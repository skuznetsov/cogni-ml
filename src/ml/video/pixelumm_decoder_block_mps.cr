# Opt-in, one-layer PixelUMM decoder execution. This path is deliberately
# independent of the existing Qwen and Hy-Nomas decoder modules.
require "../core/buffer"
require "./metal_mps_command_chain"
require "./metal_ops"
require "./pixelumm_block_profile"

lib PixelUMMDecoderBlockMPSFFI
  struct PhaseMarker
    wall_ns : UInt64
    thread_cpu_ns : UInt64
    root_count : Int32
    phase : Int32
  end
end

{% if flag?(:darwin) && !flag?(:cpu_only) %}
  @[Link(ldflags: "-framework Foundation -framework Metal -framework MetalPerformanceShaders -framework MetalPerformanceShadersGraph")]
{% end %}

{% unless flag?(:cpu_only) %}
  lib PixelUMMDecoderBlockMPSFFI
    fun plan_create = gs_pixelumm_decoder_block_plan_create(
      device : Void*, temporal : Int32, height : Int32, width : Int32,
      tile : Int32, dim : Int32, ffn : Int32, heads : Int32,
      kv_heads : Int32, head_dim : Int32, epsilon : Float32,
      input : Void*, cosine : Void*, sine : Void*, weights : Void**,
      core : Void**, captures : Void**, capture_mask : UInt64,
      scratch : Void**, max_query : Int32,
      max_key : Int32, tile_count : Int32, error : UInt8*,
      error_capacity : LibC::SizeT,
    ) : Void*
    fun plan_create_with_ffn_f32 = gs_pixelumm_decoder_block_plan_create_with_ffn_f32(
      device : Void*, temporal : Int32, height : Int32, width : Int32,
      tile : Int32, dim : Int32, ffn : Int32, heads : Int32,
      kv_heads : Int32, head_dim : Int32, epsilon : Float32,
      input : Void*, cosine : Void*, sine : Void*, weights : Void**,
      core : Void**, captures : Void**, capture_mask : UInt64,
      scratch : Void**, max_query : Int32,
      max_key : Int32, tile_count : Int32, ffn_f32 : Int32,
      error : UInt8*, error_capacity : LibC::SizeT,
    ) : Void*
    fun encode = gs_pixelumm_decoder_block_encode(plan : Void*, chain : Void*,
                                                  error : UInt8*, error_capacity : LibC::SizeT) : Int32
    fun encode_profiled = gs_pixelumm_decoder_block_encode_profiled(
      plan : Void*, command_buffer : Void*, profile_chain : Void*,
      markers : PhaseMarker*, marker_capacity : UInt32,
      error : UInt8*, error_capacity : LibC::SizeT,
    ) : Int32
    fun plan_release = gs_pixelumm_decoder_block_plan_release(plan : Void*) : Void
  end
{% end %}

module ML::Video::PixelUMMDecoderBlockMPS
  # The PixelUMM v1 decoder architecture is intentionally fixed. Geometry is
  # varied for qualification; model dimensions and rotary policy may not drift.
  struct Config
    getter temporal : Int32
    getter height : Int32
    getter width : Int32
    getter tile_tokens : Int32
    getter model_dim : Int32
    getter ff_dim : Int32
    getter query_heads : Int32
    getter kv_heads : Int32
    getter head_dim : Int32
    getter rms_norm_eps : Float32
    getter rope_theta_t : Float32
    getter rope_theta_hw : Float32

    def initialize(@temporal : Int32, @height : Int32, @width : Int32,
                   @tile_tokens : Int32 = 8, @model_dim : Int32 = 4096,
                   @ff_dim : Int32 = 12_288, @query_heads : Int32 = 32,
                   @kv_heads : Int32 = 8, @head_dim : Int32 = 128,
                   @rms_norm_eps : Float32 = 1e-6_f32,
                   @rope_theta_t : Float32 = 1_000_000_f32,
                   @rope_theta_hw : Float32 = 10_000_f32)
      validate!
    end

    def validate! : Nil
      unless @temporal > 0 && @height > 0 && @width > 0
        raise ArgumentError.new("PixelUMM temporal, height, and width must be positive")
      end
      unless @tile_tokens == 8
        raise ArgumentError.new("PixelUMM v1 requires an 8-token spatial tile")
      end
      unless {@model_dim, @ff_dim, @query_heads, @kv_heads, @head_dim} ==
               {4096, 12_288, 32, 8, 128}
        raise ArgumentError.new("unsupported PixelUMM decoder architecture")
      end
      unless @rms_norm_eps == 1e-6_f32 && @rope_theta_t == 1_000_000_f32 &&
             @rope_theta_hw == 10_000_f32
        raise ArgumentError.new("unsupported PixelUMM RMS or MRoPE parameters")
      end
      token_count = @temporal.to_i64 * @height * @width
      unless token_count > 0 && token_count <= Int32::MAX.to_i64 // @ff_dim
        raise ArgumentError.new("PixelUMM geometry exceeds prepared MPS limits")
      end
    end

    def tokens : Int32
      (@temporal.to_i64 * @height * @width).to_i32
    end

    def tiles_h : Int32
      ((@height.to_i64 + @tile_tokens - 1) // @tile_tokens).to_i32
    end

    def tiles_w : Int32
      ((@width.to_i64 + @tile_tokens - 1) // @tile_tokens).to_i32
    end

    def layout : AttentionLayout
      AttentionLayout.new(self)
    end
  end

  # CPU-only description of the upstream tile-major permutation and physical
  # query/key gathers. It is suitable for admission/spec checks and never reads
  # or writes model tensors. Queries and keys returned here are slots in the
  # already-tile-major activation, not original T/H/W offsets.
  class AttentionLayout
    getter max_query_tokens : Int32
    getter max_key_tokens : Int32
    getter tile_count : Int32

    @tile_major_to_original : Array(Int32)
    @original_to_tile_major : Array(Int32)
    @queries : Array(Array(Int32))
    @keys : Array(Array(Int32))

    def initialize(@config : Config)
      @config.validate!
      @tile_major_to_original = [] of Int32
      @queries = [] of Array(Int32)
      @keys = [] of Array(Int32)
      tiles_h = @config.tiles_h
      tiles_w = @config.tiles_w
      @tile_count = if tiles_h <= 3 && tiles_w <= 3
                      1
                    else
                      tiles_h * tiles_w
                    end
      @original_to_tile_major = Array(Int32).new(@config.tokens, 0)

      if tiles_h <= 3 && tiles_w <= 3
        @tile_major_to_original = (0...@config.tokens).to_a
        @original_to_tile_major = @tile_major_to_original.dup
        full = (0...@config.tokens).to_a
        @queries << full.dup
        @keys << full.dup
      else
        cursor = 0
        tile_members = {} of Tuple(Int32, Int32) => Array(Int32)
        tiles_h.times do |tile_h|
          tiles_w.times do |tile_w|
            members = [] of Int32
            first_h = tile_h * @config.tile_tokens
            last_h = Math.min(first_h + @config.tile_tokens, @config.height)
            first_w = tile_w * @config.tile_tokens
            last_w = Math.min(first_w + @config.tile_tokens, @config.width)
            @config.temporal.times do |t|
              (first_h...last_h).each do |h|
                (first_w...last_w).each do |w|
                  original = ((t * @config.height + h) * @config.width + w)
                  @tile_major_to_original << original
                  @original_to_tile_major[original] = cursor
                  members << cursor
                  cursor += 1
                end
              end
            end
            tile_members[{tile_h, tile_w}] = members
          end
        end
        unless cursor == @config.tokens
          raise ArgumentError.new("PixelUMM tile-major layout lost activation tokens")
        end

        tiles_h.times do |tile_h|
          tiles_w.times do |tile_w|
            @queries << tile_members[{tile_h, tile_w}]
            center_h = clamp_center(tile_h, tiles_h)
            center_w = clamp_center(tile_w, tiles_w)
            gathered = [] of Int32
            ((center_h - 1)..(center_h + 1)).each do |neighbor_h|
              ((center_w - 1)..(center_w + 1)).each do |neighbor_w|
                next unless 0 <= neighbor_h < tiles_h && 0 <= neighbor_w < tiles_w
                gathered.concat(tile_members[{neighbor_h, neighbor_w}])
              end
            end
            @keys << gathered
          end
        end
      end

      @max_query_tokens = @queries.max_of(&.size).to_i32
      @max_key_tokens = @keys.max_of(&.size).to_i32
      unless @max_query_tokens > 0 && @max_key_tokens > 0
        raise ArgumentError.new("PixelUMM attention layout is empty")
      end
      score_count = @max_query_tokens.to_i64 * @max_key_tokens * @config.query_heads
      unless score_count <= Int32::MAX
        raise ArgumentError.new("PixelUMM bounded attention score extent exceeds MPS limits")
      end
    end

    def local_attention? : Bool
      !(@config.tiles_h <= 3 && @config.tiles_w <= 3)
    end

    # Public pure-CPU aliases used by geometry admission and spec checks.
    def permutation : Array(Int32)
      tile_major_to_original
    end

    def inverse_permutation : Array(Int32)
      original_to_tile_major
    end

    def active? : Bool
      local_attention?
    end

    def tile_major_to_original : Array(Int32)
      @tile_major_to_original.dup
    end

    def original_to_tile_major : Array(Int32)
      @original_to_tile_major.dup
    end

    def query_indices(tile : Int32) : Array(Int32)
      @queries[tile]?.try(&.dup) || raise ArgumentError.new("PixelUMM query tile is outside the layout")
    end

    def key_indices(tile : Int32) : Array(Int32)
      @keys[tile]?.try(&.dup) || raise ArgumentError.new("PixelUMM key tile is outside the layout")
    end

    # Fixed-width per-tile tables padded with zero. Counts live in the matching
    # index array as an Int32 header for the native gather kernels.
    def query_index_table : Array(Int32)
      fixed_table(@queries, @max_query_tokens)
    end

    def key_index_table : Array(Int32)
      fixed_table(@keys, @max_key_tokens)
    end

    private def fixed_table(rows : Array(Array(Int32)), width : Int32) : Array(Int32)
      values = Array(Int32).new(rows.size * (width + 1), 0)
      rows.each_with_index do |row, index|
        base = index * (width + 1)
        values[base] = row.size
        row.each_with_index { |value, offset| values[base + 1 + offset] = value }
      end
      values
    end

    private def clamp_center(position : Int32, count : Int32) : Int32
      position.clamp(1, Math.max(count - 2, 1))
    end
  end

  class Plan
    # The source fallback is full attention for grids with at most 3x3 spatial
    # tiles. Keep its F32 score/probability panels bounded independently of
    # named-buffer admission (MPSGraph owns those temporary tensors).
    FULL_ATTENTION_MAX_TOKENS = 2048

    WEIGHT_NAMES = {
      "input_norm.weight", "attn.q_proj.weight", "attn.k_proj.weight",
      "attn.v_proj.weight", "attn.o_proj.weight", "attn.q_norm_t.weight",
      "attn.q_norm_hw.weight", "attn.k_norm_t.weight", "attn.k_norm_hw.weight",
      "post_attn_norm.weight", "mlp.gate_proj.weight", "mlp.up_proj.weight",
      "mlp.down_proj.weight",
    }
    CAPTURE_NAMES = {
      "input_norm", "q_proj", "k_proj", "v_proj", "q_norm", "k_norm",
      "q_rope", "k_rope", "attention", "o_proj", "attn_residual",
      "post_attn_norm", "gate_proj", "up_proj", "silu_gate", "product",
      "down_proj", "output", "input_norm_statistics", "input_norm_pre_gain",
      "q_ready", "k_ready", "q_norm_statistics", "k_norm_statistics",
      "q_norm_inverse", "k_norm_inverse",
    }
    CAPTURE_BYTES = {
      "input_norm" => :bf16_d, "q_proj" => :bf16_d,
      "k_proj" => :bf16_kv, "v_proj" => :bf16_kv,
      "q_norm" => :f32_q, "k_norm" => :f32_kv,
      "q_rope" => :f32_q, "k_rope" => :f32_kv,
      "attention" => :bf16_d, "o_proj" => :bf16_d,
      "attn_residual" => :bf16_d, "post_attn_norm" => :bf16_d,
      "gate_proj" => :bf16_f, "up_proj" => :bf16_f,
      "silu_gate" => :bf16_f, "product" => :bf16_f,
      "down_proj" => :bf16_d, "output" => :bf16_d,
      "input_norm_statistics" => :f32_rows,
      "input_norm_pre_gain" => :bf16_d,
      "q_ready" => :bf16_d, "k_ready" => :bf16_kv,
      "q_norm_statistics" => :f32_q_norm_pair,
      "k_norm_statistics" => :f32_kv_norm_pair,
      "q_norm_inverse" => :f32_q_norm_pair,
      "k_norm_inverse" => :f32_kv_norm_pair,
    }
    CORE_NAMES = {"input_norm", "q", "k", "v", "attention", "residual", "gate", "up", "output"}

    getter config : Config
    getter input : ML::MetalBuffer
    getter stages = {} of String => ML::MetalBuffer
    getter named_bytes_by_name = {} of String => Int64
    getter named_bytes : Int64 = 0_i64
    getter local_attention_query_tokens : Int32
    getter local_attention_key_tokens : Int32
    getter local_attention_tile_count : Int32
    getter attention_graph_encodes : Int32 = 0
    getter last_encode_host_seconds : Float64 = 0.0
    getter slot : Int32
    getter ffn_f32 : Bool

    @cosine : ML::MetalBuffer
    @sine : ML::MetalBuffer
    @weights : Array(ML::MetalBuffer)
    @core = {} of String => ML::MetalBuffer
    @captures = {} of String => ML::MetalBuffer
    @query_indices : ML::MetalBuffer? = nil
    @key_indices : ML::MetalBuffer? = nil
    @query_gather_f32 : ML::MetalBuffer? = nil
    @key_gather_f32 : ML::MetalBuffer? = nil
    @value_gather_f32 : ML::MetalBuffer? = nil
    @scores_f32 : ML::MetalBuffer? = nil
    @probabilities_f32 : ML::MetalBuffer? = nil
    @attended_f32 : ML::MetalBuffer? = nil
    @rms_exponent : ML::MetalBuffer? = nil
    @rms_statistics : ML::MetalBuffer? = nil
    @q_norm_statistics : ML::MetalBuffer? = nil
    @k_norm_statistics : ML::MetalBuffer? = nil
    @q_norm_inverse : ML::MetalBuffer? = nil
    @k_norm_inverse : ML::MetalBuffer? = nil
    @native = Pointer(Void).null
    @workspace : Workspace? = nil
    @owns_workspace = false
    @workspace_attached = false
    @slot = 0_i32
    @chain : ML::Metal::MPSCommandChain? = nil
    @profiled_encode = false
    @profile_markers = Slice(PixelUMMDecoderBlockMPSFFI::PhaseMarker).new(11) { PixelUMMDecoderBlockMPSFFI::PhaseMarker.new }
    @released = false
    @error_text = Bytes.new(1024, 0_u8)
    @@retained_mutex = Mutex.new
    @@retained = [] of Plan

    def initialize(@config : Config, *, input : ML::MetalBuffer,
                   weights : Hash(String, ML::MetalBuffer), cos : ML::MetalBuffer,
                   sin : ML::MetalBuffer, max_named_bytes : Int64,
                   captures : Array(String) = [] of String,
                   workspace : Workspace? = nil, slot : Int32 = 0,
                   ffn_f32 : Bool = false)
      @input = input
      @cosine = cos
      @sine = sin
      @ffn_f32 = ffn_f32
      @config.validate!
      layout = @config.layout
      @local_attention_query_tokens = layout.max_query_tokens
      @local_attention_key_tokens = layout.max_key_tokens
      @local_attention_tile_count = layout.tile_count
      if captures.uniq.size != captures.size
        raise ArgumentError.new("PixelUMM capture stages must be unique")
      end
      selected = captures.uniq
      unknown = selected.reject { |name| name == "x" || CAPTURE_NAMES.includes?(name) }
      raise ArgumentError.new("unknown PixelUMM capture stage(s): #{unknown.join(",")}") unless unknown.empty?
      @weights = [] of ML::MetalBuffer
      validate_bindings!(weights)
      metadata_bytes = weights.to_h { |name, buffer| {name, buffer.size} }
      self.class.validate_admission!(@config,
        input_bytes: @input.size, cosine_bytes: @cosine.size,
        sine_bytes: @sine.size, weights_bytes: metadata_bytes,
        max_named_bytes: max_named_bytes, captures: selected)
      if workspace
        workspace.validate_plan_binding!(@config, slot, @input, @cosine, @sine, @weights, selected)
        @workspace = workspace
        @slot = slot
      else
        @owns_workspace = true
        @slot = 0_i32
      end

      {% if flag?(:cpu_only) %}
        raise "Metal disabled (cpu_only)"
      {% else %}
        begin
          unless @workspace
            @workspace = Workspace.new(@config,
              captures_by_slot: [selected], max_owned_bytes: max_named_bytes)
          end
          workspace = @workspace.not_nil!
          workspace.attach_plan!(@slot, @input, @cosine, @sine, @weights, selected)
          @workspace_attached = true
          bind_workspace_buffers!(workspace)
          @named_bytes = workspace.named_bytes
          @named_bytes_by_name = workspace.named_bytes_by_name.dup
          setup_native!(layout, weights)
          @@retained_mutex.synchronize { @@retained << self }
        rescue ex
          release_native_plan
          if workspace = @workspace
            workspace.detach_failed_plan!(@slot) if @workspace_attached
            @workspace_attached = false
            workspace.release if @owns_workspace
          end
          @core.clear
          @captures.clear
          @stages.clear
          raise ex
        end
      {% end %}
    end

    # Pure arithmetic admission query. `max_named_bytes` is the allowance for
    # plan-owned GPU buffers only; callers account for borrowed weights and
    # inputs separately.
    def self.owned_bytes(config : Config, layout : AttentionLayout,
                         captures : Array(String) = [] of String) : Int64
      config.validate!
      j = config.tokens.to_i64
      d = config.model_dim.to_i64
      f = config.ff_dim.to_i64
      kv = config.kv_heads.to_i64 * config.head_dim
      bytes = 0_i64
      # Persistent core activations. `q` is reused for o_proj/post_attn_norm;
      # `attention` is reused for down_proj after the attention capture copy.
      {"input_norm", "q", "attention", "residual", "output"}.each do |name|
        bytes += j * d * 2
      end
      {"k", "v"}.each { bytes += j * kv * 2 }
      {"gate", "up"}.each { bytes += j * f * 2 }
      # Bounded exact-geometry attention panels. Q/K/V are explicitly gathered
      # for all query heads; scores and probabilities are separate F32
      # materialization boundaries, and PV writes F32 before BF16 scatter.
      heads = config.query_heads.to_i64
      max_q = layout.max_query_tokens.to_i64
      max_k = layout.max_key_tokens.to_i64
      dh = config.head_dim.to_i64
      bytes += 2_i64 * heads * max_q * dh * 4    # Q, attended
      bytes += 2_i64 * heads * max_k * dh * 4    # K, V
      bytes += 2_i64 * heads * max_q * max_k * 4 # scores, probabilities
      bytes += layout.tile_count.to_i64 * (layout.max_query_tokens + 1_i64) * 4
      bytes += layout.tile_count.to_i64 * (layout.max_key_tokens + 1_i64) * 4
      bytes += 4     # Immutable F32 exponent feed (= 2.0) for source pow(2).
      bytes += j * 4 # Dedicated F32 statistics rows shared by the two RMS stages.
      # Prepared Q/K split-RMS graphs write [1,J,heads,2] F32 statistics.
      bytes += j * (config.query_heads + config.kv_heads) * 2 * 4
      # Prepared Q/K inverse-root graphs retain a separate F32 output buffer.
      bytes += j * (config.query_heads + config.kv_heads) * 2 * 4
      selected = captures.uniq.reject { |name| name == "x" || name == "output" }
      selected.each do |name|
        kind = CAPTURE_BYTES[name]? || raise ArgumentError.new("unknown PixelUMM capture stage #{name}")
        bytes += case kind
                 when :bf16_d           then j * d * 2
                 when :bf16_kv          then j * kv * 2
                 when :bf16_f           then j * f * 2
                 when :f32_q            then j * d * 4
                 when :f32_kv           then j * kv * 4
                 when :f32_rows         then j * 4
                 when :f32_q_norm_pair  then j * config.query_heads * 2 * 4
                 when :f32_kv_norm_pair then j * config.kv_heads * 2 * 4
                 else                        raise ArgumentError.new("unknown PixelUMM capture type")
                 end
      end
      bytes
    end

    # CPU-only shape and budget gate. Native allocation begins only after this
    # passes. Identity overlap is checked against actual MetalBuffer ranges by
    # the constructor after the metadata gate succeeds.
    def self.validate_admission!(config : Config, *, input_bytes : Int64,
                                 cosine_bytes : Int64, sine_bytes : Int64,
                                 weights_bytes : Hash(String, Int64),
                                 max_named_bytes : Int64,
                                 captures : Array(String) = [] of String) : Nil
      config.validate!
      if config.tiles_h <= 3 && config.tiles_w <= 3 &&
         config.tokens > FULL_ATTENTION_MAX_TOKENS
        raise ArgumentError.new("PixelUMM full-attention fallback exceeds 2048 tokens")
      end
      expected_weights = {
        "input_norm.weight"     => 4096_i64 * 2,
        "attn.q_proj.weight"    => 4096_i64 * 4096 * 2,
        "attn.k_proj.weight"    => 1024_i64 * 4096 * 2,
        "attn.v_proj.weight"    => 1024_i64 * 4096 * 2,
        "attn.o_proj.weight"    => 4096_i64 * 4096 * 2,
        "attn.q_norm_t.weight"  => 64_i64 * 2,
        "attn.q_norm_hw.weight" => 64_i64 * 2,
        "attn.k_norm_t.weight"  => 64_i64 * 2,
        "attn.k_norm_hw.weight" => 64_i64 * 2,
        "post_attn_norm.weight" => 4096_i64 * 2,
        "mlp.gate_proj.weight"  => 12_288_i64 * 4096 * 2,
        "mlp.up_proj.weight"    => 12_288_i64 * 4096 * 2,
        "mlp.down_proj.weight"  => 4096_i64 * 12_288 * 2,
      }
      unless weights_bytes.keys.sort == expected_weights.keys.sort
        raise ArgumentError.new("exact 13 PixelUMM layer0 weight metadata entries required")
      end
      expected_weights.each do |name, bytes|
        unless weights_bytes[name] == bytes
          raise ArgumentError.new("PixelUMM #{name} must be #{bytes} bytes")
        end
      end
      j = config.tokens.to_i64
      unless input_bytes == j * 4096 * 2 && cosine_bytes == j * 128 * 4 &&
             sine_bytes == j * 128 * 4
        raise ArgumentError.new("PixelUMM input or MRoPE table byte extent is invalid")
      end
      if captures.uniq.size != captures.size
        raise ArgumentError.new("PixelUMM capture stages must be unique")
      end
      unknown = captures.reject { |name| name == "x" || CAPTURE_NAMES.includes?(name) }
      raise ArgumentError.new("unknown PixelUMM capture stage(s): #{unknown.join(",")}") unless unknown.empty?
      needed = owned_bytes(config, config.layout, captures)
      unless max_named_bytes > 0 && needed <= max_named_bytes
        raise ArgumentError.new("PixelUMM plan-owned buffers exceed admitted byte budget")
      end
    end

    def stage(name : String) : ML::MetalBuffer
      check_live!
      @stages[name]? || raise ArgumentError.new("PixelUMM stage was not captured: #{name}")
    end

    def workspace : Workspace
      @workspace || raise ArgumentError.new("PixelUMM plan has no workspace")
    end

    def encode(chain : ML::Metal::MPSCommandChain, *, profile : Bool = false) : Nil
      check_live!
      raise ArgumentError.new("PixelUMM phase profiling requires an enabled fresh API chain") if profile && !chain.api_profile_enabled?
      @profiled_encode = false
      started = Time.instant
      @workspace.not_nil!.submit_encode!(@slot, chain) do
        chain.encoder
        if previous = @chain
          unless previous.same?(chain) || previous.completed?
            raise ArgumentError.new("PixelUMM plan cannot move between in-flight command chains")
          end
        end
        @chain = chain
        {% if flag?(:darwin) && !flag?(:cpu_only) %}
          chain.pause!
          status = -1
          begin
            if profile
              status = PixelUMMDecoderBlockMPSFFI.encode_profiled(@native, chain.handle,
                chain.api_profile_handle, @profile_markers.to_unsafe, @profile_markers.size.to_u32,
                @error_text.to_unsafe, @error_text.size)
            else
              status = PixelUMMDecoderBlockMPSFFI.encode(@native, chain.handle,
                @error_text.to_unsafe, @error_text.size)
            end
          ensure
            chain.resume!
          end
          raise ArgumentError.new("PixelUMM MPS encode failed (#{status}): #{error_message}") unless status == 0
          # Each tile schedules separately materialized QK, softmax, and PV
          # MPSGraphs. No graph is encoded for a padding row or unused head group.
          @attention_graph_encodes = @local_attention_tile_count * 3
          @profiled_encode = profile
          @last_encode_host_seconds = (Time.instant - started).total_seconds
        {% else %}
          raise ArgumentError.new("PixelUMM prepared MPS execution requires macOS Metal Performance Shaders")
        {% end %}
      end
    end

    # CPU metadata and existing root timestamps are interpreted only after the
    # caller's successful sole terminal wait, never during inference.
    def phase_profile(chain : ML::Metal::MPSCommandChain) : ML::Video::PixelUMMBlockPhaseProfile
      unless @profiled_encode && @chain.try(&.same?(chain)) && chain.completed? && chain.wait_count == 1
        raise ArgumentError.new("PixelUMM phase profile requires the profiled chain's terminal wait")
      end
      markers = @profile_markers.map_with_index do |raw, index|
        raise ArgumentError.new("PixelUMM native phase inventory is incomplete") unless raw.phase == index
        ML::Video::PixelUMMBlockPhaseMarker.new(
          ML::Video::PixelUMMBlockPhaseProfile::MARKER_NAMES[index],
          raw.wall_ns.to_f64 / 1e9, raw.thread_cpu_ns.to_f64 / 1e9, raw.root_count)
      end.to_a
      ML::Video::PixelUMMBlockPhaseProfile.build(markers, chain.completed_root_timings,
        chain.gpu_elapsed_seconds, chain.api_profile, @local_attention_tile_count)
    end

    def release : Nil
      return if @released
      if chain = @chain
        unless chain.completed?
          raise ArgumentError.new("complete PixelUMM command chain before releasing its plan")
        end
      end
      @workspace.not_nil!.release_plan!(@slot) do
        release_native_plan
      end
      @core.clear
      @captures.clear
      @stages.clear
      @released = true
      @@retained_mutex.synchronize { @@retained.reject! { |plan| plan.same?(self) } }
      @workspace.not_nil!.release if @owns_workspace
    end

    private def validate_bindings!(weights : Hash(String, ML::MetalBuffer)) : Nil
      unless weights.keys.sort == WEIGHT_NAMES.to_a.sort
        raise ArgumentError.new("exact 13 PixelUMM layer0 weight bindings required")
      end
      j = @config.tokens.to_i64
      d = @config.model_dim.to_i64
      f = @config.ff_dim.to_i64
      kv = @config.kv_heads.to_i64 * @config.head_dim
      sizes = {
        "input_norm.weight"     => d * 2,
        "attn.q_proj.weight"    => d * d * 2,
        "attn.k_proj.weight"    => kv * d * 2,
        "attn.v_proj.weight"    => kv * d * 2,
        "attn.o_proj.weight"    => d * d * 2,
        "attn.q_norm_t.weight"  => 64_i64 * 2,
        "attn.q_norm_hw.weight" => 64_i64 * 2,
        "attn.k_norm_t.weight"  => 64_i64 * 2,
        "attn.k_norm_hw.weight" => 64_i64 * 2,
        "post_attn_norm.weight" => d * 2,
        "mlp.gate_proj.weight"  => f * d * 2,
        "mlp.up_proj.weight"    => f * d * 2,
        "mlp.down_proj.weight"  => d * f * 2,
      }
      sizes.each do |name, bytes|
        buffer = weights[name]
        check_shared_exact!(buffer, bytes, name)
        @weights << buffer
      end
      check_shared_exact!(@input, j * d * 2, "input")
      table_bytes = j * @config.head_dim * 4
      check_shared_exact!(@cosine, table_bytes, "cosine")
      check_shared_exact!(@sine, table_bytes, "sine")
      borrowed = [@input, @cosine, @sine] + @weights
      validate_disjoint!(borrowed, "borrowed PixelUMM bindings must not alias")
    end

    private def bind_workspace_buffers!(workspace : Workspace) : Nil
      @core = workspace.core_for_slot(@slot)
      @captures = workspace.captures_for_slot(@slot)
      @captures.each do |name, buffer|
        @stages[name] = buffer
      end
      @stages["x"] = @input if workspace.capture_x?(@slot)
      scratch = workspace.scratch_buffers
      @query_gather_f32 = scratch[0]
      @key_gather_f32 = scratch[1]
      @value_gather_f32 = scratch[2]
      @scores_f32 = scratch[3]
      @probabilities_f32 = scratch[4]
      @attended_f32 = scratch[5]
      @query_indices = scratch[6]
      @key_indices = scratch[7]
      @rms_exponent = scratch[8]
      @rms_statistics = scratch[9]
      @q_norm_statistics = scratch[10]
      @k_norm_statistics = scratch[11]
      @q_norm_inverse = scratch[12]
      @k_norm_inverse = scratch[13]
    end

    private def setup_native!(layout : AttentionLayout,
                              weights : Hash(String, ML::MetalBuffer)) : Nil
      {% if flag?(:darwin) && !flag?(:cpu_only) %}
        weight_handles = WEIGHT_NAMES.map { |name| weights[name].handle }.to_a
        core_handles = {"input_norm", "q", "k", "v", "attention", "residual", "gate", "up", "output"}
          .map { |name| @core[name].handle }.to_a
        capture_handles = CAPTURE_NAMES.map do |name|
          @captures[name]?.try(&.handle) || Pointer(Void).null
        end.to_a
        @native = PixelUMMDecoderBlockMPSFFI.plan_create_with_ffn_f32(
          ML::Metal::Device.instance.device_handle,
          @config.temporal, @config.height, @config.width, @config.tile_tokens,
          @config.model_dim, @config.ff_dim, @config.query_heads,
          @config.kv_heads, @config.head_dim, @config.rms_norm_eps,
          @input.handle, @cosine.handle, @sine.handle,
          weight_handles.to_unsafe, core_handles.to_unsafe,
          capture_handles.to_unsafe, capture_mask,
          [@query_gather_f32.not_nil!.handle, @key_gather_f32.not_nil!.handle,
           @value_gather_f32.not_nil!.handle, @scores_f32.not_nil!.handle,
           @probabilities_f32.not_nil!.handle, @attended_f32.not_nil!.handle,
           @query_indices.not_nil!.handle, @key_indices.not_nil!.handle,
           @rms_exponent.not_nil!.handle,
           @rms_statistics.not_nil!.handle, @q_norm_statistics.not_nil!.handle,
           @k_norm_statistics.not_nil!.handle, @q_norm_inverse.not_nil!.handle,
           @k_norm_inverse.not_nil!.handle]
            .to_unsafe,
          layout.max_query_tokens, layout.max_key_tokens, layout.tile_count,
          @ffn_f32 ? 1 : 0,
          @error_text.to_unsafe, @error_text.size)
        raise ArgumentError.new("PixelUMM plan preparation failed: #{error_message}") if @native.null?
      {% else %}
        raise ArgumentError.new("PixelUMM prepared MPS execution requires macOS Metal Performance Shaders")
      {% end %}
    end

    private def capture_mask : UInt64
      mask = 0_u64
      CAPTURE_NAMES.each_with_index do |name, index|
        mask |= 1_u64 << index if @captures.has_key?(name)
      end
      mask
    end

    private def check_shared_exact!(buffer : ML::MetalBuffer, bytes : Int64,
                                    name : String) : Nil
      unless buffer.valid? && buffer.storage_mode.shared? && buffer.size == bytes
        raise ArgumentError.new("PixelUMM #{name} must be an exact-size shared buffer (#{bytes} bytes)")
      end
    end

    private def validate_disjoint!(buffers : Array(ML::MetalBuffer), message : String) : Nil
      buffers.each_with_index do |left, left_index|
        buffers.each_with_index do |right, right_index|
          next if right_index <= left_index
          raise ArgumentError.new(message) if MetalOps.overlap?(left, right)
        end
      end
    end

    private def error_message : String
      String.new(@error_text.to_unsafe)
    end

    private def release_native_plan : Nil
      {% if flag?(:darwin) && !flag?(:cpu_only) %}
        PixelUMMDecoderBlockMPSFFI.plan_release(@native) unless @native.null?
      {% end %}
      @native = Pointer(Void).null
    end

    private def check_live! : Nil
      raise ArgumentError.new("PixelUMM MPS plan has been released") if @released || @native.null?
    end
  end

  # Owns one set of transient activations and attention/RMS scratch for one or
  # more sequential decoder plans. Each plan gets a private persistent output
  # and private capture buffers; slot N may consume slot N-1's output directly.
  class Workspace
    getter config : Config
    getter named_bytes : Int64
    getter slot_count : Int32

    @layout : AttentionLayout
    @capture_sets : Array(Array(String))
    @core_shared = {} of String => ML::MetalBuffer
    @outputs = [] of ML::MetalBuffer
    @slot_captures = [] of Hash(String, ML::MetalBuffer)
    @slot_captures_x = [] of Bool
    @scratch = [] of ML::MetalBuffer
    @scratch_names = [] of String
    @owned = [] of ML::MetalBuffer
    @named_bytes_by_name = {} of String => Int64
    @claimed : Array(Bool)
    @released_slots : Array(Bool)
    @last_chains : Array(ML::Metal::MPSCommandChain?)
    @bound_weights = [] of Array(ML::MetalBuffer)
    @bound_cosine : ML::MetalBuffer? = nil
    @bound_sine : ML::MetalBuffer? = nil
    @request_chain : ML::Metal::MPSCommandChain? = nil
    @next_encode_slot = 0_i32
    @request_failed = false
    @released = false
    @mutex = Mutex.new
    @@retained_mutex = Mutex.new
    @@retained = [] of Workspace

    def initialize(@config : Config, *, captures_by_slot : Array(Array(String)),
                   max_owned_bytes : Int64)
      @config.validate!
      if @config.tiles_h <= 3 && @config.tiles_w <= 3 &&
         @config.tokens > Plan::FULL_ATTENTION_MAX_TOKENS
        raise ArgumentError.new("PixelUMM full-attention fallback exceeds 2048 tokens")
      end
      raise ArgumentError.new("PixelUMM workspace requires at least one plan slot") if captures_by_slot.empty?
      @layout = @config.layout
      @capture_sets = captures_by_slot.map do |captures|
        self.class.validate_capture_set!(captures)
        captures.dup
      end
      @slot_count = @capture_sets.size.to_i32
      @named_bytes = self.class.owned_bytes(@config, @layout, @capture_sets)
      unless max_owned_bytes > 0 && @named_bytes <= max_owned_bytes
        raise ArgumentError.new("PixelUMM workspace buffers exceed admitted byte budget")
      end
      @claimed = Array(Bool).new(@slot_count, false)
      @released_slots = Array(Bool).new(@slot_count, false)
      @last_chains = Array(ML::Metal::MPSCommandChain?).new(@slot_count, nil)
      @bound_weights = Array(Array(ML::MetalBuffer)).new(@slot_count) { [] of ML::MetalBuffer }

      {% if flag?(:cpu_only) %}
        raise "Metal disabled (cpu_only)"
      {% else %}
        begin
          allocate_buffers!
          unless @named_bytes_by_name.values.sum == @named_bytes
            raise ArgumentError.new("PixelUMM workspace named-byte accounting mismatch")
          end
          @@retained_mutex.synchronize { @@retained << self }
        rescue ex
          @owned.each(&.release)
          @owned.clear
          @core_shared.clear
          @outputs.clear
          @slot_captures.clear
          @scratch.clear
          @named_bytes_by_name.clear
          raise ex
        end
      {% end %}
    end

    # Unique bytes across both borrowed bindings and workspace-owned buffers.
    # Slot 0 borrows the external activation; each later slot consumes the
    # previous slot's already-counted output bank.
    def self.validate_admission!(config : Config, *, input_bytes : Int64,
                                 cosine_bytes : Int64, sine_bytes : Int64,
                                 weights_bytes_by_slot : Array(Hash(String, Int64)),
                                 captures_by_slot : Array(Array(String)),
                                 max_unique_bytes : Int64) : Int64
      config.validate!
      unless weights_bytes_by_slot.size == captures_by_slot.size && !captures_by_slot.empty?
        raise ArgumentError.new("PixelUMM workspace weight/capture slot counts must match")
      end
      validate_capture_sets!(captures_by_slot)
      captures_by_slot.each_with_index do |captures, slot|
        Plan.validate_admission!(config,
          input_bytes: input_bytes,
          cosine_bytes: cosine_bytes,
          sine_bytes: sine_bytes,
          weights_bytes: weights_bytes_by_slot[slot],
          max_named_bytes: Int64::MAX,
          captures: captures)
      end
      layout = config.layout
      owned = owned_bytes(config, layout, captures_by_slot)
      borrowed = input_bytes + cosine_bytes + sine_bytes
      weights_bytes_by_slot.each do |metadata|
        borrowed += metadata.values.sum
      end
      unique = owned + borrowed
      unless max_unique_bytes > 0 && unique <= max_unique_bytes
        raise ArgumentError.new("PixelUMM unique workspace/binding bytes exceed admitted byte budget")
      end
      unique
    end

    # Core transients and native scratch are shared once. Output banks and
    # selected persistent captures are unique to each plan slot.
    def self.owned_bytes(config : Config, layout : AttentionLayout,
                         captures_by_slot : Array(Array(String))) : Int64
      config.validate!
      validate_capture_sets!(captures_by_slot)
      raise ArgumentError.new("PixelUMM workspace requires at least one plan slot") if captures_by_slot.empty?
      output_bytes = config.tokens.to_i64 * config.model_dim * 2
      one_slot = Plan.owned_bytes(config, layout)
      shared = one_slot - output_bytes
      total = shared + output_bytes * captures_by_slot.size
      captures_by_slot.each do |captures|
        total += Plan.owned_bytes(config, layout, captures) - one_slot
      end
      total
    end

    def self.validate_capture_sets!(captures_by_slot : Array(Array(String))) : Nil
      captures_by_slot.each { |captures| validate_capture_set!(captures) }
    end

    def self.validate_capture_set!(captures : Array(String)) : Nil
      if captures.uniq.size != captures.size
        raise ArgumentError.new("PixelUMM capture stages must be unique")
      end
      unknown = captures.reject { |name| name == "x" || Plan::CAPTURE_NAMES.includes?(name) }
      raise ArgumentError.new("unknown PixelUMM capture stage(s): #{unknown.join(",")}") unless unknown.empty?
    end

    def named_bytes_by_name : Hash(String, Int64)
      @named_bytes_by_name.dup
    end

    def validate_plan_binding!(config : Config, slot : Int32,
                               input : ML::MetalBuffer, cosine : ML::MetalBuffer,
                               sine : ML::MetalBuffer, weights : Array(ML::MetalBuffer),
                               captures : Array(String)) : Nil
      @mutex.synchronize do
        validate_plan_binding_unlocked!(config, slot, input, cosine, sine, weights, captures)
      end
    end

    def attach_plan!(slot : Int32, input : ML::MetalBuffer,
                     cosine : ML::MetalBuffer, sine : ML::MetalBuffer,
                     weights : Array(ML::MetalBuffer), captures : Array(String)) : Nil
      @mutex.synchronize do
        validate_plan_binding_unlocked!(@config, slot, input, cosine, sine, weights, captures)
        @claimed[slot] = true
        @bound_cosine ||= cosine
        @bound_sine ||= sine
        @bound_weights[slot] = weights.dup
      end
    end

    def detach_failed_plan!(slot : Int32) : Nil
      @mutex.synchronize do
        return if @released || slot < 0 || slot >= @slot_count || !@claimed[slot]
        @claimed[slot] = false
        @released_slots[slot] = false
        @bound_weights[slot].clear
        if @bound_weights.all?(&.empty?)
          @bound_cosine = nil
          @bound_sine = nil
        end
      end
    end

    def release_plan!(slot : Int32, &block : -> Nil) : Nil
      @mutex.synchronize do
        check_live!
        validate_slot!(slot)
        unless @claimed[slot] && !@released_slots[slot]
          raise ArgumentError.new("PixelUMM workspace plan slot is not live")
        end
        if chain = @last_chains[slot]
          unless chain.completed?
            raise ArgumentError.new("complete PixelUMM command chain before releasing its plan")
          end
        end
        yield
        @released_slots[slot] = true
      end
    end

    # The host-side submission is serialized for the duration of the native
    # encode call. This closes same-slot double admission without adding a GPU
    # wait or changing the command-chain schedule.
    def submit_encode!(slot : Int32, chain : ML::Metal::MPSCommandChain,
                       &block : -> Nil) : Nil
      @mutex.synchronize do
        check_live!
        validate_slot!(slot)
        unless @claimed[slot] && !@released_slots[slot]
          raise ArgumentError.new("PixelUMM workspace plan slot is not live")
        end
        if @slot_count > 1 && !@claimed.all?
          raise ArgumentError.new("prepare every PixelUMM workspace plan before encoding")
        end

        if previous = @last_chains[slot]
          unless previous.same?(chain) || previous.completed?
            raise ArgumentError.new("PixelUMM plan cannot move between in-flight command chains")
          end
        end

        if @slot_count > 1
          if @request_failed || @next_encode_slot == @slot_count
            if @request_chain && @request_chain.not_nil!.completed?
              @request_chain = nil
              @next_encode_slot = 0_i32
              @request_failed = false
            else
              raise ArgumentError.new("complete the prior PixelUMM workspace request before reuse")
            end
          end
          unless slot == @next_encode_slot
            raise ArgumentError.new("PixelUMM workspace plans must encode once in slot order")
          end
          if request_chain = @request_chain
            unless request_chain.same?(chain)
              raise ArgumentError.new("PixelUMM workspace slots must use one command chain")
            end
          else
            @request_chain = chain
          end
        end

        begin
          yield
          @last_chains[slot] = chain
          @next_encode_slot += 1 if @slot_count > 1
        rescue ex
          # If native submission fails part-way through, shared buffers stay
          # unavailable until that same chain reaches terminal completion.
          @last_chains[slot] = chain
          @request_failed = true if @slot_count > 1
          raise ex
        end
      end
    end

    def core_for_slot(slot : Int32) : Hash(String, ML::MetalBuffer)
      check_live!
      validate_slot!(slot)
      @core_shared.merge({"output" => @outputs[slot]})
    end

    def captures_for_slot(slot : Int32) : Hash(String, ML::MetalBuffer)
      check_live!
      validate_slot!(slot)
      @slot_captures[slot].dup
    end

    def output_buffer(slot : Int32) : ML::MetalBuffer
      check_live!
      validate_slot!(slot)
      @outputs[slot]
    end

    def capture_x?(slot : Int32) : Bool
      check_live!
      validate_slot!(slot)
      @slot_captures_x[slot]
    end

    def scratch_buffers : Array(ML::MetalBuffer)
      check_live!
      @scratch.dup
    end

    # Mutable shared buffers can be overwritten by a new request. The
    # diagnostic view refuses access while a prior command chain is in flight.
    # It intentionally omits immutable index tables, the power exponent, all
    # output banks, and all per-plan captures.
    def mutable_shared_buffers : Hash(String, ML::MetalBuffer)
      @mutex.synchronize do
        check_live!
        @last_chains.each do |chain|
          if chain && !chain.completed?
            raise ArgumentError.new("complete the PixelUMM command chain before inspecting shared transients")
          end
        end
        result = {} of String => ML::MetalBuffer
        @core_shared.each { |name, buffer| result["core.#{name}"] = buffer }
        {0, 1, 2, 3, 4, 5, 9, 10, 11, 12, 13}.each do |index|
          result[@scratch_names[index]] = @scratch[index]
        end
        result
      end
    end

    def release : Nil
      @mutex.synchronize do
        return if @released
        if chain = @request_chain
          unless chain.completed?
            raise ArgumentError.new("complete PixelUMM command chain before releasing its workspace")
          end
        end
        @claimed.each_with_index do |claimed, index|
          if claimed && !@released_slots[index]
            raise ArgumentError.new("release every PixelUMM plan before its workspace")
          end
        end
        @owned.each(&.release)
        @owned.clear
        @core_shared.clear
        @outputs.clear
        @slot_captures.clear
        @scratch.clear
        @named_bytes_by_name.clear
        @released = true
      end
      @@retained_mutex.synchronize { @@retained.reject! { |workspace| workspace.same?(self) } }
    end

    private def allocate_buffers! : Nil
      j = @config.tokens.to_i64
      d = @config.model_dim.to_i64
      f = @config.ff_dim.to_i64
      kv = @config.kv_heads.to_i64 * @config.head_dim
      {"input_norm" => j * d * 2, "q" => j * d * 2,
       "k" => j * kv * 2, "v" => j * kv * 2,
       "attention" => j * d * 2, "residual" => j * d * 2,
       "gate" => j * f * 2, "up" => j * f * 2}.each do |name, bytes|
        @core_shared[name] = own_buffer("workspace.#{name}", bytes)
      end
      output_bytes = j * d * 2
      @slot_count.times do |slot|
        name = @slot_count == 1 ? "workspace.output" : "workspace.output.slot#{slot}"
        @outputs << own_buffer(name, output_bytes)
      end

      max_q = @layout.max_query_tokens.to_i64
      max_k = @layout.max_key_tokens.to_i64
      heads = @config.query_heads.to_i64
      dh = @config.head_dim.to_i64
      @scratch_names = [
        "attention.query_gather_f32", "attention.key_gather_f32",
        "attention.value_gather_f32", "attention.scores_f32",
        "attention.probabilities_f32", "attention.attended_f32",
        "attention.query_indices_i32", "attention.key_indices_i32",
        "rms.power_exponent_f32", "rms.statistics_f32",
        "q_norm.statistics_f32", "k_norm.statistics_f32",
        "q_norm.inverse_f32", "k_norm.inverse_f32",
      ]
      scratch_sizes = [
        heads * max_q * dh * 4,
        heads * max_k * dh * 4,
        heads * max_k * dh * 4,
        heads * max_q * max_k * 4,
        heads * max_q * max_k * 4,
        heads * max_q * dh * 4,
        @layout.query_index_table.size.to_i64 * 4,
        @layout.key_index_table.size.to_i64 * 4,
        4_i64,
        j * 4,
        j * @config.query_heads * 2 * 4,
        j * @config.kv_heads * 2 * 4,
        j * @config.query_heads * 2 * 4,
        j * @config.kv_heads * 2 * 4,
      ]
      scratch_sizes.each_with_index do |bytes, index|
        @scratch << own_buffer(@scratch_names[index], bytes)
      end
      q_table = @layout.query_index_table
      k_table = @layout.key_index_table
      @scratch[6].write_bytes(q_table.to_unsafe.as(Pointer(UInt8)), q_table.size * 4)
      @scratch[7].write_bytes(k_table.to_unsafe.as(Pointer(UInt8)), k_table.size * 4)
      @scratch[8].write([2.0_f32])

      @slot_count.times do |slot|
        captures = {} of String => ML::MetalBuffer
        @slot_captures_x << @capture_sets[slot].includes?("x")
        @capture_sets[slot].each do |name|
          next if name == "x"
          if name == "output"
            captures[name] = @outputs[slot]
            next
          end
          bytes = capture_bytes(name)
          prefix = @slot_count == 1 ? "capture" : "capture.slot#{slot}"
          captures[name] = own_buffer("#{prefix}.#{name}", bytes)
        end
        @slot_captures << captures
      end
    end

    private def capture_bytes(name : String) : Int64
      j = @config.tokens.to_i64
      d = @config.model_dim.to_i64
      f = @config.ff_dim.to_i64
      kv = @config.kv_heads.to_i64 * @config.head_dim
      case Plan::CAPTURE_BYTES[name]?
      when :bf16_d           then j * d * 2
      when :bf16_kv          then j * kv * 2
      when :bf16_f           then j * f * 2
      when :f32_q            then j * d * 4
      when :f32_kv           then j * kv * 4
      when :f32_rows         then j * 4
      when :f32_q_norm_pair  then j * @config.query_heads * 2 * 4
      when :f32_kv_norm_pair then j * @config.kv_heads * 2 * 4
      else                        raise ArgumentError.new("unknown PixelUMM capture type for #{name}")
      end
    end

    private def own_buffer(name : String, bytes : Int64) : ML::MetalBuffer
      raise ArgumentError.new("invalid PixelUMM buffer extent for #{name}") unless bytes > 0
      buffer = ML::MetalBuffer.new(bytes)
      @owned << buffer
      @named_bytes_by_name[name] = bytes
      buffer
    end

    private def same_config?(other : Config) : Bool
      {@config.temporal, @config.height, @config.width, @config.tile_tokens,
       @config.model_dim, @config.ff_dim, @config.query_heads, @config.kv_heads,
       @config.head_dim, @config.rms_norm_eps, @config.rope_theta_t, @config.rope_theta_hw} ==
        {other.temporal, other.height, other.width, other.tile_tokens,
         other.model_dim, other.ff_dim, other.query_heads, other.kv_heads,
         other.head_dim, other.rms_norm_eps, other.rope_theta_t, other.rope_theta_hw}
    end

    private def validate_plan_binding_unlocked!(config : Config, slot : Int32,
                                                input : ML::MetalBuffer,
                                                cosine : ML::MetalBuffer,
                                                sine : ML::MetalBuffer,
                                                weights : Array(ML::MetalBuffer),
                                                captures : Array(String)) : Nil
      check_live!
      validate_slot!(slot)
      unless same_config?(config)
        raise ArgumentError.new("PixelUMM plan configuration does not match its workspace")
      end
      unless captures.sort == @capture_sets[slot].sort
        raise ArgumentError.new("PixelUMM plan captures do not match its workspace slot")
      end
      if @claimed[slot]
        raise ArgumentError.new("PixelUMM workspace slot is already assigned")
      end
      if slot > 0
        unless @claimed[slot - 1] && input.same?(@outputs[slot - 1])
          raise ArgumentError.new("PixelUMM workspace slot input must be the prior slot output")
        end
      end
      if shared_cosine = @bound_cosine
        unless cosine.same?(shared_cosine) && sine.same?(@bound_sine.not_nil!)
          raise ArgumentError.new("PixelUMM workspace slots must share MRoPE tables")
        end
      end
      @bound_weights.each_with_index do |previous_weights, previous_slot|
        next if previous_slot == slot || previous_weights.empty?
        previous_weights.each do |previous|
          weights.each do |current|
            if MetalOps.overlap?(previous, current)
              raise ArgumentError.new("PixelUMM layer weight bindings must be disjoint across slots")
            end
          end
        end
      end
    end

    private def validate_slot!(slot : Int32) : Nil
      unless 0 <= slot < @slot_count
        raise ArgumentError.new("PixelUMM workspace slot is outside its allocation")
      end
    end

    private def check_live! : Nil
      raise ArgumentError.new("PixelUMM workspace has been released") if @released
    end
  end
end
