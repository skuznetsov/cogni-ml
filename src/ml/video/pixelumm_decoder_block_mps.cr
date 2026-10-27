# Opt-in, one-layer PixelUMM decoder execution. This path is deliberately
# independent of the existing Qwen and Hy-Nomas decoder modules.
require "../core/buffer"
require "./metal_mps_command_chain"
require "./metal_ops"

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
    fun encode = gs_pixelumm_decoder_block_encode(plan : Void*, chain : Void*,
                                                  error : UInt8*, error_capacity : LibC::SizeT) : Int32
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

    @cosine : ML::MetalBuffer
    @sine : ML::MetalBuffer
    @weights : Array(ML::MetalBuffer)
    @core = {} of String => ML::MetalBuffer
    @captures = {} of String => ML::MetalBuffer
    @owned = [] of ML::MetalBuffer
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
    @chain : ML::Metal::MPSCommandChain? = nil
    @released = false
    @error_text = Bytes.new(1024, 0_u8)
    @@retained_mutex = Mutex.new
    @@retained = [] of Plan

    def initialize(@config : Config, *, input : ML::MetalBuffer,
                   weights : Hash(String, ML::MetalBuffer), cos : ML::MetalBuffer,
                   sin : ML::MetalBuffer, max_named_bytes : Int64,
                   captures : Array(String) = [] of String)
      @input = input
      @cosine = cos
      @sine = sin
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
      required_bytes = self.class.owned_bytes(@config, layout, selected)
      @named_bytes = required_bytes

      {% if flag?(:cpu_only) %}
        raise "Metal disabled (cpu_only)"
      {% else %}
        begin
          allocate_core!
          allocate_attention_scratch!(layout)
          allocate_captures!(selected)
          setup_native!(layout, weights)
          @@retained_mutex.synchronize { @@retained << self }
        rescue ex
          release_native_plan
          @owned.each(&.release)
          @owned.clear
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

    def encode(chain : ML::Metal::MPSCommandChain) : Nil
      check_live!
      chain.encoder
      if previous = @chain
        unless previous.same?(chain) || previous.completed?
          raise ArgumentError.new("PixelUMM plan cannot move between in-flight command chains")
        end
      end
      @chain = chain
      started = Time.instant
      {% if flag?(:darwin) && !flag?(:cpu_only) %}
        chain.pause!
        status = -1
        begin
          status = PixelUMMDecoderBlockMPSFFI.encode(@native, chain.handle,
            @error_text.to_unsafe, @error_text.size)
        ensure
          chain.resume!
        end
        raise ArgumentError.new("PixelUMM MPS encode failed (#{status}): #{error_message}") unless status == 0
        # Each tile schedules separately materialized QK, softmax, and PV
        # MPSGraphs. No graph is encoded for a padding row or unused head group.
        @attention_graph_encodes = @local_attention_tile_count * 3
        @last_encode_host_seconds = (Time.instant - started).total_seconds
      {% else %}
        raise ArgumentError.new("PixelUMM prepared MPS execution requires macOS Metal Performance Shaders")
      {% end %}
    end

    def release : Nil
      return if @released
      if chain = @chain
        unless chain.completed?
          raise ArgumentError.new("complete PixelUMM command chain before releasing its plan")
        end
      end
      release_native_plan
      @owned.each(&.release)
      @owned.clear
      @core.clear
      @captures.clear
      @stages.clear
      @released = true
      @@retained_mutex.synchronize { @@retained.reject! { |plan| plan.same?(self) } }
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

    private def allocate_core! : Nil
      j = @config.tokens.to_i64
      d = @config.model_dim.to_i64
      f = @config.ff_dim.to_i64
      kv = @config.kv_heads.to_i64 * @config.head_dim
      {"input_norm" => j * d * 2, "q" => j * d * 2,
       "k" => j * kv * 2, "v" => j * kv * 2,
       "attention" => j * d * 2, "residual" => j * d * 2,
       "gate" => j * f * 2, "up" => j * f * 2,
       "output" => j * d * 2}.each do |name, bytes|
        @core[name] = own_buffer("workspace.#{name}", bytes)
      end
    end

    private def allocate_attention_scratch!(layout : AttentionLayout) : Nil
      max_q = layout.max_query_tokens.to_i64
      max_k = layout.max_key_tokens.to_i64
      dh = @config.head_dim.to_i64
      heads = @config.query_heads.to_i64
      @query_gather_f32 = own_buffer("attention.query_gather_f32", heads * max_q * dh * 4)
      @key_gather_f32 = own_buffer("attention.key_gather_f32", heads * max_k * dh * 4)
      @value_gather_f32 = own_buffer("attention.value_gather_f32", heads * max_k * dh * 4)
      @scores_f32 = own_buffer("attention.scores_f32", heads * max_q * max_k * 4)
      @probabilities_f32 = own_buffer("attention.probabilities_f32", heads * max_q * max_k * 4)
      @attended_f32 = own_buffer("attention.attended_f32", heads * max_q * dh * 4)
      q_table = layout.query_index_table
      k_table = layout.key_index_table
      @query_indices = own_buffer("attention.query_indices_i32", q_table.size.to_i64 * 4)
      @key_indices = own_buffer("attention.key_indices_i32", k_table.size.to_i64 * 4)
      @query_indices.not_nil!.write_bytes(q_table.to_unsafe.as(Pointer(UInt8)), q_table.size * 4)
      @key_indices.not_nil!.write_bytes(k_table.to_unsafe.as(Pointer(UInt8)), k_table.size * 4)
      @rms_exponent = own_buffer("rms.power_exponent_f32", 4)
      @rms_exponent.not_nil!.write([2.0_f32])
      # Keep graph-written RMS rows separate from CPU-initialized attention
      # index tables, which are reused on every encode.
      @rms_statistics = own_buffer("rms.statistics_f32", @config.tokens.to_i64 * 4)
      @q_norm_statistics = own_buffer("q_norm.statistics_f32",
        @config.tokens.to_i64 * @config.query_heads * 2 * 4)
      @k_norm_statistics = own_buffer("k_norm.statistics_f32",
        @config.tokens.to_i64 * @config.kv_heads * 2 * 4)
      @q_norm_inverse = own_buffer("q_norm.inverse_f32",
        @config.tokens.to_i64 * @config.query_heads * 2 * 4)
      @k_norm_inverse = own_buffer("k_norm.inverse_f32",
        @config.tokens.to_i64 * @config.kv_heads * 2 * 4)
    end

    private def allocate_captures!(selected : Array(String)) : Nil
      selected.each do |name|
        if name == "x"
          @stages[name] = @input
          next
        end
        if name == "output"
          @captures[name] = @core["output"]
          @stages[name] = @core["output"]
          next
        end
        kind = CAPTURE_BYTES[name]
        j = @config.tokens.to_i64
        d = @config.model_dim.to_i64
        f = @config.ff_dim.to_i64
        kv = @config.kv_heads.to_i64 * @config.head_dim
        bytes = case kind
                when :bf16_d           then j * d * 2
                when :bf16_kv          then j * kv * 2
                when :bf16_f           then j * f * 2
                when :f32_q            then j * d * 4
                when :f32_kv           then j * kv * 4
                when :f32_rows         then j * 4
                when :f32_q_norm_pair  then j * @config.query_heads * 2 * 4
                when :f32_kv_norm_pair then j * @config.kv_heads * 2 * 4
                else                        raise ArgumentError.new("unknown PixelUMM capture type")
                end
        buffer = own_buffer("capture.#{name}", bytes)
        @captures[name] = buffer
        @stages[name] = buffer
      end
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
        @native = PixelUMMDecoderBlockMPSFFI.plan_create(
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

    private def own_buffer(name : String, bytes : Int64) : ML::MetalBuffer
      raise ArgumentError.new("invalid PixelUMM buffer extent for #{name}") unless bytes > 0
      buffer = ML::MetalBuffer.new(bytes)
      @owned << buffer
      @named_bytes_by_name[name] = bytes
      buffer
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
end
