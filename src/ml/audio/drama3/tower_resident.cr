require "./tower"
require "./resident"

{% if flag?(:drama3_metal) && flag?(:darwin) && !flag?(:cpu_only) %}
module ML::Audio::Drama3
  RESIDENT_TOWER_SOURCE = {{ read_file("#{__DIR__}/kernels/tower_resident.metal") }}

  # The device-side result deliberately exposes buffers rather than eagerly
  # reading activations back to the host. Call Resident::Context#read only at
  # terminal output or when a reference trace callback is enabled.
  struct ResidentTowerOutput
    getter hidden : Resident::Buffer
    getter head_logits : Array(Resident::Buffer)

    def initialize(@hidden : Resident::Buffer, @head_logits : Array(Resident::Buffer))
    end
  end

  # Stateful resident counterpart to Tower. Each call advances the single
  # temporal offset shared by all layer KV caches, and leaves its command graph
  # pending in the supplied context so the caller can compose backbone work.
  class TowerResident
    private struct Layer
      getter embedding : Matrix
      getter enorm : Resident::Buffer
      getter hnorm : Resident::Buffer
      getter eh_proj : Matrix
      getter attention_norm : Resident::Buffer
      getter ffn_norm : Resident::Buffer
      getter wqkv : Matrix
      getter wo : Matrix
      getter q_norm : Resident::Buffer
      getter k_norm : Resident::Buffer
      getter w1 : Matrix
      getter w3 : Matrix
      getter w2 : Matrix
      getter final_layernorm : Resident::Buffer

      def initialize(
        @embedding : Matrix,
        @enorm : Resident::Buffer,
        @hnorm : Resident::Buffer,
        @eh_proj : Matrix,
        @attention_norm : Resident::Buffer,
        @ffn_norm : Resident::Buffer,
        @wqkv : Matrix,
        @wo : Matrix,
        @q_norm : Resident::Buffer,
        @k_norm : Resident::Buffer,
        @w1 : Matrix,
        @w3 : Matrix,
        @w2 : Matrix,
        @final_layernorm : Resident::Buffer,
      )
      end
    end

    private struct LayerCache
      getter keys : Resident::Buffer
      getter values : Resident::Buffer

      def initialize(@keys : Resident::Buffer, @values : Resident::Buffer)
      end
    end

    getter config : TowerConfig
    getter offset : Int32
    getter capacity : Int32

    @codebook_embeddings : Matrix
    @codebook_heads : Array(Matrix)
    @shared_layers : Array(Layer)
    @residual_layers : Array(Layer)
    @caches : Array(LayerCache)
    @store : WeightStore
    @context : Resident::Context
    @codebook_embedding_fusion : Bool

    def initialize(@config : TowerConfig, @store : WeightStore, @context : Resident::Context, @capacity : Int32, prefix : String = "tower")
      raise ArgumentError.new("tower cache capacity must be positive") unless @capacity > 0
      @offset = 0
      @prefix = prefix
      @codebook_embedding_fusion = case ENV["DRAMA3_TOWER_EMBEDDING_FUSION"]? || "0"
                                   when "0" then false
                                   when "1" then true
                                   else raise ArgumentError.new("DRAMA3_TOWER_EMBEDDING_FUSION must be 0 or 1")
                                   end
      @codebook_embeddings = matrix("codebook_embeddings.weight", @config.num_codebooks * @config.vocab_size, @config.dim)
      @shared_layers = Array(Layer).new(@config.shared_enrichment_layers) do |index|
        load_layer("shared_layers.#{index}", "semantic_embedding")
      end
      @residual_layers = Array(Layer).new(@config.num_codebooks - 1) do |index|
        load_layer("residual_layers.#{index}", "codebook_embedding")
      end
      @codebook_heads = Array(Matrix).new(@config.num_codebooks - 1) do |index|
        matrix("codebook_heads.#{index}.weight", @config.vocab_size, @config.dim)
      end
      cache_width = @config.n_local_heads * @capacity * @config.head_dim
      @caches = Array(LayerCache).new(@config.layer_count) do
        LayerCache.new(@context.alloc(cache_width), @context.alloc(cache_width))
      end
    end

    def reset! : Nil
      @offset = 0
    end

    # Small explicit upload helpers for the CPU/device boundary. Token and code
    # ids are carried as exact integral FP32 values in resident buffers.
    def upload_hidden(hidden : Vector) : Resident::Buffer
      validate_width!(hidden.size, @config.dim, "hidden")
      @context.upload(hidden)
    end

    def upload_frame(frame : Array(Int32)) : Resident::Buffer
      validate_frame!(frame, "frame")
      @context.upload(frame.map(&.to_f32))
    end

    def upload_token(token : Int32) : Resident::Buffer
      @context.upload([token.to_f32])
    end

    # Add already-consumed VQ code embeddings to one resident backbone text
    # embedding. No frame or activation values are read back.
    def embed_backbone(frame : Resident::Buffer, token_embedding : Resident::Buffer,
                       trace : TowerTraceCallback? = nil, trace_prefix : String = "tower.embed_backbone.row0") : Resident::Buffer
      validate_frame_buffer!(frame, "frame")
      validate_buffer_width!(token_embedding, @config.dim, "text embedding")
      trace_value(trace, "#{trace_prefix}.text_embedding", [1_i32, 1_i32, @config.dim], token_embedding)

      extra = codebook_sum(frame, 1, 1 + @config.vq_backbone_codebooks)
      # The CPU reference omits the sum and mixed checkpoints on non-semantic
      # frames. This branch is only for trace fidelity; ordinary execution has
      # no control readback.
      semantic = semantic_control(frame, trace)
      if semantic && @config.vq_backbone_codebooks > 0
        trace_value(trace, "#{trace_prefix}.code_embedding_sum", [1_i32, 1_i32, @config.dim], extra)
      end

      output = @context.alloc(@config.dim)
      scale = Math.sqrt((1 + @config.vq_backbone_codebooks).to_f64).to_f32
      @context.dispatch(
        "drama3_tower_embed_backbone",
        RESIDENT_TOWER_SOURCE,
        [frame, token_embedding, extra, output],
        [@config.vocab_size.to_u64, @config.dim.to_u64, @config.vq_backbone_codebooks.to_u64,
         @config.semantic_begin.to_i64.to_u64, @config.semantic_end.to_i64.to_u64,
         scale.unsafe_as(UInt32).to_u64],
        @config.dim,
      )
      trace_value(trace, "#{trace_prefix}.mixed", [1_i32, 1_i32, @config.dim], output) if semantic && @config.vq_backbone_codebooks > 0
      trace_value(trace, "#{trace_prefix}.output", [1_i32, 1_i32, @config.dim], output)
      output
    end

    # Process one teacher-forced pair [main,c0,...,cN-1]. The output and all
    # intermediate activations remain GPU buffers unless `trace` requests an
    # explicit diagnostic readback.
    def teacher_force(hidden : Resident::Buffer, previous_frame : Resident::Buffer, next_frame : Resident::Buffer,
                      trace : TowerTraceCallback? = nil, trace_prefix : String = "tower.teacher_force.row0") : ResidentTowerOutput
      Profile.measure("tower.teacher_force", category: "tower") do
        result = teacher_force_impl(hidden, previous_frame, next_frame, trace, trace_prefix)
        # Component profiling includes GPU completion. Normal inference can
        # compose the next backbone operation in the same serial command buffer.
        @context.finish if Profile.enabled?
        result
      end
    end

    private def teacher_force_impl(hidden : Resident::Buffer, previous_frame : Resident::Buffer, next_frame : Resident::Buffer,
                                   trace : TowerTraceCallback?, trace_prefix : String) : ResidentTowerOutput
      validate_call!(hidden, previous_frame)
      validate_frame_buffer!(next_frame, "next_frame")
      ensure_capacity!
      trace_value(trace, "#{trace_prefix}.hidden_input", [1_i32, 1_i32, @config.dim], hidden)

      unseen = codebook_sum(previous_frame, 1 + @config.vq_backbone_codebooks, @config.num_codebooks)
      trace_value(trace, "#{trace_prefix}.unseen_embedding", [1_i32, 1_i32, @config.dim], unseen)
      main = next_frame.slice(0, 1)
      state = hidden
      cache_index = 0

      @shared_layers.each_with_index do |layer, index|
        code = clipped_code(next_frame.slice(1, 1))
        embedding = gated_embedding(layer.embedding, code, main)
        embedding = @context.add(embedding, unseen) if index == 0
        layer_prefix = "#{trace_prefix}.shared_layers.#{index}"
        state = run_layer(layer, state, embedding, @caches[cache_index], trace, layer_prefix)
        cache_index += 1
      end

      head_logits = Array(Resident::Buffer).new(@residual_layers.size)
      @residual_layers.each_with_index do |layer, index|
        code = clipped_code(next_frame.slice(index + 1, 1))
        embedding = gated_embedding(layer.embedding, code, main)
        layer_prefix = "#{trace_prefix}.residual_layers.#{index}"
        state = run_layer(layer, state, embedding, @caches[cache_index], trace, layer_prefix)
        logits = @context.matvec(@codebook_heads[index], state)
        trace_value(trace, "#{trace_prefix}.head_logits.#{index}", [1_i32, @config.vocab_size], logits)
        head_logits << logits
        cache_index += 1
      end

      @offset += 1
      trace_value(trace, "#{trace_prefix}.hidden_output", [1_i32, 1_i32, @config.dim], state)
      ResidentTowerOutput.new(state, head_logits)
    end

    # Generate [main,c0,...,cN-1] on device. The sampler callback receives
    # resident logits and must return one resident scalar code id; ctx.sample
    # implements GPU FP32 top-k/top-p sampling. Cutoff/draw boundaries may
    # differ from the CPU Float64 sampler.
    def sample_frame(main : Resident::Buffer, hidden : Resident::Buffer, previous_frame : Resident::Buffer,
                     trace : TowerTraceCallback? = nil, trace_prefix : String = "tower.decode.row0",
                     output_frame : Resident::Buffer? = nil,
                     &sample : Resident::Buffer -> Resident::Buffer) : Resident::Buffer
      Profile.measure("tower.sample_frame", category: "tower") do
        result = sample_frame_impl(main, hidden, previous_frame, output_frame, trace, trace_prefix) do |logits|
          yield logits
        end
        # Component profiling includes GPU completion. Normal inference can
        # compose the next backbone operation in the same serial command buffer.
        @context.finish if Profile.enabled?
        result
      end
    end

    private def sample_frame_impl(main : Resident::Buffer, hidden : Resident::Buffer, previous_frame : Resident::Buffer,
                                  output_frame : Resident::Buffer?, trace : TowerTraceCallback?, trace_prefix : String,
                                  &sample : Resident::Buffer -> Resident::Buffer) : Resident::Buffer
      validate_buffer_width!(main, 1, "main token")
      validate_call!(hidden, previous_frame)
      ensure_capacity!
      if frame = output_frame
        validate_frame_buffer!(frame, "output_frame")
        validate_disjoint!(frame, main, "output_frame", "main token")
        validate_disjoint!(frame, hidden, "output_frame", "hidden")
        validate_disjoint!(frame, previous_frame, "output_frame", "previous_frame")
      end
      trace_value(trace, "#{trace_prefix}.hidden_input", [1_i32, 1_i32, @config.dim], hidden)

      frame = output_frame || @context.alloc(@config.num_codebooks + 1)
      @context.dispatch(
        "drama3_tower_initial_frame",
        RESIDENT_TOWER_SOURCE,
        [main, frame],
        [(@config.num_codebooks + 1).to_u64, @config.semantic_begin.to_i64.to_u64, @config.vocab_size.to_u64],
        @config.num_codebooks + 1,
      )
      unseen = codebook_sum(previous_frame, 1 + @config.vq_backbone_codebooks, @config.num_codebooks)
      trace_value(trace, "#{trace_prefix}.unseen_embedding", [1_i32, 1_i32, @config.dim], unseen)
      state = hidden
      cache_index = 0

      @shared_layers.each_with_index do |layer, index|
        code = clipped_code(frame.slice(1, 1))
        embedding = gated_embedding(layer.embedding, code, main)
        embedding = @context.add(embedding, unseen) if index == 0
        layer_prefix = "#{trace_prefix}.shared_layers.#{index}"
        state = run_layer(layer, state, embedding, @caches[cache_index], trace, layer_prefix)
        cache_index += 1
      end

      @residual_layers.each_with_index do |layer, index|
        code = clipped_code(frame.slice(index + 1, 1))
        embedding = gated_embedding(layer.embedding, code, main)
        layer_prefix = "#{trace_prefix}.residual_layers.#{index}"
        state = run_layer(layer, state, embedding, @caches[cache_index], trace, layer_prefix)
        logits = @context.matvec(@codebook_heads[index], state)
        trace_value(trace, "#{trace_prefix}.head_logits.#{index}", [1_i32, @config.vocab_size], logits)
        sampled_code = yield logits
        validate_buffer_width!(sampled_code, 1, "sampled code")
        output_code = frame.slice(index + 2, 1)
        validate_disjoint!(frame, sampled_code, "output_frame", "sampled code")
        @context.copy(sampled_code, output_code)
        cache_index += 1
      end

      @offset += 1
      trace_value(trace, "#{trace_prefix}.hidden_output", [1_i32, 1_i32, @config.dim], state)
      frame
    end

    private def run_layer(layer : Layer, hidden : Resident::Buffer, embedding : Resident::Buffer,
                          cache : LayerCache, trace : TowerTraceCallback?, prefix : String) : Resident::Buffer
      trace_value(trace, "#{prefix}.embedding", [1_i32, 1_i32, @config.dim], embedding)
      normalized_embedding = @context.rms(embedding, layer.enorm, @config.norm_eps, true)
      normalized_hidden = @context.rms(hidden, layer.hnorm, @config.norm_eps, true)
      trace_value(trace, "#{prefix}.enorm", [1_i32, 1_i32, @config.dim], normalized_embedding)
      trace_value(trace, "#{prefix}.hnorm", [1_i32, 1_i32, @config.dim], normalized_hidden)

      combined = @context.alloc(@config.dim * 2)
      @context.dispatch("drama3_tower_concat", RESIDENT_TOWER_SOURCE, [normalized_embedding, normalized_hidden, combined],
        [@config.dim.to_u64], @config.dim * 2)
      trace_value(trace, "#{prefix}.eh_concat", [1_i32, 1_i32, @config.dim * 2], combined)
      mixed = @context.matvec(layer.eh_proj, combined)
      trace_value(trace, "#{prefix}.eh_proj", [1_i32, 1_i32, @config.dim], mixed)

      normalized_attention_input = @context.rms(mixed, layer.attention_norm, @config.norm_eps, true)
      trace_value(trace, "#{prefix}.attention_norm", [1_i32, 1_i32, @config.dim], normalized_attention_input)
      attended = attention(layer, normalized_attention_input, cache, trace, "#{prefix}.attention")
      state = @context.add(mixed, attended)
      trace_value(trace, "#{prefix}.attention_residual", [1_i32, 1_i32, @config.dim], state)
      normalized_feed_input = @context.rms(state, layer.ffn_norm, @config.norm_eps, true)
      trace_value(trace, "#{prefix}.ffn_norm", [1_i32, 1_i32, @config.dim], normalized_feed_input)
      feed = feed_forward(layer, normalized_feed_input, trace, "#{prefix}.feed_forward")
      residual = @context.add(state, feed)
      trace_value(trace, "#{prefix}.ffn_residual", [1_i32, 1_i32, @config.dim], residual)
      output = @context.rms(residual, layer.final_layernorm, @config.norm_eps, true)
      @context.guard(output)
      trace_value(trace, "#{prefix}.output", [1_i32, 1_i32, @config.dim], output)
      output
    end

    private def attention(layer : Layer, input : Resident::Buffer, cache : LayerCache,
                          trace : TowerTraceCallback?, prefix : String) : Resident::Buffer
      q_width = @config.n_head * @config.head_dim
      kv_width = @config.n_local_heads * @config.head_dim
      projected = @context.matvec(layer.wqkv, input)
      q = projected.slice(0, q_width)
      gate = projected.slice(q_width, q_width)
      k = projected.slice(q_width * 2, kv_width)
      v = projected.slice(q_width * 2 + kv_width, kv_width)
      trace_value(trace, "#{prefix}.q_projected", [1_i32, 1_i32, @config.n_head, @config.head_dim], q)
      trace_value(trace, "#{prefix}.gate", [1_i32, 1_i32, @config.n_head, @config.head_dim], gate)
      trace_value(trace, "#{prefix}.k_projected", [1_i32, 1_i32, @config.n_local_heads, @config.head_dim], k)
      trace_value(trace, "#{prefix}.v_projected", [1_i32, 1_i32, @config.n_local_heads, @config.head_dim], v)

      q_norm = @context.rms(q, layer.q_norm, @config.norm_eps, true, @config.n_head)
      k_norm = @context.rms(k, layer.k_norm, @config.norm_eps, true, @config.n_local_heads)
      trace_value(trace, "#{prefix}.q_norm", [1_i32, @config.n_head, 1_i32, @config.head_dim], q_norm)
      trace_value(trace, "#{prefix}.k_norm", [1_i32, @config.n_local_heads, 1_i32, @config.head_dim], k_norm)
      @context.rope!(q_norm, @config.n_head, @config.head_dim, @config.rotary_dim, @offset, @config.rope_base, true)
      @context.rope!(k_norm, @config.n_local_heads, @config.head_dim, @config.rotary_dim, @offset, @config.rope_base, true)
      trace_value(trace, "#{prefix}.q_rope", [1_i32, @config.n_head, 1_i32, @config.head_dim], q_norm)
      trace_value(trace, "#{prefix}.k_rope", [1_i32, @config.n_local_heads, 1_i32, @config.head_dim], k_norm)

      cache_position = @offset
      @context.dispatch(
        "drama3_tower_append_kv", RESIDENT_TOWER_SOURCE,
        [k_norm, v, cache.keys, cache.values],
        [@config.n_local_heads.to_u64, @config.head_dim.to_u64, @capacity.to_u64, cache_position.to_u64],
        kv_width,
      )
      length = cache_position + 1
      trace_cache(trace, "#{prefix}.key_cache", cache.keys, length)
      trace_cache(trace, "#{prefix}.value_cache", cache.values, length)

      scores = @context.alloc(@config.n_head * length)
      scale = (1.0_f64 / Math.sqrt(@config.head_dim.to_f64)).to_f32
      @context.dispatch(
        "drama3_tower_attention_scores", RESIDENT_TOWER_SOURCE,
        [q_norm, cache.keys, scores],
        [@config.n_head.to_u64, @config.n_local_heads.to_u64, @config.head_dim.to_u64,
         (@config.n_head // @config.n_local_heads).to_u64, length.to_u64, @capacity.to_u64,
         scale.unsafe_as(UInt32).to_u64],
        @config.n_head * length,
      )
      context = @context.alloc(q_width)
      @context.dispatch(
        "drama3_tower_attention_context", RESIDENT_TOWER_SOURCE,
        [scores, cache.values, gate, context],
        [@config.n_head.to_u64, @config.head_dim.to_u64,
         (@config.n_head // @config.n_local_heads).to_u64, length.to_u64, @capacity.to_u64],
        q_width,
      )
      trace_value(trace, "#{prefix}.gated_context", [1_i32, 1_i32, q_width], context)
      output = @context.matvec(layer.wo, context)
      trace_value(trace, "#{prefix}.output", [1_i32, 1_i32, @config.dim], output)
      output
    end

    private def feed_forward(layer : Layer, input : Resident::Buffer, trace : TowerTraceCallback?, prefix : String) : Resident::Buffer
      up = @context.matvec(layer.w1, input)
      gate = @context.matvec(layer.w3, input)
      activated = @context.silu(up)
      product = @context.mul(activated, gate)
      trace_value(trace, "#{prefix}.w1", [1_i32, 1_i32, @config.intermediate_size], up)
      trace_value(trace, "#{prefix}.w3", [1_i32, 1_i32, @config.intermediate_size], gate)
      trace_value(trace, "#{prefix}.silu", [1_i32, 1_i32, @config.intermediate_size], activated)
      trace_value(trace, "#{prefix}.product", [1_i32, 1_i32, @config.intermediate_size], product)
      output = @context.matvec(layer.w2, product)
      trace_value(trace, "#{prefix}.w2", [1_i32, 1_i32, @config.dim], output)
      output
    end

    private def codebook_sum(frame : Resident::Buffer, first_codebook : Int32, end_codebook : Int32) : Resident::Buffer
      @context.codebook_sum(@codebook_embeddings, frame, first_codebook, end_codebook,
        @config.vocab_size, @config.semantic_begin, @config.semantic_end,
        RESIDENT_TOWER_SOURCE, @codebook_embedding_fusion)
    end

    private def clipped_code(code : Resident::Buffer) : Resident::Buffer
      clipped = @context.alloc(1)
      @context.dispatch("drama3_tower_clip_code", RESIDENT_TOWER_SOURCE, [code, clipped], [@config.vocab_size.to_u64], 1)
      clipped
    end

    private def gated_embedding(table : Matrix, code : Resident::Buffer, main : Resident::Buffer) : Resident::Buffer
      embedding = @context.embedding(table, code)
      masked = @context.alloc(@config.dim)
      @context.dispatch(
        "drama3_tower_mask_semantic", RESIDENT_TOWER_SOURCE,
        [embedding, main, masked],
        [@config.dim.to_u64, @config.semantic_begin.to_i64.to_u64, @config.semantic_end.to_i64.to_u64],
        @config.dim,
      )
      masked
    end

    private def trace_cache(trace : TowerTraceCallback?, name : String, cache : Resident::Buffer, length : Int32) : Nil
      return unless trace
      compact = @context.alloc(@config.n_local_heads * length * @config.head_dim)
      @context.dispatch(
        "drama3_tower_compact_cache", RESIDENT_TOWER_SOURCE, [cache, compact],
        [@config.n_local_heads.to_u64, @config.head_dim.to_u64, @capacity.to_u64, length.to_u64], compact.count,
      )
      trace_value(trace, name, [1_i32, @config.n_local_heads, length, @config.head_dim], compact)
    end

    private def trace_value(trace : TowerTraceCallback?, name : String, shape : Array(Int32), value : Resident::Buffer) : Nil
      trace.try(&.call(name, shape, @context.read(value, "diagnostic")))
    end

    private def semantic_control(frame : Resident::Buffer, trace : TowerTraceCallback?) : Bool
      return false unless trace
      token = @context.read(frame.slice(0, 1), "diagnostic")[0].to_i32
      token >= @config.semantic_begin && token <= @config.semantic_end
    end

    private def ensure_capacity! : Nil
      raise ArgumentError.new("tower resident cache capacity #{@capacity} exceeded at offset #{@offset}") if @offset >= @capacity
    end

    private def validate_call!(hidden : Resident::Buffer, previous_frame : Resident::Buffer) : Nil
      validate_buffer_width!(hidden, @config.dim, "hidden")
      validate_frame_buffer!(previous_frame, "previous_frame")
    end

    private def validate_frame_buffer!(frame : Resident::Buffer, name : String) : Nil
      validate_buffer_width!(frame, @config.num_codebooks + 1, name)
    end

    private def validate_buffer_width!(buffer : Resident::Buffer, expected : Int32, name : String) : Nil
      raise ArgumentError.new("#{name} has width #{buffer.count}; expected #{expected}") unless buffer.count == expected
    end

    private def validate_disjoint!(destination : Resident::Buffer, source : Resident::Buffer,
                                   destination_name : String, source_name : String) : Nil
      return unless destination.buffer.handle.address == source.buffer.handle.address

      destination_start = destination.offset
      destination_end = destination_start + destination.count.to_i64 * 4_i64
      source_start = source.offset
      source_end = source_start + source.count.to_i64 * 4_i64
      if destination_start < source_end && source_start < destination_end
        raise ArgumentError.new("#{destination_name} overlaps #{source_name}")
      end
    end

    private def validate_frame!(frame : Array(Int32), name : String) : Nil
      expected = @config.num_codebooks + 1
      raise ArgumentError.new("#{name} must have #{expected} entries [main, c0, ..., cN-1]") unless frame.size == expected
    end

    private def validate_width!(actual : Int32, expected : Int32, name : String) : Nil
      raise ArgumentError.new("#{name} has width #{actual}; expected #{expected}") unless actual == expected
    end

    private def load_layer(layer_path : String, embedding_name : String) : Layer
      path = "#{layer_path}."
      Layer.new(
        matrix("#{path}#{embedding_name}.weight", @config.vocab_size, @config.dim),
        vector("#{path}enorm.weight", @config.dim),
        vector("#{path}hnorm.weight", @config.dim),
        matrix("#{path}eh_proj.weight", @config.dim, @config.dim * 2),
        vector("#{path}transformer_layer.attention_norm.weight", @config.dim),
        vector("#{path}transformer_layer.ffn_norm.weight", @config.dim),
        matrix("#{path}transformer_layer.attention.wqkv.weight", @config.n_head * @config.head_dim * 2 + @config.n_local_heads * @config.head_dim * 2, @config.dim),
        matrix("#{path}transformer_layer.attention.wo.weight", @config.dim, @config.n_head * @config.head_dim),
        vector("#{path}transformer_layer.attention.q_norm.weight", @config.head_dim),
        vector("#{path}transformer_layer.attention.k_norm.weight", @config.head_dim),
        matrix("#{path}transformer_layer.feed_forward.w1.weight", @config.intermediate_size, @config.dim),
        matrix("#{path}transformer_layer.feed_forward.w3.weight", @config.intermediate_size, @config.dim),
        matrix("#{path}transformer_layer.feed_forward.w2.weight", @config.dim, @config.intermediate_size),
        vector("#{path}final_layernorm.weight", @config.dim),
      )
    end

    private def matrix(suffix : String, rows : Int32, cols : Int32) : Matrix
      name = weight_name(suffix)
      value = @store.matrix(name)
      raise ArgumentError.new("#{name} has shape #{value.rows}x#{value.cols}; expected #{rows}x#{cols}") unless value.rows == rows && value.cols == cols
      value
    end

    private def vector(suffix : String, size : Int32) : Resident::Buffer
      name = weight_name(suffix)
      tensor = @store.tensor(name)
      raise ArgumentError.new("#{name} has shape #{tensor.shape}; expected [#{size}]") unless tensor.shape == [size]
      @context.tensor(tensor)
    end

    private def weight_name(suffix : String) : String
      @prefix.empty? ? suffix : "#{@prefix}.#{suffix}"
    end
  end
end
{% end %}
