require "json"
require "./native"

# Behavior is ported from mlx-audio's fish_omni/tower.py at
# e7488935ec2ebfafd38ef2b58e290d79f3d05dc7 (MIT, Copyright (c) 2024 Prince
# Canuma). Preserve the upstream MIT notice in distributions containing this
# derived implementation.
module ML
  module Audio
    module Drama3
      alias TowerTraceCallback = Proc(String, Array(Int32), Vector, Nil)

      # Configuration for the Drama 3 autoregressive audio-code tower.
      struct TowerConfig
        getter vocab_size : Int32
        getter num_codebooks : Int32
        getter dim : Int32
        getter n_head : Int32
        getter n_local_heads : Int32
        getter head_dim : Int32
        getter intermediate_size : Int32
        getter norm_eps : Float64
        getter rope_base : Float64
        getter rotary_percent : Float64
        getter shared_enrichment_layers : Int32
        getter vq_backbone_codebooks : Int32
        getter semantic_begin : Int32
        getter semantic_end : Int32

        def initialize(
          @vocab_size : Int32 = 4096,
          @num_codebooks : Int32 = 10,
          @dim : Int32 = 2048,
          @n_head : Int32 = 16,
          @n_local_heads : Int32 = 2,
          @head_dim : Int32 = 256,
          @intermediate_size : Int32 = 6144,
          @norm_eps : Float64 = 1e-6,
          @rope_base : Float64 = 10_000_000.0,
          @rotary_percent : Float64 = 0.25,
          @shared_enrichment_layers : Int32 = 4,
          @vq_backbone_codebooks : Int32 = 1,
          @semantic_begin : Int32 = 0,
          semantic_end : Int32? = nil,
        )
          @semantic_end = semantic_end || (@semantic_begin + @vocab_size - 1)
          validate!
        end

        def rotary_dim : Int32
          (@head_dim.to_f64 * @rotary_percent).to_i32
        end

        def layer_count : Int32
          @shared_enrichment_layers + @num_codebooks - 1
        end

        def self.from_json(source : String) : TowerConfig
          root = JSON.parse(source).as_h
          values = root["audio_decoder_config"]?.try(&.as_h) || root
          validate_architecture_flags!(values)
          vocab_size = int_value(values, "vocab_size", 4096_i32)
          semantic_begin = int_value(root, "semantic_start_token_id", int_value(values, "semantic_begin", 0_i32))
          semantic_end = int_value(root, "semantic_end_token_id", int_value(values, "semantic_end", semantic_begin + vocab_size - 1))
          new(
            vocab_size: vocab_size,
            num_codebooks: int_value(values, "num_codebooks", 10_i32),
            dim: int_value(values, "dim", 2048_i32),
            n_head: int_value(values, "n_head", 16_i32),
            n_local_heads: int_value(values, "n_local_heads", 2_i32),
            head_dim: int_value(values, "head_dim", 256_i32),
            intermediate_size: int_value(values, "intermediate_size", 6144_i32),
            norm_eps: float_value(values, "norm_eps", 1e-6),
            rope_base: float_value(values, "rope_base", 10_000_000.0),
            rotary_percent: float_value(values, "rotary_percent", 0.25),
            shared_enrichment_layers: int_value(values, "shared_enrichment_layers", 4_i32),
            vq_backbone_codebooks: int_value(values, "vq_backbone_codebooks", 1_i32),
            semantic_begin: semantic_begin,
            semantic_end: semantic_end,
          )
        end

        private def self.validate_architecture_flags!(values : Hash(String, JSON::Any)) : Nil
          {
            "attention_o_bias"              => false,
            "attention_qkv_bias"            => false,
            "attention_output_gate"         => true,
            "layernorm_zero_centered_gamma" => true,
            "rotary_interleaved"            => true,
            "mtp_use_swa"                   => false,
            "use_moe"                       => false,
          }.each do |name, expected|
            next unless value = values[name]?
            actual = value.raw
            unless actual.is_a?(Bool) && actual == expected
              raise ArgumentError.new("#{name} must be #{expected} for the pinned tower implementation")
            end
          end

          {
            "vq_feature_placement" => "backbone_partial",
            "decoder_type"         => "codebook_mtp_shared",
          }.each do |name, expected|
            next unless value = values[name]?
            actual = value.raw
            unless actual.is_a?(String) && actual == expected
              raise ArgumentError.new("#{name} must be #{expected.inspect} for the pinned tower implementation")
            end
          end
        end

        private def self.int_value(values : Hash(String, JSON::Any), name : String, default : Int32) : Int32
          if value = values[name]?
            value.as_i.to_i32
          else
            default
          end
        end

        private def self.float_value(values : Hash(String, JSON::Any), name : String, default : Float64) : Float64
          if value = values[name]?
            case number = value.raw
            when Float64
              number
            when Int64
              number.to_f64
            else
              raise ArgumentError.new("#{name} must be numeric")
            end
          else
            default
          end
        end

        private def validate!
          raise ArgumentError.new("vocab_size must be positive") unless @vocab_size > 0
          raise ArgumentError.new("num_codebooks must be positive") unless @num_codebooks > 0
          raise ArgumentError.new("model and attention dimensions must be positive") unless @dim > 0 && @head_dim > 0 && @n_head > 0 && @n_local_heads > 0
          raise ArgumentError.new("n_head must be divisible by n_local_heads") unless @n_head % @n_local_heads == 0
          raise ArgumentError.new("intermediate_size must be positive") unless @intermediate_size > 0
          raise ArgumentError.new("at least one shared enrichment layer is required") unless @shared_enrichment_layers > 0
          raise ArgumentError.new("vq_backbone_codebooks must be in 0..num_codebooks-1") unless @vq_backbone_codebooks >= 0 && @vq_backbone_codebooks < @num_codebooks
          raise ArgumentError.new("semantic_end must not precede semantic_begin") unless @semantic_end >= @semantic_begin
          raise ArgumentError.new("norm_eps must be finite and positive") unless @norm_eps.finite? && @norm_eps > 0.0
          raise ArgumentError.new("rope_base must be finite and positive") unless @rope_base.finite? && @rope_base > 0.0
          raise ArgumentError.new("rotary_percent must be finite") unless @rotary_percent.finite?
          rd = rotary_dim
          raise ArgumentError.new("rotary_dim must be positive, even, and no larger than head_dim") unless rd > 0 && rd.even? && rd <= @head_dim
        end
      end

      # Temporal KV state for one tower layer. Every entry is one time position,
      # with local heads concatenated in checkpoint order.
      class TowerCache
        getter keys : Array(Vector)
        getter values : Array(Vector)

        def initialize
          @keys = [] of Vector
          @values = [] of Vector
        end

        def offset : Int32
          @keys.size.to_i32
        end

        def append(key : Vector, value : Vector) : Nil
          raise ArgumentError.new("key/value cache widths differ") unless key.size == value.size
          @keys << key.dup
          @values << value.dup
        end

        def clear : Nil
          @keys.clear
          @values.clear
        end
      end

      # Final hidden state and logits for codebooks 1..N-1 for one frame pair.
      struct TowerOutput
        getter hidden : Vector
        getter head_logits : Array(Vector)

        def initialize(@hidden : Vector, @head_logits : Array(Vector))
        end
      end

      # Autoregressively sampled residual codes for one frame, including c0.
      struct TowerSample
        getter codes : Array(Int32)
        getter head_logits : Array(Vector)

        def initialize(@codes : Array(Int32), @head_logits : Array(Vector))
        end
      end

      private class TowerLayerWeights
        getter embedding : Matrix
        getter enorm : Vector
        getter hnorm : Vector
        getter eh_proj : Matrix
        getter attention_norm : Vector
        getter ffn_norm : Vector
        getter wqkv : Matrix
        getter wo : Matrix
        getter q_norm : Vector
        getter k_norm : Vector
        getter w1 : Matrix
        getter w3 : Matrix
        getter w2 : Matrix
        getter final_layernorm : Vector

        def initialize(
          @embedding : Matrix,
          @enorm : Vector,
          @hnorm : Vector,
          @eh_proj : Matrix,
          @attention_norm : Vector,
          @ffn_norm : Vector,
          @wqkv : Matrix,
          @wo : Matrix,
          @q_norm : Vector,
          @k_norm : Vector,
          @w1 : Matrix,
          @w3 : Matrix,
          @w2 : Matrix,
          @final_layernorm : Vector,
        )
        end
      end

      # CPU reference tower for teacher forcing and per-codebook head logits.
      # The implementation intentionally processes one time row at a time so
      # temporal cache offsets and causal attention have a single representation.
      class Tower
        getter config : TowerConfig
        getter codebook_embeddings : Matrix

        def initialize(@config : TowerConfig, @store : WeightStore, prefix : String = "tower")
          @prefix = prefix
          @codebook_embeddings = matrix("codebook_embeddings.weight", @config.num_codebooks * @config.vocab_size, @config.dim)
          @shared_layers = Array(TowerLayerWeights).new(@config.shared_enrichment_layers) do |index|
            load_layer("shared_layers.#{index}", "semantic_embedding")
          end
          @residual_layers = Array(TowerLayerWeights).new(@config.num_codebooks - 1) do |index|
            load_layer("residual_layers.#{index}", "codebook_embedding")
          end
          @codebook_heads = Array(Matrix).new(@config.num_codebooks - 1) do |index|
            matrix("codebook_heads.#{index}.weight", @config.vocab_size, @config.dim)
          end
        end

        def make_cache : Array(TowerCache)
          Array(TowerCache).new(@config.layer_count) { TowerCache.new }
        end

        # Process one known previous/current frame pair. `previous_frame` and
        # `next_frame` are [main, c0, ..., cN-1]; each layer appends one KV row.
        def teacher_force(hidden : Vector, previous_frame : Array(Int32), next_frame : Array(Int32), caches : Array(TowerCache), trace : TowerTraceCallback? = nil, trace_prefix : String = "tower.teacher_force.row0") : TowerOutput
          Profile.measure("tower.teacher_force", category: "tower") do
            teacher_force_impl(hidden, previous_frame, next_frame, caches, trace, trace_prefix)
          end
        end

        private def teacher_force_impl(hidden : Vector, previous_frame : Array(Int32), next_frame : Array(Int32), caches : Array(TowerCache), trace : TowerTraceCallback?, trace_prefix : String) : TowerOutput
          validate_vector!(hidden, @config.dim, "hidden")
          validate_frame!(previous_frame, "previous_frame")
          validate_frame!(next_frame, "next_frame")
          validate_caches!(caches)
          trace_value(trace, "#{trace_prefix}.hidden_input", [1_i32, 1_i32, @config.dim], hidden)

          semantic = semantic?(next_frame[0])
          unseen = if semantic?(previous_frame[0])
                     embedding_sum(previous_frame, 1 + @config.vq_backbone_codebooks, @config.num_codebooks)
                   else
                     zeros(@config.dim)
                   end
          trace_value(trace, "#{trace_prefix}.unseen_embedding", [1_i32, 1_i32, @config.dim], unseen)

          state = hidden
          cache_index = 0
          @shared_layers.each_with_index do |layer, index|
            embedding = embedding_row(layer.embedding, next_frame[1])
            zero!(embedding) unless semantic
            if index == 0
              embedding = Ops.add(embedding, unseen)
            end
            layer_prefix = "#{trace_prefix}.shared_layers.#{index}"
            state = run_layer(layer, state, embedding, caches[cache_index], trace, layer_prefix)
            cache_index += 1
          end

          head_logits = Array(Vector).new(@residual_layers.size)
          @residual_layers.each_with_index do |layer, index|
            # A residual layer embeds the preceding code and its head predicts
            # the next one: layer 0 consumes c0 and predicts c1.
            code = next_frame[index + 1]
            embedding = embedding_row(layer.embedding, code)
            zero!(embedding) unless semantic
            layer_prefix = "#{trace_prefix}.residual_layers.#{index}"
            state = run_layer(layer, state, embedding, caches[cache_index], trace, layer_prefix)
            logits = Ops.linear(state, @codebook_heads[index])
            trace_value(trace, "#{trace_prefix}.head_logits.#{index}", [1_i32, @config.vocab_size], logits)
            head_logits << logits
            cache_index += 1
          end

          trace_value(trace, "#{trace_prefix}.hidden_output", [1_i32, 1_i32, @config.dim], state)
          TowerOutput.new(state, head_logits)
        end

        # Convenience spelling for callers that already step the backbone one
        # row at a time. `next_frame` supplies teacher-forced code inputs.
        def decode_step(hidden : Vector, previous_frame : Array(Int32), next_frame : Array(Int32), caches : Array(TowerCache)) : TowerOutput
          teacher_force(hidden, previous_frame, next_frame, caches)
        end

        # Generate codes 0..N-1 for `main`, consuming each sampled residual
        # code in the following codebook layer. `previous_frame` supplies the
        # unseen code embeddings from the preceding backbone frame.
        def sample_frame(main : Int32, hidden : Vector, previous_frame : Array(Int32), caches : Array(TowerCache), trace : TowerTraceCallback? = nil, trace_prefix : String = "tower.decode.row0", &sample : Vector -> Int32) : TowerSample
          Profile.measure("tower.sample_frame", category: "tower") do
            sample_frame_impl(main, hidden, previous_frame, caches, trace, trace_prefix) { |logits| yield logits }
          end
        end

        private def sample_frame_impl(main : Int32, hidden : Vector, previous_frame : Array(Int32), caches : Array(TowerCache), trace : TowerTraceCallback?, trace_prefix : String, &sample : Vector -> Int32) : TowerSample
          validate_vector!(hidden, @config.dim, "hidden")
          validate_frame!(previous_frame, "previous_frame")
          validate_caches!(caches)
          trace_value(trace, "#{trace_prefix}.hidden_input", [1_i32, 1_i32, @config.dim], hidden)

          semantic = semantic?(main)
          code = clip_code(main - @config.semantic_begin)
          unseen = if semantic?(previous_frame[0])
                     embedding_sum(previous_frame, 1 + @config.vq_backbone_codebooks, @config.num_codebooks)
                   else
                     zeros(@config.dim)
                   end
          trace_value(trace, "#{trace_prefix}.unseen_embedding", [1_i32, 1_i32, @config.dim], unseen)

          state = hidden
          cache_index = 0
          @shared_layers.each_with_index do |layer, index|
            embedding = embedding_row(layer.embedding, code)
            zero!(embedding) unless semantic
            embedding = Ops.add(embedding, unseen) if index == 0
            layer_prefix = "#{trace_prefix}.shared_layers.#{index}"
            state = run_layer(layer, state, embedding, caches[cache_index], trace, layer_prefix)
            cache_index += 1
          end

          codes = [code]
          head_logits = Array(Vector).new(@residual_layers.size)
          @residual_layers.each_with_index do |layer, index|
            embedding = embedding_row(layer.embedding, codes.last)
            zero!(embedding) unless semantic
            layer_prefix = "#{trace_prefix}.residual_layers.#{index}"
            state = run_layer(layer, state, embedding, caches[cache_index], trace, layer_prefix)
            logits = Ops.linear(state, @codebook_heads[index])
            trace_value(trace, "#{trace_prefix}.head_logits.#{index}", [1_i32, @config.vocab_size], logits)
            head_logits << logits
            sampled_code = yield logits
            unless sampled_code >= 0 && sampled_code < @config.vocab_size
              raise ArgumentError.new("sampled code #{sampled_code} outside 0...#{@config.vocab_size}")
            end
            codes << sampled_code
            cache_index += 1
          end

          trace_value(trace, "#{trace_prefix}.hidden_output", [1_i32, 1_i32, @config.dim], state)
          TowerSample.new(codes, head_logits)
        end

        # Sequence helper. Rows are consumed in order, so caches can be reused
        # across later chunks and split-prefill behavior matches one-row stepping.
        def prefill(hidden_rows : Array(Vector), previous_frames : Array(Array(Int32)), next_frames : Array(Array(Int32)), caches : Array(TowerCache), trace : TowerTraceCallback? = nil) : Array(TowerOutput)
          count = hidden_rows.size
          raise ArgumentError.new("prefill row counts differ") unless previous_frames.size == count && next_frames.size == count
          outputs = Array(TowerOutput).new(count)
          count.times do |index|
            outputs << teacher_force(hidden_rows[index], previous_frames[index], next_frames[index], caches, trace, "tower.prefill.row#{index}")
          end
          outputs
        end

        # Add embeddings for codebooks already consumed by the backbone, but
        # only on semantic rows, matching AudioDecoder.embed_backbone.
        def embed_backbone(frames : Array(Array(Int32)), text_embeddings : Array(Vector), trace : TowerTraceCallback? = nil) : Array(Vector)
          raise ArgumentError.new("frame and text row counts differ") unless frames.size == text_embeddings.size
          frames.each_with_index.map do |frame, index|
            validate_frame!(frame, "frame")
            text = text_embeddings[index]
            validate_vector!(text, @config.dim, "text embedding")
            prefix = "tower.embed_backbone.row#{index}"
            trace_value(trace, "#{prefix}.text_embedding", [1_i32, 1_i32, @config.dim], text)
            unless semantic?(frame[0]) && @config.vq_backbone_codebooks > 0
              trace_value(trace, "#{prefix}.output", [1_i32, 1_i32, @config.dim], text)
              next text.dup
            end
            extra = embedding_sum(frame, 1, 1 + @config.vq_backbone_codebooks)
            trace_value(trace, "#{prefix}.code_embedding_sum", [1_i32, 1_i32, @config.dim], extra)
            scale = Math.sqrt((1 + @config.vq_backbone_codebooks).to_f64).to_f32
            mixed = Array(Float32).new(@config.dim) { |dimension| (text[dimension] + extra[dimension]) / scale }
            trace_value(trace, "#{prefix}.mixed", [1_i32, 1_i32, @config.dim], mixed)
            trace_value(trace, "#{prefix}.output", [1_i32, 1_i32, @config.dim], mixed)
            mixed
          end.to_a
        end

        private def load_layer(layer_path : String, embedding_name : String) : TowerLayerWeights
          path = "#{layer_path}."
          TowerLayerWeights.new(
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

        private def vector(suffix : String, size : Int32) : Vector
          name = weight_name(suffix)
          value = @store.vector(name)
          raise ArgumentError.new("#{name} has width #{value.size}; expected #{size}") unless value.size == size
          value
        end

        private def weight_name(suffix : String) : String
          @prefix.empty? ? suffix : "#{@prefix}.#{suffix}"
        end

        private def run_layer(layer : TowerLayerWeights, hidden : Vector, embedding : Vector, cache : TowerCache, trace : TowerTraceCallback?, trace_prefix : String) : Vector
          Profile.measure("tower.layer", category: "tower") do
            run_layer_impl(layer, hidden, embedding, cache, trace, trace_prefix)
          end
        end

        private def run_layer_impl(layer : TowerLayerWeights, hidden : Vector, embedding : Vector, cache : TowerCache, trace : TowerTraceCallback?, trace_prefix : String) : Vector
          trace_value(trace, "#{trace_prefix}.embedding", [1_i32, 1_i32, @config.dim], embedding)
          normalized_embedding = Ops.rms_norm(embedding, layer.enorm, @config.norm_eps, true)
          normalized_hidden = Ops.rms_norm(hidden, layer.hnorm, @config.norm_eps, true)
          trace_value(trace, "#{trace_prefix}.enorm", [1_i32, 1_i32, @config.dim], normalized_embedding)
          trace_value(trace, "#{trace_prefix}.hnorm", [1_i32, 1_i32, @config.dim], normalized_hidden)
          # Array#+ concatenates, matching the reference's cat([norm(e), norm(h)]).
          combined = normalized_embedding + normalized_hidden
          trace_value(trace, "#{trace_prefix}.eh_concat", [1_i32, 1_i32, @config.dim * 2], combined)
          mixed = Ops.linear(combined, layer.eh_proj)
          trace_value(trace, "#{trace_prefix}.eh_proj", [1_i32, 1_i32, @config.dim], mixed)
          normalized_attention_input = Ops.rms_norm(mixed, layer.attention_norm, @config.norm_eps, true)
          trace_value(trace, "#{trace_prefix}.attention_norm", [1_i32, 1_i32, @config.dim], normalized_attention_input)
          attended = attention(layer, normalized_attention_input, cache, trace, "#{trace_prefix}.attention")
          state = Ops.add(mixed, attended)
          trace_value(trace, "#{trace_prefix}.attention_residual", [1_i32, 1_i32, @config.dim], state)
          normalized_feed_input = Ops.rms_norm(state, layer.ffn_norm, @config.norm_eps, true)
          trace_value(trace, "#{trace_prefix}.ffn_norm", [1_i32, 1_i32, @config.dim], normalized_feed_input)
          feed = feed_forward(layer, normalized_feed_input, trace, "#{trace_prefix}.feed_forward")
          residual = Ops.add(state, feed)
          trace_value(trace, "#{trace_prefix}.ffn_residual", [1_i32, 1_i32, @config.dim], residual)
          output = Ops.rms_norm(residual, layer.final_layernorm, @config.norm_eps, true)
          trace_value(trace, "#{trace_prefix}.output", [1_i32, 1_i32, @config.dim], output)
          output
        end

        private def attention(layer : TowerLayerWeights, input : Vector, cache : TowerCache, trace : TowerTraceCallback?, trace_prefix : String) : Vector
          Profile.measure("tower.attention", category: "attention") do
            attention_impl(layer, input, cache, trace, trace_prefix)
          end
        end

        private def attention_impl(layer : TowerLayerWeights, input : Vector, cache : TowerCache, trace : TowerTraceCallback?, trace_prefix : String) : Vector
          q_width = @config.n_head * @config.head_dim
          kv_width = @config.n_local_heads * @config.head_dim
          projected = Ops.linear(input, layer.wqkv)
          q = projected[0, q_width]
          gate = projected[q_width, q_width]
          k = projected[q_width * 2, kv_width]
          v = projected[q_width * 2 + kv_width, kv_width]
          trace_value(trace, "#{trace_prefix}.q_projected", [1_i32, 1_i32, @config.n_head, @config.head_dim], q)
          trace_value(trace, "#{trace_prefix}.gate", [1_i32, 1_i32, @config.n_head, @config.head_dim], gate)
          trace_value(trace, "#{trace_prefix}.k_projected", [1_i32, 1_i32, @config.n_local_heads, @config.head_dim], k)
          trace_value(trace, "#{trace_prefix}.v_projected", [1_i32, 1_i32, @config.n_local_heads, @config.head_dim], v)

          q = normalize_heads(q, @config.n_head, layer.q_norm)
          k = normalize_heads(k, @config.n_local_heads, layer.k_norm)
          trace_value(trace, "#{trace_prefix}.q_norm", [1_i32, @config.n_head, 1_i32, @config.head_dim], q)
          trace_value(trace, "#{trace_prefix}.k_norm", [1_i32, @config.n_local_heads, 1_i32, @config.head_dim], k)
          position = cache.offset
          Ops.rope!(q, @config.n_head, @config.head_dim, @config.rotary_dim, position, @config.rope_base, true)
          Ops.rope!(k, @config.n_local_heads, @config.head_dim, @config.rotary_dim, position, @config.rope_base, true)
          trace_value(trace, "#{trace_prefix}.q_rope", [1_i32, @config.n_head, 1_i32, @config.head_dim], q)
          trace_value(trace, "#{trace_prefix}.k_rope", [1_i32, @config.n_local_heads, 1_i32, @config.head_dim], k)
          cache.append(k, v)
          trace_value(trace, "#{trace_prefix}.key_cache", [1_i32, @config.n_local_heads, cache.offset, @config.head_dim], cache_tensor(cache.keys))
          trace_value(trace, "#{trace_prefix}.value_cache", [1_i32, @config.n_local_heads, cache.offset, @config.head_dim], cache_tensor(cache.values))

          context = Array(Float32).new(q_width, 0.0_f32)
          heads_per_kv = @config.n_head // @config.n_local_heads
          scale = 1.0_f32 / Math.sqrt(@config.head_dim.to_f64).to_f32
          @config.n_head.times do |head|
            query_offset = head * @config.head_dim
            kv_offset = (head // heads_per_kv) * @config.head_dim
            scores = Array(Float32).new(cache.keys.size) do |time|
              score = 0.0_f32
              @config.head_dim.times do |dimension|
                score += q[query_offset + dimension] * cache.keys[time][kv_offset + dimension]
              end
              score * scale
            end
            max_score = scores.max
            weights = scores.map { |score| Math.exp((score - max_score).to_f64).to_f32 }
            denominator = weights.sum(0.0_f32)
            @config.head_dim.times do |dimension|
              value = 0.0_f32
              weights.each_with_index do |weight, time|
                value += (weight / denominator) * cache.values[time][kv_offset + dimension]
              end
              context[query_offset + dimension] = value * Ops.sigmoid(gate[query_offset + dimension])
            end
          end
          trace_value(trace, "#{trace_prefix}.gated_context", [1_i32, 1_i32, q_width], context)
          output = Ops.linear(context, layer.wo)
          trace_value(trace, "#{trace_prefix}.output", [1_i32, 1_i32, @config.dim], output)
          output
        end

        private def normalize_heads(input : Vector, head_count : Int32, weight : Vector) : Vector
          output = Array(Float32).new(input.size, 0.0_f32)
          head_count.times do |head|
            start = head * @config.head_dim
            normalized = Ops.rms_norm(input[start, @config.head_dim], weight, @config.norm_eps, true)
            @config.head_dim.times { |dimension| output[start + dimension] = normalized[dimension] }
          end
          output
        end

        private def feed_forward(layer : TowerLayerWeights, input : Vector, trace : TowerTraceCallback?, trace_prefix : String) : Vector
          Profile.measure("tower.feed_forward", category: "tower") do
            feed_forward_impl(layer, input, trace, trace_prefix)
          end
        end

        private def feed_forward_impl(layer : TowerLayerWeights, input : Vector, trace : TowerTraceCallback?, trace_prefix : String) : Vector
          up = Ops.linear(input, layer.w1)
          gate = Ops.linear(input, layer.w3)
          activated = Ops.silu(up)
          product = Array(Float32).new(activated.size) { |index| activated[index] * gate[index] }
          trace_value(trace, "#{trace_prefix}.w1", [1_i32, 1_i32, @config.intermediate_size], up)
          trace_value(trace, "#{trace_prefix}.w3", [1_i32, 1_i32, @config.intermediate_size], gate)
          trace_value(trace, "#{trace_prefix}.silu", [1_i32, 1_i32, @config.intermediate_size], activated)
          trace_value(trace, "#{trace_prefix}.product", [1_i32, 1_i32, @config.intermediate_size], product)
          output = Ops.linear(product, layer.w2)
          trace_value(trace, "#{trace_prefix}.w2", [1_i32, 1_i32, @config.dim], output)
          output
        end

        private def embedding_sum(frame : Array(Int32), first_codebook : Int32, end_codebook : Int32) : Vector
          total = zeros(@config.dim)
          return total if first_codebook >= end_codebook
          first_codebook.upto(end_codebook - 1) do |codebook|
            row = embedding_row(@codebook_embeddings, frame[codebook + 1], codebook)
            @config.dim.times { |dimension| total[dimension] += row[dimension] }
          end
          total
        end

        private def embedding_row(table : Matrix, code : Int32, codebook : Int32 = 0) : Vector
          table.row(codebook * @config.vocab_size + clip_code(code))
        end

        private def clip_code(code : Int32) : Int32
          code.clamp(0, @config.vocab_size - 1)
        end

        private def semantic?(token : Int32) : Bool
          token >= @config.semantic_begin && token <= @config.semantic_end
        end

        private def validate_frame!(frame : Array(Int32), name : String) : Nil
          expected = @config.num_codebooks + 1
          raise ArgumentError.new("#{name} must have #{expected} entries [main, c0, ..., cN-1]") unless frame.size == expected
        end

        private def validate_vector!(vector : Vector, expected : Int32, name : String) : Nil
          raise ArgumentError.new("#{name} has width #{vector.size}; expected #{expected}") unless vector.size == expected
        end

        private def validate_caches!(caches : Array(TowerCache)) : Nil
          raise ArgumentError.new("expected #{@config.layer_count} layer caches, got #{caches.size}") unless caches.size == @config.layer_count
          offset = caches.first.offset
          raise ArgumentError.new("layer cache offsets differ") unless caches.all? { |cache| cache.offset == offset }
        end

        private def trace_value(trace : TowerTraceCallback?, name : String, shape : Array(Int32), values : Vector) : Nil
          trace.try(&.call(name, shape, values.dup))
        end

        private def cache_tensor(rows : Array(Vector)) : Vector
          heads = @config.n_local_heads
          head_dim = @config.head_dim
          Array(Float32).new(heads * rows.size * head_dim) do |index|
            head = index // (rows.size * head_dim)
            time = (index // head_dim) % rows.size
            dimension = index % head_dim
            rows[time][head * head_dim + dimension]
          end
        end

        private def zeros(size : Int32) : Vector
          Array(Float32).new(size, 0.0_f32)
        end

        private def zero!(values : Vector) : Nil
          values.each_index { |index| values[index] = 0.0_f32 }
        end
      end
    end
  end
end
