require "./scratch"
require "./math"
require "./dspark_weights"
require "./dspark_dense"
require "../../../metal/device"
require "../../../metal/dispatch"
require "../../../core/buffer"

module ML::GGUF
  {% if flag?(:darwin) && !flag?(:cpu_only) %}
    # Encode-only three-layer DSpark transformer. One instance owns one
    # sequence's bounded context K/V rings; the caller submits the encoder and
    # advances this state only with confirmed target taps.
    class Inkling2DSparkTransformer
      private HIDDEN       = 2_304_i32
      private INTERMEDIATE = 9_216_i32
      private Q_HEADS      = 24_i32
      private KV_HEADS     = 8_i32
      private HEAD_DIM     = 128_i32
      private BLOCK        = 8_i32
      private LAYERS       = 3_i32
      private RING_SIZE    = 1_024_i32
      private CONTEXT_MAX  = 1_023_i32
      private EPS          = 1e-6_f32
      private ROPE_BASE    = 1_000_000_f32
      private THREADS      = 256_i32
      private SOURCE       = {{ read_file("#{__DIR__}/../../../metal/kernels/inkling2/inkling2_dspark_transformer.metal") }}

      @weights : Inkling2DSparkWeights
      @context_keys = [] of ML::MetalBuffer
      @context_values = [] of ML::MetalBuffer
      @closed = false

      def initialize(@weights : Inkling2DSparkWeights)
        raise "Metal is unavailable. Use the CPU Inkling path instead." unless ML::Metal::Device.init!

        cache_bytes = RING_SIZE.to_i64 * KV_HEADS * HEAD_DIM * sizeof(Float32)
        begin
          LAYERS.times do
            @context_keys << ML::MetalBuffer.new(cache_bytes, ML::StorageMode::Private)
            @context_values << ML::MetalBuffer.new(cache_bytes, ML::StorageMode::Private)
          end
        rescue ex
          release_context
          raise ex
        end
      end

      # Append confirmed target taps at their absolute positions. Taps are
      # row-major F32 `[rows, 2304]`. The caller tracks the committed cursor
      # and may begin with a bounded suffix of a long prefill. The context ring
      # keeps absolute-row modulo 1024 slots and every layer reads at most the
      # newest 1023 rows.
      def encode_absorb(
        encoder : ML::Metal::ComputeEncoder,
        taps : StaticArray(Inkling2View, 4),
        start : Int32,
        rows : Int32,
        scratch : Inkling2ResidentScratch,
      ) : Nil
        ensure_open!
        raise ArgumentError.new("Inkling 2 DSpark absorb start must be non-negative") if start < 0
        raise ArgumentError.new("Inkling 2 DSpark absorb rows must be between 1 and #{CONTEXT_MAX}") unless rows.in?(1..CONTEXT_MAX)
        end_position = start.to_i64 + rows
        raise ArgumentError.new("Inkling 2 DSpark context position exceeds Int32") if end_position > Int32::MAX
        validate_taps!(taps, rows)

        reuse = scratch.mark
        begin
          LAYERS.times do |layer|
            mixed = scratch.take(checked_elements(rows, HIDDEN, "context mix"))
            key = scratch.take(checked_elements(rows, KV_HEADS * HEAD_DIM, "context key projection"))
            value = scratch.take(checked_elements(rows, KV_HEADS * HEAD_DIM, "context value projection"))

            encode_mix_rows(encoder, taps, layer, mixed, rows)
            prefix = layer_prefix(layer)
            Inkling2DSparkDense.project(encoder, weight("#{prefix}.self_attn.k_proj_ctx.weight"),
              mixed, key, rows, HIDDEN, KV_HEADS * HEAD_DIM)
            Inkling2DSparkDense.project(encoder, weight("#{prefix}.self_attn.v_proj_ctx.weight"),
              mixed, value, rows, HIDDEN, KV_HEADS * HEAD_DIM)
            encode_context_commit(encoder, layer, key, value, start, rows)
            # Each layer's cache commit is the last reader of these temporary
            # projections; command order permits reusing this scratch span.
            scratch.rewind(reuse)
          end
        rescue ex
          scratch.rewind(reuse)
          raise ex
        end
        nil
      end

      # Return the normalized `[8, 2304]` F32 block states. The trained block
      # is the anchor plus seven mask embeddings. All block positions attend
      # mutually to the full block and to the bounded committed context.
      def encode_hidden(
        encoder : ML::Metal::ComputeEncoder,
        anchor_embedding : Inkling2View,
        position : Int32,
        scratch : Inkling2ResidentScratch,
      ) : Inkling2View
        ensure_open!
        raise ArgumentError.new("Inkling 2 DSpark hidden position must be non-negative") if position < 0
        validate_view!(anchor_embedding, HIDDEN, "anchor embedding")

        block_elements = checked_elements(BLOCK, HIDDEN, "draft block")
        query_elements = checked_elements(BLOCK, Q_HEADS * HEAD_DIM, "draft queries")
        key_value_elements = checked_elements(BLOCK, KV_HEADS * HEAD_DIM, "draft keys and values")
        intermediate_elements = checked_elements(BLOCK, INTERMEDIATE, "draft MLP")

        x = scratch.take(block_elements)
        encode_block_init(encoder, anchor_embedding, x)
        residual = x

        LAYERS.times do |layer|
          summed = if layer == 0
                     x
                   else
                     view = scratch.take(block_elements)
                     Inkling2ResidentMath.add(encoder, x, residual, view, block_elements)
                     view
                   end

          prefix = layer_prefix(layer)
          normalized = scratch.take(block_elements)
          encode_rms_norm(encoder, summed, weight("#{prefix}.input_layernorm.weight"), normalized, HIDDEN, BLOCK)

          query_raw = scratch.take(query_elements)
          key_raw = scratch.take(key_value_elements)
          value_raw = scratch.take(key_value_elements)
          attention = "#{prefix}.self_attn"
          Inkling2DSparkDense.project(encoder, weight("#{attention}.q_proj.weight"),
            normalized, query_raw, BLOCK, HIDDEN, Q_HEADS * HEAD_DIM)
          Inkling2DSparkDense.project(encoder, weight("#{attention}.k_proj.weight"),
            normalized, key_raw, BLOCK, HIDDEN, KV_HEADS * HEAD_DIM)
          Inkling2DSparkDense.project(encoder, weight("#{attention}.v_proj.weight"),
            normalized, value_raw, BLOCK, HIDDEN, KV_HEADS * HEAD_DIM)

          query = scratch.take(query_elements)
          encode_block_qk(encoder, query_raw, weight("#{attention}.q_norm.weight"), query,
            Q_HEADS, BLOCK, position)
          key = scratch.take(key_value_elements)
          encode_block_qk(encoder, key_raw, weight("#{attention}.k_norm.weight"), key,
            KV_HEADS, BLOCK, position)
          context = scratch.take(query_elements)
          encode_block_attention(encoder, layer, query, key, value_raw, context, position)

          attention_output = scratch.take(block_elements)
          Inkling2DSparkDense.project(encoder, weight("#{attention}.o_proj.weight"),
            context, attention_output, BLOCK, Q_HEADS * HEAD_DIM, HIDDEN)
          attention_residual = scratch.take(block_elements)
          Inkling2ResidentMath.add(encoder, attention_output, summed, attention_residual, block_elements)

          post_normalized = scratch.take(block_elements)
          encode_rms_norm(encoder, attention_residual, weight("#{prefix}.post_attention_layernorm.weight"),
            post_normalized, HIDDEN, BLOCK)
          gate = scratch.take(intermediate_elements)
          up = scratch.take(intermediate_elements)
          Inkling2DSparkDense.project(encoder, weight("#{prefix}.mlp.gate_proj.weight"),
            post_normalized, gate, BLOCK, HIDDEN, INTERMEDIATE)
          Inkling2DSparkDense.project(encoder, weight("#{prefix}.mlp.up_proj.weight"),
            post_normalized, up, BLOCK, HIDDEN, INTERMEDIATE)
          activated = scratch.take(intermediate_elements)
          encode_swiglu(encoder, gate, up, activated, intermediate_elements)
          next_x = scratch.take(block_elements)
          Inkling2DSparkDense.project(encoder, weight("#{prefix}.mlp.down_proj.weight"),
            activated, next_x, BLOCK, INTERMEDIATE, HIDDEN)

          x = next_x
          residual = attention_residual
        end

        final_sum = scratch.take(block_elements)
        Inkling2ResidentMath.add(encoder, x, residual, final_sum, block_elements)
        output = scratch.take(block_elements)
        encode_rms_norm(encoder, final_sum, weight("norm.weight"), output, HIDDEN, BLOCK)
        output
      end

      # Mix selected rows of the four taps for one draft layer or the Markov
      # head (layer 3). The caller supplies one-row tap views and row zero.
      def encode_mix(
        encoder : ML::Metal::ComputeEncoder,
        taps : StaticArray(Inkling2View, 4),
        row : Int32,
        layer : Int32,
        output : Inkling2View,
      ) : Nil
        ensure_open!
        raise ArgumentError.new("Inkling 2 DSpark mix layer must be between 0 and 3") unless layer.in?(0..3)
        raise ArgumentError.new("Inkling 2 DSpark mix row must be non-negative") if row < 0
        required_rows = row.to_i64 + 1_i64
        required_elements = required_rows * HIDDEN
        raise ArgumentError.new("Inkling 2 DSpark mix row exceeds Int32 elements") if required_elements > Int32::MAX
        taps.each do |tap|
          validate_view!(tap, required_elements.to_i32, "mix tap")
        end
        validate_view!(output, HIDDEN, "mix output")

        selected = StaticArray(Inkling2View, 4).new do |index|
          taps[index].slice(row * HIDDEN)
        end
        encode_mix_rows(encoder, selected, layer, output, 1)
      end

      def close : Nil
        return if @closed
        release_context
        @closed = true
        nil
      end

      private def encode_mix_rows(
        encoder : ML::Metal::ComputeEncoder,
        taps : StaticArray(Inkling2View, 4),
        layer : Int32,
        output : Inkling2View,
        rows : Int32,
      ) : Nil
        mix_norm = if layer < LAYERS
                     weight("#{layer_prefix(layer)}.dflare_norm.weight")
                   else
                     weight("markov_head.dflare_norm.weight")
                   end
        encoder.set_pipeline(pipeline("inkling2_dspark_mix_rms"))
        taps.each_with_index do |tap, index|
          encoder.set_buffer(tap.buffer, index, offset: tap.offset)
        end
        encoder.set_buffer(weight("fuse_weight").buffer, 4, offset: weight("fuse_weight").offset)
        encoder.set_buffer(mix_norm.buffer, 5, offset: mix_norm.offset)
        encoder.set_buffer(output.buffer, 6, ML::Metal::BufferAccess::Write, offset: output.offset)
        encoder.set_value(HIDDEN.to_u32, 7)
        encoder.set_value(rows.to_u32, 8)
        encoder.set_value(layer.to_u32, 9)
        encoder.set_value(EPS, 10)
        encoder.dispatch_threadgroups({rows, 1, 1}, {THREADS, 1, 1})
      end

      private def encode_context_commit(
        encoder : ML::Metal::ComputeEncoder,
        layer : Int32,
        key : Inkling2View,
        value : Inkling2View,
        start : Int32,
        rows : Int32,
      ) : Nil
        prefix = layer_prefix(layer)
        key_norm = weight("#{prefix}.self_attn.k_norm_ctx.weight")
        encoder.set_pipeline(pipeline("inkling2_dspark_context_commit"))
        encoder.set_buffer(key.buffer, 0, offset: key.offset)
        encoder.set_buffer(value.buffer, 1, offset: value.offset)
        encoder.set_buffer(key_norm.buffer, 2, offset: key_norm.offset)
        encoder.set_buffer(@context_keys[layer], 3, ML::Metal::BufferAccess::Write)
        encoder.set_buffer(@context_values[layer], 4, ML::Metal::BufferAccess::Write)
        encoder.set_value(start.to_u32, 5)
        encoder.set_value(rows.to_u32, 6)
        encoder.set_value(EPS, 7)
        encoder.set_value(ROPE_BASE, 8)
        encoder.dispatch_threadgroups({KV_HEADS, rows, 1}, {THREADS, 1, 1})
      end

      private def encode_block_init(
        encoder : ML::Metal::ComputeEncoder,
        anchor : Inkling2View,
        output : Inkling2View,
      ) : Nil
        mask = weight("mask_embed")
        encoder.set_pipeline(pipeline("inkling2_dspark_block_init"))
        encoder.set_buffer(anchor.buffer, 0, offset: anchor.offset)
        encoder.set_buffer(mask.buffer, 1, offset: mask.offset)
        encoder.set_buffer(output.buffer, 2, ML::Metal::BufferAccess::Write, offset: output.offset)
        encoder.dispatch_1d(BLOCK * HIDDEN, THREADS)
      end

      private def encode_rms_norm(
        encoder : ML::Metal::ComputeEncoder,
        input : Inkling2View,
        norm : Inkling2View,
        output : Inkling2View,
        width : Int32,
        rows : Int32,
      ) : Nil
        encoder.set_pipeline(pipeline("inkling2_dspark_rms_norm"))
        encoder.set_buffer(input.buffer, 0, offset: input.offset)
        encoder.set_buffer(norm.buffer, 1, offset: norm.offset)
        encoder.set_buffer(output.buffer, 2, ML::Metal::BufferAccess::Write, offset: output.offset)
        encoder.set_value(width.to_u32, 3)
        encoder.set_value(rows.to_u32, 4)
        encoder.set_value(EPS, 5)
        encoder.dispatch_threadgroups({rows, 1, 1}, {THREADS, 1, 1})
      end

      private def encode_block_qk(
        encoder : ML::Metal::ComputeEncoder,
        input : Inkling2View,
        norm : Inkling2View,
        output : Inkling2View,
        heads : Int32,
        rows : Int32,
        position : Int32,
      ) : Nil
        encoder.set_pipeline(pipeline("inkling2_dspark_block_qk_prepare"))
        encoder.set_buffer(input.buffer, 0, offset: input.offset)
        encoder.set_buffer(norm.buffer, 1, offset: norm.offset)
        encoder.set_buffer(output.buffer, 2, ML::Metal::BufferAccess::Write, offset: output.offset)
        encoder.set_value(heads.to_u32, 3)
        encoder.set_value(rows.to_u32, 4)
        encoder.set_value(position.to_u32, 5)
        encoder.set_value(EPS, 6)
        encoder.set_value(ROPE_BASE, 7)
        encoder.dispatch_threadgroups({heads, rows, 1}, {THREADS, 1, 1})
      end

      private def encode_block_attention(
        encoder : ML::Metal::ComputeEncoder,
        layer : Int32,
        query : Inkling2View,
        key : Inkling2View,
        value : Inkling2View,
        output : Inkling2View,
        position : Int32,
      ) : Nil
        encoder.set_pipeline(pipeline("inkling2_dspark_block_attention"))
        encoder.set_buffer(query.buffer, 0, offset: query.offset)
        encoder.set_buffer(key.buffer, 1, offset: key.offset)
        encoder.set_buffer(value.buffer, 2, offset: value.offset)
        encoder.set_buffer(@context_keys[layer], 3)
        encoder.set_buffer(@context_values[layer], 4)
        encoder.set_buffer(output.buffer, 5, ML::Metal::BufferAccess::Write, offset: output.offset)
        encoder.set_value(position.to_u32, 6)
        encoder.dispatch_threadgroups({Q_HEADS, BLOCK, 1}, {THREADS, 1, 1})
      end

      private def encode_swiglu(
        encoder : ML::Metal::ComputeEncoder,
        gate : Inkling2View,
        up : Inkling2View,
        output : Inkling2View,
        count : Int32,
      ) : Nil
        encoder.set_pipeline(pipeline("inkling2_dspark_swiglu"))
        encoder.set_buffer(gate.buffer, 0, offset: gate.offset)
        encoder.set_buffer(up.buffer, 1, offset: up.offset)
        encoder.set_buffer(output.buffer, 2, ML::Metal::BufferAccess::Write, offset: output.offset)
        encoder.set_value(count.to_u32, 3)
        encoder.dispatch_1d(count, THREADS)
      end

      private def validate_taps!(taps : StaticArray(Inkling2View, 4), rows : Int32) : Nil
        required = checked_elements(rows, HIDDEN, "target taps")
        taps.each_with_index do |tap, index|
          validate_view!(tap, required, "target tap #{index}")
        end
      end

      private def validate_view!(view : Inkling2View, elements : Int32, name : String) : Nil
        bytes = elements.to_i64 * sizeof(Float32)
        unless view.offset >= 0 && view.offset % sizeof(Float32) == 0 &&
               view.offset <= view.buffer.size && bytes <= view.buffer.size - view.offset
          raise ArgumentError.new("Inkling 2 DSpark #{name} view exceeds or misaligns its Metal buffer")
        end
      end

      private def checked_elements(rows : Int32, width : Int32, name : String) : Int32
        elements = rows.to_i64 * width
        unless rows >= 0 && width >= 0 && elements <= Int32::MAX
          raise ArgumentError.new("Inkling 2 DSpark #{name} element count exceeds Int32")
        end
        elements.to_i32
      end

      private def weight(name : String) : Inkling2View
        @weights.view(name)
      end

      private def layer_prefix(layer : Int32) : String
        "layers.#{layer}"
      end

      private def release_context : Nil
        @context_keys.each(&.release)
        @context_values.each(&.release)
        @context_keys.clear
        @context_values.clear
        nil
      end

      private def ensure_open! : Nil
        raise "Inkling 2 DSpark transformer is closed" if @closed
      end

      private def pipeline(name : String) : ML::Metal::ComputePipeline
        ML::Metal::PipelineCache.get_or_compile(name, SOURCE)
      end
    end
  {% end %}
end
