require "./dspark_weights"
require "./dspark_dense"
require "./linear"
require "./math"

{% if flag?(:darwin) && !flag?(:cpu_only) %}
  module ML::GGUF
    # Encode the DSpark Markov head recurrence and its direct candidate top-1
    # projection. State and proposed IDs remain in Metal scratch/output views;
    # each call starts a fresh recurrence with the supplied anchor token.
    class Inkling2DSparkMarkov
      # This first integration slice matches Inkling2DSpark's three-ID buffer.
      private RANK = 256_i32
      private HIDDEN = 2304_i32
      private MAX_CANDIDATES = 3_i32
      private THREADS = 256_i32
      private INPUT_WIDTH = 2_i32 * RANK + 2_i32 * HIDDEN
      private PROJECTION_WIDTH = 3_i32 * RANK
      private SOURCE = {{ read_file("#{__DIR__}/../../../metal/kernels/inkling2/inkling2_dspark_markov.metal") }}

      @rnn_weight : Inkling2View
      @rnn_bias : Inkling2View
      @markov_embedding : Inkling2View
      @hidden_out_weight : Inkling2View
      @hidden_out_bias : Inkling2View

      def initialize(
        @weights : Inkling2DSparkWeights,
        @target : Inkling2Weights,
        @linear : Inkling2ResidentLinear,
      )
        config = @target.config
        raise ArgumentError.new("Inkling 2 DSpark Markov expects a 2304-wide target") unless config.n_embd == HIDDEN
        raise ArgumentError.new("Inkling 2 DSpark Markov target head has an incompatible input width") unless @target.output.in_dim == HIDDEN
        raise ArgumentError.new("Inkling 2 DSpark Markov target head has an incompatible vocabulary width") unless @target.output.out_dim == config.padded_vocab_size
        unless Inkling2ResidentLinear.projection_supported?(@target.output)
          raise ArgumentError.new("Inkling 2 DSpark Markov target head type #{@target.output.type} is unsupported by resident projection")
        end

        @rnn_weight = @weights.view("markov_head.rnn_proj.weight")
        @rnn_bias = @weights.view("markov_head.rnn_proj.bias")
        @markov_embedding = @weights.view("markov_head.markov_w1.weight")
        @hidden_out_weight = @weights.view("markov_head.hidden_out.weight")
        @hidden_out_bias = @weights.view("markov_head.hidden_out.bias")
      end

      # `hidden` is the normalized DSpark block `[anchor, candidate...]` in
      # row-major F32. `last_context` is already the fuse_weight[3]-mixed,
      # dflare-normalized confirmed tap. `output_ids` stores Int32 IDs in a
      # same-sized Inkling2View; `slice` uses the common four-byte element size.
      # No command is submitted or waited here.
      def encode(
        encoder : ML::Metal::ComputeEncoder,
        anchor_id : Int32,
        hidden : Inkling2View,
        last_context : Inkling2View,
        output_ids : Inkling2View,
        count : Int32,
        scratch : Inkling2ResidentScratch,
      ) : Nil
        raise ArgumentError.new("Inkling 2 DSpark Markov count must be in 1..#{MAX_CANDIDATES}") unless count.in?(1..MAX_CANDIDATES)
        vocab = @target.config.vocab_size
        stride = @target.output.out_dim
        raise ArgumentError.new("Inkling 2 DSpark Markov anchor is outside the unpadded vocabulary") unless 0 <= anchor_id < vocab

        validate_view!(hidden, (count + 1).to_i64 * HIDDEN.to_i64, "DSpark Markov hidden block")
        validate_view!(last_context, HIDDEN.to_i64, "DSpark Markov last context")
        validate_view!(output_ids, count.to_i64, "DSpark Markov output IDs")

        recurrent_input = scratch.take(INPUT_WIDTH)
        state = scratch.take(RANK)
        raw = scratch.take(PROJECTION_WIDTH)
        latent = scratch.take(RANK)
        hidden_delta = scratch.take(HIDDEN)
        adjusted_hidden = scratch.take(HIDDEN)
        logits = scratch.take(stride)

        count.times do |step|
          encode_input(encoder, hidden, last_context, state, output_ids, recurrent_input,
            anchor_id, step)

          Inkling2DSparkDense.project(encoder, @rnn_weight, recurrent_input, raw,
            1_i32, INPUT_WIDTH, PROJECTION_WIDTH)

          encode_transition(encoder, raw, @rnn_bias, state, latent, step)

          Inkling2DSparkDense.project(encoder, @hidden_out_weight, latent, hidden_delta,
            1_i32, RANK, HIDDEN)

          encode_adjust_hidden(encoder, hidden, hidden_delta, @hidden_out_bias,
            adjusted_hidden, step)

          @linear.project(encoder, @target.output, adjusted_hidden, logits, 1_i32)

          greedy_mark = scratch.mark
          Inkling2ResidentMath.greedy_ids(encoder, scratch, logits, output_ids.slice(step),
            @target.config.logit_softcap, vocab, stride, 1_i32)
          # The serial encoder finishes reducing these tile partials before
          # the next candidate reuses the same scratch range.
          scratch.rewind(greedy_mark)
        end
      end

      private def encode_input(
        encoder : ML::Metal::ComputeEncoder,
        hidden : Inkling2View,
        last_context : Inkling2View,
        state : Inkling2View,
        output_ids : Inkling2View,
        output : Inkling2View,
        anchor_id : Int32,
        step : Int32,
      ) : Nil
        encoder.set_pipeline(pipeline("inkling2_dspark_markov_input"))
        encoder.set_buffer(hidden.buffer, 0, offset: hidden.offset)
        encoder.set_buffer(last_context.buffer, 1, offset: last_context.offset)
        encoder.set_buffer(state.buffer, 2, offset: state.offset)
        encoder.set_buffer(@markov_embedding.buffer, 3, offset: @markov_embedding.offset)
        encoder.set_buffer(output_ids.buffer, 4, offset: output_ids.offset)
        encoder.set_buffer(output.buffer, 5, ML::Metal::BufferAccess::Write, offset: output.offset)
        encoder.set_value(anchor_id.to_u32, 6)
        encoder.set_value(step.to_u32, 7)
        encoder.set_value(HIDDEN.to_u32, 8)
        encoder.set_value(RANK.to_u32, 9)
        encoder.set_value(@target.config.vocab_size.to_u32, 10)
        encoder.dispatch_threadgroups({(INPUT_WIDTH + THREADS - 1) // THREADS, 1, 1}, {THREADS, 1, 1})
      end

      private def encode_transition(
        encoder : ML::Metal::ComputeEncoder,
        raw : Inkling2View,
        bias : Inkling2View,
        state : Inkling2View,
        latent : Inkling2View,
        step : Int32,
      ) : Nil
        encoder.set_pipeline(pipeline("inkling2_dspark_markov_transition"))
        encoder.set_buffer(raw.buffer, 0, offset: raw.offset)
        encoder.set_buffer(bias.buffer, 1, offset: bias.offset)
        encoder.set_buffer(state.buffer, 2, ML::Metal::BufferAccess::ReadWrite, offset: state.offset)
        encoder.set_buffer(latent.buffer, 3, ML::Metal::BufferAccess::Write, offset: latent.offset)
        encoder.set_value(RANK.to_u32, 4)
        encoder.set_value(step.to_u32, 5)
        encoder.dispatch_threadgroups({(RANK + THREADS - 1) // THREADS, 1, 1}, {THREADS, 1, 1})
      end

      private def encode_adjust_hidden(
        encoder : ML::Metal::ComputeEncoder,
        hidden : Inkling2View,
        delta : Inkling2View,
        bias : Inkling2View,
        output : Inkling2View,
        step : Int32,
      ) : Nil
        encoder.set_pipeline(pipeline("inkling2_dspark_markov_adjust_hidden"))
        encoder.set_buffer(hidden.buffer, 0, offset: hidden.offset)
        encoder.set_buffer(delta.buffer, 1, offset: delta.offset)
        encoder.set_buffer(bias.buffer, 2, offset: bias.offset)
        encoder.set_buffer(output.buffer, 3, ML::Metal::BufferAccess::Write, offset: output.offset)
        encoder.set_value(HIDDEN.to_u32, 4)
        encoder.set_value(step.to_u32, 5)
        encoder.dispatch_threadgroups({(HIDDEN + THREADS - 1) // THREADS, 1, 1}, {THREADS, 1, 1})
      end

      private def pipeline(name : String) : ML::Metal::ComputePipeline
        ML::Metal::PipelineCache.get_or_compile(name, SOURCE)
      end

      private def validate_view!(view : Inkling2View, elements : Int64, label : String) : Nil
        bytes = elements * sizeof(Float32).to_i64
        unless view.offset >= 0 && view.offset <= view.buffer.size && bytes <= view.buffer.size - view.offset
          raise ArgumentError.new("#{label} view exceeds its Metal buffer")
        end
      end
    end
  end
{% end %}
