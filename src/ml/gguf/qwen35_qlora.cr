require "digest/sha256"
require "file/tempfile"
require "random/pcg32"
require "./qwen35_cpu"
require "./qwen35_chat"
require "./qwen35_tokenizer"
require "./qwen35_qlora_block"
require "../optim/adam"

module ML::GGUF
  module Qwen35CPU
    # The ordinary prefill route returns only the last chunk's hidden rows
    # when it owns chunking. Keep chunks here to preserve every prefix row.
    def qlora_prefix(weights : Qwen35Weights, tokens : Array(Int32),
                     pos : Int32, state : State) : Array(Float32)
      raise ArgumentError.new("invalid QLoRA prefix span") if tokens.empty? || pos < 0 || pos + tokens.size > state.max_seq

      result = Array(Float32).new(tokens.size * weights.hparams.n_embd)
      limit = Math.min(256, prefill_chunk_size(false))
      offset = 0
      with_prefill_gc_guard do
        while offset < tokens.size
          length = Math.min(limit, tokens.size - offset)
          rows = prefill_tokens_hidden(weights, tokens[offset, length], pos + offset, state,
            stop_layer: weights.layers.size - 1)
          result.concat(rows)
          offset += length
        end
      end
      result
    end
  end

  module Qwen35QLoRA
    RANK          =       8
    ALPHA         =  16_f32
    MAX_SEQ       =    4096
    EXPECTED_SIZE = 200_704
    ADAPTER_MAGIC = "COGNI-QWEN35-QLORA-1"
    private ENDIAN        = IO::ByteFormat::LittleEndian

    record Example, input : Array(Float32), targets : Array(Int32), first : Int32

    # Only completion rows visit the frozen LM head. Its dequantized view is
    # shared by the forward and transpose GEMMs, with bounded logits scratch.
    class CompletionLoss
      def initialize(weight : QuantWeight, @norm : Array(Float32), @eps : Float32)
        @width = weight.in_dim
        @vocab = weight.out_dim
        unless @width > 0 && @vocab > 0 && @norm.size == @width && @eps > 0 &&
               @vocab.to_i64 * @width <= Int32::MAX
          raise ArgumentError.new("invalid QLoRA output head")
        end
        @head = Dequant.dequantize(weight.raw, weight.type, @vocab * @width)
      end

      def calculate(hidden : Array(Float32), first : Int32,
                    targets : Array(Int32), *, backward : Bool) : {Float64, Array(Float32)}
        unless hidden.size % @width == 0 && first >= 0 && !targets.empty? &&
               first + targets.size <= hidden.size // @width && targets.all? { |id| 0 <= id < @vocab }
          raise ArgumentError.new("invalid QLoRA completion targets")
        end

        grad = backward ? Array(Float32).new(hidden.size, 0_f32) : [] of Float32
        loss = 0.0
        inv_count = 1_f32 / targets.size
        offset = 0
        while offset < targets.size
          rows = Math.min(32, targets.size - offset)
          start = (first + offset) * @width
          x = hidden[start, rows * @width]
          normalized = Array(Float32).new(x.size, 0_f32)
          inv_rms = Array(Float32).new(rows) do |i|
            row_start = i * @width
            sum = 0_f32
            @width.times do |d|
              value = x.unsafe_fetch(row_start + d)
              sum += value * value
            end
            inv = 1_f32 / Math.sqrt(sum / @width + @eps)
            @width.times { |d| normalized[row_start + d] = x.unsafe_fetch(row_start + d) * inv * @norm.unsafe_fetch(d) }
            inv
          end
          logits = MathOps.gemm(normalized, @head, rows, @vocab, @width, transpose_b: true)
          values = logits.to_unsafe
          rows.times do |i|
            row = values + i * @vocab
            label = targets[offset + i]
            correct = row[label]
            max = row[0]
            @vocab.times { |j| max = Math.max(max, row[j]) }
            sum = 0.0
            @vocab.times do |j|
              p = Math.exp(row[j] - max)
              row[j] = p
              sum += p
            end
            loss += Math.log(sum) + max.to_f64 - correct.to_f64
            if backward
              scale = (inv_count / sum).to_f32
              @vocab.times { |j| row[j] *= scale }
              row[label] -= inv_count
            end
          end

          if backward
            dx_norm = MathOps.gemm(logits, @head, rows, @width, @vocab)
            rows.times do |i|
              row = i * @width
              dot = 0_f32
              @width.times { |d| dot += dx_norm.unsafe_fetch(row + d) * @norm.unsafe_fetch(d) * x.unsafe_fetch(row + d) }
              inv = inv_rms[i]
              correction = dot * inv * inv / @width
              @width.times do |d|
                grad[start + row + d] = inv * (dx_norm.unsafe_fetch(row + d) * @norm.unsafe_fetch(d) -
                                               correction * x.unsafe_fetch(row + d))
              end
            end
          end
          offset += rows
        end
        {loss / targets.size, grad}
      end
    end

    class Model
      getter weights : Qwen35Weights
      getter tokenizer : Qwen35Tokenizer
      getter block : Block
      getter digest : String

      def initialize(path : String, seed : UInt64 = 42_u64)
        @weights = Qwen35Weights.from_gguf(path)
        metadata = GGUFFile.new(path, mmap_tensors: false)
        begin
          @tokenizer = Qwen35Tokenizer.from_gguf(metadata, path)
        ensure
          metadata.close
        end
        @digest = File.open(path) { |io| Digest::SHA256.hexdigest(io) }
        hp = @weights.hparams
        unless {hp.n_layer, hp.n_embd, hp.n_ff, hp.n_head, hp.n_head_kv, hp.head_dim} == {24, 1024, 3584, 8, 2, 256} &&
               @weights.layers.last.is_a?(Qwen35FullAttnWeights) &&
               @tokenizer.vocab.size == @weights.output.out_dim
          raise ArgumentError.new("QLoRA requires the Qwen3.5-0.8B text decoder")
        end
        @block = Block.new(@weights.layers.last.as(Qwen35FullAttnWeights), hp, seed)
        unless factors.sum(&.size) == EXPECTED_SIZE && @block.projections.all? { |p| p.rank == RANK && p.alpha == ALPHA }
          raise "unexpected QLoRA projection layout"
        end
      end

      def close : Nil
        @weights.close
      end

      def example(messages : Array(Qwen35Chat::Message)) : Example
        unless messages.size >= 2 && messages.last.role == "assistant" && !messages.last.content.empty?
          raise ArgumentError.new("QLoRA example needs a prompt and nonempty assistant completion")
        end
        prompt = encode_prompt(messages[0...-1])
        completion = @tokenizer.encode(messages.last.content + "<|im_end|>", add_bos_override: false)
        inputs = prompt + completion[0...-1]
        check_length(inputs.size)
        input = with_prefix_state(inputs.size) do |state|
          Qwen35CPU.qlora_prefix(@weights, inputs, 0, state)
        end
        Example.new(input, completion, prompt.size - 1)
      end

      def generate(messages : Array(Qwen35Chat::Message), max_new : Int32) : {String, Int32, String}
        raise ArgumentError.new("invalid QLoRA generation limit") unless 0 < max_new < MAX_SEQ

        tokens = encode_prompt(messages)
        check_length(tokens.size + max_new)
        eos = {@tokenizer.eos_id, @tokenizer.token_to_id["<|im_end|>"]?,
               @tokenizer.token_to_id["<|endoftext|>"]?}
        width = @weights.hparams.n_embd
        result = Array(Int32).new(max_new)
        stopped = false
        with_prefix_state(tokens.size + max_new) do |state|
          prefix = Qwen35CPU.qlora_prefix(@weights, tokens, 0, state)
          decode_state = DecodeState.new
          hidden = @block.prefill(prefix, decode_state)[-width, width]
          max_new.times do |i|
            token, _ = Qwen35CPU.hidden_top1(@weights, hidden)
            if eos.includes?(token)
              stopped = true
              break
            end
            result << token
            break if i + 1 == max_new
            next_row = Qwen35CPU.qlora_prefix(@weights, [token], tokens.size + i, state)
            hidden = @block.decode(next_row, decode_state)
          end
        end
        {@tokenizer.decode(result).scrub, result.size, stopped ? "eos" : "limit"}
      end

      def loss_head : CompletionLoss
        CompletionLoss.new(@weights.output, @weights.output_norm, @weights.hparams.rms_eps)
      end

      def factors : Array(Array(Float32))
        @block.projections.flat_map { |proj| [proj.a, proj.b] }
      end

      def save(path : String) : Nil
        arrays = factors
        raise "nonfinite QLoRA factors" unless arrays.all? { |array| array.all?(&.finite?) }

        file = File.tempfile(".qlora-", nil, dir: File.dirname(path))
        begin
          file << ADAPTER_MAGIC << '\n' << @digest << '\n'
          arrays.each do |factor|
            factor.each { |value| file.write_bytes(value, ENDIAN) }
          end
          file.close
          File.rename(file.path, path)
        ensure
          file.close unless file.closed?
          File.delete?(file.path)
        end
      end

      def load(path : String) : Nil
        values = File.open(path, "rb") do |file|
          unless file.gets(ADAPTER_MAGIC.bytesize + 1, chomp: true) == ADAPTER_MAGIC &&
                 file.gets(65, chomp: true) == @digest
            raise ArgumentError.new("QLoRA adapter format or GGUF digest mismatch")
          end
          arrays = factors.map do |factor|
            Array(Float32).new(factor.size) { file.read_bytes(Float32, ENDIAN) }
          end
          unless file.read_byte.nil? && arrays.all? { |array| array.all?(&.finite?) }
            raise ArgumentError.new("invalid QLoRA adapter contents")
          end
          arrays
        end
        factors.zip(values).each { |target, source| target.to_unsafe.copy_from(source.to_unsafe, source.size) }
      end

      private def encode_prompt(messages : Array(Qwen35Chat::Message)) : Array(Int32)
        @tokenizer.encode(Qwen35Chat.render(messages, enable_thinking: false), add_bos_override: false)
      end

      private def check_length(length : Int32) : Nil
        if length < 1 || length > Math.min(MAX_SEQ, @weights.hparams.context_length)
          raise ArgumentError.new("QLoRA sequence has #{length} tokens outside supported context")
        end
      end

      private def with_prefix_state(length : Int32, &)
        state = Qwen35CPU::State.new(@weights.hparams, length)
        begin
          Qwen35CPU.prepare_state_metal!(state, @weights.hparams, admit_adaptive_resident_kv: false)
          yield state
        ensure
          Qwen35CPU.release_state_metal!(state)
        end
      end
    end

    class Trainer
      @head : CompletionLoss
      @params : Array(Autograd::Variable)
      @adam : Optim::Adam

      def initialize(@model : Model, learning_rate : Float32 = 0.0001_f32)
        @head = @model.loss_head
        @params = @model.factors.map do |factor|
          Autograd::Variable.from_tensor(Tensor.from_array(factor, Shape.new([factor.size])), requires_grad: true)
        end
        @adam = Optim::Adam.new(@params, lr: learning_rate)
      end

      def loss(example : Example) : Float64
        value, _ = @head.calculate(@model.block.forward(example.input), example.first, example.targets, backward: false)
        value
      end

      def step(example : Example) : Float64
        block = @model.block
        block.projections.each(&.zero_grad)
        value, gradient = @head.calculate(block.forward(example.input), example.first, example.targets, backward: true)
        raise "nonfinite QLoRA loss" unless value.finite?
        block.backward(gradient)
        grads = block.projections.flat_map { |proj| [proj.grad_a, proj.grad_b] }
        @params.zip(grads).each do |param, grad|
          raise "nonfinite QLoRA factor gradient" unless grad.all?(&.finite?)
          param.grad = Tensor.from_array(grad, param.shape)
        end
        @adam.step
        @model.factors.zip(@params).each do |factor, param|
          factor.to_unsafe.copy_from(param.data.cpu_data.not_nil!.to_unsafe, factor.size)
        end
        value
      end
    end
  end
end
