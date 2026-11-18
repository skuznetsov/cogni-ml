require "digest/sha256"
require "file/tempfile"
require "random/pcg32"
require "./qwen35_cpu"
require "./qwen35_chat"
require "./qwen35_tokenizer"
require "./qwen35_qlora_block"
require "./qwen35_qlora_recurrent"
require "../optim/adam"

module ML::GGUF
  module Qwen35CPU
    # The ordinary prefill route returns only the last chunk's hidden rows
    # when it owns chunking. Keep chunks here to preserve every prefix row.
    def qlora_prefix(weights : Qwen35Weights, tokens : Array(Int32),
                     pos : Int32, state : State, stop_layer : Int32 = weights.layers.size - 1) : Array(Float32)
      raise ArgumentError.new("invalid QLoRA prefix span") if tokens.empty? || pos < 0 || pos + tokens.size > state.max_seq
      raise ArgumentError.new("invalid QLoRA prefix layer") unless 0 <= stop_layer < weights.layers.size

      result = Array(Float32).new(tokens.size * weights.hparams.n_embd)
      limit = Math.min(256, prefill_chunk_size(false))
      offset = 0
      with_prefill_gc_guard do
        while offset < tokens.size
          length = Math.min(limit, tokens.size - offset)
          rows = prefill_tokens_hidden(weights, tokens[offset, length], pos + offset, state,
            stop_layer: stop_layer)
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
    STACK_MAGIC   = "COGNI-QWEN35-QLORA-2"
    private ENDIAN        = IO::ByteFormat::LittleEndian
    private STACK_COUNTS  = {4, 6}
    # Up to six nonnegative Int32 indices, with one separator each.
    private PLACEMENT_LINE_LIMIT = 6 * 11

    record Example, input : Array(Float32), targets : Array(Int32), first : Int32

    # Pure native kernels for the pairwise first-token hinge. The scale fields
    # are the derivative of the weighted hinge with respect to each margin.
    module PairedObjective
      def self.calculate(sft : Float64, anchor_margin : Float32, mate_margin : Float32,
                         weight : Float32, minimum_margin : Float32) : NamedTuple(loss: Float64,
        pair: Float64, anchor_scale: Float32, mate_scale: Float32)
        unless sft.finite? && anchor_margin.finite? && mate_margin.finite? &&
               weight.finite? && weight > 0_f32 && minimum_margin.finite? && minimum_margin > 0_f32
          raise ArgumentError.new("invalid QLoRA paired objective values")
        end

        gap = minimum_margin - Math.min(anchor_margin, mate_margin)
        pair = gap > 0_f32 ? gap.to_f64 : 0.0
        loss = sft + weight.to_f64 * pair
        anchor_scale = 0_f32
        mate_scale = 0_f32
        if gap > 0_f32
          if anchor_margin < mate_margin
            anchor_scale = -weight
          elsif mate_margin < anchor_margin
            mate_scale = -weight
          else
            anchor_scale = -weight / 2_f32
            mate_scale = anchor_scale
          end
        end

        {loss: loss, pair: pair, anchor_scale: anchor_scale, mate_scale: mate_scale}
      end

      # Margin is logit(correct) - logit(other). The derivative applies the
      # selected two-row head GEMM followed by the frozen output RMSNorm rule.
      def self.two_choice_margin_and_hidden_gradient(hidden : Array(Float32), norm : Array(Float32),
                                                     correct_head : Array(Float32), other_head : Array(Float32),
                                                     eps : Float32, *, backward : Bool = true) : NamedTuple(margin: Float32, gradient: Array(Float32))
        width = hidden.size
        unless width > 0 && norm.size == width && correct_head.size == width &&
               other_head.size == width && eps.finite? && eps > 0_f32
          raise ArgumentError.new("invalid QLoRA paired margin dimensions")
        end

        normalized = Array(Float32).new(width, 0_f32)
        sum_squares = 0_f32
        width.times do |i|
          value = hidden.unsafe_fetch(i)
          sum_squares += value * value
        end
        inv_rms = 1_f32 / Math.sqrt(sum_squares / width + eps)
        width.times do |i|
          normalized[i] = hidden.unsafe_fetch(i) * inv_rms * norm.unsafe_fetch(i)
        end

        head = correct_head + other_head
        logits = MathOps.gemm(normalized, head, 1, 2, width, transpose_b: true)
        margin = logits.unsafe_fetch(0) - logits.unsafe_fetch(1)
        gradient = if backward
                     logit_grad = [1_f32, -1_f32]
                     dx_norm = MathOps.gemm(logit_grad, head, 1, width, 2)
                     dot = 0_f32
                     width.times do |i|
                       dot += dx_norm.unsafe_fetch(i) * norm.unsafe_fetch(i) * hidden.unsafe_fetch(i)
                     end
                     correction = dot * inv_rms * inv_rms / width
                     Array(Float32).new(width) do |i|
                       inv_rms * (dx_norm.unsafe_fetch(i) * norm.unsafe_fetch(i) - correction * hidden.unsafe_fetch(i))
                     end
                   else
                     [] of Float32
                   end
        {margin: margin, gradient: gradient}
      end
    end

    # Only completion rows visit the frozen LM head. Its dequantized view is
    # shared by the forward and transpose GEMMs, with bounded logits scratch.
    # Choice mode dequantizes only its two selected rows and maps external token
    # IDs to the two local logits used by the same loss and backward path.
    class CompletionLoss
      @head : Array(Float32)
      @choices : Tuple(Int32, Int32)?

      def initialize(weight : QuantWeight, @norm : Array(Float32), @eps : Float32,
                     *, choices : Tuple(Int32, Int32)? = nil)
        @width = weight.in_dim
        @choices = choices
        unless @width > 0 && weight.out_dim > 0 && @norm.size == @width && @eps > 0
          raise ArgumentError.new("invalid QLoRA output head")
        end
        if ids = choices
          unless 0 <= ids[0] < weight.out_dim && 0 <= ids[1] < weight.out_dim && ids[0] != ids[1]
            raise ArgumentError.new("invalid QLoRA two-class output head")
          end
          @vocab = 2
          @head = dequantize_row(weight, ids[0]) + dequantize_row(weight, ids[1])
        else
          @vocab = weight.out_dim
          unless @vocab.to_i64 * @width <= Int32::MAX
            raise ArgumentError.new("invalid QLoRA output head")
          end
          @head = Dequant.dequantize(weight.raw, weight.type, @vocab * @width)
        end
      end

      def calculate(hidden : Array(Float32), first : Int32,
                    targets : Array(Int32), *, backward : Bool) : {Float64, Array(Float32)}
        unless hidden.size % @width == 0 && first >= 0 && !targets.empty? &&
               first + targets.size <= hidden.size // @width && valid_targets?(targets)
          raise ArgumentError.new("invalid QLoRA completion targets")
        end

        labels = if ids = @choices
                   [targets[0] == ids[0] ? 0 : 1]
                 else
                   targets
                 end
        grad = backward ? Array(Float32).new(hidden.size, 0_f32) : [] of Float32
        loss = 0.0
        inv_count = 1_f32 / labels.size
        offset = 0
        while offset < labels.size
          rows = Math.min(32, labels.size - offset)
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
            label = labels[offset + i]
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
        {loss / labels.size, grad}
      end

      def choice_margin_and_hidden_gradient(hidden : Array(Float32), first : Int32,
                                            correct_token : Int32, *, backward : Bool = true) : NamedTuple(margin: Float32, gradient: Array(Float32))
        ids = @choices
        raise ArgumentError.new("QLoRA margin needs a two-choice output head") unless ids
        unless hidden.size % @width == 0 && first >= 0 && first < hidden.size // @width
          raise ArgumentError.new("invalid QLoRA paired margin row")
        end

        correct_index = if correct_token == ids[0]
                          0
                        elsif correct_token == ids[1]
                          1
                        else
                          raise ArgumentError.new("QLoRA first target is outside paired choices")
                        end
        other_index = 1 - correct_index
        offset = first * @width
        PairedObjective.two_choice_margin_and_hidden_gradient(
          hidden[offset, @width], @norm,
          @head[correct_index * @width, @width], @head[other_index * @width, @width], @eps,
          backward: backward)
      end

      private def valid_targets?(targets : Array(Int32)) : Bool
        if ids = @choices
          targets.size == 1 && (targets[0] == ids[0] || targets[0] == ids[1])
        else
          targets.all? { |id| 0 <= id < @vocab }
        end
      end

      private def dequantize_row(weight : QuantWeight, row : Int32) : Array(Float32)
        block_elements = weight.type.block_elements
        unless block_elements > 0 && weight.in_dim % block_elements == 0
          raise ArgumentError.new("QLoRA output row is not quantization-block aligned")
        end
        row_bytes = (weight.in_dim // block_elements).to_i64 * weight.type.block_bytes
        offset = row.to_i64 * row_bytes
        unless row_bytes > 0 && offset + row_bytes <= weight.raw.size
          raise ArgumentError.new("QLoRA output row exceeds frozen head bytes")
        end
        bytes = Bytes.new(weight.raw.to_unsafe + offset, row_bytes.to_i32, read_only: true)
        Dequant.dequantize(bytes, weight.type, weight.in_dim)
      end
    end

    class Model
      getter weights : Qwen35Weights
      getter tokenizer : Qwen35Tokenizer
      getter digest : String
      getter layer_indices : Array(Int32)
      getter projections : Array(Projection)

      @tail : Array(Block | RecurrentBlock)
      @prefix_layers : Int32

      def initialize(path : String, seed : UInt64 = 42_u64, *, full_blocks : Int32 = 1)
        unless full_blocks == 1 || STACK_COUNTS.includes?(full_blocks)
          raise ArgumentError.new("QLoRA supports one, four, or six full-attention blocks")
        end
        @weights = Qwen35Weights.from_gguf(path)
        metadata = GGUFFile.new(path, mmap_tensors: false)
        begin
          @tokenizer = Qwen35Tokenizer.from_gguf(metadata, path)
        ensure
          metadata.close
        end
        @digest = File.open(path) { |io| Digest::SHA256.hexdigest(io) }
        hp = @weights.hparams
        geometry = {hp.n_layer, hp.n_embd, hp.n_ff, hp.n_head, hp.n_head_kv, hp.head_dim}
        factors_per_block = case geometry
                            when {24, 1024, 3584, 8, 2, 256}
                              EXPECTED_SIZE
                            when {24, 2048, 6144, 8, 2, 256}
                              if {hp.full_attention_interval, hp.ssm_conv_kernel, hp.ssm_state_size,
                                  hp.ssm_group_count, hp.ssm_time_step_rank, hp.ssm_inner_size} == {4, 4, 128, 16, 16, 2048}
                                319_488
                              end
                            when {32, 2560, 9216, 16, 4, 256}
                              if {hp.full_attention_interval, hp.ssm_conv_kernel, hp.ssm_state_size,
                                  hp.ssm_group_count, hp.ssm_time_step_rank, hp.ssm_inner_size} == {4, 4, 128, 16, 32, 4096}
                                479_232
                              end
                            end
        unless factors_per_block && @weights.layers.last.is_a?(Qwen35FullAttnWeights) &&
               @tokenizer.vocab.size == @weights.output.out_dim
          raise ArgumentError.new("QLoRA requires a supported Qwen3.5-0.8B, Qwen3.5-2B, or Qwen3.5-4B text decoder")
        end
        full = @weights.layers.each_index.select { |index| @weights.layers[index].is_a?(Qwen35FullAttnWeights) }.to_a
        raise ArgumentError.new("GGUF has too few full-attention blocks") if full.size < full_blocks
        @layer_indices = full.last(full_blocks)
        @prefix_layers = @layer_indices.first
        @tail = [] of Block | RecurrentBlock
        @projections = [] of Projection
        (@prefix_layers...@weights.layers.size).each do |index|
          case layer = @weights.layers[index]
          in Qwen35FullAttnWeights
            layer_seed = index == @weights.layers.size - 1 ? seed : seed ^ index.to_u64
            adapted = Block.new(layer, hp, layer_seed)
            @tail << adapted
            @projections.concat(adapted.projections)
          in Qwen35RecurrentWeights
            @tail << RecurrentBlock.new(layer, hp)
          end
        end
        unless factors.sum(&.size) == full_blocks * factors_per_block && @projections.all? { |p| p.rank == RANK && p.alpha == ALPHA }
          raise "unexpected QLoRA projection layout"
        end
      end

      def block : Block
        @tail.last.as(Block)
      end

      def close : Nil
        @weights.close
      end

      def example(messages : Array(Qwen35Chat::Message), *, first_token_only : Bool = false) : Example
        unless messages.size >= 2 && messages.last.role == "assistant" && !messages.last.content.empty?
          raise ArgumentError.new("QLoRA example needs a prompt and nonempty assistant completion")
        end
        prompt = encode_prompt(messages[0...-1])
        completion = @tokenizer.encode(messages.last.content + "<|im_end|>", add_bos_override: false)
        if first_token_only && completion.size < 2
          raise ArgumentError.new("QLoRA first-token example needs a real assistant completion token")
        end
        targets = first_token_only ? [completion.first] : completion
        inputs = first_token_only ? prompt : prompt + completion[0...-1]
        check_length(inputs.size)
        input = with_prefix_state(inputs.size) do |state|
          Qwen35CPU.qlora_prefix(@weights, inputs, 0, state, @prefix_layers)
        end
        Example.new(input, targets, prompt.size - 1)
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
          prefix = Qwen35CPU.qlora_prefix(@weights, tokens, 0, state, @prefix_layers)
          decode_states = Array(DecodeState).new(@layer_indices.size) { DecodeState.new }
          hidden = generation_prefill(prefix, state, decode_states)[-width, width]
          max_new.times do |i|
            token, _ = Qwen35CPU.hidden_top1(@weights, hidden)
            if eos.includes?(token)
              stopped = true
              break
            end
            result << token
            break if i + 1 == max_new
            next_row = Qwen35CPU.qlora_prefix(@weights, [token], tokens.size + i, state, @prefix_layers)
            hidden = generation_tail(next_row, state, decode_states)
          end
        end
        {@tokenizer.decode(result).scrub, result.size, stopped ? "eos" : "limit"}
      end

      def loss_head : CompletionLoss
        CompletionLoss.new(@weights.output, @weights.output_norm, @weights.hparams.rms_eps)
      end

      def factors : Array(Array(Float32))
        @projections.flat_map { |proj| [proj.a, proj.b] }
      end

      def forward(input : Array(Float32), *, retain_tape : Bool) : Array(Float32)
        @tail.each do |layer|
          input = layer.forward(input, retain_tape: retain_tape)
        end
        input
      end

      def backward(grad : Array(Float32)) : Nil
        (@tail.size - 1).downto(0) do |offset|
          case layer = @tail[offset]
          in Block
            grad = layer.backward(grad, need_input: offset > 0)
          in RecurrentBlock
            grad = layer.backward(grad)
          end
        end
      end

      def self.adapter_full_blocks(path : String) : Int32
        File.open(path) do |file|
          _, indices = read_adapter_header(file)
          indices ? indices.size : 1
        end
      end

      # :nodoc:
      def self.read_adapter_header(file : IO) : {String, Array(Int32)?}
        magic = file.gets(ADAPTER_MAGIC.bytesize + 1)
        unless magic == ADAPTER_MAGIC + "\n" || magic == STACK_MAGIC + "\n"
          raise ArgumentError.new("unrecognized QLoRA adapter format")
        end
        digest_line = file.gets(65)
        unless digest_line && digest_line.matches?(/\A[0-9a-f]{64}\n\z/)
          raise ArgumentError.new("invalid QLoRA adapter GGUF digest")
        end
        digest = digest_line.chomp
        return {digest, nil} if magic == ADAPTER_MAGIC + "\n"

        line = file.gets(PLACEMENT_LINE_LIMIT)
        unless line && line.ends_with?('\n')
          raise ArgumentError.new("QLoRA adapter block placement is incomplete")
        end
        placement = line.byte_slice(0, line.bytesize - 1)
        indices = placement.split(',').compact_map(&.to_i?)
        unless STACK_COUNTS.includes?(indices.size) && indices.join(',') == placement &&
               indices.all? { |index| index >= 0 } && indices == indices.sort.uniq
          raise ArgumentError.new("QLoRA adapter block placement is invalid")
        end
        {digest, indices}
      end

      def save(path : String) : Nil
        arrays = factors
        raise "nonfinite QLoRA factors" unless arrays.all? { |array| array.all?(&.finite?) }

        file = File.tempfile(".qlora-", nil, dir: File.dirname(path))
        begin
          file << (@layer_indices.size == 1 ? ADAPTER_MAGIC : STACK_MAGIC) << '\n' << @digest << '\n'
          file << @layer_indices.join(',') << '\n' if @layer_indices.size > 1
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
        targets = factors
        skipped = 0
        values = File.open(path, "rb") do |file|
          digest, indices = self.class.read_adapter_header(file)
          unless digest == @digest
            raise ArgumentError.new("QLoRA adapter GGUF digest mismatch")
          end
          blocks = indices ? indices.size : 1
          if indices && (blocks > @layer_indices.size || indices != @layer_indices.last(blocks))
            raise ArgumentError.new("QLoRA adapter block placement mismatch")
          end
          skipped = (@layer_indices.size - blocks) * block.projections.size * 2
          arrays = targets[skipped..].map do |factor|
            Array(Float32).new(factor.size) { file.read_bytes(Float32, ENDIAN) }
          end
          unless file.read_byte.nil? && arrays.all? { |array| array.all?(&.finite?) }
            raise ArgumentError.new("invalid QLoRA adapter contents")
          end
          arrays
        end
        @projections.first(skipped // 2).each { |projection| projection.b.fill(0.0_f32) }
        targets[skipped..].zip(values).each { |target, source| target.to_unsafe.copy_from(source.to_unsafe, source.size) }
      end

      private def generation_prefill(input : Array(Float32), state : Qwen35CPU::State,
                                     decode_states : Array(DecodeState)) : Array(Float32)
        return generation_tail(input, state, decode_states) if @layer_indices.size == 1

        # Match the frozen prefix's bounded Metal prefill chunks. Each tail
        # layer keeps its own attention or recurrent state between chunks.
        width = @weights.hparams.n_embd
        chunk = Math.min(256, Qwen35CPU.prefill_chunk_size(false)) * width
        offset = 0
        output = [] of Float32
        while offset < input.size
          length = Math.min(chunk, input.size - offset)
          output = generation_tail(input[offset, length], state, decode_states)
          offset += length
        end
        output
      end

      private def generation_tail(input : Array(Float32), state : Qwen35CPU::State,
                                  decode_states : Array(DecodeState)) : Array(Float32)
        attention_index = 0
        @tail.each_with_index do |layer, offset|
          case layer
          in Block
            input = layer.prefill(input, decode_states[attention_index])
            attention_index += 1
          in RecurrentBlock
            index = @prefix_layers + offset
            lw = @weights.layers[index].as(Qwen35RecurrentWeights)
            input = Qwen35CPU.qlora_recurrent(input, lw, state.layers[index], @weights.hparams, state.max_seq)
          end
        end
        input
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
          {% unless flag?(:cpu_only) %}
            Qwen35CPU.release_state_metal!(state)
          {% end %}
        end
      end
    end

    class Trainer
      @head : CompletionLoss
      @paired_head : CompletionLoss?
      @paired_choices : Tuple(Int32, Int32)?
      @params : Array(Autograd::Variable)
      @adam : Optim::Adam

      def initialize(@model : Model, learning_rate : Float32 = 0.0001_f32,
                     *, choices : Tuple(Int32, Int32)? = nil,
                     paired_choices : Tuple(Int32, Int32)? = nil)
        if choices && paired_choices
          raise ArgumentError.new("QLoRA trainer cannot combine choice CE and paired objective modes")
        end
        @paired_choices = paired_choices
        @head = if ids = choices
                  CompletionLoss.new(@model.weights.output, @model.weights.output_norm,
                    @model.weights.hparams.rms_eps, choices: ids)
                else
                  @model.loss_head
                end
        @paired_head = if ids = paired_choices
                         CompletionLoss.new(@model.weights.output, @model.weights.output_norm,
                           @model.weights.hparams.rms_eps, choices: ids)
                       else
                         nil
                       end
        @params = @model.factors.map do |factor|
          Autograd::Variable.from_tensor(Tensor.from_array(factor, Shape.new([factor.size])), requires_grad: true)
        end
        @adam = Optim::Adam.new(@params, lr: learning_rate)
      end

      def loss(example : Example) : Float64
        value, _ = @head.calculate(@model.forward(example.input, retain_tape: false), example.first, example.targets, backward: false)
        value
      end

      def step(example : Example) : Float64
        @model.projections.each(&.zero_grad)
        value, gradient = @head.calculate(@model.forward(example.input, retain_tape: true), example.first, example.targets, backward: true)
        raise "nonfinite QLoRA loss" unless value.finite?
        @model.backward(gradient)
        grads = @model.projections.flat_map { |proj| [proj.grad_a, proj.grad_b] }
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

      def step_paired(anchor : Example, mate : Example, *, weight : Float32 = 0.5_f32,
                      minimum_margin : Float32 = 0.5_f32) : NamedTuple(loss: Float64, sft: Float64,
        pair: Float64, anchor_margin: Float32, mate_margin: Float32)
        choices = @paired_choices
        pair_head = @paired_head
        unless choices && pair_head
          raise ArgumentError.new("paired QLoRA step requires a trainer configured with paired choices")
        end
        unless weight.finite? && weight > 0_f32 && minimum_margin.finite? && minimum_margin > 0_f32
          raise ArgumentError.new("paired QLoRA weight and minimum margin must be positive and finite")
        end

        validate_paired_example!(anchor)
        validate_paired_example!(mate)
        anchor_target = anchor.targets.first
        mate_target = mate.targets.first
        unless (anchor_target == choices[0] || anchor_target == choices[1]) &&
               (mate_target == choices[0] || mate_target == choices[1]) && anchor_target != mate_target
          raise ArgumentError.new("paired QLoRA examples need oppositely labeled first targets")
        end

        @model.projections.each(&.zero_grad)
        mate_hidden = @model.forward(mate.input, retain_tape: false)
        mate_margin = pair_head.choice_margin_and_hidden_gradient(mate_hidden, mate.first, mate_target,
          backward: false).[:margin]

        anchor_hidden = @model.forward(anchor.input, retain_tape: true)
        sft, anchor_gradient = @head.calculate(anchor_hidden, anchor.first, anchor.targets, backward: true)
        raise "nonfinite QLoRA SFT loss" unless sft.finite?
        anchor_margin = pair_head.choice_margin_and_hidden_gradient(anchor_hidden, anchor.first, anchor_target,
          backward: false).[:margin]
        objective = PairedObjective.calculate(sft, anchor_margin, mate_margin, weight, minimum_margin)

        if objective[:anchor_scale] != 0_f32
          anchor_result = pair_head.choice_margin_and_hidden_gradient(anchor_hidden, anchor.first, anchor_target)
          start = anchor.first * @model.weights.hparams.n_embd
          anchor_result[:gradient].size.times do |i|
            anchor_gradient[start + i] += objective[:anchor_scale] * anchor_result[:gradient].unsafe_fetch(i)
          end
        end
        @model.backward(anchor_gradient)

        if objective[:mate_scale] != 0_f32
          mate_hidden = @model.forward(mate.input, retain_tape: true)
          mate_result = pair_head.choice_margin_and_hidden_gradient(mate_hidden, mate.first, mate_target)
          mate_gradient = Array(Float32).new(mate_hidden.size, 0_f32)
          start = mate.first * @model.weights.hparams.n_embd
          mate_result[:gradient].size.times do |i|
            mate_gradient[start + i] = objective[:mate_scale] * mate_result[:gradient].unsafe_fetch(i)
          end
          @model.backward(mate_gradient)
        end

        raise "nonfinite QLoRA paired loss" unless objective[:loss].finite? && objective[:pair].finite?
        grads = @model.projections.flat_map { |proj| [proj.grad_a, proj.grad_b] }
        @params.zip(grads).each do |param, grad|
          raise "nonfinite QLoRA factor gradient" unless grad.all?(&.finite?)
          param.grad = Tensor.from_array(grad, param.shape)
        end
        @adam.step
        @model.factors.zip(@params).each do |factor, param|
          factor.to_unsafe.copy_from(param.data.cpu_data.not_nil!.to_unsafe, factor.size)
        end

        {loss: objective[:loss], sft: sft, pair: objective[:pair],
         anchor_margin: anchor_margin, mate_margin: mate_margin}
      end

      private def validate_paired_example!(example : Example) : Nil
        width = @model.weights.hparams.n_embd
        rows = example.input.size // width
        unless example.input.size % width == 0 && !example.targets.empty? && example.first >= 0 &&
               example.first.to_i64 + example.targets.size.to_i64 <= rows &&
               example.targets.all? { |target| 0 <= target < @model.weights.output.out_dim }
          raise ArgumentError.new("invalid full-token QLoRA paired example")
        end
      end
    end
  end
end
