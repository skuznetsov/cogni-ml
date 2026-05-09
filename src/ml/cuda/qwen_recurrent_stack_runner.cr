require "./qwen_recurrent_layer_runner"

module ML::CUDA
  # Correctness-first scaffold for chaining several recurrent-layer CUDA
  # runners. This intentionally uses a host handoff between layers so it can
  # validate lifecycle/state composition before a later device-resident handoff.
  class QwenRecurrentStackRunner
    getter layer_ids : Array(Int32)
    getter runners : Array(QwenRecurrentLayerRunner)
    getter final_gpu_all : Array(Float32)

    def self.load(gguf : ML::GGUF::GGUFFile,
                  layer_ids : Array(Int32),
                  tokens : Int32,
                  xs : Array(Float32),
                  conv_state_inits : Array(Array(Float32)),
                  ssm_state_inits : Array(Array(Float32)),
                  eps : Float32 = 1.0e-6_f32) : self
      weights = layer_ids.map { |layer| QwenRecurrentLayerRunner::Weights.load(gguf, layer, eps) }
      new(layer_ids, weights, tokens, xs, conv_state_inits, ssm_state_inits)
    end

    def initialize(@layer_ids : Array(Int32),
                   weights : Array(QwenRecurrentLayerRunner::Weights),
                   @tokens : Int32,
                   @xs : Array(Float32),
                   conv_state_inits : Array(Array(Float32)),
                   ssm_state_inits : Array(Array(Float32)))
      raise ArgumentError.new("at least one recurrent layer required") if @layer_ids.empty?
      raise ArgumentError.new("layer/weight count mismatch") unless @layer_ids.size == weights.size
      raise ArgumentError.new("conv_state count mismatch") unless conv_state_inits.size == weights.size
      raise ArgumentError.new("ssm_state count mismatch") unless ssm_state_inits.size == weights.size

      hidden = weights.first.hidden
      raise ArgumentError.new("xs size mismatch") unless @xs.size == @tokens * hidden
      weights.each { |weight| raise ArgumentError.new("mixed hidden sizes are unsupported") unless weight.hidden == hidden }

      @runners = weights.map_with_index do |weight, idx|
        layer_input = idx == 0 ? @xs : Array(Float32).new(@tokens * hidden, 0.0_f32)
        QwenRecurrentLayerRunner.from_weights(weight, @tokens, layer_input, conv_state_inits[idx], ssm_state_inits[idx])
      end
      @final_gpu_all = Array(Float32).new(@tokens * hidden, 0.0_f32)
      @closed = false
    end

    def upload_weights : Nil
      @runners.each(&.upload_weights)
    end

    def run_sequence : Nil
      current = @xs
      @runners.each_with_index do |runner, idx|
        runner.replace_sequence_input(current)
        runner.reset_sequence
        runner.run_sequence
        ML::CUDA.synchronize!("cuCtxSynchronize(stack layer #{idx})")
        runner.read_outputs
        current = runner.final_gpu_all.dup
      end
      @final_gpu_all = current
    end

    def close : Nil
      return if @closed

      @runners.each(&.close)
      @closed = true
    end
  end
end
