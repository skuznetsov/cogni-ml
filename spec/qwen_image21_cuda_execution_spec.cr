require "spec"

{% if flag?(:linux) %}
  require "../src/ml/cuda/qwen_image21_block_runner"

  private def qi21_cuda_f32_weight(out_dim : Int32, in_dim : Int32, phase : Int32)
    raw = Bytes.new(out_dim * in_dim * sizeof(Float32))
    values = raw.to_unsafe.as(Float32*)
    (out_dim * in_dim).times do |index|
      values[index] = (((index * 17 + phase * 13) % 29) - 14).to_f32 / 37.0_f32
    end
    ML::GGUF::QuantWeight.new(raw, ML::GGUF::TensorType::F32, out_dim, in_dim)
  end

  private def qi21_cuda_weights(config : ML::GGUF::QwenImage21BlockConfig, phase : Int32)
    dim = config.hidden_dim
    ML::GGUF::QwenImage21BlockWeights.new(
      qi21_cuda_f32_weight(dim, dim, phase),
      qi21_cuda_f32_weight(dim, dim, phase + 1),
      qi21_cuda_f32_weight(dim, dim, phase + 2),
      qi21_cuda_f32_weight(dim, dim, phase + 3),
      [1.0_f32, 0.9_f32, 1.1_f32, 0.8_f32, 1.2_f32, 0.95_f32],
      [0.85_f32, 1.05_f32, 0.9_f32, 1.15_f32, 0.8_f32, 1.1_f32],
      qi21_cuda_f32_weight(2 * config.intermediate_dim, dim, phase + 4),
      qi21_cuda_f32_weight(dim, config.intermediate_dim, phase + 5),
    )
  end

  private def qi21_cuda_hidden(seed : Int32)
    Array(Float32).new(5 * 6) { |index| (((index * 11 + seed * 7) % 31) - 15).to_f32 / 23.0_f32 }
  end

  describe ML::CUDA::QwenImage21BlockRunner do
    it "chains two resident blocks in a nondefault CUDA graph stream and overwrites stale scratch" do
      pending!("set COGNI_CUDA_IMAGE_TEST=1 on a CUDA host") unless ENV["COGNI_CUDA_IMAGE_TEST"]? == "1"

      context = ML::CUDA::Context.create
      runner_a : ML::CUDA::QwenImage21BlockRunner? = nil
      runner_b : ML::CUDA::QwenImage21BlockRunner? = nil
      stream : ML::CUDA::CUDAStream? = nil
      graph : ML::CUDA::CUDAGraph? = nil
      graph_exec : ML::CUDA::CUDAGraphExec? = nil
      begin
        config = ML::GGUF::QwenImage21BlockConfig.new(
          hidden_dim: 6,
          heads: 1,
          head_dim: 6,
          intermediate_dim: 5,
          axes_dims: StaticArray[2, 2, 2],
        )
        tokens = 5
        positions = [
          StaticArray[0, 0, 0],
          StaticArray[1, 0, 1],
          StaticArray[2, 1, 0],
          StaticArray[2, 3, 0],
          StaticArray[3, 3, 1],
        ]
        image_ids = [-1, -1, 0, 1, 1]
        # Keep token 4 valid so token 3 must include a future key sharing its
        # image id, while token 1 remains a masked causal key.
        key_valid = [true, false, true, true, true]
        modulation_a = Array(Float32).new(tokens * 4 * 6) { |index| (((index * 7) % 31) - 15).to_f32 / 101.0_f32 }
        modulation_b = Array(Float32).new(tokens * 4 * 6) { |index| (((index * 11 + 3) % 37) - 18).to_f32 / 97.0_f32 }
        weights_a = qi21_cuda_weights(config, 1)
        weights_b = qi21_cuda_weights(config, 9)

        expect_raises(ArgumentError, "attention row has no valid keys") do
          ML::CUDA::QwenImage21BlockRunner.new(
            weights_a, config, tokens, modulation_a, positions, image_ids,
            Array(Bool).new(tokens, false)
          )
        end

        runner_a = ML::CUDA::QwenImage21BlockRunner.new(
          weights_a, config, tokens, modulation_a, positions, image_ids, key_valid
        )
        runner_b = ML::CUDA::QwenImage21BlockRunner.new(
          weights_b, config, tokens, modulation_b, positions, image_ids, key_valid
        )
        a = runner_a.not_nil!
        b = runner_b.not_nil!

        expect_raises(ArgumentError, "not been uploaded or selected") { b.enqueue }
        expect_raises(ArgumentError, "must not alias") { b.use_device_input(b.output_device_ptr) }

        input_first = qi21_cuda_hidden(1)
        input_changed = qi21_cuda_hidden(4)
        a.upload_hidden(input_first)
        b.use_device_input(a.output_device_ptr)

        stream = ML::CUDA::CUDAStream.new
        stream_value = stream.not_nil!
        ML::CUDA.with_stream(stream_value) do
          stream_value.begin_capture
          a.enqueue
          b.enqueue
          graph = stream_value.end_capture
        end
        graph_value = graph.not_nil!
        graph_exec = graph_value.instantiate
        graph_exec_value = graph_exec.not_nil!

        # Replay the captured chain for the first input; one terminal stream
        # synchronization precedes diagnostic readback.
        graph_exec_value.launch(stream_value)
        stream_value.synchronize
        first_output = b.read_checkpoint("output")
        expected_first_a = ML::GGUF::QwenImage21BlockCPU.forward(
          input_first, tokens, modulation_a, positions, image_ids, weights_a, config, key_valid
        )
        expected_first = ML::GGUF::QwenImage21BlockCPU.forward(
          expected_first_a, tokens, modulation_b, positions, image_ids, weights_b, config, key_valid
        )
        first_output.zip(expected_first).each do |actual, expected|
          actual.finite?.should be_true
          actual.should be_close(expected, 2e-5_f32)
        end

        # Reuse the same device scratch with changed input. First execute the
        # ordinary enqueue chain, then replay the graph with the same input.
        a.upload_hidden(input_changed)
        ML::CUDA.with_stream(stream_value) do
          a.enqueue
          b.enqueue
        end
        stream_value.synchronize
        uncaptured_output = b.read_checkpoint("output")

        a.upload_hidden(input_changed)
        graph_exec_value.launch(stream_value)
        stream_value.synchronize
        graph_output = b.read_checkpoint("output")
        expected_changed_a = ML::GGUF::QwenImage21BlockCPU.forward(
          input_changed, tokens, modulation_a, positions, image_ids, weights_a, config, key_valid
        )
        expected_changed = ML::GGUF::QwenImage21BlockCPU.forward(
          expected_changed_a, tokens, modulation_b, positions, image_ids, weights_b, config, key_valid
        )

        uncaptured_output.zip(graph_output).each do |actual, expected|
          actual.should be_close(expected, 2e-5_f32)
        end
        graph_output.zip(expected_changed).each do |actual, expected|
          actual.finite?.should be_true
          actual.should be_close(expected, 2e-5_f32)
        end
        graph_output.zip(first_output).any? { |after, before| (after - before).abs > 1e-5_f32 }.should be_true
      ensure
        graph_exec.try(&.close)
        graph.try(&.close)
        stream.try(&.close)
        runner_b.try(&.close)
        runner_a.try(&.close)
        context.close
      end
    end
  end
{% end %}
