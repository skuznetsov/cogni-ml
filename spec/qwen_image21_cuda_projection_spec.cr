require "spec"

{% if flag?(:linux) %}
  require "../src/ml/cuda/qwen_image21_projection"

  private def qi21_projection_weight(type, out_dim, in_dim)
    bytes = Bytes.new(out_dim * (in_dim // type.block_elements) * type.block_bytes)
    ML::GGUF::QuantWeight.new(bytes, type, out_dim, in_dim)
  end

  describe ML::CUDA::QwenImage21Projection do
    it "rejects unsupported formats, partial blocks and truncated weights before CUDA allocation" do
      expect_raises(ArgumentError, "unsupported") do
        ML::CUDA::QwenImage21Projection.validate_weight!(qi21_projection_weight(ML::GGUF::TensorType::Q4_K, 4, 256))
      end
      expect_raises(ArgumentError, "quant-block aligned") do
        ML::CUDA::QwenImage21Projection.validate_weight!(qi21_projection_weight(ML::GGUF::TensorType::Q8_0, 4, 33))
      end
      expect_raises(ArgumentError, "byte count") do
        ML::CUDA::QwenImage21Projection.validate_weight!(
          ML::GGUF::QuantWeight.new(Bytes.new(3), ML::GGUF::TensorType::F32, 1, 1))
      end
    end

    it "matches independent decoded weights for every admitted format and three device rows" do
      pending!("set COGNI_CUDA_IMAGE_TEST=1 on a CUDA host") unless ENV["COGNI_CUDA_IMAGE_TEST"]? == "1"
      context = ML::CUDA::Context.create
      begin
        types = [ML::GGUF::TensorType::F32, ML::GGUF::TensorType::BF16,
                 ML::GGUF::TensorType::Q8_0, ML::GGUF::TensorType::Q5_K, ML::GGUF::TensorType::Q6_K]
        types.each do |type|
          in_dim = type.f32? || type.bf16? ? 7 : 256
          out_dim = 5
          tokens = 3
          weight = qi21_projection_weight(type, out_dim, in_dim)
          raw = weight.raw
          if type.f32? || type.bf16?
            (out_dim * in_dim).times do |i|
              value = (((i * 17 + 3) % 37) - 18).to_f32 / 127
              if type.f32?
                raw.to_unsafe.as(Float32*)[i] = value
              else
                raw.to_unsafe.as(UInt16*)[i] = (value.unsafe_as(UInt32) >> 16).to_u16
              end
            end
          else
            raw.size.times { |i| raw[i] = ((i * 17 + 3) % 251).to_u8 }
            # Fixed small, finite scales keep the test sensitive to decoding
            # and row addressing without magnifying rounding-order differences.
            (raw.size // type.block_bytes).times do |block|
              base = block * type.block_bytes
              scale_offset = type.q6_k? ? 208 : 0
              raw[base + scale_offset] = 0
              raw[base + scale_offset + 1] = 0x18 # F16 2^-9
              if type.q5_k?
                raw[base + 2] = 0
                raw[base + 3] = 0x14 # F16 2^-10 minimum scale
              end
            end
          end
          decoded = ML::GGUF::Dequant.dequantize(raw, type, out_dim * in_dim)
          decoded_raw = Bytes.new(decoded.size * 4)
          decoded_raw.to_unsafe.copy_from(decoded.to_unsafe.as(UInt8*), decoded_raw.size)
          reference_weight = ML::GGUF::QuantWeight.new(decoded_raw, ML::GGUF::TensorType::F32, out_dim, in_dim)
          input = Array(Float32).new(tokens * in_dim) { |i| (((i * 7 + 1) % 29) - 14).to_f32 / 71 }
          expected = ML::GGUF::F32Backend.new.matmul(input, tokens, reference_weight, Array(Float32).new(out_dim, 0.0_f32))
          input_device = ML::CUDA::DeviceBuffer.new((input.size * 4).to_u64)
          output_device = ML::CUDA::DeviceBuffer.new((tokens * out_dim * 4).to_u64)
          projection = ML::CUDA::QwenImage21Projection.new(weight)
          begin
            ML::CUDA.copy_htod!(input_device.ptr, input.to_unsafe.as(Void*), (input.size * 4).to_u64, "projection test input")
            projection.enqueue(input_device.ptr, output_device.ptr, tokens)
            ML::CUDA.synchronize!("projection test terminal completion")
            actual = Array(Float32).new(tokens * out_dim, 0.0_f32)
            ML::CUDA.copy_dtoh!(actual.to_unsafe.as(Void*), output_device.ptr, (actual.size * 4).to_u64, "projection test output")
            actual.zip(expected).each do |value, reference|
              value.finite?.should be_true
              value.should be_close(reference, 2e-5_f32)
            end
            expect_raises(ArgumentError, "must not overlap") do
              projection.enqueue(input_device.ptr, input_device.ptr, tokens)
            end

            # Exercise argument snapshots for Q8's per-token launches as well
            # as the packed Q5/Q6 batch kernels on a nondefault stream.
            poison = Array(Float32).new(actual.size, Float32::NAN)
            ML::CUDA.copy_htod!(output_device.ptr, poison.to_unsafe.as(Void*), (poison.size * 4).to_u64, "projection graph sentinel")
            stream = ML::CUDA::CUDAStream.new(1_u32)
            graph = nil.as(ML::CUDA::CUDAGraph?)
            execution = nil.as(ML::CUDA::CUDAGraphExec?)
            begin
              ML::CUDA.with_stream(stream) do
                stream.begin_capture
                projection.enqueue(input_device.ptr, output_device.ptr, tokens)
                graph = stream.end_capture
              end
              execution = graph.not_nil!.instantiate
              execution.not_nil!.launch(stream)
              stream.synchronize
              replay = Array(Float32).new(actual.size, 0.0_f32)
              ML::CUDA.copy_dtoh!(replay.to_unsafe.as(Void*), output_device.ptr, (replay.size * 4).to_u64, "projection graph output")
              replay.should eq(actual)
            ensure
              execution.try(&.close)
              graph.try(&.close)
              stream.close
            end
          ensure
            projection.close
            output_device.close
            input_device.close
          end
        end
      ensure
        context.close
      end
    end
  end
{% end %}
