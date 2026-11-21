require "spec"
require "../src/ml/gguf/inkling2_engine"

{% if flag?(:darwin) && !flag?(:cpu_only) %}
  describe ML::GGUF::Inkling2MetalMatmul do
    it "uses correction bias only for selection and keeps tied expert IDs ordered" do
      pending! "Metal unavailable" unless ML::Metal::Device.init!

      logits = [Math.log(3).to_f32, 0_f32, -Math.log(3).to_f32, 0_f32]
      bias = [0_f32, 0_f32, 0.6_f32, 0_f32]
      picked = 3
      scale = 1.5_f32
      logits_buffer = ML::MetalBuffer.from_array(logits)
      bias_buffer = ML::MetalBuffer.from_array(bias)
      ids_buffer = ML::MetalBuffer.new(picked.to_i64 * sizeof(UInt32))
      weights_buffer = ML::MetalBuffer.new(picked.to_i64 * sizeof(Float32))
      begin
        pipeline = ML::Metal::PipelineCache.get_or_compile("inkling2_moe_route", ML::GGUF::Inkling2MetalMatmul::MOE_SOURCE)
        {true, false}.each do |use_bias|
          command = ML::Metal::CommandBuffer.new
          encoder = ML::Metal::ComputeEncoder.new(command)
          begin
            encoder.set_pipeline(pipeline)
            encoder.set_buffer(logits_buffer, 0)
            encoder.set_buffer(bias_buffer, 1)
            encoder.set_buffer(ids_buffer, 2, ML::Metal::BufferAccess::Write)
            encoder.set_buffer(weights_buffer, 3, ML::Metal::BufferAccess::Write)
            encoder.set_value(logits.size.to_u32, 4)
            encoder.set_value(picked.to_u32, 5)
            encoder.set_value(scale, 6)
            encoder.set_value(use_bias ? 1_u32 : 0_u32, 7)
            encoder.set_threadgroup_memory(2 * logits.size * sizeof(Float32), 0)
            encoder.dispatch_threadgroups({1, 1, 1}, {32, 1, 1})
          ensure
            encoder.end_encoding
          end
          command.commit_and_wait

          expected = ML::GGUF::Inkling2Math.route(logits, use_bias ? bias : nil, picked, scale)
          ids = Slice.new(ids_buffer.contents.as(UInt32*), picked).to_a
          ids.should eq(use_bias ? [2_u32, 0_u32, 1_u32] : [0_u32, 1_u32, 3_u32])
          weights_buffer.read(picked).zip(expected).each do |weight, (_, reference)|
            weight.should be_close(reference, 1e-6_f32)
          end
        end
      ensure
        logits_buffer.release
        bias_buffer.release
        ids_buffer.release
        weights_buffer.release
      end
    end
  end
{% end %}
