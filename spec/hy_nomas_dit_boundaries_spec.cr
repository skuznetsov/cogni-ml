require "spec"
require "./support/video_allocation_counter"
require "../src/ml/video/hy_nomas_dit_boundaries"

describe "Hy-Nomas first-segment input and output boundaries" do
  it "preserves pointwise Conv3d layout across channel and row boundaries" do
    weights, inputs = ML::Video::Arena.new, ML::Video::Arena.new
    patch = nil.as(ML::Video::HyNomasPatchMPSPlan?)
    chain = nil.as(ML::Metal::MPSCommandChain?)
    begin
      tokens = 3
      values = Array(Float32).new(32 * tokens) { |i| i.to_f32 / 8 }
      inputs.upload("latent", values)
      weights.upload("img_in.proj.weight", Array(Float32).new(2048 * 32) { |i| i % 32 == (i // 32) % 32 ? 1.0_f32 : 0.0_f32 })
      weights.upload("img_in.proj.bias", Array(Float32).new(2048) { |i| i.to_f32 / 4 })
      patch = ML::Video::HyNomasPatchMPSPlan.new(inputs["latent"], tokens, weights)
      chain = ML::Metal::MPSCommandChain.new
      before = ML::MetalBuffer.video_spec_allocations
      patch.encode(chain)
      ML::MetalBuffer.video_spec_allocations.should eq(before)
      chain.wait_count.should eq(0)
      expect_raises(ArgumentError, /Drain/) { patch.release }
      chain.commit_and_wait
      chain.wait_count.should eq(1)
      output = patch.output.read(tokens * 2048)
      tokens.times do |token|
        2048.times do |channel|
          output[token * 2048 + channel].should eq(values[(channel % 32) * tokens + token] + channel.to_f32 / 4)
        end
      end
    ensure
      chain.try(&.commit_and_wait)
      patch.try(&.release)
      chain.try(&.release)
      inputs.release
      weights.release
    end
  end

  it "runs normalization, modulation, final projection and unpatchify on one chain" do
    weights, inputs = ML::Video::Arena.new, ML::Video::Arena.new
    final = nil.as(ML::Video::HyNomasFinalMPSPlan?)
    chain = nil.as(ML::Metal::MPSCommandChain?)
    begin
      tokens = 3
      inputs.upload("hidden", Array(Float32).new(tokens * 2048) { |i| (i + i // 2048).even? ? -1.0_f32 : 1.0_f32 })
      inputs.upload("vec", Array(Float32).new(2048, 0.0_f32))
      weights.upload("final_layer.adaLN_modulation.1.weight", Array(Float32).new(4096 * 2048, 0.0_f32))
      weights.upload("final_layer.adaLN_modulation.1.bias", Array(Float32).new(4096) { |i| i < 2048 ? 0.25_f32 : 0.5_f32 })
      weights.upload("final_layer.linear.weight", Array(Float32).new(32 * 2048) { |i| i % 2048 == (i // 2048) * 63 ? 1.0_f32 : 0.0_f32 })
      weights.upload("final_layer.linear.bias", Array(Float32).new(32) { |i| i.to_f32 / 4 })
      final = ML::Video::HyNomasFinalMPSPlan.new(inputs["hidden"], inputs["vec"], tokens, weights)
      chain = ML::Metal::MPSCommandChain.new
      before = ML::MetalBuffer.video_spec_allocations
      final.encode(chain)
      ML::MetalBuffer.video_spec_allocations.should eq(before)
      chain.wait_count.should eq(0)
      expect_raises(ArgumentError, /Drain/) { final.release }
      chain.commit_and_wait
      chain.wait_count.should eq(1)
      output = final.output.read(32 * tokens)
      scale = 1.0_f32 / Math.sqrt(1.0_f32 + 1e-6_f32)
      32.times do |channel|
        tokens.times do |token|
          sign = (channel * 63 + token).even? ? -1.0_f32 : 1.0_f32
          expected = sign * scale * 1.5_f32 + 0.25_f32 + channel.to_f32 / 4
          output[channel * tokens + token].should be_close(expected, 2e-6_f32)
        end
      end
      original = weights["final_layer.linear.bias"]
      replacement = inputs.upload("replacement", Array(Float32).new(32, 0.0_f32))
      weights.buffers["final_layer.linear.bias"] = replacement
      second = ML::Metal::MPSCommandChain.new
      begin
        expect_raises(ArgumentError, /weight binding changed/) { final.encode(second) }
        second.wait_count.should eq(0)
      ensure
        weights.buffers["final_layer.linear.bias"] = original
        second.commit_and_wait
        second.release
      end
    ensure
      chain.try(&.commit_and_wait)
      final.try(&.release)
      chain.try(&.release)
      inputs.release
      weights.release
    end
  end
end
