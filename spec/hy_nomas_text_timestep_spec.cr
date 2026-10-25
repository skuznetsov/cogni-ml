require "spec"
require "json"
require "digest/sha256"
require "../src/ml/video/hy_nomas_text_refiner"
require "../examples/support/hy_nomas_text_probe"
require "../examples/support/hy_nomas_probe_instrumentation"
require "../examples/support/qwen25_prefix_probe"

describe "Hy-Nomas GPU timestep embedding" do
  if path = ENV["HY_NOMAS_TIMESTEP_FIXTURE"]?
    it "matches the pinned basis at fractional and unseen timesteps in one command chain" do
      digest = ENV["HY_NOMAS_TIMESTEP_FIXTURE_SHA256"]
      digest.should match(/\A[0-9a-f]{64}\z/)
      File.size(path).should be <= 262144
      bytes = File.read(path)
      Digest::SHA256.hexdigest(bytes).should eq(digest)
      fixture = JSON.parse(bytes)
      fixture["schema"].as_s.should eq("hy-nomas-timestep-embedding-reference-v1")
      fixture["schema_version"].as_i.should eq(1)
      fixture["upstream_revision"].as_s.should eq("911cea6458a89c2663513ff7099561ec190dd491")
      fixture["source"]["sha256"].as_s.should eq("5b055e383bbf532ea5aa49ee506031d96bea3e3d5b8da3a104d4bd875c2d0af3")
      fixture["spec"]["dim"].as_i.should eq(256)
      fixture["spec"]["max_period"].as_i.should eq(10000)
      timesteps = fixture["spec"]["timesteps"].as_a.map(&.as_f.to_f32)
      timesteps.should eq([-1.0_f32, 0.0_f32, 0.125_f32, 1.0_f32, 499.5_f32, 500.0_f32, 999.25_f32, 1000.0_f32])
      fixture["frequency_basis"]["dtype"].as_s.should eq("F32")
      fixture["frequency_basis"]["shape"].as_a.map(&.as_i).should eq([128])
      basis = fixture["frequency_basis"]["bits"].as_a.map(&.as_i64.to_u32)
      basis.size.should eq(128)
      # Check the model constant itself as well as the dynamic GPU result.
      table = ML::Video::HyNomasTextRefinerMPSPlan::SOURCE.split("constant uint HY_TEXT_TIME_FREQUENCY_BITS[128] = {")[1].split("};")[0]
      table.scan(/0x([0-9a-f]{8})u/).map { |m| m[1].to_u32(16) }.should eq(basis)
      fixture["expected"]["dtype"].as_s.should eq("F32")
      fixture["expected"]["shape"].as_a.map(&.as_i).should eq([8, 256])
      expected = fixture["expected"]["values"].as_a.map { |row| row.as_a.map(&.as_f.to_f32) }
      expected.size.should eq(8)
      expected.each { |row| row.size.should eq(256) }

      arena = ML::Video::Arena.new
      chain = nil.as(ML::Metal::MPSCommandChain?)
      begin
        pipeline = ML::Metal::ComputePipeline.new(ML::Video::HyNomasTextRefinerMPSPlan::TIME_KERNEL,
          ML::Video::HyNomasTextRefinerMPSPlan::SOURCE)
        timesteps.each_with_index do |timestep, i|
          arena.upload("input.#{i}", [timestep])
          arena.alloc("output.#{i}", 256)
        end
        chain = ML::Metal::MPSCommandChain.new
        allocations = ML::MetalBuffer.hy_text_probe_allocations
        readbacks = ML::MetalBuffer.hy_text_probe_readbacks
        timesteps.each_index do |i|
          chain.encoder.set_pipeline(pipeline)
            .set_buffer(arena["input.#{i}"], 0)
            .set_buffer(arena["output.#{i}"], 1, ML::Metal::BufferAccess::Write)
            .set_value(128_u32, 2).dispatch_1d(128)
        end
        ML::MetalBuffer.hy_text_probe_allocations.should eq(allocations)
        ML::MetalBuffer.hy_text_probe_readbacks.should eq(readbacks)
        chain.wait_count.should eq(0)
        chain.commit_and_wait
        chain.wait_count.should eq(1)
        expected.each_with_index do |reference, i|
          comparison = HyNomasTextProbe.compare_values(arena["output.#{i}"].read(256), reference)
          comparison[:failures].should eq(0), "timestep=#{timesteps[i]}: #{comparison.to_json}"
          comparison[:raw_bit_differences].should eq(0), "timestep=#{timesteps[i]}: #{comparison.to_json}"
        end
      ensure
        Qwen25PrefixProbe.cleanup_after_drain(chain) do
          chain.try(&.release)
          arena.release
        end
      end
    end
  else
    pending "requires a fresh authenticated HY_NOMAS_TIMESTEP_FIXTURE under the external guard" do
    end
  end
end
