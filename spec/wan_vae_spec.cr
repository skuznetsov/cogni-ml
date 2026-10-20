require "spec"
require "json"
require "digest/sha256"
require "../src/ml/video/wan_vae"
require "./support/video_allocation_counter"

private def vae_assert_gate(name : String, actual : Array(Float32), expected : JSON::Any) : Nil
  ref = expected.as_a
  actual.size.should eq(ref.size), name
  worst = 0_f64
  violations = 0
  actual.each_with_index do |value, index|
    target = ref[index].as_f.to_f32
    ratio = (value - target).abs.to_f64 / (2e-5 + 2e-5 * target.abs)
    violations += 1 unless value.finite? && target.finite? && ratio <= 1
    worst = Math.max(worst, ratio)
  end
  violations.should eq(0), "#{name}: violations=#{violations}, worst normalized error=#{worst}"
end

private def vae_read_fixture_f32(values : JSON::Any) : Array(Float32)
  values.as_a.map(&.as_f.to_f32)
end

private def vae_sha256(path : String) : String
  digest = Digest::SHA256.new
  File.open(path, "rb") do |io|
    bytes = Bytes.new(1024 * 1024)
    while (size = io.read(bytes)) > 0
      digest.update(bytes[0, size])
    end
  end
  digest.hexfinal
end

describe ML::Video::WanVAE do
  it "plans the pre-projection upsample peak and checks scratch element overflow" do
    cfg = ML::Video::WanVAEConfig.new(1, 16, 24)
    previous_count = cfg.batches * cfg.temporal_factor * cfg.output_channels * cfg.height * cfg.width
    nearest_count = cfg.batches * cfg.temporal_factor * 192 * cfg.height * cfg.width
    previous_count.should eq(147_456)
    nearest_count.should eq(294_912)
    ML::Video::WanVAE.activation_scratch_values(cfg).should eq(nearest_count)
    ML::Video::WanVAE.activation_scratch_values(
      ML::Video::WanVAEConfig.new(2, 16, 24)).should eq(589_824)
    ML::Video::WanVAE.activation_scratch_values(
      ML::Video::WanVAEConfig.new(1, 512, 832)).should eq(327_155_712)
    # Posterior channels can exceed the decoder peak in configurable probes.
    ML::Video::WanVAE.activation_scratch_values(
      ML::Video::WanVAEConfig.new(1, 8, 8, z_dim: 32_768)).should eq(65_536)
    ML::Video::WanVAE.activation_scratch_values(
      ML::Video::WanVAEConfig.new(1, 8, 8, dim: 1, z_dim: 1)).should eq(768)
    expect_raises(ArgumentError) do
      ML::Video::WanVAE.activation_scratch_values(
        ML::Video::WanVAEConfig.new(1, 2048, 2048))
    end
  end

  it "keeps the 22-latent decode geometry and requires a caller memory plan" do
    dim_mult = [1, 2, 4, 4]
    temporal = [false, true, true]
    cfg = ML::Video::WanVAEConfig.new(1, 8, 8, 96, 16, dim_mult, 2, temporal)
    dim_mult[1] = 5
    temporal[1] = false
    cfg.dim_mult.should eq([1, 2, 4, 4])
    cfg.temporal_downsample.should eq([false, true, true])
    cfg.dim_mult[1] = 5
    cfg.temporal_downsample[1] = false
    cfg.dim_mult.should eq([1, 2, 4, 4])
    cfg.temporal_downsample.should eq([false, true, true])
    cfg.decoded_frames(22).should eq(85)
    ML::Video::WanVAE.checkpoint_bindings(cfg).size.should eq(194)
    expect_raises(ArgumentError) { ML::Video::WanVAEConfig.new(1, 7, 8) }
    expect_raises(ArgumentError) do
      ML::Video::WanVAEConfig.new(1, 8, 8, 96, 16, [1, 5, 1, 1], 2, [false, true, true])
    end
    expect_raises(ArgumentError) do
      ML::Video::WanVAEConfig.new(1, 8, 8, 96, 16, [1, 2, 4, 4], 2, [false, false, false])
    end
  end

  it "copies source frame slices into an explicit destination offset" do
    inputs = ML::Video::Arena.new
    begin
      source = inputs.upload("vae.copy_source", [10_f32, 20_f32, 30_f32, 40_f32])
      target = inputs.upload("vae.copy_target", [-1_f32, -1_f32, -1_f32, -1_f32, -1_f32, -1_f32])
      command = ML::Metal::CommandBuffer.new
      e = ML::Metal::ComputeEncoder.new(command)
      ML::Video::WanVAE.prepare!
      ML::Video::WanVAE.copy_frames!(e, source, target, 1, 1,
        4, 1, 1, 1, 6, 3, 2)
      e.end_encoding
      command.commit
      command.wait
      target.read(6).should eq([-1_f32, -1_f32, -1_f32, 20_f32, 30_f32, -1_f32])
    ensure
      inputs.release
    end
  end

  it "matches pinned MPS image, 73-frame motion and 22-latent decode fixtures" do
    fixture_path = ENV["WAN_VAE_FIXTURE"]?
    unless fixture_path
      pending! "Set WAN_VAE_FIXTURE to a pinned MPS fixture"
    end
    fixture = JSON.parse(File.read(fixture_path))
    fixture["revision"].as_s.should eq("1ea34ff48f87168174e12956e200b1d908b1c5ff")
    fixture["source_sha256"].as_s.should eq("d0dfe9eccd0cdbcb37ffde9887579be58f4c1bb6fc2dbb1b72e8916e170df180")
    fixture["device"].as_s.should eq("mps")
    fixture["arithmetic"].as_s.should eq("F32")
    fixture["storage"].as_s.should eq("F32")
    fixture["weight_sha256"].as_s.should eq("2ae212ca06c67f00681d5e155d1a954f5ce81eeedb9f5ffd344829817718991f")
    checkpoint_path = fixture["checkpoint"].as_s
    vae_sha256(checkpoint_path).should eq(fixture["weight_sha256"].as_s)

    c = fixture["config"]
    cfg = ML::Video::WanVAEConfig.new(c["batches"].as_i, c["height"].as_i, c["width"].as_i)
    cfg.decoded_frames(c["decode_latent_frames"].as_i).should eq(c["decoded_frames"].as_i)
    c["motion_latent_frames"].as_i.should eq(19)
    c["decoded_frames"].as_i.should eq(85)
    c["delivered_source_offset"].as_i.should eq(4)
    c["deliver_frames"].as_i.should eq(80)

    # The trace is emitted from the upstream cache itself, not inferred from
    # the native cache planner. It pins first, second, and later chunk states.
    encode_trace = fixture["cache_trace"]["encode"].as_a
    encode_trace.size.should eq(3)
    encode_trace.map { |call| call["chunk_index"].as_i }.should eq([0, 1, 2])
    encode_trace.map { |call| call["input_offset"].as_i }.should eq([0, 1, 5])
    encode_trace.map { |call| call["input_frames"].as_i }.should eq([1, 4, 4])
    encode_trace.map { |call| call["output_frames"].as_i }.should eq([1, 1, 1])
    encode_trace.each do |call|
      call["feat_idx"].as_i.should eq(24)
      call["cache"].as_a.size.should eq(26)
      call["cache"].as_a[24, 2].all?(&.raw.nil?).should be_true
    end
    decode_trace = fixture["cache_trace"]["decode"].as_a
    decode_trace.size.should eq(3)
    decode_trace.map { |call| call["latent_index"].as_i }.should eq([0, 1, 2])
    decode_trace.map { |call| call["output_frames"].as_i }.should eq([1, 4, 4])
    [11, 18].each do |slot|
      spatial_scale = slot == 11 ? 1 : 2
      cache_shape = [1, 384, 2,
                     cfg.latent_height * spatial_scale,
                     cfg.latent_width * spatial_scale]
      decode_trace[0]["cache"].as_a[slot].as_s.should eq("Rep")
      decode_trace[1]["cache"].as_a[slot]["shape"].as_a.map(&.as_i).should eq(cache_shape)
      decode_trace[2]["cache"].as_a[slot]["shape"].as_a.map(&.as_i).should eq(cache_shape)
    end
    decode_trace.each do |call|
      call["feat_idx"].as_i.should eq(32)
      call["cache"].as_a.size.should eq(33)
      call["cache"].as_a[32].raw.should be_nil
    end

    file = ML::Video::CheckpointFile.new(checkpoint_path)
    weights = ML::Video::Arena.new
    workspace_arena = ML::Video::Arena.new
    inputs = ML::Video::Arena.new
    prepared = nil.as(ML::Video::PreparedWeights?)
    execution_chain = nil.as(ML::Metal::MPSCommandChain?)
    command = nil.as(ML::Metal::CommandBuffer?)
    begin
      bindings = ML::Video::WanVAE.checkpoint_bindings(cfg)
      plan = ML::Video::WeightPlan.new([file], bindings,
        8_i64 * 1024 * 1024 * 1024,
        1024_i64 * 1024 * 1024,
        resident_dtype: ML::Video::WeightDType::F32,
        require_all: true)
      prepared = ML::Video::PreparedWeights.new(plan, weights)
      ML::Video::WanVAE.prepare!
      workspace = ML::Video::WanVAE.prepare!(workspace_arena, weights, cfg,
        8_i64 * 1024 * 1024 * 1024,
        1024_i64 * 1024 * 1024,
        c["decode_latent_frames"].as_i)
      workspace.cache_names.size.should eq(56), "24 encoder plus 32 decoder cache slots"

      image_values = vae_read_fixture_f32(fixture["inputs"]["image"])
      motion_values = vae_read_fixture_f32(fixture["inputs"]["motion"])
      latent_values = vae_read_fixture_f32(fixture["inputs"]["latent"])
      image = inputs.upload("vae.image", image_values)
      motion = inputs.upload("vae.motion", motion_values)
      latent = inputs.upload("vae.latent", latent_values)
      image_latent = inputs.alloc("vae.image_latent", cfg.z_dim * cfg.latent_height * cfg.latent_width)
      motion_latent = inputs.alloc("vae.motion_latent", cfg.z_dim * c["motion_latent_frames"].as_i * cfg.latent_height * cfg.latent_width)
      decoded_count = cfg.batches * 3 * c["decoded_frames"].as_i * cfg.height * cfg.width
      delivered_count = cfg.batches * 3 * c["deliver_frames"].as_i * cfg.height * cfg.width
      decoded = inputs.alloc("vae.decoded", decoded_count)
      repeated = inputs.alloc("vae.decoded_repeated", decoded_count)
      delivered = inputs.alloc("vae.delivered", delivered_count)
      copy_source = inputs.upload("vae.copy_source", [10_f32, 20_f32, 30_f32, 40_f32])
      copy_target = inputs.upload("vae.copy_target", [-1_f32, -1_f32, -1_f32, -1_f32, -1_f32, -1_f32])

      before = ML::MetalBuffer.video_spec_allocations
      if ENV["WAN_VAE_EXECUTION_CHAIN"]? == "1"
        active = ML::Metal::MPSCommandChain.new
        execution_chain = active
        e = active.encoder
      else
        active = ML::Metal::CommandBuffer.new
        command = active
        e = ML::Metal::ComputeEncoder.new(active)
      end
      prepared.encode(e)
      ML::Video::WanVAE.copy_frames!(e, copy_source, copy_target, 1, 1,
        4, 1, 1, 1, 6, 3, 2)
      ML::Video::WanVAE.encode_image(e, image, image_latent, weights, workspace, cfg,
        execution_chain: execution_chain)
      if active = execution_chain
        e = active.encoder
      end
      ML::Video::WanVAE.encode_motion(e, motion, motion_latent, weights, workspace, cfg,
        c["motion_frames"].as_i, execution_chain: execution_chain)
      if active = execution_chain
        e = active.encoder
      end
      output_frames = ML::Video::WanVAE.decode(e, latent, decoded, weights, workspace, cfg,
        c["decode_latent_frames"].as_i, execution_chain: execution_chain)
      if active = execution_chain
        e = active.encoder
      end
      output_frames.should eq(85)
      ML::Video::WanVAE.crop_frames!(e, decoded, delivered, cfg, output_frames,
        c["delivered_source_offset"].as_i, c["deliver_frames"].as_i)
      ML::Video::WanVAE.decode(e, latent, repeated, weights, workspace, cfg,
        c["decode_latent_frames"].as_i, execution_chain: execution_chain).should eq(output_frames)
      if active = execution_chain
        e = active.encoder
      end
      ML::MetalBuffer.video_spec_allocations.should eq(before), "VAE encode/decode must allocate no GPU buffers"
      if active = execution_chain
        active.wait_count.should eq(0), "VAE chain must defer its only wait until the terminal boundary"
        active.commit_and_wait
        active.wait_count.should eq(1), "VAE chain must use one terminal wait"
        active.segment_count.should be > 1, "VAE chain must continue between completed subphases"
      else
        e.end_encoding
        command.not_nil!.commit
        command.not_nil!.wait
      end

      actual = {
        "image_latent"  => image_latent.read(cfg.z_dim * cfg.latent_height * cfg.latent_width),
        "motion_latent" => motion_latent.read(cfg.z_dim * c["motion_latent_frames"].as_i * cfg.latent_height * cfg.latent_width),
        "decoded"       => decoded.read(decoded_count),
        "delivered"     => delivered.read(delivered_count),
      }
      if actual_path = ENV["WAN_VAE_ACTUAL"]?
        File.write(actual_path, actual.to_json)
      end
      actual.each do |name, values|
        vae_assert_gate(name, values, fixture["expected"][name])
      end
      repeated.read(decoded_count).map(&.unsafe_as(UInt32)).should eq(
        actual["decoded"].map(&.unsafe_as(UInt32))), "serial decode must reset temporal history"
      copy_target.read(6).should eq([-1_f32, -1_f32, -1_f32, 20_f32, 30_f32, -1_f32])
    ensure
      begin
        if active = execution_chain
          begin
            active.commit_and_wait unless active.completed?
          ensure
            active.release if active.completed?
          end
        end
      ensure
        if execution_chain.nil? || execution_chain.not_nil!.completed?
          prepared.try(&.release)
          inputs.release
          workspace_arena.release
          weights.release
          file.close
        end
      end
    end
  end
end
