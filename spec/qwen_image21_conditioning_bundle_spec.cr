require "./spec_helper"
require "../src/ml/gguf/qwen_image21_conditioning_bundle"

private def qwen_image21_test_conditioning : {String, Bytes}
  states = Array(Float32).new(2 * 4096, 0.0_f32)
  states[0] = 0.25_f32
  latents = Array(Float32).new(4 * 64, 0.0_f32)
  latents[0] = 1.0_f32
  latents[63] = -2.0_f32
  latents[64] = 3.0_f32

  io = IO::Memory.new
  states.each { |value| io.write_bytes(value, IO::ByteFormat::LittleEndian) }
  io.write(Bytes[1, 1])
  io.write(Bytes[0, 0])
  latents.each { |value| io.write_bytes(value, IO::ByteFormat::LittleEndian) }
  states_bytes = states.size * 4
  noise_offset = states_bytes + 4
  payload = io.to_slice
  payload_sha256 = Digest::SHA256.hexdigest(payload)
  manifest = <<-JSON
  {
    "schema":"qwen-image21-conditioning","schema_version":1,
    "model":{"repo":"Qwen/Qwen-Image-2.1","revision":"test-revision"},
    "prompt":"a red cube","noise":{"seed":7},
    "image":{"width":32,"height":32,"vae_scale_factor":16,
      "latent_height":2,"latent_width":2,"img_shapes":[[1,2,2]]},
    "payload_file":"qwen_image21_conditioning.bin",
    "payload_nbytes":#{payload.size},"payload_sha256":"#{payload_sha256}",
    "tensors":{
      "encoder_hidden_states":{"dtype":"float32-le","shape":[2,4096],"offset_bytes":0,"nbytes":#{states_bytes}},
      "encoder_hidden_states_mask":{"dtype":"uint8","shape":[2],"offset_bytes":#{states_bytes},"nbytes":2},
      "encoder_img_mask":{"dtype":"uint8","shape":[2],"offset_bytes":#{states_bytes + 2},"nbytes":2},
      "initial_target_latents":{"dtype":"float32-le","shape":[4,64],"offset_bytes":#{noise_offset},"nbytes":#{latents.size * 4}}
    }
  }
  JSON
  {manifest, payload}
end

describe ML::GGUF::QwenImage21ConditioningBundle do
  it "loads the official unpatched 64-channel text-to-image tensor layout" do
    manifest, payload = qwen_image21_test_conditioning
    bundle = ML::GGUF::QwenImage21ConditioningBundle.parse(manifest, payload)

    bundle.image_width.should eq(32)
    bundle.image_height.should eq(32)
    bundle.latent_height.should eq(2)
    bundle.latent_width.should eq(2)
    bundle.img_shapes.should eq([StaticArray[1, 2, 2]])
    bundle.encoder_hidden_states.size.should eq(2 * 4096)
    bundle.encoder_hidden_states[0].should eq(0.25_f32)
    bundle.encoder_hidden_states_mask.should eq([true, true])
    bundle.encoder_img_mask.should eq([false, false])
    bundle.initial_target_latents.size.should eq(4 * 64)
    bundle.initial_target_latents[0].should eq(1.0_f32)
    bundle.initial_target_latents[63].should eq(-2.0_f32)
    bundle.initial_target_latents[64].should eq(3.0_f32)
  end

  it "rejects the older Qwen-Image 2x2-packed geometry" do
    manifest, payload = qwen_image21_test_conditioning
    packed = manifest.gsub("\"latent_height\":2", "\"latent_height\":1")
    expect_raises(ArgumentError) do
      ML::GGUF::QwenImage21ConditioningBundle.parse(packed, payload)
    end
  end

  it "rejects truncated and non-finite payloads before denoising" do
    manifest, payload = qwen_image21_test_conditioning
    expect_raises(ArgumentError) do
      ML::GGUF::QwenImage21ConditioningBundle.parse(manifest, payload[0, payload.size - 1])
    end
    corrupt = payload.dup
    IO::ByteFormat::LittleEndian.encode(Float32::NAN, corrupt[0, 4])
    expect_raises(ArgumentError) do
      ML::GGUF::QwenImage21ConditioningBundle.parse(manifest, corrupt)
    end
  end
end

baseline_ab_manifest = ENV["QWEN_IMAGE21_AB_BASELINE_MANIFEST"]?
native_ab_manifest = ENV["QWEN_IMAGE21_AB_NATIVE_MANIFEST"]?
raise ArgumentError.new("set both QWEN_IMAGE21_AB_*_MANIFEST variables") if baseline_ab_manifest.nil? != native_ab_manifest.nil?

if baseline_ab_manifest && native_ab_manifest
  describe "optional real Qwen-Image 2.1 conditioning A/B handoff" do
    it "loads both bundles with identical masks and initial latents" do
      baseline = ML::GGUF::QwenImage21ConditioningBundle.load(baseline_ab_manifest)
      native = ML::GGUF::QwenImage21ConditioningBundle.load(native_ab_manifest)

      native.prompt.should eq(baseline.prompt)
      native.seed.should eq(baseline.seed)
      native.model_revision.should eq(baseline.model_revision)
      native.image_width.should eq(baseline.image_width)
      native.image_height.should eq(baseline.image_height)
      native.img_shapes.should eq(baseline.img_shapes)
      native.encoder_hidden_states_mask.should eq(baseline.encoder_hidden_states_mask)
      native.encoder_img_mask.should eq(baseline.encoder_img_mask)
      native.initial_target_latents.should eq(baseline.initial_target_latents)
      native.encoder_hidden_states.size.should eq(10 * 4096)
      native.encoder_hidden_states.should_not eq(baseline.encoder_hidden_states)
    end
  end
end
