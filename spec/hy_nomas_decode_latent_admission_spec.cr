require "spec"
require "../examples/support/hy_nomas_decode_latent_admission"

describe HyNomasLatentDecodeAdmission do
  default_path = "/tmp/echo-continuation.json"
  default_sha = "a" * 64
  rolewright_path = "/tmp/rolewright-continuation.json"
  rolewright_sha = "b" * 64

  it "keeps the historical decoder cache pin unless an explicit pair is supplied" do
    HyNomasLatentDecodeAdmission.resolve_decoder_cache!("", "").should be_nil
    HyNomasLatentDecodeAdmission.resolve_decoder_cache!("/tmp/recovered-cache.json", rolewright_sha)
      .should eq({"/tmp/recovered-cache.json", rolewright_sha})
  end

  it "rejects one-sided or malformed recovered decoder cache identities" do
    [
      {"/tmp/recovered-cache.json", ""},
      {"", rolewright_sha},
      {"relative/cache.json", rolewright_sha},
      {"/tmp/recovered-cache.json", "B" * 64},
      {"/tmp/recovered-cache.json", "b" * 63},
    ].each do |path, sha|
      expect_raises(ArgumentError) do
        HyNomasLatentDecodeAdmission.resolve_decoder_cache!(path, sha)
      end
    end
  end

  it "preserves the legacy Echo continuation default and omits it for 30 frames" do
    HyNomasLatentDecodeAdmission.resolve_continuation!(60, true, "", "",
      default_path, default_sha).should eq({default_path, default_sha})
    HyNomasLatentDecodeAdmission.resolve_continuation!(90, true, "", "",
      default_path, default_sha).should eq({default_path, default_sha})
    HyNomasLatentDecodeAdmission.resolve_continuation!(30, true, "", "",
      default_path, default_sha).should be_nil
  end

  it "requires an explicit paired continuation for a non-default long decode" do
    expect_raises(ArgumentError, /non-default inputs require explicit/) do
      HyNomasLatentDecodeAdmission.resolve_continuation!(60, false, "", "",
        default_path, default_sha)
    end

    HyNomasLatentDecodeAdmission.resolve_continuation!(60, false, rolewright_path,
      rolewright_sha, default_path, default_sha).should eq({rolewright_path, rolewright_sha})
  end

  it "rejects one-sided, malformed, and unused continuation overrides" do
    invalid = [
      {60, true, rolewright_path, ""},
      {60, true, "", rolewright_sha},
      {60, true, "relative/continuation.json", rolewright_sha},
      {60, true, rolewright_path, "B" * 64},
      {30, true, rolewright_path, rolewright_sha},
    ]
    invalid.each do |frames, default_base, path, sha|
      expect_raises(ArgumentError) do
        HyNomasLatentDecodeAdmission.resolve_continuation!(frames, default_base, path, sha,
          default_path, default_sha)
      end
    end
  end

  it "admits the explicit two-segment previous-window inputs only as one complete set" do
    imported = HyNomasLatentDecodeAdmission.resolve_previous_window!(
      30, 30, "", "", "/tmp/previous-latent.f32le", "c" * 64,
      "/tmp/previous-noise.f32le", "d" * 64)
    imported.should eq({"/tmp/previous-latent.f32le", "c" * 64,
                        "/tmp/previous-noise.f32le", "d" * 64})

    HyNomasLatentDecodeAdmission.resolve_previous_window!(30, 30, "", "", "", "", "", "")
      .should be_nil

    invalid = [
      {"/tmp/previous-latent.f32le", "c" * 64, "", ""},
      {"", "", "/tmp/previous-noise.f32le", "d" * 64},
      {"/tmp/previous-latent.f32le", "", "/tmp/previous-noise.f32le", "d" * 64},
      {"/tmp/previous-latent.f32le", "c" * 64, "/tmp/previous-noise.f32le", ""},
    ]
    invalid.each do |previous_latent, previous_latent_sha, previous_noise, previous_noise_sha|
      expect_raises(ArgumentError, /all four previous-window/) do
        HyNomasLatentDecodeAdmission.resolve_previous_window!(30, 30, "", "",
          previous_latent, previous_latent_sha, previous_noise, previous_noise_sha)
      end
    end
  end

  it "rejects malformed digests, incompatible modes, and conflicting noise sources" do
    invalid = [
      {"relative/previous-latent.f32le", "c" * 64, "/tmp/previous-noise.f32le", "d" * 64,
       30, 30, "", ""},
      {"/tmp/previous-latent.f32le", "C" * 64, "/tmp/previous-noise.f32le", "d" * 64,
       30, 30, "", ""},
      {"/tmp/previous-latent.f32le", "c" * 63, "/tmp/previous-noise.f32le", "d" * 64,
       30, 30, "", ""},
      {"/tmp/previous-latent.f32le", "c" * 64, "relative/previous-noise.f32le", "d" * 64,
       30, 30, "", ""},
      {"/tmp/previous-latent.f32le", "c" * 64, "/tmp/previous-noise.f32le", "D" * 64,
       30, 30, "", ""},
      {"/tmp/previous-latent.f32le", "c" * 64, "/tmp/previous-noise.f32le", "d" * 63,
       30, 30, "", ""},
      {"/tmp/previous-latent.f32le", "c" * 64, "/tmp/previous-noise.f32le", "d" * 64,
       60, 60, "", ""},
      {"/tmp/previous-latent.f32le", "c" * 64, "/tmp/previous-noise.f32le", "d" * 64,
       30, 180, "", ""},
      {"/tmp/previous-latent.f32le", "c" * 64, "/tmp/previous-noise.f32le", "d" * 64,
       30, 30, "/tmp/continuation.json", "e" * 64},
    ]
    invalid.each do |previous_latent, previous_latent_sha, previous_noise, previous_noise_sha, latent_frames, latent_bank_frames, continuation_path, continuation_sha|
      expect_raises(ArgumentError) do
        HyNomasLatentDecodeAdmission.resolve_previous_window!(latent_frames, latent_bank_frames,
          continuation_path, continuation_sha, previous_latent, previous_latent_sha,
          previous_noise, previous_noise_sha)
      end
    end
  end

  it "requires exact path and SHA binding between supplied inputs and continuation base" do
    HyNomasLatentDecodeAdmission.validate_base_binding!(rolewright_path, rolewright_sha,
      rolewright_path, rolewright_sha).should be_nil

    [{"/tmp/other.json", rolewright_sha}, {rolewright_path, "c" * 64}].each do |path, sha|
      expect_raises(ArgumentError) do
        HyNomasLatentDecodeAdmission.validate_base_binding!(path, sha,
          rolewright_path, rolewright_sha)
      end
    end
  end

  it "admits only the fixed 512x288 v2 geometry and binds its dynamic Qwen fields" do
    geometry = HyNomasLatentDecodeAdmission::DEFAULT_GEOMETRY_FIELDS.dup
    geometry["qwen_crop"] = JSON.parse("[108, 1388]")
    geometry["qwen_valid_tokens"] = JSON::Any.new(377_i64)
    qwen_source = JSON.parse(%({"crop":[108,1388]}))

    HyNomasLatentDecodeAdmission.validate_default_geometry!(JSON::Any.new(geometry),
      qwen_source, 377).should be_nil

    wrong_spatial = geometry.dup
    wrong_spatial["decoder_noise_shape"] = JSON.parse("[1,3,120,416,768]")
    expect_raises(ArgumentError, /decoder_noise_shape/) do
      HyNomasLatentDecodeAdmission.validate_default_geometry!(JSON::Any.new(wrong_spatial),
        qwen_source, 377)
    end

    widened = geometry.dup
    widened["pixel_width"] = JSON::Any.new(768_i64)
    expect_raises(ArgumentError, /geometry inventory/) do
      HyNomasLatentDecodeAdmission.validate_default_geometry!(JSON::Any.new(widened),
        qwen_source, 377)
    end

    expect_raises(ArgumentError, /Qwen geometry/) do
      HyNomasLatentDecodeAdmission.validate_default_geometry!(JSON::Any.new(geometry),
        qwen_source, 369)
    end
  end

  it "requires matching fixed-geometry base and continuation RoPE records" do
    identity = {"F32", [1, 5760, 1, 128], 2_949_120_i64, "d" * 64}
    HyNomasLatentDecodeAdmission.validate_matching_rope_records!("decoder.rope_cos",
      identity, identity).should be_nil

    drifted = [
      {"BF16", identity[1], identity[2], identity[3]},
      {identity[0], [1, 5760, 1, 64], identity[2], identity[3]},
      {identity[0], identity[1], identity[2] - 4, identity[3]},
      {identity[0], identity[1], identity[2], "e" * 64},
    ]
    drifted.each do |record|
      expect_raises(ArgumentError) do
        HyNomasLatentDecodeAdmission.validate_matching_rope_records!("decoder.rope_sin",
          identity, record)
      end
    end

    wrong_but_equal_geometry = {"F32", [1, 3, 720, 288, 512], identity[2], identity[3]}
    expect_raises(ArgumentError, /geometry or digest/) do
      HyNomasLatentDecodeAdmission.validate_matching_rope_records!("decoder.rope_cos",
        wrong_but_equal_geometry, wrong_but_equal_geometry)
    end
  end
end
