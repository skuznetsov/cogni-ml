require "./spec_helper"
require "digest/sha256"
require "../src/ml/gguf/qwen3vl_text_block"
require "./support/qwen3vl_hidden_norm_fixture"

private QWEN3VL_HIDDEN_ROW_SHA256                  = "6756578c5420adb864a8463d117a1bc300ec8aff70c724d100d487b1f2c3d4c3"
private QWEN3VL_HIDDEN_GAMMA_SHA256                = "7ea7d81ee524694c26e3f13ec24eb94b7f545e6e04530a80b4e5702bb41990a4"
private QWEN3VL_HIDDEN_TEACHER_ROW_SHA256          = "6b116be9c56d2da2334e70f24f73f643ae4e57958c9b167bea20f0fe53db269b"
private QWEN3VL_HIDDEN_EXPECTED_GENERIC_MISMATCHES = 51

private def qwen3vl_hidden_norm_config(
  dim : Int32,
  hidden_backend : ML::GGUF::Qwen3VLTextBlockConfig::HiddenNormBackend? = nil,
  qkv_backend : ML::GGUF::Qwen3VLTextBlockConfig::QKVProjectionBackend? = nil,
) : ML::GGUF::Qwen3VLTextBlockConfig
  ML::GGUF::Qwen3VLTextBlockConfig.new(
    hidden_dim: dim,
    heads: 1,
    kv_heads: 1,
    head_dim: dim,
    intermediate_dim: 1,
    eps: 1e-6_f32,
    qkv_projection_backend: qkv_backend,
    hidden_norm_backend: hidden_backend,
  )
end

describe "Qwen3-VL hidden RMSNorm reduction profile" do
  it "matches the pinned fixed 4096-wide row and keeps raw-word controls discriminating" do
    fixture = Qwen3VLHiddenNormFixture.small_rows
    input_bytes = fixture["residual"]
    gamma_bytes = fixture["gamma"]
    teacher_bytes = fixture["teacher"]
    Digest::SHA256.hexdigest(input_bytes).should eq(QWEN3VL_HIDDEN_ROW_SHA256)
    Digest::SHA256.hexdigest(gamma_bytes).should eq(QWEN3VL_HIDDEN_GAMMA_SHA256)
    Digest::SHA256.hexdigest(teacher_bytes).should eq(QWEN3VL_HIDDEN_TEACHER_ROW_SHA256)

    input = Qwen3VLHiddenNormFixture.decode_bf16(input_bytes)
    gamma = Qwen3VLHiddenNormFixture.decode_bf16(gamma_bytes)
    expected = teacher_bytes
    profile = ML::GGUF::Qwen3VLTextBlockConfig::HiddenNormBackend::Torch26Arm64W4
    config = qwen3vl_hidden_norm_config(4096, hidden_backend: profile)
    actual = Qwen3VLHiddenNormFixture.encode_bf16(
      ML::GGUF::Qwen3VLTextBlock.normalize_hidden_rows(input, 1, gamma, config)
    )

    Digest::SHA256.hexdigest(actual).should eq(QWEN3VL_HIDDEN_TEACHER_ROW_SHA256)
    actual.should eq(expected)
    Qwen3VLHiddenNormFixture.mismatch_count(actual, expected).should eq(0)

    mutated = expected.dup
    mutated[0] = mutated[0] ^ 0x01_u8
    Qwen3VLHiddenNormFixture.mismatch_count(expected, expected).should eq(0)
    Qwen3VLHiddenNormFixture.mismatch_count(expected, mutated).should eq(1)
  end

  it "retains the generic default and keeps hidden selection independent of QKV selection" do
    fixture = Qwen3VLHiddenNormFixture.small_rows
    input = Qwen3VLHiddenNormFixture.decode_bf16(fixture["residual"])
    gamma = Qwen3VLHiddenNormFixture.decode_bf16(fixture["gamma"])
    expected = fixture["teacher"]
    profile = ML::GGUF::Qwen3VLTextBlockConfig::HiddenNormBackend::Torch26Arm64W4
    torch_qkv = ML::GGUF::Qwen3VLTextBlockConfig::QKVProjectionBackend::Torch26Arm64Bf16

    generic = Qwen3VLHiddenNormFixture.encode_bf16(
      ML::GGUF::Qwen3VLTextBlock.normalize_hidden_rows(
        input, 1, gamma, qwen3vl_hidden_norm_config(4096)
      )
    )
    qkv_only = Qwen3VLHiddenNormFixture.encode_bf16(
      ML::GGUF::Qwen3VLTextBlock.normalize_hidden_rows(
        input, 1, gamma, qwen3vl_hidden_norm_config(4096, qkv_backend: torch_qkv)
      )
    )
    profiled_without_qkv = Qwen3VLHiddenNormFixture.encode_bf16(
      ML::GGUF::Qwen3VLTextBlock.normalize_hidden_rows(
        input, 1, gamma, qwen3vl_hidden_norm_config(4096, hidden_backend: profile)
      )
    )
    profiled_with_qkv = Qwen3VLHiddenNormFixture.encode_bf16(
      ML::GGUF::Qwen3VLTextBlock.normalize_hidden_rows(
        input, 1, gamma, qwen3vl_hidden_norm_config(
        4096, hidden_backend: profile, qkv_backend: torch_qkv
      )
      )
    )

    generic.should eq(qkv_only)
    profiled_without_qkv.should eq(profiled_with_qkv)
    Qwen3VLHiddenNormFixture.mismatch_count(generic, expected).should eq(
      QWEN3VL_HIDDEN_EXPECTED_GENERIC_MISMATCHES
    )
    profiled_without_qkv.should eq(expected)
  end

  it "falls back at other hidden widths and does not alter the 128-wide Q/K profile" do
    values = Array(Float32).new(2 * 64) do |index|
      ((index % 19) - 9).to_f32 / 8.0_f32
    end
    gamma = Array(Float32).new(64) { |index| ((index % 7) + 1).to_f32 / 4.0_f32 }
    generic = ML::GGUF::Qwen3VLTextBlock.normalize_hidden_rows(
      values, 2, gamma, qwen3vl_hidden_norm_config(64)
    )
    selected = ML::GGUF::Qwen3VLTextBlock.normalize_hidden_rows(
      values, 2, gamma, qwen3vl_hidden_norm_config(
      64, hidden_backend: ML::GGUF::Qwen3VLTextBlockConfig::HiddenNormBackend::Torch26Arm64W4
    )
    )
    selected.should eq(generic)

    q = Array(Float32).new(2 * 128) do |index|
      ((index % 23) - 11).to_f32 / 8.0_f32
    end
    k = Array(Float32).new(2 * 128) do |index|
      ((index % 17) - 8).to_f32 / 8.0_f32
    end
    q_gamma = Array(Float32).new(128) { |index| ((index % 11) + 1).to_f32 / 8.0_f32 }
    k_gamma = Array(Float32).new(128) { |index| ((index % 13) + 1).to_f32 / 8.0_f32 }
    torch_qkv = ML::GGUF::Qwen3VLTextBlockConfig::QKVProjectionBackend::Torch26Arm64Bf16
    qk_generic = ML::GGUF::Qwen3VLTextBlock.normalize_qk(
      q, k, 2, q_gamma, k_gamma,
      qwen3vl_hidden_norm_config(128, qkv_backend: torch_qkv),
    )
    qk_hidden_selected = ML::GGUF::Qwen3VLTextBlock.normalize_qk(
      q, k, 2, q_gamma, k_gamma,
      qwen3vl_hidden_norm_config(
        128,
        hidden_backend: ML::GGUF::Qwen3VLTextBlockConfig::HiddenNormBackend::Torch26Arm64W4,
        qkv_backend: torch_qkv,
      ),
    )
    qk_hidden_selected.should eq(qk_generic)
  end

  it "rejects malformed shapes without overflowing the token-count product" do
    config = qwen3vl_hidden_norm_config(
      4096, hidden_backend: ML::GGUF::Qwen3VLTextBlockConfig::HiddenNormBackend::Torch26Arm64W4
    )
    expect_raises(ArgumentError) do
      ML::GGUF::Qwen3VLTextBlock.normalize_hidden_rows([] of Float32, 0, [] of Float32, config)
    end
    expect_raises(ArgumentError) do
      ML::GGUF::Qwen3VLTextBlock.normalize_hidden_rows(
        [0.0_f32], Int32::MAX, [1.0_f32], config
      )
    end
    expect_raises(ArgumentError) do
      ML::GGUF::Qwen3VLTextBlock.normalize_hidden_rows(
        [0.0_f32], 1, [1.0_f32], config
      )
    end
    expect_raises(ArgumentError) do
      ML::GGUF::Qwen3VLTextBlock.normalize_hidden_rows(
        Array(Float32).new(4096, 0.0_f32), 1, [1.0_f32], config
      )
    end
  end

  it "keeps the uniform-row result unchanged under the explicit profile" do
    input = Array(Float32).new(4096, 1.0_f32)
    gamma = Array(Float32).new(4096, 1.0_f32)
    generic = ML::GGUF::Qwen3VLTextBlock.normalize_hidden_rows(
      input, 1, gamma, qwen3vl_hidden_norm_config(4096)
    )
    profiled = ML::GGUF::Qwen3VLTextBlock.normalize_hidden_rows(
      input, 1, gamma, qwen3vl_hidden_norm_config(
      4096, hidden_backend: ML::GGUF::Qwen3VLTextBlockConfig::HiddenNormBackend::Torch26Arm64W4
    )
    )
    profiled.should eq(generic)
  end
end
