require "./spec_helper"
require "digest/sha256"
require "../src/ml/gguf/qwen3vl_text_block"
require "./support/qwen3vl_hidden_norm_fixture"

private QWEN3VL_HIDDEN_SIDECAR_INPUT_SHA256   = "d366a2c8f9e929874cbb75228504f1f8f9a06cd40b33e19661f9c0eb1cc56df3"
private QWEN3VL_HIDDEN_SIDECAR_TEACHER_SHA256 = "08aa01e5ca11c94b609bd76c4e40d00889e9fda96b188b26ec299a55a78e93be"
private QWEN3VL_HIDDEN_SIDECAR_GENERIC_SHA256 = "9af333ef7cefbf4ab622ea696a781596721548a16049b3b0589794a7c4b0e077"
private QWEN3VL_HIDDEN_SIDECAR_GAMMA_SHA256   = "7ea7d81ee524694c26e3f13ec24eb94b7f545e6e04530a80b4e5702bb41990a4"
private QWEN3VL_HIDDEN_SIDECAR_ELEMENTS       = 244 * 4096
private QWEN3VL_HIDDEN_SIDECAR_BYTES          = QWEN3VL_HIDDEN_SIDECAR_ELEMENTS * 2
private QWEN3VL_HIDDEN_SIDECAR_ROW50_OFFSET   = 50 * 4096 * 2

private def qwen3vl_hidden_sidecar_config(
  backend : ML::GGUF::Qwen3VLTextBlockConfig::HiddenNormBackend? = nil,
) : ML::GGUF::Qwen3VLTextBlockConfig
  ML::GGUF::Qwen3VLTextBlockConfig.new(
    hidden_dim: 4096,
    heads: 32,
    kv_heads: 8,
    head_dim: 128,
    intermediate_dim: 1,
    eps: 1e-6_f32,
    hidden_norm_backend: backend,
  )
end

private def qwen3vl_hidden_sidecar_bytes(path : String, sha256 : String) : Bytes
  bytes = File.read(path).to_slice.dup
  bytes.size.should eq(QWEN3VL_HIDDEN_SIDECAR_BYTES)
  Digest::SHA256.hexdigest(bytes).should eq(sha256)
  bytes
end

if fixture_dir = ENV["QWEN3VL_HIDDEN_NORM_FIXTURE_DIR"]?
  describe "Qwen3-VL hidden RMSNorm pinned 244-row sidecar regression" do
    it "matches every profiled BF16 output and retains the exact 51-word generic negative control" do
      input_bytes = qwen3vl_hidden_sidecar_bytes(
        File.join(fixture_dir, "post_attention_residual_torch_add.bf16le"),
        QWEN3VL_HIDDEN_SIDECAR_INPUT_SHA256,
      )
      expected_bytes = qwen3vl_hidden_sidecar_bytes(
        File.join(fixture_dir, "post_attention_norm_torch_f32_mean_rsqrt.bf16le"),
        QWEN3VL_HIDDEN_SIDECAR_TEACHER_SHA256,
      )
      generic_reference = qwen3vl_hidden_sidecar_bytes(
        File.join(fixture_dir, "native_emulation_source_shaped.bf16le"),
        QWEN3VL_HIDDEN_SIDECAR_GENERIC_SHA256,
      )
      fixture = Qwen3VLHiddenNormFixture.small_rows
      input_row = Bytes.new(Qwen3VLHiddenNormFixture::ROW_BYTES)
      teacher_row = Bytes.new(Qwen3VLHiddenNormFixture::ROW_BYTES)
      Qwen3VLHiddenNormFixture::ROW_BYTES.times do |index|
        offset = QWEN3VL_HIDDEN_SIDECAR_ROW50_OFFSET + index
        input_row[index] = input_bytes[offset]
        teacher_row[index] = expected_bytes[offset]
      end
      fixture["residual"].should eq(input_row)
      fixture["teacher"].should eq(teacher_row)

      input = Qwen3VLHiddenNormFixture.decode_bf16(input_bytes)
      gamma_bytes = fixture["gamma"]
      Digest::SHA256.hexdigest(gamma_bytes).should eq(QWEN3VL_HIDDEN_SIDECAR_GAMMA_SHA256)
      gamma = Qwen3VLHiddenNormFixture.decode_bf16(gamma_bytes)
      generic_config = qwen3vl_hidden_sidecar_config
      profiled_config = qwen3vl_hidden_sidecar_config(
        ML::GGUF::Qwen3VLTextBlockConfig::HiddenNormBackend::Torch26Arm64W4
      )
      generic = Qwen3VLHiddenNormFixture.encode_bf16(
        ML::GGUF::Qwen3VLTextBlock.normalize_hidden_rows(input, 244, gamma, generic_config)
      )
      profiled = Qwen3VLHiddenNormFixture.encode_bf16(
        ML::GGUF::Qwen3VLTextBlock.normalize_hidden_rows(input, 244, gamma, profiled_config)
      )

      Digest::SHA256.hexdigest(profiled).should eq(QWEN3VL_HIDDEN_SIDECAR_TEACHER_SHA256)
      profiled.should eq(expected_bytes)
      Digest::SHA256.hexdigest(generic).should eq(QWEN3VL_HIDDEN_SIDECAR_GENERIC_SHA256)
      generic.should eq(generic_reference)
      Qwen3VLHiddenNormFixture.mismatch_count(profiled, expected_bytes).should eq(0)
      Qwen3VLHiddenNormFixture.mismatch_count(generic, expected_bytes).should eq(51)
    end
  end
end
