require "./spec_helper"
require "digest/sha256"
require "../src/ml/gguf/qwen3vl_text_block"

# These rows were extracted only after the scratch probe pinned the source
# trace, Q/K projection tensors, official Q/K norm sidecars, native sidecars,
# weight shard, and this source file by SHA-256. The five Q coordinates
# (2,13,105), (13,13,105), (16,13,105), (240,13,105), and (243,13,105)
# share the same 128-value Q input row in the official trace.
#
# Pinned sources: trace d856c64eafca40294e6de8cda3261ab0bb3766bf2c8edc23672de941ca3b8fff;
# Q projection d770317fb1bfdf92cca3ab3e0ece3be4e0dbebd82a88da27e7d3f16145dbae93;
# K projection 773ef1a1e9600ecc8ccb6d4fb46bd93399db03df35372a750fc9ff174aa350b1;
# official Q norm 95f780c845178dc874aa86667b907f4915a18ef970ce8188346031d6eb779373;
# official K norm 6c138005385095ae62161b20d44bbcabc87b6a978d7c8af26562ec2c321eb01d.
private def qwen3vl_qnorm_spec_q_row_hex : String
  [
    "3cd43cbebc90bccf3ce53c293b193ca4bcc3bb47bbb7bd35bbdfbc813cf4bccc",
    "3cae3bab3d193bfe3d51bc933c5fbcc53d77bbc03c103d893bb43d08be433d07",
    "bd88bb953d09bcf1be663bfc3af9bd36bc9e3ba03c3d3cb3bd0dbbd73b913c34",
    "3d1a3cda3c94be763c5cbcd0bc043d373d0b3d81bbeabcf73ce23dbc3c80bd82",
    "bbb83b26bbe43c233c9d3d06bd4c3bc03b2bbc463cab3c37be713c843a08be93",
    "bc29bdad3c65bd223d85be2e3d753cddbceebc9fbce8bd8d3d09bc893d1cbcb3",
    "bbfa3c94bbb43d07bd83bc833c2a3ce2b9db3d0e3d82bc96bc103bbf3b2ebc1e",
    "3d703c873d333db13ca1bc08bca6bbb03cbc3d913d023e4b3bbebd09bd963cb0",
  ].join
end

private def qwen3vl_qnorm_spec_k_row_hex : String
  [
    "be8c3cef3e783dbebe8a3c9abd8e3da93d673dc5bc893d443df93d59bd11bc5b",
    "bbe0be383ba3bdc53d27be58bd0b3ddb3d453d8b3d28be56bca9bdebbde8bc61",
    "3dadbd143cb0bd41bdec3d053ca93e14bc37bacfbab63d0cbcc43aab3c5abb70",
    "bd9c3bf7be24beaf3b2c3d6fbb24be8bbd4cbd43bc853d843db73d5f3d863e16",
    "3eaebe2f3dbc3e22bbdebd53bca9bd933bc9bd6c3d9abd083c47bd14bdbb3b3e",
    "bd913e293c8cb9d5bd8a3de43c39bdd8bdb93c023d483dfd3d26bcd9bdfe3c8f",
    "bda13b02bc5a3d9abcedbcd8bd1a3d19bd21bd163d083e203c33bcdebc59bd23",
    "bd1c3b37bc993ddfbcec3dcd3d943be43cfcbc793ca93cf73daebe3a3c29bcde",
  ].join
end

private def qwen3vl_qnorm_spec_decode_row(hex : String) : Array(Float32)
  raise "Qwen3-VL Q-norm row fixture must contain 128 BF16 values" unless hex.bytesize == 128 * 4
  Array(Float32).new(128) do |index|
    bits = hex.byte_slice(index * 4, 4).not_nil!.to_u16(16)
    (bits.to_u32 << 16).unsafe_as(Float32)
  end
end

private def qwen3vl_qnorm_spec_bf16(bits : UInt16) : Float32
  (bits.to_u32 << 16).unsafe_as(Float32)
end

private def qwen3vl_qnorm_spec_projection(row : Array(Float32)) : Array(Float32)
  Array(Float32).new(128 * 128) do |index|
    output = index // 128
    input = index % 128
    input == 0 ? row[output] : 0.0_f32
  end
end

private def qwen3vl_qnorm_spec_weights(input_layernorm_gamma_bits : UInt16 = 0x3db5_u16) : ML::GGUF::Qwen3VLTextBlockWeights
  q_row = qwen3vl_qnorm_spec_decode_row(qwen3vl_qnorm_spec_q_row_hex)
  k_row = qwen3vl_qnorm_spec_decode_row(qwen3vl_qnorm_spec_k_row_hex)
  width = 128
  zeros = Array(Float32).new(width, 0.0_f32)
  zero_matrix = Array(Float32).new(width * width, 0.0_f32)
  q_norm = zeros.dup
  k_norm = zeros.dup
  q_norm[105] = qwen3vl_qnorm_spec_bf16(0x3fe9_u16)
  k_norm[105] = qwen3vl_qnorm_spec_bf16(0x3ffb_u16)

  ML::GGUF::Qwen3VLTextBlockWeights.new(
    # The BF16 reciprocal of the one-hot row's RMS makes input_layernorm
    # materialize exactly 1 at column 0, so the projection fixture stays exact.
    input_layernorm: Array(Float32).new(width, qwen3vl_qnorm_spec_bf16(input_layernorm_gamma_bits)),
    q_proj: qwen3vl_qnorm_spec_projection(q_row),
    k_proj: qwen3vl_qnorm_spec_projection(k_row),
    v_proj: zero_matrix.dup,
    q_norm: q_norm,
    k_norm: k_norm,
    o_proj: zero_matrix,
    post_attention_layernorm: zeros.dup,
    gate_proj: Array(Float32).new(width, 0.0_f32),
    up_proj: Array(Float32).new(width, 0.0_f32),
    down_proj: Array(Float32).new(width, 0.0_f32),
  )
end

private def qwen3vl_qnorm_spec_forward(eps : Float32 = 1e-6_f32,
                                       input_layernorm_gamma_bits : UInt16 = 0x3db5_u16,
                                       qkv_projection_backend : ML::GGUF::Qwen3VLTextBlockConfig::QKVProjectionBackend? = ML::GGUF::Qwen3VLTextBlockConfig::QKVProjectionBackend::Torch26Arm64Bf16) : Hash(String, Array(Float32))
  width = 128
  tokens = 5
  hidden = Array(Float32).new(tokens * width) do |index|
    index % width == 0 ? 1.0_f32 : 0.0_f32
  end
  config = ML::GGUF::Qwen3VLTextBlockConfig.new(
    hidden_dim: width,
    heads: 1,
    kv_heads: 1,
    head_dim: width,
    intermediate_dim: 1,
    eps: eps,
    qkv_projection_backend: qkv_projection_backend,
  )
  trace = Hash(String, Array(Float32)).new
  ML::GGUF::Qwen3VLTextBlock.forward(
    hidden,
    Array(Bool).new(tokens, true),
    qwen3vl_qnorm_spec_weights(input_layernorm_gamma_bits),
    config,
    trace: trace,
  )
  trace
end

private def qwen3vl_qnorm_spec_bits(value : Float32) : UInt16
  (value.unsafe_as(UInt32) >> 16).to_u16
end

describe "Qwen3-VL Q/K RMSNorm FP32 mean boundary" do
  it "matches the five official Q-norm midpoint coordinates" do
    q_hex = qwen3vl_qnorm_spec_q_row_hex
    q_hex.bytesize.should eq(512)
    Digest::SHA256.hexdigest(q_hex).should eq("9d975b30cc86577cb2a27ebfa69fc3fe08ddbc014381d9f892887ef02a7ca66c")

    trace = qwen3vl_qnorm_spec_forward
    q_proj = trace["layers.0.self_attn.q_proj"]
    q_norm = trace["layers.0.self_attn.q_norm"]
    # The official source tokens are [2, 13, 16, 240, 243]. Each shares the
    # same 128-value input row, so each is represented here by donor slot 0..4.
    # Check every projected donor slot before treating the RMSNorm result as
    # the captured row's output. Current reduction lands one BF16 step high.
    source_tokens = [2, 13, 16, 240, 243]
    source_tokens.each_with_index do |_source_token, slot|
      offset = slot * 128
      qwen3vl_qnorm_spec_bits(q_proj[offset + 105]).should eq(0x3d0e_u16)
    end
    source_tokens.each_with_index do |_source_token, slot|
      offset = slot * 128
      qwen3vl_qnorm_spec_bits(q_norm[offset + 105]).should eq(0x3f83_u16)
    end
  end

  it "keeps the official K-norm row as a positive control" do
    k_hex = qwen3vl_qnorm_spec_k_row_hex
    k_hex.bytesize.should eq(512)
    Digest::SHA256.hexdigest(k_hex).should eq("bbac8ebb7b82ab656ba75a90f6d9afe5e42cf6e67c003c7230aa619034f5c08e")

    trace = qwen3vl_qnorm_spec_forward
    k_proj = trace["layers.0.self_attn.k_proj"]
    k_norm = trace["layers.0.self_attn.k_norm"]
    k_input = qwen3vl_qnorm_spec_decode_row(k_hex)
    offset = 2 * 128
    qwen3vl_qnorm_spec_bits(k_proj[offset + 105]).should eq(qwen3vl_qnorm_spec_bits(k_input[105]))
    qwen3vl_qnorm_spec_bits(k_norm[offset + 105]).should eq(0xbf3e_u16)
  end

  it "distinguishes a wrong-epsilon negative control" do
    # Adjust the one-hot input LayerNorm gamma for this synthetic control so
    # its projected Q rows remain byte-identical to the pinned donors. The
    # only intended change at Q-norm is eps=1e-2.
    trace = qwen3vl_qnorm_spec_forward(1e-2_f32, 0x3e09_u16)
    q_proj = trace["layers.0.self_attn.q_proj"]
    q_norm = trace["layers.0.self_attn.q_norm"]
    [2, 13, 16, 240, 243].each_with_index do |_source_token, slot|
      offset = slot * 128
      qwen3vl_qnorm_spec_bits(q_proj[offset + 105]).should eq(0x3d0e_u16)
      qwen3vl_qnorm_spec_bits(q_norm[offset + 105]).should eq(0x3f09_u16)
    end
  end

  it "keeps the generic reduction for the default unprofiled backend" do
    trace = qwen3vl_qnorm_spec_forward(1e-6_f32, 0x3db5_u16, nil)
    q_norm = trace["layers.0.self_attn.q_norm"]
    k_norm = trace["layers.0.self_attn.k_norm"]
    [0, 1, 2, 3, 4].each do |slot|
      qwen3vl_qnorm_spec_bits(q_norm[slot * 128 + 105]).should eq(0x3f84_u16)
      qwen3vl_qnorm_spec_bits(k_norm[slot * 128 + 105]).should eq(0xbf3e_u16)
    end
  end

  it "keeps the generic reduction for non-128-wide heads even when profiled" do
    row = qwen3vl_qnorm_spec_decode_row(qwen3vl_qnorm_spec_q_row_hex)[0, 64]
    scale = Array(Float32).new(64, 1.0_f32)
    profiled = ML::GGUF::Qwen3VLTextBlockConfig.new(
      hidden_dim: 64,
      heads: 1,
      kv_heads: 1,
      head_dim: 64,
      intermediate_dim: 1,
      qkv_projection_backend: ML::GGUF::Qwen3VLTextBlockConfig::QKVProjectionBackend::Torch26Arm64Bf16,
    )
    generic = ML::GGUF::Qwen3VLTextBlockConfig.new(
      hidden_dim: 64,
      heads: 1,
      kv_heads: 1,
      head_dim: 64,
      intermediate_dim: 1,
    )

    profiled_norms = ML::GGUF::Qwen3VLTextBlock.normalize_qk(row, row, 1, scale, scale, profiled)
    generic_norms = ML::GGUF::Qwen3VLTextBlock.normalize_qk(row, row, 1, scale, scale, generic)
    profiled_norms[:q].map { |value| qwen3vl_qnorm_spec_bits(value) }.should eq(
      generic_norms[:q].map { |value| qwen3vl_qnorm_spec_bits(value) }
    )
    profiled_norms[:k].map { |value| qwen3vl_qnorm_spec_bits(value) }.should eq(
      generic_norms[:k].map { |value| qwen3vl_qnorm_spec_bits(value) }
    )
  end
end
