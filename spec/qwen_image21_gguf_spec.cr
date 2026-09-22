require "./spec_helper"
require "../src/ml/gguf/qwen_image21_gguf"

private def qwen_image21_fixture
  metadata = Hash(String, ML::GGUF::Value).new
  metadata["general.architecture"] = "qwen_image"
  metadata["general.file_type"] = 15_u32

  original_shape = Array(ML::GGUF::Value).new
  original_shape << 4096_i32
  original_shape << 64_i32
  metadata["comfy.gguf.orig_shape.img_in.weight"] = original_shape

  tensors = [] of ML::GGUF::TensorInfo
  add = ->(name : String, dims : Array(Int64), type : ML::GGUF::TensorType) do
    tensors << ML::GGUF::TensorInfo.new(name, dims, type, 0_u64)
  end

  add.call("img_in.weight", [256_i64, 1024_i64], ML::GGUF::TensorType::BF16)
  add.call("modulation.1.weight", [4096_i64, 16384_i64], ML::GGUF::TensorType::BF16)
  add.call("norm_out.linear.weight", [4096_i64, 4096_i64], ML::GGUF::TensorType::BF16)
  add.call("proj_out.weight", [4096_i64, 64_i64], ML::GGUF::TensorType::BF16)
  add.call("time_text_embed.timestep_embedder.linear_1.weight", [256_i64, 4096_i64], ML::GGUF::TensorType::BF16)
  add.call("time_text_embed.timestep_embedder.linear_2.weight", [4096_i64, 4096_i64], ML::GGUF::TensorType::BF16)
  add.call("txt_in.in_layer.weight", [4096_i64, 4096_i64], ML::GGUF::TensorType::BF16)
  add.call("txt_in.out_layer.weight", [4096_i64, 4096_i64], ML::GGUF::TensorType::BF16)
  add.call("txt_in.text_norm.weight", [4096_i64], ML::GGUF::TensorType::F32)

  32.times do |layer|
    prefix = "transformer_blocks.#{layer}"
    add.call("#{prefix}.attn.norm_k.weight", [128_i64], ML::GGUF::TensorType::F32)
    add.call("#{prefix}.attn.norm_q.weight", [128_i64], ML::GGUF::TensorType::F32)
    add.call("#{prefix}.attn.to_k.weight", [4096_i64, 4096_i64], ML::GGUF::TensorType::Q8_0)
    add.call("#{prefix}.attn.to_q.weight", [4096_i64, 4096_i64], ML::GGUF::TensorType::Q8_0)
    add.call("#{prefix}.attn.to_v.weight", [4096_i64, 4096_i64], ML::GGUF::TensorType::Q6_K)
    add.call("#{prefix}.attn.to_out.0.weight", [4096_i64, 4096_i64], ML::GGUF::TensorType::Q8_0)
    add.call("#{prefix}.img_mlp.gate_up.weight", [4096_i64, 24576_i64], ML::GGUF::TensorType::Q5_K)
    add.call("#{prefix}.img_mlp.out.weight", [12288_i64, 4096_i64], ML::GGUF::TensorType::Q6_K)
  end

  {metadata, tensors}
end

describe ML::GGUF::TensorType do
  it "uses the official GGML BF16 type id and scalar byte width" do
    ML::GGUF::TensorType::BF16.value.should eq(30_u32)
    ML::GGUF::TensorType::BF16.block_elements.should eq(1)
    ML::GGUF::TensorType::BF16.block_bytes.should eq(2)
  end
end

describe ML::GGUF::Dequant do
  it "dequantizes little-endian BF16 values" do
    data = Bytes[0x80, 0x3f, 0x20, 0xc0, 0x00, 0x00]
    result = ML::GGUF::Dequant.dequantize(data, ML::GGUF::TensorType::BF16, 3)
    result.should eq([1.0_f32, -2.5_f32, 0.0_f32])
  end
end

describe ML::GGUF::QwenImage21GGUFInventory do
  it "validates the Qwen-Image 2.1 DiT and reports actual tensor types" do
    metadata, tensors = qwen_image21_fixture
    inventory = ML::GGUF::QwenImage21GGUFInventory.new(metadata, tensors)

    inventory.block_count.should eq(32)
    inventory.orig_shape_count.should eq(1)
    inventory.logical_shape("img_in.weight").should eq([4096_i64, 64_i64])
    inventory.reader_compatible?.should be_true
    inventory.unsupported_type_labels.should be_empty
    inventory.type_counts.should eq({
      "BF16" => 8,
      "F32"  => 65,
      "Q8_0" => 96,
      "Q6_K" => 64,
      "Q5_K" => 32,
    })
  end

  it "rejects original shapes that change the element count" do
    metadata, tensors = qwen_image21_fixture
    bad_shape = Array(ML::GGUF::Value).new
    bad_shape << 4096_i32
    bad_shape << 65_i32
    metadata["comfy.gguf.orig_shape.img_in.weight"] = bad_shape

    expect_raises(ArgumentError, /element count/) do
      ML::GGUF::QwenImage21GGUFInventory.new(metadata, tensors)
    end
  end

  it "rejects an incomplete transformer block" do
    metadata, tensors = qwen_image21_fixture
    tensors.reject!(&.name.==("transformer_blocks.7.attn.to_v.weight"))

    expect_raises(ArgumentError, /transformer_blocks\.7\.attn\.to_v\.weight/) do
      ML::GGUF::QwenImage21GGUFInventory.new(metadata, tensors)
    end
  end

  it "reports a parsed but unsupported tensor codec" do
    metadata, tensors = qwen_image21_fixture
    index = tensors.index!(&.name.==("transformer_blocks.0.attn.to_q.weight"))
    previous = tensors[index]
    tensors[index] = ML::GGUF::TensorInfo.new(previous.name, previous.dims, ML::GGUF::TensorType::Q2_K, previous.offset)

    inventory = ML::GGUF::QwenImage21GGUFInventory.new(metadata, tensors)
    inventory.reader_compatible?.should be_false
    inventory.unsupported_type_labels.should eq(["Q2_K"])
  end
end
