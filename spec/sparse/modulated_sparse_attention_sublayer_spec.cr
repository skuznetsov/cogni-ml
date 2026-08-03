require "json"
require "digest/sha256"
require "../../src/ml/sparse/adaptive_layer_norm"
require "../../src/ml/sparse/full_self_attention_output"
require "../../src/ml/sparse/gated_residual"
require "../spec_helper"

private MODULATED_SPARSE_ATTENTION_TOLERANCE      = 5.0e-5_f32
private MODULATED_SPARSE_ATTENTION_SOURCE_DIGESTS = {
  "sparse_basic"             => "99dbcb7298238fdb6b6d47918ec068f618c68067653489d22b32c136c1ae0e78",
  "sparse_config"            => "6a9cb44608829cb2c11591685282959928c6081c5bc659687aa8395765c5f91b",
  "sparse_attention_modules" => "cfa99afda24e5840118814e80cefae783423d01d47e6322fe967412aff11f6cf",
  "sparse_full_attention"    => "bee0c32089f060c8136292f41a2cc7a952a8679a8b6d113d14c772f2c681e520",
  "sparse_rope"              => "0525164901c3f1c885e961b747c856b654a4d6840882f34827677fb017dcec00",
  "modulated_cross_block"    => "fab9838c79b5fa9cbc6055c4a958f5a8e6f394f94e1691140be022caab7078d2",
  "norm"                     => "f89c40abf3356f7b06fc85f0498cd77eefb677a0e0d370a43d14a735f4c40172",
  "structured_latent_flow"   => "76454ead55d112214e36db8de5e9b3d1d4128f05d25256fb6c581b4c1a588021",
}
private MODULATED_SPARSE_ATTENTION_CONFIG_DIGESTS = {
  "shape_512" => {
    "configs/gen/slat_flow_img2shape_dit_1_3B_512_bf16.json",
    "6989e77f8b5ff4eb524522649e7708bee56526544f5d059f55760fcc5567d388",
  },
  "shape_1024" => {
    "configs/gen/slat_flow_img2shape_dit_1_3B_512_bf16_ft1024.json",
    "310f9588a6d3ebc7c036b1bb5be79e96343ff232cc9c5627e0d590f101949da0",
  },
  "texture_512" => {
    "configs/gen/slat_flow_imgshape2tex_dit_1_3B_512_bf16.json",
    "a344cef8feca45a4efc2201c53e772ebd77b97f9aff1b1e91328576ab6f3e1c6",
  },
  "texture_1024" => {
    "configs/gen/slat_flow_imgshape2tex_dit_1_3B_512_bf16_ft1024.json",
    "df727c8b2bcd6fc592e4feb0489ddec57c73f2f4fdb5b4028ded8648d6d37057",
  },
}
private MODULATED_SPARSE_ATTENTION_PARAMETER_DIGESTS = {
  "scale_msa"     => "30272d07dc353dc2a03e1297e921c52a89a2eedefc499dd7f3fbf702e662ab81",
  "shift_msa"     => "da6aa829d369a3ce75fc6986428fdc04f61ff46c3fcd1a25138a8416163a5fbe",
  "gate_msa"      => "7c5ef135776935185db47ea8d88b3593098d33647e34ba555df4fa88bf735072",
  "qkv_weight"    => "c7fc08ed15e4b00b7f5c5da55675961eae1ae6c4c0a6a06601422abda0588f3f",
  "qkv_bias"      => "680c876d8d21a587453971f19669b4749243d2c28ca953fc9ee9dd816f29b903",
  "q_gamma"       => "14357446e891d6455dc9173593451f4cf6e18484cd88041f3f3de01c3d7f0ecd",
  "k_gamma"       => "06865877666c64978355de05278ea5f8d91943f60e6d2a9f01234aa1857fd80d",
  "to_out_weight" => "d41d97e08374ffd530e076edccd893dc7e2be2e8e1ed2be7945905c7b9b35532",
  "to_out_bias"   => "5baa06b330ab537e3d367b711c0d1766523c4ae2ea136787180e092d1e00a35b",
}
private MODULATED_SPARSE_ATTENTION_STAGE_DIGESTS = {
  "residual_input"        => "2c3544c2a3301d00f2b730b21e3661095e46225e5ee703c84db0384b4cf0a671",
  "normalized_features"   => "1addcfabc0546f7d6d43cbb7d3fc6dc254d60a7f3f69b70faaacd0c3fcf9ca28",
  "adaptive_affine"       => "754c21e1d496393ac772fe6909a0e55f69f68276fec2064fdc57bf51c8086549",
  "packed_qkv_at_backend" => "4fd45b5908ac1904aa0745862564e38243070982898e592f92bf8161bfe30e56",
  "attention_pre_output"  => "6b1f490d64d264dd2b0374e14c01bf3095c157ead8367d9f9f7e180c76d0ea31",
  "attention_output"      => "fd1ca3eaa3752169d6e6ac51a59649ba21ebec5f27aa0e20f871bdb40f119360",
  "gated_attention"       => "734c566678c8a72d8eb4a95faf03e79ded510862176adae2b348c51ee2092407",
  "after_self"            => "3be9512ed8bacfb5a27ac8139ffc444805d920fbe1463ea15fabd5cf71e6674e",
}

private def modulated_sparse_attention_fixture : JSON::Any
  JSON.parse(File.read(File.join(
    __DIR__,
    "../fixtures/trellis2/sparse_modulated_cross_sublayer_cpu_v1.json"
  )))
end

private def modulated_sparse_attention_f32(
  payload : JSON::Any,
  output = [] of Float32,
) : Array(Float32)
  if array = payload.as_a?
    array.each { |entry| modulated_sparse_attention_f32(entry, output) }
  else
    output << payload.as_f.to_f32
  end
  output
end

private def modulated_sparse_attention_i32(
  payload : JSON::Any,
  output = [] of Int32,
) : Array(Int32)
  if array = payload.as_a?
    array.each { |entry| modulated_sparse_attention_i32(entry, output) }
  else
    output << payload.as_i.to_i32
  end
  output
end

private def modulated_sparse_attention_shape(payload : JSON::Any) : ML::Shape
  ML::Shape.new(
    payload["shape"].as_a.map { |value| value.as_i.to_i32 }
  )
end

private def modulated_sparse_attention_tensor(payload : JSON::Any) : ML::Tensor
  ML::Tensor.from_array(
    modulated_sparse_attention_f32(payload["values"]),
    modulated_sparse_attention_shape(payload)
  )
end

private def modulated_sparse_attention_f32le_sha256(
  values : Indexable(Float32),
) : String
  bytes = Bytes.new(values.size * 4, 0_u8)
  values.each_with_index do |value, index|
    IO::ByteFormat::LittleEndian.encode(value, bytes[index * 4, 4])
  end
  Digest::SHA256.hexdigest(bytes)
end

private def modulated_sparse_attention_input(
  fixture : JSON::Any,
) : ML::Sparse::TensorCPU
  input = fixture["input"]
  residual = fixture["stages"]["residual_input"]
  map = ML::Sparse::CoordinateMap3D.new(
    modulated_sparse_attention_i32(input["coordinates"]),
    input["batch_size"].as_i.to_i32,
    {
      input["spatial_shape"][0].as_i.to_i32,
      input["spatial_shape"][1].as_i.to_i32,
      input["spatial_shape"][2].as_i.to_i32,
    }
  )
  ML::Sparse::TensorCPU.new(
    modulated_sparse_attention_tensor(residual),
    map
  )
end

private def modulated_sparse_attention_linear(
  weight_payload : JSON::Any,
  bias_payload : JSON::Any,
) : ML::NN::Linear
  weight_shape = weight_payload["shape"].as_a.map(&.as_i.to_i32)
  layer = ML::NN::Linear.new(
    weight_shape[1],
    weight_shape[0],
    device: ML::Tensor::Device::CPU
  )
  weight = modulated_sparse_attention_f32(weight_payload["values"])
  weight_data = layer.weight.data.cpu_data.not_nil!
  weight.each_with_index { |value, index| weight_data[index] = value }
  bias = modulated_sparse_attention_f32(bias_payload["values"])
  bias_data = layer.bias.not_nil!.data.cpu_data.not_nil!
  bias.each_with_index { |value, index| bias_data[index] = value }
  layer.weight.requires_grad = false
  layer.bias.not_nil!.requires_grad = false
  layer
end

private def compose_modulated_sparse_attention_sublayer(
  residual : ML::Sparse::TensorCPU,
  scale : ML::Tensor,
  shift : ML::Tensor,
  to_qkv : ML::NN::Linear,
  to_out : ML::NN::Linear,
  q_gamma : ML::Tensor,
  k_gamma : ML::Tensor,
  gate : ML::Tensor,
  max_score_bytes : Int64 = ML::Sparse::FullSelfAttentionPlanCPU::MAX_SCORE_BYTES,
) : Tuple(ML::Sparse::TensorCPU, ML::Sparse::TensorCPU, ML::Sparse::TensorCPU)
  normalized = ML::Sparse::TensorCPU.apply_adaptive_layer_norm(
    residual,
    scale,
    shift
  )
  attention = ML::Sparse::TensorCPU.apply_full_self_attention_output(
    normalized,
    to_qkv,
    to_out,
    2,
    q_gamma: q_gamma,
    k_gamma: k_gamma,
    use_rope: true,
    max_score_bytes: max_score_bytes
  )
  output = ML::Sparse::TensorCPU.apply_gated_residual(
    residual,
    attention,
    gate
  )
  {normalized, attention, output}
end

describe "TRELLIS.2 modulated sparse attention sublayer composition" do
  it "matches every pinned stage through the first gated residual" do
    fixture = modulated_sparse_attention_fixture
    fixture["schema"].as_s.should eq(
      "cogni-ml/trellis2/sparse-modulated-cross-sublayer-oracle/v1"
    )
    provenance = fixture["provenance"]
    provenance["commit"].as_s.should eq(
      "75fbf0183001ed9876c8dbb35de6b68552ee08bd"
    )
    provenance["generator"].as_s.should eq(
      "tools/trellis2_oracle/export_sparse_modulated_cross_sublayer.py"
    )
    provenance["python_version"].as_s.should eq("3.11.9")
    provenance["torch_version"].as_s.should eq("2.9.0")
    provenance["numpy_version"].as_s.should eq("2.1.3")
    provenance["network"].as_s.should eq("none")
    provenance["device"].as_s.should eq("cpu")
    provenance["weights"].as_s.should eq("synthetic patterned F32")
    provenance["upstream_forward_executed"].as_bool.should be_true
    provenance["upstream_attention_backend_executed"].as_bool.should be_false
    MODULATED_SPARSE_ATTENTION_SOURCE_DIGESTS.each do |name, digest|
      provenance["sources"][name]["sha256"].as_s.should eq(digest)
    end

    production = fixture["production_configuration"]
    configs = production["configs"].as_a
    configs.map { |config| config["name"].as_s }.should eq([
      "shape_512",
      "shape_1024",
      "texture_512",
      "texture_1024",
    ])
    production["block_attention_mode"].as_s.should eq("full")
    production["block_qkv_bias_default"].as_bool.should be_true
    production["model_use_checkpoint_default"].as_bool.should be_false
    configs.each do |config|
      config["model"].as_s.should eq("ElasticSLatFlowModel")
      expected = MODULATED_SPARSE_ATTENTION_CONFIG_DIGESTS[config["name"].as_s]
      config["path"].as_s.should eq(expected[0])
      config["sha256"].as_s.should eq(expected[1])
    end

    contract = fixture["contract"]
    contract["order"].as_a.map(&.as_s).should eq([
      "combined_modulation",
      "norm1",
      "adaptive_affine",
      "self_attn",
      "gate_msa",
      "residual_add",
      "norm2_boundary",
    ])
    contract["boundary"].as_s.should eq("before norm2")
    contract["sequence_lengths"].as_a.map(&.as_i).should eq([
      2_i64, 0_i64, 2_i64,
    ])
    contract["batch_broadcast_map"].as_a.map(&.as_i).should eq([
      0_i64, 0_i64, 2_i64, 2_i64,
    ])
    contract["coordinate_object_reused"].as_bool.should be_true
    contract["input_and_parameters_unchanged"].as_bool.should be_true

    parameters = fixture["parameters"]
    stages = fixture["stages"]
    parameters.as_h.each do |name, payload|
      next if name == "epsilon"
      values = modulated_sparse_attention_f32(payload["values"])
      modulated_sparse_attention_f32le_sha256(values).should eq(
        payload["f32le_sha256"].as_s
      )
      payload["f32le_sha256"].as_s.should eq(
        MODULATED_SPARSE_ATTENTION_PARAMETER_DIGESTS[name]
      )
    end
    stages.as_h.each do |name, payload|
      values = modulated_sparse_attention_f32(payload["values"])
      modulated_sparse_attention_f32le_sha256(values).should eq(
        payload["f32le_sha256"].as_s
      )
      payload["f32le_sha256"].as_s.should eq(
        MODULATED_SPARSE_ATTENTION_STAGE_DIGESTS[name]
      )
    end

    residual = modulated_sparse_attention_input(fixture)
    residual_before = residual.features_copy
    scale = modulated_sparse_attention_tensor(parameters["scale_msa"])
    shift = modulated_sparse_attention_tensor(parameters["shift_msa"])
    gate = modulated_sparse_attention_tensor(parameters["gate_msa"])
    q_gamma = modulated_sparse_attention_tensor(parameters["q_gamma"])
    k_gamma = modulated_sparse_attention_tensor(parameters["k_gamma"])
    to_qkv = modulated_sparse_attention_linear(
      parameters["qkv_weight"],
      parameters["qkv_bias"]
    )
    to_out = modulated_sparse_attention_linear(
      parameters["to_out_weight"],
      parameters["to_out_bias"]
    )
    parameter_copies = {
      scale.cpu_data.not_nil!.clone,
      shift.cpu_data.not_nil!.clone,
      gate.cpu_data.not_nil!.clone,
      q_gamma.cpu_data.not_nil!.clone,
      k_gamma.cpu_data.not_nil!.clone,
      to_qkv.weight.data.cpu_data.not_nil!.clone,
      to_qkv.bias.not_nil!.data.cpu_data.not_nil!.clone,
      to_out.weight.data.cpu_data.not_nil!.clone,
      to_out.bias.not_nil!.data.cpu_data.not_nil!.clone,
    }

    normalized, attention, output = compose_modulated_sparse_attention_sublayer(
      residual,
      scale,
      shift,
      to_qkv,
      to_out,
      q_gamma,
      k_gamma,
      gate
    )
    local_norm1 = ML::Sparse::TensorCPU.apply_adaptive_layer_norm(
      residual,
      ML::Tensor.zeros(3, 16),
      ML::Tensor.zeros(3, 16)
    )
    {
      {local_norm1, stages["normalized_features"]},
      {normalized, stages["adaptive_affine"]},
      {attention, stages["attention_output"]},
      {output, stages["after_self"]},
    }.each do |actual, expected_payload|
      actual.coordinate_map.same?(residual.coordinate_map).should be_true
      expected = modulated_sparse_attention_f32(expected_payload["values"])
      actual.features_copy.zip(expected).each do |value, wanted|
        value.should be_close(wanted, MODULATED_SPARSE_ATTENTION_TOLERANCE)
      end
    end
    output.features_copy.should_not eq(attention.features_copy)
    residual.features_copy.should eq(residual_before)
    {
      scale.cpu_data.not_nil!,
      shift.cpu_data.not_nil!,
      gate.cpu_data.not_nil!,
      q_gamma.cpu_data.not_nil!,
      k_gamma.cpu_data.not_nil!,
      to_qkv.weight.data.cpu_data.not_nil!,
      to_qkv.bias.not_nil!.data.cpu_data.not_nil!,
      to_out.weight.data.cpu_data.not_nil!,
      to_out.bias.not_nil!.data.cpu_data.not_nil!,
    }.zip(parameter_copies).each do |actual, before|
      actual.should eq(before)
    end
  end

  it "preserves sequential failure precedence without a combined API" do
    fixture = modulated_sparse_attention_fixture
    parameters = fixture["parameters"]
    residual = modulated_sparse_attention_input(fixture)
    scale = modulated_sparse_attention_tensor(parameters["scale_msa"])
    shift = modulated_sparse_attention_tensor(parameters["shift_msa"])
    gate = modulated_sparse_attention_tensor(parameters["gate_msa"])
    q_gamma = modulated_sparse_attention_tensor(parameters["q_gamma"])
    k_gamma = modulated_sparse_attention_tensor(parameters["k_gamma"])
    to_qkv = modulated_sparse_attention_linear(
      parameters["qkv_weight"],
      parameters["qkv_bias"]
    )
    to_out = modulated_sparse_attention_linear(
      parameters["to_out_weight"],
      parameters["to_out_bias"]
    )
    invalid_gate = ML::Tensor.zeros(1, 1)

    expect_raises(ML::Sparse::SparseTensorError, /scale shape.*\[3, 16\]/) do
      compose_modulated_sparse_attention_sublayer(
        residual,
        ML::Tensor.zeros(1, 1),
        shift,
        to_qkv,
        to_out,
        q_gamma,
        k_gamma,
        invalid_gate,
        63_i64
      )
    end

    expect_raises(
      ML::Sparse::SparseTensorBudgetError,
      /require 64 bytes.*limit is 63/
    ) do
      compose_modulated_sparse_attention_sublayer(
        residual,
        scale,
        shift,
        to_qkv,
        to_out,
        q_gamma,
        k_gamma,
        invalid_gate,
        63_i64
      )
    end

    normalized = ML::Sparse::TensorCPU.apply_adaptive_layer_norm(
      residual,
      scale,
      shift
    )
    attention = ML::Sparse::TensorCPU.apply_full_self_attention_output(
      normalized,
      to_qkv,
      to_out,
      2,
      q_gamma: q_gamma,
      k_gamma: k_gamma,
      use_rope: true
    )
    expect_raises(
      ML::Sparse::SparseTensorError,
      /gate_msa shape.*\[3, 16\]/
    ) do
      ML::Sparse::TensorCPU.apply_gated_residual(
        residual,
        attention,
        invalid_gate
      )
    end
  end
end
