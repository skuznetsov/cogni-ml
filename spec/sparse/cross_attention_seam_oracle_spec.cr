require "json"
require "digest/sha256"
require "../spec_helper"

private CROSS_ATTENTION_SEAM_SOURCE_DIGESTS = {
  "sparse_basic"             => "99dbcb7298238fdb6b6d47918ec068f618c68067653489d22b32c136c1ae0e78",
  "sparse_config"            => "6a9cb44608829cb2c11591685282959928c6081c5bc659687aa8395765c5f91b",
  "sparse_attention_modules" => "cfa99afda24e5840118814e80cefae783423d01d47e6322fe967412aff11f6cf",
  "sparse_full_attention"    => "bee0c32089f060c8136292f41a2cc7a952a8679a8b6d113d14c772f2c681e520",
  "modulated_cross_block"    => "fab9838c79b5fa9cbc6055c4a958f5a8e6f394f94e1691140be022caab7078d2",
  "norm"                     => "f89c40abf3356f7b06fc85f0498cd77eefb677a0e0d370a43d14a735f4c40172",
  "structured_latent_flow"   => "76454ead55d112214e36db8de5e9b3d1d4128f05d25256fb6c581b4c1a588021",
  "image_feature_extractor"  => "12530b23e8b6a2cc6b87d8cd01922c7b0085199a365b731f19dd0e7ef4919150",
  "trellis2_image_to_3d"     => "e2addfca672354284b23d1541a8f49228d5a727d49220fcb8512cca2cdd38ce9",
}

private CROSS_ATTENTION_SEAM_CONFIG_DIGESTS = {
  "configs/gen/slat_flow_img2shape_dit_1_3B_512_bf16.json"           => "6989e77f8b5ff4eb524522649e7708bee56526544f5d059f55760fcc5567d388",
  "configs/gen/slat_flow_img2shape_dit_1_3B_512_bf16_ft1024.json"    => "310f9588a6d3ebc7c036b1bb5be79e96343ff232cc9c5627e0d590f101949da0",
  "configs/gen/slat_flow_imgshape2tex_dit_1_3B_512_bf16.json"        => "a344cef8feca45a4efc2201c53e772ebd77b97f9aff1b1e91328576ab6f3e1c6",
  "configs/gen/slat_flow_imgshape2tex_dit_1_3B_512_bf16_ft1024.json" => "df727c8b2bcd6fc592e4feb0489ddec57c73f2f4fdb5b4028ded8648d6d37057",
}

private CROSS_ATTENTION_SEAM_LOCAL_DENSE_DIGEST =
  "6901223b87c6ec4ee39113c7684191b18c68a04edb64bc53ff800a407d82a7ce"

private def cross_attention_seam_fixture : JSON::Any
  JSON.parse(File.read(File.join(
    __DIR__,
    "../fixtures/trellis2/sparse_cross_attention_seam_cpu_v1.json"
  )))
end

private def cross_attention_seam_f32(
  payload : JSON::Any,
  output = [] of Float32,
) : Array(Float32)
  if hash = payload.as_h?
    if values = hash["values"]?
      return cross_attention_seam_f32(values, output)
    end
  elsif array = payload.as_a?
    array.each { |entry| cross_attention_seam_f32(entry, output) }
  else
    output << payload.as_f.to_f32
  end
  output
end

private def cross_attention_seam_f32le_sha256(values : Indexable(Float32)) : String
  bytes = Bytes.new(values.size * 4, 0_u8)
  values.each_with_index do |value, index|
    IO::ByteFormat::LittleEndian.encode(value, bytes[index * 4, 4])
  end
  Digest::SHA256.hexdigest(bytes)
end

describe "TRELLIS.2 sparse cross-attention seam oracle" do
  it "pins the owner, production configuration, and context boundary" do
    fixture = cross_attention_seam_fixture
    fixture["schema"].as_s.should eq(
      "cogni-ml/trellis2/sparse-cross-attention-seam-oracle/v1"
    )

    provenance = fixture["provenance"]
    provenance["commit"].as_s.should eq(
      "75fbf0183001ed9876c8dbb35de6b68552ee08bd"
    )
    provenance["generator"].as_s.should eq(
      "tools/trellis2_oracle/export_sparse_cross_attention_seam.py"
    )
    provenance["network"].as_s.should eq("none")
    provenance["device"].as_s.should eq("cpu")
    provenance["dtype"].as_s.should eq("float32")
    provenance["weights"].as_s.should eq("synthetic patterned F32")
    provenance["upstream_class_executed"].as_s.should eq(
      "ModulatedSparseTransformerCrossBlock._forward"
    )
    provenance["upstream_forward_executed"].as_bool.should be_true
    provenance["upstream_attention_backend_executed"].as_bool.should be_false
    provenance["independent_reference_executed"].as_bool.should be_true
    provenance["sources"].as_h.each do |name, source|
      CROSS_ATTENTION_SEAM_SOURCE_DIGESTS[name].should eq(source["sha256"].as_s)
    end

    production = fixture["production_configuration"]
    production["model"].as_s.should eq("ElasticSLatFlowModel")
    production["block"].as_s.should eq("ModulatedSparseTransformerCrossBlock")
    production["channels"].as_i.should eq(1536)
    production["context_channels"].as_i.should eq(1024)
    production["num_heads"].as_i.should eq(12)
    production["share_mod"].as_bool.should be_true
    production["qk_rms_norm_cross"].as_bool.should be_true
    production["context"]["standard_pipeline_representation"].as_s.should eq(
      "dense [B,N,1024]"
    )
    production["context"]["model_api_alternate_representation"].as_s.should eq(
      "VarLenTensor [B,*,1024]"
    )
    production["context"]["exact_token_count"].as_s.should eq(
      "unresolved-gated-config"
    )
    production["configs"].as_a.each do |config|
      config["model"].as_s.should eq("ElasticSLatFlowModel")
      config["model_channels"].as_i.should eq(1536)
      config["cond_channels"].as_i.should eq(1024)
      CROSS_ATTENTION_SEAM_CONFIG_DIGESTS[config["path"].as_s].should eq(
        config["sha256"].as_s
      )
    end
    production["configs"].as_a.map { |config| config["variant"].as_s }.should eq([
      "shape_512",
      "shape_1024",
      "texture_512",
      "texture_1024",
    ])
  end

  it "executes norm2, cross-attention, and the second residual only" do
    fixture = cross_attention_seam_fixture
    contract = fixture["contract"]
    contract["boundary"].as_s.should eq("before norm3")
    contract["query_sequence_lengths"].as_a.map(&.as_i).should eq([
      2_i64, 0_i64, 2_i64,
    ])
    contract["batch_broadcast_map"].as_a.map(&.as_i).should eq([
      0_i64, 0_i64, 2_i64, 2_i64,
    ])
    contract["dense_context_lengths"].as_a.map(&.as_i).uniq.size.should eq(1)
    contract["coordinate_map_identity_preserved"].as_bool.should be_true
    contract["input_and_parameters_unchanged"].as_bool.should be_true
    contract["empty_query_batch_emits_no_rows"].as_bool.should be_true
    contract["failure_precedence"].as_a.map(&.as_s).should eq([
      "runtime/source/config pins before execution",
      "CPU/F32/contiguous sparse input and dense context checks before _forward",
      "norm2 arithmetic before cross-attention",
      "cross-attention output before the second residual add",
      "norm3 input capture and boundary stop before norm3 arithmetic",
    ])

    stages = fixture["stages"]
    after_self = cross_attention_seam_f32(stages["after_self"])
    cross_output = cross_attention_seam_f32(stages["cross_attention_output"])
    after_cross = cross_attention_seam_f32(stages["after_cross"])
    after_self.zip(cross_output).map { |(x, h)| x + h }.should eq(after_cross)

    stages.as_h.each do |name, payload|
      digest = cross_attention_seam_f32le_sha256(
        cross_attention_seam_f32(payload)
      )
      payload["f32le_sha256"].as_s.should eq(digest)
      fixture["stage_f32le_sha256"][name].as_s.should eq(digest)
    end

    parameters = fixture["parameters"]
    parameters["epsilon"].as_f.should eq(1.0e-6)
    %w(norm2_weight norm2_bias to_q_weight to_q_bias to_kv_weight to_kv_bias q_gamma k_gamma to_out_weight to_out_bias).each do |name|
      parameters[name]["f32le_sha256"].as_s.should eq(
        cross_attention_seam_f32le_sha256(
          cross_attention_seam_f32(parameters[name])
        )
      )
    end
  end

  it "rejects conflating the dense oracle with the sparse carrier" do
    compatibility = cross_attention_seam_fixture["compatibility"]
    dense = compatibility["dense_trellis_cross_attention"]
    dense_source = dense["source"]
    dense_source["path"].as_s.should eq(
      "src/ml/three_d/trellis2/dense_block.cr"
    )
    dense_source["sha256"].as_s.should eq(
      CROSS_ATTENTION_SEAM_LOCAL_DENSE_DIGEST
    )
    Digest::SHA256.hexdigest(
      File.read(File.join(__DIR__, "../../", dense_source["path"].as_s))
    ).should eq(CROSS_ATTENTION_SEAM_LOCAL_DENSE_DIGEST)
    dense["verdict"].as_s.should eq("arithmetic-oracle-only")
    dense["accepts_dense_uniform_overlap"].as_bool.should be_true
    dense["represents_per_batch_query_lengths"].as_bool.should be_false
    dense["preserves_flat_sparse_layout"].as_bool.should be_false
    dense["requires_full_dense_buffers"].as_bool.should be_true
    dense["production_sparse_carrier"].as_bool.should be_false

    attention = ML::ThreeD::Trellis2::CrossAttention.new(
      channels: 4,
      context_channels: 6,
      num_heads: 2,
      device: ML::Tensor::Device::CPU
    )
    attention.to_q.weight.data.cpu_data.not_nil!.fill(0.0_f32)
    attention.to_q.bias.not_nil!.data.cpu_data.not_nil!.fill(0.0_f32)
    attention.to_kv.weight.data.cpu_data.not_nil!.fill(0.0_f32)
    attention.to_kv.bias.not_nil!.data.cpu_data.not_nil!.fill(0.0_f32)
    attention.to_out.weight.data.cpu_data.not_nil!.fill(0.0_f32)
    attention.to_out.bias.not_nil!.data.cpu_data.not_nil!.fill(1.0_f32)
    dense_query = ML::Autograd::Variable.new(
      ML::Tensor.zeros(3, 2, 4, device: ML::Tensor::Device::CPU),
      requires_grad: false
    )
    dense_context = ML::Autograd::Variable.new(
      ML::Tensor.zeros(3, 3, 6, device: ML::Tensor::Device::CPU),
      requires_grad: false
    )
    dense_output = attention.forward(
      dense_query,
      dense_context,
      valid_query_length: 2
    ).data
    dense_output.shape.should eq(ML::Shape.new(3, 2, 4))
    (dense_output.shape[0] * dense_output.shape[1]).should eq(6)
    dense_output.cpu_data.not_nil![8, 8].should eq(Array.new(8, 1.0_f32))
    cross_attention_seam_fixture["contract"]["query_sequence_lengths"]
      .as_a.sum(&.as_i).should eq(4)

    expect_raises(ArgumentError, /valid length must be within 1../) do
      attention.forward(
        dense_query,
        dense_context,
        valid_query_length: 0
      )
    end
  end
end
