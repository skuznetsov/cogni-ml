require "json"
require "digest/sha256"
require "../spec_helper"

private SPARSE_ATTENTION_CONSUMER_SOURCE_DIGESTS = {
  "sparse_transformer_blocks"    => "622e7c5374976c053fb96151c706356b44244e241afab180e0fbdde5da6770d9",
  "modulated_transformer_blocks" => "fab9838c79b5fa9cbc6055c4a958f5a8e6f394f94e1691140be022caab7078d2",
  "sparse_tensor_arithmetic"     => "99dbcb7298238fdb6b6d47918ec068f618c68067653489d22b32c136c1ae0e78",
  "structured_latent_flow"       => "76454ead55d112214e36db8de5e9b3d1d4128f05d25256fb6c581b4c1a588021",
}
private SPARSE_ATTENTION_CONSUMER_CONFIG_DIGESTS = {
  "configs/gen/slat_flow_img2shape_dit_1_3B_512_bf16.json"           => "6989e77f8b5ff4eb524522649e7708bee56526544f5d059f55760fcc5567d388",
  "configs/gen/slat_flow_img2shape_dit_1_3B_512_bf16_ft1024.json"    => "310f9588a6d3ebc7c036b1bb5be79e96343ff232cc9c5627e0d590f101949da0",
  "configs/gen/slat_flow_imgshape2tex_dit_1_3B_512_bf16.json"        => "a344cef8feca45a4efc2201c53e772ebd77b97f9aff1b1e91328576ab6f3e1c6",
  "configs/gen/slat_flow_imgshape2tex_dit_1_3B_512_bf16_ft1024.json" => "df727c8b2bcd6fc592e4feb0489ddec57c73f2f4fdb5b4028ded8648d6d37057",
}
private SPARSE_ATTENTION_CONSUMER_STAGE_DIGESTS = {
  "residual_input"               => "ebd1d95a72c83eb3d50ac0ecaa1112f78a172df0573ea7dc0b1783ea41c3a3c8",
  "attention_output"             => "038cbf2423db1f7da606c7e91709b1feadc5536c4cd646a5e2faae353a42e5d5",
  "gate_msa"                     => "c9eb25d80aa8754b3e5c6a7221d9ccb76ac75494c2df43f778e74c1526b108c9",
  "plain_residual"               => "ee9ea58f56e4a7ab0ce01a96eee7b0b1ec2e0a5f3d8310659e7c6c2acd631a40",
  "modulated_gate_plus_residual" => "6884b4b9cc4aba0874ba0f9730bcb5f71b075385dcfb7617d330a839171dd988",
}

private def sparse_attention_consumers_fixture : JSON::Any
  JSON.parse(File.read(File.join(
    __DIR__,
    "../fixtures/trellis2/sparse_attention_consumers_cpu_v1.json"
  )))
end

private def sparse_attention_consumers_f32(
  payload : JSON::Any,
  output = [] of Float32,
) : Array(Float32)
  if array = payload.as_a?
    array.each do |entry|
      sparse_attention_consumers_f32(entry, output)
    end
  else
    output << payload.as_f.to_f32
  end
  output
end

private def sparse_attention_consumers_f32le_sha256(
  values : Indexable(Float32),
) : String
  bytes = Bytes.new(values.size * 4, 0_u8)
  values.each_with_index do |value, index|
    IO::ByteFormat::LittleEndian.encode(value, bytes[index * 4, 4])
  end
  Digest::SHA256.hexdigest(bytes)
end

describe "TRELLIS.2 immediate sparse self-attention consumers" do
  it "pins separate plain and modulated source variants" do
    fixture = sparse_attention_consumers_fixture
    fixture["schema"].as_s.should eq(
      "cogni-ml/trellis2/sparse-attention-consumers-oracle/v1"
    )
    provenance = fixture["provenance"]
    provenance["commit"].as_s.should eq(
      "75fbf0183001ed9876c8dbb35de6b68552ee08bd"
    )
    provenance["generator"].as_s.should eq(
      "tools/trellis2_oracle/export_sparse_attention_consumers.py"
    )
    provenance["python_version"].as_s.should eq("3.11.9")
    provenance["torch_version"].as_s.should eq("2.9.0")
    provenance["numpy_version"].as_s.should eq("2.1.3")
    provenance["network"].as_s.should eq("none")
    provenance["device"].as_s.should eq("cpu")
    provenance["weights"].as_s.should eq("synthetic")
    provenance["upstream_consumer_methods_executed"].as_bool.should be_true
    provenance["upstream_attention_backend_executed"].as_bool.should be_false
    provenance["upstream_checkpoint_path_executed"].as_bool.should be_false

    SPARSE_ATTENTION_CONSUMER_SOURCE_DIGESTS.each do |name, digest|
      provenance["sources"][name]["sha256"].as_s.should eq(digest)
    end

    inventory = fixture["inventory"].as_a
    inventory.map { |entry| entry["class"].as_s }.should eq([
      "SparseTransformerBlock",
      "SparseTransformerCrossBlock",
      "ModulatedSparseTransformerBlock",
      "ModulatedSparseTransformerCrossBlock",
    ])
    inventory.map { |entry| entry["consumer"].as_s }.should eq([
      "plain_residual",
      "plain_residual",
      "modulated_gate_plus_residual",
      "modulated_gate_plus_residual",
    ])
    inventory[0]["formula"].as_s.should eq("x + attention_output")
    inventory[1]["formula"].as_s.should eq("x + attention_output")
    inventory[2]["formula"].as_s.should eq(
      "x + attention_output * gate_msa[batch]"
    )
    inventory[3]["formula"].as_s.should eq(
      "x + attention_output * gate_msa[batch]"
    )
    inventory[3]["production_consumer"].as_bool.should be_true
    inventory[0]["production_consumer"].as_bool.should be_false
    inventory.map { |entry| entry["production_consumer"].as_bool }.should eq([
      false, false, false, true,
    ])

    production = fixture["production_configuration"]
    production["model"].as_s.should eq("ElasticSLatFlowModel")
    production["base_model"].as_s.should eq("SLatFlowModel")
    production["block"].as_s.should eq(
      "ModulatedSparseTransformerCrossBlock"
    )
    production["share_mod"].as_bool.should be_true
    production["config_count"].as_i.should eq(4)
    production["configs"].as_a.each do |config|
      config["share_mod"].as_bool.should be_true
      config["model"].as_s.should eq("ElasticSLatFlowModel")
      config["num_blocks"].as_i.should eq(30)
      SPARSE_ATTENTION_CONSUMER_CONFIG_DIGESTS[config["path"].as_s].should eq(
        config["sha256"].as_s
      )
    end
  end

  it "executes both consumers without conflating gate semantics" do
    fixture = sparse_attention_consumers_fixture
    contract = fixture["contract"]
    contract["sequence_lengths"].as_a.map(&.as_i).should eq([2_i64, 0_i64, 2_i64])
    contract["batch_broadcast_map"].as_a.map(&.as_i).should eq([
      0_i64, 0_i64, 2_i64, 2_i64,
    ])
    contract["share_mod_changes_gate_source_only"].as_bool.should be_true
    contract["input_unchanged"].as_bool.should be_true
    contract["direct_sparse_arithmetic_reuses_coordinate_object"].as_bool.should be_true

    stages = fixture["stages"]
    input = sparse_attention_consumers_f32(stages["residual_input"])
    attention = sparse_attention_consumers_f32(stages["attention_output"])
    gate = sparse_attention_consumers_f32(stages["gate_msa"])
    plain = sparse_attention_consumers_f32(stages["plain_residual"])
    modulated = sparse_attention_consumers_f32(
      stages["modulated_gate_plus_residual"]
    )
    SPARSE_ATTENTION_CONSUMER_STAGE_DIGESTS.each do |name, digest|
      values = sparse_attention_consumers_f32(stages[name])
      sparse_attention_consumers_f32le_sha256(values).should eq(digest)
      fixture["stage_f32le_sha256"][name].as_s.should eq(digest)
    end
    channels = fixture["contract"]["channels"].as_i.to_i32
    batch_map = contract["batch_broadcast_map"].as_a.map(&.as_i.to_i32)

    input.zip(attention).map { |(x, h)| x + h }.should eq(plain)
    expected_modulated = input.map_with_index do |x, index|
      row = index // channels
      channel = index % channels
      x + attention[index] * gate[batch_map[row] * channels + channel]
    end
    expected_modulated.should eq(modulated)
    modulated.should_not eq(plain)

    consumers = fixture["executions"]
    consumers["plain"].as_a.each do |execution|
      execution["output_f32le_sha256"].as_s.should eq(
        sparse_attention_consumers_f32le_sha256(plain)
      )
    end
    consumers["modulated"].as_a.each do |execution|
      execution["output_f32le_sha256"].as_s.should eq(
        sparse_attention_consumers_f32le_sha256(modulated)
      )
    end
    consumers["plain"].as_a.size.should eq(2)
    consumers["modulated"].as_a.size.should eq(4)

    gate.should contain(0.0_f32)
    gate.any? { |value| value < 0.0_f32 }.should be_true
    gate.any? { |value| value > 1.0_f32 }.should be_true
  end
end
