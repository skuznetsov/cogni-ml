require "spec"
require "../examples/support/drama3_decode_replay_coverage"

private def drama3_decode_replay_fixture_names(frame : Int32) : Array(String)
  names = [] of String
  40.times do |layer|
    prefix = "backbone.layers.#{layer}.mlp"
    names << "#{prefix}.gate.weight"
    names << "#{prefix}.shared_expert_gate.weight"
    {"shared_expert", "switch_mlp"}.each do |expert|
      {"gate_proj", "up_proj", "down_proj"}.each do |projection|
        names << "#{prefix}.#{expert}.#{projection}.weight"
      end
    end

    if layer % 4 == 3
      {"q_proj", "k_proj", "v_proj", "o_proj"}.each do |projection|
        names << "backbone.layers.#{layer}.self_attn.#{projection}.weight"
      end
    else
      {"in_proj_qkv", "in_proj_z", "in_proj_b", "in_proj_a", "out_proj"}.each do |projection|
        names << "backbone.layers.#{layer}.linear_attn.#{projection}.weight"
      end
    end
  end

  {"shared_layers" => 4, "residual_layers" => 9}.each do |kind, count|
    count.times do |layer|
      prefix = "tower.#{kind}.#{layer}"
      {"eh_proj", "transformer_layer.attention.wqkv", "transformer_layer.attention.wo",
       "transformer_layer.feed_forward.w1", "transformer_layer.feed_forward.w3",
       "transformer_layer.feed_forward.w2"}.each do |projection|
        names << "#{prefix}.#{projection}.weight"
      end
    end
  end
  9.times { |head| names << "tower.codebook_heads.#{head}.weight" }
  names << "lm_head.weight" if frame == 29 || frame == 59
  names
end

private def drama3_decode_replay_record(frame : Int32, name : String) : JSON::Any
  JSON.parse({frame: frame, name: name}.to_json)
end

private def drama3_decode_replay_fixture : Array(JSON::Any)
  [0, 29, 59].flat_map do |frame|
    drama3_decode_replay_fixture_names(frame).map do |name|
      drama3_decode_replay_record(frame, name)
    end
  end
end

describe Drama3DecodeReplay do
  it "accepts the exact selected-frame projection inventory" do
    records = drama3_decode_replay_fixture
    Drama3DecodeReplay.assert_coverage!(records).should be_nil
    records.count { |record| record["frame"].as_i == 0 }.should eq(597)
    records.count { |record| record["frame"].as_i == 29 }.should eq(598)
    records.count { |record| record["frame"].as_i == 59 }.should eq(598)
  end

  it "rejects a missing projection even when all other frames are complete" do
    records = drama3_decode_replay_fixture
    missing = "backbone.layers.39.mlp.switch_mlp.down_proj.weight"
    records.reject! { |record| record["frame"].as_i == 0 && record["name"].as_s == missing }

    expect_raises(ArgumentError, /missing projection at frame 0/) do
      Drama3DecodeReplay.assert_coverage!(records)
    end
  end

  it "rejects a duplicate replacing another name without a count change" do
    records = drama3_decode_replay_fixture
    index = records.index do |record|
      record["frame"].as_i == 0 && record["name"].as_s == "backbone.layers.0.linear_attn.in_proj_z.weight"
    end || raise "fixture projection missing"
    records[index] = drama3_decode_replay_record(0, "backbone.layers.0.mlp.gate.weight")

    expect_raises(ArgumentError, /duplicate projection at frame 0/) do
      Drama3DecodeReplay.assert_coverage!(records)
    end
  end

  it "rejects an unexpected selected-frame index" do
    records = drama3_decode_replay_fixture
    index = records.index { |record| record["frame"].as_i == 29 } || raise "fixture frame missing"
    name = records[index]["name"].as_s
    records[index] = drama3_decode_replay_record(30, name)

    expect_raises(ArgumentError, /unexpected capture frame 30/) do
      Drama3DecodeReplay.assert_coverage!(records)
    end
  end

  it "rejects an unknown layer or projection name" do
    records = drama3_decode_replay_fixture
    records << drama3_decode_replay_record(59, "backbone.layers.40.mlp.gate.weight")

    expect_raises(ArgumentError, /unexpected projection at frame 59/) do
      Drama3DecodeReplay.assert_coverage!(records)
    end
  end

  it "rejects a wholly missing selected frame" do
    records = drama3_decode_replay_fixture.reject { |record| record["frame"].as_i == 29 }

    expect_raises(ArgumentError, /missing capture frame 29/) do
      Drama3DecodeReplay.assert_coverage!(records)
    end
  end

  it "rejects mixed kernels or reductions in one timed family" do
    base = JSON.parse({frame: 29, family: "backbone.linear_attention.qkv",
                       kernel: "dr3_dense", reduction: "tree", dtype: "BF16",
                       bits: 0, group_size: 64, threadgroup_width: 64}.to_json)
    Drama3DecodeReplay.assert_homogeneous!([base, base]).should be_nil
    {"kernel" => "dr3_dense_simd", "reduction" => "simd"}.each do |key, value|
      changed = JSON.parse(base.to_json)
      changed.as_h[key] = JSON::Any.new(value)
      expect_raises(ArgumentError, /heterogeneous replay group/) do
        Drama3DecodeReplay.assert_homogeneous!([base, changed])
      end
    end
  end
end
