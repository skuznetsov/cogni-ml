require "spec"
require "../src/ml/audio/drama3/backbone"

include ML::Audio::Drama3

describe "DRAMA-3 pinned MLX backbone parity" do
  it "matches all residual boundaries over three cached rows" do
    path = File.join(__DIR__, "support/drama3/reference_fixture")
    expected = {} of String => Vector
    JSON.parse(File.read(File.join(path, "reference_traces.json")))["traces"].as_a.each do |record|
      name = record["name"].as_s
      expected[name] = record["values"].as_a.map { |v| v.as_f.to_f32 } if name.starts_with?("backbone.")
    end
    model = Backbone.new(BackboneConfig.new(JSON.parse(File.read(File.join(path, "config.json")))), WeightStore.new(path))
    row = 0
    observed = [] of String
    model.trace = ->(name : String, values : Vector) {
      key = name.sub("backbone.", "backbone.row#{row}.")
      reference = expected[key]
      values.size.should eq(reference.size)
      values.zip(reference).each { |a, b| a.should be_close(b, 1e-5) }
      observed << key
      nil
    }
    inputs = JSON.parse(File.read(File.join(path, "inputs.json")))["input_embeddings"].as_a
    inputs.each_with_index do |input, index|
      row = index
      model.step(input.as_a.map { |v| v.as_f.to_f32 })
    end
    observed.sort.should eq(expected.keys.sort)
    model.position.should eq(3)
    model.states[0].conv.size.should eq(3)
    model.states[3].keys.size.should eq(3)
  end

  it "unpacks affine Q4 and Q8 in the same lane order as MLX" do
    path = File.join(__DIR__, "support/drama3/reference_fixture")
    expected = JSON.parse(File.read(File.join(path, "quantization_expected.json"))).as_h
    store = WeightStore.new(path)
    expected.each do |name, fixture|
      matrix = store.matrix(name + ".weight")
      dense = fixture["dense"].as_a.map { |r| r.as_a.map { |v| v.as_f.to_f32 } }
      dense.each_with_index do |reference, index|
        matrix.row(index).zip(reference).each { |a, b| a.should be_close(b, 1e-7) }
      end
      input = fixture["input"].as_a.map { |v| v.as_f.to_f32 }
      reference = fixture["matvec"].as_a.map { |v| v.as_f.to_f32 }
      matrix.matvec(input).zip(reference).each { |a, b| a.should be_close(b, 2e-6) }
      expect_raises(ArgumentError) { matrix.values }
    end
  end
end
