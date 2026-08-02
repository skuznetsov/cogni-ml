require "json"
require "digest/sha256"
require "../../src/ml/sparse/ops"
require "../spec_helper"

private def sparse_concat_fixture : JSON::Any
  JSON.parse(File.read(File.join(
    __DIR__,
    "../fixtures/trellis2/sparse_feature_concat_cpu_v1.json"
  )))
end

private def sparse_concat_i32(payload : JSON::Any) : Array(Int32)
  payload.as_a.flat_map do |row|
    row.as_a.map { |value| value.as_i.to_i32 }
  end
end

private def sparse_concat_f32(payload : JSON::Any) : Array(Float32)
  payload.as_a.flat_map do |row|
    row.as_a.map { |value| value.as_f.to_f32 }
  end
end

private def sparse_concat_f32le_sha256(values : Indexable(Float32)) : String
  bytes = Bytes.new(values.size * 4, 0_u8)
  values.each_with_index do |value, index|
    IO::ByteFormat::LittleEndian.encode(value, bytes[index * 4, 4])
  end
  Digest::SHA256.hexdigest(bytes)
end

class SparseConcatOverride < ML::Sparse::TensorCPU
  def class : ML::Sparse::TensorCPU.class
    ML::Sparse::TensorCPU
  end

  def feature(row : Int32, channel : Int32) : Float32
    Float32::NAN
  end
end

class SparseConcatMapOverride < ML::Sparse::CoordinateMap3D
  def class : ML::Sparse::CoordinateMap3D.class
    ML::Sparse::CoordinateMap3D
  end

  def same?(other : ML::Sparse::CoordinateMap3D) : Bool
    true
  end
end

class SparseConcatReceiverOverride < ML::Sparse::TensorCPU
  def self.==(other : ML::Sparse::TensorCPU.class) : Bool
    true
  end

  def initialize(
    features : Array(Float32),
    map : ML::Sparse::CoordinateMap3D,
    point_count : Int32,
    channels : Int32,
    budget : Int64,
  )
    features[0] = Float32::NAN unless features.empty?
    super(features, map, point_count, channels, budget)
  end
end

describe ML::Sparse::TensorCPU do
  it "matches the pinned two-value feature-concat oracle" do
    fixture = sparse_concat_fixture
    fixture["schema"].as_s.should eq(
      "cogni-ml/trellis2/sparse-feature-concat-oracle/v1"
    )
    provenance = fixture["provenance"]
    provenance["commit"].as_s.should eq(
      "75fbf0183001ed9876c8dbb35de6b68552ee08bd"
    )
    provenance["source_sha256"].as_s.should eq(
      "99dbcb7298238fdb6b6d47918ec068f618c68067653489d22b32c136c1ae0e78"
    )
    provenance["python_version"].as_s.should eq("3.11.9")
    provenance["torch_version"].as_s.should eq("2.9.0")
    provenance["numpy_version"].as_s.should eq("2.1.3")
    provenance["network"].as_s.should eq("none")
    provenance["device"].as_s.should eq("cpu")
    provenance["weights"].as_s.should eq("none")
    provenance["sparse_backend"].as_s.should eq("none")
    contract = fixture["contract"]
    contract["arity"].as_i.should eq(2)
    contract["axis"].as_s.should eq("feature_channels")
    contract["local_coordinate_policy"].as_s.should eq(
      "exact same immutable CoordinateMap3D object"
    )
    fixture["upstream_non_guarantee"]["mismatched_coordinate_concat_succeeded"].as_bool.should be_true

    input = fixture["input"]
    coordinates = sparse_concat_i32(input["coordinates"])
    left_values = sparse_concat_f32(input["left_features"])
    right_values = sparse_concat_f32(input["right_features"])
    expected = sparse_concat_f32(fixture["output"]["features"])
    map = ML::Sparse::CoordinateMap3D.new(coordinates, 3, {4, 3, 4})
    left_source = ML::Tensor.from_array(
      left_values,
      ML::Shape.new(6_i32, 2_i32)
    )
    right_source = ML::Tensor.from_array(
      right_values,
      ML::Shape.new(6_i32, 3_i32)
    )
    left = ML::Sparse::TensorCPU.new(left_source, map, 256_i64)
    right = ML::Sparse::TensorCPU.new(right_source, map, 128_i64)

    left_source.cpu_data.not_nil![0] = -1.0_f32
    right_source.cpu_data.not_nil![0] = -2.0_f32
    output = ML::Sparse::TensorCPU.concat_features(left, right)

    output.channels.should eq(5)
    output.point_count.should eq(6)
    output.coordinate_map.same?(map).should be_true
    output.max_feature_bytes.should eq(128_i64)
    output.features_copy.should eq(expected)
    sparse_concat_f32le_sha256(output.features_copy).should eq(
      fixture["output"]["features_f32le_sha256"].as_s
    )
    left.features_copy.should eq(left_values)
    right.features_copy.should eq(right_values)

    duplicated = ML::Sparse::TensorCPU.concat_features(left, left)
    duplicated.features_copy[0, 4].should eq(
      [1_000.0_f32, 1_001.0_f32, 1_000.0_f32, 1_001.0_f32]
    )

    copy = output.features_copy
    copy[0] = -3.0_f32
    output.feature(0, 0).should eq(1_000.0_f32)
  end

  it "accepts empty aligned values without widening the coordinate contract" do
    map = ML::Sparse::CoordinateMap3D.new([] of Int32, 2, {4, 4, 4})
    left = ML::Sparse::TensorCPU.new(
      ML::Tensor.from_array([] of Float32, ML::Shape.new(0_i32, 2_i32)),
      map
    )
    right = ML::Sparse::TensorCPU.new(
      ML::Tensor.from_array([] of Float32, ML::Shape.new(0_i32, 3_i32)),
      map
    )

    output = ML::Sparse::TensorCPU.concat_features(left, right)
    output.channels.should eq(5)
    output.features_copy.should be_empty
    output.coordinate_map.same?(map).should be_true
  end

  it "rejects equal or different reconstructed maps" do
    coordinates = [0, 0, 0, 0, 0, 1, 1, 1] of Int32
    left_map = ML::Sparse::CoordinateMap3D.new(coordinates, 1, {4, 4, 4})
    equal_map = ML::Sparse::CoordinateMap3D.new(coordinates, 1, {4, 4, 4})
    different_map = ML::Sparse::CoordinateMap3D.new(
      [0, 0, 0, 1, 0, 1, 1, 1] of Int32,
      1,
      {4, 4, 4}
    )
    left = ML::Sparse::TensorCPU.new(ML::Tensor.zeros(2, 1), left_map)
    equal = ML::Sparse::TensorCPU.new(ML::Tensor.zeros(2, 1), equal_map)
    different = ML::Sparse::TensorCPU.new(ML::Tensor.zeros(2, 1), different_map)

    expect_raises(ML::Sparse::SparseTensorError, /same immutable coordinate map/) do
      ML::Sparse::TensorCPU.concat_features(left, equal)
    end
    expect_raises(ML::Sparse::SparseTensorError, /same immutable coordinate map/) do
      ML::Sparse::TensorCPU.concat_features(left, different)
    end
  end

  it "fails closed when raw allocation bypasses initialization" do
    map = ML::Sparse::CoordinateMap3D.new([] of Int32, 1, {1, 1, 1})
    initialized = ML::Sparse::TensorCPU.new(
      ML::Tensor.from_array([] of Float32, ML::Shape.new(0_i32, 1_i32)),
      map
    )
    raw = ML::Sparse::TensorCPU.allocate

    expect_raises(ML::Sparse::SparseTensorError, /initialized sparse values/) do
      ML::Sparse::TensorCPU.concat_features(raw, initialized)
    end
    expect_raises(ML::Sparse::SparseTensorError, /initialized sparse values/) do
      ML::Sparse::TensorCPU.concat_features(initialized, raw)
    end

    populated_map = ML::Sparse::CoordinateMap3D.new(
      [0, 0, 0, 0] of Int32,
      1,
      {1, 1, 1}
    )
    base = ML::Sparse::TensorCPU.new(ML::Tensor.ones(1, 1), populated_map)
    overridden = SparseConcatOverride.new(ML::Tensor.ones(1, 1), populated_map)
    ML::Sparse::TensorCPU.concat_features(overridden, base).features_copy.should eq(
      [1.0_f32, 1.0_f32]
    )

    override_map = SparseConcatMapOverride.new([] of Int32, 1, {1, 1, 1})
    distinct_override_map = SparseConcatMapOverride.new([] of Int32, 1, {1, 1, 1})
    mapped_left = ML::Sparse::TensorCPU.new(
      ML::Tensor.from_array([] of Float32, ML::Shape.new(0_i32, 1_i32)),
      override_map
    )
    mapped_right = ML::Sparse::TensorCPU.new(
      ML::Tensor.from_array([] of Float32, ML::Shape.new(0_i32, 1_i32)),
      distinct_override_map
    )
    expect_raises(ML::Sparse::SparseTensorError, /same immutable coordinate map/) do
      ML::Sparse::TensorCPU.concat_features(mapped_left, mapped_right)
    end
  end

  it "rejects inherited subclass receivers before constructing output" do
    map = ML::Sparse::CoordinateMap3D.new(
      [0, 0, 0, 0] of Int32,
      1,
      {1, 1, 1}
    )
    value = ML::Sparse::TensorCPU.new(ML::Tensor.ones(1, 1), map)

    expect_raises(ML::Sparse::SparseTensorError, /base TensorCPU receiver/) do
      SparseConcatReceiverOverride.concat_features(value, value)
    end
  end

  it "fails closed on output channel and byte budgets" do
    empty_map = ML::Sparse::CoordinateMap3D.new([] of Int32, 1, {1, 1, 1})
    left_wide = ML::Sparse::TensorCPU.new(
      ML::Tensor.from_array([] of Float32, ML::Shape.new(0_i32, 128_i32)),
      empty_map
    )
    right_wide = ML::Sparse::TensorCPU.new(
      ML::Tensor.from_array([] of Float32, ML::Shape.new(0_i32, 129_i32)),
      empty_map
    )
    exact_wide = ML::Sparse::TensorCPU.new(
      ML::Tensor.from_array([] of Float32, ML::Shape.new(0_i32, 128_i32)),
      empty_map
    )
    ML::Sparse::TensorCPU.concat_features(left_wide, exact_wide).channels.should eq(256)
    expect_raises(ML::Sparse::SparseTensorBudgetError, /channel count 257/) do
      ML::Sparse::TensorCPU.concat_features(left_wide, right_wide)
    end

    map = ML::Sparse::CoordinateMap3D.new(
      [0, 0, 0, 0, 0, 1, 1, 1] of Int32,
      1,
      {4, 4, 4}
    )
    left = ML::Sparse::TensorCPU.new(ML::Tensor.zeros(2, 1), map, 8_i64)
    right = ML::Sparse::TensorCPU.new(ML::Tensor.zeros(2, 1), map, 16_i64)
    exact_left = ML::Sparse::TensorCPU.new(ML::Tensor.zeros(2, 1), map, 16_i64)
    ML::Sparse::TensorCPU.concat_features(exact_left, right).max_feature_bytes.should eq(16_i64)
    expect_raises(ML::Sparse::SparseTensorBudgetError, /require 16 bytes, limit is 8/) do
      ML::Sparse::TensorCPU.concat_features(left, right)
    end
  end
end
