require "json"
require "digest/sha256"
require "../../src/ml/sparse"
require "../spec_helper"

private def sparse_plain_residual_fixture : JSON::Any
  JSON.parse(File.read(File.join(
    __DIR__,
    "../fixtures/trellis2/sparse_cross_attention_seam_cpu_v1.json"
  )))
end

private def sparse_plain_residual_i32(payload : JSON::Any) : Array(Int32)
  payload.as_a.flat_map do |row|
    row.as_a.map { |value| value.as_i.to_i32 }
  end
end

private def sparse_plain_residual_f32(
  payload : JSON::Any,
  output = [] of Float32,
) : Array(Float32)
  if hash = payload.as_h?
    if values = hash["values"]?
      return sparse_plain_residual_f32(values, output)
    end
  elsif array = payload.as_a?
    array.each { |entry| sparse_plain_residual_f32(entry, output) }
  else
    output << payload.as_f.to_f32
  end
  output
end

private def sparse_plain_residual_f32le_sha256(
  values : Indexable(Float32),
) : String
  bytes = Bytes.new(values.size * 4, 0_u8)
  values.each_with_index do |value, index|
    IO::ByteFormat::LittleEndian.encode(value, bytes[index * 4, 4])
  end
  Digest::SHA256.hexdigest(bytes)
end

private def sparse_plain_residual_tensor(
  values : Array(Float32),
  rows : Int32,
  channels : Int32,
) : ML::Tensor
  ML::Tensor.from_array(values, ML::Shape.new(rows, channels))
end

class SparsePlainResidualReceiverOverride < ML::Sparse::TensorCPU
  def self.==(other : ML::Sparse::TensorCPU.class) : Bool
    true
  end
end

class SparsePlainResidualInputOverride < ML::Sparse::TensorCPU
  def initialize(
    features : Array(Float32),
    map : ML::Sparse::CoordinateMap3D,
    point_count : Int32,
    channels : Int32,
    budget : Int64,
  )
    super(features, map, point_count, channels, budget)
  end
end

describe "TRELLIS.2 bounded sparse plain residual" do
  it "matches the pinned cross-attention second residual exactly" do
    fixture = sparse_plain_residual_fixture
    input = fixture["input"]
    stages = fixture["stages"]
    channels = input["channels"].as_i.to_i32
    coordinates = sparse_plain_residual_i32(input["coordinates"])
    base_values = sparse_plain_residual_f32(stages["after_self"])
    update_values = sparse_plain_residual_f32(
      stages["cross_attention_output"]
    )
    expected = sparse_plain_residual_f32(stages["after_cross"])

    fixture["provenance"]["commit"].as_s.should eq(
      "75fbf0183001ed9876c8dbb35de6b68552ee08bd"
    )
    fixture["stage_contract"]["after_cross"]["formula"].as_s.should eq(
      "after_self + cross_attention_output"
    )
    sparse_plain_residual_f32le_sha256(expected).should eq(
      "f04bd7c46104b2072b38cba375bf9fb084961974ebda067b5b87fc1eeb0c8249"
    )

    map = ML::Sparse::CoordinateMap3D.new(
      coordinates,
      input["batch_size"].as_i.to_i32,
      {
        input["spatial_shape"][0].as_i.to_i32,
        input["spatial_shape"][1].as_i.to_i32,
        input["spatial_shape"][2].as_i.to_i32,
      }
    )
    base = ML::Sparse::TensorCPU.new(
      sparse_plain_residual_tensor(base_values, 4, channels),
      map,
      512_i64
    )
    update = ML::Sparse::TensorCPU.new(
      sparse_plain_residual_tensor(update_values, 4, channels),
      map,
      256_i64
    )
    base_before = base.features_copy
    update_before = update.features_copy

    output = ML::Sparse::TensorCPU.apply_residual(base, update)

    output.point_count.should eq(4)
    output.channels.should eq(channels)
    output.coordinate_map.same?(map).should be_true
    output.max_feature_bytes.should eq(256_i64)
    output.features_copy.should eq(expected)
    sparse_plain_residual_f32le_sha256(output.features_copy).should eq(
      "f04bd7c46104b2072b38cba375bf9fb084961974ebda067b5b87fc1eeb0c8249"
    )
    base.features_copy.should eq(base_before)
    update.features_copy.should eq(update_before)

    copy = output.features_copy
    copy[0] = -99.0_f32
    output.feature(0, 0).should eq(expected[0])
  end

  it "preserves empty and production-width standard carriers" do
    empty_map = ML::Sparse::CoordinateMap3D.new([] of Int32, 3, {1, 1, 1})
    empty_base = ML::Sparse::TensorCPU.new(
      sparse_plain_residual_tensor([] of Float32, 0, 2),
      empty_map,
      16_i64
    )
    empty_update = ML::Sparse::TensorCPU.new(
      sparse_plain_residual_tensor([] of Float32, 0, 2),
      empty_map,
      8_i64
    )
    empty_output = ML::Sparse::TensorCPU.apply_residual(
      empty_base,
      empty_update
    )
    empty_output.features_copy.should be_empty
    empty_output.coordinate_map.same?(empty_map).should be_true
    empty_output.max_feature_bytes.should eq(8_i64)

    production_map = ML::Sparse::CoordinateMap3D.new(
      [0, 0, 0, 0] of Int32,
      1,
      {1, 1, 1}
    )
    production_base = ML::Sparse::TensorCPU.production(
      ML::Tensor.ones(1, 257),
      production_map,
      2_056_i64
    )
    production_update = ML::Sparse::TensorCPU.production(
      ML::Tensor.ones(1, 257),
      production_map,
      1_028_i64
    )
    production_output = ML::Sparse::TensorCPU.apply_residual(
      production_base,
      production_update
    )
    production_output.production_width?.should be_true
    production_output.channels.should eq(257)
    production_output.max_feature_bytes.should eq(1_028_i64)
    production_output.features_copy.should eq(Array.new(257, 2.0_f32))
  end

  it "rejects different map identities, carrier roles, and shapes" do
    coordinates = [0, 0, 0, 0] of Int32
    map = ML::Sparse::CoordinateMap3D.new(coordinates, 1, {1, 1, 1})
    equal_map = ML::Sparse::CoordinateMap3D.new(
      coordinates,
      1,
      {1, 1, 1}
    )
    base = ML::Sparse::TensorCPU.new(ML::Tensor.ones(1, 1), map)

    expect_raises(
      ML::Sparse::SparseTensorError,
      /same immutable coordinate map/
    ) do
      ML::Sparse::TensorCPU.apply_residual(
        base,
        ML::Sparse::TensorCPU.new(ML::Tensor.ones(1, 1), equal_map)
      )
    end

    production = ML::Sparse::TensorCPU.production(
      ML::Tensor.ones(1, 1),
      map
    )
    expect_raises(ML::Sparse::SparseTensorError, /same carrier role/) do
      ML::Sparse::TensorCPU.apply_residual(base, production)
    end

    wider = ML::Sparse::TensorCPU.new(ML::Tensor.ones(1, 2), map)
    expect_raises(ML::Sparse::SparseTensorError, /same row and channel shape/) do
      ML::Sparse::TensorCPU.apply_residual(base, wider)
    end
  end

  it "rejects uninitialized and malformed sparse values" do
    map = ML::Sparse::CoordinateMap3D.new(
      [0, 0, 0, 0, 0, 1, 0, 0] of Int32,
      1,
      {2, 1, 1}
    )
    valid = ML::Sparse::TensorCPU.new(ML::Tensor.ones(2, 1), map)
    raw = ML::Sparse::TensorCPU.allocate

    expect_raises(ML::Sparse::SparseTensorError, /initialized sparse value/) do
      ML::Sparse::TensorCPU.apply_residual(raw, valid)
    end
    expect_raises(ML::Sparse::SparseTensorError, /initialized sparse value/) do
      ML::Sparse::TensorCPU.apply_residual(valid, raw)
    end

    wrong_points = SparsePlainResidualInputOverride.new(
      [1.0_f32],
      map,
      1_i32,
      1_i32,
      8_i64
    )
    expect_raises(
      ML::Sparse::SparseTensorError,
      /coordinate and feature point counts/
    ) do
      ML::Sparse::TensorCPU.apply_residual(wrong_points, wrong_points)
    end

    bad_storage = SparsePlainResidualInputOverride.new(
      [1.0_f32],
      map,
      2_i32,
      1_i32,
      8_i64
    )
    expect_raises(ML::Sparse::SparseTensorError, /feature storage size/) do
      ML::Sparse::TensorCPU.apply_residual(bad_storage, bad_storage)
    end
  end

  it "checks the inherited byte cap before arithmetic and rejects overflow" do
    map = ML::Sparse::CoordinateMap3D.new(
      [0, 0, 0, 0] of Int32,
      1,
      {1, 1, 1}
    )
    insufficient = SparsePlainResidualInputOverride.new(
      [Float32::NAN, Float32::NAN],
      map,
      1_i32,
      2_i32,
      7_i64
    )
    wider = SparsePlainResidualInputOverride.new(
      [1.0_f32, 1.0_f32],
      map,
      1_i32,
      2_i32,
      8_i64
    )
    expect_raises(
      ML::Sparse::SparseTensorBudgetError,
      /require 8 bytes.*limit is 7/
    ) do
      ML::Sparse::TensorCPU.apply_residual(insufficient, wider)
    end

    {Float32::NAN, Float32::INFINITY, -Float32::INFINITY}.each do |invalid|
      poisoned = SparsePlainResidualInputOverride.new(
        [invalid, 1.0_f32],
        map,
        1_i32,
        2_i32,
        8_i64
      )
      expect_raises(
        ML::Sparse::SparseTensorError,
        /output\[0\] must be finite/
      ) do
        ML::Sparse::TensorCPU.apply_residual(poisoned, wider)
      end
    end

    max_value = ML::Sparse::TensorCPU.new(
      sparse_plain_residual_tensor([Float32::MAX], 1, 1),
      map
    )
    expect_raises(ML::Sparse::SparseTensorError, /output\[0\] must be finite/) do
      ML::Sparse::TensorCPU.apply_residual(max_value, max_value)
    end
  end

  it "requires the base TensorCPU receiver" do
    map = ML::Sparse::CoordinateMap3D.new(
      [0, 0, 0, 0] of Int32,
      1,
      {1, 1, 1}
    )
    value = ML::Sparse::TensorCPU.new(ML::Tensor.ones(1, 1), map)

    expect_raises(ML::Sparse::SparseTensorError, /base TensorCPU receiver/) do
      SparsePlainResidualReceiverOverride.apply_residual(value, value)
    end
  end
end
