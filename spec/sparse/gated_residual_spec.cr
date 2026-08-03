require "json"
require "digest/sha256"
require "../../src/ml/sparse"
require "../spec_helper"

private def sparse_gated_residual_fixture : JSON::Any
  JSON.parse(File.read(File.join(
    __DIR__,
    "../fixtures/trellis2/sparse_attention_consumers_cpu_v1.json"
  )))
end

private def sparse_gated_residual_i32(payload : JSON::Any) : Array(Int32)
  payload.as_a.flat_map do |row|
    row.as_a.map { |value| value.as_i.to_i32 }
  end
end

private def sparse_gated_residual_f32(payload : JSON::Any) : Array(Float32)
  payload.as_a.flat_map do |row|
    row.as_a.map { |value| value.as_f.to_f32 }
  end
end

private def sparse_gated_residual_f32le_sha256(
  values : Indexable(Float32),
) : String
  bytes = Bytes.new(values.size * 4, 0_u8)
  values.each_with_index do |value, index|
    IO::ByteFormat::LittleEndian.encode(value, bytes[index * 4, 4])
  end
  Digest::SHA256.hexdigest(bytes)
end

private def sparse_gated_residual_tensor(
  values : Array(Float32),
  rows : Int32,
  channels : Int32,
) : ML::Tensor
  ML::Tensor.from_array(values, ML::Shape.new(rows, channels))
end

class SparseGatedResidualReceiverOverride < ML::Sparse::TensorCPU
  def self.==(other : ML::Sparse::TensorCPU.class) : Bool
    true
  end
end

class SparseGatedResidualInputOverride < ML::Sparse::TensorCPU
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

class SparseGatedResidualGateOverride < ML::Tensor
  def initialize(values : Array(Float32))
    shape = ML::Shape.new(1_i32, 1_i32)
    super(
      shape,
      ML::Strides.new(shape),
      ML::DType::F32,
      ML::Tensor::Device::CPU,
      nil,
      values
    )
  end

  def on_cpu? : Bool
    false
  end

  def contiguous? : Bool
    false
  end

  def shape : ML::Shape
    ML::Shape.new(9_i32, 9_i32)
  end

  def cpu_read : ML::Tensor::CPUReadView
    fake = ML::Tensor.from_array([7.0_f32], ML::Shape.new(1_i32, 1_i32))
    ML::Tensor::CPUReadView.new(fake, fake.cpu_data.not_nil!)
  end
end

describe "TRELLIS.2 bounded sparse gated residual" do
  it "matches the pinned production-consumer oracle" do
    fixture = sparse_gated_residual_fixture
    input = fixture["input"]
    contract = fixture["contract"]
    stages = fixture["stages"]
    channels = contract["channels"].as_i.to_i32
    coordinates = sparse_gated_residual_i32(input["coordinates"])
    residual_values = sparse_gated_residual_f32(stages["residual_input"])
    attention_values = sparse_gated_residual_f32(stages["attention_output"])
    gate_values = sparse_gated_residual_f32(stages["gate_msa"])
    expected = sparse_gated_residual_f32(
      stages["modulated_gate_plus_residual"]
    )
    plain = sparse_gated_residual_f32(stages["plain_residual"])

    fixture["schema"].as_s.should eq(
      "cogni-ml/trellis2/sparse-attention-consumers-oracle/v1"
    )
    fixture["provenance"]["commit"].as_s.should eq(
      "75fbf0183001ed9876c8dbb35de6b68552ee08bd"
    )
    contract["sequence_lengths"].as_a.map(&.as_i).should eq([
      2_i64, 0_i64, 2_i64,
    ])
    contract["batch_broadcast_map"].as_a.map(&.as_i).should eq([
      0_i64, 0_i64, 2_i64, 2_i64,
    ])
    sparse_gated_residual_f32le_sha256(residual_values).should eq(
      "ebd1d95a72c83eb3d50ac0ecaa1112f78a172df0573ea7dc0b1783ea41c3a3c8"
    )
    sparse_gated_residual_f32le_sha256(attention_values).should eq(
      "038cbf2423db1f7da606c7e91709b1feadc5536c4cd646a5e2faae353a42e5d5"
    )
    sparse_gated_residual_f32le_sha256(gate_values).should eq(
      "c9eb25d80aa8754b3e5c6a7221d9ccb76ac75494c2df43f778e74c1526b108c9"
    )
    sparse_gated_residual_f32le_sha256(plain).should eq(
      "ee9ea58f56e4a7ab0ce01a96eee7b0b1ec2e0a5f3d8310659e7c6c2acd631a40"
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
    residual = ML::Sparse::TensorCPU.new(
      sparse_gated_residual_tensor(residual_values, 4, channels),
      map,
      128_i64
    )
    attention = ML::Sparse::TensorCPU.new(
      sparse_gated_residual_tensor(attention_values, 4, channels),
      map,
      96_i64
    )
    gate = sparse_gated_residual_tensor(gate_values, 3, channels)

    output = ML::Sparse::TensorCPU.apply_gated_residual(
      residual,
      attention,
      gate
    )

    output.point_count.should eq(4)
    output.channels.should eq(channels)
    output.coordinate_map.same?(map).should be_true
    output.max_feature_bytes.should eq(96_i64)
    output.features_copy.should eq(expected)
    output.features_copy.should_not eq(plain)
    sparse_gated_residual_f32le_sha256(output.features_copy).should eq(
      "6884b4b9cc4aba0874ba0f9730bcb5f71b075385dcfb7617d330a839171dd988"
    )
    residual.features_copy.should eq(residual_values)
    attention.features_copy.should eq(attention_values)

    gate.cpu_data.not_nil![0] = 99.0_f32
    output.features_copy.should eq(expected)
    copy = output.features_copy
    copy[0] = -99.0_f32
    output.feature(0, 0).should eq(expected[0])
  end

  it "uses coordinate batches across middle, trailing, and all-empty slices" do
    middle_map = ML::Sparse::CoordinateMap3D.new(
      [0, 1, 0, 0, 2, 0, 1, 0] of Int32,
      3,
      {2, 2, 1}
    )
    residual = ML::Sparse::TensorCPU.new(
      sparse_gated_residual_tensor([10.0_f32, 20.0_f32], 2, 1),
      middle_map
    )
    update = ML::Sparse::TensorCPU.new(
      sparse_gated_residual_tensor([2.0_f32, 3.0_f32], 2, 1),
      middle_map
    )
    gate = sparse_gated_residual_tensor(
      [0.0_f32, 100.0_f32, -2.0_f32],
      3,
      1
    )
    ML::Sparse::TensorCPU.apply_gated_residual(
      residual,
      update,
      gate
    ).features_copy.should eq([10.0_f32, 14.0_f32])

    trailing_map = ML::Sparse::CoordinateMap3D.new(
      [0, 0, 0, 0, 0, 1, 0, 0] of Int32,
      3,
      {2, 1, 1}
    )
    trailing_residual = ML::Sparse::TensorCPU.new(
      sparse_gated_residual_tensor([1.0_f32, 2.0_f32], 2, 1),
      trailing_map
    )
    trailing_update = ML::Sparse::TensorCPU.new(
      sparse_gated_residual_tensor([3.0_f32, 4.0_f32], 2, 1),
      trailing_map
    )
    ML::Sparse::TensorCPU.apply_gated_residual(
      trailing_residual,
      trailing_update,
      sparse_gated_residual_tensor([2.0_f32, 9.0_f32, -4.0_f32], 3, 1)
    ).features_copy.should eq([7.0_f32, 10.0_f32])

    empty_map = ML::Sparse::CoordinateMap3D.new([] of Int32, 3, {1, 1, 1})
    empty = ML::Sparse::TensorCPU.new(
      sparse_gated_residual_tensor([] of Float32, 0, 2),
      empty_map
    )
    empty_output = ML::Sparse::TensorCPU.apply_gated_residual(
      empty,
      empty,
      sparse_gated_residual_tensor(Array(Float32).new(6, 0.0_f32), 3, 2)
    )
    empty_output.features_copy.should be_empty
    empty_output.coordinate_map.same?(empty_map).should be_true
  end

  it "rejects equal reconstructed and different coordinate maps" do
    coordinates = [0, 0, 0, 0] of Int32
    map = ML::Sparse::CoordinateMap3D.new(coordinates, 1, {1, 1, 1})
    equal_map = ML::Sparse::CoordinateMap3D.new(coordinates, 1, {1, 1, 1})
    different_map = ML::Sparse::CoordinateMap3D.new(
      [0, 0, 0, 1] of Int32,
      1,
      {1, 1, 2}
    )
    residual = ML::Sparse::TensorCPU.new(ML::Tensor.ones(1, 1), map)
    gate = ML::Tensor.ones(1, 1)

    {equal_map, different_map}.each do |other_map|
      update = ML::Sparse::TensorCPU.new(ML::Tensor.ones(1, 1), other_map)
      expect_raises(
        ML::Sparse::SparseTensorError,
        /same immutable coordinate map/
      ) do
        ML::Sparse::TensorCPU.apply_gated_residual(residual, update, gate)
      end
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
    gate = ML::Tensor.ones(1, 1)

    expect_raises(ML::Sparse::SparseTensorError, /initialized sparse values/) do
      ML::Sparse::TensorCPU.apply_gated_residual(raw, valid, gate)
    end
    expect_raises(ML::Sparse::SparseTensorError, /initialized sparse values/) do
      ML::Sparse::TensorCPU.apply_gated_residual(valid, raw, gate)
    end

    wrong_channels = ML::Sparse::TensorCPU.new(ML::Tensor.ones(2, 2), map)
    expect_raises(ML::Sparse::SparseTensorError, /same row and channel shape/) do
      ML::Sparse::TensorCPU.apply_gated_residual(
        valid,
        wrong_channels,
        gate
      )
    end

    wrong_points = SparseGatedResidualInputOverride.new(
      [1.0_f32],
      map,
      1_i32,
      1_i32,
      8_i64
    )
    expect_raises(ML::Sparse::SparseTensorError, /coordinate and feature point counts/) do
      ML::Sparse::TensorCPU.apply_gated_residual(
        wrong_points,
        wrong_points,
        gate
      )
    end

    bad_storage = SparseGatedResidualInputOverride.new(
      [1.0_f32],
      map,
      2_i32,
      1_i32,
      8_i64
    )
    expect_raises(ML::Sparse::SparseTensorError, /feature storage size/) do
      ML::Sparse::TensorCPU.apply_gated_residual(
        bad_storage,
        bad_storage,
        gate
      )
    end
  end

  it "rejects gate shape, layout, and non-finite values including empty batches" do
    map = ML::Sparse::CoordinateMap3D.new(
      [0, 0, 0, 0, 2, 0, 0, 0] of Int32,
      3,
      {1, 1, 1}
    )
    value = ML::Sparse::TensorCPU.new(ML::Tensor.ones(2, 2), map)

    {
      ML::Tensor.ones(6),
      ML::Tensor.ones(2, 2),
      ML::Tensor.ones(3, 1),
    }.each do |gate|
      expect_raises(ML::Sparse::SparseTensorError, /gate_msa shape.*\[3, 2\]/) do
        ML::Sparse::TensorCPU.apply_gated_residual(value, value, gate)
      end
    end

    noncontiguous = ML::Tensor.ones(2, 3).transpose
    expect_raises(ML::Sparse::SparseTensorError, /gate_msa.*contiguous/) do
      ML::Sparse::TensorCPU.apply_gated_residual(
        value,
        value,
        noncontiguous
      )
    end

    {Float32::NAN, Float32::INFINITY, -Float32::INFINITY}.each do |invalid|
      gate_values = Array(Float32).new(6, 1.0_f32)
      gate_values[2] = invalid # Batch 1 is empty but remains part of [B, C].
      gate = sparse_gated_residual_tensor(gate_values, 3, 2)
      expect_raises(ML::Sparse::SparseTensorError, /gate_msa\[2\] must be finite/) do
        ML::Sparse::TensorCPU.apply_gated_residual(value, value, gate)
      end
    end
  end

  it "uses canonical gate storage instead of virtual Tensor overrides" do
    map = ML::Sparse::CoordinateMap3D.new(
      [0, 0, 0, 0] of Int32,
      1,
      {1, 1, 1}
    )
    value = ML::Sparse::TensorCPU.new(ML::Tensor.ones(1, 1), map)
    gate = SparseGatedResidualGateOverride.new([0.0_f32])

    ML::Sparse::TensorCPU.apply_gated_residual(
      value,
      value,
      gate
    ).features_copy.should eq([1.0_f32])

    raw_gate = ML::Tensor.allocate
    expect_raises(ML::Sparse::SparseTensorError, /gate_msa must be on CPU/) do
      ML::Sparse::TensorCPU.apply_gated_residual(
        value,
        value,
        raw_gate
      )
    end
  end

  it "checks the inherited byte cap before arithmetic and rejects overflow" do
    map = ML::Sparse::CoordinateMap3D.new(
      [0, 0, 0, 0, 0, 1, 0, 0] of Int32,
      1,
      {2, 1, 1}
    )
    exact = ML::Sparse::TensorCPU.new(ML::Tensor.ones(2, 1), map, 8_i64)
    wider = ML::Sparse::TensorCPU.new(ML::Tensor.ones(2, 1), map, 16_i64)
    output = ML::Sparse::TensorCPU.apply_gated_residual(
      exact,
      wider,
      ML::Tensor.ones(1, 1)
    )
    output.max_feature_bytes.should eq(8_i64)

    insufficient = SparseGatedResidualInputOverride.new(
      [1.0_f32, 1.0_f32],
      map,
      2_i32,
      1_i32,
      7_i64
    )
    invalid_gate = ML::Tensor.from_array(
      [Float32::NAN],
      ML::Shape.new(1_i32, 1_i32)
    )
    expect_raises(
      ML::Sparse::SparseTensorBudgetError,
      /require 8 bytes.*limit is 7/
    ) do
      ML::Sparse::TensorCPU.apply_gated_residual(
        insufficient,
        wider,
        invalid_gate
      )
    end

    one_map = ML::Sparse::CoordinateMap3D.new(
      [0, 0, 0, 0] of Int32,
      1,
      {1, 1, 1}
    )
    residual = ML::Sparse::TensorCPU.new(
      sparse_gated_residual_tensor([0.0_f32], 1, 1),
      one_map
    )
    update = ML::Sparse::TensorCPU.new(
      sparse_gated_residual_tensor([Float32::MAX], 1, 1),
      one_map
    )
    expect_raises(ML::Sparse::SparseTensorError, /output\[0\] must be finite/) do
      ML::Sparse::TensorCPU.apply_gated_residual(
        residual,
        update,
        sparse_gated_residual_tensor([2.0_f32], 1, 1)
      )
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
      SparseGatedResidualReceiverOverride.apply_gated_residual(
        value,
        value,
        ML::Tensor.ones(1, 1)
      )
    end
  end
end
