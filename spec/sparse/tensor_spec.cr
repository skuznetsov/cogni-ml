require "json"
require "digest/sha256"
require "../../src/ml/sparse/tensor"
require "../spec_helper"

private def sparse_tensor_fixture : JSON::Any
  JSON.parse(File.read(File.join(
    __DIR__,
    "../fixtures/trellis2/sparse_tensor_cpu_v1.json"
  )))
end

private def sparse_tensor_coordinates(payload : JSON::Any) : Array(Int32)
  payload.as_a.flat_map do |row|
    row.as_a.map { |value| value.as_i.to_i32 }
  end
end

private def sparse_tensor_features(payload : JSON::Any) : Array(Float32)
  payload.as_a.flat_map do |row|
    row.as_a.map { |value| value.as_f.to_f32 }
  end
end

private def sparse_tensor_f32le_sha256(values : Indexable(Float32)) : String
  bytes = Bytes.new(values.size * 4, 0_u8)
  values.each_with_index do |value, index|
    IO::ByteFormat::LittleEndian.encode(value, bytes[index * 4, 4])
  end
  Digest::SHA256.hexdigest(bytes)
end

describe ML::Sparse::TensorCPU do
  it "snapshots aligned features and preserves immutable map identity" do
    fixture = sparse_tensor_fixture
    input = fixture["input"]
    coordinates = sparse_tensor_coordinates(input["coordinates"])
    features = sparse_tensor_features(input["features"])
    sparse_tensor_f32le_sha256(features).should eq(
      input["features_f32le_sha256"].as_s
    )
    map = ML::Sparse::CoordinateMap3D.new(coordinates, 3, {4, 3, 4})
    source = ML::Tensor.from_array(features, ML::Shape.new(6_i32, 2_i32))
    sparse = ML::Sparse::TensorCPU.new(source, map)

    sparse.shape.should eq({3, 2})
    sparse.point_count.should eq(6)
    sparse.channels.should eq(2)
    sparse.max_feature_bytes.should eq(ML::Sparse::TensorCPU::MAX_FEATURE_BYTES)
    sparse.coordinate_map.object_id.should eq(map.object_id)
    sparse.features_copy.should eq(features)
    sparse.feature(0, 0).should eq(20_010.0_f32)
    sparse.feature(5, 1).should eq(220_211.0_f32)

    source.cpu_data.not_nil![0] = -99.0_f32
    sparse.feature(0, 0).should eq(20_010.0_f32)
    copy = sparse.features_copy
    copy[0] = -77.0_f32
    sparse.feature(0, 0).should eq(20_010.0_f32)

    replacement_values = features.map { |value| value + 0.5_f32 }
    replacement = sparse.replace_features(
      ML::Tensor.from_array(replacement_values, ML::Shape.new(6_i32, 2_i32))
    )
    replacement.coordinate_map.object_id.should eq(map.object_id)
    replacement.feature(0, 0).should eq(20_010.5_f32)
    sparse.feature(0, 0).should eq(20_010.0_f32)

    new_coordinates = coordinates.dup
    new_coordinates[1] = 1
    moved = sparse.replace_coordinates(new_coordinates, 3, {4, 3, 4})
    moved.coordinate_map.object_id.should_not eq(map.object_id)
    moved.coordinate_map.index_of(0, 1, 0, 1).should eq(0)
    moved.coordinate_map.index_of(0, 2, 0, 1).should be_nil
    map.index_of(0, 2, 0, 1).should eq(0)
    moved.features_copy.should eq(sparse.features_copy)
  end

  it "accepts an empty finite CPU feature matrix" do
    map = ML::Sparse::CoordinateMap3D.new([] of Int32, 2, {4, 4, 4})
    sparse = ML::Sparse::TensorCPU.new(
      ML::Tensor.from_array([] of Float32, ML::Shape.new(0_i32, 3_i32)),
      map
    )
    sparse.shape.should eq({2, 3})
    sparse.point_count.should eq(0)
    sparse.features_copy.should be_empty
  end

  it "fails closed before publishing malformed feature alignment" do
    map = ML::Sparse::CoordinateMap3D.new(
      [0, 0, 0, 0, 0, 1, 1, 1] of Int32,
      1,
      {4, 4, 4}
    )
    expect_raises(ML::Sparse::SparseTensorError, /rank 2/) do
      ML::Sparse::TensorCPU.new(
        ML::Tensor.zeros(1, 2, 1, device: ML::Tensor::Device::CPU),
        map
      )
    end
    expect_raises(ML::Sparse::SparseTensorError, /point count/) do
      ML::Sparse::TensorCPU.new(
        ML::Tensor.zeros(3, 2, device: ML::Tensor::Device::CPU),
        map
      )
    end
    expect_raises(ML::Sparse::SparseTensorError, /channel count/) do
      ML::Sparse::TensorCPU.new(
        ML::Tensor.from_array([] of Float32, ML::Shape.new(2_i32, 0_i32)),
        map
      )
    end
    expect_raises(ML::Sparse::SparseTensorError, /byte budget/) do
      ML::Sparse::TensorCPU.new(
        ML::Tensor.zeros(2, 2, device: ML::Tensor::Device::CPU),
        map,
        0_i64
      )
    end
    expect_raises(ML::Sparse::SparseTensorBudgetError, /require 16 bytes, limit is 12/) do
      ML::Sparse::TensorCPU.new(
        ML::Tensor.zeros(2, 2, device: ML::Tensor::Device::CPU),
        map,
        12_i64
      )
    end
    expect_raises(ML::Sparse::SparseTensorError, /channel count/) do
      ML::Sparse::TensorCPU.new(
        ML::Tensor.zeros(2, 257, device: ML::Tensor::Device::CPU),
        map
      )
    end
    expect_raises(ML::Sparse::SparseTensorError, /contiguous/) do
      ML::Sparse::TensorCPU.new(
        ML::Tensor.zeros(2, 2, device: ML::Tensor::Device::CPU).transpose,
        map
      )
    end

    nonfinite = ML::Tensor.zeros(2, 2, device: ML::Tensor::Device::CPU)
    nonfinite.cpu_data.not_nil![2] = Float32::NAN
    expect_raises(ML::Sparse::SparseTensorError, /feature\[2\].*finite/) do
      ML::Sparse::TensorCPU.new(nonfinite, map)
    end

    sparse = ML::Sparse::TensorCPU.new(
      ML::Tensor.zeros(2, 2, device: ML::Tensor::Device::CPU),
      map
    )
    expect_raises(ML::Sparse::SparseTensorError, /point count/) do
      sparse.replace_coordinates([0, 0, 0, 0] of Int32, 1, {4, 4, 4})
    end
  end
end
