require "json"
require "digest/sha256"
require "../../src/ml/sparse/adaptive_layer_norm"
require "../spec_helper"

private AFFINE_FIXTURE_SHA256 =
  "3eb9ad23f6948ed5e1076a865f4dbad91b5eea4ff493c16600e02f9e7db9c28d"

private def sparse_affine_fixture_path : String
  File.join(
    __DIR__,
    "../fixtures/trellis2/sparse_adaptive_affine_cpu_v1.json"
  )
end

private def sparse_affine_fixture : JSON::Any
  path = sparse_affine_fixture_path
  Digest::SHA256.hexdigest(File.read(path).to_slice).should eq(
    AFFINE_FIXTURE_SHA256
  )
  JSON.parse(File.read(path))
end

private def sparse_affine_i32(payload : JSON::Any) : Array(Int32)
  payload.as_a.flat_map do |row|
    if values = row.as_a?
      values.map { |value| value.as_i.to_i32 }
    else
      [row.as_i.to_i32]
    end
  end
end

private def sparse_affine_f32(payload : JSON::Any) : Array(Float32)
  values = payload.as_h?.try(&.["values"]?) || payload
  values.as_a.flat_map do |row|
    if nested = row.as_a?
      nested.map { |value| value.as_f.to_f32 }
    else
      [row.as_f.to_f32]
    end
  end
end

private def sparse_affine_f32le_sha256(
  values : Indexable(Float32),
) : String
  bytes = Bytes.new(values.size * 4, 0_u8)
  values.each_with_index do |value, index|
    IO::ByteFormat::LittleEndian.encode(value, bytes[index * 4, 4])
  end
  Digest::SHA256.hexdigest(bytes)
end

class SparseAdaptiveAffineReceiverOverride < ML::Sparse::TensorCPU
  def self.==(other : ML::Sparse::TensorCPU.class) : Bool
    true
  end
end

class SparseAdaptiveAffineInputOverride < ML::Sparse::TensorCPU
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

describe "TRELLIS.2 sparse adaptive MLP affine seam" do
  it "matches the real pinned block boundary before MLP exactly" do
    fixture = sparse_affine_fixture
    fixture["schema"].as_s.should eq(
      "cogni-ml/trellis2/sparse-adaptive-affine-oracle/v1"
    )
    provenance = fixture["provenance"]
    provenance["commit"].as_s.should eq(
      "75fbf0183001ed9876c8dbb35de6b68552ee08bd"
    )
    provenance["sources"]["sparse_basic"]["sha256"].as_s.should eq(
      "99dbcb7298238fdb6b6d47918ec068f618c68067653489d22b32c136c1ae0e78"
    )
    provenance["sources"]["modulated_cross_block"]["sha256"].as_s.should eq(
      "fab9838c79b5fa9cbc6055c4a958f5a8e6f394f94e1691140be022caab7078d2"
    )
    provenance["sources"]["norm"]["sha256"].as_s.should eq(
      "f89c40abf3356f7b06fc85f0498cd77eefb677a0e0d370a43d14a735f4c40172"
    )
    provenance["upstream_method_executed"].as_s.should eq(
      "ModulatedSparseTransformerCrossBlock._forward"
    )
    provenance["network"].as_s.should eq("none")
    provenance["device"].as_s.should eq("cpu")
    provenance["dtype"].as_s.should eq("float32")
    provenance["generator_sha256"].as_s.should eq(
      "78e89c7ffd4bfeef29b2441b0b8beb7eddf0d8944c74739130d42ea51416ea9c"
    )

    source_fixture = fixture["source_fixture"]
    source_fixture["sha256"].as_s.should eq(
      "84cf93c7afa0e791439c1c41fb3d2afe0e131aff8640912cd26b94ea4773c35e"
    )
    source_fixture["generator_sha256"].as_s.should eq(
      "1cf99b1f900414908934eaa9f2e882bf67c987a8360c06b381eeb24db05283df"
    )
    contract = fixture["contract"]
    contract["source_expression"].as_s.should eq(
      "h = h * (1 + scale_mlp) + shift_mlp"
    )
    contract["boundary"].as_s.should eq(
      "after adaptive MLP affine and before MLP"
    )
    contract["modulation_chunk_order"].as_a.map(&.as_s).should eq([
      "shift_msa",
      "scale_msa",
      "gate_msa",
      "shift_mlp",
      "scale_mlp",
      "gate_mlp",
    ])

    input = fixture["input"]
    input["query_sequence_lengths"].as_a.map(&.as_i).should eq([2_i64, 0_i64, 2_i64])
    sparse_affine_i32(input["batch_broadcast_map"]).should eq(
      [0, 0, 2, 2] of Int32
    )
    coordinates = sparse_affine_i32(input["coordinates"])
    features = sparse_affine_f32(input["norm3"])
    scale_values = sparse_affine_f32(fixture["adaptive"]["scale_mlp"])
    shift_values = sparse_affine_f32(fixture["adaptive"]["shift_mlp"])
    expected = sparse_affine_f32(fixture["output"]["mlp_input"])
    sparse_affine_f32le_sha256(features).should eq(
      input["norm3"]["f32le_sha256"].as_s
    )
    sparse_affine_f32le_sha256(scale_values).should eq(
      fixture["adaptive"]["scale_mlp"]["f32le_sha256"].as_s
    )
    sparse_affine_f32le_sha256(shift_values).should eq(
      fixture["adaptive"]["shift_mlp"]["f32le_sha256"].as_s
    )
    sparse_affine_f32le_sha256(expected).should eq(
      fixture["output"]["mlp_input"]["f32le_sha256"].as_s
    )

    map = ML::Sparse::CoordinateMap3D.new(coordinates, 3, {3, 3, 3})
    sparse = ML::Sparse::TensorCPU.new(
      ML::Tensor.from_array(features, ML::Shape.new(4_i32, 16_i32)),
      map,
      256_i64
    )
    scale = ML::Tensor.from_array(
      scale_values,
      ML::Shape.new(3_i32, 16_i32)
    )
    shift = ML::Tensor.from_array(
      shift_values,
      ML::Shape.new(3_i32, 16_i32)
    )
    before = sparse.features_copy
    scale_before = scale.cpu_data.not_nil!.dup
    shift_before = shift.cpu_data.not_nil!.dup

    output = ML::Sparse::TensorCPU.apply_adaptive_affine(
      sparse,
      scale,
      shift
    )

    output.features_copy.should eq(expected)
    output.coordinate_map.same?(map).should be_true
    output.max_feature_bytes.should eq(256_i64)
    output.production_width?.should be_false
    sparse.features_copy.should eq(before)
    scale.cpu_data.not_nil!.should eq(scale_before)
    shift.cpu_data.not_nil!.should eq(shift_before)
  end

  it "does not normalize and maps scale and shift by declared batch" do
    map = ML::Sparse::CoordinateMap3D.new(
      [0, 0, 0, 0, 2, 0, 0, 0] of Int32,
      3,
      {1, 1, 1}
    )
    input = ML::Sparse::TensorCPU.new(
      ML::Tensor.from_array(
        [2.0_f32, -3.0_f32, 7.0_f32, 11.0_f32],
        ML::Shape.new(2_i32, 2_i32)
      ),
      map
    )
    scale = ML::Tensor.from_array(
      [1.0_f32, 0.0_f32, 99.0_f32, 99.0_f32, -1.0_f32, 0.5_f32],
      ML::Shape.new(3_i32, 2_i32)
    )
    shift = ML::Tensor.from_array(
      [0.0_f32, 5.0_f32, 99.0_f32, 99.0_f32, 4.0_f32, -1.0_f32],
      ML::Shape.new(3_i32, 2_i32)
    )

    output = ML::Sparse::TensorCPU.apply_adaptive_affine(
      input,
      scale,
      shift
    )

    output.features_copy.should eq([4.0_f32, 2.0_f32, 4.0_f32, 15.5_f32])
    output.coordinate_map.same?(map).should be_true
  end

  it "preserves empty trailing batches and production-width carrier authority" do
    empty_map = ML::Sparse::CoordinateMap3D.new(
      [] of Int32,
      3,
      {1, 1, 1}
    )
    empty = ML::Sparse::TensorCPU.new(
      ML::Tensor.from_array(
        [] of Float32,
        ML::Shape.new(0_i32, 2_i32)
      ),
      empty_map
    )
    empty_output = ML::Sparse::TensorCPU.apply_adaptive_affine(
      empty,
      ML::Tensor.zeros(3, 2, device: ML::Tensor::Device::CPU),
      ML::Tensor.zeros(3, 2, device: ML::Tensor::Device::CPU)
    )
    empty_output.features_copy.should be_empty
    empty_output.shape.should eq({3_i32, 2_i32})
    empty_output.coordinate_map.same?(empty_map).should be_true

    map = ML::Sparse::CoordinateMap3D.new(
      [0, 0, 0, 0] of Int32,
      3,
      {1, 1, 1}
    )
    production = ML::Sparse::TensorCPU.production(
      ML::Tensor.from_array(
        Array(Float32).new(1_536, 2.0_f32),
        ML::Shape.new(1_i32, 1_536_i32)
      ),
      map
    )
    production_output = ML::Sparse::TensorCPU.apply_adaptive_affine(
      production,
      ML::Tensor.zeros(3, 1_536, device: ML::Tensor::Device::CPU),
      ML::Tensor.zeros(3, 1_536, device: ML::Tensor::Device::CPU)
    )
    production_output.features_copy.should eq(
      Array(Float32).new(1_536, 2.0_f32)
    )
    production_output.production_width?.should be_true
    production_output.coordinate_map.same?(map).should be_true
  end

  it "rejects uninitialized, malformed, and mismatched values" do
    map = ML::Sparse::CoordinateMap3D.new(
      [0, 0, 0, 0] of Int32,
      2,
      {1, 1, 1}
    )
    scale = ML::Tensor.zeros(2, 2, device: ML::Tensor::Device::CPU)
    shift = ML::Tensor.zeros(2, 2, device: ML::Tensor::Device::CPU)
    expect_raises(ML::Sparse::SparseTensorError, /initialized sparse value/) do
      ML::Sparse::TensorCPU.apply_adaptive_affine(
        ML::Sparse::TensorCPU.allocate,
        scale,
        shift
      )
    end

    wrong_points = SparseAdaptiveAffineInputOverride.new(
      [1.0_f32, 2.0_f32],
      map,
      0_i32,
      2_i32,
      8_i64
    )
    expect_raises(ML::Sparse::SparseTensorError, /coordinate and feature point counts/) do
      ML::Sparse::TensorCPU.apply_adaptive_affine(
        wrong_points,
        scale,
        shift
      )
    end

    bad_storage = SparseAdaptiveAffineInputOverride.new(
      [1.0_f32],
      map,
      1_i32,
      2_i32,
      8_i64
    )
    expect_raises(ML::Sparse::SparseTensorError, /storage size.*\[N, C\]/) do
      ML::Sparse::TensorCPU.apply_adaptive_affine(
        bad_storage,
        scale,
        shift
      )
    end

    valid = ML::Sparse::TensorCPU.new(
      ML::Tensor.ones(1, 2, device: ML::Tensor::Device::CPU),
      map
    )
    expect_raises(ML::Sparse::SparseTensorError, /scale shape.*\[2, 2\]/) do
      ML::Sparse::TensorCPU.apply_adaptive_affine(
        valid,
        ML::Tensor.zeros(1, 2, device: ML::Tensor::Device::CPU),
        shift
      )
    end
    expect_raises(ML::Sparse::SparseTensorError, /shift shape.*\[2, 2\]/) do
      ML::Sparse::TensorCPU.apply_adaptive_affine(
        valid,
        scale,
        ML::Tensor.zeros(4, device: ML::Tensor::Device::CPU)
      )
    end
    expect_raises(ML::Sparse::SparseTensorError, /scale.*contiguous/) do
      ML::Sparse::TensorCPU.apply_adaptive_affine(
        valid,
        ML::Tensor.zeros(
          2,
          2,
          device: ML::Tensor::Device::CPU
        ).transpose,
        shift
      )
    end
  end

  it "checks the inherited byte cap before poisoned payloads" do
    map = ML::Sparse::CoordinateMap3D.new(
      [0, 0, 0, 0] of Int32,
      1,
      {1, 1, 1}
    )
    insufficient = SparseAdaptiveAffineInputOverride.new(
      [Float32::NAN, Float32::NAN],
      map,
      1_i32,
      2_i32,
      7_i64
    )
    poisoned_scale = ML::Tensor.from_array(
      [Float32::NAN, Float32::NAN],
      ML::Shape.new(1_i32, 2_i32)
    )
    expect_raises(
      ML::Sparse::SparseTensorBudgetError,
      /require 8 bytes.*limit is 7/
    ) do
      ML::Sparse::TensorCPU.apply_adaptive_affine(
        insufficient,
        poisoned_scale,
        poisoned_scale
      )
    end
  end

  it "rejects non-finite adaptive parameters and arithmetic output" do
    map = ML::Sparse::CoordinateMap3D.new(
      [0, 0, 0, 0] of Int32,
      1,
      {1, 1, 1}
    )
    input = ML::Sparse::TensorCPU.new(
      ML::Tensor.ones(1, 2, device: ML::Tensor::Device::CPU),
      map
    )
    scale = ML::Tensor.zeros(1, 2, device: ML::Tensor::Device::CPU)
    shift = ML::Tensor.zeros(1, 2, device: ML::Tensor::Device::CPU)
    scale.cpu_data.not_nil![1] = Float32::NAN
    expect_raises(ML::Sparse::SparseTensorError, /scale\[1\].*finite/) do
      ML::Sparse::TensorCPU.apply_adaptive_affine(input, scale, shift)
    end

    scale.cpu_data.not_nil![1] = 0.0_f32
    shift.cpu_data.not_nil![0] = Float32::INFINITY
    expect_raises(ML::Sparse::SparseTensorError, /shift\[0\].*finite/) do
      ML::Sparse::TensorCPU.apply_adaptive_affine(input, scale, shift)
    end

    divergent = ML::Sparse::TensorCPU.new(
      ML::Tensor.from_array(
        [Float32::MAX, 0.0_f32],
        ML::Shape.new(1_i32, 2_i32)
      ),
      map
    )
    expect_raises(ML::Sparse::SparseTensorError, /output\[0\].*finite/) do
      ML::Sparse::TensorCPU.apply_adaptive_affine(
        divergent,
        ML::Tensor.from_array(
          [1.0_f32, 0.0_f32],
          ML::Shape.new(1_i32, 2_i32)
        ),
        ML::Tensor.zeros(1, 2, device: ML::Tensor::Device::CPU)
      )
    end
  end

  it "requires the base TensorCPU receiver" do
    map = ML::Sparse::CoordinateMap3D.new(
      [0, 0, 0, 0] of Int32,
      1,
      {1, 1, 1}
    )
    sparse = ML::Sparse::TensorCPU.new(
      ML::Tensor.ones(1, 1, device: ML::Tensor::Device::CPU),
      map
    )
    expect_raises(ML::Sparse::SparseTensorError, /base TensorCPU receiver/) do
      SparseAdaptiveAffineReceiverOverride.apply_adaptive_affine(
        sparse,
        ML::Tensor.zeros(1, 1, device: ML::Tensor::Device::CPU),
        ML::Tensor.zeros(1, 1, device: ML::Tensor::Device::CPU)
      )
    end
  end
end
