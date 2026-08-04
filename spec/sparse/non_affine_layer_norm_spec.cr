require "json"
require "digest/sha256"
require "../../src/ml/sparse/adaptive_layer_norm"
require "../spec_helper"

private def sparse_norm3_fixture : JSON::Any
  JSON.parse(File.read(File.join(
    __DIR__,
    "../fixtures/trellis2/sparse_norm3_cpu_v1.json"
  )))
end

private def sparse_norm3_i32(payload : JSON::Any) : Array(Int32)
  payload.as_a.flat_map do |row|
    row.as_a.map { |value| value.as_i.to_i32 }
  end
end

private def sparse_norm3_f32(payload : JSON::Any) : Array(Float32)
  values = payload.as_h?.try(&.["values"]?) || payload
  values.as_a.map { |value| value.as_f.to_f32 }
end

private def sparse_norm3_f32le_sha256(
  values : Indexable(Float32),
) : String
  bytes = Bytes.new(values.size * 4, 0_u8)
  values.each_with_index do |value, index|
    IO::ByteFormat::LittleEndian.encode(value, bytes[index * 4, 4])
  end
  Digest::SHA256.hexdigest(bytes)
end

class SparseNonAffineLayerNormReceiverOverride < ML::Sparse::TensorCPU
  def self.==(other : ML::Sparse::TensorCPU.class) : Bool
    true
  end
end

class SparseNonAffineLayerNormInputOverride < ML::Sparse::TensorCPU
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

describe "TRELLIS.2 sparse non-affine LayerNorm32" do
  it "matches the pinned norm3 seam exactly" do
    fixture = sparse_norm3_fixture
    fixture["schema"].as_s.should eq(
      "cogni-ml/trellis2/sparse-norm3-oracle/v1"
    )
    fixture["provenance"]["commit"].as_s.should eq(
      "75fbf0183001ed9876c8dbb35de6b68552ee08bd"
    )
    fixture["provenance"]["sources"]["modulated_cross_block"]["sha256"].as_s.should eq(
      "fab9838c79b5fa9cbc6055c4a958f5a8e6f394f94e1691140be022caab7078d2"
    )
    fixture["provenance"]["sources"]["norm"]["sha256"].as_s.should eq(
      "f89c40abf3356f7b06fc85f0498cd77eefb677a0e0d370a43d14a735f4c40172"
    )
    fixture["source_fixture"]["sha256"].as_s.should eq(
      "04cc1f953acbb14dec9ca723e71d951dbeea155efb27678784c666dee3b7553a"
    )
    fixture["source_fixture"]["generator_sha256"].as_s.should eq(
      "a7cb916d8d45c81ea4b79da9fdbe060f9b2266af8d9c5367dfeba066b736e574"
    )
    contract = fixture["contract"]
    contract["owner"].as_s.should eq(
      "ModulatedSparseTransformerCrossBlock.norm3"
    )
    contract["boundary"].as_s.should eq(
      "after norm3 and before scale_mlp/shift_mlp"
    )
    contract["layer_norm_elementwise_affine"].as_bool.should be_false
    contract["epsilon"].as_f.to_f32.should eq(1e-6_f32)

    input = fixture["input"]
    source = sparse_norm3_f32(input["after_cross"])
    expected = sparse_norm3_f32(fixture["output"]["norm3"])
    sparse_norm3_f32le_sha256(source).should eq(
      "f04bd7c46104b2072b38cba375bf9fb084961974ebda067b5b87fc1eeb0c8249"
    )
    sparse_norm3_f32le_sha256(expected).should eq(
      "96c3f44b97e8c2b11e2ccfb26da886298c2a27952575dd5cce478929b88d2442"
    )
    coordinates = sparse_norm3_i32(input["coordinates"])
    map = ML::Sparse::CoordinateMap3D.new(
      coordinates,
      input["batch_size"].as_i.to_i32,
      {
        input["spatial_shape"][0].as_i.to_i32,
        input["spatial_shape"][1].as_i.to_i32,
        input["spatial_shape"][2].as_i.to_i32,
      }
    )
    sparse = ML::Sparse::TensorCPU.new(
      ML::Tensor.from_array(
        source,
        ML::Shape.new(4_i32, input["channels"].as_i.to_i32)
      ),
      map,
      256_i64
    )
    before = sparse.features_copy

    output = ML::Sparse::TensorCPU.apply_non_affine_layer_norm(sparse)

    output.features_copy.should eq(expected)
    sparse_norm3_f32le_sha256(output.features_copy).should eq(
      fixture["output"]["norm3"]["f32le_sha256"].as_s
    )
    output.coordinate_map.same?(map).should be_true
    output.max_feature_bytes.should eq(256_i64)
    output.production_width?.should be_false
    sparse.features_copy.should eq(before)
  end

  it "supports empty, single-channel, and production-width carriers" do
    empty_map = ML::Sparse::CoordinateMap3D.new(
      [] of Int32,
      3,
      {1, 1, 1}
    )
    empty = ML::Sparse::TensorCPU.new(
      ML::Tensor.from_array(
        [] of Float32,
        ML::Shape.new(0_i32, 4_i32)
      ),
      empty_map
    )
    empty_output = ML::Sparse::TensorCPU.apply_non_affine_layer_norm(empty)
    empty_output.features_copy.should be_empty
    empty_output.coordinate_map.same?(empty_map).should be_true

    map = ML::Sparse::CoordinateMap3D.new(
      [0, 0, 0, 0] of Int32,
      1,
      {1, 1, 1}
    )
    single = ML::Sparse::TensorCPU.new(
      ML::Tensor.from_array(
        [7.0_f32],
        ML::Shape.new(1_i32, 1_i32)
      ),
      map
    )
    ML::Sparse::TensorCPU.apply_non_affine_layer_norm(single)
      .features_copy.should eq([0.0_f32])

    production = ML::Sparse::TensorCPU.production(
      ML::Tensor.from_array(
        Array(Float32).new(1_536, 7.0_f32),
        ML::Shape.new(1_i32, 1_536_i32)
      ),
      map
    )
    production_output = ML::Sparse::TensorCPU.apply_non_affine_layer_norm(
      production
    )
    production_output.features_copy.should eq(
      Array(Float32).new(1_536, 0.0_f32)
    )
    production_output.production_width?.should be_true
  end

  it "rejects uninitialized, malformed, and epsilon-drifted inputs" do
    map = ML::Sparse::CoordinateMap3D.new(
      [0, 0, 0, 0] of Int32,
      1,
      {1, 1, 1}
    )
    raw = ML::Sparse::TensorCPU.allocate
    expect_raises(ML::Sparse::SparseTensorError, /initialized sparse value/) do
      ML::Sparse::TensorCPU.apply_non_affine_layer_norm(raw)
    end

    wrong_points = SparseNonAffineLayerNormInputOverride.new(
      [1.0_f32],
      map,
      0_i32,
      1_i32,
      4_i64
    )
    expect_raises(
      ML::Sparse::SparseTensorError,
      /coordinate and feature point counts/
    ) do
      ML::Sparse::TensorCPU.apply_non_affine_layer_norm(wrong_points)
    end

    bad_storage = SparseNonAffineLayerNormInputOverride.new(
      [] of Float32,
      map,
      1_i32,
      1_i32,
      4_i64
    )
    expect_raises(ML::Sparse::SparseTensorError, /storage size.*\[N, C\]/) do
      ML::Sparse::TensorCPU.apply_non_affine_layer_norm(bad_storage)
    end

    valid = ML::Sparse::TensorCPU.new(
      ML::Tensor.from_array(
        [1.0_f32],
        ML::Shape.new(1_i32, 1_i32)
      ),
      map
    )
    expect_raises(ML::Sparse::SparseTensorError, /epsilon.*1.0e-6/) do
      ML::Sparse::TensorCPU.apply_non_affine_layer_norm(valid, 1e-5_f32)
    end
  end

  it "checks the inherited byte cap before normalization arithmetic" do
    map = ML::Sparse::CoordinateMap3D.new(
      [0, 0, 0, 0] of Int32,
      1,
      {1, 1, 1}
    )
    insufficient = SparseNonAffineLayerNormInputOverride.new(
      [Float32::NAN, Float32::NAN],
      map,
      1_i32,
      2_i32,
      7_i64
    )
    expect_raises(
      ML::Sparse::SparseTensorBudgetError,
      /require 8 bytes.*limit is 7/
    ) do
      ML::Sparse::TensorCPU.apply_non_affine_layer_norm(insufficient)
    end
  end

  it "rejects non-finite normalized output" do
    map = ML::Sparse::CoordinateMap3D.new(
      [0, 0, 0, 0] of Int32,
      1,
      {1, 1, 1}
    )
    {Float32::NAN, Float32::INFINITY, -Float32::INFINITY}.each do |invalid|
      poisoned = SparseNonAffineLayerNormInputOverride.new(
        [invalid, 1.0_f32],
        map,
        1_i32,
        2_i32,
        8_i64
      )
      expect_raises(ML::Sparse::SparseTensorError, /output\[0\].*finite/) do
        ML::Sparse::TensorCPU.apply_non_affine_layer_norm(poisoned)
      end
    end
  end

  it "requires the base TensorCPU receiver" do
    map = ML::Sparse::CoordinateMap3D.new(
      [0, 0, 0, 0] of Int32,
      1,
      {1, 1, 1}
    )
    sparse = ML::Sparse::TensorCPU.new(
      ML::Tensor.from_array(
        [1.0_f32],
        ML::Shape.new(1_i32, 1_i32)
      ),
      map
    )
    expect_raises(ML::Sparse::SparseTensorError, /base TensorCPU receiver/) do
      SparseNonAffineLayerNormReceiverOverride
        .apply_non_affine_layer_norm(sparse)
    end
  end
end
