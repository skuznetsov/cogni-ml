require "json"
require "digest/sha256"
require "../../src/ml/sparse/adaptive_layer_norm"
require "../spec_helper"

private def sparse_adaln_fixture : JSON::Any
  JSON.parse(File.read(File.join(
    __DIR__,
    "../fixtures/trellis2/sparse_adaptive_layer_norm_cpu_v1.json"
  )))
end

private def sparse_adaln_i32(payload : JSON::Any) : Array(Int32)
  if payload.as_a.first?.try(&.as_a?)
    payload.as_a.flat_map do |row|
      row.as_a.map { |value| value.as_i.to_i32 }
    end
  else
    payload.as_a.map { |value| value.as_i.to_i32 }
  end
end

private def sparse_adaln_f32(payload : JSON::Any) : Array(Float32)
  if payload.as_a.first?.try(&.as_a?)
    payload.as_a.flat_map do |row|
      row.as_a.map { |value| value.as_f.to_f32 }
    end
  else
    payload.as_a.map { |value| value.as_f.to_f32 }
  end
end

private def sparse_adaln_f32le_sha256(values : Indexable(Float32)) : String
  bytes = Bytes.new(values.size * 4, 0_u8)
  values.each_with_index do |value, index|
    IO::ByteFormat::LittleEndian.encode(value, bytes[index * 4, 4])
  end
  Digest::SHA256.hexdigest(bytes)
end

class SparseAdaptiveLayerNormReceiverOverride < ML::Sparse::TensorCPU
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

class SparseAdaptiveLayerNormMapOverride < ML::Sparse::CoordinateMap3D
  def class : ML::Sparse::CoordinateMap3D.class
    ML::Sparse::CoordinateMap3D
  end

  def batch_size : Int32
    1_000_i32
  end

  def point_count : Int32
    0_i32
  end

  def coordinate(row : Int32, axis : Int32) : Int32
    999_i32
  end
end

class SparseAdaptiveLayerNormInputOverride < ML::Sparse::TensorCPU
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

describe ML::Sparse::TensorCPU do
  it "matches the pinned asymmetric adaptive LayerNorm32 oracle" do
    fixture = sparse_adaln_fixture
    fixture["schema"].as_s.should eq(
      "cogni-ml/trellis2/sparse-adaptive-layer-norm-oracle/v1"
    )
    provenance = fixture["provenance"]
    provenance["commit"].as_s.should eq(
      "75fbf0183001ed9876c8dbb35de6b68552ee08bd"
    )
    sources = provenance["sources"]
    sources["sparse_basic"]["sha256"].as_s.should eq(
      "99dbcb7298238fdb6b6d47918ec068f618c68067653489d22b32c136c1ae0e78"
    )
    sources["modulated_block"]["sha256"].as_s.should eq(
      "fab9838c79b5fa9cbc6055c4a958f5a8e6f394f94e1691140be022caab7078d2"
    )
    sources["norm"]["sha256"].as_s.should eq(
      "f89c40abf3356f7b06fc85f0498cd77eefb677a0e0d370a43d14a735f4c40172"
    )
    sources["structured_flow"]["sha256"].as_s.should eq(
      "76454ead55d112214e36db8de5e9b3d1d4128f05d25256fb6c581b4c1a588021"
    )
    provenance["python_version"].as_s.should eq("3.11.9")
    provenance["torch_version"].as_s.should eq("2.9.0")
    provenance["numpy_version"].as_s.should eq("2.1.3")
    provenance["system"].as_s.should eq("Darwin")
    provenance["machine"].as_s.should eq("arm64")
    provenance["torch_cpu_capability"].as_s.should eq("DEFAULT")
    pytorch_sources = provenance["pytorch_cpu_sources"]
    pytorch_sources["layer_norm_kernel"]["sha256"].as_s.should eq(
      "14c765e7e931ea313bf0ef42b0c9099d7e762c8f3d33ee408e96807435756711"
    )
    pytorch_sources["moments_utils"]["sha256"].as_s.should eq(
      "9b421e0b16cdf9f64c4a3e0a97201c71c58f4123522370c8cc79aae0d2f885a1"
    )
    provenance["network"].as_s.should eq("none")
    provenance["device"].as_s.should eq("cpu")
    provenance["weights"].as_s.should eq("synthetic")
    provenance["sparse_backend"].as_s.should eq("none")

    contract = fixture["contract"]
    contract["owner"].as_s.should eq(
      "ModulatedSparseTransformerCrossBlock.norm1"
    )
    contract["upstream_calls"].as_a.map(&.as_s).should eq([
      "x.replace(LayerNorm32(x.feats))",
      "h * (1 + scale) + shift",
    ])
    contract["layer_norm_elementwise_affine"].as_bool.should be_false
    contract["epsilon"].as_f.to_f32.should eq(1e-6_f32)
    contract["variance"].as_s.should eq("population")
    contract["moments_algorithm"].as_s.should eq(
      "PyTorch F32 RowwiseMoments Welford cascade"
    )
    contract["cpu_vector_width"].as_i.should eq(4)
    contract["moments_chunk_size"].as_i.should eq(16)
    contract["adaptive_parameter_shape"].as_s.should eq(
      "[batch_size, channels]"
    )

    input = fixture["input"]
    coordinates = sparse_adaln_i32(input["coordinates"])
    features = sparse_adaln_f32(input["features"])
    batch_map = sparse_adaln_i32(input["batch_broadcast_map"])
    scale_values = sparse_adaln_f32(fixture["adaptive"]["scale"])
    shift_values = sparse_adaln_f32(fixture["adaptive"]["shift"])
    expected = sparse_adaln_f32(fixture["output"]["features"])
    normalized = sparse_adaln_f32(
      fixture["intermediate"]["normalized_features"]
    )
    scaled = sparse_adaln_f32(fixture["intermediate"]["scaled_features"])
    sparse_adaln_f32le_sha256(features).should eq(
      input["features_f32le_sha256"].as_s
    )
    sparse_adaln_f32le_sha256(scale_values).should eq(
      fixture["adaptive"]["scale_f32le_sha256"].as_s
    )
    sparse_adaln_f32le_sha256(shift_values).should eq(
      fixture["adaptive"]["shift_f32le_sha256"].as_s
    )
    sparse_adaln_f32le_sha256(expected).should eq(
      fixture["output"]["features_f32le_sha256"].as_s
    )
    sparse_adaln_f32le_sha256(normalized).should eq(
      fixture["intermediate"]["normalized_features_f32le_sha256"].as_s
    )
    sparse_adaln_f32le_sha256(scaled).should eq(
      fixture["intermediate"]["scaled_features_f32le_sha256"].as_s
    )
    batch_map.should eq([0, 0, 0, 2, 2] of Int32)

    map = ML::Sparse::CoordinateMap3D.new(coordinates, 3, {3, 3, 2})
    sparse = ML::Sparse::TensorCPU.new(
      ML::Tensor.from_array(features, ML::Shape.new(5_i32, 4_i32)),
      map,
      128_i64
    )
    scale = ML::Tensor.from_array(
      scale_values,
      ML::Shape.new(3_i32, 4_i32)
    )
    shift = ML::Tensor.from_array(
      shift_values,
      ML::Shape.new(3_i32, 4_i32)
    )

    output = ML::Sparse::TensorCPU.apply_adaptive_layer_norm(
      sparse,
      scale,
      shift
    )

    output.point_count.should eq(5)
    output.channels.should eq(4)
    output.coordinate_map.same?(map).should be_true
    output.max_feature_bytes.should eq(128_i64)
    output.features_copy.should eq(expected)

    scale.cpu_data.not_nil![0] = -99.0_f32
    shift.cpu_data.not_nil![0] = -99.0_f32
    output.features_copy.should eq(expected)
  end

  it "supports empty sparse values and exact single-channel normalization" do
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
    scale = ML::Tensor.ones(3, 4)
    shift = ML::Tensor.zeros(3, 4)
    output = ML::Sparse::TensorCPU.apply_adaptive_layer_norm(
      empty,
      scale,
      shift
    )
    output.features_copy.should be_empty
    output.coordinate_map.same?(empty_map).should be_true

    map = ML::Sparse::CoordinateMap3D.new(
      [1, 0, 0, 0] of Int32,
      2,
      {1, 1, 1}
    )
    single = ML::Sparse::TensorCPU.new(
      ML::Tensor.from_array(
        [7.0_f32],
        ML::Shape.new(1_i32, 1_i32)
      ),
      map
    )
    single_scale = ML::Tensor.from_array(
      [10.0_f32, -3.0_f32],
      ML::Shape.new(2_i32, 1_i32)
    )
    single_shift = ML::Tensor.from_array(
      [100.0_f32, -2.5_f32],
      ML::Shape.new(2_i32, 1_i32)
    )
    ML::Sparse::TensorCPU.apply_adaptive_layer_norm(
      single,
      single_scale,
      single_shift
    ).features_copy.should eq([-2.5_f32])
  end

  it "matches pinned RowwiseMoments vector-tail and cascade boundaries" do
    sparse_adaln_fixture["numerical_cases"].as_a.each do |payload|
      channels = payload["channels"].as_i.to_i32
      features = sparse_adaln_f32(payload["features"])
      expected = sparse_adaln_f32(payload["output"])
      sparse_adaln_f32le_sha256(features).should eq(
        payload["features_f32le_sha256"].as_s
      )
      sparse_adaln_f32le_sha256(expected).should eq(
        payload["output_f32le_sha256"].as_s
      )

      map = ML::Sparse::CoordinateMap3D.new(
        [0, 0, 0, 0] of Int32,
        1,
        {1, 1, 1}
      )
      sparse = ML::Sparse::TensorCPU.new(
        ML::Tensor.from_array(
          features,
          ML::Shape.new(1_i32, channels)
        ),
        map
      )
      output = ML::Sparse::TensorCPU.apply_adaptive_layer_norm(
        sparse,
        ML::Tensor.zeros(1, channels),
        ML::Tensor.zeros(1, channels)
      )
      output.features_copy.zip(expected).each do |actual, wanted|
        actual.should be_close(wanted, 2e-6)
      end
    end
  end

  it "rejects uninitialized input, shape drift, and epsilon drift" do
    map = ML::Sparse::CoordinateMap3D.new(
      [0, 0, 0, 0] of Int32,
      2,
      {1, 1, 1}
    )
    sparse = ML::Sparse::TensorCPU.new(ML::Tensor.ones(1, 2), map)
    scale = ML::Tensor.zeros(2, 2)
    shift = ML::Tensor.zeros(2, 2)

    raw = ML::Sparse::TensorCPU.allocate
    expect_raises(ML::Sparse::SparseTensorError, /initialized sparse value/) do
      ML::Sparse::TensorCPU.apply_adaptive_layer_norm(raw, scale, shift)
    end
    expect_raises(ML::Sparse::SparseTensorError, /scale shape.*\[2, 2\]/) do
      ML::Sparse::TensorCPU.apply_adaptive_layer_norm(
        sparse,
        ML::Tensor.zeros(1, 2),
        shift
      )
    end
    expect_raises(ML::Sparse::SparseTensorError, /shift shape.*\[2, 2\]/) do
      ML::Sparse::TensorCPU.apply_adaptive_layer_norm(
        sparse,
        scale,
        ML::Tensor.zeros(4)
      )
    end
    expect_raises(ML::Sparse::SparseTensorError, /epsilon.*1.0e-6/) do
      ML::Sparse::TensorCPU.apply_adaptive_layer_norm(
        sparse,
        scale,
        shift,
        1e-5_f32
      )
    end

    expect_raises(ML::Sparse::SparseTensorError, /scale.*contiguous/) do
      ML::Sparse::TensorCPU.apply_adaptive_layer_norm(
        sparse,
        ML::Tensor.ones(2, 2).transpose,
        shift
      )
    end
  end

  it "rejects inherited subclass receivers before constructing output" do
    map = ML::Sparse::CoordinateMap3D.new(
      [0, 0, 0, 0] of Int32,
      1,
      {1, 1, 1}
    )
    sparse = ML::Sparse::TensorCPU.new(ML::Tensor.ones(1, 1), map)
    scale = ML::Tensor.zeros(1, 1)
    shift = ML::Tensor.zeros(1, 1)

    expect_raises(ML::Sparse::SparseTensorError, /base TensorCPU receiver/) do
      SparseAdaptiveLayerNormReceiverOverride.apply_adaptive_layer_norm(
        sparse,
        scale,
        shift
      )
    end

    overridden_map = SparseAdaptiveLayerNormMapOverride.new(
      [0, 0, 0, 0, 1, 0, 0, 0] of Int32,
      2,
      {1, 1, 1}
    )
    overridden = ML::Sparse::TensorCPU.new(
      ML::Tensor.ones(2, 1),
      overridden_map
    )
    override_scale = ML::Tensor.zeros(2, 1)
    override_shift = ML::Tensor.from_array(
      [10.0_f32, 20.0_f32],
      ML::Shape.new(2_i32, 1_i32)
    )
    override_output = ML::Sparse::TensorCPU.apply_adaptive_layer_norm(
      overridden,
      override_scale,
      override_shift
    )
    override_output.features_copy.should eq([10.0_f32, 20.0_f32])
    override_output.shape.should eq({2_i32, 1_i32})
  end

  it "revalidates private-ownership input shape and budget invariants" do
    map = ML::Sparse::CoordinateMap3D.new(
      [0, 0, 0, 0] of Int32,
      1,
      {1, 1, 1}
    )
    scale = ML::Tensor.zeros(1, 1)
    shift = ML::Tensor.zeros(1, 1)

    zero_channels = SparseAdaptiveLayerNormInputOverride.new(
      [] of Float32,
      map,
      1_i32,
      0_i32,
      4_i64
    )
    expect_raises(ML::Sparse::SparseTensorError, /channel count.*1\.\.256/) do
      ML::Sparse::TensorCPU.apply_adaptive_layer_norm(
        zero_channels,
        scale,
        shift
      )
    end

    bad_storage = SparseAdaptiveLayerNormInputOverride.new(
      [] of Float32,
      map,
      1_i32,
      1_i32,
      4_i64
    )
    expect_raises(ML::Sparse::SparseTensorError, /storage size.*\[N, C\]/) do
      ML::Sparse::TensorCPU.apply_adaptive_layer_norm(
        bad_storage,
        scale,
        shift
      )
    end

    insufficient_budget = SparseAdaptiveLayerNormInputOverride.new(
      [1.0_f32],
      map,
      1_i32,
      1_i32,
      1_i64
    )
    expect_raises(ML::Sparse::SparseTensorBudgetError, /require 4 bytes.*limit is 1/) do
      ML::Sparse::TensorCPU.apply_adaptive_layer_norm(
        insufficient_budget,
        scale,
        shift
      )
    end
  end

  it "rejects non-finite adaptive parameters and outputs" do
    map = ML::Sparse::CoordinateMap3D.new(
      [0, 0, 0, 0] of Int32,
      1,
      {1, 1, 1}
    )
    sparse = ML::Sparse::TensorCPU.new(ML::Tensor.ones(1, 2), map)
    scale = ML::Tensor.zeros(1, 2)
    shift = ML::Tensor.zeros(1, 2)
    scale.cpu_data.not_nil![1] = Float32::NAN
    expect_raises(ML::Sparse::SparseTensorError, /scale\[1\].*finite/) do
      ML::Sparse::TensorCPU.apply_adaptive_layer_norm(sparse, scale, shift)
    end

    scale.cpu_data.not_nil![1] = 0.0_f32
    shift.cpu_data.not_nil![0] = Float32::INFINITY
    expect_raises(ML::Sparse::SparseTensorError, /shift\[0\].*finite/) do
      ML::Sparse::TensorCPU.apply_adaptive_layer_norm(sparse, scale, shift)
    end

    empty_middle_map = ML::Sparse::CoordinateMap3D.new(
      [0, 0, 0, 0, 2, 0, 0, 0] of Int32,
      3,
      {1, 1, 1}
    )
    empty_middle = ML::Sparse::TensorCPU.new(
      ML::Tensor.ones(2, 2),
      empty_middle_map
    )
    unused_nonfinite_scale = ML::Tensor.zeros(3, 2)
    unused_nonfinite_scale.cpu_data.not_nil![2] = Float32::NAN
    expect_raises(ML::Sparse::SparseTensorError, /scale\[2\].*finite/) do
      ML::Sparse::TensorCPU.apply_adaptive_layer_norm(
        empty_middle,
        unused_nonfinite_scale,
        ML::Tensor.zeros(3, 2)
      )
    end

    divergent = ML::Sparse::TensorCPU.new(
      ML::Tensor.from_array(
        [1.0_f32, -1.0_f32],
        ML::Shape.new(1_i32, 2_i32)
      ),
      map
    )
    finite_scale = ML::Tensor.from_array(
      [Float32::MAX, 0.0_f32],
      ML::Shape.new(1_i32, 2_i32)
    )
    finite_shift = ML::Tensor.from_array(
      [Float32::MAX, 0.0_f32],
      ML::Shape.new(1_i32, 2_i32)
    )
    expect_raises(ML::Sparse::SparseTensorError, /output\[0\].*finite/) do
      ML::Sparse::TensorCPU.apply_adaptive_layer_norm(
        divergent,
        finite_scale,
        finite_shift
      )
    end
  end
end
