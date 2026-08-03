require "json"
require "../../src/ml/sparse/self_attention_rope"
require "../../src/ml/sparse/full_self_attention"
require "../spec_helper"

private SPARSE_SELF_ATTENTION_ROPE_TOLERANCE = 3.0e-6_f32

private def sparse_self_attention_rope_fixture : JSON::Any
  JSON.parse(File.read(File.join(
    __DIR__,
    "../fixtures/trellis2/sparse_self_attention_rope_cpu_v1.json"
  )))
end

private def flatten_sparse_self_attention_rope_f32(
  payload : JSON::Any,
  output = [] of Float32,
) : Array(Float32)
  if array = payload.as_a?
    array.each { |entry| flatten_sparse_self_attention_rope_f32(entry, output) }
  else
    output << payload.as_f.to_f32
  end
  output
end

private def sparse_self_attention_rope_i32(payload : JSON::Any) : Array(Int32)
  payload.as_a.flat_map do |row|
    row.as_a.map { |value| value.as_i.to_i32 }
  end
end

private def sparse_self_attention_rope_input(
  fixture : JSON::Any,
) : ML::Sparse::SelfAttentionQKVCPU
  input = fixture["input"]
  point_count = input["coordinates"].as_a.size.to_i32
  num_heads = input["num_heads"].as_i.to_i32
  head_dim = input["head_dim"].as_i.to_i32
  map = ML::Sparse::CoordinateMap3D.new(
    sparse_self_attention_rope_i32(input["coordinates"]),
    input["batch_size"].as_i.to_i32,
    {
      input["spatial_shape"][0].as_i.to_i32,
      input["spatial_shape"][1].as_i.to_i32,
      input["spatial_shape"][2].as_i.to_i32,
    }
  )
  flat = ML::Sparse::TensorCPU.new(
    ML::Tensor.from_array(
      flatten_sparse_self_attention_rope_f32(input["qkv_features"]),
      ML::Shape.new(point_count, 3_i32 * num_heads * head_dim)
    ),
    map
  )
  ML::Sparse::SelfAttentionQKVCPU.new(flat, num_heads)
end

class SparseSelfAttentionRoPETensorCPUOverride < ML::Sparse::TensorCPU
end

class SparseSelfAttentionRoPEQKVOverride < ML::Sparse::SelfAttentionQKVCPU
end

describe ML::Sparse::TensorCPU do
  it "matches the pinned upstream sparse 3D RoPE fixture" do
    fixture = sparse_self_attention_rope_fixture
    fixture["schema"].as_s.should eq(
      "cogni-ml/trellis2/sparse-self-attention-rope-oracle/v1"
    )
    fixture["provenance"]["commit"].as_s.should eq(
      "75fbf0183001ed9876c8dbb35de6b68552ee08bd"
    )
    fixture["provenance"]["upstream_sparse_rotary_executed"]
      .as_bool.should be_true
    fixture["output"]["qkv_features_f32le_sha256"].as_s.should eq(
      "b0ab845a95d429b3f97bcf831f97c0e378c8624404f4769c8e65489b9b5924cd"
    )
    qkv = sparse_self_attention_rope_input(fixture)
    before = qkv.features_copy
    rope_freq = fixture["input"]["rope_freq"]

    output = ML::Sparse::TensorCPU.apply_self_attention_rope(
      qkv,
      rope_freq[0].as_f.to_f32,
      rope_freq[1].as_f.to_f32
    )
    expected = flatten_sparse_self_attention_rope_f32(
      fixture["output"]["qkv_features"]
    )

    output.coordinate_map.same?(qkv.coordinate_map).should be_true
    output.shape.should eq(qkv.shape)
    output.max_feature_bytes.should eq(qkv.max_feature_bytes)
    output.features_copy.zip(expected).each do |actual, wanted|
      actual.should be_close(wanted, SPARSE_SELF_ATTENTION_ROPE_TOLERANCE)
    end
    qkv.features_copy.should eq(before)

    row_width = 3 * qkv.channels
    qkv.point_count.times do |row|
      qkv.channels.times do |channel|
        value_index = row * row_width + 2 * qkv.channels + channel
        output.features_copy[value_index].should eq(before[value_index])
      end
      2.times do |component|
        qkv.num_heads.times do |head|
          (12...qkv.head_dim).each do |channel|
            index = row * row_width + component * qkv.channels +
                    head * qkv.head_dim + channel
            output.features_copy[index].should eq(before[index])
          end
        end
      end
    end

    attention = ML::Sparse::TensorCPU.apply_full_self_attention(output)
    attention.coordinate_map.same?(qkv.coordinate_map).should be_true
    attention.features_copy.each(&.finite?.should(be_true))
  end

  it "uses x, y, z in axis-major adjacent real-pair order" do
    map = ML::Sparse::CoordinateMap3D.new(
      [0, 1, 2, 3] of Int32,
      1,
      {2, 3, 4}
    )
    values = [] of Float32
    2.times { values.concat([1.0_f32, 0.0_f32] * 3) }
    values.concat([11.0_f32, 12.0_f32, 13.0_f32, 14.0_f32, 15.0_f32, 16.0_f32])
    qkv = ML::Sparse::SelfAttentionQKVCPU.new(
      ML::Sparse::TensorCPU.new(
        ML::Tensor.from_array(values, ML::Shape.new(1_i32, 18_i32)),
        map
      ),
      1
    )

    output = ML::Sparse::TensorCPU.apply_self_attention_rope(
      qkv,
      1.0_f32,
      10000.0_f32
    )

    2.times do |component|
      3.times do |axis|
        angle = (axis + 1).to_f64
        output.feature(0, component, 0, axis * 2).should be_close(
          Math.cos(angle).to_f32,
          2.0e-6_f32
        )
        output.feature(0, component, 0, axis * 2 + 1).should be_close(
          Math.sin(angle).to_f32,
          2.0e-6_f32
        )
      end
    end
    output.features_copy[-6, 6].should eq(values[-6, 6])
  end

  it "ignores the batch column and shares phases across heads" do
    map = ML::Sparse::CoordinateMap3D.new(
      [0, 2, 1, 3, 1, 2, 1, 3] of Int32,
      2,
      {3, 2, 4}
    )
    row = Array(Float32).new(36) { |index| (index - 18).to_f32 / 5.0_f32 }
    qkv = ML::Sparse::SelfAttentionQKVCPU.new(
      ML::Sparse::TensorCPU.new(
        ML::Tensor.from_array(row + row, ML::Shape.new(2_i32, 36_i32)),
        map
      ),
      2
    )

    output = ML::Sparse::TensorCPU.apply_self_attention_rope(qkv)

    output.features_copy[0, 36].should eq(output.features_copy[36, 36])
  end

  it "preserves all-empty packed values under the local CPU policy" do
    map = ML::Sparse::CoordinateMap3D.new([] of Int32, 3, {1, 1, 1})
    qkv = ML::Sparse::SelfAttentionQKVCPU.new(
      ML::Sparse::TensorCPU.new(
        ML::Tensor.from_array([] of Float32, ML::Shape.new(0_i32, 18_i32)),
        map
      ),
      1
    )

    output = ML::Sparse::TensorCPU.apply_self_attention_rope(qkv)

    output.coordinate_map.same?(map).should be_true
    output.shape.should eq({3_i32, 3_i32, 1_i32, 6_i32})
    output.features_copy.should be_empty
  end

  it "preserves upstream all-identity floor behavior for D=2 and D=4" do
    map = ML::Sparse::CoordinateMap3D.new([0, 2, 3, 4] of Int32, 1, {3, 4, 5})
    {2_i32, 4_i32}.each do |head_dim|
      values = Array(Float32).new(3 * head_dim) do |index|
        (index - head_dim).to_f32 / 3.0_f32
      end
      qkv = ML::Sparse::SelfAttentionQKVCPU.new(
        ML::Sparse::TensorCPU.new(
          ML::Tensor.from_array(
            values,
            ML::Shape.new(1_i32, 3_i32 * head_dim)
          ),
          map
        ),
        1
      )

      output = ML::Sparse::TensorCPU.apply_self_attention_rope(qkv)

      output.features_copy.should eq(values)
      output.coordinate_map.same?(map).should be_true
    end
  end

  it "rejects odd head dimensions and invalid frequency ranges" do
    map = ML::Sparse::CoordinateMap3D.new([0, 0, 0, 0] of Int32, 1, {1, 1, 1})
    odd = ML::Sparse::SelfAttentionQKVCPU.new(
      ML::Sparse::TensorCPU.new(
        ML::Tensor.from_array(
          Array(Float32).new(9, 1.0_f32),
          ML::Shape.new(1_i32, 9_i32)
        ),
        map
      ),
      1
    )
    expect_raises(ML::Sparse::SparseTensorError, /head dimension.*even/) do
      ML::Sparse::TensorCPU.apply_self_attention_rope(odd)
    end

    even = ML::Sparse::SelfAttentionQKVCPU.new(
      ML::Sparse::TensorCPU.new(
        ML::Tensor.from_array(
          Array(Float32).new(18, 1.0_f32),
          ML::Shape.new(1_i32, 18_i32)
        ),
        map
      ),
      1
    )
    {0.0_f32, -1.0_f32, Float32::NAN}.each do |invalid|
      expect_raises(ML::Sparse::SparseTensorError, /frequencies.*positive/) do
        ML::Sparse::TensorCPU.apply_self_attention_rope(even, invalid, 10000.0_f32)
      end
    end
    expect_raises(ML::Sparse::SparseTensorError, /frequencies.*positive/) do
      ML::Sparse::TensorCPU.apply_self_attention_rope(even, 1.0_f32, Float32::INFINITY)
    end
  end

  it "rejects non-finite phases before returning an output" do
    map = ML::Sparse::CoordinateMap3D.new([0, 2, 0, 0] of Int32, 1, {3, 1, 1})
    qkv = ML::Sparse::SelfAttentionQKVCPU.new(
      ML::Sparse::TensorCPU.new(
        ML::Tensor.from_array(
          Array(Float32).new(18, 1.0_f32),
          ML::Shape.new(1_i32, 18_i32)
        ),
        map
      ),
      1
    )

    expect_raises(ML::Sparse::SparseTensorError, /RoPE angle.*finite/) do
      ML::Sparse::TensorCPU.apply_self_attention_rope(
        qkv,
        Float32::MAX,
        1.0_f32
      )
    end
  end

  it "requires canonical packed input and the base receiver" do
    qkv = sparse_self_attention_rope_input(sparse_self_attention_rope_fixture)
    inherited = SparseSelfAttentionRoPEQKVOverride.new(
      qkv.flat_projection,
      qkv.num_heads
    )
    expect_raises(ML::Sparse::SparseTensorError, /base SelfAttentionQKVCPU/) do
      ML::Sparse::TensorCPU.apply_self_attention_rope(inherited)
    end
    expect_raises(ML::Sparse::SparseTensorError, /base TensorCPU receiver/) do
      SparseSelfAttentionRoPETensorCPUOverride.apply_self_attention_rope(qkv)
    end
  end
end
