require "../../src/ml/sparse/adaptive_layer_norm"
require "../../src/ml/sparse/cross_attention_plan"
require "../../src/ml/sparse/full_self_attention"
require "../../src/ml/sparse/gated_residual"
require "../../src/ml/sparse/ops"
require "../../src/ml/sparse/self_attention_qk_rms_norm"
require "../../src/ml/sparse/self_attention_qkv"
require "../../src/ml/sparse/self_attention_rope"
require "../spec_helper"

private def freeze_production_carrier_linear(layer : ML::NN::Linear) : Nil
  layer.weight.requires_grad = false
  layer.bias.try { |bias| bias.requires_grad = false }
end

private def empty_production_carrier(channels : Int32) : ML::Sparse::TensorCPU
  map = ML::Sparse::CoordinateMap3D.new(
    [] of Int32,
    1,
    {1, 1, 1}
  )
  ML::Sparse::TensorCPU.production(
    ML::Tensor.from_array(
      [] of Float32,
      ML::Shape.new(0_i32, channels)
    ),
    map
  )
end

class ProductionCarrierReceiverOverride < ML::Sparse::TensorCPU
end

describe "TRELLIS.2 production-width sparse carrier" do
  it "keeps the legacy constructor bounded and admits width through explicit authority" do
    ML::Sparse::TensorCPU::MAX_CHANNELS.should eq(256)
    ML::Sparse::TensorCPU::PRODUCTION_MAX_CHANNELS.should eq(1_536)
    ML::Sparse::TensorCPU::MAX_SELF_ATTENTION_QKV_CHANNELS.should eq(4_608)
    ML::Sparse::TensorCPU::MAX_FEATURE_BYTES.should eq(
      64_i64 * 1024_i64 * 1024_i64
    )

    map = ML::Sparse::CoordinateMap3D.new(
      [0, 0, 0, 0] of Int32,
      1,
      {1, 1, 1}
    )
    source = ML::Tensor.zeros(
      1,
      1_536,
      device: ML::Tensor::Device::CPU
    )
    expect_raises(
      ML::Sparse::SparseTensorError,
      /channel count must be in 1\.\.256/
    ) do
      ML::Sparse::TensorCPU.new(source, map)
    end
    expect_raises(ML::Sparse::SparseTensorError, /base TensorCPU receiver/) do
      ProductionCarrierReceiverOverride.production(source, map)
    end

    carrier = ML::Sparse::TensorCPU.production(source, map, 6_144_i64)
    carrier.channels.should eq(1_536)
    carrier.point_count.should eq(1)
    carrier.coordinate_map.same?(map).should be_true
    carrier.features_copy.size.should eq(1_536)
    carrier.max_feature_bytes.should eq(6_144_i64)
    carrier.production_width?.should be_true
    carrier.packed_qkv?.should be_false

    expect_raises(
      ML::Sparse::SparseTensorError,
      /production sparse feature channel count must be in 1\.\.1536/
    ) do
      ML::Sparse::TensorCPU.production(
        ML::Tensor.zeros(1, 1_537, device: ML::Tensor::Device::CPU),
        map
      )
    end
  end

  it "preserves production authority through generic sparse leaves" do
    empty_map = ML::Sparse::CoordinateMap3D.new(
      [] of Int32,
      1,
      {1, 1, 1}
    )
    left = ML::Sparse::TensorCPU.production(
      ML::Tensor.from_array(
        [] of Float32,
        ML::Shape.new(0_i32, 768_i32)
      ),
      empty_map
    )
    right = ML::Sparse::TensorCPU.production(
      ML::Tensor.from_array(
        [] of Float32,
        ML::Shape.new(0_i32, 768_i32)
      ),
      empty_map
    )
    concatenated = ML::Sparse::TensorCPU.concat_features(left, right)
    concatenated.channels.should eq(1_536)
    concatenated.production_width?.should be_true
    concatenated.coordinate_map.same?(empty_map).should be_true

    narrow = ML::Sparse::TensorCPU.production(
      ML::Tensor.from_array(
        [] of Float32,
        ML::Shape.new(0_i32, 1_i32)
      ),
      empty_map
    )
    exact = ML::NN::Linear.new(
      1,
      1_536,
      device: ML::Tensor::Device::CPU
    )
    freeze_production_carrier_linear(exact)
    projected = ML::Sparse::TensorCPU.apply_linear(narrow, exact)
    projected.channels.should eq(1_536)
    projected.production_width?.should be_true

    bounded = ML::Sparse::TensorCPU.new(
      ML::Tensor.from_array(
        [] of Float32,
        ML::Shape.new(0_i32, 1_i32)
      ),
      empty_map
    )
    expect_raises(ML::Sparse::SparseTensorError, /same carrier role/) do
      ML::Sparse::TensorCPU.concat_features(narrow, bounded)
    end
  end

  it "transports the ragged production seam through norm, residual, and cross preflight" do
    map = ML::Sparse::CoordinateMap3D.new(
      [
        0, 0, 0, 0,
        0, 0, 0, 1,
        2, 0, 0, 0,
        2, 0, 0, 1,
      ] of Int32,
      3,
      {1, 1, 2}
    )
    carrier = ML::Sparse::TensorCPU.production(
      ML::Tensor.ones(4, 1_536, device: ML::Tensor::Device::CPU),
      map
    )
    adaptive = ML::Sparse::TensorCPU.apply_adaptive_layer_norm(
      carrier,
      ML::Tensor.zeros(3, 1_536, device: ML::Tensor::Device::CPU),
      ML::Tensor.zeros(3, 1_536, device: ML::Tensor::Device::CPU)
    )
    adaptive.channels.should eq(1_536)
    adaptive.coordinate_map.same?(map).should be_true
    adaptive.production_width?.should be_true
    adaptive.features_copy.all?(&.finite?).should be_true

    residual = ML::Sparse::TensorCPU.apply_gated_residual(
      carrier,
      adaptive,
      ML::Tensor.zeros(3, 1_536, device: ML::Tensor::Device::CPU)
    )
    residual.channels.should eq(1_536)
    residual.coordinate_map.same?(map).should be_true
    residual.production_width?.should be_true

    plan = ML::Sparse::CrossAttentionPlanCPU.preflight(
      residual,
      ML::Tensor.ones(3, 1, 1_024, device: ML::Tensor::Device::CPU),
      12
    )
    plan.query_channels.should eq(1_536)
    plan.context_channels.should eq(1_024)
    plan.max_query_length.should eq(2)
    plan.coordinate_map.same?(map).should be_true
    plan.output_elements.should eq(6_144_i64)
  end

  it "keeps exact production QKV packed until attention collapses it" do
    carrier = empty_production_carrier(1_536)
    qkv_layer = ML::NN::Linear.new(
      1_536,
      4_608,
      bias: false,
      device: ML::Tensor::Device::CPU
    )
    freeze_production_carrier_linear(qkv_layer)

    one_point_map = ML::Sparse::CoordinateMap3D.new(
      [0, 0, 0, 0] of Int32,
      1,
      {1, 1, 1}
    )
    budget_tight = ML::Sparse::TensorCPU.production(
      ML::Tensor.zeros(1, 1_536, device: ML::Tensor::Device::CPU),
      one_point_map,
      6_144_i64
    )
    expect_raises(
      ML::Sparse::SparseTensorBudgetError,
      /output features require 18432 bytes, limit is 6144/
    ) do
      ML::Sparse::TensorCPU.apply_self_attention_qkv(
        budget_tight,
        qkv_layer,
        12
      )
    end

    qkv = ML::Sparse::TensorCPU.apply_self_attention_qkv(
      carrier,
      qkv_layer,
      12
    )
    qkv.feature_shape.should eq({0, 3, 12, 128})
    qkv.flat_projection.channels.should eq(4_608)
    qkv.flat_projection.production_width?.should be_true
    qkv.flat_projection.packed_qkv?.should be_true

    ordinary_production = empty_production_carrier(12)
    expect_raises(
      ML::Sparse::SparseTensorError,
      /requires packed production QKV storage/
    ) do
      ML::Sparse::SelfAttentionQKVCPU.new(ordinary_production, 3)
    end
    expect_raises(
      ML::Sparse::SparseTensorError,
      /requires a standard sparse carrier, not packed QKV storage/
    ) do
      ML::Sparse::FullSelfAttentionPlanCPU.preflight(
        qkv.flat_projection,
        12
      )
    end

    packed_to_standard = ML::NN::Linear.new(
      4_608,
      1,
      bias: false,
      device: ML::Tensor::Device::CPU
    )
    freeze_production_carrier_linear(packed_to_standard)
    expect_raises(ML::Sparse::SparseTensorError, /requires a standard sparse carrier/) do
      ML::Sparse::TensorCPU.apply_linear(
        qkv.flat_projection,
        packed_to_standard
      )
    end
    expect_raises(ML::Sparse::SparseTensorError, /requires a standard sparse carrier/) do
      qkv.flat_projection.replace_features(
        ML::Tensor.from_array(
          [] of Float32,
          ML::Shape.new(0_i32, 12_i32)
        )
      )
    end
    expect_raises(ML::Sparse::SparseTensorError, /requires a standard sparse carrier/) do
      qkv.flat_projection.replace_coordinates(
        [] of Int32,
        1,
        {1, 1, 1}
      )
    end
    expect_raises(ML::Sparse::SparseTensorError, /requires a standard sparse carrier/) do
      ML::Sparse::TensorCPU.concat_features(
        qkv.flat_projection,
        qkv.flat_projection
      )
    end
    expect_raises(ML::Sparse::SparseTensorError, /requires a standard sparse carrier/) do
      ML::Sparse::TensorCPU.apply_adaptive_layer_norm(
        qkv.flat_projection,
        ML::Tensor.zeros(1, 1, device: ML::Tensor::Device::CPU),
        ML::Tensor.zeros(1, 1, device: ML::Tensor::Device::CPU)
      )
    end
    expect_raises(ML::Sparse::SparseTensorError, /requires a standard sparse carrier/) do
      ML::Sparse::TensorCPU.apply_gated_residual(
        qkv.flat_projection,
        qkv.flat_projection,
        ML::Tensor.zeros(1, 1, device: ML::Tensor::Device::CPU)
      )
    end
    expect_raises(ML::Sparse::SparseTensorError, /requires a standard sparse carrier/) do
      ML::Sparse::CrossAttentionPlanCPU.preflight(
        qkv.flat_projection,
        ML::Tensor.ones(1, 1, 1_024, device: ML::Tensor::Device::CPU),
        12
      )
    end

    normalized = ML::Sparse::TensorCPU.apply_self_attention_qk_rms_norm(
      qkv,
      ML::Tensor.ones(12, 128, device: ML::Tensor::Device::CPU),
      ML::Tensor.ones(12, 128, device: ML::Tensor::Device::CPU)
    )
    normalized.flat_projection.production_width?.should be_true
    normalized.flat_projection.packed_qkv?.should be_true
    rotated = ML::Sparse::TensorCPU.apply_self_attention_rope(normalized)
    rotated.flat_projection.production_width?.should be_true
    rotated.flat_projection.packed_qkv?.should be_true

    attention = ML::Sparse::TensorCPU.apply_full_self_attention(rotated)
    attention.channels.should eq(1_536)
    attention.production_width?.should be_true
    attention.packed_qkv?.should be_false
  end
end
