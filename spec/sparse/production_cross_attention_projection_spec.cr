require "../../src/ml/sparse/cross_attention_projection"
require "../spec_helper"

private PRODUCTION_CROSS_CHANNELS         =  1_536_i32
private PRODUCTION_CROSS_CONTEXT_CHANNELS =  1_024_i32
private PRODUCTION_CROSS_CONTEXT_LENGTH   =      3_i32
private PRODUCTION_CROSS_HEADS            =     12_i32
private PRODUCTION_CROSS_HEAD_DIM         =    128_i32
private PRODUCTION_CROSS_TOLERANCE        = 5.0e-5_f32

private def production_cross_map : ML::Sparse::CoordinateMap3D
  ML::Sparse::CoordinateMap3D.new(
    [
      0, 0, 0, 0,
      0, 0, 0, 1,
      2, 0, 0, 0,
      2, 0, 0, 1,
    ] of Int32,
    3,
    {1, 1, 2}
  )
end

private def production_cross_query_values : Array(Float32)
  Array(Float32).new(4 * PRODUCTION_CROSS_CHANNELS) do |index|
    row = index // PRODUCTION_CROSS_CHANNELS
    channel = index % PRODUCTION_CROSS_CHANNELS
    ((row + 1) * 0.125 + (channel % 17) * 0.01).to_f32
  end
end

private def production_cross_context_values : Array(Float32)
  Array(Float32).new(
    3 * PRODUCTION_CROSS_CONTEXT_LENGTH * PRODUCTION_CROSS_CONTEXT_CHANNELS
  ) do |index|
    token_index = index // PRODUCTION_CROSS_CONTEXT_CHANNELS
    batch = token_index // PRODUCTION_CROSS_CONTEXT_LENGTH
    token = token_index % PRODUCTION_CROSS_CONTEXT_LENGTH
    channel = index % PRODUCTION_CROSS_CONTEXT_CHANNELS
    ((batch + 1) * 0.2 + (token + 1) * 0.03 +
      (channel % 13) * 0.015).to_f32
  end
end

private def production_cross_query(
  values : Array(Float32),
  map : ML::Sparse::CoordinateMap3D,
) : ML::Sparse::TensorCPU
  ML::Sparse::TensorCPU.production(
    ML::Tensor.from_array(
      values,
      ML::Shape.new(4, PRODUCTION_CROSS_CHANNELS)
    ),
    map
  )
end

private def production_cross_context(values : Array(Float32)) : ML::Tensor
  ML::Tensor.from_array(
    values,
    ML::Shape.new(
      3,
      PRODUCTION_CROSS_CONTEXT_LENGTH,
      PRODUCTION_CROSS_CONTEXT_CHANNELS
    )
  )
end

private def production_cross_to_q : ML::NN::Linear
  layer = ML::NN::Linear.new(
    PRODUCTION_CROSS_CHANNELS,
    PRODUCTION_CROSS_CHANNELS,
    device: ML::Tensor::Device::CPU
  )
  weight = layer.weight.data.cpu_data.not_nil!
  weight.fill(0.0_f32)
  bias = layer.bias.not_nil!.data.cpu_data.not_nil!
  PRODUCTION_CROSS_CHANNELS.times do |output|
    first = (output * 17 + 3) % PRODUCTION_CROSS_CHANNELS
    second = (output * 29 + 5) % PRODUCTION_CROSS_CHANNELS
    weight[output * PRODUCTION_CROSS_CHANNELS + first] = 0.5_f32
    weight[output * PRODUCTION_CROSS_CHANNELS + second] = -0.25_f32
    bias[output] = ((output % 7) - 3).to_f32 * 0.01_f32
  end
  layer.weight.requires_grad = false
  layer.bias.not_nil!.requires_grad = false
  layer
end

private def production_cross_to_kv : ML::NN::Linear
  layer = ML::NN::Linear.new(
    PRODUCTION_CROSS_CONTEXT_CHANNELS,
    PRODUCTION_CROSS_CHANNELS * 2,
    device: ML::Tensor::Device::CPU
  )
  weight = layer.weight.data.cpu_data.not_nil!
  weight.fill(0.0_f32)
  bias = layer.bias.not_nil!.data.cpu_data.not_nil!
  (PRODUCTION_CROSS_CHANNELS * 2).times do |output|
    first = (output * 11 + 1) % PRODUCTION_CROSS_CONTEXT_CHANNELS
    second = (output * 23 + 7) % PRODUCTION_CROSS_CONTEXT_CHANNELS
    weight[output * PRODUCTION_CROSS_CONTEXT_CHANNELS + first] = -0.375_f32
    weight[output * PRODUCTION_CROSS_CONTEXT_CHANNELS + second] = 0.625_f32
    bias[output] = ((output % 9) - 4).to_f32 * 0.005_f32
  end
  layer.weight.requires_grad = false
  layer.bias.not_nil!.requires_grad = false
  layer
end

# The source-bound T2N5l fixture proves the upstream affine formula and layout
# at toy width. This falsifier isolates production dimensions and indexing with
# dense parameter storage but analytically checkable two-tap rows; it is not a
# real-checkpoint or long dense-accumulation parity claim.
describe "TRELLIS.2 production-width sparse cross-attention projection" do
  it "executes synthetic L=3 exact-width Q/KV arithmetic without padded query rows" do
    map = production_cross_map
    query_values = production_cross_query_values
    context_values = production_cross_context_values
    query = production_cross_query(query_values, map)
    context = production_cross_context(context_values)
    to_q = production_cross_to_q
    to_kv = production_cross_to_kv

    projection = ML::Sparse::CrossAttentionProjectionCPU.project(
      query,
      context,
      PRODUCTION_CROSS_HEADS,
      to_q,
      to_kv
    )

    projection.coordinate_map.same?(map).should be_true
    projection.query_feature_shape.should eq({4, PRODUCTION_CROSS_CHANNELS})
    projection.context_kv_feature_shape.should eq({
      3,
      PRODUCTION_CROSS_CONTEXT_LENGTH,
      PRODUCTION_CROSS_CHANNELS * 2,
    })
    projection.query_batch_slice(0).should eq(ML::Sparse::BatchSlice.new(0, 2))
    projection.query_batch_slice(1).should eq(ML::Sparse::BatchSlice.new(2, 2))
    projection.query_batch_slice(2).should eq(ML::Sparse::BatchSlice.new(2, 4))
    projection.query_features_copy.size.should eq(4 * PRODUCTION_CROSS_CHANNELS)
    projection.context_kv_features_copy.size.should eq(
      3 * PRODUCTION_CROSS_CONTEXT_LENGTH * PRODUCTION_CROSS_CHANNELS * 2
    )

    plan = projection.plan
    plan.score_bytes.should eq(576_i64)
    plan.projection_bytes.should eq(135_168_i64)
    plan.output_bytes.should eq(24_576_i64)
    plan.query_projection_mac_elements.should eq(9_437_184_i64)
    plan.context_kv_projection_mac_elements.should eq(28_311_552_i64)
    plan.work_elements.should eq(47_222_784_i64)

    4.times do |row|
      PRODUCTION_CROSS_CHANNELS.times do |output|
        first = (output * 17 + 3) % PRODUCTION_CROSS_CHANNELS
        second = (output * 29 + 5) % PRODUCTION_CROSS_CHANNELS
        expected = query_values[row * PRODUCTION_CROSS_CHANNELS + first] * 0.5_f32 +
                   query_values[row * PRODUCTION_CROSS_CHANNELS + second] * -0.25_f32 +
                   ((output % 7) - 3).to_f32 * 0.01_f32
        projection.query_feature(
          row,
          output // PRODUCTION_CROSS_HEAD_DIM,
          output % PRODUCTION_CROSS_HEAD_DIM
        ).should be_close(expected, PRODUCTION_CROSS_TOLERANCE)
      end
    end

    3.times do |batch|
      PRODUCTION_CROSS_CONTEXT_LENGTH.times do |token|
        (PRODUCTION_CROSS_CHANNELS * 2).times do |output|
          first = (output * 11 + 1) % PRODUCTION_CROSS_CONTEXT_CHANNELS
          second = (output * 23 + 7) % PRODUCTION_CROSS_CONTEXT_CHANNELS
          input_offset = (batch * PRODUCTION_CROSS_CONTEXT_LENGTH + token) *
                         PRODUCTION_CROSS_CONTEXT_CHANNELS
          expected = context_values[input_offset + first] * -0.375_f32 +
                     context_values[input_offset + second] * 0.625_f32 +
                     ((output % 9) - 4).to_f32 * 0.005_f32
          component = output // PRODUCTION_CROSS_CHANNELS
          within_component = output % PRODUCTION_CROSS_CHANNELS
          projection.context_kv_feature(
            batch,
            token,
            component,
            within_component // PRODUCTION_CROSS_HEAD_DIM,
            within_component % PRODUCTION_CROSS_HEAD_DIM
          ).should be_close(expected, PRODUCTION_CROSS_TOLERANCE)
        end
      end
    end

    query.features_copy.should eq(query_values)
    context.cpu_data.not_nil!.should eq(context_values)
    to_q.weight.requires_grad?.should be_false
    to_kv.weight.requires_grad?.should be_false
  end

  it "rejects the exact production work envelope before payload or parameter reads" do
    map = production_cross_map
    query = production_cross_query(production_cross_query_values, map)
    context = production_cross_context(production_cross_context_values)
    context.cpu_data.not_nil![0] = Float32::NAN
    to_q = ML::NN::Linear.new(1, 1, device: ML::Tensor::Device::CPU)
    to_kv = ML::NN::Linear.new(1, 2, device: ML::Tensor::Device::CPU)

    expect_raises(
      ML::Sparse::SparseTensorBudgetError,
      /require 47222784 MAC elements, limit is 47222783/
    ) do
      ML::Sparse::CrossAttentionProjectionCPU.project(
        query,
        context,
        PRODUCTION_CROSS_HEADS,
        to_q,
        to_kv,
        max_work_elements: 47_222_783_i64
      )
    end
  end
end
