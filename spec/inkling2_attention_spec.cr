require "./spec_helper"
require "../src/ml/gguf/inkling2_weights"
require "../src/ml/gguf/inkling2_matmul"
require "../src/ml/gguf/inkling2_attention"

private def tiny_inkling_config(
  *,
  n_head : Int32 = 1,
  n_head_kv : Int32 = 1,
  n_embd : Int32 = 1,
  rel_extent : Int32 = 2,
  sliding_window : Int32 = 2,
  conv_size : Int32 = 2,
  sliding : Bool = true,
  log_scale_floor : Int32 = 1,
  log_scale_alpha : Float32 = 0_f32,
  rel_clip : Float32 = 10_f32,
) : ML::GGUF::Inkling2Config
  ML::GGUF::Inkling2Config.new(
    n_layer: 1_i32,
    n_embd: n_embd,
    n_head: n_head,
    n_head_kv: n_head_kv,
    head_dim: 1_i32,
    rel_dim: 1_i32,
    rel_extent: rel_extent,
    sliding_window: sliding_window,
    conv_size: conv_size,
    log_scale_floor: log_scale_floor,
    log_scale_alpha: log_scale_alpha,
    qk_clip: 10_f32,
    rel_clip: rel_clip,
    rms_eps: 1e-5_f32,
    n_expert: 1_i32,
    n_expert_used: 1_i32,
    expert_dim: 1_i32,
    expert_scale: 1_f32,
    situ_beta: 1_f32,
    situ_linear_beta: 1_f32,
    res_block_size: 1_i32,
    vocab_size: 2_i32,
    padded_vocab_size: 2_i32,
    context_length: 32_i32,
    core_start: 0_i32,
    core_end: 1_i32,
    passes: 3_i32,
    gate_rank: 1_i32,
    logit_softcap: 30_f32,
    varying: false,
    sliding_pattern: [sliding],
  )
end

private def f32_weight(label : String, out_dim : Int32, in_dim : Int32, values : Array(Float32)) : ML::GGUF::QuantWeight
  raise "bad test matrix size for #{label}" unless values.size == out_dim * in_dim
  bytes = Bytes.new(values.size * sizeof(Float32), 0_u8)
  bytes.to_unsafe.as(Float32*).copy_from(values.to_unsafe, values.size)
  ML::GGUF::QuantWeight.new(bytes, ML::GGUF::TensorType::F32, out_dim, in_dim, "test:#{label}")
end

private def zero_matrix(rows : Int32, columns : Int32) : Array(Float32)
  Array(Float32).new(rows * columns, 0_f32)
end

private def test_attention_weights(
  config : ML::GGUF::Inkling2Config,
  *,
  q_values : Array(Float32)? = nil,
  k_values : Array(Float32)? = nil,
  v_values : Array(Float32)? = nil,
  output_values : Array(Float32)? = nil,
  gate_values : Array(Float32)? = nil,
  sink_values : Array(Float32)? = nil,
  rel_values : Array(Float32)? = nil,
  rel_bias : Array(Float32)? = nil,
  q_clip : Array(Float32)? = nil,
  rel_clip : Array(Float32)? = nil,
  v_conv_weights : Array(Float32)? = nil,
) : ML::GGUF::Inkling2AttentionWeights
  q_width = config.n_head * config.head_dim
  kv_width = config.n_head_kv * config.head_dim
  model_width = config.n_embd
  extent = config.sliding?(0) ? config.sliding_window : config.rel_extent

  output = output_values || zero_matrix(model_width, q_width)
  [model_width, q_width].min.times { |index| output[index * q_width + index] = 1_f32 } unless output_values

  k_projection = if config.sliding?(0)
                   nil
                 else
                   f32_weight("k_conv_proj", config.conv_size * config.n_head_kv, model_width,
                     zero_matrix(config.conv_size * config.n_head_kv, model_width))
                 end
  v_projection = if config.sliding?(0)
                   nil
                 else
                   f32_weight("v_conv_proj", config.conv_size * config.n_head_kv, model_width,
                     zero_matrix(config.conv_size * config.n_head_kv, model_width))
                 end

  ML::GGUF::Inkling2AttentionWeights.new(
    q: f32_weight("q", q_width, model_width, q_values || zero_matrix(q_width, model_width)),
    k: f32_weight("k", kv_width, model_width, k_values || zero_matrix(kv_width, model_width)),
    v: f32_weight("v", kv_width, model_width, v_values || zero_matrix(kv_width, model_width)),
    output: f32_weight("output", model_width, q_width, output),
    gate: f32_weight("gate", q_width, model_width, gate_values || zero_matrix(q_width, model_width)),
    sink: f32_weight("sink", config.n_head, model_width, sink_values || zero_matrix(config.n_head, model_width)),
    rel_proj: f32_weight("rel", config.n_head * config.rel_dim, model_width,
      rel_values || zero_matrix(config.n_head * config.rel_dim, model_width)),
    q_clip: q_clip || Array(Float32).new(q_width, 1_f32),
    rel_bias: rel_bias || Array(Float32).new(extent * config.rel_dim, 0_f32),
    rel_clip: rel_clip || Array(Float32).new(config.n_head * config.rel_dim, 1_f32),
    k_conv: Array(Float32).new(config.conv_size * kv_width, 0_f32),
    v_conv: v_conv_weights || Array(Float32).new(config.conv_size * kv_width, 0_f32),
    post_conv: Array(Float32).new(config.conv_size * model_width, 0_f32),
    k_conv_proj: k_projection,
    v_conv_proj: v_projection,
  )
end

private def forward_attention(
  weights : ML::GGUF::Inkling2AttentionWeights,
  config : ML::GGUF::Inkling2Config,
  state : ML::GGUF::Inkling2AttentionState,
  q_in : Array(Float32),
  k_in : Array(Float32),
  v_in : Array(Float32),
  position : Int32,
) : Array(Float32)
  ML::GGUF::Inkling2Attention.forward(weights, config, state, ML::GGUF::Inkling2CPUMatmul.new,
    q_in, k_in, v_in, position)
end

describe ML::GGUF::Inkling2Convolution do
  it "uses opposing static and dynamic tap orientation" do
    dynamic_conv = ML::GGUF::Inkling2Convolution.new(1_i32, 3_i32)
    dynamic_weights = [0_f32, 0_f32, 0_f32]
    dynamic_taps = [1_f32, 10_f32, 100_f32]
    dynamic_out = [2_f32, 3_f32, 5_f32].map do |value|
      dynamic_conv.apply([value], dynamic_weights, dynamic_taps)[0]
    end
    root7 = Math.sqrt(7.0)
    (dynamic_out[0] - 4.0 / root7).abs.should be < 1e-5
    (dynamic_out[1] - 26.0 / root7).abs.should be < 1e-5
    (dynamic_out[2] - 240.0 / root7).abs.should be < 1e-5

    static_conv = ML::GGUF::Inkling2Convolution.new(1_i32, 3_i32)
    static_weights = [1_f32, 10_f32, 100_f32]
    static_out = [2_f32, 3_f32, 5_f32].map do |value|
      static_conv.apply([value], static_weights)[0]
    end
    (static_out[0] - 101.0).abs.should be < 1e-5
    (static_out[1] - 161.5).abs.should be < 1e-5
    (static_out[2] - 268.5).abs.should be < 1e-5

    grouped = ML::GGUF::Inkling2Convolution.new(4_i32, 2_i32)
    grouped_weights = Array(Float32).new(8, 0_f32)
    grouped.apply([1_f32, 2_f32, 3_f32, 4_f32], grouped_weights, [1_f32, 0_f32, 10_f32, 0_f32])
    grouped_out = grouped.apply([10_f32, 20_f32, 30_f32, 40_f32], grouped_weights, [1_f32, 2_f32, 10_f32, 20_f32])
    root5 = Math.sqrt(5.0)
    (grouped_out[0] - 22.0 / root5).abs.should be < 1e-5
    (grouped_out[1] - 44.0 / root5).abs.should be < 1e-5
    (grouped_out[2] - 390.0 / root5).abs.should be < 1e-5
    (grouped_out[3] - 520.0 / root5).abs.should be < 1e-5
  end
end

describe ML::GGUF::Inkling2Attention do
  it "maps two query heads onto one shared KV head" do
    config = tiny_inkling_config(n_head: 2_i32, n_head_kv: 1_i32, n_embd: 2_i32)
    q = [1_f32, 0_f32, 0_f32, -1_f32]
    k = [1_f32, 0_f32]
    v = [0_f32, 1_f32]
    weights = test_attention_weights(config, q_values: q, k_values: k, v_values: v)
    state = ML::GGUF::Inkling2AttentionState.new(config, true, 8_i32)

    forward_attention(weights, config, state, [1_f32, 1_f32], [1_f32, 0_f32], [0_f32, 1_f32], 0_i32)
    output = forward_attention(weights, config, state, [1_f32, 1_f32], [-1_f32, 0_f32], [0_f32, -1_f32], 1_i32)

    output[0].should be > 0.01_f32
    output[1].should be < -0.01_f32
    (output[0] + output[1]).abs.should be < 1e-5_f32
  end

  it "normalizes the learned sink against one real key" do
    config = tiny_inkling_config
    weights = test_attention_weights(config, v_values: [1_f32], sink_values: [Math.log(3.0).to_f32])
    state = ML::GGUF::Inkling2AttentionState.new(config, true, 8_i32)
    output = forward_attention(weights, config, state, [1_f32], [0_f32], [1_f32], 0_i32)

    expected = (0.125 / Math.sqrt(3.0)).to_f32
    (output[0] - expected).abs.should be < 1e-5_f32
  end

  it "clips projected relative bias and uses zero bias beyond the configured extent" do
    config = tiny_inkling_config(sliding: false, rel_extent: 2_i32, rel_clip: 0.5_f32)
    weights = test_attention_weights(
      config,
      v_values: [1_f32],
      rel_values: [10_f32],
      rel_bias: [0_f32, 2_f32],
    )
    state = ML::GGUF::Inkling2AttentionState.new(config, false, 4_i32)
    forward_attention(weights, config, state, [1_f32], [1_f32], [1_f32], 0_i32)
    forward_attention(weights, config, state, [1_f32], [1_f32], [-1_f32], 1_i32)
    output = forward_attention(weights, config, state, [1_f32], [1_f32], [1_f32], 2_i32)

    e = Math.exp(1.0)
    expected = ((2.0 - e) / (3.0 + e) * 0.5 / Math.sqrt(3.0)).to_f32
    (output[0] - expected).abs.should be < 2e-5_f32
  end

  it "excludes a retained key exactly on the sliding-window boundary" do
    config = tiny_inkling_config
    weights = test_attention_weights(config, v_values: [1_f32])
    state = ML::GGUF::Inkling2AttentionState.new(config, true, 8_i32)
    forward_attention(weights, config, state, [1_f32], [0_f32], [0_f32], 0_i32)
    forward_attention(weights, config, state, [1_f32], [0_f32], [1_f32], 1_i32)
    output = forward_attention(weights, config, state, [1_f32], [0_f32], [0_f32], 3_i32)

    output[0].abs.should be < 1e-6_f32
  end

  it "applies log-scaled tau to full-attention Q and relative projections" do
    alpha = (1.0 / Math.log(2.0)).to_f32
    config = tiny_inkling_config(sliding: false, log_scale_alpha: alpha)
    weights = test_attention_weights(
      config,
      q_values: [1_f32],
      k_values: [1_f32],
      v_values: [1_f32],
      rel_values: [1_f32],
      rel_bias: [0_f32, 1_f32],
    )
    state = ML::GGUF::Inkling2AttentionState.new(config, false, 4_i32)
    forward_attention(weights, config, state, [1_f32], [1_f32], [1_f32], 0_i32)
    output = forward_attention(weights, config, state, [1_f32], [-1_f32], [-1_f32], 1_i32)

    scale = 2.0 / Math.sqrt(5.0)
    score_past = 2.0 + scale
    score_current = -scale
    numerator = Math.exp(score_past) - Math.exp(score_current)
    denominator = 1.0 + Math.exp(score_past) + Math.exp(score_current)
    expected = (numerator / denominator * 0.5 / Math.sqrt(3.0)).to_f32
    (output[0] - expected).abs.should be < 2e-5_f32
  end

  it "keeps convolution and KV state independent across state objects" do
    config = tiny_inkling_config
    weights = test_attention_weights(config, v_values: [1_f32], v_conv_weights: [1_f32, 0_f32])
    first = ML::GGUF::Inkling2AttentionState.new(config, true, 8_i32)
    interleaved = ML::GGUF::Inkling2AttentionState.new(config, true, 8_i32)
    control = ML::GGUF::Inkling2AttentionState.new(config, true, 8_i32)

    forward_attention(weights, config, first, [1_f32], [0_f32], [1_f32], 0_i32)
    forward_attention(weights, config, control, [1_f32], [0_f32], [1_f32], 0_i32)
    forward_attention(weights, config, interleaved, [1_f32], [0_f32], [-1_f32], 0_i32)
    first_output = forward_attention(weights, config, first, [1_f32], [0_f32], [0_f32], 1_i32)
    control_output = forward_attention(weights, config, control, [1_f32], [0_f32], [0_f32], 1_i32)

    (first_output[0] - control_output[0]).abs.should be < 1e-6_f32
  end

  it "bounds full-attention cache growth by max_seq" do
    config = tiny_inkling_config(sliding: false)
    weights = test_attention_weights(config)
    state = ML::GGUF::Inkling2AttentionState.new(config, false, 1_i32)
    forward_attention(weights, config, state, [1_f32], [0_f32], [0_f32], 0_i32)

    expect_raises(ArgumentError) do
      forward_attention(weights, config, state, [1_f32], [0_f32], [0_f32], 1_i32)
    end
  end
end
