require "./spec_helper"
require "../src/ml/gguf/qwen_image21_weights"
require "../src/ml/gguf/qwen_image21_metal"
require "../src/ml/gguf/qwen_image21_flow_match"
require "../src/ml/gguf/qwen_image21_conditioning_bundle"

describe ML::GGUF::QwenImage21MetalAttentionRouteTrace do
  it "aggregates the kernel name supplied by the actual dispatch site" do
    trace = ML::GGUF::QwenImage21MetalAttentionRouteTrace.new(enabled: true)
    trace.record_selected("qi21_block_causal_attention_tiled", 128, 768, 768)
    trace.record_selected("qi21_block_causal_attention_tiled", 128, 768, 768)

    trace.summary_lines.should eq([
      "qwen_image21_attention_dispatch kernel=qi21_block_causal_attention_tiled head_dim=128 total_tokens=768 query_tokens=768 dispatches=2",
    ])
  end

  it "leaves no dispatch records when disabled" do
    trace = ML::GGUF::QwenImage21MetalAttentionRouteTrace.new(enabled: false)
    trace.record_selected("qi21_block_causal_attention", 128, 768, 768)

    trace.summary_lines.should be_empty
  end

  it "records only after the selected kernel is encoded for dispatch" do
    source = File.read(File.expand_path("../src/ml/gguf/qwen_image21_metal.cr", __DIR__))
    start = source.index("private def self.encode_attention(").not_nil!
    finish = source.index("private def self.encode_copy_f32(", start).not_nil!
    encoder = source[start...finish]

    selected = encoder.index("encoder.set_pipeline(pipeline(kernel_name))").not_nil!
    dispatched = encoder.index("encoder.dispatch_threadgroups({groups, 1, 1}, {threads, 1, 1})").not_nil!
    observed = encoder.index("QwenImage21MetalAttentionRouteDiagnostics.record_selected(").not_nil!
    selected.should be < dispatched
    dispatched.should be < observed
    encoder[observed..].should contain("kernel_name, config.head_dim, total_tokens, query_tokens")
    encoder.should contain("threads = tiled || staged ? 128 : head_threads(config.head_dim)")
    encoder.should contain("query_tokens * config.heads")
    encoder.should contain("query_offset: query_offset")
  end
end

describe ML::GGUF::QwenImage21MetalAttentionPolicy do
  it "keeps BF16 stages disabled by default and opts in only for the exact admitted shape" do
    policy = ML::GGUF::QwenImage21MetalAttentionPolicy

    # The focused spec command unsets this env var; this call exercises the
    # production default rather than only an explicit nil override.
    policy.kernel_name(128, 1254, 1254, setting: nil)
      .should eq(ML::GGUF::QwenImage21MetalAttentionPolicy::LEGACY_KERNEL)
    policy.kernel_name(128, 1254, 1254, setting: nil, bf16_stages_setting: nil)
      .should eq(ML::GGUF::QwenImage21MetalAttentionPolicy::LEGACY_KERNEL)
    policy.kernel_name(128, 1254, 1254, setting: nil, bf16_stages_setting: "1")
      .should eq(ML::GGUF::QwenImage21MetalAttentionPolicy::BF16_STAGED_KERNEL)
    policy.kernel_name(128, 1254, 1254, setting: nil, bf16_stages_setting: "true")
      .should eq(ML::GGUF::QwenImage21MetalAttentionPolicy::LEGACY_KERNEL)
    policy.kernel_name(128, 1254, 1024, setting: nil,
      bf16_stages_setting: "1", query_offset: 230)
      .should eq(ML::GGUF::QwenImage21MetalAttentionPolicy::BF16_STAGED_KERNEL)
  end

  it "does not widen BF16 staging to nearby shapes or invalid query ranges" do
    policy = ML::GGUF::QwenImage21MetalAttentionPolicy
    stage = "1"

    policy.kernel_name(127, 1254, 1254, setting: nil, bf16_stages_setting: stage)
      .should eq(ML::GGUF::QwenImage21MetalAttentionPolicy::LEGACY_KERNEL)
    policy.kernel_name(128, 1253, 1253, setting: nil, bf16_stages_setting: stage)
      .should eq(ML::GGUF::QwenImage21MetalAttentionPolicy::LEGACY_KERNEL)
    policy.kernel_name(128, 1254, 0, setting: nil, bf16_stages_setting: stage)
      .should eq(ML::GGUF::QwenImage21MetalAttentionPolicy::LEGACY_KERNEL)
    policy.kernel_name(128, 1254, 1255, setting: nil, bf16_stages_setting: stage)
      .should eq(ML::GGUF::QwenImage21MetalAttentionPolicy::LEGACY_KERNEL)
    policy.kernel_name(128, 1254, 1024, setting: nil,
      bf16_stages_setting: stage, query_offset: -1)
      .should eq(ML::GGUF::QwenImage21MetalAttentionPolicy::LEGACY_KERNEL)
    policy.kernel_name(128, 1254, 1024, setting: nil,
      bf16_stages_setting: stage, query_offset: 231)
      .should eq(ML::GGUF::QwenImage21MetalAttentionPolicy::LEGACY_KERNEL)
  end

  it "preserves the fourth positional tile selector and uses the tile route outside the staged shape" do
    policy = ML::GGUF::QwenImage21MetalAttentionPolicy

    policy.kernel_name(128, 768, 768, "1")
      .should eq(ML::GGUF::QwenImage21MetalAttentionPolicy::TILED_KERNEL)
    policy.kernel_name(128, 1253, 1253, "1", "1")
      .should eq(ML::GGUF::QwenImage21MetalAttentionPolicy::TILED_KERNEL)
    policy.kernel_name(128, 1254, 1254, "1", "1")
      .should eq(ML::GGUF::QwenImage21MetalAttentionPolicy::BF16_STAGED_KERNEL)
  end

  it "rejects all-masked rows before any attention kernel dispatch" do
    policy = ML::GGUF::QwenImage21MetalAttentionPolicy

    policy.valid_rows?([-1, -1, 0, 0], [true, true, true, true]).should be_true
    policy.valid_rows?([-1, 0], [false, true]).should be_false
    policy.valid_rows?([-1, 0], [false, false]).should be_false
    policy.valid_rows?([-1, 0], [true]).should be_false
  end

  it "preserves NaN and infinity in the staged kernel's BF16 conversion guard" do
    source = File.read(File.expand_path("../src/ml/gguf/kernels/qwen_image21.metal", __DIR__))

    source.should contain("if (!isfinite(value)) return value;")
    source.should contain("attention_stages_valid")
  end
end

private def qwen_image21_attention_stage_runtime_output(
  pipeline : ML::Metal::ComputePipeline,
  q : Array(Float32), k : Array(Float32), v : Array(Float32),
  image_ids : Array(Int32), key_valid : Array(UInt8), query_offset : Int32,
) : Array(Float32)
  total_tokens = 1254
  query_tokens = 1
  heads = 1
  head_dim = 128
  buffers = [] of ML::MetalBuffer
  begin
    q_buffer = ML::MetalBuffer.from_array(q)
    buffers << q_buffer
    k_buffer = ML::MetalBuffer.from_array(k)
    buffers << k_buffer
    v_buffer = ML::MetalBuffer.from_array(v)
    buffers << v_buffer
    image_ids_buffer = ML::MetalBuffer.new(image_ids.size.to_i64 * sizeof(Int32))
    buffers << image_ids_buffer
    image_ids_buffer.write_bytes(image_ids.to_unsafe.as(Pointer(UInt8)), image_ids.size * sizeof(Int32))
    key_valid_buffer = ML::MetalBuffer.new(key_valid.size.to_i64)
    buffers << key_valid_buffer
    key_valid_buffer.write_bytes(key_valid.to_unsafe, key_valid.size)
    output_buffer = ML::MetalBuffer.new(query_tokens.to_i64 * heads * head_dim * sizeof(Float32))
    buffers << output_buffer

    command = ML::Metal::CommandBuffer.new
    encoder = ML::Metal::ComputeEncoder.new(command)
    encoder.set_pipeline(pipeline)
    encoder.set_buffer(q_buffer, 0)
    encoder.set_buffer(k_buffer, 1)
    encoder.set_buffer(v_buffer, 2)
    encoder.set_buffer(image_ids_buffer, 3)
    encoder.set_buffer(key_valid_buffer, 4)
    encoder.set_buffer(output_buffer, 5, ML::Metal::BufferAccess::Write)
    encoder.set_value(total_tokens.to_u32, 6)
    encoder.set_value(query_tokens.to_u32, 7)
    encoder.set_value(query_offset.to_u32, 8)
    encoder.set_value(heads.to_u32, 9)
    encoder.set_value(head_dim.to_u32, 10)
    encoder.set_value((1.0_f64 / Math.sqrt(head_dim)).to_f32, 11)
    encoder.dispatch_threadgroups({1, 1, 1}, {128, 1, 1})
    encoder.end_encoding
    command.commit_and_wait
    output_buffer.read(query_tokens * heads * head_dim)
  ensure
    buffers.each(&.release)
  end
end

describe "QwenImage21MetalAttentionBF16StagesRuntime" do
  it "compiles production MSL and smoke-checks one-row finite, non-finite, and empty-mask behavior" do
    pending!("set QWEN_IMAGE21_STAGE_RUNTIME=1 to compile and dispatch the tiny Metal probe") unless ENV["QWEN_IMAGE21_STAGE_RUNTIME"]? == "1"
    pending!("Metal is unavailable") unless ML::Metal::Device.init!

    source = File.read(File.expand_path("../src/ml/gguf/kernels/qwen_image21.metal", __DIR__))
    staged_name = ML::GGUF::QwenImage21MetalAttentionPolicy.kernel_name(
      128, 1254, 1, setting: nil, bf16_stages_setting: "1", query_offset: 230)
    staged_name.should eq(ML::GGUF::QwenImage21MetalAttentionPolicy::BF16_STAGED_KERNEL)
    staged = ML::Metal::ComputePipeline.new(staged_name, source)
    legacy = ML::Metal::ComputePipeline.new(
      ML::GGUF::QwenImage21MetalAttentionPolicy::LEGACY_KERNEL, source,
    )

    total_tokens = 1254
    q = Array(Float32).new(128, 0.0_f32)
    k = Array(Float32).new(total_tokens * 128, 0.0_f32)
    v = Array(Float32).new(total_tokens * 128) do |index|
      (((index * 13) % 47) - 23).to_f32 / 37.0_f32
    end
    image_ids = Array(Int32).new(total_tokens, -1)
    (230...total_tokens).each { |index| image_ids[index] = 0 }
    valid_mask = Array(UInt8).new(total_tokens, 1_u8)
    staged_output = qwen_image21_attention_stage_runtime_output(
      staged, q, k, v, image_ids, valid_mask, 230,
    )
    legacy_output = qwen_image21_attention_stage_runtime_output(
      legacy, q, k, v, image_ids, valid_mask, 230,
    )
    staged_output.all?(&.finite?).should be_true
    legacy_output.all?(&.finite?).should be_true
    staged_output.zip(legacy_output).max_of { |value, reference| (value - reference).abs }
      .should be < 1e-2_f32

    q[0] = Float32::NAN
    nonfinite_output = qwen_image21_attention_stage_runtime_output(
      staged, q, k, v, image_ids, valid_mask, 230,
    )
    nonfinite_output.all?(&.nan?).should be_true

    empty_mask_output = qwen_image21_attention_stage_runtime_output(
      staged, Array(Float32).new(128, 0.0_f32), k, v,
      image_ids, Array(UInt8).new(total_tokens, 0_u8), 230,
    )
    empty_mask_output.all?(&.nan?).should be_true
  end
end

private def qwen_image21_metal_bf16_weight(values : Array(Float32), out_dim : Int32, in_dim : Int32)
  raw = Bytes.new(values.size * 2)
  values.each_with_index do |value, index|
    bits = value.unsafe_as(UInt32) >> 16
    raw[index * 2] = (bits & 0xff).to_u8
    raw[index * 2 + 1] = (bits >> 8).to_u8
  end
  ML::GGUF::QuantWeight.new(raw, ML::GGUF::TensorType::BF16, out_dim, in_dim)
end

private def qwen_image21_metal_normalize_text(
  input : Array(Float32), rows : Int32, dim : Int32,
  weight : Array(Float32), eps : Float32,
) : Array(Float32)
  output = Array(Float32).new(input.size, 0.0_f32)
  rows.times do |row|
    offset = row * dim
    mean_square = 0.0_f64
    dim.times { |column| mean_square += input[offset + column].to_f64 ** 2 }
    inv_rms = 1.0_f64 / Math.sqrt(mean_square / dim + eps)
    dim.times do |column|
      output[offset + column] = (
        input[offset + column] * inv_rms * (weight[column] + 1.0_f32)
      ).to_f32
    end
  end
  output
end

# Isolates the new top-level BF16 route while retaining the already-verified
# mixed-quant Metal projections inside each transformer block.
private class QwenImage21CPUReferenceBF16Backend
  include ML::GGUF::ComputeBackend

  def initialize
    @cpu = ML::GGUF::F32Backend.new
    @metal = ML::GGUF::QwenImage21MetalProjectionBackend.new(strict: true)
  end

  def matmul(x : Array(Float32), rows : Int32, qw : ML::GGUF::QuantWeight,
             bias : Array(Float32)) : Array(Float32)
    if qw.type.bf16?
      @cpu.matmul(x, rows, qw, bias)
    else
      @metal.matmul(x, rows, qw, bias)
    end
  end

  def layer_norm!(x : Array(Float32), n_pos : Int32, dim : Int32,
                  w : Array(Float32), b : Array(Float32)) : Nil
    @cpu.layer_norm!(x, n_pos, dim, w, b)
  end

  def softmax_row!(scores : Array(Float32), offset : Int32, len : Int32) : Nil
    @cpu.softmax_row!(scores, offset, len)
  end

  def gelu(x : Float32) : Float32
    @cpu.gelu(x)
  end

  def dot(a : Array(Float32), a_off : Int32, b : Array(Float32),
          b_off : Int32, len : Int32) : Float32
    @cpu.dot(a, a_off, b, b_off, len)
  end
end

describe ML::GGUF::QwenImage21MetalProjectionBackend do
  it "matches the CPU reference for a BF16 batch projection" do
    pending!("Metal is unavailable") unless ML::GGUF::QwenImage21MetalProjectionBackend.available?
    weight = qwen_image21_metal_bf16_weight(
      Array(Float32).new(20) { |index| (((index * 7) % 13) - 6).to_f32 / 5.0_f32 },
      4,
      5,
    )
    input = Array(Float32).new(15) do |index|
      (((index * 11) % 17) - 8).to_f32 / 7.0_f32
    end
    bias = [0.25_f32, -0.5_f32, 0.75_f32, -1.0_f32]
    expected = ML::GGUF::F32Backend.new.matmul(input, 3, weight, bias)
    backend = ML::GGUF::QwenImage21MetalProjectionBackend.new(strict: true)

    actual = backend.matmul(input, 3, weight, bias)

    actual.zip(expected).each do |value, reference|
      value.should be_close(reference, 1e-5_f32)
    end
    backend.metal_projection_count.should eq(1)
    backend.bf16_projection_count.should eq(1)
  end

  it "keeps chained BF16 text and timestep projections in one command each" do
    pending!("Metal is unavailable") unless ML::GGUF::QwenImage21MetalProjectionBackend.available?
    cpu = ML::GGUF::F32Backend.new
    backend = ML::GGUF::QwenImage21MetalProjectionBackend.new(strict: true)
    weight = ->(out_dim : Int32, in_dim : Int32, phase : Int32) do
      qwen_image21_metal_bf16_weight(
        Array(Float32).new(out_dim * in_dim) do |index|
          (((index * 7 + phase * 5) % 23) - 11).to_f32 / 13.0_f32
        end,
        out_dim,
        in_dim,
      )
    end
    rows = 2
    input = Array(Float32).new(rows * 3) do |index|
      (((index * 11) % 17) - 8).to_f32 / 9.0_f32
    end
    first = weight.call(4, 3, 1)
    second = weight.call(4, 4, 2)
    modulation = weight.call(8, 4, 3)
    scale = weight.call(4, 4, 4)

    text_hidden = cpu.matmul(input, rows, first, Array(Float32).new(4, 0.0_f32))
    text_hidden.map! { |value| cpu.gelu(value) }
    expected_text = cpu.matmul(text_hidden, rows, second, Array(Float32).new(4, 0.0_f32))
    actual_text = backend.project_text_layers(input, rows, first, second).not_nil!

    time_hidden = cpu.matmul(input, rows, first, Array(Float32).new(4, 0.0_f32))
    time_hidden.map! { |value| value / (1.0_f32 + Math.exp(-value)) }
    time_hidden = cpu.matmul(time_hidden, rows, second, Array(Float32).new(4, 0.0_f32))
    time_hidden.map! { |value| value / (1.0_f32 + Math.exp(-value)) }
    expected_modulation = cpu.matmul(
      time_hidden, rows, modulation, Array(Float32).new(8, 0.0_f32)
    )
    expected_scale = cpu.matmul(time_hidden, rows, scale, Array(Float32).new(4, 0.0_f32))
    actual_time = backend.project_timestep_layers(
      input, rows, first, second, modulation, scale
    ).not_nil!

    actual_text.zip(expected_text).each do |value, reference|
      value.should be_close(reference, 1e-4_f32)
    end
    actual_time[0].zip(expected_modulation).each do |value, reference|
      value.should be_close(reference, 1e-4_f32)
    end
    actual_time[1].zip(expected_scale).each do |value, reference|
      value.should be_close(reference, 1e-4_f32)
    end
    backend.metal_projection_count.should eq(6)
    backend.bf16_projection_count.should eq(6)
    backend.fused_outer_command_count.should eq(2)
  end

  it "keeps finite saturated GELU activations finite in the fused BF16 text chain" do
    pending!("Metal is unavailable") unless ML::GGUF::QwenImage21MetalProjectionBackend.available?
    first = qwen_image21_metal_bf16_weight([12.0_f32, -12.0_f32, 20.0_f32, -20.0_f32], 4, 1)
    identity = qwen_image21_metal_bf16_weight(
      [1.0_f32, 0.0_f32, 0.0_f32, 0.0_f32,
       0.0_f32, 1.0_f32, 0.0_f32, 0.0_f32,
       0.0_f32, 0.0_f32, 1.0_f32, 0.0_f32,
       0.0_f32, 0.0_f32, 0.0_f32, 1.0_f32],
      4,
      4,
    )
    expected = [12.0_f32, 0.0_f32, 20.0_f32, 0.0_f32]

    actual = ML::GGUF::QwenImage21MetalBF16.project_text_layers(
      [1.0_f32], 1, first, identity
    ).not_nil!

    actual.count(&.finite?).should eq(actual.size)
    actual.zip(expected).each do |value, reference|
      value.should be_close(reference, 1e-4_f32)
    end
  end

  it "keeps fused real Qwen3-VL text projection finite against the separated Metal route" do
    path = ENV["QWEN_IMAGE21_GGUF"]?
    bundle_path = ENV["QWEN_IMAGE21_CONDITIONING"]?
    pending!("set QWEN_IMAGE21_GGUF and QWEN_IMAGE21_CONDITIONING for real text projection") unless path && File.file?(path) && bundle_path && File.file?(bundle_path)
    pending!("Metal is unavailable") unless ML::GGUF::QwenImage21MetalProjectionBackend.available?

    conditioning = ML::GGUF::QwenImage21ConditioningBundle.load(bundle_path.not_nil!)
    weights = ML::GGUF::QwenImage21Weights.from_gguf(path.not_nil!)
    begin
      config = weights.transformer_config
      rows = conditioning.encoder_hidden_states.size // config.context_dim
      normalized = qwen_image21_metal_normalize_text(
        conditioning.encoder_hidden_states, rows, config.context_dim,
        weights.text_norm, config.block.eps,
      )
      backend = ML::GGUF::QwenImage21MetalProjectionBackend.new(strict: true)
      first = backend.matmul(
        normalized, rows, weights.text_in_layer,
        Array(Float32).new(config.hidden_dim, 0.0_f32),
      )
      cpu = ML::GGUF::F32Backend.new
      first.map! { |value| cpu.gelu(value) }
      separated = backend.matmul(
        first, rows, weights.text_out_layer,
        Array(Float32).new(config.hidden_dim, 0.0_f32),
      )
      fused = ML::GGUF::QwenImage21MetalBF16.project_text_layers(
        normalized, rows, weights.text_in_layer, weights.text_out_layer,
      ).not_nil!

      fused_finite = fused.count(&.finite?)
      separated_finite = separated.count(&.finite?)
      max_abs = 0.0_f64
      if fused_finite == fused.size && separated_finite == separated.size
        fused.zip(separated).each do |actual, expected|
          max_abs = Math.max(max_abs, (actual - expected).abs)
        end
      end
      fused.size.should eq(rows * config.hidden_dim)
      separated.size.should eq(fused.size)
      separated_finite.should eq(separated.size)
      fused_finite.should eq(fused.size)
      max_abs.should be < 1.0e-2
    ensure
      weights.close
    end
  end

  it "matches the CPU reference for one real mixed-quant transformer block" do
    path = ENV["QWEN_IMAGE21_GGUF"]?
    pending!("set QWEN_IMAGE21_GGUF to run the model-backed parity check") unless path && File.file?(path)
    pending!("Metal is unavailable") unless ML::GGUF::QwenImage21MetalProjectionBackend.available?

    weights = ML::GGUF::QwenImage21Weights.from_gguf(path.not_nil!)
    begin
      config = weights.block_config
      token_count = 2
      hidden = Array(Float32).new(token_count * config.hidden_dim) do |index|
        (((index * 17 + 11) % 257) - 128).to_f32 / 193.0_f32
      end
      modulation = Array(Float32).new(token_count * 4 * config.hidden_dim) do |index|
        (((index * 13 + 7) % 101) - 50).to_f32 / 401.0_f32
      end
      positions = [StaticArray[0, 0, 0], StaticArray[1, 1, 1]]
      image_ids = [-1, 0]

      cpu = ML::GGUF::QwenImage21BlockCPU.forward(
        hidden, token_count, modulation, positions, image_ids,
        weights.layers[0], config,
      )
      backend = ML::GGUF::QwenImage21MetalProjectionBackend.new(strict: true)
      metal = ML::GGUF::QwenImage21BlockCPU.forward(
        hidden, token_count, modulation, positions, image_ids,
        weights.layers[0], config,
        backend: backend,
      )

      max_abs = 0.0_f64
      dot = 0.0_f64
      cpu_norm = 0.0_f64
      metal_norm = 0.0_f64
      cpu.each_with_index do |expected, index|
        actual = metal[index]
        max_abs = Math.max(max_abs, (expected - actual).abs)
        dot += expected.to_f64 * actual
        cpu_norm += expected.to_f64 ** 2
        metal_norm += actual.to_f64 ** 2
      end
      cosine = dot / Math.sqrt(cpu_norm * metal_norm)
      STDERR.puts "qwen_image21_block_parity max_abs=#{max_abs} cosine=#{cosine}"

      backend.metal_projection_count.should eq(6)
      metal.all?(&.finite?).should be_true
      max_abs.should be < 1.0e-2
      cosine.should be > 0.999999
    ensure
      weights.close
    end
  end

  it "executes the complete 32-block outer transformer with a real text prefix" do
    path = ENV["QWEN_IMAGE21_GGUF"]?
    pending!("set QWEN_IMAGE21_GGUF to run the model-backed outer-forward check") unless path && File.file?(path)
    pending!("Metal is unavailable") unless ML::GGUF::QwenImage21MetalProjectionBackend.available?

    weights = ML::GGUF::QwenImage21Weights.from_gguf(path.not_nil!)
    begin
      config = weights.transformer_config
      hidden = Array(Float32).new(4 * config.input_dim) do |index|
        (((index * 23 + 5) % 97) - 48).to_f32 / 127.0_f32
      end
      encoder_hidden = Array(Float32).new(config.context_dim) do |index|
        (((index * 31 + 9) % 103) - 51).to_f32 / 137.0_f32
      end
      expected = ML::GGUF::QwenImage21TransformerCPU.forward(
        hidden,
        encoder_hidden,
        0.5_f32,
        [StaticArray[1, 2, 2]],
        [false, true],
        weights.transformer_weights,
        config,
        backend: QwenImage21CPUReferenceBF16Backend.new,
      )
      backend = ML::GGUF::QwenImage21MetalProjectionBackend.new(strict: false)
      started = Time.instant
      result = ML::GGUF::QwenImage21TransformerCPU.forward(
        hidden,
        encoder_hidden,
        0.5_f32,
        [StaticArray[1, 2, 2]],
        [false, true],
        weights.transformer_weights,
        config,
        backend: backend,
      )
      elapsed = Time.instant - started
      STDERR.puts "qwen_image21_outer_forward seconds=#{elapsed.total_seconds} metal_projections=#{backend.metal_projection_count}"

      result.output.size.should eq(5 * config.output_dim)
      result.output.all?(&.finite?).should be_true
      result.layout.target_token_mask.should eq([false, true, true, true, true])
      max_abs = 0.0_f64
      dot = 0.0_f64
      expected_norm = 0.0_f64
      result_norm = 0.0_f64
      result.output.zip(expected.output).each do |value, reference|
        max_abs = Math.max(max_abs, (value - reference).abs)
        dot += value.to_f64 * reference
        expected_norm += reference.to_f64 ** 2
        result_norm += value.to_f64 ** 2
      end
      cosine = dot / Math.sqrt(expected_norm * result_norm)
      STDERR.puts "qwen_image21_outer_bf16_parity max_abs=#{max_abs} cosine=#{cosine}"
      max_abs.should be < 1e-4
      cosine.should be > 0.999999
      backend.metal_projection_count.should eq(32 * 6 + 8)
      backend.bf16_projection_count.should eq(8)
      backend.fused_outer_command_count.should eq(1)
    ensure
      weights.close
    end
  end

  it "runs two configured FlowMatch steps through all 32 real blocks" do
    path = ENV["QWEN_IMAGE21_GGUF"]?
    pending!("set QWEN_IMAGE21_GGUF to run the model-backed denoising check") unless path && File.file?(path)
    pending!("Metal is unavailable") unless ML::GGUF::QwenImage21MetalProjectionBackend.available?

    weights = ML::GGUF::QwenImage21Weights.from_gguf(path.not_nil!)
    begin
      config = weights.transformer_config
      initial = Array(Float32).new(4 * config.input_dim) do |index|
        (((index * 29 + 7) % 101) - 50).to_f32 / 131.0_f32
      end
      backend = ML::GGUF::QwenImage21MetalProjectionBackend.new(strict: false)
      started = Time.instant
      result = ML::GGUF::QwenImage21LatentDenoiser.run(
        initial,
        [] of Float32,
        [] of Float32,
        [StaticArray[1, 2, 2]],
        [] of Bool,
        weights.transformer_weights,
        config,
        num_inference_steps: 2,
        backend: backend,
      )
      elapsed = Time.instant - started
      STDERR.puts "qwen_image21_two_step_denoise seconds=#{elapsed.total_seconds} metal_projections=#{backend.metal_projection_count}"

      result.transformer_evaluations.should eq(2)
      result.latents.size.should eq(initial.size)
      result.latents.all?(&.finite?).should be_true
      result.latents.should_not eq(initial)
      backend.metal_projection_count.should eq(2 * (32 * 6 + 6))
      backend.bf16_projection_count.should eq(12)
      backend.fused_outer_command_count.should eq(2)
    ensure
      weights.close
    end
  end

  it "keeps real Qwen3-VL text-to-image denoising finite on the default Metal route" do
    path = ENV["QWEN_IMAGE21_GGUF"]?
    bundle_path = ENV["QWEN_IMAGE21_CONDITIONING"]?
    pending!("set QWEN_IMAGE21_GGUF and QWEN_IMAGE21_CONDITIONING for real prompt integration") unless path && File.file?(path) && bundle_path && File.file?(bundle_path)
    pending!("the unsafe fused text route was explicitly enabled") if ENV["QWEN_IMAGE21_FUSED_TEXT"]? == "1"
    pending!("Metal is unavailable") unless ML::GGUF::QwenImage21MetalProjectionBackend.available?

    conditioning = ML::GGUF::QwenImage21ConditioningBundle.load(bundle_path.not_nil!)
    weights = ML::GGUF::QwenImage21Weights.from_gguf(path.not_nil!)
    stack = ML::GGUF::QwenImage21MetalLayerStackBackend.new
    begin
      result = ML::GGUF::QwenImage21LatentDenoiser.run(
        conditioning.initial_target_latents,
        [] of Float32,
        conditioning.encoder_hidden_states,
        conditioning.img_shapes,
        conditioning.encoder_img_mask,
        weights.transformer_weights,
        weights.transformer_config,
        num_inference_steps: 2,
        encoder_hidden_states_mask: conditioning.encoder_hidden_states_mask,
        backend: ML::GGUF::QwenImage21MetalProjectionBackend.new(strict: true),
        layer_stack_backend: stack,
      )
      result.transformer_evaluations.should eq(2)
      result.latents.size.should eq(conditioning.initial_target_latents.size)
      result.latents.all?(&.finite?).should be_true
      result.latents.should_not eq(conditioning.initial_target_latents)
    ensure
      stack.close
      weights.close
    end
  end
end
