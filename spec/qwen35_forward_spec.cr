require "./spec_helper"
require "../src/ml/gguf/qwen35_cpu"
require "../src/ml/gguf/qwen35_weights"

QWEN_9B_FWD     = "#{ENV["HOME"]}/.cache/lm-studio/models/lmstudio-community/Qwen3.5-9B-GGUF/Qwen3.5-9B-Q4_K_M.gguf"
QWEN_08B_FWD    = "#{ENV["HOME"]}/.cache/lm-studio/models/lmstudio-community/Qwen3.5-0.8B-GGUF/Qwen3.5-0.8B-Q8_0.gguf"
QWEN_38_27B_FWD = "#{ENV["HOME"]}/.cache/lm-studio/models/lmstudio-community/Qwen3.8-27B-GGUF/Qwen3.8-27B-Q4_K_M.gguf"

private def with_qwen38_adaptive_allowed_fallback_env(layer_index : Int32, &)
  keys = [
    "QWEN35_ADAPTIVE_RESIDENT_KV_LAYER",
    "QWEN35_ADAPTIVE_RESIDENT_KV_TIER",
    "QWEN35_ADAPTIVE_RESIDENT_KV_MAP",
    "QWEN35_PREFILL_APPEND_CMD_OFF",
    "QWEN35_PREFILL_RESIDENT_BOUNDARY_OFF",
    "QWEN35_PREFILL_FUSE_FULL_REC_OFF",
    "QWEN35_FULL_PREFILL_CHUNK_OFF",
    "QWEN35_PREFILL_REC_RUN_OFF",
    "QWEN35_PREFILL_CHUNK_OFF",
    "QWEN35_HEAD_TOP1_FUSED",
  ]
  old = keys.to_h { |key| {key, ENV[key]?} }
  keys.each { |key| ENV.delete(key) }
  ENV["QWEN35_ADAPTIVE_RESIDENT_KV_LAYER"] = layer_index.to_s
  ENV["QWEN35_ADAPTIVE_RESIDENT_KV_TIER"] = "bf16"
  ENV["QWEN35_HEAD_TOP1_FUSED"] = "0"
  yield
ensure
  old.try &.each do |key, value|
    if value
      ENV[key] = value
    else
      ENV.delete(key)
    end
  end
end

private def with_qwen35_head_top1_fused(&)
  old = ENV["QWEN35_HEAD_TOP1_FUSED"]?
  ENV["QWEN35_HEAD_TOP1_FUSED"] = "1"
  yield
ensure
  if old
    ENV["QWEN35_HEAD_TOP1_FUSED"] = old
  else
    ENV.delete("QWEN35_HEAD_TOP1_FUSED")
  end
end

describe ML::GGUF::Qwen35Metal, "route policies" do
  it "reports optional GPU command timing separately from host wait timing" do
    profile = ML::GGUF::Qwen35Metal::Profile
    old_gpu_timing = ENV["QWEN35_METAL_GPU_TIMING"]?
    profile.reset
    profile.enable!
    begin
      ENV.delete("QWEN35_METAL_GPU_TIMING")
      profile.bump_gpu_command("disabled", 1.0_f64)
      profile.report_io.should_not contain("GPU command buffers")

      ENV["QWEN35_METAL_GPU_TIMING"] = "1"
      profile.bump_gpu_command("decode.layers.0-1.rr", 0.001_f64)
      profile.bump_gpu_command("decode.layers.0-1.rr", 0.002_f64)
      profile.bump_gpu_command("decode.layers.2-3.rf", 0.004_f64)

      report = profile.report_io
      report.should contain("GPU command buffers")
      report.should contain("decode.layers.0-1.rr")
      report.should contain("2 calls")
      report.should contain("3.00 ms")
      report.should contain("decode.layers.2-3.rf")
      report.should contain("4.00 ms")
    ensure
      if old_gpu_timing
        ENV["QWEN35_METAL_GPU_TIMING"] = old_gpu_timing
      else
        ENV.delete("QWEN35_METAL_GPU_TIMING")
      end
      profile.disable!
      profile.reset
    end
  end

  it "bounds automatic B64 tail fusion padding on M2 Max" do
    metal = ML::GGUF::Qwen35Metal

    metal.q4_h16_b64_tail_policy?(96, "Apple M2 Max", nil).should be_false
    metal.q4_h16_b64_tail_policy?(103, "Apple M2 Max", nil).should be_false
    metal.q4_h16_b64_tail_policy?(114, "Apple M2 Max", nil).should be_true
    metal.q4_h16_b64_tail_policy?(129, "Apple M2 Max", nil).should be_false
    metal.q4_h16_b64_tail_policy?(171, "Apple M2 Max", nil).should be_true
    metal.q4_h16_b64_tail_policy?(193, "Apple M2 Max", nil).should be_false
    metal.q4_h16_b64_tail_policy?(205, "Apple M2 Max", nil).should be_false
    metal.q4_h16_b64_tail_policy?(228, "Apple M2 Max", nil).should be_true
    metal.q4_h16_b64_tail_policy?(257, "Apple M2 Max", nil).should be_false
    metal.q4_h16_b64_tail_policy?(285, "Apple M2 Max", nil).should be_true
    metal.q4_h16_b64_tail_policy?(321, "Apple M2 Max", nil).should be_false
    metal.q4_h16_b64_tail_policy?(342, "Apple M2 Max", nil).should be_true
    metal.q4_h16_b64_tail_policy?(360, "Apple M2 Max", nil).should be_true

    metal.q4_h16_b64_tail_policy?(360, "Apple M3 Max", nil).should be_false
    metal.q4_h16_b64_tail_policy?(96, "Apple M3 Max", "96").should be_true
    metal.q4_h16_b64_tail_policy?(360, "Apple M2 Max", "0").should be_false
    metal.q4_h16_b64_tail_policy?(0, "Apple M2 Max", nil).should be_false
    metal.q4_h16_b64_tail_policy?(-1, "Apple M2 Max", nil).should be_false
    metal.q4_h16_b64_tail_policy?(Int32::MAX, "Apple M2 Max", nil).should be_true

    expect_raises(ArgumentError) { metal.q4_h16_b64_tail_policy?(360, "Apple M2 Max", "") }
    expect_raises(ArgumentError) { metal.q4_h16_b64_tail_policy?(360, "Apple M2 Max", "abc") }
    expect_raises(ArgumentError) { metal.q4_h16_b64_tail_policy?(360, "Apple M2 Max", "999999999999999999999999") }
  end
end

describe ML::GGUF::Qwen35CPU, "full decoder forward" do
  pending!("9B model not present") unless File.exists?(QWEN_9B_FWD)

  it "chooses prefill chunk defaults from physical memory size" do
    gib = ML::GGUF::Qwen35CPU::GIB
    ML::GGUF::Qwen35CPU.prefill_chunk_size_for_memory(nil).should eq(4096)
    ML::GGUF::Qwen35CPU.prefill_chunk_size_for_memory(16_u64 * gib).should eq(2048)
    ML::GGUF::Qwen35CPU.prefill_chunk_size_for_memory(24_u64 * gib).should eq(4096)
    ML::GGUF::Qwen35CPU.prefill_chunk_size_for_memory(48_u64 * gib).should eq(8192)
  end

  it "bounds shared prefill command groups by the prompt-row budget" do
    ML::GGUF::Qwen35CPU.prefill_append_group_limit(512, nil).should eq(0)
    ML::GGUF::Qwen35CPU.prefill_append_group_limit(1024, nil).should eq(2)
    ML::GGUF::Qwen35CPU.prefill_append_group_limit(2048, nil).should eq(1)
    ML::GGUF::Qwen35CPU.prefill_append_group_limit(4096, nil).should eq(1)
    ML::GGUF::Qwen35CPU.prefill_append_group_limit(8192, nil).should eq(1)
    ML::GGUF::Qwen35CPU.prefill_append_group_limit(8192, "3").should eq(3)
    ML::GGUF::Qwen35CPU.prefill_append_group_limit(8192, "0").should eq(0)
    expect_raises(ArgumentError, /non-negative integer/) do
      ML::GGUF::Qwen35CPU.prefill_append_group_limit(8192, "invalid")
    end
  end

  it "parses the prefill command cooldown fail-closed" do
    ML::GGUF::Qwen35CPU.prefill_append_cooldown_ms(nil).should eq(50)
    ML::GGUF::Qwen35CPU.prefill_append_cooldown_ms("50").should eq(50)
    ML::GGUF::Qwen35CPU.prefill_append_cooldown_ms("0").should eq(0)
    expect_raises(ArgumentError, /non-negative integer/) do
      ML::GGUF::Qwen35CPU.prefill_append_cooldown_ms("invalid")
    end
    expect_raises(ArgumentError, /non-negative integer/) do
      ML::GGUF::Qwen35CPU.prefill_append_cooldown_ms("-1")
    end
  end

  it "keeps the compositor cooldown across long-prefill chunk boundaries" do
    ML::GGUF::Qwen35CPU.prefill_chunk_boundary_cooldown_ms(1024, true, true, nil, nil).should eq(50)
    ML::GGUF::Qwen35CPU.prefill_chunk_boundary_cooldown_ms(1024, true, true, "1", "100").should eq(100)
    ML::GGUF::Qwen35CPU.prefill_chunk_boundary_cooldown_ms(1024, false, true, nil, nil).should eq(0)
    ML::GGUF::Qwen35CPU.prefill_chunk_boundary_cooldown_ms(1024, true, false, nil, nil).should eq(0)
    ML::GGUF::Qwen35CPU.prefill_chunk_boundary_cooldown_ms(512, true, true, nil, nil).should eq(0)
    ML::GGUF::Qwen35CPU.prefill_chunk_boundary_cooldown_ms(1024, true, true, "0", nil).should eq(0)
  end

  it "admits bounded CogniGraph prefill enqueue for exact and adaptive prefill" do
    qwen = ML::GGUF::Qwen35CPU
    qwen.prefill_graph_max_inflight(false, false, false, nil).should eq(0)
    qwen.prefill_graph_max_inflight(false, false, false, "0").should eq(0)
    qwen.prefill_graph_max_inflight(false, false, false, "1").should eq(1)
    qwen.prefill_graph_max_inflight(false, false, false, "2").should eq(2)

    expect_raises(ArgumentError, /between 0 and 2/) do
      qwen.prefill_graph_max_inflight(false, false, false, "3")
    end
    qwen.prefill_graph_max_inflight(true, false, false, "1").should eq(1)
    qwen.prefill_graph_max_inflight(true, false, false, "2").should eq(2)
    expect_raises(ArgumentError, /checkpoint/) do
      qwen.prefill_graph_max_inflight(false, true, false, "1")
    end
    expect_raises(ArgumentError, /profiling/) do
      qwen.prefill_graph_max_inflight(false, false, true, "1")
    end
  end

  it "appends the resident top-1 head only at the final adaptive layer" do
    qwen = ML::GGUF::Qwen35CPU
    qwen.prefill_resident_top1_append_layer?(0, 16).should be_false
    qwen.prefill_resident_top1_append_layer?(14, 16).should be_false
    qwen.prefill_resident_top1_append_layer?(15, 16).should be_true
    qwen.prefill_resident_top1_append_layer?(0, 1).should be_true
    qwen.prefill_resident_top1_append_layer?(0, 0).should be_false
  end

  it "gives concurrent CogniGraph leases disjoint scratch arenas" do
    pending!("Metal not available") unless ML::GGUF::Qwen35Metal.available?

    first = ML::GGUF::Qwen35Metal::Scratch::Arena.new("cognigraph_spec:0")
    second = ML::GGUF::Qwen35Metal::Scratch::Arena.new("cognigraph_spec:1")
    first_buf = first.with { ML::GGUF::Qwen35Metal::Scratch.get(:cognigraph_spec, 64_i64) }
    second_buf = second.with { ML::GGUF::Qwen35Metal::Scratch.get(:cognigraph_spec, 64_i64) }

    first_buf.handle.should_not eq(second_buf.handle)
    first.buffers.should eq([first_buf])
    second.buffers.should eq([second_buf])
    first.release
    second.release
    first.released?.should be_true
    second.released?.should be_true
    first_buf.valid?.should be_false
    second_buf.valid?.should be_false
  end

  it "caps automatic resident prefill row tiles while preserving explicit overrides" do
    default_size = ML::GGUF::Qwen35CPU.default_prefill_chunk_size
    ML::GGUF::Qwen35CPU.prefill_chunk_size(false, nil).should eq(default_size)
    ML::GGUF::Qwen35CPU.prefill_chunk_size(true, nil).should eq(Math.min(default_size, 2048))
    ML::GGUF::Qwen35CPU.prefill_chunk_size(true, "4096").should eq(4096)
    expect_raises(ArgumentError, /positive integer/) do
      ML::GGUF::Qwen35CPU.prefill_chunk_size(true, "invalid")
    end
  end

  it "produces finite logits at pos=0 for token 0" do
    w = ML::GGUF::Qwen35Weights.from_gguf(QWEN_9B_FWD)
    hp = w.hparams

    state = ML::GGUF::Qwen35CPU::State.new(hp, max_seq: 32)

    t0 = Time.instant
    logits = ML::GGUF::Qwen35CPU.forward(w, 0, 0, state)
    dt = Time.instant - t0
    puts "  [qwen35_forward] first-token latency: #{dt.total_milliseconds.round(1)} ms"

    logits.size.should eq(w.output.out_dim) # vocab_size (≈248k)
    logits.all? { |v| v.finite? }.should be_true

    # Logits should have some spread (not all identical)
    maxv = logits.max
    minv = logits.min
    (maxv - minv).should be > 1.0_f32

    top = logits.index(maxv).not_nil!
    top.should eq(198)
    maxv.should be_close(11.423705_f32, 1e-4_f32)
    puts "  [qwen35_forward] top token id=#{top}, logit=#{maxv}"
  end

  it "produces different logits for different token inputs" do
    w = ML::GGUF::Qwen35Weights.from_gguf(QWEN_9B_FWD)
    hp = w.hparams

    state_a = ML::GGUF::Qwen35CPU::State.new(hp, max_seq: 32)
    state_b = ML::GGUF::Qwen35CPU::State.new(hp, max_seq: 32)

    logits_a = ML::GGUF::Qwen35CPU.forward(w, 100, 0, state_a)
    logits_b = ML::GGUF::Qwen35CPU.forward(w, 5000, 0, state_b)

    # Top-1 should almost certainly differ between input 100 and input 5000
    top_a = logits_a.index(logits_a.max).not_nil!
    top_b = logits_b.index(logits_b.max).not_nil!
    top_a.should_not eq(top_b)
  end

  it "matches full logits top-1 on the fused greedy head route" do
    w = ML::GGUF::Qwen35Weights.from_gguf(QWEN_9B_FWD)
    hp = w.hparams

    state_full = ML::GGUF::Qwen35CPU::State.new(hp, max_seq: 32)
    logits = ML::GGUF::Qwen35CPU.forward(w, 0, 0, state_full)
    full_max = logits.max
    full_top = logits.index(full_max).not_nil!.to_i32

    old = ENV["QWEN35_HEAD_TOP1_FUSED"]?
    ENV["QWEN35_HEAD_TOP1_FUSED"] = "1"
    begin
      state_top1 = ML::GGUF::Qwen35CPU::State.new(hp, max_seq: 32)
      top_id, top_logit = ML::GGUF::Qwen35CPU.forward_top1(w, 0, 0, state_top1)
      top_id.should eq(full_top)
      top_logit.should be_close(full_max, 1e-4_f32)
    ensure
      if old
        ENV["QWEN35_HEAD_TOP1_FUSED"] = old
      else
        ENV.delete("QWEN35_HEAD_TOP1_FUSED")
      end
    end
  end

  it "falls back to exact allowed logits for adaptive decode when the fused head is unavailable" do
    pending!("Qwen3.8 27B model not present") unless File.exists?(QWEN_38_27B_FWD)
    pending!("Metal not available") unless ML::GGUF::Qwen35Metal.available?

    weights = ML::GGUF::Qwen35Weights.from_gguf(QWEN_38_27B_FWD)
    hp = weights.hparams
    expected_state = nil.as(ML::GGUF::Qwen35CPU::State?)
    actual_state = nil.as(ML::GGUF::Qwen35CPU::State?)

    with_qwen38_adaptive_allowed_fallback_env(hp.full_attention_layers.first) do
      expected_state = ML::GGUF::Qwen35CPU::State.new(hp, max_seq: 2)
      ML::GGUF::Qwen35CPU.prepare_state_metal!(expected_state.not_nil!, hp, clear: true)
      expected_state.not_nil!.adaptive_kv?.should be_true
      expected_state.not_nil!.adaptive_kv_layer_indices.should_not be_empty
      logits = ML::GGUF::Qwen35CPU.forward(weights, 0, 0, expected_state.not_nil!)
      full_top = logits.index(logits.max).not_nil!.to_i32
      allowed_ids = Array(Int32).new(13) do |offset|
        (full_top + offset + 1) % weights.output.out_dim
      end
      expected_id = allowed_ids.max_by { |id| logits[id] }
      expected_logit = logits[expected_id]
      allowed_ids.includes?(full_top).should be_false
      expected_state.not_nil!.adaptive_kv_layer_indices.each do |layer_index|
        expected_state.not_nil!.layers[layer_index].adaptive_kv.not_nil!.cache_len.should eq(1)
      end
      ML::GGUF::Qwen35CPU.release_state_metal!(expected_state.not_nil!)
      expected_state = nil

      actual_state = ML::GGUF::Qwen35CPU::State.new(hp, max_seq: 2)
      ML::GGUF::Qwen35CPU.prepare_state_metal!(actual_state.not_nil!, hp, clear: true)
      actual_state.not_nil!.adaptive_kv?.should be_true
      actual_state.not_nil!.adaptive_kv_layer_indices.should_not be_empty
      actual_id, actual_logit = ML::GGUF::Qwen35CPU.forward_top1_allowed(
        weights, 0, 0, actual_state.not_nil!, allowed_ids,
      )
      actual_id.should eq(expected_id)
      actual_logit.should be_close(expected_logit, 1.0e-4_f32)
      actual_state.not_nil!.adaptive_kv_layer_indices.each do |layer_index|
        actual_state.not_nil!.layers[layer_index].adaptive_kv.not_nil!.cache_len.should eq(1)
      end
    end
  ensure
    ML::GGUF::Qwen35CPU.release_state_metal!(expected_state) if expected_state
    ML::GGUF::Qwen35CPU.release_state_metal!(actual_state) if actual_state
    weights.try(&.close)
  end

  it "projects top-1 directly from a selected resident hidden row" do
    pending!("Metal not available") unless ML::GGUF::Qwen35Metal.available?

    w = ML::GGUF::Qwen35Weights.from_gguf(QWEN_9B_FWD)
    hidden_dim = w.hparams.n_embd
    first = Array(Float32).new(hidden_dim) { |i| ((i % 29) - 14).to_f32 / 29.0_f32 }
    second = Array(Float32).new(hidden_dim) { |i| ((i % 31) - 15).to_f32 / 31.0_f32 }
    resident = ML::MetalBuffer.new((2 * hidden_dim).to_i64 * sizeof(Float32))
    resident.write(first + second)

    normalized = ML::GGUF::Qwen35CPU.rms_norm(
      second, w.output_norm, w.hparams.rms_eps,
    )
    cpu_logits = ML::GGUF::QuantMatmul.matmul_add(
      normalized, 1, w.output.in_dim, w.output.raw, w.output.type,
      w.output.out_dim, Array(Float32).new(w.output.out_dim, 0.0_f32),
    )
    expected_id = cpu_logits.index(cpu_logits.max).not_nil!.to_i32
    actual = ML::GGUF::Qwen35Metal.rmsnorm_project_top1_buffer(
      resident, hidden_dim.to_i64, w.output_norm, w.output, w.hparams.rms_eps,
    ).not_nil!

    actual[0].to_i32.should eq(expected_id)
    actual[1].should be_close(cpu_logits[expected_id], 1.0e-4_f32)
    ML::GGUF::Qwen35Metal.rmsnorm_project_top1_buffer(
      resident, -1_i64, w.output_norm, w.output, w.hparams.rms_eps,
    ).should be_nil
    ML::GGUF::Qwen35Metal.rmsnorm_project_top1_buffer(
      resident, hidden_dim.to_i64 + 1_i64, w.output_norm, w.output, w.hparams.rms_eps,
    ).should be_nil
    ML::GGUF::Qwen35Metal.rmsnorm_project_top1_buffer(
      resident, Int64::MAX, w.output_norm, w.output, w.hparams.rms_eps,
    ).should be_nil
  end

  it "appends a selected resident top-1 head without committing its caller command" do
    pending!("Metal not available") unless ML::GGUF::Qwen35Metal.available?

    w = ML::GGUF::Qwen35Weights.from_gguf(QWEN_9B_FWD)
    hidden_dim = w.hparams.n_embd
    first = Array(Float32).new(hidden_dim) { |i| ((i % 17) - 8).to_f32 / 17.0_f32 }
    second = Array(Float32).new(hidden_dim) { |i| ((i % 23) - 11).to_f32 / 23.0_f32 }
    resident = ML::MetalBuffer.new((2 * hidden_dim).to_i64 * sizeof(Float32))
    resident.write(first + second)
    top1_id = ML::MetalBuffer.new(sizeof(UInt32).to_i64)
    top1_value = ML::MetalBuffer.new(sizeof(Float32).to_i64)

    normalized = ML::GGUF::Qwen35CPU.rms_norm(
      second, w.output_norm, w.hparams.rms_eps,
    )
    cpu_logits = ML::GGUF::QuantMatmul.matmul_add(
      normalized, 1, w.output.in_dim, w.output.raw, w.output.type,
      w.output.out_dim, Array(Float32).new(w.output.out_dim, 0.0_f32),
    )
    expected_id = cpu_logits.index(cpu_logits.max).not_nil!.to_i32

    cmd = ML::Metal::CommandBuffer.new
    ML::GGUF::Qwen35Metal.encode_rmsnorm_project_top1_buffer(
      cmd, resident, hidden_dim.to_i64,
      w.output_norm, w.output, w.hparams.rms_eps,
      top1_id, top1_value,
    ).should be_true
    cmd.committed?.should be_false

    cmd.commit
    cmd.wait
    actual = ML::GGUF::Qwen35Metal.read_head_top1_buffers(top1_id, top1_value)
    actual[0].to_i32.should eq(expected_id)
    actual[1].should be_close(cpu_logits[expected_id], 1.0e-4_f32)

    completed = ML::Metal::CommandBuffer.new
    completed.commit
    completed.wait
    ML::GGUF::Qwen35Metal.encode_rmsnorm_project_top1_buffer(
      completed, resident, hidden_dim.to_i64,
      w.output_norm, w.output, w.hparams.rms_eps,
      top1_id, top1_value,
    ).should be_false
  end

  it "appends resident top-1 rows without opening a second command" do
    pending!("Metal not available") unless ML::GGUF::Qwen35Metal.available?

    with_qwen35_head_top1_fused do
      w = ML::GGUF::Qwen35Weights.from_gguf(QWEN_9B_FWD)
      hidden_dim = w.hparams.n_embd
      rows = 3
      hidden = Array(Float32).new(rows * hidden_dim) do |i|
        ((i % 29) - 14).to_f32 / 29.0_f32
      end
      resident = ML::MetalBuffer.new(hidden.size.to_i64 * sizeof(Float32))
      resident.write(hidden)
      expected = ML::GGUF::Qwen35Metal.rmsnorm_project_top1_rows_buffer(
        resident, rows, w.output_norm, w.output, w.hparams.rms_eps,
      ).not_nil!
      top1_ids = ML::MetalBuffer.new(rows.to_i64 * sizeof(UInt32))
      top1_values = ML::MetalBuffer.new(rows.to_i64 * sizeof(Float32))
      arena = ML::GGUF::Qwen35Metal::Scratch::Arena.new("row_head_direct_spec")

      cmd = ML::Metal::CommandBuffer.new
      ML::GGUF::Qwen35Metal.encode_rmsnorm_project_top1_rows_buffer(
        cmd, arena, resident, rows,
        w.output_norm, w.output, w.hparams.rms_eps,
        top1_ids, top1_values,
      ).should be_true
      cmd.committed?.should be_false

      cmd.commit
      cmd.wait
      actual = ML::GGUF::Qwen35Metal.read_head_top1_rows_buffers(
        top1_ids, top1_values, rows,
      )
      actual.map(&.[0]).should eq(expected.map(&.[0]))
      actual.each_with_index do |(_, value), i|
        value.should be_close(expected[i][1], 1.0e-6_f32)
      end

      completed = ML::Metal::CommandBuffer.new
      completed.commit
      completed.wait
      ML::GGUF::Qwen35Metal.encode_rmsnorm_project_top1_rows_buffer(
        completed, arena, resident, rows,
        w.output_norm, w.output, w.hparams.rms_eps,
        top1_ids, top1_values,
      ).should be_false
      undersized_ids = ML::MetalBuffer.new((rows - 1).to_i64 * sizeof(UInt32))
      fresh = ML::Metal::CommandBuffer.new
      ML::GGUF::Qwen35Metal.encode_rmsnorm_project_top1_rows_buffer(
        fresh, arena, resident, rows,
        w.output_norm, w.output, w.hparams.rms_eps,
        undersized_ids, top1_values,
      ).should be_false
      fresh.committed?.should be_false

      scratch_count = arena.buffers.size
      ENV["QWEN35_HEAD_TOP1_FUSED"] = "0"
      disabled = ML::Metal::CommandBuffer.new
      ML::GGUF::Qwen35Metal.encode_rmsnorm_project_top1_rows_buffer(
        disabled, arena, resident, rows,
        w.output_norm, w.output, w.hparams.rms_eps,
        top1_ids, top1_values,
      ).should be_false
      ENV["QWEN35_HEAD_TOP1_FUSED"] = "1"
      invalid_rows = ML::Metal::CommandBuffer.new
      ML::GGUF::Qwen35Metal.encode_rmsnorm_project_top1_rows_buffer(
        invalid_rows, arena, resident, 0,
        w.output_norm, w.output, w.hparams.rms_eps,
        top1_ids, top1_values,
      ).should be_false
      undersized_x = ML::MetalBuffer.new(sizeof(Float32).to_i64)
      invalid_x = ML::Metal::CommandBuffer.new
      ML::GGUF::Qwen35Metal.encode_rmsnorm_project_top1_rows_buffer(
        invalid_x, arena, undersized_x, rows,
        w.output_norm, w.output, w.hparams.rms_eps,
        top1_ids, top1_values,
      ).should be_false
      undersized_values = ML::MetalBuffer.new(sizeof(Float32).to_i64)
      invalid_values = ML::Metal::CommandBuffer.new
      ML::GGUF::Qwen35Metal.encode_rmsnorm_project_top1_rows_buffer(
        invalid_values, arena, resident, rows,
        w.output_norm, w.output, w.hparams.rms_eps,
        top1_ids, undersized_values,
      ).should be_false
      arena.buffers.size.should eq(scratch_count)
      [disabled, invalid_rows, invalid_x, invalid_values].each(&.committed?.should(be_false))
      arena.release
    end
  end

  it "keeps equal-size resident row-head flights scratch-disjoint until completion" do
    pending!("Metal not available") unless ML::GGUF::Qwen35Metal.available?

    old_head = ENV["QWEN35_HEAD_TOP1_FUSED"]?
    ENV["QWEN35_HEAD_TOP1_FUSED"] = "1"
    first_cmd = nil.as(ML::Metal::CommandBuffer?)
    second_cmd = nil.as(ML::Metal::CommandBuffer?)
    first_arena = ML::GGUF::Qwen35Metal::Scratch::Arena.new("row_head_flight_spec:0")
    second_arena = ML::GGUF::Qwen35Metal::Scratch::Arena.new("row_head_flight_spec:1")

    begin
      weights = ML::GGUF::Qwen35Weights.from_gguf(QWEN_9B_FWD)
      hidden_dim = weights.hparams.n_embd
      rows = 3
      first_hidden = Array(Float32).new(rows * hidden_dim) do |i|
        ((i % 29) - 14).to_f32 / 29.0_f32
      end
      second_hidden = Array(Float32).new(rows * hidden_dim) do |i|
        ((i % 37) - 18).to_f32 / 37.0_f32
      end
      first_resident = ML::MetalBuffer.new(first_hidden.size.to_i64 * sizeof(Float32))
      second_resident = ML::MetalBuffer.new(second_hidden.size.to_i64 * sizeof(Float32))
      first_resident.write(first_hidden)
      second_resident.write(second_hidden)
      expected_first = ML::GGUF::Qwen35Metal.rmsnorm_project_top1_rows_buffer(
        first_resident, rows, weights.output_norm, weights.output, weights.hparams.rms_eps,
      ).not_nil!
      expected_second = ML::GGUF::Qwen35Metal.rmsnorm_project_top1_rows_buffer(
        second_resident, rows, weights.output_norm, weights.output, weights.hparams.rms_eps,
      ).not_nil!
      first_ids = ML::MetalBuffer.new(rows.to_i64 * sizeof(UInt32))
      first_values = ML::MetalBuffer.new(rows.to_i64 * sizeof(Float32))
      second_ids = ML::MetalBuffer.new(rows.to_i64 * sizeof(UInt32))
      second_values = ML::MetalBuffer.new(rows.to_i64 * sizeof(Float32))

      first_cmd = ML::Metal::CommandBuffer.new(queue: ML::Metal::CommandQueue.new)
      second_cmd = ML::Metal::CommandBuffer.new(queue: ML::Metal::CommandQueue.new)
      ML::GGUF::Qwen35Metal.encode_rmsnorm_project_top1_rows_buffer(
        first_cmd.not_nil!, first_arena, first_resident, rows,
        weights.output_norm, weights.output, weights.hparams.rms_eps,
        first_ids, first_values,
      ).should be_true
      ML::GGUF::Qwen35Metal.encode_rmsnorm_project_top1_rows_buffer(
        second_cmd.not_nil!, second_arena, second_resident, rows,
        weights.output_norm, weights.output, weights.hparams.rms_eps,
        second_ids, second_values,
      ).should be_true
      first_arena.buffers.size.should eq(4)
      second_arena.buffers.size.should eq(4)
      first_arena.buffers.each do |first_buffer|
        second_arena.buffers.each do |second_buffer|
          first_buffer.handle.should_not eq(second_buffer.handle)
        end
      end

      first_cmd.not_nil!.commit
      second_cmd.not_nil!.commit
      first_cmd.not_nil!.wait
      second_cmd.not_nil!.wait
      actual_first = ML::GGUF::Qwen35Metal.read_head_top1_rows_buffers(first_ids, first_values, rows)
      actual_second = ML::GGUF::Qwen35Metal.read_head_top1_rows_buffers(second_ids, second_values, rows)
      actual_first.map(&.[0]).should eq(expected_first.map(&.[0]))
      actual_second.map(&.[0]).should eq(expected_second.map(&.[0]))
      actual_first.each_with_index { |(_, value), i| value.should be_close(expected_first[i][1], 1.0e-6_f32) }
      actual_second.each_with_index { |(_, value), i| value.should be_close(expected_second[i][1], 1.0e-6_f32) }
    ensure
      [first_cmd, second_cmd].each do |command|
        next unless command
        begin
          command.wait if command.committed? && !command.completed?
        rescue
        end
      end
      first_arena.release
      second_arena.release
      if old_head
        ENV["QWEN35_HEAD_TOP1_FUSED"] = old_head
      else
        ENV.delete("QWEN35_HEAD_TOP1_FUSED")
      end
    end
  end

  it "appends a multi-tile allowed resident top-1 head without weakening to global top-1" do
    pending!("Metal not available") unless ML::GGUF::Qwen35Metal.available?

    w = ML::GGUF::Qwen35Weights.from_gguf(QWEN_9B_FWD)
    hidden_dim = w.hparams.n_embd
    first = Array(Float32).new(hidden_dim) { |i| ((i % 19) - 9).to_f32 / 19.0_f32 }
    second = Array(Float32).new(hidden_dim) { |i| ((i % 27) - 13).to_f32 / 27.0_f32 }
    resident = ML::MetalBuffer.new((2 * hidden_dim).to_i64 * sizeof(Float32))
    resident.write(first + second)
    top1_id = ML::MetalBuffer.new(sizeof(UInt32).to_i64)
    top1_value = ML::MetalBuffer.new(sizeof(Float32).to_i64)

    normalized = ML::GGUF::Qwen35CPU.rms_norm(
      second, w.output_norm, w.hparams.rms_eps,
    )
    cpu_logits = ML::GGUF::QuantMatmul.matmul_add(
      normalized, 1, w.output.in_dim, w.output.raw, w.output.type,
      w.output.out_dim, Array(Float32).new(w.output.out_dim, 0.0_f32),
    )
    full_top = cpu_logits.index(cpu_logits.max).not_nil!.to_i32
    # HEAD_TOP1_ROWS_PER_TG is 12. Thirteen candidates exercise both the tile
    # kernel and its reducer, while excluding the unrestricted global winner.
    allowed_ids = Array(Int32).new(13) do |offset|
      (full_top + offset + 1) % w.output.out_dim
    end
    expected_id = allowed_ids.max_by { |id| cpu_logits[id] }
    allowed_ids.includes?(full_top).should be_false
    expected_id.should_not eq(full_top)

    ML::GGUF::Qwen35Metal.rmsnorm_project_top1_allowed_ids_supported?(
      w.output,
    ).should be_true

    cmd = ML::Metal::CommandBuffer.new
    ML::GGUF::Qwen35Metal.encode_rmsnorm_project_top1_allowed_ids_buffer(
      cmd, resident, hidden_dim.to_i64,
      w.output_norm, w.output, w.hparams.rms_eps, allowed_ids,
      top1_id, top1_value,
    ).should be_true
    cmd.committed?.should be_false

    cmd.commit
    cmd.wait
    actual = ML::GGUF::Qwen35Metal.read_head_top1_buffers(top1_id, top1_value)
    actual[0].to_i32.should eq(expected_id)
    actual[1].should be_close(cpu_logits[expected_id], 1.0e-4_f32)

    completed = ML::Metal::CommandBuffer.new
    completed.commit
    completed.wait
    ML::GGUF::Qwen35Metal.encode_rmsnorm_project_top1_allowed_ids_buffer(
      completed, resident, hidden_dim.to_i64,
      w.output_norm, w.output, w.hparams.rms_eps, allowed_ids,
      top1_id, top1_value,
    ).should be_false

    empty_ids_cmd = ML::Metal::CommandBuffer.new
    ML::GGUF::Qwen35Metal.encode_rmsnorm_project_top1_allowed_ids_buffer(
      empty_ids_cmd, resident, hidden_dim.to_i64,
      w.output_norm, w.output, w.hparams.rms_eps, [] of Int32,
      top1_id, top1_value,
    ).should be_false
    empty_ids_cmd.committed?.should be_false

    invalid_id_cmd = ML::Metal::CommandBuffer.new
    ML::GGUF::Qwen35Metal.encode_rmsnorm_project_top1_allowed_ids_buffer(
      invalid_id_cmd, resident, hidden_dim.to_i64,
      w.output_norm, w.output, w.hparams.rms_eps, [-1_i32],
      top1_id, top1_value,
    ).should be_false
    invalid_id_cmd.committed?.should be_false

    out_of_range_cmd = ML::Metal::CommandBuffer.new
    ML::GGUF::Qwen35Metal.encode_rmsnorm_project_top1_allowed_ids_buffer(
      out_of_range_cmd, resident, hidden_dim.to_i64,
      w.output_norm, w.output, w.hparams.rms_eps, [w.output.out_dim],
      top1_id, top1_value,
    ).should be_false
    out_of_range_cmd.committed?.should be_false

    source_bounds_cmd = ML::Metal::CommandBuffer.new
    ML::GGUF::Qwen35Metal.encode_rmsnorm_project_top1_allowed_ids_buffer(
      source_bounds_cmd, resident, hidden_dim.to_i64 + 1_i64,
      w.output_norm, w.output, w.hparams.rms_eps, allowed_ids,
      top1_id, top1_value,
    ).should be_false
    source_bounds_cmd.committed?.should be_false
  end

  it "falls back to full-logit argmax when fused greedy head is disabled" do
    w = ML::GGUF::Qwen35Weights.from_gguf(QWEN_9B_FWD)
    hp = w.hparams

    state_full = ML::GGUF::Qwen35CPU::State.new(hp, max_seq: 32)
    logits = ML::GGUF::Qwen35CPU.forward(w, 0, 0, state_full)
    full_max = logits.max
    full_top = logits.index(full_max).not_nil!.to_i32

    old = ENV["QWEN35_HEAD_TOP1_FUSED"]?
    ENV["QWEN35_HEAD_TOP1_FUSED"] = "0"
    begin
      state_top1 = ML::GGUF::Qwen35CPU::State.new(hp, max_seq: 32)
      top_id, top_logit = ML::GGUF::Qwen35CPU.forward_top1(w, 0, 0, state_top1)
      top_id.should eq(full_top)
      top_logit.should be_close(full_max, 1e-4_f32)
    ensure
      if old
        ENV["QWEN35_HEAD_TOP1_FUSED"] = old
      else
        ENV.delete("QWEN35_HEAD_TOP1_FUSED")
      end
    end
  end

  it "prefills non-final prompt tokens without changing final top1" do
    w = ML::GGUF::Qwen35Weights.from_gguf(QWEN_9B_FWD)
    hp = w.hparams
    prompt = [760_i32, 6511_i32, 314_i32, 9338_i32, 369_i32] # "The capital of France is"

    live = ML::GGUF::Qwen35CPU::State.new(hp, max_seq: 32)
    final_top = 0_i32
    final_logit = 0.0_f32
    prompt.each_with_index do |token_id, pos|
      final_top, final_logit = ML::GGUF::Qwen35CPU.forward_top1(w, token_id, pos.to_i32, live)
    end

    prefilled = ML::GGUF::Qwen35CPU::State.new(hp, max_seq: 32)
    prompt[0...-1].each_with_index do |token_id, pos|
      ML::GGUF::Qwen35CPU.prefill_token(w, token_id, pos.to_i32, prefilled)
    end
    top, logit = ML::GGUF::Qwen35CPU.forward_top1(w, prompt.last, (prompt.size - 1).to_i32, prefilled)

    live_top, live_logit = ML::GGUF::Qwen35CPU.forward_top1(w, 11751_i32, prompt.size.to_i32, live)
    prefill_top, prefill_logit = ML::GGUF::Qwen35CPU.forward_top1(w, 11751_i32, prompt.size.to_i32, prefilled)

    top.should eq(final_top)
    logit.should be_close(final_logit, 1e-4_f32)
    prefill_top.should eq(live_top)
    prefill_logit.should be_close(live_logit, 1e-4_f32)
  end

  it "chunk-prefills non-final prompt tokens without changing final top1" do
    w = ML::GGUF::Qwen35Weights.from_gguf(QWEN_9B_FWD)
    hp = w.hparams
    prompt = [760_i32, 6511_i32, 314_i32, 9338_i32, 369_i32]

    serial = ML::GGUF::Qwen35CPU::State.new(hp, max_seq: 32)
    prompt[0...-1].each_with_index do |token_id, pos|
      ML::GGUF::Qwen35CPU.prefill_token(w, token_id, pos.to_i32, serial)
    end
    serial_top, serial_logit = ML::GGUF::Qwen35CPU.forward_top1(w, prompt.last, (prompt.size - 1).to_i32, serial)

    old = ENV["QWEN35_PREFILL_CHUNK_OFF"]?
    ENV.delete("QWEN35_PREFILL_CHUNK_OFF")
    begin
      chunked = ML::GGUF::Qwen35CPU::State.new(hp, max_seq: 32)
      ML::GGUF::Qwen35CPU.prefill_tokens(w, prompt[0...-1], 0, chunked)
      chunk_top, chunk_logit = ML::GGUF::Qwen35CPU.forward_top1(w, prompt.last, (prompt.size - 1).to_i32, chunked)

      chunk_top.should eq(serial_top)
      chunk_logit.should be_close(serial_logit, 1e-4_f32)
    ensure
      if old
        ENV["QWEN35_PREFILL_CHUNK_OFF"] = old
      else
        ENV.delete("QWEN35_PREFILL_CHUNK_OFF")
      end
    end
  end

  it "chunk-prefill remains deterministic after cached Metal constants are reused" do
    w = ML::GGUF::Qwen35Weights.from_gguf(QWEN_9B_FWD)
    hp = w.hparams
    prompt = [760_i32, 6511_i32, 314_i32, 9338_i32, 369_i32, 279_i32, 9821_i32, 13_i32]

    old = ENV["QWEN35_PREFILL_CHUNK_OFF"]?
    ENV.delete("QWEN35_PREFILL_CHUNK_OFF")
    begin
      first = ML::GGUF::Qwen35CPU::State.new(hp, max_seq: 32)
      second = ML::GGUF::Qwen35CPU::State.new(hp, max_seq: 32)

      ML::GGUF::Qwen35CPU.prefill_tokens(w, prompt[0...-1], 0, first)
      first_top, first_logit = ML::GGUF::Qwen35CPU.forward_top1(w, prompt.last, (prompt.size - 1).to_i32, first)

      ML::GGUF::Qwen35CPU.prefill_tokens(w, prompt[0...-1], 0, second)
      second_top, second_logit = ML::GGUF::Qwen35CPU.forward_top1(w, prompt.last, (prompt.size - 1).to_i32, second)

      second_top.should eq(first_top)
      second_logit.should be_close(first_logit, 1e-4_f32)
    ensure
      if old
        ENV["QWEN35_PREFILL_CHUNK_OFF"] = old
      else
        ENV.delete("QWEN35_PREFILL_CHUNK_OFF")
      end
    end
  end

  it "final full-attention last-row prefill matches the full final-layer fallback" do
    w = ML::GGUF::Qwen35Weights.from_gguf(QWEN_9B_FWD)
    hp = w.hparams
    prompt = [760_i32, 6511_i32, 314_i32, 9338_i32, 369_i32, 279_i32, 9821_i32, 13_i32]

    old_chunk = ENV["QWEN35_PREFILL_CHUNK_OFF"]?
    old_final = ENV["QWEN35_FINAL_FULL_LAST_OFF"]?
    ENV.delete("QWEN35_PREFILL_CHUNK_OFF")
    begin
      undersized = ML::GGUF::Qwen35CPU::State.new(hp, max_seq: 32)
      undersized.layers[hp.full_attention_layers.first].k_cache = [0.0_f32]
      ML::GGUF::Qwen35CPU.prefill_full_logits_last_supported?(w, undersized, prompt.size, 0).should be_false

      stale = ML::GGUF::Qwen35CPU::State.new(hp, max_seq: 32)
      stale.layers.first.position = 1
      ML::GGUF::Qwen35CPU.prefill_full_logits_last_supported?(w, stale, prompt.size, 0).should be_false

      fast = ML::GGUF::Qwen35CPU::State.new(hp, max_seq: 32)
      ENV.delete("QWEN35_FINAL_FULL_LAST_OFF")
      fast_top, fast_logit = ML::GGUF::Qwen35CPU.prefill_tokens_top1(w, prompt, 0, fast)
      fast_next_top, fast_next_logit = ML::GGUF::Qwen35CPU.forward_top1(w, 11751_i32, prompt.size.to_i32, fast)

      fallback = ML::GGUF::Qwen35CPU::State.new(hp, max_seq: 32)
      ENV["QWEN35_FINAL_FULL_LAST_OFF"] = "1"
      fallback_top, fallback_logit = ML::GGUF::Qwen35CPU.prefill_tokens_top1(w, prompt, 0, fallback)
      fallback_next_top, fallback_next_logit = ML::GGUF::Qwen35CPU.forward_top1(w, 11751_i32, prompt.size.to_i32, fallback)

      fast_top.should eq(fallback_top)
      fast_logit.should be_close(fallback_logit, 1e-4_f32)
      fast_next_top.should eq(fallback_next_top)
      fast_next_logit.should be_close(fallback_next_logit, 1e-4_f32)
    ensure
      if old_chunk
        ENV["QWEN35_PREFILL_CHUNK_OFF"] = old_chunk
      else
        ENV.delete("QWEN35_PREFILL_CHUNK_OFF")
      end

      if old_final
        ENV["QWEN35_FINAL_FULL_LAST_OFF"] = old_final
      else
        ENV.delete("QWEN35_FINAL_FULL_LAST_OFF")
      end
    end
  end

  it "final-row full-logit prefill preserves the complete logits and continuation" do
    w = ML::GGUF::Qwen35Weights.from_gguf(QWEN_9B_FWD)
    hp = w.hparams
    prompt = [760_i32, 6511_i32, 314_i32, 9338_i32, 369_i32, 279_i32, 9821_i32, 13_i32]
    continuation = [11751_i32, 42_i32, 997_i32, 314_i32]

    old_chunk = ENV["QWEN35_PREFILL_CHUNK_OFF"]?
    old_final = ENV["QWEN35_FINAL_FULL_LAST_OFF"]?
    ENV.delete("QWEN35_PREFILL_CHUNK_OFF")
    begin
      fast = ML::GGUF::Qwen35CPU::State.new(hp, max_seq: 32)
      ENV.delete("QWEN35_FINAL_FULL_LAST_OFF")
      ML::GGUF::Qwen35CPU.prefill_full_logits_last_supported?(w, fast, prompt.size, 0).should be_true
      fast_route = [false]
      fast_logits = ML::GGUF::Qwen35CPU.prefill_tokens_logits(w, prompt, 0, fast, fast_route)
      fast_route[0].should be_true

      fallback = ML::GGUF::Qwen35CPU::State.new(hp, max_seq: 32)
      ENV["QWEN35_FINAL_FULL_LAST_OFF"] = "1"
      ML::GGUF::Qwen35CPU.prefill_full_logits_last_supported?(w, fallback, prompt.size, 0).should be_false
      fallback_route = [true]
      fallback_logits = ML::GGUF::Qwen35CPU.prefill_tokens_logits(w, prompt, 0, fallback, fallback_route)
      fallback_route[0].should be_false

      fast_logits.size.should eq(fallback_logits.size)
      fast_logits.each_with_index do |value, index|
        value.should be_close(fallback_logits[index], 1.0e-4_f32)
      end

      continuation.each_with_index do |token, index|
        pos = prompt.size.to_i32 + index
        fast_next = ML::GGUF::Qwen35CPU.forward(w, token, pos, fast)
        fallback_next = ML::GGUF::Qwen35CPU.forward(w, token, pos, fallback)
        fast_next.size.should eq(fallback_next.size)
        fast_next.each_with_index do |value, logit_index|
          value.should be_close(fallback_next[logit_index], 1.0e-4_f32)
        end
      end
    ensure
      if old_chunk
        ENV["QWEN35_PREFILL_CHUNK_OFF"] = old_chunk
      else
        ENV.delete("QWEN35_PREFILL_CHUNK_OFF")
      end

      if old_final
        ENV["QWEN35_FINAL_FULL_LAST_OFF"] = old_final
      else
        ENV.delete("QWEN35_FINAL_FULL_LAST_OFF")
      end
    end
  end

  it "long prompt suffix chunk top1 matches final-token fallback" do
    w = ML::GGUF::Qwen35Weights.from_gguf(QWEN_9B_FWD)
    hp = w.hparams
    prompt = [760_i32, 6511_i32, 314_i32, 9338_i32, 369_i32, 279_i32, 9821_i32, 13_i32]

    old_chunk_off = ENV["QWEN35_PREFILL_CHUNK_OFF"]?
    old_chunk_size = ENV["QWEN35_PREFILL_CHUNK_SIZE"]?
    old_long = ENV["QWEN35_PREFILL_LONG_SUFFIX_OFF"]?
    ENV.delete("QWEN35_PREFILL_CHUNK_OFF")
    ENV["QWEN35_PREFILL_CHUNK_SIZE"] = "4"
    begin
      fast = ML::GGUF::Qwen35CPU::State.new(hp, max_seq: 32)
      ENV.delete("QWEN35_PREFILL_LONG_SUFFIX_OFF")
      fast_top, fast_logit = ML::GGUF::Qwen35CPU.prefill_tokens_top1(w, prompt, 0, fast)
      fast_next_top, fast_next_logit = ML::GGUF::Qwen35CPU.forward_top1(w, 11751_i32, prompt.size.to_i32, fast)

      fallback = ML::GGUF::Qwen35CPU::State.new(hp, max_seq: 32)
      ENV["QWEN35_PREFILL_LONG_SUFFIX_OFF"] = "1"
      fallback_top, fallback_logit = ML::GGUF::Qwen35CPU.prefill_tokens_top1(w, prompt, 0, fallback)
      fallback_next_top, fallback_next_logit = ML::GGUF::Qwen35CPU.forward_top1(w, 11751_i32, prompt.size.to_i32, fallback)

      fast_top.should eq(fallback_top)
      fast_logit.should be_close(fallback_logit, 1e-4_f32)
      fast_next_top.should eq(fallback_next_top)
      fast_next_logit.should be_close(fallback_next_logit, 1e-4_f32)
    ensure
      if old_chunk_off
        ENV["QWEN35_PREFILL_CHUNK_OFF"] = old_chunk_off
      else
        ENV.delete("QWEN35_PREFILL_CHUNK_OFF")
      end

      if old_chunk_size
        ENV["QWEN35_PREFILL_CHUNK_SIZE"] = old_chunk_size
      else
        ENV.delete("QWEN35_PREFILL_CHUNK_SIZE")
      end

      if old_long
        ENV["QWEN35_PREFILL_LONG_SUFFIX_OFF"] = old_long
      else
        ENV.delete("QWEN35_PREFILL_LONG_SUFFIX_OFF")
      end
    end
  end

  it "bounded shared prefill commands preserve the next-token result" do
    w = ML::GGUF::Qwen35Weights.from_gguf(QWEN_9B_FWD)
    hp = w.hparams
    prompt = [760_i32, 6511_i32, 314_i32, 9338_i32, 369_i32, 279_i32, 9821_i32, 13_i32]

    old_limit = ENV["QWEN35_PREFILL_APPEND_MAX_GROUPS"]?
    old_cooldown = ENV["QWEN35_PREFILL_APPEND_COOLDOWN_MS"]?
    old_chunk = ENV["QWEN35_PREFILL_CHUNK_SIZE"]?
    ENV["QWEN35_PREFILL_CHUNK_SIZE"] = "64"
    ENV.delete("QWEN35_PREFILL_APPEND_COOLDOWN_MS")
    begin
      unbounded = ML::GGUF::Qwen35CPU::State.new(hp, max_seq: 32)
      ENV["QWEN35_PREFILL_APPEND_MAX_GROUPS"] = "0"
      unbounded_top, unbounded_logit = ML::GGUF::Qwen35CPU.prefill_tokens_top1(w, prompt, 0, unbounded)
      unbounded_next_top, unbounded_next_logit = ML::GGUF::Qwen35CPU.forward_top1(w, 11751_i32, prompt.size.to_i32, unbounded)

      bounded = ML::GGUF::Qwen35CPU::State.new(hp, max_seq: 32)
      ENV["QWEN35_PREFILL_APPEND_MAX_GROUPS"] = "1"
      bounded_top, bounded_logit = ML::GGUF::Qwen35CPU.prefill_tokens_top1(w, prompt, 0, bounded)
      bounded_next_top, bounded_next_logit = ML::GGUF::Qwen35CPU.forward_top1(w, 11751_i32, prompt.size.to_i32, bounded)

      bounded_top.should eq(unbounded_top)
      bounded_logit.should be_close(unbounded_logit, 1e-4_f32)
      bounded_next_top.should eq(unbounded_next_top)
      bounded_next_logit.should be_close(unbounded_next_logit, 1e-4_f32)
    ensure
      if old_limit
        ENV["QWEN35_PREFILL_APPEND_MAX_GROUPS"] = old_limit
      else
        ENV.delete("QWEN35_PREFILL_APPEND_MAX_GROUPS")
      end
      if old_cooldown
        ENV["QWEN35_PREFILL_APPEND_COOLDOWN_MS"] = old_cooldown
      else
        ENV.delete("QWEN35_PREFILL_APPEND_COOLDOWN_MS")
      end
      if old_chunk
        ENV["QWEN35_PREFILL_CHUNK_SIZE"] = old_chunk
      else
        ENV.delete("QWEN35_PREFILL_CHUNK_SIZE")
      end
    end
  end

  it "CogniGraph depth two preserves exact prefill and append results" do
    pending!("Metal not available") unless ML::GGUF::Qwen35Metal.available?

    w = ML::GGUF::Qwen35Weights.from_gguf(QWEN_9B_FWD)
    hp = w.hparams
    prompt = [760_i32, 6511_i32, 314_i32, 9338_i32, 369_i32, 279_i32, 9821_i32, 13_i32]
    keys = [
      "QWEN35_COGNIGRAPH_PREFILL_MAX_INFLIGHT",
      "QWEN35_PREFILL_APPEND_MAX_GROUPS",
      "QWEN35_PREFILL_APPEND_COOLDOWN_MS",
      "QWEN35_PREFILL_CHUNK_SIZE",
      "QWEN35_PREFILL_BOUNDARY_PROFILE",
    ]
    old = keys.to_h { |key| {key, ENV[key]?} }
    ENV["QWEN35_PREFILL_APPEND_MAX_GROUPS"] = "1"
    ENV["QWEN35_PREFILL_APPEND_COOLDOWN_MS"] = "0"
    ENV["QWEN35_PREFILL_CHUNK_SIZE"] = "64"
    ENV.delete("QWEN35_PREFILL_BOUNDARY_PROFILE")
    begin
      baseline = ML::GGUF::Qwen35CPU::State.new(hp, max_seq: 32)
      ENV["QWEN35_COGNIGRAPH_PREFILL_MAX_INFLIGHT"] = "0"
      baseline_top, baseline_logit = ML::GGUF::Qwen35CPU.prefill_tokens_top1(w, prompt, 0, baseline)
      baseline_next_top, baseline_next_logit = ML::GGUF::Qwen35CPU.forward_top1(w, 11751_i32, prompt.size.to_i32, baseline)

      graph = ML::GGUF::Qwen35CPU::State.new(hp, max_seq: 32)
      ENV["QWEN35_COGNIGRAPH_PREFILL_MAX_INFLIGHT"] = "2"
      graph_top, graph_logit = ML::GGUF::Qwen35CPU.prefill_tokens_top1(w, prompt, 0, graph)
      graph_next_top, graph_next_logit = ML::GGUF::Qwen35CPU.forward_top1(w, 11751_i32, prompt.size.to_i32, graph)

      graph_top.should eq(baseline_top)
      graph_logit.should be_close(baseline_logit, 1e-4_f32)
      graph_next_top.should eq(baseline_next_top)
      graph_next_logit.should be_close(baseline_next_logit, 1e-4_f32)
    ensure
      old.each do |key, value|
        if value
          ENV[key] = value
        else
          ENV.delete(key)
        end
      end
    end
  end

  it "chunked top1 verifier matches serial greedy target steps" do
    w = ML::GGUF::Qwen35Weights.from_gguf(QWEN_9B_FWD)
    hp = w.hparams
    prompt = [760_i32, 6511_i32, 314_i32, 9338_i32, 13_i32]
    candidates = [11751_i32, 318_i32, 279_i32, 9821_i32]

    prefix_serial = ML::GGUF::Qwen35CPU::State.new(hp, max_seq: 32)
    ML::GGUF::Qwen35CPU.prefill_tokens(w, prompt, 0, prefix_serial)
    prefix_chunk = prefix_serial.fork

    serial = [] of {Int32, Float32}
    candidates.each_with_index do |token_id, i|
      serial << ML::GGUF::Qwen35CPU.forward_top1(w, token_id, prompt.size + i, prefix_serial)
    end

    chunked = ML::GGUF::Qwen35CPU.prefill_tokens_top1s(w, candidates, prompt.size, prefix_chunk)
    chunked.size.should eq(serial.size)
    chunked.each_with_index do |(top_id, top_logit), i|
      top_id.should eq(serial[i][0])
      top_logit.should be_close(serial[i][1], 1e-4_f32)
    end

    next_serial = ML::GGUF::Qwen35CPU.forward_top1(w, 11751_i32, prompt.size + candidates.size, prefix_serial)
    next_chunk = ML::GGUF::Qwen35CPU.forward_top1(w, 11751_i32, prompt.size + candidates.size, prefix_chunk)
    next_chunk[0].should eq(next_serial[0])
    next_chunk[1].should be_close(next_serial[1], 1e-4_f32)

    old_rows = ENV["QWEN35_HEAD_TOP1_ROWS"]?
    ENV["QWEN35_HEAD_TOP1_ROWS"] = "1"
    begin
      prefix_batched = ML::GGUF::Qwen35CPU::State.new(hp, max_seq: 32)
      ML::GGUF::Qwen35CPU.prefill_tokens(w, prompt, 0, prefix_batched)
      batched = ML::GGUF::Qwen35CPU.prefill_tokens_top1s(w, candidates, prompt.size, prefix_batched)
      batched.size.should eq(serial.size)
      batched.each_with_index do |(top_id, top_logit), i|
        top_id.should eq(serial[i][0])
        top_logit.should be_close(serial[i][1], 1e-4_f32)
      end
    ensure
      if old_rows
        ENV["QWEN35_HEAD_TOP1_ROWS"] = old_rows
      else
        ENV.delete("QWEN35_HEAD_TOP1_ROWS")
      end
    end
  end

  it "keeps chunked top1 verifier exact on large ambiguous verifier rows" do
    w = ML::GGUF::Qwen35Weights.from_gguf(QWEN_9B_FWD)
    hp = w.hparams
    prompt = Array(Int32).new(5) { |i| ((i * 7 + 11) % 1000).to_i32 }
    candidates = Array(Int32).new(32) { |i| ((i * 13 + 11751) % 1000).to_i32 }

    old_full = ENV["QWEN35_HEAD_FULL_ROWS"]?
    old_full_off = ENV["QWEN35_HEAD_FULL_ROWS_OFF"]?
    old_rows = ENV["QWEN35_HEAD_TOP1_ROWS"]?
    ENV.delete("QWEN35_HEAD_FULL_ROWS")
    ENV.delete("QWEN35_HEAD_FULL_ROWS_OFF")
    ENV.delete("QWEN35_HEAD_TOP1_ROWS")
    begin
      serial_state = ML::GGUF::Qwen35CPU::State.new(hp, max_seq: prompt.size + candidates.size + 8)
      ML::GGUF::Qwen35CPU.prefill_tokens(w, prompt, 0, serial_state)
      serial = [] of {Int32, Float32}
      candidates.each_with_index do |token_id, i|
        serial << ML::GGUF::Qwen35CPU.forward_top1(w, token_id, prompt.size + i, serial_state)
      end

      chunk_state = ML::GGUF::Qwen35CPU::State.new(hp, max_seq: prompt.size + candidates.size + 8)
      ML::GGUF::Qwen35CPU.prefill_tokens(w, prompt, 0, chunk_state)
      chunked = ML::GGUF::Qwen35CPU.prefill_tokens_top1s(w, candidates, prompt.size, chunk_state)

      chunked.size.should eq(serial.size)
      chunked.each_with_index do |(top_id, _top_logit), i|
        top_id.should eq(serial[i][0])
      end
    ensure
      if old_full
        ENV["QWEN35_HEAD_FULL_ROWS"] = old_full
      else
        ENV.delete("QWEN35_HEAD_FULL_ROWS")
      end
      if old_full_off
        ENV["QWEN35_HEAD_FULL_ROWS_OFF"] = old_full_off
      else
        ENV.delete("QWEN35_HEAD_FULL_ROWS_OFF")
      end
      if old_rows
        ENV["QWEN35_HEAD_TOP1_ROWS"] = old_rows
      else
        ENV.delete("QWEN35_HEAD_TOP1_ROWS")
      end
    end
  end

  it "chunk-off last hidden matches the chunked prompt boundary" do
    w = ML::GGUF::Qwen35Weights.from_gguf(QWEN_9B_FWD)
    hp = w.hparams
    prompt = [760_i32, 6511_i32, 314_i32, 9338_i32, 369_i32]

    old = ENV["QWEN35_PREFILL_CHUNK_OFF"]?
    begin
      ENV.delete("QWEN35_PREFILL_CHUNK_OFF")
      chunk_state = ML::GGUF::Qwen35CPU::State.new(hp, max_seq: 32)
      chunk_hidden = ML::GGUF::Qwen35CPU.prefill_tokens_last_hidden(w, prompt, 0, chunk_state)
      chunk_top, chunk_logit = ML::GGUF::Qwen35CPU.hidden_top1(w, chunk_hidden)
      chunk_next_top, chunk_next_logit = ML::GGUF::Qwen35CPU.forward_top1(w, chunk_top, prompt.size.to_i32, chunk_state)

      ENV["QWEN35_PREFILL_CHUNK_OFF"] = "1"
      serial_state = ML::GGUF::Qwen35CPU::State.new(hp, max_seq: 32)
      serial_hidden = ML::GGUF::Qwen35CPU.prefill_tokens_last_hidden(w, prompt, 0, serial_state)
      serial_top, serial_logit = ML::GGUF::Qwen35CPU.hidden_top1(w, serial_hidden)
      serial_next_top, serial_next_logit = ML::GGUF::Qwen35CPU.forward_top1(w, serial_top, prompt.size.to_i32, serial_state)

      serial_top.should eq(chunk_top)
      serial_logit.should be_close(chunk_logit, 1e-4_f32)
      serial_next_top.should eq(chunk_next_top)
      serial_next_logit.should be_close(chunk_next_logit, 1e-4_f32)
    ensure
      if old
        ENV["QWEN35_PREFILL_CHUNK_OFF"] = old
      else
        ENV.delete("QWEN35_PREFILL_CHUNK_OFF")
      end
    end
  end

  it "keeps chunk verifier constants model-specific across target and draft models" do
    pending!("0.8B draft model not present") unless File.exists?(QWEN_08B_FWD)
    target = ML::GGUF::Qwen35Weights.from_gguf(QWEN_9B_FWD)
    draft = ML::GGUF::Qwen35Weights.from_gguf(QWEN_08B_FWD)
    prompt = [727_i32, 73111_i32, 1393_i32, 1590_i32] # "def fibonacci(n):"
    accepted = [198_i32, 262_i32, 413_i32, 307_i32]

    # Pollute shared Metal scratch/constant caches with the smaller draft model.
    draft_state = ML::GGUF::Qwen35CPU::State.new(draft.hparams, max_seq: 32)
    ML::GGUF::Qwen35CPU.prefill_tokens_top1(draft, prompt, 0, draft_state)
    accepted.each_with_index do |token_id, i|
      ML::GGUF::Qwen35CPU.forward_top1(draft, token_id, prompt.size + i, draft_state)
    end

    serial = ML::GGUF::Qwen35CPU::State.new(target.hparams, max_seq: 32)
    ML::GGUF::Qwen35CPU.prefill_tokens_top1(target, prompt, 0, serial)
    accepted.each_with_index do |token_id, i|
      ML::GGUF::Qwen35CPU.forward_top1(target, token_id, prompt.size + i, serial)
    end

    chunk = ML::GGUF::Qwen35CPU::State.new(target.hparams, max_seq: 32)
    ML::GGUF::Qwen35CPU.prefill_tokens_top1(target, prompt, 0, chunk)
    verify = chunk.fork
    ML::GGUF::Qwen35CPU.prefill_tokens_top1s(target, accepted, prompt.size, verify)
    chunk.copy_from!(verify)

    serial_next = ML::GGUF::Qwen35CPU.forward_top1(target, 606_i32, prompt.size + accepted.size, serial)
    chunk_next = ML::GGUF::Qwen35CPU.forward_top1(target, 606_i32, prompt.size + accepted.size, chunk)
    chunk_next[0].should eq(serial_next[0])
    chunk_next[1].should be_close(serial_next[1], 1e-4_f32)
  end

  it "forks decode state into independent buffers" do
    w = ML::GGUF::Qwen35Weights.from_gguf(QWEN_9B_FWD)
    hp = w.hparams

    old_top1 = ENV["QWEN35_HEAD_TOP1_FUSED"]?
    ENV["QWEN35_HEAD_TOP1_FUSED"] = "1"
    begin
      base = ML::GGUF::Qwen35CPU::State.new(hp, max_seq: 32)
      ML::GGUF::Qwen35CPU.forward_top1(w, 0, 0, base)

      fork_a = base.fork
      fork_b = base.fork
      top_a, logit_a = ML::GGUF::Qwen35CPU.forward_top1(w, 100, 1, fork_a)
      top_b, logit_b = ML::GGUF::Qwen35CPU.forward_top1(w, 100, 1, fork_b)
      top_a.should eq(top_b)
      logit_a.should be_close(logit_b, 1e-5_f32)

      restored = ML::GGUF::Qwen35CPU::State.new(hp, max_seq: 32)
      restored.copy_from!(base)
      top_r, logit_r = ML::GGUF::Qwen35CPU.forward_top1(w, 100, 1, restored)
      top_r.should eq(top_b)
      logit_r.should be_close(logit_b, 1e-5_f32)

      checked = 0
      base.layers.each_with_index do |src_layer, i|
        fork_layer = fork_b.layers[i]
        {
          {src_layer.k_cache_buf, fork_layer.k_cache_buf},
          {src_layer.v_cache_buf, fork_layer.v_cache_buf},
          {src_layer.conv_state_buf, fork_layer.conv_state_buf},
          {src_layer.ssm_state_buf, fork_layer.ssm_state_buf},
        }.each do |src_opt, fork_opt|
          next unless src = src_opt
          forked = fork_opt.not_nil!
          src.handle.should_not eq(forked.handle)
          src.size.should eq(forked.size)

          before = forked.read(1)[0]
          src.contents.as(Pointer(Float32))[0] = before + 1.0_f32
          forked.read(1)[0].should eq(before)
          checked += 1
        end
      end

      checked.should be > 0
    ensure
      if old_top1
        ENV["QWEN35_HEAD_TOP1_FUSED"] = old_top1
      else
        ENV.delete("QWEN35_HEAD_TOP1_FUSED")
      end
    end
  end
end
