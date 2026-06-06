require "./spec_helper"
require "../src/ml/gguf/gemma4_cpu"
require "../src/ml/gguf/gemma4_metal"
require "../src/ml/gguf/qwen35_metal"

GEMMA4_METAL_BUFFER_12B_Q4KM = "#{ENV["HOME"]}/.cache/lm-studio/models/lmstudio-community/gemma-4-12B-it-GGUF/gemma-4-12B-it-Q4_K_M.gguf"

def gemma4_buffer_max_abs_diff(a : Array(Float32), b : Array(Float32)) : Float32
  max = 0.0_f32
  a.each_with_index do |av, i|
    diff = (av - b[i]).abs
    max = diff if diff > max
  end
  max
end

def gemma4_expect_resident_matmul_matches(label : String,
                                          qw : ML::GGUF::QuantWeight,
                                          x : Array(Float32)) : Nil
  expected = ML::GGUF::Qwen35Metal.matmul(qw, x, 1).not_nil!
  x_buf = ML::MetalBuffer.from_array(x)
  out_buf = ML::MetalBuffer.new(qw.out_dim.to_i64 * sizeof(Float32))

  ML::GGUF::Qwen35Metal.matmul_to_buffer(qw, x_buf, out_buf, 1).should be_true
  actual = out_buf.read(qw.out_dim)
  diff = gemma4_buffer_max_abs_diff(expected, actual)
  puts "  [#{label}] max|d|=#{diff}"
  diff.should be <= 1.0e-6_f32
end

describe "Gemma4 resident Metal matmul buffers" do
  pending!("Gemma4 12B GGUF not found") unless File.exists?(GEMMA4_METAL_BUFFER_12B_Q4KM)
  pending!("Metal not available") unless ML::GGUF::Qwen35Metal.available?

  it "keeps the graph-backed FFN elementwise tail aligned with existing helpers" do
    gate = Array(Float32).new(128) { |i| Math.sin(i.to_f32 * 0.031_f32).to_f32 }
    up = Array(Float32).new(128) { |i| Math.cos(i.to_f32 * 0.017_f32).to_f32 }
    residual = Array(Float32).new(128) { |i| (i % 7).to_f32 * 0.125_f32 - 0.4_f32 }
    scale = 0.75_f32

    combined = ML::GGUF::Gemma4Metal.gelu_mul(gate, up).not_nil!
    expected = ML::GGUF::Gemma4Metal.add_scaled_vec(residual, combined, scale).not_nil!
    actual = ML::GGUF::Gemma4Metal.gelu_mul_add_scaled_graph(gate, up, residual, scale).not_nil!

    diff = gemma4_buffer_max_abs_diff(expected, actual)
    diff.should be <= 1.0e-6_f32
  end

  it "embeds Q6 token-id batches like the existing per-token Metal path" do
    w = ML::GGUF::Gemma4Weights.from_gguf(GEMMA4_METAL_BUFFER_12B_Q4KM)
    token_ids = [42_i32, 43_i32, 44_i32, 45_i32]
    hidden = w.hparams.n_embd
    scale = Math.sqrt(hidden.to_f64).to_f32
    expected = [] of Float32

    token_ids.each do |token_id|
      row = ML::GGUF::Qwen35Metal.embedding_q6k_from_token_id(w.token_embd, token_id).not_nil!
      row.size.times { |i| row[i] *= scale }
      expected.concat(row)
    end

    out_buf = ML::MetalBuffer.new(token_ids.size.to_i64 * hidden * sizeof(Float32))
    ML::GGUF::Qwen35Metal.embedding_q6k_rows_scaled_to_buffer(w.token_embd, token_ids, out_buf, scale)
    actual = out_buf.read(token_ids.size * hidden)
    diff = gemma4_buffer_max_abs_diff(expected, actual)
    puts "  [gemma4_batch_q6_embedding] max|d|=#{diff}"
    diff.should be <= 1.0e-6_f32
  end

  it "projects Q4 and Q6 Gemma4 weights from resident input buffers like the array path" do
    w = ML::GGUF::Gemma4Weights.from_gguf(GEMMA4_METAL_BUFFER_12B_Q4KM)
    x = ML::GGUF::Gemma4CPU.embedding_lookup(w.token_embd, 42)
    lw = w.layers[0]
    x_norm = ML::GGUF::Gemma4CPU.rms_norm(x, lw.attn_norm, w.hparams.rms_eps)

    ML::GGUF::Qwen35Metal.matmul_to_buffer(lw.attn_q_qw, ML::MetalBuffer.from_array(x_norm), ML::MetalBuffer.new(lw.attn_q_qw.out_dim.to_i64 * sizeof(Float32)), 0).should be_false
    gemma4_expect_resident_matmul_matches("gemma4_resident_q4_attn_q", lw.attn_q_qw, x_norm)

    ctx = Array(Float32).new(lw.ffn_down_qw.in_dim) { |i| Math.sin(i.to_f32 * 0.013_f32).to_f32 * 0.25_f32 }
    gemma4_expect_resident_matmul_matches("gemma4_resident_q6_ffn_down", lw.ffn_down_qw, ctx)
  end

  it "runs the Gemma4 layer tail through resident intermediate buffers like the array path" do
    w = ML::GGUF::Gemma4Weights.from_gguf(GEMMA4_METAL_BUFFER_12B_Q4KM)
    lw = w.layers[0]
    x = ML::GGUF::Gemma4CPU.embedding_lookup(w.token_embd, 42)
    scale = Math.sqrt(w.hparams.n_embd.to_f64).to_f32
    x.size.times { |i| x[i] *= scale }
    attn_projected = Array(Float32).new(w.hparams.n_embd) { |i| Math.sin(i.to_f32 * 0.017_f32).to_f32 * 0.125_f32 }

    expected = ML::GGUF::Gemma4Metal.layer_tail(x, attn_projected, lw, w.hparams).not_nil!
    actual = ML::GGUF::Gemma4Metal.layer_tail_resident_buffers(x, attn_projected, lw, w.hparams).not_nil!
    diff = gemma4_buffer_max_abs_diff(expected, actual)
    puts "  [gemma4_resident_layer_tail] max|d|=#{diff}"
    diff.should be <= 1.0e-5_f32
  end

  it "runs the Gemma4 layer tail over prompt rows like serial resident tails" do
    w = ML::GGUF::Gemma4Weights.from_gguf(GEMMA4_METAL_BUFFER_12B_Q4KM)
    lw = w.layers[0]
    batch = 4
    hidden = w.hparams.n_embd
    scale = Math.sqrt(hidden.to_f64).to_f32
    x_rows = [] of Float32
    attn_rows = [] of Float32
    expected = [] of Float32

    batch.times do |r|
      x = ML::GGUF::Gemma4CPU.embedding_lookup(w.token_embd, 42 + r)
      x.size.times { |i| x[i] *= scale }
      attn_projected = Array(Float32).new(hidden) { |i| Math.sin((i + r * 17).to_f32 * 0.017_f32).to_f32 * 0.125_f32 }
      x_rows.concat(x)
      attn_rows.concat(attn_projected)
      expected.concat(ML::GGUF::Gemma4Metal.layer_tail_resident_buffers(x, attn_projected, lw, w.hparams).not_nil!)
    end

    actual = ML::GGUF::Gemma4Metal.layer_tail_batch(x_rows, attn_rows, lw, w.hparams, batch).not_nil!
    diff = gemma4_buffer_max_abs_diff(expected, actual)
    puts "  [gemma4_batch_layer_tail] max|d|=#{diff}"
    diff.should be <= 1.0e-5_f32
  end

  it "runs Gemma4 resident layer rows like serial resident layer steps" do
    w = ML::GGUF::Gemma4Weights.from_gguf(GEMMA4_METAL_BUFFER_12B_Q4KM)
    batch = 4
    hidden = w.hparams.n_embd
    scale = Math.sqrt(hidden.to_f64).to_f32
    layers = [0]
    if full_il = w.hparams.full_attention_layers.first?
      layers << full_il unless full_il == 0
    end

    layers.each do |il|
      x_rows = [] of Float32
      batch.times do |r|
        x = ML::GGUF::Gemma4CPU.embedding_lookup(w.token_embd, 42 + r)
        x.size.times { |i| x[i] *= scale }
        x_rows.concat(x)
      end

      serial_state = ML::GGUF::Gemma4Metal::ResidentState.new(w.hparams, 16)
      expected = [] of Float32
      batch.times do |r|
        row = x_rows[(r * hidden)...((r + 1) * hidden)].to_a
        expected.concat(ML::GGUF::Gemma4Metal.forward_layer_resident_cache(w, il, row, r, serial_state).not_nil!)
      end

      row_state = ML::GGUF::Gemma4Metal::ResidentState.new(w.hparams, 16)
      actual = ML::GGUF::Gemma4Metal.forward_layer_resident_cache_rows(w, il, x_rows, 0, batch, row_state).not_nil!
      diff = gemma4_buffer_max_abs_diff(expected, actual)
      puts "  [gemma4_resident_layer_rows_l#{il}] max|d|=#{diff}"
      diff.should be <= 1.0e-5_f32
    end
  end

  it "prefills Gemma4 prompt chunks through resident layer rows like serial resident prefill" do
    w = ML::GGUF::Gemma4Weights.from_gguf(GEMMA4_METAL_BUFFER_12B_Q4KM)
    prompt = [42_i32, 43_i32, 44_i32, 45_i32, 46_i32, 47_i32, 48_i32, 49_i32]
    stop_layer = 6

    serial_state = ML::GGUF::Gemma4Metal::ResidentState.new(w.hparams, 16)
    serial_last = [] of Float32
    prompt.each_with_index do |token_id, pos|
      serial_last = ML::GGUF::Gemma4Metal.forward_hidden_resident_cache(w, token_id, pos, serial_state, stop_layer).not_nil!
    end

    row_state = ML::GGUF::Gemma4Metal::ResidentState.new(w.hparams, 16)
    row_last = ML::GGUF::Gemma4Metal.prefill_tokens_last_hidden_resident_rows(w, prompt, 0, row_state, chunk_size: 4, stop_layer: stop_layer).not_nil!
    last_diff = gemma4_buffer_max_abs_diff(serial_last, row_last)
    puts "  [gemma4_resident_prefill_rows_stop6_last] max|d|=#{last_diff}"
    last_diff.should be <= 1.0e-5_f32

    serial_next = ML::GGUF::Gemma4Metal.forward_hidden_resident_cache(w, 50, prompt.size, serial_state, stop_layer).not_nil!
    row_next = ML::GGUF::Gemma4Metal.forward_hidden_resident_cache(w, 50, prompt.size, row_state, stop_layer).not_nil!
    next_diff = gemma4_buffer_max_abs_diff(serial_next, row_next)
    puts "  [gemma4_resident_prefill_rows_stop6_next] max|d|=#{next_diff}"
    next_diff.should be <= 1.0e-5_f32
  end

  it "keeps the stop-layer resident-cache hidden path aligned after resident tail promotion" do
    w = ML::GGUF::Gemma4Weights.from_gguf(GEMMA4_METAL_BUFFER_12B_Q4KM)
    host_state = ML::GGUF::Gemma4Metal::State.new(w.hparams, 8)
    resident_state = ML::GGUF::Gemma4Metal::ResidentState.new(w.hparams, 8)
    host = [] of Float32
    resident = [] of Float32

    [42, 43].each_with_index do |token_id, pos|
      host = ML::GGUF::Gemma4Metal.forward_hidden(w, token_id, pos, host_state, 2).not_nil!
      resident = ML::GGUF::Gemma4Metal.forward_hidden_resident_cache(w, token_id, pos, resident_state, 2).not_nil!
    end

    diff = gemma4_buffer_max_abs_diff(host, resident)
    puts "  [gemma4_resident_stop2_hidden] max|d|=#{diff}"
    diff.should be <= 1.0e-5_f32
  end

  it "keeps the one-command-buffer decode wave aligned with resident decode" do
    w = ML::GGUF::Gemma4Weights.from_gguf(GEMMA4_METAL_BUFFER_12B_Q4KM)
    resident_state = ML::GGUF::Gemma4Metal::ResidentState.new(w.hparams, 8)
    wave_state = ML::GGUF::Gemma4Metal::ResidentState.new(w.hparams, 8)
    resident = [] of Float32
    wave = [] of Float32

    [42, 43].each_with_index do |token_id, pos|
      resident = ML::GGUF::Gemma4Metal.forward_hidden_resident_cache(w, token_id, pos, resident_state, 6).not_nil!
      wave = ML::GGUF::Gemma4Metal.forward_hidden_resident_cache_wave(w, token_id, pos, wave_state, 6).not_nil!
    end

    diff = gemma4_buffer_max_abs_diff(resident, wave)
    puts "  [gemma4_decode_wave_stop6_hidden] max|d|=#{diff}"
    diff.should be <= 1.0e-5_f32
  end

  it "keeps resident top2 decode aligned with hidden-wave top2 projection" do
    w = ML::GGUF::Gemma4Weights.from_gguf(GEMMA4_METAL_BUFFER_12B_Q4KM)
    hidden_state = ML::GGUF::Gemma4Metal::ResidentState.new(w.hparams, 8)
    top2_state = ML::GGUF::Gemma4Metal::ResidentState.new(w.hparams, 8)
    token_id = 42
    stop_layer = 6

    hidden = ML::GGUF::Gemma4Metal.forward_hidden_resident_cache_wave(w, token_id, 0, hidden_state, stop_layer).not_nil!
    normed = ML::GGUF::Gemma4CPU.rms_norm(hidden, w.output_norm, w.hparams.rms_eps)
    expected = ML::GGUF::Qwen35Metal.project_top2_no_norm(w.token_embd, normed).not_nil!
    actual = ML::GGUF::Gemma4Metal.forward_top2_resident_cache_wave(w, token_id, 0, top2_state, stop_layer).not_nil!

    actual[0].to_i.should eq(expected[0].to_i)
    actual[2].to_i.should eq(expected[2].to_i)
    (actual[1] - expected[1]).abs.should be <= 1.0e-5_f32
    (actual[3] - expected[3]).abs.should be <= 1.0e-5_f32
  end

  it "keeps the graph-backed no-norm top1 head aligned with the existing head path" do
    w = ML::GGUF::Gemma4Weights.from_gguf(GEMMA4_METAL_BUFFER_12B_Q4KM)
    hidden_state = ML::GGUF::Gemma4Metal::ResidentState.new(w.hparams, 8)
    token_id = 42
    stop_layer = 6

    hidden = ML::GGUF::Gemma4Metal.forward_hidden_resident_cache_wave(w, token_id, 0, hidden_state, stop_layer).not_nil!
    normed = ML::GGUF::Gemma4CPU.rms_norm(hidden, w.output_norm, w.hparams.rms_eps)
    expected = ML::GGUF::Qwen35Metal.project_top1_no_norm(w.token_embd, normed).not_nil!
    actual = ML::GGUF::Qwen35Metal.project_top1_no_norm_graph(w.token_embd, normed).not_nil!

    actual[0].to_i.should eq(expected[0].to_i)
    (actual[1] - expected[1]).abs.should be <= 1.0e-5_f32
  end

  it "keeps resident allowed-token top1 aligned with full hidden-wave top2 projection" do
    w = ML::GGUF::Gemma4Weights.from_gguf(GEMMA4_METAL_BUFFER_12B_Q4KM)
    hidden_state = ML::GGUF::Gemma4Metal::ResidentState.new(w.hparams, 8)
    allowed_state = ML::GGUF::Gemma4Metal::ResidentState.new(w.hparams, 8)
    token_id = 42
    stop_layer = 6

    hidden = ML::GGUF::Gemma4Metal.forward_hidden_resident_cache_wave(w, token_id, 0, hidden_state, stop_layer).not_nil!
    normed = ML::GGUF::Gemma4CPU.rms_norm(hidden, w.output_norm, w.hparams.rms_eps)
    top2 = ML::GGUF::Qwen35Metal.project_top2_no_norm(w.token_embd, normed).not_nil!
    best_id = top2[0].to_i
    second_id = top2[2].to_i
    allowed_ids = [1_i32, second_id.to_i32, 2_i32, best_id.to_i32, 3_i32]

    actual = ML::GGUF::Gemma4Metal.forward_top1_allowed_resident_cache_wave(w, token_id, 0, allowed_ids, allowed_state, stop_layer).not_nil!
    actual[0].to_i.should eq(best_id)
    (actual[1] - top2[1]).abs.should be <= 1.0e-5_f32

    hidden_state2 = ML::GGUF::Gemma4Metal::ResidentState.new(w.hparams, 8)
    allowed_state2 = ML::GGUF::Gemma4Metal::ResidentState.new(w.hparams, 8)
    hidden2 = ML::GGUF::Gemma4Metal.forward_hidden_resident_cache_wave(w, token_id, 0, hidden_state2, stop_layer).not_nil!
    normed2 = ML::GGUF::Gemma4CPU.rms_norm(hidden2, w.output_norm, w.hparams.rms_eps)
    top2_b = ML::GGUF::Qwen35Metal.project_top2_no_norm(w.token_embd, normed2).not_nil!
    allowed_without_top1 = [1_i32, top2_b[2].to_i32, 2_i32, 3_i32]

    actual2 = ML::GGUF::Gemma4Metal.forward_top1_allowed_resident_cache_wave(w, token_id, 0, allowed_without_top1, allowed_state2, stop_layer).not_nil!
    actual2[0].to_i.should eq(top2_b[2].to_i)
    (actual2[1] - top2_b[3]).abs.should be <= 1.0e-5_f32
  end

  it "keeps full-attention K-as-V semantics aligned in the resident hidden path" do
    w = ML::GGUF::Gemma4Weights.from_gguf(GEMMA4_METAL_BUFFER_12B_Q4KM)
    host_state = ML::GGUF::Gemma4Metal::State.new(w.hparams, 8)
    resident_state = ML::GGUF::Gemma4Metal::ResidentState.new(w.hparams, 8)
    host = [] of Float32
    resident = [] of Float32

    [42, 43].each_with_index do |token_id, pos|
      host = ML::GGUF::Gemma4Metal.forward_hidden(w, token_id, pos, host_state, 6).not_nil!
      resident = ML::GGUF::Gemma4Metal.forward_hidden_resident_cache(w, token_id, pos, resident_state, 6).not_nil!
    end

    diff = gemma4_buffer_max_abs_diff(host, resident)
    puts "  [gemma4_resident_stop6_hidden] max|d|=#{diff}"
    diff.should be <= 1.0e-5_f32
  end
end
