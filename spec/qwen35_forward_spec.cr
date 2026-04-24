require "./spec_helper"
require "../src/ml/gguf/qwen35_cpu"
require "../src/ml/gguf/qwen35_weights"

QWEN_9B_FWD = "#{ENV["HOME"]}/.cache/lm-studio/models/lmstudio-community/Qwen3.5-9B-GGUF/Qwen3.5-9B-Q4_K_M.gguf"

describe ML::GGUF::Qwen35CPU, "full decoder forward" do
  pending!("9B model not present") unless File.exists?(QWEN_9B_FWD)

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
