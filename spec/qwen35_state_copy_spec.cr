require "./spec_helper"
require "../src/ml/gguf/reader"
require "../src/ml/gguf/qwen35_chat"
require "../src/ml/gguf/qwen35_cpu"
require "../src/ml/gguf/qwen35_tokenizer"

QWEN_9B_STATE_COPY = "#{ENV["HOME"]}/.cache/lm-studio/models/lmstudio-community/Qwen3.5-9B-GGUF/Qwen3.5-9B-Q4_K_M.gguf"

private def fill_state_bytes!(state : ML::GGUF::Qwen35CPU::State, value : UInt8) : Nil
  ML::Metal::Dispatch.execute_blit do |enc|
    state.layers.each do |layer|
      if buf = layer.k_cache_buf
        enc.fill_buffer(buf, value, 0, buf.size.to_i32)
      end
      if buf = layer.v_cache_buf
        enc.fill_buffer(buf, value, 0, buf.size.to_i32)
      end
      if buf = layer.conv_state_buf
        enc.fill_buffer(buf, value, 0, buf.size.to_i32)
      end
      if buf = layer.ssm_state_buf
        enc.fill_buffer(buf, value, 0, buf.size.to_i32)
      end
    end
  end
end

private def release_state_buffers!(state : ML::GGUF::Qwen35CPU::State) : Nil
  ML::Metal::Device.synchronize
  state.layers.each do |layer|
    layer.k_cache_buf.try(&.release)
    layer.v_cache_buf.try(&.release)
    layer.conv_state_buf.try(&.release)
    layer.ssm_state_buf.try(&.release)
  end
end

describe ML::GGUF::Qwen35CPU do
  it "prepares a checkpoint with recurrent Metal buffers only" do
    pending!("9B model not present") unless File.exists?(QWEN_9B_STATE_COPY)
    pending!("Metal not available") unless ML::GGUF::Qwen35Metal.available?

    gguf = ML::GGUF::GGUFFile.new(QWEN_9B_STATE_COPY)
    hp = ML::GGUF::Qwen35Hparams.new(gguf)
    checkpoint = ML::GGUF::Qwen35CPU::State.new(hp, max_seq: 32)
    ML::GGUF::Qwen35CPU.prepare_recurrent_state_metal!(checkpoint, hp)

    hp.full_attention_layers.each do |layer_index|
      checkpoint.layers[layer_index].k_cache_buf.should be_nil
      checkpoint.layers[layer_index].v_cache_buf.should be_nil
    end
    hp.recurrent_layers.each do |layer_index|
      checkpoint.layers[layer_index].conv_state_buf.should_not be_nil
      checkpoint.layers[layer_index].ssm_state_buf.should_not be_nil
    end
  ensure
    release_state_buffers!(checkpoint) if checkpoint
    gguf.try(&.close)
  end

  it "copies only live Metal KV rows plus recurrent state for branch states" do
    pending!("9B model not present") unless File.exists?(QWEN_9B_STATE_COPY)
    pending!("Metal not available") unless ML::GGUF::Qwen35Metal.available?

    gguf = ML::GGUF::GGUFFile.new(QWEN_9B_STATE_COPY)
    hp = ML::GGUF::Qwen35Hparams.new(gguf)
    max_seq = 16
    pos = 4

    src = ML::GGUF::Qwen35CPU::State.new(hp, max_seq: max_seq)
    dst = ML::GGUF::Qwen35CPU::State.new(hp, max_seq: max_seq)
    ML::GGUF::Qwen35CPU.prepare_state_metal!(src, hp)
    ML::GGUF::Qwen35CPU.prepare_state_metal!(dst, hp)
    src.layers.each { |layer| layer.position = pos }

    fill_state_bytes!(src, 0x5a_u8)
    fill_state_bytes!(dst, 0x00_u8)

    ML::GGUF::Qwen35CPU.copy_state_metal_used!(dst, src, hp, used_tokens: pos)

    rec_layer = hp.recurrent_layers.first
    dst.layers[rec_layer].conv_state_buf.not_nil!.contents.as(Pointer(UInt8)).value.should eq(0x5a_u8)
    dst.layers[rec_layer].ssm_state_buf.not_nil!.contents.as(Pointer(UInt8)).value.should eq(0x5a_u8)

    full_layer = hp.full_attention_layers.first
    live_kv_bytes = (pos * hp.head_dim * hp.n_head_kv * sizeof(Float32)).to_i
    k_ptr = dst.layers[full_layer].k_cache_buf.not_nil!.contents.as(Pointer(UInt8))
    v_ptr = dst.layers[full_layer].v_cache_buf.not_nil!.contents.as(Pointer(UInt8))

    k_ptr[0].should eq(0x5a_u8)
    v_ptr[0].should eq(0x5a_u8)
    k_ptr[live_kv_bytes - 1].should eq(0x5a_u8)
    v_ptr[live_kv_bytes - 1].should eq(0x5a_u8)

    # The branch copy intentionally leaves unused KV capacity untouched.
    k_ptr[live_kv_bytes].should eq(0x00_u8)
    v_ptr[live_kv_bytes].should eq(0x00_u8)
  end

  it "matches full state copy for subsequent exact decode" do
    pending!("9B model not present") unless File.exists?(QWEN_9B_STATE_COPY)
    pending!("Metal not available") unless ML::GGUF::Qwen35Metal.available?

    weights = ML::GGUF::Qwen35Weights.from_gguf(QWEN_9B_STATE_COPY)
    hp = weights.hparams
    prefix = [1_i32, 2_i32, 3_i32, 4_i32, 5_i32, 6_i32, 7_i32, 8_i32]

    src = ML::GGUF::Qwen35CPU::State.new(hp, max_seq: 32)
    live = ML::GGUF::Qwen35CPU::State.new(hp, max_seq: 32)
    ML::GGUF::Qwen35CPU.prepare_state_metal!(src, hp)
    ML::GGUF::Qwen35CPU.prepare_state_metal!(live, hp)

    ML::GGUF::Qwen35CPU.prefill_tokens(weights, prefix, 0, src)
    full = src.fork
    ML::GGUF::Qwen35CPU.copy_state_metal_used!(live, src, hp, used_tokens: prefix.size)

    next_token = 42_i32
    full_top1 = ML::GGUF::Qwen35CPU.forward_top1(weights, next_token, prefix.size, full)
    live_top1 = ML::GGUF::Qwen35CPU.forward_top1(weights, next_token, prefix.size, live)
    live_top1[0].should eq(full_top1[0])
    live_top1[1].should be_close(full_top1[1], 1.0e-4)

    full_next = ML::GGUF::Qwen35CPU.forward_top1(weights, full_top1[0], prefix.size + 1, full)
    live_next = ML::GGUF::Qwen35CPU.forward_top1(weights, live_top1[0], prefix.size + 1, live)
    live_next[0].should eq(full_next[0])
    live_next[1].should be_close(full_next[1], 1.0e-4)
  end

  it "rewinds recurrent state and replays only the exact completed-message suffix" do
    pending!("9B model not present") unless File.exists?(QWEN_9B_STATE_COPY)
    pending!("Metal not available") unless ML::GGUF::Qwen35Metal.available?

    gguf = ML::GGUF::GGUFFile.new(QWEN_9B_STATE_COPY)
    tokenizer = ML::GGUF::Qwen35Tokenizer.from_gguf(gguf, QWEN_9B_STATE_COPY)
    messages = [
      ML::GGUF::Qwen35Chat::Message.new("system", "Answer tersely."),
      ML::GGUF::Qwen35Chat::Message.new("user", "Reply only with checkpoint-1."),
    ]
    prompt_text = ML::GGUF::Qwen35Chat.render(messages, enable_thinking: false)
    boundary_text = ML::GGUF::Qwen35Chat.render(
      messages + [ML::GGUF::Qwen35Chat::Message.new("assistant", "checkpoint-1")],
      add_generation_prompt: false,
      enable_thinking: false,
    )
    prompt_ids = tokenizer.encode(prompt_text, add_bos_override: false)
    boundary_ids = tokenizer.encode(boundary_text, add_bos_override: false)
    prefix_len = Math.max(1, prompt_ids.size - 8).to_i32
    prompt_ids[0, prefix_len].should eq(boundary_ids[0, prefix_len])

    weights = ML::GGUF::Qwen35Weights.from_gguf(QWEN_9B_STATE_COPY)
    hp = weights.hparams
    live = ML::GGUF::Qwen35CPU::State.new(hp, max_seq: 64)
    checkpoint = ML::GGUF::Qwen35CPU::State.new(hp, max_seq: 64)
    baseline = ML::GGUF::Qwen35CPU::State.new(hp, max_seq: 64)
    ML::GGUF::Qwen35CPU.prepare_state_metal!(live, hp)
    ML::GGUF::Qwen35CPU.prepare_state_metal!(baseline, hp)

    generated, _generated_logit = ML::GGUF::Qwen35CPU.prefill_tokens_top1_recurrent_checkpoint(
      weights,
      prompt_ids,
      0,
      live,
      prefix_len - 1,
      checkpoint,
    )
    prompt_expected = ML::GGUF::Qwen35CPU.prefill_tokens_top1(
      weights,
      prompt_ids,
      0,
      baseline,
    )
    generated.should eq(prompt_expected[0])
    release_state_buffers!(baseline)
    baseline = ML::GGUF::Qwen35CPU::State.new(hp, max_seq: 64)
    ML::GGUF::Qwen35CPU.prepare_state_metal!(baseline, hp)
    ML::GGUF::Qwen35CPU.forward_top1(weights, generated, prompt_ids.size, live)

    ML::GGUF::Qwen35CPU.swap_recurrent_state_metal_buffers!(live, checkpoint, hp)
    live.layers.each { |layer| layer.position = prefix_len }
    replay_ids = boundary_ids[prefix_len, boundary_ids.size - prefix_len]
    replayed = ML::GGUF::Qwen35CPU.prefill_tokens_top1_sequential(
      weights,
      replay_ids,
      prefix_len,
      live,
    )
    ML::GGUF::Qwen35CPU.clear_kv_tail_metal!(live, hp, boundary_ids.size.to_i32)
    kv_row_bytes = hp.head_dim * hp.n_head_kv * sizeof(Float32)
    tail_offset = boundary_ids.size * kv_row_bytes
    hp.full_attention_layers.each do |layer_index|
      [live.layers[layer_index].k_cache_buf.not_nil!, live.layers[layer_index].v_cache_buf.not_nil!].each do |buffer|
        tail_size = buffer.size.to_i - tail_offset
        tail = Slice.new(buffer.contents.as(Pointer(UInt8)) + tail_offset, tail_size)
        tail.all? { |byte| byte == 0_u8 }.should be_true
      end
    end
    expected = ML::GGUF::Qwen35CPU.prefill_tokens_top1_sequential(
      weights,
      boundary_ids,
      0,
      baseline,
    )

    replayed[0].should eq(expected[0])
    replayed[1].should be_close(expected[1], 2.0e-2_f32)
    replayed_next = ML::GGUF::Qwen35CPU.forward_top1(weights, replayed[0], boundary_ids.size, live)
    expected_next = ML::GGUF::Qwen35CPU.forward_top1(weights, expected[0], boundary_ids.size, baseline)
    replayed_next[0].should eq(expected_next[0])
    replayed_next[1].should be_close(expected_next[1], 2.0e-2_f32)
  ensure
    release_state_buffers!(checkpoint) if checkpoint
    release_state_buffers!(live) if live
    release_state_buffers!(baseline) if baseline
    weights.try(&.close)
    gguf.try(&.close)
  end
end
