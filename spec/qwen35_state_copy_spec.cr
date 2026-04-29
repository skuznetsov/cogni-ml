require "./spec_helper"
require "../src/ml/gguf/reader"
require "../src/ml/gguf/qwen35_cpu"

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

describe ML::GGUF::Qwen35CPU do
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

    ML::GGUF::Qwen35CPU.copy_state_metal_used!(dst, src, hp)

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
end
