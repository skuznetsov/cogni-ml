require "./spec_helper"
require "../src/ml/gguf/inkling2_weights"
require "../src/ml/gguf/inkling2_engine"

private alias Inkling2TestValue = String | Int32 | Float32 | Bool | Array(Int32) | Array(Bool)
private alias Inkling2TestTensor = NamedTuple(name: String, dims: Array(Int64), type_id: UInt32, offset: UInt64, bytes: Int32)

private def inkling2_write_string(io : IO, value : String) : Nil
  io.write_bytes(value.bytesize.to_u64, IO::ByteFormat::LittleEndian)
  io.write(value.to_slice)
end

private def inkling2_write_metadata_value(io : IO, value : Inkling2TestValue) : Nil
  case value
  when String
    io.write_bytes(8_u32, IO::ByteFormat::LittleEndian)
    inkling2_write_string(io, value)
  when Int32
    io.write_bytes(5_u32, IO::ByteFormat::LittleEndian)
    io.write_bytes(value, IO::ByteFormat::LittleEndian)
  when Float32
    io.write_bytes(6_u32, IO::ByteFormat::LittleEndian)
    io.write_bytes(value, IO::ByteFormat::LittleEndian)
  when Bool
    io.write_bytes(7_u32, IO::ByteFormat::LittleEndian)
    io.write_byte(value ? 1_u8 : 0_u8)
  when Array(Int32)
    io.write_bytes(9_u32, IO::ByteFormat::LittleEndian)
    io.write_bytes(5_u32, IO::ByteFormat::LittleEndian)
    io.write_bytes(value.size.to_u64, IO::ByteFormat::LittleEndian)
    value.each { |item| io.write_bytes(item, IO::ByteFormat::LittleEndian) }
  when Array(Bool)
    io.write_bytes(9_u32, IO::ByteFormat::LittleEndian)
    io.write_bytes(7_u32, IO::ByteFormat::LittleEndian)
    io.write_bytes(value.size.to_u64, IO::ByteFormat::LittleEndian)
    value.each { |item| io.write_byte(item ? 1_u8 : 0_u8) }
  end
end

private def inkling2_pad_to(io : IO, alignment : Int32) : Nil
  padding = ((alignment - io.pos % alignment) % alignment).to_i
  io.write(Bytes.new(padding, 0_u8)) if padding > 0
end

private def inkling2_test_metadata(
  *,
  n_layer : Int32 = 4,
  n_embd : Int32 = 4,
  n_head : Int32 = 2,
  n_head_kv : Int32 = 1,
  head_dim : Int32 = 2,
  rel_dim : Int32 = 2,
  rel_extent : Int32 = 3,
  sliding_window : Int32 = 2,
  conv_size : Int32 = 2,
  n_expert : Int32 = 2,
  n_expert_used : Int32 = 1,
  expert_dim : Int32 = 2,
  padded_vocab_size : Int32 = 8,
  vocab_size : Int32 = 7,
  res_block_size : Int32 = 1,
  core_start : Int32 = 2,
  core_end : Int32 = 3,
  passes : Int32 = 3,
  varying : Bool = false,
  sliding_pattern : Array(Bool) = [true, false, true, false],
) : Array({String, Inkling2TestValue})
  prefix = "inkling2_loop"
  [
    {"general.architecture", "inkling2_loop".as(Inkling2TestValue)},
    {"#{prefix}.block_count", n_layer.as(Inkling2TestValue)},
    {"#{prefix}.embedding_length", n_embd.as(Inkling2TestValue)},
    {"#{prefix}.context_length", 1024_i32.as(Inkling2TestValue)},
    {"#{prefix}.attention.head_count", Array(Int32).new(n_layer, n_head).as(Inkling2TestValue)},
    {"#{prefix}.attention.head_count_kv", Array(Int32).new(n_layer, n_head_kv).as(Inkling2TestValue)},
    {"#{prefix}.attention.key_length", head_dim.as(Inkling2TestValue)},
    {"#{prefix}.attention.value_length", head_dim.as(Inkling2TestValue)},
    {"#{prefix}.attention.relative_length", rel_dim.as(Inkling2TestValue)},
    {"#{prefix}.attention.relative_extent", rel_extent.as(Inkling2TestValue)},
    {"#{prefix}.attention.sliding_window", sliding_window.as(Inkling2TestValue)},
    {"#{prefix}.attention.sliding_window_pattern", sliding_pattern.as(Inkling2TestValue)},
    {"#{prefix}.attention.log_scale_floor", 64_i32.as(Inkling2TestValue)},
    {"#{prefix}.attention.log_scale_alpha", 0.1_f32.as(Inkling2TestValue)},
    {"#{prefix}.attention.qk_clip", 22.0_f32.as(Inkling2TestValue)},
    {"#{prefix}.attention.relative_clip", 8.0_f32.as(Inkling2TestValue)},
    {"#{prefix}.attention.layer_norm_rms_epsilon", 1.0e-6_f32.as(Inkling2TestValue)},
    {"#{prefix}.shortconv.l_cache", conv_size.as(Inkling2TestValue)},
    {"#{prefix}.expert_count", n_expert.as(Inkling2TestValue)},
    {"#{prefix}.expert_used_count", n_expert_used.as(Inkling2TestValue)},
    {"#{prefix}.expert_feed_forward_length", expert_dim.as(Inkling2TestValue)},
    {"#{prefix}.expert_weights_scale", 8.0_f32.as(Inkling2TestValue)},
    {"#{prefix}.activation.situ_beta", 4.0_f32.as(Inkling2TestValue)},
    {"#{prefix}.activation.situ_linear_beta", 25.0_f32.as(Inkling2TestValue)},
    {"#{prefix}.attn_res.block_size", res_block_size.as(Inkling2TestValue)},
    {"#{prefix}.vocab_size", padded_vocab_size.as(Inkling2TestValue)},
    {"#{prefix}.vocab_size_unpadded", vocab_size.as(Inkling2TestValue)},
    {"#{prefix}.loop.start", core_start.as(Inkling2TestValue)},
    {"#{prefix}.loop.end", core_end.as(Inkling2TestValue)},
    {"#{prefix}.loop.count", passes.as(Inkling2TestValue)},
    {"#{prefix}.loop.gate_rank", 2_i32.as(Inkling2TestValue)},
    {"#{prefix}.loop.varying", varying.as(Inkling2TestValue)},
    {"#{prefix}.final_logit_softcapping", 30.0_f32.as(Inkling2TestValue)},
  ]
end

private def inkling2_test_tensors(
  n_layer : Int32,
  n_embd : Int32,
  n_head : Int32,
  n_head_kv : Int32,
  head_dim : Int32,
  rel_dim : Int32,
  rel_extent : Int32,
  sliding_window : Int32,
  conv_size : Int32,
  n_expert : Int32,
  expert_dim : Int32,
  vocab_size : Int32,
  padded_vocab_size : Int32,
  core_start : Int32,
  core_end : Int32,
  sliding_pattern : Array(Bool),
  quantized_experts : Bool,
) : Array(Inkling2TestTensor)
  tensors = [] of Inkling2TestTensor
  add = ->(name : String, dims : Array(Int64)) do
    elements = dims.product(1_i64)
    type = quantized_experts && name.ends_with?("_exps.weight") ? ML::GGUF::TensorType::Q8_0 : ML::GGUF::TensorType::F32
    bytes = (elements // type.block_elements * type.block_bytes).to_i32
    tensors << {name: name, dims: dims, type_id: type.value, offset: 0_u64, bytes: bytes}
  end
  dim = ->(values : Array(Int32)) { values.map(&.to_i64) }

  add.call("token_embd.weight", dim.call([n_embd, padded_vocab_size]))
  add.call("output.weight", dim.call([n_embd, padded_vocab_size]))
  add.call("output_norm.weight", dim.call([n_embd]))
  add.call("output_res_query.weight", dim.call([n_embd]))
  add.call("token_embd_norm.weight", dim.call([n_embd]))

  q_dim = head_dim * n_head
  kv_dim = head_dim * n_head_kv
  n_layer.times do |il|
    p = "blk.#{il}."
    extent = sliding_pattern[il] ? sliding_window : rel_extent
    add.call(p + "attn_q.weight", dim.call([n_embd, q_dim]))
    add.call(p + "attn_k.weight", dim.call([n_embd, kv_dim]))
    add.call(p + "attn_v.weight", dim.call([n_embd, kv_dim]))
    add.call(p + "attn_output.weight", dim.call([q_dim, n_embd]))
    add.call(p + "attn_gate.weight", dim.call([n_embd, q_dim]))
    add.call(p + "attn_q_clip.weight", dim.call([head_dim, n_head]))
    add.call(p + "attn_sink_proj.weight", dim.call([n_embd, n_head]))
    add.call(p + "attn_rel.weight", dim.call([n_embd, rel_dim * n_head]))
    add.call(p + "attn_rel_bias.weight", dim.call([rel_dim, extent]))
    add.call(p + "attn_rel_clip.weight", dim.call([rel_dim, n_head]))
    add.call(p + "attn_k_conv.weight", dim.call([conv_size, kv_dim]))
    add.call(p + "attn_v_conv.weight", dim.call([conv_size, kv_dim]))
    add.call(p + "attn_post_conv.weight", dim.call([conv_size, n_embd]))
    add.call(p + "ffn_post_conv.weight", dim.call([conv_size, n_embd]))

    if sliding_pattern[il]
      add.call(p + "attn_norm.weight", dim.call([n_embd]))
    else
      add.call(p + "attn_q_inp_norm.weight", dim.call([n_embd]))
      add.call(p + "attn_k_inp_norm.weight", dim.call([n_embd]))
      add.call(p + "attn_v_inp_norm.weight", dim.call([n_embd]))
      add.call(p + "attn_k_conv_proj.weight", dim.call([n_embd, conv_size * n_head_kv]))
      add.call(p + "attn_v_conv_proj.weight", dim.call([n_embd, conv_size * n_head_kv]))
    end

    add.call(p + "attn_res_query.weight", dim.call([n_embd, sliding_pattern[il] ? 1 : 3])) if il > 0
    add.call(p + "ffn_res_query.weight", dim.call([n_embd]))
    add.call(p + "ffn_norm.weight", dim.call([n_embd]))
    add.call(p + "ffn_gate_inp.weight", dim.call([n_embd, n_expert]))
    add.call(p + "ffn_gate_up_exps.weight", dim.call([n_embd, expert_dim * 2, n_expert]))
    add.call(p + "ffn_down_exps.weight", dim.call([expert_dim, n_embd, n_expert]))
    add.call(p + "exp_probs_b.bias", dim.call([n_expert])) if il.odd?

    if core_start <= il < core_end
      add.call(p + "loop_msg_query.weight", dim.call([n_embd, 4]))
      {"q", "k", "v", "ffn"}.each do |kind|
        add.call(p + "loop_#{kind}_ctrl_down.weight", dim.call([2 * n_embd, 2]))
        add.call(p + "loop_#{kind}_ctrl_gain.weight", dim.call([2, n_embd]))
        add.call(p + "loop_#{kind}_ctrl_forget.weight", dim.call([2, n_embd]))
      end
    end
  end

  tensors
end

private def write_inkling2_test_gguf(
  path : String,
  *,
  n_layer : Int32 = 4,
  n_embd : Int32 = 4,
  n_head : Int32 = 2,
  n_head_kv : Int32 = 1,
  head_dim : Int32 = 2,
  rel_dim : Int32 = 2,
  rel_extent : Int32 = 3,
  sliding_window : Int32 = 2,
  conv_size : Int32 = 2,
  n_expert : Int32 = 2,
  n_expert_used : Int32 = 1,
  expert_dim : Int32 = 2,
  padded_vocab_size : Int32 = 8,
  vocab_size : Int32 = 7,
  res_block_size : Int32 = 1,
  core_start : Int32 = 2,
  core_end : Int32 = 3,
  passes : Int32 = 3,
  varying : Bool = false,
  sliding_pattern : Array(Bool) = [true, false, true, false],
  include_all_tensors : Bool = false,
  include_truncated_embedding : Bool = false,
  patterned_weights : Bool = false,
  quantized_experts : Bool = false,
) : Nil
  entries = inkling2_test_metadata(
    n_layer: n_layer, n_embd: n_embd, n_head: n_head, n_head_kv: n_head_kv, head_dim: head_dim,
    rel_dim: rel_dim, rel_extent: rel_extent, sliding_window: sliding_window, conv_size: conv_size,
    n_expert: n_expert, n_expert_used: n_expert_used, expert_dim: expert_dim,
    padded_vocab_size: padded_vocab_size, vocab_size: vocab_size, res_block_size: res_block_size,
    core_start: core_start, core_end: core_end, passes: passes, varying: varying,
    sliding_pattern: sliding_pattern,
  )

  tensors = if include_all_tensors
              inkling2_test_tensors(
                n_layer, n_embd, n_head, n_head_kv, head_dim, rel_dim, rel_extent,
                sliding_window, conv_size, n_expert, expert_dim, vocab_size,
                padded_vocab_size, core_start, core_end, sliding_pattern, quantized_experts,
              )
            elsif include_truncated_embedding
              [{name: "token_embd.weight", dims: [n_embd.to_i64, padded_vocab_size.to_i64], type_id: 0_u32,
                offset: 0_u64, bytes: (n_embd * padded_vocab_size * 4)}] of Inkling2TestTensor
            else
              [] of Inkling2TestTensor
            end

  cursor = 0_i64
  tensors = tensors.map do |tensor|
    cursor = (cursor + 31_i64) & ~31_i64
    updated = {name: tensor[:name], dims: tensor[:dims], type_id: tensor[:type_id], offset: cursor.to_u64, bytes: tensor[:bytes]}
    cursor += tensor[:bytes]
    updated
  end

  io = IO::Memory.new
  io.write("GGUF".to_slice)
  io.write_bytes(3_u32, IO::ByteFormat::LittleEndian)
  io.write_bytes(tensors.size.to_i64, IO::ByteFormat::LittleEndian)
  io.write_bytes(entries.size.to_i64, IO::ByteFormat::LittleEndian)
  entries.each do |key, value|
    inkling2_write_string(io, key)
    inkling2_write_metadata_value(io, value)
  end
  tensors.each do |tensor|
    inkling2_write_string(io, tensor[:name])
    io.write_bytes(tensor[:dims].size.to_u32, IO::ByteFormat::LittleEndian)
    tensor[:dims].each { |dimension| io.write_bytes(dimension, IO::ByteFormat::LittleEndian) }
    io.write_bytes(tensor[:type_id], IO::ByteFormat::LittleEndian)
    io.write_bytes(tensor[:offset], IO::ByteFormat::LittleEndian)
  end
  inkling2_pad_to(io, 32)
  data_cursor = 0_i64
  random = Random.new(54321)
  tensors.each do |tensor|
    target = tensor[:offset].to_i64
    io.write(Bytes.new((target - data_cursor).to_i, 0_u8)) if target > data_cursor
    if include_all_tensors
      if patterned_weights && tensor[:type_id] == ML::GGUF::TensorType::Q8_0.value
        (tensor[:bytes] // 34).times do
          io.write_bytes(0x2400_u16, IO::ByteFormat::LittleEndian)
          32.times { io.write_byte(random.rand(-12..12).to_i8.unsafe_as(UInt8)) }
        end
      elsif patterned_weights
        name = tensor[:name]
        (tensor[:bytes] // 4).times do |i|
          value = random.rand(-100..100).to_f32 * 0.0008_f32
          value += 1_f32 if name.ends_with?("norm.weight") || name.ends_with?("_clip.weight")
          value += i * 0.25_f32 if name.ends_with?(".bias")
          io.write_bytes(value, IO::ByteFormat::LittleEndian)
        end
      else
        io.write(Bytes.new(tensor[:bytes], 0_u8))
      end
    end
    data_cursor = target + tensor[:bytes]
  end
  File.write(path, io.to_slice)
end

private def expect_inkling2_kv_parity(reference : ML::GGUF::Inkling2Engine,
                                      actual : ML::GGUF::Inkling2Engine) : Nil
  actual.layer_visits.should eq(reference.layer_visits)
  reference.layer_visits.times do |visit|
    actual.kv_count(visit).should eq(reference.kv_count(visit))
    reference.kv_count(visit).times do |index|
      expected_k, expected_v, expected_pos = reference.key_value_at(visit, index)
      actual_k, actual_v, actual_pos = actual.key_value_at(visit, index)
      actual_pos.should eq(expected_pos)
      expected_k.zip(actual_k).each { |left, right| right.should be_close(left, 2e-4_f32) }
      expected_v.zip(actual_v).each { |left, right| right.should be_close(left, 2e-4_f32) }
    end
  end
end

describe ML::GGUF::Inkling2Engine do
  it "rejects invalid prefill before changing the causal sequence" do
    path = File.tempname("inkling2-prefill", ".gguf")
    write_inkling2_test_gguf(path, include_all_tensors: true)
    begin
      [0, 129].each do |tile|
        expect_raises(ArgumentError, /prefill tile.*1.*128/) do
          ML::GGUF::Inkling2Engine.new(path, 3, metal: false, prefill_tile: tile)
        end
      end
      expect_raises(ArgumentError, /requires resident Metal/) do
        ML::GGUF::Inkling2Engine.new(path, 3, metal: false, prefill_tile: 64)
      end
      engine = ML::GGUF::Inkling2Engine.new(path, 3, metal: false)
      begin
        expect_raises(ArgumentError, /empty/) { engine.prefill([] of Int32) }
        expect_raises(ArgumentError, /vocabulary/) { engine.prefill([0, 1, 7]) }
        expect_raises(ArgumentError, /context/) { engine.prefill([0, 1, 2, 3]) }
        engine.position.should eq(0)
        engine.prefill([1, 3]).size.should eq(7)
        engine.position.should eq(2)
        engine.forward(4).not_nil!.size.should eq(7)
      ensure
        engine.close
      end
    ensure
      File.delete?(path)
    end
  end

  {% if flag?(:darwin) && !flag?(:cpu_only) %}
    it "matches staged logits and KV across a prefill tile that wraps the window" do
      pending! "Metal unavailable" unless ML::Metal::Device.init!

      path = File.tempname("inkling2-resident", ".gguf")
      write_inkling2_test_gguf(path, n_layer: 6, n_embd: 32, expert_dim: 32, core_end: 4,
        sliding_pattern: [true, false, true, false, false, true], include_all_tensors: true,
        patterned_weights: true, quantized_experts: true)
      begin
        reference = ML::GGUF::Inkling2Engine.new(path, 16, metal: true, resident: false)
        resident = ML::GGUF::Inkling2Engine.new(path, 16, metal: true)
        begin
          resident.resident?.should be_true
          reference.resident?.should be_false
          2.times do
            [[5, 1, 3], [2, 4, 1, 6], [0]].each do |ids|
              expected = reference.prefill(ids)
              actual = ids.size == 1 ? resident.forward(ids.first).not_nil! : resident.prefill(ids)
              expected.zip(actual).each { |left, right| right.should be_close(left, 2e-4_f32) }
              resident.layer_visits.should eq(10)
              expect_inkling2_kv_parity(reference, resident)
            end
            resident.reset
            reference.reset
            resident.kv_count(0).should eq(0)
          end
        ensure
          resident.close
          reference.close
        end

        trace = ->(_position : Int32, _pass : Int32, _delta : Float32) { nil }
        traced = ML::GGUF::Inkling2Engine.new(path, 16, metal: true, trace: trace)
        begin
          traced.resident?.should be_false
          traced.forward(5).not_nil!.size.should eq(7)
        ensure
          traced.close
        end
      ensure
        File.delete?(path)
      end
    end

    it "preserves logits and per-visit KV when widening prefill across a sliding window" do
      pending! "Metal unavailable" unless ML::Metal::Device.init!

      path = File.tempname("inkling2-wide-prefill", ".gguf")
      write_inkling2_test_gguf(path, n_layer: 24, n_embd: 2304, expert_dim: 32,
        core_start: 6, core_end: 18, res_block_size: 6, varying: true, sliding_window: 48,
        sliding_pattern: Array(Bool).new(24) { |layer| layer % 3 != 1 },
        include_all_tensors: true, patterned_weights: true, quantized_experts: true)
      begin
        prefix = [5, 1, 3, 4, 6, 0, 2, 1, 4]
        prompt = Array(Int32).new(131) { |i| (i * 5 + i // 7) % 7 }
        [3, 10].each do |passes|
          reference = ML::GGUF::Inkling2Engine.new(path, 160, metal: true, passes: passes)
          begin
            reference.resident?.should be_true
            [64, 128].each do |tile|
              wide = ML::GGUF::Inkling2Engine.new(path, 160, metal: true, passes: passes, prefill_tile: tile)
              begin
                wide.resident?.should be_true
                reference.reset
                reference.prefill(prefix)
                wide.prefill(prefix)
                expected = reference.prefill(prompt)
                actual = wide.prefill(prompt)
                expected.zip(actual).each { |left, right| right.should be_close(left, 2e-4_f32) }
                expect_inkling2_kv_parity(reference, wide)
                [5, 3, 1].each do |token|
                  expected = reference.forward(token).not_nil!
                  actual = wide.forward(token).not_nil!
                  expected.zip(actual).each { |left, right| right.should be_close(left, 2e-4_f32) }
                end
                expect_inkling2_kv_parity(reference, wide)
              ensure
                wide.close
              end
            end
          ensure
            reference.close
          end
        end
      ensure
        File.delete?(path)
      end
    end
  {% end %}
end

describe ML::GGUF::Inkling2Weights do
  it "loads tiny fixed-loop GGUF tensors as mmap-backed slices" do
    path = File.tempname("inkling2-weights", ".gguf")
    write_inkling2_test_gguf(path, include_all_tensors: true)
    weights = ML::GGUF::Inkling2Weights.new(path)
    begin
      config = weights.config
      config.n_layer.should eq(4)
      config.n_embd.should eq(4)
      config.passes.should eq(3)
      config.varying?.should be_false
      config.sliding?(0).should be_true
      config.sliding?(1).should be_false
      expect_raises(IndexError) { config.sliding?(4) }

      weights.layers.size.should eq(4)
      weights.layers[0].attention.q.in_dim.should eq(4)
      weights.layers[0].attention.q.out_dim.should eq(4)
      q_info = weights.gguf.tensor("blk.0.attn_q.weight").not_nil!
      mmap = weights.gguf.mmap_region.not_nil![0]
      expected_q_address = (mmap + weights.gguf.data_offset + q_info.offset.to_i64).address
      weights.layers[0].attention.q.raw.to_unsafe.address.should eq(expected_q_address)
      weights.layers[0].attention.k_conv_proj.should be_nil
      weights.layers[0].q_norm.object_id.should eq(weights.layers[0].k_norm.object_id)
      weights.layers[0].attn_res_query.should be_nil
      weights.layers[1].attention.k_conv_proj.should_not be_nil
      weights.layers[1].q_norm.object_id.should_not eq(weights.layers[1].k_norm.object_id)
      weights.layers[2].controller.not_nil!.query.size.should eq(16)
      weights.layers[2].controller.not_nil!.down.size.should eq(4)
      weights.layers[2].gate_up.expert(1).in_dim.should eq(4)
      weights.layers[2].gate_up.expert(1).out_dim.should eq(4)
      weights.layers[2].down.expert(1).in_dim.should eq(2)
      weights.layers[2].down.expert(1).out_dim.should eq(4)
      weights.embedding_row(6).size.should eq(4)
      expect_raises(IndexError) { weights.embedding_row(7) }
    ensure
      weights.close
      File.delete?(path)
    end
  end

  it "rejects a fixed-loop checkpoint whose metadata changes its three passes" do
    path = File.tempname("inkling2-fixed-passes", ".gguf")
    write_inkling2_test_gguf(path, passes: 4)
    begin
      expect_raises(ArgumentError, /fixed Inkling 2 checkpoints require exactly 3 passes/) do
        ML::GGUF::Inkling2Weights.new(path)
      end
    ensure
      File.delete?(path)
    end
  end

  it "checks tensor byte extents before taking mmap slices" do
    path = File.tempname("inkling2-truncated-tensor", ".gguf")
    write_inkling2_test_gguf(path, include_truncated_embedding: true)
    begin
      expect_raises(ArgumentError, /extends beyond the GGUF file/) do
        ML::GGUF::Inkling2Weights.new(path)
      end
    ensure
      File.delete?(path)
    end
  end

  it "admits the variable-loop metadata variant with its three-pass default" do
    path = File.tempname("inkling2-varloop-config", ".gguf")
    pattern = Array(Bool).new(24, true)
    write_inkling2_test_gguf(
      path,
      n_layer: 24,
      n_embd: 2304,
      n_head: 24,
      n_head_kv: 8,
      head_dim: 128,
      rel_dim: 16,
      rel_extent: 1024,
      sliding_window: 1024,
      conv_size: 4,
      n_expert: 128,
      n_expert_used: 8,
      expert_dim: 1152,
      padded_vocab_size: 201024,
      vocab_size: 200058,
      res_block_size: 6,
      core_start: 6,
      core_end: 18,
      varying: true,
      sliding_pattern: pattern,
    )
    gguf = ML::GGUF::GGUFFile.new(path, mmap_tensors: false)
    begin
      config = ML::GGUF::Inkling2Config.new(gguf)
      config.varying?.should be_true
      config.passes.should eq(3)
      config.core_start.should eq(6)
      config.core_end.should eq(18)
    ensure
      gguf.close
      File.delete?(path)
    end
  end

  it "rejects variable-loop metadata with an incompatible core layout" do
    path = File.tempname("inkling2-varloop-layout", ".gguf")
    write_inkling2_test_gguf(path, varying: true)
    gguf = ML::GGUF::GGUFFile.new(path, mmap_tensors: false)
    begin
      expect_raises(ArgumentError, /variable loops are admitted only/) do
        ML::GGUF::Inkling2Config.new(gguf)
      end
    ensure
      gguf.close
      File.delete?(path)
    end
  end
end
