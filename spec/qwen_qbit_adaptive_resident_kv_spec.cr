require "./spec_helper"
require "../src/ml/gguf/qwen35_cpu"
require "../src/ml/gguf/qwen_qbit_adaptive_kv"
require "../src/ml/gguf/qwen_qbit_adaptive_resident_kv"

module QwenQBitAdaptiveResidentKVSpec
  extend self

  def reference(q : Array(Float32), gate : Array(Float32),
                k_cache : Array(Float32), v_cache : Array(Float32),
                cache_len : Int32, n_head : Int32, n_head_kv : Int32,
                head_dim : Int32, heads_per_group : Int32,
                scale : Float32) : Array(Float32)
    kv_dim = n_head_kv * head_dim
    out = Array(Float32).new(n_head * head_dim, 0.0_f32)
    scores = Array(Float32).new(cache_len, 0.0_f32)

    n_head.times do |h|
      kv_h = h // heads_per_group
      q_offset = h * head_dim
      cache_len.times do |position|
        k_offset = position * kv_dim + kv_h * head_dim
        dot = 0.0_f32
        head_dim.times { |d| dot += q[q_offset + d] * k_cache[k_offset + d] }
        scores[position] = dot * scale
      end
      ML::GGUF::Qwen35CPU.softmax_slice!(scores, 0, cache_len)

      cache_len.times do |position|
        v_offset = position * kv_dim + kv_h * head_dim
        weight = scores[position]
        head_dim.times { |d| out[q_offset + d] += weight * v_cache[v_offset + d] }
      end
    end

    out.size.times do |i|
      out[i] *= ML::GGUF::Qwen35CPU.sigmoid(gate[i])
    end
    out
  end

  def chunk_reference(q : Array(Float32), gate : Array(Float32),
                      packed_k : Array(Float32), packed_v : Array(Float32),
                      current_k : Array(Float32), current_v : Array(Float32),
                      packed_len : Int32, token_count : Int32,
                      n_head : Int32, n_head_kv : Int32,
                      head_dim : Int32, heads_per_group : Int32,
                      scale : Float32) : Array(Float32)
    q_dim = n_head * head_dim
    kv_dim = n_head_kv * head_dim
    out = Array(Float32).new(token_count * q_dim, 0.0_f32)
    scores = Array(Float32).new(packed_len + token_count, 0.0_f32)

    token_count.times do |token|
      visible_len = packed_len + token + 1
      n_head.times do |h|
        kv_h = h // heads_per_group
        q_offset = token * q_dim + h * head_dim
        visible_len.times do |position|
          if position < packed_len
            k_values = packed_k
            k_offset = position * kv_dim + kv_h * head_dim
          else
            k_values = current_k
            k_offset = (position - packed_len) * kv_dim + kv_h * head_dim
          end
          dot = 0.0_f32
          head_dim.times { |d| dot += q[q_offset + d] * k_values[k_offset + d] }
          scores[position] = dot * scale
        end
        ML::GGUF::Qwen35CPU.softmax_slice!(scores, 0, visible_len)

        visible_len.times do |position|
          if position < packed_len
            v_values = packed_v
            v_offset = position * kv_dim + kv_h * head_dim
          else
            v_values = current_v
            v_offset = (position - packed_len) * kv_dim + kv_h * head_dim
          end
          weight = scores[position]
          head_dim.times do |d|
            out[q_offset + d] += weight * v_values[v_offset + d]
          end
        end
      end
    end

    out.size.times do |i|
      out[i] *= ML::GGUF::Qwen35CPU.sigmoid(gate[i])
    end
    out
  end

  def max_diff(a : Array(Float32), b : Array(Float32)) : Float32
    a.each_with_index.max_of { |value, i| (value - b[i]).abs }
  end

  def cosine(a : Array(Float32), b : Array(Float32)) : Float64
    dot = 0.0_f64
    aa = 0.0_f64
    bb = 0.0_f64
    a.each_with_index do |value, i|
      x = value.to_f64
      y = b[i].to_f64
      dot += x * y
      aa += x * x
      bb += y * y
    end
    dot / (Math.sqrt(aa) * Math.sqrt(bb))
  end
end

describe ML::GGUF::QwenQBitAdaptiveResidentKV do
  adaptive = ML::GGUF::QwenQBitAdaptiveKV
  tier = ML::GGUF::QwenQBitAdaptiveKV::Tier

  it "decodes mixed p4, p5, BF16, and F32 rows inside GQA6 Metal attention" do
    pending!("Metal not available") unless ML::GGUF::Qwen35Metal.available?

    # Cross the 16-token kernel tile boundary while cycling through every tier.
    cache_len = 19
    n_head = 24
    n_head_kv = 4
    head_dim = 256
    heads_per_group = 6
    scale = (1.0 / Math.sqrt(head_dim.to_f64)).to_f32
    q_dim = n_head * head_dim
    kv_count = cache_len * n_head_kv * head_dim
    row_count = cache_len * n_head_kv
    rng = Random.new(0xADA9717E_u64)
    q = Array(Float32).new(q_dim) { ((rng.next_float - 0.5) * 1.0).to_f32 }
    gate = Array(Float32).new(q_dim) { ((rng.next_float - 0.5) * 2.0).to_f32 }
    k = Array(Float32).new(kv_count) { ((rng.next_float - 0.5) * 1.0).to_f32 }
    v = Array(Float32).new(kv_count) { ((rng.next_float - 0.5) * 1.0).to_f32 }
    k_tiers = Array(ML::GGUF::QwenQBitAdaptiveKV::Tier).new(row_count) do |i|
      tier.from_value(i % 4)
    end
    v_tiers = Array(ML::GGUF::QwenQBitAdaptiveKV::Tier).new(row_count) do |i|
      tier.from_value((i + 2) % 4)
    end
    encoded_k = adaptive.encode(k, k_tiers, block_size: head_dim)
    encoded_v = adaptive.encode(v, v_tiers, block_size: head_dim)
    expected = QwenQBitAdaptiveResidentKVSpec.reference(
      q, gate, adaptive.decode(encoded_k), adaptive.decode(encoded_v),
      cache_len, n_head, n_head_kv, head_dim, heads_per_group, scale,
    )

    live_before = ML::MetalBuffer.stats[:live_bytes]
    resident = ML::GGUF::QwenQBitAdaptiveResidentKV.prepare(
      encoded_k, encoded_v, cache_len, n_head_kv, head_dim,
    )
    resident.compressed_bytes.should eq(
      (encoded_k.payload_bytes + encoded_v.payload_bytes).to_i64,
    )
    begin
      gpu_elapsed_seconds = 0.0_f64
      actual = ML::GGUF::QwenQBitAdaptiveResidentKV.attn_decode(
        q, gate, resident, n_head, heads_per_group, scale,
        gpu_elapsed_seconds: pointerof(gpu_elapsed_seconds),
      )
      gpu_elapsed_seconds.should be > 0.0_f64
    ensure
      resident.release
    end
    ML::MetalBuffer.stats[:live_bytes].should eq(live_before)

    cosine = QwenQBitAdaptiveResidentKVSpec.cosine(expected, actual)
    diff = QwenQBitAdaptiveResidentKVSpec.max_diff(expected, actual)
    printf "  [adaptive resident QBit] cos=%.12f max|delta|=%g ratio=%.3fx\n",
      cosine, diff, (2 * kv_count * sizeof(Float32)).to_f64 /
                    (encoded_k.payload_bytes + encoded_v.payload_bytes)
    cosine.should be > 0.9999999
    diff.should be < 2.0e-4_f32
  end

  it "packs append-only F32 Metal chunks directly into a canonical resident cache" do
    pending!("Metal not available") unless ML::GGUF::Qwen35Metal.available?

    cache_len = 19
    first_chunk = 7
    n_head = 24
    n_head_kv = 4
    head_dim = 256
    heads_per_group = 6
    scale = (1.0 / Math.sqrt(head_dim.to_f64)).to_f32
    row_count = cache_len * n_head_kv
    value_count = row_count * head_dim
    rng = Random.new(0xA991D0_u64)
    k = Array(Float32).new(value_count) { ((rng.next_float - 0.5) * 1.0).to_f32 }
    v = Array(Float32).new(value_count) { ((rng.next_float - 0.5) * 1.0).to_f32 }
    q = Array(Float32).new(n_head * head_dim) { ((rng.next_float - 0.5) * 1.0).to_f32 }
    gate = Array(Float32).new(n_head * head_dim) { ((rng.next_float - 0.5) * 2.0).to_f32 }
    k_tiers = Array(ML::GGUF::QwenQBitAdaptiveKV::Tier).new(row_count) { |i| tier.from_value(i % 4) }
    v_tiers = Array(ML::GGUF::QwenQBitAdaptiveKV::Tier).new(row_count) { |i| tier.from_value((i + 2) % 4) }
    k_plan = adaptive.plan(k_tiers)
    v_plan = adaptive.plan(v_tiers)

    live_before = ML::MetalBuffer.stats[:live_bytes]
    k_source = ML::MetalBuffer.from_array(k)
    v_source = ML::MetalBuffer.from_array(v)
    resident = ML::GGUF::QwenQBitAdaptiveResidentKV.allocate(
      k_plan, v_plan, cache_len, n_head_kv, head_dim,
    )
    begin
      ML::GGUF::QwenQBitAdaptiveResidentKV.append_from_metal(
        resident, k_source, v_source, first_chunk,
      )
      resident.cache_len.should eq(first_chunk)
      resident.live_compressed_bytes.should eq(
        first_chunk * n_head_kv *
        (2 * ML::GGUF::QwenQBitAdaptiveKV::BASE_ROW_BYTES +
         2 * ML::GGUF::QwenQBitAdaptiveKV::METADATA_BYTES) +
        k_plan.prefix_sidecar_bytes(first_chunk * n_head_kv) +
        v_plan.prefix_sidecar_bytes(first_chunk * n_head_kv),
      )
      first_k, first_v = ML::GGUF::QwenQBitAdaptiveResidentKV.snapshot(resident)
      adaptive.validate(first_k)
      adaptive.validate(first_v)

      pack_gpu_elapsed_seconds = 0.0_f64
      ML::GGUF::QwenQBitAdaptiveResidentKV.append_from_metal(
        resident, k_source, v_source, cache_len - first_chunk,
        source_token_offset: first_chunk,
        gpu_elapsed_seconds: pointerof(pack_gpu_elapsed_seconds),
      )
      pack_gpu_elapsed_seconds.should be > 0.0_f64
      resident.cache_len.should eq(cache_len)
      packed_k, packed_v = ML::GGUF::QwenQBitAdaptiveResidentKV.snapshot(resident)
      ML::GGUF::QwenQBitAdaptiveResidentKV.snapshot_k(resident).payload.should eq(packed_k.payload)
      ML::GGUF::QwenQBitAdaptiveResidentKV.snapshot_v(resident).payload.should eq(packed_v.payload)
      decoded_k = adaptive.decode(packed_k)
      decoded_v = adaptive.decode(packed_v)
      cpu_k = adaptive.decode(adaptive.encode(k, k_tiers))
      cpu_v = adaptive.decode(adaptive.encode(v, v_tiers))

      QwenQBitAdaptiveResidentKVSpec.max_diff(decoded_k, cpu_k).should be < 2.0e-5_f32
      QwenQBitAdaptiveResidentKVSpec.max_diff(decoded_v, cpu_v).should be < 2.0e-5_f32
      expected = QwenQBitAdaptiveResidentKVSpec.reference(
        q, gate, decoded_k, decoded_v, cache_len,
        n_head, n_head_kv, head_dim, heads_per_group, scale,
      )
      actual = ML::GGUF::QwenQBitAdaptiveResidentKV.attn_decode(
        q, gate, resident, n_head, heads_per_group, scale,
      )
      QwenQBitAdaptiveResidentKVSpec.cosine(expected, actual).should be > 0.9999999
      QwenQBitAdaptiveResidentKVSpec.max_diff(expected, actual).should be < 2.0e-4_f32

      unsupported_q = Array(Float32).new(4 * head_dim, 0.0_f32)
      expect_raises(ArgumentError, /GQA6/) do
        ML::GGUF::QwenQBitAdaptiveResidentKV.attn_decode(
          unsupported_q, unsupported_q, resident, 4, 1, scale,
        )
      end

      expect_raises(ArgumentError, /capacity/) do
        ML::GGUF::QwenQBitAdaptiveResidentKV.append_from_metal(
          resident, k_source, v_source, 1,
        )
      end
    ensure
      resident.release
      k_source.release
      v_source.release
    end
    ML::MetalBuffer.stats[:live_bytes].should eq(live_before)
  end

  it "keeps prefix-only pack quantization byte-identical across every tier" do
    pending!("Metal not available") unless ML::GGUF::Qwen35Metal.available?

    token_count = 8
    n_head_kv = 4
    head_dim = 256
    row_count = token_count * n_head_kv
    value_count = row_count * head_dim
    rng = Random.new(0xA991F1_u64)
    k = Array(Float32).new(value_count) { ((rng.next_float - 0.5) * 4.0).to_f32 }
    v = Array(Float32).new(value_count) { ((rng.next_float - 0.5) * 4.0).to_f32 }
    k_plan = adaptive.plan(Array.new(row_count) { |i| tier.from_value(i % 4) })
    v_plan = adaptive.plan(Array.new(row_count) { |i| tier.from_value((i + 2) % 4) })
    legacy = ML::GGUF::QwenQBitAdaptiveResidentKV.allocate(
      k_plan, v_plan, token_count, n_head_kv, head_dim,
    )
    prefix = ML::GGUF::QwenQBitAdaptiveResidentKV.allocate(
      k_plan, v_plan, token_count, n_head_kv, head_dim,
    )
    buffers = [ML::MetalBuffer.from_array(k), ML::MetalBuffer.from_array(v)]
    previous_prefix_quant = ENV["QWEN35_ADAPTIVE_PACK_PREFIX_QUANT"]?
    begin
      ENV["QWEN35_ADAPTIVE_PACK_PREFIX_QUANT"] = "0"
      ML::GGUF::QwenQBitAdaptiveResidentKV.append_from_metal(
        legacy, buffers[0], buffers[1], token_count,
      )
      ENV["QWEN35_ADAPTIVE_PACK_PREFIX_QUANT"] = "1"
      ML::GGUF::QwenQBitAdaptiveResidentKV.append_from_metal(
        prefix, buffers[0], buffers[1], token_count,
      )

      legacy_k, legacy_v = ML::GGUF::QwenQBitAdaptiveResidentKV.snapshot(legacy)
      prefix_k, prefix_v = ML::GGUF::QwenQBitAdaptiveResidentKV.snapshot(prefix)
      prefix_k.payload.should eq(legacy_k.payload)
      prefix_v.payload.should eq(legacy_v.payload)
    ensure
      if previous_prefix_quant
        ENV["QWEN35_ADAPTIVE_PACK_PREFIX_QUANT"] = previous_prefix_quant
      else
        ENV.delete("QWEN35_ADAPTIVE_PACK_PREFIX_QUANT")
      end
      buffers.each(&.release)
      prefix.release
      legacy.release
    end
  end

  it "attends over packed history plus an exact current chunk before publishing the append" do
    pending!("Metal not available") unless ML::GGUF::Qwen35Metal.available?

    cache_len = 20
    first_chunk = 7
    second_chunk = cache_len - first_chunk - 1
    n_head = 24
    n_head_kv = 4
    head_dim = 256
    heads_per_group = 6
    q_dim = n_head * head_dim
    kv_dim = n_head_kv * head_dim
    scale = (1.0 / Math.sqrt(head_dim.to_f64)).to_f32
    row_count = cache_len * n_head_kv
    rng = Random.new(0xC4A5CA1E_u64)
    q = Array(Float32).new(cache_len * q_dim) { ((rng.next_float - 0.5) * 1.0).to_f32 }
    gate = Array(Float32).new(cache_len * q_dim) { ((rng.next_float - 0.5) * 2.0).to_f32 }
    k = Array(Float32).new(cache_len * kv_dim) { ((rng.next_float - 0.5) * 1.0).to_f32 }
    v = Array(Float32).new(cache_len * kv_dim) { ((rng.next_float - 0.5) * 1.0).to_f32 }
    k_tiers = Array(ML::GGUF::QwenQBitAdaptiveKV::Tier).new(row_count) { |i| tier.from_value(i % 4) }
    v_tiers = Array(ML::GGUF::QwenQBitAdaptiveKV::Tier).new(row_count) { |i| tier.from_value((i + 2) % 4) }
    k_plan = adaptive.plan(k_tiers)
    v_plan = adaptive.plan(v_tiers)

    live_before = ML::MetalBuffer.stats[:live_bytes]
    resident = ML::GGUF::QwenQBitAdaptiveResidentKV.allocate(
      k_plan, v_plan, cache_len, n_head_kv, head_dim,
    )
    begin
      chunks = [
        {0, first_chunk},
        {first_chunk, second_chunk},
        {first_chunk + second_chunk, 1},
      ]
      decoded_history_k = [] of Float32
      decoded_history_v = [] of Float32

      chunks.each do |(start_token, token_count)|
        q_chunk = q[start_token * q_dim, token_count * q_dim]
        gate_chunk = gate[start_token * q_dim, token_count * q_dim]
        k_chunk = k[start_token * kv_dim, token_count * kv_dim]
        v_chunk = v[start_token * kv_dim, token_count * kv_dim]
        expected = QwenQBitAdaptiveResidentKVSpec.chunk_reference(
          q_chunk, gate_chunk, decoded_history_k, decoded_history_v,
          k_chunk, v_chunk, start_token, token_count,
          n_head, n_head_kv, head_dim, heads_per_group, scale,
        )

        buffers = [
          ML::MetalBuffer.from_array(q_chunk),
          ML::MetalBuffer.from_array(gate_chunk),
          ML::MetalBuffer.from_array(k_chunk),
          ML::MetalBuffer.from_array(v_chunk),
          ML::MetalBuffer.new(expected.size.to_i64 * sizeof(Float32)),
        ]
        begin
          ML::GGUF::QwenQBitAdaptiveResidentKV.prefill_chunk_and_append_from_metal(
            resident, buffers[0], buffers[1], buffers[2], buffers[3], buffers[4],
            token_count, n_head, heads_per_group, scale,
          )
          actual = buffers[4].read(expected.size.to_i32)
          QwenQBitAdaptiveResidentKVSpec.cosine(expected, actual).should be > 0.9999999
          QwenQBitAdaptiveResidentKVSpec.max_diff(expected, actual).should be < 2.0e-4_f32
        ensure
          buffers.each(&.release)
        end

        resident.cache_len.should eq(start_token + token_count)
        packed_k, packed_v = ML::GGUF::QwenQBitAdaptiveResidentKV.snapshot(resident)
        adaptive.validate(packed_k)
        adaptive.validate(packed_v)
        decoded_history_k = adaptive.decode(packed_k)
        decoded_history_v = adaptive.decode(packed_v)
      end
    ensure
      resident.release
    end
    ML::MetalBuffer.stats[:live_bytes].should eq(live_before)
  end

  it "matches the serial adaptive decode across an 8K split-K prefix" do
    pending!("Metal not available") unless ML::GGUF::Qwen35Metal.available?

    packed_len = 8191
    capacity = packed_len + 2
    n_head = 6
    n_head_kv = 1
    head_dim = 256
    heads_per_group = 6
    q_dim = n_head * head_dim
    kv_dim = n_head_kv * head_dim
    scale = (1.0 / Math.sqrt(head_dim.to_f64)).to_f32
    rng = Random.new(0x5A117A_u64)
    initial_k = Array(Float32).new(packed_len * kv_dim) { ((rng.next_float - 0.5) * 1.0).to_f32 }
    initial_v = Array(Float32).new(packed_len * kv_dim) { ((rng.next_float - 0.5) * 1.0).to_f32 }
    q = Array(Float32).new(q_dim) { ((rng.next_float - 0.5) * 1.0).to_f32 }
    gate = Array(Float32).new(q_dim) { ((rng.next_float - 0.5) * 2.0).to_f32 }
    current_k = Array(Float32).new(kv_dim) { ((rng.next_float - 0.5) * 1.0).to_f32 }
    current_v = Array(Float32).new(kv_dim) { ((rng.next_float - 0.5) * 1.0).to_f32 }
    plan = adaptive.plan(Array.new(
      capacity * n_head_kv,
      ML::GGUF::QwenQBitAdaptiveKV::Tier::P4,
    ))
    baseline = ML::GGUF::QwenQBitAdaptiveResidentKV.allocate(
      plan, plan, capacity, n_head_kv, head_dim,
    )
    t4_serial = ML::GGUF::QwenQBitAdaptiveResidentKV.allocate(
      plan, plan, capacity, n_head_kv, head_dim,
    )
    scalar_splitk = ML::GGUF::QwenQBitAdaptiveResidentKV.allocate(
      plan, plan, capacity, n_head_kv, head_dim,
    )
    t4_splitk = ML::GGUF::QwenQBitAdaptiveResidentKV.allocate(
      plan, plan, capacity, n_head_kv, head_dim,
    )
    legacy_splitk = ML::GGUF::QwenQBitAdaptiveResidentKV.allocate(
      plan, plan, capacity, n_head_kv, head_dim,
    )
    t8_splitk = ML::GGUF::QwenQBitAdaptiveResidentKV.allocate(
      plan, plan, capacity, n_head_kv, head_dim,
    )
    initial_buffers = [
      ML::MetalBuffer.from_array(initial_k),
      ML::MetalBuffer.from_array(initial_v),
    ]
    previous_splitk = ENV["QWEN35_ADAPTIVE_SPLITK"]?
    previous_dequant_t4 = ENV["QWEN35_ADAPTIVE_DEQUANT_T4"]?
    previous_stage2_fused = ENV["QWEN35_ADAPTIVE_SPLITK_STAGE2_FUSED"]?
    previous_p4_t8 = ENV["QWEN35_ADAPTIVE_P4_SPLITK_T8"]?
    begin
      [baseline, t4_serial, scalar_splitk, t4_splitk, legacy_splitk, t8_splitk].each do |cache|
        ML::GGUF::QwenQBitAdaptiveResidentKV.append_from_metal(
          cache, initial_buffers[0], initial_buffers[1], packed_len,
        )
      end
      hits_before, misses_before = ML::GGUF::Qwen35Metal::Scratch.stats
      packed_k, packed_v = ML::GGUF::QwenQBitAdaptiveResidentKV.snapshot(baseline)
      expected = QwenQBitAdaptiveResidentKVSpec.chunk_reference(
        q, gate, adaptive.decode(packed_k), adaptive.decode(packed_v),
        current_k, current_v, packed_len, 1,
        n_head, n_head_kv, head_dim, heads_per_group, scale,
      )

      outputs = [] of Array(Float32)
      cases = [
        {cache: baseline, splitk: "0", dequant_t4: "0", stage2_fused: "0", p4_t8: "0"},
        {cache: t4_serial, splitk: "0", dequant_t4: "1", stage2_fused: "1", p4_t8: "0"},
        {cache: scalar_splitk, splitk: "1", dequant_t4: "0", stage2_fused: "auto", p4_t8: "0"},
        {cache: t4_splitk, splitk: "1", dequant_t4: "1", stage2_fused: "1", p4_t8: "0"},
        {cache: legacy_splitk, splitk: "1", dequant_t4: "0", stage2_fused: "0", p4_t8: "0"},
        {cache: t8_splitk, splitk: "1", dequant_t4: "1", stage2_fused: "0", p4_t8: "1"},
      ]
      cases.each do |candidate|
        ENV["QWEN35_ADAPTIVE_SPLITK"] = candidate[:splitk]
        ENV["QWEN35_ADAPTIVE_DEQUANT_T4"] = candidate[:dequant_t4]
        ENV["QWEN35_ADAPTIVE_P4_SPLITK_T8"] = candidate[:p4_t8]
        if candidate[:stage2_fused] == "auto"
          ENV.delete("QWEN35_ADAPTIVE_SPLITK_STAGE2_FUSED")
        else
          ENV["QWEN35_ADAPTIVE_SPLITK_STAGE2_FUSED"] = candidate[:stage2_fused]
        end
        buffers = [
          ML::MetalBuffer.from_array(q),
          ML::MetalBuffer.from_array(gate),
          ML::MetalBuffer.from_array(current_k),
          ML::MetalBuffer.from_array(current_v),
          ML::MetalBuffer.new(q_dim.to_i64 * sizeof(Float32)),
        ]
        begin
          ML::GGUF::Qwen35Metal::Scratch.with_namespace("adaptive_splitk_capacity_spec") do
            ML::GGUF::QwenQBitAdaptiveResidentKV.prefill_chunk_and_append_from_metal(
              candidate[:cache], buffers[0], buffers[1], buffers[2], buffers[3], buffers[4],
              1, n_head, heads_per_group, scale,
            )
          end
          outputs << buffers[4].read(q_dim)
        ensure
          buffers.each(&.release)
        end
      end

      outputs.each do |actual|
        QwenQBitAdaptiveResidentKVSpec.cosine(expected, actual).should be > 0.9999999
        QwenQBitAdaptiveResidentKVSpec.max_diff(expected, actual).should be < 2.0e-4_f32
      end
      outputs[1..].each do |actual|
        QwenQBitAdaptiveResidentKVSpec.cosine(outputs[0], actual).should be > 0.9999999
        QwenQBitAdaptiveResidentKVSpec.max_diff(outputs[0], actual).should be < 2.0e-5_f32
      end
      baseline_k, baseline_v = ML::GGUF::QwenQBitAdaptiveResidentKV.snapshot(baseline)
      t4_serial_k, t4_serial_v = ML::GGUF::QwenQBitAdaptiveResidentKV.snapshot(t4_serial)
      scalar_splitk_k, scalar_splitk_v = ML::GGUF::QwenQBitAdaptiveResidentKV.snapshot(scalar_splitk)
      t4_splitk_k, t4_splitk_v = ML::GGUF::QwenQBitAdaptiveResidentKV.snapshot(t4_splitk)
      legacy_splitk_k, legacy_splitk_v = ML::GGUF::QwenQBitAdaptiveResidentKV.snapshot(legacy_splitk)
      t8_splitk_k, t8_splitk_v = ML::GGUF::QwenQBitAdaptiveResidentKV.snapshot(t8_splitk)
      t4_serial_k.payload.should eq(baseline_k.payload)
      t4_serial_v.payload.should eq(baseline_v.payload)
      scalar_splitk_k.payload.should eq(baseline_k.payload)
      scalar_splitk_v.payload.should eq(baseline_v.payload)
      t4_splitk_k.payload.should eq(baseline_k.payload)
      t4_splitk_v.payload.should eq(baseline_v.payload)
      legacy_splitk_k.payload.should eq(baseline_k.payload)
      legacy_splitk_v.payload.should eq(baseline_v.payload)
      t8_splitk_k.payload.should eq(baseline_k.payload)
      t8_splitk_v.payload.should eq(baseline_v.payload)

      hits_after_first, misses_after_first = ML::GGUF::Qwen35Metal::Scratch.stats
      misses_after_first.should eq(misses_before + 3)
      hits_after_first.should eq(hits_before + 9)
      ENV["QWEN35_ADAPTIVE_SPLITK"] = "1"
      ENV["QWEN35_ADAPTIVE_DEQUANT_T4"] = "1"
      ENV["QWEN35_ADAPTIVE_SPLITK_STAGE2_FUSED"] = "1"
      ENV["QWEN35_ADAPTIVE_P4_SPLITK_T8"] = "0"
      second_buffers = [
        ML::MetalBuffer.from_array(q),
        ML::MetalBuffer.from_array(gate),
        ML::MetalBuffer.from_array(current_k),
        ML::MetalBuffer.from_array(current_v),
        ML::MetalBuffer.new(q_dim.to_i64 * sizeof(Float32)),
      ]
      begin
        ML::GGUF::Qwen35Metal::Scratch.with_namespace("adaptive_splitk_capacity_spec") do
          ML::GGUF::QwenQBitAdaptiveResidentKV.prefill_chunk_and_append_from_metal(
            t4_splitk, second_buffers[0], second_buffers[1], second_buffers[2],
            second_buffers[3], second_buffers[4], 1, n_head, heads_per_group, scale,
          )
        end
      ensure
        second_buffers.each(&.release)
      end
      hits_after_second, misses_after_second = ML::GGUF::Qwen35Metal::Scratch.stats
      hits_after_second.should eq(hits_after_first + 3)
      misses_after_second.should eq(misses_after_first)
    ensure
      if previous_splitk
        ENV["QWEN35_ADAPTIVE_SPLITK"] = previous_splitk
      else
        ENV.delete("QWEN35_ADAPTIVE_SPLITK")
      end
      if previous_dequant_t4
        ENV["QWEN35_ADAPTIVE_DEQUANT_T4"] = previous_dequant_t4
      else
        ENV.delete("QWEN35_ADAPTIVE_DEQUANT_T4")
      end
      if previous_stage2_fused
        ENV["QWEN35_ADAPTIVE_SPLITK_STAGE2_FUSED"] = previous_stage2_fused
      else
        ENV.delete("QWEN35_ADAPTIVE_SPLITK_STAGE2_FUSED")
      end
      if previous_p4_t8
        ENV["QWEN35_ADAPTIVE_P4_SPLITK_T8"] = previous_p4_t8
      else
        ENV.delete("QWEN35_ADAPTIVE_P4_SPLITK_T8")
      end
      initial_buffers.each(&.release)
      baseline.release
      t4_serial.release
      scalar_splitk.release
      t4_splitk.release
      legacy_splitk.release
      t8_splitk.release
    end
  end

  it "matches the CPU reference for a uniform BF16 prefill plan" do
    pending!("Metal not available") unless ML::GGUF::Qwen35Metal.available?

    initial_tokens = 3
    token_count = 2
    capacity = initial_tokens + token_count
    n_head = 6
    n_head_kv = 1
    head_dim = 256
    heads_per_group = 6
    q_dim = n_head * head_dim
    kv_dim = n_head_kv * head_dim
    scale = (1.0 / Math.sqrt(head_dim.to_f64)).to_f32
    rng = Random.new(0xBF160001_u64)
    initial_k = Array(Float32).new(initial_tokens * kv_dim) { ((rng.next_float - 0.5) * 1.0).to_f32 }
    initial_v = Array(Float32).new(initial_tokens * kv_dim) { ((rng.next_float - 0.5) * 1.0).to_f32 }
    q = Array(Float32).new(token_count * q_dim) { ((rng.next_float - 0.5) * 1.0).to_f32 }
    gate = Array(Float32).new(token_count * q_dim) { ((rng.next_float - 0.5) * 2.0).to_f32 }
    current_k = Array(Float32).new(token_count * kv_dim) { ((rng.next_float - 0.5) * 1.0).to_f32 }
    current_v = Array(Float32).new(token_count * kv_dim) { ((rng.next_float - 0.5) * 1.0).to_f32 }
    plan = adaptive.plan(Array.new(capacity, ML::GGUF::QwenQBitAdaptiveKV::Tier::BF16))

    live_before = ML::MetalBuffer.stats[:live_bytes]
    baseline = ML::GGUF::QwenQBitAdaptiveResidentKV.allocate(
      plan, plan, capacity, n_head_kv, head_dim,
    )
    t4 = ML::GGUF::QwenQBitAdaptiveResidentKV.allocate(
      plan, plan, capacity, n_head_kv, head_dim,
    )
    previous_dequant_t4 = ENV["QWEN35_ADAPTIVE_DEQUANT_T4"]?
    begin
      initial_buffers = [
        ML::MetalBuffer.from_array(initial_k),
        ML::MetalBuffer.from_array(initial_v),
      ]
      begin
        [baseline, t4].each do |resident|
          ML::GGUF::QwenQBitAdaptiveResidentKV.append_from_metal(
            resident, initial_buffers[0], initial_buffers[1], initial_tokens,
          )
        end
      ensure
        initial_buffers.each(&.release)
      end

      packed_k, packed_v = ML::GGUF::QwenQBitAdaptiveResidentKV.snapshot(baseline)
      expected = QwenQBitAdaptiveResidentKVSpec.chunk_reference(
        q, gate, adaptive.decode(packed_k), adaptive.decode(packed_v),
        current_k, current_v, initial_tokens, token_count,
        n_head, n_head_kv, head_dim, heads_per_group, scale,
      )
      outputs = [] of Array(Float32)
      [{resident: baseline, dequant_t4: "0"}, {resident: t4, dequant_t4: "1"}].each do |candidate|
        ENV["QWEN35_ADAPTIVE_DEQUANT_T4"] = candidate[:dequant_t4]
        buffers = [
          ML::MetalBuffer.from_array(q),
          ML::MetalBuffer.from_array(gate),
          ML::MetalBuffer.from_array(current_k),
          ML::MetalBuffer.from_array(current_v),
          ML::MetalBuffer.new(expected.size.to_i64 * sizeof(Float32)),
        ]
        begin
          ML::GGUF::QwenQBitAdaptiveResidentKV.prefill_chunk_and_append_from_metal(
            candidate[:resident],
            buffers[0], buffers[1], buffers[2], buffers[3], buffers[4],
            token_count, n_head, heads_per_group, scale,
          )
          outputs << buffers[4].read(expected.size.to_i32)
        ensure
          buffers.each(&.release)
        end
      end
      outputs.each do |actual|
        QwenQBitAdaptiveResidentKVSpec.cosine(expected, actual).should be > 0.9999999
        QwenQBitAdaptiveResidentKVSpec.max_diff(expected, actual).should be < 2.0e-4_f32
      end
      QwenQBitAdaptiveResidentKVSpec.cosine(outputs[0], outputs[1]).should be > 0.9999999
      QwenQBitAdaptiveResidentKVSpec.max_diff(outputs[0], outputs[1]).should be < 2.0e-5_f32
      baseline_k, baseline_v = ML::GGUF::QwenQBitAdaptiveResidentKV.snapshot(baseline)
      t4_k, t4_v = ML::GGUF::QwenQBitAdaptiveResidentKV.snapshot(t4)
      t4_k.payload.should eq(baseline_k.payload)
      t4_v.payload.should eq(baseline_v.payload)
    ensure
      if previous_dequant_t4
        ENV["QWEN35_ADAPTIVE_DEQUANT_T4"] = previous_dequant_t4
      else
        ENV.delete("QWEN35_ADAPTIVE_DEQUANT_T4")
      end
      baseline.release
      t4.release
    end
    ML::MetalBuffer.stats[:live_bytes].should eq(live_before)
  end

  it "keeps the vector BF16 split-K loader numerically and byte equivalent at 8K" do
    pending!("Metal not available") unless ML::GGUF::Qwen35Metal.available?

    packed_len = 8191
    capacity = packed_len + 2
    n_head = 6
    n_head_kv = 1
    head_dim = 256
    heads_per_group = 6
    q_dim = n_head * head_dim
    kv_dim = n_head_kv * head_dim
    scale = (1.0 / Math.sqrt(head_dim.to_f64)).to_f32
    rng = Random.new(0xBF168008_u64)
    initial_k = Array(Float32).new(packed_len * kv_dim) { ((rng.next_float - 0.5) * 1.0).to_f32 }
    initial_v = Array(Float32).new(packed_len * kv_dim) { ((rng.next_float - 0.5) * 1.0).to_f32 }
    q = Array(Float32).new(q_dim) { ((rng.next_float - 0.5) * 1.0).to_f32 }
    gate = Array(Float32).new(q_dim) { ((rng.next_float - 0.5) * 2.0).to_f32 }
    current_k = Array(Float32).new(kv_dim) { ((rng.next_float - 0.5) * 1.0).to_f32 }
    current_v = Array(Float32).new(kv_dim) { ((rng.next_float - 0.5) * 1.0).to_f32 }
    plan = adaptive.plan(Array.new(
      capacity * n_head_kv,
      ML::GGUF::QwenQBitAdaptiveKV::Tier::BF16,
    ))
    baseline = ML::GGUF::QwenQBitAdaptiveResidentKV.allocate(
      plan, plan, capacity, n_head_kv, head_dim,
    )
    candidate = ML::GGUF::QwenQBitAdaptiveResidentKV.allocate(
      plan, plan, capacity, n_head_kv, head_dim,
    )
    initial_buffers = [
      ML::MetalBuffer.from_array(initial_k),
      ML::MetalBuffer.from_array(initial_v),
    ]
    previous_splitk = ENV["QWEN35_ADAPTIVE_SPLITK"]?
    previous_dequant_t4 = ENV["QWEN35_ADAPTIVE_DEQUANT_T4"]?
    previous_stage2_fused = ENV["QWEN35_ADAPTIVE_SPLITK_STAGE2_FUSED"]?
    previous_bf16_t8 = ENV["QWEN35_ADAPTIVE_BF16_SPLITK_T8"]?
    begin
      [baseline, candidate].each do |cache|
        ML::GGUF::QwenQBitAdaptiveResidentKV.append_from_metal(
          cache, initial_buffers[0], initial_buffers[1], packed_len,
        )
      end
      [baseline, candidate].each do |cache|
        cache.with_live_buffers do |_k_base, _k_metadata, k_sidecar, _v_base, _v_metadata, v_sidecar, _cache_len|
          (k_sidecar.contents.address % 16_u64).should eq(0_u64)
          (v_sidecar.contents.address % 16_u64).should eq(0_u64)
        end
      end
      packed_k, packed_v = ML::GGUF::QwenQBitAdaptiveResidentKV.snapshot(baseline)
      expected = QwenQBitAdaptiveResidentKVSpec.chunk_reference(
        q, gate, adaptive.decode(packed_k), adaptive.decode(packed_v),
        current_k, current_v, packed_len, 1,
        n_head, n_head_kv, head_dim, heads_per_group, scale,
      )

      outputs = [] of Array(Float32)
      [{cache: baseline, bf16_t8: "0"}, {cache: candidate, bf16_t8: "1"}].each do |variant|
        ENV["QWEN35_ADAPTIVE_SPLITK"] = "1"
        ENV["QWEN35_ADAPTIVE_DEQUANT_T4"] = "1"
        ENV["QWEN35_ADAPTIVE_SPLITK_STAGE2_FUSED"] = "1"
        ENV["QWEN35_ADAPTIVE_BF16_SPLITK_T8"] = variant[:bf16_t8]
        buffers = [
          ML::MetalBuffer.from_array(q),
          ML::MetalBuffer.from_array(gate),
          ML::MetalBuffer.from_array(current_k),
          ML::MetalBuffer.from_array(current_v),
          ML::MetalBuffer.new(q_dim.to_i64 * sizeof(Float32)),
        ]
        begin
          ML::GGUF::Qwen35Metal::Scratch.with_namespace("adaptive_bf16_splitk_t8_spec") do
            ML::GGUF::QwenQBitAdaptiveResidentKV.prefill_chunk_and_append_from_metal(
              variant[:cache], buffers[0], buffers[1], buffers[2], buffers[3], buffers[4],
              1, n_head, heads_per_group, scale,
            )
          end
          outputs << buffers[4].read(q_dim)
        ensure
          buffers.each(&.release)
        end
      end

      outputs.each do |actual|
        QwenQBitAdaptiveResidentKVSpec.cosine(expected, actual).should be > 0.9999999
        QwenQBitAdaptiveResidentKVSpec.max_diff(expected, actual).should be < 2.0e-4_f32
      end
      QwenQBitAdaptiveResidentKVSpec.cosine(outputs[0], outputs[1]).should be > 0.9999999
      QwenQBitAdaptiveResidentKVSpec.max_diff(outputs[0], outputs[1]).should be < 2.0e-5_f32
      baseline_k, baseline_v = ML::GGUF::QwenQBitAdaptiveResidentKV.snapshot(baseline)
      candidate_k, candidate_v = ML::GGUF::QwenQBitAdaptiveResidentKV.snapshot(candidate)
      candidate_k.payload.should eq(baseline_k.payload)
      candidate_v.payload.should eq(baseline_v.payload)
    ensure
      if previous_splitk
        ENV["QWEN35_ADAPTIVE_SPLITK"] = previous_splitk
      else
        ENV.delete("QWEN35_ADAPTIVE_SPLITK")
      end
      if previous_dequant_t4
        ENV["QWEN35_ADAPTIVE_DEQUANT_T4"] = previous_dequant_t4
      else
        ENV.delete("QWEN35_ADAPTIVE_DEQUANT_T4")
      end
      if previous_stage2_fused
        ENV["QWEN35_ADAPTIVE_SPLITK_STAGE2_FUSED"] = previous_stage2_fused
      else
        ENV.delete("QWEN35_ADAPTIVE_SPLITK_STAGE2_FUSED")
      end
      if previous_bf16_t8
        ENV["QWEN35_ADAPTIVE_BF16_SPLITK_T8"] = previous_bf16_t8
      else
        ENV.delete("QWEN35_ADAPTIVE_BF16_SPLITK_T8")
      end
      initial_buffers.each(&.release)
      baseline.release
      candidate.release
    end
  end

  it "matches bounded adaptive chunks when a wide span is published once" do
    pending!("Metal not available") unless ML::GGUF::Qwen35Metal.available?

    initial_tokens = 5
    token_count = 65
    capacity = initial_tokens + token_count
    chunks = [32, 33]
    n_head = 6
    n_head_kv = 1
    head_dim = 256
    heads_per_group = 6
    q_dim = n_head * head_dim
    kv_dim = n_head_kv * head_dim
    scale = (1.0 / Math.sqrt(head_dim.to_f64)).to_f32
    rng = Random.new(0xB01D5A4E_u64)
    initial_k = Array(Float32).new(initial_tokens * kv_dim) { ((rng.next_float - 0.5) * 1.0).to_f32 }
    initial_v = Array(Float32).new(initial_tokens * kv_dim) { ((rng.next_float - 0.5) * 1.0).to_f32 }
    q = Array(Float32).new(token_count * q_dim) { ((rng.next_float - 0.5) * 1.0).to_f32 }
    gate = Array(Float32).new(token_count * q_dim) { ((rng.next_float - 0.5) * 2.0).to_f32 }
    k = Array(Float32).new(token_count * kv_dim) { ((rng.next_float - 0.5) * 1.0).to_f32 }
    v = Array(Float32).new(token_count * kv_dim) { ((rng.next_float - 0.5) * 1.0).to_f32 }
    plan = adaptive.plan(Array.new(capacity, ML::GGUF::QwenQBitAdaptiveKV::Tier::P4))

    live_before = ML::MetalBuffer.stats[:live_bytes]
    reference = ML::GGUF::QwenQBitAdaptiveResidentKV.allocate(
      plan, plan, capacity, n_head_kv, head_dim,
    )
    wide = ML::GGUF::QwenQBitAdaptiveResidentKV.allocate(
      plan, plan, capacity, n_head_kv, head_dim,
    )
    reference_output = [] of Float32
    offset = 0
    previous_dequant_t4 = ENV["QWEN35_ADAPTIVE_DEQUANT_T4"]?
    begin
      initial_buffers = [
        ML::MetalBuffer.from_array(initial_k),
        ML::MetalBuffer.from_array(initial_v),
      ]
      begin
        [reference, wide].each do |resident|
          ML::GGUF::QwenQBitAdaptiveResidentKV.append_from_metal(
            resident, initial_buffers[0], initial_buffers[1], initial_tokens,
          )
        end
      ensure
        initial_buffers.each(&.release)
      end

      ENV["QWEN35_ADAPTIVE_DEQUANT_T4"] = "0"
      chunks.each do |chunk_size|
        chunk_buffers = [
          ML::MetalBuffer.from_array(q[offset * q_dim, chunk_size * q_dim]),
          ML::MetalBuffer.from_array(gate[offset * q_dim, chunk_size * q_dim]),
          ML::MetalBuffer.from_array(k[offset * kv_dim, chunk_size * kv_dim]),
          ML::MetalBuffer.from_array(v[offset * kv_dim, chunk_size * kv_dim]),
          ML::MetalBuffer.new(chunk_size.to_i64 * q_dim * sizeof(Float32)),
        ]
        begin
          ML::GGUF::QwenQBitAdaptiveResidentKV.prefill_chunk_and_append_from_metal(
            reference,
            chunk_buffers[0], chunk_buffers[1], chunk_buffers[2], chunk_buffers[3], chunk_buffers[4],
            chunk_size, n_head, heads_per_group, scale,
          )
          reference_output.concat(chunk_buffers[4].read(chunk_size * q_dim))
        ensure
          chunk_buffers.each(&.release)
        end
        offset += chunk_size
      end

      ENV["QWEN35_ADAPTIVE_DEQUANT_T4"] = "1"
      wide_buffers = [
        ML::MetalBuffer.from_array(q),
        ML::MetalBuffer.from_array(gate),
        ML::MetalBuffer.from_array(k),
        ML::MetalBuffer.from_array(v),
        ML::MetalBuffer.new(token_count.to_i64 * q_dim * sizeof(Float32)),
      ]
      begin
        command = ML::Metal::CommandBuffer.new
        ML::GGUF::QwenQBitAdaptiveResidentKV.encode_prefill_chunk_and_append(
          command, wide,
          wide_buffers[0], wide_buffers[1], wide_buffers[2], wide_buffers[3], wide_buffers[4],
          token_count, n_head, heads_per_group, scale,
          expected_start_token: initial_tokens,
        )
        wide.cache_len.should eq(initial_tokens)
        ML::GGUF::QwenQBitAdaptiveResidentKV.finalize_pending_append(command, wide)
        command.commit_and_wait
        ML::GGUF::QwenQBitAdaptiveResidentKV.finish_pending_append!(wide, command)
        wide.cache_len.should eq(capacity)

        wide_output = wide_buffers[4].read(token_count * q_dim)
        QwenQBitAdaptiveResidentKVSpec.cosine(reference_output, wide_output).should be > 0.9999999
        QwenQBitAdaptiveResidentKVSpec.max_diff(reference_output, wide_output).should be < 2.0e-4_f32
        reference_k, reference_v = ML::GGUF::QwenQBitAdaptiveResidentKV.snapshot(reference)
        wide_k, wide_v = ML::GGUF::QwenQBitAdaptiveResidentKV.snapshot(wide)
        wide_k.payload.should eq(reference_k.payload)
        wide_v.payload.should eq(reference_v.payload)
      ensure
        if command
          ML::GGUF::QwenQBitAdaptiveResidentKV.cancel_pending_append!(wide, command)
        end
        wide_buffers.each(&.release)
      end
    ensure
      if previous_dequant_t4
        ENV["QWEN35_ADAPTIVE_DEQUANT_T4"] = previous_dequant_t4
      else
        ENV.delete("QWEN35_ADAPTIVE_DEQUANT_T4")
      end
      reference.release
      wide.release
    end
    ML::MetalBuffer.stats[:live_bytes].should eq(live_before)
  end

  it "keeps a failed non-finite device append invisible and permits a clean retry" do
    pending!("Metal not available") unless ML::GGUF::Qwen35Metal.available?

    values = Array(Float32).new(256, 0.25_f32)
    invalid = values.dup
    invalid[17] = Float32::NAN
    plan = adaptive.plan([ML::GGUF::QwenQBitAdaptiveKV::Tier::P4])
    k_source = ML::MetalBuffer.from_array(invalid)
    v_source = ML::MetalBuffer.from_array(values)
    resident = ML::GGUF::QwenQBitAdaptiveResidentKV.allocate(plan, plan, 1, 1, 256)
    begin
      empty_q = Array(Float32).new(6 * 256, 0.0_f32)
      expect_raises(ArgumentError, /empty/) do
        ML::GGUF::QwenQBitAdaptiveResidentKV.attn_decode(
          empty_q, empty_q, resident, 6, 6, 1.0_f32,
        )
      end

      expect_raises(ArgumentError, /failed closed/) do
        ML::GGUF::QwenQBitAdaptiveResidentKV.append_from_metal(
          resident, k_source, v_source, 1,
        )
      end
      resident.cache_len.should eq(0)

      k_source.write(values)
      ML::GGUF::QwenQBitAdaptiveResidentKV.append_from_metal(
        resident, k_source, v_source, 1,
      )
      resident.cache_len.should eq(1)
      packed_k, packed_v = ML::GGUF::QwenQBitAdaptiveResidentKV.snapshot(resident)
      adaptive.validate(packed_k)
      adaptive.validate(packed_v)
    ensure
      resident.release
      k_source.release
      v_source.release
    end
  end

  it "keeps a failed mixed prefill invisible and permits a clean retry" do
    pending!("Metal not available") unless ML::GGUF::Qwen35Metal.available?

    token_count = 65
    head_dim = 256
    n_head = 6
    q_values = token_count * n_head * head_dim
    valid_q = Array(Float32).new(q_values, 0.125_f32)
    invalid_q = valid_q.dup
    invalid_q[40 * n_head * head_dim + 41] = Float32::NAN
    gate = Array(Float32).new(q_values, 0.0_f32)
    kv = Array(Float32).new(token_count * head_dim, 0.25_f32)
    plan = adaptive.plan(Array.new(token_count, ML::GGUF::QwenQBitAdaptiveKV::Tier::P4))
    buffers = [
      ML::MetalBuffer.from_array(invalid_q),
      ML::MetalBuffer.from_array(gate),
      ML::MetalBuffer.from_array(kv),
      ML::MetalBuffer.from_array(kv),
      ML::MetalBuffer.new(q_values.to_i64 * sizeof(Float32)),
    ]
    resident = ML::GGUF::QwenQBitAdaptiveResidentKV.allocate(
      plan, plan, token_count, 1, head_dim,
    )
    begin
      expect_raises(ArgumentError, /prefill\/pack failed closed/) do
        ML::GGUF::QwenQBitAdaptiveResidentKV.prefill_chunk_and_append_from_metal(
          resident, buffers[0], buffers[1], buffers[2], buffers[3], buffers[4],
          token_count, n_head, 6, 1.0_f32,
        )
      end
      resident.cache_len.should eq(0)

      buffers[0].write(valid_q)
      ML::GGUF::QwenQBitAdaptiveResidentKV.prefill_chunk_and_append_from_metal(
        resident, buffers[0], buffers[1], buffers[2], buffers[3], buffers[4],
        token_count, n_head, 6, 1.0_f32,
      )
      resident.cache_len.should eq(token_count)
    ensure
      resident.release
      buffers.each(&.release)
    end
  end

  it "publishes an externally encoded append only after the shared command completes" do
    pending!("Metal not available") unless ML::GGUF::Qwen35Metal.available?

    head_dim = 256
    n_head = 6
    q_values = n_head * head_dim
    q = Array(Float32).new(q_values, 0.125_f32)
    gate = Array(Float32).new(q_values, 0.0_f32)
    kv = Array(Float32).new(head_dim, 0.25_f32)
    plan = adaptive.plan([ML::GGUF::QwenQBitAdaptiveKV::Tier::P4])
    buffers = [
      ML::MetalBuffer.from_array(q),
      ML::MetalBuffer.from_array(gate),
      ML::MetalBuffer.from_array(kv),
      ML::MetalBuffer.from_array(kv),
      ML::MetalBuffer.new(q_values.to_i64 * sizeof(Float32)),
    ]
    resident = ML::GGUF::QwenQBitAdaptiveResidentKV.allocate(
      plan, plan, 1, 1, head_dim,
    )
    begin
      command = ML::Metal::CommandBuffer.new
      ML::GGUF::QwenQBitAdaptiveResidentKV.encode_prefill_chunk_and_append(
        command, resident,
        buffers[0], buffers[1], buffers[2], buffers[3], buffers[4],
        1, n_head, 6, 1.0_f32,
        expected_start_token: 0,
      )

      resident.cache_len.should eq(0)
      expect_raises(ArgumentError, /complete successfully/) do
        ML::GGUF::QwenQBitAdaptiveResidentKV.finish_pending_append!(resident, command)
      end
      expect_raises(ArgumentError, /pending/) do
        ML::GGUF::QwenQBitAdaptiveResidentKV.snapshot(resident)
      end
      expect_raises(ArgumentError, /pending/) { resident.release }

      ML::GGUF::QwenQBitAdaptiveResidentKV.finalize_pending_append(
        command, resident,
      )
      command.commit_and_wait
      ML::GGUF::QwenQBitAdaptiveResidentKV.finish_pending_append!(resident, command)
      resident.cache_len.should eq(1)
      packed_k, packed_v = ML::GGUF::QwenQBitAdaptiveResidentKV.snapshot(resident)
      adaptive.validate(packed_k)
      adaptive.validate(packed_v)
    ensure
      if command
        ML::GGUF::QwenQBitAdaptiveResidentKV.cancel_pending_append!(resident, command)
      end
      resident.release
      buffers.each(&.release)
    end
  end

  it "reserves two adjacent shared-command appends and publishes them FIFO" do
    pending!("Metal not available") unless ML::GGUF::Qwen35Metal.available?

    head_dim = 256
    n_head = 6
    q_values = n_head * head_dim
    plan = adaptive.plan(Array.new(2, ML::GGUF::QwenQBitAdaptiveKV::Tier::P4))
    inputs = Array.new(2) do |index|
      value = 0.125_f32 + index.to_f32 * 0.125_f32
      [
        ML::MetalBuffer.from_array(Array(Float32).new(q_values, value)),
        ML::MetalBuffer.from_array(Array(Float32).new(q_values, 0.0_f32)),
        ML::MetalBuffer.from_array(Array(Float32).new(head_dim, value)),
        ML::MetalBuffer.from_array(Array(Float32).new(head_dim, value)),
        ML::MetalBuffer.new(q_values.to_i64 * sizeof(Float32)),
      ]
    end
    resident = ML::GGUF::QwenQBitAdaptiveResidentKV.allocate(plan, plan, 2, 1, head_dim)
    command_queue = ML::Metal::CommandQueue.new
    foreign_queue = ML::Metal::CommandQueue.new
    command_a = ML::Metal::CommandBuffer.new(queue: command_queue)
    command_b = ML::Metal::CommandBuffer.new(queue: command_queue)
    foreign_command = ML::Metal::CommandBuffer.new(queue: foreign_queue)
    begin
      {command_a, command_b}.each_with_index do |command, index|
        buffers = inputs[index]
        ML::GGUF::QwenQBitAdaptiveResidentKV.encode_prefill_chunk_and_append(
          command, resident,
          buffers[0], buffers[1], buffers[2], buffers[3], buffers[4],
          1, n_head, 6, 1.0_f32,
          expected_start_token: index,
        )
        ML::GGUF::QwenQBitAdaptiveResidentKV.finalize_pending_append(command, resident)
        if index == 0
          expect_raises(ArgumentError, /does not own/) do
            ML::GGUF::QwenQBitAdaptiveResidentKV.cancel_pending_append!(resident, foreign_command)
          end
          foreign_buffers = inputs[1]
          expect_raises(ArgumentError, /Metal queue/) do
            ML::GGUF::QwenQBitAdaptiveResidentKV.encode_prefill_chunk_and_append(
              foreign_command, resident,
              foreign_buffers[0], foreign_buffers[1], foreign_buffers[2], foreign_buffers[3], foreign_buffers[4],
              1, n_head, 6, 1.0_f32,
              expected_start_token: 1,
            )
          end
        end
      end

      resident.cache_len.should eq(0)
      expect_raises(ArgumentError, /pending/) do
        ML::GGUF::QwenQBitAdaptiveResidentKV.snapshot(resident)
      end

      command_a.commit
      command_b.commit
      command_b.wait
      expect_raises(ArgumentError, /FIFO/) do
        ML::GGUF::QwenQBitAdaptiveResidentKV.finish_pending_append!(resident, command_b)
      end
      resident.cache_len.should eq(0)

      command_a.wait
      expect_raises(ArgumentError, /duplicate cache/) do
        ML::GGUF::QwenQBitAdaptiveResidentKV.finish_pending_appends!([resident, resident], command_a)
      end
      resident.cache_len.should eq(0)
      ML::GGUF::QwenQBitAdaptiveResidentKV.finish_pending_append!(resident, command_a)
      resident.cache_len.should eq(1)
      ML::GGUF::QwenQBitAdaptiveResidentKV.finish_pending_append!(resident, command_b)
      resident.cache_len.should eq(2)
    ensure
      {command_b, foreign_command, command_a}.each do |command|
        begin
          ML::GGUF::QwenQBitAdaptiveResidentKV.cancel_pending_append!(resident, command)
        rescue
        end
        begin
          command.discard unless command.committed?
        rescue
        end
      end
      resident.release
      inputs.each { |buffers| buffers.each(&.release) }
    end
  end

  it "fails a two-cache two-flight group atomically and discards both reserved suffixes" do
    pending!("Metal not available") unless ML::GGUF::Qwen35Metal.available?

    head_dim = 256
    n_head = 6
    q_values = n_head * head_dim
    plan = adaptive.plan(Array.new(2, ML::GGUF::QwenQBitAdaptiveKV::Tier::P4))
    valid_q = Array(Float32).new(q_values, 0.125_f32)
    invalid_q = valid_q.dup
    invalid_q[41] = Float32::NAN
    live_before = ML::MetalBuffer.stats[:live_bytes]
    2.times do |failed_cache_index|
      inputs = Array.new(2) do |flight_index|
        Array.new(2) do |cache_index|
          q = flight_index == 0 && cache_index == failed_cache_index ? invalid_q : valid_q
          [
            ML::MetalBuffer.from_array(q),
            ML::MetalBuffer.from_array(Array(Float32).new(q_values, 0.0_f32)),
            ML::MetalBuffer.from_array(Array(Float32).new(head_dim, 0.25_f32)),
            ML::MetalBuffer.from_array(Array(Float32).new(head_dim, 0.25_f32)),
            ML::MetalBuffer.new(q_values.to_i64 * sizeof(Float32)),
          ]
        end
      end
      residents = Array.new(2) do
        ML::GGUF::QwenQBitAdaptiveResidentKV.allocate(plan, plan, 2, 1, head_dim)
      end
      command_queue = ML::Metal::CommandQueue.new
      command_a = ML::Metal::CommandBuffer.new(queue: command_queue)
      command_b = ML::Metal::CommandBuffer.new(queue: command_queue)
      begin
        {command_a, command_b}.each_with_index do |command, index|
          residents.each_with_index do |resident, cache_index|
            buffers = inputs[index][cache_index]
            ML::GGUF::QwenQBitAdaptiveResidentKV.encode_prefill_chunk_and_append(
              command, resident,
              buffers[0], buffers[1], buffers[2], buffers[3], buffers[4],
              1, n_head, 6, 1.0_f32,
              expected_start_token: index,
            )
            ML::GGUF::QwenQBitAdaptiveResidentKV.finalize_pending_append(command, resident)
          end
        end

        command_a.commit
        command_b.commit
        expect_raises(ArgumentError, /in-flight GPU work/) do
          ML::GGUF::QwenQBitAdaptiveResidentKV.discard_pending_appends!(residents[0])
        end
        command_a.wait
        command_b.wait

        expect_raises(ArgumentError, /failed closed/) do
          ML::GGUF::QwenQBitAdaptiveResidentKV.finish_pending_appends!(residents, command_a)
        end
        expect_raises(ArgumentError, /FIFO/) do
          ML::GGUF::QwenQBitAdaptiveResidentKV.finish_pending_appends!(residents, command_b)
        end
        residents.each { |resident| resident.cache_len.should eq(0) }

        residents.each do |resident|
          ML::GGUF::QwenQBitAdaptiveResidentKV.discard_pending_appends!(resident)
          packed_k, packed_v = ML::GGUF::QwenQBitAdaptiveResidentKV.snapshot(resident)
          packed_k.value_count.should eq(0)
          packed_v.value_count.should eq(0)
        end
      ensure
        {command_b, command_a}.each do |command|
          residents.reverse_each do |resident|
            begin
              ML::GGUF::QwenQBitAdaptiveResidentKV.cancel_pending_append!(resident, command)
            rescue
            end
          end
          begin
            command.discard unless command.committed?
          rescue
          end
        end
        residents.each(&.release)
        inputs.each do |flight|
          flight.each { |buffers| buffers.each(&.release) }
        end
      end
      ML::MetalBuffer.stats[:live_bytes].should eq(live_before)
    end
  end

  it "requires a true command-tail marker before publishing an external append" do
    pending!("Metal not available") unless ML::GGUF::Qwen35Metal.available?

    head_dim = 256
    n_head = 6
    q_values = n_head * head_dim
    plan = adaptive.plan([ML::GGUF::QwenQBitAdaptiveKV::Tier::P4])
    buffers = [
      ML::MetalBuffer.from_array(Array(Float32).new(q_values, 0.125_f32)),
      ML::MetalBuffer.from_array(Array(Float32).new(q_values, 0.0_f32)),
      ML::MetalBuffer.from_array(Array(Float32).new(head_dim, 0.25_f32)),
      ML::MetalBuffer.from_array(Array(Float32).new(head_dim, 0.25_f32)),
      ML::MetalBuffer.new(q_values.to_i64 * sizeof(Float32)),
    ]
    resident = ML::GGUF::QwenQBitAdaptiveResidentKV.allocate(plan, plan, 1, 1, head_dim)
    begin
      command = ML::Metal::CommandBuffer.new
      ML::GGUF::QwenQBitAdaptiveResidentKV.encode_prefill_chunk_and_append(
        command, resident,
        buffers[0], buffers[1], buffers[2], buffers[3], buffers[4],
        1, n_head, 6, 1.0_f32,
        expected_start_token: 0,
      )

      command.commit_and_wait
      expect_raises(ArgumentError, /failed closed/) do
        ML::GGUF::QwenQBitAdaptiveResidentKV.finish_pending_append!(resident, command)
      end
      resident.cache_len.should eq(0)
    ensure
      if command
        ML::GGUF::QwenQBitAdaptiveResidentKV.cancel_pending_append!(resident, command)
      end
      resident.release
      buffers.each(&.release)
    end
  end

  it "keeps every cache unpublished when one shared-command append fails" do
    pending!("Metal not available") unless ML::GGUF::Qwen35Metal.available?

    head_dim = 256
    n_head = 6
    q_values = n_head * head_dim
    plan = adaptive.plan([ML::GGUF::QwenQBitAdaptiveKV::Tier::P4])
    valid_q = Array(Float32).new(q_values, 0.125_f32)
    invalid_q = valid_q.dup
    invalid_q[41] = Float32::NAN
    gate = Array(Float32).new(q_values, 0.0_f32)
    kv = Array(Float32).new(head_dim, 0.25_f32)
    buffers = [valid_q, invalid_q].map do |q|
      [
        ML::MetalBuffer.from_array(q),
        ML::MetalBuffer.from_array(gate),
        ML::MetalBuffer.from_array(kv),
        ML::MetalBuffer.from_array(kv),
        ML::MetalBuffer.new(q_values.to_i64 * sizeof(Float32)),
      ]
    end
    residents = Array.new(2) do
      ML::GGUF::QwenQBitAdaptiveResidentKV.allocate(plan, plan, 1, 1, head_dim)
    end
    begin
      command = ML::Metal::CommandBuffer.new
      residents.each_with_index do |resident, i|
        layer_buffers = buffers[i]
        ML::GGUF::QwenQBitAdaptiveResidentKV.encode_prefill_chunk_and_append(
          command, resident,
          layer_buffers[0], layer_buffers[1], layer_buffers[2], layer_buffers[3], layer_buffers[4],
          1, n_head, 6, 1.0_f32,
          expected_start_token: 0,
        )
      end
      residents.each do |resident|
        ML::GGUF::QwenQBitAdaptiveResidentKV.finalize_pending_append(command, resident)
      end
      command.commit_and_wait

      expect_raises(ArgumentError, /failed closed/) do
        ML::GGUF::QwenQBitAdaptiveResidentKV.finish_pending_appends!(residents, command)
      end
      residents.each { |resident| resident.cache_len.should eq(0) }
    ensure
      if command
        residents.each do |resident|
          ML::GGUF::QwenQBitAdaptiveResidentKV.cancel_pending_append!(resident, command)
        end
      end
      residents.each(&.release)
      buffers.each { |layer_buffers| layer_buffers.each(&.release) }
    end
  end

  it "keeps a failed externally encoded append unpublished and permits retry" do
    pending!("Metal not available") unless ML::GGUF::Qwen35Metal.available?

    head_dim = 256
    n_head = 6
    q_values = n_head * head_dim
    valid_q = Array(Float32).new(q_values, 0.125_f32)
    invalid_q = valid_q.dup
    invalid_q[41] = Float32::NAN
    gate = Array(Float32).new(q_values, 0.0_f32)
    kv = Array(Float32).new(head_dim, 0.25_f32)
    plan = adaptive.plan([ML::GGUF::QwenQBitAdaptiveKV::Tier::P4])
    buffers = [
      ML::MetalBuffer.from_array(invalid_q),
      ML::MetalBuffer.from_array(gate),
      ML::MetalBuffer.from_array(kv),
      ML::MetalBuffer.from_array(kv),
      ML::MetalBuffer.new(q_values.to_i64 * sizeof(Float32)),
    ]
    resident = ML::GGUF::QwenQBitAdaptiveResidentKV.allocate(
      plan, plan, 1, 1, head_dim,
    )
    begin
      failed_command = ML::Metal::CommandBuffer.new
      ML::GGUF::QwenQBitAdaptiveResidentKV.encode_prefill_chunk_and_append(
        failed_command, resident,
        buffers[0], buffers[1], buffers[2], buffers[3], buffers[4],
        1, n_head, 6, 1.0_f32,
        expected_start_token: 0,
      )
      ML::GGUF::QwenQBitAdaptiveResidentKV.finalize_pending_append(
        failed_command, resident,
      )
      failed_command.commit_and_wait
      expect_raises(ArgumentError, /failed closed/) do
        ML::GGUF::QwenQBitAdaptiveResidentKV.finish_pending_append!(resident, failed_command)
      end
      resident.cache_len.should eq(0)

      buffers[0].write(valid_q)
      retry_command = ML::Metal::CommandBuffer.new
      ML::GGUF::QwenQBitAdaptiveResidentKV.encode_prefill_chunk_and_append(
        retry_command, resident,
        buffers[0], buffers[1], buffers[2], buffers[3], buffers[4],
        1, n_head, 6, 1.0_f32,
        expected_start_token: 0,
      )
      ML::GGUF::QwenQBitAdaptiveResidentKV.finalize_pending_append(
        retry_command, resident,
      )
      retry_command.commit
      expect_raises(ArgumentError, /not completed/) do
        ML::GGUF::QwenQBitAdaptiveResidentKV.cancel_pending_append!(resident, retry_command)
      end
      retry_command.wait
      ML::GGUF::QwenQBitAdaptiveResidentKV.finish_pending_append!(resident, retry_command)
      resident.cache_len.should eq(1)
    ensure
      if retry_command
        ML::GGUF::QwenQBitAdaptiveResidentKV.cancel_pending_append!(resident, retry_command)
      end
      resident.release
      buffers.each(&.release)
    end
  end

  it "rejects adaptive layouts that do not match the declared KV shape" do
    values = Array(Float32).new(2 * 256, 0.0_f32)
    encoded = adaptive.encode(values, [
      ML::GGUF::QwenQBitAdaptiveKV::Tier::P4,
      ML::GGUF::QwenQBitAdaptiveKV::Tier::P5,
    ], block_size: 256)

    expect_raises(ArgumentError, /value count/) do
      ML::GGUF::QwenQBitAdaptiveResidentKV.prepare(encoded, encoded, 1, 1, 256)
    end

    plan = adaptive.plan([
      ML::GGUF::QwenQBitAdaptiveKV::Tier::P4,
      ML::GGUF::QwenQBitAdaptiveKV::Tier::P5,
    ])
    expect_raises(ArgumentError, /plan row count/) do
      ML::GGUF::QwenQBitAdaptiveResidentKV.allocate(plan, plan, 1, 1, 256)
    end
    expect_raises(ArgumentError, /maximum sequence/) do
      ML::GGUF::QwenQBitAdaptiveResidentKV.allocate(plan, plan, 0, 1, 256)
    end
  end

  it "restores a compact snapshot atomically into the matching planned owner" do
    pending!("Metal not available") unless ML::GGUF::Qwen35Metal.available?

    max_seq = 4
    cache_len = 2
    head_dim = 256
    live_tiers = [
      ML::GGUF::QwenQBitAdaptiveKV::Tier::P4,
      ML::GGUF::QwenQBitAdaptiveKV::Tier::P5,
    ]
    capacity_tiers = [
      ML::GGUF::QwenQBitAdaptiveKV::Tier::P4,
      ML::GGUF::QwenQBitAdaptiveKV::Tier::P5,
      ML::GGUF::QwenQBitAdaptiveKV::Tier::P4,
      ML::GGUF::QwenQBitAdaptiveKV::Tier::P5,
    ]
    values = Array(Float32).new(cache_len * head_dim) { |i| (i - 255).to_f32 / 89.0_f32 }
    k = adaptive.encode(values, live_tiers, block_size: head_dim)
    v = adaptive.encode(values.reverse, live_tiers, block_size: head_dim)
    plan = adaptive.plan(capacity_tiers)
    resident = ML::GGUF::QwenQBitAdaptiveResidentKV.allocate(
      plan, plan, max_seq, 1, head_dim,
    )
    wrong_plan = adaptive.plan(Array(ML::GGUF::QwenQBitAdaptiveKV::Tier).new(
      max_seq,
      ML::GGUF::QwenQBitAdaptiveKV::Tier::P5,
    ))
    wrong = ML::GGUF::QwenQBitAdaptiveResidentKV.allocate(
      wrong_plan, wrong_plan, max_seq, 1, head_dim,
    )
    begin
      ML::GGUF::QwenQBitAdaptiveResidentKV.restore_snapshot!(resident, k, v, cache_len)
      resident.cache_len.should eq(cache_len)
      restored_k, restored_v = ML::GGUF::QwenQBitAdaptiveResidentKV.snapshot(resident)
      restored_k.payload.should eq(k.payload)
      restored_v.payload.should eq(v.payload)

      expect_raises(ArgumentError, /tier plan mismatch/) do
        ML::GGUF::QwenQBitAdaptiveResidentKV.restore_snapshot!(wrong, k, v, cache_len)
      end
      wrong.cache_len.should eq(0)
    ensure
      resident.release
      wrong.release
    end
  end

  it "makes release idempotent and rejects use after release" do
    pending!("Metal not available") unless ML::GGUF::Qwen35Metal.available?

    values = Array(Float32).new(256, 0.0_f32)
    encoded = adaptive.encode(values, [ML::GGUF::QwenQBitAdaptiveKV::Tier::P4], block_size: 256)
    live_before = ML::MetalBuffer.stats[:live_bytes]
    resident = ML::GGUF::QwenQBitAdaptiveResidentKV.prepare(encoded, encoded, 1, 1, 256)
    resident.release
    resident.release
    ML::MetalBuffer.stats[:live_bytes].should eq(live_before)

    q = Array(Float32).new(6 * 256, 0.0_f32)
    expect_raises(ArgumentError, /released/) do
      ML::GGUF::QwenQBitAdaptiveResidentKV.attn_decode(q, q, resident, 6, 6, 1.0_f32)
    end
  end
end
