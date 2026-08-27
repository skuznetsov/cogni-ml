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
      actual = ML::GGUF::QwenQBitAdaptiveResidentKV.attn_decode(
        q, gate, resident, n_head, heads_per_group, scale,
      )
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

      ML::GGUF::QwenQBitAdaptiveResidentKV.append_from_metal(
        resident, k_source, v_source, cache_len - first_chunk,
        source_token_offset: first_chunk,
      )
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
    resident = ML::GGUF::QwenQBitAdaptiveResidentKV.allocate(
      plan, plan, capacity, n_head_kv, head_dim,
    )
    begin
      initial_buffers = [
        ML::MetalBuffer.from_array(initial_k),
        ML::MetalBuffer.from_array(initial_v),
      ]
      begin
        ML::GGUF::QwenQBitAdaptiveResidentKV.append_from_metal(
          resident, initial_buffers[0], initial_buffers[1], initial_tokens,
        )
      ensure
        initial_buffers.each(&.release)
      end

      packed_k, packed_v = ML::GGUF::QwenQBitAdaptiveResidentKV.snapshot(resident)
      expected = QwenQBitAdaptiveResidentKVSpec.chunk_reference(
        q, gate, adaptive.decode(packed_k), adaptive.decode(packed_v),
        current_k, current_v, initial_tokens, token_count,
        n_head, n_head_kv, head_dim, heads_per_group, scale,
      )
      buffers = [
        ML::MetalBuffer.from_array(q),
        ML::MetalBuffer.from_array(gate),
        ML::MetalBuffer.from_array(current_k),
        ML::MetalBuffer.from_array(current_v),
        ML::MetalBuffer.new(expected.size.to_i64 * sizeof(Float32)),
      ]
      begin
        ML::GGUF::QwenQBitAdaptiveResidentKV.prefill_chunk_and_append_from_metal(
          resident,
          buffers[0], buffers[1], buffers[2], buffers[3], buffers[4],
          token_count, n_head, heads_per_group, scale,
        )
        actual = buffers[4].read(expected.size.to_i32)
        QwenQBitAdaptiveResidentKVSpec.cosine(expected, actual).should be > 0.9999999
        QwenQBitAdaptiveResidentKVSpec.max_diff(expected, actual).should be < 2.0e-4_f32
      ensure
        buffers.each(&.release)
      end
    ensure
      resident.release
    end
    ML::MetalBuffer.stats[:live_bytes].should eq(live_before)
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
