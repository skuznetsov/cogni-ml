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
