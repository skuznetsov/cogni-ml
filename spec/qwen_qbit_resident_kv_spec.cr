require "./spec_helper"
require "../src/ml/gguf/qwen35_cpu"
require "../src/ml/gguf/qwen_qbit_gaussian_codec"
require "../src/ml/gguf/qwen_qbit_resident_kv"

module QwenQBitResidentKVSpec
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

describe ML::GGUF::QwenQBitResidentKV do
  codec = ML::GGUF::QwenQBitGaussianCodec

  it "decodes resident p4 and p5 KV inside GQA6 Metal attention" do
    pending!("Metal not available") unless ML::GGUF::Qwen35Metal.available?

    cache_len = 19
    n_head = 24
    n_head_kv = 4
    head_dim = 256
    heads_per_group = 6
    scale = (1.0 / Math.sqrt(head_dim.to_f64)).to_f32
    q_dim = n_head * head_dim
    kv_count = cache_len * n_head_kv * head_dim
    rng = Random.new(0x51B17_u64)
    q = Array(Float32).new(q_dim) { ((rng.next_float - 0.5) * 1.0).to_f32 }
    gate = Array(Float32).new(q_dim) { ((rng.next_float - 0.5) * 2.0).to_f32 }
    k = Array(Float32).new(kv_count) { ((rng.next_float - 0.5) * 1.0).to_f32 }
    v = Array(Float32).new(kv_count) { ((rng.next_float - 0.5) * 1.0).to_f32 }

    [4, 5].each do |precision|
      encoded_k = codec.encode(k, block_size: head_dim, precision: precision)
      encoded_v = codec.encode(v, block_size: head_dim, precision: precision)
      expected = QwenQBitResidentKVSpec.reference(
        q, gate, codec.decode(encoded_k), codec.decode(encoded_v),
        cache_len, n_head, n_head_kv, head_dim, heads_per_group, scale,
      )

      live_before = ML::MetalBuffer.stats[:live_bytes]
      resident = ML::GGUF::QwenQBitResidentKV.prepare(
        encoded_k, encoded_v, cache_len, n_head_kv, head_dim,
      )
      resident.compressed_bytes.should eq((encoded_k.payload.size + encoded_v.payload.size).to_i64)
      ML::MetalBuffer.stats[:live_bytes].should eq(live_before + resident.compressed_bytes)
      begin
        actual = ML::GGUF::QwenQBitResidentKV.attn_decode(
          q, gate, resident, n_head, heads_per_group, scale,
        )
      ensure
        resident.release
      end
      ML::MetalBuffer.stats[:live_bytes].should eq(live_before)

      cos = QwenQBitResidentKVSpec.cosine(expected, actual)
      diff = QwenQBitResidentKVSpec.max_diff(expected, actual)
      printf "  [resident_qbit p%d] cos=%.12f max|delta|=%g ratio=%.3fx\n",
        precision, cos, diff, (2 * kv_count * sizeof(Float32)).to_f64 /
                              (encoded_k.payload.size + encoded_v.payload.size)
      cos.should be > 0.9999999
      diff.should be < 2.0e-4_f32
    end
  end

  it "makes the parity oracle react to a seeded plane-bit mutation" do
    pending!("Metal not available") unless ML::GGUF::Qwen35Metal.available?

    cache_len = 8
    n_head = 6
    n_head_kv = 1
    head_dim = 256
    heads_per_group = 6
    scale = (1.0 / Math.sqrt(head_dim.to_f64)).to_f32
    q_dim = n_head * head_dim
    kv_count = cache_len * n_head_kv * head_dim
    rng = Random.new(0xC0220F7_u64)
    q = Array(Float32).new(q_dim) { ((rng.next_float - 0.5) * 1.0).to_f32 }
    gate = Array(Float32).new(q_dim) { ((rng.next_float - 0.5) * 2.0).to_f32 }
    k = Array(Float32).new(kv_count) { ((rng.next_float - 0.5) * 1.0).to_f32 }
    v = Array(Float32).new(kv_count) { ((rng.next_float - 0.5) * 1.0).to_f32 }
    encoded_k = codec.encode(k, block_size: head_dim, precision: 4)
    encoded_v = codec.encode(v, block_size: head_dim, precision: 4)
    original = QwenQBitResidentKVSpec.reference(
      q, gate, codec.decode(encoded_k), codec.decode(encoded_v),
      cache_len, n_head, n_head_kv, head_dim, heads_per_group, scale,
    )

    mutated_payload = encoded_v.payload.dup
    first_plane_last_byte = ML::GGUF::QwenQBitGaussianCodec::BLOCK_HEADER_BYTES + head_dim // 8 - 1
    mutated_payload[first_plane_last_byte] ^= 0x01_u8
    mutated_v = ML::GGUF::QwenQBitGaussianCodec::Encoded.new(
      encoded_v.value_count, encoded_v.block_size, encoded_v.precision, mutated_payload,
    )
    expected_mutated = QwenQBitResidentKVSpec.reference(
      q, gate, codec.decode(encoded_k), codec.decode(mutated_v),
      cache_len, n_head, n_head_kv, head_dim, heads_per_group, scale,
    )

    resident = ML::GGUF::QwenQBitResidentKV.prepare(
      encoded_k, mutated_v, cache_len, n_head_kv, head_dim,
    )
    begin
      actual_mutated = ML::GGUF::QwenQBitResidentKV.attn_decode(
        q, gate, resident, n_head, heads_per_group, scale,
      )
    ensure
      resident.release
    end

    QwenQBitResidentKVSpec.max_diff(expected_mutated, actual_mutated).should be < 2.0e-4_f32
    QwenQBitResidentKVSpec.max_diff(original, actual_mutated).should be > 1.0e-5_f32
  end

  it "rejects layouts that could cross semantic KV rows" do
    values = Array(Float32).new(512, 0.0_f32)
    wrong_block = codec.encode(values, block_size: 128, precision: 4)
    good = codec.encode(values, block_size: 256, precision: 4)

    expect_raises(ArgumentError, /block size/) do
      ML::GGUF::QwenQBitResidentKV.prepare(wrong_block, good, 2, 1, 256)
    end
  end

  it "rejects shapes outside the bounded Qwen3.8 specialization" do
    pending!("Metal not available") unless ML::GGUF::Qwen35Metal.available?

    head_dim = 256
    values = Array(Float32).new(head_dim, 0.0_f32)
    encoded = codec.encode(values, block_size: head_dim, precision: 4)
    resident = ML::GGUF::QwenQBitResidentKV.prepare(encoded, encoded, 1, 1, head_dim)
    begin
      q = Array(Float32).new(5 * head_dim, 0.0_f32)
      expect_raises(ArgumentError, /GQA6/) do
        ML::GGUF::QwenQBitResidentKV.attn_decode(q, q, resident, 5, 5, 1.0_f32)
      end
    ensure
      resident.release
    end
  end

  it "makes release idempotent and rejects use after release" do
    pending!("Metal not available") unless ML::GGUF::Qwen35Metal.available?

    head_dim = 256
    values = Array(Float32).new(head_dim, 0.0_f32)
    encoded = codec.encode(values, block_size: head_dim, precision: 4)
    live_before = ML::MetalBuffer.stats[:live_bytes]
    resident = ML::GGUF::QwenQBitResidentKV.prepare(encoded, encoded, 1, 1, head_dim)
    resident.release
    resident.release
    ML::MetalBuffer.stats[:live_bytes].should eq(live_before)

    q = Array(Float32).new(6 * head_dim, 0.0_f32)
    expect_raises(ArgumentError, /released/) do
      ML::GGUF::QwenQBitResidentKV.attn_decode(q, q, resident, 6, 6, 1.0_f32)
    end
  end
end
