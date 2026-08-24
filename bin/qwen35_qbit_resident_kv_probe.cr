require "../src/ml/gguf/qwen35_metal"
require "../src/ml/gguf/qwen_qbit_gaussian_codec"
require "../src/ml/gguf/qwen_qbit_resident_kv"
require "../src/ml/core/buffer"

module Qwen35QBitResidentKVProbe
  extend self

  MAX_CONTEXT = 8192
  MAX_REPEATS =   20

  def median_ms(samples : Array(Float64)) : Float64
    ordered = samples.sort
    ordered[ordered.size // 2]
  end

  def timed_ms(repeats : Int32, &block : ->) : Float64
    samples = Array(Float64).new(repeats)
    repeats.times do
      started = Time.instant
      yield
      samples << (Time.instant - started).total_milliseconds
    end
    median_ms(samples)
  end

  def max_diff(a : Array(Float32), b : Array(Float32)) : Float32
    a.each_with_index.max_of { |value, i| (value - b[i]).abs }
  end

  def run(contexts : Array(Int32), precision : Int32, repeats : Int32) : Nil
    raise "Metal not available" unless ML::GGUF::Qwen35Metal.available?
    unless precision == 4 || precision == 5
      raise ArgumentError.new("precision must be 4 or 5")
    end
    unless repeats > 0 && repeats <= MAX_REPEATS
      raise ArgumentError.new("repeats must be in 1..#{MAX_REPEATS}")
    end
    contexts.each do |context|
      unless context > 0 && context <= MAX_CONTEXT
        raise ArgumentError.new("context must be in 1..#{MAX_CONTEXT}")
      end
    end

    n_head = 6
    n_head_kv = 1
    head_dim = 256
    heads_per_group = 6
    q_dim = n_head * head_dim
    scale = (1.0 / Math.sqrt(head_dim.to_f64)).to_f32

    puts "context precision f32_mib qbit_mib ratio f32_ms qbit_ms slowdown max_diff"
    contexts.each do |cache_len|
      rng = Random.new(0x514B17_u64 + cache_len.to_u64)
      q = Array(Float32).new(q_dim) { ((rng.next_float - 0.5) * 1.0).to_f32 }
      gate = Array(Float32).new(q_dim) { ((rng.next_float - 0.5) * 2.0).to_f32 }
      kv_count = cache_len * n_head_kv * head_dim
      k = Array(Float32).new(kv_count) { ((rng.next_float - 0.5) * 1.0).to_f32 }
      v = Array(Float32).new(kv_count) { ((rng.next_float - 0.5) * 1.0).to_f32 }
      encoded_k = ML::GGUF::QwenQBitGaussianCodec.encode(k, block_size: head_dim, precision: precision)
      encoded_v = ML::GGUF::QwenQBitGaussianCodec.encode(v, block_size: head_dim, precision: precision)
      decoded_k = ML::GGUF::QwenQBitGaussianCodec.decode(encoded_k)
      decoded_v = ML::GGUF::QwenQBitGaussianCodec.decode(encoded_v)

      resident = ML::GGUF::QwenQBitResidentKV.prepare(
        encoded_k, encoded_v, cache_len, n_head_kv, head_dim,
      )
      begin
        k_buffer = ML::MetalBuffer.from_array(decoded_k)
        begin
          v_buffer = ML::MetalBuffer.from_array(decoded_v)
          begin
            expected = ML::GGUF::Qwen35Metal.attn_decode(
              q, gate, k_buffer, v_buffer,
              cache_len - 1, n_head, n_head_kv, head_dim, heads_per_group, scale,
            )
            actual = ML::GGUF::QwenQBitResidentKV.attn_decode(
              q, gate, resident, n_head, heads_per_group, scale,
            )
            diff = max_diff(expected, actual)
            raise "resident QBit parity failed: max_diff=#{diff}" unless diff < 2.0e-4_f32

            f32_ms = timed_ms(repeats) do
              ML::GGUF::Qwen35Metal.attn_decode(
                q, gate, k_buffer, v_buffer,
                cache_len - 1, n_head, n_head_kv, head_dim, heads_per_group, scale,
              )
            end
            qbit_ms = timed_ms(repeats) do
              ML::GGUF::QwenQBitResidentKV.attn_decode(
                q, gate, resident, n_head, heads_per_group, scale,
              )
            end

            f32_bytes = 2_i64 * kv_count * sizeof(Float32)
            qbit_bytes = resident.compressed_bytes
            printf "%7d p%d %7.3f %8.3f %5.2fx %7.3f %8.3f %7.2fx %.3g\n",
              cache_len, precision,
              f32_bytes.to_f64 / 1_048_576.0,
              qbit_bytes.to_f64 / 1_048_576.0,
              f32_bytes.to_f64 / qbit_bytes,
              f32_ms, qbit_ms, qbit_ms / f32_ms, diff
          ensure
            v_buffer.release
          end
        ensure
          k_buffer.release
        end
      ensure
        resident.release
      end
    end
  end
end

contexts = (ARGV[0]? || "128,512,2048").split(',').map(&.to_i)
precision = (ARGV[1]? || "4").to_i
repeats = (ARGV[2]? || "5").to_i
Qwen35QBitResidentKVProbe.run(contexts, precision, repeats)
