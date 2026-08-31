require "../src/ml/gguf/qwen35_metal"
require "../src/ml/gguf/qwen_qbit_adaptive_kv"
require "../src/ml/gguf/qwen_qbit_adaptive_resident_kv"
require "../src/ml/core/buffer"

# Bounded synthetic timing probe for the append-only adaptive QBit pack path.
# It never loads model weights. The F32 attention comparator is materialized
# only after packing and is released before the next row of the matrix.
module Qwen35QBitAdaptivePackProbe
  extend self

  MAX_REPEATS     =   20
  MAX_LIVE_TOKENS = 2048

  def median(samples : Array(Float64)) : Float64
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
    median(samples)
  end

  def max_diff(a : Array(Float32), b : Array(Float32)) : Float32
    a.each_with_index.max_of { |value, i| (value - b[i]).abs }
  end

  def tiers(mode : String, rows : Int32) : Array(ML::GGUF::QwenQBitAdaptiveKV::Tier)
    case mode
    when "p4"
      Array(ML::GGUF::QwenQBitAdaptiveKV::Tier).new(
        rows, ML::GGUF::QwenQBitAdaptiveKV::Tier::P4
      )
    when "mixed25"
      Array(ML::GGUF::QwenQBitAdaptiveKV::Tier).new(rows) do |row|
        row % 4 == 3 ? ML::GGUF::QwenQBitAdaptiveKV::Tier::BF16 : ML::GGUF::QwenQBitAdaptiveKV::Tier::P4
      end
    when "bf16"
      Array(ML::GGUF::QwenQBitAdaptiveKV::Tier).new(
        rows, ML::GGUF::QwenQBitAdaptiveKV::Tier::BF16
      )
    else
      raise ArgumentError.new("mode must be p4, bf16, or mixed25")
    end
  end

  def run(chunks : Array(Int32), repeats : Int32) : Nil
    raise "Metal not available" unless ML::GGUF::Qwen35Metal.available?
    unless repeats > 0 && repeats <= MAX_REPEATS
      raise ArgumentError.new("repeats must be in 1..#{MAX_REPEATS}")
    end
    chunks.each do |chunk|
      live_tokens = chunk.to_i64 * (repeats + 1)
      unless chunk > 0 && live_tokens <= MAX_LIVE_TOKENS
        raise ArgumentError.new(
          "chunk * (repeats + 1) must be in 1..#{MAX_LIVE_TOKENS}"
        )
      end
    end

    n_head = 24
    n_head_kv = 4
    head_dim = 256
    heads_per_group = 6
    scale = (1.0 / Math.sqrt(head_dim.to_f64)).to_f32
    puts "chunk mode live_tokens density pack_ms pack_gpu_ms pack_gib_s f32_attn_ms qbit_wall_ms qbit_gpu_ms slowdown max_diff"

    chunks.each do |chunk|
      ["p4", "bf16", "mixed25"].each do |mode|
        live_tokens = chunk * (repeats + 1)
        row_count = live_tokens * n_head_kv
        plan = ML::GGUF::QwenQBitAdaptiveKV.plan(tiers(mode, row_count))
        rng = Random.new(0xA991D0_u64 + chunk.to_u64 + (mode == "p4" ? 0_u64 : 1_u64))
        source_values = chunk * n_head_kv * head_dim
        k = Array(Float32).new(source_values) { ((rng.next_float - 0.5) * 1.0).to_f32 }
        v = Array(Float32).new(source_values) { ((rng.next_float - 0.5) * 1.0).to_f32 }
        q = Array(Float32).new(n_head * head_dim) { ((rng.next_float - 0.5) * 1.0).to_f32 }
        gate = Array(Float32).new(n_head * head_dim) { ((rng.next_float - 0.5) * 2.0).to_f32 }

        k_source = ML::MetalBuffer.from_array(k)
        v_source = ML::MetalBuffer.from_array(v)
        resident = ML::GGUF::QwenQBitAdaptiveResidentKV.allocate(
          plan, plan, live_tokens, n_head_kv, head_dim,
        )
        begin
          # First append compiles/warms the pipeline and is not timed.
          ML::GGUF::QwenQBitAdaptiveResidentKV.append_from_metal(
            resident, k_source, v_source, chunk,
          )
          pack_gpu_samples = Array(Float64).new(repeats)
          pack_ms = timed_ms(repeats) do
            pack_gpu_elapsed_seconds = 0.0_f64
            ML::GGUF::QwenQBitAdaptiveResidentKV.append_from_metal(
              resident, k_source, v_source, chunk,
              gpu_elapsed_seconds: pointerof(pack_gpu_elapsed_seconds),
            )
            pack_gpu_samples << pack_gpu_elapsed_seconds * 1000.0
          end
          pack_gpu_ms = median(pack_gpu_samples)

          encoded_k, encoded_v = ML::GGUF::QwenQBitAdaptiveResidentKV.snapshot(resident)
          decoded_k = ML::GGUF::QwenQBitAdaptiveKV.decode(encoded_k)
          decoded_v = ML::GGUF::QwenQBitAdaptiveKV.decode(encoded_v)
          k_f32 = ML::MetalBuffer.from_array(decoded_k)
          v_f32 = ML::MetalBuffer.from_array(decoded_v)
          begin
            expected = ML::GGUF::Qwen35Metal.attn_decode(
              q, gate, k_f32, v_f32,
              live_tokens - 1, n_head, n_head_kv, head_dim, heads_per_group, scale,
            )
            actual = ML::GGUF::QwenQBitAdaptiveResidentKV.attn_decode(
              q, gate, resident, n_head, heads_per_group, scale,
            )
            diff = max_diff(expected, actual)
            raise "adaptive resident QBit parity failed: max_diff=#{diff}" unless diff < 2.0e-4_f32

            f32_ms = timed_ms(repeats) do
              ML::GGUF::Qwen35Metal.attn_decode(
                q, gate, k_f32, v_f32,
                live_tokens - 1, n_head, n_head_kv, head_dim, heads_per_group, scale,
              )
            end
            gpu_samples = Array(Float64).new(repeats)
            qbit_ms = timed_ms(repeats) do
              gpu_elapsed_seconds = 0.0_f64
              ML::GGUF::QwenQBitAdaptiveResidentKV.attn_decode(
                q, gate, resident, n_head, heads_per_group, scale,
                gpu_elapsed_seconds: pointerof(gpu_elapsed_seconds),
              )
              gpu_samples << gpu_elapsed_seconds * 1000.0
            end
            qbit_gpu_ms = median(gpu_samples)

            raw_bytes = 2_i64 * live_tokens * n_head_kv * head_dim * sizeof(Float32)
            packed_input_bytes = 2_i64 * chunk * n_head_kv * head_dim * sizeof(Float32)
            density = raw_bytes.to_f64 / resident.compressed_bytes
            gib_s = packed_input_bytes.to_f64 / (1024.0 ** 3) / (pack_ms / 1000.0)
            printf "%5d %-7s %11d %7.3fx %7.3f %11.3f %10.3f %11.3f %12.3f %11.3f %8.3fx %.3g\n",
              chunk, mode, live_tokens, density, pack_ms, pack_gpu_ms, gib_s,
              f32_ms, qbit_ms, qbit_gpu_ms, qbit_ms / f32_ms, diff
          ensure
            k_f32.release
            v_f32.release
          end
        ensure
          resident.release
          k_source.release
          v_source.release
        end
      end
    end
  end
end

chunks = (ARGV[0]? || "8,32,128").split(',').map(&.to_i)
repeats = (ARGV[1]? || "7").to_i
Qwen35QBitAdaptivePackProbe.run(chunks, repeats)
