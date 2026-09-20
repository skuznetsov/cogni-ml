#!/usr/bin/env crystal

require "json"
require "option_parser"
require "../src/ml/core/buffer"
require "../src/ml/metal/device"
require "../src/ml/metal/dispatch"
require "../src/ml/gguf/qwen_qbit_adaptive_kv"

# Model-free falsifier for a runtime-only P4 row layout. Both kernels decode
# the same 136-byte rows and perform the same Float32 work. The baseline reads
# one byte from each canonical plane; the candidate reads eight adjacent
# four-bit prefixes from one little-endian UInt32.
module Qwen35QBitP4RuntimeLayoutProbe
  extend self

  MAX_TOKENS      = 16_384
  MAX_KV_HEADS    =      8
  MAX_TIMED_PAIRS =     20
  SIMD_WIDTH      =     32

  SOURCE = <<-METAL
    #include <metal_stdlib>
    using namespace metal;

    constant uint P4_ROW_VALUES = 256u;
    constant uint P4_PLANE_BYTES = P4_ROW_VALUES / 8u;
    constant uint P4_ROW_BYTES = 8u + 4u * P4_PLANE_BYTES;

    constant uint P4_CENTROID_BITS[8] = {
        0x3da18fb8u, 0x3e747262u, 0x3ecf6ceau, 0x3f158a3au,
        0x3f491a06u, 0x3f8408fbu, 0x3fb45dcfu, 0x4007469au,
    };

    inline uint read_u32_le(device const uchar* src, uint offset) {
        return ((uint)src[offset]) |
               (((uint)src[offset + 1u]) << 8u) |
               (((uint)src[offset + 2u]) << 16u) |
               (((uint)src[offset + 3u]) << 24u);
    }

    inline float p4_centroid(uint prefix) {
        return prefix < 8u
            ? as_type<float>(P4_CENTROID_BITS[prefix])
            : -as_type<float>(P4_CENTROID_BITS[15u - prefix]);
    }

    kernel void p4_canonical_checksum(
        device const uchar* base [[buffer(0)]],
        device float* output [[buffer(1)]],
        constant uint& row_count [[buffer(2)]],
        uint lane [[thread_index_in_threadgroup]],
        uint row [[threadgroup_position_in_grid]]) {
        if (row >= row_count) return;
        const uint row_base = row * P4_ROW_BYTES;
        const float mean = as_type<float>(read_u32_le(base, row_base));
        const float sigma = as_type<float>(read_u32_le(base, row_base + 4u));
        const uint byte_offset = P4_PLANE_BYTES - 1u - lane;
        const uchar p0 = base[row_base + 8u + 0u * P4_PLANE_BYTES + byte_offset];
        const uchar p1 = base[row_base + 8u + 1u * P4_PLANE_BYTES + byte_offset];
        const uchar p2 = base[row_base + 8u + 2u * P4_PLANE_BYTES + byte_offset];
        const uchar p3 = base[row_base + 8u + 3u * P4_PLANE_BYTES + byte_offset];

        float local = 0.0f;
        for (uint within = 0u; within < 8u; ++within) {
            const uint prefix = (((uint)(p0 >> within) & 1u) << 3u) |
                                (((uint)(p1 >> within) & 1u) << 2u) |
                                (((uint)(p2 >> within) & 1u) << 1u) |
                                (((uint)(p3 >> within) & 1u) << 0u);
            local += mean + sigma * p4_centroid(prefix);
        }
        const float total = simd_sum(local);
        if (lane == 0u) output[row] = total;
    }

    kernel void p4_runtime_word_checksum(
        device const uchar* base [[buffer(0)]],
        device float* output [[buffer(1)]],
        constant uint& row_count [[buffer(2)]],
        uint lane [[thread_index_in_threadgroup]],
        uint row [[threadgroup_position_in_grid]]) {
        if (row >= row_count) return;
        const uint row_base = row * P4_ROW_BYTES;
        const float mean = as_type<float>(read_u32_le(base, row_base));
        const float sigma = as_type<float>(read_u32_le(base, row_base + 4u));
        device const uint* words =
            reinterpret_cast<device const uint*>(base + row_base + 8u);
        const uint word = words[lane];

        float local = 0.0f;
        for (uint within = 0u; within < 8u; ++within) {
            const uint prefix = (word >> (within * 4u)) & 0xfu;
            local += mean + sigma * p4_centroid(prefix);
        }
        const float total = simd_sum(local);
        if (lane == 0u) output[row] = total;
    }
    METAL

  def write_u32_le(bytes : Bytes, offset : Int32, value : UInt32) : Nil
    bytes[offset] = (value & 0xff_u32).to_u8
    bytes[offset + 1] = ((value >> 8) & 0xff_u32).to_u8
    bytes[offset + 2] = ((value >> 16) & 0xff_u32).to_u8
    bytes[offset + 3] = ((value >> 24) & 0xff_u32).to_u8
  end

  def runtime_rows(row_count : Int32) : Bytes
    row_bytes = ML::GGUF::QwenQBitAdaptiveKV::BASE_ROW_BYTES
    runtime = Bytes.new(row_count * row_bytes, 0_u8)
    row_count.times do |row|
      row_offset = row * row_bytes
      mean = ((row % 257) - 128).to_f32 / 127.0_f32
      sigma = 0.125_f32 + (row % 29).to_f32 / 31.0_f32
      write_u32_le(runtime, row_offset, mean.unsafe_as(UInt32))
      write_u32_le(runtime, row_offset + 4, sigma.unsafe_as(UInt32))
      32.times do |group|
        word = 0_u32
        8.times do |within|
          prefix = (row * 3 + group * 5 + within * 7) & 15
          word |= prefix.to_u32 << (within * 4)
        end
        write_u32_le(runtime, row_offset + 8 + group * sizeof(UInt32), word)
      end
    end
    runtime
  end

  def percentile(samples : Array(Float64), fraction : Float64) : Float64
    ordered = samples.sort
    ordered[((ordered.size - 1) * fraction).round.to_i]
  end

  def gpu_ms(pipeline : ML::Metal::ComputePipeline,
             input : ML::MetalBuffer,
             output : ML::MetalBuffer,
             row_count : Int32) : Float64
    command = ML::Metal::CommandBuffer.new
    encoder = ML::Metal::ComputeEncoder.new(command)
    encoder.set_pipeline(pipeline)
    encoder.set_buffer(input, 0)
    encoder.set_buffer(output, 1, ML::Metal::BufferAccess::Write)
    encoder.set_value(row_count.to_u32, 2)
    encoder.dispatch_threadgroups({row_count, 1, 1}, {SIMD_WIDTH, 1, 1})
    encoder.end_encoding
    command.commit_and_wait_gpu_elapsed_seconds * 1000.0
  end

  def exact_output(reference : Array(Float32), candidate : Array(Float32))
    raise "output size mismatch" unless reference.size == candidate.size
    mismatches = 0
    finite = true
    max_abs = 0.0_f32
    reference.each_with_index do |expected, index|
      actual = candidate[index]
      finite &&= expected.finite? && actual.finite?
      mismatches += 1 unless expected.unsafe_as(UInt32) == actual.unsafe_as(UInt32)
      delta = (expected - actual).abs
      max_abs = delta if delta > max_abs
    end
    {finite: finite, mismatches: mismatches, max_abs: max_abs}
  end

  def validate!(tokens : Array(Int32), kv_heads : Int32,
                warmup : Int32, timed_pairs : Int32) : Nil
    raise ArgumentError.new("tokens list must not be empty") if tokens.empty?
    tokens.each do |count|
      unless count > 0 && count <= MAX_TOKENS && count % 256 == 0
        raise ArgumentError.new("tokens must be positive multiples of 256 up to #{MAX_TOKENS}")
      end
    end
    unless kv_heads > 0 && kv_heads <= MAX_KV_HEADS
      raise ArgumentError.new("kv-heads must be in 1..#{MAX_KV_HEADS}")
    end
    raise ArgumentError.new("warmup must be in 0..10") unless warmup.in?(0..10)
    unless timed_pairs.in?(4..MAX_TIMED_PAIRS)
      raise ArgumentError.new("pairs must be in 4..#{MAX_TIMED_PAIRS}")
    end
  end

  def run(tokens : Array(Int32), kv_heads : Int32,
          warmup : Int32, timed_pairs : Int32,
          dry_run : Bool) : Nil
    validate!(tokens, kv_heads, warmup, timed_pairs)
    max_rows = tokens.max * kv_heads
    row_bytes = ML::GGUF::QwenQBitAdaptiveKV::BASE_ROW_BYTES
    max_layout_bytes = max_rows.to_i64 * row_bytes
    max_gpu_bytes = 2_i64 * max_layout_bytes + 2_i64 * max_rows * sizeof(Float32)
    puts "qwen35_qbit_p4_runtime_layout_probe"
    puts "  tokens=#{tokens.join(',')} kv_heads=#{kv_heads} warmup=#{warmup} pairs=#{timed_pairs}"
    puts "  max_rows=#{max_rows} row_bytes=#{row_bytes} max_gpu_bytes=#{max_gpu_bytes}"
    puts "  gate=exact_bits,median_speedup>=1.03,wins>=8/10"
    return if dry_run

    raise "Metal not available" unless ML::Metal::Device.available?
    runtime_all = runtime_rows(max_rows)
    canonical_all = ML::GGUF::QwenQBitAdaptiveKV.p4_canonical_base_from_runtime_words(runtime_all)
    rebuilt_runtime = ML::GGUF::QwenQBitAdaptiveKV.p4_runtime_words_from_canonical_base(canonical_all)
    raise "CPU layout round-trip mismatch" unless rebuilt_runtime == runtime_all

    canonical_pipeline = ML::Metal::ComputePipeline.new(
      "p4_canonical_checksum", SOURCE
    )
    runtime_pipeline = ML::Metal::ComputePipeline.new(
      "p4_runtime_word_checksum", SOURCE
    )
    unless canonical_pipeline.thread_execution_width == SIMD_WIDTH &&
           runtime_pipeline.thread_execution_width == SIMD_WIDTH
      raise "probe requires a 32-thread SIMD-group"
    end

    results = [] of NamedTuple(
      tokens: Int32,
      rows: Int32,
      exact: Bool,
      mismatches: Int32,
      max_abs: Float32,
      canonical_p50_ms: Float64,
      runtime_p50_ms: Float64,
      speedup: Float64,
      wins: Int32,
      pairs: Int32,
      admitted: Bool,
    )

    tokens.each do |token_count|
      row_count = token_count * kv_heads
      bytes = row_count * row_bytes
      canonical_input = ML::MetalBuffer.new(bytes.to_i64)
      runtime_input = ML::MetalBuffer.new(bytes.to_i64)
      canonical_output = ML::MetalBuffer.new(row_count.to_i64 * sizeof(Float32))
      runtime_output = ML::MetalBuffer.new(row_count.to_i64 * sizeof(Float32))
      begin
        canonical_input.write_bytes(canonical_all.to_unsafe, bytes)
        runtime_input.write_bytes(runtime_all.to_unsafe, bytes)

        warmup.times do |index|
          if index.even?
            gpu_ms(canonical_pipeline, canonical_input, canonical_output, row_count)
            gpu_ms(runtime_pipeline, runtime_input, runtime_output, row_count)
          else
            gpu_ms(runtime_pipeline, runtime_input, runtime_output, row_count)
            gpu_ms(canonical_pipeline, canonical_input, canonical_output, row_count)
          end
        end

        gpu_ms(canonical_pipeline, canonical_input, canonical_output, row_count)
        gpu_ms(runtime_pipeline, runtime_input, runtime_output, row_count)
        quality = exact_output(
          canonical_output.read(row_count), runtime_output.read(row_count)
        )

        canonical_samples = Array(Float64).new(timed_pairs)
        runtime_samples = Array(Float64).new(timed_pairs)
        timed_pairs.times do |pair|
          order = case pair % 4
                  when 0, 3 then {:canonical, :runtime}
                  else           {:runtime, :canonical}
                  end
          order.each do |variant|
            if variant == :canonical
              canonical_samples << gpu_ms(
                canonical_pipeline, canonical_input, canonical_output, row_count
              )
            else
              runtime_samples << gpu_ms(
                runtime_pipeline, runtime_input, runtime_output, row_count
              )
            end
          end
        end

        canonical_p50 = percentile(canonical_samples, 0.5)
        runtime_p50 = percentile(runtime_samples, 0.5)
        speedup = canonical_p50 / runtime_p50
        wins = (0...timed_pairs).count do |index|
          runtime_samples[index] < canonical_samples[index]
        end
        required_wins = Math.min(8, timed_pairs)
        exact = quality[:finite] && quality[:mismatches] == 0
        admitted = exact && speedup >= 1.03 && wins >= required_wins
        results << {
          tokens:           token_count,
          rows:             row_count,
          exact:            exact,
          mismatches:       quality[:mismatches],
          max_abs:          quality[:max_abs],
          canonical_p50_ms: canonical_p50,
          runtime_p50_ms:   runtime_p50,
          speedup:          speedup,
          wins:             wins,
          pairs:            timed_pairs,
          admitted:         admitted,
        }
        printf "tokens=%d rows=%d exact=%s mismatches=%d max_abs=%.9g\n",
          token_count, row_count, exact, quality[:mismatches], quality[:max_abs]
        printf "  canonical_ms p10=%.6f p50=%.6f p90=%.6f\n",
          percentile(canonical_samples, 0.1), canonical_p50,
          percentile(canonical_samples, 0.9)
        printf "  runtime_ms   p10=%.6f p50=%.6f p90=%.6f speedup=%.4fx wins=%d/%d admitted=%s\n",
          percentile(runtime_samples, 0.1), runtime_p50,
          percentile(runtime_samples, 0.9), speedup, wins, timed_pairs, admitted
      ensure
        canonical_input.release
        runtime_input.release
        canonical_output.release
        runtime_output.release
      end
    end

    puts "QBIT_P4_RUNTIME_LAYOUT_JSON=#{results.to_json}"
  end
end

tokens = [6144, 8192, 16_384]
kv_heads = 4
warmup = 3
timed_pairs = 10
dry_run = false

OptionParser.parse(ARGV) do |parser|
  parser.banner = "Usage: qwen35_qbit_p4_runtime_layout_probe [options]"
  parser.on("--tokens=LIST", "Comma-separated token counts") do |value|
    tokens = value.split(',').map(&.to_i)
  end
  parser.on("--kv-heads=N", "KV heads per token") { |value| kv_heads = value.to_i }
  parser.on("--warmup=N", "Untimed balanced warmup pairs") { |value| warmup = value.to_i }
  parser.on("--pairs=N", "Timed ABBA/BAAB pairs") { |value| timed_pairs = value.to_i }
  parser.on("--dry-run", "Validate and print the bounded resource plan") { dry_run = true }
  parser.on("-h", "--help", "Show help") { puts parser; exit }
end

Qwen35QBitP4RuntimeLayoutProbe.run(
  tokens, kv_heads, warmup, timed_pairs, dry_run
)
