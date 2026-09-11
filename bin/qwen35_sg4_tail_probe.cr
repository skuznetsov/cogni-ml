#!/usr/bin/env crystal

# Model-free regression probe: exact (unpadded) SG4 rows and a separable,
# non-uniform attention oracle. Run only the corrected kernel under run_safe.sh.
require "../src/ml/core/buffer"
require "../src/ml/metal/device"
require "../src/ml/metal/dispatch"
require "../src/ml/metal/process_lease"
require "digest/sha256"

SOURCE   = {{ read_file("#{__DIR__}/../src/ml/gguf/kernels/fullattn_qwen35.metal") }}
DIM      =         256
HEADS    =          24
KV_HEADS =           4
SENTINEL = 12345.0_f32
CASES    = [{0, 1}, {0, 2}, {0, 3}, {0, 4}, {17, 5}, {17, 6}, {17, 7},
            {63, 193}, {63, 194}, {63, 195}, {63, 196}, {7839, 195}]

# A future rebuild must explicitly requalify changed shader bytes. In
# particular, never execute an old divergent whole-threadgroup barrier.
private def validate_source!(source : String)
  unless Digest::SHA256.hexdigest(source) == "a53054dd97bdfdbfa2e4a8cdc160898f9c7e6c7a5907fbb6b1884bfa1dd2eff1"
    raise ArgumentError.new("SG4 probe shader changed; requalify its synchronization before GPU use")
  end
end

# All synthetic KV values are exact multiples of 1/32 in (-1, 1), hence
# normal binary16 values (or zero); no rounding or subnormal policy is needed.
private def half_bits(value : Float32) : UInt16
  return 0_u16 if value == 0
  bits = value.unsafe_as(UInt32)
  exponent = ((bits >> 23) & 255).to_i - 127 + 15
  raise "synthetic half range" unless 1 <= exponent < 31 && (bits & 0x1fff) == 0
  (((bits >> 16) & 0x8000) | (exponent.to_u32 << 10) | ((bits >> 13) & 1023)).to_u16
end

private def kv_buffer(values : Array(Float32), half : Bool) : ML::MetalBuffer
  return ML::MetalBuffer.from_array(values) unless half
  bits = values.map { |v| half_bits(v) }
  buffer = ML::MetalBuffer.new(bits.size.to_i64 * 2)
  buffer.write_bytes(bits.to_unsafe.as(Pointer(UInt8)), bits.size * 2)
  buffer
end

private def run_case(pipe : ML::Metal::ComputePipeline, half : Bool, base : Int32, rows : Int32)
  count = rows * HEADS * DIM
  q = Array(Float32).new(count) { |i| (((i // (HEADS * DIM) + (i // DIM) % HEADS + i % DIM) % 7 - 3) / 16.0).to_f32 }
  gate = Array(Float32).new(count) { |i| ((i % 17 - 8) / 8.0).to_f32 }
  cache_count = (base + rows) * KV_HEADS * DIM
  k = Array(Float32).new(cache_count) { |i| (((i // (KV_HEADS * DIM)) % 7 - 3) * (i % DIM % 5 - 2) / 32.0).to_f32 }
  v = Array(Float32).new(cache_count) { |i| (((i // (KV_HEADS * DIM)) % 11 - 5) / 16.0 + ((i // DIM) % KV_HEADS) / 8.0 + (i % DIM % 13 - 6) / 32.0).to_f32 }
  buffers = [] of ML::MetalBuffer
  begin
    buffers << ML::MetalBuffer.from_array(q)
    buffers << ML::MetalBuffer.from_array(gate)
    buffers << kv_buffer(k, half)
    buffers << kv_buffer(v, half)
    # One full SG4 output group is poisoned, even for aligned row counts.
    buffers << ML::MetalBuffer.from_array(Array(Float32).new(count + 4 * HEADS * DIM, SENTINEL))
    command = ML::Metal::CommandBuffer.new
    enc = ML::Metal::ComputeEncoder.new(command)
    enc.set_pipeline(pipe)
    buffers.each_with_index { |buffer, i| enc.set_buffer(buffer, i) }
    enc.set_value(base.to_u32, 5)
    enc.set_value(rows.to_u32, 6)
    enc.set_value(HEADS.to_u32, 7)
    enc.set_value(KV_HEADS.to_u32, 8)
    enc.set_value(DIM.to_u32, 9)
    enc.set_value((HEADS // KV_HEADS).to_u32, 10)
    enc.set_value(1.0_f32 / 16, 11)
    enc.dispatch_threadgroups({HEADS, (rows + 3) // 4, 1}, {128, 1, 1})
    enc.end_encoding
    command.commit_and_wait
    actual = buffers.last.read(count + 4 * HEADS * DIM)
    raise "output guard overwritten" unless actual[count..].all? { |x| x == SENTINEL }
    max_error = 0.0
    rows.times do |t|
      HEADS.times do |h|
        offset = (t * HEADS + h) * DIM
        # K[j,d] = position_code[j] * dimension_code[d] / 32.
        # Factorizing the CPU dot makes long-prefix validation cheap, while
        # exercising all query coordinates and non-uniform softmax weights.
        factor = (0...DIM).sum { |d| q[offset + d].to_f64 * (d % 5 - 2) / 32.0 } / 16.0
        weights = (0...7).map { |j| Math.exp((j - 3) * factor) }
        denominator = 0.0
        numerator = 0.0
        (base + t + 1).times do |j|
          weight = weights[j % 7]
          denominator += weight
          numerator += weight * (j % 11 - 5) / 16.0
        end
        DIM.times do |d|
          expected = (numerator / denominator + (h // 6) / 8.0 + (d % 13 - 6) / 32.0) / (1.0 + Math.exp(-gate[offset + d].to_f64))
          value = actual[offset + d]
          raise "non-finite/unwritten output" unless value.finite? && value != SENTINEL
          max_error = Math.max(max_error, (value - expected).abs)
        end
      end
    end
    raise "CPU oracle mismatch #{max_error}" unless max_error <= 1.0e-5
    puts "kernel=#{pipe.name} kv=#{half ? "f16" : "f32"} base=#{base} rows=#{rows} max_abs=#{max_error} guard=PASS oracle=PASS"
    STDOUT.flush
  ensure
    # Failure is terminal for this probe; never submit another case after it.
    buffers.each(&.release)
  end
end

validate_source!(SOURCE)
if ARGV == ["--self-test"]
  {SOURCE + "\n", SOURCE.gsub("simdgroup_barrier(", "threadgroup_barrier(")}.each do |changed|
    rejected = false
    begin
      validate_source!(changed)
    rescue ArgumentError
      rejected = true
    end
    raise "source guard failed open" unless rejected
  end
  puts "source_guard=PASS negative_controls=2 gpu=not_initialized"
  exit
end
unless ARGV == ["--run"]
  abort "usage: qwen35_sg4_tail_probe [--run|--self-test]" unless ARGV.empty?
  puts "dry: #{CASES.size * 4} bounded SG4 cases; no Metal initialization; use --run under scripts/run_safe.sh"
  exit
end

lease = ML::Metal::ProcessLease.acquire
begin
  # Production SG4 assumes the 32-lane SIMD groups of Apple GPUs.
  raise "probe requires Apple GPU" unless ML::Metal::Device.instance.name.starts_with?("Apple")
  {false, true}.each do |half|
    {"qwen35_attn_decode_rows_sg4", "qwen35_attn_decode_rows_sg4_pregate"}.each do |name|
      source = half ? "#define QWEN35_KV_CACHE_F16 1\n" + SOURCE : SOURCE
      pipe = ML::Metal::ComputePipeline.new(name, source)
      CASES.each do |base, rows|
        run_case(pipe, half, base, rows)
        GC.collect
      end
    end
  end
ensure
  lease.close
end
puts "admission=PASS cases=#{CASES.size * 4}"
