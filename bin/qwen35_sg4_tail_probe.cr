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
BENCHMARK_CASES = [{7839, 193}, {7839, 194}, {7839, 195}, {7839, 196},
                   {0, 64}, {0, 195}]
BENCHMARK_BLOCKS = 3
CANARY_COUNT     = 4 * HEADS * DIM
ORACLE_LIMIT     = 1.0e-5_f64
PAIR_LIMIT       = 1.0e-6_f64

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

private def cpu_oracle(q : Array(Float32), gate : Array(Float32), base : Int32, rows : Int32) : Array(Float64)
  count = rows * HEADS * DIM
  expected = Array(Float64).new(count, 0.0_f64)
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
        expected[offset + d] = (numerator / denominator + (h // 6) / 8.0 + (d % 13 - 6) / 32.0) / (1.0 + Math.exp(-gate[offset + d].to_f64))
      end
    end
  end
  expected
end

private class ShapeFixture
  getter base : Int32
  getter rows : Int32
  getter count : Int32
  getter expected : Array(Float64)
  getter buffers : Array(ML::MetalBuffer)
  getter output_buffer : ML::MetalBuffer

  def initialize(@base : Int32, @rows : Int32, half : Bool)
    @count = @rows * HEADS * DIM
    q = Array(Float32).new(count) { |i| (((i // (HEADS * DIM) + (i // DIM) % HEADS + i % DIM) % 7 - 3) / 16.0).to_f32 }
    gate = Array(Float32).new(count) { |i| ((i % 17 - 8) / 8.0).to_f32 }
    cache_count = (@base + @rows) * KV_HEADS * DIM
    k = Array(Float32).new(cache_count) { |i| (((i // (KV_HEADS * DIM)) % 7 - 3) * (i % DIM % 5 - 2) / 32.0).to_f32 }
    v = Array(Float32).new(cache_count) { |i| (((i // (KV_HEADS * DIM)) % 11 - 5) / 16.0 + ((i // DIM) % KV_HEADS) / 8.0 + (i % DIM % 13 - 6) / 32.0).to_f32 }
    @buffers = [] of ML::MetalBuffer
    @buffers << ML::MetalBuffer.from_array(q)
    @buffers << ML::MetalBuffer.from_array(gate)
    @buffers << kv_buffer(k, half)
    kv_v = kv_buffer(v, half)
    @buffers << kv_v
    # One full SG4 output group is poisoned, even for aligned row counts.
    @output_buffer = ML::MetalBuffer.from_array(Array(Float32).new(@count + CANARY_COUNT, SENTINEL))
    @buffers << @output_buffer
    @expected = cpu_oracle(q, gate, @base, @rows)
    @poison = Array(Float32).new(@count + CANARY_COUNT, SENTINEL)
  end

  def poison_output! : Nil
    @output_buffer.write(@poison)
  end

  def read_output : Array(Float32)
    @output_buffer.read(@count + CANARY_COUNT)
  end

  def release : Nil
    @buffers.each(&.release)
  end
end

private def validate_output!(actual : Array(Float32), expected : Array(Float64), count : Int32, label : String) : Float64
  raise "#{label}: output length #{actual.size} != #{count + CANARY_COUNT}" unless actual.size == count + CANARY_COUNT
  raise "#{label}: oracle length #{expected.size} != #{count}" unless expected.size == count
  max_error = 0.0_f64
  i = 0
  while i < count
    value = actual[i]
    raise "#{label}: non-finite/unwritten output at #{i}" unless value.finite? && value != SENTINEL && expected[i].finite?
    max_error = Math.max(max_error, (value.to_f64 - expected[i]).abs)
    i += 1
  end
  while i < actual.size
    raise "#{label}: output guard overwritten at #{i}: #{actual[i]}" unless actual[i] == SENTINEL
    i += 1
  end
  raise "#{label}: CPU oracle mismatch #{max_error}" unless max_error <= ORACLE_LIMIT
  max_error
end

private def max_pair_difference!(direct : Array(Float32), pregate : Array(Float32), count : Int32) : Float64
  raise "direct/pregate output length mismatch" unless direct.size == pregate.size
  max_error = 0.0_f64
  count.times do |i|
    raise "non-finite pair" unless direct[i].finite? && pregate[i].finite?
    max_error = Math.max(max_error, (direct[i].to_f64 - pregate[i].to_f64).abs)
  end
  raise "direct/pregate mismatch #{max_error}" unless max_error <= PAIR_LIMIT
  max_error
end

private def dispatch!(pipe : ML::Metal::ComputePipeline, fixture : ShapeFixture, timed : Bool) : Tuple(Float64, Float64?)
  # This write is deliberately outside the timed interval. A no-op/stale
  # output must fail validation after every sample, including equality checks.
  fixture.poison_output!
  started = Time.instant
  command = ML::Metal::CommandBuffer.new
  enc = ML::Metal::ComputeEncoder.new(command)
  enc.set_pipeline(pipe)
  fixture.buffers.each_with_index { |buffer, i| enc.set_buffer(buffer, i) }
  enc.set_value(fixture.base.to_u32, 5)
  enc.set_value(fixture.rows.to_u32, 6)
  enc.set_value(HEADS.to_u32, 7)
  enc.set_value(KV_HEADS.to_u32, 8)
  enc.set_value(DIM.to_u32, 9)
  enc.set_value((HEADS // KV_HEADS).to_u32, 10)
  enc.set_value(1.0_f32 / 16, 11)
  enc.dispatch_threadgroups({HEADS, (fixture.rows + 3) // 4, 1}, {128, 1, 1})
  enc.end_encoding
  gpu_ms = if timed
             command.commit_and_wait_gpu_elapsed_seconds * 1000.0_f64
           else
             command.commit_and_wait
             nil
           end
  host_ms = (Time.instant - started).total_milliseconds
  {host_ms, gpu_ms}
end

private def run_case(pipe : ML::Metal::ComputePipeline, half : Bool, base : Int32, rows : Int32)
  fixture = ShapeFixture.new(base, rows, half)
  begin
    dispatch!(pipe, fixture, false)
    actual = fixture.read_output
    max_error = validate_output!(actual, fixture.expected, fixture.count, "#{pipe.name}/#{base}/#{rows}")
    puts "kernel=#{pipe.name} kv=#{half ? "f16" : "f32"} base=#{base} rows=#{rows} max_abs=#{max_error} guard=PASS oracle=PASS"
    STDOUT.flush
  ensure
    # Failure is terminal for this probe; never submit another case after it.
    fixture.release
  end
end

private def run_benchmark_shape!(direct_pipe : ML::Metal::ComputePipeline,
                                 pregate_pipe : ML::Metal::ComputePipeline,
                                 base : Int32, rows : Int32) : Nil
  fixture = ShapeFixture.new(base, rows, false)
  begin
    # Correctness is established with the same allocated inputs/output for both
    # kernels before timing starts.
    dispatch!(direct_pipe, fixture, false)
    direct_actual = fixture.read_output
    direct_error = validate_output!(direct_actual, fixture.expected, fixture.count, "direct/#{base}/#{rows}")
    dispatch!(pregate_pipe, fixture, false)
    pregate_actual = fixture.read_output
    pregate_error = validate_output!(pregate_actual, fixture.expected, fixture.count, "pregate/#{base}/#{rows}")
    pair_error = max_pair_difference!(direct_actual, pregate_actual, fixture.count)

    puts "sg4_check base=#{base} rows=#{rows} direct_max_abs=#{direct_error} pregate_max_abs=#{pregate_error} pair_max_abs=#{pair_error} guard=PASS oracle=PASS difference=PASS"
    STDOUT.flush

    # Compile/warm both pipelines outside measured samples, on the same fixture.
    dispatch!(direct_pipe, fixture, false)
    dispatch!(pregate_pipe, fixture, false)

    BENCHMARK_BLOCKS.times do |block|
      {direct_pipe, pregate_pipe, pregate_pipe, direct_pipe}.each_with_index do |pipe, order|
        host_ms, gpu_ms = dispatch!(pipe, fixture, true)
        actual = fixture.read_output
        max_error = validate_output!(actual, fixture.expected, fixture.count, "sample/#{base}/#{rows}/#{block}/#{order}")
        pair_error = max_pair_difference!(actual, pregate_actual, fixture.count)
        gpu_value = gpu_ms.not_nil!
        puts "sg4_sample base=#{base} rows=#{rows} block=#{block} order=#{order} kernel=#{pipe.name} gpu_ms=#{gpu_value} host_ms=#{host_ms} max_abs=#{max_error} pair_max_abs=#{pair_error} guard=PASS oracle=PASS"
        STDOUT.flush
      end
    end

    puts "sg4_summary base=#{base} rows=#{rows} samples=#{BENCHMARK_BLOCKS * 4} direct_max_abs=#{direct_error} pregate_max_abs=#{pregate_error} pair_max_abs=#{pair_error} guard=PASS oracle=PASS difference=PASS"
    STDOUT.flush
  ensure
    fixture.release
  end
end

private def reject_validation!(label : String, actual : Array(Float32), expected : Array(Float64), count : Int32) : Nil
  rejected = false
  begin
    validate_output!(actual, expected, count, label)
  rescue
    rejected = true
  end
  raise "#{label}: validation failed open" unless rejected
end

private def run_self_test : Nil
  {SOURCE + "\n", SOURCE.gsub("simdgroup_barrier(", "threadgroup_barrier(")}.each do |changed|
    rejected = false
    begin
      validate_source!(changed)
    rescue ArgumentError
      rejected = true
    end
    raise "source guard failed open" unless rejected
  end

  expected = [1.0_f64, 2.0_f64, 3.0_f64]
  good = expected.map(&.to_f32) + Array(Float32).new(CANARY_COUNT, SENTINEL)
  validate_output!(good, expected, expected.size.to_i32, "self-test/good")
  corrupted_output = good.dup
  corrupted_output[1] = Float32::NAN
  reject_validation!("self-test/corrupted-output", corrupted_output, expected, expected.size.to_i32)
  corrupted_canary = good.dup
  corrupted_canary[expected.size] = 0.0_f32
  reject_validation!("self-test/corrupted-canary", corrupted_canary, expected, expected.size.to_i32)
  {SENTINEL, 4.0_f32}.each do |bad|
    changed = good.dup
    changed[0] = bad
    reject_validation!("self-test/unwritten-or-wrong", changed, expected, expected.size.to_i32)
  end
  changed = good.dup
  changed[0] += 1.0e-4_f32
  rejected = false
  begin
    max_pair_difference!(good, changed, expected.size.to_i32)
  rescue
    rejected = true
  end
  raise "pair validator failed open" unless rejected
  puts "source_guard=PASS negative_controls=2 validation_controls=5 gpu=not_initialized"
end

validate_source!(SOURCE)
if ARGV == ["--self-test"]
  run_self_test
  exit
end

{% if flag?(:cpu_only) %}
  abort "CPU-only probe supports only --self-test or dry invocation" unless ARGV.empty?
  puts "dry: GPU modes unavailable in CPU-only build"
{% else %}
  if ARGV == ["--benchmark"]
    lease = ML::Metal::ProcessLease.acquire
    begin
      device = ML::Metal::Device.instance
      raise "probe requires Apple GPU" unless device.name.starts_with?("Apple")
      device_name = device.name.gsub(/\s+/, "_")
      puts "sg4_benchmark source_sha256=#{Digest::SHA256.hexdigest(SOURCE)} device=#{device_name} precision=f32 heads=#{HEADS} kv_heads=#{KV_HEADS} head_dim=#{DIM} buffers=shared_per_shape order=ABBA blocks=#{BENCHMARK_BLOCKS}"
      STDOUT.flush
      direct_pipe = ML::Metal::ComputePipeline.new("qwen35_attn_decode_rows_sg4", SOURCE)
      pregate_pipe = ML::Metal::ComputePipeline.new("qwen35_attn_decode_rows_sg4_pregate", SOURCE)
      BENCHMARK_CASES.each do |base, rows|
        run_benchmark_shape!(direct_pipe, pregate_pipe, base, rows)
        GC.collect
      end
    ensure
      lease.close
    end
    puts "benchmark=PASS shapes=#{BENCHMARK_CASES.size} samples=#{BENCHMARK_CASES.size * BENCHMARK_BLOCKS * 4}"
    exit
  end

  unless ARGV == ["--run"]
    abort "usage: qwen35_sg4_tail_probe [--run|--benchmark|--self-test]" unless ARGV.empty?
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
{% end %}
