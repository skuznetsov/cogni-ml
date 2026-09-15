#!/usr/bin/env crystal

# Model-free regression probe: exact (unpadded) SG4 rows and a separable,
# non-uniform attention oracle. Run only the corrected kernel under run_safe.sh.
require "../src/ml/core/buffer"
require "../src/ml/metal/device"
require "../src/ml/metal/dispatch"
require "../src/ml/metal/process_lease"
require "digest/sha256"

SOURCE          = {{ read_file("#{__DIR__}/../src/ml/gguf/kernels/fullattn_qwen35.metal") }}
REGISTER_SOURCE = "#define QWEN35_SG4_REGISTER_GATE 1\n" + SOURCE
REGISTER_KERNEL = "qwen35_attn_decode_rows_sg4_register"
DIM             =         256
HEADS           =          24
KV_HEADS        =           4
SENTINEL        = 12345.0_f32
CASES           = [{0, 1}, {0, 2}, {0, 3}, {0, 4}, {17, 5}, {17, 6}, {17, 7},
                   {63, 193}, {63, 194}, {63, 195}, {63, 196}, {7839, 195}]
BENCHMARK_CASES = [{7839, 193}, {7839, 194}, {7839, 195}, {7839, 196},
                   {0, 64}, {0, 195}]
BENCHMARK_BLOCKS = 3
CANARY_COUNT     = 4 * HEADS * DIM
ORACLE_LIMIT     = 1.0e-5_f64
PAIR_LIMIT       = 1.0e-6_f64
SLICE_ROWS       =         64

private def single_command_kernel(selector : String) : String
  case selector
  when "--single-command=direct"   then "qwen35_attn_decode_rows_sg4"
  when "--single-command=pregate"  then "qwen35_attn_decode_rows_sg4_pregate"
  when "--single-command=register" then REGISTER_KERNEL
  else                                  raise ArgumentError.new("single-command requires exactly direct, pregate or register")
  end
end

# Offsets apply only to Q, gate and output. K/V remain global and immutable.
private def query_slices(base : Int32, rows : Int32) : Array(Tuple(Int32, Int32, Int32, Int64))
  raise ArgumentError.new("invalid slice extent") unless base >= 0 && rows > 0 && base.to_i64 + rows <= Int32::MAX
  result = [] of Tuple(Int32, Int32, Int32, Int64)
  start = 0
  while start < rows
    size = Math.min(SLICE_ROWS, rows - start)
    result << {start, size, base + start, start.to_i64 * HEADS * DIM * sizeof(Float32)}
    start += size
  end
  result
end

# A future rebuild must explicitly requalify changed shader bytes. In
# particular, never execute an old divergent whole-threadgroup barrier.
private def validate_source!(source : String)
  unless Digest::SHA256.hexdigest(source) == "824f224369ce719cb05ad766717006915a7b25536926e9b1f3eaca913ffdf682"
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

private def validate_slice_output!(actual : Array(Float32), expected : Array(Float64), completed : Int32, previous : Array(Float32)) : Float64
  raise "slice output extent" unless actual.size == expected.size + CANARY_COUNT && previous.size <= completed <= expected.size
  max_error = 0.0_f64
  actual.each_with_index do |value, i|
    if i < completed
      raise "slice non-finite/unwritten output #{i}" unless value.finite? && value != SENTINEL && expected[i].finite?
      raise "slice overwrote completed prefix #{i}" if i < previous.size && value != previous[i]
      max_error = Math.max(max_error, (value.to_f64 - expected[i]).abs)
    else
      raise "slice premature/trailing write #{i}" unless value == SENTINEL
    end
  end
  raise "slice CPU oracle mismatch #{max_error}" unless max_error <= ORACLE_LIMIT
  max_error
end

private def dispatch!(pipe : ML::Metal::ComputePipeline, fixture : ShapeFixture, timed : Bool,
                      row_start : Int32 = 0, row_count : Int32 = fixture.rows,
                      poison : Bool = true, trace : Bool = false) : Tuple(Float64, Float64?)
  raise "dispatch query extent" unless row_start >= 0 && row_count > 0 && row_start.to_i64 + row_count <= fixture.rows
  offset = row_start.to_i64 * HEADS * DIM * sizeof(Float32)
  # This write is deliberately outside the timed interval. A no-op/stale
  # output must fail validation after every sample, including equality checks.
  fixture.poison_output! if poison
  if trace
    puts "sg4_slice_submit kernel=#{pipe.name} base=#{fixture.base} rows=#{fixture.rows} start=#{row_start} slice_rows=#{row_count} slice_base=#{fixture.base + row_start} offset_bytes=#{offset} kv_offset_bytes=0"
    STDOUT.flush
  end
  started = Time.instant
  command = ML::Metal::CommandBuffer.new
  enc = ML::Metal::ComputeEncoder.new(command)
  enc.set_pipeline(pipe)
  fixture.buffers.each_with_index do |buffer, i|
    enc.set_buffer(buffer, i, offset: (i == 0 || i == 1 || i == 4) ? offset : 0_i64)
  end
  enc.set_value((fixture.base + row_start).to_u32, 5)
  enc.set_value(row_count.to_u32, 6)
  enc.set_value(HEADS.to_u32, 7)
  enc.set_value(KV_HEADS.to_u32, 8)
  enc.set_value(DIM.to_u32, 9)
  enc.set_value((HEADS // KV_HEADS).to_u32, 10)
  enc.set_value(1.0_f32 / 16, 11)
  enc.dispatch_threadgroups({HEADS, (row_count + 3) // 4, 1}, {128, 1, 1})
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

private def run_sliced_shape!(direct_pipe : ML::Metal::ComputePipeline,
                              pregate_pipe : ML::Metal::ComputePipeline, base : Int32, rows : Int32) : Nil
  fixture = ShapeFixture.new(base, rows, false)
  begin
    direct_actual = [] of Float32
    {direct_pipe, pregate_pipe}.each do |pipe|
      fixture.poison_output!
      previous = [] of Float32
      query_slices(base, rows).each do |start, size, _slice_base, _offset|
        host_ms, gpu_ms = dispatch!(pipe, fixture, true, start, size, poison: false, trace: true)
        actual = fixture.read_output
        completed = (start + size) * HEADS * DIM
        max_error = validate_slice_output!(actual, fixture.expected, completed, previous)
        previous = actual[0, completed]
        puts "sg4_slice_complete kernel=#{pipe.name} base=#{base} rows=#{rows} start=#{start} slice_rows=#{size} gpu_ms=#{gpu_ms.not_nil!} host_ms=#{host_ms} max_abs=#{max_error} guard=PASS prefix=PASS oracle=PASS"
        STDOUT.flush
      end
      actual = fixture.read_output
      max_error = validate_output!(actual, fixture.expected, fixture.count, pipe.name)
      puts "sg4_slice_kernel kernel=#{pipe.name} base=#{base} rows=#{rows} max_abs=#{max_error} guard=PASS oracle=PASS"
      if pipe == direct_pipe
        direct_actual = actual
      else
        pair_error = max_pair_difference!(direct_actual, actual, fixture.count)
        puts "sg4_slice_pair base=#{base} rows=#{rows} pair_max_abs=#{pair_error} difference=PASS"
      end
      STDOUT.flush
    end
  ensure
    fixture.release
  end
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
  raise "direct selector" unless single_command_kernel("--single-command=direct") == "qwen35_attn_decode_rows_sg4"
  raise "pregate selector" unless single_command_kernel("--single-command=pregate") == "qwen35_attn_decode_rows_sg4_pregate"
  raise "register selector" unless single_command_kernel("--single-command=register") == REGISTER_KERNEL
  {"--single-command", "--single-command=", "--single-command=Direct", "--single-command=both", "--single-command=pregate ", "--single-command=direct=extra"}.each do |bad|
    rejected = false
    begin
      single_command_kernel(bad)
    rescue ArgumentError
      rejected = true
    end
    raise "single-command selector failed open" unless rejected
  end
  {0, 7839}.each do |base|
    {1, 63, 64, 65, 193, 194, 195, 196}.each do |rows|
      seen = [] of Int32
      query_slices(base, rows).each do |start, size, slice_base, offset|
        raise "slice size" unless 1 <= size <= 64
        raise "slice buffer bounds" unless offset >= 0 && offset + size.to_i64 * HEADS * DIM * 4 <= rows.to_i64 * HEADS * DIM * 4
        size.times do |local|
          raise "slice Q/gate/output offset" unless offset // 4 + local * HEADS * DIM == (start + local) * HEADS * DIM
          raise "slice causal end" unless slice_base + local + 1 == base + start + local + 1
          seen << start + local
        end
      end
      raise "slice coverage" unless seen == (0...rows).to_a
    end
  end
  expected_slice = [1.0_f64, 2.0_f64, 3.0_f64]
  previous = [1.0_f32]
  clean = [1.0_f32, 2.0_f32, SENTINEL] + Array(Float32).new(CANARY_COUNT, SENTINEL)
  validate_slice_output!(clean, expected_slice, 2, previous)
  {0, 1, 2, 3}.each do |index|
    corrupt = clean.dup
    # A tiny earlier-prefix change must fail even within oracle tolerance.
    corrupt[index] = index == 0 ? 1.000001_f32 : 0.0_f32
    rejected = false
    begin
      validate_slice_output!(corrupt, expected_slice, 2, previous)
    rescue
      rejected = true
    end
    raise "slice validator failed open at #{index}" unless rejected
  end
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
  puts "source_guard=PASS negative_controls=2 validation_controls=5 slice_shapes=16 slice_negative_controls=4 selector_negative_controls=6 gpu=not_initialized"
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
  if ARGV == ["--pipeline-info"] || ARGV == ["--pipeline-info-register"]
    lease = ML::Metal::ProcessLease.acquire
    begin
      device = ML::Metal::Device.instance
      raise "probe requires Apple GPU" unless device.name.starts_with?("Apple")
      puts "sg4_pipeline_info pid=#{Process.pid} source_sha256=#{Digest::SHA256.hexdigest(SOURCE)} device=#{device.name.gsub(/\s+/, "_")} precision=f32"
      STDOUT.flush
      # Same source, options and compilation order as --single-command.
      # No fixture, tensor allocation, command-buffer creation or compute submission.
      {"qwen35_attn_decode_rows_sg4", "qwen35_attn_decode_rows_sg4_pregate"}.each do |name|
        pipe = ML::Metal::ComputePipeline.new(name, SOURCE)
        bytes = pipe.static_threadgroup_memory_length
        width = pipe.thread_execution_width
        max_threads = pipe.max_total_threads_per_threadgroup
        raise "Invalid pipeline thread limit" unless max_threads >= width
        puts "sg4_pipeline kernel=#{name} static_threadgroup_bytes=#{bytes} thread_execution_width=#{width} max_total_threads=#{max_threads}"
        STDOUT.flush
      end
      if ARGV == ["--pipeline-info-register"]
        pipe = ML::Metal::ComputePipeline.new(REGISTER_KERNEL, REGISTER_SOURCE, "qwen35_attn_decode_rows_sg4_pregate")
        bytes = pipe.static_threadgroup_memory_length
        width = pipe.thread_execution_width
        max_threads = pipe.max_total_threads_per_threadgroup
        puts "sg4_pipeline kernel=#{pipe.name} compiled_source_sha256=#{Digest::SHA256.hexdigest(REGISTER_SOURCE)} static_threadgroup_bytes=#{bytes} thread_execution_width=#{width} max_total_threads=#{max_threads}"
        raise "register candidate misses resource gate" unless bytes == 4608 && width == 32 && max_threads >= 128
      end
    ensure
      lease.close
    end
    puts "pipeline_info=PASS pipelines=#{ARGV == ["--pipeline-info-register"] ? 3 : 2} compute_commands=0"
    exit
  end

  if ARGV.any? { |arg| arg.starts_with?("--single-command") }
    abort "single-command takes exactly one selector argument" unless ARGV.size == 1
    kernel = single_command_kernel(ARGV[0])
    lease = ML::Metal::ProcessLease.acquire
    begin
      device = ML::Metal::Device.instance
      raise "probe requires Apple GPU" unless device.name.starts_with?("Apple")
      puts "sg4_single_command kernel=#{kernel} pid=#{Process.pid} source_sha256=#{Digest::SHA256.hexdigest(SOURCE)} device=#{device.name.gsub(/\s+/, "_")} precision=f32 base=7839 fixture_rows=193 command_rows=64 heads=#{HEADS} kv_heads=#{KV_HEADS} head_dim=#{DIM} warmups=0 cooldown_ms=0"
      STDOUT.flush
      # Controls always compile first; the candidate adds one flagged pipeline.
      direct_pipe = ML::Metal::ComputePipeline.new("qwen35_attn_decode_rows_sg4", SOURCE)
      pregate_pipe = ML::Metal::ComputePipeline.new("qwen35_attn_decode_rows_sg4_pregate", SOURCE)
      pipe = kernel == direct_pipe.name ? direct_pipe : pregate_pipe
      if kernel == REGISTER_KERNEL
        pipe = ML::Metal::ComputePipeline.new(REGISTER_KERNEL, REGISTER_SOURCE, "qwen35_attn_decode_rows_sg4_pregate")
        raise "register candidate misses resource gate" unless pipe.static_threadgroup_memory_length == 4608 && pipe.thread_execution_width == 32 && pipe.max_total_threads_per_threadgroup >= 128
        puts "sg4_candidate compiled_source_sha256=#{Digest::SHA256.hexdigest(REGISTER_SOURCE)} storage=thread_local register_allocation=unmeasured"
        STDOUT.flush
      end
      fixture = ShapeFixture.new(7839, 193, false)
      begin
        host_ms, gpu_ms = dispatch!(pipe, fixture, true, 0, 64, trace: true)
        max_error = validate_slice_output!(fixture.read_output, fixture.expected, 64 * HEADS * DIM, [] of Float32)
        puts "sg4_single_result kernel=#{kernel} pid=#{Process.pid} commands=1 completed_rows=64 max_abs=#{max_error} gpu_ms=#{gpu_ms.not_nil!} host_ms=#{host_ms} guard=PASS oracle=PASS"
        STDOUT.flush
      ensure
        fixture.release
      end
    ensure
      lease.close
    end
    puts "single_command=PASS kernel=#{kernel} commands=1"
    exit
  end

  if ARGV == ["--slice-check"]
    lease = ML::Metal::ProcessLease.acquire
    begin
      device = ML::Metal::Device.instance
      raise "probe requires Apple GPU" unless device.name.starts_with?("Apple")
      puts "sg4_slice_check source_sha256=#{Digest::SHA256.hexdigest(SOURCE)} device=#{device.name.gsub(/\s+/, "_")} precision=f32 heads=#{HEADS} kv_heads=#{KV_HEADS} head_dim=#{DIM} max_slice_rows=#{SLICE_ROWS} cooldown_ms=0 per_slice_validation=true"
      STDOUT.flush
      direct_pipe = ML::Metal::ComputePipeline.new("qwen35_attn_decode_rows_sg4", SOURCE)
      pregate_pipe = ML::Metal::ComputePipeline.new("qwen35_attn_decode_rows_sg4_pregate", SOURCE)
      BENCHMARK_CASES.each do |base, rows|
        run_sliced_shape!(direct_pipe, pregate_pipe, base, rows)
        GC.collect
      end
    ensure
      lease.close
    end
    puts "slice_check=PASS shapes=#{BENCHMARK_CASES.size} kernels=#{BENCHMARK_CASES.size * 2} commands=#{BENCHMARK_CASES.sum { |_, rows| ((rows + SLICE_ROWS - 1) // SLICE_ROWS) * 2 }}"
    exit
  end

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
    abort "usage: qwen35_sg4_tail_probe [--run|--benchmark|--slice-check|--single-command=direct|--single-command=pregate|--single-command=register|--pipeline-info|--pipeline-info-register|--self-test]" unless ARGV.empty?
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
