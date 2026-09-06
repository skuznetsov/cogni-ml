#!/usr/bin/env crystal

# Synthetic correctness and optional timing probe for the d256 Flash-MMA
# attention kernel. This is intentionally a direct, no-model harness: it
# exercises the kernel ABI and cache geometry without changing production
# admission or routing.

require "option_parser"
require "../src/ml/core/buffer"
require "../src/ml/metal/device"
require "../src/ml/metal/dispatch"

BASELINE_SOURCE = "#define QWEN35_KV_CACHE_F16 1\n" + {{ read_file("#{__DIR__}/../src/ml/gguf/kernels/fullattn_qwen35.metal") }}
FLASH_SOURCE    = {{ read_file("#{__DIR__}/../src/ml/gguf/kernels/qwen35_attn_flash_d256.metal") }}

HEAD_DIM    =           256
KV_HEADS    =             4
MAX_TOKENS  =          2048
MAX_TOTAL   =          8192
SENTINEL    = 12345.678_f32
COSINE_MIN  =    0.9999_f64
MAX_ABS_TOL =    1.0e-4_f64
RMSE_TOL    =    2.0e-5_f64

record ShapeCase, base_pos : Int32, n_tokens : Int32

# The set is deliberately bounded while covering every requested base and
# token boundary. The CPU oracle runs on a smaller subset below.
CORRECTNESS_CASES = [
  ShapeCase.new(0, 1),
  ShapeCase.new(0, 7),
  ShapeCase.new(0, 8),
  ShapeCase.new(0, 63),
  ShapeCase.new(0, 64),
  ShapeCase.new(0, 65),
  ShapeCase.new(1, 7),
  ShapeCase.new(63, 1),
  ShapeCase.new(63, 8),
  ShapeCase.new(64, 1),
  ShapeCase.new(64, 63),
  ShapeCase.new(65, 64),
  ShapeCase.new(65, 65),
  ShapeCase.new(63, 65),
  ShapeCase.new(256, 512),
  ShapeCase.new(0, 1024),
  ShapeCase.new(0, 2048),
  ShapeCase.new(8126, 65),
  ShapeCase.new(8127, 65),
  ShapeCase.new(8191, 1),
]

# Small enough for an independent O(heads * rows * visible-keys * d) oracle,
# but crossing the scalar-tail, full-64-key, prefix, and query-tail seams.
ORACLE_CASES = [
  ShapeCase.new(0, 1),
  ShapeCase.new(0, 63),
  ShapeCase.new(0, 64),
  ShapeCase.new(0, 65),
  ShapeCase.new(63, 1),
  ShapeCase.new(64, 1),
  ShapeCase.new(65, 65),
  ShapeCase.new(8191, 1),
]

record SyntheticInputs,
  n_head : Int32,
  n_head_kv : Int32,
  padded_tokens : Int32,
  cache_tokens : Int32,
  q : Array(Float32),
  gate : Array(Float32),
  k_bits : Array(UInt16),
  v_bits : Array(UInt16),
  k : Array(Float32),
  v : Array(Float32)

record Metrics,
  finite : Bool,
  cosine : Float64,
  rmse : Float64,
  max_abs : Float64

private def round_shift(value : UInt32, shift : Int32) : UInt32
  return value if shift <= 0
  return 0_u32 if shift >= 32

  base = value >> shift
  mask = (1_u32 << shift) - 1_u32
  remainder = value & mask
  halfway = 1_u32 << (shift - 1)
  if remainder > halfway || (remainder == halfway && (base & 1_u32) != 0)
    base + 1_u32
  else
    base
  end
end

# Correct round-to-nearest-even conversion for finite normal Float32 values.
# The synthetic generator stays well inside the normal range, but handling the
# carry into the exponent here prevents the common mantissa-overflow bug.
private def f32_to_f16_bits(value : Float32) : UInt16
  bits = value.unsafe_as(UInt32)
  sign = ((bits >> 16) & 0x8000_u32).to_u16
  exponent = ((bits >> 23) & 0xff_u32).to_i32
  mantissa = bits & 0x7fffff_u32

  if exponent == 0xff
    return sign | 0x7c00_u16 | (mantissa == 0 ? 0_u16 : 0x0200_u16)
  end
  # Float32 subnormals are far below the synthetic range; preserving their
  # signed zero is sufficient for this probe.
  return sign if exponent == 0

  unbiased = exponent - 127
  if unbiased < -14
    significand = mantissa | 0x800000_u32
    shift = -unbiased - 1
    rounded = round_shift(significand, shift)
    return sign if rounded == 0
    return sign | 0x0400_u16 if rounded >= 0x400_u32
    return sign | rounded.to_u16
  end

  return sign | 0x7c00_u16 if unbiased > 15
  half_exponent = unbiased + 15
  half_mantissa = round_shift(mantissa, 13)
  if half_mantissa >= 0x400_u32
    half_exponent += 1
    half_mantissa = 0_u32
    return sign | 0x7c00_u16 if half_exponent >= 31
  end
  sign | (half_exponent.to_u16 << 10) | half_mantissa.to_u16
end

private def f16_to_f32(bits : UInt16) : Float32
  sign = (bits >> 15) & 1
  exponent = (bits >> 10) & 0x1f
  mantissa = bits & 0x03ff

  if exponent == 0
    return 0.0_f32 if mantissa == 0
    value = mantissa.to_f32 / 1024.0_f32
    value *= 2.0_f32 ** -14
    return sign == 1 ? -value : value
  end

  if exponent == 31
    return sign == 1 ? -Float32::INFINITY : Float32::INFINITY if mantissa == 0
    return Float32::NAN
  end

  value = (1.0_f32 + mantissa.to_f32 / 1024.0_f32) * (2.0_f32 ** (exponent.to_i32 - 15))
  sign == 1 ? -value : value
end

private def buffer_from_u16(values : Array(UInt16)) : ML::MetalBuffer
  buffer = ML::MetalBuffer.new(values.size.to_i64 * sizeof(UInt16))
  buffer.write_bytes(values.to_unsafe.as(Pointer(UInt8)), values.size * sizeof(UInt16))
  buffer
end

private def round_up(value : Int32, multiple : Int32) : Int32
  ((value + multiple - 1) // multiple) * multiple
end

# Values are generated from small integer numerators over powers of two, then
# explicitly rounded through H16. This keeps the oracle's Q representation
# identical to the Flash kernel's half query tile while retaining deterministic
# and visibly distinct prefix rows.
private def build_inputs(shape : ShapeCase, heads_per_group : Int32) : SyntheticInputs
  n_head = KV_HEADS * heads_per_group
  # Keep query padding plus one poisoned guard row. Flash receives the exact
  # n_tokens value and must leave every padded output row untouched, including
  # when n_tokens is already aligned. The legacy baseline is dispatched at a
  # multiple of four (see run_baseline), so its extra query/cache rows remain
  # inside allocated storage but are not part of the comparison.
  padded_tokens = round_up(shape.n_tokens, 8) + 8
  cache_tokens = shape.base_pos + round_up(shape.n_tokens, 4) + 1
  q_count = padded_tokens * n_head * HEAD_DIM
  gate_count = q_count
  cache_count = cache_tokens * KV_HEADS * HEAD_DIM
  q = Array(Float32).new(q_count, Float32::NAN)
  gate = Array(Float32).new(gate_count, Float32::NAN)
  k_bits = Array(UInt16).new(cache_count, 0x7e00_u16)
  v_bits = Array(UInt16).new(cache_count, 0x7e00_u16)
  k = Array(Float32).new(cache_count, Float32::NAN)
  v = Array(Float32).new(cache_count, Float32::NAN)

  shape.n_tokens.times do |t|
    n_head.times do |h|
      row = (t * n_head + h) * HEAD_DIM
      HEAD_DIM.times do |d|
        q_code = ((t * 37 + h * 19 + d * 11 + 7) % 257) - 128
        q_raw = q_code.to_f32 / 256.0_f32
        # Q is deliberately uploaded as Float32 after this H16 round trip;
        # Flash loads the same value into its half query tile.
        q[row + d] = f16_to_f32(f32_to_f16_bits(q_raw))

        gate_code = ((t * 13 + h * 5 + d * 3 + 1) % 65) - 32
        gate[row + d] = gate_code.to_f32 / 16.0_f32
      end
    end
  end

  (shape.base_pos + shape.n_tokens).times do |position|
    KV_HEADS.times do |kv_h|
      row = (position * KV_HEADS + kv_h) * HEAD_DIM
      HEAD_DIM.times do |d|
        k_code = ((position * 17 + kv_h * 29 + d * 7 + 31) % 257) - 128
        v_code = ((position * 43 + kv_h * 11 + d * 5 + 97) % 257) - 128
        k_raw = k_code.to_f32 / 256.0_f32
        v_raw = v_code.to_f32 / 128.0_f32
        kb = f32_to_f16_bits(k_raw)
        vb = f32_to_f16_bits(v_raw)
        k_bits[row + d] = kb
        v_bits[row + d] = vb
        k[row + d] = f16_to_f32(kb)
        v[row + d] = f16_to_f32(vb)
      end
    end
  end

  SyntheticInputs.new(n_head, KV_HEADS, padded_tokens, cache_tokens, q, gate,
    k_bits, v_bits, k, v)
end

private def run_baseline(pipe : ML::Metal::ComputePipeline,
                         q : ML::MetalBuffer,
                         gate : ML::MetalBuffer,
                         k : ML::MetalBuffer,
                         v : ML::MetalBuffer,
                         output : ML::MetalBuffer,
                         shape : ShapeCase,
                         dispatch_tokens : Int32,
                         n_head : Int32,
                         n_head_kv : Int32,
                         heads_per_group : Int32,
                         flash_reference : Bool = false) : Float64
  if flash_reference
    return run_flash(pipe, q, gate, k, v, output, shape, n_head, n_head_kv, heads_per_group)
  end
  scale = 1.0_f32 / Math.sqrt(HEAD_DIM.to_f32)
  cmd = ML::Metal::CommandBuffer.new
  enc = ML::Metal::ComputeEncoder.new(cmd)
  enc.set_pipeline(pipe)
  enc.set_buffer(q, 0)
  enc.set_buffer(gate, 1)
  enc.set_buffer(k, 2)
  enc.set_buffer(v, 3)
  enc.set_buffer(output, 4, ML::Metal::BufferAccess::Write)
  enc.set_value(shape.base_pos.to_u32, 5)
  # qwen35_attn_decode_rows_sg4 returns before its threadgroup barrier when
  # an SG4 lane is past n_tokens. Pad this legacy dispatch to four rows so
  # the comparison remains bounded and does not rely on its partial-group
  # barrier behavior. Only the requested shape.n_tokens rows are compared.
  enc.set_value(dispatch_tokens.to_u32, 6)
  enc.set_value(n_head.to_u32, 7)
  enc.set_value(n_head_kv.to_u32, 8)
  enc.set_value(HEAD_DIM.to_u32, 9)
  enc.set_value(heads_per_group.to_u32, 10)
  enc.set_value(scale, 11)
  # rows_sg4 maps x to head and y to groups of four query rows.
  enc.dispatch_threadgroups({n_head, (dispatch_tokens + 3) // 4, 1}, {128, 1, 1})
  enc.end_encoding
  cmd.commit_and_wait_gpu_elapsed_seconds * 1000.0
end

private def run_flash(pipe : ML::Metal::ComputePipeline,
                      q : ML::MetalBuffer,
                      gate : ML::MetalBuffer,
                      k : ML::MetalBuffer,
                      v : ML::MetalBuffer,
                      output : ML::MetalBuffer,
                      shape : ShapeCase,
                      n_head : Int32,
                      n_head_kv : Int32,
                      heads_per_group : Int32) : Float64
  scale = 1.0_f32 / Math.sqrt(HEAD_DIM.to_f32)
  cmd = ML::Metal::CommandBuffer.new
  enc = ML::Metal::ComputeEncoder.new(cmd)
  enc.set_pipeline(pipe)
  enc.set_buffer(q, 0)
  enc.set_buffer(gate, 1)
  enc.set_buffer(k, 2)
  enc.set_buffer(v, 3)
  enc.set_buffer(output, 4, ML::Metal::BufferAccess::Write)
  enc.set_value(shape.base_pos.to_u32, 5)
  enc.set_value(shape.n_tokens.to_u32, 6)
  enc.set_value(n_head.to_u32, 7)
  enc.set_value(n_head_kv.to_u32, 8)
  enc.set_value(HEAD_DIM.to_u32, 9)
  enc.set_value(heads_per_group.to_u32, 10)
  enc.set_value(scale, 11)
  enc.set_threadgroup_memory(16 * 1024, 0)
  enc.dispatch_threadgroups({(shape.n_tokens + 7) // 8, n_head, 1}, {32, 4, 1})
  enc.end_encoding
  cmd.commit_and_wait_gpu_elapsed_seconds * 1000.0
end

private def cpu_attention(inputs : SyntheticInputs, shape : ShapeCase) : Array(Float32)
  scale = 1.0_f64 / Math.sqrt(HEAD_DIM.to_f64)
  output = Array(Float32).new(shape.n_tokens * inputs.n_head * HEAD_DIM, 0.0_f32)
  heads_per_group = inputs.n_head // inputs.n_head_kv

  shape.n_tokens.times do |t|
    query_position = shape.base_pos + t
    inputs.n_head.times do |h|
      kv_h = h // heads_per_group
      q_row = (t * inputs.n_head + h) * HEAD_DIM
      scores = Array(Float64).new(query_position + 1, 0.0_f64)
      max_score = -Float64::INFINITY
      (query_position + 1).times do |key|
        k_row = (key * inputs.n_head_kv + kv_h) * HEAD_DIM
        dot = 0.0_f64
        HEAD_DIM.times do |d|
          dot += inputs.q[q_row + d].to_f64 * inputs.k[k_row + d].to_f64
        end
        score = dot * scale
        scores[key] = score
        max_score = score if score > max_score
      end

      denominator = 0.0_f64
      (query_position + 1).times do |visible_key|
        denominator += Math.exp(scores[visible_key] - max_score)
      end
      inv_denominator = denominator > 0.0 ? 1.0 / denominator : 0.0
      out_row = (t * inputs.n_head + h) * HEAD_DIM
      HEAD_DIM.times do |d|
        weighted = 0.0_f64
        (query_position + 1).times do |visible_key|
          v_row = (visible_key * inputs.n_head_kv + kv_h) * HEAD_DIM
          probability = Math.exp(scores[visible_key] - max_score) * inv_denominator
          weighted += probability * inputs.v[v_row + d].to_f64
        end
        gate = inputs.gate[q_row + d].to_f64
        output[out_row + d] = (weighted / (1.0 + Math.exp(-gate))).to_f32
      end
    end
  end
  output
end

private def compare(reference : Array(Float32), candidate : Array(Float32), count : Int32) : Metrics
  raise "comparison count exceeds reference" if count > reference.size || count > candidate.size
  dot = 0.0_f64
  ref_norm = 0.0_f64
  candidate_norm = 0.0_f64
  squared = 0.0_f64
  max_abs = 0.0_f64
  finite = true
  count.times do |i|
    expected = reference[i]
    actual = candidate[i]
    finite &&= expected.finite? && actual.finite?
    delta = actual.to_f64 - expected.to_f64
    abs_delta = delta.abs
    max_abs = abs_delta if abs_delta > max_abs
    squared += delta * delta
    dot += expected.to_f64 * actual.to_f64
    ref_norm += expected.to_f64 * expected.to_f64
    candidate_norm += actual.to_f64 * actual.to_f64
  end
  cosine = if ref_norm > 0.0 && candidate_norm > 0.0
             dot / Math.sqrt(ref_norm * candidate_norm)
           else
             0.0_f64
           end
  Metrics.new(finite, cosine, Math.sqrt(squared / count), max_abs)
end

private def active_guard(values : Array(Float32), shape : ShapeCase,
                         inputs : SyntheticInputs) : {Bool, Int32}
  active_count = shape.n_tokens * inputs.n_head * HEAD_DIM
  active_sentinel_count = 0
  active_finite = true
  active_count.times do |i|
    value = values[i]
    active_sentinel_count += 1 if value == SENTINEL
    active_finite &&= value.finite?
  end
  {active_finite && active_sentinel_count == 0, active_sentinel_count}
end

private def sentinel_check(values : Array(Float32), shape : ShapeCase,
                           inputs : SyntheticInputs) : {Bool, Bool, Int32}
  active = active_guard(values, shape, inputs)
  active_count = shape.n_tokens * inputs.n_head * HEAD_DIM
  raise "output shorter than active shape" if active_count > values.size
  inactive_unchanged = true
  values[active_count, values.size - active_count].each do |value|
    inactive_unchanged &&= value == SENTINEL
  end
  {active[0], inactive_unchanged, active[1]}
end

private def prefix_distinct?(inputs : SyntheticInputs, shape : ShapeCase) : Bool
  return true if shape.base_pos < 2
  row0 = 0
  row1 = inputs.n_head_kv * HEAD_DIM
  inputs.k[row0, HEAD_DIM].each_with_index do |value, d|
    return true if value != inputs.k[row1 + d]
  end
  inputs.v[row0, HEAD_DIM].each_with_index do |value, d|
    return true if value != inputs.v[row1 + d]
  end
  false
end

private def oracle_case?(shape : ShapeCase) : Bool
  ORACLE_CASES.any? { |candidate| candidate == shape }
end

private def assert_cpu_causal_mask! : Nil
  # Tiny deterministic mask control: with zero Q/K, row 0 must not see the
  # future row's value, while row 1 must average both visible rows.
  shape = ShapeCase.new(0, 2)
  inputs = build_inputs(shape, 4)
  inputs.q.fill(0.0_f32)
  inputs.gate.fill(0.0_f32)
  inputs.k.fill(0.0_f32)
  inputs.v.fill(0.0_f32)
  inputs.v_bits.fill(0_u16)
  one = f32_to_f16_bits(1.0_f32)
  row_one = KV_HEADS * HEAD_DIM
  HEAD_DIM.times do |d|
    inputs.v[row_one + d] = 1.0_f32
    inputs.v_bits[row_one + d] = one
  end
  output = cpu_attention(inputs, shape)
  first_row = output[0, HEAD_DIM]
  second_row = output[inputs.n_head * HEAD_DIM, HEAD_DIM]
  raise "CPU causal mask leaked a future row" unless first_row.all? { |value| value.abs < 1.0e-7_f32 }
  raise "CPU causal mask failed current-row inclusion" unless second_row.all? { |value| (value - 0.25_f32).abs < 1.0e-6_f32 }
end

private def percentile(values : Array(Float64), fraction : Float64) : Float64
  sorted = values.sort
  sorted[((sorted.size - 1) * fraction).round.to_i]
end

private def active_metrics!(baseline_values : Array(Float32), flash_values : Array(Float32),
                            shape : ShapeCase, inputs : SyntheticInputs,
                            label : String) : Metrics
  active_count = shape.n_tokens * inputs.n_head * HEAD_DIM
  baseline_guard = active_guard(baseline_values, shape, inputs)
  flash_guard = sentinel_check(flash_values, shape, inputs)
  raise "#{label}: baseline produced non-finite or sentinel active output" unless baseline_guard[0]
  raise "#{label}: Flash produced non-finite or sentinel active output (active_sentinels=#{flash_guard[2]})" unless flash_guard[0]
  raise "#{label}: Flash wrote inactive padded output" unless flash_guard[1]
  metrics = compare(baseline_values, flash_values, active_count)
  unless metrics.finite && metrics.cosine >= COSINE_MIN && metrics.max_abs <= MAX_ABS_TOL && metrics.rmse <= RMSE_TOL
    raise "#{label}: Flash correctness failed finite=#{metrics.finite} cosine=#{metrics.cosine} rmse=#{metrics.rmse} max_abs=#{metrics.max_abs}"
  end
  metrics
end

private def assert_flash_equal!(reference : Array(Float32), candidate : Array(Float32), count : Int32) : Nil
  count.times do |i|
    unless reference[i].finite? && candidate[i].finite? &&
           reference[i].unsafe_as(UInt32) == candidate[i].unsafe_as(UInt32)
      raise "Flash reference bit mismatch at #{i}: #{reference[i]} != #{candidate[i]}"
    end
  end
end

private def qualify_flash_comparator! : Nil
  values = [0.25_f32, -0.5_f32]
  assert_flash_equal!(values, values, values.size)
  [[0.25_f32, -0.25_f32], [0.25_f32, Float32::NAN]].each do |bad|
    rejected = false
    begin
      assert_flash_equal!(values, bad, values.size)
    rescue
      rejected = true
    end
    raise "Flash comparator failed seeded defect" unless rejected
  end
end

flash_source_path = nil.as(String?)
reference_flash_path = nil.as(String?)
run_perf = false
perf_only = false
warmup = 1
reps = 4

OptionParser.parse(ARGV) do |parser|
  parser.banner = "Usage: qwen35_attn_flash_d256_prefix_probe [--perf|--perf-only] [--flash-source=PATH]"
  parser.on("--perf", "Run optional P1024/4096 × T256/512 ABBA timing cases") { run_perf = true }
  parser.on("--perf-only", "Run only aligned base0 T256/512/1024/2048 timing controls") { run_perf = true; perf_only = true }
  parser.on("--flash-source=PATH", "Use an alternate Flash source (e.g. an old-shader red control)") { |value| flash_source_path = value }
  parser.on("--flash-reference=PATH", "Compare bit-for-bit and time against an old Flash shader instead of SG4") { |value| reference_flash_path = value }
  parser.on("--warmup=N", "ABBA warmup blocks (default: 1)") { |value| warmup = value.to_i }
  parser.on("--reps=N", "ABBA timing blocks (default: 4)") { |value| reps = value.to_i }
  parser.on("-h", "--help", "Show help") { puts parser; exit }
end

raise "Metal not available" unless ML::Metal::Device.available?
raise "warmup must be non-negative" unless warmup >= 0
raise "reps must be positive" unless reps > 0

assert_cpu_causal_mask!
qualify_flash_comparator!

flash_source = if path = flash_source_path
                 raise "Flash source not found: #{path}" unless File.file?(path)
                 File.read(path)
               else
                 FLASH_SOURCE
               end

flash_reference = !!reference_flash_path
baseline_pipe = if path = reference_flash_path
                  ML::Metal::ComputePipeline.new("qwen35_attn_flash_d256", File.read(path))
                else
                  ML::Metal::ComputePipeline.new("qwen35_attn_decode_rows_sg4", BASELINE_SOURCE)
                end
flash_pipe = ML::Metal::ComputePipeline.new("qwen35_attn_flash_d256", flash_source)

puts "qwen35_attn_flash_d256_prefix_probe"
puts "device=#{ML::Metal::Device.instance.name.inspect} baseline=#{reference_flash_path || "qwen35_attn_decode_rows_sg4"} flash_source=#{flash_source_path || "embedded"} exact_flash_comparator=PASS"
puts "correctness_cases=#{CORRECTNESS_CASES.size} oracle_cases=#{ORACLE_CASES.size} gqa_groups=4,6 cpu_causal_mask=PASS"

gqa_groups = [4_i32, 6_i32]
unless perf_only
  CORRECTNESS_CASES.each do |shape|
    raise "invalid token count" unless shape.n_tokens > 0 && shape.n_tokens <= MAX_TOKENS
    raise "invalid total context" unless shape.base_pos >= 0 && shape.base_pos + shape.n_tokens <= MAX_TOTAL
    gqa_groups.each do |heads_per_group|
      inputs = build_inputs(shape, heads_per_group)
      q_buf = ML::MetalBuffer.from_array(inputs.q)
      gate_buf = ML::MetalBuffer.from_array(inputs.gate)
      k_buf = buffer_from_u16(inputs.k_bits)
      v_buf = buffer_from_u16(inputs.v_bits)
      output_count = inputs.padded_tokens * inputs.n_head * HEAD_DIM
      baseline_out = ML::MetalBuffer.from_array(Array(Float32).new(output_count, SENTINEL))
      flash_out = ML::MetalBuffer.from_array(Array(Float32).new(output_count, SENTINEL))
      case_label = "base=#{shape.base_pos} tokens=#{shape.n_tokens} gqa=#{heads_per_group}"
      dispatch_tokens = round_up(shape.n_tokens, 4)

      run_baseline(baseline_pipe, q_buf, gate_buf, k_buf, v_buf, baseline_out,
        shape, dispatch_tokens, inputs.n_head, inputs.n_head_kv, heads_per_group, flash_reference)
      baseline_values = baseline_out.read(output_count)
      oracle = nil.as(Array(Float32)?)
      if oracle_case?(shape)
        oracle = cpu_attention(inputs, shape)
        baseline_oracle = compare(oracle, baseline_values, shape.n_tokens * inputs.n_head * HEAD_DIM)
        unless baseline_oracle.finite && baseline_oracle.cosine >= COSINE_MIN &&
               baseline_oracle.max_abs <= MAX_ABS_TOL && baseline_oracle.rmse <= RMSE_TOL
          raise "#{case_label}: baseline CPU oracle failed finite=#{baseline_oracle.finite} cosine=#{baseline_oracle.cosine} rmse=#{baseline_oracle.rmse} max_abs=#{baseline_oracle.max_abs}"
        end
      end
      raise "#{case_label}: synthetic prefix rows are not distinct" unless prefix_distinct?(inputs, shape)

      run_flash(flash_pipe, q_buf, gate_buf, k_buf, v_buf, flash_out,
        shape, inputs.n_head, inputs.n_head_kv, heads_per_group)
      flash_values = flash_out.read(output_count)
      assert_flash_equal!(baseline_values, flash_values, shape.n_tokens * inputs.n_head * HEAD_DIM) if flash_reference
      metrics = active_metrics!(baseline_values, flash_values, shape, inputs, case_label)
      if expected = oracle
        flash_oracle = compare(expected, flash_values, shape.n_tokens * inputs.n_head * HEAD_DIM)
        unless flash_oracle.finite && flash_oracle.cosine >= COSINE_MIN &&
               flash_oracle.max_abs <= MAX_ABS_TOL && flash_oracle.rmse <= RMSE_TOL
          raise "#{case_label}: Flash CPU oracle failed finite=#{flash_oracle.finite} cosine=#{flash_oracle.cosine} rmse=#{flash_oracle.rmse} max_abs=#{flash_oracle.max_abs}"
        end
      end
      puts "  #{case_label} finite=#{metrics.finite} cosine=#{metrics.cosine.round(8)} rmse=#{metrics.rmse.round(8)} max_abs=#{metrics.max_abs.round(8)} pass=true"
    end
  end
end

if run_perf
  perf_cases = if perf_only
                 [ShapeCase.new(0, 256), ShapeCase.new(0, 512), ShapeCase.new(0, 1024), ShapeCase.new(0, 2048)]
               else
                 [
                   ShapeCase.new(256, 512),
                   ShapeCase.new(1024, 256),
                   ShapeCase.new(1024, 512),
                   ShapeCase.new(4096, 256),
                   ShapeCase.new(4096, 512),
                 ]
               end
  perf_cases.each do |shape|
    # GQA6 is the target 27B shape; correctness mode separately covers GQA4.
    heads_per_group = 6_i32
    inputs = build_inputs(shape, heads_per_group)
    q_buf = ML::MetalBuffer.from_array(inputs.q)
    gate_buf = ML::MetalBuffer.from_array(inputs.gate)
    k_buf = buffer_from_u16(inputs.k_bits)
    v_buf = buffer_from_u16(inputs.v_bits)
    output_count = inputs.padded_tokens * inputs.n_head * HEAD_DIM
    baseline_out = ML::MetalBuffer.from_array(Array(Float32).new(output_count, SENTINEL))
    flash_out = ML::MetalBuffer.from_array(Array(Float32).new(output_count, SENTINEL))
    baseline_ms = [] of Float64
    flash_ms = [] of Float64
    baseline_gpu_ms = [] of Float64
    flash_gpu_ms = [] of Float64

    warmup.times do
      run_baseline(baseline_pipe, q_buf, gate_buf, k_buf, v_buf, baseline_out,
        shape, shape.n_tokens, inputs.n_head, inputs.n_head_kv, heads_per_group, flash_reference)
      run_flash(flash_pipe, q_buf, gate_buf, k_buf, v_buf, flash_out,
        shape, inputs.n_head, inputs.n_head_kv, heads_per_group)
      run_flash(flash_pipe, q_buf, gate_buf, k_buf, v_buf, flash_out,
        shape, inputs.n_head, inputs.n_head_kv, heads_per_group)
      run_baseline(baseline_pipe, q_buf, gate_buf, k_buf, v_buf, baseline_out,
        shape, shape.n_tokens, inputs.n_head, inputs.n_head_kv, heads_per_group, flash_reference)
    end

    reps.times do
      started = Time.instant
      baseline_gpu_ms << run_baseline(baseline_pipe, q_buf, gate_buf, k_buf, v_buf, baseline_out,
        shape, shape.n_tokens, inputs.n_head, inputs.n_head_kv, heads_per_group, flash_reference)
      baseline_ms << (Time.instant - started).total_milliseconds
      started = Time.instant
      flash_gpu_ms << run_flash(flash_pipe, q_buf, gate_buf, k_buf, v_buf, flash_out,
        shape, inputs.n_head, inputs.n_head_kv, heads_per_group)
      flash_ms << (Time.instant - started).total_milliseconds
      started = Time.instant
      flash_gpu_ms << run_flash(flash_pipe, q_buf, gate_buf, k_buf, v_buf, flash_out,
        shape, inputs.n_head, inputs.n_head_kv, heads_per_group)
      flash_ms << (Time.instant - started).total_milliseconds
      started = Time.instant
      baseline_gpu_ms << run_baseline(baseline_pipe, q_buf, gate_buf, k_buf, v_buf, baseline_out,
        shape, shape.n_tokens, inputs.n_head, inputs.n_head_kv, heads_per_group, flash_reference)
      baseline_ms << (Time.instant - started).total_milliseconds
    end

    baseline_values = baseline_out.read(output_count)
    flash_values = flash_out.read(output_count)
    assert_flash_equal!(baseline_values, flash_values, shape.n_tokens * inputs.n_head * HEAD_DIM) if flash_reference
    metrics = active_metrics!(baseline_values, flash_values, shape, inputs,
      "perf base=#{shape.base_pos} tokens=#{shape.n_tokens}")
    baseline_p50 = percentile(baseline_ms, 0.5)
    flash_p50 = percentile(flash_ms, 0.5)
    baseline_gpu_p50 = percentile(baseline_gpu_ms, 0.5)
    flash_gpu_p50 = percentile(flash_gpu_ms, 0.5)
    puts "  perf base=#{shape.base_pos} tokens=#{shape.n_tokens} correctness_cosine=#{metrics.cosine.round(8)} correctness_rmse=#{metrics.rmse.round(8)} correctness_max_abs=#{metrics.max_abs.round(8)} baseline_p50_ms=#{baseline_p50.round(4)} flash_p50_ms=#{flash_p50.round(4)} speedup=#{(baseline_p50 / flash_p50).round(4)}x timing_gate=none"
    puts "    gpu_interval baseline_p50_ms=#{baseline_gpu_p50.round(4)} flash_p50_ms=#{flash_gpu_p50.round(4)} speedup=#{(baseline_gpu_p50 / flash_gpu_p50).round(4)}x samples_per_path=#{baseline_gpu_ms.size}"
  end
end

puts "admission=PASS"
