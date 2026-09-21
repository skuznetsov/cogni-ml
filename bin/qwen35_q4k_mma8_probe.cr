#!/usr/bin/env crystal
# Isolated Q4_K decode mixed-MMA falsifier. Never changes production routing.
require "option_parser"
require "json"
require "digest/sha256"
require "../src/ml/gguf/qwen35_metal"

IN_DIM               =       5120
OUT_DIM              =      17408
QK_K                 =        256
Q4K_BLOCK_BYTES      =        144
ROWS_PER_THREADGROUP =          8
SIMD_WIDTH           =         32
PROMOTION_LOCAL_PCT  =       9.86
PROMOTION_WINS_PCT   =       80.0
NUMERIC_TOLERANCE    = 1.0e-3_f32
BASELINE_NAME        = "simd_mv_q4k_f32_x16"
CANDIDATE_NAME       = "simd_mv_q4k_f32_mma8"

# One SIMD group computes eight output rows. For each aligned 32-value Q4_K
# scale segment, four 8x8 mixed MMAs multiply exact half-representable nibbles
# by F32 activations. Only column zero is useful, so this intentionally spends
# 8x the arithmetic to test whether the matrix engine and fourfold SIMD-group
# reduction can overcome scalar GEMV overhead. Scale/min correction remains F32.
CANDIDATE_SOURCE = <<-METAL
kernel void #{CANDIDATE_NAME}(
    device const uint8_t* w_raw   [[buffer(0)]],
    device const float*   x       [[buffer(1)]],
    device       float*   output  [[buffer(2)]],
    constant     uint&    in_dim  [[buffer(3)]],
    constant     uint&    out_dim [[buffer(4)]],
    constant     uint&    batch   [[buffer(5)]],
    threadgroup  char*    shmem   [[threadgroup(0)]],
    uint3  tgpig [[threadgroup_position_in_grid]],
    ushort tiisg [[thread_index_in_simdgroup]])
{
    const uint first_row = tgpig.x * #{ROWS_PER_THREADGROUP};
    const uint n = tgpig.y;
    if (first_row >= out_dim || n >= batch) return;

    threadgroup half * weights = (threadgroup half *)shmem;
    threadgroup float * input_matrix =
        (threadgroup float *)(weights + 64);
    threadgroup float * product = input_matrix + 64;

    const uint blocks_per_row = in_dim / QK_K;
    const uint row_bytes = blocks_per_row * #{Q4K_BLOCK_BYTES};
    device const float * input_row = x + n * in_dim;
    float sum = 0.0f;

    // The seven unused columns stay zero for the entire dispatch. Only column
    // zero changes between K tiles; rewriting all 64 cells in the hot loop
    // would add 14 unnecessary F32 stores per useful activation pair.
    if (tiisg < 8) {
        const uint base = tiisg * 8;
        FOR_UNROLL for (uint column = 1; column < 8; ++column) {
            input_matrix[base + column] = 0.0f;
        }
    }
    simdgroup_barrier(mem_flags::mem_threadgroup);

    for (uint segment = 0; segment < blocks_per_row * 8; ++segment) {
        const uint block_index = segment >> 3;
        const uint subblock = segment & 7;
        const uint q_group = subblock >> 1;
        const bool high_nibble = (subblock & 1) != 0;
        const uint input_base = segment * 32;
        const float input_sum = simd_sum(input_row[input_base + tiisg]);
        simdgroup_float8x8 acc = make_filled_simdgroup_matrix<float, 8>(0.0f);

        FOR_UNROLL for (uint tile = 0; tile < 4; ++tile) {
            const uint weight_index0 = tiisg;
            const uint weight_index1 = tiisg + 32;
            const uint weight_row0 = weight_index0 >> 3;
            const uint weight_row1 = weight_index1 >> 3;
            const uint k0 = weight_index0 & 7;
            const uint k1 = weight_index1 & 7;
            const uint q_index0 = q_group * 32 + tile * 8 + k0;
            const uint q_index1 = q_group * 32 + tile * 8 + k1;
            device const block_q4_K * block0 =
                (device const block_q4_K *)(w_raw + (first_row + weight_row0) * row_bytes) + block_index;
            device const block_q4_K * block1 =
                (device const block_q4_K *)(w_raw + (first_row + weight_row1) * row_bytes) + block_index;
            const uchar packed0 = block0->qs[q_index0];
            const uchar packed1 = block1->qs[q_index1];
            weights[weight_index0] = half(high_nibble ? packed0 >> 4 : packed0 & 0x0f);
            weights[weight_index1] = half(high_nibble ? packed1 >> 4 : packed1 & 0x0f);

            if (tiisg < 8) {
                const uint base = tiisg * 8;
                input_matrix[base] = input_row[input_base + tile * 8 + tiisg];
            }
            simdgroup_barrier(mem_flags::mem_threadgroup);

            simdgroup_half8x8 weight_matrix;
            simdgroup_float8x8 x_matrix;
            simdgroup_load(weight_matrix, weights, 8);
            simdgroup_load(x_matrix, input_matrix, 8);
            simdgroup_multiply_accumulate(acc, weight_matrix, x_matrix, acc);
            simdgroup_barrier(mem_flags::mem_threadgroup);
        }

        simdgroup_store(acc, product, 8);
        simdgroup_barrier(mem_flags::mem_threadgroup);
        if (tiisg < 8 && first_row + tiisg < out_dim) {
            device const block_q4_K * block =
                (device const block_q4_K *)(w_raw + (first_row + tiisg) * row_bytes) + block_index;
            const uchar2 scale_min = get_scale_min_k4_scalar(subblock, block->scales);
            sum += float(block->d) * float(scale_min.x) * product[tiisg * 8] -
                   float(block->dmin) * float(scale_min.y) * input_sum;
        }
        simdgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (tiisg < 8 && first_row + tiisg < out_dim) {
        output[n * out_dim + first_row + tiisg] = sum;
    }
}
METAL

record Timing, wall_ms : Float64, gpu_ms : Float64

private def source : String
  base = ML::GGUF::Qwen35Metal::GEMM_Q4K_SOURCE
  raise "source drift: missing baseline" unless base.includes?("kernel void #{BASELINE_NAME}(")
  raise "candidate collision" if base.includes?("kernel void #{CANDIDATE_NAME}(")
  base + "\n" + CANDIDATE_SOURCE
end

private def median(values : Array(Float64)) : Float64
  sorted = values.sort
  (sorted[(sorted.size - 1) // 2] + sorted[sorted.size // 2]) / 2
end

private def input_values : Array(Float32)
  Array(Float32).new(IN_DIM) do |i|
    ((((i.to_i64 * 1103515245_i64 + 12345_i64) & 0xffff_i64) / 32768.0) - 1.0).to_f32
  end
end

private def verify_layout! : Nil
  covered = Array(Int32).new(QK_K, 0)
  8.times do |subblock|
    4.times do |tile|
      8.times do |k|
        within = subblock * 32 + tile * 8 + k
        covered[within] += 1
        group = subblock // 2
        packed_index = group * 32 + tile * 8 + k
        raise "Q4 packed index out of range" unless packed_index.in?(0...128)
      end
    end
  end
  covered.each_with_index do |count, index|
    raise "Q4 element #{index} has coverage #{count}" unless count == 1
  end

  matrix_cells = Array(Int32).new(64, 0)
  SIMD_WIDTH.times do |lane|
    matrix_cells[lane] += 1
    matrix_cells[lane + SIMD_WIDTH] += 1
  end
  raise "matrix staging is incomplete" unless matrix_cells.all?(&.==(1))
end

private def max_abs_diff(a : Array(Float32), b : Array(Float32)) : Float32
  raise "output size mismatch" unless a.size == b.size
  max = 0.0_f32
  a.each_with_index do |value, i|
    other = b[i]
    raise "non-finite output at #{i}" unless value.finite? && other.finite?
    diff = (value - other).abs
    max = diff if diff > max
  end
  max
end

private def baseline_once(qw : ML::GGUF::QuantWeight,
                          input : ML::MetalBuffer,
                          output : ML::MetalBuffer) : Timing
  ENV["QWEN35_Q4K_GEMV_X16"] = "1"
  cmd = ML::Metal::CommandBuffer.new
  enc = ML::Metal::ComputeEncoder.new(cmd)
  raise "baseline is not Metal routable" unless ML::GGUF::Qwen35Metal.encode_matmul_to_buffer(enc, qw, input, output, 1)
  enc.end_encoding
  started = Time.instant
  gpu_ms = cmd.commit_and_wait_gpu_elapsed_seconds * 1_000.0
  Timing.new((Time.instant - started).total_milliseconds, gpu_ms)
end

private def candidate_once(pipeline : ML::Metal::ComputePipeline,
                           weight : ML::MetalBuffer,
                           input : ML::MetalBuffer,
                           output : ML::MetalBuffer) : Timing
  cmd = ML::Metal::CommandBuffer.new
  enc = ML::Metal::ComputeEncoder.new(cmd)
  enc.set_pipeline(pipeline)
  enc.set_buffer(weight, 0)
  enc.set_buffer(input, 1)
  enc.set_buffer(output, 2, ML::Metal::BufferAccess::Write)
  enc.set_value(IN_DIM.to_u32, 3)
  enc.set_value(OUT_DIM.to_u32, 4)
  enc.set_value(1_u32, 5)
  enc.set_threadgroup_memory(640, 0)
  enc.dispatch_threadgroups({(OUT_DIM + ROWS_PER_THREADGROUP - 1) // ROWS_PER_THREADGROUP, 1, 1}, {SIMD_WIDTH, 1, 1})
  enc.end_encoding
  started = Time.instant
  gpu_ms = cmd.commit_and_wait_gpu_elapsed_seconds * 1_000.0
  Timing.new((Time.instant - started).total_milliseconds, gpu_ms)
end

private def load_q4_weight(gguf : ML::GGUF::GGUFFile,
                           tensor_name : String) : ML::GGUF::QuantWeight
  info = gguf.tensor(tensor_name) || raise "missing #{tensor_name}"
  unless info.type.q4_k? && info.dims == [IN_DIM.to_i64, OUT_DIM.to_i64]
    raise "unsupported #{tensor_name} shape/type #{info.dims} #{info.type.name}"
  end
  raw = gguf.read_tensor_raw(info)
  expected = OUT_DIM * (IN_DIM // QK_K) * Q4K_BLOCK_BYTES
  raise "invalid #{tensor_name} byte count #{raw.size}, expected #{expected}" unless raw.size == expected
  ML::GGUF::QuantWeight.new(raw, info.type, OUT_DIM, IN_DIM, tensor_name,
    ML::GGUF::Q4GemvX16Capability::Qwen38)
end

private def benchmark(label : String,
                      qw : ML::GGUF::QuantWeight,
                      pipeline : ML::Metal::ComputePipeline,
                      input : ML::MetalBuffer,
                      warmup : Int32,
                      pairs : Int32) : Nil
  baseline_output = ML::MetalBuffer.new(OUT_DIM.to_i64 * sizeof(Float32))
  candidate_output = ML::MetalBuffer.new(OUT_DIM.to_i64 * sizeof(Float32))
  weight = qw.fallback_metal_buffer

  warmup.times do
    baseline_once(qw, input, baseline_output)
    candidate_once(pipeline, weight, input, candidate_output)
  end

  baseline_times = [] of Timing
  candidate_times = [] of Timing
  pair_wins = 0
  pairs.times do |cycle|
    if cycle.even?
      a = baseline_once(qw, input, baseline_output)
      b = candidate_once(pipeline, weight, input, candidate_output)
    else
      b = candidate_once(pipeline, weight, input, candidate_output)
      a = baseline_once(qw, input, baseline_output)
    end
    baseline_times << a
    candidate_times << b
    pair_wins += 1 if b.gpu_ms < a.gpu_ms
    puts({event: "pair", op: label, cycle: cycle, baseline_gpu_ms: a.gpu_ms,
          candidate_gpu_ms: b.gpu_ms, baseline_wall_ms: a.wall_ms,
          candidate_wall_ms: b.wall_ms}.to_json)
  end

  baseline_once(qw, input, baseline_output)
  candidate_once(pipeline, weight, input, candidate_output)
  diff = max_abs_diff(baseline_output.read(OUT_DIM), candidate_output.read(OUT_DIM))
  baseline_gpu = baseline_times.map(&.gpu_ms)
  candidate_gpu = candidate_times.map(&.gpu_ms)
  baseline_wall = baseline_times.map(&.wall_ms)
  candidate_wall = candidate_times.map(&.wall_ms)
  baseline_gpu_p50 = median(baseline_gpu)
  candidate_gpu_p50 = median(candidate_gpu)
  reduction = 100.0 * (1.0 - candidate_gpu_p50 / baseline_gpu_p50)
  wins_required = (pairs * PROMOTION_WINS_PCT / 100.0).ceil.to_i
  numeric_passed = diff <= NUMERIC_TOLERANCE
  promotion_passed = numeric_passed && reduction >= PROMOTION_LOCAL_PCT && pair_wins >= wins_required

  puts({event: "summary", op: label, in_dim: IN_DIM, out_dim: OUT_DIM,
        baseline: BASELINE_NAME, candidate: CANDIDATE_NAME,
        baseline_gpu_p50_ms: baseline_gpu_p50,
        candidate_gpu_p50_ms: candidate_gpu_p50,
        gpu_time_reduction_pct: reduction,
        baseline_wall_p50_ms: median(baseline_wall),
        candidate_wall_p50_ms: median(candidate_wall),
        candidate_wins: pair_wins, pairs: pairs, wins_required: wins_required,
        max_abs_diff: diff, numeric_tolerance: NUMERIC_TOLERANCE,
        numeric_passed: numeric_passed,
        promotion_local_pct: PROMOTION_LOCAL_PCT,
        promotion_passed: promotion_passed}.to_json)
end

mode = ""
model = ENV["QWEN35_MODEL"]? || ""
warmup = 3
pairs = 20
OptionParser.parse do |parser|
  {"self-test", "compile-only", "run"}.each do |option|
    parser.on("--#{option}", option) { raise "choose one mode" unless mode.empty?; mode = option }
  end
  parser.on("--model=PATH", "Qwen3.8 GGUF path") { |value| model = value }
  parser.on("--warmup=N", "Warmup rounds (default: 3)") { |value| warmup = value.to_i }
  parser.on("--pairs=N", "Alternating measured pairs per operator (default: 20)") { |value| pairs = value.to_i }
  parser.on("-h", "--help", "Show help") { puts parser; exit }
end

raise "choose --self-test, --compile-only or --run" if mode.empty?
raise "--warmup must be non-negative" if warmup < 0
raise "--pairs must be positive" unless pairs > 0
verify_layout!
shader = source
puts({event: "self_test", q4_layout_complete: true, matrix_staging_complete: true,
      source_drift_guard: true}.to_json)
exit if mode == "self-test"

raise "Metal not available" unless ML::GGUF::Qwen35Metal.available?
candidate = ML::Metal::ComputePipeline.new(CANDIDATE_NAME, shader)
raise "candidate cannot launch one SIMD group" if candidate.max_total_threads_per_threadgroup < SIMD_WIDTH
puts({event: "pipeline", candidate: CANDIDATE_NAME,
      max_threads: candidate.max_total_threads_per_threadgroup,
      threadgroup_memory_bytes: 640,
      production_source_sha256: Digest::SHA256.hexdigest(ML::GGUF::Qwen35Metal::GEMM_Q4K_SOURCE),
      candidate_source_sha256: Digest::SHA256.hexdigest(shader),
      model_loaded: false}.to_json)
exit if mode == "compile-only"

raise "--model is required" if model.empty?
raise "set COGNI_RUN_SAFE_ACTIVE=1 and use scripts/run_safe.sh" unless ENV["COGNI_RUN_SAFE_ACTIVE"]? == "1"
input = ML::MetalBuffer.from_array(input_values)
original_x16 = ENV["QWEN35_Q4K_GEMV_X16"]?
gguf = ML::GGUF::GGUFFile.new(model, mmap_tensors: false)
begin
  puts({event: "run", model: model, warmup: warmup, pairs: pairs,
        device: ML::Metal::Device.instance.name}.to_json)
  benchmark("gate", load_q4_weight(gguf, "blk.0.ffn_gate.weight"), candidate, input, warmup, pairs)
  benchmark("up", load_q4_weight(gguf, "blk.0.ffn_up.weight"), candidate, input, warmup, pairs)
ensure
  gguf.close
  if value = original_x16
    ENV["QWEN35_Q4K_GEMV_X16"] = value
  else
    ENV.delete("QWEN35_Q4K_GEMV_X16")
  end
end
