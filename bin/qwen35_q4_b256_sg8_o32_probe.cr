#!/usr/bin/env crystal
# Isolated O32 x B256 Q4_K falsifier. Never changes the production shader.
require "option_parser"
require "json"
require "digest/sha256"
require "../src/ml/gguf/qwen35_metal"

IN_DIM         =  5120
OUT_DIM        = 17408
BASELINE_NAME  = "simd_mm_q4k_h16_b128_sg8"
CANDIDATE_NAME = "simd_mm_q4k_h16_b256_sg8_o32"

# The candidate keeps the production B128 kernel's 256-thread launch and
# sixteen live 8x8 accumulators per SIMD group. It trades a narrower O32 tile
# for B256 reuse: one Q4 weight tile is dequantized once for 256 input rows.
# The 18-KiB single staging tile needs an extra uniform barrier before each
# overwrite; the operator gate decides whether the saved dequantization pays.
CANDIDATE_SOURCE = <<-METAL
kernel void #{CANDIDATE_NAME}(
    device const uint8_t* w_raw   [[buffer(0)]],
    device const half*    x       [[buffer(1)]],
    device       float*   output  [[buffer(2)]],
    constant     uint&    in_dim  [[buffer(3)]],
    constant     uint&    out_dim [[buffer(4)]],
    constant     uint&    batch   [[buffer(5)]],
    threadgroup  char*    shmem   [[threadgroup(0)]],
    uint3  tgpig [[threadgroup_position_in_grid]],
    ushort tiitg [[thread_index_in_threadgroup]],
    ushort sgitg [[simdgroup_index_in_threadgroup]])
{
    const int nr0 = 32;
    const int nr1 = 256;
    const int sa_half_count = nr0 * MM_NK;
    const int sb_block_half_count = 128 * MM_NK;
    threadgroup half * sa = (threadgroup half *)shmem;
    threadgroup half * sb = sa + sa_half_count;

    const int r0 = tgpig.y * nr0;
    const int r1 = tgpig.x * nr1;
    if (r0 + nr0 > (int)out_dim || r1 + nr1 > (int)batch) return;

    const ushort tidw = tiitg & 63;
    const short lr0 = (short)(tidw / MM_NL0);
    const short lr1 = (short)(tiitg / MM_NL1);
    const short il0 = tidw % MM_NL0;
    short il = il0;

    const uint row_bytes = (in_dim / QK_K) * 144;
    device const block_q4_K * xw =
        (device const block_q4_K *)(w_raw + (r0 + lr0) * row_bytes) + il0 / MM_NL;

    const short iy = 8 * (tiitg % MM_NL1);
    device const half * y0 = x + (r1 + lr1) * in_dim + iy;
    device const half * y1 = y0 + 64 * in_dim;
    device const half * y2 = y0 + 128 * in_dim;
    device const half * y3 = y0 + 192 * in_dim;

    simdgroup_half8x8  ma[4];
    simdgroup_half8x8  mb0[2];
    simdgroup_half8x8  mb1[2];
    simdgroup_float8x8 mc0[8];
    simdgroup_float8x8 mc1[8];
    FOR_UNROLL for (short i = 0; i < 8; i++) {
        mc0[i] = make_filled_simdgroup_matrix<float, 8>(0.f);
        mc1[i] = make_filled_simdgroup_matrix<float, 8>(0.f);
    }

    if (tiitg < 64) {
        half4x4 temp_a;
        dequantize_q4_K_fn(xw, il, temp_a);
        FOR_UNROLL for (short i = 0; i < 16; i++) {
            const short sx = 2*il0 + i/8;
            const short sy = (tidw/MM_NL0)/8;
            const short lx = (tidw/MM_NL0)%8;
            const short ly = i%8;
            // O32 has four 8-row output tiles. Keep each 8x8 matrix
            // contiguous (64 halfs), while compacting the B128 kernel's
            // eight output tiles down to four.
            *(sa + 64*(4*sx + sy) + 8*ly + lx) = temp_a[i/4][i%4];
        }
    }
    {
        const short sx = tiitg % MM_NL1;
        const short sy = (tiitg/MM_NL1)/8;
        const short ly = (tiitg/MM_NL1)%8;
        threadgroup half * sb_hi = sb + sb_block_half_count;
        threadgroup half * dst00 = sb    + 64*(16*sx + sy)     + 8*ly;
        threadgroup half * dst01 = sb    + 64*(16*sx + sy + 8) + 8*ly;
        threadgroup half * dst10 = sb_hi + 64*(16*sx + sy)     + 8*ly;
        threadgroup half * dst11 = sb_hi + 64*(16*sx + sy + 8) + 8*ly;
        *(threadgroup half2x4 *)dst00 = *(device const half2x4 *)y0;
        *(threadgroup half2x4 *)dst01 = *(device const half2x4 *)y1;
        *(threadgroup half2x4 *)dst10 = *(device const half2x4 *)y2;
        *(threadgroup half2x4 *)dst11 = *(device const half2x4 *)y3;
    }
    il = (il + 2 < MM_NL) ? il + 2 : il % 2;
    xw = (il < 2) ? xw + (2 + MM_NL - 1)/MM_NL : xw;
    y0 += MM_NK;
    y1 += MM_NK;
    y2 += MM_NK;
    y3 += MM_NK;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    const uint n_iter = (in_dim + MM_NK - 1) / MM_NK;
    for (uint iter = 0; iter < n_iter; iter++) {
        threadgroup const half * lsma = sa;
        const short block = sgitg / 4;
        const short group = sgitg % 4;
        threadgroup const half * sb_sg = sb + block * sb_block_half_count;
        threadgroup const half * lsmb0 = sb_sg + 2*64*group;
        threadgroup const half * lsmb1 = sb_sg + 2*64*(group + 4);
        FOR_UNROLL for (short ik = 0; ik < MM_NK/8; ik++) {
            simdgroup_barrier(mem_flags::mem_none);
            FOR_UNROLL for (short i = 0; i < 4; i++) simdgroup_load(ma[i], lsma + 64*i, 8, 0, false);
            simdgroup_barrier(mem_flags::mem_none);
            FOR_UNROLL for (short i = 0; i < 2; i++) {
                simdgroup_load(mb0[i], lsmb0 + 64*i, 8, 0, false);
                simdgroup_load(mb1[i], lsmb1 + 64*i, 8, 0, false);
            }
            simdgroup_barrier(mem_flags::mem_none);
            FOR_UNROLL for (short i = 0; i < 8; i++) {
                simdgroup_multiply_accumulate(mc0[i], mb0[i/4], ma[i%4], mc0[i]);
                simdgroup_multiply_accumulate(mc1[i], mb1[i/4], ma[i%4], mc1[i]);
            }
            lsma += 4*64;
            lsmb0 += 16*64;
            lsmb1 += 16*64;
        }

        if (iter + 1 < n_iter) {
            // The activation tile is single-buffered. All eight SIMD groups
            // must finish reading before any thread overwrites it.
            threadgroup_barrier(mem_flags::mem_threadgroup);
            if (tiitg < 64) {
                half4x4 temp_a;
                dequantize_q4_K_fn(xw, il, temp_a);
                FOR_UNROLL for (short i = 0; i < 16; i++) {
                    const short sx = 2*il0 + i/8;
                    const short sy = (tidw/MM_NL0)/8;
                    const short lx = (tidw/MM_NL0)%8;
                    const short ly = i%8;
                    *(sa + 64*(4*sx + sy) + 8*ly + lx) = temp_a[i/4][i%4];
                }
            }
            {
                const short sx = tiitg % MM_NL1;
                const short sy = (tiitg/MM_NL1)/8;
                const short ly = (tiitg/MM_NL1)%8;
                threadgroup half * sb_hi = sb + sb_block_half_count;
                threadgroup half * dst00 = sb    + 64*(16*sx + sy)     + 8*ly;
                threadgroup half * dst01 = sb    + 64*(16*sx + sy + 8) + 8*ly;
                threadgroup half * dst10 = sb_hi + 64*(16*sx + sy)     + 8*ly;
                threadgroup half * dst11 = sb_hi + 64*(16*sx + sy + 8) + 8*ly;
                *(threadgroup half2x4 *)dst00 = *(device const half2x4 *)y0;
                *(threadgroup half2x4 *)dst01 = *(device const half2x4 *)y1;
                *(threadgroup half2x4 *)dst10 = *(device const half2x4 *)y2;
                *(threadgroup half2x4 *)dst11 = *(device const half2x4 *)y3;
            }
            il = (il + 2 < MM_NL) ? il + 2 : il % 2;
            xw = (il < 2) ? xw + (2 + MM_NL - 1)/MM_NL : xw;
            y0 += MM_NK;
            y1 += MM_NK;
            y2 += MM_NK;
            y3 += MM_NK;
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
    }

    const int block = sgitg / 4;
    const int group = sgitg % 4;
    device float * c0 = output + r0 + (r1 + block*128 + group*16)*out_dim;
    device float * c1 = c0 + 64*out_dim;
    FOR_UNROLL for (short i = 0; i < 8; i++) {
        simdgroup_store(mc0[i], c0 + 8*(i%4) + 8*out_dim*(i/4), out_dim, 0, false);
        simdgroup_store(mc1[i], c1 + 8*(i%4) + 8*out_dim*(i/4), out_dim, 0, false);
    }
}
METAL

private def source : String
  base = ML::GGUF::Qwen35Metal::GEMM_Q4K_SOURCE
  raise "source drift: missing baseline" unless base.includes?("kernel void #{BASELINE_NAME}(")
  raise "candidate collision" if base.includes?("kernel void #{CANDIDATE_NAME}(")
  base + "\n" + CANDIDATE_SOURCE
end

private def rejects!(&block : ->) : Nil
  accepted = true
  begin
    yield
  rescue
    accepted = false
  end
  raise "negative control was accepted" if accepted
end

private def input_values(batch : Int32) : Array(UInt16)
  Array(UInt16).new(batch * IN_DIM) do |i|
    hash = (i.to_i64 * 1103515245 + 12345) & 0xffffffff_i64
    (((hash >> 16) & 0x83ff) | 0x2800).to_u16
  end
end

private def exact_f32!(a : Slice(Float32), b : Slice(Float32)) : Nil
  raise "F32 size mismatch" unless a.size == b.size
  nonzero = false
  a.each_with_index do |value, i|
    other = b[i]
    raise "F32 mismatch/nonfinite at #{i}" unless value.finite? && other.finite? && value.unsafe_as(UInt32) == other.unsafe_as(UInt32)
    nonzero ||= value != 0
  end
  raise "empty F32 output" unless nonzero
end

private def complete_once!(counts : Array(Int32), label : String) : Nil
  counts.each_with_index do |count, index|
    raise "#{label} cell #{index} has coverage #{count}" unless count == 1
  end
end

private def verify_tile_layout! : Nil
  # The compact O32 weight tile must write every half exactly once. This is a
  # model-free oracle for the indexing that differs from production B128.
  weight_writes = Array(Int32).new(32 * 32, 0)
  64.times do |tidw|
    il0 = tidw % 2
    row = tidw // 2
    16.times do |i|
      sx = 2 * il0 + i // 8
      sy = row // 8
      lx = row % 8
      ly = i % 8
      index = 64 * (4 * sx + sy) + 8 * ly + lx
      raise "weight write out of range" unless index < weight_writes.size
      weight_writes[index] += 1
    end
  end
  complete_once!(weight_writes, "weight write")

  weight_reads = Array(Int32).new(32 * 32, 0)
  4.times do |iteration|
    4.times do |matrix|
      base = iteration * 4 * 64 + matrix * 64
      64.times { |lane| weight_reads[base + lane] += 1 }
    end
  end
  complete_once!(weight_reads, "weight MMA read")

  # Four H16 source bands fill two independent 128-row staging blocks.
  activation_writes = Array(Int32).new(256 * 32, 0)
  256.times do |thread|
    sx = thread % 4
    sy = (thread // 4) // 8
    ly = (thread // 4) % 8
    {0, 128 * 32}.each do |block|
      {0, 8}.each do |band|
        base = block + 64 * (16 * sx + sy + band) + 8 * ly
        8.times { |lane| activation_writes[base + lane] += 1 }
      end
    end
  end
  complete_once!(activation_writes, "activation write")

  activation_reads = Array(Int32).new(256 * 32, 0)
  8.times do |simdgroup|
    block = (simdgroup // 4) * 128 * 32
    group = simdgroup % 4
    4.times do |iteration|
      {group, group + 4}.each do |band|
        2.times do |matrix|
          base = block + iteration * 16 * 64 + 2 * 64 * band + matrix * 64
          64.times { |lane| activation_reads[base + lane] += 1 }
        end
      end
    end
  end
  complete_once!(activation_reads, "activation MMA read")
end

private def median(values : Array(Float64)) : Float64
  sorted = values.sort
  (sorted[(sorted.size - 1) // 2] + sorted[sorted.size // 2]) / 2
end

private def run_kernel(pipeline : ML::Metal::ComputePipeline, weight : ML::MetalBuffer,
                       input : ML::MetalBuffer, output : ML::MetalBuffer,
                       batch : Int32, candidate : Bool) : Float64
  cmd = ML::Metal::CommandBuffer.new
  enc = ML::Metal::ComputeEncoder.new(cmd)
  enc.set_pipeline(pipeline)
  enc.set_buffer(weight, 0)
  enc.set_buffer(input, 1)
  enc.set_buffer(output, 2, ML::Metal::BufferAccess::Write)
  enc.set_value(IN_DIM.to_u32, 3)
  enc.set_value(OUT_DIM.to_u32, 4)
  enc.set_value(batch.to_u32, 5)
  enc.set_threadgroup_memory(candidate ? 18432 : 24576, 0)
  enc.dispatch_threadgroups(candidate ? {batch // 256, OUT_DIM // 32, 1} : {batch // 128, OUT_DIM // 64, 1}, {256, 1, 1})
  enc.end_encoding
  elapsed = cmd.commit_and_wait_gpu_elapsed_seconds * 1000
  raise "invalid GPU interval" unless elapsed.finite? && elapsed > 0
  elapsed
end

private def self_test : Nil
  text = source
  raise "missing candidate" unless text.scan("kernel void #{CANDIDATE_NAME}(").size == 1
  verify_tile_layout!
  values = input_values(2)
  raise "fixture rows repeat" if values[0, IN_DIM] == values[IN_DIM, IN_DIM]
  exact_f32!(Slice[1.0_f32, -0.5_f32], Slice[1.0_f32, -0.5_f32])
  rejects! { exact_f32!(Slice[1.0_f32], Slice[1.0001_f32]) }
  rejects! { exact_f32!(Slice[Float32::NAN], Slice[Float32::NAN]) }
  rejects! { exact_f32!(Slice[0.0_f32], Slice[0.0_f32]) }
  puts "self_test=true tile_layout_complete=true source_drift_guard=true corruption_nan_empty_rejected=true"
end

mode = ""
batch = 256
model = "/Users/sergey/.cache/lm-studio/models/lmstudio-community/Qwen3.8-27B-GGUF/Qwen3.8-27B-Q4_K_M.gguf"
OptionParser.parse do |p|
  {"self-test", "compile-only", "run"}.each do |option|
    p.on("--#{option}", option) { raise "choose one mode" unless mode.empty?; mode = option }
  end
  p.on("--batch=N", "256 or 512") { |v| batch = v.to_i32 }
  p.on("--model=PATH", "Qwen3.8 GGUF") { |v| model = v }
  p.on("--help", "Help") { puts p; exit }
end
raise "choose --self-test, --compile-only or --run" if mode.empty?
raise "unsupported batch" unless {256, 512}.includes?(batch)
self_test
exit if mode == "self-test"
raise "set COGNI_RUN_SAFE_ACTIVE=1 and use scripts/run_safe.sh" if mode == "run" && ENV["COGNI_RUN_SAFE_ACTIVE"]? != "1"

shader = source
baseline = ML::Metal::ComputePipeline.new(BASELINE_NAME, shader)
candidate = ML::Metal::ComputePipeline.new(CANDIDATE_NAME, shader)
puts({event: "pipeline", baseline_max_threads: baseline.max_total_threads_per_threadgroup,
      candidate_max_threads: candidate.max_total_threads_per_threadgroup,
      baseline_shmem: 24576, candidate_shmem: 18432}.to_json)
raise "candidate cannot launch 256 threads" if candidate.max_total_threads_per_threadgroup < 256
puts({event: "source", device: ML::Metal::Device.instance.name,
      baseline_sha256: Digest::SHA256.hexdigest(ML::GGUF::Qwen35Metal::GEMM_Q4K_SOURCE),
      candidate_sha256: Digest::SHA256.hexdigest(shader), model_loaded: false}.to_json)
exit if mode == "compile-only"

raw = Bytes.empty
gguf = ML::GGUF::GGUFFile.new(model, mmap_tensors: false)
begin
  info = gguf.tensor("blk.0.ffn_gate.weight") || raise "missing gate tensor"
  raise "unsupported tensor shape/type" unless info.type.q4_k? && info.dims == [IN_DIM.to_i64, OUT_DIM.to_i64]
  raw = gguf.read_tensor_raw(info)
ensure
  gguf.close
end
raise "invalid Q4 byte count" unless raw.size == OUT_DIM * (IN_DIM // 256) * 144
puts({event: "tensor", name: "blk.0.ffn_gate.weight", bytes: raw.size,
      sha256: Digest::SHA256.hexdigest(raw), mmap_tensors: false}.to_json)
weight = ML::MetalBuffer.new(raw.size.to_i64)
weight.write_bytes(raw.to_unsafe, raw.size)
values = input_values(batch)
input_bytes = Slice.new(values.to_unsafe.as(Pointer(UInt8)), values.size * 2)
puts({event: "input", batch: batch, sha256: Digest::SHA256.hexdigest(input_bytes), first_two_rows_distinct: true}.to_json)
input = ML::MetalBuffer.new(input_bytes.size.to_i64)
input.write_bytes(input_bytes.to_unsafe, input_bytes.size)
count = batch * OUT_DIM
outputs = Array.new(2) { ML::MetalBuffer.new(count.to_i64 * 4) }
run = ->(arm : Int32) { run_kernel(arm == 0 ? baseline : candidate, weight, input, outputs[arm], batch, arm == 1) }
check = -> { exact_f32!(Slice.new(outputs[0].contents.as(Pointer(Float32)), count), Slice.new(outputs[1].contents.as(Pointer(Float32)), count)) }
poison = -> { outputs.each { |buf| Slice.new(buf.contents.as(Pointer(Float32)), count).fill(Float32::NAN) } }

poison.call
run.call(0)
run.call(1)
check.call
5.times { {0, 1, 1, 0}.each { |arm| run.call(arm) } }
baseline_times, candidate_times = [] of Float64, [] of Float64
wins = 0
10.times do |cycle|
  a0, b0, b1, a1 = run.call(0), run.call(1), run.call(1), run.call(0)
  baseline_times.concat([a0, a1])
  candidate_times.concat([b0, b1])
  wins += 1 if b0 + b1 < a0 + a1
  puts({event: "cycle", batch: batch, cycle: cycle, baseline_first_ms: a0,
        candidate_first_ms: b0, candidate_second_ms: b1, baseline_second_ms: a1}.to_json)
end
check.call
poison.call
run.call(1)
run.call(0)
check.call
baseline_median = median(baseline_times)
candidate_median = median(candidate_times)
puts({event: "summary", batch: batch, warmups: 5, cycles: 10, elements: count,
      finite_bitwise_equal: true, baseline_median_ms: baseline_median,
      candidate_median_ms: candidate_median,
      time_reduction_pct: 100 * (1 - candidate_median / baseline_median),
      candidate_wins: wins, promotion_local_pct: 10.0, promotion_wins: 8,
      promotion_passed: candidate_median <= baseline_median * 0.90 && wins >= 8}.to_json)
