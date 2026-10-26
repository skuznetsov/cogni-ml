require "./hy_nomas_q4_reduction_probe"

# Opt-in projection benchmark. Timed roots exclude qualification, poisoning,
# output comparison and coverage atomics. All arithmetic stays on GPU.
module HyNomasQ4PairedBenchmark
  extend self
  alias P = HyNomasQ4ReductionProbe

  ROOT            = "/opt/mldata_ssd/cogni-ml/hy-nomas-q4-paired-20261026/reports"
  BASELINE        = "/opt/mldata_ssd/cogni-ml/hy-nomas-q4-direct-split-20261026/reports/packed-v2.json"
  BASELINE_SHA    = "a15740f8114d3b723e5c561cbebcf73317104ddd6701eac498c615444f6c161d"
  ARMS            = ["packed.g1", "mps.resident", "packed.g2", "mps.expand"]
  QUALIFICATION   = ["direct", "dense.g2", "packed.g1", "packed.g2", "mps.resident", "mps.expand"]
  BANKS           = QUALIFICATION.flat_map { |name| ["#{name}.raw", "#{name}.bias"] }
  WARMUP_ROUNDS   = 2
  MEASURED_ROUNDS = 6
  SAMPLE_COUNT    = (WARMUP_ROUNDS + MEASURED_ROUNDS) * ARMS.size
  METRIC_WORDS    = 4
  WEIGHT_WORDS    = 2 * METRIC_WORDS
  STATS_WORDS     = WEIGHT_WORDS + (SAMPLE_COUNT + 1) * METRIC_WORDS
  MAGIC           = 0x50414952_u32

  EXTRA_SOURCE = <<-METAL
    // A geometry probe: repeat the authenticated 23 real rows on GPU. This
    // does not create a new real full-prefix activation distribution.
    kernel void hy_q4_bench_input(device const float *source [[buffer(0)]],
        device half *h16 [[buffer(1)]], device float *f32 [[buffer(2)]],
        constant uint &count [[buffer(3)]], uint gid [[thread_position_in_grid]]) {
      if (gid < count) {
        const half value = half(source[gid % (23u * 8192u)]);
        h16[gid] = value; f32[gid] = float(value);
      }
    }
    // Match the packed GEMM's F16 decoder rounding before widening to F32.
    kernel void hy_q4_bench_expand(device const block_q4_K *packed [[buffer(0)]],
        device float *destination [[buffer(1)]], constant uint &count [[buffer(2)]],
        uint gid [[thread_position_in_grid]]) {
      if (gid >= count / 16u) return;
      const short il = short(gid % 16u);
      half4x4 reg;
      dequantize_q4_K_fn(packed + gid / 16u, il, reg);
      const uint within = 64u * (uint(il) / 4u) + 32u * ((uint(il) & 3u) / 2u) + 16u * (uint(il) & 1u);
      for (uint i = 0; i < 16u; ++i)
        destination[(gid / 16u) * 256u + within + i] = float(reg[i / 4u][i % 4u]);
    }
    kernel void hy_q4_bench_weight_check(device const half *canonical [[buffer(0)]],
        device const float *expanded [[buffer(1)]], device atomic_uint *stats [[buffer(2)]],
        constant uint &count [[buffer(3)]], constant uint &injected [[buffer(4)]],
        uint gid [[thread_position_in_grid]]) {
      if (gid >= count) return;
      uint actual = as_type<uint>(expanded[gid]);
      if (gid == injected) actual ^= 1u;
      if (actual != as_type<uint>(float(canonical[gid])))
        atomic_fetch_add_explicit(stats, 1u, memory_order_relaxed);
      if (!isfinite(expanded[gid]) || !isfinite(canonical[gid]))
        atomic_fetch_add_explicit(stats + 1, 1u, memory_order_relaxed);
      if ((gid & 255u) == 0u) atomic_fetch_add_explicit(stats + 2, 1u, memory_order_relaxed);
      if (gid == count - 1u) atomic_store_explicit(stats + 3, 0x50414952u, memory_order_relaxed);
    }
    kernel void hy_q4_bench_output_check(device const float *actual [[buffer(0)]],
        device const float *reference [[buffer(1)]], device atomic_uint *stats [[buffer(2)]],
        constant uint &count [[buffer(3)]], constant uint &injected [[buffer(4)]],
        uint gid [[thread_position_in_grid]]) {
      if (gid >= count) return;
      uint bits = as_type<uint>(actual[gid]);
      if (gid == injected) bits ^= 1u;
      if (bits != as_type<uint>(reference[gid])) atomic_fetch_add_explicit(stats, 1u, memory_order_relaxed);
      if (!isfinite(actual[gid]) || !isfinite(reference[gid]))
        atomic_fetch_add_explicit(stats + 1, 1u, memory_order_relaxed);
      if ((gid & 255u) == 0u) atomic_fetch_add_explicit(stats + 2, 1u, memory_order_relaxed);
      if (gid == count - 1u) atomic_store_explicit(stats + 3, 0x50414952u, memory_order_relaxed);
    }
    METAL

  def packed_source : String
    source = P.packed_grouped_source
    start = source.index("kernel void hy_q4_packed_grouped_h16(") || raise "Missing packed grouping kernel"
    kernel = source[start..]
    kernel = sub_once(kernel, "kernel void hy_q4_packed_grouped_h16(", "kernel void hy_q4_bench_packed_h16(")
    kernel = sub_once(kernel, "device atomic_uint* coverage [[buffer(7)]],", "")
    # Ignore indentation only when removing the exact instrumentation block.
    pattern = /if \(tgpig.x == 0 && tgpig.y == 0 && tiitg == 0\) \{\s*const uint tile = start_k \/ MM_NK \+ iter;\s*atomic_fetch_add_explicit\(coverage \+ tile, 1u, memory_order_relaxed\);\s*atomic_fetch_or_explicit\(coverage \+ in_dim \/ MM_NK \+ tile, 1u << tgpig.z, memory_order_relaxed\);\s*\}/
    raise "Packed instrumentation source drift" unless kernel.scan(pattern).size == 1
    kernel = kernel.sub(pattern, "")
    raise "Timed packed kernel retains diagnostics" if kernel.includes?("atomic_") || kernel.includes?("coverage")
    raise "Timed decoder changed" unless kernel.scan("dequantize_q4_K_fn(xw, il, temp_a);").size == 2
    source + kernel + EXTRA_SOURCE
  end

  private def sub_once(source : String, needle : String, replacement : String) : String
    raise "Paired benchmark source anchor changed" unless source.scan(needle).size == 1
    source.sub(needle, replacement)
  end

  def schedule(reverse_first : Bool)
    (WARMUP_ROUNDS + MEASURED_ROUNDS).times.flat_map do |round|
      reverse = round.odd? != reverse_first
      (reverse ? ARMS.reverse : ARMS).map_with_index do |arm, position|
        {round: round, position: position, arm: arm, warmup: round < WARMUP_ROUNDS, reverse: reverse}
      end
    end.to_a
  end

  def planned_named_bytes(rows : Int32)
    out_bytes = rows.to_i64 * P::OUTPUT_DIM * sizeof(Float32)
    resources = {
      "cache.q4_payload" => P::CACHE_PAYLOAD_BYTES,
      "input.source"     => P::INPUT_BYTES,
      "input.h16"        => rows.to_i64 * P::INPUT_DIM * sizeof(UInt16),
      "input.f32"        => rows.to_i64 * P::INPUT_DIM * sizeof(Float32),
      "weight.canonical" => P::Q4_BF16_BYTES,
      "weight.stage"     => P::INPUT_DIM.to_i64 * P::OUTPUT_DIM * sizeof(Float32),
      "bias"             => P::BIAS_BYTES,
      "partials"         => 2 * out_bytes,
      "dense.coverage"   => P::GROUP_TILE_COUNT.to_i64 * sizeof(UInt32),
      "bench.output"     => out_bytes,
      "stats"            => STATS_WORDS.to_i64 * sizeof(UInt32),
      "packed"           => BANKS.size * out_bytes + STATS_WORDS * sizeof(UInt32),
    } of String => Int64
    BANKS.each { |name| resources["output.#{name}"] = out_bytes }
    {resources, resources.values.sum(0_i64)}
  end

  class Owner
    getter resource_bytes : Hash(String, Int64)
    getter named_bytes : Int64
    getter? encoded : Bool = false
    @buffers = {} of String => ML::MetalBuffer
    @pipelines = {} of String => ML::Metal::ComputePipeline
    @plans = {} of String => ML::Video::WanT5MPSLinearPlan
    @chain : ML::Metal::MPSCommandChain? = nil
    @released = false

    def initialize(@cache : ML::Video::HyNomasQ4Cache::Block, @rows : Int32, input : Array(Float32))
      @resource_bytes, @named_bytes = HyNomasQ4PairedBenchmark.planned_named_bytes(@rows)
      begin
        @cache.load!
        @resource_bytes.each { |name, bytes| @buffers[name] = ML::MetalBuffer.new(bytes) unless name == "cache.q4_payload" }
        buffer("input.source").write(input)
        buffer("stats").write_bytes(Array(UInt32).new(STATS_WORDS, 0_u32).to_unsafe.as(UInt8*), STATS_WORDS * sizeof(UInt32))
        buffer("dense.coverage").write_bytes(Array(UInt32).new(P::GROUP_TILE_COUNT, 0_u32).to_unsafe.as(UInt8*), P::GROUP_TILE_COUNT * sizeof(UInt32))
        source = HyNomasQ4PairedBenchmark.packed_source
        ["simd_mm_q4k_h16", "hy_q4_grouped_h16", "hy_q4_bench_packed_h16", "hy_q4_bench_input", "hy_q4_bench_expand", "hy_q4_bench_weight_check", "hy_q4_bench_output_check"].each do |name|
          @pipelines[name] = ML::Metal::ComputePipeline.new(name, source)
        end
        ["hy_q4_canonical", "hy_q4_copy", "hy_q4_poison_partials", "hy_q4_merge_partials"].each do |name|
          @pipelines[name] = ML::Metal::ComputePipeline.new(name, P::DIAGNOSTIC_SOURCE)
        end
        ML::Video::HyNomasQ4Cache.prepare_decode!
        ML::Video::MetalLinearMPS.prepare!
        @pipelines["bias"] = ML::Video::MetalLinearMPS.pipeline
        ["mps.resident.raw", "mps.expand.raw"].each { |name| create_plan(name, buffer("output.#{name}")) }
        create_plan("bench", buffer("bench.output"))
      rescue ex
        release_setup_only!
        raise ex
      end
    end

    def buffer(name : String) : ML::MetalBuffer
      @buffers[name]? || raise "Missing paired benchmark buffer #{name}"
    end

    def encode_setup(chain : ML::Metal::MPSCommandChain) : Nil
      raise "Paired owner already encoded/released" if @encoded || @released
      @encoded = true
      @chain = chain
      e = chain.encoder
      e.set_pipeline(@pipelines["hy_q4_bench_input"]).set_buffer(buffer("input.source"), 0)
        .set_buffer(buffer("input.h16"), 1, ML::Metal::BufferAccess::Write)
        .set_buffer(buffer("input.f32"), 2, ML::Metal::BufferAccess::Write)
        .set_value((@rows * P::INPUT_DIM).to_u32, 3).dispatch_1d(@rows * P::INPUT_DIM)
      e.set_pipeline(ML::Video::HyNomasQ4Cache.copy_pipeline)
        .set_buffer(@cache.resident_payload, 0, offset: P::BIAS_OFFSET)
        .set_buffer(buffer("bias"), 1, ML::Metal::BufferAccess::Write)
        .set_value(P::OUTPUT_DIM.to_u32, 2).dispatch_1d(P::OUTPUT_DIM)
      e.set_pipeline(@pipelines["hy_q4_canonical"])
        .set_buffer(@cache.resident_payload, 0, offset: P::Q4_WEIGHT_OFFSET)
        .set_buffer(buffer("weight.canonical"), 1, ML::Metal::BufferAccess::Write)
        .set_value((P::INPUT_DIM * P::OUTPUT_DIM).to_u32, 2).dispatch_1d(P::INPUT_DIM * P::OUTPUT_DIM)
      expand(e)
      2.times do |i|
        e.set_pipeline(@pipelines["hy_q4_bench_weight_check"])
          .set_buffer(buffer("weight.canonical"), 0).set_buffer(buffer("weight.stage"), 1)
          .set_buffer(buffer("stats"), 2, ML::Metal::BufferAccess::ReadWrite, offset: i.to_i64 * METRIC_WORDS * 4)
          .set_value((P::INPUT_DIM * P::OUTPUT_DIM).to_u32, 3)
          .set_value(i == 0 ? UInt32::MAX : 17_u32, 4).dispatch_1d(P::INPUT_DIM * P::OUTPUT_DIM)
      end
      QUALIFICATION.each do |arm|
        poison(e, "partials", 2 * output_elements)
        poison(e, "output.#{arm}.raw", output_elements)
        output = buffer("output.#{arm}.raw")
        case arm
        when "direct"
          simd(e, "simd_mm_q4k_h16", output, 1)
        when "dense.g2"
          simd(e, "hy_q4_grouped_h16", buffer("partials"), 2, dense: true)
          merge(e, output, 2)
        else
          project(e, arm, output, arm + ".raw", apply_bias: false)
        end
        copy(e, output, buffer("output.#{arm}.bias"), output_elements, 0)
        bias(e, buffer("output.#{arm}.bias"))
      end
    end

    def prepare_sample(chain : ML::Metal::MPSCommandChain) : Nil
      poison(chain.encoder, "partials", 2 * output_elements)
      poison(chain.encoder, "bench.output", output_elements)
    end

    def encode_sample(chain : ML::Metal::MPSCommandChain, arm : String, repeats : Int32) : Nil
      repeats.times { project(chain.encoder, arm, buffer("bench.output"), "bench") }
    end

    def check_sample(chain : ML::Metal::MPSCommandChain, arm : String, index : Int32, injected = false) : Nil
      e = chain.encoder
      e.set_pipeline(@pipelines["hy_q4_bench_output_check"])
        .set_buffer(buffer("bench.output"), 0).set_buffer(buffer("output.#{arm}.bias"), 1)
        .set_buffer(buffer("stats"), 2, ML::Metal::BufferAccess::ReadWrite, offset: (WEIGHT_WORDS + index * METRIC_WORDS).to_i64 * 4)
        .set_value(output_elements.to_u32, 3).set_value(injected ? 17_u32 : UInt32::MAX, 4).dispatch_1d(output_elements)
    end

    def encode_pack(chain : ML::Metal::MPSCommandChain) : Nil
      BANKS.each_with_index { |name, i| copy(chain.encoder, buffer("output.#{name}"), buffer("packed"), output_elements, i * output_elements) }
      copy(chain.encoder, buffer("stats"), buffer("packed"), STATS_WORDS, BANKS.size * output_elements)
    end

    def values(chain : ML::Metal::MPSCommandChain) : Array(Float32)
      check_terminal!(chain)
      buffer("packed").read(BANKS.size * output_elements + STATS_WORDS)
    end

    def release_after_terminal!(chain : ML::Metal::MPSCommandChain) : Nil
      check_terminal!(chain)
      release_owned!
    end

    def release_setup_only! : Nil
      raise "Cannot release in-flight paired resources" if @encoded && !@chain.try(&.completed?)
      release_owned!
    end

    private def check_terminal!(chain : ML::Metal::MPSCommandChain) : Nil
      raise "Paired resources need their sole terminal wait" unless @encoded && chain.completed? && chain.wait_count == 1 && @chain.try(&.same?(chain))
    end

    private def output_elements : Int32
      @rows * P::OUTPUT_DIM
    end

    private def create_plan(name : String, output : ML::MetalBuffer) : Nil
      config = ML::Video::WanT5Config.new(vocab: 1, dim: 1, ffn_dim: 1, heads: 1, layers: 1, batches: 1, tokens: @rows)
      @plans[name] = ML::Video::WanT5MPSLinearPlan.new(config, P::INPUT_DIM, P::OUTPUT_DIM, buffer("input.f32"), buffer("weight.stage"), output)
    end

    private def expand(e : ML::Metal::ComputeEncoder) : Nil
      e.set_pipeline(@pipelines["hy_q4_bench_expand"])
        .set_buffer(@cache.resident_payload, 0, offset: P::Q4_WEIGHT_OFFSET)
        .set_buffer(buffer("weight.stage"), 1, ML::Metal::BufferAccess::Write)
        .set_value((P::INPUT_DIM * P::OUTPUT_DIM).to_u32, 2).dispatch_1d(P::INPUT_DIM * P::OUTPUT_DIM // 16)
    end

    private def simd(e : ML::Metal::ComputeEncoder, kernel : String, output : ML::MetalBuffer, groups : Int32, dense = false) : Nil
      e.set_pipeline(@pipelines[kernel])
      if dense
        e.set_buffer(buffer("weight.canonical"), 0).set_value(groups.to_u32, 6).set_value(0_u32, 7)
          .set_buffer(buffer("dense.coverage"), 8, ML::Metal::BufferAccess::ReadWrite)
      else
        e.set_buffer(@cache.resident_payload, 0, offset: P::Q4_WEIGHT_OFFSET)
        e.set_value(groups.to_u32, 6) unless kernel == "simd_mm_q4k_h16"
      end
      e.set_buffer(buffer("input.h16"), 1).set_buffer(output, 2, ML::Metal::BufferAccess::Write)
        .set_value(P::INPUT_DIM.to_u32, 3).set_value(P::OUTPUT_DIM.to_u32, 4).set_value(@rows.to_u32, 5)
        .set_threadgroup_memory(P::MM_SHMEM, 0)
        .dispatch_threadgroups({(@rows + P::MM_NR1 - 1) // P::MM_NR1, P::OUTPUT_DIM // P::MM_NR0, groups}, {P::MM_TG, 1, 1})
    end

    private def merge(e : ML::Metal::ComputeEncoder, output : ML::MetalBuffer, groups : Int32) : Nil
      e.set_pipeline(@pipelines["hy_q4_merge_partials"]).set_buffer(buffer("partials"), 0)
        .set_buffer(output, 1, ML::Metal::BufferAccess::Write)
        .set_value(output_elements.to_u32, 2).set_value(groups.to_u32, 3).set_value(0_u32, 4).dispatch_1d(output_elements)
    end

    private def project(e : ML::Metal::ComputeEncoder, arm : String, output : ML::MetalBuffer, plan_name : String, apply_bias = true) : Nil
      case arm
      when "packed.g1", "packed.g2"
        groups = arm == "packed.g1" ? 1 : 2
        simd(e, "hy_q4_bench_packed_h16", buffer("partials"), groups)
        merge(e, output, groups)
      when "mps.resident", "mps.expand"
        expand(e) if arm == "mps.expand"
        ML::Video::MetalLinearMPS.project_preexpanded(e, buffer("input.f32"), buffer("bias"), output,
          @rows, P::INPUT_DIM, P::OUTPUT_DIM, ML::Video::WeightDType::BF16, @plans[plan_name], buffer("weight.stage"), apply_bias: false)
      else
        raise "Unknown projection arm #{arm}"
      end
      bias(e, output) if apply_bias
    end

    private def bias(e : ML::Metal::ComputeEncoder, output : ML::MetalBuffer) : Nil
      e.set_pipeline(@pipelines["bias"]).set_buffer(output, 0, ML::Metal::BufferAccess::ReadWrite)
        .set_buffer(buffer("bias"), 1).set_buffer(buffer("bias"), 2)
        .set_value(P::OUTPUT_DIM.to_u32, 3).set_value(1_u32, 4).set_value(output_elements.to_u32, 5).dispatch_1d(output_elements)
    end

    private def poison(e : ML::Metal::ComputeEncoder, name : String, count : Int32) : Nil
      e.set_pipeline(@pipelines["hy_q4_poison_partials"]).set_buffer(buffer(name), 0, ML::Metal::BufferAccess::Write)
        .set_value(count.to_u32, 1).dispatch_1d(count)
    end

    private def copy(e : ML::Metal::ComputeEncoder, source : ML::MetalBuffer, destination : ML::MetalBuffer, count : Int32, offset : Int32) : Nil
      e.set_pipeline(@pipelines["hy_q4_copy"]).set_buffer(source, 0).set_buffer(destination, 1, ML::Metal::BufferAccess::Write)
        .set_value(count.to_u32, 2).set_value(offset.to_u32, 3).dispatch_1d(count)
    end

    private def release_owned! : Nil
      return if @released
      @plans.each_value(&.release)
      @plans.clear
      @buffers.each_value { |b| b.release if b.valid? }
      @buffers.clear
      @cache.release
      @released = true
    end
  end

  def continue_async(chain : ML::Metal::MPSCommandChain) : Nil
    before = chain.observed_root_count
    chain.pause!
    chain.continue_async!
    chain.resume!
    raise "Expected exactly one asynchronous next root" unless chain.observed_root_count == before + 1 && chain.wait_count == 0
  end

  def validate_report_path!(path : String) : String
    basename = File.basename(path)
    raise ArgumentError.new("Unsafe paired report basename") unless basename.matches?(/\A[a-z0-9][a-z0-9-]*\.json\z/)
    raise ArgumentError.new("Paired report requires dedicated directory") unless File.realpath(File.dirname(path)) == File.realpath(ROOT)
    raise ArgumentError.new("Refusing existing paired report/sidecar") if File.info?(path, false) || File.info?(path + ".f32le", false)
    path
  end

  def median(values : Array(Float64)) : Float64
    sorted = values.sort
    (sorted[(sorted.size - 1) // 2] + sorted[sorted.size // 2]) / 2.0
  end
end
