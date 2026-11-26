require "./hy_nomas_q4_tile_dense_benchmark"

# Opt-in schedule experiments. Keep threadgroup publication/reuse barriers and
# the per-output full-K reduction. Production kernels and dispatch are intact.
module HyNomasQ4B64ScheduleBenchmark
  extend self
  alias B = HyNomasQ4TileDenseBenchmark
  alias P = HyNomasQ4ReductionProbe

  ROOT            = "/opt/mldata_ssd/cogni-ml/hy-nomas-q4-b64-schedule-20261026/reports"
  BASELINE        = "/opt/mldata_ssd/cogni-ml/hy-nomas-q4-tile-dense-20261026/reports/dense-m1280-forward-v1.json"
  BASELINE_SHA    = "af8d416961c1a7a982b5e67826422d911974a0ae510ca4db2f3b9208295caee3"
  ARMS            = ["dense.b64", "dense.eager", "dense.uniform", "packed.b64", "packed.eager", "packed.uniform", "mps.resident", "mps.expand"]
  CANDIDATES      = ["dense.eager", "packed.eager", "dense.uniform", "packed.uniform"]
  QUALIFICATION   = B::QUALIFICATION + CANDIDATES
  BANKS           = QUALIFICATION.flat_map { |name| ["#{name}.raw", "#{name}.bias"] }
  WARMUP_ROUNDS   = ::HyNomasQ4PairedBenchmark::WARMUP_ROUNDS
  MEASURED_ROUNDS = ::HyNomasQ4PairedBenchmark::MEASURED_ROUNDS
  SAMPLE_COUNT    = (WARMUP_ROUNDS + MEASURED_ROUNDS) * ARMS.size
  METRIC_WORDS    = ::HyNomasQ4PairedBenchmark::METRIC_WORDS
  STATS_WORDS     = ::HyNomasQ4PairedBenchmark::WEIGHT_WORDS + (SAMPLE_COUNT + 1) * METRIC_WORDS
  MAGIC           = ::HyNomasQ4PairedBenchmark::MAGIC
  PACKED_LAUNCHES = ::HyNomasQ4TileReuseBenchmark::PACKED_LAUNCHES
  DENSE_LAUNCHES  = B::DENSE_LAUNCHES
  EAGER_LAUNCHES  = ["dense.eager", "packed.eager"].map do |arm|
    {arm: arm, kernel: "hy_q4_bench_#{arm.sub('.', '_')}", m_tile: 64, n_tile: 64,
     threads: 256, shared_bytes: 16384, threadgroups: [20, 32, 1], k_groups: 1}
  end
  UNIFORM_LAUNCHES = ["dense.uniform", "packed.uniform"].map do |arm|
    {arm: arm, kernel: "hy_q4_bench_#{arm.sub('.', '_')}", m_tile: 64, n_tile: 64,
     threads: 256, shared_bytes: 16384, threadgroups: [20, 32, 1], k_groups: 1}
  end
  CANDIDATE_LAUNCHES = EAGER_LAUNCHES + UNIFORM_LAUNCHES

  def packed_source : String
    B.packed_source
  end

  def dense_source : String
    B.dense_source
  end

  def eager_kernel(source : String, name : String, following : String, replacement : String) : String
    start = source.index("kernel void #{name}(") || raise "Missing B64 schedule source"
    finish = source.index("kernel void #{following}(", start) || raise "Missing B64 schedule boundary"
    kernel = source[start...finish]
    mma = "        threadgroup const half * lsma = sa + 4*64*(sgitg % 2);"
    fill = "        if (iter + 1 < n_iter) {"
    barrier = "        threadgroup_barrier(mem_flags::mem_threadgroup);\n    }\n\n    if (nr0"
    raise "B64 schedule source drift" unless kernel.scan(mma).size == 1 && kernel.scan(fill).size == 1 && kernel.scan(barrier).size == 1
    mma_start = kernel.index(mma).not_nil!
    fill_start = kernel.index(fill).not_nil!
    fill_end = kernel.index(barrier).not_nil!
    raise "B64 schedule order drift" unless mma_start < fill_start && fill_start < fill_end
    # Initial panel publication and each loop-end barrier remain in place.
    # For iter>0 the preceding barrier certifies all reads of nxt have ended;
    # this iteration reads only cur, so filling nxt here cannot overwrite it.
    moved = kernel[0...mma_start] + kernel[fill_start...fill_end] + "\n" + kernel[mma_start...fill_start] + kernel[fill_end..]
    moved.sub("kernel void #{name}(", "kernel void #{replacement}(")
  end

  def schedule_source : String
    source = dense_source
    source + eager_kernel(source, "hy_q4_bench_dense_b64", "hy_q4_bench_dense_b128_sg8", "hy_q4_bench_dense_eager") +
      eager_kernel(source, "simd_mm_q4k_h16_b64", "simd_mm_q4k_h16_b128_sg8", "hy_q4_bench_packed_eager") +
      uniform_kernel(source, "hy_q4_bench_dense_b64", "hy_q4_bench_dense_b128_sg8", "hy_q4_bench_dense_uniform") +
      uniform_kernel(source, "simd_mm_q4k_h16_b64", "simd_mm_q4k_h16_b128_sg8", "hy_q4_bench_packed_uniform")
  end

  def uniform_kernel(source : String, name : String, following : String, replacement : String) : String
    start = source.index("kernel void #{name}(") || raise "Missing uniform B64 source"
    finish = source.index("kernel void #{following}(", start) || raise "Missing uniform B64 boundary"
    kernel = source[start...finish]
    barrier = "            simdgroup_barrier(mem_flags::mem_none);\n"
    raise "Uniform B64 barrier source drift" unless kernel.scan(barrier).size == 3
    # Apple MSL's SIMD-matrix example uses uniform load/load/MMA without these
    # execution-only barriers. Shared-memory publication/reuse still uses the
    # original threadgroup barriers; no per-output MMA or fill order changes.
    kernel.gsub(barrier, "").sub("kernel void #{name}(", "kernel void #{replacement}(")
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
    resources, _ = B.planned_named_bytes(rows)
    out_bytes = rows.to_i64 * P::OUTPUT_DIM * sizeof(Float32)
    resources["stats"] = STATS_WORDS.to_i64 * sizeof(UInt32)
    resources["packed"] = BANKS.size * out_bytes + STATS_WORDS * sizeof(UInt32)
    CANDIDATES.each do |arm|
      ["raw", "bias"].each { |suffix| resources["output.#{arm}.#{suffix}"] = out_bytes }
    end
    {resources, resources.values.sum(0_i64)}
  end

  class Owner < B::Owner
    def initialize(cache : ML::Video::HyNomasQ4Cache::Block, rows : Int32, input : Array(Float32))
      super(cache, rows, input)
      begin
        resources, named = HyNomasQ4B64ScheduleBenchmark.planned_named_bytes(rows)
        ["stats", "packed"].each do |name|
          buffer(name).release
          @buffers[name] = ML::MetalBuffer.new(resources[name])
        end
        CANDIDATES.each do |arm|
          ["raw", "bias"].each do |suffix|
            name = "output.#{arm}.#{suffix}"
            @buffers[name] = ML::MetalBuffer.new(resources[name])
          end
        end
        buffer("stats").write_bytes(Array(UInt32).new(STATS_WORDS, 0_u32).to_unsafe.as(UInt8*), STATS_WORDS * sizeof(UInt32))
        source = HyNomasQ4B64ScheduleBenchmark.schedule_source
        CANDIDATE_LAUNCHES.each { |entry| @pipelines[entry[:kernel]] = ML::Metal::ComputePipeline.new(entry[:kernel], source) }
        @resource_bytes = resources
        @named_bytes = named
      rescue ex
        release_setup_only!
        raise ex
      end
    end

    def encode_setup(chain : ML::Metal::MPSCommandChain) : Nil
      super(chain)
      CANDIDATES.each do |arm|
        e = chain.encoder
        poison(e, "partials", 2 * output_elements)
        poison(e, "output.#{arm}.raw", output_elements)
        output = buffer("output.#{arm}.raw")
        project_candidate(e, arm, output, apply_bias: false)
        copy(e, output, buffer("output.#{arm}.bias"), output_elements, 0)
        bias(e, buffer("output.#{arm}.bias"))
      end
    end

    def encode_sample(chain : ML::Metal::MPSCommandChain, arm : String, repeats : Int32) : Nil
      if CANDIDATES.includes?(arm)
        repeats.times { project_candidate(chain.encoder, arm, buffer("bench.output")) }
      else
        super(chain, arm, repeats)
      end
    end

    def encode_pack(chain : ML::Metal::MPSCommandChain) : Nil
      BANKS.each_with_index { |name, i| copy(chain.encoder, buffer("output.#{name}"), buffer("packed"), output_elements, i * output_elements) }
      copy(chain.encoder, buffer("stats"), buffer("packed"), STATS_WORDS, BANKS.size * output_elements)
    end

    def values(chain : ML::Metal::MPSCommandChain) : Array(Float32)
      check_terminal!(chain)
      buffer("packed").read(BANKS.size * output_elements + STATS_WORDS)
    end

    private def project_candidate(e : ML::Metal::ComputeEncoder, arm : String, output : ML::MetalBuffer, apply_bias = true) : Nil
      launch = CANDIDATE_LAUNCHES.find { |entry| entry[:arm] == arm } || raise "Unknown B64 schedule arm"
      e.set_pipeline(@pipelines[launch[:kernel]])
      if arm.starts_with?("dense.")
        e.set_buffer(buffer("weight.canonical"), 0)
      else
        e.set_buffer(@cache.resident_payload, 0, offset: P::Q4_WEIGHT_OFFSET)
      end
      e.set_buffer(buffer("input.h16"), 1).set_buffer(buffer("partials"), 2, ML::Metal::BufferAccess::Write)
        .set_value(P::INPUT_DIM.to_u32, 3).set_value(P::OUTPUT_DIM.to_u32, 4).set_value(@rows.to_u32, 5)
        .set_threadgroup_memory(launch[:shared_bytes], 0)
        .dispatch_threadgroups({@rows // launch[:m_tile], P::OUTPUT_DIM // launch[:n_tile], 1}, {launch[:threads], 1, 1})
      merge(e, output, 1)
      bias(e, output) if apply_bias
    end
  end

  def continue_async(chain : ML::Metal::MPSCommandChain) : Nil
    B.continue_async(chain)
  end

  def validate_report_path!(path : String) : String
    raise ArgumentError.new("Unsafe B64 schedule report basename") unless File.basename(path).matches?(/\A[a-z0-9][a-z0-9-]*\.json\z/)
    raise ArgumentError.new("B64 schedule report requires dedicated directory") unless File.realpath(File.dirname(path)) == File.realpath(ROOT)
    raise ArgumentError.new("Refusing existing B64 schedule report/sidecar") if File.info?(path, false) || File.info?(path + ".f32le", false)
    path
  end

  def median(values : Array(Float64)) : Float64
    B.median(values)
  end
end
