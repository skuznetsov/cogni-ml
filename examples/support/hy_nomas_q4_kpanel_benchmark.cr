require "./hy_nomas_q4_b64_schedule_benchmark"

# Opt-in K-panel experiment with the same full-K K8 MMA sequence.
# Production kernels and dispatch are intact; only complete tiles are admitted.
module HyNomasQ4KPanelBenchmark
  extend self
  alias B = HyNomasQ4TileDenseBenchmark
  alias P = HyNomasQ4ReductionProbe

  ROOT            = "/opt/mldata_ssd/cogni-ml/hy-nomas-q4-kpanel-20261026/reports"
  BASELINE        = "/opt/mldata_ssd/cogni-ml/hy-nomas-q4-tile-dense-20261026/reports/dense-m1280-forward-v1.json"
  BASELINE_SHA    = "af8d416961c1a7a982b5e67826422d911974a0ae510ca4db2f3b9208295caee3"
  ARMS            = ["dense.k32", "dense.k64", "packed.k32", "packed.k64", "mps.resident", "mps.expand"]
  CANDIDATES      = ["dense.k32", "dense.k64", "packed.k32", "packed.k64"]
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
  KPANEL_LAUNCHES = CANDIDATES.map do |arm|
    width = arm.ends_with?("k64") ? 64 : 32
    {arm: arm, kernel: "hy_q4_bench_#{arm.sub('.', '_')}", m_tile: 64, n_tile: 64,
     threads: 256, shared_bytes: width * 512, threadgroups: [20, 32, 1], k_groups: 1,
     k_panel: width, k8_mmas_per_output: P::INPUT_DIM // 8, panel_iterations: P::INPUT_DIM // width}
  end

  def packed_source : String
    B.packed_source
  end

  def dense_source : String
    B.dense_source
  end

  private def replace_exact(source : String, old : String, replacement : String, count : Int32) : String
    raise "K-panel source anchor drift: #{old}" unless source.scan(old).size == count
    source.gsub(old, replacement)
  end

  def panel_kernel(source : String, name : String, following : String, replacement : String, width : Int32) : String
    kernel = HyNomasQ4B64ScheduleBenchmark.uniform_kernel(source, name, following, replacement)
    return kernel if width == 32
    raise "Unsupported K panel" unless width == 64
    # A K64 weight panel assigns 16 coefficients to each of all 256 threads.
    # Input assignment retains the K32 row mapping and adds the second K32 band.
    kernel = replace_exact(kernel, "const ushort tidw = tiitg & 127;", "const ushort tidw = tiitg;", 1)
    kernel = replace_exact(kernel, "if (tiitg < 128)", "if (tiitg < 256)", 2)
    kernel = replace_exact(kernel, "il = (il + 2 < MM_NL) ? il + 2 : il % 2;", "il = (il + 4 < MM_NL) ? il + 4 : il % 4;", 2)
    kernel = replace_exact(kernel, "xw = (il < 2) ? xw + (2 + MM_NL - 1)/MM_NL : xw;", "xw = (il < 4) ? xw + (4 + MM_NL - 1)/MM_NL : xw;", 2)
    load = "*(threadgroup half2x4 *)dst = *(device const half2x4 *)y;"
    kernel = replace_exact(kernel, load, load + "\n            *(threadgroup half2x4 *)(dst + 4*8*64) = *(device const half2x4 *)(y + 32);", 2)
    # Only the clone sees these constants. Global K32 kernels stay byte-for-byte.
    {"MM64_TILE_SIZE" => "HY_K64_TILE_SIZE", "MM64_SA_SIZE" => "HY_K64_SA_SIZE",
     "MM_NL0" => "HY_K64_NL0", "MM_NK" => "HY_K64_NK"}.each do |old, value|
      kernel = kernel.gsub(old, value)
    end
    kernel
  end

  def kpanel_source : String
    source = dense_source
    constants = "\nconstant int HY_K64_NK = 64;\nconstant int HY_K64_NL0 = 4;\nconstant int HY_K64_SA_SIZE = 8192;\nconstant int HY_K64_TILE_SIZE = 16384;\n"
    source + constants + panel_kernel(source, "hy_q4_bench_dense_b64", "hy_q4_bench_dense_b128_sg8", "hy_q4_bench_dense_k32", 32) +
      panel_kernel(source, "hy_q4_bench_dense_b64", "hy_q4_bench_dense_b128_sg8", "hy_q4_bench_dense_k64", 64) +
      panel_kernel(source, "simd_mm_q4k_h16_b64", "simd_mm_q4k_h16_b128_sg8", "hy_q4_bench_packed_k32", 32) +
      panel_kernel(source, "simd_mm_q4k_h16_b64", "simd_mm_q4k_h16_b128_sg8", "hy_q4_bench_packed_k64", 64)
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
      unless rows == 1280 && P::INPUT_DIM % 64 == 0 && P::OUTPUT_DIM % 64 == 0
        raise ArgumentError.new("K-panel benchmark requires complete M64/N64/K64 tiles")
      end
      super(cache, rows, input)
      begin
        resources, named = HyNomasQ4KPanelBenchmark.planned_named_bytes(rows)
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
        source = HyNomasQ4KPanelBenchmark.kpanel_source
        KPANEL_LAUNCHES.each do |entry|
          pipeline = ML::Metal::ComputePipeline.new(entry[:kernel], source)
          unless pipeline.thread_execution_width == 32 && pipeline.max_total_threads_per_threadgroup >= entry[:threads] &&
                 pipeline.static_threadgroup_memory_length + entry[:shared_bytes] <= 32768
            raise "K-panel pipeline exceeds admitted SIMD/threadgroup resource limits"
          end
          @pipelines[entry[:kernel]] = pipeline
        end
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
      launch = KPANEL_LAUNCHES.find { |entry| entry[:arm] == arm } || raise "Unknown K-panel arm"
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
    raise ArgumentError.new("Unsafe K-panel report basename") unless File.basename(path).matches?(/\A[a-z0-9][a-z0-9-]*\.json\z/)
    raise ArgumentError.new("K-panel report requires dedicated directory") unless File.realpath(File.dirname(path)) == File.realpath(ROOT)
    raise ArgumentError.new("Refusing existing K-panel report/sidecar") if File.info?(path, false) || File.info?(path + ".f32le", false)
    path
  end

  def median(values : Array(Float64)) : Float64
    B.median(values)
  end
end
