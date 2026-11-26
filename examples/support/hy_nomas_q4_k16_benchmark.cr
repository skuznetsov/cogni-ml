require "./hy_nomas_q4_kpanel_benchmark"

# Opt-in companion diagnostic for a 16-wide/8 KiB panel against the exact
# prior 32-wide/16 KiB panel. Production kernels and dispatch remain intact.
module HyNomasQ4K16Benchmark
  extend self
  alias B = HyNomasQ4TileDenseBenchmark
  alias P = HyNomasQ4ReductionProbe

  ROOT            = "/opt/mldata_ssd/cogni-ml/hy-nomas-q4-k16-20261026/reports"
  BASELINE        = "/opt/mldata_ssd/cogni-ml/hy-nomas-q4-tile-dense-20261026/reports/dense-m1280-forward-v1.json"
  BASELINE_SHA    = "af8d416961c1a7a982b5e67826422d911974a0ae510ca4db2f3b9208295caee3"
  ARMS            = ["dense.k16", "dense.k32", "packed.k16", "packed.k32", "mps.resident", "mps.expand"]
  CANDIDATES      = ["dense.k16", "dense.k32", "packed.k16", "packed.k32"]
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
  K16_LAUNCHES    = CANDIDATES.map do |arm|
    width = arm.ends_with?("k32") ? 32 : 16
    kernel = "hy_q4_bench_#{arm.sub('.', '_')}"
    {arm: arm, kernel: kernel, pipeline: kernel, m_tile: 64, n_tile: 64,
     threads: 256, shared_memory_bytes: width * 512, threadgroups: [20, 32, 1], k_groups: 1,
     k_panel: width, k8_mmas_per_output: P::INPUT_DIM // 8, panel_iterations: P::INPUT_DIM // width}
  end

  def packed_source : String
    B.packed_source
  end

  def dense_source : String
    B.dense_source
  end

  # This pass-through freezes the prior K32/K64 source as a separately
  # fingerprintable control. K16 kernels are appended only by k16_source.
  def kpanel_source : String
    HyNomasQ4KPanelBenchmark.kpanel_source
  end

  private def replace_exact(source : String, old : String, replacement : String, count : Int32) : String
    raise "K16 source anchor drift: #{old}" unless source.scan(old).size == count
    source.gsub(old, replacement)
  end

  # K16 uses one Q4_K sub-block (16 coefficients) and two K8 MMA steps per
  # panel. The two guarded loader sets cover the whole 64x16 A and B panels:
  # 64 A loaders and 128 B loaders. Threadgroup publication/reuse barriers
  # and the serial ascending K8 MMA loop are inherited unchanged from B64.
  private def k16_kernel(source : String, name : String, following : String, replacement : String,
                         dense : Bool) : String
    kernel = HyNomasQ4B64ScheduleBenchmark.uniform_kernel(source, name, following, replacement)

    kernel = replace_exact(kernel, "const ushort tidw = tiitg & 127;", "const ushort tidw = tiitg & 63;", 1)
    kernel = replace_exact(kernel, "if (tiitg < 128)", "if (tiitg < 64)", 2)

    # NL1=2 gives 128 input-loader slots. Guard both panel fills so threads
    # 128..255 cannot spill the 2 KiB B panel into the adjacent ping-pong bank.
    kernel = replace_exact(kernel,
      "        {\n            const short sx = (tiitg % MM_NL1);",
      "        if (tiitg < 128) {\n            const short sx = (tiitg % MM_NL1);", 1)
    kernel = replace_exact(kernel,
      "            {\n                const short sx = (tiitg % MM_NL1);",
      "            if (tiitg < 128) {\n                const short sx = (tiitg % MM_NL1);", 1)

    {"MM_NK"          => {"HY_K16_NK", dense ? 6 : 5},
     "MM_NL0"         => {"HY_K16_NL0", 6},
     "MM_NL1"         => {"HY_K16_NL1", 8},
     "MM64_TILE_SIZE" => {"HY_K16_TILE_SIZE", 2},
     "MM64_SA_SIZE"   => {"HY_K16_SA_SIZE", 2}}.each do |old, entry|
      kernel = replace_exact(kernel, old, entry[0], entry[1])
    end

    kernel = replace_exact(kernel,
      "il = (il + 2 < MM_NL) ? il + 2 : il % 2;",
      "il = (il + 1 < MM_NL) ? il + 1 : il % 1;", 2)
    kernel = replace_exact(kernel,
      "xw = (il < 2) ? xw + (2 + MM_NL - 1)/MM_NL : xw;",
      "xw = (il < 1) ? xw + (1 + MM_NL - 1)/MM_NL : xw;", 2)

    unless kernel.scan("if (tiitg < 64)").size == 2 && kernel.scan("if (tiitg < 128)").size == 2 &&
           kernel.scan("threadgroup_barrier(mem_flags::mem_threadgroup);").size == 3 &&
           kernel.scan("simdgroup_multiply_accumulate(mc[i], mb[i/4], ma[i%4], mc[i]);").size == 1 &&
           kernel.scan("HY_K16_NK/8").size == 1
      raise "K16 loader, barrier or ascending MMA anchor drift"
    end
    decode_calls = kernel.scan("dequantize_q4_K_fn(xw, il, temp_a);").size
    unless (dense && decode_calls == 0) || (!dense && decode_calls == 2)
      raise "K16 weight decoder source drift"
    end
    kernel
  end

  def k16_source : String
    source = kpanel_source
    constants = <<-METAL
      constant int HY_K16_NK = 16;
      constant int HY_K16_NL0 = 1;
      constant int HY_K16_NL1 = 2;
      constant int HY_K16_SA_SIZE = 2048;
      constant int HY_K16_SB_SIZE = 2048;
      constant int HY_K16_TILE_SIZE = 4096;
      METAL
    source + constants +
      k16_kernel(source, "hy_q4_bench_dense_b64", "hy_q4_bench_dense_b128_sg8", "hy_q4_bench_dense_k16", true) +
      k16_kernel(source, "simd_mm_q4k_h16_b64", "simd_mm_q4k_h16_b128_sg8", "hy_q4_bench_packed_k16", false)
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

  # Inherit TileDense directly; this avoids allocating the old K-panel owner's
  # candidate banks while retaining the common merge, bias, and MPS controls.
  class Owner < B::Owner
    def initialize(cache : ML::Video::HyNomasQ4Cache::Block, rows : Int32, input : Array(Float32))
      unless rows == 1280 && P::INPUT_DIM == 8192 && P::OUTPUT_DIM == 2048
        raise ArgumentError.new("K16 benchmark requires the complete M1280/N2048/K8192 tile")
      end
      super(cache, rows, input)
      begin
        resources, named = HyNomasQ4K16Benchmark.planned_named_bytes(rows)
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
        source = HyNomasQ4K16Benchmark.k16_source
        K16_LAUNCHES.each do |entry|
          pipeline = ML::Metal::ComputePipeline.new(entry[:pipeline], source)
          unless pipeline.thread_execution_width == 32 && pipeline.max_total_threads_per_threadgroup >= entry[:threads] &&
                 pipeline.static_threadgroup_memory_length + entry[:shared_memory_bytes] <= 32768
            raise "K16 benchmark pipeline exceeds admitted SIMD/threadgroup resource limits"
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
      launch = K16_LAUNCHES.find { |entry| entry[:arm] == arm } || raise "Unknown K16 benchmark arm"
      e.set_pipeline(@pipelines[launch[:kernel]])
      if arm.starts_with?("dense.")
        e.set_buffer(buffer("weight.canonical"), 0)
      else
        e.set_buffer(@cache.resident_payload, 0, offset: P::Q4_WEIGHT_OFFSET)
      end
      e.set_buffer(buffer("input.h16"), 1).set_buffer(buffer("partials"), 2, ML::Metal::BufferAccess::Write)
        .set_value(P::INPUT_DIM.to_u32, 3).set_value(P::OUTPUT_DIM.to_u32, 4).set_value(@rows.to_u32, 5)
        .set_threadgroup_memory(launch[:shared_memory_bytes], 0)
        .dispatch_threadgroups({@rows // launch[:m_tile], P::OUTPUT_DIM // launch[:n_tile], 1}, {launch[:threads], 1, 1})
      merge(e, output, 1)
      bias(e, output) if apply_bias
    end
  end

  def continue_async(chain : ML::Metal::MPSCommandChain) : Nil
    B.continue_async(chain)
  end

  def validate_report_path!(path : String) : String
    raise ArgumentError.new("Unsafe K16 report basename") unless File.basename(path).matches?(/\A[a-z0-9][a-z0-9-]*\.json\z/)
    raise ArgumentError.new("K16 report requires dedicated directory") unless File.realpath(File.dirname(path)) == File.realpath(ROOT)
    raise ArgumentError.new("Refusing existing K16 report/sidecar") if File.info?(path, false) || File.info?(path + ".f32le", false)
    path
  end

  def median(values : Array(Float64)) : Float64
    B.median(values)
  end
end
