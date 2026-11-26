require "./hy_nomas_q4_paired_benchmark"

# Change only M reuse: retain the full-K reduction, one-group merge and bias.
# The inherited owner holds all controls and scratch through the final wait.
module HyNomasQ4TileReuseBenchmark
  extend self
  alias B = HyNomasQ4PairedBenchmark
  alias P = HyNomasQ4ReductionProbe

  ROOT            = "/opt/mldata_ssd/cogni-ml/hy-nomas-q4-tile-reuse-20261026/reports"
  BASELINE        = "/opt/mldata_ssd/cogni-ml/hy-nomas-q4-paired-20261026/reports/paired-m1280-forward-v1.json"
  BASELINE_SHA    = "6cbe310fb1fdd619e715f440c09daac99e308eb5ca9f61eade459a9e0abd15c6"
  ARMS            = ["packed.g1", "packed.b64", "mps.resident", "packed.b128", "mps.expand"]
  CANDIDATES      = ["packed.b64", "packed.b128"]
  QUALIFICATION   = B::QUALIFICATION + CANDIDATES
  BANKS           = QUALIFICATION.flat_map { |name| ["#{name}.raw", "#{name}.bias"] }
  WARMUP_ROUNDS   = B::WARMUP_ROUNDS
  MEASURED_ROUNDS = B::MEASURED_ROUNDS
  SAMPLE_COUNT    = (WARMUP_ROUNDS + MEASURED_ROUNDS) * ARMS.size
  METRIC_WORDS    = B::METRIC_WORDS
  WEIGHT_WORDS    = B::WEIGHT_WORDS
  STATS_WORDS     = WEIGHT_WORDS + (SAMPLE_COUNT + 1) * METRIC_WORDS
  MAGIC           = B::MAGIC
  PACKED_LAUNCHES = [
    {arm: "packed.g1", kernel: "hy_q4_bench_packed_h16", m_tile: 32, n_tile: 64,
     threads: 128, shared_bytes: 12288, threadgroups: [40, 32, 1], k_groups: 1},
    {arm: "packed.b64", kernel: "simd_mm_q4k_h16_b64", m_tile: 64, n_tile: 64,
     threads: 256, shared_bytes: 16384, threadgroups: [20, 32, 1], k_groups: 1},
    {arm: "packed.b128", kernel: "simd_mm_q4k_h16_b128_sg8", m_tile: 128, n_tile: 64,
     threads: 256, shared_bytes: 24576, threadgroups: [10, 32, 1], k_groups: 1},
  ]

  def packed_source : String
    B.packed_source
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
      raise ArgumentError.new("Tile reuse requires exact M1280 tiles") unless rows == 1280
      super(cache, rows, input)
      begin
        resources, named = HyNomasQ4TileReuseBenchmark.planned_named_bytes(rows)
        # Replace terminal diagnostic buffers before any command is encoded.
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
        ["simd_mm_q4k_h16_b64", "simd_mm_q4k_h16_b128_sg8"].each do |name|
          @pipelines[name] = ML::Metal::ComputePipeline.new(name, HyNomasQ4TileReuseBenchmark.packed_source)
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
        project_reuse(e, arm, output, apply_bias: false)
        copy(e, output, buffer("output.#{arm}.bias"), output_elements, 0)
        bias(e, buffer("output.#{arm}.bias"))
      end
    end

    def encode_sample(chain : ML::Metal::MPSCommandChain, arm : String, repeats : Int32) : Nil
      if CANDIDATES.includes?(arm)
        repeats.times { project_reuse(chain.encoder, arm, buffer("bench.output")) }
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

    private def project_reuse(e : ML::Metal::ComputeEncoder, arm : String, output : ML::MetalBuffer, apply_bias = true) : Nil
      launch = PACKED_LAUNCHES.find { |entry| entry[:arm] == arm } || raise "Unknown tile reuse arm"
      e.set_pipeline(@pipelines[launch[:kernel]])
        .set_buffer(@cache.resident_payload, 0, offset: P::Q4_WEIGHT_OFFSET)
        .set_buffer(buffer("input.h16"), 1).set_buffer(buffer("partials"), 2, ML::Metal::BufferAccess::Write)
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
    raise ArgumentError.new("Unsafe tile reuse report basename") unless File.basename(path).matches?(/\A[a-z0-9][a-z0-9-]*\.json\z/)
    raise ArgumentError.new("Tile reuse report requires dedicated directory") unless File.realpath(File.dirname(path)) == File.realpath(ROOT)
    raise ArgumentError.new("Refusing existing tile reuse report/sidecar") if File.info?(path, false) || File.info?(path + ".f32le", false)
    path
  end

  def median(values : Array(Float64)) : Float64
    B.median(values)
  end
end
