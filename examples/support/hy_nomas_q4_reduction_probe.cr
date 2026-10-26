require "digest/sha256"
require "json"
require "option_parser"
require "set"
require "./hy_nomas_probe_instrumentation"
require "../../src/ml/video/hy_nomas_q4_cache"
require "../../src/ml/video/metal_linear_mps"
require "../../src/ml/gguf/qwen35_metal"

module HyNomasQ4ReductionProbe
  extend self

  ROOT                   = "/opt/mldata_ssd/cogni-ml/hy-nomas-q4-reduction-20261025/reports"
  GROUP_ROOT             = "/opt/mldata_ssd/cogni-ml/hy-nomas-q4-grouping-20261026/reports"
  FIXTURE_PATH           = "/opt/mldata_ssd/cogni-ml/hy-nomas-text-conditioning-20261025/fixtures/text-conditioning-v2.json"
  FIXTURE_SHA256         = "167dd498c75efa396ca1f8ae33b68fa34368d2e4c96feb5d4de2d89c2666992e"
  FIXTURE_SIDECAR_SHA256 = "1668ef1699da4d66cc8d88005fb7c3e3a288feba53999d3019a81af5dfb7b849"
  FIXTURE_SIDECAR_BYTES  = 15_605_764_i64
  UPSTREAM_REVISION      = "911cea6458a89c2663513ff7099561ec190dd491"
  CACHE_MANIFEST         = "/opt/mldata_ssd/cogni-ml/hy-nomas-q4-dit-20261025/weights/q4-cache-v2/block-0-q4k.json"
  CACHE_MANIFEST_SHA256  = "d692a899855a813a2c2d92ed5f31782f856398c8dc004feb171bde3b1bb04d3f"
  SOURCE_SHA256          = "73819bcf3b82b090116dd1a1748795de95eef703c304e7443963d00ea8aa4d70"
  SOURCE_BYTES           = 34_213_991_280_i64
  CACHE_PAYLOAD_SHA256   = "7c4ab766d687187d6f2895c3521a1a15a90799a14171ec17ccf2eb2f02055b12"
  CACHE_PAYLOAD_BYTES    = 85_058_560_i64
  BLOCK_INDEX            =          0_i32
  INPUT_DIM              =       8192_i32
  OUTPUT_DIM             =       2048_i32
  ROWS                   =         23_i32
  INPUT_ELEMENTS         = INPUT_DIM.to_i64 * ROWS
  OUTPUT_ELEMENTS        = OUTPUT_DIM.to_i64 * ROWS
  INPUT_BYTES            = INPUT_ELEMENTS * sizeof(Float32)
  OUTPUT_BYTES           = OUTPUT_ELEMENTS * sizeof(Float32)
  Q4_WEIGHT_OFFSET       = 61_441_024_i64
  Q4_WEIGHT_BYTES        =  9_437_184_i64
  Q4_BF16_BYTES          = INPUT_DIM.to_i64 * OUTPUT_DIM * sizeof(UInt16)
  BIAS_OFFSET            = 61_436_928_i64
  BIAS_BYTES             = OUTPUT_DIM.to_i64 * sizeof(UInt16)
  TOLERANCE              =   2e-5_f64
  ATOL                   =   2e-5_f32
  RTOL                   =   2e-5_f32
  MM_NR0                 =     64_i32
  MM_NR1                 =     32_i32
  MM_TG                  =    128_i32
  MM_SHMEM               = 12_288_i32
  PREFIXES               = [32, 256, 2048, 4096, 6144]
  STATS_COUNT            = 10
  DECODE_STATS_WORDS     = 3 * STATS_COUNT
  GROUP_CASES            = [false, true].flat_map do |interleaved|
    [1, 2, 4, 8, 16, 32].map do |groups|
      {name: "#{interleaved ? "strided32" : "contiguous"}.g#{groups}", groups: groups, interleaved: interleaved}
    end
  end
  GROUP_MERGES         = ["fold", "tree"]
  GROUP_TILE_COUNT     = INPUT_DIM // 32
  GROUP_COVERAGE_WORDS = GROUP_CASES.size * GROUP_TILE_COUNT
  STATS_WORDS          = DECODE_STATS_WORDS + GROUP_COVERAGE_WORDS
  GROUP_OUTPUTS        = GROUP_CASES.flat_map { |c| GROUP_MERGES.flat_map { |m| ["group.#{c[:name]}.#{m}.raw", "group.#{c[:name]}.#{m}.bias"] } }
  OUTPUT_VARIANTS      = ["direct.raw", "direct.bias", "dense.raw", "dense.bias",
                          "mps.canonical.raw", "mps.canonical.bias", "mps.helper.raw", "mps.helper.bias",
                          "compensated.raw", "compensated.bias"] + PREFIXES.flat_map { |k| ["dense.k#{k}", "mps.k#{k}"] } + GROUP_OUTPUTS
  DIAGNOSTIC_SOURCE = ML::Video::HyNomasQ4Cache::SOURCE + {{ read_file("#{__DIR__}/hy_nomas_q4_reduction.metal") }}
  HELPER_SOURCE     = <<-METAL
    kernel void hy_q4_dump_helper(device const block_q4_K *packed [[buffer(0)]],
        device half *destination [[buffer(1)]], constant uint &count [[buffer(2)]],
        uint gid [[thread_position_in_grid]]) {
      if (gid >= count / 16u) return;
      const uint block_index = gid / 16u;
      const short il = short(gid % 16u);
      half4x4 reg;
      dequantize_q4_K_fn(packed + block_index, il, reg);
      const uint within = 64u * (uint(il) / 4u) + 32u * ((uint(il) & 3u) / 2u) + 16u * (uint(il) & 1u);
      for (uint i = 0; i < 16u; ++i)
        destination[block_index * 256u + within + i] = reg[i / 4u][i % 4u];
    }
    METAL

  # Change weight loading only; preserve SIMD geometry, MMA order and edge stores.
  # Refuse source drift rather than silently deriving a different control.
  def dense_source(source : String = ML::GGUF::Qwen35Metal::GEMM_Q4K_SOURCE) : String
    start = source.index("kernel void simd_mm_q4k_h16(") || raise "Missing F16 kernel"
    finish = source.index("kernel void simd_mm_q4k_h16_b48(", start) || raise "Missing F16 kernel boundary"
    kernel = source[start...finish]
    helper_call = "dequantize_q4_K_fn(xw, il, temp_a);"
    raise "Unexpected F16 decode load count" unless kernel.scan(helper_call).size == 2
    first = "FOR_UNROLL for (short d = 0; d < 16; ++d) temp_a[d/4][d%4] = ((device const half *)w_raw)[(r0 + lr0) * in_dim + 16 * il0 + d];"
    following = "FOR_UNROLL for (short d = 0; d < 16; ++d) temp_a[d/4][d%4] = ((device const half *)w_raw)[(r0 + lr0) * in_dim + (iter + 1) * MM_NK + 16 * il0 + d];"
    kernel = kernel.sub(helper_call, first).sub(helper_call, following)
    iteration = "const uint n_iter = (in_dim + MM_NK - 1) / MM_NK;"
    binding = "constant     uint&    batch   [[buffer(5)]],"
    raise "Unexpected F16 iteration/binding anchor" unless kernel.scan(iteration).size == 1 && kernel.scan(binding).size == 1
    kernel = kernel.sub(iteration, "const uint n_iter = reduction_k / MM_NK;")
      .sub(binding, binding + "\n    constant uint& reduction_k [[buffer(6)]],")
      .sub("kernel void simd_mm_q4k_h16(", "kernel void hy_q4_dense_h16(")
    source + HELPER_SOURCE + kernel
  end

  # Preserve the existing SIMD MMA, row strides and edge stores. Partition only
  # the ascending sequence of 32-element K tiles, then merge on GPU separately.
  def grouped_source : String
    source = dense_source
    start = source.index("kernel void hy_q4_dense_h16(") || raise "Missing dense control"
    kernel = source[start..]
    kernel = sub_once(kernel, "kernel void hy_q4_dense_h16(", "kernel void hy_q4_grouped_h16(")
    kernel = sub_once(kernel, "constant uint& reduction_k [[buffer(6)]],", <<-METAL)
      constant uint& groups [[buffer(6)]],
          constant uint& interleaved [[buffer(7)]],
          device atomic_uint* coverage [[buffer(8)]],
      METAL
    kernel = sub_once(kernel, "const int r0 = tgpig.y * MM_NR0;", <<-METAL)
      const uint n_iter = in_dim / MM_NK / groups;
          const uint start_k = tgpig.z * (interleaved ? MM_NK : n_iter * MM_NK);
          const uint step_k = interleaved ? groups * MM_NK : MM_NK;
          output += tgpig.z * batch * out_dim;
          const int r0 = tgpig.y * MM_NR0;
      METAL
    kernel = sub_once(kernel, "const uint n_iter = reduction_k / MM_NK;", "")
    kernel = sub_once(kernel, "[(r0 + lr0) * in_dim + 16 * il0 + d]", "[(r0 + lr0) * in_dim + start_k + 16 * il0 + d]")
    kernel = sub_once(kernel, "[(r0 + lr0) * in_dim + (iter + 1) * MM_NK + 16 * il0 + d]",
      "[(r0 + lr0) * in_dim + start_k + (iter + 1) * step_k + 16 * il0 + d]")
    kernel = sub_once(kernel, "x + (r1 + lr1) * in_dim + iy;", "x + (r1 + lr1) * in_dim + start_k + iy;")
    raise "Unexpected dense input step count" unless kernel.scan("y += MM_NK;").size == 2
    kernel = kernel.gsub("y += MM_NK;", "y += step_k;")
    kernel = sub_once(kernel, "for (uint iter = 0; iter < n_iter; iter++) {", <<-METAL)
      for (uint iter = 0; iter < n_iter; iter++) {
              if (tgpig.x == 0 && tgpig.y == 0 && tiitg == 0)
                  atomic_fetch_add_explicit(coverage + (start_k + iter * step_k) / MM_NK, 1u, memory_order_relaxed);
      METAL
    source + kernel
  end

  private def sub_once(source : String, needle : String, replacement : String) : String
    raise "Grouped SIMD source anchor changed: #{needle}" unless source.scan(needle).size == 1
    source.sub(needle, replacement)
  end

  record TensorView, source : String, offset_bytes : Int64, nbytes : Int64,
    shape : Array(Int64), encoding : String
  alias Comparison = NamedTuple(name: String, elements: Int32, failures: Int64,
    raw_bit_differences: Int64, max_abs: Float64, rmse: Float64,
    relative_l2: Float64?, finite: Bool, gate_passed: Bool,
    atol: Float32, rtol: Float32)

  class Fixture
    getter sha256 : String
    getter sidecar_sha256 : String
    getter input_values : Array(Float32)
    getter upstream_output : Array(Float32)
    getter input_name : String = "dit.txt.ff.activated"
    getter output_name : String = "dit.txt.ff.projected"

    @path : String
    @sidecar_path : String
    @json_info : File::Info
    @sidecar_info : File::Info

    def initialize(@path : String)
      integrity = ML::Video::HyNomasQ4Cache::Integrity
      @json_info = integrity.regular_file_info!(@path, "conditioning fixture")
      unless @json_info.size > 0 && @json_info.size <= 1024 * 1024
        raise ArgumentError.new("Conditioning fixture JSON size is invalid")
      end
      raw = File.read(@path)
      @sha256 = Digest::SHA256.hexdigest(raw)
      raise ArgumentError.new("Conditioning fixture SHA-256 mismatch") unless @sha256 == FIXTURE_SHA256
      integrity.same_snapshot?(@json_info, integrity.regular_file_info!(@path, "conditioning fixture")) ||
        raise ArgumentError.new("Conditioning fixture changed while reading")
      document = JSON.parse(raw)
      validate_document!(document)

      file = document["tensor_file"]
      basename = file["name"].as_s
      unless basename == "text-conditioning-v2.tensors.f32le" && integrity.safe_basename?(basename)
        raise ArgumentError.new("Conditioning sidecar basename mismatch")
      end
      @sidecar_path = File.join(File.dirname(@path), basename)
      @sidecar_info = integrity.regular_file_info!(@sidecar_path, "conditioning sidecar")
      unless @sidecar_info.size == FIXTURE_SIDECAR_BYTES &&
             file["nbytes"].as_i64 == FIXTURE_SIDECAR_BYTES &&
             file["sha256"].as_s == FIXTURE_SIDECAR_SHA256
        raise ArgumentError.new("Conditioning sidecar metadata mismatch")
      end
      @sidecar_sha256 = integrity.sha256_file(@sidecar_path)
      raise ArgumentError.new("Conditioning sidecar SHA-256 mismatch") unless @sidecar_sha256 == FIXTURE_SIDECAR_SHA256

      @input_values = read_tensor(document, input_name, [1_i64, 23_i64, 8192_i64], 4_063_232_i64, 753_664_i64)
      @upstream_output = read_tensor(document, output_name, [1_i64, 23_i64, 2048_i64], 5_947_392_i64, 188_416_i64)
      unless @input_values.all?(&.finite?) && @upstream_output.all?(&.finite?)
        raise ArgumentError.new("Conditioning fixture selected tensor contains nonfinite values")
      end
      unchanged!
    end

    def unchanged! : Nil
      integrity = ML::Video::HyNomasQ4Cache::Integrity
      integrity.same_snapshot?(@json_info, integrity.regular_file_info!(@path, "conditioning fixture")) ||
        raise ArgumentError.new("Conditioning fixture changed after authentication")
      integrity.same_snapshot?(@sidecar_info, integrity.regular_file_info!(@sidecar_path, "conditioning sidecar")) ||
        raise ArgumentError.new("Conditioning sidecar changed after authentication")
    end

    private def validate_document!(document : JSON::Any) : Nil
      unless document["schema"].as_s == "hy-nomas-text-conditioning-reference-v1" &&
             document["schema_version"].as_i == 1 &&
             document["upstream_revision"].as_s == UPSTREAM_REVISION &&
             document["tolerance"]["atol"].as_f == TOLERANCE &&
             document["tolerance"]["rtol"].as_f == TOLERANCE
        raise ArgumentError.new("Conditioning fixture schema, revision, or tolerance mismatch")
      end
      geometry = document["geometry"]
      unless geometry["video_tokens"].as_i == 6 && geometry["valid_tokens"].as_i == ROWS &&
             geometry["text_tokens"].as_i == 1280 && geometry["batches"].as_i == 1 &&
             geometry["dim"].as_i == 2048 && geometry["ffn_dim"].as_i == 8192 &&
             document["producer"]["device"].as_s == "mps"
        raise ArgumentError.new("Conditioning fixture geometry or producer mismatch")
      end
    end

    private def read_tensor(document : JSON::Any, name : String,
                            shape : Array(Int64), offset : Int64, bytes : Int64) : Array(Float32)
      descriptor = document["tensors"][name]
      unless descriptor["dtype"].as_s == "F32" &&
             descriptor["shape"].as_a.map(&.as_i64) == shape &&
             descriptor["offset_bytes"].as_i64 == offset &&
             descriptor["nbytes"].as_i64 == bytes
        raise ArgumentError.new("Conditioning tensor shape/layout mismatch: #{name}")
      end
      count = (bytes // sizeof(Float32)).to_i32
      raw = Bytes.new(bytes.to_i32)
      File.open(@sidecar_path, "rb") do |io|
        io.seek(offset)
        io.read_fully(raw)
      end
      Array(Float32).new(count) do |i|
        IO::ByteFormat::LittleEndian.decode(Float32, raw[i * sizeof(Float32), sizeof(Float32)])
      end
    end
  end

  class Owner
    getter named_bytes : Int64
    getter resource_bytes : Hash(String, Int64)
    getter? encoded : Bool = false
    @buffers = {} of String => ML::MetalBuffer
    @pipelines = {} of String => ML::Metal::ComputePipeline
    @plans = {} of String => ML::Video::WanT5MPSLinearPlan
    @chain : ML::Metal::MPSCommandChain? = nil
    @released = false

    def initialize(@cache : ML::Video::HyNomasQ4Cache::Block, input : Array(Float32))
      @resource_bytes, @named_bytes = HyNomasQ4ReductionProbe.planned_named_bytes
      begin
        @cache.load!
        @resource_bytes.each do |name, bytes|
          @buffers[name] = ML::MetalBuffer.new(bytes) unless name == "cache.q4_payload"
        end
        buffer("input.f32").write(input)
        stats = Array(UInt32).new(DECODE_STATS_WORDS, 0_u32)
        3.times { |i| stats[i * STATS_COUNT + 3] = UInt32::MAX }
        buffer("decode.stats").write_bytes(stats.to_unsafe.as(UInt8*), DECODE_STATS_WORDS * sizeof(UInt32))
        coverage = Array(UInt32).new(GROUP_COVERAGE_WORDS, 0_u32)
        buffer("group.coverage").write_bytes(coverage.to_unsafe.as(UInt8*), GROUP_COVERAGE_WORDS * sizeof(UInt32))
        qwen_source = HyNomasQ4ReductionProbe.grouped_source
        ["simd_mm_q4k_h16", "hy_q4_dump_helper", "hy_q4_dense_h16", "hy_q4_grouped_h16"].each do |name|
          @pipelines[name] = ML::Metal::ComputePipeline.new(name, qwen_source)
        end
        ["hy_q4_round_input", "hy_q4_canonical", "hy_q4_compare", "hy_q4_pack_prefix",
         "hy_q4_copy", "hy_q4_compensated", "hy_q4_poison_partials", "hy_q4_merge_partials"].each do |name|
          @pipelines[name] = ML::Metal::ComputePipeline.new(name, DIAGNOSTIC_SOURCE)
        end
        ML::Video::HyNomasQ4Cache.prepare_decode!
        ML::Video::MetalLinearMPS.prepare!
        @pipelines["bias"] = ML::Video::MetalLinearMPS.pipeline
        create_plan("mps.canonical.raw", INPUT_DIM)
        create_plan("mps.helper.raw", INPUT_DIM)
        PREFIXES.each { |k| create_plan("mps.k#{k}", k) }
      rescue ex
        release_setup_only!
        raise ex
      end
    end

    def buffer(name : String) : ML::MetalBuffer
      @buffers[name]? || raise "Missing reduction probe buffer #{name}"
    end

    def encode(chain : ML::Metal::MPSCommandChain) : Nil
      raise "Reduction owner already encoded/released" if @encoded || @released
      @encoded = true
      @chain = chain
      e = chain.encoder
      count = (INPUT_DIM.to_i64 * OUTPUT_DIM).to_u32
      e.set_pipeline(@pipelines["hy_q4_round_input"])
        .set_buffer(buffer("input.f32"), 0)
        .set_buffer(buffer("input.h16"), 1, ML::Metal::BufferAccess::Write)
        .set_buffer(buffer("input.rounded_f32"), 2, ML::Metal::BufferAccess::Write)
        .set_value(INPUT_ELEMENTS.to_u32, 3).dispatch_1d(INPUT_ELEMENTS.to_i32)
      {"hy_q4_canonical" => "weight.canonical", "hy_q4_dump_helper" => "weight.helper"}.each do |pipeline, name|
        e.set_pipeline(@pipelines[pipeline])
          .set_buffer(@cache.resident_payload, 0, offset: Q4_WEIGHT_OFFSET)
          .set_buffer(buffer(name), 1, ML::Metal::BufferAccess::Write)
          .set_value(count, 2)
          .dispatch_1d(pipeline == "hy_q4_dump_helper" ? (count // 16).to_i32 : count.to_i32)
      end
      3.times do |control|
        e.set_pipeline(@pipelines["hy_q4_compare"])
          .set_buffer(buffer(control == 0 ? "weight.canonical" : "weight.helper"), 0)
          .set_buffer(buffer("weight.helper"), 1)
          .set_buffer(buffer("decode.stats"), 2, ML::Metal::BufferAccess::ReadWrite, offset: control.to_i64 * STATS_COUNT * sizeof(UInt32))
          .set_value(count, 3).set_value(control == 2 ? 17_u32 : UInt32::MAX, 4).dispatch_1d(count.to_i32)
      end
      # The same resident BF16 bias is used by all branches.
      e.set_pipeline(ML::Video::HyNomasQ4Cache.copy_pipeline)
        .set_buffer(@cache.resident_payload, 0, offset: BIAS_OFFSET)
        .set_buffer(buffer("bias.bf16"), 1, ML::Metal::BufferAccess::Write)
        .set_value(OUTPUT_DIM.to_u32, 2).dispatch_1d(OUTPUT_DIM)

      encode_simd(e, "direct.raw", INPUT_DIM, false)
      encode_simd(e, "dense.raw", INPUT_DIM, true)
      encode_mps(e, "mps.canonical.raw", INPUT_DIM, "weight.canonical")
      encode_mps(e, "mps.helper.raw", INPUT_DIM, "weight.helper")
      e.set_pipeline(@pipelines["hy_q4_compensated"])
        .set_buffer(buffer("weight.helper"), 0).set_buffer(buffer("input.h16"), 1)
        .set_buffer(buffer("output.compensated.raw"), 2, ML::Metal::BufferAccess::Write)
        .set_value(INPUT_DIM.to_u32, 3).set_value(OUTPUT_DIM.to_u32, 4)
        .set_value(ROWS.to_u32, 5).dispatch_1d(OUTPUT_ELEMENTS.to_i32)
      ["direct", "dense", "mps.canonical", "mps.helper", "compensated"].each do |name|
        copy(e, buffer("output.#{name}.raw"), buffer("output.#{name}.bias"), OUTPUT_ELEMENTS.to_i32, 0)
        e.set_pipeline(@pipelines["bias"])
          .set_buffer(buffer("output.#{name}.bias"), 0, ML::Metal::BufferAccess::ReadWrite)
          .set_buffer(buffer("bias.bf16"), 1).set_buffer(buffer("bias.bf16"), 2)
          .set_value(OUTPUT_DIM.to_u32, 3).set_value(1_u32, 4)
          .set_value(OUTPUT_ELEMENTS.to_u32, 5).dispatch_1d(OUTPUT_ELEMENTS.to_i32)
      end
      PREFIXES.each do |k|
        encode_simd(e, "dense.k#{k}", k, true)
        encode_mps(e, "mps.k#{k}", k, "weight.helper")
      end
      GROUP_CASES.each_with_index do |c, i|
        e.set_pipeline(@pipelines["hy_q4_poison_partials"])
          .set_buffer(buffer("group.partials"), 0, ML::Metal::BufferAccess::Write)
          .set_value((32 * OUTPUT_ELEMENTS).to_u32, 1).dispatch_1d((32 * OUTPUT_ELEMENTS).to_i32)
        e.set_pipeline(@pipelines["hy_q4_grouped_h16"])
          .set_buffer(buffer("weight.helper"), 0).set_buffer(buffer("input.h16"), 1)
          .set_buffer(buffer("group.partials"), 2, ML::Metal::BufferAccess::Write)
          .set_value(INPUT_DIM.to_u32, 3).set_value(OUTPUT_DIM.to_u32, 4).set_value(ROWS.to_u32, 5)
          .set_value(c[:groups].to_u32, 6).set_value(c[:interleaved] ? 1_u32 : 0_u32, 7)
          .set_buffer(buffer("group.coverage"), 8, ML::Metal::BufferAccess::ReadWrite,
            offset: i.to_i64 * GROUP_TILE_COUNT * sizeof(UInt32))
          .set_threadgroup_memory(MM_SHMEM, 0)
          .dispatch_threadgroups({(ROWS + MM_NR1 - 1) // MM_NR1, (OUTPUT_DIM + MM_NR0 - 1) // MM_NR0, c[:groups]}, {MM_TG, 1, 1})
        GROUP_MERGES.each do |merge|
          name = "group.#{c[:name]}.#{merge}"
          e.set_pipeline(@pipelines["hy_q4_merge_partials"])
            .set_buffer(buffer("group.partials"), 0)
            .set_buffer(buffer("output.#{name}.raw"), 1, ML::Metal::BufferAccess::Write)
            .set_value(OUTPUT_ELEMENTS.to_u32, 2).set_value(c[:groups].to_u32, 3)
            .set_value(merge == "tree" ? 1_u32 : 0_u32, 4).dispatch_1d(OUTPUT_ELEMENTS.to_i32)
          copy(e, buffer("output.#{name}.raw"), buffer("output.#{name}.bias"), OUTPUT_ELEMENTS.to_i32, 0)
          e.set_pipeline(@pipelines["bias"])
            .set_buffer(buffer("output.#{name}.bias"), 0, ML::Metal::BufferAccess::ReadWrite)
            .set_buffer(buffer("bias.bf16"), 1).set_buffer(buffer("bias.bf16"), 2)
            .set_value(OUTPUT_DIM.to_u32, 3).set_value(1_u32, 4)
            .set_value(OUTPUT_ELEMENTS.to_u32, 5).dispatch_1d(OUTPUT_ELEMENTS.to_i32)
        end
      end
      OUTPUT_VARIANTS.each_with_index do |name, i|
        copy(e, buffer("output.#{name}"), buffer("output.packed"), OUTPUT_ELEMENTS.to_i32, (i * OUTPUT_ELEMENTS).to_i32)
      end
      copy(e, buffer("decode.stats"), buffer("output.packed"), DECODE_STATS_WORDS, (OUTPUT_VARIANTS.size * OUTPUT_ELEMENTS).to_i32)
      copy(e, buffer("group.coverage"), buffer("output.packed"), GROUP_COVERAGE_WORDS,
        (OUTPUT_VARIANTS.size * OUTPUT_ELEMENTS + DECODE_STATS_WORDS).to_i32)
    end

    def values(chain : ML::Metal::MPSCommandChain) : Array(Float32)
      check_terminal!(chain)
      buffer("output.packed").read((OUTPUT_VARIANTS.size * OUTPUT_ELEMENTS + STATS_WORDS).to_i32)
    end

    def release_after_terminal!(chain : ML::Metal::MPSCommandChain) : Nil
      check_terminal!(chain)
      release_owned!
    end

    def release_setup_only! : Nil
      raise "Cannot release in-flight reduction resources" if @encoded && !@chain.try(&.completed?)
      release_owned!
    end

    private def check_terminal!(chain : ML::Metal::MPSCommandChain) : Nil
      unless @encoded && chain.completed? && chain.wait_count == 1 && @chain.try(&.same?(chain))
        raise "Reduction resources require their sole completed terminal wait"
      end
    end

    private def create_plan(name : String, k : Int32) : Nil
      config = ML::Video::WanT5Config.new(vocab: 1, dim: 1, ffn_dim: 1,
        heads: 1, layers: 1, batches: 1, tokens: ROWS)
      @plans[name] = ML::Video::WanT5MPSLinearPlan.new(config, k, OUTPUT_DIM,
        buffer("input.mps_prefix"), buffer("mps.weight_stage"), buffer("output.#{name}"))
    end

    private def encode_simd(e : ML::Metal::ComputeEncoder, name : String, k : Int32, dense : Bool) : Nil
      e.set_pipeline(@pipelines[dense ? "hy_q4_dense_h16" : "simd_mm_q4k_h16"])
      if dense
        e.set_buffer(buffer("weight.helper"), 0).set_value(k.to_u32, 6)
      else
        e.set_buffer(@cache.resident_payload, 0, offset: Q4_WEIGHT_OFFSET)
      end
      e.set_buffer(buffer("input.h16"), 1)
        .set_buffer(buffer("output.#{name}"), 2, ML::Metal::BufferAccess::Write)
        .set_value(INPUT_DIM.to_u32, 3).set_value(OUTPUT_DIM.to_u32, 4).set_value(ROWS.to_u32, 5)
        .set_threadgroup_memory(MM_SHMEM, 0)
        .dispatch_threadgroups({(ROWS + MM_NR1 - 1) // MM_NR1, (OUTPUT_DIM + MM_NR0 - 1) // MM_NR0, 1}, {MM_TG, 1, 1})
    end

    private def encode_mps(e : ML::Metal::ComputeEncoder, name : String, k : Int32, weights : String) : Nil
      e.set_pipeline(@pipelines["hy_q4_pack_prefix"])
        .set_buffer(buffer(weights), 0).set_buffer(buffer("input.h16"), 1)
        .set_buffer(buffer("mps.weight_stage"), 2, ML::Metal::BufferAccess::Write)
        .set_buffer(buffer("input.mps_prefix"), 3, ML::Metal::BufferAccess::Write)
        .set_value(INPUT_DIM.to_u32, 4).set_value(k.to_u32, 5)
        .set_value(OUTPUT_DIM.to_u32, 6).set_value(ROWS.to_u32, 7).dispatch_1d(OUTPUT_DIM * k)
      ML::Video::MetalLinearMPS.project_preexpanded(e, buffer("input.mps_prefix"), buffer("bias.bf16"),
        buffer("output.#{name}"), ROWS, k, OUTPUT_DIM, ML::Video::WeightDType::BF16,
        @plans[name], buffer("mps.weight_stage"), apply_bias: false)
    end

    private def copy(e : ML::Metal::ComputeEncoder, source : ML::MetalBuffer,
                     destination : ML::MetalBuffer, count : Int32, offset : Int32) : Nil
      e.set_pipeline(@pipelines["hy_q4_copy"]).set_buffer(source, 0)
        .set_buffer(destination, 1, ML::Metal::BufferAccess::Write)
        .set_value(count.to_u32, 2).set_value(offset.to_u32, 3).dispatch_1d(count)
    end

    private def release_owned! : Nil
      return if @released
      @plans.each_value(&.release)
      @plans.clear
      @buffers.each_value { |value| value.release if value.valid? }
      @buffers.clear
      @cache.release
      @released = true
    end
  end

  def planned_named_bytes : {Hash(String, Int64), Int64}
    resources = {
      "cache.q4_payload"  => CACHE_PAYLOAD_BYTES,
      "input.f32"         => INPUT_BYTES,
      "input.h16"         => INPUT_ELEMENTS * sizeof(UInt16),
      "input.rounded_f32" => INPUT_BYTES,
      "input.mps_prefix"  => INPUT_BYTES,
      "weight.canonical"  => Q4_BF16_BYTES,
      "weight.helper"     => Q4_BF16_BYTES,
      "mps.weight_stage"  => INPUT_DIM.to_i64 * OUTPUT_DIM * sizeof(Float32),
      "bias.bf16"         => BIAS_BYTES,
      "decode.stats"      => DECODE_STATS_WORDS.to_i64 * sizeof(UInt32),
      "group.coverage"    => GROUP_COVERAGE_WORDS.to_i64 * sizeof(UInt32),
      "group.partials"    => 32_i64 * OUTPUT_BYTES,
    }
    OUTPUT_VARIANTS.each { |name| resources["output.#{name}"] = OUTPUT_BYTES }
    resources["output.packed"] = OUTPUT_VARIANTS.size.to_i64 * OUTPUT_BYTES + STATS_WORDS * sizeof(UInt32)
    {resources, resources.values.sum}
  end

  def verify_cache_tensor!(cache : ML::Video::HyNomasQ4Cache::Block,
                           name : String, expected_shape : Array(Int64),
                           encoding : String, offset : Int64, bytes : Int64) : TensorView
    matches = cache.tensors.select { |tensor| tensor.source == name }
    raise ArgumentError.new("Cache lacks exact selected tensor #{name}") unless matches.size == 1
    tensor = matches.first
    unless tensor.shape == expected_shape && tensor.encoding == encoding &&
           tensor.offset_bytes == offset && tensor.nbytes == bytes
      raise ArgumentError.new("Selected Q4 cache tensor contract mismatch: #{name}")
    end
    TensorView.new(tensor.source, tensor.offset_bytes, tensor.nbytes,
      tensor.shape, tensor.encoding)
  end

  def comparison(name : String, actual : Array(Float32), expected : Array(Float32),
                 atol : Float32 = ATOL, rtol : Float32 = RTOL) : Comparison
    raise ArgumentError.new("Comparison lengths differ") unless actual.size == expected.size
    finite = true
    failures = 0_i64
    raw_differences = 0_i64
    sum_sq_error = 0.0_f64
    sum_sq_expected = 0.0_f64
    max_abs = 0.0_f64
    actual.each_with_index do |value, i|
      target = expected[i]
      if !value.finite? || !target.finite?
        finite = false
        failures += 1
        next
      end
      raw_differences += 1 if value.unsafe_as(UInt32) != target.unsafe_as(UInt32)
      difference = (value.to_f64 - target.to_f64).abs
      max_abs = Math.max(max_abs, difference)
      sum_sq_error += difference * difference
      sum_sq_expected += target.to_f64 * target.to_f64
      failures += 1 if difference > atol.to_f64 + rtol.to_f64 * target.to_f64.abs
    end
    rmse = Math.sqrt(sum_sq_error / actual.size)
    relative_l2 = sum_sq_expected == 0.0 ? (sum_sq_error == 0.0 ? 0.0 : nil) : Math.sqrt(sum_sq_error / sum_sq_expected)
    {name: name, elements: actual.size, failures: failures,
     raw_bit_differences: raw_differences, max_abs: max_abs, rmse: rmse,
     relative_l2: relative_l2, finite: finite, gate_passed: finite && failures == 0,
     atol: atol, rtol: rtol}
  end

  def validate_report_path!(path : String) : String
    raise ArgumentError.new("Report path must be absolute") unless path.starts_with?('/')
    unless File.basename(path).matches?(/\A[A-Za-z0-9][A-Za-z0-9_.-]*\.json\z/)
      raise ArgumentError.new("Report basename is unsafe")
    end
    parent_real = File.realpath(File.dirname(path))
    unless [ROOT, GROUP_ROOT].any? { |root| File.directory?(root) && parent_real == File.realpath(root) }
      raise ArgumentError.new("Report must be a direct child of the dedicated reduction reports directory")
    end
    raise ArgumentError.new("Refusing to replace existing report") if File.info?(path, false)
    path
  end

  def write_new_artifact!(path : String, content : String) : Nil
    raise ArgumentError.new("Refusing to replace existing report") if File.info?(path, false)
    temp = File.tempfile(".q4-reduction-", ".partial", dir: File.dirname(path))
    begin
      temp << content
      temp.flush
      temp.fsync
      temp.close
      File.link(temp.path, path)
    ensure
      temp.close unless temp.closed?
      File.delete(temp.path) if File.info?(temp.path, false)
    end
  end

  def extract_output(packed : Array(Float32), index : Int32) : Array(Float32)
    raise ArgumentError.new("Output variant index is invalid") unless (0...OUTPUT_VARIANTS.size).includes?(index)
    packed[index.to_i64 * OUTPUT_ELEMENTS, OUTPUT_ELEMENTS].not_nil!
  end
end
