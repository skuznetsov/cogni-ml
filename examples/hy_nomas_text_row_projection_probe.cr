require "digest/sha256"
require "json"
require "option_parser"
require "./support/hy_nomas_q4_reduction_probe"

module HyNomasTextRowProjectionProbe
  extend self

  INPUT_DIM          = HyNomasQ4ReductionProbe::INPUT_DIM
  OUTPUT_DIM         = HyNomasQ4ReductionProbe::OUTPUT_DIM
  SOURCE_ROWS        = HyNomasQ4ReductionProbe::ROWS
  MAX_ROWS           = 1280_i32
  STORAGE_ROWS       = [23_i32, 32_i32, 64_i32, 128_i32, 256_i32, 512_i32, 1280_i32]
  COMPARE_ELEMENTS   = SOURCE_ROWS * OUTPUT_DIM
  WEIGHT_ELEMENTS    = INPUT_DIM.to_i64 * OUTPUT_DIM
  MAX_INPUT_ELEMENTS = MAX_ROWS.to_i64 * INPUT_DIM
  MAX_NAMED_BYTES    = 512_i64 * 1024 * 1024
  REPORT_ROOT        = "/opt/mldata_ssd/cogni-ml/hy-nomas-text-compaction-20261026/reports"
  ATOL               = 2e-5_f32
  RTOL               = 2e-5_f32

  KERNEL_SOURCE = <<-METAL
    #include <metal_stdlib>
    using namespace metal;

    kernel void hy_text_projection_bf16_to_f32(
        device const ushort *source [[buffer(0)]],
        device float *destination [[buffer(1)]],
        constant uint &count [[buffer(2)]],
        uint gid [[thread_position_in_grid]]) {
      if (gid >= count) return;
      destination[gid] = as_type<float>(uint(source[gid]) << 16u);
    }

    kernel void hy_text_projection_repeat_x(
        device const float *source [[buffer(0)]],
        device float *destination [[buffer(1)]],
        constant uint &source_rows [[buffer(2)]],
        constant uint &input_dim [[buffer(3)]],
        constant uint &rows [[buffer(4)]],
        uint gid [[thread_position_in_grid]]) {
      const uint count = rows * input_dim;
      if (gid >= count) return;
      const uint row = gid / input_dim;
      const uint col = gid - row * input_dim;
      destination[gid] = source[(row % source_rows) * input_dim + col];
    }
    METAL

  record Comparison, storage_rows : Int32, elements : Int32,
    failures : Int64, raw_bit_differences : Int64, nonfinite : Int64,
    max_abs : Float64, finite : Bool, gate_passed : Bool

  class Owner
    getter named_bytes : Int64
    getter resource_bytes : Hash(String, Int64)
    getter? encoded : Bool = false

    @cache : ML::Video::HyNomasQ4Cache::Block
    @fixture : HyNomasQ4ReductionProbe::Fixture
    @buffers = {} of String => ML::MetalBuffer
    @pipelines = {} of String => ML::Metal::ComputePipeline
    @plans = {} of Int32 => ML::Video::WanT5MPSLinearPlan
    @chain : ML::Metal::MPSCommandChain? = nil
    @released = false

    def initialize(@cache : ML::Video::HyNomasQ4Cache::Block,
                   @fixture : HyNomasQ4ReductionProbe::Fixture,
                   @resource_bytes : Hash(String, Int64), @named_bytes : Int64)
      begin
        @cache.load!
        @resource_bytes.each do |name, bytes|
          @buffers[name] = ML::MetalBuffer.new(bytes) unless name == "cache.q4_payload"
        end
        buffer("input.source_f32").write(@fixture.input_values)

        ML::Video::HyNomasQ4Cache.prepare_decode!
        ML::Video::MetalLinearMPS.prepare!
        @pipelines["bf16_to_f32"] = ML::Metal::ComputePipeline.new(
          "hy_text_projection_bf16_to_f32", KERNEL_SOURCE)
        @pipelines["repeat_x"] = ML::Metal::ComputePipeline.new(
          "hy_text_projection_repeat_x", KERNEL_SOURCE)

        STORAGE_ROWS.each do |rows|
          output_name = output_buffer_name(rows)
          config = ML::Video::WanT5Config.new(vocab: 1, dim: 1, ffn_dim: 1,
            heads: 1, layers: 1, batches: 1, tokens: rows)
          @plans[rows] = ML::Video::WanT5MPSLinearPlan.new(config,
            INPUT_DIM, OUTPUT_DIM, buffer("input.storage_f32"),
            buffer("weight.mps_f32"), buffer(output_name))
          ML::Video::MetalLinearMPS.validate_preexpanded!(
            buffer("input.storage_f32"), buffer("bias.bf16"), buffer(output_name),
            rows, INPUT_DIM, OUTPUT_DIM, ML::Video::WeightDType::BF16,
            @plans[rows], buffer("weight.mps_f32"))
        end
      rescue ex
        release_setup_only!
        raise ex
      end
    end

    def buffer(name : String) : ML::MetalBuffer
      @buffers[name]? || raise ArgumentError.new("Missing row projection buffer #{name}")
    end

    def encode(chain : ML::Metal::MPSCommandChain) : Nil
      raise ArgumentError.new("Text row projection owner already encoded or released") if @encoded || @released
      @encoded = true
      @chain = chain
      e = chain.encoder

      weight_count = WEIGHT_ELEMENTS.to_u32
      e.set_pipeline(ML::Video::HyNomasQ4Cache.q4_pipeline)
        .set_buffer(@cache.resident_payload, 0,
          offset: HyNomasQ4ReductionProbe::Q4_WEIGHT_OFFSET)
        .set_buffer(buffer("weight.decoded_bf16"), 1, ML::Metal::BufferAccess::Write)
        .set_value(weight_count, 2)
        .dispatch_1d(weight_count.to_i32)

      e.set_pipeline(@pipelines["bf16_to_f32"])
        .set_buffer(buffer("weight.decoded_bf16"), 0)
        .set_buffer(buffer("weight.mps_f32"), 1, ML::Metal::BufferAccess::Write)
        .set_value(weight_count, 2)
        .dispatch_1d(weight_count.to_i32)

      e.set_pipeline(ML::Video::HyNomasQ4Cache.copy_pipeline)
        .set_buffer(@cache.resident_payload, 0,
          offset: HyNomasQ4ReductionProbe::BIAS_OFFSET)
        .set_buffer(buffer("bias.bf16"), 1, ML::Metal::BufferAccess::Write)
        .set_value(OUTPUT_DIM.to_u32, 2)
        .dispatch_1d(OUTPUT_DIM)

      STORAGE_ROWS.each do |rows|
        e.set_pipeline(@pipelines["repeat_x"])
          .set_buffer(buffer("input.source_f32"), 0)
          .set_buffer(buffer("input.storage_f32"), 1, ML::Metal::BufferAccess::Write)
          .set_value(SOURCE_ROWS.to_u32, 2)
          .set_value(INPUT_DIM.to_u32, 3)
          .set_value(rows.to_u32, 4)
          .dispatch_1d((rows.to_i64 * INPUT_DIM).to_i32)

        ML::Video::MetalLinearMPS.project_preexpanded(e,
          buffer("input.storage_f32"), buffer("bias.bf16"),
          buffer(output_buffer_name(rows)), rows, INPUT_DIM, OUTPUT_DIM,
          ML::Video::WeightDType::BF16, @plans[rows], buffer("weight.mps_f32"),
          apply_bias: false)
      end
    end

    def values(chain : ML::Metal::MPSCommandChain, rows : Int32) : Array(Float32)
      check_terminal!(chain)
      buffer(output_buffer_name(rows)).read((rows.to_i64 * OUTPUT_DIM).to_i32)
    end

    def release_after_terminal!(chain : ML::Metal::MPSCommandChain) : Nil
      check_terminal!(chain)
      release_owned!
    end

    def release_setup_only! : Nil
      if @encoded && !@chain.try(&.completed?)
        raise ArgumentError.new("Cannot release in-flight text row projection resources")
      end
      release_owned!
    end

    private def check_terminal!(chain : ML::Metal::MPSCommandChain) : Nil
      unless @encoded && chain.completed? && chain.wait_count == 1 &&
             @chain.try(&.same?(chain))
        raise ArgumentError.new("Text row projection read requires its sole terminal wait")
      end
    end

    private def release_owned! : Nil
      return if @released
      @plans.each_value(&.release)
      @plans.clear
      @buffers.each_value(&.release)
      @buffers.clear
      @cache.release
      @released = true
    end

    private def output_buffer_name(rows : Int32) : String
      "output.m#{rows}"
    end
  end

  def planned_named_bytes : {Hash(String, Int64), Int64}
    resources = {
      "cache.q4_payload"    => HyNomasQ4ReductionProbe::CACHE_PAYLOAD_BYTES,
      "input.source_f32"    => HyNomasQ4ReductionProbe::INPUT_BYTES,
      "input.storage_f32"   => MAX_INPUT_ELEMENTS * sizeof(Float32),
      "weight.decoded_bf16" => HyNomasQ4ReductionProbe::Q4_BF16_BYTES,
      "weight.mps_f32"      => WEIGHT_ELEMENTS * sizeof(Float32),
      "bias.bf16"           => HyNomasQ4ReductionProbe::BIAS_BYTES,
    }
    STORAGE_ROWS.each do |rows|
      resources["output.m#{rows}"] = rows.to_i64 * OUTPUT_DIM * sizeof(Float32)
    end
    {resources, resources.values.sum}
  end

  def validate_report_path!(path : String) : String
    raise ArgumentError.new("Report path must be absolute") unless path.starts_with?('/')
    unless File.basename(path).matches?(/\A[A-Za-z0-9][A-Za-z0-9_.-]*\.json\z/)
      raise ArgumentError.new("Report basename is unsafe")
    end
    unless File.directory?(REPORT_ROOT) && File.realpath(File.dirname(path)) == File.realpath(REPORT_ROOT)
      raise ArgumentError.new("Report must be a direct child of the text compaction reports directory")
    end
    raise ArgumentError.new("Refusing to replace existing report") if File.info?(path, false)
    path
  end

  def compare_rows(storage_rows : Int32, actual : Array(Float32),
                   reference : Array(Float32)) : Comparison
    raise ArgumentError.new("Row projection comparison storage is too small") unless actual.size >= COMPARE_ELEMENTS && reference.size >= COMPARE_ELEMENTS

    failures = 0_i64
    raw_differences = 0_i64
    nonfinite = 0_i64
    max_abs = 0.0_f64
    COMPARE_ELEMENTS.times do |i|
      value = actual[i]
      target = reference[i]
      unless value.finite? && target.finite?
        nonfinite += 1
        failures += 1
        next
      end
      raw_differences += 1 if value.unsafe_as(UInt32) != target.unsafe_as(UInt32)
      difference = (value.to_f64 - target.to_f64).abs
      max_abs = Math.max(max_abs, difference)
      failures += 1 if difference > ATOL.to_f64 + RTOL.to_f64 * target.to_f64.abs
    end
    finite = nonfinite == 0
    Comparison.new(storage_rows, COMPARE_ELEMENTS, failures, raw_differences,
      nonfinite, max_abs, finite, finite && failures == 0)
  end

  def f32_sha256(values : Array(Float32)) : String
    bytes = values.to_unsafe.as(UInt8*).to_slice(values.size * sizeof(Float32))
    Digest::SHA256.hexdigest(bytes)
  end

  def output_buffer_name(rows : Int32) : String
    "output.m#{rows}"
  end
end

report_path = ""
OptionParser.parse do |parser|
  parser.banner = "Bounded native text row projection shape diagnostic; no generation."
  parser.on("--report PATH", "Fresh JSON report under the Q4 reduction reports directory") { |v| report_path = v }
end
unless ARGV.empty? && !report_path.empty?
  raise ArgumentError.new("Expected --report PATH and no positional arguments")
end
report_path = HyNomasTextRowProjectionProbe.validate_report_path!(report_path)

# Authenticate all CPU fixtures and exact cache tensor extents, then admit the
# complete named Metal footprint before loading any payload or creating Metal.
fixture = HyNomasQ4ReductionProbe::Fixture.new(HyNomasQ4ReductionProbe::FIXTURE_PATH)
cache = ML::Video::HyNomasQ4Cache::Block.new(
  HyNomasQ4ReductionProbe::CACHE_MANIFEST,
  HyNomasQ4ReductionProbe::CACHE_MANIFEST_SHA256,
  HyNomasQ4ReductionProbe::SOURCE_SHA256,
  ML::Video::HyNomasConfig.new(batches: 1, video_tokens: 5280,
    text_tokens: 1280, dim: 2048, heads: 16, head_dim: 128, ffn_dim: 8192),
  HyNomasQ4ReductionProbe::BLOCK_INDEX)
raise ArgumentError.new("Unexpected Q4 cache source size") unless cache.source_bytes == HyNomasQ4ReductionProbe::SOURCE_BYTES
raise ArgumentError.new("Unexpected Q4 cache payload identity") unless cache.payload_bytes == HyNomasQ4ReductionProbe::CACHE_PAYLOAD_BYTES &&
                                                                       cache.payload_sha256 == HyNomasQ4ReductionProbe::CACHE_PAYLOAD_SHA256
HyNomasQ4ReductionProbe.verify_cache_tensor!(cache,
  "double_blocks.0.txt_mlp.fc2.weight", [2048_i64, 8192_i64],
  "q4_k_8range", HyNomasQ4ReductionProbe::Q4_WEIGHT_OFFSET,
  HyNomasQ4ReductionProbe::Q4_WEIGHT_BYTES)
HyNomasQ4ReductionProbe.verify_cache_tensor!(cache,
  "double_blocks.0.txt_mlp.fc2.bias", [2048_i64], "bf16-le",
  HyNomasQ4ReductionProbe::BIAS_OFFSET, HyNomasQ4ReductionProbe::BIAS_BYTES)
resource_plan, required_named_bytes = HyNomasTextRowProjectionProbe.planned_named_bytes
raise ArgumentError.new("Native row projection named residency exceeds 512 MiB before Metal allocation") if required_named_bytes > HyNomasTextRowProjectionProbe::MAX_NAMED_BYTES

owner = nil.as(HyNomasTextRowProjectionProbe::Owner?)
chain = nil.as(ML::Metal::MPSCommandChain?)
begin
  allocations_before_setup = ML::MetalBuffer.hy_text_probe_allocations
  reads_before_setup = ML::MetalBuffer.hy_text_probe_readbacks
  owner = HyNomasTextRowProjectionProbe::Owner.new(cache, fixture,
    resource_plan, required_named_bytes)
  setup_allocations = ML::MetalBuffer.hy_text_probe_allocations - allocations_before_setup
  setup_readbacks = ML::MetalBuffer.hy_text_probe_readbacks - reads_before_setup
  # Cache::Block::load! creates one instrumented MetalBuffer for the payload;
  # the remaining entries are allocated directly by this probe.
  expected_allocations = resource_plan.size
  raise ArgumentError.new("Named allocation count differs from the admitted inventory") unless owner.resource_bytes == resource_plan && owner.named_bytes == required_named_bytes &&
                                                                                               setup_allocations == expected_allocations && setup_readbacks == 0

  chain = ML::Metal::MPSCommandChain.new
  allocations_before_encode = ML::MetalBuffer.hy_text_probe_allocations
  reads_before_encode = ML::MetalBuffer.hy_text_probe_readbacks
  owner.encode(chain)
  encode_allocations = ML::MetalBuffer.hy_text_probe_allocations - allocations_before_encode
  preterminal_readbacks = ML::MetalBuffer.hy_text_probe_readbacks - reads_before_encode
  raise ArgumentError.new("Encode allocated named buffers, read back, or waited") unless encode_allocations == 0 && preterminal_readbacks == 0 && chain.wait_count == 0

  chain.commit_and_wait
  raise ArgumentError.new("Expected exactly one terminal wait") unless chain.completed? && chain.wait_count == 1

  outputs = {} of Int32 => Array(Float32)
  HyNomasTextRowProjectionProbe::STORAGE_ROWS.each do |rows|
    outputs[rows] = owner.values(chain, rows)
  end
  terminal_readbacks = ML::MetalBuffer.hy_text_probe_readbacks - reads_before_encode
  raise ArgumentError.new("Expected one post-wait output read per storage shape") unless terminal_readbacks == HyNomasTextRowProjectionProbe::STORAGE_ROWS.size
  fixture.unchanged!

  reference = outputs[HyNomasTextRowProjectionProbe::MAX_ROWS]
  comparisons = HyNomasTextRowProjectionProbe::STORAGE_ROWS.map do |rows|
    comparison = HyNomasTextRowProjectionProbe.compare_rows(rows, outputs[rows], reference)
    {
      storage_rows:        rows,
      compared_rows:       HyNomasQ4ReductionProbe::ROWS,
      elements:            comparison.elements,
      failures:            comparison.failures,
      raw_bit_differences: comparison.raw_bit_differences,
      nonfinite:           comparison.nonfinite,
      max_abs:             comparison.max_abs,
      finite:              comparison.finite,
      gate_passed:         comparison.gate_passed,
      output_elements:     outputs[rows].size,
      output_sha256:       HyNomasTextRowProjectionProbe.f32_sha256(outputs[rows]),
    }
  end
  passed = comparisons.all? { |item| item[:gate_passed] }
  result = {
    schema:         "hy-nomas-text-row-projection-shape-diagnostic-v1",
    schema_version: 1,
    scope:          "Block 0 txt_mlp.fc2 raw MPS projection over authenticated text-conditioning-v2 rows; no block or diffusion generation",
    fixture:        {
      path:                HyNomasQ4ReductionProbe::FIXTURE_PATH,
      sha256:              fixture.sha256,
      sidecar_sha256:      fixture.sidecar_sha256,
      upstream_revision:   HyNomasQ4ReductionProbe::UPSTREAM_REVISION,
      input_name:          fixture.input_name,
      input_dtype:         "F32",
      input_shape:         [1, HyNomasQ4ReductionProbe::ROWS, HyNomasQ4ReductionProbe::INPUT_DIM],
      input_sha256:        HyNomasTextRowProjectionProbe.f32_sha256(fixture.input_values),
      activation_rounding: "none; fixture F32 values uploaded unchanged",
    },
    cache: {
      manifest:        HyNomasQ4ReductionProbe::CACHE_MANIFEST,
      manifest_sha256: cache.manifest_sha256,
      source_sha256:   cache.source_sha256,
      payload_sha256:  cache.payload_sha256,
      payload_bytes:   cache.payload_bytes,
      selected_weight: {
        name:         "double_blocks.0.txt_mlp.fc2.weight",
        shape:        [2048, 8192],
        encoding:     "q4_k_8range",
        offset_bytes: HyNomasQ4ReductionProbe::Q4_WEIGHT_OFFSET,
        nbytes:       HyNomasQ4ReductionProbe::Q4_WEIGHT_BYTES,
      },
      selected_bias: {
        name:         "double_blocks.0.txt_mlp.fc2.bias",
        shape:        [2048],
        encoding:     "bf16-le",
        offset_bytes: HyNomasQ4ReductionProbe::BIAS_OFFSET,
        nbytes:       HyNomasQ4ReductionProbe::BIAS_BYTES,
      },
    },
    geometry: {
      input_dim:              HyNomasTextRowProjectionProbe::INPUT_DIM,
      output_dim:             HyNomasTextRowProjectionProbe::OUTPUT_DIM,
      valid_text_rows:        HyNomasTextRowProjectionProbe::SOURCE_ROWS,
      storage_rows:           HyNomasTextRowProjectionProbe::STORAGE_ROWS,
      reference_storage_rows: HyNomasTextRowProjectionProbe::MAX_ROWS,
    },
    representation: {
      weight_path:     "authenticated Q4_K cache -> native Q4 decoder BF16 -> exact BF16-to-F32 bit expansion -> MPS GEMM",
      activation_path: "fixture F32 -> GPU row repeat into one max-1280 F32 input buffer",
      bias_applied:    false,
      tolerance:       {atol: HyNomasTextRowProjectionProbe::ATOL, rtol: HyNomasTextRowProjectionProbe::RTOL},
    },
    resource_admission: {
      cap_bytes:                                       HyNomasTextRowProjectionProbe::MAX_NAMED_BYTES,
      named_bytes:                                     owner.named_bytes,
      named_resources:                                 owner.resource_bytes,
      named_metal_allocations:                         setup_allocations,
      opaque_mps_plan_count:                           HyNomasTextRowProjectionProbe::STORAGE_ROWS.size,
      opaque_mps_plan_storage_included_in_named_bytes: false,
    },
    execution: {
      terminal_wait_count:             chain.wait_count,
      command_segments:                chain.segment_count,
      preterminal_readbacks:           preterminal_readbacks,
      post_wait_output_readbacks:      terminal_readbacks,
      named_allocations_during_encode: encode_allocations,
      q4_decode_dispatches:            1,
      bf16_to_f32_weight_dispatches:   1,
      input_repeat_dispatches:         HyNomasTextRowProjectionProbe::STORAGE_ROWS.size,
      mps_projection_dispatches:       HyNomasTextRowProjectionProbe::STORAGE_ROWS.size,
      comparison_after_terminal_wait:  true,
      diffusion_generation:            false,
    },
    comparisons: comparisons,
    passed:      passed,
  }

  owner.release_after_terminal!(chain)
  chain.release
  owner = nil
  chain = nil
  HyNomasQ4ReductionProbe.write_new_artifact!(report_path, result.to_pretty_json + "\n")
  puts({report: report_path, passed: passed, rows: HyNomasTextRowProjectionProbe::STORAGE_ROWS,
        named_bytes: required_named_bytes}.to_json)
  raise ArgumentError.new("MPS storage-row comparison gate failed") unless passed
ensure
  if active_chain = chain
    unless active_chain.completed?
      begin
        active_chain.commit_and_wait
      rescue ex
        STDERR.puts "Terminal cleanup failed; retain resources until exit: #{ex.message}"
      end
    end
    if active_chain.completed?
      owner.try { |current| current.encoded? ? current.release_after_terminal!(active_chain) : current.release_setup_only! }
      active_chain.release
    end
  else
    owner.try(&.release_setup_only!)
  end
end
