# Fixed-fixture upstream qualification for the standalone Wan S2V FFN epilogue
# probe. This admits only block 0's first FFN projection weight and bias.
require "digest/sha256"
require "json"
require "../../src/ml/video/checkpoint"
require "../../src/ml/video/metal_linear_mps"
require "../../spec/support/video_allocation_counter"

module WanFFNEpilogueUpstream
  MANIFEST_SHA256 = "fb7943d637bc8a6853707cd2ed284595e6744c4368083c36993fd1dd26d78da9"
  WEIGHTS_SHA256  = "c172519063c1c34fb36f7f930429ca7db655a0e335524d7487525db1df5ad220"
  EXPLICIT_CAP    = 1_i64 * 1024 * 1024 * 1024
  BATCHES         = 2
  TOKENS          = 2
  ROWS            = BATCHES * TOKENS
  INPUT_DIM       = 5_120
  FFN_DIM         = 13_824
  VALUE_COUNT     = ROWS * FFN_DIM
  METRIC_NAMES    = {
    "trace_vs_mps_bias", "trace_vs_expected_ffn_in", "mps_bias_vs_expected_ffn_in",
    "gelu_candidate_vs_baseline", "gelu_baseline_vs_expected", "gelu_candidate_vs_expected",
  }
  BITWISE_NAMES = {"trace_vs_mps_bias", "gelu_candidate_vs_baseline"}

  def self.sha256_file(path : String) : String
    digest = Digest::SHA256.new
    buffer = Bytes.new(1 << 20)
    File.open(path, "rb") do |file|
      while (count = file.read(buffer)) > 0
        digest.update(buffer[0, count])
      end
    end
    digest.final.hexstring
  end

  def self.f32_values(value : JSON::Any, name : String, expected_count : Int32) : Array(Float32)
    values = value.as_a
    unless values.size == expected_count
      raise ArgumentError.new("Upstream fixture #{name} length mismatch")
    end
    values.map do |element|
      number = element.as_f.to_f32
      raise ArgumentError.new("Upstream fixture #{name} contains non-finite values") unless number.finite?
      number
    end
  end

  def self.validate_binding!(manifest : JSON::Any, source : String,
                             shape : Array(Int64)) : Nil
    matches = manifest["weight_bindings"].as_a.select do |binding|
      binding["source"].as_s == source
    end
    raise ArgumentError.new("Upstream fixture must bind #{source} exactly once") unless matches.size == 1
    binding = matches.first
    unless binding["dtype"].as_s == "BF16" && binding["shape"].as_a.map(&.as_i64) == shape
      raise ArgumentError.new("Upstream fixture binding mismatch for #{source}")
    end
  end

  # All device storage and persistent MPS state are owned here until the root
  # command buffer's sole terminal wait. The caller owns command submission.
  class Prepared
    @arena : ML::Video::Arena
    @checkpoint : ML::Video::CheckpointFile?
    @prepared_weights : ML::Video::PreparedWeights?
    @plan : ML::Video::WanT5MPSLinearPlan?
    @metrics = {} of String => ML::MetalBuffer
    @bias_pipeline : ML::Metal::ComputePipeline
    @fixture_manifest_path : String
    @manifest_sha256 : String
    @weights_path : String
    @weights_sha256 : String
    @encoded = false
    @released = false

    def initialize(fixture_path : String,
                   @fused_trace_pipeline : ML::Metal::ComputePipeline,
                   @compare_pipeline : ML::Metal::ComputePipeline,
                   @zero_pipeline : ML::Metal::ComputePipeline)
      @checkpoint = nil
      @prepared_weights = nil
      @plan = nil
      @encoded = false
      @released = false
      manifest_path = File.expand_path(fixture_path)
      raise ArgumentError.new("Missing upstream fixture manifest") unless File.file?(manifest_path)
      @fixture_manifest_path = manifest_path
      @manifest_sha256 = WanFFNEpilogueUpstream.sha256_file(manifest_path)
      raise ArgumentError.new("Pinned upstream manifest SHA-256 mismatch") unless @manifest_sha256 == MANIFEST_SHA256
      manifest = JSON.parse(File.read(manifest_path))
      validate_manifest!(manifest)

      weights_file = manifest["weights_file"]
      weights_name = weights_file["path"].as_s
      unless File.basename(weights_name) == weights_name && !weights_name.empty?
        raise ArgumentError.new("Upstream checkpoint path must be a fixture-sidecar basename")
      end
      @weights_path = File.join(File.dirname(manifest_path), weights_name)
      raise ArgumentError.new("Missing upstream checkpoint sidecar") unless File.file?(@weights_path)
      expected_weights_size = weights_file["bytes"].as_i64
      unless File.size(@weights_path) == expected_weights_size
        raise ArgumentError.new("Upstream checkpoint sidecar size mismatch")
      end
      @weights_sha256 = weights_file["sha256"].as_s
      unless @weights_sha256 == WEIGHTS_SHA256 &&
             WanFFNEpilogueUpstream.sha256_file(@weights_path) == @weights_sha256
        raise ArgumentError.new("Upstream checkpoint sidecar SHA-256 mismatch")
      end

      expected = manifest["expected"]
      input_values = WanFFNEpilogueUpstream.f32_values(expected["ffn_mod"], "ffn_mod", ROWS * INPUT_DIM)
      ffn_in_values = WanFFNEpilogueUpstream.f32_values(expected["ffn_in"], "ffn_in", VALUE_COUNT)
      gelu_values = WanFFNEpilogueUpstream.f32_values(expected["ffn_gelu"], "ffn_gelu", VALUE_COUNT)

      @bias_pipeline = ML::Video::MetalLinearMPS.pipeline
      ML::Video::MetalOps.pipeline("video_copy_f32")
      ML::Video::MetalOps.pipeline("video_gelu_f32")
      ML::Video::WanT5.pipeline("wan_t5_expand_fc2")
      @arena = ML::Video::Arena.new
      begin
        input = @arena.alloc("upstream_ffn_mod", ROWS * INPUT_DIM)
        raw_projection = @arena.alloc("upstream_raw_projection", VALUE_COUNT)
        @arena.alloc("upstream_baseline", VALUE_COUNT)
        @arena.alloc("upstream_candidate", VALUE_COUNT)
        @arena.alloc("upstream_pre_gelu_trace", VALUE_COUNT)
        expected_ffn_in = @arena.alloc("upstream_expected_ffn_in", VALUE_COUNT)
        expected_gelu = @arena.alloc("upstream_expected_ffn_gelu", VALUE_COUNT)
        @arena.alloc("upstream_weight_stage_f32", INPUT_DIM * FFN_DIM)
        input.write(input_values)
        expected_ffn_in.write(ffn_in_values)
        expected_gelu.write(gelu_values)

        METRIC_NAMES.each do |name|
          @metrics[name] = @arena.alloc("upstream_metrics_#{name}", 5)
        end

        checkpoint = ML::Video::CheckpointFile.new(@weights_path)
        @checkpoint = checkpoint
        bindings = [
          ML::Video::WeightBinding.new("ffn_in_weight", "blocks.0.ffn.0.weight",
            [FFN_DIM.to_i64, INPUT_DIM.to_i64], ML::GGUF::SafeTensorDType::BF16),
          ML::Video::WeightBinding.new("ffn_in_bias", "blocks.0.ffn.0.bias",
            [FFN_DIM.to_i64], ML::GGUF::SafeTensorDType::BF16),
        ]
        weight_plan = ML::Video::WeightPlan.new([checkpoint], bindings,
          EXPLICIT_CAP, @arena.resident_bytes, ML::Video::WeightDType::BF16, require_all: false)
        unless weight_plan.entries.size == 2 && weight_plan.destination_bytes ==
               (FFN_DIM.to_i64 * INPUT_DIM + FFN_DIM) * sizeof(UInt16)
          raise ArgumentError.new("Upstream weight plan admitted unexpected tensors")
        end
        prepared = ML::Video::PreparedWeights.new(weight_plan, @arena)
        @prepared_weights = prepared
        cfg = ML::Video::WanT5Config.new(vocab: 8, dim: INPUT_DIM,
          ffn_dim: FFN_DIM, heads: 40, layers: 1, batches: BATCHES, tokens: TOKENS)
        @plan = ML::Video::WanT5MPSLinearPlan.new(cfg, INPUT_DIM, FFN_DIM,
          input, @arena["upstream_weight_stage_f32"], raw_projection)
        raise "Upstream FFN explicit allocation cap exceeded" if @arena.resident_bytes > EXPLICIT_CAP
      rescue ex
        release_after_terminal_wait!
        raise ex
      end
    end

    def setup_json : String
      {
        schema: "wan_s2v_ffn_epilogue_upstream_setup_v1",
        domain: "wan.s2v.ffn.epilogue",
        fixture_manifest_path: @fixture_manifest_path,
        fixture_manifest_sha256: @manifest_sha256,
        source_checkpoint_path: @weights_path,
        source_checkpoint_sha256: @weights_sha256,
        rows: ROWS, input_dim: INPUT_DIM, ffn_dim: FFN_DIM,
        source_weight_dtype: "BF16",
        admitted_checkpoint_tensors: ["blocks.0.ffn.0.weight", "blocks.0.ffn.0.bias"],
        require_all_checkpoint_tensors: false,
        explicit_buffer_bytes: @arena.resident_bytes,
        explicit_buffer_cap_bytes: EXPLICIT_CAP,
        command_completion_waits_owned_here: 0,
        activation_readbacks: 0,
        scope: "fixed ffn_mod slice parity; not a full block or model run",
      }.to_json
    end

    # Appends projection, cloned post-MPS arms, trace/fused and baseline checks
    # to the caller's open encoder. No wait, submit, activation readback, or
    # MetalBuffer allocation occurs in this method.
    def encode!(encoder : ML::Metal::ComputeEncoder) : Nil
      raise "Upstream FFN probe is already encoded or released" if @encoded || @released
      input = @arena["upstream_ffn_mod"]
      raw_projection = @arena["upstream_raw_projection"]
      baseline = @arena["upstream_baseline"]
      candidate = @arena["upstream_candidate"]
      trace = @arena["upstream_pre_gelu_trace"]
      expected_ffn_in = @arena["upstream_expected_ffn_in"]
      expected_gelu = @arena["upstream_expected_ffn_gelu"]
      weight = @arena["ffn_in_weight"]
      bias = @arena["ffn_in_bias"]
      allocation_count = ML::MetalBuffer.video_spec_allocations
      @prepared_weights.not_nil!.encode(encoder)
      ML::Video::WanT5.mps_projection(encoder, input, weight,
        raw_projection, ROWS, INPUT_DIM, FFN_DIM, ML::Video::WeightDType::BF16,
        @plan.not_nil!, @arena["upstream_weight_stage_f32"])
      ML::Video::MetalOps.copy(encoder, raw_projection, baseline, VALUE_COUNT)
      ML::Video::MetalOps.copy(encoder, raw_projection, candidate, VALUE_COUNT)

      encoder.set_pipeline(@bias_pipeline)
      encoder.set_buffer(baseline, 0, ML::Metal::BufferAccess::ReadWrite)
      encoder.set_buffer(bias, 1).set_buffer(bias, 2)
      encoder.set_value(FFN_DIM.to_u32, 3).set_value(1_u32, 4)
      encoder.set_value(VALUE_COUNT.to_u32, 5).dispatch_1d(VALUE_COUNT)

      encoder.set_pipeline(@fused_trace_pipeline)
      encoder.set_buffer(candidate, 0, ML::Metal::BufferAccess::ReadWrite)
      encoder.set_buffer(trace, 1, ML::Metal::BufferAccess::Write)
      encoder.set_buffer(bias, 2).set_buffer(bias, 3)
      encoder.set_value(FFN_DIM.to_u32, 4).set_value(1_u32, 5)
      encoder.set_value(VALUE_COUNT.to_u32, 6).dispatch_1d(VALUE_COUNT)

      encode_compare(encoder, "trace_vs_mps_bias", baseline, trace)
      encode_compare(encoder, "trace_vs_expected_ffn_in", expected_ffn_in, trace)
      encode_compare(encoder, "mps_bias_vs_expected_ffn_in", expected_ffn_in, baseline)

      ML::Video::MetalOps.gelu(encoder, baseline, baseline, VALUE_COUNT, approximate: true)
      encode_compare(encoder, "gelu_candidate_vs_baseline", baseline, candidate)
      encode_compare(encoder, "gelu_baseline_vs_expected", expected_gelu, baseline)
      encode_compare(encoder, "gelu_candidate_vs_expected", expected_gelu, candidate)
      unless ML::MetalBuffer.video_spec_allocations == allocation_count
        raise "Upstream FFN encode allocated a MetalBuffer"
      end
      @encoded = true
    end

    # The caller invokes this only after its single completion wait. It reads
    # six compact 5-u32 receipts and fails closed on non-finite or tolerance
    # failures, plus bitwise failures for the two explicitly exact comparisons.
    def validate_after_terminal_wait! : Nil
      raise "Upstream FFN probe has not been encoded" unless @encoded
      raise "Upstream FFN probe has been released" if @released
      failures = [] of String
      METRIC_NAMES.each do |name|
        words = @metrics[name].contents.as(UInt32*)
        bit_differences = words[0]
        tolerance_violations = words[1]
        nonfinite = words[2]
        max_abs = words[3].unsafe_as(Float32)
        max_ratio = words[4].unsafe_as(Float32)
        bitwise_required = BITWISE_NAMES.includes?(name)
        passed = tolerance_violations == 0 && nonfinite == 0 &&
          (!bitwise_required || bit_differences == 0)
        puts({
          schema: "wan_s2v_ffn_epilogue_upstream_check_v1",
          domain: "wan.s2v.ffn.epilogue",
          check: name, rows: ROWS, input_dim: INPUT_DIM, ffn_dim: FFN_DIM,
          source_weight_dtype: "BF16", compared_values: VALUE_COUNT,
          bit_differences: bit_differences,
          strict_tolerance_violations: tolerance_violations,
          nonfinite: nonfinite, max_abs_error: max_abs,
          max_normalized_error: max_ratio,
          tolerance: "2e-5+2e-5*abs(reference)",
          bitwise_required: bitwise_required, passed: passed,
          scope: "fixed ffn_mod slice parity only",
        }.to_json)
        failures << name unless passed
      end
      STDOUT.flush
      raise "Upstream FFN epilogue qualification failed: #{failures.join(", ")}" unless failures.empty?
    end

    # Must follow the root's terminal wait because the MPS plan borrows all
    # bound storage and the prepared weights use the same resident arena.
    def release_after_terminal_wait! : Nil
      return if @released
      @plan.try(&.release)
      @plan = nil
      @prepared_weights.try(&.release)
      @prepared_weights = nil
      @arena.release
      @checkpoint.try(&.close)
      @checkpoint = nil
      @released = true
    end

    private def validate_manifest!(manifest : JSON::Any) : Nil
      config = manifest["config"]
      unless config["batches"].as_i == BATCHES && config["seq"].as_i == TOKENS &&
             config["dim"].as_i == INPUT_DIM && config["ffn_dim"].as_i == FFN_DIM &&
             config["heads"].as_i == 40
        raise ArgumentError.new("Upstream fixture FFN geometry mismatch")
      end
      upstream = manifest["upstream"]
      unless upstream["oracle_device"].as_s == "mps" && upstream["fallback_disabled"].as_bool
        raise ArgumentError.new("Upstream fixture is not a fallback-disabled MPS oracle")
      end
      weights = manifest["weights_file"]
      source = upstream["checkpoint_source"]
      unless weights["dtype"].as_s == "BF16" && weights["sha256"].as_s == WEIGHTS_SHA256 &&
             weights["path"].as_s == "wan-official.weights.safetensors" &&
             source["dtype"].as_s == "BF16" && source["sha256"].as_s == WEIGHTS_SHA256 &&
             source["bytes"].as_i64 == weights["bytes"].as_i64
        raise ArgumentError.new("Upstream fixture checkpoint identity mismatch")
      end
      WanFFNEpilogueUpstream.validate_binding!(manifest,
        "blocks.0.ffn.0.weight", [FFN_DIM.to_i64, INPUT_DIM.to_i64])
      WanFFNEpilogueUpstream.validate_binding!(manifest,
        "blocks.0.ffn.0.bias", [FFN_DIM.to_i64])
      expected = manifest["expected"]
      raise ArgumentError.new("Upstream ffn_mod shape mismatch") unless expected["ffn_mod"].as_a.size == ROWS * INPUT_DIM
      raise ArgumentError.new("Upstream ffn_in shape mismatch") unless expected["ffn_in"].as_a.size == VALUE_COUNT
      raise ArgumentError.new("Upstream ffn_gelu shape mismatch") unless expected["ffn_gelu"].as_a.size == VALUE_COUNT
    end

    private def encode_compare(encoder : ML::Metal::ComputeEncoder, name : String,
                               reference : ML::MetalBuffer, actual : ML::MetalBuffer) : Nil
      metrics = @metrics[name]
      encoder.set_pipeline(@zero_pipeline).set_buffer(metrics, 0, ML::Metal::BufferAccess::Write).dispatch_1d(5)
      encoder.set_pipeline(@compare_pipeline)
      encoder.set_buffer(reference, 0).set_buffer(actual, 1)
      encoder.set_buffer(metrics, 2, ML::Metal::BufferAccess::ReadWrite)
      encoder.set_value(VALUE_COUNT.to_u32, 3).dispatch_1d(VALUE_COUNT)
    end
  end
end
