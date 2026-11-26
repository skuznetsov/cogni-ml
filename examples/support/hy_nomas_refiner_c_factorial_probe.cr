require "json"
require "digest/sha256"
require "path"
require "../../src/ml/core/buffer"
require "../../src/ml/video/checkpoint"
require "../../src/ml/video/hy_nomas_text_refiner"
require "../../src/ml/video/hy_nomas_continuation_history"
require "./hy_nomas_probe_instrumentation"
require "./hy_nomas_refiner_transfer_probe"

# Parent Fixture methods call helpers from their defining module. The old CLI
# included that module only at top level; include it explicitly here so the
# compositional admission works from this new entry point as well.
class HyNomasRefinerTransferProbe::Fixture
  include HyNomasRefinerTransferProbe
end

# Diagnostic-only extension used by the factorial probe. It deliberately
# mirrors the production prelude and block path, replacing only the computed
# `c` value after conditioning_add and before input_embed consumes it.
class ML::Video::HyNomasTextRefinerMPSPlan
  def encode_with_probe_c_override(chain : ML::Metal::MPSCommandChain,
                                   c_override : ML::MetalBuffer,
                                   c_copy_plan : ML::Video::HyNomasContinuationHistoryKernels::AppendPlan) : Nil
    validate_for!(chain)
    c_target = checkpoint("c")
    unless c_override.valid? && c_override.storage_mode.shared? &&
           c_override.size == HIDDEN_DIM.to_i64 * sizeof(Float32) &&
           c_target.size == c_override.size
      raise ArgumentError.new("factorial c override must be a shared F32 hidden-size buffer")
    end
    if @checkpoints.values.any? { |buffer| buffer.same?(c_override) }
      raise ArgumentError.new("factorial c override must not alias a refiner checkpoint")
    end
    ML::Video::MetalOps.distinct(c_target, c_override)
    @last_chain = chain

    encode_compaction(chain.encoder)
    encode_mean(chain.encoder)
    encode_sinusoidal(chain.encoder)
    encode_linear(chain, "refiner_timestep_mlp1")
    encode_activation(chain, "refiner_timestep_silu")
    encode_linear(chain, "refiner_timestep")
    encode_linear(chain, "context_linear1")
    encode_activation(chain, "context_silu")
    encode_linear(chain, "context_embed")
    encode_conditioning_add(chain.encoder)
    c_copy_plan.encode(chain.encoder, c_override, c_target, HIDDEN_DIM, 0)
    encode_linear(chain, "input_embed")
    encode_linear(chain, "time_in.mlp1")
    encode_activation(chain, "time_in.silu")
    encode_linear(chain, "time_in.mlp2")
    BLOCKS.times do |block|
      encode_block(chain, block)
    end
  end
end

module HyNomasRefinerCFactorialProbe
  extend self

  SCHEMA                  = "hy-nomas-refiner-c-factorial-reference-v1"
  PARENT_REFERENCE_PATH   = "/opt/mldata_ssd/cogni-ml/hy-nomas-refiner-transfer-20261105/source-v2.json"
  PARENT_REFERENCE_SHA256 = "8974bfce4bb557f84a1c75be168e4e7ddc623b6fd6552c796360771bd201a877"

  CASES              = ["A0", "A1", "B", "X10", "X01"]
  OUTPUT_CHECKPOINTS = HyNomasRefinerTransferProbe::OUTPUT_CHECKPOINTS
  INPUT_BYTES        = HyNomasRefinerTransferProbe::INPUT_BYTES
  VALID_ROWS         = HyNomasRefinerTransferProbe::VALID_ROWS
  INPUT_DIM          = HyNomasRefinerTransferProbe::INPUT_DIM
  HIDDEN_DIM         = HyNomasRefinerTransferProbe::HIDDEN_DIM
  ATOL               = HyNomasRefinerTransferProbe::ATOL
  RTOL               = HyNomasRefinerTransferProbe::RTOL
  SPANS              = HyNomasRefinerTransferProbe::SPANS
  MAX_MANIFEST_BYTES = 16_i64 * 1024 * 1024
  MAX_OUTPUT_BYTES   = 64_i64 * 1024 * 1024
  FACTORS            = {
    "A0"  => {text_case: "A0", c_case: "A0", override_c: false},
    "A1"  => {text_case: "A0", c_case: "A0", override_c: true},
    "B"   => {text_case: "B", c_case: "B", override_c: false},
    "X10" => {text_case: "B", c_case: "A0", override_c: true},
    "X01" => {text_case: "A0", c_case: "B", override_c: true},
  }

  record OutputTensor, name : String, shape : Array(Int32),
    offset_bytes : Int64, nbytes : Int64, sha256 : String

  record EffectStats, elements : Int32, changed_values : Int32,
    mean_effect : Float64?, max_abs : Float64?, rms_effect : Float64?,
    baseline_rms : Float64?, relative_rms_effect : Float64?,
    exact_zero : Bool, finite : Bool

  class Fixture
    include HyNomasRefinerCFactorialProbe

    getter path : String
    getter sha256 : String
    getter document : JSON::Any
    getter parent : HyNomasRefinerTransferProbe::Fixture
    getter output_payload_path : String
    getter output_payload_sha256 : String
    getter output_tensors : Hash(String, OutputTensor)
    getter expected_outputs : Hash(String, Array(Float32))

    def initialize(path : String, expected_sha256 : String)
      @path = canonical_regular_file(path, "factorial reference")
      raw = read_bounded(@path, MAX_MANIFEST_BYTES, "factorial reference")
      @sha256 = Digest::SHA256.hexdigest(raw)
      require_sha256!(expected_sha256, "trusted factorial reference SHA-256")
      raise ArgumentError.new("factorial reference SHA-256 mismatch") unless @sha256 == expected_sha256
      @document = parse_json(raw, "factorial reference")

      parent_reference = @document["parent_reference"]
      expect_exact_keys!(parent_reference, %w(path sha256), "factorial parent reference")
      require_sha256!(parent_reference["sha256"].as_s, "parent reference SHA-256")
      unless parent_reference["sha256"].as_s == PARENT_REFERENCE_SHA256 &&
             File.realpath(parent_reference["path"].as_s) == File.realpath(PARENT_REFERENCE_PATH)
        raise ArgumentError.new("factorial reference is not bound to the admitted parent source-v2")
      end
      # Reuse the complete parent admission, including all Qwen input manifests,
      # range hashes, raw-prefix equality, padding normalization and source
      # outputs, before loading this child output payload.
      @parent = HyNomasRefinerTransferProbe::Fixture.new(PARENT_REFERENCE_PATH,
        PARENT_REFERENCE_SHA256)
      validate_reference_document!
      @output_payload_path, @output_payload_sha256 = read_output_payload!
      @output_tensors = read_output_tensors!
      @expected_outputs = validate_output_payload!
      validate_source_summaries!
      validate_case_invariants!(group_outputs(@expected_outputs))
    end

    def expected_values(key : String) : Array(Float32)
      @expected_outputs[key]
    end

    private def validate_reference_document! : Nil
      expected_keys = %w(schema schema_version upstream_revision parent_reference source geometry timestep mask action_spans normalization precision execution case_factors payload tensors)
      expect_exact_keys!(@document, expected_keys, "factorial reference")
      unless @document["schema"].as_s == SCHEMA && @document["schema_version"].as_i == 1 &&
             @document["upstream_revision"].as_s == @parent.document["upstream_revision"].as_s
        raise ArgumentError.new("factorial reference schema or upstream pin mismatch")
      end
      {"source", "geometry", "timestep", "mask", "action_spans", "normalization", "precision"}.each do |key|
        unless @document[key] == @parent.document[key]
          raise ArgumentError.new("factorial reference changed inherited parent field: #{key}")
        end
      end
      validate_execution!
      validate_case_factors!(@document["case_factors"])
      payload = @document["payload"]
      expect_exact_keys!(payload, %w(basename bytes sha256 encoding), "factorial output payload")
      tensors = @document["tensors"].as_h
      expected = CASES.flat_map { |case_name| OUTPUT_CHECKPOINTS.map { |checkpoint| "#{case_name}.#{checkpoint}" } }.sort
      raise ArgumentError.new("factorial output tensor inventory mismatch") unless tensors.keys.sort == expected
    end

    private def validate_execution! : Nil
      execution = @document["execution"]
      expect_exact_keys!(execution,
        %w(case_order scope source_precision source_attention_query_chunks parent_replay source_aa),
        "factorial execution")
      parent_execution = @parent.document["execution"]
      unless execution["case_order"].as_a.map(&.as_s) == CASES &&
             execution["scope"].as_s == "refiner only; no full DiT" &&
             execution["source_precision"] == parent_execution["source_precision"] &&
             execution["source_attention_query_chunks"].as_i == parent_execution["source_attention_query_chunks"].as_i
        raise ArgumentError.new("factorial execution scope/provenance mismatch")
      end

      replay = execution["parent_replay"]
      expect_exact_keys!(replay, %w(A0 A1 B), "factorial parent replay")
      {"A0", "A1", "B"}.each do |case_name|
        case_records = replay[case_name]
        expect_exact_keys!(case_records, OUTPUT_CHECKPOINTS, "parent replay #{case_name}")
        OUTPUT_CHECKPOINTS.each do |checkpoint|
          record = case_records[checkpoint]
          validate_compare_record!(record, "parent.#{case_name}.#{checkpoint}",
            "#{case_name}.#{checkpoint}")
        end
      end

      source_aa = execution["source_aa"]
      expect_exact_keys!(source_aa, OUTPUT_CHECKPOINTS, "factorial source A/A")
      OUTPUT_CHECKPOINTS.each do |checkpoint|
        record = source_aa[checkpoint]
        validate_compare_record!(record, "A0/A1.#{checkpoint}", nil)
      end
    end

    private def validate_compare_record!(record : JSON::Any, label : String,
                                         tensor_key : String?) : Nil
      expect_exact_keys!(record,
        %w(checkpoint elements max_abs mean_abs rms_abs outside_atol_rtol all_finite passed atol rtol),
        "comparison record")
      checkpoint = record["checkpoint"].as_s
      raise ArgumentError.new("comparison checkpoint label mismatch: #{checkpoint}") unless checkpoint == label
      elements = tensor_key ? HyNomasRefinerTransferProbe.tensor_elements(tensor_key.not_nil!.split('.', 2)[1]) : HyNomasRefinerTransferProbe.tensor_elements(label.split('/', 2)[1].split('.', 2)[1])
      unless record["elements"].as_i == elements && record["all_finite"].as_bool &&
             record["passed"].as_bool && record["outside_atol_rtol"].as_i == 0 &&
             record["atol"].as_f == ATOL && record["rtol"].as_f == RTOL
        raise ArgumentError.new("factorial comparison record failed its A/A or parent-replay gate: #{label}")
      end
      %w(max_abs mean_abs rms_abs).each do |name|
        value = record[name].as_f
        raise ArgumentError.new("non-finite factorial comparison statistic: #{label}.#{name}") unless value.finite? && value >= 0.0
      end
    end

    private def validate_case_factors!(factors : JSON::Any) : Nil
      expect_exact_keys!(factors, CASES, "factorial case factors")
      CASES.each do |case_name|
        record = factors[case_name]
        expect_exact_keys!(record, %w(text_case c_case override_c), "factor map #{case_name}")
        expected = FACTORS[case_name]
        unless record["text_case"].as_s == expected[:text_case] &&
               record["c_case"].as_s == expected[:c_case] &&
               record["override_c"].as_bool == expected[:override_c]
          raise ArgumentError.new("factorial case semantics mismatch: #{case_name}")
        end
      end
    end

    private def validate_source_summaries! : Nil
      execution = @document["execution"]
      {"A0", "A1", "B"}.each do |case_name|
        OUTPUT_CHECKPOINTS.each do |checkpoint|
          record = execution["parent_replay"][case_name][checkpoint]
          child_values = @expected_outputs["#{case_name}.#{checkpoint}"]
          parent_values = @parent.expected_values("#{case_name}.#{checkpoint}")
          measured = HyNomasRefinerTransferProbe.compare_values(child_values, parent_values)
          validate_measured_record!(record, measured, child_values, parent_values,
            "parent.#{case_name}.#{checkpoint}")
        end
      end
      OUTPUT_CHECKPOINTS.each do |checkpoint|
        record = execution["source_aa"][checkpoint]
        left = @expected_outputs["A0.#{checkpoint}"]
        right = @expected_outputs["A1.#{checkpoint}"]
        measured = HyNomasRefinerTransferProbe.compare_values(left, right)
        validate_measured_record!(record, measured, left, right, "A0/A1.#{checkpoint}")
      end
    end

    private def validate_measured_record!(record : JSON::Any,
                                          measured : NamedTuple(failures: Int32, max_abs: Float64, raw_bit_differences: Int32, first_mismatch: Int32?, first_actual: Float32?, first_expected: Float32?, finite: Bool),
                                          left : Array(Float32), right : Array(Float32),
                                          label : String) : Nil
      raise ArgumentError.new("source replay/A/A comparison failed: #{label}") unless measured[:finite] && measured[:failures] == 0
      max_abs = measured[:max_abs]
      raise ArgumentError.new("source comparison max_abs does not match payload: #{label}") if (record["max_abs"].as_f - max_abs).abs > 1.0e-12
      sum_abs = 0.0_f64
      sum_sq = 0.0_f64
      left.each_with_index do |a, index|
        delta = (right[index].to_f64 - a.to_f64).abs
        sum_abs += delta
        sum_sq += delta * delta
      end
      mean_abs = sum_abs / left.size
      rms_abs = Math.sqrt(sum_sq / left.size)
      raise ArgumentError.new("source comparison mean_abs does not match payload: #{label}") if (record["mean_abs"].as_f - mean_abs).abs > Math.max(1.0e-12, mean_abs.abs * 1.0e-9)
      raise ArgumentError.new("source comparison rms_abs does not match payload: #{label}") if (record["rms_abs"].as_f - rms_abs).abs > Math.max(1.0e-12, rms_abs.abs * 1.0e-9)
    end

    private def read_output_payload! : Tuple(String, String)
      metadata = @document["payload"]
      name = safe_basename(metadata["basename"].as_s, "factorial output payload")
      path = canonical_regular_file(File.join(File.dirname(@path), name), "factorial output payload")
      bytes = metadata["bytes"].as_i64
      sha = metadata["sha256"].as_s
      require_sha256!(sha, "factorial output payload SHA-256")
      unless metadata["encoding"].as_s == "f32le_bf16_expansion" &&
             0 < bytes <= MAX_OUTPUT_BYTES && File.size(path) == bytes && sha256_file(path) == sha
        raise ArgumentError.new("factorial output payload metadata/hash mismatch")
      end
      {path, sha}
    end

    private def read_output_tensors! : Hash(String, OutputTensor)
      tensors = @document["tensors"].as_h
      result = {} of String => OutputTensor
      tensors.each do |name, item|
        case_name, checkpoint = name.split('.', 2)
        unless CASES.includes?(case_name) && OUTPUT_CHECKPOINTS.includes?(checkpoint)
          raise ArgumentError.new("unknown factorial tensor: #{name}")
        end
        expect_exact_keys!(item,
          %w(dtype source_dtype shape offset_bytes nbytes sha256 checkpoint_boundary),
          "factorial tensor #{name}")
        shape = @parent.output_tensors["A0.#{checkpoint}"].shape
        expected_nbytes = shape.reduce(1_i64) { |n, extent| n * extent } * sizeof(Float32)
        unless item["dtype"].as_s == "F32" && item["source_dtype"].as_s == "BF16" &&
               item["shape"].as_a.map(&.as_i) == shape &&
               item["checkpoint_boundary"].as_s == checkpoint && item["nbytes"].as_i64 == expected_nbytes
          raise ArgumentError.new("factorial tensor shape/dtype/boundary mismatch: #{name}")
        end
        sha = item["sha256"].as_s
        require_sha256!(sha, "factorial tensor SHA-256")
        result[name] = OutputTensor.new(name, shape, item["offset_bytes"].as_i64,
          item["nbytes"].as_i64, sha)
      end
      result
    end

    private def validate_output_payload! : Hash(String, Array(Float32))
      cursor = 0_i64
      @output_tensors.values.sort_by(&.offset_bytes).each do |tensor|
        unless tensor.offset_bytes == cursor && tensor.nbytes <= File.size(@output_payload_path) - cursor
          raise ArgumentError.new("factorial output tensors must be contiguous and disjoint")
        end
        cursor += tensor.nbytes
      end
      unless cursor == File.size(@output_payload_path)
        raise ArgumentError.new("factorial output payload has trailing or missing bytes")
      end
      raw = read_bounded(@output_payload_path, MAX_OUTPUT_BYTES, "factorial output payload")
      expected = {} of String => Array(Float32)
      @output_tensors.each do |name, tensor|
        piece = raw[tensor.offset_bytes.to_i32, tensor.nbytes.to_i32]
        raise ArgumentError.new("factorial tensor digest mismatch: #{name}") unless Digest::SHA256.hexdigest(piece) == tensor.sha256
        values = Array(Float32).new(tensor.nbytes.to_i32 // sizeof(Float32))
        (tensor.nbytes.to_i32 // sizeof(Float32)).times do |index|
          bits = IO::ByteFormat::LittleEndian.decode(UInt32, piece[index * 4, 4])
          value = bits.unsafe_as(Float32)
          unless value.finite? && (bits & 0xffff_u32) == 0
            raise ArgumentError.new("factorial output is not finite BF16-expanded F32: #{name}")
          end
          values << value
        end
        expected[name] = values
      end
      expected
    end

    private def canonical_regular_file(path : String, label : String) : String
      raise ArgumentError.new("#{label} must be a regular non-symlink file") unless File.file?(path) && !File.symlink?(path)
      File.realpath(path)
    end

    private def safe_basename(path : String, label : String) : String
      unless !path.empty? && path != "." && path != ".." && Path[path].basename == path &&
             !path.includes?('/') && !path.includes?('\\')
        raise ArgumentError.new("#{label} must be a sibling basename")
      end
      path
    end

    private def read_bounded(path : String, limit : Int64, label : String) : Bytes
      info = File.info(path)
      raise ArgumentError.new("#{label} exceeds its bounded size") unless info.size > 0 && info.size <= limit
      raw = File.read(path).to_slice.dup
      after = File.info(path)
      unless raw.size.to_i64 == info.size && after.same_file?(info) && after.size == info.size &&
             after.modification_time == info.modification_time
        raise ArgumentError.new("#{label} changed while being read")
      end
      raw
    end

    private def parse_json(raw : Bytes, label : String) : JSON::Any
      JSON.parse(String.new(raw))
    rescue ex : JSON::ParseException
      raise ArgumentError.new("#{label} is not valid JSON: #{ex.message}")
    end

    private def expect_exact_keys!(value : JSON::Any, keys : Array(String), label : String) : Nil
      raise ArgumentError.new("#{label} key inventory mismatch") unless value.as_h.keys.sort == keys.sort
    end

    private def require_sha256!(value : String, label : String) : Nil
      raise ArgumentError.new("#{label} must be lowercase SHA-256") unless value.matches?(/\A[0-9a-f]{64}\z/)
    end

    private def sha256_file(path : String) : String
      digest = Digest::SHA256.new
      slab = Bytes.new(8 * 1024 * 1024)
      File.open(path, "rb") do |io|
        loop do
          count = io.read(slab)
          break if count == 0
          digest.update(slab[0, count])
        end
      end
      digest.hexfinal
    end
  end

  def validate_case_factors!(factors : JSON::Any) : Nil
    unless factors.as_h.keys.sort == CASES.sort
      raise ArgumentError.new("factorial case-factor inventory mismatch")
    end
    CASES.each do |case_name|
      record = factors[case_name]
      unless record.as_h.keys.sort == %w(text_case c_case override_c).sort
        raise ArgumentError.new("factor map key inventory mismatch: #{case_name}")
      end
      expected = FACTORS[case_name]
      unless record["text_case"].as_s == expected[:text_case] &&
             record["c_case"].as_s == expected[:c_case] &&
             record["override_c"].as_bool == expected[:override_c]
        raise ArgumentError.new("factorial case semantics mismatch: #{case_name}")
      end
    end
  end

  def validate_case_invariants!(outputs : Hash(String, Hash(String, Array(Float32)))) : Nil
    failed = case_invariant_records(outputs).find { |record| !record[:exact] }
    raise ArgumentError.new("factorial exact-value invariant failed: #{failed.not_nil![:name]}") if failed
  end

  def case_invariant_records(outputs : Hash(String, Hash(String, Array(Float32)))) : Array(NamedTuple(name: String, left_case: String, right_case: String, checkpoint: String, elements: Int32, raw_bit_differences: Int32, finite: Bool, exact: Bool))
    raise ArgumentError.new("factorial case inventory mismatch") unless outputs.keys.sort == CASES.sort
    CASES.each do |case_name|
      raise ArgumentError.new("factorial checkpoint inventory mismatch: #{case_name}") unless outputs[case_name].keys.sort == OUTPUT_CHECKPOINTS.sort
    end
    records = [] of NamedTuple(name: String, left_case: String, right_case: String, checkpoint: String, elements: Int32, raw_bit_differences: Int32, finite: Bool, exact: Bool)
    OUTPUT_CHECKPOINTS.each do |checkpoint|
      records << exact_record("A0_A1_noop", "A0", "A1", checkpoint, outputs["A0"][checkpoint], outputs["A1"][checkpoint])
    end
    records << exact_record("X10_used_c_is_A0_c", "A0", "X10", "c", outputs["A0"]["c"], outputs["X10"]["c"])
    records << exact_record("X01_used_c_is_B_c", "B", "X01", "c", outputs["B"]["c"], outputs["X01"]["c"])
    records << exact_record("X10_mean_is_natural_B", "B", "X10", "masked_mean", outputs["B"]["masked_mean"], outputs["X10"]["masked_mean"])
    records << exact_record("X01_mean_is_natural_A", "A0", "X01", "masked_mean", outputs["A0"]["masked_mean"], outputs["X01"]["masked_mean"])
    CASES.each do |case_name|
      records << exact_record("vec_is_timestep_only", "A0", case_name, "vec", outputs["A0"]["vec"], outputs[case_name]["vec"])
    end
    records
  end

  private def exact_record(name : String, left_case : String, right_case : String,
                           checkpoint : String, left : Array(Float32),
                           right : Array(Float32)) : NamedTuple(name: String, left_case: String, right_case: String, checkpoint: String, elements: Int32, raw_bit_differences: Int32, finite: Bool, exact: Bool)
    comparison = HyNomasRefinerTransferProbe.compare_values(left, right)
    exact = comparison[:finite] && comparison[:failures] == 0 && comparison[:raw_bit_differences] == 0
    {name: name, left_case: left_case, right_case: right_case, checkpoint: checkpoint,
     elements: left.size.to_i32, raw_bit_differences: comparison[:raw_bit_differences],
     finite: comparison[:finite], exact: exact}
  end

  def group_outputs(flat : Hash(String, Array(Float32))) : Hash(String, Hash(String, Array(Float32)))
    grouped = {} of String => Hash(String, Array(Float32))
    CASES.each do |case_name|
      values = {} of String => Array(Float32)
      OUTPUT_CHECKPOINTS.each { |checkpoint| values[checkpoint] = flat["#{case_name}.#{checkpoint}"] }
      grouped[case_name] = values
    end
    grouped
  end

  def admission_self_test! : Nil
    HyNomasRefinerTransferProbe.admission_self_test!
    good = JSON.parse(%({"A0":{"text_case":"A0","c_case":"A0","override_c":false},"A1":{"text_case":"A0","c_case":"A0","override_c":true},"B":{"text_case":"B","c_case":"B","override_c":false},"X10":{"text_case":"B","c_case":"A0","override_c":true},"X01":{"text_case":"A0","c_case":"B","override_c":true}}))
    validate_case_factors!(good)
    bad = JSON.parse(%({"A0":{"text_case":"A0","c_case":"A0","override_c":false},"A1":{"text_case":"A0","c_case":"A0","override_c":true},"B":{"text_case":"B","c_case":"B","override_c":false},"X10":{"text_case":"B","c_case":"B","override_c":true},"X01":{"text_case":"A0","c_case":"B","override_c":true}}))
    rejected = false
    begin
      validate_case_factors!(bad)
    rescue ex : ArgumentError
      rejected = ex.message.try(&.includes?("semantics")) || false
    end
    raise "factorial factor-map self-test failed to reject incorrect X10 c case" unless rejected

    examples = {} of String => Hash(String, Array(Float32))
    CASES.each do |case_name|
      values = {} of String => Array(Float32)
      OUTPUT_CHECKPOINTS.each { |checkpoint| values[checkpoint] = [1.0_f32, 2.0_f32] }
      examples[case_name] = values
    end
    validate_case_invariants!(examples)
    examples["X10"]["c"][0] = 1.25_f32
    rejected = false
    begin
      validate_case_invariants!(examples)
    rescue ex : ArgumentError
      rejected = ex.message.try(&.includes?("X10_used_c_is_A0_c")) || false
    end
    raise "factorial invariant self-test failed to reject incorrect used-c checkpoint" unless rejected

    positive = HyNomasRefinerTransferProbe.compare_values([1.0_f32, 2.0_f32], [1.0_f32, 2.0_f32])
    negative = HyNomasRefinerTransferProbe.compare_values([1.0_f32, 2.0_f32], [1.0_f32, 2.25_f32])
    raise "factorial numeric self-test positive control failed" unless positive[:failures] == 0
    raise "factorial numeric self-test did not detect a seeded mismatch" unless negative[:failures] == 1

    interaction = interaction_stats([1.0_f32], [8.0_f32], [2.0_f32], [3.0_f32])
    unless interaction.finite && interaction.mean_effect == 4.0 &&
           interaction.max_abs == 4.0 && interaction.rms_effect == 4.0
      raise "factorial interaction self-test did not compute B-X10-X01+A0 elementwise"
    end
  end

  def effect_stats(baseline : Array(Float32), changed : Array(Float32)) : EffectStats
    raise ArgumentError.new("factor effect size mismatch") unless baseline.size == changed.size
    deltas = Array(Float64).new(baseline.size) { |index| changed[index].to_f64 - baseline[index].to_f64 }
    summarize_effect(baseline, deltas)
  end

  def interaction_stats(a0 : Array(Float32), b : Array(Float32),
                        x10 : Array(Float32), x01 : Array(Float32)) : EffectStats
    unless a0.size == b.size && a0.size == x10.size && a0.size == x01.size
      raise ArgumentError.new("factor interaction size mismatch")
    end
    deltas = Array(Float64).new(a0.size) do |index|
      # Keep this exact Float64 operation order; do not subtract aggregated
      # scalar statistics or round the factorial contrast through Float32.
      b[index].to_f64 - x10[index].to_f64 - x01[index].to_f64 + a0[index].to_f64
    end
    summarize_effect(a0, deltas)
  end

  private def summarize_effect(baseline : Array(Float32), deltas : Array(Float64)) : EffectStats
    sum = 0.0_f64
    sum_sq = 0.0_f64
    base_sq = 0.0_f64
    max_abs = 0.0_f64
    changed = 0_i32
    finite = true
    baseline.each_with_index do |base, index|
      delta = deltas[index]
      unless base.finite? && delta.finite?
        finite = false
        next
      end
      sum += delta
      sum_sq += delta * delta
      base_sq += base.to_f64 * base.to_f64
      max_abs = Math.max(max_abs, delta.abs)
      changed += 1 if delta != 0.0
    end
    return EffectStats.new(baseline.size.to_i32, changed, nil, nil, nil, nil, nil,
      changed == 0, false) unless finite
    mean = sum / baseline.size
    rms = Math.sqrt(sum_sq / baseline.size)
    base_rms = Math.sqrt(base_sq / baseline.size)
    relative = base_rms > 0.0 ? rms / base_rms : nil
    EffectStats.new(baseline.size.to_i32, changed, mean, max_abs, rms,
      base_rms, relative, changed == 0, true)
  end

  # The factorial decomposition is numeric and conditional: X10 uses B text
  # with A c; X01 uses A text with B c. This does not itself establish semantic
  # or perceptual control.
end
