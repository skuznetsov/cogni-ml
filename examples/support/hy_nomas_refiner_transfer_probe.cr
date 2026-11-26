require "json"
require "digest/sha256"
require "path"
require "../../src/ml/core/buffer"
require "../../src/ml/video/checkpoint"
require "../../src/ml/video/hy_nomas_text_refiner"
require "../../src/ml/video/hy_nomas_continuation_history"
require "./hy_nomas_probe_instrumentation"

# Authenticated paired-input refiner probe. The CPU side admits immutable source
# tensors and expected outputs; all refiner work and case-to-case input updates
# are GPU encoded on one serial command chain.
module HyNomasRefinerTransferProbe
  extend self

  SCHEMA                       = "hy-nomas-refiner-transfer-reference-v1"
  INPUT_SCHEMA                 = "hy-nomas-refiner-transfer-input-v1"
  UPSTREAM_REVISION            = "911cea6458a89c2663513ff7099561ec190dd491"
  TEXT_WEIGHTS_SHA256          = "58313f6141adafc12629b46c73f1bc270aa64c3e0972a8995875ba76d9068d8f"
  TEXT_WEIGHTS_MANIFEST_SHA256 = "0863d49ba53d54db5ad7ff5313099c3e93003cf304e729f4a377098f230f462f"
  TEXT_WEIGHTS_BYTES           = 583_316_840_i64
  QWEN_SOURCE_REVISION         = "eb6ba55b40ada64977147b52e50a4145ee89ac4f"
  QWEN_SOURCE_SCHEMA           = "cogni_ml.qwen25_vl_pretrained_decoder_prefix_reference"
  SOURCE_QUERY_CHUNK_TOKENS    =    256
  ATOL                         = 2.0e-5
  RTOL                         = 2.0e-5
  TOKEN_COUNT                  =   1280
  INPUT_DIM                    =   3584
  HIDDEN_DIM                   =   2048
  VALID_ROWS                   =    224
  PREFIX_END                   =    132
  ACTION2_START                =    132
  ACTION2_END                  =    224
  INPUT_BYTES                  = TOKEN_COUNT.to_i64 * INPUT_DIM * 2
  INPUT_ROW_BYTES              = INPUT_DIM * 2
  OUTPUT_CHECKPOINTS           = ["masked_mean", "c", "vec", "block0.output", "block1.output"]
  CASES                        = ["A0", "A1", "B"]
  SPANS                        = {
    "global"  => {0, 49},
    "action1" => {49, 132},
    "action2" => {132, 224},
  }
  SOURCE_BF16_CONTROLS = [
    "prelude_input_bf16_control",
    "prelude_linear_bf16_control",
    "prelude_silu_bf16_control",
    "prelude_second_linear_bf16_control",
    "prelude_add_bf16_control",
    "block_boundaries_bf16_control",
    "qkv_bf16_control",
    "source_attention_control",
    "source_projection_residual_control",
    "source_norm2_control",
    "source_fc1_control",
    "source_mlp_silu_control",
    "source_fc2_residual_control",
    "source_mean_autocast",
  ]
  MAX_MANIFEST_BYTES = 16_i64 * 1024 * 1024
  MAX_OUTPUT_BYTES   = 64_i64 * 1024 * 1024
  BF16_COPY_KERNEL   = "hy_nomas_refiner_transfer_copy_bf16"
  BF16_COPY_SOURCE   = ML::Video::MetalOps::SOURCE + <<-METAL
    kernel void hy_nomas_refiner_transfer_copy_bf16(
        device const ushort *source [[buffer(0)]],
        device ushort *destination [[buffer(1)]],
        constant uint &count [[buffer(2)]],
        uint index [[thread_position_in_grid]]) {
      if (index < count) destination[index] = source[index];
    }
    METAL

  record OutputTensor, name : String, shape : Array(Int32),
    offset_bytes : Int64, nbytes : Int64, sha256 : String

  record InputCase, manifest_path : String, manifest_sha256 : String,
    payload_path : String, payload_sha256 : String, payload_bytes : Int64,
    source_tensor_sha256 : String, normalized_tensor_sha256 : String,
    source_bytes : Bytes, normalized_bytes : Bytes

  record Comparison, case_name : String, checkpoint : String,
    elements : Int32, failures : Int32, max_abs : Float64,
    raw_bit_differences : Int32, first_mismatch : Int32?,
    first_actual : Float32?, first_expected : Float32?, finite : Bool

  record DeltaStats, elements : Int32, changed_values : Int32,
    max_abs : Float64?, rms_delta : Float64?, baseline_rms : Float64?,
    relative_rms_delta : Float64?, exact_zero : Bool, finite : Bool

  class Fixture
    getter path : String
    getter sha256 : String
    getter document : JSON::Any
    getter output_payload_path : String
    getter output_payload_sha256 : String
    getter output_tensors : Hash(String, OutputTensor)
    getter inputs : Hash(String, InputCase)
    getter text_weights_path : String
    getter text_weights_sha256 : String
    getter text_weights_manifest_path : String
    getter text_weights_manifest_sha256 : String
    getter text_weights_stored_dtype : String
    getter text_weights_info : File::Info
    getter timestep : Float32
    getter mask : Array(Int32)
    getter input_action2_changed_bytes : Int32
    getter input_action2_changed_rows : Int32
    getter raw_padding_changed_bytes : Int32
    getter expected_outputs : Hash(String, Array(Float32))

    def initialize(path : String, expected_sha256 : String)
      @path = canonical_regular_file(path, "reference manifest")
      raw = read_bounded(@path, MAX_MANIFEST_BYTES, "reference manifest")
      @sha256 = Digest::SHA256.hexdigest(raw)
      require_sha256!(expected_sha256, "trusted reference SHA-256")
      raise ArgumentError.new("reference manifest SHA-256 mismatch") unless @sha256 == expected_sha256
      @document = parse_json(raw, "refiner transfer reference")
      validate_reference_document!

      source = @document["source"]
      weights = source["text_weights"]
      @text_weights_path = canonical_regular_file(weights["path"].as_s, "text refiner weights")
      @text_weights_info = File.info(@text_weights_path)
      @text_weights_sha256 = weights["sha256"].as_s
      require_sha256!(@text_weights_sha256, "text refiner weights SHA-256")
      raise ArgumentError.new("text refiner weight byte count mismatch") unless File.size(@text_weights_path) == weights["bytes"].as_i64 && weights["bytes"].as_i64 == TEXT_WEIGHTS_BYTES
      raise ArgumentError.new("text refiner weights SHA-256 mismatch") unless sha256_file(@text_weights_path) == @text_weights_sha256
      current_weights_info = File.info(@text_weights_path)
      unless current_weights_info.same_file?(@text_weights_info) &&
             current_weights_info.size == @text_weights_info.size &&
             current_weights_info.modification_time == @text_weights_info.modification_time
        raise ArgumentError.new("text refiner weights changed during admission")
      end
      @text_weights_manifest_path = canonical_regular_file(weights["manifest_path"].as_s, "text weight manifest")
      @text_weights_manifest_sha256 = weights["manifest_sha256"].as_s
      require_sha256!(@text_weights_manifest_sha256, "text weight manifest SHA-256")
      unless @text_weights_sha256 == TEXT_WEIGHTS_SHA256 && @text_weights_manifest_sha256 == TEXT_WEIGHTS_MANIFEST_SHA256
        raise ArgumentError.new("source refiner weight pin mismatch")
      end
      raise ArgumentError.new("text weight manifest SHA-256 mismatch") unless sha256_file(@text_weights_manifest_path) == @text_weights_manifest_sha256
      @text_weights_stored_dtype = weights["stored_dtype"].as_s
      raise ArgumentError.new("source text weights must load as BF16") unless weights["loaded_dtype"].as_s == "BF16"
      raise ArgumentError.new("unsupported stored text weight dtype") unless {"BF16", "F32"}.includes?(@text_weights_stored_dtype)
      raise ArgumentError.new("text refiner weight tensor count mismatch") unless weights["tensor_count"].as_i == 42
      raise ArgumentError.new("text refiner F32-to-BF16 conversion contract mismatch") unless weights["stored_dtype"].as_s == "F32" && weights["conversion"].as_s == "F32_to_BF16_RNE"

      @timestep = @document["timestep"].as_f.to_f32
      raise ArgumentError.new("refiner transfer timestep must be 1000") unless @timestep == 1000_f32
      @mask = Array(Int32).new(TOKEN_COUNT) { |index| index < VALID_ROWS ? 1 : 0 }

      @inputs = {} of String => InputCase
      raw_by_case = {} of String => Bytes
      # A0/A1 bind the same immutable input fixture. Reuse the source-side
      # admission instead of hashing the large Qwen sidecar and generation
      # bundle twice.
      validate_aa_input_binding!
      @inputs["A0"] = read_input_case("A0")
      @inputs["A1"] = @inputs["A0"]
      raw_by_case["A0"] = @inputs["A0"].source_bytes
      raw_by_case["A1"] = @inputs["A1"].source_bytes
      @inputs["B"] = read_input_case("B")
      raw_by_case["B"] = @inputs["B"].source_bytes
      @input_action2_changed_bytes, @input_action2_changed_rows, @raw_padding_changed_bytes =
        validate_original_input_pair!(raw_by_case["A0"], raw_by_case["B"])
      validate_prefix_checks!

      # Input bytes are normalized only after the raw A/B prefix and Action2
      # intervention have been validated. The payload copy is then checked
      # against this independently produced normalized tensor.
      CASES.each do |case_name|
        input_case = @inputs[case_name]
        normalized = normalize_padding(input_case.source_bytes)
        raise ArgumentError.new("normalized Qwen tensor SHA mismatch: #{case_name}") unless Digest::SHA256.hexdigest(normalized) == input_case.normalized_tensor_sha256
        raise ArgumentError.new("normalized payload differs from validated source rows: #{case_name}") unless normalized == input_case.normalized_bytes
        @inputs[case_name] = InputCase.new(input_case.manifest_path,
          input_case.manifest_sha256, input_case.payload_path,
          input_case.payload_sha256, input_case.payload_bytes,
          input_case.source_tensor_sha256, input_case.normalized_tensor_sha256,
          input_case.source_bytes, normalized)
      end
      raise ArgumentError.new("A/A normalized inputs differ") unless @inputs["A0"].normalized_bytes == @inputs["A1"].normalized_bytes
      raise ArgumentError.new("normalized Action2 intervention disappeared") unless @inputs["A1"].normalized_bytes != @inputs["B"].normalized_bytes

      @output_payload_path, @output_payload_sha256 = read_output_payload!
      @output_tensors = read_output_tensors!
      @expected_outputs = validate_output_payload!
    end

    def normalized_input(case_name : String) : Bytes
      @inputs[case_name].normalized_bytes
    end

    def expected_values(key : String) : Array(Float32)
      @expected_outputs[key]
    end

    private def validate_reference_document! : Nil
      expect_exact_keys!(@document,
        %w(schema schema_version upstream_revision source geometry timestep mask action_spans inputs normalization precision execution payload tensors),
        "reference document")
      unless @document["schema"].as_s == SCHEMA && @document["schema_version"].as_i == 1 &&
             @document["upstream_revision"].as_s == UPSTREAM_REVISION
        raise ArgumentError.new("refiner transfer reference schema or pinned revision mismatch")
      end
      validate_source!
      validate_geometry!
      validate_action_spans!
      validate_mask!
      validate_normalization!
      validate_execution!
      validate_precision!
      payload = @document["payload"]
      expect_exact_keys!(payload, %w(basename bytes sha256 encoding), "reference payload")
      tensors = @document["tensors"]
      expected_tensors = CASES.flat_map { |case_name| OUTPUT_CHECKPOINTS.map { |checkpoint| "#{case_name}.#{checkpoint}" } }
      raise ArgumentError.new("reference output tensor inventory mismatch") unless tensors.as_h.keys.sort == expected_tensors.sort
    end

    private def validate_source! : Nil
      source = @document["source"]
      expect_exact_keys!(source,
        %w(upstream_path revision source_files text_weights source_checkpoint module_bindings),
        "reference source")
      raise ArgumentError.new("source upstream path must be absolute") unless source["upstream_path"].as_s.starts_with?("/")
      raise ArgumentError.new("source revision pin mismatch") unless source["revision"].as_s == UPSTREAM_REVISION
      files = source["source_files"]
      raise ArgumentError.new("source file hash inventory must contain 11 files") unless files.as_h.size == 11
      files.as_h.each do |name, digest|
        raise ArgumentError.new("source file path must be relative") if name.starts_with?("/")
        require_sha256!(digest.as_s, "source file SHA-256")
      end
      weights = source["text_weights"]
      expect_exact_keys!(weights,
        %w(path sha256 bytes manifest_path manifest_sha256 tensor_count stored_dtype loaded_dtype conversion),
        "source text weights")
      raise ArgumentError.new("source weight pin mismatch") unless weights["sha256"].as_s == TEXT_WEIGHTS_SHA256 &&
                                                                   weights["manifest_sha256"].as_s == TEXT_WEIGHTS_MANIFEST_SHA256 &&
                                                                   weights["bytes"].as_i64 == TEXT_WEIGHTS_BYTES && weights["tensor_count"].as_i == 42 &&
                                                                   weights["stored_dtype"].as_s == "F32" && weights["loaded_dtype"].as_s == "BF16" &&
                                                                   weights["conversion"].as_s == "F32_to_BF16_RNE"
      checkpoint = source["source_checkpoint"]
      expect_exact_keys!(checkpoint,
        %w(repository checkpoint_path checkpoint_lfs_sha256 checkpoint_lfs_sha256_status source_snapshot_revision),
        "source checkpoint")
      unless checkpoint["repository"].as_s == "tencent/Hy-Nomas" &&
             checkpoint["checkpoint_path"].as_s == "DiT/distilled/diffusion_pytorch_model_ema_0_995.safetensors" &&
             checkpoint["checkpoint_lfs_sha256"].as_s == "73819bcf3b82b090116dd1a1748795de95eef703c304e7443963d00ea8aa4d70" &&
             checkpoint["checkpoint_lfs_sha256_status"].as_s == "declared by pinned HF API; whole file not downloaded or verified" &&
             checkpoint["source_snapshot_revision"].as_s == "bac719a80c8d02f91c485fca804bb8221e149b70"
        raise ArgumentError.new("source checkpoint provenance mismatch")
      end
      modules = source["module_bindings"]
      expect_exact_keys!(modules, %w(refiner time_embedder), "source module bindings")
      unless modules["refiner"].as_s == "hymm.models.text_encoder.token_refiner.SingleTokenRefinerPerBlock" &&
             modules["time_embedder"].as_s == "hymm.models.dit.modules.embed_layers.TimestepEmbedder"
        raise ArgumentError.new("source module binding mismatch")
      end
    end

    private def validate_geometry! : Nil
      validate_geometry_record!(@document["geometry"])
    end

    private def validate_action_spans! : Nil
      spans = @document["action_spans"]
      expect_exact_keys!(spans, %w(global action1 action2 padding), "reference action spans")
      {"global" => {0, 49}, "action1" => {49, 132},
       "action2" => {132, 224}, "padding" => {224, 1280}}.each do |name, pair|
        actual = spans[name].as_a.map(&.as_i)
        raise ArgumentError.new("refiner transfer span mismatch: #{name}") unless actual == [pair[0], pair[1]]
      end
    end

    private def validate_mask! : Nil
      mask = @document["mask"]
      expect_exact_keys!(mask,
        %w(dtype shape valid_true_prefix pad_false_suffix sha256 encoding), "reference mask")
      unless mask["dtype"].as_s == "BOOL" && mask["shape"].as_a.map(&.as_i) == [1, TOKEN_COUNT] &&
             mask["valid_true_prefix"].as_i == VALID_ROWS &&
             mask["pad_false_suffix"].as_i == TOKEN_COUNT - VALID_ROWS &&
             mask["sha256"].as_s == mask_sha256 &&
             mask["encoding"].as_s == "one uint8 byte per bool; true=1,false=0"
        raise ArgumentError.new("refiner transfer attention mask mismatch")
      end
    end

    private def validate_normalization! : Nil
      normalization = @document["normalization"]
      expect_exact_keys!(normalization,
        %w(prefix_validation padding valid_rows pad_start pad_stop), "reference normalization")
      unless normalization["prefix_validation"].as_s == "raw_A_B_prefix_bitmatch_before_normalization" &&
             normalization["padding"].as_s == "zero_bf16_rows_after_validation" &&
             normalization["valid_rows"].as_i == VALID_ROWS &&
             normalization["pad_start"].as_i == VALID_ROWS &&
             normalization["pad_stop"].as_i == TOKEN_COUNT
        raise ArgumentError.new("refiner transfer normalization contract mismatch")
      end
    end

    private def validate_execution! : Nil
      execution = @document["execution"]
      expect_exact_keys!(execution,
        %w(timestep case_order scope source_precision source_attention_query_chunks source_a_a_floor),
        "reference execution")
      source_precision = execution["source_precision"]
      expect_exact_keys!(source_precision,
        %w(resident_weights autocast_device autocast_dtype attention_adapter attention_adapter_mode query_chunk_tokens masked_mean_checkpoint c_checkpoint vec_checkpoint block_output_rows),
        "source precision")
      unless source_precision["resident_weights"].as_s == "BF16 converted from authenticated F32 subset" &&
             source_precision["autocast_device"].as_s == "mps" &&
             source_precision["autocast_dtype"].as_s == "BF16" &&
             source_precision["attention_adapter"].as_s == "qualified hy_nomas_real_attention_reference._install_refiner_attention" &&
             source_precision["attention_adapter_mode"].as_s == "bf16" &&
             source_precision["query_chunk_tokens"].as_i == SOURCE_QUERY_CHUNK_TOKENS &&
             source_precision["masked_mean_checkpoint"].as_s == "F32 source masked mean cast to BF16 at c_embedder Linear input" &&
             source_precision["c_checkpoint"].as_s == "forward_pre_hook of individual_token_refiner input[1]" &&
             source_precision["vec_checkpoint"].as_s == "time_in output" &&
             source_precision["block_output_rows"].as_s == "valid prefix [0,224) only"
        raise ArgumentError.new("refiner transfer source precision provenance mismatch")
      end
      unless execution["timestep"].as_f == 1000.0 &&
             execution["case_order"].as_a.map(&.as_s) == CASES &&
             execution["scope"].as_s == "refiner only; no full DiT" &&
             execution["source_attention_query_chunks"].as_i == 2 &&
             source_precision["query_chunk_tokens"].as_i == SOURCE_QUERY_CHUNK_TOKENS
        raise ArgumentError.new("refiner transfer execution scope mismatch")
      end
      floor = execution["source_a_a_floor"]
      raise ArgumentError.new("source A/A floor checkpoint inventory mismatch") unless floor.as_h.keys.sort == OUTPUT_CHECKPOINTS.sort
      floor.as_h.each do |name, value|
        unless OUTPUT_CHECKPOINTS.includes?(name) && value["passed"].as_bool &&
               value["all_finite"].as_bool && value["outside_atol_rtol"].as_i == 0 &&
               value["atol"].as_f == ATOL && value["rtol"].as_f == RTOL &&
               value["checkpoint"].as_s == "A0/A1.#{name}" &&
               value["elements"].as_i == tensor_elements(name)
          raise ArgumentError.new("source A/A floor is not within the qualified tolerance: #{name}")
        end
      end
    end

    private def validate_precision! : Nil
      precision = @document["precision"]
      expect_exact_keys!(precision,
        %w(resident_weights compute attention source_bf16_autocast controls atol rtol),
        "reference precision")
      unless precision["resident_weights"].as_s == "BF16" &&
             precision["compute"].as_s == "F32" &&
             precision["attention"].as_s == "F32Control" &&
             precision["source_bf16_autocast"].as_bool
        raise ArgumentError.new("refiner transfer precision policy mismatch")
      end
      controls = precision["controls"]
      raise ArgumentError.new("source-BF16 control inventory mismatch") unless controls.as_h.keys.sort == SOURCE_BF16_CONTROLS.sort
      SOURCE_BF16_CONTROLS.each do |name|
        raise ArgumentError.new("source-BF16 refiner control disabled: #{name}") unless controls[name].as_bool
      end
      raise ArgumentError.new("refiner transfer comparison tolerance mismatch") unless precision["atol"].as_f == ATOL && precision["rtol"].as_f == RTOL
    end

    private def validate_aa_input_binding! : Nil
      inputs = @document["inputs"]
      expect_exact_keys!(inputs, %w(cases prefix_checks), "reference inputs")
      cases = inputs["cases"]
      raise ArgumentError.new("reference input case inventory mismatch") unless cases.as_h.keys.sort == CASES.sort
      a0 = @document["inputs"]["cases"]["A0"]
      a1 = @document["inputs"]["cases"]["A1"]
      fields = %w(manifest_path manifest_sha256 payload_path payload_bytes payload_sha256 tensor_name offset_bytes nbytes source_tensor_sha256 source_shape source_dtype normalized_tensor_sha256)
      expect_exact_keys!(a0, fields, "A0 input record")
      expect_exact_keys!(a1, fields, "A1 input record")
      fields.each do |name|
        raise ArgumentError.new("A/A input binding differs: #{name}") unless a0[name] == a1[name]
      end
      raise ArgumentError.new("A/A input name must be input.qwen_bf16") unless a0["tensor_name"].as_s == "input.qwen_bf16"
    end

    private def validate_prefix_checks! : Nil
      prefix = @document["inputs"]["prefix_checks"]
      expect_exact_keys!(prefix,
        %w(validated_before_padding_normalization unchanged_rows changed_rows action2_changed_bytes global_action1_sha256 A0_A1_same_normalized_qwen_sha256),
        "reference prefix checks")
      unless prefix["validated_before_padding_normalization"].as_bool &&
             prefix["unchanged_rows"].as_a.map(&.as_i) == [0, PREFIX_END] &&
             prefix["changed_rows"].as_a.map(&.as_i) == [ACTION2_START, ACTION2_END] &&
             prefix["action2_changed_bytes"].as_i == @input_action2_changed_bytes &&
             prefix["A0_A1_same_normalized_qwen_sha256"].as_s == @inputs["A0"].normalized_tensor_sha256
        raise ArgumentError.new("source prefix/intervention admission receipt mismatch")
      end
      raw_prefix = @inputs["A0"].source_bytes[0, PREFIX_END * INPUT_ROW_BYTES]
      expected = prefix["global_action1_sha256"].as_s
      require_sha256!(expected, "source unchanged-prefix SHA-256")
      raise ArgumentError.new("source unchanged-prefix digest mismatch") unless Digest::SHA256.hexdigest(raw_prefix) == expected
    end

    private def read_input_case(case_name : String) : InputCase
      case_doc = @document["inputs"]["cases"][case_name]
      outer_keys = %w(manifest_path manifest_sha256 payload_path payload_bytes payload_sha256 tensor_name offset_bytes nbytes source_tensor_sha256 source_shape source_dtype normalized_tensor_sha256)
      expect_exact_keys!(case_doc, outer_keys, "#{case_name} reference input")
      manifest_name = safe_basename(case_doc["manifest_path"].as_s, "case input manifest")
      payload_name = safe_basename(case_doc["payload_path"].as_s, "case normalized payload")
      root = File.dirname(@path)
      manifest_path = canonical_regular_file(File.join(root, manifest_name), "case input manifest")
      payload_path = canonical_regular_file(File.join(root, payload_name), "case normalized payload")

      manifest_raw = read_bounded(manifest_path, MAX_MANIFEST_BYTES, "case input manifest")
      manifest_sha = Digest::SHA256.hexdigest(manifest_raw)
      require_sha256!(case_doc["manifest_sha256"].as_s, "case input manifest SHA-256")
      raise ArgumentError.new("case input manifest SHA mismatch: #{case_name}") unless manifest_sha == case_doc["manifest_sha256"].as_s
      source_doc = parse_json(manifest_raw, "case input manifest")
      required_input_keys = %w(schema schema_version upstream_revision source_qwen_manifest source_qwen_payload source_qwen_range source_qwen origin_manifest normalization payload tensors)
      actual_input_keys = source_doc.as_h.keys.sort
      optional_input_keys = ["generation_input_binding"]
      unless required_input_keys.sort.all? { |key| actual_input_keys.includes?(key) } &&
             (actual_input_keys - required_input_keys - optional_input_keys).empty? &&
             (actual_input_keys - optional_input_keys).sort == required_input_keys.sort
        raise ArgumentError.new("case input manifest key inventory mismatch: #{case_name}")
      end
      unless source_doc["schema"].as_s == INPUT_SCHEMA && source_doc["schema_version"].as_i == 1 &&
             source_doc["upstream_revision"].as_s == UPSTREAM_REVISION
        raise ArgumentError.new("case input manifest schema or revision mismatch: #{case_name}")
      end
      expect_exact_keys!(source_doc["source_qwen_manifest"],
        %w(path sha256 schema source_revision), "source Qwen manifest binding")
      expect_exact_keys!(source_doc["source_qwen_payload"],
        %w(basename bytes sha256 dtype shape offset_bytes), "source Qwen payload binding")
      expect_exact_keys!(source_doc["source_qwen"],
        %w(name dtype shape offset nbytes sha256), "raw source Qwen tensor")
      expect_exact_keys!(source_doc["origin_manifest"], %w(path sha256), "source origin binding")
      expect_exact_keys!(source_doc["normalization"],
        %w(valid_prefix_rows pad_start pad_stop pad_operation normalized_sha256), "case normalization")
      expect_exact_keys!(source_doc["payload"], %w(basename bytes sha256), "case normalized payload")
      expect_exact_keys!(source_doc["tensors"], %w(input.qwen_bf16), "case normalized tensor inventory")
      expect_exact_keys!(source_doc["tensors"]["input.qwen_bf16"],
        %w(dtype shape offset nbytes sha256), "case normalized Qwen tensor")

      raw_source = load_source_qwen_tensor(source_doc, case_name)
      validate_case_origin!(source_doc, case_doc, raw_source, case_name, payload_path)
      source_qwen = source_doc["source_qwen"]
      source_hash = source_qwen["sha256"].as_s
      require_sha256!(source_hash, "source Qwen tensor SHA-256")
      raise ArgumentError.new("case source tensor SHA mismatch: #{case_name}") unless source_hash == case_doc["source_tensor_sha256"].as_s
      unless source_qwen["name"].as_s == case_doc["tensor_name"].as_s &&
             source_qwen["dtype"].as_s == case_doc["source_dtype"].as_s &&
             source_qwen["dtype"].as_s == "BF16" &&
             source_qwen["shape"].as_a.map(&.as_i) == case_doc["source_shape"].as_a.map(&.as_i) &&
             source_qwen["offset"].as_i64 == case_doc["offset_bytes"].as_i64 &&
             source_qwen["nbytes"].as_i64 == case_doc["nbytes"].as_i64 &&
             source_qwen["shape"].as_a.map(&.as_i) == [1, TOKEN_COUNT, INPUT_DIM] &&
             source_qwen["nbytes"].as_i64 == INPUT_BYTES
        raise ArgumentError.new("case source tensor geometry mismatch: #{case_name}")
      end
      raise ArgumentError.new("source Qwen tensor range SHA mismatch: #{case_name}") unless Digest::SHA256.hexdigest(raw_source) == source_hash

      normalized_sha = case_doc["normalized_tensor_sha256"].as_s
      require_sha256!(normalized_sha, "normalized Qwen tensor SHA-256")
      normalization = source_doc["normalization"]
      unless normalization["valid_prefix_rows"].as_i == VALID_ROWS &&
             normalization["pad_start"].as_i == VALID_ROWS &&
             normalization["pad_stop"].as_i == TOKEN_COUNT &&
             normalization["pad_operation"].as_s == "zero_after_prefix_validation" &&
             normalization["normalized_sha256"].as_s == normalized_sha
        raise ArgumentError.new("case input normalization metadata mismatch: #{case_name}")
      end
      payload = source_doc["payload"]
      unless payload["basename"].as_s == payload_name &&
             payload["bytes"].as_i64 == case_doc["payload_bytes"].as_i64 &&
             payload["sha256"].as_s == case_doc["payload_sha256"].as_s &&
             File.size(payload_path) == case_doc["payload_bytes"].as_i64
        raise ArgumentError.new("case normalized payload metadata mismatch: #{case_name}")
      end
      tensor = source_doc["tensors"]["input.qwen_bf16"]
      unless tensor["dtype"].as_s == "BF16" && tensor["shape"].as_a.map(&.as_i) == [1, TOKEN_COUNT, INPUT_DIM] &&
             tensor["offset"].as_i64 == 0 && tensor["nbytes"].as_i64 == INPUT_BYTES &&
             tensor["sha256"].as_s == normalized_sha && tensor["sha256"].as_s == payload["sha256"].as_s
        raise ArgumentError.new("normalized Qwen tensor record mismatch: #{case_name}")
      end
      raise ArgumentError.new("normalized input payload file SHA mismatch: #{case_name}") unless sha256_file(payload_path) == payload["sha256"].as_s
      normalized_payload = File.read(payload_path).to_slice.dup
      raise ArgumentError.new("normalized input payload size mismatch: #{case_name}") unless normalized_payload.size.to_i64 == INPUT_BYTES

      InputCase.new(manifest_path, manifest_sha, payload_path,
        payload["sha256"].as_s, payload["bytes"].as_i64,
        source_hash, normalized_sha, raw_source, normalized_payload)
    end

    private def validate_case_origin!(case_doc : JSON::Any, outer : JSON::Any,
                                      source_bytes : Bytes, case_name : String,
                                      normalized_path : String) : Nil
      binding = case_doc["origin_manifest"]
      origin_path = canonical_regular_file(binding["path"].as_s, "source input origin manifest")
      origin_raw = read_bounded(origin_path, MAX_MANIFEST_BYTES, "source input origin manifest")
      origin_sha = Digest::SHA256.hexdigest(origin_raw)
      require_sha256!(binding["sha256"].as_s, "source input origin manifest SHA-256")
      raise ArgumentError.new("source input origin manifest SHA mismatch: #{case_name}") unless origin_sha == binding["sha256"].as_s
      origin = parse_json(origin_raw, "source input origin manifest")
      source_manifest = case_doc["source_qwen_manifest"]
      raw_sha = Digest::SHA256.hexdigest(source_bytes)
      crop = origin["crop"]
      unless origin["schema"].as_s == "hy-nomas-refiner-transfer-input-origin-v1" &&
             File.realpath(origin["path"].as_s) == normalized_path &&
             origin["sha256"].as_s == outer["payload_sha256"].as_s &&
             origin["raw_qwen_tensor_sha256"].as_s == raw_sha &&
             File.realpath(origin["source_qwen_manifest"].as_s) == File.realpath(source_manifest["path"].as_s) &&
             origin["source_qwen_manifest_sha256"].as_s == source_manifest["sha256"].as_s &&
             origin["source_qwen_sidecar_sha256"].as_s == case_doc["source_qwen_payload"]["sha256"].as_s &&
             crop["state"].as_i == 26 && crop["start"].as_i == 108 &&
             crop["tokens"].as_i == TOKEN_COUNT && crop["hidden"].as_i == INPUT_DIM &&
             origin["nbytes"].as_i64 == INPUT_BYTES &&
             origin["normalization"].as_s == "zero masked padding rows [224,1280); preserve all valid source BF16 words"
        raise ArgumentError.new("source input origin provenance mismatch: #{case_name}")
      end
      if generation = case_doc["generation_input_binding"]?
        generation = generation.not_nil!
        unless case_name == "A0" || case_name == "A1"
          raise ArgumentError.new("only accepted A cases may carry generation bundle provenance")
        end
        unless File.realpath(origin["source_generation_manifest"].as_s) == File.realpath(generation["manifest_path"].as_s) &&
               origin["source_generation_manifest_sha256"].as_s == generation["manifest_sha256"].as_s &&
               origin["source_payload_sha256"].as_s == generation["payload_sha256"].as_s &&
               origin["raw_qwen_tensor_sha256"].as_s == generation["tensor_sha256"].as_s
          raise ArgumentError.new("accepted A source origin does not bind its generation input")
        end
      elsif case_name == "A0" || case_name == "A1"
        raise ArgumentError.new("accepted A source origin is missing generation provenance")
      elsif origin["source_generation_manifest"]?
        raise ArgumentError.new("counterfactual B must not bind the accepted generation bundle")
      end
    end

    private def load_source_qwen_tensor(case_doc : JSON::Any, case_name : String) : Bytes
      manifest_binding = case_doc["source_qwen_manifest"]
      manifest_path = canonical_regular_file(manifest_binding["path"].as_s, "source Qwen manifest")
      manifest_sha = manifest_binding["sha256"].as_s
      require_sha256!(manifest_sha, "source Qwen manifest SHA-256")
      manifest_raw = read_bounded(manifest_path, MAX_MANIFEST_BYTES, "source Qwen manifest")
      raise ArgumentError.new("source Qwen manifest SHA mismatch: #{case_name}") unless Digest::SHA256.hexdigest(manifest_raw) == manifest_sha
      manifest = parse_json(manifest_raw, "source Qwen manifest")
      unless manifest_binding["path"].as_s.starts_with?("/") &&
             manifest_binding["schema"].as_s == QWEN_SOURCE_SCHEMA &&
             manifest_binding["source_revision"].as_s == QWEN_SOURCE_REVISION &&
             manifest["schema"].as_s == QWEN_SOURCE_SCHEMA &&
             manifest["checkpoint"]["revision"].as_s == QWEN_SOURCE_REVISION
        raise ArgumentError.new("source Qwen manifest schema/revision mismatch: #{case_name}")
      end

      prompt = manifest["hy_nomas_prompt"]
      unless prompt["upstream_revision"].as_s == UPSTREAM_REVISION &&
             prompt["crop_start"].as_i == 108 && prompt["raw_length"].as_i == 1388 &&
             manifest["capture"]["captured_state_count"].as_i == 27 &&
             manifest["checkpoint"]["repo_id"].as_s == "Qwen/Qwen2.5-VL-7B-Instruct" &&
             prompt["cropped_attention_mask"].as_a.map(&.as_i) == Array(Int32).new(TOKEN_COUNT) { |index| index < VALID_ROWS ? 1 : 0 }
        raise ArgumentError.new("source Qwen prompt/capture provenance mismatch: #{case_name}")
      end
      hidden = manifest["capture"]["hidden_states"]
      storage = hidden["storage"]
      sidecar = manifest["sidecars"][storage["sidecar"].as_s]
      payload_binding = case_doc["source_qwen_payload"]
      payload_name = safe_basename(payload_binding["basename"].as_s, "source Qwen payload")
      require_sha256!(payload_binding["sha256"].as_s, "source Qwen payload SHA-256")
      unless hidden["dtype"].as_s == "bfloat16" &&
             hidden["shape"].as_a.map(&.as_i) == [27, 1, 1388, INPUT_DIM] &&
             payload_binding["dtype"].as_s == "BF16" &&
             payload_binding["shape"].as_a.map(&.as_i) == [27, 1, 1388, INPUT_DIM] &&
             payload_binding["offset_bytes"].as_i64 == storage["offset_bytes"].as_i64 &&
             storage["nbytes"].as_i64 == payload_binding["bytes"].as_i64 &&
             sidecar["path"].as_s == payload_name &&
             sidecar["nbytes"].as_i64 == payload_binding["bytes"].as_i64 &&
             sidecar["sha256"].as_s == payload_binding["sha256"].as_s &&
             sidecar["encoding"].as_s == "bfloat16-le"
        raise ArgumentError.new("source Qwen payload geometry/binding mismatch: #{case_name}")
      end
      payload_path = canonical_regular_file(File.join(File.dirname(manifest_path), payload_name), "source Qwen payload")
      unless File.size(payload_path) == payload_binding["bytes"].as_i64 &&
             sha256_file(payload_path) == payload_binding["sha256"].as_s
        raise ArgumentError.new("source Qwen payload hash/size mismatch: #{case_name}")
      end

      range = case_doc["source_qwen_range"]
      expect_exact_keys!(range,
        %w(tensor_name dtype shape offset_bytes nbytes sha256 state_index crop_start crop_stop selected_shape selected_nbytes selected_sha256),
        "source Qwen range")
      state = range["state_index"].as_i
      crop_start = range["crop_start"].as_i
      crop_stop = range["crop_stop"].as_i
      selected_shape = range["selected_shape"].as_a.map(&.as_i)
      expected_rows = crop_stop - crop_start
      expected_offset = ((state.to_i64 * 1388 + crop_start) * INPUT_DIM * 2)
      selected_sha = range["selected_sha256"].as_s
      require_sha256!(selected_sha, "selected Qwen tensor SHA-256")
      unless range["tensor_name"].as_s == "capture.hidden_states" &&
             range["dtype"].as_s == "BF16" && range["shape"].as_a.map(&.as_i) == [27, 1, 1388, INPUT_DIM] &&
             range["offset_bytes"].as_i64 == 0 &&
             range["nbytes"].as_i64 == payload_binding["bytes"].as_i64 &&
             range["sha256"].as_s == payload_binding["sha256"].as_s &&
             state == 26 && crop_start == 108 && crop_stop == 1388 &&
             selected_shape == [1, TOKEN_COUNT, INPUT_DIM] &&
             range["selected_nbytes"].as_i64 == INPUT_BYTES &&
             expected_rows == TOKEN_COUNT && expected_offset >= 0 &&
             expected_offset <= payload_binding["bytes"].as_i64 - INPUT_BYTES
        raise ArgumentError.new("source Qwen selected range mismatch: #{case_name}")
      end
      sidecar_sha, bytes = sha256_file_extract_range(payload_path, expected_offset, INPUT_BYTES)
      raise ArgumentError.new("source Qwen sidecar changed while admitting range: #{case_name}") unless sidecar_sha == payload_binding["sha256"].as_s
      raise ArgumentError.new("source Qwen selected range SHA mismatch: #{case_name}") unless Digest::SHA256.hexdigest(bytes) == selected_sha
      unless selected_sha == source_qwen_for(case_doc)["sha256"].as_s
        raise ArgumentError.new("selected Qwen range/source tensor binding mismatch: #{case_name}")
      end

      if generation_binding = case_doc["generation_input_binding"]?
        validate_generation_binding!(generation_binding.not_nil!, selected_sha, bytes, case_name)
      elsif case_name == "A0" || case_name == "A1"
        raise ArgumentError.new("accepted A input requires generation provenance binding")
      end
      bytes
    end

    private def source_qwen_for(case_doc : JSON::Any) : JSON::Any
      case_doc["source_qwen"]
    end

    private def validate_generation_binding!(binding : JSON::Any, selected_sha : String,
                                             selected_bytes : Bytes, case_name : String) : Nil
      expected_keys = %w(manifest_path manifest_sha256 payload_basename payload_bytes payload_sha256 tensor_name tensor_dtype tensor_shape tensor_offset tensor_nbytes tensor_sha256)
      raise ArgumentError.new("generation input binding key inventory mismatch") unless binding.as_h.keys.sort == expected_keys.sort
      manifest_path = canonical_regular_file(binding["manifest_path"].as_s, "accepted generation manifest")
      manifest_sha = binding["manifest_sha256"].as_s
      require_sha256!(manifest_sha, "accepted generation manifest SHA-256")
      manifest_raw = read_bounded(manifest_path, MAX_MANIFEST_BYTES, "accepted generation manifest")
      raise ArgumentError.new("accepted generation manifest SHA mismatch") unless Digest::SHA256.hexdigest(manifest_raw) == manifest_sha
      manifest = parse_json(manifest_raw, "accepted generation manifest")
      unless manifest["schema"].as_s == "cogni_ml.hy_nomas_generation_inputs/v2" &&
             manifest["schema_version"].as_i == 2 &&
             manifest["upstream_revision"].as_s == UPSTREAM_REVISION
        raise ArgumentError.new("accepted generation input schema or source revision mismatch")
      end
      payload_name = safe_basename(binding["payload_basename"].as_s, "accepted generation payload")
      payload_path = canonical_regular_file(File.join(File.dirname(manifest_path), payload_name), "accepted generation payload")
      payload_sha = binding["payload_sha256"].as_s
      require_sha256!(payload_sha, "accepted generation payload SHA-256")
      payload_bytes = binding["payload_bytes"].as_i64
      unless manifest["payload"].as_s == payload_name && manifest["payload_bytes"].as_i64 == payload_bytes &&
             manifest["payload_sha256"].as_s == payload_sha && File.size(payload_path) == payload_bytes &&
             sha256_file(payload_path) == payload_sha
        raise ArgumentError.new("accepted generation payload metadata/hash mismatch")
      end
      tensor = manifest["tensors"]["input.qwen_bf16"]
      unless binding["tensor_name"].as_s == "input.qwen_bf16" &&
             binding["tensor_dtype"].as_s == "BF16" && tensor["dtype"].as_s == "BF16" &&
             tensor["shape"].as_a.map(&.as_i) == [1, TOKEN_COUNT, INPUT_DIM] &&
             binding["tensor_shape"].as_a.map(&.as_i) == [1, TOKEN_COUNT, INPUT_DIM] &&
             tensor["offset"].as_i64 == binding["tensor_offset"].as_i64 &&
             tensor["nbytes"].as_i64 == INPUT_BYTES && binding["tensor_nbytes"].as_i64 == INPUT_BYTES
        raise ArgumentError.new("accepted generation Qwen tensor geometry mismatch")
      end
      tensor_sha = tensor["sha256"].as_s
      require_sha256!(tensor_sha, "accepted generation Qwen tensor SHA-256")
      unless tensor_sha == binding["tensor_sha256"].as_s && tensor_sha == selected_sha
        raise ArgumentError.new("accepted generation Qwen tensor SHA binding mismatch")
      end
      range_sha, range_bytes = sha256_file_extract_range(payload_path, tensor["offset"].as_i64, INPUT_BYTES)
      unless range_sha == payload_sha && Digest::SHA256.hexdigest(range_bytes) == tensor_sha && range_bytes == selected_bytes
        raise ArgumentError.new("accepted generation Qwen bytes differ from authenticated source crop: #{case_name}")
      end
    end

    private def validate_original_input_pair!(a : Bytes, b : Bytes) : Tuple(Int32, Int32, Int32)
      result = validate_pair_geometry!(a, b, row_bytes: INPUT_ROW_BYTES,
        rows: TOKEN_COUNT, prefix_end: PREFIX_END, valid_rows: VALID_ROWS)
      {result[:changed_bytes], result[:changed_rows], result[:padding_changed_bytes]}
    end

    private def normalize_padding(source : Bytes) : Bytes
      raise ArgumentError.new("source Qwen input size mismatch") unless source.size.to_i64 == INPUT_BYTES
      normalized = Bytes.new(source.size, 0_u8)
      valid_bytes = VALID_ROWS * INPUT_ROW_BYTES
      normalized[0, valid_bytes].copy_from(source[0, valid_bytes])
      normalized
    end

    private def read_output_payload! : Tuple(String, String)
      metadata = @document["payload"]
      name = safe_basename(metadata["basename"].as_s, "source output payload")
      path = canonical_regular_file(File.join(File.dirname(@path), name), "source output payload")
      bytes = metadata["bytes"].as_i64
      sha = metadata["sha256"].as_s
      require_sha256!(sha, "source output payload SHA-256")
      unless metadata["encoding"].as_s == "f32le_bf16_expansion" &&
             0 < bytes <= MAX_OUTPUT_BYTES && File.size(path) == bytes && sha256_file(path) == sha
        raise ArgumentError.new("source output payload metadata/hash mismatch")
      end
      {path, sha}
    end

    private def read_output_tensors! : Hash(String, OutputTensor)
      expected_names = CASES.flat_map { |case_name| OUTPUT_CHECKPOINTS.map { |checkpoint| "#{case_name}.#{checkpoint}" } }.sort
      metadata = @document["tensors"].as_h
      raise ArgumentError.new("source output tensor inventory mismatch") unless metadata.keys.sort == expected_names
      tensors = {} of String => OutputTensor
      metadata.each do |name, item|
        shape = tensor_shape(name)
        unless item["dtype"].as_s == "F32" && item["source_dtype"].as_s == "BF16" &&
               item["shape"].as_a.map(&.as_i) == shape &&
               item["checkpoint_boundary"].as_s == name.split('.', 2)[1]
          raise ArgumentError.new("source output tensor shape/dtype/boundary mismatch: #{name}")
        end
        nbytes = item["nbytes"].as_i64
        expected_nbytes = shape.reduce(1_i64) { |count, extent| count * extent } * sizeof(Float32)
        sha = item["sha256"].as_s
        require_sha256!(sha, "source output tensor SHA-256")
        raise ArgumentError.new("source output tensor byte count mismatch: #{name}") unless nbytes == expected_nbytes
        tensors[name] = OutputTensor.new(name, shape, item["offset_bytes"].as_i64, nbytes, sha)
      end
      tensors
    end

    private def validate_output_payload! : Hash(String, Array(Float32))
      cursor = 0_i64
      @output_tensors.values.sort_by(&.offset_bytes).each do |tensor|
        unless tensor.offset_bytes == cursor && tensor.nbytes <= File.size(@output_payload_path) - cursor
          raise ArgumentError.new("source output tensors must be contiguous and disjoint")
        end
        cursor += tensor.nbytes
      end
      unless cursor == File.size(@output_payload_path)
        raise ArgumentError.new("source output payload has trailing or missing bytes")
      end
      raw = read_bounded(@output_payload_path, MAX_OUTPUT_BYTES, "source output payload")
      expected = {} of String => Array(Float32)
      @output_tensors.each do |name, tensor|
        first = tensor.offset_bytes.to_i32
        size = tensor.nbytes.to_i32
        piece = raw[first, size]
        raise ArgumentError.new("source output tensor digest mismatch: #{name}") unless Digest::SHA256.hexdigest(piece) == tensor.sha256
        values = Array(Float32).new(size // sizeof(Float32))
        count = size // sizeof(Float32)
        count.times do |index|
          bits = IO::ByteFormat::LittleEndian.decode(UInt32, piece[index * 4, 4])
          value = bits.unsafe_as(Float32)
          unless value.finite? && (bits & 0xffff_u32) == 0
            raise ArgumentError.new("source output is not finite BF16-expanded F32: #{name}")
          end
          values << value
        end
        expected[name] = values
      end
      expected
    end

    private def canonical_regular_file(path : String, label : String) : String
      unless File.file?(path) && !File.symlink?(path)
        raise ArgumentError.new("#{label} must be a regular non-symlink file")
      end
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
      unless info.size > 0 && info.size <= limit
        raise ArgumentError.new("#{label} exceeds its bounded size")
      end
      raw = File.read(path).to_slice.dup
      after = File.info(path)
      unless raw.size.to_i64 == info.size && after.same_file?(info) &&
             after.size == info.size && after.modification_time == info.modification_time
        raise ArgumentError.new("#{label} changed while being read")
      end
      raw
    end

    private def require_sha256!(value : String, label : String) : Nil
      raise ArgumentError.new("#{label} must be lowercase SHA-256") unless value.matches?(/\A[0-9a-f]{64}\z/)
    end

    private def expect_exact_keys!(value : JSON::Any, keys : Array(String), label : String) : Nil
      unless value.as_h.keys.sort == keys.sort
        raise ArgumentError.new("#{label} key inventory mismatch")
      end
    end

    private def parse_json(raw : Bytes, label : String) : JSON::Any
      JSON.parse(String.new(raw))
    rescue ex : JSON::ParseException
      raise ArgumentError.new("#{label} is not valid JSON: #{ex.message}")
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

    private def sha256_file_extract_range(path : String, offset : Int64,
                                          count : Int64) : Tuple(String, Bytes)
      info = File.info(path)
      unless count > 0 && offset >= 0 && offset <= info.size && count <= info.size - offset
        raise ArgumentError.new("source tensor range is outside its authenticated sidecar")
      end
      digest = Digest::SHA256.new
      result = Bytes.new(count.to_i32)
      result_cursor = 0_i64
      position = 0_i64
      slab = Bytes.new(8 * 1024 * 1024)
      File.open(path, "rb") do |io|
        loop do
          read = io.read(slab)
          break if read == 0
          chunk = slab[0, read]
          digest.update(chunk)
          chunk_start = position
          chunk_stop = position + read
          overlap_start = Math.max(offset, chunk_start)
          overlap_stop = Math.min(offset + count, chunk_stop)
          if overlap_start < overlap_stop
            source_start = (overlap_start - chunk_start).to_i32
            dest_start = (overlap_start - offset).to_i32
            copy_count = (overlap_stop - overlap_start).to_i32
            result[dest_start, copy_count].copy_from(chunk[source_start, copy_count])
            result_cursor += copy_count
          end
          position = chunk_stop
        end
      end
      after = File.info(path)
      unless position == info.size && result_cursor == count && after.same_file?(info) &&
             after.size == info.size && after.modification_time == info.modification_time
        raise ArgumentError.new("authenticated sidecar changed or was truncated while reading")
      end
      {digest.hexfinal, result}
    end

    private def tensor_shape(name : String) : Array(Int32)
      checkpoint = name.split('.', 2)[1]
      case checkpoint
      when "masked_mean"                    then [1, INPUT_DIM]
      when "c", "vec"                       then [1, HIDDEN_DIM]
      when "block0.output", "block1.output" then [1, VALID_ROWS, HIDDEN_DIM]
      else                                       raise ArgumentError.new("unknown refiner output checkpoint: #{checkpoint}")
      end
    end

    private def mask_sha256 : String
      bytes = Bytes.new(TOKEN_COUNT, 0_u8)
      VALID_ROWS.times { |index| bytes[index] = 1_u8 }
      Digest::SHA256.hexdigest(bytes)
    end
  end

  # This gate deliberately runs before any Metal arena/weight plan exists. The
  # positive control proves Action2-only edits are admitted; the injected
  # prefix mutation must fail closed.
  def prefix_gate_self_test! : Nil
    original = Bytes[0_u8, 1_u8, 2_u8, 3_u8, 4_u8, 5_u8]
    changed_action2 = Bytes[0_u8, 1_u8, 9_u8, 9_u8, 4_u8, 5_u8]
    stats = validate_pair_geometry!(original, changed_action2, row_bytes: 1, rows: 6,
      prefix_end: 2, valid_rows: 4)
    raise "prefix admission self-test positive control failed" unless stats[:changed_bytes] == 2
    corrupted_prefix = changed_action2.dup
    corrupted_prefix[0] = 7_u8
    rejected = false
    begin
      validate_pair_geometry!(original, corrupted_prefix, row_bytes: 1, rows: 6,
        prefix_end: 2, valid_rows: 4)
    rescue ex : ArgumentError
      rejected = ex.message.try(&.includes?("prefix")) || false
    end
    raise "prefix admission self-test did not reject changed global/Action1 input" unless rejected
  end

  # CPU-only admission self-test. These call the same geometry, pair, and
  # comparator routines used by the live Fixture/runner path.
  def admission_self_test! : Nil
    prefix_gate_self_test!
    valid_geometry = JSON.parse(%({"batch_size":1,"valid_rows":224,"pad_rows":1056,"hidden_size":3584,"refiner_dim":2048,"heads":16,"head_dim":128,"blocks":2,"qwen_shape":[1,1280,3584]}))
    validate_geometry_record!(valid_geometry)
    malformed_geometry = JSON.parse(%({"batch_size":1,"valid_rows":223,"pad_rows":1057,"hidden_size":3584,"refiner_dim":2048,"heads":16,"head_dim":128,"blocks":2,"qwen_shape":[1,1280,3584]}))
    geometry_rejected = false
    begin
      validate_geometry_record!(malformed_geometry)
    rescue ex : ArgumentError
      geometry_rejected = ex.message.try(&.includes?("geometry")) || false
    end
    raise "refiner transfer geometry admission self-test did not reject malformed valid_rows" unless geometry_rejected

    clean = compare_values([1.0_f32, -2.0_f32], [1.0_f32, -2.0_f32])
    raise "refiner transfer comparator self-test positive control failed" unless clean[:failures] == 0
    seeded = compare_values([1.0_f32, -2.0_f32], [1.1_f32, -2.0_f32])
    raise "refiner transfer comparator self-test did not detect a seeded mismatch" unless seeded[:failures] == 1 && seeded[:first_mismatch] == 0
  end

  def validate_geometry_record!(geometry : JSON::Any) : Nil
    unless geometry.as_h.keys.sort == %w(batch_size blocks head_dim heads hidden_size pad_rows qwen_shape refiner_dim valid_rows).sort
      raise ArgumentError.new("refiner transfer geometry key inventory mismatch")
    end
    {
      "batch_size"  => 1,
      "valid_rows"  => VALID_ROWS,
      "pad_rows"    => TOKEN_COUNT - VALID_ROWS,
      "hidden_size" => INPUT_DIM,
      "refiner_dim" => HIDDEN_DIM,
      "heads"       => 16,
      "head_dim"    => 128,
      "blocks"      => 2,
    }.each do |key, expected|
      raise ArgumentError.new("refiner transfer geometry mismatch: #{key}") unless geometry[key].as_i == expected
    end
    raise ArgumentError.new("refiner transfer Qwen shape mismatch") unless geometry["qwen_shape"].as_a.map(&.as_i) == [1, TOKEN_COUNT, INPUT_DIM]
  end

  def validate_pair_geometry!(a : Bytes, b : Bytes, *, row_bytes : Int32,
                              rows : Int32, prefix_end : Int32,
                              valid_rows : Int32) : NamedTuple(changed_bytes: Int32,
    changed_rows: Int32, padding_changed_bytes: Int32)
    expected = row_bytes.to_i64 * rows
    unless row_bytes > 0 && rows > valid_rows && prefix_end > 0 &&
           prefix_end <= valid_rows && a.size.to_i64 == expected && b.size.to_i64 == expected
      raise ArgumentError.new("invalid input pair geometry")
    end
    unless a[0, prefix_end * row_bytes] == b[0, prefix_end * row_bytes]
      raise ArgumentError.new("raw A/B global+Action1 Qwen prefix changed")
    end
    action_start = prefix_end * row_bytes
    action_end = valid_rows * row_bytes
    changed_bytes = 0_i32
    changed_rows = 0_i32
    (prefix_end...valid_rows).each do |row|
      first = row * row_bytes
      changed = false
      row_bytes.times do |offset|
        unless a[first + offset] == b[first + offset]
          changed_bytes += 1
          changed = true
        end
      end
      changed_rows += 1 if changed
    end
    raise ArgumentError.new("Action2 Qwen intervention is byte-identical to A") if changed_bytes == 0
    padding_start = valid_rows * row_bytes
    padding_changed = 0_i32
    (padding_start...a.size).each do |index|
      padding_changed += 1 if a[index] != b[index]
    end
    {changed_bytes: changed_bytes, changed_rows: changed_rows,
     padding_changed_bytes: padding_changed}
  end

  class BF16CopyPlan
    @pipeline : ML::Metal::ComputePipeline

    def initialize
      @pipeline = ML::Metal::PipelineCache.get(BF16_COPY_KERNEL) do
        ML::Metal::ComputePipeline.new(BF16_COPY_KERNEL, BF16_COPY_SOURCE)
      end
    end

    def encode(encoder : ML::Metal::ComputeEncoder, source : ML::MetalBuffer,
               destination : ML::MetalBuffer, elements : Int32) : Nil
      unless elements > 0 && source.valid? && destination.valid? &&
             source.storage_mode.shared? && destination.storage_mode.shared? &&
             source.size >= elements.to_i64 * 2 && destination.size >= elements.to_i64 * 2
        raise ArgumentError.new("invalid refiner-transfer BF16 copy extent/storage")
      end
      ML::Video::MetalOps.distinct(source, destination)
      encoder.set_pipeline(@pipeline)
        .set_buffer(source, 0)
        .set_buffer(destination, 1, ML::Metal::BufferAccess::Write)
        .set_value(elements.to_u32, 2)
        .dispatch_1d(elements)
    end
  end

  def tensor_elements(checkpoint : String) : Int32
    case checkpoint
    when "masked_mean"                    then INPUT_DIM
    when "c", "vec"                       then HIDDEN_DIM
    when "block0.output", "block1.output" then VALID_ROWS * HIDDEN_DIM
    else                                       raise ArgumentError.new("unknown native checkpoint #{checkpoint}")
    end
  end

  def compare_values(actual : Array(Float32), expected : Array(Float32)) : NamedTuple(failures: Int32, max_abs: Float64, raw_bit_differences: Int32, first_mismatch: Int32?, first_actual: Float32?, first_expected: Float32?, finite: Bool)
    raise ArgumentError.new("source/native comparison size mismatch") unless actual.size == expected.size
    failures = 0_i32
    raw_differences = 0_i32
    max_abs = 0.0_f64
    finite = true
    first_mismatch = nil.as(Int32?)
    first_actual = nil.as(Float32?)
    first_expected = nil.as(Float32?)
    actual.each_with_index do |value, index|
      target = expected[index]
      raw_differences += 1 if value.unsafe_as(UInt32) != target.unsafe_as(UInt32)
      unless value.finite? && target.finite?
        finite = false
        failures += 1
        max_abs = Float64::MAX
      else
        delta = (value.to_f64 - target.to_f64).abs
        max_abs = Math.max(max_abs, delta)
        failures += 1 if delta > ATOL + RTOL * target.abs.to_f64
      end
      if failures > 0 && first_mismatch.nil?
        first_mismatch = index
        first_actual = value if value.finite?
        first_expected = target if target.finite?
      end
    end
    {failures: failures, max_abs: max_abs, raw_bit_differences: raw_differences,
     first_mismatch: first_mismatch, first_actual: first_actual,
     first_expected: first_expected, finite: finite}
  end

  def delta_stats(baseline : Array(Float32), changed : Array(Float32)) : DeltaStats
    raise ArgumentError.new("delta summary size mismatch") unless baseline.size == changed.size
    sum_delta = 0.0_f64
    sum_base = 0.0_f64
    max_abs = 0.0_f64
    changed_values = 0_i32
    finite = true
    baseline.each_with_index do |base, index|
      target = changed[index]
      unless base.finite? && target.finite?
        finite = false
        next
      end
      delta = (target.to_f64 - base.to_f64).abs
      sum_delta += delta * delta
      sum_base += base.to_f64 * base.to_f64
      max_abs = Math.max(max_abs, delta)
      changed_values += 1 if base.unsafe_as(UInt32) != target.unsafe_as(UInt32)
    end
    return DeltaStats.new(baseline.size.to_i32, changed_values, nil, nil, nil, nil,
      changed_values == 0, false) unless finite
    rms_delta = Math.sqrt(sum_delta / baseline.size)
    baseline_rms = Math.sqrt(sum_base / baseline.size)
    relative = baseline_rms > 0.0 ? rms_delta / baseline_rms : nil
    DeltaStats.new(baseline.size.to_i32, changed_values, max_abs, rms_delta,
      baseline_rms, relative, changed_values == 0, true)
  end

  def span(values : Array(Float32), first_row : Int32, last_row : Int32) : Array(Float32)
    unless 0 <= first_row < last_row <= VALID_ROWS && values.size == VALID_ROWS * HIDDEN_DIM
      raise ArgumentError.new("invalid valid-prefix span summary extent")
    end
    values[first_row * HIDDEN_DIM, (last_row - first_row) * HIDDEN_DIM]
  end

  def write_stats(json : JSON::Builder, stats : DeltaStats) : Nil
    json.object do
      json.field "elements", stats.elements
      json.field "changed_values", stats.changed_values
      json.field "max_abs", stats.max_abs
      json.field "rms_delta", stats.rms_delta
      json.field "baseline_rms", stats.baseline_rms
      json.field "relative_rms_delta", stats.relative_rms_delta
      json.field "exact_zero", stats.exact_zero
      json.field "finite", stats.finite
    end
  end
end
