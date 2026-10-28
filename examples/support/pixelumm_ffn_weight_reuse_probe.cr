require "json"
require "digest/sha256"
require "set"
require "process"
require "../../src/ml/video/pixelumm_decoder_block_mps"
require "./pixelumm_block_probe"
require "./pixelumm_two_block_probe"
require "./pixelumm_connected_resource_probe"
require "./pixelumm_memory_probe"
require "./hy_nomas_probe_instrumentation"

module PixelUMMFFNWeightReuseProbe
  extend self

  SCHEMA                      = "cogni_ml.pixelumm_ffn_windows_reference/v1"
  UPSTREAM_REVISION           = "911cea6458a89c2663513ff7099561ec190dd491"
  MODEL_SHA256                = "bea01a8490f8e9a4076da83d4f9fbb3cba7bd6701df5c28fb2e2c04f00131eec"
  INPUT_MANIFEST_PATH         = "/opt/mldata_ssd/cogni-ml/hy-nomas-first-video-20261028/inputs.json"
  INPUT_MANIFEST_SHA256       = "5b162b8bf4b28c6cfacd8a3d8c14a7758b699034b2a46e84c224522eb0a75a7b"
  INPUT_PAYLOAD_PATH          = "/opt/mldata_ssd/cogni-ml/hy-nomas-first-video-20261028/inputs.tensors.bin"
  INPUT_PAYLOAD_SHA256        = "627cd65f67ed7721c9b0292430bb7ac220a09ea1e5c89a369018dfbbcd920d42"
  INPUT_PAYLOAD_BYTES         = 232_386_560_i64
  WEIGHT_CACHE_PATH           = "/opt/mldata_ssd/cogni-ml/hy-nomas-pixelumm-36-20261027/source/pixelumm-thirty-six-bf16-cache.json"
  WEIGHT_CACHE_SHA256         = "74ee5e84f2b63af0189ffd7be0e4c758b8dfa9ff925e3c9cb389d207ce546dba"
  WEIGHT_CACHE_PAYLOAD_PATH   = "/opt/mldata_ssd/cogni-ml/hy-nomas-pixelumm-36-20261027/source/pixelumm-thirty-six-bf16-cache.bin"
  WEIGHT_CACHE_PAYLOAD_SHA256 = "152ba021b52d9546e4c8821b17ab11f67f46486d521b500de3766f14dd2217ca"
  WEIGHT_CACHE_PAYLOAD_BYTES  = 13_892_143_104_i64
  STEM_SUBSET_PATH            = "/opt/mldata_ssd/cogni-ml/hy-nomas-pixelumm-20261026/source/stem-f32.safetensors"
  STEM_SUBSET_SHA256          = "7eb73ebf42f6681e21870dc2d2d4b592846ba15ed5bf5f9a6c7a83d39c2d8d04"
  STEM_SUBSET_BYTES           = 52_511_088_i64
  STEM_RECEIPT_PATH           = "/opt/mldata_ssd/cogni-ml/hy-nomas-pixelumm-20261026/source/stem-f32.receipt.json"
  STEM_RECEIPT_SHA256         = "88d8dbda211d5dbe0c817d1036069e1efd57b6567f0eb01b9ab5b691c3409014"
  STEM_WEIGHTS_SHA256         = "7257a5d07644a9fddd38918a127eec94abaeb954b0527f2c33777a61c0f0ec1b"
  BLOCK_SUBSET_SHA256         = {
    "0" => "ae82bb41c121a955493f5b1502dc512ac8aaba05a6cec67e98edeaedfe720485",
    "1" => "d33d7cfec3d582bdca30029cdab23bba8fec353ed8937fccf8e66648af080361",
  }
  BLOCK_RECEIPT_SHA256 = {
    "0" => "72a55054289f432f5be145f61b09eae1560dcb084c5a4bdd5a9f7df76631902d",
    "1" => "11379cf75c6ac2b76ffe0adcc7132009512b1c628713d8a5e0793de9cd02ce0e",
  }
  BLOCK_SOURCE_EXTENT = {
    "0" => [102_887_424_i64, 874_673_152_i64],
    "1" => [874_673_152_i64, 1_646_458_880_i64],
  }

  GIB                   = 1024_i64 * 1024 * 1024
  MAX_NAMED_BYTES       = 9_i64 * GIB
  TOKENS                =            5760
  HIDDEN                =            4096
  FFN                   =           12288
  STAGE_BYTES           = 201_326_592_i64
  BLOCK_BF16_BYTES      = 385_892_864_i64
  STAGE_ELEMENT_COUNT   = 150_994_944_u32
  METRIC_WORDS          =               4
  STAGE_METRIC_SLOTS    =               4
  CAPTURE_METRIC_BASE   = STAGE_METRIC_SLOTS
  CAPTURE_METRIC_SLOTS  = 4 * 6
  POSITIVE_CONTROL_SLOT = CAPTURE_METRIC_BASE + CAPTURE_METRIC_SLOTS
  METRIC_SLOTS          = POSITIVE_CONTROL_SLOT + 1
  METRIC_BYTES          = METRIC_SLOTS.to_i64 * METRIC_WORDS * sizeof(UInt32)
  ATOL                  = 2e-5_f64
  RTOL                  = 2e-5_f64
  CAPTURE_ORDER         = ["post_attn_norm", "gate_proj", "up_proj", "product", "down_proj", "output"]
  WINDOW_NAMES          = ["w0", "w1"]
  INPUT_SOURCE_RECORDS  = {
    "latent" => {"record" => "input.latent", "dtype" => "F32", "shape" => [1, 32, 30, 9, 16],
                 "offset" => 0_i64, "nbytes" => 552_960_i64, "sha256" => "61a5ed5ec852b8abf9c3d799df4fca420eae5e320adb7596d2da0a4c1670cf33"},
    "noise" => {"record" => "decoder.noise", "dtype" => "F32", "shape" => [1, 3, 120, 288, 512],
                "offset" => 14_151_680_i64, "nbytes" => 212_336_640_i64, "sha256" => "064167f7d75c79a72f2896c61c988cebd6f71f7586f1f2177276b7ada69009b3"},
    "cos" => {"record" => "decoder.rope_cos", "dtype" => "F32", "shape" => [1, 5760, 1, 128],
              "offset" => 226_488_320_i64, "nbytes" => 2_949_120_i64, "sha256" => "c704e1397520e7a8536994143e56672bea6cab31170b0ffe7a364cb8abb4ac0e"},
    "sin" => {"record" => "decoder.rope_sin", "dtype" => "F32", "shape" => [1, 5760, 1, 128],
              "offset" => 229_437_440_i64, "nbytes" => 2_949_120_i64, "sha256" => "0f3d33d7ce2581e918bc2d91cc73639d9bdf42eb2dfd28a20785b6face97c639"},
  }

  alias Record = PixelUMMBlockProbe::Record
  alias Metric = PixelUMMConnectedResourceProbe::Metric
  alias Budget = NamedTuple(workspace_bytes: Int64, input_weight_bytes: Int64,
    stage_bytes: Int64, expected_capture_bytes: Int64, metric_bytes: Int64,
    total_bytes: Int64)
  alias ScheduleEntry = NamedTuple(index: Int32, mode: String, kind: String,
    capture_mode: Bool, baseline_schedule: String, candidate_schedule: String,
    timing_eligible: Bool)

  def metric_diagnostic(metric : Metric) : String
    "failures=#{metric[:failures]} different_words=#{metric[:different_words]} " +
      "nonfinite=#{metric[:nonfinite]} elements=#{metric[:elements]}"
  end

  def validate_request_count!(count : Int32) : Nil
    unless 1 <= count <= 140
      raise ArgumentError.new("bounded PixelUMM diagnostic requires 1..140 requests")
    end
  end

  def validate_stage_geometry!(dtype : String, shape : Array(Int32), nbytes : Int64) : Nil
    raise ArgumentError.new("FFN weight reuse captures must be BF16") unless dtype == "BF16"
    expected = case shape
               when [1, TOKENS, HIDDEN] then TOKENS.to_i64 * HIDDEN * 2
               when [1, TOKENS, FFN]    then TOKENS.to_i64 * FFN * 2
               else                          -1_i64
               end
    raise ArgumentError.new("FFN weight reuse capture shape/extent mismatch") unless nbytes == expected
  end

  def validate_input_geometry!(name : String, record : Record) : Nil
    expected_shape, expected_dtype = case name
                                     when "x"          then {[1, TOKENS, HIDDEN], "BF16"}
                                     when "cos", "sin" then {[1, TOKENS, 1, 128], "F32"}
                                     else                   raise ArgumentError.new("unknown FFN window input")
                                     end
    unless record.shape == expected_shape && record.dtype == expected_dtype
      raise ArgumentError.new("FFN window input geometry/dtype mismatch: #{name}")
    end
  end

  def validate_window_slice!(window : String, latent_start : Int32,
                             latent_length : Int32, noise_start : Int32,
                             noise_length : Int32) : Nil
    expected = case window
               when "w0" then {0, 10, 0, 40}
               when "w1" then {5, 10, 20, 40}
               else           raise ArgumentError.new("unknown saved-noise preview window")
               end
    unless {latent_start, latent_length, noise_start, noise_length} == expected
      raise ArgumentError.new("saved-noise preview window slice identity mismatch")
    end
  end

  def validate_reference_metadata!(document : JSON::Any) : Nil
    unless document["schema"].as_s == SCHEMA &&
           document["upstream_revision"].as_s == UPSTREAM_REVISION &&
           document["upstream_model_sha256"].as_s == MODEL_SHA256 &&
           document["model_sha256"].as_s == MODEL_SHA256 &&
           document["grid"].as_a.map(&.as_i) == [10, 18, 32] &&
           document["tile_tokens"].as_i == 8 &&
           document["token_order"].as_s == "tile_major" &&
           document["runtime_dtype"].as_s == "BF16" &&
           document["torch_version"].as_s == "2.6.0" &&
           document["capture_order"].as_a.map(&.as_s) == CAPTURE_ORDER &&
           document["comparison"]["atol"].as_f == ATOL &&
           document["comparison"]["rtol"].as_f == RTOL
      raise ArgumentError.new("FFN window reference provenance/layout/tolerance mismatch")
    end
    provenance = document["input_provenance"]
    stem = provenance["stem"]
    unless provenance["kind"].as_s == "saved_initial_noise_conditioning_upstream_stem" &&
           !provenance["model_inference"].as_bool &&
           stem["subset_path"].as_s == STEM_SUBSET_PATH &&
           stem["subset_sha256"].as_s == STEM_SUBSET_SHA256 &&
           stem["subset_bytes"].as_i64 == STEM_SUBSET_BYTES &&
           stem["receipt_path"].as_s == STEM_RECEIPT_PATH &&
           stem["receipt_sha256"].as_s == STEM_RECEIPT_SHA256 &&
           stem["weights_sha256"].as_s == STEM_WEIGHTS_SHA256 &&
           provenance["x_expression"].as_s == "embed(patchify(decoder.noise.to(dtype))) + _condition_tokens(input.latent.to(dtype))" &&
           provenance["window_offsets"].as_s == "latent starts 0/5; noise starts 0/20; each source window independently follows upstream streaming crop semantics"
      raise ArgumentError.new("FFN window source provenance is not the pinned saved-noise upstream stem")
    end
    source = document["input_source"]
    unless source["manifest_path"].as_s == INPUT_MANIFEST_PATH &&
           source["manifest_sha256"].as_s == INPUT_MANIFEST_SHA256 &&
           source["payload_path"].as_s == INPUT_PAYLOAD_PATH &&
           source["payload_sha256"].as_s == INPUT_PAYLOAD_SHA256 &&
           source["payload_bytes"].as_i64 == INPUT_PAYLOAD_BYTES &&
           source["tensors"].as_h.keys.sort == INPUT_SOURCE_RECORDS.keys.sort
      raise ArgumentError.new("FFN window input-source manifest binding mismatch")
    end
    unless document["block_execution"] == JSON.parse({
             block0_input: "inputs.x",
             block1_input: "captures.block0.output",
             edge:         "same_window_output_to_next_input",
           }.to_json)
      raise ArgumentError.new("FFN window block-to-block input edge declaration mismatch")
    end
    INPUT_SOURCE_RECORDS.each do |name, expected|
      item = source["tensors"][name]
      unless item["dtype"].as_s == expected["dtype"] &&
             item["shape"].as_a.map(&.as_i) == expected["shape"] &&
             item["offset"].as_i64 == expected["offset"] &&
             item["nbytes"].as_i64 == expected["nbytes"] &&
             item["sha256"].as_s == expected["sha256"]
        raise ArgumentError.new("FFN window input-source record mismatch: #{name}")
      end
    end
    slices = document["windows"]
    WINDOW_NAMES.each do |window|
      slice = slices[window]["source_slices"]
      validate_window_slice!(window, slice["latent_frame_start"].as_i,
        slice["latent_frame_count"].as_i, slice["noise_frame_start"].as_i,
        slice["noise_frame_count"].as_i)
      unless PixelUMMBlockProbe.valid_digest?(slice["latent_slice_sha256"].as_s) &&
             PixelUMMBlockProbe.valid_digest?(slice["noise_slice_sha256"].as_s)
        raise ArgumentError.new("FFN source-window slices are missing their content digests")
      end
    end
    validate_weight_cache_binding_metadata!(document["weight_cache"])
  end

  def validate_weight_cache_binding_metadata!(binding : JSON::Any) : Nil
    unless binding["manifest_path"].as_s == WEIGHT_CACHE_PATH &&
           binding["manifest_sha256"].as_s == WEIGHT_CACHE_SHA256 &&
           binding["payload_path"].as_s == WEIGHT_CACHE_PAYLOAD_PATH &&
           binding["payload_sha256"].as_s == WEIGHT_CACHE_PAYLOAD_SHA256 &&
           binding["payload_bytes"].as_i64 == WEIGHT_CACHE_PAYLOAD_BYTES &&
           binding["schema"].as_s == "cogni_ml.pixelumm_bf16_weight_cache/v1" &&
           binding["upstream_revision"].as_s == UPSTREAM_REVISION &&
           binding["upstream_model_sha256"].as_s == MODEL_SHA256 &&
           binding["blocks"].as_h.keys.sort == ["0", "1"]
      raise ArgumentError.new("FFN window fixture weight-cache binding mismatch")
    end
    {"0", "1"}.each do |key|
      block = binding["blocks"][key]
      unless block["selected_block"].as_i == key.to_i &&
             block["subset_sha256"].as_s == BLOCK_SUBSET_SHA256[key] &&
             block["source_receipt_sha256"].as_s == BLOCK_RECEIPT_SHA256[key] &&
             block["source_extent"].as_a.map(&.as_i64) == BLOCK_SOURCE_EXTENT[key] &&
             block["subset_bytes"].as_i64 == 771_787_128_i64
        raise ArgumentError.new("FFN window block #{key} cache receipt mismatch")
      end
    end
  end

  def validate_cache_document!(fixture_binding : JSON::Any,
                               cache_document : JSON::Any,
                               cache_sha256 : String,
                               cache_payload_bytes : Int64) : Nil
    validate_weight_cache_binding_metadata!(fixture_binding)
    unless PixelUMMBlockProbe.valid_digest?(cache_sha256) &&
           cache_sha256 == WEIGHT_CACHE_SHA256 &&
           cache_document["schema"].as_s == "cogni_ml.pixelumm_bf16_weight_cache/v1" &&
           cache_document["upstream_revision"].as_s == UPSTREAM_REVISION &&
           cache_document["upstream_model_sha256"].as_s == MODEL_SHA256 &&
           cache_document["source_commit"].as_s == UPSTREAM_REVISION &&
           cache_document["stored_dtype"].as_s == "F32" &&
           cache_document["runtime_dtype"].as_s == "BF16" &&
           cache_document["payload_sha256"].as_s == WEIGHT_CACHE_PAYLOAD_SHA256 &&
           cache_payload_bytes == WEIGHT_CACHE_PAYLOAD_BYTES
      raise ArgumentError.new("external 36-block BF16 cache identity mismatch")
    end
    {"0", "1"}.each do |key|
      unless fixture_binding["blocks"][key] == cache_document["blocks"][key]
        raise ArgumentError.new("fixture/cache block #{key} source receipt differs")
      end
    end
  end

  def metric_slot(plan_slot : Int32, stage_index : Int32) : Int32
    unless (0..3).includes?(plan_slot) && (0...CAPTURE_ORDER.size).includes?(stage_index)
      raise ArgumentError.new("FFN window comparator slot is outside its bounded inventory")
    end
    CAPTURE_METRIC_BASE + plan_slot * CAPTURE_ORDER.size + stage_index
  end

  def schedule(baseline_order : String = "window-major") : Array(ScheduleEntry)
    validate_baseline_schedule!(baseline_order)
    entries = [] of ScheduleEntry
    modes = [
      {"baseline", "correctness"}, {"staged_f32", "correctness"},
      {"baseline", "warmup"}, {"staged_f32", "warmup"},
      {"staged_f32", "warmup"}, {"baseline", "warmup"},
      {"baseline", "timed"}, {"staged_f32", "timed"},
      {"staged_f32", "timed"}, {"baseline", "timed"},
      {"staged_f32", "timed"}, {"baseline", "timed"},
      {"baseline", "timed"}, {"staged_f32", "timed"},
    ]
    modes.each_with_index do |item, index|
      mode, kind = item
      entries << {index: index.to_i32, mode: mode, kind: kind,
                  capture_mode: kind == "correctness", baseline_schedule: baseline_order,
                  candidate_schedule: "block-major", timing_eligible: kind == "timed"}
    end
    validate_schedule!(entries, baseline_order)
    entries
  end

  def validate_baseline_schedule!(value : String) : Nil
    unless {"window-major", "block-major"}.includes?(value)
      raise ArgumentError.new("baseline schedule must be window-major or block-major")
    end
  end

  def validate_schedule!(entries : Array(ScheduleEntry), baseline_order : String) : Nil
    validate_baseline_schedule!(baseline_order)
    unless entries.size == 14 && entries.map(&.[:index]) == (0...14).map(&.to_i32)
      raise ArgumentError.new("FFN weight reuse schedule must contain two correctness, four warmup, and eight timed requests")
    end
    unless entries[0...2].map(&.[:mode]) == ["baseline", "staged_f32"] &&
           entries[0...2].all?(&.[:capture_mode]) &&
           entries[2...6].map(&.[:mode]) == ["baseline", "staged_f32", "staged_f32", "baseline"] &&
           entries[2...6].none?(&.[:capture_mode]) &&
           entries[6..].map(&.[:mode]) == ["baseline", "staged_f32", "staged_f32", "baseline",
                                           "staged_f32", "baseline", "baseline", "staged_f32"] &&
           entries[6..].all?(&.[:timing_eligible])
      raise ArgumentError.new("FFN weight reuse request order or capture/timing boundary mismatch")
    end
    entries.each do |entry|
      unless entry[:baseline_schedule] == baseline_order &&
             entry[:candidate_schedule] == "block-major" &&
             entry[:timing_eligible] == (entry[:kind] == "timed")
        raise ArgumentError.new("FFN weight reuse schedule metadata mismatch")
      end
    end
  end

  def plan_order(mode : String, baseline_order : String) : Array(Int32)
    raise ArgumentError.new("invalid FFN weight reuse mode") unless {"baseline", "staged_f32"}.includes?(mode)
    validate_baseline_schedule!(baseline_order)
    order = mode == "staged_f32" ? "block-major" : baseline_order
    order == "window-major" ? [0, 1, 2, 3] : [0, 2, 1, 3]
  end

  def budget(config : ML::Video::PixelUMMDecoderBlockMPS::Config,
             window_input_bytes : Int64, weight_bytes : Int64,
             expected_capture_bytes : Int64) : Budget
    captures_by_slot = Array.new(4) { CAPTURE_ORDER }
    workspace_bytes = ML::Video::PixelUMMDecoderBlockMPS::Workspace.owned_bytes(
      config, config.layout, captures_by_slot)
    stage_bytes = 3_i64 * STAGE_BYTES
    total = workspace_bytes + window_input_bytes + weight_bytes + stage_bytes +
            expected_capture_bytes + METRIC_BYTES
    {workspace_bytes: workspace_bytes, input_weight_bytes: window_input_bytes + weight_bytes,
     stage_bytes: stage_bytes, expected_capture_bytes: expected_capture_bytes,
     metric_bytes: METRIC_BYTES, total_bytes: total}
  end

  # Reconstruct the complete fixed named-buffer inventory from pinned geometry
  # and cache receipts before hashing/uploading large payloads or initializing
  # Metal. Owner recomputes the same total from parsed records before allocation.
  def preflight_budget(maximum : Int64) : Budget
    config = ML::Video::PixelUMMDecoderBlockMPS::Config.new(
      temporal: 10, height: 18, width: 32)
    input_per_window = TOKENS.to_i64 * HIDDEN * 2_i64 +
                       2_i64 * TOKENS * 128 * 4_i64
    input_bytes = 2_i64 * input_per_window
    weight_bytes = 2_i64 * BLOCK_BF16_BYTES
    one_capture_set = 3_i64 * TOKENS * HIDDEN * 2_i64 +
                      3_i64 * TOKENS * FFN * 2_i64
    expected_capture_bytes = 4_i64 * one_capture_set
    result = budget(config, input_bytes, weight_bytes, expected_capture_bytes)
    validate_budget!(result, maximum)
    result
  end

  def validate_budget!(budget : Budget, maximum : Int64) : Nil
    unless maximum > 0 && maximum <= MAX_NAMED_BYTES &&
           budget[:workspace_bytes] > 0 && budget[:input_weight_bytes] > 0 &&
           budget[:stage_bytes] == 3_i64 * STAGE_BYTES &&
           budget[:expected_capture_bytes] > 0 && budget[:metric_bytes] == METRIC_BYTES &&
           budget[:total_bytes] ==
             budget[:workspace_bytes] + budget[:input_weight_bytes] +
               budget[:stage_bytes] + budget[:expected_capture_bytes] + budget[:metric_bytes] &&
           budget[:total_bytes] <= maximum
      raise ArgumentError.new("FFN weight reuse named-buffer budget exceeds the cap before Metal allocation")
    end
  end

  class Window
    getter name : String
    getter latent_start : Int32
    getter latent_length : Int32
    getter noise_start : Int32
    getter noise_length : Int32
    getter latent_slice_sha256 : String
    getter noise_slice_sha256 : String
    getter inputs : Hash(String, Record)
    getter captures : Hash(String, Hash(String, Record))

    def initialize(@name : String, @latent_start : Int32, @latent_length : Int32,
                   @noise_start : Int32, @noise_length : Int32,
                   @latent_slice_sha256 : String, @noise_slice_sha256 : String,
                   @inputs : Hash(String, Record),
                   @captures : Hash(String, Hash(String, Record)))
    end
  end

  class Fixture
    getter path : String
    getter sha256 : String
    getter payload_sha256 : String
    getter tensor_file : PixelUMMTwoBlockProbe::TensorFile
    getter document : JSON::Any
    getter windows : Hash(String, Window)
    getter all_records : Array(Record)
    getter payload : String

    def initialize(path : String, trusted_sha256 : String)
      @tensor_file = PixelUMMTwoBlockProbe::TensorFile.new(path, trusted_sha256, 4_i64 * GIB)
      @path = @tensor_file.path
      @sha256 = @tensor_file.sha256
      @payload = @tensor_file.payload
      @document = @tensor_file.document
      @payload_sha256 = @document["payload_sha256"].as_s
      PixelUMMFFNWeightReuseProbe.validate_reference_metadata!(@document)
      @windows = {} of String => Window
      @all_records = [] of Record
      WINDOW_NAMES.each do |name|
        node = @document["windows"][name]
        source_slices = node["source_slices"]
        latent_start = source_slices["latent_frame_start"].as_i
        latent_length = source_slices["latent_frame_count"].as_i
        noise_start = source_slices["noise_frame_start"].as_i
        noise_length = source_slices["noise_frame_count"].as_i
        latent_slice_sha256 = source_slices["latent_slice_sha256"].as_s
        noise_slice_sha256 = source_slices["noise_slice_sha256"].as_s
        PixelUMMFFNWeightReuseProbe.validate_window_slice!(name, latent_start,
          latent_length, noise_start, noise_length)
        inputs = parse_records(node["inputs"])
        unless inputs.keys.sort == ["cos", "sin", "x"]
          raise ArgumentError.new("FFN window input inventory mismatch: #{name}")
        end
        inputs.each { |input_name, record| PixelUMMFFNWeightReuseProbe.validate_input_geometry!(input_name, record) }
        captures_node = node["captures"].as_h
        unless captures_node.keys.sort == ["block0", "block1"]
          raise ArgumentError.new("FFN window capture block inventory mismatch: #{name}")
        end
        captures = {} of String => Hash(String, Record)
        {"block0", "block1"}.each do |block|
          stages = parse_records(captures_node[block])
          unless stages.keys.sort == CAPTURE_ORDER.sort
            raise ArgumentError.new("FFN window capture-stage inventory mismatch: #{name}/#{block}")
          end
          stages.each do |stage, record|
            shape = stage == "gate_proj" || stage == "up_proj" || stage == "product" ? [1, TOKENS, FFN] : [1, TOKENS, HIDDEN]
            PixelUMMFFNWeightReuseProbe.validate_stage_geometry!(record.dtype, record.shape, record.nbytes)
            unless record.shape == shape
              raise ArgumentError.new("FFN window capture stage shape mismatch: #{name}/#{block}/#{stage}")
            end
          end
          captures[block] = stages
        end
        @all_records.concat(inputs.values)
        captures.each_value { |stages| @all_records.concat(stages.values) }
        @windows[name] = Window.new(name, latent_start, latent_length,
          noise_start, noise_length, latent_slice_sha256, noise_slice_sha256,
          inputs, captures)
      end
      unless @windows["w0"].inputs["x"].sha256 != @windows["w1"].inputs["x"].sha256
        raise ArgumentError.new("FFN preview windows must have distinct derived stem inputs")
      end
      @tensor_file.validate_ranges!(@all_records, contiguous: true)
    end

    def verify_payload! : Nil
      @tensor_file.verify_payload!
    end

    def upload(record : Record) : ML::MetalBuffer
      buffer = ML::MetalBuffer.new(record.nbytes)
      begin
        @tensor_file.upload_into!(record, buffer)
      rescue error
        buffer.release
        raise error
      end
      buffer
    end

    private def parse_records(value : JSON::Any) : Hash(String, Record)
      value.as_h.transform_values { |raw| Record.new(raw, @tensor_file.payload_bytes) }
    end
  end

  class WeightCache
    getter tensor_file : PixelUMMTwoBlockProbe::TensorFile
    getter weights : Hash(String, Record)
    getter selected_weights : Hash(String, Record)

    def initialize(path : String, fixture : Fixture)
      @tensor_file = PixelUMMTwoBlockProbe::TensorFile.new(path,
        WEIGHT_CACHE_SHA256, PixelUMMTwoBlockProbe.weight_cache_payload_limit(36))
      PixelUMMFFNWeightReuseProbe.validate_cache_document!(
        fixture.document["weight_cache"], @tensor_file.document,
        @tensor_file.sha256, @tensor_file.payload_bytes)
      @weights = @tensor_file.records("weights")
      PixelUMMTwoBlockProbe.validate_weights!(@weights, 36)
      @tensor_file.validate_ranges!(@weights.values, contiguous: true)
      @selected_weights = {} of String => Record
      {"0", "1"}.each do |block|
        PixelUMMBlockProbe.weight_shapes.each_key do |name|
          key = "block#{block}.#{name}"
          @selected_weights[key] = @weights[key]? ||
                                   raise ArgumentError.new("pinned BF16 cache omits #{key}")
        end
      end
      @tensor_file.verify_payload!
    end

    def upload(record : Record) : ML::MetalBuffer
      buffer = ML::MetalBuffer.new(record.nbytes)
      begin
        @tensor_file.upload_into!(record, buffer)
      rescue error
        buffer.release
        raise error
      end
      buffer
    end
  end

  # CPU-only admission/lifecycle guard. The Workspace's fixed predecessor
  # ownership rule is intentionally bypassed; this probe owns the four-plan
  # binding map and serial order directly.
  class RequestLifecycle
    getter attempted : Bool = false
    getter terminal_wait_succeeded : Bool = false
    getter next_index : Int32 = 0
    getter aborted : Bool = false
    @active_entry : ScheduleEntry? = nil
    @next_plan_order = [] of Int32
    @next_plan_cursor = 0
    @staged_blocks = Set(Int32).new
    @stage_checked = false

    def begin_request!(entry : ScheduleEntry, expected_index : Int32,
                       bound_input_ids : Array(String),
                       expected_input_ids : Array(String)) : Nil
      raise ArgumentError.new("FFN request lifecycle is aborted") if @aborted
      raise ArgumentError.new("FFN request lifecycle already has an active request") if @active_entry
      unless entry[:index] == @next_index && entry[:index] == expected_index
        raise ArgumentError.new("FFN requests must be admitted once in schedule order")
      end
      unless bound_input_ids == expected_input_ids && bound_input_ids.size == 2
        raise ArgumentError.new("FFN request is bound to a stale or reordered window input")
      end
      @active_entry = entry
      @next_plan_order = PixelUMMFFNWeightReuseProbe.plan_order(
        entry[:mode], entry[:baseline_schedule])
      @next_plan_cursor = 0
      @staged_blocks.clear
      @stage_checked = false
      @attempted = false
      @terminal_wait_succeeded = false
    end

    def stage_needed?(plan_slot : Int32) : Bool
      entry = active_entry
      raise ArgumentError.new("baseline request cannot stage F32 weights") unless entry[:mode] == "staged_f32"
      raise ArgumentError.new("staged FFN stage admission is out of plan order") unless current_plan == plan_slot
      !@staged_blocks.includes?(plan_slot % 2)
    end

    # Blits and native encodes can partially submit work before reporting an
    # error. Once any command is appended, reuse and teardown stay behind the
    # same single terminal wait.
    def mark_submission_attempt! : Nil
      active_entry
      raise ArgumentError.new("FFN request already reached its terminal wait") if @terminal_wait_succeeded
      @attempted = true
    end

    def mark_stage_enqueued!(plan_slot : Int32) : Nil
      entry = active_entry
      raise ArgumentError.new("baseline request cannot stage F32 weights") unless entry[:mode] == "staged_f32"
      raise ArgumentError.new("staged FFN conversion is out of plan order") unless current_plan == plan_slot
      block = plan_slot % 2
      raise ArgumentError.new("staged FFN block was converted more than once") if @staged_blocks.includes?(block)
      expected_block = @staged_blocks.size.to_i32
      unless block == expected_block && plan_slot == block
        raise ArgumentError.new("staged FFN blocks must be converted once in block-major order")
      end
      @staged_blocks.add(block)
      @stage_checked = false
      @attempted = true
    end

    def mark_stage_checked!(plan_slot : Int32) : Nil
      entry = active_entry
      raise ArgumentError.new("baseline request cannot stage F32 weights") unless entry[:mode] == "staged_f32"
      raise ArgumentError.new("stage identity checks are correctness-only") unless entry[:capture_mode]
      raise ArgumentError.new("staged FFN stage check is out of plan order") unless current_plan == plan_slot
      raise ArgumentError.new("staged FFN identity check lacks a converted block") unless @staged_blocks.includes?(plan_slot % 2)
      raise ArgumentError.new("staged FFN identity was checked twice") if @stage_checked
      @attempted = true
      @stage_checked = true
    end

    def mark_plan_encoded!(plan_slot : Int32) : Nil
      entry = active_entry
      raise ArgumentError.new("FFN native plan is out of schedule order") unless current_plan == plan_slot
      if entry[:mode] == "staged_f32"
        raise ArgumentError.new("staged FFN consumer preceded its block conversion") unless @staged_blocks.includes?(plan_slot % 2)
        if entry[:capture_mode] && !@stage_checked
          raise ArgumentError.new("correctness FFN consumer preceded its stage identity check")
        end
      end
      @attempted = true
      @next_plan_cursor += 1
      @stage_checked = false
    end

    def finish_request!(completed : Bool, wait_count : Int32) : Nil
      entry = active_entry
      expected_blocks = entry[:mode] == "staged_f32" ? Set{0, 1} : Set(Int32).new
      unless @attempted && @next_plan_cursor == 4 && @staged_blocks == expected_blocks &&
             completed && wait_count == 1
        raise ArgumentError.new("FFN request requires four ordered plans and one terminal wait")
      end
      @terminal_wait_succeeded = true
      @next_index += 1
      @active_entry = nil
      @next_plan_order.clear
    end

    def abort! : Nil
      @aborted = true
      @active_entry = nil
      @next_plan_order.clear
    end

    # An exceptional request may still be drained once before teardown. This
    # authorization is separate from a successful request/metrics validation.
    def mark_cleanup_wait!(completed : Bool, wait_count : Int32) : Nil
      unless @aborted && @attempted && completed && wait_count == 1 &&
             !@terminal_wait_succeeded
        raise ArgumentError.new("FFN cleanup release requires its sole successful terminal wait")
      end
      @terminal_wait_succeeded = true
    end

    def release_allowed? : Bool
      !@attempted || @terminal_wait_succeeded
    end

    private def active_entry : ScheduleEntry
      @active_entry || raise ArgumentError.new("FFN request lifecycle has no active request")
    end

    private def current_plan : Int32
      @next_plan_order[@next_plan_cursor]? ||
        raise ArgumentError.new("FFN request encoded more than four native plans")
    end
  end

  {% unless flag?(:cpu_only) %}
    lib PixelUMMFFNWeightReuseBridge
      fun prepare = gs_pixelumm_ffn_weight_reuse_prepare(plan : Void*,
                                                         stages : Void**, error : UInt8*, capacity : LibC::SizeT) : Void*
      fun select = gs_pixelumm_ffn_weight_reuse_select(wrapper : Void*,
                                                       enabled : Int32, captures : Int32, error : UInt8*,
                                                       capacity : LibC::SizeT) : Int32
      fun stage = gs_pixelumm_ffn_weight_reuse_stage(wrapper : Void*,
                                                     command_buffer : Void*, error : UInt8*, capacity : LibC::SizeT) : Int32
      fun check_stage = gs_pixelumm_ffn_weight_reuse_check_stage(wrapper : Void*,
                                                                 command_buffer : Void*, metrics : Void*, slot : UInt32,
                                                                 error : UInt8*, capacity : LibC::SizeT) : Int32
      fun release = gs_pixelumm_ffn_weight_reuse_release(wrapper : Void*) : Void
      fun bindings = gs_pixelumm_ffn_weight_reuse_bindings(wrapper : Void*,
                                                           bindings : Void**, count : LibC::SizeT) : Int32
    end

    class Owner
      getter fixture : Fixture
      getter cache : WeightCache
      getter budget : Budget
      getter named_resources : Hash(String, Int64)
      getter lifecycle : RequestLifecycle
      getter capture_buffers : Array(Hash(String, ML::MetalBuffer))
      getter output_buffers : Array(ML::MetalBuffer)
      getter stage_buffers : Array(ML::MetalBuffer)
      getter metric_bytes : Int64
      getter native_plan_handles : Array(UInt64)

      CORE_ORDER           = ["input_norm", "q", "k", "v", "attention", "residual", "gate", "up", "output"]
      WEIGHT_ORDER         = ML::Video::PixelUMMDecoderBlockMPS::Plan::WEIGHT_NAMES.to_a
      NATIVE_CAPTURE_ORDER = ML::Video::PixelUMMDecoderBlockMPS::Plan::CAPTURE_NAMES.to_a

      @config : ML::Video::PixelUMMDecoderBlockMPS::Config
      @workspace : ML::Video::PixelUMMDecoderBlockMPS::Workspace? = nil
      @input_buffers = {} of String => Hash(String, ML::MetalBuffer)
      @weight_buffers = {} of String => Hash(String, ML::MetalBuffer)
      @expected_buffers = [] of Hash(String, ML::MetalBuffer)
      @expected_records = [] of Hash(String, Record)
      @capture_buffers = [] of Hash(String, ML::MetalBuffer)
      @output_buffers = [] of ML::MetalBuffer
      @stage_buffers = [] of ML::MetalBuffer
      @native_plans = [] of Pointer(Void)
      @wrappers = [] of Pointer(Void)
      @native_plan_handles = [] of UInt64
      @metrics_buffer : ML::MetalBuffer? = nil
      @pipeline : ML::Metal::ComputePipeline? = nil
      @error_text = Bytes.new(1024, 0_u8)
      @released = false
      @last_chain : ML::Metal::MPSCommandChain? = nil
      @memory : PixelUMMMemoryProbe::Recorder
      @maximum : Int64

      def initialize(@fixture : Fixture, @cache : WeightCache, @maximum : Int64,
                     @memory : PixelUMMMemoryProbe::Recorder,
                     @expected_request_count : Int32 = 14,
                     @emit_request_memory : Bool = true)
        PixelUMMFFNWeightReuseProbe.validate_request_count!(@expected_request_count)
        @lifecycle = RequestLifecycle.new
        @config = ML::Video::PixelUMMDecoderBlockMPS::Config.new(
          temporal: 10, height: 18, width: 32)
        expected_capture_bytes = @fixture.windows.values.sum do |window|
          window.captures.values.sum do |stages|
            stages.values.sum(&.nbytes)
          end
        end
        input_bytes = @fixture.windows.values.sum do |window|
          window.inputs.values.sum(&.nbytes)
        end
        selected_weight_bytes = @cache.selected_weights.values.sum(&.nbytes)
        @budget = PixelUMMFFNWeightReuseProbe.budget(@config,
          input_bytes, selected_weight_bytes, expected_capture_bytes)
        PixelUMMFFNWeightReuseProbe.validate_budget!(@budget, @maximum)
        unless @budget == PixelUMMFFNWeightReuseProbe.preflight_budget(@maximum)
          raise ArgumentError.new("parsed FFN fixture/cache inventory differs from fixed pre-Metal admission")
        end
        @metric_bytes = @budget[:metric_bytes]
        @named_resources = {
          "workspace.core_scratch_outputs_and_private_captures" => @budget[:workspace_bytes],
          "two_window_inputs_and_two_block_bf16_weights"        => @budget[:input_weight_bytes],
          "shared_three_matrix_f32_staging"                     => @budget[:stage_bytes],
          "four_private_expected_capture_sets"                  => @budget[:expected_capture_bytes],
          "packed_gpu_metrics"                                  => @budget[:metric_bytes],
        }
        @memory.emit("prebuffers", phase_bytes_kind: "planned_named_metal_bytes",
          phase_bytes: @budget[:total_bytes])
        @fixture.verify_payload!
        verify_memory_headroom!
        unless ML::Metal::Device.init!
          raise ArgumentError.new("Metal device unavailable")
        end
        begin
          allocate_workspace!
          upload_inputs_and_weights!
          upload_expected_captures!
          allocate_stage_and_metrics!
          compile_output_comparator!
          prepare_native_plans!
          validate_actual_resources!
        rescue error
          release_unsubmitted!
          raise error
        end
      end

      def named_resources_json : JSON::Any
        JSON.parse({
          named_bytes:                        @budget[:total_bytes],
          resource_bytes:                     @named_resources,
          native_plan_count:                  @native_plans.size,
          native_plan_handles_distinct:       @native_plan_handles.uniq.size == 4,
          opaque_native_plan_bytes:           nil,
          opaque_native_plan_bytes_semantics: "unknown; excluded from named MetalBuffer inventory",
          plan_input_binding_sha256:          @fixture.windows.values.map { |window| window.inputs["x"].sha256 },
          capture_slots_private:              @capture_buffers.size == 4,
          f32_stage_buffers_shared:           @stage_buffers.size == 3,
        }.to_json)
      end

      def begin_request!(entry : ScheduleEntry, expected_input_ids : Array(String)) : Nil
        check_live!
        raise ArgumentError.new("previous FFN request chain has not been released") if @last_chain
        bound_input_ids = WINDOW_NAMES.map do |window|
          @fixture.windows[window].inputs["x"].sha256
        end
        @lifecycle.begin_request!(entry, entry[:index], bound_input_ids, expected_input_ids)
        select_request_mode!(entry)
        validate_native_bindings!
      end

      def track_chain!(chain : ML::Metal::MPSCommandChain) : Nil
        check_live!
        raise ArgumentError.new("FFN request chain was already attached") if @last_chain
        @last_chain = chain
      end

      def abort_request! : Nil
        @lifecycle.abort!
      end

      def poison_gate_buffers!(chain : ML::Metal::MPSCommandChain) : Nil
        @lifecycle.mark_submission_attempt!
        blit_fill!(chain, poison_buffers)
      end

      def clear_metrics!(chain : ML::Metal::MPSCommandChain) : Nil
        @lifecycle.mark_submission_attempt!
        blit_fill!(chain, [metrics_buffer], 0_u8)
      end

      def encode_request!(chain : ML::Metal::MPSCommandChain,
                          entry : ScheduleEntry,
                          profile_markers : Array(Slice(PixelUMMDecoderBlockMPSFFI::PhaseMarker))? = nil) : NamedTuple(body_encode_seconds: Float64,
        comparator_encode_seconds: Float64,
        stage_conversion_count: Int32,
        stage_identity_check_count: Int32)
        if markers = profile_markers
          unless entry[:mode] == "baseline" && chain.api_profile_enabled? &&
                 markers.size == 4 && markers.all? { |values| values.size == 11 }
            raise ArgumentError.new("FFN phase profiling requires four fixed marker arrays and a baseline API chain")
          end
        end
        order = PixelUMMFFNWeightReuseProbe.plan_order(entry[:mode], entry[:baseline_schedule])
        stage_conversion_count = 0
        stage_identity_check_count = 0
        body_started = Time.instant
        order.each do |plan_slot|
          wrapper = @wrappers[plan_slot]
          if entry[:mode] == "staged_f32"
            if @lifecycle.stage_needed?(plan_slot)
              @lifecycle.mark_stage_enqueued!(plan_slot)
              stage_for_plan!(chain, wrapper, plan_slot)
              stage_conversion_count += 1
            end
            if entry[:capture_mode]
              check_stage!(chain, wrapper, plan_slot)
              @lifecycle.mark_stage_checked!(plan_slot)
              stage_identity_check_count += 1
            end
          end
          encode_plan!(chain, @native_plans[plan_slot], plan_slot, profile_markers.try(&.[plan_slot]))
          @lifecycle.mark_plan_encoded!(plan_slot)
        end
        body_seconds = (Time.instant - body_started).total_seconds
        comparator_started = Time.instant
        if entry[:capture_mode]
          encode_capture_comparisons!(chain)
        else
          encode_output_comparisons!(chain)
        end
        if entry[:kind] == "correctness" && entry[:index] == 0
          encode_positive_control!(chain)
        end
        {body_encode_seconds:        body_seconds,
         comparator_encode_seconds:  (Time.instant - comparator_started).total_seconds,
         stage_conversion_count:     stage_conversion_count,
         stage_identity_check_count: stage_identity_check_count}
      end

      def poison_buffers : Array(ML::MetalBuffer)
        # The Workspace exposes only mutable shared transients here; immutable
        # query/key index tables and the RMS power exponent keep setup values.
        buffers = @workspace.not_nil!.mutable_shared_buffers(include_outputs: true).values
        @capture_buffers.each do |set|
          set.each_value { |buffer| buffers << buffer }
        end
        @stage_buffers.each { |buffer| buffers << buffer }
        buffers
      end

      def finish_request!(chain : ML::Metal::MPSCommandChain,
                          entry : ScheduleEntry) : Array(Metric)
        unless chain.completed? && chain.wait_count == 1
          raise ArgumentError.new("FFN request requires one successful terminal wait")
        end
        @lifecycle.finish_request!(chain.completed?, chain.wait_count)
        metrics = read_metrics!
        validate_request_metrics!(metrics, entry)
        chain.release
        @last_chain = nil
        if @emit_request_memory
          @memory.emit("after_chain_release", request: entry[:index],
            phase_bytes_kind: "owned_named_metal_bytes", phase_bytes: @budget[:total_bytes])
        end
        metrics
      end

      def release_after_terminal! : Nil
        check_live!
        if chain = @last_chain
          raise ArgumentError.new("FFN command chain has not been released") unless chain.completed?
        end
        unless @lifecycle.next_index == @expected_request_count && !@lifecycle.aborted
          raise ArgumentError.new("FFN probe resources require all scheduled requests to finish")
        end
        release_resources!
      end

      def release_unsubmitted! : Nil
        return if @released
        if chain = @last_chain
          raise ArgumentError.new("cannot release FFN probe buffers while a command chain is in flight") unless chain.completed?
        end
        unless @lifecycle.release_allowed?
          raise ArgumentError.new("cannot release FFN probe resources before terminal completion")
        end
        release_resources!
      end

      def expected_input_ids : Array(String)
        WINDOW_NAMES.map { |window| @fixture.windows[window].inputs["x"].sha256 }
      end

      private def verify_memory_headroom! : Nil
        output = IO::Memory.new
        error = IO::Memory.new
        status = Process.run("memory_pressure", ["-Q"], output: output, error: error)
        unless status.success?
          raise ArgumentError.new("10% free-memory guard could not read memory_pressure -Q")
        end
        match = output.to_s.match(/System-wide memory free percentage:\s*(\d+)%/)
        free_percent = match.try(&.[1]).try(&.to_i?) ||
                       raise ArgumentError.new("10% free-memory guard could not parse memory_pressure output")
        unless free_percent > 10
          raise ArgumentError.new("10% free-memory guard requires system free memory above 10%")
        end
      end

      private def allocate_workspace! : Nil
        captures_by_slot = Array.new(4) { CAPTURE_ORDER }
        workspace = ML::Video::PixelUMMDecoderBlockMPS::Workspace.new(@config,
          captures_by_slot: captures_by_slot, max_owned_bytes: @budget[:workspace_bytes])
        unless workspace.slot_count == 4 && workspace.named_bytes == @budget[:workspace_bytes]
          workspace.release
          raise ArgumentError.new("four-slot PixelUMM Workspace differs from preflight")
        end
        @workspace = workspace
        4.times do |slot|
          @capture_buffers << workspace.captures_for_slot(slot)
          @output_buffers << workspace.output_buffer(slot)
        end
        @memory.emit("after_workspace_allocate", phase_bytes_kind: "owned_named_metal_bytes",
          phase_bytes: workspace.named_bytes)
      end

      private def upload_inputs_and_weights! : Nil
        WINDOW_NAMES.each do |window_name|
          window = @fixture.windows[window_name]
          buffers = {} of String => ML::MetalBuffer
          {"x", "cos", "sin"}.each do |name|
            buffers[name] = @fixture.upload(window.inputs[name])
          end
          @input_buffers[window_name] = buffers
        end
        {"0", "1"}.each do |block|
          weights = {} of String => ML::MetalBuffer
          WEIGHT_ORDER.each do |name|
            record = @cache.selected_weights["block#{block}.#{name}"]
            weights[name] = @cache.upload(record)
          end
          @weight_buffers[block] = weights
        end
        actual = @input_buffers.values.sum { |set| set.values.sum(&.size) } +
                 @weight_buffers.values.sum { |set| set.values.sum(&.size) }
        raise ArgumentError.new("FFN borrowed-input/weight inventory differs from preflight") unless actual == @budget[:input_weight_bytes]
      end

      private def upload_expected_captures! : Nil
        WINDOW_NAMES.each_with_index do |window_name, window_index|
          2.times do |block_index|
            plan_slot = window_index * 2 + block_index
            records = @fixture.windows[window_name].captures["block#{block_index}"]
            buffers = {} of String => ML::MetalBuffer
            CAPTURE_ORDER.each do |stage|
              buffers[stage] = @fixture.upload(records[stage])
            end
            @expected_records << records
            @expected_buffers << buffers
          end
        end
        actual = @expected_buffers.sum { |set| set.values.sum(&.size) }
        raise ArgumentError.new("FFN expected-capture inventory differs from preflight") unless actual == @budget[:expected_capture_bytes]
      end

      private def allocate_stage_and_metrics! : Nil
        @stage_buffers = Array.new(3) { ML::MetalBuffer.new(STAGE_BYTES) }
        @metrics_buffer = ML::MetalBuffer.new(@metric_bytes)
        actual = @stage_buffers.sum(&.size) + metrics_buffer.size
        raise ArgumentError.new("FFN stage/metric buffers differ from preflight") unless actual == @budget[:stage_bytes] + @metric_bytes
      end

      private def compile_output_comparator! : Nil
        @pipeline = ML::Metal::ComputePipeline.new("pixelumm_ffn_compare_bf16",
          PixelUMMConnectedResourceProbe::SOURCE)
      end

      private def prepare_native_plans! : Nil
        stage_handles = @stage_buffers.map(&.handle).to_a
        stages_pointer = stage_handles.to_unsafe
        4.times do |plan_slot|
          window_index = plan_slot // 2
          block_index = plan_slot % 2
          window_name = WINDOW_NAMES[window_index]
          block_name = block_index.to_s
          input = if block_index == 0
                    @input_buffers[window_name]["x"]
                  else
                    @output_buffers[plan_slot - 1]
                  end
          weights = WEIGHT_ORDER.map { |name| @weight_buffers[block_name][name].handle }.to_a
          cores_map = @workspace.not_nil!.core_for_slot(plan_slot)
          cores = CORE_ORDER.map { |name| cores_map[name].handle }.to_a
          capture_map = @capture_buffers[plan_slot]
          captures = NATIVE_CAPTURE_ORDER.map do |name|
            capture_map[name]?.try(&.handle) || Pointer(Void).null
          end.to_a
          capture_mask = 0_u64
          NATIVE_CAPTURE_ORDER.each_with_index do |name, index|
            capture_mask |= 1_u64 << index if capture_map.has_key?(name)
          end
          scratch = @workspace.not_nil!.scratch_buffers.map(&.handle).to_a
          layout = @config.layout
          @error_text.fill(0_u8)
          plan = PixelUMMDecoderBlockMPSFFI.plan_create_with_ffn_f32(
            ML::Metal::Device.instance.device_handle,
            @config.temporal, @config.height, @config.width, @config.tile_tokens,
            @config.model_dim, @config.ff_dim, @config.query_heads,
            @config.kv_heads, @config.head_dim, @config.rms_norm_eps,
            input.handle, @input_buffers[window_name]["cos"].handle,
            @input_buffers[window_name]["sin"].handle,
            weights.to_unsafe, cores.to_unsafe, captures.to_unsafe, capture_mask,
            scratch.to_unsafe, layout.max_query_tokens, layout.max_key_tokens,
            layout.tile_count, 1, @error_text.to_unsafe, @error_text.size)
          if plan.null?
            raise ArgumentError.new("raw FFN F32 plan creation failed: #{error_message}")
          end
          @native_plans << plan
          @native_plan_handles << plan.address.to_u64
          @error_text.fill(0_u8)
          wrapper = PixelUMMFFNWeightReuseBridge.prepare(plan, stages_pointer,
            @error_text.to_unsafe, @error_text.size)
          if wrapper.null?
            raise ArgumentError.new("FFN staged graph preparation failed: #{error_message}")
          end
          @wrappers << wrapper
        end
        unless @native_plan_handles.uniq.size == 4
          raise ArgumentError.new("four native FFN plans must have distinct handles")
        end
        validate_native_bindings!
      end

      private def validate_native_bindings! : Nil
        4.times do |plan_slot|
          window_name = WINDOW_NAMES[plan_slot // 2]
          block_name = (plan_slot % 2).to_s
          expected_input = if plan_slot % 2 == 0
                             @input_buffers[window_name]["x"]
                           else
                             @output_buffers[plan_slot - 1]
                           end
          expected = [expected_input.handle,
                      @input_buffers[window_name]["cos"].handle,
                      @input_buffers[window_name]["sin"].handle,
                      @output_buffers[plan_slot].handle,
                      @weight_buffers[block_name]["mlp.gate_proj.weight"].handle,
                      @weight_buffers[block_name]["mlp.up_proj.weight"].handle,
                      @weight_buffers[block_name]["mlp.down_proj.weight"].handle] +
                     @stage_buffers.map(&.handle)
          actual = Array(Pointer(Void)).new(10, Pointer(Void).null)
          status = PixelUMMFFNWeightReuseBridge.bindings(@wrappers[plan_slot],
            actual.to_unsafe, actual.size)
          unless status == 0 && actual == expected
            raise ArgumentError.new("native FFN plan #{plan_slot} binding identity changed")
          end
        end
      end

      private def select_request_mode!(entry : ScheduleEntry) : Nil
        enabled = entry[:mode] == "staged_f32" ? 1 : 0
        captures = entry[:capture_mode] ? 1 : 0
        @wrappers.each do |wrapper|
          @error_text.fill(0_u8)
          status = PixelUMMFFNWeightReuseBridge.select(wrapper, enabled, captures,
            @error_text.to_unsafe, @error_text.size)
          unless status == 0
            raise ArgumentError.new("FFN baseline/staged graph selection failed (#{status}): #{error_message}")
          end
        end
      end

      private def stage_for_plan!(chain : ML::Metal::MPSCommandChain,
                                  wrapper : Pointer(Void), plan_slot : Int32) : Nil
        @error_text.fill(0_u8)
        status = -1
        chain.pause!
        begin
          status = PixelUMMFFNWeightReuseBridge.stage(wrapper, chain.handle,
            @error_text.to_unsafe, @error_text.size)
        ensure
          chain.resume!
        end
        raise ArgumentError.new("FFN GPU staging failed (#{status}): #{error_message}") unless status == 0
      end

      private def check_stage!(chain : ML::Metal::MPSCommandChain,
                               wrapper : Pointer(Void), plan_slot : Int32) : Nil
        @error_text.fill(0_u8)
        status = -1
        chain.pause!
        begin
          status = PixelUMMFFNWeightReuseBridge.check_stage(wrapper, chain.handle,
            metrics_buffer.handle, plan_slot.to_u32, @error_text.to_unsafe,
            @error_text.size)
        ensure
          chain.resume!
        end
        raise ArgumentError.new("FFN GPU stage-word check failed (#{status}): #{error_message}") unless status == 0
      end

      private def encode_plan!(chain : ML::Metal::MPSCommandChain,
                               plan : Pointer(Void), plan_slot : Int32,
                               markers : Slice(PixelUMMDecoderBlockMPSFFI::PhaseMarker)? = nil) : Nil
        @error_text.fill(0_u8)
        status = -1
        chain.pause!
        begin
          status = if values = markers
                     PixelUMMDecoderBlockMPSFFI.encode_profiled(plan, chain.handle,
                       chain.api_profile_handle, values.to_unsafe, values.size.to_u32,
                       @error_text.to_unsafe, @error_text.size)
                   else
                     PixelUMMDecoderBlockMPSFFI.encode(plan, chain.handle,
                       @error_text.to_unsafe, @error_text.size)
                   end
        ensure
          chain.resume!
        end
        raise ArgumentError.new("FFN native plan encode failed for slot #{plan_slot} (#{status}): #{error_message}") unless status == 0
      end

      private def encode_capture_comparisons!(chain : ML::Metal::MPSCommandChain) : Nil
        4.times do |plan_slot|
          CAPTURE_ORDER.each_with_index do |stage, stage_index|
            encode_comparison!(chain, @capture_buffers[plan_slot][stage],
              @expected_buffers[plan_slot][stage],
              PixelUMMFFNWeightReuseProbe.metric_slot(plan_slot, stage_index),
              @expected_records[plan_slot][stage].nbytes // 2)
          end
        end
      end

      private def encode_output_comparisons!(chain : ML::Metal::MPSCommandChain) : Nil
        4.times do |plan_slot|
          output_stage_index = CAPTURE_ORDER.index("output") ||
                               raise ArgumentError.new("output capture slot is missing")
          encode_comparison!(chain, @output_buffers[plan_slot],
            @expected_buffers[plan_slot]["output"],
            PixelUMMFFNWeightReuseProbe.metric_slot(plan_slot, output_stage_index),
            @expected_records[plan_slot]["output"].nbytes // 2)
        end
      end

      private def encode_positive_control!(chain : ML::Metal::MPSCommandChain) : Nil
        record = @expected_records[0]["output"]
        # Inject a NaN into a self-comparison so the exact positive control
        # does not depend on legitimate differences in native BF16 rounding.
        encode_comparison!(chain, @expected_buffers[0]["output"], @expected_buffers[0]["output"],
          POSITIVE_CONTROL_SLOT, record.nbytes // 2, inject_nonfinite: true)
      end

      private def encode_comparison!(chain : ML::Metal::MPSCommandChain,
                                     actual : ML::MetalBuffer,
                                     expected : ML::MetalBuffer,
                                     slot : Int32, elements : Int64,
                                     inject_nonfinite : Bool = false) : Nil
        encoder = chain.encoder
        encoder.set_pipeline(@pipeline || raise ArgumentError.new("FFN output comparator is unavailable"))
        encoder.set_buffer(actual, 0)
        encoder.set_buffer(expected, 1)
        encoder.set_buffer(metrics_buffer, 2, ML::Metal::BufferAccess::ReadWrite,
          slot.to_i64 * 16)
        encoder.set_value(elements.to_u32, 3)
        encoder.set_value(elements.to_u32, 4)
        encoder.set_value(inject_nonfinite ? 1_u32 : 0_u32, 5)
        encoder.dispatch_threadgroups({(elements.to_i32 + 255) // 256, 1, 1}, {256, 1, 1})
      end

      private def validate_request_metrics!(metrics : Array(Metric),
                                            entry : ScheduleEntry) : Nil
        raise ArgumentError.new("packed FFN comparator metrics slot count mismatch") unless metrics.size == METRIC_SLOTS
        if entry[:mode] == "staged_f32" && entry[:capture_mode]
          STAGE_METRIC_SLOTS.times do |slot|
            metric = metrics[slot]
            unless metric[:elements] == STAGE_ELEMENT_COUNT &&
                   metric[:failures] == 0_u32 && metric[:different_words] == 0_u32 &&
                   metric[:nonfinite] == 0_u32
              raise ArgumentError.new("staged F32 weight identity failed for native plan slot #{slot}: #{PixelUMMFFNWeightReuseProbe.metric_diagnostic(metric)}")
            end
          end
        else
          STAGE_METRIC_SLOTS.times do |slot|
            raise ArgumentError.new("FFN request unexpectedly checked/staged F32 weights") unless metrics[slot] == {failures: 0_u32, different_words: 0_u32, nonfinite: 0_u32, elements: 0_u32}
          end
        end
        4.times do |plan_slot|
          CAPTURE_ORDER.each_with_index do |stage, stage_index|
            slot = PixelUMMFFNWeightReuseProbe.metric_slot(plan_slot, stage_index)
            metric = metrics[slot]
            should_compare = entry[:capture_mode] || stage == "output"
            if should_compare
              expected_elements = (@expected_records[plan_slot][stage].nbytes // 2).to_u32
              unless metric[:elements] == expected_elements && metric[:failures] == 0_u32 &&
                     metric[:nonfinite] == 0_u32
                raise ArgumentError.new("FFN upstream capture comparison failed at plan #{plan_slot}/#{stage}: #{PixelUMMFFNWeightReuseProbe.metric_diagnostic(metric)}")
              end
            else
              unless metric == {failures: 0_u32, different_words: 0_u32, nonfinite: 0_u32, elements: 0_u32}
                raise ArgumentError.new("FFN output-only request recorded an unexpected intermediate comparison")
              end
            end
          end
        end
        if entry[:index] == 0
          control = metrics[POSITIVE_CONTROL_SLOT]
          output_elements = (@expected_records[0]["output"].nbytes // 2).to_u32
          unless control[:elements] == output_elements && control[:failures] == 1_u32 &&
                 control[:different_words] == 0_u32 && control[:nonfinite] == 1_u32
            raise ArgumentError.new("FFN comparator NaN positive control was not exact: #{PixelUMMFFNWeightReuseProbe.metric_diagnostic(control)}")
          end
        else
          control = metrics[POSITIVE_CONTROL_SLOT]
          unless control[:elements] == 0_u32 && control[:failures] == 0_u32
            raise ArgumentError.new("FFN comparator positive-control slot was not clear")
          end
        end
      end

      private def read_metrics! : Array(Metric)
        words = Array(UInt32).new(METRIC_SLOTS * METRIC_WORDS, 0_u32)
        metrics_buffer.read_bytes(words.to_unsafe.as(UInt8*), @metric_bytes.to_i32)
        Array.new(METRIC_SLOTS) do |index|
          {failures: words[index * 4], different_words: words[index * 4 + 1],
           nonfinite: words[index * 4 + 2], elements: words[index * 4 + 3]}
        end
      end

      private def blit_fill!(chain : ML::Metal::MPSCommandChain,
                             buffers : Array(ML::MetalBuffer),
                             value : UInt8 = 255_u8) : Nil
        chain.pause!
        blit = Pointer(Void).null
        begin
          blit = MetalDispatchFFI.create_blit_encoder(chain.root_handle)
          raise ArgumentError.new("FFN probe GPU poison blit could not be created") if blit.null?
          seen = Set(UInt64).new
          buffers.each do |buffer|
            next unless seen.add?(buffer.handle.address.to_u64)
            MetalDispatchFFI.blit_fill_buffer(blit, buffer.handle, value, 0_i64, buffer.size)
          end
        ensure
          MetalDispatchFFI.blit_end_encoding(blit) unless blit.null?
          chain.resume!
        end
      end

      private def metrics_buffer : ML::MetalBuffer
        @metrics_buffer || raise ArgumentError.new("FFN packed metrics buffer is unavailable")
      end

      private def error_message : String
        String.new(@error_text.to_unsafe)
      end

      private def validate_actual_resources! : Nil
        workspace_bytes = @workspace.not_nil!.named_bytes
        input_weight_bytes = @input_buffers.values.sum { |set| set.values.sum(&.size) } +
                             @weight_buffers.values.sum { |set| set.values.sum(&.size) }
        expected_bytes = @expected_buffers.sum { |set| set.values.sum(&.size) }
        stage_bytes = @stage_buffers.sum(&.size)
        actual_total = workspace_bytes + input_weight_bytes + expected_bytes +
                       stage_bytes + metrics_buffer.size
        unless workspace_bytes == @budget[:workspace_bytes] &&
               input_weight_bytes == @budget[:input_weight_bytes] &&
               expected_bytes == @budget[:expected_capture_bytes] &&
               stage_bytes == @budget[:stage_bytes] &&
               metrics_buffer.size == @metric_bytes &&
               actual_total == @budget[:total_bytes] &&
               actual_total <= @maximum
          raise ArgumentError.new("FFN actual named-buffer inventory differs from pre-Metal admission")
        end
      end

      private def release_resources! : Nil
        return if @released
        @wrappers.each { |wrapper| PixelUMMFFNWeightReuseBridge.release(wrapper) }
        @wrappers.clear
        @native_plans.each { |plan| PixelUMMDecoderBlockMPSFFI.plan_release(plan) }
        @native_plans.clear
        @native_plan_handles.clear
        @workspace.try(&.release)
        @workspace = nil
        @expected_buffers.each { |set| set.each_value(&.release) }
        @expected_buffers.clear
        @input_buffers.each_value(&.each_value(&.release))
        @input_buffers.clear
        @weight_buffers.each_value(&.each_value(&.release))
        @weight_buffers.clear
        @stage_buffers.each(&.release)
        @stage_buffers.clear
        @metrics_buffer.try(&.release)
        @metrics_buffer = nil
        @released = true
        @memory.emit("after_all_named_buffers_release",
          phase_bytes_kind: "owned_named_metal_bytes", phase_bytes: 0_i64)
      end

      private def check_live! : Nil
        raise ArgumentError.new("FFN weight reuse probe owner has been released") if @released
      end
    end
  {% end %}
end
