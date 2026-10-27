require "./pixelumm_block_probe"
require "./pixelumm_memory_probe"

module PixelUMMTwoBlockProbe
  extend self
  alias Record = PixelUMMBlockProbe::Record
  alias Comparison = PixelUMMBlockProbe::Comparison
  alias Decoder = ML::Video::PixelUMMDecoderBlockMPS
  HF_REVISION   = "c67f2c52b3fbe0193cbc202769bbe5cb7264807b"
  HEADER_SHA256 = "7e7a92cfe84d54c0da3a132099a9472766ae0034042515a0f76b6198bbd09d11"
  MODEL_SHA256  = "bea01a8490f8e9a4076da83d4f9fbb3cba7bd6701df5c28fb2e2c04f00131eec"
  SOURCE_PATH   = "VAE/PixelUMM_FT/diffusion_pytorch_model.safetensors"
  GIB           = 1024_i64 * 1024 * 1024

  def validate_identity!(document : JSON::Any) : Nil
    unless document["upstream_revision"].as_s == PixelUMMBlockProbe::UPSTREAM_REVISION &&
           document["upstream_model_sha256"].as_s == MODEL_SHA256 &&
           document["hf_revision"].as_s == HF_REVISION &&
           document["hf_source_path"].as_s == SOURCE_PATH &&
           document["checkpoint_header_sha256"].as_s == HEADER_SHA256
      raise ArgumentError.new("PixelUMM pinned source identity mismatch")
    end
  end

  def validate_receipt_binding!(cache_block : JSON::Any, fixture_block : JSON::Any, slot : Int32) : Nil
    unless cache_block["selected_block"].as_i == slot &&
           cache_block["subset_sha256"].as_s == fixture_block["checkpoint_subset_sha256"].as_s &&
           cache_block["source_receipt_sha256"].as_s == fixture_block["source_receipt_sha256"].as_s &&
           cache_block["subset_bytes"].as_i64 == fixture_block["checkpoint_subset_bytes"].as_i64 &&
           cache_block["source_extent"] == fixture_block["source_extent"]
      raise ArgumentError.new("BF16 cache source receipt binding mismatch")
    end
  end

  def validate_block_count!(block_count : Int32) : Nil
    raise ArgumentError.new("serial block count must be two, four, eight, sixteen or thirty-six") unless {2, 4, 8, 16, 36}.includes?(block_count)
  end

  def validate_grid!(grid : Array(Int32), block_count : Int32) : Nil
    validate_block_count!(block_count)
    unless grid == [2, 32, 32] || grid == [6, 44, 80]
      raise ArgumentError.new("serial fixture grid is outside admitted contexts")
    end
    if block_count == 36 && grid != [6, 44, 80]
      raise ArgumentError.new("36-block fixture requires the full T6/H44/W80 grid")
    end
  end

  def validate_serial_execution_mode!(block_count : Int32, *, ffn_f32 : Bool,
                                      reuse_outputs : Bool) : Nil
    validate_block_count!(block_count)
    if block_count == 36
      raise ArgumentError.new("36-block probe requires --ffn-f32") unless ffn_f32
      raise ArgumentError.new("36-block probe requires --reuse-output-banks") unless reuse_outputs
    end
  end

  def fixture_payload_limit(block_count : Int32) : Int64
    validate_block_count!(block_count)
    case block_count
    when  8 then 8_i64 * GIB
    when 16 then 16_i64 * GIB
    when 36 then 32_i64 * GIB
    else         4_i64 * GIB
    end
  end

  def weight_cache_payload_limit(block_count : Int32) : Int64
    validate_block_count!(block_count)
    case block_count
    when 16 then 8_i64 * GIB
    when 36 then 16_i64 * GIB
    else         4_i64 * GIB
    end
  end

  def named_budget_limit(block_count : Int32) : Int64
    validate_block_count!(block_count)
    case block_count
    when 16 then 16_i64 * GIB
    when 36 then 24_i64 * GIB
    else         8_i64 * GIB
    end
  end

  def fixture_schema(block_count : Int32) : String
    validate_block_count!(block_count)
    label = {2 => "two", 4 => "four", 8 => "eight", 16 => "sixteen", 36 => "thirty_six"}[block_count]
    "cogni_ml.pixelumm_#{label}_block_reference/v1"
  end

  def capture_slot(key : String, block_count : Int32) : Int32
    validate_block_count!(block_count)
    slot = (0...block_count).find { |index| key.starts_with?("block#{index}.") }
    slot || raise ArgumentError.new("capture has unknown block ownership")
  end

  def capture_stage(key : String, block_count : Int32) : String
    slot = capture_slot(key, block_count)
    key["block#{slot}.".size..]
  end

  def validate_cache_block_keys!(keys : Array(String), block_count : Int32) : Nil
    validate_block_count!(block_count)
    unless keys.sort == (0...block_count).map(&.to_s).sort
      raise ArgumentError.new("BF16 cache block inventory mismatch")
    end
  end

  def validate_replay_output_distinguishability!(captures : Hash(String, Record),
                                                 alternate_captures : Hash(String, Record),
                                                 block_count : Int32) : Nil
    validate_block_count!(block_count)
    slots = block_count == 36 ? [34, 35] : (0...block_count).to_a
    slots.each do |slot|
      key = "block#{slot}.output"
      primary = captures[key]? || raise ArgumentError.new("replay output capture missing: #{key}")
      alternate = alternate_captures[key]? || raise ArgumentError.new("alternate replay output capture missing: #{key}")
      if primary.sha256 == alternate.sha256
        raise ArgumentError.new("alternate oracle outputs must distinguish every retained block output")
      end
    end
  end

  def thirty_six_canonical_capture_order : Array(String)
    (0...36).flat_map { |slot| ["block#{slot}.attention", "block#{slot}.output"] }
  end

  def thirty_six_survivor_capture_order : Array(String)
    (0...36).flat_map do |slot|
      slot < 34 ? ["block#{slot}.attention"] : ["block#{slot}.attention", "block#{slot}.output"]
    end
  end

  def thirty_six_unretained_output_aliases : Array(String)
    thirty_six_canonical_capture_order - thirty_six_survivor_capture_order
  end

  def validate_thirty_six_capture_policy!(document : JSON::Any) : Nil
    policy = document.as_h["full36_capture_policy"]?
    unless policy && policy.as_s == "two_bank_survivors"
      raise ArgumentError.new("36-block fixture capture policy mismatch")
    end
  end

  def validate_serial_capture_inventory!(keys : Array(String), grid : Array(Int32), block_count : Int32) : Nil
    validate_grid!(grid, block_count)
    unless {8, 16, 36}.includes?(block_count)
      raise ArgumentError.new("full serial capture inventories are defined for eight, sixteen or thirty-six blocks")
    end
    expected = if block_count == 36
                 thirty_six_survivor_capture_order
               else
                 stages = grid == [2, 32, 32] ? Decoder::Plan::CAPTURE_NAMES.to_a : ["attention", "output"]
                 (0...block_count).flat_map { |slot| stages.map { |stage| "block#{slot}.#{stage}" } }
               end
    unless keys.sort == expected.sort
      message = block_count == 36 ? "36-block fixture requires exactly the two-bank survivor capture inventory" : "#{block_count}-block fixture requires every expected stage at its admitted grid"
      raise ArgumentError.new(message)
    end
  end

  # Select the common postterminal capture subset that survives two alternating
  # output banks. This changes only which CPU diagnostics are requested; callers
  # still admit and allocate their actual workspace using their own bank mode.
  def common_ring_comparison_capture_order(keys : Array(String), grid : Array(Int32), block_count : Int32) : Array(String)
    validate_grid!(grid, block_count)
    selected = keys.select do |key|
      slot = capture_slot(key, block_count)
      stage = capture_stage(key, block_count)
      Decoder::Workspace.capture_survives?(stage, slot, block_count, reuse_outputs: true)
    end
    raise ArgumentError.new("common-ring comparison capture selection must not be empty") if selected.empty?

    expected_count = if block_count == 16
                       case grid
                       when [2, 32, 32] then 402
                       when [6, 44, 80] then 18
                       else                  nil
                       end
                     elsif block_count == 36
                       grid == [6, 44, 80] ? 38 : nil
                     else
                       nil
                     end
    if expected_count && selected.size != expected_count
      raise ArgumentError.new("common-ring comparison capture count mismatch: expected #{expected_count}, got #{selected.size}")
    end
    selected
  end

  def validate_eight_capture_inventory!(keys : Array(String), grid : Array(Int32)) : Nil
    validate_serial_capture_inventory!(keys, grid, 8)
  end

  # This checks the authenticated fixture's metadata contract. The probe does
  # not independently recompute the upstream PixelUMM stem from latent/xt.
  def validate_real_input_provenance_metadata!(document : JSON::Any, input : Record) : Nil
    provenance = document["input_provenance"]
    expected_keys = ["kind", "latent_sha256", "xt_sha256", "stem_weights_sha256", "derived_x_sha256"]
    unless provenance.as_h.keys.sort == expected_keys.sort
      raise ArgumentError.new("serial fixture provenance metadata fields mismatch")
    end
    required_hashes = {"latent_sha256", "xt_sha256", "stem_weights_sha256"}
    valid_hashes = required_hashes.all? { |key| PixelUMMBlockProbe.valid_digest?(provenance[key].as_s) }
    unless provenance["kind"].as_s == "real_latent_xt_pixelumm_stem" && valid_hashes &&
           provenance["derived_x_sha256"].as_s == input.sha256
      raise ArgumentError.new("serial fixture lacks valid real-context input provenance metadata")
    end
  end

  def validate_thirty_six_input_lineage_metadata!(document : JSON::Any) : Nil
    lineage = document["input_lineage"].as_h
    manifest_sha = document["real_input_manifest_sha256"].as_s
    payload_sha = document["real_input_payload_sha256"].as_s
    payload_bytes = document["real_input_payload_bytes"].as_i64
    unless !lineage.empty? && PixelUMMBlockProbe.valid_digest?(manifest_sha) &&
           PixelUMMBlockProbe.valid_digest?(payload_sha) && payload_bytes > 0
      raise ArgumentError.new("36-block fixture lacks authenticated real-input lineage metadata")
    end
  end

  def validate_thirty_six_weight_word_verification!(document : JSON::Any) : Nil
    verification = document["weight_word_verification"]
    expected_fields = ["all_equal", "method", "tensor_count"]
    unless verification.as_h.keys.sort == expected_fields &&
           verification["method"].as_s == "loaded_mps_bf16_int16_word_equality" &&
           verification["tensor_count"].as_i == 468 && verification["all_equal"].as_bool
      raise ArgumentError.new("36-block weight word verification metadata mismatch")
    end
  end

  def validate_thirty_six_weight_payload_order!(weights : Hash(String, Record)) : Nil
    expected = (0...36).flat_map do |slot|
      PixelUMMBlockProbe.weight_shapes.keys.sort.map { |name| "block#{slot}.#{name}" }
    end
    actual = weights.to_a.sort_by { |pair| pair[1].offset }.map { |pair| pair[0] }
    unless actual == expected
      raise ArgumentError.new("36-block BF16 weight payload records must be in numeric block/sorted-name order")
    end
  end

  def validate_thirty_six_capture_payload_order!(captures : Hash(String, Record),
                                                 capture_order : Array(String)) : Nil
    expected = thirty_six_survivor_capture_order
    payload_order = captures.to_a.sort_by { |pair| pair[1].offset }.map { |pair| pair[0] }
    unless capture_order == expected && payload_order == expected
      raise ArgumentError.new("36-block captures must be in numeric block/stage order")
    end
  end

  def validate_thirty_six_weight_cache_link_metadata!(document : JSON::Any, weights : Hash(String, Record)) : Nil
    manifest_sha = document["weight_cache_manifest_sha256"].as_s
    payload_sha = document["weight_cache_payload_sha256"].as_s
    payload_bytes = document["weight_cache_payload_bytes"].as_i64
    expected_bytes = weights.values.sum(&.nbytes)
    unless PixelUMMBlockProbe.valid_digest?(manifest_sha) &&
           PixelUMMBlockProbe.valid_digest?(payload_sha) && payload_bytes == expected_bytes
      raise ArgumentError.new("36-block fixture BF16 cache linkage metadata mismatch")
    end
  end

  def validate_thirty_six_weight_cache_binding!(document : JSON::Any, manifest_sha : String,
                                                payload_sha : String, payload_bytes : Int64) : Nil
    unless document["weight_cache_manifest_sha256"].as_s == manifest_sha &&
           document["weight_cache_payload_sha256"].as_s == payload_sha &&
           document["weight_cache_payload_bytes"].as_i64 == payload_bytes
      raise ArgumentError.new("36-block fixture is not bound to its authenticated BF16 cache")
    end
  end

  def validate_ffn_precision_slots!(slots : Array(Bool), block_count : Int32, enabled : Bool) : Nil
    validate_block_count!(block_count)
    unless slots.size == block_count && slots.all? { |value| value == enabled }
      raise ArgumentError.new("serial plans must share one FFN graph precision across every slot")
    end
  end

  def validate_output_bank_layout!(indices : Array(Int32), block_count : Int32, reuse_outputs : Bool) : Nil
    validate_block_count!(block_count)
    expected = Array.new(block_count) do |slot|
      Decoder::Workspace.output_bank_index(slot, block_count, reuse_outputs: reuse_outputs)
    end
    unless indices == expected
      raise ArgumentError.new("serial output bank identities do not match the admitted layout")
    end
  end

  def validate_input_source!(source : String, slot : Int32) : Nil
    expected = slot == 0 ? "inputs.x" : "captures.block#{slot - 1}.output"
    raise ArgumentError.new("block input must reference its immediate predecessor") unless source == expected
  end

  def validate_capture!(key : String, record : Record, tokens : Int32, block_count : Int32 = 2) : Nil
    name = capture_stage(key, block_count)
    dtype, shape = case name
                   when "input_norm_statistics"
                     {"F32", [1, tokens, 1]}
                   when "q_norm_statistics", "q_norm_inverse"
                     {"F32", [1, tokens, 32, 2]}
                   when "k_norm_statistics", "k_norm_inverse"
                     {"F32", [1, tokens, 8, 2]}
                   when "q_norm", "q_rope"
                     {"F32", [1, tokens, 32, 128]}
                   when "k_norm", "k_rope"
                     {"F32", [1, tokens, 8, 128]}
                   when "q_proj", "q_ready", "attention"
                     {"BF16", [1, tokens, 32, 128]}
                   when "k_proj", "v_proj", "k_ready"
                     {"BF16", [1, tokens, 8, 128]}
                   when "gate_proj", "up_proj", "silu_gate", "product"
                     {"BF16", [1, tokens, 12288]}
                   when "x", "input_norm_pre_gain", "input_norm", "o_proj", "attn_residual", "post_attn_norm", "down_proj", "output"
                     {"BF16", [1, tokens, 4096]}
                   else
                     raise ArgumentError.new("unknown PixelUMM capture stage")
                   end
    unless record.dtype == dtype && record.shape == shape
      raise ArgumentError.new("capture stage dtype/shape mismatch")
    end
  end

  # Metadata is small and authenticated before parsing. Large payloads are
  # streamed only after aggregate memory admission, before GPU allocation.
  class TensorFile
    getter path : String
    getter sha256 : String
    getter document : JSON::Any
    getter payload : String
    getter payload_bytes : Int64
    @verified = false

    def initialize(@path : String, trusted_sha256 : String, max_payload_bytes : Int64 = 4_i64 * 1024**3)
      unless max_payload_bytes == 4_i64 * GIB || max_payload_bytes == 8_i64 * GIB ||
             max_payload_bytes == 16_i64 * GIB || max_payload_bytes == 32_i64 * GIB
        raise ArgumentError.new("unsupported tensor payload byte limit")
      end
      unless Path[@path].absolute? && File.file?(@path) && !File.symlink?(@path) &&
             File.realpath(@path) == @path && File.size(@path) <= 1024 * 1024
        raise ArgumentError.new("tensor metadata must be a bounded canonical regular file")
      end
      raw = File.read(@path)
      @sha256 = Digest::SHA256.hexdigest(raw)
      unless PixelUMMBlockProbe.valid_digest?(trusted_sha256) && @sha256 == trusted_sha256
        raise ArgumentError.new("tensor metadata trusted identity mismatch")
      end
      @document = JSON.parse(raw)
      basename = @document["payload"].as_s
      unless basename.matches?(/\A[A-Za-z0-9_.-]+\z/) && basename != "." && basename != ".."
        raise ArgumentError.new("tensor payload must be a sibling basename")
      end
      @payload = File.join(File.dirname(@path), basename)
      unless File.file?(@payload) && !File.symlink?(@payload) && File.realpath(@payload) == @payload
        raise ArgumentError.new("tensor payload must be a canonical regular sibling")
      end
      @payload_bytes = File.size(@payload)
      unless @payload_bytes == @document["payload_bytes"].as_i64 &&
             0 < @payload_bytes <= max_payload_bytes &&
             PixelUMMBlockProbe.valid_digest?(@document["payload_sha256"].as_s)
        raise ArgumentError.new("tensor payload extent/identity mismatch")
      end
    end

    def records(key : String) : Hash(String, Record)
      @document[key].as_h.transform_values { |value| Record.new(value, @payload_bytes) }
    end

    def validate_ranges!(records : Array(Record), contiguous : Bool = false) : Nil
      cursor = 0_i64
      records.sort_by(&.offset).each do |record|
        if record.offset < cursor || (contiguous && record.offset != cursor)
          raise ArgumentError.new("overlapping or missing tensor payload ranges")
        end
        cursor = record.offset + record.nbytes
      end
      raise ArgumentError.new("trailing tensor payload bytes") if contiguous && cursor != @payload_bytes
    end

    def verify_payload! : Nil
      return if @verified
      unless PixelUMMBlockProbe.sha_file(@payload) == @document["payload_sha256"].as_s
        raise ArgumentError.new("tensor payload hash mismatch")
      end
      @verified = true
    end

    def read(record : Record) : Bytes
      raise ArgumentError.new("tensor payload was not authenticated") unless @verified
      bytes = Bytes.new(record.nbytes.to_i32)
      read_into(record, bytes)
      bytes
    end

    def upload_into!(record : Record, buffer : ML::MetalBuffer) : Nil
      raise ArgumentError.new("tensor upload extent/storage mismatch") unless buffer.valid? &&
                                                                              buffer.size == record.nbytes && buffer.storage_mode == ML::StorageMode::Shared
      raise ArgumentError.new("tensor payload was not authenticated") unless @verified
      # Startup file I/O writes the existing shared GPU allocation directly;
      # there is no transient full-weight CPU allocation or dtype conversion.
      read_into(record, buffer.contents.as(UInt8*).to_slice(record.nbytes.to_i32))
    end

    private def read_into(record : Record, bytes : Bytes) : Nil
      File.open(@payload) do |io|
        io.seek(record.offset)
        io.read_fully(bytes)
      end
      unless Digest::SHA256.hexdigest(bytes) == record.sha256
        raise ArgumentError.new("tensor record hash mismatch")
      end
    end
  end

  def validate_weights!(weights : Hash(String, Record), block_count : Int32 = 2) : Nil
    validate_block_count!(block_count)
    expected = PixelUMMBlockProbe.weight_shapes
    raise ArgumentError.new("serial blocks require exactly #{13 * block_count} weights") unless weights.size == 13 * block_count
    block_count.times do |slot|
      expected.each do |name, shape|
        record = weights["block#{slot}.#{name}"]? || raise ArgumentError.new("missing block weight")
        unless record.dtype == "BF16" && record.shape == shape
          raise ArgumentError.new("block weight dtype/shape mismatch")
        end
      end
    end
  end

  class Fixture < TensorFile
    getter grid : Array(Int32)
    getter inputs : Hash(String, Record)
    getter weights : Hash(String, Record)
    getter captures : Hash(String, Record)
    getter capture_order : Array(String)
    getter subset_hashes : Array(String)
    getter block_count : Int32

    def initialize(path : String, trusted_sha256 : String, @block_count : Int32 = 2)
      PixelUMMTwoBlockProbe.validate_block_count!(@block_count)
      # Larger serial fixtures need explicit per-depth payload ceilings.
      super(path, trusted_sha256, PixelUMMTwoBlockProbe.fixture_payload_limit(@block_count))
      d = @document
      PixelUMMTwoBlockProbe.validate_identity!(d)
      schema = PixelUMMTwoBlockProbe.fixture_schema(@block_count)
      unless d["schema"].as_s == schema &&
             d["upstream_revision"].as_s == PixelUMMBlockProbe::UPSTREAM_REVISION &&
             d["tile_tokens"].as_i == 8 && d["token_order"].as_s == "tile_major"
        raise ArgumentError.new("serial fixture provenance/layout mismatch")
      end
      PixelUMMTwoBlockProbe.validate_thirty_six_capture_policy!(d) if @block_count == 36
      @grid = d["grid"].as_a.map(&.as_i)
      PixelUMMTwoBlockProbe.validate_grid!(@grid, @block_count)
      @inputs, @weights, @captures = records("inputs"), records("weights"), records("captures")
      @capture_order = d["capture_order"].as_a.map(&.as_s)
      block_captures_valid = if @block_count == 36
                               (0...36).all? do |slot|
                                 expected = slot < 34 ? ["attention"] : ["attention", "output"]
                                 expected.all? { |name| @captures.has_key?("block#{slot}.#{name}") } &&
                                   @captures.keys.select(&.starts_with?("block#{slot}.")).map { |key| PixelUMMTwoBlockProbe.capture_stage(key, 36) }.sort == expected.sort
                               end
                             else
                               (0...@block_count).all? do |slot|
                                 {"attention", "output"}.all? { |name| @captures.has_key?("block#{slot}.#{name}") }
                               end
                             end
      unless @inputs.keys.sort == ["cos", "sin", "x"] &&
             @capture_order.uniq == @capture_order && @capture_order.sort == @captures.keys.sort &&
             block_captures_valid
        raise ArgumentError.new("serial fixture input/capture inventory mismatch")
      end
      tokens = @grid.reduce(1) { |n, v| n * v }
      @captures.each { |key, record| PixelUMMTwoBlockProbe.validate_capture!(key, record, tokens, @block_count) }
      PixelUMMTwoBlockProbe.validate_serial_capture_inventory!(@capture_order, @grid, @block_count) if @block_count >= 8
      unless @inputs["x"].shape == [1, tokens, 4096] && @inputs["x"].dtype == "BF16" &&
             {"cos", "sin"}.all? { |name| @inputs[name].shape == [1, tokens, 1, 128] && @inputs[name].dtype == "F32" }
        raise ArgumentError.new("serial input geometry/dtype mismatch")
      end
      PixelUMMTwoBlockProbe.validate_weights!(@weights, @block_count)
      PixelUMMTwoBlockProbe.validate_real_input_provenance_metadata!(d, @inputs["x"]) if {16, 36}.includes?(@block_count)
      if @block_count == 36
        PixelUMMTwoBlockProbe.validate_thirty_six_input_lineage_metadata!(d)
        PixelUMMTwoBlockProbe.validate_thirty_six_weight_word_verification!(d)
        PixelUMMTwoBlockProbe.validate_thirty_six_weight_cache_link_metadata!(d, @weights)
        PixelUMMTwoBlockProbe.validate_thirty_six_weight_payload_order!(@weights)
        PixelUMMTwoBlockProbe.validate_thirty_six_capture_payload_order!(@captures, @capture_order)
      end
      blocks = d["blocks"].as_a
      raise ArgumentError.new("serial source inventory mismatch") unless blocks.size == @block_count
      if @block_count == 36
        blocks.each_with_index do |block, index|
          subset_bytes = block["checkpoint_subset_bytes"].as_i64
          source_extent = block["source_extent"].as_a.map(&.as_i64)
          unless subset_bytes > 0 && source_extent.size == 2 &&
                 source_extent[0] >= 0 && source_extent[1] > source_extent[0]
            raise ArgumentError.new("36-block source receipt extent mismatch at slot #{index}")
          end
          unless block["weight_word_check_count"].as_i == 13
            raise ArgumentError.new("36-block weight word verification count mismatch at slot #{index}")
          end
        end
      end
      @subset_hashes = blocks.map_with_index do |block, index|
        prefix = "block#{index}."
        PixelUMMTwoBlockProbe.validate_input_source!(block["input_source"].as_s, index)
        digest = block["checkpoint_subset_sha256"].as_s
        unless block["index"].as_i == index && block["weight_prefix"].as_s == prefix &&
               block["output_key"].as_s == "captures.block#{index}.output" &&
               (if @block_count == 36
                 block["weight_keys"].as_a.map(&.as_s) == PixelUMMBlockProbe.weight_shapes.keys.sort.map { |name| prefix + name }
               else
                 block["weight_keys"].as_a.map(&.as_s).sort == @weights.keys.select(&.starts_with?(prefix)).sort
               end) &&
               (if @block_count == 36
                 expected_keys = index < 34 ? ["block#{index}.attention"] : ["block#{index}.attention", "block#{index}.output"]
                 block["capture_keys"].as_a.map(&.as_s) == expected_keys
               else
                 block["capture_keys"].as_a.map(&.as_s).sort == @captures.keys.select(&.starts_with?(prefix)).sort
               end) &&
               PixelUMMBlockProbe.valid_digest?(digest) &&
               PixelUMMBlockProbe.valid_digest?(block["source_receipt_sha256"].as_s)
          raise ArgumentError.new("serial source/ownership mismatch")
        end
        digest
      end
      unless @subset_hashes[0] == PixelUMMBlockProbe::SUBSET_SHA256
        raise ArgumentError.new("block zero source identity mismatch")
      end
      validate_ranges!(@inputs.values + @weights.values + @captures.values, contiguous: @block_count >= 8)
    end

    def captures_by_slot : Array(Array(String))
      Array.new(@block_count) do |slot|
        prefix = "block#{slot}."
        @capture_order.select(&.starts_with?(prefix)).map { |name| name[prefix.size..] }
      end
    end

    def compatible_replay!(other : Fixture) : Nil
      unless @block_count == other.block_count && @grid == other.grid && @subset_hashes == other.subset_hashes &&
             @capture_order == other.capture_order && @weights.keys.all? { |key| @weights[key].sha256 == other.weights[key].sha256 } &&
             {"cos", "sin"}.all? { |key| @inputs[key].sha256 == other.inputs[key].sha256 } &&
             @inputs["x"].sha256 != other.inputs["x"].sha256
        raise ArgumentError.new("replay must change only the initial hidden input")
      end
    end
  end

  class WeightCache < TensorFile
    getter weights : Hash(String, Record)

    def initialize(path : String, trusted_sha256 : String, fixture : Fixture)
      super(path, trusted_sha256, PixelUMMTwoBlockProbe.weight_cache_payload_limit(fixture.block_count))
      d = @document
      PixelUMMTwoBlockProbe.validate_identity!(d)
      raise ArgumentError.new("unsupported BF16 cache schema") unless d["schema"].as_s == "cogni_ml.pixelumm_bf16_weight_cache/v1"
      if fixture.block_count == 36
        PixelUMMTwoBlockProbe.validate_thirty_six_weight_cache_binding!(fixture.document, @sha256,
          d["payload_sha256"].as_s, @payload_bytes)
      end
      unless d["source_commit"].as_s == PixelUMMBlockProbe::UPSTREAM_REVISION &&
             d["stored_dtype"].as_s == "F32" && d["runtime_dtype"].as_s == "BF16" &&
             !d["full_file_hash_verified"].as_bool
        raise ArgumentError.new("BF16 cache stored/runtime source mismatch")
      end
      @weights = records("weights")
      PixelUMMTwoBlockProbe.validate_weights!(@weights, fixture.block_count)
      PixelUMMTwoBlockProbe.validate_thirty_six_weight_payload_order!(@weights) if fixture.block_count == 36
      conversion = d["conversion"]
      unless conversion["method"].as_s == "torch.Tensor.to(torch.bfloat16)" &&
             conversion["device"].as_s == "mps" && conversion["rounding"].as_s == "round_to_nearest_even" &&
             !conversion["torch_version"].as_s.empty?
        raise ArgumentError.new("BF16 cache conversion provenance mismatch")
      end
      PixelUMMTwoBlockProbe.validate_cache_block_keys!(d["blocks"].as_h.keys, fixture.block_count)
      unless d["comparison"]["all_raw_uint16_equal"].as_bool &&
             d["comparison"]["tensor_count"].as_i == fixture.block_count * 13 &&
             d["comparison"]["weights"].as_h.keys.sort == @weights.keys.sort
        raise ArgumentError.new("BF16 cache raw-word comparison inventory mismatch")
      end
      fixture.block_count.times do |slot|
        block = d["blocks"][slot.to_s]
        source = fixture.document["blocks"][slot]
        PixelUMMTwoBlockProbe.validate_receipt_binding!(block, source, slot)
        unless block["weight_keys"].as_a.map(&.as_s).sort == @weights.keys.select(&.starts_with?("block#{slot}.")).sort &&
               block["bf16_bytes"].as_i64 == @weights.select { |key, _| key.starts_with?("block#{slot}.") }.values.sum(&.nbytes)
          raise ArgumentError.new("BF16 cache block identity mismatch")
        end
      end
      @weights.each do |key, record|
        expected = fixture.weights[key]
        unless record.dtype == expected.dtype && record.shape == expected.shape &&
               record.nbytes == expected.nbytes && record.sha256 == expected.sha256
          raise ArgumentError.new("BF16 cache differs from upstream fixture weight #{key}")
        end
        comparison = d["comparison"]["weights"][key]
        unless comparison["raw_uint16_equal"].as_bool && comparison["payload_bytes_equal"].as_bool &&
               comparison["source_raw_uint16_sha256"].as_s == record.sha256 &&
               comparison["payload_sha256"].as_s == record.sha256 &&
               comparison["shape"].as_a.map(&.as_i) == record.shape
          raise ArgumentError.new("BF16 cache raw-word comparison mismatch")
        end
      end
      validate_ranges!(@weights.values, contiguous: true)
    end
  end

  class Session
    getter fixture : Fixture
    getter borrowed_bytes : Int64
    getter named_bytes : Int64
    getter ffn_f32 : Bool
    getter reuse_outputs : Bool
    getter workspace : Decoder::Workspace
    getter plans = [] of Decoder::Plan
    getter capture_order : Array(String)
    getter workspace_capture_order : Array(String)
    getter output_bank_by_slot : Array(Int32)
    getter compare_common_ring_captures : Bool
    @bindings = {} of String => ML::MetalBuffer
    @last_chain : ML::Metal::MPSCommandChain? = nil
    @released = false
    @phase_observer : PixelUMMMemoryProbe::Recorder?

    def initialize(@fixture : Fixture, cache : WeightCache, max_named_bytes : Int64, *,
                   @ffn_f32 : Bool = false, @reuse_outputs : Bool = false,
                   @compare_common_ring_captures : Bool = false,
                   @phase_observer : PixelUMMMemoryProbe::Recorder? = nil)
      grid = @fixture.grid
      config = Decoder::Config.new(temporal: grid[0], height: grid[1], width: grid[2])
      weight_bytes = Array.new(@fixture.block_count) do |slot|
        PixelUMMBlockProbe.weight_shapes.keys.to_h { |name| {name, @fixture.weights["block#{slot}.#{name}"].nbytes} }
      end
      captures_by_slot = @fixture.captures_by_slot.map_with_index do |captures, slot|
        captures.select do |name|
          Decoder::Workspace.capture_survives?(name, slot, @fixture.block_count, reuse_outputs: @reuse_outputs)
        end
      end
      @workspace_capture_order = @fixture.capture_order.select do |key|
        slot = PixelUMMTwoBlockProbe.capture_slot(key, @fixture.block_count)
        captures_by_slot[slot].includes?(PixelUMMTwoBlockProbe.capture_stage(key, @fixture.block_count))
      end
      if @fixture.block_count == 36
        stable_ring_order = PixelUMMTwoBlockProbe.common_ring_comparison_capture_order(
          @workspace_capture_order, grid, 36)
        unless stable_ring_order == @workspace_capture_order &&
               stable_ring_order.count(&.ends_with?(".attention")) == 36 &&
               stable_ring_order.count(&.ends_with?(".output")) == 2 &&
               stable_ring_order.includes?("block34.output") && stable_ring_order.includes?("block35.output")
          raise ArgumentError.new("36-block ring must retain exactly 38 stable attention/output comparisons")
        end
      end
      @capture_order = if @compare_common_ring_captures
                         PixelUMMTwoBlockProbe.common_ring_comparison_capture_order(
                           @workspace_capture_order, grid, @fixture.block_count)
                       else
                         @workspace_capture_order
                       end
      @output_bank_by_slot = [] of Int32
      @borrowed_bytes = (@fixture.inputs.values + cache.weights.values).sum(&.nbytes)
      @named_bytes = Decoder::Workspace.validate_admission!(config,
        input_bytes: @fixture.inputs["x"].nbytes, cosine_bytes: @fixture.inputs["cos"].nbytes,
        sine_bytes: @fixture.inputs["sin"].nbytes, weights_bytes_by_slot: weight_bytes,
        captures_by_slot: captures_by_slot, max_unique_bytes: max_named_bytes, reuse_outputs: @reuse_outputs)
      # No file-sized read or Metal allocation precedes the aggregate gate.
      memory_phase("before_fixture_payload_authentication")
      @fixture.verify_payload!
      memory_phase("after_fixture_payload_authentication")
      memory_phase("before_cache_payload_authentication")
      cache.verify_payload!
      memory_phase("after_cache_payload_authentication")
      @workspace = Decoder::Workspace.new(config, captures_by_slot: captures_by_slot,
        max_owned_bytes: max_named_bytes - @borrowed_bytes, reuse_outputs: @reuse_outputs)
      input_uploaded_bytes = 0_i64
      begin
        memory_phase("after_workspace_alloc", phase_bytes_kind: "workspace_named_bytes",
          phase_bytes: @workspace.named_bytes) if @phase_observer
        @fixture.inputs.each do |name, record|
          @bindings[name] = buffer = ML::MetalBuffer.new(record.nbytes)
          @fixture.upload_into!(record, buffer)
          input_uploaded_bytes += record.nbytes if @phase_observer
        end
        memory_phase("after_input_upload", phase_bytes_kind: "fixture_input_buffers_uploaded_bytes",
          phase_bytes: input_uploaded_bytes) if @phase_observer

        expected_weight_bytes_by_slot = nil.as(Array(Int64)?)
        uploaded_weight_bytes_by_slot = nil.as(Array(Int64)?)
        if @phase_observer
          expected_weight_bytes_by_slot = weight_bytes.map { |per_slot| per_slot.values.sum }
          uploaded_weight_bytes_by_slot = Array.new(@fixture.block_count, 0_i64)
        end
        cache.weights.each do |name, record|
          @bindings[name] = buffer = ML::MetalBuffer.new(record.nbytes)
          cache.upload_into!(record, buffer)
          if @phase_observer
            slot = PixelUMMTwoBlockProbe.capture_slot(name, @fixture.block_count)
            uploaded = uploaded_weight_bytes_by_slot.not_nil!
            uploaded[slot] += record.nbytes
            if uploaded[slot] == expected_weight_bytes_by_slot.not_nil![slot]
              memory_phase("after_weight_slot_upload", slot: slot,
                phase_bytes_kind: "weight_slot_uploaded_bytes", phase_bytes: uploaded[slot])
            end
          end
        end
        @fixture.block_count.times do |slot|
          weights = PixelUMMBlockProbe.weight_shapes.keys.to_h { |name| {name, @bindings["block#{slot}.#{name}"]} }
          input = slot == 0 ? @bindings["x"] : @plans[slot - 1].output
          @plans << Decoder::Plan.new(config, input: input, weights: weights,
            cos: @bindings["cos"], sin: @bindings["sin"], captures: captures_by_slot[slot],
            max_named_bytes: max_named_bytes - @borrowed_bytes, workspace: @workspace, slot: slot,
            ffn_f32: @ffn_f32)
          memory_phase("after_plan_build", slot: slot, phase_bytes_kind: "plan_named_bytes",
            phase_bytes: @plans.last.named_bytes) if @phase_observer
        end
        PixelUMMTwoBlockProbe.validate_ffn_precision_slots!(@plans.map(&.ffn_f32), @fixture.block_count, @ffn_f32)
        # This loader owns distinct allocations for every authenticated weight,
        # the external input/tables and unique output banks. Broader caller-provided
        # Workspace aliases are outside this probe's ownership contract.
        outputs = @plans.map(&.output.handle)
        banks = outputs.uniq
        @output_bank_by_slot = outputs.map { |handle| banks.index(handle).not_nil!.as(Int32) }
        PixelUMMTwoBlockProbe.validate_output_bank_layout!(@output_bank_by_slot, @fixture.block_count, @reuse_outputs)
        handles = @bindings.values.map(&.handle) + banks
        unless @named_bytes == @borrowed_bytes + @workspace.named_bytes && handles.uniq.size == handles.size
          raise ArgumentError.new("serial unique-byte/output/weight ownership mismatch")
        end
      rescue error
        cleanup = PixelUMMMemoryProbe::Cleanup.new
        cleanup.phase(@phase_observer, "session_construction_rollback_before_release")
        @plans.reverse_each { |plan| cleanup.attempt { plan.release } }
        cleanup.phase(@phase_observer, "session_construction_rollback_after_plan_release")
        cleanup.attempt { @workspace.release }
        cleanup.phase(@phase_observer, "session_construction_rollback_after_workspace_release")
        @bindings.each_value { |binding| cleanup.attempt { binding.release } }
        cleanup.phase(@phase_observer, "session_construction_rollback_after_binding_release")
        raise error
      end
    end

    def encode(chain : ML::Metal::MPSCommandChain, *, profile : Bool = false) : Nil
      check_idle!
      @last_chain = chain
      @plans.each(&.encode(chain, profile: profile))
    end

    def replay_input!(fixture : Fixture) : Nil
      check_idle!
      @fixture.compatible_replay!(fixture)
      fixture.verify_payload!
      fixture.upload_into!(fixture.inputs["x"], @bindings["x"])
      @fixture = fixture
    end

    def poison_transients! : Nil
      check_idle!
      @workspace.mutable_shared_buffers(include_outputs: true).each_value do |buffer|
        # BF16 and F32 interpret all-one words as NaN. Only diagnostic setup
        # touches shared memory on the host, after the previous terminal wait.
        buffer.contents.as(UInt8*).to_slice(buffer.size.to_i32).fill(0xff_u8)
      end
    end

    def compare(*, request : Int32? = nil,
                phase_observer : PixelUMMMemoryProbe::Recorder? = nil) : Array(Comparison)
      chain = @last_chain || raise ArgumentError.new("PixelUMM serial was not encoded")
      unless chain.completed? && chain.wait_count == 1
        raise ArgumentError.new("PixelUMM serial diagnostics require one terminal wait")
      end
      @capture_order.map do |key|
        slot = PixelUMMTwoBlockProbe.capture_slot(key, @fixture.block_count)
        name = PixelUMMTwoBlockProbe.capture_stage(key, @fixture.block_count)
        record, buffer = @fixture.captures[key], @plans[slot].stage(name)
        raise ArgumentError.new("native capture extent mismatch #{key}") unless record.nbytes == buffer.size
        comparison_bytes = record.nbytes &* 2_i64
        memory_phase("before_compare_buffers", slot: slot, request: request,
          phase_bytes_kind: "cpu_expected_and_actual_buffer_bytes", phase_bytes: comparison_bytes,
          observer: phase_observer)
        expected, actual = @fixture.read(record), Bytes.new(record.nbytes.to_i32)
        memory_phase("compare_buffers_ready", slot: slot, request: request,
          phase_bytes_kind: "cpu_expected_and_actual_buffer_bytes", phase_bytes: comparison_bytes,
          observer: phase_observer)
        buffer.read_bytes(actual.to_unsafe, actual.size)
        width = record.dtype == "BF16" ? 2 : 4
        count, failures, different = record.nbytes // width, 0_i64, 0_i64
        maximum, finite = 0.0, true
        first_failure, first_expected, first_actual = nil.as(Int64?), nil.as(String?), nil.as(String?)
        count.times do |index|
          offset = (index * width).to_i32
          a, b = decode(expected, offset, width), decode(actual, offset, width)
          different += 1 unless expected[offset, width] == actual[offset, width]
          error = (a - b).abs
          bad = !a.finite? || !b.finite? || error > PixelUMMBlockProbe::ATOL + PixelUMMBlockProbe::RTOL * a.abs
          unless a.finite? && b.finite?
            finite, maximum = false, Float64::MAX
          else
            maximum = Math.max(maximum, error)
          end
          if bad
            failures += 1
            first_failure, first_expected, first_actual = index, a.to_s, b.to_s if first_failure.nil?
          end
        end
        comparison = {stage: key, elements: count, failures: failures, different_words: different,
                      max_abs: maximum, finite: finite, actual_sha256: Digest::SHA256.hexdigest(actual),
                      first_failure_index: first_failure, first_expected: first_expected, first_actual: first_actual}
        memory_phase("compare_stage_done", slot: slot, request: request,
          phase_bytes_kind: "cpu_expected_and_actual_buffer_bytes", phase_bytes: comparison_bytes,
          observer: phase_observer)
        comparison
      end
    end

    def release : Nil
      return if @released
      check_idle!
      # The session becomes unusable before teardown starts. Even if a resource
      # fails to release, attempt every sibling and prevent reuse of partial state.
      @released = true
      cleanup = PixelUMMMemoryProbe::Cleanup.new
      cleanup.phase(@phase_observer, "before_session_plan_release")
      @plans.reverse_each { |plan| cleanup.attempt { plan.release } }
      cleanup.phase(@phase_observer, "after_session_plan_release")
      cleanup.attempt { @workspace.release }
      cleanup.phase(@phase_observer, "after_session_workspace_release")
      @bindings.each_value { |binding| cleanup.attempt { binding.release } }
      cleanup.phase(@phase_observer, "after_session_binding_release")
      cleanup.phase(@phase_observer, "after_session_release")
      cleanup.raise_first!
      nil
    end

    private def memory_phase(phase : String, *, slot : Int32? = nil, request : Int32? = nil,
                             phase_bytes_kind : String? = nil, phase_bytes : Int64? = nil,
                             observer : PixelUMMMemoryProbe::Recorder? = nil) : Nil
      (observer || @phase_observer).try do |active|
        active.emit(phase, slot: slot, request: request,
          phase_bytes_kind: phase_bytes_kind, phase_bytes: phase_bytes)
      end
      nil
    end

    private def check_idle! : Nil
      raise ArgumentError.new("PixelUMM serial session released") if @released
      if previous = @last_chain
        raise ArgumentError.new("PixelUMM serial command chain is in flight") unless previous.completed?
      end
    end

    private def decode(bytes : Bytes, offset : Int32, width : Int32) : Float64
      if width == 2
        word = bytes[offset].to_u32 | (bytes[offset + 1].to_u32 << 8)
        (word << 16).unsafe_as(Float32).to_f64
      else
        IO::ByteFormat::LittleEndian.decode(Float32, bytes[offset, 4]).to_f64
      end
    end
  end
end
