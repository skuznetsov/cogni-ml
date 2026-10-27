require "./pixelumm_block_probe"

module PixelUMMTwoBlockProbe
  extend self
  alias Record = PixelUMMBlockProbe::Record
  alias Comparison = PixelUMMBlockProbe::Comparison
  alias Decoder = ML::Video::PixelUMMDecoderBlockMPS
  HF_REVISION   = "c67f2c52b3fbe0193cbc202769bbe5cb7264807b"
  HEADER_SHA256 = "7e7a92cfe84d54c0da3a132099a9472766ae0034042515a0f76b6198bbd09d11"
  MODEL_SHA256  = "bea01a8490f8e9a4076da83d4f9fbb3cba7bd6701df5c28fb2e2c04f00131eec"
  SOURCE_PATH   = "VAE/PixelUMM_FT/diffusion_pytorch_model.safetensors"

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
    raise ArgumentError.new("serial block count must be two, four or eight") unless block_count == 2 || block_count == 4 || block_count == 8
  end

  def fixture_schema(block_count : Int32) : String
    validate_block_count!(block_count)
    label = {2 => "two", 4 => "four", 8 => "eight"}[block_count]
    "cogni_ml.pixelumm_#{label}_block_reference/v1"
  end

  def capture_slot(key : String, block_count : Int32) : Int32
    validate_block_count!(block_count)
    slot = (0...block_count).find { |index| key.starts_with?("block#{index}.") }
    slot || raise ArgumentError.new("capture has unknown block ownership")
  end

  def validate_eight_capture_inventory!(keys : Array(String), grid : Array(Int32)) : Nil
    stages = grid == [2, 32, 32] ? Decoder::Plan::CAPTURE_NAMES.to_a : ["attention", "output"]
    expected = (0...8).flat_map { |slot| stages.map { |stage| "block#{slot}.#{stage}" } }
    unless keys.sort == expected.sort
      raise ArgumentError.new("eight-block fixture requires every expected stage at its admitted grid")
    end
  end

  def validate_input_source!(source : String, slot : Int32) : Nil
    expected = slot == 0 ? "inputs.x" : "captures.block#{slot - 1}.output"
    raise ArgumentError.new("block input must reference its immediate predecessor") unless source == expected
  end

  def validate_capture!(key : String, record : Record, tokens : Int32, block_count : Int32 = 2) : Nil
    capture_slot(key, block_count)
    name = key[7..]
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
      unless max_payload_bytes == 4_i64 * 1024**3 || max_payload_bytes == 8_i64 * 1024**3
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
      # Eight owners plus diagnostic captures exceed the legacy file bound.
      # Weight caches and pair/four fixtures retain their 4 GiB ceiling.
      super(path, trusted_sha256, (@block_count == 8 ? 8_i64 : 4_i64) * 1024**3)
      d = @document
      PixelUMMTwoBlockProbe.validate_identity!(d)
      schema = PixelUMMTwoBlockProbe.fixture_schema(@block_count)
      unless d["schema"].as_s == schema &&
             d["upstream_revision"].as_s == PixelUMMBlockProbe::UPSTREAM_REVISION &&
             d["tile_tokens"].as_i == 8 && d["token_order"].as_s == "tile_major"
        raise ArgumentError.new("serial fixture provenance/layout mismatch")
      end
      @grid = d["grid"].as_a.map(&.as_i)
      unless @grid == [2, 32, 32] || @grid == [6, 44, 80]
        raise ArgumentError.new("serial fixture grid is outside admitted contexts")
      end
      @inputs, @weights, @captures = records("inputs"), records("weights"), records("captures")
      @capture_order = d["capture_order"].as_a.map(&.as_s)
      unless @inputs.keys.sort == ["cos", "sin", "x"] &&
             @capture_order.uniq == @capture_order && @capture_order.sort == @captures.keys.sort &&
             (0...@block_count).all? { |slot| {"attention", "output"}.all? { |name| @captures.has_key?("block#{slot}.#{name}") } }
        raise ArgumentError.new("serial fixture input/capture inventory mismatch")
      end
      tokens = @grid.reduce(1) { |n, v| n * v }
      @captures.each { |key, record| PixelUMMTwoBlockProbe.validate_capture!(key, record, tokens, @block_count) }
      PixelUMMTwoBlockProbe.validate_eight_capture_inventory!(@capture_order, @grid) if @block_count == 8
      unless @inputs["x"].shape == [1, tokens, 4096] && @inputs["x"].dtype == "BF16" &&
             {"cos", "sin"}.all? { |name| @inputs[name].shape == [1, tokens, 1, 128] && @inputs[name].dtype == "F32" }
        raise ArgumentError.new("serial input geometry/dtype mismatch")
      end
      PixelUMMTwoBlockProbe.validate_weights!(@weights, @block_count)
      blocks = d["blocks"].as_a
      raise ArgumentError.new("serial source inventory mismatch") unless blocks.size == @block_count
      @subset_hashes = blocks.map_with_index do |block, index|
        prefix = "block#{index}."
        PixelUMMTwoBlockProbe.validate_input_source!(block["input_source"].as_s, index)
        digest = block["checkpoint_subset_sha256"].as_s
        unless block["index"].as_i == index && block["weight_prefix"].as_s == prefix &&
               block["output_key"].as_s == "captures.block#{index}.output" &&
               block["weight_keys"].as_a.map(&.as_s).sort == @weights.keys.select(&.starts_with?(prefix)).sort &&
               block["capture_keys"].as_a.map(&.as_s).sort == @captures.keys.select(&.starts_with?(prefix)).sort &&
               PixelUMMBlockProbe.valid_digest?(digest) &&
               PixelUMMBlockProbe.valid_digest?(block["source_receipt_sha256"].as_s)
          raise ArgumentError.new("serial source/ownership mismatch")
        end
        digest
      end
      unless @subset_hashes[0] == PixelUMMBlockProbe::SUBSET_SHA256
        raise ArgumentError.new("block zero source identity mismatch")
      end
      validate_ranges!(@inputs.values + @weights.values + @captures.values, contiguous: @block_count == 8)
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
      super(path, trusted_sha256)
      d = @document
      PixelUMMTwoBlockProbe.validate_identity!(d)
      raise ArgumentError.new("unsupported BF16 cache schema") unless d["schema"].as_s == "cogni_ml.pixelumm_bf16_weight_cache/v1"
      unless d["source_commit"].as_s == PixelUMMBlockProbe::UPSTREAM_REVISION &&
             d["stored_dtype"].as_s == "F32" && d["runtime_dtype"].as_s == "BF16" &&
             !d["full_file_hash_verified"].as_bool
        raise ArgumentError.new("BF16 cache stored/runtime source mismatch")
      end
      @weights = records("weights")
      PixelUMMTwoBlockProbe.validate_weights!(@weights, fixture.block_count)
      conversion = d["conversion"]
      unless conversion["method"].as_s == "torch.Tensor.to(torch.bfloat16)" &&
             conversion["device"].as_s == "mps" && conversion["rounding"].as_s == "round_to_nearest_even" &&
             !conversion["torch_version"].as_s.empty?
        raise ArgumentError.new("BF16 cache conversion provenance mismatch")
      end
      unless d["blocks"].as_h.keys.sort == (0...fixture.block_count).map(&.to_s) &&
             d["comparison"]["all_raw_uint16_equal"].as_bool &&
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
    getter workspace : Decoder::Workspace
    getter plans = [] of Decoder::Plan
    @bindings = {} of String => ML::MetalBuffer
    @last_chain : ML::Metal::MPSCommandChain? = nil
    @released = false

    def initialize(@fixture : Fixture, cache : WeightCache, max_named_bytes : Int64, *, @ffn_f32 : Bool = false)
      grid = @fixture.grid
      config = Decoder::Config.new(temporal: grid[0], height: grid[1], width: grid[2])
      weight_bytes = Array.new(@fixture.block_count) do |slot|
        PixelUMMBlockProbe.weight_shapes.keys.to_h { |name| {name, @fixture.weights["block#{slot}.#{name}"].nbytes} }
      end
      @borrowed_bytes = (@fixture.inputs.values + cache.weights.values).sum(&.nbytes)
      @named_bytes = Decoder::Workspace.validate_admission!(config,
        input_bytes: @fixture.inputs["x"].nbytes, cosine_bytes: @fixture.inputs["cos"].nbytes,
        sine_bytes: @fixture.inputs["sin"].nbytes, weights_bytes_by_slot: weight_bytes,
        captures_by_slot: @fixture.captures_by_slot, max_unique_bytes: max_named_bytes)
      # No file-sized read or Metal allocation precedes the aggregate gate.
      @fixture.verify_payload!
      cache.verify_payload!
      @workspace = Decoder::Workspace.new(config, captures_by_slot: @fixture.captures_by_slot,
        max_owned_bytes: max_named_bytes - @borrowed_bytes)
      begin
        @fixture.inputs.each do |name, record|
          @bindings[name] = buffer = ML::MetalBuffer.new(record.nbytes)
          @fixture.upload_into!(record, buffer)
        end
        cache.weights.each do |name, record|
          @bindings[name] = buffer = ML::MetalBuffer.new(record.nbytes)
          cache.upload_into!(record, buffer)
        end
        @fixture.block_count.times do |slot|
          weights = PixelUMMBlockProbe.weight_shapes.keys.to_h { |name| {name, @bindings["block#{slot}.#{name}"]} }
          input = slot == 0 ? @bindings["x"] : @plans[slot - 1].stage("output")
          @plans << Decoder::Plan.new(config, input: input, weights: weights,
            cos: @bindings["cos"], sin: @bindings["sin"], captures: @fixture.captures_by_slot[slot],
            max_named_bytes: max_named_bytes - @borrowed_bytes, workspace: @workspace, slot: slot,
            ffn_f32: @ffn_f32)
        end
        unless @plans.all? { |plan| plan.ffn_f32 == @ffn_f32 }
          raise ArgumentError.new("serial plans must share the selected FFN graph precision")
        end
        # This loader owns distinct allocations for every authenticated weight,
        # the external input/tables and retained outputs. Broader caller-provided
        # Workspace aliases are outside this probe's ownership contract.
        handles = @bindings.values.map(&.handle) + @plans.map { |plan| plan.stage("output").handle }
        unless @named_bytes == @borrowed_bytes + @workspace.named_bytes && handles.uniq.size == handles.size
          raise ArgumentError.new("serial unique-byte/output/weight ownership mismatch")
        end
      rescue error
        @plans.reverse_each(&.release)
        @workspace.release
        @bindings.each_value(&.release)
        raise error
      end
    end

    def encode(chain : ML::Metal::MPSCommandChain) : Nil
      check_idle!
      @last_chain = chain
      @plans.each(&.encode(chain))
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
      @workspace.mutable_shared_buffers.each_value do |buffer|
        # BF16 and F32 interpret all-one words as NaN. Only diagnostic setup
        # touches shared memory on the host, after the previous terminal wait.
        buffer.contents.as(UInt8*).to_slice(buffer.size.to_i32).fill(0xff_u8)
      end
    end

    def compare : Array(Comparison)
      chain = @last_chain || raise ArgumentError.new("PixelUMM serial was not encoded")
      unless chain.completed? && chain.wait_count == 1
        raise ArgumentError.new("PixelUMM serial diagnostics require one terminal wait")
      end
      @fixture.capture_order.map do |key|
        slot = PixelUMMTwoBlockProbe.capture_slot(key, @fixture.block_count)
        name = key[7..]
        record, buffer = @fixture.captures[key], @plans[slot].stage(name)
        raise ArgumentError.new("native capture extent mismatch #{key}") unless record.nbytes == buffer.size
        expected, actual = @fixture.read(record), Bytes.new(record.nbytes.to_i32)
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
        {stage: key, elements: count, failures: failures, different_words: different,
         max_abs: maximum, finite: finite, actual_sha256: Digest::SHA256.hexdigest(actual),
         first_failure_index: first_failure, first_expected: first_expected, first_actual: first_actual}
      end
    end

    def release : Nil
      return if @released
      check_idle!
      @plans.reverse_each(&.release)
      @workspace.release
      @bindings.each_value(&.release)
      @released = true
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
