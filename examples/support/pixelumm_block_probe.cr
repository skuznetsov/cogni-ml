require "json"
require "digest/sha256"
require "../../src/ml/video/pixelumm_decoder_block_mps"

module PixelUMMBlockProbe
  extend self
  ATOL              = 2e-5
  RTOL              = 2e-5
  UPSTREAM_REVISION = "911cea6458a89c2663513ff7099561ec190dd491"
  SUBSET_SHA256     = "ae82bb41c121a955493f5b1502dc512ac8aaba05a6cec67e98edeaedfe720485"
  alias Comparison = NamedTuple(stage: String, elements: Int64, failures: Int64,
    different_words: Int64, max_abs: Float64, finite: Bool, actual_sha256: String,
    first_failure_index: Int64?, first_expected: String?, first_actual: String?)

  def sha_file(path : String) : String
    hash = Digest::SHA256.new
    File.open(path) do |io|
      chunk = Bytes.new(1024 * 1024)
      while (count = io.read(chunk)) > 0
        hash.update(chunk[0, count])
      end
    end
    hash.hexfinal
  end

  def valid_digest?(value : String) : Bool
    value.matches?(/\A[0-9a-f]{64}\z/)
  end

  def weight_shapes : Hash(String, Array(Int32))
    result = {"input_norm.weight" => [4096], "post_attn_norm.weight" => [4096]}
    {"q" => 4096, "k" => 1024, "v" => 1024, "o" => 4096}.each do |name, out_dim|
      result["attn.#{name}_proj.weight"] = [out_dim, 4096]
    end
    {"q", "k"}.each do |name|
      {"t", "hw"}.each { |section| result["attn.#{name}_norm_#{section}.weight"] = [64] }
    end
    result["mlp.gate_proj.weight"] = [12288, 4096]
    result["mlp.up_proj.weight"] = [12288, 4096]
    result["mlp.down_proj.weight"] = [4096, 12288]
    result
  end

  struct Record
    getter dtype : String
    getter shape : Array(Int32)
    getter offset : Int64
    getter nbytes : Int64
    getter sha256 : String

    def initialize(value : JSON::Any, payload_bytes : Int64)
      @dtype = value["dtype"].as_s
      @shape = value["shape"].as_a.map(&.as_i)
      @offset = value["offset"].as_i64
      @nbytes = value["nbytes"].as_i64
      @sha256 = value["sha256"].as_s
      raise ArgumentError.new("invalid fixture record dtype/shape") unless {"BF16", "F32"}.includes?(@dtype) && !@shape.empty? && @shape.all? { |n| n > 0 }
      count = @shape.reduce(1_i64) { |product, n| product * n }
      width = @dtype == "BF16" ? 2 : 4
      unless @offset >= 0 && @nbytes > 0 && @nbytes <= 1024_i64 * 1024 * 1024 &&
             @offset <= payload_bytes && @nbytes <= payload_bytes - @offset &&
             @nbytes == count * width && PixelUMMBlockProbe.valid_digest?(@sha256)
        raise ArgumentError.new("invalid fixture record extent/identity")
      end
    end
  end

  class Fixture
    getter path : String
    getter sha256 : String
    getter payload : String
    getter grid : Array(Int32)
    getter inputs : Hash(String, Record)
    getter weights : Hash(String, Record)
    getter captures : Hash(String, Record)
    getter capture_order : Array(String)

    def initialize(path : String, expected_sha256 : String)
      raise ArgumentError.new("fixture must be a regular absolute file") unless Path[path].absolute? && File.file?(path) && !File.symlink?(path) && File.realpath(path) == path
      @path = path
      raise ArgumentError.new("fixture metadata exceeds bound") if File.size(path) > 1024 * 1024
      raw = File.read(path)
      @sha256 = Digest::SHA256.hexdigest(raw)
      raise ArgumentError.new("fixture trusted identity mismatch") unless PixelUMMBlockProbe.valid_digest?(expected_sha256) && @sha256 == expected_sha256
      document = JSON.parse(raw)
      unless document["schema"].as_s == "cogni_ml.pixelumm_block_reference/v1" &&
             document["upstream_revision"].as_s == UPSTREAM_REVISION &&
             document["checkpoint_subset_sha256"].as_s == SUBSET_SHA256 &&
             document["tile_tokens"].as_i == 8 && document["token_order"].as_s == "tile_major"
        raise ArgumentError.new("fixture provenance/layout mismatch")
      end
      @grid = document["grid"].as_a.map(&.as_i)
      raise ArgumentError.new("fixture grid is outside qualified contexts") unless @grid == [2, 32, 32] || @grid == [6, 44, 80]
      basename = document["payload"].as_s
      raise ArgumentError.new("fixture payload must be a sibling basename") unless basename.matches?(/\A[A-Za-z0-9_.-]+\z/) && basename != "." && basename != ".."
      @payload = File.join(File.dirname(path), basename)
      raise ArgumentError.new("fixture payload is not a regular sibling") unless File.file?(@payload) && !File.symlink?(@payload) && File.realpath(@payload) == @payload
      size = File.size(@payload)
      unless size == document["payload_bytes"].as_i64 && 0 < size <= 4_i64 * 1024 * 1024 * 1024 &&
             PixelUMMBlockProbe.sha_file(@payload) == document["payload_sha256"].as_s
        raise ArgumentError.new("fixture payload extent/hash mismatch")
      end
      @inputs = parse_records(document["inputs"], size)
      @weights = parse_records(document["weights"], size)
      @captures = parse_records(document["captures"], size)
      @capture_order = document["capture_order"].as_a.map(&.as_s)
      unless @inputs.keys.sort == ["cos", "sin", "x"] && @capture_order.uniq == @capture_order &&
             @capture_order.sort == @captures.keys.sort && @captures.has_key?("output")
        raise ArgumentError.new("fixture input/capture inventory mismatch")
      end
      count = @grid.reduce(1) { |product, n| product * n }
      unless @inputs["x"].shape == [1, count, 4096] && @inputs["x"].dtype == "BF16" &&
             {"cos", "sin"}.all? { |name| @inputs[name].shape == [1, count, 1, 128] && @inputs[name].dtype == "F32" }
        raise ArgumentError.new("fixture input geometry/dtype mismatch")
      end
      expected_weights.each do |name, shape|
        record = @weights[name]? || raise ArgumentError.new("missing fixture weight #{name}")
        raise ArgumentError.new("fixture weight shape/dtype mismatch") unless record.shape == shape && record.dtype == "BF16"
      end
      raise ArgumentError.new("unexpected fixture weight") unless @weights.size == 13
      # Every declared range must be disjoint, including capture bytes that are
      # only read after completion. The separately trusted payload hash binds gaps.
      records = @inputs.values + @weights.values + @captures.values
      cursor = 0_i64
      records.sort_by(&.offset).each do |record|
        raise ArgumentError.new("overlapping fixture records") if record.offset < cursor
        cursor = record.offset + record.nbytes
      end
    end

    def read(record : Record) : Bytes
      bytes = Bytes.new(record.nbytes.to_i32)
      File.open(@payload) do |io|
        io.seek(record.offset)
        io.read_fully(bytes)
      end
      raise ArgumentError.new("fixture tensor hash mismatch") unless Digest::SHA256.hexdigest(bytes) == record.sha256
      bytes
    end

    private def parse_records(value : JSON::Any, size : Int64) : Hash(String, Record)
      value.as_h.transform_values { |record| Record.new(record, size) }
    end

    private def expected_weights : Hash(String, Array(Int32))
      PixelUMMBlockProbe.weight_shapes
    end
  end

  class Session
    getter named_bytes : Int64 = 0_i64
    getter borrowed_bytes : Int64 = 0_i64
    getter fixture : Fixture
    @plan : ML::Video::PixelUMMDecoderBlockMPS::Plan? = nil
    @bindings = {} of String => ML::MetalBuffer
    @last_chain : ML::Metal::MPSCommandChain? = nil
    @released = false

    def initialize(@fixture : Fixture, max_named_bytes : Int64)
      begin
        @borrowed_bytes = (@fixture.inputs.values + @fixture.weights.values).sum(&.nbytes)
        raise ArgumentError.new("borrowed bindings exceed named memory budget") unless @borrowed_bytes < max_named_bytes
        grid = @fixture.grid
        config = ML::Video::PixelUMMDecoderBlockMPS::Config.new(temporal: grid[0], height: grid[1], width: grid[2])
        ML::Video::PixelUMMDecoderBlockMPS::Plan.validate_admission!(config,
          input_bytes: @fixture.inputs["x"].nbytes,
          cosine_bytes: @fixture.inputs["cos"].nbytes,
          sine_bytes: @fixture.inputs["sin"].nbytes,
          weights_bytes: @fixture.weights.transform_values(&.nbytes),
          max_named_bytes: max_named_bytes - @borrowed_bytes,
          captures: @fixture.capture_order)
        @fixture.inputs.each { |name, record| @bindings[name] = upload(record) }
        weights = {} of String => ML::MetalBuffer
        @fixture.weights.each { |name, record| weights[name] = @bindings[name] = upload(record) }
        @plan = ML::Video::PixelUMMDecoderBlockMPS::Plan.new(config,
          input: @bindings["x"], weights: weights, cos: @bindings["cos"], sin: @bindings["sin"],
          max_named_bytes: max_named_bytes - @borrowed_bytes, captures: @fixture.capture_order)
        @named_bytes = @borrowed_bytes + plan.named_bytes
        raise ArgumentError.new("prepared PixelUMM bindings exceed named budget") if @named_bytes > max_named_bytes
      rescue error
        @plan.try(&.release)
        @bindings.each_value(&.release)
        raise error
      end
    end

    def plan : ML::Video::PixelUMMDecoderBlockMPS::Plan
      @plan || raise ArgumentError.new("PixelUMM probe plan is unavailable")
    end

    def encode(chain : ML::Metal::MPSCommandChain, *, profile : Bool = false) : Nil
      raise ArgumentError.new("PixelUMM probe session released") if @released
      if previous = @last_chain
        raise ArgumentError.new("previous PixelUMM probe command chain is in flight") unless previous.completed?
      end
      @last_chain = chain
      plan.encode(chain, profile: profile)
    end

    def compare : Array(Comparison)
      chain = @last_chain || raise ArgumentError.new("PixelUMM was not encoded")
      raise ArgumentError.new("PixelUMM diagnostics require terminal wait") unless chain.completed? && chain.wait_count == 1
      @fixture.capture_order.map do |name|
        record = @fixture.captures[name]
        buffer = plan.stages[name]
        raise ArgumentError.new("native capture extent mismatch #{name}") unless buffer.size == record.nbytes
        expected = @fixture.read(record)
        actual = Bytes.new(record.nbytes.to_i32)
        buffer.read_bytes(actual.to_unsafe, actual.size)
        width = record.dtype == "BF16" ? 2 : 4
        count = record.nbytes // width
        failures = 0_i64
        different = 0_i64
        maximum = 0.0
        finite = true
        first_failure = nil.as(Int64?)
        first_expected = nil.as(String?)
        first_actual = nil.as(String?)
        count.times do |index|
          offset = (index * width).to_i32
          a = decode(expected, offset, width)
          b = decode(actual, offset, width)
          different += 1 unless expected[offset, width] == actual[offset, width]
          if !a.finite? || !b.finite?
            finite = false
            failures += 1
            maximum = Float64::MAX
          else
            error = (a - b).abs
            maximum = Math.max(maximum, error)
            failures += 1 if error > ATOL + RTOL * a.abs
          end
          if first_failure.nil? && (!a.finite? || !b.finite? || (a - b).abs > ATOL + RTOL * a.abs)
            first_failure, first_expected, first_actual = index, a.to_s, b.to_s
          end
        end
        {stage: name, elements: count, failures: failures, different_words: different,
         max_abs: maximum, finite: finite, actual_sha256: Digest::SHA256.hexdigest(actual),
         first_failure_index: first_failure, first_expected: first_expected, first_actual: first_actual}
      end
    end

    def release : Nil
      return if @released
      if chain = @last_chain
        raise ArgumentError.new("PixelUMM probe command chain is in flight") unless chain.completed?
      end
      plan.release
      @bindings.each_value(&.release)
      @released = true
    end

    private def upload(record : Record) : ML::MetalBuffer
      bytes = @fixture.read(record)
      buffer = ML::MetalBuffer.new(record.nbytes)
      buffer.write_bytes(bytes.to_unsafe, bytes.size)
      buffer
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
