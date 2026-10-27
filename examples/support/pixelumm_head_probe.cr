require "./pixelumm_block_probe"
require "./pixelumm_memory_probe"
require "../../src/ml/video/pixelumm_decoder_head_mps"

# The saved input is the authenticated tile-major block-35 output. This probe
# qualifies only the global head; it neither runs nor budgets the 36-block chain.
module PixelUMMHeadProbe
  extend self
  ATOL         = PixelUMMBlockProbe::ATOL
  RTOL         = PixelUMMBlockProbe::RTOL
  MODEL_SHA256 = "bea01a8490f8e9a4076da83d4f9fbb3cba7bd6701df5c28fb2e2c04f00131eec"
  CAPTURES     = %w(final_norm_tile_major restored_thw head_norm_thw linear_thw output_thw)
  NAMED_LIMIT  = 2_i64 * 1024 * 1024 * 1024
  alias Record = PixelUMMBlockProbe::Record
  alias Comparison = PixelUMMBlockProbe::Comparison

  def weight_shapes : Hash(String, Array(Int32))
    {"final_norm_weight" => [4096], "head_norm_weight" => [4096],
     "head_linear_weight" => [3072, 4096], "head_linear_bias" => [3072]}
  end

  def capture_shapes(grid : Array(Int32)) : Hash(String, Array(Int32))
    tokens = grid.reduce(1) { |a, b| a * b }
    {"final_norm_tile_major" => [1, tokens, 4096], "restored_thw" => [1, tokens, 4096],
     "head_norm_thw" => [1, tokens, 4096], "linear_thw" => [1, tokens, 3072],
     "output_thw" => [1, 3, grid[0] * 4, grid[1] * 16, grid[2] * 16]}
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
      unless Path[path].absolute? && File.file?(path) && !File.symlink?(path) && File.realpath(path) == path
        raise ArgumentError.new("head fixture must be a regular canonical absolute file")
      end
      @path = path
      raise ArgumentError.new("head fixture metadata exceeds bound") if File.size(path) > 1024 * 1024
      raw = File.read(path)
      @sha256 = Digest::SHA256.hexdigest(raw)
      unless PixelUMMBlockProbe.valid_digest?(expected_sha256) && @sha256 == expected_sha256
        raise ArgumentError.new("head fixture trusted metadata identity mismatch")
      end
      document = JSON.parse(raw)
      unless document["schema"].as_s == "cogni_ml.pixelumm_head_reference/v1" &&
             document["upstream_revision"].as_s == PixelUMMBlockProbe::UPSTREAM_REVISION &&
             document["upstream_model_sha256"].as_s == MODEL_SHA256 &&
             document["tile_tokens"].as_i == 8 && document["token_order"].as_s == "tile_major" &&
             document["output_layout"].as_s == "BCTHW" &&
             document["comparison"]["atol"].as_f == ATOL && document["comparison"]["rtol"].as_f == RTOL
        raise ArgumentError.new("head fixture provenance/layout/tolerance mismatch")
      end
      @grid = document["grid"].as_a.map(&.as_i)
      raise ArgumentError.new("head fixture requires the full qualified J") unless @grid == [6, 44, 80]
      basename = document["payload"].as_s
      unless basename.matches?(/\A[A-Za-z0-9_.-]+\z/) && basename != "." && basename != ".."
        raise ArgumentError.new("head fixture payload must be a sibling basename")
      end
      @payload = File.join(File.dirname(path), basename)
      unless File.file?(@payload) && !File.symlink?(@payload) && File.realpath(@payload) == @payload
        raise ArgumentError.new("head fixture payload must be a regular canonical sibling")
      end
      size = File.size(@payload)
      unless size == document["payload_bytes"].as_i64 && 0 < size <= 2_i64 * 1024 * 1024 * 1024 &&
             PixelUMMBlockProbe.valid_digest?(document["payload_sha256"].as_s) &&
             PixelUMMBlockProbe.sha_file(@payload) == document["payload_sha256"].as_s
        raise ArgumentError.new("head fixture payload extent/hash mismatch")
      end
      @inputs = parse_records(document["inputs"], size)
      @weights = parse_records(document["weights"], size)
      @captures = parse_records(document["captures"], size)
      @capture_order = document["capture_order"].as_a.map(&.as_s)
      unless @inputs.keys == ["block35_output"] && @capture_order == CAPTURES && @captures.keys.sort == CAPTURES.sort &&
             @weights.keys.sort == PixelUMMHeadProbe.weight_shapes.keys.sort
        raise ArgumentError.new("head fixture record inventory/order mismatch")
      end
      validate_shape(@inputs["block35_output"], [1, 21120, 4096])
      PixelUMMHeadProbe.weight_shapes.each { |name, shape| validate_shape(@weights[name], shape) }
      PixelUMMHeadProbe.capture_shapes(@grid).each { |name, shape| validate_shape(@captures[name], shape) }
      cursor = 0_i64
      (@inputs.values + @weights.values + @captures.values).sort_by(&.offset).each do |record|
        raise ArgumentError.new("head fixture records must be contiguous and disjoint") unless record.offset == cursor
        cursor += record.nbytes
      end
      raise ArgumentError.new("head fixture has unaccounted payload bytes") unless cursor == size
    end

    def read(record : Record) : Bytes
      bytes = Bytes.new(record.nbytes.to_i32)
      File.open(@payload) do |io|
        io.seek(record.offset)
        io.read_fully(bytes)
      end
      unless Digest::SHA256.hexdigest(bytes) == record.sha256
        raise ArgumentError.new("head fixture tensor hash mismatch")
      end
      bytes
    end

    def compatible_replay!(other : Fixture) : Nil
      unless @grid == other.grid && @capture_order == other.capture_order &&
             @weights.all? { |name, record| record.sha256 == other.weights[name].sha256 } &&
             @inputs["block35_output"].sha256 != other.inputs["block35_output"].sha256 &&
             @captures["output_thw"].sha256 != other.captures["output_thw"].sha256
        raise ArgumentError.new("head replay must keep weights and geometry with distinguishable input/output")
      end
    end

    private def parse_records(value : JSON::Any, size : Int64) : Hash(String, Record)
      value.as_h.transform_values { |record| Record.new(record, size) }
    end

    private def validate_shape(record : Record, shape : Array(Int32)) : Nil
      raise ArgumentError.new("head fixture tensor shape/dtype mismatch") unless record.dtype == "BF16" && record.shape == shape
    end
  end

  # This comparator is also exercised on deliberately wrong permutations and
  # non-finite words by the CPU-only falsifier spec.
  def compare_bytes(name : String, expected : Bytes, actual : Bytes) : Comparison
    unless expected.size == actual.size && expected.size > 0 && expected.size.even?
      raise ArgumentError.new("head comparison byte extents differ")
    end
    count = expected.size.to_i64 // 2
    failures, different = 0_i64, 0_i64
    maximum = 0.0
    finite = true
    first_failure = nil.as(Int64?)
    first_expected = nil.as(String?)
    first_actual = nil.as(String?)
    count.times do |index|
      offset = (index * 2).to_i32
      a = decode(expected, offset)
      b = decode(actual, offset)
      different += 1 unless expected[offset, 2] == actual[offset, 2]
      failed = !a.finite? || !b.finite?
      if failed
        finite = false
        maximum = Float64::MAX
      else
        error = (a - b).abs
        maximum = Math.max(maximum, error)
        failed = error > ATOL + RTOL * a.abs
      end
      if failed
        failures += 1
        if first_failure.nil?
          first_failure, first_expected, first_actual = index, a.to_s, b.to_s
        end
      end
    end
    {stage: name, elements: count, failures: failures, different_words: different,
     max_abs: maximum, finite: finite, actual_sha256: Digest::SHA256.hexdigest(actual),
     first_failure_index: first_failure, first_expected: first_expected, first_actual: first_actual}
  end

  private def decode(bytes : Bytes, offset : Int32) : Float64
    word = bytes[offset].to_u32 | (bytes[offset + 1].to_u32 << 8)
    (word << 16).unsafe_as(Float32).to_f64
  end

  class Session
    getter fixture : Fixture
    getter named_bytes : Int64 = 0_i64
    getter borrowed_bytes : Int64 = 0_i64
    getter aliased_weights_rejected = false
    @plan : ML::Video::PixelUMMDecoderHeadMPS::Plan? = nil
    @bindings = {} of String => ML::MetalBuffer
    @last_chain : ML::Metal::MPSCommandChain? = nil
    @released = false

    def initialize(@fixture : Fixture, max_named_bytes : Int64)
      begin
        @borrowed_bytes = (@fixture.inputs.values + @fixture.weights.values).sum(&.nbytes)
        raise ArgumentError.new("head bindings exceed aggregate named budget") unless @borrowed_bytes < max_named_bytes <= NAMED_LIMIT
        grid = @fixture.grid
        config = ML::Video::PixelUMMDecoderHeadMPS::Config.new(temporal: grid[0], height: grid[1], width: grid[2])
        ML::Video::PixelUMMDecoderHeadMPS::Plan.validate_admission!(config,
          input_bytes: @fixture.inputs["block35_output"].nbytes,
          weights_bytes: @fixture.weights.transform_values(&.nbytes),
          max_named_bytes: max_named_bytes - @borrowed_bytes, captures: @fixture.capture_order)
        @bindings["block35_output"] = upload(@fixture.inputs["block35_output"])
        weights = {} of String => ML::MetalBuffer
        @fixture.weights.each { |name, record| weights[name] = @bindings[name] = upload(record) }
        aliased = weights.dup
        aliased["head_norm_weight"] = weights["final_norm_weight"]
        begin
          invalid = ML::Video::PixelUMMDecoderHeadMPS::Plan.new(config,
            input: @bindings["block35_output"], weights: aliased,
            max_named_bytes: max_named_bytes - @borrowed_bytes, captures: @fixture.capture_order)
          invalid.release
          raise "head accepted aliased borrowed weights"
        rescue error : ArgumentError
          raise error unless (error.message || "").includes?("alias")
          @aliased_weights_rejected = true
        end
        @plan = ML::Video::PixelUMMDecoderHeadMPS::Plan.new(config,
          input: @bindings["block35_output"], weights: weights,
          max_named_bytes: max_named_bytes - @borrowed_bytes, captures: @fixture.capture_order)
        @named_bytes = @borrowed_bytes + plan.named_bytes
        raise ArgumentError.new("head aggregate named budget exceeded") if @named_bytes > max_named_bytes
      rescue error
        @plan.try(&.release)
        @bindings.each_value(&.release)
        raise error
      end
    end

    def plan : ML::Video::PixelUMMDecoderHeadMPS::Plan
      @plan || raise ArgumentError.new("head plan unavailable")
    end

    def encode(chain : ML::Metal::MPSCommandChain) : Nil
      check_completed!
      @last_chain = chain
      plan.encode(chain)
    end

    def replay_input!(other : Fixture) : Nil
      check_completed!
      @fixture.compatible_replay!(other)
      bytes = other.read(other.inputs["block35_output"])
      @bindings["block35_output"].write_bytes(bytes.to_unsafe, bytes.size)
      @fixture = other
    end

    def compare : Array(Comparison)
      raise ArgumentError.new("head probe session released") if @released
      chain = @last_chain || raise ArgumentError.new("head was not encoded")
      raise ArgumentError.new("head diagnostics require terminal wait") unless chain.completed? && chain.wait_count == 1
      @fixture.capture_order.map do |name|
        record = @fixture.captures[name]
        buffer = plan.stage(name)
        raise ArgumentError.new("head capture extent mismatch") unless buffer.size == record.nbytes
        expected = @fixture.read(record)
        actual = Bytes.new(record.nbytes.to_i32)
        buffer.read_bytes(actual.to_unsafe, actual.size)
        result = PixelUMMHeadProbe.compare_bytes(name, expected, actual)
        # Diagnostics are outside inference timing; avoid retaining full-J byte
        # arrays from prior captures in the conservative GC heap.
        expected = Bytes.empty
        actual = Bytes.empty
        GC.collect
        result
      end
    end

    def release : Nil
      return if @released
      check_completed!
      plan.release
      @bindings.each_value(&.release)
      @released = true
    end

    private def check_completed! : Nil
      raise ArgumentError.new("head probe session released") if @released
      if chain = @last_chain
        raise ArgumentError.new("head probe command chain is in flight") unless chain.completed?
      end
    end

    private def upload(record : Record) : ML::MetalBuffer
      bytes = @fixture.read(record)
      buffer = ML::MetalBuffer.new(record.nbytes)
      buffer.write_bytes(bytes.to_unsafe, bytes.size)
      buffer
    end
  end
end
