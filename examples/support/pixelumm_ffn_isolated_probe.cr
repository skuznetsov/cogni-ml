require "json"
require "digest/sha256"
require "../../src/ml/video/pixelumm_decoder_block_mps"
require "./pixelumm_memory_probe"
require "./hy_nomas_probe_instrumentation"

module PixelUMMFFNIsolatedProbe
  extend self

  SCHEMA            = "cogni_ml.pixelumm_ffn_isolated_reference/v1"
  UPSTREAM_REVISION = "911cea6458a89c2663513ff7099561ec190dd491"
  MODEL_SHA256      = "bea01a8490f8e9a4076da83d4f9fbb3cba7bd6701df5c28fb2e2c04f00131eec"
  SUBSET_SHA256     = "ae82bb41c121a955493f5b1502dc512ac8aaba05a6cec67e98edeaedfe720485"
  ATOL              = 2e-5_f64
  RTOL              = 2e-5_f64
  HIDDEN            =     4096
  FFN               =    12288
  GIB               = 1024_i64 * 1024 * 1024
  MAX_NAMED_BYTES   = 4_i64 * GIB
  METRIC_SLOTS      = 25
  SOURCE            = {{ read_file("#{__DIR__}/pixelumm_ffn_dense.metal") }}

  INPUT_NAMES         = ["post_attn_norm", "product"]
  WEIGHT_NAMES        = ["mlp.gate_proj.weight", "mlp.up_proj.weight", "mlp.down_proj.weight"]
  CAPTURE_NAMES       = ["gate_proj", "up_proj", "down_proj"]
  PLAN_NAMES          = ["separate_gate", "separate_up", "shared_gate_up", "down"]
  SOURCE_RECORD_NAMES = [
    "inputs.x",
    "captures.block0.attention",
    "weights.block0.attn.o_proj.weight",
    "weights.block0.post_attn_norm.weight",
    "weights.block0.mlp.gate_proj.weight",
    "weights.block0.mlp.up_proj.weight",
    "weights.block0.mlp.down_proj.weight",
  ]
  SOURCE_PARENT_RECORDS = [
    "x", "attention", "attn.o_proj.weight", "post_attn_norm.weight",
    "mlp.gate_proj.weight", "mlp.up_proj.weight", "mlp.down_proj.weight",
  ]

  alias Metric = NamedTuple(failures: UInt32, different_words: UInt32, nonfinite: UInt32, elements: UInt32)
  alias ScheduleEntry = NamedTuple(index: Int32, mode: String, warmup: Bool)
  alias StageTiming = NamedTuple(stage: String, first_root: Int32, end_root: Int32, encode_wall_seconds: Float64)

  # A command may have been partially enqueued even when its API returns an
  # error. Once any encoder/blit is attempted, native objects stay owned until
  # the one successful terminal wait has completed.
  class SubmissionLifecycle
    @attempted = false
    @terminal_wait_succeeded = false

    def mark_attempt! : Nil
      raise ArgumentError.new("cannot submit after terminal completion") if @terminal_wait_succeeded
      @attempted = true
    end

    def mark_terminal_wait!(completed : Bool, wait_count : Int32) : Nil
      unless completed && wait_count == 1
        raise ArgumentError.new("isolated FFN terminal release requires one successful wait")
      end
      raise ArgumentError.new("terminal wait was already recorded") if @terminal_wait_succeeded
      @terminal_wait_succeeded = true
    end

    def attempted? : Bool
      @attempted
    end

    def terminal_wait_succeeded? : Bool
      @terminal_wait_succeeded
    end

    def release_allowed?(completed : Bool, wait_count : Int32) : Bool
      !@attempted || (@terminal_wait_succeeded && completed && wait_count == 1)
    end
  end

  struct TensorRecord
    getter name : String
    getter dtype : String
    getter shape : Array(Int32)
    getter offset : Int64
    getter nbytes : Int64
    getter sha256 : String

    def initialize(@name : String, value : JSON::Any)
      @dtype = value["dtype"].as_s
      @shape = value["shape"].as_a.map(&.as_i)
      @offset = value["offset"].as_i64
      @nbytes = value["nbytes"].as_i64
      @sha256 = value["sha256"].as_s
    end

    def validate!(expected_shape : Array(Int32), payload_bytes : Int64) : Nil
      unless @dtype == "BF16" && @shape == expected_shape && @shape.all? { |dim| dim > 0 } &&
             @offset >= 0 && @nbytes == elements.to_i64 * 2 && @nbytes > 0 &&
             @offset <= payload_bytes && @nbytes <= payload_bytes - @offset &&
             PixelUMMFFNIsolatedProbe.valid_digest?(@sha256)
        raise ArgumentError.new("isolated FFN record geometry/extent mismatch: #{@name}")
      end
    end

    def elements : Int64
      @shape.reduce(1_i64) { |product, dim| product * dim }
    end
  end

  struct Metadata
    getter grid : Array(Int32)
    getter tokens : Int32
    getter payload_bytes : Int64
    getter payload_sha256 : String
    getter inputs : Hash(String, TensorRecord)
    getter weights : Hash(String, TensorRecord)
    getter captures : Hash(String, TensorRecord)
    getter source_replay : JSON::Any

    def initialize(document : JSON::Any)
      unless document["schema"].as_s == SCHEMA &&
             document["upstream_revision"].as_s == UPSTREAM_REVISION &&
             document["upstream_model_sha256"].as_s == MODEL_SHA256 &&
             document["model_sha256"].as_s == MODEL_SHA256 &&
             document["comparison"]["atol"].as_f == ATOL &&
             document["comparison"]["rtol"].as_f == RTOL &&
             document["token_order"].as_s == "tile_major" &&
             document["tile_tokens"].as_i == 8 &&
             document["runtime_dtype"].as_s == "BF16"
        raise ArgumentError.new("isolated FFN fixture provenance/layout/tolerance mismatch")
      end

      @grid = document["grid"].as_a.map(&.as_i)
      @tokens = @grid.reduce(1) { |product, dim| product * dim }
      unless @grid == [2, 32, 32] || @grid == [6, 44, 80]
        raise ArgumentError.new("isolated FFN fixture grid is outside qualified contexts")
      end
      @payload_bytes = document["payload_bytes"].as_i64
      @payload_sha256 = document["payload_sha256"].as_s
      unless 0 < @payload_bytes <= MAX_NAMED_BYTES && PixelUMMFFNIsolatedProbe.valid_digest?(@payload_sha256)
        raise ArgumentError.new("isolated FFN payload identity/extent mismatch")
      end

      @inputs = parse_group(document["inputs"], "inputs")
      @weights = parse_group(document["weights"], "weights")
      @captures = parse_group(document["captures"], "captures")
      unless @inputs.keys.sort == INPUT_NAMES.sort && @weights.keys.sort == WEIGHT_NAMES.sort &&
             @captures.keys.sort == CAPTURE_NAMES.sort &&
             document["capture_order"].as_a.map(&.as_s) == CAPTURE_NAMES
        raise ArgumentError.new("isolated FFN fixture inventory mismatch")
      end

      @inputs["post_attn_norm"].validate!([1, @tokens, HIDDEN], @payload_bytes)
      @inputs["product"].validate!([1, @tokens, FFN], @payload_bytes)
      @weights["mlp.gate_proj.weight"].validate!([FFN, HIDDEN], @payload_bytes)
      @weights["mlp.up_proj.weight"].validate!([FFN, HIDDEN], @payload_bytes)
      @weights["mlp.down_proj.weight"].validate!([HIDDEN, FFN], @payload_bytes)
      @captures["gate_proj"].validate!([1, @tokens, FFN], @payload_bytes)
      @captures["up_proj"].validate!([1, @tokens, FFN], @payload_bytes)
      @captures["down_proj"].validate!([1, @tokens, HIDDEN], @payload_bytes)

      all_records = @inputs.values + @weights.values + @captures.values
      cursor = 0_i64
      all_records.sort_by(&.offset).each do |record|
        raise ArgumentError.new("isolated FFN records have a gap or overlap") unless record.offset == cursor
        cursor = record.offset + record.nbytes
      end
      raise ArgumentError.new("isolated FFN records do not exactly cover payload") unless cursor == @payload_bytes

      @source_replay = document["source_replay"]
      validate_source_replay!(@source_replay, @tokens)
    end

    private def parse_group(value : JSON::Any, group : String) : Hash(String, TensorRecord)
      records = {} of String => TensorRecord
      value.as_h.each do |name, record|
        records[name] = TensorRecord.new("#{group}.#{name}", record)
      end
      records
    end

    private def validate_source_replay!(source : JSON::Any, tokens : Int32) : Nil
      parent_schema = tokens == 2048 ? "cogni_ml.pixelumm_block_reference/v1" : "cogni_ml.pixelumm_thirty_six_block_reference/v1"
      metadata_sha = source["source_metadata_sha256"].as_s
      payload_sha = source["source_payload_sha256"].as_s
      parent_payload_bytes = source["parent_payload_bytes"].as_i64
      receipt = source["source_receipt_sha256"]?
      unless source["kind"].as_s == "saved_block0_attention_upstream_suffix_mps" &&
             source["parent_schema"].as_s == parent_schema &&
             source["parent_manifest_sha256"].as_s == metadata_sha &&
             source["parent_payload_sha256"].as_s == payload_sha &&
             PixelUMMFFNIsolatedProbe.valid_digest?(metadata_sha) &&
             PixelUMMFFNIsolatedProbe.valid_digest?(payload_sha) &&
             parent_payload_bytes > 0 && parent_payload_bytes <= 32_i64 * GIB &&
             source["selected_record_hashes_verified"].as_bool &&
             !source["parent_payload_full_hash_recomputed"].as_bool &&
             !source["cpu_fallback"].as_bool &&
             source["block0_subset_sha256"].as_s == SUBSET_SHA256
        raise ArgumentError.new("isolated FFN source replay binding is malformed")
      end
      if tokens == 21120
        unless receipt && PixelUMMFFNIsolatedProbe.valid_digest?(receipt.as_s) &&
               source["parent_block0"]["source_receipt_sha256"].as_s == receipt.as_s &&
               source["parent_input_provenance"]["kind"].as_s == "real_latent_xt_pixelumm_stem"
          raise ArgumentError.new("isolated FFN full-J source receipt/input provenance mismatch")
        end
      elsif receipt && !receipt.raw.nil?
        raise ArgumentError.new("small-J source replay must not claim a full36 receipt")
      end

      source_records = source["source_records"].as_h
      parent_records = source["selected_parent_records"].as_h
      unless source_records.keys.sort == SOURCE_RECORD_NAMES.sort &&
             parent_records.keys.sort == SOURCE_PARENT_RECORDS.sort
        raise ArgumentError.new("isolated FFN source record inventory mismatch")
      end
      source_shapes = {
        "inputs.x"                             => [1, tokens, HIDDEN],
        "captures.block0.attention"            => [1, tokens, 32, 128],
        "weights.block0.attn.o_proj.weight"    => [HIDDEN, HIDDEN],
        "weights.block0.post_attn_norm.weight" => [HIDDEN],
        "weights.block0.mlp.gate_proj.weight"  => [FFN, HIDDEN],
        "weights.block0.mlp.up_proj.weight"    => [FFN, HIDDEN],
        "weights.block0.mlp.down_proj.weight"  => [HIDDEN, FFN],
      }
      selected_names = {
        "inputs.x"                             => "x",
        "captures.block0.attention"            => "attention",
        "weights.block0.attn.o_proj.weight"    => "attn.o_proj.weight",
        "weights.block0.post_attn_norm.weight" => "post_attn_norm.weight",
        "weights.block0.mlp.gate_proj.weight"  => "mlp.gate_proj.weight",
        "weights.block0.mlp.up_proj.weight"    => "mlp.up_proj.weight",
        "weights.block0.mlp.down_proj.weight"  => "mlp.down_proj.weight",
      }
      source_ranges = [] of Tuple(Int64, Int64)
      source_shapes.each do |key, shape|
        record = source_records[key]
        parent_record = parent_records[selected_names[key]]
        validate_source_record!(record, shape, parent_payload_bytes, key)
        unless record["dtype"] == parent_record["dtype"] &&
               record["shape"] == parent_record["shape"] &&
               record["offset"] == parent_record["offset"] &&
               record["nbytes"] == parent_record["nbytes"] &&
               record["sha256"] == parent_record["sha256"]
          raise ArgumentError.new("isolated FFN source replay record alias mismatch: #{key}")
        end
        start = record["offset"].as_i64
        source_ranges << {start, start + record["nbytes"].as_i64}
      end
      ordered = source_ranges.sort_by(&.[0])
      ordered.each_cons(2) do |pair|
        raise ArgumentError.new("isolated FFN source records overlap") if pair[0][1] > pair[1][0]
      end
    end

    private def validate_source_record!(record : JSON::Any, shape : Array(Int32),
                                        payload_bytes : Int64, name : String) : Nil
      actual_shape = record["shape"].as_a.map(&.as_i)
      offset = record["offset"].as_i64
      nbytes = record["nbytes"].as_i64
      expected_bytes = shape.reduce(1_i64) { |product, dim| product * dim } * 2
      unless record["dtype"].as_s == "BF16" && actual_shape == shape &&
             nbytes == expected_bytes && offset >= 0 && offset <= payload_bytes &&
             nbytes <= payload_bytes - offset &&
             PixelUMMFFNIsolatedProbe.valid_digest?(record["sha256"].as_s)
        raise ArgumentError.new("isolated FFN source record malformed: #{name}")
      end
    end
  end

  class Fixture
    getter path : String
    getter sha256 : String
    getter payload : String
    getter metadata : Metadata
    getter document : JSON::Any

    def initialize(path : String, expected_sha256 : String)
      unless Path[path].absolute? && File.file?(path) && !File.symlink?(path) && File.realpath(path) == path
        raise ArgumentError.new("isolated FFN metadata must be a regular canonical absolute file")
      end
      raise ArgumentError.new("isolated FFN metadata exceeds bound") if File.size(path) > 1024 * 1024
      @path = path
      raw = File.read(path)
      @sha256 = Digest::SHA256.hexdigest(raw)
      unless PixelUMMFFNIsolatedProbe.valid_digest?(expected_sha256) && @sha256 == expected_sha256
        raise ArgumentError.new("isolated FFN trusted metadata SHA256 mismatch")
      end
      @document = JSON.parse(raw)
      @metadata = Metadata.new(@document)
      basename = @document["payload"].as_s
      unless basename.matches?(/\A[A-Za-z0-9_.-]+\z/) && basename != "." && basename != ".."
        raise ArgumentError.new("isolated FFN payload must be a sibling basename")
      end
      @payload = File.join(File.dirname(path), basename)
      unless File.file?(@payload) && !File.symlink?(@payload) && File.realpath(@payload) == @payload
        raise ArgumentError.new("isolated FFN payload must be a regular canonical sibling")
      end
      unless File.size(@payload) == @metadata.payload_bytes &&
             PixelUMMFFNIsolatedProbe.sha_file(@payload) == @metadata.payload_sha256
        raise ArgumentError.new("isolated FFN payload SHA256/extent mismatch")
      end
    end

    def read_into(record : TensorRecord, buffer : ML::MetalBuffer) : Nil
      slice = Slice.new(buffer.contents.as(UInt8*), record.nbytes.to_i32)
      File.open(@payload) do |io|
        io.seek(record.offset)
        io.read_fully(slice)
      end
      unless Digest::SHA256.hexdigest(slice) == record.sha256
        raise ArgumentError.new("isolated FFN tensor content changed: #{record.name}")
      end
    end
  end

  def valid_digest?(value : String) : Bool
    value.matches?(/\A[0-9a-f]{64}\z/)
  end

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

  def resource_bytes(tokens : Int32) : Hash(String, Int64)
    raise ArgumentError.new("unsupported isolated FFN row count") unless {2048, 21120}.includes?(tokens)
    hidden = tokens.to_i64 * HIDDEN * 2
    ffn = tokens.to_i64 * FFN * 2
    weight = FFN.to_i64 * HIDDEN * 2
    metrics = METRIC_SLOTS.to_i64 * 4 * sizeof(UInt32)
    {
      "input.post_attn_norm" => hidden,
      "input.product"        => ffn,
      "weight.gate"          => weight,
      "weight.up"            => weight,
      "weight.down"          => weight,
      "expected.gate"        => ffn,
      "expected.up"          => ffn,
      "expected.down"        => hidden,
      "candidate.gate"       => ffn,
      "candidate.up"         => ffn,
      "candidate.down"       => hidden,
      "down.canonical"       => hidden,
      "metrics"              => metrics,
    }
  end

  def named_bytes(tokens : Int32) : Int64
    resource_bytes(tokens).values.sum(0_i64)
  end

  def validate_budget!(tokens : Int32, maximum : Int64) : Int64
    unless maximum > 0 && maximum <= MAX_NAMED_BYTES
      raise ArgumentError.new("isolated FFN named-memory cap must be between one byte and 4 GiB")
    end
    required = named_bytes(tokens)
    raise ArgumentError.new("isolated FFN named buffers exceed cap before allocation") if required > maximum
    required
  end

  def plan_order(reverse_warmups : Bool) : Array(String)
    reverse_warmups ? PLAN_NAMES.reverse : PLAN_NAMES.dup
  end

  def schedule(reverse_warmups : Bool, reverse_first : Bool) : Array(ScheduleEntry)
    warmups = reverse_warmups ? ["shared", "separate"] : ["separate", "shared"]
    measured = reverse_first ? ["shared", "separate", "separate", "shared"] : ["separate", "shared", "shared", "separate"]
    (warmups.map_with_index { |mode, index| {index: index.to_i32, mode: mode, warmup: true} } +
      measured.map_with_index { |mode, index| {index: (index + 2).to_i32, mode: mode, warmup: false} })
  end

  def stage_order(down_first : Bool) : Array(String)
    down_first ? ["down", "gate_up"] : ["gate_up", "down"]
  end

  def median(values : Array(Float64)) : Float64
    raise ArgumentError.new("empty isolated FFN timing set") if values.empty?
    sorted = values.sort
    (sorted[(sorted.size - 1) // 2] + sorted[sorted.size // 2]) / 2
  end

  def compare(expected : Bytes, actual : Bytes) : Metric
    unless expected.size == actual.size && expected.size > 0 && expected.size.even?
      raise ArgumentError.new("invalid isolated FFN BF16 comparison extent")
    end
    failures = 0_u32
    different = 0_u32
    nonfinite = 0_u32
    (expected.size // 2).times do |i|
      expected_word = IO::ByteFormat::LittleEndian.decode(UInt16, expected[i * 2, 2])
      actual_word = IO::ByteFormat::LittleEndian.decode(UInt16, actual[i * 2, 2])
      expected_value = (expected_word.to_u32 << 16).unsafe_as(Float32)
      actual_value = (actual_word.to_u32 << 16).unsafe_as(Float32)
      different += 1 if expected_word != actual_word
      if !expected_value.finite? || !actual_value.finite?
        nonfinite += 1
        failures += 1
      elsif (expected_value - actual_value).abs > 2e-5_f32 + 2e-5_f32 * expected_value.abs
        failures += 1
      end
    end
    {failures: failures, different_words: different, nonfinite: nonfinite,
     elements: (expected.size // 2).to_u32}
  end

  def count_word_differences(expected : Bytes, actual : Bytes) : UInt32
    unless expected.size == actual.size && expected.size > 0 && expected.size.even?
      raise ArgumentError.new("invalid isolated FFN canonical word extent")
    end
    different = 0_u32
    (expected.size // 2).times do |index|
      left = IO::ByteFormat::LittleEndian.decode(UInt16, expected[index * 2, 2])
      right = IO::ByteFormat::LittleEndian.decode(UInt16, actual[index * 2, 2])
      different += 1 if left != right
    end
    different
  end

  {% unless flag?(:cpu_only) %}
    lib PixelUMMFFNIsolatedFFI
      fun step_create = gs_pixelumm_ffn_isolated_step_create(
        device : Void*, norm_input : Void*, product_input : Void*, gate_weight : Void*,
        up_weight : Void*, down_weight : Void*, gate_output : Void*, up_output : Void*,
        down_output : Void*, tokens : Int32, kind : Int32, error : UInt8*, capacity : LibC::SizeT,
      ) : Void*
      fun step_encode = gs_pixelumm_ffn_isolated_step_encode(
        plan : Void*, command : Void*, error : UInt8*, capacity : LibC::SizeT,
      ) : Int32
      fun step_release = gs_pixelumm_ffn_isolated_step_release(plan : Void*) : Void
    end

    class Owner
      getter named_bytes : Int64
      getter resources : Hash(String, Int64)
      getter plan_order : Array(String)
      getter memory : PixelUMMMemoryProbe::Recorder
      getter lifecycle : SubmissionLifecycle
      @buffers = {} of String => ML::MetalBuffer
      @plans = {} of String => Pointer(Void)
      @pipeline : ML::Metal::ComputePipeline? = nil
      @plan_order = [] of String
      @released = false
      @canonical_ready = false
      @tokens : Int32

      def initialize(@fixture : Fixture, maximum : Int64, reverse_warmups : Bool,
                     @memory : PixelUMMMemoryProbe::Recorder)
        @tokens = @fixture.metadata.tokens
        @lifecycle = SubmissionLifecycle.new
        @resources = PixelUMMFFNIsolatedProbe.resource_bytes(@tokens)
        @named_bytes = PixelUMMFFNIsolatedProbe.validate_budget!(@tokens, maximum)
        raise "Metal unavailable" unless ML::Metal::Device.init!
        @memory.emit("prebuffers", phase_bytes_kind: "planned_named_metal_bytes", phase_bytes: @named_bytes)
        begin
          @resources.each { |name, bytes| @buffers[name] = ML::MetalBuffer.new(bytes) }
          upload_fixture!
          initialize_outputs!
          @memory.emit("postupload", phase_bytes_kind: "owned_named_metal_bytes", phase_bytes: @named_bytes)
          @plan_order = PixelUMMFFNIsolatedProbe.plan_order(reverse_warmups)
          @plan_order.each_with_index do |name, index|
            compile_plan!(name)
            @memory.emit("postcompile:#{name}", slot: index, phase_bytes_kind: "owned_named_metal_bytes", phase_bytes: @named_bytes)
          end
          @pipeline = ML::Metal::ComputePipeline.new("pixelumm_ffn_compare_bf16", PixelUMMFFNIsolatedProbe::SOURCE)
          @memory.emit("postcompile:comparator", slot: @plan_order.size, phase_bytes_kind: "owned_named_metal_bytes", phase_bytes: @named_bytes)
        rescue error
          release_unsubmitted!
          raise error
        end
      end

      def buffer(name : String) : ML::MetalBuffer
        @buffers[name]
      end

      def canonical_ready? : Bool
        @canonical_ready
      end

      def preencode! : Nil
        @memory.emit("preencode", phase_bytes_kind: "owned_named_metal_bytes", phase_bytes: @named_bytes)
      end

      def poison!(chain : ML::Metal::MPSCommandChain, names : Array(String)) : Nil
        @lifecycle.mark_attempt!
        chain.pause!
        blit = MetalDispatchFFI.create_blit_encoder(chain.root_handle)
        raise "isolated FFN poison blit unavailable" if blit.null?
        names.each do |name|
          MetalDispatchFFI.blit_fill_buffer(blit, buffer(name).handle, 255_u8, 0_i64, buffer(name).size)
        end
        MetalDispatchFFI.blit_end_encoding(blit)
        chain.resume!
      end

      def encode_gate_up(chain : ML::Metal::MPSCommandChain, mode : String) : Nil
        case mode
        when "separate"
          encode_plan!(chain, "separate_gate")
          encode_plan!(chain, "separate_up")
        when "shared"
          encode_plan!(chain, "shared_gate_up")
        else
          raise ArgumentError.new("unknown isolated FFN mode")
        end
      end

      def encode_down(chain : ML::Metal::MPSCommandChain) : Nil
        encode_plan!(chain, "down")
      end

      def compare_gate_up(chain : ML::Metal::MPSCommandChain, sample : Int32) : Nil
        check!(chain, "candidate.gate", "expected.gate", sample * 4, @tokens * FFN)
        check!(chain, "candidate.up", "expected.up", sample * 4 + 1, @tokens * FFN)
      end

      def copy_down_canonical!(chain : ML::Metal::MPSCommandChain) : Nil
        raise ArgumentError.new("down canonical output was already captured") if @canonical_ready
        @lifecycle.mark_attempt!
        chain.pause!
        blit = MetalDispatchFFI.create_blit_encoder(chain.root_handle)
        raise "isolated FFN canonical blit unavailable" if blit.null?
        MetalDispatchFFI.blit_copy_buffer(blit, buffer("candidate.down").handle, 0_i64,
          buffer("down.canonical").handle, 0_i64, buffer("candidate.down").size)
        MetalDispatchFFI.blit_end_encoding(blit)
        chain.resume!
        @canonical_ready = true
      end

      def compare_down(chain : ML::Metal::MPSCommandChain, sample : Int32) : Nil
        raise ArgumentError.new("down canonical output is not ready") unless @canonical_ready
        count = @tokens * HIDDEN
        check!(chain, "candidate.down", "expected.down", sample * 4 + 2, count)
        check!(chain, "candidate.down", "down.canonical", sample * 4 + 3, count)
      end

      def inject_nonfinite_control(chain : ML::Metal::MPSCommandChain) : Nil
        raise ArgumentError.new("down canonical output is not ready") unless @canonical_ready
        check!(chain, "candidate.down", "expected.down", 24, @tokens * HIDDEN, inject: true)
      end

      def poststage!(stage : String, sample : Int32) : Nil
        @memory.emit("poststage:#{stage}", slot: sample, request: sample,
          phase_bytes_kind: "owned_named_metal_bytes", phase_bytes: @named_bytes)
      end

      def metrics(chain : ML::Metal::MPSCommandChain) : Array(Metric)
        raise ArgumentError.new("isolated FFN metrics require the sole terminal wait") unless chain.completed? && chain.wait_count == 1
        words = Array(UInt32).new(METRIC_SLOTS * 4, 0_u32)
        buffer("metrics").read_bytes(words.to_unsafe.as(UInt8*), words.size * sizeof(UInt32))
        Array.new(METRIC_SLOTS) do |index|
          {failures: words[index * 4], different_words: words[index * 4 + 1],
           nonfinite: words[index * 4 + 2], elements: words[index * 4 + 3]}
        end
      end

      def release_after_terminal!(chain : ML::Metal::MPSCommandChain) : Nil
        unless @lifecycle.release_allowed?(chain.completed?, chain.wait_count)
          raise ArgumentError.new("isolated FFN plans/buffers cannot be released before terminal completion")
        end
        release_plans!
        @buffers.keys.each do |name|
          @buffers.delete(name).try(&.release)
        end
        @memory.emit("after_all_buffers_release", phase_bytes_kind: "owned_named_metal_bytes", phase_bytes: 0_i64)
        @released = true
      end

      def postwait!(chain : ML::Metal::MPSCommandChain) : Nil
        @lifecycle.mark_terminal_wait!(chain.completed?, chain.wait_count)
        @memory.emit("postwait", phase_bytes_kind: "owned_named_metal_bytes", phase_bytes: @named_bytes)
      end

      def api_profile_json(profile : ML::Video::MPSAPIProfile) : Array(JSON::Any)
        profile.phases.map do |phase|
          operations = {} of String => JSON::Any
          phase.operations.each do |name, stats|
            operations[name] = JSON.parse({count: stats.count, wall_total_ns: stats.wall_total_ns,
                                           wall_max_ns: stats.wall_max_ns, thread_cpu_total_ns: stats.thread_cpu_total_ns,
                                           thread_cpu_max_ns: stats.thread_cpu_max_ns, nested_count: stats.nested_count,
                                           nested_wall_total_ns: stats.nested_wall_total_ns,
                                           nested_thread_cpu_total_ns: stats.nested_thread_cpu_total_ns,
                                           exclusive_wall_total_ns: stats.exclusive_wall_total_ns,
                                           exclusive_thread_cpu_total_ns: stats.exclusive_thread_cpu_total_ns}.to_json)
          end
          JSON.parse({phase_id: phase.phase_id, phase: phase.name, operations: operations}.to_json)
        end
      end

      def release_plan!(name : String, index : Int32) : Nil
        plan = @plans.delete(name)
        PixelUMMFFNIsolatedFFI.step_release(plan) if plan
        @memory.emit("after_plan_release:#{name}", slot: index,
          phase_bytes_kind: "owned_named_metal_bytes", phase_bytes: @named_bytes)
      end

      private def release_plans! : Nil
        @plan_order.each_with_index { |name, index| release_plan!(name, index) }
      end

      def release_unsubmitted! : Nil
        return if @released
        unless @lifecycle.release_allowed?(false, 0)
          raise ArgumentError.new("isolated FFN objects cannot be released after a submission attempt")
        end
        release_plans!
        @buffers.each_value(&.release)
        @buffers.clear
        @released = true
      end

      private def upload_fixture! : Nil
        fixture = @fixture
        {
          "input.post_attn_norm" => fixture.metadata.inputs["post_attn_norm"],
          "input.product"        => fixture.metadata.inputs["product"],
          "weight.gate"          => fixture.metadata.weights["mlp.gate_proj.weight"],
          "weight.up"            => fixture.metadata.weights["mlp.up_proj.weight"],
          "weight.down"          => fixture.metadata.weights["mlp.down_proj.weight"],
          "expected.gate"        => fixture.metadata.captures["gate_proj"],
          "expected.up"          => fixture.metadata.captures["up_proj"],
          "expected.down"        => fixture.metadata.captures["down_proj"],
        }.each { |name, record| fixture.read_into(record, buffer(name)) }
      end

      private def initialize_outputs! : Nil
        {"candidate.gate", "candidate.up", "candidate.down", "down.canonical", "metrics"}.each do |name|
          Slice.new(buffer(name).contents.as(UInt8*), buffer(name).size.to_i32).fill(0_u8)
        end
      end

      private def compile_plan!(name : String) : Nil
        kind = case name
               when "separate_gate"  then 0
               when "separate_up"    then 1
               when "shared_gate_up" then 2
               when "down"           then 3
               else                       raise ArgumentError.new("unknown isolated FFN graph name")
               end
        error = Bytes.new(1024, 0_u8)
        plan = PixelUMMFFNIsolatedFFI.step_create(
          ML::Metal::Device.instance.device_handle,
          buffer("input.post_attn_norm").handle,
          buffer("input.product").handle,
          buffer("weight.gate").handle,
          buffer("weight.up").handle,
          buffer("weight.down").handle,
          buffer("candidate.gate").handle,
          buffer("candidate.up").handle,
          buffer("candidate.down").handle,
          @tokens, kind, error.to_unsafe, error.size)
        raise "isolated FFN graph preparation #{name}: #{String.new(error.to_unsafe)}" if plan.null?
        @plans[name] = plan
      end

      private def encode_plan!(chain : ML::Metal::MPSCommandChain, name : String) : Nil
        plan = @plans[name]? || raise ArgumentError.new("isolated FFN graph is unavailable: #{name}")
        @lifecycle.mark_attempt!
        chain.pause!
        error = Bytes.new(1024, 0_u8)
        status = PixelUMMFFNIsolatedFFI.step_encode(plan, chain.handle, error.to_unsafe, error.size)
        chain.resume!
        raise "isolated FFN graph encode #{name}: #{status}: #{String.new(error.to_unsafe)}" unless status == 0
      end

      private def check!(chain : ML::Metal::MPSCommandChain, actual : String, expected : String,
                         slot : Int32, count : Int32, inject = false) : Nil
        pipeline = @pipeline || raise ArgumentError.new("isolated FFN comparator is unavailable")
        @lifecycle.mark_attempt!
        chain.encoder.set_pipeline(pipeline)
          .set_buffer(buffer(actual), 0)
          .set_buffer(buffer(expected), 1)
          .set_buffer(buffer("metrics"), 2, ML::Metal::BufferAccess::ReadWrite, slot.to_i64 * 16)
          .set_value(count.to_u32, 3)
          .set_value(count.to_u32, 4)
          .set_value(inject ? 1_u32 : 0_u32, 5)
          .dispatch_threadgroups({(count + 255) // 256, 1, 1}, {256, 1, 1})
      end
    end

    def next_root(chain : ML::Metal::MPSCommandChain) : Nil
      before = chain.observed_root_count
      chain.pause!
      chain.continue_async!
      chain.resume!
      unless chain.observed_root_count == before + 1 && chain.wait_count == 0
        raise "isolated FFN async boundary changed root/wait state"
      end
    end

    def timed_stage(chain : ML::Metal::MPSCommandChain, stage : String, &block : -> Nil) : StageTiming
      next_root(chain)
      first_root = chain.observed_root_count - 1
      started = Time.instant
      yield
      wall = (Time.instant - started).total_seconds
      end_root = chain.observed_root_count
      next_root(chain)
      {stage: stage, first_root: first_root, end_root: end_root, encode_wall_seconds: wall}
    end

    def root_timing(timing : StageTiming, roots : Array(ML::Video::MPSCommandRootTiming)) : NamedTuple(gpu_root_union_seconds: Float64, outside_root_overlap_seconds: Float64, isolated_root_timing: Bool)
      selected = roots.select { |root| timing[:first_root] <= root.sequence < timing[:end_root] }
      unless !selected.empty? && selected.all? { |root| root.completed && !root.has_error &&
             root.gpu_start_seconds.finite? && root.gpu_end_seconds.finite? &&
             root.gpu_start_seconds > 0 && root.gpu_end_seconds >= root.gpu_start_seconds }
        raise ArgumentError.new("isolated FFN timed root is missing or failed")
      end
      ranges = selected.sort_by(&.gpu_start_seconds)
      union = 0.0
      cursor = ranges.first.gpu_start_seconds
      ranges.each do |root|
        left = Math.max(cursor, root.gpu_start_seconds)
        union += Math.max(0.0, root.gpu_end_seconds - left)
        cursor = Math.max(cursor, root.gpu_end_seconds)
      end
      outside = roots.reject { |root| timing[:first_root] <= root.sequence < timing[:end_root] }
      intersections = [] of Tuple(Float64, Float64)
      selected.each do |root|
        outside.each do |other|
          start = Math.max(root.gpu_start_seconds, other.gpu_start_seconds)
          finish = Math.min(root.gpu_end_seconds, other.gpu_end_seconds)
          intersections << {start, finish} if finish > start
        end
      end
      overlap = interval_union_seconds(intersections)
      raise ArgumentError.new("isolated FFN timed root union is empty") unless union > 0
      {gpu_root_union_seconds: union, outside_root_overlap_seconds: overlap,
       isolated_root_timing: overlap == 0.0}
    end

    private def interval_union_seconds(intervals : Array(Tuple(Float64, Float64))) : Float64
      return 0.0 if intervals.empty?
      sorted = intervals.sort_by(&.[0])
      start, finish = sorted.first
      total = 0.0
      sorted.skip(1).each do |left, right|
        if left <= finish
          finish = Math.max(finish, right)
        else
          total += finish - start
          start, finish = left, right
        end
      end
      total + finish - start
    end
  {% end %}
end
