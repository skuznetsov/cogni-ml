# Matched-input late-layer arithmetic diagnostics. Activations stay on Metal;
# only the final compact metric records are read back.
require "digest/sha256"
require "json"
require "set"
require "../src/ml/audio/drama3/generator_resident"

{% if flag?(:drama3_metal) && flag?(:darwin) && !flag?(:cpu_only) %}
module Drama3LateCompare
  include ML::Audio::Drama3
  alias Buffer = Resident::Buffer
  alias Context = Resident::Context
  SOURCE = {{ read_file("#{__DIR__}/support/drama3_late_compare.metal") }}
  WORDS_PER_RECEIPT = 26
  FLOATS_PER_RECEIPT = WORDS_PER_RECEIPT * 2

  class TensorRecord
    getter name : String
    getter file : String
    getter shape : Array(Int64)
    getter dtype : String
    getter bytes : Int64
    getter sha256 : String
    property buffer : Buffer?

    def initialize(value : JSON::Any)
      @name = value["name"].as_s
      @file = value["file"].as_s
      @shape = value["shape"].as_a.map(&.as_i64)
      @dtype = value["dtype"].as_s
      @bytes = value["bytes"].as_i64
      @sha256 = value["sha256"].as_s
      @buffer = nil
      raise "invalid fixture tensor shape: #{@name}" if @shape.empty? || @shape.any? { |d| d <= 0 }
      expected_bytes = @shape.reduce(1_i64) { |count, dimension| count * dimension } * 4_i64
      raise "unsupported fixture dtype #{@dtype}: #{@name}" unless @dtype.in?("float32", "int32")
      raise "fixture byte count mismatch: #{@name}" unless @bytes == expected_bytes
    end

    def json : Hash(String, JSON::Any)
      {
        "name"   => Drama3LateCompare.j(@name), "sha256" => Drama3LateCompare.j(@sha256),
        "shape"  => Drama3LateCompare.j(@shape), "dtype" => Drama3LateCompare.j(@dtype),
      }
    end
  end

  class Fixture
    getter path : String
    getter root : String
    getter manifest : JSON::Any
    getter sha256 : String
    getter input_sha256 : String
    getter rows : Int32
    getter width : Int32
    @records = {"scalar" => {} of String => TensorRecord, "sequence" => {} of String => TensorRecord}

    def initialize(@path : String)
      @path = File.expand_path(@path)
      @root = File.dirname(@path)
      @manifest = JSON.parse(File.read(@path))
      raise "unexpected late fixture format" unless @manifest["format"].as_s == "drama3-late-matched-intermediates-v1"
      @sha256 = Drama3LateCompare.sha(@path)
      @input_sha256 = @manifest["input"]["used_f32_sha256"].as_s
      @rows = @manifest["input"]["used_rows"].as_i
      @width = @manifest["input"]["width"].as_i
      ["scalar", "sequence"].each do |arm|
        @manifest["arms"][arm]["tensors"].as_a.each do |value|
          record = TensorRecord.new(value)
          raise "duplicate fixture tensor #{arm}:#{record.name}" if @records[arm].has_key?(record.name)
          @records[arm][record.name] = record
        end
      end
      input = @manifest["input"]
      raise "fixture input byte count" unless input["used_f32_bytes"].as_i64 == @rows.to_i64 * @width * 4
    end

    def record(arm : String, name : String) : TensorRecord
      @records[arm][name]? || raise "missing fixture tensor #{arm}:#{name}"
    end

    def tensor(ctx : Context, arm : String, name : String) : TensorRecord
      result = record(arm, name)
      result.buffer ||= load(ctx, result.file, result.bytes, result.dtype, result.sha256)
      result
    end

    def clear_buffers : Nil
      @records.each_value { |records| records.each_value { |record| record.buffer = nil } }
    end

    def input(ctx : Context) : Buffer
      info = @manifest["input"]
      relative = info["used_f32_file"].as_s
      bytes = info["used_f32_bytes"].as_i64
      raise "fixture input width/rows" unless @rows > 0 && @width > 0
      raise "fixture input hash mismatch" unless @input_sha256 == info["used_f32_sha256"].as_s
      load(ctx, relative, bytes, "float32", @input_sha256)
    end

    private def load(ctx : Context, relative : String, expected_bytes : Int64, dtype : String, expected_sha : String) : Buffer
      path = File.expand_path(relative, @root)
      raise "fixture path escaped root" unless path.starts_with?(@root + "/")
      raise "fixture tensor byte count on disk" unless File.size(path) == expected_bytes
      raise "fixture tensor SHA mismatch: #{relative}" unless Drama3LateCompare.sha(path) == expected_sha
      raise "fixture tensor alignment" unless expected_bytes > 0 && expected_bytes % 4 == 0 && expected_bytes // 4 <= Int32::MAX
      buffer = ctx.alloc((expected_bytes // 4).to_i32)
      File.open(path) do |io|
        target = Bytes.new(buffer.buffer.contents.as(UInt8*) + buffer.offset, expected_bytes.to_i32)
        io.read_fully(target)
      end
      buffer
    end
  end

  class Operand
    getter name : String
    getter shape : Array(Int64)
    getter origin : String
    getter record : TensorRecord?
    getter produced_by : String?

    def initialize(@name : String, @shape : Array(Int64), @origin : String,
                   @record : TensorRecord? = nil, @produced_by : String? = nil)
    end

    def self.upstream(record : TensorRecord) : Operand
      Operand.new(record.name, record.shape, "upstream_fixture", record)
    end

    def self.native(name : String, shape : Array(Int64), produced_by : String) : Operand
      Operand.new(name, shape, "native_device_result", nil, produced_by)
    end
  end

  class CaseGroup
    getter id : String
    getter kind : String
    getter stage : String
    getter ref_arm : String
    getter shape : Array(Int64)
    getter axes : Array(String)
    getter atol : Float32
    getter rtol : Float32
    getter reference : TensorRecord
    getter operands : Array(Operand)
    getter route_ids : TensorRecord?
    getter model_tensors : Array(Hash(String, JSON::Any))
    getter slots = [] of Int32

    def initialize(@id : String, @kind : String, @stage : String, @ref_arm : String,
                   @shape : Array(Int64), @axes : Array(String), @atol : Float32,
                   @rtol : Float32, @reference : TensorRecord, @operands : Array(Operand),
                   @route_ids : TensorRecord?, @model_tensors : Array(Hash(String, JSON::Any)))
    end
  end

  class Comparator
    getter ctx : Context
    getter fixture : Fixture
    getter groups : Hash(String, CaseGroup)
    getter order = [] of String
    getter metrics : Buffer
    getter merged_metrics : Buffer
    getter slots = 0
    @dummy_routes : Buffer

    def initialize(@ctx : Context, @fixture : Fixture, max_slots : Int32)
      @groups = {} of String => CaseGroup
      @metrics = @ctx.alloc(max_slots * FLOATS_PER_RECEIPT)
      @merged_metrics = @ctx.alloc(330 * FLOATS_PER_RECEIPT)
      @dummy_routes = @ctx.zeros(1)
    end

    def compare(id : String, kind : String, stage : String, arm : String,
                actual : Buffer, reference : TensorRecord, reference_slice : Buffer,
                atol : Float32, rtol : Float32, axes : Array(String), offset : UInt64 = 0_u64,
                operands : Array(Operand) = [] of Operand, route_ids : TensorRecord? = nil,
                route_buffer : Buffer? = nil, model_tensors : Array(Hash(String, JSON::Any)) = [] of Hash(String, JSON::Any)) : Nil
      raise "comparison element count: #{id}" unless actual.count == reference_slice.count
      shape = reference.shape
      if existing = @groups[id]?
        raise "case metadata drift: #{id}" unless existing.reference.same?(reference) && existing.ref_arm == arm
        group = existing
      else
        group = CaseGroup.new(id, kind, stage, arm, shape, axes, atol, rtol,
          reference, operands, route_ids, model_tensors)
        @groups[id] = group
        @order << id
      end
      raise "comparator metrics capacity" if @slots >= @metrics.count // FLOATS_PER_RECEIPT
      parts_count = (actual.count + 1023) // 1024
      parts = @ctx.alloc(parts_count * WORDS_PER_RECEIPT)
      output = @metrics.slice(@slots * FLOATS_PER_RECEIPT, FLOATS_PER_RECEIPT)
      ids = route_buffer || @dummy_routes
      hidden = (shape.last? || 1_i64).to_u64
      params = [actual.count.to_u64, atol.unsafe_as(UInt32).to_u64,
        rtol.unsafe_as(UInt32).to_u64, offset, route_buffer ? 1_u64 : 0_u64, hidden]
      @ctx.dispatch("dr3_late_compare_tile", SOURCE,
        [actual, reference_slice, parts, ids], params, parts_count, 1)
      @ctx.dispatch("dr3_late_compare_reduce", SOURCE, [parts, output], [parts_count.to_u64], 1, 1)
      group.slots << @slots
      @slots += 1
    end

    def reports(manifest_sha : String, input_sha : String, model_config_sha : String,
                model_index_sha : String, chunk : Int32, depth : Int32,
                consumed : Int32) : Array(Hash(String, JSON::Any))
      raise "no comparison receipts" unless consumed == @slots
      raise "comparator group capacity" if @order.size > @merged_metrics.count // FLOATS_PER_RECEIPT
      @order.each_with_index do |id, index|
        group = @groups[id]
        params = [group.slots.size.to_u64] + group.slots.map(&.to_u64)
        @ctx.dispatch("dr3_late_compare_merge", SOURCE,
          [@metrics, @merged_metrics.slice(index * FLOATS_PER_RECEIPT, FLOATS_PER_RECEIPT)],
          params, 1, 1)
      end
      values = @ctx.read(@merged_metrics.slice(0, @order.size * FLOATS_PER_RECEIPT), "terminal")
      @order.map_with_index do |id, index|
        group = @groups[id]
        merged = Array(UInt32).new(WORDS_PER_RECEIPT, 0_u32)
        offset = index * FLOATS_PER_RECEIPT
        WORDS_PER_RECEIPT.times do |i|
          low = values[offset + i * 2].to_u32
          high = values[offset + i * 2 + 1].to_u32
          merged[i] = low | (high << 16)
        end
        case_json(group, merged, manifest_sha, input_sha, model_config_sha, model_index_sha, chunk, depth)
      end
    end

    private def witness_index(words : Array(UInt32), start : Int32) : UInt64
      low, high = words[start], words[start + 1]
      return UInt64::MAX if low == UInt32::MAX && high == UInt32::MAX
      (high.to_u64 << 32) | low.to_u64
    end

    private def case_json(group : CaseGroup, words : Array(UInt32), manifest_sha : String,
                          input_sha : String, model_config_sha : String, model_index_sha : String,
                          chunk : Int32, depth : Int32) : Hash(String, JSON::Any)
      sources = [] of JSON::Any
      seen = Set(String).new
      add_source = ->(record : TensorRecord, role : String) do
        key = "#{group.ref_arm}|#{record.name}|#{record.sha256}"
        if !seen.includes?(key)
          entry = record.json
          entry["origin"] = Drama3LateCompare.j("upstream_fixture")
          entry["ref_arm"] = Drama3LateCompare.j(group.ref_arm)
          entry["uses"] = Drama3LateCompare.j([role])
          sources << JSON::Any.new(entry)
          seen.add(key)
        else
          sources.each do |entry|
            if entry["name"].as_s == record.name && entry["sha256"].as_s == record.sha256
              entry["uses"].as_a << Drama3LateCompare.j(role)
              break
            end
          end
        end
      end
      add_source.call(group.reference, "reference")
      group.operands.each_with_index do |operand, index|
        if record = operand.record
          add_source.call(record, "operand:#{index}")
        end
      end
      if route = group.route_ids
        add_source.call(route, "route_ids")
      end
      group.operands.each_with_index do |operand, index|
        next if operand.record
        if operand.origin == "input_fixture"
          sources << Drama3LateCompare.j({
            "name" => operand.name, "shape" => operand.shape, "origin" => operand.origin,
            "sha256" => input_sha, "manifest_sha256" => manifest_sha,
            "operand_index" => index,
          })
        else
          sources << Drama3LateCompare.j({
            "name" => operand.name, "shape" => operand.shape, "origin" => operand.origin,
            "produced_by" => operand.produced_by, "manifest_sha256" => manifest_sha,
            "input_sha256" => input_sha, "model_config_sha256" => model_config_sha,
            "model_index_sha256" => model_index_sha, "operand_index" => index,
          })
        end
      end
      reference_json = group.reference.json
      operand_json = group.operands.map do |operand|
        value = {"name" => Drama3LateCompare.j(operand.name), "shape" => Drama3LateCompare.j(operand.shape),
          "origin" => Drama3LateCompare.j(operand.origin)} of String => JSON::Any
        if record = operand.record
          value["sha256"] = Drama3LateCompare.j(record.sha256)
          value["ref_arm"] = Drama3LateCompare.j(group.ref_arm)
        elsif operand.origin == "input_fixture"
          value["sha256"] = Drama3LateCompare.j(input_sha)
        else
          value["produced_by"] = Drama3LateCompare.j(operand.produced_by)
        end
        JSON::Any.new(value)
      end
      route_json = if route = group.route_ids
                     JSON::Any.new(route.json)
                   else
                     JSON::Any.new(nil)
                   end
      metrics = {
        "bit_mismatch_count" => Drama3LateCompare.j(words[0].to_i64),
        "tolerance_failure_count" => Drama3LateCompare.j(words[1].to_i64),
        "nonfinite_count" => Drama3LateCompare.j(words[2].to_i64),
        "max_abs" => Drama3LateCompare.j(Drama3LateCompare.float_text(Drama3LateCompare.bits_float(words[3]))),
        "max_ratio" => Drama3LateCompare.j(Drama3LateCompare.float_text(Drama3LateCompare.bits_float(words[4]))),
        "first_bit" => witness_json(words, 5, group.shape),
        "first_tolerance" => witness_json(words, 12, group.shape),
        "first_nonfinite" => witness_json(words, 19, group.shape),
      } of String => JSON::Any
      identity = {
        "manifest_sha256" => Drama3LateCompare.j(manifest_sha), "input_sha256" => Drama3LateCompare.j(input_sha),
        "model_config_sha256" => Drama3LateCompare.j(model_config_sha), "model_index_sha256" => Drama3LateCompare.j(model_index_sha),
        "native_chunk" => Drama3LateCompare.j(chunk), "native_graph_depth" => Drama3LateCompare.j(depth),
        "source_tensors" => JSON::Any.new(sources), "model_tensors" => JSON::Any.new(group.model_tensors.map { |x| JSON::Any.new(x) }),
      } of String => JSON::Any
      {
        "id" => Drama3LateCompare.j(group.id), "kind" => Drama3LateCompare.j(group.kind), "stage" => Drama3LateCompare.j(group.stage),
        "ref_arm" => Drama3LateCompare.j(group.ref_arm), "shape" => Drama3LateCompare.j(group.shape), "axes" => Drama3LateCompare.j(group.axes),
        "atol" => Drama3LateCompare.j(group.atol), "rtol" => Drama3LateCompare.j(group.rtol),
        "reference" => JSON::Any.new(reference_json), "operands" => JSON::Any.new(operand_json),
        "route_ids" => route_json, "identity" => JSON::Any.new(identity),
        "metrics" => JSON::Any.new(metrics),
      } of String => JSON::Any
    end

    private def witness_json(words : Array(UInt32), start : Int32, shape : Array(Int64)) : JSON::Any
      index = witness_index(words, start)
      return JSON::Any.new(nil) if index == UInt64::MAX
      actual, reference, error, allowance = words[start + 2], words[start + 3], words[start + 4], words[start + 5]
      expert_id = words[start + 6].unsafe_as(Int32)
      value = {
        "index" => Drama3LateCompare.j(index), "actual_f32" => Drama3LateCompare.j(Drama3LateCompare.float_text(Drama3LateCompare.bits_float(actual))),
        "reference_f32" => Drama3LateCompare.j(Drama3LateCompare.float_text(Drama3LateCompare.bits_float(reference))),
        "error_f32" => Drama3LateCompare.j(Drama3LateCompare.float_text(Drama3LateCompare.bits_float(error))),
        "allowance_f32" => Drama3LateCompare.j(Drama3LateCompare.float_text(Drama3LateCompare.bits_float(allowance))),
        "actual_bits" => Drama3LateCompare.j(Drama3LateCompare.hex_bits(actual)), "reference_bits" => Drama3LateCompare.j(Drama3LateCompare.hex_bits(reference)),
        "coordinates" => JSON::Any.new(coordinates(index, shape)),
      } of String => JSON::Any
      value["expert_id"] = Drama3LateCompare.j(expert_id) if expert_id >= 0
      JSON::Any.new(value)
    end

    private def coordinates(index : UInt64, shape : Array(Int64)) : Hash(String, JSON::Any)
      coordinates = Array(Int64).new(shape.size, 0_i64)
      remainder = index.to_i64
      (shape.size - 1).downto(0) do |axis|
        coordinates[axis] = remainder % shape[axis]
        remainder //= shape[axis]
      end
      result = {"indices" => Drama3LateCompare.j(coordinates)} of String => JSON::Any
      result["row"] = Drama3LateCompare.j(coordinates[0]) unless coordinates.empty?
      result["rank"] = Drama3LateCompare.j(coordinates[1]) if coordinates.size >= 3
      result["channel"] = Drama3LateCompare.j(coordinates[-1]) if coordinates.size >= 2
      result
    end
  end

  def self.j(value) : JSON::Any
    JSON.parse(value.to_json)
  end

  def self.sha(path : String) : String
    digest = Digest::SHA256.new
    File.open(path) do |io|
      bytes = Bytes.new(1024 * 1024)
      while (n = io.read(bytes)) > 0
        digest.update(bytes[0, n])
      end
    end
    digest.hexfinal
  end

  def self.bits_float(value : UInt32) : Float32
    value.unsafe_as(Float32)
  end

  def self.hex_bits(value : UInt32) : String
    "0x" + value.to_s(16).upcase.rjust(8, '0')
  end

  def self.float_text(value : Float32) : String
    return "NaN" if value.nan?
    return value > 0 ? "+Inf" : "-Inf" if value.infinite?
    value.to_s
  end

  def self.weight_json(store : WeightStore, name : String) : Hash(String, JSON::Any)
    tensor = store.tensor(name)
    digest = Digest::SHA256.new
    if raw = tensor.raw_bytes
      digest.update(raw)
    else
      bytes = Bytes.new(4)
      tensor.values.each do |value|
        IO::ByteFormat::LittleEndian.encode(value, bytes)
        digest.update(bytes)
      end
    end
    {
      "name" => j(name), "sha256" => j(digest.hexfinal),
      "shape" => j(tensor.shape.map(&.to_i64)), "dtype" => j(tensor.dtype),
    } of String => JSON::Any
  end

  def self.affine_matrix_json(store : WeightStore, weight_name : String) : Array(Hash(String, JSON::Any))
    stem = weight_name.ends_with?(".weight") ? weight_name[0...-7] : weight_name
    [weight_json(store, weight_name), weight_json(store, stem + ".scales"), weight_json(store, stem + ".biases")]
  end

  def self.input_record(fixture : Fixture) : Operand
    Operand.new("input.used_f32_file", [fixture.rows.to_i64, fixture.width.to_i64],
      "input_fixture", nil, nil)
  end

  def self.axes_for(shape : Array(Int64)) : Array(String)
    case shape.size
    when 3 then ["row", "rank", "channel"]
    when 2 then ["row", "channel"]
    else Array(String).new(shape.size) { |i| "axis_#{i}" }
    end
  end

  def self.layer_mlp(layer : Int32, name : String) : String
    "backbone.layers.#{layer}.mlp.#{name}"
  end

  def self.case_kind_order(kind : String) : Int32
    case kind
    when "post_attention_rms_replay" then 0
    when "expert_down_native" then 1
    when "expert_down_mlx_q4" then 2
    when "shared_projection" then 3
    when "shared_activation_replay" then 4
    when "shared_down_replay" then 5
    when "shared_weighted_replay" then 6
    when "moe_sum_add_replay" then 7
    when "moe_weighted_add_replay" then 8
    when "whole_moe_replay" then 9
    when "residual_upstream_add_control" then 10
    when "residual_native_moe_replay" then 11
    else 99
    end
  end

  def self.run : Nil
    args = ARGV.dup
    raise "usage: drama3_late_compare MODEL_DIR FIXTURE_MANIFEST OUTPUT.json [--chunk=16] [--depth=0|2]" unless args.size.in?(3..5)
    model_path, manifest_path, output_path = args.shift(3)
    chunk, depth = 16, 0
    args.each do |argument|
      key, value = argument.split("=", 2)
      case key
      when "--chunk"
        chunk = value.to_i
        raise "chunk must be 1..32" unless chunk.in?(1..32)
      when "--depth"
        depth = value.to_i
        raise "depth must be 0 or 2" unless depth == 0 || depth == 2
      else raise "unknown option #{argument}"
      end
    end
    ENV["DRAMA3_COGNIGRAPH_MAX_INFLIGHT"] = depth.to_s
    ENV["DRAMA3_MOE_BATCH"] = "1"
    ENV["DRAMA3_BACKBONE_DENSE_REDUCTION"] = "tree"
    ENV["DRAMA3_EXPERT_DOWN_REDUCTION"] = "native"
    raise "refusing existing output" if File.exists?(output_path)

    fixture = Fixture.new(manifest_path)
    config_path = File.join(model_path, "config.json")
    index_path = File.join(model_path, "model.safetensors.index.json")
    model_config_sha = sha(config_path)
    model_index_sha = sha(index_path)
    checkpoint = fixture.manifest["checkpoint"]
    raise "model config hash differs from fixture" unless checkpoint["config_sha256"].as_s == model_config_sha
    raise "model index hash differs from fixture" unless checkpoint["index_sha256"].as_s == model_index_sha
    raise "fixture input dimensions" unless fixture.rows == 158 && fixture.width > 0

    config = BackboneConfig.new(JSON.parse(File.read(config_path)))
    raise "input width does not match model" unless config.hidden == fixture.width
    store = WeightStore.new(model_path)
    ctx = Context.new
    max_slots = 81 * 2 * ((fixture.rows + chunk - 1) // chunk) + 14 * 6 * 2
    comparator = Comparator.new(ctx, fixture, max_slots)
    input = fixture.input(ctx)

    full_stages = [] of String
    config.layers.times do |layer|
      full_stages << "backbone.layers.#{layer}.attention_residual"
      full_stages << "backbone.layers.#{layer}.hidden"
    end
    full_stages << "backbone.hidden"
    raise "expected exactly 81 boundary stages" unless full_stages.size == 81

    boundary_model = BackboneResident.new(config, store, ctx, fixture.rows + 1)
    base = 0
    boundary_model.prefill_trace = ->(stage : String, actual : Buffer) {
      arms = ["scalar", "sequence"]
      arms.each do |arm|
        reference = fixture.tensor(ctx, arm, stage)
        width = reference.shape.last.to_i32
        slice = reference.buffer.not_nil!.slice(base * width, actual.count)
        group_id = "full_boundary|#{arm}|#{stage.starts_with?("backbone.hidden") ? "final" : stage.split(".")[2]}|#{stage}"
        comparator.compare(group_id, "full_boundary", stage, arm, actual, reference, slice,
          stage == "backbone.hidden" ? 3e-5_f32 : 6e-5_f32, 3e-4_f32,
          ["row", "channel"], (base * width).to_u64,
          [input_record(fixture)], nil, nil,
          [] of Hash(String, JSON::Any))
      end
      nil
    }
    while base < fixture.rows
      rows = Math.min(chunk, fixture.rows - base)
      boundary_model.prefill(input.slice(base * fixture.width, rows * fixture.width), rows)
      ctx.submit
      base += rows
    end
    boundary_model.prefill_trace = nil
    # Run exact-upstream-operand replays for both arms, one arm at a time to
    # bound retained fixture buffers. Cumulative boundary comparisons remain
    # separate from same-input operation probes.
    replay_cases = [] of String
    moefull = BackboneResident.new(config, store, ctx, fixture.rows + 1)
    ["scalar", "sequence"].each do |arm|
      34.upto(39) do |layer|
        prefix = "backbone.layers.#{layer}"
        mlp = prefix + ".mlp."
        normalized = fixture.tensor(ctx, arm, mlp + "normalized_input")
        attention_residual = fixture.tensor(ctx, arm, prefix + ".attention_residual")
        moe_output_ref = fixture.tensor(ctx, arm, mlp + "moe_output")
        hidden_ref = fixture.tensor(ctx, arm, prefix + ".hidden")
        rows = fixture.rows
        hidden = fixture.width
        rms_name = prefix + ".post_attention_layernorm.weight"
        rms_weight = store.tensor(rms_name)
        rms = ctx.rms(attention_residual.buffer.not_nil!, ctx.tensor(rms_weight), config.eps, heads: rows)
        reference = normalized
        id = "post_attention_rms_replay|#{arm}|#{layer}|#{mlp}normalized_input"
        comparator.compare(id, "post_attention_rms_replay", mlp + "normalized_input", arm,
          rms, reference, reference.buffer.not_nil!, 6e-5_f32, 3e-4_f32,
          ["row", "channel"], 0_u64, [Operand.upstream(attention_residual)], nil, nil,
          [weight_json(store, rms_name)])
        replay_cases << id

        activation = fixture.tensor(ctx, arm, mlp + "canonical_expert_activation_outputs")
        expert_ids = fixture.tensor(ctx, arm, mlp + "canonical_expert_ids")
        down_name = mlp + "switch_mlp.down_proj.weight"
        down_matrix = store.matrix(down_name, 0)
        float_ids = ctx.alloc(expert_ids.buffer.not_nil!.count)
        ctx.dispatch("dr3_late_ids_i32_to_f32", SOURCE,
          [expert_ids.buffer.not_nil!, float_ids], [float_ids.count.to_u64], float_ids.count)
        down = ctx.experts_rows(down_matrix, activation.buffer.not_nil!, float_ids,
          rows, config.top_k, per_rank_input: true, reduction: "native")
        reference = fixture.tensor(ctx, arm, mlp + "canonical_expert_down_outputs")
        id = "expert_down_native|#{arm}|#{layer}|#{mlp}canonical_expert_down_outputs"
        comparator.compare(id, "expert_down_native", mlp + "canonical_expert_down_outputs", arm,
          down, reference, reference.buffer.not_nil!, 6e-5_f32, 3e-4_f32,
          ["row", "rank", "channel"], 0_u64,
          [Operand.upstream(activation), Operand.upstream(expert_ids)], expert_ids,
          expert_ids.buffer, affine_matrix_json(store, down_name))
        replay_cases << id

        down_q4 = ctx.experts_rows(down_matrix, activation.buffer.not_nil!, float_ids,
          rows, config.top_k, per_rank_input: true, reduction: "mlx_q4")
        id = "expert_down_mlx_q4|#{arm}|#{layer}|#{mlp}canonical_expert_down_outputs"
        comparator.compare(id, "expert_down_mlx_q4", mlp + "canonical_expert_down_outputs", arm,
          down_q4, reference, reference.buffer.not_nil!, 6e-5_f32, 3e-4_f32,
          ["row", "rank", "channel"], 0_u64,
          [Operand.upstream(activation), Operand.upstream(expert_ids)], expert_ids,
          expert_ids.buffer, affine_matrix_json(store, down_name))
        replay_cases << id

        shared_gate_name = mlp + "shared_expert.gate_proj.weight"
        shared_up_name = mlp + "shared_expert.up_proj.weight"
        shared_down_name = mlp + "shared_expert.down_proj.weight"
        shared_gate = ctx.matvec_rows(store.matrix(shared_gate_name), normalized.buffer.not_nil!, rows, dense_reduction: "tree")
        shared_up = ctx.matvec_rows(store.matrix(shared_up_name), normalized.buffer.not_nil!, rows, dense_reduction: "tree")
        { {"shared_gate_output", shared_gate, shared_gate_name}, {"shared_up_output", shared_up, shared_up_name} }.each do |name, actual, weight_name|
          stage = mlp + name
          ref = fixture.tensor(ctx, arm, stage)
          id = "shared_projection|#{arm}|#{layer}|#{stage}"
          comparator.compare(id, "shared_projection", stage, arm, actual, ref, ref.buffer.not_nil!,
            6e-5_f32, 3e-4_f32, ["row", "channel"], 0_u64,
            [Operand.upstream(normalized)], nil, nil, [weight_json(store, weight_name)])
          replay_cases << id
        end

        upstream_gate = fixture.tensor(ctx, arm, mlp + "shared_gate_output")
        upstream_up = fixture.tensor(ctx, arm, mlp + "shared_up_output")
        shared_activation = ctx.mul(ctx.silu(upstream_gate.buffer.not_nil!), upstream_up.buffer.not_nil!)
        stage = mlp + "shared_activation_output"
        ref = fixture.tensor(ctx, arm, stage)
        id = "shared_activation_replay|#{arm}|#{layer}|#{stage}"
        comparator.compare(id, "shared_activation_replay", stage, arm, shared_activation, ref,
          ref.buffer.not_nil!, 6e-5_f32, 3e-4_f32, ["row", "channel"], 0_u64,
          [Operand.upstream(upstream_gate), Operand.upstream(upstream_up)])
        replay_cases << id

        upstream_activation = fixture.tensor(ctx, arm, mlp + "shared_activation_output")
        shared_down = ctx.matvec_rows(store.matrix(shared_down_name), upstream_activation.buffer.not_nil!, rows, dense_reduction: "tree")
        stage = mlp + "shared_expert_output"
        ref = fixture.tensor(ctx, arm, stage)
        id = "shared_down_replay|#{arm}|#{layer}|#{stage}"
        comparator.compare(id, "shared_down_replay", stage, arm, shared_down, ref,
          ref.buffer.not_nil!, 6e-5_f32, 3e-4_f32, ["row", "channel"], 0_u64,
          [Operand.upstream(upstream_activation)], nil, nil, [weight_json(store, shared_down_name)])
        replay_cases << id

        upstream_shared = fixture.tensor(ctx, arm, mlp + "shared_expert_output")
        upstream_gate_logits = fixture.tensor(ctx, arm, mlp + "shared_gate_logits")
        shared_weighted = ctx.zeros(rows * hidden)
        ctx.weighted_add_rows!(shared_weighted, upstream_shared.buffer.not_nil!, upstream_gate_logits.buffer.not_nil!, rows, hidden)
        stage = mlp + "shared_weighted_output"
        ref = fixture.tensor(ctx, arm, stage)
        id = "shared_weighted_replay|#{arm}|#{layer}|#{stage}"
        comparator.compare(id, "shared_weighted_replay", stage, arm, shared_weighted, ref,
          ref.buffer.not_nil!, 6e-5_f32, 3e-4_f32, ["row", "channel"], 0_u64,
          [Operand.upstream(upstream_shared), Operand.upstream(upstream_gate_logits)])
        replay_cases << id

        upstream_routed = fixture.tensor(ctx, arm, mlp + "routed_sum")
        upstream_weighted = fixture.tensor(ctx, arm, mlp + "shared_weighted_output")
        sum_add = ctx.add(upstream_routed.buffer.not_nil!, upstream_weighted.buffer.not_nil!)
        stage = mlp + "moe_output"
        ref = moe_output_ref
        id = "moe_sum_add_replay|#{arm}|#{layer}|#{stage}"
        comparator.compare(id, "moe_sum_add_replay", stage, arm, sum_add, ref,
          ref.buffer.not_nil!, 6e-5_f32, 3e-4_f32, ["row", "channel"], 0_u64,
          [Operand.upstream(upstream_routed), Operand.upstream(upstream_weighted)])
        replay_cases << id

        weighted_add = ctx.alloc(rows * hidden)
        ctx.copy(upstream_routed.buffer.not_nil!, weighted_add)
        ctx.weighted_add_rows!(weighted_add, upstream_shared.buffer.not_nil!, upstream_gate_logits.buffer.not_nil!, rows, hidden)
        id = "moe_weighted_add_replay|#{arm}|#{layer}|#{stage}"
        comparator.compare(id, "moe_weighted_add_replay", stage, arm, weighted_add, ref,
          ref.buffer.not_nil!, 6e-5_f32, 3e-4_f32, ["row", "channel"], 0_u64,
          [Operand.upstream(upstream_routed), Operand.upstream(upstream_shared), Operand.upstream(upstream_gate_logits)])
        replay_cases << id

        moefull.diagnostic_layer = layer
        native_moe = moefull.diagnose_moe_rows(normalized.buffer.not_nil!, layer, rows)
        id = "whole_moe_replay|#{arm}|#{layer}|#{stage}"
        comparator.compare(id, "whole_moe_replay", stage, arm, native_moe, ref,
          ref.buffer.not_nil!, 6e-5_f32, 3e-4_f32, ["row", "channel"], 0_u64,
          [Operand.upstream(normalized)], nil, nil,
          [weight_json(store, mlp + "gate.weight"), weight_json(store, mlp + "switch_mlp.gate_proj.weight"),
           weight_json(store, mlp + "switch_mlp.up_proj.weight"), weight_json(store, down_name),
           weight_json(store, shared_gate_name), weight_json(store, shared_up_name), weight_json(store, shared_down_name),
           weight_json(store, mlp + "shared_expert_gate.weight")])
        replay_cases << id

        residual = attention_residual.buffer.not_nil!
        upstream_moe = fixture.tensor(ctx, arm, mlp + "moe_output")
        hidden_actual = ctx.add(residual, upstream_moe.buffer.not_nil!)
        hidden_stage = prefix + ".hidden"
        id = "residual_upstream_add_control|#{arm}|#{layer}|#{hidden_stage}"
        comparator.compare(id, "residual_upstream_add_control", hidden_stage, arm,
          hidden_actual, hidden_ref, hidden_ref.buffer.not_nil!, 6e-5_f32, 3e-4_f32,
          ["row", "channel"], 0_u64,
          [Operand.upstream(attention_residual), Operand.upstream(upstream_moe)])
        replay_cases << id

        native_hidden = ctx.add(residual, native_moe)
        id = "residual_native_moe_replay|#{arm}|#{layer}|#{hidden_stage}"
        comparator.compare(id, "residual_native_moe_replay", hidden_stage, arm,
          native_hidden, hidden_ref, hidden_ref.buffer.not_nil!, 6e-5_f32, 3e-4_f32,
          ["row", "channel"], 0_u64,
          [Operand.upstream(attention_residual), Operand.native(mlp + "native_moe_output", moe_output_ref.shape, "whole_moe_replay|#{arm}|#{layer}|#{stage}")])
        replay_cases << id
      end
    end

    reports = comparator.reports(fixture.sha256, fixture.input_sha256,
      model_config_sha, model_index_sha, chunk, depth, comparator.slots)
    boundary = reports.select { |value| value["kind"].as_s == "full_boundary" }
    strict_pass = boundary.all? do |value|
      metrics = value["metrics"]
      metrics["tolerance_failure_count"].as_i64 == 0 && metrics["nonfinite_count"].as_i64 == 0
    end
    # Order the 81 official boundaries per reference arm before the local
    # matched-operand replays, making scalar and sequence earliest claims
    # independently auditable.
    reports.sort_by! do |value|
      if value["kind"].as_s == "full_boundary"
        arm_order = value["ref_arm"].as_s == "scalar" ? 0 : 1
        stage_index = full_stages.index(value["stage"].as_s) || 81
        {0_i32, arm_order, stage_index.to_i32, 0_i32, ""}
      else
        {1_i32, value["ref_arm"].as_s == "scalar" ? 0 : 1,
          value["stage"].as_s.split(".")[2].to_i? || 0,
          case_kind_order(value["kind"].as_s), value["stage"].as_s}
      end
    end
    report = {} of String => JSON::Any
    report["format"] = j("drama3-late-native-comparison-v1")
    report["manifest"] = j({"path" => fixture.path, "sha256" => fixture.sha256,
      "input_sha256" => fixture.input_sha256})
    report["identity"] = j({"model_config_sha256" => model_config_sha, "model_index_sha256" => model_index_sha,
      "native_chunk" => chunk, "native_graph_depth" => depth})
    report["boundary_count"] = j(boundary.size)
    report["replay_count"] = j(replay_cases.size)
    report["strict_full_native_pass"] = j(strict_pass)
    report["readback_policy"] = j("single terminal compact metrics read; activations remain device resident")
    report["readbacks"] = j(ctx.readbacks)
    report["terminal_readback_bytes"] = j(ctx.readbacks["terminal"])
    report["control_readback_bytes"] = j(ctx.readbacks["control"])
    report["diagnostic_readback_bytes"] = j(ctx.readbacks["diagnostic"])
    report["queue"] = j(ctx.graph_stats)
    report["cases"] = JSON::Any.new(reports.map { |value| JSON::Any.new(value) })
    File.write(output_path, report.to_pretty_json + "\n")
    puts({output: output_path, boundaries: boundary.size, replays: replay_cases.size,
      receipts: comparator.slots, strict_full_native_pass: strict_pass,
      terminal_readbacks: ctx.readbacks["terminal"], control_readbacks: ctx.readbacks["control"]}.to_json)
    raise "full native boundary tolerance/nonfinite gate failed" unless strict_pass
  end
end
Drama3LateCompare.run
{% else %}
  abort "build with -Ddrama3_metal on macOS"
{% end %}
