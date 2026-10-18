# Decode-path dense-fusion gate on the three pinned backbone rows. Activations
# remain on Metal; only compact comparison receipts cross the terminal boundary.
require "digest/sha256"
require "json"
require "set"
require "../src/ml/audio/drama3/backbone_resident"

{% if flag?(:drama3_metal) && flag?(:darwin) && !flag?(:cpu_only) %}
module ML::Audio::Drama3::Resident
  class Context
    # The context selector is frozen when constructed, so the baseline arm uses
    # this diagnostic-only override to execute the unfused reference while the
    # candidate arm reaches the real context-selected kernel.
    property decode_compare_unfused_expert_down_sum = false
    getter decode_compare_reference_expert_down_sum_calls = 0_i64
    property decode_compare_unfused_expert_swiglu = false
    getter decode_compare_reference_expert_swiglu_calls = 0_i64
    getter decode_compare_reference_dense_swiglu_calls = 0_i64

    def experts_swiglu(gate_matrix : Matrix, up_matrix : Matrix, x : Buffer, ids : Buffer,
                       capture_projections : Bool = false) : NamedTuple(output: Buffer, gate: Buffer?, up: Buffer?)
      return previous_def unless @decode_compare_unfused_expert_swiglu

      gate = experts(gate_matrix, x, ids)
      up = experts(up_matrix, x, ids)
      output = mul(silu(gate), up)
      @decode_compare_reference_expert_swiglu_calls += 1
      {output: output, gate: capture_projections ? gate : nil, up: capture_projections ? up : nil}
    end

    def matvec_swiglu(gate_matrix : Matrix, up_matrix : Matrix, x : Buffer,
                      dense_reduction : String? = nil,
                      capture_projections : Bool = false) : NamedTuple(output: Buffer, gate: Buffer?, up: Buffer?)
      return previous_def unless @decode_compare_unfused_expert_swiglu

      gate = matvec(gate_matrix, x, dense_reduction: dense_reduction)
      up = matvec(up_matrix, x, dense_reduction: dense_reduction)
      output = mul(silu(gate), up)
      @decode_compare_reference_dense_swiglu_calls += 1
      {output: output, gate: capture_projections ? gate : nil, up: capture_projections ? up : nil}
    end

    def experts_down_sum(matrix : Matrix, x : Buffer, ids : Buffer, weights : Buffer,
                         reduction : String = "native", capture_down : Bool = false) : NamedTuple(output: Buffer, down: Buffer?)
      return previous_def unless @decode_compare_unfused_expert_down_sum

      down = experts(matrix, x, ids, per_rank_input: true, reduction: reduction)
      routing = alloc(ids.count * 2)
      copy(ids, routing.slice(0, ids.count))
      copy(weights, routing.slice(ids.count, weights.count))
      output = alloc(matrix.rows)
      dispatch("dr3_expert_sum", BackboneResident::SOURCE, [down, routing, output],
        [output.count.to_u64, ids.count.to_u64], output.count)
      @decode_compare_reference_expert_down_sum_calls += 1
      {output: output, down: capture_down ? down : nil}
    end
  end
end

module Drama3DenseDecodeCompare
  include ML::Audio::Drama3
  alias Context = Resident::Context
  alias Buffer = Resident::Buffer

  SOURCE = {{ read_file("#{__DIR__}/support/drama3_prefill_compare.metal") }}
  INPUT_SHA256 = "1fea4a810ba88680ea031c9496c982ad7b8bf32ac23c05e18964c1bf80a1a6f3"
  TRACE_SHA256 = "b83a511d4b91c5930b9e2eed8be0ce541f8296045f9ee8e75e4719566e9f8e79"
  TRACE_MANIFEST_SHA256 = "a82b5940c42ffa92eb3f405b0b79dd91c83fab14ab1d8c5fbf7b115eba035686"
  MODEL_REPOSITORY = "cguldogan/drama-3-mlx-q4"
  MODEL_REVISION = "5331ab509359c0e4a7e353a19e255ddf3e8152a8"
  UPSTREAM_REVISION = "e7488935ec2ebfafd38ef2b58e290d79f3d05dc7"
  CONFIG_SHA256 = "9d52aa7960ba723efcfe47d2ec03f52c5e22e84a6c2a7d773a845db1bf48563c"
  INDEX_SHA256 = "da812cc5b08188ccef048b9074e4cf70c8a1896859c5b492f374d7833e63b855"
  ROWS = 3
  WIDTH = 2048
  TILES = (WIDTH + 1023) // 1024
  RECEIPT_WIDTH = 5
  ATOL = 6e-5_f32
  FINAL_ATOL = 3e-5_f32
  RTOL = 3e-4_f32

  record Capture, row : Int32, stage : String, buffer : Buffer
  record Comparison, row : Int32, stage : String, kind : String, atol : Float32, slot : Int32

  def self.sha(path : String) : String
    digest = Digest::SHA256.new
    File.open(path, "rb") do |io|
      bytes = Bytes.new(1024 * 1024)
      while (count = io.read(bytes)) > 0
        digest.update(bytes[0, count])
      end
    end
    digest.hexfinal
  end

  def self.j(value) : JSON::Any
    JSON.parse(value.to_json)
  end

  def self.number(value : JSON::Any, label : String) : Float32
    raw = value.raw
    number = case raw
             when Float64 then raw.to_f32
             when Int64   then raw.to_f32
             else              raise ArgumentError.new("#{label}: expected a JSON number")
             end
    raise ArgumentError.new("#{label}: non-finite value") unless number.finite?
    number
  end

  def self.expected_stages(layer_count : Int32) : Array(String)
    stages = [] of String
    layer_count.times do |layer|
      stages << "backbone.layers.#{layer}.attention_residual"
      stages << "backbone.layers.#{layer}.hidden"
    end
    stages << "backbone.hidden"
    stages
  end

  def self.reference_name(row : Int32, stage : String) : String
    suffix = stage == "backbone.hidden" ? "hidden" : stage.sub("backbone.", "")
    "backbone.row#{row}.#{suffix}"
  end

  def self.pin_inputs(path : String, raw : String, config : BackboneConfig) : Array(Array(Float32))
    raise "input fixture SHA mismatch" unless Digest::SHA256.hexdigest(raw) == INPUT_SHA256
    fixture = JSON.parse(raw)
    raise "input fixture format mismatch" unless fixture["format"].as_s == "drama3-dense-backbone-inputs-v1"
    raise "input fixture provenance mismatch" unless fixture["provenance"].as_s ==
      "Pinned checkpoint quantized embedding via mlx.nn.QuantizedEmbedding, BF16 scales/biases promoted exactly to FP32; first three retained native prompt IDs."
    rows = fixture["input_embeddings"].as_a.map_with_index do |row, row_index|
      values = row.as_a.map_with_index { |value, column| number(value, "input row #{row_index} column #{column}") }
      raise "input fixture row width mismatch" unless values.size == config.hidden && values.size == WIDTH
      values
    end
    raise "input fixture row count mismatch" unless rows.size == ROWS
    raise "input fixture path must be inputs.json" unless File.basename(path) == "inputs.json"
    rows
  end

  def self.pin_references(path : String, input_path : String, config : BackboneConfig) : Array(NamedTuple(name: String, values: Array(Float32)))
    raise "pinned trace SHA mismatch" unless sha(path) == TRACE_SHA256
    manifest_path = path + ".manifest.json"
    raise "pinned trace manifest SHA mismatch" unless sha(manifest_path) == TRACE_MANIFEST_SHA256
    manifest = JSON.parse(File.read(manifest_path))
    raise "trace manifest type" unless manifest["kind"].as_s == "actual_pinned_q4_backbone_replay"
    raise "trace manifest trace hash" unless manifest["trace_sha256"].as_s == TRACE_SHA256
    raise "trace manifest input hash" unless manifest["inputs"]["sha256"].as_s == INPUT_SHA256
    raise "trace manifest input path" unless manifest["inputs"]["path"].as_s == File.basename(input_path)
    raise "trace manifest row/trace counts" unless manifest["row_count"].as_i == ROWS && manifest["trace_count"].as_i == ROWS * (2 * config.layers + 1)
    raise "trace manifest model identity" unless manifest["model"]["repository"].as_s == MODEL_REPOSITORY &&
      manifest["model"]["revision"].as_s == MODEL_REVISION
    raise "trace manifest upstream identity" unless manifest["upstream"]["mlx-audio"]["revision"].as_s == UPSTREAM_REVISION
    raise "trace manifest model file hashes" unless manifest["model_files"]["config.json"].as_s == CONFIG_SHA256 &&
      manifest["model_files"]["model.safetensors.index.json"].as_s == INDEX_SHA256
    raise "trace manifest embedded trace hash" unless manifest["trace_sha256"].as_s == TRACE_SHA256
    raise "trace fixture path must be mlx-backbone-no-capture.json" unless File.basename(path) == "mlx-backbone-no-capture.json"

    reference = JSON.parse(File.read(path))
    raise "upstream trace format mismatch" unless reference["format"].as_s == "drama3-reference-traces-v1"
    stages = expected_stages(config.layers)
    expected_names = Array(String).new(ROWS * stages.size) do |index|
      row = index // stages.size
      stage = stages[index % stages.size]
      reference_name(row, stage)
    end
    names_in_manifest = manifest["trace_names"].as_a.map(&.as_s)
    raise "trace manifest has duplicate names" unless names_in_manifest.uniq.size == names_in_manifest.size
    raise "trace manifest names differ from expected boundaries" unless names_in_manifest.to_set == expected_names.to_set

    seen = Set(String).new
    records = reference["traces"].as_a.map do |record|
      name = record["name"].as_s
      raise "duplicate upstream trace #{name}" unless seen.add?(name)
      raise "unexpected upstream trace #{name}" unless expected_names.includes?(name)
      raise "upstream trace dtype/shape #{name}" unless record["dtype"].as_s == "float32" &&
        record["shape"].as_a.map(&.as_i) == [config.hidden]
      values = record["values"].as_a.map_with_index do |value, index|
        number(value, "upstream trace #{name} index #{index}")
      end
      raise "upstream trace width #{name}" unless values.size == config.hidden
      {name: name, values: values}
    end
    raise "upstream trace set incomplete" unless seen == expected_names.to_set && records.size == ROWS * stages.size
    records
  end

  def self.configure_environment(expert_down_sum_mode : Bool = false, expert_swiglu_mode : Bool = false) : Nil
    ENV.keys.each { |key| ENV.delete(key) if key.starts_with?("DRAMA3_") }
    {
      "DRAMA3_BACKEND"                    => "metal",
      "DRAMA3_COGNIGRAPH_MAX_INFLIGHT"    => "2",
      "DRAMA3_MOE_BATCH"                  => "1",
      "DRAMA3_BACKBONE_DENSE_REDUCTION"   => "tree",
      "DRAMA3_DENSE_REDUCTION"            => "tree",
      "DRAMA3_BACKBONE_DENSE_FUSION"      => "0",
      "DRAMA3_DELTA_TRANSPOSE"            => "1",
      "DRAMA3_EXPERT_DOWN_REDUCTION"      => "native",
      "DRAMA3_EXPERT_DOWN_SUM_FUSION"     => expert_down_sum_mode ? "1" : "0",
      "DRAMA3_EXPERT_SWIGLU_FUSION"       => expert_swiglu_mode ? "1" : "0",
      "DRAMA3_EXPERT_THREADS"             => "64",
      "DRAMA3_EXPERT_PACKED_ROWS"         => "0",
      "DRAMA3_METAL_TIMELINE"             => "0",
      "DRAMA3_METAL_RESOURCES"            => "0",
      "DRAMA3_HOST_PROFILE"               => "0",
      "DRAMA3_MEMORY_PROFILE"             => "0",
      "DRAMA3_ALLOCATION_PROFILE"         => "0",
    }.each { |key, value| ENV[key] = value }
    ENV.delete("DRAMA3_PROFILE")
    Profile.configure_from_env!
    raise "profiling must be disabled" if Profile.enabled?
  end

  def self.compare(ctx : Context, actual : Buffer, reference : Buffer, parts : Buffer,
                   output : Buffer, atol : Float32) : Nil
    raise "comparison width mismatch" unless actual.count == reference.count && actual.count == WIDTH
    ctx.dispatch("dr3_prefill_compare", SOURCE, [actual, reference, parts],
      [actual.count.to_u64, atol.unsafe_as(UInt32).to_u64, RTOL.unsafe_as(UInt32).to_u64], TILES)
    ctx.dispatch("dr3_prefill_compare_reduce", SOURCE, [parts, output], [TILES.to_u64], 1, 1)
  end

  def self.metric(values : Array(Float32), slot : Int32) : Array(Float32)
    Array(Float32).new(RECEIPT_WIDTH) { |column| values[slot * RECEIPT_WIDTH + column] }
  end

  def self.metric_json(values : Array(Float32)) : JSON::Any
    JSON::Any.new({
      "bit_differences" => j(values[0].to_i64),
      "tolerance_failures" => j(values[1].to_i64),
      "max_abs" => j(values[2]),
      "max_ratio" => j(values[3]),
      "nonfinite" => j(values[4].to_i64),
    } of String => JSON::Any)
  end

  def self.metric_equal?(left : Array(Float32), right : Array(Float32)) : Bool
    RECEIPT_WIDTH.times.all? { |index| left[index].unsafe_as(UInt32) == right[index].unsafe_as(UInt32) }
  end

  def self.exact?(values : Array(Float32)) : Bool
    values.all?(&.finite?) && values[0] == 0_f32 && values[1] == 0_f32 &&
      values[2] == 0_f32 && values[3] == 0_f32 && values[4] == 0_f32
  end

  def self.upstream_pass?(values : Array(Float32)) : Bool
    values.all?(&.finite?) && values[1] == 0_f32 && values[4] == 0_f32
  end

  def self.run : Nil
    context = nil.as(Context?)
    usage = "usage: drama3_dense_decode_compare MODEL_DIR INPUTS_JSON MLX_SCALAR_TRACE OUTPUT_JSON [--expert-down-sum|--expert-swiglu]"
    raise usage unless ARGV.size == 4 || (ARGV.size == 5 && ARGV[4].in?("--expert-down-sum", "--expert-swiglu"))
    expert_down_sum_mode = ARGV[4]? == "--expert-down-sum"
    expert_swiglu_mode = ARGV[4]? == "--expert-swiglu"
    dense_mode = !expert_down_sum_mode && !expert_swiglu_mode
    model_dir, input_path, trace_path, output_path = ARGV.first(4).map { |value| File.expand_path(value) }
    raise "refusing existing output" if File.exists?(output_path)
    config_path = File.join(model_dir, "config.json")
    index_path = File.join(model_dir, "model.safetensors.index.json")
    raise "model config SHA mismatch" unless sha(config_path) == CONFIG_SHA256
    raise "model index SHA mismatch" unless sha(index_path) == INDEX_SHA256
    config = BackboneConfig.new(JSON.parse(File.read(config_path)))
    raise "pinned model geometry mismatch" unless config.hidden == WIDTH && config.layers == 40 && config.full_interval == 4

    input_bytes = File.read(input_path)
    inputs = pin_inputs(input_path, input_bytes, config)
    references = pin_references(trace_path, input_path, config)
    reference_records = {} of String => Array(Float32)
    references.each { |record| reference_records[record[:name]] = record[:values] }
    references = [] of NamedTuple(name: String, values: Array(Float32))

    configure_environment(expert_down_sum_mode, expert_swiglu_mode)
    store = WeightStore.new(model_dir)
    ctx = Context.new
    context = ctx
    raise "decode comparator requires graph depth 2" unless ctx.graph_stats["depth"] == 2
    raise "decode comparator requires tree dense reduction" unless ctx.dense_reduction == "tree"
    raise "decode comparator requires 64 expert threads" unless ctx.expert_threads == 64
    raise "decode comparator requires packed rows disabled" if ctx.expert_packed_rows_enabled
    raise "dense comparison must freeze expert-down-sum fusion off" if !expert_down_sum_mode && ctx.expert_down_sum_fusion_enabled
    raise "diagnostic unfused override must start disabled" if ctx.decode_compare_unfused_expert_down_sum
    raise "SwiGLU selector mismatch" unless ctx.expert_swiglu_fusion_enabled == expert_swiglu_mode
    raise "SwiGLU diagnostic override must start disabled" if ctx.decode_compare_unfused_expert_swiglu
    baseline = BackboneResident.new(config, store, ctx, ROWS)
    raise "baseline fusion selector not frozen off" if baseline.dense_fusion_enabled
    raise "baseline MoE batching selector mismatch" unless baseline.batch_experts
    raise "baseline expert-down reduction mismatch" unless baseline.expert_down_reduction == "native"
    raise "expert-down-sum selector not frozen on" if expert_down_sum_mode && !ctx.expert_down_sum_fusion_enabled
    ENV["DRAMA3_BACKBONE_DENSE_FUSION"] = dense_mode ? "1" : "0"
    candidate = BackboneResident.new(config, store, ctx, ROWS)
    unless dense_mode
      raise "candidate dense fusion selector mismatch" if candidate.dense_fusion_enabled
    else
      raise "candidate fusion selector not frozen on" unless candidate.dense_fusion_enabled
    end
    raise "candidate MoE batching selector mismatch" unless candidate.batch_experts
    raise "candidate expert-down reduction mismatch" unless candidate.expert_down_reduction == "native"
    ENV["DRAMA3_BACKBONE_DENSE_FUSION"] = "0"
    raise "baseline selector changed after candidate construction" if baseline.dense_fusion_enabled
    raise "candidate selector changed after environment reset" unless candidate.dense_fusion_enabled == dense_mode
    raise "reduction control mismatch" unless baseline.dense_reduction == "tree" && candidate.dense_reduction == "tree"

    reference_buffers = {} of String => Buffer
    reference_records.each do |name, values|
      reference_buffers[name] = ctx.upload(values)
    end
    input_buffers = inputs.map { |values| ctx.upload(values) }
    baseline_captures = [] of Capture
    candidate_captures = [] of Capture
    expert_down_sum_row_proof = [] of JSON::Any
    expert_swiglu_row_proof = [] of JSON::Any
    active_row = 0
    baseline.buffer_trace = ->(name : String, buffer : Buffer) {
      baseline_captures << Capture.new(active_row, name, buffer)
      nil
    }
    candidate.buffer_trace = ->(name : String, buffer : Buffer) {
      candidate_captures << Capture.new(active_row, name, buffer)
      nil
    }
    stages = expected_stages(config.layers)
    input_buffers.each_with_index do |input, row|
      active_row = row
      baseline_dispatch_before = ctx.expert_down_sum_fused_dispatches
      baseline_reference_before = ctx.decode_compare_reference_expert_down_sum_calls
      baseline_swiglu_before = {ctx.expert_swiglu_fused_dispatches, ctx.dense_swiglu_fused_dispatches,
                               ctx.decode_compare_reference_expert_swiglu_calls, ctx.decode_compare_reference_dense_swiglu_calls}
      if expert_down_sum_mode
        ctx.decode_compare_unfused_expert_down_sum = true
        baseline.step(input)
        ctx.decode_compare_unfused_expert_down_sum = false
      elsif expert_swiglu_mode
        ctx.decode_compare_unfused_expert_swiglu = true
        baseline.step(input)
        ctx.decode_compare_unfused_expert_swiglu = false
      else
        baseline.step(input)
      end
      baseline_dispatch_delta = ctx.expert_down_sum_fused_dispatches - baseline_dispatch_before
      baseline_reference_delta = ctx.decode_compare_reference_expert_down_sum_calls - baseline_reference_before
      if expert_down_sum_mode
        raise "baseline fused expert-down-sum dispatches after row #{row}" unless baseline_dispatch_delta == 0
        raise "baseline unfused reference calls after row #{row}" unless baseline_reference_delta == config.layers
      end
      if expert_swiglu_mode
        raise "baseline executed fused SwiGLU after row #{row}" unless
          ctx.expert_swiglu_fused_dispatches == baseline_swiglu_before[0] &&
          ctx.dense_swiglu_fused_dispatches == baseline_swiglu_before[1]
        raise "baseline did not execute all unfused SwiGLU calls after row #{row}" unless
          ctx.decode_compare_reference_expert_swiglu_calls - baseline_swiglu_before[2] == config.layers &&
          ctx.decode_compare_reference_dense_swiglu_calls - baseline_swiglu_before[3] == config.layers
      end

      candidate_dispatch_before = ctx.expert_down_sum_fused_dispatches
      candidate_reference_before = ctx.decode_compare_reference_expert_down_sum_calls
      candidate_swiglu_before = {ctx.expert_swiglu_fused_dispatches, ctx.dense_swiglu_fused_dispatches,
                                ctx.decode_compare_reference_expert_swiglu_calls, ctx.decode_compare_reference_dense_swiglu_calls}
      candidate.step(input)
      candidate_dispatch_delta = ctx.expert_down_sum_fused_dispatches - candidate_dispatch_before
      candidate_reference_delta = ctx.decode_compare_reference_expert_down_sum_calls - candidate_reference_before
      if expert_down_sum_mode
        raise "candidate fused expert-down-sum dispatches after row #{row}" unless candidate_dispatch_delta == config.layers
        raise "candidate unexpectedly used unfused reference after row #{row}" unless candidate_reference_delta == 0
        expert_down_sum_row_proof << j({
          "row" => row,
          "baseline_fused_dispatch_delta" => baseline_dispatch_delta,
          "baseline_reference_call_delta" => baseline_reference_delta,
          "candidate_fused_dispatch_delta" => candidate_dispatch_delta,
          "candidate_reference_call_delta" => candidate_reference_delta,
        })
      end
      if expert_swiglu_mode
        expert_delta = ctx.expert_swiglu_fused_dispatches - candidate_swiglu_before[0]
        dense_delta = ctx.dense_swiglu_fused_dispatches - candidate_swiglu_before[1]
        raise "candidate did not execute all fused SwiGLU calls after row #{row}" unless
          expert_delta == config.layers && dense_delta == config.layers
        raise "candidate unexpectedly executed an unfused SwiGLU override after row #{row}" unless
          ctx.decode_compare_reference_expert_swiglu_calls == candidate_swiglu_before[2] &&
          ctx.decode_compare_reference_dense_swiglu_calls == candidate_swiglu_before[3]
        expert_swiglu_row_proof << j({row: row, candidate_expert_fused_dispatches: expert_delta,
          candidate_dense_fused_dispatches: dense_delta, baseline_expert_reference_calls: config.layers,
          baseline_dense_reference_calls: config.layers, baseline_fused_dispatches: 0})
      end
      expected_count = (row + 1) * stages.size
      raise "baseline boundary count after row #{row}" unless baseline_captures.size == expected_count
      raise "candidate boundary count after row #{row}" unless candidate_captures.size == expected_count
      expected_row_stages = baseline_captures.last(stages.size).map(&.stage)
      raise "baseline boundary order mismatch on row #{row}" unless expected_row_stages == stages
      candidate_row_stages = candidate_captures.last(stages.size).map(&.stage)
      raise "candidate boundary order mismatch on row #{row}" unless candidate_row_stages == stages
    end
    if expert_down_sum_mode
      expected_dispatches = (ROWS * config.layers).to_i64
      raise "expert-down-sum reference call count mismatch" unless ctx.decode_compare_reference_expert_down_sum_calls == expected_dispatches
      raise "expert-down-sum fused dispatch count mismatch" unless ctx.expert_down_sum_fused_dispatches == expected_dispatches
      raise "expert-down-sum per-row proof count mismatch" unless expert_down_sum_row_proof.size == ROWS
    end
    if expert_swiglu_mode
      expected_dispatches = (ROWS * config.layers).to_i64
      raise "SwiGLU reference call count mismatch" unless
        ctx.decode_compare_reference_expert_swiglu_calls == expected_dispatches &&
        ctx.decode_compare_reference_dense_swiglu_calls == expected_dispatches
      raise "SwiGLU fused dispatch count mismatch" unless ctx.expert_swiglu_fused_dispatches == expected_dispatches &&
        ctx.dense_swiglu_fused_dispatches == expected_dispatches && expert_swiglu_row_proof.size == ROWS
    end
    expected_capture_count = ROWS * stages.size
    raise "baseline boundary count mismatch" unless baseline_captures.size == expected_capture_count
    raise "candidate boundary count mismatch" unless candidate_captures.size == expected_capture_count
    raise "baseline capture identity mismatch" unless baseline_captures.map { |capture| {capture.row, capture.stage} } ==
      candidate_captures.map { |capture| {capture.row, capture.stage} }

    before_comparison = ctx.graph_stats
    raise "unexpected submission/wait before comparison" unless before_comparison["submitted"] == 0 &&
      before_comparison["completion_waits"] == 0 && before_comparison["pending"] == 0
    raise "unexpected readback before comparison" unless ctx.readbacks.values.all?(&.zero?)

    comparison_count = expected_capture_count * 3
    metrics = ctx.alloc(comparison_count * RECEIPT_WIDTH)
    parts = ctx.alloc(TILES * RECEIPT_WIDTH)
    comparisons = [] of Comparison
    slot = 0
    expected_capture_count.times do |index|
      left = baseline_captures[index]
      right = candidate_captures[index]
      row, stage = left.row, left.stage
      raise "capture map out of order" unless row == index // stages.size && stage == stages[index % stages.size]
      reference_name = reference_name(row, stage)
      reference = reference_buffers[reference_name]? || raise "missing pinned trace buffer #{reference_name}"
      atol = stage == "backbone.hidden" ? FINAL_ATOL : ATOL
      [
        {"baseline_vs_candidate", right.buffer, left.buffer},
        {"baseline_vs_mlx", left.buffer, reference},
        {"candidate_vs_mlx", right.buffer, reference},
      ].each do |kind, actual, expected|
        output = metrics.slice(slot * RECEIPT_WIDTH, RECEIPT_WIDTH)
        compare(ctx, actual, expected, parts, output, atol)
        comparisons << Comparison.new(row, stage, kind, atol, slot)
        slot += 1
      end
    end
    raise "comparison receipt count mismatch" unless slot == comparison_count
    raise "submission/readback before terminal drain" unless ctx.graph_stats["submitted"] == 0 &&
      ctx.graph_stats["completion_waits"] == 0 && ctx.readbacks.values.all?(&.zero?)

    graph_before_drain = ctx.graph_stats.dup
    ctx.finish
    graph_after_drain = ctx.graph_stats.dup
    raise "expected one graph submit and terminal wait" unless graph_after_drain["submitted"] == 1 &&
      graph_after_drain["completed"] == 1 && graph_after_drain["completion_waits"] == 1 &&
      graph_after_drain["pending"] == 0 && graph_after_drain["depth"] == 2
    raw_metrics = ctx.read(metrics, "terminal")
    graph_after_read = ctx.graph_stats.dup
    raise "receipt read introduced another wait" unless graph_after_read == graph_after_drain
    raise "unexpected control/diagnostic/terminal ledger" unless ctx.readbacks["control"] == 4 &&
      ctx.readbacks["diagnostic"] == 0 && ctx.readbacks["terminal"] == comparison_count * RECEIPT_WIDTH * 4

    case_reports = [] of JSON::Any
    native_exact_pass = true
    upstream_metrics_identical = true
    strict_upstream_pass = true
    first_native_difference = nil.as(JSON::Any?)
    first_baseline_upstream_failure = nil.as(JSON::Any?)
    first_candidate_upstream_failure = nil.as(JSON::Any?)
    expected_capture_count.times do |case_index|
      native_cmp = comparisons[case_index * 3]
      baseline_cmp = comparisons[case_index * 3 + 1]
      candidate_cmp = comparisons[case_index * 3 + 2]
      native_values = metric(raw_metrics, native_cmp.slot)
      baseline_values = metric(raw_metrics, baseline_cmp.slot)
      candidate_values = metric(raw_metrics, candidate_cmp.slot)
      native_pass = exact?(native_values)
      native_exact_pass &&= native_pass
      unless native_pass
        first_native_difference ||= j({"row" => native_cmp.row, "stage" => native_cmp.stage})
      end
      same_upstream_metrics = metric_equal?(baseline_values, candidate_values)
      upstream_metrics_identical &&= same_upstream_metrics
      baseline_pass = upstream_pass?(baseline_values)
      candidate_pass = upstream_pass?(candidate_values)
      strict_upstream_pass &&= baseline_pass && candidate_pass
      unless baseline_pass
        first_baseline_upstream_failure ||= j({"row" => baseline_cmp.row, "stage" => baseline_cmp.stage})
      end
      unless candidate_pass
        first_candidate_upstream_failure ||= j({"row" => candidate_cmp.row, "stage" => candidate_cmp.stage})
      end
      record_json = {
        "row" => j(native_cmp.row), "stage" => j(native_cmp.stage),
        "atol" => j(native_cmp.atol), "rtol" => j(RTOL),
        "baseline_vs_candidate" => metric_json(native_values),
        "baseline_vs_mlx" => metric_json(baseline_values),
        "candidate_vs_mlx" => metric_json(candidate_values),
        "native_exact_pass" => j(native_pass),
        "upstream_metrics_identical" => j(same_upstream_metrics),
        "baseline_upstream_pass" => j(baseline_pass),
        "candidate_upstream_pass" => j(candidate_pass),
      } of String => JSON::Any
      case_reports << JSON::Any.new(record_json)
    end

    root = File.expand_path("..", __DIR__)
    source_hashes = {
      "harness" => sha(__FILE__),
      "comparison_metal" => Digest::SHA256.hexdigest(SOURCE),
      "resident_cr" => sha(File.join(root, "src/ml/audio/drama3/resident.cr")),
      "backbone_resident_cr" => sha(File.join(root, "src/ml/audio/drama3/backbone_resident.cr")),
      "resident_metal_embedded" => Digest::SHA256.hexdigest(Resident::SOURCE),
      "backbone_resident_metal_embedded" => Digest::SHA256.hexdigest(BackboneResident::SOURCE),
    }
    report = {
      "format" => j("drama3-dense-decode-comparison-v1"),
      "model" => j({
        "path" => model_dir, "repository" => MODEL_REPOSITORY, "revision" => MODEL_REVISION,
        "config_sha256" => CONFIG_SHA256, "index_sha256" => INDEX_SHA256,
        "input_rows" => ROWS, "hidden_width" => WIDTH, "layers" => config.layers,
        "full_attention_interval" => config.full_interval,
        "linear_attention_layers" => config.layers - (config.layers // config.full_interval),
      }),
      "input" => j({"path" => input_path, "sha256" => INPUT_SHA256,
        "provenance" => "pinned three-row dense-20261014 input_embeddings; row order is file order"}),
      "reference" => j({"path" => trace_path, "sha256" => TRACE_SHA256,
        "manifest_path" => trace_path + ".manifest.json", "manifest_sha256" => TRACE_MANIFEST_SHA256,
        "repository" => "https://github.com/cguldogan/mlx-audio", "revision" => UPSTREAM_REVISION,
        "trace_count" => ROWS * stages.size, "shard_sha256_recorded" => false}),
      "selectors" => j({"baseline_dense_fusion" => false, "candidate_dense_fusion" => true,
        "frozen_after_construction" => true, "dense_reduction" => "tree", "moe_batch" => true,
        "expert_down_reduction" => "native", "expert_threads" => 64, "queue_depth" => 2,
        "profile" => false, "timeline" => false}),
      "source_sha256" => j(source_hashes),
      "capture" => j({"baseline_boundaries" => baseline_captures.size,
        "candidate_boundaries" => candidate_captures.size, "comparisons" => comparison_count,
        "comparison_order" => "row-major; layer attention_residual, layer hidden, final hidden; native pair then each MLX arm"}),
      "result" => j({
        "candidate_native_exact_pass" => native_exact_pass,
        "upstream_metrics_identical" => upstream_metrics_identical,
        "strict_upstream_pass" => strict_upstream_pass,
        "first_native_difference" => first_native_difference,
        "first_baseline_upstream_failure" => first_baseline_upstream_failure,
        "first_candidate_upstream_failure" => first_candidate_upstream_failure,
      }),
      "readback_policy" => j("all 243 boundary captures remain device-resident; one terminal read contains only compact five-float comparison receipts"),
      "graph_before_drain" => j(graph_before_drain),
      "graph_after_drain" => j(graph_after_drain),
      "graph_after_receipt_read" => j(graph_after_read),
      "readbacks" => j(ctx.readbacks),
      "cases" => JSON::Any.new(case_reports),
    } of String => JSON::Any
    if expert_down_sum_mode
      report["format"] = j("drama3-expert-down-sum-decode-comparison-v1")
      report["mode"] = j("expert_down_sum")
      report["selectors"] = j({
        "baseline_dense_fusion" => false, "candidate_dense_fusion" => false,
        "frozen_after_construction" => true, "dense_reduction" => "tree", "moe_batch" => true,
        "expert_down_reduction" => "native", "expert_threads" => 64, "queue_depth" => 2,
        "expert_down_sum_fusion" => ctx.expert_down_sum_fusion_enabled,
        "baseline_unfused_override" => true, "profile" => false, "timeline" => false,
      })
      report["expert_down_sum_dispatch_proof"] = j({
        "reference_unfused_calls" => ctx.decode_compare_reference_expert_down_sum_calls,
        "baseline_fused_dispatches" => 0_i64,
        "candidate_fused_dispatches" => ctx.expert_down_sum_fused_dispatches,
        "expected_candidate_dispatches_per_row" => config.layers,
        "expected_total_dispatches" => ROWS * config.layers,
        "per_row" => expert_down_sum_row_proof,
      })
    end
    if expert_swiglu_mode
      report["format"] = j("drama3-expert-swiglu-decode-comparison-v1")
      report["mode"] = j("expert_swiglu")
      report["selectors"] = j({baseline_dense_fusion: false, candidate_dense_fusion: false,
        frozen_after_construction: true, dense_reduction: "tree", moe_batch: true,
        expert_down_reduction: "native", expert_threads: 64, queue_depth: 2,
        expert_swiglu_fusion: ctx.expert_swiglu_fusion_enabled, expert_down_sum_fusion: false,
        baseline_unfused_override: true, profile: false, timeline: false})
      report["expert_swiglu_dispatch_proof"] = j({
        expert_reference_calls: ctx.decode_compare_reference_expert_swiglu_calls,
        dense_reference_calls: ctx.decode_compare_reference_dense_swiglu_calls,
        candidate_expert_fused_dispatches: ctx.expert_swiglu_fused_dispatches,
        candidate_dense_fused_dispatches: ctx.dense_swiglu_fused_dispatches,
        expected_per_family_per_row: config.layers, per_row: expert_swiglu_row_proof})
    end
    File.write(output_path, JSON::Any.new(report).to_pretty_json + "\n")
    puts({output: output_path, native_exact_pass: native_exact_pass,
      upstream_metrics_identical: upstream_metrics_identical, strict_upstream_pass: strict_upstream_pass,
      first_native_difference: first_native_difference, first_baseline_upstream_failure: first_baseline_upstream_failure,
      first_candidate_upstream_failure: first_candidate_upstream_failure,
      comparisons: comparison_count, graph: graph_after_read, readbacks: ctx.readbacks}.to_json)
    raise "candidate differs from baseline at a decode boundary" unless native_exact_pass
    raise "baseline/candidate MLX receipts differ" unless upstream_metrics_identical
  rescue ex
    context.try(&.abort)
    raise ex
  end
end

Drama3DenseDecodeCompare.run
{% else %}
  abort "build with -Ddrama3_metal on macOS"
{% end %}
