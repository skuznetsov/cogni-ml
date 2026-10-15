require "json"
require "digest/sha256"
require "../src/ml/audio/drama3/backbone_resident"
require "./support/drama3_moe_provenance"

{% if flag?(:drama3_metal) && flag?(:darwin) && !flag?(:cpu_only) %}
  include ML::Audio::Drama3

  def q4_stage(stages : JSON::Any, name : String, shape : Array(Int32), kind : String) : Vector
    stage = stages[name]
    raise ArgumentError.new("#{name}: shape mismatch") unless stage["shape"].as_a.map(&.as_i) == shape
    raise ArgumentError.new("#{name}: dtype mismatch") unless stage["dtype"].as_s == "float32"
    raise ArgumentError.new("#{name}: capture kind mismatch") unless stage["capture_kind"].as_s == kind
    values = stage["values"].as_a.map do |value|
      number = case raw = value.raw
               when Float64 then raw.to_f32
               when Int64   then raw.to_f32
               else              raise ArgumentError.new("#{name}: non-numeric value")
               end
      raise ArgumentError.new("#{name}: non-finite value") unless number.finite?
      number
    end
    raise ArgumentError.new("#{name}: value count mismatch") unless values.size == shape.reduce(1) { |a, b| a * b }
    values
  end

  def q4_f32_sha(values : Vector) : String
    io = IO::Memory.new
    values.each { |value| io.write_bytes(value, IO::ByteFormat::LittleEndian) }
    Digest::SHA256.hexdigest(io.to_slice)
  end

  abort "usage: drama3_mlx_q4_probe MODEL_DIR UPSTREAM_FULL_MOE_TRACE OUTPUT_JSON" unless ARGV.size == 3
  model_path, reference_path, output_path = ARGV
  reference_bytes = File.read(reference_path)
  reference = JSON.parse(reference_bytes)
  raise ArgumentError.new("unexpected upstream trace format") unless reference["format"].as_s == "drama3-moe-layer-diagnostic-v1"
  raise ArgumentError.new("expected full-run upstream capture") unless reference["mode"].as_s == "full_run"
  raise ArgumentError.new("expected row 1, layer 39") unless reference["coordinate"]["row"].as_i == 1 && reference["coordinate"]["layer"].as_i == 39
  raise ArgumentError.new("upstream capture control failed") unless reference["capture_control"]["passed"].as_bool
  raise ArgumentError.new("model pin mismatch") unless reference["model"]["repository"].as_s == "cguldogan/drama-3-mlx-q4" && reference["model"]["revision"].as_s == "5331ab509359c0e4a7e353a19e255ddf3e8152a8"
  raise ArgumentError.new("upstream pin mismatch") unless reference["upstream"]["repository"].as_s == "https://github.com/cguldogan/mlx-audio" && reference["upstream"]["revision"].as_s == "e7488935ec2ebfafd38ef2b58e290d79f3d05dc7"
  stages = reference["stages"]
  ids_stage = stages["expert_ids_argpartition_order"]
  raise ArgumentError.new("expert ID shape/dtype mismatch") unless ids_stage["shape"].as_a.map(&.as_i) == [8] && ids_stage["dtype"].as_s == "int32"
  ids = ids_stage["values"].as_a.map(&.as_i)
  raise ArgumentError.new("invalid expert IDs") unless ids.size == 8 && ids.uniq.size == 8 && ids.all? { |id| 0 <= id < 256 }
  rank = ids.index(108) || raise ArgumentError.new("expert 108 is not routed")
  rank_map = reference["expert_rank_to_id"].as_a.sort_by { |entry| entry["rank"].as_i }
  raise ArgumentError.new("rank-to-expert map mismatch") unless rank_map.map { |entry| entry["rank"].as_i } == (0...8).to_a && rank_map.map { |entry| entry["expert_id"].as_i } == ids
  {"expert_activation_outputs", "expert_down_outputs"}.each do |name|
    raise ArgumentError.new("#{name}: expert order mismatch") unless stages[name]["expert_ids"].as_a.map(&.as_i) == ids
  end
  activation = q4_stage(stages, "expert_activation_outputs", [8, 512], "upstream_swiglu_return")
  expected = q4_stage(stages, "expert_down_outputs", [8, 2048], "upstream_switch_mlp_return")
  activation_sha = q4_f32_sha(activation)
  # Reject a different activation before creating a Metal context. This probe
  # intentionally targets the retained expert-108 discrepancy, not arbitrary data.
  raise ArgumentError.new("pinned SwiGLU activation mismatch") unless activation_sha == "8b9086fd8e4a2ab31549429513dbe482d8abb289fae70a3be4289318bfd77244"
  Profile.configure_from_env!
  raise ArgumentError.new("profiling must be disabled for the single-drain probe") if Profile.enabled?
  store = WeightStore.new(model_path)
  provenance = Drama3MoeProvenance.verify!(model_path, store, reference["source_hashes"].as_h)
  matrix = store.matrix("backbone.layers.39.mlp.switch_mlp.down_proj.weight", 0)
  raise ArgumentError.new("expected affine Q4 2048x512/group64") unless matrix.rows == 2048 && matrix.cols == 512 && matrix.bits == 4 && matrix.group_size == 64 && matrix.scales && matrix.biases
  ctx = Resident::Context.new
  raise ArgumentError.new("probe requires graph depth 0") unless ctx.graph_stats["depth"] == 0
  x = ctx.upload(activation)
  selected = ctx.upload(ids.map(&.to_f32))
  one_x = x.slice(rank * 512, 512)
  outputs = {
    "native_all" => ctx.experts(matrix, x, selected, per_rank_input: true),
    "mlx_all"    => ctx.experts(matrix, x, selected, per_rank_input: true, reduction: "mlx_q4"),
    "mlx_repeat" => ctx.experts(matrix, x, selected, per_rank_input: true, reduction: "mlx_q4"),
    "native_108" => ctx.expert(matrix, one_x, selected, rank),
    "mlx_108"    => ctx.expert(matrix, one_x, selected, rank, reduction: "mlx_q4"),
  }
  raise ArgumentError.new("readback before drain") unless ctx.readbacks.values.all?(&.zero?)
  before = ctx.graph_stats
  raise ArgumentError.new("submission or wait before terminal drain") unless before["submitted"] == 0 && before["completion_waits"] == 0
  ctx.finish
  after = ctx.graph_stats
  raise ArgumentError.new("expected exactly one terminal completion wait") unless after["completion_waits"] == 1 && after["submitted"] == 1
  values = outputs.transform_values { |buffer| ctx.read(buffer, "terminal") }
  final = ctx.graph_stats
  raise ArgumentError.new("terminal reads added GPU waits") unless final["completion_waits"] == after["completion_waits"]
  raise ArgumentError.new("non-terminal activation readback") unless ctx.readbacks["diagnostic"] == 0 && ctx.readbacks["control"] == 0
  File.write(output_path, {
    format: "drama3-native-mlx-q4-probe-v1",
    model: reference["model"], upstream: reference["upstream"],
    reference_sha256: Digest::SHA256.hexdigest(reference_bytes),
    model_weight_provenance: provenance,
    activation_sha256: activation_sha,
    expert_108_activation_sha256: q4_f32_sha(activation[rank * 512, 512]),
    expert_ids: ids, expert_108_rank: rank,
    shape: {experts: 8, n: 2048, k: 512, bits: 4, group_size: 64},
    reference_outputs_sha256: q4_f32_sha(expected),
    graph_before_drain: before, graph_after_drain: after, graph_after_reads: final,
    readbacks: ctx.readbacks,
    outputs: values,
  }.to_json)
{% else %}
  abort "build with -Ddrama3_metal on Darwin"
{% end %}
