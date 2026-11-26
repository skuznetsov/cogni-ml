require "json"
require "digest/sha256"
require "set"
require "../src/ml/audio/drama3/backbone_resident"
require "./support/drama3_moe_provenance"

{% if flag?(:drama3_metal) && flag?(:darwin) && !flag?(:cpu_only) %}
  include ML::Audio::Drama3

  MODEL_REPOSITORY    = "cguldogan/drama-3-mlx-q4"
  MODEL_REVISION      = "5331ab509359c0e4a7e353a19e255ddf3e8152a8"
  UPSTREAM_REPOSITORY = "https://github.com/cguldogan/mlx-audio"
  UPSTREAM_REVISION   = "e7488935ec2ebfafd38ef2b58e290d79f3d05dc7"
  TARGET_ROW          =  1
  TARGET_LAYER        = 39

  alias MoeStage = NamedTuple(shape: Array(Int32), dtype: String, values: Vector, capture_kind: String)

  def moe_float(value : JSON::Any) : Float32
    raw = value.raw
    number = case raw
             when Int64   then raw.to_f32
             when Float64 then raw.to_f32
             else              raise ArgumentError.new("MoE fixture contains a non-numeric value")
             end
    raise ArgumentError.new("MoE fixture contains a non-finite value") unless number.finite?
    number
  end

  def moe_stage(ctx : Resident::Context, buffer : Resident::Buffer, kind : String) : MoeStage
    {shape: [buffer.count], dtype: "float32", values: ctx.read(buffer, "terminal"), capture_kind: kind}
  end

  def moe_capture_kind(name : String) : String
    if name.starts_with?("expert_weighted_outputs.") || name.starts_with?("diagnostic_prefix_sums_") || name == "shared_weighted_output"
      "diagnostic_auxiliary"
    else
      "official_in_path"
    end
  end

  def moe_rank_buffers(buffers : Hash(String, Resident::Buffer), prefix : String, ranks : Int32) : Array(Resident::Buffer)
    Array(Resident::Buffer).new(ranks) do |rank|
      buffers["#{prefix}.rank_#{rank}"]? || raise ArgumentError.new("missing MoE capture #{prefix}.rank_#{rank}")
    end
  end

  abort "usage: drama3_moe_trace MODEL_DIR FULL_INPUTS_JSON UPSTREAM_INPUT_FIXTURE OUTPUT_JSON" unless ARGV.size == 4
  model_path, full_inputs_path, fixture_path, output_path = ARGV
  fixture_bytes = File.read(fixture_path)
  fixture = JSON.parse(fixture_bytes)
  abort "unexpected MoE input fixture format" unless fixture["format"]?.try(&.as_s) == "drama3-moe-input-v1"
  coordinate = fixture["coordinate"]
  abort "MoE fixture must identify row 1, layer 39" unless coordinate["row"].as_i == TARGET_ROW && coordinate["layer"].as_i == TARGET_LAYER
  abort "MoE fixture activation must be post_attention_layernorm" unless fixture["activation_name"].as_s == "post_attention_layernorm"
  abort "MoE fixture activation dtype must be float32" unless fixture["activation_dtype"].as_s == "float32"
  activation_shape = fixture["activation_shape"].as_a
  abort "MoE fixture activation shape must be [2048]" unless activation_shape.size == 1 && activation_shape[0].as_i == 2048

  model_metadata = fixture["model"]?
  upstream_metadata = fixture["upstream"]?
  unless model_metadata && upstream_metadata
    raise ArgumentError.new("MoE fixture must carry pinned model and upstream metadata")
  end
  unless model_metadata["repository"].as_s == MODEL_REPOSITORY && model_metadata["revision"].as_s == MODEL_REVISION
    raise ArgumentError.new("MoE fixture model pin differs from the supported checkpoint")
  end
  unless upstream_metadata["repository"].as_s == UPSTREAM_REPOSITORY && upstream_metadata["revision"].as_s == UPSTREAM_REVISION
    raise ArgumentError.new("MoE fixture upstream pin differs from the supported source")
  end
  raise ArgumentError.new("MoE fixture source hashes are required") unless fixture["source_hashes"]?.try(&.as_h)

  config = BackboneConfig.new(JSON.parse(File.read(File.join(model_path, "config.json"))))
  raise ArgumentError.new("model hidden width must be 2048") unless config.hidden == 2048
  raise ArgumentError.new("model must contain layer 39") unless config.layers > TARGET_LAYER
  full_input_bytes = File.read(full_inputs_path)
  full_inputs = JSON.parse(full_input_bytes)["input_embeddings"].as_a
  raise ArgumentError.new("full input fixture must contain row 1") unless full_inputs.size > TARGET_ROW

  Profile.configure_from_env!
  raise ArgumentError.new("DRAMA3_PROFILE must be disabled for the single-drain capture control") if Profile.enabled?
  store = WeightStore.new(model_path)
  model_weight_provenance = Drama3MoeProvenance.verify!(model_path, store, fixture["source_hashes"].as_h)
  ctx = Resident::Context.new

  # Run a no-detailed-capture control first on the same context and weight
  # store. Keep only the target hidden buffer and terminal outputs; no callback
  # allocates snapshots or reads device memory.
  control_model = BackboneResident.new(config, store, ctx, full_inputs.size.to_i32)
  control_model.moe_trace = nil
  control_outputs = [] of Resident::Buffer
  control_layer_buffers = {} of String => Resident::Buffer
  full_inputs.each_with_index do |row, index|
    if index == TARGET_ROW
      control_model.diagnostic_layer = TARGET_LAYER
      control_model.layer_buffer_trace = ->(name : String, buffer : Resident::Buffer) {
        control_layer_buffers[name] = buffer if name == "hidden_output"
        nil
      }
    else
      control_model.diagnostic_layer = nil
      control_model.layer_buffer_trace = nil
    end
    input = row.as_a.map { |value| moe_float(value) }
    raise ArgumentError.new("full input row width must be 2048") unless input.size == 2048
    control_outputs << control_model.step(ctx.upload(input))
  end
  control_hidden = control_layer_buffers["hidden_output"]? || raise ArgumentError.new("capture control omitted target hidden output")

  model = BackboneResident.new(config, store, ctx, full_inputs.size.to_i32)
  model.diagnostic_layer = TARGET_LAYER
  fixture_directory = File.dirname(File.expand_path(fixture_path))
  output_directory = File.dirname(File.expand_path(output_path))
  raise ArgumentError.new("upstream fixture and native trace must be siblings") unless fixture_directory == output_directory
  full_moe_buffers = {} of String => Resident::Buffer
  full_layer_buffers = {} of String => Resident::Buffer
  full_outputs = [] of Resident::Buffer
  full_batch_path = false
  full_inputs.each_with_index do |row, index|
    if index == TARGET_ROW
      model.diagnostic_layer = TARGET_LAYER
      model.moe_trace_auxiliary = true
      model.moe_trace = ->(name : String, buffer : Resident::Buffer) { full_moe_buffers[name] = buffer; nil }
      model.layer_buffer_trace = ->(name : String, buffer : Resident::Buffer) { full_layer_buffers[name] = buffer; nil }
    else
      model.moe_trace = nil
      model.layer_buffer_trace = nil
      model.diagnostic_layer = nil
    end
    input = row.as_a.map { |value| moe_float(value) }
    raise ArgumentError.new("full input row width must be 2048") unless input.size == 2048
    full_outputs << model.step(ctx.upload(input))
    full_batch_path = model.last_moe_batched if index == TARGET_ROW
  end

  # The isolated run reuses the exact same context and queued command stream;
  # the only host boundary is the final drain followed by these readbacks.
  model.diagnostic_layer = TARGET_LAYER
  isolated_buffers = {} of String => Resident::Buffer
  model.moe_trace = ->(name : String, buffer : Resident::Buffer) { isolated_buffers[name] = buffer; nil }
  model.layer_buffer_trace = nil
  model.moe_trace_auxiliary = true
  activation = fixture["activation"].as_a.map { |value| moe_float(value) }
  raise ArgumentError.new("isolated activation width must be 2048") unless activation.size == 2048
  isolated_output = model.diagnose_moe(ctx.upload(activation), TARGET_LAYER)
  isolated_batch_path = model.last_moe_batched
  raise ArgumentError.new("diagnostic hooks read back before final drain") unless ctx.readbacks.values.all?(&.zero?)
  before_drain = ctx.graph_stats
  ctx.finish
  after_drain = ctx.graph_stats

  raise ArgumentError.new("capture control output count differs from captured run") unless control_outputs.size == full_outputs.size
  captured_hidden = full_layer_buffers["hidden_output"]? || raise ArgumentError.new("captured run omitted target hidden output")
  capture_mismatches = [] of String
  compared_values = 0_i64
  control_outputs.each_with_index do |control_output, row|
    expected = ctx.read(control_output, "terminal")
    actual = ctx.read(full_outputs[row], "terminal")
    compared_values += expected.size.to_i64
    if expected.map(&.unsafe_as(UInt32)) != actual.map(&.unsafe_as(UInt32))
      capture_mismatches << "final_output_row_#{row}"
    end
  end
  expected_hidden = ctx.read(control_hidden, "terminal")
  actual_hidden = ctx.read(captured_hidden, "terminal")
  compared_values += expected_hidden.size.to_i64
  if expected_hidden.map(&.unsafe_as(UInt32)) != actual_hidden.map(&.unsafe_as(UInt32))
    capture_mismatches << "target_layer_hidden"
  end
  capture_control = {
    passed:          capture_mismatches.empty?,
    comparison:      "Float32 bitwise equality",
    count:           control_outputs.size + 1,
    compared_values: compared_values,
    mismatches:      capture_mismatches,
  }

  top_k = config.top_k
  full_stages = {} of String => MoeStage
  full_moe_buffers.each do |name, buffer|
    kind = moe_capture_kind(name)
    full_stages[name] = moe_stage(ctx, buffer, kind)
  end
  full_layer_buffers.each do |name, buffer|
    full_stages[name] = moe_stage(ctx, buffer, "official_in_path")
  end
  full_outputs.each_with_index do |buffer, row|
    full_stages["final_output_row_#{row}"] = moe_stage(ctx, buffer, "official_in_path")
  end

  # Present per-expert captures as explicit rank-major tensors to simplify
  # comparison while retaining the source path's per-rank buffers until drain.
  [
    {source: "expert_gate_outputs", output: "expert_gate_outputs", kind: "official_in_path"},
    {source: "expert_up_outputs", output: "expert_up_outputs", kind: "official_in_path"},
    {source: "expert_activation_outputs", output: "expert_activation_outputs", kind: "official_in_path"},
    {source: "expert_down_outputs", output: "expert_down_outputs", kind: "official_in_path"},
    {source: "expert_weighted_outputs", output: "expert_weighted_outputs", kind: "diagnostic_auxiliary"},
  ].each do |entry|
    ranked = moe_rank_buffers(full_moe_buffers, entry[:source], top_k)
    values = ranked.flat_map { |buffer| ctx.read(buffer, "terminal") }
    full_stages[entry[:output]] = {shape: [top_k, ranked.first.count], dtype: "float32", values: values, capture_kind: entry[:kind]}
  end
  prefix_source = full_batch_path ? "diagnostic_prefix_sums_argpartition_order" : "routed_prefix_sums_argpartition_order"
  prefix_kind = full_batch_path ? "diagnostic_auxiliary" : "official_in_path"
  prefix_values = moe_rank_buffers(full_moe_buffers, prefix_source, top_k).flat_map { |buffer| ctx.read(buffer, "terminal") }
  full_stages["diagnostic_prefix_sums_argpartition_order"] = {shape: [top_k, config.hidden], dtype: "float32", values: prefix_values, capture_kind: prefix_kind}

  isolated_stages = {} of String => MoeStage
  isolated_buffers.each do |name, buffer|
    kind = moe_capture_kind(name)
    isolated_stages[name] = moe_stage(ctx, buffer, kind)
  end
  [
    {source: "expert_gate_outputs", output: "expert_gate_outputs", kind: "official_in_path"},
    {source: "expert_up_outputs", output: "expert_up_outputs", kind: "official_in_path"},
    {source: "expert_activation_outputs", output: "expert_activation_outputs", kind: "official_in_path"},
    {source: "expert_down_outputs", output: "expert_down_outputs", kind: "official_in_path"},
    {source: "expert_weighted_outputs", output: "expert_weighted_outputs", kind: "diagnostic_auxiliary"},
  ].each do |entry|
    ranked = moe_rank_buffers(isolated_buffers, entry[:source], top_k)
    values = ranked.flat_map { |buffer| ctx.read(buffer, "terminal") }
    isolated_stages[entry[:output]] = {shape: [top_k, ranked.first.count], dtype: "float32", values: values, capture_kind: entry[:kind]}
  end
  isolated_prefix_source = isolated_batch_path ? "diagnostic_prefix_sums_argpartition_order" : "routed_prefix_sums_argpartition_order"
  isolated_prefix_kind = isolated_batch_path ? "diagnostic_auxiliary" : "official_in_path"
  isolated_prefixes = moe_rank_buffers(isolated_buffers, isolated_prefix_source, top_k).flat_map { |buffer| ctx.read(buffer, "terminal") }
  isolated_stages["diagnostic_prefix_sums_argpartition_order"] = {shape: [top_k, config.hidden], dtype: "float32", values: isolated_prefixes, capture_kind: isolated_prefix_kind}
  isolated_stages["isolated_output"] = moe_stage(ctx, isolated_output, "official_in_path")

  def rank_map(stage : MoeStage) : Array(NamedTuple(rank: Int32, expert_id: Int32))
    result = stage[:values].map_with_index do |value, rank|
      id = value.to_i32
      raise ArgumentError.new("expert ID is not integral") unless id.to_f32 == value
      {rank: rank.to_i32, expert_id: id}
    end
    raise ArgumentError.new("expert IDs must be unique across selected ranks") unless result.map(&.[:expert_id]).uniq.size == result.size
    result
  end

  full_rank_map = rank_map(full_stages["expert_ids_argpartition_order"])
  isolated_rank_map = rank_map(isolated_stages["expert_ids_argpartition_order"])
  native_sources = {
    "backbone_resident.cr"     => Digest::SHA256.hexdigest(File.read(File.join(__DIR__, "../src/ml/audio/drama3/backbone_resident.cr"))),
    "backbone_resident.metal"  => Digest::SHA256.hexdigest(File.read(File.join(__DIR__, "../src/ml/audio/drama3/kernels/backbone_resident.metal"))),
    "resident.cr"              => Digest::SHA256.hexdigest(File.read(File.join(__DIR__, "../src/ml/audio/drama3/resident.cr"))),
    "resident.metal"           => Digest::SHA256.hexdigest(File.read(File.join(__DIR__, "../src/ml/audio/drama3/kernels/resident.metal"))),
    "native.cr"                => Digest::SHA256.hexdigest(File.read(File.join(__DIR__, "../src/ml/audio/drama3/native.cr"))),
    "drama3_moe_trace.cr"      => Digest::SHA256.hexdigest(File.read(File.join(__DIR__, "drama3_moe_trace.cr"))),
    "drama3_moe_provenance.cr" => Digest::SHA256.hexdigest(File.read(File.join(__DIR__, "support/drama3_moe_provenance.cr"))),
    "full_inputs"              => Digest::SHA256.hexdigest(full_input_bytes),
    "upstream_input_fixture"   => Digest::SHA256.hexdigest(fixture_bytes),
  }
  output = {
    format:                  "drama3-moe-trace-native-v1",
    coordinate:              {row: TARGET_ROW, layer: TARGET_LAYER},
    model:                   model_metadata,
    upstream:                upstream_metadata,
    input_fixture:           {path: File.basename(fixture_path), sha256: Digest::SHA256.hexdigest(fixture_bytes)},
    source_hashes:           {native: native_sources, fixture: fixture["source_hashes"]},
    model_weight_provenance: model_weight_provenance,
    execution:               {batch_experts_requested: model.batch_experts, full_native_batch_path: full_batch_path,
                isolated_batch_path: isolated_batch_path, dense_reduction: model.dense_reduction},
    capture_control: capture_control,
    runs:            [
      {name: "full_native", row: TARGET_ROW, stages: full_stages, expert_rank_to_id: full_rank_map},
      {name: "isolated_upstream_input", row: TARGET_ROW, stages: isolated_stages, expert_rank_to_id: isolated_rank_map},
    ],
    graph_before_drain:        before_drain,
    graph_after_drain:         after_drain,
    graph_after_serialization: ctx.graph_stats,
    readbacks:                 ctx.readbacks,
  }
  File.write(output_path, output.to_json + "\n")
  puts({trace: output_path, row: TARGET_ROW, layer: TARGET_LAYER, isolated: true,
        capture_control: capture_control}.to_json)
  Profile.write_from_env
  exit 2 unless capture_control[:passed]
{% else %}
  abort "build with -Ddrama3_metal on Darwin"
{% end %}
