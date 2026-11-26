require "digest/sha256"
require "../src/ml/audio/drama3/backbone_resident"

{% if flag?(:drama3_metal) && flag?(:darwin) && !flag?(:cpu_only) %}
  include ML::Audio::Drama3
  abort "usage: drama3_backbone_gpu_trace MODEL_DIR INPUTS_JSON OUTPUT_JSON" unless ARGV.size == 3
  model_path, input_path, output_path = ARGV
  Profile.configure_from_env!
  raise ArgumentError.new("profiling must be disabled for single-drain capture") if Profile.enabled?
  input_bytes = File.read(input_path)
  input_json = JSON.parse(input_bytes)
  config = BackboneConfig.new(JSON.parse(File.read(File.join(model_path, "config.json"))))
  inputs = input_json["input_embeddings"].as_a.map do |row|
    values = row.as_a.map do |value|
      number = case raw = value.raw
               when Float64 then raw.to_f32
               when Int64   then raw.to_f32
               else              raise ArgumentError.new("input contains a non-numeric value")
               end
      raise ArgumentError.new("input contains a non-finite value") unless number.finite?
      number
    end
    raise ArgumentError.new("input width mismatch") unless values.size == config.hidden
    values
  end
  raise ArgumentError.new("input rows must not be empty") if inputs.empty?
  store = WeightStore.new(model_path)
  ctx = Resident::Context.new
  raise ArgumentError.new("single-drain capture requires graph depth 0") unless ctx.graph_stats["depth"] == 0
  model = BackboneResident.new(config, store, ctx, 1024)
  captured = [] of NamedTuple(name: String, buffer: Resident::Buffer)
  row = 0
  # Each observed boundary owns a fresh GPU buffer. Retain those buffers and
  # serialize them only after every row has been enqueued and drained.
  model.buffer_trace = ->(name : String, buffer : Resident::Buffer) {
    captured << {name: name.sub("backbone.", "backbone.row#{row}."), buffer: buffer}
    nil
  }
  inputs.each_with_index do |input, index|
    row = index
    model.step(ctx.upload(input))
  end
  expected_count = inputs.size * (2 * config.layers + 1)
  raise ArgumentError.new("trace boundary count mismatch") unless captured.size == expected_count && captured.map(&.[:name]).uniq.size == expected_count
  raise ArgumentError.new("readback before final drain") unless ctx.readbacks.values.all?(&.zero?)
  before = ctx.graph_stats
  raise ArgumentError.new("submission or wait before final drain") unless before["submitted"] == 0 && before["completion_waits"] == 0
  ctx.finish
  after = ctx.graph_stats
  raise ArgumentError.new("expected one terminal completion wait") unless after["submitted"] == 1 && after["completion_waits"] == 1
  records = captured.map do |entry|
    buffer = entry[:buffer]
    {name: entry[:name], shape: [buffer.count], dtype: "float32", values: ctx.read(buffer, "terminal")}
  end
  final = ctx.graph_stats
  raise ArgumentError.new("terminal capture added waits") unless final["completion_waits"] == after["completion_waits"]
  raise ArgumentError.new("diagnostic activation readback") unless ctx.readbacks["diagnostic"] == 0
  File.write(output_path, {
    format: "drama3-reference-traces-v1", traces: records,
    input_sha256: Digest::SHA256.hexdigest(input_bytes),
    execution: {expert_down_reduction: model.expert_down_reduction, dense_reduction: model.dense_reduction, batch_experts: model.batch_experts},
    graph_before_drain: before, graph_after_drain: after, graph_after_reads: final,
    readbacks: ctx.readbacks,
  }.to_json)
{% else %}
  abort "build with -Ddrama3_metal on Darwin"
{% end %}
