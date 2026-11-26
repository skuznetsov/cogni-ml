require "../src/ml/audio/drama3/backbone_resident"

{% if flag?(:drama3_metal) && flag?(:darwin) && !flag?(:cpu_only) %}
  include ML::Audio::Drama3
  abort "usage: drama3_resident_trace MODEL_DIR INPUTS_JSON OUTPUT_JSON" unless ARGV.size == 3
  model_path, input_path, output_path = ARGV
  Profile.configure_from_env!
  ctx = Resident::Context.new
  config = BackboneConfig.new(JSON.parse(File.read(File.join(model_path, "config.json"))))
  store = WeightStore.new(model_path)
  model = BackboneResident.new(config, store, ctx, 1024)
  inputs = JSON.parse(File.read(input_path))["input_embeddings"].as_a
  records = [] of NamedTuple(name: String, shape: Array(Int32), dtype: String, values: Vector)
  row = 0
  model.trace = ->(name : String, values : Vector) {
    records << {name: name.sub("backbone.", "backbone.row#{row}."), shape: [values.size], dtype: "float32", values: values.dup}
    nil
  }
  inputs.each_with_index do |input, index|
    row = index
    started = Time.instant
    model.step(ctx.upload(input.as_a.map { |v| v.as_f.to_f32 }))
    ctx.finish
    STDERR.puts "row #{index} #{Time.instant - started}"
  end
  File.write(output_path, {format: "drama3-reference-traces-v1", traces: records, readbacks: ctx.readbacks}.to_json)
  Profile.write_from_env
{% else %}
  abort "build with -Ddrama3_metal on Darwin"
{% end %}
