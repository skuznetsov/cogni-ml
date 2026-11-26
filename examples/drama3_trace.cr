require "../src/ml/audio/drama3/backbone"

# Fixed embedding replay avoids sampling/tokenizer differences when comparing
# every layer against a pinned reference. JSON stays outside the hot path.
abort "usage: drama3_trace WEIGHTS_DIR INPUTS_JSON OUTPUT_JSON" unless ARGV.size == 3
model_path, input_path, output_path = ARGV
ML::Audio::Drama3::Profile.configure_from_env!
ML::Audio::Drama3::MetalBackend.enabled?
config = ML::Audio::Drama3::BackboneConfig.new(JSON.parse(File.read(File.join(model_path, "config.json"))))
store = ML::Audio::Drama3::WeightStore.new(model_path)
model = ML::Audio::Drama3::Backbone.new(config, store)
inputs = JSON.parse(File.read(input_path))["input_embeddings"].as_a
records = [] of NamedTuple(name: String, shape: Array(Int32), dtype: String, values: ML::Audio::Drama3::Vector)
row = 0
model.trace = ->(name : String, values : ML::Audio::Drama3::Vector) {
  records << {name: name.sub("backbone.", "backbone.row#{row}."), shape: [values.size], dtype: "float32", values: values.dup}
  nil
}
inputs.each_with_index do |input, index|
  row = index
  started = Time.instant
  model.step(input.as_a.map { |v| v.as_f.to_f32 })
  STDERR.puts "row #{index} #{Time.instant - started}"
end
File.write(output_path, {format: "drama3-reference-traces-v1", traces: records}.to_json)
ML::Audio::Drama3::Profile.write_from_env
