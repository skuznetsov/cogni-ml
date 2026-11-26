require "json"
require "../../../src/ml/audio/drama3/tower"

include ML::Audio::Drama3

abort "usage: native_tower_trace MODEL_DIR INPUTS_JSON OUTPUT_JSON" unless ARGV.size == 3

model_dir = File.expand_path(ARGV[0])
inputs = JSON.parse(File.read(ARGV[1])).as_h
output_path = File.expand_path(ARGV[2])
Profile.configure_from_env!
MetalBackend.enabled?

def json_vector(value : JSON::Any, name : String) : Vector
  value.as_a.map do |item|
    case number = item.raw
    when Float64
      number.to_f32
    when Int64
      number.to_f32
    else
      raise ArgumentError.new("#{name} must contain numeric values")
    end
  end
end

def json_frame(value : JSON::Any, name : String) : Array(Int32)
  value.as_a.map do |item|
    number = item.as_i64
    number.to_i32
  end
end

def argmax(values : Vector) : Int32
  raise ArgumentError.new("cannot sample an empty logit vector") if values.empty?
  best_index = 0
  best_value = values[0]
  1.upto(values.size - 1) do |index|
    if values[index] > best_value
      best_index = index
      best_value = values[index]
    end
  end
  best_index.to_i32
end

config = TowerConfig.from_json(File.read(File.join(model_dir, "config.json")))
tower = Tower.new(config, WeightStore.new(model_dir))
hidden_rows = inputs["tower_hidden_rows"].as_a?.try do |rows|
  rows.map_with_index { |row, index| json_vector(row, "tower_hidden_rows[#{index}]") }
end
prefill_hidden = if hidden_rows
                   hidden_rows[0] || raise ArgumentError.new("tower_hidden_rows has no prefill row")
                 else
                   json_vector(inputs["prefill_hidden"], "prefill_hidden")
                 end
decode_hidden = if hidden_rows
                  hidden_rows[1] || raise ArgumentError.new("tower_hidden_rows has no decode row")
                else
                  json_vector(inputs["decode_hidden"], "decode_hidden")
                end
prefill_previous = json_frame(inputs["prefill_previous_frame"], "prefill_previous_frame")
prefill_next = json_frame(inputs["prefill_next_frame"], "prefill_next_frame")
decode_previous = json_frame(inputs["decode_previous_frame"], "decode_previous_frame")
decode_main = inputs["decode_main"].as_i64.to_i32

traces = [] of Tuple(String, Array(Int32), Vector)
capture = ->(name : String, shape : Array(Int32), values : Vector) do
  traces << {name, shape, values}
end

caches = tower.make_cache
tower.teacher_force(
  prefill_hidden,
  prefill_previous,
  prefill_next,
  caches,
  capture,
  "tower.prefill.row0",
)
sample = tower.sample_frame(
  decode_main,
  decode_hidden,
  decode_previous,
  caches,
  capture,
  "tower.decode.row0",
) { |logits| argmax(logits) }

unless caches.all? { |cache| cache.offset == 2 }
  raise "tower cache did not advance to position 2 after prefill and decode"
end

output = JSON.build do |json|
  json.object do
    json.field "format", "drama3-reference-traces-v1"
    json.field "traces" do
      json.array do
        traces.sort_by(&.[0]).each do |name, shape, values|
          json.object do
            json.field "name", name
            json.field "shape", shape
            json.field "dtype", "float32"
            json.field "values" do
              # Emit the complete binary32 value through Float64. Consumers
              # still canonicalize decimals to the declared Float32 dtype.
              json.array { values.each { |value| json.number value.to_f64 } }
            end
          end
        end
      end
    end
    json.field "native_decoded_frame", sample.codes
  end
end
File.write(output_path, output)
Profile.write_from_env
