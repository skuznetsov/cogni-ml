require "json"
require "../src/ml/audio/drama3/tower_resident"

{% if flag?(:drama3_metal) && flag?(:darwin) && !flag?(:cpu_only) %}
  include ML::Audio::Drama3
  abort "usage: drama3_tower_resident_trace MODEL_DIR INPUTS_JSON OUTPUT_JSON" unless ARGV.size == 3

  model_dir = File.expand_path(ARGV[0])
  inputs = JSON.parse(File.read(ARGV[1])).as_h
  output_path = File.expand_path(ARGV[2])
  Profile.configure_from_env!
  MetalBackend.enabled?

  def drama3_tower_json_vector(value : JSON::Any, name : String) : Vector
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

  def drama3_tower_json_frame(value : JSON::Any, name : String) : Array(Int32)
    value.as_a.map do |item|
      item.as_i64.to_i32
    end
  end

  config = TowerConfig.from_json(File.read(File.join(model_dir, "config.json")))
  store = WeightStore.new(model_dir)
  context = Resident::Context.new
  tower = TowerResident.new(config, store, context, 2)
  hidden_rows = inputs["tower_hidden_rows"]?.try do |rows|
    rows.as_a.map_with_index { |row, index| drama3_tower_json_vector(row, "tower_hidden_rows[#{index}]") }
  end
  prefill_hidden = if hidden_rows
                     hidden_rows[0] || raise ArgumentError.new("tower_hidden_rows has no prefill row")
                   else
                     drama3_tower_json_vector(inputs["prefill_hidden"], "prefill_hidden")
                   end
  decode_hidden = if hidden_rows
                    hidden_rows[1] || raise ArgumentError.new("tower_hidden_rows has no decode row")
                  else
                    drama3_tower_json_vector(inputs["decode_hidden"], "decode_hidden")
                  end
  prefill_previous = drama3_tower_json_frame(inputs["prefill_previous_frame"], "prefill_previous_frame")
  prefill_next = drama3_tower_json_frame(inputs["prefill_next_frame"], "prefill_next_frame")
  decode_previous = drama3_tower_json_frame(inputs["decode_previous_frame"], "decode_previous_frame")
  decode_main = inputs["decode_main"].as_i64.to_i32

  traces = [] of Tuple(String, Array(Int32), Vector)
  capture = ->(name : String, shape : Array(Int32), values : Vector) do
    traces << {name, shape, values}
  end

  tower.teacher_force(
    tower.upload_hidden(prefill_hidden),
    tower.upload_frame(prefill_previous),
    tower.upload_frame(prefill_next),
    capture,
    "tower.prefill.row0",
  )

  draw = context.upload([0.0_f32])
  decoded = tower.sample_frame(
    context.upload([decode_main.to_f32]),
    tower.upload_hidden(decode_hidden),
    tower.upload_frame(decode_previous),
    capture,
    "tower.decode.row0",
  ) do |logits|
    context.sample(logits, draw, 1.0, 1, 1.0)
  end
  unless tower.offset == 2
    raise "tower cache did not advance to position 2 after prefill and decode"
  end
  decoded_frame = context.read(decoded, "terminal").map(&.to_i32)
  context.finish

  File.write(output_path, JSON.build do |json|
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
                json.array { values.each { |value| json.number value.to_f64 } }
              end
            end
          end
        end
      end
      json.field "native_decoded_frame", decoded_frame
      json.field "resident_readbacks", context.readbacks
      json.field "resident_submissions", context.submissions
    end
  end)
  Profile.write_from_env
{% else %}
  abort "build with -Ddrama3_metal on Darwin"
{% end %}
