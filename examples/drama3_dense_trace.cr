require "json"
require "set"
require "digest/sha256"
require "../src/ml/audio/drama3/backbone_resident"

{% if flag?(:drama3_metal) && flag?(:darwin) && !flag?(:cpu_only) %}
  include ML::Audio::Drama3

  def dense_float(value : JSON::Any) : Float32
    case raw = value.raw
    when Int64
      raw.to_f32
    when Float64
      raw.to_f32
    else
      raise ArgumentError.new("dense fixture contains a non-numeric value")
    end
  end

  abort "usage: drama3_dense_trace MODEL_DIR INPUT_FIXTURE OUTPUT_JSON tree|simd" unless ARGV.size == 4
  model_path, fixture_path, output_path, mode = ARGV
  abort "dense reduction must be tree or simd" unless mode.in?("tree", "simd")

  fixture_bytes = File.read(fixture_path)
  fixture_sha256 = Digest::SHA256.hexdigest(fixture_bytes)
  fixture = JSON.parse(fixture_bytes)
  abort "unexpected dense input fixture format" unless fixture["format"].as_s == "drama3-dense-projection-inputs-v1"
  abort "unexpected dense input projection" unless fixture["projection"].as_s == "backbone.layers.0.linear_attn"
  abort "dense input dtype must be float32" unless fixture["input_dtype"].as_s == "float32"
  input_shape = fixture["input_shape"].as_a
  abort "dense input shape must be [2048]" unless input_shape.size == 1 && input_shape[0].as_i == 2048

  input_rows = [] of NamedTuple(row: Int32, values: Vector)
  seen_rows = Set(Int32).new
  fixture["rows"].as_a.each do |item|
    row = item["row"].as_i.to_i32
    raise ArgumentError.new("duplicate or negative dense fixture row") if row < 0 || seen_rows.includes?(row)
    seen_rows << row
    values = item["values"].as_a.map { |value| dense_float(value) }
    raise ArgumentError.new("dense fixture row width must be 2048") unless values.size == 2048
    raise ArgumentError.new("dense fixture row contains non-finite input") unless values.all?(&.finite?)
    input_rows << {row: row, values: values}
  end
  raise ArgumentError.new("dense fixture has no rows") if input_rows.empty?

  expected_widths = {"qkv" => 8192, "z" => 4096, "b" => 32, "a" => 32}
  prefixes = {
    "qkv" => "backbone.layers.0.linear_attn.in_proj_qkv.weight",
    "z"   => "backbone.layers.0.linear_attn.in_proj_z.weight",
    "b"   => "backbone.layers.0.linear_attn.in_proj_b.weight",
    "a"   => "backbone.layers.0.linear_attn.in_proj_a.weight",
  }

  store = WeightStore.new(model_path)
  matrices = {} of String => Matrix
  weight_metadata = {} of String => NamedTuple(shape: Array(Int32), dtype: String)
  prefixes.each do |name, tensor_name|
    matrix = store.matrix(tensor_name)
    expected = expected_widths[name]
    raise ArgumentError.new("#{name} projection shape must be [#{expected}, 2048]") unless matrix.rows == expected && matrix.cols == 2048
    raise ArgumentError.new("#{name} projection weight dtype must remain BF16") unless matrix.tensor.dtype == "BF16"
    matrices[name] = matrix
    weight_metadata[name] = {shape: [matrix.rows, matrix.cols], dtype: matrix.tensor.dtype}
  end

  Profile.configure_from_env!
  context = Resident::Context.new
  rows = [] of NamedTuple(
    row: Int32,
    projections: Hash(String, NamedTuple(shape: Array(Int32), dtype: String, values: Vector)),
  )
  input_rows.each do |input_row|
    input = context.upload(input_row[:values])
    outputs = {} of String => Resident::Buffer
    prefixes.each_key do |name|
      outputs[name] = context.matvec(matrices[name], input, dense_reduction: mode)
    end
    # One completion boundary for the four queued projections; the diagnostic
    # readbacks below are outside any benchmark interval.
    context.finish
    projections = {} of String => NamedTuple(shape: Array(Int32), dtype: String, values: Vector)
    prefixes.each_key do |name|
      projections[name] = {
        shape:  [expected_widths[name]],
        dtype:  "float32",
        values: context.read(outputs[name], "diagnostic"),
      }
    end
    rows << {row: input_row[:row], projections: projections}
  end

  trace = {
    format:               "drama3-dense-projection-native-v1",
    projection:           "backbone.layers.0.linear_attn",
    mode:                 mode,
    model:                model_path,
    input_fixture:        fixture_path,
    input_fixture_sha256: fixture_sha256,
    input_fixture_format: fixture["format"].as_s,
    weight_metadata:      weight_metadata,
    rows:                 rows,
  }
  File.write(output_path, trace.to_json + "\n")
  puts({"trace": output_path, "rows": rows.size, "mode": mode}.to_json)
  Profile.write_from_env
{% else %}
  abort "build with -Ddrama3_metal on Darwin"
{% end %}
