# Compare the frozen tower FFN-pair selector on the same pinned MLX inputs.
# All activations and comparisons stay on Metal; only a compact receipt leaves.
require "digest/sha256"
require "json"
require "set"
require "../src/ml/audio/drama3/tower_resident"

{% if flag?(:drama3_metal) && flag?(:darwin) && !flag?(:cpu_only) %}
module Drama3TowerPairCompare
  include ML::Audio::Drama3
  alias Context = Resident::Context
  alias Buffer = Resident::Buffer

  SOURCE = {{ read_file("#{__DIR__}/support/drama3_prefill_compare.metal") }}
  MODEL_REPOSITORY = "cguldogan/drama-3-mlx-q4"
  MODEL_REVISION = "5331ab509359c0e4a7e353a19e255ddf3e8152a8"
  UPSTREAM_REVISION = "e7488935ec2ebfafd38ef2b58e290d79f3d05dc7"
  GENERATION_SHA256 = "ba22da53fa60afc34e80b23594ca311c6eb5fba74c5eed47605e87499bbf1fcf"
  CONFIG_SHA256 = "9d52aa7960ba723efcfe47d2ec03f52c5e22e84a6c2a7d773a845db1bf48563c"
  INDEX_SHA256 = "da812cc5b08188ccef048b9074e4cf70c8a1896859c5b492f374d7833e63b855"
  INTERNAL_ATOL = 6e-4_f32
  FINAL_ATOL = 3e-5_f32
  RTOL = 3e-4_f32
  RECEIPT_WIDTH = 5

  record Capture, name : String, shape : Array(Int32), buffer : Buffer
  record Comparison, name : String, kind : String, actual : Buffer, reference : Buffer,
                    atol : Float32, rtol : Float32, exact_native : Bool

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

  def self.number(value : JSON::Any, label : String) : Float32
    number = case raw = value.raw
             when Float64 then raw.to_f32
             when Int64   then raw.to_f32
             else              raise ArgumentError.new("#{label}: expected a JSON number")
             end
    raise ArgumentError.new("#{label}: non-finite value") unless number.finite?
    number
  end

  def self.integer(value : JSON::Any, label : String) : Int32
    raw = value.raw
    unless raw.is_a?(Int64) && raw.in?(Int32::MIN.to_i64..Int32::MAX.to_i64)
      raise ArgumentError.new("#{label}: expected an int32 JSON integer")
    end
    raw.to_i32
  end

  def self.vector(record : JSON::Any, label : String, expected_shape : Array(Int32)? = nil) : Array(Float32)
    raise ArgumentError.new("#{label}: dtype must be float32") unless record["dtype"].as_s == "float32"
    shape = record["shape"].as_a.map_with_index { |value, index| integer(value, "#{label}.shape[#{index}]") }
    raise ArgumentError.new("#{label}: fixture shape mismatch") if expected_shape && shape != expected_shape
    count = shape.reduce(1_i64) { |product, dimension| product * dimension }
    raise ArgumentError.new("#{label}: invalid shape") unless count > 0 && count <= Int32::MAX
    values = record["values"].as_a.map_with_index { |value, index| number(value, "#{label}.values[#{index}]") }
    raise ArgumentError.new("#{label}: value count mismatch") unless values.size.to_i64 == count
    values
  end

  def self.expected_layer_names(mode : String, group : String, index : Int32) : Array(String)
    prefix = "tower.#{mode}.row#{mode == "prefill" ? 85 : 0}.#{group}.#{index}"
    names = %w[
      embedding enorm hnorm eh_concat eh_proj attention_norm attention.q_projected
      attention.gate attention.k_projected attention.v_projected attention.q_norm
      attention.k_norm attention.q_rope attention.k_rope attention.gated_context
      attention.output attention_residual ffn_norm feed_forward.w1 feed_forward.w3
      feed_forward.silu feed_forward.product feed_forward.w2 ffn_residual output
    ].map { |suffix| "#{prefix}.#{suffix}" }
    names
  end

  def self.expected_trace_names(mode : String) : Set(String)
    row = mode == "prefill" ? 85 : 0
    prefix = "tower.#{mode}.row#{row}"
    names = Set(String).new
    names << "#{prefix}.hidden_input"
    names << "#{prefix}.unseen_embedding"
    names << "#{prefix}.hidden_output"
    {"shared_layers" => 4, "residual_layers" => 9}.each do |group, count|
      count.times do |index|
        expected_layer_names(mode, group, index).each { |name| names << name }
        names << "#{prefix}.head_logits.#{index}" if group == "residual_layers"
      end
    end
    names
  end

  def self.configure_environment(pair_fusion : String) : Nil
    ENV.keys.each { |key| ENV.delete(key) if key.starts_with?("DRAMA3_") }
    {
      "DRAMA3_BACKEND"                     => "metal",
      "DRAMA3_COGNIGRAPH_MAX_INFLIGHT"     => "2",
      "DRAMA3_DENSE_REDUCTION"             => "tree",
      "DRAMA3_TOWER_EMBEDDING_FUSION"      => "0",
      "DRAMA3_TOWER_FFN_PAIR_FUSION"       => pair_fusion,
      "DRAMA3_METAL_TIMELINE"              => "0",
      "DRAMA3_METAL_RESOURCES"             => "0",
      "DRAMA3_HOST_PROFILE"                => "0",
      "DRAMA3_MEMORY_PROFILE"              => "0",
      "DRAMA3_ALLOCATION_PROFILE"          => "0",
    }.each { |key, value| ENV[key] = value }
    ENV.delete("DRAMA3_PROFILE")
    Profile.configure_from_env!
    raise "profiling must be disabled" if Profile.enabled?
  end

  def self.metrics_source_names : Array(String)
    ["native_pair_exact", "baseline_vs_mlx", "candidate_vs_mlx"]
  end

  def self.capture(context : Context, name : String, shape : Array(Int32), buffer : Buffer) : Capture
    # The observer borrows live buffers. RoPE mutates q/k norm in place after
    # this callback, so preserve those boundaries with an ordered GPU copy.
    if name.ends_with?(".attention.q_norm") || name.ends_with?(".attention.k_norm")
      snapshot = context.alloc(buffer.count)
      context.copy(buffer, snapshot)
      Capture.new(name, shape.dup, snapshot)
    else
      Capture.new(name, shape.dup, buffer)
    end
  end

  def self.schedule(ctx : Context, comparisons : Array(Comparison), parts : Buffer, output : Buffer) : Nil
    comparisons.each_with_index do |comparison, slot|
      count = comparison.actual.count
      raise "comparison widths differ for #{comparison.name}" unless count == comparison.reference.count
      tiles = (count + 1023) // 1024
      result = output.slice(slot * RECEIPT_WIDTH, RECEIPT_WIDTH)
      ctx.dispatch("dr3_prefill_compare", SOURCE,
        [comparison.actual, comparison.reference, parts],
        [count.to_u64, comparison.atol.unsafe_as(UInt32).to_u64,
         comparison.rtol.unsafe_as(UInt32).to_u64], tiles)
      ctx.dispatch("dr3_prefill_compare_reduce", SOURCE, [parts, result], [tiles.to_u64], 1, 1)
    end
  end

  def self.read_metrics(ctx : Context, output : Buffer, count : Int32) : Array(Array(Float32))
    pointer = output.buffer.contents.as(Float32*) + output.offset // 4
    ctx.readbacks["terminal"] += count.to_i64 * RECEIPT_WIDTH * sizeof(Float32).to_i64
    Array(Array(Float32)).new(count) do |index|
      Array(Float32).new(RECEIPT_WIDTH) { |column| pointer[index * RECEIPT_WIDTH + column] }
    end
  end

  def self.metric_json(values : Array(Float32)) : JSON::Any
    JSON::Any.new({
      "bit_differences" => JSON::Any.new(values[0].to_i64),
      "tolerance_failures" => JSON::Any.new(values[1].to_i64),
      "max_abs" => JSON::Any.new(values[2].to_f64),
      "max_ratio" => JSON::Any.new(values[3].to_f64),
      "nonfinite" => JSON::Any.new(values[4].to_i64),
    } of String => JSON::Any)
  end

  def self.exact_native?(values : Array(Float32)) : Bool
    values.all?(&.finite?) && values[0] == 0_f32 && values[1] == 0_f32 && values[4] == 0_f32
  end

  def self.upstream_pass?(values : Array(Float32)) : Bool
    values.all?(&.finite?) && values[1] == 0_f32 && values[4] == 0_f32
  end

  def self.run : Nil
    raise "usage: drama3_tower_pair_compare MODEL_DIR PAIR_FIXTURE EXPECTED_FIXTURE_SHA256 OUTPUT_JSON" unless ARGV.size == 4
    model_path, fixture_path = File.expand_path(ARGV[0]), File.expand_path(ARGV[1])
    expected_fixture_sha, output_path = ARGV[2], File.expand_path(ARGV[3])
    raise "refusing existing output" if File.exists?(output_path)
    raise "fixture digest changed" unless expected_fixture_sha.size == 64 && sha(fixture_path) == expected_fixture_sha
    fixture = JSON.parse(File.read(fixture_path))
    raise "fixture format/model/upstream pin mismatch" unless fixture["format"].as_s == "drama3-tower-pair-fixture-v1" &&
      fixture["model_repository"].as_s == MODEL_REPOSITORY && fixture["model_revision"].as_s == MODEL_REVISION &&
      fixture["upstream_revision"].as_s == UPSTREAM_REVISION
    hashes = fixture["source_hashes"]
    config_path = File.join(model_path, "config.json")
    index_path = File.join(model_path, "model.safetensors.index.json")
    raise "model config SHA mismatch" unless sha(config_path) == CONFIG_SHA256 && hashes["config.json"].as_s == CONFIG_SHA256
    raise "model index SHA mismatch" unless sha(index_path) == INDEX_SHA256 && hashes["model.safetensors.index.json"].as_s == INDEX_SHA256
    provenance = fixture["provenance"]
    raise "fixture generation report pin mismatch" unless provenance["generation_report_sha256"].as_s == GENERATION_SHA256
    control = provenance["instrumentation_control"]
    raise "upstream trace instrumentation control failed" unless control["unhooked_prefill_hidden_output_bit_identical"].as_bool &&
      control["unhooked_prefill_hidden_output_finite"].as_bool &&
      control["unhooked_prefill_final_layer_bit_identical"].as_bool &&
      control["unhooked_prefill_final_layer_finite"].as_bool &&
      control["unhooked_decode_final_layer_bit_identical"].as_bool &&
      control["unhooked_decode_final_layer_finite"].as_bool &&
      control["unhooked_greedy_decode_frame_identical"].as_bool && control["control_backbone_reruns"].as_i == 0

    replay = fixture["replay"]
    hidden_rows = replay["prefill_hidden_rows"].as_a.map_with_index do |row, index|
      values = row.as_a.map_with_index { |value, column| number(value, "hidden row #{index} column #{column}") }
      raise "hidden row geometry changed" unless values.size == 2048
      values
    end
    raise "expected 87 backbone hidden rows" unless hidden_rows.size == 87
    frames = replay["prompt_frames"].as_a.map_with_index do |row, index|
      values = row.as_a.map_with_index { |value, column| integer(value, "prompt frame #{index} column #{column}") }
      raise "prompt frame geometry changed" unless values.size == 11
      values
    end
    raise "prompt frame/hidden row count mismatch" unless frames.size == hidden_rows.size
    decode_previous = replay["decode_previous_frame"].as_a.map_with_index do |value, index|
      integer(value, "decode previous frame[#{index}]")
    end
    raise "decode previous frame does not equal final prompt frame" unless decode_previous == frames.last
    decode_main = integer(replay["decode_main"], "decode main")
    upstream_frame = replay["decode_reference_frame"].as_a.map_with_index do |value, index|
      integer(value, "decode reference frame[#{index}]")
    end
    raise "upstream decode frame geometry/main mismatch" unless upstream_frame.size == 11 && upstream_frame[0] == decode_main

    references = {} of String => NamedTuple(shape: Array(Int32), values: Array(Float32))
    fixture["traces"].as_a.each do |record|
      name = record["name"].as_s
      raise "duplicate upstream trace #{name}" if references.has_key?(name)
      shape = record["shape"].as_a.map_with_index { |value, index| integer(value, "#{name}.shape[#{index}]") }
      values = vector(record, name, shape)
      references[name] = {shape: shape, values: values}
    end
    expected_names = expected_trace_names("prefill")
    expected_trace_names("decode").each { |name| expected_names << name }
    raise "upstream reference trace set mismatch" unless references.keys.to_set == expected_names

    ctx = nil.as(Context?)
    begin
      store = WeightStore.new(model_path)
      configure_environment("0")
      context = Context.new
      ctx = context
      raise "tower comparison requires tree dense reduction" unless context.dense_reduction == "tree"
      baseline = TowerResident.new(TowerConfig.from_json(File.read(config_path)), store, context, 87)
      raise "baseline FFN pair selector did not freeze off" if baseline.ffn_pair_fusion_enabled
      configure_environment("1")
      candidate = TowerResident.new(TowerConfig.from_json(File.read(config_path)), store, context, 87)
      raise "candidate FFN pair selector did not freeze on" unless candidate.ffn_pair_fusion_enabled
      ENV["DRAMA3_TOWER_FFN_PAIR_FUSION"] = "0"
      raise "baseline selector changed after candidate construction" if baseline.ffn_pair_fusion_enabled
      raise "candidate selector changed after environment reset" unless candidate.ffn_pair_fusion_enabled

      hidden_buffers = hidden_rows.map { |row| context.upload(row) }
      frame_buffers = frames.map { |frame| context.upload(frame.map(&.to_f32)) }
      baseline_captures = [] of Capture
      candidate_captures = [] of Capture
      capture_active = false
      baseline.buffer_trace = ->(name : String, shape : Array(Int32), buffer : Buffer) {
        baseline_captures << capture(context, name, shape, buffer) if capture_active
        nil
      }
      candidate.buffer_trace = ->(name : String, shape : Array(Int32), buffer : Buffer) {
        candidate_captures << capture(context, name, shape, buffer) if capture_active
        nil
      }

      (0...86).each do |row|
        capture_active = row == 85
        prefix = "tower.prefill.row#{row}"
        trace_prefix = row == 85 ? "tower.prefill.row85" : prefix
        baseline.teacher_force(hidden_buffers[row], frame_buffers[row], frame_buffers[row + 1], nil, trace_prefix)
        candidate.teacher_force(hidden_buffers[row], frame_buffers[row], frame_buffers[row + 1], nil, trace_prefix)
      end
      raise "prefill cache replay offset mismatch" unless baseline.offset == 86 && candidate.offset == 86

      main_buffer = context.upload([decode_main.to_f32])
      previous_buffer = frame_buffers.last
      capture_active = true
      baseline_frame = baseline.sample_frame(main_buffer, hidden_buffers[86], previous_buffer, nil, "tower.decode.row0") do |logits|
        context.argmax(logits)
      end
      candidate_frame = candidate.sample_frame(main_buffer, hidden_buffers[86], previous_buffer, nil, "tower.decode.row0") do |logits|
        context.argmax(logits)
      end
      capture_active = false
      raise "decode cache replay offset mismatch" unless baseline.offset == 87 && candidate.offset == 87
      raise "native capture set differs" unless baseline_captures.map(&.name).to_set == expected_names &&
        candidate_captures.map(&.name).to_set == expected_names
      raise "native/MLX trace order or count differs" unless baseline_captures.size == expected_names.size &&
        candidate_captures.size == expected_names.size

      baseline_by_name = baseline_captures.to_h { |capture| {capture.name, capture} }
      candidate_by_name = candidate_captures.to_h { |capture| {capture.name, capture} }
      comparisons = [] of Comparison
      native_exact_slots = [] of Int32
      baseline_upstream_slots = [] of Int32
      candidate_upstream_slots = [] of Int32
      comparison_index = 0
      decode_frame_native_slot = -1
      decode_frame_baseline_slot = -1
      decode_frame_candidate_slot = -1
      add_triplet = ->(name : String, baseline_buffer : Buffer, candidate_buffer : Buffer,
                       reference_buffer : Buffer, atol : Float32, rtol : Float32) {
        comparisons << Comparison.new(name, "native_pair_exact", candidate_buffer, baseline_buffer,
          0_f32, 0_f32, true)
        native_exact_slots << comparison_index
        comparison_index += 1
        comparisons << Comparison.new(name, "baseline_vs_mlx", baseline_buffer, reference_buffer, atol, rtol, false)
        baseline_upstream_slots << comparison_index
        comparison_index += 1
        comparisons << Comparison.new(name, "candidate_vs_mlx", candidate_buffer, reference_buffer, atol, rtol, false)
        candidate_upstream_slots << comparison_index
        comparison_index += 1
      }
      (baseline_by_name.keys + ["tower.decode.row0.frame"]).sort.uniq.each do |name|
        if name == "tower.decode.row0.frame"
          reference_values = upstream_frame.map(&.to_f32)
          reference_buffer = context.upload(reference_values)
          add_triplet.call(name, baseline_frame, candidate_frame, reference_buffer, 0_f32, 0_f32)
          decode_frame_native_slot = native_exact_slots.last
          decode_frame_baseline_slot = baseline_upstream_slots.last
          decode_frame_candidate_slot = candidate_upstream_slots.last
          next
        end
        left = baseline_by_name[name]
        right = candidate_by_name[name]
        reference = references[name]?
        raise "missing MLX reference #{name}" unless reference
        raise "native trace shape differs from MLX for #{name}" unless left.shape == right.shape && left.shape == reference[:shape]
        ref_buffer = context.upload(reference[:values])
        terminal = name.ends_with?(".hidden_output") || name.includes?(".head_logits.")
        atol = terminal ? FINAL_ATOL : INTERNAL_ATOL
        add_triplet.call(name, left.buffer, right.buffer, ref_buffer, atol, RTOL)
      end

      max_tiles = comparisons.max_of { |comparison| (comparison.actual.count + 1023) // 1024 }
      parts = context.alloc(max_tiles * RECEIPT_WIDTH)
      metric_output = context.alloc(comparisons.size * RECEIPT_WIDTH)
      schedule(context, comparisons, parts, metric_output)
      context.finish
      metric_values = read_metrics(context, metric_output, comparisons.size)

      entries = comparisons.map_with_index do |comparison, index|
        values = metric_values[index]
        {"name" => comparison.name, "kind" => comparison.kind,
         "atol" => comparison.atol.to_f64, "rtol" => comparison.rtol.to_f64,
         "metrics" => metric_json(values), "pass" => comparison.exact_native ? exact_native?(values) : upstream_pass?(values)}
      end
      native_pair_pass = native_exact_slots.all? { |slot| exact_native?(metric_values[slot]) }
      baseline_upstream_pass = baseline_upstream_slots.all? { |slot| upstream_pass?(metric_values[slot]) }
      candidate_upstream_pass = candidate_upstream_slots.all? { |slot| upstream_pass?(metric_values[slot]) }
      decode_frame_equal = exact_native?(metric_values[decode_frame_native_slot]) &&
        upstream_pass?(metric_values[decode_frame_baseline_slot]) &&
        upstream_pass?(metric_values[decode_frame_candidate_slot])
      output = {
        "format" => "drama3-tower-pair-comparison-v1",
        "fixture_sha256" => expected_fixture_sha,
        "model_repository" => MODEL_REPOSITORY,
        "model_revision" => MODEL_REVISION,
        "upstream_revision" => UPSTREAM_REVISION,
        "source_hashes" => {
          "config.json" => hashes["config.json"].as_s,
          "model.safetensors.index.json" => hashes["model.safetensors.index.json"].as_s,
          "actual_projection_tensors" => hashes["actual_projection_tensors"].as_s,
          "comparator" => sha(__FILE__),
          "tower_resident.cr" => sha(File.join(File.expand_path("..", __DIR__), "src/ml/audio/drama3/tower_resident.cr")),
          "resident_metal" => Digest::SHA256.hexdigest(Resident::SOURCE),
          "tolerance_kernel" => Digest::SHA256.hexdigest(SOURCE),
        },
        "provenance" => provenance,
        "common_input_policy" => "same pinned 87 MLX backbone hidden rows and prompt frames for teacher_force rows 0..85; decode uses actual first report main token, final prompt frame, and context.argmax for each residual codebook",
        "trace_count" => expected_names.size,
        "tolerances" => {"internal_atol" => INTERNAL_ATOL.to_f64, "hidden_and_logits_atol" => FINAL_ATOL.to_f64, "rtol" => RTOL.to_f64},
        "native_pair_bit_exact" => native_pair_pass,
        "baseline_strict_upstream_pass" => baseline_upstream_pass,
        "candidate_strict_upstream_pass" => candidate_upstream_pass,
        "strict_upstream_pass" => baseline_upstream_pass && candidate_upstream_pass && decode_frame_equal,
        "decode_frame_equal_to_mlx" => decode_frame_equal,
        "comparisons" => entries,
        "graph" => context.graph_stats,
        "readbacks" => context.readbacks,
        "terminal_metric_read_bytes" => comparisons.size.to_i64 * RECEIPT_WIDTH * sizeof(Float32).to_i64,
      }
      output_path_parent = File.dirname(output_path)
      Dir.mkdir_p(output_path_parent)
      File.write(output_path, output.to_json + "\n")
      puts({event: "tower_pair_compare", fixture_sha256: expected_fixture_sha,
            native_pair_bit_exact: native_pair_pass, baseline_strict_upstream_pass: baseline_upstream_pass,
            candidate_strict_upstream_pass: candidate_upstream_pass, strict_upstream_pass: output["strict_upstream_pass"],
            decode_frame_equal_to_mlx: decode_frame_equal, trace_count: expected_names.size,
            terminal_metric_read_bytes: comparisons.size.to_i64 * RECEIPT_WIDTH * sizeof(Float32).to_i64,
            output: output_path}.to_json)
      raise "native off/on tower comparison failed" unless native_pair_pass
    rescue ex
      ctx.try(&.abort)
      raise ex
    end
  end
end
Drama3TowerPairCompare.run
{% else %}
  abort "build with -Ddrama3_metal on Darwin"
{% end %}
