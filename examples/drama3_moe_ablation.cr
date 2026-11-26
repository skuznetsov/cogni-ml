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
  HIDDEN_WIDTH        = 2048
  INTERMEDIATE_WIDTH  =  512
  TOP_K               =    8
  TARGET_LAYER        =   39

  alias MoeStage = NamedTuple(shape: Array(Int32), dtype: String, values: Array(Float32), capture_kind: String)

  def moe_float(value : JSON::Any) : Float32
    raw = value.raw
    number = case raw
             when Int64   then raw.to_f32
             when Float64 then raw.to_f32
             else              raise ArgumentError.new("MoE trace contains a non-numeric value")
             end
    raise ArgumentError.new("MoE trace contains a non-finite value") unless number.finite?
    number
  end

  def moe_stage_values(stages : JSON::Any, name : String, shape : Array(Int32), where : String) : Array(Float32)
    stage = stages[name]?
    raise ArgumentError.new("#{where}: missing stage #{name}") unless stage
    actual_shape = stage["shape"].as_a.map { |value| value.as_i.to_i32 }
    raise ArgumentError.new("#{where}.#{name}: expected shape #{shape}, found #{actual_shape}") unless actual_shape == shape
    values = stage["values"].as_a.map { |value| moe_float(value) }
    expected = shape.reduce(1) { |product, width| product * width }
    raise ArgumentError.new("#{where}.#{name}: expected #{expected} values, found #{values.size}") unless values.size == expected
    values
  end

  def moe_stage_kind(stages : JSON::Any, name : String, where : String) : String
    stage = stages[name]?
    raise ArgumentError.new("#{where}: missing stage #{name}") unless stage
    stage["capture_kind"].as_s
  end

  def moe_hash_map(value : JSON::Any, where : String) : Hash(String, String)
    value.as_h.to_h do |name, digest|
      key = name
      hash = digest.as_s
      raise ArgumentError.new("#{where}.#{key}: malformed SHA-256") unless hash.matches?(/\A[0-9a-f]{64}\z/)
      {key, hash}
    end
  end

  def moe_ids(values : Array(Float32), where : String) : Array(Int32)
    values.map_with_index do |value, index|
      id = value.to_i32
      raise ArgumentError.new("#{where}[#{index}]: expected integral expert ID") unless id.to_f32 == value && id >= 0 && id < 256
      id
    end.tap do |ids|
      raise ArgumentError.new("#{where}: expected #{TOP_K} expert IDs") unless ids.size == TOP_K
      raise ArgumentError.new("#{where}: duplicate expert IDs") unless ids.uniq.size == ids.size
    end
  end

  def moe_upstream_ids(stages : JSON::Any) : Array(Int32)
    stage = stages["expert_ids_argpartition_order"]
    shape = stage["shape"].as_a.map { |value| value.as_i.to_i32 }
    raise ArgumentError.new("upstream expert ID shape") unless shape == [TOP_K]
    ids = stage["values"].as_a.map do |value|
      raw = value.raw
      id = case raw
           when Int64 then raw.to_i32
           when Float64
             converted = raw.to_i32
             raise ArgumentError.new("upstream expert ID is not integral") unless converted.to_f64 == raw
             converted
           else raise ArgumentError.new("upstream expert ID is not numeric")
           end
      id
    end
    raise ArgumentError.new("upstream expert ID count") unless ids.size == TOP_K
    raise ArgumentError.new("upstream expert ID range") unless ids.all? { |id| id >= 0 && id < 256 }
    raise ArgumentError.new("upstream duplicate expert IDs") unless ids.uniq.size == TOP_K
    ids
  end

  def moe_rank_ids(run : JSON::Any, ids : Array(Int32)) : Nil
    rows = run["expert_rank_to_id"].as_a
    pairs = rows.map do |row|
      {rank: row["rank"].as_i.to_i32, expert_id: row["expert_id"].as_i.to_i32}
    end.sort_by(&.[:rank])
    raise ArgumentError.new("native expert rank map count") unless pairs.size == TOP_K
    raise ArgumentError.new("native expert rank map differs from routed IDs") unless pairs.map(&.[:expert_id]) == ids
  end

  def moe_explicit_ids(stage : JSON::Any, ids : Array(Int32), where : String) : Nil
    values = stage["expert_ids"]?.try(&.as_a)
    raise ArgumentError.new("#{where}: missing explicit expert ID map") unless values
    mapped = values.map { |value| value.as_i.to_i32 }
    raise ArgumentError.new("#{where}: explicit expert ID map differs from route order") unless mapped == ids
  end

  def moe_reorder_rows(values : Array(Float32), width : Int32, source_ids : Array(Int32), target_ids : Array(Int32)) : Array(Float32)
    raise ArgumentError.new("expert matrix row count mismatch") unless values.size == width * source_ids.size
    target_ids.flat_map do |id|
      rank = source_ids.index(id) || raise ArgumentError.new("expert ID #{id} is missing from source rows")
      Array(Float32).new(width) { |index| values[rank * width + index] }
    end
  end

  def moe_reorder_weights(values : Array(Float32), source_ids : Array(Int32), target_ids : Array(Int32)) : Array(Float32)
    raise ArgumentError.new("expert weight count mismatch") unless values.size == source_ids.size
    target_ids.map do |id|
      rank = source_ids.index(id) || raise ArgumentError.new("expert ID #{id} is missing from source weights")
      values[rank]
    end
  end

  def moe_bitwise_check(actual : Array(Float32), expected : Array(Float32), name : String) : NamedTuple(name: String, passed: Bool, count: Int32, mismatch_count: Int32, mismatches: Array(NamedTuple(index: Int32, expected_bits: UInt32, actual_bits: UInt32)))
    raise ArgumentError.new("#{name}: bitwise comparison shape mismatch") unless actual.size == expected.size
    mismatches = [] of NamedTuple(index: Int32, expected_bits: UInt32, actual_bits: UInt32)
    actual.each_with_index do |value, index|
      expected_bits = expected[index].unsafe_as(UInt32)
      actual_bits = value.unsafe_as(UInt32)
      mismatches << {index: index.to_i32, expected_bits: expected_bits, actual_bits: actual_bits} if expected_bits != actual_bits && mismatches.size < 32
    end
    # Keep the complete mismatch count separate from the bounded witness list.
    mismatch_count = actual.each_with_index.count { |value, index| value.unsafe_as(UInt32) != expected[index].unsafe_as(UInt32) }.to_i32
    {name: name, passed: mismatch_count == 0, count: actual.size.to_i32, mismatch_count: mismatch_count, mismatches: mismatches}
  end

  def moe_enqueue_sum(ctx : Resident::Context, terms : Array(Float32), weights : Array(Float32), ids : Array(Int32)) : Resident::Buffer
    raise ArgumentError.new("routed sum input shape") unless terms.size == HIDDEN_WIDTH * TOP_K && weights.size == TOP_K && ids.size == TOP_K
    routing = ids.map(&.to_f32) + weights
    term_buffer = ctx.upload(terms)
    routing_buffer = ctx.upload(routing)
    output = ctx.alloc(HIDDEN_WIDTH)
    ctx.dispatch("dr3_expert_sum", BackboneResident::SOURCE, [term_buffer, routing_buffer, output], [HIDDEN_WIDTH.to_u64, TOP_K.to_u64], HIDDEN_WIDTH)
    output
  end

  abort "usage: drama3_moe_ablation MODEL_DIR UPSTREAM_FULL_TRACE NATIVE_TRACE OUTPUT_JSON" unless ARGV.size == 4
  model_path, upstream_path, native_path, output_path = ARGV
  upstream_bytes = File.read(upstream_path)
  native_bytes = File.read(native_path)
  upstream = JSON.parse(upstream_bytes)
  native = JSON.parse(native_bytes)
  raise ArgumentError.new("unexpected upstream diagnostic format") unless upstream["format"].as_s == "drama3-moe-layer-diagnostic-v1"
  raise ArgumentError.new("unexpected native trace format") unless native["format"].as_s == "drama3-moe-trace-native-v1"

  coordinate = upstream["coordinate"]
  raise ArgumentError.new("upstream trace must identify row 1, layer #{TARGET_LAYER}") unless coordinate["row"].as_i == 1 && coordinate["layer"].as_i == TARGET_LAYER
  raise ArgumentError.new("native trace must identify row 1, layer #{TARGET_LAYER}") unless native["coordinate"]["row"].as_i == 1 && native["coordinate"]["layer"].as_i == TARGET_LAYER
  [upstream, native].each do |trace|
    model = trace["model"]
    upstream_pin = trace["upstream"]
    raise ArgumentError.new("model pin mismatch") unless model["repository"].as_s == MODEL_REPOSITORY && model["revision"].as_s == MODEL_REVISION
    raise ArgumentError.new("upstream pin mismatch") unless upstream_pin["repository"].as_s == UPSTREAM_REPOSITORY && upstream_pin["revision"].as_s == UPSTREAM_REVISION
  end
  raise ArgumentError.new("upstream trace must be a full-run capture") unless upstream["mode"].as_s == "full_run"
  model_config = upstream["model_config"]
  expected_model_config = {
    "hidden_size"           => HIDDEN_WIDTH,
    "moe_intermediate_size" => INTERMEDIATE_WIDTH,
    "num_experts"           => 256,
    "num_experts_per_tok"   => TOP_K,
  }
  expected_model_config.each do |name, expected|
    actual = model_config[name]?.try(&.as_i)
    raise ArgumentError.new("upstream.model_config.#{name}: expected #{expected}, found #{actual}") unless actual == expected
  end
  config_json = JSON.parse(File.read(File.join(model_path, "config.json")))
  config = BackboneConfig.new(config_json)
  raise ArgumentError.new("model MoE dimensions differ from the pinned trace") unless config.hidden == HIDDEN_WIDTH && config.experts == 256 && config.top_k == TOP_K && config.layers > TARGET_LAYER
  text_config = config_json["text_config"]? || config_json
  raise ArgumentError.new("model MoE intermediate width differs from the pinned trace") unless text_config["moe_intermediate_size"]?.try(&.as_i) == INTERMEDIATE_WIDTH
  raise ArgumentError.new("upstream capture control failed") unless upstream["capture_control"]["passed"].as_bool
  raise ArgumentError.new("native capture control failed") unless native["capture_control"]["passed"].as_bool
  raise ArgumentError.new("native isolated experts must use batched path") unless native["execution"]["isolated_batch_path"].as_bool

  upstream_stages = upstream["stages"]
  native_run = native["runs"].as_a.find { |run| run["name"].as_s == "isolated_upstream_input" } || raise ArgumentError.new("native trace has no isolated upstream-input run")
  native_stages = native_run["stages"]
  upstream_input_record = upstream["input"]
  native_input_fixture = native["input_fixture"]
  upstream_input_sha = upstream_input_record["sha256"].as_s
  native_input_sha = native_input_fixture["sha256"].as_s
  raise ArgumentError.new("upstream and native input fixture bytes differ") unless upstream_input_sha == native_input_sha
  upstream_source_hashes = moe_hash_map(upstream["source_hashes"], "upstream.source_hashes")
  native_fixture_hashes = moe_hash_map(native["source_hashes"]["fixture"], "native.source_hashes.fixture")
  raise ArgumentError.new("native and upstream fixture source hashes differ") unless native_fixture_hashes == upstream_source_hashes
  raise ArgumentError.new("upstream input metadata mismatch") unless upstream_input_record["format"].as_s == "drama3-moe-input-v1" && upstream_input_record["activation_name"].as_s == "post_attention_layernorm" && upstream_input_record["dtype"].as_s == "float32" && upstream_input_record["shape"].as_a.map(&.as_i) == [HIDDEN_WIDTH]

  upstream_input = moe_stage_values(upstream_stages, "normalized_input", [HIDDEN_WIDTH], "upstream")
  native_input = moe_stage_values(native_stages, "normalized_input", [HIDDEN_WIDTH], "native isolated")
  input_check = moe_bitwise_check(native_input, upstream_input, "normalized_input")
  raise ArgumentError.new("native isolated activation differs from exact upstream input") unless input_check[:passed]

  upstream_ids = moe_upstream_ids(upstream_stages)
  native_ids = moe_ids(moe_stage_values(native_stages, "expert_ids_argpartition_order", [TOP_K], "native isolated"), "native expert IDs")
  raise ArgumentError.new("native and upstream expert sets differ") unless upstream_ids.to_set == native_ids.to_set
  moe_rank_ids(native_run, native_ids)

  ref_activation = moe_stage_values(upstream_stages, "expert_activation_outputs", [TOP_K, INTERMEDIATE_WIDTH], "upstream")
  ref_down = moe_stage_values(upstream_stages, "expert_down_outputs", [TOP_K, HIDDEN_WIDTH], "upstream")
  ref_weights = moe_stage_values(upstream_stages, "selected_weights", [TOP_K], "upstream")
  nat_activation = moe_stage_values(native_stages, "expert_activation_outputs", [TOP_K, INTERMEDIATE_WIDTH], "native isolated")
  nat_down = moe_stage_values(native_stages, "expert_down_outputs", [TOP_K, HIDDEN_WIDTH], "native isolated")
  nat_weights = moe_stage_values(native_stages, "selected_weights", [TOP_K], "native isolated")
  ref_official_sum = moe_stage_values(upstream_stages, "routed_sum_official", [HIDDEN_WIDTH], "upstream")
  nat_official_sum = moe_stage_values(native_stages, "routed_sum_official", [HIDDEN_WIDTH], "native isolated")
  raise ArgumentError.new("native normalized-input capture kind is not official") unless moe_stage_kind(native_stages, "normalized_input", "native isolated") == "official_in_path"
  raise ArgumentError.new("native activation capture kind is not official") unless moe_stage_kind(native_stages, "expert_activation_outputs", "native isolated") == "official_in_path"
  raise ArgumentError.new("native down capture kind is not official") unless moe_stage_kind(native_stages, "expert_down_outputs", "native isolated") == "official_in_path"
  raise ArgumentError.new("upstream activation is not an actual SwiGLU return") unless moe_stage_kind(upstream_stages, "expert_activation_outputs", "upstream") == "upstream_swiglu_return"
  raise ArgumentError.new("upstream down output is not an actual SwitchGLU return") unless moe_stage_kind(upstream_stages, "expert_down_outputs", "upstream") == "upstream_switch_mlp_return"
  {"expert_gate_outputs", "expert_up_outputs", "expert_activation_outputs", "expert_down_outputs"}.each do |name|
    stage = upstream_stages[name]
    moe_explicit_ids(stage, upstream_ids, "upstream.#{name}")
  end

  activation_sha = Digest::SHA256.hexdigest(String.build do |io|
    upstream_input.each do |value|
      bits = value.unsafe_as(UInt32)
      4.times { |byte| io.write_byte(((bits >> (8 * byte)) & 0xff).to_u8) }
    end
  end)
  raise ArgumentError.new("upstream activation hash mismatch") unless activation_sha == upstream_input_record["activation_sha256"].as_s

  Profile.configure_from_env!
  raise ArgumentError.new("DRAMA3_PROFILE must be disabled for the single-drain ablation") if Profile.enabled?
  store = WeightStore.new(model_path)
  model_weight_provenance = Drama3MoeProvenance.verify!(model_path, store, upstream["source_hashes"].as_h)
  ctx = Resident::Context.new
  down_matrix = store.matrix("backbone.layers.#{TARGET_LAYER}.mlp.switch_mlp.down_proj.weight", 0)
  raise ArgumentError.new("target down matrix must be 2048x512 Q4 affine") unless down_matrix.rows == HIDDEN_WIDTH && down_matrix.cols == INTERMEDIATE_WIDTH && down_matrix.bits == 4 && down_matrix.group_size == 64 && down_matrix.scales && down_matrix.biases

  # These two GPU executions isolate the down projection from its upstream
  # inputs. The second must reproduce the native captured down rows exactly.
  reference_activation_buffer = ctx.upload(ref_activation)
  reference_ids_buffer = ctx.upload(upstream_ids.map(&.to_f32))
  native_ref_activation_down = ctx.experts(down_matrix, reference_activation_buffer, reference_ids_buffer, per_rank_input: true)
  native_activation_buffer = ctx.upload(nat_activation)
  native_ids_buffer = ctx.upload(native_ids.map(&.to_f32))
  native_replayed_down = ctx.experts(down_matrix, native_activation_buffer, native_ids_buffer, per_rank_input: true)

  refdown_native_order = moe_reorder_rows(ref_down, HIDDEN_WIDTH, upstream_ids, native_ids)
  refweights_native_order = moe_reorder_weights(ref_weights, upstream_ids, native_ids)
  natdown_reference_order = moe_reorder_rows(nat_down, HIDDEN_WIDTH, native_ids, upstream_ids)
  natweights_reference_order = moe_reorder_weights(nat_weights, native_ids, upstream_ids)

  sum_buffers = {} of String => Resident::Buffer
  case_specs = [
    {name: "reference_terms_reference_weights_reference_order", terms: ref_down, weights: ref_weights, ids: upstream_ids,
     terms_source: "upstream expert_down_outputs", weights_source: "upstream selected_weights"},
    {name: "reference_terms_reference_weights_native_order", terms: refdown_native_order, weights: refweights_native_order, ids: native_ids,
     terms_source: "upstream expert_down_outputs reordered by expert ID", weights_source: "upstream selected_weights reordered by expert ID"},
    {name: "native_terms_reference_weights_reference_order", terms: natdown_reference_order, weights: ref_weights, ids: upstream_ids,
     terms_source: "native captured expert_down_outputs reordered by expert ID", weights_source: "upstream selected_weights"},
    {name: "native_terms_native_weights_reference_order", terms: natdown_reference_order, weights: natweights_reference_order, ids: upstream_ids,
     terms_source: "native captured expert_down_outputs reordered by expert ID", weights_source: "native selected_weights reordered by expert ID"},
    {name: "native_terms_native_weights_native_order", terms: nat_down, weights: nat_weights, ids: native_ids,
     terms_source: "native captured expert_down_outputs", weights_source: "native selected_weights"},
  ]
  case_specs.each do |spec|
    sum_buffers[spec[:name]] = moe_enqueue_sum(ctx, spec[:terms], spec[:weights], spec[:ids])
  end

  raise ArgumentError.new("unexpected GPU readback before final drain: #{ctx.readbacks}") unless ctx.readbacks.values.all?(&.zero?)
  graph_before_drain = ctx.graph_stats
  ctx.finish
  graph_after_drain = ctx.graph_stats

  ref_activation_native_down_values = ctx.read(native_ref_activation_down, "terminal")
  native_replay_down_values = ctx.read(native_replayed_down, "terminal")
  native_down_replay_control = moe_bitwise_check(native_replay_down_values, nat_down, "native_down_replay")
  native_sum_values = ctx.read(sum_buffers[case_specs.last[:name]], "terminal")
  native_sum_control = moe_bitwise_check(native_sum_values, nat_official_sum, "native_sum_native_order")
  controls = [native_down_replay_control, native_sum_control]
  controls_passed = controls.all?(&.[:passed])
  control_mismatches = controls.reject(&.[:passed]).map do |control|
    "#{control[:name]}: #{control[:mismatch_count]} bitwise mismatches"
  end
  control_count = controls.sum(&.[:count])
  control_mismatch_count = controls.sum(&.[:mismatch_count])

  direct_down_check = moe_bitwise_check(ref_activation_native_down_values, ref_down, "native_down_on_reference_activation")
  cases = case_specs.map do |spec|
    values = ctx.read(sum_buffers[spec[:name]], "terminal")
    {
      name:           spec[:name],
      capture_kind:   "diagnostic_auxiliary",
      expert_ids:     spec[:ids],
      terms_source:   spec[:terms_source],
      weights_source: spec[:weights_source],
      shape:          [HIDDEN_WIDTH],
      dtype:          "float32",
      values:         values,
    }
  end
  reference_activation_native_down = {
    name:                        "native_down_on_upstream_activation",
    capture_kind:                "diagnostic_auxiliary",
    expert_ids:                  upstream_ids,
    shape:                       [TOP_K, HIDDEN_WIDTH],
    dtype:                       "float32",
    values:                      ref_activation_native_down_values,
    comparison_to_upstream_down: direct_down_check,
  }
  output = {
    format:                  "drama3-moe-ablation-v1",
    coordinate:              {row: 1, layer: TARGET_LAYER},
    model:                   upstream["model"],
    upstream:                upstream["upstream"],
    model_weight_provenance: model_weight_provenance,
    input_hashes:            {
      upstream_trace_sha256:          Digest::SHA256.hexdigest(upstream_bytes),
      native_trace_sha256:            Digest::SHA256.hexdigest(native_bytes),
      upstream_input_bundle_sha256:   upstream_input_sha,
      upstream_activation_sha256:     activation_sha,
      native_fixture_sha256:          native_input_sha,
      input_fixture_bytes_equal:      true,
      normalized_input_bitwise_equal: input_check[:passed],
    },
    expert_ids:                       {upstream_order: upstream_ids, native_order: native_ids},
    reference_activation_native_down: reference_activation_native_down,
    native_down_replay_control:       native_down_replay_control,
    cases:                            cases,
    self_controls:                    controls,
    capture_control:                  {passed: controls_passed, count: control_count, mismatch_count: control_mismatch_count, mismatches: control_mismatches},
    graph_before_drain:               graph_before_drain,
    graph_after_drain:                graph_after_drain,
    graph_after_serialization:        ctx.graph_stats,
    readbacks:                        ctx.readbacks,
  }
  File.write(output_path, output.to_json + "\n")
  puts({trace: output_path, capture_control: output[:capture_control], reference_activation_down: direct_down_check}.to_json)
  Profile.write_from_env
  exit 2 unless controls_passed
{% else %}
  abort "build with -Ddrama3_metal on Darwin"
{% end %}
