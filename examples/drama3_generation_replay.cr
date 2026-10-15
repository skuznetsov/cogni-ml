require "digest/sha256"
require "json"
require "../src/ml/audio/drama3/backbone_resident"
require "../src/ml/audio/drama3/tower_resident"

{% if flag?(:drama3_metal) && flag?(:darwin) && !flag?(:cpu_only) %}
  include ML::Audio::Drama3

  module Drama3GenerationReplay
    extend self

    CHECKPOINTS        = [0, 1, 2, 4, 8, 16, 32, 64, 128, 192, 256, 384, 512, 640]
    TAIL_WINDOW_FRAMES = 128

    def integer(value : JSON::Any, label : String) : Int32
      raw = value.raw
      unless raw.is_a?(Int64) && raw.in?(Int32::MIN.to_i64..Int32::MAX.to_i64)
        raise ArgumentError.new("#{label} must be an int32 JSON integer")
      end
      raw.to_i32
    end

    def read_prompt_ids(report : JSON::Any) : Array(Int32)
      values = report["prompt_ids"].as_a
      raise ArgumentError.new("prompt_ids must not be empty") if values.empty?
      values.map_with_index { |value, index| integer(value, "prompt_ids[#{index}]") }
    end

    def read_frames(report : JSON::Any, codebooks : Int32, semantic_begin : Int32,
                    semantic_end : Int32, vocab_size : Int32) : Array(Array(Int32))
      report["frames"].as_a.map_with_index do |frame, frame_index|
        values = frame.as_a.map_with_index do |value, column|
          integer(value, "frames[#{frame_index}][#{column}]")
        end
        raise ArgumentError.new("frames[#{frame_index}] width mismatch") unless values.size == codebooks + 1
        unless values[0].in?(semantic_begin..semantic_end)
          raise ArgumentError.new("frames[#{frame_index}] main token outside semantic range")
        end
        raise ArgumentError.new("frames[#{frame_index}] initial residual code mismatch") unless values[1] == values[0] - semantic_begin
        unless values.each_with_index.all? { |value, index| index == 0 || value.in?(0...vocab_size) }
          raise ArgumentError.new("frames[#{frame_index}] code token outside vocabulary")
        end
        values
      end
    end

    def capture_steps(frame_count : Int32) : Array(Int32)
      raise ArgumentError.new("frame_count must be non-negative") if frame_count < 0
      tail_start = frame_count > TAIL_WINDOW_FRAMES ? frame_count - TAIL_WINDOW_FRAMES : 0
      (CHECKPOINTS.select { |index| index <= frame_count } + (tail_start..frame_count).to_a).uniq.sort
    end
  end

  abort "usage: drama3_generation_replay MODEL_DIR GENERATED_WAV_JSON OUTPUT_JSON" unless ARGV.size == 3
  model_path = File.expand_path(ARGV[0])
  report_path = File.expand_path(ARGV[1])
  output_path = File.expand_path(ARGV[2])
  abort "model directory is missing: #{model_path}" unless Dir.exists?(model_path)
  abort "generation report is missing: #{report_path}" unless File.file?(report_path)
  abort "output already exists: #{output_path}" if File.exists?(output_path)

  Profile.configure_from_env!
  raise ArgumentError.new("profiling must be disabled for common-prefix replay") if Profile.enabled?
  replay_started = Time.instant

  report_bytes = File.read(report_path)
  report = JSON.parse(report_bytes)
  raise ArgumentError.new("generation report must come from resident Metal") unless report["backend"].as_s == "metal_resident"
  generation_truncated = report["truncated"].as_bool
  stopped_on_eos = report["stopped_on_eos"].as_bool
  raise ArgumentError.new("generation report termination flags disagree") unless generation_truncated == !stopped_on_eos
  prompt_ids = Drama3GenerationReplay.read_prompt_ids(report)
  prompt_rows = Drama3GenerationReplay.integer(report["prompt_rows"], "prompt_rows")
  raise ArgumentError.new("prompt_rows must match prompt_ids") unless prompt_rows == prompt_ids.size

  config_path = File.join(model_path, "config.json")
  index_path = File.join(model_path, "model.safetensors.index.json")
  config_bytes = File.read(config_path)
  raise ArgumentError.new("model index is missing: #{index_path}") unless File.file?(index_path)
  index_bytes = File.read(index_path)
  config_json = JSON.parse(config_bytes)
  backbone_config = BackboneConfig.new(config_json)
  tower_config = TowerConfig.from_json(config_json.to_json)
  raise ArgumentError.new("common-prefix replay requires ten codebooks") unless tower_config.num_codebooks == 10
  frames = Drama3GenerationReplay.read_frames(report, tower_config.num_codebooks,
    tower_config.semantic_begin, tower_config.semantic_end, tower_config.vocab_size)
  raise ArgumentError.new("common-prefix replay requires generated frames") if frames.empty?
  total_rows = prompt_rows + frames.size
  raise ArgumentError.new("resident cache capacity exceeds 1024 rows") unless total_rows.in?(1..1024)

  setup_started = Time.instant
  store_started = Time.instant
  store = WeightStore.new(model_path)
  head = store.matrix("lm_head.weight")
  expected_main_rows = 1 + tower_config.semantic_end - tower_config.semantic_begin + 1
  raise ArgumentError.new("main head dimensions mismatch") unless head.cols == backbone_config.hidden && head.rows == expected_main_rows
  embedding_rows = store.matrix("backbone.embed_tokens.weight").rows
  unless prompt_ids.all? { |id| id.in?(0...embedding_rows) }
    raise ArgumentError.new("prompt token outside embedding vocabulary")
  end
  model_load_seconds = (Time.instant - store_started).total_seconds

  context = Resident::Context.new
  backbone = BackboneResident.new(backbone_config, store, context, total_rows)
  tower = TowerResident.new(tower_config, store, context, total_rows)
  resident_setup_seconds = (Time.instant - setup_started).total_seconds
  rows = [] of Array(Int32)
  prompt_ids.each { |id| rows << ([id] + Array(Int32).new(tower_config.num_codebooks, 0)) }
  rows.concat(frames)
  steps = Drama3GenerationReplay.capture_steps(frames.size.to_i32)
  captured = [] of NamedTuple(name: String, buffer: Resident::Buffer)
  capture_times = [] of NamedTuple(step: Int32, enqueue_elapsed_seconds: Float64)

  initial_graph = context.graph_stats
  raise ArgumentError.new("resident context must start with zero submissions") unless initial_graph["submitted"] == 0 && initial_graph["completion_waits"] == 0
  enqueue_started = Time.instant
  rows.each_with_index do |current_values, row_index|
    current = tower.upload_frame(current_values)
    token = backbone.embedding(current.slice(0, 1))
    mixed = tower.embed_backbone(current, token)
    hidden = backbone.step(mixed)

    step = row_index - prompt_rows + 1
    selected_step = steps.includes?(step)
    main_logits = selected_step ? context.matvec(head, hidden) : nil
    next_row = rows[row_index + 1]?
    tower_output = if next_values = next_row
                     next_frame = tower.upload_frame(next_values)
                     tower.teacher_force(hidden, current, next_frame)
                   else
                     nil
                   end

    if selected_step
      captured << {name: "generation.step#{step}.backbone_hidden", buffer: hidden}
      captured << {name: "generation.step#{step}.main_logits", buffer: main_logits.not_nil!}
      if output = tower_output
        captured << {name: "generation.step#{step}.tower_hidden", buffer: output.hidden}
        output.head_logits.each_with_index do |logits, index|
          captured << {name: "generation.step#{step}.tower_logits.#{index}", buffer: logits}
        end
      end
    end

    # Match resident prefill's bounded-resource cadence while keeping every
    # activation and cache update on the same ordered command queue.
    context.submit
    if selected_step
      capture_times << {step: step, enqueue_elapsed_seconds: (Time.instant - enqueue_started).total_seconds}
    end
  end
  enqueue_seconds = (Time.instant - enqueue_started).total_seconds

  expected_capture_count = steps.size * 2 + steps.count { |step| step < frames.size } * 10
  raise ArgumentError.new("captured stage count mismatch") unless captured.size == expected_capture_count
  raise ArgumentError.new("duplicate captured stage name") unless captured.map(&.[:name]).uniq.size == captured.size
  raise ArgumentError.new("capture caused a host readback before the terminal drain") unless context.readbacks.values.all?(&.zero?)
  raise ArgumentError.new("diagnostic readback before terminal drain") unless context.readbacks["diagnostic"] == 0

  before_drain = context.graph_stats
  drain_started = Time.instant
  context.finish
  drain_seconds = (Time.instant - drain_started).total_seconds
  after_drain = context.graph_stats
  raise ArgumentError.new("final drain left pending work") unless after_drain["pending"] == 0 && after_drain["completed"] == after_drain["submitted"]
  raise ArgumentError.new("expected every replay row to be submitted") unless after_drain["submitted"] == total_rows

  read_started = Time.instant
  trace_records = captured.map do |entry|
    buffer = entry[:buffer]
    {name: entry[:name], shape: [buffer.count], dtype: "float32", values: context.read(buffer, "terminal")}
  end
  readback_seconds = (Time.instant - read_started).total_seconds
  after_reads = context.graph_stats
  unless after_reads["submitted"] == after_drain["submitted"] &&
         after_reads["completed"] == after_drain["completed"] &&
         after_reads["completion_waits"] == after_drain["completion_waits"]
    raise ArgumentError.new("terminal reads added a submission or completion wait")
  end
  expected_terminal_bytes = captured.sum { |entry| entry[:buffer].count.to_i64 * 4 }
  raise ArgumentError.new("terminal readback ledger mismatch") unless context.readbacks["terminal"] == expected_terminal_bytes
  raise ArgumentError.new("unexpected diagnostic readback") unless context.readbacks["diagnostic"] == 0
  raise ArgumentError.new("finite-status readback ledger mismatch") unless context.readbacks["control"] == captured.size.to_i64 * 4

  execution = {
    context_dense_reduction:  context.dense_reduction,
    backbone_dense_reduction: backbone.dense_reduction,
    expert_down_reduction:    backbone.expert_down_reduction,
    batch_experts:            backbone.batch_experts,
  }
  File.write(output_path, {
    format:            "drama3-generation-common-prefix-v1",
    source_report:     {path: report_path, sha256: Digest::SHA256.hexdigest(report_bytes)},
    source_generation: {
      elapsed_seconds:    report["elapsed_seconds"],
      generation_seconds: report["phase_seconds"]["generation"],
      stopped_on_eos:     stopped_on_eos,
      truncated:          generation_truncated,
    },
    prompt_ids:   prompt_ids,
    prompt_rows:  prompt_rows,
    frames:       frames,
    frame_count:  frames.size,
    model_config: {
      hidden_size:      backbone_config.hidden,
      main_vocab_size:  expected_main_rows,
      tower_dim:        tower_config.dim,
      tower_vocab_size: tower_config.vocab_size,
      num_codebooks:    tower_config.num_codebooks,
    },
    model_files: {
      "config.json"                  => Digest::SHA256.hexdigest(config_bytes),
      "model.safetensors.index.json" => Digest::SHA256.hexdigest(index_bytes),
    },
    capture_steps:  steps,
    capture_policy: {
      checkpoints:        Drama3GenerationReplay::CHECKPOINTS,
      tail_window_frames: Drama3GenerationReplay::TAIL_WINDOW_FRAMES,
      tail_inclusive:     true,
    },
    capture_enqueue_times: capture_times,
    execution:             execution,
    timings:               {
      model_load_seconds:        model_load_seconds,
      resident_setup_seconds:    resident_setup_seconds,
      enqueue_seconds:           enqueue_seconds,
      final_drain_seconds:       drain_seconds,
      terminal_readback_seconds: readback_seconds,
      total_seconds:             (Time.instant - replay_started).total_seconds,
    },
    terminal_step: {
      index:                  frames.size,
      measured_in_generation: !generation_truncated,
      interpretation:         generation_truncated ? "counterfactual_after_truncated_prefix" : "actual_terminal_prediction_after_reported_frames",
    },
    graph_before_final_drain: before_drain,
    graph_after_final_drain:  after_drain,
    graph_after_reads:        after_reads,
    readbacks:                context.readbacks,
    traces:                   trace_records,
  }.to_pretty_json)
{% else %}
  abort "build with -Ddrama3_metal on Darwin"
{% end %}
