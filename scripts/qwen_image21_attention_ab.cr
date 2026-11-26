#!/usr/bin/env crystal

# Real 768x768 Qwen-Image 2.1 attention A/B. This is an evidence runner, not
# an automatic performance-promotion gate: it fails closed on pinned inputs,
# route invariants, finite outputs, parity, and cache-hit equivalence, while
# reporting paired wall/GPU timings and host-load observations without a speed
# threshold.
require "../src/ml/bench_load_guard"
require "../src/ml/gguf/qwen_image21_conditioning_bundle"
require "../src/ml/gguf/qwen_image21_metal"
require "../src/ml/gguf/qwen_image21_weights"
require "../src/ml/gguf/qwen_image21_flow_match"

private QWEN_IMAGE21_AB_REVISION         = "790c92633540aa0cb11d9abf19eb46d861714758"
private QWEN_IMAGE21_AB_CONDITIONING_SHA = "14f1c790d0edeb86e007ee61a43e8495d3eec02b83a08fd43e9faef34edcb86b"
private QWEN_IMAGE21_AB_GGUF_SHA         = "51998ad7c068ce7d68e233237537900ffe874ab4d5c72e20758f5f18ceb15b8a"
private QWEN_IMAGE21_AB_MAX_ABS          =  5.0e-2
private QWEN_IMAGE21_AB_MIN_COSINE       = 0.99999
private QWEN_IMAGE21_AB_CACHE_MAX_ABS    =  1.0e-4

class QwenImage21AttentionABUncachedStack
  include ML::GGUF::QwenImage21LayerStackBackend
  include ML::GGUF::QwenImage21ResidentInputStackBackend

  def initialize(@inner : ML::GGUF::QwenImage21MetalLayerStackBackend)
  end

  def forward_layers(
    hidden : Array(Float32), token_count : Int32,
    modulation : Array(Float32), positions : Array(StaticArray(Int32, 3)),
    image_ids : Array(Int32), layers : Array(ML::GGUF::QwenImage21BlockWeights),
    config : ML::GGUF::QwenImage21BlockConfig, key_valid : Array(Bool)?,
    target_start : Int32?,
  ) : Array(Float32)
    @inner.forward_layers(
      hidden, token_count, modulation, positions, image_ids, layers,
      config, key_valid, target_start,
    )
  end

  def forward_resident_input(
    image_input : Array(Float32), projected_text : Array(Float32),
    time_input : Array(Float32), img_mask : Array(Bool),
    layout : ML::GGUF::QwenImage21TokenLayout,
    weights : ML::GGUF::QwenImage21TransformerWeights,
    config : ML::GGUF::QwenImage21TransformerConfig,
  ) : Array(Float32)?
    @inner.forward_resident_input(
      image_input, projected_text, time_input, img_mask, layout, weights, config,
    )
  end

  # TransformerCPU.forward selects this method for the causal image suffix.
  # Deliberately use the full resident-input path to make the reference uncached.
  def forward_resident_cached_input(
    image_input : Array(Float32), encoder_hidden : Array(Float32),
    projected_text : Array(Float32), time_input : Array(Float32),
    img_mask : Array(Bool), layout : ML::GGUF::QwenImage21TokenLayout,
    weights : ML::GGUF::QwenImage21TransformerWeights,
    config : ML::GGUF::QwenImage21TransformerConfig, prefix_tokens : Int32,
  ) : Array(Float32)?
    @inner.forward_resident_input(
      image_input, projected_text, time_input, img_mask, layout, weights, config,
    )
  end
end

private class QwenImage21AttentionABRoute
  getter name : String
  getter result : ML::GGUF::QwenImage21TransformerResult
  getter stats : ML::GGUF::QwenImage21MetalBlockStats
  getter wall_ms : Float64
  getter gpu_ms : Float64?
  getter host_before : ML::BenchLoadGuard::HostLoad?
  getter host_after : ML::BenchLoadGuard::HostLoad?

  def initialize(
    @name, @result, @stats, @wall_ms, @gpu_ms, @host_before, @host_after,
  )
  end
end

private class QwenImage21AttentionABModeRun
  getter mode : String
  getter build : QwenImage21AttentionABRoute
  getter hit : QwenImage21AttentionABRoute
  getter uncached : QwenImage21AttentionABRoute

  def initialize(@mode, @build, @hit, @uncached)
  end

  def route(name : String) : QwenImage21AttentionABRoute
    case name
    when "build"    then @build
    when "hit"      then @hit
    when "uncached" then @uncached
    else                 raise ArgumentError.new("unknown route: #{name}")
    end
  end
end

private record QwenImage21AttentionABObservation,
  pair : Int32,
  phase : String,
  run : QwenImage21AttentionABModeRun

private def qwen_image21_ab_usage : NoReturn
  abort <<-USAGE
  usage: crystal run scripts/qwen_image21_attention_ab.cr -- MODEL.gguf CONDITIONING.json [pairs=6] [steps=40]

  Optional forms: --pairs=N --steps=N
                  --load-warning-threshold=20 --load-total-warning-threshold=50
  N=1 is a pilot only; promotion-quality sampling requires both measured orders
  and a larger paired set. Host-load thresholds annotate observations; they do
  not alter timings or impose a performance cutoff.
  USAGE
end

private def qwen_image21_ab_parse_args : {String, String, Int32, Int32, Float64, Float64}
  args = ARGV.dup
  qwen_image21_ab_usage if args.size < 2 || args.includes?("--help") || args.includes?("-h")
  gguf_path = args.shift
  conditioning_path = args.shift
  pairs = 6
  steps = 40
  per_process_threshold = 20.0
  total_threshold = 50.0
  positional = 0

  args.each do |arg|
    if arg.starts_with?("--pairs=")
      pairs = arg.split("=", 2)[1].to_i? || qwen_image21_ab_usage
    elsif arg.starts_with?("--steps=")
      steps = arg.split("=", 2)[1].to_i? || qwen_image21_ab_usage
    elsif arg.starts_with?("--load-warning-threshold=")
      per_process_threshold = arg.split("=", 2)[1].to_f? || qwen_image21_ab_usage
    elsif arg.starts_with?("--load-total-warning-threshold=")
      total_threshold = arg.split("=", 2)[1].to_f? || qwen_image21_ab_usage
    elsif arg.starts_with?("-")
      qwen_image21_ab_usage
    else
      positional += 1
      case positional
      when 1 then pairs = arg.to_i? || qwen_image21_ab_usage
      when 2 then steps = arg.to_i? || qwen_image21_ab_usage
      else        qwen_image21_ab_usage
      end
    end
  end

  qwen_image21_ab_usage unless pairs > 0 && pairs <= 100 && steps >= 2 && steps <= 100
  qwen_image21_ab_usage unless per_process_threshold >= 0.0 && total_threshold >= 0.0
  {gguf_path, conditioning_path, pairs, steps, per_process_threshold, total_threshold}
end

private def qwen_image21_ab_sha256_file(path : String) : String
  digest = Digest::SHA256.new
  File.open(path) { |file| digest.update(file) }
  digest.final.hexstring
end

private def qwen_image21_ab_metrics(
  expected : Array(Float32), actual : Array(Float32),
) : {Float64, Float64, Float64}
  raise ArgumentError.new("transformer output sizes differ") unless expected.size == actual.size
  unless expected.all?(&.finite?) && actual.all?(&.finite?)
    raise ArgumentError.new("transformer output contains non-finite values")
  end

  max_abs = 0.0_f64
  squared_error = 0.0_f64
  dot = 0.0_f64
  expected_norm = 0.0_f64
  actual_norm = 0.0_f64
  expected.each_with_index do |reference, index|
    value = actual[index]
    difference = (reference - value).to_f64
    max_abs = Math.max(max_abs, difference.abs)
    squared_error += difference ** 2
    dot += reference.to_f64 * value
    expected_norm += reference.to_f64 ** 2
    actual_norm += value.to_f64 ** 2
  end
  unless expected_norm > 0.0_f64 && actual_norm > 0.0_f64
    raise ArgumentError.new("transformer output has a zero norm")
  end
  {max_abs, Math.sqrt(squared_error / expected.size), dot / Math.sqrt(expected_norm * actual_norm)}
end

private def qwen_image21_ab_compare(
  label : String, expected : Array(Float32), actual : Array(Float32),
  strict_max_abs : Float64 = QWEN_IMAGE21_AB_MAX_ABS,
  enforce_cosine : Bool = true,
) : {Float64, Float64, Float64}
  metrics = qwen_image21_ab_metrics(expected, actual)
  puts "#{label} max_abs=#{metrics[0]} rms=#{metrics[1]} cosine=#{metrics[2]}"
  accepted = metrics[0] < strict_max_abs
  accepted &&= metrics[2] > QWEN_IMAGE21_AB_MIN_COSINE if enforce_cosine
  abort "#{label} exceeded numerical gate (max_abs < #{strict_max_abs}, cosine > #{QWEN_IMAGE21_AB_MIN_COSINE})" unless accepted
  metrics
end

private def qwen_image21_ab_host_status(load : ML::BenchLoadGuard::HostLoad?) : String
  return "unavailable" unless host = load
  return "unknown" if host.process_loads.empty?
  host.busy? ? "busy" : "quiet"
end

private def qwen_image21_ab_host_detail(load : ML::BenchLoadGuard::HostLoad?) : String
  return "unavailable" unless host = load
  status = qwen_image21_ab_host_status(host)
  detail = "#{status}(other_cpu_sum=#{host.total_cpu.round(1)}%)"
  if status == "busy"
    top = host.process_loads.first(3).map do |process|
      "#{process.pid}:#{process.cpu.round(1)}%"
    end.join(",")
    detail += "[#{top}]"
  end
  detail
end

private def qwen_image21_ab_host_sample(
  per_process_threshold : Float64, total_threshold : Float64,
) : ML::BenchLoadGuard::HostLoad
  ML::BenchLoadGuard.sample(per_process_threshold, total_threshold)
end

private def qwen_image21_ab_forward(
  route_name : String,
  stack : ML::GGUF::QwenImage21MetalLayerStackBackend,
  backend : ML::GGUF::QwenImage21MetalProjectionBackend,
  model : ML::GGUF::QwenImage21Weights,
  latents : Array(Float32),
  conditioning : ML::GGUF::QwenImage21ConditioningBundle,
  img_mask : Array(Bool),
  shapes : Array(StaticArray(Int32, 3)),
  layout : ML::GGUF::QwenImage21TokenLayout,
  timestep : Float32,
  target_offset : Int32,
  target_count : Int32,
  uncached : Bool,
  per_process_threshold : Float64,
  total_threshold : Float64,
) : QwenImage21AttentionABRoute
  host_before = qwen_image21_ab_host_sample(per_process_threshold, total_threshold)
  result = nil.as(ML::GGUF::QwenImage21TransformerResult?)
  wall = Time.measure do
    if uncached
      adapter = QwenImage21AttentionABUncachedStack.new(stack)
      result = ML::GGUF::QwenImage21TransformerCPU.forward(
        latents, conditioning.encoder_hidden_states, timestep, shapes, img_mask,
        model.transformer_weights, model.transformer_config,
        encoder_hidden_states_mask: conditioning.encoder_hidden_states_mask,
        backend: backend, layer_stack_backend: adapter,
      )
    else
      result = ML::GGUF::QwenImage21TransformerCPU.forward(
        latents, conditioning.encoder_hidden_states, timestep, shapes, img_mask,
        model.transformer_weights, model.transformer_config,
        encoder_hidden_states_mask: conditioning.encoder_hidden_states_mask,
        backend: backend, layer_stack_backend: stack,
      )
    end
  end
  host_after = qwen_image21_ab_host_sample(per_process_threshold, total_threshold)
  stats = stack.last_stats || abort("#{route_name} did not expose Metal stats")
  output = result || abort("#{route_name} did not return a transformer output")
  expected_output_size = layout.token_count * model.transformer_config.output_dim
  abort "#{route_name} returned a partial transformer output" unless output.output.size == expected_output_size
  abort "#{route_name} returned an unexpected target slice" unless output.output[target_offset, target_count].size == target_count
  QwenImage21AttentionABRoute.new(
    route_name, output, stats, wall.total_milliseconds, stats.gpu_command_ms,
    host_before, host_after,
  )
end

private def qwen_image21_ab_assert_stats(
  route : QwenImage21AttentionABRoute, active_tokens : Int32,
  target_tokens : Int32, stack : ML::GGUF::QwenImage21MetalLayerStackBackend,
  invocations : Int32, builds : Int32, hits : Int32,
) : Nil
  stats = route.stats
  abort "#{route.name} expected one Metal command buffer, got #{stats.command_buffers}" unless stats.command_buffers == 1
  abort "#{route.name} had intermediate Metal readbacks" unless stats.intermediate_readbacks == 0
  abort "#{route.name} expected one final Metal readback, got #{stats.final_readbacks}" unless stats.final_readbacks == 1
  abort "#{route.name} active tokens=#{stats.active_tokens}, expected #{active_tokens}" unless stats.active_tokens == active_tokens
  abort "#{route.name} image projection rows=#{stats.image_projection_rows}, expected #{target_tokens}" unless stats.image_projection_rows == target_tokens
  abort "#{route.name} invocation count changed unexpectedly" unless stack.invocations == invocations
  abort "#{route.name} prefix build count changed unexpectedly" unless stack.prefix_cache_builds == builds
  abort "#{route.name} prefix hit count changed unexpectedly" unless stack.prefix_cache_hits == hits
end

private def qwen_image21_ab_run_mode(
  mode : String,
  mode_value : String,
  model : ML::GGUF::QwenImage21Weights,
  backend : ML::GGUF::QwenImage21MetalProjectionBackend,
  conditioning : ML::GGUF::QwenImage21ConditioningBundle,
  img_mask : Array(Bool),
  shapes : Array(StaticArray(Int32, 3)),
  layout : ML::GGUF::QwenImage21TokenLayout,
  initial_latents : Array(Float32),
  step1_latents : Array(Float32),
  timestep0 : Float32,
  timestep1 : Float32,
  target_offset : Int32,
  target_count : Int32,
  target_tokens : Int32,
  per_process_threshold : Float64,
  total_threshold : Float64,
) : QwenImage21AttentionABModeRun
  ENV["QWEN_IMAGE21_ATTENTION_TILE"] = mode_value
  policy = ML::GGUF::QwenImage21MetalAttentionPolicy
  expected_kernel = mode == "candidate" ? ML::GGUF::QwenImage21MetalAttentionPolicy::TILED_KERNEL : ML::GGUF::QwenImage21MetalAttentionPolicy::LEGACY_KERNEL
  {layout.token_count, target_tokens}.each do |query_tokens|
    selected_kernel = policy.kernel_name(
      model.transformer_config.block.head_dim, layout.token_count, query_tokens,
    )
    abort "#{mode} did not select #{expected_kernel} at #{query_tokens} queries" unless selected_kernel == expected_kernel
  end
  stack = ML::GGUF::QwenImage21MetalLayerStackBackend.new
  begin
    build = qwen_image21_ab_forward(
      "build", stack, backend, model, initial_latents, conditioning, img_mask,
      shapes, layout, timestep0, target_offset, target_count, false,
      per_process_threshold, total_threshold,
    )
    qwen_image21_ab_assert_stats(build, layout.token_count, target_tokens, stack, 1, 1, 0)

    hit = qwen_image21_ab_forward(
      "hit", stack, backend, model, step1_latents, conditioning, img_mask,
      shapes, layout, timestep1, target_offset, target_count, false,
      per_process_threshold, total_threshold,
    )
    qwen_image21_ab_assert_stats(hit, target_tokens, target_tokens, stack, 2, 1, 1)

    uncached = qwen_image21_ab_forward(
      "uncached", stack, backend, model, step1_latents, conditioning, img_mask,
      shapes, layout, timestep1, target_offset, target_count, true,
      per_process_threshold, total_threshold,
    )
    qwen_image21_ab_assert_stats(uncached, layout.token_count, target_tokens, stack, 3, 1, 1)
    run = QwenImage21AttentionABModeRun.new(mode, build, hit, uncached)
    {"build", "hit", "uncached"}.each do |name|
      route = run.route(name)
      puts "mode=#{mode} route=#{name} wall_ms=#{route.wall_ms.round(3)} " \
           "gpu_command_ms=#{route.gpu_ms.try(&.round(3).to_s) || "unavailable"} " \
           "host_before=#{qwen_image21_ab_host_detail(route.host_before)} " \
           "host_after=#{qwen_image21_ab_host_detail(route.host_after)}"
    end
    hit = run.hit
    uncached = run.uncached
    qwen_image21_ab_compare(
      "mode=#{mode} hit_vs_uncached_full", uncached.result.output, hit.result.output,
      QWEN_IMAGE21_AB_CACHE_MAX_ABS, false,
    )
    qwen_image21_ab_compare(
      "mode=#{mode} hit_vs_uncached_target",
      uncached.result.output[target_offset, target_count],
      hit.result.output[target_offset, target_count],
      QWEN_IMAGE21_AB_CACHE_MAX_ABS, false,
    )
    run
  ensure
    stack.close
  end
end

private def qwen_image21_ab_assert_inputs_unchanged(
  conditioning : ML::GGUF::QwenImage21ConditioningBundle,
  initial_latents : Array(Float32),
  fixed_initial_latents : Array(Float32),
  step1_latents : Array(Float32),
  fixed_step1_latents : Array(Float32),
  img_mask : Array(Bool),
  fixed_img_mask : Array(Bool),
  shapes : Array(StaticArray(Int32, 3)),
  fixed_shapes : Array(StaticArray(Int32, 3)),
  encoder : Array(Float32),
  fixed_encoder : Array(Float32),
  encoder_mask : Array(Bool),
  fixed_encoder_mask : Array(Bool),
) : Nil
  abort "transformer route mutated the held-fixed step-1 latent" unless step1_latents == fixed_step1_latents
  abort "transformer route mutated the conditioning initial latent" unless initial_latents == fixed_initial_latents
  abort "transformer route mutated encoder states" unless conditioning.encoder_hidden_states == fixed_encoder && encoder == fixed_encoder
  abort "transformer route mutated encoder mask" unless conditioning.encoder_hidden_states_mask == fixed_encoder_mask && encoder_mask == fixed_encoder_mask
  abort "transformer route mutated the image mask" unless img_mask == fixed_img_mask
  abort "transformer route mutated image shapes" unless shapes == fixed_shapes
end

private def qwen_image21_ab_compare_modes(
  pair_label : String,
  baseline : QwenImage21AttentionABModeRun,
  candidate : QwenImage21AttentionABModeRun,
  target_offset : Int32,
  target_count : Int32,
) : Nil
  {"build", "hit", "uncached"}.each do |name|
    expected = baseline.route(name).result.output
    actual = candidate.route(name).result.output
    qwen_image21_ab_compare("#{pair_label} route=#{name} full", expected, actual)
    qwen_image21_ab_compare(
      "#{pair_label} route=#{name} target",
      expected[target_offset, target_count], actual[target_offset, target_count],
    )
  end
end

private def qwen_image21_ab_median(values : Array(Float64)) : Float64
  raise ArgumentError.new("cannot compute median of an empty sample") if values.empty?
  sorted = values.sort
  (sorted[(sorted.size - 1) // 2] + sorted[sorted.size // 2]) / 2.0
end

private def qwen_image21_ab_ratio(numerator : Float64?, denominator : Float64?) : String
  return "unavailable" unless top = numerator
  return "unavailable" unless bottom = denominator
  return "unavailable" unless bottom > 0.0
  (top / bottom).round(4).to_s
end

private def qwen_image21_ab_report_pair_ratios(
  pair : Int32,
  baseline : QwenImage21AttentionABModeRun,
  candidate : QwenImage21AttentionABModeRun,
) : Nil
  {"build", "hit", "uncached"}.each do |name|
    base = baseline.route(name)
    cand = candidate.route(name)
    puts "pair=#{pair} route=#{name} candidate_over_baseline " \
         "wall_ratio=#{qwen_image21_ab_ratio(cand.wall_ms, base.wall_ms)} " \
         "gpu_command_ratio=#{qwen_image21_ab_ratio(cand.gpu_ms, base.gpu_ms)} " \
         "decision=reported_not_gated"
  end
end

private def qwen_image21_ab_restore_env(key : String, value : String?) : Nil
  if value
    ENV[key] = value
  else
    ENV.delete(key)
  end
end

gguf_path, conditioning_path, pairs, steps, per_process_threshold, total_threshold = qwen_image21_ab_parse_args
abort "GGUF file not found: #{gguf_path}" unless File.file?(gguf_path)
abort "conditioning manifest not found: #{conditioning_path}" unless File.file?(conditioning_path)
abort "Metal backend unavailable" unless ML::GGUF::QwenImage21MetalProjectionBackend.available?

prior_profile = ENV["QWEN_IMAGE21_PROFILE"]?
prior_attention_tile = ENV["QWEN_IMAGE21_ATTENTION_TILE"]?
model = nil.as(ML::GGUF::QwenImage21Weights?)
begin
  ENV["QWEN_IMAGE21_PROFILE"] = "1"
  ENV["QWEN_IMAGE21_ATTENTION_TILE"] = "0"

  actual_gguf_sha = qwen_image21_ab_sha256_file(gguf_path)
  abort "GGUF SHA-256 mismatch: got #{actual_gguf_sha}, expected #{QWEN_IMAGE21_AB_GGUF_SHA}" unless actual_gguf_sha == QWEN_IMAGE21_AB_GGUF_SHA
  conditioning = ML::GGUF::QwenImage21ConditioningBundle.load(conditioning_path)
  abort "conditioning seed must be 7" unless conditioning.seed == 7
  abort "conditioning model revision mismatch" unless conditioning.model_revision == QWEN_IMAGE21_AB_REVISION
  abort "conditioning payload SHA-256 mismatch" unless conditioning.conditioning_payload_sha256 == QWEN_IMAGE21_AB_CONDITIONING_SHA
  abort "expected 768x768 conditioning" unless conditioning.image_width == 768 && conditioning.image_height == 768
  target_tokens = conditioning.latent_width * conditioning.latent_height
  abort "expected 2304 target tokens" unless target_tokens == 2304
  encoder_tokens = conditioning.encoder_hidden_states.size // ML::GGUF::QwenImage21Weights::HIDDEN_DIM
  abort "expected exactly 105 text tokens" unless encoder_tokens == 105 &&
                                                  conditioning.encoder_hidden_states.size == encoder_tokens * ML::GGUF::QwenImage21Weights::HIDDEN_DIM

  loaded_model = ML::GGUF::QwenImage21Weights.from_gguf(gguf_path)
  model = loaded_model
  abort "expected the real 32-layer Qwen-Image 2.1 GGUF" unless loaded_model.layers.size == 32
  config = loaded_model.transformer_config
  weights = loaded_model.transformer_weights
  shapes = conditioning.img_shapes.dup
  img_mask = conditioning.encoder_img_mask.dup
  (target_tokens // 4).times { img_mask << true }
  layout = ML::GGUF::QwenImage21TransformerCPU.build_layout(
    img_mask, shapes, encoder_tokens, conditioning.encoder_hidden_states_mask,
  )
  abort "expected the 105 text + 2304 target = 2409 token layout" unless layout.token_count == 2409 &&
                                                                         layout.target_token_mask.first(encoder_tokens).none? && layout.target_token_mask[encoder_tokens..].all?
  abort "conditioning latent tensor shape mismatch" unless conditioning.initial_target_latents.size == target_tokens * config.input_dim
  abort "input/output dimensions must match for FlowMatch Euler" unless config.input_dim == config.output_dim

  schedule = ML::GGUF::QwenImage21FlowMatch.schedule(steps, target_tokens)
  timestep0 = schedule.model_timestep(0)
  timestep1 = schedule.model_timestep(1)
  abort "expected distinct step-0 and step-1 timesteps" if timestep0 == timestep1
  target_offset = encoder_tokens * config.output_dim
  target_count = target_tokens * config.output_dim
  projection_backend = ML::GGUF::QwenImage21MetalProjectionBackend.new(strict: true)
  fixed_initial_latents = conditioning.initial_target_latents.dup
  fixed_encoder = conditioning.encoder_hidden_states.dup
  fixed_encoder_mask = conditioning.encoder_hidden_states_mask.dup
  fixed_img_mask = img_mask.dup
  fixed_shapes = shapes.dup

  # Produce the common x1 once with the baseline step-0 Euler forward. Candidate
  # build output is never allowed to redefine later inputs in its own mode.
  ENV["QWEN_IMAGE21_ATTENTION_TILE"] = "0"
  derive_stack = ML::GGUF::QwenImage21MetalLayerStackBackend.new
  step1_latents = [] of Float32
  begin
    derive = qwen_image21_ab_forward(
      "derive_step0", derive_stack, projection_backend, loaded_model,
      conditioning.initial_target_latents, conditioning, img_mask, shapes,
      layout, timestep0, target_offset, target_count, false,
      per_process_threshold, total_threshold,
    )
    qwen_image21_ab_assert_stats(derive, layout.token_count, target_tokens, derive_stack, 1, 1, 0)
    step1_latents = schedule.step(
      conditioning.initial_target_latents, derive.result.output.last(target_count), 0,
    )
  ensure
    derive_stack.close
  end
  abort "baseline step-0 Euler did not change the target latent" if step1_latents == fixed_initial_latents
  abort "baseline step-0 Euler produced non-finite step-1 latents" unless step1_latents.all?(&.finite?)
  fixed_step1_latents = step1_latents.dup
  puts "fixed_step1=baseline_step0_euler t0=#{timestep0} t1=#{timestep1} " \
       "initial_target_tokens=#{target_tokens} held_identical=true"

  puts "probe=real_768_attention_ab gguf=#{File.basename(gguf_path)} gguf_sha256=#{actual_gguf_sha} " \
       "conditioning_revision=#{conditioning.model_revision} conditioning_payload_sha256=#{conditioning.conditioning_payload_sha256} " \
       "seed=#{conditioning.seed} image=768x768 text_tokens=#{encoder_tokens} " \
       "target_tokens=#{target_tokens} total_tokens=#{layout.token_count} layers=#{loaded_model.layers.size} " \
       "steps=#{steps} pairs=#{pairs} profile=1 attention_tile_baseline=0 candidate=1 " \
       "command_buffers=1 intermediate_readbacks=0 final_readbacks=1 " \
       "parity_max_abs<#{QWEN_IMAGE21_AB_MAX_ABS} parity_cosine>#{QWEN_IMAGE21_AB_MIN_COSINE} " \
       "cache_hit_vs_uncached_max_abs<#{QWEN_IMAGE21_AB_CACHE_MAX_ABS} speed_cutoff=none"
  puts "host_load_observation=ps_per_process_threshold=#{per_process_threshold}% " \
       "other_process_cpu_sum_threshold=#{total_threshold}% warning_only=true"

  all_runs = [] of QwenImage21AttentionABObservation
  observations = [] of QwenImage21AttentionABObservation

  # Warm each full route triplet once, baseline then candidate, so both
  # attention kernels and the distinct cache-hit route are exercised before
  # measured pairs. The measured set itself alternates AB/BA order.
  puts "phase=warmup order=baseline,candidate forwards_per_mode=build+hit+uncached"
  warm_baseline = qwen_image21_ab_run_mode(
    "baseline", "0", loaded_model, projection_backend, conditioning, img_mask,
    shapes, layout, conditioning.initial_target_latents, step1_latents,
    timestep0, timestep1, target_offset, target_count, target_tokens,
    per_process_threshold, total_threshold,
  )
  qwen_image21_ab_assert_inputs_unchanged(
    conditioning, conditioning.initial_target_latents, fixed_initial_latents,
    step1_latents, fixed_step1_latents, img_mask, fixed_img_mask, shapes,
    fixed_shapes, conditioning.encoder_hidden_states, fixed_encoder,
    conditioning.encoder_hidden_states_mask, fixed_encoder_mask,
  )
  warm_candidate = qwen_image21_ab_run_mode(
    "candidate", "1", loaded_model, projection_backend, conditioning, img_mask,
    shapes, layout, conditioning.initial_target_latents, step1_latents,
    timestep0, timestep1, target_offset, target_count, target_tokens,
    per_process_threshold, total_threshold,
  )
  qwen_image21_ab_compare_modes("warmup", warm_baseline, warm_candidate, target_offset, target_count)
  qwen_image21_ab_assert_inputs_unchanged(
    conditioning, conditioning.initial_target_latents, fixed_initial_latents,
    step1_latents, fixed_step1_latents, img_mask, fixed_img_mask, shapes,
    fixed_shapes, conditioning.encoder_hidden_states, fixed_encoder,
    conditioning.encoder_hidden_states_mask, fixed_encoder_mask,
  )
  all_runs << QwenImage21AttentionABObservation.new(0, "warm", warm_baseline)
  all_runs << QwenImage21AttentionABObservation.new(0, "warm", warm_candidate)

  pairs.times do |index|
    pair = index + 1
    order = index.even? ? ["baseline", "candidate"] : ["candidate", "baseline"]
    puts "phase=measured pair=#{pair}/#{pairs} order=#{order.join(",")} " \
         "pilot_only=#{pairs == 1}"
    outputs = Hash(String, QwenImage21AttentionABModeRun).new
    order.each do |mode|
      value = mode == "baseline" ? "0" : "1"
      outputs[mode] = qwen_image21_ab_run_mode(
        mode, value, loaded_model, projection_backend, conditioning, img_mask,
        shapes, layout, conditioning.initial_target_latents, step1_latents,
        timestep0, timestep1, target_offset, target_count, target_tokens,
        per_process_threshold, total_threshold,
      )
      qwen_image21_ab_assert_inputs_unchanged(
        conditioning, conditioning.initial_target_latents, fixed_initial_latents,
        step1_latents, fixed_step1_latents, img_mask, fixed_img_mask, shapes,
        fixed_shapes, conditioning.encoder_hidden_states, fixed_encoder,
        conditioning.encoder_hidden_states_mask, fixed_encoder_mask,
      )
    end
    baseline = outputs["baseline"]
    candidate = outputs["candidate"]
    qwen_image21_ab_compare_modes("pair=#{pair}", baseline, candidate, target_offset, target_count)
    qwen_image21_ab_report_pair_ratios(pair, baseline, candidate)
    observations << QwenImage21AttentionABObservation.new(pair, "measured", baseline)
    observations << QwenImage21AttentionABObservation.new(pair, "measured", candidate)
    all_runs << observations[-2]
    all_runs << observations[-1]
  end

  {"build", "hit", "uncached"}.each do |route_name|
    {"baseline", "candidate"}.each do |mode|
      sample = observations.select { |observation| observation.run.mode == mode }
        .map { |observation| observation.run.route(route_name) }
      wall_median = qwen_image21_ab_median(sample.map(&.wall_ms))
      gpu_values = sample.compact_map(&.gpu_ms)
      gpu_median = gpu_values.empty? ? "unavailable" : qwen_image21_ab_median(gpu_values).round(3).to_s
      puts "median route=#{route_name} mode=#{mode} n=#{sample.size} " \
           "wall_ms=#{wall_median.round(3)} gpu_command_ms=#{gpu_median}"
    end
    baseline_wall = observations.select { |observation| observation.run.mode == "baseline" }
      .map { |observation| observation.run.route(route_name).wall_ms }
    candidate_wall = observations.select { |observation| observation.run.mode == "candidate" }
      .map { |observation| observation.run.route(route_name).wall_ms }
    baseline_gpu = observations.select { |observation| observation.run.mode == "baseline" }
      .compact_map { |observation| observation.run.route(route_name).gpu_ms }
    candidate_gpu = observations.select { |observation| observation.run.mode == "candidate" }
      .compact_map { |observation| observation.run.route(route_name).gpu_ms }
    gpu_median_ratio = if baseline_gpu.empty? || candidate_gpu.empty?
                         "unavailable"
                       else
                         qwen_image21_ab_ratio(
                           qwen_image21_ab_median(candidate_gpu),
                           qwen_image21_ab_median(baseline_gpu),
                         )
                       end
    puts "summary route=#{route_name} candidate_over_baseline_wall_median=" \
         "#{(qwen_image21_ab_median(candidate_wall) / qwen_image21_ab_median(baseline_wall)).round(4)} " \
         "gpu_command_median_ratio=#{gpu_median_ratio} " \
         "decision=reported_not_gated"
  end

  host_samples = all_runs.flat_map do |observation|
    [observation.run.build.host_before, observation.run.build.host_after,
     observation.run.hit.host_before, observation.run.hit.host_after,
     observation.run.uncached.host_before, observation.run.uncached.host_after]
  end
  noisy_host = false
  unknown_host = false
  host_samples.each do |load|
    status = qwen_image21_ab_host_status(load)
    noisy_host ||= status == "busy"
    unknown_host ||= status == "unknown" || status == "unavailable"
  end
  puts "host_load_summary noise_observed=#{noisy_host} observation_unknown=#{unknown_host} " \
       "policy=warning_only"
  puts "parity=PASS routes=build,hit,uncached scope=full_and_target_outputs " \
       "promotion=NOT_DECIDED (review paired medians and host-load observations)"
ensure
  model.try(&.close)
  qwen_image21_ab_restore_env("QWEN_IMAGE21_PROFILE", prior_profile)
  qwen_image21_ab_restore_env("QWEN_IMAGE21_ATTENTION_TILE", prior_attention_tile)
end
