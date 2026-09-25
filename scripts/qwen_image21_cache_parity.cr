#!/usr/bin/env crystal

# Compare cached-prefix build/hit against the uncached resident-input route for
# one real 768x768 conditioning bundle. This measures transformer-output
# numerical parity only; it does not decode or assess image quality.
# The optional step count selects the two scheduler timesteps; this probe always
# makes exactly three transformer forwards.
require "../src/ml/gguf/qwen_image21_conditioning_bundle"
require "../src/ml/gguf/qwen_image21_metal"
require "../src/ml/gguf/qwen_image21_weights"
require "../src/ml/gguf/qwen_image21_flow_match"

class QwenImage21CacheParityUncachedStack
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

  # TransformerCPU.forward selects this method for a causal target suffix.
  # Delegate it to the same Metal stack's full resident-input method to bypass
  # prefix reuse while retaining identical production projections and inputs.
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

private def output_metrics(expected : Array(Float32), actual : Array(Float32)) : {Float64, Float64, Float64}
  raise ArgumentError.new("transformer output sizes differ") unless expected.size == actual.size
  raise ArgumentError.new("transformer output contains non-finite values") unless expected.all?(&.finite?) && actual.all?(&.finite?)

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
  raise ArgumentError.new("transformer output has a zero norm") unless expected_norm > 0.0_f64 && actual_norm > 0.0_f64
  cosine = dot / Math.sqrt(expected_norm * actual_norm)
  rms = Math.sqrt(squared_error / expected.size)
  {max_abs, rms, cosine}
end

private def metric_passes?(metrics : {Float64, Float64, Float64}, max_abs_limit : Float64) : Bool
  metrics[0] < max_abs_limit && metrics[2] > 0.99999
end

private def report_metrics(label : String, full : {Float64, Float64, Float64}, target : {Float64, Float64, Float64}) : Nil
  puts "#{label} full_max_abs=#{full[0]} full_rms=#{full[1]} full_cosine=#{full[2]} " \
       "target_max_abs=#{target[0]} target_rms=#{target[1]} target_cosine=#{target[2]}"
end

gguf_path = ARGV[0]? || abort "usage: crystal run scripts/qwen_image21_cache_parity.cr -- MODEL.gguf CONDITIONING.json [steps=40]"
conditioning_path = ARGV[1]? || abort "missing CONDITIONING.json"
steps = ARGV[2]?.try(&.to_i) || 40
abort "steps must be in 2..100" unless steps >= 2 && steps <= 100
abort "GGUF file not found: #{gguf_path}" unless File.file?(gguf_path)
abort "conditioning manifest not found: #{conditioning_path}" unless File.file?(conditioning_path)
abort "Metal backend unavailable" unless ML::GGUF::QwenImage21MetalProjectionBackend.available?

conditioning = ML::GGUF::QwenImage21ConditioningBundle.load(conditioning_path)
abort "expected the pinned 768x768 conditioning bundle" unless conditioning.image_width == 768 && conditioning.image_height == 768
target_tokens = conditioning.latent_width * conditioning.latent_height
abort "expected 2304 target tokens" unless target_tokens == 2304
abort "target tokens must form complete image slots" unless target_tokens.divisible_by?(4)
encoder_tokens = conditioning.encoder_hidden_states.size // ML::GGUF::QwenImage21Weights::HIDDEN_DIM
abort "conditioning hidden states are not whole tokens" unless conditioning.encoder_hidden_states.size == encoder_tokens * ML::GGUF::QwenImage21Weights::HIDDEN_DIM

model = ML::GGUF::QwenImage21Weights.from_gguf(gguf_path)
stack = ML::GGUF::QwenImage21MetalLayerStackBackend.new
begin
  abort "expected a real 32-layer Qwen-Image 2.1 GGUF" unless model.layers.size == 32
  config = model.transformer_config
  weights = model.transformer_weights
  shapes = conditioning.img_shapes
  img_mask = conditioning.encoder_img_mask.dup
  (target_tokens // 4).times { img_mask << true }
  layout = ML::GGUF::QwenImage21TransformerCPU.build_layout(
    img_mask, shapes, encoder_tokens, conditioning.encoder_hidden_states_mask,
  )
  abort "expected 105 text tokens followed by a 2304-token target" unless encoder_tokens == 105 && layout.token_count == 2409 &&
                                                                          layout.target_token_mask.first(encoder_tokens).none? &&
                                                                          layout.target_token_mask[encoder_tokens..].all?

  target_start = encoder_tokens
  target_count = target_tokens * config.output_dim
  target_offset = target_start * config.output_dim
  schedule = ML::GGUF::QwenImage21FlowMatch.schedule(steps, target_tokens)
  projection_backend = ML::GGUF::QwenImage21MetalProjectionBackend.new(strict: true)
  fixed_latents = conditioning.initial_target_latents.dup
  fixed_encoder = conditioning.encoder_hidden_states.dup
  fixed_encoder_mask = conditioning.encoder_hidden_states_mask.dup
  fixed_img_mask = img_mask.dup
  fixed_shapes = shapes.dup
  timestep0 = schedule.model_timestep(0)
  timestep1 = schedule.model_timestep(1)
  abort "expected distinct step-0 and step-1 model timesteps" if timestep0 == timestep1

  puts "probe=transformer-output-numerical-parity-only gguf=#{File.basename(gguf_path)} " \
       "conditioning_revision=#{conditioning.model_revision} " \
       "conditioning_payload_sha256=#{conditioning.conditioning_payload_sha256}"
  puts "image=768x768 target_tokens=#{target_tokens} text_tokens=#{encoder_tokens} " \
       "total_tokens=#{layout.token_count} layers=#{model.layers.size} scheduler_steps=#{steps} " \
       "seed=#{conditioning.seed} timestep0=#{timestep0} timestep1=#{timestep1} forward_calls=3"
  puts "routes=step0_cached_resident_input_build; step1_cached_resident_input_hit " \
       "vs step1_uncached_forward_resident_input " \
       "shared_inputs=gguf_encoder_conditioning_masks_shapes " \
       "step1_inputs=FlowMatch(initial_latents,step0_output),same_latent_and_timestep_for_hit_and_reference"

  # Build and later forwards use the same model, stack, Metal projection
  # backend, conditioning, layout, and masks. Build at the real initial
  # latent/timestep, then advance one FlowMatch Euler step. Hit and reference
  # use the exact same step-1 latent and timestep. The final call uses this
  # stack's uncached full resident-input route, which invalidates the cache
  # before processing all rows. Build output is intentionally not compared to
  # the step-1 reference: that would compare different trajectory inputs.
  build = ML::GGUF::QwenImage21TransformerCPU.forward(
    conditioning.initial_target_latents, conditioning.encoder_hidden_states,
    timestep0, shapes, img_mask, weights, config,
    encoder_hidden_states_mask: conditioning.encoder_hidden_states_mask,
    backend: projection_backend, layer_stack_backend: stack,
  )
  build_stats = stack.last_stats.not_nil!
  abort "expected exactly one prefix build and no hit after build" unless stack.prefix_cache_builds == 1 && stack.prefix_cache_hits == 0
  abort "cache build mutated the bundle inputs" unless conditioning.initial_target_latents == fixed_latents &&
                                                       conditioning.encoder_hidden_states == fixed_encoder &&
                                                       conditioning.encoder_hidden_states_mask == fixed_encoder_mask &&
                                                       img_mask == fixed_img_mask && shapes == fixed_shapes
  step1_latents = schedule.step(
    conditioning.initial_target_latents, build.output.last(target_count), 0,
  )
  abort "FlowMatch step 0 did not change the target latents" if step1_latents == conditioning.initial_target_latents
  fixed_step1_latents = step1_latents.dup

  hit = ML::GGUF::QwenImage21TransformerCPU.forward(
    step1_latents, conditioning.encoder_hidden_states,
    timestep1, shapes, img_mask, weights, config,
    encoder_hidden_states_mask: conditioning.encoder_hidden_states_mask,
    backend: projection_backend, layer_stack_backend: stack,
  )
  hit_stats = stack.last_stats.not_nil!
  abort "expected one prefix build and one cache hit" unless stack.prefix_cache_builds == 1 && stack.prefix_cache_hits == 1

  uncached_adapter = QwenImage21CacheParityUncachedStack.new(stack)
  uncached = ML::GGUF::QwenImage21TransformerCPU.forward(
    step1_latents, conditioning.encoder_hidden_states,
    timestep1, shapes, img_mask, weights, config,
    encoder_hidden_states_mask: conditioning.encoder_hidden_states_mask,
    backend: projection_backend, layer_stack_backend: uncached_adapter,
  )
  uncached_stats = stack.last_stats.not_nil!
  abort "uncached route unexpectedly built or hit a prefix" unless stack.prefix_cache_builds == 1 && stack.prefix_cache_hits == 1
  abort "a transformer route mutated the held-fixed conditioning inputs" unless step1_latents == fixed_step1_latents &&
                                                                                conditioning.initial_target_latents == fixed_latents &&
                                                                                conditioning.encoder_hidden_states == fixed_encoder &&
                                                                                conditioning.encoder_hidden_states_mask == fixed_encoder_mask &&
                                                                                img_mask == fixed_img_mask && shapes == fixed_shapes

  [build, hit, uncached].each do |result|
    abort "expected full transformer output" unless result.output.size == layout.token_count * config.output_dim
  end
  [build_stats, hit_stats, uncached_stats].each do |stats|
    abort "expected one Metal command buffer per forward" unless stats.command_buffers == 1
    abort "unexpected intermediate Metal readback" unless stats.intermediate_readbacks == 0
    abort "expected one final Metal readback" unless stats.final_readbacks == 1
    abort "unexpected image projection row count" unless stats.image_projection_rows == target_tokens
  end
  abort "cache build should process the full sequence" unless build_stats.active_tokens == layout.token_count
  abort "cache hit should process only target rows" unless hit_stats.active_tokens == target_tokens
  abort "uncached route should process the full sequence" unless uncached_stats.active_tokens == layout.token_count

  uncached_target = uncached.output[target_offset, target_count]
  hit_target = hit.output[target_offset, target_count]
  hit_target_metrics = output_metrics(uncached_target, hit_target)
  hit_full = output_metrics(uncached.output, hit.output)

  batch_threshold = ENV["QWEN35_GEMM_BATCH_THRESHOLD"]?.try(&.to_i?) || 8
  same_batch_route = (target_tokens > batch_threshold) == (layout.token_count > batch_threshold)
  max_abs_limit = same_batch_route ? 1.0e-4 : 5.0e-3
  puts "route_check=active_batches cached_target=#{target_tokens} uncached_full=#{layout.token_count} " \
       "gemm_threshold=#{batch_threshold} same_projection_route=#{same_batch_route} " \
       "max_abs_limit=#{max_abs_limit} cosine_limit=0.99999"
  report_metrics("cache_hit_vs_uncached", hit_full, hit_target_metrics)
  accepted = metric_passes?(hit_full, max_abs_limit) &&
             metric_passes?(hit_target_metrics, max_abs_limit)
  abort "cache hit parity failed for full output and/or target suffix" unless accepted
  puts "parity=PASS full_output_and_target_suffix_within_numeric_bounds image_quality=not_measured"
ensure
  stack.close
  model.close
end
