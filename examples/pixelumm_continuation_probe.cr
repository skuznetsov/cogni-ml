require "json"
require "../src/ml/video/pixelumm_continuation"
require "../examples/support/hy_nomas_probe_instrumentation"

alias PixelUMMContinuation = ML::Video::PixelUMMContinuation
alias PixelUMMGeneration = ML::Video::PixelUMMGeneration

def continuation_probe_values(elements : Int64, seed : Int32) : Array(Float32)
  raise ArgumentError.new("fixture is larger than native array indexing") unless elements > 0 && elements <= Int32::MAX
  Array(Float32).new(elements.to_i32) do |index|
    # Values stay exactly representable and identify both the buffer and its
    # flat source offset throughout this small geometry.
    (index.to_i64 * 4 + seed).to_f32
  end
end

def continuation_probe_display(elements : Int64) : Array(Float32)
  raise ArgumentError.new("display fixture is larger than native array indexing") unless elements > 0 && elements <= Int32::MAX
  Array(Float32).new(elements.to_i32) { |index| -(index + 1).to_f32 }
end

def continuation_probe_close?(actual : Float32, expected : Float32) : Bool
  actual.finite? && expected.finite? && (actual - expected).abs <= 2e-5_f32 + 2e-5_f32 * expected.abs
end

config = PixelUMMContinuation::Config.new(
  previous_latent_frames: 20, current_latent_frames: 10,
  latent_height: 1, latent_width: 16)
global_config = PixelUMMGeneration::Config.new(
  total_latent_frames: 30, latent_height: 1, latent_width: 16)
local_config = config.decoder_config
fresh_config = config.current_config

previous_latent_values = continuation_probe_values(config.previous_latent_elements, 0)
previous_noise_values = continuation_probe_values(config.previous_noise_elements, 1)
current_latent_values = continuation_probe_values(config.current_latent_elements, 2)
current_noise_values = continuation_probe_values(config.current_noise_elements, 3)

previous_latent = ML::MetalBuffer.from_array(previous_latent_values)
previous_noise = ML::MetalBuffer.from_array(previous_noise_values)
current_latent = ML::MetalBuffer.from_array(current_latent_values)
current_noise = ML::MetalBuffer.from_array(current_noise_values)

continuation = nil.as(PixelUMMContinuation::Plan?)
global_assembly = nil.as(PixelUMMGeneration::AssemblyPlan?)
local_assembly = nil.as(PixelUMMGeneration::AssemblyPlan?)
fresh_assembly = nil.as(PixelUMMGeneration::AssemblyPlan?)
window = nil.as(ML::MetalBuffer?)
chain = nil.as(ML::Metal::MPSCommandChain?)
wait_attempted = false

begin
  plan = PixelUMMContinuation::Plan.new(config,
    previous_latent: previous_latent, previous_noise: previous_noise,
    current_latent: current_latent, current_noise: current_noise)
  continuation = plan
  global = PixelUMMGeneration::AssemblyPlan.new(global_config)
  global_assembly = global
  local = PixelUMMGeneration::AssemblyPlan.new(local_config)
  local_assembly = local
  fresh = PixelUMMGeneration::AssemblyPlan.new(fresh_config)
  fresh_assembly = fresh
  shared_window = ML::MetalBuffer.new(local_config.window_output_elements * 2)
  window = shared_window
  fixture_source = {{ read_file("#{__DIR__}/support/pixelumm_continuation_probe.metal") }}
  fill_window = ML::Metal::ComputePipeline.new("pcp_fill_bf16_window", fixture_source)
  active_chain = ML::Metal::MPSCommandChain.new
  chain = active_chain

  allocations_before = ML::MetalBuffer.hy_text_probe_allocations
  readbacks_before = ML::MetalBuffer.hy_text_probe_readbacks

  plan.encode_inputs(active_chain)
  global_config.window_starts.each do |start|
    active_chain.encoder.set_pipeline(fill_window)
      .set_buffer(shared_window, 0, ML::Metal::BufferAccess::Write)
      .set_value(start.to_u32, 1)
      .set_value(global_config.window_output_elements.to_u32, 2)
      .set_value(global_config.pixel_height.to_u32, 3)
      .set_value(global_config.pixel_width.to_u32, 4)
      .dispatch_1d(global_config.window_output_elements.to_i32)
    global.encode_merge(active_chain, window_bf16: shared_window, start_latent_frame: start)
  end

  local_config.window_starts.each do |start|
    active_chain.encoder.set_pipeline(fill_window)
      .set_buffer(shared_window, 0, ML::Metal::BufferAccess::Write)
      .set_value((15 + start).to_u32, 1)
      .set_value(local_config.window_output_elements.to_u32, 2)
      .set_value(local_config.pixel_height.to_u32, 3)
      .set_value(local_config.pixel_width.to_u32, 4)
      .dispatch_1d(local_config.window_output_elements.to_i32)
    local.encode_merge(active_chain, window_bf16: shared_window, start_latent_frame: start)
  end

  active_chain.encoder.set_pipeline(fill_window)
    .set_buffer(shared_window, 0, ML::Metal::BufferAccess::Write)
    .set_value(20_u32, 1)
    .set_value(fresh_config.window_output_elements.to_u32, 2)
    .set_value(fresh_config.pixel_height.to_u32, 3)
    .set_value(fresh_config.pixel_width.to_u32, 4)
    .dispatch_1d(fresh_config.window_output_elements.to_i32)
  fresh.encode_merge(active_chain, window_bf16: shared_window, start_latent_frame: 0)

  global.encode_finalize(active_chain)
  local.encode_finalize(active_chain)
  fresh.encode_finalize(active_chain)
  plan.encode_output(active_chain, source: local.output_f32)

  encode_allocations = ML::MetalBuffer.hy_text_probe_allocations - allocations_before
  encode_readbacks = ML::MetalBuffer.hy_text_probe_readbacks - readbacks_before
  preterminal_waits = active_chain.wait_count
  unless encode_allocations == 0 && encode_readbacks == 0 && preterminal_waits == 0
    raise "continuation encode allocated named buffers, read back, or waited before terminal completion"
  end

  wait_attempted = true
  active_chain.commit_and_wait
  unless active_chain.wait_count == 1 && active_chain.completed?
    raise "continuation probe requires exactly one completed terminal wait"
  end

  joined_latent = plan.latent_f32.read(config.latent_elements.to_i32)
  config.latent_elements.to_i32.times do |index|
    source, offset = config.latent_join_source(index)
    expected = source == :previous ? previous_latent_values[offset.to_i] : current_latent_values[offset.to_i]
    unless joined_latent[index] == expected
      raise "joined latent mismatch at flat NCTHW element #{index} from #{source}[#{offset}]"
    end
  end

  joined_noise = plan.noise_f32.read(config.noise_elements.to_i32)
  config.noise_elements.to_i32.times do |index|
    source, offset = config.noise_join_source(index)
    expected = source == :previous ? previous_noise_values[offset.to_i] : current_noise_values[offset.to_i]
    unless joined_noise[index] == expected
      raise "joined noise mismatch at flat NCTHW element #{index} from #{source}[#{offset}]"
    end
  end

  cropped = plan.output_f32.read(config.output_elements.to_i32)
  local_display = local.output_f32.read(local_config.output_elements.to_i32)
  global_display = global.output_f32.read(global_config.output_elements.to_i32)
  fresh_display = fresh.output_f32.read(fresh_config.output_elements.to_i32)
  spatial = config.current_config.pixel_height.to_i64 * config.current_config.pixel_width
  global_frames = global_config.pixel_frames.to_i64
  global_start_frame = config.previous_latent_frames.to_i64 * PixelUMMContinuation::TUBELET_T
  max_crop_oracle_error = 0.0_f32
  max_global_suffix_error = 0.0_f32
  max_no_context_delta = 0.0_f32
  config.output_elements.to_i32.times do |index|
    local_source_index = config.output_source_index(index)
    local_expected = local_display[local_source_index.to_i]
    crop_error = (cropped[index] - local_expected).abs
    max_crop_oracle_error = {max_crop_oracle_error, crop_error}.max
    unless continuation_probe_close?(cropped[index], local_expected)
      raise "display crop mismatch at NCTHW element #{index}"
    end

    channel_stride = fresh_config.pixel_frames.to_i64 * spatial
    channel = index.to_i64 // channel_stride
    within_channel = index.to_i64 % channel_stride
    frame = within_channel // spatial
    cell = within_channel % spatial
    global_index = channel * global_frames * spatial + (global_start_frame + frame) * spatial + cell
    global_expected = global_display[global_index.to_i]
    global_error = (cropped[index] - global_expected).abs
    max_global_suffix_error = {max_global_suffix_error, global_error}.max
    unless continuation_probe_close?(cropped[index], global_expected)
      raise "continuation suffix differs from full-timeline BF16 assembly at element #{index}"
    end

    no_context_delta = (cropped[index] - fresh_display[index]).abs
    max_no_context_delta = {max_no_context_delta, no_context_delta}.max
  end
  unless max_no_context_delta > 2e-5_f32
    raise "fresh-current-only positive control did not expose the lost context seam"
  end

  result = {
    schema:   "cogni_ml.pixelumm_continuation_probe/v1",
    scope:    "synthetic GPU join/crop and BF16 blender fixture only; no neural inference",
    geometry: {
      previous_latent_frames: config.previous_latent_frames,
      current_latent_frames:  config.current_latent_frames,
      joined_latent_shape:    config.latent_shape,
      joined_noise_shape:     config.noise_shape,
      output_shape:           config.output_shape,
    },
    latent_elements_verified:       joined_latent.size,
    noise_elements_verified:        joined_noise.size,
    crop_elements_verified:         cropped.size,
    max_crop_oracle_abs_error:      max_crop_oracle_error,
    max_full_timeline_suffix_error: max_global_suffix_error,
    no_context_positive_delta:      max_no_context_delta,
    encode_named_allocations:       encode_allocations,
    encode_readbacks:               encode_readbacks,
    preterminal_waits:              preterminal_waits,
    terminal_waits:                 active_chain.wait_count,
    postterminal_readbacks:         ML::MetalBuffer.hy_text_probe_readbacks - readbacks_before,
    owned_named_bytes:              plan.named_bytes,
    passed:                         encode_allocations == 0 && encode_readbacks == 0 &&
            preterminal_waits == 0 && active_chain.wait_count == 1 &&
            max_crop_oracle_error <= 2e-5_f32 &&
            max_global_suffix_error <= 2e-5_f32 &&
            max_no_context_delta > 2e-5_f32,
  }
  puts result.to_json
  raise "PixelUMM continuation GPU fixture failed" unless result[:passed]
ensure
  if active_chain = chain
    active_chain.commit_and_wait unless wait_attempted || active_chain.completed?
    if active_chain.completed?
      continuation.try(&.release_after_terminal_wait!)
      global_assembly.try(&.release_after_terminal_wait!)
      local_assembly.try(&.release_after_terminal_wait!)
      fresh_assembly.try(&.release_after_terminal_wait!)
    end
    active_chain.release
  else
    continuation.try(&.release_after_terminal_wait!)
    global_assembly.try(&.release_after_terminal_wait!)
    local_assembly.try(&.release_after_terminal_wait!)
    fresh_assembly.try(&.release_after_terminal_wait!)
  end
  window.try(&.release)
  previous_latent.release
  previous_noise.release
  current_latent.release
  current_noise.release
end
