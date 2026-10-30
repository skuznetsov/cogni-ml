require "json"
require "../src/ml/video/pixelumm_generation"
require "./support/hy_nomas_probe_instrumentation"
require "./support/pixelumm_global_assembly_probe"

alias PixelUMMConfig = ML::Video::PixelUMMGeneration::Config
alias PixelUMMAssembly = ML::Video::PixelUMMGeneration::AssemblyPlan

default_config = PixelUMMConfig.new
global_config = PixelUMMConfig.new(total_latent_frames: 60)
unless default_config.window_starts == [0, 5, 10, 15, 20] &&
       default_config.pixel_frames == 120 &&
       default_config.pixel_frame_coverage_counts.all? { |count| count > 0 && count <= 2 } &&
       global_config.window_starts == (0..50).step(5).to_a &&
       global_config.window_count == 11 && global_config.window_starts.includes?(25) &&
       global_config.pixel_frame_coverage_counts.all? { |count| count == 1 || count == 2 }
  raise "default T30 or global T60 decoder geometry failed its CPU preflight"
end

global = PixelUMMAssembly.new(global_config)
first = PixelUMMAssembly.new(default_config)
second = PixelUMMAssembly.new(default_config)
window = ML::MetalBuffer.new(global_config.window_output_elements * 2)
sample_count = PixelUMMGlobalAssemblyProbe.sample_count(global_config.pixel_frames)
samples = ML::MetalBuffer.new(sample_count.to_i64 * 2 * sizeof(Float32))

source = {{ read_file("#{__DIR__}/support/pixelumm_global_assembly_probe.metal") }}
fill = ML::Metal::ComputePipeline.new("pga_fill_window", source)
gather = ML::Metal::ComputePipeline.new("pga_gather_samples", source)
chain = ML::Metal::MPSCommandChain.new
wait_attempted = false

begin
  allocations_before = ML::MetalBuffer.hy_text_probe_allocations
  readbacks_before = ML::MetalBuffer.hy_text_probe_readbacks

  global_config.window_starts.each do |start|
    chain.encoder.set_pipeline(fill).set_buffer(window, 0, ML::Metal::BufferAccess::Write)
      .set_value(start.to_u32, 1)
      .set_value(global_config.window_output_elements.to_u32, 2)
      .dispatch_1d(global_config.window_output_elements.to_i32)
    global.encode_merge(chain, window_bf16: window, start_latent_frame: start)
  end

  default_config.window_starts.each do |start|
    chain.encoder.set_pipeline(fill).set_buffer(window, 0, ML::Metal::BufferAccess::Write)
      .set_value(start.to_u32, 1)
      .set_value(default_config.window_output_elements.to_u32, 2)
      .dispatch_1d(default_config.window_output_elements.to_i32)
    first.encode_merge(chain, window_bf16: window, start_latent_frame: start)
  end

  default_config.window_starts.each do |start|
    chain.encoder.set_pipeline(fill).set_buffer(window, 0, ML::Metal::BufferAccess::Write)
      .set_value((start + 30).to_u32, 1)
      .set_value(default_config.window_output_elements.to_u32, 2)
      .dispatch_1d(default_config.window_output_elements.to_i32)
    second.encode_merge(chain, window_bf16: window, start_latent_frame: start)
  end

  global.encode_finalize(chain)
  first.encode_finalize(chain)
  second.encode_finalize(chain)
  chain.encoder.set_pipeline(gather)
    .set_buffer(global.output_f32, 0)
    .set_buffer(first.output_f32, 1)
    .set_buffer(second.output_f32, 2)
    .set_buffer(samples, 3, ML::Metal::BufferAccess::Write)
    .set_value(sample_count.to_u32, 4)
    .dispatch_1d(sample_count)

  encode_allocations = ML::MetalBuffer.hy_text_probe_allocations - allocations_before
  encode_readbacks = ML::MetalBuffer.hy_text_probe_readbacks - readbacks_before
  preterminal_waits = chain.wait_count
  unless encode_allocations == 0 && encode_readbacks == 0 && preterminal_waits == 0
    raise "global assembly encode allocated a named buffer, read back, or waited before terminal"
  end

  wait_attempted = true
  chain.commit_and_wait
  unless chain.wait_count == 1 && chain.completed?
    raise "global assembly probe requires exactly one completed terminal wait"
  end

  expected = PixelUMMGlobalAssemblyProbe.expected_samples
  actual = samples.contents.as(Float32*)
  max_error = 0.0_f32
  first_seam_max_error = 0.0_f32
  first_seam_max_delta = 0.0_f32
  sample_index = 0
  240.times do |frame|
    PixelUMMGlobalAssemblyProbe::CHANNELS.times do
      PixelUMMGlobalAssemblyProbe::SPATIAL_POSITIONS.each do
        global_actual = actual[sample_index * 2]
        independent_actual = actual[sample_index * 2 + 1]
        global_expected = expected[sample_index * 2]
        independent_expected = expected[sample_index * 2 + 1]
        raise "non-finite sampled assembly result" unless global_actual.finite? && independent_actual.finite?

        global_error = (global_actual - global_expected).abs
        independent_error = (independent_actual - independent_expected).abs
        max_error = {max_error, global_error, independent_error}.max
        if 120 <= frame < 140
          first_seam_max_error = {first_seam_max_error, global_error, independent_error}.max
          first_seam_max_delta = {first_seam_max_delta, (global_actual - independent_actual).abs}.max
        end
        sample_index += 1
      end
    end
  end

  assembly_bytes = global.named_bytes + first.named_bytes + second.named_bytes
  window_bytes = global_config.window_output_elements * 2
  sample_bytes = sample_count.to_i64 * 2 * sizeof(Float32)
  named_bytes = assembly_bytes + window_bytes + sample_bytes
  result = {
    schema:                                     "cogni_ml.pixelumm_global_assembly_probe/v1",
    scope:                                      "synthetic fixture math only; no neural inference",
    default_t30_window_starts:                  default_config.window_starts,
    global_t60_window_starts:                   global_config.window_starts,
    crossing_window_start_latent:               25,
    crossing_window_pixel_frames:               [100, 139],
    first_seam_pixel_frames:                    [120, 139],
    sample_count:                               sample_count,
    max_sample_abs_error:                       max_error,
    first_seam_max_abs_error:                   first_seam_max_error,
    first_seam_global_vs_independent_max_delta: first_seam_max_delta,
    encode_named_buffer_allocations:            encode_allocations,
    encode_readbacks:                           encode_readbacks,
    preterminal_waits:                          preterminal_waits,
    terminal_waits:                             chain.wait_count,
    gpu_seconds:                                chain.gpu_elapsed_seconds,
    named_allocation_bytes:                     {
      assembly_plans:       assembly_bytes,
      reusable_bf16_window: window_bytes,
      sample_pairs:         sample_bytes,
      total:                named_bytes,
      scope:                "named MetalBuffer payload only; excludes driver/MPS pipeline and command metadata",
    },
  }
  puts result.to_json
  unless max_error <= 2e-5_f32 && first_seam_max_error <= 2e-5_f32 &&
         first_seam_max_delta > 2e-5_f32 && encode_allocations == 0 &&
         encode_readbacks == 0 && preterminal_waits == 0 && chain.wait_count == 1
    raise "global T60 assembly failed the pinned CPU oracle or independent-concatenation falsifier"
  end
ensure
  chain.commit_and_wait unless wait_attempted || chain.completed?
  global.release_after_terminal_wait!
  first.release_after_terminal_wait!
  second.release_after_terminal_wait!
  window.release
  samples.release
  chain.release
end
