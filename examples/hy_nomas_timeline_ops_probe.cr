require "json"
require "../src/ml/video/hy_nomas_timeline_ops"
require "./support/hy_nomas_probe_instrumentation"

module HyNomasTimelineOpsProbe
  TOTAL_FRAMES   =        9
  SEGMENT_FRAMES =        3
  CHANNELS       =        2
  HEIGHT         =        1
  WIDTH          =        2
  ROPE_HEAD_DIM  =        2
  ATOL           = 2e-5_f32
  RTOL           = 2e-5_f32

  PROBE_SOURCE = <<-METAL
    #include <metal_stdlib>
    using namespace metal;

    kernel void htop_fixture_denoise(
        device float *segment [[buffer(0)]],
        uint index [[thread_position_in_grid]]) {
      segment[index] = 0.125f - segment[index];
    }

    kernel void htop_fixture_snapshot(
        device const float *segment [[buffer(0)]],
        device float *snapshot [[buffer(1)]],
        constant uint &count [[buffer(2)]],
        uint index [[thread_position_in_grid]]) {
      if (index < count)
        snapshot[index] = segment[index];
    }

    // This marker stands in only for the schedule's downstream consumer. It
    // does not execute the upstream native update_cache attention operation.
    kernel void htop_fixture_cache_consumer(
        device const float *segment [[buffer(0)]],
        device float *sink [[buffer(1)]],
        uint index [[thread_position_in_grid]]) {
      if (index == 0u)
        sink[0] = segment[0];
    }
    METAL

  def self.bf16_bits(value : Float32) : UInt16
    bits = value.unsafe_as(UInt32)
    bias = 0x7fff_u32 + ((bits >> 16) & 1_u32)
    ((bits + bias) >> 16).to_u16
  end

  def self.bf16_value(value : Float32) : Float32
    (bf16_bits(value).to_u32 << 16).unsafe_as(Float32)
  end

  def self.initial_latent(geometry : ML::Video::HyNomasTimelineOps::Geometry) : Array(Float32)
    Array(Float32).new(geometry.timeline_elements) do |index|
      value = ((index * 17 + 3) % 41 - 20).to_f32 / 1024.0_f32
      raise "fixture latent must already be BF16-exact" unless bf16_value(value) == value
      value
    end
  end

  def self.cosine_bank(geometry : ML::Video::HyNomasTimelineOps::Geometry) : Array(Float32)
    Array(Float32).new(geometry.rope_bank_elements) do |index|
      (index - 18).to_f32 / 16.0_f32
    end
  end

  def self.sine_bank(geometry : ML::Video::HyNomasTimelineOps::Geometry) : Array(Float32)
    Array(Float32).new(geometry.rope_bank_elements) do |index|
      (index + 7).to_f32 / 32.0_f32
    end
  end

  def self.epsilon_words(geometry : ML::Video::HyNomasTimelineOps::Geometry) : Array(UInt16)
    Array(UInt16).new(geometry.timeline_elements) do |index|
      value = ((index * 7 + 1) % 19 - 9).to_f32 / 16.0_f32
      bf16_bits(value)
    end
  end

  def self.index_in_timeline(geometry : ML::Video::HyNomasTimelineOps::Geometry,
                             segment_index : Int32, segment_index_in_buffer : Int32) : Int32
    spatial = geometry.latent_height * geometry.latent_width
    frame_area = geometry.segment_latent_frames * spatial
    channel = segment_index_in_buffer // frame_area
    local_frame = (segment_index_in_buffer // spatial) % geometry.segment_latent_frames
    spatial_index = segment_index_in_buffer % spatial
    global_frame = geometry.segment_start(segment_index) + local_frame
    (channel * geometry.total_latent_frames + global_frame) * spatial + spatial_index
  end

  def self.expected_scattered_timeline(
    geometry : ML::Video::HyNomasTimelineOps::Geometry,
    initial : Array(Float32),
  ) : Array(Float32)
    expected = initial.dup
    geometry.segment_elements.times do |index|
      timeline_index = index_in_timeline(geometry, 0, index)
      expected[timeline_index] = 0.125_f32 - initial[timeline_index]
    end
    expected
  end

  def self.expected_segment_gather(
    geometry : ML::Video::HyNomasTimelineOps::Geometry,
    initial : Array(Float32), segment_index : Int32,
  ) : Array(Float32)
    Array(Float32).new(geometry.segment_elements) do |index|
      initial[index_in_timeline(geometry, segment_index, index)]
    end
  end

  def self.expected_cache_noise(
    geometry : ML::Video::HyNomasTimelineOps::Geometry,
    clean_latent : Array(Float32), epsilon : Array(UInt16),
  ) : Array(Float32)
    sigma = geometry.cache_sigma
    Array(Float32).new(geometry.segment_elements) do |index|
      latent_bf16 = bf16_value(clean_latent[index])
      epsilon_bf16 = (epsilon[index].to_u32 << 16).unsafe_as(Float32)
      clean_component = (1.0_f32 - sigma) * latent_bf16
      noise_component = sigma * epsilon_bf16
      bf16_value(clean_component + noise_component)
    end
  end

  def self.expected_prefix(
    geometry : ML::Video::HyNomasTimelineOps::PrefixGeometry,
    source : Array(Float32),
  ) : Array(Float32)
    spatial = geometry.height * geometry.width
    prefix_frame_area = geometry.prefix_frames * spatial
    Array(Float32).new(geometry.prefix_elements) do |index|
      channel = index // prefix_frame_area
      frame = (index // spatial) % geometry.prefix_frames
      spatial_index = index % spatial
      source_index = (channel * geometry.total_frames + frame) * spatial + spatial_index
      source[source_index]
    end
  end

  def self.pixel_noise_fixture(geometry : ML::Video::HyNomasTimelineOps::PrefixGeometry) : Array(Float32)
    Array(Float32).new(geometry.source_elements) do |index|
      (index - geometry.source_elements // 2).to_f32 / 128.0_f32
    end
  end

  def self.max_error(actual : Array(Float32), expected : Array(Float32), label : String) : Float32
    raise "#{label}: length mismatch #{actual.size} != #{expected.size}" unless actual.size == expected.size
    max_abs = 0.0_f32
    actual.each_with_index do |value, index|
      target = expected[index]
      raise "#{label}[#{index}]: non-finite value" unless value.finite? && target.finite?
      error = (value - target).abs
      max_abs = error if error > max_abs
      tolerance = ATOL + RTOL * target.abs
      raise "#{label}[#{index}]: #{value} != #{target} (error #{error}, limit #{tolerance})" if error > tolerance
    end
    max_abs
  end

  def self.run! : Nil
    geometry = ML::Video::HyNomasTimelineOps::Geometry.new(
      total_latent_frames: TOTAL_FRAMES,
      segment_latent_frames: SEGMENT_FRAMES,
      channels: CHANNELS,
      latent_height: HEIGHT,
      latent_width: WIDTH,
      rope_head_dim: ROPE_HEAD_DIM)
    latent_prefix_geometry = ML::Video::HyNomasTimelineOps::PrefixGeometry.new(
      channels: CHANNELS,
      total_frames: TOTAL_FRAMES,
      prefix_frames: 6,
      height: HEIGHT,
      width: WIDTH)
    pixel_noise_geometry = ML::Video::HyNomasTimelineOps::PrefixGeometry.new(
      channels: 3,
      total_frames: TOTAL_FRAMES,
      prefix_frames: 6,
      height: 2,
      width: WIDTH)
    initial = initial_latent(geometry)
    cosine_values = cosine_bank(geometry)
    sine_values = sine_bank(geometry)
    epsilon_values = epsilon_words(geometry)
    pixel_noise_values = pixel_noise_fixture(pixel_noise_geometry)

    buffers = [] of ML::MetalBuffer
    buffers << ML::MetalBuffer.from_array(initial)
    buffers << ML::MetalBuffer.new(geometry.segment_bytes)
    buffers << ML::MetalBuffer.from_array(cosine_values)
    buffers << ML::MetalBuffer.from_array(sine_values)
    buffers << ML::MetalBuffer.new(geometry.rope_elements_per_segment.to_i64 * sizeof(Float32))
    buffers << ML::MetalBuffer.new(geometry.rope_elements_per_segment.to_i64 * sizeof(Float32))
    buffers << ML::MetalBuffer.new(geometry.cache_noise_bank_bytes)
    buffers << ML::MetalBuffer.new(sizeof(Float32).to_i64)
    buffers << ML::MetalBuffer.new(geometry.segment_bytes) # cache-noise capture
    buffers << ML::MetalBuffer.new(sizeof(Float32).to_i64) # fixture consumer sink
    buffers << ML::MetalBuffer.new(latent_prefix_geometry.prefix_bytes)
    buffers << ML::MetalBuffer.from_array(pixel_noise_values)
    buffers << ML::MetalBuffer.new(pixel_noise_geometry.prefix_bytes)

    timeline, segment, cosine_source, sine_source, cosine, sine, epsilon, timestep, noisy_capture, consumer_sink, latent_prefix, pixel_noise_source, pixel_noise_prefix = buffers
    epsilon.write_bytes(epsilon_values.to_unsafe.as(Pointer(UInt8)),
      epsilon_values.size * sizeof(UInt16))

    plan = ML::Video::HyNomasTimelineOps::Plan.new(geometry,
      timeline: timeline,
      segment_latent: segment,
      cosine_bank: cosine_source,
      sine_bank: sine_source,
      cosine: cosine,
      sine: sine,
      cache_eps_bf16: epsilon,
      cache_timestep: timestep,
      run_segments: 3)
    latent_prefix_plan = ML::Video::HyNomasTimelineOps::PrefixPlan.new(
      latent_prefix_geometry, source: timeline, destination: latent_prefix)
    pixel_noise_prefix_plan = ML::Video::HyNomasTimelineOps::PrefixPlan.new(
      pixel_noise_geometry, source: pixel_noise_source, destination: pixel_noise_prefix)
    transform = ML::Metal::ComputePipeline.new("htop_fixture_denoise", PROBE_SOURCE)
    snapshot = ML::Metal::ComputePipeline.new("htop_fixture_snapshot", PROBE_SOURCE)
    consumer = ML::Metal::ComputePipeline.new("htop_fixture_cache_consumer", PROBE_SOURCE)
    chain = ML::Metal::MPSCommandChain.new
    plan_released = false
    latent_prefix_released = false
    pixel_noise_prefix_released = false

    begin
      allocations_before = ML::MetalBuffer.hy_text_probe_allocations
      readbacks_before = ML::MetalBuffer.hy_text_probe_readbacks

      # Reuse the temporal gather kernel for latent and decoder-noise prefixes.
      latent_prefix_plan.encode!(chain)
      pixel_noise_prefix_plan.encode!(chain)

      # Segment 0: gather -> deterministic fixture denoise -> clean scatter ->
      # upstream-shaped BF16 cache-noise write -> downstream consumer marker.
      plan.prepare_segment!(chain, 0)
      chain.encoder.set_pipeline(transform).set_buffer(segment, 0, ML::Metal::BufferAccess::ReadWrite)
        .dispatch_1d(geometry.segment_elements)
      plan.mark_denoise_complete!(chain, 0)
      plan.scatter_generated!(chain, 0)
      plan.write_cache_noise!(chain, 0)
      chain.encoder.set_pipeline(snapshot)
        .set_buffer(segment, 0)
        .set_buffer(noisy_capture, 1, ML::Metal::BufferAccess::Write)
        .set_value(geometry.segment_elements.to_u32, 2)
        .dispatch_1d(geometry.segment_elements)
      chain.encoder.set_pipeline(consumer)
        .set_buffer(segment, 0)
        .set_buffer(consumer_sink, 1, ML::Metal::BufferAccess::Write)
        .dispatch_1d(1)
      plan.record_cache_forward_enqueued!(chain, 0)
      plan.advance_segment!(chain, 0)

      # Segment 1 exercises nonzero global latent and RoPE offsets. This probe
      # deliberately stops after prepare to exercise partial-plan release.
      plan.prepare_segment!(chain, 1)

      encode_allocations = ML::MetalBuffer.hy_text_probe_allocations - allocations_before
      encode_readbacks = ML::MetalBuffer.hy_text_probe_readbacks - readbacks_before
      preterminal_waits = chain.wait_count
      chain.commit_and_wait

      expected_scatter = expected_scattered_timeline(geometry, initial)
      expected_gather = expected_segment_gather(geometry, initial, 1)
      expected_noisy = expected_cache_noise(geometry,
        expected_segment_gather(geometry, expected_scatter, 0), epsilon_values)
      expected_cosine = cosine_values[geometry.rope_offset(1), geometry.rope_elements_per_segment]
      expected_sine = sine_values[geometry.rope_offset(1), geometry.rope_elements_per_segment]

      scatter_error = max_error(timeline.read(geometry.timeline_elements), expected_scatter, "clean scatter")
      gather_error = max_error(segment.read(geometry.segment_elements), expected_gather, "segment-1 gather")
      cosine_error = max_error(cosine.read(geometry.rope_elements_per_segment), expected_cosine, "global cosine slice")
      sine_error = max_error(sine.read(geometry.rope_elements_per_segment), expected_sine, "global sine slice")
      noise_error = max_error(noisy_capture.read(geometry.segment_elements), expected_noisy, "BF16 cache-noise formula")
      timestep_value = timestep.read(1)[0]
      timestep_error = max_error([timestep_value], [geometry.cache_timestep], "cache timestep")
      consumer_value = consumer_sink.read(1)[0]
      consumer_error = max_error([consumer_value], [expected_noisy[0]], "fixture cache consumer order")
      latent_prefix_error = max_error(latent_prefix.read(latent_prefix_geometry.prefix_elements),
        expected_prefix(latent_prefix_geometry, initial), "latent temporal prefix")
      pixel_noise_prefix_error = max_error(pixel_noise_prefix.read(pixel_noise_geometry.prefix_elements),
        expected_prefix(pixel_noise_geometry, pixel_noise_values), "decoder-noise temporal prefix")
      postterminal_readbacks = ML::MetalBuffer.hy_text_probe_readbacks - readbacks_before
      waits = chain.wait_count
      plan.release_after_drain!(chain)
      plan_released = true
      latent_prefix_plan.release_after_drain!(chain)
      latent_prefix_released = true
      pixel_noise_prefix_plan.release_after_drain!(chain)
      pixel_noise_prefix_released = true

      result = {
        schema:          "cogni_ml.hy_nomas_timeline_ops_probe/v1",
        scope:           "synthetic GPU gather/scatter, global RoPE slicing, latent/noise temporal prefixes, and cache-noise formula; no neural model or native update_cache attention",
        segments_tested: [0, 1],
        prefix_ranges:   {latent_frames: [0, 6], decoder_noise_frames: [0, 6]},
        tolerance:       {atol: ATOL, rtol: RTOL},
        max_abs_error:   {
          clean_scatter:          scatter_error,
          segment1_gather:        gather_error,
          global_cosine_slice:    cosine_error,
          global_sine_slice:      sine_error,
          cache_noise_formula:    noise_error,
          cache_timestep:         timestep_error,
          fixture_consumer_order: consumer_error,
          latent_temporal_prefix: latent_prefix_error,
          decoder_noise_prefix:   pixel_noise_prefix_error,
        },
        encode_allocations:                encode_allocations,
        encode_readbacks:                  encode_readbacks,
        preterminal_waits:                 preterminal_waits,
        terminal_waits:                    waits,
        postterminal_readbacks:            postterminal_readbacks,
        partial_plan_released_after_drain: plan_released,
        prefix_plans_released_after_drain: latent_prefix_released && pixel_noise_prefix_released,
        passed:                            encode_allocations == 0 && encode_readbacks == 0 &&
                preterminal_waits == 0 && waits == 1 && postterminal_readbacks == 9 &&
                plan_released && latent_prefix_released && pixel_noise_prefix_released,
      }
      puts result.to_pretty_json
      raise "Hy-Nomas timeline-ops probe contract failed" unless result[:passed]
    ensure
      chain.commit_and_wait unless chain.drained?
      if chain.drained?
        unless plan_released
          plan.release_after_drain!(chain)
          plan_released = true
        end
        unless latent_prefix_released
          latent_prefix_plan.release_after_drain!(chain)
          latent_prefix_released = true
        end
        unless pixel_noise_prefix_released
          pixel_noise_prefix_plan.release_after_drain!(chain)
          pixel_noise_prefix_released = true
        end
        buffers.each(&.release)
        chain.release
      end
    end
  end
end

HyNomasTimelineOpsProbe.run!
