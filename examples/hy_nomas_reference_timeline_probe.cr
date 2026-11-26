require "json"
require "../src/ml/video/hy_nomas_timeline_ops"
require "./support/hy_nomas_probe_instrumentation"

# A bounded native admission test, not a DiT or video quality test. Tensor work
# stays on GPU; expected values and diagnostic reads run after the terminal drain.
module HyNomasReferenceTimelineProbe
  @@in_flight = [] of Proc(Nil)

  def self.bf16(value : Float32) : UInt16
    bits = value.unsafe_as(UInt32)
    ((bits + 0x7fff_u32 + ((bits >> 16) & 1_u32)) >> 16).to_u16
  end

  def self.expand(bits : UInt16) : Float32
    (bits.to_u32 << 16).unsafe_as(Float32)
  end

  def self.check(actual : Array(Float32), expected : Array(Float32), label : String) : Float32
    raise "#{label}: extent mismatch" unless actual.size == expected.size
    maximum = 0.0_f32
    actual.each_with_index do |value, index|
      target = expected[index]
      error = (value - target).abs
      raise "#{label}: non-finite result" unless value.finite? && target.finite?
      raise "#{label}: mismatch at #{index}" if error > 2e-5_f32 + 2e-5_f32 * target.abs
      maximum = error if error > maximum
    end
    maximum
  end

  def self.run! : Nil
    geometry = ML::Video::HyNomasTimelineOps::Geometry.new
    reference = Array(UInt16).new(geometry.segment_elements) { |i| bf16(((i * 13) % 997 - 498).to_f32 / 128_f32) }
    clean = reference.map { |word| expand(word) }
    original = Array(Float32).new(geometry.timeline_elements) { |i| -8_f32 - (i % 31).to_f32 / 32_f32 }
    epsilon = Array(UInt16).new(geometry.timeline_elements) { |i| bf16(((i * 7) % 53 - 26).to_f32 / 16_f32) }
    cosine = Array(Float32).new(geometry.rope_bank_elements) { |i| (i % 29).to_f32 / 32_f32 }
    sine = Array(Float32).new(geometry.rope_bank_elements) { |i| -(i % 23).to_f32 / 32_f32 }
    expected_timeline = original.dup
    spatial = geometry.latent_height * geometry.latent_width
    frame_area = geometry.segment_latent_frames * spatial
    geometry.segment_elements.times do |i|
      channel = i // frame_area
      expected_timeline[channel * geometry.total_latent_frames * spatial + i % frame_area] = clean[i]
    end
    expected_t1 = Array(Float32).new(geometry.segment_elements) do |i|
      channel = i // frame_area
      original[channel * geometry.total_latent_frames * spatial + frame_area + i % frame_area]
    end
    expected_noise = Array(Float32).new(geometry.segment_elements) do |i|
      clean_component = (1_f32 - geometry.cache_sigma) * clean[i]
      noise_component = geometry.cache_sigma * expand(epsilon[i])
      expand(bf16(clean_component + noise_component))
    end
    buffers = [] of ML::MetalBuffer
    plan = nil.as(ML::Video::HyNomasTimelineOps::Plan?)
    chain = nil.as(ML::Metal::MPSCommandChain?)
    released = false
    begin
      timeline = ML::MetalBuffer.from_array(original); buffers << timeline
      segment = ML::MetalBuffer.new(geometry.segment_bytes); buffers << segment
      cos_bank = ML::MetalBuffer.from_array(cosine); buffers << cos_bank
      sin_bank = ML::MetalBuffer.from_array(sine); buffers << sin_bank
      cos = ML::MetalBuffer.new(geometry.rope_elements_per_segment.to_i64 * 4); buffers << cos
      sin = ML::MetalBuffer.new(geometry.rope_elements_per_segment.to_i64 * 4); buffers << sin
      eps = ML::MetalBuffer.new(geometry.cache_noise_bank_bytes); buffers << eps
      eps.write_bytes(epsilon.to_unsafe.as(UInt8*), epsilon.size * 2)
      timestep = ML::MetalBuffer.new(4_i64); buffers << timestep
      ref = ML::MetalBuffer.new(reference.size.to_i64 * 2); buffers << ref
      ref.write_bytes(reference.to_unsafe.as(UInt8*), reference.size * 2)
      clean_capture = ML::MetalBuffer.new(geometry.segment_bytes); buffers << clean_capture
      noise_capture = ML::MetalBuffer.new(geometry.segment_bytes); buffers << noise_capture
      plan = ML::Video::HyNomasTimelineOps::Plan.new(geometry,
        timeline: timeline, segment_latent: segment, cosine_bank: cos_bank,
        sine_bank: sin_bank, cosine: cos, sine: sin, cache_eps_bf16: eps,
        cache_timestep: timestep, run_segments: 2, reference_initialization: true)
      copy = ML::Metal::PipelineCache.get(ML::Video::HyNomasTimelineOps::COPY_KERNEL) do
        ML::Metal::ComputePipeline.new(ML::Video::HyNomasTimelineOps::COPY_KERNEL,
          ML::Video::HyNomasTimelineOps::SOURCE)
      end
      chain = ML::Metal::MPSCommandChain.new
      allocations = ML::MetalBuffer.hy_text_probe_allocations
      readbacks = ML::MetalBuffer.hy_text_probe_readbacks
      plan.admit_reference_latent!(chain, ref)
      plan.prepare_reference_segment!(chain, 0)
      chain.encoder.set_pipeline(copy).set_buffer(segment, 0)
        .set_buffer(clean_capture, 1, ML::Metal::BufferAccess::Write)
        .set_value(0_u32, 2).set_value(geometry.segment_elements.to_u32, 3)
        .dispatch_1d(geometry.segment_elements)
      plan.write_cache_noise!(chain, 0)
      chain.encoder.set_pipeline(copy).set_buffer(segment, 0)
        .set_buffer(noise_capture, 1, ML::Metal::BufferAccess::Write)
        .set_value(0_u32, 2).set_value(geometry.segment_elements.to_u32, 3)
        .dispatch_1d(geometry.segment_elements)
      # No model is executed: this marks only the consumer's schedule boundary.
      plan.record_cache_forward_enqueued!(chain, 0)
      plan.advance_segment!(chain, 0)
      plan.prepare_segment!(chain, 1)
      encode_allocations = ML::MetalBuffer.hy_text_probe_allocations - allocations
      encode_readbacks = ML::MetalBuffer.hy_text_probe_readbacks - readbacks
      preterminal_waits = chain.wait_count
      raise "reference admission allocated, read back, or waited" unless encode_allocations == 0 && encode_readbacks == 0 && preterminal_waits == 0
      chain.commit_and_wait
      errors = {
        timeline:        check(timeline.read(geometry.timeline_elements), expected_timeline, "T0-only timeline write"),
        clean_reference: check(clean_capture.read(geometry.segment_elements), clean, "clean T0 gather"),
        noisy_reference: check(noise_capture.read(geometry.segment_elements), expected_noise, "cache noise"),
        next_segment:    check(segment.read(geometry.segment_elements), expected_t1, "unchanged T1 noise"),
        cosine:          check(cos.read(geometry.rope_elements_per_segment), cosine[geometry.rope_offset(1), geometry.rope_elements_per_segment], "T1 cosine"),
        sine:            check(sin.read(geometry.rope_elements_per_segment), sine[geometry.rope_offset(1), geometry.rope_elements_per_segment], "T1 sine"),
        timestep:        check(timestep.read(1), [geometry.cache_timestep], "cache timestep"),
      }
      unless Bytes.new(ref.contents.as(UInt8*), reference.size * 2) == Bytes.new(reference.to_unsafe.as(UInt8*), reference.size * 2)
        raise "reference source was modified"
      end
      plan.release_after_drain!(chain)
      released = true
      puts({schema: "cogni_ml.hy_nomas_reference_timeline_probe/v1",
            scope: "native BF16 T0 admission and cache-noise preparation; no DiT cache writer or video quality claim",
            shape: [1, 32, 30, 9, 16], atol: 2e-5, rtol: 2e-5,
            max_abs_error: errors, encode_allocations: encode_allocations,
            encode_readbacks: encode_readbacks, preterminal_waits: preterminal_waits,
            terminal_waits: chain.wait_count, passed: chain.wait_count == 1}.to_pretty_json)
      raise "unexpected terminal wait count" unless chain.wait_count == 1
    ensure
      chain.try do |active|
        begin
          active.commit_and_wait unless active.drained?
        ensure
          unless active.drained?
            @@in_flight << -> { buffers.object_id; plan.try(&.object_id); active.object_id; nil }
          end
        end
      end
      if chain.nil? || chain.not_nil!.drained?
        if active = chain
          plan.try(&.release_after_drain!(active)) unless released
          active.release
        end
        buffers.each(&.release)
      end
    end
  end
end

HyNomasReferenceTimelineProbe.run!
