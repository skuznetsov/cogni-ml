require "digest/sha256"
require "json"
require "option_parser"
require "../src/ml/video/hy_nomas_segment_edit"
require "../src/ml/video/hy_nomas_timeline_ops"
require "./support/hy_nomas_probe_instrumentation"

module HyNomasSegmentEditProbe
  TOTAL_FRAMES           = 18
  SEGMENT_FRAMES         =  3
  CHANNELS               =  2
  HEIGHT                 =  1
  WIDTH                  =  2
  ROPE_HEAD_DIM          =  2
  RUN_SEGMENTS           =  3
  TARGET_SEGMENT         =  1 # B
  SOURCE_SEGMENT         =  5 # F, resident and outside the active A/B/C run
  RESIDENT               =  6
  UPSTREAM_REVISION      = "911cea6458a89c2663513ff7099561ec190dd491"
  UPSTREAM_SOURCE_HASHES = {
    "hymm/pipeline/causal_ar.py"   => "f9aae49b14becb87d958665f763e510c28eeaaad9292f00ac511d1146202949f",
    "hymm/models/dit/nomas_cpt.py" => "acf827bb7feb84c9046618737146544843a3f50c3500a3b16925dadeb6c5bdee",
  }

  FIXTURE_SOURCE = <<-METAL
    #include <metal_stdlib>
    using namespace metal;

    kernel void hsed_fixture_denoise_a(
        device float *segment [[buffer(0)]],
        uint index [[thread_position_in_grid]]) {
      segment[index] = 0.5f + float(index) * 0.125f;
    }

    kernel void hsed_fixture_denoise_b(
        device float *segment [[buffer(0)]],
        uint index [[thread_position_in_grid]]) {
      segment[index] = 4.0f + float(index) * 0.0625f;
    }

    kernel void hsed_fixture_copy(
        device const float *source [[buffer(0)]],
        device float *destination [[buffer(1)]],
        constant uint &count [[buffer(2)]],
        uint index [[thread_position_in_grid]]) {
      if (index < count)
        destination[index] = source[index];
    }

    kernel void hsed_fixture_cache_consumer(
        device const float *segment [[buffer(0)]],
        device float *sink [[buffer(1)]],
        uint index [[thread_position_in_grid]]) {
      if (index == 0u)
        sink[0] = segment[0];
    }
    METAL

  def self.initial_latent(geometry : ML::Video::HyNomasTimelineOps::Geometry) : Array(Float32)
    Array(Float32).new(geometry.timeline_elements) { |index| (index + 1).to_f32 / 16.0_f32 }
  end

  def self.cosine_bank(geometry : ML::Video::HyNomasTimelineOps::Geometry) : Array(Float32)
    Array(Float32).new(geometry.rope_bank_elements) { |index| (index + 1).to_f32 / 16.0_f32 }
  end

  def self.sine_bank(geometry : ML::Video::HyNomasTimelineOps::Geometry) : Array(Float32)
    Array(Float32).new(geometry.rope_bank_elements) { |index| (index + 65).to_f32 / 32.0_f32 }
  end

  def self.timeline_index(geometry : ML::Video::HyNomasTimelineOps::Geometry,
                          segment_index : Int32, segment_element : Int32) : Int32
    spatial = geometry.latent_height * geometry.latent_width
    frame_area = geometry.segment_latent_frames * spatial
    channel = segment_element // frame_area
    local_frame = (segment_element // spatial) % geometry.segment_latent_frames
    spatial_index = segment_element % spatial
    global_frame = geometry.segment_start(segment_index) + local_frame
    (channel * geometry.total_latent_frames + global_frame) * spatial + spatial_index
  end

  def self.segment_values(geometry : ML::Video::HyNomasTimelineOps::Geometry,
                          bank : Array(Float32), segment_index : Int32) : Array(Float32)
    Array(Float32).new(geometry.segment_elements) do |index|
      bank[timeline_index(geometry, segment_index, index)]
    end
  end

  def self.rope_values(geometry : ML::Video::HyNomasTimelineOps::Geometry,
                       bank : Array(Float32), segment_index : Int32) : Array(Float32)
    offset = geometry.rope_offset(segment_index)
    bank[offset, geometry.rope_elements_per_segment]
  end

  def self.denoised_values(geometry : ML::Video::HyNomasTimelineOps::Geometry,
                           base : Float32, step : Float32) : Array(Float32)
    Array(Float32).new(geometry.segment_elements) do |index|
      base + index.to_f32 * step
    end
  end

  def self.expected_after_a(geometry : ML::Video::HyNomasTimelineOps::Geometry,
                            initial : Array(Float32)) : Array(Float32)
    expected = initial.dup
    clean_a = denoised_values(geometry, 0.5_f32, 0.125_f32)
    geometry.segment_elements.times do |index|
      expected[timeline_index(geometry, 0, index)] = clean_a[index]
    end
    expected
  end

  def self.expected_after_b(geometry : ML::Video::HyNomasTimelineOps::Geometry,
                            after_a : Array(Float32)) : Array(Float32)
    expected = after_a.dup
    clean_b = denoised_values(geometry, 4.0_f32, 0.0625_f32)
    geometry.segment_elements.times do |index|
      expected[timeline_index(geometry, TARGET_SEGMENT, index)] = clean_b[index]
    end
    expected
  end

  def self.require_exact!(actual : Array(Float32), expected : Array(Float32), label : String) : Nil
    raise "#{label}: extent mismatch #{actual.size} != #{expected.size}" unless actual.size == expected.size
    actual.each_with_index do |value, index|
      target = expected[index]
      unless value.finite? && target.finite? && value.unsafe_as(UInt32) == target.unsafe_as(UInt32)
        raise "#{label}[#{index}] mismatch: #{value} != #{target}"
      end
    end
  end

  def self.write_fresh(path : String, bytes : Bytes) : Nil
    fd = LibC.open(path, LibC::O_WRONLY | LibC::O_CREAT | LibC::O_EXCL, 0o600)
    raise IO::Error.new("refusing to replace segment-edit output #{path}") if fd < 0
    io = IO::FileDescriptor.new(fd)
    begin
      io.write(bytes)
    ensure
      io.close
    end
  end

  def self.write_artifact(directory : String, name : String, values : Array(Float32),
                          shape : Array(Int32)) : NamedTuple(path: String, sha256: String,
    bytes: Int32, shape: Array(Int32))
    path = File.join(directory, name + ".f32le")
    bytes = Bytes.new(values.to_unsafe.as(UInt8*), values.size * sizeof(Float32))
    write_fresh(path, bytes)
    {path: File.basename(path), sha256: Digest::SHA256.hexdigest(bytes),
     bytes: bytes.size, shape: shape}
  end

  def self.run!(output_dir : String) : Nil
    unless Path[output_dir].absolute? && File.directory?(output_dir) &&
           !File.symlink?(output_dir) && Dir.empty?(output_dir) &&
           File.realpath(output_dir).starts_with?(File.realpath("/opt/mldata_ssd") + "/")
      raise ArgumentError.new("--output-dir must be an existing empty directory on /opt/mldata_ssd")
    end
    output_dir = File.realpath(output_dir)

    geometry = ML::Video::HyNomasTimelineOps::Geometry.new(
      total_latent_frames: TOTAL_FRAMES,
      segment_latent_frames: SEGMENT_FRAMES,
      channels: CHANNELS,
      latent_height: HEIGHT,
      latent_width: WIDTH,
      rope_head_dim: ROPE_HEAD_DIM)
    initial = initial_latent(geometry)
    cosine_values = cosine_bank(geometry)
    sine_values = sine_bank(geometry)
    expected_a = expected_after_a(geometry, initial)
    expected_final = expected_after_b(geometry, expected_a)
    expected_b_noise = segment_values(geometry, initial, SOURCE_SEGMENT)
    expected_b_cosine = rope_values(geometry, cosine_values, TARGET_SEGMENT)
    expected_b_sine = rope_values(geometry, sine_values, TARGET_SEGMENT)
    expected_c_noise = segment_values(geometry, initial, 2)
    expected_c_cosine = rope_values(geometry, cosine_values, 2)
    expected_c_sine = rope_values(geometry, sine_values, 2)

    # Keep six residents while processing A/B/C. F remains unused by the run;
    # its original noise can supply a new B take without changing B's RoPE.
    timeline = ML::MetalBuffer.from_array(initial)
    after_a_timeline = ML::MetalBuffer.new(geometry.timeline_bytes)
    segment = ML::MetalBuffer.new(geometry.segment_bytes)
    b_noise = ML::MetalBuffer.new(geometry.segment_bytes)
    cosine_bank = ML::MetalBuffer.from_array(cosine_values)
    sine_bank = ML::MetalBuffer.from_array(sine_values)
    cosine = ML::MetalBuffer.new(geometry.rope_elements_per_segment.to_i64 * sizeof(Float32))
    sine = ML::MetalBuffer.new(geometry.rope_elements_per_segment.to_i64 * sizeof(Float32))
    b_cosine = ML::MetalBuffer.new(geometry.rope_elements_per_segment.to_i64 * sizeof(Float32))
    b_sine = ML::MetalBuffer.new(geometry.rope_elements_per_segment.to_i64 * sizeof(Float32))
    c_noise = ML::MetalBuffer.new(geometry.segment_bytes)
    c_cosine = ML::MetalBuffer.new(geometry.rope_elements_per_segment.to_i64 * sizeof(Float32))
    c_sine = ML::MetalBuffer.new(geometry.rope_elements_per_segment.to_i64 * sizeof(Float32))
    epsilon = ML::MetalBuffer.new(geometry.cache_noise_bank_bytes)
    epsilon_words = Array(UInt16).new(geometry.timeline_elements, 0_u16)
    epsilon.write_bytes(epsilon_words.to_unsafe.as(Pointer(UInt8)),
      epsilon_words.size * sizeof(UInt16))
    timestep = ML::MetalBuffer.new(sizeof(Float32).to_i64)
    cache_sink = ML::MetalBuffer.new(sizeof(Float32).to_i64)
    buffers = [timeline, after_a_timeline, segment, b_noise, cosine_bank, sine_bank,
               cosine, sine, b_cosine, b_sine, c_noise, c_cosine, c_sine, epsilon, timestep, cache_sink]

    chain = ML::Metal::MPSCommandChain.new
    edit = ML::Video::HyNomasSegmentNoiseEdit.new(RUN_SEGMENTS, TARGET_SEGMENT,
      SOURCE_SEGMENT, RESIDENT)
    plan_a = ML::Video::HyNomasTimelineOps::Plan.new(geometry,
      timeline: timeline, segment_latent: segment, cosine_bank: cosine_bank, sine_bank: sine_bank,
      cosine: cosine, sine: sine, cache_eps_bf16: epsilon, cache_timestep: timestep,
      run_segments: RUN_SEGMENTS)
    plan_fork = nil.as(ML::Video::HyNomasTimelineOps::Plan?)
    denoise_a = ML::Metal::ComputePipeline.new("hsed_fixture_denoise_a", FIXTURE_SOURCE)
    denoise_b = ML::Metal::ComputePipeline.new("hsed_fixture_denoise_b", FIXTURE_SOURCE)
    copy = ML::Metal::ComputePipeline.new("hsed_fixture_copy", FIXTURE_SOURCE)
    cache_consumer = ML::Metal::ComputePipeline.new("hsed_fixture_cache_consumer", FIXTURE_SOURCE)
    plan_a_released = false
    wait_attempted = false

    begin
      allocation_start = ML::MetalBuffer.hy_text_probe_allocations
      readback_start = ML::MetalBuffer.hy_text_probe_readbacks

      # Create a clean A and queue its checkpoint cache boundary.
      plan_a.prepare_segment!(chain, 0)
      chain.encoder.set_pipeline(denoise_a)
        .set_buffer(segment, 0, ML::Metal::BufferAccess::ReadWrite)
        .dispatch_1d(geometry.segment_elements)
      plan_a.mark_denoise_complete!(chain, 0)
      plan_a.scatter_generated!(chain, 0)
      plan_a.write_cache_noise!(chain, 0)
      chain.encoder.set_pipeline(cache_consumer)
        .set_buffer(segment, 0)
        .set_buffer(cache_sink, 1, ML::Metal::BufferAccess::Write)
        .dispatch_1d(1)
      plan_a.record_cache_forward_enqueued!(chain, 0)
      plan_a.advance_segment!(chain, 0)

      # Preserve the actual after-A state before either edited or resumed B
      # work can touch the timeline.
      chain.encoder.set_pipeline(copy)
        .set_buffer(timeline, 0)
        .set_buffer(after_a_timeline, 1, ML::Metal::BufferAccess::Write)
        .set_value(geometry.timeline_elements.to_u32, 2)
        .dispatch_1d(geometry.timeline_elements)

      # Restore a fresh schedule at the exact parent boundary. B gathers from
      # F but copies its own global RoPE; snapshot both before B's denoise.
      plan_fork = ML::Video::HyNomasTimelineOps::Plan.new(geometry,
        timeline: timeline, segment_latent: segment, cosine_bank: cosine_bank, sine_bank: sine_bank,
        cosine: cosine, sine: sine, cache_eps_bf16: epsilon, cache_timestep: timestep,
        run_segments: RUN_SEGMENTS)
      plan_fork.not_nil!.restore_boundary!(TARGET_SEGMENT)
      plan_fork.not_nil!.prepare_segment!(chain, TARGET_SEGMENT,
        noise_source_segment: edit.source_for(TARGET_SEGMENT))
      chain.encoder.set_pipeline(copy)
        .set_buffer(segment, 0)
        .set_buffer(b_noise, 1, ML::Metal::BufferAccess::Write)
        .set_value(geometry.segment_elements.to_u32, 2)
        .dispatch_1d(geometry.segment_elements)
      chain.encoder.set_pipeline(copy)
        .set_buffer(cosine, 0)
        .set_buffer(b_cosine, 1, ML::Metal::BufferAccess::Write)
        .set_value(geometry.rope_elements_per_segment.to_u32, 2)
        .dispatch_1d(geometry.rope_elements_per_segment)
      chain.encoder.set_pipeline(copy)
        .set_buffer(sine, 0)
        .set_buffer(b_sine, 1, ML::Metal::BufferAccess::Write)
        .set_value(geometry.rope_elements_per_segment.to_u32, 2)
        .dispatch_1d(geometry.rope_elements_per_segment)

      # Regenerate B, scatter only into B, and consume the same cache boundary
      # needed before advancing the sequence to C.
      chain.encoder.set_pipeline(denoise_b)
        .set_buffer(segment, 0, ML::Metal::BufferAccess::ReadWrite)
        .dispatch_1d(geometry.segment_elements)
      plan_fork.not_nil!.mark_denoise_complete!(chain, TARGET_SEGMENT)
      plan_fork.not_nil!.scatter_generated!(chain, TARGET_SEGMENT)
      plan_fork.not_nil!.write_cache_noise!(chain, TARGET_SEGMENT)
      chain.encoder.set_pipeline(cache_consumer)
        .set_buffer(segment, 0)
        .set_buffer(cache_sink, 1, ML::Metal::BufferAccess::Write)
        .dispatch_1d(1)
      plan_fork.not_nil!.record_cache_forward_enqueued!(chain, TARGET_SEGMENT)
      plan_fork.not_nil!.advance_segment!(chain, TARGET_SEGMENT)

      # Prepare C on its normal initial-noise/global-RoPE path, then capture
      # those buffers for verification without denoising or scattering C.
      plan_fork.not_nil!.prepare_segment!(chain, 2)
      chain.encoder.set_pipeline(copy)
        .set_buffer(segment, 0)
        .set_buffer(c_noise, 1, ML::Metal::BufferAccess::Write)
        .set_value(geometry.segment_elements.to_u32, 2)
        .dispatch_1d(geometry.segment_elements)
      chain.encoder.set_pipeline(copy)
        .set_buffer(cosine, 0)
        .set_buffer(c_cosine, 1, ML::Metal::BufferAccess::Write)
        .set_value(geometry.rope_elements_per_segment.to_u32, 2)
        .dispatch_1d(geometry.rope_elements_per_segment)
      chain.encoder.set_pipeline(copy)
        .set_buffer(sine, 0)
        .set_buffer(c_sine, 1, ML::Metal::BufferAccess::Write)
        .set_value(geometry.rope_elements_per_segment.to_u32, 2)
        .dispatch_1d(geometry.rope_elements_per_segment)

      encode_allocations = ML::MetalBuffer.hy_text_probe_allocations - allocation_start
      encode_readbacks = ML::MetalBuffer.hy_text_probe_readbacks - readback_start
      preterminal_waits = chain.wait_count
      unless encode_allocations == 0 && encode_readbacks == 0 && preterminal_waits == 0
        raise "segment-edit encode allocated, read back, or waited before terminal"
      end

      wait_attempted = true
      chain.commit_and_wait
      unless chain.completed? && chain.wait_count == 1
        raise "segment-edit probe requires exactly one terminal wait"
      end

      # All readbacks happen after the sole drain. The separate after-A copy
      # proves prefix equivalence, while the final timeline checks B-only scatter.
      actual_after_a = after_a_timeline.read(geometry.timeline_elements)
      actual_timeline = timeline.read(geometry.timeline_elements)
      actual_b_noise = b_noise.read(geometry.segment_elements)
      actual_b_cosine = b_cosine.read(geometry.rope_elements_per_segment)
      actual_b_sine = b_sine.read(geometry.rope_elements_per_segment)
      actual_c_noise = c_noise.read(geometry.segment_elements)
      actual_c_cosine = c_cosine.read(geometry.rope_elements_per_segment)
      actual_c_sine = c_sine.read(geometry.rope_elements_per_segment)

      require_exact!(actual_after_a, expected_a, "after-A boundary snapshot")
      require_exact!(actual_timeline, expected_final, "B-only scatter preserves A and C/F suffix")
      require_exact!(actual_b_noise, expected_b_noise, "B noise gathered from F")
      require_exact!(actual_b_cosine, expected_b_cosine, "B global cosine")
      require_exact!(actual_b_sine, expected_b_sine, "B global sine")
      require_exact!(actual_c_noise, expected_c_noise, "C uses its own initial noise")
      require_exact!(actual_c_cosine, expected_c_cosine, "C global cosine")
      require_exact!(actual_c_sine, expected_c_sine, "C global sine")
      raise "fixture must distinguish B and F RoPE" if expected_b_cosine == rope_values(geometry, cosine_values, SOURCE_SEGMENT)
      raise "fixture must distinguish B and F sine" if expected_b_sine == rope_values(geometry, sine_values, SOURCE_SEGMENT)

      # The edited checkpoint identity begins only after B. It retains the
      # original input/conditioning/executable contract in its base hash.
      base_contract = "cogni_ml.hy_nomas_segment_edit_timeline_probe/base-after-a/v1"
      boundary_contract = edit.bind_contract(base_contract, TARGET_SEGMENT)
      edited_contract = edit.bind_contract(base_contract, TARGET_SEGMENT + 1)
      unless boundary_contract == base_contract && edited_contract != base_contract
        raise "segment-edit contract must preserve the parent through B entry and fork after B"
      end

      postterminal_readbacks = ML::MetalBuffer.hy_text_probe_readbacks - readback_start
      unless postterminal_readbacks == 8
        raise "segment-edit probe expected eight post-terminal artifact readbacks"
      end

      artifacts = {
        "after_a_timeline" => write_artifact(output_dir, "after-a-timeline", actual_after_a,
          [1, CHANNELS, TOTAL_FRAMES, HEIGHT, WIDTH]),
        "timeline" => write_artifact(output_dir, "timeline", actual_timeline,
          [1, CHANNELS, TOTAL_FRAMES, HEIGHT, WIDTH]),
        "b_noise" => write_artifact(output_dir, "b-noise", actual_b_noise,
          [1, CHANNELS, SEGMENT_FRAMES, HEIGHT, WIDTH]),
        "b_cosine" => write_artifact(output_dir, "b-cosine", actual_b_cosine,
          [SEGMENT_FRAMES * HEIGHT * WIDTH, ROPE_HEAD_DIM]),
        "b_sine" => write_artifact(output_dir, "b-sine", actual_b_sine,
          [SEGMENT_FRAMES * HEIGHT * WIDTH, ROPE_HEAD_DIM]),
        "c_noise" => write_artifact(output_dir, "c-noise", actual_c_noise,
          [1, CHANNELS, SEGMENT_FRAMES, HEIGHT, WIDTH]),
        "c_cosine" => write_artifact(output_dir, "c-cosine", actual_c_cosine,
          [SEGMENT_FRAMES * HEIGHT * WIDTH, ROPE_HEAD_DIM]),
        "c_sine" => write_artifact(output_dir, "c-sine", actual_c_sine,
          [SEGMENT_FRAMES * HEIGHT * WIDTH, ROPE_HEAD_DIM]),
      }
      receipt = {
        schema: "cogni_ml.hy_nomas_segment_edit_timeline_probe/v1",
        scope: "synthetic GPU after-A boundary, B-from-F gather/scatter, and C prepare; no neural model or decoder",
        upstream: {
          revision:       UPSTREAM_REVISION,
          source_sha256:  UPSTREAM_SOURCE_HASHES,
          contract_scope: "channel-first noise slicing and global RoPE offset source anchors; no DiT parity claim",
        },
        resident_segments: RESIDENT, run_segments: RUN_SEGMENTS,
        checkpoint_boundary_next_segment: TARGET_SEGMENT,
        target_segment: TARGET_SEGMENT, source_segment: SOURCE_SEGMENT,
        noise_edit: edit.identity,
        noise_edit_identity_sha256: Digest::SHA256.hexdigest(edit.identity.to_json),
        boundary_contract_sha256: Digest::SHA256.hexdigest(boundary_contract),
        fork_contract_sha256: Digest::SHA256.hexdigest(edited_contract),
        checks: {
          clean_a_retained: true, target_b_noise_from_f: true,
          target_b_scatter_only_to_b: true, target_b_rope_uses_b: true,
          c_noise_uses_c: true, c_rope_uses_c: true,
          future_suffix_untouched: true, after_a_snapshot_preserved: true,
        },
        metrics: {
          encode_allocations:     encode_allocations,
          encode_readbacks:       encode_readbacks,
          preterminal_waits:      preterminal_waits,
          terminal_waits:         chain.wait_count,
          postterminal_readbacks: postterminal_readbacks,
        },
        artifacts: artifacts,
        passed: true,
      }
      report_path = File.join(output_dir, "receipt.json")
      write_fresh(report_path, receipt.to_pretty_json.to_slice)
      puts receipt.to_pretty_json
    ensure
      unless wait_attempted || chain.completed?
        wait_attempted = true
        chain.commit_and_wait
      end
      if chain.drained?
        unless plan_a_released
          plan_a.release_after_drain!(chain)
          plan_a_released = true
        end
        if fork_plan = plan_fork
          fork_plan.release_after_drain!(chain)
        end
        buffers.each(&.release)
        chain.release
      end
    end
  end
end

output_dir = ""
OptionParser.parse do |options|
  options.on("--output-dir PATH", "Existing empty result directory on /opt/mldata_ssd") { |value| output_dir = value }
end
raise ArgumentError.new("--output-dir is required") if output_dir.empty?
HyNomasSegmentEditProbe.run!(output_dir)
