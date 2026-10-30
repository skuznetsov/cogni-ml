require "option_parser"
require "json"
require "digest/sha256"
require "../src/ml/video/hy_nomas_euler"
require "./support/hy_nomas_probe_instrumentation"

# Independent fixture contract for the opt-in native Euler boundary probe.
# The fixture values are produced by the pinned PyTorch EulerSolver source and
# carry its exact source identity; no model weights or generation inputs load.
module HyNomasEulerGPUProbe
  UPSTREAM_REVISION      = "911cea6458a89c2663513ff7099561ec190dd491"
  UPSTREAM_SOLVER_SHA256 = "e7996f947b46835f97b66059e0f43019b744ecec5cda380432124f70e169e59a"
  EXPECTED_ORACLE_PATH   = "hymm/pipeline/solvers/euler.py"
  ATOL                   = 2e-5_f32
  MAX_ELEMENTS           =      256

  CAPTURE_SOURCE = <<-METAL
    #include <metal_stdlib>
    using namespace metal;

    kernel void hy_nomas_euler_probe_capture_timestep(
        const device float *timestep [[buffer(0)]],
        device float *diagnostics [[buffer(1)]],
        constant uint &step [[buffer(2)]],
        uint tid [[thread_position_in_grid]]) {
      if (tid == 0u) diagnostics[step] = timestep[0];
    }

    kernel void hy_nomas_euler_probe_capture_latent(
        const device float *latent [[buffer(0)]],
        device float *diagnostics [[buffer(1)]],
        constant uint &count [[buffer(2)]],
        constant uint &base_offset [[buffer(3)]],
        uint tid [[thread_position_in_grid]]) {
      if (tid < count) diagnostics[base_offset + tid] = latent[tid];
    }
    METAL

  record Fixture,
    sha256 : String,
    steps : Int32,
    element_count : Int32,
    initial_latent : Array(Float32),
    expected_initial_latent_bf16 : Array(Float32),
    velocity : Array(Float32),
    expected_sigmas : Array(Float32),
    expected_timesteps : Array(Float32),
    expected_dts : Array(Float32),
    expected_step_latents : Array(Float32),
    expected_final_latent : Array(Float32)

  def self.valid_sha?(value : String) : Bool
    value.matches?(/\A[0-9a-f]{64}\z/)
  end

  def self.float_array(document : JSON::Any, key : String) : Array(Float32)
    document[key].as_a.map do |value|
      number = value.as_f.to_f32
      raise ArgumentError.new("fixture #{key} contains a non-finite value") unless number.finite?
      number
    end
  end

  def self.load_fixture(path : String, expected_sha : String) : Fixture
    unless valid_sha?(expected_sha) && Path[path].absolute? && File.file?(path) &&
           !File.symlink?(path) && File.realpath(path) == path && File.size(path) <= 1_048_576
      raise ArgumentError.new("fixture must be a canonical absolute JSON file with a trusted SHA-256")
    end
    raw = File.read(path)
    digest = Digest::SHA256.hexdigest(raw)
    raise ArgumentError.new("Euler fixture trusted identity mismatch") unless digest == expected_sha

    document = JSON.parse(raw)
    v2 = document["schema"].as_s == "cogni_ml.hy_nomas_euler_probe_fixture/v2"
    expected_keys = %w(
      schema upstream_revision oracle steps shift num_train_timesteps target_dtype
      initial_latent_f32 expected_initial_latent_bf16_f32 velocity_f32
      expected_timesteps_f32 expected_final_latent_f32
    )
    expected_keys.concat(%w(expected_sigmas_f32 expected_dts_f32 expected_step_latents_f32)) if v2
    unless document.as_h.keys.sort == expected_keys.sort &&
           (v2 || document["schema"].as_s == "cogni_ml.hy_nomas_euler_probe_fixture/v1") &&
           document["upstream_revision"].as_s == UPSTREAM_REVISION
      raise ArgumentError.new("Euler fixture schema or pinned revision mismatch")
    end
    oracle = document["oracle"]
    unless oracle.as_h.keys.sort == %w(path sha256).sort &&
           oracle["path"].as_s == EXPECTED_ORACLE_PATH &&
           oracle["sha256"].as_s == UPSTREAM_SOLVER_SHA256
      raise ArgumentError.new("Euler fixture does not identify the pinned upstream solver source")
    end
    steps = document["steps"].as_i
    unless (1..ML::Video::HyNomasEulerPlan::MAX_STEPS).includes?(steps) && (v2 || steps == 8) && document["shift"].as_f == 5.0 &&
           document["num_train_timesteps"].as_i == 1000 &&
           document["target_dtype"].as_s == "bfloat16"
      raise ArgumentError.new("Euler fixture must describe 1..128 steps, shift 5, 1000 train steps, BF16 target; v1 requires eight steps")
    end

    latent = float_array(document, "initial_latent_f32")
    initial_bf16 = float_array(document, "expected_initial_latent_bf16_f32")
    velocity = float_array(document, "velocity_f32")
    timesteps = float_array(document, "expected_timesteps_f32")
    sigmas = v2 ? float_array(document, "expected_sigmas_f32") : [] of Float32
    dts = v2 ? float_array(document, "expected_dts_f32") : [] of Float32
    states = v2 ? float_array(document, "expected_step_latents_f32") : [] of Float32
    expected = float_array(document, "expected_final_latent_f32")
    count = latent.size.to_i32
    unless (32..MAX_ELEMENTS).includes?(count) && (count == 32 || count == MAX_ELEMENTS) &&
           initial_bf16.size == count && velocity.size == count &&
           expected.size == count && timesteps.size == steps &&
           (!v2 || (sigmas.size == steps && dts.size == steps && states.size == count * steps))
      raise ArgumentError.new("Euler fixture requires 32 or 256 elements and matching schedule/state extents")
    end
    if v2 && states[-count, count] != expected
      raise ArgumentError.new("Euler fixture final state differs from its last intermediate state")
    end
    Fixture.new(digest, steps, count, latent, initial_bf16, velocity, sigmas, timesteps, dts, states, expected)
  end

  def self.close!(label : String, actual : Float32, expected : Float32) : Float32
    unless actual.finite? && expected.finite?
      raise ArgumentError.new("#{label} contains a non-finite result")
    end
    difference = (actual - expected).abs
    raise ArgumentError.new("#{label} differs by #{difference}, tolerance is #{ATOL}") if difference > ATOL
    difference
  end

  def self.run(fixture : Fixture) : Nil
    count = fixture.element_count
    steps = fixture.steps
    latent = ML::MetalBuffer.from_array(fixture.initial_latent)
    velocity = ML::MetalBuffer.from_array(fixture.velocity)
    timestep = ML::MetalBuffer.new(sizeof(Float32).to_i64)
    diagnostic_count = steps + (steps + 1) * count
    diagnostics = ML::MetalBuffer.new(diagnostic_count.to_i64 * sizeof(Float32))
    plan = ML::Video::HyNomasEulerPlan.new(latent, velocity, timestep, steps: steps, shift: 5.0_f32)
    capture_timestep = ML::Metal::ComputePipeline.new(
      "hy_nomas_euler_probe_capture_timestep", CAPTURE_SOURCE)
    capture_latent = ML::Metal::ComputePipeline.new(
      "hy_nomas_euler_probe_capture_latent", CAPTURE_SOURCE)
    chain = ML::Metal::MPSCommandChain.new
    begin
      expected_schedule_max_abs = 0.0_f32
      sigma_max_abs = 0.0_f32
      dt_max_abs = 0.0_f32
      schedule = plan.schedule
      steps.times do |step|
        expected_schedule_max_abs = Math.max(expected_schedule_max_abs,
          close!("CPU schedule timestep[#{step}]", plan.timestep_at(step), fixture.expected_timesteps[step]))
        unless fixture.expected_sigmas.empty?
          sigma_max_abs = Math.max(sigma_max_abs,
            close!("CPU schedule sigma[#{step}]", schedule[step].sigma, fixture.expected_sigmas[step]))
          dt_max_abs = Math.max(dt_max_abs,
            close!("CPU schedule dt[#{step}]", plan.dt_at(step), fixture.expected_dts[step]))
        end
      end

      reads_before = ML::MetalBuffer.hy_text_probe_readbacks
      plan_timestep = 0
      steps.times do |step|
        plan.encode_timestep(chain, step)
        chain.encoder.set_pipeline(capture_timestep)
          .set_buffer(timestep, 0)
          .set_buffer(diagnostics, 1, ML::Metal::BufferAccess::Write)
          .set_value(step.to_u32, 2)
          .dispatch_1d(1)
        if step == 0
          chain.encoder.set_pipeline(capture_latent)
            .set_buffer(latent, 0)
            .set_buffer(diagnostics, 1, ML::Metal::BufferAccess::Write)
            .set_value(count.to_u32, 2)
            .set_value(steps.to_u32, 3)
            .dispatch_1d(count)
        end
        plan.encode_update(chain, step)
        chain.encoder.set_pipeline(capture_latent)
          .set_buffer(latent, 0)
          .set_buffer(diagnostics, 1, ML::Metal::BufferAccess::Write)
          .set_value(count.to_u32, 2)
          .set_value((steps + (step + 1) * count).to_u32, 3)
          .dispatch_1d(count)
        plan_timestep += 1
      end

      preterminal_readbacks = ML::MetalBuffer.hy_text_probe_readbacks - reads_before
      preterminal_waits = chain.wait_count
      unless preterminal_readbacks == 0 && preterminal_waits == 0 && plan_timestep == steps
        raise ArgumentError.new("Euler encode performed a preterminal read/wait or missed a step")
      end

      chain.commit_and_wait
      raise ArgumentError.new("Euler probe requires one terminal wait") unless chain.wait_count == 1
      actual = diagnostics.read(diagnostic_count)
      terminal_readbacks = ML::MetalBuffer.hy_text_probe_readbacks - reads_before
      raise ArgumentError.new("Euler probe requires one packed terminal readback") unless terminal_readbacks == 1
      timesteps = actual[0, steps]
      initial_latent_bf16 = actual[steps, count]
      all_step_latents = actual[steps + count, count * steps]
      final_latent = all_step_latents[-count, count]
      timestep_max_abs = 0.0_f32
      timesteps.each_with_index do |value, step|
        timestep_max_abs = Math.max(timestep_max_abs,
          close!("GPU timestep[#{step}]", value, fixture.expected_timesteps[step]))
      end
      initial_latent_max_abs = 0.0_f32
      initial_latent_bf16.each_with_index do |value, index|
        initial_latent_max_abs = Math.max(initial_latent_max_abs,
          close!("initial BF16 boundary[#{index}]", value, fixture.expected_initial_latent_bf16[index]))
      end
      final_max_abs = 0.0_f32
      step_max_abs = Array(Float32).new(steps, 0.0_f32)
      fixture.expected_step_latents.each_with_index do |expected, index|
        step = index // count
        step_max_abs[step] = Math.max(step_max_abs[step],
          close!("step #{step} latent[#{index % count}]", all_step_latents[index], expected))
      end
      final_latent.each_with_index do |value, index|
        final_max_abs = Math.max(final_max_abs,
          close!("final latent[#{index}]", value, fixture.expected_final_latent[index]))
      end

      puts({schema: "cogni_ml.hy_nomas_euler_probe_result/v1",
            fixture_sha256: fixture.sha256, upstream_revision: UPSTREAM_REVISION,
            upstream_solver_sha256: UPSTREAM_SOLVER_SHA256, element_count: count,
            steps: steps, shift: 5.0, tolerance_atol: ATOL,
            schedule_cpu_max_abs: expected_schedule_max_abs,
            sigma_cpu_max_abs: sigma_max_abs, dt_cpu_max_abs: dt_max_abs,
            timestep_gpu_max_abs: timestep_max_abs,
            initial_latent_bf16_max_abs: initial_latent_max_abs,
            final_latent_max_abs: final_max_abs,
            intermediate_latents_checked: !fixture.expected_step_latents.empty?,
            step_latent_max_abs: step_max_abs,
            preterminal_readbacks: preterminal_readbacks,
            preterminal_waits: preterminal_waits, terminal_readbacks: terminal_readbacks,
            terminal_waits: chain.wait_count,
            gpu_seconds: chain.gpu_elapsed_seconds, passed: true}.to_json)
    ensure
      chain.commit_and_wait unless chain.completed?
      plan.release
      diagnostics.release
      timestep.release
      velocity.release
      latent.release
      chain.release
    end
  end
end

fixture_path = ""
fixture_sha = ""
OptionParser.parse do |parser|
  parser.banner = "GPU Euler scheduler probe against a pinned independent PyTorch fixture"
  parser.on("--fixture PATH", "Independent upstream Euler fixture JSON") { |value| fixture_path = value }
  parser.on("--fixture-sha SHA256", "Trusted fixture JSON digest") { |value| fixture_sha = value }
end
unless ARGV.empty? && !fixture_path.empty? && !fixture_sha.empty?
  raise ArgumentError.new("--fixture and --fixture-sha are required")
end

HyNomasEulerGPUProbe.run(HyNomasEulerGPUProbe.load_fixture(fixture_path, fixture_sha))
