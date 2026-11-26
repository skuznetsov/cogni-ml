require "option_parser"
require "digest/sha256"
require "json"
require "./support/hy_nomas_t0_rgb_history"

module HyNomasT0RgbHistoryProbe
  extend self

  record Args,
    mode : String,
    inputs_path : String,
    inputs_sha256 : String,
    source_pixels_path : String,
    source_receipt_path : String,
    encoder_weights_path : String,
    encoder_weights_sha256 : String,
    expected_clip_sha256 : String,
    output_dir : String,
    budget_mib : Int64,
    plan_only : Bool

  def parse_args : Args
    mode = ""
    inputs_path = HyNomasT0RgbHistory::CONTINUATION_PATH
    inputs_sha256 = HyNomasT0RgbHistory::CONTINUATION_SHA256
    source_pixels_path = HyNomasT0RgbHistory::SOURCE_PIXELS_PATH
    source_receipt_path = HyNomasT0RgbHistory::SOURCE_RECEIPT_PATH
    encoder_weights_path = HyNomasT0RgbHistory::ENCODER_WEIGHTS_PATH
    encoder_weights_sha256 = HyNomasT0RgbHistory::ENCODER_WEIGHTS_SHA256
    expected_clip_sha256 = ""
    output_dir = ""
    budget_mib = 32_768_i64
    plan_only = false

    OptionParser.parse do |parser|
      parser.banner = "Connected source-BF16 T0 RGB-history control; moving or repeat-last, then frozen T1"
      parser.on("--context-mode MODE", "Required: moving or repeat-last") { |value| mode = value }
      parser.on("--inputs PATH", "Authenticated frozen continuation manifest") { |value| inputs_path = value }
      parser.on("--inputs-sha SHA", "Trusted continuation manifest SHA-256") { |value| inputs_sha256 = value }
      parser.on("--source-pixels PATH", "Pinned generated T0 F32 planar RGB bank") { |value| source_pixels_path = value }
      parser.on("--source-receipt PATH", "Pinned generated T0 RGB producer receipt") { |value| source_receipt_path = value }
      parser.on("--encoder-weights PATH", "Pinned BF16 DCAE encoder checkpoint") { |value| encoder_weights_path = value }
      parser.on("--encoder-weights-sha SHA", "Trusted DCAE encoder checkpoint SHA-256") { |value| encoder_weights_sha256 = value }
      parser.on("--expected-normalized-clip-sha256 SHA", "Required source-derived BF16 clip digest") { |value| expected_clip_sha256 = value }
      parser.on("--output-dir PATH", "Existing empty result directory beneath the authorized experiment root") { |value| output_dir = value }
      parser.on("--budget-mib N", "Named-byte bound; the external 10% host-memory guard is also required") { |value| budget_mib = value.to_i64 }
      parser.on("--plan-only", "CPU-only input/checkpoint authentication and aggregate memory admission") { plan_only = true }
      parser.on("-h", "--help", "Show this help") do
        puts parser
        exit
      end
    end

    raise ArgumentError.new("--context-mode is required") if mode.empty?
    HyNomasT0RgbHistory.validate_context_mode!(mode)
    HyNomasT0RgbHistory.validate_expected_clip_sha256!(expected_clip_sha256)
    raise ArgumentError.new("--inputs-sha must match the pinned continuation") unless inputs_sha256 == HyNomasT0RgbHistory::CONTINUATION_SHA256
    unless (24_576_i64..49_152_i64).includes?(budget_mib)
      raise ArgumentError.new("T0 RGB named budget must be within 24..48 GiB")
    end
    unless plan_only || !output_dir.empty?
      raise ArgumentError.new("--output-dir is required unless --plan-only is set")
    end

    Args.new(mode, inputs_path, inputs_sha256, source_pixels_path,
      source_receipt_path, encoder_weights_path, encoder_weights_sha256,
      expected_clip_sha256, output_dir, budget_mib, plan_only)
  end

  def checkpoint_inventory!(path : String, sha256 : String,
                            named_bytes : Int64, weight_bytes : Int64) : Nil
    unless sha256 == HyNomasT0RgbHistory::ENCODER_WEIGHTS_SHA256
      raise ArgumentError.new("DCAE encoder checkpoint SHA-256 differs from the pinned BF16 source")
    end
    file = ML::Video::BF16WeightCache.authenticate(path, sha256)
    begin
      bindings = ML::Video::DCAEVEncoder.checkpoint_bindings(HyNomasT0RgbHistory::CONFIG)
      plan = ML::Video::WeightPlan.new([file], bindings, named_bytes,
        reserve_bytes: named_bytes - weight_bytes,
        resident_dtype: ML::Video::WeightDType::BF16, require_all: true)
      unless bindings.size == 169 && plan.staging_bytes == 0 &&
             plan.destination_bytes == weight_bytes && plan.peak_bytes == named_bytes
        raise ArgumentError.new("DCAE encoder checkpoint inventory differs from the fixed plan")
      end
    ensure
      file.close
    end
  end

  def frozen_identity_json(identity : NamedTuple) : JSON::Any
    JSON.parse(identity.to_json)
  end

  def run(args : Args) : Nil
    started = Time.monotonic
    source = HyNomasT0RgbHistory.authenticate_source!(args.source_pixels_path, args.source_receipt_path)
    inputs = HyNomasT0RgbHistory.authenticate_continuation!(args.inputs_path, args.inputs_sha256)
    identity = HyNomasT0RgbHistory.frozen_input_identity(inputs)
    budget_bytes = args.budget_mib * 1024 * 1024
    plan = HyNomasT0RgbHistory::CombinedPlan.for(inputs, budget_bytes)
    checkpoint_inventory!(args.encoder_weights_path, args.encoder_weights_sha256,
      plan.dcae.named_bytes, plan.dcae.weights_bytes)

    HyNomasGeneration.emit("t0_rgb_memory_plan",
      schema: HyNomasT0RgbHistory::SCHEMA, context_mode: args.mode,
      named_budget_bytes: plan.named_budget_bytes,
      connected_steady_named_bytes: plan.connected_steady_bytes,
      connected_peak_named_bytes: plan.connected_peak_bytes,
      dit_peak_named_bytes: plan.dit_peak_bytes,
      dit_steady_named_bytes: plan.dit_steady_bytes,
      dcae_named_bytes: plan.dcae.named_bytes,
      dcae_weights_bytes: plan.dcae.weights_bytes,
      dcae_workspace_bytes: plan.dcae.workspace_bytes,
      normalized_clip_bytes: plan.dcae.clip_bytes,
      source_rgb_bytes: plan.dcae.source_bytes,
      shared_t0_latent_bytes: plan.dcae.latent_bytes,
      source_pixels_sha256: source[:pixels_sha256],
      expected_normalized_clip_sha256: args.expected_clip_sha256,
      frozen_input_identity: identity,
      memory_scope: HyNomasT0RgbHistory::MEMORY_SCOPE)

    if args.plan_only
      HyNomasGeneration.emit("t0_rgb_plan_only_completed",
        context_mode: args.mode, preflight_seconds: (Time.monotonic - started).total_seconds)
      return
    end

    output_dir = HyNomasT0RgbHistory.output_dir!(args.output_dir)
    timeline_arena = ML::Video::Arena.new
    current_arena = ML::Video::Arena.new
    timeline = nil.as(ML::MetalBuffer?)
    latent = nil.as(ML::MetalBuffer?)
    reference_latent = nil.as(ML::MetalBuffer?)
    dit = nil.as(HyNomasSequence::DiT?)
    dcae = nil.as(HyNomasT0RgbHistory::Session?)
    chain = nil.as(ML::Metal::MPSCommandChain?)
    release_errors = [] of String
    request_error : Exception? = nil
    metrics = {} of String => Int64
    timings = {} of String => Float64
    artifacts = {} of String => String
    terminal_proven = false

    begin
      timeline = inputs.upload!("input.latent_bank", timeline_arena)
      latent = current_arena.alloc("t0.current_latent", 32 * 30 * 9 * 16)
      reference_latent = current_arena.alloc_weight("t0.dcae.shared_bf16_latent",
        (plan.dcae.latent_bytes // sizeof(UInt16)).to_i32, ML::Video::WeightDType::BF16)

      admission = HyNomasGeneration.authenticate_bf16_index
      setup = Time.monotonic
      dit = HyNomasSequence::DiT.new(inputs, timeline.not_nil!, latent.not_nil!, budget_bytes,
        2, admission, HyNomasSequence::HISTORY_TEXT_POLICY,
        reference_latent: reference_latent.not_nil!, source_bf16: true)
      unless dit.not_nil!.peak_named_bytes == plan.dit_peak_bytes &&
             dit.not_nil!.named_bytes == plan.dit_steady_bytes
        raise ArgumentError.new("DiT actual allocation plan differs from the connected T0 RGB admission")
      end
      timings["dit_setup_seconds"] = (Time.monotonic - setup).total_seconds
      HyNomasGeneration.emit("t0_rgb_stage", stage: "dit_prepared",
        elapsed_seconds: (Time.monotonic - started).total_seconds,
        named_bytes: dit.not_nil!.named_bytes)

      dcae = HyNomasT0RgbHistory::Session.new(reference_latent.not_nil!, args.encoder_weights_path,
        args.encoder_weights_sha256, budget_bytes)
      setup = Time.monotonic
      dcae.not_nil!.prepare!
      dcae.not_nil!.upload_source!(source[:pixels_path])
      unless dcae.not_nil!.named_bytes == plan.dcae.named_bytes &&
             dcae.not_nil!.uploaded_source_sha256 == source[:pixels_sha256]
        raise ArgumentError.new("DCAE setup/source copy differs from the preflight identity")
      end
      timings["dcae_setup_seconds"] = (Time.monotonic - setup).total_seconds
      HyNomasGeneration.emit("t0_rgb_stage", stage: "dcae_prepared",
        elapsed_seconds: (Time.monotonic - started).total_seconds,
        named_bytes: plan.connected_steady_bytes)

      chain = ML::Metal::MPSCommandChain.new
      allocations = ML::MetalBuffer.hy_text_probe_allocations
      readbacks = ML::MetalBuffer.hy_text_probe_readbacks
      run_started = Time.monotonic
      HyNomasGeneration.emit("t0_rgb_stage", stage: "dcae_encode_begin",
        elapsed_seconds: (Time.monotonic - started).total_seconds)
      dcae.not_nil!.encode!(chain.not_nil!, args.mode)
      HyNomasGeneration.emit("t0_rgb_stage", stage: "dcae_encode_enqueued",
        elapsed_seconds: (Time.monotonic - started).total_seconds)
      dit.not_nil!.encode(chain.not_nil!)
      HyNomasGeneration.emit("t0_rgb_stage", stage: "dit_encode_enqueued",
        elapsed_seconds: (Time.monotonic - started).total_seconds)
      metrics["chain_preterminal_allocations"] = ML::MetalBuffer.hy_text_probe_allocations - allocations
      metrics["chain_preterminal_readbacks"] = ML::MetalBuffer.hy_text_probe_readbacks - readbacks
      metrics["chain_preterminal_waits"] = chain.not_nil!.wait_count.to_i64
      HyNomasT0RgbHistory.validate_preterminal_metrics!(
        metrics["chain_preterminal_allocations"], metrics["chain_preterminal_readbacks"],
        metrics["chain_preterminal_waits"])

      chain.not_nil!.commit_and_wait
      terminal_proven = chain.not_nil!.drained? && chain.not_nil!.wait_count == 1
      metrics["terminal_waits"] = chain.not_nil!.wait_count.to_i64
      HyNomasT0RgbHistory.validate_encode_metrics!(
        metrics["chain_preterminal_allocations"], metrics["chain_preterminal_readbacks"],
        metrics["chain_preterminal_waits"], metrics["terminal_waits"])
      timings["chain_wall_seconds"] = (Time.monotonic - run_started).total_seconds
      timings["chain_gpu_seconds"] = chain.not_nil!.gpu_elapsed_seconds

      # These are the exact shared buffers consumed by the connected encoder
      # and continuation. Hash and export only after the one terminal drain.
      clip_sha = HyNomasT0RgbHistory.write_fresh_buffer!(
        File.join(output_dir, "normalized-clip.bf16le"), dcae.not_nil!.normalized_clip,
        plan.dcae.clip_bytes)
      HyNomasT0RgbHistory.validate_actual_clip_sha256!(clip_sha, args.expected_clip_sha256)
      endpoint_sha = HyNomasT0RgbHistory.write_latent_f32!(
        File.join(output_dir, "dcae-t0-latent.f32le"), dcae.not_nil!.latent)
      bank = Bytes.new(timeline.not_nil!.contents.as(UInt8*), timeline.not_nil!.size.to_i32)
      bank_hash = Digest::SHA256.hexdigest(bank)
      bank_path = File.join(output_dir, "latent-bank.f32le")
      HyNomasGeneration.write_fresh_output(bank_path) { |io| io.write(bank) }
      bank_slices = HyNomasT0RgbHistory.logical_prefix_digests(bank)
      unless endpoint_sha == bank_slices[:t0_sha256]
        raise ArgumentError.new("DCAE endpoint bytes differ from the final logical T0 bank slice")
      end
      artifacts = {
        "latent_bank" => bank_path,
        "latent_bank_sha256" => bank_hash,
        "t0_sha256" => bank_slices[:t0_sha256],
        "t1_sha256" => bank_slices[:t1_sha256],
        "prefix60_sha256" => bank_slices[:prefix60_sha256],
        "normalized_clip_bf16" => File.join(output_dir, "normalized-clip.bf16le"),
        "normalized_clip_bf16_sha256" => clip_sha,
        "dcae_latent_f32" => File.join(output_dir, "dcae-t0-latent.f32le"),
        "dcae_latent_f32_sha256" => endpoint_sha,
      }

      receipt = {
        schema: HyNomasT0RgbHistory::SCHEMA,
        upstream_revision: HyNomasGeneration::TimelineInputs::PINNED_REVISION,
        context_mode: args.mode,
        inputs: inputs.file.path,
        inputs_sha256: inputs.file.sha256,
        prompt: inputs.base.prompt,
        qwen_valid_tokens: inputs.base.valid_text_tokens,
        text_cache_policy: "once_per_scene",
        source_bf16: true,
        segments: 2,
        steps_per_segment: 4,
        t0_denoise_steps: dit.not_nil!.t0_denoise_steps,
        t0_cache_writer_forwards: dit.not_nil!.t0_cache_writer_forwards,
        source_pixels: source[:pixels_path],
        source_pixels_sha256: dcae.not_nil!.uploaded_source_sha256.not_nil!,
        source_pixels_bytes: HyNomasT0RgbHistory::SOURCE_BYTES,
        source_pixels_dtype: "F32 little-endian planar RGB",
        source_pixels_shape: HyNomasT0RgbHistory::SOURCE_SHAPE,
        source_receipt: source[:receipt_path],
        source_receipt_sha256: source[:receipt_sha256],
        source_selection: args.mode == "moving" ? "frames 0..119" : "frame 119 repeated 120 times",
        normalization: "GPU F32 clamp to [0,1], then BF16 RNE(2*x-1)",
        expected_normalized_clip_bf16_sha256: args.expected_clip_sha256,
        dcae_encoder: File.realpath(args.encoder_weights_path),
        dcae_encoder_sha256: args.encoder_weights_sha256,
        frozen_input_identity: identity,
        named_memory: {
          budget_bytes: plan.named_budget_bytes,
          connected_steady_bytes: plan.connected_steady_bytes,
          connected_peak_bytes: plan.connected_peak_bytes,
          dit_peak_bytes: plan.dit_peak_bytes,
          dit_steady_bytes: plan.dit_steady_bytes,
          dcae_named_bytes: plan.dcae.named_bytes,
          dcae_weights_bytes: plan.dcae.weights_bytes,
          dcae_workspace_bytes: plan.dcae.workspace_bytes,
          normalized_clip_bytes: plan.dcae.clip_bytes,
          source_rgb_bytes: plan.dcae.source_bytes,
          shared_t0_latent_bytes: plan.dcae.latent_bytes,
          memory_scope: HyNomasT0RgbHistory::MEMORY_SCOPE,
        },
        metrics: metrics,
        timings: timings,
        output: artifacts,
        full_upstream_video_parity: false,
      }
      HyNomasGeneration.write_fresh_output(File.join(output_dir, "receipt.json")) do |io|
        io.puts(receipt.to_pretty_json)
      end
      HyNomasGeneration.emit("t0_rgb_history_completed", context_mode: args.mode,
        t0_sha256: bank_slices[:t0_sha256], t1_sha256: bank_slices[:t1_sha256],
        latent_bank_sha256: bank_hash, normalized_clip_bf16_sha256: clip_sha,
        dcae_latent_f32_sha256: endpoint_sha, terminal_waits: metrics["terminal_waits"],
        receipt: File.join(output_dir, "receipt.json"))
    rescue ex
      request_error = ex
    ensure
      may_release = true
      if active_chain = chain
        unless active_chain.drained?
          begin
            active_chain.commit_and_wait
          rescue wait_error
            release_errors << "terminal wait: #{wait_error.class}: #{wait_error.message}"
          end
        end
        may_release = active_chain.drained?
        terminal_proven ||= may_release && active_chain.wait_count == 1
        unless may_release
          release_errors << "terminal wait did not prove GPU drain; owners retained"
          HyNomasSequence.retain_in_flight do
            inputs.object_id
            timeline.try(&.object_id)
            latent.try(&.object_id)
            reference_latent.try(&.object_id)
            dit.try(&.object_id)
            dcae.try(&.object_id)
            active_chain.object_id
            timeline_arena.object_id
            current_arena.object_id
            nil
          end
        end
      end
      if may_release
        if sequence = dit
          begin
            sequence.release
          rescue ex
            release_errors << "DiT release: #{ex.class}: #{ex.message}"
          end
        end
        if encoder = dcae
          begin
            if encoder.encode_started?
              encoder.release_after_drain!(chain.not_nil!)
            else
              encoder.close_before_encode!
            end
          rescue ex
            release_errors << "DCAE release: #{ex.class}: #{ex.message}"
          end
        end
        chain.try(&.release)
        begin
          current_arena.release
          timeline_arena.release
        rescue ex
          release_errors << "arena release: #{ex.class}: #{ex.message}"
        end
      end
      if errors = request_error
        STDERR.puts("T0 RGB history failed: #{errors.class}: #{errors.message}")
      end
      unless release_errors.empty?
        STDERR.puts("T0 RGB cleanup: #{release_errors.join("; ")}")
      end
    end

    raise request_error.not_nil! if request_error
    unless terminal_proven && release_errors.empty?
      raise RuntimeError.new("T0 RGB history did not prove a clean terminal/release state")
    end
  end
end

begin
  HyNomasT0RgbHistoryProbe.run(HyNomasT0RgbHistoryProbe.parse_args)
rescue ex
  STDERR.puts("T0 RGB history probe: #{ex.class}: #{ex.message}")
  exit(1)
end
