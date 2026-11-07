require "option_parser"
require "digest/sha256"
require "./support/hy_nomas_sequence"

module HyNomasPortraitT0HistoryProbe
  extend self

  ROOT                  = "/opt/mldata_ssd/cogni-ml/hy-nomas-native-context-20261106.ui7736ue"
  INPUTS_PATH           = File.join(ROOT, "photo-continuation-v2.json")
  INPUTS_SHA            = "0f35bb09964f2b28d28069e8d39a8a1d6194607d4269de6bb824ff54d828e2ac"
  BASELINE_RECEIPT      = File.join(ROOT, "photo-native-v2/receipt.json")
  BASELINE_RECEIPT_SHA  = "ca8be135ef659497594dec9647c72240f911be8cfd6be989f0740fbf76cd66f5"
  BASELINE_BANK         = File.join(ROOT, "photo-native-v2/latent-bank.f32le")
  BASELINE_BANK_SHA     = "1717edc3d6cf33a3a374852a3dc58cd9d82d5f74965fab6dd0239bbe93da943b"
  BASELINE_BANK_BYTES   = 3_317_760_i64
  CHANNELS              =            32
  BANK_FRAMES           =           180
  LATENT_FRAMES         =            30
  PIXELS_PER_FRAME      = 9 * 16
  T0_BYTES              = CHANNELS.to_i64 * LATENT_FRAMES * PIXELS_PER_FRAME * sizeof(Float32)
  PROBE_GPU_EXTRA_BYTES = BASELINE_BANK_BYTES + T0_BYTES

  record Args,
    inputs_path : String,
    inputs_sha : String,
    baseline_receipt_path : String,
    baseline_receipt_sha : String,
    baseline_bank_path : String,
    baseline_bank_sha : String,
    output_dir : String,
    keep_t0_latents : Int32,
    hide_t0_caption_keys : Bool,
    t1_current_text_kv : Bool,
    budget_mib : Int64,
    plan_only : Bool

  def sha256(path : String) : String
    digest = Digest::SHA256.new
    File.open(path) do |io|
      buffer = Bytes.new(1024 * 1024)
      while (count = io.read(buffer)) > 0
        digest.update(buffer[0, count])
      end
    end
    digest.hexfinal
  end

  def read_u32_le(bytes : Bytes, element : Int64) : UInt32
    offset = (element * sizeof(Float32)).to_i
    bytes[offset].to_u32 |
      (bytes[offset + 1].to_u32 << 8) |
      (bytes[offset + 2].to_u32 << 16) |
      (bytes[offset + 3].to_u32 << 24)
  end

  def authenticate_baseline!(receipt_path : String, receipt_sha : String,
                             bank_path : String, bank_sha : String,
                             inputs_sha : String) : NamedTuple(path: String, contents: String)
    receipt_contents = File.read(receipt_path)
    unless Digest::SHA256.hexdigest(receipt_contents) == receipt_sha && receipt_sha == BASELINE_RECEIPT_SHA
      raise ArgumentError.new("saved native baseline receipt identity mismatch")
    end
    receipt = JSON.parse(receipt_contents)
    diagnostics = receipt["diagnostics"]
    unless receipt["schema"].as_s == "cogni_ml.hy_nomas_native_continuation/v1" &&
           receipt["upstream_revision"].as_s == HyNomasGeneration::TimelineInputs::PINNED_REVISION &&
           receipt["inputs_sha256"].as_s == inputs_sha && receipt["segments"].as_i == 2 &&
           receipt["clean_latent_frames"].as_i == 60 &&
           receipt["latent_bank_shape"].as_a.map(&.as_i) == [1, 32, 180, 9, 16] &&
           receipt["source_bf16"].as_bool && receipt["reference_initialization"].as_bool &&
           diagnostics["latent_bank_sha256"].as_s == bank_sha && bank_sha == BASELINE_BANK_SHA
      raise ArgumentError.new("saved baseline is not the authenticated source-BF16 two-segment bank")
    end
    unless File.file?(bank_path) && !File.symlink?(bank_path) && File.size(bank_path) == BASELINE_BANK_BYTES
      raise ArgumentError.new("saved baseline latent-bank extent or file type mismatch")
    end
    bank_contents = File.read(bank_path)
    unless bank_contents.bytesize == BASELINE_BANK_BYTES && Digest::SHA256.hexdigest(bank_contents) == bank_sha
      raise ArgumentError.new("saved baseline consumed latent-bank bytes or identity mismatch")
    end
    {path: bank_path, contents: bank_contents}
  end

  # Check the authenticated T0 representation without transforming it on the
  # CPU. The actual channel/frame gather and F32-to-BF16 conversion are queued
  # on Metal with the DiT work.
  def validate_t0_bf16_expansion!(baseline : Bytes) : Nil
    unless baseline.size == BASELINE_BANK_BYTES
      raise ArgumentError.new("saved T0 source has the wrong extent")
    end
    channel_elements = BANK_FRAMES.to_i64 * PIXELS_PER_FRAME
    (CHANNELS.to_i64 * LATENT_FRAMES * PIXELS_PER_FRAME).times do |index|
      channel = index // (LATENT_FRAMES.to_i64 * PIXELS_PER_FRAME)
      within_channel = index % (LATENT_FRAMES.to_i64 * PIXELS_PER_FRAME)
      bits = read_u32_le(baseline, channel * channel_elements + within_channel)
      unless bits.unsafe_as(Float32).finite? && (bits & 0x0000_ffff_u32) == 0
        raise ArgumentError.new("saved T0 latent is not a finite BF16-expanded F32 value")
      end
    end
  end

  def logical_prefix_digests(source : Bytes) : NamedTuple(t0_sha256: String, t1_sha256: String, prefix60_sha256: String)
    unless source.size == BASELINE_BANK_BYTES
      raise ArgumentError.new("latent bank has the wrong extent for T0/T1 digesting")
    end
    t0_digest = Digest::SHA256.new
    t1_digest = Digest::SHA256.new
    prefix_digest = Digest::SHA256.new
    channel_bytes = BANK_FRAMES.to_i64 * PIXELS_PER_FRAME * sizeof(Float32)
    t0_channel_bytes = LATENT_FRAMES.to_i64 * PIXELS_PER_FRAME * sizeof(Float32)
    t1_start_bytes = LATENT_FRAMES.to_i64 * PIXELS_PER_FRAME * sizeof(Float32)
    CHANNELS.times do |channel|
      base_channel = channel.to_i64 * channel_bytes
      t0_slice = Bytes.new(source.to_unsafe + base_channel, t0_channel_bytes.to_i32)
      t1_slice = Bytes.new(source.to_unsafe + base_channel + t1_start_bytes, t0_channel_bytes.to_i32)
      t0_digest.update(t0_slice)
      t1_digest.update(t1_slice)
      prefix_digest.update(t0_slice)
      prefix_digest.update(t1_slice)
    end
    {t0_sha256: t0_digest.hexfinal, t1_sha256: t1_digest.hexfinal,
     prefix60_sha256: prefix_digest.hexfinal}
  end

  def output_checks!(bank_path : String, timeline : ML::MetalBuffer,
                     baseline : Bytes, keep_t0_latents : Int32,
                     hide_t0_caption_keys : Bool = false) : NamedTuple(
    t0_sha256_matches_baseline: Bool,
    t1_sha256_matches_baseline: Bool,
    t0_sha256: String,
    t1_sha256: String,
    baseline_t1_sha256: String,
    prefix60_sha256: String)
    output = Bytes.new(timeline.contents.as(UInt8*), timeline.size.to_i32)
    actual = logical_prefix_digests(output)
    expected = logical_prefix_digests(baseline)
    raise ArgumentError.new("native output latent bank was not written") unless File.file?(bank_path)
    t0_matches = actual[:t0_sha256] == expected[:t0_sha256]
    raise ArgumentError.new("native DiT changed the authenticated clean T0 prefix") unless t0_matches
    t1_matches = actual[:t1_sha256] == expected[:t1_sha256]
    if keep_t0_latents == 30 && !hide_t0_caption_keys && !t1_matches
      raise ArgumentError.new("all-visible T1 A/A latent digest differs from the saved native baseline")
    end
    {
      t0_sha256_matches_baseline: t0_matches,
      t1_sha256_matches_baseline: t1_matches,
      t0_sha256:                  actual[:t0_sha256],
      t1_sha256:                  actual[:t1_sha256],
      baseline_t1_sha256:         expected[:t1_sha256],
      prefix60_sha256:            actual[:prefix60_sha256],
    }
  end

  def parse_args : Args
    inputs_path = INPUTS_PATH
    inputs_sha = INPUTS_SHA
    receipt_path = BASELINE_RECEIPT
    receipt_sha = BASELINE_RECEIPT_SHA
    bank_path = BASELINE_BANK
    bank_sha = BASELINE_BANK_SHA
    output_dir = ""
    keep_t0_latents = 0
    hide_t0_caption_keys = false
    t1_current_text_kv = false
    budget_mib = 30_720_i64
    plan_only = false
    OptionParser.parse do |p|
      p.banner = "Authenticated native T0 history-key visibility diagnostic; two segments, four Euler steps"
      p.on("--inputs PATH", "Authenticated continuation manifest") { |v| inputs_path = v }
      p.on("--inputs-sha SHA", "Trusted continuation manifest SHA-256") { |v| inputs_sha = v }
      p.on("--baseline-receipt PATH", "Saved native baseline receipt") { |v| receipt_path = v }
      p.on("--baseline-receipt-sha SHA", "Trusted saved baseline receipt SHA-256") { |v| receipt_sha = v }
      p.on("--baseline-bank PATH", "Saved native full latent bank") { |v| bank_path = v }
      p.on("--baseline-bank-sha SHA", "Trusted saved baseline bank SHA-256") { |v| bank_sha = v }
      p.on("--keep-t0-latents N", "30 for all-visible masked-workspace A/A, or 10 for the intervention") { |v| keep_t0_latents = v.to_i }
      p.on("--hide-t0-caption-keys", "Mask T0 caption keys for the T1 reader while keeping all 30 T0 video latents visible") { hide_t0_caption_keys = true }
      p.on("--t1-current-text-kv", "Expose current T1 text K/V after hiding T0 caption keys") { t1_current_text_kv = true }
      p.on("--output-dir PATH", "Existing empty output directory on /opt/mldata_ssd") { |v| output_dir = v }
      p.on("--budget-mib N", "Named buffer cap; use with the external 10% GPU memory-floor guard") { |v| budget_mib = v.to_i64 }
      p.on("--plan-only", "Authenticate inputs and emit CPU-only memory plan") { plan_only = true }
    end
    unless {10, 30}.includes?(keep_t0_latents)
      raise ArgumentError.new("--keep-t0-latents must be 10 or 30")
    end
    HyNomasSequence.validate_t0_history_visibility!(keep_t0_latents, hide_t0_caption_keys,
      t1_current_text_kv)
    unless (18_432..49_152).includes?(budget_mib)
      raise ArgumentError.new("named buffer budget must be within 18..48 GiB")
    end
    raise ArgumentError.new("--output-dir is required unless --plan-only is used") if output_dir.empty? && !plan_only
    Args.new(inputs_path, inputs_sha, receipt_path, receipt_sha, bank_path, bank_sha,
      output_dir, keep_t0_latents, hide_t0_caption_keys, t1_current_text_kv,
      budget_mib, plan_only)
  end

  def run : Nil
    request_started = Time.monotonic
    args = parse_args
    inputs = HyNomasGeneration::TimelineInputs.new(File.realpath(args.inputs_path), args.inputs_sha)
    inputs.verify_payload!
    inputs.base.file.verify_payload!
    HyNomasSequence.validate_shared_scene_prompt!(inputs.base.prompt)
    baseline_source = authenticate_baseline!(File.realpath(args.baseline_receipt_path), args.baseline_receipt_sha,
      File.realpath(args.baseline_bank_path), args.baseline_bank_sha, inputs.file.sha256)
    baseline_bank = baseline_source[:path]
    baseline_contents = baseline_source[:contents]
    baseline = baseline_contents.to_slice
    unless baseline.size == BASELINE_BANK_BYTES
      raise ArgumentError.new("saved native baseline bank extent changed after authentication")
    end
    validate_t0_bf16_expansion!(baseline)
    baseline_t0_t1_digests = logical_prefix_digests(baseline)
    admission = HyNomasGeneration.authenticate_bf16_index
    HyNomasSequence.t0_history_attention_profiles(inputs.base.valid_text_tokens, args.keep_t0_latents,
      t0_hide_caption_keys: args.hide_t0_caption_keys,
      t1_current_text_kv: args.t1_current_text_kv)
    dit_plan = HyNomasSequence.peak_named_bytes(inputs, 2,
      reference_latent_bytes: HyNomasSequence::REFERENCE_LATENT_BYTES,
      source_bf16: true, t0_key_visible_latents: args.keep_t0_latents,
      t0_hide_caption_keys: args.hide_t0_caption_keys,
      t1_current_text_kv: args.t1_current_text_kv)
    probe_plan = dit_plan + PROBE_GPU_EXTRA_BYTES
    steady = dit_plan - HyNomasSequence.old_attention_bytes(inputs.base)
    budget = args.budget_mib * 1024 * 1024
    raise ArgumentError.new("native T0 history probe named plan exceeds its budget") if probe_plan > budget
    HyNomasGeneration.emit("portrait_t0_history_memory_plan",
      segments: 2, steps_per_segment: 4, keep_t0_latents: args.keep_t0_latents,
      t0_hide_caption_keys: args.hide_t0_caption_keys,
      t1_current_text_kv: args.t1_current_text_kv,
      dit_peak_named_bytes: dit_plan, probe_peak_named_bytes: probe_plan,
      dit_steady_named_bytes: steady,
      baseline_f32_bank_buffer_bytes: BASELINE_BANK_BYTES,
      compact_t0_f32_buffer_bytes: T0_BYTES,
      named_budget_bytes: budget, reserve_pct: 10,
      mask_path: if args.t1_current_text_kv
        "masked T1 workspace with current text K/V keys"
      else
        "same-size T1 masked workspace in both arms"
      end,
      t0_writer: "unmasked", t1_key_rows: args.t1_current_text_kv ? 8_640 + 2 * inputs.base.valid_text_tokens : 8_640 + inputs.base.valid_text_tokens,
      t1_query_rows: 4_320 + inputs.base.valid_text_tokens,
      hidden_t0_video_key_rows: args.keep_t0_latents == 10 ? 2_880 : 0,
      visible_t0_video_key_rows: args.keep_t0_latents == 10 ? 1_440 : 4_320,
      hidden_t0_caption_key_rows: args.hide_t0_caption_keys ? inputs.base.valid_text_tokens : 0,
      retained_t0_text_key_rows: args.hide_t0_caption_keys ? 0 : inputs.base.valid_text_tokens,
      t1_current_text_key_rows: args.t1_current_text_kv ? inputs.base.valid_text_tokens : 0,
      t1_current_video_key_rows: 4_320,
      t1_total_key_rows: args.t1_current_text_kv ? 8_640 + 2 * inputs.base.valid_text_tokens : 8_640 + inputs.base.valid_text_tokens,
      baseline_receipt_sha256: args.baseline_receipt_sha,
      baseline_bank_sha256: args.baseline_bank_sha,
      inputs_sha256: inputs.file.sha256,
      qwen_payload_sha256: inputs.base.file.document["payload_sha256"].as_s,
      source_bf16: true, full_upstream_video_parity: false)
    exit if args.plan_only
    unless Path[args.output_dir].absolute? && File.directory?(args.output_dir) && Dir.empty?(args.output_dir)
      raise ArgumentError.new("output directory must be an existing empty absolute directory")
    end
    output_dir = File.realpath(args.output_dir)
    unless output_dir.starts_with?(File.realpath("/opt/mldata_ssd") + "/")
      raise ArgumentError.new("native T0 history outputs must be on /opt/mldata_ssd")
    end

    timeline_arena = ML::Video::Arena.new
    current_arena = ML::Video::Arena.new
    reference_arena = ML::Video::Arena.new
    dit = nil.as(HyNomasSequence::DiT?)
    chain = nil.as(ML::Metal::MPSCommandChain?)
    prefix_plan = nil.as(ML::Video::HyNomasTimelineOps::PrefixPlan?)
    request_error : Exception? = nil
    cleanup_errors = [] of String
    diagnostics = {} of String => String
    metrics = {} of String => Int64
    timings = {} of String => Float64
    checks = nil.as(NamedTuple(t0_sha256_matches_baseline: Bool,
      t1_sha256_matches_baseline: Bool, t0_sha256: String,
      t1_sha256: String, baseline_t1_sha256: String,
      prefix60_sha256: String)?)
    begin
      timeline = inputs.upload!("input.latent_bank", timeline_arena)
      latent = current_arena.alloc("input.latent", 32 * 30 * 9 * 16)
      baseline_f32 = reference_arena.alloc("probe.baseline_f32_bank", (BASELINE_BANK_BYTES // sizeof(Float32)).to_i32)
      compact_t0_f32 = reference_arena.alloc("probe.compact_t0_f32", (T0_BYTES // sizeof(Float32)).to_i32)
      reference = reference_arena.alloc_weight("t0.reference_latent", 32 * 30 * 9 * 16,
        ML::Video::WeightDType::BF16)
      baseline_f32.write_bytes(baseline.to_unsafe, baseline.size)
      prefix_geometry = ML::Video::HyNomasTimelineOps::PrefixGeometry.new(
        channels: CHANNELS, total_frames: BANK_FRAMES, prefix_frames: LATENT_FRAMES,
        height: 9, width: 16)
      prefix_plan = ML::Video::HyNomasTimelineOps::PrefixPlan.new(prefix_geometry,
        source: baseline_f32, destination: compact_t0_f32)
      bf16_pipeline = ML::Video::PixelUMMGeneration.pipeline("pixelumm_generation_round_f32_to_bf16")
      setup_start = Time.monotonic
      dit = HyNomasSequence::DiT.new(inputs, timeline, latent, budget, 2, admission,
        reference_latent: reference, source_bf16: true,
        t0_key_visible_latents: args.keep_t0_latents,
        t0_hide_caption_keys: args.hide_t0_caption_keys,
        t1_current_text_kv: args.t1_current_text_kv)
      raise "native T0 history preflight/allocation admission disagree" unless dit.named_bytes == steady
      timings["dit_setup_seconds"] = (Time.monotonic - setup_start).total_seconds
      chain = ML::Metal::MPSCommandChain.new
      allocations = ML::MetalBuffer.hy_text_probe_allocations
      readbacks = ML::MetalBuffer.hy_text_probe_readbacks
      run_start = Time.monotonic
      prefix_plan.not_nil!.encode!(chain)
      chain.encoder.set_pipeline(bf16_pipeline)
        .set_buffer(compact_t0_f32, 0)
        .set_buffer(reference, 1, ML::Metal::BufferAccess::Write)
        .set_value((reference.size // sizeof(UInt16)).to_u32, 2)
        .dispatch_1d((reference.size // sizeof(UInt16)).to_i32)
      dit.encode(chain)
      metrics["chain_preterminal_allocations"] = ML::MetalBuffer.hy_text_probe_allocations - allocations
      metrics["chain_preterminal_readbacks"] = ML::MetalBuffer.hy_text_probe_readbacks - readbacks
      metrics["chain_preterminal_waits"] = chain.wait_count.to_i64
      unless metrics["chain_preterminal_allocations"] == 0 && metrics["chain_preterminal_readbacks"] == 0 && chain.wait_count == 0
        raise "native T0 history input preparation or DiT encode allocated, read back, or waited before terminal"
      end
      chain.commit_and_wait
      timings["dit_wall_seconds"] = (Time.monotonic - run_start).total_seconds
      timings["dit_gpu_seconds"] = chain.gpu_elapsed_seconds
      metrics["dit_segments"] = chain.segment_count.to_i64
      metrics["dit_terminal_waits"] = chain.wait_count.to_i64
      raise "native T0 history probe requires one terminal wait" unless chain.wait_count == 1 && chain.completed?

      bank_path = File.join(output_dir, "latent-bank.f32le")
      HyNomasGeneration.write_fresh_output(bank_path) do |io|
        io.write(Bytes.new(timeline.contents.as(UInt8*), timeline.size.to_i32))
      end
      diagnostics["latent_bank"] = bank_path
      diagnostics["latent_bank_sha256"] = sha256(bank_path)
      checks = output_checks!(bank_path, timeline, baseline, args.keep_t0_latents,
        args.hide_t0_caption_keys)
      diagnostics["t0_sha256"] = checks.not_nil![:t0_sha256]
      diagnostics["t1_sha256"] = checks.not_nil![:t1_sha256]
      diagnostics["prefix60_sha256"] = checks.not_nil![:prefix60_sha256]
      diagnostics["baseline_t0_sha256"] = baseline_t0_t1_digests[:t0_sha256]
      diagnostics["baseline_t1_sha256"] = baseline_t0_t1_digests[:t1_sha256]
      timings["request_seconds"] = (Time.monotonic - request_started).total_seconds
    rescue ex
      request_error = ex
    ensure
      may_release = true
      chain.try do |active_chain|
        unless active_chain.drained?
          begin
            active_chain.commit_and_wait
          rescue ex
            cleanup_errors << "terminal wait: #{ex.class}: #{ex.message}"
          end
        end
        may_release = active_chain.drained?
        if may_release && active_chain.wait_count == 1
          prefix_plan.try do |plan|
            if plan.encoded? && !plan.released?
              begin
                plan.release_after_drain!(active_chain)
              rescue ex
                cleanup_errors << "prefix plan release: #{ex.class}: #{ex.message}"
              end
            end
          end
        end
        unless may_release
          cleanup_errors << "terminal wait returned without proving GPU drain"
          HyNomasSequence.retain_in_flight do
            inputs.object_id
            dit.try(&.object_id)
            prefix_plan.try(&.object_id)
            active_chain.object_id
            reference_arena.object_id
            current_arena.object_id
            timeline_arena.object_id
            nil
          end
        end
      end
      actions = [] of Proc(Nil)
      if may_release
        dit.try { |owner| actions << -> { owner.release } }
        chain.try { |owner| actions << -> { owner.release } }
        actions << -> { reference_arena.release }
        actions << -> { current_arena.release }
        actions << -> { timeline_arena.release }
      end
      cleanup_errors.concat(HyNomasSequence.release_actions(actions))
    end
    cleanup_errors.each { |message| STDERR.puts("portrait T0 history cleanup: #{message}") }
    raise request_error.not_nil! if request_error
    raise "native T0 history cleanup failed: #{cleanup_errors.join("; ")}" unless cleanup_errors.empty?

    report = {
      schema: "cogni_ml.hy_nomas_portrait_t0_history_probe/v1",
      inputs: inputs.file.path, inputs_sha256: inputs.file.sha256,
      qwen_manifest_sha256: inputs.base.file.sha256,
      qwen_payload_sha256: inputs.base.file.document["payload_sha256"].as_s,
      baseline_receipt: args.baseline_receipt_path,
      baseline_receipt_sha256: args.baseline_receipt_sha,
      baseline_latent_bank: baseline_bank,
      baseline_latent_bank_sha256: args.baseline_bank_sha,
      keep_t0_video_latents: args.keep_t0_latents,
      t0_hide_caption_keys: args.hide_t0_caption_keys,
      t1_current_text_kv: args.t1_current_text_kv,
      arm: if args.hide_t0_caption_keys
        if args.t1_current_text_kv
          "all_t0_video_visible_t0_caption_keys_hidden_t1_current_text_kv"
        else
          "all_t0_video_visible_t0_caption_keys_hidden"
        end
      elsif args.keep_t0_latents == 30
        "all_visible_masked_workspace_aa"
      else
        "first_20_t0_video_latents_hidden"
      end,
      upstream_revision: HyNomasGeneration::TimelineInputs::PINNED_REVISION,
      weight_index_sha256: admission.sha256,
      weight_source_sha256: admission.source_sha256,
      globals_bf16_sha256: admission.globals_bf16_sha256,
      admitted_cache_payload_bytes: admission.payload_bytes,
      source_bf16: true, segments: 2, steps_per_segment: 4,
      t0_writer_attention: "unmasked nil profile",
      t1_reader_attention: if args.t1_current_text_kv
        "T0 caption mask; current T1 text K/V keys included"
      else
        "masked profile; same workspace geometry in both arms"
      end,
      t0_video_key_rows: 4_320,
      hidden_t0_video_key_rows: args.keep_t0_latents == 10 ? 2_880 : 0,
      visible_t0_video_key_rows: args.keep_t0_latents == 10 ? 1_440 : 4_320,
      t0_text_key_rows: inputs.base.valid_text_tokens,
      hidden_t0_caption_key_rows: args.hide_t0_caption_keys ? inputs.base.valid_text_tokens : 0,
      retained_t0_text_key_rows: args.hide_t0_caption_keys ? 0 : inputs.base.valid_text_tokens,
      t1_current_text_key_rows: args.t1_current_text_kv ? inputs.base.valid_text_tokens : 0,
      t1_video_key_rows: 4_320,
      t1_total_key_rows: args.t1_current_text_kv ? 8_640 + 2 * inputs.base.valid_text_tokens : 8_640 + inputs.base.valid_text_tokens,
      t1_query_rows: 4_320 + inputs.base.valid_text_tokens,
      noise_scope: "same authenticated input latent/noise/RoPE banks; no edits or reseeding",
      diagnostic_scope: "native DiT only; no DCAE encoder, PixelUMM decoder, or upstream video parity",
      metrics: metrics, timings: timings,
      output: diagnostics,
      checks: checks,
      dit_peak_named_bytes: dit_plan, probe_peak_named_bytes: probe_plan,
      dit_steady_named_bytes: steady,
      baseline_f32_bank_buffer_bytes: BASELINE_BANK_BYTES,
      compact_t0_f32_buffer_bytes: T0_BYTES,
      named_budget_bytes: budget, full_upstream_video_parity: false,
    }
    HyNomasGeneration.write_fresh_output(File.join(output_dir, "receipt.json")) do |io|
      io.puts(report.to_pretty_json)
    end
    HyNomasGeneration.emit("portrait_t0_history_completed", output: diagnostics,
      keep_t0_video_latents: args.keep_t0_latents,
      t0_hide_caption_keys: args.hide_t0_caption_keys,
      t1_current_text_kv: args.t1_current_text_kv, metrics: metrics)
  end
end

HyNomasPortraitT0HistoryProbe.run
