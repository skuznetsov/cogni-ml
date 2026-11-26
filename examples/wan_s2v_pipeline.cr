#!/usr/bin/env crystal
# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
# Apache-2.0. Native Wan S2V pipeline orchestration.
#
# `--plan` is setup-only and does not initialize Metal. `--run` uses one
# serial MPS command chain and event-ordered checkpoint reloads. It remains
# fail-closed until its exact evidence receipt admits the selected mode.

require "json"
require "digest/sha256"
require "option_parser"
require "set"
require "c/unistd"
require "c/sys/stat"
require "../src/ml/video/wan_t5"
require "../src/ml/video/wan_audio_encoder"
require "../src/ml/video/wan_s2v_audio"
require "../src/ml/video/wan_s2v_transformer"
require "../src/ml/video/wan_s2v_memory"
require "../src/ml/video/wan_vae"
require "../src/ml/video/wan_unipc"
require "../src/ml/video/metal_mps_command_chain"
require "../src/ml/video/metal_stage_io"

private class WanS2VCLI
  property run = false
  property probe_dit_block = false
  property memory_plan = false
  property mode_selected = false
  property help = false
  property inputs = ""
  property checkpoint_root = ""
  property output = ""
  property admission_receipt = ""
  property allow_recorded_numerical_drift = false
  property validate_receipt = ""
  property budget_mib = 0_i64
  property reserve_mib = 0_i64
  property steps = 40
  property requests = 1
  property projection_backend = "scalar"
  property attention_backend = "tiled"

  def parse!(args : Array(String)) : Nil
    OptionParser.parse(args) do |parser|
      parser.banner = "Usage: crystal run examples/wan_s2v_pipeline.cr -- [--plan | --run] [options]"
      parser.on("--plan", "Print a device-free exact geometry/lifetime plan (default)") do
        raise ArgumentError.new("Choose only one of --plan and --run") if @mode_selected
        @mode_selected = true
        @run = false
      end
      parser.on("--run", "Execute the opt-in native pipeline, subject to its admission receipt") do
        raise ArgumentError.new("Choose only one of --plan and --run") if @mode_selected
        @mode_selected = true
        @run = true
      end
      parser.on("--probe-dit-block", "Run one real-width block 0 + audio injector 0 as a cost diagnostic") do
        raise ArgumentError.new("Choose only one execution mode") if @mode_selected
        @mode_selected = true
        @probe_dit_block = true
      end
      parser.on("--memory-plan", "Print a header-only, device-free aggregate residency plan") do
        raise ArgumentError.new("Choose only one execution mode") if @mode_selected
        @mode_selected = true
        @memory_plan = true
      end
      parser.on("--inputs=DIR", "Input manifest directory produced by wan_s2v_inputs.py") { |v| @inputs = v }
      parser.on("--checkpoint-root=DIR", "Local Wan checkpoint root") { |v| @checkpoint_root = v }
      parser.on("--output=FILE", "Final raw F32 BCTHW output path") { |v| @output = v }
      parser.on("--admission-receipt=FILE", "Root-issued no-new-sync qualification receipt") { |v| @admission_receipt = v }
      parser.on("--allow-recorded-numerical-drift", "Allow only the two exact recorded audio/real-stack drift ledgers") { @allow_recorded_numerical_drift = true }
      parser.on("--validate-receipt=FILE", "Validate an admission receipt without initializing Metal") { |v| @validate_receipt = v }
      parser.on("--budget-mib=N", "Maximum request residency budget in MiB") { |v| @budget_mib = v.to_i64 }
      parser.on("--reserve-mib=N", "Caller reserve in MiB, including unrelated residency") { |v| @reserve_mib = v.to_i64 }
      parser.on("--steps=N", "UniPC steps; production requires exactly 40") { |v| @steps = v.to_i }
      parser.on("--requests=N", "Requests in one process, retaining pools and model plans (1 or 2)") { |v| @requests = v.to_i }
      parser.on("--projection-backend=NAME", "Projection backend for the DIT probe or experimental run: scalar or mps") { |v| @projection_backend = v }
      parser.on("--attention-backend=NAME", "Self-attention backend for the DIT probe or experimental run: tiled or mps-chunked") { |v| @attention_backend = v }
      parser.on("-h", "--help", "Show this help") { @help = true }
      parser.invalid_option do |option|
        raise ArgumentError.new("Unknown option #{option}")
      end
    end
    raise ArgumentError.new("Unexpected positional arguments: #{args.join(" ")}") unless args.empty?
    raise ArgumentError.new("--steps must be exactly 40 for the production request") unless steps == 40
    raise ArgumentError.new("--requests must be 1 or 2") unless {1, 2}.includes?(requests)
    raise ArgumentError.new("--projection-backend must be scalar or mps") unless {"scalar", "mps"}.includes?(projection_backend)
    raise ArgumentError.new("--attention-backend must be tiled or mps-chunked") unless {"tiled", "mps-chunked"}.includes?(attention_backend)
    if !validate_receipt.empty?
      raise ArgumentError.new("--validate-receipt cannot be combined with an execution mode") if mode_selected
      raise ArgumentError.new("--validate-receipt requires --inputs and --checkpoint-root") if inputs.empty? || checkpoint_root.empty?
    end
    if memory_plan
      raise ArgumentError.new("--memory-plan requires --checkpoint-root") if checkpoint_root.empty?
      raise ArgumentError.new("--memory-plan requires a positive --budget-mib and nonnegative --reserve-mib") unless budget_mib > 0 && reserve_mib >= 0
    elsif probe_dit_block
      raise ArgumentError.new("--probe-dit-block requires --checkpoint-root") if checkpoint_root.empty?
      raise ArgumentError.new("--probe-dit-block requires a positive --budget-mib and nonnegative --reserve-mib") unless budget_mib > 0 && reserve_mib >= 0
    elsif run
      {inputs, checkpoint_root, output, admission_receipt}.each do |value|
        raise ArgumentError.new("--run requires --inputs, --checkpoint-root, --output, and --admission-receipt") if value.empty?
      end
      raise ArgumentError.new("--run requires a positive --budget-mib and nonnegative --reserve-mib") unless budget_mib > 0 && reserve_mib >= 0
    end
  end
end

private struct WanS2VInputTensor
  getter path : String
  getter dtype : String
  getter shape : Array(Int64)
  getter bytes : Int64
  getter sha256 : String

  def initialize(root : String, value : JSON::Any)
    object = value.as_h
    name = object["file"].as_s
    raise ArgumentError.new("Input tensor path must be a basename") unless File.basename(name) == name
    @path = File.join(root, name)
    @dtype = object["dtype"].as_s
    @shape = object["shape"].as_a.map(&.as_i64)
    @bytes = object["bytes"].as_i64
    @sha256 = object["sha256"].as_s
  end

  def validate! : Nil
    expected_type_bytes = case dtype
                          when "F32_LE" then 4_i64
                          when "I32_LE" then 4_i64
                          else               raise ArgumentError.new("Unsupported input dtype #{dtype}")
                          end
    count = shape.reduce(1_i64) { |product, extent|
      raise ArgumentError.new("Invalid input tensor shape") unless extent > 0 && product <= Int64::MAX // extent
      product * extent
    }
    unless count <= Int64::MAX // expected_type_bytes && count * expected_type_bytes == bytes
      raise ArgumentError.new("Input tensor byte count does not match shape")
    end
    raise ArgumentError.new("Missing input tensor #{path}") unless File.file?(path) && File.size(path) == bytes
  end
end

private class WanS2VInputSet
  getter root : String
  getter manifest : JSON::Any

  def initialize(path : String)
    @root = File.expand_path(path)
    manifest_path = File.join(@root, "manifest.json")
    raise ArgumentError.new("Missing input manifest #{manifest_path}") unless File.file?(manifest_path)
    @manifest = JSON.parse(File.read(manifest_path))
    raise ArgumentError.new("Unsupported input manifest schema") unless @manifest["schema"].as_s == "wan_s2v_native_inputs_v1"
    tensors = @manifest["tensors"].as_h
    required = Set{"reference.f32", "audio.f32", "positive_ids.i32", "positive_mask.i32",
                   "negative_ids.i32", "negative_mask.i32", "noise.f32"}
    raise ArgumentError.new("Input manifest tensor set differs from the runner contract") unless tensors.keys.to_set == required
    expected_shapes = {
      "reference.f32"     => [1_i64, 3, 1, 512, 832],
      "positive_ids.i32"  => [512_i64],
      "positive_mask.i32" => [512_i64],
      "negative_ids.i32"  => [512_i64],
      "negative_mask.i32" => [512_i64],
      "noise.f32"         => [1_i64, 16, 21, 64, 104],
    }
    tensors.each do |name, value|
      tensor = WanS2VInputTensor.new(@root, value)
      tensor.validate!
      if expected = expected_shapes[name]?
        raise ArgumentError.new("Input shape mismatch for #{name}") unless tensor.shape == expected
      elsif name == "audio.f32"
        audio = @manifest["audio"]
        unless tensor.dtype == "F32_LE" && tensor.shape.size == 1 &&
               audio["sample_rate"].as_i == 16_000 && audio["sample_count"].as_i == tensor.shape[0] &&
               audio["sample_count"].as_i == 80_000 && audio["requested_seconds"].as_f == 5.0 &&
               audio["source_sample_count"].as_i >= 80_000 &&
               audio["crop"].as_s == "leading_samples_after_16khz_mono_resample; no padding" &&
               audio["source_file_sha256"].as_s.matches?(/\A[0-9a-f]{64}\z/)
          raise ArgumentError.new("Audio tensor metadata does not match the runner contract")
        end
      else
        raise ArgumentError.new("Unexpected input tensor #{name}")
      end
    end
    geometry = @manifest["geometry"]
    expected = {"canvas_height" => 512_i64, "canvas_width" => 832_i64,
                "deliver_frames" => 80_i64, "fps" => 16_i64,
                "internal_frames" => 84_i64, "latent_frames" => 21_i64,
                "latent_height" => 64_i64, "latent_width" => 104_i64,
                "text_tokens" => 512_i64}
    expected.each do |key, value|
      raise ArgumentError.new("Input geometry mismatch for #{key}") unless geometry[key].as_i64 == value
    end
  end

  def tensor(name : String) : WanS2VInputTensor
    WanS2VInputTensor.new(root, manifest["tensors"][name].not_nil!)
  end

  def int32_values(name : String) : Array(Int32)
    tensor_info = tensor(name)
    raise ArgumentError.new("Expected I32 input #{name}") unless tensor_info.dtype == "I32_LE"
    bytes = File.read(tensor_info.path).to_slice
    raise ArgumentError.new("Input tensor has invalid I32 bytes") unless bytes.size == tensor_info.bytes
    raise ArgumentError.new("Input checksum mismatch #{name}") unless Digest::SHA256.hexdigest(bytes) == tensor_info.sha256
    count = bytes.size // 4
    pointer = bytes.to_unsafe.as(Int32*)
    Array(Int32).new(count) { |i| pointer[i] }
  end

  # Validate tokenizer payloads before device initialization. This guards the
  # setup boundary independently of the upload helper that later repeats the
  # same checks before materializing T5 inputs.
  def validate_text_inputs! : Nil
    cfg = ML::Video::WanT5Config.new
    {
      {"positive_ids.i32", "positive_mask.i32"},
      {"negative_ids.i32", "negative_mask.i32"},
    }.each do |ids_name, mask_name|
      ids = int32_values(ids_name)
      mask = int32_values(mask_name)
      unless ids.size == cfg.rows && mask.size == cfg.rows &&
             ids.all? { |id| id >= 0 && id < cfg.vocab } &&
             mask.all? { |value| value == 0 || value == 1 } &&
             mask.any? { |value| value == 1 }
        raise ArgumentError.new("Invalid token IDs or mask in #{ids_name}/#{mask_name}")
      end
    end
  end

  # Reject non-finite CPU payloads before initializing Metal. The values are
  # later checksum-verified again by upload; this bounded scan does not retain
  # a second copy of the large initial-noise tensor.
  def validate_finite_inputs! : Nil
    {"reference.f32", "audio.f32", "noise.f32"}.each do |name|
      info = tensor(name)
      raise ArgumentError.new("Expected F32 input #{name}") unless info.dtype == "F32_LE"
      File.open(info.path, "rb") do |file|
        chunk = Bytes.new(1 << 20)
        carry = Bytes.new(4)
        carry_count = 0
        loop do
          chunk.copy_from(carry.to_unsafe, carry_count) if carry_count > 0
          count = file.read(chunk[carry_count..])
          break if count == 0
          total = carry_count + count
          complete = total - total % 4
          (0...complete).step(4) do |index|
            bits = chunk[index].to_u32 |
                   (chunk[index + 1].to_u32 << 8) |
                   (chunk[index + 2].to_u32 << 16) |
                   (chunk[index + 3].to_u32 << 24)
            if ((bits >> 23) & 0xff_u32) == 0xff_u32
              raise ArgumentError.new("F32 input contains a non-finite value: #{name}")
            end
          end
          carry_count = total - complete
          carry.copy_from(chunk.to_unsafe + complete, carry_count) if carry_count > 0
        end
        raise ArgumentError.new("Truncated F32 input #{name}") unless carry_count == 0
      end
    end
  end

  def upload(arena : ML::Video::Arena, name : String, buffer_name : String) : ML::MetalBuffer
    tensor_info = tensor(name)
    raise ArgumentError.new("Expected F32 input #{name}") unless tensor_info.dtype == "F32_LE"
    count = (tensor_info.bytes // 4).to_i32
    buffer = arena.alloc(buffer_name, count)
    File.open(tensor_info.path, "rb") do |file|
      bytes = Bytes.new(tensor_info.bytes.to_i32)
      file.read_fully(bytes)
      raise ArgumentError.new("Input checksum mismatch #{name}") unless Digest::SHA256.hexdigest(bytes) == tensor_info.sha256
      unless finite_f32_bytes?(bytes)
        raise ArgumentError.new("F32 input contains a non-finite value: #{name}")
      end
      buffer.write_bytes(bytes.to_unsafe, bytes.size)
    end
    buffer
  end

  private def finite_f32_bytes?(bytes : Bytes) : Bool
    return false if bytes.size % 4 != 0
    words = bytes.to_unsafe.as(UInt32*)
    (bytes.size // 4).times.all? do |index|
      ((words[index] >> 23) & 0xff_u32) != 0xff_u32
    end
  end
end

# A file-backed checkpoint phase whose payloads are staged directly into
# caller-owned no-copy pool views. BF16 tensors stay BF16 in the transformer;
# operators requiring F32 receive an exact GPU decode from a raw pool view.
# The Arena and source CheckpointFiles must remain alive until StageIO.finish!
# and the compute-chain terminal completion.
private class WanS2VStageWeights
  SOURCE = ML::Video::PreparedWeights::SOURCE

  getter arena : ML::Video::Arena
  getter plan : ML::Video::WeightPlan
  getter phase : String
  getter io_plan : ML::Video::StageIOPlan?

  @files : Array(ML::Video::CheckpointFile)
  @loads = [] of ML::Video::StageLoad
  @conversions = [] of Tuple(ML::Video::CheckpointTensor, ML::MetalBuffer, ML::MetalBuffer)
  @owned_names = [] of String
  @released = false
  @io_plan : ML::Video::StageIOPlan? = nil
  @plans = {} of UInt64 => ML::Video::StageIOPlan

  def initialize(paths : Array(String), bindings : Array(ML::Video::WeightBinding),
                 @arena : ML::Video::Arena, pool : ML::Video::MetalStagePool,
                 @phase : String, stage_io : ML::Video::StageIO,
                 budget_bytes : Int64, reserve_bytes : Int64,
                 resident_dtype : ML::Video::WeightDType,
                 require_all : Bool = false,
                 shared_pool_budget : Bool = false)
    raise ArgumentError.new("Empty Wan S2V checkpoint phase") if @phase.empty?
    @files = paths.map { |path| ML::Video::CheckpointFile.new(File.expand_path(path)) }
    @plan = begin
      # In unified-pool mode, the outer preflight owns the aggregate request
      # budget. WeightPlan still validates every binding and payload extent,
      # while this phase's page-rounded capacity is checked below.
      plan_reserve = shared_pool_budget ? 0_i64 : reserve_bytes
      ML::Video::WeightPlan.new(@files, bindings, budget_bytes,
        plan_reserve, resident_dtype, require_all)
    rescue ex
      @files.each(&.close)
      raise ex
    end
    begin
      if !shared_pool_budget && arena.resident_bytes > reserve_bytes
        raise ArgumentError.new("Checkpoint phase exceeds its caller reserve")
      end
      @plan.entries.each_with_index do |entry, index|
        info = entry.info
        entry.file.validate_tensor!(info)
        destination_bytes = info.n_elements * resident_dtype.bytes_per_element
        destination_view = pool.allocate!(@phase, "weights", destination_bytes)
        destination = arena.adopt_weight(entry.binding.target, destination_view, resident_dtype)
        @owned_names << entry.binding.target

        source = destination
        if info.dtype.bytes_per_element != resident_dtype.bytes_per_element ||
           (resident_dtype.f32? && !info.dtype.f32?)
          raw_name = "wan_stage_raw.#{@phase}.#{index}"
          # Arena owns raw storage as F32-sized words; the decode kernel reads
          # bytes using the checkpoint dtype. Odd BF16/F16 element counts need
          # word padding, but StageLoad must still read only the payload bytes.
          # VM pages are word-aligned, so this adds no page-rounded pool space.
          raw_bytes = ((info.data_bytes + sizeof(Float32) - 1) // sizeof(Float32)) * sizeof(Float32)
          raw_view = pool.allocate!(@phase, "raw", raw_bytes)
          source = arena.adopt(raw_name, raw_view)
          @owned_names << raw_name
          @conversions << {info, source, destination}
        end
        @loads << ML::Video::StageLoad.new(source,
          File.expand_path(entry.file.path), entry.file.data_offset + info.data_start,
          info.data_bytes)
      end
      io_plan = stage_io.plan(@loads)
      @io_plan = io_plan
      @plans[stage_io.object_id] = io_plan
    rescue ex
      @owned_names.reverse_each { |name| arena.remove(name) }
      @files.each(&.close)
      raise ex
    end
  end

  def encode_conversions(e : ML::Metal::ComputeEncoder) : Nil
    raise ArgumentError.new("Checkpoint stage weights have been released") if @released
    return if @conversions.empty?
    pipeline = ML::Metal::PipelineCache.get("video_weight_decode") do
      ML::Metal::ComputePipeline.new("video_weight_decode", SOURCE)
    end
    @conversions.each do |info, source, destination|
      count = info.n_elements.to_i32
      tag = case info.dtype
            in .f32?  then 0_u32
            in .bf16? then 1_u32
            in .f16?  then 2_u32
            end
      e.set_pipeline(pipeline).set_buffer(source, 0)
      e.set_buffer(destination, 1, ML::Metal::BufferAccess::Write)
      e.set_value(count.to_u32, 2).set_value(tag, 3).dispatch_1d(count)
    end
  end

  # Plans are immutable and may be created for each request before its first
  # StageIO begin. This preserves the one-chain no-host-wait ordering while
  # allowing retained pool views to be reloaded on a warm repeat.
  def plan_for(stage_io : ML::Video::StageIO) : ML::Video::StageIOPlan
    raise ArgumentError.new("Checkpoint stage weights have been released") if @released
    @plans[stage_io.object_id]? || begin
      plan = stage_io.plan(@loads)
      @plans[stage_io.object_id] = plan
      plan
    end
  end

  def validate_sources! : Nil
    @files.each(&.unchanged!)
    @plan.entries.each { |entry| entry.file.validate_tensor!(entry.info) }
  end

  # Call only after the terminal stage-I/O/compute completion. Pool views are
  # released by their Arena after every overlapping phase has also completed.
  def close_sources! : Nil
    return if @released
    @files.each(&.close)
    @released = true
  end
end

# Keep the complete native owner graph rooted before any GPU work can start.
# A failed terminal drain leaves this object in RUN_RESOURCES until process
# teardown; merely skipping explicit release would still allow GC finalizers
# to release buffers or graph plans while the request is in flight.
private class WanS2VRunResources
  property chain : ML::Metal::MPSCommandChain? = nil
  property request_io : ML::Video::StageIO? = nil
  property t5_plans : ML::Video::WanT5MPSPlans? = nil
  property xlsr_plans : ML::Video::WanAudioMPSPlans? = nil
  property normalizer : ML::Video::WanAudioMPSNormalizationPlan? = nil
  property gelu_plans : ML::Video::WanAudioMPSGeluPlans? = nil
  property position_plan : ML::Video::WanAudioMPSPositionPlan? = nil
  property linear_stage : ML::Video::WanS2VMPSLinearStage? = nil
  property positive_workspace : ML::Video::WanS2VMPSLinearWorkspace? = nil
  property negative_workspace : ML::Video::WanS2VMPSLinearWorkspace? = nil
  property attention_workspace : ML::Video::WanAttentionMPSChunkWorkspace? = nil
  getter extra_arenas = [] of ML::Video::Arena

  def initialize(@pool : ML::Video::MetalStagePool,
                 @stage_io : ML::Video::StageIO,
                 @state : ML::Video::Arena,
                 @constants : ML::Video::Arena,
                 @phase_weights : Hash(String, ML::Video::Arena),
                 @phase_scratch : Hash(String, ML::Video::Arena),
                 @stages : Hash(String, WanS2VStageWeights))
  end
end

private module WanS2VRunner
  RUN_RESOURCES   = [] of WanS2VRunResources
  CANVAS_HEIGHT   = 512
  CANVAS_WIDTH    = 832
  LATENT_HEIGHT   =  64
  LATENT_WIDTH    = 104
  INTERNAL_FRAMES =  84
  DIT_FRAMES      =  21
  DELIVER_FRAMES  =  80
  DELIVER_HEIGHT  = 480
  STEPS           =  40
  # Kept only for the reviewer-visible legacy staged path below.
  STAGED_TERMINAL_WAITS = 5
  RUN_RECEIPT_SCHEMA    = "wan_s2v_no_new_sync_admission_v1"
  AUDIO_DRIFT_LEDGER    = "/opt/mldata_ssd/cogni-ml/video-scene-20261019/evidence/wan-full/audio-conditioning-drift-ledger.json"
  AUDIO_DRIFT_GATE      = "audio-conditioning-full-output"
  STACK_DRIFT_LEDGER    = "/opt/mldata_ssd/cogni-ml/video-scene-20261019/evidence/wan-full/s2v-real-stack-drift-ledger.json"
  STACK_DRIFT_GATE      = "s2v-real-stack-full-depth"
  REQUIRED_GATES        = %w[t5_full_context audio_encoder_full causal_audio transformer_40_layers full_shape_plan_guards vae_decode_encode unipc short_pipeline]
  IMPLEMENTATION_FILES  = %w[
    examples/wan_s2v_pipeline.cr
    src/ml/video/metal_stage_pool.cr
    src/ml/video/metal_stage_io.cr
    src/ml/video/metal_stage_io.mm
    src/ml/video/metal_mps_command_chain.cr
    src/ml/video/metal_mps_command_chain.mm
    src/ml/video/wan_s2v_memory.cr
    src/ml/video/metal_ops.cr
    src/ml/metal/bridge.mm
    src/ml/video/checkpoint.cr
    src/ml/video/metal_linear_mps.cr
    src/ml/video/kernels/linear_mps.metal
    src/ml/video/kernels/common.metal
    src/ml/video/metal_attention.cr
    src/ml/video/metal_attention_mps.cr
    src/ml/video/wan_attention_mps_bridge.mm
    src/ml/video/kernels/attention_tiled.metal
    src/ml/video/wan_s2v_transformer.cr
    src/ml/video/wan_s2v_block.cr
    src/ml/video/kernels/wan_s2v_transformer.metal
    src/ml/video/kernels/wan_s2v.metal
    src/ml/video/wan_s2v_audio.cr
    src/ml/video/kernels/wan_s2v_audio.metal
    src/ml/video/wan_audio_encoder.cr
    src/ml/video/kernels/wan_audio_encoder.metal
    src/ml/video/wan_audio_mps_bridge.mm
    src/ml/video/wan_audio_mps_normalize_bridge.mm
    src/ml/video/wan_t5.cr
    src/ml/video/kernels/wan_t5.metal
    src/ml/video/wan_t5_mps_bridge.mm
    src/ml/video/wan_t5_attention_mps_bridge.mm
    src/ml/video/wan_t5_rms_mps_bridge.mm
    src/ml/video/wan_vae.cr
    src/ml/video/kernels/wan_vae.metal
    src/ml/video/wan_unipc.cr
    src/ml/video/kernels/wan_unipc.metal
    src/ml/video/kernels/weight_decode.metal
    src/ml/metal/dispatch.cr
    src/ml/metal/device.cr
    src/ml/gguf/safetensors.cr
    spec/wan_s2v_real_stack_spec.cr
    spec/support/wan_s2v_real_stack_reference.py
    examples/wan_s2v_inputs.py
  ]

  def self.transformer_config : ML::Video::WanS2VTransformerConfig
    ML::Video::WanS2VTransformerConfig.new(
      batches: 1, context_tokens: 512, dim: 5120, heads: 40, ffn_dim: 13824,
      layers: 40, video_frames: DIT_FRAMES, spatial_tokens: (LATENT_HEIGHT // 2) * (LATENT_WIDTH // 2),
      latent_height: LATENT_HEIGHT, latent_width: LATENT_WIDTH, text_dim: 4096,
      text_tokens: 512, freq_dim: 256, in_dim: 16, out_dim: 16,
      audio_dim: 1024, drop_motion_frames: true)
  end

  def self.print_plan(cli : WanS2VCLI) : Nil
    cfg = transformer_config
    sequence = cfg.sequence_tokens
    video_values = 1_i64 * 16 * DIT_FRAMES * LATENT_HEIGHT * LATENT_WIDTH
    token_values = cfg.rows.to_i64 * cfg.dim
    transformer_weights = 32_591_511_218_i64
    dit_hidden_bytes = token_values * 4
    plan = {
      schema: "wan_s2v_native_plan_v1",
      execution: "plan-only; no Metal device initialization, model load, or neural execution",
      config: {transformer_dim: cfg.dim, heads: cfg.heads, layers: cfg.layers,
               audio_injection_layers: cfg.injection_ids, context_tokens: cfg.text_tokens,
               canvas: [CANVAS_HEIGHT, CANVAS_WIDTH], latent_grid: [LATENT_HEIGHT, LATENT_WIDTH],
               predicted_latent_frames: DIT_FRAMES, sequence_tokens: sequence,
               noisy_tokens: cfg.video_tokens, reference_tokens: cfg.reference_tokens,
               framepack_tokens: cfg.framepack_tokens, first_request_no_pose: cfg.drop_motion_frames},
      temporal_recipe: {infer_internal_frames: INTERNAL_FRAMES, latent_frames: DIT_FRAMES,
                        prepend_reference_latents_for_decode: 1, decoded_raw_frames: 85,
                        upstream_window_offset: 4, upstream_window_frames: 81,
                        trailing_crop_frames: 1, delivered_frames: DELIVER_FRAMES, fps: 16},
      spatial_recipe: {vae_canvas: [CANVAS_HEIGHT, CANVAS_WIDTH], center_crop_top: 16,
                       center_crop_bottom: 16, delivered_height: DELIVER_HEIGHT, width: CANVAS_WIDTH},
      precision: {checkpoint_transformer_storage: "BF16", transformer_math: "F32",
                  T5_Wav2Vec_CausalAudio_VAE_math: "F32", input_noise: "CPU PyTorch Generator; differs from upstream MPS RNG"},
      stage_lifetimes: [
        {name: "T5", retain: ["positive_context", "negative_context"], intermediate_host_waits: 0},
        {name: "XLSR+causal_audio", retain: ["local_audio", "global_audio"], intermediate_host_waits: 0},
        {name: "VAE_reference", retain: ["reference_latent", "zero_condition"], intermediate_host_waits: 0},
        {name: "40-layer-DiT+UniPC", retain: ["final_noise_latent"], intermediate_host_waits: 0},
        {name: "VAE_decode+crop", retain: ["final_output"], terminal_host_waits: 1},
      ],
      boundaries: {planned_terminal_host_waits_per_request: 1,
                   internal_layer_or_operator_waits: 0, neural_activation_readbacks_between_stages: 0,
                   final_output_readback: "one final raw F32 export after decode completion",
                   run_admitted: false,
                   reason: "runtime execution requires a receipt for the assembled one-chain path and exact current source hashes"},
      memory_estimate: {dit_bf16_checkpoint_bytes: transformer_weights,
                        one_full_transformer_sequence_buffer_bytes: dit_hidden_bytes,
                        full_transformer_activation_sequence_bytes_each: dit_hidden_bytes,
                        values_note: "estimates exclude 40-block scratch, CFG, T5, audio, VAE caches/output, and external residency"},
      requested_budget_mib: cli.budget_mib, requested_reserve_mib: cli.reserve_mib,
    }
    puts plan.to_json
  end

  private def self.sha256(path : String) : String
    Digest::SHA256.hexdigest(File.read(path))
  end

  private def self.current_implementation_hashes : Hash(String, String)
    IMPLEMENTATION_FILES.to_h do |relative|
      path = File.expand_path(File.join(__DIR__, "..", relative))
      raise ArgumentError.new("Missing implementation source #{relative}") unless File.file?(path)
      {relative, sha256(path)}
    end
  end

  def self.validate_run_receipt!(path : String, cli : WanS2VCLI,
                                 input : WanS2VInputSet, root : String) : String
    raise ArgumentError.new("Admission receipt does not exist: #{path}") unless File.file?(path)
    json = JSON.parse(File.read(path))
    experimental = cli.allow_recorded_numerical_drift
    expected_status = experimental ? "experimental_run_admitted" : "qualified"
    unless json["schema"].as_s == RUN_RECEIPT_SCHEMA && json["status"].as_s == expected_status
      raise ArgumentError.new("--run rejected: admission receipt status must be #{expected_status}")
    end
    if experimental
      unless cli.projection_backend == "mps" && cli.attention_backend == "mps-chunked"
        raise ArgumentError.new("--run rejected: recorded drift ledgers admit only the prepared MPS projection and chunk512 attention backends")
      end
      backends = json["backends"]
      unless backends["projection"].as_s == "mps" &&
             backends["self_attention"].as_s == "mps-chunked"
        raise ArgumentError.new("--run rejected: receipt backend selection differs from the known numeric-drift ledgers")
      end
    end
    gates = json["gates"].as_h
    unless gates.keys.to_set == REQUIRED_GATES.to_set
      raise ArgumentError.new("--run rejected: receipt gate set differs from the required boundary set")
    end
    REQUIRED_GATES.each do |gate|
      value = gates[gate]
      if experimental
        case gate
        when "audio_encoder_full", "transformer_40_layers"
          next if value.as_s? == "recorded_numerical_drift"
        when "short_pipeline"
          next if value.as_bool? == false
        else
          next if value.as_bool? == true
        end
      else
        if value.as_bool? == true
          if known_drift_ledger_current?(gate)
            raise ArgumentError.new("--run rejected: current recorded evidence still rejects gate #{gate}")
          end
          next
        end
      end
      raise ArgumentError.new("--run rejected: required gate #{gate} is not qualified")
    end
    sync = json["sync_policy"]
    unless sync["terminal_host_waits_per_request"].as_i == 1 &&
           sync["intermediate_host_waits"].as_i == 0 &&
           sync["neural_readbacks_between_stages"].as_i == 0 &&
           sync["final_output_readbacks_per_request"].as_i == 1
      raise ArgumentError.new("--run rejected: receipt does not qualify the single-terminal-wait policy")
    end
    raise ArgumentError.new("--run rejected: receipt request count differs from this invocation") unless json["requests"].as_i == cli.requests
    geometry = json["geometry"]
    unless geometry["canvas_height"].as_i == CANVAS_HEIGHT && geometry["canvas_width"].as_i == CANVAS_WIDTH &&
           geometry["sequence_tokens"].as_i == 36_608 && geometry["predicted_latent_frames"].as_i == DIT_FRAMES &&
           geometry["deliver_frames"].as_i == DELIVER_FRAMES && geometry["deliver_height"].as_i == DELIVER_HEIGHT &&
           geometry["fps"].as_i == 16
      raise ArgumentError.new("--run rejected: receipt geometry is not the 5-second 480p target")
    end
    revision = git_revision
    raise ArgumentError.new("--run rejected: source revision differs") unless json["source_revision"].as_s == revision
    raise ArgumentError.new("--run rejected: input manifest hash differs") unless json["input_manifest_sha256"].as_s == sha256(File.join(input.root, "manifest.json"))
    config_path = File.join(root, "config.json")
    raise ArgumentError.new("Missing checkpoint config #{config_path}") unless File.file?(config_path)
    raise ArgumentError.new("--run rejected: checkpoint config hash differs") unless json["checkpoint_config_sha256"].as_s == sha256(config_path)
    unless json["checkpoint_identities"] == checkpoint_identities(root)
      raise ArgumentError.new("--run rejected: checkpoint file identity set differs")
    end
    expected_implementation = current_implementation_hashes
    supplied_implementation = json["implementation_sha256"].as_h.transform_values(&.as_s)
    unless supplied_implementation == expected_implementation
      raise ArgumentError.new("--run rejected: implementation source hashes differ")
    end
    if experimental
      validate_drift_ledgers!(json)
      tolerance = json["numerical_tolerance"]
      unless tolerance["absolute"].as_f == 2e-5 && tolerance["relative"].as_f == 2e-5
        raise ArgumentError.new("--run rejected: experimental mode cannot widen numerical tolerances")
      end
      "experimental_run_admitted"
    else
      raise ArgumentError.new("--run rejected: strict mode cannot include drift exceptions") if json.as_h.has_key?("numeric_drift")
      "qualified"
    end
  end

  private def self.validate_drift_ledgers!(receipt : JSON::Any) : Nil
    records = receipt["numeric_drift"].as_a
    raise ArgumentError.new("--run rejected: experimental mode requires exactly two recorded drift ledgers") unless records.size == 2
    expected = {
      AUDIO_DRIFT_GATE => {AUDIO_DRIFT_LEDGER, "audio_encoder_full"},
      STACK_DRIFT_GATE => {STACK_DRIFT_LEDGER, "transformer_40_layers"},
    }
    seen = Set(String).new
    current_hashes = current_implementation_hashes
    records.each do |record|
      gate = record["gate"].as_s
      path, receipt_gate = expected[gate]? || raise ArgumentError.new("--run rejected: drift gate is outside the exact allowlist")
      digest = record["ledger_sha256"].as_s
      unless seen.add?(gate) && record["ledger_path"].as_s == path &&
             digest.matches?(/\A[0-9a-f]{64}\z/) && File.file?(path) && sha256(path) == digest &&
             receipt["gates"][receipt_gate].as_s? == "recorded_numerical_drift"
        raise ArgumentError.new("--run rejected: drift receipt does not match its exact ledger and gate")
      end
      ledger = JSON.parse(File.read(path))
      unless ledger["schema"].as_s == "cogni.wan.numeric-drift.v1" &&
             ledger["status"].as_s == "recorded_not_admitted" && ledger["all_values_finite"].as_bool &&
             ledger["allowed_numeric_gates"].as_a.map(&.as_s) == [gate] &&
             ledger["absolute_tolerance"].as_f == 2e-5 && ledger["relative_tolerance"].as_f == 2e-5
        raise ArgumentError.new("--run rejected: recorded drift ledger exceeds its exact finite 2e-5 scope")
      end
      case gate
      when AUDIO_DRIFT_GATE
        config = ledger["configuration"]
        input_shape = ledger["input_shape"]
        unless ledger["label"].as_s == "experimental-audio-conditioning-2026-10-19" &&
               config["use_mps"].as_bool && config["source_boundary_normalizer"].as_bool &&
               config["graph_gelu"].as_bool && config["graph_position"].as_bool &&
               input_shape["input_samples"].as_i == 80_000 &&
               ledger["output_shape"].as_a.map(&.as_i) == [25_i64, 1024_i64, 84_i64]
          raise ArgumentError.new("--run rejected: audio drift ledger configuration differs from the 80,000-sample MPS source path")
        end
      when STACK_DRIFT_GATE
        config = ledger["configuration"]
        unless ledger["label"].as_s == "experimental-s2v-real-stack-prepared-mps-chunk512-2026-10-19" &&
               config["official_layers"].as_i == 40 && config["dim"].as_i == 5_120 &&
               config["heads"].as_i == 40 && config["ffn_dim"].as_i == 13_824
          raise ArgumentError.new("--run rejected: real-stack drift ledger is not the prepared MPS chunk512 run")
        end
      end
      ledger_hashes = ledger["implementation_sha256"].as_h.transform_values(&.as_s)
      raise ArgumentError.new("--run rejected: drift ledger has no implementation hashes") if ledger_hashes.empty?
      ledger_hashes.each do |relative, implementation_digest|
        unless safe_implementation_path?(relative) && current_hashes[relative]? == implementation_digest
          raise ArgumentError.new("--run rejected: implementation changed since drift evidence for #{gate}")
        end
      end
    end
    unless seen == Set{AUDIO_DRIFT_GATE, STACK_DRIFT_GATE}
      raise ArgumentError.new("--run rejected: both known drift gates must be recorded exactly once")
    end
  end

  private def self.known_drift_ledger_current?(receipt_gate : String) : Bool
    path, gate = case receipt_gate
                 when "audio_encoder_full"    then {AUDIO_DRIFT_LEDGER, AUDIO_DRIFT_GATE}
                 when "transformer_40_layers" then {STACK_DRIFT_LEDGER, STACK_DRIFT_GATE}
                 else                              return false
                 end
    return false unless File.file?(path)
    ledger = JSON.parse(File.read(path))
    return false unless ledger["schema"].as_s? == "cogni.wan.numeric-drift.v1" &&
                        ledger["status"].as_s? == "recorded_not_admitted" &&
                        ledger["allowed_numeric_gates"].as_a.map(&.as_s) == [gate]
    current_hashes = current_implementation_hashes
    hashes = ledger["implementation_sha256"].as_h.transform_values(&.as_s)
    !hashes.empty? && hashes.all? do |relative, digest|
      safe_implementation_path?(relative) && current_hashes[relative]? == digest
    end
  end

  private def self.safe_implementation_path?(relative : String) : Bool
    path = Path[relative]
    path.relative? && !path.parts.includes?("..") &&
      IMPLEMENTATION_FILES.includes?(relative) && File.file?(File.expand_path(File.join(__DIR__, "..", relative)))
  end

  # Device-free receipt validation for gate authors. This uses the exact same
  # source, input, checkpoint, geometry, synchronization, and ledger checks as
  # --run, but never initializes Metal or reads model payloads.
  def self.validate_receipt!(cli : WanS2VCLI) : Nil
    input = WanS2VInputSet.new(cli.inputs)
    root = File.expand_path(cli.checkpoint_root)
    mode = validate_run_receipt!(cli.validate_receipt, cli, input, root)
    puts({schema: "wan_s2v_receipt_validation_v1", status: mode,
          receipt: File.expand_path(cli.validate_receipt),
          requests: cli.requests, source_revision: git_revision,
          input_manifest_sha256: sha256(File.join(input.root, "manifest.json")),
          checkpoint_config_sha256: sha256(File.join(root, "config.json")),
          numerical_tolerance: {absolute: 2e-5, relative: 2e-5},
          device_initialized: false, model_payloads_loaded: false}.to_json)
  end

  private def self.git_revision : String
    output = IO::Memory.new
    status = Process.run("git", ["rev-parse", "HEAD"], output: output,
      error: Process::Redirect::Close)
    raise ArgumentError.new("Cannot resolve source revision") unless status.success?
    output.to_s.strip
  end

  private def self.dit_paths(root : String) : Array(String)
    (1..4).map do |index|
      shard = index.to_s.rjust(5, '0')
      path = File.join(root, "diffusion_pytorch_model-#{shard}-of-00004.safetensors")
      raise ArgumentError.new("Missing DIT checkpoint shard #{path}") unless File.file?(path)
      path
    end
  end

  # Bind exact checkpoint files without hashing tens of GiB on every run.
  # CheckpointFile and StageIO recheck file identity before their payloads are
  # consumed by the serial GPU chain.
  private def self.checkpoint_identities(root : String) : JSON::Any
    xlsr_root = File.join(root, "wav2vec2-large-xlsr-53-english")
    paths = dit_paths(root) + [
      File.join(root, "models_t5_umt5-xxl-enc-bf16.safetensors"),
      File.join(root, "Wan2.1_VAE.safetensors"),
      File.join(xlsr_root, "model.safetensors"),
    ]
    records = paths.map do |path|
      raise ArgumentError.new("Missing checkpoint file #{path}") unless File.file?(path)
      real = File.real_path(path)
      stat = uninitialized LibC::Stat
      raise ArgumentError.new("Cannot stat checkpoint file #{real}") unless LibC.stat(real.to_unsafe, pointerof(stat)) == 0
      mtime_ns = stat.st_mtimespec.tv_sec.to_i64 * 1_000_000_000_i64 + stat.st_mtimespec.tv_nsec.to_i64
      {path: real, size: stat.st_size.to_i64, device: stat.st_dev.to_i64,
       inode: stat.st_ino.to_i64, mtime_ns: mtime_ns}
    end.sort_by(&.[:path])
    JSON.parse(records.to_json)
  end

  private def self.page_size : Int64
    value = LibC.sysconf(LibC::SC_PAGESIZE).to_i64
    raise "Unable to read VM page size" unless value > 0
    value
  end

  private def self.rounded_bytes(bytes : Int64, page : Int64) : Int64
    raise ArgumentError.new("Invalid stage-pool allocation size") unless bytes > 0
    raise ArgumentError.new("Stage-pool accounting overflow") if bytes > Int64::MAX - (page - 1)
    ((bytes + page - 1) // page) * page
  end

  private def self.checked_product(label : String, *factors : Int64) : Int64
    factors.reduce(1_i64) do |product, factor|
      unless factor >= 0 && (factor == 0 || product <= Int64::MAX // factor)
        raise ArgumentError.new("#{label} byte-count overflow")
      end
      product * factor
    end
  end

  private def self.checked_sum(label : String, values : Enumerable(Int64)) : Int64
    values.reduce(0_i64) do |total, value|
      raise ArgumentError.new("#{label} byte-count overflow") if value < 0 || value > Int64::MAX - total
      total + value
    end
  end

  # Arena.alloc on a StagePool creates a page-aligned view for each buffer.
  # Keep the per-buffer rounding here; rounding the total would undercount.
  private def self.pool_bytes_for_counts(counts : Enumerable(Int64), page : Int64) : Int64
    counts.reduce(0_i64) do |total, count|
      unless count > 0 && count <= Int32::MAX
        raise ArgumentError.new("Stage-pool element count is outside Arena.alloc's Int32 range")
      end
      bytes = checked_product("Stage-pool allocation", count, sizeof(Float32).to_i64)
      value = rounded_bytes(bytes, page)
      raise ArgumentError.new("Stage-pool aggregate overflows") if value > Int64::MAX - total
      total + value
    end
  end

  # Reads safetensors headers only and accounts the exact stage-pool views
  # required by WanS2VStageWeights. No payload bytes or Metal device are read.
  private def self.header_weight_phase(paths : Array(String),
                                       bindings : Array(ML::Video::WeightBinding),
                                       dtype : ML::Video::WeightDType,
                                       page : Int64,
                                       require_all : Bool = false)
    files = paths.map { |path| ML::Video::CheckpointFile.new(File.expand_path(path)) }
    begin
      plan = ML::Video::WeightPlan.new(files, bindings, Int64::MAX, 0_i64, dtype, require_all)
      views = plan.entries.reduce(0_i64) do |total, entry|
        destination = rounded_bytes(checked_product("checkpoint destination",
          entry.info.n_elements, dtype.bytes_per_element.to_i64), page)
        convert = entry.info.dtype.bytes_per_element != dtype.bytes_per_element ||
                  (dtype.f32? && !entry.info.dtype.f32?)
        raw = convert ? rounded_bytes(entry.info.data_bytes, page) : 0_i64
        amount = destination + raw
        raise ArgumentError.new("Checkpoint phase pool accounting overflows") if amount > Int64::MAX - total
        total + amount
      end
      {pool_bytes: views, destination_bytes: plan.destination_bytes,
       raw_payload_bytes: plan.staging_bytes, tensor_count: plan.entries.size.to_i64}
    ensure
      files.each(&.close)
    end
  end

  private def self.push_vae_cache!(counts : Array(Int64), cfg : ML::Video::WanVAEConfig,
                                   channels : Int32, height : Int32, width : Int32) : Nil
    plane = checked_product("VAE cache plane", cfg.batches.to_i64, channels.to_i64,
      height.to_i64, width.to_i64)
    # WanVAE allocates two cache buffers, each containing two frames.
    per_buffer = checked_product("VAE cache buffer", plane, 2_i64)
    raise ArgumentError.new("VAE cache exceeds Arena.alloc's Int32 range") if per_buffer > Int32::MAX
    counts << per_buffer << per_buffer
  end

  private def self.push_vae_residual_caches!(counts : Array(Int64), cfg : ML::Video::WanVAEConfig,
                                             input : Int32, output : Int32,
                                             height : Int32, width : Int32) : Nil
    push_vae_cache!(counts, cfg, input, height, width)
    push_vae_cache!(counts, cfg, output, height, width)
  end

  # Mirrors WanVAE's pinned encoder_cache_specs + decoder_cache_specs so the
  # plan includes every two-frame double-buffered cache before any allocation.
  private def self.vae_pool_counts(cfg : ML::Video::WanVAEConfig,
                                   latent_frames : Int32) : Array(Int64)
    counts = [] of Int64
    activation = ML::Video::WanVAE.activation_scratch_values(cfg).to_i64
    5.times { counts << activation }

    dims = cfg.encoder_dims
    height, width = cfg.height, cfg.width
    push_vae_cache!(counts, cfg, 3, height, width) # encoder.conv1
    index = 0
    cfg.dim_mult.size.times do |stage|
      input, output = dims[stage], dims[stage + 1]
      cfg.num_res_blocks.times do
        push_vae_residual_caches!(counts, cfg, input, output, height, width)
        input = output
        index += 1
      end
      if stage < cfg.dim_mult.size - 1
        height //= 2
        width //= 2
        push_vae_cache!(counts, cfg, output, height, width) if cfg.temporal_downsample[stage]
        index += 1
      end
    end
    push_vae_residual_caches!(counts, cfg, dims.last, dims.last, height, width) # middle.0
    push_vae_residual_caches!(counts, cfg, dims.last, dims.last, height, width) # middle.2
    push_vae_cache!(counts, cfg, dims.last, height, width)                      # encoder.head.2

    dims = cfg.decoder_dims
    height, width = cfg.latent_height, cfg.latent_width
    push_vae_cache!(counts, cfg, cfg.z_dim, height, width) # decoder.conv1
    push_vae_residual_caches!(counts, cfg, dims.first, dims.first, height, width)
    push_vae_residual_caches!(counts, cfg, dims.first, dims.first, height, width)
    reversed_temporal = cfg.temporal_downsample.reverse
    cfg.dim_mult.size.times do |stage|
      source_in, output = dims[stage], dims[stage + 1]
      input = stage >= 1 ? source_in // 2 : source_in
      (cfg.num_res_blocks + 1).times do
        push_vae_residual_caches!(counts, cfg, input, output, height, width)
        input = output
      end
      if stage < cfg.dim_mult.size - 1
        push_vae_cache!(counts, cfg, output, height, width) if reversed_temporal[stage]
        height *= 2
        width *= 2
      end
    end
    push_vae_cache!(counts, cfg, dims.last, height, width) # decoder.head.2

    decoded_frames = cfg.decoded_frames(latent_frames)
    counts << checked_product("VAE latent sequence", cfg.batches.to_i64, cfg.z_dim.to_i64,
      latent_frames.to_i64, cfg.latent_height.to_i64, cfg.latent_width.to_i64)
    counts << checked_product("VAE decoded output", cfg.batches.to_i64, 3_i64,
      decoded_frames.to_i64, cfg.height.to_i64, cfg.width.to_i64)
    counts << checked_product("VAE temporal window", cfg.batches.to_i64, 3_i64,
      81_i64, cfg.height.to_i64, cfg.width.to_i64)
    counts
  end

  private def self.memory_plan(cli : WanS2VCLI) : JSON::Any
    build_memory_plan(cli)
  end

  # Header-only aggregate for the exact 832x512 / 84-frame request. Scratch
  # phase views overlap only where their serial lifetimes permit it; persistent
  # state, direct-upload constants, one reusable MPS linear stage, and caller
  # reserve are added without overlap. This is a budget calculation, not an
  # admission receipt or a performance/quality claim.
  private def self.build_memory_plan(cli : WanS2VCLI) : JSON::Any
    root = File.expand_path(cli.checkpoint_root)
    budget = checked_product("budget", cli.budget_mib, 1024_i64, 1024_i64)
    reserve = checked_product("reserve", cli.reserve_mib, 1024_i64, 1024_i64)
    page = page_size
    cfg = transformer_config
    dit = dit_paths(root)
    t5_cfg = ML::Video::WanT5Config.new
    xlsr_root = File.join(root, "wav2vec2-large-xlsr-53-english")
    xlsr_cfg = ML::Video::WanAudioEncoderConfig.from_hf_json(
      File.join(xlsr_root, "config.json"), 80_000, INTERNAL_FRAMES)
    causal_cfg = ML::Video::WanS2VCausalAudioConfig.new(bucket_frames: INTERNAL_FRAMES)
    vae_cfg = ML::Video::WanVAEConfig.new(1, CANVAS_HEIGHT, CANVAS_WIDTH)

    phases = {} of String => NamedTuple(pool_bytes: Int64, destination_bytes: Int64,
      raw_payload_bytes: Int64, tensor_count: Int64)
    phases["t5"] = header_weight_phase(
      [File.join(root, "models_t5_umt5-xxl-enc-bf16.safetensors")],
      ML::Video::WanT5.checkpoint_bindings(t5_cfg), ML::Video::WeightDType::BF16, page, true)
    phases["xlsr"] = header_weight_phase(
      [File.join(xlsr_root, "model.safetensors")],
      ML::Video::WanAudioEncoder.checkpoint_bindings(xlsr_cfg),
      ML::Video::WeightDType::F32, page)
    phases["causal_audio"] = header_weight_phase(
      dit, ML::Video::WanS2VCausalAudioEncoder.checkpoint_bindings(causal_cfg),
      ML::Video::WeightDType::F32, page)
    phases["vae"] = header_weight_phase(
      [File.join(root, "Wan2.1_VAE.safetensors")],
      ML::Video::WanVAE.checkpoint_bindings(vae_cfg, ML::GGUF::SafeTensorDType::F32),
      ML::Video::WeightDType::F32, page)
    phases["dit"] = header_weight_phase(
      dit, ML::Video::WanS2VTransformer.checkpoint_bindings(cfg),
      ML::Video::WeightDType::BF16, page)
    weight_pool_bytes = phases.values.max_of(&.[:pool_bytes])

    t5_counts = [] of Int64
    t5_rows = t5_cfg.rows.to_i64
    9.times { t5_counts << t5_rows * t5_cfg.dim }
    t5_counts << t5_rows
    2.times { t5_counts << t5_rows * t5_cfg.ffn_dim }
    t5_counts << t5_cfg.dim.to_i64 * Math.max(t5_cfg.dim, t5_cfg.ffn_dim)
    4.times { t5_counts << ML::Video::WanT5.attention_score_count(t5_cfg).to_i64 }
    t5_pool_bytes = pool_bytes_for_counts(t5_counts, page)

    xlsr_counts = [] of Int64
    frame = xlsr_cfg.sequence_values.to_i64
    ffn = xlsr_cfg.ffn_values.to_i64
    conv = xlsr_cfg.max_conv_values.to_i64
    xlsr_counts.concat([xlsr_cfg.input_samples.to_i64, 1_i64, 1_i64,
                        xlsr_cfg.input_samples.to_i64, conv, conv])
    9.times { xlsr_counts << frame }
    2.times { xlsr_counts << ffn }
    xlsr_counts << xlsr_cfg.position_kernel.to_i64
    xlsr_extract_pool_bytes = pool_bytes_for_counts(xlsr_counts, page)
    causal_counts = [
      causal_cfg.input_frames.to_i64 * causal_cfg.input_dim,
      causal_cfg.max_activation_values.to_i64,
      causal_cfg.max_activation_values.to_i64,
    ]
    causal_pool_bytes = pool_bytes_for_counts(causal_counts, page)

    vae_counts = vae_pool_counts(vae_cfg, DIT_FRAMES + 1)
    vae_pool_bytes = pool_bytes_for_counts(vae_counts, page)

    dit_counts = ML::Video::WanS2VTransformer.transformer_scratch_plan(cfg)
      .map { |_, count| count.to_i64 }
    dit_counts.concat([
      cfg.text_tokens.to_i64 * cfg.dim, cfg.text_tokens.to_i64 * cfg.dim,
      cfg.rows.to_i64 * cfg.dim, cfg.rows.to_i64 * cfg.dim, cfg.dim.to_i64,
      12_i64 * cfg.dim, cfg.video_tokens.to_i64 * cfg.out_dim * 4,
    ])
    video_values = 16_i64 * DIT_FRAMES * LATENT_HEIGHT * LATENT_WIDTH
    3.times { dit_counts << video_values } # CFG predictions
    5.times { dit_counts << video_values } # UniPC history/scratch
    dit_counts.concat(ML::Video::WanS2VBlock.scratch_plan(cfg.block_config).map { |_, n| n.to_i64 })
    shared = ML::Video::WanS2VTransformer::REUSED_AUDIO_SCRATCH.keys.to_set
    dit_counts.concat(ML::Video::WanS2VAudioInjector.scratch_plan(cfg.audio_config)
      .reject { |name, _| shared.includes?(name) }.map { |_, n| n.to_i64 })
    if cli.attention_backend == "mps-chunked"
      # Three bounded chunk outputs are retained by the prepared attention
      # workspace; the query/key/value and attention arrays already belong to
      # the block scratch plan.
      dit_counts << 3_i64 * 512 * cfg.dim
    end
    dit_pool_bytes = pool_bytes_for_counts(dit_counts, page)
    scratch_phases = {"t5" => t5_pool_bytes, "xlsr" => xlsr_extract_pool_bytes,
                      "causal_audio" => causal_pool_bytes, "vae" => vae_pool_bytes,
                      "dit" => dit_pool_bytes}
    phase_residency = scratch_phases.to_h do |phase, scratch_bytes|
      weight_bytes = phases[phase == "vae" ? "vae" : phase == "dit" ? "dit" : phase][:pool_bytes]
      {phase, {weight_pool_bytes: weight_bytes, scratch_pool_bytes: scratch_bytes,
               combined_pool_bytes: checked_sum("#{phase} shared pool", [weight_bytes, scratch_bytes])}}
    end
    shared_pool_bytes = phase_residency.values.max_of(&.[:combined_pool_bytes])

    state_counts = [
      3_i64 * CANVAS_HEIGHT * CANVAS_WIDTH, # reference image
      80_000_i64,                           # cropped 16 kHz waveform
      3_i64 * video_values,                 # initial noise and sample ping-pong
      2_i64 * t5_cfg.rows * t5_cfg.dim,     # positive and negative contexts
      xlsr_cfg.output_values.to_i64,        # Wav2Vec hidden states retained for causal audio
      causal_cfg.feature_values.to_i64,     # first 84-frame feature bucket
      causal_cfg.local_output_values.to_i64,
      causal_cfg.global_output_values.to_i64,
      causal_cfg.local_output_values.to_i64, # unconditional causal-audio output
      causal_cfg.global_output_values.to_i64,
      16_i64 * LATENT_HEIGHT * LATENT_WIDTH,                  # reference latent
      16_i64 * DIT_FRAMES * LATENT_HEIGHT * LATENT_WIDTH,     # no-pose condition
      3_i64 * DELIVER_FRAMES * DELIVER_HEIGHT * CANVAS_WIDTH, # delivered output
      1_i64,                                                  # GPU non-finite flag
    ]
    state_bytes = checked_product("persistent state", checked_sum("persistent state values", state_counts), 4_i64)
    constants_bytes = checked_sum("direct-upload constants", [
      4_i64 * t5_cfg.tokens * sizeof(Int32),                         # positive/negative IDs and masks
      cfg.freq_dim.to_i64 * sizeof(Float32),                         # paired time-frequency correction
      cfg.sequence_tokens.to_i64 * 3 * sizeof(Float32),              # RoPE position metadata
      cfg.block_config.head_dim.to_i64 * sizeof(Float32),            # block RoPE frequency pair
      2_i64 * cfg.dim * sizeof(Float32),                             # block and injector zero biases
      16_i64 * 2 * sizeof(Float32),                                  # VAE latent scale constants
      STEPS.to_i64 * sizeof(Float32),                                # timestep upload values
      16_i64 * LATENT_HEIGHT * LATENT_WIDTH * sizeof(Float32),       # zero motion input
      Math.max(t5_cfg.dim, t5_cfg.ffn_dim).to_i64 * sizeof(Float32), # T5 zero vector
    ])
    mps_stage_bytes = ML::Video::WanS2VMPSLinearStage.required_weight_count(
      cfg.dim, cfg.ffn_dim).to_i64 * sizeof(Float32)
    raw_weight_traffic = phases.values.max_of(&.[:raw_payload_bytes])
    total = checked_sum("aggregate request residency", [shared_pool_bytes,
                                                        state_bytes, constants_bytes, mps_stage_bytes, reserve])
    result = {
      schema:   "wan_s2v_memory_plan_v1",
      status:   total <= budget ? "within_requested_budget" : "rejected_over_budget",
      scope:    "header-only; exact checkpoint tensor binding/layout and page-rounded unified phase pool; no payload reads or Metal initialization",
      geometry: {canvas: [CANVAS_HEIGHT, CANVAS_WIDTH], delivered: [DELIVER_FRAMES, DELIVER_HEIGHT, CANVAS_WIDTH],
                 sequence_tokens: cfg.sequence_tokens, noisy_tokens: cfg.video_tokens,
                 reference_tokens: cfg.reference_tokens, framepack_tokens: cfg.framepack_tokens},
      pools: {unified_phase_pool_capacity_bytes: shared_pool_bytes,
              phase_capacities: phase_residency,
              checkpoint_phases: phases,
              scratch_phases: scratch_phases,
              allocation_model: "weights and scratch share one phase cursor; distinct serial phases alias the same pool offsets",
              all_views_are_page_rounded: true},
      retained: {persistent_state_bytes: state_bytes,
                 direct_upload_constants_bytes: constants_bytes,
                 reusable_mps_linear_stage_bytes: mps_stage_bytes,
                 opaque_mps_graph_plan_storage: "covered only by caller reserve"},
      checkpoint_phase_max_raw_payload_bytes: raw_weight_traffic,
      caller_reserve_bytes:                   reserve,
      total_estimated_resident_bytes:         total,
      budget_bytes:                           budget,
      over_budget_bytes:                      Math.max(0_i64, total - budget),
      headroom_note:                          "The user reserve must include unrelated residency and opaque Metal/MPS plan storage; the 15% host free-memory floor is checked again before runtime allocation.",
      run_admitted:                           false,
      attention_backend:                      cli.attention_backend,
      reason:                                 "A memory plan is a preflight only; full-run execution still requires current strict or exact-ledger experimental admission and successful runtime guards.",
    }
    JSON.parse(result.to_json)
  end

  def self.print_memory_plan(cli : WanS2VCLI) : Nil
    result = memory_plan(cli)
    puts result.to_json
    if result["status"].as_s == "rejected_over_budget"
      raise ArgumentError.new("--memory-plan rejected: aggregate estimate exceeds --budget-mib by #{result["over_budget_bytes"].as_i64} bytes")
    end
  end

  # Header-only preflight for one official block and its injector. The stage
  # pool needs one page-rounded no-copy view per BF16 binding.
  private def self.probe_weight_capacity(paths : Array(String),
                                         bindings : Array(ML::Video::WeightBinding),
                                         page : Int64) : Tuple(Int64, Int64)
    files = paths.map { |path| ML::Video::CheckpointFile.new(path) }
    begin
      plan = ML::Video::WeightPlan.new(files, bindings, Int64::MAX,
        resident_dtype: ML::Video::WeightDType::BF16)
      capacity = plan.entries.sum do |entry|
        rounded_bytes(entry.info.n_elements * 2, page)
      end
      {capacity, plan.destination_bytes}
    ensure
      files.each(&.close)
    end
  end

  private def self.probe_fill_source : String
    <<-METAL
    #include <metal_stdlib>
    using namespace metal;
    kernel void wan_s2v_probe_fill(
        device float *target [[buffer(0)]], constant uint &count [[buffer(1)]],
        constant uint &seed [[buffer(2)]], constant float &scale [[buffer(3)]],
        uint i [[thread_position_in_grid]]) {
      if (i >= count) return;
      const int value = int((i + seed) % 23u) - 11;
      target[i] = float(value) * (scale / 11.0f);
    }
    METAL
  end

  private def self.probe_finite_source : String
    <<-METAL
    #include <metal_stdlib>
    using namespace metal;
    kernel void wan_s2v_probe_finite(
        device const float *first [[buffer(0)]], device const float *second [[buffer(1)]],
        device atomic_uint *nonfinite [[buffer(2)]], constant uint &count [[buffer(3)]],
        uint i [[thread_position_in_grid]]) {
      if (i >= count) return;
      if (!isfinite(first[i]) || !isfinite(second[i])) {
        atomic_fetch_or_explicit(nonfinite, 1u, memory_order_relaxed);
      }
    }
    METAL
  end

  private def self.prepare_probe_fill! : ML::Metal::ComputePipeline
    ML::Metal::PipelineCache.get("wan_s2v_probe_fill") do
      ML::Metal::ComputePipeline.new("wan_s2v_probe_fill", probe_fill_source)
    end
  end

  private def self.prepare_probe_finite! : ML::Metal::ComputePipeline
    ML::Metal::PipelineCache.get("wan_s2v_probe_finite") do
      ML::Metal::ComputePipeline.new("wan_s2v_probe_finite", probe_finite_source)
    end
  end

  private def self.probe_fill(e : ML::Metal::ComputeEncoder, buffer : ML::MetalBuffer,
                              count : Int32, seed : UInt32, scale : Float32) : Nil
    ML::Video::MetalOps.check(buffer, count.to_i64)
    pipeline = prepare_probe_fill!
    e.set_pipeline(pipeline).set_buffer(buffer, 0, ML::Metal::BufferAccess::Write)
    e.set_value(count.to_u32, 1).set_value(seed, 2).set_value(scale, 3)
    e.dispatch_1d(count)
  end

  private def self.fill_zero(e : ML::Metal::ComputeEncoder, buffer : ML::MetalBuffer,
                             count : Int32) : Nil
    probe_fill(e, buffer, count, 0_u32, 0_f32)
  end

  private def self.probe_finite(e : ML::Metal::ComputeEncoder,
                                first : ML::MetalBuffer, second : ML::MetalBuffer,
                                nonfinite : ML::MetalBuffer, count : Int32) : Nil
    ML::Video::MetalOps.check(first, count.to_i64)
    ML::Video::MetalOps.check(second, count.to_i64)
    ML::Video::MetalOps.check(nonfinite, 1_i64)
    pipeline = prepare_probe_finite!
    e.set_pipeline(pipeline).set_buffer(first, 0).set_buffer(second, 1)
    e.set_buffer(nonfinite, 2, ML::Metal::BufferAccess::ReadWrite)
    e.set_value(count.to_u32, 3).dispatch_1d(count)
  end

  # Explicit diagnostic only: runs one genuine block-0 + audio-injector-0
  # with production channel width and the requested full sequence length. It
  # makes no denoising, parity, or video-quality claim.
  private def self.probe_dit_block!(cli : WanS2VCLI) : Nil
    unless cli.attention_backend == "tiled"
      raise ArgumentError.new("--probe-dit-block currently exercises only tiled self-attention; mps-chunked is wired for --run only")
    end
    root = File.expand_path(cli.checkpoint_root)
    paths = dit_paths(root)
    budget = cli.budget_mib * 1024_i64 * 1024
    reserve = cli.reserve_mib * 1024_i64 * 1024
    memory_admission!(budget)

    # For the 832x512 canvas, 21 latent frames each contain a 32x52 DiT
    # patch grid. The sequence adds the matching reference grid; text512 is
    # a separate cross-attention context and is not part of this sequence.
    block_cfg = ML::Video::WanS2VConfig.new(1, 36_608, 512, 5_120, 40,
      13_824, 34_944, 10_000.0_f32, 21, 1_664)
    audio_cfg = ML::Video::WanS2VAudioConfig.new(1, 36_608, 5_120, 40,
      21, 1_664, 5)
    bindings = ML::Video::WanS2VBlock.checkpoint_bindings(block_cfg,
      ML::GGUF::SafeTensorDType::BF16, 0, 0, "block.0.", "audio.0.")
    page = page_size
    weight_capacity, weight_bytes = probe_weight_capacity(paths, bindings, page)

    sequence_values = block_cfg.seq * block_cfg.dim
    input_counts = [sequence_values, 12 * block_cfg.dim,
                    block_cfg.seq * 3, block_cfg.context_tokens * block_cfg.dim,
                    block_cfg.video_frames * audio_cfg.audio_tokens * block_cfg.dim,
                    block_cfg.video_frames * block_cfg.dim, sequence_values, sequence_values]
    input_pool_bytes = input_counts.sum { |count| rounded_bytes(count.to_i64 * 4, page) }
    block_scratch_counts = ML::Video::WanS2VBlock.scratch_plan(block_cfg).map(&.[1])
    audio_scratch_counts = ML::Video::WanS2VAudioInjector.scratch_plan(audio_cfg).map(&.[1])
    scratch_pool_bytes = (block_scratch_counts + audio_scratch_counts)
      .sum { |count| rounded_bytes(count.to_i64 * 4, page) }
    scratch_values = (block_scratch_counts + audio_scratch_counts).sum(&.to_i64)
    constants_bytes = ML::Video::WanS2VBlock.scratch_bytes(block_cfg) +
                      ML::Video::WanS2VAudioInjector.scratch_bytes(audio_cfg) - scratch_values * 4
    projection_stage_bytes = if cli.projection_backend == "mps"
                               ML::Video::WanS2VMPSLinearStage.required_weight_count(block_cfg.dim, block_cfg.ffn_dim).to_i64 * sizeof(Float32)
                             else
                               0_i64
                             end
    finite_flag_bytes = sizeof(UInt32).to_i64
    workspace_bytes = input_pool_bytes + scratch_pool_bytes + constants_bytes + projection_stage_bytes + finite_flag_bytes
    total_estimate = weight_capacity + workspace_bytes + reserve
    raise ArgumentError.new("Block probe estimate exceeds --budget-mib: #{total_estimate} > #{budget}") if total_estimate > budget

    # WeightPlan includes no-copy view page slack in its reserve to retain the
    # same finite-budget invariant used by the stage pool itself.
    plan_reserve = workspace_bytes + reserve + (weight_capacity - weight_bytes)
    weight_pool = ML::Video::MetalStagePool.new(weight_capacity)
    scratch_pool = ML::Video::MetalStagePool.new(input_pool_bytes + scratch_pool_bytes)
    weights = ML::Video::Arena.new(weight_pool, "dit_probe_weights")
    inputs = ML::Video::Arena.new(scratch_pool, "dit_probe")
    scratch = ML::Video::Arena.new(scratch_pool, "dit_probe")
    stage_io = ML::Video::StageIO.new
    phase = nil.as(WanS2VStageWeights?)
    chain = nil.as(ML::Metal::MPSCommandChain?)
    io_started = false
    projection_stage = nil.as(ML::Video::WanS2VMPSLinearStage?)
    projection_workspace = nil.as(ML::Video::WanS2VMPSLinearWorkspace?)
    finite_state = nil.as(ML::Video::Arena?)
    begin
      ML::Video::WanS2VBlock.prepare!
      ML::Video::WanS2VAudioInjector.prepare!
      ML::Video::MetalOps.prepare_tiled_attention!
      prepare_probe_fill!
      prepare_probe_finite!
      ML::Video::WanS2VBlock.prepare_scratch!(scratch, block_cfg)
      ML::Video::WanS2VAudioInjector.prepare_scratch!(scratch, audio_cfg)

      phase = WanS2VStageWeights.new(paths, bindings, weights, weight_pool,
        "dit_probe", stage_io, budget, plan_reserve,
        ML::Video::WeightDType::BF16)
      phase.not_nil!.validate_sources!

      x = inputs.alloc("probe.x", sequence_values)
      modulation = inputs.alloc("probe.modulation", 12 * block_cfg.dim)
      positions = inputs.alloc("probe.positions", block_cfg.seq * 3)
      context = inputs.alloc("probe.context", block_cfg.context_tokens * block_cfg.dim)
      local_audio = inputs.alloc("probe.local_audio",
        block_cfg.video_frames * audio_cfg.audio_tokens * block_cfg.dim)
      global_audio = inputs.alloc("probe.global_audio", block_cfg.video_frames * block_cfg.dim)
      block_output = inputs.alloc("probe.block_output", sequence_values)
      output = inputs.alloc("probe.output", sequence_values)
      if cli.projection_backend == "mps"
        stage_count = ML::Video::WanS2VMPSLinearStage.required_weight_count(block_cfg.dim, block_cfg.ffn_dim)
        projection_stage = ML::Video::WanS2VMPSLinearStage.new(stage_count)
        projection_workspace = ML::Video::WanS2VMPSLinearWorkspace.new(
          block_cfg, audio_cfg, scratch, scratch, context, local_audio,
          projection_stage.not_nil!)
      end
      finite_state = ML::Video::Arena.new
      nonfinite = finite_state.not_nil!.alloc("probe.nonfinite_count", 1)
      nonfinite.write([0.0_f32])

      # Build every immutable I/O plan before the first stage begins.
      chain = ML::Metal::MPSCommandChain.new
      chain.not_nil!.pause!
      stage_io.begin!(phase.not_nil!.io_plan.not_nil!, chain.not_nil!)
      io_started = true
      chain.not_nil!.resume!
      phase.not_nil!.encode_conversions(chain.not_nil!.encoder)
      e = chain.not_nil!.encoder
      probe_fill(e, x, sequence_values, 1_u32, 0.025_f32)
      probe_fill(e, modulation, 12 * block_cfg.dim, 2_u32, 0.01_f32)
      probe_fill(e, positions, block_cfg.seq * 3, 3_u32, 0.0_f32)
      probe_fill(e, context, block_cfg.context_tokens * block_cfg.dim, 4_u32, 0.02_f32)
      probe_fill(e, local_audio, block_cfg.video_frames * audio_cfg.audio_tokens * block_cfg.dim, 5_u32, 0.015_f32)
      probe_fill(e, global_audio, block_cfg.video_frames * block_cfg.dim, 6_u32, 0.015_f32)
      ML::Video::WanS2VBlock.encode_with_audio(e, x, modulation, positions,
        context, local_audio, global_audio, block_output, output, weights, scratch,
        block_cfg, audio_cfg, block_weight_prefix: "block.0.",
        audio_weight_prefix: "audio.0.", tiled_attention: true,
        execution_chain: chain.not_nil!, mps_workspace: projection_workspace)
      e = chain.not_nil!.encoder
      probe_finite(e, block_output, output, nonfinite, sequence_values)
      chain.not_nil!.pause!
      stage_io.retire!(chain.not_nil!)
      chain.not_nil!.commit_and_wait
      stage_io.finish!(chain.not_nil!)
      flag_bytes = Bytes.new(sizeof(UInt32))
      nonfinite.read_bytes(flag_bytes.to_unsafe, flag_bytes.size)
      nonfinite_count = flag_bytes.to_unsafe.as(UInt32*).value
      chain.not_nil!.release
      projection_workspace.try(&.release_after_terminal_wait!)
      projection_stage.try(&.release_after_terminal_wait!)
      stage_io.release
      phase.not_nil!.close_sources!
      finite_state.not_nil!.release
      scratch.release
      inputs.release
      weights.release
      scratch_pool.release
      weight_pool.release
      puts({schema: "wan_s2v_dit_block_probe_v1", status: "completed_cost_diagnostic",
            claim_scope: "one real BF16 block 0 plus audio injector 0; no video quality or full-stack parity claim",
            sequence_tokens: block_cfg.seq, split_tokens: block_cfg.split,
            video_tokens: block_cfg.video_tokens, dim: block_cfg.dim, heads: block_cfg.heads,
            head_dim: block_cfg.head_dim, ffn_dim: block_cfg.ffn_dim,
            attention: "tiled", checkpoint_dtype: "BF16", operator_math: "F32",
            weight_bindings: bindings.size, weight_pool_bytes: weight_capacity,
            scratch_pool_bytes: scratch_pool_bytes, procedural_input_pool_bytes: input_pool_bytes,
            projection_backend: cli.projection_backend,
            projection_stage_bytes: projection_stage_bytes,
            nonfinite_output_elements: nonfinite_count,
            finite_output_check: "GPU scan of real block and injector outputs; 4-byte counter read only after the sole terminal wait",
            outputs_finite: nonfinite_count == 0,
            estimated_bytes_including_reserve: total_estimate, budget_bytes: budget,
            terminal_host_waits: chain.not_nil!.wait_count,
            inter_operator_host_waits: 0, neural_activation_readbacks: 0,
            gpu_elapsed_seconds: chain.not_nil!.gpu_elapsed_seconds,
            command_segments: chain.not_nil!.segment_count}.to_json)
    rescue ex
      terminal = chain.nil?
      if active_chain = chain
        begin
          if stage_io.open?
            begin
              active_chain.pause!
            rescue ArgumentError
            end
            stage_io.retire!(active_chain)
          end
          active_chain.commit_and_wait unless active_chain.completed?
          stage_io.finish!(active_chain) if io_started && !stage_io.finished?
        rescue
        end
        terminal = active_chain.completed?
        active_chain.release if terminal
      end
      if terminal
        projection_workspace.try(&.release_after_terminal_wait!)
        projection_stage.try(&.release_after_terminal_wait!)
      end
      begin
        stage_io.release
      rescue
      end
      phase.try(&.close_sources!)
      if terminal
        finite_state.try(&.release)
        scratch.release
        inputs.release
        weights.release
        scratch_pool.release if scratch_pool.live_view_count == 0
        weight_pool.release if weight_pool.live_view_count == 0
      end
      raise ex
    end
  end

  private def self.prepare_weights(paths : Array(String), bindings : Array(ML::Video::WeightBinding),
                                   arena : ML::Video::Arena, budget : Int64, reserve : Int64,
                                   dtype : ML::Video::WeightDType,
                                   require_all : Bool = false) : Tuple(Array(ML::Video::CheckpointFile), ML::Video::PreparedWeights)
    files = paths.map { |path| ML::Video::CheckpointFile.new(path) }
    begin
      plan = ML::Video::WeightPlan.new(files, bindings, budget, reserve,
        resident_dtype: dtype, require_all: require_all)
      prepared = ML::Video::PreparedWeights.new(plan, arena)
      {files, prepared}
    rescue ex
      files.each(&.close)
      raise ex
    end
  end

  private def self.wait_and_close(command : ML::Metal::CommandBuffer) : Nil
    command.commit_and_wait
  end

  private def self.bucket_prefix_source : String
    <<-METAL
    #include <metal_stdlib>
    using namespace metal;
    kernel void wan_s2v_bucket_prefix84(
        device const float *source [[buffer(0)]], device float *target [[buffer(1)]],
        constant uint &layers [[buffer(2)]], constant uint &channels [[buffer(3)]],
        constant uint &source_frames [[buffer(4)]], constant uint &target_frames [[buffer(5)]],
        uint i [[thread_position_in_grid]]) {
      const uint count = layers * channels * target_frames;
      if (i >= count) return;
      const uint frame = i % target_frames;
      const uint layer_channel = i / target_frames;
      target[i] = source[layer_channel * source_frames + frame];
    }
    METAL
  end

  private def self.output_crop_source : String
    <<-METAL
    #include <metal_stdlib>
    using namespace metal;
    kernel void wan_s2v_output_crop(
        device const float *source [[buffer(0)]], device float *target [[buffer(1)]],
        constant uint &channels [[buffer(2)]], constant uint &source_frames [[buffer(3)]],
        constant uint &source_height [[buffer(4)]], constant uint &source_width [[buffer(5)]],
        constant uint &target_frames [[buffer(6)]], constant uint &target_height [[buffer(7)]],
        constant uint &target_width [[buffer(8)]], constant uint &source_y [[buffer(9)]],
        uint i [[thread_position_in_grid]]) {
      const uint count = channels * target_frames * target_height * target_width;
      if (i >= count) return;
      const uint x = i % target_width;
      const uint y = (i / target_width) % target_height;
      const uint t = (i / (target_width * target_height)) % target_frames;
      const uint c = i / (target_width * target_height * target_frames);
      const uint source_index = ((c * source_frames + t) * source_height + source_y + y) * source_width + x;
      target[i] = source[source_index];
    }
    METAL
  end

  private def self.encode_bucket_prefix(e : ML::Metal::ComputeEncoder,
                                        source : ML::MetalBuffer,
                                        target : ML::MetalBuffer,
                                        layers : Int32, channels : Int32,
                                        source_frames : Int32, target_frames : Int32) : Nil
    raise ArgumentError.new("Audio bucket output must contain at least 84 frames") unless source_frames >= target_frames && target_frames == 84
    ML::Video::MetalOps.check(source, layers.to_i64 * channels * source_frames)
    ML::Video::MetalOps.check(target, layers.to_i64 * channels * target_frames)
    ML::Video::MetalOps.distinct(target, source)
    pipeline = ML::Metal::PipelineCache.get("wan_s2v_bucket_prefix84") do
      ML::Metal::ComputePipeline.new("wan_s2v_bucket_prefix84", bucket_prefix_source)
    end
    e.set_pipeline(pipeline).set_buffer(source, 0).set_buffer(target, 1, ML::Metal::BufferAccess::Write)
    e.set_value(layers.to_u32, 2).set_value(channels.to_u32, 3)
    e.set_value(source_frames.to_u32, 4).set_value(target_frames.to_u32, 5)
    e.dispatch_1d(layers * channels * target_frames)
  end

  # T5 stage. Raw projected contexts remain resident in `state` after the
  # stage's one terminal wait; all model/scratch/plan storage is retired then.
  private def self.run_t5!(input : WanS2VInputSet, root : String, state : ML::Video::Arena,
                           budget : Int64, reserve : Int64) : Tuple(ML::MetalBuffer, ML::MetalBuffer)
    cfg = ML::Video::WanT5Config.new
    stage = ML::Video::Arena.new
    context_pos = state.alloc("s2v.context_positive", cfg.rows * cfg.dim)
    context_neg = state.alloc("s2v.context_negative", cfg.rows * cfg.dim)
    source = File.join(root, "models_t5_umt5-xxl-enc-bf16.safetensors")
    raise ArgumentError.new("Missing T5 checkpoint #{source}") unless File.file?(source)
    files, prepared = prepare_weights([source], ML::Video::WanT5.checkpoint_bindings(cfg),
      stage, budget, reserve, ML::Video::WeightDType::BF16, true)
    plans = nil.as(ML::Video::WanT5MPSPlans?)
    chain = ML::Metal::MPSCommandChain.new
    begin
      ML::Video::WanT5.prepare_scratch!(stage, cfg)
      ids_pos = input.int32_values("positive_ids.i32")
      mask_pos = input.int32_values("positive_mask.i32")
      ids_neg = input.int32_values("negative_ids.i32")
      mask_neg = input.int32_values("negative_mask.i32")
      ML::Video::WanT5.upload_inputs!(stage, ids_pos, mask_pos, cfg, "t5.pos.")
      ML::Video::WanT5.upload_inputs!(stage, ids_neg, mask_neg, cfg, "t5.neg.")
      plans = ML::Video::WanT5.prepare_mps_plans(cfg, stage)
      prepared.encode(chain.encoder)
      ML::Video::WanT5.encode(chain, stage, cfg, stage["t5.pos.ids"], stage["t5.pos.mask"],
        context_pos, plans.not_nil!)
      ML::Video::WanT5.mask_context(chain.encoder, context_pos, stage["t5.pos.mask"], context_pos, cfg)
      ML::Video::WanT5.encode(chain, stage, cfg, stage["t5.neg.ids"], stage["t5.neg.mask"],
        context_neg, plans.not_nil!)
      ML::Video::WanT5.mask_context(chain.encoder, context_neg, stage["t5.neg.mask"], context_neg, cfg)
      chain.commit_and_wait
      chain.release
      plans.not_nil!.release
      prepared.release
      stage.release
      files.each(&.close)
      {context_pos, context_neg}
    rescue ex
      chain.commit_and_wait unless chain.completed?
      chain.release if chain.completed?
      plans.try(&.release)
      prepared.release
      stage.release
      files.each(&.close)
      raise ex
    end
  end

  # XLSR hidden-state extraction followed by the native CausalAudioEncoder.
  # XLSR may emit multiple 84-frame buckets for long utterances; the exact
  # first bucket is sliced on GPU before causal conditioning.
  private def self.run_audio!(input : WanS2VInputSet, root : String,
                              dit_files : Array(String), state : ML::Video::Arena,
                              budget : Int64, reserve : Int64) : Tuple(ML::MetalBuffer, ML::MetalBuffer)
    wav_tensor = input.tensor("audio.f32")
    sample_count = (wav_tensor.bytes // 4).to_i32
    xlsr_root = File.join(root, "wav2vec2-large-xlsr-53-english")
    xlsr_cfg = ML::Video::WanAudioEncoderConfig.from_hf_json(
      File.join(xlsr_root, "config.json"), sample_count, 84)
    causal_cfg = ML::Video::WanS2VCausalAudioConfig.new(bucket_frames: 84)
    xlsr_stage = ML::Video::Arena.new
    causal_stage = ML::Video::Arena.new
    waveform = input.upload(xlsr_stage, "audio.f32", "audio.waveform")
    bucketed = state.alloc("s2v.audio_bucket84", causal_cfg.feature_values)
    local = state.alloc("s2v.local_audio", causal_cfg.local_output_values)
    global = state.alloc("s2v.global_audio", causal_cfg.global_output_values)
    xlsr_weights : ML::Video::WanAudioEncoderWeights? = nil
    xlsr_scratch : ML::Video::WanAudioEncoderScratch? = nil
    xlsr_plans : ML::Video::WanAudioMPSPlans? = nil
    causal_scratch : ML::Video::WanS2VCausalAudioScratch? = nil
    causal_weights : ML::Video::WanS2VCausalAudioWeights? = nil
    xlsr_files, xlsr_prepared = prepare_weights([File.join(xlsr_root, "model.safetensors")],
      ML::Video::WanAudioEncoder.checkpoint_bindings(xlsr_cfg), xlsr_stage,
      budget, reserve, ML::Video::WeightDType::F32)
    causal_bindings = ML::Video::WanS2VCausalAudioEncoder.checkpoint_bindings(causal_cfg)
    causal_files, causal_prepared = prepare_weights(dit_files, causal_bindings,
      causal_stage, budget, reserve, ML::Video::WeightDType::F32)
    chain = ML::Metal::MPSCommandChain.new
    begin
      xlsr_weights = ML::Video::WanAudioEncoder.bind_weights(xlsr_stage, xlsr_cfg)
      xlsr_scratch = ML::Video::WanAudioEncoder.prepare_scratch!(xlsr_stage, xlsr_cfg)
      xlsr_plans = ML::Video::WanAudioMPSPlans.new(xlsr_cfg, xlsr_weights.not_nil!, xlsr_scratch.not_nil!)
      xlsr_output = xlsr_stage.alloc("audio.hidden_states", xlsr_cfg.output_values)
      ML::Video::WanS2VCausalAudioEncoder.prepare!
      causal_scratch = ML::Video::WanS2VCausalAudioEncoder.prepare_scratch!(causal_stage, causal_cfg)
      causal_weights = ML::Video::WanS2VCausalAudioEncoder.bind_weights(causal_stage, causal_cfg)
      xlsr_prepared.encode(chain.encoder)
      ML::Video::WanAudioEncoder.encode(chain, xlsr_cfg, waveform, xlsr_output,
        xlsr_weights.not_nil!, xlsr_scratch.not_nil!, xlsr_plans.not_nil!)
      encode_bucket_prefix(chain.encoder, xlsr_output, bucketed,
        xlsr_cfg.hidden_state_count, xlsr_cfg.hidden_size, xlsr_cfg.bucket_frames, 84)
      causal_prepared.encode(chain.encoder)
      ML::Video::WanS2VCausalAudioEncoder.encode(chain.encoder, causal_cfg, bucketed,
        local, global, causal_weights.not_nil!, causal_scratch.not_nil!)
      chain.commit_and_wait
      chain.release
      xlsr_plans.not_nil!.release
      xlsr_prepared.release
      causal_prepared.release
      xlsr_stage.release
      causal_stage.release
      xlsr_files.each(&.close)
      causal_files.each(&.close)
      {local, global}
    rescue ex
      chain.commit_and_wait unless chain.completed?
      chain.release if chain.completed?
      xlsr_plans.try(&.release)
      xlsr_prepared.release
      causal_prepared.release
      xlsr_stage.release
      causal_stage.release
      xlsr_files.each(&.close)
      causal_files.each(&.close)
      raise ex
    end
  end

  # Setup-only, conservative budget check. This has no neural data access.
  private def self.memory_admission!(budget : Int64, floor_pct : Int32 = 15) : Nil
    raise ArgumentError.new("Memory free floor must be at least 15 percent") unless floor_pct >= 15 && floor_pct < 100
    output = IO::Memory.new
    status = Process.run("/usr/bin/memory_pressure", ["-Q"], output: output, error: Process::Redirect::Close)
    raise "Cannot read memory pressure" unless status.success?
    free = output.to_s.match(/System-wide memory free percentage:\s*(\d+)%/).try(&.[1].to_i) || raise "Missing free-memory signal"
    raise ArgumentError.new("System free memory #{free}% does not exceed the #{floor_pct}% floor") unless free > floor_pct
    output.clear
    status = Process.run("/usr/sbin/sysctl", ["-n", "hw.memsize"], output: output, error: Process::Redirect::Close)
    raise "Cannot read physical memory size" unless status.success?
    physical = output.to_s.strip.to_i64
    usable = physical * (free - floor_pct) // 100
    raise ArgumentError.new("Requested budget exceeds available headroom estimate") if budget > usable
  end

  private def self.validate_output_target!(path : String) : Nil
    target = File.expand_path(path)
    raise ArgumentError.new("Output path already exists: #{target}") if File.exists?(target) || File.symlink?(target)
    parent = File.dirname(target)
    raise ArgumentError.new("Output directory does not exist: #{parent}") unless Dir.exists?(parent)
  end

  private def self.run_reference_vae!(image : ML::MetalBuffer, root : String,
                                      state : ML::Video::Arena, budget : Int64,
                                      reserve : Int64) : Tuple(ML::MetalBuffer, ML::MetalBuffer)
    cfg = ML::Video::WanVAEConfig.new(1, CANVAS_HEIGHT, CANVAS_WIDTH)
    weight_arena = ML::Video::Arena.new
    workspace_arena = ML::Video::Arena.new
    latent = state.alloc("s2v.reference_latent", 16 * LATENT_HEIGHT * LATENT_WIDTH)
    condition = state.alloc("s2v.condition_latents", 16 * DIT_FRAMES * LATENT_HEIGHT * LATENT_WIDTH)
    path = File.join(root, "Wan2.1_VAE.safetensors")
    raise ArgumentError.new("Missing VAE checkpoint #{path}") unless File.file?(path)
    bindings = ML::Video::WanVAE.checkpoint_bindings(cfg, ML::GGUF::SafeTensorDType::F32)
    files, prepared = prepare_weights([path], bindings, weight_arena, budget, reserve,
      ML::Video::WeightDType::F32)
    workspace = ML::Video::WanVAE.prepare!(workspace_arena, weight_arena, cfg,
      budget, reserve, 1)
    command = ML::Metal::CommandBuffer.new
    begin
      ML::Video::WanVAE.prepare!
      ML::Video::WanS2VTransformer.prepare!
      e = ML::Metal::ComputeEncoder.new(command)
      prepared.encode(e)
      ML::Video::WanVAE.encode_image(e, image, latent, weight_arena, workspace, cfg)
      ML::Video::WanS2VTransformer.zero_no_pose_condition(e, condition, transformer_config)
      e.end_encoding
      wait_and_close(command)
      prepared.release
      workspace_arena.release
      weight_arena.release
      files.each(&.close)
      {latent, condition}
    rescue ex
      command.commit_and_wait unless command.completed?
      prepared.release
      workspace_arena.release
      weight_arena.release
      files.each(&.close)
      raise ex
    end
  end

  private def self.run_transformer!(initial : ML::MetalBuffer,
                                    condition : ML::MetalBuffer,
                                    reference : ML::MetalBuffer,
                                    context_pos : ML::MetalBuffer,
                                    context_neg : ML::MetalBuffer,
                                    local_audio : ML::MetalBuffer,
                                    global_audio : ML::MetalBuffer,
                                    paths : Array(String), root : String,
                                    state : ML::Video::Arena, budget : Int64,
                                    reserve : Int64) : ML::MetalBuffer
    cfg = transformer_config
    weight_arena = ML::Video::Arena.new
    transformer_scratch = ML::Video::Arena.new
    activation_scratch = ML::Video::Arena.new
    block_scratch = ML::Video::Arena.new
    audio_scratch = ML::Video::Arena.new
    solver_scratch = ML::Video::Arena.new
    sequence_values = cfg.rows * cfg.dim
    video_values = 16 * DIT_FRAMES * LATENT_HEIGHT * LATENT_WIDTH
    bindings = ML::Video::WanS2VTransformer.checkpoint_bindings(cfg)
    files, prepared = prepare_weights(paths, bindings, weight_arena, budget,
      reserve, ML::Video::WeightDType::BF16)
    command = ML::Metal::CommandBuffer.new
    begin
      ML::Video::WanS2VTransformer.prepare!
      ML::Video::MetalOps.prepare_tiled_attention!
      ML::Video::WanS2VTransformer.prepare_transformer_scratch!(transformer_scratch, cfg)
      ML::Video::WanS2VTransformer.prepare_reused_scratch!(block_scratch, audio_scratch, cfg)
      scheduler = ML::Video::WanUniPC.new(video_values, STEPS, shift: 3.0)
      scheduler.prepare_scratch!(solver_scratch)

      text_hidden = activation_scratch.alloc("s2v.text_hidden", cfg.text_tokens * cfg.dim)
      projected = activation_scratch.alloc("s2v.text_projected", cfg.text_tokens * cfg.dim)
      sequence = activation_scratch.alloc("s2v.sequence", sequence_values)
      alternate = activation_scratch.alloc("s2v.sequence_alternate", sequence_values)
      time_embedding = activation_scratch.alloc("s2v.time_embedding", cfg.dim)
      modulation = activation_scratch.alloc("s2v.modulation", 12 * cfg.dim)
      prediction = activation_scratch.alloc("s2v.prediction", cfg.video_tokens * cfg.out_dim * 4)
      conditional = activation_scratch.alloc("s2v.prediction_conditional", video_values)
      unconditional = activation_scratch.alloc("s2v.prediction_unconditional", video_values)
      guided = activation_scratch.alloc("s2v.prediction_guided", video_values)
      positions = activation_scratch.upload("s2v.positions", ML::Video::WanS2VTransformer.position_metadata(cfg))
      motion = state.upload("s2v.motion_zero", Array(Float32).new(16 * LATENT_HEIGHT * LATENT_WIDTH, 0_f32))
      sample_a = initial
      sample_b = state.alloc("s2v.sample_ping", video_values)
      timesteps = scheduler.timesteps
      step_times = Array(ML::MetalBuffer).new(STEPS) do |step|
        activation_scratch.upload("s2v.timestep_#{step}", [timesteps[step].to_f32])
      end

      e = ML::Metal::ComputeEncoder.new(command)
      prepared.encode(e)
      scheduler.reset!
      STEPS.times do |step|
        timestep = step_times[step]
        ML::Video::WanS2VTransformer.encode(e, sample_a, condition, reference,
          motion, context_pos, text_hidden, projected, timestep, time_embedding,
          modulation, positions, local_audio, global_audio, sequence, alternate,
          prediction, conditional, weight_arena, transformer_scratch, block_scratch,
          audio_scratch, cfg, 1, tiled_attention: true)
        ML::Video::WanS2VTransformer.encode(e, sample_a, condition, reference,
          motion, context_neg, text_hidden, projected, timestep, time_embedding,
          modulation, positions, local_audio, global_audio, sequence, alternate,
          prediction, unconditional, weight_arena, transformer_scratch, block_scratch,
          audio_scratch, cfg, 1, tiled_attention: true)
        scheduler.guidance(e, conditional, unconditional, guided, 4.5_f32)
        scheduler.encode_step(e, solver_scratch, guided, sample_a, sample_b)
        sample_a, sample_b = sample_b, sample_a
      end
      e.end_encoding
      wait_and_close(command)
      prepared.release
      solver_scratch.release
      audio_scratch.release
      block_scratch.release
      activation_scratch.release
      transformer_scratch.release
      weight_arena.release
      files.each(&.close)
      state.remove("s2v.sample.initial") unless sample_a.same?(initial)
      state.remove("s2v.sample_ping") unless sample_a.same?(state["s2v.sample_ping"])
      sample_a
    rescue ex
      command.commit_and_wait unless command.completed?
      prepared.release
      solver_scratch.release
      audio_scratch.release
      block_scratch.release
      activation_scratch.release
      transformer_scratch.release
      weight_arena.release
      files.each(&.close)
      raise ex
    end
  end

  private def self.run_decode!(sample : ML::MetalBuffer, reference : ML::MetalBuffer,
                               paths : Array(String), root : String,
                               state : ML::Video::Arena, budget : Int64,
                               reserve : Int64, output_path : String) : String
    cfg = ML::Video::WanVAEConfig.new(1, CANVAS_HEIGHT, CANVAS_WIDTH)
    weight_arena = ML::Video::Arena.new
    workspace_arena = ML::Video::Arena.new
    bindings = ML::Video::WanVAE.checkpoint_bindings(cfg, ML::GGUF::SafeTensorDType::F32)
    vae_path = File.join(root, "Wan2.1_VAE.safetensors")
    files, prepared = prepare_weights([vae_path], bindings, weight_arena, budget,
      reserve, ML::Video::WeightDType::F32)
    workspace = ML::Video::WanVAE.prepare!(workspace_arena, weight_arena, cfg,
      budget, reserve, DIT_FRAMES + 1)
    latent_sequence = workspace_arena.alloc("s2v.decode_latents", 16 * (DIT_FRAMES + 1) * LATENT_HEIGHT * LATENT_WIDTH)
    decoded = workspace_arena.alloc("s2v.decoded_85", 3 * 85 * CANVAS_HEIGHT * CANVAS_WIDTH)
    temporal = workspace_arena.alloc("s2v.window_81", 3 * 81 * CANVAS_HEIGHT * CANVAS_WIDTH)
    delivered = state.alloc("s2v.output_80x480", 3 * DELIVER_FRAMES * DELIVER_HEIGHT * CANVAS_WIDTH)
    command = ML::Metal::CommandBuffer.new
    begin
      ML::Video::WanVAE.prepare!
      e = ML::Metal::ComputeEncoder.new(command)
      prepared.encode(e)
      ML::Video::WanVAE.copy_frames!(e, reference, latent_sequence,
        1, 16, 1, 0, LATENT_HEIGHT, LATENT_WIDTH, DIT_FRAMES + 1, 0, 1)
      ML::Video::WanVAE.copy_frames!(e, sample, latent_sequence,
        1, 16, DIT_FRAMES, 0, LATENT_HEIGHT, LATENT_WIDTH, DIT_FRAMES + 1, 1, DIT_FRAMES)
      ML::Video::WanVAE.decode(e, latent_sequence, decoded, weight_arena,
        workspace, cfg, DIT_FRAMES + 1)
      ML::Video::WanVAE.crop_frames!(e, decoded, temporal, cfg, 85, 4, 81)
      crop_pipeline = ML::Metal::PipelineCache.get("wan_s2v_output_crop") do
        ML::Metal::ComputePipeline.new("wan_s2v_output_crop", output_crop_source)
      end
      count = 3 * DELIVER_FRAMES * DELIVER_HEIGHT * CANVAS_WIDTH
      e.set_pipeline(crop_pipeline).set_buffer(temporal, 0).set_buffer(delivered, 1, ML::Metal::BufferAccess::Write)
      e.set_value(3_u32, 2).set_value(81_u32, 3).set_value(CANVAS_HEIGHT.to_u32, 4)
      e.set_value(CANVAS_WIDTH.to_u32, 5).set_value(DELIVER_FRAMES.to_u32, 6)
      e.set_value(DELIVER_HEIGHT.to_u32, 7).set_value(CANVAS_WIDTH.to_u32, 8)
      e.set_value(16_u32, 9).dispatch_1d(count)
      e.end_encoding
      wait_and_close(command)
      prepared.release
      workspace_arena.release
      weight_arena.release
      files.each(&.close)
      raise ArgumentError.new("Output path already exists: #{output_path}") if File.exists?(output_path)
      output_bytes = Bytes.new(count * sizeof(Float32))
      delivered.read_bytes(output_bytes.to_unsafe, output_bytes.size)
      File.open(output_path, "wb") { |file| file.write(output_bytes) }
      puts({schema: "wan_s2v_run_receipt_v1", status: "staged_unqualified",
            terminal_host_waits: STAGED_TERMINAL_WAITS,
            internal_layer_or_operator_waits: 0,
            neural_readbacks_between_stages: 0,
            final_output_readbacks: 1, output: output_path,
            output_bytes: output_bytes.size,
            output_sha256: Digest::SHA256.hexdigest(output_bytes), fps: 16,
            frames: DELIVER_FRAMES, height: DELIVER_HEIGHT, width: CANVAS_WIDTH}.to_json)
      output_path
    rescue ex
      command.commit_and_wait unless command.completed?
      prepared.release
      workspace_arena.release
      weight_arena.release
      files.each(&.close)
      raise ex
    end
  end

  # Full staged implementation retained for integration review. It executes
  # exactly the declared five terminal stage waits; current `run!` does not call
  # it because the user-facing no-new-sync admission gate is still closed.
  private def self.run_staged!(cli : WanS2VCLI) : Nil
    input = WanS2VInputSet.new(cli.inputs)
    validate_output_target!(cli.output)
    root = File.expand_path(cli.checkpoint_root)
    budget = cli.budget_mib * 1024 * 1024
    reserve = cli.reserve_mib * 1024 * 1024
    memory_admission!(budget)
    dit = dit_paths(root)
    input_arena = ML::Video::Arena.new
    state = ML::Video::Arena.new
    begin
      image = input.upload(input_arena, "reference.f32", "input.reference")
      noise = input.upload(state, "noise.f32", "s2v.sample.initial")
      context_pos, context_neg = run_t5!(input, root, state, budget, reserve)
      local_audio, global_audio = run_audio!(input, root, dit, state, budget, reserve)
      ref_latent, condition_latent = run_reference_vae!(image, root, state, budget, reserve)
      final_latent = run_transformer!(noise, condition_latent, ref_latent,
        context_pos, context_neg, local_audio, global_audio, dit, root, state, budget, reserve)
      run_decode!(final_latent, ref_latent, dit, root, state, budget, reserve, cli.output)
    ensure
      input_arena.release
      state.release
    end
  end

  # Stage events serialize aliased phase ranges on the same compute chain.
  # Submissions may continue asynchronously; only the request owner waits.
  private def self.encode_phase!(chain : ML::Metal::MPSCommandChain,
                                 io : ML::Video::StageIO,
                                 weights : WanS2VStageWeights, &block) : Nil
    chain.pause!
    io.begin!(weights.plan_for(io), chain)
    chain.resume!
    weights.encode_conversions(chain.encoder)
    yield
    chain.pause!
    io.retire!(chain)
    chain.continue_async!
    chain.resume!
  end

  private def self.drain_request!(chain : ML::Metal::MPSCommandChain,
                                  io : ML::Video::StageIO) : Nil
    # Also used after an encoding exception. Never release aliased storage
    # while an I/O operation or a partially encoded compute root is live.
    begin
      if io.open?
        begin
          chain.pause!
        rescue ArgumentError
          # The exception may have occurred with the encoder already paused.
        end
        io.retire!(chain)
      end
    ensure
      # An event-retirement error must not bypass the terminal drain.
      begin
        chain.commit_and_wait unless chain.completed?
      ensure
        if chain.completed? && !io.open?
          begin
            io.finish!(chain)
          ensure
            chain.release if io.finished?
          end
        end
      end
    end
  end

  def self.run!(cli : WanS2VCLI) : Nil
    setup_started = Time.monotonic
    input = WanS2VInputSet.new(cli.inputs)
    root = File.expand_path(cli.checkpoint_root)
    mode = validate_run_receipt!(cli.admission_receipt, cli, input, root)
    input.validate_text_inputs!
    input.validate_finite_inputs!
    outputs = Array(String).new(cli.requests) do |request|
      request == 0 ? File.expand_path(cli.output) : File.expand_path(cli.output + ".repeat")
    end
    outputs.each { |path| validate_output_target!(path) }
    outputs.each { |path| validate_output_target!(path + ".json") }
    unless cli.projection_backend == "mps" && cli.attention_backend == "mps-chunked"
      raise ArgumentError.new("The pooled full runner currently requires mps projections and mps-chunked self-attention")
    end

    budget = checked_product("budget", cli.budget_mib, 1024_i64, 1024_i64)
    reserve = checked_product("reserve", cli.reserve_mib, 1024_i64, 1024_i64)
    plan = memory_plan(cli)
    unless plan["status"].as_s == "within_requested_budget"
      raise ArgumentError.new("--run rejected: aggregate pool/state/reserve estimate exceeds --budget-mib")
    end
    memory_admission!(budget)

    cfg = transformer_config
    t5_cfg = ML::Video::WanT5Config.new
    xlsr_root = File.join(root, "wav2vec2-large-xlsr-53-english")
    xlsr_cfg = ML::Video::WanAudioEncoderConfig.from_hf_json(
      File.join(xlsr_root, "config.json"), 80_000, INTERNAL_FRAMES)
    causal_cfg = ML::Video::WanS2VCausalAudioConfig.new(bucket_frames: INTERNAL_FRAMES)
    vae_cfg = ML::Video::WanVAEConfig.new(1, CANVAS_HEIGHT, CANVAS_WIDTH)
    dit = dit_paths(root)
    t5_path = File.join(root, "models_t5_umt5-xxl-enc-bf16.safetensors")
    xlsr_path = File.join(xlsr_root, "model.safetensors")
    vae_path = File.join(root, "Wan2.1_VAE.safetensors")

    pool_bytes = plan["pools"]["unified_phase_pool_capacity_bytes"].as_i64
    pool = ML::Video::MetalStagePool.new(pool_bytes)
    stage_io = ML::Video::StageIO.new
    state = ML::Video::Arena.new
    constants = ML::Video::Arena.new
    phase_weights = {} of String => ML::Video::Arena
    phase_scratch = {} of String => ML::Video::Arena
    stages = {} of String => WanS2VStageWeights
    block_scratch = nil.as(ML::Video::Arena?)
    audio_scratch = nil.as(ML::Video::Arena?)
    scheduler_scratch = nil.as(ML::Video::Arena?)
    activation_scratch = nil.as(ML::Video::Arena?)
    t5_plans = nil.as(ML::Video::WanT5MPSPlans?)
    xlsr_plans = nil.as(ML::Video::WanAudioMPSPlans?)
    normalizer = nil.as(ML::Video::WanAudioMPSNormalizationPlan?)
    gelu_plans = nil.as(ML::Video::WanAudioMPSGeluPlans?)
    position_plan = nil.as(ML::Video::WanAudioMPSPositionPlan?)
    linear_stage = nil.as(ML::Video::WanS2VMPSLinearStage?)
    positive_workspace = nil.as(ML::Video::WanS2VMPSLinearWorkspace?)
    negative_workspace = nil.as(ML::Video::WanS2VMPSLinearWorkspace?)
    attention_workspace = nil.as(ML::Video::WanAttentionMPSChunkWorkspace?)
    resources_drained = true
    first_output_sha256 = nil.as(String?)
    owners = WanS2VRunResources.new(pool, stage_io, state, constants,
      phase_weights, phase_scratch, stages)
    RUN_RESOURCES << owners
    begin
      {"t5", "xlsr", "causal_audio", "vae", "dit"}.each do |phase|
        # Weight and scratch arenas share the same phase cursor, so their
        # allocations are disjoint within a phase. The next phase starts at
        # offset zero only after StageIO's GPU event retires this one.
        phase_weights[phase] = ML::Video::Arena.new(pool, phase, "weights")
        phase_scratch[phase] = phase == "t5" ? phase_weights[phase] : ML::Video::Arena.new(pool, phase, "scratch")
      end

      # Allocate every phase's checkpoint views before the first StageIO load.
      # StageIO plans are immutable; a new plan will be created per request
      # while these model buffers and prepared plans remain resident.
      stages["t5"] = WanS2VStageWeights.new([t5_path],
        ML::Video::WanT5.checkpoint_bindings(t5_cfg), phase_weights["t5"],
        pool, "t5", stage_io, budget, reserve, ML::Video::WeightDType::BF16,
        true, true)
      stages["xlsr"] = WanS2VStageWeights.new([xlsr_path],
        ML::Video::WanAudioEncoder.checkpoint_bindings(xlsr_cfg), phase_weights["xlsr"],
        pool, "xlsr", stage_io, budget, reserve, ML::Video::WeightDType::F32,
        false, true)
      stages["causal_audio"] = WanS2VStageWeights.new(dit,
        ML::Video::WanS2VCausalAudioEncoder.checkpoint_bindings(causal_cfg),
        phase_weights["causal_audio"], pool, "causal_audio", stage_io,
        budget, reserve, ML::Video::WeightDType::F32, false, true)
      stages["vae"] = WanS2VStageWeights.new([vae_path],
        ML::Video::WanVAE.checkpoint_bindings(vae_cfg, ML::GGUF::SafeTensorDType::F32),
        phase_weights["vae"], pool, "vae", stage_io, budget, reserve,
        ML::Video::WeightDType::F32, false, true)
      stages["dit"] = WanS2VStageWeights.new(dit,
        ML::Video::WanS2VTransformer.checkpoint_bindings(cfg), phase_weights["dit"],
        pool, "dit", stage_io, budget, reserve, ML::Video::WeightDType::BF16,
        false, true)
      stages.each_value(&.validate_sources!)

      # Persistent request state is deliberately outside the phase pool. The
      # positive and negative causal outputs are separate because upstream
      # CFG zeros audio_input before CausalAudioEncoder, not after it.
      image = input.upload(state, "reference.f32", "s2v.reference_image")
      noise = input.upload(state, "noise.f32", "s2v.initial_noise")
      waveform = input.upload(state, "audio.f32", "s2v.waveform_80k")
      context_pos = state.alloc("s2v.context_positive", t5_cfg.rows * t5_cfg.dim)
      context_neg = state.alloc("s2v.context_negative", t5_cfg.rows * t5_cfg.dim)
      xlsr_hidden = state.alloc("s2v.xlsr_hidden", xlsr_cfg.output_values)
      audio_bucket = state.alloc("s2v.audio_bucket84", causal_cfg.feature_values)
      local_pos = state.alloc("s2v.local_audio_positive", causal_cfg.local_output_values)
      global_pos = state.alloc("s2v.global_audio_positive", causal_cfg.global_output_values)
      local_neg = state.alloc("s2v.local_audio_negative", causal_cfg.local_output_values)
      global_neg = state.alloc("s2v.global_audio_negative", causal_cfg.global_output_values)
      reference = state.alloc("s2v.reference_latent", 16 * LATENT_HEIGHT * LATENT_WIDTH)
      condition = state.alloc("s2v.condition_latents", 16 * DIT_FRAMES * LATENT_HEIGHT * LATENT_WIDTH)
      sample_a = state.alloc("s2v.sample_a", 16 * DIT_FRAMES * LATENT_HEIGHT * LATENT_WIDTH)
      sample_b = state.alloc("s2v.sample_b", 16 * DIT_FRAMES * LATENT_HEIGHT * LATENT_WIDTH)
      delivered = state.alloc("s2v.output_80x480", 3 * DELIVER_FRAMES * DELIVER_HEIGHT * CANVAS_WIDTH)
      nonfinite = state.alloc("s2v.nonfinite_count", 1)
      nonfinite.write([0_f32])

      # Prepare phase scratch with the exact admitted storage and arithmetic
      # variants. MPS plans, the reusable linear stage, and the denoise loop
      # are the continuation point for the root-owned run body.
      ML::Video::WanT5.prepare_scratch!(phase_scratch["t5"], t5_cfg, graph_attention: true)
      ML::Video::WanT5.upload_inputs!(phase_scratch["t5"], input.int32_values("positive_ids.i32"),
        input.int32_values("positive_mask.i32"), t5_cfg, "t5.pos.")
      ML::Video::WanT5.upload_inputs!(phase_scratch["t5"], input.int32_values("negative_ids.i32"),
        input.int32_values("negative_mask.i32"), t5_cfg, "t5.neg.")
      xlsr_weights = ML::Video::WanAudioEncoder.bind_weights(phase_weights["xlsr"], xlsr_cfg)
      xlsr_scratch = ML::Video::WanAudioEncoder.prepare_scratch!(phase_scratch["xlsr"], xlsr_cfg)
      ML::Video::WanS2VCausalAudioEncoder.prepare!
      causal_weights = ML::Video::WanS2VCausalAudioEncoder.bind_weights(phase_weights["causal_audio"], causal_cfg)
      causal_scratch = ML::Video::WanS2VCausalAudioEncoder.prepare_scratch!(phase_scratch["causal_audio"], causal_cfg)
      ML::Video::WanVAE.prepare!
      vae_workspace = ML::Video::WanVAE.prepare!(phase_scratch["vae"], phase_weights["vae"],
        vae_cfg, budget, reserve, DIT_FRAMES + 1)
      ML::Video::WanS2VTransformer.prepare!
      ML::Video::WanS2VTransformer.prepare_transformer_scratch!(phase_scratch["dit"], cfg)
      block_scratch = ML::Video::Arena.new(pool, "dit", "block")
      audio_scratch = ML::Video::Arena.new(pool, "dit", "injector")
      owners.extra_arenas << block_scratch.not_nil! << audio_scratch.not_nil!
      ML::Video::WanS2VTransformer.prepare_reused_scratch!(block_scratch.not_nil!, audio_scratch.not_nil!, cfg)
      scheduler = ML::Video::WanUniPC.new(16 * DIT_FRAMES * LATENT_HEIGHT * LATENT_WIDTH,
        STEPS, shift: 3.0)
      scheduler_scratch = ML::Video::Arena.new(pool, "dit", "solver")
      owners.extra_arenas << scheduler_scratch.not_nil!
      scheduler.prepare_scratch!(scheduler_scratch.not_nil!)
      # Keep direct uploads in a dedicated ordinary-buffer Arena; constants
      # are not allowed to occupy phase-pool offsets.
      motion = constants.upload("s2v.motion_zero",
        Array(Float32).new(16 * LATENT_HEIGHT * LATENT_WIDTH, 0_f32))
      stages.each_value(&.validate_sources!)

      owners.t5_plans = t5_plans = ML::Video::WanT5.prepare_mps_plans(t5_cfg, phase_scratch["t5"], graph_attention: true)
      owners.xlsr_plans = xlsr_plans = ML::Video::WanAudioMPSPlans.new(xlsr_cfg, xlsr_weights, xlsr_scratch, graph_gelu: true)
      owners.normalizer = normalizer = ML::Video::WanAudioMPSNormalizationPlan.new(xlsr_cfg, waveform, xlsr_scratch, source_boundaries: true)
      owners.gelu_plans = gelu_plans = ML::Video::WanAudioMPSGeluPlans.new(xlsr_cfg, xlsr_scratch)
      owners.position_plan = position_plan = ML::Video::WanAudioMPSPositionPlan.new(xlsr_cfg, xlsr_weights, xlsr_scratch)
      ts = phase_scratch["dit"]
      # The transformer validates every entry in its scratch arena against
      # caller inputs/outputs. Keep request activations in their own owner;
      # the shared phase cursor preserves the same disjoint ranges and budget.
      activation_scratch = ML::Video::Arena.new(pool, "dit", "request")
      owners.extra_arenas << activation_scratch.not_nil!
      a = activation_scratch.not_nil!
      sequence_values = cfg.rows * cfg.dim
      video_values = 16 * DIT_FRAMES * LATENT_HEIGHT * LATENT_WIDTH
      text_hidden = a.alloc("s2v.text_hidden", cfg.text_tokens * cfg.dim)
      projected = a.alloc("s2v.text_projected", cfg.text_tokens * cfg.dim)
      sequence = a.alloc("s2v.sequence", sequence_values)
      alternate = a.alloc("s2v.sequence_alternate", sequence_values)
      time_embedding = a.alloc("s2v.time_embedding", cfg.dim)
      modulation = a.alloc("s2v.modulation", 12 * cfg.dim)
      prediction = a.alloc("s2v.prediction", cfg.video_tokens * cfg.out_dim * 4)
      conditional = a.alloc("s2v.prediction_conditional", video_values)
      unconditional = a.alloc("s2v.prediction_unconditional", video_values)
      guided = a.alloc("s2v.prediction_guided", video_values)
      positions = constants.upload("s2v.positions", ML::Video::WanS2VTransformer.position_metadata(cfg))
      timesteps = scheduler.timesteps
      step_times = Array(ML::MetalBuffer).new(STEPS) do |step|
        constants.upload("s2v.timestep_#{step}", [timesteps[step].to_f32])
      end
      owners.linear_stage = linear_stage = ML::Video::WanS2VMPSLinearStage.new(
        ML::Video::WanS2VMPSLinearStage.required_weight_count(cfg.dim, cfg.ffn_dim))
      owners.positive_workspace = positive_workspace = ML::Video::WanS2VMPSLinearWorkspace.new(cfg.block_config, cfg.audio_config,
        block_scratch.not_nil!, audio_scratch.not_nil!, projected, local_pos, linear_stage)
      owners.negative_workspace = negative_workspace = ML::Video::WanS2VMPSLinearWorkspace.new(cfg.block_config, cfg.audio_config,
        block_scratch.not_nil!, audio_scratch.not_nil!, projected, local_neg, linear_stage)
      chunk_q = ts.alloc("s2v.attention_chunk_q", 512 * cfg.dim)
      chunk_a = ts.alloc("s2v.attention_chunk_a", 512 * cfg.dim)
      chunk_b = ts.alloc("s2v.attention_chunk_b", 512 * cfg.dim)
      bs = block_scratch.not_nil!
      owners.attention_workspace = attention_workspace = ML::Video::WanAttentionMPSChunkWorkspace.new(1,
        cfg.sequence_tokens, cfg.sequence_tokens, cfg.heads, cfg.dim // cfg.heads, 512,
        bs["wan_b_self_q"], bs["wan_b_self_k"], bs["wan_b_self_v"], bs["wan_b_self_attn"],
        chunk_q, chunk_a, chunk_b)
      latent_sequence = phase_scratch["vae"].alloc("s2v.decode_latents", 16 * (DIT_FRAMES + 1) * LATENT_HEIGHT * LATENT_WIDTH)
      decoded = phase_scratch["vae"].alloc("s2v.decoded_85", 3 * 85 * CANVAS_HEIGHT * CANVAS_WIDTH)
      temporal = phase_scratch["vae"].alloc("s2v.window_81", 3 * 81 * CANVAS_HEIGHT * CANVAS_WIDTH)
      ML::Video::MetalOps.prepare_tiled_attention!
      prepare_probe_fill!
      prepare_probe_finite!
      crop_pipeline = ML::Metal::PipelineCache.get("wan_s2v_output_crop") do
        ML::Metal::ComputePipeline.new("wan_s2v_output_crop", output_crop_source)
      end
      # Verify the real page-rounded cursors, including all plan-owned views,
      # against the header-only budget before any checkpoint I/O is enqueued.
      plan["pools"]["phase_capacities"].as_h.each do |phase, capacity|
        actual = pool.phase_cursor(phase)
        expected = capacity["combined_pool_bytes"].as_i64
        raise ArgumentError.new("Runtime pool cursor differs for #{phase}: #{actual} != #{expected}") unless actual == expected
      end
      setup_seconds = (Time.monotonic - setup_started).total_seconds
      outputs.each_with_index do |output_path, request|
        request_started = Time.monotonic
        # The aggregate pool is already retained. Only reserve headroom is
        # additional at this boundary; applying the whole budget twice would
        # reject a valid retained-resource repeat.
        memory_admission!(reserve)
        io = request == 0 ? stage_io : ML::Video::StageIO.new
        owners.request_io = io
        chain = nil.as(ML::Metal::MPSCommandChain?)
        request_error = nil.as(Exception?)
        begin
          stages.each_value do |stage|
            stage.validate_sources!
            stage.plan_for(io)
          end
          active = ML::Metal::MPSCommandChain.new
          owners.chain = active
          chain = active
          resources_drained = false
          fill_zero(active.encoder, nonfinite, 1)
          ML::Video::MetalOps.copy(active.encoder, noise, sample_a, video_values)
          scheduler.reset!
          STDERR.puts "request=#{request + 1} queued setup; encoding T5"
          encode_phase!(active, io, stages["t5"]) do
            a = phase_scratch["t5"]
            ML::Video::WanT5.encode(active, a, t5_cfg, a["t5.pos.ids"], a["t5.pos.mask"],
              context_pos, t5_plans.not_nil!, source_gelu: true)
            ML::Video::WanT5.mask_context(active.encoder, context_pos, a["t5.pos.mask"], context_pos, t5_cfg)
            ML::Video::WanT5.encode(active, a, t5_cfg, a["t5.neg.ids"], a["t5.neg.mask"],
              context_neg, t5_plans.not_nil!, source_gelu: true)
            ML::Video::WanT5.mask_context(active.encoder, context_neg, a["t5.neg.mask"], context_neg, t5_cfg)
            probe_finite(active.encoder, context_pos, context_neg, nonfinite, t5_cfg.rows * t5_cfg.dim)
          end
          STDERR.puts "request=#{request + 1} queued T5; encoding XLSR"
          encode_phase!(active, io, stages["xlsr"]) do
            ML::Video::WanAudioEncoder.encode(active, xlsr_cfg, waveform, xlsr_hidden,
              xlsr_weights, xlsr_scratch, xlsr_plans.not_nil!, normalizer: normalizer,
              gelu_plans: gelu_plans, position_plan: position_plan)
            encode_bucket_prefix(active.encoder, xlsr_hidden, audio_bucket,
              xlsr_cfg.hidden_state_count, xlsr_cfg.hidden_size, xlsr_cfg.bucket_frames, INTERNAL_FRAMES)
            probe_finite(active.encoder, xlsr_hidden, xlsr_hidden, nonfinite, xlsr_cfg.output_values)
          end
          encode_phase!(active, io, stages["causal_audio"]) do
            ML::Video::WanS2VCausalAudioEncoder.encode(active.encoder, causal_cfg, audio_bucket,
              local_pos, global_pos, causal_weights, causal_scratch)
            # Match upstream arg_null: zero raw features BEFORE the biased,
            # nonlinear causal encoder. Its null outputs are not zero.
            fill_zero(active.encoder, audio_bucket, causal_cfg.feature_values)
            ML::Video::WanS2VCausalAudioEncoder.encode(active.encoder, causal_cfg, audio_bucket,
              local_neg, global_neg, causal_weights, causal_scratch)
            probe_finite(active.encoder, local_pos, local_neg, nonfinite, causal_cfg.local_output_values)
            probe_finite(active.encoder, global_pos, global_neg, nonfinite, causal_cfg.global_output_values)
          end
          STDERR.puts "request=#{request + 1} queued audio; encoding reference VAE"
          encode_phase!(active, io, stages["vae"]) do
            ML::Video::WanVAE.encode_image(active.encoder, image, reference, phase_weights["vae"], vae_workspace, vae_cfg,
              execution_chain: active)
            probe_finite(active.encoder, reference, reference, nonfinite, 16 * LATENT_HEIGHT * LATENT_WIDTH)
          end
          ML::Video::WanS2VTransformer.zero_no_pose_condition(active.encoder, condition, cfg)
          STDERR.puts "request=#{request + 1} queued reference VAE; encoding DiT"
          encode_phase!(active, io, stages["dit"]) do
            STEPS.times do |step|
              timestep = step_times[step]
              ML::Video::WanS2VTransformer.encode(active.encoder, sample_a, condition, reference,
                motion, context_pos, text_hidden, projected, timestep, time_embedding, modulation,
                positions, local_pos, global_pos, sequence, alternate, prediction, conditional,
                phase_weights["dit"], ts, block_scratch.not_nil!, audio_scratch.not_nil!, cfg, 1,
                tiled_attention: true, execution_chain: active, mps_workspace: positive_workspace,
                attention_workspace: attention_workspace)
              ML::Video::WanS2VTransformer.encode(active.encoder, sample_a, condition, reference,
                motion, context_neg, text_hidden, projected, timestep, time_embedding, modulation,
                positions, local_neg, global_neg, sequence, alternate, prediction, unconditional,
                phase_weights["dit"], ts, block_scratch.not_nil!, audio_scratch.not_nil!, cfg, 1,
                tiled_attention: true, execution_chain: active, mps_workspace: negative_workspace,
                attention_workspace: attention_workspace)
              e = active.encoder
              scheduler.guidance(e, conditional, unconditional, guided, 4.5_f32)
              scheduler.encode_step(e, scheduler_scratch.not_nil!, guided, sample_a, sample_b)
              sample_a, sample_b = sample_b, sample_a
              probe_finite(e, sample_a, sample_a, nonfinite, video_values)
              STDERR.puts "request=#{request + 1} queued denoise_step=#{step + 1}/#{STEPS} elapsed_host_seconds=#{(Time.monotonic - request_started).total_seconds}"
            end
          end
          encode_phase!(active, io, stages["vae"]) do
            e = active.encoder
            ML::Video::WanVAE.copy_frames!(e, reference, latent_sequence,
              1, 16, 1, 0, LATENT_HEIGHT, LATENT_WIDTH, DIT_FRAMES + 1, 0, 1)
            ML::Video::WanVAE.copy_frames!(e, sample_a, latent_sequence,
              1, 16, DIT_FRAMES, 0, LATENT_HEIGHT, LATENT_WIDTH, DIT_FRAMES + 1, 1, DIT_FRAMES)
            ML::Video::WanVAE.decode(e, latent_sequence, decoded, phase_weights["vae"],
              vae_workspace, vae_cfg, DIT_FRAMES + 1, execution_chain: active)
            e = active.encoder
            ML::Video::WanVAE.crop_frames!(e, decoded, temporal, vae_cfg, 85, 4, 81)
            count = 3 * DELIVER_FRAMES * DELIVER_HEIGHT * CANVAS_WIDTH
            e.set_pipeline(crop_pipeline).set_buffer(temporal, 0).set_buffer(delivered, 1, ML::Metal::BufferAccess::Write)
            e.set_value(3_u32, 2).set_value(81_u32, 3).set_value(CANVAS_HEIGHT.to_u32, 4)
            e.set_value(CANVAS_WIDTH.to_u32, 5).set_value(DELIVER_FRAMES.to_u32, 6)
            e.set_value(DELIVER_HEIGHT.to_u32, 7).set_value(CANVAS_WIDTH.to_u32, 8)
            e.set_value(16_u32, 9).dispatch_1d(count)
            probe_finite(e, delivered, delivered, nonfinite, count)
          end
          STDERR.puts "request=#{request + 1} queued decode and finite checks; terminal wait"
          active.commit_and_wait
          io.finish!(active)
          stages.each_value(&.validate_sources!)
          flag = Bytes.new(4)
          nonfinite.read_bytes(flag.to_unsafe, flag.size)
          raise "Non-finite neural values detected; discard generated output" unless flag.to_unsafe.as(UInt32*).value == 0
          bytes = Bytes.new(3 * DELIVER_FRAMES * DELIVER_HEIGHT * CANVAS_WIDTH * sizeof(Float32))
          delivered.read_bytes(bytes.to_unsafe, bytes.size)
          File.open(output_path, "wx") { |file| file.write(bytes) }
          output_sha256 = Digest::SHA256.hexdigest(bytes)
          first_output_sha256 = output_sha256 if request == 0
          output_manifest_sha256 = Digest::SHA256.hexdigest(File.read(File.join(input.root, "manifest.json")))
          output_implementation_sha256 = current_implementation_hashes
          output_admission_sha256 = Digest::SHA256.hexdigest(File.read(cli.admission_receipt))
          seconds = (Time.monotonic - request_started).total_seconds
          receipt = {schema: "wan_s2v_run_receipt_v1", status: mode,
                     request: request + 1, requests: cli.requests,
                     run_kind: request == 0 ? "cold_process" : "retained_pool_repeat",
                     setup_seconds: request == 0 ? setup_seconds : 0.0,
                     request_seconds: seconds, setup_plus_request_seconds: seconds + (request == 0 ? setup_seconds : 0.0),
                     timing_boundary: "request starts before admission and per-request I/O planning; ends after raw output write and receipt hashes; excludes receipt serialization and resource release",
                     complete_process_wall_time: "external timer only; this receipt covers one request",
                     seconds_per_video_second: (seconds + (request == 0 ? setup_seconds : 0.0)) / 5.0,
                     gpu_segment_seconds_sum: active.gpu_elapsed_seconds, command_segments: active.segment_count,
                     terminal_host_waits: active.wait_count, internal_layer_or_operator_waits: 0,
                     neural_readbacks_between_stages: 0, final_output_readbacks: 1,
                     diagnostic_counter_readbacks: 1, nonfinite_values: 0,
                     checkpoint_load_stages: io.stage_count, checkpoint_load_batches: io.batch_count,
                     checkpoint_load_batch_limit_bytes: io.max_batch_bytes, output: output_path,
                     output_bytes: bytes.size, output_sha256: output_sha256,
                     first_request_output_sha256: first_output_sha256,
                     matches_first_request_bitwise: output_sha256 == first_output_sha256,
                     input_manifest_sha256: output_manifest_sha256,
                     initial_noise_sha256: input.tensor("noise.f32").sha256,
                     seed: input.manifest["seed"].as_i64,
                     noise_generation: "precomputed PyTorch CPU F32; exact input tensor reused",
                     filesystem_cache_state: "uncontrolled; cold means a new process and prepared resources",
                     fps: 16, frames: DELIVER_FRAMES, height: DELIVER_HEIGHT, width: CANVAS_WIDTH,
                     working_height: CANVAS_HEIGHT, minimum_system_free_pct: 15,
                     source_revision: git_revision, implementation_sha256: output_implementation_sha256,
                     admission_receipt_sha256: output_admission_sha256,
                     finite_drift_is_not_strict_parity: mode == "experimental_run_admitted"}
          File.open(output_path + ".json", "wx") { |file| file.puts(receipt.to_json) }
          puts receipt.to_json
          bytes = Bytes.empty
        rescue ex
          request_error = ex
          raise ex
        ensure
          begin
            if active = chain
              begin
                drain_request!(active, io)
              ensure
                resources_drained = active.completed? && io.finished?
                io.release if resources_drained
              end
            else
              io.release
            end
          rescue cleanup_error
            # Keep the first execution failure visible. Cleanup can report
            # pending I/O after a timeout; owners stay pinned unless drained.
            raise cleanup_error unless request_error
            STDERR.puts "Request cleanup also failed: #{cleanup_error.message}"
          end
        end
      end
    ensure
      # A standalone failed run exits immediately. If completion could not be
      # established, retain the native owners and mmap until process teardown
      # rather than explicitly freeing storage under pending GPU/I/O work.
      if resources_drained
        attention_workspace.try(&.release)
        negative_workspace.try(&.release_after_terminal_wait!)
        positive_workspace.try(&.release_after_terminal_wait!)
        linear_stage.try(&.release_after_terminal_wait!)
        position_plan.try(&.release)
        gelu_plans.try(&.release)
        normalizer.try(&.release)
        xlsr_plans.try(&.release)
        t5_plans.try(&.release)
        stages.each_value(&.close_sources!)
        stages.each_value { |stage| stage.arena.release }
        phase_scratch.each_value(&.release)
        phase_weights.each_value do |arena|
          arena.release unless arena.buffers.empty? && arena.resident_bytes == 0
        end
        block_scratch.try(&.release)
        audio_scratch.try(&.release)
        scheduler_scratch.try(&.release)
        activation_scratch.try(&.release)
        state.release
        constants.release
        stage_io.release
        pool.release if pool.live_view_count == 0
        RUN_RESOURCES.delete(owners)
      else
        STDERR.puts "Retaining GPU/I/O owners until process teardown: terminal drain was not established"
      end
    end
  end

  def self.probe!(cli : WanS2VCLI) : Nil
    probe_dit_block!(cli)
  end
end

cli = WanS2VCLI.new
begin
  cli.parse!(ARGV)
  if cli.help
    puts "Usage: crystal run examples/wan_s2v_pipeline.cr -- [--plan | --memory-plan --checkpoint-root=DIR --budget-mib=N --reserve-mib=N | --validate-receipt=FILE --inputs=DIR --checkpoint-root=DIR [--allow-recorded-numerical-drift] | --probe-dit-block --projection-backend=scalar|mps --checkpoint-root=DIR --budget-mib=N --reserve-mib=N | --run --inputs=DIR --checkpoint-root=DIR --output=FILE --admission-receipt=FILE --budget-mib=N --reserve-mib=N --projection-backend=scalar|mps]"
    exit 0
  end
  if !cli.validate_receipt.empty?
    WanS2VRunner.validate_receipt!(cli)
  elsif cli.memory_plan
    WanS2VRunner.print_memory_plan(cli)
  elsif cli.probe_dit_block
    WanS2VRunner.probe!(cli)
  elsif cli.run
    WanS2VRunner.run!(cli)
  else
    WanS2VRunner.print_plan(cli)
  end
rescue ex
  STDERR.puts "wan_s2v_pipeline: #{ex.message}"
  exit 2
end
