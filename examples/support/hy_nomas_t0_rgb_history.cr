require "digest/sha256"
require "json"
require "../../src/ml/video/bf16_weight_cache"
require "../../src/ml/video/dcae_v_encoder"
require "./hy_nomas_generation"
require "./hy_nomas_sequence"
require "./hy_nomas_probe_instrumentation"

module HyNomasT0RgbHistory
  extend self

  ROOT                   = "/opt/mldata_ssd/cogni-ml/hy-nomas-native-context-20261106.ui7736ue"
  RUN_ROOT               = File.join(ROOT, "moving-static-dcae-1b2d7289")
  CONTINUATION_PATH      = File.join(ROOT, "photo-continuation-v2.json")
  CONTINUATION_SHA256    = "0f35bb09964f2b28d28069e8d39a8a1d6194607d4269de6bb824ff54d828e2ac"
  QWEN_INPUTS_SHA256     = "31528051793c3a184fc91a4c8d0bf45debab8cf38e6dd16d54fea780c3dd99eb"
  SOURCE_PIXELS_PATH     = File.join(ROOT, "photo-generated-t0-v1/native-decoder-generated-v1/pixels.f32le")
  SOURCE_PIXELS_SHA256   = "83f045adf0b94e7311f7a5b3bab757e4d178c80d3c43f0a4a17bbf06bf9b310b"
  SOURCE_RECEIPT_PATH    = File.join(ROOT, "photo-generated-t0-v1/native-decoder-generated-v1/receipt.json")
  SOURCE_RECEIPT_SHA256  = "7e5a3ee991fdb3bbcb6604ef0008ae2254e96f8161d4c81053df77e9b0d7af07"
  SOURCE_BYTES           = 637_009_920_i64
  SOURCE_FRAMES          = 360_i32
  OUTPUT_FRAMES          = 120_i32
  CHANNELS               = 3_i32
  HEIGHT                 = 288_i32
  WIDTH                  = 512_i32
  SOURCE_SHAPE           = [1_i32, CHANNELS, SOURCE_FRAMES, HEIGHT, WIDTH]
  CLIP_SHAPE             = [1_i32, CHANNELS, OUTPUT_FRAMES, HEIGHT, WIDTH]
  CONFIG                 = ML::Video::DCAEVEncoderConfig.new(HEIGHT, WIDTH, OUTPUT_FRAMES)
  ENCODER_WEIGHTS_PATH   = "/opt/mldata_ssd/models/hy-nomas/dcae-v-encoder-bf16.safetensors"
  ENCODER_WEIGHTS_SHA256 = "d0bbacbebe8960b147388d0bf61ec8bbc0eb8d575574d6ec5c5abe4878c8d6c1"
  SCHEMA                 = "cogni_ml.hy_nomas_t0_rgb_history_probe/v1"
  MEMORY_SCOPE           = "MPSGraph private workspace is opaque to MetalBuffer accounting; the process memory guard remains authoritative"

  KERNEL_SOURCE = <<-METAL
  #include <metal_stdlib>
  using namespace metal;

  inline ushort hy_t0_rgb_f32_to_bf16(float value) {
      const uint bits = as_type<uint>(value);
      const uint magnitude = bits & 0x7fffffffu;
      if (magnitude > 0x7f800000u) {
          return ushort((bits >> 16) | 0x0040u);
      }
      return ushort((bits + 0x7fffu + ((bits >> 16) & 1u)) >> 16);
  }

  kernel void hy_nomas_t0_rgb_planar_gather_normalize_bf16(
      device const float *source [[buffer(0)]],
      device ushort *clip [[buffer(1)]],
      constant uint &source_frames [[buffer(2)]],
      constant uint &output_frames [[buffer(3)]],
      constant uint &height [[buffer(4)]],
      constant uint &width [[buffer(5)]],
      constant uint &repeat_last [[buffer(6)]],
      uint id [[thread_position_in_grid]]) {
      const uint spatial = height * width;
      const uint count = 3u * output_frames * spatial;
      if (id >= count) return;
      const uint channel = id / (output_frames * spatial);
      const uint frame = (id / spatial) % output_frames;
      const uint pixel = id % spatial;
      const uint source_frame = repeat_last != 0 ? output_frames - 1u : frame;
      const uint source_index = (channel * source_frames + source_frame) * spatial + pixel;
      const float bounded = clamp(source[source_index], 0.0f, 1.0f);
      clip[id] = hy_t0_rgb_f32_to_bf16(2.0f * bounded - 1.0f);
  }
  METAL

  record ResidencyPlan,
    weights_bytes : Int64,
    workspace_bytes : Int64,
    clip_bytes : Int64,
    source_bytes : Int64,
    latent_bytes : Int64,
    named_bytes : Int64 do
    def self.for(cfg : ML::Video::DCAEVEncoderConfig) : self
      HyNomasT0RgbHistory.validate_source_geometry!("F32LE", SOURCE_SHAPE, SOURCE_BYTES)
      weights = ML::Video::DCAEVEncoder.weight_bytes(cfg)
      workspace = ML::Video::DCAEVEncoder.workspace_required_bytes(cfg)
      clip = cfg.input_bytes
      source = SOURCE_BYTES
      latent = cfg.output_bytes
      total = [weights, workspace, clip, source, latent].reduce(0_i64) do |sum, bytes|
        unless bytes >= 0 && bytes <= Int64::MAX - sum
          raise ArgumentError.new("DCAE T0 RGB named-byte plan overflows")
        end
        sum + bytes
      end
      new(weights, workspace, clip, source, latent, total)
    end
  end

  record CombinedPlan,
    dcae : ResidencyPlan,
    dit_peak_bytes : Int64,
    dit_steady_bytes : Int64,
    connected_steady_bytes : Int64,
    connected_peak_bytes : Int64,
    named_budget_bytes : Int64 do
    def self.for(inputs : HyNomasGeneration::TimelineInputs, budget_bytes : Int64) : self
      dcae = ResidencyPlan.for(CONFIG)
      dit_peak = HyNomasSequence.peak_named_bytes(inputs, 2,
        reference_latent_bytes: HyNomasSequence::REFERENCE_LATENT_BYTES,
        source_bf16: true)
      dit_steady = dit_peak - HyNomasSequence.old_attention_bytes(inputs.base)
      unless dit_steady >= 0 && dcae.latent_bytes == HyNomasSequence::REFERENCE_LATENT_BYTES
        raise ArgumentError.new("DCAE and source-BF16 DiT T0 latent plans disagree")
      end
      # The DCAE endpoint is the same buffer borrowed by DiT; count it once.
      connected_steady = dit_steady + dcae.named_bytes - dcae.latent_bytes
      # DiT is set up before DCAE buffers, so its old-attention replacement
      # peak does not overlap the connected encoder-plus-steady-DiT phase.
      connected_peak = {dit_peak, connected_steady}.max
      HyNomasT0RgbHistory.validate_named_budget!(connected_peak, budget_bytes)
      new(dcae, dit_peak, dit_steady, connected_steady, connected_peak, budget_bytes)
    end
  end

  def validate_context_mode!(mode : String) : String
    unless {"moving", "repeat-last"}.includes?(mode)
      raise ArgumentError.new("T0 RGB context mode must be moving or repeat-last")
    end
    mode
  end

  def source_frame(mode : String, output_frame : Int32) : Int32
    validate_context_mode!(mode)
    unless 0 <= output_frame < OUTPUT_FRAMES
      raise ArgumentError.new("T0 RGB output frame is outside the 120-frame DCAE clip")
    end
    mode == "moving" ? output_frame : OUTPUT_FRAMES - 1
  end

  def validate_source_geometry!(dtype : String, shape : Array(Int32), bytes : Int64) : Bool
    unless dtype == "F32LE"
      raise ArgumentError.new("T0 RGB source dtype must be F32LE")
    end
    unless shape == SOURCE_SHAPE
      raise ArgumentError.new("T0 RGB source shape must be [1,3,360,288,512] planar CTHW")
    end
    unless bytes == SOURCE_BYTES
      raise ArgumentError.new("T0 RGB source extent must be exactly 637009920 bytes")
    end
    true
  end

  def validate_source_binding!(path : String, expected_path : String,
                               actual_bytes : Int64, expected_sha256 : String,
                               pinned_bytes : Int64, actual_sha256 : String) : Bool
    unless Path[path].absolute? && path == expected_path
      raise ArgumentError.new("T0 RGB source must use the pinned source path")
    end
    unless actual_bytes == pinned_bytes && pinned_bytes == SOURCE_BYTES
      raise ArgumentError.new("T0 RGB source byte extent differs from the pinned F32 bank")
    end
    unless valid_sha256?(expected_sha256) && actual_sha256 == expected_sha256 &&
           expected_sha256 == SOURCE_PIXELS_SHA256
      raise ArgumentError.new("T0 RGB source SHA-256 differs from the pinned bank")
    end
    true
  end

  def validate_named_budget!(named_bytes : Int64, budget_bytes : Int64) : Nil
    unless budget_bytes > 0 && named_bytes >= 0 && named_bytes <= budget_bytes
      raise ArgumentError.new("connected T0 RGB named plan exceeds budget")
    end
  end

  def validate_encode_metrics!(allocations : Int64, readbacks : Int64,
                               preterminal_waits : Int64, terminal_waits : Int64) : Nil
    unless allocations == 0 && readbacks == 0 && preterminal_waits == 0 && terminal_waits == 1
      raise ArgumentError.new("connected T0 RGB chain requires zero preterminal allocations/readbacks/waits and one terminal wait")
    end
  end

  def validate_preterminal_metrics!(allocations : Int64, readbacks : Int64,
                                    waits : Int64) : Nil
    unless allocations == 0 && readbacks == 0 && waits == 0
      raise ArgumentError.new("connected T0 RGB encode requires zero preterminal allocations/readbacks/waits")
    end
  end

  def validate_expected_clip_sha256!(sha256 : String) : String
    unless valid_sha256?(sha256)
      raise ArgumentError.new("expected normalized clip SHA-256 must be lowercase hex")
    end
    sha256
  end

  def validate_actual_clip_sha256!(actual : String, expected : String) : String
    validate_expected_clip_sha256!(expected)
    unless valid_sha256?(actual) && actual == expected
      raise ArgumentError.new("GPU normalized clip differs from the pinned source-derived BF16 bytes")
    end
    actual
  end

  def bf16_f32_le_bytes(bits : UInt16) : Tuple(UInt8, UInt8, UInt8, UInt8)
    f32 = bits.to_u32 << 16
    {
      (f32 & 0xff_u32).to_u8,
      ((f32 >> 8) & 0xff_u32).to_u8,
      ((f32 >> 16) & 0xff_u32).to_u8,
      ((f32 >> 24) & 0xff_u32).to_u8,
    }
  end

  def valid_sha256?(sha : String) : Bool
    sha.matches?(/\A[0-9a-f]{64}\z/)
  end

  def sha_file(path : String) : String
    digest = Digest::SHA256.new
    File.open(path) do |io|
      buffer = Bytes.new(1024 * 1024)
      while (count = io.read(buffer)) > 0
        digest.update(buffer[0, count])
      end
    end
    digest.hexfinal
  end

  def canonical_file!(path : String, label : String) : String
    unless Path[path].absolute? && File.file?(path) && !File.symlink?(path)
      raise ArgumentError.new("#{label} must be an absolute regular non-symlink file")
    end
    File.realpath(path)
  end

  def authenticate_source!(source_path : String = SOURCE_PIXELS_PATH,
                           receipt_path : String = SOURCE_RECEIPT_PATH) : NamedTuple(
    pixels_path: String, pixels_sha256: String,
    receipt_path: String, receipt_sha256: String)
    pixels_path = canonical_file!(source_path, "native PixelUMM T0 RGB source")
    receipt_path = canonical_file!(receipt_path, "native PixelUMM source receipt")
    actual_sha = sha_file(pixels_path)
    validate_source_binding!(pixels_path, File.realpath(SOURCE_PIXELS_PATH), File.size(pixels_path),
      SOURCE_PIXELS_SHA256, SOURCE_BYTES, actual_sha)
    receipt_bytes = File.read(receipt_path)
    receipt_sha = Digest::SHA256.hexdigest(receipt_bytes)
    unless receipt_sha == SOURCE_RECEIPT_SHA256
      raise ArgumentError.new("native PixelUMM T0 source receipt SHA-256 mismatch")
    end
    receipt = JSON.parse(receipt_bytes)
    unless receipt["schema"].as_s == "cogni_ml.hy_nomas_decode_latent/v1" &&
           receipt["upstream_revision"].as_s == "911cea6458a89c2663513ff7099561ec190dd491" &&
           receipt["inputs_sha256"].as_s == QWEN_INPUTS_SHA256 &&
           receipt["continuation_manifest_sha256"].as_s == CONTINUATION_SHA256 &&
           receipt["pixel_noise_manifest_sha256"].as_s == CONTINUATION_SHA256 &&
           receipt["output_dtype"].as_s == "F32 little-endian planar RGB" &&
           receipt["output_bytes"].as_i64 == SOURCE_BYTES &&
           receipt["output_sha256"].as_s == SOURCE_PIXELS_SHA256 &&
           receipt["output_shape"].as_a.map { |value| value.as_i.to_i32 } == SOURCE_SHAPE
      raise ArgumentError.new("native PixelUMM source receipt provenance or geometry mismatch")
    end
    {pixels_path: pixels_path, pixels_sha256: SOURCE_PIXELS_SHA256,
     receipt_path: receipt_path, receipt_sha256: receipt_sha}
  end

  def authenticate_continuation!(path : String = CONTINUATION_PATH,
                                 sha256 : String = CONTINUATION_SHA256) : HyNomasGeneration::TimelineInputs
    unless sha256 == CONTINUATION_SHA256
      raise ArgumentError.new("T0 RGB history requires the pinned frozen continuation manifest")
    end
    inputs = HyNomasGeneration::TimelineInputs.new(File.realpath(path), sha256)
    inputs.verify_payload!
    inputs.base.file.verify_payload!
    HyNomasSequence.validate_shared_scene_prompt!(inputs.base.prompt)
    unless inputs.base.file.sha256 == QWEN_INPUTS_SHA256
      raise ArgumentError.new("T0 RGB history Qwen input identity differs from the frozen T1 bundle")
    end
    inputs
  end

  def output_dir!(path : String) : String
    unless Path[path].absolute? && File.directory?(path) && !File.symlink?(path) && Dir.empty?(path)
      raise ArgumentError.new("output directory must be an existing empty absolute directory")
    end
    real = File.realpath(path)
    unless real.starts_with?(File.realpath(RUN_ROOT) + "/")
      raise ArgumentError.new("T0 RGB outputs must remain beneath the authorized experiment directory")
    end
    real
  end

  def frozen_input_identity(inputs : HyNomasGeneration::TimelineInputs) : NamedTuple(
    t0_initial_noise_sha256: String, t1_initial_noise_sha256: String,
    scene_qwen_bf16_sha256: String, t0_rope_cos_sha256: String,
    t0_rope_sin_sha256: String, t1_rope_cos_sha256: String,
    t1_rope_sin_sha256: String, t0_cache_epsilon_bf16_sha256: String)
    bank = inputs.records["input.latent_bank"]
    bank_channel_bytes = 180_i64 * 9 * 16 * sizeof(Float32)
    segment_channel_bytes = 30_i64 * 9 * 16 * sizeof(Float32)
    slice_sha = ->(start : Int64) do
      digest = Digest::SHA256.new
      File.open(inputs.file.payload) do |io|
        32.times do |channel|
          io.seek(bank.offset + channel.to_i64 * bank_channel_bytes + start * 9 * 16 * sizeof(Float32))
          bytes = Bytes.new(segment_channel_bytes.to_i32)
          io.read_fully(bytes)
          digest.update(bytes)
        end
      end
      digest.hexfinal
    end
    rope_cos = inputs.records["input.rope_cos"]
    rope_sin = inputs.records["input.rope_sin"]
    cache_epsilon = inputs.records["cache.eps_bf16"]
    rope_segment_bytes = 4320_i64 * 128 * sizeof(Float32)
    epsilon_segment_bytes = 32_i64 * 30 * 9 * 16 * sizeof(UInt16)
    {
      t0_initial_noise_sha256:      slice_sha.call(0_i64),
      t1_initial_noise_sha256:      slice_sha.call(30_i64),
      scene_qwen_bf16_sha256:       inputs.base.records["input.qwen_bf16"].sha256,
      t0_rope_cos_sha256:           HyNomasGeneration.sha_range(inputs.file.payload, rope_cos.offset, rope_segment_bytes),
      t0_rope_sin_sha256:           HyNomasGeneration.sha_range(inputs.file.payload, rope_sin.offset, rope_segment_bytes),
      t1_rope_cos_sha256:           HyNomasGeneration.sha_range(inputs.file.payload, rope_cos.offset + rope_segment_bytes, rope_segment_bytes),
      t1_rope_sin_sha256:           HyNomasGeneration.sha_range(inputs.file.payload, rope_sin.offset + rope_segment_bytes, rope_segment_bytes),
      t0_cache_epsilon_bf16_sha256: HyNomasGeneration.sha_range(inputs.file.payload, cache_epsilon.offset, epsilon_segment_bytes),
    }
  end

  def write_fresh_buffer!(path : String, buffer : ML::MetalBuffer, expected_bytes : Int64) : String
    unless buffer.valid? && buffer.size == expected_bytes
      raise ArgumentError.new("T0 RGB output buffer extent mismatch")
    end
    bytes = Bytes.new(buffer.contents.as(UInt8*), expected_bytes.to_i32)
    digest = Digest::SHA256.hexdigest(bytes)
    HyNomasGeneration.write_fresh_output(path) { |io| io.write(bytes) }
    digest
  end

  def write_latent_f32!(path : String, latent_bf16 : ML::MetalBuffer) : String
    unless latent_bf16.valid? && latent_bf16.size == CONFIG.output_bytes
      raise ArgumentError.new("DCAE BF16 endpoint extent mismatch")
    end
    source = Bytes.new(latent_bf16.contents.as(UInt8*), latent_bf16.size.to_i32)
    # Prepare this small post-terminal export before entering the writer block.
    # Crystal 1.22.1 cannot codegen the conversion loop inside its typed yield.
    bytes = Bytes.new((CONFIG.output_bytes * 2).to_i32)
    (source.size // 2).times do |item|
      bf16 = (source[item * 2].to_u32 |
              (source[item * 2 + 1].to_u32 << 8)).to_u16
      f32 = bf16_f32_le_bytes(bf16)
      offset = item * 4
      bytes[offset] = f32[0]
      bytes[offset + 1] = f32[1]
      bytes[offset + 2] = f32[2]
      bytes[offset + 3] = f32[3]
    end
    digest = Digest::SHA256.hexdigest(bytes)
    HyNomasGeneration.write_fresh_output(path) { |io| io.write(bytes) }
    digest
  end

  def logical_prefix_digests(bank : Bytes) : NamedTuple(t0_sha256: String, t1_sha256: String, prefix60_sha256: String)
    bank_bytes = 32_i64 * 180 * 9 * 16 * sizeof(Float32)
    unless bank.size.to_i64 == bank_bytes
      raise ArgumentError.new("T0 RGB continuation output bank extent mismatch")
    end
    per_channel = 180_i64 * 9 * 16 * sizeof(Float32)
    half = 30_i64 * 9 * 16 * sizeof(Float32)
    t0 = Digest::SHA256.new
    t1 = Digest::SHA256.new
    prefix = Digest::SHA256.new
    32.times do |channel|
      base = channel.to_i64 * per_channel
      t0_slice = bank[base.to_i, half.to_i]
      t1_slice = bank[(base + half).to_i, half.to_i]
      t0.update(t0_slice)
      t1.update(t1_slice)
      prefix.update(t0_slice)
      prefix.update(t1_slice)
    end
    {t0_sha256: t0.hexfinal, t1_sha256: t1.hexfinal, prefix60_sha256: prefix.hexfinal}
  end

  class Session
    getter named_bytes : Int64
    getter residency : ResidencyPlan
    @weights_file : ML::Video::CheckpointFile?
    @weight_plan : ML::Video::WeightPlan?
    @weights_arena : ML::Video::Arena? = nil
    @workspace_arena : ML::Video::Arena? = nil
    @clip_arena : ML::Video::Arena? = nil
    @source : ML::MetalBuffer? = nil
    @clip : ML::MetalBuffer? = nil
    @latent : ML::MetalBuffer
    @workspace : ML::Video::DCAEVEncoderWorkspace? = nil
    @mps_plans : ML::Video::DCAEVEncoderMPSPlanSet? = nil
    @prepared : ML::Video::PreparedWeights? = nil
    @pipeline : ML::Metal::ComputePipeline? = nil
    @uploaded_source_sha256 : String? = nil
    getter uploaded_source_sha256 : String? = nil
    @prepared_for_encode = false
    @encode_started = false
    @released = false

    def encode_started? : Bool
      @encode_started
    end

    def initialize(@latent : ML::MetalBuffer, weights_path : String,
                   weights_sha256 : String, @budget_bytes : Int64)
      unless @latent.valid? && @latent.size == CONFIG.output_bytes
        raise ArgumentError.new("T0 RGB DCAE latent must be one exact-size shared BF16 endpoint")
      end
      raise ArgumentError.new("T0 RGB encoder budget must be positive") unless @budget_bytes > 0
      unless weights_sha256 == ENCODER_WEIGHTS_SHA256
        raise ArgumentError.new("T0 RGB encoder weights must use the pinned BF16 checkpoint")
      end
      @residency = ResidencyPlan.for(CONFIG)
      @named_bytes = @residency.named_bytes
      weights_file = ML::Video::BF16WeightCache.authenticate(weights_path, weights_sha256)
      begin
        bindings = ML::Video::DCAEVEncoder.checkpoint_bindings(CONFIG)
        weight_plan = ML::Video::WeightPlan.new([weights_file], bindings, @named_bytes,
          reserve_bytes: @named_bytes - @residency.weights_bytes,
          resident_dtype: ML::Video::WeightDType::BF16, require_all: true)
        unless weight_plan.staging_bytes == 0 && bindings.size == 169 &&
               weight_plan.destination_bytes == @residency.weights_bytes &&
               weight_plan.peak_bytes == @named_bytes
          raise ArgumentError.new("T0 RGB encoder weight inventory differs from the pinned DCAE plan")
        end
        @weight_plan = weight_plan
        @weights_file = weights_file
      rescue ex
        weights_file.close
        raise ex
      end
    end

    def prepare! : Nil
      raise ArgumentError.new("T0 RGB DCAE session is released") if @released
      raise ArgumentError.new("T0 RGB DCAE session is already prepared") if @prepared_for_encode
      begin
        @weights_arena = ML::Video::Arena.new
        @workspace_arena = ML::Video::Arena.new
        @clip_arena = ML::Video::Arena.new
        @source = ML::MetalBuffer.new(@residency.source_bytes)
        @prepared = ML::Video::PreparedWeights.new(@weight_plan.not_nil!, @weights_arena.not_nil!)
        @workspace = ML::Video::DCAEVEncoder.prepare!(@workspace_arena.not_nil!, CONFIG,
          @named_bytes, @named_bytes - @residency.workspace_bytes)
        @clip = @clip_arena.not_nil!.alloc_weight("t0.rgb.normalized_clip_bf16",
          CONFIG.input_values, ML::Video::WeightDType::BF16)
        @pipeline = ML::Metal::ComputePipeline.new("hy_nomas_t0_rgb_planar_gather_normalize_bf16", KERNEL_SOURCE)
        actual = @weights_arena.not_nil!.resident_bytes +
                 @workspace_arena.not_nil!.resident_bytes +
                 @clip_arena.not_nil!.resident_bytes +
                 @source.not_nil!.size + @latent.size
        unless actual == @named_bytes && actual <= @budget_bytes
          raise ArgumentError.new("T0 RGB DCAE setup differs from its admitted named residency")
        end
        @mps_plans = ML::Video::DCAEVEncoder.prepare_mps!(@clip.not_nil!, @latent,
          @weights_arena.not_nil!, @workspace.not_nil!, CONFIG)
        @prepared_for_encode = true
      rescue ex
        release_setup_owners!
        raise ex
      end
    end

    def upload_source!(path : String = SOURCE_PIXELS_PATH) : Nil
      raise ArgumentError.new("T0 RGB DCAE session is not prepared") unless @prepared_for_encode
      info = File.info(path)
      unless File.file?(path) && !File.symlink?(path) && File.realpath(path) == File.realpath(SOURCE_PIXELS_PATH) &&
             info.size == SOURCE_BYTES
        raise ArgumentError.new("T0 RGB F32 source changed before GPU upload")
      end
      bytes = Bytes.new(@source.not_nil!.contents.as(UInt8*), SOURCE_BYTES.to_i32)
      File.open(path) { |io| io.read_fully(bytes) }
      after = File.info(path)
      actual_sha = Digest::SHA256.hexdigest(bytes)
      unless info.same_file?(after) && info.size == after.size &&
             info.modification_time == after.modification_time &&
             actual_sha == SOURCE_PIXELS_SHA256
        raise ArgumentError.new("uploaded T0 RGB source bytes differ from the pinned F32 bank")
      end
      @uploaded_source_sha256 = actual_sha
    end

    def normalized_clip : ML::MetalBuffer
      @clip || raise ArgumentError.new("T0 RGB normalized clip is unavailable")
    end

    def latent : ML::MetalBuffer
      @latent
    end

    def encode!(chain : ML::Metal::MPSCommandChain, mode : String) : Nil
      raise ArgumentError.new("T0 RGB DCAE session is not ready") unless @prepared_for_encode && @mps_plans && !@released
      raise ArgumentError.new("T0 RGB DCAE may run only once") if @encode_started
      unless !chain.drained? && chain.wait_count == 0
        raise ArgumentError.new("T0 RGB DCAE requires a fresh undrained command chain")
      end
      repeat = HyNomasT0RgbHistory.validate_context_mode!(mode) == "repeat-last"
      @encode_started = true
      chain.encoder.set_pipeline(@pipeline.not_nil!)
        .set_buffer(@source.not_nil!, 0)
        .set_buffer(@clip.not_nil!, 1, ML::Metal::BufferAccess::Write)
        .set_value(SOURCE_FRAMES.to_u32, 2)
        .set_value(OUTPUT_FRAMES.to_u32, 3)
        .set_value(HEIGHT.to_u32, 4)
        .set_value(WIDTH.to_u32, 5)
        .set_value(repeat ? 1_u32 : 0_u32, 6)
        .dispatch_1d(CONFIG.input_values)
      ML::Video::DCAEVEncoder.encode(chain.encoder, @clip.not_nil!, @latent,
        @weights_arena.not_nil!, @workspace.not_nil!, CONFIG, chain, nil,
        @mps_plans.not_nil!)
    end

    def release_after_drain!(chain : ML::Metal::MPSCommandChain) : Nil
      unless @encode_started && chain.drained? && chain.wait_count == 1
        raise ArgumentError.new("T0 RGB DCAE release requires the one connected terminal drain")
      end
      release_setup_owners!(chain)
      @weights_file.try(&.close)
      @weights_file = nil
      @weight_plan = nil
      @released = true
    end

    def close_before_encode! : Nil
      raise ArgumentError.new("T0 RGB DCAE cannot close before the connected terminal drain") if @encode_started
      release_setup_owners!
      @weights_file.try(&.close)
      @weights_file = nil
      @weight_plan = nil
      @released = true
    end

    private def release_setup_owners!(chain : ML::Metal::MPSCommandChain? = nil) : Nil
      if plans = @mps_plans
        plans.release(chain)
        @mps_plans = nil
      end
      @prepared.try(&.release)
      @prepared = nil
      @clip_arena.try(&.release)
      @clip_arena = nil
      @workspace_arena.try(&.release)
      @workspace_arena = nil
      @weights_arena.try(&.release)
      @weights_arena = nil
      @source.try(&.release)
      @source = nil
      @workspace = nil
      @clip = nil
      @pipeline = nil
      @prepared_for_encode = false
    end
  end
end
