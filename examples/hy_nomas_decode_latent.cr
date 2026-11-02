require "option_parser"
require "digest/sha256"
require "json"
require "./support/hy_nomas_generation"
require "./support/hy_nomas_timeline_inputs"
require "./support/hy_nomas_decode_latent_admission"
require "../src/ml/video/hy_nomas_timeline_ops"

module HyNomasLatentDecode
  extend self

  BASE_INPUTS_PATH         = "/opt/mldata_ssd/cogni-ml/hy-nomas-echo-20261030/conditioning/inputs-echo-20s.json"
  BASE_INPUTS_SHA          = "f5001fcbc786401488de1eb652061899ae4fdda88febb67ebd8bd6a22e569bcb"
  CONTINUATION_INPUTS_PATH = "/opt/mldata_ssd/cogni-ml/hy-nomas-echo-20261030/conditioning/continuation-inputs-echo-20s.json"
  CONTINUATION_INPUTS_SHA  = "2807e11814f07975c807f367b0a4f15e3c6bdde1eb62ce9bda851814d09a6e47"
  DEFAULT_LATENT_FRAMES    =  30
  LONG_LATENT_FRAMES       =  60
  REPEATED_LATENT_FRAMES   =  90
  LATENT_BANK_FRAMES       = 180
  LATENT_CHANNELS          =  32
  LATENT_HEIGHT            =   9
  LATENT_WIDTH             =  16
  NOISE_CHANNELS           =   3
  NOISE_FRAMES_PER_LATENT  =   4
  NOISE_HEIGHT             = 288
  NOISE_WIDTH              = 512

  def latent_shape(frames : Int32) : Array(Int32)
    [1, LATENT_CHANNELS, frames, LATENT_HEIGHT, LATENT_WIDTH]
  end

  def latent_elements(frames : Int32) : Int64
    LATENT_CHANNELS.to_i64 * frames * LATENT_HEIGHT * LATENT_WIDTH
  end

  def latent_bytes(frames : Int32) : Int64
    latent_elements(frames) * sizeof(Float32)
  end

  def pixel_noise_frames(latent_frames : Int32) : Int32
    latent_frames * NOISE_FRAMES_PER_LATENT
  end

  def pixel_noise_elements(latent_frames : Int32) : Int64
    NOISE_CHANNELS.to_i64 * pixel_noise_frames(latent_frames) * NOISE_HEIGHT * NOISE_WIDTH
  end

  def validate_latent_frames!(frames : Int32) : Nil
    unless {DEFAULT_LATENT_FRAMES, LONG_LATENT_FRAMES, REPEATED_LATENT_FRAMES}.includes?(frames)
      raise ArgumentError.new("diagnostic latent frames must be 30 or the opt-in 60/90")
    end
  end

  def validate_latent_bank_frames!(latent_frames : Int32, bank_frames : Int32) : Nil
    return if bank_frames == latent_frames
    if {LONG_LATENT_FRAMES, REPEATED_LATENT_FRAMES}.includes?(latent_frames) && bank_frames == LATENT_BANK_FRAMES
      return
    end
    raise ArgumentError.new("latent bank frames must equal --latent-frames, or be 180 for a 60/90-frame decode")
  end

  # The continuation payload digest authenticates this exact range; scan it
  # separately for finite F32 values before uploading the full noise bank.
  def validate_pixel_noise_prefix_finite!(inputs : HyNomasGeneration::TimelineInputs,
                                          latent_frames : Int32) : Nil
    record = inputs.records["decoder.noise"]
    expected_shape = [1, NOISE_CHANNELS, 720, NOISE_HEIGHT, NOISE_WIDTH]
    unless record.dtype == "F32" && record.shape == expected_shape
      raise ArgumentError.new("authenticated continuation pixel-noise source geometry mismatch")
    end

    prefix_frame_bytes = pixel_noise_frames(latent_frames).to_i64 * NOISE_HEIGHT * NOISE_WIDTH * sizeof(Float32)
    source_channel_bytes = 720_i64 * NOISE_HEIGHT * NOISE_WIDTH * sizeof(Float32)
    scratch = Bytes.new(1024 * 1024)
    NOISE_CHANNELS.times do |channel|
      File.open(inputs.file.payload) do |io|
        io.seek(record.offset + channel.to_i64 * source_channel_bytes)
        remaining = prefix_frame_bytes
        channel_offset = 0_i64
        while remaining > 0
          byte_count = Math.min(remaining, scratch.size.to_i64).to_i32
          byte_count -= byte_count % sizeof(UInt32)
          chunk = scratch[0, byte_count]
          io.read_fully(chunk)
          words = scratch.to_unsafe.as(UInt32*)
          (byte_count // sizeof(UInt32)).times do |word_index|
            if (words[word_index] & 0x7f80_0000_u32) == 0x7f80_0000_u32
              index = channel.to_i64 * (prefix_frame_bytes // sizeof(Float32)) +
                      channel_offset // sizeof(Float32) + word_index
              raise ArgumentError.new("pixel noise contains non-finite F32 at element #{index}")
            end
          end
          remaining -= byte_count
          channel_offset += byte_count
        end
      end
    end
  end

  def pixel_noise_shape(latent_frames : Int32) : Array(Int32)
    [1, NOISE_CHANNELS, pixel_noise_frames(latent_frames), NOISE_HEIGHT, NOISE_WIDTH]
  end

  class RetainedResources
    def initialize(@inputs : HyNomasGeneration::Inputs,
                   @metadata : HyNomasGeneration::DecoderMetadata,
                   @decoder : HyNomasGeneration::Decoder?,
                   @chain : ML::Metal::MPSCommandChain?,
                   @shared : ML::Video::Arena?,
                   @timeline_inputs : HyNomasGeneration::TimelineInputs?,
                   @noise_prefix_plan : ML::Video::HyNomasTimelineOps::PrefixPlan?,
                   @latent_prefix_plan : ML::Video::HyNomasTimelineOps::PrefixPlan?)
    end
  end

  @@unreleased = [] of RetainedResources

  def retain_until_process_exit!(inputs : HyNomasGeneration::Inputs,
                                 metadata : HyNomasGeneration::DecoderMetadata,
                                 decoder : HyNomasGeneration::Decoder?,
                                 chain : ML::Metal::MPSCommandChain?,
                                 shared : ML::Video::Arena?,
                                 timeline_inputs : HyNomasGeneration::TimelineInputs?,
                                 noise_prefix_plan : ML::Video::HyNomasTimelineOps::PrefixPlan?,
                                 latent_prefix_plan : ML::Video::HyNomasTimelineOps::PrefixPlan?) : Nil
    @@unreleased << RetainedResources.new(inputs, metadata, decoder, chain, shared,
      timeline_inputs, noise_prefix_plan, latent_prefix_plan)
  end

  def load_latent!(path : String, trusted_sha256 : String, bank_frames : Int32) : String
    unless Path[path].absolute? && File.file?(path) && !File.symlink?(path)
      raise ArgumentError.new("latent must be an absolute regular file without a symlink leaf")
    end
    unless PixelUMMBlockProbe.valid_digest?(trusted_sha256)
      raise ArgumentError.new("--latent-sha must be a lowercase SHA-256 digest")
    end
    expected_bytes = latent_bytes(bank_frames)
    expected_shape = latent_shape(bank_frames)
    unless File.size(path) == expected_bytes
      raise ArgumentError.new("latent bank extent must be #{expected_bytes} bytes for F32 NCTHW #{expected_shape}")
    end

    raw = File.read(path)
    digest = Digest::SHA256.hexdigest(raw)
    raise ArgumentError.new("latent SHA-256 mismatch") unless digest == trusted_sha256

    bytes = raw.to_slice
    latent_elements(bank_frames).times do |index|
      value = IO::ByteFormat::LittleEndian.decode(Float32, bytes[index * 4, 4])
      raise ArgumentError.new("latent bank contains non-finite F32 at element #{index}") unless value.finite?
    end
    raw
  end

  def validate_output_dir!(path : String) : String
    unless Path[path].absolute? && File.directory?(path) && !File.symlink?(path) && Dir.empty?(path)
      raise ArgumentError.new("output directory must be an existing empty absolute directory")
    end
    canonical = File.realpath(path)
    unless canonical.starts_with?(File.realpath("/opt/mldata_ssd") + "/")
      raise ArgumentError.new("diagnostic outputs must be on /opt/mldata_ssd")
    end
    canonical
  end
end

input_path = ""
input_sha = ""
continuation_path_option = ""
continuation_sha_option = ""
latent_path = ""
latent_sha = ""
output_dir = ""
latent_frames = HyNomasLatentDecode::DEFAULT_LATENT_FRAMES
latent_bank_frames_option = nil.as(Int32?)
budget_mib = 22_528_i64
plan_only = false
OptionParser.parse do |p|
  p.banner = "Diagnostic PixelUMM decode of an authenticated terminal F32 latent bank"
  p.on("--inputs PATH", "Authenticated conditioned v2 input bundle (Echo default identity retained)") { |v| input_path = v }
  p.on("--inputs-sha SHA", "Trusted input metadata identity") { |v| input_sha = v }
  p.on("--continuation PATH", "Authenticated long decoder noise bundle; required for non-Echo inputs") { |v| continuation_path_option = v }
  p.on("--continuation-sha SHA", "Trusted continuation metadata identity") { |v| continuation_sha_option = v }
  p.on("--latent PATH", "Imported terminal F32 NCTHW latent or latent bank") { |v| latent_path = v }
  p.on("--latent-sha SHA", "Trusted latent artifact SHA-256") { |v| latent_sha = v }
  p.on("--latent-frames N", "Latent frames: 30 by default, or authenticated 60/90-frame opt-in") { |v| latent_frames = v.to_i32 }
  p.on("--latent-bank-frames N", "Authenticated input bank frames; defaults to --latent-frames, or 180 for a 60/90-frame GPU prefix") { |v| latent_bank_frames_option = v.to_i32 }
  p.on("--output-dir PATH", "Existing empty result directory on external scratch storage") { |v| output_dir = v }
  p.on("--budget-mib N", "Aggregate named buffers, 16..24 GiB") { |v| budget_mib = v.to_i64 }
  p.on("--plan-only", "CPU-only authentication and memory admission") { plan_only = true }
end

raise ArgumentError.new("--inputs, --inputs-sha, --latent and --latent-sha are required") if {input_path, input_sha, latent_path, latent_sha}.any?(&.empty?)
HyNomasLatentDecode.validate_latent_frames!(latent_frames)
latent_bank_frames = latent_frames
if bank_frames = latent_bank_frames_option
  latent_bank_frames = bank_frames
end
HyNomasLatentDecode.validate_latent_bank_frames!(latent_frames, latent_bank_frames)
unless (16_384..24_576).includes?(budget_mib)
  raise ArgumentError.new("named budget must be within 16..24 GiB")
end

unless Path[input_path].absolute? && File.file?(input_path) && !File.symlink?(input_path)
  raise ArgumentError.new("inputs must be an absolute regular conditioned input manifest without a symlink leaf")
end
input_path = File.realpath(input_path)
default_inputs_path = if File.file?(HyNomasLatentDecode::BASE_INPUTS_PATH)
                        File.realpath(HyNomasLatentDecode::BASE_INPUTS_PATH)
                      else
                        HyNomasLatentDecode::BASE_INPUTS_PATH
                      end
default_base = input_path == default_inputs_path && input_sha == HyNomasLatentDecode::BASE_INPUTS_SHA
continuation_identity = HyNomasLatentDecodeAdmission.resolve_continuation!(latent_frames, default_base,
  continuation_path_option, continuation_sha_option,
  HyNomasLatentDecode::CONTINUATION_INPUTS_PATH, HyNomasLatentDecode::CONTINUATION_INPUTS_SHA)

request_start = Time.instant
inputs = HyNomasGeneration::Inputs.new(input_path, input_sha, conditioned: true)
HyNomasLatentDecodeAdmission.validate_default_geometry!(inputs.file.document["geometry"],
  inputs.file.document["qwen_source"], inputs.valid_text_tokens)
executable_path = Process.executable_path || raise ArgumentError.new("decoder receipt requires an identifiable executable")
executable_sha256 = PixelUMMBlockProbe.sha_file(executable_path)
latent_raw = HyNomasLatentDecode.load_latent!(latent_path, latent_sha, latent_bank_frames)
latent_path = File.realpath(latent_path)
source_latent_bank_bytes = if latent_bank_frames > latent_frames
                             HyNomasLatentDecode.latent_bytes(latent_bank_frames)
                           else
                             0_i64
                           end
timeline_inputs = nil.as(HyNomasGeneration::TimelineInputs?)
source_noise_bytes = 0_i64
if identity = continuation_identity
  continuation_path, continuation_sha = identity
  unless Path[continuation_path].absolute? && File.file?(continuation_path) && !File.symlink?(continuation_path)
    raise ArgumentError.new("continuation must be an absolute regular manifest without a symlink leaf")
  end
  continuation = HyNomasGeneration::TimelineInputs.new(File.realpath(continuation_path), continuation_sha)
  HyNomasLatentDecodeAdmission.validate_base_binding!(continuation.base.file.path,
    continuation.base.file.sha256, inputs.file.path, inputs.file.sha256)
  %w(decoder.rope_cos decoder.rope_sin).each do |name|
    base_record = inputs.records[name]
    continuation_record = continuation.records[name]
    HyNomasLatentDecodeAdmission.validate_matching_rope_records!(name,
      {base_record.dtype, base_record.shape, base_record.nbytes, base_record.sha256},
      {continuation_record.dtype, continuation_record.shape, continuation_record.nbytes, continuation_record.sha256})
  end
  continuation.verify_payload!
  HyNomasLatentDecode.validate_pixel_noise_prefix_finite!(continuation, latent_frames)
  timeline_inputs = continuation
  source_noise_bytes = continuation.records["decoder.noise"].nbytes
end
metadata = HyNomasGeneration::DecoderMetadata.new(inputs, budget_mib * 1024 * 1024,
  total_latent_frames: latent_frames)
metadata.cache.verify_payload!
decoder_noise_prefix_bytes = metadata.config.output_elements * sizeof(Float32)
unless decoder_noise_prefix_bytes == HyNomasLatentDecode.pixel_noise_elements(latent_frames) * sizeof(Float32)
  raise ArgumentError.new("admitted decoder noise and 4x latent pixel-noise prefix geometry differ")
end
decoder_peak_named_bytes = metadata.named_bytes + source_noise_bytes + source_latent_bank_bytes
decoder_external_named_bytes = source_noise_bytes + source_latent_bank_bytes
unless decoder_peak_named_bytes <= budget_mib * 1024 * 1024
  raise ArgumentError.new("decoder source banks and admitted timeline exceed the named buffer budget")
end
preflight_seconds = (Time.instant - request_start).total_seconds
HyNomasGeneration.emit("decoder_memory_plan", decoder_named_bytes: decoder_peak_named_bytes,
  decoder_metadata_named_bytes: metadata.named_bytes, source_noise_bytes: source_noise_bytes,
  decoder_external_named_bytes: decoder_external_named_bytes,
  decoder_noise_prefix_bytes: decoder_noise_prefix_bytes,
  source_latent_bank_bytes: source_latent_bank_bytes,
  named_budget_bytes: budget_mib * 1024 * 1024,
  latent_shape: HyNomasLatentDecode.latent_shape(latent_frames),
  latent_bank_shape: HyNomasLatentDecode.latent_shape(latent_bank_frames),
  latent_bank_frames: latent_bank_frames,
  decoder_noise_shape: HyNomasLatentDecode.pixel_noise_shape(latent_frames),
  latent_sha256: latent_sha, preflight_seconds: preflight_seconds,
  pixel_noise_source_sha256: timeline_inputs.try(&.records["decoder.noise"].sha256),
  scope: "terminal imported latent; no DiT or solver execution")

if plan_only
  metadata.close
  exit
end

output_dir = HyNomasLatentDecode.validate_output_dir!(output_dir)
shared = nil.as(ML::Video::Arena?)
decoder = nil.as(HyNomasGeneration::Decoder?)
chain = nil.as(ML::Metal::MPSCommandChain?)
noise_prefix_plan = nil.as(ML::Video::HyNomasTimelineOps::PrefixPlan?)
latent_prefix_plan = nil.as(ML::Video::HyNomasTimelineOps::PrefixPlan?)
timings = {"preflight_seconds" => preflight_seconds} of String => Float64
metrics = {} of String => Int64
request_error : Exception? = nil

begin
  shared = ML::Video::Arena.new
  latent_bank = shared.alloc("diagnostic_latent_bank", HyNomasLatentDecode.latent_elements(latent_bank_frames).to_i32)
  latent_bank.write_bytes(latent_raw.to_slice.to_unsafe, latent_raw.bytesize)
  latent = if latent_bank_frames == latent_frames
             latent_bank
           else
             latent_geometry = ML::Video::HyNomasTimelineOps::PrefixGeometry.new(
               channels: HyNomasLatentDecode::LATENT_CHANNELS,
               total_frames: latent_bank_frames,
               prefix_frames: latent_frames,
               height: HyNomasLatentDecode::LATENT_HEIGHT,
               width: HyNomasLatentDecode::LATENT_WIDTH)
             unless latent_geometry.source_bytes == source_latent_bank_bytes &&
                    latent_geometry.prefix_elements.to_i64 == HyNomasLatentDecode.latent_elements(latent_frames)
               raise ArgumentError.new("latent-bank prefix geometry differs from authenticated bank admission")
             end
             prefix = shared.alloc("diagnostic_latent_prefix", latent_geometry.prefix_elements)
             latent_prefix_plan = ML::Video::HyNomasTimelineOps::PrefixPlan.new(latent_geometry,
               source: latent_bank, destination: prefix)
             prefix
           end
  timeline_noise = nil.as(ML::MetalBuffer?)
  if continuation = timeline_inputs
    source_noise = continuation.upload!("decoder.noise", shared)
    noise_geometry = ML::Video::HyNomasTimelineOps::PrefixGeometry.new(
      channels: HyNomasLatentDecode::NOISE_CHANNELS,
      total_frames: 720,
      prefix_frames: HyNomasLatentDecode.pixel_noise_frames(latent_frames),
      height: HyNomasLatentDecode::NOISE_HEIGHT,
      width: HyNomasLatentDecode::NOISE_WIDTH)
    unless noise_geometry.prefix_elements.to_i64 == HyNomasLatentDecode.pixel_noise_elements(latent_frames)
      raise ArgumentError.new("pixel-noise prefix element count differs from the admitted latent timeline")
    end
    timeline_noise = shared.alloc("diagnostic_noise_prefix", noise_geometry.prefix_elements)
    noise_prefix_plan = ML::Video::HyNomasTimelineOps::PrefixPlan.new(noise_geometry,
      source: source_noise, destination: timeline_noise.not_nil!)
  end
  setup = Time.instant
  decoder = if supplied_noise = timeline_noise
              HyNomasGeneration::Decoder.new(inputs, latent, metadata, timeline_noise: supplied_noise)
            else
              HyNomasGeneration::Decoder.new(inputs, latent, metadata)
            end
  timings["decoder_setup_seconds"] = (Time.instant - setup).total_seconds

  chain = ML::Metal::MPSCommandChain.new
  allocations = ML::MetalBuffer.hy_text_probe_allocations
  readbacks = ML::MetalBuffer.hy_text_probe_readbacks
  run = Time.instant
  latent_prefix_plan.try(&.encode!(chain))
  noise_prefix_plan.try(&.encode!(chain))
  decoder.encode(chain)
  metrics["decoder_encode_allocations"] = ML::MetalBuffer.hy_text_probe_allocations - allocations
  metrics["decoder_encode_readbacks"] = ML::MetalBuffer.hy_text_probe_readbacks - readbacks
  metrics["decoder_preterminal_waits"] = chain.wait_count.to_i64
  unless metrics["decoder_encode_allocations"] == 0 && metrics["decoder_encode_readbacks"] == 0 && chain.wait_count == 0
    raise "decoder encode introduced allocation/readback/wait"
  end

  chain.commit_and_wait
  timings["decoder_wall_seconds"] = (Time.instant - run).total_seconds
  timings["decoder_gpu_seconds"] = chain.gpu_elapsed_seconds
  metrics["decoder_segments"] = chain.segment_count.to_i64
  metrics["decoder_waits"] = chain.wait_count.to_i64
  metrics["latent_prefix_gpu_gathered"] = latent_prefix_plan.nil? ? 0_i64 : 1_i64
  raise "decoder did not use exactly one terminal wait" unless chain.wait_count == 1

  pixels = decoder.output
  pixels_path = File.join(output_dir, "pixels.f32le")
  HyNomasGeneration.write_fresh_output(pixels_path) do |f|
    f.write(Bytes.new(pixels.contents.as(UInt8*), pixels.size.to_i32))
  end
  timings["request_seconds"] = (Time.instant - request_start).total_seconds
  noise_source_path = timeline_inputs.try(&.file.path) || inputs.file.path
  noise_manifest_sha = timeline_inputs.try(&.file.sha256) || inputs.file.sha256
  noise_record_sha = timeline_inputs.try(&.records["decoder.noise"].sha256) || inputs.records["decoder.noise"].sha256
  noise_selection = if timeline_inputs
                      "global pixel-noise frames [0, #{HyNomasLatentDecode.pixel_noise_frames(latent_frames)}) from each NCTHW channel using HyNomasTimelineOps::PrefixPlan"
                    else
                      "authenticated decoder.noise record consumed directly"
                    end
  latent_selection = if latent_prefix_plan
                       "global latent frames [0, #{latent_frames}) from each NCTHW channel using HyNomasTimelineOps::PrefixPlan"
                     else
                       "authenticated latent bank global frames [0, #{latent_frames}) consumed directly"
                     end
  receipt = {
    schema:                              "cogni_ml.hy_nomas_decode_latent/v1",
    upstream_revision:                   PixelUMMBlockProbe::UPSTREAM_REVISION,
    executable:                          executable_path,
    executable_sha256:                   executable_sha256,
    inputs:                              inputs.file.path,
    inputs_sha256:                       inputs.file.sha256,
    latent:                              latent_path,
    latent_sha256:                       latent_sha,
    latent_bank_sha256:                  latent_sha,
    latent_bytes:                        latent_raw.bytesize,
    latent_bank_bytes:                   latent_raw.bytesize,
    latent_dtype:                        "F32 little-endian",
    latent_shape:                        HyNomasLatentDecode.latent_shape(latent_frames),
    latent_frames:                       latent_frames,
    latent_selected_frames:              latent_frames,
    latent_prefix_start_frame:           0,
    latent_bank_shape:                   HyNomasLatentDecode.latent_shape(latent_bank_frames),
    latent_bank_frames:                  latent_bank_frames,
    latent_selection:                    latent_selection,
    latent_boundary:                     "caller-supplied terminal diagnostic latent bank; any shorter decode prefix is gathered on GPU; not produced by this request's DiT or solver",
    decoder_weight_cache_sha256:         HyNomasGeneration::PIXEL_CACHE_SHA,
    stem_weights_sha256:                 HyNomasGeneration::STEM_SHA,
    head_weights_sha256:                 HyNomasGeneration::HEAD_SHA,
    continuation_source:                 timeline_inputs.try(&.file.path),
    continuation_manifest_sha256:        timeline_inputs.try(&.file.sha256),
    decoder_rope_source:                 inputs.file.path,
    decoder_rope_manifest_sha256:        inputs.file.sha256,
    decoder_rope_cos_record_sha256:      inputs.records["decoder.rope_cos"].sha256,
    decoder_rope_sin_record_sha256:      inputs.records["decoder.rope_sin"].sha256,
    continuation_rope_cos_record_sha256: timeline_inputs.try(&.records["decoder.rope_cos"].sha256),
    continuation_rope_sin_record_sha256: timeline_inputs.try(&.records["decoder.rope_sin"].sha256),
    pixel_noise_source:                  noise_source_path,
    pixel_noise_manifest_sha256:         noise_manifest_sha,
    pixel_noise_record_sha256:           noise_record_sha,
    pixel_noise_shape:                   HyNomasLatentDecode.pixel_noise_shape(latent_frames),
    pixel_noise_frames:                  HyNomasLatentDecode.pixel_noise_frames(latent_frames),
    pixel_noise_prefix_start_frame:      0,
    pixel_noise_selection:               noise_selection,
    pixelumm:                            {steps: 1, window_latents: 10, overlap_latents: 5,
               starts: metadata.config.window_starts},
    precision:                        "BF16 decoder block/head weights; F32 stem weights and native activation/MPS arithmetic; original fixed noise and PixelUMM path",
    frames:                           metadata.config.pixel_frames,
    width:                            metadata.config.pixel_width,
    height:                           metadata.config.pixel_height,
    fps:                              24,
    output_dtype:                     "F32 little-endian planar RGB",
    output:                           pixels_path,
    output_bytes:                     pixels.size,
    output_sha256:                    PixelUMMBlockProbe.sha_file(pixels_path),
    output_shape:                     [1, 3, metadata.config.pixel_frames, metadata.config.pixel_height, metadata.config.pixel_width],
    decoder_named_bytes:              decoder_peak_named_bytes,
    decoder_metadata_named_bytes:     metadata.named_bytes,
    decoder_noise_prefix_bytes:       decoder_noise_prefix_bytes,
    decoder_source_noise_bytes:       source_noise_bytes,
    decoder_source_latent_bank_bytes: source_latent_bank_bytes,
    decoder_external_named_bytes:     decoder_external_named_bytes,
    metric_scope:                     "encode counters cover instrumented MetalBuffer methods; terminal pixel export is direct shared-buffer file output",
    timings:                          timings,
    metrics:                          metrics,
    full_upstream_video_parity:       false,
  }
  HyNomasGeneration.write_fresh_output(File.join(output_dir, "receipt.json")) { |f| f.puts(receipt.to_pretty_json) }
  HyNomasGeneration.emit("decoder_pixels_saved", output: pixels_path,
    total_seconds: timings["request_seconds"])
rescue ex
  request_error = ex
ensure
  drain_confirmed = chain.nil?
  begin
    chain.try do |value|
      value.commit_and_wait unless value.completed?
    end
  rescue cleanup_error
    request_error ||= cleanup_error
  end
  if value = chain
    drain_confirmed = value.drained?
  end
  if drain_confirmed
    begin
      noise_prefix_plan.try do |prefix|
        if prefix.encoded? && !prefix.released?
          value = chain || raise ArgumentError.new("encoded pixel-noise prefix has no command chain")
          prefix.release_after_drain!(value)
        end
      end
    rescue cleanup_error
      request_error ||= cleanup_error
    end
    begin
      latent_prefix_plan.try do |prefix|
        if prefix.encoded? && !prefix.released?
          value = chain || raise ArgumentError.new("encoded latent prefix has no command chain")
          prefix.release_after_drain!(value)
        end
      end
    rescue cleanup_error
      request_error ||= cleanup_error
    end
    begin
      decoder.try(&.release)
    rescue cleanup_error
      request_error ||= cleanup_error
    end
    begin
      chain.try(&.release)
    rescue cleanup_error
      request_error ||= cleanup_error
    end
    begin
      shared.try(&.release)
    rescue cleanup_error
      request_error ||= cleanup_error
    end
    begin
      metadata.close
    rescue cleanup_error
      request_error ||= cleanup_error
    end
  else
    request_error ||= ArgumentError.new("MPS drain was not confirmed; retaining Metal resources until process exit")
    HyNomasLatentDecode.retain_until_process_exit!(inputs, metadata, decoder, chain, shared,
      timeline_inputs, noise_prefix_plan, latent_prefix_plan)
  end
end

raise request_error.not_nil! if request_error
