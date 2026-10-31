require "option_parser"
require "digest/sha256"
require "json"
require "./support/hy_nomas_generation"

module HyNomasLatentDecode
  extend self

  BASE_INPUTS_PATH = "/opt/mldata_ssd/cogni-ml/hy-nomas-echo-20261030/conditioning/inputs-echo-20s.json"
  BASE_INPUTS_SHA  = "f5001fcbc786401488de1eb652061899ae4fdda88febb67ebd8bd6a22e569bcb"
  LATENT_SHAPE     = [1, 32, 30, 9, 16]
  LATENT_ELEMENTS  = 1 * 32 * 30 * 9 * 16
  LATENT_BYTES     = LATENT_ELEMENTS * sizeof(Float32)

  class RetainedResources
    def initialize(@inputs : HyNomasGeneration::Inputs,
                   @metadata : HyNomasGeneration::DecoderMetadata,
                   @decoder : HyNomasGeneration::Decoder?,
                   @chain : ML::Metal::MPSCommandChain?,
                   @shared : ML::Video::Arena?)
    end
  end

  @@unreleased = [] of RetainedResources

  def retain_until_process_exit!(inputs : HyNomasGeneration::Inputs,
                                 metadata : HyNomasGeneration::DecoderMetadata,
                                 decoder : HyNomasGeneration::Decoder?,
                                 chain : ML::Metal::MPSCommandChain?,
                                 shared : ML::Video::Arena?) : Nil
    @@unreleased << RetainedResources.new(inputs, metadata, decoder, chain, shared)
  end

  def load_latent!(path : String, trusted_sha256 : String) : String
    unless Path[path].absolute? && File.file?(path) && !File.symlink?(path)
      raise ArgumentError.new("latent must be an absolute regular file without a symlink leaf")
    end
    unless PixelUMMBlockProbe.valid_digest?(trusted_sha256)
      raise ArgumentError.new("--latent-sha must be a lowercase SHA-256 digest")
    end
    unless File.size(path) == LATENT_BYTES
      raise ArgumentError.new("latent extent must be #{LATENT_BYTES} bytes for F32 NCTHW #{LATENT_SHAPE}")
    end

    raw = File.read(path)
    digest = Digest::SHA256.hexdigest(raw)
    raise ArgumentError.new("latent SHA-256 mismatch") unless digest == trusted_sha256

    bytes = raw.to_slice
    LATENT_ELEMENTS.times do |index|
      value = IO::ByteFormat::LittleEndian.decode(Float32, bytes[index * 4, 4])
      raise ArgumentError.new("latent contains non-finite F32 at element #{index}") unless value.finite?
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
latent_path = ""
latent_sha = ""
output_dir = ""
budget_mib = 22_528_i64
plan_only = false
OptionParser.parse do |p|
  p.banner = "Diagnostic PixelUMM decode of one authenticated terminal F32 latent"
  p.on("--inputs PATH", "Pinned Echo conditioned input bundle") { |v| input_path = v }
  p.on("--inputs-sha SHA", "Trusted input metadata identity") { |v| input_sha = v }
  p.on("--latent PATH", "Imported terminal F32 NCTHW latent") { |v| latent_path = v }
  p.on("--latent-sha SHA", "Trusted latent artifact SHA-256") { |v| latent_sha = v }
  p.on("--output-dir PATH", "Existing empty result directory on external scratch storage") { |v| output_dir = v }
  p.on("--budget-mib N", "Aggregate named buffers, 16..24 GiB") { |v| budget_mib = v.to_i64 }
  p.on("--plan-only", "CPU-only authentication and memory admission") { plan_only = true }
end

raise ArgumentError.new("--inputs, --inputs-sha, --latent and --latent-sha are required") if {input_path, input_sha, latent_path, latent_sha}.any?(&.empty?)
unless (16_384..24_576).includes?(budget_mib)
  raise ArgumentError.new("named budget must be within 16..24 GiB")
end

unless Path[input_path].absolute? && File.file?(input_path) && !File.symlink?(input_path) &&
       File.realpath(input_path) == File.realpath(HyNomasLatentDecode::BASE_INPUTS_PATH) &&
       input_sha == HyNomasLatentDecode::BASE_INPUTS_SHA
  raise ArgumentError.new("inputs must be the pinned Echo conditioned bundle and SHA-256")
end
input_path = File.realpath(input_path)

request_start = Time.instant
inputs = HyNomasGeneration::Inputs.new(input_path, input_sha, conditioned: true)
latent_raw = HyNomasLatentDecode.load_latent!(latent_path, latent_sha)
latent_path = File.realpath(latent_path)
metadata = HyNomasGeneration::DecoderMetadata.new(inputs, budget_mib * 1024 * 1024,
  total_latent_frames: 30)
metadata.cache.verify_payload!
preflight_seconds = (Time.instant - request_start).total_seconds
HyNomasGeneration.emit("decoder_memory_plan", decoder_named_bytes: metadata.named_bytes,
  named_budget_bytes: budget_mib * 1024 * 1024, latent_shape: HyNomasLatentDecode::LATENT_SHAPE,
  latent_sha256: latent_sha, preflight_seconds: preflight_seconds,
  scope: "terminal imported latent; no DiT or solver execution")

if plan_only
  metadata.close
  exit
end

output_dir = HyNomasLatentDecode.validate_output_dir!(output_dir)
shared = nil.as(ML::Video::Arena?)
decoder = nil.as(HyNomasGeneration::Decoder?)
chain = nil.as(ML::Metal::MPSCommandChain?)
timings = {"preflight_seconds" => preflight_seconds} of String => Float64
metrics = {} of String => Int64
request_error : Exception? = nil

begin
  shared = ML::Video::Arena.new
  latent = shared.alloc("diagnostic_latent", HyNomasLatentDecode::LATENT_ELEMENTS)
  latent.write_bytes(latent_raw.to_slice.to_unsafe, latent_raw.bytesize)
  setup = Time.instant
  decoder = HyNomasGeneration::Decoder.new(inputs, latent, metadata)
  timings["decoder_setup_seconds"] = (Time.instant - setup).total_seconds

  chain = ML::Metal::MPSCommandChain.new
  allocations = ML::MetalBuffer.hy_text_probe_allocations
  readbacks = ML::MetalBuffer.hy_text_probe_readbacks
  run = Time.instant
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
  raise "decoder did not use exactly one terminal wait" unless chain.wait_count == 1

  pixels = decoder.output
  pixels_path = File.join(output_dir, "pixels.f32le")
  HyNomasGeneration.write_fresh_output(pixels_path) do |f|
    f.write(Bytes.new(pixels.contents.as(UInt8*), pixels.size.to_i32))
  end
  timings["request_seconds"] = (Time.instant - request_start).total_seconds
  receipt = {
    schema:                      "cogni_ml.hy_nomas_decode_latent/v1",
    upstream_revision:           PixelUMMBlockProbe::UPSTREAM_REVISION,
    inputs:                      inputs.file.path,
    inputs_sha256:               inputs.file.sha256,
    latent:                      latent_path,
    latent_sha256:               latent_sha,
    latent_bytes:                latent_raw.bytesize,
    latent_dtype:                "F32 little-endian",
    latent_shape:                HyNomasLatentDecode::LATENT_SHAPE,
    latent_boundary:             "caller-supplied terminal diagnostic latent; not produced by this request's DiT or solver",
    decoder_weight_cache_sha256: HyNomasGeneration::PIXEL_CACHE_SHA,
    stem_weights_sha256:         HyNomasGeneration::STEM_SHA,
    head_weights_sha256:         HyNomasGeneration::HEAD_SHA,
    pixelumm:                    {steps: 1, window_latents: 10, overlap_latents: 5, starts: [0, 5, 10, 15, 20]},
    precision:                   "BF16 decoder block/head weights; F32 stem weights and native activation/MPS arithmetic; original fixed noise and PixelUMM path",
    frames:                      120,
    width:                       512,
    height:                      288,
    fps:                         24,
    output_dtype:                "F32 little-endian planar RGB",
    output:                      pixels_path,
    output_bytes:                pixels.size,
    output_sha256:               PixelUMMBlockProbe.sha_file(pixels_path),
    decoder_named_bytes:         metadata.named_bytes,
    metric_scope:                "encode counters cover instrumented MetalBuffer methods; terminal pixel export is direct shared-buffer file output",
    timings:                     timings,
    metrics:                     metrics,
    full_upstream_video_parity:  false,
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
    HyNomasLatentDecode.retain_until_process_exit!(inputs, metadata, decoder, chain, shared)
  end
end

raise request_error.not_nil! if request_error
