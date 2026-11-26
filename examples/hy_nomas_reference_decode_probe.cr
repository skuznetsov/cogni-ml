require "digest/sha256"
require "json"
require "option_parser"
require "set"
require "./support/hy_nomas_generation"
require "./support/hy_nomas_reference_latent"
require "./support/pixelumm_block_probe"
require "./support/hy_nomas_probe_instrumentation"

module HyNomasReferenceDecodeProbe
  extend self

  RUN_ROOT              = "/opt/mldata_ssd/cogni-ml/hy-nomas-reference-handoff-20261101.pDnd7TV1"
  DEFAULT_INPUTS        = "/opt/mldata_ssd/cogni-ml/hy-nomas-echo-20261030/conditioning/inputs-echo-20s.json"
  DEFAULT_INPUTS_SHA    = "f5001fcbc786401488de1eb652061899ae4fdda88febb67ebd8bd6a22e569bcb"
  LATENT_SHA            = "115628ac1db4dca3533e410d4affbe9b9ce8828b0e93ff8d8be60645a3aa4a38"
  UPSTREAM_ARTIFACT_SHA = "8b7dcc20b55b27d0588fc8531f59c3f9157af8de57d0cf9bf6e5d226e2ab8ed7"
  UPSTREAM_REVISION     = "a31c0fe0bcc493be188fd37ddba4ce2f674becb4"
  CHECKPOINT_REVISION   = "bb6a8ca451bfe84947f0ea1c6d51957f983ffffe"
  CHECKPOINT_SHA        = "ebbee483c80890c7795a2babc06e1cb4d42564174cf3bae2b698c0589d95cc3b"
  DCAE_MODULE_SHA       = "f090a77806e250203618742a80263457cd2b44a5f9677bd35b40cfe9ce0fdc1d"
  NATIVE_SCHEMA         = "cogni_ml.hy_nomas_reference_decode_native/v1"
  OPERATION             = "saved_dcae_latent_pixelumm_decoder"
  PRECISION             = "BF16 PixelUMM block/head weights; F32 stem and native MPS activations; saved BF16 latent expanded to F32 on GPU"
  LATENT_F32_BYTES      =     552_960_i64
  PIXEL_BYTES           = 212_336_640_i64
  PIXEL_SHAPE           = [1_i32, 3_i32, 120_i32, 288_i32, 512_i32]

  record RetainedResources,
    inputs : HyNomasGeneration::Inputs,
    latent_input : HyNomasReferenceLatent::Input,
    latent_bf16 : ML::MetalBuffer,
    latent_f32 : ML::MetalBuffer,
    metadata : HyNomasGeneration::DecoderMetadata,
    decoder : HyNomasGeneration::Decoder?,
    chain : ML::Metal::MPSCommandChain?

  @@retained = [] of RetainedResources

  def retain(resources : RetainedResources) : Nil
    @@retained << resources
  end

  def valid_digest?(value : String) : Bool
    value.matches?(/\A[0-9a-f]{64}\z/)
  end

  def canonical_file!(path : String, label : String) : String
    unless Path[path].absolute? && File.file?(path) && !File.symlink?(path)
      raise ArgumentError.new("#{label} must be an absolute regular non-symlink file")
    end
    File.realpath(path)
  end

  def canonical_dir!(path : String) : String
    unless Path[path].absolute? && File.directory?(path) && !File.symlink?(path) && Dir.empty?(path)
      raise ArgumentError.new("output directory must be an existing empty absolute directory")
    end
    real = File.realpath(path)
    unless real.starts_with?(File.realpath(RUN_ROOT) + "/")
      raise ArgumentError.new("decoder outputs must remain beneath the authorized handoff run root")
    end
    real
  end

  def sha_json(path : String, label : String) : Tuple(JSON::Any, String)
    canonical = canonical_file!(path, label)
    bytes = File.read(canonical)
    raise ArgumentError.new("#{label} exceeds the metadata size limit") if bytes.bytesize > 2 * 1024 * 1024
    {JSON.parse(bytes), Digest::SHA256.hexdigest(bytes)}
  end

  def validate_dcae_lineage!(latent_path : String,
                             reference_path : String,
                             upstream_artifact_path : String,
                             native_receipt_path : String,
                             comparison_path : String,
                             bitwise_path : String) : Hash(String, JSON::Any)
    latent_path = canonical_file!(latent_path, "saved DCAE latent")
    unless PixelUMMBlockProbe.sha_file(latent_path) == LATENT_SHA
      raise ArgumentError.new("saved DCAE latent file SHA-256 mismatch")
    end
    reference_path = canonical_file!(reference_path, "DCAE upstream reference")
    upstream_artifact_path = canonical_file!(upstream_artifact_path, "DCAE upstream artifact")
    native_receipt_path = canonical_file!(native_receipt_path, "DCAE native receipt")
    comparison_path = canonical_file!(comparison_path, "DCAE comparison")
    bitwise_path = canonical_file!(bitwise_path, "DCAE bitwise comparison")
    reference, reference_sha = sha_json(reference_path, "DCAE upstream reference")
    native, native_sha = sha_json(native_receipt_path, "DCAE native receipt")
    comparison, comparison_sha = sha_json(comparison_path, "DCAE comparison")
    bitwise, bitwise_sha = sha_json(bitwise_path, "DCAE bitwise comparison")

    unless reference["schema"].as_s == "hy-nomas-dcae-v-upstream-reference-v1" &&
           reference["upstream"]["repository"].as_s == "Tencent-Hunyuan/Hy-Nomas" &&
           reference["upstream"]["reference_image_revision"].as_s == UPSTREAM_REVISION &&
           reference["upstream"]["vae_source_sha256"].as_s == DCAE_MODULE_SHA &&
           reference["checkpoint"]["repository_revision"].as_s == CHECKPOINT_REVISION &&
           reference["checkpoint"]["sha256"].as_s == CHECKPOINT_SHA &&
           reference["execution"]["device"].as_s == "mps" &&
           !reference["execution"]["cpu_fallback_enabled"].as_bool &&
           reference["execution"]["dtype"].as_s == "BF16" &&
           reference["execution"]["latent_postprocess"].as_s ==
             "identity: upstream _LatentDist.mode returns the encoder output"
      raise ArgumentError.new("pinned DCAE upstream reference provenance mismatch")
    end
    input = reference["input"]
    unless input["width"].as_i == 512 && input["height"].as_i == 288 &&
           input["frames"].as_i == 120 && input["fps"].as_i == 24 &&
           input["tile_frames"].as_i == 40 && input["temporal_feature_cache"].as_bool &&
           input["repeat"].as_s == "identical pixel frame repeated in pixel space before DCAE-V encode"
      raise ArgumentError.new("DCAE upstream reference is not the admitted 512x288, 120-frame repeated-image path")
    end
    latent_spec = reference["artifact"]["tensors"]["latent"]
    unless latent_spec["dtype"].as_s == "BF16" &&
           latent_spec["shape"].as_a.map(&.as_i) == [1, 32, 30, 9, 16]
      raise ArgumentError.new("DCAE upstream reference latent shape/dtype mismatch")
    end
    upstream_sha = reference["artifact"]["sha256"].as_s
    unless upstream_sha == UPSTREAM_ARTIFACT_SHA &&
           upstream_artifact_path.ends_with?(reference["artifact"]["filename"].as_s) &&
           File.size(upstream_artifact_path) == reference["artifact"]["bytes"].as_i64 &&
           PixelUMMBlockProbe.sha_file(upstream_artifact_path) == upstream_sha
      raise ArgumentError.new("upstream DCAE reference tensor artifact identity mismatch")
    end
    unless native["event"].as_s == "dcae_native_completed" &&
           native["output_sha256"].as_s == LATENT_SHA &&
           File.realpath(native["output"].as_s) == latent_path &&
           native["metrics"]["allocations"].as_i == 0 &&
           native["metrics"]["readbacks"].as_i == 0 &&
           native["metrics"]["preterminal_waits"].as_i == 0 &&
           native["waits"].as_i == 1
      raise ArgumentError.new("DCAE candidate receipt does not bind the pinned native latent and terminal metrics")
    end
    expected_names = {"input.clip", "trace.stages.5.tile0", "trace.stages.5.tile1",
                      "trace.stages.5.tile2", "latent"}.to_set
    comparisons = comparison["comparisons"].as_a
    names = comparisons.map { |entry| entry["name"].as_s }.to_set
    unless comparison["schema"].as_s == "hy-nomas-dcae-v-comparison-v1" &&
           comparison["all_passed"].as_bool &&
           comparison["candidate_sha256"].as_s == LATENT_SHA &&
           comparison["upstream_reference_sha256"].as_s == UPSTREAM_ARTIFACT_SHA &&
           comparisons.size == expected_names.size && names == expected_names &&
           comparisons.all? { |entry| entry["passed"].as_bool && entry["max_abs_error"].as_f == 0.0 }
      raise ArgumentError.new("DCAE upstream/candidate comparison does not establish exact saved-latent parity")
    end
    tensors = bitwise["tensors"].as_h
    unless bitwise["all_bitwise_equal"].as_bool && tensors.keys.to_set == expected_names &&
           tensors.values.all? { |entry| entry["bitwise_equal"].as_bool }
      raise ArgumentError.new("DCAE bitwise comparison evidence is incomplete or failed")
    end

    {
      "upstream_reference" => JSON::Any.new({"path" => JSON::Any.new(reference_path), "sha256" => JSON::Any.new(reference_sha)}),
      "upstream_artifact"  => JSON::Any.new({"path" => JSON::Any.new(upstream_artifact_path), "sha256" => JSON::Any.new(upstream_sha)}),
      "native_receipt"     => JSON::Any.new({"path" => JSON::Any.new(native_receipt_path), "sha256" => JSON::Any.new(native_sha)}),
      "comparison"         => JSON::Any.new({"path" => JSON::Any.new(comparison_path), "sha256" => JSON::Any.new(comparison_sha)}),
      "bitwise_comparison" => JSON::Any.new({"path" => JSON::Any.new(bitwise_path), "sha256" => JSON::Any.new(bitwise_sha)}),
    }
  end

  def write_buffer!(path : String, buffer : ML::MetalBuffer, expected_bytes : Int64) : Hash(String, JSON::Any)
    raise ArgumentError.new("decoder output buffer extent mismatch") unless buffer.valid? && buffer.size == expected_bytes
    HyNomasGeneration.write_fresh_output(path) do |f|
      f.write(Bytes.new(buffer.contents.as(UInt8*), buffer.size.to_i32))
    end
    {
      "path"  => JSON::Any.new(path),
      "dtype" => JSON::Any.new("F32LE"),
      "shape" => JSON::Any.new(if expected_bytes == LATENT_F32_BYTES
        [1, 32, 30, 9, 16].map { |n| JSON::Any.new(n) }
      else
        PIXEL_SHAPE.map { |n| JSON::Any.new(n) }
      end),
      "nbytes" => JSON::Any.new(expected_bytes),
      "sha256" => JSON::Any.new(PixelUMMBlockProbe.sha_file(path)),
    }
  end

  def json_ref(path : String, digest : String) : JSON::Any
    JSON::Any.new({"path" => JSON::Any.new(path), "sha256" => JSON::Any.new(digest)})
  end
end

latent_path = ""
latent_sha = ""
reference_path = ""
upstream_artifact_path = ""
dcae_receipt_path = ""
dcae_comparison_path = ""
dcae_bitwise_path = ""
inputs_path = HyNomasReferenceDecodeProbe::DEFAULT_INPUTS
inputs_sha = HyNomasReferenceDecodeProbe::DEFAULT_INPUTS_SHA
output_dir = ""
budget_mib = 22_528_i64
plan_only = false
OptionParser.parse do |p|
  p.banner = "Decode an authenticated saved DCAE-V latent through native PixelUMM only"
  p.on("--latent PATH", "Pinned saved DCAE-V native-full.safetensors") { |v| latent_path = v }
  p.on("--latent-sha256 SHA", "Expected full saved DCAE safetensors SHA-256") { |v| latent_sha = v }
  p.on("--dcae-reference PATH", "Pinned upstream DCAE reference manifest") { |v| reference_path = v }
  p.on("--dcae-upstream-artifact PATH", "Pinned upstream DCAE reference safetensors") { |v| upstream_artifact_path = v }
  p.on("--dcae-native-receipt PATH", "Receipt for the saved native DCAE latent") { |v| dcae_receipt_path = v }
  p.on("--dcae-comparison PATH", "Source-upstream versus native DCAE comparison") { |v| dcae_comparison_path = v }
  p.on("--dcae-bitwise PATH", "Saved BF16 bitwise payload comparison") { |v| dcae_bitwise_path = v }
  p.on("--inputs PATH", "Pinned conditioned PixelUMM noise and input bundle") { |v| inputs_path = v }
  p.on("--inputs-sha256 SHA", "Trusted input manifest SHA-256") { |v| inputs_sha = v }
  p.on("--output-dir PATH", "Existing empty result directory under the handoff root") { |v| output_dir = v }
  p.on("--budget-mib N", "Aggregate named buffers, 16..24 GiB") { |v| budget_mib = v.to_i64 }
  p.on("--plan-only", "CPU-only admission; do not create a Metal device or run PixelUMM") { plan_only = true }
end

required = {latent_path, latent_sha, reference_path, upstream_artifact_path, dcae_receipt_path,
            dcae_comparison_path, dcae_bitwise_path}
raise ArgumentError.new("all saved-latent and DCAE lineage arguments are required") if required.any?(&.empty?)
unless latent_sha == HyNomasReferenceDecodeProbe::LATENT_SHA && inputs_sha == HyNomasReferenceDecodeProbe::DEFAULT_INPUTS_SHA
  raise ArgumentError.new("saved DCAE latent and conditioned decoder-input identities are fixed for this qualification")
end
unless (16_384..24_576).includes?(budget_mib)
  raise ArgumentError.new("decoder budget must be between 16 and 24 GiB")
end

started = Time.instant
latent_path = HyNomasReferenceDecodeProbe.canonical_file!(latent_path, "saved DCAE latent")
lineage = HyNomasReferenceDecodeProbe.validate_dcae_lineage!(latent_path, reference_path,
  upstream_artifact_path, dcae_receipt_path, dcae_comparison_path, dcae_bitwise_path)
latent_input = HyNomasReferenceLatent::Input.new(latent_path, latent_sha)
inputs_path = File.realpath(inputs_path)
inputs = HyNomasGeneration::Inputs.new(inputs_path, inputs_sha, conditioned: true,
  geometry: HyNomasGeneration::SpatialGeometry::DEFAULT)
input_noise = inputs.records["decoder.noise"]
unless input_noise.dtype == "F32" && input_noise.shape == HyNomasReferenceDecodeProbe::PIXEL_SHAPE &&
       input_noise.nbytes == HyNomasReferenceDecodeProbe::PIXEL_BYTES
  raise ArgumentError.new("authenticated conditioned decoder noise geometry mismatch")
end
metadata = HyNomasGeneration::DecoderMetadata.new(inputs, budget_mib * 1024 * 1024,
  capture_raw: true, total_latent_frames: 30)
metadata.cache.verify_payload!
external_source_bytes = HyNomasReferenceLatent::BF16_BYTES
unless metadata.named_bytes + external_source_bytes <= budget_mib * 1024 * 1024
  raise ArgumentError.new("saved BF16 latent source exceeds decoder aggregate memory budget")
end

puts({event: "reference_decoder_memory_plan", decoder_named_bytes: metadata.named_bytes,
      source_bf16_latent_bytes: external_source_bytes,
      aggregate_named_bytes: metadata.named_bytes + external_source_bytes,
      named_budget_bytes: budget_mib * 1024 * 1024,
      latent_source_sha256: latent_sha,
      input_manifest_sha256: inputs.file.sha256,
      decoder_noise_sha256: input_noise.sha256,
      dcae_comparison: "source-upstream and saved native artifacts reauthenticated; no DCAE execution",
      scope: "saved DCAE-V latent and authenticated PixelUMM decoder only"}.to_json)
if plan_only
  metadata.close
  latent_input.close
  exit
end

output_dir = HyNomasReferenceDecodeProbe.canonical_dir!(output_dir)
latent_bf16 = latent_input.upload_bf16!
latent_f32 = ML::MetalBuffer.new(HyNomasReferenceLatent::F32_BYTES)
expansion = HyNomasReferenceLatent::ExpansionPlan.new(latent_bf16, latent_f32)
decoder = nil.as(HyNomasGeneration::Decoder?)
chain = nil.as(ML::Metal::MPSCommandChain?)
request_error : Exception? = nil
timings = {"preflight_seconds" => (Time.instant - started).total_seconds} of String => Float64
metrics = {} of String => Int64

begin
  setup = Time.instant
  decoder = HyNomasGeneration::Decoder.new(inputs, latent_f32, metadata)
  timings["decoder_setup_seconds"] = (Time.instant - setup).total_seconds
  chain = ML::Metal::MPSCommandChain.new
  allocation_baseline = ML::MetalBuffer.hy_text_probe_allocations
  readback_baseline = ML::MetalBuffer.hy_text_probe_readbacks
  encode_start = Time.instant
  expansion.encode(chain)
  decoder.encode(chain)
  metrics["decoder_encode_allocations"] = ML::MetalBuffer.hy_text_probe_allocations - allocation_baseline
  metrics["decoder_encode_readbacks"] = ML::MetalBuffer.hy_text_probe_readbacks - readback_baseline
  metrics["decoder_preterminal_waits"] = chain.wait_count.to_i64
  unless metrics["decoder_encode_allocations"] == 0 &&
         metrics["decoder_encode_readbacks"] == 0 && chain.wait_count == 0
    raise "saved-latent decode encode introduced allocation/readback/wait"
  end
  chain.commit_and_wait
  timings["decoder_wall_seconds"] = (Time.instant - encode_start).total_seconds
  timings["decoder_gpu_seconds"] = chain.gpu_elapsed_seconds
  metrics["decoder_waits"] = chain.wait_count.to_i64
  raise "saved-latent decoder did not use one terminal wait" unless chain.wait_count == 1 && chain.completed?

  source_f32 = HyNomasReferenceDecodeProbe.write_buffer!(File.join(output_dir, "latent.f32le"),
    latent_f32, HyNomasReferenceDecodeProbe::LATENT_F32_BYTES)
  raw = HyNomasReferenceDecodeProbe.write_buffer!(File.join(output_dir, "raw-pixels.f32le"),
    decoder.raw_output, HyNomasReferenceDecodeProbe::PIXEL_BYTES)
  display = HyNomasReferenceDecodeProbe.write_buffer!(File.join(output_dir, "display-pixels.f32le"),
    decoder.output, HyNomasReferenceDecodeProbe::PIXEL_BYTES)
  receipt = {
    "schema" => JSON::Any.new(HyNomasReferenceDecodeProbe::NATIVE_SCHEMA),
    "operation" => JSON::Any.new(HyNomasReferenceDecodeProbe::OPERATION),
    "upstream_revision" => JSON::Any.new(PixelUMMBlockProbe::UPSTREAM_REVISION),
    "precision" => JSON::Any.new(HyNomasReferenceDecodeProbe::PRECISION),
    "resolution" => JSON::Any.new("512x288"),
    "width" => JSON::Any.new(512), "height" => JSON::Any.new(288),
    "frames" => JSON::Any.new(120), "fps" => JSON::Any.new(24),
    "source_latent" => JSON::Any.new({
      "path" => JSON::Any.new(latent_path), "sha256" => JSON::Any.new(latent_sha),
      "tensor_key" => JSON::Any.new("latent"), "dtype" => JSON::Any.new("BF16"),
      "shape" => JSON::Any.new([1, 32, 30, 9, 16].map { |n| JSON::Any.new(n) }),
      "nbytes" => JSON::Any.new(HyNomasReferenceLatent::BF16_BYTES),
    }),
    "dcae_lineage" => JSON::Any.new(lineage),
    "decoder_inputs" => JSON::Any.new({
      "manifest_path"   => JSON::Any.new(inputs.file.path),
      "manifest_sha256" => JSON::Any.new(inputs.file.sha256),
      "payload_sha256"  => inputs.file.document["payload_sha256"],
      "prompt"          => JSON::Any.new(inputs.prompt),
      "noise"           => JSON::Any.new({
        "key"    => JSON::Any.new("decoder.noise"),
        "dtype"  => JSON::Any.new(input_noise.dtype),
        "shape"  => JSON::Any.new(input_noise.shape.map { |n| JSON::Any.new(n) }),
        "nbytes" => JSON::Any.new(input_noise.nbytes),
        "sha256" => JSON::Any.new(input_noise.sha256),
      }),
    }),
    "weights" => JSON::Any.new({
      "decoder_cache_manifest_sha256" => JSON::Any.new(metadata.cache.sha256),
      "decoder_cache_payload_sha256"  => metadata.cache.document["payload_sha256"],
      "stem_subset_sha256"            => JSON::Any.new(HyNomasGeneration::STEM_SHA),
      "head_subset_sha256"            => JSON::Any.new(HyNomasGeneration::HEAD_SHA),
    }),
    "outputs" => JSON::Any.new({
      "latent_f32"     => JSON::Any.new(source_f32),
      "raw_pixels"     => JSON::Any.new(raw),
      "display_pixels" => JSON::Any.new(display),
    }),
    "pixelumm" => JSON::Any.new({
      "window_latents" => JSON::Any.new(10), "overlap_latents" => JSON::Any.new(5),
      "starts" => JSON::Any.new(metadata.config.window_starts.map { |n| JSON::Any.new(n) }),
      "prediction_count_per_window" => JSON::Any.new(1),
    }),
    "execution" => JSON::Any.new({
      "device"                                   => JSON::Any.new("mps"),
      "terminal_drain_confirmed"                 => JSON::Any.new(true),
      "exports_after_terminal_drain"             => JSON::Any.new(true),
      "metrics"                                  => JSON::Any.new(metrics.transform_values { |n| JSON::Any.new(n) }),
      "timings"                                  => JSON::Any.new(timings.transform_values { |n| JSON::Any.new(n) }),
      "decoder_named_bytes"                      => JSON::Any.new(decoder.not_nil!.named_bytes),
      "saved_latent_source_bytes"                => JSON::Any.new(latent_bf16.size),
      "aggregate_named_bytes"                    => JSON::Any.new(decoder.not_nil!.named_bytes + latent_bf16.size),
      "opaque_mps_private_allocation_visibility" => JSON::Any.new("not observable through MetalBuffer instrumentation"),
    }),
  }
  receipt_path = File.join(output_dir, "receipt.json")
  HyNomasGeneration.write_fresh_output(receipt_path) { |f| f.puts(JSON::Any.new(receipt).to_pretty_json) }
  puts({event: "reference_decoder_completed", receipt: receipt_path,
        raw_pixels_sha256: raw["sha256"].as_s,
        display_pixels_sha256: display["sha256"].as_s,
        latent_f32_sha256: source_f32["sha256"].as_s,
        timings: timings, metrics: metrics}.to_json)
rescue ex
  request_error = ex
ensure
  drain_confirmed = chain.nil?
  begin
    chain.try { |value| value.commit_and_wait unless value.completed? }
  rescue cleanup_error
    request_error ||= cleanup_error
  end
  drain_confirmed = chain.try(&.drained?) || drain_confirmed
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
      latent_f32.release
      latent_bf16.release
      latent_input.close
      metadata.close
    rescue cleanup_error
      request_error ||= cleanup_error
    end
  else
    request_error ||= ArgumentError.new("MPS drain was not confirmed; retaining decoder Metal resources until process exit")
    HyNomasReferenceDecodeProbe.retain(HyNomasReferenceDecodeProbe::RetainedResources.new(
      inputs, latent_input, latent_bf16, latent_f32, metadata, decoder, chain))
  end
end

raise request_error.not_nil! if request_error
