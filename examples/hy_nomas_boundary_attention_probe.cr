require "json"
require "option_parser"
require "digest/sha256"
require "./support/hy_nomas_timeline_inputs"
require "./support/qwen25_prefix_probe"
require "../src/ml/video/hy_nomas_timeline_ops"
require "../src/ml/video/hy_nomas_continuation_history"
require "../src/ml/video/hy_nomas_history_attention"

struct HyNomasBoundaryCapture
  getter name : String
  getter shape : Array(Int32)
  getter buffer : ML::MetalBuffer

  def initialize(@name : String, @shape : Array(Int32), @buffer : ML::MetalBuffer)
  end
end

module HyNomasBoundaryCaptureSpec
  extend self

  SCHEMA              = "cogni_ml.hy_nomas_boundary_attention_capture/v1"
  UPSTREAM_REVISION   = "911cea6458a89c2663513ff7099561ec190dd491"
  CONDITIONING_SHA    = "2807e11814f07975c807f367b0a4f15e3c6bdde1eb62ce9bda851814d09a6e47"
  PREVIEW_ROOT        = "/opt/mldata_ssd/cogni-ml/hy-nomas-cache-fix-20261030/preview-10s/native"
  PREVIEW_RECEIPT_SHA = "16cd52cf89e4a370f0b0112e14e8eefba08db76fcb2da6789e98f888ab6f24a4"
  PREVIEW_LATENT_SHA  = "003b116c48e55809c092ce67f64242f80e6f24a26c89bc2d2fe38352e4da365b"

  TENSORS = begin
    values = [] of Tuple(String, Array(Int32))
    latent = [1, 32, 30, 9, 16]
    rope = [4320, 128]
    txt = [1, 576, 2048]
    vec = [1, 2048]
    img = [1, 4320, 2048]
    qkv = [1, 4896, 16, 128]
    history = [1, 4877, 16, 128]
    {
      {"input.latent.writer_clean", latent},
      {"input.latent.writer_noisy", latent},
      {"input.latent.reader_initial", latent},
      {"input.latent.reader_model", latent},
      {"input.rope_cos.writer", rope},
      {"input.rope_sin.writer", rope},
      {"input.rope_cos.reader", rope},
      {"input.rope_sin.reader", rope},
      {"input.txt.writer", txt},
      {"input.txt.reader", txt},
      {"input.vec.writer", vec},
      {"input.vec.reader", vec},
      {"input.vec_txt.writer", vec},
      {"input.vec_txt.reader", vec},
      {"input.img.writer", img},
      {"input.img.reader", img},
    }.each { |name, shape| values << {name, shape} }
    {"writer", "reader"}.each do |phase|
      {"query", "key", "value"}.each do |axis|
        values << {"#{phase}.#{axis}", qkv}
      end
    end
    values << {"history.key", history}
    values << {"history.value", history}
    values << {"reader.history_attention", history}
    values
  end

  def elements(shape : Array(Int32)) : Int64
    shape.reduce(1_i64) { |count, extent| count * extent }
  end

  def bytes : Int64
    TENSORS.sum { |_, shape| elements(shape) * sizeof(Float32) }
  end

  def sha256(path : String) : String
    digest = Digest::SHA256.new
    fd = LibC.open(path, LibC::O_RDONLY)
    raise IO::Error.from_errno("open #{path}") if fd < 0
    begin
      scratch = Bytes.new(1024 * 1024)
      loop do
        count = LibC.read(fd, scratch.to_unsafe, scratch.size.to_u64)
        if count < 0
          next if Errno.value == Errno::EINTR
          raise IO::Error.from_errno("read #{path}")
        end
        break if count == 0
        digest.update(scratch[0, count.to_i])
      end
    ensure
      raise IO::Error.from_errno("close #{path}") if LibC.close(fd) < 0
    end
    digest.hexfinal
  end

  def copy_preview_prefix!(path : String, timeline : ML::MetalBuffer) : Nil
    fd = LibC.open(path, LibC::O_RDONLY)
    raise IO::Error.from_errno("open #{path}") if fd < 0
    preview_bytes = Bytes.new(829_440 * sizeof(Float32))
    begin
      offset = 0
      while offset < preview_bytes.size
        count = LibC.read(fd, preview_bytes.to_unsafe + offset, (preview_bytes.size - offset).to_u64)
        if count < 0
          next if Errno.value == Errno::EINTR
          raise IO::Error.from_errno("read #{path}")
        end
        raise IO::EOFError.new("preview latent bank ended early: #{path}") if count == 0
        offset += count.to_i
      end
    ensure
      raise IO::Error.from_errno("close #{path}") if LibC.close(fd) < 0
    end
    timeline_ptr = timeline.contents.as(UInt8*)
    preview_ptr = preview_bytes.to_unsafe
    channel_stride = 180 * 9 * 16 * sizeof(Float32)
    prefix_stride = 30 * 9 * 16 * sizeof(Float32)
    32.times do |channel|
      (timeline_ptr + channel * channel_stride).copy_from(preview_ptr + channel * channel_stride, prefix_stride)
    end
  end

  def open_fresh_output(path : String) : Int32
    fd = LibC.open(path, LibC::O_WRONLY | LibC::O_CREAT | LibC::O_EXCL, 0o600)
    raise IO::Error.from_errno("create fresh output #{path}") if fd < 0
    fd
  end

  def write_all!(fd : Int32, bytes : Bytes, path : String) : Nil
    offset = 0
    while offset < bytes.size
      count = LibC.write(fd, bytes.to_unsafe + offset, (bytes.size - offset).to_u64)
      if count < 0
        next if Errno.value == Errno::EINTR
        raise IO::Error.from_errno("write #{path}")
      end
      raise IO::Error.new("zero-length write to #{path}") if count == 0
      offset += count.to_i
    end
  end

  def close_output!(fd : Int32, path : String) : Nil
    raise IO::Error.from_errno("close #{path}") if LibC.close(fd) < 0
  end
end

manifest = ""
manifest_sha = ""
output = ""
budget = 0_i64
OptionParser.parse do |parser|
  parser.banner = "Capture the authentic first Hy-Nomas writer/read boundary at block 0."
  parser.on("--inputs PATH", "Authenticated continuation input manifest") { |value| manifest = value }
  parser.on("--inputs-sha256 SHA", "Trusted continuation manifest SHA-256") { |value| manifest_sha = value }
  parser.on("--output PATH", "Fresh capture JSON path with sibling tensor payload") { |value| output = value }
  parser.on("--max-named-bytes N", "Finite named GPU buffer budget, at most 8 GiB") { |value| budget = value.to_i64 }
end
unless ARGV.empty? && !manifest.empty? && manifest_sha == HyNomasBoundaryCaptureSpec::CONDITIONING_SHA &&
       !output.empty? && output.ends_with?(".json") && 0 < budget <= 8_i64 * 1024**3
  raise ArgumentError.new("an admitted inputs manifest/digest, fresh .json output, and budget up to 8 GiB are required")
end
unless Path[manifest].absolute? && File.file?(manifest) && !File.symlink?(manifest)
  raise ArgumentError.new("conditioning manifest must be an absolute regular file")
end
manifest = File.realpath(manifest)
output_dir = File.realpath(File.dirname(output))
external_root = File.realpath("/opt/mldata_ssd")
unless output_dir.starts_with?(external_root + "/")
  raise ArgumentError.new("boundary capture output must be under /opt/mldata_ssd")
end
output = File.join(output_dir, File.basename(output))
payload = output.sub(/\.json\z/, ".tensors.bin")
{output, payload}.each do |path|
  raise ArgumentError.new("refusing to replace capture #{path}") if File.exists?(path) || File.symlink?(path)
end

preview_root = File.realpath(HyNomasBoundaryCaptureSpec::PREVIEW_ROOT)
preview_receipt = File.join(preview_root, "receipt.json")
preview_latent = File.join(preview_root, "latent-bank.f32le")
unless File.file?(preview_receipt) && !File.symlink?(preview_receipt) &&
       File.realpath(preview_receipt) == preview_receipt &&
       HyNomasBoundaryCaptureSpec.sha256(preview_receipt) == HyNomasBoundaryCaptureSpec::PREVIEW_RECEIPT_SHA
  raise ArgumentError.new("saved preview receipt identity mismatch")
end
receipt = JSON.parse(File.read(preview_receipt))
diagnostics = receipt["diagnostics"]
unless receipt["schema"].as_s == "cogni_ml.hy_nomas_native_continuation/v1" &&
       receipt["upstream_revision"].as_s == HyNomasBoundaryCaptureSpec::UPSTREAM_REVISION &&
       receipt["inputs_sha256"].as_s == manifest_sha && File.realpath(receipt["inputs"].as_s) == manifest &&
       File.realpath(diagnostics["latent_bank"].as_s) == preview_latent &&
       diagnostics["latent_bank_sha256"].as_s == HyNomasBoundaryCaptureSpec::PREVIEW_LATENT_SHA &&
       receipt["latent_bank_shape"].as_a.map(&.as_i) == [1, 32, 180, 9, 16] &&
       receipt["clean_latent_frames"].as_i == 60 && receipt["segments"].as_i == 2 &&
       receipt["frames"].as_i == 240 && receipt["fps"].as_i == 24 &&
       receipt["width"].as_i == 512 && receipt["height"].as_i == 288 &&
       receipt["steps_per_segment"].as_i == 4 && receipt["shift"].as_i == 5 &&
       receipt["cfg"].as_i == 1 && receipt["text_cache_policy"].as_s == "once_per_scene" &&
       receipt["qwen_valid_tokens"].as_i == 557 &&
       receipt["continuation_experimental"].as_bool && !receipt["full_upstream_video_parity"].as_bool
  raise ArgumentError.new("saved preview receipt does not bind the admitted two-segment run")
end
unless File.file?(preview_latent) && !File.symlink?(preview_latent) &&
       File.realpath(preview_latent) == preview_latent &&
       File.size(preview_latent) == 829_440 * sizeof(Float32) &&
       HyNomasBoundaryCaptureSpec.sha256(preview_latent) == HyNomasBoundaryCaptureSpec::PREVIEW_LATENT_SHA
  raise ArgumentError.new("saved preview latent-bank identity mismatch")
end

inputs = HyNomasGeneration::TimelineInputs.new(manifest, manifest_sha)
inputs.verify_payload!
base = inputs.base
config = HyNomasGeneration::CONFIG.compact_valid_text_prefix(base.valid_text_tokens, base.compact_text_tokens)
unless base.valid_text_tokens == 557 && base.compact_text_tokens == 576
  raise ArgumentError.new("real boundary requires valid557/compact576 text geometry")
end
index = HyNomasGeneration.authenticate_bf16_index
pin = index.blocks.first
files = [] of ML::Video::CheckpointFile
arenas = [] of ML::Video::Arena
prepared = [] of ML::Video::PreparedWeights
writer_patch : ML::Video::HyNomasPatchMPSPlan? = nil
writer_refiner : ML::Video::HyNomasTextRefinerMPSPlan? = nil
reader_refiner : ML::Video::HyNomasTextRefinerMPSPlan? = nil
workspace : ML::Video::HyNomasMPSWorkspace? = nil
reader_euler : ML::Video::HyNomasEulerPlan? = nil
timeline_plan : ML::Video::HyNomasTimelineOps::Plan? = nil
history_workspace : ML::Video::HyNomasHistoryAttentionWorkspace? = nil
history_plan : ML::Video::HyNomasHistoryAttentionMPSPlan? = nil
chain : ML::Metal::MPSCommandChain? = nil
begin
  globals_file = ML::Video::BF16WeightCache.authenticate(index.globals_bf16_path, index.globals_bf16_sha256)
  files << globals_file
  block_file = ML::Video::BF16WeightCache.authenticate(pin.bf16_path, pin.bf16_sha256)
  files << block_file
  global_bindings = HyNomasQ4DiTIO::GLOBAL_BINDINGS.map do |binding|
    ML::Video::WeightBinding.new(binding.target, binding.source, binding.shape,
      ML::GGUF::SafeTensorDType::BF16)
  end
  globals_plan = ML::Video::WeightPlan.new([globals_file], global_bindings, budget,
    resident_dtype: ML::Video::WeightDType::BF16, require_all: true)
  block_plan = ML::Video::WeightPlan.new([block_file],
    ML::Video::HyNomasDualStreamBlock.checkpoint_bindings(config,
      ML::GGUF::SafeTensorDType::BF16, block_index: 0), budget,
    resident_dtype: ML::Video::WeightDType::BF16, require_all: true)
  unless globals_plan.staging_bytes == 0 && block_plan.staging_bytes == 0
    raise ArgumentError.new("boundary diagnostic requires exact resident BF16 checkpoint weights")
  end

  segment_elements = 32 * 30 * 9 * 16
  compact_elements = 576 * 2048
  vec_elements = 2048
  rope_elements = 4320 * 128
  scratch_bytes = ML::Video::HyNomasDualStreamBlock.scratch_plan(config).sum do |_, count|
    count.to_i64 * sizeof(Float32)
  end
  refiner_bytes = ML::Video::HyNomasTextRefinerMPSPlan.named_bytes(base.mask, retain_padding: true).total
  compact_attention_bytes = ML::Video::HyNomasCompactAttentionMPSPlan.workspace_geometry(
    1, 4320 + 557, 16, 128, 256, 16).resident_bytes
  history_geometry = ML::Video::HyNomasHistoryAttentionWorkspace.geometry(4877, 4877, 16, 128, 4320)
  patch_bytes = (4320_i64 * 32 + 4320_i64 * 2048 + 2048_i64 * 32) * sizeof(Float32)
  source_bytes = inputs.bytes(["input.latent_bank", "input.rope_cos", "input.rope_sin", "cache.eps_bf16"]) +
                 base.bytes(["input.qwen_bf16"])
  work_bytes = (2_i64 * segment_elements + 2_i64 * compact_elements + 2_i64 * vec_elements +
                2_i64 * rope_elements) * sizeof(Float32)
  capture_bytes = HyNomasBoundaryCaptureSpec.bytes
  workspace_fixed = ML::Video::HyNomasMPSWorkspace.required_weight_stage_count(config).to_i64 * sizeof(Float32) +
                    compact_attention_bytes
  # Named block RMS/normalization stages are bounded by the fixed cushion;
  # opaque MPSGraph temporaries are outside the named-buffer contract.
  preflight = globals_plan.destination_bytes + block_plan.destination_bytes + scratch_bytes +
              source_bytes + work_bytes + capture_bytes + patch_bytes + 2_i64 * refiner_bytes +
              workspace_fixed + history_geometry.workspace_bytes +
              HyNomasGeneration.euler_schedule_bytes(4) + 2_i64 * 1024**3
  unless preflight <= budget
    raise ArgumentError.new("boundary capture named preflight exceeds budget: #{preflight}")
  end
  STDOUT.puts("boundary_attention_admitted named_bytes=#{preflight} capture_bytes=#{capture_bytes}")

  input_arena = ML::Video::Arena.new
  arenas << input_arena
  global_arena = ML::Video::Arena.new
  arenas << global_arena
  block_arena = ML::Video::Arena.new
  arenas << block_arena
  scratch = ML::Video::Arena.new
  arenas << scratch
  capture_arena = ML::Video::Arena.new
  arenas << capture_arena
  globals_prepared = ML::Video::PreparedWeights.new(globals_plan, global_arena)
  prepared << globals_prepared
  block_prepared = ML::Video::PreparedWeights.new(block_plan, block_arena)
  prepared << block_prepared

  timeline = inputs.upload!("input.latent_bank", input_arena)
  cosine_bank = inputs.upload!("input.rope_cos", input_arena)
  sine_bank = inputs.upload!("input.rope_sin", input_arena)
  cache_eps = inputs.upload!("cache.eps_bf16", input_arena)
  qwen = base.upload!("input.qwen_bf16", input_arena)
  latent = input_arena.alloc("segment.latent", segment_elements)
  velocity = input_arena.alloc("segment.velocity", segment_elements)
  text_work = input_arena.alloc("segment.text_work", compact_elements)
  vec_work = input_arena.alloc("segment.vec_work", vec_elements)
  cosine = input_arena.alloc("segment.rope_cos", rope_elements)
  sine = input_arena.alloc("segment.rope_sin", rope_elements)

  # Replace the first 30 clean latent frames with the authenticated saved
  # preview bytes. NCTHW frame slices are strided per channel.
  HyNomasBoundaryCaptureSpec.copy_preview_prefix!(preview_latent, timeline)

  captures = {} of String => HyNomasBoundaryCapture
  HyNomasBoundaryCaptureSpec::TENSORS.each do |name, shape|
    buffer = capture_arena.alloc("capture.#{name}", HyNomasBoundaryCaptureSpec.elements(shape).to_i32)
    captures[name] = HyNomasBoundaryCapture.new(name, shape, buffer)
  end
  writer_patch = ML::Video::HyNomasPatchMPSPlan.new(latent, 4320, global_arena)
  writer_refiner = ML::Video::HyNomasTextRefinerMPSPlan.new(qwen, base.mask, 206.897_f32,
    global_arena, refiner_bytes, retain_padding: true)
  reader_refiner = ML::Video::HyNomasTextRefinerMPSPlan.new(qwen, base.mask, 1000_f32,
    global_arena, refiner_bytes, retain_padding: true)
  reader_euler = ML::Video::HyNomasEulerPlan.new(latent, velocity,
    reader_refiner.checkpoints["timestep"], steps: 4, shift: 5_f32)
  ML::Video::HyNomasDualStreamBlock.prepare_scratch!(scratch, config)
  block = ML::Video::HyNomasDualStreamBlock.new(config, block_arena, scratch)
  workspace = ML::Video::HyNomasMPSWorkspace.new(block, writer_patch.output, text_work,
    vec_work, vec_work, cosine, sine, attention_query_chunk: 256,
    attention_heads_per_group: 16, valid_text_tokens: 557,
    text_rms_tokens: 1280, allow_shared_modulation: true)
  timeline_plan = ML::Video::HyNomasTimelineOps::Plan.new(
    ML::Video::HyNomasTimelineOps::Geometry.new, timeline: timeline, segment_latent: latent,
    cosine_bank: cosine_bank, sine_bank: sine_bank, cosine: cosine, sine: sine,
    cache_eps_bf16: cache_eps, cache_timestep: writer_refiner.checkpoints["timestep"], run_segments: 2)

  history_key = captures["history.key"].buffer
  history_value = captures["history.value"].buffer
  history_workspace = ML::Video::HyNomasHistoryAttentionWorkspace.new(4877, 4877, 16, 128, 4320)
  history_workspace.register_history_pair!(history_key, history_value)
  history_workspace.freeze_history_inventory!
  history_plan = ML::Video::HyNomasHistoryAttentionMPSPlan.new(history_workspace,
    block.checkpoints["joint.query"], block.checkpoints["joint.key"],
    block.checkpoints["joint.value"], captures["reader.history_attention"].buffer)
  append = ML::Video::HyNomasContinuationHistoryKernels::AppendPlan.new
  gather = ML::Metal::PipelineCache.get(ML::Video::HyNomasTimelineOps::GATHER_KERNEL) do
    ML::Metal::ComputePipeline.new(ML::Video::HyNomasTimelineOps::GATHER_KERNEL,
      ML::Video::HyNomasTimelineOps::SOURCE)
  end
  rope_copy = ML::Metal::PipelineCache.get(ML::Video::HyNomasTimelineOps::COPY_KERNEL) do
    ML::Metal::ComputePipeline.new(ML::Video::HyNomasTimelineOps::COPY_KERNEL,
      ML::Video::HyNomasTimelineOps::SOURCE)
  end

  actual_named = arenas.sum(&.resident_bytes) + writer_patch.resident_bytes +
                 writer_refiner.resident_bytes + reader_refiner.resident_bytes +
                 workspace.resident_bytes + reader_euler.resident_bytes +
                 history_workspace.resident_bytes + timeline_plan.plan_buffer_bytes
  unless actual_named <= preflight && actual_named <= budget
    raise ArgumentError.new("actual named buffers exceeded preflight/budget: #{actual_named} > #{preflight}")
  end

  chain = ML::Metal::MPSCommandChain.new
  before_allocations = ML::MetalBuffer.hy_text_probe_allocations
  before_readbacks = ML::MetalBuffer.hy_text_probe_readbacks

  globals_prepared.encode(chain.encoder)
  block_prepared.encode(chain.encoder)

  # Writer0: start from saved clean latent, apply the authenticated separate
  # cache-noise kernel, then run the production patch/refiner/block0 plans.
  timeline_plan.prepare_segment!(chain, 0)
  ML::Video::MetalOps.copy(chain.encoder, latent, captures["input.latent.writer_clean"].buffer, segment_elements)
  timeline_plan.mark_denoise_complete!(chain, 0)
  timeline_plan.scatter_generated!(chain, 0)
  timeline_plan.write_cache_noise!(chain, 0)
  ML::Video::MetalOps.copy(chain.encoder, latent, captures["input.latent.writer_noisy"].buffer, segment_elements)
  ML::Video::MetalOps.copy(chain.encoder, cosine, captures["input.rope_cos.writer"].buffer, rope_elements)
  ML::Video::MetalOps.copy(chain.encoder, sine, captures["input.rope_sin.writer"].buffer, rope_elements)
  writer_patch.encode(chain)
  writer_refiner.encode(chain)
  ML::Video::MetalOps.copy(chain.encoder, writer_refiner.text_output, text_work, compact_elements)
  ML::Video::MetalOps.copy(chain.encoder, writer_refiner.vec, vec_work, vec_elements)
  ML::Video::MetalOps.copy(chain.encoder, writer_patch.output, captures["input.img.writer"].buffer, 4320 * 2048)
  ML::Video::MetalOps.copy(chain.encoder, writer_refiner.text_output, captures["input.txt.writer"].buffer, compact_elements)
  ML::Video::MetalOps.copy(chain.encoder, writer_refiner.vec, captures["input.vec.writer"].buffer, vec_elements)
  ML::Video::MetalOps.copy(chain.encoder, writer_refiner.vec, captures["input.vec_txt.writer"].buffer, vec_elements)
  block.encode_mps(chain, writer_patch.output, text_work, vec_work, vec_work, cosine, sine, workspace)
  %w(query key value).each do |axis|
    ML::Video::MetalOps.copy(chain.encoder, block.checkpoints["joint.#{axis}"],
      captures["writer.#{axis}"].buffer, 4896 * 2048)
  end
  append.encode(chain.encoder, block.checkpoints["joint.key"], history_key, 4877 * 2048, 0)
  append.encode(chain.encoder, block.checkpoints["joint.value"], history_value, 4877 * 2048, 0)

  # Reader1: gather authentic continuation latents and global RoPE table 1,
  # preserve the pre/post initial BF16 Euler values, then run its own t=1000
  # text refiner and block0 current-only compact path.
  chain.encoder.set_pipeline(gather)
    .set_buffer(timeline, 0)
    .set_buffer(latent, 1, ML::Metal::BufferAccess::Write)
    .set_value(30_u32, 2).set_value(180_u32, 3).set_value(30_u32, 4)
    .set_value(9_u32, 5).set_value(16_u32, 6).set_value(segment_elements.to_u32, 7)
    .dispatch_1d(segment_elements)
  chain.encoder.set_pipeline(rope_copy)
    .set_buffer(cosine_bank, 0).set_buffer(cosine, 1, ML::Metal::BufferAccess::Write)
    .set_value((4320 * 128).to_u32, 2).set_value(rope_elements.to_u32, 3).dispatch_1d(rope_elements)
  chain.encoder.set_pipeline(rope_copy)
    .set_buffer(sine_bank, 0).set_buffer(sine, 1, ML::Metal::BufferAccess::Write)
    .set_value((4320 * 128).to_u32, 2).set_value(rope_elements.to_u32, 3).dispatch_1d(rope_elements)
  ML::Video::MetalOps.copy(chain.encoder, latent, captures["input.latent.reader_initial"].buffer, segment_elements)
  ML::Video::MetalOps.copy(chain.encoder, cosine, captures["input.rope_cos.reader"].buffer, rope_elements)
  ML::Video::MetalOps.copy(chain.encoder, sine, captures["input.rope_sin.reader"].buffer, rope_elements)
  reader_euler.encode_timestep(chain, 0)
  ML::Video::MetalOps.copy(chain.encoder, latent, captures["input.latent.reader_model"].buffer, segment_elements)
  writer_patch.encode(chain)
  reader_refiner.encode(chain)
  ML::Video::MetalOps.copy(chain.encoder, reader_refiner.text_output, text_work, compact_elements)
  ML::Video::MetalOps.copy(chain.encoder, reader_refiner.vec, vec_work, vec_elements)
  ML::Video::MetalOps.copy(chain.encoder, writer_patch.output, captures["input.img.reader"].buffer, 4320 * 2048)
  ML::Video::MetalOps.copy(chain.encoder, reader_refiner.text_output, captures["input.txt.reader"].buffer, compact_elements)
  ML::Video::MetalOps.copy(chain.encoder, reader_refiner.vec, captures["input.vec.reader"].buffer, vec_elements)
  ML::Video::MetalOps.copy(chain.encoder, reader_refiner.vec, captures["input.vec_txt.reader"].buffer, vec_elements)
  block.encode_mps(chain, writer_patch.output, text_work, vec_work, vec_work, cosine, sine, workspace)
  %w(query key value).each do |axis|
    ML::Video::MetalOps.copy(chain.encoder, block.checkpoints["joint.#{axis}"],
      captures["reader.#{axis}"].buffer, 4896 * 2048)
  end
  history_plan.encode(chain, 0)

  allocations = ML::MetalBuffer.hy_text_probe_allocations - before_allocations
  readbacks = ML::MetalBuffer.hy_text_probe_readbacks - before_readbacks
  unless allocations == 0 && readbacks == 0 && chain.wait_count == 0
    raise "boundary encode allocated, read back, or waited before the terminal drain"
  end
  chain.commit_and_wait
  unless chain.wait_count == 1
    raise "boundary diagnostic requires exactly one terminal chain wait"
  end

  records = {} of String => NamedTuple(dtype: String, shape: Array(Int64), offset: Int64, nbytes: Int64, sha256: String)
  offset = 0_i64
  payload_fd = HyNomasBoundaryCaptureSpec.open_fresh_output(payload)
  payload_digest = Digest::SHA256.new
  begin
    HyNomasBoundaryCaptureSpec::TENSORS.each do |name, shape|
      buffer = captures[name].buffer
      bytes = Bytes.new(buffer.contents.as(UInt8*), buffer.size.to_i32)
      HyNomasBoundaryCaptureSpec.write_all!(payload_fd, bytes, payload)
      tensor_digest = Digest::SHA256.new
      tensor_digest.update(bytes)
      payload_digest.update(bytes)
      records[name] = {dtype: "F32", shape: shape.map(&.to_i64), offset: offset,
                       nbytes: buffer.size, sha256: tensor_digest.hexfinal}
      offset += buffer.size
    end
  ensure
    HyNomasBoundaryCaptureSpec.close_output!(payload_fd, payload)
  end
  payload_sha = payload_digest.hexfinal
  report = JSON.build(indent: "  ") do |json|
    json.object do
      json.field "schema", HyNomasBoundaryCaptureSpec::SCHEMA
      json.field "upstream_revision", HyNomasBoundaryCaptureSpec::UPSTREAM_REVISION
      json.field "conditioning_manifest", manifest
      json.field "conditioning_sha256", manifest_sha
      json.field "preview_receipt", preview_receipt
      json.field "preview_receipt_sha256", HyNomasBoundaryCaptureSpec::PREVIEW_RECEIPT_SHA
      json.field "preview_latent_bank", preview_latent
      json.field "preview_latent_bank_sha256", HyNomasBoundaryCaptureSpec::PREVIEW_LATENT_SHA
      json.field "weight_index", File.realpath(HyNomasGeneration::INDEX)
      json.field "weight_index_sha256", HyNomasGeneration::INDEX_SHA
      json.field "block_weights", File.realpath(pin.bf16_path)
      json.field "block_weights_sha256", pin.bf16_sha256
      json.field "global_weights", File.realpath(index.globals_bf16_path)
      json.field "global_weights_sha256", index.globals_bf16_sha256
      json.field "geometry", {
        batch: 1, segments: 2, video_tokens: 4320, text_storage_tokens: 576,
        valid_text_tokens: 557, joint_storage_rows: 4896, active_rows: 4877,
        heads: 16, head_dim: 128, dim: 2048, history_rows: 4877,
        current_rows: 4320, query_rows: 4877, key_rows: 9197,
      }
      json.field "timesteps", {writer_cache: 206.897, reader_initial: 1000}
      json.field "cache_noise", {
        sigma: 0.206897, timestep: 206.897,
        writer_latent_output: "BF16-RNE expanded to F32 after separate F32 mul/add",
        eps_source: "conditioning manifest cache.eps_bf16 segment0",
      }
      json.field "history", {
        writer_rows: 4877, reader_history_rows: 4877, reader_current_rows: 4320,
        reader_key_rows: 9197, writer_kv_source: "block0 QKV first 4877 rows",
        append: "HyNomasContinuationHistoryKernels::AppendPlan",
        format: "F32 BQHD row-major",
        reader_attention: "direct HyNomasHistoryAttentionMPSPlan output; separate from block0 compact current-only attention",
      }
      json.field "aliases", {native_vec_txt_alias_of_vec: true, capture_vec_txt_is_owned_copy: true}
      json.field "precision", "BF16 checkpoint weights; F32 native activation/MPS arithmetic; cache-noise BF16 RNE; imported initial reader latent Euler BF16 RNE; F32 persistent history K/V"
      json.field "scope", "authentic writer0 and reader1 patch/refiner/block0 inputs/QKV; direct history attention only; block0 default attention remains native current-only compact attention and is unused; no attached history provider, history-aware block output, layers>0, full DiT, or media generation"
      json.field "named_bytes", actual_named
      json.field "encode_allocations", allocations
      json.field "encode_readbacks", readbacks
      json.field "preterminal_waits", 0
      json.field "terminal_waits", chain.wait_count
      json.field "payload", File.basename(payload)
      json.field "payload_bytes", offset
      json.field "payload_sha256", payload_sha
      json.field "tensors", records
    end
  end
  report_fd = HyNomasBoundaryCaptureSpec.open_fresh_output(output)
  begin
    HyNomasBoundaryCaptureSpec.write_all!(report_fd, report.to_slice, output)
  ensure
    HyNomasBoundaryCaptureSpec.close_output!(report_fd, output)
  end
  STDOUT.puts("boundary_attention_captured capture=#{output} named_bytes=#{actual_named} tensors=#{records.size}")
ensure
  Qwen25PrefixProbe.cleanup_after_drain(chain) do
    history_plan.try(&.release)
    history_workspace.try(&.release)
    workspace.try(&.release)
    reader_euler.try(&.release)
    reader_refiner.try(&.release)
    writer_refiner.try(&.release)
    writer_patch.try(&.release)
    if active = chain
      timeline_plan.try(&.release_after_drain!(active))
    end
    chain.try(&.release)
    prepared.reverse_each(&.release)
    arenas.reverse_each(&.release)
    files.reverse_each(&.close)
  end
end
