require "option_parser"
require "./support/hy_nomas_generation"

input_path = ""
input_sha = ""
reference_path = ""
reference_sha = ""
report = ""
resolution = "512x288"
OptionParser.parse do |p|
  p.on("--inputs PATH", "Fixed generation input manifest") { |v| input_path = v }
  p.on("--inputs-sha SHA", "Trusted input identity") { |v| input_sha = v }
  p.on("--reference PATH", "Pinned upstream new-boundary oracle") { |v| reference_path = v }
  p.on("--reference-sha SHA", "Trusted reference identity") { |v| reference_sha = v }
  p.on("--report PATH", "New report") { |v| report = v }
  p.on("--resolution RES", "Authenticated v3 probe geometry (512x288 or 768x416)") { |v| resolution = v }
end
geometry = HyNomasGeneration::SpatialGeometry.parse(resolution)
inputs = HyNomasGeneration::Inputs.new(input_path, input_sha,
  conditioned: geometry.high_resolution?, geometry: geometry)
oracle = HyNomasGeneration::TensorFile.new(File.realpath(reference_path), reference_sha)
doc = oracle.document
reference_schema = geometry.high_resolution? ? "cogni_ml.pixelumm_generation_reference/v2" : "cogni_ml.pixelumm_generation_reference/v1"
reference_version = geometry.high_resolution? ? 2 : 1
unless doc["schema"].as_s == reference_schema && doc["schema_version"].as_i == reference_version &&
       doc["inputs_sha256"].as_s == inputs.file.sha256
  raise ArgumentError.new("PixelUMM generation oracle lineage mismatch")
end
if geometry.high_resolution?
  unless doc["resolution"].as_s == "768x416" &&
         doc["inputs_schema"].as_s == "cogni_ml.hy_nomas_generation_inputs/v3" &&
         doc["geometry"]["latent_grid"].as_a.map(&.as_i) == [30, 13, 24] &&
         doc["geometry"]["stem_token_grid"].as_a.map(&.as_i) == [10, 26, 48] &&
         doc["geometry"]["stem_shape"].as_a.map(&.as_i) == [1, 12_480, 4096] &&
         doc["geometry"]["pixel_shape"].as_a.map(&.as_i) == [1, 3, 120, 416, 768]
    raise ArgumentError.new("high-resolution PixelUMM oracle geometry mismatch")
  end
end
expected = oracle.records("tensors")
config = inputs.geometry.pixelumm_config
stem_shape = [1, config.token_count, ML::Video::PixelUMMGeneration::HIDDEN]
unless expected.keys.sort == %w(assembly.output_f32 stem.output_bf16) &&
       expected["stem.output_bf16"].dtype == "BF16" && expected["stem.output_bf16"].shape == stem_shape &&
       expected["assembly.output_f32"].dtype == "F32" && expected["assembly.output_f32"].shape == config.output_shape
  raise ArgumentError.new("PixelUMM generation oracle inventory mismatch")
end
oracle.validate_ranges!(expected.values, contiguous: true)
oracle.verify_payload!
inputs.file.verify_payload!
file = HyNomasGeneration.authenticate_f32(HyNomasGeneration::STEM_FILE, HyNomasGeneration::STEM_SHA)
plan = ML::Video::WeightPlan.new([file], HyNomasGeneration.stem_bindings, 512_i64 * 1024 * 1024, require_all: true)
arena = ML::Video::Arena.new
weights = ML::Video::Arena.new
prepared = ML::Video::PreparedWeights.new(plan, weights)
latent = inputs.upload!("input.latent", arena)
noise = inputs.upload!("decoder.noise", arena)
expected.each do |name, record|
  dtype = record.dtype == "BF16" ? ML::Video::WeightDType::BF16 : ML::Video::WeightDType::F32
  b = arena.alloc_weight(name, (record.nbytes // dtype.bytes_per_element).to_i32, dtype)
  oracle.upload_into!(record, b)
end
stem = ML::Video::PixelUMMGeneration::StemPlan.new(config, latent: latent, noise: noise,
  weights: ML::Video::PixelUMMGeneration::StemWeights.new(
    embed_weight: weights["embed.weight"], embed_bias: weights["embed.bias"],
    cond_weight: weights["cond_proj.weight"], cond_bias: weights["cond_proj.bias"]))
assembly = ML::Video::PixelUMMGeneration::AssemblyPlan.new(config)
window = arena.alloc_weight("synthetic_window", config.window_output_elements.to_i32, ML::Video::WeightDType::BF16)
status = arena.alloc("status", 6)
status.contents.as(UInt8*).to_slice(status.size.to_i32).fill(0_u8)
diagnostics = arena.alloc("frame_diagnostics", config.pixel_frames * 4)
diagnostics.contents.as(UInt32*).to_slice(config.pixel_frames * 4).fill(0_u32)
config.pixel_frames.times { |frame| diagnostics.contents.as(UInt32*)[frame * 4 + 1] = UInt32::MAX }
source = {{ read_file("#{__DIR__}/support/pixelumm_generation_probe.metal") }}
fill = ML::Metal::ComputePipeline.new("pgen_probe_window", source)
compare_bf16 = ML::Metal::ComputePipeline.new("pgen_probe_compare_bf16", source)
compare_f32 = ML::Metal::ComputePipeline.new("pgen_probe_compare_f32", source)
sample_frames = ML::Metal::ComputePipeline.new("pgen_probe_sample_frames", source)
nan = ML::Metal::ComputePipeline.new("pgen_probe_nan", source)
chain = ML::Metal::MPSCommandChain.new
begin
  allocations = ML::MetalBuffer.hy_text_probe_allocations
  readbacks = ML::MetalBuffer.hy_text_probe_readbacks
  prepared.encode(chain.encoder)
  stem.encode(chain, 0)
  stem_count = config.token_count * ML::Video::PixelUMMGeneration::HIDDEN
  chain.encoder.set_pipeline(compare_bf16).set_buffer(stem.output_bf16, 0)
    .set_buffer(arena["stem.output_bf16"], 1).set_buffer(status, 2, ML::Metal::BufferAccess::ReadWrite)
    .set_value(stem_count.to_u32, 3).dispatch_1d(stem_count)
  ML::Video::PixelUMMGeneration::WINDOW_STARTS.each_with_index do |start, i|
    chain.encoder.set_pipeline(fill).set_buffer(window, 0, ML::Metal::BufferAccess::Write)
      .set_value(i.to_u32, 1).set_value(config.window_output_elements.to_u32, 2)
      .dispatch_1d(config.window_output_elements.to_i32)
    assembly.encode_merge(chain, window_bf16: window, start_latent_frame: start)
  end
  assembly.encode_finalize(chain)
  chain.encoder.set_pipeline(compare_f32).set_buffer(assembly.output_f32, 0)
    .set_buffer(arena["assembly.output_f32"], 1).set_buffer(status, 2, ML::Metal::BufferAccess::ReadWrite, offset: 8)
    .set_value(config.output_elements.to_u32, 3).set_buffer(diagnostics, 4, ML::Metal::BufferAccess::ReadWrite)
    .set_value((config.pixel_height * config.pixel_width).to_u32, 5).set_value(config.pixel_frames.to_u32, 6)
    .dispatch_1d(config.output_elements.to_i32)
  chain.encoder.set_pipeline(sample_frames).set_buffer(assembly.output_f32, 0)
    .set_buffer(arena["assembly.output_f32"], 1).set_buffer(diagnostics, 2, ML::Metal::BufferAccess::ReadWrite)
    .set_value(config.pixel_frames.to_u32, 3).dispatch_1d(config.pixel_frames)
  chain.encoder.set_pipeline(nan).set_buffer(assembly.output_f32, 0, ML::Metal::BufferAccess::Write).dispatch_1d(1)
  chain.encoder.set_pipeline(compare_f32).set_buffer(assembly.output_f32, 0)
    .set_buffer(arena["assembly.output_f32"], 1).set_buffer(status, 2, ML::Metal::BufferAccess::ReadWrite, offset: 16)
    .set_value(1_u32, 3).set_buffer(diagnostics, 4, ML::Metal::BufferAccess::ReadWrite)
    .set_value((config.pixel_height * config.pixel_width).to_u32, 5).set_value(config.pixel_frames.to_u32, 6)
    .dispatch_1d(1)
  encode_allocations = ML::MetalBuffer.hy_text_probe_allocations - allocations
  encode_readbacks = ML::MetalBuffer.hy_text_probe_readbacks - readbacks
  preterminal_waits = chain.wait_count
  chain.commit_and_wait
  s = status.contents.as(UInt32*)
  d = diagnostics.contents.as(UInt32*)
  failed_frames = (0...config.pixel_frames).select { |frame| d[frame * 4] > 0 }.map do |frame|
    {frame: frame, failures: d[frame * 4], first_index: d[frame * 4 + 1],
     actual: d[frame * 4 + 2].unsafe_as(Float32), expected: d[frame * 4 + 3].unsafe_as(Float32)}
  end
  result = {schema: "cogni_ml.pixelumm_generation_comparison/v1",
            inputs_sha256: input_sha, reference_sha256: reference_sha,
            atol: 2e-5, rtol: 2e-5, stem_failures: s[0], stem_max_abs: s[1].unsafe_as(Float32),
            assembly_failures: s[2], assembly_max_abs: s[3].unsafe_as(Float32), nonfinite_negative_failures: s[4],
            encode_allocations: encode_allocations, encode_readbacks: encode_readbacks, preterminal_waits: preterminal_waits,
            waits: chain.wait_count, gpu_seconds: chain.gpu_elapsed_seconds,
            failed_frames: failed_frames,
            passed: s[0] == 0 && s[2] == 0 && s[4] == 1 && encode_allocations == 0 && encode_readbacks == 0 && preterminal_waits == 0}
  HyNomasQ4DiTIO.write_new!(report, result.to_pretty_json + "\n")
  puts result.to_json
  raise "PixelUMM generation boundaries failed unchanged upstream tolerances" unless result[:passed]
ensure
  chain.commit_and_wait
  assembly.release
  stem.release
  prepared.release
  weights.release
  arena.release
  file.close
  chain.release
end
