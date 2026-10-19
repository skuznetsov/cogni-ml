# Bounded residency probe, not a complete S2V inference pipeline.
# Run under scripts/run_safe.sh with COGNI_RUN_SAFE_MIN_FREE_PCT=15.
# Arguments: checkpoint directory, total budget MiB, other-residency reserve MiB.
require "json"
require "digest/sha256"
require "../src/ml/video/checkpoint"

private WAN_WEIGHT_REVISION = "dab4e9c55bbe4c8c4d03db1c2c98c7f0ac9c454b"
private WAN_WEIGHT_BYTES    = 32_591_511_218_i64
private WAN_HEADER_SHA256   = {
  "diffusion_pytorch_model-00001-of-00004.safetensors" => "0bd9ec9ac327117b16bd9acc09231d391dc6b527fb60871ff1f3a5881309a568",
  "diffusion_pytorch_model-00002-of-00004.safetensors" => "e8c4b8221c736f60691935f4a2d9ab8dc5389ce7b3628234934461b32f4a973d",
  "diffusion_pytorch_model-00003-of-00004.safetensors" => "3d572fab1887a7b32b9c357de709eb7c20f08e8b26de816c070b51ea6a79ff52",
  "diffusion_pytorch_model-00004-of-00004.safetensors" => "dcbd76c7066e5c6d17351d0218a643597f5d736003b0facbbc121ba817735c9d",
}

# Every BF16 word is read on GPU. Each chunk exports an integer checksum and
# non-finite count only at the terminal boundary; no neural activation is read.
private WAN_SCAN_WORDS  = 262_144
private WAN_SCAN_SOURCE = <<-METAL
#include <metal_stdlib>
using namespace metal;
kernel void wan_residency_scan(
    device const ushort *weights [[buffer(0)]], device uint *result [[buffer(1)]],
    constant uint &count [[buffer(2)]], constant uint &output_offset [[buffer(3)]],
    uint chunk [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]], uint sg [[simdgroup_index_in_threadgroup]]) {
    threadgroup uint sums[8], invalids[8];
    const uint start = chunk * 262144u, end = min(start + 262144u, count);
    uint sum = 0, invalid = 0;
    for (uint i = start + tid; i < end; i += 256u) {
        const uint bits = uint(weights[i]);
        sum += bits * (i + 1u);
        invalid += ((bits & 0x7f80u) == 0x7f80u);
    }
    sum = simd_sum(sum);
    invalid = simd_sum(invalid);
    if (lane == 0) { sums[sg] = sum; invalids[sg] = invalid; }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid == 0) {
        uint total = 0, bad = 0;
        for (uint i = 0; i < 8u; ++i) { total += sums[i]; bad += invalids[i]; }
        result[2u * (output_offset + chunk)] = total;
        result[2u * (output_offset + chunk) + 1u] = bad;
    }
}
METAL

private def wan_probe_command(command : String, args : Array(String)) : String
  output = IO::Memory.new
  status = Process.run(command, args, output: output, error: Process::Redirect::Close)
  raise "Cannot read memory admission signal #{command}" unless status.success?
  output.to_s
end

unless ARGV.size == 3
  abort "Usage: wan_s2v_weight_residency CHECKPOINT_DIR BUDGET_MIB RESERVE_MIB"
end
directory = File.expand_path(ARGV[0])
budget = ARGV[1].to_i64 * 1024 * 1024
reserve = ARGV[2].to_i64 * 1024 * 1024
floor = ENV["COGNI_RUN_SAFE_MIN_FREE_PCT"]?.try(&.to_i) || 15
raise "Memory floor must be at least 15% and below 100%" unless 15 <= floor < 100
pressure = wan_probe_command("/usr/bin/memory_pressure", ["-Q"])
free_pct = pressure.match(/System-wide memory free percentage:\s*(\d+)%/).try(&.[1].to_i) || raise "Missing memory headroom signal"
physical_bytes = wan_probe_command("/usr/sbin/sysctl", ["-n", "hw.memsize"]).strip.to_i64
raise "Insufficient memory headroom" unless free_pct > floor
headroom_estimate = physical_bytes * (free_pct - floor) // 100
budget = Math.min(budget, headroom_estimate)

files = [] of ML::Video::CheckpointFile
arena = nil.as(ML::Video::Arena?)
prepared = nil.as(ML::Video::PreparedWeights?)
scan = nil.as(ML::MetalBuffer?)
started = Time.instant
begin
  index = JSON.parse(File.read(File.join(directory, "diffusion_pytorch_model.safetensors.index.json")))
  mapping = index["weight_map"].as_h.transform_values(&.as_s)
  raise "Unexpected full checkpoint index" unless mapping.size == 1260 &&
                                                  index["metadata"]["total_size"].as_i64 == WAN_WEIGHT_BYTES &&
                                                  mapping.values.uniq.sort == WAN_HEADER_SHA256.keys.sort
  bindings = [] of ML::Video::WeightBinding
  WAN_HEADER_SHA256.each do |name, expected_sha|
    path = File.join(directory, name)
    file = ML::Video::CheckpointFile.new(path)
    files << file
    File.open(path, "rb") do |io|
      io.skip(8)
      header = Bytes.new((file.data_offset - 8).to_i32)
      io.read_fully(header)
      raise "Pinned checkpoint header mismatch #{name}" unless Digest::SHA256.hexdigest(header) == expected_sha
    end
    file.tensors.each do |tensor_name, info|
      raise "Checkpoint index/shard mismatch #{tensor_name}" unless mapping[tensor_name]? == name
      bindings << ML::Video::WeightBinding.new(tensor_name, tensor_name, info.shape, ML::GGUF::SafeTensorDType::BF16)
    end
  end
  raise "Incomplete full checkpoint" unless bindings.size == mapping.size
  plan = ML::Video::WeightPlan.new(files, bindings, budget, reserve,
    resident_dtype: ML::Video::WeightDType::BF16, require_all: true)
  raise "Wrong full checkpoint byte count" unless plan.destination_bytes == WAN_WEIGHT_BYTES && plan.staging_bytes == 0
  entries = plan.entries
  chunks = entries.sum(0_i64) { |entry| (entry.info.n_elements + WAN_SCAN_WORDS - 1) // WAN_SCAN_WORDS }
  scan_bytes = chunks * 2 * sizeof(UInt32)
  raise "Scan output must fit caller reserve" unless reserve >= scan_bytes

  arena = ML::Video::Arena.new
  device = ML::Metal::Device.instance
  pipeline = ML::Metal::ComputePipeline.new("wan_residency_scan", WAN_SCAN_SOURCE)
  baseline = ML::MetalBuffer.stats
  device_before = device.current_allocated_size
  prepared = ML::Video::PreparedWeights.new(plan, arena)
  loaded = ML::MetalBuffer.stats
  device_loaded = device.current_allocated_size
  raise "Unexpected buffer count" unless loaded[:live_buffers] - baseline[:live_buffers] == 1260
  raise "Unexpected weight mirror or missing storage" unless loaded[:live_bytes] - baseline[:live_bytes] == WAN_WEIGHT_BYTES
  loaded_seconds = (Time.instant - started).total_seconds
  scan = ML::MetalBuffer.new(scan_bytes)
  command = ML::Metal::CommandBuffer.new
  encoder = ML::Metal::ComputeEncoder.new(command)
  prepared.encode(encoder)
  offset = 0_u32
  entries.each do |entry|
    count = entry.info.n_elements
    count_chunks = (count + WAN_SCAN_WORDS - 1) // WAN_SCAN_WORDS
    encoder.set_pipeline(pipeline)
    encoder.set_buffer(arena[entry.binding.target], 0)
    encoder.set_buffer(scan, 1, ML::Metal::BufferAccess::Write)
    encoder.set_value(count.to_u32, 2).set_value(offset, 3)
    encoder.dispatch_threadgroups({count_chunks.to_i32, 1, 1}, {256, 1, 1})
    offset += count_chunks.to_u32
  end
  encoder.end_encoding
  command.commit_and_wait
  result = Bytes.new(scan_bytes.to_i32)
  scan.read_bytes(result.to_unsafe, result.size)
  words = result.to_unsafe.as(UInt32*)
  invalid = (0...chunks.to_i32).sum(0_i64) { |i| words[2 * i + 1].to_i64 }
  raise "Non-finite original BF16 weights: #{invalid}" unless invalid == 0
  puts({
    scope: "Complete Wan diffusion-transformer BF16 residency and GPU word scan; not full S2V inference",
    checkpoint_revision: WAN_WEIGHT_REVISION, tensor_count: entries.size,
    destination_bytes: plan.destination_bytes, staging_bytes: plan.staging_bytes,
    reserve_bytes: reserve, admission_budget_bytes: budget, admission_free_pct: free_pct,
    configured_free_floor_pct: floor, admission_headroom_estimate_bytes: headroom_estimate,
    device: device.name, unified_memory: device.has_unified_memory?,
    recommended_working_set_bytes: device.recommended_working_set_size,
    buffer_baseline: baseline, buffer_loaded: loaded,
    device_before_bytes: device_before, device_loaded_bytes: device_loaded,
    device_delta_bytes: device_loaded - device_before,
    gpu_scan_chunks: chunks, gpu_scan_output_bytes: scan_bytes, non_finite_weights: invalid,
    gpu_scan_sha256: Digest::SHA256.hexdigest(result),
    setup_seconds: loaded_seconds, total_seconds: (Time.instant - started).total_seconds,
  }.to_json)
ensure
  scan.try(&.release)
  prepared.try(&.release)
  arena.try(&.release)
  files.each(&.close)
end
