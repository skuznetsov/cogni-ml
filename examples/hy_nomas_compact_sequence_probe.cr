require "option_parser"
require "./support/hy_nomas_sequence"

# Opt-in DiT-only interval probe. Every process has one terminal drain; compact
# checkpoints preserve GPU-selected moments and codes without CPU quantization.
input_path = ""
input_sha = ""
output_dir = ""
state_path = ""
state_sha = ""
segments = 3
start_segment = 0
stop_after_segments = 3
state_format = "exact"
cache_roundtrip = "exact"
budget_mib = 30_720_i64
plan_only = false
noise_edit_segment : Int32? = nil
noise_source_segment : Int32? = nil
OptionParser.parse do |p|
  p.banner = "Hy-Nomas bounded multi-process compact continuation qualification"
  p.on("--inputs PATH", "Authenticated continuation input manifest") { |v| input_path = v }
  p.on("--inputs-sha SHA", "Trusted manifest SHA-256") { |v| input_sha = v }
  p.on("--output-dir PATH", "Existing empty SSD output directory") { |v| output_dir = v }
  p.on("--segments N", "Total scene segments, two through six") { |v| segments = v.to_i }
  p.on("--start-segment N", "First segment in this process, zero based") { |v| start_segment = v.to_i }
  p.on("--stop-after-segments N", "Exclusive segment stop") { |v| stop_after_segments = v.to_i }
  p.on("--state-format MODE", "exact or qbit checkpoint representation") { |v| state_format = v }
  p.on("--cache-roundtrip MODE", "exact or qbit at each nonfinal boundary") { |v| cache_roundtrip = v }
  p.on("--state PATH", "Input checkpoint, required for nonzero start") { |v| state_path = v }
  p.on("--state-sha SHA", "Trusted complete checkpoint SHA-256") { |v| state_sha = v }
  p.on("--budget-mib N", "Named GPU buffer cap; external 10% guard required") { |v| budget_mib = v.to_i64 }
  p.on("--plan-only", "Authenticate inputs and report budget without Metal") { plan_only = true }
  p.on("--noise-edit-segment N", "New take for one active noninitial segment; shared scene prompt is unchanged") { |v| noise_edit_segment = v.to_i }
  p.on("--noise-source-segment N", "Unused noise donor in the same six-segment bank; paired with noise-edit-segment") { |v| noise_source_segment = v.to_i }
end
HyNomasSequence.validate_segments!(segments)
unless noise_edit_segment.nil? == noise_source_segment.nil?
  raise ArgumentError.new("noise-edit-segment and noise-source-segment must be supplied together")
end
noise_edit = if target = noise_edit_segment
               ML::Video::HyNomasSegmentNoiseEdit.new(segments, target, noise_source_segment.not_nil!)
             end
unless 0 <= start_segment < stop_after_segments <= segments
  raise ArgumentError.new("interval must satisfy 0 <= start < stop <= segments")
end
unless %w(exact qbit).includes?(state_format) && %w(exact qbit).includes?(cache_roundtrip)
  raise ArgumentError.new("checkpoint and roundtrip modes must be exact or qbit")
end
if state_format == "qbit" && cache_roundtrip != "exact"
  raise ArgumentError.new("compact checkpoint intervals cannot also roundtrip their boundaries")
end
if start_segment > 0
  raise ArgumentError.new("nonzero start requires an authenticated state") if state_path.empty? || state_sha.empty?
elsif !state_path.empty? || !state_sha.empty?
  raise ArgumentError.new("zero start cannot consume a state")
end
raise ArgumentError.new("inputs and inputs-sha are required") if input_path.empty? || input_sha.empty?
raise ArgumentError.new("named budget must be within 24..48 GiB") unless (24_576..49_152).includes?(budget_mib)
request_start = Time.monotonic
mode = start_segment > 0 ? "resume" : stop_after_segments < segments ? "save" : "control"
budget = budget_mib * 1024 * 1024
inputs = HyNomasGeneration::TimelineInputs.new(File.realpath(input_path), input_sha)
inputs.verify_payload!
inputs.base.file.verify_payload!
HyNomasSequence.validate_shared_scene_prompt!(inputs.base.prompt)
admission = HyNomasGeneration.authenticate_bf16_index
contracts = {} of Int32 => String
inventories = {} of Int32 => Hash(String, Int64)
(1...segments).each do |boundary|
  contracts[boundary] = HyNomasSequence.checkpoint_contract(inputs, segments, boundary, noise_edit)
  inventories[boundary] = HyNomasSequence.checkpoint_inventory(inputs, segments, boundary)
end
counts = inventories.transform_values do |inventory|
  inventory.reject { |name, _| name == "timeline" }.values.map { |n| (n // sizeof(Float32)).to_i32 }
end
capture_bytes = (stop_after_segments - start_segment).to_i64 * 4 * 32 * 30 * 9 * 16 * sizeof(Float32)
dit_peak = HyNomasSequence.peak_named_bytes(inputs, segments)
roundtrip_bytes = 0_i64
if cache_roundtrip == "qbit"
  ((start_segment + 1)...stop_after_segments).each do |boundary|
    roundtrip_bytes += HyNomasCacheQBit::Plan.named_bytes(counts[boundary].max, counts[boundary].size)
  end
end
pack_bytes = state_format == "qbit" && stop_after_segments < segments ? HyNomasCacheQBitCheckpoint::PackPlan.named_bytes(counts[stop_after_segments]) : 0_i64
restore_bytes = state_format == "qbit" && start_segment > 0 ? HyNomasCacheQBitCheckpoint::PackPlan.named_bytes(counts[start_segment]) : 0_i64
codec_bytes = roundtrip_bytes + pack_bytes + restore_bytes
peak = dit_peak + capture_bytes + codec_bytes
resident_f32_kv_bytes = inventories[segments - 1].reject { |name, _| name == "timeline" }.values.sum
raise ArgumentError.new("interval peak named buffers exceed budget") if peak > budget
HyNomasGeneration.emit("compact_resume_memory_plan", mode: mode, segments: segments,
  start_segment: start_segment, stop_after_segments: stop_after_segments, state_format: state_format,
  cache_roundtrip: cache_roundtrip, dit_peak_named_bytes: dit_peak, capture_bytes: capture_bytes,
  codec_bytes: codec_bytes, peak_named_bytes: peak, named_budget_bytes: budget,
  resident_f32_kv_bytes: resident_f32_kv_bytes, noise_edit: noise_edit.try(&.identity),
  reserve_pct: 10, decoder: false, full_generation: false)
exit if plan_only
unless Path[output_dir].absolute? && File.directory?(output_dir) && Dir.empty?(output_dir)
  raise ArgumentError.new("output must be an existing empty absolute directory")
end
output_dir = File.realpath(output_dir)
unless output_dir.starts_with?(File.realpath("/opt/mldata_ssd") + "/")
  raise ArgumentError.new("outputs must be on /opt/mldata_ssd")
end

# Authenticate the complete input bundle, including all compact metadata, before
# allocating a model. The setup reader remains owned until restore is prepared.
exact_reader = nil.as(ML::Video::HyNomasResumeBundle::Reader?)
compact_reader = nil.as(HyNomasCacheQBitCheckpoint::Preflight?)
if start_segment > 0
  if state_format == "exact"
    exact_reader = ML::Video::HyNomasResumeBundle.open(state_path, state_sha,
      contracts[start_segment], inventories[start_segment])
  else
    compact_reader = HyNomasCacheQBitCheckpoint::Preflight.open(state_path, state_sha,
      contracts[start_segment], inventories[start_segment])
  end
end
timeline_arena = ML::Video::Arena.new
current_arena = ML::Video::Arena.new
capture_arena = ML::Video::Arena.new
dit = nil.as(HyNomasSequence::DiT?)
pack = nil.as(HyNomasCacheQBitCheckpoint::PackPlan?)
roundtrips = {} of Int32 => HyNomasCacheQBit::Plan
chain = nil.as(ML::Metal::MPSCommandChain?)
timings = {"preflight_seconds" => (Time.monotonic - request_start).total_seconds}
metrics = {} of String => Int64
diagnostics = {} of String => String
codec_reports = {} of String => JSON::Any
checkpoint_file_bytes = 0_i64
logical_packed_kv_bytes = 0_i64
request_error : Exception? = nil
cleanup_errors = [] of String
begin
  timeline = inputs.upload!("input.latent_bank", timeline_arena)
  latent = current_arena.alloc("input.latent", 32 * 30 * 9 * 16)
  captures = {} of Int32 => Array(ML::MetalBuffer)
  (start_segment...stop_after_segments).each do |segment|
    captures[segment] = Array.new(4) { |step| capture_arena.alloc("segment#{segment}-step#{step + 1}", 32 * 30 * 9 * 16) }
  end
  setup = Time.monotonic
  dit = HyNomasSequence::DiT.new(inputs, timeline, latent, budget - capture_bytes - codec_bytes,
    segments, admission, prepare_capture: true, noise_edit: noise_edit)
  timings["dit_setup_seconds"] = (Time.monotonic - setup).total_seconds
  prepared_bindings = {} of Int32 => Hash(String, Tuple(ML::MetalBuffer, Int32))
  if cache_roundtrip == "qbit"
    ((start_segment + 1)...stop_after_segments).each do |boundary|
      prepared_bindings[boundary] = dit.prepare_checkpoint_bindings!(boundary)
      bindings = prepared_bindings[boundary].reject { |name, _| name == "timeline" }.map do |name, entry|
        HyNomasCacheQBit::Binding.new(name, entry[0], entry[1])
      end
      roundtrips[boundary] = HyNomasCacheQBit::Plan.new(bindings)
    end
  end
  if state_format == "qbit" && stop_after_segments < segments
    prepared_bindings[stop_after_segments] = dit.prepare_checkpoint_bindings!(stop_after_segments)
    bindings = prepared_bindings[stop_after_segments].reject { |name, _| name == "timeline" }.map do |name, entry|
      HyNomasCacheQBit::Binding.new(name, entry[0], entry[1])
    end
    pack = HyNomasCacheQBitCheckpoint::PackPlan.new(bindings)
    raise "pack admission mismatch" unless pack.resident_bytes == pack_bytes
  end
  if reader = exact_reader
    restore = Time.monotonic
    dit.restore_boundary!(reader, start_segment)
    timings["restore_seconds"] = (Time.monotonic - restore).total_seconds
    reader.close
    exact_reader = nil
  end
  if reader = compact_reader
    restore = Time.monotonic
    dit.restore_compact_boundary!(reader, start_segment)
    raise "restore admission mismatch" unless dit.compact_restore_resident_bytes <= restore_bytes
    codec_reports["restore"] = reader.stats
    timings["restore_seconds"] = (Time.monotonic - restore).total_seconds
    reader.close
    compact_reader = nil
  end
  chain = ML::Metal::MPSCommandChain.new
  allocations = ML::MetalBuffer.hy_text_probe_allocations
  readbacks = ML::MetalBuffer.hy_text_probe_readbacks
  packed_boundary = 0
  callback = ->(boundary : Int32, active_chain : ML::Metal::MPSCommandChain) do
    roundtrips[boundary]?.try(&.encode(active_chain))
    if owner = pack
      if boundary == stop_after_segments
        owner.encode_pack(active_chain)
        packed_boundary = boundary
      end
    end
    nil
  end
  run = Time.monotonic
  dit.encode(chain, stop_after_segments: stop_after_segments, capture_steps: captures, after_boundary: callback)
  metrics["encode_allocations"] = ML::MetalBuffer.hy_text_probe_allocations - allocations
  metrics["encode_readbacks"] = ML::MetalBuffer.hy_text_probe_readbacks - readbacks
  metrics["preterminal_waits"] = chain.wait_count.to_i64
  raise "encode introduced allocation/readback/wait" unless metrics.values.all?(&.zero?)
  raise "compact pack missed terminal boundary" if pack && packed_boundary != stop_after_segments
  chain.commit_and_wait
  timings["dit_wall_seconds"] = (Time.monotonic - run).total_seconds
  timings["dit_gpu_seconds"] = chain.gpu_elapsed_seconds
  metrics["command_segments"] = chain.segment_count.to_i64
  metrics["terminal_waits"] = chain.wait_count.to_i64
  raise "interval must have one terminal wait" unless chain.wait_count == 1
  roundtrips.each do |boundary, owner|
    report = owner.report_after_drain(chain)
    codec_reports["roundtrip#{boundary}"] = report
    raise "nonfinite QBit rows" unless report["invalid_rows"].as_i64 == 0
  end
  export = Time.monotonic
  exports = {"timeline" => timeline}
  captures.each do |segment, buffers|
    buffers.each_with_index { |buffer, step| exports["segment#{segment}-step#{step + 1}"] = buffer }
  end
  exports.each do |name, buffer|
    path = File.join(output_dir, "#{name}.f32le")
    HyNomasGeneration.write_fresh_output(path) do |io|
      io.write(Bytes.new(buffer.contents.as(UInt8*), buffer.size.to_i32))
    end
    diagnostics[name] = path
    diagnostics["#{name}_sha256"] = PixelUMMBlockProbe.sha_file(path)
  end
  if stop_after_segments < segments
    path = File.join(output_dir, state_format == "qbit" ? "boundary.hnqc" : "boundary.hnrs")
    if owner = pack
      timeline_binding = HyNomasCacheQBit::Binding.new("timeline", timeline, (timeline.size // sizeof(Float32)).to_i32)
      diagnostics["state_sha256"] = owner.write_after_drain(chain, path, contracts[stop_after_segments], timeline_binding)
      report = owner.report_after_drain(chain)
      codec_reports["pack"] = report
      logical_packed_kv_bytes = report["logical_packed_kv_bytes"].as_i64
    else
      diagnostics["state_sha256"] = dit.save_boundary!(path)
    end
    diagnostics["state"] = path
    checkpoint_file_bytes = File.size(path)
  end
  timings["terminal_export_seconds"] = (Time.monotonic - export).total_seconds
  timings["request_seconds"] = (Time.monotonic - request_start).total_seconds
  receipt = {schema: noise_edit ? "cogni_ml.hy_nomas_segment_edit_probe/v1" : "cogni_ml.hy_nomas_compact_sequence_probe/v1", mode: mode,
             pid: Process.pid, executable_sha256: PixelUMMBlockProbe.sha_file(Process.executable_path.not_nil!),
             inputs: inputs.file.path, inputs_sha256: inputs.file.sha256,
             payload_sha256: inputs.file.document["payload_sha256"].as_s,
             conditioning_sha256: inputs.base.file.sha256, model_index_sha256: HyNomasGeneration::INDEX_SHA,
             contract_sha256: start_segment > 0 ? Digest::SHA256.hexdigest(contracts[start_segment]) : "",
             boundary_contract_sha256: contracts.transform_keys(&.to_s).transform_values { |v| Digest::SHA256.hexdigest(v) },
             state_input_sha256: state_sha, state_format: state_format, cache_roundtrip: cache_roundtrip,
             qbit_roundtrip_boundaries: roundtrips.keys.sort, compact_pack_boundary: packed_boundary,
             upstream_revision: HyNomasGeneration::TimelineInputs::PINNED_REVISION,
             segments: segments, start_segment: start_segment, stop_after_segments: stop_after_segments,
             resume_next_segment: start_segment, steps_per_segment: 4,
             valid_text_tokens: inputs.base.valid_text_tokens, dit_peak_named_bytes: dit_peak,
             capture_bytes: capture_bytes, codec_bytes: codec_bytes, codec: codec_reports,
             peak_named_bytes: peak, named_budget_bytes: budget,
             checkpoint_file_bytes: checkpoint_file_bytes, logical_packed_kv_bytes: logical_packed_kv_bytes,
             resident_f32_kv_bytes: resident_f32_kv_bytes,
             checkpoint_payload_bytes: stop_after_segments < segments ? inventories[stop_after_segments].values.sum : 0_i64,
             timeline_shape: [1, 32, 180, 9, 16], step_shape: [1, 32, 30, 9, 16],
             timings: timings, metrics: metrics, diagnostics: diagnostics,
             noise_edit: noise_edit.try(&.identity),
             invalidation: noise_edit.try { |edit| {latent_frames: [edit.target_segment * 30, segments * 30],
                                                    decoder_pixel_frames: [edit.target_segment * 120 - 20, segments * 120],
                                                    prior_segment_overlap_pixel_frames: [edit.target_segment * 120 - 20, edit.target_segment * 120],
                                                    units: "half-open global frame ranges; regenerate every descendant"} },
             noise_scope: noise_edit ? "one initial-noise gather from an unused donor in the authenticated state; target RoPE/cache epsilon and shared conditioning unchanged" : "unchanged authenticated latent/epsilon banks; no reseeding",
             io_scope: "setup/terminal shared-buffer file I/O; GPU codec math",
             metric_scope: "instrumented MetalBuffer methods; opaque MPS resources and terminal I/O excluded",
             upstream_scope: "pinned small cache-contract F32/MPS oracle; real DiT native restart parity",
             decoder: false, full_generation: false, full_upstream_video_parity: false}
  HyNomasGeneration.write_fresh_output(File.join(output_dir, "receipt.json")) { |io| io.puts(receipt.to_pretty_json) }
  HyNomasGeneration.emit("compact_resume_completed", start_segment: start_segment,
    stop_after_segments: stop_after_segments, checkpoint_file_bytes: checkpoint_file_bytes, metrics: metrics)
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
    unless may_release
      cleanup_errors << "terminal wait returned without proving GPU drain"
      HyNomasSequence.retain_in_flight do
        inputs.object_id; dit.try(&.object_id); pack.try(&.object_id); roundtrips.object_id
        active_chain.object_id; capture_arena.object_id; current_arena.object_id; timeline_arena.object_id
        nil
      end
    end
  end
  actions = [] of Proc(Nil)
  if may_release
    pack.try { |owner| actions << -> { owner.release_after_drain(chain) } }
    roundtrips.each_value { |owner| actions << -> { owner.release_after_drain(chain) } }
    dit.try { |owner| actions << -> { owner.release } }
    chain.try { |owner| actions << -> { owner.release } }
    actions << -> { capture_arena.release }
    actions << -> { current_arena.release }
    actions << -> { timeline_arena.release }
  end
  exact_reader.try { |owner| actions << -> { owner.close } }
  compact_reader.try { |owner| actions << -> { owner.close } }
  cleanup_errors.concat(HyNomasSequence.release_actions(actions))
end
cleanup_errors.each { |message| STDERR.puts("compact interval cleanup: #{message}") }
raise request_error.not_nil! if request_error
raise "compact interval cleanup failed: #{cleanup_errors.join("; ")}" unless cleanup_errors.empty?
