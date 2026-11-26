require "option_parser"
require "./support/hy_nomas_sequence"
require "./support/hy_nomas_cache_qbit"

# Bounded, DiT-only qualification. The opt-in cache roundtrip isolates lossy
# representation error; exact F32 remains the default and the restart control.
mode = ""
input_path = ""
input_sha = ""
output_dir = ""
state_path = ""
state_sha = ""
budget_mib = 30_720_i64
plan_only = false
cache_roundtrip = "exact"
OptionParser.parse do |p|
  p.banner = "Hy-Nomas BF16 A/save/restart/B qualification; two segments, four steps each"
  p.on("--mode MODE", "control, save, or resume") { |v| mode = v }
  p.on("--inputs PATH", "Authenticated continuation input manifest") { |v| input_path = v }
  p.on("--inputs-sha SHA", "Trusted continuation manifest SHA-256") { |v| input_sha = v }
  p.on("--output-dir PATH", "Existing empty directory on /opt/mldata_ssd") { |v| output_dir = v }
  p.on("--state PATH", "Saved post-A boundary; required by resume") { |v| state_path = v }
  p.on("--state-sha SHA", "Trusted complete state SHA-256; required by resume") { |v| state_sha = v }
  p.on("--budget-mib N", "Named GPU buffer cap; external 10% memory guard is mandatory") { |v| budget_mib = v.to_i64 }
  p.on("--plan-only", "Authenticate and report budget without Metal allocation") { plan_only = true }
  p.on("--cache-roundtrip MODE", "exact (default), or qbit diagnostic before resumed B") { |v| cache_roundtrip = v }
end
unless %w(control save resume).includes?(mode)
  raise ArgumentError.new("--mode must be control, save, or resume")
end
raise ArgumentError.new("--inputs and --inputs-sha are required") if input_path.empty? || input_sha.empty?
unless %w(exact qbit).includes?(cache_roundtrip) && (cache_roundtrip == "exact" || mode == "resume")
  raise ArgumentError.new("QBit roundtrip is admitted only for authenticated resume")
end
unless (24_576..49_152).includes?(budget_mib)
  raise ArgumentError.new("resume probe named budget must be within 24..48 GiB")
end
if mode == "resume"
  raise ArgumentError.new("resume requires --state and --state-sha") if state_path.empty? || state_sha.empty?
elsif !state_path.empty? || !state_sha.empty?
  raise ArgumentError.new("only resume accepts an input state")
end
request_start = Time.monotonic
budget = budget_mib * 1024 * 1024
inputs = HyNomasGeneration::TimelineInputs.new(File.realpath(input_path), input_sha)
inputs.verify_payload!
inputs.base.file.verify_payload!
HyNomasSequence.validate_shared_scene_prompt!(inputs.base.prompt)
admission = HyNomasGeneration.authenticate_bf16_index
contract = HyNomasSequence.checkpoint_contract(inputs, 2, 1)
inventory = HyNomasSequence.checkpoint_inventory(inputs, 2, 1)
capture_bytes = mode == "save" ? 0_i64 : 4_i64 * 32 * 30 * 9 * 16 * sizeof(Float32)
dit_peak = HyNomasSequence.peak_named_bytes(inputs, 2)
cache_sizes = inventory.reject { |name, _| name == "timeline" }.values
codec_bytes = cache_roundtrip == "qbit" ? HyNomasCacheQBit::Plan.named_bytes(
  (cache_sizes.max // sizeof(Float32)).to_i32, cache_sizes.size) : 0_i64
peak = dit_peak + capture_bytes + codec_bytes
raise ArgumentError.new("resume probe peak named buffers exceed budget") if peak > budget
HyNomasGeneration.emit("resume_memory_plan", mode: mode, segments: 2,
  dit_peak_named_bytes: dit_peak, capture_bytes: capture_bytes, codec_bytes: codec_bytes, peak_named_bytes: peak,
  checkpoint_payload_bytes: inventory.values.sum, named_budget_bytes: budget,
  reserve_pct: 10, full_upstream_video_parity: false, decoder: false)
exit if plan_only
unless Path[output_dir].absolute? && File.directory?(output_dir) && Dir.empty?(output_dir)
  raise ArgumentError.new("output directory must be an existing empty absolute directory")
end
output_dir = File.realpath(output_dir)
unless output_dir.starts_with?(File.realpath("/opt/mldata_ssd") + "/")
  raise ArgumentError.new("resume probe outputs must be on /opt/mldata_ssd")
end

# Full file authentication precedes allocation of any model or GPU state.
reader = mode == "resume" ? ML::Video::HyNomasResumeBundle.open(
  state_path, state_sha, contract, inventory) : nil
timeline_arena = ML::Video::Arena.new
current_arena = ML::Video::Arena.new
capture_arena = ML::Video::Arena.new
dit = nil.as(HyNomasSequence::DiT?)
codec = nil.as(HyNomasCacheQBit::Plan?)
codec_report = nil.as(JSON::Any?)
chain = nil.as(ML::Metal::MPSCommandChain?)
timings = {"preflight_seconds" => (Time.monotonic - request_start).total_seconds}
metrics = {} of String => Int64
diagnostics = {} of String => String
request_error : Exception? = nil
cleanup_errors = [] of String
begin
  timeline = inputs.upload!("input.latent_bank", timeline_arena)
  latent = current_arena.alloc("input.latent", 32 * 30 * 9 * 16)
  captures = [] of ML::MetalBuffer
  unless mode == "save"
    4.times { |step| captures << capture_arena.alloc("step#{step}", 32 * 30 * 9 * 16) }
  end
  setup = Time.monotonic
  dit = HyNomasSequence::DiT.new(inputs, timeline, latent, budget - capture_bytes - codec_bytes, 2,
    admission, prepare_capture: mode != "save")
  timings["dit_setup_seconds"] = (Time.monotonic - setup).total_seconds
  if saved = reader
    restore = Time.monotonic
    bindings = [] of HyNomasCacheQBit::Binding
    visitor = if cache_roundtrip == "qbit"
                ->(name : String, buffer : ML::MetalBuffer, count : Int32) do
                  bindings << HyNomasCacheQBit::Binding.new(name, buffer, count)
                  nil
                end
              else
                nil
              end
    dit.restore_boundary!(saved, 1, visit_cache: visitor)
    timings["restore_seconds"] = (Time.monotonic - restore).total_seconds
    saved.close
    reader = nil
    if cache_roundtrip == "qbit"
      codec = HyNomasCacheQBit::Plan.new(bindings)
      raise "QBit scratch admission mismatch" unless codec.resident_bytes == codec_bytes
    end
  end
  chain = ML::Metal::MPSCommandChain.new
  allocations = ML::MetalBuffer.hy_text_probe_allocations
  readbacks = ML::MetalBuffer.hy_text_probe_readbacks
  run = Time.monotonic
  codec.try(&.encode(chain))
  if mode == "save"
    dit.encode(chain, stop_after_segments: 1)
  else
    dit.encode(chain, capture_segment: 1, captures: captures)
  end
  metrics["encode_allocations"] = ML::MetalBuffer.hy_text_probe_allocations - allocations
  metrics["encode_readbacks"] = ML::MetalBuffer.hy_text_probe_readbacks - readbacks
  metrics["preterminal_waits"] = chain.wait_count.to_i64
  raise "resume encode introduced allocation/readback/wait" unless metrics.values.all?(&.zero?)
  chain.commit_and_wait
  timings["dit_wall_seconds"] = (Time.monotonic - run).total_seconds
  timings["dit_gpu_seconds"] = chain.gpu_elapsed_seconds
  metrics["command_segments"] = chain.segment_count.to_i64
  metrics["terminal_waits"] = chain.wait_count.to_i64
  raise "resume probe must have one terminal wait" unless chain.wait_count == 1
  if active_codec = codec
    report = active_codec.report_after_drain(chain)
    codec_report = report
    HyNomasGeneration.emit("qbit_cache_report", report: report)
    raise "QBit cache contains nonfinite rows" unless report["invalid_rows"].as_i64 == 0
  end
  export = Time.monotonic
  {"timeline" => timeline}.merge(captures.map_with_index { |b, i| {"b-step#{i + 1}", b} }.to_h).each do |name, buffer|
    path = File.join(output_dir, "#{name}.f32le")
    HyNomasGeneration.write_fresh_output(path) do |io|
      io.write(Bytes.new(buffer.contents.as(UInt8*), buffer.size.to_i32))
    end
    diagnostics[name] = path
    diagnostics["#{name}_sha256"] = PixelUMMBlockProbe.sha_file(path)
  end
  if mode == "save"
    path = File.join(output_dir, "after-a.hnrs")
    diagnostics["state_sha256"] = dit.save_boundary!(path)
    diagnostics["state"] = path
  end
  timings["terminal_export_seconds"] = (Time.monotonic - export).total_seconds
  timings["request_seconds"] = (Time.monotonic - request_start).total_seconds
  receipt = {schema: "cogni_ml.hy_nomas_sequence_resume_probe/v1", mode: mode,
             pid: Process.pid, inputs: inputs.file.path, inputs_sha256: inputs.file.sha256,
             contract_sha256: Digest::SHA256.hexdigest(contract), state_input_sha256: state_sha,
             upstream_revision: HyNomasGeneration::TimelineInputs::PINNED_REVISION,
             segments: 2, resume_next_segment: 1, steps_per_segment: 4,
             valid_text_tokens: inputs.base.valid_text_tokens,
             dit_peak_named_bytes: dit_peak, capture_bytes: capture_bytes,
             peak_named_bytes: peak, checkpoint_payload_bytes: inventory.values.sum,
             cache_roundtrip: cache_roundtrip, codec_bytes: codec_bytes, codec: codec_report,
             timeline_shape: [1, 32, 180, 9, 16], step_shape: [1, 32, 30, 9, 16],
             timings: timings, metrics: metrics, diagnostics: diagnostics,
             noise_scope: "unchanged authenticated latent/epsilon banks; no reseeding",
             checkpoint: "after A's separate noisy cache writer; exact F32 active K/V and full latent bank",
             io_scope: "shared-buffer file I/O in setup/after terminal wait, outside GPU encode",
             metric_scope: "instrumented MetalBuffer methods; opaque MPS resources and terminal I/O excluded",
             upstream_scope: "small pinned upstream cache-contract oracle is qualified separately; real DiT test is native restart parity",
             decoder: false, full_generation: false, full_upstream_video_parity: false}
  HyNomasGeneration.write_fresh_output(File.join(output_dir, "receipt.json")) { |io| io.puts(receipt.to_pretty_json) }
  HyNomasGeneration.emit("resume_probe_completed", mode: mode, diagnostics: diagnostics, metrics: metrics)
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
        inputs.object_id
        dit.try(&.object_id)
        codec.try(&.object_id)
        active_chain.object_id
        capture_arena.object_id
        current_arena.object_id
        timeline_arena.object_id
        nil
      end
    end
  end
  actions = [] of Proc(Nil)
  if may_release
    codec.try { |owner| actions << -> { owner.release_after_drain(chain) } }
    dit.try { |owner| actions << -> { owner.release } }
    chain.try { |owner| actions << -> { owner.release } }
    actions << -> { capture_arena.release }
    actions << -> { current_arena.release }
    actions << -> { timeline_arena.release }
  end
  reader.try { |owner| actions << -> { owner.close } }
  cleanup_errors.concat(HyNomasSequence.release_actions(actions))
end
cleanup_errors.each { |message| STDERR.puts("resume probe cleanup: #{message}") }
if error = request_error
  raise error
end
raise "resume probe cleanup failed: #{cleanup_errors.join("; ")}" unless cleanup_errors.empty?
