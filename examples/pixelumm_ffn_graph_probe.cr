require "option_parser"
require "process"
require "./support/pixelumm_ffn_graph_probe"

alias G = PixelUMMFFNGraphProbe
alias A = PixelUMMFFNAaProbe
alias C = PixelUMMFFNAaContract
alias B = PixelUMMFFNWeightReuseProbe
alias I = PixelUMMFFNIsolatedProbe

reference = ""
reference_sha = ""
weight_cache = ""
report = ""
split_down = false
stage_outputs = false
OptionParser.parse do |parser|
  parser.banner = "Diagnostic-only J5760 whole-FFN MPSGraph correctness falsifier; no generation; launch under the external cache-aware memory guard"
  parser.on("--reference PATH", "Pinned J5760 windows fixture") { |value| reference = value }
  parser.on("--reference-sha256 SHA", "Trusted fixture manifest digest") { |value| reference_sha = value }
  parser.on("--weight-cache PATH", "Pinned BF16 block-weight cache") { |value| weight_cache = value }
  parser.on("--report PATH", "Fresh absolute JSON path under /Volumes") { |value| report = value }
  parser.on("--split-down", "Use a BF16 product graph boundary and a separate production down graph") { split_down = true }
  parser.on("--stage-outputs", "Materialize gate/up/SiLU/product BF16 stages before the separate down graph") { stage_outputs = true }
  parser.invalid_option { |value| raise ArgumentError.new("unknown whole-FFN graph option: #{value}") }
end
topology = G::Topology.select(split_down, stage_outputs)

raise ArgumentError.new("unexpected whole-FFN graph positional argument") unless ARGV.empty?
unless reference.starts_with?("/") && weight_cache.starts_with?("/") &&
       reference_sha == A::REFERENCE_SHA256
  raise ArgumentError.new("whole-FFN graph probe requires the pinned J5760 fixture and absolute input paths")
end
unless File.file?(reference) && !File.symlink?(reference) && File.realpath(reference) == reference &&
       File.file?(weight_cache) && !File.symlink?(weight_cache) && File.realpath(weight_cache) == weight_cache
  raise ArgumentError.new("whole-FFN graph inputs must be canonical absolute files")
end
unless report.starts_with?("/") && report.ends_with?(".json") &&
       File.directory?(File.dirname(report)) && File.realpath(File.dirname(report)).starts_with?("/Volumes/") &&
       !File.exists?(report) && !File.symlink?(report)
  raise ArgumentError.new("whole-FFN graph report must be a fresh absolute JSON path under /Volumes")
end
report = File.join(File.realpath(File.dirname(report)), File.basename(report))

def free_memory_percent! : Int32
  output = IO::Memory.new
  error = IO::Memory.new
  status = Process.run("memory_pressure", ["-Q"], output: output, error: error)
  raise ArgumentError.new("10% free-memory guard could not read memory_pressure -Q") unless status.success?
  match = output.to_s.match(/System-wide memory free percentage:\s*(\d+)%/)
  value = match.try(&.[1]).try(&.to_i?) ||
          raise ArgumentError.new("10% free-memory guard could not parse memory_pressure output")
  raise ArgumentError.new("10% free-memory guard requires system free memory above 10%") unless value > 10
  value
end

source_paths = [
  __FILE__, "examples/support/pixelumm_ffn_graph_probe.cr",
  "examples/support/pixelumm_ffn_graph_bridge.mm",
  "examples/support/pixelumm_ffn_aa_bridge.mm",
  "examples/support/pixelumm_ffn_aa_probe.cr",
  "examples/support/pixelumm_ffn_aa_contract.cr",
  "examples/support/pixelumm_ffn_isolated_probe.cr",
  "examples/support/pixelumm_ffn_weight_reuse_probe.cr",
  "examples/support/wan_memory_guard.py",
  "examples/support/pixelumm_ffn_dense.metal",
  "src/ml/video/pixelumm_decoder_block_mps.cr",
  "src/ml/video/pixelumm_decoder_block_mps_bridge.mm",
  "src/ml/video/metal_mps_command_chain.cr",
  "src/ml/video/metal_mps_command_chain.mm",
  "src/ml/metal/bridge.mm",
]
source_hashes = source_paths.to_h { |path| {path, G.sha_file(path)} }

fixture = B::Fixture.new(reference, reference_sha)
fixture.verify_payload!
cache = B::WeightCache.new(weight_cache, fixture)
free_percent_before = free_memory_percent!

baseline = nil.as(A::Owner?)
candidate = nil.as(G::Owner?)
chain = nil.as(ML::Metal::MPSCommandChain?)
wait_attempted = false
report_written = false
failure = nil.as(String?)
metrics = [] of G::Metric
stage_failures = [] of String
baseline_counts = [] of UInt64
candidate_counts = [] of UInt64
encode_allocations = -1_i64
preterminal_readbacks = -1_i64
postterminal_readbacks = -1_i64
wait_count = 0_i32
free_percent_after = -1
named_bytes = 0_i64
resources = {} of String => Int64

begin
  active_baseline = A::Owner.new(fixture, cache)
  baseline = active_baseline
  active_candidate = G::Owner.new(active_baseline, topology)
  candidate = active_candidate
  named_bytes = active_candidate.named_bytes
  resources = active_candidate.resources
  free_percent_after = free_memory_percent!

  command = ML::Metal::MPSCommandChain.new
  chain = command
  allocations_before = ML::MetalBuffer.hy_text_probe_allocations
  readbacks_before = ML::MetalBuffer.hy_text_probe_readbacks

  # One fixed correctness request. Baseline capture and both candidate graph
  # executables are encoded into this same MPS chain; no sample is timed.
  active_baseline.encode!(command, true)
  active_baseline.copy_canonical!(command)

  active_candidate.encode_output_only!(command)
  active_candidate.compare!(command, "work.down", "expected.down", 0,
    C::DOWN_ELEMENTS)
  active_candidate.compare!(command, "work.down", "canonical.down", 1,
    C::DOWN_ELEMENTS)
  if topology.uses_split_down?
    active_candidate.compare!(command, "graph.product", "expected.product", 2,
      C::PRODUCT_ELEMENTS)
    active_candidate.compare!(command, "graph.product", "work.up", 3,
      C::PRODUCT_ELEMENTS)
  end
  if topology.stage_outputs?
    active_candidate.compare!(command, "graph.gate", "expected.gate", 4,
      C::RAW_FFN_ELEMENTS)
    active_candidate.compare!(command, "graph.gate", "capture.gate", 5,
      C::RAW_FFN_ELEMENTS)
    active_candidate.compare!(command, "graph.up", "expected.up", 6,
      C::RAW_FFN_ELEMENTS)
    active_candidate.compare!(command, "graph.up", "capture.up", 7,
      C::RAW_FFN_ELEMENTS)
    active_candidate.compare!(command, "graph.silu", "work.gate", 8,
      C::RAW_FFN_ELEMENTS)
  end

  active_candidate.encode_captured!(command)
  active_candidate.compare!(command, "capture.gate", "expected.gate", 9,
    C::RAW_FFN_ELEMENTS)
  active_candidate.compare!(command, "capture.up", "expected.up", 10,
    C::RAW_FFN_ELEMENTS)
  active_candidate.compare!(command, "graph.gate", "expected.gate", 11,
    C::RAW_FFN_ELEMENTS)
  active_candidate.compare!(command, "graph.gate", "capture.gate", 12,
    C::RAW_FFN_ELEMENTS)
  active_candidate.compare!(command, "graph.up", "expected.up", 13,
    C::RAW_FFN_ELEMENTS)
  active_candidate.compare!(command, "graph.up", "capture.up", 14,
    C::RAW_FFN_ELEMENTS)

  # Saved J5760 captures omit silu_gate. Derive the oracle on GPU from the
  # authenticated saved gate_proj with the production SiLU kernel.
  active_candidate.derive_reference_silu!(command)
  active_candidate.compare!(command, "work.gate", "expected.gate", 15,
    C::RAW_FFN_ELEMENTS)
  active_candidate.compare!(command, "graph.silu", "expected.gate", 16,
    C::RAW_FFN_ELEMENTS)
  active_candidate.compare!(command, "graph.silu", "work.gate", 17,
    C::RAW_FFN_ELEMENTS)

  active_candidate.compare!(command, "work.up", "expected.product", 18,
    C::PRODUCT_ELEMENTS)
  active_candidate.compare!(command, "graph.product", "expected.product", 19,
    C::PRODUCT_ELEMENTS)
  active_candidate.compare!(command, "graph.product", "work.up", 20,
    C::PRODUCT_ELEMENTS)
  active_candidate.compare!(command, "canonical.down", "expected.down", 21,
    C::DOWN_ELEMENTS)
  active_candidate.compare!(command, "work.down", "expected.down", 22,
    C::DOWN_ELEMENTS)
  active_candidate.compare!(command, "work.down", "canonical.down", 23,
    C::DOWN_ELEMENTS)
  active_candidate.positive_control!(command)

  encode_allocations = ML::MetalBuffer.hy_text_probe_allocations - allocations_before
  preterminal_readbacks = ML::MetalBuffer.hy_text_probe_readbacks - readbacks_before
  wait_attempted = true
  command.commit_and_wait
  active_baseline.postwait!(command)
  active_candidate.postwait!(command)
  metrics = active_candidate.metrics(command)
  postterminal_readbacks = ML::MetalBuffer.hy_text_probe_readbacks - readbacks_before
  wait_count = command.wait_count
  baseline_counts = active_baseline.counts
  candidate_counts = active_candidate.counts
  stage_failures = G.validate_metrics!(metrics, topology)

  stage_failures << "baseline operation counts differ: #{baseline_counts}" unless baseline_counts == [1_u64, 1_u64, 1_u64, 1_u64, 1_u64, 1_u64, 2_u64]
  stage_failures.concat(G.validate_candidate_counts!(topology, candidate_counts))
  stage_failures << "encoding allocated named Metal buffers: #{encode_allocations}" unless encode_allocations == 0
  stage_failures << "readbacks occurred before the terminal wait: #{preterminal_readbacks}" unless preterminal_readbacks == 0
  stage_failures << "expected exactly one packed post-wait readback: #{postterminal_readbacks}" unless postterminal_readbacks == 1
  stage_failures << "expected exactly one terminal wait: #{wait_count}" unless wait_count == 1

  numerical_pass = G.validate_metrics!(metrics, topology).empty?
  execution_pass = encode_allocations == 0 && preterminal_readbacks == 0 &&
                   postterminal_readbacks == 1 && wait_count == 1 &&
                   baseline_counts == [1_u64, 1_u64, 1_u64, 1_u64, 1_u64, 1_u64, 2_u64] &&
                   G.validate_candidate_counts!(topology, candidate_counts).empty? && named_bytes <= A::MAX_NAMED_BYTES
  expected_stage_counts = G.expected_stage_counts(topology)
  metric_rows = metrics.map_with_index do |metric, index|
    expected = expected_stage_counts[index].to_u32
    passed = if index == G::STAGE_NAMES.size - 1
      metric == {failures: 1_u32, different_words: 0_u32,
                 nonfinite: 1_u32, elements: expected}
    elsif expected == 0
      metric == {failures: 0_u32, different_words: 0_u32,
                 nonfinite: 0_u32, elements: 0_u32}
    else
      metric[:elements] == expected && metric[:failures] == 0 && metric[:nonfinite] == 0
    end
    {name: G::STAGE_NAMES[index], expected_elements: expected,
     failures: metric[:failures], different_words: metric[:different_words],
     nonfinite: metric[:nonfinite], elements: metric[:elements],
     passed: passed}
  end
  candidate_sequence = case topology
  when G::Topology::FullGraph
    "each output-only and captured graph computes through down and writes work.down"
  when G::Topology::SplitDown
    "output-only and captured front graphs each write BF16 graph.product; a separate production F32 down graph consumes that buffer and writes work.down on the same MPS command buffer"
  when G::Topology::StageOutputs
    "output-only and captured front graphs each materialize BF16 gate/up/silu/product into the existing stage buffers; a separate production F32 down graph consumes graph.product and writes work.down on the same MPS command buffer"
  end
  result = {
    schema: G::SCHEMA,
    scope: "diagnostic-only output-only and captured whole-FFN MPSGraph correctness falsifier; no generation",
    probe_completed: stage_failures.empty?, production_promoted: false,
    full_generation_executed: false, correctness_only: true,
    fixture: {path: File.realpath(reference), metadata_sha256: fixture.sha256,
              payload_sha256: fixture.payload_sha256,
              weight_cache_metadata_sha256: cache.tensor_file.sha256,
              weight_cache_payload_sha256: cache.tensor_file.document["payload_sha256"],
              input_window: "w0", block: 0,
              normalized_input_sha256: fixture.windows["w0"].captures["block0"]["post_attn_norm"].sha256,
              input_provenance: "saved_initial_noise_conditioning_upstream_stem",
              model_inference: false, upstream_revision: B::UPSTREAM_REVISION,
              upstream_model_sha256: B::MODEL_SHA256},
    comparison: {atol: B::ATOL, rtol: B::RTOL,
                 silu_reference: "production pixelumm_silu applied in place on GPU to authenticated saved gate_proj; source has no saved silu_gate capture"},
    protocol: {tokens: C::TOKENS, hidden: C::HIDDEN, ffn: C::FFN,
               candidate_topology: topology.label,
               candidate_graph_sequence: candidate_sequence,
               output_only_front_stage_outputs: topology.stage_outputs? ?
                 ["graph.gate", "graph.up", "graph.silu", "graph.product"] :
                 (topology.split_down? ? ["graph.product"] : ["work.down"]),
               stage_outputs_are_mandatory_targets: topology.stage_outputs?,
               candidate_graph_operation_count_names: G::GRAPH_OPERATION_NAMES,
               expected_candidate_graph_operation_counts: G.operation_count_report(G.expected_candidate_counts(topology)),
               candidate_timing_samples: 0, baseline_timing_samples: 0,
               output_only_and_captured_graphs_compiled_separately: true,
               output_only_down_compared_to_saved_and_native_baseline: true,
               output_only_product_compared_to_saved_and_native_baseline_before_capture: topology.uses_split_down?,
               output_only_front_stages_compared_before_capture: topology.stage_outputs?,
               captured_stages_compared_to_saved_and_native_baseline: true,
               explicit_bf16_rounds: ["gate", "up", "silu", "product", "down"],
               terminal_waits: 1, comparator_readback_passes: 1,
               encoded_named_buffer_allocations_allowed: 0,
               preterminal_readbacks_allowed: 0,
               local_free_memory_preflight: "memory_pressure -Q reports more than 10 percent free before and after allocation; this does not establish memory admission",
               external_memory_admission_required: "launch under examples/support/wan_memory_guard.py with the existing cache-aware 10 percent reserve and process/swap limits",
               local_free_percent_preflight_threshold: 10,
               no_generation: true, no_promotion: true},
    execution: {named_bytes: named_bytes, maximum_named_bytes: A::MAX_NAMED_BYTES,
                resources: resources, free_memory_percent_before_allocate: free_percent_before,
                free_memory_percent_after_allocate: free_percent_after,
                encoding_named_allocation_delta: encode_allocations,
                preterminal_readbacks: preterminal_readbacks,
                total_readbacks_after_wait: postterminal_readbacks,
                wait_count: wait_count, baseline_operation_counts: baseline_counts,
                candidate_graph_operation_counts: G.operation_count_report(candidate_counts)},
    stages: metric_rows, failures: stage_failures,
    numerical_pass: numerical_pass, execution_pass: execution_pass,
    pass: numerical_pass && execution_pass && stage_failures.empty?,
    source_sha256: source_hashes,
  }
  fd = LibC.open(report, LibC::O_WRONLY | LibC::O_CREAT | LibC::O_EXCL, 0o600)
  raise "whole-FFN graph report reservation failed" if fd < 0
  io = IO::FileDescriptor.new(fd)
  begin
    io << result.to_json << '\n'
    io.flush
  ensure
    io.close
  end
  report_written = true
  puts({report: report, pass: result[:pass], numerical_pass: numerical_pass,
        execution_pass: execution_pass, failures: stage_failures}.to_json)
  raise "whole-FFN graph parity/execution gate failed; raw report retained" unless result[:pass]
rescue error
  failure = error.message || error.class.name
  raise error
ensure
  if active_chain = chain
    if !active_chain.completed? && !wait_attempted
      wait_attempted = true
      active_chain.commit_and_wait
    end
    if active_chain.completed?
      if active_candidate = candidate
        active_candidate.postwait!(active_chain) unless active_candidate.lifecycle.terminal_wait_succeeded?
        active_candidate.release_after_terminal!(active_chain)
      end
      if active_baseline = baseline
        active_baseline.postwait!(active_chain) unless active_baseline.lifecycle.terminal_wait_succeeded?
        active_baseline.release_after_terminal!(active_chain)
      end
      active_chain.release
    end
  else
    candidate.try(&.release_unsubmitted!)
    baseline.try(&.release_unsubmitted!)
  end
  unless report_written
    STDERR.puts("PIXELUMM_FFN_GRAPH_PROBE_INCOMPLETE #{failure || "probe did not reach its report"}")
  end
end
