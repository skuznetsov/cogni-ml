require "option_parser"
require "./support/hy_nomas_refiner_attention_probe"
require "../src/ml/video/metal_mps_command_chain"

include HyNomasRefinerAttentionProbe

inputs_path = ""
inputs_sha256 = ""
output_path = ""
max_named_bytes = 0_i64
bf16_boundaries_control = false
vector_tail_control = false
OptionParser.parse do |parser|
  parser.banner = "Capture native Hy-Nomas block-0 Q/K/softmax/PV attention stages."
  parser.on("--inputs PATH", "Authenticated native attention replay manifest") { |value| inputs_path = value }
  parser.on("--inputs-sha256 SHA", "Trusted replay manifest SHA-256") { |value| inputs_sha256 = value }
  parser.on("--output PATH", "New capture JSON and adjacent tensor sidecar") { |value| output_path = value }
  parser.on("--max-named-bytes N", "Positive named-buffer cap, at most 512 MiB") { |value| max_named_bytes = value.to_i64 }
  parser.on("--bf16-boundaries-control", "Opt-in BF16 scaled-logit/P/PV graph with a separate P witness") { bf16_boundaries_control = true }
  parser.on("--vector-tail-control", "Opt-in native one-pass BF16 vector tail; requires BF16 boundaries control") { vector_tail_control = true }
end
unless ARGV.empty? && !inputs_path.empty? && !inputs_sha256.empty? &&
       !output_path.empty? && output_path.ends_with?(".json")
  raise ArgumentError.new("inputs, trusted inputs digest, fresh .json output and named-buffer cap are required")
end

# Authenticate every payload byte and every tensor record before constructing
# Arena or touching Metal. Input/output path and memory checks are also pure.
replay = ReplayInputs.new(inputs_path, inputs_sha256)
output_manifest, output_payload = checked_output_paths!(output_path)
ensure_fresh_outputs!(output_manifest, output_payload)
outputs_are_distinct = {output_manifest, output_payload}.all? do |candidate|
  candidate != replay.packet.path && candidate != replay.packet.payload_path
end
unless outputs_are_distinct
  raise ArgumentError.new("attention probe inputs and outputs must be distinct files")
end
expected_named_bytes = validate_budget!(max_named_bytes, bf16_boundaries_control, vector_tail_control)
ops = operation_documents

# The existing reduction helper consumes HyNomasConfig for geometry and scale
# validation. This adapter sums to 263 rows; it assigns no video/text meaning.
config = ML::Video::HyNomasConfig.new(
  batches: BATCHES, video_tokens: 1, text_tokens: QUERY_ROWS - 1,
  dim: HEADS * HEAD_DIM, heads: HEADS, head_dim: HEAD_DIM,
  ffn_dim: 4 * HEADS * HEAD_DIM,
)
config.validate!

arena : ML::Video::Arena? = nil
controls : HyAttentionProbe::MPSReductionControls? = nil
bf16_plan : ML::Video::HyNomasAttentionMPSPlan? = nil
instrumented_plan : ML::Video::HyNomasAttentionMPSPlan? = nil
vector_plan : ML::Video::HyNomasVectorAttentionPlan? = nil
chain : ML::Metal::MPSCommandChain? = nil
wait_attempted = false
begin
  arena = ML::Video::Arena.new
  input_buffers = replay.upload!(arena)
  bind_inputs_and_allocate_outputs!(arena, input_buffers)
  controls = HyAttentionProbe::MPSReductionControls.new(config, ops, arena)
  raise "attention probe must prepare six native MPS graphs" unless controls.plan_count == 6
  if bf16_boundaries_control
    output = arena.alloc("bf16.output", Q_ELEMENTS.to_i32)
    instrumented_output = arena.alloc("bf16.instrumented_output", Q_ELEMENTS.to_i32)
    probabilities = arena.alloc("bf16.probabilities", SCORE_ELEMENTS.to_i32)
    bf16_plan = ML::Video::HyNomasAttentionMPSPlan.new(
      BATCHES, QUERY_ROWS, HEADS, HEAD_DIM, input_buffers["q"],
      input_buffers["k"], input_buffers["v"], output, true)
    instrumented_plan = ML::Video::HyNomasAttentionMPSPlan.new(
      BATCHES, QUERY_ROWS, HEADS, HEAD_DIM, input_buffers["q"],
      input_buffers["k"], input_buffers["v"], instrumented_output, true, probabilities)
  end
  if vector_tail_control
    output = arena.alloc("vector.output", TAIL_ELEMENTS.to_i32)
    vector_plan = ML::Video::HyNomasVectorAttentionPlan.new(
      BATCHES, QUERY_ROWS, QUERY_ROWS - TAIL_ROWS, TAIL_ROWS, KEY_ROWS,
      HEADS, HEAD_DIM, input_buffers["q"], input_buffers["k"], input_buffers["v"], output)
  end
  setup_named_bytes = arena.resident_bytes
  setup_named_bytes += bf16_plan.try(&.resident_bytes) || 0_i64
  setup_named_bytes += instrumented_plan.try(&.resident_bytes) || 0_i64
  setup_named_bytes += vector_plan.try(&.resident_bytes) || 0_i64
  unless setup_named_bytes == expected_named_bytes && setup_named_bytes <= max_named_bytes
    raise ArgumentError.new("attention probe setup buffers differ from admitted named-byte budget")
  end

  chain = ML::Metal::MPSCommandChain.new
  allocations_before = allocation_counter
  readbacks_before = readback_counter
  ops.each_with_index do |op, index|
    controls.encode(chain, op, index.to_i32)
  end
  bf16_plan.try(&.encode(chain))
  instrumented_plan.try(&.encode(chain))
  vector_plan.try(&.encode(chain))
  encode_allocations = allocation_counter - allocations_before
  encode_readbacks = readback_counter - readbacks_before
  preterminal_waits = chain.wait_count
  unless encode_allocations == 0 && encode_readbacks == 0 && preterminal_waits == 0
    raise "attention stage encode allocated, read back or waited before terminal"
  end

  wait_attempted = true
  chain.commit_and_wait
  unless chain.completed? && chain.wait_count == 1
    raise "attention probe requires exactly one successful terminal wait"
  end

  write_capture!(output_manifest, output_payload, replay.packet,
    capture_sources(arena, bf16_boundaries_control, vector_tail_control), setup_named_bytes, max_named_bytes,
    encode_allocations, encode_readbacks, preterminal_waits, chain.wait_count,
    bf16_boundaries_control, vector_tail_control)
  puts "attention_probe_complete capture=#{output_manifest} setup_named_bytes=#{setup_named_bytes} terminal_waits=#{chain.wait_count}"
ensure
  if active_chain = chain
    unless wait_attempted || active_chain.completed?
      wait_attempted = true
      begin
        active_chain.commit_and_wait
      rescue
      end
    end
    if active_chain.completed?
      vector_plan.try(&.release)
      instrumented_plan.try(&.release)
      bf16_plan.try(&.release)
      controls.try(&.release)
      active_chain.release
      arena.try(&.release)
    end
  else
    vector_plan.try(&.release)
    instrumented_plan.try(&.release)
    bf16_plan.try(&.release)
    controls.try(&.release)
    arena.try(&.release)
  end
end
