require "option_parser"
require "./support/hy_nomas_prompt_refiner_probe"
require "./support/hy_nomas_probe_instrumentation"
require "./support/qwen25_prefix_probe"

fixture_path = ""
fixture_sha256 = ""
inputs_path = ""
inputs_sha256 = ""
report = ""
timestep_label = ""
max_named_bytes = 3_i64 * 1024 * 1024 * 1024
OptionParser.parse do |parser|
  parser.banner = "Refiner-only comparison for authenticated 557/585-token prompt inputs; no DiT block or generation."
  parser.on("--fixture PATH", "Fresh pinned-upstream valid-prefix refiner fixture") { |value| fixture_path = value }
  parser.on("--fixture-sha256 SHA256", "Trusted oracle manifest digest") { |value| fixture_sha256 = value }
  parser.on("--inputs PATH", "Authenticated conditioned generation_inputs v2 manifest") { |value| inputs_path = value }
  parser.on("--inputs-sha256 SHA256", "Trusted generation input manifest digest") { |value| inputs_sha256 = value }
  parser.on("--timestep VALUE", "One case per process: 1000 or 206.897") { |value| timestep_label = value }
  parser.on("--report PATH", "New post-terminal native comparison report below the task scratch root") { |value| report = value }
  parser.on("--max-named-bytes N", "Aggregate named-buffer cap; default 3 GiB") { |value| max_named_bytes = value.to_i64 }
end
raise ArgumentError.new("fixture, inputs, timestep, report, and positive budget are required") if fixture_path.empty? || fixture_sha256.empty? || inputs_path.empty? || inputs_sha256.empty? || report.empty? || timestep_label.empty? || max_named_bytes <= 0
raise ArgumentError.new("unexpected positional arguments") unless ARGV.empty?
report_parent = File.realpath(File.dirname(report))
output_root = File.realpath(HyNomasPromptRefinerProbe::OUTPUT_ROOT)
raise ArgumentError.new("report path must be below #{output_root}") unless Path[report].absolute? && report_parent.starts_with?("#{output_root}/")
raise ArgumentError.new("refusing to replace native report") if File.exists?(report) || File.symlink?(report)

fixture = HyNomasPromptRefinerProbe::Fixture.new(fixture_path, fixture_sha256, timestep_label)
inputs = HyNomasGeneration::Inputs.new(inputs_path, inputs_sha256, conditioned: true)
fixture.validate_inputs!(inputs)

weight_file : ML::Video::CheckpointFile? = nil
input_arena : ML::Video::Arena? = nil
weight_arena : ML::Video::Arena? = nil
prepared : ML::Video::PreparedWeights? = nil
refiner : ML::Video::HyNomasTextRefinerMPSPlan? = nil
chain : ML::Metal::MPSCommandChain? = nil
wait_attempted = false

begin
  weight_file = ML::Video::CheckpointFile.new(fixture.text_weights)
  bindings = ML::Video::HyNomasTextRefinerMPSPlan.checkpoint_bindings
  weight_plan = ML::Video::WeightPlan.new([weight_file.not_nil!], bindings,
    max_named_bytes, resident_dtype: ML::Video::WeightDType::BF16,
    require_all: true, allow_f32_to_bf16: true)
  text_bytes = ML::Video::HyNomasTextRefinerMPSPlan.named_bytes(inputs.mask, retain_padding: true)
  input_bytes = inputs.bytes(["input.qwen_bf16"])
  aggregate_bytes = weight_plan.peak_bytes + text_bytes.total + input_bytes
  raise ArgumentError.new("refiner-only session exceeds aggregate named-buffer cap") if aggregate_bytes > max_named_bytes

  input_arena = ML::Video::Arena.new
  weight_arena = ML::Video::Arena.new
  source = inputs.upload!("input.qwen_bf16", input_arena.not_nil!)
  prepared = ML::Video::PreparedWeights.new(weight_plan, weight_arena.not_nil!)
  refiner = ML::Video::HyNomasTextRefinerMPSPlan.new(source, inputs.mask,
    fixture.timestep, weight_arena.not_nil!, text_bytes.total,
    ML::Video::HyNomasAttentionPrecision::F32Control, retain_padding: true)
  raise "retained-padding refiner unexpectedly compacted its storage" unless refiner.not_nil!.processed_tokens == 1280
  raise "refiner named-byte admission differs from its plan" unless refiner.not_nil!.resident_bytes == text_bytes.total

  chain = ML::Metal::MPSCommandChain.new
  before_allocations = ML::MetalBuffer.hy_text_probe_allocations
  before_readbacks = ML::MetalBuffer.hy_text_probe_readbacks
  started = Time.instant
  prepared.not_nil!.encode(chain.not_nil!.encoder)
  refiner.not_nil!.encode(chain.not_nil!)
  encode_seconds = (Time.instant - started).total_seconds
  encode_allocations = ML::MetalBuffer.hy_text_probe_allocations - before_allocations
  encode_readbacks = ML::MetalBuffer.hy_text_probe_readbacks - before_readbacks
  raise "encoding allocated named GPU buffers" unless encode_allocations == 0
  raise "encoding read back an activation" unless encode_readbacks == 0
  raise "encoding performed an intermediate wait" unless chain.not_nil!.wait_count == 0

  wait_started = Time.instant
  wait_attempted = true
  chain.not_nil!.commit_and_wait
  wait_seconds = (Time.instant - wait_started).total_seconds
  raise "expected exactly one terminal wait" unless chain.not_nil!.completed? && chain.not_nil!.wait_count == 1

  projected_name = "projected_text.#{fixture.timestep_label}"
  expected_text = fixture.values(projected_name)
  expected_vec = fixture.values("vec.#{fixture.timestep_label}")
  text_elements = fixture.valid_tokens * 2048
  raise ArgumentError.new("oracle valid-prefix projected_text extent mismatch") unless expected_text.size == text_elements
  raise ArgumentError.new("oracle vec extent mismatch") unless expected_vec.size == 2048
  text_buffer = refiner.not_nil!.text_output
  vec_buffer = refiner.not_nil!.vec
  raise ArgumentError.new("native retained projected_text storage extent mismatch") unless text_buffer.size == 1280_i64 * 2048 * 4
  raise ArgumentError.new("native vec storage extent mismatch") unless vec_buffer.size == 2048_i64 * 4
  # read() begins at byte zero. Since valid text is an authenticated prefix,
  # this compares only the semantically consumed rows while retaining all
  # 1280 native storage slots. No padded-row equality is asserted.
  actual_text = text_buffer.read(text_elements)
  actual_vec = vec_buffer.read(2048)
  comparisons = [
    {name: "projected_text.valid_prefix", actual: actual_text, expected: expected_text},
    {name: "vec", actual: actual_vec, expected: expected_vec},
  ].map do |item|
    result = HyNomasPromptRefinerProbe.compare_values(item[:actual], item[:expected])
    {name: item[:name], elements: item[:expected].size, failures: result[:failures],
     max_abs: result[:max_abs], raw_bit_differences: result[:raw_bit_differences],
     first_mismatch: result[:first_mismatch], first_actual: result[:first_actual],
     first_expected: result[:first_expected]}
  end
  after_wait_readbacks = ML::MetalBuffer.hy_text_probe_readbacks - before_readbacks
  raise "unexpected post-wait diagnostic read count" unless after_wait_readbacks == 2
  failures = comparisons.select { |comparison| comparison[:failures] > 0 }
  native_report = {
    "schema" => "cogni_ml.hy_nomas_prompt_refiner_native_probe/v1",
    "schema_version" => 1,
    "fixture" => fixture.path,
    "fixture_sha256" => fixture.sha256,
    "inputs_manifest_sha256" => inputs.file.sha256,
    "inputs_payload_sha256" => inputs.file.document["payload_sha256"].as_s,
    "valid_text_tokens" => fixture.valid_tokens,
    "timestep_label" => fixture.timestep_label,
    "timestep_f32" => fixture.timestep,
    "precision_scope" => "BF16 resident weights; F32 refiner math and attention; pinned upstream uses BF16-rounded F32 weights",
    "padded_slot_policy" => "native retain_padding=true stores 1280 rows; only valid prefix rows and vec are compared; no padding equality claim",
    "aggregate_named_bytes" => aggregate_bytes,
    "named_allocations_during_encode" => encode_allocations,
    "activation_readbacks_before_terminal_wait" => encode_readbacks,
    "terminal_wait_count" => chain.not_nil!.wait_count,
    "diagnostic_readbacks_after_terminal_wait" => after_wait_readbacks,
    "encode_seconds" => encode_seconds,
    "terminal_wait_seconds" => wait_seconds,
    "gpu_elapsed_seconds" => chain.not_nil!.gpu_elapsed_seconds,
    "tolerance" => {"atol" => HyNomasPromptRefinerProbe::ATOL, "rtol" => HyNomasPromptRefinerProbe::RTOL},
    "comparisons" => comparisons,
    "first_mismatch" => failures.first?.try(&.[:name]),
    "passed" => failures.empty?,
  }
  report_json = JSON.build(indent: "  ") do |json|
    json.object { native_report.each { |key, value| json.field(key, value) } }
  end
  HyNomasGeneration.write_fresh_output(report) do |io|
    io.puts(report_json)
  end
  comparisons.each { |comparison| puts comparison.to_json }
  raise "first native refiner mismatch: #{failures.first[:name]}" unless failures.empty?
  puts "PASS: authenticated prompt #{fixture.valid_tokens}, timestep #{fixture.timestep_label}, valid projected_text plus vec; one terminal wait"
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
      refiner.try(&.release)
      prepared.try(&.release)
      input_arena.try(&.release)
      weight_arena.try(&.release)
      weight_file.try(&.close)
      active_chain.release
    end
  else
    refiner.try(&.release)
    prepared.try(&.release)
    input_arena.try(&.release)
    weight_arena.try(&.release)
    weight_file.try(&.close)
  end
end
