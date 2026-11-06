require "json"
require "option_parser"
require "./support/hy_nomas_soft_route_render"

include HyNomasSoftRouteRender

parent_path = DEFAULT_PARENT
parent_sha = PARENT_SHA
anchor_path = DEFAULT_ANCHOR
anchor_sha = ANCHOR_SHA
bias_path = DEFAULT_BIAS
bias_sha = DEFAULT_BIAS_PAYLOAD_SHA
noise_path = NOISE_SELECTION_PATH
noise_sha = NOISE_SELECTION_SHA
report = ""
budget = 21_i64 * 1024 * 1024 * 1024
plan_only = false
admit_only = false
execute = false
OptionParser.parse do |parser|
  parser.banner = "Authenticated four-step Hy-Nomas soft-route render; keeps accepted M11 hard support."
  parser.on("--parent PATH", "Accepted frozen direct-route parent") { |value| parent_path = value }
  parser.on("--parent-sha256 SHA", "Trusted parent reference digest") { |value| parent_sha = value }
  parser.on("--anchor PATH", "Authenticated T1 Qwen conditioning crop") { |value| anchor_path = value }
  parser.on("--anchor-sha256 SHA", "Trusted conditioning manifest digest") { |value| anchor_sha = value }
  parser.on("--bias-manifest PATH", "Pinned authored soft-route priors") { |value| bias_path = value }
  parser.on("--bias-sha256 SHA", "Trusted bias manifest digest") { |value| bias_sha = value }
  parser.on("--noise-selection PATH", "Pinned selection of the accepted parent initial latent") { |value| noise_path = value }
  parser.on("--noise-selection-sha256 SHA", "Trusted noise-selection digest") { |value| noise_sha = value }
  parser.on("--report PATH", "Fresh native manifest path; payload is a sibling .f32le file") { |value| report = value }
  parser.on("--max-named-bytes N", "Named Metal allocation ceiling") { |value| budget = value.to_i64 }
  parser.on("--plan-only", "Authenticate inputs and print the CPU-only allocation plan") { plan_only = true }
  parser.on("--admit-only", "Authenticate all frozen CPU inputs without creating Metal resources") { admit_only = true }
  parser.on("--execute", "Run the four-step native render") { execute = true }
end
raise ArgumentError.new("unexpected positional arguments") unless ARGV.empty?
raise ArgumentError.new("choose exactly one of --plan-only, --admit-only, or --execute") unless [plan_only, admit_only, execute].count(true) == 1
raise ArgumentError.new("named-buffer ceiling must be positive") unless budget > 0
raise ArgumentError.new("--report is required only with --execute") unless execute == !report.empty?

admission = Admission.new(parent_path, parent_sha, anchor_path, anchor_sha,
  bias_path, bias_sha, noise_path, noise_sha)
if plan_only || admit_only
  cursor_contract = validate_stack_cursor_lifecycle!
  plan = Runner.memory_plan(admission.parent, admission.anchor,
    {"zero"    => admission.bias.values["zero"],
     "A1_high" => admission.bias.values["A1_high"],
     "A2_high" => admission.bias.values["A2_high"]})
  stage_bytes = plan["block_weight_stage"]
  reserve_bytes = Runner.bf16_stack_reserve_bytes(plan, stage_bytes,
    plan["resident_block_weights"])
  stack_admission = {
    "bf16_resident_bytes"      => plan["resident_block_weights"],
    "stage_bytes"              => stage_bytes,
    "reserve_bytes"            => reserve_bytes,
    "admitted_bytes"           => plan["resident_block_weights"] + stage_bytes + reserve_bytes,
    "q4_payload_bytes"         => 0_i64,
    "q4_substitution_rejected" => false,
  }
  if admit_only
    cache_admission = ML::Video::HyNomasQ4StackIndex::Admission.new(
      admission.parent.cache_index_path, admission.parent.cache_index_sha256,
      UPSTREAM_SHA, CACHE_CONFIG)
    checked = Runner.validate_bf16_stack_budget!(plan, stage_bytes,
      cache_admission.payload_bytes)
    stack_admission["q4_payload_bytes"] = checked[:q4_payload_bytes]
    stack_admission["q4_substitution_rejected"] = checked[:q4_substitution_rejected]
  end
  puts({"schema" => SCHEMA, "status" => plan_only ? "plan_validated" : "inputs_admitted_cpu_only",
        "model_launched" => false, "metal_resources_created" => false,
        "stack_cursor_contract" => cursor_contract,
        "bindings" => HyNomasSoftRouteRender.bindings(admission),
        "resource_accounting" => plan, "bf16_stack_admission" => stack_admission,
        "max_named_bytes" => budget,
        "budget_passed" => plan["peak_named_bytes"] <= budget}.to_json)
  raise ArgumentError.new("four-step named-buffer plan exceeds the supplied ceiling") if plan["peak_named_bytes"] > budget
  admission.close
  exit
end

unless Path[report].absolute? && File.realpath(File.dirname(report)) == File.dirname(report)
  raise ArgumentError.new("native report parent must be an existing canonical absolute directory")
end
report = File.join(File.dirname(report), Path[report].basename)
payload = File.join(File.dirname(report), "#{Path[report].stem}.f32le")
[report, payload].each do |path|
  raise ArgumentError.new("refusing to replace native output: #{path}") if File.exists?(path) || File.symlink?(path)
end

runner : HyNomasSoftRouteRender::Runner? = nil
begin
  runner = Runner.new(admission, budget)
  result = runner.run
  outputs = result[:outputs]
  checks = {} of String => HyNomasDiTRouteFactorialProbe::Comparison
  {
    "zero.step1.velocity"   => admission.parent.inputs["t1.step1.velocity"],
    "zero.step1.patch"      => admission.parent.inputs["t1.step1.patch"],
    "zero.step1.text_valid" => admission.parent.inputs["t1.step1.text_valid"],
    "zero.step1.vector"     => admission.parent.inputs["t1.step1.vector"],
  }.each do |name, expected|
    checks[name] = HyNomasDiTRouteFactorialProbe.compare_values(expected, outputs[name])
  end
  accepted_final = HyNomasSoftRouteRender.read_f32(ACCEPTED_FINAL_PATH, ACCEPTED_FINAL_SHA, LATENT)
  checks["zero.step4.latent"] = HyNomasDiTRouteFactorialProbe.compare_values(accepted_final, outputs["zero.step4.latent"])
  outputs.each do |name, values|
    unless values.all? { |value| value.finite? && (value.unsafe_as(UInt32) & 0xffff_u32).zero? }
      raise "native render output is not finite lossless BF16 expansion: #{name}"
    end
  end
  passed = checks.values.all? { |value| value.failures == 0 && value.exact_bit_differences == 0 }

  records = {} of String => NamedTuple(shape: Array(Int32), dtype: String, source_dtype: String,
    offset_bytes: Int64, nbytes: Int64, sha256: String)
  cursor = 0_i64
  HyNomasDiTRouteFactorialProbe.with_fresh_output(payload) do |io|
    TENSOR_ORDER.each do |name|
      values = outputs[name]
      raw = Slice(UInt8).new(values.to_unsafe.as(UInt8*), values.size * sizeof(Float32))
      io.write(raw)
      records[name] = {shape: TENSOR_SHAPES[name], dtype: "F32", source_dtype: "BF16",
                       offset_bytes: cursor, nbytes: raw.size.to_i64,
                       sha256: Digest::SHA256.hexdigest(raw)}
      cursor += raw.size.to_i64
    end
  end
  payload_sha = HyNomasDiTRouteFactorialProbe.sha256_file(payload)
  observed_calls = result[:case_calls]
  observed_t0_calls = observed_calls["T0"]
  observed_t1_calls = CASES.sum { |name| observed_calls[name] }
  report_json = JSON.build(indent: "  ") do |json|
    json.object do
      json.field "schema", SCHEMA
      json.field "schema_version", 1
      json.field "case_order", CASES
      json.field "inputs" do
        json.object do
          json.field "parent" do
            json.object do
              json.field "path", admission.parent.path
              json.field "sha256", admission.parent.sha256
              json.field "payload_path", admission.parent.output_payload_path
              json.field "payload_sha256", admission.parent.output_payload_sha256
              json.field "frozen_inputs_path", admission.parent.document["source_inputs"]["frozen_inputs"]["path"]
              json.field "frozen_inputs_sha256", admission.parent.document["source_inputs"]["frozen_inputs"]["sha256"]
              json.field "initial_latent_sha256", PARENT_LATENT_SHA
              json.field "roi_source_path", admission.parent.document["source_inputs"]["roi"]["manifest_path"]
              json.field "roi_source_sha256", ROI_SOURCE_SHA
              json.field "roi_payload_path", admission.parent.document["source_inputs"]["roi"]["payload_path"]
              json.field "roi_payload_sha256", HyNomasDiTRouteFactorialProbe::ROI_PAYLOAD_SHA256
            end
          end
          json.field "anchor" do
            qwen = admission.anchor.records["input.qwen_bf16"]
            unused = admission.anchor.records["input.latent"]
            json.object do
              json.field "path", admission.anchor.file.path
              json.field "sha256", admission.anchor.file.sha256
              json.field "payload_path", admission.anchor_payload_path
              json.field "payload_bytes", admission.anchor.file.payload_bytes
              json.field "payload_sha256", admission.anchor_payload_sha256
              json.field "qwen_record" do
                json.object do
                  json.field "name", "input.qwen_bf16"
                  json.field "dtype", qwen.dtype
                  json.field "shape", qwen.shape
                  json.field "offset_bytes", qwen.offset
                  json.field "nbytes", qwen.nbytes
                  json.field "sha256", qwen.sha256
                end
              end
              json.field "unused_latent" do
                json.object do
                  json.field "name", "input.latent"
                  json.field "dtype", unused.dtype
                  json.field "shape", unused.shape
                  json.field "offset_bytes", unused.offset
                  json.field "nbytes", unused.nbytes
                  json.field "sha256", unused.sha256
                  json.field "used", false
                end
              end
            end
          end
          json.field "bias_manifest" do
            json.object do
              json.field "path", admission.bias.path
              json.field "sha256", admission.bias.sha256
              json.field "case_order", HyNomasSoftRouteProbe::CASES
              json.field "reference_image", admission.bias.document["reference_image"]
              json.field "maps", admission.bias.document["maps"]
              json.field "bias_f32le_sha256", admission.bias.document["bias_f32le_sha256"]
            end
          end
          json.field "noise_selection" do
            json.object do
              json.field "path", admission.noise_path
              json.field "sha256", admission.noise_sha256
              json.field "selected_record", "t1.initial_latent"
              json.field "anchor_input_latent_used", false
            end
          end
          json.field "accepted_zero_step4" do
            json.object do
              json.field "path", ACCEPTED_FINAL_PATH
              json.field "sha256", ACCEPTED_FINAL_SHA
              json.field "shape", [1, 32, 30, 9, 16]
              json.field "dtype", "F32 lossless expansion of BF16"
            end
          end
        end
      end
      json.field "bindings", bindings(admission)
      json.field "geometry" do
        json.object do
          json.field "latent_shape", [1, 32, 30, 9, 16]
          json.field "video_rows", VIDEO
          json.field "history_text_valid_rows", T0_VALID
          json.field "action_text_valid_rows", T1_VALID
          json.field "text_storage_rows", TEXT
          json.field "hard_support", "accepted M11; additive bias applies only to its allowed video-query/action-text edges"
          json.field "steps", OUTER_STEPS
          json.field "shift", 5.0
        end
      end
      json.field "precision" do
        json.object do
          json.field "resident_weights", "BF16"
          json.field "input_storage", "F32 lossless promotion of authenticated BF16 saved states"
          json.field "activation_dtype", "BF16 boundaries with F32 expansion in named native buffers"
          json.field "attention", "source BF16 Q/K/V; compact SDPA chunks of 256"
          json.field "euler", "F32 update from BF16-rounded velocity, then BF16 round latent"
          json.field "mps_fallback", false
          json.field "cuda_parity_claimed", false
        end
      end
      json.field "resource_accounting", runner.not_nil!.plan
      json.field "max_named_bytes", budget
      json.field "opaque_mps_resources_included", false
      json.field "payload", {"basename" => Path[payload].basename, "bytes" => cursor,
                             "sha256" => payload_sha, "encoding" => "little-endian F32 lossless BF16 expansion"}
      json.field "tensors" do
        json.object do
          TENSOR_ORDER.each do |name|
            record = records[name]
            json.field name do
              json.object do
                json.field "dtype", record[:dtype]
                json.field "source_dtype", record[:source_dtype]
                json.field "shape", record[:shape]
                json.field "offset_bytes", record[:offset_bytes]
                json.field "nbytes", record[:nbytes]
                json.field "sha256", record[:sha256]
                json.field "checkpoint_boundary", name.split('.', 2).last
              end
            end
          end
        end
      end
      json.field "execution" do
        json.object do
          json.field "case_order", CASES
          json.field "schedule", "four outer Euler steps; each step stages each of 54 blocks once and shares it across T0 plus three T1 arms with one-layer K/V"
          json.field "steps", OUTER_STEPS
          json.field "weight_stage_pass_cap", WEIGHT_STAGE_PASSES_PER_CHAIN
          json.field "weight_stage_passes_observed", OUTER_STEPS
          json.field "logical_full_forward_calls", LOGICAL_FULL_FORWARD_CALLS
          json.field "full_block_calls", observed_t0_calls + observed_t1_calls
          json.field "t0_block_calls", observed_t0_calls
          json.field "t1_block_calls", observed_t1_calls
          json.field "observed_attention_calls" do
            json.object do
              observed_calls.each do |name, count|
                json.field name, count
              end
            end
          end
          json.field "refiner_calls", OUTER_STEPS
          json.field "patch_calls", 12
          json.field "final_head_calls", 12
          json.field "euler_updates", 12
          json.field "command_chains", 1
          json.field "terminal_waits", result[:terminal_waits]
          json.field "named_allocations_during_encode", result[:encode_allocations]
          json.field "activation_readbacks_before_terminal_wait", result[:encode_readbacks]
          json.field "postwait_tensor_readbacks", records.size
          json.field "forward_seconds", result[:seconds]
          json.field "scheduler_updates", 12
          json.field "decoder_calls", 0
          json.field "accepted_zero_replay" do
            json.object do
              checks.each do |name, value|
                json.field name do
                  json.object do
                    json.field "elements", value.elements
                    json.field "failures", value.failures
                    json.field "max_abs", value.max_abs
                    json.field "rms_abs", value.rms_abs
                    json.field "exact_bit_differences", value.exact_bit_differences
                  end
                end
              end
            end
          end
        end
      end
      json.field "passed_native_invariants", passed
      json.field "interpretation_scope", "Four-step frozen-state numerical render; does not establish visual acting or semantic quality."
    end
  end
  HyNomasDiTRouteFactorialProbe.with_fresh_output(report) { |io| io.puts(report_json) }
  puts({"schema" => SCHEMA, "status" => passed ? "native_render_completed" : "replay_gate_failed",
        "manifest" => report, "payload" => payload, "passed_native_invariants" => passed,
        "checks" => checks.size, "forward_seconds" => result[:seconds],
        "peak_named_bytes" => runner.not_nil!.plan["peak_named_bytes"]}.to_json)
  raise "native render exact zero replay failed; diagnostic artifacts were preserved" unless passed
ensure
  runner.try(&.release)
end
