require "option_parser"
require "./support/pixelumm_ffn_projection_probe"

alias B = PixelUMMFFNProjectionProbe
reference = ""
reference_sha = ""
report = ""
projection = "gate"
rows = 2048
rounds = 4
reverse_first = false
max_named_bytes = 4_i64 * 1024 * 1024 * 1024
OptionParser.parse do |parser|
  parser.banner = "Bounded pretrained PixelUMM FFN projection diagnostics; no generation."
  parser.on("--reference PATH", "Authenticated saved J2048 upstream fixture") { |v| reference = v }
  parser.on("--reference-sha256 SHA", "Trusted fixture metadata SHA256") { |v| reference_sha = v }
  parser.on("--report PATH", "Fresh absolute external JSON report") { |v| report = v }
  parser.on("--projection NAME", "gate, up or down") { |v| projection = v }
  parser.on("--rows N", "2048 or 21120; larger M repeats saved rows on GPU") { |v| rows = v.to_i32 }
  parser.on("--rounds N", "2 or 4 measured alternating-order rounds after one warmup") { |v| rounds = v.to_i32 }
  parser.on("--reverse-first", "Reverse the first measured order") { reverse_first = true }
  parser.on("--max-named-bytes N", "Preallocation cap, at most 4 GiB") { |v| max_named_bytes = v.to_i64 }
end
raise ArgumentError.new("invalid probe arguments") unless ARGV.empty? && {2, 4}.includes?(rounds) &&
                                                          {2048, 21120}.includes?(rows) && max_named_bytes > 0 && max_named_bytes <= 4_i64 * 1024 * 1024 * 1024
B.projection(projection)
raise ArgumentError.new("report must be fresh, regular-parent absolute external JSON path") unless Path[report].absolute? &&
                                                                                                   File.realpath(File.dirname(report)).starts_with?("/Volumes/") && File.extname(report) == ".json" && !File.info?(report, false)
fixture = PixelUMMBlockProbe::Fixture.new(reference, reference_sha)
schedule = (rounds + 1).times.flat_map do |round|
  reverse = round > 0 && ((round - 1).odd? != reverse_first)
  (reverse ? B::ARMS.reverse : B::ARMS).map { |arm| {arm: arm, round: round, warmup: round == 0, reverse: reverse} }
end.to_a
samples = [] of NamedTuple(arm: String, round: Int32, warmup: Bool, reverse: Bool,
  first_root: Int32, end_root: Int32, encode_seconds: Float64)
owner = nil.as(B::Owner?)
chain = nil.as(ML::Metal::MPSCommandChain?)
begin
  setup_started = Time.instant
  owner = B::Owner.new(fixture, projection, rows, schedule.size, max_named_bytes)
  setup_seconds = (Time.instant - setup_started).total_seconds
  chain = ML::Metal::MPSCommandChain.new
  allocations = ML::MetalBuffer.hy_text_probe_allocations
  reads = ML::MetalBuffer.hy_text_probe_readbacks
  encode_started = Time.instant
  owner.setup(chain)
  schedule.each_with_index do |sample, i|
    owner.poison(chain, sample[:arm])
    B.next_root(chain)
    first_root = chain.observed_root_count - 1
    started = Time.instant
    owner.encode(chain, sample[:arm])
    encode_seconds = (Time.instant - started).total_seconds
    end_root = chain.observed_root_count
    B.next_root(chain)
    owner.check(chain, sample[:arm], i)
    samples << sample.merge({first_root: first_root, end_root: end_root, encode_seconds: encode_seconds})
  end
  owner.check(chain, "mps.bf16", schedule.size, inject: true)
  encode_seconds = (Time.instant - encode_started).total_seconds
  encode_allocations = ML::MetalBuffer.hy_text_probe_allocations - allocations
  early_reads = ML::MetalBuffer.hy_text_probe_readbacks - reads
  raise "projection probe allocated or read/waited during encode" unless encode_allocations == 0 && early_reads == 0 && chain.wait_count == 0
  started = Time.instant
  chain.commit_and_wait
  terminal_wait_seconds = (Time.instant - started).total_seconds
  roots = chain.completed_root_timings
  metrics = owner.metrics(chain)
  expected_elements = (rows * B.projection(projection)[:n]).to_u32
  raise "GPU comparator skipped or overcounted output elements" unless metrics.all? { |m| m[:elements] == expected_elements && m[:failures] <= expected_elements && m[:different_words] <= expected_elements && m[:nonfinite] <= expected_elements }
  raise "comparator missed injected nonfinite defect" unless metrics.last[:failures] == 1 && metrics.last[:nonfinite] == 1 && metrics.last[:elements] == (rows * B.projection(projection)[:n]).to_u32
  prefixes = owner.prefix_comparisons(chain)
  roundtrip = owner.roundtrip(chain)
  raise "F16 operand counter missed known underflow/overflow" unless roundtrip["positive"] == [2_u32, 1_u32, 1_u32]
  prefixes.each do |prefix|
    # Independently constructed CPU comparison of the last request's first
    # 2048 rows must agree with the GPU metric for small-J qualification.
    index = samples.rindex { |sample| sample[:arm] == prefix[:arm] }.not_nil!
    raise "CPU/GPU numerical comparator disagreement" if rows == 2048 && metrics[index] != prefix[:metric]
  end
  timed = samples.map_with_index do |sample, index|
    selected = roots[sample[:first_root]...sample[:end_root]]
    raise "missing or failed timed GPU root" unless !selected.empty? && selected.all? { |r| r.completed && !r.has_error && r.gpu_start_seconds.finite? && r.gpu_end_seconds.finite? && r.gpu_start_seconds > 0 && r.gpu_end_seconds >= r.gpu_start_seconds }
    ranges = selected.sort_by(&.gpu_start_seconds)
    union_seconds = 0.0
    cursor = ranges.first.gpu_start_seconds
    ranges.each do |root|
      left = Math.max(cursor, root.gpu_start_seconds)
      union_seconds += Math.max(0.0, root.gpu_end_seconds - left)
      cursor = Math.max(cursor, root.gpu_end_seconds)
    end
    # Adjacent Metal command buffers can have overlapping GPU lifetimes even
    # on this serial queue. Preserve the numerical result, but disqualify an
    # exclusive projection timing claim when diagnostic roots overlap it.
    overlap = roots.reject { |r| sample[:first_root] <= r.sequence < sample[:end_root] }.sum do |other|
      selected.sum { |r| Math.max(0.0, Math.min(r.gpu_end_seconds, other.gpu_end_seconds) - Math.max(r.gpu_start_seconds, other.gpu_start_seconds)) }
    end
    raise "empty projection GPU interval" unless union_seconds > 0
    sample.merge({gpu_seconds: union_seconds, metric: metrics[index], outside_root_overlap_seconds: overlap,
                  isolated_timing: overlap == 0.0})
  end
  summaries = B::ARMS.map do |arm|
    measured = timed.select { |sample| sample[:arm] == arm && !sample[:warmup] }
    all = timed.select { |sample| sample[:arm] == arm }
    {arm: arm, median_root_union_seconds: B.median(measured.map { |s| s[:gpu_seconds] }),
     min_root_union_seconds: measured.min_of { |s| s[:gpu_seconds] }, max_root_union_seconds: measured.max_of { |s| s[:gpu_seconds] },
     gate_passed: all.all? { |s| s[:metric][:failures] == 0 && s[:metric][:nonfinite] == 0 },
     isolated_timings: measured.all? { |s| s[:isolated_timing] },
     maximum_failures: all.max_of { |s| s[:metric][:failures] }, maximum_different_words: all.max_of { |s| s[:metric][:different_words] }}
  end
  control_ok = summaries.find { |s| s[:arm] == "mps.bf16" }.not_nil![:gate_passed]
  result = {schema: "cogni_ml.pixelumm_ffn_projection/v1", scope: "one pretrained projection; no decoder or generation",
            geometry: {projection: projection, m: rows, k: B.projection(projection)[:k], n: B.projection(projection)[:n],
                       input_origin: "saved seeded upstream J2048 capture; cyclic rows repeated on GPU, including at J21120"},
            fixture: {path: fixture.path, sha256: fixture.sha256, upstream_revision: PixelUMMBlockProbe::UPSTREAM_REVISION,
                      input_sha256: owner.input_record.sha256, weight_sha256: owner.weight_record.sha256, expected_sha256: owner.expected_record.sha256},
            tolerance: {atol: PixelUMMBlockProbe::ATOL, rtol: PixelUMMBlockProbe::RTOL},
            kernel_sha256: Digest::SHA256.hexdigest(B::SOURCE), graph_matmul_data_types: owner.product_types,
            f16_operand_roundtrip: {counter_order: ["different_f32_values", "nonzero_to_zero", "finite_to_infinity"], counters: roundtrip},
            timing: {unit: "union of completed GPU root lifetimes per projection; exclusive timing only when isolated_timings is true", rounds: rounds, reverse_first: reverse_first,
                     per_request_casts_included: true, excluded: "fixture upload, setup/compilation, GPU row repetition, poison, comparisons and terminal readback",
                     summaries: summaries, samples: timed},
            execution: {named_bytes: owner.named_bytes, resources: owner.resources, setup_seconds: setup_seconds,
                        encode_seconds: encode_seconds, terminal_wait_seconds: terminal_wait_seconds, terminal_waits: chain.wait_count,
                        encode_named_allocations: encode_allocations, preterminal_readbacks: early_reads,
                        postterminal_readbacks: ML::MetalBuffer.hy_text_probe_readbacks - reads,
                        command_roots: roots.size, diagnostic_async_boundaries: 2 * samples.size,
                        root_timings: roots.map { |r| {sequence: r.sequence, start: r.gpu_start_seconds, end: r.gpu_end_seconds} }},
            gpu_positive_control: metrics.last, postterminal_cpu_prefix_comparisons: prefixes,
            summary: {diagnostic_completed: true, mps_control_gate_passed: control_ok, production_promoted: false}}
  fd = LibC.open(report, LibC::O_WRONLY | LibC::O_CREAT | LibC::O_EXCL, 0o600)
  raise IO::Error.from_errno("create projection report") if fd < 0
  file = IO::FileDescriptor.new(fd)
  begin
    file.puts(result.to_pretty_json)
  ensure
    file.close
  end
  puts result[:summary].to_json
  raise "current MPS control failed saved upstream gate" unless control_ok
ensure
  chain.try { |c| c.commit_and_wait unless c.completed? }
  owner.try(&.release)
  chain.try(&.release)
end
