require "option_parser"
require "./support/hy_nomas_probe_instrumentation"
require "./support/hy_nomas_q4_dit_probe"

# Paired, same-process full54 forward. Diagnostics bracket the timed samples;
# all model work, including the compact input copy, joins one command chain.
module HyNomasTextCompactionBenchmark
  extend self

  def run_forward(session : HyNomasQ4DiTProbe::Session, mode : String,
                  *, request : Int32? = nil) : JSON::Any
    chain = ML::Metal::MPSCommandChain.new
    begin
      allocations = ML::MetalBuffer.hy_text_probe_allocations
      reads = ML::MetalBuffer.hy_text_probe_readbacks
      poison = request == 1
      started = Time.instant
      session.encode(chain, poison: poison,
        poison_text_suffix: poison && session.compact_text)
      encode_seconds = (Time.instant - started).total_seconds
      encode_allocations = ML::MetalBuffer.hy_text_probe_allocations - allocations
      early_reads = ML::MetalBuffer.hy_text_probe_readbacks - reads
      raise "Compaction forward allocated, read activations or waited during encode" unless encode_allocations == 0 && early_reads == 0 && chain.wait_count == 0
      early_release_rejected = false
      if request == 0
        begin
          session.release
        rescue ex : ArgumentError
          raise ex unless (ex.message || "").includes?("command chain")
          early_release_rejected = true
        end
        raise "Compaction session accepted in-flight release" unless early_release_rejected
      end
      wait_started = Time.instant
      chain.commit_and_wait
      wait_seconds = (Time.instant - wait_started).total_seconds
      total_seconds = (Time.instant - started).total_seconds
      record = JSON.parse({mode: mode, encode_seconds: encode_seconds,
                           terminal_wait_seconds: wait_seconds, total_seconds: total_seconds,
                           gpu_elapsed_seconds: chain.gpu_elapsed_seconds,
                           command_segments: chain.segment_count,
                           named_allocations_during_encode: encode_allocations,
                           activation_readbacks_before_terminal_wait: early_reads,
                           terminal_wait_count: chain.wait_count,
                           diagnostic_readbacks: 0}.to_json)
      if diagnostic_request = request
        comparisons = session.compare(diagnostic_request)
        diagnostic_reads = ML::MetalBuffer.hy_text_probe_readbacks - reads
        raise "Compaction diagnostics must read the same 14 outputs after completion" unless diagnostic_reads == 14
        passed = chain.wait_count == 1 && comparisons.all? do |r|
          r[:finite] && r[:failures] == 0 && r[:first_request_sha256_match] != false
        end
        record.as_h["request"] = JSON::Any.new(diagnostic_request.to_i64)
        record.as_h["passed"] = JSON::Any.new(passed)
        record.as_h["scratch_poisoned"] = JSON::Any.new(poison)
        record.as_h["excluded_suffix_poisoned"] = JSON::Any.new(poison && session.compact_text)
        record.as_h["poison_buffer_count"] = JSON::Any.new((poison ? session.poison_count : 0).to_i64)
        record.as_h["early_release_rejected"] = JSON::Any.new(early_release_rejected)
        record.as_h["diagnostic_readbacks"] = JSON::Any.new(diagnostic_reads.to_i64)
        record.as_h["comparisons"] = JSON.parse(comparisons.to_json)
      end
      raise "Compaction forward gained an extra wait/read" unless chain.wait_count == 1 && (request || ML::MetalBuffer.hy_text_probe_readbacks == reads)
      record
    ensure
      chain.commit_and_wait unless chain.completed?
      chain.release
    end
  end
end

reference, reference_sha, index, index_sha, report = "", "", "", "", ""
budget = 0_i64
combined_budget = 0_i64
compact_text_tokens = 64
OptionParser.parse do |parser|
  parser.banner = "Paired padded/compact full54 DiT forwards; external 10% memory guard required."
  parser.on("--reference PATH", "Frozen padded upstream Q4 fixture") { |v| reference = v }
  parser.on("--reference-sha256 SHA", "Trusted padded fixture SHA256") { |v| reference_sha = v }
  parser.on("--cache-index PATH", "Authenticated ordered cache index") { |v| index = v }
  parser.on("--cache-index-sha256 SHA", "Trusted cache index SHA256") { |v| index_sha = v }
  parser.on("--max-named-bytes N", "Per-session named budget") { |v| budget = v.to_i64 }
  parser.on("--max-combined-named-bytes N", "Combined budget for both resident sessions") { |v| combined_budget = v.to_i64 }
  parser.on("--compact-text-tokens N", "Aligned DiT text storage rows; attention retains 23 valid rows") { |v| compact_text_tokens = v.to_i32 }
  parser.on("--report PATH", "New paired benchmark JSON") { |v| report = v }
end
unless ARGV.empty? && {reference, reference_sha, index, index_sha, report}.all? { |v| !v.empty? } &&
       0 < budget <= 14_i64 * 1024**3 && 0 < combined_budget <= 24_i64 * 1024**3
  raise ArgumentError.new("Expected authenticated inputs, new report, per-session budget<=14GiB and combined<=24GiB")
end
raise ArgumentError.new("Refusing to replace compaction report") if File.info?(report, false)
admitted_index = ML::Video::HyNomasQ4StackIndex.load!(index, index_sha,
  HyNomasQ4DiTIO::SOURCE_SHA, HyNomasQ4DiTIO::CONFIG)
fixture = HyNomasQ4DiTProbe::Fixture.new(reference, reference_sha, "q4", admitted_index)
# Derive both exact admissions before either model is allocated. Globals,
# weights and raw inputs are the same authenticated bytes in both sessions.
expected_padded = HyNomasQ4DiTProbe.q4_named_bytes(admitted_index)
expected_compact = HyNomasQ4DiTProbe.q4_named_bytes(admitted_index,
  compact_text: true, compact_text_tokens: compact_text_tokens)
unless expected_padded <= budget && expected_compact <= budget &&
       expected_padded <= combined_budget - expected_compact
  raise ArgumentError.new("Both full54 sessions exceed the admitted named budgets")
end
sessions = {} of String => HyNomasQ4DiTProbe::Session
validations = [] of JSON::Any
warmups = [] of JSON::Any
samples = [] of JSON::Any
epochs = [] of JSON::Any
begin
  sessions["padded"] = HyNomasQ4DiTProbe::Session.new(fixture, index, index_sha,
    admitted_index, "q4", budget)
  sessions["compact"] = HyNomasQ4DiTProbe::Session.new(fixture, index, index_sha,
    admitted_index, "q4", budget, compact_text: true, compact_text_tokens: compact_text_tokens)
  unless sessions["padded"].named_bytes == expected_padded && sessions["compact"].named_bytes == expected_compact
    raise "Paired full54 actual allocation differs from pre-admitted totals"
  end
  {0, 1}.each do |request|
    # Clean validation precedes timing; poisoned/reuse validation follows it.
    {"padded", "compact"}.each do |mode|
      entry = HyNomasTextCompactionBenchmark.run_forward(sessions[mode], mode, request: request)
      validations << entry
      STDERR.puts "validation #{mode}/#{request}: #{entry["passed"]}"
    end
    break unless validations.all? { |v| v["passed"].as_bool }
    next if request == 1
    2.times do |epoch|
      initial = epoch == 0 ? ["padded", "compact"] : ["compact", "padded"]
      epochs << JSON.parse({epoch: epoch, warmup_order: initial, pairs: 3}.to_json)
      initial.each_with_index do |mode, position|
        entry = HyNomasTextCompactionBenchmark.run_forward(sessions[mode], mode)
        entry.as_h["epoch"] = JSON::Any.new(epoch.to_i64)
        entry.as_h["position"] = JSON::Any.new(position.to_i64)
        warmups << entry
      end
      3.times do |pair|
        order = pair.even? ? initial : initial.reverse
        order.each_with_index do |mode, position|
          entry = HyNomasTextCompactionBenchmark.run_forward(sessions[mode], mode)
          entry.as_h["epoch"] = JSON::Any.new(epoch.to_i64)
          entry.as_h["pair"] = JSON::Any.new(pair.to_i64)
          entry.as_h["position"] = JSON::Any.new(position.to_i64)
          samples << entry
          STDERR.puts "epoch #{epoch}/pair #{pair}/#{mode}: #{entry["total_seconds"]}s"
        end
      end
    end
  end
  passed = validations.size == 4 && validations.all? { |v| v["passed"].as_bool } && samples.size == 12
  modes = sessions.transform_values do |session|
    {named_bytes: session.named_bytes, dit_text_tokens: session.dit_text_tokens}
  end
  result = {schema: "cogni_ml.hy_nomas_text_compaction_benchmark", schema_version: 1,
            reference_sha256: fixture.sha256, cache_index_sha256: index_sha,
            precision_mode: "q4", geometry: {batches: 1, video_tokens: 5280,
                                             text_tokens: 1280, valid_text_tokens: 23,
                                             compact_text_tokens: compact_text_tokens, blocks: 54},
            tolerance: {atol: HyNomasQ4DiTProbe::ATOL, rtol: HyNomasQ4DiTProbe::RTOL},
            diffusion_generation: false, modes: modes,
            max_combined_named_bytes: combined_budget, epochs: epochs,
            validations: validations, warmups: warmups, samples: samples, passed: passed}
  HyNomasQ4DiTIO.write_new!(report, result.to_pretty_json + "\n")
  puts({report: report, passed: passed}.to_json)
  raise "Compaction upstream/poison gate failed; unchanged tolerances" unless passed
ensure
  sessions.each_value(&.release)
end
