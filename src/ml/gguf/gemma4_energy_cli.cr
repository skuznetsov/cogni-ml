require "option_parser"
require "digest/sha256"
require "./gemma4_energy_profile"
require "./gemma4_energy_runner"
require "../energy/victoria_metrics"

module ML::GGUF
  module Gemma4EnergyCLI
    private def self.claim_option!(seen : Set(String), key : String)
      raise ArgumentError.new("duplicate or conflicting #{key} option") unless seen.add?(key)
    end

    private def self.validate_phase!(phase : JSON::Any, name : String, allowance : Int32)
      count = phase["predicted_tokens"].as_i64
      prompt = phase["prompt_tokens"].as_i64
      returned = phase["returned_token_ids"].as_i64
      overhead = phase["control_overhead"].as_i64
      unless phase["budget_tokens"].as_i64 == allowance && count >= 0 && count <= allowance &&
             prompt >= 0 && returned >= 0 && returned <= count && overhead == count - returned
        raise ArgumentError.new("inconsistent imported phase token accounting")
      end
      outcome = phase["outcome"].as_s
      stop = phase["stop_type"].as_s
      word = phase["stopping_word"].as_s
      forced = phase["forced_close"].as_bool
      truncated = phase["truncated"].as_bool
      valid = if name == "reasoning" && allowance == 0
                count == 0 && prompt == 0 && outcome == "skipped_zero_budget" &&
                  stop == "skipped" && word.empty? && !forced && !truncated
              elsif count <= 0 || prompt <= 0
                false
              elsif name == "reasoning"
                !truncated && (
                  (outcome == "forced_close" && stop == "limit" && count == allowance && forced && word.empty?) ||
                    (outcome == "natural_channel_close" && stop == "word" && word == Gemma4EnergyRunner::CHANNEL_END && !forced)
                )
              else
                !forced && (
                  (outcome == "answer_limit" && stop == "limit" && count == allowance && truncated && word.empty?) ||
                    (outcome == "eos" && stop == "eos" && !truncated && word.empty?) ||
                    (outcome == "turn_end" && stop == "word" && word == Gemma4EnergyRunner::TURN_END && !truncated)
                )
              end
      raise ArgumentError.new("inconsistent imported phase completion outcome") unless valid
    end

    # Imported windows must satisfy the same contract as freshly generated runs.
    # In particular an empty report cannot make --require-measurement succeed.
    def self.validate_report!(report : JSON::Any) : Nil
      unless report["schema"].as_s == "cogni.gemma4.energy.v1" && report["backend"].as_s == "llama.cpp" && report["full_depth"].as_bool
        raise ArgumentError.new("unsupported energy report contract")
      end
      raise ArgumentError.new("invalid prompt digest") unless /\A[0-9a-f]{64}\z/.matches?(report["prompt_sha256"].as_s)
      runs = report["runs"].as_a
      raise ArgumentError.new("report must contain measured runs") if runs.empty?
      identities = [] of String
      runs.each do |run|
        status = run["status"].as_s
        raise ArgumentError.new("invalid run status") unless {"completed", "failed"}.includes?(status)
        start_ms, end_ms = run["started_at_epoch_ms"].as_i64, run["ended_at_epoch_ms"].as_i64
        raise ArgumentError.new("invalid run window") unless start_ms > 0 && end_ms >= start_ms && (status == "failed" || end_ms > start_ms)
        profile = run["profile"]
        resolved = Gemma4EnergyProfile.for_name(profile["name"].as_s, profile["reasoning_tokens"].as_i, profile["answer_tokens"].as_i)
        if identity = run["model_identity"]?.try(&.as_s?)
          raise ArgumentError.new("run model identity must identify Gemma 4") unless identity.downcase.gsub(/[^a-z0-9]/, "").includes?("gemma4")
          identities << identity
        end
        if run["comparison_eligible"]?.try(&.as_bool?)
          raise ArgumentError.new("automatic or imported answer-quality approval is unsupported")
        end
        run.as_h["comparison_eligible"] = JSON::Any.new(false)
        run.as_h["quality_review_required"] = JSON::Any.new(true)
        if status == "completed"
          raise ArgumentError.new("completed run lacks model identity or answer") unless run["model_identity"].as_s.size > 0 && !run["answer"].as_s.strip.empty?
          raise ArgumentError.new("completed run lacks usage") unless run["usage_complete"].as_bool
          {"reasoning" => resolved.reasoning_tokens, "answer" => resolved.answer_tokens}.each do |phase_name, allowance|
            validate_phase!(run["phases"][phase_name], phase_name, allowance)
          end
        else
          raise ArgumentError.new("failed run cannot claim complete usage") unless run["usage_complete"].as_bool == false
        end
      end
      raise ArgumentError.new("backend model identity changed during comparison") if identities.uniq.size > 1
      if identity = identities.first?
        raise ArgumentError.new("top-level model identity does not match runs") unless report["model_identity"].as_s == identity
      end
    rescue ex : ArgumentError
      raise ex
    rescue
      raise ArgumentError.new("incomplete or malformed energy report")
    end

    def self.run(args : Array(String)) : Int32
      server = ENV["GEMMA4_SERVER_URL"]? || "http://127.0.0.1:8080"
      vm_url = ENV["COGNI_VM_URL"]? || "http://127.0.0.1:8428"
      selector = %(fluke_current_amperes{mode="ac"})
      names = ["eco", "balanced", "quality"]
      reasoning = nil.as(Int32?)
      answer = nil.as(Int32?)
      prompt = nil.as(String?)
      prompt_file = nil.as(String?)
      output = nil.as(String?)
      measure_only = nil.as(String?)
      token_file = nil.as(String?)
      model_file = nil.as(String?)
      kind = "current_ac"
      voltage = 116.0
      power_factor = nil.as(Float64?)
      max_gap_ms = 5000_i64
      measurement_wait = 8.0
      repeats = 1
      warmups = 0
      dry_run = false
      require_measurement = false
      help = false
      seen = Set(String).new
      OptionParser.parse(args) do |p|
        p.banner = "usage: gemma4_energy [prompt] [options]\nFull-depth local llama.cpp inference and VictoriaMetrics raw measurement comparison."
        p.on("--server-url URL", "Local llama-server base URL (GEMMA4_SERVER_URL)") { |v| server = v }
        p.on("--profile NAME", "Run eco, balanced or quality") { |v| claim_option!(seen, "profiles"); names = [v] }
        p.on("--profiles NAMES", "Comma-separated profiles, default eco,balanced,quality") { |v| claim_option!(seen, "profiles"); names = v.split(',') }
        p.on("--reasoning-tokens N", "Override reasoning allowance; zero disables thinking") { |v| claim_option!(seen, "reasoning budget"); reasoning = v.to_i }
        p.on("--answer-tokens N", "Override independent answer allowance") { |v| claim_option!(seen, "answer budget"); answer = v.to_i }
        p.on("--prompt TEXT", "User request") { |v| claim_option!(seen, "prompt source"); prompt = v }
        p.on("--prompt-file PATH", "Read user request from file") { |v| claim_option!(seen, "prompt source"); prompt_file = v }
        p.on("--model-file PATH", "Hash the local GGUF for report provenance") { |v| model_file = v }
        p.on("--repeats N", "Measured rounds, profile order rotates each round") { |v| repeats = v.to_i }
        p.on("--warmups N", "Unmeasured 8-token Eco warmups, default zero") { |v| warmups = v.to_i }
        p.on("--vm-url URL", "VictoriaMetrics read base URL (COGNI_VM_URL)") { |v| vm_url = v }
        p.on("--vm-selector SELECTOR", "Select exactly one physical measurement series") { |v| selector = v }
        p.on("--vm-token-file PATH", "Read bearer token; alternatively COGNI_VM_BEARER_TOKEN") { |v| token_file = v }
        p.on("--measurement-kind KIND", "current_ac (Fluke) or active_power (watts)") { |v| kind = v }
        p.on("--voltage V", "Assumed RMS grid voltage, default 116") { |v| voltage = v.to_f }
        p.on("--assumed-power-factor PF", "Explicit PF assumption for estimated joules") { |v| power_factor = v.to_f }
        p.on("--max-gap-ms N", "Maximum allowed raw sample gap, default 5000") { |v| max_gap_ms = v.to_i64 }
        p.on("--measurement-wait SECONDS", "Wait for collector delivery and a trailing sample, default 8") { |v| measurement_wait = v.to_f }
        p.on("--measure-only PATH", "Re-query measurements for an existing report; no inference") { |v| measure_only = v }
        p.on("--require-measurement", "Exit 3 if any run lacks valid measurements") { require_measurement = true }
        p.on("--output PATH", "Save JSON report (stdout is also JSON)") { |v| output = v }
        p.on("--dry-run", "Print resolved budgets without contacting servers") { dry_run = true }
        p.on("-h", "--help", "Show help") { puts p; help = true }
        p.unknown_args do |before, after|
          raise ArgumentError.new("pass a single prompt or --prompt-file") if before.size > 1 || !after.empty?
          if positional = before.first?
            claim_option!(seen, "prompt source")
            prompt = positional
          end
        end
      end
      return 0 if help
      raise ArgumentError.new("profiles must be unique and nonempty") if names.empty? || names.uniq.size != names.size
      profiles = names.map { |name| Gemma4EnergyProfile.for_name(name, reasoning, answer) }
      raise ArgumentError.new("profiles must be unique") unless profiles.map(&.name).uniq.size == profiles.size
      raise ArgumentError.new("repeats must be in 1..100 and warmups in 0..10") unless (1..100).includes?(repeats) && (0..10).includes?(warmups)
      raise ArgumentError.new("measurement wait must be finite and in 0..60") unless measurement_wait.finite? && (0.0..60.0).includes?(measurement_wait)
      raise ArgumentError.new("measurement kind must be current_ac or active_power") unless {"current_ac", "active_power"}.includes?(kind)
      raise ArgumentError.new("voltage must be finite and positive") unless voltage.finite? && voltage > 0
      raise ArgumentError.new("max gap must be in 1..60000 ms") unless (1_i64..60000_i64).includes?(max_gap_ms)
      if pf = power_factor
        raise ArgumentError.new("PF must be in (0,1] and requires current_ac") unless pf.finite? && pf > 0 && pf <= 1 && kind == "current_ac"
      end
      if dry_run
        puts({profiles: profiles.map { |profile| {name: profile.name, reasoning_tokens: profile.reasoning_tokens, answer_tokens: profile.answer_tokens} }, backend: "llama.cpp", full_depth: true}.to_json)
        return 0
      end
      token = token_file.try { |path| File.read(path).strip } || ENV["COGNI_VM_BEARER_TOKEN"]?
      vm = ML::Energy::VictoriaMetrics.new(vm_url, token)
      report = if source = measure_only
                 imported = JSON.parse(File.read(source))
                 imported.as_h["inference_provenance"] = JSON::Any.new("imported_unverified; structural validation does not attest inference")
                 imported
               else
                 raise ArgumentError.new("use either --prompt or --prompt-file") if prompt && prompt_file
                 request = prompt_file.try { |path| File.read(path) } || prompt || ENV["GEMMA4_PROMPT"]? || "What is 17 times 23? Give a short answer."
                 raise ArgumentError.new("prompt must not be empty") if request.strip.empty?
                 runner = Gemma4EnergyRunner.new(server)
                 warmups.times { runner.run("Reply OK.", Gemma4EnergyProfile.for_name("eco", 0, 8)) }
                 runs = [] of JSON::Any
                 repeats.times do |round|
                   profiles.rotate(round % profiles.size).each do |profile|
                     started = Time.utc.to_unix_ms
                     monotonic_start = Time.instant
                     result = begin
                       runner.run(request, profile)
                     rescue ex
                       JSON.parse({status: "failed", usage_complete: false, comparison_eligible: false, model_identity: runner.model_identity, phases: nil, profile: {name: profile.name, reasoning_tokens: profile.reasoning_tokens, answer_tokens: profile.answer_tokens}, error: ex.message || ex.class.name, started_at_epoch_ms: started, ended_at_epoch_ms: Time.utc.to_unix_ms, elapsed_ms: (Time.instant - monotonic_start).total_milliseconds.to_i64}.to_json)
                     end
                     result.as_h["round"] = JSON::Any.new(round.to_i64)
                     runs << result
                   end
                 end
                 identities = runs.compact_map { |run| run["model_identity"]?.try(&.as_s?) }.uniq
                 raise ArgumentError.new("backend model identity changed during comparison") if identities.size > 1
                 identity = identities.first?
                 digest = model_file.try do |path|
                   # Backend failure still yields a cost window even when no
                   # validated loaded identity is available for hashing.
                   next nil unless identity
                   unless File.realpath(path) == File.realpath(identity)
                     raise ArgumentError.new("--model-file must identify the loaded model on the same filesystem as llama-server")
                   end
                   File.open(path) { |file| Digest::SHA256.hexdigest(file) }
                 end
                 JSON.parse({schema: "cogni.gemma4.energy.v1", backend: "llama.cpp", model_identity: identity, model_sha256: digest, prompt_sha256: Digest::SHA256.hexdigest(request), full_depth: true, cache_prompt: false, temperature: 0, warmups: warmups, runs: runs}.to_json)
               end
      validate_report!(report)
      sleep measurement_wait.seconds if measurement_wait > 0
      runs = report["runs"].as_a
      runs.each do |run|
        measurement = begin
          vm.measure(selector, run["started_at_epoch_ms"].as_i64, run["ended_at_epoch_ms"].as_i64, kind, voltage, power_factor, max_gap_ms)
        rescue ex
          JSON.parse({status: "unavailable", reason: ex.message || ex.class.name}.to_json)
        end
        run.as_h["measurement"] = measurement
      end
      report.as_h["measurement_selector"] = JSON::Any.new(selector)
      report.as_h["notes"] = JSON::Any.new("Whole circuit request windows; includes both phase HTTP requests and re-prefill. Excludes model load, warmup, and post-run collector wait. Cross-host clocks must be synchronized. No automatic answer-quality ranking; inspect answer and truncation before comparing cost.")
      json = report.to_pretty_json
      File.write(output.not_nil!, json + "\n") if output
      puts json
      return 1 if runs.any? { |run| run["status"]?.try(&.as_s?) == "failed" }
      return 3 if require_measurement && runs.any? { |run| run["measurement"]["status"].as_s != "available" }
      0
    end
  end
end
