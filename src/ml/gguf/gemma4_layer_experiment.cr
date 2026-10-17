require "option_parser"
require "digest/sha256"
require "http/client"
require "../energy/victoria_metrics"
require "../energy/prometheus_remote_write"
require "./gemma4_chat"
require "./gemma4_token_progress"

module ML::GGUF
  # A deliberately separate, fixed-work experiment. It does not change the
  # production profiles or treat a pruned decoder as a validated language model.
  module Gemma4LayerExperiment
    PROMPT = "Explain how an offline enterprise server processes language model requests. Discuss GPUs, memory, latency, and energy consumption in a continuous paragraph."
    GENERATED_TOKEN_METRIC = "cogni_llm_generated_tokens_total"

    def self.validate_completion!(response : JSON::Any, requested : Int32) : Int64
      actual = response["tokens_predicted"].as_i64
      raise ArgumentError.new("backend did not generate the exact fixed token count") unless actual == requested
      raise ArgumentError.new("backend did not stop at the fixed token limit") unless response["stop_type"].as_s == "limit"
      raise ArgumentError.new("backend reported a truncated prompt") if response["truncated"]?.try(&.as_bool)
      raise ArgumentError.new("backend prompt usage is missing") unless response["tokens_evaluated"].as_i64 > 0
      tokens = response["tokens"].as_a
      unless tokens.size == actual && tokens.all? { |token| token.as_i64 >= 0 }
        raise ArgumentError.new("returned token IDs disagree with fixed-work usage")
      end
      actual
    end

    def self.request(base : String, path : String, payload : String? = nil) : JSON::Any
      uri = URI.parse(base.rstrip('/') + path)
      unless uri.scheme == "http" && {"127.0.0.1", "localhost", "::1", "[::1]"}.includes?(uri.host) && uri.user.nil? && uri.password.nil?
        raise ArgumentError.new("experiment backend must be loopback HTTP without credentials")
      end
      response = HTTP::Client.new(uri) do |client|
        client.connect_timeout = 5.seconds
        client.read_timeout = 90.seconds
        payload ? client.post(uri.request_target, HTTP::Headers{"Content-Type" => "application/json"}, payload) : client.get(uri.request_target)
      end
      raise ArgumentError.new("backend HTTP #{response.status_code}") unless response.status_code == 200
      parsed = JSON.parse(response.body)
      raise ArgumentError.new("backend response must be a JSON object") unless parsed.as_h?
      parsed
    end

    def self.request_payload(rendered : String, budget : Int32, token_progress : Bool = false) : String
      {
        prompt: rendered, n_predict: budget, temperature: 0, seed: 1,
        ignore_eos: true, stop: [] of String, cache_prompt: false,
        stream: token_progress, return_tokens: true,
        timings_per_token: token_progress,
      }.to_json
    end

    def self.stream_completion(base : String, payload : String,
                               collector : Gemma4TokenProgress::Collector,
                               monotonic_start : Time::Instant) : Tuple(JSON::Any, String)
      uri = URI.parse(base.rstrip('/') + "/completion")
      unless uri.scheme == "http" && {"127.0.0.1", "localhost", "::1", "[::1]"}.includes?(uri.host) && uri.user.nil? && uri.password.nil?
        raise ArgumentError.new("experiment backend must be loopback HTTP without credentials")
      end
      final_event = nil.as(JSON::Any?)
      content = String::Builder.new
      event_data = [] of String
      HTTP::Client.new(uri) do |client|
        client.connect_timeout = 5.seconds
        client.read_timeout = 90.seconds
        client.exec("POST", uri.request_target, HTTP::Headers{"Content-Type" => "application/json", "Accept" => "text/event-stream"}, payload) do |response|
          unless response.status_code == 200
            raise ArgumentError.new("backend HTTP #{response.status_code}")
          end
          consume_frame = -> {
            unless event_data.empty?
              data = event_data.join("\n")
              event_data.clear
              unless data == "[DONE]"
                received_epoch_ms = Time.utc.to_unix_ms
                received_monotonic_ms = monotonic_start.elapsed.total_milliseconds.to_i64
                event = JSON.parse(data)
                if event.as_h["error"]?
                  raise ArgumentError.new("backend returned a streaming error")
                end
                chunk = event["content"]?
                if chunk && !chunk.raw.nil?
                  content << chunk.as_s
                end
                collector.consume(data, received_epoch_ms, received_monotonic_ms)
                final_event = event if event["stop"]?.try(&.as_bool)
              end
            end
          }
          response.body_io.each_line do |raw_line|
            line = raw_line.chomp
            if line.empty?
              consume_frame.call
              next
            end
            next if line.starts_with?(':')
            if line.starts_with?("data:")
              value = line[5..-1]
              event_data << (value.starts_with?(' ') ? value[1..-1] : value)
            end
          end
          consume_frame.call
        end
      end
      raise ArgumentError.new("stream truncated before final response event") unless final_event
      final = final_event.not_nil!
      streamed_content = content.to_s
      aggregate = final.as_h.dup
      aggregate["tokens"] = JSON.parse(collector.token_ids.to_json)
      aggregate["content"] = JSON::Any.new(streamed_content)
      {JSON::Any.new(aggregate), streamed_content}
    end

    private def self.remote_write_labels(campaign : String, label : String,
                                         layers : Int32, trial : Int32) : Hash(String, String)
      {"job" => "cogni-gemma4", "host" => "gputer", "campaign" => campaign,
       "label" => label, "layers" => layers.to_s, "trial" => trial.to_s,
       "phase" => "answer"}
    end

    private def self.remote_write_selector(labels : Hash(String, String)) : String
      label_list = labels.to_a.sort_by(&.[0]).map do |name, value|
        %(#{name}="#{ML::Energy::PrometheusRemoteWrite.escape_label_value(value)}")
      end.join(",")
      "#{GENERATED_TOKEN_METRIC}{#{label_list}}"
    end

    def self.run(args : Array(String)) : Int32
      backend = "http://127.0.0.1:8080"
      vm_url = "http://127.0.0.1:8428"
      layers = 0
      label = ""
      campaign = ""
      model_file = ""
      output = ""
      tokens = 256
      trials = 3
      wait = 8
      settle = 3
      token_progress = false
      OptionParser.parse(args) do |parser|
        parser.on("--backend-url URL", "Loopback llama-server") { |v| backend = v }
        parser.on("--vm-url URL", "VictoriaMetrics URL") { |v| vm_url = v }
        parser.on("--layers N", "Decoder count verified in the loader log") { |v| layers = v.to_i }
        parser.on("--label NAME", "Depth and campaign position") { |v| label = v }
        parser.on("--campaign-id ID", "GPU collector run label") { |v| campaign = v }
        parser.on("--model-file PATH", "Loaded GGUF on the same filesystem") { |v| model_file = v }
        parser.on("--output PATH", "New report file") { |v| output = v }
        parser.on("--tokens N", "Exact generated work; EOS is ignored") { |v| tokens = v.to_i }
        parser.on("--trials N", "Measured requests") { |v| trials = v.to_i }
        parser.on("--measurement-wait N", "VM flush/after-sample wait in seconds") { |v| wait = v.to_i }
        parser.on("--settle N", "Seconds of idle time between requests") { |v| settle = v.to_i }
        parser.on("--token-progress", "Stream token IDs and write observed progress to VictoriaMetrics") { token_progress = true }
      end
      unless layers.in?(1..48) && tokens.in?(1..1024) && trials.in?(1..10) && wait.in?(0..30) && settle.in?(0..30)
        raise ArgumentError.new("invalid bounded layer/token/trial/wait settings")
      end
      raise ArgumentError.new("campaign ID and label are required") if campaign.empty? || label.empty?
      if token_progress
        unless campaign.matches?(/\A[A-Za-z0-9._-]{1,64}\z/) && label.matches?(/\A[A-Za-z0-9._-]{1,64}\z/)
          raise ArgumentError.new("token-progress campaign ID and label must be bounded Prometheus-safe IDs")
        end
      end
      raise ArgumentError.new("model and new output paths are required") if model_file.empty? || output.empty? || File.exists?(output)
      # Reject invalid measurement configuration before spending GPU work.
      vm = ML::Energy::VictoriaMetrics.new(vm_url)
      remote_writer = token_progress ? ML::Energy::PrometheusRemoteWrite.new(vm_url) : nil
      props = request(backend, "/props")
      raise ArgumentError.new("experiment requires a single server slot") unless props["total_slots"].as_i == 1
      identity = props["model_path"].as_s
      raise ArgumentError.new("model path does not match the loaded model") unless File.realpath(model_file) == File.realpath(identity)
      raise ArgumentError.new("loaded model must identify Gemma 4") unless identity.downcase.gsub(/[^a-z0-9]/, "").includes?("gemma4")
      context = props["default_generation_settings"]["n_ctx"].as_i
      raise ArgumentError.new("experiment requires at least 2048 context tokens") unless context >= 2048
      digest = File.open(model_file) { |file| Digest::SHA256.hexdigest(file) }
      rendered = Gemma4Chat.render_user_prompt(PROMPT, enable_thinking: false)
      # Exclude model loading, hashing and one short warmup from request windows.
      warmup = request(backend, "/completion", request_payload(rendered, 8))
      validate_completion!(warmup, 8)
      sleep settle.seconds
      runs = [] of JSON::Any
      trials.times do |index|
        started = Time.utc.to_unix_ms
        monotonic = Time.instant
        response = nil.as(JSON::Any?)
        actual = nil.as(Int64?)
        error = nil.as(String?)
        collector = token_progress ? Gemma4TokenProgress::Collector.new(started, tokens) : nil
        streamed_content = nil.as(String?)
        begin
          if progress = collector
            response, streamed_content = stream_completion(backend, request_payload(rendered, tokens, true), progress, monotonic)
          else
            response = request(backend, "/completion", request_payload(rendered, tokens))
          end
          actual = validate_completion!(response, tokens)
        rescue ex
          error = ex.message || ex.class.name
        end
        ended = Time.utc.to_unix_ms
        if progress = collector
          if error.nil?
            begin
              progress.finish!(ended, monotonic.elapsed.total_milliseconds.to_i64)
            rescue ex
              error = ex.message || ex.class.name
            end
          end
        end
        run = JSON.parse({label: label, decoder_layers: layers, trial: index + 1,
                          status: error ? "failed" : "completed", error: error,
                          model_identity: identity, model_sha256: digest,
                          started_at_epoch_ms: started, ended_at_epoch_ms: ended,
                          elapsed_ms: monotonic.elapsed.total_milliseconds,
                          actual_tokens: actual, requested_tokens: tokens,
                          tokens_reported: response.try { |r| r["tokens_predicted"]?.try(&.as_i64?) },
                          backend_timings: response.try { |r| r["timings"]? },
                          generated_text_sha256: if stream_content = streamed_content
                            Digest::SHA256.hexdigest(stream_content)
                          else
                            response.try { |r| r["content"]?.try(&.as_s?).try { |text| Digest::SHA256.hexdigest(text) } }
                          end}.to_json)
        if progress = collector
          run.as_h["token_progress"] = progress.to_json(status: error ? "partial" : "available")
        end
        runs << run
        # A failed inference is retained with its cost window; stop this batch.
        break if error
        sleep settle.seconds
      end
      telemetry_error = nil.as(String?)
      if writer = remote_writer
        write_series = [] of ML::Energy::PrometheusRemoteWrite::Series
        run_selectors = {} of Int32 => String
        runs.each do |run|
          run_number = run["trial"].as_i
          labels = remote_write_labels(campaign, label, layers, run_number)
          selector = remote_write_selector(labels)
          run_selectors[run_number] = selector
          progress = run["token_progress"]
          start_ms = run["started_at_epoch_ms"].as_i64
          points = progress["points"].as_a.map do |point|
            timestamp = start_ms + point["elapsed_ms"].as_i64
            count = point["generated_tokens"].as_i64
            ML::Energy::PrometheusRemoteWrite::Sample.new(timestamp, count.to_f64)
          end
          write_series << ML::Energy::PrometheusRemoteWrite::Series.new(GENERATED_TOKEN_METRIC, labels, points)
        end
        begin
          writer.write(write_series)
        rescue ex
          telemetry_error = ex.message || ex.class.name
        end
        runs.each do |run|
          run_number = run["trial"].as_i
          point_count = run["token_progress"]["points"].as_a.size
          run["token_progress"].as_h["remote_write"] = JSON.parse({
            status: telemetry_error ? "failed" : "available",
            metric: GENERATED_TOKEN_METRIC,
            selector: run_selectors[run_number],
            endpoint_path: writer.endpoint.path,
            sample_count: point_count,
            reason: telemetry_error,
          }.to_json)
        end
      end
      sleep wait.seconds if wait > 0
      gpu_selector = "cogni_gpu_power_watts{host=\"gputer\",gpu=\"0\",run=#{campaign.to_json}}"
      utilization_selector = "cogni_gpu_utilization_percent{host=\"gputer\",gpu=\"0\",run=#{campaign.to_json}}"
      runs.each do |run|
        start_ms = run["started_at_epoch_ms"].as_i64
        end_ms = run["ended_at_epoch_ms"].as_i64
        {"gpu_measurement"     => {gpu_selector, "active_power"},
         "circuit_measurement" => {"fluke_current_amperes{circuit=\"gputer\",mode=\"ac\"}", "current_ac"}}.each do |name, setting|
          run.as_h[name] = begin
            vm.measure(setting[0], start_ms, end_ms, setting[1], max_gap_ms: 2000_i64)
          rescue ex
            JSON.parse({status: "unavailable", reason: ex.message || ex.class.name}.to_json)
          end
        end
        run.as_h["gpu_utilization_measurement"] = begin
          series = vm.read_series(utilization_selector, start_ms, end_ms, 2000_i64)
          values = series.samples.select { |time, _| start_ms <= time <= end_ms }.map(&.[1])
          raise ArgumentError.new("no utilization samples within request window") if values.empty?
          raise ArgumentError.new("utilization exceeds 100 percent") if values.any? { |value| value > 100 }
          JSON.parse({status: "available", mean_percent: values.sum / values.size,
                      sample_count: values.size, method: "arithmetic mean of samples within window",
                      source: "whole-board utilization sample"}.to_json)
        rescue ex
          JSON.parse({status: "unavailable", reason: ex.message || ex.class.name}.to_json)
        end
      end
      report = {schema: "cogni.gemma4.layer_depth.v1", campaign_id: campaign,
                prompt_sha256: Digest::SHA256.hexdigest(rendered), requested_tokens: tokens,
                layers_evidence: "decoder count is supplied after inspection of this model's llama.cpp loader log",
                full_depth: layers == 48, ignore_eos: true, cache_prompt: false,
                temperature: 0, seed: 1, warmup_tokens: 8, model_file_bytes: File.size(model_file),
                notes: "Experimental prefix exit; answer quality is unvalidated. NVIDIA measures whole-board W/J on a shared GPU; Fluke measures whole-circuit VA/VA s at assumed 116 V. Fixed work includes generation past EOS. No idle-power subtraction. Loading, hashing, warmup and waits are excluded. Sensor averaging and host-clock offset remain measurement limits.",
                runs: runs}
      File.write(output, report.to_pretty_json + "\n")
      STDOUT.puts report.to_json
      runs.any? { |run| run["status"].as_s == "failed" } || telemetry_error ? 1 : 0
    end
  end
end
