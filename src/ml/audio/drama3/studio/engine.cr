require "../generator_resident"
require "../codec_resident"
require "../codec_encoder_resident"
require "../reference_audio"
require "../wav"

{% if flag?(:drama3_metal) && flag?(:darwin) && !flag?(:cpu_only) %}
  module ML::Audio::Drama3::Studio
    # A single worker owns the model and context. Only static converted weights
    # and compiled pipelines survive a successful job; temporal state is fresh.
    class Engine
      @config : JSON::Any
      @store : WeightStore
      @tokenizer : Tokenizer
      @tower_config : TowerConfig
      @context : Resident::Context
      @jobs = 0
      getter load_seconds : Float64

      def initialize(@model : String)
        started = Time.instant
        raise ArgumentError.new("studio generation requires resident Metal") unless MetalBackend.enabled?
        @config = JSON.parse(File.read(File.join(@model, "config.json")))
        raise ArgumentError.new("only fish_omni at 44100 Hz is supported") unless @config["model_type"].as_s == "fish_omni" && @config["sample_rate"].as_i == 44100
        @tower_config = TowerConfig.from_json(@config.to_json)
        raise ArgumentError.new("ten codec codebooks are required") unless @tower_config.num_codebooks == 10
        @tokenizer = Tokenizer.new(File.join(@model, "tokenizer.json"), @config["eos_token_id"].as_i, @config["pad_token_id"].as_i)
        @store = WeightStore.new(@model)
        @context = Resident::Context.new
        @load_seconds = (Time.instant - started).total_seconds
      end

      # A failure is fatal to the worker. The supervisor starts a fresh process
      # for the next job instead of resuming potentially partial GPU state.
      def generate(input : JSON::Any, output : String, &progress : String -> Nil) : JSON::Any
        started = Time.instant
        host_before = @context.host_profile.snapshot
        raise ArgumentError.new("output already exists") if File.exists?(output) || File.exists?(output + ".json")
        text = input["text"].as_s
        max_frames = input["max_frames"].as_i
        seed = input["seed"].as_i64.to_u64
        temperature = input["temperature"].as_f
        top_k = input["top_k"].as_i
        top_p = input["top_p"].as_f
        raise ArgumentError.new("empty or excessive text") if text.strip.empty? || text.bytesize > 16_384
        raise ArgumentError.new("invalid frame bound") unless max_frames.in?(1..4096)
        reference_path = input["reference_audio"]?.try(&.as_s?)
        reference_text = input["reference_text"]?.try(&.as_s?) || ""
        reference = reference_path ? Wav.read_reference(reference_path) : nil
        plan = reference ? Prompt.reference_plan(text, reference_text, reference.not_nil!.frames, @tokenizer,
          @tower_config.num_codebooks, @config["semantic_start_token_id"].as_i, @tower_config.vocab_size) : nil
        prompt = plan ? [] of Array(Int32) : Prompt.build(text, @tokenizer, @tower_config.num_codebooks)
        rows = plan.try(&.row_count) || prompt.size
        capacity = rows.to_i64 + max_frames - 1
        raise ArgumentError.new("prompt plus frame bound exceeds 4096 positions") unless capacity.in?(1_i64..4096_i64)
        before_graph = @context.graph_stats
        before_reads = @context.readbacks.dup
        before_alloc = @context.allocation_stats
        generator = ResidentGenerator.new(@config, @store, @context, capacity.to_i32)
        # Progress is telemetry, not a per-frame pipe/disk transaction. This
        # clock check does not submit, wait for or read any device work.
        last_progress = Time.instant - 1.second
        generator.progress = ->(message : String) do
          now = Time.instant
          if now - last_progress >= 1.second
            progress.call(message)
            last_progress = now
          end
          nil
        end
        generation_started = Time.instant
        generation_alloc = @context.allocation_stats
        host_generation = @context.host_profile.snapshot
        result = if reference && plan
                   progress.call("Кодирование референса")
                   audio = ReferenceAudio.upload(reference, @context)
                   encoded = CodecEncoderResident.new(@store, @context).encode(audio, reference.samples)
                   raise "reference frame count changed" unless encoded.frames == plan.audio_row_count
                   resident_prompt = PromptResident.build(plan, encoded.codes, @context)
                   raise "reference handoff introduced a read" unless @context.readbacks == before_reads
                   generator.generate(resident_prompt, max_frames, seed, temperature, top_k, top_p)
                 else
                   generator.generate(prompt, max_frames, seed, temperature, top_k, top_p)
                 end
        generation_seconds = (Time.instant - generation_started).total_seconds
        host_codec = @context.host_profile.snapshot
        codec_alloc = @context.allocation_stats
        count = result.generation.frames.size
        raise ArgumentError.new("model produced EOS without audio") if count == 0
        progress.call("Декодирование #{count} кадров")
        codec_started = Time.instant
        codec = CodecResident.new(@store, @context)
        waveform = codec.decode(result.deviceframes.slice(0, count * (@tower_config.num_codebooks + 1)), count)
        samples = @context.read(waveform, "terminal")
        codec_seconds = (Time.instant - codec_started).total_seconds
        host_after_codec = @context.host_profile.snapshot
        peak = samples.max_of(&.abs)
        rms = Math.sqrt(samples.sum { |value| value.to_f64 * value } / samples.size)
        raise ArgumentError.new("waveform is silent") unless peak > 1e-6 && rms > 1e-7
        Wav.write(output, samples)
        host_after_wav = @context.host_profile.snapshot
        graph = @context.graph_stats
        raise "unfinished GPU job" unless graph["pending"] == 0 && graph["completed"] == graph["submitted"]
        reads = @context.readbacks.transform_values { |value, key| value - (before_reads[key]? || 0_i64) }
        allocation = @context.allocation_stats
        report = JSON.parse({backend: "metal_resident", worker_job: @jobs + 1, warm_worker: @jobs > 0,
                             model_path: @model, model_load_seconds: @load_seconds, text: text, seed: seed,
                             max_frames: max_frames, temperature: temperature, top_k: top_k, top_p: top_p,
                             reference_sha256: reference.try(&.sha256), prompt_rows: rows,
                             frames: result.generation.frames, stopped_on_eos: result.generation.stopped,
                             truncated: !result.generation.stopped, sample_rate: 44100, samples: samples.size,
                             seconds: samples.size / 44100.0, peak: peak, rms: rms,
                             sampler_inputs: generator.sampler_inputs, resident_readbacks: reads,
                             resident_graph: graph.transform_values { |value, key| key.in?("submitted", "completed", "completion_waits") ? value - before_graph[key] : value },
                             resident_allocation_stats: {allocations: allocation[:allocations] - before_alloc[:allocations],
                                                         requested_bytes: allocation[:requested_bytes] - before_alloc[:requested_bytes],
                                                         allocation_profile: allocation[:allocation_profile],
                                                         allocation_ns: allocation[:allocation_ns] - before_alloc[:allocation_ns]},
                             allocation_phases: {initialize: {allocations: generation_alloc[:allocations] - before_alloc[:allocations],
                                                              requested_bytes: generation_alloc[:requested_bytes] - before_alloc[:requested_bytes],
                                                              allocation_ns: generation_alloc[:allocation_ns] - before_alloc[:allocation_ns]},
                                                 generation: {allocations: codec_alloc[:allocations] - generation_alloc[:allocations],
                                                              requested_bytes: codec_alloc[:requested_bytes] - generation_alloc[:requested_bytes],
                                                              allocation_ns: codec_alloc[:allocation_ns] - generation_alloc[:allocation_ns]},
                                                 codec: {allocations: allocation[:allocations] - codec_alloc[:allocations],
                                                         requested_bytes: allocation[:requested_bytes] - codec_alloc[:requested_bytes],
                                                         allocation_ns: allocation[:allocation_ns] - codec_alloc[:allocation_ns]}},
                             resident_dac_scratch: codec.dac_scratch_stats,
                             resident_generation_scratch: generator.workspace_stats,
                             host_profile: {enabled: @context.host_profile.enabled?,
                                            initialize: HostProfile.delta(host_before, host_generation),
                                            generation: HostProfile.delta(host_generation, host_codec),
                                            codec: HostProfile.delta(host_codec, host_after_codec),
                                            postprocess: HostProfile.delta(host_after_codec, host_after_wav),
                                            through_wav: HostProfile.delta(host_before, host_after_wav)},
                             elapsed_seconds: (Time.instant - started).total_seconds,
                             phase_seconds: {initialize: (generation_started - started).total_seconds,
                                             generation: generation_seconds, codec: codec_seconds},
        }.to_json)
        File.write(output + ".json", report.to_pretty_json)
        @jobs += 1
        report
      rescue ex
        @context.abort
        raise ex
      end
    end
  end
{% end %}
