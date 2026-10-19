require "../generator_resident"
require "../codec_resident"
require "../codec_encoder_resident"
require "../reference_audio"
require "../wav"
require "../window_memory"
require "./frame_window"
require "./memory_guard"
require "digest/sha256"

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
      @memory_guard : MemoryGuard
      @memory_stages = {} of String => JSON::Any
      @sampled_footprint = 0_i64
      @sampled_metal = 0_i64
      @memory_samples = 0
      @minimum_free_percent = 100.0
      @jobs = 0
      getter load_seconds : Float64

      def initialize(@model : String)
        started = Time.instant
        @memory_guard = MemoryGuard.from_env
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

      # Only host metadata is examined. No tensor read, submit or wait is added.
      # The caller invokes this after load and the existing between-job GC.
      def memory_status(stage : String) : JSON::Any
        sample = observe_memory(stage, retain: false)
        cache = @context.converted_weight_cache_signature
        sample.as_h["converted_weight_cache"] = JSON.parse({tensors:         cache.size,
                                                            bytes:           cache.sum(0_i64) { |entry| entry[3].to_i64 * 4 },
                                                            identity_sha256: Digest::SHA256.hexdigest(cache.to_json)}.to_json)
        sample.as_h["resident_graph"] = JSON.parse(@context.graph_stats.to_json)
        sample.as_h["worker_pid"] = JSON::Any.new(Process.pid.to_i64)
        sample.as_h["completed_jobs"] = JSON::Any.new(@jobs.to_i64)
        sample
      end

      private def observe_memory(stage : String, retain : Bool = true) : JSON::Any
        sample = JSON.parse(WindowMemory.snapshot.to_json)
        @memory_guard.check!(sample, stage)
        @sampled_footprint = Math.max(@sampled_footprint, sample["process_footprint_bytes"].as_i64)
        @sampled_metal = Math.max(@sampled_metal, sample["metal_allocated_bytes"].as_i64)
        @minimum_free_percent = Math.min(@minimum_free_percent,
          100.0 * sample["free_reclaimable_bytes"].as_i64 / sample["physical_bytes"].as_i64)
        @memory_samples += 1
        @memory_stages[stage] = sample if retain
        sample
      end

      private def frame_window(rows : Int32, cap : Int32, automatic : Bool, reference : Bool,
                               sample : JSON::Any) : FrameWindow::Selection
        return FrameWindow.manual(rows, cap) unless automatic
        inventory = @store.resident_memory_inventory
        text_config = @config["text_config"]
        request = Drama3WindowBudget::Request.new(
          BackboneConfig.new(text_config), @tower_config, CodecConfig.new, rows, 1, cap,
          inventory[:mapped_bytes], inventory[:largest_checkpoint_mapped_buffer_bytes],
          inventory[:converted_static_bytes], inventory[:largest_converted_buffer_bytes],
          1024_i64 * 1024 * 1024,
          text_config["moe_intermediate_size"]?.try(&.as_i),
          text_config["shared_expert_intermediate_size"]?.try(&.as_i))
        allocated = sample["metal_allocated_bytes"].as_i64
        recommended = sample["metal_recommended_bytes"].as_i64
        if remaining = @memory_guard.remaining_bytes(sample)
          recommended = Math.min(recommended, allocated + remaining)
        end
        telemetry = Drama3WindowBudget::Telemetry.new(
          sample["physical_bytes"].as_i64, sample["free_reclaimable_bytes"].as_i64,
          recommended, allocated, MetalBackend.max_buffer_length)
        FrameWindow.automatic(request, telemetry, reference, ENV["DRAMA3_DAC_SCRATCH"]? == "1")
      end

      # A failure is fatal to the worker. The supervisor starts a fresh process
      # for the next job instead of resuming potentially partial GPU state.
      def generate(input : JSON::Any, output : String, &progress : String -> Nil) : JSON::Any
        started = Time.instant
        @memory_stages.clear
        @sampled_footprint = @sampled_metal = 0_i64
        @memory_samples = 0
        @minimum_free_percent = 100.0
        memory_begin = observe_memory("request_begin")
        resources_before = HostResourceSnapshot.capture if @context.resources_enabled
        host_before = @context.host_profile.snapshot
        raise ArgumentError.new("output already exists") if File.exists?(output) || File.exists?(output + ".json")
        text = input["text"].as_s
        requested_frames = input["max_frames"].as_i
        automatic = input["auto_frames"]?.try(&.as_bool) || false
        seed = input["seed"].as_i64.to_u64
        temperature = input["temperature"].as_f
        top_k = input["top_k"].as_i
        top_p = input["top_p"].as_f
        raise ArgumentError.new("empty or excessive text") if text.strip.empty? || text.bytesize > 16_384
        raise ArgumentError.new("invalid frame bound") unless requested_frames.in?(1..4096)
        reference_path = input["reference_audio"]?.try(&.as_s?)
        reference_text = input["reference_text"]?.try(&.as_s?) || ""
        reference = reference_path ? Wav.read_reference(reference_path) : nil
        plan = reference ? Prompt.reference_plan(text, reference_text, reference.not_nil!.frames, @tokenizer,
          @tower_config.num_codebooks, @config["semantic_start_token_id"].as_i, @tower_config.vocab_size) : nil
        prompt = plan ? [] of Array(Int32) : Prompt.build(text, @tokenizer, @tower_config.num_codebooks)
        rows = plan.try(&.row_count) || prompt.size
        window = frame_window(rows, requested_frames, automatic, !!reference, memory_begin)
        max_frames = window.frames
        capacity = window.capacity
        before_graph = @context.graph_stats
        before_reads = @context.readbacks.dup
        before_alloc = @context.allocation_stats
        before_down_sum_dispatches = @context.expert_down_sum_fused_dispatches
        before_expert_swiglu_dispatches = @context.expert_swiglu_fused_dispatches
        before_gdn_post_dispatches = @context.gdn_post_fused_dispatches
        before_dense_swiglu_dispatches = @context.dense_swiglu_fused_dispatches
        generator = ResidentGenerator.new(@config, @store, @context, capacity.to_i32)
        observe_memory("initialize_end")
        # Progress is telemetry, not a per-frame pipe/disk transaction. This
        # clock check does not submit, wait for or read any device work.
        last_progress = Time.instant - 1.second
        generator.progress = ->(message : String) do
          # The guard runs on every existing row/frame callback, independently
          # of the throttled pipe writes. It never synchronizes GPU work.
          observe_memory("generation_progress", retain: false)
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
        resources_generation = HostResourceSnapshot.capture if @context.resources_enabled
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
        observe_memory("generation_end")
        host_codec = @context.host_profile.snapshot
        resources_codec = HostResourceSnapshot.capture if @context.resources_enabled
        codec_alloc = @context.allocation_stats
        count = result.generation.frames.size
        raise ArgumentError.new("model produced EOS without audio") if count == 0
        progress.call("Декодирование #{count} кадров")
        codec_started = Time.instant
        codec = CodecResident.new(@store, @context)
        waveform = codec.decode(result.deviceframes.slice(0, count * (@tower_config.num_codebooks + 1)), count)
        samples = @context.read(waveform, "terminal")
        observe_memory("codec_end")
        codec_seconds = (Time.instant - codec_started).total_seconds
        host_after_codec = @context.host_profile.snapshot
        peak = samples.max_of(&.abs)
        rms = Math.sqrt(samples.sum { |value| value.to_f64 * value } / samples.size)
        raise ArgumentError.new("waveform is silent") unless peak > 1e-6 && rms > 1e-7
        Wav.write(output, samples)
        observe_memory("wav_end")
        host_after_wav = @context.host_profile.snapshot
        resources_after_wav = HostResourceSnapshot.capture if @context.resources_enabled
        graph = @context.graph_stats
        raise "unfinished GPU job" unless graph["pending"] == 0 && graph["completed"] == graph["submitted"]
        reads = @context.readbacks.transform_values { |value, key| value - (before_reads[key]? || 0_i64) }
        allocation = @context.allocation_stats
        report = JSON.parse({backend: "metal_resident", worker_pid: Process.pid, worker_job: @jobs + 1, warm_worker: @jobs > 0,
                             model_path: @model, model_load_seconds: @load_seconds, text: text, seed: seed,
                             max_frames: max_frames, temperature: temperature, top_k: top_k, top_p: top_p,
                             requested_max_frames: requested_frames, capacity: capacity,
                             window_mode: automatic ? "experimental_auto" : "manual", window_plan: window.plan,
                             memory_stages: @memory_stages,
                             memory_observations: {sample_count: @memory_samples, sampled_max_footprint_bytes: @sampled_footprint,
                                                   sampled_max_metal_bytes: @sampled_metal, minimum_free_reclaimable_percent: @minimum_free_percent,
                                                   limit_bytes: @memory_guard.limit_bytes, min_free_percent: 15,
                                                   hard_peak_bound: false, warm_static_allowance: "conservative_cold_inventory"},
                             reference_sha256: reference.try(&.sha256), prompt_rows: rows,
                             frames: result.generation.frames, stopped_on_eos: result.generation.stopped,
                             truncated: !result.generation.stopped, sample_rate: 44100, samples: samples.size,
                             seconds: samples.size / 44100.0, peak: peak, rms: rms,
                             sampler_inputs: generator.sampler_inputs, resident_readbacks: reads,
                             resident_tower_ffn_pair_fusion: generator.tower.ffn_pair_fusion_enabled,
                             resident_expert_down_sum_fusion: @context.expert_down_sum_fusion_enabled,
                             resident_expert_down_sum_fused_dispatches: @context.expert_down_sum_fused_dispatches - before_down_sum_dispatches,
                             resident_expert_swiglu_fusion: @context.expert_swiglu_fusion_enabled,
                             resident_expert_swiglu_fused_dispatches: @context.expert_swiglu_fused_dispatches - before_expert_swiglu_dispatches,
                             resident_gdn_post_fusion: @context.gdn_post_fusion_enabled,
                             resident_gdn_post_fused_dispatches: @context.gdn_post_fused_dispatches - before_gdn_post_dispatches,
                             resident_dense_swiglu_fused_dispatches: @context.dense_swiglu_fused_dispatches - before_dense_swiglu_dispatches,
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
                             host_resources: @context.resources_enabled ? {request_begin: resources_before,
                                                                           generation_begin: resources_generation,
                                                                           generation_end: resources_codec,
                                                                           wav_end: resources_after_wav} : nil,
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
        @jobs += 1
        report
      rescue ex
        @context.abort
        raise ex
      end
    end
  end
{% end %}
