require "option_parser"
require "digest/sha256"
require "../src/ml/audio/drama3/generator"
require "../src/ml/audio/drama3/codec"
require "../src/ml/audio/drama3/wav"
require "../src/ml/audio/drama3/generator_resident"
require "../src/ml/audio/drama3/codec_resident"
require "../src/ml/audio/drama3/codec_encoder_resident"
require "../src/ml/audio/drama3/reference_audio"
require "../src/ml/audio/drama3/window_budget"
require "../src/ml/audio/drama3/window_memory"

# Offline native inference entry point. MLX/Python are reference tools only.
model_path = ""
text = ""
output = "drama3.wav"
max_frames = 160
auto_frames = false
frame_cap = 4096
frame_cap_given = false
memory_plan_only = false
seed = 0_u64
temperature = 0.8
top_k = 30
top_p = 0.9
reference_audio = ""
reference_text = ""
OptionParser.parse do |parser|
  parser.banner = "usage: drama3_generate --model DIR --text TEXT [--output FILE.wav]"
  parser.on("--model DIR", "Pinned MLX-converted model directory") { |v| model_path = File.expand_path(v) }
  parser.on("--text TEXT", "Short single-speaker text") { |v| text = v }
  parser.on("--reference-audio FILE", "Mono PCM16 44.1 kHz reference WAV, at most 10 seconds (Metal)") { |v| reference_audio = File.expand_path(v) }
  parser.on("--reference-text TEXT", "Exact transcript of the reference WAV, speaker 0") { |v| reference_text = v }
  parser.on("--output FILE", "PCM16 mono WAV destination") { |v| output = File.expand_path(v) }
  parser.on("--max-frames N|auto", "Bound generation; auto is experimental (default 160)") do |v|
    auto_frames = v == "auto"
    max_frames = v.to_i unless auto_frames
  end
  parser.on("--frame-cap N", "Generated-frame ceiling for auto, at most 4096") { |v| frame_cap = v.to_i; frame_cap_given = true }
  parser.on("--memory-plan-only", "Print the auto budget without running inference") { memory_plan_only = true }
  parser.on("--seed N", "Native sampling seed") { |v| seed = v.to_u64 }
  parser.on("--temperature N", "Sampling temperature (default 0.8)") { |v| temperature = v.to_f64 }
  parser.on("--top-k N", "Sampling top-k (default 30)") { |v| top_k = v.to_i }
  parser.on("--top-p N", "Sampling top-p (default 0.9)") { |v| top_p = v.to_f64 }
  parser.on("-h", "--help", "Show usage") { puts parser; exit }
end
abort "--model and --text are required" if model_path.empty? || text.strip.empty?
abort "--reference-audio and --reference-text are required together" if reference_audio.empty? != reference_text.empty?
abort "max-frames must be in 1..4096 or auto" unless auto_frames || max_frames.in?(1..4096)
abort "frame-cap must be in 1..4096" unless frame_cap.in?(1..4096)
abort "--frame-cap and --memory-plan-only require --max-frames auto" if !auto_frames && (frame_cap_given || memory_plan_only)
abort "output already exists: #{output}" if File.exists?(output) || File.exists?(output + ".json")
ML::Audio::Drama3::Profile.configure_from_env!
metal = ML::Audio::Drama3::MetalBackend.enabled?
has_reference = !reference_audio.empty?
abort "reference audio requires resident Metal" if has_reference && !metal
abort "automatic frame budgeting requires resident Metal" if auto_frames && !metal
abort "automatic budgeting for the reference encoder is not yet calibrated" if auto_frames && has_reference
abort "CPU max-frames must be in 1..1024" if !metal && max_frames > 1024
reference_input = has_reference ? ML::Audio::Drama3::Wav.read_reference(reference_audio) : nil
reference_sha256 = reference_input.try(&.sha256)
config = JSON.parse(File.read(File.join(model_path, "config.json")))
abort "only fish_omni at 44100 Hz is supported" unless config["model_type"].as_s == "fish_omni" && config["sample_rate"].as_i == 44100
abort "codec.safetensors is required" unless File.file?(File.join(model_path, "codec.safetensors"))
tower_config = ML::Audio::Drama3::TowerConfig.from_json(config.to_json)
abort "initial DAC slice requires 10 codebooks" unless tower_config.num_codebooks == 10
tokenizer = ML::Audio::Drama3::Tokenizer.new(File.join(model_path, "tokenizer.json"), config["eos_token_id"].as_i, config["pad_token_id"].as_i)
{"<|semantic:0|>"                              => config["semantic_start_token_id"].as_i,
 "<|semantic:#{tower_config.vocab_size - 1}|>" => config["semantic_end_token_id"].as_i,
 "<|im_end|>"                                  => config["eos_token_id"].as_i}.each do |piece, id|
  abort "tokenizer/config mismatch for #{piece}" unless tokenizer.token_id(piece) == id
end
reference_plan = if input = reference_input
                   ML::Audio::Drama3::Prompt.reference_plan(text, reference_text, input.frames, tokenizer,
                     tower_config.num_codebooks, config["semantic_start_token_id"].as_i, tower_config.vocab_size)
                 end
prompt = reference_plan ? [] of Array(Int32) : ML::Audio::Drama3::Prompt.build(text, tokenizer, tower_config.num_codebooks)
prompt_rows = reference_plan.try(&.row_count) || prompt.size
abort "prompt plus frame bound exceeds the 4096-row resident cache" if metal && !auto_frames && prompt_rows.to_i64 + max_frames - 1 > 4096
store = ML::Audio::Drama3::WeightStore.new(model_path)
window_plan : JSON::Any? = nil
resident_memory = {} of String => JSON::Any
resident_allocation_stats : JSON::Any? = nil
resident_dac_scratch : JSON::Any? = nil
{% if flag?(:drama3_metal) && flag?(:darwin) && !flag?(:cpu_only) %}
  if metal
    snapshot = ML::Audio::Drama3::WindowMemory.snapshot
    resident_memory["before_model"] = JSON.parse(snapshot.to_json)
    if auto_frames
      setting = ENV["DRAMA3_DAC_SCRATCH"]?
      abort "automatic budgeting requires DRAMA3_DAC_SCRATCH=1 (or unset)" if setting && setting != "1"
      ENV["DRAMA3_DAC_SCRATCH"] = "1"
      inventory = store.resident_memory_inventory
      telemetry = ML::Audio::Drama3::Drama3WindowBudget::Telemetry.new(
        physical_bytes: snapshot[:valid] ? snapshot[:physical_bytes] : nil,
        free_reclaimable_bytes: snapshot[:valid] ? snapshot[:free_reclaimable_bytes] : nil,
        metal_recommended_bytes: snapshot[:metal_recommended_bytes],
        metal_current_allocated_bytes: snapshot[:metal_allocated_bytes],
        max_buffer_length: ML::Audio::Drama3::MetalBackend.max_buffer_length,
      )
      request = ML::Audio::Drama3::Drama3WindowBudget::Request.new(
        backbone_config: ML::Audio::Drama3::BackboneConfig.new(config["text_config"]),
        tower_config: tower_config, codec_config: ML::Audio::Drama3::CodecConfig.default,
        prompt_frames: prompt_rows, min_frames: 1, frame_cap: frame_cap,
        checkpoint_mapped_bytes: inventory[:mapped_bytes],
        largest_checkpoint_mapped_buffer_bytes: inventory[:largest_checkpoint_mapped_buffer_bytes],
        converted_static_bytes: inventory[:converted_static_bytes],
        largest_converted_buffer_bytes: inventory[:largest_converted_buffer_bytes],
        additional_margin_bytes: 1024_i64 * 1024 * 1024,
        moe_intermediate_size: config["text_config"]["moe_intermediate_size"]?.try(&.as_i),
        shared_expert_intermediate_size: config["text_config"]["shared_expert_intermediate_size"]?.try(&.as_i),
      )
      result = ML::Audio::Drama3::Drama3WindowBudget.experimental_plan(request, telemetry)
      unless plan = result.plan?
        puts({mode: "experimental_auto", admitted: false, failure: result.failure.to_s,
              inventory: inventory, memory: snapshot, reserved_percent: 15,
              scope: "Advisory estimate; no memory was reserved and no model inference ran."}.to_json)
        abort "automatic frame budget rejected: #{result.failure}"
      end
      max_frames = plan.generated_frames
      window_plan = JSON.parse(plan.to_json)
      if memory_plan_only
        puts({mode: "experimental_auto", admitted: true, plan: window_plan, memory: snapshot,
              scope: "Advisory allocation estimate; shared-host load and GC retention can change the actual peak."}.to_json)
        exit
      end
      STDERR.puts "experimental auto budget: #{max_frames} frames, #{plan.total_positions} positions; allocation estimate, not a reservation"
    end
  end
{% end %}
started = Time.instant
progress = ->(message : String) { STDERR.puts "#{message} elapsed=#{(Time.instant - started).total_seconds.round(2)}s"; nil }
resident_readbacks = {} of String => Int64
resident_submissions = 0_i64
resident_graph = {} of String => Int64
resident_sampler_inputs : NamedTuple(dtype: String, uniform_count: Int32, sha256: String, provenance: String)? = nil
resident_dense_reduction : String? = nil
resident_backbone_dense_reduction : String? = nil
resident_expert_down_reduction : String? = nil
initialize_seconds = 0.0
generation_seconds = 0.0
codec_seconds = 0.0
reference_enqueue_seconds = 0.0
resident = false
generation, samples = {% if flag?(:drama3_metal) && flag?(:darwin) && !flag?(:cpu_only) %}
                        if metal
                          resident = true
                          context = ML::Audio::Drama3::Resident::Context.new
                          generator = ML::Audio::Drama3::ResidentGenerator.new(config, store, context, prompt_rows + max_frames - 1)
                          resident_dense_reduction = context.dense_reduction
                          resident_backbone_dense_reduction = generator.backbone.dense_reduction
                          resident_expert_down_reduction = generator.backbone.expert_down_reduction
                          generator.progress = progress
                          generation_started = Time.instant
                          initialize_seconds = (generation_started - started).total_seconds
                          result = if plan = reference_plan
                                     reference_started = Time.instant
                                     audio = ML::Audio::Drama3::ReferenceAudio.upload(reference_input.not_nil!, context)
                                     encoder = ML::Audio::Drama3::CodecEncoderResident.new(store, context)
                                     encoded = encoder.encode(audio, reference_input.not_nil!.samples)
                                     abort "reference encoder frame count changed" unless encoded.frames == plan.audio_row_count
                                     device_prompt = ML::Audio::Drama3::PromptResident.build(plan, encoded.codes, context)
                                     # No fence or code read between encoding and prefill. This
                                     # duration measures host enqueue time; execution also falls
                                     # within the surrounding generation wall-clock interval.
                                     reference_enqueue_seconds = (Time.instant - reference_started).total_seconds
                                     raise "reference handoff introduced a host read" unless context.readbacks.empty?
                                     generator.generate(device_prompt, max_frames, seed, temperature, top_k, top_p)
                                   else
                                     generator.generate(prompt, max_frames, seed, temperature, top_k, top_p)
                                   end
                          generation_seconds = (Time.instant - generation_started).total_seconds
                          resident_memory["after_generation"] = JSON.parse(ML::Audio::Drama3::WindowMemory.snapshot.to_json)
                          abort "model produced EOS before any audio frame" if result.generation.frames.empty?
                          STDERR.puts "decoding #{result.generation.frames.size} frames with resident Metal DAC"
                          codec_started = Time.instant
                          waveform = ML::Audio::Drama3::Profile.measure("codec.decode", category: "codec") do
                            codec = ML::Audio::Drama3::CodecResident.new(store, context)
                            frame_count = result.generation.frames.size
                            output_buffer = codec.decode(result.deviceframes.slice(0, frame_count * (tower_config.num_codebooks + 1)), frame_count)
                            resident_dac_scratch = JSON.parse(codec.dac_scratch_stats.to_json)
                            resident_memory["after_decode"] = JSON.parse(ML::Audio::Drama3::WindowMemory.snapshot.to_json)
                            context.read(output_buffer, "terminal")
                          end
                          codec_seconds = (Time.instant - codec_started).total_seconds
                          resident_readbacks = context.readbacks
                          resident_submissions = context.submissions
                          resident_graph = context.graph_stats
                          resident_sampler_inputs = generator.sampler_inputs
                          resident_allocation_stats = JSON.parse(context.allocation_stats.to_json)
                          {result.generation, waveform}
                        else
                          generator = ML::Audio::Drama3::Generator.new(config, store)
                          generator.progress = progress
                          generation_started = Time.instant
                          initialize_seconds = (generation_started - started).total_seconds
                          generated = generator.generate(prompt, max_frames, seed, temperature, top_k, top_p)
                          generation_seconds = (Time.instant - generation_started).total_seconds
                          abort "model produced EOS before any audio frame" if generated.frames.empty?
                          STDERR.puts "decoding #{generated.frames.size} frames with native DAC"
                          codec_started = Time.instant
                          waveform = ML::Audio::Drama3::Codec.new(store).decode(generated.codebooks)
                          codec_seconds = (Time.instant - codec_started).total_seconds
                          {generated, waveform}
                        end
                      {% else %}
                        generator = ML::Audio::Drama3::Generator.new(config, store)
                        generator.progress = progress
                        generation_started = Time.instant
                        initialize_seconds = (generation_started - started).total_seconds
                        generated = generator.generate(prompt, max_frames, seed, temperature, top_k, top_p)
                        generation_seconds = (Time.instant - generation_started).total_seconds
                        abort "model produced EOS before any audio frame" if generated.frames.empty?
                        STDERR.puts "decoding #{generated.frames.size} frames with native DAC"
                        codec_started = Time.instant
                        waveform = ML::Audio::Drama3::Codec.new(store).decode(generated.codebooks)
                        codec_seconds = (Time.instant - codec_started).total_seconds
                        {generated, waveform}
                      {% end %}
peak = samples.max_of(&.abs)
rms = Math.sqrt(samples.sum { |v| v.to_f64 * v } / samples.size)
abort "waveform is silent" unless peak > 1e-6 && rms > 1e-7
wav_started = Time.instant
ML::Audio::Drama3::Profile.measure("wav.write", category: "io") do
  ML::Audio::Drama3::Wav.write(output, samples)
end
wav_seconds = (Time.instant - wav_started).total_seconds
report = {
  backend: resident ? "metal_resident" : "cpu",
  resident_readbacks: resident_readbacks, resident_submissions: resident_submissions,
  resident_graph: resident_graph, sampler_inputs: resident_sampler_inputs,
  resident_dense_reduction: resident_dense_reduction,
  resident_backbone_dense_reduction: resident_backbone_dense_reduction,
  resident_expert_down_reduction: resident_expert_down_reduction,
  window_mode: auto_frames ? "experimental_auto" : "manual", window_plan: window_plan,
  resident_memory: resident_memory, resident_allocation_stats: resident_allocation_stats,
  resident_dac_scratch: resident_dac_scratch,
  model_path: model_path, text: text, seed: seed, max_frames: max_frames, temperature: temperature, top_k: top_k, top_p: top_p,
  reference: has_reference ? {audio: reference_audio, audio_sha256: reference_sha256,
                              text: reference_text, text_sha256: Digest::SHA256.hexdigest(reference_text),
                              sample_rate: reference_input.not_nil!.sample_rate, samples: reference_input.not_nil!.samples,
                              frames: reference_input.not_nil!.frames, conditioning: "resident GPU DAC encoder and prompt",
                              audio_row_start: reference_plan.not_nil!.audio_row_start,
                              host_code_read_before_prefill: false} : nil,
  prompt_rows: prompt_rows, prompt_ids: has_reference ? nil : prompt.map(&.[0]), frames: generation.frames,
  stopped_on_eos: generation.stopped, truncated: !generation.stopped,
  sample_rate: 44100, samples: samples.size, seconds: samples.size / 44100.0,
  peak: peak, rms: rms, elapsed_seconds: (Time.instant - started).total_seconds,
  phase_seconds: {model_initialize: initialize_seconds, generation: generation_seconds,
                  reference_host_enqueue: reference_enqueue_seconds, codec: codec_seconds, wav_write: wav_seconds},
}
File.write(output + ".json", report.to_pretty_json)
ML::Audio::Drama3::Profile.write_from_env
STDERR.puts "frame bound reached; WAV is a truncated prefix" unless generation.stopped
puts({output: output, frames: generation.frames.size, stopped_on_eos: generation.stopped, samples: samples.size, peak: peak, rms: rms}.to_json)
