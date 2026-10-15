require "option_parser"
require "../src/ml/audio/drama3/generator"
require "../src/ml/audio/drama3/codec"
require "../src/ml/audio/drama3/wav"
require "../src/ml/audio/drama3/generator_resident"
require "../src/ml/audio/drama3/codec_resident"

# Offline native inference entry point. MLX/Python are reference tools only.
model_path = ""
text = ""
output = "drama3.wav"
max_frames = 160
seed = 0_u64
temperature = 0.8
top_k = 30
top_p = 0.9
OptionParser.parse do |parser|
  parser.banner = "usage: drama3_generate --model DIR --text TEXT [--output FILE.wav]"
  parser.on("--model DIR", "Pinned MLX-converted model directory") { |v| model_path = File.expand_path(v) }
  parser.on("--text TEXT", "Short single-speaker text") { |v| text = v }
  parser.on("--output FILE", "PCM16 mono WAV destination") { |v| output = File.expand_path(v) }
  parser.on("--max-frames N", "Bound generation (default 160)") { |v| max_frames = v.to_i }
  parser.on("--seed N", "Native sampling seed") { |v| seed = v.to_u64 }
  parser.on("--temperature N", "Sampling temperature (default 0.8)") { |v| temperature = v.to_f64 }
  parser.on("--top-k N", "Sampling top-k (default 30)") { |v| top_k = v.to_i }
  parser.on("--top-p N", "Sampling top-p (default 0.9)") { |v| top_p = v.to_f64 }
  parser.on("-h", "--help", "Show usage") { puts parser; exit }
end
abort "--model and --text are required" if model_path.empty? || text.strip.empty?
abort "max-frames must be in 1..1024" unless max_frames.in?(1..1024)
abort "output already exists: #{output}" if File.exists?(output) || File.exists?(output + ".json")
ML::Audio::Drama3::Profile.configure_from_env!
metal = ML::Audio::Drama3::MetalBackend.enabled?
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
prompt = ML::Audio::Drama3::Prompt.build(text, tokenizer, tower_config.num_codebooks)
abort "prompt plus frame bound exceeds the 1024-row resident cache" if metal && prompt.size + max_frames - 1 > 1024
store = ML::Audio::Drama3::WeightStore.new(model_path)
started = Time.instant
progress = ->(message : String) { STDERR.puts "#{message} elapsed=#{(Time.instant - started).total_seconds.round(2)}s"; nil }
resident_readbacks = {} of String => Int64
resident_submissions = 0_i64
resident_graph = {} of String => Int64
resident_dense_reduction : String? = nil
resident_backbone_dense_reduction : String? = nil
resident_expert_down_reduction : String? = nil
initialize_seconds = 0.0
generation_seconds = 0.0
codec_seconds = 0.0
resident = false
generation, samples = {% if flag?(:drama3_metal) && flag?(:darwin) && !flag?(:cpu_only) %}
                        if metal
                          resident = true
                          context = ML::Audio::Drama3::Resident::Context.new
                          generator = ML::Audio::Drama3::ResidentGenerator.new(config, store, context, prompt.size + max_frames - 1)
                          resident_dense_reduction = context.dense_reduction
                          resident_backbone_dense_reduction = generator.backbone.dense_reduction
                          resident_expert_down_reduction = generator.backbone.expert_down_reduction
                          generator.progress = progress
                          generation_started = Time.instant
                          initialize_seconds = (generation_started - started).total_seconds
                          result = generator.generate(prompt, max_frames, seed, temperature, top_k, top_p)
                          generation_seconds = (Time.instant - generation_started).total_seconds
                          abort "model produced EOS before any audio frame" if result.generation.frames.empty?
                          STDERR.puts "decoding #{result.generation.frames.size} frames with resident Metal DAC"
                          codec_started = Time.instant
                          waveform = ML::Audio::Drama3::Profile.measure("codec.decode", category: "codec") do
                            codec = ML::Audio::Drama3::CodecResident.new(store, context)
                            frame_count = result.generation.frames.size
                            output_buffer = codec.decode(result.deviceframes.slice(0, frame_count * (tower_config.num_codebooks + 1)), frame_count)
                            context.read(output_buffer, "terminal")
                          end
                          codec_seconds = (Time.instant - codec_started).total_seconds
                          resident_readbacks = context.readbacks
                          resident_submissions = context.submissions
                          resident_graph = context.graph_stats
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
  resident_graph: resident_graph,
  resident_dense_reduction: resident_dense_reduction,
  resident_backbone_dense_reduction: resident_backbone_dense_reduction,
  resident_expert_down_reduction: resident_expert_down_reduction,
  model_path: model_path, text: text, seed: seed, temperature: temperature, top_k: top_k, top_p: top_p,
  prompt_rows: prompt.size, prompt_ids: prompt.map(&.[0]), frames: generation.frames,
  stopped_on_eos: generation.stopped, truncated: !generation.stopped,
  sample_rate: 44100, samples: samples.size, seconds: samples.size / 44100.0,
  peak: peak, rms: rms, elapsed_seconds: (Time.instant - started).total_seconds,
  phase_seconds: {model_initialize: initialize_seconds, generation: generation_seconds, codec: codec_seconds, wav_write: wav_seconds},
}
File.write(output + ".json", report.to_pretty_json)
ML::Audio::Drama3::Profile.write_from_env
STDERR.puts "frame bound reached; WAV is a truncated prefix" unless generation.stopped
puts({output: output, frames: generation.frames.size, stopped_on_eos: generation.stopped, samples: samples.size, peak: peak, rms: rms}.to_json)
