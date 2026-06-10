require "option_parser"
require "../src/ml/gguf/diffusion_gemma_meta"

DEFAULT_MODEL = "#{ENV["HOME"]}/.cache/lm-studio/models/unsloth/diffusiongemma-26B-A4B-it-GGUF/diffusiongemma-26B-A4B-it-Q4_K_M.gguf"

model = ENV["DIFFUSION_GEMMA_MODEL"]? || DEFAULT_MODEL
show_tensors = false

OptionParser.parse do |p|
  p.banner = "Usage: diffusion_gemma_inventory [--model PATH] [--tensors]"
  p.on("--model PATH", "DiffusionGemma GGUF path") { |v| model = v }
  p.on("--tensors", "Print compact tensor type/shape histogram") { show_tensors = true }
  p.on("-h", "--help", "Show help") do
    puts p
    exit
  end
end

g = ML::GGUF::GGUFFile.new(model)
begin
  hp = ML::GGUF::DiffusionGemmaHparams.new(g)

  puts "model=#{model}"
  puts "arch=#{hp.arch}"
  puts "layers=#{hp.n_layer} embd=#{hp.n_embd} ffn=#{hp.n_ff} expert_ff=#{hp.expert_ff} vocab=#{hp.vocab_size}"
  puts "ctx=#{hp.context_length} canvas=#{hp.canvas_length} causal_attention=#{hp.causal_attention}"
  puts "experts=#{hp.expert_count} experts_used=#{hp.expert_used_count}"
  puts "heads=#{hp.n_head} head_dim_full=#{hp.head_dim} head_dim_swa=#{hp.head_dim_swa}"
  puts "rope_dim_full=#{hp.rope_dim_count} rope_dim_swa=#{hp.rope_dim_count_swa}"
  puts "rope_base_full=#{hp.rope_freq_base} rope_base_swa=#{hp.rope_freq_base_swa}"
  puts "rms_eps=#{hp.rms_eps} final_logit_softcap=#{hp.final_logit_softcapping}"
  puts "sliding_window=#{hp.sliding_window} shared_kv_layers=#{hp.shared_kv_layers}"
  puts "swa_layers=#{hp.sliding_window_layers.join(",")}"
  puts "full_layers=#{hp.full_attention_layers.join(",")}"

  kv_groups = hp.n_head_kv_by_layer.group_by { |v| v }.transform_values(&.size)
  puts "n_head_kv_hist=#{kv_groups.map { |k, v| "#{k}:#{v}" }.join(",")}"

  if show_tensors
    hist = Hash(String, Int32).new(0)
    g.tensors.each do |t|
      shape = t.dims.join("x")
      hist["#{t.type.name}:#{shape}"] += 1
    end
    puts "tensor_shape_hist:"
    hist.to_a.sort_by { |kv| {-kv[1], kv[0]} }.each do |key, count|
      puts "  #{count} #{key}"
    end
  end
ensure
  g.close
end
