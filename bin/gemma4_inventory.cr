require "option_parser"
require "../src/ml/gguf/gemma4_meta"

DEFAULT_MODEL = "#{ENV["HOME"]}/.cache/lm-studio/models/lmstudio-community/gemma-4-12B-it-GGUF/gemma-4-12B-it-Q4_K_M.gguf"

model = ENV["GEMMA4_MODEL"]? || DEFAULT_MODEL
show_tensors = false

OptionParser.parse do |p|
  p.banner = "Usage: gemma4_inventory [--model PATH] [--tensors]"
  p.on("--model PATH", "Gemma4 GGUF path") { |v| model = v }
  p.on("--tensors", "Print compact tensor type/shape histogram") { show_tensors = true }
  p.on("-h", "--help", "Show help") do
    puts p
    exit
  end
end

g = ML::GGUF::GGUFFile.new(model)
begin
  hp = ML::GGUF::Gemma4Hparams.new(g)

  puts "model=#{model}"
  puts "arch=#{hp.arch}"
  puts "layers=#{hp.n_layer} embd=#{hp.n_embd} ffn=#{hp.n_ff} vocab=#{hp.vocab_size} ctx=#{hp.context_length}"
  puts "heads=#{hp.n_head} head_dim_full=#{hp.head_dim} head_dim_swa=#{hp.head_dim_swa}"
  puts "rope_dim_full=#{hp.rope_dim_count} rope_dim_swa=#{hp.rope_dim_count_swa}"
  puts "rope_base_full=#{hp.rope_freq_base} rope_base_swa=#{hp.rope_freq_base_swa}"
  puts "rms_eps=#{hp.rms_eps} final_logit_softcap=#{hp.final_logit_softcapping}"
  puts "sliding_window=#{hp.sliding_window} shared_kv_layers=#{hp.shared_kv_layers}"
  puts "swa_layers=#{hp.sliding_window_layers.join(",")}"
  puts "full_layers=#{hp.full_attention_layers.join(",")}"

  kv_groups = hp.n_head_kv_by_layer.group_by { |v| v }.transform_values(&.size)
  puts "n_head_kv_hist=#{kv_groups.map { |k, v| "#{k}:#{v}" }.join(",")}"

  first_shapes = (0...[hp.n_layer, 12].min).map do |il|
    kind = hp.sliding_window?(il) ? "swa" : "full"
    "L#{il}:#{kind}:kv#{hp.n_head_kv(il)}:hd#{hp.head_dim_for_layer(il)}"
  end
  puts "first_layer_shapes=#{first_shapes.join(" ")}"

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
