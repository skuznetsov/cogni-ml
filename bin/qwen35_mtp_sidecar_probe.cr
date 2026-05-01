#!/usr/bin/env crystal

require "option_parser"
require "../src/ml/gguf/qwen35_meta"
require "../src/ml/gguf/qwen35_mtp"
require "../src/ml/gguf/reader"

DEFAULT_MODEL = "#{ENV["HOME"]}/.cache/lm-studio/models/lmstudio-community/Qwen3.6-27B-GGUF/Qwen3.6-27B-Q4_K_M.gguf"
DEFAULT_MTP   = "#{ENV["HOME"]}/.cache/cogni-ml/qwen36_mtp/Qwen3.6-27B-mtp.safetensors"

model_path = DEFAULT_MODEL
mtp_path = DEFAULT_MTP

OptionParser.parse do |p|
  p.banner = "Usage: qwen35_mtp_sidecar_probe [--model GGUF] [--mtp SIDE_SAFETENSORS]"
  p.on("--model PATH", "Qwen3.6 GGUF target model path") { |v| model_path = v }
  p.on("--mtp PATH", "MTP-only safetensors sidecar path") { |v| mtp_path = v }
  p.on("-h", "--help", "Show help") do
    puts p
    exit
  end
end

abort "model not found: #{model_path}" unless File.exists?(model_path)
abort "MTP sidecar not found: #{mtp_path}" unless File.exists?(mtp_path)

gguf = ML::GGUF::GGUFFile.new(model_path)
hparams = ML::GGUF::Qwen35Hparams.new(gguf)
mtp = ML::GGUF::Qwen35MTPWeights.from_safetensors(mtp_path)
mtp.validate_for_qwen35!(hparams)

puts "qwen35_mtp_sidecar_probe: ok"
puts "model=#{model_path}"
puts "mtp=#{mtp_path}"
puts "hparams hidden=#{hparams.n_embd} layers=#{hparams.n_layer} heads=#{hparams.n_head} kv_heads=#{hparams.n_head_kv} head_dim=#{hparams.head_dim} ffn=#{hparams.n_ff}"
puts "mtp_bytes=#{(mtp.total_raw_bytes / 1_048_576.0).round(2)} MiB"
puts "fc=#{mtp.fc.out_dim}x#{mtp.fc.in_dim}"
puts "attn q=#{mtp.q_proj.out_dim}x#{mtp.q_proj.in_dim} k=#{mtp.k_proj.out_dim}x#{mtp.k_proj.in_dim} v=#{mtp.v_proj.out_dim}x#{mtp.v_proj.in_dim} o=#{mtp.o_proj.out_dim}x#{mtp.o_proj.in_dim}"
puts "ffn gate=#{mtp.ffn_gate.out_dim}x#{mtp.ffn_gate.in_dim} up=#{mtp.ffn_up.out_dim}x#{mtp.ffn_up.in_dim} down=#{mtp.ffn_down.out_dim}x#{mtp.ffn_down.in_dim}"
