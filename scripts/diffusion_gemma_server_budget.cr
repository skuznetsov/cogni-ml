require "option_parser"
require "../src/ml/gguf/diffusion_gemma_meta"

DEFAULT_MODEL = "#{ENV["HOME"]}/.cache/lm-studio/models/unsloth/diffusiongemma-26B-A4B-it-GGUF/diffusiongemma-26B-A4B-it-Q4_K_M.gguf"

model = ENV["DIFFUSION_GEMMA_MODEL"]? || DEFAULT_MODEL
steps = 10

OptionParser.parse do |p|
  p.banner = "Usage: diffusion_gemma_server_budget [--model PATH] [--steps N]"
  p.on("--model PATH", "DiffusionGemma GGUF path") { |v| model = v }
  p.on("--steps N", "Denoise steps to estimate (default: 10)") { |v| steps = v.to_i }
  p.on("-h", "--help", "Show help") do
    puts p
    exit
  end
end

raise "--steps must be positive" unless steps > 0

def fmt_bytes(bytes : Int64) : String
  units = {"B", "KiB", "MiB", "GiB", "TiB"}
  value = bytes.to_f64
  unit = 0
  while value >= 1024.0 && unit < units.size - 1
    value /= 1024.0
    unit += 1
  end
  "#{value.round(2)} #{units[unit]}"
end

g = ML::GGUF::GGUFFile.new(model)
begin
  hp = ML::GGUF::DiffusionGemmaHparams.new(g)
  response_bytes = hp.canvas_length.to_i64 * hp.vocab_size.to_i64 * 4_i64
  total_bytes = response_bytes * steps

  puts "model=#{model}"
  puts "canvas=#{hp.canvas_length}"
  puts "vocab=#{hp.vocab_size}"
  puts "steps=#{steps}"
  puts "server_response_bytes_per_step=#{response_bytes}"
  puts "server_response_per_step=#{fmt_bytes(response_bytes)}"
  puts "server_response_total_bytes=#{total_bytes}"
  puts "server_response_total=#{fmt_bytes(total_bytes)}"
  puts "decision=full_logits_file_protocol_is_not_a_speed_path_without_server_side_sampling"
ensure
  g.close
end
