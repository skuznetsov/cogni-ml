require "../src/ml/gguf/reader"
require "../src/ml/gguf/qwen35_meta"
require "../src/ml/gguf/qwen35_weights"

DEFAULT_MODEL_PATH = "#{ENV["HOME"]}/.cache/lm-studio/models/lmstudio-community/Qwen3.5-9B-GGUF/Qwen3.5-9B-Q4_K_M.gguf"

def usage(io : IO) : Nil
  io << "Usage: qwen35_gguf_info [--model PATH] [--load-weights] [--list-tensors]\n"
  io << "\n"
  io << "CPU-only Qwen 3.5/3.6 GGUF metadata smoke. Build with -Dcpu_only on Linux/CUDA hosts.\n"
  io << "\n"
  io << "Options:\n"
  io << "  --model PATH     GGUF file path. Defaults to QWEN35_MODEL or the 9B LM Studio path.\n"
  io << "  --load-weights   Also instantiate Qwen35Weights without Metal registration.\n"
  io << "  --list-tensors   Print tensor inventory lines after the summary.\n"
  io << "  -h, --help       Show this help.\n"
end

model_path = ENV["QWEN35_MODEL"]? || DEFAULT_MODEL_PATH
load_weights = false
list_tensors = false

args = ARGV.dup
until args.empty?
  arg = args.shift
  case arg
  when "--model"
    model_path = args.shift? || raise ArgumentError.new("--model requires a path")
  when "--load-weights"
    load_weights = true
  when "--list-tensors"
    list_tensors = true
  when "-h", "--help"
    usage(STDOUT)
    exit 0
  else
    if arg.starts_with?("-")
      raise ArgumentError.new("unknown option: #{arg}")
    else
      model_path = arg
    end
  end
end

unless File.exists?(model_path)
  STDERR.puts "model not found: #{model_path}"
  STDERR.puts "set QWEN35_MODEL or pass --model PATH"
  exit 2
end

gguf = ML::GGUF::GGUFFile.new(model_path)
hparams = ML::GGUF::Qwen35Hparams.new(gguf)

puts "model=#{model_path}"
puts "arch=#{hparams.arch}"
puts "layers=#{hparams.n_layer}"
puts "embedding_length=#{hparams.n_embd}"
puts "feed_forward_length=#{hparams.n_ff}"
puts "context_length=#{hparams.context_length}"
puts "heads=#{hparams.n_head}"
puts "kv_heads=#{hparams.n_head_kv}"
puts "head_dim=#{hparams.head_dim}"
puts "full_attention_interval=#{hparams.full_attention_interval}"
puts "full_attention_layers=#{hparams.full_attention_layers.join(",")}"
puts "recurrent_layers=#{hparams.recurrent_layers.size}"
puts "ssm_state_size=#{hparams.ssm_state_size}"
puts "ssm_group_count=#{hparams.ssm_group_count}"
puts "ssm_time_step_rank=#{hparams.ssm_time_step_rank}"
puts "ssm_inner_size=#{hparams.ssm_inner_size}"
puts "metadata=#{gguf.metadata.size}"
puts "tensors=#{gguf.tensors.size}"
puts "file_bytes=#{File.size(model_path)}"

type_counts = Hash(String, Int32).new(0)
total_tensor_bytes = 0_i64
gguf.tensors.each do |tensor|
  type_counts[tensor.type.name] += 1
  total_tensor_bytes += tensor.data_bytes
end
puts "tensor_bytes=#{total_tensor_bytes}"
puts "tensor_types=#{type_counts.keys.sort.map { |key| "#{key}:#{type_counts[key]}" }.join(",")}"

if load_weights
  weights = ML::GGUF::Qwen35Weights.new(gguf, hparams)
  full_count = weights.layers.count(&.is_a?(ML::GGUF::Qwen35FullAttnWeights))
  recurrent_count = weights.layers.count(&.is_a?(ML::GGUF::Qwen35RecurrentWeights))
  puts "weights=loaded"
  puts "weight_layers=#{weights.layers.size}"
  puts "weight_full_attention_layers=#{full_count}"
  puts "weight_recurrent_layers=#{recurrent_count}"
  puts "token_embd=#{weights.token_embd.out_dim}x#{weights.token_embd.in_dim}:#{weights.token_embd.type.name}"
  puts "output=#{weights.output.out_dim}x#{weights.output.in_dim}:#{weights.output.type.name}"
else
  puts "weights=not_loaded"
  gguf.close
end

if list_tensors
  gguf.tensors.each do |tensor|
    dims = tensor.dims.join("x")
    puts "tensor=#{tensor.name}\tdims=#{dims}\ttype=#{tensor.type.name}\tbytes=#{tensor.data_bytes}"
  end
end
