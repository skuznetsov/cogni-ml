require "../src/ml/gguf/gemma4_layer_experiment"

begin
  exit ML::GGUF::Gemma4LayerExperiment.run(ARGV)
rescue ex
  STDERR.puts "Gemma 4 layer experiment: #{ex.message}"
  exit 2
end
