require "../src/ml/gguf/gemma4_energy_cli"

begin
  exit ML::GGUF::Gemma4EnergyCLI.run(ARGV)
rescue ex
  STDERR.puts "gemma4_energy: #{ex.message}"
  exit 1
end
