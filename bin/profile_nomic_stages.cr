require "option_parser"
require "../src/ml/gguf/metal_backend"
require "../src/ml/gguf/nomic_bert"

DEFAULT_MODEL = ENV["EMBED_MODEL"]? || (Path.home / ".cache/lm-studio/models/nomic-ai/nomic-embed-text-v2-moe-GGUF/nomic-embed-text-v2-moe.Q5_K_M.gguf").to_s

SINGLE_TEXTS = {
  "short"  => "def embed(text : String) : Array(Float32)",
  "medium" => "The Crystal programming language is a statically typed compiled language with syntax inspired by Ruby. It features type inference macros and generics making it both expressive and performant. Crystal compiles to native code via LLVM.",
  "long"   => "The Crystal programming language is a statically typed, compiled language with syntax inspired by Ruby. " \
    "It features type inference, macros, and generics, making it both expressive and performant. Crystal compiles " \
    "to native code via LLVM and achieves C-like performance while maintaining developer-friendly syntax. The " \
    "language supports concurrency through fibers and channels, similar to Go's goroutines. Crystal's type system " \
    "catches errors at compile time, eliminating many runtime bugs common in dynamic languages. The standard " \
    "library includes HTTP servers, JSON parsing, database drivers, and cryptographic primitives. Crystal also " \
    "supports C bindings through its lib declaration syntax, allowing direct FFI without wrapper libraries. The " \
    "compiler itself is written in Crystal, demonstrating the language's capability for systems programming.",
}

BATCH_TEXTS = [
  "def embed(text : String) : Array(Float32)",
  "class ComputeGraph\n  @ops : Array(Op)\n  @waves : Array(Array(Int32))?\nend",
  "kernel void simd_mm_q5k(device const uint8_t* w_raw, device const half* x, device const float* bias, device half* output)",
  "The Crystal programming language is a statically typed, compiled language with Ruby-like syntax. It features type inference, macros, and generics.",
  "PostgreSQL extension for vector search with pgvector and HNSW indexing enables fast approximate nearest neighbor queries on embedding vectors.",
  "fn main() {\n  let mut v = Vec::new();\n  for i in 0..100 {\n    v.push(i * i);\n  }\n  println!(\"{:?}\", v);\n}",
  "SELECT e.entity_id, e.payload, e.embedding <=> $1::vector AS distance FROM crystal_facts e ORDER BY distance LIMIT 10",
  "module ML::Metal\n  class Device\n    @@instance : Device?\n    def self.available? : Bool\n      instance.available?\n    end\n  end\nend",
]

struct AveragedProfile
  getter text_count : Int32
  getter total_tokens : Int32
  getter max_seq_len : Int32
  getter tokenize_ms : Float64
  getter prepare_ms : Float64
  getter reorder_ms : Float64
  getter graph_lookup_ms : Float64
  getter cmd_setup_ms : Float64
  getter token_write_ms : Float64
  getter lengths_write_ms : Float64
  getter prepass_encode_ms : Float64
  getter graph_encode_ms : Float64
  getter submit_wait_ms : Float64
  getter readback_ms : Float64
  getter total_ms : Float64

  def initialize(@text_count, @total_tokens, @max_seq_len, @tokenize_ms, @prepare_ms, @reorder_ms,
                 @graph_lookup_ms, @cmd_setup_ms, @token_write_ms, @lengths_write_ms,
                 @prepass_encode_ms, @graph_encode_ms, @submit_wait_ms, @readback_ms, @total_ms)
  end

  def self.from_profiles(profiles : Array(ML::GGUF::EmbedProfile)) : AveragedProfile
    raise "no profiles" if profiles.empty?
    n = profiles.size.to_f64
    AveragedProfile.new(
      profiles.first.text_count,
      (profiles.sum(&.total_tokens) / profiles.size).to_i32,
      (profiles.sum(&.max_seq_len) / profiles.size).to_i32,
      profiles.sum(&.tokenize_ms) / n,
      profiles.sum(&.prepare_ms) / n,
      profiles.sum(&.reorder_ms) / n,
      profiles.sum(&.backend.graph_lookup_ms) / n,
      profiles.sum(&.backend.cmd_setup_ms) / n,
      profiles.sum(&.backend.token_write_ms) / n,
      profiles.sum(&.backend.lengths_write_ms) / n,
      profiles.sum(&.backend.prepass_encode_ms) / n,
      profiles.sum(&.backend.graph_encode_ms) / n,
      profiles.sum(&.backend.submit_wait_ms) / n,
      profiles.sum(&.backend.readback_ms) / n,
      profiles.sum(&.total_ms) / n,
    )
  end
end

def fmt(ms : Float64) : String
  ms.round(2).to_s.rjust(8)
end

mode = "all"
runs = 5
warmup = 3
model_path = DEFAULT_MODEL

OptionParser.parse do |p|
  p.banner = "Usage: profile_nomic_stages [options]"
  p.on("--mode=MODE", "single | batch | all (default: all)") { |v| mode = v }
  p.on("--runs=N", "Number of measured runs per case (default: 5)") { |v| runs = v.to_i }
  p.on("--warmup=N", "Warmup runs before measuring (default: 3)") { |v| warmup = v.to_i }
  p.on("--model=PATH", "Path to GGUF model") { |v| model_path = v }
end

unless File.exists?(model_path)
  STDERR.puts "Model not found: #{model_path}"
  exit 1
end

ML::Metal::Device.init!
model = ML::GGUF::NomicBertMoE.from_gguf(model_path, ML::GGUF::MetalBackend.new)

warmup.times { model.embed("warmup") }

if mode == "single" || mode == "all"
  STDERR.puts "=== Single Text Stage Profile ==="
  STDERR.puts "#{"label".ljust(8)} #{"tok".rjust(5)} #{"tokenize".rjust(8)} #{"prepare".rjust(8)} #{"graph".rjust(8)} #{"cmd".rjust(8)} #{"tokwrite".rjust(8)} #{"prepass".rjust(8)} #{"encode".rjust(8)} #{"wait".rjust(8)} #{"read".rjust(8)} #{"total".rjust(8)}"
  STDERR.puts "-" * 108

  SINGLE_TEXTS.each do |label, text|
    profiles = Array(ML::GGUF::EmbedProfile).new(runs)
    runs.times { profiles << model.profile_embed(text) }
    avg = AveragedProfile.from_profiles(profiles)
    STDERR.puts "#{label.ljust(8)} #{avg.total_tokens.to_s.rjust(5)} #{fmt(avg.tokenize_ms)} #{fmt(avg.prepare_ms)} #{fmt(avg.graph_lookup_ms)} #{fmt(avg.cmd_setup_ms)} #{fmt(avg.token_write_ms)} #{fmt(avg.prepass_encode_ms)} #{fmt(avg.graph_encode_ms)} #{fmt(avg.submit_wait_ms)} #{fmt(avg.readback_ms)} #{fmt(avg.total_ms)}"
  end
  STDERR.puts
end

if mode == "batch" || mode == "all"
  STDERR.puts "=== Batch Stage Profile ==="
  STDERR.puts "#{"texts".rjust(5)} #{"tok".rjust(5)} #{"max".rjust(5)} #{"tokenize".rjust(8)} #{"prepare".rjust(8)} #{"reorder".rjust(8)} #{"graph".rjust(8)} #{"cmd".rjust(8)} #{"tokwrite".rjust(8)} #{"lenwrite".rjust(8)} #{"prepass".rjust(8)} #{"encode".rjust(8)} #{"wait".rjust(8)} #{"read".rjust(8)} #{"total".rjust(8)}"
  STDERR.puts "-" * 136

  profiles = Array(ML::GGUF::EmbedProfile).new(runs)
  runs.times { profiles << model.profile_embed_batch(BATCH_TEXTS) }
  avg = AveragedProfile.from_profiles(profiles)
  STDERR.puts "#{avg.text_count.to_s.rjust(5)} #{avg.total_tokens.to_s.rjust(5)} #{avg.max_seq_len.to_s.rjust(5)} #{fmt(avg.tokenize_ms)} #{fmt(avg.prepare_ms)} #{fmt(avg.reorder_ms)} #{fmt(avg.graph_lookup_ms)} #{fmt(avg.cmd_setup_ms)} #{fmt(avg.token_write_ms)} #{fmt(avg.lengths_write_ms)} #{fmt(avg.prepass_encode_ms)} #{fmt(avg.graph_encode_ms)} #{fmt(avg.submit_wait_ms)} #{fmt(avg.readback_ms)} #{fmt(avg.total_ms)}"
end
