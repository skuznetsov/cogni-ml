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

struct AveragedLayerEmbedProfile
  getter seq_len : Int32
  getter tokenize_ms : Float64
  getter prepare_ms : Float64
  getter prepass_wait_ms : Float64
  getter pool_wait_ms : Float64
  getter readback_ms : Float64
  getter layers : Array(ML::GGUF::LayerStageProfile)

  def initialize(@seq_len, @tokenize_ms, @prepare_ms, @prepass_wait_ms, @pool_wait_ms, @readback_ms, @layers)
  end

  def layer_wait_ms : Float64
    @layers.sum(&.total_ms)
  end

  def total_ms : Float64
    @tokenize_ms + @prepare_ms + @prepass_wait_ms + layer_wait_ms + @pool_wait_ms + @readback_ms
  end

  def self.from_profiles(profiles : Array(ML::GGUF::LayerEmbedProfile)) : AveragedLayerEmbedProfile
    raise "no profiles" if profiles.empty?
    n = profiles.size.to_f64
    base_layers = profiles.first.layers
    avg_layers = Array(ML::GGUF::LayerStageProfile).new(base_layers.size)
    base_layers.each_with_index do |base, idx|
      avg_layers << ML::GGUF::LayerStageProfile.new(
        layer_index: base.layer_index,
        kind: base.kind,
        attn_proj_ms: profiles.sum { |p| p.layers[idx].attn_proj_ms } / n,
        attn_core_ms: profiles.sum { |p| p.layers[idx].attn_core_ms } / n,
        attn_out_norm_ms: profiles.sum { |p| p.layers[idx].attn_out_norm_ms } / n,
        ffn_route_ms: profiles.sum { |p| p.layers[idx].ffn_route_ms } / n,
        ffn_up_ms: profiles.sum { |p| p.layers[idx].ffn_up_ms } / n,
        ffn_down_ms: profiles.sum { |p| p.layers[idx].ffn_down_ms } / n,
        ffn_scatter_norm_ms: profiles.sum { |p| p.layers[idx].ffn_scatter_norm_ms } / n,
      )
    end

    AveragedLayerEmbedProfile.new(
      profiles.first.seq_len,
      profiles.sum(&.tokenize_ms) / n,
      profiles.sum(&.prepare_ms) / n,
      profiles.sum(&.prepass_wait_ms) / n,
      profiles.sum(&.pool_wait_ms) / n,
      profiles.sum(&.readback_ms) / n,
      avg_layers,
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
layers_case = "all"

OptionParser.parse do |p|
  p.banner = "Usage: profile_nomic_stages [options]"
  p.on("--mode=MODE", "single | batch | layers | all (default: all)") { |v| mode = v }
  p.on("--runs=N", "Number of measured runs per case (default: 5)") { |v| runs = v.to_i }
  p.on("--warmup=N", "Warmup runs before measuring (default: 3)") { |v| warmup = v.to_i }
  p.on("--model=PATH", "Path to GGUF model") { |v| model_path = v }
  p.on("--layers-case=LABEL", "short | medium | long | all (default: all)") { |v| layers_case = v }
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
    warmup.times { model.profile_embed(text) }
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

  warmup.times { model.profile_embed_batch(BATCH_TEXTS) }
  profiles = Array(ML::GGUF::EmbedProfile).new(runs)
  runs.times { profiles << model.profile_embed_batch(BATCH_TEXTS) }
  avg = AveragedProfile.from_profiles(profiles)
  STDERR.puts "#{avg.text_count.to_s.rjust(5)} #{avg.total_tokens.to_s.rjust(5)} #{avg.max_seq_len.to_s.rjust(5)} #{fmt(avg.tokenize_ms)} #{fmt(avg.prepare_ms)} #{fmt(avg.reorder_ms)} #{fmt(avg.graph_lookup_ms)} #{fmt(avg.cmd_setup_ms)} #{fmt(avg.token_write_ms)} #{fmt(avg.lengths_write_ms)} #{fmt(avg.prepass_encode_ms)} #{fmt(avg.graph_encode_ms)} #{fmt(avg.submit_wait_ms)} #{fmt(avg.readback_ms)} #{fmt(avg.total_ms)}"
end

if mode == "layers" || mode == "all"
  cases = if layers_case == "all"
            SINGLE_TEXTS.to_a
          else
            text = SINGLE_TEXTS[layers_case]?
            raise "unknown layers case: #{layers_case}" unless text
            [{layers_case, text}]
          end

  cases.each do |label, text|
    warmup.times { model.profile_embed_layers(text) }
    profiles = Array(ML::GGUF::LayerEmbedProfile).new(runs)
    runs.times { profiles << model.profile_embed_layers(text) }
    avg = AveragedLayerEmbedProfile.from_profiles(profiles)

    STDERR.puts
    STDERR.puts "=== Layer Profile: #{label} (seq=#{avg.seq_len}) ==="
    STDERR.puts "tokenize=#{fmt(avg.tokenize_ms).strip}ms prepare=#{fmt(avg.prepare_ms).strip}ms prepass_wait=#{fmt(avg.prepass_wait_ms).strip}ms pool_wait=#{fmt(avg.pool_wait_ms).strip}ms read=#{fmt(avg.readback_ms).strip}ms total=#{fmt(avg.total_ms).strip}ms"
    STDERR.puts "#{"layer".rjust(5)} #{"kind".ljust(11)} #{"a_proj".rjust(8)} #{"a_core".rjust(8)} #{"a_out".rjust(8)} #{"route".rjust(8)} #{"ffn_up".rjust(8)} #{"ffn_dn".rjust(8)} #{"scatter".rjust(8)} #{"total".rjust(8)}"
    STDERR.puts "-" * 96

    avg.layers.each do |layer|
      STDERR.puts "#{layer.layer_index.to_s.rjust(5)} #{layer.kind.ljust(11)} #{fmt(layer.attn_proj_ms)} #{fmt(layer.attn_core_ms)} #{fmt(layer.attn_out_norm_ms)} #{fmt(layer.ffn_route_ms)} #{fmt(layer.ffn_up_ms)} #{fmt(layer.ffn_down_ms)} #{fmt(layer.ffn_scatter_norm_ms)} #{fmt(layer.total_ms)}"
    end

    attn_total = avg.layers.sum(&.attention_ms)
    dense_total = avg.layers.select { |l| l.kind == "dense" }.sum(&.ffn_ms)
    moe_total = avg.layers.reject { |l| l.kind == "dense" }.sum(&.ffn_ms)
    hot = avg.layers.max_by(&.total_ms)
    STDERR.puts "-" * 96
    STDERR.puts "attention_total=#{fmt(attn_total).strip}ms dense_ffn_total=#{fmt(dense_total).strip}ms moe_ffn_total=#{fmt(moe_total).strip}ms hottest=L#{hot.layer_index}(#{hot.kind}) #{fmt(hot.total_ms).strip}ms"
  end
end
