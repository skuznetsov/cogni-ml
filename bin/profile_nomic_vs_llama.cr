require "option_parser"
require "../src/ml/gguf/metal_backend"
require "../src/ml/gguf/nomic_bert"
require "../src/ml/llm/llama"

DEFAULT_MODEL = ENV["EMBED_MODEL"]? || (Path.home / ".cache/lm-studio/models/nomic-ai/nomic-embed-text-v2-moe-GGUF/nomic-embed-text-v2-moe.Q5_K_M.gguf").to_s

TEXTS = {
  "short"  => "def embed(text : String) : Array(Float32)",
  "medium" => "The Crystal programming language is a statically typed compiled language with syntax inspired by Ruby. It features type inference macros and generics making it both expressive and performant. Crystal compiles to native code via LLVM.",
  "long"   => "The Crystal programming language is a statically typed, compiled language with syntax inspired by Ruby. It features type inference, macros, and generics, making it both expressive and performant. Crystal compiles to native code via LLVM and achieves C-like performance while maintaining developer-friendly syntax. The language supports concurrency through fibers and channels, similar to Go's goroutines. Crystal's type system catches errors at compile time, eliminating many runtime bugs common in dynamic languages. The standard library includes HTTP servers, JSON parsing, database drivers, and cryptographic primitives. Crystal also supports C bindings through its lib declaration syntax, allowing direct FFI without wrapper libraries. The compiler itself is written in Crystal, demonstrating the language's capability for systems programming.",
}

record Stats, avg_ms : Float64, p50_ms : Float64, p95_ms : Float64
record OneShot, ms : Float64, vec : Array(Float32)

def l2_normalize(vec : Array(Float32)) : Array(Float32)
  norm = Math.sqrt(vec.sum { |x| x * x }.to_f64)
  return vec if norm <= 1e-12
  inv = (1.0 / norm).to_f32
  vec.map { |x| x * inv }
end

def cosine(a : Array(Float32), b : Array(Float32)) : Float64
  raise "dim mismatch" unless a.size == b.size
  dot = 0.0_f64
  na = 0.0_f64
  nb = 0.0_f64
  a.size.times do |i|
    av = a[i].to_f64
    bv = b[i].to_f64
    dot += av * bv
    na += av * av
    nb += bv * bv
  end
  return 0.0 if na <= 1e-12 || nb <= 1e-12
  dot / (Math.sqrt(na) * Math.sqrt(nb))
end

def measure(runs : Int32, &block : -> Array(Float32)) : {Stats, Array(Float32)}
  times = Array(Float64).new(runs)
  last = [] of Float32
  runs.times do
    t0 = Time.instant
    last = yield
    times << (Time.instant - t0).total_milliseconds
  end
  sorted = times.sort
  p50 = sorted[sorted.size // 2]
  p95 = sorted[(sorted.size * 95 // 100).clamp(0, sorted.size - 1)]
  avg = times.sum / times.size
  {Stats.new(avg_ms: avg, p50_ms: p50, p95_ms: p95), last}
end

def measure_once(&block : -> Array(Float32)) : OneShot
  t0 = Time.instant
  vec = yield
  OneShot.new(ms: (Time.instant - t0).total_milliseconds, vec: vec)
end

def fmt(ms : Float64) : String
  ms.round(2).to_s.rjust(8)
end

model_path = DEFAULT_MODEL
runs = 7
warmup = 3
n_gpu_layers = 99
n_batch = 512
n_threads = 0
llama_flash_attn = true
llama_fresh_context = false

OptionParser.parse do |p|
  p.banner = "Usage: profile_nomic_vs_llama [options]"
  p.on("--model=PATH", "Path to GGUF model") { |v| model_path = v }
  p.on("--runs=N", "Measured runs per case (default: 7)") { |v| runs = v.to_i }
  p.on("--warmup=N", "Warmup runs per case (default: 3)") { |v| warmup = v.to_i }
  p.on("--n-gpu-layers=N", "llama.cpp GPU layers (default: 99)") { |v| n_gpu_layers = v.to_i }
  p.on("--n-batch=N", "llama.cpp context batch size (default: 512)") { |v| n_batch = v.to_i }
  p.on("--n-threads=N", "llama.cpp CPU threads (default: auto)") { |v| n_threads = v.to_i }
  p.on("--llama-flash-attn", "Enable llama.cpp flash attention (default)") { llama_flash_attn = true }
  p.on("--llama-no-flash-attn", "Disable llama.cpp flash attention") { llama_flash_attn = false }
  p.on("--llama-fresh-context", "Recreate llama.cpp context per embedding call") { llama_fresh_context = true }
end

unless File.exists?(model_path)
  STDERR.puts "Model not found: #{model_path}"
  exit 1
end

ML::Metal::Device.init!
native = ML::GGUF::NomicBertMoE.from_gguf(model_path, ML::GGUF::MetalBackend.new)

ML::LLM.init
llama_model = ML::LLM::Model.new(model_path, n_gpu_layers: n_gpu_layers)
llama_ctx = if llama_fresh_context
              nil
            else
              llama_model.create_context(
                n_ctx: 512,
                n_batch: n_batch,
                n_threads: n_threads,
                flash_attn: llama_flash_attn,
                embeddings: true,
              )
            end

def llama_embed(
  llama_model : ML::LLM::Model,
  llama_ctx : ML::LLM::Context?,
  text : String,
  n_batch : Int32,
  n_threads : Int32,
  flash_attn : Bool,
  fresh_context : Bool
) : Array(Float32)
  tokens = llama_model.tokenize(text, add_bos: true)

  if fresh_context
    ctx = llama_model.create_context(
      n_ctx: 512,
      n_batch: n_batch,
      n_threads: n_threads,
      flash_attn: flash_attn,
      embeddings: true,
    )
    begin
      raise "llama encode failed" unless ctx.encode(tokens)
      return l2_normalize(ctx.get_seq_embeddings(0).to_a)
    ensure
      ctx.free
    end
  end

  ctx = llama_ctx.not_nil!
  ctx.reset
  raise "llama encode failed" unless ctx.encode(tokens)
  l2_normalize(ctx.get_seq_embeddings(0).to_a)
end

begin
  STDERR.puts "settings: llama_flash_attn=#{llama_flash_attn} llama_fresh_context=#{llama_fresh_context} runs=#{runs} warmup=#{warmup}"
  STDERR.puts
  STDERR.puts "=== Native Metal vs llama.cpp: cold one-shot ==="
  STDERR.puts "#{"label".ljust(8)} #{"tok(native)".rjust(11)} #{"tok(llama)".rjust(11)} #{"native cold".rjust(11)} #{"llama cold".rjust(11)} #{"speedup".rjust(8)} #{"cos".rjust(8)}"
  STDERR.puts "-" * 86

  TEXTS.each do |label, text|
    native_tokens = native.tokenize(text).size
    llama_tokens = llama_model.tokenize(text, add_bos: true).size

    native_cold = measure_once { native.embed(text) }
    llama_cold = measure_once { llama_embed(llama_model, llama_ctx, text, n_batch, n_threads, llama_flash_attn, llama_fresh_context) }

    cold_speedup = llama_cold.ms / native_cold.ms
    cold_cos = cosine(native_cold.vec, llama_cold.vec)

    STDERR.puts "#{label.ljust(8)} #{native_tokens.to_s.rjust(11)} #{llama_tokens.to_s.rjust(11)} #{fmt(native_cold.ms)} #{fmt(llama_cold.ms)} #{cold_speedup.round(2).to_s.rjust(8)} #{cold_cos.round(6).to_s.rjust(8)}"
  end

  STDERR.puts
  STDERR.puts "=== Native Metal vs llama.cpp: steady state ==="
  STDERR.puts "#{"label".ljust(8)} #{"tok(native)".rjust(11)} #{"tok(llama)".rjust(11)} #{"native p50".rjust(10)} #{"native p95".rjust(10)} #{"llama p50".rjust(10)} #{"llama p95".rjust(10)} #{"speedup".rjust(8)} #{"cos".rjust(8)}"
  STDERR.puts "-" * 118

  warmup.times do
    TEXTS.each_value do |text|
      native.embed(text)
      llama_embed(llama_model, llama_ctx, text, n_batch, n_threads, llama_flash_attn, llama_fresh_context)
    end
  end

  TEXTS.each do |label, text|
    native_tokens = native.tokenize(text).size
    llama_tokens = llama_model.tokenize(text, add_bos: true).size

    native_stats, native_vec = measure(runs) { native.embed(text) }
    llama_stats, llama_vec = measure(runs) { llama_embed(llama_model, llama_ctx, text, n_batch, n_threads, llama_flash_attn, llama_fresh_context) }

    speedup = llama_stats.p50_ms / native_stats.p50_ms
    cos = cosine(native_vec, llama_vec)

    STDERR.puts "#{label.ljust(8)} #{native_tokens.to_s.rjust(11)} #{llama_tokens.to_s.rjust(11)} #{fmt(native_stats.p50_ms)} #{fmt(native_stats.p95_ms)} #{fmt(llama_stats.p50_ms)} #{fmt(llama_stats.p95_ms)} #{speedup.round(2).to_s.rjust(8)} #{cos.round(6).to_s.rjust(8)}"
  end
ensure
  llama_ctx.try(&.free)
  llama_model.free
  ML::LLM.cleanup
end
