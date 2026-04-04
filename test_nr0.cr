require "./src/ml"
require "./src/ml/gguf/nomic_bert"
require "./src/ml/gguf/metal_backend"

MODEL = "/Users/sergey/.cache/lm-studio/models/nomic-ai/nomic-embed-text-v2-moe-GGUF/nomic-embed-text-v2-moe.Q5_K_M.gguf"

ML::Metal::Device.init!
cpu = ML::GGUF::NomicBertMoE(ML::GGUF::F32Backend).from_gguf(MODEL)
gpu = ML::GGUF::NomicBertMoE.from_gguf(MODEL, ML::GGUF::MetalBackend.new)

texts = [
  "Hello world",
  "Crystal programming language",
  "def method_name(arg : String) : Bool",
  "PostgreSQL extension for vector search with pgvector and HNSW indexing",
  # ~300 token paragraph for llama.cpp comparison target
  "The Crystal programming language is a statically typed, compiled language with syntax inspired by Ruby. " \
  "It features type inference, macros, and generics, making it both expressive and performant. Crystal compiles " \
  "to native code via LLVM and achieves C-like performance while maintaining developer-friendly syntax. The " \
  "language supports concurrency through fibers and channels, similar to Go's goroutines. Crystal's type system " \
  "catches errors at compile time, eliminating many runtime bugs common in dynamic languages. The standard " \
  "library includes HTTP servers, JSON parsing, database drivers, and cryptographic primitives. Crystal also " \
  "supports C bindings through its lib declaration syntax, allowing direct FFI without wrapper libraries. The " \
  "compiler itself is written in Crystal, demonstrating the language's capability for systems programming. " \
  "Recent versions have added support for Windows, multi-threading with the -Dpreview_mt flag, and improved " \
  "incremental compilation. The community maintains a package manager called Shards, with hundreds of libraries " \
  "available for web development, machine learning, and system administration tasks.",
]

texts.each do |text|
  # Warmup
  gpu.embed(text)

  # CPU reference
  t0 = Time.instant
  ec = cpu.embed(text)
  cm = (Time.instant - t0).total_milliseconds

  # GPU best-of-10
  best = Float64::MAX
  eg = [] of Float32
  10.times do
    t0 = Time.instant
    eg = gpu.embed(text)
    ms = (Time.instant - t0).total_milliseconds
    best = ms if ms < best
  end

  # cos(GPU, CPU)
  dot = 0.0_f64
  ec.size.times { |i| dot += ec[i].to_f64 * eg[i].to_f64 }

  STDERR.puts "#{text[0, 40].ljust(42)} GPU=#{best.round(1)}ms CPU=#{cm.round(1)}ms cos=#{dot.round(4)}"
end
