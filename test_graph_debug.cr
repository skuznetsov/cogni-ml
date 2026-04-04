require "./src/ml"
require "./src/ml/metal/compute_graph"
require "./src/ml/gguf/nomic_bert"
require "./src/ml/gguf/metal_backend"

MODEL = "/Users/sergey/.cache/lm-studio/models/nomic-ai/nomic-embed-text-v2-moe-GGUF/nomic-embed-text-v2-moe.Q5_K_M.gguf"

ML::Metal::Device.init!
gpu = ML::GGUF::NomicBertMoE.from_gguf(MODEL, ML::GGUF::MetalBackend.new)
cpu = ML::GGUF::NomicBertMoE(ML::GGUF::F32Backend).from_gguf(MODEL)

# Long text to trigger GPU routing path
text = "The Crystal programming language is a statically typed, compiled language with syntax inspired by Ruby. " \
  "It features type inference, macros, and generics, making it both expressive and performant. Crystal compiles " \
  "to native code via LLVM and achieves C-like performance while maintaining developer-friendly syntax. The " \
  "language supports concurrency through fibers and channels, similar to Go's goroutines. Crystal's type system " \
  "catches errors at compile time, eliminating many runtime bugs common in dynamic languages. The standard " \
  "library includes HTTP servers, JSON parsing, database drivers, and cryptographic primitives."

ec = cpu.embed(text)

# Enable debug for one encode
ML::Metal::ComputeGraph.debug_encode = true
eg = gpu.embed(text)
ML::Metal::ComputeGraph.debug_encode = false

dot = 0.0_f64
ec.size.times { |i| dot += ec[i].to_f64 * eg[i].to_f64 }
STDERR.puts "cos = #{dot.round(6)}"
