require "./src/ml"
require "./src/ml/metal/compute_graph"
require "./src/ml/gguf/nomic_bert"
require "./src/ml/gguf/metal_backend"

MODEL = "/Users/sergey/.cache/lm-studio/models/nomic-ai/nomic-embed-text-v2-moe-GGUF/nomic-embed-text-v2-moe.Q5_K_M.gguf"

ML::Metal::Device.init!

# Test 1: Graph API + wave compilation
g = ML::Metal::ComputeGraph.new
STDERR.puts "ComputeGraph: size=#{g.size}"

buf_a = ML::MetalBuffer.new(1024_i64)
buf_b = ML::MetalBuffer.new(1024_i64)
buf_c = ML::MetalBuffer.new(1024_i64)

# Simulate 3 ops: A→B, A→C (independent), B+C→A
# Create a simple kernel for testing (reuse zero_region)
src = {{ read_file("#{__DIR__}/src/ml/gguf/kernels/bert_fp16.metal") }}
pipe = ML::Metal::ComputePipeline.new("zero_region", src)

g.add_op(pipe) do |op|
  op.buffer(buf_a, 0, :read)
  op.buffer(buf_b, 1, :write)
  op.dispatch_1d(256, 256)
end

g.add_op(pipe) do |op|
  op.buffer(buf_a, 0, :read)
  op.buffer(buf_c, 1, :write)
  op.dispatch_1d(256, 256)
end

g.add_op(pipe) do |op|
  op.buffer(buf_b, 0, :read)
  op.buffer(buf_c, 0, :read)  # different binding index, same buffer
  op.buffer(buf_a, 1, :write)
  op.dispatch_1d(256, 256)
end

g.compile!
st = g.stats
STDERR.puts "Compiled: ops=#{st.n_ops} waves=#{st.n_waves} barriers=#{st.n_barriers} max_width=#{st.max_wave_width}"
# Expected: 3 ops, 2 waves (op0+op1 concurrent, op2 after), 1 barrier, max_width=2

# Test 2: Full embedding with graph (just verify old path still works)
gpu = ML::GGUF::NomicBertMoE.from_gguf(MODEL, ML::GGUF::MetalBackend.new)
eg = gpu.embed("Hello world")
STDERR.puts "Embed OK: dim=#{eg.size} norm=#{Math.sqrt(eg.sum { |v| v * v })}"
