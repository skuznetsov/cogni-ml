# Cogni-ML

Crystal machine learning library with native Apple Silicon GPU acceleration.

**Highlights:**
- Native Metal GPU embedding pipeline — **43ms** for 260 tokens on M2 Max (2.2x faster than baseline)
- GGUF model loading with Q5_K/Q6_K quantization support
- `simdgroup_matrix_multiply_accumulate` GEMM kernels
- Compute graph with automatic wave-based barrier optimization
- Autograd engine, NN layers, Adam optimizer
- llama.cpp bindings for any GGUF model

## Architecture

```
src/ml/
  core/         Tensor, Shape, MetalBuffer
  autograd/     Variable, GradFn (backward pass)
  nn/           Linear, LayerNorm, MultiHeadAttention, ViT
  optim/        Adam/AdamW
  llm/          llama.cpp FFI bindings
  gguf/         GGUF reader, tokenizer, dequantization, NomicBertMoE
  metal/        Device, ComputeEncoder, ComputeGraph, GraphEncoder
```

## GPU Embedding Pipeline

The crown jewel: a fully native Metal compute pipeline for nomic-embed-text-v2-moe BERT embeddings.

```crystal
require "ml"
require "ml/gguf/nomic_bert"
require "ml/gguf/metal_backend"
require "ml/metal/compute_graph"

ML::Metal::Device.init!
model = ML::GGUF::NomicBertMoE.from_gguf("path/to/model.gguf", ML::GGUF::MetalBackend.new)

embedding = model.embed("Your text here")  # → Array(Float32), dim=768
```

### Performance (Apple M2 Max, 38 GPU cores)

| Tokens | Latency |
|--------|---------|
| 20     | 14ms    |
| 94     | 16ms    |
| 196    | 33ms    |
| 433    | 70ms    |

### What's inside

- **simdgroup_matrix GEMM** — hardware-accelerated 8x8 matrix tiles for Q5_K/Q6_K dequant+multiply
- **Batched expert GEMM** — all 8 MoE experts in 1 dispatch (LTP Diamond surgery)
- **ComputeGraph** — automatic wave scheduling with offset-aware + Block Integrity dependency analysis
- **GraphEncoder** — drop-in ComputeEncoder replacement that builds the compute graph
- **Fused kernels** — QKV split+RoPE, gate+softmax+topk, atomic scatter, f32 norm2
- **Indirect dispatch** — GPU-driven threadgroup counts, zero CPU-GPU sync for MoE routing

### Supported models

| Model | Format | Status |
|-------|--------|--------|
| nomic-embed-text-v2-moe | GGUF Q5_K_M | Full native Metal pipeline |
| Any BERT-like encoder | GGUF | Via NomicBertMoE (if architecture matches) |
| Llama, Qwen, Mistral, etc. | GGUF | Via llama.cpp bindings |

## Installation

```yaml
# shard.yml
dependencies:
  cogni-ml:
    github: anthropics/cogni-ml  # or local path
    version: ~> 0.10.0
```

### Build with Metal GPU

```sh
make build    # Compiles bridge.mm + links Metal frameworks
make spec     # Run tests with GPU
EMBED_MODEL=/path/to/nomic.gguf make profile_nomic  # Stage breakdown for native Metal embeddings
```

### CPU-only build

```sh
crystal build -Dcpu_only your_app.cr
```

## Quick Start

### Tensor + Autograd (CPU)

```crystal
require "ml"

x = ML::Autograd::Variable.rand(2, 3, requires_grad: true, device: ML::Tensor::Device::CPU)
layer = ML::NN::Linear.new(3, 4, device: ML::Tensor::Device::CPU)

out = layer.forward(x)
loss = out.mean
loss.backward

opt = ML::Optim::Adam.new(layer.parameters)
opt.step
opt.zero_grad
```

### LLM Inference (llama.cpp)

```crystal
require "ml/llm/llama"

ML::LLM.init
model = ML::LLM::Model.new("path/to/model.gguf")
gen = ML::LLM::Generator.new(model)
puts gen.ask("What is Crystal?", max_tokens: 100)
ML::LLM.cleanup
```

### GGUF Embeddings (Metal GPU)

```crystal
require "ml"
require "ml/gguf/nomic_bert"
require "ml/gguf/metal_backend"
require "ml/metal/compute_graph"

ML::Metal::Device.init!
model = ML::GGUF::NomicBertMoE.from_gguf(
  "nomic-embed-text-v2-moe.Q5_K_M.gguf",
  ML::GGUF::MetalBackend.new
)

# Single embedding
vec = model.embed("Crystal programming language")
puts "dim=#{vec.size}"  # 768

# Batch embedding
vecs = model.embed_batch(["Hello", "World", "Crystal"])
```

## Metal Kernels

11 Metal shader files implementing:

| Kernel | Purpose |
|--------|---------|
| `gemm_mm.metal` | simdgroup_matrix GEMM for Q5_K/Q6_K + batched expert variants |
| `gemm_simd.metal` | Scalar SIMD GEMM (small batch fallback) |
| `attention_matmul.metal` | Flash attention with simdgroup_matrix Q*K^T |
| `bert_fp16.metal` | Fused ops: QKV split+RoPE, gate+softmax+topk, norms, scatter, routing |
| `gemm_mm_f16.metal` | FP16 GEMM (experimental) |
| `nn.metal` | General NN ops (linear, layernorm, GELU) |

## Platform Support

| Platform | GPU | CPU | Status |
|----------|-----|-----|--------|
| macOS (Apple Silicon) | Metal | Yes | Primary target |
| macOS (Intel) | Metal | Yes | Supported |
| Linux | - | Yes | `-Dcpu_only` |

## Build Flags

| Flag | Effect |
|------|--------|
| `-Dcpu_only` | Disable Metal, pure CPU |
| `-Duse_gguf` | Enable GGUF model loading (requires llama.cpp for LLM, standalone for embeddings) |

## License

MIT
