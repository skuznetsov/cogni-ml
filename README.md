# Cogni-ML

Crystal machine learning library with native Apple Silicon GPU acceleration.

Cogni-ML is currently two things:

- A general Crystal ML toolkit: tensors, autograd, NN layers, optimizers, GGUF readers, and llama.cpp bindings.
- A native Metal inference lab for GGUF models, with production-oriented work on `nomic-embed-text-v2-moe` embeddings and Qwen 3.5 text generation.

## Highlights

- Native Metal embedding pipeline for `nomic-embed-text-v2-moe`.
- Native Qwen 3.5 9B GGUF inference path for Apple Silicon Metal.
- Q4_K/Q5_K/Q6_K/Q8_0 quantized matmul kernels.
- Chunked Qwen 3.5 prefill, decode wave scheduling, prompt-state cache restore, and exact speculative decode harnesses.
- ComputeGraph wave scheduling with offset-aware barrier optimization.
- Crystal autograd engine, NN layers, and Adam/AdamW optimizers.
- llama.cpp FFI bindings for general GGUF model access.

## Architecture

```text
src/ml/
  core/         Tensor, Shape, MetalBuffer
  autograd/     Variable, GradFn backward pass
  nn/           Linear, LayerNorm, MultiHeadAttention, ViT
  optim/        Adam/AdamW
  llm/          llama.cpp FFI bindings
  gguf/         GGUF reader, tokenizer, dequantization, Qwen35, NomicBertMoE
  metal/        Device, ComputeEncoder, ComputeGraph, GraphEncoder
```

## Qwen 3.5 Native Metal

The native Qwen path targets `Qwen3.5-9B-Q4_K_M.gguf` on Apple Silicon. The code supports:

- Qwen 3.5 GGUF metadata and tokenizer loading.
- Q4_K, Q5_K, Q6_K, and Q8_0 quantized projections.
- Full-attention layers with GQA, partial RoPE, KV cache writes, and fused output projection.
- DeltaNet/recurrent layers with GPU-resident recurrent state and chunked prefill scan.
- Chunked prefill with final-token top1 shortcut.
- Decode wave scheduling to reduce command-buffer boundaries.
- Exact prompt-state save/restore and longest-prefix prompt cache.
- Exact speculative decode harnesses:
  - neural draft with Qwen 3.5 0.8B Q8_0,
  - n-gram/cache draft for repeated/generated-template text,
  - target-verifier chunks with row-batched top1 for larger accepted chunks.

The 9B Q4_K_M path is the primary verified target. Qwen 3.6 27B is a scale-up target, but it should be treated as experimental until local correctness and performance runs are completed.

### Model Layout

The developer CLIs default to local LM Studio / llama.cpp-style paths:

```text
~/.cache/lm-studio/models/lmstudio-community/Qwen3.5-9B-GGUF/Qwen3.5-9B-Q4_K_M.gguf
~/.cache/lm-studio/models/lmstudio-community/Qwen3.5-0.8B-GGUF/Qwen3.5-0.8B-Q8_0.gguf
~/SrcArchives/AI/llama.cpp/build/bin/llama-tokenize
~/SrcArchives/AI/llama.cpp/build/bin/llama-bench
```

Most benchmark/probe CLIs also accept `--model`, `--target`, `--draft`, `--tokenizer-bin`, or environment overrides. `bin/qwen35_generate.cr` is intentionally a small demo and currently uses its constants at the top of the file.

### Build Qwen CLIs

Build the Metal bridge once:

```sh
make build/bridge.o
```

Build the practical generation demo:

```sh
crystal build --release --no-debug \
  --link-flags="$(pwd)/build/bridge.o -framework Metal -framework Foundation -lc++" \
  bin/qwen35_generate.cr \
  -o build/qwen35_generate
```

Run greedy generation:

```sh
./build/qwen35_generate "The capital of France is" 64
```

Enable exact n-gram speculative decode for repeated text:

```sh
QWEN35_NGRAM_DECODE=1 ./build/qwen35_generate "The capital of France is" 64
```

Enable exact prompt cache:

```sh
QWEN35_PROMPT_CACHE=1 \
QWEN35_SESSION_ID=demo \
./build/qwen35_generate "The capital of France is" 64
```

Useful Qwen environment switches:

| Variable | Effect |
|---|---|
| `QWEN35_PROMPT_CACHE=1` | Enable exact prompt-state cache lookup/save in `qwen35_generate`. |
| `QWEN35_PROMPT_CACHE_ROOT=/path` | Override prompt-cache artifact root. |
| `QWEN35_NGRAM_DECODE=1` | Enable exact n-gram speculative decode in `qwen35_generate`. |
| `QWEN35_NGRAM_GAMMA=32` | Maximum n-gram verifier chunk size. |
| `QWEN35_NGRAM_MIN=6` | Minimum repeated suffix length before n-gram drafting. |
| `QWEN35_NGRAM_MAX=8` | Maximum suffix length to search for n-gram drafting. |
| `QWEN35_NGRAM_RECURSIVE_OFF=1` | Disable recursive n-gram extension through scratch history. |
| `QWEN35_NGRAM_DISABLE_AFTER_REJECT_OFF=1` | Exploration mode: keep trying n-gram chunks after first rejection. |
| `QWEN35_PREFILL_CHUNK_OFF=1` | Force older non-chunked prefill path. |
| `QWEN35_DECODE_WAVE_OFF=1` | Force older non-wave decode path. |

### Library Integration

The current Qwen API is low-level and intended for native inference experiments:

```crystal
require "ml/gguf/qwen35_cpu"
require "ml/gguf/qwen35_weights"

model = "/path/to/Qwen3.5-9B-Q4_K_M.gguf"
weights = ML::GGUF::Qwen35Weights.from_gguf(model)
state = ML::GGUF::Qwen35CPU::State.new(weights.hparams, max_seq: 1024)

prompt_ids = [760_i32, 6511_i32, 314_i32, 9338_i32, 13_i32]
next_id, next_logit = ML::GGUF::Qwen35CPU.prefill_tokens_top1(weights, prompt_ids, 0, state)

64.times do |i|
  puts next_id
  next_id, next_logit = ML::GGUF::Qwen35CPU.forward_top1(weights, next_id, prompt_ids.size + i, state)
end
```

When linking an executable that uses Metal, include the bridge object and Apple frameworks:

```sh
crystal build your_app.cr \
  --link-flags="$(pwd)/build/bridge.o -framework Metal -framework Foundation -lc++"
```

For CPU-only builds:

```sh
crystal build -Dcpu_only your_app.cr
```

### Benchmark Against llama.cpp

Build the matched benchmark:

```sh
crystal build --release --no-debug \
  --link-flags="$(pwd)/build/bridge.o -framework Metal -framework Foundation -lc++" \
  bin/benchmark_qwen_vs_llama.cr \
  -o build/benchmark_qwen_vs_llama
```

Run a normal first-run prefill/decode comparison:

```sh
./build/benchmark_qwen_vs_llama \
  --model ~/.cache/lm-studio/models/lmstudio-community/Qwen3.5-9B-GGUF/Qwen3.5-9B-Q4_K_M.gguf \
  --llama-bench ~/SrcArchives/AI/llama.cpp/build/bin/llama-bench \
  --prompt=64 \
  --gen=64 \
  --reps=5 \
  --warmup=2
```

For publishable measurements, wait for a quiet host:

```sh
./build/benchmark_qwen_vs_llama \
  --prompt=64 \
  --gen=64 \
  --reps=5 \
  --warmup=2 \
  --wait-quiet-ms=60000 \
  --require-quiet
```

Additional benchmark modes:

```sh
# State buffers allocated once, then reset between reps.
./build/benchmark_qwen_vs_llama --native-prefill-prealloc

# Exact prompt-cache restore after one seeded native prefill.
./build/benchmark_qwen_vs_llama --native-prefill-cache
```

Fresh local M2 Max 64GB snapshot, Qwen 3.5 9B Q4_K_M, llama.cpp `llama-bench`, `prompt=64`, `gen=64`, `reps=3`, `warmup=1`, flash-attention off:

| Mode | cogni-ml | llama.cpp | Gap |
|---|---:|---:|---:|
| First-run prefill | 406.21 tok/s p50 | 436.62 tok/s avg | -6.96% |
| Prefill with preallocated state | 427.40 tok/s p50 | 436.15 tok/s avg | -2.01% |
| Prompt-cache restore | 1264.99 tok/s p50 | 437.55 tok/s avg | +189.11% |
| Decode top1 | 39.89 tok/s p50 | 39.75 tok/s avg | +0.34% |

Notes:

- The table is a local engineering snapshot, not a lab-clean public benchmark.
- `--native-prefill-cache` measures exact restore of a previously computed prompt state; it is not a first-run prefill replacement.
- Desktop CPU/GPU load can move short decode numbers by several percent. Use `--require-quiet` for comparisons that matter.

### Speculative Decode Harnesses

Neural draft harness with Qwen 3.5 0.8B Q8_0:

```sh
crystal build --release --no-debug \
  --link-flags="$(pwd)/build/bridge.o -framework Metal -framework Foundation -lc++" \
  bin/qwen35_speculative_accept.cr \
  -o build/qwen35_speculative_accept

./build/qwen35_speculative_accept \
  --target ~/.cache/lm-studio/models/lmstudio-community/Qwen3.5-9B-GGUF/Qwen3.5-9B-Q4_K_M.gguf \
  --draft ~/.cache/lm-studio/models/lmstudio-community/Qwen3.5-0.8B-GGUF/Qwen3.5-0.8B-Q8_0.gguf \
  --tokens 64 \
  --ngram \
  "The capital of France is"
```

Target-only n-gram speculative harness:

```sh
crystal build --release --no-debug \
  --link-flags="$(pwd)/build/bridge.o -framework Metal -framework Foundation -lc++" \
  bin/qwen35_ngram_speculative.cr \
  -o build/qwen35_ngram_speculative

./build/qwen35_ngram_speculative \
  --tokens 64 \
  --gamma 32 \
  --min-ngram 6 \
  "The capital of France is"
```

Both harnesses replay/check exact greedy target output by default unless their CLI explicitly says otherwise.

## Native Metal Embeddings

The embedding path targets `nomic-embed-text-v2-moe` with a fully native Metal compute pipeline.

```crystal
require "ml"
require "ml/gguf/nomic_bert"
require "ml/gguf/metal_backend"
require "ml/metal/compute_graph"

ML::Metal::Device.init!
model = ML::GGUF::NomicBertMoE.from_gguf("path/to/model.gguf", ML::GGUF::MetalBackend.new)

embedding = model.embed("Your text here")
```

### Embedding Performance

Apple M2 Max, 38 GPU cores:

| Tokens | Latency |
|---:|---:|
| 20 | 14 ms |
| 94 | 16 ms |
| 196 | 33 ms |
| 433 | 70 ms |

### Embedding Pipeline Internals

- simdgroup-matrix GEMM for Q5_K/Q6_K dequant+multiply.
- Batched expert GEMM for MoE experts.
- ComputeGraph wave scheduling with offset-aware dependency analysis.
- Fused QKV split/RoPE, gate/softmax/top-k, scatter, and norm kernels.
- GPU-driven dispatch where useful.

## Supported Models

| Model | Format | Status |
|---|---|---|
| `Qwen3.5-9B` | GGUF Q4_K_M | Native Metal text generation path, active optimization target. |
| `Qwen3.5-0.8B` | GGUF Q8_0 | Native draft model path for speculative decode harnesses. |
| `Qwen3.6-27B` | GGUF Q4_K_M target | Planned/experimental scale-up target. |
| `nomic-embed-text-v2-moe` | GGUF Q5_K_M | Native Metal embedding pipeline. |
| BERT-like encoders | GGUF | Via `NomicBertMoE` when the architecture matches. |
| Other Llama/Qwen/Mistral-style models | GGUF | Via llama.cpp bindings. |

## Installation

```yaml
# shard.yml
dependencies:
  cogni-ml:
    github: skuznetsov/cogni-ml
    version: ~> 0.40.0
```

## Build And Test

```sh
make build
make spec
```

CPU-only:

```sh
make build_cpu
make spec_cpu
```

llama.cpp helper targets:

```sh
make llama
make llama_env
```

The Makefile searches common local, Homebrew, and system library locations for `libllama`. Override with `LLAMA_DIR`, `LLAMA_BUILD`, or `LLAMA_LIB_DIR` if needed.

## Quick Start

### Tensor + Autograd

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

### LLM Inference Through llama.cpp

```crystal
require "ml/llm/llama"

ML::LLM.init
model = ML::LLM::Model.new("path/to/model.gguf")
gen = ML::LLM::Generator.new(model)
puts gen.ask("What is Crystal?", max_tokens: 100)
ML::LLM.cleanup
```

### GGUF Embeddings

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

vec = model.embed("Crystal programming language")
puts "dim=#{vec.size}"

vecs = model.embed_batch(["Hello", "World", "Crystal"])
```

## Metal Kernels

| Kernel | Purpose |
|---|---|
| `gemm_q4k.metal` | Q4_K GEMV/GEMM paths for Qwen. |
| `gemm_q56k.metal` | Q5_K/Q6_K/Q8_0 GEMV, top1, and helper kernels for Qwen. |
| `gemm_mm.metal` | simdgroup-matrix GEMM for Q5_K/Q6_K and batched expert variants. |
| `gemm_simd.metal` | Scalar SIMD GEMM fallback. |
| `ffn_qwen35.metal` | Qwen FFN, add, RMSNorm, and activation helpers. |
| `delta_net.metal` | Qwen 3.5 DeltaNet/recurrent kernels. |
| `fullattn_qwen35.metal` | Qwen full-attention prefill/decode helpers. |
| `attn_decode_qwen35.metal` | Qwen gated attention decode. |
| `attention_matmul.metal` | Flash-style attention matrix helpers. |
| `bert_fp16.metal` | Nomic/BERT fused ops. |
| `nn.metal` | General NN ops. |

## Platform Support

| Platform | GPU | CPU | Status |
|---|---|---|---|
| macOS Apple Silicon | Metal | Yes | Primary target. |
| macOS Intel | Metal | Yes | Supported for general Metal paths; Qwen performance focus is Apple Silicon. |
| Linux | No native Metal | Yes | Use `-Dcpu_only` or llama.cpp bindings. |
| FreeBSD | No native Metal | Untested CPU-only | Not a primary CI target. |

NVIDIA/CUDA support is not implemented. The Qwen native path is Metal-first.

## Build Flags

| Flag | Effect |
|---|---|
| `-Dcpu_only` | Disable Metal and build pure CPU paths. |
| `-Duse_gguf` | Enable GGUF model loading where applicable. |

## License

MIT
