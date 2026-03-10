# Cogni-ML

Experimental Crystal ML library extracted from internal projects (weather, folding, 3d_scanner).

Status: Early-stage. Metal and LLM features require native dependencies. See `docs/CRITICAL_REVIEW.md` for remaining gaps and risks.

## Features
- Tensor core with shapes, strides, and CPU/GPU storage
- Autograd engine for a small set of ops (add/sub/mul/div/matmul/relu/sigmoid/sum/mean)
- NN layers: Linear, LayerNorm, RMSNorm, Multi-Head Attention, ViT encoder blocks
- Optimizers: Adam/AdamW, SGD, LR schedulers
- LLM wrapper around llama.cpp (optional)
- Metal GPU kernels (experimental)

## Repository layout
- `src/ml/core` Tensor/Shape/Buffer
- `src/ml/autograd` Variable + GradFn
- `src/ml/nn` Layers and GPU ops
- `src/ml/optim` Optimizers + schedulers
- `src/ml/llm` llama.cpp bindings and high-level API
- `src/ml/metal` Metal stubs and kernels

## Installation
This repository is not published as a shard. Use it via a local path in your app:

```yaml
# shard.yml
name: my_app
version: 0.1.0

dependencies:
  cogni-ml:
    path: ../cogni-ml
```

Then require it:

```crystal
require "ml"
```

### CPU-only build
If you want a CPU-only build (no Metal usage), compile with `-Dcpu_only`:

```sh
make spec_cpu
make build_cpu
```

## Quick start (CPU)
The default device for `Tensor` is GPU. For CPU-only usage, pass `device: ML::Tensor::Device::CPU`.

```crystal
require "ml"

x = ML::Autograd::Variable.rand(2, 3, requires_grad: true, device: ML::Tensor::Device::CPU)
layer = ML::NN::Linear.new(3, 4, device: ML::Tensor::Device::CPU)

# Forward + loss
out = layer.forward(x)
loss = out.mean

# Backward
loss.backward

# Optimize
opt = ML::Optim::Adam.new(layer.parameters)
opt.step
opt.zero_grad
```

## GPU / Metal
Metal support relies on a small Objective‑C++ bridge. Use the provided `Makefile` to compile and link it:

```sh
make build
make spec
```

This compiles `src/ml/metal/bridge.mm` and links it with Metal + Foundation. You’ll need Xcode/Command Line Tools installed.

## Platform Support
- macOS: Metal GPU backend supported.
- Linux/FreeBSD: CPU-only build via `make build_cpu` or `-Dcpu_only`.
- Planned: CUDA backend (not implemented, no test hardware yet).

## LLM (llama.cpp)
The LLM wrapper is optional and lives under `ML::LLM`. It requires linking `libllama` from llama.cpp. The bindings in `src/ml/llm/llama_ffi.cr` target llama.cpp API version ~7340 (Dec 2024).

To build llama.cpp locally:

```sh
make llama
```

`make llama` will build from `LLAMA_DIR` if it exists, or use an already-installed `libllama` (e.g., Homebrew on macOS) if found.

Then make sure the library is discoverable:

```sh
eval "$(make llama_env)"
```

```crystal
require "ml/llm/llama"

ML::LLM.init
begin
  model = ML::LLM::Model.new("/path/to/model.gguf")
  ctx = model.create_context
  ctx.setup_greedy_sampler
  ctx.eval(model.tokenize("Hello"))
  puts model.token_to_piece(ctx.sample)
ensure
  ctx.try(&.free)
  model.try(&.free)
  ML::LLM.cleanup
end
```

See also `docs/LLM.md` and `examples/llm_inference.cr`.

## Docs
- `docs/USAGE.md`
- `docs/CRITICAL_REVIEW.md`

## License
MIT
