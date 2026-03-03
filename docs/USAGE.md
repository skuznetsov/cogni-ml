# Usage

Note: Metal support now requires compiling the bridge (`src/ml/metal/bridge.mm`) and linking it. Use `make spec` or `make build` on macOS. See `docs/CRITICAL_REVIEW.md` for remaining gaps.

CPU-only mode is available with `-Dcpu_only` (see `make spec_cpu` / `make build_cpu`).

## Tensor basics

```crystal
require "ml"

# CPU tensor (portable default)
x = ML::Tensor.zeros(2, 3, device: ML::Tensor::Device::CPU)
x[0, 1] = 3.5_f32
puts x.to_a

# Reshape and transpose
x2 = x.reshape(3, 2)
xt = x2.t
puts xt.shape
```

## Autograd basics

```crystal
require "ml"

x = ML::Autograd::Variable.ones(2, 2, requires_grad: true, device: ML::Tensor::Device::CPU)
w = ML::Autograd::Variable.full(2, 2, 2.0_f32, requires_grad: true, device: ML::Tensor::Device::CPU)

# y = sum(x * w)
y = (x * w).sum

# Backprop

y.backward

puts x.grad.try(&.to_a)
puts w.grad.try(&.to_a)
```

## LayerNorm and MLP

```crystal
require "ml"

x = ML::Autograd::Variable.rand(4, 8, requires_grad: false, device: ML::Tensor::Device::CPU)
ln = ML::NN::LayerNorm.new(8, device: ML::Tensor::Device::CPU)
mlp = ML::NN::MLP.new(8, 16, 8, device: ML::Tensor::Device::CPU)

out = mlp.forward(ln.forward(x))
puts out.data.shape
```

## Multi-Head Attention (inference)

```crystal
require "ml"

x = ML::Autograd::Variable.rand(2, 4, 8, requires_grad: false, device: ML::Tensor::Device::CPU)
attn = ML::NN::MultiHeadAttention.new(8, 2, device: ML::Tensor::Device::CPU)

out = attn.self_attention(x)
puts out.data.shape
```

Note: Attention autograd runs on the CPU path; GPU attention remains inference‑only.

## ViT encoder (inference)

```crystal
require "ml"

img = ML::Autograd::Variable.rand(1, 3, 32, 32, requires_grad: false, device: ML::Tensor::Device::CPU)
vit = ML::NN::ViTEncoder.new(img_size: 32, patch_size: 8, embed_dim: 64, depth: 2, num_heads: 4, device: ML::Tensor::Device::CPU)

seq = vit.forward(img)
puts seq.data.shape
```

## LLM (llama.cpp)

```crystal
require "ml/llm/llama"

ML::LLM.init
model = ML::LLM::Model.new("/path/to/model.gguf")
ctx = model.create_context
ctx.setup_greedy_sampler
ctx.eval(model.tokenize("Hello"))
puts model.token_to_piece(ctx.sample)
ML::LLM.cleanup
```

## Environment variables

- `GS_BUFFER_POOL_MAX_CACHED` limits cached buffer count.
- `GS_BUFFER_POOL_MAX_BUFFER_BYTES` caps max buffer size for pooling.
- `GS_BUFFER_POOL_MAX_BYTES` caps total cached bytes.
- `GS_PURGEABLE_POOL` controls Metal purgeable state (`empty`, `volatile`, `off`).
- `GS_ATTN_RESHAPE_CPU` forces CPU reshape in attention.
- `GS_PATCH_EMBED_CPU` forces CPU patch embedding.
