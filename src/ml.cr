# Crystal ML - Machine Learning Library
#
# Provides:
# - Autograd engine (automatic differentiation)
# - Tensor operations with Metal GPU support
# - Neural network layers (Linear, LayerNorm, Attention, ViT)
# - Optimizers (Adam, SGD) with LR scheduling
#
# Unified ML shard extracted from 3d_scanner and folding projects.

# Core tensor operations
require "./ml/core/dtype"
require "./ml/core/shape"
require "./ml/core/buffer"
require "./ml/core/floating_storage"
require "./ml/core/tensor"

# Bounded immutable CPU sparse values
require "./ml/sparse"

# Graphless CPU reference operations
require "./ml/ops/normalization"

# Device optimization admission and evidence contracts
require "./ml/metal/wba_phi_atlas"

# Autograd - automatic differentiation
require "./ml/autograd/grad_fn"
require "./ml/autograd/variable"

# Neural network layers
require "./ml/nn/gpu_ops"
require "./ml/nn/linear"
require "./ml/sparse/linear"
require "./ml/sparse/self_attention_qkv"
require "./ml/sparse/full_self_attention_plan"
require "./ml/sparse/full_self_attention"
require "./ml/nn/layernorm"
require "./ml/nn/attention"
require "./ml/nn/vit"

# Bounded graphless vision references
require "./ml/vision/dino_v3"

# Optimizers
require "./ml/optim/adam"

# Native 3D model artifact contracts
require "./ml/three_d/trellis2"

module ML
  VERSION = "0.1.0"

  # Classes are defined in submodules, access via:
  # ML::Tensor, ML::Autograd::Variable, ML::NN::Linear, etc.
end
