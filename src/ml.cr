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
require "./ml/core/shape"
require "./ml/core/buffer"
require "./ml/core/tensor"

# Autograd - automatic differentiation
require "./ml/autograd/grad_fn"
require "./ml/autograd/variable"

# Neural network layers
require "./ml/nn/gpu_ops"
require "./ml/nn/linear"
require "./ml/nn/layernorm"
require "./ml/nn/attention"
require "./ml/nn/vit"

# Optimizers
require "./ml/optim/adam"

module ML
  VERSION = "0.1.0"

  # Classes are defined in submodules, access via:
  # ML::Tensor, ML::Autograd::Variable, ML::NN::Linear, etc.
end
