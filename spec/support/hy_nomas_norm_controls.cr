require "../../src/ml/video/hy_nomas_mps_norm"

# Same-input diagnostic access to the prepared norm implementation used by the
# opt-in block backend. Kernel source and PyTorch v2.8.0 BSD attribution are in
# `src/ml/video/kernels/hy_nomas_mps_norm.metal` and `third_party/VIDEO_NOTICES.md`.
module HyNomasNormProbe
  alias Controls = ML::Video::HyNomasMPSLayerNorm::Plan
end
