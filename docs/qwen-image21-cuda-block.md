# Qwen-Image 2.1 CUDA block frontier

Status: VERIFIED within the fixtures below on 2026-10-16. This slice implements
one DiT block, not a complete CUDA image pipeline. The existing Metal backend
and package defaults remain active. Base revision: `7cccdd8bfb02ac388001ab7a61832087d8d1b450`.

The CUDA block reuses the Qwen3.5 CUDA driver and packed Q8_0, Q5_K and Q6_K
projection kernels. The Q8 PTX was extracted byte-for-byte from the existing
probe; Q5/Q6 PTX is unchanged. It adds conditional LayerNorm, head RMSNorm,
three-axis interleaved RoPE, block-causal image attention, and gated residuals.
Weights retain their GGUF representation; activations and accumulators are F32.

Construction uploads weights and constants and allocates all scratch storage.
`enqueue` consumes a device pointer, launches on the selected CUDA stream, and
returns a device output pointer. The successful enqueue path does not allocate,
synchronize, copy to the host, or inspect activation values on the CPU.
Diagnostic checkpoints remain on the device until the caller completes the
entire block. Setup computes RoPE constants and uploads weights, conditioning
and initial input. The caller owns terminal synchronization, readback, and
quiescence before resource destruction. One runner is not safe for concurrent
host dispatch or reuse of its scratch across in-flight streams.

`read_checkpoint` and `read_checkpoints` use blocking DtoH copies after the
terminal boundary. They are diagnostic APIs outside enqueue. The probe's
`terminal_syncs: 2` counts explicit synchronization calls: one after the chained
pair and one after its repeated-input guard. It does not count blocking copies.

The highest-risk errors are a wrong joint text/image mask, a different GGUF
scale/minimum interpretation, and a reduction-order discrepancy hidden by a
loose whole-block comparison. The gate therefore checks every projection,
normalization, RoPE, attention, residual, and MLP checkpoint on identical
inputs and weights against a pinned official Diffusers block on CUDA. TF32 is
disabled. Existing precision bounds are retained; any failed checkpoint blocks
promotion. Mixed images, invalid keys, non-power-of-two dimensions, and a
seeded comparator defect are required guards.

Definition of Done: build and execute the CUDA block probe on the RTX 5060 Ti;
compare all named checkpoints with upstream for the tiny fixture and the
actual mixed-quant block 0; check device-to-device chaining and stream capture
without intermediate synchronization. Record the executed commands, numerical
bounds and observed results here before a verified claim. Synthetic parity
alone does not establish real-weight or full-model parity.

## Executed gate

Host: Ubuntu 24.04.4, NVIDIA GeForce RTX 5060 Ti (16,311 MiB), compute capability
12.0, driver 595.104.02. Toolchain: Crystal 1.22.0 / LLVM 22.1.3, Python 3.12,
PyTorch 2.8.0+cu128. The host lacked pkg-config; an isolated pkgconf 2.5.1 build
under the task tools directory allowed Crystal to detect installed OpenSSL
3.0.13. No global package installation was needed.

The oracle executes official `QwenImage21TransformerBlock.forward`,
`QwenImage21Rope` and `apply_rotary_emb_qwen` from Diffusers commit
`8b3c707ebd3ec4881f4190cf42931da07eaf3b65` on CUDA, in F32 with TF32 disabled.
Intermediate values are captured by hooks around official operators. The real
fixture uses exact GGUF block-0 packed bytes for the native GPU kernels and an
F32 dequantization of those same weights for Diffusers. Weight export and
dequantization happen during fixture preparation, outside inference.

Pinned asset: `realrebelai/Qwen-Image-2.1_GGUFs` at
`8d393b750593a72ee040fe7d40611f479ccee679`, file `Qwen-Image-2.1-Q4.gguf`,
5,959,127,264 bytes, SHA-256
`51998ad7c068ce7d68e233237537900ffe874ab4d5c72e20758f5f18ceb15b8a`.
The filename is not a per-tensor format guarantee: block 0 uses Q8_0 for Q/K
and attention output, Q6_K for V/MLP output, Q5_K for gate/up, and F32 norms.
The export probe verifies the full file identity and reads byte counts as Int64.

Both upstream fixtures contain two text tokens and a 2x2 target-image grid,
including an invalid text key. Tiny dimensions are D=6, H=1, I=5; real block-0
dimensions are D=4096, H=32, I=12288. All 16 checkpoints passed:

| Checkpoint | Tiny max absolute error | Real block-0 max absolute error |
| --- | ---: | ---: |
| norm1 | 2.38419e-7 | 2.86102e-6 |
| q_proj | 3.57628e-7 | 3.05176e-5 |
| k_proj | 2.38419e-7 | 2.47955e-5 |
| v | 2.38419e-7 | 1.52588e-5 |
| q_norm | 1.78814e-7 | 4.76837e-6 |
| k_norm | 2.38419e-7 | 2.38419e-6 |
| q_rope | 1.41561e-7 | 4.76837e-6 |
| k_rope | 2.38419e-7 | 2.38419e-6 |
| attended | 2.38419e-7 | 1.14441e-5 |
| attn_out | 8.94070e-8 | 2.74658e-4 |
| state | 5.21541e-8 | 3.81470e-5 |
| norm2 | 3.57628e-7 | 3.81470e-5 |
| gate_up | 2.38419e-7 | 1.43051e-5 |
| activated | 8.94070e-8 | 1.37329e-4 |
| mlp_out | 4.28408e-8 | 5.61523e-3 |
| output | 5.96046e-8 | 5.79834e-4 |
| Two-block device chain | 8.94070e-8 | 6.71387e-4 |

Frozen bounds were not widened: tiny max_abs <= 2e-5; real max_abs < 1e-2 at
every checkpoint, with output cosine > 0.999999. Observed real output cosine
was 0.9999999999988148 and chained output cosine was 0.9999999999990593.
Same-input scratch reuse produced bitwise-identical output.

The CUDA specs passed 3 examples with no failures or pending tests. They cover
F32/BF16/Q8_0/Q5_K/Q6_K projection dispatch and graph capture/replay, an odd
output dimension and three input rows, NaN output poisoning before capture,
two resident blocks on a nondefault stream, changed-input replay, output alias
rejection, and the all-invalid-key guard. The multi-image fixture independently
checks a blocked future different-image key and an allowed future same-image
key against the existing CPU block. Existing CPU block and CUDA sequence
regressions passed 6 examples; the Python fixture/comparator suite passed 4.
The comparator rejects a seeded checkpoint/chain defect, tolerance widening,
missing checkpoints and non-finite values.

Commands executed from the task checkout on the CUDA host (tool directories
were added to PATH; `CRYSTAL_CACHE_DIR` pointed into the task directory):

```sh
export PATH="$PWD/tools/pkgconf-install/bin:$PWD/tools/crystal-1.22.0-1/bin:$PWD/venv/bin:$PATH"
export CRYSTAL_CACHE_DIR="$PWD/crystal-cache"
crystal build -Dcpu_only --release --error-trace bin/qwen_image21_cuda_block_probe.cr -o build/qwen_image21_cuda_block_probe
PYTHONDONTWRITEBYTECODE=1 python -m unittest spec/qwen_image21_cuda_reference_spec.py
COGNI_CUDA_IMAGE_TEST=1 crystal spec -Dcpu_only --error-trace spec/qwen_image21_cuda_projection_spec.cr spec/qwen_image21_cuda_execution_spec.cr
crystal spec -Dcpu_only --error-trace spec/cuda_resident_sequence_runner_spec.cr spec/qwen_image21_block_spec.cr
python scripts/qwen_image21_cuda_reference.py prepare-tiny fixtures/tiny-v2
python scripts/qwen_image21_cuda_reference.py oracle fixtures/tiny-v2
build/qwen_image21_cuda_block_probe --run fixtures/tiny-v2 candidates/tiny-v2
python scripts/qwen_image21_cuda_reference.py compare fixtures/tiny-v2 candidates/tiny-v2
python scripts/qwen_image21_cuda_reference.py prepare-real fixtures/real-block0
build/qwen_image21_cuda_block_probe --export-gguf weights/Qwen-Image-2.1-Q4.gguf fixtures/real-block0
python scripts/qwen_image21_cuda_reference.py oracle fixtures/real-block0
build/qwen_image21_cuda_block_probe --run fixtures/real-block0 candidates/real-block0
python scripts/qwen_image21_cuda_reference.py compare fixtures/real-block0 candidates/real-block0
```

Small numerical reports, hashed fixture manifests and source checksums are
retained at `/opt/mldata_ssd/cogni-ml/image-cuda-block-20261016/results/`;
large fixtures and the remote toolchain are in
`/root/cogni-image-cuda-block-20261016/` on the CUDA host. Real fixture manifest
SHA-256: `02ac98d2f3439fa1e0eecda0a934812477830600853bc4ba17890939d2ef915f`;
tiny: `5a9250f40b92a6e8f948ee03af399e6b9444badafcbd3e7807dafae45e48a00c`.
Comparator success alone is a numerical check, not GPU provenance attestation;
the executed oracle/probe commands above establish the device path for this run.

The targeted source review returned ROBUST for this bounded claim after the
first-key attention read and JSON byte-count overflow were corrected. It
checked enqueue, mask semantics, graph parameter lifetimes and readback scope;
the review did not itself execute GPU tests. These kernels prioritize numerical
correctness. Long sequences, the full 32-block model, production integration
and speedup are not established by this gate.

Rollback is removal of the opt-in runner/probe and their tests. No production
dispatch switch is introduced. A change to the weights, upstream revision,
mask, arithmetic, PTX, or driver requires refreshing the numerical gate.
