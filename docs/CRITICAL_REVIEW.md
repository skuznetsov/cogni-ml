# Critical Review

Date: 2026-02-02

This is a focused audit of correctness, buildability, and operational risks. It is based on static code inspection. Updated after fixes on 2026-02-02.

## Summary
Metal API mismatches and missing bridge code were addressed (bridge + device/dispatch wrappers added). CPU-only builds are supported via `-Dcpu_only`. Core tests now exist; attention autograd is implemented on the CPU path while the GPU path remains inference-only. LLM linking still requires native libs.

## Build blockers (resolved)
- Metal bridge added at `src/ml/metal/bridge.mm`.
- `Metal::Dispatch.execute` and encoder helpers implemented.
- `Shape#[]` and `Strides#[]` now support negative indices.

## Correctness gaps (remaining)
- Attention autograd is implemented on the CPU path; GPU path remains inference‑only.
- `DType` is still effectively F32-only (explicitly enforced now).
  - File: `src/ml/core/tensor.cr`

## Portability and ergonomics (medium severity)
- Default `Tensor` device is GPU unless compiled with `-Dcpu_only`; non‑Metal systems should use CPU-only build targets.
- No CUDA backend yet; NVIDIA support is planned but unimplemented.

## Testing gaps (medium severity)
- Minimal specs were added, but no CI or automated build for Metal is in place.

## Recommended fixes (prioritized)
1. Implement attention autograd on GPU or keep inference-only usage clearly documented.
2. Add a CUDA backend once test hardware is available.
3. Add CI for macOS Metal builds and CPU-only Linux/FreeBSD builds.
4. Keep LLM linkage requirements documented and tested in CI if possible.

## Minimal path to a CPU-only build
- Use `-Dcpu_only` or `make build_cpu` / `make spec_cpu`.
- GPU/Metal APIs are stubbed under `-Dcpu_only` to avoid linking `bridge.mm`.
- Default device resolves to CPU under `-Dcpu_only`; avoid calling GPU-only paths at runtime.
