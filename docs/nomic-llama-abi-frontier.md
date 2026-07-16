# Nomic llama.cpp Comparator Frontier SDD

Document status: GEMM/MoE verified on M5 Max; attention-tail repair verified on M2 and pending on M5
Current frontier: repaired b9960 comparator, race-free M5 Max GEMM/MoE, and guarded matrix-attention tails
Bounded context: `src/ml/llm` bindings used by `bin/profile_nomic_vs_llama.cr`

## Admitted surface

- The Crystal FFI layout may track the installed b9960 `llama.h` exactly.
- The Nomic comparator may use one resident llama.cpp context after ABI size and runtime smoke checks pass.
- Performance comparison is admitted only after token IDs, pooling, output dimensions, and embedding parity pass.
- `Apple M5 Max` uses matrix GEMM and batched matrix MoE by default after the
  shared-memory handoff race passed repeated, boundary, and mixed-batch parity.
- Matrix attention remains an independent fail-closed capability on M5 Max.

## Rejected surface

- Calling by-value parameter APIs with a layout from another llama.cpp build.
- Treating a successful link or model load as ABI compatibility evidence.
- Publishing Metal latency as a win while embedding parity is red.
- Default routing into M5 matrix attention before the guarded partial-tile path
  passes the same M5 boundary and mixed-batch parity gate.

## Guard-only future

- A generated or C-shimmed binding can replace version-pinned Crystal layouts later.
- CLI/server comparison remains an independent oracle when the in-process ABI gate is red.
- `NOMIC_MATRIX_ATTENTION=on` permits an explicit M5 attention diagnostic probe;
  it is not a production recommendation until the guarded tail loads pass on M5.

## Design laws

- Current installed `llama.h` is authoritative for struct order, size, and function signatures.
- ABI mismatch must fail in a focused test or smoke before benchmark results are consumed.
- Correctness gates precede ABBA latency gates.

## Falsifier roster

- `spec/llama_ffi_abi_spec.cr`: b9960 context/model parameter sizes.
- `spec/nomic_metal_policy_spec.cr`: hardware default and explicit override policy.
- `bin/profile_nomic_vs_llama.cr`: model/context smoke plus ABBA timing; every
  measured vector is checked and the command fails on token or cosine mismatch.
- Metal/F32/llama.cpp cosine matrix: minimum accepted cosine is defined by the parity test added with the Metal fix.
- M2 forced-fallback Metal/F32 cosine: `0.999999547`, `0.999999640`,
  `0.999999720` for 20/54/196 tokens.
- M5 Max default-vs-llama.cpp, 15 runs after 5 warmups: minimum reported cosine
  `0.999999`; native p50 `3.75`, `4.13`, `7.89` ms for 20/54/196 tokens.
- M5 default-vs-forced-fallback boundary/mixed-batch stress: 136 vectors over
  15/16/17, 31/32/33, 63/64/65, 196, 511/512-token and mixed-batch cases;
  minimum cosine `0.999978343`.
- M2 forced-on attention-tail runtime smoke: exact token IDs and minimum cosine
  above `0.999999` for 20/54/196-token prompts; this checks runtime MSL
  compilation and partial K/V tiles, but does not certify M5 behavior.

## Stop rules

- Stop benchmark interpretation on any ABI-size mismatch, crash, token mismatch, non-finite vector, or failed cosine threshold.
- Do not stop or perturb unrelated host processes to obtain a quiet benchmark.

## Implementation seals

- Slice: b9960 context-parameter ABI repair
  - Source/spec: `src/ml/llm/llama_ffi.cr`, `spec/llama_ffi_abi_spec.cr`
  - Falsifiers: exact struct sizes plus comparator runtime smoke
  - Boundary: b9960 layout only; future llama.cpp layout changes reopen this seal
- Slice: M5 Max matrix-corridor repair
  - Source/spec: `src/ml/gguf/nomic_metal_policy.cr`, `src/ml/gguf/metal_backend.cr`, `spec/nomic_metal_policy_spec.cr`
  - Root cause: matrix kernels reused shared staging memory for output before
    every simdgroup had completed its final tile read
  - Falsifiers: 144 depth/repetition comparisons, 15-run llama.cpp ABBA, and
    136 boundary/mixed-batch comparisons on M5 Max
  - Boundary: matrix attention remains fail-closed on M5 Max
- Slice: matrix-attention tail repair
  - Source/spec: `src/ml/gguf/kernels/attention_matmul.metal`,
    `spec/attention_matmul_tail_safety_spec.cr`
  - Root cause: the last 8-row K/V matrix load could cross the logical sequence
    allocation; guarded per-simdgroup scratch now zero-pads partial rows
  - Falsifiers: runtime MSL compile plus M2 llama.cpp parity are green; M5
    boundary and mixed-batch parity remain required before changing its default
