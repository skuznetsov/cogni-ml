# Nomic llama.cpp Comparator Frontier SDD

Document status: implemented; M5 runtime verification pending
Current frontier: restore the b9960 comparator and fail closed around measured-red M5 Max matrix kernels
Bounded context: `src/ml/llm` bindings used by `bin/profile_nomic_vs_llama.cr`

## Admitted surface

- The Crystal FFI layout may track the installed b9960 `llama.h` exactly.
- The Nomic comparator may use one resident llama.cpp context after ABI size and runtime smoke checks pass.
- Performance comparison is admitted only after token IDs, pooling, output dimensions, and embedding parity pass.
- `Apple M5 Max` uses the scalar-SIMD Metal corridor by default until its
  `simdgroup_matrix` kernels pass the same parity gate; M2 default routing is unchanged.

## Rejected surface

- Calling by-value parameter APIs with a layout from another llama.cpp build.
- Treating a successful link or model load as ABI compatibility evidence.
- Publishing Metal latency as a win while embedding parity is red.
- Default routing into Nomic matrix GEMM, matrix attention, or batched matrix
  MoE on measured-red M5 Max hardware.

## Guard-only future

- A generated or C-shimmed binding can replace version-pinned Crystal layouts later.
- CLI/server comparison remains an independent oracle when the in-process ABI gate is red.
- `NOMIC_SIMDGROUP_MATRIX=on` permits an explicit M5 diagnostic probe; it is
  not a production recommendation until M5 parity is green.

## Design laws

- Current installed `llama.h` is authoritative for struct order, size, and function signatures.
- ABI mismatch must fail in a focused test or smoke before benchmark results are consumed.
- Correctness gates precede ABBA latency gates.

## Falsifier roster

- `spec/llama_ffi_abi_spec.cr`: b9960 context/model parameter sizes.
- `spec/nomic_metal_policy_spec.cr`: hardware default and explicit override policy.
- `bin/profile_nomic_vs_llama.cr --runs=1 --warmup=0`: model/context creation and one embedding per prompt class without crash; steady-state measurements use ABBA order.
- Metal/F32/llama.cpp cosine matrix: minimum accepted cosine is defined by the parity test added with the Metal fix.
- M2 forced-fallback Metal/F32 cosine: `0.999999547`, `0.999999640`,
  `0.999999720` for 20/54/196 tokens.

## Stop rules

- Stop benchmark interpretation on any ABI-size mismatch, crash, token mismatch, non-finite vector, or failed cosine threshold.
- Do not stop or perturb unrelated host processes to obtain a quiet benchmark.

## Implementation seals

- Slice: b9960 context-parameter ABI repair
  - Source/spec: `src/ml/llm/llama_ffi.cr`, `spec/llama_ffi_abi_spec.cr`
  - Falsifiers: exact struct sizes plus comparator runtime smoke
  - Boundary: b9960 layout only; future llama.cpp layout changes reopen this seal
- Slice: M5 Max matrix-corridor guard
  - Source/spec: `src/ml/gguf/nomic_metal_policy.cr`, `src/ml/gguf/metal_backend.cr`, `spec/nomic_metal_policy_spec.cr`
  - Falsifiers: M2 default/forced-off parity and one M5 default parity run
  - Boundary: M5 default is fail-closed; exact defective kernel remains unlocalized without M5 access
