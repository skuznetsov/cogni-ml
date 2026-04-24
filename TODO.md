# TODO — Qwen 3.5/3.6 Native Metal Port

**Goal:** Native Metal inference for Qwen 3.5 9B and 3.6 27B (arch=qwen35), targeting ≥10% faster than llama.cpp HEAD on decode, with cosine≈1.0 correctness.

**Target model:** `Qwen3.5-9B-Q4_K_M.gguf` (primary debug target) → scale to `Qwen3.6-27B-Q4_K_M.gguf`.

**llama.cpp baseline source:** `~/SrcArchives/AI/llama.cpp` (HEAD).

**Protocol:** v13.2. Quadrumvirate on each phase start. LOGBOOK in conversation, persistent state in LANDMARKS.md.

---

## Phase 0 — Verification & infrastructure

- [x] **0.1** Create LANDMARKS.md + TODO.md
- [x] **0.2** Survey llama.cpp HEAD for: qwen35 arch implementation, DeltaNet (not Mamba), M-RoPE, Q4_K Metal kernels
- [x] **0.3** llama.cpp HEAD baseline: 9B Q4_K_M pp64=458, tg64=43.5 tok/s on M2 Max (FA=0)
- [x] **0.4** qwen35 hparams parsing: `get_int_array` helper + `Qwen35Hparams` struct; 5 spec examples passing

## Phase 1a — Q4_K CPU path

- [x] **1a.1** `Dequant.dequantize_q4_k` in `src/ml/gguf/dequant.cr` (block: d/dmin/scales[12]/qs[128] = 144 B)
- [x] **1a.2** `QuantMatmul.matmul_add_q4k` in `src/ml/gguf/quant_matmul.cr`
- [x] **1a.3** Correctness vs libggml: dequant bit-identical (cos=1.0, max_diff=0) on ssm_alpha/ffn_up; fused matmul cos≈1.0, max_diff≈2.4e-7 on 4096×4096

## Phase 1b — CPU-reference Qwen 3.5 9B forward

- [ ] **1b.1** `CausalDecoder` extensions for qwen35: partial M-RoPE with sections, GQA 4:1 broadcast, Mamba SSM layer, full-attn every 4 layers, SwiGLU FFN, RMSNorm
- [ ] **1b.2** LM head with Q4_K
- [ ] **1b.3** Correctness: cosine≈1.0 vs llama.cpp across all 32 layers on 1 token

## Phase 2 — Metal Q4_K kernels

- [ ] **2.1** `dequantize_q4_K_fn` in Metal (Q5_K pattern minus qh)
- [ ] **2.2** `simd_gemm_q4k` + `simd_gemm_q4k_moe` in `gemm_simd.metal`
- [ ] **2.3** `simd_mm_q4k` + `simd_mm_q4k_moe` + `batched_mm_q4k` in `gemm_mm.metal`
- [ ] **2.4** Correctness: cosine≈1.0 vs CPU Q4_K, speedup vs reference

## Phase 3a — Metal attention: head_dim=256 + GQA

- [ ] **3a.1** Read llama.cpp M-RoPE implementation — document exact formula (**BLOCKER for 3a**)
- [ ] **3a.2** Rewrite `attention_causal.metal`: dynamic head_dim (no fixed `float4 qr[16]`), kv-head broadcast
- [ ] **3a.3** Rewrite `attention_flash.metal`: same + flash version
- [ ] **3a.4** Partial M-RoPE fused in QKV
- [ ] **3a.5** Correctness: full-attn layer 9B cosine≈1.0, no regression on nomic BERT

## Phase 3b — Metal Mamba

- [ ] **3b.1** SoTA harvest: Mamba-2 paper, llama.cpp ggml-cuda/mamba.cu, reference selective_scan
- [ ] **3b.2** `selective_scan_fwd` Metal kernel
- [ ] **3b.3** SSM state management in BufferPool
- [ ] **3b.4** Correctness: SSM layer 9B cosine≈1.0

## Phase 3.5 — State save/load

- [x] **3.5.1** Serialize: KV-buffers (full-attn layers) + DeltaNet recurrent state (conv + SSM) + metadata
- [x] **3.5.2** Deserialize / restore
- [x] **3.5.3** Prompt KV cache: exact lookup by session/hash, artifact restore, fail-closed artifact hash/model checks
- [x] **3.5.4** pg_sorted_heap metadata adapter for prompt KV cache (external artifact path + sorted session/hash index)
- [x] **3.5.5** Correctness: exact top1 + full-logit close restore on 9B
- [x] **3.5.6** Longest-prefix prompt cache: token-hash prefix lookup + exact suffix replay

## Phase 4 — Optimization: beat llama.cpp

- [ ] **4.1** Fused RMSNorm + partial M-RoPE + QKV projection
- [ ] **4.2** Fused SwiGLU
- [ ] **4.3** Weight-tile cache in threadgroup memory
- [ ] **4.4** ComputeGraph wave scheduling for decoder
- [ ] **4.5** Benchmark: ≥10% faster than llama.cpp HEAD on 9B decode
- [x] **4.6** Exact first-run prefill shortcut: skip output RMSNorm/lm-head for non-final prompt tokens
- [x] **4.7** Port llama.cpp-style chunked fused DeltaNet scan primitive
- [x] **4.8** Port chunked recurrent prep primitives: conv-shift, alpha/beta, Q/K L2 norm
- [ ] **4.9** True layerwise/microbatch prefill for known prompt tokens
  - [x] Experimental CPU-orchestrated layerwise recurrent chunk path (`QWEN35_PREFILL_CHUNK=1`) with correctness gate
  - [x] Falsifier: CPU-orchestrated chunking is slower than current whole-token Metal decode wave; keep default off
  - [x] GPU-resident recurrent layer chunk primitive; default-on prefill chunking improves pp64 p50 from `52.87 tok/s` to `56.70 tok/s`
  - [x] Reuse Q4_K simdgroup-matrix GEMM inside GPU-resident recurrent prefill chunks; pp64 p50 improves from `56.70 tok/s` to `70.38 tok/s`
  - [x] Chunk full-attention prefill layers on Metal; pp64 p50 improves from `70.38 tok/s` to `143.18 tok/s`
  - [x] Enable Q5_K/Q6_K batch GEMM in prefill chunks with GPU-side F32↔F16 conversion; pp64 p50 improves from `143.18 tok/s` to `278.50 tok/s`
  - [x] Keep DeltaNet chunk state rows register-resident across the token scan; pp64 p50 improves from `278.50 tok/s` to `308.86 tok/s`
  - [x] Keep consecutive recurrent prefill layers GPU-resident; pp64 p50 improves from `308.86 tok/s` to `327.54 tok/s`
  - [x] Batch the final prompt token into prefill and run only fused lm-head top1; pp64 p50 improves from `308.82 tok/s` to `358.44 tok/s`
  - [x] Fast-path full Q4_K GEMM output tiles with direct simdgroup stores; pp64 p50 improves from `358.44 tok/s` to `373.80 tok/s`
  - [x] Fuse full-attention chunk boundaries into following recurrent runs; total pp64 syncs drop from `17` to `10`, warm profile improves `177.20 ms` → `172.53 ms` (small/noisy but positive paired A/B)
  - [x] Add prefill attribution harness and matmul shape counters; pp64 shows FFN up/gate Q4_K dominates logical weight traffic (`1728 MiB`), Q6/Q4 FFN-down follows (`1062 MiB` combined), and Q5/Q6 batch GEMM is mandatory (`~170 ms` default vs `~367 ms` with `QWEN35_Q56K_BATCH_GEMM_OFF=1`)
  - [x] In-place SwiGLU staging for chunked prefill FFN paths; exact aliasing path passes specs and small pp64 A/B improves default by `~0.48 ms` vs `QWEN35_SWIGLU_INPLACE_OFF=1`
  - [x] Falsifier: dual Q4_K gate/up+SwiGLU GEMM reduced dispatch count but was slower at pp64 (`174.78 ms` default vs `168.65 ms` off), likely from register pressure and extra barriers
  - [x] Specialize final full-attention prefill layer for last-row logits; final layer still writes all K/V but computes attention+FFN only for the final row, improving pp64 A/B from `173.86 ms` off to `170.00 ms` default
  - [x] Falsifier: direct SwiGLU-to-F16 activation staging for Q6 FFN-down was exact in focused specs but slower at pp64 (`170.02 ms` default vs `167.52 ms` off), so activation staging is not the current wall
  - [x] Falsifier: double-buffered Q6_K GEMM compiled and passed focused forward specs, but pp64 A/B was neutral (`168.22 ms` default vs `168.25 ms` off; p50 slightly worse), so Q6 barriers are not the next leverage point
  - [x] Raise default prefill chunk size from `64` to `1024`; pp256 A/B improves from `711.92 ms` p50 at chunk 64 to `522.76 ms` p50 default, while pp64 is unchanged within noise
  - [x] Batch the final long-prompt suffix instead of falling back to single-token final decode when prompt exceeds chunk size; pp2048 A/B improves from `4987.29 ms` p50 off to `4750.20 ms` default
  - [x] Raise default prefill chunk size again from `1024` to `2048`; pp2048 p50 improves from `4755.48 ms` at chunk 1024 to `4578.57 ms` at chunk 2048, while pp1024 is unchanged within noise
  - [x] Raise default prefill chunk size from `2048` to `4096` for long prompts; pp4096 one-shot p50 improves from `11862.78 ms` at chunk 2048 to `11524.63 ms` at chunk 4096
  - [x] Make default prefill chunk size memory-aware; 64GB M2 Max selects `8192` and pp8192 one-shot A/B improves from `35219.27 ms` at chunk 4096 to `33981.42 ms` default, while smaller systems fall back to `4096`/`2048`
  - [x] Add grouped command-buffer attribution for fused prefill waves; pp256 shows all `full+rec` groups are nearly flat (`~61.2-61.7 ms` wait) and `rec0-2` is smaller (`~46.4 ms`), so there is no single pathological layer group to attack
  - [ ] Next: attack FFN weight traffic only with lower-level Q4/Q6 tile changes or eliminate work; speculative/sparsity only behind eval harness

## Phase 5 — Scale to 27B

- [ ] **5.1** Run all tests on 27B Q4_K_M
- [ ] **5.2** mlock balance — avoid SIGBUS (page eviction) vs SIGKILL (wired memory pressure)
- [ ] **5.3** Final tok/s on 27B

## Phase VL — optional

- [ ] **V.1** CLIP vision encoder (27 blocks, 1152d, patch 16, 768×768 images)
- [ ] **V.2** qwen3vl_merger projector (1152 → 4096, spatial_merge=2)
- [ ] **V.3** Deepstack multi-layer injection
- [ ] **V.4** Image preprocessing pipeline
- [ ] **V.5** `VLAdapter` → `CausalDecoder.forward(embeddings, mrope_positions)` integration

---

## Correctness criterion (all phases)

Per-token cosine similarity vs llama.cpp HEAD on standard test prompt:
- Prefill 64 tokens from `"The quick brown fox jumps over the lazy dog. Describe this scene in detail:"`-style prompt
- Decode 64 tokens
- Target: cosine ≥ 0.9999 per logit vector

## Speed criterion

Wall-clock tok/s measured with `/usr/bin/time`:
- Prefill tok/s (full 64 tokens)
- Decode tok/s (averaged over 64 tokens)
- KV save/load round-trip time
- Baseline: llama.cpp HEAD with identical settings (no-flash-attention default, as in previous bench)

## Status

- **Active phase:** 4 (optimization: beat llama.cpp)
- **Active task:** 4.9 (true layerwise/microbatch prefill)
- **Baseline (llama.cpp 86db42e97):** 9B Q4_K_M pp64=458 / tg64=43.5 tok/s (FA=0) → targets ≥504 / ≥48
- **Blocked:** nothing
- **Last updated:** 2026-04-23
