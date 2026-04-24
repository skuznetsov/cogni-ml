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
- [ ] **4.7** True layerwise/microbatch prefill for known prompt tokens

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
- **Active task:** 4.7 (true layerwise/microbatch prefill)
- **Baseline (llama.cpp 86db42e97):** 9B Q4_K_M pp64=458 / tg64=43.5 tok/s (FA=0) → targets ≥504 / ≥48
- **Blocked:** nothing
- **Last updated:** 2026-04-23
