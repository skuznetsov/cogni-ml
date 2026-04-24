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
  - [x] Add scoped matmul attribution for prefill/decode phases; pp256 confirms `prefill.rec.ffn_upgate` dominates logical weight traffic (`1296 MiB`), followed by recurrent Q5 projection (`528 MiB`) and recurrent FFN-down (`796.5 MiB` combined Q6/Q4); decode prompt64/gen4 shows the same recurrent FFN/upgate and projection dominance (`5184 MiB` and `2112 MiB`)
  - [x] Harden prefill A/B attribution: `--compare-env` now runs paired interleaved trials and reports pair wins, avoiding block-order drift false positives
  - [x] Falsifier: lowering the GEMM route threshold so b8 prefill uses Q4/Q5/Q6 GEMM switches the routes correctly but slows pp8 (`140.70 ms` p50 default GEMV-at-b8 vs `158.74 ms` p50 GEMM-at-b8); keep `GEMM_BATCH_THRESHOLD=8`
  - [x] Falsifier: fusing final full-attn-last and lm-head top1 into one command buffer reduced pp64 syncs (`10` → `9`) but slowed wall reps in sequential release checks (`162.32 ms` p50 off vs `165.67 ms` p50 fused), so the removed sync is not worth the larger final command buffer
  - [x] Falsifier: a 256-thread Q4_K batch-64 GEMM reduced nominal batch tiles but did not survive benchmark adversary; same-process attribution was slightly positive (`164.51 ms` default vs `165.41 ms` off), but matched llama benchmark regressed (`378.72 tok/s` b64 vs `394.94 tok/s` current), so the branch was removed
  - [x] Falsifier: direct Q56_K FFN-down half-output add avoided an explicit `f16_to_f32` buffer but was neutral at pp64 (`167.22 ms` default vs `167.30 ms` branch p50, wins `4/8`), so conversion removal is not enough leverage
  - [x] Falsifier: llama.cpp-style single-buffer Q4_K GEMM with Q4-only `6144` byte threadgroup memory passed focused specs and improved standalone `4096x12288 b64` op attribution (`3.171 ms` → `2.975 ms`), but pp64 wall regressed (`167.39 ms` default p50 vs `176.01 ms` branch p50), so it is not a default prefill win
  - [x] Route tiny Q4_K prefill projections (`out_dim <= 64`, especially recurrent alpha/beta) through GEMV instead of underfilled 64-row GEMM tiles; pp64 paired A/B improves default from `164.04 ms` off to `156.14 ms` on (`8/8` wins), and matched llama comparison moves native prefill to `408.02 tok/s` p50 vs llama `463.3 tok/s`
  - [x] Falsifier: extending the tiny-Q4 GEMV rule to `out_dim <= 1024` makes pp64 slower (`165.86 ms` p50 at threshold 64 vs `169.08 ms` at threshold 1024, wins `8/8` for threshold 64), so keep the rule narrow
  - [x] Falsifier: routing small Q5/Q6 prefill projections (`out_dim <= 1024`) through GEMV instead of batch GEMM is also slower at pp64 (`165.07 ms` default vs `167.05 ms` GEMV, wins `8/8` for default)
  - [x] Disable Crystal GC during multi-token prefill hot path with `QWEN35_PREFILL_GC_GUARD_OFF=1` escape hatch; pp64 paired A/B improves default from `158.00 ms` off to `151.80 ms` on (`8/8` wins), and matched llama comparison moves native prefill to `421.41 tok/s` p50 vs llama `463.04 tok/s`
  - [x] Falsifier: direct Q5_K F32-input/F32-output batch GEMM compiled and passed focused specs, but pp64 paired A/B was neutral/negative (`152.56 ms` default vs `152.52 ms` branch p50, avg worse and wins `3/8`), so Q5 conversion removal is not enough leverage
  - [x] Falsifier: using fast command buffers for prefill helper/group paths compiled and passed specs, but pp64 was neutral (`152.02 ms` default vs `151.97 ms` fast, wins `2/8`) and pp256 slightly worse (`490.49 ms` default vs `490.79 ms` fast), so command-buffer creation is not the remaining gap
  - [x] Remove avoidable host churn in Qwen35 prefill boundaries and Q5/Q6 GEMM staging: shared-buffer reads no longer zero-fill before copy, Q56 zero bias buffers are cleared directly once, and Q56 F16 conversion scratch is reused by size; focused specs pass and pp64 smoke remains around `151.80 ms` p50
  - [x] Falsifier: adding a Q5/Q6 GEMM `no-bias` mode (`apply_gelu=2`) compiled and passed focused specs, but pp64 paired A/B was neutral/negative (`151.84 ms` default vs `151.80 ms` bias path off, wins `3/8`), so the bias add/read is not material enough to justify a kernel mode
  - [x] Add exact Q4_K half-input prefill GEMM (`QWEN35_Q4K_H16_GEMM_OFF=1` disables it): Q4 prefill matmuls preconvert F32 activations to F16 once per matmul instead of per output tile; pp64 paired A/B improves default by `~1.28 ms` (`8/8` wins), pp256 improves by `~5.06 ms` (`6/6` wins), and matched pp64 moves native prefill to `424.96 tok/s` p50 vs llama.cpp `462.9 tok/s`
  - [x] Add exact Q5_K recurrent-qkv half-output prefill path (`QWEN35_Q5_QKV_H16_CONV_OFF=1` disables it): Q5 qkv GEMM output stays F16 and the recurrent conv-shift chunk reads half directly; pp64 paired A/B improves default by `~0.28 ms` (`6/8` wins), pp256 by `~0.52 ms` (`5/6` wins)
  - [x] Reuse Q4_K half-input conversion across paired FFN gate/up matmuls for large prefill batches (`QWEN35_Q4K_PAIR_H16_GEMM_OFF=1` disables it, active at batch `>=256`): pp256 isolated paired A/B improves default by `~1.56 ms` avg / `~2.21 ms` p50 (`7/10` wins), pp512 opt-in A/B improved by `~1.82 ms` (`4/4` pair wins), and pp64 is gated off after adversary A/B showed the pair path is not reliable at short prompts
  - [x] Falsifier: sharing one H16 activation conversion between recurrent Q5 qkv and Q4 gate/z projections compiled and passed focused specs, but pp256 paired A/B was flat (`485.56 ms` default vs `485.59 ms` opt-in, wins `4/8`), so recurrent projection conversion reuse is not enough leverage
  - [x] Add conversion-kernel attribution to the prefill profile report so future quiet runs can separate duplicated `F32<->F16` activation/output traffic from quantized weight traffic; smoke pp16 report shows per-row percentages, matmul/conversion logical traffic totals, and mix (`97.50%` matmul / `2.50%` conversion)
  - [x] Add matched benchmark support for exact prompt-cache restore mode (`--native-prefill-cache`): after one seeded prefill, restoring the saved `.qkv` state measures pp64 p50 `52.28 ms` / `1224.28 tok/s` vs llama.cpp `432.72 tok/s` (`+182.93%`) under the same harness; this is a repeated/session prompt-cache speedup, not a first-run prefill replacement
  - [x] Harden benchmark quiet-mode checks with both per-process and aggregate CPU thresholds (`--load-warning-threshold`, `--load-total-warning-threshold`) so noisy multi-process desktops do not silently pass `--require-quiet`; aggregate warnings now print top CPU contributors
  - [x] Add `--wait-quiet-ms` / `--quiet-poll-ms` to Qwen35 benchmark harnesses so queued A/B runs can wait for a quiet host before measuring instead of immediately aborting or recording noisy timings
  - [x] Guarded baseline refresh: with `--require-quiet --wait-quiet-ms=60000`, pp64 attribution reports p50 `151.73 ms` / `421.80 tok/s`, traffic mix `90.69%` matmul / `9.31%` conversion; matched llama.cpp prompt64/gen64 reports native prefill `426.01 tok/s` p50 vs llama `461.90 tok/s` (`-7.77%`) and native decode `47.60 tok/s` p50 vs llama `45.35 tok/s` (`+4.96%`)
  - [x] Falsifier: removing the inner `simdgroup_barrier(mem_none)` calls from the Q4_K H16 prefill GEMM passed focused Qwen specs (`14 examples, 0 failures`) but regressed guarded pp64 attribution from baseline p50 `151.73 ms` to `154.49 ms`, so keep the barriers
  - [x] Add a guarded paired decode A/B harness for env-tuned wave experiments; guarded trials confirm `QWEN35_WAVE_CHUNK_LAYERS=2` beats unchunked `0` by `~1.34 ms/tok` (`6/6` wins), while `2` vs `4` and `1` vs `2` remain neutral/noisy, so keep default `2`
  - [x] Falsifier: writing Q4 FFN pair input as H16 directly from add+RMSNorm was exact in focused specs, but pp256 paired A/B was neutral (`506.53 ms` default vs `506.29 ms` opt-in, wins `5/10`), so the separate conversion kernel is not the current wall
  - [x] Relaxed-load baseline after CPU-core clarification: sequential pp64 attribution remains stable at p50 `150.80 ms` / `424.39 tok/s`, pp256 at p50 `486.56 ms` / `526.14 tok/s`; matched pp64/gen64 reports native prefill `422.41 tok/s` p50 vs llama.cpp `461.35 tok/s` (`-8.44%`) and native decode `46.59 tok/s` p50 vs llama.cpp `44.38 tok/s` (`+4.98%`)
  - [x] Falsifier: nearby decode scheduler/glue knobs do not recover the missing margin: `QWEN35_WAVE_CHUNK_LAYERS=3` is neutral vs default `2` (`4/8`, delta `+0.013 ms/tok`), `QWEN35_WAVE_FAST_CMD=1` is neutral vs off (`3/6`, delta `-0.018 ms/tok`), `QWEN35_REC_CONVSHIFT_FUSED=1` is slightly worse/noisy (`3/8`, delta `+0.025 ms/tok`), and `QWEN35_DN_POST_FUSED=1` is neutral (`3/8`, delta `-0.014 ms/tok`)
  - [x] Falsifier: matching llama.cpp's Q6_K GEMV `NR0=2` passed focused specs but slowed decode sync profile (`25.67 ms/tok` vs `24.60 ms/tok` old binary in same sequence), so keep local `MV6_NR0=1`
  - [x] Add native Q8_0 draft-model support for Qwen3.5 0.8B: CPU dequant/matmul, Metal GEMV/top1, tied-output fallback when `output.weight` is absent, focused Q8_0 Metal spec, and reusable `--model` sync profiler
  - [x] Add first greedy speculative acceptance harness for Qwen35 target/draft pairs; serial verifier is correctness-only, but establishes acceptance-rate evidence before building a batched verifier
  - [x] Draft baseline: llama.cpp Qwen3.5 0.8B Q8_0 tg64 `145.53 tok/s` (`6.87 ms/tok`), native Q8_0 top1 `11.99 ms/tok`; greedy exact acceptance smoke ranges `13.33%` to `57.14%`, so exact greedy speculative needs either a faster draft, a closer draft, or a sampling-acceptance path before it can beat plain target decode
  - [x] Fix Q8_0 GEMV attribution block accounting and retune Q8_0 GEMV row parallelism; `MV_Q8_NR0=1` improves native 0.8B Q8_0 draft decode from `11.59 ms/tok` to `9.06 ms/tok` in the same 64-token profile, while `NR0=3/4` regress
  - [x] Add exact chunked target top1 verifier primitive (`prefill_tokens_top1s`) and harden shared Metal constant caching for mixed target+draft models; regression spec covers 9B target after 0.8B draft cache pollution
  - [x] Falsifier: chunking only the target verifier body is not enough for speedup with the current 0.8B draft. On `def fibonacci(n):`, gamma=4/tokens=24 has the same acceptance (`85.19%`) but serial verifier is `31.47 ms/tok` while chunk-body verifier is `32.97 ms/tok`; target verification time drops (`503.9` → `471.5 ms`) but fork/copy and per-row lm-head overhead erase it
  - [x] Falsifier: batched Q6_K lm-head top1 for verifier rows is exact but slower at gamma=4. On `def fibonacci(n):`, chunk verifier default is `30.03 ms/tok`; opt-in `QWEN35_HEAD_TOP1_ROWS=1` regresses to `32.44 ms/tok`, so keep row-batched top1 default-off
  - [x] Add exact `chunk-inplace` verifier mode that mutates target state directly and only restores a backup on rejection; on `def fibonacci(n):`, gamma=4/tokens=24 improves verifier wall from serial `29.09 ms/tok` and copy-back chunk `29.13 ms/tok` to `28.36 ms/tok`, but still trails plain target decode (`21.91 ms/tok`)
  - [x] Retune Q8_0 draft GEMV row parallelism from `MV_Q8_NSG=2` to `4` and make Q8 top1 dispatch width follow the Q8 simdgroup count; 0.8B Q8_0 draft decode improves from `~8.88-9.07 ms/tok` to `~8.36-8.38 ms/tok`, while `NSG=8/16/32` are slower or unstable
  - [x] After Q8_0 `NSG=4`, exact speculative `chunk-inplace` on `def fibonacci(n):` improves from `28.36 ms/tok` to `27.70 ms/tok` at the same `85.19%` acceptance, but still trails plain target decode (`21.96 ms/tok`)
  - [x] Add exact Q8_0 FFN gate/up dual GEMV for the 0.8B draft decode wave; it halves FFN-upgate dispatch count and improves interleaved draft decode from `8.54/8.85 ms/tok` with `QWEN35_Q8_DUAL_GEMV_OFF=1` to `8.44/8.51 ms/tok` default, while speculative `chunk-inplace` smoke improves draft time `231.0` -> `223.2 ms` over 32 generated tokens
  - [x] Remove repeated speculative harness rollback allocations by reusing preallocated target/draft backup states and report rollback timing; default prompt `gamma=4/tokens=32` improves from `23.83` to `22.66 ms/tok`, and `gamma=16/tokens=64` proves the high-acceptance path can beat plain target (`15.67` vs `22.24 ms/tok`)
  - [x] Add opt-in adaptive gamma (`--adaptive`, `--max-gamma`) as an exploration harness; with no-regrow-after-reject default, it helps high-acceptance prompts (`19.34 ms/tok` vs plain `22.33`) and avoids the earlier rejection-heavy collapse on `def fibonacci(n):` (`28.55 ms/tok` vs fixed gamma=4 `28.99`, plain target `21.61`)
  - [x] Add exact first-candidate early reject before draft span generation: if `draft_next != target_next`, skip draft backup, skip the rest of the draft candidates, advance one corrected target token, and continue from the corrected draft state. On `def fibonacci(n):`, gamma=4/tokens=64 improves from `32.43 ms/tok` with `QWEN35_SPEC_EARLY_REJECT_OFF=1` to `28.99 ms/tok` default in the same built harness
  - [x] Promote rejection-aware adaptive gamma to the speculative harness default, with `--no-adaptive` / `QWEN35_SPEC_ADAPTIVE=0` for fixed-gamma A/B. In the same built harness, default prompt improves from fixed `23.39 ms/tok` to adaptive `19.30 ms/tok`, and `def fibonacci(n):` improves from fixed `29.59 ms/tok` to adaptive `28.60 ms/tok`
  - [x] Add exact gamma=1 accepted-token fast path for chunk verifiers: when adaptive has fallen to one candidate and `draft_next == target_next`, advance target and draft directly without rollback backups or chunk verifier. On `def fibonacci(n):`, interleaved A/B improved from `27.93/32.02 ms/tok` with `QWEN35_SPEC_SINGLE_FAST_OFF=1` to `27.61/26.72 ms/tok` default
  - [x] Add exact target-only fallback once rejection-aware adaptive drops to low gamma. With `QWEN35_SPEC_PLAIN_FALLBACK_GAMMA=2` default, `def fibonacci(n):` exits speculative after the first rejection and measures `21.57 ms/tok` vs plain target `22.89` in the same final run, while high-acceptance prompt remains faster than plain target (`20.93` vs `23.51 ms/tok`)
  - [x] Raise adaptive `max_gamma` default from `16` to `32` while keeping initial `gamma=4`: final default-prompt smoke is `19.11 ms/tok` vs plain target `21.96`, while `def fibonacci(n):` still falls back after rejection and measures `21.21 ms/tok` vs plain target `21.94`; starting at `gamma=8` was rejected because it regressed `def fibonacci(n):` to `26.56 ms/tok`
  - [x] Re-check Q4_K dual FFN gate/up GEMV as a lower-level decode-kernel branch and reject it again: focused forward spec stayed green (`13 examples, 0 failures`), but paired A/B lost `0.367 ms/tok` (`0/6` wins) versus the separate-GEMV path, confirming this is register-pressure/occupancy bound rather than dispatch-bound
  - [x] Skip draft-state advance/resync before imminent target-only fallback (`QWEN35_SPEC_SKIP_DRAFT_BEFORE_FALLBACK_OFF=1` disables): early-reject `def fibonacci(n):` avoids one unused draft advance, and chunk-reject prompts avoid unused resync (`Once upon a time`: `22.28` vs `22.85 ms/tok`; `The quick brown fox`: `22.56` vs `22.84 ms/tok`); high-acceptance prompt has `draft_skip=0` and stays unchanged within noise
  - [x] Falsifier: deriving a faster local Q4_K_M draft from the downloaded 0.8B Q8_0 GGUF is blocked by llama.cpp (`requantizing from type q8_0 is disabled`); the partial output artifact was removed, so Q4 draft testing needs an F16/BF16 source or a separately downloaded Q4 GGUF
  - [x] Falsifier/tuning note: initial `gamma=3` is not a safe default even though it is faster on the 100%-accept prompt (`~18.3 ms/tok` vs `~19.4`) and mixed partial-reject prompts (`Once upon a time` `21.56` vs gamma4 `22.42`), because it regresses `def fibonacci(n):` to `22.74 ms/tok`; `--max-gamma 16/24` did not recover the high-accept gain, so the effect is the initial chunk schedule, not just the cap
  - [x] Enable exact early-reject and gamma=1 accepted-token fast paths for `--verify serial` too: serial verifier is still not the default for high-accept prompts (`28.83 ms/tok` vs chunk `19.34`), but partial-reject profiling now avoids artificial draft/span waste (`Once upon a time` serial-fast `21.77` vs chunk `22.31`; `The quick brown fox` `21.77` vs chunk `22.41`)
  - [x] Add opt-in `--verify hybrid` mode: first speculative cycle uses serial verifier, later cycles use chunk-inplace. This helps first-cycle partial-reject prompts (`Once upon a time` `21.44` vs chunk `22.47`; `The quick brown fox` `21.73` vs `22.49`) but is not default because high-accept prompt slightly regresses (`19.32` vs `19.08`) and `def fibonacci(n):` regresses (`21.68` vs `21.50`)
  - [x] Falsifier: a progressive verifier that serial-probes 1/2/3 first-cycle candidates and chunks the rest preserved exactness, but it is not worth carrying: high-accept regressed (`19.41-19.70 ms/tok` vs chunk `19.04`), partial-reject prompts were weaker than `hybrid`, and `def fibonacci(n):` regressed (`21.67-21.90` vs chunk `21.51`); temporary code was removed
  - [x] Falsifier: Q8_0 mixed-size dual GEMV for draft recurrent `qkv + gate` lowers `rec.proj` encode accounting (`~3.00 ms` -> `~2.36-2.42 ms` over 16 tokens), but worsens end-to-end draft decode (`~7.85-7.88 ms/tok` separate GEMVs vs `~8.05-8.06 ms/tok` mixed), so this is another dispatch-saving/register-pressure tradeoff rather than a speed path
  - [x] Reuse the existing Q8_0 dual GEMV kernel for draft recurrent `alpha + beta` projections (`QWEN35_Q8_ALPHA_BETA_DUAL_GEMV_OFF=1` disables it): 0.8B draft sync profile improves from `8.77/8.90 ms/tok` off to `8.51/8.43 ms/tok` default over 64 tokens, while focused Qwen specs stay green
  - [x] Skip speculative draft-state backup when a future rejection is guaranteed to enter target-only fallback (`QWEN35_SPEC_SKIP_DRAFT_BACKUP_BEFORE_FALLBACK_OFF=1` disables): exact harness output remains matched to greedy target, and partial-reject prompts avoid the unused draft copy (`draft_backup_skip=1`, `draft_backup=0.0 ms`)
  - [x] Add guarded fast regrowth after adaptive gamma reaches `8` (`QWEN35_SPEC_FAST_REGROW_MIN_GAMMA=0` disables): high-accept prompt now reaches `gamma=32` in `5` cycles instead of `7`, improving `The capital of France is` from `~19.2 ms/tok` to `~16.0-17.8 ms/tok`; `def fibonacci(n):` stays on the safe path (`max_seen=4`, fallback after reject)
  - [x] Reuse the existing Q8_0 dual GEMV kernel for full-attention `K + V` projections (`QWEN35_Q8_KV_DUAL_GEMV_OFF=1` disables): draft full.qkv encode drops from `~3.7 ms` to `~2.7 ms` over 64 tokens and high-accept speculative decode gets a small exact win (`15.73 ms/tok` off -> `15.68/15.71 ms/tok` default)
  - [x] Falsifier: switching fallback-bound low-gamma verifier cycles from `chunk-inplace` to forked `chunk` removes target backup but is not a win; high-accept regresses (`~15.8` -> `~16.0 ms/tok`) and reject prompts are neutral, so keep `chunk-inplace` default
  - [x] Falsifier: raising initial speculative `gamma` from `4` to `5` or `6` is worse under the current fast-regrow schedule; high-accept prompts slow slightly, `def fibonacci(n):` regresses (`gamma=5` verifies one extra reject row, `gamma=6` breaks into many low-gamma cycles), and partial-reject prompts lose acceptance efficiency
  - [x] Falsifier: enabling batched lm-head top1 only for large verifier chunks (`>=8/16/24/32` rows) is still slower than per-row fused top1; high-accept target verifier rises from `~552-556 ms` to `~627-675 ms`, so leave `QWEN35_HEAD_TOP1_ROWS` default-off
  - [x] Refresh: lowering Q4_K pair H16 prefill threshold to b64 is still not a robust pp64 win; it halves FFN upgate conversion accounting but wall remains neutral/noisy (`~156.3-156.6 ms` default vs `~156.1-156.3 ms` branch under host load), so keep pair route gated at batch `>=256`
  - [x] Falsifier: switching Q5_K batch GEMM from the current double-buffered schedule to a llama-style single-buffer schedule is exact but slower on pp64; focused specs pass (`22 examples, 0 failures`), but paired A/B loses (`159.99 ms` default vs `161.55 ms` single-buffer p50, default wins `7/8`), so keep Q5 double-buffering
  - [x] Falsifier: skipping full-attention KV copies in speculative target rollback is exact but not faster; focused rollback spec passed, yet high-accept smoke stayed neutral/slightly worse (`16.30 ms/tok` skip-KV vs `16.20 ms/tok` full backup) and `target_backup` remained `10.6 ms`, showing rollback cost is recurrent state buffers rather than KV
  - [x] Falsifier: Q8_0 GEMV shared-X threadgroup staging is exact but slower for the 0.8B draft; the focused Q8 GEMV spec worsened (`~1.9 ms` -> `24.3 ms`) and draft decode regressed from `8.26` to `9.18 ms/tok`, so current Q8 speed is not limited by repeated x loads
  - [x] Falsifier: making speculative gamma regrow more aggressive is not a safe default. `QWEN35_SPEC_FULL_ACCEPT_STREAK=1`, `QWEN35_SPEC_FAST_REGROW_MIN_GAMMA=4`, and initial `--gamma 8` all stayed neutral/slower on the 100%-accept prompt (`~16.4-16.9 ms/tok` vs default `16.44`) and regressed `def fibonacci(n):` (`24.75-24.97 ms/tok` vs default `22.98`), so keep the conservative current schedule
  - [x] Add opt-in staged speculative verifier mode for high-accept prompts (`--verify staged --stage-gate 4`). It drafts/verifies a guard stage before the rest of a large gamma cycle and preserves exact greedy output; with fixed `--gamma 32`, high-accept `The capital of France is` improves from default `16.45` to `15.04 ms/tok`, but rejection-sensitive `def fibonacci(n):` regresses to `32.58 ms/tok`, so the mode stays opt-in and not default
  - [x] Falsifier: Q6 lm-head top1 tile rows `12 -> 16` is correctness-safe but not a verifier win; focused specs passed (`13 examples, 0 failures`), yet high-accept target verifier stayed flat (`~573.8 ms` default, `~478-480 ms` staged) and reject default worsened (`22.84 -> 23.45 ms/tok`), so keep rows12
  - [x] Falsifier/fix: full-row batched Q6 lm-head verifier is fast but not exact because its F16 GEMM path can flip close argmax rows (`row24`: exact `220@8.900501` vs F16 full-row `198@8.8984375`). Keep it opt-in with `QWEN35_HEAD_FULL_ROWS=1`; default verifier returned to exact per-row fused top1 and a rows32 ambiguous regression spec now covers this case
  - [x] Add exact Q5_K/Q6_K prefill GEMM f32-output kernels that preserve the previous half-rounded output contract but skip the separate half-output → f32 conversion dispatch; also vectorize Q4_K H16 input tile loads. Focused specs pass (`23 examples, 0 failures`), standalone pp64 op attribution drops measured weighted hot-shape lower bound from `8.209 ms` to `2.913 ms`, while full pp64/pp256 wall remains host/noise limited around `153 ms` / `494 ms`
  - [x] Add matched benchmark mode for preallocated prompt state (`--native-prefill-prealloc`): it allocates GPU state once, resets only recurrent conv/SSM state between reps, and preserves exact `start_pos=0` output (`fresh=72:9.439041`, `reuse=72:9.439041`). Prompt64/gen64 smoke improves native prefill from `165.90 ms` p50 default to `156.27 ms` p50 preallocated, moving the matched gap from `-3.10%` to `-1.57%`; this is session-style state reuse/harness attribution, not a first-run kernel breakthrough.
  - [x] WBA/LTP next-frame saved: Qwen35 already uses wave scheduling, so the next breakthrough target is not "add WBA" but lift WBA from command-buffer groups to algebraic operator diamonds. Prime candidate is FFN `normed -> gate/up -> SwiGLU -> down -> residual` as a tile-streamed exact diamond, plus speculative verifier as a time-axis WBA batch.
  - [x] Falsifier: extending the existing in-place SwiGLU activation trick from prefill into decode wave is correctness-safe but not a default win. `QWEN35_DECODE_SWIGLU_INPLACE=1` passes targeted specs (`17 examples, 0 failures`) but paired decode A/B loses/noisy versus default (`25.135 ms/tok` default vs `25.786 ms/tok` opt-in; default wins `5/8`), so keep it default-off and treat full tile-streaming FFN as the real diamond, not buffer aliasing.
  - [x] Promote exact row-batched verifier top1 only for large chunks (`QWEN35_HEAD_TOP1_ROWS_MIN`, default `8`; `QWEN35_HEAD_TOP1_ROWS_OFF=1` disables): targeted specs pass (`17 examples, 0 failures`), high-accept speculative A/B wins `4/4` against rows-off (`16.29/21.04/21.13/19.90 ms/tok` auto vs `17.63/21.30/22.10/23.93 ms/tok` off), while rejection-heavy paths with `gamma=4` stay below the threshold and use the old per-row exact top1 path.
  - [ ] Next: target true batched speculative verification, FFN tile-streaming WBA diamond, and lower-level Q4/Q6 tile-prepack/kernel changes; Q8_0 draft is now closer, but exact speculative still trails plain target decode until verifier overhead is cut

## Deferred research backlog — efficient attention / long context

- [ ] **R.1** FlashAttention-style exact full-attention prefill kernel for Qwen35 full-attn layers; first add intra-group phase attribution and only implement if `full.attn` is material at pp2048+.
- [ ] **R.2** DeepSeek Sparse Attention / NSA-style indexer experiment; requires calibration + sparse adaptation, not an exact drop-in for existing Qwen weights.
- [ ] **R.3** Linear-attention replacement track; treat as new architecture or distillation/continued-training project, not an inference-only optimization.
- [ ] **R.4** Training-free long-context inference hacks under eval gates: attention sinks / sliding window, SnapKV/H2O-like KV retention, and KV quantization. Keep all default-off until LongBench/RULER/top-logit drift evidence exists.

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
- **Latest guarded benchmark:** prompt64/gen64 reps=5/warmup=2 with `--require-quiet --wait-quiet-ms=60000`: native prefill `426.01 tok/s` p50 vs llama.cpp `461.90 tok/s` (`-7.77%`), native decode `47.60 tok/s` p50 vs llama.cpp `45.35 tok/s` (`+4.96%`)
- **Blocked:** nothing
- **Last updated:** 2026-04-24
