# LANDMARKS — Qwen 3.5/3.6 Native Metal Port

Persistent landmark graph for the Qwen port project.
Format per v13.2: `[LM-name]: claim (how verified) {F/G/R} [status]`
Rich landmarks include full State/Relations/Evidence structure.

## Active Landmarks

### [LM-claude-1] Qwen 3.6 27B architecture verified
**status:** verified
**trust:** {F:0.95, G:high, R:0.95}
**context:** ml (Qwen port)
**evidence:**
- claim: "arch=qwen35, 64 layers, 5120 dim, 24/4 GQA, head_dim=256, full_attn_interval=4"
  source: `python3` GGUF metadata read on `Qwen3.6-27B-Q4_K_M.gguf`
  verified_at: 2026-04-22
  decay_trigger: model file replaced
- claim: "VL model — mmproj-Qwen3.6-27B-BF16.gguf present (931 MB)"
  source: ls `~/.cache/lm-studio/models/lmstudio-community/Qwen3.6-27B-GGUF/`
  verified_at: 2026-04-22
  decay_trigger: user deletes mmproj file

### [LM-claude-2] Architecture gap vs current cogni-ml
**status:** verified
**trust:** {F:0.85, G:narrow, R:0.85}
**context:** ml
**evidence:**
- claim: "current attention kernels overflow at head_dim>64 (float4 qr[16] fixed)"
  source: `docs/cogniformerus-native-model.md` Section 6, GPT-5.4 review findings
  verified_at: 2026-04-22
  decay_trigger: attention kernels rewritten
- claim: "current kernels assume n_heads == n_kv_heads (no GQA)"
  source: same (GPT-5.4 findings)
  verified_at: 2026-04-22
  decay_trigger: GQA added to kernels
- claim: "cogni-ml has zero Mamba/SSM infrastructure"
  source: grep `mamba|ssm|selective_scan` in src/ → 0 results
  verified_at: 2026-04-22
  decay_trigger: Фаза 3b completes

### [LM-claude-3] GEMM landscape in cogni-ml
**status:** verified
**trust:** {F:0.9, G:medium, R:0.9}
**context:** ml (Metal kernels)
**evidence:**
- claim: "3 GEMM paths: gemm_q5k (scalar tiled), simd_gemm_q5k (simdgroup gemv), simd_mm_q5k (simdgroup_matrix 8x8 with double-buffered shmem)"
  source: src/ml/gguf/kernels/gemm_q5k.metal, gemm_simd.metal, gemm_mm.metal
  verified_at: 2026-04-22
  decay_trigger: kernels refactored
- claim: "simd_mm_q5k adapted from llama.cpp kernel_mul_mm with improvements: double-buffered barriers (1 instead of 2 per iter), cooperative output write (128 threads vs 32)"
  source: gemm_mm.metal:1, gemm_mm.metal:238, gemm_mm.metal:256
  verified_at: 2026-04-22
  decay_trigger: N/A
- claim: "Q4_K support MISSING in all GEMM paths and in quant_matmul.cr"
  source: grep Q4_K in src/ml/gguf/ — only in reader.cr enum (type 12, 144 bytes/block)
  verified_at: 2026-04-22
  decay_trigger: Q4_K implemented

### [LM-claude-4] Qwen 3.5 9B architecture verified
**status:** verified
**trust:** {F:0.95, G:high, R:0.95}
**context:** ml
**evidence:**
- claim: "arch=qwen35 (identical to 27B), 32 layers, 4096 dim, 16/4 GQA, head_dim=256, full_attn_interval=4, rope.dimension_sections=[11,11,10,0], rope.dim_count=64, freq_base=10M"
  source: `python3` GGUF metadata on `Qwen3.5-9B-Q4_K_M.gguf`
  verified_at: 2026-04-22
  decay_trigger: model file replaced
- claim: "9B is ALSO VL — mmproj-Qwen3.5-9B-BF16.gguf present (456M params, BF16)"
  source: ls `~/.cache/lm-studio/models/lmstudio-community/Qwen3.5-9B-GGUF/`
  verified_at: 2026-04-22
  decay_trigger: user deletes mmproj
- claim: "mmproj: CLIP vision, 27 blocks, 1152d, image 768x768 patch 16, projection_dim=4096, qwen3vl_merger, spatial_merge=2, deepstack (27-bool array)"
  source: GGUF metadata on mmproj file
  verified_at: 2026-04-22
  decay_trigger: mmproj replaced

### [LM-QUADRUMVIRATE-1] Vision в дизайне декодера
**status:** decided
**trust:** {F:0.7, G:0.8, R:0.75}
**context:** ml (architecture decision)
**decision:** "Modal-agnostic decoder с M-RoPE в ядре, vision adapter отдельным модулем"
**cassandra:** "IMPLICIT_ASSUMPTION_BROKEN — 9B тоже VL, MISSING_INVARIANT если RoPE не готов к M-RoPE изначально"
**daedalus:** "observation shift: граница модальности = после embed-phase; всё после работает на (embeddings, positions)"
**maieutic:** "STUMBLE — точная формула partial M-RoPE в text-only режиме неизвестна, нужен llama.cpp код"
**adversary:** "обычный RoPE + патч потом = BROKEN (переписать все attention). CLIP сразу = VULNERABLE (over-engineering). modal-agnostic + M-RoPE = ROBUST."
**blocker:** "прочитать ggml-rope / Qwen3-VL M-RoPE в llama.cpp ДО Фазы 3a"
**evidence:**
- claim: "rope.dimension_sections=[11,11,10,0] значит M-RoPE T/H/W/reserved, 64 dims partial RoPE на 25% head_dim=256"
  source: GGUF metadata 9B and 27B
  verified_at: 2026-04-22
  decay_trigger: N/A

### [LM-QWEN35-DRAFT-08B] Qwen 3.5 0.8B Q8_0 draft model support
**status:** verified
**trust:** {F:0.9, G:medium, R:0.9}
**context:** ml (speculative decode)
**evidence:**
- claim: "0.8B draft GGUF uses arch=qwen35, Q8_0 tensors, 24 layers, 1024 dim, 8/2 GQA, full_attention_interval=4, vocab=248320, and omits output.weight."
  source: local GGUF metadata read on `~/.cache/lm-studio/models/lmstudio-community/Qwen3.5-0.8B-GGUF/Qwen3.5-0.8B-Q8_0.gguf`
  verified_at: 2026-04-24
  decay_trigger: model file replaced
- claim: "Native Qwen35 loader and Metal top1 path can run the 0.8B Q8_0 model with tied lm-head fallback."
  source: `tmp_qwen08_smoke.cr` output `hp layers=24 dim=1024 heads=8/2 output=Q8_0 1024x248320`, `top=198 logit=12.774761`
  verified_at: 2026-04-24
  decay_trigger: Q8_0 matmul/tied-output path changed
- claim: "Focused Q8_0 Metal GEMV correctness passes: cosine=1.0, max|Δ|=7.4505806e-8 on 1024→3584."
  source: `CRYSTAL_CACHE_DIR=/tmp/cogni_ml_crystal_cache_q8_spec2 crystal spec spec/qwen35_metal_spec.cr spec/qwen35_forward_spec.cr --link-flags=...` → `20 examples, 0 failures`
  verified_at: 2026-04-24
  decay_trigger: Q8_0 kernel or quant matmul changed

### [LM-SPECULATIVE-08B-1] 0.8B draft speculative acceptance baseline
**status:** verified
**trust:** {F:0.85, G:medium, R:0.85}
**context:** ml (speculative decode)
**evidence:**
- claim: "llama.cpp runs Qwen3.5 0.8B Q8_0 draft at tg64=145.53 tok/s (~6.87 ms/tok) and pp64=3291 tok/s on this M2 Max."
  source: `/Users/sergey/SrcArchives/AI/llama.cpp/build/bin/llama-bench -m Qwen3.5-0.8B-Q8_0.gguf -p 64 -n 64 -r 5 -ngl 99 -fa 0 -t 8 -o json`
  verified_at: 2026-04-24
  decay_trigger: llama.cpp rebuild, model file replaced, power/thermal state changes
- claim: "Native 0.8B Q8_0 top1 decode is correct but currently slower than llama.cpp: 11.99 ms/tok (~83.4 tok/s), with 100% logical traffic in Q8_0 matmuls."
  source: `/tmp/qwen35_sync_profile_qwen08 --model Qwen3.5-0.8B-Q8_0.gguf 64 64` with `QWEN35_PROFILE_TOP1=1`
  verified_at: 2026-04-24
  decay_trigger: Q8_0 kernel retuned or decode scheduler changed
- claim: "Q8_0 GEMV attribution must use 32-element blocks, not QK_K=256; after the fix, 16 draft tokens report 8067.56 MiB logical matmul traffic instead of 1008.45 MiB."
  source: `/tmp/qwen35_sync_profile_qwen08_profilefix --model Qwen3.5-0.8B-Q8_0.gguf 16 16` with `QWEN35_PROFILE_TOP1=1`
  verified_at: 2026-04-24
  decay_trigger: profile accounting changed
- claim: "Q8_0 GEMV `NR0=1` beats `NR0=2/3/4` for 0.8B draft decode: 9.06 ms/tok vs 11.59/13.53/14.58 ms/tok in same 64-token sync-profile loop."
  source: bounded local retune loop rebuilding `bin/qwen35_sync_profile.cr` for `MV_Q8_NR0 ∈ {1,2,3,4}` and running `--model Qwen3.5-0.8B-Q8_0.gguf 64 64`
  verified_at: 2026-04-24
  decay_trigger: Q8_0 kernel, scheduler, or model shape changes
- claim: "Greedy exact draft acceptance with 9B target + 0.8B draft is prompt-sensitive and low/moderate in short smoke prompts: 13.33%, 40.48%, 57.14%, 48.72%, 51.35%."
  source: `/tmp/qwen35_speculative_accept --gamma 4 --tokens 16/24 ...`
  verified_at: 2026-04-24
  decay_trigger: draft/target model, tokenizer, decoding mode, or prompt suite changes
- claim: "Shared Metal constant-cache writes must be keyed by buffer and source data identity, not only by a semantic tag; mixed 9B target + 0.8B draft runs can otherwise reuse stale constants after scratch/model changes."
  source: regression `keeps chunk verifier constants model-specific across target and draft models` in `spec/qwen35_forward_spec.cr`
  verified_at: 2026-04-24
  decay_trigger: ConstCache or Scratch implementation changes
- claim: "Chunking the target verifier body alone is exact after the ConstCache fix but not a speed win yet: on `def fibonacci(n):`, gamma=4/tokens=24, serial verifier reports 31.47 ms/tok and chunk verifier reports 32.97 ms/tok with the same 85.19% acceptance."
  source: `/tmp/qwen35_speculative_accept_modes --verify serial|chunk --gamma 4 --tokens 24 "def fibonacci(n):"`
  verified_at: 2026-04-24
  decay_trigger: draft GEMV, target verifier, state fork/copy, or lm-head batching changes
- claim: "Batched Q6_K lm-head top1 over verifier rows is exact but slower for gamma=4: default chunk verifier is 30.03 ms/tok; opt-in `QWEN35_HEAD_TOP1_ROWS=1` is 32.44 ms/tok."
  source: `/tmp/qwen35_speculative_accept_batched_head2 --verify chunk --gamma 4 --tokens 24 "def fibonacci(n):"` with and without `QWEN35_HEAD_TOP1_ROWS=1`
  verified_at: 2026-04-24
  decay_trigger: batched top1 kernel retuned, gamma changes, or verifier state-copy overhead removed
- claim: "`chunk-inplace` exact verifier avoids copy-back on fully accepted cycles and is the fastest current verifier mode, but still not a net speedup: serial 29.09 ms/tok, copy-back chunk 29.13 ms/tok, chunk-inplace 28.36 ms/tok, plain target 21.91 ms/tok on the same gamma=4/tokens=24 prompt."
  source: `/tmp/qwen35_speculative_accept_inplace --verify serial|chunk|chunk-inplace --gamma 4 --tokens 24 "def fibonacci(n):"`
  verified_at: 2026-04-24
  decay_trigger: state fork/copy, draft GEMV, acceptance, or verifier rollback strategy changes
- claim: "Q8_0 draft GEMV row parallelism `MV_Q8_NSG=4` is the best tested setting after dispatch-width hardening: NSG=2 gives ~8.88-9.07 ms/tok, NSG=4 gives ~8.36-8.38 ms/tok, NSG=8 gives ~8.78-8.80 ms/tok, and NSG=16/32 regress."
  source: rebuild sweep of `bin/qwen35_sync_profile.cr` on Qwen3.5-0.8B-Q8_0 with `QWEN35_PROFILE_TOP1=1`, prefill64/decode64
  verified_at: 2026-04-24
  decay_trigger: Q8_0 GEMV kernel, top1 dispatch, or Metal scheduler changes
- claim: "With Q8_0 NSG=4, exact speculative `chunk-inplace` improves to 27.70 ms/tok on `def fibonacci(n):` at 85.19% acceptance, but still trails plain target decode at 21.96 ms/tok."
  source: `/tmp/qwen35_speculative_accept_after_q8nsg4 --verify chunk-inplace --gamma 4 --tokens 24 "def fibonacci(n):"`
  verified_at: 2026-04-24
  decay_trigger: draft speed, verifier mode, acceptance, or target decode changes
- claim: "A Q8_0-only FFN gate/up dual GEMV is an exact small win for the 0.8B draft decode wave: interleaved 64-token top1 profiles changed from fallback `8.54/8.85 ms/tok` to default `8.44/8.51 ms/tok`, while `rec.ffn_upgate` phase time dropped from `6.04/7.68 ms` to `3.69/3.71 ms` and `full.ffn_upgate` from `2.20/2.78 ms` to `1.35/1.59 ms`."
  source: `CRYSTAL_CACHE_DIR=/tmp/cogni_ml_crystal_cache_q8_dual_bench QWEN35_PROFILE_TOP1=1 crystal run bin/qwen35_sync_profile.cr --link-flags=... -- --model Qwen3.5-0.8B-Q8_0.gguf 64 64`, interleaved `QWEN35_Q8_DUAL_GEMV_OFF=1` vs default
  verified_at: 2026-04-24
  decay_trigger: Q8_0 GEMV kernel, FFN decode route, or Metal scheduler changes
- claim: "The same Q8_0 dual-GEMV path only slightly improves the current exact speculative smoke because target verification dominates: default `chunk-inplace` changed from `23.96 ms/tok` to `23.83 ms/tok` on 32 generated tokens, with draft time `231.0 ms` -> `223.2 ms` and target verify time roughly unchanged (`483.2` -> `484.8 ms`)."
  source: `CRYSTAL_CACHE_DIR=/tmp/cogni_ml_crystal_cache_q8_dual_specacc crystal run bin/qwen35_speculative_accept.cr --link-flags=... -- --tokens 32 --gamma 4`, `QWEN35_Q8_DUAL_GEMV_OFF=1` vs default
  verified_at: 2026-04-24
  decay_trigger: draft speed, verifier mode, acceptance, or target decode changes
- claim: "Preallocating and reusing speculative rollback states removes repeated fork allocations in the harness and exposes rollback copy cost. On the default prompt at gamma=4/tokens=32, `chunk-inplace` improved from `23.83 ms/tok` to `22.66 ms/tok`; remaining measured backup copies were about `16.0 ms` target and `5.1 ms` draft over 8 cycles."
  source: `CRYSTAL_CACHE_DIR=/tmp/cogni_ml_crystal_cache_spec_prealloc crystal run bin/qwen35_speculative_accept.cr --link-flags=... -- --tokens 32 --gamma 4`
  verified_at: 2026-04-24
  decay_trigger: speculative state layout, backup strategy, or harness timing changes
- claim: "High-acceptance speculative decode is already a potential exact win when gamma is large enough: default prompt gamma=16/tokens=64 measured `15.67 ms/tok` speculative versus `22.24 ms/tok` plain target, but the same gamma on `def fibonacci(n):` fell to `43.25 ms/tok` because acceptance dropped to `51.35%` and draft resync cost rose."
  source: `bin/qwen35_speculative_accept.cr --tokens 64 --gamma 16` on default prompt and `def fibonacci(n):`
  verified_at: 2026-04-24
  decay_trigger: draft model, acceptance distribution, verifier chunking, or resync strategy changes
- claim: "Opt-in adaptive gamma with conservative two-full-cycle growth and no-regrow-after-reject keeps the high-acceptance win while avoiding the earlier rejection-heavy collapse: default prompt `--adaptive --gamma 4 --max-gamma 16` measured `19.34 ms/tok` versus plain `22.33`, while `def fibonacci(n):` measured `28.55 ms/tok` versus fixed gamma=4 `28.99` and plain target `21.61`."
  source: `/tmp/qwen35_speculative_accept_adapt_regrow --tokens 64 --gamma 4 --max-gamma 16 --adaptive`, default prompt and `def fibonacci(n):`
  verified_at: 2026-04-24
  decay_trigger: adaptive policy, acceptance predictor, verifier, or resync strategy changes
- claim: "First-candidate early reject is an exact win for rejection-heavy speculative chunks when performed before draft span generation: if `draft_next != target_next`, the harness can skip draft backup, skip the remaining draft candidates, advance one corrected target token, and continue from the corrected draft state. On `def fibonacci(n):` gamma=4/tokens=64, a built harness improved from `32.43 ms/tok` with `QWEN35_SPEC_EARLY_REJECT_OFF=1` to `28.99 ms/tok` default, with `early_rejects=2`; target verifier time dropped `1397.9 -> 1236.5 ms`, draft time `514.5 -> 473.6 ms`, and draft resync `123.3 -> 108.0 ms`."
  source: `/tmp/qwen35_speculative_accept_early_final --tokens 64 --gamma 4 "def fibonacci(n):"`, `QWEN35_SPEC_EARLY_REJECT_OFF=1` vs default
  verified_at: 2026-04-24
  decay_trigger: speculative verifier control flow, target_next semantics, or rejection distribution changes
- claim: "Rejection-aware adaptive gamma is now the better harness default than fixed gamma=4 on measured prompts. In the same built harness, default prompt improved from fixed `23.39 ms/tok` to adaptive `19.30 ms/tok`; `def fibonacci(n):` improved from fixed `29.59 ms/tok` to adaptive `28.60 ms/tok`. Fixed gamma remains available through `--no-adaptive` or `QWEN35_SPEC_ADAPTIVE=0`."
  source: `/tmp/qwen35_speculative_accept_adaptive_default --tokens 64 --gamma 4 [--no-adaptive]`, default prompt and `def fibonacci(n):`
  verified_at: 2026-04-24
  decay_trigger: adaptive policy, prompt acceptance distribution, verifier mode, or draft model changes
- claim: "A gamma=1 accepted-token fast path is an exact win once adaptive has fallen to one candidate: if `draft_next == target_next`, the harness can advance target and draft directly without target/draft rollback backups or chunk verifier. On `def fibonacci(n):`, interleaved A/B improved from `27.93/32.02 ms/tok` with `QWEN35_SPEC_SINGLE_FAST_OFF=1` to `27.61/26.72 ms/tok` default; `single_fast=24` and backup timing roughly halved."
  source: `/tmp/qwen35_speculative_accept_single_fast_ab --tokens 64 --gamma 4 "def fibonacci(n):"`, interleaved `QWEN35_SPEC_SINGLE_FAST_OFF=1` vs default
  verified_at: 2026-04-24
  decay_trigger: adaptive policy, verifier control flow, target_next semantics, or draft model changes
- claim: "Exact target-only fallback is the right behavior after rejection-aware adaptive gamma has fallen to low gamma: at that point speculation adds draft cost with little batching upside. With `QWEN35_SPEC_PLAIN_FALLBACK_GAMMA=2` default, `def fibonacci(n):` generated 59 of 64 tokens through plain target after the first rejection and measured `21.57 ms/tok` versus plain target `22.89`; high-acceptance default prompt remained faster than plain target at `20.93` versus `23.51 ms/tok` with `plain_fallback=0`."
  source: `/tmp/qwen35_speculative_accept_fallback_final --tokens 64 --gamma 4`, default prompt and `def fibonacci(n):`
  verified_at: 2026-04-24
  decay_trigger: fallback policy, adaptive policy, prompt acceptance distribution, verifier mode, or draft model changes
- claim: "Adaptive `max_gamma=32` is a better default than 16 for high-acceptance prompts while preserving the rejection-heavy fallback behavior. With initial `gamma=4`, the default prompt measured `19.11 ms/tok` versus plain target `21.96`, while `def fibonacci(n):` still fell back after rejection and measured `21.21 ms/tok` versus plain target `21.94`. Starting at `gamma=8` was rejected because it improved high-acceptance speed but regressed `def fibonacci(n):` to `26.56 ms/tok`."
  source: `/tmp/qwen35_speculative_accept_max32_default --tokens 64 --gamma 4`, default prompt and `def fibonacci(n):`, plus `--gamma 8 --max-gamma 32` adversary run
  verified_at: 2026-04-24
  decay_trigger: adaptive policy, prompt acceptance distribution, verifier mode, or draft model changes
- claim: "When a rejection will immediately enter target-only fallback, updating draft state is unused exact work. Skipping early-reject draft advance and chunk-reject draft resync preserves greedy target output and improves rejection prompts: `Once upon a time` measured `22.28` vs `22.85 ms/tok`, `The quick brown fox` `22.56` vs `22.84`, and early-reject `def fibonacci(n):` avoids one draft advance. The high-acceptance default prompt has `draft_skip=0` and stays unchanged within noise."
  source: `/tmp/qwen35_speculative_accept_skipdraft2 --tokens 64 --gamma 4`, with and without `QWEN35_SPEC_SKIP_DRAFT_BEFORE_FALLBACK_OFF=1`, on 2026-04-24
  verified_at: 2026-04-24
  decay_trigger: early-reject policy, fallback policy, draft resync semantics, or adaptive gamma changes
- claim: "A faster Q4_K_M draft cannot be produced from the currently downloaded 0.8B Q8_0 GGUF with local llama.cpp: `llama-quantize ... Q4_K_M` fails with `requantizing from type q8_0 is disabled`. The partial `.local.gguf` artifact was removed."
  source: `/Users/sergey/SrcArchives/AI/llama.cpp/build/bin/llama-quantize Qwen3.5-0.8B-Q8_0.gguf Qwen3.5-0.8B-Q4_K_M.local.gguf Q4_K_M` on 2026-04-24
  verified_at: 2026-04-24
  decay_trigger: availability of F16/BF16 0.8B source GGUF, separately downloaded Q4 draft, or llama.cpp requantization policy changes
- claim: "Initial `gamma=3` is a useful opt-in but not a safe default: repeated high-acceptance runs improved from about `19.40 ms/tok` at gamma4 to `18.28-18.40`, and mixed partial-reject prompts improved (`Once upon a time` `21.56` vs gamma4 `22.42`), but `def fibonacci(n):` regressed to `22.74 ms/tok`. `--max-gamma 16/24` with initial gamma4 did not reproduce the high-accept gain."
  source: `/tmp/qwen35_speculative_accept_skipdraft2 --tokens 64 --gamma {3,4} [--max-gamma 16|24]`, prompts `The capital of France is`, `Once upon a time`, `The quick brown fox`, `def fibonacci(n):`, on 2026-04-24
  verified_at: 2026-04-24
  decay_trigger: adaptive gamma policy, verifier chunk implementation, prompt acceptance distribution, or draft model changes
- claim: "The exact early-reject and gamma=1 fast paths should apply to `--verify serial` as well as chunk verifiers. Enabling them makes serial useful for partial-reject profiling (`Once upon a time` serial-fast `21.77 ms/tok` vs chunk `22.31`, `The quick brown fox` `21.77` vs chunk `22.41`) while confirming serial is still bad for high-accept prompts (`28.83` vs chunk `19.34`)."
  source: `/tmp/qwen35_speculative_accept_serialfast --tokens 64 --gamma 4 --verify serial|chunk-inplace`, prompts `Once upon a time`, `The quick brown fox`, `def fibonacci(n):`, `The capital of France is`, on 2026-04-24
  verified_at: 2026-04-24
  decay_trigger: serial verifier control flow, early-reject policy, or chunk verifier performance changes
- claim: "Opt-in `--verify hybrid` uses serial verification for the first speculative cycle and chunk-inplace afterwards. It helps first-cycle partial-reject prompts (`Once upon a time` `21.44 ms/tok` vs chunk `22.47`, `The quick brown fox` `21.73` vs `22.49`) but is not a default because high-acceptance prompt slightly regresses (`19.32` vs `19.08`) and `def fibonacci(n):` regresses (`21.68` vs `21.50`)."
  source: `/tmp/qwen35_speculative_accept_hybrid --tokens 64 --gamma 4 --verify hybrid|chunk-inplace`, prompts `The capital of France is`, `Once upon a time`, `The quick brown fox`, `def fibonacci(n):`, `Today I want to`, on 2026-04-24
  verified_at: 2026-04-24
  decay_trigger: verifier mode policy, serial verifier performance, chunk verifier performance, or prompt acceptance distribution changes
- claim: "A progressive first-cycle verifier that serial-probes 1/2/3 candidates and chunks the rest is exact but not worth carrying. High-acceptance prompt regressed (`19.41-19.70 ms/tok` vs chunk `19.04`), partial-reject prompts were weaker than `hybrid`, and `def fibonacci(n):` regressed (`21.67-21.90` vs chunk `21.51`). The temporary code was removed."
  source: temporary `/tmp/qwen35_speculative_accept_progressive --tokens 64 --gamma 4 --verify progressive --progressive-probe 1|2|3`, same prompts, on 2026-04-24
  verified_at: 2026-04-24
  decay_trigger: verifier implementation, prompt acceptance distribution, or progressive verifier redesign

### [LM-PREFILL-PREALLOC-1] Preallocated prompt state benchmark mode
**status:** verified
**trust:** {F:0.85, G:medium, R:0.85}
**context:** ml (prefill benchmarking)
**evidence:**
- claim: "`--native-prefill-prealloc` is exact for repeated `start_pos=0` prompt timing when recurrent conv/SSM state is reset and full-attention K/V rows are left allocated; prompt rows are overwritten before read."
  source: temporary `build/tmp/qwen35_prealloc_correctness.cr` output `fresh=72:9.439041 seed=72:9.439041 reuse=72:9.439041`
  verified_at: 2026-04-24
  decay_trigger: prefill state layout, full-attention start_pos semantics, or recurrent state reset changes
- claim: "Prompt64/gen64 smoke with the same built benchmark improves native prefill p50 from `165.90 ms` default to `156.27 ms` preallocated, moving matched llama.cpp gap from `-3.10%` to `-1.57%`; decode remains faster than llama.cpp in both runs."
  source: `/tmp/benchmark_qwen_vs_llama_prealloc --prompt=64 --gen=64 --warmup=1 --reps=3` with and without `--native-prefill-prealloc`
  verified_at: 2026-04-24
  decay_trigger: benchmark harness, host load, prefill state allocation behavior, or llama.cpp rebuild changes
**note:** This is a fair session-style measurement and attribution tool, not a first-run prefill kernel optimization. It shows allocation/state setup costs are measurable but not the remaining breakthrough lever.

### [LM-WBA-FFN-DIAMOND-1] Qwen35 WBA should move from group waves to operator diamonds
**status:** active hypothesis with one falsified sub-branch
**trust:** {F:0.75, G:medium, R:0.8}
**context:** ml (WBA/LTP performance)
**evidence:**
- claim: "Qwen35 already uses WBA-style wave scheduling: full decode wave, recurrent prefill runs, and full-attention-plus-recurrent groups. The next useful WBA level is algebraic operator diamonds, not merely more command-buffer grouping."
  source: `src/ml/gguf/qwen35_metal.cr` functions `forward_decode_wave`, `recurrent_layer_chunk_project_many`, and `full_attn_then_recurrent_chunk_project_many`; `src/ml/metal/compute_graph.cr` Block Integrity partition logic
  verified_at: 2026-04-24
  decay_trigger: Qwen35 scheduler or ComputeGraph rewritten
- claim: "The primary exact operator-diamond candidate is FFN: `normed -> gate/up -> SwiGLU -> down -> residual`. A real win likely needs tile-streaming across the diamond rather than only aliasing the activation buffer."
  source: pp64 attribution in `/tmp/qwen35_prefill_attr` showed FFN up/gate and down dominate logical weight traffic; decode wave source materializes `ffn_gate`, `ffn_up`, activation, and `ffn_down`
  verified_at: 2026-04-24
  decay_trigger: FFN route, matmul kernels, or attribution counters changed
- claim: "A simple decode-wave in-place SwiGLU sub-branch is exact but not a default speed win. Opt-in `QWEN35_DECODE_SWIGLU_INPLACE=1` passes targeted Qwen specs (`17 examples, 0 failures`), but paired decode A/B reports default `25.135 ms/tok` vs opt-in `25.786 ms/tok`, default wins `5/8`."
  source: `QWEN35_DECODE_SWIGLU_INPLACE=1 CRYSTAL_CACHE_DIR=/tmp/cogni_ml_crystal_cache_wba_decode_optin_on_spec crystal spec spec/qwen35_forward_spec.cr spec/qwen35_delta_net_spec.cr --link-flags=...`; `/tmp/qwen35_ab_profile_wba2 --env=QWEN35_DECODE_SWIGLU_INPLACE --a='<unset>' --b=1 --prompt=64 --gen=32 --trials=8 --warmup=1`
  verified_at: 2026-04-24
  decay_trigger: decode wave, SwiGLU kernel, FFN-down GEMV, or host load changes
**decision:** Keep decode in-place SwiGLU default-off as a research knob. Do not count it as the WBA breakthrough; next branch should prototype true tile-streamed FFN or batched speculative verifier.

### [LM-SPEC-VERIFY-ROWS-1] Large speculative verifier chunks should use exact row-batched top1
**status:** verified
**trust:** {F:0.85, G:medium, R:0.8}
**context:** ml (speculative decode / WBA verifier)
**evidence:**
- claim: "Exact row-batched Q6 top1 is now a default win for large verifier chunks but should stay off for small `gamma=4` chunks. The policy uses `QWEN35_HEAD_TOP1_ROWS_MIN` default `8`, with `QWEN35_HEAD_TOP1_ROWS_OFF=1` as an escape hatch."
  source: `src/ml/gguf/qwen35_cpu.cr` `output_project_top1s_routed`; targeted specs `CRYSTAL_CACHE_DIR=/tmp/cogni_ml_crystal_cache_headrows_auto_spec crystal spec spec/qwen35_forward_spec.cr spec/qwen35_delta_net_spec.cr --link-flags=...` -> `17 examples, 0 failures`
  verified_at: 2026-04-24
  decay_trigger: verifier head kernels, Q6 lm-head quantization, or speculative verifier policy changes
- claim: "On high-accept prompt `The capital of France is`, interleaved rows>=8 auto vs rows-off won all 4 pairs: auto `16.29/21.04/21.13/19.90 ms/tok`; off `17.63/21.30/22.10/23.93 ms/tok`. The largest gain comes from lower `target_verify` time on gamma>=8 chunks."
  source: `/tmp/qwen35_speculative_accept_headrows_auto --tokens 64 --gamma 4 --max-gamma 32 --verify chunk-inplace 'The capital of France is'`, interleaved with `QWEN35_HEAD_TOP1_ROWS_OFF=1`, on 2026-04-24
  verified_at: 2026-04-24
  decay_trigger: host load, draft speed, adaptive gamma policy, or head top1 kernel changed
- claim: "Reject-heavy prompts that fall back after a `gamma=4` cycle do not use the new batched path under the default threshold, preserving the old per-row verifier behavior."
  source: prompt suite with `def fibonacci(n):`, `Once upon a time`, and `The quick brown fox`; all reported `max_seen=4` and `plain_fallback` after the first cycle
  verified_at: 2026-04-24
  decay_trigger: adaptive gamma or fallback policy changes
**decision:** Keep full-row F16 GEMM top1 default-off due exactness risk, but enable exact tile-reduce row-batched top1 for chunks at or above the WBA threshold.

## Graph Visualization

```
[LM-claude-1] Qwen 3.6 27B arch
       ↓ builds_on
[LM-claude-4] Qwen 3.5 9B arch ←─── (same architecture, both VL)
       ↓ contrasts
[LM-claude-2] Gap analysis (head_dim, GQA, DeltaNet missing)
       ↓ informs
[LM-QUADRUMVIRATE-1] Vision decision (modal-agnostic + M-RoPE in core)
       ↑ uses
[LM-claude-3] GEMM landscape (Q5/Q6 done, Q4_K missing)
       ↓ ref
[LM-claude-Q4K-KERNEL-REF] Q4_K Metal kernels in llama.cpp
       ↓
[LM-claude-QWEN35-HPARAMS] 9B SSM dimensions (d_inner=4096, 24 DeltaNet + 8 full-attn)
       ↓
[LM-claude-CADENCE-1] layer cadence formula ((i+1)%4 != 0 → recurrent)
       ↓ decomposes into
 ┌─────────────┴─────────────┐
 ↓                           ↓
[LM-claude-FULLATTN-STRUCTURE]   [LM-claude-DELTANET-1]
 Q+gate, QK-norm, M-RoPE,          build_qkvz + conv1d + silu
 attn_post_norm, SwiGLU            + L2norm + build_delta_net(AR)
       ↓                           ↓
[LM-claude-MROPE-1] M-RoPE formula (NeoX pairing, sectioned pos)
```

### [LM-claude-MROPE-1] M-RoPE formula from llama.cpp kernel_rope_multi
**status:** verified
**trust:** {F:0.9, G:high, R:0.9}
**context:** ml (Metal kernels / attention)
**evidence:**
- claim: "M-RoPE partitions rotation dims across 4 position channels via sections=[sect_0..3]. sect_dims=sum(sections). sector=(ic)%sect_dims. theta_base=pos[i2 + ne02*k] where k=0/1/2/3 based on sector range. Only dims i0 < n_dims rotated; rest copied."
  source: `ggml-metal.metal:4438-4518` kernel_rope_multi template
  verified_at: 2026-04-22
  decay_trigger: llama.cpp changes ggml_rope_multi signature
- claim: "NeoX-style pairing: x0=src[0], x1=src[n_dims/2]; dst[0]=x0*cos-x1*sin, dst[n_dims/2]=x0*sin+x1*cos"
  source: same kernel, lines 4505-4509
  verified_at: 2026-04-22
  decay_trigger: N/A
- claim: "For Qwen 3.5/3.6 text-only decoding: positions pos[i2+0*ne02], pos[i2+1*ne02], pos[i2+2*ne02] all equal actual sequence position, so effectively degenerates to standard RoPE on first n_dims=64 of head_dim=256"
  source: reasoning from M-RoPE design (VL-mode differentiates only when batching vision+text)
  verified_at: 2026-04-22
  decay_trigger: confirm by reading llama-graph.cpp ubatch.pos construction

### [LM-claude-DELTANET-1] DeltaNet dispatch and autoregressive formula
**status:** verified
**trust:** {F:0.9, G:high, R:0.9}
**context:** ml (Qwen port)
**evidence:**
- claim: "build_delta_net dispatches by n_tokens: n_tokens==1 → autoregressive (or fused_gdn_ar); n_tokens>1 → chunking (or fused_gdn_ch)"
  source: `delta-net-base.cpp:423-445`
  verified_at: 2026-04-22
  decay_trigger: N/A
- claim: "Autoregressive (decode) formula: s ← s*exp(g); sk ← sum_rows(s*k); d ← (v - sk^T)*b; s ← s + k*d^T (outer product after repeat); o ← sum_rows(s*q)"
  source: `delta-net-base.cpp:288-370` build_delta_net_autoregressive
  verified_at: 2026-04-22
  decay_trigger: N/A
- claim: "State shape: [S_v, S_v, H_v, n_seqs]; output shape: [S_v, H_v, n_tokens, n_seqs]"
  source: same
  verified_at: 2026-04-22
  decay_trigger: N/A

### [LM-claude-CADENCE-1] Full-attention vs recurrent layer cadence
**status:** verified
**trust:** {F:0.95, G:high, R:0.95}
**context:** ml
**evidence:**
- claim: "recurrent_layer_arr[i] = ((i + 1) % full_attention_interval != 0). With interval=4 and 32 layers: full-attn at il ∈ {3,7,11,15,19,23,27,31} (every 4th, 0-indexed), remaining 24 are DeltaNet"
  source: `llama-model.cpp:2794-2796` (qwen35 case) and `llama-hparams.cpp:196` is_recurrent
  verified_at: 2026-04-22
  decay_trigger: N/A
- claim: "For 27B (64 layers): 16 full-attn, 48 recurrent (same pattern il∈{3,7,...,63})"
  source: extrapolation from same cadence formula
  verified_at: 2026-04-22
  decay_trigger: N/A

### [LM-claude-QWEN35-HPARAMS] Qwen 3.5 9B SSM/DeltaNet dimensions
**status:** verified
**trust:** {F:0.95, G:high, R:0.95}
**context:** ml
**evidence:**
- claim: "ssm.conv_kernel=4 | ssm.state_size=128 (head_k_dim, also head_v_dim) | ssm.group_count=16 (num_k_heads) | ssm.time_step_rank=32 (num_v_heads) | ssm.inner_size=4096 (d_inner)"
  source: `python3` GGUF metadata on Qwen3.5-9B-Q4_K_M.gguf
  verified_at: 2026-04-22
  decay_trigger: model file replaced
- claim: "Derived: head_v_dim = d_inner/num_v_heads = 4096/32 = 128. S_k==S_v==128. num_v_heads != num_k_heads (32 vs 16, ratio 2:1) → needs ggml_repeat if not fused"
  source: arithmetic on above + qwen35.cpp:209,329-333
  verified_at: 2026-04-22
  decay_trigger: N/A
- claim: "Per-layer recurrent state: ssm_state [128,128,32] = 524288 floats (2MB fp32), conv_states [3,8192] = 96KB. 24 recurrent layers → ~50MB total SSM state per sequence"
  source: derivation
  verified_at: 2026-04-22
  decay_trigger: N/A

### [LM-claude-FULLATTN-STRUCTURE] Qwen 3.5 full-attention layer structure
**status:** verified
**trust:** {F:0.9, G:high, R:0.9}
**context:** ml
**evidence:**
- claim: "wq emits COMBINED Q+gate: output dim = (n_embd_head*2)*n_head = 256*2*16 = 8192. View-0 (Q): stride 2*head_dim per head, offset 0. View-1 (gate): stride 2*head_dim per head, offset head_dim. Gate applied AFTER attention: cur = attn_out * sigmoid(gate)"
  source: `qwen35.cpp:129-156,186-190`
  verified_at: 2026-04-22
  decay_trigger: N/A
- claim: "Separate wk, wv with LoRA: out_dim = head_dim*n_head_kv = 256*4 = 1024 each. QK-norm applied via attn_q_norm / attn_k_norm (RMSNorm) BEFORE M-RoPE. Then M-RoPE on first 64 dims of head_dim=256 (partial, n_rot=64)"
  source: `qwen35.cpp:141-172`
  verified_at: 2026-04-22
  decay_trigger: N/A
- claim: "attn_post_norm (RMSNorm) BETWEEN attention output and FFN (in addition to attn_norm pre-attention). FFN = SwiGLU via LLM_FFN_SILU, LLM_FFN_PAR (build_ffn): up + silu(gate) combined, then down. Residual pattern: inpSA saved → attn → +inpSA → ffn_residual save → attn_post_norm → ffn → +ffn_residual"
  source: `qwen35.cpp:27-72,376-381`
  verified_at: 2026-04-22
  decay_trigger: N/A

### [LM-claude-Q4K-KERNEL-REF] Q4_K Metal kernel reference from llama.cpp
**status:** verified
**trust:** {F:0.9, G:high, R:0.9}
**context:** ml (Metal kernels)
**evidence:**
- claim: "Q4_K block = {d:half, dmin:half, scales[12]:u8, qs[128]:u8} = 144 bytes. 256 elements per block (QK_K). Dequant via get_scale_min_k4_just2(is, k, scales) → (sc, min); mask 0x0F for il<2, 0xF0 for il>=2"
  source: `ggml-metal.metal:680-697` dequantize_q4_K template
  verified_at: 2026-04-22
  decay_trigger: N/A
- claim: "kernel_mul_mm_q4_K_f32 uses half8x8 simdgroup matrix: template instantiation at ggml-metal.metal:10077. Same infra as kernel_mul_mm_q5_K_f32 that we adapted."
  source: `ggml-metal.metal:10077`
  verified_at: 2026-04-22
  decay_trigger: N/A
- claim: "kernel_mul_mv_q4_K_f32 (GEMV for decode, batch=1) at ggml-metal.metal:7716-7824. N_R0_Q4_K template for row-parallelism."
  source: `ggml-metal.metal:7716-7824`
  verified_at: 2026-04-22
  decay_trigger: N/A

### [LM-prefill-FR-FUSION-1] Full-attention to recurrent boundary fusion
**status:** verified
**trust:** {F:0.85, G:narrow, R:0.85}
**context:** ml (Qwen35 prefill)
**evidence:**
- claim: "Qwen35 prefill now fuses each full-attention chunk followed by a recurrent run into one Metal command buffer, preserving exact layer order/state updates while removing seven CPU read/write/sync boundaries at pp64."
  source: `src/ml/gguf/qwen35_cpu.cr`, `src/ml/gguf/qwen35_metal.cr`
  verified_at: 2026-04-24
  decay_trigger: prefill scheduler or Qwen35 layer cadence changes
- claim: "Correctness gate passed: `crystal spec spec/qwen35_metal_spec.cr spec/qwen35_forward_spec.cr spec/qwen35_delta_net_spec.cr spec/qwen35_state_snapshot_spec.cr spec/qwen35_prompt_cache_spec.cr --link-flags=\"$(pwd)/build/bridge.o -framework Metal -framework Foundation -lc++\"` -> 27 examples, 0 failures."
  source: local command output
  verified_at: 2026-04-24
  decay_trigger: Metal kernels, state layout, or Qwen35 forward path changes
- claim: "Warm prefill profile syncs drop from 17 to 10 and wall improves from 177.20 ms with `QWEN35_PREFILL_FUSE_FULL_REC_OFF=1` to 172.53 ms default; post-fix paired native A/B won 9/12 with p50 174.77 ms vs 179.81 ms, but the standard llama comparison run was noisy and did not monotonically improve."
  source: local paired A/B, warm profile scripts, and `bin/benchmark_qwen_vs_llama.cr`
  verified_at: 2026-04-24
  decay_trigger: benchmark harness, power state, or kernel scheduler changes
**adversary:** "Effect size is small and near system noise; keep env-off fallback and treat this as a boundary-overhead reduction, not a kernel breakthrough. Earlier A/B before layer-unique constant buffers is stale."

### [LM-prefill-ATTRIBUTION-1] Prefill matmul shape attribution
**status:** verified
**trust:** {F:0.9, G:narrow, R:0.9}
**context:** ml (Qwen35 prefill)
**evidence:**
- claim: "`bin/qwen35_prefill_attribution.cr` profiles pp prefill and reports matmul route/quant/shape/batch counts plus logical weight traffic from the exact `encode_matmul` route."
  source: `src/ml/gguf/qwen35_metal.cr`, `bin/qwen35_prefill_attribution.cr`
  verified_at: 2026-04-24
  decay_trigger: prefill scheduler or matmul route selection changes
- claim: "At pp64, dominant logical weight traffic is `q4_gemm Q4_K 4096x12288 b64` with 64 calls / 1728 MiB; Q6 down-projection traffic is `q6_gemm Q6_K 12288x4096 b64` with 16 calls / 630 MiB; Q5 recurrent QKV traffic is 24 calls / 528 MiB."
  source: `crystal run bin/qwen35_prefill_attribution.cr -- --prompt=64 --warmup=1 --reps=3 --compare-env=QWEN35_Q56K_BATCH_GEMM_OFF`
  verified_at: 2026-04-24
  decay_trigger: model quant mix, batch size, or route selection changes
- claim: "Q5/Q6 batch GEMM remains mandatory: same attribution run measured default p50 around 170.72 ms versus 367.23 ms with `QWEN35_Q56K_BATCH_GEMM_OFF=1`."
  source: same command
  verified_at: 2026-04-24
  decay_trigger: Q5/Q6 kernels or benchmark environment changes
**decision:** "Next exact optimization should target FFN traffic/staging first; more Q5/Q6 fallback experimentation is refuted for pp64."

### [LM-prefill-SWIGLU-INPLACE-1] In-place SwiGLU staging for chunked prefill
**status:** verified
**trust:** {F:0.85, G:narrow, R:0.85}
**context:** ml (Qwen35 prefill)
**evidence:**
- claim: "Chunked prefill FFN paths can write `silu(gate) * up` back into the up buffer and feed that buffer to FFN-down; the Metal elementwise kernel is index-local, so this aliasing is exact."
  source: `src/ml/gguf/kernels/ffn_qwen35.metal`, `src/ml/gguf/qwen35_metal.cr`
  verified_at: 2026-04-24
  decay_trigger: SwiGLU kernel indexing or FFN buffer ownership changes
- claim: "Correctness gate passed after enabling the in-place default: qwen35 targeted specs -> 27 examples, 0 failures."
  source: `crystal spec spec/qwen35_metal_spec.cr spec/qwen35_forward_spec.cr spec/qwen35_delta_net_spec.cr spec/qwen35_state_snapshot_spec.cr spec/qwen35_prompt_cache_spec.cr --link-flags=\"$(pwd)/build/bridge.o -framework Metal -framework Foundation -lc++\"`
  verified_at: 2026-04-24
  decay_trigger: prefill FFN path changes
- claim: "A/B with `bin/qwen35_prefill_attribution.cr -- --prompt=64 --warmup=2 --reps=6 --compare-env=QWEN35_SWIGLU_INPLACE_OFF` measured default avg 170.98 ms vs off avg 171.47 ms."
  source: local command output
  verified_at: 2026-04-24
  decay_trigger: benchmark harness or power state changes
**adversary:** "Effect size is tiny; keep `QWEN35_SWIGLU_INPLACE_OFF=1` fallback and do not treat this as attacking the dominant weight-read wall."

### [LM-prefill-Q4-SWIGLU-GEMM-FALSIFIER] Dual Q4_K gate/up+SwiGLU GEMM
**status:** refuted
**trust:** {F:0.8, G:narrow, R:0.85}
**context:** ml (Qwen35 prefill)
**evidence:**
- claim: "A fused Q4_K kernel that computes gate/up GEMMs together and writes SwiGLU directly reduced logical dispatch count, but slowed pp64."
  source: temporary `simd_mm_q4k_swiglu_f32` branch in `src/ml/gguf/kernels/gemm_q4k.metal` and `src/ml/gguf/qwen35_metal.cr`
  verified_at: 2026-04-24
  decay_trigger: Q4_K GEMM tile design or Apple GPU register/threadgroup limits change
- claim: "A/B with `bin/qwen35_prefill_attribution.cr -- --prompt=64 --warmup=2 --reps=6 --compare-env=QWEN35_Q4K_SWIGLU_GEMM_OFF` measured fused default avg 174.78 ms vs off avg 168.65 ms."
  source: local command output
  verified_at: 2026-04-24
  decay_trigger: benchmark harness, power state, or kernel implementation changes
**decision:** "Do not retry the same dual-accumulator Q4_K SwiGLU GEMM shape; if revisited, change the frame to a lower-register design or fused down-projection tiling."

### [LM-prefill-FINAL-FULL-LAST-1] Final full-attention last-row prefill
**status:** verified
**trust:** {F:0.9, G:narrow, R:0.85}
**context:** ml (Qwen35 prefill)
**evidence:**
- claim: "For `prefill_tokens_top1`, the final decoder layer only needs K/V cache updates for all prompt rows and the last row hidden state for logits; non-final output rows from the final layer are not future state."
  source: Qwen35 autoregressive state model and implementation in `src/ml/gguf/qwen35_cpu.cr`
  verified_at: 2026-04-24
  decay_trigger: prompt cache starts storing final hidden rows or downstream consumers require all final layer outputs
- claim: "`Qwen35Metal.full_attn_layer_chunk_project_last` projects K/V for the whole final-layer chunk, writes final-layer K/V cache, then computes Q/attention/FFN output only for the final prompt row."
  source: `src/ml/gguf/qwen35_metal.cr`
  verified_at: 2026-04-24
  decay_trigger: full-attn cache layout or residual/FFN ordering changes
- claim: "Correctness gate passed: targeted qwen35 specs -> 28 examples, 0 failures."
  source: `crystal spec spec/qwen35_metal_spec.cr spec/qwen35_forward_spec.cr spec/qwen35_delta_net_spec.cr spec/qwen35_state_snapshot_spec.cr spec/qwen35_prompt_cache_spec.cr --link-flags=\"$(pwd)/build/bridge.o -framework Metal -framework Foundation -lc++\"`
  verified_at: 2026-04-24
  decay_trigger: Qwen35 forward path changes
- claim: "A/B with `bin/qwen35_prefill_attribution.cr -- --prompt=64 --warmup=2 --reps=8 --compare-env=QWEN35_FINAL_FULL_LAST_OFF` measured default avg 170.01 ms / p50 170.00 ms versus off avg 174.88 ms / p50 173.86 ms."
  source: local command output
  verified_at: 2026-04-24
  decay_trigger: benchmark harness or power state changes
**adversary:** "The branch must choose fallback before mutating state through layers 0..30; a discarded earlier A/B violated that and was invalid."

### [LM-prefill-SWIGLU-F16-DOWN-FALSIFIER] SwiGLU-to-F16 FFN-down staging
**status:** refuted
**trust:** {F:0.8, G:narrow, R:0.85}
**context:** ml (Qwen35 prefill)
**evidence:**
- claim: "Computing SwiGLU directly into an F16 activation buffer before Q6 FFN-down is exact relative to the existing Q5/Q6 batch GEMM path, which already consumes F16 inputs internally."
  source: temporary `qwen35_swiglu_mul_f16` route in `src/ml/gguf/kernels/ffn_qwen35.metal` and `src/ml/gguf/qwen35_metal.cr`
  verified_at: 2026-04-24
  decay_trigger: Q5/Q6 GEMM input precision or FFN-down route changes
- claim: "Focused correctness smoke passed while the route was enabled: `spec/qwen35_forward_spec.cr` -> 10 examples, 0 failures."
  source: `CRYSTAL_CACHE_DIR=/tmp/cogni_ml_crystal_cache_swiglu_f16_spec crystal spec spec/qwen35_forward_spec.cr --link-flags=\"$(pwd)/build/bridge.o -framework Metal -framework Foundation -lc++\"`
  verified_at: 2026-04-24
  decay_trigger: prefill FFN path changes
- claim: "A/B with `bin/qwen35_prefill_attribution.cr -- --prompt=64 --warmup=2 --reps=6 --compare-env=QWEN35_SWIGLU_F16_DOWN_OFF` measured default avg 170.02 ms / p50 170.68 ms versus off avg 167.52 ms / p50 169.64 ms."
  source: local command output
  verified_at: 2026-04-24
  decay_trigger: benchmark harness or power state changes
**decision:** "Do not keep or retry direct SwiGLU-to-F16 staging as a default optimization; activation staging removal is smaller than its scheduling/conversion overhead on pp64."

### [LM-prefill-Q6-DB-GEMM-FALSIFIER] Q6_K double-buffered GEMM
**status:** refuted
**trust:** {F:0.8, G:narrow, R:0.85}
**context:** ml (Qwen35 prefill)
**evidence:**
- claim: "A Q6_K batch GEMM variant can mirror the Q5_K double-buffered K-tile schedule and preserve forward correctness."
  source: temporary `simd_mm_q6k_db` route in `src/ml/gguf/kernels/gemm_mm.metal` and `src/ml/gguf/qwen35_metal.cr`
  verified_at: 2026-04-24
  decay_trigger: Q6_K GEMM kernel or prefill route changes
- claim: "Focused correctness smoke passed while the route was enabled: `spec/qwen35_forward_spec.cr` -> 9 examples, 0 failures."
  source: `CRYSTAL_CACHE_DIR=/tmp/cogni_ml_crystal_cache_q6db_spec crystal spec spec/qwen35_forward_spec.cr --link-flags=\"$(pwd)/build/bridge.o -framework Metal -framework Foundation -lc++\"`
  verified_at: 2026-04-24
  decay_trigger: Q6_K GEMM kernel or forward path changes
- claim: "Repeated pp64 A/B with `bin/qwen35_prefill_attribution.cr -- --prompt=64 --warmup=3 --reps=16 --compare-env=QWEN35_Q6K_DB_GEMM_OFF` measured default avg 168.22 ms / p50 168.81 ms versus off avg 168.25 ms / p50 168.62 ms."
  source: local command output
  verified_at: 2026-04-24
  decay_trigger: benchmark harness or power state changes
**decision:** "Do not keep the double-buffered Q6_K variant; the observed effect is noise-level and p50 is slightly worse."

### [LM-prefill-ADAPTIVE-CHUNK-SIZE] Memory-aware default prefill chunks
**status:** verified
**trust:** {F:0.85, G:medium, R:0.85}
**context:** ml (Qwen35 prefill)
**evidence:**
- claim: "Default `QWEN35_PREFILL_CHUNK_SIZE=64` created avoidable chunk boundaries for prompts above 64 tokens; chunk sizes matching the prompt reduced Metal syncs and improved throughput in sweep runs."
  source: local chunk sweep with `bin/qwen35_prefill_attribution.cr` for prompt sizes 64, 128, 256, 512, and 1024
  verified_at: 2026-04-24
  decay_trigger: prefill scheduler, memory budget, or full-attention chunk implementation changes
- claim: "Changing the default chunk policy preserved the env override and passed focused forward/prompt-cache specs, including a pure memory-threshold regression check."
  source: `CRYSTAL_CACHE_DIR=/tmp/cogni_ml_crystal_cache_adaptive_chunk_spec crystal spec spec/qwen35_forward_spec.cr spec/qwen35_prompt_cache_spec.cr --link-flags=\"$(pwd)/build/bridge.o -framework Metal -framework Foundation -lc++\"` -> 16 examples, 0 failures
  verified_at: 2026-04-24
  decay_trigger: Qwen35 prefill or prompt cache changes
- claim: "Full targeted Qwen gate passed after adaptive chunk selection."
  source: `CRYSTAL_CACHE_DIR=/tmp/cogni_ml_crystal_cache_adaptive_chunk_gate crystal spec spec/qwen35_metal_spec.cr spec/qwen35_forward_spec.cr spec/qwen35_delta_net_spec.cr spec/qwen35_state_snapshot_spec.cr spec/qwen35_prompt_cache_spec.cr --link-flags=\"$(pwd)/build/bridge.o -framework Metal -framework Foundation -lc++\"` -> 30 examples, 0 failures
  verified_at: 2026-04-24
  decay_trigger: Qwen35 prefill, Metal kernels, or prompt cache changes
- claim: "pp256 A/B with `--compare-env=QWEN35_PREFILL_CHUNK_SIZE --compare-off=64` measured default avg 524.34 ms / p50 522.76 ms versus chunk64 avg 708.19 ms / p50 711.92 ms."
  source: local command output
  verified_at: 2026-04-24
  decay_trigger: benchmark harness or power state changes
- claim: "pp64 A/B stayed within noise: default avg 169.25 ms / p50 170.57 ms versus chunk64 avg 170.03 ms / p50 170.89 ms."
  source: local command output
  verified_at: 2026-04-24
  decay_trigger: benchmark harness or power state changes
- claim: "Matched pp256/gen16 comparison after the default change measured cogni-ml prefill p50 493.41 tok/s versus llama.cpp 575.71 tok/s; cogni-ml decode stayed ahead at 48.07 tok/s versus llama.cpp 45.32 tok/s."
  source: `CRYSTAL_CACHE_DIR=/tmp/cogni_ml_crystal_cache_vs_llama_chunk1024 crystal run --link-flags=\"$(pwd)/build/bridge.o -framework Metal -framework Foundation -lc++\" bin/benchmark_qwen_vs_llama.cr -- --prompt=256 --gen=16 --reps=3 --warmup=1`
  verified_at: 2026-04-24
  decay_trigger: benchmark harness, llama.cpp build, or power state changes
- claim: "pp2048 sweep measured chunk2048 p50 4578.57 ms versus chunk1024 p50 4755.48 ms; pp1024 stayed within noise, chunk2048 p50 2041.48 ms versus chunk1024 p50 2038.57 ms."
  source: local `bin/qwen35_prefill_attribution.cr` chunk sweep
  verified_at: 2026-04-24
  decay_trigger: benchmark harness or power state changes
- claim: "pp4096 one-shot sweep measured chunk4096 p50 11524.63 ms versus chunk2048 p50 11862.78 ms."
  source: local `bin/qwen35_prefill_attribution.cr` chunk sweep
  verified_at: 2026-04-24
  decay_trigger: benchmark harness, memory pressure, or power state changes
- claim: "Adaptive default chooses chunk 8192 on the local 64GB M2 Max and pp8192 one-shot A/B measured default p50 33981.42 ms versus explicit chunk4096 p50 35219.27 ms."
  source: `bin/qwen35_prefill_attribution.cr -- --prompt=8192 --warmup=0 --reps=1 --compare-env=QWEN35_PREFILL_CHUNK_SIZE --compare-off=4096`
  verified_at: 2026-04-24
  decay_trigger: benchmark harness, memory pressure, or power state changes
**adversary:** "Larger chunks increase peak scratch memory; memory-aware default uses 8192 only at >=48 GiB, 4096 at >=24 GiB, and 2048 below that; `QWEN35_PREFILL_CHUNK_SIZE` remains an override."

### [LM-prefill-GROUP-ATTRIBUTION-1] Fused prefill group timing
**status:** verified
**trust:** {F:0.82, G:medium, R:0.80}
**context:** ml (Qwen35 prefill)
**evidence:**
- claim: "`Qwen35Metal::Profile` now reports per grouped command-buffer labels for the fused recurrent run and full-attention-plus-recurrent waves, including encode/wait/read timing."
  source: `src/ml/gguf/qwen35_metal.cr`, `src/ml/gguf/qwen35_cpu.cr`
  verified_at: 2026-04-24
  decay_trigger: Profile instrumentation or prefill scheduler changes
- claim: "Focused forward/prompt-cache specs passed after adding group labels."
  source: `CRYSTAL_CACHE_DIR=/tmp/cogni_ml_crystal_cache_group_profile_spec crystal spec spec/qwen35_forward_spec.cr spec/qwen35_prompt_cache_spec.cr --link-flags=\"$(pwd)/build/bridge.o -framework Metal -framework Foundation -lc++\"` -> 16 examples, 0 failures
  verified_at: 2026-04-24
  decay_trigger: Qwen35 prefill/profile code changes
- claim: "Full targeted Qwen gate passed after group-attribution instrumentation."
  source: `CRYSTAL_CACHE_DIR=/tmp/cogni_ml_crystal_cache_group_profile_gate crystal spec spec/qwen35_metal_spec.cr spec/qwen35_forward_spec.cr spec/qwen35_delta_net_spec.cr spec/qwen35_state_snapshot_spec.cr spec/qwen35_prompt_cache_spec.cr --link-flags=\"$(pwd)/build/bridge.o -framework Metal -framework Foundation -lc++\"` -> 30 examples, 0 failures
  verified_at: 2026-04-24
  decay_trigger: Qwen35 Metal/profile code changes
- claim: "Two pp256 profile runs show no layer-group outlier: all seven `full+rec` groups wait around `61.2-61.7 ms`, while the initial recurrent-only `rec0-2` group waits around `46.4 ms`."
  source: `bin/qwen35_prefill_attribution.cr -- --prompt=256 --warmup=1 --reps=3` and repeat with `--reps=2`
  verified_at: 2026-04-24
  decay_trigger: benchmark harness, kernel route, or power state changes
- claim: "A cold pp2048 profile initially showed `rec0-2` as an apparent outlier (`901.24 ms` wait), but a warm run refuted that interpretation: `rec0-2` dropped to `344.42 ms`, while `full+rec` groups waited around `562-568 ms`."
  source: `CRYSTAL_CACHE_DIR=/tmp/cogni_ml_crystal_cache_current_prefill_pp2048 crystal run --link-flags=\"$(pwd)/build/bridge.o -framework Metal -framework Foundation -lc++\" bin/qwen35_prefill_attribution.cr -- --prompt=2048 --warmup=0 --reps=1`
  verified_at: 2026-04-24
  decay_trigger: benchmark harness, chunk size, kernel route, or power state changes
  adversary_update: `CRYSTAL_CACHE_DIR=/tmp/cogni_ml_crystal_cache_current_prefill_pp2048_warm crystal run --link-flags=\"$(pwd)/build/bridge.o -framework Metal -framework Foundation -lc++\" bin/qwen35_prefill_attribution.cr -- --prompt=2048 --warmup=1 --reps=1` -> warm profile wall `4560.86 ms`, wall rep `4547.89 ms`, `rec0-2` wait `344.42 ms`.
**adversary:** "This refutes a single bad layer-group hypothesis for pp256 and refutes the cold pp2048 recurrent-outlier hypothesis. The next exact optimization should first add intra-group phase attribution before changing kernels."

### [LM-prefill-SCOPED-MATMUL-ATTRIBUTION-1] Scoped prefill/decode matmul attribution
**status:** verified
**trust:** {F:0.86, G:medium, R:0.82}
**context:** ml (Qwen35 prefill/decode attribution)
**evidence:**
- claim: "`Qwen35Metal::Profile` now maintains a trace scope stack and prefixes matmul shape counters with the active decode or prefill phase, without double-counting `encode_matmul` GEMV fallback."
  source: `src/ml/gguf/qwen35_metal.cr`
  verified_at: 2026-04-24
  decay_trigger: Profile instrumentation, `encode_matmul`, or wave/prefill scheduling changes
- claim: "Focused forward/prompt-cache specs pass after scoped attribution instrumentation."
  source: `CRYSTAL_CACHE_DIR=/tmp/cogni_ml_crystal_cache_scoped_profile_spec2 crystal spec spec/qwen35_forward_spec.cr spec/qwen35_prompt_cache_spec.cr --link-flags=\"$(pwd)/build/bridge.o -framework Metal -framework Foundation -lc++\"` -> 16 examples, 0 failures
  verified_at: 2026-04-24
  decay_trigger: Qwen35 Metal/profile code changes
- claim: "A pp256 prefill profile attributes logical weight traffic by phase: `prefill.rec.ffn_upgate q4_gemm Q4_K 4096x12288 b256` is largest at 48 calls / 1296.00 MiB; recurrent Q5 projection is 24 calls / 528.00 MiB; recurrent FFN-down totals 796.50 MiB across Q6/Q4 routes."
  source: `CRYSTAL_CACHE_DIR=/tmp/cogni_ml_crystal_cache_scoped_prefill_pp256b crystal run --link-flags=\"$(pwd)/build/bridge.o -framework Metal -framework Foundation -lc++\" bin/qwen35_prefill_attribution.cr -- --prompt=256 --warmup=1 --reps=1`
  verified_at: 2026-04-24
  decay_trigger: prefill chunk size, matmul route, profile code, kernel route, or power state changes
- claim: "A prompt64/gen4 top1 decode profile attributes logical weight traffic by phase: recurrent FFN up/gate is largest at 192 calls / 5184.00 MiB, recurrent Q5 projection is 96 calls / 2112.00 MiB, and recurrent FFN-down totals 3186.00 MiB across Q6/Q4 routes."
  source: `CRYSTAL_CACHE_DIR=/tmp/cogni_ml_crystal_cache_scoped_decode_b QWEN35_PROFILE_TOP1=1 crystal run --link-flags=\"$(pwd)/build/bridge.o -framework Metal -framework Foundation -lc++\" bin/qwen35_sync_profile.cr -- 64 4`
  verified_at: 2026-04-24
  decay_trigger: decode wave scheduling, matmul route, profile code, kernel route, or power state changes
**adversary:** "This is logical weight-traffic attribution, not per-kernel GPU-time attribution. It is sufficient to rank exact next targets: recurrent FFN up/gate and recurrent projection traffic dominate both prefill and decode. It does not justify attention rewrites yet because `full.attn` encode/wait remains small in these profiles."

### [LM-prefill-B8-GEMM-THRESHOLD-FALSIFIER] Batch-8 prefill should stay on GEMV
**status:** refuted
**trust:** {F:0.78, G:narrow, R:0.74}
**context:** ml (Qwen35 prefill matmul routing)
**evidence:**
- claim: "A bounded runtime override experiment changed the effective GEMM route threshold from `batch > 8` to `batch > 7`, causing b8 prefill projection routes to switch from `gemv` to `q4_gemm`/`q5_gemm`/`q6_gemm`."
  source: temporary local patch to `src/ml/gguf/qwen35_metal.cr`, verified by `bin/qwen35_prefill_attribution.cr -- --prompt=8 --warmup=3 --reps=8`
  verified_at: 2026-04-24
  decay_trigger: GEMM kernels, GEMV kernels, or prefill routing changes
- claim: "The b8 GEMM route was slower in the pp8 attribution run: default `GEMM_BATCH_THRESHOLD=8` measured p50 `140.70 ms`; threshold `7` measured p50 `158.74 ms`."
  source: `CRYSTAL_CACHE_DIR=/tmp/cogni_ml_crystal_cache_threshold_ab crystal run --link-flags=\"$(pwd)/build/bridge.o -framework Metal -framework Foundation -lc++\" bin/qwen35_prefill_attribution.cr -- --prompt=8 --warmup=3 --reps=8`, repeated with `QWEN35_GEMM_BATCH_THRESHOLD=7`
  verified_at: 2026-04-24
  decay_trigger: GEMM kernels, GEMV kernels, prefill routing, benchmark harness, or power state changes
**adversary:** "This was a narrow, noisy local falsifier, but the effect direction was large enough to reject lowering the default threshold now. The temporary runtime override was removed to avoid adding env lookup overhead in the encode path."

### [LM-prefill-FINAL-TOP1-FUSE-FALSIFIER] Final full-attn-last plus top1 fusion saves a sync but slows pp64
**status:** refuted
**trust:** {F:0.82, G:narrow, R:0.78}
**context:** ml (Qwen35 prefill final-layer/head boundary)
**evidence:**
- claim: "A bounded exact branch encoded final full-attn-last output RMSNorm and fused lm-head top1 into the same command buffer as `full_attn_layer_chunk_project_last`, reducing pp64 total Metal syncs from 10 to 9."
  source: temporary local patch to `src/ml/gguf/qwen35_metal.cr` and `src/ml/gguf/qwen35_cpu.cr`; `bin/qwen35_prefill_attribution.cr -- --prompt=64 --warmup=4 --reps=12`
  verified_at: 2026-04-24
  decay_trigger: final full-attn-last route, head top1 kernel, or command-buffer scheduler changes
- claim: "Focused forward/prompt-cache specs passed with the branch before benchmarking."
  source: `CRYSTAL_CACHE_DIR=/tmp/cogni_ml_crystal_cache_full_last_top1_spec2 crystal spec spec/qwen35_forward_spec.cr spec/qwen35_prompt_cache_spec.cr --link-flags=\"$(pwd)/build/bridge.o -framework Metal -framework Foundation -lc++\"` -> 16 examples, 0 failures
  verified_at: 2026-04-24
  decay_trigger: Qwen35 final prefill route changes
- claim: "Sequential release pp64 checks refuted promotion: in reverse-order A/B, old path measured p50 `162.32 ms`, while fused final-top1 measured p50 `165.67 ms` despite using one fewer sync."
  source: `QWEN35_FINAL_FULL_LAST_TOP1_OFF=1 ... bin/qwen35_prefill_attribution.cr -- --prompt=64 --warmup=4 --reps=12`, then default branch with the same command/cache
  verified_at: 2026-04-24
  decay_trigger: Metal command-buffer overhead, head top1 kernel, final route, or benchmark harness changes
**adversary:** "This is another boundary-fusion trap: removing a synchronization point is not automatically a win when it moves lm-head work into a larger command buffer and changes scheduling. The branch was removed; only this falsifier remains."

### [LM-prefill-Q4-B64-GEMM-FALSIFIER] Wider Q4_K batch-64 GEMM is not a default prefill win
**status:** refuted
**trust:** {F:0.80, G:narrow, R:0.76}
**context:** ml (Qwen35 prefill Q4_K GEMM)
**evidence:**
- claim: "A bounded exact branch added a 256-thread `simd_mm_q4k_f32_b64` kernel that covers 64 batch rows per threadgroup, so pp64 Q4_K GEMM reads each output-row weight tile once instead of once per 32-token batch tile."
  source: temporary local patch to `src/ml/gguf/kernels/gemm_q4k.metal`, `src/ml/gguf/qwen35_metal.cr`, and `spec/qwen35_forward_spec.cr`
  verified_at: 2026-04-24
  decay_trigger: Q4_K GEMM tiling, Metal compiler scheduling, or prefill chunk routing changes
- claim: "Correctness passed with the b64 route enabled: targeted forward/DeltaNet specs returned `15 examples, 0 failures`; a fuller opt-in gate returned `34 examples, 0 failures`."
  source: `CRYSTAL_CACHE_DIR=/tmp/cogni_ml_crystal_cache_b64_default_spec crystal spec spec/qwen35_forward_spec.cr spec/qwen35_delta_net_spec.cr ...` and `CRYSTAL_CACHE_DIR=/tmp/cogni_ml_crystal_cache_b64_gate QWEN35_Q4K_B64_GEMM=1 crystal spec ...`
  verified_at: 2026-04-24
  decay_trigger: Qwen35 prefill specs or route toggles change
- claim: "Same-process pp64 attribution was only weakly positive for b64: default b64 p50 `164.51 ms` versus b64-off p50 `165.41 ms` over 10 reps."
  source: `CRYSTAL_CACHE_DIR=/tmp/cogni_ml_crystal_cache_b64_default_ab crystal run --release ... bin/qwen35_prefill_attribution.cr -- --prompt=64 --warmup=3 --reps=10 --compare-env=QWEN35_Q4K_B64_GEMM_OFF --compare-off=1`
  verified_at: 2026-04-24
  decay_trigger: benchmark harness, host load, or Q4_K GEMM route changes
- claim: "Matched llama benchmark adversary did not support promotion: b64 default measured native pp64 p50 `378.72 tok/s`, while the current b32 route with `QWEN35_Q4K_B64_GEMM_OFF=1` measured `394.94 tok/s` immediately after; decode was unchanged."
  source: `CRYSTAL_CACHE_DIR=/tmp/cogni_ml_crystal_cache_vs_llama_q4b64 crystal run --release ... bin/benchmark_qwen_vs_llama.cr -- --prompt=64 --gen=64 --reps=3 --warmup=1` and same with `QWEN35_Q4K_B64_GEMM_OFF=1`
  verified_at: 2026-04-24
  decay_trigger: clean-load rerun, llama.cpp rebuild, or Q4_K b64 kernel rewrite
**adversary:** "This is a lower-weight-traffic idea that likely lost to lower occupancy/register pressure and scheduling effects. Do not reintroduce a b64 Q4_K route without a paired benchmark harness showing both pp64 and longer prompts improve; the temporary code was removed."

### [LM-prefill-PAIRED-AB-HARNESS-1] Prefill compare-env uses paired interleaving
**status:** verified
**trust:** {F:0.84, G:medium, R:0.82}
**context:** ml (Qwen35 prefill benchmarking)
**evidence:**
- claim: "`bin/qwen35_prefill_attribution.cr --compare-env` now alternates default and alternate env settings inside each measured pair, balances pair order by trial parity, and reports pair wins."
  source: local patch to `bin/qwen35_prefill_attribution.cr`
  verified_at: 2026-04-24
  decay_trigger: prefill attribution harness rewrite
- claim: "Smoke verification completed on pp8 with `QWEN35_PREFILL_FINAL_CHUNK_OFF`: paired output reported `wins=2/2` and preserved the normal profile report."
  source: `CRYSTAL_CACHE_DIR=/tmp/cogni_ml_crystal_cache_prefill_attr_paired crystal run --release ... bin/qwen35_prefill_attribution.cr -- --prompt=8 --warmup=1 --reps=2 --compare-env=QWEN35_PREFILL_FINAL_CHUNK_OFF --compare-off=1`
  verified_at: 2026-04-24
  decay_trigger: harness output format or env toggle semantics change
**adversary:** "This does not remove the need for matched llama benchmarks before public claims; it only makes local env-toggle prefill decisions less vulnerable to block-order drift."

### [LM-prefill-Q56-H16-ADD-FALSIFIER] Removing Q56 FFN-down F16->F32 conversion is not enough
**status:** refuted
**trust:** {F:0.80, G:narrow, R:0.78}
**context:** ml (Qwen35 prefill Q56_K FFN-down)
**evidence:**
- claim: "A bounded exact branch kept Q5/Q6 GEMM FFN-down output as half and fused `residual + half(ffn_down)` in a new add kernel, avoiding the explicit `f16_to_f32` output conversion buffer for Q56 FFN-down paths."
  source: temporary local patch to `src/ml/gguf/kernels/ffn_qwen35.metal` and `src/ml/gguf/qwen35_metal.cr`
  verified_at: 2026-04-24
  decay_trigger: Q56 GEMM output format, FFN-down routing, or add kernel changes
- claim: "Correctness passed with the branch enabled: targeted forward/DeltaNet specs returned `14 examples, 0 failures`."
  source: `CRYSTAL_CACHE_DIR=/tmp/cogni_ml_crystal_cache_q56h16_spec QWEN35_Q56_DOWN_H16_ADD=1 crystal spec spec/qwen35_forward_spec.cr spec/qwen35_delta_net_spec.cr ...`
  verified_at: 2026-04-24
  decay_trigger: Qwen35 correctness specs or route toggles change
- claim: "Paired pp64 attribution was neutral: default p50 `167.22 ms`, branch p50 `167.30 ms`, wins `4/8`."
  source: `CRYSTAL_CACHE_DIR=/tmp/cogni_ml_crystal_cache_q56h16_ab crystal run --release ... bin/qwen35_prefill_attribution.cr -- --prompt=64 --warmup=2 --reps=8 --compare-env=QWEN35_Q56_DOWN_H16_ADD --compare-off=1`
  verified_at: 2026-04-24
  decay_trigger: benchmark harness, host load, or Q56 FFN-down route changes
**adversary:** "The conversion kernel/output buffer is not a material wall at pp64; the branch was removed. Future Q56 work needs to change matmul throughput or eliminate work, not only fold the final conversion."

### [LM-prefill-SMALL-Q4-GEMV-1] Tiny Q4_K prefill projections should stay on GEMV
**status:** verified
**trust:** {F:0.86, G:narrow, R:0.84}
**context:** ml (Qwen35 prefill routing)
**evidence:**
- claim: "For small Q4_K prefill projections with `out_dim <= 64`, the 64-row Q4 GEMM tile is underfilled enough that the existing Q4 GEMV route is faster despite batch 64."
  source: local patch to `src/ml/gguf/qwen35_metal.cr`, routed behind `QWEN35_SMALL_Q4_GEMV_OFF=1`
  verified_at: 2026-04-24
  decay_trigger: Q4_K GEMM/GEMV kernel rewrite, alpha/beta projection shapes, or batch threshold changes
- claim: "Correctness passed after default-enabling the small-Q4 GEMV route: focused forward/DeltaNet specs returned `14 examples, 0 failures`."
  source: `CRYSTAL_CACHE_DIR=/tmp/cogni_ml_crystal_cache_smallq4gemv_default_spec crystal spec spec/qwen35_forward_spec.cr spec/qwen35_delta_net_spec.cr ...`
  verified_at: 2026-04-24
  decay_trigger: Qwen35 correctness specs or prefill routing changes
- claim: "Paired pp64 A/B favored the new default: default p50 `156.14 ms`, route-off p50 `164.04 ms`, wins `8/8`."
  source: `CRYSTAL_CACHE_DIR=/tmp/cogni_ml_crystal_cache_smallq4gemv_default_ab crystal run --release ... bin/qwen35_prefill_attribution.cr -- --prompt=64 --warmup=2 --reps=8 --compare-env=QWEN35_SMALL_Q4_GEMV_OFF --compare-off=1`
  verified_at: 2026-04-24
  decay_trigger: benchmark harness, host load, or route toggle semantics change
- claim: "Matched llama comparison after promotion measured native pp64 p50 `408.02 tok/s` versus llama.cpp `463.3 tok/s`; decode remained ahead at native p50 `47.14 tok/s` versus llama `45.27 tok/s`."
  source: `CRYSTAL_CACHE_DIR=/tmp/cogni_ml_crystal_cache_smallq4gemv_vs_llama crystal run --release ... bin/benchmark_qwen_vs_llama.cr -- --prompt=64 --gen=64 --reps=3 --warmup=1`
  verified_at: 2026-04-24
  decay_trigger: llama.cpp rebuild, benchmark settings, thermal/power state, or Qwen35 route changes
**adversary:** "This is a narrow routing win, not a general GEMV-over-GEMM rule. Keep GEMM for large Q4 prefill projections; only the tiny alpha/beta-style projections are defaulted to GEMV."

### [LM-prefill-SMALL-Q4-1024-REFUTE-1] Do not extend the tiny-Q4 GEMV rule to 1024-wide projections
**status:** refuted
**trust:** {F:0.78, G:narrow, R:0.78}
**context:** ml (Qwen35 prefill routing)
**evidence:**
- claim: "A temporary env-threshold branch allowed `QWEN35_SMALL_Q4_GEMV_MAX=1024`, routing Q4_K projections with `out_dim <= 1024` through GEMV instead of only `out_dim <= 64`."
  source: temporary local patch to `src/ml/gguf/qwen35_metal.cr`
  verified_at: 2026-04-24
  decay_trigger: Q4_K GEMM/GEMV kernel rewrite or projection mix changes
- claim: "Correctness still passed on the branch: focused forward/DeltaNet specs returned `14 examples, 0 failures`."
  source: `CRYSTAL_CACHE_DIR=/tmp/cogni_ml_crystal_cache_smallq4max_spec crystal spec spec/qwen35_forward_spec.cr spec/qwen35_delta_net_spec.cr ...`
  verified_at: 2026-04-24
  decay_trigger: Qwen35 correctness specs or prefill routing changes
- claim: "Paired pp64 A/B refuted promotion: default threshold 64 p50 `165.86 ms`, threshold 1024 p50 `169.08 ms`, wins `8/8` for threshold 64."
  source: `CRYSTAL_CACHE_DIR=/tmp/cogni_ml_crystal_cache_smallq4max_ab QWEN35_SMALL_Q4_GEMV_MAX=64 crystal run --release ... bin/qwen35_prefill_attribution.cr -- --prompt=64 --warmup=2 --reps=8 --compare-env=QWEN35_SMALL_Q4_GEMV_MAX --compare-off=1024`
  verified_at: 2026-04-24
  decay_trigger: benchmark harness, host load, or Q4 route changes
**adversary:** "The small-Q4 win is shape-specific. Once `out_dim` reaches 1024, batch GEMM's weight reuse beats GEMV's per-token rereads; keep the default cutoff at 64."

### [LM-prefill-SMALL-Q56-GEMV-REFUTE-1] Small Q5/Q6 prefill projections should stay on batch GEMM
**status:** refuted
**trust:** {F:0.78, G:narrow, R:0.78}
**context:** ml (Qwen35 prefill routing)
**evidence:**
- claim: "A temporary branch routed Q5_K/Q6_K prefill projections with `out_dim <= 1024` through GEMV instead of the batch GEMM route."
  source: temporary local patch to `src/ml/gguf/qwen35_metal.cr` behind `QWEN35_SMALL_Q56_GEMV=1`
  verified_at: 2026-04-24
  decay_trigger: Q5/Q6 GEMM conversion path, GEMV kernels, or projection mix changes
- claim: "Correctness still passed on the branch: focused forward/DeltaNet specs returned `14 examples, 0 failures`."
  source: `CRYSTAL_CACHE_DIR=/tmp/cogni_ml_crystal_cache_smallq56_spec crystal spec spec/qwen35_forward_spec.cr spec/qwen35_delta_net_spec.cr ...`
  verified_at: 2026-04-24
  decay_trigger: Qwen35 correctness specs or prefill routing changes
- claim: "Paired pp64 A/B refuted promotion: default p50 `165.07 ms`, small-Q56-GEMV p50 `167.05 ms`, wins `8/8` for default."
  source: `CRYSTAL_CACHE_DIR=/tmp/cogni_ml_crystal_cache_smallq56_ab crystal run --release ... bin/qwen35_prefill_attribution.cr -- --prompt=64 --warmup=2 --reps=8 --compare-env=QWEN35_SMALL_Q56_GEMV`
  verified_at: 2026-04-24
  decay_trigger: benchmark harness, host load, or Q56 route changes
**adversary:** "Even with F32/F16 conversion overhead, Q5/Q6 batch GEMM remains better for 1024-wide projections at batch 64. Do not copy the Q4 `out_dim <= 64` rule to Q56 without new evidence."

### [LM-prefill-GC-GUARD-1] Disable Crystal GC during multi-token prefill hot path
**status:** verified
**trust:** {F:0.82, G:medium, R:0.80}
**context:** ml (Qwen35 prefill runtime)
**evidence:**
- claim: "Multi-token prefill allocates several large boundary `Array(Float32)` objects while moving hidden states between fused Metal layer groups; disabling Crystal GC only during the hot prefill call reduces GC/jitter without changing numerical semantics."
  source: local patch to `src/ml/gguf/qwen35_cpu.cr`, guarded by `QWEN35_PREFILL_GC_GUARD_OFF=1`
  verified_at: 2026-04-24
  decay_trigger: GPU-resident layer scheduler, boundary Array allocation pattern, or Crystal GC behavior changes
- claim: "Correctness passed after default-enabling the guard: focused forward/DeltaNet specs returned `14 examples, 0 failures`."
  source: `CRYSTAL_CACHE_DIR=/tmp/cogni_ml_crystal_cache_gc_guard_spec crystal spec spec/qwen35_forward_spec.cr spec/qwen35_delta_net_spec.cr ...`
  verified_at: 2026-04-24
  decay_trigger: Qwen35 correctness specs or prefill control flow changes
- claim: "Paired pp64 A/B favored the guard: default p50 `151.80 ms`, guard-off p50 `158.00 ms`, wins `8/8`."
  source: `CRYSTAL_CACHE_DIR=/tmp/cogni_ml_crystal_cache_gc_guard_ab64 crystal run --release ... bin/qwen35_prefill_attribution.cr -- --prompt=64 --warmup=2 --reps=8 --compare-env=QWEN35_PREFILL_GC_GUARD_OFF --compare-off=1`
  verified_at: 2026-04-24
  decay_trigger: benchmark harness, host load, or prefill allocation changes
- claim: "At pp256, profile read spikes dropped from a prior `31.34 ms` read total to `3.23 ms` with the guard; paired wall was mixed but average favored the guard (`489.36 ms` vs `493.41 ms`) while p50 was effectively neutral."
  source: `CRYSTAL_CACHE_DIR=/tmp/cogni_ml_crystal_cache_gc_guard_ab256 crystal run --release ... bin/qwen35_prefill_attribution.cr -- --prompt=256 --warmup=1 --reps=6 --compare-env=QWEN35_PREFILL_GC_GUARD_OFF --compare-off=1`
  verified_at: 2026-04-24
  decay_trigger: host load, GC behavior, or GPU-resident hidden-state refactor
- claim: "Matched llama comparison after promotion measured native pp64 p50 `421.41 tok/s` versus llama.cpp `463.04 tok/s`; decode measured native p50 `48.02 tok/s` versus llama `44.43 tok/s`."
  source: `CRYSTAL_CACHE_DIR=/tmp/cogni_ml_crystal_cache_gc_guard_vs_llama crystal run --release ... bin/benchmark_qwen_vs_llama.cr -- --prompt=64 --gen=64 --reps=3 --warmup=1`
  verified_at: 2026-04-24
  decay_trigger: llama.cpp rebuild, benchmark settings, thermal/power state, or prefill runtime changes
**adversary:** "This is a runtime-jitter optimization, not a substitute for GPU-resident hidden-state scheduling. It is scoped to multi-token prefill, has an env escape hatch, and should be revisited if boundary Arrays are eliminated."

### [LM-prefill-Q5-F32-GEMM-FALSIFIER] Direct Q5_K F32 GEMM is not a pp64 wall win
**status:** refuted
**trust:** {F:0.78, G:narrow, R:0.76}
**context:** ml (Qwen35 prefill Q5_K GEMM)
**evidence:**
- claim: "A temporary branch added `simd_mm_q5k_f32`, a direct Q5_K F32-input/F32-output GEMM adapted from the Q4_K F32 kernel, and routed Q5_K batch prefill projections behind `QWEN35_Q5K_F32_GEMM=1`."
  source: temporary local patch to `src/ml/gguf/kernels/gemm_mm.metal` and `src/ml/gguf/qwen35_metal.cr`
  verified_at: 2026-04-24
  decay_trigger: Q5_K GEMM implementation, conversion path, or prefill projection mix changes
- claim: "Correctness passed with the branch enabled: focused forward/DeltaNet specs returned `14 examples, 0 failures`."
  source: `CRYSTAL_CACHE_DIR=/tmp/cogni_ml_crystal_cache_q5f32_spec QWEN35_Q5K_F32_GEMM=1 crystal spec spec/qwen35_forward_spec.cr spec/qwen35_delta_net_spec.cr ...`
  verified_at: 2026-04-24
  decay_trigger: Qwen35 correctness specs or Q5 route changes
- claim: "Paired pp64 A/B did not support promotion: default p50 `152.56 ms`, branch p50 `152.52 ms`, default average `152.42 ms`, branch average `153.67 ms`, wins `3/8` for default-as-first."
  source: `CRYSTAL_CACHE_DIR=/tmp/cogni_ml_crystal_cache_q5f32_ab crystal run --release ... bin/qwen35_prefill_attribution.cr -- --prompt=64 --warmup=2 --reps=8 --compare-env=QWEN35_Q5K_F32_GEMM`
  verified_at: 2026-04-24
  decay_trigger: benchmark harness, host load, or Q5 route changes
- claim: "Standalone op attribution showed at most a narrow Q5 micro-win (`4096x8192 b64` p50 `4.112 ms` branch vs `4.253 ms` default) that did not survive the fused prefill wall test."
  source: `bin/qwen35_op_attribution.cr -- --batch=64 --warmup=3 --runs=9 --limit=8` with and without `QWEN35_Q5K_F32_GEMM=1`
  verified_at: 2026-04-24
  decay_trigger: op attribution harness or Q5 kernel changes
**adversary:** "Removing Q5 F32/F16 conversion is not a reliable wall-clock lever in the fused prefill wave. The temporary code was removed; revisit only with a materially different Q5 tile design or a code-variant harness showing pp64/pp256 wall improvement."

### [LM-prefill-FAST-CMD-FALSIFIER] Fast command buffers do not move prefill wall time
**status:** refuted
**trust:** {F:0.78, G:narrow, R:0.78}
**context:** ml (Qwen35 prefill command-buffer overhead)
**evidence:**
- claim: "A temporary branch added `QWEN35_PREFILL_FAST_CMD=1` and routed prefill helper/group command-buffer creation through `CommandBuffer.new(fast: true)`."
  source: temporary local patch to `src/ml/gguf/qwen35_metal.cr`
  verified_at: 2026-04-24
  decay_trigger: Metal command-buffer bridge, prefill grouping, or command-buffer creation path changes
- claim: "Correctness passed with the branch enabled: focused forward/DeltaNet specs returned `14 examples, 0 failures`."
  source: `CRYSTAL_CACHE_DIR=/tmp/cogni_ml_crystal_cache_prefill_fastcmd_spec QWEN35_PREFILL_FAST_CMD=1 crystal spec spec/qwen35_forward_spec.cr spec/qwen35_delta_net_spec.cr ...`
  verified_at: 2026-04-24
  decay_trigger: Qwen35 correctness specs or prefill command-buffer path changes
- claim: "Paired pp64 A/B was neutral: default p50 `152.02 ms`, fast-command-buffer p50 `151.97 ms`, wins `2/8`."
  source: `CRYSTAL_CACHE_DIR=/tmp/cogni_ml_crystal_cache_prefill_fastcmd_ab64 crystal run --release ... bin/qwen35_prefill_attribution.cr -- --prompt=64 --warmup=2 --reps=8 --compare-env=QWEN35_PREFILL_FAST_CMD`
  verified_at: 2026-04-24
  decay_trigger: benchmark harness, host load, or command-buffer bridge changes
- claim: "Paired pp256 A/B slightly favored the current default: default p50 `490.49 ms`, fast-command-buffer p50 `490.79 ms`, wins `3/6` for default-as-first."
  source: `CRYSTAL_CACHE_DIR=/tmp/cogni_ml_crystal_cache_prefill_fastcmd_ab256 crystal run --release ... bin/qwen35_prefill_attribution.cr -- --prompt=256 --warmup=1 --reps=6 --compare-env=QWEN35_PREFILL_FAST_CMD`
  verified_at: 2026-04-24
  decay_trigger: benchmark harness, host load, or command-buffer bridge changes
**adversary:** "The remaining prefill gap is not command-buffer creation overhead at current grouping. The temporary code was removed; focus on weight-traffic/matmul throughput or eliminating CPU hidden round-trips structurally."

### [LM-prefill-Q56-HOST-SCRATCH-CLEANUP-1] Q56 host/scratch cleanup is exact but small
**status:** verified
**trust:** {F:0.78, G:narrow, R:0.78}
**context:** ml (Qwen35 prefill host/runtime cleanup)
**evidence:**
- claim: "Shared F32 buffer reads no longer allocate a zero-filled Array before copying from unified-memory `contents`; the destination array is built and immediately overwritten."
  source: `src/ml/gguf/qwen35_metal.cr`, commit `ce98bfd`
  verified_at: 2026-04-24
  decay_trigger: Metal buffer read path or Crystal Array.build semantics change
- claim: "Q5/Q6 batch GEMM zero bias buffers are cleared directly through shared `contents` once, avoiding a heap zero Array before `ConstCache.write_once` in the steady prefill encode path."
  source: `src/ml/gguf/qwen35_metal.cr`, commit `57ceaad`
  verified_at: 2026-04-24
  decay_trigger: Q56 GEMM bias handling changes
- claim: "Q5/Q6 F16 conversion scratch buffers are reused by byte size instead of by weight offset; dispatches are encoded in order and the intermediate buffers are consumed before reuse."
  source: `src/ml/gguf/qwen35_metal.cr`, commit `06dd28d`
  verified_at: 2026-04-24
  decay_trigger: Q56 conversion scheduling or command-buffer hazard assumptions change
- claim: "Focused correctness gate passed after each cleanup; the final scratch cleanup run measured pp64 smoke avg `151.73 ms` / p50 `151.80 ms`."
  source: `CRYSTAL_CACHE_DIR=/tmp/cogni_ml_crystal_cache_q56scratch_spec crystal spec spec/qwen35_forward_spec.cr spec/qwen35_delta_net_spec.cr ...` and `bin/qwen35_prefill_attribution.cr -- --prompt=64 --warmup=2 --reps=8`
  verified_at: 2026-04-24
  decay_trigger: benchmark harness, host load, or Q56 staging code changes
- claim: "Matched llama comparison after the cleanup commits measured native pp64 p50 `420.49 tok/s` versus llama.cpp `461.16 tok/s`; decode remained ahead at native p50 `47.72 tok/s` versus llama `45.23 tok/s`."
  source: `CRYSTAL_CACHE_DIR=/tmp/cogni_ml_crystal_cache_after_q56_cleanup_vs_llama crystal run --release ... bin/benchmark_qwen_vs_llama.cr -- --prompt=64 --gen=64 --reps=3 --warmup=1`
  verified_at: 2026-04-24
  decay_trigger: llama.cpp rebuild, benchmark settings, thermal/power state, or Q56 staging code changes
**adversary:** "This is exact cleanup and memory-footprint reduction, not a robust standalone speed breakthrough. The remaining gap is still dominated by Q4/Q5/Q6 matmul weight traffic."

### [LM-prefill-Q56-NOBIAS-GEMM-FALSIFIER] Q56 no-bias GEMM mode is not a pp64 win
**status:** refuted
**trust:** {F:0.78, G:narrow, R:0.76}
**context:** ml (Qwen35 prefill Q56_K GEMM)
**evidence:**
- claim: "A temporary exact branch added `apply_gelu=2` to `simd_mm_q5k` and `simd_mm_q6k`, skipping the zero-bias read/add for Qwen35 Q5/Q6 batch GEMM calls."
  source: temporary local patch to `src/ml/gguf/kernels/gemm_mm.metal` and `src/ml/gguf/qwen35_metal.cr`
  verified_at: 2026-04-24
  decay_trigger: Q56 GEMM output epilogue changes
- claim: "Focused correctness passed with the branch: `spec/qwen35_forward_spec.cr spec/qwen35_delta_net_spec.cr` -> 14 examples, 0 failures."
  source: `CRYSTAL_CACHE_DIR=/tmp/cogni_ml_crystal_cache_q56nobias_spec crystal spec ...`
  verified_at: 2026-04-24
  decay_trigger: Qwen35 correctness specs or Q56 route changes
- claim: "Paired pp64 A/B refuted promotion: no-bias default avg `152.30 ms` / p50 `151.84 ms`, old bias path avg `152.00 ms` / p50 `151.80 ms`, wins `3/8`."
  source: `CRYSTAL_CACHE_DIR=/tmp/cogni_ml_crystal_cache_q56nobias_ab crystal run --release ... bin/qwen35_prefill_attribution.cr -- --prompt=64 --warmup=2 --reps=8 --compare-env=QWEN35_Q56_NOBIAS_GEMM_OFF --compare-off=1`
  verified_at: 2026-04-24
  decay_trigger: benchmark harness, host load, or Q56 kernel rewrite
**decision:** "Do not add a Q56 no-bias epilogue mode now; it adds kernel branch surface without wall-time evidence. The temporary code was removed."

### [LM-prefill-Q4-H16-GEMM-1] Q4_K half-input GEMM reduces prefill tile conversion work
**status:** verified
**trust:** {F:0.86, G:medium, R:0.84}
**context:** ml (Qwen35 prefill Q4_K GEMM)
**evidence:**
- claim: "The existing Q4_K prefill GEMM already rounds F32 activations to half before simdgroup MMA, but it does that while loading each output-row tile. `simd_mm_q4k_h16` preconverts the F32 activation matrix to F16 once per matmul and reuses that half input across output-row tiles, preserving the same multiplication precision."
  source: `src/ml/gguf/kernels/gemm_q4k.metal`, `src/ml/gguf/qwen35_metal.cr`
  verified_at: 2026-04-24
  decay_trigger: Q4_K GEMM input precision or prefill route changes
- claim: "Correctness passed after default-enabling the half-input route with `QWEN35_Q4K_H16_GEMM_OFF=1` escape hatch: focused forward/DeltaNet specs returned `14 examples, 0 failures`."
  source: `CRYSTAL_CACHE_DIR=/tmp/cogni_ml_crystal_cache_q4h16_default_spec crystal spec spec/qwen35_forward_spec.cr spec/qwen35_delta_net_spec.cr ...`
  verified_at: 2026-04-24
  decay_trigger: Qwen35 correctness specs or Q4_K route changes
- claim: "Paired pp64 A/B favored the new default: default avg `152.57 ms` / p50 `152.54 ms`, old F32-input path avg `153.86 ms` / p50 `153.83 ms`, wins `8/8`."
  source: `CRYSTAL_CACHE_DIR=/tmp/cogni_ml_crystal_cache_q4h16_default_ab64 crystal run --release ... bin/qwen35_prefill_attribution.cr -- --prompt=64 --warmup=2 --reps=8 --compare-env=QWEN35_Q4K_H16_GEMM_OFF --compare-off=1`
  verified_at: 2026-04-24
  decay_trigger: benchmark harness, host load, or Q4_K kernel changes
- claim: "Paired pp256 A/B also favored the half-input route before promotion: old F32-input path avg `491.72 ms` / p50 `491.08 ms`, half-input path avg `486.66 ms` / p50 `486.50 ms`, wins `6/6` for half-input."
  source: `CRYSTAL_CACHE_DIR=/tmp/cogni_ml_crystal_cache_q4h16_ab256 crystal run --release ... bin/qwen35_prefill_attribution.cr -- --prompt=256 --warmup=1 --reps=6 --compare-env=QWEN35_Q4K_H16_GEMM --compare-off=1`
  verified_at: 2026-04-24
  decay_trigger: benchmark harness, host load, or Q4_K kernel changes
- claim: "Matched llama comparison after promotion measured native pp64 p50 `424.96 tok/s` versus llama.cpp `462.9 tok/s`; decode stayed ahead at native p50 `48.38 tok/s` versus llama `45.89 tok/s`."
  source: `CRYSTAL_CACHE_DIR=/tmp/cogni_ml_crystal_cache_q4h16_vs_llama crystal run --release ... bin/benchmark_qwen_vs_llama.cr -- --prompt=64 --gen=64 --reps=3 --warmup=1`
  verified_at: 2026-04-24
  decay_trigger: llama.cpp rebuild, benchmark settings, power state, or Q4_K route changes
**adversary:** "The win comes from reducing repeated activation conversion/load work, not from changing quantized weight traffic. The route is exact for the current Q4 GEMM precision model but should be rechecked if the Q4 kernel moves to true F32 input accumulation."

### [LM-prefill-Q5-QKV-H16-CONV-1] Q5_K recurrent qkv output can stay half through conv-shift
**status:** verified
**trust:** {F:0.82, G:narrow, R:0.80}
**context:** ml (Qwen35 prefill Q5_K recurrent projection)
**evidence:**
- claim: "Q5_K recurrent `attn_qkv` batch GEMM already emits F16 internally, then the old route expanded that output to F32 only for `qwen35_recurrent_conv_shift_chunk` to read it immediately. The half-output route skips that expansion and uses `qwen35_recurrent_conv_shift_chunk_h16`, which casts the same half value to float inside the conv calculation."
  source: `src/ml/gguf/kernels/recurrent_qwen35.metal`, `src/ml/gguf/qwen35_metal.cr`
  verified_at: 2026-04-24
  decay_trigger: Q5 GEMM output precision or recurrent conv-shift semantics change
- claim: "Correctness passed after default-enabling the route with `QWEN35_Q5_QKV_H16_CONV_OFF=1` escape hatch: focused forward/DeltaNet specs returned `14 examples, 0 failures`."
  source: `CRYSTAL_CACHE_DIR=/tmp/cogni_ml_crystal_cache_q5qkvh16_default_spec crystal spec spec/qwen35_forward_spec.cr spec/qwen35_delta_net_spec.cr ...`
  verified_at: 2026-04-24
  decay_trigger: Qwen35 correctness specs or Q5 recurrent path changes
- claim: "Paired pp64 A/B favored the new default: default avg `150.73 ms` / p50 `150.56 ms`, off avg `151.01 ms` / p50 `150.97 ms`, wins `6/8`."
  source: `CRYSTAL_CACHE_DIR=/tmp/cogni_ml_crystal_cache_q5qkvh16_default_ab64 crystal run --release ... bin/qwen35_prefill_attribution.cr -- --prompt=64 --warmup=2 --reps=8 --compare-env=QWEN35_Q5_QKV_H16_CONV_OFF --compare-off=1`
  verified_at: 2026-04-24
  decay_trigger: benchmark harness, host load, or Q5 recurrent route changes
- claim: "Paired pp256 A/B also favored the default: default avg `486.90 ms` / p50 `486.62 ms`, off avg `487.43 ms` / p50 `487.22 ms`, wins `5/6`."
  source: `CRYSTAL_CACHE_DIR=/tmp/cogni_ml_crystal_cache_q5qkvh16_default_ab256 crystal run --release ... bin/qwen35_prefill_attribution.cr -- --prompt=256 --warmup=1 --reps=6 --compare-env=QWEN35_Q5_QKV_H16_CONV_OFF --compare-off=1`
  verified_at: 2026-04-24
  decay_trigger: benchmark harness, host load, or Q5 recurrent route changes
- claim: "Matched llama smoke after promotion measured native pp64 p50 `424.96 tok/s`; the llama.cpp side was noisy in this run (`pp` stddev `20.97 tok/s`, `tg` stddev `5.93 tok/s`), so do not use it as a refreshed public baseline."
  source: `CRYSTAL_CACHE_DIR=/tmp/cogni_ml_crystal_cache_q5qkvh16_vs_llama crystal run --release ... bin/benchmark_qwen_vs_llama.cr -- --prompt=64 --gen=64 --reps=3 --warmup=1`
  verified_at: 2026-04-24
  decay_trigger: quiet-load matched llama rerun
**adversary:** "The effect is small but consistent in paired local prefill A/B. It only covers recurrent Q5 qkv batch prefill, not decode GEMV and not full-attention q/v projections."

### [LM-prefill-Q4-SINGLE-BUFFER-FALSIFIER] Single-buffer Q4_K GEMM is not a default prefill win
**status:** refuted
**trust:** {F:0.78, G:narrow, R:0.74}
**context:** ml (Qwen35 prefill Q4_K GEMM)
**evidence:**
- claim: "A bounded exact branch changed `simd_mm_q4k_f32` from the current double-buffered 2-tile threadgroup layout to a llama.cpp-style single-buffer loop and used Q4-only `6144` byte threadgroup memory."
  source: temporary local patch to `src/ml/gguf/kernels/gemm_q4k.metal` and `src/ml/gguf/qwen35_metal.cr`
  verified_at: 2026-04-24
  decay_trigger: Q4_K GEMM loop, threadgroup memory, or Metal compiler scheduling changes
- claim: "Correctness passed with the branch: focused forward/DeltaNet specs returned `14 examples, 0 failures`."
  source: `CRYSTAL_CACHE_DIR=/tmp/cogni_ml_crystal_cache_q4single6144_spec crystal spec spec/qwen35_forward_spec.cr spec/qwen35_delta_net_spec.cr ...`
  verified_at: 2026-04-24
  decay_trigger: Qwen35 correctness specs or Q4_K kernel routing changes
- claim: "Standalone op attribution showed a narrow Q4_K up/gate micro-win: `4096x12288 b64` p50 improved from default `3.171 ms` to single-buffer `2.975 ms`."
  source: `bin/qwen35_op_attribution.cr -- --batch=64 --warmup=3 --runs=7 --limit=6` on default and temporary branch
  verified_at: 2026-04-24
  decay_trigger: op attribution harness, host load, or Q4_K kernel changes
- claim: "End-to-end pp64 attribution did not support promotion: default p50 was `167.39 ms`, while the single-buffer/Q4-6144 branch measured `176.01 ms` p50 under the same warmup/reps shape."
  source: `bin/qwen35_prefill_attribution.cr -- --prompt=64 --warmup=2 --reps=6` on default and temporary branch
  verified_at: 2026-04-24
  decay_trigger: clean-load rerun, paired code-variant harness, or command-buffer scheduling changes
**adversary:** "This is a microbench-vs-wall mismatch: the dominant standalone Q4 shape got faster, but the fused prefill wave became slower. Do not promote the single-buffer route without a code-variant A/B harness that shows pp64 and long-prompt wall improve; the temporary code was removed."

### [LM-attention-RESEARCH-BACKLOG-1] Efficient attention options for Qwen35
**status:** proposed
**trust:** {F:0.65, G:medium, R:0.70}
**context:** ml (Qwen35 long-context research)
**evidence:**
- claim: "FlashAttention is an exact IO-aware implementation strategy for softmax attention, so it can be applied without retraining if a Qwen35 full-attention phase is proven to dominate long-context prefill."
  source: FlashAttention paper (`https://arxiv.org/abs/2205.14135`) and current Qwen35 full-attn/recurrent split in `LANDMARKS.md`
  verified_at: 2026-04-24
  decay_trigger: full-attention kernel or profiling evidence changes
- claim: "DeepSeek Sparse Attention / NSA-style DSA is not an exact drop-in for Qwen35 weights: the paper introduces a lightning indexer and token selection through continued training."
  source: DeepSeek-V3.2 paper (`https://huggingface.co/deepseek-ai/DeepSeek-V3.2/resolve/main/assets/paper.pdf`)
  verified_at: 2026-04-24
  decay_trigger: DSA implementation details or Qwen sparse adaptation experiments change
- claim: "Linear attention changes the attention math and should be treated as a new architecture or distillation/continued-training branch rather than a runtime-only patch."
  source: Linear Transformers paper (`https://proceedings.mlr.press/v119/katharopoulos20a.html`)
  verified_at: 2026-04-24
  decay_trigger: exact linear-attention equivalence proof or successful Qwen distillation changes this
- claim: "Training-free long-context hacks exist but are approximate: attention sinks / StreamingLLM, SnapKV/H2O-like retention, and KV quantization should stay behind eval gates."
  source: StreamingLLM (`https://arxiv.org/abs/2309.17453`), SnapKV (`https://arxiv.org/abs/2404.14469`), H2O (`https://proceedings.neurips.cc/paper_files/paper/2023/file/6ceefa7b15572587b78ecfcebb2827f8-Paper-Conference.pdf`)
  verified_at: 2026-04-24
  decay_trigger: local eval harness or quality measurements change
**adversary:** "These are not the immediate Qwen35/M2 Max bottleneck until profiling proves attention dominates. Current prefill attribution points to repeated FFN/GEMM traffic, while decode already has whole-token GPU residency."

### [LM-prefill-LONG-SUFFIX-TOP1] Batched final chunk for long prompts
**status:** verified
**trust:** {F:0.85, G:medium, R:0.85}
**context:** ml (Qwen35 prefill)
**evidence:**
- claim: "When `prefill_tokens_top1` receives more tokens than the chunk size, it can prefill the prefix and recursively run `prefill_tokens_top1` on the final chunk; this keeps the final token in a batched prefill chunk instead of falling back to single-token decode."
  source: `src/ml/gguf/qwen35_cpu.cr`
  verified_at: 2026-04-24
  decay_trigger: prefill chunking or final-token top1 semantics change
- claim: "Correctness smoke with forced chunk size 4 passed against the final-token fallback including next decode state: `spec/qwen35_forward_spec.cr` -> 10 examples, 0 failures."
  source: `CRYSTAL_CACHE_DIR=/tmp/cogni_ml_crystal_cache_long_suffix_spec crystal spec spec/qwen35_forward_spec.cr --link-flags=\"$(pwd)/build/bridge.o -framework Metal -framework Foundation -lc++\"`
  verified_at: 2026-04-24
  decay_trigger: Qwen35 forward path changes
- claim: "pp2048 A/B with `--compare-env=QWEN35_PREFILL_LONG_SUFFIX_OFF` measured default avg 4748.53 ms / p50 4750.20 ms versus off avg 4982.20 ms / p50 4987.29 ms."
  source: local command output
  verified_at: 2026-04-24
  decay_trigger: benchmark harness or power state changes
**adversary:** "This only affects prompts longer than the active chunk size; shorter prompts are unchanged."

### [LM-codex-prefill-q4k-gemm] Q4_K GEMM inside recurrent prefill chunks
**status:** verified
**trust:** {F:0.9, G:medium, R:0.9}
**context:** ml (Qwen prefill)
**evidence:**
- claim: "GPU-resident recurrent prefill chunks should route Q4_K projections through `simd_mm_q4k_f32` when `batch > GEMM_BATCH_THRESHOLD`; Q5/Q6 and small batches remain on existing GEMV routes."
  source: `src/ml/gguf/qwen35_metal.cr` encoder selection in `recurrent_layer_chunk_project`
  verified_at: 2026-04-23
  decay_trigger: recurrent prefill encoder or Q4_K GEMM kernel rewritten
- claim: "Correctness gates passed after routing Q4_K chunk projections to GEMM."
  source: `crystal spec spec/qwen35_metal_spec.cr ...` => 8 examples, 0 failures; `crystal spec spec/qwen35_forward_spec.cr spec/qwen35_delta_net_spec.cr spec/qwen35_state_snapshot_spec.cr spec/qwen35_prompt_cache_spec.cr ...` => 18 examples, 0 failures
  verified_at: 2026-04-23
  decay_trigger: model forward path or state snapshot format changes
- claim: "pp64 benchmark improved from prior GPU recurrent chunk p50 `56.70 tok/s` to `70.38 tok/s`; decode remains ahead of llama.cpp by `5.81%` on gen16 in the same harness."
  source: `bin/benchmark_qwen_vs_llama.cr -- --prompt=64 --gen=16 --reps=3 --warmup=1`
  verified_at: 2026-04-23
  decay_trigger: benchmark harness, model file, power state, or llama.cpp HEAD changes

### [LM-codex-prefill-fullattn-chunk] Full-attention prefill chunks on Metal
**status:** verified
**trust:** {F:0.9, G:medium, R:0.9}
**context:** ml (Qwen prefill)
**evidence:**
- claim: "Full-attention prompt rows can be processed as an exact causal chunk by projecting Q/K/V for all rows, writing chunk K/V to the position-major cache, then letting row `t` attend only to `0..start_pos+t`."
  source: `qwen35_split_qgate_rows`, `qwen35_rmsnorm_heads_rows`, `qwen35_rope_partial_rows`, `qwen35_kv_write_rows`, `qwen35_attn_decode_rows` in `src/ml/gguf/kernels/fullattn_qwen35.metal`
  verified_at: 2026-04-23
  decay_trigger: full-attention kernel layout or KV cache layout changes
- claim: "Default full-attn prefill chunk matches serial fallback closely on a 16-token full-logit A/B."
  source: temporary A/B harness: same top1 `30`, cosine `0.9999999977579328`, max_abs `0.001364708`
  verified_at: 2026-04-23
  decay_trigger: Qwen forward numerics, kernel layout, or fallback path changes
- claim: "Full-attn chunking reduces profile `attn` syncs from 504 to 8 and pp64 p50 improves to `143.18 tok/s`; decode remains ahead of llama.cpp by `6.65%` on gen16."
  source: `bin/benchmark_qwen_vs_llama.cr -- --prompt=64 --gen=16 --reps=3 --warmup=1`
  verified_at: 2026-04-23
  decay_trigger: benchmark harness, model file, power state, or llama.cpp HEAD changes

### [LM-claude-BASELINE-1] llama.cpp HEAD 9B tok/s reference
**status:** verified
**trust:** {F:0.95, G:narrow, R:0.9}
**context:** ml (benchmarking)
**evidence:**
- claim: "Qwen 3.5 9B Q4_K_M on M2 Max, llama.cpp 86db42e97 (build 8894), MTL+BLAS backend, 8 threads: pp64=458.4 ± 1.4 t/s | tg64=43.5 ± 0.2 t/s (FA=0); pp256=542 ± 22 | tg256=38.8 ± 2.5 (FA=0)"
  source: `llama-bench` invocation 2026-04-22
  verified_at: 2026-04-22
  decay_trigger: llama.cpp HEAD updated OR model requantized
- claim: "FA=1 yields ~identical numbers (451 pp / 43.4 tg) → flash attention not active for head_dim=256, or bandwidth-dominated"
  source: same benchmark
  verified_at: 2026-04-22
  decay_trigger: llama.cpp adds FA support for head_dim=256
- claim: "Speed targets for cogni-ml Qwen port: decode ≥48 t/s (+10%), prefill ≥504 t/s (+10%)"
  source: project goal stated by user
  verified_at: 2026-04-22
  decay_trigger: user revises target

### [LM-claude-WEIGHTS-1] Qwen35 weight loader structures layer types
**status:** verified
**trust:** {F:0.9, G:high, R:0.9}
**context:** ml (Qwen port, Phase 1b.1)
**evidence:**
- claim: "Qwen35Weights.from_gguf loads 9B: 32 layers dispatched as 8 full-attn + 24 recurrent. Tensor shapes match Qwen35Hparams (attn_q=[4096,8192] combined Q+gate; attn_qkv=[4096,8192] recurrent; ssm_conv1d=[4,8192] covers full q+k+v)"
  source: spec/qwen35_weights_spec.cr, loaded from Qwen3.5-9B-Q4_K_M.gguf
  verified_at: 2026-04-22
  decay_trigger: tensor name convention changes OR hparams struct changes
- claim: "ssm_conv1d channel count 8192 = 2*num_k_heads*state + num_v_heads*state (full q+k+v stream), NOT just 2*num_k_heads*state as one might infer from Mamba-style conv. Relevant for Phase 1b.4 DeltaNet CPU reference."
  source: dims inspection on blk.0.ssm_conv1d vs hparams derivation
  verified_at: 2026-04-22
  decay_trigger: N/A

### [LM-claude-Q4K-CPU-VERIFIED] Q4_K CPU path bit-identical to llama.cpp
**status:** verified
**trust:** {F:1.0, G:medium, R:1.0}
**context:** ml (Qwen port, Phase 1a)
**evidence:**
- claim: "Crystal Dequant.dequantize_q4_k byte-identical to libggml's dequantize_row_q4_K on blk.0.ssm_alpha.weight (131072 elts) and blk.0.ffn_up.weight (50M elts): cos=1.0, max_abs_diff=0"
  source: spec/q4k_dequant_spec.cr via spec/support/q4k_ref (links libggml-base.dylib from llama.cpp HEAD)
  verified_at: 2026-04-22
  decay_trigger: Dequant code modified OR libggml algorithm changes
- claim: "Crystal QuantMatmul.matmul_add_q4k matches bulk-dequant + dense matmul on 4096×4096 attn_gate: cos=1.0, max_abs_diff≈2.4e-7 (fp32 accumulation order variance, well below precision threshold)"
  source: spec/q4k_matmul_spec.cr
  verified_at: 2026-04-22
  decay_trigger: quant_matmul.cr modified
- claim: "Adversary check passed: edge cases (rows=0 safe), regression (45 examples 0 failures, was 42), no new heap overflow surface"
  source: make spec output 2026-04-22 23:14
  verified_at: 2026-04-22
  decay_trigger: matmul interface changes

### [LM-claude-PHASE1B-VERIFIED] Qwen 3.5 9B CPU end-to-end generates adequate output
**status:** verified
**trust:** {F:0.9, G:0.6, R:0.9}
**context:** ml (Qwen port, Phase 1b complete)
**evidence:**
- claim: "Prompt 'The capital of France is' → greedy token ' Paris' (id=11751), continuation '.\\nThe capital'"
  source: bin/qwen35_generate_bin, /tmp/qwen35_ours2.log 2026-04-22
  verified_at: 2026-04-22
  decay_trigger: qwen35_cpu.cr modified
- claim: "spec/qwen35_forward_spec passes: logits finite, spread >1.0, distinct top-1 across different token inputs. First-token latency 137s on M2 Max (CPU only — expected, Phase 2 adds Metal)"
  source: crystal spec spec/qwen35_forward_spec.cr 2026-04-22
  verified_at: 2026-04-22
  decay_trigger: qwen35_cpu.cr modified
- claim: "Root bug fixed: ssm_conv1d accessed as [t*qkv_dim+ch]. GGUF dims=[4, 8192] with dims[0] innermost means layout is [ch*K+t]. Pre-fix output was garbage '{'_人家'."
  source: bin/qwen35_tensor_dump.cr + llama.cpp ops.cpp:9297 `c[i0 + i1*nc]` + commit 4dfafdb
  verified_at: 2026-04-22
  decay_trigger: N/A (foundational)
**builds_on:** [LM-claude-Q4K-CPU-VERIFIED], [LM-claude-WEIGHTS-1], [LM-claude-DELTANET-1]
**unblocks:** [Phase 2: Metal Q4_K kernels]

### [LM-claude-GGUF-LAYOUT-LESSON] GGUF dimension convention (learned the hard way)
**status:** verified
**trust:** {F:1.0, G:high, R:1.0}
**context:** ml (weights loading, any arch)
**lesson:** "dims[0] is the INNERMOST (contiguous) axis in memory. For a weight with dims=[A, B], memory layout is flat[i*A + a] where i ∈ [0, B), a ∈ [0, A). Always check dims AND expected inner axis before writing indexing code."
**evidence:**
- claim: "ssm_conv1d dims=[4, 8192] — conv_kernel (K=4) is innermost, channels (qkv_dim=8192) outermost. Indexing: [ch*K + t]. Same convention for matmul weights: [K=in_dim, N=out_dim] → flat[col*in_dim + row_inner]."
  source: Phase 1b debugging + llama.cpp ggml tensor layout convention
  verified_at: 2026-04-22
  decay_trigger: GGUF spec changes (extremely unlikely)

### [LM-claude-PHASE2_6-VERIFIED] Metal Q4_K + Q5_K + Q6_K matmul, integrated
**status:** verified
**trust:** {F:0.9, G:0.6, R:0.9}
**context:** ml (Qwen port, Phase 2.5+2.6)
**evidence:**
- claim: "Metal Q5_K GEMV (4096→8192 attn_qkv Q5_K): cos=1.0, max|Δ|=1.10e-6 vs CPU reference; 63ms GPU vs 711ms CPU = 11× (small matmul)"
  source: spec/qwen35_metal_spec.cr 2026-04-23 07:08
  verified_at: 2026-04-23
  decay_trigger: kernels/gemm_q56k.metal or qwen35_metal.cr modified
- claim: "Metal Q6_K GEMV (12288→4096 ffn_down Q6_K): cos=1.0, max|Δ|=4.77e-7; 8ms GPU vs 1080ms CPU = 135×"
  source: spec/qwen35_metal_spec.cr
  verified_at: 2026-04-23
  decay_trigger: kernel modified
- claim: "Metal Q6_K GEMV on lm_head (4096→248320 Q6_K, 796 MB tensor): cos=1.0, max|Δ|=3.58e-7; 143ms GPU vs 21867ms CPU = 152×"
  source: spec/qwen35_metal_spec.cr
  verified_at: 2026-04-23
  decay_trigger: kernel modified
- claim: "End-to-end decoder forward on Qwen 3.5 9B: 665 ms/token (vs 137 s/token in Phase 1b CPU = 206×). 80% of weights now on Metal (Q4_K+Q5_K+Q6_K), only tiny Q4_K ssm_alpha and all non-matmul ops remain on CPU."
  source: bin/qwen35_profile.cr single-token timing 2026-04-23
  verified_at: 2026-04-23
  decay_trigger: qwen35_cpu.cr or metal kernels changed
- claim: "Output on prompt 'The capital of France is' unchanged: still generates ' Paris.\\n'"
  source: bin/qwen35_generate.cr 2026-04-23 07:09
  verified_at: 2026-04-23
  decay_trigger: kernel modified or qmatvec logic changed
**builds_on:** [LM-claude-PHASE2-VERIFIED]
**unblocks:** [Phase 4: persistent buffers, Phase 3: attention/Mamba Metal]
**note:** Q5_K/Q6_K GEMV only (batch=1). Prefill for these types still on CPU. Persistent weight buffers (avoid per-call upload) are the biggest remaining single win — ~800 MB lm_head re-uploaded every token.

### [LM-claude-PHASE2-VERIFIED] Metal Q4_K matmul (GEMV+GEMM) correct and fast
**status:** verified
**trust:** {F:0.9, G:0.6, R:0.9}
**context:** ml (Qwen port, Phase 2 — standalone Metal kernel, not integrated into BERT compute graph)
**evidence:**
- claim: "simd_mv_q4k_f32 (GEMV, batch=1, 4096→12288 Q4_K ffn_up): cos=1.0, max|Δ|=3.28e-7 vs CPU matmul_add reference"
  source: spec/qwen35_metal_spec.cr on blk.0.ffn_up.weight
  verified_at: 2026-04-23
  decay_trigger: kernels/gemm_q4k.metal or qwen35_metal.cr modified
- claim: "simd_mm_q4k_f32 (GEMM, batch=16, 4096→12288 Q4_K ffn_up): cos=1.0, max|Δ|=4.9e-4 vs CPU reference"
  source: spec/qwen35_metal_spec.cr
  verified_at: 2026-04-23
  decay_trigger: kernels/gemm_q4k.metal modified
- claim: "Inverse shape (12288→4096 Q4_K ffn_down) works: cos=1.0, max|Δ|=5.96e-7"
  source: spec/qwen35_metal_spec.cr on blk.4.ffn_down.weight
  verified_at: 2026-04-23
  decay_trigger: kernel modified
- claim: "Performance (M2 Max, includes upload+download overhead, warm): GEMV 8.8ms vs CPU 783ms = 89x; GEMM batch=16 7.1ms vs CPU 12540ms = 1766x. First-run JIT warmup adds ~120ms to pipeline compile."
  source: spec output 2026-04-23 06:57
  verified_at: 2026-04-23
  decay_trigger: optimizations applied OR M-series architecture changes
**builds_on:** [LM-claude-Q4K-CPU-VERIFIED]
**unblocks:** [Phase 3a: Metal attention, Phase 3b: Metal Mamba]
**note:** Auto-selects GEMV for batch≤8, GEMM for batch>8. F32 in/out (no GELU, no bias — unlike BERT path). Full-upload per call — persistent weight buffers are a Phase 4 optimization.

### [LM-claude-PHASE4a-VERIFIED] Zero-copy mmap weight buffers via newBufferWithBytesNoCopy
**status:** verified
**trust:** {F:0.9, G:0.7, R:0.9}
**context:** ml (Qwen port, Phase 4a — unified-memory zero-copy for weights)
**evidence:**
- claim: "lm_head Q6_K (4096→248320, 796 MB) dispatch: upload-per-call 80.2 ms → zero-copy 3.0 ms = 26.9× speedup"
  source: bin/qwen35_metal_micro run 2026-04-23
  verified_at: 2026-04-23
  decay_trigger: qwen35_metal.cr register_mmap OR bridge.mm gs_create_buffer_no_copy changed
- claim: "End-to-end Qwen 3.5 9B decode per-token: 665 ms (Phase 2.6) → ~130-140 ms (Phase 4a steady-state) ≈ 5× speedup. First token 2.1 s includes kernel JIT."
  source: bin/qwen35_generate_bin "The capital of France is" 5 on 2026-04-23
  verified_at: 2026-04-23
  decay_trigger: qwen35_cpu.cr forward path or any Metal kernel modified
- claim: "Output unchanged vs Phase 2.6: 'The capital of France is Paris.\\nThe capital' — model still coherent and deterministic under greedy decoding"
  source: bin/qwen35_generate_bin same run
  verified_at: 2026-04-23
  decay_trigger: any numeric kernel change
- claim: "All 68 existing specs pass (6 Metal Q4_K/Q5_K/Q6_K specs show cos=1.0); no regressions in QuantWeight struct→class refactor"
  source: make spec 2026-04-23 (1 min, 0 failures, 0 errors)
  verified_at: 2026-04-23
  decay_trigger: specs or compute.cr changed
- claim: "Tensor data region inside GGUF is NOT page-aligned (data_offset=10967456, mod 16384 = 6560; 0/427 tensors page-aligned), but mmap base IS page-aligned. Therefore wrap whole mmap as one MTLBuffer and dispatch per-tensor via setBuffer:offset:."
  source: bin/qwen35_align_check run 2026-04-23
  verified_at: 2026-04-23
  decay_trigger: GGUF file format or mmap allocator behavior changes
**architecture:**
- bridge.mm: new `gs_create_buffer_no_copy(void*, size, mode)` wrapping `[device newBufferWithBytesNoCopy:length:options:deallocator:nil]`. `nil` deallocator = Metal does NOT free bytes (caller/mmap owns them).
- core/buffer.cr: `MetalBuffer.wrap_no_copy(ptr, size, mode)` + `new_from_external`/`init_external` for non-owned handles. `release` always releases MTLBuffer ObjC handle, sets purgeable only for owned.
- gguf/reader.cr: `mmap_region : {Pointer(UInt8), UInt64}?` exposes base+size.
- gguf/compute.cr: `QuantWeight` changed `struct` → `class` (reference semantics for lazy cached MetalBuffer fallback when raw bytes lie outside mmap region).
- gguf/qwen35_metal.cr: module-level `@@mmap_buf` holds whole-file MTLBuffer. `register_mmap` sets it up once. `mmap_slot_for(raw)` returns `{buf, offset}` if Bytes lies within mmap; `weight_slot`/`matmul(qw, ...)` dispatch to `matmul_gemv_buf`/`matmul_q4k_gemm_buf` with per-call `w_offset`.
- gguf/qwen35_weights.cr: keeps `@gguf : GGUFFile` alive for model lifetime so mmap stays mapped; drops `.dup` on raw weight bytes; calls `Qwen35Metal.register_mmap` after constructing all weights.
- gguf/qwen35_cpu.cr: `metal_matvec_or_nil` simplified to single `Qwen35Metal.matmul(qw, x, 1)` call.
**builds_on:** [LM-claude-PHASE2_6-VERIFIED]
**unblocks:** [Phase 4b: KV-cache persistent buffers, Phase 3a: Metal attention — now worthwhile since upload isn't the bottleneck]
**note:** User correction "Apple Silicon has unified memory, only locks matter, not uploads" was the key insight. Pre-correction design (copy once + reuse) would still have paid the 80ms memcpy on first call; true zero-copy is free. Owes entire 27× to `newBufferWithBytesNoCopy` + per-dispatch `setBuffer:offset:`.

### [LM-claude-PHASE3b-VERIFIED] Metal DeltaNet / GatedDeltaRule recurrent step
**status:** verified
**trust:** {F:0.9, G:0.7, R:0.9}
**context:** ml (Qwen port, Phase 3b — recurrent state update on GPU)
**evidence:**
- claim: "Metal `delta_net_step` bit-matches CPU reference on Qwen 3.5 9B shapes (h_k=16, h_v=32, s=128, seeded random inputs): y cos≈1.0 max|Δ|=1.21e-8, state cos≈1.0 max|Δ|=2.98e-8"
  source: spec/qwen35_delta_net_spec.cr (2026-04-23)
  verified_at: 2026-04-23
  decay_trigger: kernels/delta_net.metal OR qwen35_cpu.cr delta_net_step! changed
- claim: "End-to-end decode 418 ms/token (Phase 4a, CPU DeltaNet) → 78.3 ms/token (Phase 3b, Metal DeltaNet) = 5.3× total speedup. Recurrent layers 15.1 → 2.29 ms/layer = 6.6×"
  source: bin/qwen35_phase_profile on 2026-04-23, n_runs=5, pos starting at 6
  verified_at: 2026-04-23
  decay_trigger: delta_net kernel tuning OR forward_recurrent_layer changes
- claim: "Per-phase allocation after Phase 3b: total=78 ms (full_attn=14.1 ms/18%, recurrent=54.9 ms/70%, head=2.6 ms/3%, embedding=6.7 ms/9%). Recurrent still dominant — Phase 3a (Metal attention head_dim=256) is next highest-leverage target."
  source: same profiler run
  verified_at: 2026-04-23
  decay_trigger: any forward-path change
- claim: "Greedy output with Metal DeltaNet identical to Phase 4a CPU path: 'The capital of France is Paris.\\nThe capital' — deterministic under the new routing"
  source: bin/qwen35_generate_bin "The capital of France is" 5 on 2026-04-23
  verified_at: 2026-04-23
  decay_trigger: numeric kernel change
- claim: "All 69 specs pass (new qwen35_delta_net_spec + updated qwen35_deltanet_spec both pass; no regressions)"
  source: make spec 2026-04-23 (57 s, 0 failures, 0 errors)
  verified_at: 2026-04-23
  decay_trigger: kernels/delta_net.metal or dispatch glue changed
**architecture:**
- kernels/delta_net.metal: one threadgroup per v-head (dispatch (h_v, 1, 1)), 32 threads each. Four sequential stages with `threadgroup_barrier(mem_flags::mem_threadgroup)` between: (1) state *= ghead decay, (2) sk[d2] = Σ state[d2,:]·K via float4 dot, (3) state[d2,:] += K·(β·(V-sk))[d2] outer-product, (4) out[d2] = Σ state[d2,:]·Q · scale. Uses `float4` aligned loads/stores on s_v-stride rows (128·4B = 512B, 16-B aligned).
- qwen35_metal.cr: `DELTA_NET_SOURCE` embedded; `@@dn_pipeline` lazy-cached; `delta_net_step(state_buf, q/k/v/g/β, h_k/h_v/s, scale)` allocates per-call q/k/v/g/β/out buffers (~s·(h_k+h_v+h_v+2)·4B ≈ few KB) and reads out buffer back to `Array(Float32)`. State buffer is *persistent*, passed in and written in place.
- qwen35_cpu.cr: `delta_net_step!` extracted as top-level module fn (callable from specs as CPU reference). `LayerState.ssm_state_buf : ML::MetalBuffer?` added alongside `ssm_state : Array(Float32)?`. `delta_net_step_routed` picks whichever backend is available on first call and populates the matching field; later calls for that LayerState reuse the same backend — the two fields never coexist.
- spec/qwen35_deltanet_spec.cr updated: tests read SSM state via backend-agnostic helper `ssm_state_as_array.call(ls, expected_size)` → `ls.ssm_state_buf.read(n) ?? ls.ssm_state.not_nil!`.
**builds_on:** [LM-claude-PHASE4a-VERIFIED]
**unblocks:** [Phase 3a: Metal attention — now the critical path at ~18%, Phase 4: fused kernels + ComputeGraph]
**note:** Per-call q/k/v/g/β/out buffer allocation (small, 1–4 KB per layer) is probably wasteful at 24 layers × decode step; a persistent set of scratch buffers in `Qwen35Metal` is a cheap Phase 4 win. The `h_v=32` threadgroup dispatch leaves most of the M2 Max (30 cores × 4 concurrent TGs ≈ 120 TGs) idle per layer — likely why the kernel is still 2.3 ms/layer despite being tiny (128³ fma ≈ 2 Mops). Fusing across layers or across the two halves of the step would help.

### [LM-claude-PHASE3a-VERIFIED] Metal gated attention decode (head_dim=256 + GQA)
**status:** verified
**trust:** {F:0.9, G:0.7, R:0.9}
**context:** ml (Qwen port, Phase 3a — full-attention decode on GPU)
**evidence:**
- claim: "Qwen35-specific Metal kernel `attn_decode_qwen35.metal` bit-matches CPU reference on 9B shapes (n_head=40, n_head_kv=8, head_dim=256, heads_per_group=5, pos=17): cos=1.0 max|Δ|=4.47e-8"
  source: spec/qwen35_attn_decode_spec.cr (2026-04-23)
  verified_at: 2026-04-23
  decay_trigger: kernels/attn_decode_qwen35.metal OR forward_full_attn_layer attn inner loop changed
- claim: "Metal attention scales better than CPU with position. Per full-attn-layer decode time (Metal vs CPU, 8 layers): pos=257 3.70 vs 6.19 ms (1.67×); pos=513 6.70 vs 8.68 ms (1.30×); pos=1025 5.76 vs 13.68 ms (2.38×)"
  source: bin/qwen35_long_ctx_profile 2026-04-23, n_runs=5 each, QWEN35_ATTN_CPU=1 env gate for A/B
  verified_at: 2026-04-23
  decay_trigger: kernel/routing changes
- claim: "Total decode savings vs CPU attn at long context — pos=1025: 118.1 ms/tok (Metal) vs 188.2 ms/tok (CPU) = 70 ms/tok saved (37% faster)"
  source: same profiler
  verified_at: 2026-04-23
  decay_trigger: forward-path change
- claim: "Short-context (pos<32) cost is neutral to small: Phase 3a integration did not regress end-to-end greedy output — first-token top logit 11.423702 numerically identical to pre-integration; full generation 'The capital of France is Paris.\\nThe capital' unchanged"
  source: bin/qwen35_generate_bin 2026-04-23
  verified_at: 2026-04-23
  decay_trigger: numeric kernel change
- claim: "All 70 specs pass (new qwen35_attn_decode_spec + updated qwen35_fullattn_spec both pass; no regressions)"
  source: make spec 2026-04-23
  verified_at: 2026-04-23
  decay_trigger: kernel or dispatch glue changed
**architecture:**
- kernels/attn_decode_qwen35.metal: one threadgroup per query head (n_head=40 TGs), 32 threads each. Q and gate for the head are cached in threadgroup memory (QA_HD=256 bound). Online (flash-style) softmax over position tiles of 32 with simd_max for the max update and simd_sum for the normalizer correction. GQA broadcast in-kernel via `kv_h = h / heads_per_group`. Per-lane output accumulators (8 slots for head_dim=256 / warp=32). Fused `sigmoid(gate)` multiply at write-out.
- qwen35_metal.cr: `ATTN_DECODE_SOURCE` embedded via `read_file`; `@@attn_pipeline` lazy-cached; `Qwen35Metal.attn_decode(q, gate, k_cache_buf, v_cache_buf, pos, ...)` allocates per-call q/gate/out MetalBuffers and dispatches `(n_head, 1, 1)` threadgroups.
- qwen35_cpu.cr: `LayerState.k_cache_buf`/`v_cache_buf : ML::MetalBuffer?` added alongside Array-based `k_cache`/`v_cache`. `attn_decode_routed` picks Metal first; on first call lazily allocates persistent max_seq·kv_dim KV buffers in unified memory, writes per-position K/V slice via `contents.as(Pointer(Float32)) + base` pointer arithmetic (zero-copy), then dispatches attn_decode. CPU path writes Array cache and runs the inner loop inline.
- ENV `QWEN35_ATTN_CPU=1` forces CPU attention for A/B profiling without disabling the rest of the Metal path.
- spec/qwen35_fullattn_spec.cr updated: reads KV via backend-agnostic `read_kv` helper that picks `k_cache_buf.read(n)` or `k_cache.not_nil!` depending on which LayerState field is populated.
**builds_on:** [LM-claude-PHASE3b-VERIFIED]
**unblocks:** [Phase 3.5: state save/load, Phase 4: fused kernels]
**note:** Attention is now ~18% at short context (pos<64) and dominates the remaining CPU/GPU disparity at long context. Per-call q/gate/out allocation can be pooled in Phase 4 (same pattern as DeltaNet). The `kv_dim.times { k_ptr[i] = k[i] }` write is a small trivial CPU loop inside the decode step — could be fused into a Metal compute shader once K/V projections themselves land on GPU. Phase 4 fusion (qkv gemm → rope → kv-write → attn → gate → o_proj) is the next big lever vs llama.cpp.

### [LM-claude-PHASE4.0-VERIFIED] Instrumentation — sync overhead is the dominant cost
**status:** verified
**trust:** {F:0.9, G:0.7, R:0.9}
**context:** ml (Qwen port, Phase 4.0 — bottleneck identification)
**evidence:**
- claim: "At pos=6+, wall clock 96.8 ms/tok on 9B Q4_K_M. Per-token Metal dispatches: 201 gemv + 24 delta_net + 8 attn = 233 sync barriers. gemv wait 60.5 ms (62%), dn wait 12.9 ms (13%), attn wait 2.0 ms (2%), encode total 5.8 ms (6%), read total 2.9 ms (3%). Remaining ~12.7 ms is CPU glue (norm/activations) + 48/tok CPU-fallback matvecs below Metal size threshold."
  source: bin/qwen35_sync_profile 5 10 2026-04-23 (Qwen35Metal::Profile instrumentation)
  verified_at: 2026-04-23
  decay_trigger: forward path changes dispatch count or adds batching
- claim: "Per-gemv wait = 301 μs (kernel exec + commit/sync overhead). Encode = 23 μs/dispatch (setup only). Sync overhead per barrier ≈ 100–150 μs based on kernel-size estimate."
  source: same profiler run
  verified_at: 2026-04-23
  decay_trigger: batched-submission landing
- claim: "All 70 specs still pass with the no-op-when-disabled Profile hooks added to matmul_gemv_buf / matmul_q4k_gemm_buf / delta_net_step / attn_decode."
  source: make spec 2026-04-23
  verified_at: 2026-04-23
  decay_trigger: dispatch paths refactored
**architecture:**
- qwen35_metal.cr `Qwen35Metal::Profile` submodule: module-level counters + ns accumulators for gemv/gemm/dn/attn, split into encode (everything up to end_encoding), wait (commit→wait), read (out_buf.read). `Profile.enable!` / `Profile.disable!` / `Profile.reset` / `Profile.report_io`. Each bump checks `@@enabled` first; disabled path is a branch-and-return, ~1 ns.
- All four Metal dispatch sites changed from `cmd.commit_and_wait` to `cmd.commit; cmd.wait` sandwiched by `Time.instant` markers; the commit/wait split lets us measure synchronization cost vs kernel compute cost separately in future phases.
- `metal_matvec_or_nil` in qwen35_cpu.cr bumps `Profile.cpu_fallback` when size thresholds or availability rule out Metal, quantifying how much work is missing the GPU.
- bin/qwen35_sync_profile: prefills N tokens then runs n_runs decode steps with Profile enabled, prints the report + wall clock.
**builds_on:** [LM-claude-PHASE3a-VERIFIED]
**unblocks:** [Phase 4.1: batched command buffer, Phase 4.2–4.4]
**note:** Bedrock found. The decode step is **sync-bound**, not compute-bound. 233 `commit_and_wait` barriers dominate the budget at ~150 μs sync overhead each. Phase 4.1 (single command buffer per token) predicts ~35 ms/tok saved → ~62 ms/tok. CPU-fallback count (48/tok) hints that small matmuls (k_norm, β projection) are round-tripping to CPU — Phase 4.4 will move them to GPU. The fact that Metal's `attn` kernel is only 2% of budget (vs CPU attn at ~10%) means Phase 3a's real benefit was already captured by enabling the batched-dispatch opportunity for attention, not the raw kernel win; long-context benefit remains as measured in [LM-claude-PHASE3a-VERIFIED].

### [LM-claude-PHASE4-PLAN-v2] Phase 4 plan revision — evidence-driven
**status:** proposed
**trust:** {F:0.7, G:0.7, R:0.9}
**context:** ml (Qwen port, Phase 4 — close gap to bandwidth floor)
**target:** approach 12.5 ms/tok theoretical floor (M2 Max, 400 GB/s, 5 GB Q4_K_M model). Prior mandate ≤20.7 ms/tok; user escalation (2026-04-23) to maximalist target near the floor.
**evidence:**
- claim: "Phase 4 split into 4.0/4.1/4.2/4.3/4.4 with testable hypotheses at each step, based on the Phase 4.0 instrumentation findings."
  source: Quadrumvirate discussion with user 2026-04-23; prediction anchored in 233-sync / 150-μs-sync-overhead measurement
  verified_at: 2026-04-23 (plan; steps to be verified individually)
  decay_trigger: plan revised after a phase reveals different bottleneck
**plan:**
- **4.0** (done): instrument Metal dispatches; confirm sync-bound.
- **4.1**: single command buffer per token — encode all 233 dispatches, one commit+wait. Hypothesis: save ≥25 ms/tok. Risk: command buffer size limits, dependency tracking between kernels.
- **4.2**: persistent scratch buffers (DeltaNet q/k/v/g/β/out, attn q/gate/out, matmul x_buf/out_buf). Hypothesis: save ~5 ms/tok from MetalBuffer.new + free + zero on every dispatch.
- **4.3**: kernel fusion — rms_norm+projection, pre-attn (K/V projection + position write), post-attn (gate+o_proj). Hypothesis: reduce dispatch count from 233 to ~80–120 and remove readback of intermediates.
- **4.4**: move remaining CPU glue to GPU — rms_norm (currently CPU), softplus/sigmoid (currently CPU), small matmuls below threshold. Eliminate all 48 cpu_fallback matvecs per token. Hypothesis: close remaining gap and enable full single-command-buffer flow.
**guard rails:**
- After each step: re-run `bin/qwen35_sync_profile` to quantify the gain + catch regressions.
- After each step: run `make spec` + `./ml "The capital of France is" 5` to confirm output unchanged.
- Quadrumvirate (Cassandra/Daedalus/Maieutic/Adversary) before starting and after completing each sub-phase.

### [LM-claude-PHASE4.1a-VERIFIED] Batched within-layer projections — 18 ms/tok saved
**status:** verified
**trust:** {F:0.9, G:0.7, R:0.9}
**context:** ml (Qwen port, Phase 4 — sync reduction via batch encoding)
**result:** batching Q/K/V (full-attn) + gate/up (ffn) + qkv/gate/α/β (recurrent) + gate/up (recurrent ffn) into one `matmul_many` call each cuts 48 `commit_and_wait` barriers per token (2330 → 1850) and **shaves ~18 ms/tok** (85.45 vs 103.83 ms/tok, paired t=3.57 across 8 interleaved trials, p≈0.009). Long-context pos=257: 109 ms vs 146 ms total forward (confirmed at longer context too).
**evidence:**
- claim: "4.1a saves 18.38 ± 5.15 ms/tok (mean paired diff over 8 trials, 17.7% faster)"
  source: `./bin/qwen35_sync_profile 5 10` interleaved with `QWEN35_BATCH_OFF=1` toggle, 8 trials each
  verified_at: 2026-04-23
  decay_trigger: kernel fusion (Phase 4.3) changes the projection dispatch structure
- claim: "correctness preserved — 70/70 specs, top logit 11.423702 unchanged"
  source: `make spec` after implementation; `attn_decode_spec` cos=1.0 max|Δ|=4.47e-8
  verified_at: 2026-04-23
  decay_trigger: kernel code changes
- claim: "single-run measurements are unreliable for sub-20ms effects — thermal variance is ~10 ms/tok"
  source: first single-run showed 126 ms (regression); paired-interleaved showed -18 ms (real win). Within-condition spread 20-34 ms.
  verified_at: 2026-04-23
  decay_trigger: M2 Max thermal characteristics change (different machine, background load)
**implementation:**
- `Qwen35CPU.qmatvec_many(qws, x)` — dispatches all eligible qws in one encoder; falls back to per-qw when any is below Metal threshold or Metal unavailable.
- `Qwen35Metal.matmul_many(qws, x)` — uploads `x` once, encodes N dispatches on one `MTLComputeEncoder`, one `commit_and_wait`, reads N outputs. Counts as ONE `gemv` sync in Profile.
- `ENV["QWEN35_BATCH_OFF"]="1"` disables batching at runtime for future A/B.
- 4 call sites wired: full-attn Q/K/V + ffn gate/up; recurrent qkv/gate/α/β + ffn gate/up.
**builds_on:** [LM-claude-PHASE4.0-VERIFIED]
**insight:** Predicted 48×150μs = 7 ms saved from sync count alone. Measured 18 ms — ~11 ms extra comes from reduced Metal state transitions and amortized encoder setup. Single-run measurement was misleading (126 ms regression vanished under interleaved A/B). Lesson: for sub-20ms effects on a thermally-constrained laptop, paired interleaved trials are mandatory.

### [LM-claude-PHASE4.2-VERIFIED] Persistent Metal scratch pool — 9.6 ms/tok saved
**status:** verified
**trust:** {F:0.9, G:0.7, R:0.9}
**context:** ml (Qwen port, Phase 4 — cut per-dispatch allocator churn)
**result:** per-dispatch scratch buffers (q/gate/out for attn_decode, q/k/v/g/b/out for delta_net, x/out for matmul_gemv/gemm, x + up-to-8 out slots for matmul_many) are now allocated once on first use and reused per `(tag, size)` key. Cuts the ~550 `MetalBuffer.new`+`finalize` pairs per token to effectively zero steady-state. **Saves ~9.6 ms/tok** (17 cool paired trials, 15/17 wins, paired t=-3.75, p≈0.0017). Long-context pos=257 recurrent stack: 70 vs 83 ms (~15%).
**evidence:**
- claim: "4.2 saves ~9.57 ms/tok (mean paired diff over 17 interleaved cool trials, 10% faster)"
  source: `./bin/qwen35_sync_profile 5 10` interleaved with `QWEN35_SCRATCH_OFF=1` toggle
  verified_at: 2026-04-23
  decay_trigger: Phase 4.3 (kernel fusion) consolidates scratch shapes; Phase 4.4 eliminates CPU→Metal upload sites
- claim: "correctness preserved — 70/70 specs, top logit 11.423702 unchanged, attn_decode cos=1.0 max|Δ|=4.47e-8"
  source: `make spec` after implementation
  verified_at: 2026-04-23
  decay_trigger: kernel code changes
- claim: "thermal throttling above ~100 ms/tok regime masks the effect — 4.2 ON even appeared slower than OFF under sustained load (T14-T16 hit 103/126/145 ms)"
  source: A/B trials T14-T16 under sustained load; recovery on cool-down at T17-T20
  verified_at: 2026-04-23
  decay_trigger: machine changes / ambient temperature changes
**implementation:**
- `Qwen35Metal::Scratch.get(tag, bytes)` — `{Symbol, Int64}`-keyed Hash lookup; miss allocates `ML::MetalBuffer.new` and inserts; `ENV["QWEN35_SCRATCH_OFF"]="1"` bypasses.
- `MANY_SLOT_TAGS` array of 8 per-slot tags for `matmul_many` output buffers (must not alias within one encoder).
- Replaced 10 call-site `MetalBuffer.new` allocations: attn_decode(3), delta_net_step(6), matmul_gemv_buf(2), matmul_q4k_gemm_buf(2), matmul_many(1 x + N out).

### [LM-codex-DN-FUSE-1] Recurrent post-processing fused on GPU
**status:** verified
**trust:** {F:0.9, G:0.7, R:0.9}
**context:** ml (Qwen port, decode optimization after Phase 3b)
**target:** remove the extra recurrent-layer round-trip `delta_net_step -> CPU RMSNorm*SiLU -> ssm_out matvec`.
**evidence:**
- claim: "Added fused recurrent Metal route: `delta_net_step -> delta_net_post_norm_gate -> ssm_out GEMV` in one command buffer, with CPU fallback preserved behind `QWEN35_DN_FUSE_OFF=1`."
  source: `src/ml/gguf/kernels/delta_net.metal`, `src/ml/gguf/qwen35_metal.cr`, `src/ml/gguf/qwen35_cpu.cr`
  verified_at: 2026-04-23
  decay_trigger: delta_net kernel glue or recurrent forward path changes
- claim: "Correctness preserved: `spec/qwen35_delta_net_spec.cr` passes (`y` cos=1.0, max|Δ|=1.21e-8; `state` cos=1.0, max|Δ|=2.98e-8) and `spec/qwen35_deltanet_spec.cr` passes (2 examples, 0 failures)."
  source: targeted specs on 2026-04-23 with Metal bridge linked
  verified_at: 2026-04-23
  decay_trigger: recurrent route modified
- claim: "Warm A/B on `bin/qwen35_sync_profile.cr` (prefill=5, decode=5) improved wall time from 246.66 ms/tok (`QWEN35_DN_FUSE_OFF=1`) to 239.34 ms/tok (default), saving 7.32 ms/tok (~3.0%)."
  source: `crystal run bin/qwen35_sync_profile.cr --link-flags=\".../build/bridge.o ...\" -- 5 5` on 2026-04-23
  verified_at: 2026-04-23
  decay_trigger: profiling conditions changed materially
- claim: "Metal sync count dropped from 885 to 765 over 5 decode tokens (24 fewer syncs/token), and GEMV calls dropped from 725 to 605, matching one removed `ssm_out` round-trip per recurrent layer."
  source: same warm A/B profile run
  verified_at: 2026-04-23
  decay_trigger: profiler accounting changed
**architecture:**
- `kernels/delta_net.metal`: new `delta_net_post_norm_gate` kernel computes per-head RMSNorm in place on the DeltaNet output and multiplies by `silu(z)`.
- `qwen35_metal.cr`: new `dn_post_pipeline`; new `delta_net_project(...)` entrypoint encodes three kernels in one command buffer: DeltaNet step, post-norm gate, `ssm_out` GEMV.
- `qwen35_cpu.cr`: `forward_recurrent_layer` now prefers the fused route and falls back to the previous CPU post-processing path when disabled or unsupported.
**builds_on:** [LM-claude-PHASE3b-VERIFIED]
**unblocks:** [Phase 4.1: larger decode-wave fusion, Phase 4.4: move remaining CPU fallback ops to GPU]
**note:** This was the right class of optimization but not the final lever. Saving ~7.3 ms/tok is meaningful and proves recurrent post-processing was still paying a real sync tax, yet decode remains far from the 14.1 ms/tok M2 Max bandwidth floor. The next likely gains are broader per-token command-buffer fusion and eliminating the remaining 240 CPU fallback matvecs over 5 tokens.

### [LM-codex-BATCH-SPLIT-1] Mixed qmatvec batches keep large projections on Metal
**status:** verified
**trust:** {F:0.9, G:0.7, R:0.9}
**context:** ml (Qwen port, recurrent projection batching)
**evidence:**
- claim: "Changed `qmatvec_many` from all-or-nothing routing to mixed routing: supported qweights are batched on Metal, only unsupported/disabled slots fall back."
  source: `src/ml/gguf/qwen35_cpu.cr`
  verified_at: 2026-04-23
  decay_trigger: qmatvec routing changed
- claim: "Escalated batched routing to include the small recurrent `alpha` / `beta` projections too when they ride in the same Metal batch; `cpu_fallback matvecs` dropped from 240 to 0 over 5 decode tokens."
  source: `bin/qwen35_sync_profile.cr` 2026-04-23
  verified_at: 2026-04-23
  decay_trigger: batching heuristics changed
- claim: "Wall decode improved from ~239 ms/tok (pre-change profile) to ~139–156 ms/tok, then stabilized much lower after follow-up fusions."
  source: same profile sequence on 2026-04-23
  verified_at: 2026-04-23
  decay_trigger: later optimizations supersede
**note:** The hidden bug was not 'big projections go CPU' but 'big projections lose batching because tiny projections poison `all_eligible`'. Fixing that unlocked the later, larger wins.

### [LM-codex-FFN-FUSE-1] SwiGLU FFN fused on GPU for all layers
**status:** verified
**trust:** {F:0.9, G:0.7, R:0.9}
**context:** ml (Qwen port, Phase 4 decode optimization)
**evidence:**
- claim: "Added `qwen35_swiglu_mul` Metal kernel and fused route `gate_proj + up_proj -> swiglu -> down_proj` in one command buffer for both full-attn and recurrent layers."
  source: `src/ml/gguf/kernels/ffn_qwen35.metal`, `src/ml/gguf/qwen35_metal.cr`, `src/ml/gguf/qwen35_cpu.cr`
  verified_at: 2026-04-23
  decay_trigger: FFN route modified
- claim: "End-to-end correctness preserved: `spec/qwen35_fullattn_spec.cr`, `spec/qwen35_deltanet_spec.cr`, and `spec/qwen35_forward_spec.cr` all pass; forward top token remains 198 with logit ≈11.423701."
  source: targeted specs on 2026-04-23
  verified_at: 2026-04-23
  decay_trigger: FFN kernels or layer glue changed
- claim: "Profile after FFN fusion: total metal syncs 485 over 5 tokens, warm decode ≈95.68 ms/tok."
  source: `bin/qwen35_sync_profile.cr` 2026-04-23
  verified_at: 2026-04-23
  decay_trigger: profiling conditions changed
**note:** This was the first clear structural win after the mixed-batch routing fix: removing one FFN round-trip per layer pays on all 32 layers.

### [LM-codex-READ-SHARED-1] Hot-path results read from unified memory directly
**status:** verified
**trust:** {F:0.85, G:0.7, R:0.8}
**context:** ml (Apple Silicon unified-memory optimization)
**evidence:**
- claim: "Replaced hot `gs_buffer_read` calls in `Qwen35Metal` with direct `contents` copies from unified-memory MTLBuffers."
  source: `src/ml/gguf/qwen35_metal.cr`
  verified_at: 2026-04-23
  decay_trigger: MetalBuffer read path refactored
- claim: "On warm sync-profile, `gemv read` dropped to ~1.23 ms over 5 tokens and wall decode reached ~95.68 ms/tok."
  source: `bin/qwen35_sync_profile.cr` 2026-04-23
  verified_at: 2026-04-23
  decay_trigger: profiling conditions changed
**note:** Another direct application of the unified-memory lesson from the mmap work: avoid crossing the bridge when the bytes are already CPU-visible.

### [LM-codex-RECURRENT-PREP-1] Recurrent conv/L2 prep moved to GPU and fused into DeltaNet wave
**status:** verified
**trust:** {F:0.9, G:0.7, R:0.9}
**context:** ml (Qwen port, recurrent layers dominate decode)
**evidence:**
- claim: "Added GPU recurrent-prep kernels (`qwen35_recurrent_ab`, `qwen35_recurrent_conv`, `qwen35_recurrent_shift`, `qwen35_l2_heads`) and new fused route from recurrent projections through conv/L2 into DeltaNet."
  source: `src/ml/gguf/kernels/recurrent_qwen35.metal`, `src/ml/gguf/qwen35_metal.cr`, `src/ml/gguf/qwen35_cpu.cr`
  verified_at: 2026-04-23
  decay_trigger: recurrent kernels or route changed
- claim: "Then fused recurrent-prep and DeltaNet/project into a single GPU wave, cutting total metal syncs from 485 to 365 over 5 tokens."
  source: `bin/qwen35_sync_profile.cr` 2026-04-23
  verified_at: 2026-04-23
  decay_trigger: recurrent command-buffer structure changed
- claim: "Warm sync-profile reached 58.43 ms/tok; phase profile reached 60.5 ms total forward with recurrent layers down to 42.5 ms total (1.77 ms/layer)."
  source: `bin/qwen35_sync_profile.cr`, `bin/qwen35_phase_profile.cr` 2026-04-23
  verified_at: 2026-04-23
  decay_trigger: profiling conditions changed
- claim: "Recurrent layer correctness preserved: `spec/qwen35_deltanet_spec.cr` passes after adapting it to read `conv_state_buf` / `ssm_state_buf`."
  source: `spec/qwen35_deltanet_spec.cr` 2026-04-23
  verified_at: 2026-04-23
  decay_trigger: spec or recurrent route changed
**note:** This was the biggest win of the session. The remaining path is now much closer to a 'one sync per semantic chunk' design, though still far from the single-wave ideal needed to approach the M2 Max bandwidth floor.
- `upload_weights` (test-only path) intentionally left unchanged — called once, not per token.
**builds_on:** [LM-claude-PHASE4.1a-VERIFIED]
**insight:** Predicted 500 allocs × ~10 μs = 5 ms/tok. Measured 9.6 ms — the extra likely comes from `finalize` / ARC release work on the old buffers, which amortizes outside Metal timing but shows up in wall clock. Thermal throttling above ~100 ms/tok swamped the signal; cool-trial filtering was necessary to extract a clean measurement.

### [LM-codex-FULLATTN-PREP-1] Full-attention prep/output fused on GPU
**status:** verified
**trust:** {F:0.9, G:0.7, R:0.9}
**context:** ml (Qwen port, full-attention decode)
**evidence:**
- claim: "Added `qwen35_split_qgate`, `qwen35_rmsnorm_heads`, `qwen35_rope_partial`, and `qwen35_kv_write` kernels plus `Qwen35Metal.full_attn_project`, so decode full-attn now runs `q/k/v -> split/norm/rope -> KV write -> attention -> out proj` in one GPU wave."
  source: `src/ml/gguf/kernels/fullattn_qwen35.metal`, `src/ml/gguf/qwen35_metal.cr`, `src/ml/gguf/qwen35_cpu.cr`
  verified_at: 2026-04-23
  decay_trigger: full-attn kernels or route changed
- claim: "Correctness preserved: `spec/qwen35_fullattn_spec.cr` and `spec/qwen35_forward_spec.cr` pass; forward top token remains 198 with logit ≈11.4237."
  source: targeted specs on 2026-04-23
  verified_at: 2026-04-23
  decay_trigger: full-attn math or routing changed
- claim: "A/B sync profile improved from 55.53 ms/tok and 365 syncs (fuse off) to 51.71 ms/tok and 325 syncs (fuse on) over 5 decode tokens."
  source: `bin/qwen35_sync_profile.cr` 2026-04-23
  verified_at: 2026-04-23
  decay_trigger: profiling conditions changed
**note:** This removed one per-full-attn-layer round-trip but did not change the session's main bottleneck; recurrent layers still dominated after this step.

### [LM-codex-RECURRENT-LAYER-FUSE-1] Recurrent layers fused through FFN and residuals
**status:** verified
**trust:** {F:0.92, G:0.75, R:0.92}
**context:** ml (Qwen port, dominant decode hotspot)
**evidence:**
- claim: "Added generic vector kernels `qwen35_add_rmsnorm` and `qwen35_add_vec`, then extended the recurrent GPU route to keep `inp -> recurrent attention -> residual add + post-attn RMSNorm -> SwiGLU FFN -> residual add` in one command buffer."
  source: `src/ml/gguf/kernels/ffn_qwen35.metal`, `src/ml/gguf/qwen35_metal.cr`, `src/ml/gguf/qwen35_cpu.cr`
  verified_at: 2026-04-23
  decay_trigger: recurrent layer route or vector kernels changed
- claim: "Correctness preserved: `spec/qwen35_deltanet_spec.cr`, `spec/qwen35_delta_net_spec.cr`, and `spec/qwen35_forward_spec.cr` pass; forward top token remains 198 with logit ≈11.423702."
  source: targeted specs on 2026-04-23
  verified_at: 2026-04-23
  decay_trigger: recurrent layer math changed
- claim: "A/B sync profile improved from 64.92 ms/tok and 325 syncs (recurrent-layer fuse off) to 39.62 ms/tok and 205 syncs (recurrent-layer fuse on) over 5 decode tokens."
  source: `bin/qwen35_sync_profile.cr` 2026-04-23
  verified_at: 2026-04-23
  decay_trigger: profiling conditions changed
- claim: "Phase profile with the fused recurrent layer reached ~40.4 ms total forward; recurrent layers dropped to ~27.4 ms total (1.14 ms/layer)."
  source: `bin/qwen35_phase_profile.cr` 2026-04-23
  verified_at: 2026-04-23
  decay_trigger: profiling conditions changed
**note:** This was the decisive structural win after the earlier prep fusions. It eliminated the separate FFN round-trip on all 24 recurrent layers.

### [LM-codex-FULL-LAYER-FUSE-1] Full-attention layers fused through FFN and residuals
**status:** verified
**trust:** {F:0.9, G:0.7, R:0.88}
**context:** ml (Qwen port, finishing per-layer glue removal)
**evidence:**
- claim: "Extended the full-attn GPU route to keep `inp -> full-attn -> residual add + post-attn RMSNorm -> SwiGLU FFN -> residual add` in one command buffer."
  source: `src/ml/gguf/qwen35_metal.cr`, `src/ml/gguf/qwen35_cpu.cr`
  verified_at: 2026-04-23
  decay_trigger: full-attn layer route changed
- claim: "Correctness preserved: `spec/qwen35_fullattn_spec.cr`, `spec/qwen35_forward_spec.cr`, `spec/qwen35_deltanet_spec.cr`, and `spec/qwen35_delta_net_spec.cr` all pass."
  source: targeted specs on 2026-04-23
  verified_at: 2026-04-23
  decay_trigger: full-attn layer math changed
- claim: "Best current profile after recurrent + full-attn layer fusion reached ~39.6–40.5 ms/tok with 165 total Metal syncs over 5 decode tokens."
  source: `bin/qwen35_sync_profile.cr`, `bin/qwen35_phase_profile.cr` 2026-04-23
  verified_at: 2026-04-23
  decay_trigger: profiling conditions changed
- claim: "At this point syncs equal one round-trip per layer plus the output head: 165 syncs / 5 tokens = 33 syncs/token = 32 layers + 1 head."
  source: arithmetic on sync-profile counts
  verified_at: 2026-04-23
  decay_trigger: decoder scheduling changes
**note:** This likely exhausts the easy per-layer fusion space. The next ceiling is cross-layer residency of the hidden state, not another local fusion.

### [LM-codex-NEXT-BOTTLENECK-1] Remaining structural ceiling is layer-to-layer CPU round-trip
**status:** verified
**trust:** {F:0.88, G:0.8, R:0.9}
**context:** ml (Qwen port, next optimization branch)
**evidence:**
- claim: "After the current fusions, the hidden state still returns to CPU after every layer because `forward_*_layer` APIs return `Array(Float32)` and the decoder loop feeds the next layer from CPU-owned activations."
  source: `src/ml/gguf/qwen35_cpu.cr`, `src/ml/gguf/qwen35_metal.cr`
  verified_at: 2026-04-23
  decay_trigger: decoder loop rewritten for GPU-resident activations
- claim: "The profile signature confirms this: 33 syncs/token remain (32 layers + output head), so orchestration cost is now dominated by layer boundaries rather than intra-layer glue."
  source: `bin/qwen35_sync_profile.cr` 2026-04-23
  verified_at: 2026-04-23
  decay_trigger: compute-graph / decoder-wave scheduling implemented
**note:** The next serious optimization is TODO 4.4 in spirit: ping-pong hidden-state buffers across the whole decoder token wave and read back only the final logits.

### [LM-codex-DECODE-WAVE-1] Whole-token decode wave removes per-layer round-trips
**status:** verified
**trust:** {F:0.92, G:0.75, R:0.92}
**context:** ml (Qwen port, decoder scheduling)
**evidence:**
- claim: "Added a GPU-resident decode route that uploads the embedding once, keeps hidden activations on GPU across all 32 layers with ping-pong buffers, and reads back only final logits."
  source: `src/ml/gguf/qwen35_cpu.cr`, `src/ml/gguf/qwen35_metal.cr`
  verified_at: 2026-04-23
  decay_trigger: decoder routing or layer kernels changed
- claim: "Correctness preserved vs the previous layer-by-layer path: first-token A/B gives top token 198 on both paths, cosine ≈ 1.0, max diff ≈ 1.24e-5; targeted Qwen specs still pass."
  source: local A/B check + `spec/qwen35_forward_spec.cr`, `spec/qwen35_fullattn_spec.cr`, `spec/qwen35_deltanet_spec.cr`, `spec/qwen35_delta_net_spec.cr`
  verified_at: 2026-04-23
  decay_trigger: wave path math or buffer residency changes
- claim: "Sync profile dropped from 165 total Metal syncs over 5 decode tokens on the fused layer-by-layer path to 5 syncs over 5 tokens on the decode-wave path."
  source: `bin/qwen35_sync_profile.cr` with and without `QWEN35_DECODE_WAVE_OFF=1` on 2026-04-23
  verified_at: 2026-04-23
  decay_trigger: profiling conditions changed
- claim: "Short warm decode improved from ~34.6 ms/tok (layer-by-layer fused path) to ~27.0-29.0 ms/tok on the decode-wave path; longer 20-token run measured ~29.95 ms/tok."
  source: `bin/qwen35_sync_profile.cr` 2026-04-23
  verified_at: 2026-04-23
  decay_trigger: thermal state / profiling conditions changed
- claim: "A fresh top1 decode profile after prefill work measured 8 decode-wave syncs over 8 tokens, `164.53 ms` GPU wait, and `24.05 ms/tok` wall at prompt=64/gen=8."
  source: `CRYSTAL_CACHE_DIR=/tmp/cogni_ml_crystal_cache_current_decode_profile QWEN35_PROFILE_TOP1=1 crystal run --link-flags=\"$(pwd)/build/bridge.o -framework Metal -framework Foundation -lc++\" bin/qwen35_sync_profile.cr -- 64 8`
  verified_at: 2026-04-24
  decay_trigger: thermal state, power state, decode wave route, or profiling conditions changed
**note:** This completes the obvious orchestration optimization. The remaining gap to llama.cpp is now mostly kernel efficiency and long-context attention cost, not CPU/GPU boundary churn.

### [LM-codex-QWEN-VS-LLAMA-1] Matched 64/64 benchmark shows decode gap is now moderate
**status:** verified
**trust:** {F:0.9, G:0.8, R:0.92}
**context:** ml (Qwen port, benchmarking discipline)
**evidence:**
- claim: "Added a dedicated harness `bin/benchmark_qwen_vs_llama.cr` that measures cogni-ml and local `llama-bench` under matched `pp/tg` settings."
  source: `bin/benchmark_qwen_vs_llama.cr`
  verified_at: 2026-04-23
  decay_trigger: harness logic or benchmark settings changed
- claim: "On `64/64`, decode measured `cogni-ml p50 ≈ 33.31 tok/s` vs `llama.cpp avg ≈ 39.49 tok/s`, a gap of about `-15.66%`."
  source: `crystal run bin/benchmark_qwen_vs_llama.cr -- --reps=3 --warmup=1 --prompt=64 --gen=64` on 2026-04-23
  verified_at: 2026-04-23
  decay_trigger: wave path or local llama.cpp build changed
- claim: "The same harness reports prefill at `~33 tok/s` vs `~447 tok/s`, but this is not an apples-to-apples prefill comparison yet because cogni-ml still performs prompt ingestion as repeated decode steps rather than a true batched prefill path."
  source: same harness run + code inspection of `Qwen35CPU.forward`
  verified_at: 2026-04-23
  decay_trigger: true Qwen prefill path implemented
**note:** This is the right current framing: decode is now close enough for kernel work to matter directly; prefill remains a separate engineering problem, not a fair speed comparison today.

### [LM-codex-Q5K-DISPATCH-1] Q5_K decode kernel was under-dispatched relative to llama.cpp
**status:** verified
**trust:** {F:0.9, G:0.82, R:0.92}
**context:** ml (Qwen port, recurrent decode kernels)
**evidence:**
- claim: "Our Q5_K GEMV path was using `NR0=2` semantics everywhere, but llama.cpp Metal uses `N_R0_Q5_K=1` while keeping `N_SG_Q5_K=2`."
  source: `src/ml/gguf/kernels/gemm_q56k.metal`, `src/ml/gguf/qwen35_metal.cr`, `/Users/sergey/SrcArchives/AI/llama.cpp/ggml/src/ggml-metal/ggml-metal-impl.h`
  verified_at: 2026-04-23
  decay_trigger: GEMV tiling changed again
- claim: "Fixing the Q5_K kernel constant to `MV5_NR0=1` and making Crystal-side grid sizing pipeline-aware improved the recurrent `attn_qkv` microbench from `~0.733 ms` to `~0.643 ms` for `4096 -> 8192`."
  source: local `tmp_q5_micro.cr` microbench on 2026-04-23
  verified_at: 2026-04-23
  decay_trigger: Q5_K kernel or dispatch wrapper changes
- claim: "After the fix, targeted Qwen specs still pass and the decode-wave sync profile improved to `~27.96 ms/tok` on a `20/20` run (`~28.74 ms/tok` on `5/5`)."
  source: `spec/qwen35_forward_spec.cr`, `spec/qwen35_fullattn_spec.cr`, `spec/qwen35_deltanet_spec.cr`, `spec/qwen35_delta_net_spec.cr`, `bin/qwen35_sync_profile.cr` on 2026-04-23
  verified_at: 2026-04-23
  decay_trigger: decode-wave kernels or profiling conditions changed
**note:** The main verified win in this branch is the Q5_K dispatch correction, not a broad GEMV retune.

### [LM-codex-DELTANET-128-REFUTE-1] 128-thread DeltaNet rewrite is falsified and should stay reverted
**status:** refuted
**trust:** {F:0.93, G:0.75, R:0.94}
**context:** ml (Qwen port, recurrent kernel experiments)
**evidence:**
- claim: "A rewrite of `delta_net_step` that tried to use `128` threads per head and stripe rows across four simdgroups produced incorrect output: only rows `0,4,8,...` were written, and `spec/qwen35_delta_net_spec.cr` dropped to `y cos ≈ 0.497`."
  source: local debug repro `tmp_delta_debug.cr` + `spec/qwen35_delta_net_spec.cr` on 2026-04-23
  verified_at: 2026-04-23
  decay_trigger: the experiment is retried with a different indexing model
- claim: "The branch was reverted and the previous one-simdgroup DeltaNet kernel restored; correctness returned immediately (`7 examples, 0 failures`)."
  source: targeted Qwen specs on 2026-04-23
  verified_at: 2026-04-23
  decay_trigger: DeltaNet kernel changed again
**note:** Do not retry this exact frame. If DeltaNet is revisited, first prove the simdgroup/threadgroup indexing model with a minimal probe kernel before touching recurrence math again.

### [LM-codex-DELTANET-128-V2-1] Corrected 128-thread DeltaNet path is now the default decode path
**status:** verified
**trust:** {F:0.92, G:0.78, R:0.90}
**context:** ml (Qwen port, recurrent kernel experiments)
**evidence:**
- claim: "The earlier `rows 0,4,8,...` failure was not caused by Metal simdgroup indexing or row-striping itself. Minimal probes for `simdgroup_index_in_threadgroup`, row ownership, and per-row `float4` read/modify/write all behaved correctly."
  source: local probes `tmp_metal_simd_probe.cr`, `tmp_metal_row_probe.cr`, `tmp_metal_row_math_probe.cr` on 2026-04-23
  verified_at: 2026-04-23
  decay_trigger: N/A
- claim: "A standalone `delta_net_step_128_probe` using four simdgroups per head matched the CPU reference at machine precision (`y_cos≈1.0`, `s_cos≈1.0`, `max|Δ|≈1e-8`)."
  source: local probe `tmp_delta128_probe.cr` on 2026-04-23
  verified_at: 2026-04-23
  decay_trigger: the kernel shape is modified
- claim: "In a direct kernel microbench on Qwen 9B shapes (`h_k=16`, `h_v=32`, `s=128`), the corrected 128-thread kernel was materially faster than the 32-thread kernel (`avg 0.5239 ms` vs `1.0199 ms`, `p50 0.3742 ms` vs `1.0235 ms`)."
  source: local `tmp_delta_microbench.cr` on 2026-04-23
  verified_at: 2026-04-23
  decay_trigger: kernel math or dispatch changes
- claim: "A narrow runtime toggle (`QWEN35_DN_128`) enabled apples-to-apples A/B testing. With the same built `qwen35_sync_profile` binary, decode improved from `28.33 -> 24.77 ms/tok` mean on `5/5` runs and from `30.60 -> 25.80 ms/tok` mean on `20/20` runs."
  source: built `/tmp/qwen35_sync_profile_bench` with repeated local runs on 2026-04-23
  verified_at: 2026-04-23
  decay_trigger: decode-wave scheduler or recurrent kernels change
- claim: "On the matched `64/64` benchmark against local `llama.cpp`, decode improved from `32.04 tok/s` (`-21.73%` vs llama) to `39.10 tok/s` (`-4.06%` vs llama). Prefill also improved from `31.78` to `37.91 tok/s`, though that comparison remains structurally unfair because cogni-ml still ingests prompt tokens as repeated decode."
  source: built `/tmp/benchmark_qwen_vs_llama` with `--reps=3 --warmup=1 --prompt=64 --gen=64` on 2026-04-23
  verified_at: 2026-04-23
  decay_trigger: benchmark harness or decode path changes
- claim: "The 128-thread variant is now the default path, with `QWEN35_DN_128=0` preserving the old 32-thread fallback; targeted Qwen specs stay green in both modes."
  source: `src/ml/gguf/qwen35_metal.cr`, `spec/qwen35_forward_spec.cr`, `spec/qwen35_fullattn_spec.cr`, `spec/qwen35_deltanet_spec.cr`, `spec/qwen35_delta_net_spec.cr` on 2026-04-23
  verified_at: 2026-04-23
  decay_trigger: DeltaNet dispatch logic changes
**note:** This is now a verified product win for decode, not just a kernel-only lead. The gap to `llama.cpp` on matched decode shrank to about `4%`, so the next work should attack the remaining recurrent/full-layer overhead rather than revisit DeltaNet threadgroup math.

### [LM-codex-RECURRENT-MICROFUSE-REFUTE-1] Two small recurrent micro-fusions did not improve decode and were reverted
**status:** refuted
**trust:** {F:0.86, G:0.72, R:0.88}
**context:** ml (Qwen port, recurrent kernel experiments)
**evidence:**
- claim: "Fusing `recurrent_shift` into `recurrent_conv` looked structurally safe but regressed end-to-end decode badly (`~48 ms/tok` on repeated `20/20` sync-profile runs), so it was reverted."
  source: local patch + `bin/qwen35_sync_profile.cr` on 2026-04-23
  verified_at: 2026-04-23
  decay_trigger: recurrent conv/state update design changes substantially
- claim: "Fusing the separate Q/K L2-norm kernels into one `qk-norm` kernel preserved correctness but did not improve decode (`~28.36 ms/tok` vs prior `~27.86 ms/tok` on `5/5`), so it was reverted."
  source: local patch + `bin/qwen35_sync_profile.cr` on 2026-04-23
  verified_at: 2026-04-23
  decay_trigger: recurrent norm scheduling changes substantially
**note:** Current evidence says the next recurrent win is unlikely to come from tiny encoder-count reductions. Focus on larger arithmetic kernels or a better-validated DeltaNet redesign.

### [LM-codex-VECRMS-SG-REFUTE-1] Simdgroup-first vector RMSNorm reductions were not a robust decode win
**status:** refuted
**trust:** {F:0.89, G:0.70, R:0.87}
**context:** ml (Qwen port, vector norm kernels)
**evidence:**
- claim: "Alternative `qwen35_rmsnorm_vec` / `qwen35_add_rmsnorm` kernels that used simdgroup-first reductions preserved correctness (`7 examples, 0 failures`) but only gave mixed performance signals."
  source: local toggle experiment `QWEN35_VEC_RMS_SG=1` with targeted Qwen specs on 2026-04-23
  verified_at: 2026-04-23
  decay_trigger: vector RMSNorm kernel shape changes
- claim: "On the same built `qwen35_sync_profile` binary, `20/20` improved (`25.94 -> 25.16 ms/tok`) but `64/64` was effectively flat (`26.05 -> 26.06 ms/tok`) and short `5/5` runs were noisy."
  source: built `/tmp/qwen35_sync_profile_vecsg` with repeated local runs on 2026-04-23
  verified_at: 2026-04-23
  decay_trigger: decode workload or RMSNorm kernel changes
- claim: "A matched `64/64` benchmark against `llama.cpp` was too thermally noisy to support promotion; native decode moved only slightly (`40.41 -> 41.24 tok/s`) while llama.cpp varied widely between runs."
  source: built `/tmp/benchmark_qwen_vs_llama_vecsg` on 2026-04-23
  verified_at: 2026-04-23
  decay_trigger: fresh benchmark under calmer conditions
**note:** This path was reverted. The reduction strategy may still be viable later, but current evidence is too weak and inconsistent to justify extra kernel complexity.

### [LM-codex-Q6WIDE-REFUTE-1] Wider Q6_K GEMV threadgroups did not improve end-to-end decode
**status:** refuted
**trust:** {F:0.90, G:0.76, R:0.90}
**context:** ml (Qwen port, Q6_K decode kernels)
**evidence:**
- claim: "A bounded constant search on real Qwen weights found that `Q6_K` variants with different `(NSG, NR0)` trade off differently by shape: `NSG=4, NR0=2` helped recurrent `ffn_down`, while `NSG=1, NR0=4` helped the huge `lm_head` microbench."
  source: local `tmp_q6_variant_bench.cr` on 2026-04-23
  verified_at: 2026-04-23
  decay_trigger: Q6_K kernel math changes
- claim: "Promoting the most plausible global compromise (`NSG=4, NR0=2`) behind `QWEN35_Q6_WIDE=1` preserved correctness (`7 examples, 0 failures`) but did not improve decode on the same built `qwen35_sync_profile` binary."
  source: `spec/qwen35_forward_spec.cr`, `spec/qwen35_fullattn_spec.cr`, `spec/qwen35_deltanet_spec.cr`, `spec/qwen35_delta_net_spec.cr`, plus built `/tmp/qwen35_sync_profile_q6wide` on 2026-04-23
  verified_at: 2026-04-23
  decay_trigger: dispatch wrapper or benchmark workload changes
- claim: "Matched native-only runs were flat or slightly worse: `5/5` mean `24.93 -> 25.22 ms/tok`, `20/20` mean `25.53 -> 25.79 ms/tok`, `64/64` mean `27.47 -> 27.52 ms/tok`."
  source: repeated `/tmp/qwen35_sync_profile_q6wide` runs on 2026-04-23
  verified_at: 2026-04-23
  decay_trigger: fresh rerun after broader decode-path changes
**note:** This experiment was reverted. A single global Q6_K tiling choice is too blunt; future Q6_K work should be shape-aware or path-specific, not a repo-wide constant flip.

### [LM-codex-RECURRENT-PROJ-MAP-1] Current recurrent decode is dominated by a small set of repeated projections
**status:** verified
**trust:** {F:0.87, G:0.74, R:0.88}
**context:** ml (Qwen port, recurrent decode hotspots)
**evidence:**
- claim: "For one representative recurrent layer in Qwen 3.5 9B, the large routed projections are: `attn_qkv=Q5_K 4096->8192`, `attn_gate=Q4_K 4096->4096`, `ssm_out=Q4_K 4096->4096`, `ffn_gate=Q4_K 4096->12288`, `ffn_up=Q4_K 4096->12288`, `ffn_down=Q6_K 12288->4096`."
  source: local `tmp_recurrent_proj_bench.cr` type dump on 2026-04-23
  verified_at: 2026-04-23
  decay_trigger: model file or layer routing changes
- claim: "Production-route microbenches on those recurrent projections show the main repeated costs cluster around `attn_qkv` (`~0.66 ms`), the fused FFN pair `ffn_gate_up_many` (`~0.74 ms` for both together), and `ffn_down` (`~0.56 ms`), with `attn_gate`/`ssm_out` somewhat smaller."
  source: local `tmp_recurrent_proj_bench.cr` on 2026-04-23
  verified_at: 2026-04-23
  decay_trigger: GEMV batching or projection routing changes
**note:** The next likely win is not another broad Q6_K retune; it is more likely in the recurrent FFN / projection bundle, especially the repeated Q4_K `ffn_gate/up` + Q6_K `ffn_down` path.

### [LM-codex-FFN-INPLACE-REFUTE-1] In-place SwiGLU buffer reuse looked good in microbench but did not improve decode
**status:** refuted
**trust:** {F:0.90, G:0.76, R:0.89}
**context:** ml (Qwen port, FFN bundle experiments)
**evidence:**
- claim: "A standalone recurrent FFN bundle microbench using real Qwen weights showed a strong local win when reusing `ffn_up` as the SwiGLU output buffer instead of writing to a separate `ffn_comb` buffer (`1.1716 -> 0.9323 ms`)."
  source: local `tmp_ffn_bundle_bench.cr` on 2026-04-23
  verified_at: 2026-04-23
  decay_trigger: FFN kernel or buffer schedule changes
- claim: "A narrow production toggle (`QWEN35_FFN_INPLACE=1`) preserved correctness (`7 examples, 0 failures`) but did not produce a robust decode win on the same built `qwen35_sync_profile` binary."
  source: targeted Qwen specs plus built `/tmp/qwen35_sync_profile_ffn` on 2026-04-23
  verified_at: 2026-04-23
  decay_trigger: decode-wave FFN schedule changes
- claim: "Strict native-only `64/64` runs refuted the optimization: baseline mean `26.16 ms/tok` vs in-place mean `26.45 ms/tok`, despite one looser `64/64` two-run sample looking slightly better."
  source: repeated `/tmp/qwen35_sync_profile_ffn` runs on 2026-04-23
  verified_at: 2026-04-23
  decay_trigger: fresh rerun after other FFN-path changes
**note:** This was reverted. The gap between local bundle microbench and decode-level behavior suggests the next FFN win has to cut more than one scratch-buffer hop or reduce an actual kernel/dispatch bottleneck, not just alias a buffer.

### [LM-codex-WAVE-REC-PACK-REFUTE-1] Collapsing recurrent wave work into fewer compute encoders was not a decode win
**status:** refuted
**trust:** {F:0.90, G:0.74, R:0.88}
**context:** ml (Qwen port, decode-wave scheduling)
**evidence:**
- claim: "A bounded wave-path experiment packed each recurrent layer into fewer compute encoders (`recurrent chain` + `FFN chain`) behind `QWEN35_WAVE_REC_PACK=1`, while preserving correctness (`7 examples, 0 failures`)."
  source: local toggle experiment with `spec/qwen35_forward_spec.cr`, `spec/qwen35_fullattn_spec.cr`, `spec/qwen35_deltanet_spec.cr`, and `spec/qwen35_delta_net_spec.cr` on 2026-04-23
  verified_at: 2026-04-23
  decay_trigger: recurrent wave-path scheduling changes substantially
- claim: "Short `5/5` runs looked mildly better (`25.07 -> 24.74 ms/tok` mean), but the stricter native-only falsifiers were worse: `20/20` mean `25.28 -> 25.40 ms/tok`, `64/64` mean `27.82 -> 28.40 ms/tok` on the same built binary."
  source: repeated `/tmp/qwen35_sync_profile_wavepack` runs on 2026-04-23
  verified_at: 2026-04-23
  decay_trigger: fresh rerun after larger command-buffer or kernel changes
- claim: "After reverting the experiment, targeted specs were green again and a fresh local `20/20` rerun gave `27.15 ms/tok`, which is noisier/slower than the earlier `~25.3 ms/tok` anchor and therefore should not be promoted as a new headline benchmark."
  source: `CRYSTAL_CACHE_DIR=/tmp/cogni_ml_crystal_cache_wavepack_revert crystal spec ...` and `CRYSTAL_CACHE_DIR=/tmp/cogni_ml_crystal_cache_wavepack_revert_sync crystal run bin/qwen35_sync_profile.cr -- ... -- 20 20` on 2026-04-23
  verified_at: 2026-04-23
  decay_trigger: calm thermal rerun or broader decode-path changes
**note:** This path was reverted. Fewer compute encoders per recurrent layer are not automatically better; the next win likely needs kernel-level work or a more targeted reduction in recurrent arithmetic cost, not encoder-count reduction by itself.

### [LM-codex-REC1D-TG-REFUTE-1] Smaller recurrent `conv/shift` threadgroups were not a robust decode win
**status:** refuted
**trust:** {F:0.89, G:0.72, R:0.88}
**context:** ml (Qwen port, recurrent arithmetic dispatch)
**evidence:**
- claim: "A narrow dispatch-only experiment exposed `QWEN35_REC_1D_TG={64,128,256}` for recurrent `conv`/`shift` kernels without changing math, and correctness stayed green (`7 examples, 0 failures`)."
  source: local toggle experiment with targeted Qwen specs on 2026-04-23
  verified_at: 2026-04-23
  decay_trigger: recurrent kernel shape or dispatch changes
- claim: "Initial native-only runs suggested `128` might slightly help `20/20` and `64/64`, but the stricter alternating falsifier showed the signal was not robust: `20/20` mean `24.50 -> 24.20 ms/tok`, while `64/64` was effectively flat-to-worse (`27.00 -> 27.08 ms/tok`)."
  source: repeated `/tmp/qwen35_sync_profile_rec1d` runs on 2026-04-23
  verified_at: 2026-04-23
  decay_trigger: fresh rerun after recurrent kernel rewrites
- claim: "After reverting the dispatch toggle, targeted specs were green and a fresh `20/20` rerun returned to `24.80 ms/tok`."
  source: `CRYSTAL_CACHE_DIR=/tmp/cogni_ml_crystal_cache_rec1d_revert crystal spec ...` and `CRYSTAL_CACHE_DIR=/tmp/cogni_ml_crystal_cache_rec1d_revert_sync crystal run bin/qwen35_sync_profile.cr -- ... -- 20 20` on 2026-04-23
  verified_at: 2026-04-23
  decay_trigger: thermal rerun or broader decode changes
**note:** This path was reverted. The remaining recurrent gap is unlikely to be solved by a blind threadgroup-size tweak for `conv`/`shift`; the next credible win probably needs either a true recurrent-kernel rewrite or a more targeted projection-side improvement.

### [LM-codex-REC-CONV4-REFUTE-1] A vectorized `recurrent_conv4` rewrite changed logits and was rejected
**status:** refuted
**trust:** {F:0.93, G:0.78, R:0.93}
**context:** ml (Qwen port, recurrent arithmetic kernels)
**evidence:**
- claim: "A bounded rewrite introduced `qwen35_recurrent_conv4`, where each thread computed 4 adjacent channels, behind `QWEN35_REC_CONV4=1`."
  source: local patch to `src/ml/gguf/kernels/recurrent_qwen35.metal` and `src/ml/gguf/qwen35_metal.cr` on 2026-04-23
  verified_at: 2026-04-23
  decay_trigger: future conv-kernel redesign
- claim: "The existing targeted specs did not fail immediately, but an explicit forward check showed the regression clearly: baseline `top=198 logit=11.423705`, conv4 `top=318 logit=16.414322` for the same token-0 forward pass."
  source: local `tmp_qwen35_conv4_check.cr` A/B run with `QWEN35_REC_CONV4=0/1` on 2026-04-23
  verified_at: 2026-04-23
  decay_trigger: prompt/model/output-path changes
- claim: "After reverting the conv4 branch and strengthening `spec/qwen35_forward_spec.cr` to assert `top=198` and `logit≈11.423705`, targeted specs were green again and a fresh `20/20` rerun gave `24.40 ms/tok`."
  source: `CRYSTAL_CACHE_DIR=/tmp/cogni_ml_crystal_cache_conv4_revert crystal spec ...` and `CRYSTAL_CACHE_DIR=/tmp/cogni_ml_crystal_cache_conv4_revert_sync crystal run bin/qwen35_sync_profile.cr -- ... -- 20 20` on 2026-04-23
  verified_at: 2026-04-23
  decay_trigger: when the golden forward expectation changes intentionally
**note:** This branch was reverted. The useful outcome was not a speed win but a stronger guardrail: `qwen35_forward_spec` now catches semantic regressions that previously slipped through.

### [LM-codex-REC-CONV4-CORRECT-REFUTE-1] A corrected `recurrent_conv4` rewrite preserved logits but still did not speed up decode
**status:** refuted
**trust:** {F:0.92, G:0.78, R:0.91}
**context:** ml (Qwen port, recurrent arithmetic kernels)
**evidence:**
- claim: "A second bounded `qwen35_recurrent_conv4` attempt fixed the weight-layout bug by gathering channel-major `conv1d` weights with stride `conv_k`, and explicit forward A/B matched again: `top=198 logit=11.423705` in both modes."
  source: local `tmp_qwen35_conv4_check.cr` A/B run with corrected conv4 on 2026-04-23
  verified_at: 2026-04-23
  decay_trigger: future conv-kernel rewrite
- claim: "With `QWEN35_REC_CONV4=1`, the strengthened targeted specs stayed green (`7 examples, 0 failures`)."
  source: `CRYSTAL_CACHE_DIR=/tmp/cogni_ml_crystal_cache_conv4_fixed_spec crystal spec ...` on 2026-04-23
  verified_at: 2026-04-23
  decay_trigger: forward golden changes intentionally
- claim: "Strict native-only decode A/B on `/tmp/qwen35_sync_profile_conv4_fixed` refuted the optimization: `5/5` mean `22.81 -> 22.98 ms/tok`, `20/20` mean `24.18 -> 24.76 ms/tok`, `64/64` mean `26.09 -> 26.12 ms/tok`."
  source: repeated `/tmp/qwen35_sync_profile_conv4_fixed` runs on 2026-04-23
  verified_at: 2026-04-23
  decay_trigger: broader recurrent kernel changes
- claim: "After reverting the corrected conv4 branch, targeted specs were green and a fresh `20/20` rerun returned to `24.47 ms/tok`."
  source: `CRYSTAL_CACHE_DIR=/tmp/cogni_ml_crystal_cache_conv4_fixed_revert crystal spec ...` and `CRYSTAL_CACHE_DIR=/tmp/cogni_ml_crystal_cache_conv4_fixed_revert_sync crystal run bin/qwen35_sync_profile.cr -- ... -- 20 20` on 2026-04-23
  verified_at: 2026-04-23
  decay_trigger: thermal rerun or larger decode-path changes
**note:** This second branch was also reverted. The idea is now falsified in both forms: the naive vectorized rewrite was wrong, and the corrected one was correct but not faster.

### [LM-codex-REC-STAGE-BENCH-1] Fused recurrent stage timing is much lower than the sum of old standalone projection microbenches
**status:** verified
**trust:** {F:0.88, G:0.75, R:0.89}
**context:** ml (Qwen port, recurrent hotpath prioritization)
**evidence:**
- claim: "On the current fused GPU helpers with real Qwen 3.5 9B weights, `recurrent_attn_project` measured about `0.8685 ms` avg / `0.8678 ms` p50, and full `recurrent_layer_project` about `1.4164 ms` avg / `1.6725 ms` p50."
  source: local `tmp_recurrent_stage_bench.cr` on 2026-04-23
  verified_at: 2026-04-23
  decay_trigger: recurrent helper rewrites or wider decode-path changes
- claim: "These numbers are much lower than the older sum of standalone single-op projection microbenches, which means those isolated per-op timings are not a faithful proxy for in-wave prioritization."
  source: comparison of `tmp_recurrent_stage_bench.cr` with prior local `tmp_recurrent_proj_bench.cr` landmark data on 2026-04-23
  verified_at: 2026-04-23
  decay_trigger: fresh per-op microbench methodology or helper-route changes
**note:** The next optimization branch should prioritize measurements that stay inside the fused recurrent route or the full decode wave, rather than relying on standalone per-op microbenches alone.

### [LM-codex-Q6-HEAD-REFUTE-1] A head-only `Q6_K` wide pipeline did not survive stricter falsification
**status:** refuted
**trust:** {F:0.91, G:0.77, R:0.90}
**context:** ml (Qwen port, output head / Q6_K decode)
**evidence:**
- claim: "A bounded output-head-only experiment added a separate `Q6_K` GEMV pipeline for the huge `lm_head` shape (`QWEN35_Q6_HEAD=1`) and preserved correctness (`7 examples, 0 failures`)."
  source: local patch to `src/ml/gguf/kernels/gemm_q56k.metal` and `src/ml/gguf/qwen35_metal.cr`, plus targeted specs on 2026-04-23
  verified_at: 2026-04-23
  decay_trigger: output-head routing changes
- claim: "Initial decode runs looked promising on shorter workloads (`20/20` mean `24.43 -> 23.84 ms/tok`), but long-context `64/64` A/B was consistently worse (`26.86 -> 27.16 ms/tok`)."
  source: repeated `/tmp/qwen35_sync_profile_q6head` runs on 2026-04-23
  verified_at: 2026-04-23
  decay_trigger: fresh rerun after decode-path changes
- claim: "A direct `rmsnorm_project` microbench for the head path failed the hypothesis: baseline `2.8742 ms` avg / `2.5436 ms` p50 versus head-wide `2.9280 ms` avg / `2.5420 ms` p50."
  source: local `tmp_q6_head_bench.cr` on 2026-04-23
  verified_at: 2026-04-23
  decay_trigger: output-head kernel rewrite
- claim: "After reverting the branch, targeted specs were green and a fresh `20/20` rerun gave `23.78 ms/tok`."
  source: `CRYSTAL_CACHE_DIR=/tmp/cogni_ml_crystal_cache_q6head_revert crystal spec ...` and `CRYSTAL_CACHE_DIR=/tmp/cogni_ml_crystal_cache_q6head_revert_sync crystal run bin/qwen35_sync_profile.cr -- ... -- 20 20` on 2026-04-23
  verified_at: 2026-04-23
  decay_trigger: thermal rerun or broader decode changes
**note:** This branch was reverted. The isolated head microbench falsified the supposed shape-specific win, so future Q6_K work should focus on a genuinely different kernel strategy rather than dispatch constants alone.

### [LM-codex-DN-AB-FUSE-REFUTE-1] Folding `recurrent_ab` into DeltaNet did not produce a robust decode win
**status:** refuted
**trust:** {F:0.90, G:0.78, R:0.91}
**context:** ml (Qwen port, recurrent DeltaNet scheduling)
**evidence:**
- claim: "A bounded branch fused `recurrent_ab` into DeltaNet by adding `delta_net_step_ab` / `delta_net_step_ab_128`, wiring raw `alpha/beta + dt_bias + ssm_a` directly into the recurrent DeltaNet callsites under `QWEN35_DN_AB_FUSE=1`."
  source: local patches to `src/ml/gguf/kernels/delta_net.metal` and `src/ml/gguf/qwen35_metal.cr` on 2026-04-23
  verified_at: 2026-04-23
  decay_trigger: future DeltaNet kernel redesign
- claim: "The branch preserved correctness under the strengthened Qwen gate: `QWEN35_DN_AB_FUSE=1 crystal spec spec/qwen35_forward_spec.cr spec/qwen35_fullattn_spec.cr spec/qwen35_deltanet_spec.cr spec/qwen35_delta_net_spec.cr ...` returned `7 examples, 0 failures`, with `top token id=198` and `logit=11.423705`."
  source: `CRYSTAL_CACHE_DIR=/tmp/cogni_ml_crystal_cache_dn_ab_spec crystal spec ...` on 2026-04-23
  verified_at: 2026-04-23
  decay_trigger: forward golden changes intentionally
- claim: "Matched native-only decode A/B on one built `/tmp/qwen35_sync_profile_dn_ab` binary did not support promotion: `5/5` mean `22.95 -> 23.90 ms/tok`, `20/20` mean `24.08 -> 23.77 ms/tok`, `64/64` mean `25.58 -> 25.56 ms/tok`."
  source: alternating local runs of `/tmp/qwen35_sync_profile_dn_ab` with `QWEN35_DN_AB_FUSE=0/1` on 2026-04-23
  verified_at: 2026-04-23
  decay_trigger: wider recurrent or decode-wave changes
- claim: "After reverting the branch, the fused kernels and toggle path were removed, targeted specs were green again, and a fresh `20/20` rerun gave `24.93 ms/tok`."
  source: `rg -n \"dn_ab|delta_net_step_ab\" ...` returning no matches, `CRYSTAL_CACHE_DIR=/tmp/cogni_ml_crystal_cache_dn_ab_revert_spec crystal spec ...`, and `CRYSTAL_CACHE_DIR=/tmp/cogni_ml_crystal_cache_dn_ab_revert_sync crystal run bin/qwen35_sync_profile.cr -- ... -- 20 20` on 2026-04-23
  verified_at: 2026-04-23
  decay_trigger: thermal rerun or broader decode changes
**note:** This branch was correctness-safe but not performance-robust. Removing one recurrent helper dispatch was not enough; future DeltaNet work should target a more structural arithmetic or residency win, not this specific fuse.

### [LM-codex-DN-QKNORM-REFUTE-1] Folding recurrent Q/K L2 normalization into DeltaNet was correctness-safe but not faster
**status:** refuted
**trust:** {F:0.91, G:0.79, R:0.91}
**context:** ml (Qwen port, recurrent DeltaNet / QK normalization)
**evidence:**
- claim: "A bounded branch added `delta_net_step_qknorm` / `delta_net_step_qknorm_128` and a narrow `QWEN35_DN_QKNORM=1` toggle, so recurrent DeltaNet could compute per-head Q/K L2 normalization internally and skip the standalone `qwen35_l2_heads` kernels."
  source: local patches to `src/ml/gguf/kernels/delta_net.metal` and `src/ml/gguf/qwen35_metal.cr` on 2026-04-23
  verified_at: 2026-04-23
  decay_trigger: future DeltaNet or recurrent normalization redesign
- claim: "The branch preserved correctness under the strengthened Qwen gate: `QWEN35_DN_QKNORM=1 crystal spec spec/qwen35_forward_spec.cr spec/qwen35_fullattn_spec.cr spec/qwen35_deltanet_spec.cr spec/qwen35_delta_net_spec.cr ...` returned `7 examples, 0 failures`, with `top token id=198` and `logit=11.423705`."
  source: `CRYSTAL_CACHE_DIR=/tmp/cogni_ml_crystal_cache_dn_qknorm_spec crystal spec ...` on 2026-04-23
  verified_at: 2026-04-23
  decay_trigger: forward golden changes intentionally
- claim: "Initial matched decode A/B on one built `/tmp/qwen35_sync_profile_dn_qknorm` binary looked only marginally positive (`5/5` mean `23.15 -> 22.69 ms/tok`, `20/20` mean `23.74 -> 23.67 ms/tok`, `64/64` mean `25.36 -> 25.27 ms/tok`)."
  source: alternating local runs of `/tmp/qwen35_sync_profile_dn_qknorm` with `QWEN35_DN_QKNORM=0/1` on 2026-04-23
  verified_at: 2026-04-23
  decay_trigger: wider recurrent or decode-wave changes
- claim: "A stricter alternating falsifier refuted the branch: `20/20` mean `23.570 -> 23.625 ms/tok`, `64/64` mean `25.422 -> 25.788 ms/tok`."
  source: second alternating local A/B run of `/tmp/qwen35_sync_profile_dn_qknorm` on 2026-04-23
  verified_at: 2026-04-23
  decay_trigger: broader recurrent or decode-path changes
- claim: "After reverting the branch, the qknorm-specific kernels and toggle path were removed, targeted specs were green again, and a fresh `20/20` rerun gave `24.36 ms/tok`."
  source: `rg -n \"qknorm|delta_net_step_qknorm\" ...` returning no matches, `CRYSTAL_CACHE_DIR=/tmp/cogni_ml_crystal_cache_dn_qknorm_revert_spec crystal spec ...`, and `CRYSTAL_CACHE_DIR=/tmp/cogni_ml_crystal_cache_dn_qknorm_revert_sync crystal run bin/qwen35_sync_profile.cr -- ... -- 20 20` on 2026-04-23
  verified_at: 2026-04-23
  decay_trigger: thermal rerun or broader decode changes
**note:** This branch was also reverted. Removing two recurrent normalization kernels inside a single decode wave was not enough; the remaining gap is not explained by these standalone L2-normalization passes.

### [LM-codex-REC-SPLIT-BENCH-2] Recurrent FFN and recurrent-attention helpers are now roughly the same size
**status:** verified
**trust:** {F:0.87, G:0.76, R:0.88}
**context:** ml (Qwen port, recurrent hotpath prioritization)
**evidence:**
- claim: "On the current production path with real Qwen 3.5 9B weights, standalone helper timings on one recurrent layer were `recurrent_attn_project avg=0.8890 ms p50=0.8797`, `ffn_project avg=0.9346 ms p50=0.9331`, and `recurrent_layer_project avg=1.6926 ms p50=1.6904`."
  source: local `tmp_qwen35_recurrent_split_bench.cr` run on 2026-04-23
  verified_at: 2026-04-23
  decay_trigger: recurrent or FFN helper rewrites
- claim: "The measured `recurrent_layer_project - recurrent_attn_project` gap was about `0.8036 ms`, which is close to the standalone `ffn_project` cost. This means recurrent FFN is no longer a minor tail; it is effectively co-dominant with recurrent attention in the helper path."
  source: same local `tmp_qwen35_recurrent_split_bench.cr` run on 2026-04-23
  verified_at: 2026-04-23
  decay_trigger: helper-route restructuring or wider decode changes
**note:** The next bounded structural branch should target FFN projection math, not only recurrent-attention glue. The most credible candidate is a fused or otherwise more structural optimization around `gate/up/down`, rather than another small scheduler tweak.

### [LM-codex-FULL-SPLIT-BENCH-1] Full-attention FFN is also co-dominant with full-attention helper cost
**status:** verified
**trust:** {F:0.87, G:0.77, R:0.88}
**context:** ml (Qwen port, shared FFN prioritization)
**evidence:**
- claim: "On the current production path with real Qwen 3.5 9B weights, standalone helper timings on one full-attention layer were `full_attn_project avg=0.7005 ms p50=0.6959`, `ffn_project avg=0.9365 ms p50=0.9340`, and `full_attn_layer_project avg=1.4991 ms p50=1.5041`."
  source: local `tmp_qwen35_full_split_bench.cr` run on 2026-04-23
  verified_at: 2026-04-23
  decay_trigger: full-attention or FFN helper rewrites
- claim: "The measured `full_attn_layer_project - full_attn_project` gap was about `0.7986 ms`, again close to the standalone `ffn_project` cost. This means FFN is not just a recurrent-side issue; it is a shared hotspot across both layer families."
  source: same local `tmp_qwen35_full_split_bench.cr` run on 2026-04-23
  verified_at: 2026-04-23
  decay_trigger: helper-route restructuring or wider decode changes
**note:** Together with `LM-codex-REC-SPLIT-BENCH-2`, this shifts the next credible optimization target to a shared FFN projection-math branch. Another recurrent-only or full-attention-only glue tweak is less likely to move end-to-end decode meaningfully.

### [LM-codex-FFN-Q4PAIR-REFUTE-1] A fused Q4_K gate/up pair projection had a local helper win but failed decode-level promotion
**status:** refuted
**trust:** {F:0.91, G:0.79, R:0.90}
**context:** ml (Qwen port, shared FFN projection math)
**evidence:**
- claim: "A bounded shared FFN branch added `simd_mv_q4k_pair_f32` and wired it behind `QWEN35_FFN_Q4PAIR=1` for Q4_K `gate/up` pairs with matching `4096->12288` shape in both recurrent and full-attention FFN paths."
  source: local patches to `src/ml/gguf/kernels/gemm_q4k.metal` and `src/ml/gguf/qwen35_metal.cr` on 2026-04-23
  verified_at: 2026-04-23
  decay_trigger: future FFN projection rewrite
- claim: "The branch preserved correctness under the strengthened Qwen gate: `QWEN35_FFN_Q4PAIR=1 crystal spec spec/qwen35_forward_spec.cr spec/qwen35_fullattn_spec.cr spec/qwen35_deltanet_spec.cr spec/qwen35_delta_net_spec.cr ...` returned `7 examples, 0 failures`, with `top token id=198` and `logit=11.423705`."
  source: `CRYSTAL_CACHE_DIR=/tmp/cogni_ml_crystal_cache_ffn_q4pair_spec crystal spec ...` on 2026-04-23
  verified_at: 2026-04-23
  decay_trigger: forward golden changes intentionally
- claim: "Initial decode A/B on one built `/tmp/qwen35_sync_profile_ffn_q4pair` binary was mildly positive across all workloads: `5/5` mean `24.413 -> 24.313 ms/tok`, `20/20` mean `25.023 -> 24.803 ms/tok`, `64/64` mean `26.720 -> 26.485 ms/tok`."
  source: initial alternating local runs of `/tmp/qwen35_sync_profile_ffn_q4pair` with `QWEN35_FFN_Q4PAIR=0/1` on 2026-04-23
  verified_at: 2026-04-23
  decay_trigger: broader decode-path changes
- claim: "A stricter alternating falsifier refuted the decode claim: `20/20` mean `24.962 -> 24.723 ms/tok`, but `64/64` mean `27.645 -> 28.008 ms/tok`."
  source: second alternating local A/B run of `/tmp/qwen35_sync_profile_ffn_q4pair` on 2026-04-23
  verified_at: 2026-04-23
  decay_trigger: broader decode or thermal conditions
- claim: "Helper-level evidence did show a local effect on the full-attention FFN helper (`full avg 0.9418 -> 0.8935 ms`), but that did not survive end-to-end decode promotion."
  source: local `tmp_qwen35_ffn_pair_helper_bench.cr` on 2026-04-23
  verified_at: 2026-04-23
  decay_trigger: FFN helper rewrites
- claim: "After reverting the branch, the q4pair-specific kernel and toggle path were removed, targeted specs were green again, and a fresh `20/20` rerun gave `24.10 ms/tok`."
  source: `rg -n \"q4pair|simd_mv_q4k_pair_f32|QWEN35_FFN_Q4PAIR\" ...` returning no matches, `CRYSTAL_CACHE_DIR=/tmp/cogni_ml_crystal_cache_ffn_q4pair_revert_spec crystal spec ...`, and `CRYSTAL_CACHE_DIR=/tmp/cogni_ml_crystal_cache_ffn_q4pair_revert_sync crystal run bin/qwen35_sync_profile.cr -- ... -- 20 20` on 2026-04-23
  verified_at: 2026-04-23
  decay_trigger: thermal rerun or wider decode changes
**note:** This branch was reverted. Sharing `gate/up` y-loads inside a pair kernel was a real local helper improvement, but not a robust decode win, especially at `64/64`. That suggests the remaining FFN cost is not solved by this level of local Q4_K pairing alone.

### [LM-codex-FFN-DOWN-FUSE-REFUTE-1] Fusing SwiGLU with Q6_K down-projection was correctness-safe but consistently slower
**status:** refuted
**trust:** {F:0.92, G:0.81, R:0.92}
**context:** ml (Qwen port, shared FFN structural branch)
**evidence:**
- claim: "A bounded shared FFN branch added `simd_mv_q6k_swiglu_f32` and wired it behind `QWEN35_FFN_DOWN_FUSE=1` to compute `ffn_down(W_q6k @ (silu(gate) * up))` without the standalone `swiglu` dispatch and `ffn_comb` buffer."
  source: local patches to `src/ml/gguf/kernels/gemm_q56k.metal` and `src/ml/gguf/qwen35_metal.cr` on 2026-04-23
  verified_at: 2026-04-23
  decay_trigger: future FFN projection rewrite
- claim: "The branch preserved correctness under the strengthened Qwen gate: `QWEN35_FFN_DOWN_FUSE=1 crystal spec spec/qwen35_forward_spec.cr spec/qwen35_fullattn_spec.cr spec/qwen35_deltanet_spec.cr spec/qwen35_delta_net_spec.cr ...` returned `7 examples, 0 failures`, with `top token id=198` and `logit=11.423702`."
  source: `CRYSTAL_CACHE_DIR=/tmp/cogni_ml_crystal_cache_ffn_downfuse_spec crystal spec ...` on 2026-04-23
  verified_at: 2026-04-23
  decay_trigger: forward golden changes intentionally
- claim: "Matched native-only decode A/B on one built `/tmp/qwen35_sync_profile_ffn_downfuse` binary refuted the branch immediately: `5/5` mean `23.117 -> 24.730 ms/tok`, `20/20` mean `23.813 -> 25.380 ms/tok`, and `64/64` mean `25.230 -> 27.060 ms/tok`."
  source: alternating local runs of `/tmp/qwen35_sync_profile_ffn_downfuse` with `QWEN35_FFN_DOWN_FUSE` unset/set on 2026-04-23
  verified_at: 2026-04-23
  decay_trigger: broader FFN or decode-path changes
- claim: "After reverting the branch, the q6-swiglu kernel and toggle path were removed, targeted specs were green again, and a fresh `20/20` rerun gave `24.12 ms/tok`."
  source: `rg -n \"QWEN35_FFN_DOWN_FUSE|simd_mv_q6k_swiglu_f32|mv6_swiglu|q6_swiglu_down|can_use_ffn_down_fuse\" ...` returning no matches, `CRYSTAL_CACHE_DIR=/tmp/cogni_ml_crystal_cache_ffn_downfuse_revert_spec crystal spec ...`, and `CRYSTAL_CACHE_DIR=/tmp/cogni_ml_crystal_cache_ffn_downfuse_revert_sync crystal run bin/qwen35_sync_profile.cr -- ... -- 20 20` on 2026-04-23
  verified_at: 2026-04-23
  decay_trigger: thermal rerun or wider decode changes
**note:** This branch was reverted. Eliminating the explicit `swiglu` dispatch and `ffn_comb` buffer looked structurally promising, but on the actual decode wave it slowed all tested workloads. The remaining FFN cost is not dominated by this local down-projection glue.

### [LM-codex-INTERLAYER-NORM-REFUTE-1] Precomputing next-layer RMSNorm at the FFN residual boundary did not produce a robust decode win
**status:** refuted
**trust:** {F:0.90, G:0.82, R:0.89}
**context:** ml (Qwen port, decode-wave inter-layer boundary optimization)
**evidence:**
- claim: "A bounded decode-wave branch added `QWEN35_INTERLAYER_NORM=1`, replacing the end-of-layer `add_vec` plus the next layer's initial `rmsnorm_vec` with one `add_rmsnorm` that wrote both the next hidden state and its next-layer normalized view."
  source: local patch to `src/ml/gguf/qwen35_metal.cr` on 2026-04-23
  verified_at: 2026-04-23
  decay_trigger: wider decode-wave restructuring
- claim: "The branch preserved correctness under the strengthened Qwen gate: `QWEN35_INTERLAYER_NORM=1 crystal spec spec/qwen35_forward_spec.cr spec/qwen35_fullattn_spec.cr spec/qwen35_deltanet_spec.cr spec/qwen35_delta_net_spec.cr ...` returned `7 examples, 0 failures`, with `top token id=198` and `logit=11.423705`."
  source: `CRYSTAL_CACHE_DIR=/tmp/cogni_ml_crystal_cache_interlayer_norm_spec crystal spec ...` on 2026-04-23
  verified_at: 2026-04-23
  decay_trigger: forward golden changes intentionally
- claim: "Initial matched native-only A/B on one built `/tmp/qwen35_sync_profile_interlayer_norm` binary was mixed and small: `5/5` mean `22.980 -> 22.890 ms/tok`, `20/20` mean `23.580 -> 23.620 ms/tok`, `64/64` mean `25.400 -> 25.225 ms/tok`."
  source: first alternating local runs of `/tmp/qwen35_sync_profile_interlayer_norm` with `QWEN35_INTERLAYER_NORM` unset/set on 2026-04-23
  verified_at: 2026-04-23
  decay_trigger: broader decode-path changes
- claim: "A stricter alternating falsifier still did not make the result robust: `20/20` mean `23.810 -> 23.655 ms/tok`, while `64/64` mean `25.843 -> 25.483 ms/tok` depended on a single slow OFF outlier; the medians do not support a clean promotion."
  source: second alternating local A/B run of `/tmp/qwen35_sync_profile_interlayer_norm` on 2026-04-23
  verified_at: 2026-04-23
  decay_trigger: wider decode or thermal conditions
- claim: "After reverting the branch, the toggle path was removed, targeted specs were green again, and a fresh `20/20` rerun gave `24.62 ms/tok`."
  source: `rg -n \"QWEN35_INTERLAYER_NORM|pre_norm_ready|next_layer_norm_buf\" ...` returning no matches, `CRYSTAL_CACHE_DIR=/tmp/cogni_ml_crystal_cache_interlayer_norm_revert_spec crystal spec ...`, and `CRYSTAL_CACHE_DIR=/tmp/cogni_ml_crystal_cache_interlayer_norm_revert_sync crystal run bin/qwen35_sync_profile.cr -- ... -- 20 20` on 2026-04-23
  verified_at: 2026-04-23
  decay_trigger: thermal rerun or wider decode changes
**note:** This branch was reverted. The idea was structurally stronger than simple encoder packing because it eliminated an actual inter-layer `add + rmsnorm` boundary, but the measured effect remained too small and too noise-sensitive to keep.

### [LM-codex-FFN-SPLIT-BENCH-1] In the current production FFN helper, `gate/up` and `down` are already comparable
**status:** verified
**trust:** {F:0.86, G:0.79, R:0.88}
**context:** ml (Qwen port, shared FFN prioritization after multiple refutations)
**evidence:**
- claim: "On a temporary local split-bench using real Qwen 3.5 9B weights and the current production Metal path, recurrent FFN measured `whole_ffn avg=1.2924 ms`, `gate_up_proj avg=0.7495 ms`, `down_proj avg=0.5590 ms`."
  source: local `tmp_qwen35_ffn_split_bench.cr` run on 2026-04-23
  verified_at: 2026-04-23
  decay_trigger: FFN helper or GEMV routing changes
- claim: "The same split on a full-attention FFN measured `whole_ffn avg=0.9620 ms`, `gate_up_proj avg=0.6557 ms`, `down_proj avg=0.5365 ms`."
  source: same local `tmp_qwen35_ffn_split_bench.cr` run on 2026-04-23
  verified_at: 2026-04-23
  decay_trigger: FFN helper or GEMV routing changes
- claim: "The measured `proj+down` totals (`1.3086 ms` recurrent, `1.1922 ms` full) sit close to the whole-helper timings, so the remaining FFN cost is not hiding in a tiny glue kernel; it is split across both projection sides."
  source: same local `tmp_qwen35_ffn_split_bench.cr` run on 2026-04-23
  verified_at: 2026-04-23
  decay_trigger: FFN helper scheduling changes
**note:** This makes the recent refutations coherent: `Q4_K gate/up pairing` and `Q6_K down fusion` each targeted only one side of a shared cost center. The next credible FFN branch probably needs a more radical shared projection rewrite or a different bottleneck family entirely, not another one-sided local tweak.

### [LM-codex-DN-FUSED-EXACT-1] Fused DeltaNet update/output removes one dense state pass and improves decode
**status:** verified
**trust:** {F:0.93, G:0.82, R:0.91}
**context:** ml (Qwen port, recurrent arithmetic algorithm rewrite)
**evidence:**
- claim: "Added `delta_net_step_128_fused`, an algebraic rewrite of the 128-thread DeltaNet kernel. It computes `sk = dot(old_state * g, K)`, then writes `new_state = old_state * g + K * delt` while accumulating `out = dot(new_state, Q) * scale`, avoiding the separate materialized decay pass and final output read pass."
  source: local patch to `src/ml/gguf/kernels/delta_net.metal` and `src/ml/gguf/qwen35_metal.cr` on 2026-04-23
  verified_at: 2026-04-23
  decay_trigger: future DeltaNet kernel redesign
- claim: "Correctness preserved under direct DeltaNet and strengthened Qwen gates: default fused path returned `7 examples, 0 failures`, with `top token id=198`, `logit=11.423705`, DeltaNet `y cos=1.0`, and state max diff about `3.17e-08`; fallback `QWEN35_DN_FUSED=0` also passed DeltaNet/recurrent specs."
  source: `CRYSTAL_CACHE_DIR=/tmp/cogni_ml_crystal_cache_dn_fused_default_spec crystal spec ...` and `QWEN35_DN_FUSED=0 CRYSTAL_CACHE_DIR=/tmp/cogni_ml_crystal_cache_dn_fused_fallback_spec crystal spec ...` on 2026-04-23
  verified_at: 2026-04-23
  decay_trigger: forward golden changes intentionally
- claim: "Initial matched A/B on one built `/tmp/qwen35_sync_profile_dn_fused` binary was positive across all workloads: `5/5` mean `24.720 -> 23.250 ms/tok`, `20/20` mean `25.233 -> 24.023 ms/tok`, and `64/64` mean `26.140 -> 25.120 ms/tok`."
  source: alternating local runs of `/tmp/qwen35_sync_profile_dn_fused` with `QWEN35_DN_FUSED` unset/set on 2026-04-23
  verified_at: 2026-04-23
  decay_trigger: thermal rerun or wider recurrent changes
- claim: "A stricter alternating falsifier confirmed the win: `20/20` median `23.735 -> 22.330 ms/tok` and `64/64` median `25.280 -> 24.130 ms/tok`."
  source: second alternating local A/B run of `/tmp/qwen35_sync_profile_dn_fused` on 2026-04-23
  verified_at: 2026-04-23
  decay_trigger: thermal rerun or wider recurrent changes
- claim: "The fused kernel is now the default path; `QWEN35_DN_FUSED=0` preserves the previous 128-thread kernel fallback."
  source: `src/ml/gguf/qwen35_metal.cr` default path on 2026-04-23
  verified_at: 2026-04-23
  decay_trigger: DeltaNet pipeline routing changes
- claim: "Fresh matched `64/64` benchmark after promotion measured decode `cogni-ml p50=42.38 tok/s` vs `llama.cpp avg=45.12 tok/s`, gap `-6.08%`. This is not a win over llama.cpp yet, but it improves the same harness shape from the prior `-8.04%` run."
  source: `CRYSTAL_CACHE_DIR=/tmp/cogni_ml_crystal_cache_vs_llama_dn_fused crystal run bin/benchmark_qwen_vs_llama.cr -- ... --reps=3 --warmup=1 --prompt=64 --gen=64` on 2026-04-23
  verified_at: 2026-04-23
  decay_trigger: fresh benchmark rerun or benchmark harness change
**note:** This is the useful kind of paradigm shift for this path: exact algebraic reassociation with the same weights and state, not an approximation. It reduces recurrent memory traffic without accepting quality loss.

### [LM-codex-DN-SINGLEPASS-REFUTE-1] Keeping per-lane DeltaNet row vectors in registers was correctness-safe but slower
**status:** refuted
**trust:** {F:0.90, G:0.78, R:0.88}
**context:** ml (Qwen port, recurrent kernel algorithm rewrite)
**evidence:**
- claim: "A bounded `delta_net_step_128_singlepass` experiment kept each lane's old row, K, and Q float4 in registers across the simd reduction, then wrote `new_state` and `out` without the second row/K reload."
  source: local temporary patch to `src/ml/gguf/kernels/delta_net.metal` and `src/ml/gguf/qwen35_metal.cr` on 2026-04-23
  verified_at: 2026-04-23
  decay_trigger: future DeltaNet kernel redesign or compiler codegen change
- claim: "Correctness was preserved under direct DeltaNet and strengthened Qwen gates: `QWEN35_DN_SINGLEPASS=1 crystal spec spec/qwen35_forward_spec.cr spec/qwen35_fullattn_spec.cr spec/qwen35_deltanet_spec.cr spec/qwen35_delta_net_spec.cr ...` returned `7 examples, 0 failures`, with `top token id=198`, `logit=11.423705`, DeltaNet `y cos=1.0`, and state max diff about `3.17e-08`."
  source: `QWEN35_DN_SINGLEPASS=1 CRYSTAL_CACHE_DIR=/tmp/cogni_ml_crystal_cache_dn_single_qwen crystal spec ...` on 2026-04-23
  verified_at: 2026-04-23
  decay_trigger: forward golden changes intentionally
- claim: "Matched native-only decode A/B on one built `/tmp/qwen35_sync_profile_dn_single` binary refuted the branch: `64/64` mean `25.400 -> 25.546 ms/tok` with wins `2/5`, and `20/20` mean `22.784 -> 26.916 ms/tok` with wins `0/5`."
  source: paired local runs of `/tmp/qwen35_sync_profile_dn_single` with `QWEN35_DN_SINGLEPASS=0/1` on 2026-04-23
  verified_at: 2026-04-23
  decay_trigger: thermal rerun or wider recurrent changes
- claim: "After reverting the branch, default strengthened Qwen specs stayed green: `7 examples, 0 failures`, with `top token id=198` and `logit=11.423705`."
  source: `CRYSTAL_CACHE_DIR=/tmp/cogni_ml_crystal_cache_dn_single_revert_spec crystal spec ...` on 2026-04-23
  verified_at: 2026-04-23
  decay_trigger: forward golden changes intentionally
**note:** This branch was removed. The exact math was sound, but the register-pressure/codegen tradeoff lost to the two-pass fused kernel on the actual decode path. Avoid retrying this shape unless Metal compiler behavior or the row/head layout changes.

### [LM-codex-Q6-NR1-1] Q6_K one-row-per-simdgroup is a small release-build decode win on M2 Max
**status:** verified
**trust:** {F:0.89, G:0.72, R:0.86}
**context:** ml (Qwen port, Q6_K decode kernels)
**evidence:**
- claim: "Changed Q6_K GEMV tiling from `MV6_NR0=2` to `MV6_NR0=1` while keeping `MV6_NSG=2`, and mirrored the Crystal dispatch constants."
  source: local patch to `src/ml/gguf/kernels/gemm_q56k.metal` and `src/ml/gguf/qwen35_metal.cr` on 2026-04-23
  verified_at: 2026-04-23
  decay_trigger: Q6_K kernel or dispatch rewrite
- claim: "Correctness stayed green across Q4/Q5/Q6 Metal specs and the strengthened Qwen gates: `13 examples, 0 failures`, with Qwen top token `198` and logit `11.423705`."
  source: `CRYSTAL_CACHE_DIR=/tmp/cogni_ml_crystal_cache_q6nr1_spec crystal spec spec/qwen35_metal_spec.cr spec/qwen35_forward_spec.cr spec/qwen35_fullattn_spec.cr spec/qwen35_deltanet_spec.cr spec/qwen35_delta_net_spec.cr ...` on 2026-04-23
  verified_at: 2026-04-23
  decay_trigger: forward golden changes intentionally
- claim: "Matched release-build A/B on separate NR2/NR1 binaries showed a small decode win: `64/64` mean `24.060 -> 23.870 ms/tok` with NR1 wins `3/5`, and `20/20` mean `22.504 -> 22.452 ms/tok` with NR1 wins `3/5`."
  source: paired local runs of `/tmp/qwen35_sync_profile_q6nr2` and `/tmp/qwen35_sync_profile_q6nr1` on 2026-04-23
  verified_at: 2026-04-23
  decay_trigger: thermal rerun or Q6 kernel rewrite
- claim: "A fresh release `64/64` benchmark after promotion measured decode `cogni-ml p50=44.36 tok/s` vs `llama.cpp avg=43.89 tok/s`, gap `+1.07%`, but llama variance was high (`stddev=4.48 tok/s`). A stricter repeat with `reps=5,warmup=2` still led (`40.33` vs `35.22 tok/s`) but was clearly affected by machine drift."
  source: `/tmp/benchmark_qwen_vs_llama_q6nr1_release --reps=3 --warmup=1 --prompt=64 --gen=64` and `--reps=5 --warmup=2 --prompt=64 --gen=64` on 2026-04-23
  verified_at: 2026-04-23
  decay_trigger: fresh benchmark rerun under stable thermal conditions or llama.cpp rebuild
**note:** Keep the claim narrow. The internal NR2-vs-NR1 A/B is reliable enough to promote the tiling change; the external llama.cpp lead is promising but not stable enough yet to call the project `>10% faster than llama.cpp`.

### [LM-codex-Q4-NR1-REFUTE-1] Q4_K one-row-per-simdgroup preserved correctness but slowed decode
**status:** refuted
**trust:** {F:0.91, G:0.77, R:0.91}
**context:** ml (Qwen port, Q4_K decode kernels)
**evidence:**
- claim: "A bounded Q4_K tiling experiment changed `MV_NR0/MV_Q4_NR0` from `2` to `1` while keeping `MV_NSG/MV_Q4_NSG=2`."
  source: local temporary patch to `src/ml/gguf/kernels/gemm_q4k.metal` and `src/ml/gguf/qwen35_metal.cr` on 2026-04-23
  verified_at: 2026-04-23
  decay_trigger: Q4_K decode kernel rewrite
- claim: "Correctness stayed green across Q4/Q5/Q6 Metal specs and the strengthened Qwen gates: `13 examples, 0 failures`, with Qwen top token `198` and logit `11.423705`."
  source: `CRYSTAL_CACHE_DIR=/tmp/cogni_ml_crystal_cache_q4nr1_spec crystal spec spec/qwen35_metal_spec.cr spec/qwen35_forward_spec.cr spec/qwen35_fullattn_spec.cr spec/qwen35_deltanet_spec.cr spec/qwen35_delta_net_spec.cr ...` on 2026-04-23
  verified_at: 2026-04-23
  decay_trigger: forward golden changes intentionally
- claim: "Matched release-build A/B on separate NR2/NR1 binaries refuted the change: `64/64` mean `23.894 -> 25.252 ms/tok`, NR1 wins `0/5`; `20/20` mean `22.282 -> 23.202 ms/tok`, NR1 wins `0/5`."
  source: paired local runs of `/tmp/qwen35_sync_profile_q4nr2` and `/tmp/qwen35_sync_profile_q4nr1` on 2026-04-23
  verified_at: 2026-04-23
  decay_trigger: thermal rerun or Q4 kernel rewrite
- claim: "The branch was reverted; Q4_K remains at `MV_NR0/MV_Q4_NR0=2`."
  source: `rg -n \"MV_NR0|MV_Q4_NR0\" src/ml/gguf/kernels/gemm_q4k.metal src/ml/gguf/qwen35_metal.cr` on 2026-04-23
  verified_at: 2026-04-23
  decay_trigger: future Q4_K tiling change
**note:** Unlike Q6_K, Q4_K benefits strongly from amortizing input loads over two output rows. Do not retry the Q4 NR1 shape unless the Q4 kernel itself changes materially.

### [LM-codex-AB-Q4PAIR-REFUTE-1] Pairing recurrent alpha/beta Q4_K projections did not improve decode
**status:** refuted
**trust:** {F:0.89, G:0.76, R:0.88}
**context:** ml (Qwen port, recurrent projection micro-fusion)
**evidence:**
- claim: "A bounded experiment added an opt-in `simd_mv_q4k_pair_f32` kernel for same-shaped recurrent `ssm_alpha`/`ssm_beta` projections (`4096->32`) to share input reads and remove one Q4 dispatch per recurrent layer."
  source: local temporary patch to `src/ml/gguf/kernels/gemm_q4k.metal` and `src/ml/gguf/qwen35_metal.cr` on 2026-04-23
  verified_at: 2026-04-23
  decay_trigger: recurrent projection scheduling rewrite
- claim: "Correctness stayed green under the strengthened Qwen gates: `QWEN35_AB_PAIR=1 crystal spec spec/qwen35_forward_spec.cr spec/qwen35_fullattn_spec.cr spec/qwen35_deltanet_spec.cr spec/qwen35_delta_net_spec.cr ...` returned `7 examples, 0 failures`, with Qwen top token `198` and logit `11.423705`."
  source: `QWEN35_AB_PAIR=1 CRYSTAL_CACHE_DIR=/tmp/cogni_ml_crystal_cache_abpair_spec crystal spec ...` on 2026-04-23
  verified_at: 2026-04-23
  decay_trigger: forward golden changes intentionally
- claim: "Matched release env-toggle A/B on one built `/tmp/qwen35_sync_profile_abpair` binary refuted promotion: `64/64` mean `27.182 -> 27.604 ms/tok` with pair wins `2/5`; `20/20` mean `25.640 -> 25.474 ms/tok` but median slightly regressed and pair wins were only `2/5`."
  source: paired local runs of `/tmp/qwen35_sync_profile_abpair` with `QWEN35_AB_PAIR` unset/set on 2026-04-23
  verified_at: 2026-04-23
  decay_trigger: thermal rerun or recurrent projection rewrite
- claim: "The branch was removed; no `AB_PAIR`, `ab_pair`, `q4_pair`, or `simd_mv_q4k_pair` code remains, and default strengthened Qwen specs are green."
  source: `rg -n \"AB_PAIR|ab_pair|q4_pair|simd_mv_q4k_pair\" src/ml/gguf`, plus `CRYSTAL_CACHE_DIR=/tmp/cogni_ml_crystal_cache_abpair_revert_spec crystal spec ...` on 2026-04-23
  verified_at: 2026-04-23
  decay_trigger: future recurrent projection rewrite
**note:** This is another micro-fusion trap: removing one small dispatch inside the wave did not overcome extra codegen/register pressure. Future alpha/beta work should be part of a larger recurrent projection rewrite, not a standalone pair kernel.

### [LM-codex-ROPE-TABLE-REFUTE-1] Per-token RoPE cos/sin table avoided transcendentals but did not improve decode
**status:** refuted
**trust:** {F:0.89, G:0.74, R:0.88}
**context:** ml (Qwen port, full-attention RoPE)
**evidence:**
- claim: "A bounded decode-wave experiment added an opt-in `qwen35_rope_partial_table` kernel that used per-token CPU-computed cos/sin tables instead of recomputing `pow/cos/sin` inside every Q/K head and full-attention layer."
  source: local temporary patch to `src/ml/gguf/kernels/fullattn_qwen35.metal` and `src/ml/gguf/qwen35_metal.cr` on 2026-04-23
  verified_at: 2026-04-23
  decay_trigger: full-attention RoPE scheduling rewrite
- claim: "Correctness stayed green under the strict token/logit gate: `QWEN35_ROPE_TABLE=1 crystal spec spec/qwen35_forward_spec.cr spec/qwen35_delta_net_spec.cr ...` returned `3 examples, 0 failures`, with Qwen top token `198` and logit `11.423705`."
  source: `QWEN35_ROPE_TABLE=1 CRYSTAL_CACHE_DIR=/tmp/cogni_ml_crystal_cache_rope_table_spec crystal spec ...` on 2026-04-23
  verified_at: 2026-04-23
  decay_trigger: forward golden changes intentionally
- claim: "Matched release env-toggle A/B on one built `/tmp/qwen35_sync_profile_rope_table` binary refuted promotion: `64/64` mean `26.624 -> 26.944 ms/tok` with table wins `1/5`; `20/20` mean `25.540 -> 26.080 ms/tok` with table wins `2/5`."
  source: paired local runs of `/tmp/qwen35_sync_profile_rope_table` with `QWEN35_ROPE_TABLE` unset/set on 2026-04-23
  verified_at: 2026-04-23
  decay_trigger: thermal rerun or full-attention rewrite
- claim: "The branch was removed; default Qwen forward/DeltaNet specs stayed green."
  source: `CRYSTAL_CACHE_DIR=/tmp/cogni_ml_crystal_cache_rope_revert_spec crystal spec spec/qwen35_forward_spec.cr spec/qwen35_delta_net_spec.cr ...` on 2026-04-23
  verified_at: 2026-04-23
  decay_trigger: future RoPE code changes
**note:** At this scale, avoiding GPU transcendentals was not enough to overcome CPU table construction/upload and extra buffer reads. A future RoPE optimization should fuse RoPE into a larger Q/K path rather than add a standalone table.

### [LM-codex-DN-POST-FUSED-1] Fusing DeltaNet output with recurrent post RMSNorm removes one recurrent dispatch
**status:** verified
**trust:** {F:0.90, G:0.78, R:0.86}
**context:** ml (Qwen port, recurrent arithmetic/kernel fusion)
**evidence:**
- claim: "Added `delta_net_step_128_fused_post`, which computes the fused DeltaNet step and then applies per-head RMSNorm plus `silu(z)` inside the same threadgroup, removing the separate `delta_net_post_norm_gate` dispatch and avoiding a global intermediate read/write."
  source: local patch to `src/ml/gguf/kernels/delta_net.metal` and `src/ml/gguf/qwen35_metal.cr` on 2026-04-23
  verified_at: 2026-04-23
  decay_trigger: DeltaNet or recurrent post-processing rewrite
- claim: "Correctness stayed green under the strengthened Qwen gates with fused post enabled: `7 examples, 0 failures`, top token `198`, logit `11.423702`; default-on and fallback `QWEN35_DN_POST_FUSED=0` both pass."
  source: `QWEN35_DN_POST_FUSED=1 CRYSTAL_CACHE_DIR=/tmp/cogni_ml_crystal_cache_dn_post_fused_spec crystal spec ...`, `CRYSTAL_CACHE_DIR=/tmp/cogni_ml_crystal_cache_dn_post_default_spec crystal spec ...`, and `QWEN35_DN_POST_FUSED=0 CRYSTAL_CACHE_DIR=/tmp/cogni_ml_crystal_cache_dn_post_fallback_spec crystal spec ...` on 2026-04-23
  verified_at: 2026-04-23
  decay_trigger: forward golden changes intentionally
- claim: "Initial paired release A/B on one built `/tmp/qwen35_sync_profile_dn_post_fused` binary was positive but modest: `64/64` mean `26.998 -> 26.560 ms/tok`, `20/20` mean `24.134 -> 23.828 ms/tok`, both with fused wins `3/5`."
  source: paired local runs of `/tmp/qwen35_sync_profile_dn_post_fused` with `QWEN35_DN_POST_FUSED` unset/set on 2026-04-23
  verified_at: 2026-04-23
  decay_trigger: thermal rerun or recurrent kernel changes
- claim: "A stricter alternating repeat kept the result positive but narrow: `64/64` median `28.620 -> 28.175 ms/tok`; `20/20` median `27.305 -> 27.300 ms/tok`."
  source: second paired local A/B run of `/tmp/qwen35_sync_profile_dn_post_fused` on 2026-04-23
  verified_at: 2026-04-23
  decay_trigger: thermal rerun or recurrent kernel changes
- claim: "The fused-post path is now default-on; `QWEN35_DN_POST_FUSED=0` preserves the previous separate post-kernel path."
  source: `src/ml/gguf/qwen35_metal.cr` default routing on 2026-04-23
  verified_at: 2026-04-23
  decay_trigger: DeltaNet pipeline routing changes
**note:** This is a small but structurally sound exact fusion. Treat external llama.cpp measurements from the same period as noisy because both native and llama throughput drifted heavily under machine load.

### [LM-codex-FFN-SWIGLU-INPLACE-REFUTE-1] Reusing `ffn_up` as SwiGLU output was not a robust wave-level win
**status:** refuted
**trust:** {F:0.88, G:0.75, R:0.88}
**context:** ml (Qwen port, FFN memory traffic)
**evidence:**
- claim: "A bounded decode-wave experiment made `qwen35_swiglu_mul` write into `ffn_up_buf` and fed that buffer directly into `ffn_down`, avoiding the separate `ffn_comb_buf` write/read for the active wave path."
  source: local temporary patch to `src/ml/gguf/qwen35_metal.cr` on 2026-04-23
  verified_at: 2026-04-23
  decay_trigger: FFN wave rewrite
- claim: "Correctness stayed green under the strict token/logit gate: `QWEN35_FFN_SWIGLU_INPLACE=1 crystal spec spec/qwen35_forward_spec.cr spec/qwen35_delta_net_spec.cr ...` returned `3 examples, 0 failures`, with top token `198` and logit `11.423702`."
  source: `QWEN35_FFN_SWIGLU_INPLACE=1 CRYSTAL_CACHE_DIR=/tmp/cogni_ml_crystal_cache_swiglu_inplace_spec crystal spec ...` on 2026-04-23
  verified_at: 2026-04-23
  decay_trigger: forward golden changes intentionally
- claim: "Matched release env-toggle A/B on one built `/tmp/qwen35_sync_profile_swiglu_inplace` binary was mixed: `64/64` mean `25.068 -> 24.924 ms/tok` with wins `4/5`, but `20/20` mean `22.944 -> 23.054 ms/tok` with wins `2/5`."
  source: paired local runs of `/tmp/qwen35_sync_profile_swiglu_inplace` with `QWEN35_FFN_SWIGLU_INPLACE` unset/set on 2026-04-23
  verified_at: 2026-04-23
  decay_trigger: thermal rerun or FFN wave changes
- claim: "The branch was removed; no `SWIGLU_INPLACE` code remains, and default forward/DeltaNet specs stayed green."
  source: `rg -n \"SWIGLU_INPLACE|swiglu_inplace\" src/ml/gguf`, plus `CRYSTAL_CACHE_DIR=/tmp/cogni_ml_crystal_cache_swiglu_inplace_revert_spec crystal spec ...` on 2026-04-23
  verified_at: 2026-04-23
  decay_trigger: future FFN wave rewrite
**note:** The old standalone helper microbench did not predict wave-level promotion. Future FFN work needs a deeper fused projection strategy or a better profiler, not another isolated buffer-alias tweak.

### [LM-codex-FFN-Q4-SWIGLU-PAIR-REFUTE-1] Fusing Q4 gate/up projection with SwiGLU was slower in the decode wave
**status:** refuted
**trust:** {F:0.89, G:0.75, R:0.88}
**context:** ml (Qwen port, FFN projection fusion)
**evidence:**
- claim: "A bounded decode-wave experiment added an opt-in `simd_mv_q4k_swiglu_pair_f32` kernel and `QWEN35_FFN_Q4_SWIGLU_PAIR=1`, computing Q4 gate and up projections plus `silu(gate) * up` into the FFN combine buffer."
  source: local temporary patch to `src/ml/gguf/kernels/gemm_q4k.metal` and `src/ml/gguf/qwen35_metal.cr` on 2026-04-23
  verified_at: 2026-04-23
  decay_trigger: future Q4 decode kernel rewrite
- claim: "Correctness stayed green under the strict token/logit gate: `QWEN35_FFN_Q4_SWIGLU_PAIR=1 crystal spec spec/qwen35_forward_spec.cr spec/qwen35_delta_net_spec.cr ...` returned `3 examples, 0 failures`, with top token `198` and logit `11.423702`."
  source: `QWEN35_FFN_Q4_SWIGLU_PAIR=1 CRYSTAL_CACHE_DIR=/tmp/cogni_ml_crystal_cache_q4_swiglu_pair_spec crystal spec ...` on 2026-04-23
  verified_at: 2026-04-23
  decay_trigger: forward golden changes intentionally
- claim: "Matched release env-toggle A/B on one built `/tmp/qwen35_sync_profile_q4_swiglu_pair` binary refuted promotion: `64/64` mean `25.570 -> 26.154 ms/tok` with fused wins `0/5`; `20/20` medians `25.330 -> 26.080 ms/tok` with fused wins `1/5`."
  source: paired local runs of `/tmp/qwen35_sync_profile_q4_swiglu_pair` with `QWEN35_FFN_Q4_SWIGLU_PAIR` unset/set on 2026-04-23
  verified_at: 2026-04-23
  decay_trigger: thermal rerun or FFN/Q4 kernel rewrite
- claim: "The branch was removed; no `QWEN35_FFN_Q4_SWIGLU_PAIR` or `simd_mv_q4k_swiglu_pair_f32` code remains, and default forward/DeltaNet specs stayed green."
  source: `rg -n "SWIGLU_PAIR|swiglu_pair|q4_swiglu_pair|simd_mv_q4k_swiglu_pair" src/ml/gguf spec bin`, plus `CRYSTAL_CACHE_DIR=/tmp/cogni_ml_crystal_cache_q4_swiglu_pair_revert_spec crystal spec spec/qwen35_forward_spec.cr spec/qwen35_delta_net_spec.cr ...` on 2026-04-23
  verified_at: 2026-04-23
  decay_trigger: future FFN/Q4 wave rewrite
**note:** The branch is an explicit `LOCAL_OPTIMIZATION`/`MICRO_OPTIMIZATION_TRAP` example: exact arithmetic and fewer logical buffers did not compensate for doubled per-row Q4 work and worse kernel pressure. Avoid one-off paired Q4 projection kernels unless a local microbench shows a large win before wave integration.

### [LM-codex-DN-DOT-REUSE-REFUTE-1] Reusing DeltaNet dot products was exact but did not improve long decode
**status:** refuted
**trust:** {F:0.90, G:0.76, R:0.88}
**context:** ml (Qwen port, DeltaNet algebra rewrite)
**evidence:**
- claim: "A bounded opt-in branch added `delta_net_step_128_fused_post_dot` behind `QWEN35_DN_POST_DOT=1`, using the identity `dot(old*g + K*delt, Q) = g*dot(old,Q) + dot(K,Q)*delt` to avoid the second state-row dot against Q."
  source: local temporary patch to `src/ml/gguf/kernels/delta_net.metal` and `src/ml/gguf/qwen35_metal.cr` on 2026-04-23
  verified_at: 2026-04-23
  decay_trigger: future DeltaNet kernel redesign
- claim: "Correctness stayed green under the strict gate: `QWEN35_DN_POST_DOT=1 crystal spec spec/qwen35_forward_spec.cr spec/qwen35_fullattn_spec.cr spec/qwen35_deltanet_spec.cr spec/qwen35_delta_net_spec.cr ...` returned `7 examples, 0 failures`, with top token `198` and logit `11.423703`."
  source: `QWEN35_DN_POST_DOT=1 CRYSTAL_CACHE_DIR=/tmp/cogni_ml_crystal_cache_dn_post_dot_spec crystal spec ...` on 2026-04-23
  verified_at: 2026-04-23
  decay_trigger: forward golden changes intentionally
- claim: "Matched release env-toggle A/B on one built `/tmp/qwen35_sync_profile_dn_post_dot` binary refuted promotion: `20/20` mean `23.16 -> 23.05 ms/tok` was within noise with wins `2/5` plus one tie, while `64/64` mean `24.44 -> 24.71 ms/tok` regressed with wins `1/5`."
  source: paired local runs of `/tmp/qwen35_sync_profile_dn_post_dot` with `QWEN35_DN_POST_DOT=0/1` on 2026-04-23
  verified_at: 2026-04-23
  decay_trigger: thermal rerun or DeltaNet kernel rewrite
- claim: "The branch was removed; no `QWEN35_DN_POST_DOT` or `delta_net_step_128_fused_post_dot` code remains, and default strengthened Qwen specs stayed green."
  source: `rg -n "DN_POST_DOT|fused_post_dot|delta_net_step_128_fused_post_dot" src/ml/gguf spec bin`, plus `CRYSTAL_CACHE_DIR=/tmp/cogni_ml_crystal_cache_dn_post_dot_revert_spec crystal spec spec/qwen35_forward_spec.cr spec/qwen35_fullattn_spec.cr spec/qwen35_deltanet_spec.cr spec/qwen35_delta_net_spec.cr ...` on 2026-04-23
  verified_at: 2026-04-23
  decay_trigger: future DeltaNet code changes
**note:** The algebra was exact, but moving work from the second output dot into the first pass plus extra `K·Q` setup did not survive wave-level timing. Future DeltaNet work needs either a larger state-residency change or shape-specific GPU profiling, not this dot-reuse formula.

### [LM-codex-Q5-NR2-REFUTE-1] Q5_K two-rows-per-simdgroup helps short decode but regresses the 64-token target
**status:** refuted
**trust:** {F:0.88, G:0.74, R:0.87}
**context:** ml (Qwen port, Q5_K decode kernels)
**evidence:**
- claim: "A bounded branch changed Q5_K GEMV tiling from `MV5_NR0=1` to `MV5_NR0=2` in both the Metal kernel and Crystal dispatch constants."
  source: local temporary patch to `src/ml/gguf/kernels/gemm_q56k.metal` and `src/ml/gguf/qwen35_metal.cr` on 2026-04-23
  verified_at: 2026-04-23
  decay_trigger: Q5_K kernel or dispatch rewrite
- claim: "Correctness stayed green under the strengthened Qwen gates: `7 examples, 0 failures`, top token `198`, logit `11.423702`."
  source: `CRYSTAL_CACHE_DIR=/tmp/cogni_ml_crystal_cache_q5nr2_spec crystal spec spec/qwen35_forward_spec.cr spec/qwen35_fullattn_spec.cr spec/qwen35_deltanet_spec.cr spec/qwen35_delta_net_spec.cr ...` on 2026-04-23
  verified_at: 2026-04-23
  decay_trigger: forward golden changes intentionally
- claim: "Paired release A/B against separate `q5nr1` and `q5nr2` binaries was mixed: `20/20` mean `24.17 -> 23.66 ms/tok` with NR2 wins `5/5`, but target `64/64` mean `24.91 -> 25.15 ms/tok` regressed with NR2 wins only `2/5`."
  source: paired local runs of `/tmp/qwen35_sync_profile_q5nr1` and `/tmp/qwen35_sync_profile_q5nr2` on 2026-04-23
  verified_at: 2026-04-23
  decay_trigger: thermal rerun or Q5 kernel rewrite
- claim: "The branch was reverted to `MV5_NR0=1`; strengthened Qwen specs stayed green."
  source: `CRYSTAL_CACHE_DIR=/tmp/cogni_ml_crystal_cache_q5nr1_final_spec crystal spec spec/qwen35_forward_spec.cr spec/qwen35_fullattn_spec.cr spec/qwen35_deltanet_spec.cr spec/qwen35_delta_net_spec.cr ...` on 2026-04-23
  verified_at: 2026-04-23
  decay_trigger: future Q5 tiling changes
**note:** Q5_K NR2 may be useful for very short interactive loops, but the active target is 64-token decode / llama-bench parity. Keep NR1 unless the benchmark target changes or the Q5 kernel is redesigned.

### [LM-codex-TOP1-READBACK-REFUTE-1] GPU top-1 logits reduction did not beat full unified-memory logits readback
**status:** refuted
**trust:** {F:0.88, G:0.72, R:0.86}
**context:** ml (Qwen port, decode readback/sampling)
**evidence:**
- claim: "A bounded opt-in branch added a `qwen35_argmax_f32` Metal kernel and a `forward_top1`/`QWEN35_PROFILE_TOP1=1` profile route that read back only top token id and logit instead of the full `248320`-float logits vector."
  source: local temporary patch to `src/ml/gguf/kernels/ffn_qwen35.metal`, `src/ml/gguf/qwen35_metal.cr`, `src/ml/gguf/qwen35_cpu.cr`, `bin/qwen35_sync_profile.cr`, and `spec/qwen35_forward_spec.cr` on 2026-04-23
  verified_at: 2026-04-23
  decay_trigger: output head or sampling rewrite
- claim: "Correctness matched the full-logits top-1 route and kept the strengthened DeltaNet gate green: `4 examples, 0 failures`, top token `198`, logit `11.423702`."
  source: `CRYSTAL_CACHE_DIR=/tmp/cogni_ml_crystal_cache_top1_spec crystal spec spec/qwen35_forward_spec.cr spec/qwen35_delta_net_spec.cr ...` on 2026-04-23
  verified_at: 2026-04-23
  decay_trigger: forward golden changes intentionally
- claim: "Matched release env-toggle A/B on one built `/tmp/qwen35_sync_profile_top1` binary refuted promotion: `20/20` was noisy and small (`24.70 -> 24.47 ms/tok`), while target `64/64` regressed (`27.07 -> at least 27.52 ms/tok`, with the last top1 run at `29.01 ms/tok`)."
  source: paired local runs of `/tmp/qwen35_sync_profile_top1` with `QWEN35_PROFILE_TOP1=0/1` on 2026-04-23
  verified_at: 2026-04-23
  decay_trigger: thermal rerun or argmax/readback implementation rewrite
- claim: "The branch was removed; no `QWEN35_PROFILE_TOP1`, `forward_top1`, or `qwen35_argmax_f32` code remains, and strengthened Qwen specs stayed green."
  source: `rg -n "QWEN35_PROFILE_TOP1|forward_top1|qwen35_argmax_f32|argmax_pipeline|wave_top_id|top1:" src/ml/gguf spec bin`, plus `CRYSTAL_CACHE_DIR=/tmp/cogni_ml_crystal_cache_top1_revert_spec crystal spec spec/qwen35_forward_spec.cr spec/qwen35_fullattn_spec.cr spec/qwen35_deltanet_spec.cr spec/qwen35_delta_net_spec.cr ...` on 2026-04-23
  verified_at: 2026-04-23
  decay_trigger: future sampling or output-head route changes
**note:** On this unified-memory path, full logits readback is not the dominant remaining bottleneck. A future sampling optimization should be fused into the output head or avoid materializing full logits altogether; adding a standalone argmax pass after full logits is a local-optimization trap.

### [LM-codex-Q6-NSG4-REFUTE-1] Q6_K four-simdgroup threadgroups were correctness-safe but not a stable target-workload win
**status:** refuted
**trust:** {F:0.87, G:0.72, R:0.84}
**context:** ml (Qwen port, Q6_K decode kernels)
**evidence:**
- claim: "A bounded compile-time branch changed Q6_K GEMV from `MV6_NSG=2` to `MV6_NSG=4` at `MV6_NR0=1`, with matching Crystal dispatch constants and a temporary per-quant threadgroup-size helper."
  source: local temporary patch to `src/ml/gguf/kernels/gemm_q56k.metal` and `src/ml/gguf/qwen35_metal.cr` on 2026-04-23
  verified_at: 2026-04-23
  decay_trigger: Q6_K kernel or dispatch rewrite
- claim: "The first attempt failed the strict top-token gate (`198 -> 220`) because the dispatch still launched 64 threads while the kernel expected 4 simdgroups. After matching threadgroup size to 128, correctness returned green: `7 examples, 0 failures`, top token `198`, logit `11.423702`."
  source: failed `CRYSTAL_CACHE_DIR=/tmp/cogni_ml_crystal_cache_q6nsg4_spec crystal spec ...`, then passing `CRYSTAL_CACHE_DIR=/tmp/cogni_ml_crystal_cache_q6nsg4_spec2 crystal spec ...` on 2026-04-23
  verified_at: 2026-04-23
  decay_trigger: Q6 dispatch-shape changes
- claim: "Initial paired release A/B was noisy and only weakly positive on `64/64` (`33.81 -> 33.23 ms/tok`, wins `4/5`) while `20/20` was effectively tied (`29.77 -> 29.79 ms/tok`)."
  source: paired local runs of `/tmp/qwen35_sync_profile_q6nsg2` and `/tmp/qwen35_sync_profile_q6nsg4` on 2026-04-23
  verified_at: 2026-04-23
  decay_trigger: thermal rerun or Q6 kernel rewrite
- claim: "A second `64/64` repeat refuted promotion: excluding one obvious scheduler outlier (`3184.90 ms/tok`), NSG4 had no stable advantage and often regressed."
  source: additional paired local runs of `/tmp/qwen35_sync_profile_q6nsg2` and `/tmp/qwen35_sync_profile_q6nsg4` on 2026-04-23
  verified_at: 2026-04-23
  decay_trigger: controlled thermal rerun or Q6 kernel rewrite
- claim: "The branch was reverted to `MV6_NSG=2`; no `MV6_NSG=4`, `MV_Q6_NSG=4`, or temporary `gemv_threads_per_tg_for` code remains, and targeted specs stayed green."
  source: `rg -n "MV6_NSG = 4|MV_Q6_NSG = 4|gemv_threads_per_tg_for" src/ml/gguf`, plus `CRYSTAL_CACHE_DIR=/tmp/cogni_ml_crystal_cache_q6nsg4_revert_spec crystal spec spec/qwen35_forward_spec.cr spec/qwen35_delta_net_spec.cr ...` on 2026-04-23
  verified_at: 2026-04-23
  decay_trigger: future Q6 tiling changes
**note:** Q6 dispatch shape is sensitive enough that mismatched host/kernel constants can silently corrupt logits. Future tiling experiments must keep launch shape tied to kernel constants and require strict token/logit specs before timing.

### [LM-codex-Q6-HEAD-TOP1-FUSED-1] Fused Q6 lm-head top1 avoids full logits materialization for greedy decode
**status:** verified
**trust:** {F:0.90, G:0.70, R:0.86}
**context:** ml (Qwen port, output head / greedy decode)
**evidence:**
- claim: "Added an opt-in fused greedy output-head path: `simd_mv_q6k_top1_tiles_f32` computes Q6_K lm-head dot products and emits per-tile maxima without writing full vocab logits; `qwen35_top1_reduce_tiles` reduces those tile maxima to one token id/logit. The path is enabled by using `QWEN35_HEAD_TOP1_FUSED=1` with `Qwen35CPU.forward_top1` or `QWEN35_PROFILE_TOP1=1` in `bin/qwen35_sync_profile.cr`."
  source: local patch to `src/ml/gguf/kernels/gemm_q56k.metal`, `src/ml/gguf/qwen35_metal.cr`, `src/ml/gguf/qwen35_cpu.cr`, `bin/qwen35_sync_profile.cr`, and `spec/qwen35_forward_spec.cr` on 2026-04-23
  verified_at: 2026-04-23
  decay_trigger: output head, Q6_K kernel, or sampling route rewrite
- claim: "Correctness matched full logits top-1 under the strengthened Qwen gate; final rows12 spec run returned `8 examples, 0 failures`, top token `198`, logit `11.423702`, and DeltaNet max state diff about `3.17e-08`."
  source: `CRYSTAL_CACHE_DIR=/tmp/cogni_ml_crystal_cache_head_top1_rows12_final_spec crystal spec spec/qwen35_forward_spec.cr spec/qwen35_fullattn_spec.cr spec/qwen35_deltanet_spec.cr spec/qwen35_delta_net_spec.cr ...` on 2026-04-23
  verified_at: 2026-04-23
  decay_trigger: forward golden changes intentionally
- claim: "AC-power paired release A/B on one built `/tmp/qwen35_sync_profile_head_top1` binary showed a robust short-decode win: `20/20` mean `22.13 -> 21.46 ms/tok`, fused wins `5/5`."
  source: paired local runs with `QWEN35_PROFILE_TOP1=0 QWEN35_HEAD_TOP1_FUSED=0` vs `QWEN35_PROFILE_TOP1=1 QWEN35_HEAD_TOP1_FUSED=1` on AC power on 2026-04-23
  verified_at: 2026-04-23
  decay_trigger: thermal rerun or output-head changes
- claim: "AC-power `64/64` data was positive but modest: first paired run mean `24.09 -> 23.96 ms/tok`, median `23.64 -> 23.41`; repeat median improved about `25.13 -> 24.89 ms/tok` with fused wins `4/6`."
  source: paired local runs and repeat of `/tmp/qwen35_sync_profile_head_top1` on AC power on 2026-04-23
  verified_at: 2026-04-23
  decay_trigger: fixed benchmark harness rerun
- claim: "The original `bin/qwen35_sync_profile.cr` allocated `State(max_seq: 64)` regardless of workload, so `64/64` measurements wrote KV positions past the allocated cache after the warm-up token. The harness now allocates `prefill + n_runs + 2`, and corrected AC `64/64` data remains positive: mean `25.63 -> 24.88 ms/tok`, median `25.61 -> 24.99 ms/tok`, fused wins `5/5`."
  source: local fix to `bin/qwen35_sync_profile.cr`, `CRYSTAL_CACHE_DIR=/tmp/cogni_ml_crystal_cache_sync_maxseq_spec crystal spec spec/qwen35_forward_spec.cr spec/qwen35_delta_net_spec.cr ...`, and paired runs of `/tmp/qwen35_sync_profile_head_top1_fixed` on AC power on 2026-04-23
  verified_at: 2026-04-23
  decay_trigger: benchmark harness or output-head changes
- claim: "A bounded tile-size search kept `MV6_TOP_ROWS_PER_TG=12`: rows12 beat rows16 on corrected AC `64/64` in mean latency (`24.07 -> 23.61 ms/tok`, wins `4/5`), while rows32 was noisier and worse than rows16."
  source: paired local runs of `/tmp/qwen35_sync_profile_head_top1_rows12`, `/tmp/qwen35_sync_profile_head_top1_rows16`, and `/tmp/qwen35_sync_profile_head_top1_rows32` on AC power on 2026-04-23
  verified_at: 2026-04-23
  decay_trigger: output-head tile-size retune
- claim: "Corrected AC `64/64` comparison against the full lm-head path showed rows12 fused top1 as a repeatable but insufficient win: mean `23.85 -> 23.22 ms/tok`, fused wins `5/5`. On `0/128`, fused rows12 averaged about `22.77 ms/tok` (`43.9 tok/s`) versus fresh llama.cpp `tg128 = 45.85 +/- 0.96 tok/s`; do not claim llama.cpp parity yet."
  source: paired local runs of `/tmp/qwen35_sync_profile_head_top1_rows12` on AC power and `~/SrcArchives/AI/llama.cpp/build/bin/llama-bench -m ... -pg 64,64 -fa 0 -ngl 99 -r 5 -o md` on 2026-04-23
  verified_at: 2026-04-23
  decay_trigger: benchmark harness, power state, llama.cpp build/model, or output-head changes
**note:** This is an opt-in greedy-decode optimization, not a replacement for full-logits `forward` and not yet a public llama.cpp-beating claim. It is structurally different from the refuted standalone argmax branch because it avoids materializing full logits in the first place. Treat pre-fix `64/64` sync-profile numbers as stale because the KV cache was undersized.

### [LM-codex-WAVE-CHUNK-CMD-1] Chunking decode wave command buffers overlaps CPU encoding with GPU execution
**status:** verified
**trust:** {F:0.86, G:0.68, R:0.78}
**context:** ml (Qwen port, Metal decode scheduling)
**evidence:**
- claim: "The wave decode path uses fast command buffers by default and now splits the 32-layer decode wave into 2-layer command-buffer chunks by default. `QWEN35_WAVE_FAST_CMD=0` disables unretained-reference command buffers; `QWEN35_WAVE_CHUNK_LAYERS=0` disables chunking; explicit `QWEN35_WAVE_CHUNK_LAYERS=N` keeps the runtime override."
  source: local patch to `src/ml/gguf/qwen35_metal.cr` on 2026-04-23
  verified_at: 2026-04-24
  decay_trigger: Metal command-buffer wrapper or wave scheduler rewrite
- claim: "Correctness stayed green with default chunk4 scheduling: strengthened Qwen specs returned `8 examples, 0 failures`, top token `198`, logit `11.423702`, and DeltaNet max state diff about `3.17e-08`."
  source: `CRYSTAL_CACHE_DIR=/tmp/cogni_ml_crystal_cache_wave_chunk4_final_spec crystal spec spec/qwen35_forward_spec.cr spec/qwen35_fullattn_spec.cr spec/qwen35_deltanet_spec.cr spec/qwen35_delta_net_spec.cr ...` on 2026-04-23
  verified_at: 2026-04-23
  decay_trigger: forward golden or wave scheduler changes
- claim: "AC-power tile search showed chunking was a large scheduler win before later background-load drift: `0/32` quick scan improved `22.20 -> 21.12 ms/tok` at chunk4, and paired `64/64` improved mean `24.27 -> 23.22 ms/tok`, chunk4 wins `5/5`."
  source: paired local runs of `/tmp/qwen35_sync_profile_wave_chunk` with `QWEN35_WAVE_CHUNK_LAYERS=0/4`, `QWEN35_PROFILE_TOP1=1`, and `QWEN35_HEAD_TOP1_FUSED=1` on AC power on 2026-04-23
  verified_at: 2026-04-23
  decay_trigger: power state, background load, or command-buffer scheduling changes
- claim: "Later absolute timings degraded under visible host load (`WindowServer`, iTerm, Codex, Chrome, Claude), so keep the relative A/B result but do not use the late absolute numbers for public llama.cpp comparisons."
  source: `ps -Ao pid,pcpu,pmem,comm | sort -k2 -nr | head -20`, plus late reruns on 2026-04-23
  verified_at: 2026-04-23
  decay_trigger: clean-load rerun
- claim: "A later rebuilt chunk scan on the current top1 wave path showed chunk2 as a possible lead in the first low-drift pass (`22.49 ms/tok` for chunk2 vs `22.65 ms/tok` for chunk4 on `64/32`), and a paired but thermally drifting run had chunk2 win `4/6` after the first two trials. This was not promoted because `ps` showed `/tmp/cv2_system_fd_fix` at ~99% CPU and WindowServer at ~43% during the late runs."
  source: `/tmp/qwen35_sync_profile_current` scans with `QWEN35_WAVE_CHUNK_LAYERS=0/1/2/3/4/5/6/8/16/32` and paired `2/4` runs on 2026-04-23
  verified_at: 2026-04-23
  decay_trigger: quiet-load rerun or wave scheduler changes
- claim: "After stopping the contaminating `/tmp/cv2_system_fd_fix` process, a paired `chunk2/chunk4` rerun still did not justify promotion: chunk4 won `6/8` paired trials on `64/32`, although absolute timings remained UI-load sensitive."
  source: paired local runs of `/tmp/qwen35_sync_profile_current` with `QWEN35_WAVE_CHUNK_LAYERS=2/4`, `QWEN35_PROFILE_TOP1=1`, and `QWEN35_HEAD_TOP1_FUSED=1` on 2026-04-23
  verified_at: 2026-04-23
  decay_trigger: quiet-load rerun or wave scheduler changes
- claim: "Fresh same-binary release A/B after current prefill/top1 changes promoted chunk2 over chunk4 on prompt64/gen32: chunk2 won `8/10`, mean `22.385 ms/tok`, p50 `22.383`; chunk4 mean `22.518 ms/tok`, p50 `22.517`."
  source: `CRYSTAL_CACHE_DIR=/tmp/cogni_ml_crystal_cache_wave_chunk_refresh24 crystal run --release --link-flags=\"$(pwd)/build/bridge.o -framework Metal -framework Foundation -lc++\" bin/qwen35_ab_profile.cr -- --env=QWEN35_WAVE_CHUNK_LAYERS --a=2 --b=4 --prompt=64 --gen=32 --trials=10 --warmup=1`
  verified_at: 2026-04-24
  decay_trigger: wave scheduler, command buffer implementation, background load, or host power state changes
- claim: "Correctness stayed green after changing the default to chunk2: targeted Qwen Metal/forward/full-attn/DeltaNet/prompt-cache specs returned `23 examples, 0 failures`."
  source: `CRYSTAL_CACHE_DIR=/tmp/cogni_ml_crystal_cache_wave_chunk2_spec crystal spec spec/qwen35_forward_spec.cr spec/qwen35_fullattn_spec.cr spec/qwen35_deltanet_spec.cr spec/qwen35_delta_net_spec.cr spec/qwen35_prompt_cache_spec.cr --link-flags=\"$(pwd)/build/bridge.o -framework Metal -framework Foundation -lc++\"`
  verified_at: 2026-04-24
  decay_trigger: Qwen35 wave scheduler or correctness gates change
- claim: "Fresh prompt64/gen64 llama comparison after chunk2 default measured native decode `47.01 tok/s` p50 versus llama.cpp `45.36 tok/s`; pp64 prefill remained behind at native `394.46 tok/s` p50 versus llama.cpp `462.28 tok/s`."
  source: `CRYSTAL_CACHE_DIR=/tmp/cogni_ml_crystal_cache_vs_llama_wave_chunk2 crystal run --release --link-flags=\"$(pwd)/build/bridge.o -framework Metal -framework Foundation -lc++\" bin/benchmark_qwen_vs_llama.cr -- --prompt=64 --gen=64 --reps=3 --warmup=1`
  verified_at: 2026-04-24
  decay_trigger: llama.cpp rebuild, host load, power state, or benchmark harness changes
**note:** This branch is a scheduling/frame-shift win rather than a math-kernel win. It reduces idle GPU time by committing earlier chunks while the CPU encodes later chunks. The current clean same-binary A/B now justifies chunk2 as the default, but public claims should still use fresh matched llama.cpp reruns.

### [LM-codex-Q4-NSG4-REFUTE-1] Q4_K four-simdgroup threadgroups corrupt the strict Qwen top-token gate
**status:** refuted
**trust:** {F:0.88, G:0.70, R:0.86}
**context:** ml (Qwen port, Q4_K decode kernels)
**evidence:**
- claim: "Operator mix shows Q4_K dominates decode MACs (`6.30B` counted dense-equivalent MACs across decode operators), especially FFN `4096->12288` gate/up projections (`3.22B`)."
  source: temporary local weight-mix script over `Qwen35Weights` on `Qwen3.5-9B-Q4_K_M.gguf` on 2026-04-23
  verified_at: 2026-04-23
  decay_trigger: model quantization, operator list, or Qwen routing changes
- claim: "A bounded exact-semantics branch changed Q4_K GEMV from `MV_NSG/MV_Q4_NSG=2` to `4` while keeping `NR0=2`."
  source: temporary patch to `src/ml/gguf/kernels/gemm_q4k.metal` and `src/ml/gguf/qwen35_metal.cr` on 2026-04-23
  verified_at: 2026-04-23
  decay_trigger: Q4_K kernel rewrite
- claim: "The branch failed the strict correctness gate immediately: forward top token changed from `198` to `3140` under `spec/qwen35_forward_spec.cr`."
  source: `CRYSTAL_CACHE_DIR=/tmp/cogni_ml_crystal_cache_q4nsg4_spec crystal spec spec/qwen35_forward_spec.cr spec/qwen35_delta_net_spec.cr ...` on 2026-04-23
  verified_at: 2026-04-23
  decay_trigger: Q4_K launch-shape or kernel rewrite
- claim: "The branch was reverted; targeted specs returned green with top token `198`, logit `11.423702`, and DeltaNet state max diff about `3.17e-08`."
  source: `CRYSTAL_CACHE_DIR=/tmp/cogni_ml_crystal_cache_q4nsg4_revert_spec crystal spec spec/qwen35_forward_spec.cr spec/qwen35_delta_net_spec.cr ...` on 2026-04-23
  verified_at: 2026-04-23
  decay_trigger: Q4_K launch-shape changes
**note:** Q4_K remains the right bottleneck family, but simdgroup-count changes are not a free launch-geometry knob: the kernel's per-simdgroup row mapping and host dispatch must preserve exact coverage. Do not retry NSG4 without a real kernel rewrite and a token/logit spec first.

### [LM-codex-Q4-NSG1-REFUTE-1] Q4_K one-simdgroup threadgroups are correctness-safe but much slower
**status:** refuted
**trust:** {F:0.88, G:0.72, R:0.87}
**context:** ml (Qwen port, Q4_K decode kernels)
**evidence:**
- claim: "A bounded exact branch changed Q4_K GEMV from `MV_NSG/MV_Q4_NSG=2` to `1` while keeping `NR0=2`."
  source: temporary patch to `src/ml/gguf/kernels/gemm_q4k.metal` and `src/ml/gguf/qwen35_metal.cr` on 2026-04-23
  verified_at: 2026-04-23
  decay_trigger: Q4_K kernel or launch-shape rewrite
- claim: "Correctness passed under the strict gate: top token `198`, logit `11.423702`, DeltaNet state max diff about `3.17e-08`."
  source: `CRYSTAL_CACHE_DIR=/tmp/cogni_ml_crystal_cache_q4nsg1_spec crystal spec spec/qwen35_forward_spec.cr spec/qwen35_delta_net_spec.cr ...` on 2026-04-23
  verified_at: 2026-04-23
  decay_trigger: Q4_K kernel rewrite
- claim: "Release A/B on AC power refuted promotion: `64/64` mean regressed from about `23.19 ms/tok` to `28.87 ms/tok`, baseline wins `3/3`."
  source: paired local runs of `/tmp/qwen35_sync_profile_q4nsg2` and `/tmp/qwen35_sync_profile_q4nsg1` with `QWEN35_PROFILE_TOP1=1` and `QWEN35_HEAD_TOP1_FUSED=1` on 2026-04-23
  verified_at: 2026-04-23
  decay_trigger: Q4_K kernel rewrite or clean-load rerun
- claim: "The branch was reverted; targeted specs returned green with top token `198`, logit `11.423702`, and DeltaNet state max diff about `3.17e-08`."
  source: `CRYSTAL_CACHE_DIR=/tmp/cogni_ml_crystal_cache_q4nsg1_revert_spec crystal spec spec/qwen35_forward_spec.cr spec/qwen35_delta_net_spec.cr ...` on 2026-04-23
  verified_at: 2026-04-23
  decay_trigger: Q4_K launch-shape changes
**note:** The llama-compatible `NSG=2, NR0=2` shape remains the local optimum among the narrow Q4 launch-shape probes tried so far. Further Q4 gains need a real kernel rewrite, not just threadgroup geometry.

### [LM-codex-Q4-NR4-REFUTE-1] Q4_K four-rows-per-simdgroup is correctness-safe but not a decode win
**status:** refuted
**trust:** {F:0.88, G:0.72, R:0.87}
**context:** ml (Qwen port, Q4_K decode kernels)
**evidence:**
- claim: "A bounded exact branch changed Q4_K GEMV from `MV_NR0/MV_Q4_NR0=2` to `4` while keeping `NSG=2`."
  source: temporary patch to `src/ml/gguf/kernels/gemm_q4k.metal` and `src/ml/gguf/qwen35_metal.cr` on 2026-04-23
  verified_at: 2026-04-23
  decay_trigger: Q4_K kernel or launch-shape rewrite
- claim: "Correctness passed under the strict gate: top token `198`, logit `11.423702`, DeltaNet state max diff about `3.17e-08`."
  source: `CRYSTAL_CACHE_DIR=/tmp/cogni_ml_crystal_cache_q4nr4_spec crystal spec spec/qwen35_forward_spec.cr spec/qwen35_delta_net_spec.cr ...` on 2026-04-23
  verified_at: 2026-04-23
  decay_trigger: Q4_K kernel rewrite
- claim: "Release A/B on AC power refuted promotion: `64/64` mean regressed from about `23.60 ms/tok` to `23.86 ms/tok`, with NR4 wins only `2/5`."
  source: paired local runs of `/tmp/qwen35_sync_profile_q4nr2_baseline` and `/tmp/qwen35_sync_profile_q4nr4` with `QWEN35_PROFILE_TOP1=1` and `QWEN35_HEAD_TOP1_FUSED=1` on 2026-04-23
  verified_at: 2026-04-23
  decay_trigger: Q4_K kernel rewrite or clean-load rerun
- claim: "The branch was reverted; targeted specs returned green with top token `198`, logit `11.423702`, and DeltaNet state max diff about `3.17e-08`."
  source: `CRYSTAL_CACHE_DIR=/tmp/cogni_ml_crystal_cache_q4nr4_revert_spec crystal spec spec/qwen35_forward_spec.cr spec/qwen35_delta_net_spec.cr ...` on 2026-04-23
  verified_at: 2026-04-23
  decay_trigger: Q4_K launch-shape changes
**note:** Together with the earlier NR1, NSG1, and NSG4 refutations, this closes the simple Q4 launch-geometry search. Further Q4 improvements need a different kernel algorithm or prefill/batched path, not constant tuning.

### [LM-codex-WAVE-ENCODE-BUDGET-1] Current decode-wave encode overhead bounds ICB/replay ROI
**status:** verified
**trust:** {F:0.86, G:0.70, R:0.84}
**context:** ml (Qwen port, Metal command scheduling)
**evidence:**
- claim: "On the current decode-wave top1 path, `64/64` profile reports wave encode overhead around `40.85 ms` over 64 measured tokens, about `0.64 ms/tok`; wait time dominates at `1414.76 ms`."
  source: `QWEN35_PROFILE_TOP1=1 QWEN35_HEAD_TOP1_FUSED=1 /tmp/qwen35_sync_profile_q4nr2_baseline 64 64` on AC power on 2026-04-23
  verified_at: 2026-04-23
  decay_trigger: wave scheduler, command-buffer bridge, or profiling code changes
- claim: "Chunking still helps by reducing GPU wait/idle despite slightly more encode work: on a `64/32` profile, chunk0 had encode `50.48 ms`, wait `744.71 ms`, wall `25.00 ms/tok`; chunk4 had encode `53.14 ms`, wait `668.61 ms`, wall `22.72 ms/tok`."
  source: paired `QWEN35_WAVE_CHUNK_LAYERS=0/4` profile runs on `/tmp/qwen35_sync_profile_q4nr2_baseline` on AC power on 2026-04-23
  verified_at: 2026-04-23
  decay_trigger: scheduler or host-load changes
**note:** ICB/replay remains a plausible low-risk engineering task, but the measured ceiling is sub-millisecond per token on the current path. It cannot close the main gap by itself; kernel efficiency or batched/speculative verification has higher upside.

### [LM-codex-WAVE-OUTPUT-SCRATCH-SPLIT-REFUTE-1] Branching output scratch allocation does not improve greedy decode
**status:** refuted
**trust:** {F:0.86, G:0.66, R:0.84}
**context:** ml (Qwen port, decode-wave scratch management)
**evidence:**
- claim: "A bounded branch allocated either full-logits scratch or top1 scratch in `forward_decode_wave`, instead of looking up both sets unconditionally."
  source: temporary patch to `src/ml/gguf/qwen35_metal.cr` on 2026-04-23
  verified_at: 2026-04-23
  decay_trigger: wave scratch management rewrite
- claim: "Correctness stayed green under the strengthened gate: `8 examples, 0 failures`, top token `198`, logit `11.423702`, DeltaNet state max diff about `3.17e-08`."
  source: `CRYSTAL_CACHE_DIR=/tmp/cogni_ml_crystal_cache_output_scratch_split_spec crystal spec spec/qwen35_forward_spec.cr spec/qwen35_fullattn_spec.cr spec/qwen35_deltanet_spec.cr spec/qwen35_delta_net_spec.cr ...` on 2026-04-23
  verified_at: 2026-04-23
  decay_trigger: wave output path changes
- claim: "Release A/B on AC power did not justify promotion: split won only `2/5` and mean latency was essentially worse/noisy (`~24.50 -> ~24.61 ms/tok`)."
  source: paired local runs of `/tmp/qwen35_sync_profile_q4nr2_baseline` and `/tmp/qwen35_sync_profile_output_scratch_split` with `QWEN35_PROFILE_TOP1=1` and `QWEN35_HEAD_TOP1_FUSED=1` on 2026-04-23
  verified_at: 2026-04-23
  decay_trigger: clean-load rerun or scratch pool rewrite
- claim: "The branch was reverted; targeted specs returned green with top token `198`, logit `11.423702`, and DeltaNet state max diff about `3.17e-08`."
  source: `CRYSTAL_CACHE_DIR=/tmp/cogni_ml_crystal_cache_output_scratch_split_revert_spec crystal spec spec/qwen35_forward_spec.cr spec/qwen35_delta_net_spec.cr ...` on 2026-04-23
  verified_at: 2026-04-23
  decay_trigger: wave output path changes
**note:** The unconditional scratch lookups are not a measurable bottleneck; the branch/nilable overhead can erase the tiny saved work.

### [LM-codex-OP-ATTRIBUTION-1] Standalone operator attribution ranks Q4 FFN projections as the main exact kernel target
**status:** verified
**trust:** {F:0.86, G:0.72, R:0.84}
**context:** ml (Qwen port, bottleneck attribution)
**evidence:**
- claim: "Added `bin/qwen35_op_attribution.cr`, a release-build microbench that groups real Qwen3.5 9B decode matvec operators by `(quant_type, in_dim, out_dim)`, measures standalone `Qwen35Metal.matmul` p50 latency, and reports `calls_per_token * p50_ms` as a ranking signal."
  source: `bin/qwen35_op_attribution.cr` on 2026-04-23
  verified_at: 2026-04-23
  decay_trigger: operator list, Qwen routing, or Metal matmul path changes
- claim: "Current standalone attribution top shapes are Q4_K `4096->12288` FFN gate/up (`64` calls, p50 `0.400 ms`, weighted `25.592`), Q4_K `4096->4096` recurrent gate/out (`56` calls, weighted `11.107`), Q4_K `4096->32` alpha/beta (`48` calls, weighted `7.640`), and Q5_K `4096->8192` recurrent qkv (`24` calls, weighted `5.497`)."
  source: `/tmp/qwen35_op_attribution --runs=7 --warmup=3` on AC power on 2026-04-23
  verified_at: 2026-04-23
  decay_trigger: kernel rewrite, power state, or benchmark harness changes
- claim: "The local LM Studio cache has Qwen3.5 9B, Qwen3.6 27B, and Qwen3.5 35B-A3B GGUFs, but no Qwen 0.5B draft model; speculative decode would require acquiring a draft model and building a batched verifier path."
  source: `find ~/.cache/lm-studio/models -maxdepth 4 -iname '*Qwen*gguf' -o -iname '*Qwen*.gguf'` on 2026-04-23
  verified_at: 2026-04-23
  decay_trigger: local model cache changes
**note:** The reported weighted total is not token latency because each shape is measured as a standalone matmul; it intentionally overweights dispatch overhead for tiny projections. Use it to rank kernel families, not as a perf accounting model. This makes the next exact target Q4 FFN `4096->12288`, unless we pivot to speculative decode infrastructure.

### [LM-codex-BATCHED-VERIFY-MATMUL-1] Current matmul kernels scale well for batched verifier shapes
**status:** verified
**trust:** {F:0.84, G:0.66, R:0.82}
**context:** ml (Qwen port, speculative decode prerequisites)
**evidence:**
- claim: "`bin/qwen35_op_attribution.cr` now supports `--batch=N`, reporting whole-batch p50, per-row p50, and per-row weighted shape rankings."
  source: `bin/qwen35_op_attribution.cr` on 2026-04-23
  verified_at: 2026-04-23
  decay_trigger: attribution harness or Metal matmul path changes
- claim: "For the dominant Q4_K FFN `4096->12288` shape, standalone per-row p50 improved from `0.642 ms` at batch1 to `0.064 ms` at batch16; Q4_K `4096->4096` improved from `0.211 ms` to `0.025 ms`; Q5_K `4096->8192` improved from `0.240 ms` to `0.091 ms`."
  source: `/tmp/qwen35_op_attribution --runs=9 --warmup=5 --batch=1 --limit=5` and `--batch=16 --limit=5` on AC power on 2026-04-23
  verified_at: 2026-04-23
  decay_trigger: matmul kernels, power state, or benchmark harness changes
**note:** This does not mean full batched verification is implemented: attention, recurrent state branching, KV/SSM state handling, and batched wave scheduling are still missing. It does show the dominant matmul kernels are not the blocker for a speculative verifier; the blocker is sequence/state orchestration.

### [LM-codex-HEAD-TOP1-ROWS10-14-REFUTE-1] Q6 lm-head top1 rows10/rows14 are correctness-safe but not stable wins
**status:** refuted
**trust:** {F:0.82, G:0.62, R:0.76}
**context:** ml (Qwen port, output head / greedy decode)
**evidence:**
- claim: "A bounded tile-size branch tested `MV6_TOP_ROWS_PER_TG` / `HEAD_TOP1_ROWS_PER_TG` at `10` and `14` around the current rows12 default."
  source: temporary patches to `src/ml/gguf/kernels/gemm_q56k.metal` and `src/ml/gguf/qwen35_metal.cr` on 2026-04-23
  verified_at: 2026-04-23
  decay_trigger: output-head top1 kernel rewrite
- claim: "Both rows10 and rows14 preserved the strict top-token gate: top token `198`, logit `11.423702`, DeltaNet state max diff about `3.17e-08`."
  source: `CRYSTAL_CACHE_DIR=/tmp/cogni_ml_crystal_cache_head_rows10_spec crystal spec ...` and `CRYSTAL_CACHE_DIR=/tmp/cogni_ml_crystal_cache_head_rows14_spec crystal spec ...` on 2026-04-23
  verified_at: 2026-04-23
  decay_trigger: output-head top1 kernel rewrite
- claim: "AC release A/B did not produce a stable win: rows14 won the first interleaved run, rows12 won the third, and all absolute timings drifted upward during the test."
  source: interleaved local runs of `/tmp/qwen35_sync_profile_head_rows10`, `/tmp/qwen35_sync_profile_head_rows12_baseline`, and `/tmp/qwen35_sync_profile_head_rows14` on 2026-04-23
  verified_at: 2026-04-23
  decay_trigger: quiet-load rerun or output-head rewrite
- claim: "The branch was reverted to rows12; targeted specs returned green with top token `198`, logit `11.423702`, and DeltaNet state max diff about `3.17e-08`."
  source: `CRYSTAL_CACHE_DIR=/tmp/cogni_ml_crystal_cache_head_rows_retune_revert_spec crystal spec spec/qwen35_forward_spec.cr spec/qwen35_delta_net_spec.cr ...` on 2026-04-23
  verified_at: 2026-04-23
  decay_trigger: output-head top1 retune
**note:** Keep rows12 until a quiet-load retune shows a repeatable improvement. Rows10/14 are safe but not promoted.

### [LM-codex-QWEN-STATE-FORK-1] Decode state can be deep-copied for exact speculative branches
**status:** verified
**trust:** {F:0.86, G:0.72, R:0.84}
**context:** ml (Qwen port, speculative decode prerequisites)
**evidence:**
- claim: "`Qwen35CPU::State#fork`, `State#copy_from!`, `LayerState#fork`, and `LayerState#copy_from!` deep-copy CPU arrays and GPU-resident KV/conv/SSM `MetalBuffer`s using `MetalBuffer#copy_from`, preserving `max_seq` and per-layer `position`."
  source: `src/ml/gguf/qwen35_cpu.cr` on 2026-04-23
  verified_at: 2026-04-23
  decay_trigger: Qwen state layout or Metal buffer ownership changes
- claim: "A focused forward spec verifies two forked branches and one restored branch produce the same next-token top1/logit from the same parent state, that cloned Metal buffers have distinct handles and identical sizes, and that mutating the parent buffer does not alter the forked buffer."
  source: `spec/qwen35_forward_spec.cr` on 2026-04-23
  verified_at: 2026-04-23
  decay_trigger: speculative state branching or wave scheduling rewrite
- claim: "Targeted Qwen forward spec passed with the strengthened gate: `4 examples, 0 failures`; top token `198`, logit `11.423702`."
  source: `CRYSTAL_CACHE_DIR=/tmp/cogni_ml_crystal_cache_state_restore_spec crystal spec spec/qwen35_forward_spec.cr --link-flags=\"$(pwd)/build/bridge.o -framework Metal -framework Foundation -lc++\"` on 2026-04-23
  verified_at: 2026-04-23
  decay_trigger: compiler, model weights, Qwen forward route, or Metal buffer copy changes
- claim: "After one decoded token at `max_seq=32`, full `State#fork` measured p50 `4.220 ms`, while `copy_from!` restore into preallocated buffers measured p50 `1.046 ms`."
  source: temporary local release probes `tmp_qwen_fork_bench.cr` and `tmp_qwen_restore_bench.cr` on AC power on 2026-04-23
  verified_at: 2026-04-23
  decay_trigger: state size, buffer-copy implementation, or power/load changes
- claim: "A naive optimization that copied only `position / max_seq` bytes for Metal K/V buffers was not exact with the current state invariant. Focused verifier specs failed with logit mismatches in `chunked top1 verifier matches serial greedy target steps` and `keeps chunk verifier constants model-specific`, so the code was reverted."
  source: `CRYSTAL_CACHE_DIR=/tmp/cogni_ml_crystal_cache_prefixcopy_spec crystal spec spec/qwen35_forward_spec.cr spec/qwen35_state_snapshot_spec.cr --link-flags=...` on 2026-04-24
  verified_at: 2026-04-24
  decay_trigger: per-layer KV valid-length tracking, full-attention position semantics, or state copy implementation changes
**note:** This is infrastructure, not a speed win by itself. It removes a blocker for exact speculative verification; naive per-candidate fork is too expensive, but preallocated restore is cheap enough to use as a rollback primitive while building a batched verifier.

### [LM-codex-DUAL-Q4-FFN-REFUTE-1] Fusing FFN gate/up Q4 GEMV into one kernel is slower
**status:** refuted
**trust:** {F:0.82, G:0.66, R:0.78}
**context:** ml (Qwen port, Q4 FFN kernel optimization)
**evidence:**
- claim: "A bounded branch added `simd_mv_q4k_dual_f32` and routed FFN gate/up Q4 projections through one dispatch when both matrices shared shape and input."
  source: temporary patch to `src/ml/gguf/kernels/gemm_q4k.metal` and `src/ml/gguf/qwen35_metal.cr` on 2026-04-23
  verified_at: 2026-04-23
  decay_trigger: Q4 GEMV kernel rewrite
- claim: "Correctness was preserved under targeted checks: `5 examples, 0 failures`; top token `198`, logit `11.423702`; DeltaNet state max diff about `3.17e-08`."
  source: `CRYSTAL_CACHE_DIR=/tmp/cogni_ml_crystal_cache_dual_q4_spec crystal spec spec/qwen35_forward_spec.cr spec/qwen35_delta_net_spec.cr ...` on 2026-04-23
  verified_at: 2026-04-23
  decay_trigger: Q4 FFN route changes
- claim: "AC A/B on `/tmp/qwen35_sync_profile_dual_q4 64 32` showed the dual kernel slower: fallback wait `~688-692 ms` over 32 tokens, dual wait `~704-706 ms` over 32 tokens."
  source: interleaved local runs with `QWEN35_DUAL_Q4_FFN=0` vs default on 2026-04-23
  verified_at: 2026-04-23
  decay_trigger: quiet-load rerun or Q4 GEMV compiler behavior changes
- claim: "The branch was reverted; targeted specs returned green with top token `198`, logit `11.423702`, and DeltaNet state max diff about `3.17e-08`."
  source: `CRYSTAL_CACHE_DIR=/tmp/cogni_ml_crystal_cache_dual_q4_revert_spec crystal spec spec/qwen35_forward_spec.cr spec/qwen35_delta_net_spec.cr ...` on 2026-04-23
  verified_at: 2026-04-23
  decay_trigger: Q4 FFN route changes
- claim: "A 2026-04-24 re-check reproduced the falsifier with the current decode wave: focused forward spec stayed green (`13 examples, 0 failures`), but paired A/B lost `0.367 ms/tok` with dual Q4 enabled (`23.152` vs separate-GEMV `22.785 ms/tok`, wins `0/6`)."
  source: `/tmp/qwen35_ab_profile_q4dual --env QWEN35_Q4_DUAL_GEMV_OFF --a '<unset>' --b 1 --prompt 64 --gen 24 --trials 6 --warmup 1`
  verified_at: 2026-04-24
  decay_trigger: Q4 GEMV kernel rewrite, Metal compiler, decode wave scheduler, or FFN route changes
**note:** Sharing input loads did not beat the extra register pressure/occupancy cost. Do not retry this shape as a simple dual-row kernel; any future FFN work needs a different frame, such as true batched/speculative GEMM or approximate activation sparsity with evals.

### [LM-codex-AB-PROFILE-1] In-process paired A/B harness reduces scheduler-noise false positives
**status:** verified
**trust:** {F:0.84, G:0.72, R:0.82}
**context:** ml (Qwen port, performance measurement)
**evidence:**
- claim: "Added `bin/qwen35_ab_profile.cr`, an in-process paired A/B harness that loads the model once, builds one base prefill state, forks that state per trial/config, alternates A/B order, and reports per-token latency plus paired wins."
  source: `bin/qwen35_ab_profile.cr` on 2026-04-23
  verified_at: 2026-04-23
  decay_trigger: benchmark harness changes
- claim: "The harness avoids the shell-loop false positives caused by model reloads, pipeline compile, and external scheduler drift; it was validated by rerunning `QWEN35_WAVE_CHUNK_LAYERS=2/4`, where chunk4 still won `3/4` in the first short run and later `3/5`/`4/5` depending on run."
  source: `/tmp/qwen35_ab_profile --env=QWEN35_WAVE_CHUNK_LAYERS --a=2 --b=4 --prompt=64 --gen=16 --trials=4 --warmup=1` and later short runs on 2026-04-23
  verified_at: 2026-04-23
  decay_trigger: harness or state-fork semantics changes
- claim: "Existing exact toggles measured with the corrected harness: `QWEN35_DN_FUSED=1` is a real win over `0` (`~1.33 ms/tok`, wins `5/5`); `QWEN35_DN_POST_FUSED=1` is a small win (`~0.09 ms/tok`, wins `4/5`); `QWEN35_WAVE_FAST_CMD=1` is effectively noise (`~0.017 ms/tok`, wins `3/5`)."
  source: sequential `/tmp/qwen35_ab_profile` runs with `--prompt=64 --gen=12 --trials=5 --warmup=1` on 2026-04-23
  verified_at: 2026-04-23
  decay_trigger: toggle implementation or benchmark harness changes
**note:** Use this harness for noisy local optimization decisions. Avoid parallel benchmark runs; they contaminate unified-memory/GPU scheduling and create fake regressions or wins.

### [LM-codex-TOP1-FALLBACK-FIX-1] `forward_top1` now handles non-fused full-logit fallback correctly
**status:** verified
**trust:** {F:0.90, G:0.82, R:0.88}
**context:** ml (Qwen port, greedy decode correctness)
**evidence:**
- claim: "`Qwen35CPU.forward_top1` previously assumed any wave result for `top1: true` was packed `[id, logit]`; when `QWEN35_HEAD_TOP1_FUSED=0`, the wave route materialized full logits, so the helper could misinterpret logits as a packed top1 pair."
  source: inspection of `src/ml/gguf/qwen35_cpu.cr` on 2026-04-23
  verified_at: 2026-04-23
  decay_trigger: greedy decode helper rewrite
- claim: "The helper now checks `packed.size == 2`; otherwise it computes argmax over the full-logit vector returned by the wave path."
  source: `src/ml/gguf/qwen35_cpu.cr` on 2026-04-23
  verified_at: 2026-04-23
  decay_trigger: wave top1 return format changes
- claim: "Added a spec that disables fused top1 and verifies `forward_top1` matches full-logit argmax; targeted forward specs pass: `5 examples, 0 failures`, top token `198`, logit `11.423702`."
  source: `CRYSTAL_CACHE_DIR=/tmp/cogni_ml_crystal_cache_top1_fallback_spec crystal spec spec/qwen35_forward_spec.cr ...` on 2026-04-23
  verified_at: 2026-04-23
  decay_trigger: greedy decode helper or strict top-token gate changes
**note:** This was found while hardening the perf harness. It is a correctness fix, not a speed win.

### [LM-codex-WAVE-TRACE-1] Wave trace confirms recurrent layers dominate encode orchestration too
**status:** verified
**trust:** {F:0.84, G:0.68, R:0.82}
**context:** ml (Qwen port, wave profiling)
**evidence:**
- claim: "Added lightweight wave encode tracing to `Qwen35Metal::Profile`: when profiling is enabled, it now records counts and CPU encode time for named wave sections."
  source: `src/ml/gguf/qwen35_metal.cr` on 2026-04-23
  verified_at: 2026-04-23
  decay_trigger: wave scheduler or profiling module changes
- claim: "Added `bin/qwen35_wave_trace_profile.cr`, a focused profile tool for top1/full-logits wave runs."
  source: `bin/qwen35_wave_trace_profile.cr` on 2026-04-23
  verified_at: 2026-04-23
  decay_trigger: profiling script changes
- claim: "A `64/16` top1 wave trace measured wall `22.75 ms/tok`, wave wait `341.53 ms`, and encode trace `rec.layer 384 calls / 14.62 ms`, `full.layer 128 calls / 4.93 ms`, `head 16 calls / 0.18 ms`."
  source: `/tmp/qwen35_wave_trace_profile --prefill=64 --decode=16` on 2026-04-23
  verified_at: 2026-04-23
  decay_trigger: wave instrumentation, model route, or host-load changes
- claim: "Targeted correctness specs stayed green after instrumentation: `6 examples, 0 failures`; top token `198`, logit `11.423702`; DeltaNet state max diff about `3.17e-08`."
  source: `CRYSTAL_CACHE_DIR=/tmp/cogni_ml_crystal_cache_wave_trace_spec crystal spec spec/qwen35_forward_spec.cr spec/qwen35_delta_net_spec.cr ...` on 2026-04-23
  verified_at: 2026-04-23
  decay_trigger: Qwen forward or profiling changes
**note:** This trace measures CPU encode/orchestration time, not per-kernel GPU time. It still confirms the shape of the bottleneck: recurrent layers dominate both semantic work and encode volume; output-head encode is no longer meaningful.

## Future Landmarks (TBD)

- [LM-claude-SOTA-1] DeltaNet/GatedDeltaRule SoTA harvest (before Фаза 3b)

### [LM-codex-WAVE-FINE-TRACE-1] Fine-grained wave encode attribution separates orchestration from GPU work
**status:** verified
**trust:** {F:0.82, G:0.66, R:0.80}
**context:** ml (Qwen port, performance measurement)
**evidence:**
- claim: "Expanded `Qwen35Metal::Profile.trace` labels inside the whole-token wave path to split full-attn, recurrent, and head encode sections into norm/projection/attention/DeltaNet/FFN/add buckets."
  source: `src/ml/gguf/qwen35_metal.cr` on 2026-04-23
  verified_at: 2026-04-23
  decay_trigger: wave scheduler refactor
- claim: "Targeted correctness stayed green after instrumentation: `6 examples, 0 failures`; top token `198`, logit `11.423702`; DeltaNet state max diff about `3.17e-08`."
  source: `CRYSTAL_CACHE_DIR=/tmp/cogni_ml_crystal_cache_fine_trace_spec crystal spec spec/qwen35_forward_spec.cr spec/qwen35_delta_net_spec.cr ...` on 2026-04-23
  verified_at: 2026-04-23
  decay_trigger: Qwen forward or profiling changes
- claim: "A `64/16` top1 wave trace showed CPU encode sections are only milliseconds across the whole run, while GPU wait dominates; the trace is useful for orchestration hygiene but not sufficient to identify GPU kernel bottlenecks."
  source: `/tmp/qwen35_wave_trace_profile_current --prefill=64 --decode=16` on 2026-04-23
  verified_at: 2026-04-23
  decay_trigger: profiler or workload changes
**note:** Treat encode trace as a dispatch/orchestration attribution, not a kernel-time profiler. For kernel choices, keep using paired A/B or operator microbenchmarks.

### [LM-codex-CONVSHIFT-FUSION-REFUTE-1] Recurrent conv+shift exact fusion did not show a reliable speed win
**status:** refuted-for-default
**trust:** {F:0.78, G:0.55, R:0.74}
**context:** ml (Qwen port, recurrent layer fusion)
**evidence:**
- claim: "Added an exact `qwen35_recurrent_conv_shift` Metal kernel and a `QWEN35_REC_CONVSHIFT_FUSED=1` wave-path toggle; the default remains off because the measured result was neutral/noisy."
  source: `src/ml/gguf/kernels/recurrent_qwen35.metal` and `src/ml/gguf/qwen35_metal.cr` on 2026-04-23
  verified_at: 2026-04-23
  decay_trigger: recurrent conv/state layout changes
- claim: "Correctness with the fused kernel code present stayed green: `6 examples, 0 failures`; top token `198`, logit `11.423702`; DeltaNet state max diff about `3.17e-08`."
  source: `CRYSTAL_CACHE_DIR=/tmp/cogni_ml_crystal_cache_convshift_spec crystal spec spec/qwen35_forward_spec.cr spec/qwen35_delta_net_spec.cr ...` on 2026-04-23
  verified_at: 2026-04-23
  decay_trigger: recurrent kernel changes
- claim: "Paired A/B with prompt=64/gen=32/trials=8 gave fused wins `5/8` but mean delta `-0.058 ms/tok` (A=split, B=fused), so the evidence does not justify enabling it by default."
  source: `/tmp/qwen35_ab_profile_convshift --env=QWEN35_REC_CONVSHIFT_FUSED --a=0 --b=1 --prompt=64 --gen=32 --trials=8 --warmup=1` on 2026-04-23
  verified_at: 2026-04-23
  decay_trigger: quieter rerun or recurrent kernel rewrite
**note:** This is an exact optimization candidate, but current evidence says dispatch-count reduction here is not the bottleneck. Do not spend more time on simple conv+shift fusion unless a GPU kernel profiler contradicts this.

### [LM-codex-TOP1-DEFAULT-1] Greedy Qwen path uses fused top1 by default
**status:** verified
**trust:** {F:0.86, G:0.76, R:0.84}
**context:** ml (Qwen port, greedy decode)
**evidence:**
- claim: "Changed `QWEN35_HEAD_TOP1_FUSED` semantics so fused top1 is enabled by default for `forward_top1`; setting `QWEN35_HEAD_TOP1_FUSED=0` still forces the full-logit fallback."
  source: `src/ml/gguf/qwen35_metal.cr` and `src/ml/gguf/qwen35_cpu.cr` on 2026-04-23
  verified_at: 2026-04-23
  decay_trigger: greedy decode helper or lm-head path changes
- claim: "Paired A/B showed fused top1 is a real decode win for greedy path: `QWEN35_HEAD_TOP1_FUSED=1` beat `0` in `5/5` trials, mean improvement about `0.512 ms/tok` for prompt=64/gen=16."
  source: `/tmp/qwen35_ab_profile_convshift --env=QWEN35_HEAD_TOP1_FUSED --a=0 --b=1 --prompt=64 --gen=16 --trials=5 --warmup=1` on 2026-04-23
  verified_at: 2026-04-23
  decay_trigger: lm-head kernel changes
- claim: "Updated `bin/qwen35_generate.cr` to use `forward_top1` for greedy prefill/decode; smoke run generated `The capital of France is Paris` with post-compile token latencies around `0.02s` on the 5-token prompt."
  source: `/tmp/qwen35_generate_check "The capital of France is" 1` on 2026-04-23
  verified_at: 2026-04-23
  decay_trigger: generation script or tokenizer changes
- claim: "Updated `bin/benchmark_qwen_vs_llama.cr` to measure native decode as top1 by default, with `--native-full-logits` preserving the old full-logit measurement. A small `8/8` smoke benchmark completed successfully."
  source: `/tmp/benchmark_qwen_vs_llama_check --prompt=8 --gen=8 --reps=1 --warmup=0` on 2026-04-23
  verified_at: 2026-04-23
  decay_trigger: benchmark harness changes
**note:** This is exact for greedy decode only. Do not use it to claim arbitrary-temperature sampling speed; full logits are still required for general sampling.

### [LM-codex-PREFILL-GAP-1] Qwen prefill is currently sequential decode, not optimized prefill
**status:** verified
**trust:** {F:0.88, G:0.78, R:0.84}
**context:** ml (Qwen port, prefill performance)
**evidence:**
- claim: "`benchmark_qwen_vs_llama.cr` measures native prefill by looping `prompt.each_with_index { forward(single token) }`; there is no Qwen-specific layerwise/batched prefill engine in the current path."
  source: `bin/benchmark_qwen_vs_llama.cr` and `src/ml/gguf/qwen35_cpu.cr` inspection on 2026-04-23
  verified_at: 2026-04-23
  decay_trigger: Qwen prefill API or benchmark rewrite
- claim: "Fresh local `64/8` benchmark measured native prefill p50 `1445.35 ms` = `44.28 tok/s`, while llama.cpp prefill was `448.63 tok/s`; native decode top1 was roughly competitive at `45.56 tok/s` vs llama.cpp `43.63 tok/s`."
  source: `/tmp/benchmark_qwen_vs_llama_prefill_check --prompt=64 --gen=8 --reps=3 --warmup=1` on 2026-04-23
  verified_at: 2026-04-23
  decay_trigger: benchmark harness, host load, llama.cpp version, or prefill path changes
- claim: "Changing sequential prompt processing from full logits each token to top1 each token did not materially improve 64-token prompt throughput: both stayed around `43.6 tok/s`."
  source: `/tmp/qwen35_prefill_modes 64 3` on 2026-04-23
  verified_at: 2026-04-23
  decay_trigger: output head or top1 path changes
**note:** The main prefill problem is not output-head materialization; it is that native prefill has not crossed from token-by-token decode to layerwise sequence/batch processing.

### [LM-codex-MICROBATCH-LOWERBOUND-1] Batch matmul attribution supports known-token microbatch as the next breakthrough
**status:** verified-lower-bound
**trust:** {F:0.82, G:0.68, R:0.80}
**context:** ml (Qwen port, microbatch/speculative verifier)
**evidence:**
- claim: "Standalone shape attribution shows Qwen projection matmuls have large per-row speedups when processed as a batch: top weighted shapes total `88.29 ms` at batch=1, `12.92 ms` at batch=32, and `9.81 ms` at batch=64."
  source: `/tmp/qwen35_op_attr --batch=1/32/64 --warmup=2 --runs=5..7 --limit=10` on 2026-04-23
  verified_at: 2026-04-23
  decay_trigger: matmul kernel route or op attribution harness changes
- claim: "The current Qwen wrapper only routes `Q4_K` batch>8 through the Q4 GEMM path; `Q5_K` and `Q6_K` still use GEMV for batch inputs, despite `gemm_mm.metal` containing `simd_mm_q5k` and `simd_mm_q6k` kernels used elsewhere in the repo."
  source: `src/ml/gguf/qwen35_metal.cr` and `src/ml/gguf/kernels/gemm_mm.metal` inspection on 2026-04-23
  verified_at: 2026-04-23
  decay_trigger: Qwen matmul routing changes
**note:** These are standalone matmul lower bounds, not full prefill latency predictions. They justify the paradigm shift, but a correct implementation must handle recurrent state scan and avoid fake parallelism.

### [LM-codex-QWEN35-PROMPT-CACHE-1] Exact prompt-prefix restore is implemented through `.qkv` artifacts
**status:** verified
**trust:** {F:0.86, G:0.72, R:0.84}
**context:** ml (Qwen 3.5 prompt cache, state save/load)
**evidence:**
- claim: "`Qwen35StateSnapshot` captures and restores full-attention KV buffers plus DeltaNet conv/SSM state, preserving layer positions and active storage owner bytes."
  source: `src/ml/gguf/qwen35_state_snapshot.cr` and `spec/qwen35_state_snapshot_spec.cr` on 2026-04-23
  verified_at: 2026-04-23
  decay_trigger: Qwen state layout, MetalBuffer ownership, or recurrent/full-attention state changes
- claim: "The `.qkv` artifact format is fail-closed for SHA mismatch and corrupt/trailing bytes; exact restore preserves next-token top1 and full-logit cosine within the current tolerance on the 9B prompt fixture."
  source: `CRYSTAL_CACHE_DIR=/tmp/cogni_ml_crystal_cache_prompt_cache_spec2 crystal spec spec/qwen35_state_snapshot_spec.cr spec/qwen35_prompt_cache_spec.cr ...` -> `6 examples, 0 failures` on 2026-04-23
  verified_at: 2026-04-23
  decay_trigger: artifact version, snapshot codec, or restore path changes
- claim: "`Qwen35PromptCache::Store` provides local JSONL manifest lookup by exact `(model_id, tokenizer_id, prompt_hash, prefix_len)`, by session, and by longest compatible token-prefix hash; restore validates SHA before loading artifacts."
  source: `src/ml/gguf/qwen35_prompt_cache.cr` and `spec/qwen35_prompt_cache_spec.cr` on 2026-04-23
  verified_at: 2026-04-23
  decay_trigger: prompt-hash scheme, manifest schema, or PG metadata adapter changes
- claim: "Longest-prefix restore plus exact suffix replay matches live full prefill on the 9B prompt fixture."
  source: `CRYSTAL_CACHE_DIR=/tmp/cogni_ml_crystal_cache_prefix_all crystal spec spec/qwen35_forward_spec.cr spec/qwen35_delta_net_spec.cr spec/qwen35_state_snapshot_spec.cr spec/qwen35_prompt_cache_spec.cr ...` -> `14 examples, 0 failures` on 2026-04-23
  verified_at: 2026-04-23
  decay_trigger: token-hash scheme, replay helper, or Qwen forward state semantics changes
- claim: "`bin/qwen35_generate.cr` can use the prompt cache when `QWEN35_PROMPT_CACHE=1`; it stores the state before the last prompt token and, on a later run, restores the 4/5-token prefix for `The capital of France is`, replays 1 token, and generates token `11751` (`Paris`)."
  source: `/tmp/qwen35_generate_cache_check "The capital of France is" 1` run twice with `QWEN35_PROMPT_CACHE_ROOT=/tmp/cogni_ml_qwen35_generate_cache_check` on 2026-04-23
  verified_at: 2026-04-23
  decay_trigger: generator prompt loop, tokenizer, or cache metadata changes
- claim: "The pg_sorted_heap metadata adapter is dependency-free in core: it generates the `USING sorted_heap` schema, parameterized upsert SQL, and rejects unsafe SQL identifiers."
  source: `docs/sql/qwen35_prompt_cache_pg_sorted_heap.sql`, `src/ml/gguf/qwen35_prompt_cache.cr`, and `spec/qwen35_prompt_cache_spec.cr` on 2026-04-23
  verified_at: 2026-04-23
  decay_trigger: pg_sorted_heap table AM naming, schema, or PostgreSQL shard integration changes
**note:** This avoids repeated prompt prefill for exact hits and reduces repeated prefill for shared-prefix prompts. Live PostgreSQL execution, approximate KV recall, and layerwise prefill microbatching remain separate work.

### [LM-codex-QWEN35-PROMPT-CACHE-BENCH-1] Prompt-cache restore beats llama.cpp for repeated prompts
**status:** verified-benchmark-mode
**trust:** {F:0.78, G:0.56, R:0.70}
**context:** ml (Qwen 3.5 prompt-cache benchmark)
**evidence:**
- claim: "Added `--native-prefill-cache` to `bin/benchmark_qwen_vs_llama.cr`. The mode seeds the exact prompt cache once, then measures restoring the saved `.qkv` state as the native prefill replacement for repeated/session prompts."
  source: `bin/benchmark_qwen_vs_llama.cr` on 2026-04-24
  verified_at: 2026-04-24
  decay_trigger: benchmark harness, prompt-cache artifact format, or prefill state semantics changes
- claim: "The prompt-cache spec gate still passes after adding the benchmark mode."
  source: `CRYSTAL_CACHE_DIR=/tmp/cogni_ml_crystal_cache_cachebench_spec crystal spec spec/qwen35_prompt_cache_spec.cr --link-flags=...` -> `5 examples, 0 failures` on 2026-04-24
  verified_at: 2026-04-24
  decay_trigger: prompt-cache specs, state snapshot format, or restore path changes
- claim: "Matched pp64/gen64 cache benchmark under the same harness reports cached native prefill p50 `52.28 ms` / `1224.28 tok/s` vs llama.cpp `432.72 tok/s`, a `+182.93%` gap. Decode stayed positive at `+4.6%`."
  source: `/tmp/benchmark_qwen_vs_llama_cache --prompt=64 --gen=64 --warmup=1 --reps=3 --native-prefill-cache --load-warning-threshold=150 --load-total-warning-threshold=500` on 2026-04-24
  verified_at: 2026-04-24
  decay_trigger: host load, disk/cache location, prompt-cache snapshot size, or llama.cpp benchmark changes
**note:** This is the useful paradigm shift for repeated prompts and session resumes: eliminate first-run prefill by restoring exact state. It must not be compared against llama.cpp as a first-run prefill kernel win; first-run pp64 still needs lower Q4/FFN wall time.

### [LM-codex-Q56K-BATCH-GEMM-1] Q5/Q6 simdgroup batch GEMM for prefill chunks
**status:** verified-default
**trust:** {F:0.9, G:0.62, R:0.9}
**context:** ml (Qwen matmul routing, prefill/microbatch)
**evidence:**
- claim: "Historical baseline: Qwen access to existing `simd_mm_q5k`/`simd_mm_q6k` kernels was initially opt-in via `QWEN35_Q56K_BATCH_GEMM=1`; default routing stayed GEMV until chunk-level profiling showed the route is beneficial inside prefill command buffers."
  source: `src/ml/gguf/qwen35_metal.cr` on 2026-04-23
  verified_at: 2026-04-23
  decay_trigger: Q5/Q6 matrix-matrix kernel rewrite or benchmark retune
- claim: "Correctness of opt-in Q5/Q6 batch GEMM against CPU reference passed: `spec/qwen35_metal_spec.cr` -> `8 examples, 0 failures`; Q5 batch=16 cosine `1.0`, max delta `0.00337`; Q6 batch=16 cosine `1.0`, max delta `0.00489`."
  source: `CRYSTAL_CACHE_DIR=/tmp/cogni_ml_crystal_cache_q56_gemm_spec2 crystal spec spec/qwen35_metal_spec.cr ...` on 2026-04-23
  verified_at: 2026-04-23
  decay_trigger: quant kernel or tolerance changes
- claim: "Default Qwen correctness gate stayed green with Q5/Q6 batch GEMM disabled by default: `13 examples, 0 failures`; top token `198`, logit `11.423702`."
  source: `CRYSTAL_CACHE_DIR=/tmp/cogni_ml_crystal_cache_final_qwen_gate crystal spec spec/qwen35_forward_spec.cr spec/qwen35_delta_net_spec.cr spec/qwen35_state_snapshot_spec.cr spec/qwen35_prompt_cache_spec.cr ...` on 2026-04-23
  verified_at: 2026-04-23
  decay_trigger: Qwen default matmul routing changes
- claim: "Q5/Q6 batch GEMM is now default-on for Qwen chunked prefill with GPU-side F32->F16 input conversion and F16->F32 output conversion; disable with `QWEN35_Q56K_BATCH_GEMM_OFF=1`."
  source: `src/ml/gguf/qwen35_metal.cr`, `src/ml/gguf/kernels/ffn_qwen35.metal` on 2026-04-23
  verified_at: 2026-04-23
  decay_trigger: Q5/Q6 matrix-matrix kernel rewrite, conversion kernel rewrite, or precision policy change
- claim: "Default correctness gates pass with Q5/Q6 batch GEMM enabled: `26 examples, 0 failures`; A/B full-logit harness against old default on 16 tokens gave same top1 `30`, cosine `0.9999999901126796`, max_abs `0.003421545`."
  source: `crystal spec spec/qwen35_metal_spec.cr spec/qwen35_forward_spec.cr spec/qwen35_delta_net_spec.cr spec/qwen35_state_snapshot_spec.cr spec/qwen35_prompt_cache_spec.cr ...` and temporary A/B harness on 2026-04-23
  verified_at: 2026-04-23
  decay_trigger: Qwen default matmul routing changes
- claim: "pp64 benchmark improved from `143.18 tok/s` p50 to `278.50 tok/s` p50; decode remains ahead of llama.cpp by `6.96%` in the same prompt=64/gen=16 harness."
  source: `bin/benchmark_qwen_vs_llama.cr -- --prompt=64 --gen=16 --reps=3 --warmup=1` on 2026-04-23
  verified_at: 2026-04-23
  decay_trigger: benchmark harness, power state, model file, or llama.cpp HEAD changes
**note:** The older opt-in evidence is retained as history; the current default route is guarded by `QWEN35_Q56K_BATCH_GEMM_OFF=1`.

### [LM-codex-QWEN35-BENCH-20260423-1] First-run prefill is still the major gap; decode currently beats llama.cpp
**status:** verified-benchmark
**trust:** {F:0.82, G:0.62, R:0.78}
**context:** ml (Qwen benchmark vs llama.cpp)
**evidence:**
- claim: "On prompt=64/gen=64/reps=3/warmup=1, current native first-run prefill measured `41.41 tok/s` p50 while llama.cpp measured `437.92 tok/s`; gap `-90.54%`."
  source: `CRYSTAL_CACHE_DIR=/tmp/cogni_ml_crystal_cache_bench_current_6464 crystal run --link-flags="build/bridge.o -framework Metal -framework Foundation -lc++" bin/benchmark_qwen_vs_llama.cr -- --prompt=64 --gen=64 --reps=3 --warmup=1` on 2026-04-23
  verified_at: 2026-04-23
  decay_trigger: benchmark harness, host load, llama.cpp version, or Qwen prefill path changes
- claim: "On the same prompt=64/gen=64 run, native greedy decode measured `41.71 tok/s` p50 while llama.cpp measured `35.23 tok/s`; gap `+18.4%`."
  source: same benchmark command on 2026-04-23
  verified_at: 2026-04-23
  decay_trigger: benchmark harness, host load, llama.cpp version, or decode path changes
- claim: "Shorter prompt=64/gen=16/reps=3/warmup=1 showed the same shape: native prefill `41.58 tok/s` p50 vs llama.cpp `434.41 tok/s`; native decode `42.91 tok/s` p50 vs llama.cpp `40.09 tok/s`."
  source: `CRYSTAL_CACHE_DIR=/tmp/cogni_ml_crystal_cache_bench_current crystal run --link-flags="build/bridge.o -framework Metal -framework Foundation -lc++" bin/benchmark_qwen_vs_llama.cr -- --prompt=64 --gen=16 --reps=3 --warmup=1` on 2026-04-23
  verified_at: 2026-04-23
  decay_trigger: benchmark harness, host load, llama.cpp version, or decode/prefill path changes
**note:** Prompt-cache optimizes repeated/shared-prefix prompts, not the first-run prefill measured here. First-run prefill still needs a real layerwise/microbatch implementation.

### [LM-codex-PREFILL-NOHEAD-1] First-run prefill skips lm-head on non-final prompt tokens
**status:** verified
**trust:** {F:0.84, G:0.66, R:0.82}
**context:** ml (Qwen port, exact prefill shortcut)
**evidence:**
- claim: "Added `Qwen35CPU.prefill_token` for prompt tokens whose logits are not needed; it runs the full layer stack and updates full-attention KV plus DeltaNet conv/SSM state, but skips output RMSNorm/lm-head."
  source: `src/ml/gguf/qwen35_cpu.cr` and `src/ml/gguf/qwen35_metal.cr` on 2026-04-23
  verified_at: 2026-04-23
  decay_trigger: Qwen forward state semantics, decode wave routing, or output head path changes
- claim: "Generator prefill and prompt-cache suffix replay use `prefill_token` for non-final prompt tokens and keep `forward_top1` for the final token, preserving greedy next-token behavior."
  source: `bin/qwen35_generate.cr`, `src/ml/gguf/qwen35_prompt_cache.cr`, and `spec/qwen35_forward_spec.cr` on 2026-04-23
  verified_at: 2026-04-23
  decay_trigger: prompt loop, prompt-cache replay, tokenizer, or state snapshot changes
- claim: "Correctness gate stayed green after the no-head prefill path: `15 examples, 0 failures` across Qwen forward, DeltaNet, state snapshot, and prompt-cache specs."
  source: `CRYSTAL_CACHE_DIR=/tmp/cogni_ml_crystal_cache_prefill_final_specs crystal spec spec/qwen35_forward_spec.cr spec/qwen35_delta_net_spec.cr spec/qwen35_state_snapshot_spec.cr spec/qwen35_prompt_cache_spec.cr ...` on 2026-04-23
  verified_at: 2026-04-23
  decay_trigger: spec fixture, Qwen forward path, or Metal bridge changes
- claim: "On prompt=64/gen=64/reps=3/warmup=1, exact first-run native prefill improved from previous `41.41 tok/s` p50 to `52.73 tok/s` p50; llama.cpp still measured `463.99 tok/s`, so this is a useful shortcut, not the final prefill architecture."
  source: `CRYSTAL_CACHE_DIR=/tmp/cogni_ml_crystal_cache_bench_prefill_final crystal run --link-flags="build/bridge.o -framework Metal -framework Foundation -lc++" bin/benchmark_qwen_vs_llama.cr -- --prompt=64 --gen=64 --reps=3 --warmup=1` on 2026-04-23
  verified_at: 2026-04-23
  decay_trigger: benchmark harness, host load, llama.cpp version, or Qwen prefill path changes
**note:** This is exact because intermediate prompt-token logits are unobserved in greedy prefill. The remaining prefill gap needs a layerwise/microbatch known-token engine, not more output-head trimming.

### [LM-codex-LLAMACPP-PREFILL-GDN-1] llama.cpp Qwen35 prefill uses ubatch graph plus fused chunked GDN
**status:** verified-source
**trust:** {F:0.86, G:0.72, R:0.82}
**context:** ml (Qwen prefill architecture)
**evidence:**
- claim: "llama.cpp builds Qwen35 prefill as a multi-token `ubatch` graph, not as repeated single-token decode. `llm_build_qwen35` runs projections over `n_tokens`, limits final-layer outputs through `inp_out_ids`, and routes recurrent layers through `build_delta_net`."
  source: `/Users/sergey/SrcArchives/AI/llama.cpp/src/models/qwen35.cpp` and `/Users/sergey/SrcArchives/AI/llama.cpp/src/llama-graph.h` inspection on 2026-04-23
  verified_at: 2026-04-23
  decay_trigger: llama.cpp Qwen35 graph builder rewrite
- claim: "For `n_seq_tokens > 1`, llama.cpp selects `build_delta_net_fused` when fused chunked GDN is supported; otherwise it falls back to a chunking graph. The Metal backend has `kernel_gated_delta_net_*` that loops over token time inside one dispatch per head/layer."
  source: `/Users/sergey/SrcArchives/AI/llama.cpp/src/models/delta-net-base.cpp`, `/Users/sergey/SrcArchives/AI/llama.cpp/ggml/src/ggml-metal/ggml-metal.metal`, and `/Users/sergey/SrcArchives/AI/llama.cpp/ggml/src/ggml-metal/ggml-metal-ops.cpp` inspection on 2026-04-23
  verified_at: 2026-04-23
  decay_trigger: llama.cpp Gated Delta Net op or Metal backend rewrite
**note:** Claude's serial-scan warning is directionally correct but incomplete. The recurrence remains serial over tokens, but llama.cpp removes token-by-token orchestration and batches projections around the scan.

### [LM-codex-DN-CHUNK-PRIMITIVE-1] Chunked fused DeltaNet Metal primitive is correct and faster than repeated step dispatch
**status:** verified-primitive
**trust:** {F:0.84, G:0.58, R:0.80}
**context:** ml (Qwen prefill primitive)
**evidence:**
- claim: "Added `delta_net_chunk_128_fused`, a token-major multi-token DeltaNet scan kernel with the same recurrence as the single-token fused path, plus `Qwen35Metal.delta_net_chunk` wrapper."
  source: `src/ml/gguf/kernels/delta_net.metal` and `src/ml/gguf/qwen35_metal.cr` on 2026-04-23
  verified_at: 2026-04-23
  decay_trigger: DeltaNet state layout, recurrent tensor layout, or Metal wrapper changes
- claim: "Correctness against repeated CPU `delta_net_step!` passed on Qwen35 9B shapes for an 8-token chunk: output cosine `1.0`, max output diff `1.49e-08`, state cosine `1.0`, max state diff `5.96e-08`."
  source: `CRYSTAL_CACHE_DIR=/tmp/cogni_ml_crystal_cache_dn_chunk_spec crystal spec spec/qwen35_delta_net_spec.cr ...` on 2026-04-23
  verified_at: 2026-04-23
  decay_trigger: DeltaNet CPU reference or kernel launch geometry changes
- claim: "Microbench comparing 8 repeated Metal `delta_net_step` calls to one `delta_net_chunk` call measured p50 `2.8041 ms` vs `0.9214 ms`, a `3.04x` p50 speedup for the scan primitive under the current upload/readback wrapper."
  source: temporary `/tmp` Crystal microbench run with `CRYSTAL_CACHE_DIR=/tmp/cogni_ml_crystal_cache_dn_chunk_bench` on 2026-04-23
  verified_at: 2026-04-23
  decay_trigger: Metal command-buffer path, host load, or DeltaNet wrapper changes
**note:** This is not yet an end-to-end prefill speedup because the Qwen forward path still lacks layerwise hidden-state buffers and batched full-attention/FFN plumbing. It establishes the central recurrent primitive needed for that path.

### [LM-codex-RECURRENT-PREP-CHUNK-1] Recurrent prefill prep primitives are chunked on Metal
**status:** verified-primitive
**trust:** {F:0.84, G:0.58, R:0.80}
**context:** ml (Qwen prefill primitive)
**evidence:**
- claim: "Added chunked Metal kernels for Qwen35 recurrent prep: `qwen35_recurrent_conv_shift_chunk`, `qwen35_recurrent_ab_chunk`, and `qwen35_l2_heads_chunk`, plus `Qwen35Metal.recurrent_prep_chunk` wrapper. The wrapper emits token-major Q/K/V/g/beta arrays ready for `delta_net_chunk` and updates convolution state exactly across the chunk."
  source: `src/ml/gguf/kernels/recurrent_qwen35.metal` and `src/ml/gguf/qwen35_metal.cr` on 2026-04-23
  verified_at: 2026-04-23
  decay_trigger: recurrent conv layout, alpha/beta transform, L2 norm semantics, or prefill tensor layout changes
- claim: "Correctness against repeated CPU recurrent prep passed for an 8-token Qwen35-shaped chunk: max diffs q `5.96e-08`, k `5.96e-08`, v `7.45e-09`, g `1.19e-07`, beta `1.19e-07`, conv_state `0`."
  source: `CRYSTAL_CACHE_DIR=/tmp/cogni_ml_crystal_cache_recurrent_prep_spec crystal spec spec/qwen35_delta_net_spec.cr ...` on 2026-04-23
  verified_at: 2026-04-23
  decay_trigger: recurrent prep spec, CPU reference formulas, or kernel launch geometry changes
- claim: "Full Qwen gate stayed green with the new primitives present: `17 examples, 0 failures` across forward, DeltaNet, state snapshot, and prompt-cache specs."
  source: `CRYSTAL_CACHE_DIR=/tmp/cogni_ml_crystal_cache_recurrent_prep_gate crystal spec spec/qwen35_forward_spec.cr spec/qwen35_delta_net_spec.cr spec/qwen35_state_snapshot_spec.cr spec/qwen35_prompt_cache_spec.cr ...` on 2026-04-23
  verified_at: 2026-04-23
  decay_trigger: Qwen forward path, prompt-cache replay, or Metal bridge changes
- claim: "Microbench comparing eight repeated single-token recurrent prep wrapper calls to one 8-token chunk call measured p50 `2.0605 ms` vs `0.3000 ms`, a `6.87x` p50 speedup for the prep primitive under current upload/readback wrappers."
  source: temporary `/tmp` Crystal microbench run with `CRYSTAL_CACHE_DIR=/tmp/cogni_ml_crystal_cache_recurrent_prep_bench` on 2026-04-23
  verified_at: 2026-04-23
  decay_trigger: Metal command-buffer path, host load, or recurrent prep wrapper changes
**note:** Together with [LM-codex-DN-CHUNK-PRIMITIVE-1], this covers the recurrent scan core and its per-token prep. End-to-end prefill still needs batched layer hidden-state flow, SSM output/post/FFN for all chunk tokens, and full-attention layer chunking.

### [LM-codex-PREFILL-CPU-CHUNK-FALSIFIER-1] CPU-orchestrated layerwise prefill chunking is correct but slower
**status:** verified-falsifier
**trust:** {F:0.82, G:0.62, R:0.78}
**context:** ml (Qwen prefill architecture)
**evidence:**
- claim: "Added `Qwen35CPU.prefill_tokens`, a correctness-gated experimental path that processes known non-final prompt spans layerwise: recurrent layers use batched projections plus `recurrent_prep_chunk`/`delta_net_chunk`, while full-attention layers remain serial. The path is opt-in through `QWEN35_PREFILL_CHUNK=1` because it is not a speed win."
  source: `src/ml/gguf/qwen35_cpu.cr`, `bin/qwen35_generate.cr`, `bin/benchmark_qwen_vs_llama.cr`, and `src/ml/gguf/qwen35_prompt_cache.cr` on 2026-04-23
  verified_at: 2026-04-23
  decay_trigger: prefill orchestration, whole-token decode wave, or full-attention chunk path changes
- claim: "Correctness gate passed with opt-in chunk prefill: `10 examples, 0 failures` for forward plus DeltaNet specs, and `18 examples, 0 failures` across forward, DeltaNet, state snapshot, and prompt-cache specs."
  source: `CRYSTAL_CACHE_DIR=/tmp/cogni_ml_crystal_cache_prefill_chunk_spec3 crystal spec spec/qwen35_delta_net_spec.cr spec/qwen35_forward_spec.cr ...` and `CRYSTAL_CACHE_DIR=/tmp/cogni_ml_crystal_cache_prefill_chunk_gate2 crystal spec spec/qwen35_forward_spec.cr spec/qwen35_delta_net_spec.cr spec/qwen35_state_snapshot_spec.cr spec/qwen35_prompt_cache_spec.cr ...` on 2026-04-23
  verified_at: 2026-04-23
  decay_trigger: Qwen forward state semantics, prompt-cache replay, or recurrent chunk wrapper changes
- claim: "Benchmark falsified CPU-orchestrated layerwise chunking as a default optimization on prompt=64/gen=16/reps=2/warmup=1: default whole-token wave prefill measured p50 `52.35 tok/s`; opt-in `QWEN35_PREFILL_CHUNK=1` measured p50 `23.01 tok/s`; llama.cpp measured about `461 tok/s` prefill. Decode stayed ahead of llama.cpp in this short run: p50 `47.16 tok/s` vs llama `44.48 tok/s`."
  source: `bin/benchmark_qwen_vs_llama.cr -- --prompt=64 --gen=16 --reps=2 --warmup=1` with and without `QWEN35_PREFILL_CHUNK=1` on 2026-04-23
  verified_at: 2026-04-23
  decay_trigger: benchmark harness, host load, llama.cpp version, or prefill wave implementation changes
**note:** This refuted the first CPU-layer-loop integration. It was superseded by [LM-codex-PREFILL-GPU-RECURRENT-CHUNK-1], which keeps each recurrent layer chunk GPU-resident and turns the same architectural idea into a small speed win.

### [LM-codex-PREFILL-GPU-RECURRENT-CHUNK-1] GPU-resident recurrent layer chunks improve exact prefill
**status:** verified
**trust:** {F:0.84, G:0.60, R:0.80}
**context:** ml (Qwen prefill optimization)
**evidence:**
- claim: "Added a GPU-resident recurrent layer chunk primitive: row RMSNorm, batched qkv/z/alpha/beta projections, chunked conv/L2/alpha-beta, chunked DeltaNet, chunked post norm/gate, batched ssm_out, row add+RMSNorm, batched FFN, and final add all run in one command buffer per recurrent layer chunk."
  source: `src/ml/gguf/qwen35_metal.cr`, `src/ml/gguf/kernels/ffn_qwen35.metal`, and `src/ml/gguf/kernels/delta_net.metal` on 2026-04-23
  verified_at: 2026-04-23
  decay_trigger: recurrent layer kernel layout, batch GEMV semantics, or prefill chunk routing changes
- claim: "Correctness gate passed after making recurrent chunk prefill default-on with a 64-token chunk cap: `18 examples, 0 failures` across forward, DeltaNet, state snapshot, and prompt-cache specs."
  source: `CRYSTAL_CACHE_DIR=/tmp/cogni_ml_crystal_cache_gpu_chunk_fullgate crystal spec spec/qwen35_forward_spec.cr spec/qwen35_delta_net_spec.cr spec/qwen35_state_snapshot_spec.cr spec/qwen35_prompt_cache_spec.cr ...` on 2026-04-23
  verified_at: 2026-04-23
  decay_trigger: Qwen forward state semantics, prompt-cache replay, or chunk-size routing changes
- claim: "On prompt=64/gen=16/reps=3/warmup=1, default GPU recurrent chunk prefill measured p50 `56.70 tok/s`; `QWEN35_PREFILL_CHUNK_OFF=1` measured p50 `52.87 tok/s`; llama.cpp measured about `463 tok/s` prefill. Decode remained ahead of llama.cpp in the same short run: p50 `47.65 tok/s` vs llama `45.07 tok/s`."
  source: `bin/benchmark_qwen_vs_llama.cr -- --prompt=64 --gen=16 --reps=3 --warmup=1` with and without `QWEN35_PREFILL_CHUNK_OFF=1` on 2026-04-23
  verified_at: 2026-04-23
  decay_trigger: benchmark harness, host load, llama.cpp version, or prefill wave implementation changes
**note:** This is still only a partial prefill win because full-attention layers are serial inside each chunk and recurrent layer outputs still return to the host between layers. The next speed step is a true Metal-side prefill wave with persistent chunk hidden buffers across layers and causal full-attention chunk kernels.

### [LM-codex-DN-CHUNK-ROWWISE-1] DeltaNet prefill chunk keeps state rows register-resident
**status:** verified-default
**trust:** {F:0.88, G:0.62, R:0.86}
**context:** ml (Qwen prefill optimization)
**evidence:**
- claim: "Added `delta_net_chunk_128_rowwise`, a default-on s=128 prefill chunk kernel that assigns four state rows per threadgroup and keeps each row stripe in simdgroup registers across the token scan; disable with `QWEN35_DN_CHUNK_ROWWISE_OFF=1`."
  source: `src/ml/gguf/kernels/delta_net.metal` and `src/ml/gguf/qwen35_metal.cr` on 2026-04-23
  verified_at: 2026-04-23
  decay_trigger: DeltaNet recurrence semantics, state layout, or prefill chunk launch geometry changes
- claim: "Correctness gates pass with rowwise DeltaNet enabled: isolated DeltaNet spec `3 examples, 0 failures`; targeted Qwen gate `26 examples, 0 failures`; full-logit A/B against rowwise-off on a 16-token prompt gives same top1 `198`, cosine `1.0`, max_abs `0.0`."
  source: `crystal spec spec/qwen35_delta_net_spec.cr ...`, `crystal spec spec/qwen35_metal_spec.cr spec/qwen35_forward_spec.cr spec/qwen35_delta_net_spec.cr spec/qwen35_state_snapshot_spec.cr spec/qwen35_prompt_cache_spec.cr ...`, and temporary A/B harness on 2026-04-23
  verified_at: 2026-04-23
  decay_trigger: spec fixtures, full-logit comparison harness, or Metal bridge changes
- claim: "pp64 profile improved DeltaNet wait from `148.06 ms` to `121.28 ms`; total profiled prefill improved from `229.31 ms` (`279.10 tok/s`) to `200.06 ms` (`319.90 tok/s`)."
  source: temporary pp64 profile harness with and without `QWEN35_DN_CHUNK_ROWWISE_OFF=1` on 2026-04-23
  verified_at: 2026-04-23
  decay_trigger: host load, power state, benchmark harness, or Metal driver behavior changes
- claim: "Matched prompt=64/gen=16/reps=3/warmup=1 benchmark measured native prefill p50 `308.86 tok/s` vs llama.cpp `457.81 tok/s`; decode stayed ahead at native `46.80 tok/s` vs llama.cpp `44.08 tok/s`."
  source: `CRYSTAL_CACHE_DIR=/tmp/cogni_ml_crystal_cache_vs_llama_rowwise crystal run --link-flags="build/bridge.o -framework Metal -framework Foundation -lc++" bin/benchmark_qwen_vs_llama.cr -- --prompt=64 --gen=16 --reps=3 --warmup=1` on 2026-04-23
  verified_at: 2026-04-23
  decay_trigger: benchmark harness, host load, llama.cpp version, or Qwen prefill path changes
**note:** This closes the obvious state-traffic waste inside the chunked DeltaNet scan. The remaining first-run prefill gap is now more likely inter-layer host readbacks plus non-DeltaNet work than the recurrent scan kernel itself.

### [LM-codex-PREFILL-RECURRENT-RUN-1] Consecutive recurrent prefill layers stay GPU-resident
**status:** verified-default
**trust:** {F:0.88, G:0.62, R:0.84}
**context:** ml (Qwen prefill optimization)
**evidence:**
- claim: "Added `Qwen35Metal.recurrent_layer_chunk_project_many` and routed consecutive recurrent layer runs through it by default; disable with `QWEN35_PREFILL_REC_RUN_OFF=1`. The primitive keeps the token-major hidden matrix on GPU across recurrent runs between full-attention layers."
  source: `src/ml/gguf/qwen35_metal.cr` and `src/ml/gguf/qwen35_cpu.cr` on 2026-04-23
  verified_at: 2026-04-23
  decay_trigger: Qwen layer ordering, recurrent prefill chunk kernels, or hidden-buffer layout changes
- claim: "Correctness gates pass: targeted specs `10 examples, 0 failures`; full targeted Qwen gate `26 examples, 0 failures`; full-logit A/B against `QWEN35_PREFILL_REC_RUN_OFF=1` on a 16-token prompt gives same top1 `198`, cosine `1.0`, max_abs `0.0`."
  source: `crystal spec spec/qwen35_forward_spec.cr spec/qwen35_delta_net_spec.cr ...`, full targeted spec command, and temporary A/B harness on 2026-04-23
  verified_at: 2026-04-23
  decay_trigger: spec fixtures, prefill routing, or Metal bridge changes
- claim: "pp64 profile improved total syncs from `33` to `17`; profiled prefill improved from `204.45 ms` (`313.04 tok/s`) with `QWEN35_PREFILL_REC_RUN_OFF=1` to `197.98 ms` (`323.27 tok/s`) default."
  source: temporary pp64 profile harness with and without `QWEN35_PREFILL_REC_RUN_OFF=1` on 2026-04-23
  verified_at: 2026-04-23
  decay_trigger: host load, power state, benchmark harness, or Metal driver behavior changes
- claim: "Matched prompt=64/gen=16/reps=3/warmup=1 benchmark measured native prefill p50 `327.54 tok/s` vs llama.cpp `459.46 tok/s`; decode remained slightly ahead at native `44.89 tok/s` vs llama.cpp `44.23 tok/s`."
  source: `CRYSTAL_CACHE_DIR=/tmp/cogni_ml_crystal_cache_vs_llama_recrun crystal run --link-flags="build/bridge.o -framework Metal -framework Foundation -lc++" bin/benchmark_qwen_vs_llama.cr -- --prompt=64 --gen=16 --reps=3 --warmup=1` on 2026-04-23
  verified_at: 2026-04-23
  decay_trigger: benchmark harness, host load, llama.cpp version, or Qwen prefill path changes
- claim: "A bounded `QWEN35_ROPE_ROWS_TABLE=1` experiment was correct but not a speed win: pp64 was `201.27 ms` default vs `201.73 ms` with row RoPE tables, so the branch was discarded."
  source: temporary pp64 profile harness with and without `QWEN35_ROPE_ROWS_TABLE=1` on 2026-04-23
  verified_at: 2026-04-23
  decay_trigger: full-attention chunk kernel rewrite or RoPE implementation changes
**note:** This reduces orchestration/readback overhead within recurrent stretches. The remaining exact prefill gap is now the boundary between full-attention and recurrent chunks plus the first-run final-token decode included in the native benchmark.

### [LM-codex-PREFILL-FINAL-CHUNK-1] Final prompt token is batched into exact prefill
**status:** verified-default
**trust:** {F:0.88, G:0.66, R:0.86}
**context:** ml (Qwen prefill optimization)
**evidence:**
- claim: "Added `Qwen35CPU.prefill_tokens_top1`, which processes the full prompt span including the final token through the chunked prefill path, then applies a fused Metal lm-head top1 to the final hidden row. The old final-token decode path remains available with `QWEN35_PREFILL_FINAL_CHUNK_OFF=1`."
  source: `src/ml/gguf/qwen35_cpu.cr`, `src/ml/gguf/qwen35_metal.cr`, `bin/benchmark_qwen_vs_llama.cr`, `bin/qwen35_generate.cr`, and `src/ml/gguf/qwen35_prompt_cache.cr` on 2026-04-23
  verified_at: 2026-04-23
  decay_trigger: prompt prefill routing, output head top1 kernel, or benchmark semantics changes
- claim: "Correctness gate passes: targeted Qwen specs `15 examples, 0 failures`; full targeted gate `26 examples, 0 failures`; A/B against `QWEN35_PREFILL_FINAL_CHUNK_OFF=1` on a 64-token prompt gives same top1 `72`, logit delta `0.00035572052`."
  source: `crystal spec spec/qwen35_forward_spec.cr spec/qwen35_delta_net_spec.cr spec/qwen35_prompt_cache_spec.cr ...`, full targeted spec command, and temporary A/B harness on 2026-04-23
  verified_at: 2026-04-23
  decay_trigger: spec fixtures, top1 tolerance, or prompt-cache replay changes
- claim: "Matched prompt=64/gen=16/reps=3/warmup=1 benchmark improved native prefill p50 from `308.82 tok/s` with `QWEN35_PREFILL_FINAL_CHUNK_OFF=1` to `358.44 tok/s` default; llama.cpp measured `458.39 tok/s` in the default run. Decode measured native `47.02 tok/s` vs llama.cpp `44.06 tok/s`."
  source: `CRYSTAL_CACHE_DIR=/tmp/cogni_ml_crystal_cache_vs_llama_final_chunk crystal run --link-flags="build/bridge.o -framework Metal -framework Foundation -lc++" bin/benchmark_qwen_vs_llama.cr -- --prompt=64 --gen=16 --reps=3 --warmup=1` and same command with `QWEN35_PREFILL_FINAL_CHUNK_OFF=1` on 2026-04-23
  verified_at: 2026-04-23
  decay_trigger: benchmark harness, host load, llama.cpp version, or Qwen prefill path changes
- claim: "A same-binary A/B refuted changing `QWEN35_WAVE_CHUNK_LAYERS` default from `4` to `2`; paired 64/16 decode runs were effectively equal around `22.5 ms/tok`, so the default stayed `4`."
  source: `/tmp/qwen35_sync_profile_wavechunk_ab` paired runs with `QWEN35_WAVE_CHUNK_LAYERS=2/4` on 2026-04-23
  verified_at: 2026-04-23
  decay_trigger: decode wave scheduler or command-buffer behavior changes
**note:** This is exact for greedy prompt ingestion because intermediate prompt-token logits are unobserved, and the final prompt token still produces next-token top1 before generation starts.

### [LM-codex-Q4K-GEMM-DIRECT-STORE-1] Q4_K prefill GEMM skips shmem output staging on full tiles
**status:** verified-default
**trust:** {F:0.86, G:0.64, R:0.82}
**context:** ml (Qwen prefill optimization)
**evidence:**
- claim: "Changed `simd_mm_q4k_f32` to write full 64x32 output tiles directly from simdgroup accumulators to device memory, keeping the previous cooperative shmem staging path for edge tiles. This follows the local llama.cpp Metal `kernel_mul_mm` fast path and is exact for full tiles."
  source: `src/ml/gguf/kernels/gemm_q4k.metal` on 2026-04-23
  verified_at: 2026-04-23
  decay_trigger: Q4_K GEMM tiling, output layout, or batch GEMM routing changes
- claim: "Correctness gate passes after the direct-store fast path: targeted Qwen specs `26 examples, 0 failures`; Q4_K GEMM spec reports cosine `1.0`, max delta `0.0004899502`."
  source: `CRYSTAL_CACHE_DIR=/tmp/cogni_ml_crystal_cache_q4_direct_gate crystal spec spec/qwen35_metal_spec.cr spec/qwen35_forward_spec.cr spec/qwen35_delta_net_spec.cr spec/qwen35_state_snapshot_spec.cr spec/qwen35_prompt_cache_spec.cr ...` on 2026-04-23
  verified_at: 2026-04-23
  decay_trigger: Metal compiler behavior, spec fixtures, or Q4_K kernel changes
- claim: "Matched prompt=64/gen=16/reps=5/warmup=1 benchmark measured native prefill p50 `373.80 tok/s` vs llama.cpp `464.02 tok/s`; decode p50 reached `48.22 tok/s` vs llama.cpp `45.13 tok/s`."
  source: `CRYSTAL_CACHE_DIR=/tmp/cogni_ml_crystal_cache_q4_direct_bench_final crystal run --link-flags="build/bridge.o -framework Metal -framework Foundation -lc++" bin/benchmark_qwen_vs_llama.cr -- --prompt=64 --gen=16 --reps=5 --warmup=1` on 2026-04-23
  verified_at: 2026-04-23
  decay_trigger: host load, power state, benchmark harness, llama.cpp version, or Qwen prefill path changes
- claim: "A no-bias F32-output Q5_K/Q6_K GEMM branch was correct but slower in isolated specs (`Q5_K batch=16` around `138.9 ms`, `Q6_K batch=16` around `42.5 ms`), so it was discarded and the F16 activation/output Q5/Q6 path remains."
  source: temporary Q5/Q6 F32 GEMM branch spec run on 2026-04-23
  verified_at: 2026-04-23
  decay_trigger: Q5/Q6 GEMM kernel rewrite or Metal compiler behavior changes
**note:** The direct-store Q4_K change is a local kernel win that moved pp64 p50 from the previous `358.44 tok/s` landmark to `373.80 tok/s`. It does not solve the remaining prefill gap by itself; the next measured wall is still recurrent/full-attn chunk work dominated by quantized batched matmuls and remaining readbacks.

### [LM-codex-Q4K-PAIR-H16-1] Large-batch Q4_K FFN gate/up reuses one half-input conversion
**status:** verified-default-gated
**trust:** {F:0.86, G:0.56, R:0.82}
**context:** ml (Qwen prefill optimization)
**evidence:**
- claim: "Added a default-on, batch-gated Q4_K FFN gate/up pair route for prefill. It converts the shared normalized F32 activation matrix to F16 once, then feeds both Q4_K half-input GEMMs. The route is disabled with `QWEN35_Q4K_PAIR_H16_GEMM_OFF=1` and only activates at batch `>=256` because pp64 adversary runs were not reliable."
  source: `src/ml/gguf/qwen35_metal.cr` on 2026-04-24
  verified_at: 2026-04-24
  decay_trigger: Q4_K half-input GEMM semantics, FFN gate/up layout, or prefill batch routing changes
- claim: "Correctness gates pass with the default-gated route: targeted Qwen forward and DeltaNet specs report `14 examples, 0 failures`."
  source: `CRYSTAL_CACHE_DIR=/tmp/cogni_ml_crystal_cache_q4pair_gate_spec crystal spec spec/qwen35_forward_spec.cr spec/qwen35_delta_net_spec.cr --link-flags=...` on 2026-04-24
  verified_at: 2026-04-24
  decay_trigger: spec fixtures, Metal kernels, or Qwen35 prefill path changes
- claim: "Isolated pp256 paired A/B shows the gated default route faster than `QWEN35_Q4K_PAIR_H16_GEMM_OFF=1`: avg `488.35 ms` default vs `489.91 ms` off, p50 `486.48 ms` vs `488.69 ms`, wins `7/10`."
  source: `CRYSTAL_CACHE_DIR=/tmp/cogni_ml_crystal_cache_q4pair_gate_ab256_iso crystal run --release ... bin/qwen35_prefill_attribution.cr -- --prompt=256 --warmup=2 --reps=10 --compare-env=QWEN35_Q4K_PAIR_H16_GEMM_OFF --compare-off=1` on 2026-04-24
  verified_at: 2026-04-24
  decay_trigger: host load, power state, benchmark harness, or Q4_K prefill route changes
- claim: "pp64 is intentionally not routed through the pair helper: after adding the `batch >= 256` gate, an isolated pp64 A/B is neutral within noise (`150.94 ms` default vs `150.84 ms` off avg, wins `3/6`), while an earlier ungated pp64 run was noisy/regression-prone."
  source: `CRYSTAL_CACHE_DIR=/tmp/cogni_ml_crystal_cache_q4pair_gate_ab64_iso crystal run --release ... --prompt=64 --warmup=2 --reps=6 --compare-env=QWEN35_Q4K_PAIR_H16_GEMM_OFF --compare-off=1` on 2026-04-24
  verified_at: 2026-04-24
  decay_trigger: prefill batch threshold, host load, or Q4_K pair routing changes
**note:** This is a small exact long-prefill cleanup, not a paradigm shift. It removes duplicated activation conversion for paired FFN projections but does not reduce the dominant weight traffic.

### [LM-codex-REC-PROJ-SHARED-H16-FALSIFIER-1] Recurrent projection shared H16 conversion is neutral
**status:** stale historical falsifier; superseded by [LM-codex-QWEN35-REC-PROJ-SHARED-H16-1]
**trust:** {F:0.82, G:0.34, R:0.42}
**context:** ml (Qwen prefill optimization)
**evidence:**
- claim: "An opt-in branch preconverted the recurrent prefill normed activation matrix to F16 once and reused it for Q5_K qkv half-output GEMM plus Q4_K gate/z half-input GEMM. The branch preserved the existing half-input arithmetic and passed focused Qwen forward/DeltaNet specs."
  source: temporary `QWEN35_REC_PROJ_SHARED_H16=1` branch in `src/ml/gguf/qwen35_metal.cr` on 2026-04-24
  verified_at: 2026-04-24
  decay_trigger: recurrent projection layout, Q5/Q4 GEMM kernels, or prefill batch routing changes
- claim: "The branch was not a speed win at pp256: paired A/B measured default avg `485.56 ms` / p50 `485.59 ms` vs opt-in avg `485.59 ms` / p50 `485.70 ms`, wins `4/8`."
  source: `CRYSTAL_CACHE_DIR=/tmp/cogni_ml_crystal_cache_recproj_shared_ab256 QWEN35_Q4K_PAIR_H16_GEMM_OFF=1 crystal run --release ... bin/qwen35_prefill_attribution.cr -- --prompt=256 --warmup=2 --reps=8 --compare-env=QWEN35_REC_PROJ_SHARED_H16 --compare-off=1` on 2026-04-24
  verified_at: 2026-04-24
  decay_trigger: host load, power state, benchmark harness, or Q4/Q5 projection kernels change
**note:** This was valid for the temporary 2026-04-24 branch with `QWEN35_Q4K_PAIR_H16_GEMM_OFF=1` and older surrounding routes. The 2026-04-25 retained implementation is a refreshed default-on cleanup after later Q4/Q5 routing changes; it remains small, not a breakthrough.

### [LM-codex-CONVERSION-ATTRIBUTION-1] Prefill profile reports F32/F16 conversion traffic
**status:** verified-instrumentation
**trust:** {F:0.84, G:0.62, R:0.82}
**context:** ml (Qwen prefill attribution)
**evidence:**
- claim: "Added `Qwen35Metal::Profile.bump_conversion` and report rows for conversion kernels. The report now separates logical `F32->F16` and `F16->F32` traffic from quantized matmul weight traffic, scoped by the current profile trace label."
  source: `src/ml/gguf/qwen35_metal.cr` on 2026-04-24
  verified_at: 2026-04-24
  decay_trigger: Profile report format, Q4/Q56 GEMM conversion path, or prefill attribution harness changes
- claim: "Correctness gate remains green after instrumentation: targeted Qwen forward and DeltaNet specs report `14 examples, 0 failures`."
  source: `CRYSTAL_CACHE_DIR=/tmp/cogni_ml_crystal_cache_conversion_profile_spec crystal spec spec/qwen35_forward_spec.cr spec/qwen35_delta_net_spec.cr --link-flags=...` on 2026-04-24
  verified_at: 2026-04-24
  decay_trigger: spec fixtures, Metal kernels, or Qwen35 prefill path changes
- claim: "A non-baseline pp16 smoke run printed per-row traffic percentages plus logical traffic totals/mix: top matmul row `prefill.rec.ffn_upgate q4_h16_gemm` was `32.46%` of logical weight traffic; totals were matmul `3992.53 MiB`, conversion `102.47 MiB`, traffic mix `97.50%` matmul / `2.50%` conversion. The run was only used to verify report formatting/totals, not speed."
  source: `CRYSTAL_CACHE_DIR=/tmp/cogni_ml_crystal_cache_rowpct_smoke crystal build --release bin/qwen35_prefill_attribution.cr ... && /tmp/qwen35_prefill_attribution_rowpct --prompt=16 --warmup=0 --reps=1 --load-warning-threshold=0 --load-total-warning-threshold=0 | sed -n '/matmul shapes:/,/logical traffic mix:/p' | head -32` on 2026-04-24
  verified_at: 2026-04-24
  decay_trigger: report formatting or conversion path changes
**note:** This is measurement infrastructure, not a speedup. It supports the next quiet-host phase by quantifying whether a proposed conversion-elimination branch can matter before changing kernels.

### [LM-codex-GUARDED-BASELINE-20260424-1] Guarded pp64 baseline after attribution/cleanup
**status:** verified-baseline
**trust:** {F:0.86, G:0.56, R:0.84}
**context:** ml (Qwen prefill/decode benchmark)
**evidence:**
- claim: "Guarded prefill attribution for prompt=64 measured p50 `151.73 ms` / `421.80 tok/s` with 10 Metal syncs. Logical traffic was matmul `3992.53 MiB`, conversion `409.88 MiB`, mix `90.69%` matmul / `9.31%` conversion. Top logical-weight row remains `prefill.rec.ffn_upgate q4_h16_gemm` at `32.46%` of matmul traffic."
  source: `CRYSTAL_CACHE_DIR=/tmp/cogni_ml_crystal_cache_guarded_attr_build crystal build --release bin/qwen35_prefill_attribution.cr ... && /tmp/qwen35_prefill_attribution_guarded --prompt=64 --warmup=2 --reps=5 --require-quiet --wait-quiet-ms=60000 --quiet-poll-ms=1000` on 2026-04-24
  verified_at: 2026-04-24
  decay_trigger: host load guard, benchmark harness, Qwen35 prefill path, or Metal kernels change
- claim: "Guarded matched prompt64/gen64 benchmark measured native prefill p50 `426.01 tok/s` vs llama.cpp `461.90 tok/s` (`-7.77%`), and native decode p50 `47.60 tok/s` vs llama.cpp `45.35 tok/s` (`+4.96%`)."
  source: `CRYSTAL_CACHE_DIR=/tmp/cogni_ml_crystal_cache_guarded_llama_build crystal build --release bin/benchmark_qwen_vs_llama.cr ... && /tmp/benchmark_qwen_vs_llama_guarded --prompt=64 --gen=64 --warmup=2 --reps=5 --require-quiet --wait-quiet-ms=60000 --quiet-poll-ms=1000` on 2026-04-24
  verified_at: 2026-04-24
  decay_trigger: host load guard, llama.cpp build/version, benchmark harness, or Qwen35 prefill/decode path changes
**note:** This supersedes earlier noisy matched runs. Decode is ahead of llama.cpp by about 5%; prefill is still behind by about 8%, and attribution says the remaining exact target is quantized matmul/work elimination, not conversion cleanup.

### [LM-codex-Q4K-H16-BARRIER-FALSIFIER-1] Q4_K H16 simdgroup load barriers should stay
**status:** verified-falsifier
**trust:** {F:0.82, G:0.42, R:0.78}
**context:** ml (Qwen prefill kernel optimization)
**evidence:**
- claim: "A temporary branch removed the three inner `simdgroup_barrier(mem_flags::mem_none)` calls around `simdgroup_load` and `simdgroup_multiply_accumulate` in `simd_mm_q4k_h16`, leaving the F32 Q4_K kernel unchanged. Focused Qwen forward/DeltaNet specs still passed."
  source: `src/ml/gguf/kernels/gemm_q4k.metal` temporary branch and `CRYSTAL_CACHE_DIR=/tmp/cogni_ml_crystal_cache_q4h16_nosgb_spec crystal spec spec/qwen35_forward_spec.cr spec/qwen35_delta_net_spec.cr --link-flags=...` on 2026-04-24
  verified_at: 2026-04-24
  decay_trigger: Q4_K H16 GEMM kernel, Metal compiler, or simdgroup_matrix load behavior changes
- claim: "The branch regressed guarded pp64 attribution: baseline p50 `151.73 ms` / `421.80 tok/s` versus barrier-removal p50 `154.49 ms` / `414.26 tok/s`, with the same logical traffic profile."
  source: `/tmp/qwen35_prefill_attribution_q4h16_nosgb --prompt=64 --warmup=2 --reps=5 --require-quiet --wait-quiet-ms=60000 --quiet-poll-ms=1000` on 2026-04-24
  verified_at: 2026-04-24
  decay_trigger: host load guard, Metal compiler, benchmark harness, or Q4_K H16 GEMM route changes
**note:** Correctness alone was insufficient here; this is a measured micro-optimization trap. Do not retry barrier removal unless the kernel structure or compiler behavior changes.

### [LM-codex-DECODE-WAVE-CHUNK-GUARD-1] Decode wave chunk default remains 2 under guarded A/B
**status:** verified-falsifier
**trust:** {F:0.84, G:0.52, R:0.82}
**context:** ml (Qwen decode wave scheduling)
**evidence:**
- claim: "The paired decode A/B helper now supports the same quiet-host guard options as the main benchmark harness, so wave scheduler experiments can wait for/require low background CPU load before measuring."
  source: `bin/qwen35_ab_profile.cr --help` after release build on 2026-04-24
  verified_at: 2026-04-24
  decay_trigger: benchmark harness or `BenchLoadGuard` changes
- claim: "`QWEN35_WAVE_CHUNK_LAYERS=2` remains materially better than unchunked `0`: guarded prompt64/gen64 trials measured A mean `22.848 ms/tok` vs B mean `24.183 ms/tok`, wins `6/6`, delta `-1.335 ms/tok`."
  source: `/tmp/qwen35_ab_profile_guarded --env=QWEN35_WAVE_CHUNK_LAYERS --a=2 --b=0 --prompt=64 --gen=64 --trials=6 --warmup=1 --require-quiet --wait-quiet-ms=60000 --quiet-poll-ms=1000` on 2026-04-24
  verified_at: 2026-04-24
  decay_trigger: decode wave scheduler, Metal command queue behavior, or host-load guard changes
- claim: "Smaller/larger nearby chunking does not justify a default change: guarded `2` vs `4` was neutral (`3/6`, delta `-0.071 ms/tok`), and guarded `1` vs `2` was also neutral/noisy (`4/6`, delta `-0.066 ms/tok`)."
  source: guarded `/tmp/qwen35_ab_profile_guarded` runs for `2/4` and `1/2` on 2026-04-24
  verified_at: 2026-04-24
  decay_trigger: decode wave scheduler or benchmark harness changes
**note:** The missing decode margin is not recovered by retuning command-buffer chunk size. Keep default `2` and look at decode compute work or output-head path next.

### [LM-codex-ADDNORM-H16-FFN-FALSIFIER-1] Direct H16 addnorm output for Q4 FFN pair is neutral
**status:** verified-falsifier
**trust:** {F:0.78, G:0.44, R:0.74}
**context:** ml (Qwen prefill optimization)
**evidence:**
- claim: "A temporary opt-in branch added `qwen35_add_rmsnorm_rows_h16` and routed large-batch Q4_K FFN gate/up pair prefill through direct H16 normalized rows. This preserves the effective Q4 H16 GEMM input precision because the default path already casts normalized F32 rows to H16 before the Q4 GEMM."
  source: temporary `QWEN35_ADDNORM_H16_FFN=1` branch in `src/ml/gguf/kernels/ffn_qwen35.metal` and `src/ml/gguf/qwen35_metal.cr` on 2026-04-24
  verified_at: 2026-04-24
  decay_trigger: add+RMSNorm row kernel, Q4_K pair route, or Q4 H16 GEMM input semantics changes
- claim: "Correctness stayed green with the opt-in route: targeted Qwen forward/DeltaNet specs reported `14 examples, 0 failures`, top token `198`, logit `11.423702`."
  source: `CRYSTAL_CACHE_DIR=/tmp/cogni_ml_crystal_cache_addnorm_h16_spec2 QWEN35_ADDNORM_H16_FFN=1 crystal spec spec/qwen35_forward_spec.cr spec/qwen35_delta_net_spec.cr --link-flags=...` on 2026-04-24
  verified_at: 2026-04-24
  decay_trigger: spec fixtures or Qwen35 prefill path changes
- claim: "Relaxed-load paired pp256 A/B did not support promotion: default avg `506.53 ms` / p50 `510.47 ms` versus opt-in avg `506.29 ms` / p50 `509.64 ms`, wins `5/10`."
  source: `/tmp/qwen35_prefill_attribution_addnorm_h16_2 --prompt=256 --warmup=2 --reps=10 --compare-env=QWEN35_ADDNORM_H16_FFN --compare-off=1 --load-warning-threshold=150 --load-total-warning-threshold=500` on 2026-04-24
  verified_at: 2026-04-24
  decay_trigger: host load, benchmark harness, or Q4 pair route changes
**note:** This is another conversion-elimination falsifier. The branch was removed; future exact prefill wins need to reduce dominant quantized weight traffic or change the lower-level Q4/Q6 tile work, not just move the F32->H16 cast into addnorm.

### [LM-codex-RELAXED-CPU-BASELINE-20260424-1] Relaxed-load baseline after CPU-core clarification
**status:** verified-baseline
**trust:** {F:0.76, G:0.50, R:0.74}
**context:** ml (Qwen prefill/decode benchmark)
**evidence:**
- claim: "After treating one busy M2 CPU core as normal host activity rather than a hard benchmark blocker, sequential relaxed-load prefill attribution stayed consistent with the guarded baseline: pp64 p50 `150.80 ms` / `424.39 tok/s`, 10 Metal syncs, logical traffic mix `90.69%` matmul / `9.31%` conversion."
  source: `/tmp/qwen35_prefill_attribution_relaxed --prompt=64 --warmup=2 --reps=5 --load-warning-threshold=150 --load-total-warning-threshold=500` on 2026-04-24
  verified_at: 2026-04-24
  decay_trigger: host load, power state, benchmark harness, or Qwen35 prefill path changes
- claim: "Sequential relaxed pp256 prefill attribution measured p50 `486.56 ms` / `526.14 tok/s`, 10 Metal syncs, logical traffic mix `73.31%` matmul / `26.69%` conversion."
  source: `/tmp/qwen35_prefill_attribution_relaxed --prompt=256 --warmup=1 --reps=3 --load-warning-threshold=150 --load-total-warning-threshold=500` on 2026-04-24
  verified_at: 2026-04-24
  decay_trigger: host load, power state, benchmark harness, or Qwen35 prefill path changes
- claim: "Relaxed matched prompt64/gen64 benchmark measured native prefill p50 `422.41 tok/s` vs llama.cpp `461.35 tok/s` (`-8.44%`), and native decode p50 `46.59 tok/s` vs llama.cpp `44.38 tok/s` (`+4.98%`)."
  source: `/tmp/benchmark_qwen_vs_llama_relaxed --prompt=64 --gen=64 --warmup=2 --reps=5 --load-warning-threshold=150 --load-total-warning-threshold=500` on 2026-04-24
  verified_at: 2026-04-24
  decay_trigger: host load, llama.cpp build/version, benchmark harness, or Qwen35 prefill/decode path changes
**note:** Relaxed-load evidence is useful for iteration on a busy desktop but weaker than `--require-quiet` evidence. Do not use it to claim a public speedup unless a guarded rerun agrees.

### [LM-codex-DECODE-GLUE-KNOBS-FALSIFIER-1] Decode scheduler/glue knobs are not the next win
**status:** verified-falsifier
**trust:** {F:0.76, G:0.44, R:0.74}
**context:** ml (Qwen decode wave scheduling)
**evidence:**
- claim: "Changing `QWEN35_WAVE_CHUNK_LAYERS` from default `2` to `3` is neutral: paired prompt64/gen64 A/B measured A mean `23.009 ms/tok` vs B mean `22.997 ms/tok`, wins `4/8`, delta `+0.013 ms/tok`."
  source: `/tmp/qwen35_ab_profile_guarded --env=QWEN35_WAVE_CHUNK_LAYERS --a=2 --b=3 --prompt=64 --gen=64 --trials=8 --warmup=1 --load-warning-threshold=150 --load-total-warning-threshold=500` on 2026-04-24
  verified_at: 2026-04-24
  decay_trigger: decode wave scheduler, host load, or benchmark harness changes
- claim: "`QWEN35_WAVE_FAST_CMD=1` remains neutral vs off on decode: paired A/B measured A mean `22.971 ms/tok` vs B mean `22.989 ms/tok`, wins `3/6`, delta `-0.018 ms/tok`."
  source: `/tmp/qwen35_ab_profile_guarded --env=QWEN35_WAVE_FAST_CMD --a=1 --b=0 --prompt=64 --gen=64 --trials=6 --warmup=1 --load-warning-threshold=150 --load-total-warning-threshold=500` on 2026-04-24
  verified_at: 2026-04-24
  decay_trigger: Metal command-buffer behavior or benchmark harness changes
- claim: "`QWEN35_REC_CONVSHIFT_FUSED=1` is not a decode win: paired A/B measured fused A mean `22.912 ms/tok` vs default/off B mean `22.887 ms/tok`, wins `3/8`, delta `+0.025 ms/tok`."
  source: `/tmp/qwen35_ab_profile_guarded --env=QWEN35_REC_CONVSHIFT_FUSED --a=1 --b=0 --prompt=64 --gen=64 --trials=8 --warmup=1 --load-warning-threshold=150 --load-total-warning-threshold=500` on 2026-04-24
  verified_at: 2026-04-24
  decay_trigger: recurrent conv-shift kernels, decode wave scheduler, or benchmark harness changes
- claim: "`QWEN35_DN_POST_FUSED=1` is neutral in the current decode wave: paired A/B measured A mean `23.200 ms/tok` vs B mean `23.214 ms/tok`, wins `3/8`, delta `-0.014 ms/tok`."
  source: `/tmp/qwen35_ab_profile_guarded --env=QWEN35_DN_POST_FUSED --a=1 --b=0 --prompt=64 --gen=64 --trials=8 --warmup=1 --load-warning-threshold=150 --load-total-warning-threshold=500` on 2026-04-24
  verified_at: 2026-04-24
  decay_trigger: DeltaNet post-fusion kernels, decode wave scheduler, or benchmark harness changes
**note:** These measurements point back to quantized GEMV traffic. Retuning command-buffer chunks or small glue kernels is now a micro-optimization trap unless a new trace shows scheduler overhead rising again.

### [LM-codex-Q6-NR2-FALSIFIER-1] Q6_K GEMV NR0=2 is slower on local decode
**status:** verified-falsifier
**trust:** {F:0.72, G:0.38, R:0.68}
**context:** ml (Qwen decode kernel optimization)
**evidence:**
- claim: "A temporary branch changed local `MV6_NR0` from `1` to llama.cpp-style `2` in `simd_mv_q6k_f32`. Focused Qwen forward/DeltaNet specs still passed (`14 examples, 0 failures`)."
  source: temporary `src/ml/gguf/kernels/gemm_q56k.metal` branch and `CRYSTAL_CACHE_DIR=/tmp/cogni_ml_crystal_cache_q6nr2_spec crystal spec spec/qwen35_forward_spec.cr spec/qwen35_delta_net_spec.cr --link-flags=...` on 2026-04-24
  verified_at: 2026-04-24
  decay_trigger: Q6_K GEMV kernel, Metal compiler, or Qwen35 decode path changes
- claim: "The branch slowed same-sequence decode sync profile: old binary `MV6_NR0=1` measured `24.60 ms/tok`, while new `MV6_NR0=2` measured `25.67 ms/tok`; the branch also increased encode accounting. This was not a strict interleaved A/B, so the falsifier is narrower than the guarded env-based ones."
  source: `QWEN35_PROFILE_TOP1=1 /tmp/qwen35_sync_profile_relaxed 64 16 && QWEN35_PROFILE_TOP1=1 /tmp/qwen35_sync_profile_q6nr2 64 16` on 2026-04-24
  verified_at: 2026-04-24
  decay_trigger: host load, Metal compiler, Q6_K GEMV implementation, or decode wave scheduler changes
**note:** Keep local `MV6_NR0=1`. This is one of the cases where blindly matching llama.cpp launch geometry is worse on the M2 Max local path.

### [LM-codex-Q8-MIXED-DUAL-GEMV-FALSIFIER-1] Q8_0 mixed-size recurrent projection dual GEMV is slower
**status:** verified-falsifier
**trust:** {F:0.72, G:0.38, R:0.70}
**context:** ml (Qwen speculative draft decode kernel optimization)
**evidence:**
- claim: "A temporary branch added `simd_mv_q8_0_dual_mixed_f32` and routed Q8_0 draft recurrent `qkv + gate` GEMVs through one mixed-output kernel. The first draft had an invalid collective placement (`simd_sum` under `tiisg == 0`); it was corrected before measurement."
  source: temporary `src/ml/gguf/kernels/gemm_q56k.metal` and `src/ml/gguf/qwen35_metal.cr` branch on 2026-04-24
  verified_at: 2026-04-24
  decay_trigger: Q8_0 GEMV kernel, recurrent projection route, or Metal compiler behavior changes
- claim: "The corrected mixed kernel reduced encode attribution for recurrent projection from about `2.96-3.00 ms` to `2.36-2.42 ms` across 288 calls in a 16-token Qwen3.5 0.8B Q8_0 draft decode profile."
  source: `QWEN35_MODEL=...Qwen3.5-0.8B-Q8_0.gguf QWEN35_PROFILE_TOP1=1 /tmp/qwen35_sync_profile_q8mixed2 64 16`, interleaved `QWEN35_Q8_MIXED_DUAL_GEMV_OFF=1` vs default on 2026-04-24
  verified_at: 2026-04-24
  decay_trigger: host load, profiler accounting, or decode wave route changes
- claim: "End-to-end wall time regressed despite lower encode attribution: separate GEMVs measured `126.1 ms` and `125.6 ms` for 16 tokens (`7.88` and `7.85 ms/tok`), while mixed dual measured `128.8 ms` and `129.0 ms` (`8.05` and `8.06 ms/tok`)."
  source: same interleaved `/tmp/qwen35_sync_profile_q8mixed2 64 16` run on 2026-04-24
  verified_at: 2026-04-24
  decay_trigger: host load, Q8_0 GEMV kernel, Metal compiler, or draft model quantization changes
**note:** Do not re-add mixed-size Q8 recurrent `qkv + gate` dual GEMV as a default path without a different kernel design. Saving dispatch/input reads was outweighed by lower occupancy/register pressure, matching the earlier Q4 dual-GEMV failure pattern.

### [LM-codex-Q8-ALPHA-BETA-DUAL-GEMV-1] Q8_0 alpha/beta dual GEMV improves draft decode
**status:** verified
**trust:** {F:0.82, G:0.46, R:0.78}
**context:** ml (Qwen speculative draft decode kernel optimization)
**evidence:**
- claim: "The Qwen3.5 0.8B Q8_0 recurrent `alpha` and `beta` projections have matching input/output dimensions and can reuse the existing exact `simd_mv_q8_0_dual_f32` kernel without a new Metal kernel. The route is guarded by `QWEN35_Q8_ALPHA_BETA_DUAL_GEMV_OFF=1` and also respects the broader `QWEN35_Q8_DUAL_GEMV_OFF=1`."
  source: `src/ml/gguf/qwen35_metal.cr` on 2026-04-24
  verified_at: 2026-04-24
  decay_trigger: Q8_0 dual GEMV kernel, recurrent projection layout, or draft model quantization changes
- claim: "Correctness checks passed after enabling the route: focused Qwen forward specs reported `13 examples, 0 failures`; Metal quantized matmul specs reported `9 examples, 0 failures`."
  source: `CRYSTAL_CACHE_DIR=/tmp/cogni_ml_crystal_cache_q8ab_spec crystal spec spec/qwen35_forward_spec.cr --link-flags=...` and `CRYSTAL_CACHE_DIR=/tmp/cogni_ml_crystal_cache_q8ab_metal_spec crystal spec spec/qwen35_metal_spec.cr --link-flags=...` on 2026-04-24
  verified_at: 2026-04-24
  decay_trigger: spec fixtures, Qwen35 route selection, or Metal bridge changes
- claim: "In a 64-token Qwen3.5 0.8B Q8_0 draft sync profile, default alpha/beta dual routing improved wall time from `561.6 ms` and `569.6 ms` off (`8.77` and `8.90 ms/tok`) to `544.3 ms` and `539.7 ms` default (`8.51` and `8.43 ms/tok`). Recurrent projection encode accounting dropped from `13.24/15.94 ms` to `9.80/9.78 ms` over 1152 calls."
  source: interleaved `QWEN35_Q8_ALPHA_BETA_DUAL_GEMV_OFF=1` vs default with `QWEN35_MODEL=...Qwen3.5-0.8B-Q8_0.gguf QWEN35_PROFILE_TOP1=1 /tmp/qwen35_sync_profile_q8ab 64 64` on 2026-04-24
  verified_at: 2026-04-24
  decay_trigger: host load, Metal compiler, Q8_0 GEMV kernels, or decode wave scheduler changes
- claim: "Speculative harness smokes stayed exact across high-accept and rejection prompts. The gain is mostly visible in draft-only profiles; end-to-end speculative wall is dominated by target verification/fallback on many prompts."
  source: `/tmp/qwen35_speculative_accept_q8ab --tokens 64 --gamma 4 --max-gamma 32` on prompts `The capital of France is`, `def fibonacci(n):`, `Once upon a time`, and `The quick brown fox` on 2026-04-24
  verified_at: 2026-04-24
  decay_trigger: speculative scheduler, target verifier, or draft model changes
**note:** This is a small exact draft-kernel win. It does not change the main conclusion: the next large speculative gain needs lower target verifier cost or a much faster draft, not more mixed-size dual GEMV fusions.

### [LM-codex-SPEC-DRAFT-BACKUP-SKIP-1] Skip unused draft backup before target-only fallback
**status:** verified
**trust:** {F:0.78, G:0.44, R:0.74}
**context:** ml (Qwen speculative scheduler cleanup)
**evidence:**
- claim: "When adaptive speculative decoding has no regrow and `current_gamma // 2` would fall to the target-only fallback threshold, any rejection in the current chunk will skip draft resync. In that case the pre-candidate `draft_cycle_base.copy_from!` is unused and can be skipped exactly."
  source: `bin/qwen35_speculative_accept.cr` on 2026-04-24
  verified_at: 2026-04-24
  decay_trigger: speculative adaptive fallback policy, draft resync policy, or rollback state model changes
- claim: "The harness now reports `draft_backup_skip` and has a guard raise if a resync path is reached after a skipped draft backup."
  source: `bin/qwen35_speculative_accept.cr` on 2026-04-24
  verified_at: 2026-04-24
  decay_trigger: speculative harness reporting or control-flow changes
- claim: "Exact output stayed matched to greedy target in A/B smokes for `The capital of France is`, `def fibonacci(n):`, and `Once upon a time`. On partial-reject prompts the default path reports `draft_backup_skip=1` and `draft_backup=0.0 ms`; `Once upon a time` improved from about `22.50 ms/tok` off to `22.36 ms/tok` default in two sequential pairs."
  source: interleaved `QWEN35_SPEC_SKIP_DRAFT_BACKUP_BEFORE_FALLBACK_OFF=1` vs default with `/tmp/qwen35_speculative_accept_draft_backup --tokens 64 --gamma 4 --max-gamma 32` on 2026-04-24
  verified_at: 2026-04-24
  decay_trigger: host load, speculative scheduler, or target/draft model changes
**note:** This is a small exact cleanup, not the main speculative breakthrough. It removes known-dead state copying on fallback-bound cycles; target verification remains the dominant cost.

### [LM-codex-SPEC-FAST-REGROW-1] Fast adaptive regrowth after gamma 8 improves high-accept prompts
**status:** verified
**trust:** {F:0.80, G:0.48, R:0.76}
**context:** ml (Qwen speculative scheduler optimization)
**evidence:**
- claim: "Unconditional grow-after-one-full-accept is unsafe as a default: it improves the 100%-accept prompt but lets `def fibonacci(n):` grow from `gamma=4` to `8` before the first rejection, regressing to about `23.7 ms/tok` versus about `21.4 ms/tok` on the conservative schedule."
  source: interleaved `QWEN35_SPEC_FULL_ACCEPT_STREAK=2` vs `1` with `/tmp/qwen35_speculative_accept_streak --tokens 64 --gamma 4 --max-gamma 32` on 2026-04-24
  verified_at: 2026-04-24
  decay_trigger: speculative scheduler, draft model, target model, or fallback policy changes
- claim: "A two-stage schedule keeps the conservative two-full-accept requirement at `gamma=4`, then grows after each full accept once `gamma>=8`. This preserves the rejection-heavy trajectory for `def fibonacci(n):`: `cycles=2`, `max_seen=4`, fallback after reject, and matched greedy target output."
  source: `QWEN35_SPEC_FAST_REGROW_MIN_GAMMA=8` default in `bin/qwen35_speculative_accept.cr`; A/B against `QWEN35_SPEC_FAST_REGROW_MIN_GAMMA=0` on `def fibonacci(n):` on 2026-04-24
  verified_at: 2026-04-24
  decay_trigger: adaptive gamma update logic or fallback policy changes
- claim: "On the 100%-accept prompt `The capital of France is`, fast regrowth reduces verifier cycles from `7` to `5` and improves end-to-end exact speculative decode from off runs `18.95/19.37/19.17 ms/tok` to default runs `17.82/16.03/16.27 ms/tok`. Target verifier time drops from about `741-753 ms` to `554-618 ms`."
  source: interleaved `QWEN35_SPEC_FAST_REGROW_MIN_GAMMA=0` vs default with `/tmp/qwen35_speculative_accept_fastregrow --tokens 64 --gamma 4 --max-gamma 32 "The capital of France is"` on 2026-04-24
  verified_at: 2026-04-24
  decay_trigger: host load, target verifier, draft speed, or adaptive schedule changes
**note:** Keep `QWEN35_SPEC_FULL_ACCEPT_STREAK=2` for the initial gamma. The win comes from faster regrowth only after the first safe promotion to gamma 8, avoiding the known `def fibonacci` regression.

### [LM-codex-Q8-KV-DUAL-GEMV-1] Q8_0 full-attention K/V dual GEMV is a small draft win
**status:** verified
**trust:** {F:0.80, G:0.44, R:0.76}
**context:** ml (Qwen speculative draft decode kernel optimization)
**evidence:**
- claim: "Qwen3.5 0.8B Q8_0 full-attention `K` and `V` projections have matching `1024x512` shapes and can reuse the existing exact Q8 dual GEMV kernel. The route is guarded by `QWEN35_Q8_KV_DUAL_GEMV_OFF=1` and respects the broader `QWEN35_Q8_DUAL_GEMV_OFF=1`."
  source: `src/ml/gguf/qwen35_metal.cr` on 2026-04-24
  verified_at: 2026-04-24
  decay_trigger: full-attention projection layout, Q8_0 dual GEMV kernel, or draft quantization changes
- claim: "Focused Qwen forward specs passed after enabling K/V dual routing: `13 examples, 0 failures`."
  source: `CRYSTAL_CACHE_DIR=/tmp/cogni_ml_crystal_cache_q8kv_spec crystal spec spec/qwen35_forward_spec.cr --link-flags=...` on 2026-04-24
  verified_at: 2026-04-24
  decay_trigger: Qwen35 full-attention decode route or spec fixtures change
- claim: "In 64-token Qwen3.5 0.8B Q8_0 draft sync profiles, full.qkv encode accounting dropped from about `3.73-3.75 ms` off to `2.68-2.74 ms` default, and wall time moved from `525.7/521.3 ms` off to `520.5/520.2 ms` default."
  source: interleaved `QWEN35_Q8_KV_DUAL_GEMV_OFF=1` vs default with `QWEN35_MODEL=...Qwen3.5-0.8B-Q8_0.gguf QWEN35_PROFILE_TOP1=1 /tmp/qwen35_sync_profile_q8kv 64 64` on 2026-04-24
  verified_at: 2026-04-24
  decay_trigger: host load, Metal compiler, or Q8_0 GEMV route changes
- claim: "High-accept exact speculative smokes show a small end-to-end gain: K/V dual off measured `15.73/15.73 ms/tok`, default measured `15.68/15.71 ms/tok`, with draft time dropping from about `446 ms` to `443-445 ms`."
  source: interleaved `QWEN35_Q8_KV_DUAL_GEMV_OFF=1` vs default with `/tmp/qwen35_speculative_accept_q8kv --tokens 64 --gamma 4 --max-gamma 32 "The capital of France is"` on 2026-04-24
  verified_at: 2026-04-24
  decay_trigger: speculative scheduler, draft model, or host load changes
**note:** This is worth keeping because it is exact and low-risk, but the gain is small; target verifier and bulk draft GEMV remain the major costs.

### [LM-codex-FALLBACK-BOUND-CHUNK-VERIFY-FALSIFIER-1] Forked chunk verifier is not better for fallback-bound cycles
**status:** verified-falsifier
**trust:** {F:0.76, G:0.42, R:0.74}
**context:** ml (Qwen speculative verifier state strategy)
**evidence:**
- claim: "A temporary branch routed fallback-bound low-gamma cycles from `chunk-inplace` to forked `chunk` verification. The hypothesis was that avoiding `target_backup_state.copy_from!` would help cycles likely to reject and enter target-only fallback."
  source: temporary `bin/qwen35_speculative_accept.cr` branch on 2026-04-24
  verified_at: 2026-04-24
  decay_trigger: verifier state-copy implementation or fallback policy changes
- claim: "The branch removed target backup accounting on rejection prompts but did not improve wall time: `def fibonacci(n):` stayed around `21.5 ms/tok`, and `Once upon a time` stayed around `22.46-22.47 ms/tok`."
  source: interleaved `QWEN35_SPEC_FALLBACK_BOUND_CHUNK_VERIFY=1` vs default with `/tmp/qwen35_speculative_accept_fbchunk --tokens 64 --gamma 4 --max-gamma 32` on 2026-04-24
  verified_at: 2026-04-24
  decay_trigger: host load, verifier state-copy cost, or speculative scheduler changes
- claim: "High-accept prompts regressed because accepted fallback-bound early cycles must copy the forked verifier state back: default `chunk-inplace` measured about `15.79/15.81 ms/tok`, while the forked chunk branch measured about `15.96/16.02 ms/tok`."
  source: same interleaved `/tmp/qwen35_speculative_accept_fbchunk` run on `The capital of France is` on 2026-04-24
  verified_at: 2026-04-24
  decay_trigger: target State.copy_from! cost or verifier implementation changes
**note:** Keep `chunk-inplace` as the default state strategy. Removing a small target backup copy is not enough when accepted cycles become more expensive.

### [LM-codex-SPEC-INITIAL-GAMMA-FALSIFIER-1] Initial gamma 5/6 is worse than gamma 4
**status:** verified-falsifier
**trust:** {F:0.78, G:0.50, R:0.76}
**context:** ml (Qwen speculative scheduler tuning)
**evidence:**
- claim: "Initial `gamma=5` looked plausible because `def fibonacci(n):` accepts four draft tokens before rejection, but it is slower: it verifies one extra rejected row in the first cycle and regresses from `~21.0-21.5 ms/tok` at `gamma=4` to `~22.6-22.7 ms/tok`."
  source: interleaved `/tmp/qwen35_speculative_accept_gamma --tokens 64 --gamma 4|5|6 --max-gamma 32 "def fibonacci(n):"` on 2026-04-24
  verified_at: 2026-04-24
  decay_trigger: draft acceptance pattern, verifier cost, or adaptive fallback logic changes
- claim: "Initial `gamma=6` is worse: it causes many low-gamma cycles and draft resyncs on `def fibonacci(n):` and `The quick brown fox`, measuring about `24.7-28.2 ms/tok` instead of the `gamma=4` fallback path."
  source: same interleaved `/tmp/qwen35_speculative_accept_gamma` run on 2026-04-24
  verified_at: 2026-04-24
  decay_trigger: draft model, target model, or adaptive scheduler changes
- claim: "High-accept prompt `The capital of France is` also prefers `gamma=4`: `gamma=4` measured about `15.75/16.11 ms/tok`, while `gamma=5` measured `16.03/16.63 ms/tok` and `gamma=6` measured `16.44/16.83 ms/tok`."
  source: same interleaved `/tmp/qwen35_speculative_accept_gamma` run on 2026-04-24
  verified_at: 2026-04-24
  decay_trigger: host load, verifier route, or fast-regrow schedule changes
**note:** Keep initial `gamma=4`. The successful scheduler change is fast regrowth after gamma 8, not a larger first chunk.

### [LM-codex-HEAD-TOP1-ROWS-MIN-FALSIFIER-1] Batched lm-head top1 is slower even only for large chunks
**status:** verified-falsifier
**trust:** {F:0.76, G:0.44, R:0.74}
**context:** ml (Qwen speculative verifier output head)
**evidence:**
- claim: "A temporary branch added `QWEN35_HEAD_TOP1_ROWS_MIN` so the batched lm-head top1 route was used only for verifier chunks at or above thresholds `8`, `16`, `24`, or `32`, leaving small reject chunks on the per-row fused top1 path."
  source: temporary `src/ml/gguf/qwen35_cpu.cr` branch on 2026-04-24
  verified_at: 2026-04-24
  decay_trigger: output head kernels, verifier chunk schedule, or Q6_K lm-head route changes
- claim: "Even at large thresholds, the batched route regressed high-accept exact speculative verification. Default per-row fused top1 measured target_verify about `552-556 ms`; thresholds `8/16` measured `~662-675 ms`, and thresholds `24/32` measured `~627 ms`. Wall time moved from about `15.8-16.2 ms/tok` to `17.2-17.8 ms/tok`."
  source: `/tmp/qwen35_speculative_accept_headrowsmin --tokens 64 --gamma 4 --max-gamma 32 "The capital of France is"`, varying `QWEN35_HEAD_TOP1_ROWS_MIN`, on 2026-04-24
  verified_at: 2026-04-24
  decay_trigger: host load, output head kernel, or speculative gamma schedule changes
**note:** Keep `QWEN35_HEAD_TOP1_ROWS` default-off. The current per-row fused top1 remains better than the available row-batched top1, even when restricted to gamma 16/32 verifier chunks.

### [LM-codex-Q4-PAIR-B64-REFRESH-1] Q4 pair H16 at b64 remains neutral/noisy
**status:** verified-falsifier
**trust:** {F:0.68, G:0.36, R:0.62}
**context:** ml (Qwen prefill Q4_K FFN gate/up optimization)
**evidence:**
- claim: "A refreshed temporary branch lowered `Q4_PAIR_H16_MIN_BATCH` from `256` to `64`, activating the shared H16 input conversion for Q4_K FFN gate/up at pp64. The route changed conversion attribution from `prefill.rec.ffn_upgate f32_to_f16 q4_gemm_input` 48 calls / `72 MiB` to `q4_pair_input` 24 calls / `36 MiB`."
  source: temporary `src/ml/gguf/qwen35_metal.cr` branch and `/tmp/qwen35_prefill_attribution_q4pair64 --prompt=64 --warmup=1 --reps=3` on 2026-04-24
  verified_at: 2026-04-24
  decay_trigger: Q4 pair route, conversion kernels, or prefill attribution harness changes
- claim: "Despite lower conversion accounting, pp64 wall remained neutral/noisy under host load: current default measured p50 `156.27` and `156.55 ms`; b64 pair branch measured p50 `156.30` and `156.09 ms`. This is too small and noisy to promote, especially given earlier stronger pp64 pair regressions."
  source: interleaved old binary `/tmp/qwen35_prefill_attribution_current2` vs temporary `/tmp/qwen35_prefill_attribution_q4pair64` on 2026-04-24
  verified_at: 2026-04-24
  decay_trigger: host load, Q4 pair kernel, or prefill chunk schedule changes
**note:** Keep `Q4_PAIR_H16_MIN_BATCH=256`. Pair conversion reuse remains useful for larger prompts but not robust enough for pp64 default.

### [LM-codex-Q5-SINGLE-BUFFER-FALSIFIER-1] Q5_K GEMM should stay double-buffered
**status:** verified-falsifier
**trust:** {F:0.78, G:0.46, R:0.75}
**context:** ml (Qwen prefill Q5_K GEMM kernel optimization)
**evidence:**
- claim: "A temporary opt-in branch added `simd_mm_q5k_single`, mirroring the Q6_K single-buffer schedule, and routed Q5_K batch GEMM through it with `QWEN35_Q5_SINGLE_GEMM=1`. The hypothesis was that Q5's extra prefetch state/register pressure might outweigh the saved barrier in the current double-buffered Q5 kernel."
  source: temporary `src/ml/gguf/kernels/gemm_mm.metal` and `src/ml/gguf/qwen35_metal.cr` branch on 2026-04-24
  verified_at: 2026-04-24
  decay_trigger: Q5_K GEMM kernel rewrite, Metal compiler change, or recurrent qkv route change
- claim: "Correctness was preserved under the opt-in branch: `QWEN35_Q5_SINGLE_GEMM=1 crystal spec spec/qwen35_metal_spec.cr spec/qwen35_forward_spec.cr --link-flags=...` passed `22 examples, 0 failures`; Q5 batch GEMM kept cosine `1.0` and max delta `0.0033721924`."
  source: `CRYSTAL_CACHE_DIR=/tmp/cogni_ml_crystal_cache_q5single_spec QWEN35_Q5_SINGLE_GEMM=1 crystal spec spec/qwen35_metal_spec.cr spec/qwen35_forward_spec.cr --link-flags="$(pwd)/build/bridge.o -framework Metal -framework Foundation -lc++"` on 2026-04-24
  verified_at: 2026-04-24
  decay_trigger: Q5_K kernel or Qwen35 correctness fixtures change
- claim: "The opt-in branch was slower in paired pp64 prefill A/B: current double-buffer default measured avg `159.40 ms`, p50 `159.99 ms`; single-buffer measured avg `160.51 ms`, p50 `161.55 ms`; default won `7/8` pairs. Standalone attribution also showed Q5 `4096x8192 b64` p50 worsening from `3.911 ms` to `4.316 ms`."
  source: `CRYSTAL_CACHE_DIR=/tmp/cogni_ml_crystal_cache_q5single_ab crystal run --release --link-flags=... bin/qwen35_prefill_attribution.cr -- --prompt=64 --warmup=2 --reps=8 --compare-env=QWEN35_Q5_SINGLE_GEMM --compare-off=1`, plus `qwen35_op_attribution` on 2026-04-24
  verified_at: 2026-04-24
  decay_trigger: host load, Q5_K GEMM schedule, or benchmark harness change
**note:** Keep the current double-buffered Q5_K GEMM. This closes the inverse of the earlier Q6 double-buffer falsifier: Q5 and Q6 prefer different schedules in the current kernels.

### [LM-codex-SPEC-SKIP-KV-BACKUP-FALSIFIER-1] Speculative target rollback is not KV-copy bound
**status:** verified-falsifier
**trust:** {F:0.74, G:0.42, R:0.72}
**context:** ml (Qwen speculative verifier rollback)
**evidence:**
- claim: "A temporary branch added recurrent-only target backup/restore for `chunk-inplace` speculative verification, leaving full-attention KV rows written after the cycle start in place because causal attention ignores future rows and later corrections overwrite them."
  source: temporary `copy_recurrent_from!` branch in `src/ml/gguf/qwen35_cpu.cr` and `bin/qwen35_speculative_accept.cr` on 2026-04-24
  verified_at: 2026-04-24
  decay_trigger: attention cache semantics, rollback policy, or verifier state model changes
- claim: "The shortcut preserved exactness in a focused rollback spec: a rejected speculative span followed by recurrent-only restore and correction matched full state restore on top1 and logit."
  source: `CRYSTAL_CACHE_DIR=/tmp/cogni_ml_crystal_cache_skipkv_spec crystal spec spec/qwen35_forward_spec.cr --link-flags=...` -> `14 examples, 0 failures` on 2026-04-24
  verified_at: 2026-04-24
  decay_trigger: Qwen35 state layout or verifier correctness fixtures change
- claim: "The shortcut was not a speed win on the high-accept path. `The capital of France is`, tokens=64/gamma=4/max_gamma=32 measured skip-KV default `16.30 ms/tok` and full backup `16.20 ms/tok`; `target_backup` stayed `10.6 ms` in both runs."
  source: `bin/qwen35_speculative_accept.cr -- --tokens 64 --gamma 4 --max-gamma 32 "The capital of France is"`, with and without `QWEN35_SPEC_TARGET_BACKUP_KV_OFF=1`, on 2026-04-24
  verified_at: 2026-04-24
  decay_trigger: host load, state buffer sizes, or backup implementation changes
**note:** Do not spend more time on KV-only rollback copies for current short-context speculative decode. The remaining rollback cost is recurrent conv/SSM state, while the larger wall remains target verification plus draft generation.

### [LM-codex-Q8-SHARED-X-FALSIFIER-1] Q8_0 GEMV is not improved by threadgroup x staging
**status:** verified-falsifier
**trust:** {F:0.78, G:0.44, R:0.76}
**context:** ml (Qwen 0.8B Q8_0 draft GEMV optimization)
**evidence:**
- claim: "A temporary opt-in branch added `simd_mv_q8_0_shared_x_f32`, loading each token activation vector into threadgroup memory once per threadgroup so the four Q8 simdgroups could share x reads."
  source: temporary `src/ml/gguf/kernels/gemm_q56k.metal` and `src/ml/gguf/qwen35_metal.cr` branch on 2026-04-24
  verified_at: 2026-04-24
  decay_trigger: Q8_0 GEMV kernel rewrite, threadgroup memory behavior, or M2 Max compiler change
- claim: "The shared-X branch was numerically exact in focused specs: `QWEN35_Q8_SHARED_X_GEMV=1 crystal spec spec/qwen35_metal_spec.cr spec/qwen35_forward_spec.cr --link-flags=...` passed `22 examples, 0 failures`; Q8_0 GEMV cosine stayed `1.0`."
  source: `CRYSTAL_CACHE_DIR=/tmp/cogni_ml_crystal_cache_q8shared_spec QWEN35_Q8_SHARED_X_GEMV=1 crystal spec spec/qwen35_metal_spec.cr spec/qwen35_forward_spec.cr --link-flags=...` on 2026-04-24
  verified_at: 2026-04-24
  decay_trigger: Q8_0 spec fixtures or GEMV kernel change
- claim: "The branch was substantially slower. Focused Q8_0 GEMV spec time worsened from roughly `1.9 ms` current to `24.3 ms` with shared-X, and Qwen3.5 0.8B Q8_0 draft sync profile regressed from `528.4 ms` / `8.26 ms/tok` current to `587.5 ms` / `9.18 ms/tok` with shared-X."
  source: `QWEN35_PROFILE_TOP1=1 bin/qwen35_sync_profile.cr -- --model Qwen3.5-0.8B-Q8_0.gguf 64 64`, with and without `QWEN35_Q8_SHARED_X_GEMV=1`, on 2026-04-24
  verified_at: 2026-04-24
  decay_trigger: host load, Q8_0 launch shape, or Metal compiler change
**note:** Do not retry simple threadgroup staging for Q8_0 x reads. The added load loop/barrier costs more than repeated cached x loads; future Q8 gains need a different dot-product algorithm or fewer GEMV calls.

### [LM-codex-SPEC-GAMMA-AGGRESSIVE-FALSIFIER-1] Aggressive gamma regrowth is not a safe default
**status:** verified-falsifier
**trust:** {F:0.70, G:0.42, R:0.68}
**context:** ml (Qwen speculative scheduler tuning)
**evidence:**
- claim: "A schedule sweep tested more aggressive exact speculative growth policies against the current conservative default: `QWEN35_SPEC_FULL_ACCEPT_STREAK=1`, `QWEN35_SPEC_FAST_REGROW_MIN_GAMMA=4`, and initial `--gamma 8`, all with `--tokens 64 --max-gamma 32`."
  source: `bin/qwen35_speculative_accept.cr` schedule sweep on 2026-04-24
  verified_at: 2026-04-24
  decay_trigger: speculative verifier implementation, draft speed, target verifier batch path, or gamma policy changes
- claim: "On the 100%-accept prompt `The capital of France is`, aggressive policies were neutral or slower than default: default `16.44 ms/tok`, streak1 `16.93 ms/tok`, regrow4 `16.78 ms/tok`, gamma8 `16.54 ms/tok`."
  source: `crystal run --release bin/qwen35_speculative_accept.cr -- --tokens 64 --gamma 4/8 --max-gamma 32 "The capital of France is"` with schedule env variants on 2026-04-24
  verified_at: 2026-04-24
  decay_trigger: host load, gamma schedule, or target verifier route changes
- claim: "On rejection-sensitive `def fibonacci(n):`, all aggressive policies regressed versus default: default `22.98 ms/tok`, streak1 `24.97 ms/tok`, regrow4 `24.83 ms/tok`, gamma8 `24.75 ms/tok`."
  source: `crystal run --release bin/qwen35_speculative_accept.cr -- --tokens 64 --gamma 4/8 --max-gamma 32 "def fibonacci(n):"` with schedule env variants on 2026-04-24
  verified_at: 2026-04-24
  decay_trigger: host load, gamma schedule, or early-reject/fallback policy changes
- claim: "After exact row-batched top1 was promoted for large verifier chunks, re-enabling adaptive regrow remained a strong regression. `def fibonacci(n):` moved from default `23.77 ms/tok` to `48.00 ms/tok` with `QWEN35_SPEC_ADAPTIVE_REGROW=1`, because draft/resync costs returned (`draft=820.9 ms`, `draft_resync=192.8 ms`)."
  source: `/tmp/qwen35_spec_ab_current --tokens 64 --gamma 4 --max-gamma 32 --verify chunk-inplace "def fibonacci(n):"` with and without `QWEN35_SPEC_ADAPTIVE_REGROW=1` on 2026-04-24
  verified_at: 2026-04-24
  decay_trigger: draft model speed, target verifier cost, or rejection fallback policy changes
**note:** Keep the current default schedule. The next exact win is unlikely to come from more schedule aggressiveness; it needs lower target verifier cost or a true batched verifier path.

### [LM-codex-SPEC-STAGED-VERIFY-1] Staged verifier is useful only as high-accept opt-in turbo
**status:** verified-feature-with-caveat
**trust:** {F:0.76, G:0.38, R:0.72}
**context:** ml (Qwen speculative verifier scheduling)
**evidence:**
- claim: "Added `--verify staged` and `--stage-gate N` to the speculative harness. The mode drafts and verifies a small guard stage before drafting/verifying the rest of a large gamma cycle, preserving exact greedy output while reducing high-accept cycle count."
  source: `bin/qwen35_speculative_accept.cr` on 2026-04-24
  verified_at: 2026-04-24
  decay_trigger: speculative harness control flow, target state rollback, or draft resync changes
- claim: "Correctness and build hygiene passed: `crystal build --release bin/qwen35_speculative_accept.cr` succeeded, a staged smoke with `--tokens 16 --gamma 8 --verify staged --stage-gate 4` matched target greedy output, and focused Qwen forward specs passed `13 examples, 0 failures`."
  source: `crystal build --release --link-flags=... bin/qwen35_speculative_accept.cr -o /tmp/qwen35_speculative_accept_staged`; `/tmp/qwen35_speculative_accept_staged --tokens 16 --gamma 8 --verify staged --stage-gate 4 "The capital of France is"`; `crystal spec spec/qwen35_forward_spec.cr --link-flags=...` on 2026-04-24
  verified_at: 2026-04-24
  decay_trigger: Qwen forward specs, tokenizer behavior, or speculative exactness check changes
- claim: "As a fixed-gamma high-accept opt-in, staged mode improves the 64-token `The capital of France is` smoke from default `16.45 ms/tok` to `15.04 ms/tok` with `--gamma 32 --verify staged --stage-gate 4`; target verification drops from `571.3 ms` to `476.3 ms`."
  source: `/tmp/qwen35_speculative_accept_staged --tokens 64 --gamma 4 --max-gamma 32 --verify chunk-inplace "The capital of France is"` vs `/tmp/qwen35_speculative_accept_staged --tokens 64 --gamma 32 --max-gamma 32 --verify staged --stage-gate 4 "The capital of France is"` on 2026-04-24
  verified_at: 2026-04-24
  decay_trigger: host load, gamma schedule, draft model speed, or target verifier implementation changes
- claim: "The same fixed-gamma staged mode is not safe as default: on `def fibonacci(n):`, it regresses from default `23.82 ms/tok` to `32.58 ms/tok` because acceptance drops and draft resync dominates."
  source: same `/tmp/qwen35_speculative_accept_staged` A/B on `def fibonacci(n):` on 2026-04-24
  verified_at: 2026-04-24
  decay_trigger: prompt distribution, rejection fallback policy, or staged guard policy changes
- claim: "After row-batched top1 promotion, adaptive staged mode with larger gates was not a default win either: high-accept `The capital of France is` regressed from chunk-inplace `18.76 ms/tok` to staged gate8 `26.86` and gate16 `21.79`, while rejection-heavy prompts were neutral/slower."
  source: `/tmp/qwen35_spec_ab_current --tokens 64 --gamma 4 --max-gamma 32 --verify chunk-inplace|staged --stage-gate 8|16` prompt suite on 2026-04-24
  verified_at: 2026-04-24
  decay_trigger: verifier route, row-batched top1 threshold, staged policy, or host load changes
**note:** Keep staged verifier opt-in. It is an exact high-accept turbo lever, but default scheduling still needs a prompt-safe acceptance predictor or cheaper verifier before using gamma 32 broadly.

### [LM-codex-SPEC-BOOTSTRAP-GAMMA-1] Bootstrap gamma jump is an opt-in high-accept turbo
**status:** verified-feature-with-caveat
**trust:** {F:0.74, G:0.34, R:0.70}
**context:** ml (Qwen speculative scheduler tuning)
**evidence:**
- claim: "Added `--bootstrap-gamma N` / `QWEN35_SPEC_BOOTSTRAP_GAMMA` as an exact scheduler knob: after one fully accepted initial chunk, adaptive decode can jump directly to a larger gamma instead of waiting for the conservative two-cycle growth."
  source: `bin/qwen35_speculative_accept.cr` on 2026-04-24
  verified_at: 2026-04-24
  decay_trigger: speculative scheduler, acceptance policy, or CLI/env option changes
- claim: "On the 100%-accept prompt `The capital of France is`, bootstrap32 reduced verifier cycles from `5` to `3` and improved exact speculative decode from default `17.89 ms/tok` to `15.10 ms/tok`; target verification dropped from `621.8 ms` to `455.5 ms`."
  source: `/tmp/qwen35_spec_bootstrap --tokens 64 --gamma 4 --max-gamma 32 --verify chunk-inplace "The capital of France is"` with and without `QWEN35_SPEC_BOOTSTRAP_GAMMA=32` on 2026-04-24
  verified_at: 2026-04-24
  decay_trigger: host load, draft model speed, verifier chunk route, or gamma policy changes
- claim: "Bootstrap is not a safe default: `def fibonacci(n):` regressed from default `23.36 ms/tok` to bootstrap16 `28.83` and bootstrap32 `29.93`, because an initially accepted prefix was not enough to predict continued high acceptance."
  source: same `/tmp/qwen35_spec_bootstrap` A/B on `def fibonacci(n):` on 2026-04-24
  verified_at: 2026-04-24
  decay_trigger: prompt distribution, acceptance predictor, or fallback policy changes
**note:** Keep bootstrap default-off. It is useful for controlled high-accept/repetitive workloads, but default promotion needs a stronger predictor than "first gamma=4 chunk fully accepted".

### [LM-codex-PREFILL-PP2048-ATTR-1] Qwen35 pp2048 prefill is recurrent-scan bound, not full-attention bound
**status:** verified-falsifier
**trust:** {F:0.78, G:0.46, R:0.72}
**context:** ml (Qwen prefill optimization)
**evidence:**
- claim: "A pp2048 prefill attribution run showed full-attention is not currently the dominant long-prompt wall: attention reported `26.35 ms` wait inside a `10170.50 ms` profiled wall."
  source: `/tmp/qwen35_prefill_attr_current --prompt 2048 --warmup 0 --reps 1 --load-warning-threshold=0 --load-total-warning-threshold=0` on 2026-04-24
  verified_at: 2026-04-24
  decay_trigger: full-attention implementation, recurrent prefill implementation, profiling labels, or prompt length changes
- claim: "The material pp2048 wall is recurrent grouped prefill / DeltaNet serial work: `dn` reported `8` calls with `9991.69 ms` wait and grouped command buffers each spent about `0.87-1.56 s` waiting."
  source: same pp2048 attribution run on 2026-04-24
  verified_at: 2026-04-24
  decay_trigger: DeltaNet chunk kernel, recurrent scan algorithm, or grouped command-buffer composition changes
- claim: "Logical conversion traffic at pp2048 is also large: `10848.00 MiB` conversion traffic versus `3992.53 MiB` logical matmul weights, so activation f32->f16 conversion pressure is a secondary exact target."
  source: same pp2048 attribution run on 2026-04-24
  verified_at: 2026-04-24
  decay_trigger: GEMM input format, fused conversion kernels, or profile accounting changes
**note:** Defer FlashAttention as a performance project until recurrent scan and conversion pressure are reduced. The next long-prefill breakthrough is parallel/chunked DeltaNet scan or an exact recurrent WBA diamond, not attention.

### [LM-codex-QWEN-LLAMA-BENCH-20260424-1] Current Qwen35 native path beats llama.cpp on pp64 prefill and matches decode
**status:** verified-benchmark
**trust:** {F:0.76, G:0.42, R:0.70}
**context:** ml (Qwen performance benchmark)
**evidence:**
- claim: "On the current M2 Max run, native pp64 prefill with preallocated prompt state beat llama.cpp by `+35.53%`: cogni-ml p50 `335.25 tok/s` versus llama.cpp avg `247.37 tok/s`."
  source: `/tmp/benchmark_qwen_vs_llama_current --prompt=64 --gen=64 --reps=3 --warmup=1 --native-prefill-prealloc --load-warning-threshold=0 --load-total-warning-threshold=0` on 2026-04-24
  verified_at: 2026-04-24
  decay_trigger: host load, llama.cpp build/model, native prefill route, or benchmark harness changes
- claim: "Decode is currently parity/slightly ahead in the same run: cogni-ml p50 `16.32 tok/s` versus llama.cpp avg `16.15 tok/s`, a `+1.03%` gap."
  source: same benchmark run on 2026-04-24
  verified_at: 2026-04-24
  decay_trigger: host load, decode wave route, speculative mode, llama.cpp build/model, or benchmark harness changes
**note:** Treat this as current local evidence, not a stable public claim. Rerun under quiet-load conditions before publishing; the next decode target is a meaningful margin beyond parity, not another small scheduler tweak.

### [LM-codex-Q6-TOP1-ROWS16-FALSIFIER-1] Q6 top1 rows16 does not improve verifier cost
**status:** verified-falsifier
**trust:** {F:0.72, G:0.36, R:0.68}
**context:** ml (Qwen Q6 lm-head top1 kernel retune)
**evidence:**
- claim: "A temporary branch changed `HEAD_TOP1_ROWS_PER_TG` and `MV6_TOP_ROWS_PER_TG` from `12` to `16` to reduce Q6 lm-head top1 tile count."
  source: temporary `src/ml/gguf/qwen35_metal.cr` and `src/ml/gguf/kernels/gemm_q56k.metal` branch on 2026-04-24
  verified_at: 2026-04-24
  decay_trigger: output-head top1 kernel rewrite or Metal compiler change
- claim: "Rows16 preserved the strict Qwen top-token gate: focused `spec/qwen35_forward_spec.cr` passed `13 examples, 0 failures`, top token `198`, logit `11.423702`."
  source: `CRYSTAL_CACHE_DIR=/tmp/cogni_ml_crystal_cache_head16_spec crystal spec spec/qwen35_forward_spec.cr --link-flags=...` on 2026-04-24
  verified_at: 2026-04-24
  decay_trigger: Qwen forward specs or output-head route changes
- claim: "Rows16 did not lower verifier target time. On high-accept default chunk-inplace, target verification stayed `573.8 ms` for rows12 and rows16; staged high-accept moved from `477.8 ms` rows12 to `479.9 ms` rows16."
  source: `/tmp/qwen35_speculative_accept_staged` rows12 vs `/tmp/qwen35_speculative_accept_head16` rows16 on `The capital of France is`, 64 tokens, on 2026-04-24
  verified_at: 2026-04-24
  decay_trigger: host load, output-head top1 kernel, or speculative verifier schedule changes
- claim: "Rows16 regressed rejection-sensitive default smoke: `def fibonacci(n):` default chunk-inplace moved from `22.84 ms/tok` rows12 to `23.45 ms/tok` rows16."
  source: same `/tmp/qwen35_speculative_accept_staged` rows12 vs `/tmp/qwen35_speculative_accept_head16` rows16 A/B on 2026-04-24
  verified_at: 2026-04-24
  decay_trigger: host load, output-head top1 kernel, or speculative verifier schedule changes
**note:** Keep Q6 top1 tile rows at 12. Larger tiles reduce tile count but do not reduce the measured verifier bottleneck in this implementation.

### [LM-codex-HEAD-FULL-ROWS-VERIFY-1] Batched full lm-head is faster for verifier chunks
**status:** refuted as default; opt-in only
**trust:** {F:0.88, G:0.62, R:0.86}
**context:** ml (Qwen speculative target verifier)
**evidence:**
- claim: "llama.cpp speculative-simple evaluates the target model on `[id_last, draft0, ..., draftN-1]` as one batch and samples from those batched logits; our verifier body was already chunked, but its lm-head path emitted per-row fused top1."
  source: `/Users/sergey/SrcArchives/AI/llama.cpp/examples/speculative-simple/speculative-simple.cpp` lines around target batch construction, inspected on 2026-04-24
  verified_at: 2026-04-24
  decay_trigger: llama.cpp speculative implementation changes or local verifier control-flow changes
- claim: "Full-row Q6 lm-head verifier is fast but not exact against the greedy target route because it converts activations/logits through F16; an all-row parity check found row24 mismatch at rows32/64: exact per-row `220@8.900501` vs full-row F16 `198@8.8984375`."
  source: `build/tmp/qwen35_headfull_all_rows_correctness.cr` with `ROWS=32/64`, comparing all rows against `QWEN35_HEAD_FULL_ROWS_OFF=1`, on 2026-04-24
  verified_at: 2026-04-24
  decay_trigger: output-head full-row route, activation precision, or verifier exactness policy changes
- claim: "Default verifier was restored to exact per-row fused top1; full-row F16 verifier is now opt-in via `QWEN35_HEAD_FULL_ROWS=1`, and a rows32 ambiguous regression spec keeps default `prefill_tokens_top1s` aligned with serial greedy target steps."
  source: `src/ml/gguf/qwen35_cpu.cr` and `spec/qwen35_forward_spec.cr` on 2026-04-24
  verified_at: 2026-04-24
  decay_trigger: verifier routing env gates or top1 regression spec changes
- claim: "The previous speed numbers remain useful only as an approximate/opt-in upper-bound: rows64 p50 improved to `165.69 ms` and staged high-accept reached `12.26 ms/tok`, but those figures must not be labeled exact until an exact guard or F32-equivalent batched lm-head exists."
  source: earlier `build/tmp/qwen35_headfull_default_ab.cr` and `/tmp/qwen35_speculative_accept_headgpu ... --verify staged` runs on 2026-04-24, downgraded after all-row parity falsifier
  verified_at: 2026-04-24
  decay_trigger: exact full-row verifier implementation or margin-guard validation
- claim: "A follow-up attempt to switch the full-row route to f32 output plus f32 top1 reduce did not rescue exactness. Focused specs still passed (`17 examples, 0 failures`), but the live speculative harness diverged on `The capital of France is` at token 17 (`13` in plain target greedy vs `32` in the f32-output full-row route), so the temporary code was removed."
  source: `QWEN35_HEAD_FULL_ROWS=1 /tmp/qwen35_spec_fullrows_f32_fixed --tokens 64 --gamma 4 --max-gamma 32 --verify chunk-inplace 'The capital of France is'` on 2026-04-24
  verified_at: 2026-04-24
  decay_trigger: exact full-row verifier implementation, f32 activation input path, or margin-guard validation
**note:** The useful next direction is not default-on F16 batched logits; it is exact guarded batching: compute fast batched candidates, detect close top1/top2 margins, and fall back to exact F32 per-row top1 for ambiguous rows, or build a batched lm-head that preserves the greedy route's precision enough to pass all-row parity.

### [LM-codex-Q56-PREFILL-F32OUT-1] Q5/Q6 prefill GEMM can skip the f16-output conversion dispatch
**status:** verified-feature-with-caveat
**trust:** {F:0.82, G:0.52, R:0.76}
**context:** ml (Qwen prefill Metal GEMM)
**evidence:**
- claim: "Added `simd_mm_q5k_f32out` and `simd_mm_q6k_f32out` for Qwen prefill. They keep the existing F16 activation input and explicitly round the accumulator through `half` before writing f32 output, preserving the prior half-output numeric contract while removing the separate `f16_to_f32` conversion dispatch."
  source: `src/ml/gguf/kernels/gemm_mm.metal` and `src/ml/gguf/qwen35_metal.cr` on 2026-04-24
  verified_at: 2026-04-24
  decay_trigger: Q5/Q6 GEMM precision policy, prefill output format, or Metal kernel routing changes
- claim: "Focused Qwen specs passed after the kernel change, including forward and Metal coverage: `23 examples, 0 failures`."
  source: `CRYSTAL_CACHE_DIR=/tmp/cogni_ml_crystal_cache_q56f32_fullspec crystal spec spec/qwen35_forward_spec.cr spec/qwen35_metal_spec.cr --link-flags=...` on 2026-04-24
  verified_at: 2026-04-24
  decay_trigger: spec fixtures, quant kernels, or Qwen forward routing changes
- claim: "Standalone pp64 op attribution lower bound improved materially: total weighted measured hot-shape time dropped from the refreshed baseline `8.209 ms` to `2.913 ms`; Q5_K 4096x8192 dropped from `4.186 ms` p50 to `0.421 ms`, and Q6_K 12288x4096 from `3.733 ms` to `0.779 ms`."
  source: `/tmp/qwen35_op_q56f32 --batch=64 --warmup=3 --runs=9 --limit=12` compared with the earlier same-session refreshed baseline on 2026-04-24
  verified_at: 2026-04-24
  decay_trigger: op attribution harness, host load, or kernel pipeline changes
- claim: "End-to-end prefill wall did not yet move by the same amount under current host noise/scheduling: pp64 measured around `152.66-153.82 ms` and pp256 around `494.02-496.99 ms`, so the remaining >10% llama.cpp gap is not solved by this dispatch-elimination alone."
  source: `/tmp/qwen35_prefill_q4h16vec --prompt=64 --warmup=2 --reps=8` and `--prompt=256 --warmup=1 --reps=4` on 2026-04-24
  verified_at: 2026-04-24
  decay_trigger: quiet-system benchmark rerun, prefill grouping, or Metal scheduler behavior changes
**note:** Keep the exact f32-output Q5/Q6 kernels, but do not count this as the pp64 breakthrough. The next prefill step should attack Q4_K 4096x12288/12288x4096 traffic or eliminate materialized FFN intermediates; this change mainly removes a provable conversion dispatch and improves isolated operator lower bounds.

### [LM-codex-DRAFT-LAYER-TRUNC-REFUTE-1] Truncating the 0.8B Q8_0 draft collapses speculative acceptance
**status:** verified-falsifier
**trust:** {F:0.74, G:0.48, R:0.70}
**context:** ml (Qwen speculative decode, approximate draft)
**evidence:**
- claim: "A bounded temporary branch added a draft-only `--draft-layers N` probe. The target verifier remained exact, so any accepted speedup would have been a valid speculative-decoding win; the only question was draft acceptance versus draft cost."
  source: temporary local patch to `bin/qwen35_speculative_accept.cr` and `src/ml/gguf/qwen35_cpu.cr` on 2026-04-24
  verified_at: 2026-04-24
  decay_trigger: draft model architecture, speculative harness, or draft approximation method changes
- claim: "The downloaded Qwen3.5 0.8B Q8_0 draft has `24` layers in this harness. Requests for `--draft-layers 24`, `26`, or `28` normalize to the full draft and therefore are not truncation experiments."
  source: `/tmp/qwen35_speculative_accept_draft_limited --draft-layers 24/26/28 --tokens 64 ...` prompt line reported `draft_layers=full` on 2026-04-24
  verified_at: 2026-04-24
  decay_trigger: model file or harness layer-limit normalization changes
- claim: "The first real truncation tested, `--draft-layers 22`, destroyed the high-accept case: `The capital of France is` moved from full-draft `100.0%` acceptance and about `15.55 ms/tok` to `0.0%` acceptance, one immediate reject, and about `20.63 ms/tok` target-fallback behavior."
  source: `/tmp/qwen35_speculative_accept_draft_limited --tokens 64 'The capital of France is'` versus `--draft-layers 22` on 2026-04-24
  verified_at: 2026-04-24
  decay_trigger: prompt distribution, draft model, or fallback policy changes
- claim: "The same pattern held on partial-reject probes: `Once upon a time` and `The quick brown fox` dropped from `75%` first-cycle acceptance with the full draft to `0%` at 22 layers, and `def fibonacci(n):` dropped from `80%` to `25%`."
  source: same temporary `/tmp/qwen35_speculative_accept_draft_limited` grid over four prompts on 2026-04-24
  verified_at: 2026-04-24
  decay_trigger: prompt distribution, draft model, or draft approximation method changes
- claim: "No code was retained after the falsifier; `rg -n \"draft-layers|DRAFT_LAYERS|forward_top1_limited\" bin/qwen35_speculative_accept.cr src/ml/gguf/qwen35_cpu.cr` returns no matches."
  source: local source tree after reverting the temporary branch on 2026-04-24
  verified_at: 2026-04-24
  decay_trigger: future draft approximation feature work
**note:** Approximate drafts are still a valid exact-speculative paradigm, but layer truncation is too destructive for this 0.8B model. Future work should try a separately quantized smaller/faster full draft, n-gram/cache draft, or trained early-exit/draft head rather than naively skipping transformer layers.

### [LM-codex-TOP1-ROWS-MIN4-REFUTE-1] Lowering verifier row-batch threshold to 4 is not a safe default
**status:** refuted
**trust:** {F:0.62, G:0.34, R:0.56}
**context:** ml (Qwen speculative target verifier)
**evidence:**
- claim: "A bounded env-only smoke tested `QWEN35_HEAD_TOP1_ROWS_MIN=4` against the current default threshold `8`, with no source changes."
  source: `/tmp/qwen35_speculative_accept_probe --tokens 64 ...` with and without `QWEN35_HEAD_TOP1_ROWS_MIN=4` on 2026-04-24
  verified_at: 2026-04-24
  decay_trigger: host load, output-head top1 kernel, or speculative verifier schedule changes
- claim: "The four-prompt run was noisy but adversarially mixed: `def fibonacci(n):` improved in that run (`32.61 -> 30.91 ms/tok`), while high-accept `The capital of France is` regressed (`19.58 -> 22.60 ms/tok`) and `The quick brown fox` regressed (`31.08 -> 33.63 ms/tok`)."
  source: same env-only smoke on 2026-04-24
  verified_at: 2026-04-24
  decay_trigger: quiet rerun, changed gamma policy, or top1 rows kernel rewrite
**note:** Keep `QWEN35_HEAD_TOP1_ROWS_MIN=8`. A lower threshold can help a rejection-heavy first chunk under some load, but it is not robust enough to trade away high-accept speed.

### [LM-codex-NGRAM-SPEC-HARNESS-1] N-gram/cache draft is an exact speculative path for repeated text
**status:** verified-feature-with-caveat
**trust:** {F:0.78, G:0.40, R:0.72}
**context:** ml (Qwen speculative decode, non-neural draft)
**evidence:**
- claim: "Added `bin/qwen35_ngram_speculative.cr`, an opt-in target-only speculative harness that uses repeated token suffixes as a zero-model-cost draft. It verifies candidates with the normal Qwen35 target path, and by default replays plain greedy target output to assert exact equality."
  source: `bin/qwen35_ngram_speculative.cr` on 2026-04-24
  verified_at: 2026-04-24
  decay_trigger: harness control flow, target verifier semantics, tokenizer behavior, or default n-gram gates change
- claim: "The harness is intentionally gated: default `min_ngram=6`, `max_ngram=8`, `gamma=16`. This avoids the earlier weak 1-gram/2-gram failure mode where random local repeats created expensive rejected verifier chunks."
  source: `bin/qwen35_ngram_speculative.cr` defaults and temporary min-ngram grid on 2026-04-24
  verified_at: 2026-04-24
  decay_trigger: prompt distribution or acceptance predictor changes
- claim: "On the high-repeat prompt `The capital of France is`, the default n-gram harness accepted `55/55` n-gram candidates with `7` verifier cycles and `9` plain target seed steps, measuring `1142.6 ms` / `17.85 ms/tok` in the smoke."
  source: `/tmp/qwen35_ngram_speculative2 --tokens 64 'The capital of France is'` on 2026-04-24
  verified_at: 2026-04-24
  decay_trigger: host load, target verifier speed, n-gram defaults, or prompt output changes
- claim: "On `def fibonacci(n):`, the default gate proposed no n-gram candidates (`0/0`, `cycles=0`) and fell back to `64` plain target steps, preserving exact output without spending verifier chunks on weak repeats."
  source: `/tmp/qwen35_ngram_speculative2 --tokens 64 'def fibonacci(n):'` on 2026-04-24
  verified_at: 2026-04-24
  decay_trigger: prompt output, n-gram defaults, or fallback policy changes
- claim: "The suffix matcher was extracted to reusable `ML::GGUF::NgramDraft` with focused specs for no-match, longest-suffix, weak-repeat rejection, and parameter validation."
  source: `src/ml/gguf/ngram_draft.cr`, `spec/ngram_draft_spec.cr`, and `CRYSTAL_CACHE_DIR=/tmp/cogni_ml_crystal_cache_ngram_spec crystal spec spec/ngram_draft_spec.cr` returning `4 examples, 0 failures` on 2026-04-24
  verified_at: 2026-04-24
  decay_trigger: n-gram candidate semantics or spec fixtures change
- claim: "Recursive n-gram extension lets the draft append accepted-looking candidates into scratch history and continue proposing up to `gamma=32`, while target verification still preserves exact greedy output."
  source: `src/ml/gguf/ngram_draft.cr`, `bin/qwen35_ngram_speculative.cr`, and `spec/ngram_draft_spec.cr` on 2026-04-24
  verified_at: 2026-04-24
  decay_trigger: recursive n-gram semantics or verifier control flow changes
- claim: "On `The capital of France is`, recursive n-gram extension with `gamma=32` reduced verifier cycles from non-recursive `7` to `2` and improved the smoke from about `17.46 ms/tok` to about `9.0 ms/tok`, with `55/55` candidates accepted and exact-check replay enabled."
  source: `/tmp/qwen35_ngram_speculative_rec --tokens 64 --gamma 32 --min-ngram 6 'The capital of France is'` with and without `--no-recursive-ngram` on 2026-04-24
  verified_at: 2026-04-24
  decay_trigger: host load, prompt output, n-gram defaults, or target verifier speed changes
- claim: "Adversary prompts did not show the weak-repeat failure mode with the default `min_ngram=6`: `def fibonacci(n):` and `Once upon a time` proposed no candidates, while `The quick brown fox` remained exact and near plain target speed."
  source: same `/tmp/qwen35_ngram_speculative_rec` adversary grid on 2026-04-24
  verified_at: 2026-04-24
  decay_trigger: prompt distribution, n-gram defaults, or acceptance policy changes
- claim: "The n-gram harness now disables further n-gram drafting after the first rejected n-gram chunk by default, with `--keep-ngram-after-reject` / `QWEN35_NGRAM_DISABLE_AFTER_REJECT_OFF=1` as the exploration escape hatch."
  source: `bin/qwen35_ngram_speculative.cr` and `/tmp/qwen35_ngram_disable` A/B on `The quick brown fox` and high-repeat/no-repeat prompts on 2026-04-24
  verified_at: 2026-04-24
  decay_trigger: n-gram rejection policy, prompt distribution, or target verifier behavior changes
- claim: "The main neural speculative harness now has an opt-in n-gram front-runner (`--ngram` / `QWEN35_SPEC_NGRAM=1`). It verifies n-gram chunks with the exact target path and lazy-resyncs the Q8_0 draft only if neural drafting is needed after n-gram-generated tokens."
  source: `bin/qwen35_speculative_accept.cr` on 2026-04-24
  verified_at: 2026-04-24
  decay_trigger: speculative harness control flow, draft state semantics, or n-gram option defaults change
- claim: "In the same built harness on `The capital of France is`, neural-only speculative measured `17.08 ms/tok`, while `--ngram` measured `11.24 ms/tok`; n-gram accepted `48/48` candidates in `2` n-gram cycles and left `48` draft tokens pending because no later neural resync was needed."
  source: `/tmp/qwen35_speculative_accept_ngram --tokens 64 'The capital of France is'` with and without `--ngram` on 2026-04-24
  verified_at: 2026-04-24
  decay_trigger: host load, prompt output, n-gram defaults, or neural speculative scheduler changes
- claim: "Adversary prompts stayed effectively unchanged under `--ngram`: `def fibonacci(n):` and `Once upon a time` proposed no n-gram candidates, while `The quick brown fox` used one n-gram chunk, disabled n-gram after rejection, and avoided repeated n-gram verifier waste."
  source: `/tmp/qwen35_speculative_accept_ngram --tokens 64 --ngram ...` adversary grid on 2026-04-24
  verified_at: 2026-04-24
  decay_trigger: prompt distribution, n-gram defaults, or fallback policy changes
- claim: "A target-only seed experiment before the first n-gram match was rejected. It improved high-repeat `The capital of France is` to about `8.85 ms/tok`, but `The quick brown fox` diverged from greedy at token 31; disabling row-batched top1 for the n-gram chunk did not fix the divergence."
  source: temporary local `bin/qwen35_speculative_accept.cr` branch and `/tmp/qwen35_speculative_accept_ngram_seed_exact --tokens 64 --ngram 'The quick brown fox'` on 2026-04-24
  verified_at: 2026-04-24
  decay_trigger: exact verifier implementation, n-gram staging policy, or target-seed algorithm changes
- claim: "No target-seed code was retained after the falsifier; the committed n-gram front-runner still waits for neural/normal generation to create a repeated suffix before proposing n-gram chunks."
  source: `git diff -- bin/qwen35_speculative_accept.cr` after reverting the temporary target-seed branch on 2026-04-24
  verified_at: 2026-04-24
  decay_trigger: future target-seed or staged n-gram verifier work
**note:** This is a paradigm shift rather than a universal decode win: it helps repeated/generated-template text and should eventually be composed with neural draft speculative decode as a cheap first-choice draft, but it is not a replacement for a faster learned draft on arbitrary prompts.

### [LM-codex-NGRAM-GENERATE-CLI-1] Practical generation CLI has opt-in exact n-gram speculative decode
**status:** verified-feature-with-caveat
**trust:** {F:0.78, G:0.38, R:0.74}
**context:** ml (Qwen decode, generation CLI, non-neural draft)
**evidence:**
- claim: "`bin/qwen35_generate.cr` now has an opt-in `QWEN35_NGRAM_DECODE=1` path that uses `ML::GGUF::NgramDraft` candidates, verifies them with the exact target `prefill_tokens_top1s` path, restores/replays corrected tokens on rejection, and disables further n-gram drafting after the first rejection by default."
  source: `bin/qwen35_generate.cr` on 2026-04-24
  verified_at: 2026-04-24
  decay_trigger: generation decode loop, n-gram draft semantics, or target verifier semantics change
- claim: "The generation CLI still builds after adding the n-gram path."
  source: `CRYSTAL_CACHE_DIR=/tmp/cogni_ml_crystal_cache_ngram_generate_build crystal build --release --no-debug --link-flags="$(pwd)/build/bridge.o -framework Metal -framework Foundation -lc++" bin/qwen35_generate.cr -o /tmp/qwen35_generate_ngram` on 2026-04-24
  verified_at: 2026-04-24
  decay_trigger: Crystal compiler, Metal bridge, or Qwen source changes
- claim: "Focused n-gram draft specs still pass after reusing the primitive from the CLI."
  source: `CRYSTAL_CACHE_DIR=/tmp/cogni_ml_crystal_cache_ngram_generate_spec crystal spec spec/ngram_draft_spec.cr` returning `5 examples, 0 failures` on 2026-04-24
  verified_at: 2026-04-24
  decay_trigger: n-gram primitive or spec fixtures change
- claim: "On `The capital of France is`, `QWEN35_NGRAM_DECODE=1 /tmp/qwen35_generate_ngram ... 64` produced exactly the same generated token IDs as the default greedy CLI path and reported `accepted=55/55`, `cycles=2`, `plain_steps=9`, `disabled=false`."
  source: `/tmp/qwen35_generate_ngram "The capital of France is" 64` compared with `QWEN35_NGRAM_DECODE=1 /tmp/qwen35_generate_ngram "The capital of France is" 64` on 2026-04-24
  verified_at: 2026-04-24
  decay_trigger: prompt output, verifier path, or n-gram defaults change
- claim: "On adversary `The quick brown fox`, the opt-in n-gram CLI path also produced exactly the same generated token IDs as default greedy, accepted the first partial repeat chunk only partly (`18/32`), then disabled further n-gram drafting (`disabled=true`)."
  source: `/tmp/qwen35_generate_ngram "The quick brown fox" 64` compared with `QWEN35_NGRAM_DECODE=1 /tmp/qwen35_generate_ngram "The quick brown fox" 64` on 2026-04-24
  verified_at: 2026-04-24
  decay_trigger: prompt output, rejection replay, verifier path, or n-gram defaults change
**note:** This makes the repeated-text n-gram speed path usable from the practical demo without making it a default universal decode strategy. Keep it opt-in until a broader prompt suite confirms the fail-closed policy does not create unacceptable overhead.

### [LM-codex-HEAD-TOP1-ROWS-AUTO-REPAIR-1] Thresholded row-batched verifier top1 is active by default again
**status:** verified-feature-with-caveat
**trust:** {F:0.80, G:0.42, R:0.76}
**context:** ml (Qwen speculative verifier output head)
**evidence:**
- claim: "The source had a stale outer env gate in `prefill_tokens_top1s`: `output_project_top1s_routed` had the intended `rows >= QWEN35_HEAD_TOP1_ROWS_MIN` policy, but it was only called when `QWEN35_HEAD_TOP1_ROWS=1` or full-row research mode was set. Removing the outer gate makes the routed helper's threshold and `QWEN35_HEAD_TOP1_ROWS_OFF=1` escape hatch effective."
  source: `src/ml/gguf/qwen35_cpu.cr` on 2026-04-24
  verified_at: 2026-04-24
  decay_trigger: output-head routing, verifier chunk policy, or env gate changes
- claim: "Focused Qwen forward and DeltaNet specs pass after the routing change."
  source: `CRYSTAL_CACHE_DIR=/tmp/cogni_ml_crystal_cache_headrows_default_spec crystal spec spec/qwen35_forward_spec.cr spec/qwen35_delta_net_spec.cr --link-flags=...` returning `17 examples, 0 failures` on 2026-04-24
  verified_at: 2026-04-24
  decay_trigger: Qwen specs, Metal kernels, or verifier route changes
- claim: "Current high-accept smoke on `The capital of France is` shows the default thresholded route faster than rows-off: one direct A/B was `16.67 ms/tok` default versus `17.0 ms/tok` with `QWEN35_HEAD_TOP1_ROWS_OFF=1`; a four-pair noisy sequence had default wins/near-ties in all pairs (`16.43 vs 16.90`, `16.62 vs 16.95`, `19.82 vs 19.84`, `21.93 vs 22.16`)."
  source: `/tmp/qwen35_speculative_accept_headrows_default --tokens 64 --gamma 4 --max-gamma 32 --verify chunk-inplace "The capital of France is"` interleaved with `QWEN35_HEAD_TOP1_ROWS_OFF=1` on 2026-04-24
  verified_at: 2026-04-24
  decay_trigger: host load, draft speed, output-head kernel, or adaptive gamma schedule changes
- claim: "Reject-sensitive adversary prompts stayed exact in the same harness; `def fibonacci(n):` used one gamma-4 verifier cycle then target fallback (`22.46 ms/tok`), and `The quick brown fox` stayed exact while falling back after the first partial chunk (`23.62 ms/tok`)."
  source: `/tmp/qwen35_speculative_accept_headrows_default --tokens 64 --gamma 4 --max-gamma 32 --verify chunk-inplace "def fibonacci(n):"` and `"The quick brown fox"` on 2026-04-24
  verified_at: 2026-04-24
  decay_trigger: prompt distribution, adaptive gamma policy, or rows-min threshold changes
**note:** This repairs a routing inconsistency, not a new universal verifier architecture. It helps large accepted verifier chunks and should stay guarded by `QWEN35_HEAD_TOP1_ROWS_MIN=8`; earlier falsifiers still argue against full-row F16 top1 and against lowering the threshold to 4.

### [LM-codex-FFN-DOWN-ADD-WBA-1] Decode FFN-down can fold the residual add into Q4/Q6 GEMV
**status:** verified-feature-with-caveat
**trust:** {F:0.82, G:0.44, R:0.72}
**context:** ml (Qwen decode WBA / FFN diamond)
**evidence:**
- claim: "The decode wave now has Q4_K and Q6_K GEMV variants that write `residual + dot` directly, allowing the FFN-down path to skip the separate `add_vec` kernel for both recurrent and full-attention layers. The feature is default-on and can be disabled with `QWEN35_FFN_DOWN_ADD_FUSED_OFF=1`."
  source: `src/ml/gguf/kernels/gemm_q4k.metal`, `src/ml/gguf/kernels/gemm_q56k.metal`, and `src/ml/gguf/qwen35_metal.cr` on 2026-04-24
  verified_at: 2026-04-24
  decay_trigger: decode wave routing, Q4/Q6 GEMV kernels, or FFN residual semantics change
- claim: "Focused Qwen forward and DeltaNet specs pass with the fused-add route default-on."
  source: `CRYSTAL_CACHE_DIR=/tmp/cogni_ml_crystal_cache_ffn_add_default_spec crystal spec spec/qwen35_forward_spec.cr spec/qwen35_delta_net_spec.cr --link-flags=...` returning `17 examples, 0 failures` on 2026-04-24
  verified_at: 2026-04-24
  decay_trigger: specs, Metal kernels, or Qwen forward path changes
- claim: "The default fused route and the old route produce identical generated token IDs on `The quick brown fox` for 32 generated tokens."
  source: `/tmp/qwen35_generate_ffn_add "The quick brown fox" 32` compared with `QWEN35_FFN_DOWN_ADD_FUSED_OFF=1 /tmp/qwen35_generate_ffn_add "The quick brown fox" 32` on 2026-04-24
  verified_at: 2026-04-24
  decay_trigger: prompt output, decode wave route, or FFN precision changes
- claim: "Sync profile confirms the intended structural effect: default fused trace reports `rec.ffn_down_add` / `full.ffn_down_add` and no `rec.add` / `full.add`; off-route reports the old `rec.ffn_down` plus `rec.add` and `full.ffn_down` plus `full.add`. A short top1 sync smoke measured `24.82 ms/tok` default versus `29.57 ms/tok` off."
  source: `/tmp/qwen35_sync_profile_ffn_add` with and without `QWEN35_FFN_DOWN_ADD_FUSED_OFF=1`, prompt64/gen4, on 2026-04-24
  verified_at: 2026-04-24
  decay_trigger: profile harness, host load, or wave command grouping changes
- claim: "Paired decode A/B under active host load was positive but not clean enough to call a large benchmark win: opt-in A/B won `7/8` with `+0.312 ms/tok`, while a later default-vs-off A/B under heavier load was `4/8` with `+0.067 ms/tok`."
  source: `/tmp/qwen35_ab_profile_ffn_add --env=QWEN35_FFN_DOWN_ADD_FUSED --a='<unset>' --b=1 ...` and `/tmp/qwen35_ab_profile_ffn_add_default --env=QWEN35_FFN_DOWN_ADD_FUSED_OFF --a='<unset>' --b=1 ...` on 2026-04-24
  verified_at: 2026-04-24
  decay_trigger: quiet rerun, host load, or decode wave route changes
**note:** This is a small exact WBA sub-diamond, not the full tile-streamed FFN breakthrough. It removes a dispatch and one residual/output buffer pass, but the real large lever remains avoiding or reusing FFN activation/down traffic across the whole `gate/up -> SwiGLU -> down -> residual` diamond.

### [LM-codex-PREFILL-FFN-DOWN-ADD-WBA-1] Prefill FFN-down residual-add fusion is correctness-safe but not default-worthy
**status:** falsified-default-off-feature
**trust:** {F:0.84, G:0.48, R:0.78}
**context:** ml (Qwen prefill WBA / FFN diamond)
**evidence:**
- claim: "A Q6_K batch-GEMM prefill route can fold the residual add into the FFN-down output write while preserving the prior Q6 f32-output numeric contract by rounding the accumulator through `half` before adding. The route is opt-in with `QWEN35_PREFILL_FFN_DOWN_ADD_FUSED=1`; default remains the old separate FFN-down plus add path."
  source: `src/ml/gguf/kernels/gemm_mm.metal` and `src/ml/gguf/qwen35_metal.cr` on 2026-04-24
  verified_at: 2026-04-24
  decay_trigger: Q6_K prefill GEMM output precision, prefill chunk routing, or FFN residual semantics change
- claim: "Default prefill route correctness still passes the focused Qwen Metal/forward/DeltaNet specs after adding the opt-in route."
  source: `CRYSTAL_CACHE_DIR=/tmp/cogni_ml_crystal_cache_prefill_wba_default crystal spec spec/qwen35_metal_spec.cr spec/qwen35_forward_spec.cr spec/qwen35_delta_net_spec.cr --link-flags=...` returning `26 examples, 0 failures` on 2026-04-24
  verified_at: 2026-04-24
  decay_trigger: specs, Metal kernels, or Qwen35 prefill path changes
- claim: "The opt-in prefill FFN-down-add route also passes focused forward specs."
  source: `CRYSTAL_CACHE_DIR=/tmp/cogni_ml_crystal_cache_prefill_wba_optin QWEN35_PREFILL_FFN_DOWN_ADD_FUSED=1 crystal spec spec/qwen35_forward_spec.cr --link-flags=...` returning `14 examples, 0 failures` on 2026-04-24
  verified_at: 2026-04-24
  decay_trigger: specs, Metal kernels, or Qwen35 prefill path changes
- claim: "Paired prefill A/B rejects default-on at prompt64: default measured avg `160.72 ms`, p50 `162.31 ms`; opt-in measured avg `163.81 ms`, p50 `164.02 ms`; default won `8/8` pairs."
  source: `/tmp/qwen35_prefill_attribution_wba_final --prompt=64 --warmup=1 --reps=8 --compare-env=QWEN35_PREFILL_FFN_DOWN_ADD_FUSED --load-warning-threshold=0 --load-total-warning-threshold=0` on 2026-04-24
  verified_at: 2026-04-24
  decay_trigger: host load, prefill grouping, Q6_K add kernel, or benchmark harness changes
- claim: "Prompt256 A/B was not strong enough to override the prompt64 regression: default measured avg `537.37 ms`, p50 `537.40 ms`; opt-in measured avg `533.29 ms`, p50 `533.72 ms`; opt-in won `4/6` pairs."
  source: `/tmp/qwen35_prefill_attribution_wba_final --prompt=256 --warmup=1 --reps=6 --compare-env=QWEN35_PREFILL_FFN_DOWN_ADD_FUSED --load-warning-threshold=0 --load-total-warning-threshold=0` on 2026-04-24
  verified_at: 2026-04-24
  decay_trigger: host load, prompt length, prefill grouping, or benchmark harness changes
- claim: "Q4_K batch-add variants were explicitly rejected for now: direct `simdgroup_matrix` add is not supported by the Metal matrix type, and the cooperative-store fallback lost the existing direct simdgroup-store advantage. An accidental batch-Q4-to-GEMV add route was caught as a catastrophic regression before defaulting."
  source: local prefill WBA experiment on 2026-04-24; pp64 cooperative Q4+Q6 was neutral (`158.77 ms` default vs `159.29 ms` branch), and the accidental Q4 GEMV route regressed pp64 (`409.88 ms` opt-in vs `297.55 ms` off) plus pp256 (`1429.69 ms` opt-in vs `927.95 ms` off)
  verified_at: 2026-04-24
  decay_trigger: Q4_K batch GEMM output-store strategy, Metal simdgroup_matrix capabilities, or prefill add routing changes
**note:** This is useful as an opt-in falsifier harness and future Q6 experiment hook, not a current speed win. The WBA/LTP lesson is that simply folding the residual write is too local for prefill; the remaining exact opportunity is a larger FFN tile-streaming diamond or a lower-level Q4/Q6 tile algorithm that reduces dominant weight/activation traffic.

### [LM-codex-DN-POST-LOCAL-Y-FALSIFIER-1] Local-caching DeltaNet post values is not the recurrent prefill breakthrough
**status:** falsified-removed
**trust:** {F:0.78, G:0.44, R:0.72}
**context:** ml (Qwen prefill DeltaNet / recurrent WBA)
**evidence:**
- claim: "The current prefill wall is still dominated by grouped recurrent/DeltaNet command buffers: prompt64 profile reported `dn wait 141.49 ms` inside `153.92 ms` profiled wall, while the grouped `full+rec` waits were flat around `17.92-18.63 ms` and `rec0-2` was `13.32 ms`."
  source: `/tmp/qwen35_prefill_attribution_wba_final --prompt=64 --warmup=1 --reps=3 --load-warning-threshold=0 --load-total-warning-threshold=0` on 2026-04-24
  verified_at: 2026-04-24
  decay_trigger: prefill grouping, DeltaNet kernels, benchmark harness, or host load changes
- claim: "Caching each lane's four `Y[d]` values in `delta_net_post_norm_gate` and `delta_net_post_norm_gate_chunk` is correctness-safe in focused Metal specs but does not improve wall time."
  source: temporary patch to `src/ml/gguf/kernels/delta_net.metal`; `CRYSTAL_CACHE_DIR=/tmp/cogni_ml_crystal_cache_dn_post_delta_escalated crystal spec spec/qwen35_delta_net_spec.cr --link-flags=...` returned `3 examples, 0 failures`, and `CRYSTAL_CACHE_DIR=/tmp/cogni_ml_crystal_cache_dn_post_forward_escalated crystal spec spec/qwen35_forward_spec.cr --link-flags=...` returned `14 examples, 0 failures` on 2026-04-24
  verified_at: 2026-04-24
  decay_trigger: DeltaNet post kernel implementation, Metal compiler behavior, or spec coverage changes
- claim: "The local-cache patch regressed prompt64 and was neutral at prompt256: old pp64 p50 `152.56 ms` versus patched `155.71 ms`; old pp256 p50 `492.55 ms` versus patched `492.23 ms`. The patch was removed before commit."
  source: old `/tmp/qwen35_prefill_attribution_wba_final` versus patched `/tmp/qwen35_prefill_attribution_dnpost_new`, both run with `--warmup=1 --load-warning-threshold=0 --load-total-warning-threshold=0` on 2026-04-24
  verified_at: 2026-04-24
  decay_trigger: host load, DeltaNet post kernel, or Metal compiler scheduling changes
**note:** This falsifies a local post-kernel cache optimization, not DeltaNet as a target. The next exact recurrent lever needs a larger boundary change: either fuse DeltaNet output generation with post-normalization at a scan-compatible granularity, or replace the serial scan with a parallel/chunked associative formulation.

### [LM-codex-SPEC-GUARDED-FULL-ROWS-1] Guarded full-row verifier accelerates wide accepted speculative chunks, default-off
**status:** verified-opt-in-feature-with-caveat
**trust:** {F:0.78, G:0.36, R:0.72}
**context:** ml (Qwen speculative verifier output head)
**evidence:**
- claim: "Added `QWEN35_HEAD_FULL_ROWS_GUARDED=1`, a default-off verifier route that runs the fast full-row Q6/F16 lm-head path for large chunks, reduces top1/top2 per row, and falls back through exact row-batched fused top1 for rows whose F16 top1-top2 margin is below `QWEN35_HEAD_FULL_ROWS_MARGIN` (default `0.25`)."
  source: `src/ml/gguf/qwen35_cpu.cr`, `src/ml/gguf/qwen35_metal.cr`, and `src/ml/gguf/kernels/gemm_q56k.metal` on 2026-04-24
  verified_at: 2026-04-24
  decay_trigger: output-head kernels, Q6_K GEMM precision, speculative verifier route, or margin policy changes
- claim: "Default exactness remains intact after adding the opt-in route. Focused Qwen forward specs passed with default env and with `QWEN35_HEAD_FULL_ROWS_GUARDED=1 QWEN35_HEAD_FULL_ROWS_MARGIN=0.05`."
  source: `CRYSTAL_CACHE_DIR=/tmp/cogni_ml_crystal_cache_guarded_fullrows crystal spec spec/qwen35_forward_spec.cr --link-flags=...` and `QWEN35_HEAD_FULL_ROWS_GUARDED=1 QWEN35_HEAD_FULL_ROWS_MARGIN=0.05 CRYSTAL_CACHE_DIR=/tmp/cogni_ml_crystal_cache_guarded_fullrows_optin crystal spec spec/qwen35_forward_spec.cr --link-flags=...`, both returning `14 examples, 0 failures`
  verified_at: 2026-04-24
  decay_trigger: Qwen specs, Metal kernels, or verifier route changes
- claim: "Exact speculative harness smokes stayed aligned with plain greedy target output on high-accept and rejection prompts. `The capital of France is` used guarded rows16/rows32 chunks with zero fallback rows at margin `0.25` and measured `14.32 ms/tok`; `def fibonacci(n):` and `The quick brown fox` stayed exact and mostly bypassed the guarded path after early rejection/fallback."
  source: `/tmp/qwen35_speculative_accept_guarded --tokens 64 --gamma 4 --max-gamma 32 --verify chunk-inplace`, with `QWEN35_HEAD_FULL_ROWS_GUARDED=1`, on prompts `The capital of France is`, `def fibonacci(n):`, and `The quick brown fox` on 2026-04-24
  verified_at: 2026-04-24
  decay_trigger: prompt distribution, host load, speculative scheduler, or margin policy changes
- claim: "Paired high-accept A/B was noisy but favorable: neural-only guard won `3/4` pairs (default avg `1178.1 ms`, guarded avg `1126.2 ms`), and n-gram+neural guard won `4/4` pairs (default avg `762.5 ms`, guarded avg `672.0 ms`)."
  source: Python interleaved shell harness over `/tmp/qwen35_speculative_accept_guarded`, `tokens=64`, `gamma=4`, `max_gamma=32`, margin `0.05`, on 2026-04-24
  verified_at: 2026-04-24
  decay_trigger: host load, prompt mix, verifier chunk sizes, or Q6 full-row kernel changes
**note:** This deliberately does not default the route on. It is a useful upper-bound/opt-in speed path for accepted large chunks, but the accepted fast rows rely on a margin guard over an F16 full-row GEMM, not a formally exact argmax proof. Keep the final greedy-output equality check in the harness when using it for research.

### [LM-codex-Q8-FFN-DOWN-ADD-WBA-1] Q8 draft FFN-down can fold residual add exactly
**status:** verified-feature-with-caveat
**trust:** {F:0.82, G:0.46, R:0.76}
**context:** ml (Qwen speculative draft decode WBA)
**evidence:**
- claim: "Added `simd_mv_q8_0_f32_add` and routed Q8_0 FFN-down through the existing decode-wave FFN-down-add policy. The 0.8B draft now emits `rec.ffn_down_add` / `full.ffn_down_add` instead of separate `ffn_down` plus `add` kernels; `QWEN35_FFN_DOWN_ADD_FUSED_OFF=1` still disables the route."
  source: `src/ml/gguf/kernels/gemm_q56k.metal` and `src/ml/gguf/qwen35_metal.cr` on 2026-04-24
  verified_at: 2026-04-24
  decay_trigger: Q8_0 GEMV kernels, decode wave routing, or FFN residual semantics change
- claim: "Focused Metal and Qwen forward specs pass after enabling Q8 FFN-down-add by default."
  source: `CRYSTAL_CACHE_DIR=/tmp/cogni_ml_crystal_cache_q8add_spec crystal spec spec/qwen35_metal_spec.cr spec/qwen35_forward_spec.cr --link-flags=...` returning `23 examples, 0 failures` on 2026-04-24
  verified_at: 2026-04-24
  decay_trigger: specs, Metal kernels, or Qwen forward path changes
- claim: "Draft wave trace confirms the structural effect: default route reports `rec.ffn_down_add` and `full.ffn_down_add`; off-route reports `rec.ffn_down` + `rec.add` and `full.ffn_down` + `full.add`. Single trace wall was neutral (`8.10 ms/tok` both), so paired A/B is the stronger signal."
  source: `/tmp/qwen35_wave_trace_profile_q8add --model Qwen3.5-0.8B-Q8_0.gguf --prefill=64 --decode=64`, with and without `QWEN35_FFN_DOWN_ADD_FUSED_OFF=1`, on 2026-04-24
  verified_at: 2026-04-24
  decay_trigger: host load, trace harness, or wave route changes
- claim: "Paired 0.8B draft A/B favors the fused route: default mean `8.102 ms/tok`, off mean `8.257 ms/tok`, default wins `7/8` pairs. End-to-end speculative smoke on `The capital of France is` remained neutral because target verification dominated."
  source: `/tmp/qwen35_ab_profile_q8add --model Qwen3.5-0.8B-Q8_0.gguf --env=QWEN35_FFN_DOWN_ADD_FUSED_OFF --a='<unset>' --b=1 --prompt=64 --gen=64 --trials=8 --warmup=1`, plus `/tmp/qwen35_speculative_accept_q8add --tokens 64 --gamma 4 --max-gamma 32`, on 2026-04-24
  verified_at: 2026-04-24
  decay_trigger: host load, draft model, speculative scheduler, or target verifier cost changes
**note:** This is a real exact draft-kernel cleanup, but it is not a speculative breakthrough by itself. It reduces draft wave overhead; high-accept end-to-end decode now needs verifier-side gains or a smaller/faster draft to expose more of the benefit.

### [LM-codex-Q4-PAIR-H16-THRESHOLD-128-FALSIFIER-1] Q4 pair-H16 prefill threshold 128 was neutral before later H16 routing changes
**status:** stale historical falsifier; superseded by refreshed threshold64 default
**trust:** {F:0.76, G:0.30, R:0.45}
**context:** ml (Qwen prefill Q4 conversion reuse)
**evidence:**
- claim: "Added `QWEN35_Q4K_PAIR_H16_MIN_BATCH` as an experiment knob while keeping the default threshold at `256`."
  source: `src/ml/gguf/qwen35_metal.cr` and `README.md` on 2026-04-24
  verified_at: 2026-04-24
  decay_trigger: prefill Q4 gate/up route, Q4 H16 GEMM conversion path, or chunk-size policy changes
- claim: "A temporary default threshold of `128` activated shared Q4 H16 input conversion at pp128 and pp256 while leaving pp64 off. pp128 was neutral (`default-other: -0.01 ms`, wins `3/6`), pp64 was noise (`-0.77 ms`, wins `5/6` even though route stayed off), and pp256 preserved the known positive signal (`-1.53 ms`, wins `5/6`)."
  source: `/tmp/qwen35_prefill_attribution_pair128 --prompt=64/128/256 --warmup=1 --reps=6 --compare-env=QWEN35_Q4K_PAIR_H16_GEMM_OFF --load-warning-threshold=0 --load-total-warning-threshold=0` on 2026-04-24
  verified_at: 2026-04-24
  decay_trigger: host load, Q4 pair route, or prompt-length mix changes
**note:** This closes the old untested threshold gap for the 2026-04-24 route state. Later H16 routing cleanups changed the surrounding traffic enough that a 2026-04-25 refresh promoted threshold `64` as a small default win.

### [LM-codex-Q4-PAIR-H16-THRESHOLD64-REFRESH-1] Q4 pair-H16 prefill sharing now starts at pp64
**status:** verified small default tuning
**trust:** {F:0.84, G:0.44, R:0.80}
**context:** ml (Qwen35 prefill Q4 conversion reuse)
**evidence:**
- claim: "After the shared-H16 recurrent projection cleanup, lowering `QWEN35_Q4K_PAIR_H16_MIN_BATCH` from `256` to `64` is a small exact pp64/pp128 win."
  source: `src/ml/gguf/qwen35_metal.cr` default `Q4_PAIR_H16_MIN_BATCH = 64`, implemented on 2026-04-25
  verified_at: 2026-04-25
  decay_trigger: Q4 H16 GEMM route, FFN gate/up pairing, benchmark harness, or host load changes
- claim: "Focused correctness passes with the threshold64 default."
  source: `CRYSTAL_CACHE_DIR=/tmp/cogni_ml_crystal_cache_pair64_spec crystal spec spec/qwen35_metal_spec.cr spec/qwen35_forward_spec.cr --link-flags="$(pwd)/build/bridge.o -framework Metal -framework Foundation -lc++"` -> `23 examples, 0 failures` on 2026-04-25
  verified_at: 2026-04-25
  decay_trigger: specs, Metal kernels, or Qwen35 forward routes change
- claim: "Threshold64 versus historical threshold256 wins pp64 in repeated paired A/B: first run `150.03 ms` threshold64 vs `150.18 ms` default256, branch wins `6/8`; repeat `150.05 ms` threshold64 vs `150.38 ms` default256, branch wins `12/12`."
  source: `qwen35_prefill_attribution.cr --prompt=64 --compare-env=QWEN35_Q4K_PAIR_H16_MIN_BATCH --compare-off=64` runs on 2026-04-25
  verified_at: 2026-04-25
  decay_trigger: host load or prefill route changes
- claim: "Threshold64 also improves pp128 in the same comparison style: `263.66 ms` threshold64 vs `264.07 ms` default256, branch wins `5/6`."
  source: `CRYSTAL_CACHE_DIR=/tmp/cogni_ml_crystal_cache_pair128_refresh crystal run --release --link-flags=... bin/qwen35_prefill_attribution.cr -- --prompt=128 --warmup=1 --reps=6 --compare-env=QWEN35_Q4K_PAIR_H16_MIN_BATCH --compare-off=64` on 2026-04-25
  verified_at: 2026-04-25
  decay_trigger: host load or prefill route changes
- claim: "Default threshold64 versus fully disabling Q4 pair-H16 remains small-positive: pp64 `150.27 ms` default vs `150.40 ms` off (`5/8` wins), pp128 `263.51 ms` default vs `264.09 ms` off (`5/6` wins)."
  source: `qwen35_prefill_attribution.cr --prompt=64/128 --compare-env=QWEN35_Q4K_PAIR_H16_GEMM_OFF --compare-off=1` on 2026-04-25
  verified_at: 2026-04-25
  decay_trigger: host load, Q4 pair route, or prompt-length mix changes
**adversary:** The absolute delta is small (`~0.1-0.6 ms`), so this is not a breakthrough and should not be sold as closing the llama.cpp prefill gap. It is retained because it is exact, reversible by env, and consistently non-negative in the refreshed pp64/pp128 checks.

### [LM-codex-GENERATE-NEURAL-SPEC-1] qwen35_generate has opt-in exact neural speculative decode
**status:** verified-feature-with-caveat
**trust:** {F:0.82, G:0.42, R:0.76}
**context:** ml (Qwen CLI / speculative decode)
**evidence:**
- claim: "`qwen35_generate` now supports opt-in exact neural speculative decode via `QWEN35_SPECULATIVE_DECODE=1` and optional `QWEN35_DRAFT_MODEL`. The CLI uses the target chunk verifier, adaptive gamma, early-reject, and target-only fallback; it remains separate from the existing n-gram mode for this commit."
  source: `bin/qwen35_generate.cr` on 2026-04-24
  verified_at: 2026-04-24
  decay_trigger: generation CLI control flow, speculative harness policy, or state semantics change
- claim: "Generated token IDs match greedy target output for three 32-token smoke prompts: `The capital of France is`, `def fibonacci(n):`, and `The quick brown fox`."
  source: Python parity script running `/tmp/qwen35_generate_spec PROMPT 32` with and without `QWEN35_SPECULATIVE_DECODE=1` on 2026-04-24
  verified_at: 2026-04-24
  decay_trigger: tokenizer, prompt outputs, target verifier, or draft resync policy changes
- claim: "On a 64-token high-accept practical CLI smoke, `The capital of France is` measured `21.29 ms/tok` greedy, `17.48 ms/tok` neural speculative, and `15.73 ms/tok` with `QWEN35_HEAD_FULL_ROWS_GUARDED=1`; n-gram-only remained faster for this repeated-output prompt at `8.01 ms/tok`."
  source: `/tmp/qwen35_generate_spec 'The capital of France is' 64`, with env variants `QWEN35_SPECULATIVE_DECODE=1`, `QWEN35_HEAD_FULL_ROWS_GUARDED=1`, and `QWEN35_NGRAM_DECODE=1`, on 2026-04-24
  verified_at: 2026-04-24
  decay_trigger: host load, draft speed, target verifier, prompt distribution, or CLI timing changes
- claim: "The 32-token smoke shows fixed speculative overhead can dominate short runs: high-accept speculative measured `22.04 ms/tok` versus greedy `20.71 ms/tok`, so the CLI mode should stay opt-in and documented as useful for longer high-accept generations."
  source: same 32-token parity script on 2026-04-24
  verified_at: 2026-04-24
  decay_trigger: scheduler, draft cost, or default gamma policy changes
**note:** This moves the neural speculative speed path from benchmark harness into the practical CLI. It does not yet compose neural and n-gram modes in `qwen35_generate`; that should be a separate state-machine change.

### [LM-codex-GENERATE-NGRAM-GUARD-FALSIFIER-1] N-gram verifier chunks must not use guarded full-row top1
**status:** verified-fix-plus-refutation
**trust:** {F:0.84, G:0.46, R:0.78}
**context:** ml (Qwen CLI / n-gram speculative verifier)
**evidence:**
- claim: "A temporary ngram+neural composition in `qwen35_generate` diverged on `The quick brown fox` only when `QWEN35_HEAD_FULL_ROWS_GUARDED=1` was active. The same composition without the guard matched greedy output, isolating the hazard to guarded full-row verification on partial n-gram reject chunks."
  source: `/tmp/qwen35_generate_compose 'The quick brown fox' 48` with `QWEN35_SPECULATIVE_DECODE=1 QWEN35_NGRAM_DECODE=1 QWEN35_HEAD_FULL_ROWS_GUARDED=1` versus without the guard on 2026-04-24
  verified_at: 2026-04-24
  decay_trigger: guarded verifier margin policy, n-gram verifier route, or target top1 exactness changes
- claim: "The retained CLI fix wraps n-gram verifier calls with `with_guarded_full_rows_disabled`, so n-gram-only generation preserves exact greedy token IDs even if the guarded verifier env is set."
  source: `bin/qwen35_generate.cr` on 2026-04-24
  verified_at: 2026-04-24
  decay_trigger: generation CLI n-gram path or output-head routing changes
- claim: "Post-fix parity smokes with `QWEN35_NGRAM_DECODE=1 QWEN35_HEAD_FULL_ROWS_GUARDED=1` match greedy token IDs for `The capital of France is`, `The quick brown fox`, and `def fibonacci(n):` at 32 generated tokens."
  source: Python parity script over `/tmp/qwen35_generate_final PROMPT 32` on 2026-04-24
  verified_at: 2026-04-24
  decay_trigger: prompt output drift, n-gram policy, or verifier route changes
- claim: "The ngram+neural composition itself was not retained because it was not a speed win: on `The capital of France is` 64-token smoke, n-gram-only measured `7.76 ms/tok`, while neural speculative with guard measured `15.10 ms/tok`; partial/no-repeat prompts also favored fail-closed n-gram or greedy fallback over composition."
  source: `/tmp/qwen35_generate_final 'The capital of France is' 64` with greedy/spec/spec_guard/ngram_guard env variants on 2026-04-24
  verified_at: 2026-04-24
  decay_trigger: prompt mix, draft cost, n-gram policy, or verifier performance changes
**note:** Keep the modes separate in the practical CLI for now: use n-gram for repeated/template text, neural speculative for longer high-accept non-repeated text, and guarded full-row verifier only on neural speculative chunks.

### [LM-codex-GENERATE-DECODE-POLICY-1] qwen35_generate has explicit decode-mode policy
**status:** verified-feature
**trust:** {F:0.84, G:0.48, R:0.80}
**context:** ml (Qwen CLI / decode policy)
**evidence:**
- claim: "`qwen35_generate` now accepts `QWEN35_DECODE_POLICY=greedy|ngram|speculative|auto`. Explicit policy overrides legacy mode envs; `auto` currently maps to the exact fail-closed n-gram path instead of composing n-gram and neural speculative state machines."
  source: `bin/qwen35_generate.cr` on 2026-04-24
  verified_at: 2026-04-24
  decay_trigger: generation CLI control flow, speculative mode composition, or n-gram verifier semantics change
- claim: "Legacy ambiguous mode selection now fails before model load: setting both `QWEN35_SPECULATIVE_DECODE=1` and `QWEN35_NGRAM_DECODE=1` without an explicit policy exits with `QWEN35_SPECULATIVE_DECODE and QWEN35_NGRAM_DECODE are mutually exclusive`; invalid `QWEN35_DECODE_POLICY` values also fail before model load."
  source: `/tmp/qwen35_generate_policy` fail-fast smoke commands on 2026-04-24
  verified_at: 2026-04-24
  decay_trigger: env parsing or CLI startup behavior changes
- claim: "Real CLI parity checks with `QWEN35_DECODE_POLICY=auto QWEN35_HEAD_FULL_ROWS_GUARDED=1` match greedy generated token IDs for `The capital of France is`, `The quick brown fox`, and `def fibonacci(n):` at 32 generated tokens. The same smoke measured auto at `10.40`, `16.79`, and `20.83 ms/tok` respectively versus greedy `19.73`, `20.20`, and `19.77 ms/tok`."
  source: Python parity script over `/tmp/qwen35_generate_policy PROMPT 32` on 2026-04-24
  verified_at: 2026-04-24
  decay_trigger: prompt output drift, n-gram policy, guarded verifier interaction, or host load changes
**note:** This is a policy-safety cleanup rather than a new kernel win. Neural speculative stays opt-in; automatic neural selection needs a reliable acceptance/cost predictor before it is safe.

### [LM-codex-FINAL-FULL-LAST-FFN-ADD-FALSIFIER-1] Final full-attn prefill path should keep separate FFN-down add
**status:** falsified-no-code-retained
**trust:** {F:0.80, G:0.36, R:0.76}
**context:** ml (Qwen prefill WBA / final full-attention layer)
**evidence:**
- claim: "A temporary branch extended the existing exact FFN-down residual-add GEMV path into `full_attn_layer_chunk_project_last`, replacing the final full-attention prefill/top1 path's separate `ffn_down` plus `add_vec` with `encode_matmul_add` for batch 1."
  source: temporary local patch to `src/ml/gguf/qwen35_metal.cr` on 2026-04-24
  verified_at: 2026-04-24
  decay_trigger: final full-attention prefill route, GEMV-add kernels, or FFN residual semantics change
- claim: "Correctness checks passed with the temporary fusion: focused Qwen forward and Metal specs returned `23 examples, 0 failures`."
  source: `CRYSTAL_CACHE_DIR=/tmp/cogni_ml_crystal_cache_final_ffn_add crystal spec spec/qwen35_forward_spec.cr spec/qwen35_metal_spec.cr --link-flags=...` on 2026-04-24
  verified_at: 2026-04-24
  decay_trigger: specs, Metal kernels, or Qwen prefill route changes
- claim: "The pp64 prefill wall did not improve. Paired A/B with `QWEN35_FFN_DOWN_ADD_FUSED_OFF=1` favored the old separate route: fused default avg `150.64 ms`, old route avg `150.45 ms`, fused wins `3/8`."
  source: `/tmp/qwen35_prefill_attr_finalffn --prompt=64 --warmup=1 --reps=8 --compare-env=QWEN35_FFN_DOWN_ADD_FUSED_OFF --load-warning-threshold=0 --load-total-warning-threshold=0` on 2026-04-24
  verified_at: 2026-04-24
  decay_trigger: host load, final prefill path, or GEMV-add occupancy changes
**note:** This is another local-WBA falsifier: removing one tiny add dispatch is not enough when the alternate GEMV-add form changes kernel balance. Do not re-add this exact patch without a different lower-level kernel design.

### [LM-codex-NGRAM-RECURSIVE-COPY-1] Recursive n-gram drafting avoids scratch copies on no-match paths
**status:** verified-small-optimization
**trust:** {F:0.82, G:0.58, R:0.78}
**context:** ml (Qwen CLI / n-gram speculative decode)
**evidence:**
- claim: "`ML::GGUF::NgramDraft.candidates(..., recursive: true)` now runs the first suffix match against the existing history and only duplicates/extends scratch history after a first candidate chunk exists. No-candidate calls return without `history.dup`."
  source: `src/ml/gguf/ngram_draft.cr` on 2026-04-24
  verified_at: 2026-04-24
  decay_trigger: n-gram candidate semantics or recursive extension policy changes
- claim: "Focused n-gram specs still pass after the copy-avoidance change."
  source: `CRYSTAL_CACHE_DIR=/tmp/cogni_ml_crystal_cache_ngram_fast crystal spec spec/ngram_draft_spec.cr` returning `5 examples, 0 failures` on 2026-04-24
  verified_at: 2026-04-24
  decay_trigger: spec coverage or n-gram matching semantics changes
- claim: "An isolated 2048-token no-match microbench improved from `370.43 ms` to `358.58 ms` over 5000 recursive calls; a repeated-history microbench improved from `13.77 ms` to `13.09 ms`."
  source: `build/tmp/ngram_draft_bench.cr` comparing old local implementation vs current module on 2026-04-24
  verified_at: 2026-04-24
  decay_trigger: Crystal optimizer/runtime, NgramDraft implementation, or benchmark shape changes
- claim: "Real CLI auto parity still matches greedy token IDs for `The capital of France is`, `The quick brown fox`, and `def fibonacci(n):` at 32 generated tokens with `QWEN35_DECODE_POLICY=auto QWEN35_HEAD_FULL_ROWS_GUARDED=1`."
  source: Python parity script over `/tmp/qwen35_generate_ngramfast PROMPT 32` on 2026-04-24
  verified_at: 2026-04-24
  decay_trigger: prompt outputs, n-gram policy, guarded verifier interaction, or tokenizer changes
**note:** This is a small CPU-side cleanup, not a model-kernel breakthrough. It mostly protects auto/ngram mode when repeated suffixes do not appear.

### [LM-codex-GENERATE-SPEC-BOOTSTRAP-1] qwen35_generate exposes opt-in speculative bootstrap gamma
**status:** verified-feature-with-caveat
**trust:** {F:0.78, G:0.42, R:0.74}
**context:** ml (Qwen CLI / neural speculative decode)
**evidence:**
- claim: "`qwen35_generate` now accepts `QWEN35_SPEC_BOOTSTRAP_GAMMA=N` for neural speculative decode. After a fully accepted initial adaptive chunk, the CLI can jump directly to the configured gamma, bounded by `QWEN35_SPEC_MAX_GAMMA`."
  source: `bin/qwen35_generate.cr` on 2026-04-24
  verified_at: 2026-04-24
  decay_trigger: neural speculative scheduler or CLI env parsing changes
- claim: "The knob is default-off because prior harness evidence showed it can speed 100%-accept prompts but regress prompts that reject after an accepted prefix (`The capital of France is`: `17.89 -> 15.10 ms/tok`; `def fibonacci(n):` `23.36 -> 29.93 ms/tok`)."
  source: `TODO.md` bootstrap falsifier/feature entry from the existing speculative harness work on 2026-04-24
  verified_at: 2026-04-24
  decay_trigger: draft model, verifier cost, or adaptive scheduling changes
- claim: "Practical CLI parity with `QWEN35_DECODE_POLICY=speculative QWEN35_SPEC_BOOTSTRAP_GAMMA=32` matches greedy token IDs for `The capital of France is`, `def fibonacci(n):`, and `The quick brown fox` at 32 generated tokens. The same smoke shows why it stays manual: high-accept `The capital of France is` measured `16.65 ms/tok`, while rejection prompts regressed to `23.25` and `26.16 ms/tok`."
  source: Python parity script over `/tmp/qwen35_generate_bootstrap PROMPT 32` on 2026-04-24
  verified_at: 2026-04-24
  decay_trigger: draft acceptance, prompt mix, or speculative scheduler changes
**note:** This ports a known exact harness option into the practical CLI without changing defaults. It is a manual high-accept lever, not an auto policy.

### [LM-codex-GENERATE-SPEC-SKIP-DRAFT-FALLBACK-1] qwen35_generate skips draft backup/resync before target-only fallback
**status:** verified-feature
**trust:** {F:0.82, G:0.48, R:0.78}
**context:** ml (Qwen CLI / neural speculative decode)
**evidence:**
- claim: "`qwen35_generate` now skips draft-state backup and later draft resync when a speculative rejection would force the scheduler into target-only fallback (`current_gamma // 2 <= QWEN35_SPEC_PLAIN_FALLBACK_GAMMA`). Escape hatches are `QWEN35_SPEC_SKIP_DRAFT_BEFORE_FALLBACK_OFF=1` and `QWEN35_SPEC_SKIP_DRAFT_BACKUP_BEFORE_FALLBACK_OFF=1`."
  source: `bin/qwen35_generate.cr` on 2026-04-24
  verified_at: 2026-04-24
  decay_trigger: neural speculative scheduler, fallback policy, or draft state semantics change
- claim: "Practical CLI parity matches greedy token IDs and the off-route token IDs for `The capital of France is`, `def fibonacci(n):`, and `The quick brown fox` at 32 generated tokens."
  source: Python parity script over `/tmp/qwen35_generate_skipdraft PROMPT 32`, comparing greedy, default speculative, and skip-off speculative envs on 2026-04-24
  verified_at: 2026-04-24
  decay_trigger: prompt outputs, draft acceptance, or speculative verifier changes
- claim: "The same smoke shows the intended partial-reject effect: `The quick brown fox` avoids one draft backup/resync and improves from `25.99` to `25.09 ms/tok`; `def fibonacci(n):` improves slightly from `23.41` to `23.24 ms/tok`; high-accept `The capital of France is` is neutral/noisy."
  source: `/tmp/qwen35_generate_skipdraft` practical CLI smoke with `QWEN35_DECODE_POLICY=speculative`, with and without skip-off envs, on 2026-04-24
  verified_at: 2026-04-24
  decay_trigger: host load, prompt mix, draft acceptance, or fallback threshold changes
**note:** This is exact because once the scheduler enters target-only fallback, draft state is no longer read for the rest of generation. It reduces rejection overhead but does not change the broader conclusion that neural speculative should stay opt-in.

### [LM-codex-AUTO-NGRAM-MISS-LIMIT-FALSIFIER-1] Auto n-gram miss-limit is not a useful speed policy
**status:** falsified-no-code-retained
**trust:** {F:0.78, G:0.40, R:0.74}
**context:** ml (Qwen CLI / n-gram auto policy)
**evidence:**
- claim: "A temporary `QWEN35_NGRAM_PLAIN_MISS_LIMIT` policy disabled n-gram drafting after 16 consecutive no-candidate plain steps in `QWEN35_DECODE_POLICY=auto` mode while keeping explicit n-gram mode unlimited."
  source: temporary local patch to `bin/qwen35_generate.cr` on 2026-04-24
  verified_at: 2026-04-24
  decay_trigger: auto decode policy or n-gram candidate overhead changes
- claim: "The policy preserved exact token IDs versus greedy and no-limit auto mode for `The capital of France is`, `The quick brown fox`, and `def fibonacci(n):` at 32 generated tokens."
  source: Python parity script over `/tmp/qwen35_generate_automiss PROMPT 32` on 2026-04-24
  verified_at: 2026-04-24
  decay_trigger: prompt outputs, n-gram policy, or tokenizer changes
- claim: "It was not a speed win: no-repeat `def fibonacci(n):` measured `20.29 ms/tok` with miss-limit versus `20.22 ms/tok` without, while high-repeat and partial-repeat prompts were neutral/noisy. Since it can also suppress late n-gram wins, no source change was retained."
  source: `/tmp/qwen35_generate_automiss` practical CLI smoke with `QWEN35_DECODE_POLICY=auto`, with and without `QWEN35_NGRAM_PLAIN_MISS_LIMIT=0`, on 2026-04-24
  verified_at: 2026-04-24
  decay_trigger: host load, prompt mix, or NgramDraft overhead changes
**note:** Keep auto simple for now. A better auto policy needs a cheaper predictor of future repetition, not a blind consecutive-miss cutoff.

### [LM-codex-GENERATE-SPEC-FALLBACK-OFF-1] qwen35_generate has harness-compatible fallback-off switch
**status:** verified-feature
**trust:** {F:0.82, G:0.50, R:0.78}
**context:** ml (Qwen CLI / neural speculative decode)
**evidence:**
- claim: "`qwen35_generate` now supports `QWEN35_SPEC_PLAIN_FALLBACK_OFF=1`, matching the speculative harness. Default behavior is unchanged: target-only fallback remains enabled unless the env is set."
  source: `bin/qwen35_generate.cr` on 2026-04-24
  verified_at: 2026-04-24
  decay_trigger: neural speculative scheduler or CLI env parsing changes
- claim: "The existing draft-backup/resync skip is now gated by `spec_plain_fallback_enabled`, so fallback-off A/B does not accidentally skip draft state needed by continued speculation."
  source: `bin/qwen35_generate.cr` on 2026-04-24
  verified_at: 2026-04-24
  decay_trigger: draft state semantics or fallback policy changes
- claim: "Practical CLI parity with fallback enabled and disabled matches greedy token IDs for `The capital of France is`, `def fibonacci(n):`, and `The quick brown fox` at 32 generated tokens. The smoke also confirms fallback-off remains an experiment knob, not a speed default: `The quick brown fox` regressed from `25.05` to `30.56 ms/tok`."
  source: Python parity script over `/tmp/qwen35_generate_fallbackoff PROMPT 32`, comparing greedy, default speculative, and `QWEN35_SPEC_PLAIN_FALLBACK_OFF=1`, on 2026-04-24
  verified_at: 2026-04-24
  decay_trigger: prompt outputs, draft acceptance, or fallback scheduler changes
**note:** This is an experimental-control feature, not a default speed win. Use it to reproduce harness fallback-off comparisons inside the practical CLI.

### [LM-codex-GENERATE-NGRAM-LAZY-BACKUP-1] qwen35_generate allocates n-gram rollback backup lazily
**status:** verified-small-cleanup
**trust:** {F:0.82, G:0.54, R:0.76}
**context:** ml (Qwen CLI / n-gram speculative decode)
**evidence:**
- claim: "`qwen35_generate` now allocates the n-gram verifier rollback `State` only after an n-gram candidate chunk exists, instead of allocating it at the start of every n-gram/auto decode run."
  source: `bin/qwen35_generate.cr` on 2026-04-24
  verified_at: 2026-04-24
  decay_trigger: n-gram verifier rollback policy or state allocation semantics change
- claim: "Token parity still matches greedy for `QWEN35_DECODE_POLICY=auto` and `QWEN35_DECODE_POLICY=ngram` on `The capital of France is`, `The quick brown fox`, and `def fibonacci(n):` at 32 generated tokens."
  source: Python parity script over `/tmp/qwen35_generate_lazybackup PROMPT 32` on 2026-04-24
  verified_at: 2026-04-24
  decay_trigger: prompt outputs, n-gram verifier route, or state rollback semantics change
- claim: "Wall timing on the 32-token smoke is neutral/noisy, so this should be treated as allocation/overhead hygiene rather than a measured tok/s win."
  source: same `/tmp/qwen35_generate_lazybackup` parity smoke on 2026-04-24
  verified_at: 2026-04-24
  decay_trigger: host load, prompt mix, or CLI timing changes
**note:** This removes avoidable work from no-candidate auto/ngram runs without changing candidate behavior. It is not a breakthrough path.

### [LM-codex-GENERATE-TRACE-OFF-1] qwen35_generate can suppress per-step trace output
**status:** verified-feature
**trust:** {F:0.82, G:0.62, R:0.78}
**context:** ml (Qwen CLI / benchmark hygiene)
**evidence:**
- claim: "`qwen35_generate` now accepts `QWEN35_TRACE_STEPS_OFF=1` or `QWEN35_QUIET=1` to suppress per-token/per-cycle trace lines while keeping mode headers, summaries, generated token IDs, generated text, and full output."
  source: `bin/qwen35_generate.cr` and `README.md` on 2026-04-24
  verified_at: 2026-04-24
  decay_trigger: CLI output format or generation loop logging changes
- claim: "Trace-off parity matches baseline token IDs for greedy, auto n-gram, and neural speculative modes on `The capital of France is`, `The quick brown fox`, and `def fibonacci(n):` at 32 generated tokens; the smoke also confirmed no `gen`, `spec cycle`, or `ngram cycle` trace lines were printed."
  source: Python parity script over `/tmp/qwen35_generate_quiet PROMPT 32` on 2026-04-24
  verified_at: 2026-04-24
  decay_trigger: prompt outputs, decode policy, or CLI logging changes
**note:** This is benchmark/CLI hygiene, not a kernel-speed claim. Use trace-off for cleaner practical timing and logs.

### [LM-codex-GENERATE-SPEC-SINGLE-FAST-1] qwen35_generate has gamma=1 accepted-token fast path
**status:** verified-feature
**trust:** {F:0.82, G:0.42, R:0.76}
**context:** ml (Qwen CLI / neural speculative decode)
**evidence:**
- claim: "`qwen35_generate` now has the same exact gamma=1 accepted-token fast path as the speculative harness: when `cycle_gamma == 1` and `draft_next == target_next`, it advances target and draft directly instead of taking a draft backup, target backup, and chunk verifier path."
  source: `bin/qwen35_generate.cr` on 2026-04-24
  verified_at: 2026-04-24
  decay_trigger: neural speculative scheduler or state semantics changes
- claim: "The fast path is controlled by `QWEN35_SPEC_SINGLE_FAST_OFF=1` and is reported in the speculative summary as `single_fast=N`."
  source: `bin/qwen35_generate.cr` and `README.md` on 2026-04-24
  verified_at: 2026-04-24
  decay_trigger: CLI env parsing or output format changes
- claim: "Fallback-off parity on `Once upon a time` at 32 generated tokens produced identical token IDs with the fast path on and off; the smoke hit `single_fast=1`. Wall timing was neutral/noisy (`27.94/27.95 ms/tok` on vs `27.81/28.12 ms/tok` off), so this is a correctness-preserving control/overhead cleanup rather than a measured default speed win."
  source: Python parity/A-B script over `/tmp/qwen35_generate_singlefast "Once upon a time" 32` with `QWEN35_DECODE_POLICY=speculative QWEN35_SPEC_PLAIN_FALLBACK_OFF=1`, on 2026-04-24
  verified_at: 2026-04-24
  decay_trigger: host load, prompt acceptance, fallback policy, or draft model changes
**note:** Default target-only fallback usually exits speculation before gamma=1 accepted cycles matter. Keep this path because it aligns the practical CLI with the harness and removes avoidable work in fallback-off/adversarial experiments, but do not count it as a default performance breakthrough.

### [LM-codex-GENERATE-SPEC-VERIFY-MODE-1] qwen35_generate exposes exact neural verifier modes
**status:** verified-feature
**trust:** {F:0.84, G:0.52, R:0.78}
**context:** ml (Qwen CLI / neural speculative decode)
**evidence:**
- claim: "`qwen35_generate` now accepts `QWEN35_SPEC_VERIFY=chunk-inplace|hybrid|serial`. The default remains `chunk-inplace`; `hybrid` verifies only the first speculative cycle serially and then uses chunk-inplace; `serial` is a slow exact diagnostic mode."
  source: `bin/qwen35_generate.cr` and `README.md` on 2026-04-24
  verified_at: 2026-04-24
  decay_trigger: neural speculative verifier control flow changes
- claim: "Generated token IDs matched greedy for `chunk-inplace`, `hybrid`, and `serial` on `The capital of France is`, `Once upon a time`, and `def fibonacci(n):` at 16 generated tokens."
  source: Python parity script over `/tmp/qwen35_generate_verify PROMPT 16` on 2026-04-24
  verified_at: 2026-04-24
  decay_trigger: prompt outputs, verifier modes, tokenizer, or model file changes
- claim: "In practical 64-token CLI smokes, `hybrid` helps first-cycle partial-reject prompts by avoiding the chunk rollback path: `Once upon a time` improved from `22.84/23.06` to `22.48/22.11 ms/tok`, and `The quick brown fox` from `22.86/22.85` to `22.12/22.22`. The same mode regresses a high-accept prompt: `The capital of France is` moved from `16.33/16.40` to `16.94/16.72 ms/tok`."
  source: interleaved practical CLI smoke with `/tmp/qwen35_generate_verify PROMPT 64`, default vs `QWEN35_SPEC_VERIFY=hybrid`, on 2026-04-24
  verified_at: 2026-04-24
  decay_trigger: host load, prompt acceptance distribution, verifier implementation, or fallback policy changes
**note:** Do not make `hybrid` default without a reliable first-cycle rejection predictor. It is useful as an exact opt-in for partial-reject workloads and as evidence that verifier-policy selection, not more generic scheduler glue, is still a lever.

### [LM-codex-Q8-MIXED-DUAL-GEMV-FALSIFIER-1] Unequal-output Q8 dual GEMV is not a draft win
**status:** falsified-no-code-retained
**trust:** {F:0.78, G:0.42, R:0.74}
**context:** ml (Qwen speculative draft / Q8 kernels)
**evidence:**
- claim: "A temporary exact kernel fused the 0.8B recurrent draft `attn_qkv` and `attn_gate` Q8_0 GEMVs into one mixed dual dispatch, covering unequal output sizes `1024x6144` and `1024x2048` with the same normalized input."
  source: temporary local patch to `src/ml/gguf/kernels/gemm_q56k.metal` and `src/ml/gguf/qwen35_metal.cr` on 2026-04-24
  verified_at: 2026-04-24
  decay_trigger: Q8 kernel design or recurrent projection layout changes
- claim: "Focused verifier spec still passed after the temporary kernel route."
  source: `/tmp/qwen35_forward_spec_q8mixed --location spec/qwen35_forward_spec.cr:377` -> `1 examples, 0 failures`
  verified_at: 2026-04-24
  decay_trigger: Q8 kernel route or verifier spec changes
- claim: "Despite lower `rec.proj` encode time, wall regressed in the 0.8B draft profile: default mixed measured `7.62/7.66/7.70 ms/tok`, while `QWEN35_Q8_MIXED_DUAL_GEMV_OFF=1` measured `7.47/7.39/7.39 ms/tok` over interleaved 64-prefill/32-decode runs."
  source: `/tmp/qwen35_sync_profile_q8mixed --model Qwen3.5-0.8B-Q8_0.gguf 64 32` interleaved with and without `QWEN35_Q8_MIXED_DUAL_GEMV_OFF=1`, on 2026-04-24
  verified_at: 2026-04-24
  decay_trigger: Q8 kernel, threadgroup shape, Metal scheduler, or host load changes
**note:** Do not retry dispatch-only mixed fusion for this Q8 projection shape without a different kernel design. The next draft win needs better per-row throughput or less memory traffic, not just merging two unequal GEMV dispatches.

### [LM-codex-QWEN35-REC-PROJ-SHARED-H16-1] Recurrent prefill projection can share one H16 input between Q5 qkv and Q4 gate
**status:** verified default-on exact micro-optimization
**trust:** {F:0.86, G:0.54, R:0.82}
**context:** ml (Qwen35 prefill / Metal conversion traffic / WBA)
**evidence:**
- claim: "The recurrent prefill projection path now converts the normalized hidden chunk to H16 once, then feeds the same H16 input to the Q5 qkv half-output GEMM and the Q4 gate GEMM. This removes one duplicate `4096*b` F32->F16 conversion per recurrent layer without changing quantized weights or arithmetic kernels."
  source: `src/ml/gguf/qwen35_metal.cr` shared-H16 route guarded by `QWEN35_REC_PROJ_SHARED_H16_OFF=1`, implemented on 2026-04-25
  verified_at: 2026-04-25
  decay_trigger: recurrent prefill projection routing, Q5 h16 conv route, Q4 h16 GEMM route, or scratch-buffer ownership changes
- claim: "Focused correctness still passes after the route change."
  source: `CRYSTAL_CACHE_DIR=/tmp/cogni_ml_crystal_cache_shared_h16_spec crystal spec spec/qwen35_metal_spec.cr spec/qwen35_forward_spec.cr --link-flags="$(pwd)/build/bridge.o -framework Metal -framework Foundation -lc++"` -> `23 examples, 0 failures` on 2026-04-25
  verified_at: 2026-04-25
  decay_trigger: focused specs, Metal kernels, or Qwen35 forward route changes
- claim: "Fresh attribution shows the optimization reduces profiled conversion traffic and is not a short-prefill regression: pp64 conversion traffic fell from `385.50 MiB` to `349.50 MiB`, with paired A/B `150.45 ms` default vs `150.55 ms` off (`4/8` wins)."
  source: `CRYSTAL_CACHE_DIR=/tmp/cogni_ml_crystal_cache_shared_h16_ab64 crystal run --release --link-flags="$(pwd)/build/bridge.o -framework Metal -framework Foundation -lc++" bin/qwen35_prefill_attribution.cr -- --prompt=64 --warmup=1 --reps=8 --compare-env=QWEN35_REC_PROJ_SHARED_H16_OFF --compare-off=1 --load-warning-threshold=0 --load-total-warning-threshold=0` on 2026-04-25
  verified_at: 2026-04-25
  decay_trigger: host load, benchmark harness, or prefill route changes
- claim: "Longer prompt A/B is small-positive but not a breakthrough: pp256 measured `483.28 ms` default vs `484.48 ms` off (`4/6` wins), and pp1024 measured `1916.54 ms` default vs `1917.83 ms` off (`3/3` wins)."
  source: `qwen35_prefill_attribution.cr` paired A/B at pp256 and pp1024 with `QWEN35_REC_PROJ_SHARED_H16_OFF=1` on 2026-04-25
  verified_at: 2026-04-25
  decay_trigger: host load, benchmark harness, prompt length policy, or projection route changes
**adversary:** This is an exact traffic cleanup, not the next 10% prefill breakthrough. It helps the conversion-pressure direction found in pp1024 attribution, but the wall delta is around `0.1-1.3 ms` in current measurements; the larger remaining levers are still FFN tile-streaming/prepack and speculative verifier batching.

### [LM-codex-DELTANET-ASSOCIATIVE-SCAN-PLAN-1] DeltaNet may admit a RetNet-style block scan, but not the cheap RetNet formula
**status:** active research hypothesis
**trust:** {F:0.72, G:0.46, R:0.68}
**context:** ml (Qwen prefill / recurrent scan / LTP-WBA)
**evidence:**
- claim: "The old CogniFusion RetNet parallelization applies to a first-order linear recurrence `h_t = A*h_{t-1}+u_t`, where `A` is cheap to exponentiate or convolve over the sequence. That is why the prior `cumsum`/`conv1d`/cached-power frame was practical."
  source: `../../ML/cognifusion_llm/cogni_fusion_model.py` classes `MPSOptimisedParallelRetentionLogic` and `ParallelRetentionLogic`, inspected on 2026-04-25
  verified_at: 2026-04-25
  decay_trigger: old CogniFusion reference changes or a different RetNet formulation becomes the comparison target
- claim: "Qwen35 DeltaNet is also affine over its recurrent state, but with a token-dependent rank-1 transition: `S_t = g_t*S_{t-1}*(I - beta_t*K_t*K_t^T) + beta_t*V_t*K_t^T` after aligning row/column convention with the current kernel. Affine transform composition is associative, so a block prefix-scan is mathematically plausible."
  source: `src/ml/gguf/qwen35_cpu.cr` `delta_net_step!` and `src/ml/gguf/kernels/delta_net.metal` `delta_net_chunk_128_rowwise`, inspected on 2026-04-25
  verified_at: 2026-04-25
  decay_trigger: DeltaNet recurrence semantics, Q/K/V layout, or beta/g decay computation changes
- claim: "A naive dense block-summary scan is likely too expensive at Qwen35 shape `s=128`: transition summaries become dense `128x128` matrices per head, and composition would add matrix-matrix or matrix-state work that can exceed the current rowwise serial scan."
  source: algebraic work estimate from the same recurrence and the current `delta_net_chunk_128_rowwise` implementation, 2026-04-25
  verified_at: 2026-04-25
  decay_trigger: new Metal microbench showing dense summary composition is cheap enough
- claim: "The viable exact research path is a compact low-rank/WY-style block summary exploiting the rank-1 form of `I - beta*K*K^T`, plus serial replay inside blocks after a prefix scan over block summaries."
  source: synthesis from current DeltaNet rank-1 transition structure and prior RetNet parallel-scan pattern, 2026-04-25
  verified_at: 2026-04-25
  decay_trigger: CPU proof disproves equivalence, rank growth becomes unbounded in practice, or work model predicts no pp256+ win
- claim: "A tiny-shape CPU spec now proves the algebraic checkpoint: composed affine transforms match serial DeltaNet final state, and block-prefix replay reproduces serial intermediate outputs."
  source: `CRYSTAL_CACHE_DIR=/tmp/cogni_ml_crystal_cache_affine_scan crystal spec spec/qwen35_deltanet_affine_scan_spec.cr` -> `2 examples, 0 failures` on 2026-04-25
  verified_at: 2026-04-25
  decay_trigger: affine proof spec, DeltaNet recurrence, or row/column convention changes
- claim: "The first critical-path work model with a `1.25x` gate rejects naive dense summaries for `s=128` and gives a concrete compact-summary budget. At pp256, block sizes `8/16/32` can pass `1.25x` if one summary composition costs less than about `37.76/43.20/46.93` serial-token-equivalent DeltaNet steps; a naive dense estimate is about `170.67`, so dense block summaries are a non-starter."
  source: `CRYSTAL_CACHE_DIR=/tmp/cogni_ml_crystal_cache_scan_model crystal spec spec/qwen35_deltanet_affine_scan_spec.cr spec/qwen35_deltanet_scan_model_spec.cr` -> `5 examples, 0 failures`; `CRYSTAL_CACHE_DIR=/tmp/cogni_ml_crystal_cache_scan_model_bin crystal run bin/qwen35_deltanet_scan_model.cr -- --target=1.25 --prompts=64,256,1024 --blocks=8,16,32 --compose=8,32,170.67`
  verified_at: 2026-04-25
  decay_trigger: work model, DeltaNet rowwise cost model, or target speedup gate changes
- claim: "A dense CPU block-scan baseline confirms exactness but also exposes the expected overhead. It reproduces serial final state on `s=16,tokens=64,block=16` and `s=32,tokens=128,block=32` with max state deltas below `1e-16`; sequential dense summary+replay is much slower than serial (`~12.87 ms` vs `~1.08 ms`, then `~184.36 ms` vs `~9.89 ms`)."
  source: `CRYSTAL_CACHE_DIR=/tmp/cogni_ml_crystal_cache_dense_checkpoint crystal run bin/qwen35_deltanet_dense_scan_cpu.cr -- --s=16 --tokens=64 --block=16 --runs=3` and `--s=32 --tokens=128 --block=32 --runs=2` on 2026-04-25
  verified_at: 2026-04-25
  decay_trigger: dense baseline implementation, CPU compiler, or block summary algorithm changes
- claim: "The block-scan algebra is now a reusable module instead of duplicated spec/harness code: `ML::GGUF::Qwen35DeltaNetBlockScan` owns serial step, affine transform, composition, replay, and compact additive-summary formulas. The proof spec and dense CPU baseline both use it."
  source: `src/ml/gguf/qwen35_deltanet_block_scan.cr`, `spec/qwen35_deltanet_affine_scan_spec.cr`, and `bin/qwen35_deltanet_dense_scan_cpu.cr`; verified by `CRYSTAL_CACHE_DIR=/tmp/cogni_ml_crystal_cache_blockscan_refactor crystal spec spec/qwen35_deltanet_affine_scan_spec.cr spec/qwen35_deltanet_scan_model_spec.cr` -> `6 examples, 0 failures` on 2026-04-25
  verified_at: 2026-04-25
  decay_trigger: block-scan module, proof spec, or dense baseline changes
- claim: "Compact block summaries now avoid materializing dense `B`: `CompactDeltaSummary` stores dense `A` plus low-rank `B` factors, supports exact apply, and supports exact composition by transforming the first summary's right factors through the second summary's `A`. Focused specs cover compact apply and compact composition against dense affine composition."
  source: `CRYSTAL_CACHE_DIR=/tmp/cogni_ml_crystal_cache_compact_scan crystal spec spec/qwen35_deltanet_affine_scan_spec.cr spec/qwen35_deltanet_scan_model_spec.cr` -> `8 examples, 0 failures` on 2026-04-25
  verified_at: 2026-04-25
  decay_trigger: compact summary representation, composition formula, or DeltaNet recurrence changes
- claim: "Sequential CPU compact-summary baseline is still slower than serial but faster than dense-summary CPU on checked shapes: `s=16,tokens=64,block=16` measured compact `~8.41 ms` vs dense `~12.96 ms` and serial `~1.09 ms`; `s=32,tokens=128,block=32` measured compact `~108.38 ms` vs prior dense `~184 ms`. Storage is not favorable on tiny shapes with large blocks, but for Qwen `s=128`, dense-A plus low-rank-B storage is favorable for block sizes below `64`."
  source: `bin/qwen35_deltanet_compact_scan_cpu.cr` and `bin/qwen35_deltanet_dense_scan_cpu.cr` smokes on 2026-04-25
  verified_at: 2026-04-25
  decay_trigger: compact baseline implementation, storage model, or benchmark rerun
- claim: "The transition matrix `A` also has an exact compact representation: for a block, `A = gamma * (I + U V^T)` where each token contributes one rank-1 update from `I - beta*K*K^T`. The module now supports compact transition build, apply, and composition, plus fully compact summaries that avoid materializing dense `A` and dense `B`."
  source: `src/ml/gguf/qwen35_deltanet_block_scan.cr`; `CRYSTAL_CACHE_DIR=/tmp/cogni_ml_crystal_cache_fully_compact crystal spec spec/qwen35_deltanet_affine_scan_spec.cr spec/qwen35_deltanet_scan_model_spec.cr` -> `11 examples, 0 failures` on 2026-04-25
  verified_at: 2026-04-25
  decay_trigger: compact transition formula, DeltaNet recurrence, or composition semantics changes
- claim: "Fully compact CPU baseline is exact but still not a sequential win: `s=16,tokens=64,block=16 --fully` measured `~8.99 ms` vs serial `~1.16 ms`; `s=32,tokens=128,block=32 --fully` measured `~101.14 ms` vs serial `~7.49 ms`, with max state deltas below `1e-16`. For Qwen shape `s=128`, fully compact summary storage should be favorable at block sizes `8/16/32`, but only a Metal microbench can decide wall time."
  source: `CRYSTAL_CACHE_DIR=/tmp/cogni_ml_crystal_cache_fully_compact_final crystal run bin/qwen35_deltanet_compact_scan_cpu.cr -- --s=16 --tokens=64 --block=16 --runs=3 --fully` and `--s=32 --tokens=128 --block=32 --runs=2 --fully` on 2026-04-25
  verified_at: 2026-04-25
  decay_trigger: fully compact baseline implementation, CPU compiler, or Metal prototype evidence
- claim: "The first Metal microbench for fully compact summary apply is green at synthetic Qwen shape. A two-kernel `coeffs + apply` path over `s=128` measured p50 `~0.257/0.269/0.293/0.322 ms` for block/rank `8/16/32/64`, with max f32-vs-CPU drift below `3e-8`."
  source: `CRYSTAL_CACHE_DIR=/tmp/cogni_ml_crystal_cache_compact_metal crystal run --release --link-flags="$(pwd)/build/bridge.o -framework Metal -framework Foundation -lc++" bin/qwen35_deltanet_compact_metal_micro.cr -- --s=128 --block=8|16|32|64 --runs=5 --warmup=2` on 2026-04-25
  verified_at: 2026-04-25
  decay_trigger: compact Metal microbench, Metal compiler, host load, or summary apply kernel changes
- claim: "The compact-`A` construction Metal microbench is exact but only promising at larger block sizes. At `s=128`, p50 was `~0.223/0.281/0.442/1.108 ms` for block/rank `8/16/32/64`; versus the CPU construction reference this is `0.07x/0.24x/2.13x/1.24x`. Block `32` is the first plausible construction point; block `8/16` need fusion with apply/replay or they lose to setup/dispatch overhead."
  source: `CRYSTAL_CACHE_DIR=/tmp/cogni_ml_crystal_cache_compact_build_metal crystal run --release --link-flags="$(pwd)/build/bridge.o -framework Metal -framework Foundation -lc++" bin/qwen35_deltanet_compact_build_metal_micro.cr -- --s=128 --block=8|16|32|64 --runs=5 --warmup=2` on 2026-04-25
  verified_at: 2026-04-25
  decay_trigger: compact-A construction kernel, CPU reference timing, Metal compiler, or block-size policy changes
- claim: "The compact-`B` right-factor construction Metal microbench is exact and identifies the current construction bottleneck. At `s=128`, p50 was `~1.206/1.086/2.034/2.835 ms` for block/rank `8/16/32/64`, with `max_right_delta < 5e-8`; this dominates the earlier compact-`A` construction and apply probes. Block `16` currently has the best construction+apply lower-bound among tested sizes."
  source: `CRYSTAL_CACHE_DIR=/tmp/cogni_ml_crystal_cache_compact_b_metal crystal run --release --link-flags="$(pwd)/build/bridge.o -framework Metal -framework Foundation -lc++" bin/qwen35_deltanet_compact_b_metal_micro.cr -- --s=128 --block=8|16|32|64 --runs=5 --warmup=2` on 2026-04-25
  verified_at: 2026-04-25
  decay_trigger: compact-B construction kernel, Metal compiler, host load, or block-size policy changes
- claim: "The current multi-dispatch compact block-scan design is falsified as a prefill integration candidate. A synthetic lower-bound path that only computes final state (`build(A+B)+apply`) loses badly to the existing `delta_net_chunk_128_rowwise` kernel at `s=128`: compact p50 `~1.300/2.324/3.566 ms` vs rowwise `~0.270/0.287/0.318 ms` for block `8/16/32`. This is especially decisive because rowwise also computes all per-token outputs, while the compact path does not yet replay outputs."
  source: `CRYSTAL_CACHE_DIR=/tmp/cogni_ml_crystal_cache_compact_vs_rowwise crystal run --release --link-flags="$(pwd)/build/bridge.o -framework Metal -framework Foundation -lc++" bin/qwen35_deltanet_compact_vs_rowwise_micro.cr -- --s=128 --block=8|16|32 --runs=5 --warmup=2` on 2026-04-25
  verified_at: 2026-04-25
  decay_trigger: compact-B construction fused with replay/apply, rowwise kernel changes, or a new associative scan formulation
**decision:** Add a default-off research backlog item before touching production kernels: prove tiny-shape equivalence against serial `delta_net_step!`, estimate `s=128` work for pp64/pp256/pp1024/pp2048, and only then prototype compact summaries and a Metal block size `8` or `16` path.
**adversary:** This is unlikely to help decode (`T=1`) and may not help short pp64 prefill. Dense summaries are now kept only as a long-prefill GPU baseline; compact summaries remain mathematically valid but the current multi-dispatch implementation is not a speed path. Do not replace the existing rowwise kernel unless a future design fuses compact-`B` construction with replay/apply or finds a different parallel scan formulation that beats `delta_net_chunk_128_rowwise` in a full A/B.

### [LM-codex-Q4-F16-PREPACK-FALSIFIER-1] Full-F16 expansion is slower than compressed Q4_H16 for the hot FFN prefill shape
**status:** refuted optimization branch
**trust:** {F:0.72, G:0.50, R:0.76}
**context:** ml (Qwen35 prefill / Q4_K GEMM / LTP-WBA)
**evidence:**
- claim: "For the hot Q4_K FFN gate/up shape `4096x12288`, the current compressed Q4_H16 kernel is faster than a synthetic full-F16 expanded-weight GEMM at tested prefill batch sizes. Measured p50 wait times were b64 `1.401 ms` Q4_H16 vs `1.927 ms` F16, b128 `2.413 ms` vs `3.731 ms`, and b256 `3.132 ms` vs `6.872 ms`."
  source: `/tmp/qwen35_q4_wait_micro.cr` and `/tmp/qwen35_f16_prepack_micro.cr`, built with `crystal build --release --link-flags="$(pwd)/build/bridge.o -framework Metal -framework Foundation -lc++"` and run at `BATCH=64,128,256 RUNS=7 WARMUP=2` on 2026-04-25
  verified_at: 2026-04-25
  decay_trigger: Q4_H16 kernel, F16 GEMM kernel, Metal compiler, or batch tiling changes
- claim: "Standalone attribution with readback still shows the same direction: `Q4_K 4096x12288 b64` measured `3.133 ms` p50 through `Qwen35Metal.matmul`, while the no-readback F16 synthetic path measured about `1.95 ms`; after removing the readback mismatch with the Q4 wait-only microbench, compressed Q4_H16 is the faster route."
  source: `CRYSTAL_CACHE_DIR=/tmp/cogni_ml_crystal_cache_op_attr crystal run --release --link-flags="$(pwd)/build/bridge.o -framework Metal -framework Foundation -lc++" bin/qwen35_op_attribution.cr -- --batch=64 --warmup=2 --runs=7 --limit=8` and the wait-only microbench above
  verified_at: 2026-04-25
  decay_trigger: attribution harness, readback behavior, or scratch-buffer route changes
**decision:** Do not spend the next production pass on full-F16 predequantized FFN weights. If a prepack branch is revisited, it should preserve compressed/blocked Q4 layout or fuse downstream FFN work; expanding the hot weights to F16 loses on memory traffic before integration overhead is counted.
**adversary:** The F16 microbench is synthetic and uses the current generic `simd_mm_f16` kernel, so this does not formally rule out a hand-tuned F16 GEMM. It does rule out the cheap "predequantize Q4 weights to full F16 and reuse existing F16 GEMM" path as a plausible short-term win.

### [LM-codex-FFN-SWIGLU-ONLY-WBA-FALSIFIER-1] SwiGLU-only fusion is too small for a prefill breakthrough
**status:** refuted optimization branch
**trust:** {F:0.70, G:0.48, R:0.74}
**context:** ml (Qwen35 prefill / FFN WBA / activation staging)
**evidence:**
- claim: "For the hot `4096x12288` Q4 FFN gate/up pair, adding the standalone SwiGLU combine to the same command buffer is almost free relative to the two Q4_H16 GEMMs. Measured p50 at b64 was `1.861 ms` for pair GEMMs, `0.193 ms` for SwiGLU alone, and `1.877 ms` for pair+SwiGLU; b128 was `3.255/0.220/3.268 ms`; b256 was `6.063/0.254/6.119 ms`."
  source: `/tmp/qwen35_ffn_gate_swiglu_micro.cr`, built with `crystal build --release --link-flags="$(pwd)/build/bridge.o -framework Metal -framework Foundation -lc++"` and run at `BATCH=64,128,256 RUNS=9 WARMUP=3` on 2026-04-25
  verified_at: 2026-04-25
  decay_trigger: Q4_H16 pair route, SwiGLU kernel, command-buffer scheduling, or batch tiling changes
- claim: "The current pp64 prefill profile still shows FFN gate/up as the dominant logical weight traffic (`1674 MiB` combined recurrent+full Q4_H16 upgate), but the elementwise activation stage is not separately large enough to recover the remaining `~7-8%` llama.cpp first-run prefill gap."
  source: `CRYSTAL_CACHE_DIR=/tmp/cogni_ml_crystal_cache_prefill_attr_wba crystal run --release --link-flags="$(pwd)/build/bridge.o -framework Metal -framework Foundation -lc++" bin/qwen35_prefill_attribution.cr -- --prompt=64 --warmup=1 --reps=3 --load-warning-threshold=0 --load-total-warning-threshold=0` on 2026-04-25
  verified_at: 2026-04-25
  decay_trigger: profile instrumentation, prompt length, host load, or FFN route changes
**decision:** Do not implement a narrow `SwiGLU`-only fusion as the next production branch. A useful WBA move must either fuse a larger `gate/up -> activation -> down` tile-streaming diamond or improve the Q4/Q6 matmul kernels; just eliminating the elementwise activation dispatch/intermediate read is too small.
**adversary:** This microbench does not prove that a full FFN tile-streaming kernel is unprofitable. It only rules out the narrow activation-stage fusion hypothesis; the larger diamond remains a research branch because it could avoid more global writes/reads but has much higher kernel complexity and exactness risk.

### [LM-codex-QWEN35-PREPARE-STATE-METAL-1] Preparing Metal state buffers removes first-touch allocation from prefill latency
**status:** implemented
**trust:** {F:0.82, G:0.58, R:0.82}
**context:** ml (Qwen35 benchmark / state preparation / prefill latency)
**evidence:**
- claim: "A fresh `State` whose Metal KV/DeltaNet buffers are allocated and cleared before timing runs pp64 prefill about `7.95 ms` faster than a fresh lazy-allocation state in the timed region: fresh p50 `150.19 ms`, prepared p50 `142.24 ms`."
  source: `/tmp/qwen35_prefill_state_prep_micro.cr` with `PROMPT=64 RUNS=6 WARMUP=1` on 2026-04-25
  verified_at: 2026-04-25
  decay_trigger: state allocation policy, MetalBuffer behavior, or benchmark harness timing boundary changes
- claim: "The reusable API and benchmark mode are now implemented as `Qwen35CPU.prepare_state_metal!(state, hp)` and `benchmark_qwen_vs_llama --native-prefill-prepare-state`. Focused Qwen Metal/forward specs pass."
  source: `CRYSTAL_CACHE_DIR=/tmp/cogni_ml_crystal_cache_prepare_state_specs crystal spec spec/qwen35_metal_spec.cr spec/qwen35_forward_spec.cr --link-flags="$(pwd)/build/bridge.o -framework Metal -framework Foundation -lc++"` -> `23 examples, 0 failures` on 2026-04-25
  verified_at: 2026-04-25
  decay_trigger: state preparation code, specs, or Metal state layout changes
- claim: "Matched llama.cpp snapshot with prepared state measured native prefill `142.31 ms` / `449.73 tok/s` p50 vs llama.cpp `464.78 tok/s` (`-3.24%`), while default first-run in the same binary measured `149.99 ms` / `426.70 tok/s` vs llama.cpp `455.10 tok/s` (`-6.24%`)."
  source: `/tmp/benchmark_qwen_vs_llama_prepare_state --prompt=64 --gen=64 --reps=3 --warmup=1 --native-prefill-prepare-state --load-warning-threshold=0 --load-total-warning-threshold=0` and the same command without `--native-prefill-prepare-state` on 2026-04-25
  verified_at: 2026-04-25
  decay_trigger: host load, llama.cpp build, benchmark mode, or state preparation semantics changes
- claim: "`qwen35_generate` now uses eager Metal state preparation by default for the target state and neural-speculative draft/backup states; `QWEN35_PREPARE_STATE_OFF=1` restores lazy state-buffer allocation for A/B."
  source: `bin/qwen35_generate.cr`, implemented and smoke-tested on 2026-04-25
  verified_at: 2026-04-25
  decay_trigger: CLI state lifecycle, prompt-cache restore path, or speculative state management changes
- claim: "`qwen35_prefill_attribution --prepare-state` now measures the same prepared-state boundary for operator/profile work. Current pp64 prepared profile is `142.31 ms` p50 / `449.72 tok/s`, with matmul/conversion traffic still dominated by Q4 FFN up/gate and recurrent projection/down paths."
  source: `/tmp/qwen35_prefill_attribution_prepare --prompt=64 --warmup=1 --reps=3 --prepare-state --load-warning-threshold=0 --load-total-warning-threshold=0` on 2026-04-25
  verified_at: 2026-04-25
  decay_trigger: attribution harness, prepared-state code, or prefill route changes
**decision:** Keep default first-run semantics unchanged, but expose prepared-state latency as an explicit server/session mode. This is not a kernel speedup; it is a timing-boundary and product-latency optimization for callers that can create a session before prompt ingest.
**adversary:** This must not be marketed as making cold state+prefill cheaper end-to-end. The allocation/zeroing work still exists; it is moved before the latency-sensitive prefill measurement. It is nevertheless useful and comparable to systems that initialize KV/recurrent cache before timing prompt ingest.
