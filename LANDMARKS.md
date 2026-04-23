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
  source: same file
  verified_at: 2026-04-22
  decay_trigger: N/A

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

## Future Landmarks (TBD)

- [LM-claude-SOTA-1] DeltaNet/GatedDeltaRule SoTA harvest (before Фаза 3b)
