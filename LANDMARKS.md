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

## Future Landmarks (TBD)

- [LM-claude-SOTA-1] DeltaNet/GatedDeltaRule SoTA harvest (before Фаза 3b)
