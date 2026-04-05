# Cogniformerus Native Model: Architecture Design

**Status:** Phase 0.5 validated, Phase 0 next
**Date:** 2026-04-05
**Context:** crystal_ball SWE agent, cogni-ml Metal inference, Cogniformerus MCP

## 1. Core Thesis

SSM/Mamba/HGRN2 layers have exponential memory decay — proven in cogniversum_v2
(EMA hierarchy plateaus at ~26.7% for complex tasks) and cognifusion_llm
(alpha=0.999 causes gradient short-circuit, working range 0.9-0.99).

**No O(N) recurrent architecture can provide long-range precision.**
This is an information-theoretic limit of fixed-size state:
- State h in R^D = D floats = 8KB
- 100K tokens of context = ~200K bits of unique information
- Compression ratio 25:1 through lossy exponential decay

**Solution:** External memory (Cogniformerus pg_sorted_heap) provides precision.
The model only needs a small hot window for syntax + cross-attention to
retrieved facts for knowledge.

## 2. Architecture

```
┌─────────────────────────────────────────────────────┐
│ nomic BERT MoE encoder (137M, already on Metal GPU) │
│ 768d, 12 heads, 8 experts top-2, bidirectional      │
│ Input: current hot window (512 tok) + retrieved facts│
│ Output: token-level representations [N, 768]         │
│ Dual-use: same model for retrieval embeddings        │
│ 2x faster than llama.cpp on same hardware            │
└──────────────┬──────────────────────────────────────┘
               │ encoded representations
┌──────────────▼──────────────────────────────────────┐
│ Fact Cross-Attention (position-free!)                │
│ Q = from decoder (learned projection)                │
│ K = facts @ Wk_cross (learned, NOT raw embedding)    │
│ V = facts @ Wv_cross (learned, separate from K)      │
│ No RoPE — pure content matching                      │
│ With fact gating: g = sigmoid(Wg * x)                │
│                   x = x + g * cross_attn(...)        │
└──────────────┬──────────────────────────────────────┘
               │ fact-enriched representations
┌──────────────▼──────────────────────────────────────┐
│ Causal Decoder (~500M params)                        │
│ - Causal self-attention with ALiBi or RoPE (hot window)
│ - SwiGLU FFN                                         │
│ - GQA (grouped query attention)                      │
│ - Cross-attention every N layers (not all)            │
│ - Optionally: HGRN2/BitLinear for efficiency         │
└──────────────┬──────────────────────────────────────┘
               │
               ▼ generated tokens / tool calls
```

### Why position-free fact attention?

RoPE encodes token ORDER. Facts from Cogniformerus are atomic knowledge
units — their order doesn't matter, only their content:
- "class App < Widget, file: src/tui/app.cr" — position irrelevant
- "method render takes buffer parameter" — position irrelevant

Cross-attention to facts needs content matching, not positional bias.
With ALiBi for self-attention: K vectors are position-independent,
can be freely cached, moved, offloaded.

### Why separate K/V projections for facts?

Phase 0.5 experiment showed nomic raw embeddings as K=V:
- Tool discrimination: PASS (avg cosine 0.53 between different tools)
- Semantic discrimination: PASS 7/8 (avg 0.66)
- **Semantic grouping: SOFT** (avg 0.72, below 0.85 target)
- Edge case: increment vs decrement = 0.97 (can't distinguish)

Learned K/V projections will:
- Map stylistic variants to common key space (Group B improvement)
- Allow decoder to learn distinct "what to match" (K) vs "what to inject" (V)
- Enable multi-head specialization

### Resource budget (M2 Max 64GB)

| Component | Parameters | RAM (Q4) | Role |
|-----------|-----------|----------|------|
| nomic BERT | 137M | ~100MB | Encoder (dual-use with retrieval) |
| Decoder | 500M | ~300MB | Generator |
| Cogniformerus PG | — | ~1GB | External memory |
| **Total** | **637M** | **~1.4GB** | |
| vs GPT-OSS 20B | 20B | ~10GB | All-in-one |

30x fewer parameters, 7x less RAM. Room for 10+ parallel sessions.

## 3. What exists already

### In cogni-ml:
- [x] nomic BERT MoE full forward pass on Metal GPU (`src/ml/gguf/nomic_bert.cr`)
- [x] Custom Metal shaders: flash attention, GEMM Q5_K/Q6_K, fused BERT ops
- [x] ComputeGraph with automatic wave/barrier scheduling
- [x] MetalBuffer with unified memory, BufferPool
- [x] Tokenizer (Unigram)
- [x] Weight loading from GGUF
- [x] 2x faster than llama.cpp (measured)
- [x] Causal attention Metal kernel — prefill + decode + cross-attention (`kernels/attention_causal.metal`)
- [x] Sampling Metal kernel — top-k + softmax (`kernels/sampling.metal`)
- [x] CausalDecoder Crystal scaffold — full forward pass CPU reference (`src/ml/gguf/decoder.cr`)
- [x] KV cache with bounds checking, V transpose, GQA support
- [x] Phase 0.5 code discrimination test (`bin/test_code_discrimination.cr`)

### In Cogniformerus:
- [x] Hierarchical memory (pg_sorted_heap + GraphRAG)
- [x] nomic-embed for retrieval (same vector space)
- [x] MCP tools: memory_store, memory_retrieve, memory_hybrid_retrieve
- [x] 4K fixed window + fine-tuned extraction layers
- [x] GptOssPointerHead (bi-linear attention for query→memory selection)

### In crystal_ball:
- [x] GPT-OSS native format prompt builder
- [x] Tool reminder in SWA window
- [x] LocalDirect provider with KV cache prefix matching
- [x] LocalInferenceServer with OS thread inference (non-blocking TUI)

## 4. Phase 0.5 Experiment Results

**Test:** Can nomic-embed distinguish semantically different code?

```
Group A (must discriminate — different operations):
  ✓ cos=0.69  read vs write file
  ✓ cos=0.51  push vs delete
  ✓ cos=0.50  find vs create (DB)
  ✓ cos=0.55  send vs receive
  ✗ cos=0.97  increment vs decrement (subtle — +=1 vs -=1)
  ✓ cos=0.75  process vs delete file
  ✓ cos=0.59  constructor vs destructor
  ✓ cos=0.73  GET vs POST request
  avg=0.66

Group B (must group — same semantics, different syntax):
  ~ cos=0.83  select vs reject
  ✗ cos=0.72  map vs manual loop
  ✗ cos=0.64  nil guard styles
  ✗ cos=0.71  File.open vs File.read
  avg=0.72

Group C (baseline — completely different):
  avg=0.18

Group D (tool discrimination — critical for agent):
  ✓ cos=0.46  grep vs read_file
  ✓ cos=0.56  edit vs read
  ✓ cos=0.43  shell vs grep
  ✓ cos=0.65  list_dir vs read_file
  ✓ cos=0.54  intent-to-search vs fact-about-function
  avg=0.53
```

**Verdict: SOFT PASS.**
- Operation types (Group D): excellent discrimination
- Read vs write (Group A): good discrimination, 7/8 pass
- Subtle changes (+=1 vs -=1): fail — nomic can't see single-char diffs
- Style equivalents (Group B): below 0.85 target — needs learned projections

## 5. Implementation Phases

### Phase 0: GPT-OSS + Cogniformerus fact injection (NOW, zero training)

```
user message → Cogniformerus.retrieve(query) → inject facts into prompt
tool result  → Cogniformerus.store(key facts) → continue
```

Facts injected in developer message within SWA=128 window of GPT-OSS.
Validates that fact injection improves tool calling quality.

Implementation: ~30 lines in LocalDirect.generate_with_tools.

### Phase 1: Fine-tune existing small model with Cogniformerus

Take Qwen 3.5 3B or 9B (already has SSM + attention):
- Add cross-attention adapter layers (LoRA) to Cogniformerus facts
- Fine-tune on filtered session data (successful tool chains only)
- Tests WHETHER retrieval-grounded generation works
- WITHOUT training decoder from scratch

### Phase 2: Distillation

Use GPT-OSS 20B / Grok 4.1 as teacher:
- Generate "golden" trajectories for 1000+ coding tasks
- Logits distillation into custom 500M decoder
- Hidden-state matching for cross-attention layers
- Fact dropout during training (robustness)

### Phase 3: Custom architecture (full Cogniformerus-native)

nomic encoder + HGRN2/BitLinear decoder, trained from scratch:
- Distilled data from Phase 2
- 1.4GB total deployment
- 10 parallel sessions on 64GB M2 Max
- Deploy as Cogniformerus-native model

## 6. GPT-5.4 Review Findings (2026-04-05)

Three-model review: Grok 4.1 + GPT-5.4 Mini + GPT-5.4 Full.
Self-review found the most critical bug.

### Critical bugs fixed:
- KV cache `write_bytes` with nonexistent `offset` parameter — every append
  overwrote buffer start (fixed: direct pointer access via `contents`)
- `v_t_buf` allocated but never populated with transposed V
- Config validation: dim == n_heads * head_dim, divisibility checks
- LM head type mismatch (Array vs QuantWeight)

### GPT-5.4 Full unique findings (27 issues total):
- Metal kernels assume n_heads == n_kv_heads (GQA not supported in kernels)
- Metal kernels overflow at head_dim > 64 (`float4 qr[16]` fixed array)
- Facts shape [n_facts, dim] incompatible with kernel layout [n_heads, kv_len, head_dim]
- Cross-attention entirely unimplemented (stubs only)
- RoPE applied consistently (half-split matches Llama convention — confirmed correct)

### GPT-5.4 architectural recommendations:
1. Fact gating: `g = sigmoid(Wg * x); x = x + g * cross_attn(...)`
2. Metadata embeddings for facts (type, source, recency, hierarchy level)
3. Memory slot compression (compress facts → latent slots, attend to slots)
4. Cross-attention schedule: not every layer, selected mid layers only
5. Fact dropout during training (prevent overdependence on retrieval)
6. Distillation from 20B teacher (logits + hidden states)
7. Train/infer parity tests (full prefill vs incremental decode)

## 7. Local Model Landscape (2026-04)

### Available GGUF models and their architectures:

| Model | Arch | SSM | SWA | Context | RoPE | Key insight |
|-------|------|-----|-----|---------|------|-------------|
| GPT-OSS 20B | TransMamba | implied | **128** | 131K (YaRN 32x from 4K) | 150K base | Tiny SWA → needs tool reminder in window |
| Gemma 4 31B | gemma4 | no SSM | **1024** | 262K | 1M + 10K (dual) | SWA hybrid, no Mamba |
| Gemma 4 26B-A4B | gemma4 | no SSM | **1024** | 262K | 1M + 10K (dual) | MoE variant |
| Qwen 3.5 35B-A3B | qwen35moe | **Mamba** | no SWA | 262K | 10M base | Full attention + SSM, best for KV offload |
| Qwen 3.5 9B | qwen35 | **Mamba** | no SWA | 262K | 10M base | Smaller, same arch |

### KV cache implications:
- GPT-OSS (SWA=128): KV cache ~constant, offload pointless
- Gemma 4 (SWA=1024): KV grows for full-attn layers, offload useful
- Qwen 3.5 (no SWA): KV grows linearly, offload most valuable
- RoPE freq_base 10M (Qwen): positions work natively at long range, no YaRN degradation

## 8. Related Research

### User's prior work:
- **cogniversum_v2**: RetNet + hierarchical EMA memory (α=0.95..0.9999)
  - Proved: streaming stability over 10+ steps, no catastrophic forgetting
  - Limitation: tactile plateau at 26.7%, vocabulary imbalance
  - Key: EMA + explicit buffers + surprise-triggered consolidation

- **cognifusion_llm**: ParallelRetention + ACT (Adaptive Computation Time)
  - Achieved: 15.6 PPL in <24h training
  - Key finding: α∈[0.9,0.99] stability > α=0.999 max retention
  - "Stability and selectivity beat maximum theoretical retention"

- **FlashHadamard** (clustered_pg): no-codebook vector quantizer
  - 91% recall@10 at 5.7ms on 103K vectors
  - KV memory routing: tail-similarity REFUTED, random beats sketch
  - Selective memory track: PARKED pending coverage/diversity routing

### External:
- **MatMul-Free LLM** (HGRN2 + BitLinear): O(1) per token, ternary weights
  - Max tested: 2.7B params, no tool-calling model exists
  - Same memory decay limit as Mamba (h = f*h + i, diagonal A)
  - Potential for efficiency if someone trains 20B+

- **RETRO** (DeepMind): 7.5B + retrieval = 25x smaller than GPT-3
- **Atlas** (Meta): T5 770M + Contriever = SOTA on knowledge tasks
- **FiD**: T5-base 220M decoder with cross-attention to passages

## 9. Key Insight

The name: **Cogni*formerus*** = Cogni(versum) + (trans)formerus.
Hierarchical memory from cogniversum research attached to a transformer.

The model architecture is not "small LLM trying to be big LLM".
It is: **memory-augmented generator** where:
- Generator (decoder) handles syntax and tool format (small, fast)
- Memory (Cogniformerus) handles knowledge and facts (unbounded, precise)
- Encoder (nomic BERT) bridges both spaces (dual-use, already running)

The generator doesn't need to memorize — it needs to USE memory.
This is fundamentally different from scaling up parametric knowledge.
