# Wan long-key self-attention tuning

The experiment isolates prepared MPSGraph self-attention at B=1,
Q=K=36,608, H=40, D=128 from input initialization and output validation.
The baseline query chunk is 512; candidates are 128, 256 and 1,024.
Only independent query rows are split. Every query retains all keys and values.
The unchanged gate is `abs(actual-reference) <= 2e-5 + 2e-5*abs(reference)`,
with finite operands. The production defaults are unchanged by this experiment.

## Admission and measurement

`examples/wan_s2v_self_attention_benchmark.cr` fills nontrivial Q/K/V on GPU
using a deterministic integer avalanche and dyadic F32 values. Input generation
has its own terminal boundary. Each attention arm contains only the existing
chunk workspace, its GPU copies/tail padding and one terminal wait. All output
validation runs in a separate diagnostic chain. Only compact comparison metrics
are read by the host; model activations remain on GPU.

The two prepared workspaces and their output buffers are allocated before timed
arms. Each is warmed before three measured pairs in AB/BA/AB order. Both arms
finish before either output is validated, including the reversed BA pair.
Candidate admission requires positive savings in every pair and an improved
median; it is not a default-selection decision. Encoding
must allocate zero explicit Metal buffers. Host wall time from attention encoding
through its terminal wait is the comparison metric. Summed continued-command
GPU intervals can overlap and cannot be interpreted as independent stage costs.

The PyTorch 2.8.0 F32 MPS SDPA oracle reimplements the existing Wan reference
adapter's layout and attention subset, with CPU fallback disabled. It does not
invoke that adapter or its separate manual-equation cross-check. Its independent
query chunks retain full K/V. The manifest records geometry, input formula,
source/generator/output hashes and runtime; native validation checks all
187,432,960 output values against it. Manifest/source consistency is local
provenance, not independent attestation or a complete model quality gate.

All runs are serialized with a 15% system free-memory floor and per-process RSS
guard. Explicit buffers are admitted before allocation; the opaque MPS reserve
is a heuristic, not a bound on driver/private allocations. A fully materialized
F32 score tile would occupy 0.698/1.396/2.793/5.586 GiB for the four chunk sizes;
these are risk estimates, not measured MPS peaks. The guard's half-second tick
counter is not wall time or a strict real-time timeout.

The older video process is paused only for serial probes, with a cleanup trap
that resumes it. CPU STOP alone does not drain its submitted GPU work. A read-only
AGX client observer requires three stable submission/completion-counter samples
spaced by two seconds after STOP. This establishes observed target-client
quiescence, not system-wide GPU idleness. Other applications remain untouched.

## Whole-block discriminator

`examples/wan_s2v_full_block_benchmark.cr --comparison=self-chunks
--candidate-self-chunk=N` pairs baseline 512 with a candidate using real BF16
block-0 weights and identical synthetic GPU inputs. The text cross-attention
setting remains 320 in both arms. Each arm retains one terminal wait and zero
explicit encode allocations. Every candidate output is compared with the same
pair's baseline output at the unchanged strict gate; this is not a separate
full-geometry upstream block oracle.
The original cross-segmentation mode still requires bit identity.

A standalone attention improvement admits a candidate for whole-block testing;
it does not establish a full-video speedup. The block excludes the audio injector.
The known strict 40-layer stack drift remains recorded separately in
`video-numerical-drift.md`; this experiment cannot erase that rejection.

## Evidence and refresh

External logs, fixtures, binaries and control receipts live under
`/opt/mldata_ssd/cogni-ml/video-scene-20261019/evidence/wan-self-attention/`.
Refresh results after changes to implementation/lifetimes, weights or inputs,
geometry, precision, device/runtime, or GPU queue isolation. Results and the
limits of bottleneck attribution are recorded below.

The fixture pins PyTorch 2.8.0 and the Wan attention source at commit
`1ea34ff48f87168174e12956e200b1d908b1c5ff`. Its expected-output SHA-256 is
`8bf37c04b5effec694ec888a0a3c052160e827fcb7e0fb3603f5bfd939a709ef`;
the manifest SHA-256 is
`7586da395e4acedcd667639c91350119135183a0ed692fa994faeb683c6b850a`.
The full-block checkpoint is the previously qualified BF16 block-0 extraction:
1,017,447,344 bytes, SHA-256
`c172519063c1c34fb36f7f930429ca7db655a0e335524d7487525db1df5ad220`.

Build with the external scratch directory and the existing Metal bridge objects:

```sh
export TMPDIR=/opt/mldata_ssd/cogni-ml/video-scene-20261019/tmp
export CRYSTAL_CACHE_DIR=/opt/mldata_ssd/cogni-ml/crystal-cache
make BUILD_DIR=/opt/mldata_ssd/cogni-ml/video-scene-20261019/build/wan-block-benchmark \
  /opt/mldata_ssd/cogni-ml/video-scene-20261019/build/wan-block-benchmark/wan_s2v_self_attention_benchmark \
  /opt/mldata_ssd/cogni-ml/video-scene-20261019/build/wan-block-benchmark/wan_s2v_full_block_benchmark
```

After serializing GPU access and establishing the memory guard, create the
oracle and run each candidate separately:

```sh
env PYTORCH_ENABLE_MPS_FALLBACK=0 \
  python examples/support/wan_s2v_self_attention_benchmark_reference.py \
  --output=/path/to/reference --query-chunk-rows=128
env COGNI_RUN_SAFE_MIN_FREE_PCT=15 RUN_SAFE_PASSTHROUGH_STDIO=1 \
  scripts/run_safe.sh /path/to/wan_s2v_self_attention_benchmark 600 16384 \
  --fixture=/path/to/reference --candidate=128
env COGNI_RUN_SAFE_MIN_FREE_PCT=15 RUN_SAFE_PASSTHROUGH_STDIO=1 \
  scripts/run_safe.sh /path/to/wan_s2v_full_block_benchmark 900 20480 \
  --weights=/path/to/wan-official.weights.safetensors \
  --comparison=self-chunks --candidate-self-chunk=128 --pairs=3
```

Use a PyTorch build with MPS support for the oracle, with CPU fallback disabled.
Repeat the standalone command with 256 and 1,024. The `run_safe.sh` timeout
argument uses its existing tick semantics; it is not a strict wall deadline.
Require every warmup/measured validation, all three pairs, the injected-error
control and a successful process exit before admitting a result. The discarded
`self-chunk-128-v1.log` predates the current-pair BA comparison fix and is not
selection evidence.

## Measured results: 2026-10-20

The machine was an Apple M2 Max with 64 GiB unified memory on macOS 26.6.2.
Each row below is a separate AB/BA/AB cohort with its own 512-row baseline;
the table is not a direct randomized comparison between candidate sizes.
Times are host wall seconds from encoding through terminal completion,
excluding input fill and output validation.

| Candidate rows | Paired 512 median | Candidate median | Candidate time change | Pair wins |
| ---: | ---: | ---: | ---: | ---: |
| 128 | 25.4884 s | 16.9625 s | -33.45% | 3/3 |
| 256 | 23.5564 s | 19.4015 s | -17.64% | 3/3 |
| 1,024 | 26.2424 s | 28.3771 s | +8.13% | 0/3 |

All eight warmup/measured validations per cohort check every one of the
187,432,960 outputs against the F32 MPS oracle. All pass, with no nonfinites,
maximum absolute error `1.8328428e-6` and maximum normalized gate error
`0.08037424`. Candidates are bit-identical to their current pair's 512-row
baseline. All three injected-error controls reject exactly one changed value;
all processes exit successfully. Attention encoding allocates zero explicit
Metal buffers and each arm has one terminal wait.

GPU input setup took 0.0761/0.0919/1.1217 s in the 128/256/1,024 cohorts;
separate validation chains took 0.0374–1.0665 s across all cohorts. These costs
do not explain the attention intervals above. The standalone oracle's 11.8721 s
is from a different harness and is not a matched native/PyTorch speed comparison.

The strongest within-run candidate, 128, was then checked inside the complete
block-0 forward, keeping cross segmentation at 320:

| Pair order | 512 rows | 128 rows | Saved |
| --- | ---: | ---: | ---: |
| AB | 32.7227 s | 26.3915 s | 6.3312 s |
| BA | 36.0253 s | 27.0151 s | 9.0102 s |
| AB | 36.8494 s | 30.0596 s | 6.7898 s |
| Median | 36.0253 s | 27.0151 s | 25.01% lower median |

Warmup and all three paired block outputs are bit-identical across all
187,432,960 values. The corruption control rejects one changed value and the
process exits successfully. Both arms measure one terminal wait and zero
explicit Metal-buffer allocations during encoding. The block source preserves
asynchronous continuations without activation readbacks or intermediate host
waits; the corresponding emitted zero fields are source-supported labels,
not separate runtime counters. Segment counts increase from 589 to 624 at 128,
so fewer command segments are not a reliable performance proxy here.

The matrix and whole-block memory samples both bottom out at 31%, above the
configured 15% floor. The sampler is observational; `run_safe.sh` provides the
existing stop guard. The older video process was resumed after each experiment.
The correlated mini-Quadrumvirate adversary verdict is ROBUST for these bounded
comparisons. Keep 128 as the next pipeline candidate; production defaults and
the already-running frozen video executable remain unchanged. Three pairs do
not prove global optimality, and there is no full-video speed or quality claim.

External receipts (SHA-256):

| Receipt | SHA-256 |
| --- | --- |
| `self-chunk-128-v2.log` | `46d7c5d9849c602db814cf97ede83e48e815e0b8ac5c11826103a5c30672ce4c` |
| `self-chunk-256-v2.log` | `f532e8914a9485d3df24b0cd1d7b484c8e8fcee217afee264c881ebeffeba135` |
| `self-chunk-1024-v2.log` | `f20dcf031e84b7c62ac5d9b8946e9aeefa491b2f1f0cf979a6b6c51c284db4d8` |
| `full-block-128-v1.log` | `d6cd15930da1ee7a66dab728beda29328cebb259c5200cea59fc145f7f8dcd9f` |

`results-summary-v2.json` and `full-block-summary-v1.json` are convenience
summaries. Admission rests on the complete raw records, controls and successful
exits, with independent recomputation of pair membership and statistics.
The focused attention spec passes two examples with both chunk sizes, tail
padding, changed-input replay and the pinned upstream fixtures. It is not a
fresh complete-stack parity check.

## Cost map and limits of attribution

For this geometry, the block topology implies the following arithmetic counts.
These are source-derived work estimates, not measured stage durations or DRAM
traffic. Bias, normalization, RoPE, softmax, GELU and residual work are excluded.

| Work per block | TFLOP | Calls per block | Calls per 40-step CFG request |
| --- | ---: | ---: | ---: |
| Self SDPA QK + PV | 27.446 | 1 logical attention | 3,200 |
| Six full-row square projections | 11.516 | 6 | 19,200 |
| FFN input + output projections | 10.364 | 2 | 6,400 |
| Cross SDPA QK + PV | 0.384 | 1 logical attention | 3,200 |
| Short-context K/V projections | 0.0537 | 2 | 6,400 |

The request multiplier assumes 40 solver steps, two CFG evaluations and 40
transformer layers. The twelve audio injectors add 960 injector evaluations
(and 4,800 projections); this benchmark excludes those injectors, conditioning,
VAE decode and packaging. It does not measure the entire request.

The frozen first-window pipeline explicitly uses `drop_motion_frames: true`:
21 latent video frames contribute 34,944 tokens, and the reference contributes
1,664, yielding 36,608. The delivered 80 video frames are a different temporal
representation. With motion framepack enabled, the same grid instead gives
39,104 rows, adding about 6.8% linear work and 14.1% self-attention arithmetic.
Results must be refreshed for that continuation geometry.

| Query chunk | Graph calls | Q/output blits | Explicit chunk scratch | Hypothetical F32 score slab |
| ---: | ---: | ---: | ---: | ---: |
| 128 | 286 | 572 | 7.5 MiB | 0.698 GiB |
| 256 | 143 | 286 | 15 MiB | 1.396 GiB |
| 512 | 72 | 144 | 30 MiB | 2.793 GiB |
| 1,024 | 36 | 72 | 60 MiB | 5.586 GiB |

All choices copy the same valid Q/output volume, 1,430 MiB per block. Chunks
512 and 1,024 also pad 256 query rows; 128 and 256 divide the query count exactly.
Fewer graph calls might reduce scheduling overhead but need not reduce wall
time: the prior cross-attention experiment reduced segment count and lost in
all three whole-block pairs. Only measured paired results select the next move.

Each BF16 projection currently expands weights to F32 in one reusable workspace,
executes a prepared MPS linear operation, then adds bias. Across ten projections,
this implies about 1.96 GiB of logical weight conversion traffic per block. The
prior resident-F32 comparison did not establish a consistent speedup across the
four projection shapes. Keeping another complete F32 model is not justified by
that experiment.

The block already separates phases through asynchronous command continuation.
Its reported summed GPU root intervals may overlap. Adding them, or subtracting
standalone attention wall time from an unrelated whole-block sample, does not
establish exclusive phase costs. The `encode_cpu_seconds` field is wall time
inside encoding, not sampled CPU execution time; driver backpressure and GPU
overlap can contribute. A later diagnostic can attach stage IDs to
existing root boundaries and report interval envelopes, unions and overlap,
without adding waits; causal whole-block A/B remains the promotion gate.

DeepInfer's [published scope](https://github.com/deepseek-ai/DeepInfer#deepinfer)
is DeepSeek inference on NVIDIA SM100 and Ascend 950, including specialized
attention kernels and asynchronous scheduling. Those are useful design prompts,
not a Metal/Wan drop-in. Transferable hypotheses are reducing score traffic and
preserving asynchronous execution; its autoregressive KV replay and draft-token
techniques do not directly certify reuse of changing diffusion activations.

## SonicBoom: applicable ideas and next discriminators

The reviewed revision is
[`b74d683b19529dfec794c02aa18e3dcf66b6156b`](https://github.com/hardenedlinux/SonicBoom/tree/b74d683b19529dfec794c02aa18e3dcf66b6156b).
Its [published comparison](https://github.com/hardenedlinux/SonicBoom/blob/b74d683b19529dfec794c02aa18e3dcf66b6156b/README.md#numbers)
reports 1.13–1.20x steady-state decode on an RTX 3050 for two quantized LLMs;
Qwen at KV depth 512 is 1.01x. Its generated prefill attention remains 3.12x
slower than llama.cpp. These are author measurements, not a Wan/Metal result.

The most relevant attention ideas are staged K/V reuse and conditional output
accumulator rescaling. The
[code generator](https://github.com/hardenedlinux/SonicBoom/blob/b74d683b19529dfec794c02aa18e3dcf66b6156b/core/compiler/lib/AttentionCodegen.cpp#L945-L1028)
skips the accumulator multiplication when the online row maximum does not grow
and the factor is exactly one. It retains the normalizer's FMA and reduction
association. That algebraic move is a candidate for the native F32 Metal kernel,
not evidence that copying its FP16 probability path would preserve our gate.
Any implementation must first pass all full-output fixtures and then whole-block
timing. No SonicBoom code is incorporated by this change.

Its [runtime tuner](https://github.com/hardenedlinux/SonicBoom/blob/b74d683b19529dfec794c02aa18e3dcf66b6156b/core/src/nn/cuda_elementwise.cu)
illustrates warmup-only selection, device/operator/version-specific cache keys
and a known-safe fallback. Numerical eligibility must remain a separate gate.
The timing inputs are zero Q/K/V: with conditional rescaling, that may favor
the branch that skips accumulator work. This is a benchmark-proxy risk, not a
demonstrated performance defect. Retain our nontrivial inputs when tuning.
The [benchmark harness](https://github.com/hardenedlinux/SonicBoom/blob/b74d683b19529dfec794c02aa18e3dcf66b6156b/tools/bench.sh)
also has alternating two-build comparisons and an A/A noise control. Its
canonical SonicBoom/llama.cpp runs use separate series, so do not describe the
published cross-runtime numbers as alternating paired measurements.

For Wan, the next major discriminator is K/V layout reuse. The current MPS
bridge places Q/K/V transposes and opaque SDPA inside each repeated chunk plan.
The [PyTorch 2.8 MPS path](https://github.com/pytorch/pytorch/blob/v2.8.0/aten/src/ATen/native/mps/operations/Attention.mm)
used by the oracle instead executes an explicit QK/scale/softmax/PV graph with
contiguous head-major K/V. The compiled MPSGraph's physical memory behavior is
opaque, so repeated physical transposes are not established. Test a matched
explicit graph and one GPU K/V preparation per block against the current plan;
then compare a native F32 tiled attention route with K/V reuse. Keep the same
strict gate, prepared buffers, terminal boundary and whole-block discriminator.
Stage IDs on existing asynchronous roots can separately test submission and
projection hypotheses; the current aggregate does not rank non-attention costs.
