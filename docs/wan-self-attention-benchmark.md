# Wan long-key self-attention tuning

The experiment isolates prepared MPSGraph self-attention at B=1,
Q=K=36,608, H=40, D=128 from input initialization and output validation.
The query-chunk experiment uses baseline 512 and candidates 128, 256 and 1,024;
the later K/V-layout ablation holds both arms at 128.
Only independent query rows are split. Every query retains all keys and values.
The unchanged gate is `abs(actual-reference) <= 2e-5 + 2e-5*abs(reference)`,
with finite operands. The production defaults are unchanged by this experiment.

## Explicit-graph frontier

Status: optional experiment implemented. Standalone strict fixture checks pass
and the three timed pairs improve; the whole-block warmup fails the unchanged
numerical gate before measured pairs. Promotion is rejected and defaults remain
unchanged. Further GPU diagnosis is suspended after the memory incident below.
Both arms use query chunk 128, cross segments 320, and the same caller-owned
prepared head-major K/V pair. Each encode refreshes that pair on GPU; each arm
finishes at its existing terminal wait before the other arm may reuse it.
Sharing across overlapping workspace encodes is forbidden by the caller
contract; per-workspace lifetime checks cannot enforce cross-workspace ownership.

The opt-in `explicit_qk_softmax_pv` plan/workspace argument requires head-major
K/V and defaults to false. Its graph follows PyTorch 2.8.0's F32 MPS math path:
transpose the final two K dimensions, QK matmul, add the upstream `1e-20f`
macOS 15+ workaround, multiply by F32 `1/sqrt(head_dim)`, softmax over keys,
then PV matmul. Only the context tensor is exported. This matches the selected
operation sequence, not PyTorch's dispatch selection or opaque compiler lowering.
The pinned source is [PyTorch Attention.mm](https://github.com/pytorch/pytorch/blob/v2.8.0/aten/src/ATen/native/mps/operations/Attention.mm),
SHA-256 `167ee89bddb489c154b5d5ca72cf042745443f557d624fc48175f2d29f45a351`.

The strongest failure modes are changed reduction rounding, opaque score-memory
growth, and accidental sharing before completion. Before production changes,
new API/fixture specs must fail against the old implementation. Qualification
requires all six pinned fixture cases, changed-input/tail/reuse guards, the
full-geometry strict upstream oracle, and the real BF16 whole-block regression.
Only then run three AB/BA/AB timed pairs at standalone and whole-block scope.
All three savings and the median must improve for performance admission.
No tolerance change, new host wait/readback, or explicit encode allocation is
admitted. The default SDPA path is the rollback.

A score tile at B=1,H=40,Q=128,K=36,608 occupies 749,731,840 bytes.
Preflight adds three such tiles to the existing opaque reserve before graph
creation. This conservative heuristic is not a driver-allocation bound. The
standalone graph-comparison cap is 12 GiB; other standalone modes retain 8 GiB.
Adding the three tiles makes the whole-block estimate 16.012 GiB, so that
comparison alone has an 18 GiB cap; existing block modes retain 16 GiB.
Explicit shared K/V is counted once.
The executed GPU runs were serial under the configured 15% `memory_pressure -Q`
guard and the paused-client quiescence observer. That percentage and the RSS
limit did not bound physical footprint or swap; they cannot certify 15% unused
physical memory. A numerical or memory failure rejects promotion and preserves
the failed probe for analysis. New heavy runs require stronger memory admission.

## Head-major K/V frontier

Status: implemented as an optional path; bounded correctness checks pass, but
the three-pair performance admission fails. The candidate prepares shared F32
K/V from BTHD to BHTD with one fused GPU dispatch
at the beginning of every chunk-workspace `encode` call. All query chunks in
that call consume the same prepared buffers. Q and output layouts, SDPA math,
precision and the strict gate above remain unchanged. This tests a layout and
reuse hypothesis; the existing opaque MPSGraph executable does not establish
how many physical transposes its compiler currently performs.

The caller allocates and owns both extra full-size K/V buffers before encoding;
their combined 1,499,463,680 bytes at this geometry count toward admission and
workspace scratch. The optional pair must be disjoint from all original and
chunk buffers, remain live until terminal completion, and cannot be shared by
in-flight chains. Each encode refreshes both inputs; there is no cross-call
activation cache. Legacy workspace construction is the rollback and remains the
default.

The falsifiers are exact transpose-layout checks, changed-K/V ABA reuse in one
chain and after completion, alias/extent/lifetime rejection, and strict full
output comparison against the pinned reference. Performance admission requires
three AB/BA/AB pairs of the whole real-weight block with both arms using 128-row
query chunks and 320-row cross segments. Preparation is inside each timed arm;
allocations, input fill and output checking remain outside. Every pair must
improve and the median must improve, with zero explicit encode allocations
and one terminal wait. Historical runs used the configured 15% pressure guard;
new runs also require the stronger admission described below. A failed numerical
gate or a slower whole block rejects promotion without changing tolerances.

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

Runs used a configured 15% `memory_pressure -Q` threshold and a process-tree RSS
guard. Neither is a bound on physical footprint or swap. Explicit buffers are
admitted before allocation; the opaque MPS reserve
is a heuristic, not a bound on driver/private allocations. A fully materialized
F32 score tile would occupy 0.698/1.396/2.793/5.586 GiB for the four chunk sizes;
these are risk estimates, not measured MPS peaks. The guard's half-second tick
counter is not wall time or a strict real-time timeout.

During the recorded cohorts the older video process was paused for serial probes,
with a cleanup trap that resumed it. It was subsequently terminated with explicit
user authorization after the memory incident below; automatic restart is forbidden.
CPU STOP alone does not drain submitted GPU work. A read-only
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
it does not establish a full-video speedup. The explicitly requested K/V-layout
experiment also runs the whole-block discriminator if standalone timing is
inconclusive. `--comparison=kv-layout --candidate-self-chunk=128` holds both
query chunks at 128 and both text cross-attention segment groups at 320. Only
the candidate prepares and uses head-major K/V; preparation is charged inside
its block encode. The block excludes the audio injector.
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

Historical build command, retained as provenance. Builds and GPU probes are
currently suspended pending stronger memory admission; this is not a replay
procedure.

```text
export TMPDIR=/opt/mldata_ssd/cogni-ml/video-scene-20261019/tmp
export CRYSTAL_CACHE_DIR=/opt/mldata_ssd/cogni-ml/crystal-cache
make BUILD_DIR=/opt/mldata_ssd/cogni-ml/video-scene-20261019/build/wan-block-benchmark \
  /opt/mldata_ssd/cogni-ml/video-scene-20261019/build/wan-block-benchmark/wan_s2v_self_attention_benchmark \
  /opt/mldata_ssd/cogni-ml/video-scene-20261019/build/wan-block-benchmark/wan_s2v_full_block_benchmark
```

Historical oracle/candidate invocations follow. The old guard shown here cannot
bound footprint or swap; do not execute these commands as an admission procedure.

```text
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

The oracle used PyTorch with MPS support and CPU fallback disabled; standalone
candidates also included 256 and 1,024. The `run_safe.sh` timeout
argument uses its existing tick semantics; it is not a strict wall deadline.
Require every warmup/measured validation, all three pairs, the injected-error
control and a successful process exit before admitting a result. The discarded
`self-chunk-128-v1.log` predates the current-pair BA comparison fix and is not
selection evidence.

## Query-chunk measurements: 2026-10-20

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

## Head-major K/V measurements: 2026-10-20

The optional path prepares K/V once per workspace encode and reuses it across
all 286 query chunks. Both arms use 128 rows and the full block keeps cross
segments at 320. Preparation is included in each candidate's timed encode;
allocation, input initialization and comparison are outside. These are fresh
within-run comparisons on the machine above. The earlier chunk-size cohorts
have different host state and are not controls for this change.

| Attention pair | Legacy K/V | Prepared K/V | Saved |
| --- | ---: | ---: | ---: |
| AB | 15.372098 s | 15.333370 s | 0.038728 s |
| BA | 15.809275 s | 15.012460 s | 0.796815 s |
| AB | 13.870758 s | 14.484962 s | -0.614203 s |
| Median | 15.372098 s | 15.012460 s | 2.34% lower median |

| Whole block pair | Legacy K/V | Prepared K/V | Saved |
| --- | ---: | ---: | ---: |
| AB | 18.227773 s | 18.091101 s | 0.136672 s |
| BA | 18.224448 s | 17.990366 s | 0.234082 s |
| AB | 18.432724 s | 19.068676 s | -0.635952 s |
| Median | 18.227773 s | 18.091101 s | 0.75% lower median |

Both cohorts win only two of three pairs, so neither meets the predeclared
admission rule. The full-block result does not establish a stable speedup.
The path remains optional and production defaults are unchanged. The extra
prepared pair costs 1,499,463,680 bytes (1.396 GiB), making default promotion
particularly unjustified by this result. Opaque MPSGraph compilation prevents
attributing the result to an exact physical transpose count or exclusive cost.

All eight standalone warmup/measured outputs pass the unchanged pinned F32 MPS
upstream gate across all 187,432,960 values: no nonfinites or violations, max
absolute error `1.8328428e-6`, max normalized gate error `0.08037424`. Candidate
outputs are bit-identical to their native controls. Warmup and all three whole
block pairs are also bit-identical across every output value; both corruption
controls reject exactly one changed value. The block uses real BF16 block-0
weights with synthetic GPU activations and excludes the audio injector. It is
not a full-spatial upstream block oracle or a full-stack/video quality check.

The focused attention spec passes 5/0 examples, covering exact B=2 transpose
mapping, separately changed K/V, restored inputs, completed-chain reuse, and
alias/extent/in-flight guards. The official upstream BF16 block/injector
intermediate regression passes 6/0 examples at its smaller sequence geometry.
Every timed arm has one terminal wait and zero explicit encode MetalBuffer
allocations; production source adds no intermediate host wait or activation
readback. Private MPS allocations remain outside that counter. Sampled pressure
metric minima were 40% (focused spec), 58% (block regression), 34% (attention)
and 29% (whole block), above the configured 15% telemetry threshold, without
certifying physical headroom. The older video process was resumed after those
serial probes and was later terminated with explicit user authorization.

The correlated mini-Quadrumvirate adversary verdict is ROBUST for the stated
code/lifetime and bounded correctness claims; the speedup claim is VULNERABLE
and promotion is rejected. The known strict 40-layer drift remains open.

Historical head-major invocations, not a replay procedure:

```text
env COGNI_RUN_SAFE_MIN_FREE_PCT=15 RUN_SAFE_PASSTHROUGH_STDIO=1 \
  scripts/run_safe.sh /path/to/wan_s2v_self_attention_benchmark 1800 16384 \
  --fixture=/path/to/reference-v1 --comparison=kv-layout --candidate=128
env COGNI_RUN_SAFE_MIN_FREE_PCT=15 RUN_SAFE_PASSTHROUGH_STDIO=1 \
  scripts/run_safe.sh /path/to/wan_s2v_full_block_benchmark 1800 24576 \
  --weights=/path/to/wan-official.weights.safetensors \
  --comparison=kv-layout --candidate-self-chunk=128 --pairs=3
```

The old external `with-wan-paused.sh` controller hardcodes the terminated video
PID and must not be reused. A future replay needs the separate footprint/swap
admission gate and a fresh queue-isolation check.

The standalone DoD requires all eight strict oracle/baseline checks, zero
nonfinites, a rejected corruption control and successful exit. The whole-block
DoD requires warmup plus all three complete pair comparisons, the same control
and successful exit. Performance additionally requires every paired saving and
the median to improve; that separate gate fails here. Argument guards reject a
256-row K/V-layout comparison before GPU setup in both rebuilt executables.
Additional CPU probes accept the pinned manifest's native `B,Q,H,D` layout
through an intentionally missing-output boundary and reject a substituted
`B,K,H,D` fixture layout. These four checks initialize no GPU workspace.

External receipts in the evidence directory above (SHA-256):

| Receipt | SHA-256 |
| --- | --- |
| `head-kv-attention-v1.log` | `1a2205778e69b60e949118b447d895e0ba7d6e049b5961be9e2e05e558f82b16` |
| `head-kv-block-guard-v1.log` | `c803020bcb621b78d563cffe8e1b0dcae2262f75e68fa0bb5056b10861ad290c` |
| `head-kv-self-v1.log` | `8efaf1a242d715c9e81dd5eeb7ae930e807020883fb372f38deb54fbdf8e14b4` |
| `head-kv-full-block-v1.log` | `a1707a200cd84b7ce11fabcf8552113a6329879bc5feae11d6e2663f37578a02` |

`head-kv-qualified-summary-v1.json` records independently recomputed pair
statistics, memory samples, raw log hashes and measured source/binary hashes.
Its SHA-256 is
`16952d005c875c31937d531961d48588999e23e99d62636aa7e027b7e3775870`.
Raw v1 baseline metadata says `B,Q,H,D`; its correct K/V label is `B,K,H,D`
(Q=K=36,608 here). Committed benchmark emitters correct only that label.
The semantic input-manifest contract remains `B,Q,H,D` as measured.
A final formatter change only moves the FFI declaration's closing delimiter and
adds a trailing comma. The summary records exact reversal checks to measured
source hashes; measured executables are retained with the suffix
`-head-kv-measured-v1`. These cosmetic changes have CPU rebuild checks, not new
GPU timing receipts. The same implementation, runtime, input and queue-isolation
refresh triggers above apply.

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

For Wan, the head-major K/V ablation above did not establish a stable gain.
The legacy MPS bridge places Q/K/V transposes and opaque SDPA inside each
repeated chunk plan; the optional path prepares head-major K/V once per call.
The [PyTorch 2.8 MPS path](https://github.com/pytorch/pytorch/blob/v2.8.0/aten/src/ATen/native/mps/operations/Attention.mm)
used by the oracle instead executes an explicit QK/scale/softmax/PV graph with
contiguous head-major K/V. The compiled MPSGraph's physical memory behavior is
opaque, so repeated physical transposes are not established. The next hypothesis
was a matched explicit QK/scale/softmax/PV graph using the prepared K/V. Its
standalone result and whole-block rejection are recorded below. Native F32 tiled
attention with K/V reuse remains a separate candidate. Keep the same
strict gate, prepared buffers, terminal boundary and whole-block discriminator.
Stage IDs on existing asynchronous roots can separately test submission and
projection hypotheses; the current aggregate does not rank non-attention costs.

## Explicit-graph measurements: 2026-10-20

Both arms use the same caller-owned head-major K/V pair, refresh it inside every
timed encode, use query chunk 128 and finish serially at one terminal wait.
Cross segmentation remains 320 in the full block. Input fill, output validation,
buffer allocation and graph preparation are outside the timed arms.

| Standalone pair | MPSGraph SDPA | Explicit graph | Saved |
| --- | ---: | ---: | ---: |
| AB | 18.338896 s | 11.670051 s | 6.668844 s |
| BA | 17.060943 s | 11.678053 s | 5.382890 s |
| AB | 16.666125 s | 12.028706 s | 4.637419 s |
| Median | 17.060943 s | 11.678053 s | 31.55% lower median |

All eight standalone full-output checks pass the unchanged pinned PyTorch gate
over 187,432,960 values per check, with no nonfinites. All four explicit outputs
are bit-identical to this fixture oracle. The SDPA control differs from the
oracle by at most `1.8328428e-6`, normalized gate error `0.08037424`; candidate
versus SDPA also passes. The corruption control rejects exactly one changed
value. This is bounded evidence for these deterministic Q/K/V inputs, not a
general claim of bit identity to upstream on projected model activations.

The real BF16 block-0 experiment exits 1 during warmup: **5,085 / 187,432,960**
values violate the unchanged `2e-5 + 2e-5*abs(reference)` gate. Maximum absolute
error is `6.0796738e-5`, maximum normalized error `2.5658143`, and nonfinite count
is zero. There are 184,471,886 bit differences. The reference here is the same
run's native SDPA block, not a full-geometry upstream block oracle. The harness
stops before any measured arm records or corruption control. There are **zero
measured block pairs** and no admissible whole-block speedup result.

Arithmetic differences amplified by later projections are a hypothesis, not an
established cause. The next numerical discriminator is a separate same-input
trace at `self_attention`, then `self_projection` and `self_residual` if needed,
plus SDPA A/A replay to distinguish state reuse from arithmetic. Its external
generator was written but not executed; no trace or A/A result exists. Tolerances
remain unchanged, the default stays SDPA, and the known strict stack rejection
remains open.

The focused upstream attention/reuse/guard spec passes 7/0 examples. The existing
official BF16 block/audio regression passes 6/0 examples at its smaller fixture
geometry; it does not qualify the rejected full-shape graph substitution.
The old API fails the added spec at compile time, and both rebuilt benchmark
argument guards reject unequal query chunks before GPU initialization. Every
standalone timed arm records one terminal wait and zero explicit MetalBuffer
allocations inside encoding. Model activations remain on GPU; compact diagnostic
metrics are read only after separate validation chains.

The explicit path has 573 command segments versus 48 for SDPA. Its host time
inside encoding is about 10.3–10.8 s, versus roughly 0.05–0.64 s for SDPA, and
is already included in the wall times above. Encoding may overlap submitted GPU
work or block inside framework calls; this is not an exclusive CPU cost. Summed
continued-command GPU intervals can overlap and must not be added to wall time.

Preflight explicit-plus-reserve estimates are 10,410,262,576 bytes for standalone
and 17,192,380,948 bytes for the block, beneath their respective 12/18 GiB caps.
These are not measured driver peaks. The minimum sampled `memory_pressure -Q`
percentages are 30% (focused spec), 30% (block regression), 31% (standalone) and
27% (full block). They did not establish safe physical headroom.

After these probes, the user reported renewed memory growth in the resumed older
full-video process (PID 34536), then explicitly authorized its termination at
step 19/40. Before termination, `vmmap -summary` reported footprint 32.1 GiB and
30.9 GiB writable memory swapped out, despite `ps` RSS around 45 MiB.
`memory_pressure -Q` reported 76% nearby, illustrating why that guard plus RSS
is insufficient. The process and its AGX client were confirmed absent afterward.
Other projects were untouched. No further GPU probe or build was launched;
strengthening footprint/swap admission precedes any new heavy run. These
observations do not establish a leak or attribute total host memory to Wan.

The historical CLI settings were `--comparison=attention-graph --candidate=128`
for standalone attention and `--comparison=attention-graph
--candidate-self-chunk=128 --pairs=3` for the block. They are provenance, not a
replay procedure: the former pressure/RSS guard is insufficient. Do not replay
these runs until a separate footprint/swap admission gate has been implemented
and checked, followed by queue isolation. No replacement admission gate has
been implemented by this change.

Raw logs are `explicit-graph-attention-v1.log`,
`explicit-graph-block-guard-v1.log`, `explicit-graph-self-v1.log` and
`explicit-graph-full-block-v1.log` in the evidence directory above. Each has
separate memory and target-client observation receipts. The independently
recomputed `explicit-graph-qualified-summary-v1.json` records the rejected
whole-block qualification, descriptive standalone result and raw hashes;
its name does not imply whole-block admission. `explicit-graph-bindings-v1.json`
pins the five measured source files, bridge object and four executables against
base commit `060c4b37e46d0953339a5d91765f20e043b0c4bc`. All remain unchanged
after measurement. `user-memory-release-20261020.json` confirms termination.
The subsequent CPU receipt check verifies all ten bound source/binary files,
all twenty raw-receipt hashes and the reported medians; it performs no GPU or
build operation. Its receipt is `explicit-graph-cpu-receipt-check-v1.json`.
The updated summary SHA-256 is
`65ab6514dec0dff77b6b676a510cab9ac328ec5c2942e408b6bc8c225ad3bb78`.

Correlated mini-Quadrumvirate review supports the bounded standalone comparison
and rejection decision. Numerical admission of the whole-block candidate is
BROKEN; a whole-block performance claim has no measured evidence. Refresh all
qualification after source, input, geometry, device/runtime, memory admission or
queue-isolation changes. Full-video speed and quality remain unmeasured here.
