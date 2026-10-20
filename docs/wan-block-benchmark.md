# Full-size Wan block performance probes

This experiment keeps the production default at 320 cross-attention groups per
segment. `WanS2VBlock.encode` and `encode_with_audio` accept an optional
`cross_attention_groups_per_segment` for the tiled text cross-attention chain.
The override changes command boundaries, not a query's ascending key reduction.
It does not change self-attention. Nonpositive values fail before encoding; a
nondefault tiled override requires an execution chain.

The target geometry is B=1, N=36,608, C=5,120, FF=13,824, H=40, D=128 and
text K=512. There are 183,040 query/head groups. Segments of 320 and 1,024 yield
572 and 179 segments respectively. The previously rejected 4,096-group dispatch
used **long-key self-attention K=36,608**; it is not a measured cross-attention
limit.

## Isolation and checks

The probes run in a separate worktree based on `2210c5cc`. The older full-video
process is paused only while a serial GPU experiment runs and resumed by an
external cleanup trap. Its frozen checkout and admission receipt stay intact.
`scripts/run_safe.sh` checks the 15% system memory floor and a per-process-tree
RSS budget, with a nominal timeout. Its elapsed-time counter counts half-second
loop ticks plus monitoring work, so it is not a real-time deadline or latency
measurement. The full-block process took 237.12 real seconds by `/usr/bin/time`,
although the guard printed approximately 108 seconds. The guards are unchanged.
No unrelated user process is changed.

Pausing the CPU submitter does not drain GPU work already submitted by it.
A rapid cross-attention repeat had large declining times and is retained as
confounded evidence. The reported repeat waits 40 seconds on CPU after the pause,
before benchmark setup; this adds no synchronization to the model path.
Settling reduces this plausible interference source but does not independently
prove an idle GPU. The original full-video process was resumed after each probe.

GPU model operations retain caller-owned buffers and prepared plans, with no
intermediate host waits or activation readbacks. Benchmark setup and terminal
validation are outside production execution. Whole-arm wall time is the
authoritative latency metric. A continued command chain's summed GPU intervals
can overlap and must not be added up as independent stage costs.

The official BF16 block-0 fixture passes both segment settings: 6 examples,
zero failures/errors/pending per run. It checks intermediate results at full
model width with B=2 and sequence length 2, not full spatial geometry. Its
upstream source is Wan2.2 `1ea34ff48f87168174e12956e200b1d908b1c5ff`, checkpoint
revision `dab4e9c55bbe4c8c4d03db1c2c98c7f0ac9c454b`. The gate remains
`abs(actual-reference) <= 2e-5 + 2e-5*abs(reference)` with finite values.

## Full-geometry cross attention

`examples/wan_s2v_block_benchmark.cr` compares tiled segments of 320 and 1,024
with the existing prepared MPSGraph attention workspace using query chunks of
512. All use B=1, Q=36,608, K=512, H=40 and D=128. Each path is warmed before
three measured rounds. The order rotates ABC/BCA/CAB, balancing each path's
position over the three rounds. Smaller repetition counts do not provide this
balance.

`examples/support/wan_s2v_block_benchmark_reference.py` produces the independent
F32 MPS SDPA output with PyTorch 2.8.0, CPU fallback disabled. A shared integer
avalanche formula initializes distinct, nontrivial Q/K/V inputs; native input
initialization and all attention arithmetic run on GPU. The oracle chunks only
independent query rows, retaining full K/V. Native B,Q,H,D output is compared on
GPU with contiguous B,H,Q,D reference output across all 187,432,960 values.
Only compact validation metrics are read at a separate terminal boundary.

The fixture records source and generator digests, geometry, layouts, formula,
salts, runtime version and expected-output digest. The native harness checks
these before execution. This establishes local source consistency, not an
independent source attestation or a full-block upstream result. Recorded source
digests are Wan attention `23fe7c6f6e4065242d95e5e188cb2d1a16bd283f05dca7e7e878977158fcfdbc`
and the existing S2V reference adapter
`863f0020418f3da40caf6f2863dbef9ed40d147127c34b0748cf3415433333a3`.

The explicit-buffer preflight is 3,051,356,208 bytes, below the 4 GiB cap.
Validation runs outside timed arms and includes a separate one-value corruption
control. Every timed arm retains one terminal wait and performs zero explicit
Metal-buffer allocations. Setup and diagnostic terminal waits are not added to
the production block path.

The settled repeat gives the following medians of three whole-arm wall times:

| Cross-attention path | Seconds | Range, seconds | Command segments |
| --- | ---: | ---: | ---: |
| Tiled, 320 groups | 0.4307 | 0.3884–0.4390 | 572 |
| Tiled, 1,024 groups | 0.2771 | 0.2653–0.3039 | 179 |
| MPSGraph, 512 query rows | 0.1191 | 0.1167–0.1251 | 12–13 |

Thus the isolated 1,024 path is 1.55 times faster and MPSGraph is 3.62 times
faster than tiled 320 by the ratio of medians. Both tiled settings are
bit-identical. All paths pass the unchanged upstream gate across all values;
maximum absolute error is 2.6226044e-6 for tiled and 2.3841858e-6 for MPSGraph.
The comparator rejects the deliberate one-value corruption. Minimum sampled
free memory was 61% in this native repeat and 39% in the MPS oracle run.

The initial cross run also ranked MPSGraph, 1,024, then 320, with medians
0.1265, 0.3005 and 0.4454 seconds. Its warmup output preceded a JSON-format
repair, while the timed values and gates passed. The rapid repeat between the
initial and settled runs also passed all arithmetic checks, but its times fell
sharply within the run; those times are excluded from the table, not discarded.
Absolute latency remains sensitive to host/device conditions.

## Full-block comparison

`examples/wan_s2v_full_block_benchmark.cr` runs real official block-0 BF16 weights
at the full target geometry, with GPU-generated synthetic F32 activations,
prepared MPS projections, query-chunked self attention and tiled cross attention.
It excludes the audio injector and performs no video-quality qualification.
It warms both segment settings, then measures AB/BA/AB pairs. Every complete
output is compared outside the timed arms and must be bit-identical, finite and
within the unchanged strict tolerance. A deliberate one-value corruption must
be rejected.

The preflight estimate is 13,459,450,388 bytes under a 16 GiB admission budget,
including 1 GiB reserved for opaque MPS allocations. That reserve is a heuristic,
not a bound on MPS internals; the external memory-floor and RSS guards still
apply. The checkpoint sidecar is SHA-256
`c172519063c1c34fb36f7f930429ca7db655a0e335524d7487525db1df5ad220`.
It contains 39 block/injector bindings; unused injector plans use tiny scratch.

`host_wall_seconds` includes block encoding and its single terminal wait.
The historical field name `encode_cpu_seconds` measures wall time inside the
encoding calls, including driver backpressure; it is not CPU execution time.
Comparison kernels and their terminal waits are outside both intervals.
The full-size equivalence claim is relative to the native 320 baseline. Absolute
upstream intermediate coverage remains the separate full-width, short-sequence
official block gate and full-geometry attention primitive gate above.

Measured full-block wall time, seconds:

| Cross segment groups | Pair 1 | Pair 2 | Pair 3 | Median | Command segments |
| --- | ---: | ---: | ---: | ---: | ---: |
| 320 | 26.5768 | 27.3868 | 28.9348 | 27.3868 | 589 |
| 1,024 | 27.7389 | 28.1977 | 29.7441 | 28.1977 | 196 |

The 1,024 arm is slower in all three pairs, by 4.37%, 2.96% and 2.80%.
Warmup and all three comparisons are bit-identical over 187,432,960 finite
output values. The corruption control is rejected, and minimum sampled free
memory is 30%, above the configured 15% floor. These observations support
keeping 320 as the default. They do not establish a causal slowdown under all
conditions: time drifts upward, the sample is small, and the test is block 0
with synthetic activations. Fewer command segments and faster isolated cross
attention have not produced a measured full-block win. MPSGraph cross attention
has not yet been measured inside this complete block.

## Projection cache diagnostic

`examples/wan_s2v_projection_benchmark.cr` measures the existing BF16-to-F32
expansion plus prepared MPS multiplication against the same multiplication
consuming an already expanded F32 stage. It uses synthetic dyadic GPU inputs at
production matrix dimensions. This is a cache upper-bound diagnostic, not a
proposal to retain an F32 copy of the whole model or an upstream parity claim.

After warming both paths, three pairs alternate AB/BA/AB. Each timed arm has one
terminal wait and zero explicit encode-time Metal-buffer allocations. Separate
GPU scans compare every output exactly; all four shapes pass. Injecting one
changed output value is rejected by both bit and strict-tolerance comparisons.

Measured on Apple M2 Max, median seconds:

| Projection | Rows | BF16 expansion + MPS wall | Resident F32 MPS wall | Block calls per request |
| --- | ---: | ---: | ---: | ---: |
| Hidden square | 36,608 | 0.3514 | 0.3246 | 19,200 |
| FFN input | 36,608 | 1.0192 | 1.1299 | 3,200 |
| FFN output | 36,608 | 1.2745 | 1.3307 | 3,200 |
| Cross K/V square | 512 | 0.0082 | 0.0084 | 6,400 |

Counts assume 40 denoise steps, two CFG passes and 40 blocks. Each block has six
full-N square projections, two short-K square projections, and the two FFN
projections. Audio-injector projections are additional. Multiplying isolated
medians by counts is not a full-request latency prediction.

The expansion/resident wall ratios change sign across pairs for every shape.
These data do not support promoting persistent F32 weights. The memory monitor
observed a minimum 28% free percentage across this serial projection suite and
the separate self-attention diagnostic; the configured floor stayed 15%.

## Next discriminating probes

1. Long-key self attention is the highest-priority cost candidate. A separate
   Q=K=36,608, H=40, D=128 diagnostic took 23.5103 seconds over its command chain,
   but that interval includes GPU input fills and a full-output scan. It is not
   a pure attention measurement or a measured fraction of the full block. Split
   those costs using existing GPU timing boundaries before comparing query
   chunk sizes 128/256/512/1,024, without intermediate host synchronization.
2. Integrate the already faster short-key MPSGraph path as an opt-in full-block
   candidate and repeat the complete-block and upstream gates. Its isolated
   3.62-times ratio is not an end-to-end prediction.
3. The six large square projections are the most frequent linear shape:
   19,200 calls per request. The two FFN projections are individually slower
   in the isolated diagnostic. Persistent F32 expansion alone did not win
   consistently; a future projection kernel must preserve BF16 storage and
   F32 accumulation and demonstrate a full-block gain.

Self and text cross attention each occur 3,200 times in the stated request.
Their Q*K arithmetic ratio is 71.5, but this count is not a runtime ratio.
Replaying prepared fixed-shape work is a separate host-submission hypothesis;
the current `encode_cpu_seconds` field includes GPU driver backpressure and
does not prove that CPU computation is the dominant cost.

## Reproduction and evidence

Build benchmarks with external scratch and the bundled Wan bridge objects:

```sh
SCRATCH=/opt/mldata_ssd/cogni-ml/video-scene-20261019
BUILD="$SCRATCH/build/wan-block-benchmark"
TMPDIR="$SCRATCH/tmp" CRYSTAL_CACHE_DIR="$SCRATCH/crystal-cache" \
  make BUILD_DIR="$BUILD" "$BUILD/wan_s2v_projection_benchmark" \
  "$BUILD/wan_s2v_block_benchmark" "$BUILD/wan_s2v_full_block_benchmark"
COGNI_RUN_SAFE_MIN_FREE_PCT=15 scripts/run_safe.sh \
  "$BUILD/wan_s2v_projection_benchmark" 600 10240 --operator=ffn-in --pairs=3
```

Run the GPU jobs serially with other owned submitters paused and settled,
retaining cleanup that resumes them. The MPS oracle requires an existing
PyTorch MPS environment; no dependencies are installed by these probes:

```sh
EVIDENCE="$SCRATCH/evidence/wan-block-benchmark"
MPS_PYTHON="$SCRATCH/../drama3-fish-reference-20261015/venv/bin/python"
COGNI_RUN_SAFE_MIN_FREE_PCT=15 scripts/run_safe.sh "$MPS_PYTHON" 900 12288 \
  examples/support/wan_s2v_block_benchmark_reference.py \
  --output="$EVIDENCE/cross-full-v1" \
  --wan-attention-source="$SCRATCH/wan-audit/wan_modules_attention.py"
COGNI_RUN_SAFE_MIN_FREE_PCT=15 scripts/run_safe.sh \
  "$BUILD/wan_s2v_block_benchmark" 900 12288 \
  --fixture="$EVIDENCE/cross-full-v1" --repetitions=3
COGNI_RUN_SAFE_MIN_FREE_PCT=15 scripts/run_safe.sh \
  "$BUILD/wan_s2v_full_block_benchmark" 900 20480 \
  --weights="$SCRATCH/evidence/wan-official-block0/wan-official.weights.safetensors" \
  --pairs=3
```

The oracle requires a new or empty output directory; reuse the existing fixture
for native reruns. Expected signals are exit zero, all three upstream comparisons
within the stated gate with finite values, bit-identical tiled segmentation
outputs, zero encode buffer allocations, one terminal wait per timed arm, and
rejection of one corrupted value. Official block runs additionally require
6 examples with zero failures, errors and pending cases for each segment size.
These are separate scopes, not a full-spatial upstream block certificate.

Local raw logs, memory samples and hashed summaries are in
`/opt/mldata_ssd/cogni-ml/video-scene-20261019/evidence/wan-block-benchmark`.
Projection evidence is `projection-{hidden,ffn-in,ffn-out,cross-kv}.log` and
`projection-summary.json`; official block gates are `block-upstream-{320,1024}.log`.
The source benchmark and upstream manifest define what was measured; a log
pointer alone is not an attestation.

Full-size evidence is `full-block-native.log`, `cross-native-settled.log` and
`cross-upstream.log`, with matching memory logs. Earlier cross repeats remain in
`cross-native-initial.log` and `cross-native.log`. `analyze_measurements.py`
checks the recorded gates, order, counts and controls and writes hashed
`measurement-summary.json`; it indexes observations, not their independent
authenticity. The correlated mini-Quadrumvirate review supports the bounded
decision to retain 320; it does not certify an optimal backend or full-video
speed or quality.

Refresh these conclusions after a change to kernels, command-chain behavior,
buffer lifetimes, inputs/weights, geometry, precision, hardware or runtime.
Existing strict forty-layer drift and full-video qualification remain separate
open boundaries; this experiment does not relax either.

## DeepInfer transfer assessment

DeepInfer was inspected at `774ce244b87cc0eb8d4b94ab9f5de2ab22049f01` without
installing dependencies or copying implementation code. Its bounded scheduler
work and shape buckets support testing execution granularity; they do not
prescribe a Metal attention tile size. Its [CUDA graph runner](https://github.com/deepseek-ai/DeepInfer/blob/774ce244b87cc0eb8d4b94ab9f5de2ab22049f01/deepinfer/worker/cuda_graph_runner.py)
captures fixed-shape work, retains graph memory and replays it. This is a useful
future experiment if Metal host submission remains limiting, with compatibility,
lifetime and memory costs still to prove for mixed MPS/custom command chains.

Wan already retains its buffers and prepared plans. DeepInfer's asynchronous
token scheduler does not remove GPU command-buffer backpressure, and its NVIDIA
batch-invariance option is not a transferable Metal kernel or a demonstrated Wan
speedup. The pinned [source audit](https://github.com/deepseek-ai/DeepInfer/tree/774ce244b87cc0eb8d4b94ab9f5de2ab22049f01)
and local `deepinfer-audit.md` distinguish those hypotheses from measurements.
