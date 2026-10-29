# Hy-Nomas performance introspection, 2026-10-26

## Full-J callback-pending window, 2026-10-29

The existing J21120 36-block-plus-head traces reproduce the 63-root pattern
found in the J5760 FFN probe. This qualification reanalyzes four pinned raw
reports at parent `60480968`; no model, GPU replay, Studio or generation ran.
Only the offline analyzer, its tests and documentation change. Runtime math,
buffers, queue policy, waits and production selectors remain unchanged.

### Observation and limits

| Saved schedule | Traced requests | Continuation events | Calls strictly longer than 10 ms | Earlier callback-pending roots at entry, every long call |
| --- | ---: | ---: | ---: | ---: |
| Order, normal | 4 | 18,312 | 1,030 | 63 |
| Order, reverse | 4 | 18,312 | 992 | 63 |
| Paced A/A, normal | 2 | 9,156 | 510 | 63 |
| Paced A/A, reverse, including captured request 1 | 2 | 9,156 | 509 | 63 |

All 3,041 long calls are nested phase-2 `self_attention` continuations. Only
roots with sequence less than `before_root_sequence` are counted: the root
being submitted is excluded. In every case the oldest pending root is
`current_sequence - 63`, and its callback is observed during the API call.
The call returns 0.012458-0.210292 ms after that callback observation. Every
root in these reports has a source-1 callback observation; there are no
source-2 terminal-wait fallback roots. Callback-pending describes observation
time, not the number of physically running GPU commands.

Excluding the externally captured A/A request from timing totals leaves
2,780 long calls: caller-thread CPU totals 66.323 ms against 337.096 s of
call wall time. Within each eligible whole-chain GPU-root envelope, gaps
outside the interval union sum to 2.603-4.747 ms. This is envelope-minus-union,
not whole-request time minus the union. These are interval observations,
not active shader cycles, GPU occupancy, driver-lock identification or proof
that the host wait can be removed from end-to-end latency. Source inspection
finds a default `newCommandQueue`, with no explicit 64-buffer setting;
`maxPendingRoots` records a high-water mark rather than enforcing capacity.

The caller's attention marker does not identify the work being waited for.
For 840 long calls the oldest root has only a gate/up phase candidate; for
420 it has only down. Another 420 have mixed FFN candidates. Mixed candidates
stay grouped, and these counts are not an exclusive GPU cost breakdown or
proof that FFN caused the wait. They explain why moving or combining host
markers cannot by itself establish an attention speedup.

In the normal-order profile study, baseline request 2 takes 42.792485 s and request 5 takes
60.195153 s, despite identical callback-pending entry histograms. Both have
4,480 calls at 63 pending roots; long-call count rises from 251 to 285 and
median long-call duration from 107.522 to 161.782 ms. The pending window is
therefore insufficient to explain the drift. Reverse-order baseline times
are 42.584312 and 42.345675 s, contradicting a simple later-is-slower rule.

The separate 44.939 -> 84.561 -> 50.567 s row-tiling series has no continuation
trace. This analysis does not directly trace or resolve that transition.
The older 43.356/69.456 s pair compares timed input A with a poisoned,
different-input gate; it is not a same-input drift measurement. Paced A/A
reverse request 1 was externally captured. Its trace is included above only
for the descriptive pending-window check; `--external-capture reverse:1`
continues to exclude it from the A/A timing and continuation aggregates.

### Analyzer contract and evidence

The shared order analyzer now reports callback-pending entry/return counts,
fallback/unknown-source censoring, signed oldest-root return offsets and
candidate-phase groups by API phase/depth. Pending predicates and the strict
10 ms cutoff use integer Mach ticks, avoiding subtraction of large converted
timestamps at the threshold. Unknown or fallback source markers cannot become
callback observations. Empty groups cannot satisfy nonempty universal claims.
The FFN single-event overlap helper remains compatible with minimal legacy
roots whose observation source is unknown.

Evidence root:
`/opt/mldata_ssd/cogni-ml/hy-nomas-pixelumm-pending-window-20261029`.
`verify_pending.py` independently uses a timestamp heap instead of the shared
analyzer's prior-root scan for callback-pending counts. Candidate phase labels
and whole-root metrics come from the saved report and are separately validated
by the repository analyzer. `independent-check.json` pins the four raw reports
and their two prior integrity receipts, rechecks 1,204 saved finite bitwise-
equal upstream comparison records across 28 requests at unchanged
`atol=rtol=2e-5`, and checks one terminal wait, zero named encode allocations,
zero preterminal activation reads and 43 post-wait diagnostics per request.
Those are saved adapted-Torch-MPS controls, not fresh upstream execution or
new artifact-payload authentication. Historical guard receipts retain their
original 10% cache-aware reserve; no new GPU memory-admission result is claimed.

`order-analysis.json` and `aa-analysis.json` replay the repository validators;
`cross-check.json` compares their pending summaries with the independent
recomputation, including an explicitly descriptive check of the captured arm.
`ffn-analysis.json` exercises the other shared-helper caller on the pinned
four-arm J5760 reports. `analysis-source/` and `analysis-manifest.json` freeze
the offline code, tests, inputs and outputs. The qualification commands are:

```sh
python3 -B -m unittest spec.pixelumm_order_analysis_test \
  spec.pixelumm_aa_analysis_test spec.pixelumm_ffn_aa_trace_analysis_test
python3 -B /opt/mldata_ssd/cogni-ml/hy-nomas-pixelumm-pending-window-20261029/verify_pending.py
git diff --check
```

The verifier creates a fresh output exclusively; preserve existing results
when replaying. The three test suites pass 28 tests, all three raw-report
analyzer CLIs exit 0, and the cross-check matches all 12 traced requests
(11 timing-eligible plus the separate captured request). `cpu-checks.json`
and `cli-checks.json` retain the observed command results. Correlated
mini-Quad review is ROBUST only for the tested
offline contract and pinned observations. Cause of drift and a speed benefit
from changing queue capacity remain unverified. Rollback removes the offline
fields; no runtime rollback is needed. Refresh after input/report, trace
semantics, clock, source, backend/device, precision, schedule or toolchain
changes; each future GPU process needs fresh memory admission.

The next causal discriminator remains a matched full-J replay that changes
only inter-request CPU diagnostic policy while retaining input, resources,
captures and spacing. For acceleration, measure actual GPU service in the
remaining FFN/attention work and require a complete-request gain. The present
evidence does not justify counting overlapping `commitAndContinue` wall time
as removable overhead or increasing queue capacity as the next speed lever.

## J5760 FFN continuation attribution, 2026-10-29

A diagnostic-only continuation trace now localizes the late host-encoding
increase observed in the baseline below. Four arms ran sequentially on one
process and owner thread in OFF/ON/ON/OFF order. API profiling stays enabled
in all arms; only the existing continuation trace changes. Each arm constructs
a fresh one-shot owner and command chain, reuses its buffers/plans across the
same two warmups and sixteen measured samples, and releases them after its
single terminal wait. The fixture/cache are authenticated once before all arms.
No generation, Studio, production selector, arithmetic, or synchronization
policy changed.

### Result and interpretation

| Arm | Continuation trace | FFN GPU-root union median | Continuation wall total | Continuation owner-thread CPU total | Graph-exclusive wall total |
| --- | --- | ---: | ---: | ---: | ---: |
| 0 | off | 159.260 ms | 1,861.133 ms | 3.466 ms | 14.591 ms |
| 1 | on | 161.206 ms | 1,847.810 ms | 2.932 ms | 11.854 ms |
| 2 | on | 166.150 ms | 1,941.864 ms | 3.134 ms | 11.545 ms |
| 3 | off | 159.701 ms | 1,813.581 ms | 1.807 ms | 11.138 ms |

The late per-sample host encode increase reproduces in every arm: early
samples need about 0.16-0.58 ms, while late samples need about 104-140 ms.
Almost all encoding-loop wall time is inside `commitAndContinue`; graph
exclusive time is only about 11-15 ms over all 54 graph calls. Graph inclusive
time already contains nested continuations and must not be added to them.
The sample timer excludes explicit leading/trailing continuation boundaries;
the serialized API totals and trace include them.

For the two traced arms, the union of **earlier** GPU command-root envelopes
overlaps 99.9881% and 99.9914% of continuation-call wall time. Root *i*, being
submitted by event *i*, is excluded: only sequences less than *i* contribute.
Intervals are merged before summing so overlapping roots cannot double-count
time. The entire chain has only 1.817 ms and 1.696 ms between root envelopes
in the traced arms. This is evidence against large gaps from a CPU producer
failing to submit work at this shape. It is not a measurement of shader
occupancy or ALU-active cycles within a command-buffer envelope.

The observed increase is therefore localized to low-owner-thread-CPU calls
while prior GPU work remains in flight. Backend backpressure is consistent
with this pattern, but timing overlap does not prove why an Apple API blocked
or descheduled the thread. Removing 1.8-1.9 seconds of reported host waits is
not a supported latency-saving prediction: that host wall overlaps GPU work.
These measurements do not justify another lock/GC/command-encoding rewrite
as the main acceleration lever for this FFN. They also do not explain the
separate J21120 full-chain drift.

A second discriminator counts earlier roots whose completion callbacks had
not yet been observed when each continuation began. In both traced arms,
all 33 calls longer than 10 ms begin with exactly 63 such roots; none begins
below 63. The oldest pending root is always `current_sequence - 63`. Calls
return 0.016-0.167 ms after that root's callback observation, or 0.109-1.323 ms
after its GPU end timestamp. Their median wall times are 51.335 and 53.651 ms.
This supports admission backpressure at a fixed pending-root window. Callback
pending is not a count of physically active GPU commands, and the pattern
does not identify a driver lock or prove a specific Metal/MPS capacity limit.
Increasing queue capacity alone is not a supported speedup prediction.

The two traced GPU medians are higher than the two API-only medians, with
variation even between the traced repeats. Four arms do not establish zero
observer effect or equivalence. This control measures incremental trace cost
conditional on API profiling; it does not measure the API profiler's own cost.
No performance candidate is promoted.

### Numerical, lifecycle, and memory checks

All four arms match all 56 normal metric slots against the pinned upstream
and canonical BF16 words exactly. Each seeded comparator NaN produces exactly
one failure and one nonfinite count. Tolerances remain `atol=rtol=2e-5`.
Each arm has 162 completed error-free roots, 54 graph calls and 161
continuations: 125 graph-nested and 36 explicit sample boundaries. Every
selected FFN root union has zero overlap with outside roots. Each arm takes
one terminal wait and one 912-byte metric readback, with zero named
`MetalBuffer` allocations or readbacks during encoding. Trace setup allocates
its fixed host event array before sampling, and trace/API snapshots occur
only after the existing wait. Named counters do not measure opaque MPS or
ordinary host allocations.

The external guard completes with return code 0 and 43 samples under the
unchanged cache-aware 10% reserve, 8 GiB process-group footprint cap,
64 MiB swap-growth stop and 600-second timeout. Sampled peak footprint is
3,746,681,824 bytes (3.489 GiB); minimum headroom is 40,379,039,744 bytes
(37.606 GiB), minimum pressure-free percentage 86%, and swap growth zero.
File-backed/speculative pages are counted once. These sampled limits do not
exclude shorter spikes or establish that every cached page is reclaimable.

### Evidence, verification, and next discriminator

Evidence root: `/opt/mldata_ssd/cogni-ml/hy-nomas-ffn-aa-trace-20261029`.
`source/` pins 172 build/support files from parent `75140810` plus this
runner/profile patch. The release binary SHA256 is
`17953d92488002ce3145227c4dcc7ed846c03d833cf2aaa23c3937c831d24027`;
`build-manifest.json` SHA256 is
`3347d1999abf25906085e22d81d41ca141015ca1b111b514418b9dc639b34a76`.
`launch.command` reproduces the guarded `--trace-sweep` using fresh result
paths. Raw reports are `results/sweep-{0-api,1-trace,2-trace,3-api}.json`;
`results/sweep.json` is written only after all four pass. Independent
`verify_trace.py` recomputes numerical counts, unions, outside overlap,
prior-root overlap, guard accounting and source/binary identities into
`results/verification.json`. The repository analyzer separately validates
native trace clocks, event topology, nesting and full API statistics.
Its `results/trace-analysis-2.json` rechecks all four arms; per-phase raw
continuation wall/CPU totals must fit their separately sampled API aggregates.
Hostile cases reject forged-small API totals, shifted sample ranges, and
boundary/internal nesting swaps even when aggregate counts still agree.
`analyze_pending_roots.py` independently recomputes the callback-pending
distribution into `results/pending-roots.json`, pinning both input hashes;
the main agent separately reproduced the decisive counts. Source inspection
finds a default `[gs_device newCommandQueue]` on this chain, with no explicit
64-buffer setting. The observed 63-root window is not a configured-cap claim.
`analysis-source/` freezes the final offline analyzers and hostile tests;
`analysis-manifest.json` pins these files and result hashes (SHA256
`9c9ba55fa31e1e73739f262e5b8435c5f328a97fa359ade47d5670c5396dee9b`).

Executed qualification commands and signals:

```sh
# Run make from the frozen source/ directory.
make BUILD_DIR=/opt/mldata_ssd/cogni-ml/hy-nomas-ffn-aa-trace-20261029/build \
  pixelumm_ffn_aa_probe spec_pixelumm_ffn_aa_probe
crystal spec -Dcpu_only spec/pixelumm_ffn_aa_contract_spec.cr \
  spec/wan_mps_api_profile_spec.cr spec/wan_mps_continuation_trace_spec.cr
python3 -B -m unittest spec/pixelumm_ffn_aa_trace_analysis_test.py \
  spec/pixelumm_order_analysis_test.py
```

The frozen-source release build and seven contract examples pass; the three
CPU-only Crystal suites pass 21 examples. The two analyzer suites pass 15
tests, including 11 new analyzer tests. The unchanged fixture/guard suites
pass another 53 Python tests. The native API-clock qualification passes two
examples; format and diff checks pass. These CPU checks do not replace the
four-arm guarded native run. Correlated mini-Quad review is ROBUST for the
bounded measurement and numerical claims; full-chain causal attribution
remains unverified.

The next discriminating experiment remains a matched J21120 replay with the
same 36-block/head chain, input, captures, resources and spacing, changing only
inter-request CPU diagnostic policy. For an actual acceleration candidate,
use measured GPU work or the algorithmic video budget; this trace gives no
basis for counting overlapping host waits as removable GPU latency. No full
generation is required for that replay. Refresh this evidence after any
source, input, weight, geometry, precision, backend, device, schedule, profiler,
guard or toolchain change. Rollback removes the optional diagnostic flags and
offline analyzer; production math is unaffected.

## Native J5760 connected FFN A/A, 2026-10-29

The bounded calibration proposed below has now run. It establishes one short
baseline series on the pinned saved input; it does not resolve the separate
J21120 full-chain drift or establish a production speedup. No generation or
Studio ran, and no production implementation or selector changed.

The diagnostic reuses the production `pixel_linear_step(..., true)` graphs
and native pointwise kernels, in order: gate, up, BF16 SiLU, BF16 product,
down. Each projection converts its BF16 operands to F32 for computation and
rounds its output to BF16. The native product feeds down directly. One owner
retains the same plans and GPU buffers for two warmups and sixteen measured
samples at J5760, D4096, F12288. Only w0/block0 post-attention-normalized input
is fixed; saved product/down tensors are comparison oracles, not substitute
inputs to the connected computation.

### Observed result

| Measure | Observed value |
| --- | ---: |
| Timed FFN GPU-root union, median | 160.083 ms |
| Minimum / maximum | 158.050 / 175.393 ms |
| p90, interpolated | 168.479 ms |
| Median absolute deviation / median | 0.828% |
| First-four / last-four median | 166.610 / 160.247 ms |
| Last-four / first-four ratio | 0.96181 |
| Eight adjacent later/earlier ratios, range | 0.97320-1.10973 |

This series has no growing early-to-late median; one adjacent pair still
differs by 10.97%. It does not establish stable latency across long exposure
or a causal explanation of previous drift. The three matrix products count
1.739462 TFLOP/sample; dividing by the whole FFN root-union median gives
10.866 projection-equivalent TFLOP/s. This includes pointwise work, casts and
any in-root stalls, and is not a GEMM-only rate or ALU utilization.

All eighteen product/down comparisons and both raw gate/up captures match
the authenticated upstream BF16 words exactly. Every down output also matches
the first warmup's canonical GPU output bitwise. The unchanged finite
`atol=rtol=2e-5` checks pass, with expected element counts in all 56 normal
metric slots. The separate local NaN control produces exactly one failure
and one nonfinite value; its word-difference count is zero because word
comparison precedes local float injection.

### Measurement and memory boundary

There are no tensor reads, per-capture host comparisons, hashing, memory
diagnostic sampling or GC diagnostics inside the sampling loop. Host clocks
and native root-cursor bookkeeping remain around each sample; the report's
`cpu_diagnostics_between_samples:false` means no CPU tensor diagnostics,
not an absence of host timing instrumentation. The external sampled memory
guard remains active throughout.

Three GPU comparisons follow each FFN. Raw captures occur only in warmup 0,
and their comparisons and the NaN control follow the loop. An independent
CPU recomputation of all 162 completed, error-free root timestamps confirms
each selected FFN interval union and zero overlap with every outside root.
Comparison roots are excluded from the timed sets. Their fixed cadence may
still affect subsequent device state. The metric is a union of command-buffer
GPU intervals, not exclusive shader-active cycles or end-to-end latency.

The successful run has one terminal wait, one packed 912-byte post-wait
readback, zero preterminal tensor reads and zero tracked `ML::MetalBuffer`
allocations during encoding. Named buffers total 1,481,638,800 bytes; this does
not account for opaque framework allocations. Host stage-encoding wall time
changes from about 0.5 ms in early measured samples to 106-122 ms later, while
GPU-root unions stay within the range above. Internal MPS backpressure or
scheduling is not attributed by this probe; no extra explicit wait was added.

The guard completes with exit 0 and 43 samples, under a cache-aware 10%
reserve, 8 GiB process-group cap, 64 MiB swap-growth stop and 600 s timeout.
Sampled group footprint peaks at 2,264,811,344 bytes (2.109 GiB); minimum
headroom is 40,592,424,960 bytes (37.805 GiB), minimum pressure-free reading
88%, and positive swap growth zero. Headroom is
`free_speculative - speculative + file_backed`, counting speculative pages
once. File-backed memory remains a reclaimability estimate, and sampled
limits do not rule out between-poll spikes.

### Evidence and decision

Measured source parent: `762caf4f5ad1e7afb4d6a2897277a769fd7b5484`, with the
new diagnostic files preserved alongside 171 pinned source files in
`/opt/mldata_ssd/cogni-ml/hy-nomas-ffn-aa-j5760-20261029/source-2/`.
The same external directory contains `build-manifest-2.json` and:

- `results/aa-3.json`, SHA256
  `f4871703cb9f57f3f3fb869d3f9b3f26f35590fe6bcbfed2da7a347cc2686851`;
- `results/guard-3.json`, SHA256
  `66759e3c971b358c9ea74c0b1fc67d379625a136221c3b55014dc02830c98d7b`;
- measured binary SHA256
  `f9525148dbb63c0107f49e9175b928de4df3f150db70a179891e8c0e2fe9d082`;
- `verify_aa.py` and its recomputed `results/verification.json`.

The fixture is the authenticated Torch 2.6.0 windows reference, manifest
SHA256 `e9b1ddce57fac94efb0262bd3e5f0be2963ef9df4255dc74cf254ff4b0217ddc`,
with the authenticated thirty-six-block BF16 cache. Provenance remains
`saved_initial_noise_conditioning_upstream_stem`, `model_inference:false`;
this is not generated-DiT activation throughput or a quality result.
Only three selected weights are uploaded. Full fixture/cache authentication
happens before device preparation and timing. Earlier launch attempts 1/2
produced no measured GPU result: attempt 1 rejected a symlinked input path
(fixed by canonicalization); attempt 2 stopped during file authentication
without a final guard event. Its cause remains unassigned. A separate CPU
launcher control and the complete guarded attempt 3 succeed; failed attempts
are excluded from timing evidence and retained.

Executed DoD:

```sh
AA_ROOT=/opt/mldata_ssd/cogni-ml/hy-nomas-ffn-aa-j5760-20261029
TMPDIR="$AA_ROOT/tmp" CRYSTAL_CACHE_DIR="$AA_ROOT/crystal-cache" \
  make BUILD_DIR="$AA_ROOT/build" pixelumm_ffn_aa_probe spec_pixelumm_ffn_aa_probe
PYTHONDONTWRITEBYTECODE=1 python3 -B -m unittest \
  spec/hy_nomas_pixelumm_ffn_windows_reference_test.py spec/wan_memory_guard_spec.py
PYTHONDONTWRITEBYTECODE=1 python3 -B "$AA_ROOT/verify_aa.py"
crystal tool format --check examples/pixelumm_ffn_aa_probe.cr \
  examples/support/pixelumm_ffn_aa_probe.cr examples/support/pixelumm_ffn_aa_contract.cr \
  spec/pixelumm_ffn_aa_contract_spec.cr
git diff --check
```

Release build, 7 Crystal contract examples and 53 Python fixture/guard tests
pass. The verifier recomputes all numerical counts, sample unions, outside
overlap, statistics and guard accounting, and checks source/binary identities.
Three correlated mini-Quad reviews return ROBUST for this bounded parity and
measurement claim; attributing or resolving old full-chain drift remains
VULNERABLE. Rollback removes the standalone diagnostic and its build targets.
Refresh on source/input/weight/geometry/precision/runtime/backend/device/
schedule/guard change or artifact loss; every launch requires fresh admission.

The next attribution experiment remains the matched J21120 36-block-plus-head
replay described below, changing only the inter-request CPU diagnostic policy
while preserving resources and spacing. A new kernel candidate needs matched
A/B timing and a full-request check; this A/A does not select or promote one.

## Current reassessment: distinguish nominal headroom from measured loss, 2026-10-29

This section supersedes the historical next-action list below. It is a
read-only executable-source audit at parent `73909256`, with CPU recomputation
of frozen reports and three fresh-context, correlated Quad reviews. No model,
GPU probe, Studio, build, profiler or generation ran. Production selectors,
precision and synchronization are unchanged.

### The missing denominator

The often-quoted twofold gap is **not a measured native-versus-upstream speed
ratio**. It divides 734.577 TFLOP of modeled preview matrix work by a nominal
13.6 TFLOP/s and compares the resulting 54.013 s with a historical 103.315 s
neural execution. That ideal omits elementwise work, conversions, reductions,
resource preparation and scheduling. The nominal rate is not calibrated
sustained throughput, hardware utilization or a hard bound for every opaque
BF16-fed MPS operation. See the
[operation-count model](hy-nomas-generation-frontier.md#operation-count-roofline-for-the-fixed-bf16-preview-2026-10-29).
Saved upstream tensors establish the stated numerical checkpoints; they do not
provide an upstream timing denominator.

Three workloads must remain distinct:

- The saved complete preview is 120 frames at 512x288/24 fps, with eight DiT
  steps and five overlapping J5760 PixelUMM windows, using saved Qwen state.
- The latest integrated diagnostic is one J21120 chain of 36 decoder blocks
  and the head, with no diffusion or generation.
- The isolated FFN probe uses actual J21120 rows once, one block's saved
  weights/activations and stage GPU-root intervals. Its timings are not a
  complete request, exclusive shader cycles or a prediction for J5760.

The original non-CUDA upstream decoder uses eager dense attention rather than
the bounded local implementation. Do not launch it as an alleged fair MPS
speed control: it has a different work/memory contract. A projected dense F32
score tensor at J21120 alone is 53.174 GiB. The existing adapted Torch 2.6 MPS
oracle remains a numerical control, not a fresh original-upstream speed run.

### The large GEMMs already have much less than twofold nominal headroom

For J21120, D4096 and F12288, one FFN projection costs
`2*J*D*F = 2,126,008,811,520` FLOPs. The frozen isolated baseline yields:

| Phase | Counted TFLOP | GPU-root interval, ms | Effective TFLOP/s | Fraction of nominal 13.6 |
| --- | ---: | ---: | ---: | ---: |
| Gate + up | 4.252018 | 371.140 | 11.457 | 84.24% |
| Down | 2.126009 | 196.807 | 10.803 | 79.43% |

These graph intervals include their casts and are admitted without overlap
with other observed roots in the measured samples. They do not measure ALU
utilization. At the assumed nominal rate, their arithmetic-only times would be
312.648 and 156.324 ms: conditional reductions of 15.76% and 20.57%, rather
than 50%. This does not prove the optimum or bound a different arithmetic
implementation. Do not multiply these intervals by 36 or transfer them to the
different preview geometry. An overall work/wall ratio near half the nominal
rate does not imply that each dense GEMM has half its potential throughput.

The large-row workload also differs from token decoding in Qwen/Nomic:
weights are reused across thousands of rows. The existing one-pass interface
model gives about 1,002 FLOP/byte for the J5760 F32 gate projection. It is not
a physical DRAM measurement, but it invalidates an automatic transfer of
memory-bound single-token compression/lock-removal gains to this GEMM.

### Why the unsuccessful experiments are not equivalent

- Shared/packed gate-up, transposition and M-row tiling keep full-K arithmetic.
  Fewer graphs or shorter host encoding need not reduce GPU service. In the
  packed experiment, encoding fell 15.6% while terminal waiting grew 14.3%;
  whole-request time did not improve. Direct MPS reduced roots from 125 to 97
  but worsened paired request time by 7.7%.
- The native F32 kernel passed its saved numerical gate but was about 4.6x
  slower than the relevant MPS projection. More custom code is not evidence
  of a better dense implementation.
- Native F16 and SDPA were rejected by unchanged numerical gates. No speed
  qualification followed; these are accuracy failures, not measurements
  proving that faster arithmetic cannot help. Algebraic equivalence does not
  preserve the reference's reduction and BF16 rounding boundaries.
- The earlier explicit-F32 FFN graph did reduce a bounded connected block's
  latency by 25-29%. This is evidence that MPS lowering/algorithm selection can
  matter; it is not a fresh full-generation gain. See the
  [FFN qualification](hy-nomas-pixelumm-block-frontier.md#earlier-single-block-f32-ffn-graph-qualification-2026-10-27).

Concatenating the SwiGLU gate/up matrices leaves the two dense products and
the intervening nonlinear product intact. Unlike an affine recurrent scan,
the row-parallel FFN has no identified sequential recurrence to expose through
associative composition. No exact algebraic reduction in its arithmetic has
been established here. Surrogates, changed attention reductions and fewer
diffusion steps need their own error/quality contract; they cannot silently
inherit the unchanged implementation-parity claim.

### Real drift remains unexplained

The same latest baseline takes 44.939 s during warmup, then 84.561 s on its
first timed request and 50.567 s later. Apparent row-tiling improvements are
14.895% in normal order and 0.483% in reverse order, failing the prospective
3% criterion. These samples neither prove equivalence nor a stable slowdown.

Each full request has 43 retained capture records and post-wait diagnostic
reads; 516 comparisons belong to the twelve-request set, not each request.
These are not 43 mandatory blits: head captures use resident buffers and some
block outputs can alias the workspace rather than create copy buffers.
The 37.9-45.6 s host diagnostic work is outside the request timer and cannot
directly explain a slow measured GPU envelope. It can change the next
request's host/device state. The nearly continuous root-interval coverage
also weakens CPU producer starvation as the leading explanation, while not
proving active GPU saturation. Clock/power, competing GPU work, backend
resource pressure and diagnostic carryover remain competing hypotheses.
GC-off, weight staging and root-count reductions have existing negative
controls; do not reopen those routes solely from source-level suspicion.

### Decision and next falsifier

Stop treating twofold nominal headroom as a promised optimization target, and
pause further tile-size/fusion grids that preserve the same arithmetic.
The next bounded native calibration should use the **J5760 preview geometry**,
reused buffers/plans and repeated baseline FFN samples. Pin the operand and
oracle provenance before running: the existing J5760 drift fixture uses saved
initial-noise conditioning, explicitly not generated DiT latents. It can
calibrate that fixture/shape, not establish representative preview activation
throughput or quality. Retain GPU comparisons against the same saved upstream
values, one existing terminal wait, unchanged `atol=rtol=2e-5` and the
cache-aware 10% reserve. Do not put per-capture host allocation, hashing or GC
between the measured samples. A short replicated A/A precedes any new
candidate; no full generation is required.

The second obligation is separate: attribution of severe drift in the
J21120 36-block-plus-head chain requires a matched replay of that same chain,
input, captures, resources and request spacing, changing only the inter-request
host diagnostic policy. An isolated J5760 result cannot answer that question.
If drift survives without those diagnostics in the matched chain, they are not
a necessary cause; if it disappears, reinstating only that policy must
reproduce it before claiming causality. GPU clock or contention attribution
requires direct observations, not a timing label.
Only then choose an attention/reduction or backend-resource probe from the
remaining measured wall cost. Existing overlapping phase intervals cannot be
summed into an exclusive cost breakdown. Root tracking/callback overhead is
also a hypothesis until its CPU cost is observed; it does not justify a new
queue/runtime rewrite.

Use actual phase time for pipeline predictions. The saved preview has
35.768158459 s DiT, 67.547219917 s decoder windows/display and 149.110485917 s
total application time. Even a hypothetical halving of that entire decoder
phase gives **1.485659x neural** or **1.292826x application** speedup, holding
everything else fixed. These are Amdahl calculations, not measured gains. A
twofold whole-pipeline target must also address DiT/setup or reduce work.

Executed CPU DoD:

```sh
python3 /opt/mldata_ssd/cogni-ml/hy-nomas-deep-quad-20261029/verify_headroom.py
git diff --check
```

The verifier reports PASS for five pinned reports, projection rates, both
orders, 516 records, baseline drift and conditional phase-Amdahl arithmetic.
It checks all twelve requests retain one wait and zero preterminal reads or
tracked encode allocations. The report is
`/opt/mldata_ssd/cogni-ml/hy-nomas-deep-quad-20261029/headroom-audit.json`,
SHA256 `cc653be8c7c6af00b634c547281aa81c98fad7675ec08670036a266ecb57e2ae`.
It pins the five source reports. The row-tiling runtime evidence refers to its
archived measured sources; the later comment-only correction is documented in
the [generation frontier](hy-nomas-generation-frontier.md#gateup-row-tiling-does-not-qualify-as-acceleration-2026-10-29),
not a fresh live-source GPU audit. Correlated Adversary verdict: ROBUST for
the arithmetic and narrower scope; VULNERABLE for any claimed cause of drift,
attainable optimum or stable upstream speed ratio. Refresh on source,
operand/geometry/precision, backend/device/toolchain, schedule, guard or
artifact changes. No new GPU performance or quality result is established.

## Historical audit, 2026-10-26

This is a source and evidence audit at native parent `5b7803f8`, not a new GPU
benchmark or a generation-quality result. The three fresh-context Quad audits
are correlated review. No model, Studio, build, profiler or generation was
launched for this audit; production code and numerical gates are unchanged.

Subsequent implementation and guarded full-forward measurements are recorded
in [the text-compaction frontier](hy-nomas-text-compaction-frontier.md): M64
storage with the original text RMS reduction shape passes the same-Q4 upstream
gate and saves 797 MiB. All six adjacent timing pairs favor it, with median
10.64% lower compact time; absolute drift prevents a stable-latency claim.

## Decision

Pause K16 loader tuning as the primary speed frontier. Keep K32 and the current
MPS control. The first decision probe is an inventory of the actual PixelUMM
checkpoint, target segment geometry and aggregate residency, before starting a
decoder port or timing it. The first low-cost implementation probe is
elimination of unused DiT text rows. Use actual image-stream activations before
another large-M Q4 kernel experiment. Qualify representation quality separately
from kernel parity; current uniform Q4 has not qualified as a quality-preserving
replacement for BF16.

The intended value remains usable video at roughly 60-120 seconds of wall time
per second of output, with acceptable appearance and motion. Neither a fast
projection nor a small weight file establishes that value. Required counter
metrics are decoder time, complete pipeline latency, peak resident memory,
swap growth, BF16-relative model error, and decoded visual/temporal quality.

## What the existing work establishes

- All 54 DiT blocks execute on GPU from a saved Qwen producer boundary, with
  zero tracked encode allocations, zero preterminal activation reads and one
  terminal wait. Fourteen captured outputs match the corresponding upstream
  BF16 or identically decoded Q4 control exactly. Scratch reuse and serial
  weight staging are already implemented; they are not new opportunities.
- BF16-weight/F32-math MPS saved-input forwards took 6.415-6.420 seconds;
  staged Q4 took 6.545-6.563 seconds. The runs were not interleaved, so no
  relative speed or staging-cost conclusion follows. Named storage falls from 22,616,779,332 to
  10,895,796,804 bytes. Compression has a demonstrated residency benefit even
  without a demonstrated arithmetic speedup.
- Q4 versus original BF16 reaches final-latent relative L2 1.184475 and cosine
  0.638643. This does not prove visible artifacts. It rejects a claim that this
  quantization error is negligible. The unchanged finite `atol=rtol=2e-5`
  implementation gate compares identical decoded operands, not lossy Q4 to
  original BF16.
- The K16/K32 probe uses block-0 text FFN-down weights and 23 saved input rows
  repeated to M1280. K16 loses all 12 paired rounds per representation: median
  duration ratios 1.155866 dense and 1.229616 Q4. B128/SG8 weight-panel reuse
  was already tested and had no material gain over B64. Direct Q4 still has
  no demonstrated advantage over the relevant MPS controls.
- A prior synthetic one-block profile reports 137.626 ms wall, 1.853 ms CPU
  encode and 136.704 ms GPU-root union. Attention-only interior roots sum to
  57.623 ms, but shared boundary roots and stalls prevent exact attribution.
  This is neither a phase profile of the saved-input full54 forward nor a
  video latency measurement.

Evidence: [full54 forward and quality](hy-nomas-q4-dit-frontier.md),
[K16/K32 and earlier tile controls](hy-nomas-q4-reduction-frontier.md),
[whole-block profile](hy-nomas-native-frontier.md#previous-slice-representative-one-block-profiling).
Frozen artifacts remain under `/opt/mldata_ssd/cogni-ml/hy-nomas-q4-dit-20261025`,
`hy-nomas-q4-panel-profile-20261026` and `hy-nomas-full-block-profile-20261024`.

## The overlooked decoder can be comparable to the denoiser

External sources were inspected at upstream main
`37372925fd4bbb7977a89ce3248584926fc9e953`. The native port remains pinned to
`911cea6458a89c2663513ff7099561ec190dd491`; this audit does not update that pin.
The current-main PixelUMM defaults are a Qwen3-like generation expert with
36 layers, hidden width 4096, FFN width 12288, 32 query heads and 8 KV heads
of width 128. A latent `[1,32,6,22,40]` produces a `6*44*80 = 21,120` token
grid. Its intended pixel shape is 24 frames at 704x1280; no frames were decoded
in the native full54 qualification.

Counting multiply-add as two FLOPs, the decoder's layer projections alone cost

```text
Ndec = 6 * (2*22) * (2*40) = 21,120
Pdec/layer = 2*4096^2 + 2*4096*1024 + 3*4096*12288 = 192,937,984
Fdec/stack = 2 * Ndec * 36 * Pdec/layer = 293.389 TFLOP
```

For comparison, eight passes through the current DiT geometry require about
285.272 TFLOP in their analogous token-wise projections, including Q/K/V and
attention-output projections as well as FFN linears on both sides. These
counts exclude pairwise QK/PV attention, norms, activations, decoder input/head
projections, blending and I/O. They do not predict elapsed time: operand
precision, GQA, locality, kernel efficiency and window overlap differ. They
show why treating the decoder as cheap postprocessing is unsafe. Decoder layer matrices alone
contain 13,891,534,848 BF16 bytes, excluding the input/head and other tensors.

The current upstream preset uses one decoder denoising step, temporal windows
of 10 latents with overlap 5, no spatial crop tiling and a local-attention tile
of 128 pixels. Local attention limits attention work; it does not remove the
dense FFN/projection cost. More decoder steps or overlapping windows increase
work. Do not silently substitute a smaller architecture or another checkpoint.
For this short input there is one temporal window and one spatial task, so the
upstream preset's eight-way window distribution does not split that decoder
task over eight GPUs.

This comparison is restricted to the six-latent, one-window fixture. The README
example uses 30-latent segments; overlapping windows can process latent positions
more than once, while DiT attention scales differently from the linears above.
Do not extrapolate the decoder/denoiser ratio to a five-second segment or a
complete video without recomputing the target geometry and window schedule.

There is also a reference-backend trap. The current `_flex_attention` compiles
its fused kernel only on CUDA; its docstring explicitly says eager execution
materializes the full score matrix. The default non-CUDA, non-compiled path
uses that eager implementation; the preset sets `compile_decoder: false`.
A hypothetical F32 score tensor at this fixture is
`21120^2 * 32 * 4 = 57,094,963,200 bytes`, or 53.174 GiB, before other tensors.
This is a size calculation, not an observed allocation or a verified MPS
execution. Do not launch the full eager decoder as an MPS oracle or implement
local attention as a dense score matrix plus a mask. Physically bounded
neighborhood/tiled computation needs its own unchanged-tolerance check against
the upstream semantics, initially at small shapes; an adapted oracle must be
identified as adapted rather than claimed as a CUDA execution.

Next discriminator: inventory the actual decoder checkpoint and its required
architecture, then decide whether a bounded decoder-block probe is feasible at
the target token/window shape. Admit aggregate Qwen + DiT + decoder weights and
overlapping workspace lifetimes before allocation. Reuse suitable CogniQwen
RMS/GQA/MLP building blocks, but explicitly implement the decoder's T/H/W RoPE,
bidirectional local attention and precision boundaries; a causal Qwen3.5 graph
is not a drop-in decoder. A block timing is only an estimate until the head,
windowing and complete decode are measured.

Sources: [decoder architecture and token mapping](https://github.com/Tencent-Hunyuan/Hy-Nomas/blob/37372925fd4bbb7977a89ce3248584926fc9e953/hymm/models/vae/pixelumm_decoder/model.py#L34-L68),
[decoder forward](https://github.com/Tencent-Hunyuan/Hy-Nomas/blob/37372925fd4bbb7977a89ce3248584926fc9e953/hymm/models/vae/pixelumm_decoder/model.py#L289-L350),
[local attention and eager fallback](https://github.com/Tencent-Hunyuan/Hy-Nomas/blob/37372925fd4bbb7977a89ce3248584926fc9e953/hymm/models/vae/pixelumm_decoder/model.py#L125-L173),
[decode windows and sampler](https://github.com/Tencent-Hunyuan/Hy-Nomas/blob/37372925fd4bbb7977a89ce3248584926fc9e953/hymm/models/vae/pixelumm_decoder/autoencoder.py#L350-L418),
[inference preset](https://github.com/Tencent-Hunyuan/Hy-Nomas/blob/37372925fd4bbb7977a89ce3248584926fc9e953/configs/experiment/inference.yaml).

## Exact candidate: remove unused text rows after the refiner

The saved-input fixture has 5,280 image tokens, 1,280 text rows, and 23 valid
text tokens. Attention already uses only the 5,303-token valid prefix.
`HyNomasConfig.text_rows` and all text linear/FFN operations still use 1,280
rows. Padded rows have per-row norms, modulation and residuals, and never enter
valid attention queries or keys. Under the admitted batch-one, single-segment,
no-history contract, eliminating them after the full text refiner should not
change image outputs or the 23 valid text outputs. This is a dataflow argument,
not a full-size numerical result: changing M can change MPS reductions.

For D2048/F8192, token-wise Q/K/V, output and FFN linears cost

```text
Flin/row = 2*(4*D^2 + 2*D*F) = 100,663,296
Flin/block = Flin/row * (5280+1280) = 660.351 GFLOP
Fattn_QK_PV/block = 4*(5280+23)^2*D = 230.374 GFLOP
Fremoved/block = 1257 * Flin/row = 126.534 GFLOP
Fremoved / (Flin + Fattn_QK_PV) = 14.206%
```

This is an arithmetic opportunity, not a 14.2% latency claim. The fraction
depends on prompt length and excludes other work. It does not remove image
tokens or reduce already-compact attention. A short valid text prefix may
also favor a different small-M backend; existing M23 split-K evidence is not
permission to change the full model's operand/reduction contract.

Falsifier: keep Qwen/refiner inputs and outputs unchanged, bind a compact valid
DiT text prefix on GPU, and compare image outputs at blocks 0/7/53, valid text
outputs and final latent against the current same-operand upstream/control at
finite `atol=rtol=2e-5`. Poison the excluded suffix; valid outputs must be
invariant. These selected checkpoints and the final result establish only that
scoped contract; claiming all 54 intermediate boundaries match requires
streaming comparisons at every boundary. Rebuild fixed plans and verify
lifetimes, zero encode allocations, zero preterminal reads and one terminal
wait. Run opposite-order A/B pairs on the complete saved-input forward only
after correctness passes. Reject on any
valid-output failure; do not widen tolerances. Unused padded text outputs are
explicitly outside this proposed consumer contract, so the old all-1280-row
capture hash is not an appropriate equivalence claim. Multi-segment/history
support must reopen the proof before reusing this optimization.

Anchors: [block dataflow and linear shapes](../src/ml/video/hy_nomas_block.cr),
[valid-prefix attention plan](../src/ml/video/hy_nomas_block_mps.cr),
[existing tiny prefix test](../spec/hy_nomas_mps_padded_prefix_spec.cr),
[saved-input geometry](../examples/support/hy_nomas_q4_dit_probe.cr).

## Algorithmic leverage and the latency budget

The inspected current upstream README explicitly supports distilled 4/6/8-step
inference and shows a four-step Diffusers example; the native inference preset
still defaults to eight. The local source checkpoint is already the distilled
DiT. Check immutable checkpoint identity, scheduler and precision behavior
before qualifying a four-step native path; fewer steps are an approximate
generation policy, not an exact kernel rewrite.

Using the observed 6.42-second saved-input forward only as a conditional unit:

| DiT forward calls | Conditional DiT seconds |
| --- | ---: |
| 4 | 25.68 |
| 6 | 38.52 |
| 8 | 51.36 |
| 9, if a non-final cache-write forward is required | 57.78 |

These are not measured seconds per second of video. They exclude live Qwen
composition, scheduler, PixelUMM and output handling; input/segment geometry
and step-dependent runtime can differ. Do not multiply by CFG=2 blindly: the
inspected default guidance is 1.0. Do not remove a required cache write as if
it were redundant. A four-step path can halve denoiser calls while saving much
less than half the complete pipeline time if decoding is expensive.

Source: [distilled model and four-step example](https://github.com/Tencent-Hunyuan/Hy-Nomas/blob/37372925fd4bbb7977a89ce3248584926fc9e953/README.md#model-assets).
Qualify step-count quality with identical initial noise, prompts and output
settings against the eight-step control when decoding is available. Report
motion, consistency and semantic adherence as well as latency; implementation
parity alone does not establish acceptable four-step generation.

## Quantization: optimize the error seen by the model

For a fixed input X and weight error E, the projection error is `X E^T`:
`||X E^T||_F^2 = tr(E (X^T X) E^T)`. Weight-space MSE, a Gaussian weight
histogram, and largest scalar residuals omit activation covariance and
downstream amplification. Repeating 23 rows 56 times provides no new coverage.
The earlier Wan activation-fit experiment already improved its two training
rows and regressed on the other two in both folds. Do not repeat that route
without genuinely new calibration and independent held-out prompts/timesteps.

Prefer a bounded Q8 or mixed BF16/Q8/Q4 quality screen before adding codec or
GEMM complexity. Allocate precision to sensitive matrices using identical real
inputs and downstream outputs, with the complete resident byte budget counted.
Sparse deltas can repair isolated errors; they are not automatically a compact
model of structured error. A more ambitious hypothesis is a high-precision
low-rank branch extracted before quantizing the residual, evaluated against
higher-bit and sparse-patch alternatives at equal complete bytes. SVDQuant
provides a primary example of this design and of fusing the branches to avoid
activation traffic. Its NVIDIA W4A4 results do not establish M2 Max W4A16 speed
or Hy-Nomas quality, and a post-hoc low-rank fit of already-quantized error is
not the same algorithm.

Source: [SVDQuant, sections 4.2-4.3](https://arxiv.org/html/2411.05007v4#S4.SS2).
Prior falsifier: [Wan held-out calibration regression](wan-joint-affine-benchmark.md#distortion-and-allocation).
Every representation keeps a separate unchanged same-operand implementation
gate. Candidate representations must reduce distortion relative to original
BF16 model outputs, not only pass against their own decoded oracle. No
visible-quality claim is admitted before decode.

## Kernel work that earns another probe

Large-M GEMM reuses each weight over many input rows. Smaller stored weights
can improve feasibility and reduce pressure without reducing MMA count or
being the limiting bandwidth term. Register decoding, panel reloads, shared
memory and accumulator pressure can consume the benefit. The tested B128
weight reuse and K16 storage reduction are concrete counterexamples to a
universal smaller-payload/faster-kernel claim; they do not identify a causal
hardware stall.

Measure actual image weights/activations and the saved-input stack's phase
share before another large-M kernel branch. Amdahl's bound for a phase of
fraction p accelerated by s is `1 / (1-p+p/s)`; without p there is no defensible
video speed estimate. Existing timestamps can orient this without new waits.
If that share justifies kernel work, candidates are FC1 + unchanged F32 GELU
+ next-input rounding, and FC2 + bias + gated residual epilogues. Compare
intermediate values first, preserve contraction/rounding order, then time the
whole FFN/block. These fusions currently apply to the experimental direct-Q4
route; they are not already-qualified changes to the staged MPS production path.
Do not reopen split-K at M1280 without new evidence addressing its 949 failures
and measured slowdown.

## Next frontier and failure policy

1. Inventory the actual PixelUMM checkpoint, target segment/window geometry and
   aggregate resident memory. This is a decision probe, not a decoder-port
   commitment or a phase-timing result.
2. Keep the qualified opt-in M64 text-compaction route and padded control;
   [numerical checks and paired timings](hy-nomas-text-compaction-frontier.md)
   now pass on the fixed saved-input fixture. Reopen on text/geometry or backend
   changes instead of assuming exact-prefix storage preserves MPS arithmetic.
3. Qualify and time a bounded decoder block only after the inventory admits it,
   with physically local attention and an explicitly scoped upstream control.
   Treat block timing as provisional until a complete decode is measured.
4. Screen higher-bit/mixed weights on real held-out inputs; retain original
   BF16 as the quality control and compact staging as the memory fallback.
5. Qualify the distilled four-step schedule against eight-step output quality
   after the decoder and live producer handoffs are available.

All GPU probes remain serial, fully resident in the model path, with no added
preterminal synchronization, unchanged finite tolerances and the cache-aware
10% physical-memory reserve plus existing footprint/swap/deadline guards.
Temporary files stay under `/opt/mldata_ssd/`. No full generation is authorized
by this audit. Rollback is to leave proposed paths unused and retain existing
BF16/MPS controls. Stop a branch when its scoped falsifier fails or its maximum
pipeline headroom cannot affect the value target; another geometry tweak alone
does not reopen a refuted hypothesis.

The correlated Adversary review is ROBUST within this source/evidence audit's
scope. It does not establish future runtime, end-to-end quality or independent
replication. Audit verification: inspected current source and frozen report
summaries; recomputed the FLOP, storage and conditional-time arithmetic directly; checked
document links and `git diff --check`. These checks validate this decision's
anchors and arithmetic, not future speed or quality. Refresh on source,
checkpoint, token/mask contract, segment/history, precision, toolchain, device,
or missing artifact changes. The decoder architecture is explicitly a
current-main observation and must be matched to its chosen checkpoint before
implementation.
