# Hy-Nomas first native video frontier

## Combined-attention order localization (2026-10-29)

Status: VERIFIED for bounded host-phase/API localization, retained saved-output
parity and sampled memory fit. The physical cause of the timing drift and a
stable production speedup remain unresolved. Production selectors stay off;
there was no diffusion or full generation.

`pixelumm_decoder_integrated_probe --profile-combined-attention` adds a separate
eight-request diagnostic on authenticated input A. Both modes warm with poisoned
scratch and profiling off; the next four requests use profiled ABBA, followed
by two unprofiled tails. A second serial process reverses every mode assignment.
Shared gate/up stays disabled. Completed root intervals are collected after the
existing terminal wait on all eight requests; phase wall/thread CPU, API and
continuation records are collected only on the middle four. Exact graph counts
and all 36 latched per-Plan selectors guard against fusion misattribution.
The prospective discriminator was whether the difference followed the mode,
ordinal position, or broad continuation waiting with little caller CPU.

The following seconds are observer-on host measurements, not kernel timings:

| Process/request | Mode | Whole request | Attention marker | Attention continuation API wall | Continuation thread CPU |
| --- | --- | ---: | ---: | ---: | ---: |
| normal-1/2 | baseline | 42.792485 | 40.050632 | 39.841887 | 0.035886 |
| normal-1/3 | combined | 39.870393 | 37.603717 | 37.400612 | 0.035003 |
| normal-1/4 | combined | 39.832005 | 37.566830 | 37.352387 | 0.037410 |
| normal-1/5 | baseline | 60.195153 | 56.773947 | 56.536751 | 0.040737 |
| reverse-1/2 | combined | 40.381760 | 38.068375 | 37.888100 | 0.032238 |
| reverse-1/3 | baseline | 42.584312 | 39.855497 | 39.661136 | 0.033747 |
| reverse-1/4 | baseline | 42.345675 | 39.654472 | 39.471536 | 0.032657 |
| reverse-1/5 | combined | 39.862891 | 37.602929 | 37.427409 | 0.030485 |

The largest same-mode drift is normal baseline request2 to request5:
whole-request wall grows 17.402668 s, attention marker wall 16.723315 s
(96.096% of that change), and nested continuation wall 16.694864 s.
Attention marker CPU grows only 0.033059 s. All 36 block host spans increase;
this is not a single early/late block. FFN gate/up plus down marker wall changes
0.973432 to 1.204150 s, and terminal wait 0.921548 to 1.233585 s. The dominant
change appears while encoding attention, inside the MPS continuation call.
Attention graph-encode wall after excluding nested continuations is 0.157588 to
0.208488 s across these requests. Inclusive graph and nested continuation
spans must not be added together.

The raw Mach trace brackets `[super commitAndContinue]`; our root bookkeeping
lies outside that span. Joining it to GPU root intervals excludes the root
being committed and unions intersections with earlier roots. Earlier-root
intervals cover 99.98794%/99.98795% of attention continuation wall in normal/
reverse processes under the existing 2 microsecond clock bound. Raw attention
continuation wall sums are 171.118798/154.436725 s, while caller thread CPU sums
are 0.140415/0.121149 s. This supports waiting during previously submitted GPU
work, not CPU instruction cost of encoding or our bookkeeping locks. It does
not identify the GPU kernel responsible, distinguish bandwidth/compute/thermal
or other scheduling effects, or prove exclusive GPU busy time. A host attention
span can wait for preceding FFN work; the small FFN host markers do not bound
FFN GPU cost. Parent/depth checks are syntactic because not every graph API
scope is exported; callback observation time is not exact GPU completion time.

All profiled combined arms remain 39.832005 to 40.381760 s at both ends/orders.
Three baseline controls are 42.345675 to 42.792485 s; the fourth is 60.195153 s.
Reverse request5 is fast, contradicting a simple monotonically later-is-slower
explanation. This is a repeatable mode association in these two profile runs,
not causal separation of the selector from run-state interactions. The normal
unprofiled tails remain shifted at 49.432697 s baseline and 42.480520 s combined.
Off warm/tail controls do not calibrate observer overhead. Every request has
4,579 roots; every profile has 6,480 attention graph encodes and 3,960 attention
continuations. Whole-chain root unions track the broad wall shift; shared
phase candidate sums are non-additive and do not attribute individual kernels.

Existing memory-event clocks also expose substantial off-timer observer work:
after the first request, the 43-capture comparison span is 36.023 to 36.183 s.
Profile extraction is about 48 to 50 ms when enabled versus 6 to 8 ms off;
post-diagnostic GC spans are 1.090 to 4.088 ms. These spans include recorder
overhead and are not instrumented GC stop-time or isolated comparator timings.
They cannot directly explain 17.4 s inside request timing, but the long gaps
can change device state and limit extrapolation to uninterrupted production.

Both processes return zero with identical 197 measured source files, release
binary, objects and fixtures. All 16 requests pass 688 finite bitwise-equal
retained saved-upstream comparisons at unchanged atol=rtol=2e-5. They preserve
the direct block35/head GPU handoff, one terminal wait, zero tracked named
encode allocations/preterminal activation reads and the existing 43 post-wait
diagnostic reads, using the same 330 MiB host comparison pair. Retained coverage
is 36 attention outputs, block34/35 outputs and five head stages, not all
overwritten early outputs or internal F32 probabilities. Saved upstream MPS
controls are not fresh connected independent-backend inference.

All 2,612 guard samples pass the unchanged 27.5 GiB cap, cache-aware 10% reserve,
64 MiB positive-swap-growth stop and 900 s per-process deadline. Maximum sampled
footprint is 26.347140 GiB, minimum headroom 19.517944 GiB, observed swap growth
zero. The historical headroom formula label remains wrong; pinned code and
samples use `free_spec - speculative + file_backed`, without double counting
speculative pages. Sampling does not bound opaque MPS transients or future fit.
Native processes have ended.

DoD executed: release build; 62 focused Crystal examples; four Python analysis
tests including 15 corrupted-report controls; eight integrity corruptions;
three incompatible-CLI rejection controls on the measured binary; format,
diff and build-freshness checks. `run_integrated.py normal-1` and `reverse-1`
execute the serial native requests. `audit_integrity.py --label normal-1
--label reverse-1 --output results/integrity-final.json` authenticates source,
binary, fixture, receipt, report, lifecycle and guard identities. The checked-in
`examples/support/analyze_pixelumm_order_profile.py --report
results/normal-1.json --report results/reverse-1.json --output
results/analysis-final.json` recomputes phase, API, root-overlap and observer
gaps; it does not itself authenticate execution receipts or assert bitwise
parity. Correlated Mini-Quad reviews are not independent backend evidence.

Evidence root:
`/opt/mldata_ssd/cogni-ml/hy-nomas-pixelumm-order-profile-20261029/`.
Integrity index `results/integrity-final.json` SHA256:
`37b3b84cad62fe08540b22e8e0bb138b6e976c2e83fcefb42fdc8cbba8d2bd3c`.
Analysis `results/analysis-final.json` SHA256:
`13f8bb895f0033e11877b1489540016739bf164091f86bb50cd88c5f59394cce`.
Analyzer source SHA256:
`2933ce822cffa30accaa3e5d03a50a60b904f1d19607ffe9f36d57a822f948c4`.
External integrity auditor source SHA256:
`84046dfc6aa7d22681bac5af12cb0501f402b69a2ff0c9c1eba0d8009e783a54`.
Measured release binary SHA256:
`b5e501416d6b69685a199df8d2414455a8b4c955a33f6710e5533ce690aeffa5`.
Next discriminator: fixed-mode A/A with matched between-request spacing and
kernel-level GPU attribution of the delayed roots. Do not optimize host locks
or promote a stable speed claim from phase wall alone. Rollback omits the new
profile flag. Refresh on scoped source/artifact, fixture, geometry/precision,
schedule/lifetime, clock, device/backend/toolchain or guard changes; each new
process requires live admission. Earlier unprofiled epochs below are not pooled.

## Combined attention on 36 PixelUMM blocks and head (2026-10-29)

Status: VERIFIED for the bounded opt-in diagnostic, saved-output parity and
sampled memory fit. The prospective timing screen passes descriptively;
a causal or stable generation-speed claim remains VULNERABLE. Production
selectors stay off. No diffusion or full generation was run.

`pixelumm_decoder_integrated_probe --compare-combined-attention` prepares the
existing scale/softmax alternative once on every Plan, then selects it together
with fused K/V gather on all 36 idle Plans. Shared gate/up remains disabled.
Four poisoned gates cover both modes on both authenticated inputs. Four timed
same-input requests use ABBA; another process reverses warm and timed orders.
A third process repeats ABBA. All three runs use the same 197 source files,
runner, objects, fixtures and release binary. Its SHA256 is
`ed8fca7b1ad8d65d79c74232c939de7edac3c3bcec4bcf65a4f38502df0de290`.

The connected T6/H44/W80/J21,120 decoder and head completed 24 requests and
1,032 finite bitwise-equal comparisons at unchanged atol=rtol=2e-5. Every
request retains the direct same-buffer block35/head handoff, one terminal
wait, zero tracked named encode allocations and no preterminal activation
readbacks. Its 43 existing diagnostic CPU reads happen after that wait and
outside request timing. Retained captures are 36 attention outputs, block34/35
outputs and five head stages. Outputs 0-33 are overwritten GPU handoffs, not
separate saved goldens. The oracle composes authenticated saved upstream MPS
controls; it is not fresh connected independent-backend inference or internal
softmax-probability parity. In-flight selector rejection and poisoned replay
checks pass; exceptional failed-wait quarantine is source-reviewed, not injected.

| Run | Timed order | Baseline median | Combined median | Median adjacent paired time reduction |
| --- | --- | ---: | ---: | ---: |
| normal-1 | ABBA | 48.999847 s | 45.762902 s | 6.5132% |
| reverse-1 | BAAB | 49.977961 s | 47.533238 s | 4.9058% |
| normal-2 | ABBA | 49.965031 s | 48.135272 s | 3.2761% |

All three per-run medians exceed the prospective 3% screen, with both orders
and a confirming repeat. The six paired reductions range from -1.4346% to
+9.8273%; four pairs improve. Baseline-then-candidate pairs have a +9.1331%
median, while candidate-then-baseline pairs have a -0.0158% median. The pooled
paired median is +5.9400%, but six pairs and this order sensitivity do not
establish a causal or stable speedup. Do not promote defaults from the screen
alone or interpret request wall time as kernel-only work or video throughput.

The diagnostic preallocates one distinct expected/actual host pair after
aggregate admission and before GPU setup: 173,015,040 bytes each, 346,030,080
bytes total (330 MiB). All 43 captures and requests reuse exact prefix slices.
Expected records are reauthenticated on every read; actual reads, element
scans, tolerance checks and hashes remain unchanged. Post-return collection
runs equally outside both arms' timed intervals. Comparison allocation churn
is bounded by the audited 64 MiB limit, and the reported post-GC host heap does
not grow across these eight-request processes. This bounds those two arrays,
not every host allocation or opaque MPS resource.

All 4,559 guard samples pass the unchanged 27.5 GiB process cap, cache-aware
10% reserve, 64 MiB positive-swap-growth stop and 900 s per-process deadline.
The largest sampled peak is 26.286456 GiB, minimum headroom 19.468338 GiB,
and observed swap growth is zero. The guard's historical formula label is
inaccurate: pinned code and all samples use `free_spec - speculative +
file_backed`, avoiding speculative-page double counting. The final audit
explicitly records this label mismatch; it does not reinterpret the label as
arithmetic or weaken admission. Sampled bounds do not cover between-poll spikes
or guarantee future admission. Native processes have ended.

Release build, formatting, 60 focused CPU examples and 30 validator corruption
controls pass. The native DoD was executed serially by `run_integrated.py`
with labels `normal-1`, `reverse-1`, `normal-2`, each returning zero with
unchanged sources; `validate_integrated.py` over all three reports returns zero.
The runner refuses consumed labels. Preserve evidence rather than overwrite it.
Correlated Mini-Quad reviews support the bounded parity/lifecycle/guard claim;
they are not independent backend agreement.

Evidence root:
`/opt/mldata_ssd/cogni-ml/hy-nomas-pixelumm-combined-full-reuse-20261029/`.
Final index: `results/audit-final.json`, SHA256
`6c09bd6bbf51ce3372dbc94ab1dcda806f967346c904c666e66daabf984d1e40`.
Validator SHA256:
`a58d12cd75811eef8ca8f0e10649a299361a5850f36e4ab2fcc301e8d59db322`.
Receipts pin report, fixture and measured-source identities; source snapshots
and raw guard/log/report files remain external. Rollback omits combined mode
or reverts the scoped diagnostic change. Refresh on scoped source/artifact,
fixture, geometry, precision, schedule/lifetime, backend/device/toolchain or
guard changes; each new process still requires live memory admission.

Earlier epochs remain excluded from timing: the original 28.5 GiB launch was
refused before native execution; the stricter-cap attempt stopped at request5
with a 27.597 GiB sampled peak; post-return GC alone stopped before request4
completed at 27.595798 GiB. Tracked Metal inventory stayed constant while the
old host heap grew; this did not prove a GPU leak or its retention mechanism.
The reusable comparison pair is the subsequent successful source epoch. Raw
attempts remain in the sibling `combined-full`, `combined-full-cap27p5` and
`combined-full-postgc` evidence directories, all dated 20261029.

## Combined K/V gather and score-scale/softmax experiment (2026-10-29)

Prospective bounded test: `pixelumm_fused_kv_gather_probe --scale-softmax`
selects both existing alternatives together, after the owner resets each
request and before creating its command chain. The same four plans retain
the prepared scale graphs in both arms. Baseline disables both flags;
candidate enables both. Production defaults remain off. Rollback leaves both
native selectors disabled; omitting the flag restores the K/V-only diagnostic.

Hypothesis: the isolated gains may compose, but MPS scheduling and queue
behavior can make them non-additive. Correctness must first pass all saved
upstream stage/output comparisons at unchanged atol=rtol=2e-5. Only then admit
unprofiled balanced AB/BA timing on the same release binary and source bytes.
The prospective promotion screen remains at least 3% paired whole-request
improvement and a confirming repeat; fewer dispatches alone are insufficient.

Scope is four J5760 whole-block passes: two windows, each with native block0
output feeding block1. Full captures qualify both arms, followed by output-only
requests. No raw K/V panel or combined internal-F32 comparator is asserted.
Geometry, precision, scratch bindings and GPU-resident handoffs stay fixed.
Each request must retain one terminal wait, one 464-byte packed post-wait
metrics read, zero tracked encode-time named allocations and no preterminal
readbacks. Setup owns the extra softmax graphs; their opaque MPS storage is
outside the named-buffer inventory. Run under the cache-aware 10% reserve,
16 GiB process cap, 64 MiB positive-swap-growth stop and 600 s deadline. No full
generation is admitted. Refresh evidence on scoped source/artifact, fixture,
geometry, precision, schedule/lifetime, backend/device/toolchain or guard changes.

Two serial unprofiled 40-request runs passed that screen on identical measured
source bytes and release binary:

| Run | Baseline request median | Combined request median | Median paired time reduction | Candidate wins |
| --- | ---: | ---: | ---: | ---: |
| 1 | 1.636905 s | 1.585109 s | 4.1034% | 16/16 |
| 2 | 1.500376 s | 1.387504 s | 7.2021% | 16/16 |

Percentages are medians of adjacent pair reductions, not ratios of the two
arm medians. The pooled paired median is 5.6375%, with 32/32 wins. AB/BA
medians are 3.9808/4.3595% and 7.2021/7.2864%. The absolute regime and effect
size vary between runs; these are descriptive bounded results, not a promised
7% production gain or proof of super-additivity over isolated experiments.
Whole-request time includes owner setup/selection, comparator encoding, the
sole terminal wait and metrics read. Body encode excludes comparator encoding;
root durations are not exclusive GPU busy time. Both arms retain 125 roots.

All 400 compared saved-upstream stage/output slots across timing runs are
finite and bitwise equal. Every request retained one terminal wait, one packed
metrics read, zero tracked encode-time named allocation and no preterminal
readback. Sampled peak footprint was 7.625 GiB, minimum cache-aware headroom
22.402 GiB and positive swap growth zero under the declared limits. Samples
do not bound opaque MPS allocation transients. The artifact validator rejects
14 damaged controls covering schedule, coverage, NaN control, receipt, combined
flags, tolerance and memory floor. Twenty-one focused CPU examples pass.
Correlated Mini-Quad source and artifact reviews return ROBUST within this
successful unprofiled, saved-stage/output scope. Independent pair recomputation
also distinguishes the 3.1643/7.5230% ratios of arm medians from the stated
paired estimates; neither estimator establishes a stable production effect.

The combination passes the prospective local request screen. Keep production
selectors off pending connected 36-block/head validation; no full generation
was executed. DoD executed: the release `pixelumm_fused_kv_gather_probe` target,
four-request correctness gate, then two balanced timing runs under
`examples/support/wan_memory_guard.py`. Numerical failure would block timing.
The pinned fixture, weight cache, common driver and comparison stages are
unchanged. Evidence, raw reports, exact measured-source archives, runner and
validator are under:
`/opt/mldata_ssd/cogni-ml/hy-nomas-pixelumm-fused-kv-scale-softmax-20261029/`.
Summary SHA256 `b8ebc68b5c949196791a15b03d04f8f58c569b10609a817e7942fc0a2bf96bd9`;
timing binary `7282e0b3897be2dcefa2586aad67ef94552248e95c0368fd318375405e35b073`;
raw reports `660ca83379046c0a5ee69dd2bb2be1eff4d3b6159a173ac181696d2ab141995c`
and `a0149bb970ece3c07b38ae586d22d7a48957b9a738a963b8a85624ee10e4fd75`.

Source review found that a partial preparation failure could label the graph
configuration as already prepared. A post-timing setup/report-only fix now
sets that flag only after all four plans finish preparation; request clocks,
selection, encode, wait and comparison code are unchanged. The original
timing binary and source archives remain intact. A separate final release
build passed the guarded four-request correctness gate; its report SHA256 is
`0ea67e27250711ea9f030ada36c8d92115885d7dbcbfe939b9bf3d5bec89131a` and
binary is `280831e14493976b71ea42a367ef0963605582907588ca721294cb54046c62e6`.
`results/final-qualification.json` authenticates the source delta. Timing
claims belong to the archived measured binary, not a new final-binary timing.
The optional Instruments failure-cleanup route was not exercised; the
successful unprofiled request lifecycle is the admitted scope.

## Exact FFN/attention rewrites rejected by the request screen (2026-10-29)

The remaining dense FFN projections remain the strongest hotspot candidate by
arithmetic volume and overlapping matrix-shader intervals; those intervals do
not establish exclusive FFN GPU cost. A refined timestamp-containment audit
uniquely assigns all 1,138 matrix-shader intervals to a GPU root. In captured
output-only requests, the definite FFN/attention matrix timeline unions were
378.341/119.211 ms (baseline) and 288.138/90.792 ms (packed). These subsets do
not cover unresolved intervals, active cycles or unprofiled request speed.
The reproducible strict-containment calculation and input identities are in
`/opt/mldata_ssd/cogni-ml/hy-nomas-pixelumm-hotspot-20261028/results/matrix-attribution-20261029.json`
(SHA256 `56d563dcd90858748daee36725b1cccec0de79e7ef72f2045d40cc00ddd43942`).
Shader rows do not carry command-buffer IDs; the assignments use timestamps.
Two rows partially intersect a second root while remaining fully contained in
exactly one. The companion Python script recomputes these checks from the raw
capture exports; this does not establish causal or exclusive shader costs.
Reorienting all three F32 GEMMs as
`transpose(W @ transpose(X))` passed all six saved upstream BF16 stages on
both J5760 windows and blocks 0/1, but failed the whole-request speed screen:
-0.6382% median paired improvement, 4/16 wins, negative AB and BA medians.
Roots increased from 125 to 137. Reject this unchanged route; shorter terminal
waits did not compensate for longer encoding/MPS call intervals.

Evidence (diagnostic source archived with receipts):
`/opt/mldata_ssd/cogni-ml/hy-nomas-pixelumm-ffn-orientation-20261028/results/analysis-timing-1.json`.
No full-J timing, connected decoder, exclusive GPU-cost or generation claim.

The output-only softmax/PV rewrite also passed both J2048 and J21120 raw
F32 attended-output comparisons and all saved BF16 captures, but failed the
whole-request speed screen: -0.7061% median paired improvement, 6/16 wins,
negative AB and BA medians, unchanged 125 roots. Removing an external output
binding did not establish removal of internal probability storage. Keep it
explicitly diagnostic and off by default; do not repeat the unchanged route.
Evidence: `/opt/mldata_ssd/cogni-ml/hy-nomas-pixelumm-output-only-pv-20261029/results/analysis-timing-1.json`.

The explicit macOS 15 SDPA candidate failed the first BF16 rejection screen:
plan 0 post-attention norm had 4,899 tolerance failures / 7,776 differing words,
followed by downstream FFN/output failures. Baseline passed. No SDPA timing was
admitted. The diagnostic implementation was archived externally and removed
from the live core rather than leaving a known-inaccurate alternative there.
Evidence: `/opt/mldata_ssd/cogni-ml/hy-nomas-pixelumm-sdpa-20261029/results/correctness-1.json`.

Pairing adjacent query groups with identical ordered K lists reduced 12 groups
to 6 but failed the unchanged BF16 gate: plan 0 post-attention norm had 206
tolerance failures / 329 differing words, followed by larger FFN and successor
failures. Baseline passed, with no nonfinite values. This is the first captured
failure, not a localization to QK, softmax or PV. No timing was admitted.
Evidence: `/opt/mldata_ssd/cogni-ml/hy-nomas-pixelumm-query-pair-20261029/results/correctness-1.json`.

Classic `MPSMatrixMultiplication` for gate/up/down also passed the unchanged
saved upstream stage gate with bitwise equality, including the final-source
correctness-only repetition. Explicit GPU conversion retains BF16-to-F32
operands, full-K F32 multiplication and the original BF16 output boundaries.
All descriptors, kernels and the serialized 192 MiB F32 weight stage are
prepared before requests. Combined named storage is 7,582,827,060 bytes;
a 7 GiB negative admission rejects before fixture/Metal initialization.

The final-source unprofiled run completed 40 requests, including 16 balanced
AB/BA timed pairs. Baseline/candidate request medians were 1.582481/1.708753 s;
the paired median time reduction was -7.7081%, with zero candidate wins and
negative medians in both orders. Encoding was 26.99% shorter but the terminal
wait was 37.71% longer. GPU root-duration sums also worsened by 7.39%; neither
root durations nor API waits establish exclusive kernel costs. Root count
fell 125 to 97. Reject the unchanged candidate: fewer roots and shorter host
encoding are not a request speedup.

All 200 compared slots were finite and bitwise equal. Each request kept one
terminal wait, one 464-byte post-wait metrics read, no preterminal read and
zero tracked encode-time named allocations. Guard peak was 7.807 GiB, minimum
cache-aware headroom 26.041 GiB and positive swap growth zero. The 10% reserve,
16 GiB process cap, 64 MiB swap-growth stop and 600 s deadline held; sampled
guard evidence does not bound unsampled MPS allocations. The final-source
build and artifact validator passed; four damaged CPU artifacts were rejected.
Static correlated adversary review found the transactional release and
reciprocal diagnostic-mode guards robust within their serialized lifecycle.

Evidence: `/opt/mldata_ssd/cogni-ml/hy-nomas-pixelumm-ffn-direct-mps-20261029/results/analysis-timing-1.json`;
raw report SHA256 `d8f497d76d65502621cce994ef2c52a023d8b5d4ae5d67660c610e6e07c3bdd7`,
binary `4d54eb37e6c0c95edfa19a3e4496d896a5d90d58a95650fcedd175f971fe082c`.
Archived measured-source copies and execution receipts preserve the precise
source state. Rollback leaves the MPSGraph projections selected. No full
generation or improvement was established by this candidate.

Scope correction: the frozen common driver and runtime binding verifier used
in these probes connect each native block-0 output to its block-1 input. Earlier
text describing independent saved block-1 inputs was incorrect. The driver
SHA256 is `e83f70829a67652baa0b37226a066c0acaac8bf23277cc8e621133f478ccdf39`;
its archived copies and execution receipts agree across scale-softmax,
orientation, output-only, query-pair and packed gate/up runs. This establishes
two connected blocks per window, not a complete decoder/head or generation.

## Fused K/V gather qualification (2026-10-29)

The next opt-in candidate gathers K and V in a single native dispatch per
attention tile, retaining the original head/token mapping and two disjoint F32
destinations. At J5760 this replaces 24 K/V dispatches with 12 per block. It
does not reduce tensor storage or claim lower physical DRAM traffic. The
baseline selector remains the rollback; native CogniGraph islands are mutually
excluded. Plans and the added pipeline are prepared before requests.

The falsifier is the unchanged upstream stage gate on two connected blocks in
each of two windows, including full captures, output-only checks and the NaN
control. Require atol=rtol=2e-5, one terminal wait and packed post-wait read,
zero preterminal reads or tracked encode-time buffer allocations, a 10%
cache-aware reserve and the existing 16 GiB/64 MiB swap-growth/600 s limits.
Only after correctness may 16 balanced AB/BA pairs test the prospective 3%
whole-request improvement screen. No raw K/V panel bitwise comparator or full
generation is part of this candidate's claim.

The release build, correctness-only qualification and subsequent final-source
40-request run passed. All 200 compared slots were finite and bitwise equal;
the NaN and malformed-selector controls rejected as expected. Sixteen balanced
pairs gave baseline/candidate medians 1.448511/1.435620 s, paired median time
reduction 1.2331%, 16/16 candidate wins, and positive AB/BA medians
1.1986/1.6199%. Root count stayed 125. This is a small descriptive gain below
the prospective 3% default-promotion screen; keep the selector off by default.
Do not interpret differences from previous runs' control medians as a gain.

Each request retained one terminal wait and one 464-byte post-wait read, zero
preterminal reads and zero tracked encode-time named buffer allocations.
Sampled peak footprint was 7.618 GiB, minimum cache-aware headroom 24.321 GiB
and positive swap growth zero. The unchanged limits held. The artifact
validator passed and rejected four damaged CPU controls; the focused CPU
schedule/lifecycle suite passed 29 examples. Correlated source review found
the default selector, buffer mapping and ABI/link isolation robust in the
serialized, unprofiled request scope. The added pipeline is created at plan
setup even with the selector off: preservation applies to default request
execution, not zero setup overhead or setup failure independence.

Evidence and archived measured source:
`/opt/mldata_ssd/cogni-ml/hy-nomas-pixelumm-fused-kv-20261029/results/analysis-timing-1.json`;
raw report SHA256 `94cac65039df4e67cf9c033d44ed4f130e0f140fc0475a38eccade9c26e27f7c`,
binary `6e22a64d909cd4613ab4601060fb2b79f0beaf88d07365c65193ecb207457e68`.
Refresh after scoped source, geometry, backend/device, schedule or guard
changes. No full generation or exclusive GPU-utilization claim is established.

## Operation-count roofline for the fixed BF16 preview (2026-10-29)

This is a CPU-only arithmetic model, not a new model run or measured hardware
utilization. The local device has 38 M2 Max GPU cores and 64 GiB unified memory.
[Apple specifies 400 GB/s bandwidth](https://support.apple.com/en-us/111838).
The nominal 13.6 TFLOP/s FP32 figure comes from the methodology of
[Benazir and Lin's primary study](https://arxiv.org/html/2508.08531v1), not an
Apple-published FP32 guarantee or a calibrated sustained MPS rate on this host.

Count two FLOPs per MAC. The fixed first-segment preset has 120 frames at
512x288/24 fps, 54 DiT blocks over eight steps and 36 PixelUMM blocks over five
overlapping windows. Saved Qwen state is an input; live Qwen encoding is
excluded. DiT linears process 4,320 video plus 64 stored text rows; attention
uses 4,320 plus 23 valid text rows. For D=2,048 and F=8,192, each DiT block
costs `24D^2 + 8D^2*S + 4D*F*S + 4D*A^2`, with S=4,384 and A=4,343.

PixelUMM uses J=5,760, D=4,096, F=12,288, 32 query heads, eight KV heads and
head width 128. The 3x4 local-tile layout has eight 640-query and four
160-query groups; clamped neighborhoods give every group 4,320 keys. One block
therefore has 1.739462 TFLOP FFN, 0.483184 TFLOP Q/K/V/O and 0.407686 TFLOP
QK/PV: 2.630332 TFLOP of counted matrix work.

| Counted matrix work | TFLOP |
| --- | ---: |
| DiT blocks, 54 x 8 | 257.439 |
| Text refiner and DiT input/output adapters | 2.222 |
| PixelUMM blocks, 36 x 5 | 473.460 |
| PixelUMM stem and head, five windows | 1.457 |
| Total | 734.577 |

The fresh unchanged-baseline diagnostic is 10.521328 TFLOP across four full
J5760 block passes, divided by 1.448511 s: 7.264 TFLOP/s of counted work per
request wall second, or 53.4% of the nominal FP32 rate. The earlier 1.582481 s
control gives 48.9%; between-run drift is not a candidate speedup. These ratios
are not ALU utilization. The conditional all-work-at-FP32-peak time is 0.774 s;
only the explicit-F32 subset gives 0.632 s. BF16-fed Q/K/V/O graph lowering is
opaque.

The historical BF16 preview receipt was reread, not rerun: 103.315 s neural
wall, 45.098 s setup and 149.110 s application total. This is 20.663 s neural
time per video second, or 29.822 s including cold setup. All counted work at
13.6 TFLOP/s would take 54.013 s; the explicit-F32 subset alone is 647.604 TFLOP
and takes 47.618 s at that assumed peak. A hypothetical double rate for opaque
BF16-fed projections gives 50.816 s. These conditional compute-only models
omit elementwise work, conversions, reductions, resource preparation and
scheduling; they are not achievable latency predictions or physical floors.
The conditional gap is about twofold, not evidence for a 10-100x kernel gain.
It does not extrapolate to higher resolutions, longer videos or quality.

The bandwidth ridge at the cited nominal rates is 34 FLOP/byte. An ideal
one-pass F32 gate GEMM has about 1,002 FLOP per interface byte. This explains
why smaller weight storage alone need not accelerate the large-row GEMMs;
actual tile reloads, conversions and attention traffic still require direct
measurement. Allocation sizes and logical interface traffic are not physical
DRAM bytes, and these estimates do not prove the whole pipeline compute-bound.

Reproduction, exact integer counts, source/artifact identities and limitations:
`python3 /opt/mldata_ssd/cogni-ml/hy-nomas-roofline-20261029/calculate_roofline.py`.
The report is `results/roofline.json` in that directory. Refresh after changes
to shapes, steps/windows, precision, kernels/backend, hardware or scoped source
and measurement identities. No new wait, readback, model or generation ran for
this calculation.

## F32 score scale inside softmax: bounded paired gain (2026-10-28)

The candidate replaces the standalone in-place F32 score multiply with an F32
multiply inside the rank-4, axis-3 softmax graph. QK and PV retain rank-3 batch
32 geometry. The scale uses the same Float32 value as the baseline. Selection
remains opt-in; rollback is the unchanged separate-scale graph selection.

The principal failure is a changed rounding/reduction boundary hidden by the
later BF16 cast. Before timing, compare probabilities and attended F32 values
tile-by-tile from identical raw QK/V, candidate first so the baseline's in-place
scale cannot corrupt its input. Require finite values and unchanged
`atol=rtol=2e-5`, followed by the saved upstream BF16 block captures. The existing
F32 softmax/PV checker consumes already-scaled scores and must not be reused
unchanged. No persisted upstream fixture currently contains all internal F32
attention tensors; native same-input checks supplement, rather than replace,
the authenticated upstream block oracle.

DoD: the correctness-only plan-reuse probe passes the direct F32 checks and
NaN control; the scale-softmax probe passes the two-window/block-0/1 upstream
captures and completes balanced ABBA/BAAB unprofiled whole-request pairs.
Every request keeps one terminal wait, one packed post-wait metrics read, no
preterminal reads or tracked encode-time named buffer allocations. Use the
cache-aware 10% reserve, 16 GiB process cap, 64 MiB swap-growth stop and 600 s
timeout. No full generation. A graph-level multiply alone does not establish
that lowering removed a panel sweep; promotion requires a measured request win.

The correctness-only release probe passed both J2048 and full J21120 at the
unchanged tolerances. At full J it compared 2,229,534,720 probability values and
86,507,520 attended F32 values tile-by-tile: zero failures, differing words or
nonfinite values. All seven saved upstream block captures also matched bitwise.
These internal F32 checks are against the native same-input baseline; they are
not a newly captured independent upstream F32 attention oracle. The NaN control
failed exactly once as intended. An external mutant applying scale twice was
rejected: 75,379,622 probability and 8,379,834 attended values exceeded tolerance
on J2048, and the saved upstream attention gate failed. No tolerance changed.

Two serial unprofiled processes each completed 40 requests: four correctness,
four ABBA warmups, and 32 timed requests in balanced ABBA/BAAB cycles. Each
request contains four complete J5760 block passes, two initial noise windows
times blocks 0/1. Each window starts from its saved upstream input; the native
block-0 output feeds block 1. This is connected two-block diagnostic timing,
not full-J timing, a complete decoder/head or generation qualification.

| Whole-request wall metric | Run 1 | Run 2 |
| --- | ---: | ---: |
| Baseline median | 1.590239 s | 2.904161 s |
| Scale/softmax median | 1.559931 s | 2.860022 s |
| Paired median time reduction | 2.17145% | 2.00291% |
| Candidate wins | 14/16 | 14/16 |
| Baseline-first paired reduction | 1.07183% | 2.03262% |
| Candidate-first paired reduction | 2.77269% | 2.00291% |

Across all 32 pairs the descriptive median reduction is 2.04714%, with 28
candidate wins. This is below the pre-existing prospective 3% promotion
screen; the candidate remains off by default. The much larger absolute drift
between processes remains
unexplained. Do not infer generation throughput, a stable host baseline or a
causal bandwidth limit from these measurements. Both arms retain 125 MPS
roots/request. The candidate omits one explicit Metal scale dispatch per tile;
the physical multiply/softmax lowering and DRAM traffic were not profiled.

All 80 requests passed their numerical gates. Correctness requests compare
six captures per plan, and every request compares all four final outputs;
all compared values match bitwise. The separate direct-F32 checker is excluded
from timing. Every request retains one terminal wait and one 464-byte packed
metrics read, zero tracked encode-time named allocations and preterminal reads.
The direct checker instead uses its existing single 176-byte metrics read.
Sampled timing-run peak footprint was 7.625 GiB, minimum cache-aware headroom
26.411 GiB and positive swap growth zero. Full-J correctness peaked at 9.606
GiB. These are sampled overlapping memory counters, not bounds on transients
or evidence of candidate-specific memory savings; both alternatives coexist.

Release builds, 37 focused CPU examples, format/diff checks and the artifact
validator pass. The validator verifies schedule, stage coverage, positive
controls, execution/report identity and source stability; eight deliberately
damaged CPU artifact controls were rejected. A correlated Mini-Quad review
independently recomputed the raw pairs and returned ROBUST for this bounded
descriptive claim. It found an incorrect timer-description string: plan body
encoding excludes comparator encoding, which is reported separately and is
included in full-request wall. Only this metadata string changed after timing;
the evidence packet pins both versions and leaves the raw reports intact.

Evidence: `/opt/mldata_ssd/cogni-ml/hy-nomas-pixelumm-scale-softmax-20261028/results/summary.json`
SHA256 `48207f1eff554fd0f7b3daadd7da72d870f05fe8485fc487b1fefc29b5213aab`.
Reproduce direct qualification with `pixelumm_plan_reuse_probe --scale-softmax
--check-scale-softmax-f32` and the pinned small/full manifests; use
`pixelumm_scale_softmax_probe` with the pinned two-window reference and BF16
weight cache for timing. Exact guarded argv, source copies, fixture hashes,
binary/object identities and guard logs are indexed by the packet. The 10%
admission expires at each launch. Refresh on computational source, fixture,
weights, geometry, precision, scheduling, lifetime, backend/device/OS/compiler,
clock or guard changes, or artifact loss. Next falsifier: a larger exact
FFN/attention change must first meet the prospective whole-request screen,
then pass connected decoder/head validation before default selection. No full
generation ran; the larger FFN/attention bottlenecks remain open.

## PixelUMM hotspot profile and packed gate/up refutation (2026-10-28)

The optimization space is not established as exhausted. The admitted profile
and candidate experiment completed; the candidate demonstrated no acceleration
and remains diagnostic-only. Production graph selection and GPU arithmetic
are unchanged. The useful pivot is from graph/encoder counts to measured work
and intermediate traffic, with whole-request time as the promotion metric.

Scope: J5760, two saved initial-noise windows and blocks 0/1, four Plan passes
per request, BF16 handoffs and F32 FFN. Within each window, native block 0
feeds block 1. This is a connected two-block diagnostic, not a complete decoder,
decoder head or full-generation qualification. The oracle is the existing
authenticated Torch 2.6.0 MPS upstream-source-method capture, not a newly run
independent backend. Keep `atol=rtol=2e-5`, attention QK/PV rank-3 batch 32 and
rank-4 softmax axis 3. No tolerance, synchronization or generation expansion.

The first profile exposed mismatched CPU clock domains: a nested Mach trace
could exceed its enclosing CLOCK_MONOTONIC API interval. API wall timers and
trace resolution admission now use CLOCK_UPTIME_RAW, matching Mach raw uptime
after timebase conversion. The existing 2 us accounting bound is unchanged.
A CPU-only nested-clock probe qualifies the relation, while preserving the
old two-output CPU-probe ABI. No Metal object or GPU wait is created by it.

The repeated baseline profile completed 102 requests: two correctness, four
warmup and 96 balanced ABBA/BAAB timed requests, 48 each with API profiling on
and off. In the off arm, median request wall time was 1.555039 s, current-thread
CPU 8.842 ms, body encode wall 723.879 ms and terminal wait 824.499 ms. Encode
wall includes MPS backpressure; it is not active CPU encoding cost. These CPU
figures exclude other driver threads. Three collections totalled 1.863 ms over
the 156.176 s observed loop. GC/host-lock removal is therefore a weak primary
lever for this workload; this does not establish a general runtime verdict.
First/last eight off-arm requests drifted from 1.427 to 1.614 s; the older
4.6 s severe drift was not reproduced or causally explained.

The tested candidate concatenates BF16 gate/up weights inside one MPS graph,
performs one logical F32 matmul, rounds its combined output to BF16, then
slices into the original output buffers. Runtime concatenation is included.
The shared input conversion does not remove gate/up arithmetic. Logical
matmul count is not a statement about MPS's physical kernel decomposition.
The bridge is linked only into `pixelumm_packed_gate_up_probe`; it retains and
restores Plan selector state under a serialized idle-caller precondition.

The unprofiled run completed 40 requests: four correctness, four warmup and
32 balanced timed requests, 16 per arm. Adjacent-pair comparisons:

| Metric | Baseline median | Packed median | Paired candidate change |
| --- | ---: | ---: | ---: |
| Whole request | 1.425029 s | 1.427755 s | +0.1228% |
| Body encode wall | 675.196 ms | 570.635 ms | -15.5643% |
| Terminal wait | 749.863 ms | 857.328 ms | +14.2570% |
| GPU root-duration sum | 1.402999 s | 1.404167 s | +0.0251% |

Packed won only 5/16 whole-request pairs; the paired range was -0.6415% to
+0.5934%. The apparent encode win moved into terminal waiting. Neither an
acceleration nor equivalence is established. Roots fell from 125 to 117 per
request, providing another counterexample to root-count-as-performance.
All compared finite capture/output words matched upstream bitwise, and the
NaN positive control fired. Each request retained one terminal wait, one
464-byte packed metrics read, zero preterminal tensor reads and zero tracked
encode-time named MetalBuffer allocations. These counters do not measure
opaque MPS/native allocations. Failed sole waits are never retried; an
undrained owner is retained rather than explicitly freeing in-flight resources.

A fresh Instruments capture covered four correctness requests, two per arm;
none qualifies for speed comparisons. Exact-PID mapping accounts for all 484
roots and 862 target Compute rows, with zero unmapped target rows. Root-duration
error and single-clock-offset spread were both about 42 ns, within the unchanged
2 us gate. Shader rows are associated by time overlap, not command-buffer
identity. Their phase/root coverage is non-exclusive, and capture/output-only
modes expose different labels. Do not add phase totals or infer exclusive
matrix cost from them. The GPU-counter export exceeded its 90 s limit; partial
counter XML is excluded. Bandwidth, occupancy, frequency and ALU saturation
remain unqualified. The trace narrows candidate areas without proving a roofline.

### What still offers a plausible lever

The following are source-derived work counts or hypotheses, not measured gains:

- At J5760/D4096/FFN12288, one FFN projection requires 579.821 GFLOPs; gate,
  up and down together require about 1.739 TFLOPs per block. Combining calls
  leaves that arithmetic intact. A faster dense path would matter, but the
  prior native F32 GEMM and lower-precision reductions failed speed or accuracy
  gates. Do not reopen them without addressing the recorded failure.
- Local attention processes 796,262,400 F32 score values per block pass (one
  window), a cumulative
  3,185,049,600 bytes of panels, plus the same probability volume. This is not
  resident memory or measured DRAM traffic: peak scores+probabilities are
  675 MiB, and caches/lowering affect physical traffic. QK and PV together
  require about 407.686 GFLOPs. Removing materialization is a potentially large
  structural lever, but its actual latency contribution remains unmeasured.
- The smallest exact attention probe is folding the standalone F32 score scale
  into softmax while preserving the F32 rounding boundary and batch geometry.
  Verify that lowering actually removes a panel sweep; a shared graph alone
  is insufficient. Compare scaled scores/probabilities/attended outputs on
  identical inputs before the existing BF16 block gates. The existing helper
  expects already-scaled scores and cannot be reused blindly for this test.
- A streamed QK/softmax/PV path could avoid score/probability storage. It is a
  higher-risk route because online softmax and tiled accumulation change
  reductions. It remains a hypothesis until unchanged intermediate/output
  gates pass; numerical algebra alone is not an equivalence certificate.
- One-time weight layout preparation or batching FFN rows from two independent
  windows might improve matrix reuse. MPS may already absorb transposes, and
  a larger M dimension may select different reductions. Require evidence of
  repeated conversion/layout work, account for extra live storage and setup,
  and do not infer a gain from fewer calls. Prior F32 staging alone was neutral
  or worse.
- Cold pipeline preparation and decoder overlap deserve separate accounting.
  The earlier five-second preview spent about 45 s preparing DiT/decoder;
  those are historical measurements, not this run. Reusing owners may reduce
  startup latency but retaining every phase's weights needs a fresh memory
  budget. Five overlapping decoder windows process 200 frame positions for
  120 displayed frames; context changes prevent naive cross-window activation
  reuse. Neither timestamp equality nor identical weights proves cache safety.

The scale/softmax experiment proposed here is reported above. The prospective
promotion policy requires repeated unprofiled balanced whole-request improvement in
both orderings, outside the observed A/A variation, with setup/deferred work
and resource growth included. Use a prospective 3% median improvement screen,
not a claim that this threshold proves significance. Hardware-limit claims
also need qualified counters and stable device state; none is made here.

All three native runs passed the 10% cache-aware reserve, 16 GiB group cap,
64 MiB swap-growth stop and 600 s timeout. Sampled group peaks were 8,180,976,952 B
(baseline), 8,989,412,664 B (packed) and 10,329,643,728 B (Instruments), with zero
positive swap growth. Lowest sampled headroom across them was 28,249,341,952 B.
Cache headroom is a reclaimability estimate and polling cannot prevent transient
spikes. The diagnostic's named inventory remains 7,381,500,468 B, including
unused 576 MiB staging and saved comparison captures; it is not a pipeline budget.

DoD commands: `make pixelumm_packed_gate_up_probe`,
`make spec_mps_api_profile_clock`, and `crystal spec -Dcpu_only` over the six
profile/schedule/weight-reuse spec files listed in the execution logs. Release
build, 44 CPU examples, two native CPU clock examples, eight Python accounting
tests and format/diff checks passed. The clock spec is excluded from CPU-only
and non-Darwin runs. Correlated adversary review is ROBUST for these bounded
numerical, lifecycle and descriptive measurement claims; severe-drift causality,
full-pipeline throughput and GPU saturation remain unverified. Rollback leaves
the packed candidate unselected; production defaults are unchanged.

Evidence root: `/opt/mldata_ssd/cogni-ml/hy-nomas-pixelumm-hotspot-20261028`.
`results/hotspot-summary-1.json` SHA256
`4a9417ef0948d3605394d751e576dd5ac87480274cb6f2efd8c4d59985a523ee`
pins reports, guard logs, execution receipts, selected analysis and final CPU
checks. Measured sources were unchanged during each run and archived per run.
The unprofiled 40-request run predates attribution-only report/attach-gate
additions and restoration of the old CPU-probe ABI; the four-request trace
qualifies the current ABI/numerical path, not a new timed comparison. Final
formatting, the test guard and Makefile test target followed the trace; the final
release binary built successfully but has no separate GPU performance claim.
Refresh on source, fixture, weights, runtime, device, geometry, precision,
schedule, backend or guard changes, or evidence loss. Admit memory afresh
before every GPU replay. No full generation ran in this experiment.

## Active experiment: prepared CogniGraph native islands (2026-10-28)

Implemented opt-in, CAUTION: share the existing CogniGraph dependency compiler with
prepared native PixelUMM islands. MPSGraph remains the projection, normalization
and attention backend. Compile access/range dependencies and copy fixed dispatch
parameters once during plan preparation. Each island borrows the current root;
it must close its encoder before any MPSGraph call or capture blit. Retain all
bound resources under the existing serialized Plan/Workspace lifetime.

The first bounded candidates are independent Q/K preparation, Q/K/V panel
gathers together with the preceding tile's independent output scatter, and
SiLU followed by product with one dependency barrier. The final tile retains
its trailing scatter. Capture blits remain ordered and may split the FFN island.
No operation may cross an opaque MPSGraph boundary. The control keeps the
original encoders; selection is explicit and defaults off. This is reuse of CogniGraph scheduling,
not a new LTP/WBA certificate or a claim about the historical Nomic MoE speedup.

Prediction: fewer native encoders can reduce whole-block request time without
changing BF16 rounding or MPS reduction geometry. Strongest failure: concurrent
dispatches alias mutable scratch or escape the current MPS root. Guard it with
shared scheduler dependency tests, prepared-plan validation, poisoned transients,
the six upstream capture points per block (`post_attn_norm`, `gate_proj`,
`up_proj`, `product`, `down_proj`, `output`), output-only candidate comparisons, and
one terminal wait/readback. Keep `atol=rtol=2e-5`, a cache-aware 10% reserve,
the existing 16 GiB process cap and 64 MiB swap-growth bound. No full generation.

Implementation/rollback: factor the scheduler in `src/ml/metal/compute_graph.cr`,
add its CPU-testable native schedule adapter, and integrate opt-in prepared
islands in `src/ml/video/pixelumm_decoder_block_mps_bridge.mm`. Add a balanced
same-process probe over the pinned J5760 two-window/two-block fixture. Rollback
is leaving the candidate disabled; do not change production selection without
numerical and paired whole-block evidence. DoD: focused scheduler/probe specs,
release `pixelumm_cognigraph_probe`, unchanged upstream gates including a seeded
nonfinite control, then guarded ABBA/BAAB whole-request timing with both arms
sharing weights, inputs, scratch and MPS graphs. Record root/encoder counts,
host encode wall/CPU, terminal wait, GPU intervals and peak memory separately.

Observed result: all 40 requests pass the unchanged upstream gates. The baseline
and candidate correctness requests each compare six stages for each of four
plans; every request compares all four block outputs. All compared words are
bitwise equal to the saved authenticated upstream-source-method MPS captures;
the seeded NaN control detects exactly once. This is J5760, blocks 0/1 on two
windows, not a fresh connected upstream run, full-J21,120 qualification, the
36-block decoder, head or full generation. Optional Q/K captures and the
SiLU-capture fallback have source review but no direct runtime qualification
from this fixture.

Four ABBA+BAAB cycles give 16 adjacent timed pairs. Raw full-request medians are
1.599111 s baseline and 1.602950 s candidate; paired median change is -0.1124%
(mean +0.0897%, range -2.145% to +2.716%, eight wins in sixteen pairs). This run
does not demonstrate a meaningful speedup; it does not prove zero effect.
Calling-thread CPU medians are 12.057/13.718 ms, with a noisy paired median
change of +20.165%. Each arm retains 125 MPS roots/request. Native encoder
counts of 66/29 per block are prepared-plan model counters, not measured API
calls. Recipes, bindings and dependency waves are reused; Metal commands are
still encoded into each current root, not replayed from a preencoded command
buffer. Candidate selection remains off by default.

Every request has one terminal wait, zero tracked named encode allocations or
preterminal readbacks, and one 464-byte metrics read after completion. Sampled
process-group peak is 7.619 GiB against the 16 GiB cap; minimum cache-aware
headroom is 29.868 GiB and swap growth is zero. The 10% reserve remains intact.
The footprint includes both retained alternatives; it is not a candidate-only
memory comparison. Sampled limits do not bound inter-poll transients.

Validation: release build and fresh `make -q`, 37 focused CPU examples, four
Python analyzer tests, three existing GPU ComputeGraph examples, format/diff
checks, and the guarded 40-request run pass. The guard launches the binary
directly so its PID matches the report. Its analyzer rejects missing process
liveness, wrong guard policy, changed tolerance/pins, extra reads and wrong
mode transitions. The execution receipt pins unchanged measured source and
binary hashes. Correlated Mini-Quad verdict: ROBUST for bounded correctness and
the absence of a demonstrated gain in this run; no general speedup claim.

Evidence: `/opt/mldata_ssd/cogni-ml/hy-nomas-pixelumm-cognigraph-20261028/results/summary.json`,
SHA256 `26b34c70cb12f4cbcdb9e0e452086f62a5a4c7d37f1acae9220e53f97d830dac`.
It pins the raw report, guard, execution receipt, analysis sources and curator.
Refresh after scoped source/artifact/backend/device/OS/toolchain, geometry,
precision, scheduling, lifetime, guard or clock changes, or evidence loss;
memory admission is fresh per launch. Next falsifier: use the existing phase
and root profile on identical operands to target dominant GPU FFN/attention
work; fewer host encoders alone did not improve this request. Ordinary GC
remains selected. The historical preview boundary below is not a new generation
launch request.

## Preview scope and admitted composition

The user authorized a real five-second preview before further kernel tuning.
The first bounded target is one first segment, 120 frames at 24 fps,
512x288, seed 42, CFG 1, eight shifted Euler steps (shift 5), camera/action
conditioning off. Its DiT grid is `[30,9,16]` (4320 tokens). PixelUMM uses the
pinned preset's ten-latent windows, five-latent overlap, no spatial cropping,
and the existing eight-token local attention layout on grid `[10,18,32]`.

BF16 remains the default DiT control. The user also authorized an experimental
Q4_K DiT preview with the identical pinned inputs and schedule for visual
comparison. PixelUMM and the global DiT adapters remain BF16. Q4 is decoded on
GPU into the existing single-block BF16 stage; it does not introduce another
wait or a CPU activation handoff. This experiment does not promote Q4 quality
or change the numerical tolerance. Same-decoded-Q4 numerical parity does not
establish acceptable image quality: the recorded Q4 versus
BF16 full-forward discrepancy is large. The first prompt may reuse the
authenticated saved Qwen state-26 crop; that boundary must be stated in the
run receipt rather than presented as live native text encoding.

## Existing anchors and missing composition

- The 54-block DiT, input/final adapters and text/time refiner are qualified
  separately on the saved first-segment fixture.
- The 36-block PixelUMM decoder plus head are qualified separately on real
  stem activations. That fixture is not a generated video.
- The missing production edges are GPU Euler integration, ordered repeated
  forwards with a changing GPU timestep, PixelUMM input stem, window decode
  and cosine merge, display-range conversion, and terminal MP4 export.
- Pinned upstream: `911cea6458a89c2663513ff7099561ec190dd491`.
  Existing source and weight authentication remain mandatory.

## Risk, resource boundary and rollback

CAUTION: this composes previously separate model phases and introduces repeated
use of fixed scratch. The strongest failure is stale/overwritten data from an
incorrect pass or window order, followed by plausible but invalid pixels.
Reject wrong ordering before appending work, retain all GPU owners until their
chain drains, and check finite output and nonconstant frames. Keep the old
single-forward loader contract as the default; repeated forwards are bounded
and explicit. New generation entry points stay opt-in.

Keep a cache-aware 10% physical-memory reserve, a fixed process cap, and a
64 MiB maximum system swap increase. Run heavy GPU tasks serially. Admit each
weight phase before allocation. DiT weights and decoder weights cannot simply
be added to the same residency budget. If a phase handoff requires draining a
chain to release weights, report that synchronization explicitly; retain the
small latent GPU buffer across the handoff and do not round-trip it through
the CPU. There are no waits or activation readbacks inside diffusion steps,
decoder blocks or window assembly.

Rollback is removal of the opt-in entry point and new glue, with the default
single-forward probes and current MPS baseline unchanged. Preserve unrelated
working-tree files. All temporary outputs and build products live on
`/opt/mldata_ssd/`.

## Definition of done and falsifiers

1. Run the scheduler/order specs and the new stem/merge oracle comparison at
   the unchanged `atol=rtol=2e-5`; nonfinite negative controls must fail.
2. Run a guarded native first-segment request through eight DiT forwards,
   all 36 decoder blocks and head for each of five windows, followed by GPU
   merge/postprocess. Record named allocations, encode readbacks, terminal
   waits, elapsed times, peak footprint, and guard verdict.
3. Export only completed display-range F32 pixels using
   `examples/hy_nomas_export.py`; verify 120 frames, 512x288, 24 fps, five
   seconds, finite and nonconstant decoded frames. Inspect a contact sheet and
   the actual clip. A valid MP4 alone does not certify generation quality.
4. Compare the newly composed numerical boundaries with pinned upstream on
   identical inputs. Keep execution parity, image quality and performance as
   separate claims. A failing boundary blocks a verified parity claim; never
   expand the tolerance to pass it.

The bounded native execution and terminal export below are VERIFIED within
their stated scope. Full-video upstream parity and general prompt support
remain open. No native generated clip was available at the start of this work.

## Boundary checks, 2026-10-28

The fixed input manifest is
`5b162b8bf4b28c6cfacd8a3d8c14a7758b699034b2a46e84c224522eb0a75a7b`.
It carries the saved natural-tram conditioning, MPS-generated seed-42 noise,
and pinned upstream RoPE tables for the preview geometry. This is a fixed
pilot, not a general prompt CLI. Initial F32 noise is rounded through BF16 on
GPU before the first DiT forward. The native DiT retains its qualified F32
arithmetic; the Euler boundary rounds model velocity and each updated latent
through BF16 as the source pipeline does.

The pinned `EulerSolver` module produces a 256-element MPS oracle under
Diffusers 0.36.0. Native initial rounding, all eight timestep values and the
final latent are bit-exact. The probe has one packed terminal diagnostic read
and one wait, with no preterminal activation read or wait. Its fixture SHA is
`3fce6402ec76036437dc0bdfee9f07008610bdca23c040c2bab55d339777d6de`.

The real-weight PixelUMM stem and synthetic five-window temporal assembly
also match the pinned MPS source bit-exactly, with a NaN rejection control.
The probe records zero encode allocations/readbacks/preterminal waits and one
terminal wait. The corrected oracle manifest SHA is
`3f6976cbd8d1efeeb3aa663b817980cba78b8d793e7ed56e90384278a535409f`.
An earlier oracle was invalid: large MPS int64 remainder operations corrupted
the synthetic inputs. Constructing the exact 8192-value period before repeating
it removes that oracle error. The source display conversion also requires
separate BF16 rounding after multiply and add, before F32 export. Neither fix
widens the tolerance. The earlier failing artifacts remain outside the repo.

The immutable DiT weight cache records its V5280 qualification geometry.
The preview authenticates that original cache geometry separately from its
V4320 activations. A CPU check permits only this video-token extent to differ
and requires identical parameter bindings; cache validation itself is unchanged.
Runtime buffers, plans, RoPE and memory accounting use V4320 throughout.

Exact named residency is 21,011,384,964 bytes for DiT and 16,563,852,432 bytes
for the decoder phase. The external guard uses a 23 GiB process-tree cap,
10% cache-aware reserve, 64 MiB swap-growth stop and 900 s timeout. These are
admission and sampled runtime guards, not guarantees against between-poll spikes.

## First completed preview, 2026-10-28

The accepted `generation-v3` process completed on Apple M2 Max, 64 GiB unified
memory. It generated the fixed prompt about a yellow tram in a wet downtown
street. `preview.mp4` contains 120 frames, 512x288, 24 fps and exactly five
seconds, independently checked with ffprobe and decoded to a contact sheet.
The sampled frames contain a recognizable tram, buildings and umbrellas;
the tram position changes over time. This inspection is not an upstream
quality comparison or a certification of temporal consistency.

| Phase | Measured wall time |
| --- | ---: |
| DiT preparation | 22.877 s |
| Eight DiT steps | 35.768 s |
| PixelUMM preparation | 22.221 s |
| Five decoder windows and display conversion | 67.547 s |
| Guarded process, including CPU admission and terminal file output | 155.672 s |
| MP4 export | 0.417 s |

This is about 31.2 seconds per second of video including startup and export,
or 20.7 seconds per second for the two neural execution phases. These are one
preview's measurements at 512x288, not a warmed benchmark, a result for 480p
or 720p, or a comparison with Wan or another backend.

The guard recorded 300 samples: peak process-group footprint 21,340,353,160
bytes (19.874 GiB), minimum cache-aware headroom 23,171,694,592 bytes
(21.580 GiB), swap growth zero, exit zero. Both neural phases report one
terminal wait and zero preterminal waits. Allocation/readback counters report
zero only for instrumented MetalBuffer methods; they do not measure opaque
MPS resources or the direct terminal pixel-file export. The receipt now states
that scope explicitly. The one weight-phase handoff is an intentional wait;
there is no CPU round trip for the latent.

The completed pixels have SHA256
`f13481b1d5790761df94943b6eac9e768177a240998c859d3ea9f21a633c2f85`;
the exported MP4 has SHA256
`7a323eb0b9e02443eedf45a5576f36a3c738ea41b687d1a476cbbdcfd0ea1cd3`.
All 120 display-range frames are finite and nonconstant. Frame standard
deviation ranges from 0.2316 to 0.2433; consecutive-frame mean absolute
differences range from 0.02968 to 0.06836. These statistics reject empty or
frozen output; they do not establish acceptable motion quality.

Artifacts live outside Git under
`/opt/mldata_ssd/cogni-ml/hy-nomas-first-video-20261028/`: `inputs.json`,
`generation-v3.log`, `generation-v3.guard.jsonl`, `results/receipt.json`,
`export.json`, `ffprobe.json`, `video-analysis.json`, `contact-sheet.jpg`, and
`preview.mp4`. The original receipt precedes the metric-scope wording addition;
its counter interpretation is the scoped one above. Earlier failed attempts
are retained separately. One stopped before loading weights on the recorded
cache-geometry mismatch; another completed GPU work but rejected Crystal's
unsupported `wx` file mode. The writer now uses O_EXCL and a CPU no-clobber
regression check. Neither failure was a successful video result.

Final checks: release builds, 39 Crystal examples, 13 Python tests, formatting
and diff checks; bit-exact upstream checks for the newly composed Euler and
stem/merge boundaries as described above. Correlated adversary review found
no enqueue-order/lifetime blocker and required the metric-scope correction.
This does not add a full generated-video upstream comparison.

## Experimental Q4 preview, 2026-10-28

The opt-in `--dit-weights q4-k` selector uses the authenticated existing
`HyNomasQ4Stack`, including its eight-forward order/chain limit. The default
remains `bf16`. Only DiT block weights differ: all pinned input bytes, global
adapters, Euler schedule, decoder noise, BF16 PixelUMM weights, and export
settings match the accepted BF16 preview. This is the staged Q4_K path, not
the separate direct-Q4 GEMM experiment. Compact block payloads occupy
4,593,162,240 bytes. Their GPU decode reuses one BF16 block stage; the complete
Q4 DiT phase admits 9,290,402,436 bytes instead of 21,011,384,964 for BF16.
The decoder admission remains 16,563,852,432 bytes and dominates the Q4 peak.

DoD: release build, focused selection/admission/order specs, CPU admission in
both modes, invalid-mode rejection before Metal, and one guarded complete
preview with the identical input manifest. Memory estimates must match actual
owner admission. A mode mismatch, input/schedule mismatch, added encode-time
readback/wait, guard stop or nonfinite output rejects the execution result.
Q4-versus-BF16 quality remains a visual experiment; it is not a passing 2e-5
parity test. Existing same-decoded-weight kernel gates retain that tolerance.

The Q4 run completed all 120 frames at 512x288/24 fps. Contact frames show a
recognizable tram but substantially more blurred and distorted buildings and
tram contours than the BF16 control. Q4 quality is not promoted. The raw output
is finite and nonconstant in all 120 frames, and ffprobe independently confirms
the frame count, geometry, frame rate and five-second duration. These checks
establish execution and export, not acceptable perceptual quality.

| Phase | Prior BF16 launch | Q4 launch |
| --- | ---: | ---: |
| DiT preparation | 22.877 s | 7.296 s |
| Eight DiT steps | 35.768 s | 48.976 s |
| PixelUMM preparation | 22.221 s | 22.313 s |
| Five decoder windows/display | 67.547 s | 85.906 s |
| Guarded process | 155.672 s | 168.534 s |
| MP4 export | 0.417 s | 0.311 s |
| Sampled peak group footprint | 19.874 GiB | 16.784 GiB |

These are separate single launches, not an interleaved speed experiment.
The unchanged decoder also took longer in the Q4 launch. Do not attribute the
whole timing difference to quantization or claim an acceleration. Faster DiT
preparation and lower resident bytes do not remove PixelUMM compute cost.
All weight files currently resolve through `/opt/mldata_ssd` to the external
SSD. Disk reads occur during setup; the block payloads are GPU resident during
encode. A local mirror could improve startup but has not been benchmarked.

The Q4 guard recorded 324 samples, peak footprint 18,021,842,016 bytes, minimum
cache-aware headroom 23,878,877,184 bytes, zero swap growth and exit zero under
the unchanged 10% reserve, 23 GiB process cap, 64 MiB swap stop and 900 s deadline.
There is one terminal wait per phase, zero preterminal waits/readbacks and zero
instrumented MetalBuffer encode allocations. Opaque MPS resources and terminal
pixel-file export remain outside the counter scope. The Q4 receipt explicitly
records the mode, experimental quantization, BF16 global/decoder weights and
Euler boundary rounding; full generated-video upstream parity remains untested.

Artifacts: `/opt/mldata_ssd/cogni-ml/hy-nomas-q4-preview-20261028/` contains
`launch.json` (source/binary/baseline digests), `plan-bf16.jsonl`,
`plan-q4.jsonl`, `specs.log`, `invalid-mode.log`, `generation.log`,
`generation.guard.jsonl`, `results/receipt.json`, `comparison.json`,
`export.json`, `ffprobe.json`, `contact-sheet-comparison.jpg`,
`preview-q4.mp4`, and `comparison-bf16-left-q4-right.mp4`.
Q4 pixels SHA256:
`dc7a80be9c802b134e91ae784f7be0c05a5bae398cdd1fdd7e8cfe2d6cff94ae`.
Q4 MP4 SHA256:
`fb4cfacc8f7a8abd2be264e9a2f083dad2bbe6c6837afa572b62a505f3434074`.
The focused suite passed 21 examples. Correlated adversary review found no
composition/lifetime blocker; it required explicit Euler-rounding wording.
The final artifact review returned ROBUST for matching input/schedule/decoder
identities, guarded execution and finite output; it does not qualify visual
quality, full upstream parity or a speed improvement.

## Duration boundary

The native configuration explicitly admits one first segment and rejects
history/multiple segments. The current input, latent, RoPE and assembly are
fixed to 120 output frames. Five overlapping decoder windows are not five
autoregressive segments. Upstream implements segment continuation with K/V
history and temporal RoPE offsets; that state contract is not ported here.

Pure division of the first BF16 launch gives about 1 minute 55 seconds of
independent five-second output per wall-clock hour at 512x288. Dividing only
the neural phases gives about 2 minutes 54 seconds per hour, excluding setup
and requiring a future reuse design. Neither number is sustained throughput
or a maximum continuous duration. In particular, two continuous minutes
cannot yet be requested from this entry point. Long-duration timing and memory
must be measured after continuation is implemented; history adds work and
memory absent from the present preview.

## Next optimization frontier

The Q4 preview was visually rejected by the user. BF16 remains the quality
control and the active performance target; do not promote Q4 on memory savings
or projection-level numerical tests alone.

PixelUMM accounts for 65.4% of the two neural phases. Weight authentication,
loading and plan construction account for another 45.1 seconds of the native
request and should be separated before proposing a persistent owner. Do not
remove integrity checks to make startup appear faster.

The next exact-path probe should group independent same-sized local-attention
tiles into batched MPS graphs while retaining each tile's context, BF16
boundaries and reduction order. First measure the current 5760-token window
by FFN, projections, attention and command transitions; a reduction in API
calls alone does not establish a full-window speedup. Compare identical inputs
against the current implementation and pinned source at the unchanged 2e-5,
then measure the complete decoder window. This is PROPOSED, not implemented.
The overlapping windows decode 200 frame positions to produce 120 output
frames, but later-layer overlap activations depend on different attention
contexts and cannot be reused solely because their frame indices match.

The bounded block-zero probe now admits T10/H18/W32 (J5760), the preview's
window geometry, including all 26 intermediate captures. It uses actual
pretrained weights and seeded synthetic hidden states, not the generated
clip's activations. The unchanged native implementation matches all 26 tensors
bitwise against the pinned upstream source under PyTorch 2.6.0 MPS at the
unchanged 2e-5 gate. The existing decoder-head control and poisoned alternate
replay also match all five tensors bitwise. Each request retains one terminal
wait, zero instrumented encode-time named allocations and zero preterminal
activation readbacks. All launches passed the cache-aware 10% floor, process
footprint cap and 64 MiB swap-growth guard. Evidence is under
`/opt/mldata_ssd/cogni-ml/hy-nomas-pixelumm-bf16-perf-20261028/results/`:
`native-torch26-all.json`, `native-torch26-admission-all.json`,
`head-torch26-original.json` and their guard logs.

An initially mixed PyTorch 2.8 oracle changed inverse-root lowering and falsely
suggested a native RMS regression. A proposed RMS replacement matched that
oracle but failed the existing 2.6 head control; it was reverted. Preview MPS
fixture production now rejects a runtime other than 2.6.0 before GPU work and
records interpreter/package provenance. The native J5760 consumer also rejects
other runtimes or CPU fixtures before payload access and reports the reference
producer's preflight separately from its own memory guard.
A runtime migration requires a separate
consistent block/head lineage, not a relaxed tolerance.

The J5760 layout has 12 query tiles and two distinct ordered key lists. A
candidate reduced K/V gathers from 24 to four while preserving intermediate
results in the isolated 2.8 experiment. Four ABBA launches gave profile-off
full-block medians of 0.515/0.531/0.571/0.594 seconds; GPU drift dominated the
roughly 0.6% balanced difference. No reliable speedup was established, so that
candidate and the experimental RMS code remain outside production. Preserve
the rank-three batch-32 matmuls and rank-four softmax axis 3. The retained
external patch and reports document the rejected route, not a production gain.

### Measured BF16 startup improvement

The BF16 entry point previously authenticated all Q4 payloads three times
through the shared weight index before computing with separately authenticated
BF16 controls. It now creates one request-scoped `HyNomasQ4StackIndex::Admission`
and refreshes its captured file identities, sizes and modification times at the
two later admission boundaries. Construction still performs the original full
index/manifest/payload hash and inventory checks. Every consumed BF16 control
and the globals still receive their full SHA-256, inventory and direct-copy
checks. Q4 inference retains its full loader path and rejects the reusable BF16
admission. No GPU arithmetic, dispatch, synchronization or precision changed.

The freshness guarantee is stat-based: replacement, removal or an mtime change
rejects across all five file classes. A same-inode, same-size Q4 byte mutation
with restored mtime can evade refresh; this is not fresh content authentication.
The BF16 runtime does not consume those Q4 bytes. The corresponding same-stat
mutation of a consumed BF16 control still fails its full hash before parsing
or Metal allocation. Reuse is local to this request, not a persistent trust
cache or authorization for skipping BF16 hashes.

A native Crystal CPU benchmark compared three complete index loads with one
complete admission plus two refreshes, including the first full scan in each
case. It used the same 54 ordered blocks and 4,593,162,240 Q4 payload bytes,
one untimed full-scan warmup, no cache purge, and ABBA followed by BAAB. Across
four timed cases per arm, medians were **6.136654 s versus 2.055800 s**, a
2.985x ratio and 4.080854 s difference for this authentication portion. Both
order groups retained approximately 3x ratios. The last two cases slowed
together (10.050/3.427 s); these are warm-cache startup observations, not a
cold-start guarantee or a GPU/complete-video speedup. All identity fingerprints
matched and tracked Metal allocations were zero. The cache-aware guard passed
the 10% floor, 2 GiB process cap, 64 MiB swap-growth stop and 180 s timeout:
85 samples, 8,470,936 B sampled peak, no swap growth.

The real supplied-admission BF16Stack constructor also authenticated all 54
BF16 controls (16,314,144,768 payload bytes), admitted exact raw-copy plans and
released with no Metal allocation. The release preview binary passed
`--plan-only` on the pinned input, reporting BF16 selection and the unchanged
21,011,384,964/16,563,852,432 B DiT/decoder named budgets. No generation ran.
The receipt now records `preflight_seconds` and `request_seconds`; the latter
includes preflight and input upload through pixel-file export. Historical
`total_seconds` continues to start after preflight/input upload, so compare
like scopes. External MP4 encoding and process startup remain excluded.

DoD: release `make hy_nomas_generate`; CPU-only specs
`hy_nomas_q4_stack_spec.cr`, `hy_nomas_generation_admission_spec.cr` and
`hy_nomas_bf16_serial_admission_spec.cr` (23 examples passed); native
`make spec_hy_nomas_bf16_cache` (18 examples passed); real guarded authentication
benchmark and BF16 control admission; guarded preview `--plan-only`;
format/diff checks. An initial CPU-only run of the broader BF16 cache suite hit
two existing nested-`pending` errors in its Metal-required cases; those cases
passed in the proper native suite. No unrelated skip/stub code was changed.
Evidence in the same external namespace: `tmp/authentication_benchmark.cr`,
`results/authentication-paired.jsonl`, `authentication-summary.json`,
`authentication-bf16-controls.jsonl`, `generation-plan-admission.jsonl` and
their guard logs. The summary pins benchmark source/binary and preview binary
hashes. Correlated mini-Quad verdict is ROBUST for this startup change and its
stated stat boundary; unchanged block/head math retains the numerical control
above. This does not requalify complete-video quality or throughput.

### Bounded FFN weight staging: exact, no established speedup

Each independent decoder window currently feeds BF16 FFN weights into graphs
containing F32 casts. Amortizing those casts across five windows was a possible
lever; the bounded two-window experiment below did not establish a gain.
Simple loop reordering is
invalid: prepared graphs bind fixed inputs/outputs and the shared two-bank
workspace cursor requires complete slot-ordered passes. A block-major runner
needs separate window activation banks and explicit per-window graph bindings.
One block's three F32 FFN matrices require 576 MiB of staging; retained current
and next states for five windows require 450 MiB, about 315--360 MiB more than
the current stem/output banks. Opaque MPS resources need separate measurement.

The diagnostic compares two J5760 windows through blocks 0 and 1: ordinary
BF16 graph feeds against block-major reuse of one 576 MiB F32 staging set.
It supports both window-major and block-major baseline ordering so a same-order
control can isolate staging from loop reordering.
Each window/block has a distinct output binding; all four plans share transient
scratch and both graph variants remain resident. GPU conversion of all three
FFN matrices is included once per block in the staged timing. Staging must not
alias weights, live window states, captures or metrics, and the next block may
overwrite it only after both windows have consumed the current block.

Inputs are overlapping slices of the saved preview initial-noise conditioning,
passed through the actual pretrained upstream stem, at latent starts 0 and 5
(noise-frame starts 0 and 20). They are not final generated DiT latents. The
fixture must retain this provenance limitation and exact source, payload and
weight identities. Local temporal RoPE resets in each upstream decoder window.

DoD: build `pixelumm_ffn_weight_reuse_probe`; pass its CPU schedule/admission
specs and strict GPU comparison of post-attention normalization, gate, up,
product, down and output for both windows/blocks against pinned Torch 2.6.0
upstream at unchanged `atol=rtol=2e-5`; reject poison/known mismatches and verify
staged weight words exactly. Only then run four ABBA warmups and eight timed
ABBA/BAAB requests with intermediate capture copies disabled, staging included,
one terminal wait and one packed metrics read. Track named and opaque resources,
fresh cache-aware admission, the 10% reserve and the 64 MiB swap-growth stop.
No full generation or production schedule change is admitted by this probe.

Risk: stale fixed graph bindings or staging overwrite could silently mix
windows/blocks; distinct retained outputs, schedule guards, poison and per-boundary
upstream checks are the falsifiers. Rollback is removal of the new diagnostic
target and sources; production math and defaults remain unchanged.

Both arms matched all 24 upstream captures bitwise. The staged correctness
request additionally checked 603,979,776 expanded F32 weight words across its
four consumers, with no differences, failures or nonfinite values. Poisoned
buffers were refreshed before every request and the NaN positive control fired.
Each launch used two correctness requests, four ABBA warmups and eight timed
ABBA/BAAB requests, four per arm. Timed requests disabled intermediate capture
copies and exact-weight checker kernels, included both block conversions,
and retained four terminal output comparisons. Every request had one terminal
wait, one 464-byte packed metrics read, zero tracked encode-time MetalBuffer
allocations and zero preterminal readbacks. These counters do not prove zero
Crystal or Objective-C allocations.

Profile-off request medians on Apple M2 Max, in seconds:

| Baseline order / run | Ordinary BF16 graph | Shared F32 staging | Wall reduction |
| --- | ---: | ---: | ---: |
| Window-major / initial | 1.459559 | 1.465904 | -0.435% |
| Block-major / first control | 2.319832 | 2.312451 | +0.318% |
| Block-major / corrected report | 1.700959 | 1.710128 | -0.539% |

The initial comparison also changed scheduling. The two same-order controls
have mixed signs and substantial process-to-process drift; the corrected run
was slower in three of four adjacent pairs. Its body-encoding median fell
from 0.793150 to 0.628362 seconds while terminal waiting rose from 0.907483 to
1.086075 seconds. Encoding overlaps GPU work and may block inside Metal/MPS;
this is not evidence of a corresponding CPU saving or faster inference.
The measured GPU interval is the command-chain span, not isolated kernel time.
Request wall time also includes common poison/comparators, terminal readback
and a memory sample. No reliable staging speedup or default promotion follows.

The named live inventory was 7,381,500,468 bytes, including the 603,979,776-byte
staging set and 2,264,924,160 bytes of expected captures. Native launches passed
the cache-aware 10% reserve, 16 GiB footprint cap, 64 MiB swap-growth stop and
600-second timeout. Across them, sampled peak footprint was 8,182,353,184 bytes
and swap growth was zero. The corrected launch retained at least
37,270,994,944 bytes of cache-aware headroom. Upstream production passed its
separate 12 GiB cap with a 5,910,612,224-byte sampled peak and zero swap growth.
Named bytes, GC heap, RSS, footprint and Metal device allocation overlap and
must not be added; opaque MPS bytes were not independently estimated. These
diagnostic capture inventories do not forecast full-pipeline memory.

The first two reports accidentally retained an alias to the native handle
array, which release then cleared. Preparation guards and nested resource
metadata still verified four distinct plans. The runner now snapshots the
array before release; the corrected run also verifies all four top-level
handles. Earlier reports and their measured source/binary remain preserved
under `measured-source-v1/` and `results/summary-v1.json`, without rewriting
the original evidence.

Evidence root:
`/opt/mldata_ssd/cogni-ml/hy-nomas-pixelumm-ffn-staging-20261028/`.
`results/summary.json` SHA-256 is
`8905d2e45b8ac37f14ff5c2c9b84c9de7ede986dc5b4cb4415d6dfbe80123bec`;
it pins execution sources, binaries, the upstream fixture, raw reports and
guards, including the preserved earlier lineage. Fixture metadata SHA-256 is
`e9b1ddce57fac94efb0262bd3e5f0be2963ef9df4255dc74cf254ff4b0217ddc`.
Release build, eight Crystal CPU examples, nine Python CPU tests, format/diff
checks and the guarded numerical/timing replays passed. Correlated mini-Quad
adversary verdict is ROBUST for this fixed two-window/two-block numerical reuse
boundary; a speedup claim remains unsupported. No full generation ran.

The next proposed exact-path falsifier is batching independent windows' FFN
rows, following Qwen's prefill GEMM routing, while keeping attention contexts
and local RoPE separate. Another small candidate is fusing SiLU/product while
retaining the intermediate BF16 rounding in registers. Current shared scratch,
direct block-to-block GPU handoff and prepared graphs already reuse the Qwen
lifetime pattern. Qwen decode GEMV bandwidth gains and GDN affine recurrence
parallelization do not directly apply to J5760 PixelUMM FFN/softmax attention.
None of these proposals has a speedup certificate here.

Refresh all execution claims after changes to source, weights, input identity,
geometry, precision, schedule, command ordering, backend/device or guard.
Each launch requires fresh memory admission; missing external artifacts reopen
the corresponding numerical or performance claim.

### Bounded PixelUMM drift localization (2026-10-28)

The GC experiment below observed a common GPU-chain increase from about 2 to
4.6 seconds, while its single natural collection lasted about one millisecond.
The diagnostic keeps the same J5760, two-window/two-block BF16 inputs,
four resident Plans, shared scratch and window-major schedule. Compare ordinary
unprofiled requests with interleaved requests using the existing native phase,
Metal/MPS API and continuation collectors. Profiling must not split roots or
add commands, waits, tensor transfers or conversions.

This is CAUTION diagnostic work. The strongest false positive is assigning
Metal/MPS submission backpressure to CPU computation, or summing overlapping
root intervals as independent GPU phase costs. Preserve shared root candidates
at phase and plan boundaries. Keep graph-inclusive and nested continuation
times separate, and report whole-chain interval union and envelope. A
profile-on/off comparison is required because CPU instrumentation can perturb
submission timing. No production optimization is admitted by this probe alone.

DoD: release-build `make pixelumm_drift_probe`, pass its focused CPU accounting
tests, and run two correctness requests, four warmups and 96 balanced timed
requests under the existing memory guard: 16 GiB process-group footprint,
10% cache-aware reserve, 64 MiB swap-growth stop and 600-second timeout. Both
correctness requests must pass all 24 pinned upstream captures and the existing
NaN positive control at unchanged atol=rtol=2e-5. Each request retains one
terminal wait and one packed metrics read; tracked encode-time buffer
allocations and preterminal readbacks remain zero. CPU-only profile extraction
follows that wait. Full generation is excluded. If drift does not recur, report
its non-reproduction instead of inventing a causal attribution.

Rollback removes this diagnostic and its optional probe-only profiling hook;
production math, synchronization and GC defaults remain unchanged. Refresh the
evidence after changing sources, inputs, geometry, precision, scheduling,
backend/device or guard settings.

Observed result: all 102 requests completed (two correctness, four warmup,
96 timed; 48 timed/profile arm) in a 167.121-second measured loop. The original
4.6-second severe drift was not reproduced; its preceding GC experiment ran
454.663 seconds. This shorter run reproduced a smaller chronological increase
in both arms. The last GPU root-duration sum was 2.076724 seconds. First/last
16 timed-request medians, including both profile arms:

| Observation | First 16 | Last 16 |
| --- | ---: | ---: |
| Sum of GPU command-root intervals | 1.405981 s | 1.862077 s |
| Host encode wall | 0.676150 s | 0.877546 s |
| Host encode thread CPU | 11.095 ms | 9.218 ms |
| Terminal wait envelope | 0.751987 s | 1.004449 s |

The normal-path terminal wait includes encoder closure. The native GPU elapsed
value is the sum of command-root intervals, not exclusive kernel-active time
or the whole-chain envelope. In the 48 profiled timed requests, the median
interval-union gap was 74.4 us (maximum 88.2 us), with 1.63 ms median overlap
between roots. This rules out large gaps *between those root intervals* for
this run; it does not measure stalls inside them or establish GPU utilization.

The same root topology, 125 roots/request, was retained in both arms. Comparing
the first/last eight profiled timed requests, nested `commitAndContinue` wall
time increased as follows. These call spans identify the CPU-side location of
waiting and must not be added to inclusive MPSGraph time or treated as the
exclusive GPU cost of the named phase.

| Metal/MPS API phase | First eight | Last eight |
| --- | ---: | ---: |
| FFN | 273.532 ms | 381.803 ms |
| Self-attention preparation | 158.545 ms | 210.163 ms |
| Self-attention | 226.477 ms | 270.613 ms |
| Self projection/residual | 7.663 ms | 9.128 ms |

FFN continuation wall overlaps earlier submitted roots for a median 99.983%
of call time early and 99.988% late (per-request wall-weighted fractions).
The root being submitted by the current call is excluded from this calculation.
Root intervals whose sole marker-candidate phase kind is gate/up also increase
from 0.400821 to 0.566966 seconds; shared boundary roots retain every candidate.
These are root durations rather than exclusive kernel timings. FFN is the
largest measured continuation-wait contributor, while several phases grow.
The evidence supports descriptive localization to widening command-root
intervals and Metal/MPS call waiting; the underlying cause remains open.

Instrumentation control: the 48 adjacent on/off pairs have median GPU
root-duration-sum change +0.0808% (range -3.292% to +5.868%) and total host wall
change +0.1036%. Encode thread CPU changes +8.425%, so the observer is not
neutral. Both arms drift, but these results do not establish equivalence or a
production performance change. Three natural collections occur after the wait,
during profile extraction: 2.070416 ms total collection envelope, 1.829666 ms
rendezvous-inclusive pause, 1.809873 ms fully stopped, overlapping categories.
They cannot account for the observed seconds-scale change. Advisory thermal
state changes from nominal to fair at request 50, after drift has started;
this is not a GPU clock measurement or evidence of thermal causation.

Numerical and guard DoD passed: both correctness requests match all 24 saved
upstream Torch2.6.0 captures bitwise, all timed output comparisons pass, and
the NaN positive control fires at unchanged atol=rtol=2e-5. Every request has
one terminal wait, one 464-byte packed read afterward, zero preterminal reads,
zero tracked encode MetalBuffer allocations and zero stage conversions.
No GPU math, command geometry, production defaults or GC policy changed.
Sampled peak footprint was 8,180,731,168 bytes (7.62 GiB), minimum cache-aware
headroom 32,139,329,536 bytes, minimum pressure-free 80%, with zero positive
swap growth across 360 guard samples. Categories overlap; the diagnostic's
7,381,500,468 named bytes include an unused 576 MiB stage and are not a full
pipeline memory forecast. Correlated adversary verdict: ROBUST for this bounded
descriptive localization; causal attribution and extrapolation to 4.6 seconds
remain unverified.

Reproduction: `make pixelumm_drift_probe` builds the native diagnostic;
keep `BUILD_DIR` and `TMPDIR` under the external evidence root for builds.
`make spec_pixelumm_drift_probe spec_pixelumm_block_profile` passes 36 CPU
examples, and `python3 -B -m unittest spec/pixelumm_drift_analysis_test.py`
passes eight accounting/guard falsifier tests. The existing GC observer and
reuse suite passes its real-collection qualification and 12 CPU examples.
Use the guarded argv in the external execution receipt, with fresh report and
guard paths and fresh memory admission, rather than running the binary bare.
Run the CPU analyzer with its report positional argument, `--guard LOG` and
`--output FRESH_JSON`. Source/binary snapshots, raw report, guard, validated
analysis and a compact independently recomputed index are under
`/opt/mldata_ssd/cogni-ml/hy-nomas-pixelumm-drift-20261028`; the index is
`results/summary.json`. The raw report SHA256 is
`e9758ab1c917493afad4e7960147f7f86fb6a2adc46a42b6c8e75ff13605ecc4`.
The compact index SHA256 is
`b7f254bf9b29879963309a68be1241f519088f8a88683e11f363db5d59a46d88`;
it pins the final analyzer, execution receipt, binary/source lineage and raw
guard/report hashes. Earlier exploratory analyzer summaries are not the
selected evidence.

Next falsifier for the severe drift: a longer matched serial exposure that
actually reproduces the hump, with direct device frequency/power/utilization
telemetry and a check of profiler intrusion. Separate system competition and
device state from per-Plan resource behavior before changing GC or command
scheduling. This diagnostic alone does not select a production optimization.

### Bounded host-allocation and GC experiment (2026-10-28)

Compare normal Boehm GC with GC disabled only from request admission through
the existing terminal GPU wait. Use the authenticated two-window/two-block
J5760 BF16 diagnostic above, with both arms executing the ordinary baseline
graphs in window-major order. A standalone driver is required: GC disabling
is process-wide and is not admitted in Studio or concurrent inference here.
Restore GC in an exception-safe scope, before metrics, teardown and reporting.
Keep those costs and following-request collection debt visible in end-to-end
time; a shorter encode interval alone cannot qualify a speedup.

This is CAUTION diagnostic work. The strongest false positive is shifting a
collection outside the disabled interval or mistaking Metal/MPS backpressure
for GC. Record allocation bytes, collection events and rendezvous-inclusive
and fully stopped intervals at existing CPU phase boundaries, without new GPU
waits or tensor readbacks. Qualify the allocation-free native observer with
real collections in a separate CPU-only positive control. Refuse an occupied
collector callback, unsupported GC or an already disabled collector. No forced
collections or artificial allocation pressure belong in measured GPU requests.
Zero observed collections after warmup cannot establish a general GC verdict.

DoD: release-build `make pixelumm_gc_probe`, run its CPU lifecycle/observer
qualification, then run two correctness trials, four warmups and 64 balanced
timed trials, each followed by one matched normal-GC tail (140 requests total),
under `examples/support/wan_memory_guard.py` with a 16 GiB
footprint cap, 10% cache-aware reserve, 64 MiB swap-growth stop and 600-second
timeout. Expected signal: both arms pass all 24 pinned Torch 2.6.0 captures at
unchanged atol=rtol=2e-5; the NaN positive control fires; each request retains
one terminal wait, one packed metrics read and zero tracked encode-time named
MetalBuffer allocations/preterminal reads. Numerical checking stays on GPU.
Report setup and diagnostics separately from the bounded inference body.

Rollback removes the new diagnostic and restores the previous diagnostic's
default request-count gate. Production inference, precision, synchronization
and GC policy are unchanged.

The guarded native run completed all 140 requests in 474.119 seconds, including
admission/setup/reporting. The observed request loop was 454.663 seconds.
Both correctness primaries matched all 24 saved Torch 2.6.0 captures bitwise;
the unchanged tolerance gate and NaN positive control passed. All requests
retained their single terminal wait and 464-byte packed metrics read, with zero
tracked encode-time named-buffer allocations or preterminal reads. No full
generation or production policy change ran.

The 64 timed primaries used ABBA/BAAB repeated eight times, 32 per arm. Results
below are descriptive because both arms experienced large common drift:

| Metric | Ordinary GC | Scoped `GC.disable` |
| --- | ---: | ---: |
| Primary wall median | 3.516415 s | 3.560862 s |
| Primary plus ordinary-GC tail median | 7.019055 s | 7.102461 s |
| Boehm allocation bytes per primary, median | 22,056 B | 24,624 B |
| Collections in timed primaries | 0 | 0 |
| Collections including matched tails | 0 | 1 |

The median paired `(normal - disabled) / normal` difference was -0.9701% for
primaries and -0.4741% including tails; disabled won 13/32 pairs. This is no
established speedup, slowdown or equivalence. Keep ordinary GC enabled.

The entire loop allocated 3,301,280 Boehm-managed bytes. Its heap remained
7,159,808 bytes at both endpoints. Exactly one natural collection occurred in
trial 56's ordinary-GC tail after a disabled primary, during encoding. Its
collector envelope was 1.104083 ms, rendezvous-inclusive pause 0.888000 ms and
fully stopped interval 0.840875 ms. These windows overlap; do not add them.
This one event cannot characterize a pause distribution or establish a causal
transfer of GC debt. The ordinary timed arm had no collections to contrast.
Allocation volume covers this diagnostic's process-wide Boehm activity, not
allocation counts, production inference or Objective-C/Metal allocation bytes.

Primary GPU command-chain spans also drifted: ordinary-GC early/late medians
1.950924 -> 4.650484 s, disabled 2.051612 -> 4.618436 s. Body encoding and
terminal waits grew together. The reported GPU span includes queue/gap effects;
it is not pure kernel active time, and encoding includes Metal/MPS backpressure.
The observed GC event cannot explain this seconds-scale drift. Its cause is
still unlocalized; further bounded GPU/MPS/queue-state measurement is a better
next falsifier than removing Crystal's collector or rewriting the runtime.

Sampled peak process-group footprint was 8,180,682,040 bytes (7.62 GiB), minimum
cache-aware headroom 25,697,992,704 bytes (23.93 GiB), with zero positive swap
growth and no guard trigger across 879 samples. The named inventory stayed
7,381,500,468 bytes, including an unused 576 MiB F32 staging set inherited from
the diagnostic; it is not an optimized production memory forecast. Limits are
sampled, and file-cache headroom remains a reclaimability estimate.

CPU qualification passed real collection/suppression/callback-ownership tests,
12 Crystal examples and 7 Python analyzer tests. An uncontended 10,001-point
CPU sampling qualification allocated zero bytes, observed no collection and
averaged 0.111 microseconds/point. It does not measure contended GC locks;
instrumentation costs were not subtracted. Release build and format/diff checks
passed. The correlated adversary verdict is ROBUST for these bounded numerical
and measurement claims, not a general GC-policy or production performance claim.

Evidence root: `/opt/mldata_ssd/cogni-ml/hy-nomas-pixelumm-gc-20261028`.
`results/summary.json` SHA256
`3dff8dbcc9f413ba20206bcb111b08a013e1ceb639839e485c0e0834823790f5`
pins the raw report, guard, execution receipt, analyzer v2 and qualified sources.
Measured executable sources/binary are unchanged; the subsequent Makefile edit
only adds transitive build prerequisites. Request phase deltas are recomputed
from raw points; trial-inclusive values have lower/outer bounds rather than
exact endpoint reconstruction because those two points were not serialized.
The single ordinary tail exposes immediate carryover only, not all future GC
debt. Refresh on measured source, fixture, weights, runtime, geometry, precision,
schedule, backend, device or guard changes, or artifact loss; admit memory afresh
before any GPU replay.
