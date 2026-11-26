# PixelUMM independent-window attention batching

## Active frontier

All 36 full-J blocks plus the global head pass both saved-input numerical gates,
using one shared B2 panel bank and an explicit common-Workspace/serial-slot guard.
The phase/continuation diagnostic below also passes 516 retained comparisons
bitwise against the saved upstream controls. B2 reduces command count and active
host-thread CPU time, but its full-forward contrast changes sign with order.
Much of the apparent host attention saving moves into later API waits; GPU
intervals belonging to the unchanged FFN remain substantial. Keep B2 opt-in:
neither a stable speedup nor the physical cause of the drift is qualified.
No full video generation is admitted.

The opt-in bounded diagnostic localizes the missing full-forward benefit without
changing the attention math. `--profile-phases --timed` enables the existing
block phase markers, API timing and continuation trace on four poisoned gates,
the middle ABBA requests and the candidate tail. Four observer-off requests
bracket that ABBA (baseline/B2 before it, B2/baseline after it); two observer-off
warmups precede both. `--reverse` flips the mode order, not observer placement.
The fifteen-request schedule retains the existing 43-stage saved-upstream gates,
one terminal wait and zero encode-time activation reads/allocations. Profile
metadata is extracted after the wait, outside request timing. Its CPU work can
affect the next request, so profiled phase contrasts are diagnostic and the
observer-off arms do not by themselves establish an unperturbed speedup.

The competing explanations are a more expensive paired attention phase,
non-attention drift, or altered Metal/MPS blocking. The discriminator is the
per-block phase wall/thread-CPU split, exclusive Graph versus continuation API
time, and root intervals associated with each phase. Roots crossing a boundary
keep all candidate phases; their duration sums are not active shader time or
additive phase costs. A root count reduction alone cannot qualify acceleration.

This diagnostic-only instrumentation is a CAUTION change. The main failure mode
is accepting the baseline Graph-call contract for B2, or attributing overlapping
API/root spans twice. An explicit paired-window topology must require positive
even tile counts and validate 3,240 attention / 3,785 total Graph calls for full36
B2, versus 6,480 / 7,025 baseline; continuations must match roots minus one.
CPU negative tests must reject mismatched counts and odd tiles. DoD is those
tests, the full36 schedule/CLI tests, release build and guarded normal/reverse
cohorts with unchanged numerical gates and a phase/trace consistency analysis.
The existing 28.5 GiB process cap, 10% cache-aware reserve, 64 MiB swap-growth cap
and bounded timeout apply. Missing trace events, changed topology, early reads,
additional waits or any guard failure stop the diagnostic. Rollback omits the
new flag; all existing execution defaults remain unchanged.

The full36 experiment first compares four independently poisoned connected
requests: both saved inputs in both baseline/B2 modes. It checks all 43 retained
captures (36 attention outputs, block34/35 outputs and five head stages), with
the original tolerances and a direct GPU block35-to-head edge. Only passing
gates admit two warmups followed by same-input ABBA then BAAB timing, without
CPU tensor diagnostics between timed requests. The last timed output is checked
before capture reuse. Both orders then run the same untimed candidate tail and
check its 43 captures, testing reuse after the timed cohort. A reverse process
flips the preceding mode assignments and warmup order; interpret order-sensitive
or drifted results separately.

The baseline's unique named storage is 24,108,145,128 bytes. Admission includes
one shared candidate bank plus 36 per-plan metric buffers: 931,136,064 bytes,
for an aggregate 25,039,281,192 bytes. This is an explicit allocation inventory,
not process footprint or a measured incremental RSS delta. The wrapper's live
named-buffer counter does not include native B2 objects. Opaque MPS resources
remain covered by the process-tree
guard: 28.5 GiB cap, 10% cache-aware reserve, 64 MiB swap-growth limit, and a
bounded timeout. Any numerical/guard/lifetime failure stops timing. Rollback
omits the new diagnostic executable; inference defaults remain unchanged.

This diagnostic tests whether two independent local attention windows can share
one MPSGraph batch without changing either window's Q/K extents, ordered key
list, scale, softmax axis or full-K reduction. It changes no production default.
No diffusion or full video generation is admitted. The existing BF16 weights,
F32 FFN, shared block workspace and direct decoder-head edge remain the control.

The supported grid is T6/H44/W80, J=21,120, with 32 heads of dimension 128.
There are 40 windows of Q384/K3456, 10 of Q384/K2880 and 10 of Q192/K2880.
Pair windows only within exactly equal Q/K geometry. Equal geometry does not
imply equal context: gather each lane using that window's own ordered key table.
The batch extent changes from 32 to 64 heads; query and reduction extents do not.

The previous query-row concatenation route changed Q and failed a downstream
BF16 gate. It is rejected in this experiment. MPS may nevertheless select a
different reduction when batch size changes; the following gates decide this.

## Admitted transitions and falsifiers

1. CPU layout checks cover every query once, preserve each ordered context and
   separate boundary geometries. A seeded cross-lane context mistake must fail.
2. Prepare one shared B2 panel bank before encoding. Report every allocation,
   including metrics. Retaining baseline panels means the candidate bank adds
   888 MiB, not the 444 MiB net capacity difference. Graph-owned temporary memory
   is opaque and remains covered by the external process-tree memory guard.
3. Using real post-projection/RoPE Q/K/V already on GPU, compare raw B2 QK to
   baseline per-window QK. Then compare PV with the same baseline probabilities
   supplied to both paths. A seeded comparator fault must be detected.
4. Only after these gates pass, compare the complete block's attention and output
   to authenticated saved upstream controls, with poisoned baseline transients
   and all six candidate panels (idle-only test setup, excluded from timing).
   Tolerances remain `atol=rtol=2e-5`; nonfinite values fail. The first isolated
   fixture has the same full-J geometry, weights and RoPE tables as the full36
   control but a different real input. Passing it does not qualify all 36 blocks.
5. Only passing connected paths may enter API-profiler-off full-block timing. Use
   ABBA then BAAB in the same resident process, without CPU tensor diagnostics
   between timed requests, and recheck the last candidate's retained captures.
   Do not infer a speedup from a drifted control. The full36 plus head gate on
   both saved inputs is necessary but insufficient for promotion: full-forward
   timing and memory must also qualify.

All checks use the existing terminal wait once per request. Metrics and saved
captures are read only after it. No intermediate CPU activation copies, new
GPU waits, per-window synchronization or model generation are allowed. Selection
and resource release require all affected plans to be idle. The pair bank is
shared only across serialized plans with an explicit owner lifetime.

The cache-aware guard retains 10% of physical RAM and stops on excess process
footprint, excess swap growth or missing telemetry. A failed guard, incomplete
capture set, changed fixture lineage or numerical failure stops progression.
Rollback omits the diagnostic executable; production attention remains intact.

## Evidence and completion boundary

DoD is the CPU geometry/lifetime contract suite, release build, positive-control
rejection, unchanged raw-QK/PV and connected saved-upstream block gates, then
guarded paired full-block timing if the numerical gates permit it. Full36 adds
the common-Workspace/serial-slot negative cases, two-input 43-stage gates,
connected head, aggregate admission and normal/reverse timing below. A correct
diagnostic that refutes B2 is a useful result, not permission to widen tolerances.

The oracle is the existing full-J fixture set at upstream revision
`911cea6458a89c2663513ff7099561ec190dd491`. This is a comparison with saved
upstream tensors, not a freshly regenerated upstream neural forward. Raw QK/PV
gates isolate the changed MPS operation against the current native control.
Source, fixture, cache, binary and guard-log digests must accompany any result.
Changing those inputs or the GPU/backend invalidates the numerical certificate;
timing additionally decays with host load, pacing and resource-lifetime changes.

## Full36 phase and continuation diagnosis, 2026-10-29

Two fresh processes complete the fifteen-request phase schedule above. Each
profiles four poisoned gates, the middle four measured requests and the final
B2 tail. Four observer-off measured requests bracket that middle ABBA; reversing
the process flips modes while preserving observer placement. Each timing cell
below has only two samples per path. Post-wait profile processing can affect
the following request, so these are diagnostic contrasts, not an unperturbed
production benchmark.

| Observer-off full36 + head, encode through terminal wait | Baseline | B2 | B2 time change |
| --- | ---: | ---: | ---: |
| Normal process | 54.506625 s | 55.870329 s | +2.502% |
| Reverse process | 62.911368 s | 58.049763 s | -7.728% |

Segments remain 4,579/2,779 and total Graph calls are 7,025/3,785
(baseline/B2). Observer-off active encode-thread CPU is only
0.352493/0.255823 s normally and 0.495779/0.355494 s in reverse. The saved
0.097/0.140 s is small compared with the 55 to 63 s request means. The profiled
request means are 56.849202/59.235752 s normally and 83.452031/71.558872 s
in reverse. Neither those contrasts nor a pooled mean establish a stable effect.

Host phase markers show a repeated relocation of waiting: attention is
53.564675/45.049650 s normally, while projection plus residual is
0.351656/9.206310 s. Reverse gives 79.128385/54.575204 s for attention and
0.434962/11.180452 s for projection plus residual. These host intervals include
blocking inside Metal/MPS and are not GPU phase execution time.

The API trace localizes that relocation to `commitAndContinue`. Mean exclusive
continuation wall time per profiled request is:

| API phase | Normal baseline | Normal B2 | Reverse baseline | Reverse B2 |
| --- | ---: | ---: | ---: | ---: |
| Local attention | 53.314359 s | 44.870862 s | 78.702951 s | 54.292348 s |
| Projection/residual | 0.345937 s | 9.191624 s | 0.426300 s | 11.159442 s |
| FFN | 1.146201 s | 2.471207 s | 1.474651 s | 2.812014 s |
| All phases | 55.391520 s | 57.683215 s | 81.333127 s | 69.544837 s |

The corresponding all-phase continuation thread CPU totals are only
0.059865/0.059071 s and 0.089929/0.073455 s. Exclusive Graph wall time,
which excludes nested continuations, is 0.251993/0.214322 s and
0.430529/0.338592 s. Adding inclusive Graph time to continuation time would
double-count the principal waiting span.

Across the eight profiled measured requests, all 2,690 continuation calls over
10 ms intersect intervals of earlier submitted GPU roots. Their aggregate wall
is 414.105985 s, thread CPU 0.119216 s, and prior-root interval overlap union
414.086355 s. Each intersection excludes the current submission; the existing
cross-clock bound is 2 microseconds. Because the GPU-root envelope is almost
continuously covered, this observation alone cannot identify a blocking root,
semaphore, queue capacity or driver mechanism.

GPU timestamps tell a different story from the host attention markers. The
following duration sums include only roots assigned unambiguously to the named
phase. Shared boundary roots are excluded; intervals can overlap. These are
command-buffer interval measurements, not active shader time or additive phase
costs.

| Unambiguous GPU-root interval duration sum | Normal baseline | Normal B2 | Reverse baseline | Reverse B2 |
| --- | ---: | ---: | ---: | ---: |
| Attention | 13.317544 s | 12.483761 s | 17.093919 s | 14.128867 s |
| FFN gate/up | 18.129663 s | 19.470187 s | 27.556886 s | 23.800254 s |
| FFN down | 9.477239 s | 10.199575 s | 15.278755 s | 13.062383 s |

Attention's candidate-root sums, including ambiguous boundary roots, are
19.066814/18.746846 s and 25.504964/21.676393 s. They cannot be added to
other candidate-phase sums. Whole-chain root interval unions are
56.842806/59.230821 s and 83.435886/71.550872 s, within 4 ms of their
envelopes. This rules out large uncovered command-buffer gaps in these traces;
it does not measure shader occupancy or identify the cause of a slow interval.
The unchanged FFN intervals also drift substantially between processes. Fewer
attention submissions do not remove those GPU operations, and the host saving
must not be confused with equivalent removed GPU work.

All 30 requests retain one terminal wait, zero tracked encode allocations,
zero pre-wait activation reads and zero stage-metric reads. All 516 retained
captures are finite and bitwise equal to authenticated saved upstream tensors
at unchanged `atol=rtol=2e-5`; no fresh upstream forward was run. Both guards
finish with exit 0: shared-resident sampled peaks 27.142947/27.148151 GiB,
minimum cache-aware headroom 11.801331/11.698792 GiB above the 6.4 GiB floor,
and zero swap growth. Polling does not bound between-sample spikes or isolate
B2's incremental footprint. Named live buffers return to zero after release;
the higher residual Metal allocation counter remains unexplained.

Release build, 49 focused CPU examples, eight offline analyzer falsifiers and
the pre-observer CLI rejection pass. The first offline analysis exposed a field
name mismatch between two summary helpers; a composed-path regression test
covers the repair. It changed only offline analysis after both cohorts. Frozen
runtime sources, binary, fixtures and raw reports remain the measured lineage.
The post-analysis receipt verifies that the only changed member of the measured
417-file manifest is the offline analyzer, and snapshots its repaired version
and the regression test separately. It does not assert whole-checkout invariance.

Observed CPU DoD commands (49 Crystal examples and eight Python tests, no failures):

```sh
crystal spec -Dcpu_only spec/pixelumm_window_batch_full_probe_spec.cr \
  spec/pixelumm_decoder_profile_spec.cr spec/pixelumm_block_profile_spec.cr \
  spec/wan_mps_api_profile_spec.cr spec/wan_mps_continuation_trace_spec.cr \
  spec/wan_mps_command_phase_profile_spec.cr \
  spec/video_mps_api_profile_clock_bridge_spec.cr
python3 -B spec/pixelumm_b2_phases_analysis_test.py
```

The release target is `make pixelumm_window_batch_full_probe`, with build,
temporary and Crystal cache paths directed to the external SSD. Each cohort's
launch receipt records the complete authenticated fixture argv, the existing
memory limits and `--timed --profile-phases`; reverse also adds `--reverse`.
The binary negative check rejects `--profile-phases` without `--timed` before
creating a report or observer. Each GPU cohort exits 0 after all 15 requests.

Evidence directory:
`/opt/mldata_ssd/cogni-ml/hy-nomas-pixelumm-b2-phases-20261029`.
`normal/` and `reverse/` retain frozen source snapshots, binaries, launch and
completion receipts, raw reports and guard logs. `provenance-analysis.json`
validates the original measured lineage and sampled memory guards;
`phase-analysis.json` validates and localizes the saved reports with the repaired
offline analyzer. `post-analysis-receipt.json` binds that repair to these inputs.
The measured parent is `fb3101dbcf54ea1a62b196cc01a451614a78d0b7`; saved upstream
lineage remains `911cea6458a89c2663513ff7099561ec190dd491`.

| Evidence | SHA-256 |
| --- | --- |
| Measured 417-file source manifest | `0410d01f90f81048b187dedd991db924fba1c2aa9ae3bcebfe65517e8aa5a699` |
| Measured executable | `2fa171511244126ddc9147112a16a4187878ab044a80b3be9c5c36aae29d3e31` |
| Normal report | `94d4ff3942c8a67e2f697f062206e905150f665eb7c67f188390a2aa2d148759` |
| Reverse report | `bb6be0ce6345239805a911d9ca36da24da798efe83adab14f1805a77950d7142` |
| Provenance and sampled-guard analysis | `c75654dba8306db5ea29449393b56eede51093cfba21c797fef57912800b90a0` |
| Phase/continuation analysis | `7cf5cee581b0fdc4c590bc9272eaeecafee8a947cc39ae21c04fb3c3c925aad4` |
| Post-analysis lineage receipt | `0b77416790ece9b030ba11699d9854857e673f08e6e4cc53f839c9d158ceedd3` |

Correlated Mini-Quad verdict: ROBUST for the command reduction, captured-output
parity, bounded sampled memory and lack of a qualified stable speedup.
The stronger claim that B2 cannot accelerate this forward is VULNERABLE:
both reverse observer-off adjacent pairs favor B2, so a modest benefit remains
plausible. Keep B2 opt-in and separate host API waiting from GPU execution.

The next speed discriminator is a longer observer-off, quiet, order-balanced
full36 crossover with matched pacing. The next mechanism discriminator is a
bounded Metal System Trace or per-encoder GPU timing that can separate active
attention/FFN execution from queued command-buffer intervals. Retain the one
terminal wait and all numerical/memory gates. Repeating command-count changes
or blind FFN tiling cannot resolve the physical cause of this drift. Refresh
this evidence on scoped source, fixture, precision, geometry, backend/device,
toolchain, guard, schedule, diagnostic policy or resource-lifetime changes;
timings also decay with host load and pacing. No full generation ran.

## Full36 plus head result before phase profiling, 2026-10-29

Two fresh processes use the same frozen executable, 416-file runtime/build
source snapshot, BF16 weights, F32 FFN, full-J geometry and saved controls.
Each completes 15 requests: four independently poisoned both-input/both-mode
gates, two warmups, eight balanced timed requests and an untimed B2 tail.
Normal timing is ABBA then BAAB (A=baseline, B=B2); reverse timing is BAAB then
ABBA, with reversed warmup order. The last timed request and each unpoisoned B2
tail receive all 43 comparisons. Gates, warmups and tails are excluded from the
means below; each process contributes four timed samples per path.

| Full 36 blocks + head, encode through terminal wait | Baseline | B2 | B2 time change |
| --- | ---: | ---: | ---: |
| Normal process | 68.694039 s | 71.226950 s | +3.687% |
| Reverse process | 64.175671 s | 64.888191 s | +1.110% |

Command segments fall from 4,579 to 2,779 (-39.310%). Active encode-thread CPU
time falls from 0.676507 to 0.452379 s normally and 0.482651 to 0.355825 s in
reverse. This clock measures thread CPU consumption; encode wall time includes
blocking and scheduling. Mean encode wall time is 67.233813/69.593413 s normally
and 62.780435/63.496479 s in reverse (baseline/B2). Mean terminal wait is
1.460220/1.633530 s and 1.395228/1.391706 s respectively. These phases overlap
GPU work; their partition does not establish the physical cause of waiting.

Timed requests span 49.474 to 87.621 s across the cohorts. The four balanced
cell contrasts (B2 minus baseline) are +2.523235, +2.542585, +5.780940 and
-4.355899 s. Thus the pooled means, 66.434855/68.057570 s (+2.443% B2 time),
are descriptive only. Both process means point against improvement, but the
drift and sign-changing cell contrasts do not identify a stable causal effect.
The one-block timing result below does not transfer to this full forward.

All 30 requests pass. Each compared request checks 36 attention outputs,
block34/35 outputs and five global-head stages: 258 captures per process,
516 total, finite and bitwise equal (`max_abs=0`, `different_words=0`), at
unchanged `atol=rtol=2e-5`. The head directly aliases block35's GPU output.
Every request reports one terminal wait, zero tracked encode allocations,
zero pre-wait activation reads and zero stage-metric reads. All command roots
complete without errors. Idle-only selection, poison, comparison and release
guards reject in-flight use. Full36 does not separately read every raw QK/PV
intermediate; those isolated controls remain the historical one-block evidence.

Both cache-aware guards finish with status 0, no stop events and zero sampled
swap growth. Sampled process-group footprint peaks are 27.077502 and
27.078784 GiB, under the 28.5 GiB cap. Minimum available headroom is 11.363052
and 10.385696 GiB, above the 6.4 GiB runtime floor (10% of 64 GiB).
File-backed cache is counted once by the guard. Both paths remain resident;
these peaks do not measure an isolated incremental B2 footprint. Sampling
does not exclude between-poll spikes.

Separate phase counters peak at 28,729,363,656 bytes of physical footprint,
25,372,966,912 bytes RSS and 27,853,766,656 bytes Metal allocation. They overlap
and must not be summed. Candidate setup increases the overall Metal allocation
by exactly 888 MiB, matching its shared panels; the 576 bytes of per-plan metrics
are included in admission but are not independently attributed by the counters.
After owner release, named live bytes return to zero and physical footprint is
below 0.56 GiB, while the Metal allocation counter remains higher and differs
between processes. The cause of that divergence is unresolved; this result
proves neither a backing-store leak nor complete device deallocation.

Validation passes: release build, 88 CPU examples, four pre-observer binary CLI
rejections, both guarded GPU cohorts and the offline receipt/source/schedule/
capture/lifetime analyzer. Correlated Mini-Quad review rates bounded numerical
correctness and sampled guard containment robust; durable acceleration and an
isolated incremental B2 footprint remain unqualified. Production defaults are
unchanged. Rollback omits the diagnostic target.

Observed CPU DoD commands (59 + 29 examples, zero failures):

```sh
crystal spec -Dcpu_only spec/pixelumm_decoder_probe_spec.cr \
  spec/pixelumm_two_block_probe_spec.cr spec/pixelumm_memory_probe_spec.cr \
  spec/pixelumm_head_probe_spec.cr
crystal spec -Dcpu_only spec/pixelumm_window_batch_full_probe_spec.cr \
  spec/pixelumm_window_batch_probe_spec.cr \
  spec/pixelumm_decoder_block_admission_spec.cr spec/pixelumm_block_probe_spec.cr
```

Release target: `make pixelumm_window_batch_full_probe`, with `BUILD_DIR`,
`TMPDIR` and `CRYSTAL_CACHE_DIR` directed to the external SSD. Each cohort's
`launch-receipt.json` records the exact authenticated fixture argv and guard
invocation: 29,184 MiB footprint cap, 64 MiB swap-growth cap, 2,400 s timeout,
`--memory-accounting file-cache --runtime-floor-pct 10`, and a named-byte budget
of 25,039,281,192. `completion-receipt.json` pins unchanged source, executable,
fixtures, report and guard hashes. Replay requires fresh evidence paths and
fresh memory admission; do not overwrite these cohorts.

Evidence directory:
`/opt/mldata_ssd/cogni-ml/hy-nomas-pixelumm-b2-full36-20261029`.
`normal/` and `reverse/` contain the raw reports, guards, receipts, binaries and
source snapshots. `analyze.py` reproduces `analysis.json` from both cohorts.
Its parser was corrected for lexical `/opt` paths, the reversed mode schedule
and 1-based memory-phase sequence; no measured source or raw report changed.
The measured parent is `6ed52b78837bb97d8393e4179ee21583fde090a5`; numerical
upstream lineage remains `911cea6458a89c2663513ff7099561ec190dd491`.

| Evidence | SHA-256 |
| --- | --- |
| 416-file source manifest | `acf3d17968225670ff039f386f0877e383a368377f5919f0f440947da600c029` |
| Measured executable | `7740281a4bd144150882b7d593554247741b065ac2323970f8b879fbb1e934e6` |
| Normal report | `ccce65f992080389e89331b00f8fbeaddd43fbcdd0c4a4e5ab34293879200fd0` |
| Reverse report | `a6993ef7cdeaf6fb1e6470ea5ec6f4b2be9286849c70756aa5d5811a7ca9d96e` |
| Validated analysis | `f914e3ccb77b08186027ad8735c65410c64252c135bbcfdad58964a27606df80` |

Decision: retain B2 only as a diagnostic. Reducing command count and host CPU
consumption is insufficient here; use full-forward wall time as the promotion
gate. Numerical evidence decays on source, input, weight, geometry, precision,
backend, device or toolchain changes. Timing additionally decays on host load,
pacing, schedule, diagnostics and resource-lifetime changes. No full video or
diffusion generation was run or qualified.

## Historical one-block result, 2026-10-29

Two fresh processes retained the same release binary, saved block0 fixture,
input, weights and graphs. Each completed five qualification requests, two
warmups and eight timed requests (ABBA then BAAB). The warmup order was baseline
then candidate in both processes. These are conditional descriptive contrasts,
not a population confidence interval or an end-to-end pipeline benchmark.

| Mean across 16 timed requests | Baseline | B2 | Change |
| --- | ---: | ---: | ---: |
| Full block encode through terminal wait | 1.208287 s | 1.182598 s | -2.126% |
| CPU encode interval | 357.526 ms | 227.216 ms | -36.448% |
| Command segments | 128 | 78 | -39.063% |

The four ABBA/BAAB cell contrasts are -2.763%, -1.478%, -2.249% and -2.012%.
Terminal wait increases from 850.760 to 955.381 ms as CPU encode finishes sooner.
Those intervals overlap GPU execution; subtracting CPU encode time is not an
end-to-end speedup estimate. Command-buffer interval unions are telemetry, not
active-shader or physical queue-stall measurements. B2 also packs two lanes'
K/V in one gather, so the contrast covers batching plus that gather change.

In each process, raw QK compares all 2,229,534,720 values and common-P PV all
86,507,520 values: zero differing finite words, zero tolerance failures and zero
nonfinite pairs. The comparator-only positive control records exactly one
injected failure among 84,934,656 values. All eight retained attention/output
comparisons per process match the saved upstream bitwise, including the final
timed candidate. This does not prove every F32 softmax intermediate identical.
Every request has one terminal wait, no tracked encode allocations or early
activation reads, and the declared exact post-wait capture/metric read counts.

The six extra candidate panels occupy 931,135,488 bytes (888 MiB), plus 16 bytes
of metrics. Both paths' buffer objects remain live: total unique named buffers
are 4,154,449,396 bytes. This is an allocation inventory, not a measured
incremental RSS delta or a future replacement-only workspace budget.
Both guards pass all 110 samples, with a 10 GiB process cap, 10%
cache-aware reserve and 64 MiB swap-growth limit. Sampled footprint peaks are
7.337 and 7.244 GiB; minimum available headroom is 36.027 GiB; swap growth is
zero. Sampling does not rule out between-sample spikes.

The first setup attempt is retained separately and excluded from these numbers.
It stopped before chain creation on a recursive Workspace mutex in the poison
helper. Snapshotting retained buffer membership before reacquiring the guarded
configuration lock removes the nested acquisition. Both completed runs exercise
the corrected poison path; no synchronization was added.

Validation: release build, 24 CPU examples, production non-diagnostic Objective-C++
syntax check and three pre-observer CLI rejections. Same-lineage Mini-Quad source
and evidence reviews support only the bounded one-block qualification. The
common-Workspace identity guard required at this stage is exercised by the
full36 experiment above.

Observed CPU DoD command (24 examples, zero failures):

```sh
crystal spec -Dcpu_only spec/pixelumm_window_batch_probe_spec.cr \
  spec/pixelumm_decoder_block_admission_spec.cr spec/pixelumm_block_probe_spec.cr
```

The release target is `make pixelumm_window_batch_probe`; direct build and cache
paths to the external SSD with `BUILD_DIR`, `TMPDIR` and `CRYSTAL_CACHE_DIR`.
Each launch receipt records the complete guarded GPU DoD argv and fixture pin.
Expected signal is exit 0, all 15 requests passing, the exact stage extents above,
bitwise attention/output comparisons and a successful guard completion. Replay
only with fresh report/guard/receipt paths, never overwrite an existing cohort.

Evidence directory:
`/opt/mldata_ssd/cogni-ml/hy-nomas-pixelumm-window-batch-20261029`.
`results/` retains the failed setup; `results-v2/` and `results-v3/` contain
completed raw reports, guards, source manifests and launch/completion receipts.
`source-v2/` and `source-v3/` snapshot all 414 pinned runtime/build source files.
`analyze.py` recomputes `results-analysis.json` from both completed cohorts.
The fixture metadata SHA is
`ecec716b1204cf7aae12827164ab9af31122c073f55749505cab934bd64575ca`;
the binary SHA is
`01bf186d7c8900fbd3534d31280b41714e19723fdb08d485bc81db1a28820581`.
Post-run cleanup removes only two empty EOF lines from the native diagnostic.
`postbuild-receipt.json` binds that source delta to a rebuilt executable with
exactly the same SHA as both measured binaries; no numerical replay is needed
for this byte-identical build.

Decision at this stage: keep B2 diagnostic-only. The observed command-count
reduction yields a small full-block gain and costs additional memory in this
A/B layout. Before
production use, require full36 plus head on both saved inputs, a qualified
workspace/lifetime contract and a bounded aggregate memory/timing result.
Prioritize GPU execution costs over command-count targets when choosing the
next optimization; a shorter encode interval alone is insufficient.
