# PixelUMM independent-window attention batching

## Active frontier

All 36 full-J blocks plus the global head pass both saved-input numerical gates
in two bounded cohorts, using one shared B2 panel bank and an explicit
common-Workspace/serial-slot guard. All 516 retained stage comparisons are
bitwise equal to the saved upstream controls. B2 reduces command count and
active host-thread CPU time, but neither cohort shows a full-forward speedup.
The diagnostic remains opt-in; production promotion is unqualified. No full
video generation is admitted.

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

## Full36 plus head result, 2026-10-29

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
