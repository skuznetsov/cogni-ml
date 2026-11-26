# PixelUMM CPU diagnostic placement

## Shared scope and original discriminator

This bounded experiment tests whether post-request CPU comparison changes the
following native PixelUMM request. It does not change inference mathematics,
production defaults, generation policy or the Metal/MPS backend. No diffusion,
Studio or full video generation is run.

The diagnostic uses all 36 blocks and the global decoder head at J=21,120,
grid 6x44x80, with the same authenticated input A, BF16 weights, F32 FFN graphs,
two output banks, shared block workspace and direct block35-to-head buffer edge
in every request. API profiling is off. Comparison scratch is prepared
separately from candidate attention graphs; preparing it does not enable those
graphs. Unique named GPU storage remains 24,108,145,128 bytes. The two reusable
CPU comparison arrays total 346,030,080 bytes.

Each process runs six requests: warm, condition 0, probe 0, condition 1, probe 1,
tail. The normal order compares condition 0 eagerly and condition 1 after the
whole sequence. The reverse process swaps these assignments. Warm, probes and
tail use the existing full comparison. The primary contrast is **the following
probe's encoding-through-terminal-wait time**, paired by condition in each
process. Compare both orders separately; do not pool a drifted control into a
speedup claim. No direction or minimum effect is assumed in advance.

All inference requests have one existing terminal wait, zero tracked named
MetalBuffer allocations during encoding and zero preterminal activation reads.
Each next pre-recorder milestone targets the preceding terminal return plus
50 seconds. The existing deadline check rejects diagnostic overrun before the
next request is submitted; the 250 ms allowance covers scheduler lateness after
sleep. This is a matched interval, not 50 seconds of idle sleep. Recorder setup
still follows the timestamp, so it is not the exact first GPU encode call.

## Preserve captures before moving their comparison

Live outputs cannot be compared after the sequence: subsequent requests reuse
their buffers. Every condition therefore reads the same 43 retained BF16
captures once after the existing wait and streams them into an exclusive disk
file before buffer reuse. A snapshot contains 7,353,139,200 bytes. The two
snapshots per process share the existing comparison arrays and remain on the
external scratch SSD. Logical file extent does not reserve APFS disk space;
write/flush errors abort the experiment.

Both comparison policies read actual bytes from those saved files through the
same CPU comparator and scratch. Eager comparison completes before its probe;
deferred comparison cannot start until all six requests have completed. A
deferred request is explicitly pending until then. A final passing report
requires all 258 stage comparisons, finite values and unchanged
`atol=rtol=2e-5`. Snapshot files are preserved for inspection. Reusing their
paths is rejected; the reverse process uses a different directory.

This changes the placement of snapshot rereads, expected-fixture reads,
hashing and CPU numerical comparison together. Common snapshot writes can
perturb both arms. The experiment does not isolate pure comparator cost or
establish the cause of earlier uninstrumented drift.

Persistent GPU plans and buffers are identical, but the completed command
chain is released and post-request GC runs after that request's diagnostic
work. Eager comparison therefore keeps transient command resources alive
longer and collects closer to the next request. These are consequences of the
tested policy, not separately matched resource-lifetime interventions. A pure
CPU-comparison attribution would require releasing the completed chain after
the common snapshot write and aligning GC before either comparison policy.

The oracle is the existing authenticated upstream Torch/MPS fixture set, pinned
to upstream revision `911cea6458a89c2663513ff7099561ec190dd491` and joined at the
exact block35-output/head-input edge. This is a repeat against saved upstream
controls, not a newly regenerated connected upstream neural run. Coverage is
36 retained attention captures, block34/35 outputs and five head stages;
overwritten early intermediates are outside this gate.

## Aligned lifecycle follow-up

The opt-in `--align-cpu-diagnostic-lifecycle` follow-up separates snapshot
writing from eager comparison. In both policies it copies completed-chain
metadata after the sole terminal wait, saves condition captures, releases the
native command chain, and runs explicit GC before comparison or deferred
return. Standard warm/probe/tail comparisons also follow release and GC.
Persistent session-owned GPU plans and buffers remain live. The default and
legacy diagnostic path retain their existing ordering.

The discriminator is whether the following-probe contrast survives this
common release/GC point in both normal and reverse schedules. Its estimand is
the residual eager/deferred diagnostic-placement effect conditional on that
lifecycle; it does not isolate a speedup caused by earlier release or GC.
Cache, thermal, scheduler and background-load effects remain possible. Reject
missing or reordered lifecycle receipts, metadata access after native release,
comparison before GC, capture reuse before saving, any additional wait, changed
upstream tolerance, pacing failure or memory-guard failure. Keep the same
guarded six-request schedule, with no generation. Rollback omits the new flag.

DoD: the integrated CPU specs and release build pass; invalid lifecycle CLI
combinations reject before loading fixtures; both guarded native schedules
complete with matching capture hashes and unchanged saved-upstream numerical
gates; the offline auditor verifies lifecycle order, source lineage and guard
telemetry. Pending measurements must not be reported as acceleration.

The first aligned attempt at the original 50-second interval stopped before
creating the following probe's chain: common snapshot save plus eager
comparison took 51.775841792 seconds. Both completed requests passed their
saved-upstream gates, and cleanup and the memory guard completed. This failed
attempt remains under the `hy-nomas-pixelumm-cpu-lifecycle-20261029` evidence
root and is excluded from timing contrasts. Its
`spacing_abort_before_submission=false` refers only to the later empty-chain
quarantine path; the deadline rejection happened before chain construction.

The revised comparison explicitly selects
`--cpu-diagnostic-pause-seconds 60` in both orders. Only 50 or 60 seconds are
admitted, only in CPU-diagnostic mode; 50 seconds remains the default. The same
deadline rejection and 250 ms scheduler-lateness check remain. The new
`hy-nomas-pixelumm-cpu-lifecycle60-20261029` root pins the revised contract,
source and binary separately. Do not pool the failed 50-second attempt or the
earlier unaligned experiment with this 60-second conditional contrast.

## Aligned lifecycle result (2026-10-29)

Both 60-second schedules completed with the same measured source, binary,
resident resources and ordered release/GC lifecycle. All 516 finite retained
comparisons are bitwise equal to the authenticated saved upstream controls:
zero different BF16 words and max absolute error zero at unchanged
`atol=rtol=2e-5`. The final deferred comparisons completed after all six
request cleanups in each process. This does not extend coverage to overwritten
intermediates or constitute a fresh upstream neural run.

| Process | Preceding diagnostic | Following request seconds | Encode seconds | Terminal wait seconds | Recomputed root union seconds |
| --- | --- | ---: | ---: | ---: | ---: |
| Normal | Eager | 44.535279 | 43.624829 | 0.910224 | 44.521169 |
| Normal | Deferred | 43.250263 | 42.378163 | 0.871873 | 43.217373 |
| Reverse | Deferred | 43.399179 | 42.452734 | 0.946249 | 43.384911 |
| Reverse | Eager | 58.362926 | 57.242693 | 1.118647 | 58.337225 |

The after-eager/after-deferred ratios are 1.029711 and 1.344793: eager-context
requests are 2.9711% and 34.4793% longer in these two descriptive contrasts.
Do not pool them into one speed estimate. The reverse tail also takes
47.391884 s, so the delayed state is not confined to a single point. Aligning
chain release and GC did not remove observed variation. A stable effect size,
pure CPU-comparator causality, an isolated early-release/GC benefit, driver
cause and production acceleration remain unproven. The earlier 50-second
experiments are separate schedules and cannot establish those causal claims
by before/after subtraction.

The raw GPU-root interval union changes by 1.303796 s in normal order and
14.952314 s in reverse. The corresponding terminal-wait differences are only
0.038350/0.172398 s. Thus these samples are not explained by transferring all
delay into the terminal wait. Root envelopes still do not identify active
shader time, a particular lock, frequency or thermal state. Every request has
4,579 completed error-free roots and one existing terminal wait, with 43
post-wait capture reads, zero preterminal activation reads and zero tracked
named allocations during encoding.

Host lifecycle receipts place native release at 0.020500..0.066542 ms and the
explicit GC interval at 0.809333..1.886042 ms per request. These are local
post-wait intervals, not estimates of downstream resource-reclamation effects.
Eager condition diagnostics take 52.273859/45.903829 s; deferred condition
snapshot/cleanup takes 9.398680/9.291419 s and its comparison is moved to the
end, taking another 43.723363/44.387793 s. The diagnostic work is not removed.
Cache, CPU load, shared power/thermal state and background activity can still
differ despite matched terminal-to-pre-recorder gaps.

All 2,667 guard samples pass: sampled peak process-tree footprint is
26.201907 GiB against the 27.5 GiB cap, minimum cache-aware headroom is
13.673111 GiB against the 6.4 GiB reserve, and no sample exceeds its process's
admission swap baseline.
Actual pacing gaps are 60.000188416..60.008679000 s, within the unchanged
250 ms lateness allowance. Sampled accounting does not exclude shorter
transients or guarantee that all file-backed pages are immediately reclaimable.
No full generation ran and no inference default or mathematical graph changed.

The release build and 77 integrated CPU examples pass. Seven matched malformed
CLI controls derive from the passing native argv and reject before report
creation or the memory observer; each has its own 10% cache-aware guard with
a 1 GiB cap and 10-second deadline. These establish early rejection of the
tested combinations, not branch coverage from a shared error string. The
offline verifier's mutation controls reject old 50-second evidence, missing or
reordered release/GC events, premature deferred comparison, changed hashes,
waits, root unions, pacing and guard failures. The Mini-Quad review is
correlated; its scope is bounded numerical, lifecycle, lineage and descriptive
timing evidence, not independent physical-cause identification.

Executed commands (the runner rejects reused reports and snapshot directories):

```sh
make spec_pixelumm_decoder_integrated
make BUILD_DIR=/opt/mldata_ssd/cogni-ml/hy-nomas-pixelumm-cpu-lifecycle60-20261029/build pixelumm_decoder_integrated_probe
python3 -B /opt/mldata_ssd/cogni-ml/hy-nomas-pixelumm-cpu-lifecycle60-20261029/run_integrated.py normal-1
python3 -B /opt/mldata_ssd/cogni-ml/hy-nomas-pixelumm-cpu-lifecycle60-20261029/run_integrated.py reverse-1
python3 -B /opt/mldata_ssd/cogni-ml/hy-nomas-pixelumm-cpu-lifecycle60-20261029/verify_results.py --self-test
python3 -B /opt/mldata_ssd/cogni-ml/hy-nomas-pixelumm-cpu-lifecycle60-20261029/verify_results.py
python3 -B /opt/mldata_ssd/cogni-ml/hy-nomas-pixelumm-cpu-lifecycle60-20261029/check_cli_negatives_matched.py
```

Evidence root:
`/opt/mldata_ssd/cogni-ml/hy-nomas-pixelumm-cpu-lifecycle60-20261029`.
Measured source parent is `af34cece`; both processes preserve all 202 measured
source hashes. Binary SHA256 is
`afc72a5d4dab022f40e153a923f816156679b3aa962640d215ac9b76395aa237`.
Paired analysis SHA256 is
`4589940af59d8eb7b179e2de776e6b22488bce79b5c1486d9d468b8d4e4eb0b0`;
normal/reverse report SHA256 values are
`d72807dc1438ca9491f7e78e744084fef51b02a54d47587faf010db1baed7911` /
`1ce672e4c462d61bf51285c9e22ef7a333bf998136b65809bec9cf7dcb41b1e8`.
The frozen contract, input manifest, source copies, runner and verifier bind
the conditional estimand and exclude the aborted 50-second attempt. Snapshot
extents are checked and in-run comparisons hash every actual stage, but the
snapshot payloads were not hashed at write time or separately after the run.
The artifacts therefore do not independently bind bytes at capture time to
the later comparison. Receipts contain a redundant null
`cpu_diagnostic_pause_seconds` key; their canonical `aa_pause_seconds`, recorded
argv and every per-request gap agree on 60 seconds. Refresh
after source, fixtures, weights, geometry, precision, backend, toolchain,
device, pacing, lifecycle, guard or evidence changes. Rollback omits the
alignment/diagnostic flags; ordinary inference retains its existing ordering.

`results/final-evidence-index-paired.json` SHA256
`d4a5b260969c4ab113c3480963a9fb39ac7596b394d425413973aa778adb8651`
binds the runner/verifier/contract, both report/receipt/guard sets, binary and
object hashes, all 202 measured source hashes, stage vectors and both CLI
control files. Its reconstruction compares JSON-normalized summaries because
persisted request-index object keys are strings; no numerical or lifecycle
gate is relaxed. Earlier CPU checks are labeled as interactive observations,
not invented durable test logs. The index is a curated evidence map, not an
independent proof of physical causality.

## Next performance falsifier

The next proposed candidate batches independent attention windows along the
batch axis, first B2, retaining each window's Q length, K context and reduction
dimensions. It does not concatenate query rows. At the full-J geometry the
shape classes are 40 windows of Q384/K3456, 10 of Q384/K2880 and 10 of
Q192/K2880. A B2 bank adds approximately 444 MiB of named F32 scratch, reused
across blocks; opaque MPS temporaries still need admission measurement.
The first gate must compare raw QK, then PV with the same baseline P, and then
the connected output against unchanged upstream tolerances before timing the
full block/decoder. The earlier same-K query-row concatenation failure remains
rejected. Fewer graph calls alone are not a speedup certificate.

## Verification and safeguards

The CPU-only contract tests overwrite the live source after saving it and
require the complete saved comparison vector to remain unchanged. They also
check 43-stage call counts, a rejecting NaN positive control, premature or
duplicate comparisons, wrong slot/request binding, changed file extent,
insufficient scratch and exclusive creation without overwriting existing files.
The native warm request rejects an attempted snapshot read before the wait.

```sh
make spec_pixelumm_decoder_integrated
make BUILD_DIR=/opt/mldata_ssd/cogni-ml/hy-nomas-pixelumm-cpu-diagnostics-20261029/build pixelumm_decoder_integrated_probe
python3 -B /opt/mldata_ssd/cogni-ml/hy-nomas-pixelumm-cpu-diagnostics-20261029/run_integrated.py normal-1
python3 -B /opt/mldata_ssd/cogni-ml/hy-nomas-pixelumm-cpu-diagnostics-20261029/run_integrated.py reverse-1
python3 -B /opt/mldata_ssd/cogni-ml/hy-nomas-pixelumm-cpu-diagnostics-20261029/verify_results.py --self-test
python3 -B /opt/mldata_ssd/cogni-ml/hy-nomas-pixelumm-cpu-diagnostics-20261029/verify_results.py
```

The fresh cache-aware admission gate and runtime guard preserve a 10% physical
memory reserve, a 28,160 MiB process-tree footprint cap, a 64 MiB swap-growth
stop and a 900-second deadline per process. File-backed and speculative pages
are counted once. Sampled telemetry cannot exclude shorter transients. All
large temporary files and builds remain under `/opt/mldata_ssd/`.

## Earlier unaligned 50-second result

VERIFIED for the completed bounded policy comparison, retained numerical
parity, lifecycle and sampled guard checks. Isolated CPU-comparator causality,
a stable effect size and production acceleration remain unproven. The baseline
Metal/MPS graphs and production defaults are unchanged.

Both native processes exited normally, their measured source pins remained
unchanged, and the final deferred comparisons completed. The following table
uses complete-request timing, including intervening memory telemetry, with
preparation and post-wait diagnostics excluded:

| Process | Preceding diagnostic | Request seconds | Encode seconds | Terminal wait seconds | Recomputed root union seconds |
| --- | --- | ---: | ---: | ---: | ---: |
| Normal | Eager | 63.953500 | 62.710485 | 1.242825 | 63.934997 |
| Normal | Deferred | 48.236443 | 47.284352 | 0.951796 | 48.215806 |
| Reverse | Deferred | 43.263825 | 42.333889 | 0.929649 | 43.240620 |
| Reverse | Eager | 48.303293 | 47.373546 | 0.929593 | 48.294255 |

The probe following deferred diagnostics took 24.5758% less time in the normal
order and 10.4330% less in reverse. This is two descriptive contrasts with the
same direction, not a replicated stable speed estimate. The condition requests
also vary: normal eager/deferred take 44.047282/52.613556 s; reverse
deferred/eager take 45.287303/54.428825 s. Host/device state and background load
were not independently controlled. Do not pool those controls or attribute the
whole difference to CPU comparison.

The difference also appears in the completed GPU root intervals. Each request
has 4,579 completed error-free roots; the analyzer recomputes their interval
union from raw timestamps. These intervals include time inside command
envelopes and do not measure active shader cycles or identify a driver wait.
Reverse's terminal wait is essentially unchanged while its encoding and root
union differ. Moving a terminal wait does not explain that pair.

The eager condition spends 46.466257/45.505355 s saving and comparing captures;
the deferred condition spends 9.237327/9.279095 s saving them. The deferred
comparison takes another 44.364224/44.106267 s after all six requests. This work
is moved, not eliminated. Snapshot I/O, expected-fixture reads, CPU comparison,
hashing, transient command retention and GC timing remain coupled in this
experiment. The next isolating contrast should release the completed chain
after the common snapshot write and align GC before either comparison policy.

All 12 requests pass all 516 finite retained comparisons at unchanged
atol=rtol=2e-5, with zero different BF16 words and max absolute error zero
against the saved upstream controls. Their 43-stage actual hash vectors are
identical within and across both processes. Every request has one existing
terminal wait, 43 post-wait capture reads, zero preterminal activation reads
and zero tracked named encode allocations. Actual pacing gaps are
50.000150625..50.010128625 s; no diagnostic overrun or lateness rejection occurs.

All 2,574 guard samples pass. The largest sampled process-tree footprint is
26.195773 GiB against the 27.5 GiB cap; the smallest file-cache headroom is
16.343918 GiB, above the 6.4 GiB reserve. Positive swap growth is zero. Both
processes start and end on AC power. These are sampled bounds and do not prove
thermal, clock or exclusive-host equivalence.

The release build, 74 integrated CPU contract examples, format/diff checks and
the external analyzer self-test pass. The self-test rejects false pass/pending,
extra wait, changed capture hashes, reversed predecessor binding, gap lateness,
forged root union/count, guard stop and excess swap growth. Four malformed CLI
configurations reject before the memory observer and report creation. The
correlated Mini-Quad verdict is ROBUST for these bounded observations and
VULNERABLE for isolated CPU causality or a stable speedup claim.

Evidence root:
`/opt/mldata_ssd/cogni-ml/hy-nomas-pixelumm-cpu-diagnostics-20261029`.
The prospective contract, runner, frozen source copies, binary/object hashes,
fixture pins and process receipts preserve the measured lineage. Rollback
omits `--compare-cpu-diagnostics`; the ordinary inference path is unchanged.
Measured source parent is `cfeb368b37416b1e3e5c1101c99214b80839d9e8` and the
release binary SHA256 is
`93f9ede7c8134b33093a3ed8217e27ca33cdb27d3d4eda2d48e4202678873454`.
`results/analysis-paired.json` SHA256
`8abe90461b91962d767ceabbcfef099dbfa025e1a6d4302645615b16783ac749`
records both qualified reports, exact hashes, per-order timings and guard
summaries. Normal/reverse report SHA256 values are
`92e48b90903e9493938c18ee66f4b5637c44bfbc1d5a3e4864c0d2c8837a8f84` /
`0e34913ad87fdc682e18fa6601e1c5b3acc98964dfa3357c58557e4527d171e2`.
`results/cli-negatives.json` SHA256
`a6bb703051e00cb8491d5f85f7763e13598cbaf320751febee24b0b1186a5937`
retains the four negative CLI results. Snapshot file extents are checked;
the entire concatenated files are not separately hashed. Comparison records
hash the actual bytes of every retained stage.
`results/final-evidence-index.json` SHA256
`bcb7c4380211e1f191f1432efc99e9a5d278964ebd5ef8a71786fa9721ad6708`
pins the runner, analyzer, contract, reports, receipts and guard logs, records
the observed CPU checks and confirms all 202 measured source files still match.
Refresh after source, input, weights, precision, geometry, backend, toolchain,
device, schedule, diagnostic, guard or artifact changes.
