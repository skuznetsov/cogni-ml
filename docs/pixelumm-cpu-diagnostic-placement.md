# PixelUMM CPU diagnostic placement

## Scope and prospective discriminator

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

## Result

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
