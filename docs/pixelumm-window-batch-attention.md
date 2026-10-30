# PixelUMM independent-window attention batching

## Active frontier

One full-J block passes numerical qualification and two bounded timing cohorts.
The diagnostic remains opt-in; full36 plus head and production promotion are
unqualified. No full video generation was run.

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
   Do not infer a speedup from a drifted control. A promising candidate still
   needs the full36 plus head gate on both saved inputs before promotion.

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
guarded paired full-block timing if the numerical gates permit it. A correct diagnostic
that refutes B2 is a useful result, not permission to widen tolerances.

The oracle is the existing full-J fixture set at upstream revision
`911cea6458a89c2663513ff7099561ec190dd491`. This is a comparison with saved
upstream tensors, not a freshly regenerated upstream neural forward. Raw QK/PV
gates isolate the changed MPS operation against the current native control.
Source, fixture, cache, binary and guard-log digests must accompany any result.
Changing those inputs or the GPU/backend invalidates the numerical certificate;
timing additionally decays with host load, pacing and resource-lifetime changes.

## Observed result, 2026-10-29

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
and evidence reviews support only the bounded qualification above. The 36-plan
helper additionally needs a common-Workspace identity guard before broader use.

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

Decision: keep B2 diagnostic-only. The observed command-count reduction yields
a small full-block gain and costs additional memory in this A/B layout. Before
production use, require full36 plus head on both saved inputs, a qualified
workspace/lifetime contract and a bounded aggregate memory/timing result.
Prioritize GPU execution costs over command-count targets when choosing the
next optimization; a shorter encode interval alone is insufficient.
