# Ordinary prefill command diagnostics

`QWEN35_PREFILL_COMMAND_TRACE=1` adds flushed stderr records around the existing
ordinary shared-command commit/wait in `qwen35_cpu.cr`. Unset, zero and other
values disable it. It does not enable GPU timing, change command grouping,
insert sleeps or retries, alter cache publication, or modify CogniGraph.

The motivating failure is the second-call `Impacting Interactivity` observed
with FFN capacity reuse both on and off (see `qwen-memory-sampling.md`). The
existing boundary profiler writes after successful wait/publication and cannot
identify a failed wait through its completed timing records alone.

Each command emits `qwen35_prefill_command` with:

- `phase=submit_wait_begin` before submission, followed by `submit_wait_end`
  or `submit_wait_failed` if control returns through the corresponding path;
- trace-local sequence and command object identity (not a Metal command ID);
- `start_pos`, logical `rows`, and the existing fused-group counter;
- loop cursor at the previous ordinary flush (initially zero) and at this flush;
- host elapsed milliseconds on the terminal record, **not GPU execution time**.

Cursor positions localize control flow; they are not an exact encoded layer
interval. For example, standalone adaptive work flushes before incrementing
the current layer. Group count likewise is the existing rotation counter, not
an exhaustive dispatch or layer count. Trace identity is local to one process.
CogniGraph, private/internal command buffers, finalization before submission,
publication after wait, and hard process/device termination are not covered.
A successful wait record does not certify successful cache publication.

Logging is best-effort and flushed before submission. Logging exceptions are
suppressed; command exceptions are rethrown unchanged without logging their
payload. A blocked stderr sink can still perturb latency. The safe runner may
buffer stderr until process exit, so a flushed record is not necessarily live
in the caller's log. Do not use diagnostic timing as a performance promotion.

## Verification and rollback

Six no-GPU specs exercise explicit opt-in, pre-submit visibility/flush, result
preservation, failed-wait emission/original exception identity, broken logging
and repeated command identity. A source guard checks ordinary-branch placement
before publication and the adaptive flush-before-cursor-increment distinction;
it is not runtime route coverage. Commands:

```sh
CRYSTAL_CACHE_DIR=/private/tmp/qwen-command-trace-spec crystal spec spec/qwen_prefill_command_trace_spec.cr
crystal tool format --check src/ml/gguf/qwen_prefill_command_trace.cr src/ml/gguf/qwen35_cpu.cr spec/qwen_prefill_command_trace_spec.cr
git diff --check
```

CPU-only compilation and the release CrystalBall two-call provider build also
passed on 2026-09-11. These checks establish the diagnostic helper and compile
integration, not GPU stability or complete route coverage. Rollback is unset
flag; reverting this diagnostic slice removes it. The separate capacity-reuse
experiment remains default-off and uncommitted.

Before GPU use, pin the new binary and source/input identities, retain the
existing 35% free-memory floor, 24,576-MiB cap, 300-second timeout, 2,048-row
chunks, group limit 1 and cooldown 50 ms. Run one feature-OFF diagnostic replay
and stop at its first failure; no uninstrumented retry loop. A source, build,
route or device change invalidates causal comparisons until rechecked.

## First guarded diagnostic replay (2026-09-11)

One feature-OFF replay with trace enabled failed with the same Metal
`Impacting Interactivity` / completion status `-6`. No retry followed. There
were 131 begin records and 131 matching terminal records, including one failed
wait and no unmatched command identities. The helper's failure path therefore
also has real-route evidence, not only a simulated throwing block.

Call one completed the captured 27-token output in 91.333 seconds. Call two
hit the resident prefix at 7,839 tokens with 195 helper rows. Its ordinary
shared-command trace was:

| Sequence | Cursor before / at flush | Groups | Host submit/wait ms | Result |
| --- | --- | --- | --- | --- |
| 1 | 0 / 7 | 1 | 656.780 | returned |
| 2 | 7 / 11 | 1 | 273.420 | returned |
| 3 | 11 / 15 | 1 | 210.074 | failed |

These observations localize the failure to the third ordinary shared-command
wait in the suffix. They do not identify a kernel, prove an individual-command
duration threshold, or exonerate cumulative occupancy, driver state or resource
history. The first command took longer to return successfully than the failed
command took to return an error; host elapsed time alone is not a GPU watchdog
threshold measurement. The two-call quality gate remains red, with no second
output. Do not classify the trace analyzer's successful pairing check as a
successful inference replay.

Launcher wall time was 99.971 seconds, exit 1; the observer exited 0 with 50
samples and no collection errors. Sampled free memory started at 76% and reached
43%, above the 35% floor; no floor kill was logged. All preceding launch guards
were retained. Logging/rebuilding can perturb timing, so this is not an exact
performance comparison with the uninstrumented binary.

Evidence root: `/private/tmp/qwen-command-trace.KC3KjM/`, with `run_inventory.py`,
`check_trace.py`, `command-trace-off-{dry,gpu}` logs/result/samples and source
hash manifests for cogni-ml `src` plus `build/bridge.o`, and the tracked
CrystalBall provider sources/probe/lockfile. The manifest files were captured
during the build and rechecked afterward; they are not a complete hermetic
toolchain/dependency attestation. Binary SHA256:
`07bb89c7fb055d28bd0cc3e217b0831a62ce73b3cc4d27a0416d1533d79f7f6a`.
The experimental FFN capacity-reuse source remained present but disabled.

ROBUST for bounded ordinary-command failure localization and exception-preserving
telemetry; the underlying interactivity failure is unresolved. Next inspect the
encoded operations at cursor 11 / 15 before designing a smaller-command
falsifier. No scheduling change, default enablement or further GPU run is
justified merely by this location. Temporary artifact loss or changes to the
source/model/device/route require refreshed evidence.
