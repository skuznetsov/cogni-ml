# DRAMA-3 native expert row packing

Source base: `3a83aadb`, Apple M2 Max, 2026-10-16. This experiment preserves
native F32 arithmetic while sharing a 64-thread group across several expert
output rows. It establishes a local kernel improvement, not a request speedup.
The selector remains **off by default**.

## Scope and rollback

`DRAMA3_EXPERT_PACKED_ROWS=1` enables the candidate in batched native
`Context#experts` only. The setting is validated as exactly `0|1` and frozen
when the Context is created. Unset or `0` uses the existing kernel.

Eligibility requires affine Q4/Q8 weights, group size 64, effective native
thread width 64, and 1–32 groups dividing 64. The candidate maps
`64 / groups` output rows into a threadgroup. Each lane retains the original
byte-assembled packed-word reads, affine dequantization, serial F32 group
subtotal and ascending final group sum. All lanes write scratch and reach
the barrier; partial final row groups suppress their reads and output writes.
The full matrix row count remains the weight and output rank stride.

Scalar expert/matvec, MLX-Q4 reduction, requested 32-thread launches and
unsupported shapes retain their prior paths. Decode's batched MoE and
sequence prefill can reach this kernel; the `experts_rows` helper itself is
prefill only. The candidate adds no output allocation, dispatch, command,
submit, completion wait, control read or activation read. Threadgroup scratch
is 256 bytes in the supported branch. Rollback is `DRAMA3_EXPERT_PACKED_ROWS=0`.

## Discriminators

The standalone `examples/drama3_expert_load_bench.cr` uses pinned layer-39
weights and captured upstream inputs, rank order and per-rank down activations.
Gate/up are 512×2048 and down is 2048×512, with eight selected experts,
Q4/group64. The legacy production selector is explicitly pinned to zero in
the final probe so neither an inherited setting nor a future default change
can replace its baseline.

Two excluded warmup pairs precede seven measured alternating pairs per
projection, with 32 repeated dispatches per arm. Both projection commands are
queued before a diagnostic receipt command. Timings are existing command GPU
intervals in this paired workload; overlapping intervals are not additive
family costs or isolated single-launch latency. Every repeated output is
compared on GPU for exact bits and nonfinites. An injected bit/NaN control
must be rejected. The excluded odd-tail probe covers 511 of 512 gate rows;
it does not by itself establish down-tail correctness.

The executed production-kernel specs cover down-like per-rank and gate-like
shared-input odd row tails in Q4/Q8, supported group1, fallback groups3/33/65,
requested32, scalar and MLX-Q4 controls, frozen/unset/invalid selection, and
invalid expert IDs. A compact GPU receipt requires exact bits and the expected
nonfinite count; its ULP/NaN negative control is rejected. Pipeline-cache deltas
check branch selection alongside output parity. All 13 Metal examples pass.
Only 16-byte receipts cross the new tests' terminal comparison boundary.

The complete-request driver uses profiler-off order `0,1,1,0`, with coalesced
DAC fixed, one excluded cold request and two warm requests per fresh worker.
Every request must retain all codes, sampler inputs, WAV digest and existing
submit/wait/read ledgers. Warm repeats within a worker are correlated, and an
active host can change paging or competing load abruptly; ABBA controls a
linear trend only. The saved baseline is a correctness control, not a timing
comparison to the newly measured workers.

## Observed result

Both local runs compare 7,077,888 values with exact bits and no nonfinites.
All seven measured pairs per projection favor the candidate in each run.
Initial native/candidate command ratios are 1.02353 gate, 1.02413 up and
1.30653 down. The final source replay, with the baseline explicitly pinned
off, gives 1.02022 gate, 1.02347 up and 1.29316 down. Final median command
durations for 32 repeats are 38.692750/37.933333 ms gate,
37.535208/36.677250 ms up and 19.198208/14.995875 ms down. These ratios
qualify only the local paired workload; the later replay is not independent
host replication. The 81 commands complete with no pending work and only
348 bytes of terminal receipts, including the negative and tail controls.

All 12 complete requests preserve codes, sampler inputs, WAV and read/wait
ledgers. Warm worker medians in seconds are:

| Arm | Packed rows | Request | Generation | DAC |
| --- | --- | ---: | ---: | ---: |
| 0 | off | 25.090711 | 22.357012 | 2.694725 |
| 1 | on | 29.388533 | 25.365839 | 3.927782 |
| 2 | on | 25.688299 | 21.974873 | 3.655346 |
| 3 | off | 26.168557 | 22.218319 | 3.851342 |

Mirrored request ratios are 0.853759 and 1.018696. Median worker summaries
are 25.629634 s off / 27.538416 s on for 2.786395 s audio, or RTF
9.198 / 9.883. No full-request benefit is established. DAC timing also varies
despite its fixed implementation; this run does not isolate a causal kernel
regression. Keep the selector off; a local win does not admit promotion.

With the production option on, the full matched-input MLX report is identical
to the prior native report: all 162 boundaries and 156 replays, metrics,
readbacks and queue statistics. Both reports have SHA256
`cee6b1f404d6be0642a3357f914fb3b9e681edec8d641b36e489eb87ad59bd68`.
Its expected nonzero exit preserves the existing failed strict gate, rather
than establishing upstream compatibility. The strengthened final 13-example
Metal run also passes, including environment flips before dispatch in both
directions to check that the selected kernel is frozen. Existing upstream
failures remain a separate numerical frontier; tolerances are unchanged.

## Reproduce and continue

Build with the existing Metal bridge/framework flags and `-Ddrama3_metal
--release`, then use:

```text
drama3-expert-load-bench MODEL_DIR FULL_MOE_TRACE --candidate=packed-rows --repeats=32
python -B scripts/drama3/measure_host_pairs.py --binary STUDIO --model MODEL_DIR --output NEW_DIR --baseline BASELINE --expert-packed-rows-schedule 0,1,1,0
crystal spec --release -Ddrama3_metal spec/drama3_resident_spec.cr [bridge/framework flags]
```

Raw source/build/binary joins, GPU receipts and complete-request results are
under `/opt/mldata_ssd/cogni-ml/drama3-realtime-20261016/`; the curated index is
`.landmark/evidence/drama3-expert-packed-rows.json`. Serial runs use the 15%
free-memory floor and external scratch, with local weights and unrelated
processes untouched. Refresh after source, compiler, model, operand, reference,
queue, device, OS or host-load changes, or artifact loss.

A distinct next hypothesis is to encode qkv/z/b/a, which share one decode
input, with a multi-output dense dispatch while preserving each row's exact
reduction tree. No current additive profile establishes it as the dominant
cost. Require captured-input GPU parity and a local paired probe before
complete-request qualification; do not repeat previously rejected blanket
SIMD selector trials without a new discriminator.
