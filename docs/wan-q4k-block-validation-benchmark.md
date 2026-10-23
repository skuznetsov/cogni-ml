# Q4_K sparse-delta block validation

Status: VERIFIED within the bounded projection experiment (2026-10-23).
Block-shared validation preserves exact decoded values and reduces K4/K8
decoder latency. K8 also reduces paired complete-projection latency by
4.59%/5.27% in two fresh opposite-order processes. This is an opt-in decoder
experiment; no model path or default is changed. The parent qualification is
[Q4_K sparse F16 deltas](wan-q4k-delta-benchmark.md).

## Experiment boundary

The current `wan_q4k_delta_decode` checks the same correction list, including
duplicate indices, for every output value in its 256-value block. The prediction
is that sharing that validation within one threadgroup reduces decode latency,
especially for eight corrections, without changing any decoded value. The
strongest counterexample is a malformed block contaminating a valid neighbor,
or a divergent threadgroup barrier. Focused multi-block tests must reject both.

This is a CAUTION performance change. Keep the existing per-value kernel and
API as the control and add an explicit `decode_block_validated` API. One
threadgroup owns one complete 256-value block; one lane validates its correction
records and shares indices, F32 deltas, and the malformed flag. A threadgroup
barrier is local GPU coordination. Add no global scratch, dispatch, host wait,
activation readback, or tracked explicit `ML::MetalBuffer` encode allocation.
Retain the exact separate F32 base-plus-delta arithmetic and sticky status
semantics: malformed correction lists zero their entire block; invalid base or
result values zero only the affected value. A preexisting status error must not
suppress a valid block. The packer, wire bytes, and tolerances remain unchanged.
Rollback is to continue using the existing API and omit the comparison flag.

## Paired experiment and Definition of Done

Use the same frozen `blocks.0.ffn.0.weight` projection (13824 by 5120), payloads,
sidecars, inputs, and independent upstream MPS fixtures as the parent experiment.
`--compare-block-validation` adds four execution arms ending in `_block` to the
four existing format arms. `q4_k_block` executes the same uncorrected decoder as
`q4_k` and is an A/A drift control. The other pairs differ only in correction
validation placement. Keep format storage accounting separate from execution
arms. Keep each old/new format pair adjacent, rotate four pair positions, swap
each pair every four measured rounds, and flip the complete order between
halves (A reverses the second half; B reverses the first). This balances all
eight execution positions and both pair orders
within each half. Run 16 measured rounds plus two opposite warmups per mode.
Repeat M4480 in a fresh
process with the opposite first order. M4 supplies a short control. M4480
repeats four seeded source rows and provides execution-scale evidence, not new
quality diversity or video-quality evidence.

DoD commands use `examples/support/wan_q4k_delta_experiment.py`: guarded native
`spec`, `analysis-spec`, `build`, and `probe` runs (the comparison flag applies
only to `probe`), then
`examples/support/wan_q4k_delta_results.py` over all three new probe receipts.
Expected signals are focused wire/isolated-block tests passing; exact GPU packed
bytes; full-weight and full-output finite parity before and after timing against
the unchanged upstream gate `abs(actual-ref) <= 2e-5 + 2e-5*abs(ref)`; balanced
pairing; zero new inter-operator waits, activation readbacks, or tracked encode
allocations; and intact source/fixture/binary/guard hashes. Paired GPU, wall, and
encode times must include the complete existing decode-to-F32-stage plus MPS
projection, as well as decode-only timing. Report drift and both process orders,
not just the fastest sample. Also stratify direct old/new pairs by slot and
half/order. The initial nonadjacent cyclic schedule was rejected before timing
because carry-over could confound its pair ratios. No speed claim is admitted
if A/A drift or an order/slot effect explains it. An aggregate ratio alone is
insufficient.

In comparison mode additionally require zero bit differences against each
unchanged same-format weight/output fixture, for both decoder placements in
both phases. This is a separate unchanged-arithmetic invariant; passing only
the existing finite tolerance cannot establish it.

Keep the cache-aware 15% physical reserve and pressure floor, 4 GiB process-group
footprint cap, and 64 MiB system swap-growth stop. All build/cache/evidence files
stay on `/opt/mldata_ssd`. No full model, Studio, block, video, or generation run
is admitted by this experiment. Existing opaque MPS and compiler-private storage
remain outside the explicit allocation counter and inside the sampled guard.

## Observed latency and drift

All times below are GPU milliseconds, with sixteen measured samples per arm
and mode. A and B are fresh M4480 processes with opposite initial order;
within each process every old/new pair is adjacent, all eight positions are
balanced, and both pair orders occur in each half. These are medians of the
individual arms, not estimates of the paired percentage change.

| Corrections | Decode A, old -> block | Decode B, old -> block | Complete projection A, old -> block | Complete projection B, old -> block |
| --- | ---: | ---: | ---: | ---: |
| 0, A/A control | 2.39898 -> 2.39177 | 2.39762 -> 2.39915 | 58.00442 -> 57.81365 | 64.63119 -> 64.35019 |
| 1 | 2.50329 -> 2.48369 | 2.50458 -> 2.48377 | 58.21425 -> 57.97185 | 64.45256 -> 64.61144 |
| 4 | 3.62633 -> 2.53417 | 3.59940 -> 2.52477 | 59.04198 -> 58.14679 | 65.94490 -> 64.43456 |
| 8 | 7.39960 -> 4.24546 | 7.43042 -> 4.24323 | 62.97452 -> 59.71060 | 71.39210 -> 67.63685 |

Direct same-round paired latency changes use
`100 * (block - old) / old`; negative means faster. Ranges are observed
minima/maxima, not confidence intervals.

| Corrections | Decode median change A / B | Projection A median [min, max] | Projection B median [min, max] |
| --- | ---: | ---: | ---: |
| 0, A/A | -0.12% / -0.09% | -0.29% [-1.63%, +2.23%] | +0.51% [-1.64%, +1.83%] |
| 1 | -0.94% / -0.71% | -0.31% [-1.63%, +1.62%] | -0.02% [-2.71%, +1.20%] |
| 4 | -29.06% / -29.24% | -1.50% [-2.81%, -0.03%] | -2.54% [-4.07%, -0.50%] |
| 8 | -43.00% / -42.71% | -4.59% [-7.36%, -3.75%] | -5.27% [-6.52%, -3.53%] |

K8 is faster in all sixteen full-projection pairs in each process. Its
eight old-first and eight block-first pairs improve in each process,
addressing the immediate order carry-over counterexample. Its
old/block GPU ratio medians by old execution position range from
1.04424 to 1.06025 in A and 1.04841 to 1.06332 in B; half medians are
1.05326/1.04813 and 1.05347/1.06255. Every nominal rotation median also
exceeds one. Even its least favorable pair exceeds the largest favorable
A/A deviation observed in that process. This supports a bounded K8
projection improvement beyond the observed drift, without establishing a
model-wide gain or a universal latency bound.

K4 has a clear decoder improvement in all pairs, slots and halves. All
full-projection pairs also improve, but the 1.50%/2.54% median changes are
close to the observed A/A spread; the exact small full-path magnitude is
less reliable. K1's decode and complete-path ranges cross zero and overlap
the A/A noise: no stable speed difference or equivalence is established.
The uncorrected complete-path median changes by about 11% between A and B,
so the two processes are reported separately rather than pooled. The
analyzer deliberately reports statistics without automatically promoting a
speed winner; schedule/hash validation cannot adjudicate thermal causality.

Complete K8 wall-time medians are 63.32319 -> 60.00477 ms in A and
71.70521 -> 67.95235 ms in B, with paired median changes of -4.56%/-5.23%.
Complete-path encode medians across all arms and both full-size processes
range from 0.04454 to 0.05054 ms. Their noisy percentage changes establish
no encode-speed improvement. Decode-only K8 encode medians change from
0.00346 to 0.00371 ms in A and 0.00381 to 0.00471 ms in B; the small
absolute CPU cost is retained in wall measurements.

The short M4 control gives K8 complete-path GPU 8.07525 -> 4.96067 ms
(paired -38.58%) and decode 7.26729 -> 4.16756 ms (paired -42.68%). K4
complete-path GPU is 4.31167 -> 3.14667 ms. These results do not substitute
for the full-size controls. The common 270 MiB F32 stage and existing MPS
projection remain; this experiment does not implement fused quantized GEMM.

## Verification, failure controls and memory

The 9 independent CPU reference tests, 15 analyzer tests and 9 native Metal
examples pass, with zero native failures, errors or pending examples.
Before implementation, the native spec failed because the new API was
absent; the extended analyzer spec rejected eight-arm commands/inventories.
The same tests pass after implementation. Native controls independently
check the old and new status buffers so the control kernel cannot mask a
missing error update in the candidate. They cover multiple adjacent
malformed blocks (duplicate positions, NaN, Inf) followed by a valid block,
invalid base values with a valid neighbor, nonzero corrections at positions
0/255, all quotas 1/4/8, sticky status, separate F32 addition/rounding, invalid
geometry, undersized buffers and aliases. Finite legal Q4_K/F16 operands
cannot reach a result-only F32 overflow; its unchanged guard remains but
that unreachable branch is not claimed as dynamically exercised.

The wrapper requires pipeline support for a full 256-thread group and uses
an exact group count. The old decoder requests 256-thread groups through
the existing dispatch default. All lanes in the new kernel reach one
unconditional threadgroup-memory barrier before any return. There is no
additional dispatch, global scratch buffer, host wait or readback.

All three probes pass full-weight and full-output upstream checks for all
eight execution arms before and after timing: zero bit differences,
tolerance violations and nonfinite values. Native GPU sidecars match the
independent oracle byte for byte. The BF16 control and injected finite/NaN
corruption controls pass. The finite `2e-5 + 2e-5*abs(reference)` tolerance
is unchanged, with a separate stricter bit-exact gate in comparison mode.
Lossy BF16-relative distortion is unchanged from the parent qualification:
K0/K1/K4/K8 pooled projection relative L2 remains
4.63165%/4.57702%/4.44768%/4.30013% on the four seeded rows. These metrics do
not establish video quality or original-BF16 exactness.

There are 864 timing samples including warmups across both modes and all
three processes. Each retains one existing terminal wait, zero inter-op host
waits, zero activation readbacks and zero tracked explicit
`ML::MetalBuffer` encode allocations. Parity exports, scalar status checks
and fixture construction occur outside the measured GPU intervals; no new
activation transfer or inter-op synchronization is added to the native path.

Native probes peak at a sampled 1.78563 GiB process-group footprint, minimum
estimated cache-aware headroom is 18.94270 GiB, pressure-free is at least 73%,
and sampled swap growth is zero. Fresh oracle construction peaks at
1.78839 GiB, with minimum headroom 19.70567 GiB and zero swap growth. The
same 15% physical reserve/pressure floor, 4 GiB cap, 64 MiB swap-growth stop
and deadlines apply throughout; file-backed bytes are only a reclaimability
estimate and between-poll spikes remain unobserved. No full model, block,
Studio, video or generation ran.

## Reproduction and lineage

Starting source is `fa135a1c` on `codex/wan-cross-segments`, Apple M2 Max,
macOS 26.6.2. Build/cache/evidence remain under
`/opt/mldata_ssd/cogni-ml/video-scene-20261019`. Let
`L=examples/support/wan_q4k_delta_experiment.py`. These commands all returned
zero with `changed_files: []`:

| Command after `python3 -B $L` | Observed signal |
| --- | --- |
| `cpu-spec block-green-v1` | 9 tests, OK |
| `analysis-spec block-green-v1` | 15 tests, OK |
| `spec block-green-v1` | 9 examples, zero failures/errors/pending |
| `build block-v1` | Source-bound release binary |
| `oracle block-r4-v1 --rows 4` | Independent four-row fixture |
| `oracle block-r4480-v1 --rows 4480` | Independent repeated-row fixture |
| `probe block-r4-v1 --build-tag block-v1 --fixture-tag block-r4-v1 --rounds 16 --compare-block-validation` | Eight-arm byte/parity/control checks |
| `probe block-a4480-v1 --build-tag block-v1 --fixture-tag block-r4480-v1 --rounds 16 --compare-block-validation` | Same checks, full-size paired timings |
| `probe block-b4480-v1 --build-tag block-v1 --fixture-tag block-r4480-v1 --rounds 16 --compare-block-validation --reverse-first` | Same checks, opposite initial order |

The old oracle receipts also pinned the earlier launcher source. Instead of
weakening those pins after adding the comparison flag, fresh guarded
oracles were generated with the unchanged reference producers. Their full
manifests are byte-identical to the parent fixtures. The complete pinned
source tensor, payload, sidecars, inputs, expected weights and expected
outputs are consequently unchanged. Independent PyTorch MPS supplies the
offline reference with CPU fallback disabled; the timed process is native
cogni-ml. The six reused native objects and their ancestry limitations are
unchanged: only the T5 linear bridge has complete historical source/object
ancestry; the other five have byte pins and current source pins, not a
reconstructed historical compiler lineage. The pinned canonical ggml
quantizer was not rebuilt in this experiment.

Analysis used the same guard and returned zero with unchanged analyzer,
spec and guard source pins. `EVIDENCE` is the external base plus
`evidence/wan-q4k-delta-20261023`:

```sh
python3 -B examples/support/wan_memory_guard.py \
  --memory-accounting=file-cache --max-footprint-mib=4096 \
  --max-swap-growth-mib=64 --timeout-sec=300 \
  --log="$EVIDENCE/analysis-block-v1.guard.jsonl" -- \
  python3 -B examples/support/wan_q4k_delta_results.py "$EVIDENCE" \
  block-r4-v1 block-a4480-v1 block-b4480-v1 \
  > "$EVIDENCE/summary-block-v1.json"
```

Intent/completion receipts bind this command, inputs and output hashes. Use
fresh exclusive tags for reproduction; never overwrite existing evidence.

| Artifact | SHA256 |
| --- | --- |
| `cpu-spec-block-green-v1.completion.json` | `848e145c4676ab44f9eb250fbfe6d407e703b37e6ef992b6246c688e8940975e` |
| `analysis-spec-block-green-v1.completion.json` | `b006eab091291b3c7dae1501016d92622b4a8bbcbbfb4591d0169ad722050cbf` |
| `spec-block-green-v1.completion.json` | `37fee9cb7a5e52661750be00b05ea9dc3110771a6bc3e2d661f0eafe9e11af47` |
| `build-block-v1.completion.json` | `a36666c3ee1a1f69ab3e8f11491dc34430d9f78164720c03a794e08b3177636e` |
| `oracle-block-r4-v1.completion.json` | `47461d021298e6c0d2a26716368999e8a08524e8fa2b87fa181175e1a8f21f6d` |
| `oracle-block-r4480-v1.completion.json` | `207a6453a48f9f918952446e81fb518d9896fbfe374ae2d5870e2ceaaabecbfd` |
| `probe-block-r4-v1.completion.json` | `388d15139d31e4ec64757483c02238f0737e91212b763924d64c2f2ff53e640e` |
| `probe-block-a4480-v1.completion.json` | `21cf03ec7db7ac6cc001d742b951510c991c4483e590b4ca2ead915ee1c46cf7` |
| `probe-block-b4480-v1.completion.json` | `1b2dae75e4571260de5072d89b0b800b7ec397c81a0ffd7750a2d40905d98cce` |
| `analysis-block-v1.completion.json` | `b153c33b8fe9a1ebbb7ee5780e0e05f573fd0c5456e395665c9a837548783a1b` |
| `summary-block-v1.json` | `c8669179f12ec37d567814356232390cce024945670bb5da314c463f9cab8b2c` |
| Release binary `probe-block-v1` | `9ec563ac1240c57099291562eddea07ffd52844a298a9c6e71924879986e765e` |
| Four-row manifest | `36b796df8ff172609cc78e75078d11dbf60446e82f3e4861fada4a2e76542891` |
| Repeated-row manifest | `204432b92773cf54b9db9482d1216e533e249a2e3d26a9ae244c2902860a5af3` |

## Decision and refresh boundary

The correlated Mini-Quad adversary verdict is ROBUST for the pinned
implementation, exact parity/failure controls, observed guards and bounded
K8 projection result. Small K1/K4 full-path magnitudes, representative
quality and model-wide speed remain VULNERABLE to drift or scope mismatch;
no such broader claim is admitted. Source/receipt agreement establishes
the tested context, not independence of model lineage or video quality.

Keep the block-shared decoder as an explicit opt-in API alongside the
per-value control. No checkpoint rewrite or production default promotion
is justified by a single-projection experiment. The remaining execution
frontier is reducing K8's record lookup cost or a competitive fused
decode/GEMM that removes the F32 stage; neither is implemented here. The
quality frontier remains held-out captured Wan activations and correction
allocation at matched serialized bytes.

Refresh parity, malformed-wire controls, guards and paired timings after
changes to decoder/wrapper/dispatch code, sidecars or payload, geometry or
inputs, compiler/native objects, OS/SDK/device, numerical policy or schedule.
Until refreshed, these observations do not certify the changed context.
