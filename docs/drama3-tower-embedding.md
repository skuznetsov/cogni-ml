# DRAMA-3 codebook embedding experiment

The tower's unseen-codebook sum previously used 26 allocations/dispatches for
eight codebooks. `DRAMA3_TOWER_EMBEDDING_FUSION=1` replaces that chain with one
GPU dispatch reading the original BF16/F16/F32 checkpoint bytes. It adds each
codebook in the same sequential FP32 order, including the initial positive
zero. Scaled tensors retain the legacy path. Mapped backing storage, frame and
output remain retained by the existing command lease. No intermediate host read
or completion wait is added.

The default stays `0`: a faster isolated operation did not produce a confirmed
whole-generation speedup on the shared host.

## Measurements

On Apple M2 Max, the benchmark uses the real checkpoint's `[40960, 2048]` BF16
embedding table, two warmup pairs and seven alternating measured pairs of 64
operations per arm. At each graph depth all 1,179,648 compared output elements
are bitwise equal. Existing pair-boundary waits/readbacks are unchanged.

| Graph depth | Legacy median GPU us/op | Fused median GPU us/op | Median paired speedup |
| --- | ---: | ---: | ---: |
| 0 | 77.18 | 6.42 | 10.00x |
| 2 | 69.33 | 6.54 | 10.69x |

These are GPU timestamps for the embedding operation. The ratio of separate
medians need not equal the median of paired ratios. Timings are exploratory
shared-host measurements, not a quiet-host promotion gate.

Four native worker arms in legacy/fused/fused/legacy order each run a cold and
warm job with `Hello, this is a voice test.`, seed 1 and a 64-frame cap. All eight
jobs stop at EOS after 60 frames and produce identical full code IDs and WAV
bytes. Warm allocation counts fall from 135,590 to 133,034, but median warm
request wall time is 29.53 seconds legacy versus 31.54 seconds fused. This small
sample rejects promotion based solely on the kernel result; it does not prove a
general regression. DAC scratch and allocation profiling were disabled in both
arms, with graph depth 2 and private mappings.

The same full 60-frame code trajectory matches a fresh pinned MLX F32 core
control with identical F32 inverse-CDF sampler inputs. First-prefill hidden and
main logits pass the existing `3e-5 + 3e-4 * abs(reference)` gate: maxAbs
`4.7683716e-6` and `1.4305115e-5`, zero pointwise failures/nonfinite. This is
bounded evidence; existing longer-generation MoE and DAC discrepancies remain.

## Reproduction and next discriminator

`examples/drama3_embedding_benchmark.cr` takes a checkpoint path and optional
`--ops=64 --pairs=7 --depth=0|2`. Build with `--release -Ddrama3_metal` and the same
Metal bridge/MPS flags as the studio, under `scripts/run_safe.sh`. Run serially at
`--depth=0` and `--depth=2`; keep model, reduction flags and host
conditions matched. Focused coverage is in `spec/drama3_tower_resident_spec.cr`.

The next useful probe separates host command encoding/allocation time from GPU
time for backbone, tower and DAC in interleaved full-job pairs. The remaining
135,590 warm allocations are an opportunity to investigate scratch ownership,
not evidence that allocations dominate runtime. Any reusable workspace must
prove backing-handle liveness, error-drain behavior, unchanged arithmetic and
the same command/readback ledger before promotion. This experiment is ordinary
fusion, not an LTP/WBA certificate.

Raw runs are under the local `drama3-runs/studio-20261015/` directory; the compact
index is `.landmark/evidence/drama3-tower-embedding.json`. Refresh on source,
checkpoint, fixture/tape, runtime/device/OS, graph or precision-policy changes.
Rollback by leaving the fusion flag at `0`, or reverting the isolated commit.
