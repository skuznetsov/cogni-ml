# DRAMA-3 ordered GDN postprojection fusion

The opt-in Metal path combines per-head RMS normalization, SiLU of z and the
final multiply after the GDN recurrence. Each eligible call replaces three
dispatches and three result allocations with one. The frozen selector is
`DRAMA3_GDN_POST_FUSION=0|1`, default 0. Rollback is selector 0 followed by
construction of a new resident Context.

## Arithmetic and state boundary

Lane zero retains the original ascending F32 sum of squares for each head,
division by width, epsilon addition and inverse square root. After a local
threadgroup barrier, lanes apply the same normalization products, stable sigmoid,
SiLU product and final product. Contraction remains disabled. There is no tree
reassociation, precision conversion or changed activation function.

The API checks owners, nonempty compatible shapes, positive finite epsilon and
pipeline support for 64 threads before using the candidate. Other valid device
capabilities retain the original sequence. Offsets and CogniGraph buffer
lifetimes are preserved. Actual fused dispatches are counted, and the ordinary
Studio worker reports per-request deltas. A selector alone cannot establish
candidate coverage.

`BackboneResident#linear_attention_ready` calls this operation after the existing
recurrence dispatch. Convolution history, q/k normalization, recurrent state
writes and out projection are unchanged. The shared call covers scalar decode
and layer-major prefill. Production adds no readback, submission or wait; the
existing CPU EOS control boundary remains. This is kernel fusion, not LTP/WBA or
a temporal GDN scan.

## Qualification scope

The pinned device is Apple M2 Max, 38 GPU cores and 64 GiB unified memory. The
local checkpoint is
`cguldogan/drama-3-mlx-q4@5331ab509359c0e4a7e353a19e255ddf3e8152a8`.
The full-request input is `Hello, this is a voice test.`, seed 1, max 64 frames,
temperature 0.8, top-k 30 and top-p 0.9, without a voice reference. It produces
60 frames, 122,880 samples and 2.786395 seconds of mono 44.1-kHz audio.

The saved upstream oracle is community port `cguldogan/mlx-audio` revision
`e7488935ec2ebfafd38ef2b58e290d79f3d05dc7`, MLX 0.31.1. Qualification hashes and
reuses it; it does not regenerate MLX or certify the original Fish generator.
Keep existing tolerances `6e-5 + 3e-4*abs(reference)` and final hidden
`3e-5 + 3e-4*abs(reference)`. The old layer-39 strict failure remains open.

## Observed results — 2026-10-18

The final release Metal suite passes 25 examples. It covers default/frozen/strict
selection, head widths 1/31/32/33/64/127/128/129/256, nonzero offsets, graph depths
0 and 2, varying well-scaled and extreme inputs, nonfinite classes and rejected
owners/shapes/epsilon. Finite outputs must match bits; NaN payloads are outside
the comparison contract. Batched versus scalar prefill compares boundary/state
bits and decode continuation with the candidate both off and on.

The final standalone diagnostic captures the actual backbone positions for
decode rows 0/29/59, with all 30 linear-attention layers in their expected order.
All 90 candidate outputs match the explicit old sequence. GPU receipts establish
that x and z differ across both row intervals in all layers; 120 varying-input
comparisons pass. Local replay checks 7,372,800 values per arm, including two
warmup pairs, with zero finite-bit or nonfinite-class mismatches. Seeded ULP,
NaN and opposite-infinity controls are detected.

At the three saved upstream rows, all 243 native boundary pairs are bit-exact;
all 729 three-way comparisons retain the prior tolerances. Candidate and control
have identical MLX error metrics. Both retain the existing strict failure at
row 1, `backbone.layers.39.hidden`. This is native no-regression, not strict
upstream parity. The saved upstream oracle was not regenerated.

Two counterbalanced blocks, schedules 0/1/1/0 and 1/0/0/1, run four fresh workers
each. Each worker has one cold and two warm requests: 24 requests total, with
16 warm requests used for timing. Every request preserves all 60 generated code
frames, the 704-entry F32 sampler tape, EOS and WAV SHA-256
`38f2530ed1c90f0caba45e222432909ac5d92cc3633e4c34a21fbe39d693e07d`.
Both paths retain 122 submissions/completions/waits, zero pending/retained bytes,
496 control-read bytes, 494,160 terminal-read bytes and no diagnostic activation
reads. Candidate requests execute 2,670 fused calls; controls execute zero.

Warm allocation counts fall from 135,590 to 130,250, a reduction of 5,340.
Cumulative requested bytes fall by 87,490,560. These counters are neither peak
resident memory nor measured DRAM traffic.

## Timing and decision

Local replay uses the same 90 captured calls twice per arm, two warmup pairs
and eight measured pairs with counterbalanced order. Existing completion events
are joined by unique command index and checked for successful completion and
valid timestamps. Median command intervals are 6.369 ms unfused and 2.486 ms
fused; the median paired ratio is 2.5806x, with all eight pairs favoring fusion.
Median host encoding times are 3.449 ms and 1.150 ms. Commands run with depth
two and may overlap. These are not isolated shader latencies or additive
request costs.

The full-WAV table uses medians of each worker's two warm requests, then medians
across workers in each mode. Times exclude model loading.

| Block | Generation off/on, s | Codec off/on, s | Complete request off/on, s |
| --- | --- | --- | --- |
| Forward | 21.510 / 20.532 | 4.185 / 3.514 | 25.777 / 24.247 |
| Reverse | 20.013 / 19.265 | 2.905 / 2.580 | 22.953 / 21.923 |

The four mirrored request contrasts have off/on ratios 1.1652, 0.9689, 0.9622
and 1.1342; two regress. The unchanged DAC also moves materially. Shared-host
load was uncontrolled, and repeated requests in one worker are not independent
host samples. Keep default 0: repeatable causal full-request benefit is not
established. The complete-request observations correspond to about 7.87–9.25 s
of wall time per second of output audio, not real-time generation.

The mini-Quadrumvirate found and closed diagnostic gaps in temporal row identity,
exact layer order, explicit selector-off baseline proof, source pins and frame-cap
coverage. The final offline driver suite passes 36 checks. A separate root
receipt audit rejects 29 deliberately damaged fixtures, including missing rows,
fallback counters, stale input snapshots, widened tolerances, changed tape,
additional waits and missing/duplicate GPU timestamps, without modifying the
original artifacts. Correlated source review is ROBUST for the bounded claims;
it is not independent runtime replication. Candidate model files are hashed,
but the saved control does not authenticate its model payload. Direct diagnostic
sources are checked against compile-time hashes; this is not a whole-toolchain
attestation.

An additional broad Python discovery attempt ran 314 tests and ended with one
import error in the existing `drama3_mlx_moe_capture_spec`: `torch` is absent in
that Python environment. It is not counted as a passing qualification gate;
no dependency was installed. The focused 36-driver-check command passes.

The next mathematical falsifier is known-input prefill GDN block composition:
compare every output and the committed state, including prefix and replay costs,
against the existing rowwise path at production dimensions. An affine identity
over real numbers alone cannot establish unchanged F32 results. Surrogate
decode remains a separate approximate proposal/verification experiment; the
historical Qwen greedy verifier does not certify stochastic DRAMA sampling.

## Reproduction and refresh

The focused and adjacent release Metal suite is:

```text
DRAMA3_BACKEND=metal crystal spec --release -Ddrama3_metal \
  spec/drama3_gdn_post_spec.cr spec/drama3_resident_spec.cr \
  spec/drama3_prefill_spec.cr spec/drama3_graph_resident_spec.cr
```

Use the existing bridge object and Metal/Foundation/MetalPerformanceShaders
link flags recorded in the evidence packet. Run builds and GPU work serially
under `scripts/run_safe.sh`, with external `TMPDIR`/Crystal cache, local weights
and the authorized 15% free-memory floor. No unrelated processes or live Studio
are controlled by these checks.

`examples/drama3_dense_decode_compare.cr` accepts `--gdn-post` in addition to its
previous modes. The local benchmark is `examples/drama3_gdn_post_benchmark.cr`
with `MODEL_DIR BASELINE_JSON OUTPUT_DIR`; use a current arm-0 worker report
with an explicit disabled GDN selector and zero candidate dispatches. The paired
worker driver is `scripts/drama3/measure_host_pairs.py` with
`--gdn-post-schedule 0,1,1,0` and its reverse, `--warm-repeats 2`.

Commands, artifact hashes and bounded outcomes are indexed by
`.landmark/evidence/drama3-gdn-post.json`. Raw diagnostics remain on the external
SSD under `/opt/mldata_ssd/cogni-ml/drama3-gdn-post-20261018/`. Missing evidence,
changed source, compiler/driver, device, checkpoint, sampler, text, reference,
queue depth or combined flag stack reopens the associated claim. This is a
bounded native no-regression check, not a universal cross-device bit guarantee
or an independent build/model attestation.
