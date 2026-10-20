# Recorded Wan conditioning differences

The user authorized experimental video generation on 2026-10-19 while finite
conditioning differences are recorded for later investigation. This does not
admit the strict upstream gate or widen its tolerance:
`abs(actual - reference) <= 2e-5 + 2e-5 * abs(reference)`.
Non-finite data, changed inputs or weights, invalid geometry, memory pressure
and failed GPU completion remain hard failures.

## Audio encoder receipt

The production fixture uses the original XLSR checkpoint, 80,000 waveform
samples, 24 layers, F32 arithmetic, and output shape `[25, 1024, 84]`. Waveform
normalization preserves upstream eager operation boundaries; GELU and position
convolution use prepared MPS plans. Bucket interpolation now follows the pinned
PyTorch 2.8 MPS scale, coordinate and weighted-sum operation order.

The current native receipt is
`/opt/mldata_ssd/cogni-ml/video-scene-20261019/evidence/wan-full/audio-interpolation-native.log`:
six examples, one strict numerical failure, zero errors or pending examples.
The comparison captures all values before its first failing assertion, so the
external ledger also records later hidden states instead of stopping at the
first failure.

| Captured point | Maximum absolute difference | Values outside the unchanged gate |
| --- | ---: | ---: |
| Waveform and seven feature-extractor stages | 0 | 0 |
| Hidden state 0 after interpolation | 1.716614e-5 | 0 / 86,016 |
| Hidden state 1 | 3.814697e-5 | 0 / 86,016 |
| Hidden state 2 | 3.814697e-5 | 0 / 86,016 |
| Hidden state 3, first rejected captured state | 4.863739e-5 | 1 / 86,016 |
| Hidden state 23 | 1.274109e-3 | 15,382 / 86,016 |
| Hidden state 24 | 2.899766e-5 | 9 / 86,016 |
| Complete conditioning output | 1.274109e-3 | 52,450 / 2,150,400 |

All captured values are finite. The differences are not uniformly of order
1e-6. Neither a visible video artifact nor its absence has been established;
quality impact and the remaining accumulated encoder drift are open.

The full per-point metrics, fixture/capture hashes, source hashes, revisions,
configuration and scope are in the external `audio-conditioning-drift-ledger.json`
beside that log. Its current SHA256 is
`40703d6c21e15cbcf67f5c4b1b632823fde3dd099cdbd37d3df873e53aa3a876`.
Only `audio-conditioning-full-output` is eligible for the explicit experimental
exception. Default execution remains closed when required qualification is
missing. An experimental result must retain the ledger label/hash and cannot
be reported as strict upstream parity.

This receipt decays when any captured implementation, fixture, shape, source
revision, checkpoint or arithmetic route changes. Refresh by rebuilding and
running `spec/wan_audio_encoder_spec.cr` with the pinned production fixture,
then recomputing all captured comparisons with the same finite per-value gate.
If video artifacts appear, compare native and upstream from common latent,
noise and conditioning inputs before attributing them to these differences.

## Full-depth stack receipt

The next comparison executed all 40 distinct local checkpoint blocks and all
12 audio injectors on both normal and drop-motion branches, with exact oracle
prefix activations. This retains model width 5,120, 40 heads and FFN width
13,824, but uses two video frames, 16 spatial tokens and three context tokens.
It isolates the stack; it does not check native prefix conditioning or the
36,608-token generation trajectory.

The refreshed comparison uses prepared MPS F32 projections with the actual
typed-bias kernel and a shared F32 expansion stage; resident weights remain
BF16. All 104 captured points are finite. The unchanged gate rejects 23 of
31,948,800 compared values. The maximum absolute difference across the points
is 2.059937e-4; the worst normalized error is 2.134. The first rejected block
is `normal.block.11` (12 values, maximum absolute difference 4.094839e-5).
The strict spec reports one numerical failure, zero errors or pending cases;
the complete comparison is written before the assertion. It records one
terminal host wait and zero explicit MetalBuffer allocations during encoding.
The previous scalar route had 205 rejected values and maximum drift 4.959106e-4;
its separate historical ledger is preserved and does not bound the MPS route.

The external `s2v-real-stack-drift-ledger.json` records every point, exact
implementation and fixture hashes, configuration and source revisions. Its
SHA256 is `cf461cf1f5b5bcad2aa2167057e0a39a36880f30324d3e43a692db98d4fec702`.
The corresponding log is `s2v-real-stack-40-mps-g320-native.log`. Rebuilding
after changing long-attention command boundaries preserves all reported drift
metrics; this reduced fixture does not exercise those asynchronous boundaries.
This is a second
recorded numerical exception for experimentation, not strict stack parity.
The local BF16 tensor shapes, dtypes, shard mapping and index hash were
checked; full shard payload hashes were not computed.

Changing the projection backend, arithmetic route, weights, prefix or shape
invalidates this comparison as a numerical bound and requires fresh evidence.
No visible-quality conclusion follows from the small fraction of rejected
values. Keep the exact-prefix and intermediate-output probes available if
the eventual video contains artifacts.
