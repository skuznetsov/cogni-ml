# DRAMA-3 exact-tree four-projection experiment

Source base: `932e7b6b`, Apple M2 Max, 2026-10-17 UTC.

`DRAMA3_BACKBONE_DENSE_FUSION=1` combines decode-time qkv/z/b/a projections
into one dispatch. Unset or `0` keeps the original four inline matvec calls.
The selector is strict and frozen when the backbone is constructed. Keep it
off by default: the local projection experiment improves, but the complete
request experiment does not establish a benefit. Realtime generation and the
existing upstream precision failures remain open.

## Implementation and rollback

Thirty linear-attention layers each save three dispatches per backbone decode
step. The candidate preserves the four fresh output allocations in their
original order, mapped weight views, input view, byte decoding, each lane's
F32 multiply/add order and the 64-lane tree reduction. It does not repack
weights, share output storage, change prefill, or add commands, waits or reads.
The four dispatches already share a command stream; this saves dispatch and
binding work rather than four synchronization boundaries. Bytes and FLOPs are
unchanged. This is ordinary dispatch fusion, without an LTP/WBA claim.

Only tree reduction with four rank-two, unscaled/unbiased F32/BF16/F16
matrices, expert zero and positive bounded row counts is eligible. Other
formats and reductions use the established four matvecs. Owner and all four
input widths are checked before allocations or dispatch. Each projection has
its own weight offset, encoding, row count and output binding. Rollback is
`DRAMA3_BACKBONE_DENSE_FUSION=0`; no data migration or weight conversion is
involved.

## Local paired projection gate

The release Metal benchmark uses three pinned upstream normalized inputs and
actual layer-zero BF16 qkv/z/b/a matrices of shapes 8192/4096/32/32 by 2048.
It verifies the actual mapped payload hash, checkpoint config/index hashes,
input/reference hashes and reference model/source revisions before GPU work.
Each row has two warmup pairs and seven measured pairs, with eight projection
groups per arm and alternating separate/fused order. Existing GPU timestamps
measure command intervals; they are neither isolated shader costs nor an
additive model profile.

Final production-API run `dense-fusion-r2`:

- All 2,668,032 native output values compare bitwise on GPU, with no nonfinites.
  Injected one-bit and NaN controls produce the required `[2,1]` failure receipt.
- Median paired separate/fused GPU ratios are 1.19018, 1.15811 and 1.16329
  for the three rows; pooled median is 1.16824. All 21 measured pairs favor
  fusion in this bounded sample.
- All 37,056 MLX projection values pass the unchanged `atol=6e-5, rtol=3e-4`.
  Maximum absolute errors are qkv 3.815e-6, z 2.027e-6 and b/a 9.537e-7.
  Native and MLX outputs are not bitwise identical.
- Eighty-four diagnostic commands complete with no pending/retained resources.
  Only 464 bytes of compact terminal receipts are read; activations stay GPU.

The earlier copied-kernel `r1` probe is preliminary; the qualified `r2` calls
the production API. External analysis joins build/run source hashes, unique
output binary, invocation IDs, paired command indices and valid timestamps.

## Decode fidelity and guards

Seventeen release Metal examples pass: thirteen existing resident examples and
four new fusion examples. Guards cover uneven row domains; widths
31/32/33/63/64/65/2048; mixed F32/BF16/F16; cancellation; odd and aligned raw
weight offsets; sliced input; sliced outputs and sentinel storage; SIMD/Q4
fallbacks; early owner/width rejection; strict frozen selection; and rejecting
one-bit/NaN controls. Nonzero output offsets use a test-only direct invocation
of the production kernel, because the public method allocates fresh outputs.

`drama3_dense_decode_compare` executes `BackboneResident#step` in both arms on
the same three uploaded input rows, preserving each arm's recurrent state.
It observes actual frozen off/on selectors and compares all 243 boundaries
(497,664 native F32 values) on GPU. Both arms also compare to the pinned MLX
scalar trace, producing 729 compact comparison receipts. All native values
match bitwise; all upstream metric records are identical between arms.

The full strict upstream result remains **false**, with the same sole failing
boundary at row 1, layer 39 hidden. Intermediate tolerance remains 6e-5/3e-4,
and final hidden remains 3e-5/3e-4. This proves preservation within the tested
decode fixture, not general upstream compatibility. One terminal diagnostic
command/wait drains all captures; 14,580 terminal metric bytes and the existing
four-byte finite status cross the boundary. No activation is read to CPU.
The older late-layer prefill harness does not cover this decode dispatch.

## Complete profiler-off request experiment

The counterbalanced driver uses one binary, four fresh workers in off/on/on/off
order, and one excluded cold plus two warm requests per worker. Text is
“Hello, this is a voice test.”, no reference, seed 1, cap 64, temperature 0.8,
top-k 30 and top-p 0.9. Both arms use graph depth two, tree reduction, batched
native experts, private mmap and coalesced DAC. Scratch and observers are off.
Runs are serial under `run_safe.sh`, with a 15% memory floor and unrelated
host work left running.

All 12 requests preserve the complete 60-frame code sequence, uniform sampler
tape, EOS, 122,880 samples, WAV SHA, readbacks and command/wait counts. Cold and
warm allocation counts/bytes also match their corresponding controls. A warm
request has 135,590 allocations requesting 4,079,562,724 bytes cumulatively;
these are allocation totals, not simultaneous physical residency. Each request
has 122 submissions/completions/waits, final pending/retained zero, 496 control
read bytes and 494,160 terminal read bytes. There is no new synchronization.

| Metric | Original | Fused |
| --- | ---: | ---: |
| Median worker warm request, seconds | 24.626830 | 25.464262 |
| Generation, seconds | 21.286234 | 21.241404 |
| Unchanged DAC, seconds | 3.285651 | 4.149602 |
| Request seconds per audio second (RTF) | 8.83824 | 9.13879 |

The WAV is 2.786394558 seconds. Mirrored original/fused request ratios are
0.983651 and 0.952369; generation ratios are 1.035217 and 0.972357. These do
not establish a request speedup. Unchanged DAC ratios also vary substantially
(0.717717/0.855016), so the request slowdown cannot be attributed solely to
the fused projection. Worker warm repeats are correlated, and ABBA does not
remove abrupt paging or competing workload changes. Studio records requested
environment controls; actual frozen arm selection is observed separately by
the decode harness and selector specs. There is no claim of an independently
instrumented dispatch receipt in every Studio request.

## Reproduction and evidence

Build the three examples with release Metal flags and the existing native
Metal bridge. Execute heavy work serially under `scripts/run_safe.sh`:

```text
drama3_dense_fusion_benchmark MODEL DENSE_INPUTS_JSON DENSE_REFERENCE_JSON --repeats=8
drama3_dense_decode_compare MODEL INPUTS_JSON MLX_SCALAR_TRACE OUTPUT_JSON
crystal spec --release -Ddrama3_metal --link-flags METAL_BRIDGE_FLAGS spec/drama3_resident_spec.cr spec/drama3_dense_fusion_spec.cr
python3 scripts/drama3/measure_host_pairs.py --binary STUDIO_BINARY --model MODEL --output OUTPUT --baseline BASELINE_JSON --prefix UNIQUE_PREFIX --dense-fusion-schedule 0,1,1,0 --warm-repeats 2
```

Exact paths, commands, environment, log/source/binary/input hashes and derived
analyses are indexed by `.landmark/evidence/drama3-dense-fusion.json`. Raw
evidence stays in `/opt/mldata_ssd/cogni-ml/drama3-realtime-20261016/`; weights
stay local. The same-lineage mini-Quad review is correlated; root executed the
builds, tests and GPU experiments. Its bounded verdict is ROBUST for native
parity/local timing, VULNERABLE for a request-speedup or full upstream claim.

Refresh on source/compiler/model/input/reference/tolerance/queue/device/OS
drift or evidence loss. Host-load drift invalidates timing generalization.
Keep the opt-in candidate for controlled experiments; remove it if refreshed
full-request controls continue to show no benefit and no further measured
consumer needs it. A separate next hypothesis is exact-tree tower FFN w1/w3
fusion on its shared input; current profiles establish neither its dominance
nor an expected large RTF gain. Sustained RTF 0.8–0.9 remains the open target.
