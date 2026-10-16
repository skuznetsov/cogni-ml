# DRAMA-3 mapped weight resource extents

Base source: `308338117ed75f2a50005efd38320aef922deff4`.
The claim is limited to the Context-mediated diagnostic, fixed-workload
numerical/queue guards and descriptive resource-layout experiment below.
Ordinary inference retains persistent whole-shard resources.

## Intervention and falsifier

The previous [views/wrappers experiment](drama3-weight-views.md) did not
establish a stable gain. An older per-tensor-view experiment also failed to establish
a gain across separate processes. This test reopens the latter route with one
Engine, retained converted GPU caches, an excluded prime for every layout
block, and a view-cache sham in each layout.

- S / `shard`: one page-rounded no-copy Metal buffer per used mmap file.
- N / `tensor`: page-rounded no-copy buffers covering each complete raw Tensor.
  Equal page envelopes share a resource. Partially overlapping read-only
  envelopes remain separate; their union is reported separately from their sum.
- P / `persistent`: keep raw views and Metal resources within the layout.
- V / `views`: clear/rebuild raw TensorBuffer objects while keeping the same
  Metal resources and converted GPU cache.

One sequential Engine runs S/N/N/S blocks, each with an excluded persistent
prime followed by P/V/V/P. There are four primes and sixteen scored requests,
four observations per layout/treatment. Actual predecessors cross block
boundaries. The two opposite-time-direction layout contrasts use block1
versus block0 and block2 versus block3, separately for P and V. These are
descriptive repeated measurements, not independent samples or an equivalence
test. A gain confined to one treatment or one time direction cannot promote N.

Resource extent and resource count change together. Any observed difference
would be a net layout effect, not an isolated byte-length effect or proof of
internal driver registration behavior. All arms use the same observer, but
resource-list serialization and observation overhead can depend on the layout.

## Ownership, range and capture contract

The opt-in bank is limited to a single Context. Preparation requires its
existing terminal boundary: no active encoder, command, workspace, lease,
retained pending buffers, graph work or aborted state. It never drains to make
a transition legal. All previous Metal handles are released before any new
handle is constructed. The same immutable mmap owners, pointers and sizes
remain retained; every known raw view is rebuilt outside scored generation.
Failure aborts the Context and conservatively quarantines owners.

`MappedBuffer` accepts a checked file interval and rounds its boundaries to
pages, with checked arithmetic and the device buffer-length limit. This follows
the alignment and lifetime requirements of Apple's
[no-copy Metal buffer API](https://developer.apple.com/documentation/metal/mtldevice/makebuffer%28bytesnocopy%3Alength%3Aoptions%3Adeallocator%3A%29?language=objc).
TensorBuffer subtracts the view's file offset from the original absolute byte
offset and checks that the complete raw interval fits. GPU-selected experts
therefore retain the entire rank-3 expert bank; this diagnostic does not choose
an expert on CPU or construct a buffer for just that expert.

Passive first-command capture retains per-resource file offsets, lengths and
binding counts, plus per-mmap totals. A binding is recorded only after the
underlying host `set_buffer` returns. Publication requires the existing first
submit and terminal boundary. A resource replacement invalidates any previous
receipt, preventing old command counts from being published against new
resources. Same-layout view clearing preserves it because resource identities
remain unchanged. These records describe host API exposure, not shader reads,
physical residency, hardware IDs or actual driver registration work.

Direct bank calls do not enforce the Context guard themselves and are not a
concurrent management API. The ordinary global cache remains the default.
No shader, float operation/order, precision, sampler policy or inference
submit/wait/read boundary changes. No writable-resource alias generalization
is admitted.

## Numerical and lifecycle verification

The genuine pre-implementation API witness failed because the range constructor
was absent. A second genuine RED exposed stale receipt publication after
resource replacement. Corrected source passes 76 relevant Metal examples.
New witnesses cover page boundaries, rounded tails, out-of-range and foreign
owners, exact-envelope deduplication, overlap unions, old-handle invalidation,
rebased dense/packed weights and GPU-selected experts, multi-page scales and
biases, cold GPU conversion at nonzero offsets, cache identity and graph/read
ledgers. Numerical loops and active/pending/workspace/aborted rejection run
at graph depths 0 and 2. Same-layout view clears preserve receipts; layout
changes and legacy refresh reject stale receipts.

The new driver passes seven offline schedule/lifetime/resource mutation tests;
the existing wrapper driver passes fourteen. Offline checks reject wrong
predecessors, layout/treatment labels, reported lazy resource creation,
inconsistent backing/cache receipts, misaligned offsets, incorrect interval unions or inconsistent binding
totals. These checks qualify their executed properties, not driver internals.

All twenty requests, including the four excluded primes, reproduce the fixed
baseline's full 60x11 codes and WAV SHA256
`38f2530ed1c90f0caba45e222432909ac5d92cc3633e4c34a21fbe39d693e07d`.
The output is 122880 samples at 44100 Hz, or 2.786395 seconds. Each request
retains 122 submissions/completions/waits, depth 2/max-pending 2, terminal
pending/retained zero, 496 control and 494160 terminal readback bytes. The
same 704 F32 sampler inputs have prefix SHA256
`e13a73aa1f41d156e7c2f811bb941c1b63993aa97989b267a14b65a5665a545b`.
There are no activation CPU reads or added inference synchronizations.

Four representative S/P, S/V, N/P and N/V full-code comparisons against the
pinned saved MLX F32 control pass exact GPU integer equality for main tokens,
codebooks and all tokens. This is not a new MLX generation or stock-RNG test.
The final-source 158-row prefill guard also passes: final hidden (2048) and
main logits (4097) are bitwise identical between native arms and have zero
pointwise/nonfinite failures against saved MLX at atol 3e-5/rtol 3e-4. Arm 2
also enables existing backbone/tower-prefill workspaces, so this is a combined
numerical guard, not an extent-only timing or every-layer/intermediate gate.
Both arms retain 159 existing submissions/waits; a single terminal GPU metric
command brings the combined count to 319, followed by 16 control/40 diagnostic
bytes. Reading the reduced metrics adds no submit or wait.

## Observed resource exposure

The passive histogram is invariant across all nineteen warm requests (nine S,
ten N), including the three warm primes. The excluded cold S prime has
6067 total/1174 raw bindings while discovering converted weights. Both warm
layouts issue 5442 total bindings and 862 raw-weight bindings in the first
command. Per-mmap raw counts remain 100/183/198/190/191 for the five model
shards and 0 for the codec; these ordinals are worker-local owner identities.
All 566 converted GPU cache entries keep their host identities after the cold
prime. This is not a GPU-content checksum.

| Extent accounting (bytes) | S: shard | N: tensor |
| --- | ---: | ---: |
| All resource objects | 6 | 1310 |
| All resource lengths, summed | 25867452416 | 24752242688 |
| All intervals, per-mmap union | 25867452416 | 24732073984 |
| Overlap counted by the sum | 0 | 20168704 |
| Resources bound by the first command | 5 | 853 |
| Bound resource lengths, summed | 23993368576 | 23987830784 |
| Bound intervals, per-mmap union | 23993368576 | 23974952960 |

N preserves complete GPU-selected expert banks. Its bound sum falls only
5537792 bytes (0.0231%) and its bound union only 18415616 bytes (0.0768%).
Most of the all-resource interval-union reduction is outside first-command exposure,
including unused codec extents. The diagnostic therefore does not produce a
large reduction in declared active-weight extent: approximately 24 GB remains
bound, with many more resource objects. None of these byte totals is physical
memory use, residency, actual bytes read or driver work.

## Measured latency and decision

Each entry below is the median of two scored requests in that block, in
seconds. P/V/V/P follows the excluded prime. The three warm primes are kept
out of these statistics; the initial cold prime is also excluded.

| Block | Layout | P request-to-WAV | V request-to-WAV |
| --- | --- | ---: | ---: |
| 0 | S | 28.302495 | 23.448035 |
| 1 | N | 24.824810 | 32.941053 |
| 2 | N | 26.395670 | 28.731601 |
| 3 | S | 31.754992 | 26.501395 |

N-minus-S differences are -3.477684/-5.359322 seconds for P in the two time
directions, but +9.493018/+2.230206 seconds for V. Thus P favors N and V
favors S in both directions. A uniform layout gain is not established;
claiming that N can never help would also exceed this evidence. Ordinary
inference keeps persistent whole-shard resources. No default is promoted.

The following pooled medians/ranges each contain four scored requests and
are descriptive, not independent-sample estimates:

| Metric | S/P | S/V | N/P | N/V |
| --- | ---: | ---: | ---: | ---: |
| Request-to-WAV median (s) | 28.999397 | 24.970622 | 25.792889 | 29.434425 |
| First encoding median (ms) | 14.3 | 9.3 | 10.2 | 16.9 |
| First driver interval range (s) | 3.089–9.421 | 0.316–0.520 | 0.395–0.612 | 0.598–8.749 |
| First GPU execution range (s) | 0.169–0.172 | 0.160–0.198 | 0.165–0.197 | 0.168–0.198 |
| Summed GPU-duration median (s) | 23.085761 | 22.890718 | 23.803869 | 23.759901 |
| Preparation median (us) | 6.542 | 7.500 | 115.709 | 151.250 |
| Explicit post-job GC median (ms) | 33.574 | 32.201 | 42.603 | 28.382 |

The first driver interval is kernelEnd-kernelStart scheduling duration, not
GPU execution. First encoding uses one host clock family; no driver/GPU
cross-clock subtraction is used. GPU-duration sums do not partition wall time
or individual shader families. Common observation can have layout-dependent
overhead, including larger resource lists. Scored requests construct/retire
zero mapped Metal buffers; V rebuilds raw views lazily as intended.

S-to-N transition preparation takes 24.866833 ms, including 4.502250 ms retiring
six wrappers and 20.199458 ms constructing 1310. N-to-S takes 6.994750 ms,
including 5.392625 ms retirement and 1.542291 ms construction of six wrappers.
All old handles are invalid before replacements are made. The same-layout N
prime constructs nothing; its preparation takes 0.136334 ms. Transition work is
outside scored generation, and the full prime request is separately reported.

For request_begin-to-wav_end process snapshots, median user+system CPU seconds
are 7.94/3.73/3.79/7.16 for S/P, S/V, N/P, N/V. Page-ins are zero in all scored
requests; request disk-read deltas span 0–64 KiB. First-command counters likewise
do not prove residency. Process CPU outliers co-occur with long driver/wall
intervals in S/P and N/V, while first GPU execution stays near 0.16–0.20 s.
Host/driver variability and carryover remain strong competing explanations;
contiguous blocks and only two observations per mode/block prevent a causal
extent-only attribution. Quiet-host was off and unrelated processes were left
running. Process windows overlap
and must not be summed as disjoint command costs.

The performance falsifier is met: layout direction does not survive the view
sham. Repeating this route would require a new discriminator, such as a
mirrored block schedule with qualified host-CPU/scheduling variation. Given
the nearly unchanged bound-weight volume, this extent route is deprioritized.
Nomic's cross-token MoE batching is a separate hypothesis: current DRAMA
prompt prefill and decode execute one backbone/MoE row at a time, and DRAMA
already performs GPU routing plus selected-rank batching. A transfer first
requires a real multi-token backbone path; this slice makes no such change.
See `generator_resident.cr`'s prompt-row loops and `backbone_resident.cr`'s
single-vector contract, alongside Nomic's route/gather/indirect-GEMM/scatter
path in `src/ml/gguf/metal_backend.cr`. Nomic commit `23f775a5` records
8.807 ms/text versus prior scalar approximately 12.422 ms/text for its depth-12
embedding workload. This historical record was inspected, not rerun, and is
not a DRAMA speed prediction. Nomic's different quantization/arithmetic is
not admitted as a precision-preserving kernel replacement.

## Reproduction and evidence scope

Build `examples/drama3_weight_wrappers.cr` in release mode with
`-Ddrama3_metal`, the ordinary Metal bridge and framework link flags. Then,
under `scripts/run_safe.sh`, run:

```sh
python3 -B scripts/drama3/measure_weight_extents.py \
  --binary /absolute/path/weight-extents-final \
  --model /absolute/path/pinned-model \
  --output /absolute/path/new-run \
  --baseline /absolute/path/baseline-worker.json
```

The driver also validates a mirrored `--layout-order tensor-shard-shard-tensor`
schedule offline; the recorded experiment uses the default S/N/N/S schedule.
Runs are serial with a 15% free-memory floor and quiet-host off. Unrelated
processes remain running. Checkpoint weights stay on the local SSD; temporary
builds, raw measurements and reference scratch stay under
`/opt/mldata_ssd/cogni-ml/drama3-weight-extents-20261016/`.
The host is Apple M2 Max, macOS 26.6.2 (25G83). All arms use MAP_PRIVATE,
CogniGraph depth 2, expert 64, both dense tree reductions, generation/DAC scratch
off, and common host/timeline/resource observation. This is an active-host
measurement with order/carryover limits.

Checkpoint: `cguldogan/drama-3-mlx-q4` revision
`5331ab509359c0e4a7e353a19e255ddf3e8152a8`; MLX-audio revision
`e7488935ec2ebfafd38ef2b58e290d79f3d05dc7`. Config/index/tokenizer hashes,
indexed file set and file sizes qualify the model. Full weight shards were
not rehashed. The unchanged bridge object was reused with SHA256
`4e76b260561c8f79442c273397148720ff67ce1b31ff87029c4034ee54898d49`;
no fresh-bridge or reproducible-build attestation follows.

`commands.json` retains exact guarded commands, configured controls, source,
binary/input/log hashes, statuses and before/after memory snapshots. The raw
matrix manifest and stderr retain all twenty requests and 2440 command events.
`analysis.json` is produced by `analyze-extents-v3`; `static_checks.json`
cross-checks twelve required exit-zero command receipts, live measured sources,
log hashes, the fixed inputs and final numerical gates. The ten worker source
pins are checked against live bytes; the imported older wrapper/host driver
helpers are manifest-only in the matrix's outer receipt, a disclosed coverage
limit. The imported timeline helper is retrospectively hashed and matches the
unchanged base source; it was not pinned by an execution receipt. Metadata and
packet structure do not prove model execution or causality.

Two failed offline analysis receipts are retained: v1 assumed a single baseline
row instead of selecting the established first row of the fixed three-row
baseline; v2 incorrectly required the cold discovery histogram to equal warm
histograms. V3 checks baseline-row invariants, preserves the excluded cold
histogram and qualifies all nineteen subsequent histograms. These were analysis
scope/format corrections; no matrix request was rerun, dropped or relabeled.
Initial invalid spec syntax, a constructor compile error and a mistyped suite
path are also retained separately from the genuine API/stale-receipt REDs.

The curated index is `.landmark/evidence/drama3-weight-extents.json`.
Correlated mini-Quad source and raw-result reviews, root cross-checks,
executed graph0/2 witnesses and negative receipt cases support a bounded
ROBUST verdict for the diagnostic, fixed numerical/queue contract and
descriptive observations. They do not provide independent replication,
concurrent bank safety, writable aliases, fault injection at native bind/submit,
GPU cache checksums, default observer overhead or internal-driver attribution.

Rollback: omit diagnostic enabling or revert this isolated slice. Refresh
dependent claims on source, model/layout/input/tape/fixture/tolerance/queue,
observer, device/OS or storage drift, contradictory evidence or raw artifact
eviction. Known MoE39 full-intermediate and DAC post-module numerical
frontiers remain open; endpoint/code parity does not establish general voice,
prosody, whisper or long-speech parity.
