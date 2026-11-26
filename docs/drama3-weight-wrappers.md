# DRAMA-3 persistent versus recreated mapped weight buffers

Status: VERIFIED for the bounded diagnostic and numerical contract below.
The executed experiment demonstrates wrapper recreation over
unchanged backing pages and exact fixed-workload outputs. It does not demonstrate
an end-to-end speed improvement or identify internal Metal registration work.

## Result

Two workers each execute one unscored persistent cold prime and eight warm jobs.
The warm order is `P/R/R/P` twice in worker 0 and `R/P/P/R` twice in worker 1,
where P keeps the bank's Metal buffers and R releases all its old handles before
creating replacements. This gives eight adjacent P/R pairs, not 16 independent
replicates. Both workers are sequential; other projects remain active.

| Warm measurement | Persistent, n=8 | Recreated, n=8 |
| --- | ---: | ---: |
| Wrapper preparation, median | 2.646 us | 1.087167 ms |
| Summed six-wrapper construction, median | 0 | 1.078605 ms |
| First command encoding before commit, median | 10.495 ms | 10.082 ms |
| First command driver scheduling, median | 0.375094 s | 0.444442 s |
| First command driver scheduling, range | 0.311427–7.685558 s | 0.312435–7.228574 s |
| First command GPU execution, median | 0.164900 s | 0.171223 s |
| Request to result, including worker GC, median | 23.761333 s | 24.620183 s |
| Request to result, range | 23.151490–30.787480 s | 23.033646–30.391518 s |

The adjacent paired R-minus-P request difference has median **-0.000816 s**, mean
-0.108522 s and range **-5.439325 to +6.644850 s**; four pairs favor each arm.
The pooled medians and paired median answer different descriptive questions.
Neither supports a speed promotion on this small, noisy host sample. The paired
first-driver difference likewise ranges from -5.400462 to +6.861156 s.

| Worker | Adjacent warm jobs | R minus P request seconds | R minus P first-driver seconds |
| --- | --- | ---: | ---: |
| 0 | 1 / 2 | -4.353628 | -5.052899 |
| 0 | 3 / 4 | -2.004747 | -2.941574 |
| 0 | 5 / 6 | +0.116213 | +0.036931 |
| 0 | 7 / 8 | -5.439325 | -5.400462 |
| 1 | 1 / 2 | +4.036963 | +4.648318 |
| 1 | 3 / 4 | +6.644850 | +6.861156 |
| 1 | 5 / 6 | +0.249338 | -0.025214 |
| 1 | 7 / 8 | -0.117844 | +0.014871 |

Fresh preparation ranges from 0.776125 to 1.664792 ms. Its measured remainder
outside the release and creation timers is 5.126–11.208 us, including cache/hash
bookkeeping and pointer qualification. The tiny release timer is resolution
limited and does not measure delayed driver teardown. Fresh jobs also rebuild
raw tensor-view objects lazily during encoding: the first refresh clears 1421
views in each worker, later refreshes clear 855. Thus preparation alone is not
the complete CPU cost of the intervention. Observed post-job GC medians are
25.305 ms / 27.112 ms for P/R; the paired GC difference ranges from -4.173 to
+23.864 ms. No zero-overhead or isolated GC-causality claim follows.

Direct host wrapper creation at this scale cannot explain the multi-second
first-command spread by itself. Persistent handles also retain that spread.
This does **not** rule out lazy driver registration, per-submission validation,
page-table work, memory pressure, or interference below the existing Metal
submission trap. Internal registration reuse is unobserved. Keep the ordinary
persistent cache; do not promote a wrapper lifetime optimization.

## Controlled intervention and lifetime guards

The diagnostic is explicitly enabled on a new `Resident::Context`; it has no
environment flag or ordinary Studio UI path. Its bank supplies every raw
Resident binding, including quantized weights/scales/biases, dense matrices,
embeddings and expert paths. One `MTLBuffer` wraps each whole mmap shard using
`newBufferWithBytesNoCopy`, with tensor offsets into that buffer. Files, mmap
owners, addresses, page-rounded extents, storage modes and offsets stay fixed
inside each worker. There is no remap, file-cache purge or weight-byte copy.

The live checkpoint's five indexed model shards plus `codec.safetensors` give
six mappings and **25867452416 declared buffer bytes**, rounded to 16384-byte
pages. This is a declared extent, not a measurement of physical residency.
Config/index digests and the indexed shard set are pinned; whole weight-shard
contents were not rehashed in this experiment. Same-worker pointer/owner
checks establish backing identity for the intervention.

| File | File bytes | Declared rounded buffer bytes |
| --- | ---: | ---: |
| model-00001-of-00005.safetensors | 5498947446 | 5498961920 |
| model-00002-of-00005.safetensors | 5390036166 | 5390041088 |
| model-00003-of-00005.safetensors | 5402946528 | 5402951680 |
| model-00004-of-00005.safetensors | 5423590660 | 5423595520 |
| model-00005-of-00005.safetensors | 2277811252 | 2277818368 |
| codec.safetensors | 1874070622 | 1874083840 |

Already converted GPU F32 weights remain cached in the same Context. Rotation
recreates all six previously used raw shard wrappers; it does not rebind all six
in the first prefill or recreate the converted GPU allocations. Conversion-only
raw tensor views need not be requested again on warm jobs. The contrast is raw
mapped-wrapper lifetime with a fixed converted-weight cache, not the lifetime
of every GPU weight resource.

Refresh requires an already terminal queue: no active encoder/command/lease,
workspace, pending Context buffers or pending CogniGraph command. The check
rejects nonterminal and aborted states without draining or waiting. All old
Metal handles are released before any new one is created. Old Crystal wrapper
objects and mmap owners stay alive through pointer/extent/mode qualification;
old wrappers must be invalid and replacements valid. A failure aborts the
Context; its bank is conservatively quarantined with retained owners rather
than making a failed drain a lifetime certificate.

Objective-C handle addresses may be recycled. The checks certify actual
no-copy factory calls, host wrapper replacement and backing identity, not a
new driver registration identifier. No handle-address reuse occurred in the
eight measured refreshes, but that observation is not required for correctness.
Fresh-only `contents` calls retrieve storage addresses without dereferencing
weight bytes. See Apple's [MTLBuffer contents contract](https://developer.apple.com/documentation/metal/mtlbuffer/contents%28%29)
and [no-copy buffer factory](https://developer.apple.com/documentation/metal/mtldevice/makebuffer(bytesnocopy:length:options:deallocator:)).

Normal inference still selects the original global raw-weight cache. The
shared binding helper adds an optional-bank branch; this experiment does not
measure a before/after overhead of that branch. No shader, arithmetic order,
precision, queue policy or inference read/wait boundary changes.

## Paging, scheduling and observation limits

All 16 warm jobs record zero process page-ins and zero process disk-read bytes
in the first command's encoding-begin to after-wait window. First-command
system CPU is 0.314067–7.165735 s across warm jobs. Wrapper preparation records
zero page-ins and disk-read bytes. Whole-request disk-read accounting is not
always zero: warm jobs include 0–204800 bytes outside that first window.

The two new-worker primes have first-driver intervals 12.928803 / 11.433072 s,
about 23.995 GB of disk-read accounting and 1464462 / 1464466 page-ins in the
first-command windows. Both primes use P; they do not compare modes. The first
warm job is a scheduling outlier in both workers under opposite modes (P
5.386872 s / R 4.976785 s), another reason to retain order/worker alternatives.
Primes are separate observations, not scored warm arms.
Cold means new worker/mappings, not purged filesystem caches or physical-disk
reads. Process counters cover all threads and overlapping command windows are
nonadditive. Zero page-ins cannot prove GPU residency. Metal pair-local
scheduling and GPU durations do not prove a pure queue-wait decomposition.

Host: Apple M2 Max, macOS 26.6.2 (25G83). Weights remain on the local SSD;
run/build/reference scratch is under `/opt/mldata_ssd/cogni-ml`. Both arms use
host/timeline/resource observation, dense `tree`, expert64, MAP_PRIVATE,
CogniGraph depth2, generation/DAC scratch off, a 15% `run_safe` free-memory
floor and quiet-host off. Unrelated processes are untouched. These controls
avoid an intentional per-arm residency change but cannot exclude external
activity or certify a stationary GPU/driver state.

## Numerical and synchronization checks

The fixed text is `Hello, this is a voice test.`, no reference, seed1,
max_frames64, temperature0.8, top_k30, top_p0.9. The uploaded 704-value F32
sampler tape prefix SHA256 is
`e13a73aa1f41d156e7c2f811bb941c1b63993aa97989b267a14b65a5665a545b`.
Every request preserves all 60 frames × 11 IDs, sampler inputs, WAV SHA256
`38f2530ed1c90f0caba45e222432909ac5d92cc3633e4c34a21fbe39d693e07d`,
122880 samples at 44100 Hz (2.786395 s), and the existing graph/read ledger:
122 submissions, completions and completion waits, depth2/max_pending2,
zero terminal pending/retained bytes, 496 control and 494160 terminal bytes.
All 2196 existing command events have valid scheduling/GPU timing and resource
snapshots. Rotation adds no inference submission, synchronization or activation
readback; input/control/terminal transfers retain their existing scope.

- The focused witnesses cover shared-shard tensor offsets, foreign-owner
  rejection, packed expert bindings, cached GPU F32 conversions, exact GPU
  outputs and graph/read ledgers at depths0/2, and refusal at nonterminal,
  workspace, late-enable and aborted boundaries. The final relevant Metal
  suite passes **67 examples, 0 failures/errors**.
- One P and one R output both pass an executed MLX-GPU exact comparison of all
  60×11 IDs against the saved pinned F32 MLX-core control with the same F32
  sampler inputs. This is saved-control parity, not fresh stock MLX generation
  or stock RNG equivalence.
- The executed terminal-output comparison after a 158-row prefill rotates five model-shard
  wrappers between two disjoint native prefill states sharing immutable
  converted weights. Hidden2048 and logits4097 are bitwise equal between native
  arms; against the pinned MLX fixture they have zero pointwise failures and
  worst error/allowance ratios 0.05126616 / 0.0079710195 at unchanged
  `atol=3e-5`, `rtol=3e-4`. The second arm also uses the previously tested
  backbone/tower workspaces; this is a combined numerical guard, not a wrapper
  performance measurement. Only the final hidden/logits endpoints are compared,
  not every row/layer or the discarded teacher-force outputs. Each arm retains
  159 submissions/waits; one terminal
  GPU comparison command brings the total to 319. Only the existing 16 control
  and 40 reduced diagnostic bytes are read; activation tensors stay on GPU.

Checkpoint: `cguldogan/drama-3-mlx-q4` revision
`5331ab509359c0e4a7e353a19e255ddf3e8152a8`; MLX-audio revision
`e7488935ec2ebfafd38ef2b58e290d79f3d05dc7`. The known MoE39 full-intermediate
failure remains open. Exact selected outputs do not close it or establish
voice, DAC, prosody, whisper or long-speech parity.

## Reproduction, provenance and next discriminator

Base source: `99a4f447`. The ordinary cache remains the default. Rollback is to
omit diagnostic enabling or revert the isolated diagnostic commit. Refresh is
intended only for the dedicated sequential worker; it is not a concurrent
bank-management API.

The dedicated release example is `examples/drama3_weight_wrappers.cr`.
Build with `-Ddrama3_metal` and the ordinary Metal bridge/framework link flags,
then run under `scripts/run_safe.sh`:

```sh
python3 scripts/drama3/measure_weight_wrappers.py \
  --binary /absolute/path/weight-wrappers \
  --model /absolute/path/pinned-model \
  --output /absolute/path/new-run \
  --baseline /absolute/path/baseline-worker.json --cycles 2
```

Raw local run root:
`/opt/mldata_ssd/cogni-ml/drama3-weight-wrappers-20261016/`.
`commands.json` records exact guarded commands, effective harness controls,
source/input/log hashes and exit statuses; `pairs/wrapper-worker.json` retains
all requests; `analysis.json` and `analyze.py` qualify exact executed order,
model metadata/extents, baseline hash, full outputs and timing windows. The
reused `bridge.o` has SHA256
`4e76b260561c8f79442c273397148720ff67ce1b31ff87029c4034ee54898d49`;
its sources are unchanged. This is not a fresh native-bridge build or a
reproducible-build attestation. Large artifacts stay outside Git.

Source/control adversarial review is correlated same-lineage review; measured
receipts and executed checks supply the evidence. The bounded verdict is ROBUST
for the dedicated-worker lifetime guards, executed numerical contract and
descriptive comparison. The initial exact-run objection to the generic driver's
unconstrained shard/cycle counts is resolved by the executed artifact qualifier,
which checks six mappings, two cycles, receipts and pinned baseline. This does
not turn the generic driver into a hardcoded six-shard validator or establish
pure buffer-identity causality. The curated evidence index is
`.landmark/evidence/drama3-weight-wrappers.json`; `static_checks.json` binds the
passing execution receipts, output checks and source hashes. Its structural
checks alone do not prove execution or causality.
A new model, backing layout, wrapper/queue implementation, host/OS,
or observer setting requires refreshing these scoped measurements. Missing
artifacts or failed gates downgrade the dependent claim.

The next useful discriminator is a view-only sham arm: clear/rebuild raw
`TensorBuffer` views while retaining the same mapped Metal buffers, and compare
it with persistence and full refresh in balanced order on one Engine. Passive
per-shard binding counts can establish which refreshed resources actually reach
the first command. This would separate view-cache churn from release/recreation
without activation reads or extra waits. Internal driver registration would
still be unobserved. A later whole-shard versus narrower declared-buffer
comparison could test validation granularity, but requires its own page
alignment, offset, ownership and numerical witnesses. These are hypotheses,
not promoted optimizations; wrapper caching already exists and did not remove
the observed driver spread.

Follow-up: [Raw weight views and binding exposure](drama3-weight-views.md)
executes the view-only sham on one Engine with all six P/V/R permutations and
passive first-command counts. All three modes retain seconds-long spread;
the follow-up establishes no stable speed gain or default promotion.
