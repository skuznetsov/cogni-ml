# DRAMA-3 raw tensor views versus mapped Metal wrappers

Status: VERIFIED for the bounded Context-mediated diagnostic, executed
numerical/queue contract and descriptive measurements below. No performance
promotion is admitted. Base source: `c37f4d91`.

## Measured result

One excluded prime and all 18 warm requests completed. Six requests per arm
give these medians; brackets show the observed minimum and maximum, not a
confidence interval:

| Arm | Preparation | First driver scheduling, s | First GPU execution, ms | Request to result, s |
| --- | ---: | ---: | ---: | ---: |
| P: persistent | 2.542 us | 0.486 [0.391, 5.862] | 169.983 [165.788, 198.683] | 23.759 [23.327, 29.582] |
| V: views only | 3.917 us | 1.212 [0.393, 9.054] | 171.479 [167.951, 197.572] | 25.717 [23.419, 32.122] |
| R: fresh wrappers | 1.075104 ms | 2.570 [0.377, 6.375] | 167.568 [166.429, 192.860] | 26.996 [23.329, 30.086] |

Preparation measures the immediate reset; lazy view reconstruction occurs
inside generation. R's construction median is 1.066646 ms. First-command host
encoding medians are 12.699 / 11.232 / 10.835 ms for P/V/R; existing worker GC
medians are 23.940 / 34.903 / 27.062 ms. These host timers do not measure
deferred driver resource work. GPU execution and driver scheduling durations
use their respective observed clocks; no cross-family timestamps are
subtracted to manufacture a decomposition.

Within each consecutive triplet, subtract the requests with the indicated
treatments. Positive means the first named treatment took longer:

| Actual triplet | V minus P, s | R minus V, s | R minus P, s |
| --- | ---: | ---: | ---: |
| V R P | +8.718096 | -8.756479 | -0.038383 |
| P R V | +3.025639 | -3.024013 | +0.001626 |
| V P R | +0.928176 | -1.771856 | -0.843680 |
| R P V | -0.479758 | +5.615816 | +5.136058 |
| P V R | +0.445628 | +6.019616 | +6.465244 |
| R V P | -1.222895 | +0.172986 | -1.049909 |

R-minus-P has median -0.018379 s, range -1.049909 to +6.465244 s, three wins
each. R-minus-V also changes sign three times each; its median is -0.799435 s.
V-minus-P has median +0.686902 s, with four positive and two negative samples.
The large arm-median differences do not establish treatment effects in this
active-host, ordered experiment. All arms retain seconds-long first-driver
spread; immediate host construction alone cannot explain it. Persistent
wrappers remain the ordinary policy. Neither equivalence nor a stable speed
gain, causal penalty from view clearing, or internal registration reuse is
established.

All 18 warm first commands have exactly 5442 buffer bindings, including 862
raw-weight bindings. Worker-local shard counts are **100, 183, 198, 190, 191,
0**: all five model shards reach the first command; the codec shard does not.
The six mappings declare 25867452416 bytes, including 23993368576 bytes in the
five model shards. This is the bound resource extent, not bytes accessed or
physically resident. All warm jobs retain the same 566 converted GPU cache
entries and identities. The prime has a different histogram and is excluded.
The first V clears 1421 raw views; subsequent V/R jobs clear 855. P neither
clears nor creates views, V makes no new Metal buffers, and every R retires and
constructs six buffers with qualified identical backing.

Warm first-command process windows have zero page-ins and disk-read bytes in
every arm. Their system-CPU medians/ranges are P 0.483 [0.395, 5.727],
V 1.215 [0.394, 8.253], R 2.558 [0.379, 5.942] seconds. This supports retaining
driver/kernel work as the next investigation, without identifying its cause.
Whole-request disk reads reach 65536 bytes; page-ins remain zero. Counters are
process-wide, overlapping windows are nonadditive, and zero reads do not prove
page residency or exclude unobserved memory-management work. Unavailable or
reset cumulative counters stay null; valid gauge deltas may be negative.

## Controlled intervention

The prior [P/R experiment](drama3-weight-wrappers.md) combined raw TensorBuffer
cache clearing with Metal wrapper recreation. This diagnostic adds a sham arm
to separate those interventions. One sequential Engine runs an excluded
persistent prime, followed by all six permutations of three treatments:

- **P / persistent:** keep raw TensorBuffer views and mapped Metal buffers.
- **V / views:** clear/rebuild raw TensorBuffer views; retain the same live
  MetalBuffer objects and mmap owners.
- **R / fresh:** clear/rebuild views and release/recreate the six mapped Metal
  buffers over the same mmap owners, pointers, extents, modes and tensor offsets.

The seeded schedule uses `--design views --cycles 1 --seed 8421`, giving 18 warm
requests, six per arm. Each arm occupies each within-triplet position twice;
each ordered distinct neighboring pair occurs twice inside the six triplets.
Transitions across triplet boundaries are not perfectly balanced. Actual
predecessor counts, load averages and all individual measurements are retained;
the six within-triplet differences are descriptive, not independent replicates
or an equivalence test. No cache purge, remap or weight-byte copy is performed.

Converted GPU F32 weights remain cached. A host-only signature checks the same
cache keys, Buffer/MetalBuffer object identities, counts and offsets before and
after each warm job. It does not checksum GPU contents. View clearing requires
no Metal factory calls; R must retire/recreate every previously mapped wrapper
and qualify identical backing. Factory and view counters qualify the actual
intervention rather than relying on a mode label or a recycled native address.

The supported diagnostic is Context-mediated and single-worker only. Context
refuses preparation, capture start and publication with an active workspace,
encoder, command, lease, pending retained buffers, pending graph work or an
aborted state. It never drains to make a reset legal. Failure aborts the Context
and conservatively quarantines owners. The lower-level bank methods do not
independently enforce this boundary and are not a concurrent management API.

## Passive first-command exposure

At the existing idle job boundary, capture records the current Context submit
index. HostEncoder counts binding calls after the underlying `set_buffer`
returns, seals after the first existing submit succeeds, and publishes only
through Context after the existing terminal boundary. Later commands cannot
change the snapshot; abort discards publication. Lookups and view construction
are not binding events. The shard list includes all mappings known at terminal,
including zero-count mappings created after capture began.

Stable worker-local owner ordinals and live MetalBuffer object identities
associate calls with raw mapped shards. These are host API exposure counts,
not shader dereferences, physical residency, hardware resource IDs or evidence
of internal driver registration. Current Resident raw-binding paths were
source-audited through Context/HostEncoder. Legacy direct Metal paths, future
bypasses and alternate host aliases are outside this scoped certificate.

No shader, arithmetic order, precision, queue policy or inference read/wait
boundary changes. HostEncoder gains an optional probe check even when disabled;
its unobserved default cost is not claimed to be zero. All three measured arms
use the same observation path.

## Numerical and boundary guards

The focused GPU witnesses cover same-shard offsets and foreign owners; V keeps
the exact live MetalBuffer while producing distinct TensorBuffer objects; R
recreates over the same backing. Dense/packed outputs and graph/read ledgers
are exact at graph depths0/2. Binding witnesses reject lookup-as-exposure,
include an unused zero-count shard, exclude cached converted weights and later
commands, and reject unpublished/nonterminal/aborted publication. The relevant
suite passes **69 Metal examples**; the driver has **14 offline schedule and
qualification tests** including mutated receipts. The initial view witness
failed to compile against the old API before implementation; no hardware bind
or submit failure injection is claimed.

Separate V and R terminal-output gates each execute a 158-row prefill against
the pinned MLX fixture. Final hidden2048/logits4097 are bitwise equal between
native control and candidate, with zero nonfinite values and pointwise failures
against MLX at unchanged `atol=3e-5`, `rtol=3e-4`. Worst error/allowance ratios
are 0.05126616 and 0.0079710195. Each native arm has 159 existing submits/waits;
one terminal GPU metric command gives 319 total. Only 16 control and 40 reduced
diagnostic bytes are read. Activations remain on GPU. Candidate prefill also
uses the previously tested workspaces; this is a combined numerical guard,
not a timing treatment or comparison of every row/layer/teacher-force output.

All 19 requests produce the exact baseline 60x11 codes and identical WAV
SHA256 `38f2530ed1c90f0caba45e222432909ac5d92cc3633e4c34a21fbe39d693e07d`:
122880 samples at 44100 Hz, or 2.786395 seconds. The common 704-value F32
sampler prefix has SHA256
`e13a73aa1f41d156e7c2f811bb941c1b63993aa97989b267a14b65a5665a545b`.
Every request retains 122 submissions/completions/waits, depth2/max-pending2,
zero terminal pending/retained buffers, 496 control and 494160 terminal bytes.
All 2318 timeline command events have valid timing/resource observations.
There are no new inference submits, waits or activation readbacks.

One warm output from each arm also passes the saved pinned MLX GPU control:
main tokens, codebooks and all 60 complete frames are exact on the shared F32
sampler tape. This executes three comparisons to the saved control; it does
not rerun stock MLX sampling or generation. The same-code equality of all 19
native requests extends only that fixed workload's output check.

The known MoE39 full-intermediate failure remains open. Endpoint/code parity
does not establish general long-speech, voice, whisper, prosody or DAC parity.

## Reproduction and evidence scope

Build `examples/drama3_weight_wrappers.cr` in release mode with
`-Ddrama3_metal`, the ordinary Metal bridge and framework link flags. Under
`scripts/run_safe.sh`, run:

```sh
python3 scripts/drama3/measure_weight_wrappers.py \
  --binary /absolute/path/weight-views \
  --model /absolute/path/pinned-model \
  --output /absolute/path/new-run \
  --baseline /absolute/path/baseline-worker.json \
  --design views --cycles 1 --seed 8421
```

The default `wrappers` design retains the earlier two-worker P/R schedule.
Raw artifacts, builds and reference scratch stay under
`/opt/mldata_ssd/cogni-ml/drama3-weight-views-20261016/`; weights stay on the
local SSD. Runs are serial, with a 15% free-memory floor, quiet-host off and
unrelated processes untouched. All arms use MAP_PRIVATE, CogniGraph depth2,
dense tree reductions, expert64, generation/DAC scratch off, and common
host/timeline/resource observation. The host is Apple M2 Max, macOS 26.6.2
(25G83). These are active-host measurements, not stationary or isolated trials.

Checkpoint: `cguldogan/drama-3-mlx-q4` revision
`5331ab509359c0e4a7e353a19e255ddf3e8152a8`; MLX-audio revision
`e7488935ec2ebfafd38ef2b58e290d79f3d05dc7`. Config/index and fixture hashes,
indexed file set, file sizes and page-rounded extents are checked. Whole weight
shards were not rehashed. The reused bridge object has SHA256
`4e76b260561c8f79442c273397148720ff67ce1b31ff87029c4034ee54898d49`;
its sources are unchanged. No fresh bridge or reproducible-build claim follows.

`commands.json` records guarded commands, source/input/log hashes and statuses;
`pairs/views-worker.json` retains the 19 requests, and `analysis.json` retains
all observations, actual predecessor counts and per-triplet differences.
Every row pins seven source files. The outer execution receipt separately pins
six; `measure_host_pairs.py` is manifest-only and was checked against live
source. `static_checks.json` binds the passing gates and unchanged measured
sources. The first offline analysis failed on an external-SSD symlink path
alias before producing results; that failed receipt remains visible and
`analyze-views-v2` passes with the alias preserved. No GPU run was repeated or
excluded to repair this analysis error.

The curated index is `.landmark/evidence/drama3-weight-views.json`. Correlated
mini-Quad source and measurement reviews, root inspection, executed graph0/2
witnesses and mutated-receipt rejection support a ROBUST verdict only within
the stated diagnostic and numerical scope. They are not independent
replication. Default observer overhead, native bind/submit fault injection,
concurrent bank use, GPU cache contents and hidden driver behavior are outside
that verdict.

The next candidate is a separately gated comparison of whole-shard resources
with narrower page-aligned no-copy buffers. The current native encoder passes
the Metal buffer and offset; its `length` parameter is GraphEncoder API
compatibility metadata, so changing that value alone does not change the
Metal resource extent. Binding exposure does not prove that Metal validates
the entire resource. An earlier per-tensor-view probe failed to establish an
end-to-end gain; a new contrast must address its cache/order confounds using
the now-qualified view sham and measured exposure, while preserving mmap
ownership, alignment/range guards, identical codes and the same queue ledger.
This was a hypothesis for the follow-up slice; the completed
[mapped resource extent contrast](drama3-weight-extents.md) retains the view
sham and finds no uniform layout gain. It keeps whole-shard resources as the
ordinary default and does not infer driver internals from binding exposure.

Rollback: omit diagnostic enabling or revert this isolated slice. Source,
model/layout/input/tape/observer/queue/device/OS drift, contradictions or raw
artifact eviction require refreshing dependent claims. Structural evidence
checks index execution and output artifacts; they do not establish execution,
causality or performance by themselves.
