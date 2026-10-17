# DRAMA-3 native text-to-WAV frontier

Status: native CPU text-to-WAV and opt-in resident Metal text/reference-to-WAV
observed on the pinned Q4 checkpoint. Resident inference keeps model computations,
activations, logits, routing, code selection, temporal state, and DAC on GPU.
The exact control/terminal read boundaries and measured scope are below.
Goal: one short, single-speaker text prompt to a mono 44.1 kHz WAV, with all
inference computations in Crystal/native libraries. Python/MLX/PyTorch are
reference and fixture tools only, never the native inference implementation.
The pinned MLX upstream is a community port. Original Fish DAC controls below
cover four saved code streams, not the original text-generation pipeline.

## Active slice: tower FFN paired projections — 2026-10-17

At `2149cd03`, tower FFN w1/w3 use two dense matvec dispatches over the same
resident normalized input. The pinned BF16 matrices are 6144 by 2048 in each
of thirteen tower layers. Hypothesis: one two-output dispatch can reduce
encoding/binding cost while preserving the original 64-lane F32 products and
tree. This family is not established as the current dominant request cost.

CAUTION: changed compiler reduction, mixed row/offset selection, unsafe output
reuse, or a local win without WAV benefit. Keep two output allocations in the
same ordinal order and retain the original inline calls as default/rollback.
No new production GPU allocation, submission, wait or read is admitted. This
is ordinary dispatch fusion. Unsupported formats/reductions must fall back.

First falsifier: alternating paired separate/fused production calls over
captured tower normalized inputs, GPU exact-bit/finite receipts, rejecting
bit/NaN controls and existing command timestamps. Then compare w1/w3, SiLU,
product, w2 and layer outputs against pinned MLX with unchanged tolerances,
and run offset/tail/format/fallback/frozen-selector guards. The legacy CPU
activation-reading tower trace is not the qualification instrument.

DoD: release Metal specs; captured-input local pairs; device-only native-arm
and upstream intermediate comparisons; complete profiler-off WAV ABBA/BAAB with
identical full codes, sampler tape, WAV, allocations, reads and waits. Enable
by default only after full-request benefit is confirmed without regressions;
otherwise retain an off-by-default experiment or reject the candidate.
Heavy work stays serial under run_safe with the authorized 15% memory floor,
external scratch, local weights and unrelated processes untouched. Existing
strict backbone/DAC upstream failures remain open and cannot be relabeled.

The first combined-row implementation preserves 2,654,208 local values exactly
and passes the six captured MLX projection controls at the old tolerances.
Its local paired command median is only 1.04135x (17/21 wins), with one case
at 0.99875x. Full-request ABBA gives 1.21078/1.05162 mirrored ratios, but reverse
BAAB gives 1.03778/0.92573; unchanged DAC also varies. No reproducible complete
request benefit is established. Default remains off.

The second implementation computes both original trees for equal-row,
aligned BF16 pairs in one threadgroup, sharing each input load and using
exact ushort expansion. It guards both mapped ranges/alignment and launches
only the common row count; other supported dense pairs retain the combined
row domain. Twenty release Metal examples pass, including specialized and
fallback branch receipts, exact-end weights, input/output views, cancellation,
and BF16 zero/subnormal/Inf/NaN bit controls. All 2,654,208 local values remain
exact; all six MLX projection controls pass the old tolerances. The pooled
local command ratio is 1.16337x (21/21 wins), with case medians
1.20037x/1.15581x/1.14107x. This is not isolated kernel timing or WAV benefit.

The 87-row tower replay compares 674 traced states at prompt row85 and first
decode row0, plus the decode frame, exactly. At each of the 675 sites,
off/on upstream metrics match; the complete V1/V2 comparison arrays also
match. Three late residual8 baseline comparisons exceed the unchanged MLX
tolerances: decode w2, decode FFN
residual, and prefill row85 FFN residual. No-regression is established for
this capture; strict upstream compatibility remains false. Q/K norm buffers
are snapshotted on GPU before in-place RoPE only in the diagnostic observer.
Normal requests have no observer or added copy/wait/read.

Complete profiler-off V2 ABBA gives warm worker median request times
23.175658/24.907944s (separate/fused), with mirrored ratios0.904047/0.960349.
Reverse BAAB repeat gives26.576364/28.817057s, ratios0.787350/1.078039.
All24 qualifying requests preserve full codes, sampler tape, WAV, allocations,
reads and waits. No reproducible WAV gain is established; keep default off.
These are active-host measurements for2.786395s audio, not an isolated causal
kernel regression or real-time performance.

The first reverse attempt is excluded: after9 exact results, an unfused
worker aborted during exit in the GCD watchdog's global mutex lock. All
request graphs were drained; bridge source is unchanged. Static-destruction
race is an unproven hypothesis, and the clean repeat does not fix that risk.
Next discriminators: full decode GPU attribution before choosing another
kernel family, and a separate minimal watchdog/exit reproducer. See
[tower pair qualification](drama3-tower-pair-fusion.md) and
`.landmark/evidence/drama3-tower-pair-fusion.json` for scope and lineage.

## Prior slice: precision-preserving decode acceleration — 2026-10-16

Target sustained generation below real time, preferably RTF 0.8–0.9, with
first-audio latency measured separately. A 2048-sample frame at 44.1 kHz gives
46.440 ms of audio and a 37.152–41.796 ms compute budget at that target.
Earlier saved timeline controls have about 196 ms GPU decode per frame plus
separate DAC work; they are a bounded baseline, not a current steady-state
guarantee. Prefill batching alone does not close that gap.

First discriminator: paired native affine expert projections with the same
captured operands, replacing byte-assembled packed-word loads with guarded
aligned loads while preserving every F32 product, group subtotal and ascending
group sum. Risk CAUTION: incorrect alignment, changed compiler arithmetic or
a microbenchmark win without request benefit. Use a standalone probe first;
guard aligned and odd offsets, per-rank input, bitwise output and GPU timing.
No production precision, synchronization or default change is admitted by a
local timing win. Require full fixed-trajectory request comparison before
promotion. Rollback is omitting the probe or retaining the current loader.
Keep serial run_safe, 15% floor, external scratch/local weights and existing
upstream tolerances. Numerical return pointer is the late-layer slice below.

The packed-load probe preserves all 7,077,888 compared native output values
bitwise on GPU, including negative/nonfinite and odd-offset controls, but its
seven-pair GPU command-duration medians show no sustained gain (native/word:
0.996851 gate, 0.999277 up, 0.994630 down). Overlapping command intervals do
not establish an additive MoE cost. Retain the production byte loader.

Next discriminator: WN ungrouped DAC Conv1d currently puts neighboring output
lanes at distant weight addresses. Transpose the effective weights inside the
existing normalization-apply pass and change only the Conv1d weight index.
CAUTION: a transpose error, different F32 reduction or a DAC-only gain lost in
complete requests. Keep native/coalesced selection frozen per decoder and native
as rollback. Preserve the same scale reduction, allocation sizes/counts, dispatch
count, padding and in-channel/tap accumulation; grouped and transpose kernels
remain unchanged. DoD: synthetic bitwise tails/dilation guards, all 58 real
DAC stages compared on GPU, unchanged upstream metrics at existing tolerances,
then counterbalanced profiler-off complete codes/WAV/tape/read/wait controls.
The existing full upstream DAC gate is already red; exact native parity and
identical upstream failures establish no regression, not upstream compatibility.

Observed: all 58 DAC stages (157,782,720 values) compare bitwise on GPU with
zero nonfinites; negative controls and seven Metal specs pass. Standalone DAC
ABBA/BAAB medians are 6.298823/2.841204 s (2.217x). Full-request ABBA gives
two pair wins of 1.35797x/1.15125x with exact complete codes, WAV, sampler
inputs and read/wait ledgers in all 12 requests. Median worker warm request
time is 28.994506/23.131649 s for 2.786395 s audio (RTF 10.406/8.302).
Generation-phase timing variation is not attributed to DAC. `coalesced` is
the default; `DRAMA3_DAC_CONV_LAYOUT=native` is rollback. Full upstream DAC
metrics remain identical and red. Details: `docs/drama3-dac-layout.md`.
The subsequent expert-row experiment packs multiple output rows into the
existing 64-thread group without changing native F32 subtotals or group-sum
order. The actual layer-39 probe preserves 7,077,888 values bitwise and shows
a local down-projection improvement. Production Metal specs cover selection,
odd tails, Q4/Q8, per-rank input, invalid IDs and fallbacks; all 13 pass.
The complete matched-input MLX report is identical to the native baseline
(162 boundaries,156 replays), including its existing failed strict gate.
No tolerance, production allocation, dispatch, wait or readback is added.

Full-request ABBA gives mirrored native/candidate ratios0.853759/1.018696;
all12 codes/WAV/tape/read/wait controls pass, but a request speedup is not
established. `DRAMA3_EXPERT_PACKED_ROWS=1` remains opt-in; unset/0 is the
frozen default. Fixed DAC timing also varies, so this run does not isolate
a causal expert-kernel regression. See `docs/drama3-expert-packed-rows.md`.

Current hypothesis at `932e7b6b`: qkv/z/b/a share one decode input and could
use one multi-output dense dispatch while retaining every row's original
64-lane F32 accumulation and reduction tree. No current additive profile
establishes this family's dominance. CAUTION: dynamic resource selection can
cost more than three saved dispatches, or change F32 code generation. Reuse
the same four output allocations in ordinal order, weight mappings and input
view; do not combine live scratch slots, repack weights, or add queue waits.

First discriminator: `drama3_dense_fusion_benchmark MODEL DENSE_FIXTURE MLX_REFERENCE` with
alternating separate/fused groups on pinned upstream inputs, existing command
timestamps and device-side bit/nonfinite receipts. A seeded bit/NaN error
must be rejected. Guard uneven row counts, column tails, F32/F16/BF16 and
input/output/weight offsets before production use. If this local workload
does not improve, retain four dispatches and stop this formulation. Otherwise
qualify an opt-in implementation with exact fixed-trajectory request pairs,
unchanged upstream metrics, allocation/submit/wait/read ledgers and frozen
selection/fallback tests before changing the default. Rollback is the existing
four-dispatch path. Target RTF0.8–0.9 remains open; the DAC-qualified RTF8.302
still needs about9.2–10.4x overall improvement at this fixed workload. Prior
blanket SIMD trials do not justify another selector retry without a new
discriminator.

Observed: the production-API qkv/z/b/a probe preserves 2,668,032 native values
bitwise on GPU, rejects injected bit/NaN errors, and passes all 37,056 pinned
MLX projection values at the existing tolerances. All 21 measured local pairs
favor fusion; pooled paired command ratio is 1.16824. Seventeen Metal examples
pass. A separate `#step` gate covers 243 decode boundaries / 497,664 values,
with exact native parity and identical baseline/candidate MLX metric records.
The old full upstream failure at row 1/layer 39 hidden remains; the prefill-only
late-layer harness is not used as proof for this changed decode path.

Full profiler-off 12-request ABBA preserves codes, WAV, sampler tape, allocation
counts/bytes, reads and waits. Mirrored original/fused request ratios are
0.983651/0.952369, with generation 1.035217/0.972357 and unchanged DAC also
varying. Request medians 24.626830/25.464262 s for 2.786395 s audio do not establish
a speedup. Keep `DRAMA3_BACKBONE_DENSE_FUSION=0` (unset also off), with the
original four inline matvecs as rollback. Local command wins are not realtime
throughput. Details and evidence: `docs/drama3-dense-fusion.md` and
`.landmark/evidence/drama3-dense-fusion.json`. A distinct possible next probe
is exact-tree tower FFN w1/w3 fusion; current dominance and request benefit
remain unproven.

## Prior slice: late-layer matched-input fidelity — 2026-10-16

From `d05bc45d`, locate the first tolerance violation in execution boundaries
against pinned MLX scalar and sequence arms, then distinguish accumulated
input drift from local expert-down, shared-expert and residual arithmetic.
The prior full 158-row gate first exceeds tolerance at sequence layer 35 hidden
and scalar layer 37 hidden; bit differences begin earlier and are a separate
question. These observations do not establish a local operator defect.

Risk: CAUTION for diagnostic evidence and attribution; production arithmetic,
defaults and synchronization boundaries remain unchanged. Use immutable,
hashed upstream operands and original operator return values. Native replay,
route restoration, comparison and first-coordinate selection stay on GPU.
Only compact metrics and witness coordinates/values cross the existing
terminal diagnostic boundary. Upstream raw fixture serialization is an offline
terminal operation, not an inference path. Scratch stays external; run heavy
work serially with the standing 15% memory floor and unrelated processes untouched.

DoD: GPU comparator positive/negative controls distinguish first bit difference,
first tolerance violation and nonfinite; compare all decoder boundaries to
locate the earliest failing layer/row/channel; replay expert down, shared
projections/activation/gating and residual add separately from exactly the same
upstream operands. Preserve native/upstream arm identity, route order, weights,
source hashes and the unchanged 6e-5/3e-4 intermediate tolerance. Check graph
0/2 and zero pre-terminal activation readbacks. Targeted diagnostic tests plus
the original full-prompt intermediate gate are the fallback guards. Report an
open root cause if matched-input observations cannot separate hypotheses.
Rollback is omitting the new diagnostics or reverting this isolated slice.

Observed: one Metal negative-oracle example and sixteen offline tests pass.
Both graph depths produce all 318 comparisons with identical metrics and
witnesses, no nonfinites, one 66144-byte terminal metrics read and the existing
4-byte finite status. No activation read or intermediate finish is added.
All 162 full-boundary metrics per depth reproduce the prior strict gate.
First strict sequence failure is layer35.hidden row154/channel1277
(7.4863434e-5 > 6.1382445e-5); scalar is layer37.hidden row154/channel1557
(6.592274e-5 > 6.5157314e-5). Bit differences begin at layer0 instead.

At exact upstream operands, tested shared operations remain within tolerance;
two-term MoE sum and residual add are bitwise exact throughout layers34–39.
Native down has layer39-only local failures (7 scalar / 1 sequence), with
first witnesses at expert108. Existing mlx_q4 down is bitwise exact on all
15,532,032 scalar values, but has 7 sequence layer39 failures. Whole native
MoE and its residual also fail locally only at39. These findings separate
local39 error from earlier accumulated drift; they do not causally explain
the earlier35/37 crossing. Strict full CLI exits1 after writing both reports.
No tolerance/default/production arithmetic is changed.

Correlated mini-Quad: ROBUST for bounded earliest-boundary and matched-replay
observations; VULNERABLE for a single root cause or upstream promotion.
Report `docs/drama3-late-matched.md`, index
`.landmark/evidence/drama3-late-matched.json`. Next numerical falsifier:
compare the upstream sequence quantized down reduction on these exact inputs,
then decompose the earlier failing cumulative coordinates on identical inputs.
Refresh on source/model/input/reference/tolerance/queue/device/OS drift or raw
artifact loss. Depth2's retained high-water count is 22.93 GB, not physical
residency; numerical equality does not establish a memory/performance benefit.

## Prior slice: native sequence prefill — 2026-10-16

From `a15d4253`, produce multi-row GPU activations through the entire backbone
in layer order, then batch MoE projections across token/rank pairs. The current
path batches selected ranks for only one token. Preserve each token's dense
and quantized reduction order, routing tie order, shared-expert contribution,
causal full-attention prefix and sequential convolution/delta updates.
Teacher-force exactly the existing N-1 prompt pairs in temporal order.

Risk: CAUTION. Begin behind a default-off prompt chunk-size switch, with the
existing rowwise path as rollback. Bound scratch by chunk size; reject invalid
width/capacity before encoding. No precision changes, inference activation
readbacks, extra submits or completion waits are admitted. Component profiling
that drains the GPU is excluded from inference timing. Same-lineage mini-Quad
review is correlated review, not independent replication.

DoD: a genuine missing multi-row API witness, followed by scalar/batch GPU
intermediate and continuation checks at N=2 and N=5, semantic/nonsemantic prompt
rows, graph depths 0/2, recurrent wrap, causal isolation and invalid shapes.
Compare real full-prompt intermediates with the pinned MLX source at existing
6e-5/3e-4 tolerances, and terminal hidden/logits at 3e-5/3e-4. Report native
scalar/batch and both upstream comparisons separately: the known MoE39 mismatch
is open and a passing endpoint cannot close it. Use identical sampler inputs
for end-to-end codes/WAV and queue/read ledgers. Counterbalanced warm completed
GPU measurements must distinguish MoE, backbone prefill and complete requests.
Any newly failing numerical/state/queue guard blocks default promotion.

The cheapest falsifier is a five-row tiny fixture crossing convolution wrap
and a full-attention layer, followed by one ordinary decode step. A query
seeing a future K/V, reordered delta update, lost tower pair or changed expert
sum rejects the implementation. Strong fallback: unchanged scalar path plus
saved upstream fixtures. Standing authority remains serial run_safe, 15% floor,
unrelated processes untouched, external scratch/local weights, exact-path
atomic local commit after observed checks; no push.

Observed: 86 relevant Metal and 29 offline examples pass. For the real 158-row
prompt, all 932 scalar/batch GPU checks per depth (0/2), saved MLX terminal
hidden/logits and all twelve bounded 60-frame request codes/WAV/tape/read
ledgers pass. Normal prefill adds no read/wait/submit. One empty final-row
submission is elided for the multirow tail: request counters are 310/309 at
depth 0 and 251/250 at depth 2, with complete drainage.

Warm scalar/chunk-16 backbone medians are 20.296115/8.747524s (depth0)
and 18.912527/8.031174s (depth2). MoE39 wall ratios are 2.192x/1.357x; full
bounded-request ratios are 1.217x/1.366x. GPU completion intervals accompany
these active-host descriptive measurements; observer overhead and the fixed
60-frame truncation limit extrapolation.

The strict upstream intermediate gate stays OPEN: per native arm, scalar MLX
has 100 failing elements across eight stages and sequence MLX 222 across eleven.
Both native arms have identical metrics, also identical across graph depths.
The final CLI writes all 1307 checks then exits 1 on this red gate. No tolerance
is widened and default remains chunk 1. Correlated mini-Quad verdict ROBUST
for bounded native mechanics/regression and VULNERABLE for upstream promotion.
Report `docs/drama3-batched-prefill.md`, index
`.landmark/evidence/drama3-batched-prefill.json`. Next numerical falsifier:
locate the earliest late residual failure on identical inputs and distinguish
expert-down from shared/residual accumulation; not every coordinate is proven
to be the historical expert108 case. Direct GPU token-row expert addressing is
an unmeasured performance candidate. Refresh on source/model/input/tape/fixture/
tolerance/queue/workspace/observer/device/OS/storage drift or artifact loss.

## Prior slice: mapped weight resource extents — 2026-10-16

From `30833811`, test the net warm-layout effect of page-rounded per-Tensor
no-copy resources against whole-shard resources in one Engine. This reopens
the previously rejected per-tensor route only with an excluded prime per
layout block, retained converted GPU caches, a views-only sham in both layouts,
and measured resource intervals/binding exposure. It does not isolate length
from resource count or establish internal driver registration cost.

Risk: CAUTION. Only the explicitly enabled diagnostic bank may change layout,
at the existing terminal boundary. Retire all old wrappers before creating any
replacement; retain immutable mmap owners. Rebase byte offsets and guard the
entire raw Tensor interval, including all dynamically selectable experts.
Page envelopes may overlap for read-only inputs: report their union separately
from summed resource lengths. Reject writable-resource generalization.
No shader, arithmetic, precision, inference submit/wait/readback or ordinary
inference default change is admitted. Rollback is omitting diagnostic enabling
or reverting this slice.

DoD: a genuine failing range/rebase witness, then focused graph-0/2 GPU checks
for dense/packed/conversion and terminal-boundary rejection without draining;
offline schedule and receipt mutation tests; S/N/N/S blocks with P/V/V/P jobs
and an excluded prime per block; exact full codes/WAV, sampler and queue/read
ledgers; unchanged saved-MLX full-code and 158-row final hidden/logit gates at
atol 3e-5 / rtol 3e-4. Falsifiers include a tensor outside its buffer, partial
expert coverage, a live retired wrapper, lazy construction in scored jobs,
changed converted-cache identity or numerical/queue drift. Active-host results
are descriptive and do not certify residency or an extent-only speedup.

Observed DoD: 76 relevant Metal examples and 7 new/14 existing offline tests
pass. All 20 full-code/WAV/tape and 122-submit/wait/read ledgers are exact;
four saved-control MLX GPU full-code comparisons and the final-source 158-row
final hidden/logit gate pass at unchanged 3e-5/3e-4 tolerances. The prefill
candidate also enables existing workspaces, so it is a combined numerical
guard. It does not close every intermediate or the known MoE39 discrepancy.

S/N/N/S block medians (P,V seconds) are (28.302495,23.448035),
(24.824810,32.941053), (26.395670,28.731601), (31.754992,26.501395).
N-minus-S favors N in P (-3.477684/-5.359322s) but favors S in V
(+9.493018/+2.230206s). No uniform layout gain or default promotion follows.
Warm first-command raw bindings stay 862; active resources rise 5 to 853,
while the active-bound interval union falls only 0.0768% (about 24 GB remains).
All resources rise 6 to 1310; their summed lengths are not physical residency.
Scored construction/retirement is zero; all 566 converted-cache host identities
stay fixed. First-driver ranges remain 0.316–9.421s across arms; process CPU
outliers/carryover remain competing explanations. The view sham falsifies a
uniform layout-speed claim, not all possible benefits of narrower resources.

Report [Mapped weight resource extents](drama3-weight-extents.md), evidence
`.landmark/evidence/drama3-weight-extents.json`. Keep ordinary whole-shard
resources. Reopen this layout route only with a new discriminator for the
layout-by-view interaction and host scheduling. Next bounded kernel question:
which Nomic MoE batching ideas remain applicable beyond DRAMA's existing
GPU route/top-8 projection batching? A real multi-token activation producer
must exist before promoting cross-token batching. Return numeric frontier:
MoE39/common-prefix and DAC post-module. Standing authority remains serial
run_safe, 15% floor, unrelated processes untouched, external scratch/local
weights, exact-path local atomic commit/no push.

## Prior slice: raw weight views and binding exposure — 2026-10-16

Completed bounded contrast from `c37f4d91`: one Engine and a separate persistent
prime, followed by all six permutations of persistent / views-only / fresh
wrappers (18 warm requests). Views-only clears raw TensorBuffer objects while
preserving identical live MetalBuffer objects, mmap owners and GPU F32 caches.
Fresh continues to clear views and replace whole-shard wrappers. A passive,
job-local first-command histogram counts encoder set_buffer calls after host
return, with zero counts retained for unbound shards.
Opaque owner IDs identify shards within this worker, not Metal resource IDs.

Risk remains CAUTION: resetting views with work in flight could invalidate
bindings; all resets and probe starts require the existing terminal boundary.
Rollback is disabling the diagnostic executable or reverting this slice.
No new inference submit, wait, readback, shader or precision change is admitted.
Falsifiers: a views-only factory call or changed buffer identity; a lookup
counted as a binding; an unused shard reported as exposed; a second command
included in the first-command histogram; changed codes/WAV or wait/read ledgers.
DoD: genuine failing witnesses followed by focused graph-0/2 GPU and negative
boundary cases, offline schedule/qualification mutation tests, balanced warm
P/V/R requests with exact outputs, and unchanged saved-MLX full-code and
158-row terminal hidden/logit gates at the existing tolerances. Active-host
timings can reject a simple cost explanation but cannot establish equivalence
or identify internal driver registration from wrapper addresses.

Observed DoD: 69 relevant Metal and 14 offline tests pass, with all 19 full
60x11 codes/WAV and 122-submit/wait/read ledgers exact. Three saved-MLX GPU
full-code checks and both V/R 158-row final hidden/logit gates pass unchanged
3e-5/3e-4 tolerances. Activations remain on GPU. Warm first-command histograms
are exact: 5442 total bindings, 862 raw-weight bindings across five model
shards (100/183/198/190/191), zero for the codec shard; 566 converted GPU
cache entries retain their identities. These are host exposure counts.

Preparation medians P/V/R are 2.542us / 3.917us / 1.075104ms; first-driver
ranges remain 0.391–5.862 / 0.393–9.054 / 0.377–6.375s. Within-triplet request
R-minus-P median is -0.018379s, range -1.049909..+6.465244s, three wins each.
No stable speed gain, equivalence or default promotion is established. Warm
first-command page-ins/disk reads are zero; process system-CPU time still
spreads into seconds. This does not identify driver work or prove residency.
The active host and cross-triplet carryover limit causal inference.
See [Raw weight views and binding exposure](drama3-weight-views.md) and its
evidence index for exact observations and scoped ROBUST mini-Quad verdict.
Next falsifier: a separately guarded narrower-resource-extent contrast with
the view sham and identical ownership/numerical/queue witnesses. An older
per-tensor-view probe had no stable gain; changing an encoder length alone
does not change the bound Metal resource. Return numeric frontier remains
MoE39/common-prefix and DAC post-module; this slice does not close either.

## Prior slice: mapped weight wrapper lifetime — 2026-10-16

From `99a4f447`, compare persistent raw mapped buffers with recreated host Metal
wrappers over the same mmap owners/pages. A faster warm request alone cannot
identify driver registration cost. Risk: CAUTION; the diagnostic bank is opt-in
on a new Context, rotates only at an already terminal boundary, retains mmap
owners and abort-quarantines them after failure. Active encoding, workspace,
pending commands and aborted contexts reject rotation without a wait/drain.
All raw bindings use the bank when enabled; ordinary inference retains the
existing global cache through an optional-bank branch. No shader, arithmetic,
precision, queue or inference read/wait policy changes. Rollback is omitting
diagnostic enabling or reverting this isolated slice.

DoD: exact backing/offset and GPU packed/dense/conversion witnesses at graph
0/2; terminal boundary counterexamples; counterbalanced same-worker pairs with
separate cold primes; unchanged saved MLX full-code and final hidden/logits
gates after a 158-row prefill.

The completed matrix has 18 requests: two cold primes and 16 warm jobs, with
P/R/R/P twice and R/P/P/R twice inside two workers. Every fresh job retires and
recreates six whole-shard wrappers over the same 25867452416 declared bytes;
converted GPU F32 weights remain cached. Warm preparation medians are 2.646 us
persistent / 1.087167 ms recreated. The paired request-to-result difference has
median -0.000816 s and range -5.439325 to +6.644850 s, with four pairs favoring
each arm. First-driver intervals still span 0.311–7.686 s in the persistent
arm and 0.312–7.229 s after recreation. No end-to-end gain or default promotion
is admitted. Host wrapper creation alone cannot explain the seconds-long
spread; internal registration reuse and work below submission remain unknown.

All full 60x11 codes, WAV, sampler and graph/read ledgers are exact: 122 existing
submissions/waits per request, 496 control and 494160 terminal bytes, no new
inference synchronization or activation read. The 67 relevant Metal tests,
both saved-MLX full-code GPU comparisons and final hidden/logits gate after a
158-row prefill pass unchanged 3e-5/3e-4 tolerances. This does not compare every
row/layer or close the existing MoE39 failure. Warm first-command windows have
zero process page-ins/disk-read bytes; whole-request reads are not always zero, and zero
counters do not certify residency. Converted-cache and lazy view-rebuild
asymmetry, overlapping counter windows and quiet-host-off noise limit inference.
See [Mapped weight wrapper comparison](drama3-weight-wrappers.md) for exact
pairing, scope, receipts and lineage. The subsequent
[view-only sham](drama3-weight-views.md) executes the proposed separation and
passive binding counts. Driver validation granularity remains a hypothesis
requiring its own ownership/numerical probes. Keep MoE39/common-prefix and DAC post-module as
the open numerical return frontier.

## Active slice: first-prefill startup attribution — 2026-10-16

From `ccb0d178`, separate host preparation, process page activity, Metal driver
scheduling and GPU execution before changing the startup path. The existing
before-GPU wait is an observation, not proof of GPU contention or page-in.
Competing predictions: CPU preparation appears before commit; driver work
appears in the command buffer kernel scheduling duration. Do not infer a
scheduling-to-GPU residual without a certified shared clock epoch.
Process page-in/fault and disk-read deltas can support residency activity but
cannot assign its duration or exclude kernel-managed/other-process activity.

Risk: CAUTION. Admit default-off boundary resource snapshots and passive Metal
scheduling timestamps only. Preserve arithmetic, precision, queue ownership,
existing completion waits and control/terminal reads. Do not insert callbacks,
waitUntilScheduled, warm-up commands, page purges or CPU activation reads.
Rollback is disabling diagnostics or reverting the isolated slice commit.

DoD: fake native command tests preserve one commit/wait and authoritative
failure status even with unavailable metrics; offline tests reject malformed
or reset counters and invalid scheduling order without inventing cost. Run
counterbalanced diagnostic-off/on cold-plus-warm fixed-input workers and retain
exact full codes, WAV, sampler, graph/read ledgers. Execute the unchanged pinned
MLX full-code control and 158-row prefill pointwise gate. Attribute only the
intervals and counter windows observed; leave unresolved residual causes open.

The 12-request observer-off/on/on/off matrix preserves the complete codes,
WAV, sampler and graph/read ledgers. In the six resource-observed requests,
request entry to first commit is 10.7–24.7 ms, compared with 0.35–11.96 s of
first-command driver scheduling and 0.163–0.195 s GPU execution. Both new
processes register about 24 GB of process disk-read accounting and 1464429
page-ins after commit. Warm requests retain 0.35–4.83 s of scheduling, with
0.35–4.82 s process system CPU and zero registered page-ins/disk reads in that
window. This rejects page-in as the sole explanation for the warm spread; it
does not time page-in or prove a physical-disk cache state. Separate own-worker
CPU samples observe the Metal dispatch thread in
`IOGPUCommandQueueSubmitCommandBuffers -> iokit_user_client_trap`, while the
main thread uses the existing completion wait. Work below that trap remains
unresolved. Sample counts are not CPU time. Overlapping command resource
windows are explicitly nonadditive.

Native fake-command and 63 Metal cases preserve the existing wait/read and
status behavior; 23 offline and 6 CPU cases pass. The saved MLX shared-tape
full-code and 158-row prefill gates pass without widening tolerances. The
known MoE39 full intermediate failure remains open. See
[First-prefill startup attribution](drama3-prefill-startup.md) for exact samples,
lineage, observer cost and the interrupted-wrapper limitation. No startup
optimization or default change is promoted. The subsequent
[mapped wrapper comparison](drama3-weight-wrappers.md) executes that contrast:
recreation costs about 1 ms, while multi-second warm scheduling
spread survives both modes. Driver resource validation remains an unresolved
hypothesis; wrapper caching alone has no measured end-to-end promotion.

## Active slice: GPU kernel attribution — 2026-10-16

From `22c99628`, profile backbone/tower dispatches and distinguish command
execution from commit-to-start and completion-wait variation before selecting
one optimization. Existing command-duration sums do not identify kernel cost
or the cause of wait variance. Start with an external Metal System Trace of
the unchanged binary; keep traced and ordinary paired timings separate.

Risk: CAUTION. Profiling can perturb dispatch pacing or report only an
aggregate interval. A kernel optimization must preserve each row's arithmetic
order and precision, finite/status guards, dynamic expert order, ownership,
queue depth and the existing control/terminal read boundaries. No new waits,
GPU/CPU activation transfers, or widened tolerances are admitted. Reject a
profiling route that requires per-kernel completion waits on the normal path.

DoD: qualify the profiler against named kernels, capture a warm fixed-input
trajectory, and identify a measured optimization candidate. If the installed
profiler cannot supply additive per-family cost, preserve that limit and use
homogeneous actual-weight probes instead of ranking sampled row counts.
Compare baseline/candidate
in counterbalanced pairs, including profiler-off request-to-WAV measurements;
retain full codes/WAV/sampler/read/wait identity and the unchanged saved MLX
shared-input and prefill gates. Use focused synthetic boundary tests for the
selected kernel. A local GPU speedup alone does not promote an end-to-end
speed claim. Roll back through the isolated feature commit or its diagnostic
flag; generation scratch remains opt-in.

The bounded Instruments export qualifies shader coverage but has overlapping
sample intervals and no additive family weight. The native timeline instead
localizes a 6.52 s warm-request spread mainly to before-GPU wait. The first
prefill command shows the dominant pre-start delay; its cause remains
unresolved. This is not evidence that its arithmetic became slower. The generic
64-lane SIMD dense route is bitwise equal in 12 full requests but has no stable
request-to-WAV advantage in the counterbalanced experiment.

An opt-in 32-lane dense kernel gives each lane the two original
64-lane partial accumulators, combines them in the original tree order, and
removes the cross-SIMD-group barrier. It must match the original kernel bitwise
for dtype/offset/tail/cancellation witnesses and real tower weights. Keep the
default `tree` until profiler-off full-request pairs and unchanged upstream
controls admit a promotion. Unsupported SIMD width must retain the original
64-lane tree path. All 12 full-request jobs remain exact, but pooled warm
medians are 34.2330 / 34.9154 s for tree/simd32; no default promotion is
admitted. This is an ordinary launch/reduction implementation change, not an
LTP/WBA claim.

The affine MoE candidate compares the existing expert kernel with 32- and
64-thread launches on the captured upstream layer-39 normalized input and
per-rank down-projection activations. The shader's fixed `g += 64` partition
allows 32 threads only when group count is at most 32; group 33 must retain
64 threads. Require bitwise native baseline/candidate parity and no added
intermediate reads or waits. A launch comparison does not establish native
expert output parity with upstream or close the known MoE39 failure.
The captured-input probe gives a median paired GPU duration ratio of
1.04383 (repeat 1.06120) for 64/32 threads, with 32 faster in all seven measured
pairs per capture and 1769472 values per arm bitwise equal across all pairs. Admit an opt-in
`DRAMA3_EXPERT_THREADS=32` and independent Q4/Q8 group-32/33 boundary witnesses;
all 12 profiler-off full requests preserve the complete codes/WAV/sampler and
graph/readback contract. Pooled warm medians are 29.7862 / 30.5885 s for 64/32
threads, so the default remains 64. Final 62 Metal and 10 offline tests,
saved-MLX full-code comparison, and combined simd32/expert32 prefill pass.
The known MoE39 intermediate numerical failure remains open. Details and
lineage limits are in `drama3-gpu-kernels.md`.

Next performance discriminator: separate pre-start residency/resource work
from competing GPU/driver work, or test a guarded aligned Q4 packed-word load
against the current byte loader on captured inputs. Neither is yet a measured
optimization. Keep the numerical return pointer at MoE39/common-prefix and
the DAC post-module frontier. Temporary/reference/run data now live on
`/opt/mldata_ssd/cogni-ml`; the DRAMA checkpoint remains on the local SSD.

Streaming is a source-backed design branch only in this slice. It must name
DAC lookahead/state, ownership of a PCM chunk through playback, backpressure
and cancellation before runtime admission. No low-latency or uninterrupted
playback claim follows from an offline chunk experiment or short-text speed.

### Future continuous audio output

The resident DAC is structurally causal: Conv1d uses left history and the
post-transformer attention reads only past positions (a 128-position window
by default). This supports an incremental design, but calling the current
whole-file `decode` independently for each code chunk resets required state.
Keep the rolling per-layer K/V, absolute RoPE position, depthwise/residual
convolution history, and each transposed convolution's overlapping tail.
Preserve accumulation order and apply bias once when finalizing an overlap.
Source: `kernels/codec_resident.metal:58-96,185-245` and
`codec_resident.cr:211-233,347-418,453-477`.

The first discriminator is GPU whole-stream versus stateful chunked DAC on
identical saved codes, including one-frame and irregular chunk boundaries.
Use the existing pointwise tolerances; do not substitute signal level or
audible plausibility for numerical parity. Establish which PCM prefix is
final before allowing a consumer to play it.

Keep finalized PCM in a bounded GPU output ring outside scratch workspaces.
An output slot needs both checked GPU completion and transport/playback
consumption before reuse. A queue lease alone cannot establish that the audio
consumer has finished. Backpressure must pause production without overwriting
live audio; cancellation must drain or quarantine live slots. Model state,
sampling and DAC remain on GPU. Copying finalized PCM to an audio device or
network is an output boundary, not an activation round trip.

Studio currently generates all codes, decodes all PCM and writes a complete
WAV before its file endpoint serves it (`studio/engine.cr`, `studio/server.cr`,
`studio/assets/app.js`). A future live endpoint and WebAudio consumer can consume
the output ring while the next codes/PCM are being generated. Sentence-sized
independent jobs are an earlier playback option, but require separate tests
for voice/prosody continuity and boundary artifacts. Streaming reduces time
to first sound; uninterrupted playback also requires generation to keep up
with consumption or sufficient initial buffering.

## Host cost of buffer reuse — 2026-10-16

From `b3144c85`, investigate the inconsistent request-to-WAV result before
promoting generation scratch. Competing explanations are host encoding and
resource-retention work, managed allocation/GC, and shared-host/order variation.
Admit optional per-context host timers/counters and phase GC snapshots only;
they must not submit, wait, read activations, force inference-time collections,
or change arithmetic, precision, sampler inputs or existing gates.

DoD: focused timer/nesting/disabled-path and resident depth-0/2 checks pass;
counterbalanced fresh/reuse jobs with profiling off and on preserve complete
codes, WAV bytes and graph/readback ledgers. Compare against the pinned shared-
input MLX control and unchanged prefill gate. Report cold and warm jobs
separately, profiler overhead, phase host/GPU/GC scopes and seconds per audio
second. Nested wall timers overlap device execution; they are not GPU time or
CPU utilization. Managed allocated bytes are not a typed object census.

Risk: CAUTION. The main falsifier is measurement overhead or a hidden wait
misattributed to command encoding. Use leaf encoding timers, separately time
existing waits, and test a nested/exception case. Disabling `DRAMA3_HOST_PROFILE`
removes fine clocks/counters; revert the isolated commit to remove the facade
and coarse GC snapshots as well. Scratch
remains opt-in until paired end-to-end evidence supports a speed claim.

Observed: 24 primary jobs, 12 legacy/current controls and eight existing-
timestamp controls preserve complete codes/WAV bytes and sampler/read/wait
ledgers. Warm generation median backing construction 770.3->8.2 ms and full-GC
delta 1222.5->20.5 ms; encoder API 54.6/46.3 ms and retention 16.9/19.7 ms
(fresh/reuse). Full-GC deltas overlap other scopes. Timeline-off request
medians are 26.853/28.466 s for 2.786 s audio, approximately 9.6/10.2 request
seconds per audio second, excluding model load. Paired deltas vary by seconds;
timestamp command sums stay near 15.5 s generation and 6.0 s DAC. A unique cause
of the earlier slowdown and zero profiler overhead are not established.

Final 33 CPU and 51 Metal examples pass. Complete 60-frame codes match the
saved MLX shared-input F32-core control; 158-row fresh/reused prefill is bitwise
equal and passes the unchanged MLX pointwise gate, without added read/wait
boundaries. No default promotion or arithmetic change. Details and source-
bound raw receipts: [drama3-host-attribution.md](drama3-host-attribution.md),
`.landmark/evidence/drama3-host-attribution.json`. Next discriminator: GPU
kernel attribution and command-wait variation, with DAC measured separately.

## Generation buffer reuse — 2026-10-16

From base `ce592825`, `DRAMA3_GENERATION_SCRATCH=1` enables four bounded
GPU workspaces with per-call ordinal slots, persistent escaping frames and
explicit terminal release. Uploads, converted weights, finite status and
retained diagnostics remain outside reused storage. Replaced backings survive
until terminal release or abort quarantine. No new wait, readback, arithmetic
or tolerance change is admitted; default scratch remains 0.

Observed on the pinned short-text trajectory: warm generation backing
allocations 135,139 -> 1,894, requested bytes 1,596,632,160 -> 22,125,244;
construction clocks 819–892 -> 7.15–9.37 ms in the v1 timer-on experiment.
All 18 paired native jobs preserve full code/WAV bytes and graph/readback
ledgers; four use final source after two lifetime/validation guard fixes.
The final native 60-frame trajectory matches the saved pinned MLX F32 control
on shared draws. Actual reused 158-row prefill hidden/logits are bitwise equal
to fresh and pass the unchanged MLX pointwise gate. All 49 focused Metal and
29 CPU examples pass, with a successful final release build.

Request-to-WAV timing is not consistently faster, including the final-source
warm comparison. No default speed promotion is made. Whole-pipeline GC removal,
cross-job pooling and older numerical/voice-quality frontiers remain open.
Details, tested source scope, commands and refresh/rollback bounds are in
[drama3-allocation-reuse.md](drama3-allocation-reuse.md) and
`.landmark/evidence/drama3-allocation-reuse.json`.

## Active slice: GPU reference voice — 2026-10-15

Status: VERIFIED within the one-reference slice on Apple M2 Max, from base
`04b254dd`. The native Metal CLI accepts one reference WAV and its non-empty
speaker-0 transcript in the existing single-prompt path. File parsing and text tokenization are host
input preparation; audio normalization, DAC encoding, vector-quantizer code
selection, audio-row construction, prefill and generation stay on GPU. Codes
must pass directly from encoder buffers to prefill. Do not substitute an
external encoder in native inference or add intermediate activation readbacks.
The no-reference API and its arithmetic remain supported unchanged.

The first file boundary is uncompressed mono PCM16 WAV at 44.1 kHz, at most
10 seconds; reject
other rates/layouts rather than silently resampling. Broader audio loading,
multiple references/speakers, history and streaming are outside this slice.
The reference encoder uses the pinned checkpoint's F32 weights and loaded
rotary coefficients. Existing decoder numeric frontiers remain separate.

Falsifiers/DoD: malformed/partial reference arguments and WAVs fail before
model work; prompt segment IDs and all audio rows match the pinned MLX prompt
builder; native encoder intermediate comparisons use the existing codec
stage gates (max absolute 1e-3, RMS and normalized RMS 2e-4), with exact code
IDs required independently. GPU handoff tests must show no new model-activation
readbacks and preserve graph-depth 0/2 behavior. Run no-reference regression,
then native/MLX reference/no-reference pairs on the same plain text and exact
F32 sampler tape. Terminal waveform windows measure quiet intervals, not words
or sentence alignment; frame caps and numerical failures remain explicit.
No gate, precision or sampling setting may be relaxed to admit this slice.

Observed DoD: all 102 encoder/VQ stages pass those unchanged gates at graph
depths 0 and 2; all 850 code IDs and 168 by 11 prompt elements are exact.
First prefill hidden/logits pass `3e-5 + 3e-4 * abs(reference)` with zero
pointwise failures. All stage/prompt/prefill metrics agree between depths;
terminal diagnostics add no graph work. The 27 resident GPU tests, 43 CPU
contract tests and five CLI rejection controls pass on the final source.

The fresh paired Russian plain-text run uses identical actual 5,632-element
F32 sampler inputs, verified by run-side hashes and a producer/tape proof.
Native and pinned MLX agree on all 11 IDs in every generated frame: 463
without reference, 218 with reference, both ending at EOS. Waveform duration
changes from 21.502 to 10.124 seconds; internal quiet runs of at least one
second change from four (8.5 seconds total) to zero. Native and MLX quiet
windows agree. This measures one reference/transcript, text and seed; it
does not certify words, identity, a unique cause or a universal pause remedy.
See `docs/drama3-reference-voice.md` and
`.landmark/evidence/drama3-reference-voice.json` for source-bound raw evidence,
controls, graph/read boundaries and the scoped correlated Adversary review.

Risk: CAUTION, because incorrect causal padding, residual order or near-tie
code selection can condition a different voice while still producing audio.
The strongest guard is a full exact-code comparison plus stage-local metrics
on the same waveform and checkpoint, followed by unchanged-path regression.
Rollback is the isolated feature commit; existing text-only inference remains
available without reference flags. Refresh all evidence on source, checkpoint,
waveform/transcript, sampler, runtime/device or gate changes.

## Emotion placement, pauses and voice controls — 2026-10-15

At base `51789b83`, four one-prompt cases run through native resident Metal
and the pinned MLX shared-uniform control: Russian/English with five emotion
cues and whisper last, and the same spoken bodies without cues. The native
Russian final phrase is heard by the user; the English one is not. All512
English end-whisper frames match native/MLX exactly, with no EOS and a quiet
tail from6.5s through the23.777s frame cap (peak9–22PCM16 per half-second
window). The native/MLX code fork is not necessary for this English result;
original Fish text generation and unique cause remain untested.

Russian no-cue463 and English no-cue239-frame trajectories also match exactly.
The Russian no-cue pair still has contained2–2.5s low intervals at peak2PCM16
per half-second window. These are not word/sentence alignments. Emotion tags
are not necessary for those intervals. Russian end-whisper diverges at
zero-based frame262 and finishes512/494frames native/MLX. One seed and these
two final-phrase reports do not establish language prevalence or full delivery.

Source/tokenizer audit: original Fish and MLX pass emotion cues as text;
`[whisper]` is four ordinary BPE tokens, with no explicit whisper-state reset.
The loops emit codes until EOS/cap, with no timed-pause or energy-based stop.
Their one-shot repetition retry checks the previous16 main IDs, not silence,
and does not guarantee a non-repeated retry. Offline WAV writing adds no gaps.
Speaker tags name participants; a specific voice is conditioned by reference
audio codes plus transcript. MLX ignores its generic `voice` argument. All
eight controls omit references; at that snapshot native had text-only
speaker0 and no reference-input/encoder interface. The later reference slice
above implements one native reference waveform plus matching transcript.

See `docs/drama3-emotion-placement.md` and
`.landmark/evidence/drama3-emotion-placement.json` for exact prompts, source
anchors, input/output hashes, completed diagnostics and correlated review.
Raw `emotion-whisper-end-20261015/` stays outside Git. No arithmetic, precision,
tolerance, defaults or read boundaries changed; timings are not benchmarks.
Next causal controls: isolate only whisper position, then separately add one
fixed reference to the same plain prompt/draws. Larger frame bounds alone
cannot certify restored words. Keep loaded-RoPE replay, code-fork111 and
MoE39 frontiers open. Refresh on source/model/tokenizer/conditioning/sampler/
artifact/runtime/device/OS/gate or bound listening-evidence drift.

## Original Fish full-stream GPU DAC control — 2026-10-15

At base `c259151d`, the original Fish `DAC.from_indices` independently decodes
the same four complete ten-book streams used below. Source revision is
`541566932abbaebcb0a40e562a1f846738383f38`; the original DRAMA-3 model revision
is `e0a9218aa48888270f06ef1ed9ea67622c0e17d2`, with `codec.pth` SHA-256
`74fc41c5a7151c6f350af8bd7e5d6e3accfcc7f3dfbfac23afd35af07052bb2f`.
This follows the original direct DAC CLI's checkpoint F32 loader semantics:
535 F32 parameters, preserved parametrized weight norm, three constructed
BF16 RoPE buffers and three boolean masks. It does not test the complete
Fish text-to-speech serving loader's BF16/folded-weight-norm configuration.

Both the unobserved and observed original decodes run on PyTorch 2.8.0 MPS,
with CPU fallback disabled and a dispatch guard rejecting host tensors and
scalar extraction during each full model call. A GPU default-device context
also keeps factory-created causal masks on MPS. The observer captures 58
actual intermediate boundaries and matches the separate unobserved full
waveform elementwise in all four cases. Reference export occurs afterwards.
Completed Fish/MLX fixtures upload once and are compared on native Metal;
native-vs-Fish comparisons observe actual native buffers during decode.
No model activation is read on CPU before completed decoding. Native depth-2
CogniGraph still submits/completes/waits for 32 command buffers; terminal
reads add no graph work or waits. Existing control boundaries remain.

| Saved codes | Frames | Fish vs MLX first failed observed stage | Fish/MLX max PCM16 delta | Native/Fish max PCM16 delta |
| --- | ---: | --- | ---: | ---: |
| Audible bilingual seed 1 | 197 | none; full gate PASS | 1 | 16 |
| Native styled first chunk | 427 | quantizer.upsample.1.convnext | 1 | 4 |
| MLX styled first chunk | 741 | none; full gate PASS | 1 | 13 |
| Common original text without cues | 768 | quantizer.post_module.layers.7 | 1 | 3 |

Fish/MLX waveform pointwise and one-LSB gates pass in all four cases, but
strict full parity passes only two: the 427-frame case fails eight stage
gates and the 768-frame case fails one. No gate or reference precision is
relaxed. Native/Fish fails every full gate, first at
`quantizer.post_module.layers.0`, as native/MLX did. The new native WAVs are byte-identical to the four prior
native cross-decodes; the MLX-reference regression also reproduces its prior
58 stage metrics and WAV exactly. Numerical DAC parity remains open.

The original Fish decoder preserves the same quiet continuation. Every
half-second PCM peak agrees among all three decoders in the measured regions:
36–57 units from 4.0–19.830 s for native styled codes, 37–51 from
12.0–34.412 s for MLX styled codes, and one unit from 5.5–35.666 s without
cues. These are signal measurements, not transcripts or an audibility test.
The existing human listening reports identify the missing cat sentence in
the two styled first chunks. Substituting the original Fish DAC does not
restore signal in their quiet continuation. This rejects a decoder-port-only
explanation for these saved streams, but does not establish the unique cause:
conditioning/generation, code representation and a shared checkpoint defect
still need discriminating evidence. Original Fish text generation was not run.

A separate CPU serialized-weight audit finds all 541 original tensors match
the MLX checkpoint exactly after the pinned converter's BF16-to-F32 promotion;
the other types, names, shapes and values are preserved. Original counts are
535 F32 / 3 BF16 / 3 bool; converted counts are 538 F32 / 3 bool. Runtime
sanitization renames 180 keys without collisions. This is a byte/source audit,
not CPU inference or proof of checkpoint semantic correctness.

The audit resolves an important RoPE distinction: MLX's F32 constructor cache
is overwritten by the serialized caches. Every loaded F32 coefficient is
exactly the original BF16-rounded value expanded to F32. Our resident Metal
path instead recomputes frequency/angle/cos/sin in F32 and does not consume
those caches. This is a concrete numerical-path difference, not yet the
measured cause of the layer-0 discrepancy or silence. Next numeric falsifier:
use the exact loaded RoPE coefficients on GPU with matched activations and
compare the first failing block without widening tolerances. Next silence
falsifier: locate when generation enters the quiet-code continuation with
matched conditioning and sampler inputs; keep the code-fork111/MoE39 return
frontiers distinct until an onset discriminator connects them.

The measured diagnostic DoD, guard negative controls, unchanged-path
regression, source/runtime/command hashes and correlated mini-Quadrumvirate
review are indexed in `.landmark/evidence/drama3-fish-codec-cross.json`.
Raw receipts, full stages and WAVs remain under `fish-codec-cross-20261015/`
outside Git. Diagnostic timings are not performance benchmarks. Overall
silence-cause investigation remains IN_PROGRESS; the scoped decoder
substitution result is established for these four inputs. Refresh on source,
checkpoint, input/artifact, upstream, runtime/device/OS, precision/layout/gate
or directly bound listening-evidence drift. Reverting this diagnostic commit
removes the new tools; production arithmetic/defaults are unchanged.

## Full-stream same-code GPU DAC comparison — 2026-10-15

At base `1898cc55`, four complete saved streams are decoded independently by
native resident Metal and the pinned official MLX `DAC.decode`: the audible
bilingual seed-1 positive, both independently generated styled first chunks,
and the common no-cue stream. Each arm receives exactly the same ten DAC
codebooks; the main token is excluded. No frame truncation, code clipping,
resampling, gain normalization, precision or production kernel change.

The new optional native device observer captures actual buffers at 58
boundaries: ten embeddings/projections, the combined latent sum, eight
post-transformer layers, post-module output, both upsampling stages, all
decoder blocks and waveform. Comparisons, compensated reductions and
half-second waveform statistics run on GPU. There are no activation host
reads during native decode. All four runs retain the existing 32 submitted
and completed command buffers at CogniGraph depth 2; terminal reduced-metric
and completed-waveform reads add no submission or completion wait.
This preserves existing drain boundaries, not a claim of zero GPU waits.

The reference observer retains actual upstream module outputs on MLX GPU.
For every case its full waveform is elementwise identical to a separate
unobserved official decode. Only after complete evaluation are reference
fixtures exported; native uploads them once before its decode. These fixture
and final diagnostic/WAV serialization boundaries are outside model arithmetic.

| Complete code stream | Frames | Duration | First failed stage | Max PCM16 delta |
| --- | ---: | ---: | --- | ---: |
| Bilingual seed 1, audible positive | 197 | 9.149 s | post-transformer layer 0 | 16 |
| Native styled first chunk | 427 | 19.830 s | post-transformer layer 0 | 4 |
| MLX styled first chunk | 741 | 34.412 s | post-transformer layer 0 | 13 |
| Common original text without cues | 768 | 35.666 s | post-transformer layer 0 | 3 |

All ten embedding outputs are bit-exact in all four cases. Every book
projection and combined latent sum passes the existing stage limits. The
first violated limit is max absolute error at
`quantizer.post_module.layers.0`: 0.0249863 / 0.0291367 / 0.0316620 / 0.0228043
in table order. This names the earliest failed observed boundary, not the
first nonzero numerical error or a particular arithmetic primitive.
The limits remain max absolute `1e-3`, RMS `2e-4`, normalized RMS `2e-4`;
waveform remains `max(5e-5, abs(reference)*1e-3)` per sample, plus the separate
finite one-PCM16-LSB gate and nonzero-waveform sanity check. Every full parity
gate fails. A successful probe process means a valid measurement, not parity.

The measured quiet continuation survives both decoders over the entire saved
stream, including the two first chunks whose codes diverge after frame 111.
For native-generated codes from 4.0–19.830 s, every half-second window peaks
at 36–57 PCM16 units; for MLX-generated codes from 12.0–34.412 s, 37–51 units.
For the no-cue codes from 5.5–35.666 s, every window peaks at one unit.
Native/reference window peaks agree in every window of these regions.
These are low residual signals, not literal digital zeros or a transcript.
The earlier user listening report remains the source for missing words in
the two named first chunks. No new agent listening or ASR accuracy claim.

The three reproduced native WAVs are byte-identical to their archived
generation outputs. For all three available archived MLX controls, terminal
F32 output reproduces their PCM exactly with the legacy half-away rounding.
The new cross-comparison WAVs instead use native ties-to-even rounding in both
arms. This separates a terminal one-unit serialization difference from model
arithmetic; it does not erase any failed gate.

This refutes a native-DAC-only explanation for the gross quiet continuation
in the tested streams: substituting MLX does not restore it. It does not
establish why the codes/reference enter that region, correctness of the
codes as speech, original Fish parity, or absence of a shared codec/checkpoint/
representation defect. Numeric DAC parity remains open independently.
Source inspection matches the 128-token inclusive causal window, RMS
epsilon `1e-5`, RoPE formula and block equations. Different reduction/fused
attention paths and precomputed versus recomputed RoPE remain hypotheses,
not localized causes.

A separate lazy-GPU metadata/binding audit closes the `strict=False` missing
independent-weight hypothesis for this checkpoint. All 541 sanitized tensor
names/shapes match the independent model slots, and official loading binds
all 541 to the supplied checkpoint arrays. A naive strict load rejects 120
extra nested convolution slots in 60 modules. These are runtime weight/bias
destinations: the pinned causal weight-normalization wrapper overwrites them
from loaded `weight_g`, `weight_v` and bias before every call/step. The audit
matches every missing slot to exactly those destinations and checks their
lazy replacement; no decode or tensor evaluation is required. It does not
certify checkpoint semantics or original Fish agreement.

The GPU reducer self-test passes against a small independent F64 fixture
oracle, including corruption across a reduction partition, equality,
nonfinite input, zero-reference normalization, large indices and waveform
windows. The resident spec passes two examples, including observer-on/off
bitwise equality and both input layouts. Ten reference-input CPU specs,
Crystal formatting and diff checks pass. Raw activations remain outside Git.
Diagnostic timings include retention/reductions/I/O and are not benchmarks.
A correlated source/report adversary finds the measured substitution claim
ROBUST within these four streams, while preserving every failed parity gate.
The observer callback receives a mutable buffer; its read-only behavior is
checked for this probe, not guaranteed by the public callback type.

Index: `.landmark/evidence/drama3-codec-cross.json`; raw source/build/run and
fixture receipts: `codec-cross-20261015/`. Tools and commands are documented
in `scripts/drama3/README.md`. The existing generation/MoE39 return frontiers
remain open. Next silence discriminator: the same complete codes/checkpoint
through the original Fish codec, or independently validated speech codes, to
challenge shared-reference failure. Next numeric discriminator: layer-0
attention norm/QKV/RoPE/attention/residual/FFN boundaries on matched input,
with GPU reductions and terminal-only reads. Refresh on source, compiler,
checkpoint, input/artifact, upstream, runtime/device/OS, precision/layout/
gate or directly bound listening-evidence drift. Rollback is the atomic
diagnostic commit; the production observer defaults to nil.

## Listening follow-up and local ASR limits — 2026-10-15

At base `3156210b`, the user confirms that neither
`native-original-first-chunk.wav` nor
`mlx-official-chunks-styled.chunk-001.wav` delivers the sentence about the cat.
This closes the listening question for those two named clips as a negative
content result. A separate delayed reply describes a first clip ending at
“шорох” and a second whispered clip ending at “происходит”. Its question was
labelled “without tags”, which conflicts with that description. Preserve the
reply without assigning it to both no-cue WAVs; their per-file listening
identity remains unresolved.

Direct audio forwarding was attempted, but this session returned
`audio content omitted because you do not support audio input`. No agent
listening claim follows from a local path or an audio player rendering.
Instead, a blind local ASR control uses the pinned
`mlx-community/whisper-large-v3-turbo-asr-fp16` checkpoint at
`624c19c9af5603fa73b83bce14d4aeea96156d18`, through the existing clean MLX-audio
revision. All twelve model files are SHA-bound, with available Hub LFS digests
checked. ASR model inference selects MLX GPU with FP16 weights; audio
resampling and ASR decoding control include host work, separate from DRAMA
inference. No DRAMA precision, tolerances or generation boundary changes.

Ten full-file/negative controls and seven isolated controls run with automatic
language detection, temperature zero, no initial prompt or hotwords, and no
conditioning on earlier transcript text. The expected transcript is never
supplied to the recognizer. The observed limits are material:

- Digital zero and a quiet ten-second no-cue suffix both produce the invented
  text “Thank you.” The recognizer therefore fails the empty-silence content
  guard. It cannot certify that a word was actually spoken or absent.
- Both full no-cue WAVs yield the first sentence through “шорох”, followed by
  “Продолжение следует...”, which is absent from the requested text. Native
  styled chunk 0 also yields only that first sentence; the MLX styled first
  chunk yields the whispered continuation through “происходит”. Neither ASR
  output contains the cat sentence. These are candidate transcripts, not an
  additional listening confirmation or proof of missing speech.
- Both MLX final chunks yield “Ну и напугал же он нас сегодня, маленький
  проказник.” This supports a candidate delivered final sentence, not a
  full-text pass for the preceding chunk.
- Seed 1 and seed 3 both yield the Russian and English sentences. Independent
  crops from four seconds onward also yield the English sentence, with auto
  language `en`; their first four seconds yield Russian. Seed-3 crop RMS is
  0.109728 on normalized F32 PCM. This conflicts with the earlier human
  Russian-only report. A byte-exact check binds every seed source to its
  complete segment in the listening sequence, including seed 3 and both
  half-second zero gaps. The content disagreement is unresolved; neither
  source is silently promoted over the other.

Consequently, the earlier interpretation of seed 3 as an established English
omission is reopened. Its 179 equal code frames and EOS remain measured facts;
content is contested. The original long quiet-onset result remains unchanged:
its common native/MLX code prefix extends well past the measured quiet onset.
No unique silence cause, speech fix, strict sampler/DAC pass, or ASR transcript
accuracy is claimed. A full-stream matched-code cross-decode is the next
causal discriminator for the new diverging first-chunk pair; shared-prefix
PCM agreement alone does not establish decoder parity for their continuation.

Artifacts and source/run bindings are indexed in
`.landmark/evidence/drama3-listening-asr.json`; raw controls are under
`sampler-voices-20261015/asr/`. The earlier sampler packet remains a historical
snapshot at its source state. This follow-up supersedes its seed-3 content
interpretation, not its code/EOS measurements. Refresh on source, checkpoint,
file identity, preprocessing, ASR options, runtime or listening-evidence drift.

## Shared-uniform sampling and bilingual listening — 2026-10-15

At base `3fc3de68`, the new diagnostic compares production Metal sampling
with the pinned MLX filters on the same F32 logits. It captures both input
arms, top-k order, full-vocabulary softmax/CDF, retained probabilities, exact
sample IDs, and rank-indexed common Gumbel noise. Generation controls upload
a common F32 uniform tape once, using the native Crystal RNG order: main,
unconditional repetition retry, then nine residual heads. MLX uses an explicit
inverse-CDF shadow for these controls. This isolates common-variate behavior;
it does not establish stock seeded MLX RNG parity. The reference keeps the
existing F32-promoted core and packed Q4 tensors, with checkpoint-dtype DAC.
These matched controls call the one-prompt `generate_codes` path. They do not
reproduce the default long-text orchestration of `Model.generate`.

The original thirty-word native clip and the matched reference have identical
prompt IDs and **448 complete 11-code frames**, about 20.8 seconds. Both have
the near-silent suffix from roughly 9.5 seconds. Their first emitted-code
difference is frame 448, after the quiet transition. A denser teacher-forced
onset replay captures steps 0/1/2/4/8/16/32/64 and every step 96–223:
6,127,752 actual-generation values pass unchanged `atol=6e-5, rtol=3e-4`.
The terminal prediction at step 224 is reported separately as counterfactual.
This excludes native/reference code divergence as a necessary cause of the
original quiet onset under the common-uniform control. It does not identify
why the checkpoint/reference trajectory enters the quiet region.

Actual random draws and main-token history explain persistence of the
repeated token. At frame 198, token 250598 is new to the 16-frame window and
the retry is unused. At 199, the first and retry samples both select 250598.
At 208, both ordinary and retry filters retain only that candidate. Upstream
repetition-aware sampling makes one retry and accepts its result even when
it repeats; it is not an exclusion mask. All nine captured RAS decisions
match the native generation report. The repeated run starts after the signal
activity transition, so this explains persistence, not the onset of silence.

Three native randomized variants use the exact requested text:

> Я живу в Голливуде, Флорида, США. I live in Holliwood, Florida, USA.

The user hears both languages in seeds 1 and 2, and only Russian in seed 3.
This is direct human content feedback, not an energy-based transcript or a
controlled speaker-identity comparison. Seed 3 and the matched MLX shadow
have identical prompt IDs, all 179 emitted frames, and EOS at the same point.
The local ASR follow-up above conflicts with the Russian-only interpretation;
English omission is not established for this file. Code equality and EOS do
not resolve content. Random seeds do not guarantee distinct, reproducible
speaker identities.

| Prompt / seed | Native / MLX frames | Identical complete prefix | Termination |
| --- | --- | --- | --- |
| Original whisper + laughter / 1 | 489 / 513 | 448 | EOS / EOS |
| Plain bilingual / 1 | 197 / 197 | 197, entire output | EOS / EOS |
| Plain bilingual / 2 | 148 / 321 | 12 | EOS / EOS |
| Plain bilingual / 3 | 179 / 179 | 179, entire output | EOS / EOS |
| Bilingual whisper wrapper / 1 | 384 / 213 | 26 | cap / EOS |
| Bilingual laughter cue / 1 | 384 / 384 | 198 | cap / cap |
| Bilingual whisper + laughter / 1 | 287 / 287 | 287, entire output | EOS / EOS |
| Original thirty words, both cues removed / 1 | 768 / 768 | 768, entire output | cap / cap |

Removing both cues from the original thirty-word text also produces a quiet
suffix: every half-second window from 5.5 seconds onward peaks at one PCM16
unit in both outputs. All 768 frames agree, and both hit the configured cap.
This is a prompt ablation, not a speech-content or termination pass. Native
wrapper time is 279.029 s and MLX 101.881 s on the shared host; this is not a
matched performance experiment. Full-text listening for this pair is pending.

Source inspection identifies an API-path difference for the long prompts:
the styled original is 360 UTF-8 bytes and the no-cue variant is 332, while
the pinned high-level `Model.generate` defaults to 300-byte text chunks with
generated-code history. Its first chunk contains the first three sentences
(265/237 bytes respectively), followed by the 94-byte final sentence. The
matched one-prompt controls bypass this split. The plain bilingual input is
only 93 bytes, so this difference does not explain its seed-3 omission.

Two additional controls now call official `Model.generate` with its default
splitter/history policy, preserving the same F32 core, Q4 weights, row-prefill,
sampling settings and common uniform prefix. The no-cue prompt emits 515 + 89
frames (23.917 + 4.133 seconds); the styled prompt emits 741 + 78 frames
(34.412 + 3.622 seconds). All four chunks terminate by EOS. In the no-cue
first chunk, every half-second from 7 seconds onward peaks at at most five
PCM16 units; the styled first chunk has peaks of 37–51 from 12 seconds onward.
Thus this high-level orchestration does not eliminate the long quiet suffix
inside the first chunk in these runs. The second chunks have active signal,
but signal activity does not establish word delivery. The user confirms that
styled chunk 0 omits the cat sentence; listening for the remaining chunks is
not independently resolved. The within-text comparison changes prompt splitting, history and DAC segment
boundaries together; the styled and no-cue arms do not isolate chunking from
markup. The 768-frame cap is per chunk versus per prompt, so total termination
budgets are not matched. Wrapper times of 121.732 / 162.294 seconds are shared-host observations,
not a performance comparison or promotion.

The opt-in `--chunk-text` reference adapter uses the official host prompt
builder with zero-code history placeholders, then replaces only guarded audio
spans using GPU MLX operations. Actual history-code values remain on GPU during
generation. After terminal evaluation, all four prompt tensors match the
restored official `build_chunk_prompt` byte-for-byte. A single uploaded tape
is consumed across the chunks; `--max-frames` is per chunk and capture-step
indices are global. The native first styled chunk has the same text, 71 prompt
IDs, model path, seed, settings and initial F32 variates as reference chunk 0.
It emits 427 frames versus 741, sharing 111 complete frames before columns 9/10
diverge at frame 111. That new fork is not numerically localized or a parity
pass. Native stops before residual sampling on terminal EOS, while MLX samples
all residual heads before its EOS check. The tape has an 11-slot stride;
native uses 11 slots on non-EOS frames and two on EOS, versus 11 on MLX EOS.
No native second-chunk or cross-chunk draw-consumption parity is claimed.
This control tests pinned high-level text orchestration, with a sampler shadow
and custom GPU history adapter; it is not stock full-program execution.

The short whisper control isolates another failure mode. All 1,216,539
captured actual-generation intermediate values through step 26 pass the
existing tolerance, yet residual codebook 4 diverges at step 26. Candidates
135 and 764 reverse rank by only 1.9–3.8e-6 in temperature-scaled logits.
The shared draw 0.33058440685272217 selects rank 6, away from a CDF boundary.
Both GPU samplers choose 135 on the native vector and 764 on the MLX vector.
The trajectory fork is explained by a near-tie input ranking change; this
does not localize the arithmetic operation that produced it. No tie band,
precision, tolerance, or production filter was changed.

The fixed-logit onset probe executes 199 cases, including one synthetic top-k
boundary tie and both native/reference vectors at nine steps. Exact selected
IDs, top-k order, kept masks/counts and common-noise IDs agree in those cases.
The strict gate nevertheless fails: two upstream full CDFs decrease by one
F32 ULP in their discarded tails. The invariant remains `1e-7`; violations
are recorded rather than silently relaxed or allowed to abort the remaining
diagnostics. Malformed/nonfinite data and gross arithmetic corruption still
reject. Agreement of selected IDs is narrower than full sampler parity.
The separate whisper fixed-logit probe passes all 67 cases, including 3
actual RAS decisions, with no CDF invariant failure.

The verifier independently enforces canonical normal/retry settings and
reopens both source traces by path and SHA-256. It re-derives the captured
vectors and input comparison statistics, rejecting unbound trace cases or
edited vectors even when their probe outputs agree. Twenty sampler CPU tests
and nine generation-control tests pass, including these negative mutations,
history-span guards and per-chunk/global capture-budget checks;
Python ASTs, Crystal formatting and `git diff --check` pass. A correlated
source adversary verdict is ROBUST for consistency with the retained local
artifacts. Strict sampler parity and a unique causal explanation remain
unsupported. Trace metadata alone does not authenticate all weight shards
or attest execution; the external receipts and checkpoint manifest retain
their stated local provenance limits.

Same-code full WAV comparisons retain the failed one-LSB DAC gate:
plain seed 1 maximum/RMS delta 16/0.618570 PCM16 units, seed 3 10/0.464103,
and whisper + laughter 10/0.228216. Lengths and declared 44.1 kHz clocks agree.
The full thirty-word no-cue pair has maximum/RMS delta 3/0.182296, also failing
the one-LSB gate.
These bounds do not establish general codec, pitch, or voice-identity parity.
Energy analysis is never used to certify delivery of either language.

The new native sampler probe queues all cases and production sample calls on
GPU, with zero pre-drain reads or completion waits, one submission and one
final drain. Terminal serialization adds no submission or completion wait.
The existing scalar EOS/status boundary in generation remains; the new tape
and captures add no per-step host reads. MLX retains its official row-prefill
evaluation, frame-EOS evaluation/read and per-chunk decoder-length scalar read.
The history adapter adds no in-loop code-value readback by source inspection.
Prompt/provenance preparation
and terminal reports/WAV serialization run on CPU.

Risk is CAUTION, limited to diagnostic examples, reference helpers and tests.
Production arithmetic, defaults, graph scheduling, MAP_SHARED and precision
are unchanged. Rollback is to revert this diagnostic commit. Local artifacts
and command receipts are in `sampler-voices-20261015/`; the curated index is
`.landmark/evidence/drama3-sampler-voices.json`. All nine local checkpoint
files are SHA-bound; remote model revision identity is a label, not an
authenticated-byte claim. Refresh on source, input, checkpoint, reference,
precision, runtime, compiler, device/OS or artifact drift. Full text delivery,
the unique silence cause, strict sampler/DAC parity and the prior MoE39
hidden[33] gate remain open. No performance or LTP/WBA promotion is claimed.

To reproduce the sampler diagnostic, use `compare_sampler_common.py prepare`
with native and MLX replay traces, a native-order random tape and selected
`--step` values. Run `examples/drama3_sampler_compare.cr` on the resulting
fixture, merge the pinned GPU reference with `sampler_common_reference.py`,
then run `compare_sampler_common.py compare`. The CLI help documents the
exact input/output flags. The tape is generated by
`Random.new(seed)` and `Array(Float32).new(frame_cap * 11) { random.rand.to_f32 }`,
with no skipped retry draw. The `uniforms` JSON array is uploaded once by
`generate_mlx_control.py --random-tape`; omit that option for stock MLX
sampling. Add `--chunk-text` for the bounded official text orchestration control
and budget the tape for `max_frames * chunk_count * 11` values. Source/build
receipts are part of the evidence index: shader
hashes inside a probe report alone do not attest the Crystal harness build.

## Listening feedback and decoder clock — 2026-10-15

After listening to `generate-native-seed1.wav` and `generate-f32-control.wav`,
the user reports that **both stop delivering speech after the whispered
phrase**. The user also hears the native whisper as faster and higher. This
closes the earlier listening question for these two files as a negative
full-text-delivery result; the other reference controls have not received
the same listening check. Pitch remains a perceptual observation, not a
measured fundamental-frequency difference.

Read-only analysis at base `f908862d` finds that all three saved WAVs, including
`upstream-codec.wav` decoded from the native codes, are mono PCM16 at 44,100 Hz
with exactly 2,048 samples per code frame. The same-code pair has identical
1,001,472-sample lengths and maximum/RMS differences of 7/0.168956 PCM16 units.
This excludes a substantial decoder or declared WAV clock change for that
pair, while retaining the failed one-LSB codec gate. It does not establish
general codec parity or rule out a separate playback-device issue.

With 20 ms windows and RMS above 32 PCM16 units, the last active window ends
at 9.02 s in native and 10.36 s in the independently sampled F32 MLX clip.
The roughly 1.34 s difference measures signal activity, not word timestamps,
uniform time scaling, or speech rate. Native and MLX choose different main
tokens in frame 0: 250521 versus 248764. There is no identical generated
frame prefix. Text, all 87 prompt IDs and explicit seed/temperature/top-k/top-p
match, but RNG implementations differ and per-step draws were neither
aligned nor captured. These files cannot isolate numerical error as the
cause of the perceived voice or timing difference.

The analysis and hash-bound input/WAV identities are local in
`~/.cache/cogni-ml/drama3-runs/listening-feedback-20261015/analysis.json`
(SHA-256 `44f78b93d5cb1884595e5a48ce85040c5a5f8bafc8333e462794b9b224470751`).
The NumPy probe ran through `scripts/run_safe.sh` with a 15% reserve; a
separate standard-library PCM calculation reproduced the three RMS-window
counts and endpoints, and checked hashes, prompt IDs and settings. A direct
check also matches all ten codebook columns in the saved MLX codec input to
the 489 native frames, and checks the two WAV headers for equality. This
inspection reuses the earlier generated artifacts; it is not a new model run.
The missing continuation and its root cause remain open. Refresh these
file-specific findings on artifact drift. The next generation discriminator
still needs a common prefix, equal reference precision and explicitly aligned
sampling inputs, rather than equal integer seeds alone.

## Thirty-word whisper/laughter measurement — 2026-10-15

At base `a21ee640`, the native resident path was timed on Apple M2 Max / 64 GiB,
macOS 26.6.2, Crystal 1.21.0 and the same pinned Q4 checkpoint. The input has
30 whitespace-delimited spoken words, excluding the `<whisper>` tags and
`[laughs]` cue:

> <whisper>Тише, кажется, за дверью снова шорох. Давай подождём немного и послушаем, что происходит.</whisper> [laughs] Ах, это всего лишь наш кот уронил коробку! Ну и напугал же он нас сегодня, маленький проказник.

Settings: temperature 0.8, top-k 30, top-p 0.9, dense `tree`, expert down
`native`, MAP_SHARED off and graph depth 2. The official prompt builder and
native report have identical 87 prompt IDs. This is a single-prompt native
smoke; upstream's high-level API normally splits at 300 UTF-8 bytes, whereas
this input has 360 bytes. Tags are ordinary model input, not an implemented
native style controller. The user reports that the whisper sounds excellent
but the rest of the requested speech is absent.

| Seed / frame cap | Result | WAV length | Whole wrapper wall time | Native generation / DAC phase |
| --- | --- | --- | --- | --- |
| 0 / 512 | 512 frames, capped | 23.777 s | 192.047 s | 135.092 / 56.013 s |
| 0 / 768 | 768 frames, capped | 35.666 s | 269.553 s | 188.797 / 79.596 s |
| 1 / 768 | 489 frames, EOS | 22.709 s | 186.776 s | 133.147 / 52.538 s |

The selected seed1 run takes 185.727 s inside the native timer. The wrapper
includes process/preflight overhead; the native timer starts after WeightStore
construction and prompt preparation. Phase timers use CPU monotonic clocks and
add no GPU operation, read or wait. The reported initialization phase is
resident object construction, not complete eager model loading; first-use work
can occur in generation. These are individual shared-host runs with uncontrolled
cache state, not a matched speedup or steady-state latency result. The two
seed0 runs have an identical 512-frame prefix.

**Full-text delivery failed.** EOS is not evidence that all 30 words were
spoken. From 9.5 s to the end, every half-second window peaks at only 1–2
PCM16 units. Official MLX DAC decoding of the same 489 code frames is equally
quiet there. Across the entire 1,001,472-sample clip the DAC difference is
maximum 7 PCM16 units, RMS 0.168956; the existing one-LSB gate fails. The
second half has maximum DAC delta 1 unit and RMS delta 0.046976. This localizes
the gross silence to the code stream, without claiming full codec parity or
identifying why the generator chose those codes. A 65-frame repeated main
token run begins at frame 198 (nominal 9.195 s); it is an observation, not a
proved cause. PCM activity is not speech recognition or word alignment.

The new common-prefix probes consume all 489 native frame rows on GPU and
compare backbone hidden/main logits and tower hidden/nine codebook heads.
They capture steps 0/1/2/4/8/16/32/64/128/192/256 plus every step 361–489:
140 steps, 1,670 stages and 6,269,068 F32 values per arm. The pinned official
MLX architecture uses rowwise prefill and F32-promoted floating checkpoint
tensors, with packed integers preserved. This is the existing numerical
reference precision, not stock BF16 upstream generation. The native frame
prefix is teacher-forced in both implementations; sampling is disabled in
these replays.

At unchanged `atol=6e-5, rtol=3e-4`, both native down and opt-in `mlx_q4` first
fail at step 447; all captured steps through 446 pass. Native has 1,051,036
failing values and maximum absolute error 0.861520767 at step 455 main logits;
the opt-in has 1,051,066 and 0.861519337. Main-logit top1 differs at steps 448
and 463 in both arms; minimum top30 set overlap is 28/30. These ranks describe
forced-prefix predictions, not newly sampled output. Dense-tail capture
reproduces all 146 overlapping sparse stages / 546,829 values bitwise in both
the native and upstream runs, controlling for capture perturbation on those
overlaps. The observed late failure around 20.8 s cannot explain the earlier
onset of silence; unsampled earlier boundaries and sampler behavior remain
open. It does not isolate a router, reduction or unique rounding operation.

A separate low-level official `generate_codes` control uses the same prompt
IDs and settings without forced codes. Its core parameters retain canonical
BF16/Q4 dtypes, with F32 DAC, and prefill is one row per chunk instead of the
official default 128. This bypasses high-level text chunking. The initial
official-loader attempt was stopped by run_safe at sampled free memory 13%,
below the authorized 15% reserve, without a behavioral result. Disabling the
MLX allocator's free-buffer cache let the retry complete in 46.820 s wrapper
time: 768 frames / 35.666 s, at the frame cap. After the initial second,
half-second PCM peaks range from 42 to 90 units. This does not establish
successful requested speech. Different RNGs and reference precision prevent
an exact native trajectory or silence-cause inference. The official loop
retains its per-frame EOS evaluation/read and DAC-length scalar read; this
control is not synchronization-free. Decode-call and terminal-evaluation
timings split lazy execution and are not comparable native phase latencies.

A second independently sampled control promotes core floating tensors to F32
using the replay's shardwise loader, preserving packed integers and the official
sampler/DAC. It returns 384 frames / 17.833 s before the cap in 72.111 s wrapper
time (40.215 s loading, 27.677 s generation). Its waveform has activity through
10.5 s, followed by half-second peaks of only 5–10 PCM16 units to the end.
There is also a quiet interval at 4–6 s. This is another incomplete behavioral
control, not a speech-content pass or a matched-RNG native comparison. The
official API does not expose its termination reason in this report. F32 alone
does not establish recovery of the requested text.

A third control uses the pinned public `Model.generate` path with the same
custom F32 core loader, default 300-byte text chunking and 128-row prefill.
It returns both expected chunks in 108.865 s wrapper time, including 45.113 s
loading and 61.867 s in the official iterator. The first chunk includes the
whisper, laughter cue and the sentence ending with "коробку!": 71 prompt rows,
534 frames / 24.799 s. Every half-second from 8 s to that chunk's end peaks
at only 20–28 PCM16 units. The history-conditioned second chunk has 633 prompt
rows and 114 frames / 5.294 s; it has active PCM through 4.5 s, followed by
a quiet tail. Neither chunk reaches its frame cap; the API does not report
the precise termination reason. Public chunking therefore does not remove
the long quiet tail inside the first chunk. This is a public-path control,
not a single-cause experiment: chunking, prefill and independent RNG change
the trajectory. PCM activity does not establish which words or styles were
delivered. Full speech delivery and the unique silence cause remain open.

The pinned port documents `[laughs]` as an inline cue. Its generic splitter
protects paired tags such as `<whisper>...</whisper>`, while the prompt builder
passes their text to the tokenizer unchanged; this is not an explicit style
controller or a source-backed guarantee of the model's learned tag semantics.
A fourth public-path F32 control removes only that wrapper, retaining every
spoken word, `[laughs]`, seed/settings and the two chunk boundaries. It returns
451 + 104 frames in 129.996 s wrapper time. The first chunk is 20.944 s, with
half-second peaks of only 1–4 PCM16 units from 7 s to its end; the second is
4.830 s, with activity through 4 s and a quiet tail. Both expected chunks
return before the cap. Removing the wrapper does not remove the quiet suffix
in this sampled control. The wrapper is therefore not necessary to reproduce
this symptom under these conditions; its possible contribution to the original
trajectory remains open. This single prompt ablation neither establishes
word delivery nor identifies the generator's silence or stop cause.

Native inference computations and state remain on GPU. The selected generation
records 3,928 control-read bytes and 4,027,404 terminal-read bytes; graph depth 2
drains with no pending work or retained bytes. Native diagnostic replays assert
zero pre-drain readbacks and no additional submissions/completion waits during
terminal serialization. MLX uses official per-row cache evaluation; its
internal backend submissions/waits are not instrumented. CPU performs prompt,
checkpoint, terminal WAV and report work.

Risk: CAUTION, bounded diagnostic and example CLI change. The CLI frame limit
now permits 1–1024 and rejects a resident prompt/frame bound exceeding the
existing 1024-row cache. Defaults and kernel arithmetic are unchanged. Rollback:
revert this measurement/probe commit. The important falsifier is a silent
suffix despite a non-silent global waveform and EOS, so neither global RMS nor
EOS is treated as a successful text-delivery gate. The strict comparator rejects
incomplete/malformed capture schedules, mismatched inputs/config/index, invalid
F32 data and widened tolerances. A native schedule bug that included checkpoints
above the frame count was caught before its GPU run and corrected.
The comparator checks capture contents and identity fields; it does not itself
attest source files or checkpoint shards. Exact trace hashes, primary-source
execution receipts and local checkpoint hashes are bound externally in the
evidence index. The older external control/activity helpers self-report hashes
matched to their final files; these are not independent launch-time receipts.
Observed guards: 18 focused Crystal generation/graph examples and 15 Python
capture/comparator tests pass; real CLI bounds and all 87 prompt IDs pass.
Mini-Quadrumvirate verdict: ROBUST for these bounded measurements and negative
gates; VULNERABLE for full-text delivery, which remains open. Correlated review
does not replace the executed controls.

Raw WAVs/traces and command logs remain local in
`~/.cache/cogni-ml/drama3-runs/whisper30-20261015/`; the curated index is
`.landmark/evidence/drama3-whisper30.json`. All actual cached checkpoint files
were hashed, with unchanged file metadata checked after the measurements;
this binds local bytes, not authenticated remote revision identity. Refresh
on source/checkpoint/input/reference/precision/runtime/compiler/device/OS or
artifact drift. Independent upstream sampling and speech-content verification
are separate from the common-prefix numerical gate. The next generation
discriminator needs equal reference precision, aligned random draws and
captures around the first trajectory divergence / silence onset. The original MoE39
hidden[33] gate and its next substitution probe below remain the return frontier.

## Native MLX Q4 arithmetic and full gate — 2026-10-15

At base `31920e41`, the opt-in native Metal down projection reproduces the
pinned MLX0.31.1 result bitwise on the retained row1/layer39 SwiGLU activation
and identical mapped packed weights. The original full-backbone gate still
fails one value at unchanged `atol=6e-5, rtol=3e-4`. This section supersedes
the older gate measurements below; production parity remains IN_PROGRESS.

`kernels/mlx_q4.metal` reproduces the fast Q4/group64 path: 16 activation
values per lane, packed-nibble input rescaling, `scale*dot + sum*bias` per
lane/group, 512-column block accumulation and `simd_sum`. It uses F32
arithmetic and GPU expert IDs; no dequantized weight expansion is introduced.
The separate source preserves the existing kernels' contraction settings.
The native two-dimensional dispatch permutes independent row/rank axes;
scalar and batched results are checked on the actual checkpoint. Source
expressions and `simd_sum` are not a disassembly proof of compiler contraction
or the hardware reduction tree. Upstream attribution is in
`kernels/mlx_q4.LICENSE`.

`DRAMA3_EXPERT_DOWN_REDUCTION=mlx_q4` selects only routed-expert down
projections; the value is captured when constructing the backbone. The default
remains `native`, with dense `tree` and MAP_SHARED off. Unsupported bit width,
group size, K tails or N tails reject before encoding. The admitted shape
requires affine Q4/group64, K divisible by 512 and N divisible by 8.

| Executed check | Observed result |
| --- | --- |
| Exact upstream activation, native MLX-style down | All 16,384 values across eight experts bitwise upstream; expert108 scalar and batched 2,048 values bitwise upstream |
| Repeat and retained-native controls | Candidate repeat bitwise; native down reproduces the previous three tolerance failures at expert108 469/766/1555 |
| Full upstream gate, candidate | 243 boundaries / 497,664 values; exactly one failure, row1/layer39.hidden[33] |
| Full upstream gate, default native | Same one failure as before; all 497,664 captured native values bitwise equal to the retained native trace |
| Synthetic native guards | 24 Crystal examples pass, including distinct K=1024 blocks, odd scale/bias offsets, F32/BF16/F16 metadata, GPU IDs, scalar/batch and frozen selector |
| Adjacent reference/capture guards | 32 Python tests pass at unchanged tolerance |

At hidden[33], upstream is `0.9219169616699219`; the candidate gives
`0.9215669631958008`, absolute error `0.00034999847412109375` against allowance
`0.00033657508850097655`. The earlier native error was
`0.00081634521484375`. The candidate is still outside the gate; neither the
smaller error nor synthetic exactly representable fixtures establish general
MLX bitwise equivalence or a complete correction.

All inference stays on GPU. `examples/drama3_mlx_q4_probe.cr` enqueues native,
candidate, repeat and scalar arms before one final drain. There are zero
pre-drain readbacks/submissions/waits, one final submission/wait and no added
wait during terminal serialization (212,992 terminal bytes). The passive
buffer callback in `examples/drama3_backbone_gpu_trace.cr` captures all three
rows in one GPU command buffer, then reads 1,990,656 terminal bytes and 972
finite-flag control bytes after that drain. It replaces the per-boundary host
callback for this diagnostic. CPU checkpoint/provenance preparation and
post-drain report analysis are outside inference.

A bounded follow-up uses the existing MoE trace on the exact upstream
normalized input. Gate/up, SwiGLU, candidate down, routed sum and shared stages
all pass fixed tolerance, but are not generally bitwise upstream. The first
failing official boundary is final `moe_output_official`, coordinates 1012 and
1111. The full run has a different normalized input and is explicitly
unqualified as an exact-input comparison. An initial depth0 MoE comparison
exited 2 because that comparator requires depth2; the depth2 rerun is valid
and exits 1 for those two numeric failures. Capture/no-capture control matches
8,192 values bitwise, with one final wait and no intermediate readback.

Risk: CAUTION. Rollback is disable the opt-in or revert this atomic change.
DoD was the real exact-input probe, original full comparator and adjacent
guards through `scripts/run_safe.sh` at sampled reserve 15%, with profiling
unset. The expected failure mode was wrong affine grouping, stride, mapping,
contraction or a misleading local pass; actual checkpoint controls, independent
synthetic indexing fixtures and the unchanged full gate discriminate these.
Mini-Quadrumvirate verdict: ROBUST for measured local down reproduction and
the bounded negative full-gate result; a production parity claim is VULNERABLE.
No latency, default or LTP/WBA promotion is claimed.

Exact commands, source/artifact hashes, runtime and next signal are indexed by
`.landmark/evidence/drama3-native-mlx-q4.json`; raw artifacts remain local in
`~/.cache/cogni-ml/drama3-runs/mlx-q4-20261015/`. Actual 14 target-MoE tensor
bytes match the pinned upstream digest before the probe. The historical full
reference manifest binds reference/input/source/config/index, but does not
hash every backbone weight shard; it is not an authenticated all-weight
certificate. Refresh on source, checkpoint, input, runtime/reference,
precision, compiler, device/OS or artifact drift.

Next falsifier: on common normalized input, substitute upstream gate/up-derived
activations and shared output independently on GPU, then compare the official
final MoE sum. Passing separate relative-tolerance stages does not guarantee
passing their sum near cancellation. Keep the original hidden[33] full gate
as the admission check before promoting any correction.

## Expert-108 quantized versus decoded F32 down — 2026-10-15

The common-input GPU experiment reproduces the upstream quantized down return
bitwise, while `mx.dequantize` followed by F32 `mx.matmul` differs at the
unchanged tolerance `atol=6e-5, rtol=3e-4`. This comparison resolves the next
falsifier from the MoE investigation below; it does not fix production parity.
No native arithmetic, precision, default or scheduling changes are included.

`scripts/drama3/probe_mlx_expert_down.py` loads the retained row1/layer39
SwiGLU return, preserves all eight explicit expert IDs and selects expert 108
at rank 7. It verifies the actual 14 target-MoE raw tensors and config/index
against the prior trace before importing MLX. The three down tensors retain
packed uint32 Q4 weights and BF16 scales/biases promoted to F32 exactly as in
the pinned reference. Actual lazy-array shape/dtype assertions pin direct
matmul input to `[1,512]`, decoded weights to `[2048,512]`, and outputs to F32.
The normalized-input fixture and actual expert activation have separate hashes.

| GPU arm or control | Observed comparison |
| --- | --- |
| Official-shape eight-expert `gather_qmm` | All 16,384 down values bitwise equal to retained upstream |
| Single-expert `gather_qmm` and `quantized_matmul` | Each bitwise equal to upstream expert 108, all 2,048 values |
| `dequantize → F32 matmul` versus upstream | Two tolerance failures: 469 and 766; maximum absolute difference 0.00390625 |
| Retained native down on exact upstream activation versus decoded F32 matmul | All 2,048 values pass unchanged tolerance; not bitwise equal |
| Retained native down versus quantized arm | Three failures: 469, 766 and 1555, reproducing the prior result |
| Explicit GPU uint4 `scale*q+bias` decode versus `mx.dequantize` | All 1,048,576 F32 weight bit patterns equal; zero maximum error |

At coordinate 33, upstream/quantized gives -9.041109085083008, decoded F32
matmul gives -9.04194164276123, and retained native gives
-9.041943550109863. Native and decoded F32 differ by only 1.9073486e-6 here.
The maximum absolute error in the table occurs on a large output and is not
the tolerance-failure criterion; every output is checked separately.

This rejects a mismatch between the two tested decode formulas and demonstrates
a difference between fused quantized dot arithmetic and decoded-weight F32
matmul on these exact bytes. It does not identify a unique rounding operation,
establish which result is more accurate, or prove a complete full-backbone
counterfactual. The explicit decode control implements the native-style
formula in MLX; it is not a capture of the native Metal decoder itself.

The pinned MLX v0.31.1 source (commit
`ce45c52505c8158ea48d2a54e8caae05efd86bfe`) factors group scale/bias outside
partial packed dots and uses SIMD reduction in its fast Q4 QMV path, whereas
the native kernel multiplies individually decoded weights and sums group
partials. This makes affine regrouping and reduction order the next target.
FMA remains unisolated. Source dispatch predicts ordinary QMV/GEMV for the
actual M=1 shapes, with no NAX/TF32 route; the run also sets
`MLX_ENABLE_TF32=0` before importing MLX. Kernel names and internal backend
wait counts were not profiled. Primary source/hash references are in the
local `source-audit.json` indexed by the evidence packet.

All projection, decode and weight-comparison arithmetic runs on GPU. Six
terminal result/metric reads follow one explicit `mx.eval`; no intermediate
GPU-to-host activation read or explicit synchronize call occurs in the probe.
This source/API audit does not measure MLX's internal submissions or waits.
CPU work is checkpoint loading/provenance, fixture setup and terminal analysis.
The original 497,664-value full gate remains open at hidden[33]; no full-model
run or latency claim is added by this diagnostic.

Risk: CAUTION, bounded reference probe only. Rollback is revert this diagnostic
commit. DoD: execute the real probe through `scripts/run_safe.sh` with the 15%
sampled reserve, reproduce the official full-eight capture bitwise, record all
2,048 target values for each arm, then run the negative provenance/tolerance/
readback guards and adjacent comparator/capture tests. Observed: probe exit 0,
32 Python tests pass, and all decoded-weight bits match. A second GPU run after
closing a replay-source hash guard reproduces all 22,528 projection values
bitwise across the four arms. Numeric differences
between diagnostic arms are report data, not an exit-0 assertion of parity.
Mini-Quadrumvirate verdict: ROBUST for this bounded discriminator, with the
full-backbone parity gate still open. Review and exact commands/source/artifact hashes are indexed
in `.landmark/evidence/drama3-down108-qmm.json`; raw artifacts remain local in
`~/.cache/cogni-ml/drama3-runs/down108-20261015/`. Refresh on source, checkpoint,
activation, reference/runtime, precision, device/OS or artifact drift.

Next falsifier: reproduce the pinned low-M packed Q4 dot's affine grouping and
reduction in an opt-in GPU diagnostic, preserving Q4 storage, F32 arithmetic
and one terminal drain. Qualify exact-input down first, then rerun the original
full upstream gate and adjacent guards before considering production adoption.

## Layer-39 MoE discrepancy investigation — 2026-10-15

Status: diagnostic localization observed; the production parity correction is
IN_PROGRESS. The retained full-backbone gate at base `7bf40df8` still fails at
exactly one of 497,664 values: row1/layer39 hidden[33], native
0.9211006164550781 versus upstream 0.9219169616699219. Absolute error is
0.00081634521484375; the unchanged allowance is 0.00033657508850097655.
Dense tree/SIMD arms are identical. Neither dense nor MAP_SHARED defaults are
promoted. No production arithmetic or Metal kernel changes in this diagnosis.

The probes retain actual GPU inputs, router logits, selected weights, expert
gate/up/activation/down returns, official routed/MoE returns and residual
boundaries. Expert rows are compared by explicit ID while retaining raw rank
order. Weighted terms and prefix sums computed by separate diagnostic kernels
are labelled auxiliary; they do not substitute for official return values.
The pinned MLX model runs with floating checkpoint tensors promoted to F32,
as in the previous reference; this is not stock BF16 execution parity.

| Boundary on exact upstream normalized input | Observation |
| --- | --- |
| Router logits | Maximum absolute difference 7.153e-7; passes fixed tolerance |
| Selected experts | Same eight IDs; upstream raw order 159,194,245,154,230,191,171,108; native order 108,171,191,230,154,245,194,159 |
| Selected weights by ID | Maximum absolute difference 5.960e-8; passes |
| Expert gate/up and activation | All pass; equality is tolerance based, not bitwise |
| Expert down returns | First failing stage; expert 108 fails at coordinates 469 and 766 |
| Official routed sum | Fails at coordinates 469,766,791 |
| Official MoE return | Fails at coordinates 1012,1111,1191,1640 |

The strongest intervention bypasses gate/up/SwiGLU: replay the **exact stored
upstream activation** through the existing native affine-Q4 down kernel. For
expert 108 at coordinate 33, native is -9.041943550109863 versus upstream
-9.041109085083008, delta -0.0008344650268554688. That delta survives identical
activation bytes. The same native down kernel replaying its own captured
activation reproduces all 16,384 captured down values bitwise.

Five GPU reduction arms separately hold terms, weights and order fixed.
Native reduction of upstream down rows and weights in upstream order matches
the actual upstream routed sum **bitwise across all 2,048 values**. Reordering
fixed terms with their paired weights changes bits at other coordinates but
causes no tolerance failures and no change at coordinate 33. Native terms
with upstream weights/order retain a -0.000834465 coordinate-33 sum delta;
native weights change that by only +0.000000954. Native terms/weights in
native order reproduce the captured native sum bitwise. These are bounded
interventions on this workload, not a universal claim about MLX reduction.

Full-native normalized input differs in 1,936/2,048 F32 bit patterns (maximum
1.90735e-6), so its stages are not qualified as an exact-input comparison.
At original coordinate 33, native-minus-upstream deltas are +0.000015259 for
the attention residual, -0.000831604 for MoE, and -0.000816345 for hidden.
Residual cancellation around 13.28 + (-12.36) exposes the final violation.
Replacing full-native input with upstream input changes native MoE[33] by
only about 1.907e-6. This supports the down projection as the dominant
contribution; it does not prove a complete counterfactual explanation of
every full-pass difference.

Capture controls: all 243 upstream full-backbone arrays match the retained
no-capture reference bitwise; detailed upstream full and isolated MoE stages
match bitwise. Native detailed/no-detailed captures match four output vectors
(8,192 values) bitwise, and three final rows plus the selected residual/hidden
match the retained native baseline bitwise. In the real trace and ablation,
graph submissions/waits are zero before the final drain and one afterward,
unchanged by serialization. Guard/status and diagnostic readbacks occur only
after that drain. No ordinary-inference diagnostic allocations, dispatches,
readbacks or completion boundaries are added when hooks are unset. Diagnostic
calculations and reference inference stay on GPU; CPU code handles checkpoint
loading/provenance, fixture indexing and terminal report analysis.

Both native examples hash the actual 14 target-MoE mapped tensor payloads from
their loaded `WeightStore`, plus config/index bytes, before GPU context
creation. Their target digest matches upstream:
`648d4c736dd60b6c077c032565aa647b6e90f6fd96d2ee957f22cbcd69a889dc`.
Checkpoint files must remain immutable during a run. This binds the selected
layer's weights, not every checkpoint shard or authenticated remote contents.
Final release checks pass 21 Crystal examples and 21 Python tests (20 report
guards plus one upstream capture regression); both real GPU examples exit 0.

Risk: CAUTION. Rollback is unset diagnostic callbacks or revert this diagnostic
commit. Tolerances remain `atol=6e-5, rtol=3e-4`; Q4 weights, floating precision
and ordinary inference scheduling are unchanged. Runs use the 15% sampled
reserve and do not modify unrelated processes. `DRAMA3_PROFILE` must be unset
(a value of `0` still enables profiling).

Diagnostic DoD: reproduce the original failure, qualify capture fidelity and
single-drain counters, compare exact-input stages by ID, then intervene on
down input and summation order. The observed first failing boundary is the
expert down projection. The mini-Quadrumvirate adversary verdict is ROBUST for
that bounded localization. Exact Q4 decode/product/reduction cause and the
production parity fix remain open. The down-path experiment above tests the
next falsifier on the same activation and raw weights. The subsequent pinned
MLX core source audit strengthens the affine-grouping/reduction hypothesis;
the exact rounding suboperation still needs a controlled GPU intervention.

Reproduction tools: `scripts/drama3/replay_mlx_moe.py` (full/isolated upstream),
`examples/drama3_moe_trace.cr`, `examples/drama3_moe_ablation.cr`, and
`scripts/drama3/moe_compare.py`. The comparator's exit 1 is the expected
preserved numerical mismatch; malformed pins, shapes, capture provenance,
expert maps or graph/readback controls are rejected separately. Run native
examples with `--release -Ddrama3_metal`, `DRAMA3_BACKEND=metal`, graph depth 2
and profiling unset through `scripts/run_safe.sh`. The exact commands,
test signals, source/artifact hashes and remaining frontier are indexed in
`.landmark/evidence/drama3-moe39-diagnostic.json`. Raw artifacts remain local
under `~/.cache/cogni-ml/drama3-runs/moe39-20261015/`; refresh on source, model,
reference/runtime, input, precision, compiler, device/OS or artifact drift.

## Dense backbone projection experiment — 2026-10-14

Status: opt-in candidate implemented; default promotion withheld. Measure the
actual checkpoint shapes of the shared-input `qkv/z/b/a` projections before
promoting an optimization. The baseline dense
kernel uses 64 lanes, lane-strided FP32 accumulation and a fixed binary tree
with seven threadgroup barriers. Candidate reductions must preserve every
lane's product/add order and that tree, raw BF16/F16/F32 decoding, GPU input
and output allocations, mapped-weight ownership, and command scheduling.

Risk: CAUTION. The strongest failure is a SIMD reduction that changes the
addition tree or assumes a SIMD width without a guard. First require bitwise
baseline/candidate equality on cancellation, tails, unaligned offsets and real
checkpoint shapes. Measure paired AB/BA GPU command intervals with preuploaded
inputs, depth two and only the final pair drain; reject additional submissions,
waits or reads. Retain the baseline selector for rollback. Promotion requires
repeatable projection improvement, unchanged full-generation WAV/codes/EOS and
readback/graph ledger, and refreshed pinned upstream intermediate comparisons.
The missing temporary reference was reconstructed from pinned MLX-audio
revision `e7488935ec2ebfafd38ef2b58e290d79f3d05dc7`, with MLX 0.31.1.
No new quantization or precision path is admitted. MAP_SHARED remains
default-off and outside this experiment.

DoD: release-build and run `examples/drama3_dense_benchmark.cr`, qualify its
paired timeline/counter/output records, execute the focused dense and resident
graph specs, replay backbone intermediates against pinned MLX-audio and compare
matched baseline/candidate full generation. Expected signal: bitwise native
outputs, unchanged upstream tolerances, zero intermediate waits/readbacks, and
a measured improvement without a material full-pipeline regression. If the
candidate fails, keep the baseline and pivot using the measured discriminator.

The shuffle-only candidate did not improve whole-generation GPU time. The
bounded follow-up uses an aligned BF16 `ushort` load in the candidate only,
guarded by dtype and even byte offset; odd offsets and other dtypes retain the
original decoder. Before the final full-generation matrix, the admission rule
is: all three real input rows must show a paired median kernel speedup above
1.03 for each projection, native bitwise/upstream gates must pass, and matched
full-generation GPU median candidate/baseline ratios must be at most 1.01 at
depths zero and two. Host wall time remains descriptive on this shared host.

The checkpoint header audit finds all 120 projections across 30 linear blocks
are dense BF16, have even mapped byte offsets, and occupy 1,517,813,760 bytes.
The layer-0 shapes are `qkv=[8192,2048]`, `z=[4096,2048]`, and
`b/a=[32,2048]`. The candidate keeps FP32 products/accumulation and the exact
64-lane summation tree. With 32-lane SIMD groups, one threadgroup barrier
publishes partials and shuffle instructions implement the remaining tree.
Other SIMD widths retain the shared-memory tree. Only aligned BF16 uses the
16-bit load; other dtypes and odd byte offsets retain the original decoder.
The non-32-lane branch is source-reviewed, not exercised on this M2 Max.

Three successive real prompt embeddings (token IDs `248045,8678,198`) were
replayed through upstream layer-0 normalization to obtain the timed inputs.
Each input has two warmups and seven alternating AB/BA measured pairs per
projection. Each arm repeats `qkv/z` eight times and `b/a` 1024 times in one
command buffer. Both buffers submit at graph depth two before one final pair
drain. GPU times are command-buffer timestamps, amortized over calls, not
isolated dispatch latency. These are hot, repeated-input microbenchmarks.

| Projection | Row 0 paired median speedup | Row 1 | Row 2 |
| --- | ---: | ---: | ---: |
| qkv | 1.083x | 1.082x | 1.068x |
| z | 1.101x | 1.095x | 1.109x |
| b | 1.591x | 1.587x | 1.635x |
| a | 1.581x | 1.611x | 1.578x |

Every measured pair improves in this bounded sample. A separate tree/tree
control gives paired medians 0.994, 0.997, 0.998 and 1.002 respectively. All
1,474,560 output elements per invocation match bitwise after drain. Each
invocation has 72 command buffers, maximum pending two, final pending/retained
zero, zero intermediate waits/readbacks, and 11,796,480 terminal verification
bytes. There are no new production submissions, completion boundaries,
readbacks, copies, scratch allocations or cache ownership rules.

The standalone upstream comparison passes all 37,056 values across three rows
and four projections at the unchanged `atol=6e-5, rtol=3e-4`. Maximum absolute
errors are `qkv=3.815e-6`, `z=2.027e-6`, `b/a=9.537e-7`; tree/SIMD match
bitwise. The reference promotes checkpoint floating tensors exactly to F32,
preserving packed integers, and executes the pinned MLX model. This is not a
claim of bitwise equality with stock BF16 execution. Fixtures bind input bytes,
source/checkpoint revisions, selected tensor hash and shape/dtype metadata;
they do not authenticate edited logs or hash the entire checkpoint.

Full native backbone traces also match bitwise: 243 tensors, 497,664 F32
values. However, the refreshed full upstream comparison fails for **both**
tree and SIMD at `backbone.row1.layers.39.hidden[33]`: MLX `0.921916962`,
native `0.921100616`, absolute error `8.164e-4`, allowed `3.366e-4`.
This is the sole out-of-tolerance coordinate. A fresh upstream run without
dense capture reproduces all 243 reference tensors exactly, excluding the
capture instrumentation as the cause. The discrepancy is downstream of the
last attention residual, in the existing MoE/residual path. Different expert
ordering/reduction is a source-supported hypothesis, not an established root
cause. Next discriminator: exact row1/layer39 router logits, expert IDs/order,
weighted expert outputs and partial sums. Do not widen the tolerance.

`DRAMA3_BACKBONE_DENSE_REDUCTION=simd` selects the candidate only for these
four projections in all linear blocks; rollback/default is `tree`. Generic
`DRAMA3_DENSE_REDUCTION` independently defaults to `tree`. The generation JSON
records the frozen, actual selectors, preventing a vacuous two-baseline timing
comparison. Default promotion remains blocked by the full upstream gate even
if the isolated performance and native bitwise gates pass. The existing
mapping-promotion frontier remains separate and default-off.

Matched release full generation (`Hello.`, seed zero, max 128 frames,
MAP_PRIVATE, profiling off, GPU timeline on) completed two warmups, four
AB/BA pairs at depth two and two at depth zero. Median paired
candidate/baseline ratios are:

| Graph depth | Pairs | GPU command sum | Inference elapsed | Whole process |
| --- | ---: | ---: | ---: | ---: |
| 2 | 4 | 0.99239 | 1.01218 | 1.02404 |
| 0 | 2 | 0.97089 | 0.88189 | 0.90373 |

The GPU non-regression gate passes, but these whole-process samples do not
establish a general latency improvement: depth two is 2.4% slower in process
time on the shared host. Host interference and pipeline setup have not been
separated. Do not equate an isolated 1.6x tiny-projection speedup with a
comparable whole-model gain. Earlier shuffle-only depth-two GPU ratios had
median 1.00402 and did not justify promotion.

All 14 wide-candidate runs reproduce the same 23 prompt IDs, 17 frame/code
rows, EOS stop, 34,816 waveform samples and WAV SHA-256
`3bcc331bd234ba6a3950d2122215bdc663aa4f7427a112e8f8df108085908a49`.
Both arms retain 152 control and 140,012 terminal readback bytes. Depth two
uses 73 submissions/completions/waits and maximum pending two; depth zero uses
90 and maximum pending zero. Every run drains pending/retained resources.
The initial harness incorrectly required equal submission counts across
depths; that rejected depth-zero run was excluded and the corrected matrix
checks the established per-depth counts and the actual selectors.

The final release GPU suite passes 15 examples covering dense BF16/F16/F32,
32 mapped dtype/width/offset fixtures, sliced input, strong cancellation,
selector/context ownership, resident operations and graph resource lifetimes.
Fourteen Python tests include negative controls for timing/counter/stream mutations,
stale fixtures, wrong model/revision/weight metadata, wrong candidate bits,
both native arms exceeding upstream tolerance, and attempted tolerance
widening. Adversary: ROBUST for bounded opt-in arithmetic, device ownership
and measured results; VULNERABLE for default promotion and general process
speedup. Correlated agent source reviews support, but do not replace, the
executed GPU and pinned-reference checks.

A separate warm-driver pipeline diagnostic records 0.263 ms total library,
function and pipeline creation for `dr3_dense_simd`; this small warm setup cost
does not explain the observed process-time difference. It is not a cold-driver
startup measurement. The diagnostic uses the existing host pipeline profiler
and adds no GPU synchronization.

Canonical evidence and executable commands are indexed in
`.landmark/evidence/drama3-dense-backbone.json`; raw local artifacts are in
`~/.cache/cogni-ml/drama3-runs/dense-20261014/`. Refresh on source, assets,
reference/runtime, precision, compiler, input, device/OS or artifact drift;
performance also decays with host/cache state. The sampled free-memory floor
was 15%, quiet-host gating was off, and unrelated processes were untouched.

## Sampler GPU cost and next target — 2026-10-14

Status: VERIFIED for the bounded diagnostic below; full-inference sampler
attribution remains open. `examples/drama3_sampler_benchmark.cr` builds with a
relative `require` and the normal Metal link flags, resolving the earlier
unsupported `-I` / `--path` attempts. No inference kernel or default changed.
Risk is CAUTION for measurement interpretation and queue lifetimes; rollback
is removal of the diagnostic and its analysis tools.

The release probe calls unchanged resident `Context#sample`, including its
finite-input guard. Each pair encodes 36 main-head calls (18 normal at
temperature 0.8 / top-p 0.9, 18 retry at 1.0 / 0.95) and 153 tower-head calls
(0.8 / 0.9), all at top-k 30, in two command buffers. Logits and uniform draws
are uploaded before timing. Graph depth two admits both submissions before
one final drain. Both operation groups have zero intermediate completion
waits and readbacks. Per-output checks run only after that drain and must add
no GPU waits/submissions. Warmups exercise both submission orders; seven
measured pairs per profile alternate main/tower order and reverse profile
order. GPU durations are `GPUEndTime - GPUStartTime` for whole command buffers,
not CPU encode wall time or per-dispatch timestamps.

Final run `root-measured-r3`, invocation
`80ab95a36d308772874d86d58da91914`, on M2 Max/macOS 26.6.2:

| Fixed-input profile | Main/retry 36 calls, median ms | Tower 153 calls, median ms | Pair GPU median ms | Pair min–max ms |
| --- | ---: | ---: | ---: | ---: |
| Retained decode head 0 / main proxy | 82.069 | 349.449 | 431.568 | 430.113–432.153 |
| Retained prefill head 0 / main proxy | 83.665 | 357.682 | 441.401 | 440.347–442.127 |
| Retained decode head 8 / main proxy | 78.760 | 334.678 | 413.687 | 412.930–414.760 |
| Synthetic ascending logits | 839.504 | 3569.096 | 4407.456 | 4404.510–4419.828 |
| Synthetic descending logits | 67.851 | 286.451 | 354.414 | 353.952–355.536 |
| Synthetic flat ties | 71.604 | 304.844 | 376.440 | 375.298–376.956 |
| Synthetic top-k cutoff ties | 67.468 | 288.039 | 355.561 | 353.644–356.542 |

Each column has its own median; component medians need not sum to the median
of paired sums. The three trace-based profiles cost about 2.19–2.34 ms/call
amortized over each command group, not measured isolated-call latency.
Only the tower rows are retained checkpoint logits; main rows prepend a
synthetic EOS logit to a tower row because no actual main-head logits were
retained. Reusing one row per group is a hot-input microbenchmark, not replay
of all 189 production distributions. Ascending input increases insertion
comparisons about 29.9 times and measured pair time about 12.4 times versus
descending. This confirms data-dependent cost, not an observed production
worst case. Host quiet gating was off; the runner's 15% sampled reserve passed.

All 11,907 outputs (2 warmup + 7 measured pairs, 7 profiles) match a scalar
FP32 oracle, including main retries. There are 126 submitted/completed command
buffers and final completion waits, maximum pending two, final pending and
retained bytes zero, and exactly 47,628 control plus 47,628 terminal read bytes
for post-drain verification. Deliberately wrong expected IDs and depth one
both fail. Nine Python analysis controls pass, including cross-run stream
mixing, shifted/missing/duplicate timestamps, invalid durations, extra waits,
readbacks, partial output verification and retained resources. Matching random
invocation stamps and zero-based command indices reject accidental mixed-run
logs; they are not authentication against edited logs. Global profile order
is also checked against the executable's alternating schedule.

`scripts/drama3_sampler_reference.py` extracts and executes the exact pinned
Fish-Speech sampler on CPU at revision
`541566932abbaebcb0a40e562a1f846738383f38`; no model import or new dependency is
needed beyond the existing Torch reference environment. All ten non-tie normal
role-cases have matching retained support, with conditional probability
max absolute error at most 5.96e-8. Main retry support matches in all five
non-tie cases, with max probability error 7.45e-9; its other two cases have
the same deliberate tie discrepancy. The four deliberate flat/cutoff tie cases
disagree: native ties prefer lower token IDs, while upstream CPU
`torch.sort(stable=False)` reorders equal logits. Uniform inverse-CDF draws
and upstream exponential-race draws also have different RNG contracts.
Injected exponential noise verifies in-support target selection only; it
does not prove RNG or general sampler parity. Expected GPU IDs come from the
scalar native oracle, whose host `exp` rounded to FP32 can differ from Metal
near a decision boundary. Prior backbone/tower/DAC intermediate and WAV gates
remain unchanged-source evidence, not freshly rerun inference in this probe.

Next attribution experiment: actual-shape backbone shared-input projections,
starting with `qkv/z/b/a` in `BackboneResident#linear_attention`. No projection
bottleneck has been established. Earlier command sums
place about 6.67 s in combined prefill/generation versus 1.75 s in DAC, while
this bounded sampler costs 0.414–0.441 s on trace-based profiles. These are
different measurement boundaries; do not subtract them or infer a production
speedup. First measure the projection groups with the same asynchronous
method. Reject fusion if actual-shape timing or intermediate replay regresses;
preserve each output's accumulation order. Sampler parallelization remains a
candidate, especially for the ascending tail, but changing denominator
reduction order risks top-p boundaries and is not justified by insertion
comparison counts alone. Any kernel change must rerun its relevant pinned
upstream intermediate replay and full-WAV check before promotion.

DoD commands, source/binary identities, the qualified report and reference
audit are indexed in `.landmark/evidence/drama3-sampler-cost.json`. Raw local
artifacts are under
`~/.cache/cogni-ml/drama3-runs/sampler-cost-20261014-r2/`; early runs remain
historical diagnostics, and only `root-measured-r3` supplies the table above.
Refresh on source/reference/assets/precision/compiler/input/device/OS or
artifact drift; timing additionally decays with host state. The mapping gate
below remains incomplete and default-off.

## Active mapping-promotion experiment

Status: IN_PROGRESS / default promotion withheld. The final controlled attempt
completed nine of sixteen required pairs, then failed warm-cache preparation
after both allowed repair passes. All eighteen completed arms passed exact
baseline WAV/prompt/codes/EOS/readbacks and graph-drain checks; the lowest
sampled inference headroom was 26%, above the authorized 15% reserve. The
blocker is repeatable cache conditioning, not an observed correctness failure
or the memory floor. Default and rollback remain `DRAMA3_MMAP_SHARED=0`.

The private default remains admitted until this experiment passes. Compare the
same release binary with mapping modes 0/1 at graph depths 0/2, separating
verified cold and warm OS file-cache states. Residency is measured with
`mincore`; read-only, model-scoped `msync(MS_SYNC | MS_INVALIDATE)` is eligible
only after a known-content fixture proves eviction without byte/stat changes.
This does not control SSD/controller caches. No global purge or interference
with unrelated processes is admitted.

Risk: CAUTION. Rollback is mapping mode 0. Each completed run must reproduce
the baseline prompt/codes/EOS/WAV/readback ledger and drain all graph leases.
Cold preparation requires zero resident model pages; warm preparation requires
all model pages resident immediately before launch. Record preparation time
separately from complete process time and retain both. The user explicitly
authorized a 15% system-free-memory reserve for this experiment after the
initial 30% reserve blocked its first arm. Require 60% before cache preparation
and cold launch, using the observed at-least-45-point drop plus the 15% reserve
as an admission estimate. Warm launch requires 20%, because cache preparation
has already made the model pages resident; it leaves a five-point margin above
the sampled reserve. Monitor preparation and inference; both the parent and
the owned `run_safe` wrapper stop at or below 15%, recording an invalid sample.
This is sampled pressure control, not a proof of a continuous physical-memory
bound, and resident file pages are not equivalent to permanently pinned RAM.
A post-warm high-headroom requirement was rejected because it could exclude the intended
cache state itself. Interrupted or cache/headroom-invalid pairs cannot promote
the default and remain visible in the evidence.

Predeclared promotion gate: at least four complete alternating-order pairs in
each cache/depth cell; shared/private process-time median at most 0.95 in every
cell, no pair above 1.05, no correctness/ownership/readback regression, and no
memory-guard or reserve violation. This is a local workload/device gate, not a
general model-performance guarantee. A failed cell keeps private as default
and redirects work to attribution. The strongest counterexample is that the
previous improvement disappears when initial file-cache residency is equal.

The first controlled attempt (`shared-controlled-20261014-a`) normalized all
1,578,824 model pages to zero residency and launched private/depth-zero at 75%
free. It reached the then-active 30% reserve at 14.912 seconds; the owned
workload was stopped and reaped. Zero pairs completed, so the interruption is
not a speed or parity result. Source/binary identities and model file stats
remained unchanged. The new 15% policy is a separate experiment; it does not
retroactively validate this arm. Fresh private and shared fixture mappings
must also see warm pages before invalidation, preventing an always-zero
fresh-map residency observation from qualifying the cache control.

The first 15%-reserve attempt (`shared-controlled-20261014-b`) completed three
exact-parity pairs before rejecting a partially resident preparation for
resident/depth-two/private. Its private/shared process times were 26.53/19.70 s
(nonresident/depth zero), 22.95/18.46 s (nonresident/depth two), and 15.62/9.77 s
(resident/depth zero). Its lowest sampled inference headroom was 21%; cache
conditioning, rather than the reserve, blocked this attempt. These three pairs
do not satisfy the matrix gate. That harness discarded the failed preparation's
cache snapshot before recording its exception; a later query is only a later
observation and cannot reconstruct that failed snapshot.

Unmapping each shard immediately after reading it allowed earlier shards to
be reclaimed during later reads. A cache-only discriminator kept all six
read-only mappings alive through the complete touch, then released them all:
fresh queries observed all 1,578,824 pages resident both while helpers existed
and after their release. The revised preparation creates all helper mappings
before touching any, releases all on normal/error/reserve exits, and requires
fresh all-resident mincore after cleanup and again immediately before launch.
No helper mapping or open model descriptor survives into the timed child.
The same conditioning applies to both mapping modes; preparation time remains
outside process time. A partial state fails closed with its per-file snapshot
retained. This is cache conditioning, not page pinning or an inference change.

Keeping mappings alive alone was insufficient: attempt
`shared-controlled-20261014-c` completed eight exact-parity pairs, then rejected
resident/depth-zero/private preparation at 1,501,733/1,578,824 resident pages.
The held and post-cleanup snapshots agreed: 77,091 pages had already been
reclaimed from the first model shard while all helpers were mapped. Preparation
headroom was 64–74%; the lowest completed-inference sample was 24%. This is a
cache-conditioning failure, not a reserve violation. Its eight pairs remain
separate and cannot promote the default.

A cache-only probe from a retained partial state restored all pages with one
missing-page pass and observed full residency after helper cleanup. The next
matrix starts from scratch. Warm conditioning now performs the same full read
for both modes, followed by at most two mincore-directed missing-range reads
while all six helpers remain mapped. It records the ranges, bytes, duration,
and before/after snapshots for each repair. Two passes are a fixed bound; a
partial post-cleanup or prelaunch state still rejects the arm and stops the
matrix. Repaired pages are newer than retained pages: exact mincore occupancy
and alternating order do not establish identical cache ages.

Final attempt `shared-controlled-20261014-d` stopped before launching
resident/depth-two/private pair two. Its full read left 1,447,197 pages resident;
repair one read 2,420,670,464 bytes and reached 1,487,301 pages; repair two read
3,728,264,286 bytes but fell to 1,462,154 pages. After helper cleanup only
1,456,273/1,578,824 pages (92.24%) remained resident. Preparation took 63.105 s
with sampled free-memory readings of 71–78%. More repair passes, weaker cache
criteria, and pooling earlier partial matrices are not admitted by this gate.
This refutes reliable full-residency conditioning by the bounded read/repair
procedure on the current host; it does not refute shared mapping's speedup.

The completed pairs are encouraging but insufficient for promotion:

| Initial file cache | Graph depth | Complete pairs | Median paired shared/private process time |
| --- | ---: | ---: | ---: |
| Zero resident pages | 0 | 2 | 0.730317 |
| Zero resident pages | 2 | 2 | 0.786786 |
| All resident pages | 0 | 3 | 0.638089 |
| All resident pages | 2 | 2 | 0.611097 |

Every completed pair favored shared mapping. A separate artifact recheck read
all eighteen WAVs and reports, recomputed the four cell summaries, checked
cache snapshots, readback/drain ledgers, and sampled reserves, and agreed that
the matrix cannot promote. This is same-lineage verification, not independent
replication. The harness/guard claim is ROBUST within the executed controls;
the speed/default-promotion claim remains VULNERABLE because no cell has the
required four pairs. The next mapping discriminator requires a changed cache
conditioning premise or a controlled host, rather than another identical retry.

Canonical identities, failed attempts, prior unchanged-source upstream lineage,
and refresh boundaries are in
[the controlled mapping packet](../.landmark/evidence/drama3-mapping-controlled.json).

Reproduce the guard checks and the full matrix from this worktree:

```bash
PYTHONDONTWRITEBYTECODE=1 COGNI_RUN_SAFE_MIN_FREE_PCT=15 \
  scripts/run_safe.sh python3 120 1024 \
  -m unittest spec/drama3_mapping_benchmark_spec.py
PYTHONDONTWRITEBYTECODE=1 COGNI_RUN_SAFE_MIN_FREE_PCT=15 \
  scripts/run_safe.sh python3 60 512 scripts/drama3_mapping_benchmark.py \
  --model /Users/sergey/.cache/cogni-ml/drama3-mlx-q4-5331ab50 --self-test
PYTHONDONTWRITEBYTECODE=1 python3 scripts/drama3_mapping_benchmark.py \
  --model /Users/sergey/.cache/cogni-ml/drama3-mlx-q4-5331ab50 \
  --binary /tmp/drama3_shared_final_generate \
  --baseline-wav /Users/sergey/.cache/cogni-ml/drama3-runs/hello-speed-baseline.wav \
  --output-dir /Users/sergey/.cache/cogni-ml/drama3-runs/shared-controlled-new
```

The output directory must be new. Existing output is never overwritten; each
arm and any interrupted attempt remain in `progress.json`. `--self-test` uses
only a 1 MiB fixture on the model volume and pure gate/report controls; the
five Darwin guard specs use real owned CPU children with injected pressure or
cache readings, including rejection before child launch. Five additional
controls cover post-cleanup rejection, all-helper cleanup after preparation
failure, missing-range clipping, repair convergence, and the two-pass bound.
Neither check establishes model speed or upstream numerical parity.

The next optimization discriminator is GPU stage attribution inside the
backbone, without adding completion waits. Existing command timelines assign
about 6.67 seconds of GPU execution to the combined prefill/generation commands
and 1.75 seconds to DAC. These are command-group sums, not individual operator
times. Depth two reduces command completions from 90 to 73 while leaving those
GPU sums essentially unchanged; it is not evidence that deeper queuing will
reduce compute time. The current component profiler deliberately calls
`Context#finish` in `BackboneResident#step`, so its wall spans cannot identify
the next asynchronous kernel bottleneck. Attribute a measured operator before
changing kernels; require the existing intermediate upstream gates and matched
complete-process timings for any resulting optimization. The main-token EOS
control read remains an explicit boundary, with only 152 control bytes in the
current smoke; generated frames already pass directly between GPU operations.
The existing bridge exposes GPU start/end timestamps for whole command buffers,
not individual dispatches. A bounded fixed-input sampler/projection probe can
submit isolated groups at depth two and drain once, using the existing timeline
without adding per-operation completion waits. Such synthetic group timing
does not attribute the complete generation path by itself.

A bounded external sampler probe was initially drafted but produced no measurements:
both allowed build attempts exited on unsupported Crystal include-path flags
(`-I`, then `--path`) before compilation. No GPU workload ran. Its source,
latest build log, and a curated two-attempt diagnostic are retained outside
the repository and indexed by the mapping packet. The operator hypotheses
below were unmeasured at that point. The sampler-cost section above supersedes
that build blocker and records the subsequent bounded measurements.

Source-level follow-up candidates, pending operator attribution and matched
measurements:

- `dr3_sample` runs top-k insertion and the full-logit exponential denominator
  in one GPU thread. The actual main head has 4,097 EOS/semantic candidates,
  and the nine residual tower heads have 4,096 each; this is not the 252,288
  text vocabulary. The 17-frame/EOS smoke executes 36 main/retry and 153 tower
  samples at top-k 30. Time batches matching these 189 calls with fixed logits
  and draw slots. Parallel top-k may help, but a reassociated Float32
  denominator can change top-p cutoffs; retain tie order, threshold rules,
  ordered draw subtraction, and both main/retry draws. Existing Qwen top-2
  reductions are a structural reference, not a resident sampler replacement.
- `BackboneResident#linear_attention` dispatches `qkv`, `z`, `b`, and `a`
  projections from the same input. Test a combined projection dispatch on the
  actual shapes, preserving each output's accumulation order; fewer launches
  alone do not establish a full-process improvement.
- `TowerResident#codebook_sum` allocates a scalar row, embedding, and sum for
  every codebook. Test one gather-and-ordered-sum kernel, preserving code
  clipping, masking, and accumulation order, against the existing tower trace.
- DAC's convolution kernels and separate Snake materialization warrant
  attribution within the measured 1.75-second phase. A naive fusion that
  recomputes Snake for every convolution multiply may make it slower; require
  actual-shape GPU timings before choosing fusion or convolution tiling.

These are hypotheses, not measured speedups. Each candidate needs a fixed-input
baseline/candidate replay with one final drain, followed by the unchanged
intermediate upstream and complete-WAV gates. The command-phase reanalysis and
original timeline identities are retained with the mapping experiment evidence.

## Prior first-submit slice

Status: read-only shared checkpoint mapping implemented as opt-in;
performance/default promotion remains provisional. Set `DRAMA3_MMAP_SHARED=1`
to use `MAP_SHARED`; default and rollback are `0` (`MAP_PRIVATE`). Both modes
keep `PROT_READ`, existing mmap owners, whole-shard no-copy Metal views, Shared
storage, tracked hazards, kernels, and serial queue ordering. This adds no CPU
activation handoff. Checkpoint files must remain stable and immutable until all
referencing contexts are terminal; shader `const` inputs provide the current
in-tree GPU read-only constraint. Raw public pointers/native bindings are unsafe
plumbing, not an enforcement boundary. Reset/unload before every referencing
context drains remains unsupported.

The predecessor binary reproduced 14.310 seconds from commit to GPU start,
followed by 0.173 seconds of GPU execution. The final diagnostic's four fresh
processes used one identical tiny kernel and five identical dispatches per
command: first waits were 0.881/0.817 ms for a 16 KiB view versus 518.645/
519.265 ms for the same file's 5.50 GB shard view. All terminal words matched;
GPU intervals were about 24–25 microseconds. File-cache state was uncontrolled.
This reproduces a large extent-associated delay without the backbone, tower,
or DAC; it does not assign the entire model delay to a particular driver
pinning, page-in, or deferred-compilation mechanism.

Two complete same-binary shared/private AB/BA pairs at graph depth two measured
11.00/20.97 seconds and 18.60/25.20 seconds with `/usr/bin/time` (process launch,
loading, preparation, generation, WAV/report output, and cleanup included).
These pairs used the earlier timed binary; the final build differs only in
formatter whitespace and was separately replayed below. Full-run cache and host
state were uncontrolled. Both pairs favor shared mapping, with ratios 0.524559
and 0.738095. The additional unpaired shared run took 9.89 seconds. All five
completed runs reproduced the same 17 frames, EOS, and exact WAV bytes, plus
152 control / 140,012 terminal
readback bytes; graph leases drained. Per-process peak footprint was
1.709 GB shared versus 1.709–1.778 GB private. This counter excludes or differently
charges some file-backed/Metal residency; it is not a complete physical-memory
budget. No RAM saving or stable speedup is established from these samples.

Three later full attempts were terminated by the unchanged 25% system-free
memory guard: one private, one shared, and the final shared graph-depth-two
smoke. They remain recorded as incomplete attempts and contribute no successful
timing/parity result. The host had unrelated active workloads; their processes
were not modified. The final owned-binary synchronous shared route at graph
depth zero completed with exact WAV/readback parity (11.08 seconds application
time). A separate profiled owned-binary run measured 2.079 seconds
commit-to-GPU-start and 0.159
seconds GPU execution, with 12.16 seconds full-process time. This is a separate
cache/host state from the predecessor 14.310-second observation; the pair is
not an isolated causal estimate. Fresh upstream replays passed 243 backbone /
497,664 values, 726 tower / 2,042,880 values, 20 strict-head / 77,824 values,
29 DAC internal stages, and 4,096 waveform values at the unchanged gates below.
All component traces are bitwise equal to the preceding native route. Metal
passed 63 examples at each of depths zero and two; CPU passed 20, Python
contracts six.
The new mapping guard covers both modes, invalid configuration/ranges, queued
weight consumption through GC, terminal drain, and unchanged fixture-file bytes.

Rejected experiments: CPU page touching took 24.02 seconds of setup plus
5.74 seconds of first wait, worsening the measured full probe. Per-tensor views
had full-process view/default ratios 1.029940, 1.045088, 0.887771, and 1.025412
(median 1.027676); three of four pairs regressed despite exact WAV/ledger parity.
The tensor-view code was removed. Its lower retained-byte counter was backing
extent bookkeeping, not measured physical RAM. A single MADV_RANDOM scratch
probe did not improve the preceding uncontrolled-cache probe; no inference
madvise change was retained.

Risk: CAUTION (file sharing and resource preparation). Capability DoD is the
executed mapped-input guards, pinned upstream replays, and exact native full
WAV/readback checks. Performance promotion additionally requires repeated
same-binary full-process AB/BA runs with sufficient host headroom and no guard
terminations, at graph depths zero and two. The strongest failure is an
unintended mapped weight write or owner release before completion. Current
read-only shaders, GC/drain guards, and exact file/output checks cover the
in-tree serial route; external mutation/concurrency remain outside admission.
Ordinary mmap selection is not an LTP/WBA transformation. No Collapse or default
speed promotion follows from the small timing sample; global memory admission
and repeatable complete-wall descent remain open.

Evidence, source identities, exact commands, aborted attempts, and refresh
rules: [startup evidence](../.landmark/evidence/drama3-startup.json).
Diagnostic source: [startup extent probe](../examples/drama3_startup_probe.cr).
Next falsifier: repeat matched full-process pairs after host headroom recovers,
separating file-cache state from mapping mode and recording the first-submit
interval alongside the complete timer. Refresh on source/reference, checkpoint,
precision, compiler, input, device/OS, or artifact-availability drift.

## Admitted CogniGraph handoff slice

Status: bounded CogniGraph route admitted as opt-in; speed promotion withheld.
`DRAMA3_COGNIGRAPH_MAX_INFLIGHT` defaults to `0` (legacy synchronous route);
`1` or `2` enables the bounded graph queue. DRAMA's `Resident::Context` wraps
CogniGraph's `GraphSubmissionQueue` over one explicit Metal queue. Buffer owner
identity is scoped to the context and inherited by slices; cross-context reads,
dispatches, and native bindings reject ordinary foreign views. This is not a
security boundary: code that reconstructs a `Buffer` from its exposed backing
handle/owner or performs raw external writes can bypass the tag. Treat the
constructor/backing handle as internal unsafe plumbing. The scoped safe route is
the serial single-context native CLI without external mutation, rebinding, or
concurrent context use. Leases retain temporary bindings through checked
completion. Directly bound persistent attention caches and mapped weights stay
rooted in the resident model and MetalBackend until completion; callers must
keep these owners and the Context alive until drained. Scratch allocations are
fresh; there is no recycling pool.

Graph mode submits prefill rows without immediate waits and carries each
request-owned generated-frame view directly from tower output into the next
backbone step and DAC. The next main-token read remains a CPU control boundary;
terminal codes and PCM are read once at the output boundary. EOS-first exits
before a generated row. The output view is range-checked against main, hidden,
and previous-frame inputs before dispatch; sampled codes must not alias the
frame destination. Finite checks, explicit diagnostic reads, synchronous
component profiling, and single-use model semantics remain. An encoding/GPU
failure drains or cancels known leases and poisons the context; this does not
undo model state already mutated by completed commands. Reset/unload is legal
only after every resident context is terminal. Depth bounds live leases, not
bytes per lease, and is not long-form memory admission.

Measured scope: same kernels and eight complete runs across four same-binary
depth-0/graph pairs produced exactly 17 frames, EOS, and WAV bytes in every run.
Ordinary command submissions and completion waits each fell from 90 to 73.
Readback ledgers stayed at 152 control bytes and 140,012 terminal bytes, with no
activation or diagnostic readbacks. Graph/legacy complete-wall-time ratios
were 0.870419, 1.048293, 0.980546, and 0.974302 (median 0.977424); one pair
was 4.8% slower. `/usr/bin/time` per-process peak footprint rose from
1.576–1.639 GB at depth zero to 1.709–1.765 GB with graph mode. These
observations do not establish a reliable whole-run speedup, and depth zero
remains the default.

Numerical replays pass at the pinned inputs: 243 backbone traces / 497,664
values (gates 6e-5 / 3e-4); 726 tower traces / 2,042,880 values (6e-4 / 3e-4);
and 20 strict-head traces / 77,824 values (3e-5 / 3e-4). Codec gates cover 29
internal stages (max absolute error 1e-3, RMS 2e-4, normalized RMS 2e-4);
4,096 waveform values use the per-sample absolute-error gate
`abs(error) <= max(5e-5, abs(reference) * 1e-3)`. The final owned-binary
depth-0/2 replays passed with exact WAV bytes; all three upstream trace sets
match and intermediate values are bitwise equal to the previous native path.
The full Metal suite passed at depths zero and two (60 examples each, no
failures, errors, or pending examples); the CPU suite passed 18 examples and
the Python contract suite passed six tests. These results are recorded in the
evidence packet below.

The retained-byte high-water of 31,903,032,324 is a conservative sum of whole
backing allocations per live lease: mapped dense shards and shared allocations
can be counted more than once, packed weights bound directly by native methods
are omitted, and the number is not RAM usage. The per-process peak ranges above
are `/usr/bin/time` observations; the 40,960 MB process-tree RSS limit and 25%
minimum-free-memory floor are separate run guards.

Risk: CAUTION (submission ordering, ownership, and memory). Rollback is
`DRAMA3_COGNIGRAPH_MAX_INFLIGHT=0`; context failure is terminal. The local
mechanism evidence is bounded: a Spike found and removed redundant row waits;
the Ladder queue passed lifecycle/parity replays. LTP/WBA Collapse and default
speed promotion remain withheld because whole-run ratios are inconsistent and
measured peak footprint increased. The audit coordinates are ordered as
`(semantic/ownership error, undrained lease, memory-guard breach, full-corridor
wall time, avoidable waits)`; these measurements establish a lower final
coordinate, not global potential descent. The transport corridor is one FIFO
Metal queue; the Ladder move submits fresh/persistent owners, passes resident
outputs to dependent work, waits at control/terminal boundaries, then releases
only completed leases. Failed work drains/cancels and poisons the context; it is
not retried against mutated state. Depth zero is the dual frame. The first-submit
extent probe and opt-in shared-mapping results are recorded
in the active slice above; stable performance and global memory admission
remain separate from the admitted correctness and wait-count result.

Evidence and exact run inputs: [CogniGraph evidence](drama3-cognigraph-evidence.json).
Source anchors: [resident ownership/queue](../src/ml/audio/drama3/resident.cr),
[generator handoff](../src/ml/audio/drama3/generator_resident.cr),
[tower output view](../src/ml/audio/drama3/tower_resident.cr), and
[CogniGraph queue](../src/ml/metal/compute_graph.cr). Refresh parity, ownership,
readback, and matched timing claims on source/reference, precision, asset,
compiler, input, device/OS, or evidence-availability drift.

## Source identity and boundary

- MLX reference: cguldogan/mlx-audio, branch drama3-mlx, commit
  e7488935ec2ebfafd38ef2b58e290d79f3d05dc7.
- Official reference: fishaudio/fish-speech, commit
  541566932abbaebcb0a40e562a1f846738383f38.
- Official model: fishaudio/drama-3, revision
  e0a9218aa48888270f06ef1ed9ea67622c0e17d2.
- Converted weights: cguldogan/drama-3-mlx-q4, revision
  5331ab509359c0e4a7e353a19e255ddf3e8152a8. All five model shards, codec,
  tokenizer, configuration, and index have SHA-256 identities in the evidence
  report linked below.
- Existing Qwen runtime and dirty primary checkout remain untouched.
- The initial runtime is an offline CPU correctness path. Metal now has a
  resident model graph; the earlier packed-expert/DAC hybrid is historical.
  Streaming, cloning, and speaker switching remain outside this slice.
  Unsupported configuration must fail explicitly.

## Design laws

- Keep residuals and zero-centered RMSNorm arithmetic in FP32. Do not pre-fold
  1 + w into BF16 weights.
- Backbone uses half-split partial RoPE; audio tower uses interleaved RoPE.
- Tower caches advance over time; prompt row p uses target frame p + 1.
- MoE routing is top-8 with normalized selected softmax weights plus a sigmoid
  shared expert. Nomic's MoE semantics must not be reused unchanged.
- Load quantized experts lazily; never expand the entire model to FP32/BF16.
- Reject non-finite intermediate results, invalid tensor shapes, and incomplete
  assets. WAV existence alone does not establish speech quality or parity.

## Execution plan and falsifiers

Risk: CAUTION (new model/weight/audio boundaries). Rollback: isolated worktree
and exact-path atomic commit; no changes to existing Qwen admission paths.

1. Native tensor math and mapped converted weights: compare dense/quantized
   projections, routing, norm, convolution, and both RoPE layouts to Python
   reference arrays; malformed shapes and corrupt inputs must reject.
2. Native backbone and tower: compare fixed-input intermediate tensors to
   pinned MLX; compare the numerically sensitive tower against official
   PyTorch as well. Test temporal cache continuation and prompt/target shift.
3. Native DAC decoder: compare codebook reconstruction, convolution stages,
   and waveform against reference. Encoder is deferred for text-only scope.
4. Prompt/tokenizer, generation loop, and PCM WAV: short bounded end-to-end
   run, retain token/code traces and numerical reports. Run under run_safe
   memory/time guards. No real-time or broad quality claim from this smoke.

DoD: focused Crystal specs pass; fixture comparison rejects a seeded defect;
real-weight intermediate reports identify revision/input/tolerance; native CLI
produces finite, non-silent 44.1 kHz mono audio from text. Until every part is
observed, report IN_PROGRESS and name the unresolved path.

Evidence decays on model/reference/native-source/precision changes. Refresh by
regenerating fixed-input intermediate traces and comparing before generation.

## Native use

The model directory must contain the pinned MLX-converted snapshot, including
all indexed safetensors shards, `config.json`, `tokenizer.json`, and
`codec.safetensors`. Inference uses Crystal and native libraries; reference
Python environments are optional and separate.

```sh
crystal build --release -Dcpu_only examples/drama3_generate.cr \
  -o /tmp/drama3_generate
COGNI_RUN_SAFE_MIN_FREE_PCT=25 RUN_SAFE_PASSTHROUGH_STDIO=1 \
  scripts/run_safe.sh /tmp/drama3_generate 1200 40960 \
  --model "$MODEL_DIR" --text 'Hello.' \
  --output /tmp/hello-native.wav --max-frames 128 --seed 0
```

On the tested macOS/Xcode SDK, the build also needs
`--link-flags='-fuse-ld=/usr/bin/ld'` because the default lld rejects SDK stub
files. The output path and its `.json` sidecar must not already exist. The
sidecar retains prompt IDs, every generated code frame, sampling parameters,
EOS/truncation status, waveform statistics, and elapsed time. A bounded prefix
is explicitly reported as truncated. The runtime API lives under
`ML::Audio::Drama3`; the example directly composes `Tokenizer`, `Prompt`,
`Generator`, `Codec`, and `Wav` in one process.

## Native CPU verification baseline

The measured source/asset identities and compact numerical results are in
[drama3-native-evidence.json](drama3-native-evidence.json). Reference replay
commands and dependencies are in [the reference tooling guide](../scripts/drama3/README.md).

- Focused native suite: 16 examples, zero failures/errors. The six Python
  contract checks include rejection of a seeded numerical defect, malformed
  traces, missing traces, non-finite values, and a Float32 JSON roundtrip case.
- Synthetic upstream fixtures exercise three cached backbone rows, temporal
  tower continuation, both RoPE layouts, zero-centered norms, and MLX affine
  Q4/Q8 lane order. Synthetic codec parity additionally uses unequal upsample
  rates, multiple residual books, clipping, and raw weight-normalization input.
- Tower configuration rejects unsupported bias, gating, normalization, RoPE,
  SWA/MoE, and feature-placement settings, including non-finite numeric values.
- Actual Q4 backbone: 243 boundary traces / 497,664 values across three cached
  rows compared with pinned MLX at `atol=6e-5, rtol=3e-4`. Maximum absolute
  error is 0.001220703125 at a residual magnitude of about 1712; the relative
  error there is 7.13e-7. The original `atol=3e-5` envelope failed four small
  residual values; the larger floor is limited to this recorded probe.
- Actual tower: 726 traces / 2,042,880 values compared with official PyTorch
  at identical hidden inputs and cache positions 0/1. The default
  `atol=3e-5, rtol=3e-4` envelope fails 55 intermediate values. A scoped internal
  floor of `6e-4` passes; aggregate relative L2 error is 7.41e-7. Final hidden
  and all head logits independently pass the original strict envelope. Logit
  max error is 1.14441e-5, all 18 argmaxes agree, and the smallest top-1 margin
  is 0.02979. This is measured drift, not bitwise parity or a universal bound.
- Actual DAC two-frame probe: all intermediate gates pass (max absolute
  error <=1e-3, RMS <=2e-4, normalized RMS <=2e-4). Waveform retains the stricter
  per-sample `max(5e-5, abs(reference)*1e-3)` bound; observed max/RMS errors
  are 2.8044e-5 / 6.2792e-6.
- Native `Hello.` run: 23 prompt rows, 17 generated frames, EOS observed,
  34,816 PCM16 samples at 44,100 Hz (0.78948 seconds). Peak 0.04967, RMS
  0.01073, no normalization. Model generation plus DAC took 241.83 seconds
  on an M2 Max. This is a correctness baseline with a slow scalar DAC.
- Final-source repeat: identical prompt IDs, all 17 code frames, and WAV bytes;
  elapsed 244.14 seconds. Final native backbone and tower traces were also
  regenerated against the same fixed inputs before committing.
- Full utterance DAC replay uses exactly those 17 native-generated frames in
  pinned MLX: 34,374 of 34,816 PCM16 samples match exactly; maximum difference
  is one PCM16 count, RMS difference 0.11267 counts. No peak or RMS
  renormalization is applied. This checks decoding of the complete smoke,
  independently of the two-frame intermediate probe.

Actual-checkpoint comparisons promote stored BF16 floating values to F32 on
both sides while preserving packed quantized tensors. They establish the
declared F32 execution path, not stock MLX BF16 equality. JSON trace values are
rounded to their declared Float32 type before comparison. Different RNGs and
floating-point reductions mean stochastic sequences need not match upstream.

## Remaining boundaries

Only short, single-speaker text generation is admitted by this smoke. Voice
cloning, reference-audio encoding, streaming, speaker switching,
long-form stability, broad intelligibility/quality evaluation, and real-time
performance remain untested or unimplemented. No subjective listening claim
is made. Source and weight licensing are recorded separately in
[THIRD_PARTY_DRAMA3.md](../THIRD_PARTY_DRAMA3.md).

Refresh the evidence after a weight/tokenizer/config change, a source or
precision change, or a backend change. A failure reopens the affected parity
claim; retaining a WAV from an earlier run does not close it.

## Resident Metal execution

`DRAMA3_BACKEND=metal` in the current Metal-capable CLI selects
`ResidentGenerator` and `CodecResident`. The preceding hybrid implementation
remains recorded in the historical section below. CPU remains the default
backend and explicit rollback. No existing Qwen admission path is changed.

Risk: CAUTION (GPU reductions, dynamic routing, cache ownership, and sampling
precision). Rollback: `DRAMA3_BACKEND=cpu`, or the preceding resident `b91ad1e7`.
The highest-risk failure is altered routing/cache arithmetic hidden by a
plausible waveform. DoD: focused primitive/cache/failure tests, actual
backbone/tower/DAC gates at their existing tolerances, complete text-to-WAV,
call-site audit plus readback accounting, and matched component/total timing.
The executed evidence is indexed in
[drama3-resident-evidence.json](drama3-resident-evidence.json), with the current
latency changes in
[drama3-resident-speed-evidence.json](drama3-resident-speed-evidence.json).

### Device/host contract

- Backbone: dense and packed projections, embeddings, norms, MoE top-k and
  shared-expert gating, both RoPE layouts, full attention, convolution ring,
  and Delta state updates execute in Metal. Packed experts stay packed.
- Tower: teacher forcing, temporal KV caches, enrichment/residual layers,
  logits, FP32 top-k/top-p selection, repeat fallback, and generated frames
  remain resident. The next backbone step consumes the original device frame.
- DAC: codebook reconstruction, raw-weight conversion and normalization,
  transformer, ConvNeXt, convolutions, Snake, GELU, and waveform run on GPU.
  Frame layout is `[main,c0,c1,...,c9]`; the decoder ignores main, uses c0 as
  semantic, and c1..c9 as residual codes. It consumes the valid prefix of the
  generator's original device allocation without a CPU code re-upload.
- Host work is tokenizer/config/shape validation, input and seeded-uniform
  preparation/upload, scheduling, EOS branching, terminal statistics and
  WAV/JSON I/O. A chosen main token (4 bytes) and a sticky finite-status word
  (4 bytes) are read per attempted frame. Completed codes are read once for
  the sidecar and PCM once for WAV; each terminal read checks 4 status bytes.
  Literal zero host reads is not claimed.
- Trace callbacks are diagnostic opt-ins. The normal CLI leaves them unset.
  `Context#read` rejects an `activation` purpose, and records explicit control,
  terminal, or diagnostic reads. Caller-supplied labels alone do not prove
  residency: the actual default call sites were also audited.

A serial command encoder retains temporary allocations and bound views until
checked GPU completion. Views retain their allocations; model/global mapped
weight caches retain mmap owners. Weight loads use 64-bit byte offsets and
unaligned-safe scalar decoding. Row/stage completion bounds live scratch.
GPU finite guards cover layer/component boundaries and sampling inputs; a
sticky fault rejects the next read even if the terminal tensor is finite.
This does not assert a finite check after every primitive.

Runtime objects are for one serialized inference. Do not reset global mapped
weight caches while a resident command/context is active. Finish all work
before `CodecMetal.clear!` and `MetalBackend.reset!` when unloading a model.
Concurrent inference/reset is outside the tested contract. Backbone cache
capacity is 1..1024; sampling top-k is 1..1024. CLI frame bounds do not guarantee
an aggregate RAM budget: long DAC tensors can be large. The executed admission
is the short 17-frame smoke, protected by run_safe, not long-form stability.

### Build and profile

```sh
make build/bridge.o
crystal build --release -Ddrama3_metal examples/drama3_generate.cr \
  --link-flags="$(pwd)/build/bridge.o -framework Metal -framework Foundation -lc++ -fuse-ld=/usr/bin/ld" \
  -o /tmp/drama3_generate_resident
DRAMA3_BACKEND=metal DRAMA3_PROFILE=/tmp/drama3-resident-profile.json \
  COGNI_RUN_SAFE_MIN_FREE_PCT=25 RUN_SAFE_PASSTHROUGH_STDIO=1 \
  scripts/run_safe.sh /tmp/drama3_generate_resident 300 40960 \
  --model "$MODEL_DIR" --text 'Hello.' \
  --output /tmp/hello-resident.wav --max-frames 128 --seed 0
```

Omit `DRAMA3_PROFILE` for ordinary inference. Profiling synchronizes component
boundaries, so enclosing backbone, tower, and DAC spans include actual GPU
completion; inner MoE/dispatch spans can be enqueue time and are not kernel
measurements. Sum disjoint component spans only. Normal execution composes
adjacent component commands and synchronizes at row/control/stage boundaries.
First-use mapping, conversion, and pipeline compilation occurring during
inference are included. The current sidecar timer starts before generator
initialization and ends after WAV writing; it excludes tokenizer/config and
`WeightStore` construction, process startup, and compilation of the executable.
It is not a process cold-start metric. The JSON sidecar also records backend,
readback bytes by purpose, and submission count.

### Observed upstream and adversary gates

- 42 focused Metal examples pass; 18 CPU examples and six Python contracts
  also pass. They include eight recurrent/cache rows
  across convolution-ring wrap, capacity overflow, immediate EOS/empty prefix,
  invalid GPU expert ID/rank, non-finite intermediates hidden behind a finite
  terminal, tie order, top-p crossing, both codec layouts, poisoned main token,
  checkpoint singleton shapes `[1,C,1]` / `[C,1,1]`, and the highest semantic
  candidate in both primary and repeat-fallback selections. The last case
  failed before correcting an EOS-plus-vocabulary off-by-one bound.
- Final-source actual backbone: 243 traces / 497,664 values pass unchanged
  `atol=6e-5, rtol=3e-4`; max absolute error 0.0010986328125 at a large residual
  (relative error there 6.42e-7).
- Final-source actual tower: 726 traces / 2,042,880 values pass unchanged
  internal `atol=6e-4, rtol=3e-4`. Its final hidden states and 18 head logits
  separately pass the strict `3e-5, 3e-4` envelope: 20 traces / 77,824 values,
  max absolute error 1.04904e-5.
- Final-source DAC: all 29 internal stages pass `max_abs<=1e-3`, `RMS<=2e-4`,
  `normalized_RMS<=2e-4`. Worst observed values are 0.0007963181,
  0.0001249242, and 0.0000448816, respectively. All 4096 waveform values pass
  the per-sample `max(5e-5, abs(reference)*1e-3)` gate; max error 2.84016e-5.
- Full 17-frame DAC replay in pinned MLX: 34,359/34,816 PCM16 samples agree
  exactly, maximum error one PCM16 count, RMS error 0.1145694 counts. No
  normalization is used. Native CPU/GPU codes and prompt IDs match for this
  smoke; GPU/CPU PCM differs by at most one count.
- The normal smoke reads exactly 152 control bytes (18 attempts * 8 plus
  two terminal status checks * 4), and 140,012 terminal bytes (748 code bytes
  plus 139,264 F32 PCM bytes). Diagnostic and intermediate activation reads
  are zero; the DAC consumes device codes even though the sidecar also reads
  a terminal copy.

The sampler implements descending ranking, lower-ID ties and the upstream
top-p crossing rule in FP32. The later sampler audit above finds that upstream
CPU unstable-sort tie membership differs; universal upstream tie equivalence
is not supported. A precision
falsifier deliberately sets top_p=0.99999999 on equal logits: it rounds to 1
on GPU and can retain a candidate that CPU Float64 excludes. Reject universal
seeded CPU equivalence; do not hide the difference by widening trace gates.
Seed uniforms are prepared in the CPU RNG sequence and uploaded once, with
Float32 rounding. Real checkpoint comparisons continue to promote floating
weights to FP32 and preserve quantized weights; stock upstream BF16 equality
is outside this scope.

Correlated source adversary review is ROBUST for the default single-inference
residency/ownership path; executed upstream replays supply the numerical
support. The stronger exact-sampler, concurrent/reset, long-form, subjective
speech-quality, and general performance claims remain outside the evidence.

### Measured M2 Max latency at resident baseline `b91ad1e7`

The final resident profile for `Hello.`, seed 0, 23 prompt rows and 17 frames
through EOS takes 19.9842 s: backbone 15.0293 s (40 rows), tower 2.0170 s
(22 teacher-forced plus 17 sampled frames), DAC 2.7622 s. It uses 129
submissions with profiling enabled. Earlier profiled runs took 22.1805 and
23.9230 s; an unprofiled run took 22.7126 s with 90 submissions. Those runs
preceded the final shape-metadata cleanup and/or highest-candidate guard fix;
their selected codes and WAV bytes are identical to the corrected build.
These repeats show local variability, not a statistical performance bound.

The same corrected binary takes 244.1005 s with `DRAMA3_BACKEND=cpu`, versus
19.9842 s resident Metal: 12.21x for this measured workload. Disjoint completed
component spans are:

| Component | CPU seconds | Resident Metal seconds |
| --- | ---: | ---: |
| Backbone | 96.4977 | 15.0293 |
| Tower | 8.2569 | 2.0170 |
| DAC | 139.2602 | 2.7622 |

Prompt IDs, all 17 generated code frames and EOS agree; 34,741/34,816 PCM16
samples agree exactly, with maximum difference one count. The earlier hybrid
run was 51.6059 s (2.58x slower than this resident run), but used the preceding
source and excluded generator initialization from its timer. Treat that as a
historical comparison, not the matched same-binary benchmark.

The first final-run prefill row takes 9.34 s, later rows about 0.19 s, and
generated rows about 0.21 s. Initial preparation versus execution is the next
discriminating profiling question; no attribution to a specific driver,
mapping, or compilation mechanism is established here. Host load was not
controlled and other processes were preserved. The result is a local M2 Max
observation, not a real-time, cross-device, or cold-start guarantee.

Refresh fixed-input gates, matched generation, readback audit/accounting,
and timings after changes to source, precision, assets, compiler, device/OS,
workload, or evidence availability. A failed gate reopens the affected claim;
an earlier WAV or a structurally valid evidence JSON does not close it.

### Packed rank batching and coalesced Delta state

The current resident backbone batches independent selected expert ranks into
one dispatch per packed projection. Gate/up share the input vector; down reads
a separate intermediate per rank. Quantized weights remain packed, router IDs
remain on GPU, and affine-group arithmetic retains its original order. A final
kernel adds ranks in their original order, then the existing shared expert is
added. This reduces the rank-dependent dispatches from 49 to 6 per MoE layer.
Dense fixture matrices retain the scalar expert route.

The private Delta state now stores `[value_head,key,value]`, making neighboring
value lanes contiguous. Each lane still processes keys in exactly the same
order. The state starts at zero and has no external layout contract. Both
choices are frozen at backbone initialization; setting `DRAMA3_MOE_BATCH=0`
and `DRAMA3_DELTA_TRANSPOSE=0` restores the previous kernels in the same binary.
Both default to 1 and reject values other than 0 or 1.

Risk: CAUTION (performance and recurrent layout). The baseline is `b91ad1e7`;
the rollback flags above or CPU restore the earlier routes. The strongest
falsifier is a kernel-only gain that disappears in complete text-to-WAV or
changes routing, cache continuation, codes, or the readback boundary. Promotion
requires the existing upstream gates, exact previous-native replay, focused
failure/cache checks and matched completed inference timing. Heavy probes run
serially under `scripts/run_safe.sh`; other processes are preserved.

Four same-binary fresh-process AB/BA pairs on M2 Max, `Hello.`, seed 0,
23 prompt rows and 17 generated frames through EOS, give:

| Completed interval | Median optimized / previous ratio | Time reduction |
| --- | ---: | ---: |
| Autoregressive backbone, 17 rows | 0.8812 | 11.9% |
| All GPU command intervals | 0.9438 | 5.6% |
| Complete inference through WAV writing | 0.9611 | 3.9% |

These are medians of four paired ratios, not ratios of aggregate medians.
Previous complete runs took 22.3971–23.6972 s; optimized runs took
21.5529–21.7429 s, with every pair improving. Both arms use identical synchronous
profiling and GPU-timeline flags (129 commands). Preparation and completion
remain inside the original sidecar timer. One extra ordinary unprofiled pair
(90 commands) took 23.8942 s previous / 19.3659 s optimized, with identical
output; this single pair does not establish a general 19% gain. Host load was
not controlled, and these are bounded local observations.

The isolated actual-shape Delta benchmark uses two warmups and seven AB/BA
pairs of 30 dispatches. Its median GPU ratio is 0.6969 (30.3% less kernel time),
with every final state/output bit equal. This is a small part of backbone
runtime, not a 30% whole-model speedup. Reproduce it with
`examples/drama3_delta_benchmark.cr` and the normal Metal build flags.

All 243 backbone traces, 726 tower traces (including 20 strict terminal gates)
and 30 DAC stage/waveform traces pass the unchanged upstream tolerances and
match the previous native values bitwise. Every full run has the same prompt,
codes, EOS and WAV bytes (SHA-256
`3bcc331bd234ba6a3950d2122215bdc663aa4f7427a112e8f8df108085908a49`).
The normal readback ledger remains 152 control / 140,012 terminal bytes, with
no activation/diagnostic reads. Full MLX PCM replay remains within one count.
New regressions compare Q4/Q8 shared/per-rank projections at unaligned offsets,
duplicate ranks, group tails, invalid IDs/widths, rank-sum cancellation, and
every Delta state/output bit across varied recurrent rows. The focused suite
passes 42 Metal / 18 CPU examples and six Python contracts.

The first submission remains the dominant open latency target. One instrumented
baseline command spends 10.9795 s before GPU start and 0.1348 s on GPU; all
library/function/pipeline API calls together account for about 13 ms. These
timestamps do not distinguish deferred driver work, resource preparation,
pinning, or page faults. No weight-view, hazard-tracking, scheduling or timer
hoisting change is promoted. Optional `DRAMA3_METAL_TIMELINE=1` emits GPU command
intervals; `COGNI_METAL_SUBMIT_PROFILE=1` adds the existing bridge timestamps.
First attribute this pre-GPU interval before choosing another startup change.

Correlated source adversary review is ROBUST within the validated matrix,
layout and arithmetic bounds; executed replay supplies numerical evidence.
The source/binary/input identities, per-pair observations and refresh policy
are in [drama3-resident-speed-evidence.json](drama3-resident-speed-evidence.json).
Refresh on the source, precision, assets, compiler, input, device/OS or artifact
availability changes listed above. Long-form, concurrent/reset, stock BF16,
subjective quality, realtime and general cross-device speed remain unclaimed.

## Historical hybrid Metal slice (`9622936a`)

The following records the preceding hybrid source state. Its implementation
and timing-boundary statements apply to that commit, not the current resident
CLI. Its preserved evidence is [drama3-metal-evidence.json](drama3-metal-evidence.json).

This slice profiles the existing CPU execution before choosing kernels.
Risk: CAUTION (parallel reductions and CPU/GPU ownership). Rollback is the
explicit CPU backend and the preceding native implementation commit.

- Measure backbone prefill/decode, tower teacher forcing/sampling, and DAC
  separately, including per-operation inclusive/exclusive host time. Record
  loading/setup separately from inference. Synchronous Metal timings must
  include transfer and command completion; kernel-only time is supplemental.
- First candidate: packed MLX affine projections and/or DAC convolutions,
  selected by measured cost. Preserve Float32 weights/activations and the
  existing cache and sampling semantics. No full-model dense expansion.
- The strongest failure mode is accumulated numerical drift that changes MoE
  routing or audio codes despite a plausible final waveform. Retain the
  upstream intermediate traces as gates at their existing scoped tolerances;
  do not widen them to make an optimization pass.
- DoD: profile all three components on the pinned checkpoint; accelerate a
  measured bottleneck with a selectable Metal backend; pass the synthetic and
  real-weight intermediate gates and the bounded native text-to-WAV smoke;
  report measured component and total latency including synchronization.
- Mini-Quadrumvirate at each key transition: predict the bottleneck/failure,
  test the discriminating assumption, implement the bounded move, and attack
  both parity and performance. A failed parity gate rejects that kernel path.

Performance evidence is local to this M2 Max, input, source, backend, and host
load. It expires after changes to kernels, dispatch, precision, weights, or
workload shape. CPU and Metal runs use the same bounded inputs; speed does not
substitute for correctness or establish subjective speech quality.

### Build and profiling

```sh
make build/bridge.o
crystal build --release -Ddrama3_metal examples/drama3_generate.cr \
  --link-flags="$(pwd)/build/bridge.o -framework Metal -framework Foundation -lc++ -fuse-ld=/usr/bin/ld" \
  -o /tmp/drama3_generate_metal
DRAMA3_BACKEND=metal DRAMA3_PROFILE=/tmp/drama3-metal-profile.json \
  COGNI_RUN_SAFE_MIN_FREE_PCT=25 RUN_SAFE_PASSTHROUGH_STDIO=1 \
  scripts/run_safe.sh /tmp/drama3_generate_metal 300 40960 \
  --model "$MODEL_DIR" --text 'Hello.' \
  --output /tmp/hello-metal.wav --max-frames 128 --seed 0
```

`DRAMA3_BACKEND=cpu` is the default, including in a Metal-capable binary. An
explicit Metal request fails if the binary/device cannot support it; it does
not silently fall back. Metal requires Darwin, unified memory, and a device
whose maximum buffer length can cover each mapped shard. All arithmetic in
this slice remains FP32, with packed Q4/Q8 weights and raw F32/BF16/F16 affine
scales/biases. Dense projections, attention, routing, gating, and sampling
continue on the CPU/Accelerate path.

`DRAMA3_PROFILE` enables nested synchronous host spans. JSON records each
scope path, category, call count, inclusive nanoseconds, and exclusive
nanoseconds. Sum disjoint component scopes; summing parent and child inclusive
times double-counts work. Metal spans include binding, submission, completion,
and readback; first-use mapping and pipeline compilation are included in
their enclosing spans. `weights.load` and `generator.initialize` are separate
setup spans. The WAV sidecar's `elapsed_seconds` starts after setup and ends
after WAV writing; it is inference latency, not process cold-start latency.
Profiling is disabled when the variable is absent.

`examples/drama3_trace.cr` and `spec/support/drama3/native_tower_trace.cr` accept
the same backend/profile variables. `examples/drama3_codec_trace.cr` accepts
`MODEL_DIR CODES_JSON TRACE_JSON [WAV_OUTPUT]` to decode fixed generated frames;
use `-` for `TRACE_JSON` to omit the potentially large trace bundle.

### Implementation and guards

The packed affine kernel computes each group's subtotal in CPU order, then
adds groups in ascending order. Contraction/fast math is disabled. Independent
gate/up projections share one command buffer; the down projections share the
next. CPU routing and final expert accumulation retain their original order.
Mapped weights remain packed and retain their mmap owners. Byte loads and
64-bit offsets support the actual checkpoint's unaligned U32 tensor starts
and scale offsets beyond 4 GiB. A real-weight replay rejected the initial
alignment assumption before this correction; focused tests cover odd byte
offsets, mmap tails, 65 groups, expert selection, reset, and F16 subnormals.

DAC Conv1d uses one output owner and the CPU accumulation order. Transposed
convolution gathers contributions in the original input-time/channel order,
with the same causal crop and bias placement. No atomics or reordered
reductions are used. Weight normalization remains on the CPU. Codec weight
buffers have a 128 MiB / 128-entry cache with value-change detection. Shared
scratch is serialized from host write through GPU completion and readback.
When unloading a model in a library host, call `CodecMetal.clear!` and
`MetalBackend.reset!` between inferences to release retained device views.

### Observed M2 Max result

The final matched pair uses the same release binary, pinned Q4 checkpoint,
`Hello.`, seed 0, 23 prompt rows, and all 17 generated frames through EOS.
Times below are disjoint component scopes and inference wall time in seconds:

| Scope | CPU | Metal enabled |
| --- | ---: | ---: |
| Backbone, 40 steps | 101.54 | 35.36 |
| Tower, 22 teacher-forced + 17 sampled frames | 7.84 | 11.27 |
| DAC | 141.07 | 4.91 |
| Generation + DAC + WAV write | 250.55 | 51.61 |

This pair observes 4.86x total acceleration and 28.72x DAC acceleration. Tower
arithmetic remains on CPU; these measurements do not establish a tower gain.
The host had unrelated CPU work and Spotlight activity, which were preserved.
There is no quiet-host or statistical performance claim. The earlier batched
Metal run took 48.54 s; the individual-submission Metal stage took 62.82 s.
Expert batching reduces 38,400 waits to 3,200, plus 44 DAC commands. It bypasses
`Matrix#matvec`, so compare `metal.affine_batch` and enclosing MoE counts,
rather than expecting Metal's `matvec.quant` call count to match the CPU's.

Both final backends reproduce the same prompt IDs, every code frame, EOS,
and WAV bytes. All 243 backbone traces and 726 tower traces equal the prior
CPU results exactly, then pass their existing upstream gates. Real two-frame
MLX DAC gates and the synthetic unequal-upsampling probe pass. Full 17-frame
MLX DAC PCM comparison retains the previous maximum one-count PCM16 error.
Final checks: 25 Metal examples including real codec parity, 18 CPU examples,
four focused synthetic codec examples, and six Python contract checks; no
failures, errors, or pending Crystal examples.

First-use preparation is substantial: the final Metal run's first prefill row
takes 23.48 s; later rows take about 0.38 s. Dense CPU projections and tower,
and bounded per-expert GPU weight views, are the next candidates. Separate
mapping/page preparation from lazy floating conversion before attributing
startup cost. This slice does not establish real-time generation.

The measured CPU baseline and both Metal stages, exact-output checks, source
hashes, commands, and remaining bottlenecks are in
[drama3-metal-evidence.json](drama3-metal-evidence.json). Existing upstream
tolerances above remain unchanged. A correlated read-only adversary review
found no in-scope indexing, ownership, or expert-ordering blocker; numerical
and timing claims rely on the executed replays, not that review alone.
