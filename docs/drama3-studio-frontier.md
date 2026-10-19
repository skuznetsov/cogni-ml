# DRAMA-3 local studio frontier

Status: verified for the bounded local workflow below. Generation quality and
general upstream numeric parity remain separate open frontiers.

The local studio extends the native Metal path with persistent editing sessions,
reference voices, immutable audio takes, a serialized generation queue and local
WAV/MP3/Opus export. A session is an editable document, not reused model KV state.
Each text segment generates independently; concatenation does not promise voice
continuity, sentence alignment or seamless crossfades.

Voice references can be uploaded or recorded through a user-initiated local
microphone flow. Both are previewed before the existing import API saves them;
recordings use mono PCM16 WAV at the browser AudioContext sample rate and stop at
9.5 seconds under the backend's ten-second limit. Permission is requested only
after Record is clicked, and the editable transcript remains the user's exact
text. Example readings offer English and Russian phonetic variety without
claiming complete sound coverage or guaranteed voice quality. Synthetic capture
tests do not establish microphone behavior across physical devices or browsers.

The server binds to loopback only. All audio, transcripts and metadata stay in
the selected local data directory. Browser requests cannot choose filesystem
paths or model paths. Same-origin mutation checks, bounded requests and
identifier validation protect the local API. No network hosting or user accounts
are admitted. Reference conversion uses a local ffmpeg process; native inference
still uses the GPU DAC encoder and resident prompt handoff.

Edits carry a revision. A queued job holds an immutable text/settings/reference
snapshot. A late result is retained as a take but must not replace the selection
of a subsequently edited segment. Previous takes remain playable and selectable.
Export snapshots the selected takes in document order. Missing takes reject the
export; an incomplete model result is explicitly labeled as truncated. Startup
marks interrupted queued/running jobs failed rather than replaying them silently.

One persistent worker loads one pinned model. Only one job uses its GPU state at
a time; each generation receives fresh recurrent/KV state. Model mappings,
converted static tensors and compiled pipelines may be reused between jobs.
No activation or sampling arithmetic changes are authorized by the studio slice.
Existing control reads and terminal WAV output remain; no new activation reads
or completion waits may enter the generation loop. Cleanup between jobs is not
a claim that the entire application is allocation-free or GC-free.

Risk: CAUTION. The main failure is stale asynchronous work overwriting newer
edits, followed by state leaking between repeated inferences. Rollback is the
isolated feature commit and the existing standalone generator remains available.

DoD and falsifiers:

- Store tests: revision conflict, late completion, immutable take selection,
  restart recovery, invalid IDs, referenced-asset deletion and atomic reload.
- HTTP tests: origin/Host checks, body limits, traversal, missing assets, invalid
  generation settings and explicit backend failures.
- Export tests: ordered WAV samples, actual MP3/Opus decoding, missing-take
  rejection and conversion failure without a false successful asset.
- Browser test: create two sessions, upload a reference with transcript, edit
  segments, generate, select an earlier take and export/download audio.
- Recorder tests: exercise the worklet with synthetic samples and fake streams;
  verify permission cancellation, stale completion cancellation, WAV encoding,
  duration bounds and track cleanup without opening a physical microphone.
- Native repeated-job probe: identical seed/text yields identical code IDs and
  WAV bytes, unchanged sampler tape/readback ledger; compare the existing pinned
  upstream fixture with unchanged tolerances. Measure cold and warm phases
  separately. A faster startup is not a claim of faster token generation.

Separate hot-path performance changes require paired measurements and output
equality before promotion. Existing MoE/DAC upstream numeric discrepancies stay
open; the studio cannot turn a successful process exit into numeric parity.

Observed verification on 2026-10-16: 18 CPU examples and four focused Metal
examples pass. Eight native repeated jobs have identical full codes and WAV
bytes across legacy/fused and cold/warm runs; the shared F32 sampler tape yields
60 exact frames against the pinned MLX F32 control. First-prefill hidden/logit
gates pass at the unchanged tolerances. The browser submitted three real Metal
generations (one without and two with a reference), selected earlier takes,
exported WAV/MP3/Opus, downloaded Opus and managed two sessions across restarts.
Ordered WAV PCM equals the selected source takes exactly. Failure probes cover stale
revisions, persistence faults, interrupted jobs and hung child processes.

The correlated mini-Quadrumvirate verdict is ROBUST for these bounded results,
VULNERABLE for broader identity/prosody, crash durability or performance claims.
Hard crashes may leave unregistered output files; the manifest has no fsync
guarantee. Refresh the evidence after source, checkpoint, runtime, graph policy,
input/tape or tolerance changes. Curated receipts live in
`.landmark/evidence/drama3-studio.json` and
`.landmark/evidence/drama3-tower-embedding.json`.

## Single-model ownership and request memory frontier (2026-10-18)

This slice admits a per-user lease on the canonical checkpoint directory,
acquired by the native worker before loading weights, independently of server
port or library directory. It prevents cooperating Studio workers from loading
the same model twice; it does not govern standalone generators or other users.
The existing per-library lock and serialized queue remain necessary.

Request memory observations must distinguish model caches from temporal buffers.
Record OS footprint, driver-reported Metal allocation, tracked buffers and static
cache identity at request boundaries and after the existing between-job GC.
Never add footprint, RSS and Metal bytes as disjoint memory. Budget and 15% free
headroom checks are sampled stop guards, with a separate process-group watchdog;
neither is a hard physical peak guarantee. They add no GPU waits or tensor reads.

Optional automatic windows use the existing advisory planner before allocating
KV state, bounded by the requested frame cap and 4096 total positions. Output
length remains determined by EOS, not a text-duration prediction. The initial
integration keeps the conservative cold static-weight allowance on warm jobs,
requires the calibrated DAC scratch layout, and rejects reference encoding and
unknown telemetry. Manual settings and old manifests retain their behavior.
Mid-generation cache relocation and automatic reference windows are not admitted.

Risk: CAUTION. The strongest failures are an ineffective cross-process lease,
mistaking a growing cache/high-water mark for a temporal leak, and underestimating
automatic-window memory. Rollback is the isolated change; manual frame bounds
and the standalone generator remain available.

DoD: CPU lease contention/release/alias tests, repeated fake-worker PID reuse,
memory threshold/unknown-telemetry tests, and persisted immutable auto settings.
Then build the native Studio and run three identical short HTTP jobs serially in
one worker under run_safe (15% floor, 40960 MiB RSS limit, finite timeout). Compare
post-GC live allocations/cache identity, full codes/WAV/sampler/readback ledgers,
then a bounded auto/manual capacity control and the pinned upstream fixture at
unchanged tolerances. Stop on any failed job or breached observed budget; reap
the worker and server and verify listeners are gone. A short plateau establishes
only the tested request count, capacity and source/checkpoint state.

Observed result: the first serial HTTP pilot completed three manual 128-frame
jobs in worker 84823. Its fourth, automatic request rejected with
`InsufficientBudget`; the supervisor stopped and cleanup passed. The second
pilot completed a cold automatic request followed by three manual requests in
worker 99716. Both pilots rejected a second worker before its ready event.
The converted cache kept the same 566 tensor identities within each worker.
No job retained pending graph work; DAC scratch was released. Readbacks stayed
at 496 control bytes and 494160 terminal bytes per successful job.

Post-GC tracked bytes in the first pilot were 26610326412, 26610338704 and
26610338704. In the second they were 26610271116, 26610271116, 26610404364 and
26610420748. The last pair therefore grew by 16KiB; the three manual jobs in the
second pilot grew by 149632 bytes in total. Driver-reported Metal allocation
also drifted separately. These short samples do not establish zero retention
or long-running leak freedom; the residual allocation identities remain open.
The predeclared 32MiB warm-pair drift limit passed, but is only a regression
guard. Minimum sampled free/reclaimable memory was 20.0339%, above the 15% floor;
maximum pre-GC sampled Metal allocation was 27029962752 bytes (25.174GiB).

The cold automatic plan admitted 128 generated frames and 156 total positions,
including the 29-row prompt, with an estimate of 28048786244 bytes and a budget
of 32569809305 bytes. It emitted 60 frames and EOS, exactly matching the manual
requests in codes, WAV and sampler-input digest. The warm rejection is retained:
raw mapping identities do not certify physical page residency, so subtracting
the cold mapped-weight allowance is not admitted. Reference auto sizing and
cache growth during generation remain outside this slice.

A fresh pinned MLX control used the same text, seed, sampler settings, prompt
rows, 128-frame cap and F32 sampler-tape prefix. All seven native outputs matched
its full 60 generated code frames and EOS exactly. This is a sampled-token
control, not hidden/logit or waveform numeric parity; the existing layer-39
upstream discrepancy and strict tolerances remain unchanged. The 41 CPU specs,
17 JS tests, formatting checks and native Studio build passed. Both pilot
servers/workers and the sequential MLX control exited; listeners were absent.

Evidence: `.landmark/evidence/drama3-studio-memory.json`. Private reports,
comparison analysis, harness and audio remain under
`/opt/mldata_ssd/cogni-ml/drama3-studio-memory-20261018/`. Refresh after source,
checkpoint, capacity, input/tape, runtime/device, allocation policy or concurrent
workload changes. Timings include telemetry and are not a performance claim.

## Growing resident windows (2026-10-18, bounded verification)

This slice supersedes the previous exclusion of cache growth. Automatic
no-reference requests may start with a smaller KV capacity and grow geometrically
up to their admitted final position limit, at the existing non-EOS scalar-control
boundary. The output frame cap, sampler tape, prompt and attention reduction path
remain fixed for the request. Prefill fits in the initial capacity. Manual windows
keep their existing allocation policy; automatic reference windows remain excluded.

Backbone token-major prefixes and tower head-major prefixes must be copied exactly
on the GPU, with the latter relocated using both capacity strides. Persistent
replacement buffers cannot use temporary workspace slots. Bound command leases
retain old storage until copies complete; no additional submit, wait, tensor read,
GC or CPU activation copy is admitted. A request whose final capacity exceeds 1024
uses the wide attention implementation from its first row, even while its backing
is smaller, so growth never changes reduction order.

Risk: CAUTION. Admit the full request before allocation, including the conservative
cold static inventory, downstream DAC reserve, and every earlier cache generation
in the planned growth schedule without assuming prompt GC reclamation. Recheck
current host/Metal/process headroom before each replacement allocation, retaining
the 15% floor, configured observed budget and per-buffer limits. Reject before
allocation on failed admission; an allocation/copy failure aborts the worker.
These remain sampled guards rather than physical reservations. Rollback is manual
allocation and the isolated commit.

DoD: require bit-identical
valid caches, generated codes, EOS and sampler inputs against fixed final capacity
at graph depths 0 and 2, including multiple growths, head-major relocation,
1024/1025, prompt/last-frame bounds and EOS before growth. Rejected growth must
allocate nothing; inference readback and completion-wait counts must not increase.
Run bounded serial full-model WAV controls under the 15% floor and 40GiB observed
budget, compare with the same pinned upstream control at unchanged tolerances,
and stop/reap owned processes after the probe. The 149632-byte post-GC residual
is parked and is not part of this slice.

Observed result: 49 CPU examples, 17 JS tests, seven generator Metal examples,
five cache Metal examples, formatting/syntax checks and the native Studio build
passed. Generator tests cover repeated expansion, EOS before expansion, final
frame bounds and admission/allocation failure at graph depths 0 and 2. The cache
tests compare real two-head relocation and wide-attention output exactly across
1024/1025 using a synthetic cached prefix; they do not execute 1024 full-model
steps. A three-row diagnostic compared 243 backbone boundaries per arm, with
729 comparisons: growing 1 -> 2 -> 4 and fixed capacity 8 were bit-identical, and
both arms had identical upstream metrics. The first strict MLX failure remains
row 1, `backbone.layers.39.hidden`; tolerances were not widened.

Two serial HTTP pilots each completed four full-model WAV requests in one
worker: cold automatic capacity 64 -> 128 followed by three fixed-capacity 540
controls, with a 29-row prompt and a 512-frame cap. All eight emitted 60 frames
and EOS; native full codes, WAV bytes and sampler inputs matched exactly.
Readbacks remained 496 control and 494160 terminal bytes per request. Submitted
commands and completion waits matched within each graph depth: 182 at depth 0,
122 at depth 2, with no pending work after completion. Each pilot rejected a
second model owner before loading; servers and workers exited normally, with
no listeners or owned children remaining. A fresh pinned MLX F32 core control
with the same text, cap and sampler tape matched all 60 code frames and EOS.
This does not establish upstream waveform or general hidden/logit parity.

Minimum sampled free/reclaimable memory was 20.3375% / 20.2565% and maximum
sampled Metal allocation was 27052982272 / 27066695680 bytes at depth 0 / 2.
The 15% floor, 40GiB observed process limit and finite deadlines were unchanged.
Earlier depth-2 attempts rejected before replacement allocation
(`InsufficientHostBudget`, cap 512) and before generation/cache allocation
(`InsufficientBudget`, cap 128). Those failures did not persist numeric admission
reports, so their exact causes remain unproven; the successful recheck used the
same binary without relaxing guards. Admission is sampled, not a reservation.

The correlated mini-Quadrumvirate verdict is ROBUST for guarded Studio growth
within this tested no-reference scope. Direct low-level generator callers own
request planning and growth admission. Full 4096-position model runs, automatic
reference windows, sustained leak freedom, voice quality and speed gains remain
outside the claim. Manual mode remains the default. Evidence:
`.landmark/evidence/drama3-window-growth.json`; raw receipts and audio remain
under `/opt/mldata_ssd/cogni-ml/drama3-window-growth-20261018/`. Refresh after
source, checkpoint/config, runtime/device/driver, allocator/graph policy, capacity,
input/tape or tolerance changes. Admission also depends on current host workload.
