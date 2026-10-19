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
