# DRAMA-3 local studio frontier

Status: verified for the bounded local workflow below. Generation quality and
general upstream numeric parity remain separate open frontiers.

The local studio extends the native Metal path with persistent editing sessions,
reference voices, immutable audio takes, a serialized generation queue and local
WAV/MP3/Opus export. A session is an editable document, not reused model KV state.
Each text segment generates independently; concatenation does not promise voice
continuity, sentence alignment or seamless crossfades.

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
