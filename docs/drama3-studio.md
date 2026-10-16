# Local DRAMA Studio

DRAMA Studio runs the native DRAMA-3 Metal generator behind a loopback web UI.
Sessions, voice references, immutable takes, job reports and exports stay in a
local data directory. It requires macOS, a converted DRAMA-3 checkpoint and
`ffmpeg` on PATH for reference import and WAV/MP3/Opus export.

## Build and run

From the repository root, build once under the repository resource guard:

```sh
COGNI_RUN_SAFE_MIN_FREE_PCT=15 COGNI_RUN_SAFE_REQUIRE_QUIET=0 \
  scripts/run_safe.sh /usr/bin/make 900 40960 drama3_studio
```

Then start the studio with your checkpoint directory:

```sh
COGNI_RUN_SAFE_MIN_FREE_PCT=15 COGNI_RUN_SAFE_REQUIRE_QUIET=0 \
  scripts/run_safe.sh build/drama3_studio 43200 40960 \
  --model /absolute/path/to/drama3-mlx-q4 --port 8787
```

Open `http://127.0.0.1:8787`. The default data directory is
`~/.local/share/cogni-ml/drama3-studio`; override it with `--data DIR`.
Only one studio process may use a data directory at a time. Stop with Ctrl-C.
The model loads lazily on the first generation, then remains in one native
worker for subsequent jobs. The HTTP server stays responsive during inference.
Do not run additional GPU benchmark jobs concurrently with generation.

## Working with voices and sessions

1. Create a session and add one or more text segments.
2. To clone a voice, upload a clean recording of up to ten seconds and enter
   its exact transcript. The import normalizes it to mono 44.1 kHz PCM16.
   Select that saved reference on each segment that should use the voice.
3. Generate a segment. Advanced settings expose seed, frame budget and sampler
   settings. A fixed seed with the same text/reference/settings reuses the same
   sampler inputs. A changed prompt can still produce different speech.
4. Edit and regenerate just the segment you want to change. Earlier takes remain
   in its history; select any take to use in the final export. Editing during a
   running generation keeps the resulting take in history but does not select
   it automatically for the changed revision.
5. Once every segment has a selected take, export the session as WAV, MP3 or
   Opus. Export concatenates selected takes in segment order without adding
   silence. Download individual takes or exports from the audio library.
6. Rename/archive/restore sessions and archive unused voices. Selected takes,
   files used by active jobs and source takes of existing exports are protected
   from deletion. Select another take or remove the dependent export first.

## Execution and recovery boundaries

Each generation has fresh KV/recurrent state; a web session is an editing
container, not a shared model context. Static model mappings, pipelines and
converted tensors can stay warm between jobs. Reference encoding, model
inference and DAC decoding use the native resident GPU path. The existing EOS
control reads and terminal code/WAV reads remain; the studio adds no activation
readback or per-token GPU synchronization. PCM import/export and file I/O are
host-side boundaries. GC may run between completed jobs.

The frame budget is bounded by the current 4096-position native window, including
prompt/reference rows. The worker rejects requests whose prompt and budget do
not fit. A take that reaches its budget is marked as truncated. References do
not guarantee identical voice identity across all texts or languages.

Jobs execute serially. Errors fail the job and restart the native worker when
needed. A worker that produces no response for five minutes is killed and
reaped; this is a quiet timeout, refreshed by progress messages. Transient
manifest I/O errors pause the consumer and are retried; the UI exposes the
pause. On application restart, previously queued/running jobs
become failed rather than silently replaying. Asset deletion uses a reversible
file rename before committing metadata and recovers pending deletions on
startup. A hard process or host crash between writing an output and registering
it can leave an unregistered WAV or report sidecar; startup does not sweep
unknown files. This is a local single-process manifest store without an fsync
durability guarantee. Back up the data directory while the studio is stopped.

The interface binds to loopback and checks Host and mutation Origin. It has no
multi-user authentication; keep it local. No cloud service or CDN is required.

## Performance experiment

`DRAMA3_TOWER_EMBEDDING_FUSION=1` enables the sequential FP32 GPU codebook-sum
kernel. The default remains the legacy path until end-to-end promotion evidence
is sufficient. The focused benchmark is
`examples/drama3_embedding_benchmark.cr`; it tests real checkpoint BF16 weights,
bitwise output equality, command/readback counts and alternating paired timings.
An isolated kernel improvement does not establish total generation speedup.
Existing upstream MoE and DAC numeric frontiers remain separate from this UI
and are not closed by unchanged output versus the previous native path.
The measured paired results and next performance probe are in
[drama3-tower-embedding.md](drama3-tower-embedding.md).

## Observed validation

The 2026-10-16 run passed 18 CPU examples for store, HTTP, media and worker
behavior, including injected persistence failures and child timeouts. Four
focused Metal examples cover codebook-sum arithmetic, mapped offsets, ownership
and command lifetimes. Eight repeated native jobs retain exact full code IDs,
WAV bytes and readback counts across cold/warm and legacy/fused variants. The
60-frame short-text trajectory also matches the pinned MLX F32 control with
identical F32 sampler inputs. This does not establish stock MLX RNG parity or
close the older MoE/DAC numeric frontiers.

Browser checks submitted three real native GPU jobs, including two with a
reference. They exercised two persistent sessions, segment editing and history
selection, voice import, archive/restore, WAV/MP3/Opus export and an exact Opus
download. The WAV export's PCM exactly matches the selected takes concatenated
in order. Muted Chromium playback checks decode and advance all saved formats.
Editor refresh checks cover dirty drafts, focus/caret, disclosures, voice choices
and the chosen export format. These checks establish functionality, not a
listening or voice-identity quality certificate.

CPU API/store/media checks:

```sh
COGNI_RUN_SAFE_MIN_FREE_PCT=15 COGNI_RUN_SAFE_REQUIRE_QUIET=0 \
  scripts/run_safe.sh crystal 900 40960 spec -Dcpu_only \
  spec/drama3_studio_store_spec.cr spec/drama3_studio_media_spec.cr \
  spec/drama3_studio_http_spec.cr spec/drama3_studio_worker_spec.cr
```

The admitted scope and verification requirements are in
[drama3-studio-frontier.md](drama3-studio-frontier.md).
