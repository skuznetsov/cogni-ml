# DRAMA-3 generation buffer reuse

Status: VERIFIED for the bounded storage, allocation and numerical checks below.
End-to-end speed improvement is not established; the default remains fresh
allocation (`DRAMA3_GENERATION_SCRATCH=0`). Source base: `ce592825`.

## Ownership and execution

`DRAMA3_GENERATION_SCRATCH=1` enables four generator-owned workspaces:
`backbone`, `tower.prefill`, `tower.sample` and `main`. Each allocation within
one component call gets its own ordinal backing slot, including simultaneously
live residuals. The first call allocates the slots; subsequent calls reuse them
on the same tracked serial Metal queue, including with two pending graph
commands. No reuse occurs within one call. Dynamic tower attention scores
reserve configured capacity while kernels retain their original active length.
This is ordinary GPU storage reuse, not an LTP/WBA certificate.

Final hidden state is consumed by tower/main before the next backbone call.
Generated frames escape into the separate persistent frame buffer before the
next step. Host uploads, converted immutable weights and sticky finite status
always allocate outside the workspaces. GPU zeros are filled on every use.
The generator disables reuse when a diagnostic observer is already installed;
late trace enablement is rejected before a borrowed buffer can escape.
Standalone component calls retain fresh storage unless explicitly scoped.

Workspace entry/exit adds no submit, finish, wait or readback. Explicit release
uses the existing terminal queue boundary. Growth retains replaced backings
until that boundary too; abort quarantines current and retired storage for the
process lifetime. Foreign-context, nested, active-release, aborted and released
uses reject. The native command wait calls Metal `waitUntilCompleted` before
checking terminal error status. Recovery from foreign queue implementations or
hardware loss is outside this certificate.

The four plans occupy 22,110,080 bytes (21.09 MiB) in the short tested job. They
are released before generation returns, with zero live workspace bytes/slots.
Plans are per generation, not pooled across studio jobs. Crystal wrappers,
command metadata, persistent state, input preparation and codec temporaries
still allocate; GC remains enabled. A stable repeated component scope can have
zero new backing allocations without the entire request being allocation-free.

## Allocation attribution

`DRAMA3_ALLOCATION_PROFILE=1` adds monotonic host clocks only around actual
`MetalBuffer.new` construction. It measures driver/backing construction, not
all wrappers, encoding, GC or pipeline time. Counts and requested bytes remain
available with this timer off. Studio reports `allocation_phases` for model
initialization, generation and codec, plus `resident_generation_scratch` plan
statistics. The synchronous component profiler was disabled in these runs.

Serial runs used Apple M2 Max/64 GiB, macOS 26.6.2 (25G83), Crystal 1.22.0,
MLX 0.31.1 and graph depth 2, under `scripts/run_safe.sh` with a 15% free-memory
floor and the quiet-host check off. Unrelated processes were untouched.
Checkpoint: `cguldogan/drama-3-mlx-q4` at
`5331ab509359c0e4a7e353a19e255ddf3e8152a8`. Private mmap, DAC scratch and tower
embedding fusion were held at 0; arithmetic, precision and reduction settings
were unchanged. The host and GPU were not isolated from other projects.

All full jobs used “Hello, this is a voice test.”, seed 1, maximum 64 frames,
temperature 0.8, top-k 30 and top-p 0.9, without a reference. They stopped at EOS
after 60 frames (2.7864 seconds of WAV). The native 704-element F32 sampler tape
has SHA-256 `e13a73aa1f41d156e7c2f811bb941c1b63993aa97989b267a14b65a5665a545b`.

| Warm generation measurement | Fresh | Reuse |
| --- | ---: | ---: |
| GPU backing allocations | 135,139 | 1,894 |
| Requested bytes | 1,596,632,160 | 22,125,244 |
| Construction time, two timer-on samples | 819–892 ms | 7.15–9.37 ms |

The count falls 98.60%; the timer is diagnostic rather than a speed certificate.
The six timer-on jobs and eight timer-off jobs used the v1 implementation.
Final growth/quarantine retention and fresh-path capacity-validation guards
were added afterwards; four additional full jobs and the focused specs verify
the final implementation. The final full jobs reproduce both allocation counts.
The raw manifests retain each source state; v1 timings are not relabelled as
measurements of final source.

## Timing decision

The timer-off v1 order was fresh/reuse/reuse/fresh, each with a cold and warm
request in a persistent worker. Four final-source requests followed in
reuse/fresh order, again cold and warm. Model loading is outside request time;
codec and output writing remain inside it. Cold jobs also construct pipelines
and converted tensors, so compare warm jobs separately.

| Timer-off warm request | Fresh | Reuse |
| --- | ---: | ---: |
| v1 generation median, two jobs per mode | 22.515 s | 21.724 s |
| v1 request-to-WAV median, same jobs | 28.588 s | 29.546 s |
| Final-source generation, one job per mode | 21.563 s | 25.769 s |
| Final-source request-to-WAV, same jobs | 27.618 s | 31.935 s |

These small shared-host samples do not support a stable total speedup, or a
claim that the final-source slowdown has a unique cause. Keep the opt-in flag.
The existing completed-command timestamps total about 21.5–21.7 seconds per
timer-off full job. For this fixed trajectory only, the first 90 commands cover
generation (about 15.5 seconds) and the last 32 cover DAC (about 6.0–6.1 seconds).
No added phase fence was used. Whole-request GPU totals must not be subtracted
from generation-only host times.

The next discriminator is host encoding/bookkeeping/GC versus DAC host-time
variation, with the same paired workload and existing terminal timeline. More
allocation-count reductions alone do not justify default promotion. This slice
does not change the older DAC arena default or close its numeric frontier.

## Numerical and lifetime verification

All 18 paired full native jobs retain the same 60 frames/all 11 IDs, actual
sampler-input hash and byte-identical WAV. WAV SHA-256:
`38f2530ed1c90f0caba45e222432909ac5d92cc3633e4c34a21fbe39d693e07d`.
Each has 122 submissions/completions/waits, maximum two pending commands,
496 control bytes and 494,160 terminal bytes. Reuse adds no synchronization or
model activation readback. Graph retained-byte accounting counts aliases and
is not physical GPU memory or a peak allocation bound.

A fresh GPU comparison of the final native codes against the saved MLX F32-core
control matches the complete 60-frame trajectory. That control was generated
at 2026-10-16 04:05 UTC from clean pinned MLX source
`e7488935ec2ebfafd38ef2b58e290d79f3d05dc7`; its source and model config/index
hashes were rechecked. Native uploads the first 704 F32 values of the control's
saved 8,448-value uniform tape. The shared effective draws are identical;
this is not stock MLX RNG parity or a freshly regenerated MLX waveform.

The separate 158-row prefill probe at capacity 2205 explicitly runs real
backbone and tower.prefill workspaces; it does not enable a trace that would
silently turn reuse off. After the first call, new backing allocations are zero
for all 157 subsequent backbone calls and 156 subsequent teacher-force calls.
Final persistent hidden (2,048 elements) and logits (4,097 elements) have zero
fresh/reuse bit mismatches. Both pass the unchanged MLX pointwise gate
`abs(actual-reference) <= 3e-5 + 3e-4*abs(reference)` with zero failures or
nonfinite values; worst error/allowance is 0.05127/0.00798 respectively.
Both arms have the same 159 existing submits/waits; terminal reduced comparison
reads add no graph work. Immutable weight conversions are shared between arms,
so whole-arm allocation totals are not a symmetric speed comparison.

The final source passes 49 focused Metal examples: depths 0/2, two pending
commands, forced GC, real slot growth, retired backing retention, sticky finite
status, nonzero offsets, zero refill, ownership, retained diagnostics, late
trace rejection, EOS/three-frame generator handoff and reference prompts.
Another 29 CPU examples cover studio, profiling and memory-budget contracts.
The final release studio build passes. No shader arithmetic changed.

The correlated mini-Quadrumvirate found two issues before closure: replaced
backings needed retention through abort, and disabled scratch must validate
only the active tower-score length. Both were repaired and tested. Final
Adversary verdict is ROBUST for these bounded storage/numeric checks; total
speed promotion remains unsupported. General MoE39/common-prefix/DAC parity,
long-output voice quality, cross-job pooling and whole-pipeline GC removal
remain open.

## Reproduce and refresh

Build the studio using [drama3-studio.md](drama3-studio.md), then start it with
`DRAMA3_GENERATION_SCRATCH=1` in the environment to enable reuse. Keep
`DRAMA3_ALLOCATION_PROFILE=0` for timing comparisons. The standalone native
resident generator uses the same flag.

Focused GPU checks, with the normal Metal bridge object:

```sh
DRAMA3_GENERATION_SCRATCH=1 DRAMA3_ALLOCATION_PROFILE=0 \
COGNI_RUN_SAFE_MIN_FREE_PCT=15 COGNI_RUN_SAFE_REQUIRE_QUIET=0 \
  scripts/run_safe.sh crystal 900 40960 spec -Ddrama3_metal \
  spec/drama3_workspace_spec.cr spec/drama3_tower_workspace_spec.cr \
  spec/drama3_generator_workspace_spec.cr spec/drama3_graph_resident_spec.cr \
  spec/drama3_graph_generator_spec.cr spec/drama3_generator_resident_spec.cr \
  spec/drama3_reference_prompt_spec.cr spec/drama3_tower_resident_spec.cr \
  spec/drama3_resident_spec.cr \
  --link-flags='build/bridge.o -framework Metal -framework Foundation -lc++ -framework MetalPerformanceShaders -fuse-ld=/usr/bin/ld'
```

Curated evidence: `.landmark/evidence/drama3-allocation-reuse.json`. Raw local
commands, source/binary seals, paired reports/WAVs and GPU comparison helpers
are in `/Users/sergey/.cache/cogni-ml/drama3-runs/allocation-reuse-20261016/`.
The packet is an index to observed results, not a structural proof. The first
prefill helper failed on a top-level config lookup before model work; the
corrected helper uses native `TowerConfig` and its successful run is recorded
separately. Failed probes are retained, not counted as passing evidence.

Risk: CAUTION. The main falsifier is overwriting a still-live input, residual,
escaping frame or diagnostic capture. Rollback is flag 0 or the isolated feature
commit. Refresh measurements and gates after source, shader, checkpoint,
fixture/text/tape, precision, graph policy, runtime, device or OS changes. Foreign
queue/hardware-loss recovery and concurrent workspace entry are not admitted.
