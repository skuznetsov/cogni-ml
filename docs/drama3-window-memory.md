# DRAMA-3 window and scratch memory frontier

Status: implementation present; measured scope and remaining gates below.

The active slice is an opt-in window of at most 4096 total resident positions,
including the prompt, and an experimental memory-based frame budget. The prior
implementation supported 1024 total positions and fresh DAC allocations; 768
generated frames were exercised. A model configuration limit is not a native
capacity certificate.

The largest current DAC decoder activation is 768 KiB per audio frame. Fresh
residual units allocate five outputs while retaining the skip input; GC can
retain storage after graph completion. The decoder arena has three
preallocated GPU slabs, excludes both the current input and live residual skip
from destination selection, and reuses storage on the same ordered Metal queue.
Only the separate mono result escapes. Diagnostic traces retain fresh immutable
buffers. Arena release follows the existing terminal finish or error-path drain,
never an additional inference synchronization.

The attention extension preserves all scalar dot/softmax/value reduction order.
The existing <=1024 kernel remains the regression control. Larger contexts use
persistent device scores with explicit in-kernel visibility barriers. KV and
score products must be checked before allocation, including maxBufferLength.

Automatic sizing is advisory admission, not reserved system memory. Before model
work it budgets whole mapped checkpoint bytes once, converted static tensors,
recurrent state, backbone/tower caches, decoder slabs, other codec activations,
outputs and runtime margin. It requires live OS and Metal telemetry and retains
at least 15% physical memory plus an additional margin. The serial run_safe
pressure guard remains required for shared-host experiments. Missing telemetry,
overflow, insufficient headroom and unsupported shapes reject; no precision,
CPU inference fallback or tolerance changes are permitted.

Falsifiers and DoD:

- Pure budget specs: exact-fit/one-byte-short, reserve and prompt subtraction,
  unknown telemetry, overflow, per-buffer limits and monotonic selection.
- GPU attention: old/new outputs at <=1024, actual lengths 1025/2048/4096,
  no head aliasing, graph depths 0/2, unchanged scalar oracle tolerance.
- DAC: fresh/arena output bits and submission/readback ledgers equal at depths
  0/2; repeated decode and retained diagnostic captures remain valid; abort
  drains before release. Compare actual checkpoint stages with pinned upstream
  at existing gates, reporting any pre-existing numeric frontier separately.
- Serial run_safe probes measure Metal allocated bytes and process footprint,
  then exercise 2048-frame DAC allocation and a full native generation within
  admitted capacity. Short/EOS output is not evidence of full-cap DAC use.

Risk: CAUTION. The strongest failure is overwriting a still-live residual or
diagnostic alias, followed by an optimistic budget under changing host load.
Rollback: disable the opt-in arena/auto flags or revert this feature commit.
Rejected: unbounded/rolling context, >4096 total rows, allocation guarantees on a
shared host, whole-pipeline GC elimination and full-generation upstream parity.
Refresh evidence on source, checkpoint, fixture, device/OS or gate changes.

## Allocation and lifetime contract

`DRAMA3_DAC_SCRATCH=1` opts into the arena; the default remains `0`. The three
buffers are allocated once per decode and reused across decoder layers, then
released explicitly after the existing finish. This removes GC from the large
decoder backing-storage lifetime. It does not disable Crystal GC or eliminate
allocations elsewhere in generation, quantizer/post-transformer work, command
metadata or host output construction. Reusing the arena between separate
decodes is outside this slice.

Backbone recurrent state and backbone/tower KV caches already have persistent
allocations. The later opt-in generation workspaces now reuse the dense,
normalization, attention and feed-forward `Context#alloc` temporaries across
steps while preserving residual inputs, pending submissions and diagnostics;
see [drama3-allocation-reuse.md](drama3-allocation-reuse.md). That flag remains
off by default. Disabling GC globally still does not establish safe reuse for
other allocations or across separate jobs.

The current input and residual skip are excluded by backing Metal handle, so
views cannot conceal an alias. Traces use fresh storage because observers may
retain activations. Commands use tracked resources, serial encoders and one
ordered Metal queue. No CPU activation handoff or additional completion wait is
introduced. Failed drains quarantine the arena for the process lifetime instead
of releasing potentially in-flight memory. Depth-zero error ledgers cannot
certify a successful drain and therefore quarantine conservatively.

## Paired checkpoint DAC measurements

Apple M2 Max, 64 GiB, the same saved 768-frame native code sequence and the same
probe binary. Runs were serial under `scripts/run_safe.sh`, pressure floor 15%,
private mmap, with allocation profiling in both variants. Order was fresh then
arena at depth 0, arena then fresh at depth 2. Shared-host timings are
exploratory; these runs do not establish a quiet-host performance comparison.

| Graph depth | Allocation | Decode seconds | Requested GiB | Sampled peak Metal GiB |
| --- | --- | ---: | ---: | ---: |
| 0 | Fresh | 82.531 | 27.004 | 21.855 |
| 0 | Arena | 108.817 | 3.549 | 5.276 |
| 2 | Fresh | 87.173 | 27.004 | 25.030 |
| 2 | Arena | 106.528 | 3.549 | 5.276 |

All four terminal F32 outputs have SHA-256
`80edbb7227017acdf411143298fd007208244504d2965cfaf8e82a8e4f83fe63`.
Each has the same 32 submissions, 32 existing completion waits, four control
bytes and 6,291,456 terminal bytes; terminal reads add no graph work. The arena
uses 1.6875 GiB in three slabs, reuses slots 68 times and releases them before
the output read. After decode, sampled Metal allocation is about 2.45 GiB with
the arena, versus 21.86/25.03 GiB with fresh storage retained until GC/process
exit. Allocation requests fall 7.61 times; sampled peaks fall 4.14/4.74 times.

Both arena timings are slower (32% and 22%). Memory reduction is established
for this input; speed improvement is not. No default performance promotion is
made. Sampled Metal resource bytes and process footprint are distinct metrics,
and neither the allocation counter nor graph retained-byte ledger is a global
physical-memory peak bound.

## Experimental automatic window

Use `--max-frames auto --frame-cap N --memory-plan-only` to inspect admission
without model inference. Remove `--memory-plan-only` to generate with the
selected bound. The default remains manual 160 frames. Automatic admission
requires the decoder arena, live telemetry, the checkpoint inventory and the
existing model shapes; reference-audio encoder sizing is rejected until
calibrated. Total resident positions are prompt rows plus generated frames
minus one, with a hard ceiling of 4096.

Mapping inventory counts each owner once at OS page granularity. The estimate
includes whole checkpoint mappings, persistent F32 conversions, recurrent and
KV state, the three decoder slabs, modelled command-group temporaries and
outputs. Free plus inactive OS pages are headroom observations, not guaranteed
allocatable memory. Driver allocations, changing host load and ordinary
temporary retention until GC are not certified. The report explicitly marks
the plan advisory and host GC retention uncertified. Keep the serial pressure
guard; a successful plan is not a reservation or an OOM guarantee.

The first live plan-only probe rejected with `InsufficientBudget` before inference:
35.10 GiB free plus inactive, 24.09 GiB mapped checkpoint allowance and
1.47 GiB converted-static allowance, before caches and other temporaries.
The additional physical reserve and margin are not waived on a busy host.
After headroom increased to 38.12 GiB free plus inactive, the final-source
plan-only probe admitted 687 generated frames (844 total positions): an
estimated 27.523 GiB within its 27.524 GiB budget. Both live admission outcomes
were observed; automatic generation itself was not run. Neither outcome reserves
memory or certifies ordinary temporary retention until GC.

## Intermediate correctness checks

The final focused suites pass 17 CPU examples (budget, mapping inventory and
native math) and 18 GPU examples (resident math, wider attention, DAC scratch
and mapping). The GPU suite includes exact legacy/wide attention output at
1024 positions, scalar-oracle gates at 1025, 2048 and 4096, repeated scratch
decode at graph depths 0/2, retained traces and late-error/quarantine behavior.

The pinned MLX first-prefill fixture for the 158-row news prompt passes at
resident capacity 2205 (2048-frame budget) without changing the existing
`3e-5 + 3e-4 * abs(reference)` pointwise gate. Hidden/logit maxima are
`4.7683716e-6` / `1.4305115e-5`, with zero violating or nonfinite elements.
Terminal metric reads add no graph work. This probe generates one frame; it
does not establish 2048-step model parity. The fixture pins MLX-audio
`e7488935ec2ebfafd38ef2b58e290d79f3d05dc7` and the q4 checkpoint revision
`5331ab509359c0e4a7e353a19e255ddf3e8152a8`.

Two adversarial per-buffer fixtures initially rejected the estimator: linear
convolution history and the doubled full-attention q-projection could exceed
the device buffer limit without being included in its maximum. Exact-fit and
one-byte-short regressions now pass after adding their checked byte counts.
These repairs do not certify unmodelled GC or driver retention.

## Longer decoder replay and upstream frontier

A full 2048-frame DAC replay completed at graph depth 2 with three scratch
slabs totalling 4.5 GiB, 68 slot reuses and explicit release after the existing
finish. It returned 4,194,304 samples (95.109 seconds at 44.1 kHz) in 237.252
seconds. Sampled peak Metal resource allocation was 9.639 GiB; after-decode
Metal allocation was 2.521 GiB. The readback ledger contains four control bytes
and 16,777,216 terminal bytes. The 32 submissions and 32 completion waits are
unchanged by terminal reading. The input cyclically replays the saved 768-frame
codes: this proves the decoder allocation path, not 2048 model-generation steps
or long-form speech quality.

The fresh diagnostic capture against the pinned MLX DAC reproduces the prior
58-stage metrics and gates exactly. The existing first failure remains
`quantizer.post_module.layers.0`; full-stage and one-PCM-LSB gates still fail
(maximum error 16 PCM units). This is unchanged numeric behavior, not full DAC
parity. Trace capture deliberately bypasses scratch; bitwise fresh/arena
equivalence is established separately by the paired saved-code runs above.

## Native generation with expanded allocation

The same 158-row news prompt, seed 1 and sampling settings were run with a
1100-frame bound, resident capacity 1257, graph depth 2 and DAC scratch enabled.
Generation ended on EOS after 773 frames (35.898 seconds of output), with
202.876 seconds in generation and 81.038 seconds in the DAC. All 768 earlier
native frames match exactly across all 11 IDs in the GPU comparison. The first
upstream mismatch remains frame 704, using the same saved MLX control; this is
not a new upstream generation or full-trajectory parity result.

The actual native sampler producer emitted 12,100 F32 uniforms. Its full digest
matches the generation upload, and the first 8,448 values equal the prior native
and MLX input tape exactly. Seed equality alone was not used as control. The
small external producer probe initially failed to compile because `Slice#first`
returned an array; bounded slice indexing fixed the probe without changing
production sampling.

EOS occurred before the old 1024-position execution boundary. This run exercises
the expanded cache allocation and wide-attention path but does not establish
1100 generation steps or actual model attention at positions above 1024. The
synthetic GPU attention sweep separately executes lengths 1025/2048/4096 at
depths 0/2. The full native run recorded 1,419,434 `Context#alloc` calls; most
temporary-buffer ownership remains outside the decoder arena. This count is
not a measure of allocator time or physical peak memory.

The next performance discriminator is an interleaved fresh/arena comparison
with matched warmups, separate host encoding time and per-command GPU timing.
It should distinguish GPU placement/locality from host bookkeeping and shared
host noise before promoting reuse as a speed optimization. Extending explicit
workspace ownership to backbone/tower is a separate implementation slice.

Raw commands, source/binary hashes, logs and results are under
`/Users/sergey/.cache/cogni-ml/drama3-runs/window-scratch-20261015/`.
