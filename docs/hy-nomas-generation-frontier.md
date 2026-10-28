# Hy-Nomas first native video frontier

## Active request and admitted scope

The user authorized a real five-second preview before further kernel tuning.
The first bounded target is one first segment, 120 frames at 24 fps,
512x288, seed 42, CFG 1, eight shifted Euler steps (shift 5), camera/action
conditioning off. Its DiT grid is `[30,9,16]` (4320 tokens). PixelUMM uses the
pinned preset's ten-latent windows, five-latent overlap, no spatial cropping,
and the existing eight-token local attention layout on grid `[10,18,32]`.

BF16 remains the default DiT control. The user also authorized an experimental
Q4_K DiT preview with the identical pinned inputs and schedule for visual
comparison. PixelUMM and the global DiT adapters remain BF16. Q4 is decoded on
GPU into the existing single-block BF16 stage; it does not introduce another
wait or a CPU activation handoff. This experiment does not promote Q4 quality
or change the numerical tolerance. Same-decoded-Q4 numerical parity does not
establish acceptable image quality: the recorded Q4 versus
BF16 full-forward discrepancy is large. The first prompt may reuse the
authenticated saved Qwen state-26 crop; that boundary must be stated in the
run receipt rather than presented as live native text encoding.

## Existing anchors and missing composition

- The 54-block DiT, input/final adapters and text/time refiner are qualified
  separately on the saved first-segment fixture.
- The 36-block PixelUMM decoder plus head are qualified separately on real
  stem activations. That fixture is not a generated video.
- The missing production edges are GPU Euler integration, ordered repeated
  forwards with a changing GPU timestep, PixelUMM input stem, window decode
  and cosine merge, display-range conversion, and terminal MP4 export.
- Pinned upstream: `911cea6458a89c2663513ff7099561ec190dd491`.
  Existing source and weight authentication remain mandatory.

## Risk, resource boundary and rollback

CAUTION: this composes previously separate model phases and introduces repeated
use of fixed scratch. The strongest failure is stale/overwritten data from an
incorrect pass or window order, followed by plausible but invalid pixels.
Reject wrong ordering before appending work, retain all GPU owners until their
chain drains, and check finite output and nonconstant frames. Keep the old
single-forward loader contract as the default; repeated forwards are bounded
and explicit. New generation entry points stay opt-in.

Keep a cache-aware 10% physical-memory reserve, a fixed process cap, and a
64 MiB maximum system swap increase. Run heavy GPU tasks serially. Admit each
weight phase before allocation. DiT weights and decoder weights cannot simply
be added to the same residency budget. If a phase handoff requires draining a
chain to release weights, report that synchronization explicitly; retain the
small latent GPU buffer across the handoff and do not round-trip it through
the CPU. There are no waits or activation readbacks inside diffusion steps,
decoder blocks or window assembly.

Rollback is removal of the opt-in entry point and new glue, with the default
single-forward probes and current MPS baseline unchanged. Preserve unrelated
working-tree files. All temporary outputs and build products live on
`/opt/mldata_ssd/`.

## Definition of done and falsifiers

1. Run the scheduler/order specs and the new stem/merge oracle comparison at
   the unchanged `atol=rtol=2e-5`; nonfinite negative controls must fail.
2. Run a guarded native first-segment request through eight DiT forwards,
   all 36 decoder blocks and head for each of five windows, followed by GPU
   merge/postprocess. Record named allocations, encode readbacks, terminal
   waits, elapsed times, peak footprint, and guard verdict.
3. Export only completed display-range F32 pixels using
   `examples/hy_nomas_export.py`; verify 120 frames, 512x288, 24 fps, five
   seconds, finite and nonconstant decoded frames. Inspect a contact sheet and
   the actual clip. A valid MP4 alone does not certify generation quality.
4. Compare the newly composed numerical boundaries with pinned upstream on
   identical inputs. Keep execution parity, image quality and performance as
   separate claims. A failing boundary blocks a verified parity claim; never
   expand the tolerance to pass it.

The bounded native execution and terminal export below are VERIFIED within
their stated scope. Full-video upstream parity and general prompt support
remain open. No native generated clip was available at the start of this work.

## Boundary checks, 2026-10-28

The fixed input manifest is
`5b162b8bf4b28c6cfacd8a3d8c14a7758b699034b2a46e84c224522eb0a75a7b`.
It carries the saved natural-tram conditioning, MPS-generated seed-42 noise,
and pinned upstream RoPE tables for the preview geometry. This is a fixed
pilot, not a general prompt CLI. Initial F32 noise is rounded through BF16 on
GPU before the first DiT forward. The native DiT retains its qualified F32
arithmetic; the Euler boundary rounds model velocity and each updated latent
through BF16 as the source pipeline does.

The pinned `EulerSolver` module produces a 256-element MPS oracle under
Diffusers 0.36.0. Native initial rounding, all eight timestep values and the
final latent are bit-exact. The probe has one packed terminal diagnostic read
and one wait, with no preterminal activation read or wait. Its fixture SHA is
`3fce6402ec76036437dc0bdfee9f07008610bdca23c040c2bab55d339777d6de`.

The real-weight PixelUMM stem and synthetic five-window temporal assembly
also match the pinned MPS source bit-exactly, with a NaN rejection control.
The probe records zero encode allocations/readbacks/preterminal waits and one
terminal wait. The corrected oracle manifest SHA is
`3f6976cbd8d1efeeb3aa663b817980cba78b8d793e7ed56e90384278a535409f`.
An earlier oracle was invalid: large MPS int64 remainder operations corrupted
the synthetic inputs. Constructing the exact 8192-value period before repeating
it removes that oracle error. The source display conversion also requires
separate BF16 rounding after multiply and add, before F32 export. Neither fix
widens the tolerance. The earlier failing artifacts remain outside the repo.

The immutable DiT weight cache records its V5280 qualification geometry.
The preview authenticates that original cache geometry separately from its
V4320 activations. A CPU check permits only this video-token extent to differ
and requires identical parameter bindings; cache validation itself is unchanged.
Runtime buffers, plans, RoPE and memory accounting use V4320 throughout.

Exact named residency is 21,011,384,964 bytes for DiT and 16,563,852,432 bytes
for the decoder phase. The external guard uses a 23 GiB process-tree cap,
10% cache-aware reserve, 64 MiB swap-growth stop and 900 s timeout. These are
admission and sampled runtime guards, not guarantees against between-poll spikes.

## First completed preview, 2026-10-28

The accepted `generation-v3` process completed on Apple M2 Max, 64 GiB unified
memory. It generated the fixed prompt about a yellow tram in a wet downtown
street. `preview.mp4` contains 120 frames, 512x288, 24 fps and exactly five
seconds, independently checked with ffprobe and decoded to a contact sheet.
The sampled frames contain a recognizable tram, buildings and umbrellas;
the tram position changes over time. This inspection is not an upstream
quality comparison or a certification of temporal consistency.

| Phase | Measured wall time |
| --- | ---: |
| DiT preparation | 22.877 s |
| Eight DiT steps | 35.768 s |
| PixelUMM preparation | 22.221 s |
| Five decoder windows and display conversion | 67.547 s |
| Guarded process, including CPU admission and terminal file output | 155.672 s |
| MP4 export | 0.417 s |

This is about 31.2 seconds per second of video including startup and export,
or 20.7 seconds per second for the two neural execution phases. These are one
preview's measurements at 512x288, not a warmed benchmark, a result for 480p
or 720p, or a comparison with Wan or another backend.

The guard recorded 300 samples: peak process-group footprint 21,340,353,160
bytes (19.874 GiB), minimum cache-aware headroom 23,171,694,592 bytes
(21.580 GiB), swap growth zero, exit zero. Both neural phases report one
terminal wait and zero preterminal waits. Allocation/readback counters report
zero only for instrumented MetalBuffer methods; they do not measure opaque
MPS resources or the direct terminal pixel-file export. The receipt now states
that scope explicitly. The one weight-phase handoff is an intentional wait;
there is no CPU round trip for the latent.

The completed pixels have SHA256
`f13481b1d5790761df94943b6eac9e768177a240998c859d3ea9f21a633c2f85`;
the exported MP4 has SHA256
`7a323eb0b9e02443eedf45a5576f36a3c738ea41b687d1a476cbbdcfd0ea1cd3`.
All 120 display-range frames are finite and nonconstant. Frame standard
deviation ranges from 0.2316 to 0.2433; consecutive-frame mean absolute
differences range from 0.02968 to 0.06836. These statistics reject empty or
frozen output; they do not establish acceptable motion quality.

Artifacts live outside Git under
`/opt/mldata_ssd/cogni-ml/hy-nomas-first-video-20261028/`: `inputs.json`,
`generation-v3.log`, `generation-v3.guard.jsonl`, `results/receipt.json`,
`export.json`, `ffprobe.json`, `video-analysis.json`, `contact-sheet.jpg`, and
`preview.mp4`. The original receipt precedes the metric-scope wording addition;
its counter interpretation is the scoped one above. Earlier failed attempts
are retained separately. One stopped before loading weights on the recorded
cache-geometry mismatch; another completed GPU work but rejected Crystal's
unsupported `wx` file mode. The writer now uses O_EXCL and a CPU no-clobber
regression check. Neither failure was a successful video result.

Final checks: release builds, 39 Crystal examples, 13 Python tests, formatting
and diff checks; bit-exact upstream checks for the newly composed Euler and
stem/merge boundaries as described above. Correlated adversary review found
no enqueue-order/lifetime blocker and required the metric-scope correction.
This does not add a full generated-video upstream comparison.

## Experimental Q4 preview, 2026-10-28

The opt-in `--dit-weights q4-k` selector uses the authenticated existing
`HyNomasQ4Stack`, including its eight-forward order/chain limit. The default
remains `bf16`. Only DiT block weights differ: all pinned input bytes, global
adapters, Euler schedule, decoder noise, BF16 PixelUMM weights, and export
settings match the accepted BF16 preview. This is the staged Q4_K path, not
the separate direct-Q4 GEMM experiment. Compact block payloads occupy
4,593,162,240 bytes. Their GPU decode reuses one BF16 block stage; the complete
Q4 DiT phase admits 9,290,402,436 bytes instead of 21,011,384,964 for BF16.
The decoder admission remains 16,563,852,432 bytes and dominates the Q4 peak.

DoD: release build, focused selection/admission/order specs, CPU admission in
both modes, invalid-mode rejection before Metal, and one guarded complete
preview with the identical input manifest. Memory estimates must match actual
owner admission. A mode mismatch, input/schedule mismatch, added encode-time
readback/wait, guard stop or nonfinite output rejects the execution result.
Q4-versus-BF16 quality remains a visual experiment; it is not a passing 2e-5
parity test. Existing same-decoded-weight kernel gates retain that tolerance.

The Q4 run completed all 120 frames at 512x288/24 fps. Contact frames show a
recognizable tram but substantially more blurred and distorted buildings and
tram contours than the BF16 control. Q4 quality is not promoted. The raw output
is finite and nonconstant in all 120 frames, and ffprobe independently confirms
the frame count, geometry, frame rate and five-second duration. These checks
establish execution and export, not acceptable perceptual quality.

| Phase | Prior BF16 launch | Q4 launch |
| --- | ---: | ---: |
| DiT preparation | 22.877 s | 7.296 s |
| Eight DiT steps | 35.768 s | 48.976 s |
| PixelUMM preparation | 22.221 s | 22.313 s |
| Five decoder windows/display | 67.547 s | 85.906 s |
| Guarded process | 155.672 s | 168.534 s |
| MP4 export | 0.417 s | 0.311 s |
| Sampled peak group footprint | 19.874 GiB | 16.784 GiB |

These are separate single launches, not an interleaved speed experiment.
The unchanged decoder also took longer in the Q4 launch. Do not attribute the
whole timing difference to quantization or claim an acceleration. Faster DiT
preparation and lower resident bytes do not remove PixelUMM compute cost.
All weight files currently resolve through `/opt/mldata_ssd` to the external
SSD. Disk reads occur during setup; the block payloads are GPU resident during
encode. A local mirror could improve startup but has not been benchmarked.

The Q4 guard recorded 324 samples, peak footprint 18,021,842,016 bytes, minimum
cache-aware headroom 23,878,877,184 bytes, zero swap growth and exit zero under
the unchanged 10% reserve, 23 GiB process cap, 64 MiB swap stop and 900 s deadline.
There is one terminal wait per phase, zero preterminal waits/readbacks and zero
instrumented MetalBuffer encode allocations. Opaque MPS resources and terminal
pixel-file export remain outside the counter scope. The Q4 receipt explicitly
records the mode, experimental quantization, BF16 global/decoder weights and
Euler boundary rounding; full generated-video upstream parity remains untested.

Artifacts: `/opt/mldata_ssd/cogni-ml/hy-nomas-q4-preview-20261028/` contains
`launch.json` (source/binary/baseline digests), `plan-bf16.jsonl`,
`plan-q4.jsonl`, `specs.log`, `invalid-mode.log`, `generation.log`,
`generation.guard.jsonl`, `results/receipt.json`, `comparison.json`,
`export.json`, `ffprobe.json`, `contact-sheet-comparison.jpg`,
`preview-q4.mp4`, and `comparison-bf16-left-q4-right.mp4`.
Q4 pixels SHA256:
`dc7a80be9c802b134e91ae784f7be0c05a5bae398cdd1fdd7e8cfe2d6cff94ae`.
Q4 MP4 SHA256:
`fb4cfacc8f7a8abd2be264e9a2f083dad2bbe6c6837afa572b62a505f3434074`.
The focused suite passed 21 examples. Correlated adversary review found no
composition/lifetime blocker; it required explicit Euler-rounding wording.
The final artifact review returned ROBUST for matching input/schedule/decoder
identities, guarded execution and finite output; it does not qualify visual
quality, full upstream parity or a speed improvement.

## Duration boundary

The native configuration explicitly admits one first segment and rejects
history/multiple segments. The current input, latent, RoPE and assembly are
fixed to 120 output frames. Five overlapping decoder windows are not five
autoregressive segments. Upstream implements segment continuation with K/V
history and temporal RoPE offsets; that state contract is not ported here.

Pure division of the first BF16 launch gives about 1 minute 55 seconds of
independent five-second output per wall-clock hour at 512x288. Dividing only
the neural phases gives about 2 minutes 54 seconds per hour, excluding setup
and requiring a future reuse design. Neither number is sustained throughput
or a maximum continuous duration. In particular, two continuous minutes
cannot yet be requested from this entry point. Long-duration timing and memory
must be measured after continuation is implemented; history adds work and
memory absent from the present preview.

## Next optimization frontier

PixelUMM accounts for 65.4% of the two neural phases. Weight authentication,
loading and plan construction account for another 45.1 seconds of the native
request and should be separated before proposing a persistent owner. Do not
remove integrity checks to make startup appear faster.

The next exact-path probe should group independent same-sized local-attention
tiles into batched MPS graphs while retaining each tile's context, BF16
boundaries and reduction order. First measure the current 5760-token window
by FFN, projections, attention and command transitions; a reduction in API
calls alone does not establish a full-window speedup. Compare identical inputs
against the current implementation and pinned source at the unchanged 2e-5,
then measure the complete decoder window. This is PROPOSED, not implemented.
The overlapping windows decode 200 frame positions to produce 120 output
frames, but later-layer overlap activations depend on different attention
contexts and cannot be reused solely because their frame indices match.

Refresh all execution claims after changes to source, weights, input identity,
geometry, precision, schedule, command ordering, backend/device or guard.
Each launch requires fresh memory admission; missing external artifacts reopen
the corresponding numerical or performance claim.
