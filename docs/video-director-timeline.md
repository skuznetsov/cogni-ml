# Offline Hy-Nomas director timeline

`examples/hy_nomas_timeline.py` compiles an editable JSON project into a
reviewable prompt list and explicit Hy-Nomas overrides. It is a **compile-only
tool**: it does not start inference, validate model quality, or claim that the
native Hy-Nomas path supports complete video generation.

## Project and compiler

The project schema is `hy_nomas_director_project_v1`. It records the pinned
24 fps preset, dimensions, requested duration, and ordered prompt changes. A
prompt holds until the next change. The first prompt must begin at zero;
timestamps must be strictly ordered and inside the requested duration. The
compiler currently requires dimensions that are already multiples of 16, so
the pinned pipeline's `align_to(..., 16)` does not silently change geometry.

Pinned upstream parses event timestamps with `int(seconds * fps)` and uses
`video_length: event` to derive the video length from the last event boundary.
The causal pipeline computes latent length as
`(num_frames - 1) // 4 + 1`, and maps non-final pixel segment boundaries to
latent boundaries with `pixel_frame // 4`, keeping them strictly increasing.
The compiler records the upstream timestamp-to-frame truncation as the
requested frame boundary, then rounds the exact requested time **up** to the
next multiple of four pixel frames. It reports requested and effective
times/frames separately. The requested frame is computed with exact decimal
arithmetic; it does not reproduce binary-float parser rounding for timestamps
extremely close to a frame boundary. Explicit integer boundaries govern
execution. The compiler rejects changes that collapse to one boundary or
leave a zero-length interval, and
sets the final `num_frames` to the aligned end boundary. No duration or segment
is silently rescaled. For example, the sample's requested 1.4-second change
maps to frame 33 under upstream truncation and becomes effective at frame 36
(1.5 seconds); a three-second duration becomes 72 pixel frames and 18 latent
frames.

The emitted `upstream_input` contains one item whose `segments` carry a
`prompt` field. Use the emitted overrides together: set both description
fields to `prompt` for multi-segment projects, set `desc_field_rest` to the
empty string for the single-segment case, retain `rollout_type=seg_ar` and
`video_length=event`, and pass the emitted `num_frames`, `video_fps`,
`video_size=[height,width]`, `max_segments`, and `segment_frame_boundaries`.
The top-level width and height are descriptive; `video_size` is what upstream
passes to the pipeline. The input JSON omits timestamps on purpose; explicit
boundaries avoid a second timestamp conversion and avoid upstream's
`video_length=frame` boundary rescaling. The compiled document wraps the loader
array under `upstream_input`, so it cannot itself be passed as the loader's
top-level JSON file. Extract that array to a new path first (exclusive creation
prevents replacing an existing input):

```python
import json
from pathlib import Path

root = Path("/opt/mldata_ssd/cogni-ml/director")
compiled = json.loads((root / "kite-to-sea.compiled.json").read_text())
with (root / "kite-to-sea.upstream-input.json").open("x") as stream:
    json.dump(compiled["upstream_input"], stream, indent=2)
    stream.write("\n")
```

The emitted prompts and overrides are prepared input, not evidence that
inference ran. A CPU probe executed the original pinned `_load_from_json`
function and its pure helpers, extracted from the source AST, without importing
Torch or the inference module. Both the two-prompt sample and a single-prompt
640x1024 project preserved the exact prompt strings, prompt count and absence
of JSON-derived boundaries. The probe also checked the nondefault geometry
override. Hydra composition, pipeline execution and pixel-to-latent boundary
conversion remain source-traced only.

Compile the included project to a **new** output path; existing files are
never replaced:

```sh
mkdir -p /opt/mldata_ssd/cogni-ml/director
python3 examples/hy_nomas_timeline.py \
  --project examples/data/hy_nomas_director_project.json \
  --output /opt/mldata_ssd/cogni-ml/director/kite-to-sea.compiled.json
```

Review `segments`, `requested_frame_boundaries`, `segment_frame_boundaries`,
and `segment_latent_boundaries` before passing the overrides to upstream. The
compiler rejects zero/negative duration, non-24 fps, non-16-aligned geometry,
unsorted or out-of-range timestamps, prompt changes that collapse after frame
alignment, prompts containing the upstream-reserved `|||` segment separator,
and outputs that already exist. Upstream joins prompt strings with `|||` and
splits on the same literal token; no escaping contract is provided for prompts
that contain it.

This timeline has no arbitrary subframe updates, resumable execution, camera
trajectory track, or image/video conditioning. Pinned `seg_ar` updates prompts
at segment boundaries and keeps the inference request's autoregressive KV
history across those segments; a completed request clears its cache by default.
Those are upstream behaviors, not timeline execution or persistence features
provided here.

## Terminal export

`examples/hy_nomas_export.py` encodes an already-completed terminal file only.
Its input contract is one batch of three-channel little-endian F32 pixels in
`[B=1,C=3,T,H,W]` order, stored as the corresponding contiguous
`[C,T,H,W]` bytes. It checks the exact byte count, rejects non-finite values,
requires 24 fps and 16-aligned dimensions, and checks the supplied latent frame
count against the pinned formula before starting FFmpeg. It does not load a
model or establish the provenance or correctness of the file.

Hy-Nomas display output uses the upstream conversion
`(frame * 255).clamp(0, 255).to(uint8)`, so its exporter explicitly selects
`zero_to_one`. The input must already have undergone upstream display-range
postprocessing; raw decoder pixels are not this file contract. A future native
pipeline must apply that postprocessing before its terminal output.
`examples/wan_s2v_export.py` retains its default Wan
`minus_one_to_one` normalization. Do not use Wan's range normalization for
Hy-Nomas output.

```sh
python3 examples/hy_nomas_export.py \
  --input /path/to/completed-display.f32 \
  --output /path/to/kite-to-sea.mp4 \
  --frames 72 --latent-frames 18 --height 720 --width 1280 --fps 24
```

The exporter writes a temporary MP4 beside the destination and publishes it
without replacing an existing path. FFmpeg encoding is terminal file handling,
not model execution.

## Bounded verification

Run the two CPU suites with temporary files on the external SSD:

```sh
mkdir -p /opt/mldata_ssd/cogni-ml/director/tmp
TMPDIR=/opt/mldata_ssd/cogni-ml/director/tmp \
  python3 -B spec/hy_nomas_timeline_test.py
TMPDIR=/opt/mldata_ssd/cogni-ml/director/tmp \
  python3 -B spec/hy_nomas_export_test.py
```

The tests cover exact/aligned timestamps, geometry overrides, single and
multiple prompts, rejected collisions and reserved delimiters, exclusive
publication, explicit Hy-Nomas/Wan pixel ranges, malformed terminal files and
a four-frame 16x16 FFmpeg round trip. They do not validate generated scenes,
prompt adherence or a connected native pipeline.

The 2026-10-27 run passes all 17 tests. Its compact evidence index is
`.landmark/evidence/2026-10-27-hy-nomas-offline-director.json`; source hashes,
test logs, the compiled sample and the AST-extracted upstream loader probe are
under `/opt/mldata_ssd/cogni-ml/hy-nomas-pixelumm-ring-20261027`.

## Pinned upstream references

The reference checkout used for this contract is
`/opt/mldata_ssd/cogni-ml/video-scene-20261019/upstream/hy-nomas-20261024`,
at Git revision `911cea6458a89c2663513ff7099561ec190dd491`.
Relevant source locations:

- `configs/experiment/inference.yaml:79-84`: `seg_ar`, 24 fps, and event
  length defaults.
- `hymm/inference.py:414-430`: supported timestamp formats;
  `int(start_sec * fps)` and `int(end_sec * fps)` occur at
  `hymm/inference.py:1253-1256`.
- `hymm/inference.py:2040-2092`: explicit CLI boundaries take precedence;
  event length comes from JSON timestamps only when CLI boundaries are absent.
  The compiler sets `num_frames` as well as boundaries because its JSON omits
  timestamps.
- `hymm/inference.py:1381,1989,2204-2206,2227-2228`: prompt segments are joined
  and split on `|||`; inference consumes `inference.video_size` as
  `[height,width]`. The compiler rejects prompts containing that delimiter and
  emits the geometry override.
- `hymm/pipeline/causal_ar.py:1613`: latent frame count formula;
  `hymm/pipeline/causal_ar.py:1728-1746`: pixel-boundary to latent-boundary
  conversion and strict monotonicity.
- `hymm/utils/video_output.py:6-20`: `[C,T,H,W]` display-range output
  conversion to uint8.

These references pin the compiler's mapping assumptions. A changed upstream
revision, FPS preset, temporal VAE ratio, timestamp parser, boundary conversion,
or output range invalidates this contract and requires rechecking the source.
