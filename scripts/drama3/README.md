# DRAMA-3 reference tooling

These scripts compare the native implementation with pinned MLX-audio and
Fish-Speech code. The checked-in fixture is deterministic, synthetic, and does
not contain released checkpoint weights. See [THIRD_PARTY_DRAMA3.md](../../THIRD_PARTY_DRAMA3.md)
for source and model attribution.

For native reference-voice inference, the 102-stage encoder/prompt and prefill
gates, and paired pause controls with shared sampler inputs, see
[Native DRAMA-3 reference voice](../../docs/drama3-reference-voice.md).
`reference_voice_reference.py` exports official encoder/prompt fixtures,
`prefill_reference.py` exports captured prefill boundaries, and
`compare_reference_pauses.py` validates the four completed generation reports
and measures tokens/quiet windows on GPU. These are reference tools; native
inference uses the resident Crystal/Metal encoder directly.

Use Python 3.12 with `mlx`, `torch`, `numpy`, and `safetensors`, plus Crystal.
The pinned source checkouts are imported directly. Check them out at the
revisions below; replay scripts reject a different source `HEAD`.

```sh
git clone https://github.com/cguldogan/mlx-audio "$MLX_ROOT"
git -C "$MLX_ROOT" checkout --detach e7488935ec2ebfafd38ef2b58e290d79f3d05dc7
git clone https://github.com/fishaudio/fish-speech "$FISH_ROOT"
git -C "$FISH_ROOT" checkout --detach 541566932abbaebcb0a40e562a1f846738383f38

OUT=/tmp/drama3-reference
python scripts/drama3/export_reference.py \
  --mlx-root "$MLX_ROOT" --fish-root "$FISH_ROOT" --output "$OUT/fixture"
crystal run --release -Dcpu_only examples/drama3_trace.cr -- \
  "$OUT/fixture" "$OUT/fixture/inputs.json" "$OUT/native-backbone.json"
crystal run --release -Dcpu_only spec/support/drama3/native_tower_trace.cr -- \
  "$OUT/fixture" "$OUT/fixture/inputs.json" "$OUT/native-tower.json"
python scripts/drama3/compare_traces.py \
  --reference "$OUT/fixture/reference_traces.json" \
  --actual "$OUT/native-backbone.json" --prefix backbone
python scripts/drama3/compare_traces.py \
  --reference "$OUT/fixture/reference_traces.json" \
  --actual "$OUT/native-tower.json" --prefix tower
```

For an already-cached model snapshot at
`cguldogan/drama-3-mlx-q4@5331ab509359c0e4a7e353a19e255ddf3e8152a8`, set
`MODEL_DIR` to its local directory and `INPUTS` to a matching native trace
input bundle. These commands read only the indexed backbone or tower tensors;
the backbone replay instantiates the text backbone, while the tower replay
loads only tower tensors. They do not load unrelated checkpoint components or
hash multi-gigabyte shards unless `--hash-shards` is explicitly requested.

```sh
crystal run --release -Dcpu_only examples/drama3_trace.cr -- \
  "$MODEL_DIR" "$INPUTS" "$OUT/native-backbone-q4.json"
python scripts/drama3/replay_mlx_backbone.py \
  --mlx-root "$MLX_ROOT" --model-dir "$MODEL_DIR" \
  --inputs "$INPUTS" --output "$OUT/mlx-backbone-q4.json"
python scripts/drama3/compare_traces.py \
  --reference "$OUT/mlx-backbone-q4.json" \
  --actual "$OUT/native-backbone-q4.json" --prefix backbone --atol 6e-5

crystal run --release -Dcpu_only spec/support/drama3/native_tower_trace.cr -- \
  "$MODEL_DIR" "$INPUTS" "$OUT/native-tower-q4.json"
python scripts/drama3/replay_torch_tower.py \
  --fish-root "$FISH_ROOT" --model-dir "$MODEL_DIR" \
  --inputs "$INPUTS" --output "$OUT/torch-tower-q4.json"
python scripts/drama3/compare_traces.py \
  --reference "$OUT/torch-tower-q4.json" \
  --actual "$OUT/native-tower-q4.json" --prefix tower \
  --atol 6e-4 --rtol 3e-4
```

The real Q4 tower replay has 55 intermediate values outside the fixture
default tolerance at `3e-5 + 3e-4 * abs(reference)`; the scoped real-checkpoint
comparison above uses `6e-4 + 3e-4 * abs(reference)`. This does not assert
terminal-output parity. See [the frontier record](../../docs/drama3-native-frontier.md)
for the measured failures and stricter terminal checks. On macOS, add the local
linker flag only if the installed SDK's `lld` setup requires it.

`replay_mlx_embedding.py` independently checks one packed embedding row using
the upstream `QuantizedEmbedding` call. Pass the matching native inspection
file and token ID with `--inspect` and `--token-id`; it reads only that row's
packed weights, scales, and biases.

Trace records declare `float32`. The comparator rounds JSON numbers to IEEE
binary32 before comparison, so equivalent shortest and full-precision decimal
spellings compare identically. Actual-checkpoint backbone replay promotes
floating checkpoint tensors to float32 to match the native CPU path; results do
not claim stock MLX BF16 or bit-identical execution. Fixture comparisons use
the comparator defaults (`atol=3e-5`, `rtol=3e-4`); real-checkpoint thresholds
must stay tied to the recorded source, input bundle, and precision path.

For the native codec's intermediate-stage probes, run the focused codec spec
with `DRAMA3_CODEC_REFERENCE=synthetic` or `real`,
`DRAMA3_CODEC_PYTHON` pointing to the reference Python executable,
`DRAMA3_MLX_AUDIO_ROOT="$MLX_ROOT"`, and (for real mode)
`DRAMA3_CODEC_WEIGHTS="$MODEL_DIR/codec.safetensors"`.

To compare a complete native utterance against the MLX DAC without rerunning
native inference, feed the generated sidecar back into the reference helper:

```sh
PYTHONPATH="$MLX_ROOT" python spec/support/drama3/codec_reference.py real \
  --codec "$MODEL_DIR/codec.safetensors" --codes-json /tmp/hello-native.wav.json \
  --wav-output "$OUT/hello-mlx-codec.wav" --compare-wav /tmp/hello-native.wav \
  --output "$OUT/hello-mlx-codec.json"
```

This reports PCM count errors and waveform-unit errors using the fixed
PCM16 scale, with no peak or RMS renormalization. A code replay validates the
decoder; it does not establish identical upstream stochastic generation or
subjective speech quality.

## Full-stream GPU DAC cross comparison

`codec_cross_reference.py` decodes a complete saved stream through the pinned
upstream `DAC.decode` API. It accepts native `[main,c0,...,c9]` frame reports,
MLX `output.frames`/book-major codes, and an explicit `--chunk-index` for a
chunked report. The main token is excluded; exactly ten valid, equally long
DAC codebooks are required. The codec checksum and clean upstream revision
are checked before model import. F32 checkpoint weights are not converted.

```sh
python scripts/drama3/codec_cross_reference.py \
  --codec "$MODEL_DIR/codec.safetensors" --mlx-root "$MLX_ROOT" \
  --codes-json /tmp/utterance.wav.json --output-dir "$OUT/codec-reference"

crystal build --release -Ddrama3_metal examples/drama3_codec_cross_compare.cr \
  --link-flags="$(pwd)/build/bridge.o -framework Metal -framework Foundation -framework MetalPerformanceShaders -lc++ -fuse-ld=/usr/bin/ld" \
  -o "$OUT/codec-cross"
"$OUT/codec-cross" --self-test
"$OUT/codec-cross" "$MODEL_DIR" "$OUT/codec-reference/reference.json" \
  "$OUT/codec-comparison.json" "$OUT/codec-native.wav"
PYTHONDONTWRITEBYTECODE=1 python -m unittest spec/drama3_codec_cross_reference_spec.py
```

Use the project's guarded runner for long model jobs. Reference output
directories and native output files must be new. The native probe supports
1–1024 complete frames at CogniGraph depth 2. It requires all 58 named
boundaries: each codebook embedding/projection, their sum, eight post-transformer
layers, post-module output, upsampling, decoder blocks, and waveform. Layouts
are checked at every boundary; reference F32 files are hashed before use.

The MLX observer retains actual upstream outputs and must reproduce an
unobserved official decode exactly. Export happens after full GPU evaluation.
The native device observer enqueues GPU comparisons without intermediate host
reads or extra submissions. It reads reduced diagnostics and the completed
waveform after the existing terminal drain; graph counters must stay unchanged
across those reads. Timings include diagnostics and are not performance
benchmarks. The probe does not select new model arithmetic.

### Original Fish DAC control

`fish_codec_reference.py` adds the original Fish `DAC.from_indices` control
using `fishaudio/drama-3@e0a9218aa48888270f06ef1ed9ea67622c0e17d2`'s
`codec.pth` (SHA-256
`74fc41c5a7151c6f350af8bd7e5d6e3accfcc7f3dfbfac23afd35af07052bb2f`).
It requires the clean Fish source revision above, PyTorch/torchaudio 2.8.0,
`descript-audio-codec==1.0.0`, `descript-audiotools==0.7.2`, Hydra and
OmegaConf. Use a separate Python environment; the measured MLX fixtures use
MLX/MLX-Metal 0.31.1. Runtime package/source hashes accompany the evidence.

The loader follows the original direct DAC CLI's F32 checkpoint path. It
preserves parametrized weight norm, all 535 F32 parameters, and the three
constructed BF16 RoPE buffers; it does not cast the model. This is a decoder
control, not a test of Fish's complete text-to-speech serving path, whose
loader can fold weight norm and cast the model to BF16. Independent state
keys/shapes must match exactly; original nonpersistent caches/masks are
excluded from the state load as in the official model.

```sh
PYTORCH_ENABLE_MPS_FALLBACK=0 PYTORCH_MPS_FAST_MATH=0 \
python scripts/drama3/fish_codec_reference.py \
  --fish-root "$FISH_ROOT" --codec "$FISH_CODEC" \
  --codes-json /tmp/utterance.wav.json --output-dir "$OUT/fish-reference"

# Native Metal versus original Fish, on exactly the same complete codes.
DRAMA3_COGNIGRAPH_MAX_INFLIGHT=2 "$OUT/codec-cross" "$MODEL_DIR" \
  "$OUT/fish-reference/reference.json" \
  "$OUT/native-fish-comparison.json" "$OUT/native-fish.wav"

# Completed original Fish decode versus completed MLX decode, on GPU.
"$OUT/codec-cross" --compare-fixtures \
  "$OUT/fish-reference/reference.json" "$OUT/codec-reference/reference.json" \
  "$OUT/fish-mlx-comparison.json"

PYTHONDONTWRITEBYTECODE=1 python -m unittest \
  spec/drama3_codec_cross_reference_spec.py spec/drama3_fish_codec_reference_spec.py
```

Both original Fish decodes (unobserved and observed) run on MPS under a
dispatch guard that rejects host tensors and scalar extraction. The observer
captures actual module results, including per-instance codebook embeddings,
and must match the unobserved full decode elementwise before export. CPU
fallback is disabled. Factory-created causal masks are allocated on the
selected GPU. Weights/codes are uploaded before model execution; completed
fixtures and final diagnostics/WAVs are serialized after execution.

`--compare-fixtures ACTUAL.json REFERENCE.json OUTPUT.json` is generic and
also supports identity controls. The four retained cross-backend runs use
Fish as actual and MLX as reference; reports record both formats, source
revisions, checkpoint hashes and manifest hashes. It rejects different full
code arrays, wrong pins, missing stages, shape/layout errors and stale raw
file hashes. Each fixture uploads once before GPU reductions. No model is
rerun in this mode, and terminal metric reads must add no graph work or waits.
All gates below are shared with the native comparison and remain unchanged.

The stage limits remain max absolute `1e-3`, RMS `2e-4`, normalized RMS `2e-4`;
the waveform limit remains `max(5e-5, abs(reference)*1e-3)` per sample. The
one-PCM16-LSB limit is a separate finite-waveform gate. `full_gate_passes`
requires both numerical gates and the existing nonzero-waveform sanity check.
A valid measurement exits zero even when parity gates fail: consume the
explicit gate fields, not process success, as the parity result. The GPU
reducer self-test covers equality, deliberately corrupted values across a
reduction boundary, nonfinite input, zero reference, large indices and window
statistics against a small independent F64 oracle.

Both WAVs use the native writer's F32 scale and ties-to-even PCM rounding.
Older `generate_mlx_control.py` artifacts use half-away rounding and can differ
by one PCM unit at ties. Half-second GPU RMS/peak diagnostics describe signal
activity; they do not certify words, voice identity or speech delivery.
