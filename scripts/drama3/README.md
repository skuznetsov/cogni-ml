# DRAMA-3 reference tooling

These scripts compare the native implementation with pinned MLX-audio and
Fish-Speech code. The checked-in fixture is deterministic, synthetic, and does
not contain released checkpoint weights. See [THIRD_PARTY_DRAMA3.md](../../THIRD_PARTY_DRAMA3.md)
for source and model attribution.

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
