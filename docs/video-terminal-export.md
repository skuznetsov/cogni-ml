# Terminal Wan video export

After the native GPU request completes, `examples/wan_s2v_export.py` maps its
raw little-endian F32 output as `[1,3,T,H,W]`, checks finiteness, and converts
one frame at a time to RGB24. It follows Wan's clamp, normalization and uint8
truncation order. FFmpeg writes H.264 MP4 and can mux the input audio as AAC.
This is terminal host file encoding; it performs no neural inference and adds
no intermediate GPU wait or activation readback.

```sh
python examples/wan_s2v_export.py --input /absolute/path/video.f32 \
  --output /absolute/path/video.mp4 --audio /absolute/path/voice.wav
```

Defaults are 80 frames, 832x480 and 16 fps, delivering five seconds. Explicit
geometry options support smaller controls. The destination must not exist;
publication preserves this rule even if another writer creates it during
encoding. Non-finite or incorrectly sized input is rejected before FFmpeg.
Temporary files are kept beside the output and removed on failure.

A two-frame 8x8 red/green control with audio passed FFprobe's frame, resolution,
duration and audio-stream checks. Decoding the resulting MP4 confirmed channel
order. Existing-output and NaN controls were rejected. This checks export,
not model generation or visual quality. Refresh after exporter, normalization,
FFmpeg or output-layout changes.
