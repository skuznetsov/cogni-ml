#!/usr/bin/env python3
"""Prepare setup-only CPU inputs for the native Wan S2V runner.

Image/audio decoding, Wan text cleanup/tokenization, and seeded initial noise
are performed on the host. No model is loaded and no neural inference runs.
The runner consumes the exact raw little-endian tensors described by manifest.
"""
from __future__ import annotations

import argparse
import hashlib
import html
import json
import math
from pathlib import Path
import re
import struct
import string

import ftfy
import librosa
import numpy as np
from PIL import Image
import torch
from transformers import AutoTokenizer


NEGATIVE_PROMPT = (
    "画面模糊，最差质量，画面模糊，细节模糊不清，情绪激动剧烈，手快速抖动，字幕，"
    "丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，"
    "形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走"
)
TEXT_TOKENS = 512
HEIGHT = 512
WIDTH = 832
VIDEO_FRAMES = 21
LATENT_CHANNELS = 16
LATENT_HEIGHT = HEIGHT // 8
LATENT_WIDTH = WIDTH // 8


def prepare_reference_image(image: Image.Image) -> np.ndarray:
    """Match torchvision Resize(512), CenterCrop((512, 832)), ToTensor, [-1, 1].

    Pillow handles the upstream PIL bilinear resize. The explicit integer
    sizing, symmetric zero padding, and rounded center offsets mirror
    torchvision's PIL implementation without importing its native operators.
    """
    image = image.convert("RGB")
    width, height = image.size
    if min(width, height) == HEIGHT:
        resized_width, resized_height = width, height
    elif width <= height:
        resized_width = HEIGHT
        resized_height = int(HEIGHT * height / width)
    else:
        resized_height = HEIGHT
        resized_width = int(HEIGHT * width / height)
    image = image.resize((resized_width, resized_height), Image.Resampling.BILINEAR)

    pad_left = max((WIDTH - resized_width) // 2, 0)
    pad_right = max((WIDTH - resized_width + 1) // 2, 0)
    pad_top = max((HEIGHT - resized_height) // 2, 0)
    pad_bottom = max((HEIGHT - resized_height + 1) // 2, 0)
    if pad_left or pad_top or pad_right or pad_bottom:
        from PIL import ImageOps

        image = ImageOps.expand(image, border=(pad_left, pad_top, pad_right, pad_bottom), fill=0)

    crop_left = int(round((image.width - WIDTH) / 2.0))
    crop_top = int(round((image.height - HEIGHT) / 2.0))
    image = image.crop((crop_left, crop_top, crop_left + WIDTH, crop_top + HEIGHT))

    pixels = np.asarray(image, dtype=np.float32) / np.float32(255.0)
    pixels = pixels * np.float32(2.0) - np.float32(1.0)
    return np.ascontiguousarray(pixels.transpose(2, 0, 1)[None, :, None, :, :], dtype=np.float32)


def clean_text(text: str) -> str:
    # Mirrors wan_modules_tokenizers.py basic_clean + whitespace_clean.
    text = ftfy.fix_text(text)
    text = html.unescape(html.unescape(text))
    return re.sub(r"\s+", " ", text).strip()


def write_array(path: Path, values: np.ndarray, dtype: str) -> dict[str, object]:
    array = np.ascontiguousarray(values, dtype=np.dtype(dtype).newbyteorder("<"))
    payload = memoryview(array).cast("B")
    with path.open("xb") as output:
        output.write(payload)
        output.flush()
    return {
        "file": path.name,
        "dtype": "F32_LE" if dtype == "f4" else "I32_LE",
        "shape": list(array.shape),
        "bytes": len(payload),
        "sha256": hashlib.sha256(payload).hexdigest(),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", type=Path, required=True, help="reference image, RGB decoded by Pillow")
    parser.add_argument("--audio", type=Path, required=True, help="full audio utterance, resampled to 16 kHz by librosa")
    parser.add_argument("--audio-seconds", type=float, default=5.0,
                        help="leading audio duration used by the fixed 5-second S2V bucket (default: 5.0)")
    parser.add_argument("--output-dir", type=Path, required=True, help="new directory; existing paths are rejected")
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--tokenizer", default="google/umt5-xxl")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=False)

    image_tensor = prepare_reference_image(Image.open(args.image))

    waveform, sample_rate = librosa.load(args.audio, sr=16_000, mono=True)
    if sample_rate != 16_000 or waveform.ndim != 1 or waveform.size == 0:
        raise ValueError("Expected a nonempty mono waveform resampled to 16 kHz")
    exact_samples = args.audio_seconds * sample_rate
    target_samples = round(exact_samples)
    if args.audio_seconds <= 0 or not math.isclose(exact_samples, target_samples, rel_tol=0.0, abs_tol=1e-8):
        raise ValueError("--audio-seconds must map to an integral positive sample count at 16 kHz")
    source_sample_count = int(waveform.size)
    if source_sample_count < target_samples:
        raise ValueError(
            f"Audio has {source_sample_count} resampled samples; {target_samples} are required; padding is not implicit"
        )
    waveform = waveform[:target_samples]
    waveform = np.asarray(waveform, dtype=np.float32)
    audio_source_sha256 = hashlib.sha256(args.audio.read_bytes()).hexdigest()

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, local_files_only=True)
    text = clean_text(args.prompt)
    negative = clean_text(NEGATIVE_PROMPT)

    def tokenize(value: str) -> tuple[np.ndarray, np.ndarray]:
        encoded = tokenizer(
            [value],
            max_length=TEXT_TOKENS,
            padding="max_length",
            truncation=True,
            return_tensors="np",
        )
        ids = np.asarray(encoded["input_ids"], dtype=np.int32).reshape(TEXT_TOKENS)
        mask = np.asarray(encoded["attention_mask"], dtype=np.int32).reshape(TEXT_TOKENS)
        return ids, mask

    positive_ids, positive_mask = tokenize(text)
    negative_ids, negative_mask = tokenize(negative)
    generator = torch.Generator(device="cpu").manual_seed(args.seed)
    noise = torch.randn(
        (1, LATENT_CHANNELS, VIDEO_FRAMES, LATENT_HEIGHT, LATENT_WIDTH),
        generator=generator,
        dtype=torch.float32,
        device="cpu",
    ).numpy()

    tensors = {
        "reference.f32": write_array(output_dir / "reference.f32", image_tensor, "f4"),
        "audio.f32": write_array(output_dir / "audio.f32", waveform, "f4"),
        "positive_ids.i32": write_array(output_dir / "positive_ids.i32", positive_ids, "i4"),
        "positive_mask.i32": write_array(output_dir / "positive_mask.i32", positive_mask, "i4"),
        "negative_ids.i32": write_array(output_dir / "negative_ids.i32", negative_ids, "i4"),
        "negative_mask.i32": write_array(output_dir / "negative_mask.i32", negative_mask, "i4"),
        "noise.f32": write_array(output_dir / "noise.f32", noise, "f4"),
    }
    manifest = {
        "schema": "wan_s2v_native_inputs_v1",
        "geometry": {
            "canvas_height": HEIGHT,
            "canvas_width": WIDTH,
            "deliver_frames": 80,
            "fps": 16,
            "internal_frames": 84,
            "latent_frames": VIDEO_FRAMES,
            "latent_height": LATENT_HEIGHT,
            "latent_width": LATENT_WIDTH,
            "text_tokens": TEXT_TOKENS,
        },
        "seed": args.seed,
        "conditioning": "first-request-no-pose; COND zeroed on GPU; reference image and zero motion input",
        "audio": {
            "sample_rate": sample_rate,
            "sample_count": int(waveform.size),
            "source_sample_count": source_sample_count,
            "requested_seconds": args.audio_seconds,
            "crop": "leading_samples_after_16khz_mono_resample; no padding",
            "source_file": args.audio.name,
            "source_file_sha256": audio_source_sha256,
        },
        "tokenizer": args.tokenizer,
        "prompt_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
        "negative_prompt_sha256": hashlib.sha256(negative.encode("utf-8")).hexdigest(),
        "tensors": tensors,
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"output_dir": str(output_dir), "audio_source_samples": source_sample_count,
                      "audio_samples": int(waveform.size), "seed": args.seed}))


if __name__ == "__main__":
    main()
