#!/usr/bin/env python3
"""Postterminal, CPU-only byte checks for the bounded RGB-history experiment.

These checks do not execute model arithmetic or measure inference speed. They
independently reconstruct the input bytes consumed by DCAE from the retained
F32 PixelUMM output, including channel-major frame selection and BF16 RNE.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np

SOURCE_SHAPE = (1, 3, 360, 288, 512)
CLIP_SHAPE = (1, 3, 120, 288, 512)
SOURCE_SHA256 = "83f045adf0b94e7311f7a5b3bab757e4d178c80d3c43f0a4a17bbf06bf9b310b"
SOURCE_BYTES = 637_009_920
CLIP_BYTES = 106_168_320
MODES = ("moving", "repeat-last")
# Computed independently from the frozen RGB source before either model run.
# These are input-construction bindings, not hashes inferred from receipts.
EXPECTED_CLIP_SHA256 = {
    "moving": "9ef6c012fb2e8c4a24b5fb28d7454ff699965e358e97a0e1f3fdf8a7580934ff",
    "repeat-last": "5510cda6059422576b9c3be253f348d08fe80211a5e25825bdae0d646d7e1ebe",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def normalized_bf16(values: np.ndarray) -> bytes:
    """Independent F32 clamp -> exact binary scale/subtract -> BF16 RNE."""
    values = np.asarray(values, dtype="<f4")
    if not np.isfinite(values).all():
        raise ValueError("source RGB contains non-finite values")
    values = np.clip(values, np.float32(0), np.float32(1))
    values = np.subtract(np.multiply(values, np.float32(2)), np.float32(1))
    bits = values.view("<u4")
    rounding = np.uint32(0x7FFF) + ((bits >> np.uint32(16)) & np.uint32(1))
    return ((bits + rounding) >> np.uint32(16)).astype("<u2").tobytes(order="C")


def expected_clip_chunks(source: np.ndarray, mode: str):
    if mode not in MODES:
        raise ValueError("RGB history mode must be moving or repeat-last")
    if source.shape != SOURCE_SHAPE or source.dtype != np.dtype("<f4"):
        raise ValueError("source RGB must have the frozen F32 channel-major extent")
    yield from _selected_chunks(source, mode)


def _selected_chunks(source: np.ndarray, mode: str):
    # Separated only to exercise frame/channel transport with tiny CPU
    # fixtures. Public artifact admission above retains the fixed geometry.
    if mode not in MODES:
        raise ValueError("RGB history mode must be moving or repeat-last")
    for channel in range(3):
        for frame in range(120):
            selected = frame if mode == "moving" else 119
            yield normalized_bf16(source[0, channel, selected])


def _compare_chunks(stream, expected_chunks) -> str:
    digest = hashlib.sha256()
    for index, expected in enumerate(expected_chunks):
        actual = stream.read(len(expected))
        if actual != expected:
            raise ValueError(f"normalized DCAE clip differs at channel/frame chunk {index}")
        digest.update(actual)
    if stream.read(1):
        raise ValueError("normalized DCAE clip has trailing bytes")
    return digest.hexdigest()


def validate_clip_bytes(source: np.ndarray, clip_path: Path, mode: str) -> dict:
    path = Path(clip_path)
    if path.is_symlink() or not path.is_file() or path.stat().st_size != CLIP_BYTES:
        raise ValueError("normalized DCAE clip must be a regular exact-size BF16 file")
    with path.open("rb") as stream:
        digest = _compare_chunks(stream, expected_clip_chunks(source, mode))
    if digest != EXPECTED_CLIP_SHA256[mode]:
        raise ValueError("normalized DCAE clip differs from the prelaunch arm digest")
    return {"mode": mode, "shape": list(CLIP_SHAPE), "dtype": "BF16LE",
            "bytes": CLIP_BYTES, "sha256": digest,
            "all_input_bytes_equal": True,
            "scope": "postterminal independent input construction; no model or motion claim"}


def validate_source(source_path: Path) -> np.memmap:
    path = Path(source_path)
    if path.is_symlink() or not path.is_file() or path.stat().st_size != SOURCE_BYTES:
        raise ValueError("source RGB file type or extent differs from frozen PixelUMM output")
    if sha256(path) != SOURCE_SHA256:
        raise ValueError("source RGB identity differs from frozen PixelUMM output")
    return np.memmap(path, mode="r", dtype="<f4", shape=SOURCE_SHAPE)
