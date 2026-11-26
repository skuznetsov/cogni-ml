#!/usr/bin/env python3
"""Decode the saved portrait latents with pinned upstream PixelUMM.

Default mode authenticates the native capture, its exact decoder-noise prefix,
the current full-depth BF16 weight cache, and upstream source without importing
PyTorch or touching MPS. ``--execute`` runs only the PixelUMM decoder: it does
not rerun the supplied T0 encoder or DiT. The comparison covers generated T1
display pixels only (frames [120, 240)).
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Mapping


DEFAULT_ROOT = Path("/opt/mldata_ssd/cogni-ml/hy-nomas-native-context-20261106.ui7736ue")
DEFAULT_UPSTREAM = Path("/opt/mldata_ssd/models/hy-nomas/upstream-911cea6458a8")
DEFAULT_CACHE = Path("/opt/mldata_ssd/models/hy-nomas/pixelumm-ft-bac719a8/weights/pixelumm-thirty-six-bf16-cache.json")
DEFAULT_STEM = DEFAULT_CACHE.parent / "stem-f32.safetensors"
DEFAULT_HEAD = DEFAULT_CACHE.parent / "head-f32.safetensors"

UPSTREAM_REVISION = "911cea6458a89c2663513ff7099561ec190dd491"
MODEL_SHA256 = "bea01a8490f8e9a4076da83d4f9fbb3cba7bd6701df5c28fb2e2c04f00131eec"
DCAE_SOURCE_SHA256 = "38163ebead9a6bb643d27843022c1757c45c6b8638c334a476aee5b373edf8c1"
BLENDER_SOURCE_SHA256 = "65e0d77b9f2e9a3ca457d6a51bfdb239f7b9dc3fc8a0ad014025c36d6be9f0cf"
DISPLAY_SOURCE_SHA256 = "77a039514ff494cb496a194c60ca3270f09c15b194bda25448a036a6fcc79582"

NATIVE_RECEIPT_SHA256 = "ca8be135ef659497594dec9647c72240f911be8cfd6be989f0740fbf76cd66f5"
INPUT_MANIFEST_SHA256 = "0f35bb09964f2b28d28069e8d39a8a1d6194607d4269de6bb824ff54d828e2ac"
INPUT_PAYLOAD_SHA256 = "283fecfdd64107f5b9e7b0ff9784fcca977e4170c416bc11fc4d14e0664035ad"
NOISE_SHA256 = "12fc5a9263e6784264b6b91e7e980acd00f81a9b0f9b19742dc9618cc498f0b2"
LATENT_BANK_SHA256 = "1717edc3d6cf33a3a374852a3dc58cd9d82d5f74965fab6dd0239bbe93da943b"
NATIVE_PIXELS_SHA256 = "44af661a5157136472124a72bc12a0ba1b7bb00a4510941f6f8d44d776bad898"
CACHE_MANIFEST_SHA256 = "7cbf6d8a3587f0cd23e07120a9ad1a222b4ffadfa4c1c840609c425d16e3476d"
CACHE_PAYLOAD_SHA256 = "152ba021b52d9546e4c8821b17ab11f67f46486d521b500de3766f14dd2217ca"
STEM_SHA256 = "7eb73ebf42f6681e21870dc2d2d4b592846ba15ed5bf5f9a6c7a83d39c2d8d04"
HEAD_SHA256 = "d0e985e45b6b5fcebde381dd4b61f3b77394bd288323cd8afe2a7604584e6e9f"
HEADER_SHA256 = "7e7a92cfe84d54c0da3a132099a9472766ae0034042515a0f76b6198bbd09d11"
ROPE_COS_SHA256 = "c704e1397520e7a8536994143e56672bea6cab31170b0ffe7a364cb8abb4ac0e"
ROPE_SIN_SHA256 = "0f3d33d7ce2581e918bc2d91cc73639d9bdf42eb2dfd28a20785b6face97c639"

LATENT_BANK_SHAPE = (1, 32, 180, 9, 16)
LATENT_SHAPE = (1, 32, 60, 9, 16)
NOISE_SHAPE = (1, 3, 720, 288, 512)
NOISE_PREFIX_SHAPE = (1, 3, 240, 288, 512)
DECODER_ROPE_SHAPE = (1, 5760, 1, 128)
PIXELS_SHAPE = (1, 3, 240, 288, 512)
WINDOW_LATENTS = 10
OVERLAP_LATENTS = 5
WINDOW_STARTS = tuple(range(0, 51, 5))
T1_PIXEL_CROP = (120, 240)
ATOL = RTOL = 2e-5
GIB = 1024**3
MAX_PROCESS_BYTES = 18 * GIB
RESERVE_FRACTION = 0.10
SCHEMA = "cogni_ml.hy_nomas_portrait_decoder_control/v1"


class ControlError(ValueError):
    """An input, source, cache, runtime, or output boundary was rejected."""


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ControlError(f"duplicate JSON field: {key}")
        result[key] = value
    return result


def _read_json(path: Path, what: str, max_bytes: int = 8 * 1024 * 1024) -> tuple[dict, bytes]:
    raw = _read_regular(path, what, max_bytes=max_bytes)
    try:
        value = json.loads(raw.decode("utf-8"), object_pairs_hook=_unique_object)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ControlError(f"cannot parse {what}") from error
    if type(value) is not dict:
        raise ControlError(f"{what} must be a JSON object")
    return value, raw


def _read_regular(path: Path, what: str, *, max_bytes: int | None = None) -> bytes:
    if path.is_symlink() or not path.is_file():
        raise ControlError(f"{what} must be a regular non-symlink file: {path}")
    try:
        with path.open("rb") as stream:
            raw = stream.read() if max_bytes is None else stream.read(max_bytes + 1)
    except OSError as error:
        raise ControlError(f"cannot read {what}: {path}") from error
    if max_bytes is not None and len(raw) > max_bytes:
        raise ControlError(f"{what} exceeds its size limit")
    return raw


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_range(path: Path, offset: int, nbytes: int) -> str:
    if type(offset) is not int or type(nbytes) is not int or offset < 0 or nbytes <= 0:
        raise ControlError("invalid authenticated tensor range")
    digest = hashlib.sha256()
    remaining = nbytes
    with path.open("rb") as stream:
        stream.seek(offset)
        while remaining:
            chunk = stream.read(min(8 * 1024 * 1024, remaining))
            if not chunk:
                raise ControlError("truncated authenticated tensor range")
            digest.update(chunk)
            remaining -= len(chunk)
    return digest.hexdigest()


def _require_digest(actual: str, expected: str, what: str) -> None:
    if actual != expected:
        raise ControlError(f"{what} SHA-256 mismatch: got {actual}")


def _shape_bytes(shape: object, width: int) -> int:
    if (type(shape) is not list or not shape
            or any(type(dim) is not int or dim <= 0 for dim in shape)):
        raise ControlError("weight-cache tensor shape is malformed")
    return math.prod(shape) * width


def _prefix_sha256(bank_path: Path) -> str:
    """Hash [1,32,60,9,16] in its contiguous NCTHW serialization."""
    digest = hashlib.sha256()
    per_channel = math.prod(LATENT_BANK_SHAPE[2:]) * 4
    prefix = math.prod(LATENT_SHAPE[2:]) * 4
    with bank_path.open("rb") as stream:
        for channel in range(LATENT_BANK_SHAPE[1]):
            stream.seek(channel * per_channel)
            chunk = stream.read(prefix)
            if len(chunk) != prefix:
                raise ControlError("native latent bank truncated during prefix selection")
            digest.update(chunk)
    return digest.hexdigest()


def _check_source(upstream: Path) -> dict[str, str]:
    if upstream.is_symlink() or not upstream.is_dir():
        raise ControlError("upstream root must be a non-symlink directory")
    upstream = upstream.resolve(strict=True)
    files = {
        "hymm/models/vae/pixelumm_decoder/model.py": MODEL_SHA256,
        "hymm/models/vae/autoencoder_dcae_v.py": DCAE_SOURCE_SHA256,
        "hymm/models/vae/pixelumm_decoder/autoencoder.py": BLENDER_SOURCE_SHA256,
        "hymm/pipeline/base.py": DISPLAY_SOURCE_SHA256,
    }
    try:
        revision = subprocess.check_output(
            ["git", "-C", str(upstream), "rev-parse", "HEAD"],
            text=True, stderr=subprocess.DEVNULL,
        ).strip()
        dirty = subprocess.check_output(
            ["git", "-C", str(upstream), "status", "--porcelain", "--", *files],
            text=True, stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError) as error:
        raise ControlError("cannot verify pinned upstream git source") from error
    if revision != UPSTREAM_REVISION or dirty:
        raise ControlError("upstream revision or decoder source worktree identity mismatch")
    result = {"revision": revision}
    for relative, expected in files.items():
        path = upstream / relative
        if path.is_symlink() or not path.is_file():
            raise ControlError(f"missing pinned upstream source: {relative}")
        actual = _sha256_file(path)
        _require_digest(actual, expected, f"upstream {relative}")
        result[relative] = actual
    return result


def _check_cache(manifest_path: Path) -> dict[str, object]:
    if manifest_path.is_symlink() or not manifest_path.is_file():
        raise ControlError("weight-cache manifest must be a regular file")
    raw = _read_regular(manifest_path, "weight-cache manifest", max_bytes=32 * 1024 * 1024)
    _require_digest(hashlib.sha256(raw).hexdigest(), CACHE_MANIFEST_SHA256, "weight-cache manifest")
    try:
        manifest = json.loads(raw.decode("utf-8"), object_pairs_hook=_unique_object)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ControlError("cannot parse weight-cache manifest") from error
    if (type(manifest) is not dict
            or manifest.get("schema") != "cogni_ml.pixelumm_bf16_weight_cache/v1"
            or manifest.get("upstream_revision") != UPSTREAM_REVISION
            or manifest.get("upstream_model_sha256") != MODEL_SHA256):
        raise ControlError("weight-cache identity differs from the pinned PixelUMM source")
    if (manifest.get("payload") != "pixelumm-thirty-six-bf16-cache.bin"
            or type(manifest.get("payload_bytes")) is not int
            or manifest.get("payload_bytes") != 13_892_143_104
            or manifest.get("payload_sha256") != CACHE_PAYLOAD_SHA256
            or manifest.get("checkpoint_header_sha256") != HEADER_SHA256):
        raise ControlError("weight-cache payload identity or extent is unsupported")
    blocks, weights = manifest.get("blocks"), manifest.get("weights")
    if (type(blocks) is not dict or set(blocks) != {str(i) for i in range(36)}
            or type(weights) is not dict or len(weights) != 468):
        raise ControlError("weight-cache must contain the exact 36-block/468-tensor inventory")
    ranges = []
    for index in range(36):
        block = blocks[str(index)]
        expected_prefix = f"block{index}."
        keys = block.get("weight_keys") if type(block) is dict else None
        if (type(keys) is not list or len(keys) != 13
                or keys != sorted(key for key in weights if key.startswith(expected_prefix))):
            raise ControlError(f"weight-cache block {index} ownership is malformed")
        for key in keys:
            record = weights[key]
            if (type(record) is not dict or record.get("dtype") != "BF16"
                    or len(record.get("sha256", "")) != 64
                    or type(record.get("offset")) is not int
                    or type(record.get("nbytes")) is not int
                    or record["nbytes"] != _shape_bytes(record.get("shape"), 2)
                    or record["offset"] < 0 or record["offset"] % 2):
                raise ControlError(f"weight-cache tensor record is malformed: {key}")
            ranges.append((record["offset"], record["offset"] + record["nbytes"], key))
    ranges.sort()
    cursor = 0
    for start, end, key in ranges:
        if start != cursor or end <= start:
            raise ControlError(f"weight-cache payload ranges have a gap or overlap at {key}")
        cursor = end
    if cursor != manifest["payload_bytes"]:
        raise ControlError("weight-cache tensor ranges do not cover the payload")
    payload = manifest_path.parent / manifest["payload"]
    if payload.is_symlink() or not payload.is_file() or payload.stat().st_size != manifest["payload_bytes"]:
        raise ControlError("weight-cache payload is missing or has the wrong extent")
    _require_digest(_sha256_file(payload), CACHE_PAYLOAD_SHA256, "weight-cache payload")
    return {"manifest": manifest, "manifest_sha256": CACHE_MANIFEST_SHA256,
            "payload_path": payload.resolve(strict=True), "payload_sha256": CACHE_PAYLOAD_SHA256}


def _fresh_output_dir(path: Path) -> Path:
    path = path.expanduser()
    if not path.is_absolute():
        path = Path.cwd() / path
    if path.is_symlink() or path.exists() or not path.name:
        raise ControlError("--output-dir must name a fresh, non-symlink directory")
    try:
        parent = path.parent.resolve(strict=True)
    except OSError as error:
        raise ControlError("--output-dir parent must already exist") from error
    if not parent.is_dir():
        raise ControlError("--output-dir parent is not a directory")
    return parent / path.name


def preflight(args) -> dict[str, object]:
    root = args.run_root.expanduser()
    if root.is_symlink() or not root.is_dir():
        raise ControlError("portrait run root must be a non-symlink directory")
    root = root.resolve(strict=True)
    receipt_path = root / "photo-native-v2/receipt.json"
    receipt, receipt_raw = _read_json(receipt_path, "native portrait receipt")
    receipt_sha = hashlib.sha256(receipt_raw).hexdigest()
    _require_digest(receipt_sha, NATIVE_RECEIPT_SHA256, "native portrait receipt")
    expected_receipt = {
        "schema": "cogni_ml.hy_nomas_native_continuation/v1",
        "upstream_revision": UPSTREAM_REVISION,
        "inputs_sha256": INPUT_MANIFEST_SHA256,
        "frames": 240, "width": 512, "height": 288, "fps": 24,
        "generated_segment_start_index": 1,
        "generated_latent_frames": 30,
        "generated_duration_seconds": 5,
        "pixel_output_duration_seconds": 10,
        "latent_bank_shape": list(LATENT_BANK_SHAPE),
        "pixelumm": {"steps": 1, "window_latents": 10,
                     "overlap_latents": 5, "global_windows": 11},
        "source_bf16": True,
        "full_upstream_video_parity": False,
    }
    if any(receipt.get(key) != value for key, value in expected_receipt.items()):
        raise ControlError("native portrait receipt differs from the admitted two-segment capture")
    diagnostics = receipt.get("diagnostics")
    if type(diagnostics) is not dict:
        raise ControlError("native portrait receipt diagnostics are missing")
    bank = root / "photo-native-v2/latent-bank.f32le"
    pixels = root / "photo-native-v2/pixels.f32le"
    for path, expected_sha, expected_bytes, label in (
        (bank, LATENT_BANK_SHA256, math.prod(LATENT_BANK_SHAPE) * 4, "native latent bank"),
        (pixels, NATIVE_PIXELS_SHA256, math.prod(PIXELS_SHAPE) * 4, "native display pixels"),
    ):
        if path.is_symlink() or not path.is_file() or path.stat().st_size != expected_bytes:
            raise ControlError(f"{label} has an unexpected file type or extent")
        _require_digest(_sha256_file(path), expected_sha, label)
    if (diagnostics.get("latent_bank_sha256") != LATENT_BANK_SHA256
            or diagnostics.get("pixels_sha256") != NATIVE_PIXELS_SHA256):
        raise ControlError("native receipt artifact hashes differ from the pinned portrait files")

    manifest_path = root / "photo-continuation-v2.json"
    manifest, manifest_raw = _read_json(manifest_path, "continuation input manifest")
    manifest_sha = hashlib.sha256(manifest_raw).hexdigest()
    _require_digest(manifest_sha, INPUT_MANIFEST_SHA256, "continuation input manifest")
    if (manifest.get("schema") != "cogni_ml.hy_nomas_continuation_inputs/v1"
            or manifest.get("upstream_revision") != UPSTREAM_REVISION):
        raise ControlError("continuation input manifest identity mismatch")
    payload_path = root / manifest.get("payload", "")
    if (not isinstance(manifest.get("payload"), str)
            or payload_path.name != manifest["payload"] or payload_path.is_symlink()
            or not payload_path.is_file()
            or payload_path.stat().st_size != manifest.get("payload_bytes")):
        raise ControlError("continuation payload path or extent is invalid")
    _require_digest(_sha256_file(payload_path), INPUT_PAYLOAD_SHA256, "continuation payload")
    tensors = manifest.get("tensors")
    noise_record = tensors.get("decoder.noise") if type(tensors) is dict else None
    if (type(noise_record) is not dict or noise_record.get("dtype") != "F32"
            or noise_record.get("shape") != list(NOISE_SHAPE)
            or noise_record.get("nbytes") != math.prod(NOISE_SHAPE) * 4
            or type(noise_record.get("offset")) is not int
            or noise_record["offset"] < 0 or noise_record["offset"] % 4
            or noise_record.get("sha256") != NOISE_SHA256):
        raise ControlError("continuation decoder.noise record geometry or digest mismatch")
    if noise_record["offset"] + noise_record["nbytes"] > payload_path.stat().st_size:
        raise ControlError("continuation decoder.noise range exceeds payload extent")
    _require_digest(_sha256_range(payload_path, noise_record["offset"], noise_record["nbytes"]),
                    NOISE_SHA256, "continuation decoder.noise record")
    rope_records = {}
    for name, expected_sha in (("decoder.rope_cos", ROPE_COS_SHA256),
                               ("decoder.rope_sin", ROPE_SIN_SHA256)):
        record = tensors.get(name)
        if (type(record) is not dict or record.get("dtype") != "F32"
                or record.get("shape") != list(DECODER_ROPE_SHAPE)
                or record.get("nbytes") != math.prod(DECODER_ROPE_SHAPE) * 4
                or type(record.get("offset")) is not int or record["offset"] < 0
                or record["offset"] % 4 or record.get("sha256") != expected_sha):
            raise ControlError(f"continuation {name} record geometry or digest mismatch")
        if record["offset"] + record["nbytes"] > payload_path.stat().st_size:
            raise ControlError(f"continuation {name} range exceeds payload extent")
        _require_digest(_sha256_range(payload_path, record["offset"], record["nbytes"]),
                        expected_sha, f"continuation {name} record")
        rope_records[name] = record

    prefix_sha = _prefix_sha256(bank)
    selected_latents = bank
    if args.input_latents is not None:
        if args.input_latents_sha256 is None:
            raise ControlError("--input-latents requires --input-latents-sha256")
        selected_latents = args.input_latents.expanduser()
        if not selected_latents.is_absolute():
            selected_latents = Path.cwd() / selected_latents
        if (selected_latents.is_symlink() or not selected_latents.is_file()
                or selected_latents.stat().st_size != math.prod(LATENT_SHAPE) * 4):
            raise ControlError("--input-latents must be a regular F32 file with [1,32,60,9,16] extent")
        source_sha = _sha256_file(selected_latents)
        _require_digest(source_sha, args.input_latents_sha256, "--input-latents")
        _require_digest(source_sha, prefix_sha, "upstream 60-latent file vs native bank prefix")
    elif args.input_latents_sha256 is not None:
        raise ControlError("--input-latents-sha256 requires --input-latents")

    cache = _check_cache(args.weight_cache_manifest.expanduser())
    stem = args.stem_weights.expanduser()
    head = args.head_weights.expanduser()
    for path, expected, label in ((stem, STEM_SHA256, "PixelUMM stem"),
                                  (head, HEAD_SHA256, "PixelUMM head")):
        if path.is_symlink() or not path.is_file():
            raise ControlError(f"{label} safetensors file is missing")
        _require_digest(_sha256_file(path), expected, label)
    header = args.weight_cache_manifest.expanduser().parent / "pixelumm-header.json"
    if header.is_symlink() or not header.is_file():
        raise ControlError("pinned PixelUMM checkpoint header is missing")
    _require_digest(_sha256_file(header), HEADER_SHA256, "PixelUMM checkpoint header")
    source = _check_source(args.upstream.expanduser())
    output_dir = _fresh_output_dir(args.output_dir)
    try:
        torch_version = importlib.metadata.version("torch")
    except importlib.metadata.PackageNotFoundError:
        torch_version = "not-installed"
    if torch_version != "2.6.0":
        raise ControlError(f"decoder execution requires Torch 2.6.0; environment metadata reports {torch_version}")
    return {
        "root": root, "receipt": receipt, "receipt_sha256": receipt_sha,
        "manifest": manifest, "manifest_path": manifest_path, "manifest_sha256": manifest_sha,
        "payload_path": payload_path.resolve(strict=True), "payload_sha256": INPUT_PAYLOAD_SHA256,
        "noise_record": noise_record, "bank_path": bank.resolve(strict=True),
        "rope_records": rope_records,
        "bank_sha256": LATENT_BANK_SHA256, "pixels_path": pixels.resolve(strict=True),
        "pixels_sha256": NATIVE_PIXELS_SHA256, "prefix_sha256": prefix_sha,
        "latent_path": selected_latents.resolve(strict=True),
        "latent_sha256": prefix_sha, "custom_latents": args.input_latents is not None,
        "cache_manifest": cache["manifest"], "cache_manifest_sha256": cache["manifest_sha256"],
        "cache_payload_path": cache["payload_path"], "cache_payload_sha256": cache["payload_sha256"],
        "stem_path": stem.resolve(strict=True), "head_path": head.resolve(strict=True),
        "upstream": args.upstream.expanduser().resolve(strict=True),
        "source_metadata": source, "output_dir": output_dir,
        "torch_version": torch_version,
    }


def _runtime_preflight(torch, block_reference, arch, cache_bytes: int) -> dict[str, object]:
    if torch.__version__ != "2.6.0":
        raise ControlError(f"decoder execution requires Torch 2.6.0; got {torch.__version__}")
    if not torch.backends.mps.is_available():
        raise ControlError("MPS unavailable; CPU fallback is prohibited")
    if os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK") != "0":
        raise ControlError("set PYTORCH_ENABLE_MPS_FALLBACK=0 for the decoder execution")
    grid = (10, 18, 32)
    base = block_reference._estimated_peak_bytes(
        grid, arch, layers=36, resident_layers=1,
        capture_stages=block_reference.FULL_TARGET_STAGES,
        full36_capture_policy=block_reference.THIRTY_SIX_CAPTURE_POLICY,
    )
    physical = block_reference._physical_memory_bytes()
    allocated = int(torch.mps.driver_allocated_memory())
    # The 13.9 GB cache is file mapped on CPU; at most one BF16 block is resident
    # on MPS. Include complete display comparison, merge/output, and allocator room.
    one_block = 402_653_184  # 385,892,864-byte largest block plus BF16 transfer workspace.
    compare = 6 * (120 * 3 * 288 * 512 * 4)
    estimate = base + one_block + compare + 2 * GIB
    reserve = int(physical * RESERVE_FRACTION)
    if estimate > MAX_PROCESS_BYTES:
        raise ControlError("estimated decoder footprint exceeds the 18 GiB guard cap")
    if physical - allocated - estimate < reserve:
        raise ControlError("estimated decoder footprint would leave less than 10% physical memory free")
    return {
        "physical_bytes": physical, "mps_allocated_bytes": allocated,
        "estimated_peak_bytes": estimate, "estimated_peak_gib": estimate / GIB,
        "process_cap_bytes": MAX_PROCESS_BYTES, "reserve_floor_bytes": reserve,
        "cache_payload_bytes_cpu_mapped": cache_bytes,
        "active_bf16_block_allowance_bytes": one_block,
        "guard": "wan_memory_guard.py --memory-accounting file-cache --runtime-floor-pct 10 --max-footprint-mib 18432 --max-swap-growth-mib 64",
        "estimate_scope": "one active BF16 block; 11 windows; 120-frame T1 display crop and comparison buffers",
    }


def _cache_layer(torch, source, arch, cache_bytes, manifest, block_index: int, device):
    """Build one upstream GenLayer directly from authenticated BF16 cache records."""
    prefix = f"block{block_index}."
    block = manifest["blocks"][str(block_index)]
    with torch.device("meta"):
        layer = source.GenLayer(arch)
    expected_state = layer.state_dict()
    expected_names = {prefix + name for name in expected_state}
    if set(block["weight_keys"]) != expected_names:
        raise ControlError(f"cache block {block_index} keys differ from upstream GenLayer")
    state = {}
    for name, meta_tensor in expected_state.items():
        record = manifest["weights"][prefix + name]
        if tuple(record["shape"]) != tuple(meta_tensor.shape):
            raise ControlError(f"cache block {block_index} shape mismatch: {name}")
        count = record["nbytes"] // 2
        tensor = cache_bytes.narrow(0, record["offset"], record["nbytes"])
        tensor = tensor.view(torch.bfloat16).view(record["shape"])
        if tensor.numel() != count or tensor.dtype is not torch.bfloat16:
            raise ControlError(f"cache block {block_index} BF16 view mismatch: {name}")
        state[name] = tensor
    result = layer.load_state_dict(state, strict=True, assign=True)
    if result.missing_keys or result.unexpected_keys:
        raise ControlError(f"cache block {block_index} strict parameter assignment failed")
    layer.eval().requires_grad_(False).to(device=device)
    if any(parameter.dtype is not torch.bfloat16 for parameter in layer.parameters()):
        raise ControlError(f"cache block {block_index} did not remain BF16")
    return layer


def _write_fresh(path: Path, data) -> str:
    try:
        with path.open("xb") as stream:
            data.tofile(stream)
            stream.flush()
            os.fsync(stream.fileno())
    except OSError as error:
        raise ControlError(f"cannot write output artifact: {path.name}") from error
    return _sha256_file(path)


def execute(admission: Mapping[str, object]) -> dict[str, object]:
    started_at = time.perf_counter()
    import torch

    support = Path(__file__).resolve().parent
    if str(support) not in sys.path:
        sys.path.insert(0, str(support))
    import hy_nomas_pixelumm_real_input as real_input
    import hy_nomas_pixelumm_reference as block_reference
    import hy_nomas_pixelumm_checkpoint as checkpoint_inventory
    import hy_nomas_pixelumm_head_checkpoint as head_checkpoint
    import pixelumm_generation_reference as pixelumm_reference

    upstream = admission["upstream"]
    source, model_sha = block_reference.load_upstream(upstream)
    if model_sha != MODEL_SHA256:
        raise ControlError("upstream model source changed after CPU preflight")
    _, _, autoencoder_source = real_input._load_modules(upstream)
    if _sha256_file(Path(autoencoder_source.__file__)) != BLENDER_SOURCE_SHA256:
        raise ControlError("pinned PixelUMM blender source changed after CPU preflight")
    try:
        block_reference.validate_preview_oracle_runtime((10, 18, 32), "mps")
    except block_reference.ReferenceError as error:
        raise ControlError(str(error)) from error
    device = torch.device("mps")
    arch = source.QWEN3_8B_GEN
    memory = _runtime_preflight(torch, block_reference, arch,
                                admission["cache_payload_path"].stat().st_size)

    raw_header = (admission["cache_payload_path"].parent / "pixelumm-header.json").read_bytes()
    try:
        checkpoint_inventory.parse_pinned_header(raw_header)
    except checkpoint_inventory.InventoryError as error:
        raise ControlError(f"pinned checkpoint header rejected: {error}") from error
    real_input.STEM_SUBSET_PATH = admission["stem_path"]
    stem = real_input._load_stem(source, device)
    from safetensors import safe_open
    try:
        with safe_open(str(admission["head_path"]), framework="pt", device="cpu") as handle:
            if set(handle.keys()) != set(head_checkpoint.HEAD_TENSORS):
                raise ControlError("head safetensors keys differ from the pinned four-tensor inventory")
            head_state = {name: handle.get_tensor(name) for name in head_checkpoint.HEAD_TENSORS}
    except ControlError:
        raise
    except Exception as error:
        raise ControlError("cannot load pinned PixelUMM head safetensors") from error
    expected_head = {
        "decoder.final_norm.weight": (4096,),
        "decoder.head.linear.bias": (3072,),
        "decoder.head.linear.weight": (3072, 4096),
        "decoder.head.norm.weight": (4096,),
    }
    if any(value.dtype != torch.float32 or tuple(value.shape) != expected_head[name]
           for name, value in head_state.items()):
        raise ControlError("head safetensors dtype or shape differs from the pinned source")
    final_norm = source.RMSNorm(arch.hidden_size, arch.rms_norm_eps).to(
        device=device, dtype=torch.bfloat16
    )
    pixel_head = source.PixelHead(arch.hidden_size).to(device=device, dtype=torch.bfloat16)
    head_destinations = {
        "decoder.final_norm.weight": final_norm.weight,
        "decoder.head.norm.weight": pixel_head.norm.weight,
        "decoder.head.linear.weight": pixel_head.linear.weight,
        "decoder.head.linear.bias": pixel_head.linear.bias,
    }
    with torch.no_grad():
        for name, parameter in head_destinations.items():
            parameter.copy_(head_state[name].to(device=device, dtype=torch.bfloat16))
    del head_state

    if admission["custom_latents"]:
        latent_cpu = torch.from_file(
            str(admission["latent_path"]), shared=False,
            size=math.prod(LATENT_SHAPE), dtype=torch.float32,
        ).view(LATENT_SHAPE)
    else:
        latent_bank_cpu = torch.from_file(
            str(admission["bank_path"]), shared=False,
            size=math.prod(LATENT_BANK_SHAPE), dtype=torch.float32,
        ).view(LATENT_BANK_SHAPE)
        latent_cpu = latent_bank_cpu[:, :, :LATENT_SHAPE[2]].contiguous()
        del latent_bank_cpu
    latent = latent_cpu.to(device=device)
    del latent_cpu
    pixels_cpu = torch.from_file(
        str(admission["pixels_path"]), shared=False,
        size=math.prod(PIXELS_SHAPE), dtype=torch.float32,
    ).view(PIXELS_SHAPE)
    native_t1 = pixels_cpu[:, :, T1_PIXEL_CROP[0]:T1_PIXEL_CROP[1]].to(device=device)
    del pixels_cpu
    payload_bytes = admission["cache_payload_path"].stat().st_size
    cache_bytes = torch.from_file(
        str(admission["cache_payload_path"]), shared=False,
        size=payload_bytes, dtype=torch.uint8,
    )
    payload = torch.from_file(
        str(admission["payload_path"]), shared=False,
        size=admission["payload_path"].stat().st_size // 4,
        dtype=torch.float32,
    )
    noise_record = admission["noise_record"]
    noise = payload.narrow(0, noise_record["offset"] // 4,
                           noise_record["nbytes"] // 4).view(NOISE_SHAPE)
    noise = noise.narrow(2, 0, NOISE_PREFIX_SHAPE[2]).to(device=device)
    rope_cos_record = admission["rope_records"]["decoder.rope_cos"]
    rope_sin_record = admission["rope_records"]["decoder.rope_sin"]
    cos = payload.narrow(0, rope_cos_record["offset"] // 4,
                         rope_cos_record["nbytes"] // 4).view(DECODER_ROPE_SHAPE).to(device=device)
    sin = payload.narrow(0, rope_sin_record["offset"] // 4,
                         rope_sin_record["nbytes"] // 4).view(DECODER_ROPE_SHAPE).to(device=device)
    del payload

    plan, inverse_cpu, _ = block_reference.make_attention_plan(source, (10, 18, 32), tile=8)
    permutation = torch.tensor(plan.permutation, dtype=torch.long, device=device)
    inverse = inverse_cpu.to(device=device)
    # The saved decoder RoPE records are already in the upstream tile-major order.
    condition = type("_ConditionProjection", (), {})()
    condition.cond_proj = stem.cond_proj
    condition.arch = arch
    windows = []
    blender = autoencoder_source._TemporalBlender(total_frames=PIXELS_SHAPE[2])
    with torch.inference_mode():
        with torch.autocast(device_type="mps", dtype=torch.bfloat16):
            for start in WINDOW_STARTS:
                latent_window = latent[:, :, start:start + WINDOW_LATENTS]
                noise_window = noise[:, :, start * 4:(start + WINDOW_LATENTS) * 4]
                xt_tokens = stem.embed(source.patchify(noise_window.to(stem.embed.weight.dtype)))
                condition_tokens = source.DCAEVPixelUMMDecoder._condition_tokens(
                    condition, latent_window.to(stem.cond_proj.weight.dtype)
                )
                windows.append((xt_tokens + condition_tokens).index_select(1, permutation))
                del latent_window, noise_window, xt_tokens, condition_tokens
            for block_index in range(36):
                layer = _cache_layer(torch, source, arch, cache_bytes,
                                     admission["cache_manifest"], block_index, device)
                for index, current in enumerate(windows):
                    windows[index], captures = block_reference.run_layer_reference(
                        layer, current, cos, sin, plan
                    )
                    if captures:
                        raise ControlError("unexpected intermediate capture was retained")
                del layer
        for index, start in enumerate(WINDOW_STARTS):
            with torch.autocast(device_type="mps", dtype=torch.bfloat16):
                decoder_tokens = final_norm(windows[index]).index_select(1, inverse)
            prediction = source.unpatchify(pixel_head(decoder_tokens), (10, 18, 32))
            blender.add(prediction, start * 4)
            del decoder_tokens, prediction

    raw = blender.result()
    if raw.device.type != "mps" or tuple(raw.shape) != PIXELS_SHAPE or raw.dtype is not torch.bfloat16:
        raise ControlError("upstream PixelUMM result has unexpected dtype, device, or geometry")
    display = pixelumm_reference.postprocess_bf16(torch, raw)
    upstream_t1 = display[:, :, T1_PIXEL_CROP[0]:T1_PIXEL_CROP[1]]
    actual = upstream_t1.reshape(-1)
    expected = native_t1.reshape(-1)
    maxima, sums, mismatches, finite = [], [], [], []
    chunk_size = 1 << 20
    for begin in range(0, actual.numel(), chunk_size):
        stop = min(begin + chunk_size, actual.numel())
        left, right = actual[begin:stop], expected[begin:stop]
        delta = (left - right).abs()
        maxima.append(delta.max())
        sums.append(delta.sum())
        mismatches.append((delta > (ATOL + RTOL * right.abs())).any())
        finite.append(torch.isfinite(left).all().logical_and(torch.isfinite(right).all()))
    metrics = torch.stack((
        torch.stack(maxima).max(),
        torch.stack(sums).sum() / actual.numel(),
        torch.stack(mismatches).any().to(torch.float32),
        torch.stack(finite).all().logical_not().to(torch.float32),
        torch.isfinite(raw).all().logical_not().to(torch.float32),
    ))
    packed = torch.cat((upstream_t1.contiguous().view(-1), metrics))
    # This is the sole synchronization and sole device-to-host transfer.
    cpu = packed.cpu()
    del packed, metrics, latent, noise, native_t1, display, raw, windows, cache_bytes

    pixel_count = math.prod((1, 3, T1_PIXEL_CROP[1] - T1_PIXEL_CROP[0], 288, 512))
    pixels_array = cpu[:pixel_count].numpy().astype("<f4", copy=False)
    max_abs, mean_abs, mismatch, nonfinite, raw_nonfinite = cpu[pixel_count:].tolist()
    out_dir = admission["output_dir"]
    out_dir.mkdir(mode=0o700)
    output_pixels = out_dir / "upstream-t1-display.f32le"
    output_sha = _write_fresh(output_pixels, pixels_array)
    result = {
        "schema": admission.get("output_schema", SCHEMA),
        "upstream_revision": UPSTREAM_REVISION,
        "upstream_model_sha256": model_sha,
        "native_receipt_sha256": admission["receipt_sha256"],
        "native_latent_bank_sha256": admission["bank_sha256"],
        "native_selected_latent_sha256": admission["prefix_sha256"],
        "upstream_input_latents_path": str(admission["latent_path"]),
        "upstream_input_latents_sha256": admission["latent_sha256"] if admission["custom_latents"] else None,
        "upstream_input_latents_match_native_prefix": True if admission["custom_latents"] else None,
        "continuation_manifest_sha256": admission["manifest_sha256"],
        "continuation_payload_sha256": admission["payload_sha256"],
        "decoder_noise_record_sha256": admission["noise_record"]["sha256"],
        "decoder_noise_source_shape": list(NOISE_SHAPE),
        "decoder_noise_selected_axis2": [0, 240],
        "decoder_rope_cos_sha256": ROPE_COS_SHA256,
        "decoder_rope_sin_sha256": ROPE_SIN_SHA256,
        "decoder_rope_already_tile_major": True,
        "native_display_sha256": admission["pixels_sha256"],
        "decoder_cache_manifest_sha256": admission["cache_manifest_sha256"],
        "decoder_cache_payload_sha256": admission["cache_payload_sha256"],
        "stem_weights_sha256": STEM_SHA256,
        "head_weights_sha256": HEAD_SHA256,
        "head_subset_sha256": HEAD_SHA256,
        "decoder": {
            "latent_shape": list(LATENT_SHAPE),
            "window_starts": list(WINDOW_STARTS),
            "window_latents": WINDOW_LATENTS,
            "overlap_latents": OVERLAP_LATENTS,
            "pixel_frame_starts": [start * 4 for start in WINDOW_STARTS],
            "merge": "pinned upstream _TemporalBlender cosine overlap",
            "blocks": 36,
            "source_t0_encoder_rerun": False,
            "dit_rerun": False,
        },
        "comparison": {
            "scope": admission.get("comparison_scope", "generated T1 pixels only; native clean T0 frames [0,120) excluded"),
            "native_pixel_crop": list(T1_PIXEL_CROP),
            "shape": [1, 3, 120, 288, 512],
            "representation": "display F32 planar RGB [0,1] after pinned BF16 postprocess",
            "atol": ATOL, "rtol": RTOL,
            "max_abs_error": max_abs, "mean_abs_error": mean_abs,
            "any_value_outside_tolerance": bool(mismatch),
            "all_values_finite": not bool(nonfinite or raw_nonfinite),
            "result": "match" if not mismatch and not nonfinite and not raw_nonfinite else "mismatch",
        },
        "attention_adapter": "gathered_spatial_tiles_qk_softmax_pv_fp32_accum",
        "unmodified_eager_flex_attention_parity": False,
        "memory_preflight": memory,
        "runtime": {
            "torch_version": torch.__version__,
            "pytorch_enable_mps_fallback": os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK"),
            "wall_seconds": time.perf_counter() - started_at,
            "intermediate_readbacks": 0,
            "explicit_sync_calls": 0,
            "terminal_transfer_count": 1,
        },
        "upstream_t1_display_path": str(output_pixels),
        "upstream_t1_display_sha256": output_sha,
        "single_terminal_mps_sync_and_pack_transfer": True,
    }
    if "photo_diagnostic" in admission:
        result["photo_diagnostic"] = admission["photo_diagnostic"]
    receipt_path = out_dir / "receipt.json"
    try:
        with receipt_path.open("x", encoding="utf-8") as stream:
            json.dump(result, stream, indent=2, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
    except OSError as error:
        raise ControlError("cannot write fresh decoder receipt") from error
    return result


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--preflight", action="store_true", help="authenticate inputs only (default)")
    mode.add_argument("--execute", action="store_true", help="run the MPS decoder control")
    parser.add_argument("--run-root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--upstream", type=Path, default=DEFAULT_UPSTREAM)
    parser.add_argument("--weight-cache-manifest", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--stem-weights", type=Path, default=DEFAULT_STEM)
    parser.add_argument("--head-weights", type=Path, default=DEFAULT_HEAD)
    parser.add_argument("--input-latents", type=Path,
                        help="optional [1,32,60,9,16] F32 latent capture")
    parser.add_argument("--input-latents-sha256",
                        help="required digest; input must byte-match the native bank prefix")
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        admission = preflight(args)
        if not args.execute:
            print(json.dumps({
                "schema": SCHEMA, "mode": "preflight", "admission": "passed",
                "torch_version": admission["torch_version"],
                "native_receipt_sha256": admission["receipt_sha256"],
                "native_latent_bank_sha256": admission["bank_sha256"],
                "native_selected_latent_sha256": admission["prefix_sha256"],
                "upstream_input_latents_sha256": admission["latent_sha256"] if admission["custom_latents"] else None,
                "continuation_manifest_sha256": admission["manifest_sha256"],
                "continuation_payload_sha256": admission["payload_sha256"],
                "decoder_noise_sha256": admission["noise_record"]["sha256"],
                "decoder_noise_source_shape": list(NOISE_SHAPE),
                "decoder_noise_selected_axis2": [0, 240],
                "decoder_rope_cos_sha256": ROPE_COS_SHA256,
                "decoder_rope_sin_sha256": ROPE_SIN_SHA256,
                "decoder_cache_manifest_sha256": admission["cache_manifest_sha256"],
                "decoder_cache_payload_sha256": admission["cache_payload_sha256"],
                "latent_shape": list(LATENT_SHAPE), "pixel_shape": list(PIXELS_SHAPE),
                "window_starts": list(WINDOW_STARTS), "t1_pixel_crop": list(T1_PIXEL_CROP),
                "output_dir": str(admission["output_dir"]),
                "source_metadata": admission["source_metadata"],
                "encoder_and_dit_rerun": False,
            }, indent=2))
            return 0
        result = execute(admission)
    except (ControlError, OSError, RuntimeError, ValueError, KeyError, TypeError) as error:
        print(f"portrait PixelUMM decoder control failed: {error}", file=sys.stderr)
        return 2
    print(json.dumps({
        "output": result["upstream_t1_display_path"],
        "result": result["comparison"]["result"],
        "max_abs_error": result["comparison"]["max_abs_error"],
        "mean_abs_error": result["comparison"]["mean_abs_error"],
        "memory_preflight": result["memory_preflight"],
    }, indent=2))
    return 0 if result["comparison"]["result"] == "match" else 1


if __name__ == "__main__":
    raise SystemExit(main())
