#!/usr/bin/env python3
"""Admission and pinned-source K/V oracle for native T0 block-0 inputs."""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile
from typing import Any


SCHEMA = "cogni_ml.hy_nomas_reference_cache_inputs/v1"
SCHEMA_VERSION = 1
UPSTREAM_REVISION = "a31c0fe0bcc493be188fd37ddba4ce2f674becb4"
TIMELINE_UPSTREAM_REVISION = "911cea6458a89c2663513ff7099561ec190dd491"
SCOPE = (
    "native T0 timeline/noise/patch/refiner inputs for a standalone block-0 cache writer; "
    "not a complete upstream T0 handoff"
)
PREPROCESSING = {
    "producer": "native",
    "stages": [
        "timeline_reference_scatter", "timeline_cache_noise", "patch_projection",
        "text_refiner",
    ],
    "full_upstream_t0_preprocessing_parity": False,
    "timeline_upstream_revision": TIMELINE_UPSTREAM_REVISION,
}
CACHE_WRITER = {
    "intent": "single_reference_initialization_forward",
    "technical_label": "T0",
    "segment_index": 0,
    "denoising_steps": 0,
    "sigma": 0.206897,
    "timestep": 206.897,
}

# (name, dtype, shape). JSON object insertion order is part of this fixture contract.
_FIXED_TENSORS = (
    ("reference.latent_bf16", "BF16", (1, 32, 30, 9, 16)),
    ("cache.noisy_latent_f32", "F32", (1, 32, 30, 9, 16)),
    ("block.img_f32", "F32", (1, 4320, 2048)),
    ("block.vec_f32", "F32", (1, 2048)),
    ("block.vec_txt_f32", "F32", (1, 2048)),
    ("block.rope_cos_f32", "F32", (4320, 128)),
    ("block.rope_sin_f32", "F32", (4320, 128)),
    ("cache.timestep_f32", "F32", (1,)),
)
MANIFEST_KEYS = {
    "schema", "schema_version", "upstream_revision", "scope", "preprocessing",
    "reference_latent_artifact_sha256", "timeline_manifest_sha256", "cache_writer",
    "valid_text_tokens", "payload", "payload_bytes", "payload_sha256", "tensors",
}
TENSOR_RECORD_KEYS = {"dtype", "shape", "offset", "nbytes", "sha256"}
_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_HEADER_LIMIT = 16 * 1024 * 1024
BLOCK_WEIGHTS_DEFAULT = Path(
    "/opt/mldata_ssd/cogni-ml/hy-nomas-q4-dit-20261025/weights/"
    "q4-cache-v2/block-0-bf16.safetensors"
)
BLOCK_WEIGHTS_SHA256 = "01be7388fa4267f1f566c3ced405418f94d0ed2b4ee969cc880fb2518ab7c87b"
UPSTREAM_ROOT_DEFAULT = Path(
    "/opt/mldata_ssd/cogni-ml/hy-nomas-reference-handoff-20261101.pDnd7TV1/upstream-a31"
)
INFERENCE_DTYPE_SOURCE_SHA256 = {
    "configs/experiment/inference.yaml": "d4db57b63597f98d4bddddd98355198e4e8c25da366d2e27bc7e74a79d26ec67",
    "hymm/pipeline/base.py": "77a039514ff494cb496a194c60ca3270f09c15b194bda25448a036a6fcc79582",
    "hymm/pipeline/causal_ar.py": "fc5d8eb566fabf9d9fc75c2382a52358d170e11f0c0511e13072a860e209113c",
}
ORACLE_SCHEMA = "cogni_ml.hy_nomas_reference_block0_kv_oracle/v1"
ORACLE_KEYS = (
    "img.q_projection", "img.k_projection", "img.v_projection",
    "img.q_qknorm", "img.k_qknorm", "img.q_rope", "img.k_rope",
    "txt.q_projection", "txt.k_projection", "txt.v_projection",
    "txt.q_qknorm", "txt.k_qknorm", "cache.k_bf16", "cache.v_bf16",
)
PREQKV_SCHEMA = "cogni_ml.hy_nomas_reference_block0_preqkv_oracle/v1"
PREQKV_SCOPE = (
    "diagnostic eager boundaries from the same pinned block-0 forward and native "
    "preblock inputs as the linked K/V oracle; not a complete upstream T0 handoff"
)
PREQKV_NAMES = (
    "img.mod_silu", "img.mod_params", "img.norm1", "img.qkv_input",
    "txt.mod_silu", "txt.mod_params", "txt.norm1", "txt.qkv_input",
)
PREQKV_BF16_BOUNDARIES = frozenset({
    "img.mod_silu", "img.mod_params", "txt.mod_silu", "txt.mod_params",
})
PREQKV_MIXED_ARTIFACT_DTYPE = "MIXED"


class ReferenceCacheError(ValueError):
    """A cache manifest or payload failed CPU admission."""


def _expected_tensors(valid_text_tokens: int) -> tuple[tuple[str, str, tuple[int, ...]], ...]:
    if type(valid_text_tokens) is not int or not 1 <= valid_text_tokens <= 1280:
        raise ReferenceCacheError("valid_text_tokens must be an integer in [1, 1280]")
    return (
        *_FIXED_TENSORS[:3],
        ("block.txt_valid_f32", "F32", (1, valid_text_tokens, 2048)),
        *_FIXED_TENSORS[3:],
    )


def _matches_exact_json(value: object, expected: object) -> bool:
    """Compare values without JSON's bool/int equality loophole."""
    if type(value) is not type(expected):
        return False
    if type(expected) is dict:
        return (set(value) == set(expected) and all(  # type: ignore[arg-type]
            _matches_exact_json(value[key], expected[key])  # type: ignore[index]
            for key in expected
        ))
    if type(expected) is list:
        return (len(value) == len(expected) and all(  # type: ignore[arg-type]
            _matches_exact_json(actual, wanted)
            for actual, wanted in zip(value, expected)  # type: ignore[arg-type]
        ))
    return value == expected


def _is_digest(value: object) -> bool:
    return isinstance(value, str) and _SHA256.fullmatch(value) is not None


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ReferenceCacheError(f"duplicate JSON field: {key}")
        result[key] = value
    return result


def _read_json(path: Path, *, limit: int = 1024 * 1024) -> tuple[dict, bytes]:
    if path.is_symlink() or not path.is_file():
        raise ReferenceCacheError("cache manifest must be a regular non-symlink file")
    raw = path.read_bytes()
    if len(raw) > limit:
        raise ReferenceCacheError("cache manifest exceeds the metadata size limit")
    try:
        value = json.loads(raw.decode("utf-8"), object_pairs_hook=_unique_object)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ReferenceCacheError("cache manifest JSON is invalid") from error
    if type(value) is not dict:
        raise ReferenceCacheError("cache manifest must be a JSON object")
    return value, raw


def _canonical_file(value: str | Path, what: str) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute() or path.is_symlink() or not path.is_file():
        raise ReferenceCacheError(f"{what} must be an absolute regular non-symlink file")
    try:
        return path.resolve(strict=True)
    except (OSError, RuntimeError) as error:
        raise ReferenceCacheError(f"cannot resolve {what}") from error


def _hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _hash_range(path: Path, offset: int, size: int) -> str:
    digest = hashlib.sha256()
    remaining = size
    with path.open("rb") as stream:
        stream.seek(offset)
        while remaining:
            chunk = stream.read(min(1024 * 1024, remaining))
            if not chunk:
                raise ReferenceCacheError("cache payload tensor range is truncated")
            digest.update(chunk)
            remaining -= len(chunk)
    return digest.hexdigest()


def _safetensors_latent(path: str | Path, expected_artifact_sha256: str) -> dict[str, object]:
    """Authenticate the source artifact and locate its exact BF16 latent range."""
    source = _canonical_file(path, "reference latent artifact")
    if _hash_file(source) != expected_artifact_sha256:
        raise ReferenceCacheError("reference latent artifact SHA-256 mismatch")
    size = source.stat().st_size
    with source.open("rb") as stream:
        prefix = stream.read(8)
        if len(prefix) != 8:
            raise ReferenceCacheError("reference latent safetensors header is truncated")
        header_size = int.from_bytes(prefix, "little", signed=False)
        if header_size < 2 or header_size > _HEADER_LIMIT or 8 + header_size > size:
            raise ReferenceCacheError("reference latent safetensors header size is invalid")
        raw_header = stream.read(header_size)
    try:
        header = json.loads(raw_header.decode("utf-8"), object_pairs_hook=_unique_object)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ReferenceCacheError("reference latent safetensors header is invalid") from error
    record = header.get("latent") if type(header) is dict else None
    offsets = record.get("data_offsets") if type(record) is dict else None
    shape = [1, 32, 30, 9, 16]
    if (type(record) is not dict or record.get("dtype") != "BF16"
            or record.get("shape") != shape
            or type(offsets) is not list or len(offsets) != 2
            or any(type(value) is not int for value in offsets)
            or offsets[0] < 0 or offsets[1] - offsets[0] != 276_480
            or offsets[1] > size - 8 - header_size):
        raise ReferenceCacheError("reference latent safetensors key, shape, dtype, or range is invalid")
    data_offset = 8 + header_size + offsets[0]
    return {
        "path": str(source),
        "artifact_sha256": expected_artifact_sha256,
        "header_sha256": hashlib.sha256(raw_header).hexdigest(),
        "data_offset": data_offset,
        "nbytes": 276_480,
        "tensor_sha256": _hash_range(source, data_offset, 276_480),
    }


def _compare_ranges(left_path: Path, left_offset: int, right_path: Path,
                    right_offset: int, size: int) -> None:
    with left_path.open("rb") as left, right_path.open("rb") as right:
        left.seek(left_offset)
        right.seek(right_offset)
        remaining = size
        while remaining:
            amount = min(1024 * 1024, remaining)
            left_chunk = left.read(amount)
            right_chunk = right.read(amount)
            if len(left_chunk) != amount or len(right_chunk) != amount:
                raise ReferenceCacheError("reference latent or cache payload range is truncated")
            if left_chunk != right_chunk:
                raise ReferenceCacheError(
                    "reference.latent_bf16 payload differs from the authenticated source latent"
                )
            remaining -= amount


def _expected_nbytes(dtype: str, shape: tuple[int, ...]) -> int:
    element_bytes = 2 if dtype == "BF16" else 4
    count = 1
    for dimension in shape:
        count *= dimension
    return count * element_bytes


def validate_reference_cache_manifest(
    manifest_path: str | Path,
    *,
    reference_latent_path: str | Path,
    expected_reference_latent_artifact_sha256: str,
    expected_timeline_manifest_sha256: str,
    expected_manifest_sha256: str | None = None,
) -> dict[str, object]:
    """Validate exact metadata, ordered tensor ranges, and authenticated source bytes.

    The source latent and timeline hashes are caller authorizations. The manifest
    hash is returned and may also be pinned by the caller with
    ``expected_manifest_sha256``.
    """
    if not _is_digest(expected_reference_latent_artifact_sha256):
        raise ReferenceCacheError("expected reference latent artifact SHA-256 is malformed")
    if not _is_digest(expected_timeline_manifest_sha256):
        raise ReferenceCacheError("expected timeline manifest SHA-256 is malformed")
    if expected_manifest_sha256 is not None and not _is_digest(expected_manifest_sha256):
        raise ReferenceCacheError("expected cache manifest SHA-256 is malformed")

    manifest = _canonical_file(manifest_path, "cache manifest")
    document, raw_manifest = _read_json(manifest)
    manifest_sha256 = hashlib.sha256(raw_manifest).hexdigest()
    if expected_manifest_sha256 is not None and manifest_sha256 != expected_manifest_sha256:
        raise ReferenceCacheError("cache manifest SHA-256 differs from caller authorization")
    if set(document) != MANIFEST_KEYS:
        missing = sorted(MANIFEST_KEYS - set(document))
        extra = sorted(set(document) - MANIFEST_KEYS)
        raise ReferenceCacheError(f"cache manifest fields are invalid; missing={missing}, extra={extra}")

    if (document["schema"] != SCHEMA or type(document["schema_version"]) is not int
            or document["schema_version"] != SCHEMA_VERSION
            or document["upstream_revision"] != UPSTREAM_REVISION):
        raise ReferenceCacheError("cache manifest schema, version, or upstream revision mismatch")
    if document["scope"] != SCOPE:
        raise ReferenceCacheError("cache manifest scope does not identify standalone native T0 inputs")
    if not _matches_exact_json(document["preprocessing"], PREPROCESSING):
        raise ReferenceCacheError("cache manifest preprocessing contract mismatch")
    if (document["reference_latent_artifact_sha256"] != expected_reference_latent_artifact_sha256
            or not _is_digest(document["reference_latent_artifact_sha256"])):
        raise ReferenceCacheError("reference latent artifact identity differs from caller authorization")
    if (document["timeline_manifest_sha256"] != expected_timeline_manifest_sha256
            or not _is_digest(document["timeline_manifest_sha256"])):
        raise ReferenceCacheError("timeline manifest identity differs from caller authorization")
    writer = document["cache_writer"]
    if not _matches_exact_json(writer, CACHE_WRITER):
        raise ReferenceCacheError("cache writer is not the admitted single T0 initialization forward")
    valid_text_tokens = document["valid_text_tokens"]
    expected_tensors = _expected_tensors(valid_text_tokens)
    if type(document["payload_bytes"]) is not int or document["payload_bytes"] != sum(
            _expected_nbytes(dtype, shape) for _, dtype, shape in expected_tensors):
        raise ReferenceCacheError("cache payload byte extent does not match the exact tensor inventory")
    if not _is_digest(document["payload_sha256"]):
        raise ReferenceCacheError("cache payload SHA-256 is malformed")

    expected_payload_name = manifest.with_suffix(".tensors.bin").name
    payload_name = document["payload"]
    if (not isinstance(payload_name, str) or payload_name != expected_payload_name
            or Path(payload_name).name != payload_name):
        raise ReferenceCacheError("cache payload must be the manifest's sibling .tensors.bin file")
    payload = _canonical_file(manifest.parent / payload_name, "cache tensor payload")
    if payload.stat().st_size != document["payload_bytes"]:
        raise ReferenceCacheError("cache tensor payload file has the wrong byte extent")
    if _hash_file(payload) != document["payload_sha256"]:
        raise ReferenceCacheError("cache tensor payload SHA-256 mismatch")

    tensors = document["tensors"]
    expected_names = tuple(name for name, _, _ in expected_tensors)
    if type(tensors) is not dict or tuple(tensors) != expected_names:
        raise ReferenceCacheError("cache tensor inventory names or insertion order mismatch")
    offset = 0
    for name, dtype, shape in expected_tensors:
        record = tensors[name]
        if type(record) is not dict or set(record) != TENSOR_RECORD_KEYS:
            raise ReferenceCacheError(f"cache tensor record fields are invalid: {name}")
        nbytes = _expected_nbytes(dtype, shape)
        if (type(record["dtype"]) is not str or record["dtype"] != dtype
                or type(record["shape"]) is not list or record["shape"] != list(shape)
                or any(type(value) is not int for value in record["shape"])
                or type(record["offset"]) is not int or record["offset"] != offset
                or type(record["nbytes"]) is not int or record["nbytes"] != nbytes
                or not _is_digest(record["sha256"])):
            raise ReferenceCacheError(f"cache tensor schema, offset, extent, or digest is invalid: {name}")
        if _hash_range(payload, offset, nbytes) != record["sha256"]:
            raise ReferenceCacheError(f"cache tensor payload range SHA-256 mismatch: {name}")
        offset += nbytes
    if offset != payload.stat().st_size:
        raise ReferenceCacheError("cache tensor ranges do not cover the complete payload")

    source_latent = _safetensors_latent(
        reference_latent_path, expected_reference_latent_artifact_sha256
    )
    latent_record = tensors["reference.latent_bf16"]
    if latent_record["sha256"] != source_latent["tensor_sha256"]:
        raise ReferenceCacheError("cache latent record digest differs from the authenticated source latent")
    _compare_ranges(
        Path(source_latent["path"]), source_latent["data_offset"],
        payload, latent_record["offset"], latent_record["nbytes"],
    )

    return {
        "schema": SCHEMA,
        "manifest_path": str(manifest),
        "manifest_sha256": manifest_sha256,
        "payload_path": str(payload),
        "payload_sha256": document["payload_sha256"],
        "payload_bytes": document["payload_bytes"],
        "upstream_revision": UPSTREAM_REVISION,
        "scope": SCOPE,
        "preprocessing": PREPROCESSING,
        "cache_writer": CACHE_WRITER,
        "reference_latent_artifact_sha256": expected_reference_latent_artifact_sha256,
        "reference_latent_tensor_sha256": latent_record["sha256"],
        "reference_latent_source": source_latent,
        "timeline_manifest_sha256": expected_timeline_manifest_sha256,
        "valid_text_tokens": valid_text_tokens,
        "tensor_order": list(expected_names),
        "tensor_count": len(expected_names),
    }


class _CacheCaptureComplete(Exception):
    """Private control flow: Q/K/V cache bytes captured before attention."""


def _canonicalize_preqkv_capture(capture: dict[str, Any]) -> None:
    """Validate complete inventory and normalize actual hook-call order."""
    if len(capture) != len(PREQKV_NAMES) or set(capture) != set(PREQKV_NAMES):
        raise ReferenceCacheError(
            "source pre-QKV trace inventory mismatch; "
            f"missing={sorted(set(PREQKV_NAMES) - set(capture))}, "
            f"extra={sorted(set(capture) - set(PREQKV_NAMES))}"
        )
    ordered = {name: capture[name] for name in PREQKV_NAMES}
    capture.clear()
    capture.update(ordered)


def _load_support_oracle() -> Any:
    path = Path(__file__).with_name("hy_nomas_reference.py").resolve(strict=True)
    spec = importlib.util.spec_from_file_location("_hy_nomas_reference_existing", path)
    if spec is None or spec.loader is None:
        raise ReferenceCacheError("cannot load the pinned single-block source helper")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _verify_upstream_root(root_value: str | Path, support: Any) -> tuple[Path, dict[str, str], str]:
    root = Path(root_value).expanduser()
    if not root.is_absolute() or root.is_symlink() or not root.is_dir():
        raise ReferenceCacheError("upstream root must be an absolute directory without a final symlink")
    root = root.resolve(strict=True)
    try:
        revision = subprocess.check_output(
            ["git", "-C", str(root), "rev-parse", "HEAD"],
            text=True, stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError) as error:
        raise ReferenceCacheError("cannot identify the upstream source revision") from error
    if revision != UPSTREAM_REVISION:
        raise ReferenceCacheError(
            f"upstream source revision mismatch: {revision} != {UPSTREAM_REVISION}"
        )
    hashes: dict[str, str] = {}
    for relative, expected in support.PINNED_SOURCE_SHA256.items():
        source = root / relative
        if not source.is_file() or source.is_symlink():
            raise ReferenceCacheError(f"pinned upstream source is missing or unsafe: {relative}")
        actual = _hash_file(source)
        if actual != expected:
            raise ReferenceCacheError(f"pinned upstream source hash mismatch: {relative}")
        hashes[relative] = actual
    for relative, expected in INFERENCE_DTYPE_SOURCE_SHA256.items():
        source = root / relative
        if not source.is_file() or source.is_symlink():
            raise ReferenceCacheError(f"inference dtype source is missing or unsafe: {relative}")
        actual = _hash_file(source)
        if actual != expected:
            raise ReferenceCacheError(f"inference dtype source hash mismatch: {relative}")
        hashes[relative] = actual
    return root, hashes, revision


def _read_bundle_tensors(
    validated: dict[str, object], *, torch: Any, device: Any, input_dtype: Any,
) -> tuple[dict[str, Any], dict[str, dict[str, object]]]:
    manifest_path = Path(str(validated["manifest_path"]))
    document, raw_manifest = _read_json(manifest_path)
    if hashlib.sha256(raw_manifest).hexdigest() != validated["manifest_sha256"]:
        raise ReferenceCacheError("cache input manifest changed after CPU admission")
    payload = Path(str(validated["payload_path"]))
    tensors: dict[str, Any] = {}
    records = document["tensors"]
    for name in (
        "block.img_f32", "block.txt_valid_f32", "block.vec_f32",
        "block.vec_txt_f32", "block.rope_cos_f32", "block.rope_sin_f32",
    ):
        record = records[name]
        dtype_name = record["dtype"]
        if dtype_name != "F32":
            raise ReferenceCacheError(f"source block input must be F32: {name}")
        with payload.open("rb") as stream:
            stream.seek(record["offset"])
            raw = stream.read(record["nbytes"])
        if len(raw) != record["nbytes"]:
            raise ReferenceCacheError(f"source block input payload is truncated: {name}")
        host = torch.frombuffer(bytearray(raw), dtype=torch.float32).reshape(record["shape"])
        if not bool(torch.isfinite(host).all().item()):
            raise ReferenceCacheError(f"source block input contains non-finite values: {name}")
        if name in ("block.rope_cos_f32", "block.rope_sin_f32"):
            tensors[name] = host.to(device=device, dtype=torch.float32)
        else:
            tensors[name] = host.to(device=device, dtype=input_dtype)
    if _hash_file(payload) != validated["payload_sha256"]:
        raise ReferenceCacheError("cache input payload changed while loading source tensors")
    if _hash_file(manifest_path) != validated["manifest_sha256"]:
        raise ReferenceCacheError("cache input manifest changed while loading source tensors")
    return tensors, records


def _load_source_weights(
    torch: Any, support: Any, weights_value: str | Path, expected_sha256: str,
    *, parameter_dtype: Any,
) -> tuple[Path, dict[str, Any], str]:
    path = Path(weights_value).expanduser()
    if not path.is_absolute() or path.is_symlink() or not path.is_file():
        raise ReferenceCacheError("block weights must be an absolute regular non-symlink file")
    path = path.resolve(strict=True)
    if path.suffix.lower() != ".safetensors" or not _is_digest(expected_sha256):
        raise ReferenceCacheError("block weights require safetensors and a lowercase SHA-256")
    if _hash_file(path) != expected_sha256:
        raise ReferenceCacheError("block weight source SHA-256 mismatch")
    config = dict(support.REAL_WIDTH)
    state, source_dtype = support._load_checkpoint_state(torch, path, config)
    if source_dtype != "BF16" or len(state) != 32:
        raise ReferenceCacheError("cache oracle weights must be the exact 32-tensor BF16 block-0 source")
    if _hash_file(path) != expected_sha256:
        raise ReferenceCacheError("block weight source changed while loading tensors")
    if parameter_dtype == torch.float32:
        # Localization control only: widen the authenticated BF16 values
        # without restoring the original F32 checkpoint values.
        state = {name: tensor.to(dtype=torch.float32) for name, tensor in state.items()}
    elif parameter_dtype != torch.bfloat16:
        raise ReferenceCacheError("cache oracle parameters must be BF16 or F32 control values")
    return path, state, source_dtype


def _capture_block0_cache(
    *, torch: Any,
    block: Any,
    block_class_globals: dict[str, Any],
    inputs: dict[str, Any],
    cache_buffers: dict[str, Any],
    mode: str,
    preqkv_capture: dict[str, Any] | None = None,
) -> dict[str, Any]:
    capture: dict[str, Any] = {}
    hooks: list[Any] = []
    heads, head_dim = 16, 128

    def keep(name: str, tensor: Any) -> None:
        if name in capture:
            raise ReferenceCacheError(f"duplicate source trace boundary: {name}")
        capture[name] = tensor.detach()

    for stream in ("img", "txt"):
        if preqkv_capture is not None:
            modulation = getattr(block, f"{stream}_mod")
            norm1 = getattr(block, f"{stream}_norm1")
            q_projection = getattr(block, f"{stream}_attn_q")

            def preqkv_keep(name: str, tensor: Any) -> None:
                if name in preqkv_capture:
                    raise ReferenceCacheError(f"duplicate source pre-QKV boundary: {name}")
                preqkv_capture[name] = tensor.detach()

            def silu_hook(_module: Any, _args: tuple[Any, ...], output: Any,
                          key: str = f"{stream}.mod_silu") -> None:
                preqkv_keep(key, output)

            def modulation_hook(_module: Any, _args: tuple[Any, ...], output: Any,
                                key: str = f"{stream}.mod_params") -> None:
                preqkv_keep(key, output)

            def norm1_hook(_module: Any, _args: tuple[Any, ...], output: Any,
                           key: str = f"{stream}.norm1") -> None:
                preqkv_keep(key, output)

            def qkv_input_hook(_module: Any, args: tuple[Any, ...],
                               key: str = f"{stream}.qkv_input") -> None:
                if len(args) != 1:
                    raise ReferenceCacheError("source q projection received unexpected positional inputs")
                preqkv_keep(key, args[0])

            hooks.append(modulation.act.register_forward_hook(silu_hook))
            hooks.append(modulation.register_forward_hook(modulation_hook))
            hooks.append(norm1.register_forward_hook(norm1_hook))
            hooks.append(q_projection.register_forward_pre_hook(qkv_input_hook))

        for projection in ("q", "k", "v"):
            module = getattr(block, f"{stream}_attn_{projection}")

            def projection_hook(_module: Any, _args: tuple[Any, ...], output: Any,
                                key: str = f"{stream}.{projection}_projection") -> None:
                keep(key, output.reshape(output.shape[0], output.shape[1], heads, head_dim))

            hooks.append(module.register_forward_hook(projection_hook))
        for projection, norm_name in (("q", "q_norm"), ("k", "k_norm")):
            module = getattr(block, f"{stream}_attn_{norm_name}")

            def norm_hook(_module: Any, _args: tuple[Any, ...], output: Any,
                          key: str = f"{stream}.{projection}_qknorm") -> None:
                keep(key, output)

            hooks.append(module.register_forward_hook(norm_hook))

    original_rope = block_class_globals["apply_rotary_emb"]
    original_cache_attention = block_class_globals.get("causal_attention_with_cache_mp")

    def rope_capture(q: Any, k: Any, freqs: Any, head_first: bool = False) -> tuple[Any, Any]:
        q_rot, k_rot = original_rope(q, k, freqs, head_first=head_first)
        keep("img.q_rope", q_rot)
        keep("img.k_rope", k_rot)
        return q_rot, k_rot

    # The direct block source has one shared RoPE global. Replace it only for
    # this call so the captured tensors are exactly the values consumed below.
    block_class_globals["apply_rotary_emb"] = rope_capture

    def cache_boundary(
        q: tuple[Any, Any], k: tuple[Any, Any], v: tuple[Any, Any], *,
        kv_cache: dict[str, Any], cache_valid_len: int = 0, update_cache: bool = False,
        attn_mode: str | None = None, text_mask: Any = None, cache_txt: bool = True,
        cache_mask: Any = None, **_kwargs: Any,
    ) -> Any:
        if cache_valid_len != 0 or not update_cache or not cache_txt or cache_mask is not None:
            raise ReferenceCacheError("source oracle admits only the first unmasked T0 cache append")
        if kv_cache is not cache_buffers:
            raise ReferenceCacheError("source cache adapter did not receive the prepared K/V buffers")
        if text_mask is not None or attn_mode != "flash3":
            raise ReferenceCacheError("source oracle cache adapter received unexpected mask or attention mode")
        img_q, txt_q = q
        img_k, txt_k = k
        img_v, txt_v = v
        for name, tensor in (
            ("img.q_rope", img_q), ("img.k_rope", img_k),
            ("txt.q_qknorm", txt_q), ("txt.k_qknorm", txt_k),
        ):
            if name not in capture:
                keep(name, tensor)
        kernel_dtype = kv_cache["k"].dtype
        if kernel_dtype != torch.bfloat16 or kv_cache["v"].dtype != kernel_dtype:
            raise ReferenceCacheError("source cache boundary requires BF16 K/V storage")
        current_k = torch.cat((img_k.to(kernel_dtype), txt_k.to(kernel_dtype)), dim=1)
        current_v = torch.cat((img_v.to(kernel_dtype), txt_v.to(kernel_dtype)), dim=1)
        token_count = current_k.shape[1]
        if tuple(current_k.shape) != tuple(cache_buffers["k"].shape) or tuple(current_v.shape) != tuple(cache_buffers["v"].shape):
            raise ReferenceCacheError("source cache adapter extent differs from its prepared buffers")
        kv_cache["k"][:, :token_count].copy_(current_k.detach())
        kv_cache["v"][:, :token_count].copy_(current_v.detach())
        keep("cache.k_bf16", kv_cache["k"][:, :token_count])
        keep("cache.v_bf16", kv_cache["v"][:, :token_count])
        raise _CacheCaptureComplete

    block_class_globals["causal_attention_with_cache_mp"] = cache_boundary
    block.eval()
    try:
        with torch.inference_mode(), torch.autocast(device_type="mps", dtype=torch.bfloat16,
                                                    enabled=mode == "bf16"):
            block.forward_with_cache(
                img=inputs["block.img_f32"],
                txt=inputs["block.txt_valid_f32"],
                vec=inputs["block.vec_f32"],
                vec_txt=inputs["block.vec_txt_f32"],
                freqs_cis=(inputs["block.rope_cos_f32"], inputs["block.rope_sin_f32"]),
                text_mask=None,
                kv_cache=cache_buffers,
                cache_valid_len=0,
                update_cache=True,
                attn_mode="flash3",
                cache_txt=True,
                cache_mask=None,
            )
    except _CacheCaptureComplete:
        pass
    finally:
        block_class_globals["apply_rotary_emb"] = original_rope
        if original_cache_attention is None:
            block_class_globals.pop("causal_attention_with_cache_mp", None)
        else:
            block_class_globals["causal_attention_with_cache_mp"] = original_cache_attention
        for hook in hooks:
            hook.remove()
    if tuple(capture) != ORACLE_KEYS:
        raise ReferenceCacheError(
            "source trace inventory mismatch; "
            f"missing={sorted(set(ORACLE_KEYS) - set(capture))}, "
            f"extra={sorted(set(capture) - set(ORACLE_KEYS))}"
        )
    if preqkv_capture is not None:
        _canonicalize_preqkv_capture(preqkv_capture)
    for name, tensor in capture.items():
        if tensor.device.type != "mps" or not bool(torch.isfinite(tensor).all().item()):
            raise ReferenceCacheError(f"source trace is non-MPS or non-finite: {name}")
    if preqkv_capture is not None:
        for name, tensor in preqkv_capture.items():
            if tensor.device.type != "mps" or not bool(torch.isfinite(tensor).all().item()):
                raise ReferenceCacheError(f"source pre-QKV trace is non-MPS or non-finite: {name}")
    return capture


def _save_oracle_artifact(
    capture: dict[str, Any], *, output_safetensors: str | Path,
    receipt_path: str | Path, torch: Any, metadata: dict[str, Any],
) -> dict[str, Any]:
    from safetensors.torch import save_file

    output = Path(output_safetensors).expanduser()
    receipt = Path(receipt_path).expanduser()
    for path, suffix in ((output, ".safetensors"), (receipt, ".json")):
        if not path.is_absolute() or path.suffix.lower() != suffix:
            raise ReferenceCacheError(f"oracle output must be an absolute {suffix} file")
        if path.is_symlink() or path.exists():
            raise ReferenceCacheError(f"refusing to replace source oracle output: {path}")
        if not path.parent.is_dir():
            raise ReferenceCacheError(f"oracle output parent does not exist: {path.parent}")
    if output.parent.resolve() != receipt.parent.resolve():
        raise ReferenceCacheError("oracle tensor and receipt files must share one parent directory")

    stage = Path(tempfile.mkdtemp(prefix=f".{output.stem}.stage-", dir=output.parent))
    staged_tensors = stage / output.name
    staged_receipt = stage / receipt.name
    cpu_tensors = {
        name: value.detach().contiguous().to(device="cpu")
        for name, value in capture.items()
    }
    try:
        save_file(cpu_tensors, str(staged_tensors), metadata={
            "format": ORACLE_SCHEMA,
            "upstream_revision": UPSTREAM_REVISION,
            "weight_sha256": str(metadata["weight_source"]["sha256"]),
            "cache_dtype": "BF16",
        })
        tensor_records: dict[str, dict[str, object]] = {}
        for name in ORACLE_KEYS:
            tensor = cpu_tensors[name]
            # Hash raw little-endian tensor bytes without creating a second array.
            raw = tensor.contiguous().view(torch.uint8).numpy().tobytes()
            tensor_records[name] = {
                "dtype": "BF16" if tensor.dtype == torch.bfloat16 else "F32",
                "shape": list(tensor.shape),
                "nbytes": len(raw),
                "sha256": hashlib.sha256(raw).hexdigest(),
            }
        receipt_doc = {
            "schema": ORACLE_SCHEMA,
            "operation": "upstream_block0_first_chunk_kv_cache_append",
            **metadata,
            "tensor_artifact": {
                "path": output.name,
                "sha256": _hash_file(staged_tensors),
                "tensor_count": len(tensor_records),
                "tensors": tensor_records,
            },
        }
        with staged_receipt.open("x", encoding="utf-8") as stream:
            json.dump(receipt_doc, stream, sort_keys=True, indent=2)
            stream.write("\n")
        os.link(staged_tensors, output)
        published_output = output.stat(follow_symlinks=False)
        try:
            os.link(staged_receipt, receipt)
        except OSError:
            try:
                current_output = output.stat(follow_symlinks=False)
            except FileNotFoundError:
                current_output = None
            if (current_output is not None
                    and (current_output.st_dev, current_output.st_ino)
                    == (published_output.st_dev, published_output.st_ino)):
                output.unlink()
            raise
    finally:
        for child in stage.iterdir():
            child.unlink(missing_ok=True)
        stage.rmdir()
    receipt_doc["tensor_artifact"]["path"] = str(output)
    receipt_doc["receipt_path"] = str(receipt)
    return receipt_doc


def _validate_new_artifact_paths(
    tensor_path_value: str | Path, receipt_path_value: str | Path, *, label: str,
) -> tuple[Path, Path]:
    paths: list[Path] = []
    for value, suffix in ((tensor_path_value, ".safetensors"), (receipt_path_value, ".json")):
        path = Path(value).expanduser()
        if not path.is_absolute() or path.suffix.lower() != suffix:
            raise ReferenceCacheError(f"{label} output must be an absolute {suffix} file")
        if path.is_symlink() or path.exists():
            raise ReferenceCacheError(f"refusing to replace {label} output: {path}")
        if not path.parent.is_dir():
            raise ReferenceCacheError(f"{label} output parent does not exist: {path.parent}")
        paths.append(path.parent.resolve(strict=True) / path.name)
    if paths[0].parent != paths[1].parent:
        raise ReferenceCacheError(f"{label} tensor artifact and receipt must share one parent directory")
    return paths[0], paths[1]


def _save_preqkv_artifact(
    capture: dict[str, Any], *, output_safetensors: str | Path,
    receipt_path: str | Path, torch: Any, precision: str,
    validation: dict[str, Any], weights_sha256: str,
    parent_source_receipt_path: str | Path,
    parent_source_artifact_path: str | Path,
) -> dict[str, Any]:
    """Save opt-in eager boundaries tied to the unchanged main source oracle."""
    from safetensors.torch import save_file

    output, receipt = _validate_new_artifact_paths(
        output_safetensors, receipt_path, label="source pre-QKV companion"
    )
    parent_receipt = _canonical_file(parent_source_receipt_path, "parent source oracle receipt")
    parent_artifact = _canonical_file(parent_source_artifact_path, "parent source oracle artifact")
    parent_doc, _parent_raw = _read_json(parent_receipt)
    parent_tensor_record = parent_doc.get("tensor_artifact")
    if (parent_doc.get("schema") != ORACLE_SCHEMA
            or parent_doc.get("operation") != "upstream_block0_first_chunk_kv_cache_append"
            or type(parent_tensor_record) is not dict
            or parent_tensor_record.get("path") != parent_artifact.name
            or parent_tensor_record.get("sha256") != _hash_file(parent_artifact)):
        raise ReferenceCacheError("pre-QKV companion parent source receipt/artifact identity mismatch")
    parent_input = parent_doc.get("input_bundle")
    parent_weights = parent_doc.get("weight_source")
    parent_precision = parent_doc.get("precision")
    expected_precision = "BF16BoundaryControl" if precision == "bf16" else "F32Control"
    if (type(parent_input) is not dict or type(parent_weights) is not dict
            or type(parent_precision) is not dict
            or parent_input.get("manifest_sha256") != validation["manifest_sha256"]
            or parent_input.get("payload_sha256") != validation["payload_sha256"]
            or parent_input.get("reference_latent_artifact_sha256")
            != validation["reference_latent_artifact_sha256"]
            or parent_input.get("timeline_manifest_sha256") != validation["timeline_manifest_sha256"]
            or parent_input.get("valid_text_tokens") != validation["valid_text_tokens"]
            or parent_weights.get("sha256") != weights_sha256
            or parent_precision.get("mode") != precision):
        raise ReferenceCacheError("pre-QKV companion parent receipt binds different inputs, weights, or precision")
    parent_receipt_sha256 = _hash_file(parent_receipt)
    parent_artifact_sha256 = _hash_file(parent_artifact)

    stage = Path(tempfile.mkdtemp(prefix=f".{output.stem}.stage-", dir=output.parent))
    staged_tensors = stage / output.name
    staged_receipt = stage / receipt.name
    cpu_tensors = {
        name: capture[name].detach().contiguous().to(device="cpu")
        for name in PREQKV_NAMES
    }
    try:
        disk_dtypes = {"BF16" if value.dtype == torch.bfloat16 else "F32"
                       for value in cpu_tensors.values()}
        if not disk_dtypes or not disk_dtypes <= {"BF16", "F32"}:
            raise ReferenceCacheError("pre-QKV capture has an unsupported tensor dtype")
        disk_dtype = (
            next(iter(disk_dtypes)) if len(disk_dtypes) == 1
            else PREQKV_MIXED_ARTIFACT_DTYPE
        )
        save_file(cpu_tensors, str(staged_tensors), metadata={
            "format": PREQKV_SCHEMA,
            "upstream_revision": UPSTREAM_REVISION,
            "precision_control": expected_precision,
            "parent_source_receipt_sha256": parent_receipt_sha256,
            "parent_source_artifact_sha256": parent_artifact_sha256,
        })
        tensor_records: list[dict[str, Any]] = []
        for name in PREQKV_NAMES:
            tensor = cpu_tensors[name]
            raw = tensor.contiguous().view(torch.uint8).numpy().tobytes()
            tensor_records.append({
                "name": name,
                "dtype": "BF16" if tensor.dtype == torch.bfloat16 else "F32",
                "shape": list(tensor.shape),
                "nbytes": len(raw),
                "sha256": hashlib.sha256(raw).hexdigest(),
            })
        document = {
            "schema": PREQKV_SCHEMA,
            "schema_version": 1,
            "upstream_revision": UPSTREAM_REVISION,
            "scope": PREQKV_SCOPE,
            "precision_control": expected_precision,
            "input_bundle_manifest_sha256": validation["manifest_sha256"],
            "input_bundle_payload_sha256": validation["payload_sha256"],
            "reference_latent_artifact_sha256": validation["reference_latent_artifact_sha256"],
            "timeline_manifest_sha256": validation["timeline_manifest_sha256"],
            "valid_text_tokens": validation["valid_text_tokens"],
            "weights_sha256": weights_sha256,
            "parent_source_receipt_sha256": parent_receipt_sha256,
            "parent_source_artifact_sha256": parent_artifact_sha256,
            "tensor_artifact": {
                "basename": output.name,
                "sha256": _hash_file(staged_tensors),
                "bytes": staged_tensors.stat().st_size,
                "dtype": disk_dtype,
                "tensors": tensor_records,
            },
        }
        with staged_receipt.open("x", encoding="utf-8") as stream:
            json.dump(document, stream, sort_keys=True, indent=2)
            stream.write("\n")
        if (_hash_file(parent_receipt) != parent_receipt_sha256
                or _hash_file(parent_artifact) != parent_artifact_sha256):
            raise ReferenceCacheError("pre-QKV companion parent source changed while writing")
        os.link(staged_tensors, output)
        published_output = output.stat(follow_symlinks=False)
        try:
            os.link(staged_receipt, receipt)
        except OSError:
            try:
                current_output = output.stat(follow_symlinks=False)
            except FileNotFoundError:
                current_output = None
            if (current_output is not None
                    and (current_output.st_dev, current_output.st_ino)
                    == (published_output.st_dev, published_output.st_ino)):
                output.unlink()
            raise
    finally:
        for child in stage.iterdir():
            child.unlink(missing_ok=True)
        stage.rmdir()
    document["receipt_path"] = str(receipt)
    return document


def run_reference_block0_cache_oracle(
    manifest_path: str | Path,
    *,
    reference_latent_path: str | Path,
    expected_reference_latent_artifact_sha256: str,
    expected_timeline_manifest_sha256: str,
    expected_manifest_sha256: str,
    weights_path: str | Path = BLOCK_WEIGHTS_DEFAULT,
    expected_weights_sha256: str = BLOCK_WEIGHTS_SHA256,
    upstream_root: str | Path = UPSTREAM_ROOT_DEFAULT,
    output_safetensors: str | Path,
    receipt_path: str | Path,
    precision: str = "bf16",
    capture_preqkv: bool = False,
    preqkv_output_safetensors: str | Path | None = None,
    preqkv_receipt_path: str | Path | None = None,
) -> dict[str, Any]:
    """Run pinned block 0 through its pre-attention cache append boundary on MPS.

    The input bundle is the native preblock boundary. It is not a complete
    upstream T0 preprocessing fixture. BF16 is the source inference path;
    ``f32-control`` is an explicitly non-parity precision control.
    """
    if precision not in ("bf16", "f32-control"):
        raise ReferenceCacheError("precision must be bf16 or f32-control")
    if type(capture_preqkv) is not bool:
        raise ReferenceCacheError("capture_preqkv must be a boolean")
    if capture_preqkv != (preqkv_output_safetensors is not None and preqkv_receipt_path is not None):
        raise ReferenceCacheError("pre-QKV capture requires explicit tensor and receipt output paths")
    if not capture_preqkv and (preqkv_output_safetensors is not None or preqkv_receipt_path is not None):
        raise ReferenceCacheError("pre-QKV output paths require capture_preqkv")
    if capture_preqkv:
        primary_paths = _validate_new_artifact_paths(
            output_safetensors, receipt_path, label="source oracle"
        )
        companion_paths = _validate_new_artifact_paths(
            preqkv_output_safetensors, preqkv_receipt_path,
            label="source pre-QKV companion",
        )
        if len(set((*primary_paths, *companion_paths))) != 4:
            raise ReferenceCacheError("primary and pre-QKV output paths must be distinct")
    validation = validate_reference_cache_manifest(
        manifest_path,
        reference_latent_path=reference_latent_path,
        expected_reference_latent_artifact_sha256=expected_reference_latent_artifact_sha256,
        expected_timeline_manifest_sha256=expected_timeline_manifest_sha256,
        expected_manifest_sha256=expected_manifest_sha256,
    )
    support = _load_support_oracle()
    upstream, upstream_hashes, upstream_revision = _verify_upstream_root(upstream_root, support)
    fallback = os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK")
    if fallback not in (None, "0"):
        raise ReferenceCacheError("PYTORCH_ENABLE_MPS_FALLBACK must be unset or 0")
    os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "0"
    import torch

    if not torch.backends.mps.is_built() or not torch.backends.mps.is_available():
        raise ReferenceCacheError("the source block oracle requires MPS; CPU fallback is forbidden")
    device = torch.device("mps")
    dtype = torch.bfloat16 if precision == "bf16" else torch.float32
    weights, state, source_dtype = _load_source_weights(
        torch, support, weights_path, expected_weights_sha256, parameter_dtype=dtype
    )
    inputs, input_records = _read_bundle_tensors(
        validation, torch=torch, device=device, input_dtype=dtype
    )
    helpers = support._install_upstream_modules(upstream)
    posemb_layers = helpers[-1]

    def reject_parallel_attention(*_args: Any, **_kwargs: Any) -> Any:
        raise ReferenceCacheError("source cache oracle unexpectedly entered non-cache attention")

    block_type = support._extract_pinned_block(
        upstream, torch, reject_parallel_attention,
        posemb_layers.apply_rotary_emb, helpers,
    )
    block = block_type(
        hidden_size=2048,
        heads_num=16,
        mlp_width_ratio=4.0,
        mlp_act_type="gelu_tanh",
        attn_mode="magi_mp_causal",
        qk_norm=True,
        qk_norm_type="rms",
        qkv_bias=True,
        dtype=dtype,
        device="meta",
    )
    block.load_state_dict(state, strict=True, assign=True)
    block.to(device=device)
    block.eval().requires_grad_(False)
    parameters = list(block.parameters())
    if not parameters or any(parameter.dtype != dtype or parameter.device.type != "mps"
                             for parameter in parameters):
        raise ReferenceCacheError("source block parameter dtype/device differs from the admitted mode")
    autocast_available = getattr(getattr(torch, "amp", None), "is_autocast_available", None)
    if autocast_available is None:
        autocast_available = getattr(torch, "is_autocast_available", None)
    if precision == "bf16" and (
            autocast_available is None or not autocast_available("mps")):
        raise ReferenceCacheError("PyTorch MPS BF16 autocast is unavailable")

    text_tokens = int(validation["valid_text_tokens"])
    cache_shape = (1, 4320 + text_tokens, 16, 128)
    cache_buffers = {
        "k": torch.empty(cache_shape, dtype=torch.bfloat16, device=device),
        "v": torch.empty(cache_shape, dtype=torch.bfloat16, device=device),
    }
    preqkv_capture: dict[str, Any] | None = {} if capture_preqkv else None
    capture = _capture_block0_cache(
        torch=torch,
        block=block,
        block_class_globals=block.forward_with_cache.__globals__,
        inputs=inputs,
        cache_buffers=cache_buffers,
        mode=precision,
        preqkv_capture=preqkv_capture,
    )
    _verify_typed_capture(capture, torch, dtype=dtype, text_tokens=text_tokens)
    if preqkv_capture is not None:
        _verify_typed_preqkv_capture(
            preqkv_capture, torch, dtype=dtype, text_tokens=text_tokens
        )

    metadata: dict[str, Any] = {
        "upstream_revision": upstream_revision,
        "upstream_source_root": str(upstream),
        "upstream_source_file_sha256": upstream_hashes["hymm/models/dit/nomas_cpt.py"],
        "upstream_dependency_sha256": {
            name: upstream_hashes[name] for name in support.PINNED_SOURCE_SHA256
        },
        "inference_dtype_source_sha256": {
            name: upstream_hashes[name] for name in INFERENCE_DTYPE_SOURCE_SHA256
        },
        "weight_source": {
            "path": str(weights),
            "sha256": expected_weights_sha256,
            "source_dtype": source_dtype,
            "parameter_dtype": "BF16" if dtype == torch.bfloat16 else "F32",
            "parameter_values": (
                "exact authenticated BF16 values"
                if dtype == torch.bfloat16 else "authenticated BF16 values widened to F32"
            ),
            "tensor_prefix": "double_blocks.0.",
            "tensor_count": len(state),
        },
        "parameter_dtype_basis": [
            "inference.yaml sets dit.master_weight_type=bf16",
            "BasePipeline maps master_weight_type to torch.bfloat16 target_dtype",
            "BasePipeline moves the loaded transformer to target_dtype",
            "CausalAR casts reference latents to target_dtype before block inputs",
        ],
        "precision": {
            "mode": precision,
            "block_parameters": "BF16" if dtype == torch.bfloat16 else "F32",
            "block_inputs": "BF16 cast from authenticated F32 preblock bundle" if dtype == torch.bfloat16 else "F32",
            "autocast": "MPS BF16" if dtype == torch.bfloat16 else "disabled",
            "cache_storage": "BF16",
        },
        "scope": (
            "Pinned upstream MMDoubleStreamBlock.forward_with_cache on the exact native "
            "preblock input tensors; captures block-0 K/V at first-chunk cache append. "
            "Does not claim parity for upstream timeline, noise, patch, or text-refiner preprocessing."
        ),
        "cache_append": {
            "cache_valid_len": 0,
            "update_cache": True,
            "cache_txt": True,
            "attention_executed": False,
            "cache_tensor_order": "image tokens followed by valid text prefix",
            "cache_tokens": cache_shape[1],
        },
        "input_bundle": {
            "manifest_path": validation["manifest_path"],
            "manifest_sha256": validation["manifest_sha256"],
            "payload_sha256": validation["payload_sha256"],
            "reference_latent_artifact_sha256": validation["reference_latent_artifact_sha256"],
            "reference_latent_tensor_sha256": validation["reference_latent_tensor_sha256"],
            "timeline_manifest_sha256": validation["timeline_manifest_sha256"],
            "valid_text_tokens": text_tokens,
            "consumed_tensor_names": [
                "block.img_f32", "block.txt_valid_f32", "block.vec_f32",
                "block.vec_txt_f32", "block.rope_cos_f32", "block.rope_sin_f32",
            ],
            "authenticated_but_not_consumed": [
                "reference.latent_bf16", "cache.noisy_latent_f32", "cache.timestep_f32",
            ],
            "tensor_records": {
                name: {
                    "dtype": record["dtype"], "shape": record["shape"],
                    "sha256": record["sha256"],
                }
                for name, record in input_records.items()
            },
        },
    }
    receipt = _save_oracle_artifact(
        capture, output_safetensors=output_safetensors,
        receipt_path=receipt_path, torch=torch, metadata=metadata,
    )
    if preqkv_capture is not None:
        companion = _save_preqkv_artifact(
            preqkv_capture,
            output_safetensors=preqkv_output_safetensors,
            receipt_path=preqkv_receipt_path,
            torch=torch,
            precision=precision,
            validation=validation,
            weights_sha256=expected_weights_sha256,
            parent_source_receipt_path=receipt_path,
            parent_source_artifact_path=output_safetensors,
        )
        receipt["preqkv_companion"] = {
            "receipt_path": companion["receipt_path"],
            "tensor_artifact": companion["tensor_artifact"],
        }
    del block, state, inputs, capture, cache_buffers, preqkv_capture
    return receipt


def _verify_typed_capture(capture: dict[str, Any], torch: Any, *, dtype: Any,
                          text_tokens: int) -> None:
    expected_shapes = {
        "img.q_projection": (1, 4320, 16, 128),
        "img.k_projection": (1, 4320, 16, 128),
        "img.v_projection": (1, 4320, 16, 128),
        "img.q_qknorm": (1, 4320, 16, 128),
        "img.k_qknorm": (1, 4320, 16, 128),
        "img.q_rope": (1, 4320, 16, 128),
        "img.k_rope": (1, 4320, 16, 128),
        "txt.q_projection": (1, text_tokens, 16, 128),
        "txt.k_projection": (1, text_tokens, 16, 128),
        "txt.v_projection": (1, text_tokens, 16, 128),
        "txt.q_qknorm": (1, text_tokens, 16, 128),
        "txt.k_qknorm": (1, text_tokens, 16, 128),
        "cache.k_bf16": (1, 4320 + text_tokens, 16, 128),
        "cache.v_bf16": (1, 4320 + text_tokens, 16, 128),
    }
    for name, expected_shape in expected_shapes.items():
        tensor = capture[name]
        expected_dtype = torch.bfloat16 if name.endswith("_bf16") or name.endswith("bf16") else dtype
        if tuple(tensor.shape) != expected_shape or tensor.dtype != expected_dtype:
            raise ReferenceCacheError(
                f"source capture shape or dtype mismatch for {name}: "
                f"{tuple(tensor.shape)} {tensor.dtype}"
            )


def _expected_preqkv_shapes(text_tokens: int) -> dict[str, tuple[int, ...]]:
    if type(text_tokens) is not int or not 1 <= text_tokens <= 1280:
        raise ReferenceCacheError("pre-QKV text extent must be an integer in [1, 1280]")
    return {
        "img.mod_silu": (1, 2048),
        "img.mod_params": (1, 12_288),
        "img.norm1": (1, 4320, 2048),
        "img.qkv_input": (1, 4320, 2048),
        "txt.mod_silu": (1, 2048),
        "txt.mod_params": (1, 12_288),
        "txt.norm1": (1, text_tokens, 2048),
        "txt.qkv_input": (1, text_tokens, 2048),
    }


def _verify_typed_preqkv_capture(
    capture: dict[str, Any], torch: Any, *, dtype: Any, text_tokens: int,
) -> None:
    shapes = _expected_preqkv_shapes(text_tokens)
    actual_inventory = {
        name: f"shape={tuple(tensor.shape)},dtype={tensor.dtype}"
        for name, tensor in capture.items()
    }
    expected_dtype_by_name = {
        name: (
            torch.bfloat16
            if dtype == torch.bfloat16 and name in PREQKV_BF16_BOUNDARIES
            else torch.float32
        )
        for name in PREQKV_NAMES
    }
    mismatches: list[str] = []
    if len(capture) != len(PREQKV_NAMES) or set(capture) != set(PREQKV_NAMES):
        mismatches.append(
            f"names missing={sorted(set(PREQKV_NAMES) - set(capture))}, "
            f"extra={sorted(set(capture) - set(PREQKV_NAMES))}"
        )
    for name, expected_shape in shapes.items():
        if name not in capture:
            continue
        tensor = capture[name]
        expected_dtype = expected_dtype_by_name[name]
        if tuple(tensor.shape) != expected_shape or tensor.dtype != expected_dtype:
            mismatches.append(
                f"{name} expected shape={expected_shape},dtype={expected_dtype}; "
                f"actual shape={tuple(tensor.shape)},dtype={tensor.dtype}"
            )
    if mismatches or tuple(capture) != PREQKV_NAMES:
        raise ReferenceCacheError(
            "source pre-QKV capture profile mismatch; "
            f"issues={mismatches or ['capture order differs from canonical order']}; "
            f"actual_inventory={actual_inventory}"
        )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--reference-latent", type=Path, required=True)
    parser.add_argument("--reference-latent-sha256", required=True)
    parser.add_argument("--timeline-manifest-sha256", required=True)
    parser.add_argument("--manifest-sha256", required=True,
                        help="trusted SHA-256 of the native preblock manifest")
    parser.add_argument("--weights", type=Path, default=BLOCK_WEIGHTS_DEFAULT)
    parser.add_argument("--weights-sha256", default=BLOCK_WEIGHTS_SHA256)
    parser.add_argument("--upstream-root", type=Path, default=UPSTREAM_ROOT_DEFAULT)
    parser.add_argument("--output-safetensors", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--precision", choices=("bf16", "f32-control"), default="bf16")
    parser.add_argument("--capture-preqkv", action="store_true",
                        help="write a separate diagnostic companion tied to this run's main oracle output")
    parser.add_argument("--preqkv-output-safetensors", type=Path)
    parser.add_argument("--preqkv-receipt", type=Path)
    args = parser.parse_args(argv)
    if args.capture_preqkv and (args.preqkv_output_safetensors is None or args.preqkv_receipt is None):
        parser.error("--capture-preqkv requires --preqkv-output-safetensors and --preqkv-receipt")
    if not args.capture_preqkv and (
            args.preqkv_output_safetensors is not None or args.preqkv_receipt is not None):
        parser.error("pre-QKV output paths require --capture-preqkv")
    result = run_reference_block0_cache_oracle(
        args.manifest,
        reference_latent_path=args.reference_latent,
        expected_reference_latent_artifact_sha256=args.reference_latent_sha256,
        expected_timeline_manifest_sha256=args.timeline_manifest_sha256,
        expected_manifest_sha256=args.manifest_sha256,
        weights_path=args.weights,
        expected_weights_sha256=args.weights_sha256,
        upstream_root=args.upstream_root,
        output_safetensors=args.output_safetensors,
        receipt_path=args.receipt,
        precision=args.precision,
        capture_preqkv=args.capture_preqkv,
        preqkv_output_safetensors=args.preqkv_output_safetensors,
        preqkv_receipt_path=args.preqkv_receipt,
    )
    print(json.dumps({
        "schema": result["schema"],
        "receipt_path": result["receipt_path"],
        "tensor_artifact": result["tensor_artifact"],
        "precision": result["precision"],
        "cache_append": result["cache_append"],
        "preqkv_companion": result.get("preqkv_companion"),
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
