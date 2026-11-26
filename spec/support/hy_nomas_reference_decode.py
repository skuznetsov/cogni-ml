#!/usr/bin/env python3
"""Authenticate and compare a saved DCAE-V latent through PixelUMM on MPS.

This is a decoder-only diagnostic.  It consumes the exact saved BF16 DCAE-V
latent and authenticated PixelUMM noise/weights; it does not synthesize DiT
seed, step, or guidance metadata and does not claim full video generation.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import sys
from typing import Mapping


SUPPORT_DIR = Path(__file__).resolve().parent
if str(SUPPORT_DIR) not in sys.path:
    sys.path.insert(0, str(SUPPORT_DIR))

NATIVE_SCHEMA = "cogni_ml.hy_nomas_reference_decode_native/v1"
ORACLE_SCHEMA = "cogni_ml.hy_nomas_reference_decode_oracle/v1"
NATIVE_OPERATION = "saved_dcae_latent_pixelumm_decoder"
NATIVE_PRECISION = (
    "BF16 PixelUMM block/head weights; F32 stem and native MPS activations; "
    "saved BF16 latent expanded to F32 on GPU"
)
DCAE_REFERENCE_REVISION = "a31c0fe0bcc493be188fd37ddba4ce2f674becb4"
PIXELUMM_UPSTREAM_REVISION = "911cea6458a89c2663513ff7099561ec190dd491"
DCAE_CHECKPOINT_REVISION = "bb6a8ca451bfe84947f0ea1c6d51957f983ffffe"
DCAE_CHECKPOINT_SHA256 = "ebbee483c80890c7795a2babc06e1cb4d42564174cf3bae2b698c0589d95cc3b"
DCAE_MODULE_SHA256 = "f090a77806e250203618742a80263457cd2b44a5f9677bd35b40cfe9ce0fdc1d"
DCAE_LATENT_FILE_SHA256 = "115628ac1db4dca3533e410d4affbe9b9ce8828b0e93ff8d8be60645a3aa4a38"
DCAE_UPSTREAM_ARTIFACT_SHA256 = "8b7dcc20b55b27d0588fc8531f59c3f9157af8de57d0cf9bf6e5d226e2ab8ed7"
DCAE_LATENT_SHAPE = (1, 32, 30, 9, 16)
LATENT_F32_BYTES = 552_960
PIXEL_SHAPE = (1, 3, 120, 288, 512)
PIXEL_F32_BYTES = 212_336_640
INPUT_NOISE_SHA_FIELD = "decoder.noise"
ATTENTION_SCOPE = "gathered_spatial_tiles_qk_softmax_pv_fp32_accum"
ATOL = RTOL = 2e-5
DEFAULT_RUN_ROOT = Path(
    "/opt/mldata_ssd/cogni-ml/hy-nomas-reference-handoff-20261101.pDnd7TV1"
)


class ReferenceDecodeError(ValueError):
    """The native capture, DCAE lineage, or MPS oracle admission failed."""


def _digest(value: object) -> bool:
    return (
        isinstance(value, str) and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ReferenceDecodeError(f"duplicate JSON field: {key}")
        result[key] = value
    return result


def _read_json(path: Path, what: str, *, max_bytes: int = 2 * 1024 * 1024) -> tuple[dict, bytes]:
    if path.is_symlink() or not path.is_file():
        raise ReferenceDecodeError(f"{what} must be a regular non-symlink file")
    raw = path.read_bytes()
    if len(raw) > max_bytes:
        raise ReferenceDecodeError(f"{what} exceeds the metadata size limit")
    try:
        document = json.loads(raw.decode("utf-8"), object_pairs_hook=_unique_object)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ReferenceDecodeError(f"cannot parse {what}") from error
    if type(document) is not dict:
        raise ReferenceDecodeError(f"{what} must be a JSON object")
    return document, raw


def _canonical_file(value: str | Path, what: str) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute() or path.is_symlink() or not path.is_file():
        raise ReferenceDecodeError(f"{what} must be an absolute regular non-symlink file")
    try:
        return path.resolve(strict=True)
    except (OSError, RuntimeError) as error:
        raise ReferenceDecodeError(f"cannot resolve {what}") from error


def _require_exact_keys(value: Mapping, keys: set[str], what: str) -> None:
    if type(value) is not dict or set(value) != keys:
        actual = set(value) if type(value) is dict else set()
        missing = sorted(keys - actual)
        extra = sorted(actual - keys)
        raise ReferenceDecodeError(
            f"{what} schema fields are invalid; missing={missing}, extra={extra}"
        )


def _validate_record_ref(value: object, what: str) -> dict[str, object]:
    if type(value) is not dict or set(value) != {"path", "sha256"}:
        raise ReferenceDecodeError(f"{what} reference schema is invalid")
    if not isinstance(value["path"], str) or not Path(value["path"]).is_absolute():
        raise ReferenceDecodeError(f"{what} path must be absolute")
    if not _digest(value["sha256"]):
        raise ReferenceDecodeError(f"{what} SHA-256 is malformed")
    return dict(value)


def validate_native_receipt_document(
    document: Mapping, *, expected_inputs_sha256: str,
    expected_source_latent_sha256: str = DCAE_LATENT_FILE_SHA256,
) -> dict[str, object]:
    """Validate the decoder-only receipt contract without opening large files."""
    required = {
        "schema", "operation", "upstream_revision", "precision", "resolution",
        "width", "height", "frames", "fps", "source_latent", "dcae_lineage",
        "decoder_inputs", "weights", "outputs", "pixelumm", "execution",
    }
    _require_exact_keys(document, required, "native decoder receipt")
    if document.get("schema") != NATIVE_SCHEMA:
        raise ReferenceDecodeError("unsupported native decoder receipt schema")
    if document.get("operation") != NATIVE_OPERATION:
        raise ReferenceDecodeError("native receipt operation is not saved-latent decode")
    if document.get("upstream_revision") != PIXELUMM_UPSTREAM_REVISION:
        raise ReferenceDecodeError("native receipt PixelUMM source revision mismatch")
    if document.get("precision") != NATIVE_PRECISION:
        raise ReferenceDecodeError("native receipt precision contract is unknown")
    if (document.get("resolution") != "512x288"
            or type(document.get("width")) is not int or document["width"] != 512
            or type(document.get("height")) is not int or document["height"] != 288
            or type(document.get("frames")) is not int or document["frames"] != 120
            or type(document.get("fps")) is not int or document["fps"] != 24):
        raise ReferenceDecodeError("only the default 512x288, 120-frame decode is admitted")

    source = document.get("source_latent")
    _require_exact_keys(source, {"path", "sha256", "tensor_key", "dtype", "shape", "nbytes"},
                        "saved DCAE latent")
    if not isinstance(source["path"], str) or not Path(source["path"]).is_absolute():
        raise ReferenceDecodeError("saved DCAE latent path must be absolute")
    source_sha = source["sha256"]
    if not _digest(source_sha) or source_sha != expected_source_latent_sha256:
        raise ReferenceDecodeError("saved DCAE latent file SHA-256 mismatch")
    if (source["tensor_key"] != "latent" or source["dtype"] != "BF16"
            or source["shape"] != list(DCAE_LATENT_SHAPE)
            or type(source["nbytes"]) is not int or source["nbytes"] != 276_480):
        raise ReferenceDecodeError("saved DCAE latent key, dtype, or shape mismatch")

    lineage = document.get("dcae_lineage")
    _require_exact_keys(lineage,
                        {"upstream_reference", "upstream_artifact", "native_receipt",
                         "comparison", "bitwise_comparison"}, "DCAE lineage")
    for key in ("upstream_reference", "upstream_artifact", "native_receipt",
                "comparison", "bitwise_comparison"):
        _validate_record_ref(lineage[key], f"DCAE {key}")
    if lineage["upstream_artifact"]["sha256"] != DCAE_UPSTREAM_ARTIFACT_SHA256:
        raise ReferenceDecodeError("DCAE upstream reference artifact SHA-256 mismatch")

    inputs = document.get("decoder_inputs")
    _require_exact_keys(inputs, {"manifest_path", "manifest_sha256", "payload_sha256", "prompt", "noise"},
                        "decoder input binding")
    if not isinstance(inputs["manifest_path"], str) or not Path(inputs["manifest_path"]).is_absolute():
        raise ReferenceDecodeError("decoder input manifest path must be absolute")
    for key in ("manifest_sha256", "payload_sha256"):
        if not _digest(inputs[key]):
            raise ReferenceDecodeError(f"decoder input {key} is malformed")
    if not _digest(expected_inputs_sha256) or inputs["manifest_sha256"] != expected_inputs_sha256:
        raise ReferenceDecodeError("decoder input manifest SHA-256 differs from caller authorization")
    if type(inputs["prompt"]) is not str or not inputs["prompt"].strip():
        raise ReferenceDecodeError("authenticated decoder prompt is missing")
    noise = inputs["noise"]
    _require_exact_keys(noise, {"key", "dtype", "shape", "nbytes", "sha256"},
                        "decoder noise binding")
    if (noise["key"] != INPUT_NOISE_SHA_FIELD or noise["dtype"] != "F32"
            or noise["shape"] != list(PIXEL_SHAPE)
            or type(noise["nbytes"]) is not int or noise["nbytes"] != PIXEL_F32_BYTES
            or not _digest(noise["sha256"])):
        raise ReferenceDecodeError("authenticated decoder noise key, dtype, shape, or SHA-256 mismatch")

    weights = document.get("weights")
    _require_exact_keys(weights, {"decoder_cache_manifest_sha256", "decoder_cache_payload_sha256",
                                  "stem_subset_sha256", "head_subset_sha256"},
                        "PixelUMM weight identities")
    if any(not _digest(value) for value in weights.values()):
        raise ReferenceDecodeError("PixelUMM weight identity SHA-256 is malformed")

    outputs = document.get("outputs")
    _require_exact_keys(outputs, {"latent_f32", "raw_pixels", "display_pixels"}, "native decoder outputs")
    expected_outputs = {
        "latent_f32": ("F32LE", list(DCAE_LATENT_SHAPE), LATENT_F32_BYTES),
        "raw_pixels": ("F32LE", list(PIXEL_SHAPE), PIXEL_F32_BYTES),
        "display_pixels": ("F32LE", list(PIXEL_SHAPE), PIXEL_F32_BYTES),
    }
    for name, (dtype, shape, nbytes) in expected_outputs.items():
        record = outputs[name]
        _require_exact_keys(record, {"path", "dtype", "shape", "nbytes", "sha256"},
                            f"native output {name}")
        if (not isinstance(record["path"], str) or not Path(record["path"]).is_absolute()
                or record["dtype"] != dtype or record["shape"] != shape
                or type(record["nbytes"]) is not int or record["nbytes"] != nbytes
                or not _digest(record["sha256"])):
            raise ReferenceDecodeError(f"native output {name} path, dtype, shape, extent, or SHA-256 mismatch")

    pixelumm = document.get("pixelumm")
    _require_exact_keys(pixelumm, {"window_latents", "overlap_latents", "starts",
                                   "prediction_count_per_window"}, "PixelUMM window contract")
    if pixelumm != {"window_latents": 10, "overlap_latents": 5,
                    "starts": [0, 5, 10, 15, 20], "prediction_count_per_window": 1}:
        raise ReferenceDecodeError("PixelUMM window contract mismatch")

    execution = document.get("execution")
    _require_exact_keys(execution, {"device", "terminal_drain_confirmed",
                                    "exports_after_terminal_drain", "metrics", "timings",
                                    "decoder_named_bytes", "saved_latent_source_bytes",
                                    "aggregate_named_bytes",
                                    "opaque_mps_private_allocation_visibility"},
                        "native execution evidence")
    metrics = execution["metrics"]
    _require_exact_keys(metrics, {"decoder_encode_allocations", "decoder_encode_readbacks",
                                  "decoder_preterminal_waits", "decoder_waits"},
                        "native encode metrics")
    if (any(type(execution[name]) is not int or execution[name] < 0
            for name in ("decoder_named_bytes", "saved_latent_source_bytes",
                         "aggregate_named_bytes"))
            or execution["saved_latent_source_bytes"] != 276_480
            or execution["aggregate_named_bytes"] != (
                execution["decoder_named_bytes"] + execution["saved_latent_source_bytes"]
            )):
        raise ReferenceDecodeError("native decoder aggregate_named_bytes receipt is inconsistent")
    if (type(execution["timings"]) is not dict
            or any(type(value) not in (int, float) or not math.isfinite(value) or value < 0
                   for value in execution["timings"].values())
            or execution["opaque_mps_private_allocation_visibility"]
            != "not observable through MetalBuffer instrumentation"):
        raise ReferenceDecodeError("native decoder timing or opaque allocation boundary is invalid")
    if (execution["device"] != "mps"
            or execution["terminal_drain_confirmed"] is not True
            or execution["exports_after_terminal_drain"] is not True
            or any(type(metrics[name]) is not int or metrics[name] != 0
                   for name in ("decoder_encode_allocations", "decoder_encode_readbacks",
                                "decoder_preterminal_waits"))
            or type(metrics["decoder_waits"]) is not int or metrics["decoder_waits"] != 1):
        raise ReferenceDecodeError(
            "native encode preterminal allocations/readbacks/waits must be zero before one terminal drain"
        )

    return {
        "latent": outputs["latent_f32"]["path"],
        "latent_sha256": outputs["latent_f32"]["sha256"],
        "raw_pixels": outputs["raw_pixels"]["path"],
        "raw_pixels_sha256": outputs["raw_pixels"]["sha256"],
        "display_pixels": outputs["display_pixels"]["path"],
        "display_pixels_sha256": outputs["display_pixels"]["sha256"],
        "width": 512,
        "height": 288,
        "resolution": "512x288",
        "geometry": {
            "latent_shape": DCAE_LATENT_SHAPE,
            "raw_shape": PIXEL_SHAPE,
            "latent_bytes": LATENT_F32_BYTES,
            "raw_bytes": PIXEL_F32_BYTES,
        },
    }


def _validate_safetensors_latent(path: Path, expected_sha256: str) -> dict[str, object]:
    """Verify the pinned file digest and the exact BF16 `latent` header range."""
    path = _canonical_file(path, "saved DCAE latent safetensors")
    if _sha256_file(path) != expected_sha256:
        raise ReferenceDecodeError("saved DCAE latent safetensors SHA-256 mismatch")
    size = path.stat().st_size
    with path.open("rb") as stream:
        prefix = stream.read(8)
        if len(prefix) != 8:
            raise ReferenceDecodeError("saved DCAE safetensors header is truncated")
        header_bytes = int.from_bytes(prefix, "little", signed=False)
        if header_bytes < 2 or header_bytes > 16 * 1024 * 1024 or 8 + header_bytes > size:
            raise ReferenceDecodeError("saved DCAE safetensors header size is invalid")
        raw_header = stream.read(header_bytes)
    try:
        header = json.loads(raw_header.decode("utf-8"), object_pairs_hook=_unique_object)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ReferenceDecodeError("saved DCAE safetensors header is invalid") from error
    if type(header) is not dict or "latent" not in header:
        raise ReferenceDecodeError("saved DCAE safetensors is missing the latent tensor")
    record = header["latent"]
    if (type(record) is not dict or record.get("dtype") != "BF16"
            or record.get("shape") != list(DCAE_LATENT_SHAPE)):
        raise ReferenceDecodeError("saved DCAE safetensors latent dtype or shape mismatch")
    offsets = record.get("data_offsets")
    if (type(offsets) is not list or len(offsets) != 2
            or any(type(value) is not int for value in offsets)
            or offsets[0] < 0 or offsets[1] - offsets[0] != 276_480
            or offsets[1] > size - 8 - header_bytes):
        raise ReferenceDecodeError("saved DCAE safetensors latent payload extent is invalid")
    return {"path": str(path), "sha256": expected_sha256, "header_sha256": hashlib.sha256(raw_header).hexdigest(),
            "data_offset": 8 + header_bytes + offsets[0], "nbytes": offsets[1] - offsets[0]}


def _verify_exact_bf16_to_f32_expansion(source_info: Mapping, output_path: str | Path) -> None:
    """Require each BF16 payload word to widen to its exact little-endian F32 word."""
    source_path = _canonical_file(source_info["path"], "saved DCAE latent safetensors")
    output = _canonical_file(output_path, "expanded F32 latent")
    if (source_info.get("nbytes") != 276_480
            or type(source_info.get("data_offset")) is not int
            or source_info["data_offset"] < 0
            or output.stat().st_size != LATENT_F32_BYTES):
        raise ReferenceDecodeError("saved latent expansion source or output extent is invalid")
    source_offset = source_info["data_offset"]
    remaining = source_info["nbytes"]
    try:
        with source_path.open("rb") as source, output.open("rb") as actual:
            source.seek(source_offset)
            while remaining:
                chunk = source.read(min(64 * 1024, remaining))
                if not chunk or len(chunk) % 2:
                    raise ReferenceDecodeError("saved BF16 latent payload is truncated or misaligned")
                expected = bytearray(len(chunk) * 2)
                for index in range(0, len(chunk), 2):
                    target = index * 2
                    expected[target + 2] = chunk[index]
                    expected[target + 3] = chunk[index + 1]
                if actual.read(len(expected)) != expected:
                    raise ReferenceDecodeError(
                        "native F32 latent is not the exact GPU BF16-word expansion of the saved DCAE latent"
                    )
                remaining -= len(chunk)
            if actual.read(1):
                raise ReferenceDecodeError("native F32 latent has trailing bytes")
    except OSError as error:
        raise ReferenceDecodeError(f"cannot compare saved BF16 latent expansion: {error}") from error


def _authenticate_dcae_lineage(receipt: Mapping, source_latent_path: Path) -> dict[str, object]:
    """Authenticate the existing DCAE source/candidate manifests and their claims."""
    refs = receipt["dcae_lineage"]
    resolved: dict[str, Path] = {}
    docs: dict[str, dict] = {}
    actual_hashes: dict[str, str] = {}
    for key in ("upstream_reference", "native_receipt", "comparison", "bitwise_comparison"):
        ref = refs[key]
        path = _canonical_file(ref["path"], f"DCAE {key}")
        digest = _sha256_file(path)
        if digest != ref["sha256"]:
            raise ReferenceDecodeError(f"DCAE {key} SHA-256 differs from native receipt")
        resolved[key] = path
        actual_hashes[key] = digest
    for key in ("upstream_reference", "native_receipt", "comparison", "bitwise_comparison"):
        docs[key], _ = _read_json(resolved[key], f"DCAE {key}")

    reference = docs["upstream_reference"]
    upstream = reference.get("upstream")
    checkpoint = reference.get("checkpoint")
    reference_input = reference.get("input")
    execution = reference.get("execution")
    artifact = reference.get("artifact")
    if any(type(value) is not dict for value in
           (upstream, checkpoint, reference_input, execution, artifact)):
        raise ReferenceDecodeError("pinned DCAE source reference sections are malformed")
    if (reference.get("schema") != "hy-nomas-dcae-v-upstream-reference-v1"
            or upstream.get("repository") != "Tencent-Hunyuan/Hy-Nomas"
            or upstream.get("reference_image_revision") != DCAE_REFERENCE_REVISION
            or upstream.get("vae_source_sha256") != DCAE_MODULE_SHA256
            or checkpoint.get("repository_revision") != DCAE_CHECKPOINT_REVISION
            or checkpoint.get("sha256") != DCAE_CHECKPOINT_SHA256
            or checkpoint.get("bytes") != 4_277_389_995
            or checkpoint.get("encoder_only_materialization") is not True
            or checkpoint.get("load_policy") != "torch.load(weights_only=True,mmap=True,map_location=cpu)"):
        raise ReferenceDecodeError("pinned upstream DCAE reference provenance mismatch")
    expected_reference_contract = {
        "frames": 120, "fps": 24, "height": 288, "width": 512,
        "tile_frames": 40, "temporal_feature_cache": True,
        "repeat": "identical pixel frame repeated in pixel space before DCAE-V encode",
    }
    source_input = reference_input
    if any(source_input.get(key) != value for key, value in expected_reference_contract.items()):
        raise ReferenceDecodeError("pinned DCAE upstream reference geometry or image path mismatch")
    if (execution.get("device") != "mps"
            or execution.get("cpu_fallback_enabled") is not False
            or execution.get("dtype") != "BF16"
            or execution.get("torch_version") != "2.8.0"
            or execution.get("latent_postprocess")
            != "identity: upstream _LatentDist.mode returns the encoder output"):
        raise ReferenceDecodeError("DCAE reference is not a source-MPS identity latent capture")
    latent_spec = artifact.get("tensors", {}).get("latent")
    if latent_spec != {"dtype": "BF16", "shape": list(DCAE_LATENT_SHAPE)}:
        raise ReferenceDecodeError("upstream DCAE latent manifest shape or dtype mismatch")
    upstream_artifact = refs["upstream_artifact"]
    expected_upstream = artifact
    upstream_path = _canonical_file(upstream_artifact["path"], "upstream DCAE reference safetensors")
    if (upstream_artifact["sha256"] != expected_upstream.get("sha256")
            or upstream_artifact["sha256"] != DCAE_UPSTREAM_ARTIFACT_SHA256
            or upstream_path.name != expected_upstream.get("filename")
            or upstream_path.stat().st_size != expected_upstream.get("bytes")
            or _sha256_file(upstream_path) != upstream_artifact["sha256"]):
        raise ReferenceDecodeError("upstream DCAE reference artifact identity mismatch")
    _validate_safetensors_latent(upstream_path, DCAE_UPSTREAM_ARTIFACT_SHA256)

    source_info = _validate_safetensors_latent(source_latent_path, DCAE_LATENT_FILE_SHA256)
    native_latent = receipt["outputs"]["latent_f32"]
    _verify_exact_bf16_to_f32_expansion(source_info, native_latent["path"])
    native = docs["native_receipt"]
    if (native.get("event") != "dcae_native_completed"
            or native.get("output_sha256") != DCAE_LATENT_FILE_SHA256
            or _canonical_file(native.get("output", ""), "native DCAE candidate") != source_latent_path
            or native.get("metrics") != {"allocations": 0, "readbacks": 0, "preterminal_waits": 0}
            or type(native.get("waits")) is not int or native.get("waits") != 1):
        raise ReferenceDecodeError("native DCAE candidate receipt identity or terminal metrics mismatch")
    comparison = docs["comparison"]
    expected_comparisons = {"input.clip", "trace.stages.5.tile0", "trace.stages.5.tile1",
                            "trace.stages.5.tile2", "latent"}
    entries = comparison.get("comparisons")
    if (comparison.get("schema") != "hy-nomas-dcae-v-comparison-v1"
            or comparison.get("all_passed") is not True
            or comparison.get("atol") != 2e-5 or comparison.get("rtol") != 2e-5
            or comparison.get("candidate_sha256") != DCAE_LATENT_FILE_SHA256
            or comparison.get("upstream_reference_sha256") != DCAE_UPSTREAM_ARTIFACT_SHA256
            or type(entries) is not list
            or len(entries) != len(expected_comparisons)
            or any(type(entry) is not dict for entry in entries)
            or {entry.get("name") for entry in entries} != expected_comparisons
            or any(entry.get("passed") is not True
                   or type(entry.get("max_abs_error")) not in (int, float)
                   or entry.get("max_abs_error") != 0.0
                   for entry in entries)):
        raise ReferenceDecodeError("upstream/native DCAE comparison evidence does not show exact saved-latent parity")
    bitwise = docs["bitwise_comparison"]
    tensors = bitwise.get("tensors")
    if (bitwise.get("all_bitwise_equal") is not True or type(tensors) is not dict
            or set(tensors) != expected_comparisons
            or any(type(tensors[name]) is not dict
                   or tensors[name].get("bitwise_equal") is not True
                   or tensors[name].get("bytes") != expected_bytes
                   for name, expected_bytes in {
                       "input.clip": 106_168_320,
                       "trace.stages.5.tile0": 2_949_120,
                       "trace.stages.5.tile1": 2_949_120,
                       "trace.stages.5.tile2": 2_949_120,
                       "latent": 276_480,
                   }.items())):
        raise ReferenceDecodeError("DCAE bitwise comparison evidence is incomplete or failed")
    return {
        "dcae_reference_image_revision": DCAE_REFERENCE_REVISION,
        "checkpoint_revision": DCAE_CHECKPOINT_REVISION,
        "checkpoint_sha256": DCAE_CHECKPOINT_SHA256,
        "module_sha256": DCAE_MODULE_SHA256,
        "source_latent": source_info,
        "native_latent_expansion": "exact BF16 payload word widened to F32; verified byte-for-byte",
        "upstream_reference": str(resolved["upstream_reference"]),
        "upstream_reference_sha256": actual_hashes["upstream_reference"],
        "upstream_artifact": str(upstream_path),
        "upstream_artifact_sha256": upstream_artifact["sha256"],
        "native_receipt_sha256": actual_hashes["native_receipt"],
        "comparison_sha256": actual_hashes["comparison"],
        "bitwise_comparison_sha256": actual_hashes["bitwise_comparison"],
        "comparison_result": "bitwise_match",
    }


def _regular_artifact(path_value: str, what: str, expected_sha256: str, expected_bytes: int) -> Path:
    path = _canonical_file(path_value, what)
    if path.stat().st_size != expected_bytes:
        raise ReferenceDecodeError(f"{what} byte extent does not match its admitted shape")
    if not _digest(expected_sha256) or _sha256_file(path) != expected_sha256:
        raise ReferenceDecodeError(f"{what} SHA-256 mismatch")
    return path


def _output_paths(path_arg: Path) -> tuple[Path, Path, Path, Path]:
    path = path_arg.expanduser()
    if not path.is_absolute() or path.suffix.lower() != ".json":
        raise ReferenceDecodeError("--output must be an absolute .json path")
    try:
        root = DEFAULT_RUN_ROOT.resolve(strict=True)
        parent = path.parent.resolve(strict=True)
    except OSError as error:
        raise ReferenceDecodeError("handoff scratch root and output parent must already exist") from error
    if root != parent and root not in parent.parents:
        raise ReferenceDecodeError("oracle outputs must remain beneath the authorized handoff root")
    report = parent / path.name
    raw = parent / (path.stem + "-upstream-raw.f32le")
    first = parent / (path.stem + "-upstream-first-window.f32le")
    display = parent / (path.stem + "-upstream-display.f32le")
    internal = parent / (path.stem + ".same-latent-internal.json")
    if any(os.path.lexists(candidate) for candidate in (report, raw, first, display, internal)):
        raise ReferenceDecodeError("refusing to replace an existing decoder oracle result")
    return report, raw, first, display


def _same_latent_module():
    if str(SUPPORT_DIR) not in sys.path:
        sys.path.insert(0, str(SUPPORT_DIR))
    try:
        import hy_nomas_decoder_same_latent as same_latent
    except (ImportError, RuntimeError) as error:
        raise ReferenceDecodeError(f"cannot import the pinned same-latent MPS adapter: {error}") from error
    return same_latent


class _DecoderOnlyReceiptView(Mapping):
    """Narrow adapter view for legacy internals; absent DiT fields stay absent."""

    def __init__(self, values: Mapping[str, object]):
        self._values = dict(values)

    def __getitem__(self, key):
        if key in self._values:
            return self._values[key]
        # Legacy run_oracle records these two return-only fields. Null means
        # not supplied; no DiT seed or step value is synthesized here.
        if key in ("seed", "steps"):
            return None
        raise KeyError(key)

    def __iter__(self):
        return iter(self._values)

    def __len__(self):
        return len(self._values)


def _validate_inputs_and_preflight(receipt: Mapping, diagnostics: Mapping, inputs_path: Path,
                                   inputs_sha256: str, same_latent):
    bundle = receipt["decoder_inputs"]
    if str(_canonical_file(bundle["manifest_path"], "receipt decoder input manifest")) != str(
            _canonical_file(inputs_path, "decoder input manifest")):
        raise ReferenceDecodeError("caller decoder input path differs from native receipt")
    if bundle["manifest_sha256"] != inputs_sha256:
        raise ReferenceDecodeError("caller decoder input hash differs from native receipt")
    receipt_view = _DecoderOnlyReceiptView({
        "inputs": str(inputs_path),
        "inputs_sha256": inputs_sha256,
        "prompt": bundle["prompt"],
        "precision": receipt["precision"],
    })
    input_document, input_payload_path, noise_record = same_latent._validate_inputs(
        inputs_path, inputs_sha256, receipt_view
    )
    if (input_document.get("payload_sha256") != bundle["payload_sha256"]
            or noise_record.get("name") not in (None, INPUT_NOISE_SHA_FIELD)
            or noise_record.get("dtype") != "F32"
            or noise_record.get("shape") != list(PIXEL_SHAPE)
            or noise_record.get("nbytes") != PIXEL_F32_BYTES
            or noise_record.get("sha256") != bundle["noise"]["sha256"]):
        raise ReferenceDecodeError("decoder noise record differs from its native receipt binding")
    geometry = same_latent._resolve_geometry(diagnostics, input_document)
    artifact_admission = same_latent._preflight_artifacts(
        receipt=receipt_view,
        diagnostics=diagnostics,
        input_document=input_document,
        input_payload_path=input_payload_path,
        geometry=geometry,
    )
    return receipt_view, input_document, input_payload_path, noise_record, geometry, artifact_admission


def _write_json_fresh(path: Path, document: Mapping) -> None:
    try:
        with path.open("x", encoding="utf-8") as stream:
            json.dump(document, stream, indent=2)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
    except OSError as error:
        raise ReferenceDecodeError(f"cannot write decoder oracle receipt: {error}") from error


def run_oracle(*, receipt: Mapping, native_receipt_sha256: str, diagnostics: Mapping,
               inputs_path: Path, inputs_sha256: str, output: Path) -> dict:
    """Run existing pinned source-MPS same-latent functions for this receipt."""
    same_latent = _same_latent_module()
    same_latent._validate_execution_runtime(same_latent.DEFAULT_GEOMETRY)
    validated = validate_native_receipt_document(
        receipt, expected_inputs_sha256=inputs_sha256,
    )
    source_lineage = _authenticate_dcae_lineage(
        receipt, _canonical_file(receipt["source_latent"]["path"], "saved DCAE latent")
    )
    for path_key, hash_key, byte_count in (
        ("latent", "latent_sha256", LATENT_F32_BYTES),
        ("raw_pixels", "raw_pixels_sha256", PIXEL_F32_BYTES),
        ("display_pixels", "display_pixels_sha256", PIXEL_F32_BYTES),
    ):
        _regular_artifact(validated[path_key], f"native {path_key}", validated[hash_key], byte_count)

    receipt_view, input_document, input_payload_path, noise_record, geometry, admission = (
        _validate_inputs_and_preflight(receipt, validated, inputs_path, inputs_sha256, same_latent)
    )
    output_json, output_raw, output_first, output_display = _output_paths(output)
    internal_json = output_json.with_name(output_json.stem + ".same-latent-internal.json")
    legacy_result = same_latent.run_oracle(
        receipt=receipt_view,
        native_receipt_sha256=native_receipt_sha256,
        diagnostics=validated,
        input_document=input_document,
        input_payload_path=input_payload_path,
        noise_record=noise_record,
        artifact_admission=admission,
        output_json=internal_json,
        output_raw=output_raw,
        output_first_window=output_first,
        output_display=output_display,
        geometry=geometry,
    )
    try:
        internal_json.unlink()
    except OSError as error:
        raise ReferenceDecodeError(f"cannot remove internal compatibility receipt: {error}") from error

    public_result = {
        "schema": ORACLE_SCHEMA,
        "scope": "saved DCAE-V BF16 latent through PixelUMM only; no DiT, solver, or DCAE execution",
        "upstream_revision": legacy_result["upstream_revision"],
        "upstream_model_sha256": legacy_result["upstream_model_sha256"],
        "native_receipt_sha256": native_receipt_sha256,
        "native_dcae_latent_sha256": receipt["source_latent"]["sha256"],
        "native_latent_f32_sha256": receipt["outputs"]["latent_f32"]["sha256"],
        "decoder_input_manifest_sha256": receipt["decoder_inputs"]["manifest_sha256"],
        "decoder_input_payload_sha256": receipt["decoder_inputs"]["payload_sha256"],
        "decoder_noise_sha256": receipt["decoder_inputs"]["noise"]["sha256"],
        "pixelumm_weight_identities": receipt["weights"],
        "dcae_source_lineage": source_lineage,
        "decoder_semantics": {
            "window_latent_starts": [0, 5, 10, 15, 20],
            "window_latents": 10,
            "overlap_latents": 5,
            "pixel_frame_starts": [0, 20, 40, 60, 80],
            "prediction_count_per_window": 1,
            "noise_source": "authenticated generation-input decoder.noise F32 tensor",
            "merge": "pinned upstream _TemporalBlender cosine overlap with BF16 values",
        },
        "comparison_scope": {
            "raw_pixels_before_display_normalization": True,
            "display_pixels_after_pinned_bf16_postprocess": True,
            "attention_adapter": ATTENTION_SCOPE,
            "unmodified_eager_flex_attention_parity": False,
            "runtime_dtype": "BF16",
            "atol": ATOL,
            "rtol": RTOL,
        },
        "geometry": legacy_result["geometry"],
        "memory_preflight": legacy_result["memory_preflight"],
        "decoder_cache_manifest_sha256": legacy_result["decoder_cache_manifest_sha256"],
        "decoder_cache_payload_sha256": legacy_result["decoder_cache_payload_sha256"],
        "stem_subset_sha256": legacy_result["stem_subset_sha256"],
        "head_subset_sha256": legacy_result["head_subset_sha256"],
        "head_receipt_sha256": legacy_result["head_receipt_sha256"],
        "weight_word_checks": legacy_result["weight_word_checks"],
        "metrics": legacy_result["metrics"],
        "upstream_raw_pixels": legacy_result["upstream_raw_pixels"],
        "upstream_raw_pixels_bytes": legacy_result["upstream_raw_pixels_bytes"],
        "upstream_raw_pixels_sha256": legacy_result["upstream_raw_pixels_sha256"],
        "upstream_display_pixels": legacy_result["upstream_display_pixels"],
        "upstream_display_pixels_bytes": legacy_result["upstream_display_pixels_bytes"],
        "upstream_display_pixels_sha256": legacy_result["upstream_display_pixels_sha256"],
        "first_window_raw_pixels": legacy_result["first_window_raw_pixels"],
        "first_window_raw_pixels_shape": legacy_result["first_window_raw_pixels_shape"],
        "first_window_raw_pixels_sha256": legacy_result["first_window_raw_pixels_sha256"],
    }
    _write_json_fresh(output_json, public_result)
    return public_result


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--native-receipt", type=Path, required=True)
    parser.add_argument("--native-receipt-sha256", required=True)
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--inputs-sha256", required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--admission-only", action="store_true",
                        help="authenticate artifacts and memory plan without loading models or using MPS")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    same_latent = None
    try:
        receipt_path = _canonical_file(args.native_receipt, "native decoder receipt")
        receipt, raw_receipt = _read_json(receipt_path, "native decoder receipt")
        actual_receipt_sha256 = hashlib.sha256(raw_receipt).hexdigest()
        if not _digest(args.native_receipt_sha256) or actual_receipt_sha256 != args.native_receipt_sha256:
            raise ReferenceDecodeError("native receipt SHA-256 does not match caller authorization")
        diagnostics = validate_native_receipt_document(
            receipt, expected_inputs_sha256=args.inputs_sha256,
        )
        same_latent = _same_latent_module()
        if not args.admission_only:
            # The admitted geometry is fixed to 512x288. Fail the MPS/Torch
            # gate before hashing the multi-gigabyte decoder assets.
            same_latent._validate_execution_runtime(same_latent.DEFAULT_GEOMETRY)
        source_lineage = _authenticate_dcae_lineage(
            receipt, _canonical_file(receipt["source_latent"]["path"], "saved DCAE latent")
        )
        for path_key, hash_key, byte_count in (
            ("latent", "latent_sha256", LATENT_F32_BYTES),
            ("raw_pixels", "raw_pixels_sha256", PIXEL_F32_BYTES),
            ("display_pixels", "display_pixels_sha256", PIXEL_F32_BYTES),
        ):
            _regular_artifact(diagnostics[path_key], f"native {path_key}",
                              diagnostics[hash_key], byte_count)
        receipt_view, input_document, input_payload_path, noise_record, geometry, admission = (
            _validate_inputs_and_preflight(receipt, diagnostics, args.inputs.expanduser(),
                                           args.inputs_sha256, same_latent)
        )
        if args.admission_only:
            summary = {
                "schema": ORACLE_SCHEMA,
                "admission": "passed",
                "scope": "saved DCAE-V BF16 latent through PixelUMM only",
                "native_receipt_sha256": actual_receipt_sha256,
                "native_dcae_latent_sha256": receipt["source_latent"]["sha256"],
                "decoder_noise_sha256": receipt["decoder_inputs"]["noise"]["sha256"],
                "dcae_source_lineage": source_lineage,
                "resolution": "512x288",
                "memory_preflight_source": "source-MPS same-latent preflight; no execution",
                "latent": str(admission["latent_path"]),
                "native_raw_pixels": str(admission["native_raw_path"]),
                "native_display_pixels": str(admission["native_display_path"]),
            }
            print(json.dumps(summary, indent=2))
            return 0
        if args.output is None:
            raise ReferenceDecodeError("--output is required unless --admission-only is selected")
        result = run_oracle(
            receipt=receipt,
            native_receipt_sha256=actual_receipt_sha256,
            diagnostics=diagnostics,
            inputs_path=args.inputs.expanduser(),
            inputs_sha256=args.inputs_sha256,
            output=args.output,
        )
    except (ReferenceDecodeError, OSError, RuntimeError, ValueError) as error:
        print(f"saved-latent decoder oracle failed: {error}", file=sys.stderr)
        return 2
    print(json.dumps({"output": str(args.output), "result": result["metrics"]["result"]}, indent=2))
    return 0 if result["metrics"]["result"] == "match" else 1


if __name__ == "__main__":
    raise SystemExit(main())
