#!/usr/bin/env python3
"""Build and compare one pinned Torch-MPS BF16 Hy-Nomas patch projection.

This is an arithmetic diagnostic for the first authenticated latent segment.
It does not run the DiT, refine text, decode video, or establish perceptual
quality. The native probe rounds its F32 input in-place with an explicit Metal
BF16 RNE kernel and uses the production patch token transpose.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import sys
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
SUPPORT = ROOT / "spec" / "support"
if str(SUPPORT) not in sys.path:
    sys.path.insert(0, str(SUPPORT))

import hy_nomas_real_attention_reference as capture_reference  # noqa: E402
import hy_nomas_reference as pinned_reference  # noqa: E402


SCHEMA = "cogni_ml.hy_nomas_bf16_patch_case/v1"
NATIVE_SCHEMA = "cogni_ml.hy_nomas_bf16_patch_capture/v1"
PINNED_REVISION = "911cea6458a89c2663513ff7099561ec190dd491"
CHANNELS, FRAMES, HEIGHT, WIDTH, DIM = 32, 30, 9, 16, 2048
ROWS = FRAMES * HEIGHT * WIDTH
INPUT_SHAPE = (1, CHANNELS, FRAMES, HEIGHT, WIDTH)
OUTPUT_SHAPE = (1, ROWS, DIM)
INPUT_ELEMENTS = math.prod(INPUT_SHAPE)
OUTPUT_ELEMENTS = math.prod(OUTPUT_SHAPE)
INPUT_BYTES = INPUT_ELEMENTS * 4
OUTPUT_BF16_BYTES = OUTPUT_ELEMENTS * 2
OUTPUT_F32_BYTES = OUTPUT_ELEMENTS * 4
ATOL = 2e-5
RTOL = 2e-5
OPERATION = (
    "pinned PatchEmbed.forward with BF16 input and resident BF16 pointwise "
    "Conv3d weights; flatten/transpose [T,H,W]"
)
HEX = set("0123456789abcdef")


class ReferenceError(RuntimeError):
    """A fail-closed capture, source, or comparison error."""


def _digest(value: Any, label: str) -> str:
    if (not isinstance(value, str) or len(value) != 64
            or any(char not in HEX for char in value)):
        raise ReferenceError(f"{label} must be a lowercase SHA-256 digest")
    return value


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _regular_file(path_arg: str | Path, label: str) -> Path:
    path = Path(path_arg).expanduser()
    if not path.is_absolute():
        path = Path.cwd() / path
    try:
        resolved = path.resolve(strict=True)
    except OSError as error:
        raise ReferenceError(f"{label} does not exist: {path}") from error
    if path.is_symlink() or not resolved.is_file():
        raise ReferenceError(f"{label} must be a regular non-symlink file: {path}")
    return resolved


def _read_json(path: Path, label: str) -> dict[str, Any]:
    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ReferenceError(f"duplicate JSON key in {label}: {key}")
            result[key] = value
        return result

    try:
        value = json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=unique)
    except ReferenceError:
        raise
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, RecursionError) as error:
        raise ReferenceError(f"invalid {label} JSON") from error
    if type(value) is not dict:
        raise ReferenceError(f"{label} root must be an object")
    return value


def _publish_noreplace(path: Path, data: bytes, label: str) -> None:
    try:
        with path.open("xb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
    except FileExistsError as error:
        raise ReferenceError(f"refusing to replace existing {label}: {path}") from error


def _read_capture_latent(capture: dict[str, Any], payload: Path) -> bytes:
    record = capture["tensors"]["input.latent"]
    if (record["dtype"] != "F32" or record["shape"] != [INPUT_ELEMENTS]
            or type(record["offset"]) is not int or record["offset"] < 0
            or record["nbytes"] != INPUT_BYTES):
        raise ReferenceError("authenticated input.latent descriptor differs from first-segment F32 geometry")
    with payload.open("rb") as stream:
        stream.seek(record["offset"])
        data = stream.read(record["nbytes"])
    if len(data) != INPUT_BYTES or _sha256_bytes(data) != record["sha256"]:
        raise ReferenceError("authenticated input.latent bytes differ from their capture descriptor")
    return data


def _load_patch(torch: Any, upstream: Path, global_weights: Path) -> Any:
    try:
        from safetensors import safe_open
    except ImportError as error:
        raise ReferenceError("safetensors is required for pinned BF16 globals") from error

    # Import only the actual pinned PatchEmbed class. Building the wider
    # prelude also imports SingleTokenRefinerPerBlock, which pulls in the
    # optional FlashAttention package even though this probe executes no text
    # refiner or attention.
    import importlib

    package_paths = {
        "hymm": upstream / "hymm",
        "hymm.models": upstream / "hymm" / "models",
        "hymm.models.dit": upstream / "hymm" / "models" / "dit",
        "hymm.models.dit.modules": upstream / "hymm" / "models" / "dit" / "modules",
        "hymm.utils": upstream / "hymm" / "utils",
    }
    for name, directory in package_paths.items():
        pinned_reference._namespace(name, directory)
    embed_layers = importlib.import_module("hymm.models.dit.modules.embed_layers")
    patch = embed_layers.PatchEmbed(
        patch_size=(1, 1, 1), in_chans=CHANNELS, embed_dim=DIM,
        dtype=torch.bfloat16, device=torch.device("meta"),
    )
    expected = {
        "img_in.proj.weight": (DIM, CHANNELS, 1, 1, 1),
        "img_in.proj.bias": (DIM,),
    }
    state = {}
    with safe_open(str(global_weights), framework="pt", device="cpu") as source:
        for source_name, shape in expected.items():
            tensor = source.get_tensor(source_name)
            if tensor.dtype != torch.bfloat16 or tuple(tensor.shape) != shape:
                raise ReferenceError(f"pinned patch checkpoint dtype/shape mismatch: {source_name}")
            state[source_name.removeprefix("img_in.")] = tensor.contiguous()
    if set(state) != {"proj.weight", "proj.bias"}:
        raise ReferenceError("pinned patch checkpoint binding inventory mismatch")
    if set(patch.state_dict()) != set(state):
        raise ReferenceError("pinned PatchEmbed state inventory differs from checkpoint binding")
    patch.load_state_dict(state, strict=True, assign=True)
    patch.to(device=torch.device("mps"), dtype=torch.bfloat16).eval()
    return patch


def build_case(args: argparse.Namespace) -> dict[str, Any]:
    output_parent = args.output.expanduser().parent.resolve(strict=True)
    args.output = output_parent / args.output.name
    payload_path = args.output.with_suffix(".tensors.bin")
    if args.output.suffix.lower() != ".json" or args.output.exists() or args.output.is_symlink():
        raise ReferenceError("--output must name a new .json file")
    if payload_path.exists() or payload_path.is_symlink():
        raise ReferenceError("refusing to replace an existing patch case payload")

    capture_path = _regular_file(args.capture, "native first-step capture")
    capture_sha = _digest(args.capture_sha256, "capture manifest SHA-256")
    capture = _read_json(capture_path, "native first-step capture")
    capture_payload, _capture_names = capture_reference._validate_flat_capture(
        capture, capture_path, capture_sha,
    )
    source_info, _prelude_inputs = capture_reference._authenticate_sources(capture)
    capture_reference._validate_capture_aliases(capture, capture_payload, source_info)

    upstream = args.upstream.resolve(strict=True)
    upstream_hashes = pinned_reference._verify_upstream(upstream)
    prelude_hashes = capture_reference._verify_prelude_sources(upstream)
    fallback = os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK")
    if fallback not in (None, "0"):
        raise ReferenceError("PYTORCH_ENABLE_MPS_FALLBACK must be unset or 0")
    os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "0"
    try:
        import numpy as np
        import torch
    except ImportError as error:
        raise ReferenceError("NumPy and PyTorch are required to build the Torch-MPS case") from error
    if not torch.backends.mps.is_available():
        raise ReferenceError("PyTorch MPS is unavailable; refusing a CPU fallback")

    latent_bytes = _read_capture_latent(capture, capture_payload)
    latent_words = np.frombuffer(latent_bytes, dtype="<u4")
    latent_values = np.frombuffer(latent_bytes, dtype="<f4")
    if not np.isfinite(latent_values).all():
        raise ReferenceError("authenticated first-step latent contains nonfinite F32 values")
    lsb = (latent_words >> 16) & 1
    bf16_rounded_words = (latent_words.astype(np.uint64) + 0x7FFF + lsb) & 0xFFFF0000
    changed_count = int(np.count_nonzero(latent_words.astype(np.uint64) != bf16_rounded_words))
    nonzero_low16_count = int(np.count_nonzero(latent_words & 0xFFFF))

    latent_np = np.frombuffer(latent_bytes, dtype="<f4").copy().reshape(INPUT_SHAPE)
    latent = torch.from_numpy(latent_np).to(device="mps", dtype=torch.bfloat16)
    patch = _load_patch(torch, upstream, Path(source_info["global_weights"]))
    with torch.inference_mode():
        expected = patch(latent)
    if (expected.device.type != "mps" or expected.dtype != torch.bfloat16
            or tuple(expected.shape) != OUTPUT_SHAPE):
        raise ReferenceError("pinned PatchEmbed.forward returned unexpected device, dtype, or geometry")
    expected_cpu = expected.detach().contiguous().cpu()
    expected_words = expected_cpu.view(torch.uint16).numpy()
    expected_bytes = expected_words.astype("<u2", copy=False).tobytes(order="C")
    if len(expected_bytes) != OUTPUT_BF16_BYTES:
        raise ReferenceError("pinned MPS patch result has an unexpected physical BF16 extent")

    payload_bytes = latent_bytes + expected_bytes
    _publish_noreplace(payload_path, payload_bytes, "patch case tensor payload")
    records = {}
    for name, dtype, shape, offset, data in (
        ("input.latent", "F32", list(INPUT_SHAPE), 0, latent_bytes),
        ("patch.expected_bf16", "BF16", list(OUTPUT_SHAPE), INPUT_BYTES, expected_bytes),
    ):
        records[name] = {
            "dtype": dtype,
            "shape": shape,
            "offset": offset,
            "nbytes": len(data),
            "sha256": _sha256_bytes(data),
        }

    document = {
        "schema": SCHEMA,
        "upstream_revision": PINNED_REVISION,
        "upstream_source_sha256": upstream_hashes,
        "prelude_source_sha256": prelude_hashes,
        "operation": OPERATION,
        "torch_device": "mps",
        "torch_input_dtype": "bfloat16",
        "torch_output_dtype": "bfloat16",
        "capture_manifest": str(capture_path),
        "capture_manifest_sha256": capture_sha,
        "capture_payload_sha256": capture["payload_sha256"],
        "capture_latent_sha256": capture["tensors"]["input.latent"]["sha256"],
        "weight_index_sha256": source_info["cache_index_sha256"],
        "global_weights_sha256": source_info["global_weights_sha256"],
        "geometry": {
            "input_shape": list(INPUT_SHAPE), "rows": ROWS,
            "input_width": CHANNELS, "output_width": DIM,
            "patch_size": [1, 1, 1], "token_order": "T,H,W",
        },
        "weights": {
            "weight_name": "img_in.proj.weight",
            "weight_shape": [DIM, CHANNELS, 1, 1, 1],
            "bias_name": "img_in.proj.bias",
            "bias_shape": [DIM],
        },
        "input_rounding": "torch F32-to-BF16 conversion on MPS before pinned PatchEmbed.forward",
        "input_nonzero_low16_count": nonzero_low16_count,
        "input_rounding_changed_count": changed_count,
        "named_buffer_budget_bytes": 128 * 1024 * 1024,
        "payload": payload_path.name,
        "payload_bytes": len(payload_bytes),
        "payload_sha256": _sha256_bytes(payload_bytes),
        "tensors": records,
    }
    manifest_bytes = (json.dumps(document, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()
    _publish_noreplace(args.output, manifest_bytes, "patch case manifest")
    return {
        "case_manifest": str(args.output),
        "case_manifest_sha256": _sha256_bytes(manifest_bytes),
        "payload": str(payload_path),
        "payload_bytes": len(payload_bytes),
        "operation": OPERATION,
        "device": "mps",
        "input_nonzero_low16_count": nonzero_low16_count,
        "input_rounding_changed_count": changed_count,
        "output_dtype": "bfloat16",
    }


def _record_bytes(doc: dict[str, Any], payload: Path, name: str, dtype: str,
                  shape: list[int], offset: int, nbytes: int) -> bytes:
    record = doc.get("tensors", {}).get(name)
    if (type(record) is not dict or record.get("dtype") != dtype
            or record.get("shape") != shape or type(record.get("offset")) is not int
            or record["offset"] != offset or type(record.get("nbytes")) is not int
            or record["nbytes"] != nbytes):
        raise ReferenceError(f"{name} descriptor mismatch")
    with payload.open("rb") as stream:
        stream.seek(offset)
        data = stream.read(nbytes)
    if len(data) != nbytes or _sha256_bytes(data) != _digest(record.get("sha256"), f"{name} SHA-256"):
        raise ReferenceError(f"{name} payload extent/digest mismatch")
    return data


def _load_case(case_path: Path, case_sha: str) -> tuple[dict[str, Any], Path, bytes, bytes]:
    case = _read_json(case_path, "patch case")
    if _sha256_file(case_path) != _digest(case_sha, "case manifest SHA-256"):
        raise ReferenceError("case manifest digest differs from supplied digest")
    if (case.get("schema") != SCHEMA or case.get("upstream_revision") != PINNED_REVISION
            or case.get("operation") != OPERATION or case.get("torch_device") != "mps"
            or case.get("torch_input_dtype") != "bfloat16"
            or case.get("torch_output_dtype") != "bfloat16"
            or case.get("weight_index_sha256") != capture_reference.PINNED_CACHE_INDEX_SHA256
            or case.get("geometry") != {
                "input_shape": list(INPUT_SHAPE), "rows": ROWS,
                "input_width": CHANNELS, "output_width": DIM,
                "patch_size": [1, 1, 1], "token_order": "T,H,W",
            }
            or case.get("weights") != {
                "weight_name": "img_in.proj.weight", "weight_shape": [DIM, CHANNELS, 1, 1, 1],
                "bias_name": "img_in.proj.bias", "bias_shape": [DIM],
            }):
        raise ReferenceError("patch case source/geometry contract mismatch")
    case_tensors = case.get("tensors")
    if type(case_tensors) is not dict:
        raise ReferenceError("patch case tensor inventory is malformed")
    if case.get("capture_latent_sha256") != case_tensors.get("input.latent", {}).get("sha256"):
        raise ReferenceError("patch case latent is not bound to the authenticated capture latent")
    if type(case.get("input_nonzero_low16_count")) is not int or not 0 <= case["input_nonzero_low16_count"] <= INPUT_ELEMENTS:
        raise ReferenceError("patch case low16 count is malformed")
    if type(case.get("input_rounding_changed_count")) is not int or not 0 <= case["input_rounding_changed_count"] <= INPUT_ELEMENTS:
        raise ReferenceError("patch case rounding count is malformed")
    payload_name = case.get("payload")
    if not isinstance(payload_name, str) or Path(payload_name).name != payload_name or not payload_name.endswith(".tensors.bin"):
        raise ReferenceError("patch case payload must be a sibling tensor file")
    payload = _regular_file(case_path.parent / payload_name, "patch case payload")
    if (payload.stat().st_size != INPUT_BYTES + OUTPUT_BF16_BYTES
            or case.get("payload_bytes") != INPUT_BYTES + OUTPUT_BF16_BYTES
            or _sha256_file(payload) != _digest(case.get("payload_sha256"), "case payload SHA-256")):
        raise ReferenceError("patch case payload aggregate extent/digest mismatch")
    if set(case.get("tensors", {})) != {"input.latent", "patch.expected_bf16"}:
        raise ReferenceError("patch case tensor inventory mismatch")
    latent = _record_bytes(case, payload, "input.latent", "F32", list(INPUT_SHAPE), 0, INPUT_BYTES)
    import numpy as np

    latent_words = np.frombuffer(latent, dtype="<u4")
    wide_words = latent_words.astype(np.uint64)
    expected_rounded_words = (
        (wide_words + np.uint64(0x7FFF) + ((wide_words >> np.uint64(16)) & np.uint64(1)))
        & np.uint64(0xFFFF0000)
    ).astype("<u4")
    exact_low16_count = int(np.count_nonzero(latent_words & np.uint32(0xFFFF)))
    exact_changed_count = int(np.count_nonzero(latent_words != expected_rounded_words))
    if (case["input_nonzero_low16_count"] != exact_low16_count
            or case["input_rounding_changed_count"] != exact_changed_count):
        raise ReferenceError("patch case input-rounding counts do not match authenticated latent bytes")
    expected = _record_bytes(case, payload, "patch.expected_bf16", "BF16", list(OUTPUT_SHAPE), INPUT_BYTES, OUTPUT_BF16_BYTES)
    return case, payload, latent, expected


def compare_capture(args: argparse.Namespace) -> dict[str, Any]:
    case_path = _regular_file(args.case, "patch case manifest")
    case_sha = _digest(args.case_sha256, "case manifest SHA-256")
    case, _case_payload, latent, expected_bytes = _load_case(case_path, case_sha)
    native_path = _regular_file(args.native, "native patch capture")
    native_sha = _digest(args.native_sha256, "native capture SHA-256")
    native = _read_json(native_path, "native patch capture")
    if _sha256_file(native_path) != native_sha:
        raise ReferenceError("native patch manifest digest differs from supplied digest")
    if (native.get("schema") != NATIVE_SCHEMA
            or native.get("case_manifest") != str(case_path)
            or native.get("case_manifest_sha256") != case_sha
            or native.get("case_payload_sha256") != case["payload_sha256"]
            or native.get("capture_manifest_sha256") != case["capture_manifest_sha256"]
            or native.get("weight_index_sha256") != case["weight_index_sha256"]
            or native.get("global_weights_sha256") != case["global_weights_sha256"]
            or native.get("upstream_revision") != PINNED_REVISION
            or native.get("operation") != OPERATION
            or native.get("input_rounding") != "explicit Metal BF16 RNE in-place on F32 storage before production channel-to-token transpose"
            or native.get("input_nonzero_low16_count") != case["input_nonzero_low16_count"]
            or native.get("input_rounding_changed_count") != case["input_rounding_changed_count"]
            or native.get("variant") not in ("bf16_graph_ops", "f32_accumulation_bf16_output")
            or native.get("terminal_wait_count") != 1
            or native.get("named_allocations_during_encode") != 0
            or native.get("activation_readbacks_during_encode") != 0
            or type(native.get("named_bytes")) is not int
            or native["named_bytes"] > case["named_buffer_budget_bytes"]):
        raise ReferenceError("native patch capture source or execution contract mismatch")
    geometry = native.get("geometry")
    if geometry != {"batches": 1, "frames": FRAMES, "height": HEIGHT, "width": WIDTH,
                    "rows": ROWS, "input_width": CHANNELS, "output_width": DIM}:
        raise ReferenceError("native patch capture geometry mismatch")
    payload_name = native.get("payload")
    if not isinstance(payload_name, str) or Path(payload_name).name != payload_name or not payload_name.endswith(".tensors.bin"):
        raise ReferenceError("native patch payload must be a sibling tensor file")
    payload = _regular_file(native_path.parent / payload_name, "native patch payload")
    if (payload.stat().st_size != INPUT_BYTES + OUTPUT_BF16_BYTES + OUTPUT_F32_BYTES
            or native.get("payload_bytes") != INPUT_BYTES + OUTPUT_BF16_BYTES + OUTPUT_F32_BYTES
            or _sha256_file(payload) != _digest(native.get("payload_sha256"), "native payload SHA-256")):
        raise ReferenceError("native patch payload aggregate extent/digest mismatch")
    native_tensors = native.get("tensors")
    if (type(native_tensors) is not dict
            or set(native_tensors) != {"patch.input_bf16_f32", "patch.output_bf16", "patch.output_f32"}):
        raise ReferenceError("native rounded-input/output tensor inventory mismatch")
    rounded_input = _record_bytes(native, payload, "patch.input_bf16_f32", "F32", list(INPUT_SHAPE), 0, INPUT_BYTES)
    actual_bf16 = _record_bytes(native, payload, "patch.output_bf16", "BF16", list(OUTPUT_SHAPE), INPUT_BYTES, OUTPUT_BF16_BYTES)
    actual_f32 = _record_bytes(native, payload, "patch.output_f32", "F32", list(OUTPUT_SHAPE), INPUT_BYTES + OUTPUT_BF16_BYTES, OUTPUT_F32_BYTES)

    import numpy as np

    input_words = np.frombuffer(rounded_input, dtype="<u4")
    input_source_words = np.frombuffer(latent, dtype="<u4")
    input_wide = input_source_words.astype(np.uint64)
    expected_rounded_words = (
        (input_wide + np.uint64(0x7FFF) + ((input_wide >> np.uint64(16)) & np.uint64(1)))
        & np.uint64(0xFFFF0000)
    ).astype("<u4")
    if not np.array_equal(input_words, expected_rounded_words):
        raise ReferenceError("native input is not the exact explicit BF16 RNE rounding of authenticated latent")
    if np.count_nonzero(input_words & np.uint32(0xFFFF)):
        raise ReferenceError("native BF16-rounded F32 input contains nonzero low16 bits")

    expected_words = np.frombuffer(expected_bytes, dtype="<u2")
    actual_words = np.frombuffer(actual_bf16, dtype="<u2")
    actual_f32_words = np.frombuffer(actual_f32, dtype="<u4")
    if len(expected_words) != OUTPUT_ELEMENTS or len(actual_words) != OUTPUT_ELEMENTS:
        raise ReferenceError("patch BF16 output element count mismatch")
    expanded_actual_words = actual_words.astype(np.uint32) << np.uint32(16)
    expanded_actual_f32 = actual_f32_words.view("<f4")
    if not np.array_equal(actual_f32_words, expanded_actual_words):
        raise ReferenceError("native F32 output is not the exact physical BF16 expansion")
    if np.count_nonzero(actual_f32_words & np.uint32(0xFFFF)):
        raise ReferenceError("native F32 output contains nonzero low16 bits")

    expected_nonfinite = 0
    actual_nonfinite = 0
    bad_count = 0
    bitwise_diff_count = int(np.count_nonzero(expected_words != actual_words))
    max_abs = 0.0
    sum_sq = 0.0
    rms_count = 0
    chunk = 1_000_000
    for start in range(0, OUTPUT_ELEMENTS, chunk):
        stop = min(start + chunk, OUTPUT_ELEMENTS)
        expected_f32 = (expected_words[start:stop].astype(np.uint32) << np.uint32(16)).view("<f4")
        actual = expanded_actual_f32[start:stop]
        expected_finite = np.isfinite(expected_f32)
        actual_finite = np.isfinite(actual)
        expected_nonfinite += int(np.count_nonzero(~expected_finite))
        actual_nonfinite += int(np.count_nonzero(~actual_finite))
        if not expected_finite.all() or not actual_finite.all():
            continue
        delta = np.abs(actual - expected_f32)
        limit = ATOL + RTOL * np.abs(expected_f32)
        bad_count += int(np.count_nonzero(delta > limit))
        max_abs = max(max_abs, float(delta.max(initial=0.0)))
        sum_sq += float(np.dot(delta.astype(np.float64), delta.astype(np.float64)))
        rms_count += len(delta)
    rms = math.sqrt(sum_sq / rms_count) if rms_count else math.nan
    allclose = expected_nonfinite == 0 and actual_nonfinite == 0 and bad_count == 0
    return {
        "status": "completed",
        "scope": "single patch projection from authenticated first-step latent; no DiT block, history, VAE, or perceptual assessment",
        "case_manifest": str(case_path),
        "case_manifest_sha256": case_sha,
        "native_capture": str(native_path),
        "native_capture_sha256": native_sha,
        "variant": native["variant"],
        "input_nonzero_low16_count": case["input_nonzero_low16_count"],
        "input_rounding_changed_count": case["input_rounding_changed_count"],
        "native_input_matches_exact_bf16_rne": True,
        "native_input_rounding_changed_count": int(np.count_nonzero(input_source_words != input_words)),
        "native_f32_is_exact_bf16_expansion": True,
        "native_output_low16_nonzero_count": 0,
        "torch_mps_bf16_vs_native": {
            "allclose_atol_2e-5_rtol_2e-5": allclose,
            "compared_elements": OUTPUT_ELEMENTS,
            "outside_tolerance": bad_count,
            "raw_bf16_bitwise_differences": bitwise_diff_count,
            "max_absolute_error": max_abs,
            "rms_error": rms,
            "expected_nonfinite": expected_nonfinite,
            "actual_nonfinite": actual_nonfinite,
        },
        "native_capture_contract": {
            "named_bytes": native["named_bytes"],
            "named_allocations_during_encode": native["named_allocations_during_encode"],
            "activation_readbacks_during_encode": native["activation_readbacks_during_encode"],
            "terminal_wait_count": native["terminal_wait_count"],
        },
    }


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(description=__doc__)
    subparsers = root.add_subparsers(dest="command", required=True)
    build = subparsers.add_parser("build", help="run the pinned Torch-MPS BF16 patch projection")
    build.add_argument("--capture", type=Path, required=True)
    build.add_argument("--capture-sha256", required=True)
    build.add_argument("--upstream", type=Path, default=pinned_reference.DEFAULT_UPSTREAM)
    build.add_argument("--output", type=Path, required=True)
    build.set_defaults(function=build_case)
    compare = subparsers.add_parser("compare", help="compare native physical BF16 output with Torch-MPS")
    compare.add_argument("--case", type=Path, required=True)
    compare.add_argument("--case-sha256", required=True)
    compare.add_argument("--native", type=Path, required=True)
    compare.add_argument("--native-sha256", required=True)
    compare.set_defaults(function=compare_capture)
    return root


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    try:
        result = args.function(args)
    except (ReferenceError, capture_reference.ReferenceError, RuntimeError, OSError, ValueError) as error:
        raise SystemExit(str(error)) from error
    print(json.dumps(result, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
