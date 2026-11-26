#!/usr/bin/env python3
"""Execute pinned upstream Hy-Nomas cached attention on a native Q/K/V bank.

This is a tiny cache-contract oracle, not a DiT or model run. It AST-extracts
the pinned ``causal_attention_with_cache_mp`` function without rewriting its
body, executes the torch attention path in F32, and compares each native
forward and final dense cache against the upstream result.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
import math
import os
from pathlib import Path
import re
import subprocess
import sys
from types import ModuleType, SimpleNamespace
from typing import Dict, List, Optional

import torch
import torch.nn.functional as F


NATIVE_REPORT_SCHEMA = "cogni_ml.hy_nomas_continuation_history_probe"
REFERENCE_SCHEMA = "cogni_ml.hy_nomas_continuation_history_reference"
UPSTREAM_REVISION = "911cea6458a89c2663513ff7099561ec190dd491"
UPSTREAM = Path(
    "/opt/mldata_ssd/cogni-ml/video-scene-20261019/upstream/hy-nomas-20261024"
)
UPSTREAM_SOURCE_SHA256 = "acf827bb7feb84c9046618737146544843a3f50c3500a3b16925dadeb6c5bdee"
CACHE_STORAGE_SHA256 = "53fdb67aaa967b9195f17c1d829feb12cd1e17dfa21a4e5bc0a8492790f8f0c3"
SOURCE_RELATIVE = Path("hymm/models/dit/nomas_cpt.py")
CACHE_STORAGE_RELATIVE = Path("hymm/models/dit/modules/kv_cache_storage.py")
FORWARD_COUNT = 14
SEGMENT_COUNT = 3
LAYERS = 54
VIDEO_TOKENS = 2
TEXT_TOKENS = 1
ACTIVE_QUERY_ROWS = VIDEO_TOKENS + TEXT_TOKENS
STORAGE_ROWS = 4
HEADS = 2
HEAD_DIM = 2
MAX_CACHE_ROWS = 5
ATOL = RTOL = 2e-5
INPUT_SHAPE = (FORWARD_COUNT, LAYERS, STORAGE_ROWS, HEADS, HEAD_DIM)
OUTPUT_SHAPE = (FORWARD_COUNT, LAYERS, ACTIVE_QUERY_ROWS, HEADS, HEAD_DIM)
CACHE_SHAPE = (LAYERS, MAX_CACHE_ROWS, HEADS, HEAD_DIM)
ARTIFACT_SHAPES = {
    "query": INPUT_SHAPE,
    "key": INPUT_SHAPE,
    "value": INPUT_SHAPE,
    "output": OUTPUT_SHAPE,
    "cache_key": CACHE_SHAPE,
    "cache_value": CACHE_SHAPE,
}
ARTIFACT_NAMES = tuple(ARTIFACT_SHAPES)
REPORT_FIELDS = {
    "segments": SEGMENT_COUNT,
    "layers": LAYERS,
    "denoise_forwards_per_segment": 4,
    "writer_forwards": 2,
    "forward_count": FORWARD_COUNT,
    "video_tokens": VIDEO_TOKENS,
    "valid_text_tokens": TEXT_TOKENS,
    "active_query_rows": ACTIVE_QUERY_ROWS,
    "storage_rows": STORAGE_ROWS,
    "heads": HEADS,
    "head_dim": HEAD_DIM,
}
PROJECT_ROOT = Path(__file__).resolve().parents[2]
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
OUTPUT_NAMES = (
    "reference-output.f32le",
    "reference-cache-key.f32le",
    "reference-cache-value.f32le",
    "reference.json",
)


class ReferenceError(ValueError):
    """The native report, pinned source, or reference execution is invalid."""


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ReferenceError(f"duplicate JSON field: {key}")
        result[key] = value
    return result


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path: Path, *, max_bytes: int = 1024 * 1024) -> tuple[dict, bytes]:
    if path.is_symlink() or not path.is_file():
        raise ReferenceError("native report must be a regular non-symlink file")
    raw = path.read_bytes()
    if len(raw) > max_bytes:
        raise ReferenceError("native report exceeds the metadata size limit")
    try:
        document = json.loads(raw.decode("utf-8"), object_pairs_hook=_unique_object)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ReferenceError("cannot parse native report JSON") from error
    if type(document) is not dict:
        raise ReferenceError("native report must be a JSON object")
    return document, raw


def _resolve_report_artifact(report_path: Path, value: object, name: str) -> Path:
    if not isinstance(value, str) or not value:
        raise ReferenceError(f"{name} artifact path is malformed")
    parent = report_path.parent.resolve(strict=True)
    raw = Path(value)
    if raw.is_absolute():
        candidate = raw
        if candidate.parent.resolve(strict=True) != parent:
            raise ReferenceError(f"{name} artifact must be beside the native report")
    else:
        if raw.name != value or value in {".", ".."}:
            raise ReferenceError(f"{name} artifact path must be a sibling basename")
        candidate = parent / raw
    if candidate.is_symlink() or not candidate.is_file():
        raise ReferenceError(f"{name} artifact must be a regular non-symlink file")
    try:
        resolved = candidate.resolve(strict=True)
    except (OSError, RuntimeError) as error:
        raise ReferenceError(f"cannot resolve {name} artifact") from error
    if resolved.parent != parent:
        raise ReferenceError(f"{name} artifact must remain beside the native report")
    return resolved


def validate_native_report_document(document: dict, report_path: Path) -> dict:
    """Authenticate report geometry, sibling paths, exact extents, and hashes."""
    if type(document) is not dict or document.get("schema") != NATIVE_REPORT_SCHEMA:
        raise ReferenceError("unsupported native report schema")
    if type(document.get("schema_version")) is not int or document["schema_version"] != 2:
        raise ReferenceError("unsupported native report schema_version")
    for field, expected in REPORT_FIELDS.items():
        if type(document.get(field)) is not int or document[field] != expected:
            raise ReferenceError(f"native report {field} does not match the admitted geometry")
    if document.get("text_cache_policy") != "once_per_scene":
        raise ReferenceError("native report text_cache_policy must be once_per_scene")

    artifacts = document.get("artifacts")
    if type(artifacts) is not dict or set(artifacts) != set(ARTIFACT_NAMES):
        raise ReferenceError("native report artifacts must name exactly the six required banks")
    resolved_artifacts = {}
    for name, shape in ARTIFACT_SHAPES.items():
        record = artifacts[name]
        if type(record) is not dict or set(record) != {"path", "sha256", "shape"}:
            raise ReferenceError(f"{name} artifact record fields are malformed")
        if record["shape"] != list(shape) or any(type(dim) is not int for dim in record["shape"]):
            raise ReferenceError(f"{name} artifact shape does not match the admitted geometry")
        expected_sha = record["sha256"]
        if not isinstance(expected_sha, str) or not SHA256_RE.fullmatch(expected_sha):
            raise ReferenceError(f"{name} artifact SHA-256 is malformed")
        path = _resolve_report_artifact(report_path, record["path"], name)
        expected_bytes = math.prod(shape) * 4
        if path.stat().st_size != expected_bytes:
            raise ReferenceError(f"{name} artifact byte extent does not match its shape")
        if _sha256_file(path) != expected_sha:
            raise ReferenceError(f"{name} artifact SHA-256 mismatch")
        resolved_artifacts[name] = {
            "path": path,
            "sha256": expected_sha,
            "shape": shape,
            "nbytes": expected_bytes,
        }
    return {"artifacts": resolved_artifacts}


def _validate_pinned_upstream(upstream: Path) -> tuple[Path, Path, dict[str, str]]:
    if upstream.is_symlink() or not upstream.is_dir():
        raise ReferenceError("pinned upstream directory is missing or a symlink")
    root = upstream.resolve(strict=True)
    try:
        revision = subprocess.run(
            ["git", "-C", str(root), "rev-parse", "HEAD"],
            check=True, capture_output=True, text=True, timeout=10,
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError) as error:
        raise ReferenceError("cannot read pinned upstream revision") from error
    if revision != UPSTREAM_REVISION:
        raise ReferenceError("upstream checkout revision differs from the admitted pin")
    source = root / SOURCE_RELATIVE
    cache_storage = root / CACHE_STORAGE_RELATIVE
    for path in (source, cache_storage):
        if path.is_symlink() or not path.is_file():
            raise ReferenceError("pinned upstream source file is missing or a symlink")
    source_hash = _sha256_file(source)
    cache_hash = _sha256_file(cache_storage)
    if source_hash != UPSTREAM_SOURCE_SHA256:
        raise ReferenceError("pinned causal attention source digest changed")
    if cache_hash != CACHE_STORAGE_SHA256:
        raise ReferenceError("pinned dense cache storage source digest changed")
    return source, cache_storage, {
        "revision": revision,
        "source_sha256": source_hash,
        "cache_storage_sha256": cache_hash,
    }


def _load_actual_cache_storage(module_name: str, cache_path: Path):
    """Load pinned cache_format/cache_geometry under an isolated package name."""
    package_name = module_name.rsplit(".", 1)[0]
    package_names = [
        module_name.split(".")[0],
        ".".join(module_name.split(".")[:2]),
        ".".join(module_name.split(".")[:3]),
        package_name,
    ]
    for name in package_names:
        if name not in sys.modules:
            package = ModuleType(name)
            package.__path__ = []
            sys.modules[name] = package
    existing = sys.modules.get(module_name)
    if existing is not None:
        return existing
    try:
        from importlib.util import module_from_spec, spec_from_file_location

        spec = spec_from_file_location(module_name, cache_path)
        if spec is None or spec.loader is None:
            raise ReferenceError("cannot load pinned dense cache storage module")
        module = module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)
    except (ImportError, OSError, ValueError) as error:
        raise ReferenceError("cannot import pinned dense cache storage helpers") from error
    return module


def load_upstream_attention(upstream: Path = UPSTREAM):
    """Compile the pinned function's original AST node in a constrained namespace."""
    source_path, cache_path, metadata = _validate_pinned_upstream(Path(upstream))
    try:
        tree = ast.parse(source_path.read_text(encoding="utf-8"), filename=str(source_path))
    except (OSError, SyntaxError, UnicodeDecodeError) as error:
        raise ReferenceError("cannot parse pinned causal attention source") from error
    matches = [
        node for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name == "causal_attention_with_cache_mp"
    ]
    if len(matches) != 1 or not isinstance(matches[0], ast.FunctionDef):
        raise ReferenceError("pinned source must define exactly one synchronous attention function")

    prefix = f"_hy_nomas_cache_oracle_{UPSTREAM_REVISION[:12]}"
    package_name = f"{prefix}.hymm.models.dit"
    cache_module_name = f"{package_name}.modules.kv_cache_storage"
    cache_module = _load_actual_cache_storage(cache_module_name, cache_path)
    namespace = {
        "__name__": f"{package_name}.nomas_cpt_reference",
        "__package__": package_name,
        "torch": torch,
        "F": F,
        "Dict": Dict,
        "List": List,
        "Optional": Optional,
        "get_sequence_parallel_state": lambda: False,
        "nccl_info": SimpleNamespace(sp_group=None),
        "_HAS_AR_VIS_PROFILER": False,
        "ARCHIVED_HISTORY_SAMPLINGS": frozenset(),
    }
    extracted = ast.Module(body=[matches[0]], type_ignores=[])
    try:
        code = compile(ast.fix_missing_locations(extracted), str(source_path), "exec")
        exec(code, namespace)
    except (SyntaxError, TypeError, NameError) as error:
        raise ReferenceError("cannot compile pinned causal attention AST") from error
    attention = namespace.get("causal_attention_with_cache_mp")
    if not callable(attention) or cache_module.cache_format is None:
        raise ReferenceError("pinned upstream attention or dense cache helper is unavailable")
    metadata = {
        **metadata,
        "attention_source": str(source_path),
        "cache_storage_source": str(cache_path),
        "attention_mode": "torch",
        "seg_ar_attention_backend": "magi",
        "cache_dtype": "float32",
    }
    return attention, metadata


def _forward_parameters(forward: int) -> tuple[int, str, bool, bool]:
    """Return (segment, phase, update_cache, cache_txt) for fixed 14-call order."""
    if type(forward) is not int or not 0 <= forward < FORWARD_COUNT:
        raise ReferenceError("forward index is outside the admitted schedule")
    if forward < 4:
        return 0, f"denoise{forward}", False, True
    if forward == 4:
        return 0, "writer", True, True
    if forward < 9:
        return 1, f"denoise{forward - 5}", False, False
    if forward == 9:
        return 1, "writer", True, False
    return 2, f"denoise{forward - 10}", False, False


def run_one_layer(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention,
    *,
    device: str | torch.device = "cpu",
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Run the 14-call contract for one layer using the actual pinned function."""
    expected_shape = (FORWARD_COUNT, STORAGE_ROWS)
    if any(tensor.ndim != 4 or tuple(tensor.shape[:2]) != expected_shape for tensor in (query, key, value)):
        raise ReferenceError("one-layer Q/K/V tensors must have shape [14,4,H,D]")
    if query.shape != key.shape or query.shape != value.shape:
        raise ReferenceError("one-layer Q/K/V tensor shapes differ")
    if query.shape[2] <= 0 or query.shape[3] <= 0:
        raise ReferenceError("one-layer Q/K/V head geometry is empty")
    device = torch.device(device)
    query = query.to(device=device, dtype=torch.float32)
    key = key.to(device=device, dtype=torch.float32)
    value = value.to(device=device, dtype=torch.float32)
    heads, head_dim = query.shape[2:]
    cache = {
        "k": torch.zeros((1, MAX_CACHE_ROWS, heads, head_dim), device=device, dtype=torch.float32),
        "v": torch.zeros((1, MAX_CACHE_ROWS, heads, head_dim), device=device, dtype=torch.float32),
    }
    text_mask = torch.ones((1, TEXT_TOKENS), device=device, dtype=torch.bool)
    outputs = torch.empty((FORWARD_COUNT, ACTIVE_QUERY_ROWS, heads, head_dim), device=device, dtype=torch.float32)
    valid_len = 0

    with torch.no_grad():
        for forward in range(FORWARD_COUNT):
            _, _, update_cache, cache_txt = _forward_parameters(forward)
            q = query[forward, :ACTIVE_QUERY_ROWS].unsqueeze(0)
            k = key[forward, :ACTIVE_QUERY_ROWS].unsqueeze(0)
            v = value[forward, :ACTIVE_QUERY_ROWS].unsqueeze(0)
            img_attn, txt_attn = attention(
                (q[:, :VIDEO_TOKENS], q[:, VIDEO_TOKENS:]),
                (k[:, :VIDEO_TOKENS], k[:, VIDEO_TOKENS:]),
                (v[:, :VIDEO_TOKENS], v[:, VIDEO_TOKENS:]),
                kv_cache=cache,
                cache_valid_len=valid_len,
                attn_mode="torch",
                text_mask=text_mask,
                update_cache=update_cache,
                cache_txt=cache_txt,
                seg_ar_attention_backend="magi",
            )
            outputs[forward] = torch.cat(
                (
                    img_attn.reshape(1, VIDEO_TOKENS, heads, head_dim),
                    txt_attn.reshape(1, TEXT_TOKENS, heads, head_dim),
                ),
                dim=1,
            )[0]
            if update_cache:
                valid_len += VIDEO_TOKENS + (TEXT_TOKENS if cache_txt else 0)
                if valid_len > MAX_CACHE_ROWS:
                    raise ReferenceError("pinned attention wrote beyond the admitted dense cache")
    if valid_len != MAX_CACHE_ROWS:
        raise ReferenceError("pinned attention final cache length differs from the 5-row contract")
    return outputs, cache["k"][0, :valid_len].clone(), cache["v"][0, :valid_len].clone()


def run_all_layers(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention,
    *,
    device: str | torch.device = "cpu",
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Run 54 independent layer caches; no DiT blocks or weights are loaded."""
    for name, tensor in (("query", query), ("key", key), ("value", value)):
        if tuple(tensor.shape) != INPUT_SHAPE:
            raise ReferenceError(f"{name} bank shape does not match the fixed 54-layer contract")
    device = torch.device(device)
    query = query.to(device=device, dtype=torch.float32)
    key = key.to(device=device, dtype=torch.float32)
    value = value.to(device=device, dtype=torch.float32)
    outputs = torch.empty(OUTPUT_SHAPE, device=device, dtype=torch.float32)
    cache_key = torch.empty(CACHE_SHAPE, device=device, dtype=torch.float32)
    cache_value = torch.empty(CACHE_SHAPE, device=device, dtype=torch.float32)
    for layer in range(LAYERS):
        layer_output, layer_key, layer_value = run_one_layer(
            query[:, layer], key[:, layer], value[:, layer], attention, device=device
        )
        outputs[:, layer] = layer_output
        cache_key[layer] = layer_key
        cache_value[layer] = layer_value
    return outputs, cache_key, cache_value


def _load_bank(record: dict) -> torch.Tensor:
    path = record["path"]
    expected_bytes = record["nbytes"]
    payload = path.read_bytes()
    if len(payload) != expected_bytes or _sha256_bytes(payload) != record["sha256"]:
        raise ReferenceError(f"{path.name} changed after native-report admission")
    if sys.byteorder != "little":
        raise ReferenceError("native little-endian F32 banks require a little-endian host")
    tensor = torch.frombuffer(bytearray(payload), dtype=torch.float32).view(record["shape"])
    if not bool(torch.isfinite(tensor).all().item()):
        raise ReferenceError(f"{path.name} contains non-finite values")
    return tensor


def _comparison(reference: torch.Tensor, native: torch.Tensor) -> dict[str, object]:
    reference = reference.detach().to(device="cpu", dtype=torch.float32).contiguous()
    native = native.detach().to(device="cpu", dtype=torch.float32).contiguous()
    if reference.shape != native.shape:
        raise ReferenceError("native and reference comparison shapes differ")
    if not bool(torch.isfinite(reference).all().item()) or not bool(torch.isfinite(native).all().item()):
        raise ReferenceError("native/reference comparison contains non-finite values")
    delta = (reference - native).abs()
    bound = ATOL + RTOL * reference.abs()
    failed = delta > bound
    relative = delta / reference.abs().clamp_min(1e-12)
    return {
        "shape": list(reference.shape),
        "value_count": reference.numel(),
        "failed_value_count": int(failed.sum().item()),
        "max_absolute_error": float(delta.max().item()) if delta.numel() else 0.0,
        "max_relative_error": float(relative.max().item()) if relative.numel() else 0.0,
        "atol": ATOL,
        "rtol": RTOL,
        "passed": not bool(failed.any().item()),
    }


def _prepare_output_dir(path_value: str | Path) -> Path:
    path = Path(path_value).expanduser()
    if not path.is_absolute() or path.is_symlink():
        raise ReferenceError("output directory must be an absolute non-symlink path")
    parent = path.parent.resolve(strict=True)
    target = parent / path.name
    if target == PROJECT_ROOT or PROJECT_ROOT in target.parents:
        raise ReferenceError("reference outputs must be outside the cogni-ml checkout")
    if target.exists():
        if not target.is_dir() or any(target.iterdir()):
            raise ReferenceError("output directory must be fresh or empty")
    else:
        target.mkdir()
    for name in OUTPUT_NAMES:
        if (target / name).exists() or (target / name).is_symlink():
            raise ReferenceError("refusing to replace an existing reference artifact")
    return target


def _write_raw_f32(path: Path, tensor: torch.Tensor) -> dict[str, object]:
    payload = tensor.detach().to(device="cpu", dtype=torch.float32).contiguous().numpy()
    payload = payload.astype("<f4", copy=False).tobytes(order="C")
    try:
        with path.open("xb") as stream:
            stream.write(payload)
    except FileExistsError as error:
        raise ReferenceError(f"refusing to replace {path.name}") from error
    return {
        "path": path.name,
        "sha256": _sha256_bytes(payload),
        "shape": list(tensor.shape),
        "dtype": "F32LE",
        "nbytes": len(payload),
    }


def _mps_device() -> torch.device:
    if not torch.backends.mps.is_available():
        raise ReferenceError("PyTorch MPS is unavailable; CPU fallback is prohibited for CLI runs")
    if os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK", "").strip().lower() in {
        "1", "true", "yes", "on",
    }:
        raise ReferenceError("PYTORCH_ENABLE_MPS_FALLBACK must be disabled")
    return torch.device("mps")


def run_reference(
    native_report: str | Path,
    output_dir: str | Path,
    *,
    upstream: Path = UPSTREAM,
) -> dict[str, object]:
    """Authenticate the native bank, execute the pinned F32 MPS path, compare, save."""
    report_path = Path(native_report).expanduser()
    if not report_path.is_absolute() or report_path.is_symlink() or not report_path.is_file():
        raise ReferenceError("--native-report must be an absolute regular non-symlink file")
    report_path = report_path.resolve(strict=True)
    document, report_raw = _read_json(report_path)
    admitted = validate_native_report_document(document, report_path)
    banks = {name: _load_bank(record) for name, record in admitted["artifacts"].items()}
    attention, source = load_upstream_attention(upstream)
    device = _mps_device()
    destination = _prepare_output_dir(output_dir)

    with torch.no_grad():
        reference_output, reference_key, reference_value = run_all_layers(
            banks["query"], banks["key"], banks["value"], attention, device=device
        )
        torch.mps.synchronize()
    output_metrics = _comparison(reference_output, banks["output"])
    cache_key_metrics = _comparison(reference_key, banks["cache_key"])
    cache_value_metrics = _comparison(reference_value, banks["cache_value"])
    per_forward = [
        _comparison(reference_output[index], banks["output"][index])
        for index in range(FORWARD_COUNT)
    ]
    artifacts = {
        "output": _write_raw_f32(destination / OUTPUT_NAMES[0], reference_output),
        "cache_key": _write_raw_f32(destination / OUTPUT_NAMES[1], reference_key),
        "cache_value": _write_raw_f32(destination / OUTPUT_NAMES[2], reference_value),
    }
    receipt = {
        "schema": REFERENCE_SCHEMA,
        "schema_version": 1,
        "native_report": str(report_path),
        "native_report_sha256": _sha256_bytes(report_raw),
        "native_artifact_sha256": {
            name: record["sha256"] for name, record in admitted["artifacts"].items()
        },
        "upstream": source,
        "execution": {
            "device": str(device),
            "torch_version": torch.__version__,
            "dtype": "float32",
            "attention_mode": "torch SDPA",
            "seg_ar_attention_backend": "magi (cache policy only; no Magi kernel invoked)",
            "text_mask": "all valid",
            "cache_storage": "dense F32",
            "sequence": [
                {
                    "forward": index,
                    "segment": _forward_parameters(index)[0],
                    "phase": _forward_parameters(index)[1],
                    "update_cache": _forward_parameters(index)[2],
                    "cache_txt": _forward_parameters(index)[3],
                }
                for index in range(FORWARD_COUNT)
            ],
            "parity_scope": (
                "Pinned upstream causal_attention_with_cache_mp function body, "
                "F32 Q/K/V and dense F32 cache, PyTorch SDPA on MPS; this does "
                "not establish BF16/Flash/Magi-kernel numerical parity."
            ),
        },
        "comparison": {
            "tolerance": {"atol": ATOL, "rtol": RTOL},
            "output": output_metrics,
            "cache_key": cache_key_metrics,
            "cache_value": cache_value_metrics,
            "per_forward_output": per_forward,
            "passed": all((
                output_metrics["passed"], cache_key_metrics["passed"],
                cache_value_metrics["passed"],
            )),
        },
        "artifacts": artifacts,
    }
    receipt_path = destination / OUTPUT_NAMES[3]
    try:
        with receipt_path.open("x", encoding="utf-8") as stream:
            json.dump(receipt, stream, indent=2, sort_keys=True, allow_nan=False)
            stream.write("\n")
    except FileExistsError as error:
        raise ReferenceError("refusing to replace the reference receipt") from error
    return receipt


def _main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--native-report", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--upstream-dir", type=Path, default=UPSTREAM)
    args = parser.parse_args(argv)
    try:
        receipt = run_reference(args.native_report, args.output_dir, upstream=args.upstream_dir)
    except (ReferenceError, OSError, RuntimeError) as error:
        print(f"continuation reference rejected: {error}", file=sys.stderr)
        return 2
    print(json.dumps({
        "passed": receipt["comparison"]["passed"],
        "receipt": str(args.output_dir / OUTPUT_NAMES[3]),
        "output": receipt["comparison"]["output"],
        "cache_key": receipt["comparison"]["cache_key"],
        "cache_value": receipt["comparison"]["cache_value"],
    }, sort_keys=True))
    return 0 if receipt["comparison"]["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(_main())
