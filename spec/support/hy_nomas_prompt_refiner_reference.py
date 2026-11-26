#!/usr/bin/env python3
"""Create a prompt-bound pinned-upstream Hy-Nomas text-refiner fixture.

The helper authenticates one conditioned generation_inputs v2 bundle, runs only
the two-block text refiner and timestep embedder on MPS, and compares the valid
rows of the padded and compact refiner paths before publishing. It does not
run a DiT block, the full model, or a decoder.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import math
import os
from pathlib import Path
import sys
import tempfile
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
SUPPORT = ROOT / "spec" / "support"
OUTPUT_ROOT = Path("/opt/mldata_ssd")
UPSTREAM_REVISION = "911cea6458a89c2663513ff7099561ec190dd491"
DEFAULT_UPSTREAM = Path("/opt/mldata_ssd/cogni-ml/video-scene-20261019/upstream/hy-nomas-20261024")
DEFAULT_TEXT_WEIGHTS = Path("/opt/mldata_ssd/cogni-ml/hy-nomas-text-conditioning-20261025/weights/text-time-f32.safetensors")
DEFAULT_TEXT_MANIFEST = Path(str(DEFAULT_TEXT_WEIGHTS) + ".manifest.json")
TEXT_WEIGHTS_SHA256 = "58313f6141adafc12629b46c73f1bc270aa64c3e0972a8995875ba76d9068d8f"
TEXT_MANIFEST_SHA256 = "0863d49ba53d54db5ad7ff5313099c3e93003cf304e729f4a377098f230f462f"
ATOL = 2e-5
RTOL = 2e-5
SIDE_CAR_LIMIT = 64 * 1024 * 1024
TIMESTEPS = (("1000", 1000.0), ("206.897", 0.206897 * 1000.0))
SCHEMA = "cogni_ml.hy_nomas_prompt_refiner_reference/v1"

sys.path.insert(0, str(SUPPORT))


class ReferenceError(RuntimeError):
    """An authentication, pinned-source, execution, or parity failure."""


def _digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def _publish_noreplace(path: Path, data: bytes) -> None:
    path = path.resolve()
    if not path.parent.is_dir() or path.exists() or path.is_symlink():
        raise FileExistsError(f"output must be new and its parent must exist: {path}")
    with tempfile.NamedTemporaryFile(dir=path.parent, prefix=f".{path.name}.", delete=False) as stream:
        temporary = Path(stream.name)
        stream.write(data)
        stream.flush()
        os.fsync(stream.fileno())
    try:
        os.link(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _inputs(args: argparse.Namespace) -> dict[str, Any]:
    import hy_nomas_prompt_edit_resume_reference as prompt_edit

    return prompt_edit._load_prompt_inputs({
        "manifest_path": str(args.inputs_manifest),
        "manifest_sha256": args.inputs_manifest_sha256,
        "payload_path": str(args.inputs_payload),
        "payload_sha256": args.inputs_payload_sha256,
    })


def _load_weights(torch: Any, weights_path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    import hy_nomas_text_conditioning_reference as text_reference

    text_state = text_reference._text_state(torch, weights_path)
    time_state = text_reference._time_state(torch, weights_path)
    return text_state, time_state


def _compaction_metrics(torch: Any, full: Any, compact: Any) -> tuple[Any, dict[str, Any]]:
    if tuple(full.shape) != tuple(compact.shape):
        raise ReferenceError("full and compact valid projected_text shapes differ")
    left = full.float().reshape(-1)
    right = compact.float().reshape(-1)
    delta = (left - right).abs()
    failed = (~torch.isfinite(left)) | (~torch.isfinite(right)) | (delta > ATOL + RTOL * right.abs())
    fail_count = failed.to(torch.float32).sum()
    max_abs = torch.where(torch.isfinite(delta), delta, torch.full_like(delta, float("inf"))).max()
    index_range = torch.arange(delta.numel(), device=delta.device, dtype=torch.int64)
    first = torch.where(failed, index_range, torch.full_like(index_range, delta.numel())).min()
    first = torch.where(fail_count > 0, first, torch.full_like(first, -1))
    safe_index = first.clamp(min=0)
    metric = torch.stack((max_abs, fail_count, first.to(torch.float32), left[safe_index], right[safe_index]))
    return metric, {"shape": list(compact.shape), "elements": int(compact.numel())}


def produce(args: argparse.Namespace) -> dict[str, Any]:
    if os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK") not in (None, "0"):
        raise ReferenceError("PYTORCH_ENABLE_MPS_FALLBACK must be unset or 0")
    os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "0"

    output = args.output.resolve()
    if output.suffix != ".json" or not output.parent.resolve().is_relative_to(OUTPUT_ROOT.resolve()):
        raise ReferenceError(f"output must be a JSON file below {OUTPUT_ROOT}")
    sidecar_path = output.with_suffix(".f32le")
    if output.exists() or output.is_symlink() or sidecar_path.exists() or sidecar_path.is_symlink():
        raise ReferenceError("reference JSON and sidecar paths must be new")

    input_record = _inputs(args)
    document = input_record["document"]
    source = document["qwen_source"]
    valid_tokens = input_record["valid_text_tokens"]
    if valid_tokens not in (557, 585):
        raise ReferenceError(f"this focused real-prompt oracle admits 557 or 585 visible tokens, got {valid_tokens}")

    import hy_nomas_text_conditioning_reference as text_reference

    weights_path = args.text_weights.resolve(strict=True)
    if not weights_path.is_file() or weights_path.is_symlink() or _digest(weights_path) != TEXT_WEIGHTS_SHA256:
        raise ReferenceError("text/time weights do not match the pinned F32 subset")
    text_manifest_path = args.text_manifest.resolve(strict=True)
    text_manifest = text_reference._load_trusted_manifest(
        text_manifest_path, args.text_manifest_sha256, kind="text"
    )
    if args.text_manifest_sha256 != TEXT_MANIFEST_SHA256:
        raise ReferenceError("text/time manifest does not match the pinned source receipt")
    text_reference._verify_text_manifest(text_manifest, weights_path, TEXT_WEIGHTS_SHA256)

    upstream = args.upstream.resolve(strict=True)
    from hy_nomas_reference import (
        _install_upstream_modules,
        _namespace,
        _round_parameters_bf16_in_place,
        _verify_upstream,
    )

    source_hashes = _verify_upstream(upstream)
    for relative, expected in text_reference.PINNED_SOURCE_SHA256.items():
        path = upstream / relative
        if not path.is_file() or _digest(path) != expected:
            raise ReferenceError(f"pinned upstream source mismatch: {relative}")
        source_hashes[relative] = expected

    import numpy as np
    import torch

    if not torch.backends.mps.is_built() or not torch.backends.mps.is_available():
        raise ReferenceError("MPS is required; CPU fallback is forbidden")
    device = torch.device("mps")
    if torch.get_default_dtype() != torch.float32:
        torch.set_default_dtype(torch.float32)

    sidecar = Path(source["sidecar_path"]).resolve(strict=True)
    qwen_manifest = Path(source["reference_path"]).resolve(strict=True)
    crop_start, crop_end = source["crop"]
    raw_length = source["raw_sequence_length"]
    if crop_end - crop_start != 1280 or crop_end != raw_length or source["state_index"] != 26:
        raise ReferenceError("authenticated Qwen state-26 crop geometry is unsupported")
    qwen_map = np.memmap(sidecar, mode="r", dtype="<u2", shape=(27, 1, raw_length, 3584))
    raw_rows = np.array(qwen_map[26, 0, crop_start:crop_end], copy=True)
    del qwen_map
    qwen_cpu = torch.from_numpy(raw_rows).view(torch.bfloat16)
    del raw_rows
    qwen_full = qwen_cpu.unsqueeze(0).to(device=device, dtype=torch.float32)
    del qwen_cpu

    valid_indices = list(range(valid_tokens))
    valid_mask = source["cropped_attention_mask"]
    if valid_mask != [1] * valid_tokens + [0] * (1280 - valid_tokens):
        raise ReferenceError("authenticated mask is not the expected visible prefix")
    full_mask = torch.tensor([valid_mask], dtype=torch.bool, device=device)
    compact_input = qwen_full[:, :valid_tokens].contiguous()
    compact_mask = torch.ones((1, valid_tokens), dtype=torch.bool, device=device)
    if tuple(qwen_full.shape) != (1, 1280, 3584) or tuple(compact_input.shape) != (1, valid_tokens, 3584):
        raise ReferenceError("authenticated Qwen input dimensions differ from the native refiner")

    _install_upstream_modules(upstream)
    attention_state: dict[str, Any] = {"current": None, "attention_block": 0}
    attention_module, attention = text_reference._install_text_attention(
        torch, valid_indices=valid_indices, precision="f32", capture=attention_state
    )
    attention_module.attention = attention  # type: ignore[attr-defined]
    text_dir = upstream / "hymm/models/text_encoder"
    _namespace("hymm.models.text_encoder", text_dir)
    refiner_module = importlib.import_module("hymm.models.text_encoder.token_refiner")
    embed_layers = importlib.import_module("hymm.models.dit.modules.embed_layers")

    text_state, time_state = _load_weights(torch, weights_path)
    refiner = refiner_module.SingleTokenRefinerPerBlock(
        in_channels=3584,
        hidden_size=2048,
        heads_num=16,
        depth=2,
        mlp_width_ratio=4.0,
        mlp_drop_rate=0.0,
        act_type="silu",
        qk_norm=False,
        qkv_bias=True,
        attn_mode="torch",
        seg_pos_emb_num=0,
        dtype=torch.float32,
        device=torch.device("meta"),
    )
    refiner.load_state_dict(text_state, strict=True, assign=True)
    refiner.to(device=device, dtype=torch.float32).eval()
    del text_state
    time_in = embed_layers.TimestepEmbedder(
        2048, torch.nn.SiLU, dtype=torch.float32, device=torch.device("meta")
    )
    time_in.load_state_dict(time_state, strict=True, assign=True)
    time_in.to(device=device, dtype=torch.float32).eval()
    del time_state
    _round_parameters_bf16_in_place(torch, refiner, device)
    _round_parameters_bf16_in_place(torch, time_in, device)

    output_records: dict[str, Any] = {}
    compaction_metrics: list[Any] = []
    compaction_descriptors: list[dict[str, Any]] = []
    with torch.inference_mode():
        for label, timestep_value in TIMESTEPS:
            timestep = torch.tensor([timestep_value], dtype=torch.float32, device=device)
            vec = time_in(timestep)
            full_output = refiner(qwen_full, timestep, full_mask)
            compact_output = refiner(compact_input, timestep, compact_mask)
            full_valid = full_output[:, :valid_tokens].contiguous()
            compact_output = compact_output.contiguous()
            metric, descriptor = _compaction_metrics(torch, full_valid, compact_output)
            compaction_metrics.append(metric)
            compaction_descriptors.append({"timestep": timestep_value, **descriptor})
            output_records[f"projected_text.{label}"] = full_valid
            output_records[f"vec.{label}"] = vec
            del timestep, vec, full_output, compact_output, full_valid

    names = sorted(output_records)
    flat_values: list[Any] = []
    finite_values: list[Any] = []
    tensor_metadata: dict[str, dict[str, Any]] = {}
    offset = 0
    for name in names:
        value = output_records[name]
        if value.device.type != "mps" or value.dtype != torch.float32 or not bool(value.is_contiguous()):
            raise ReferenceError(f"oracle output has an unsupported device, dtype, or layout: {name}")
        count = int(value.numel())
        nbytes = count * 4
        tensor_metadata[name] = {
            "dtype": "F32",
            "shape": list(value.shape),
            "offset_bytes": offset,
            "nbytes": nbytes,
        }
        offset += nbytes
        flat_values.append(value.reshape(-1))
        finite_values.append(torch.isfinite(value).all().to(dtype=torch.float32).reshape(1))
    if not flat_values or offset > SIDE_CAR_LIMIT:
        raise ReferenceError("refiner fixture exceeds the 64 MiB sidecar limit")

    metric_names = ["max_abs", "failures", "first_mismatch", "actual", "expected"]
    packed = torch.cat(
        flat_values
        + finite_values
        + [metric.reshape(-1).to(dtype=torch.float32) for metric in compaction_metrics]
    ).to(device="cpu", dtype=torch.float32).contiguous()
    packed_array = packed.numpy()
    finite_start = offset // 4
    finite_end = finite_start + len(finite_values)
    finite_ok = bool((packed_array[finite_start:finite_end] == 1.0).all())
    metric_values = packed_array[finite_end:].tolist()
    if len(metric_values) != len(compaction_metrics) * len(metric_names):
        raise ReferenceError("packed compaction gate metrics have the wrong extent")
    compaction_reports = []
    for index, descriptor in enumerate(compaction_descriptors):
        values = metric_values[index * 5 : (index + 1) * 5]
        max_abs, failures_f, first_f, actual, expected = values
        failures = int(failures_f)
        first = int(first_f)
        compaction_reports.append({
            **descriptor,
            "failures": failures,
            "max_abs": max_abs if math.isfinite(max_abs) else None,
            "first_mismatch": None if first < 0 else {
                "flat_index": first,
                "actual_full": actual if math.isfinite(actual) else None,
                "expected_compact": expected if math.isfinite(expected) else None,
            },
            "passed": failures == 0,
        })
    if not finite_ok or any(not row["passed"] for row in compaction_reports):
        raise ReferenceError("MPS full/compact refiner control failed")

    tensor_count = offset // 4
    raw = packed_array[:tensor_count].astype("<f4", copy=False).tobytes()
    if len(raw) != offset:
        raise ReferenceError("packed fixture byte count mismatch")
    sidecar_sha = hashlib.sha256(raw).hexdigest()
    tensor_name = sidecar_path.name
    manifest = {
        "schema": SCHEMA,
        "schema_version": 1,
        "upstream_revision": UPSTREAM_REVISION,
        "upstream_source_sha256": {
            name: source_hashes[name]
            for name in (
                "hymm/models/text_encoder/token_refiner.py",
                "hymm/models/dit/modules/embed_layers.py",
            )
        },
        "source": {
            "inputs_manifest_sha256": input_record["manifest_sha256"],
            "inputs_payload_sha256": input_record["payload_sha256"],
            "prompt_sha256": input_record["prompt_sha256"],
            "valid_tokens": valid_tokens,
            "qwen_reference_sha256": _digest(qwen_manifest),
            "qwen_sidecar_sha256": _digest(sidecar),
            "state_index": 26,
            "crop": [crop_start, crop_end],
        },
        "weights": {
            "path": str(weights_path),
            "sha256": TEXT_WEIGHTS_SHA256,
            "manifest_sha256": TEXT_MANIFEST_SHA256,
        },
        "geometry": {
            "batches": 1,
            "text_tokens": 1280,
            "valid_tokens": valid_tokens,
            "input_dim": 3584,
            "dim": 2048,
            "heads": 16,
            "head_dim": 128,
            "ffn_dim": 8192,
        },
        "precision": {
            "resident_weights": "BF16",
            "compute": "F32",
            "attention": "f32",
            "text_output_policy": "valid_prefix_only",
            "padded_slots": "native retains 1280 storage slots; only valid prefix rows are compared; no padding equality claim",
        },
        "timestep_cases": [
            {"label": label, "value": value} for label, value in TIMESTEPS
        ],
        "compaction": {
            "atol": ATOL,
            "rtol": RTOL,
            "comparisons": compaction_reports,
            "passed": True,
        },
        "tensor_file": {
            "name": tensor_name,
            "nbytes": len(raw),
            "sha256": sidecar_sha,
        },
        "tensors": tensor_metadata,
    }
    sidecar_bytes = raw
    _publish_noreplace(sidecar_path, sidecar_bytes)
    manifest_bytes = (json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode("utf-8")
    _publish_noreplace(output, manifest_bytes)
    return manifest


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs-manifest", required=True, type=Path)
    parser.add_argument("--inputs-manifest-sha256", required=True)
    parser.add_argument("--inputs-payload", required=True, type=Path)
    parser.add_argument("--inputs-payload-sha256", required=True)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--text-weights", type=Path, default=DEFAULT_TEXT_WEIGHTS)
    parser.add_argument("--text-manifest", type=Path, default=DEFAULT_TEXT_MANIFEST)
    parser.add_argument("--text-manifest-sha256", default=TEXT_MANIFEST_SHA256)
    parser.add_argument("--upstream", type=Path, default=DEFAULT_UPSTREAM)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        manifest = produce(args)
    except Exception as error:
        raise SystemExit(f"FAIL: {error}") from error
    print(json.dumps({
        "manifest": str(args.output.resolve()),
        "valid_tokens": manifest["source"]["valid_tokens"],
        "precision": manifest["precision"],
        "compaction": manifest["compaction"],
        "passed": manifest["compaction"]["passed"],
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
