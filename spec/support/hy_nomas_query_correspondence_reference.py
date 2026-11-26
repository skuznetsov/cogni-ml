#!/usr/bin/env python3
"""F32 GPU affinity screening on saved BF16-expanded Hy-Nomas Q/K.

This is neither BF16 attention nor a motion controller. It tests the
source-frame self-correspondence premise motivating query-only transport
(QWERTY, arXiv:2607.01869, section 3). It does not implement that paper's
STCD, spatial warp, temporal RoPE, or latent optimization. PyTorch/MPS is
only the equation oracle; the paired diagnostic is native Metal.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
from typing import Any

import numpy as np

SHAPE = (144, 16, 128)
SCORE_SHAPE = (16, 144, 144)
METRIC_SHAPE = (16, 144, 6)
METRICS = ["diagonal", "max", "margin", "greater_count", "argmax", "self_mass"]
ATOL = RTOL = 2e-5
INPUT_SCHEMA = "hy-nomas-query-correspondence-input-v1"
SOURCE_SCHEMA = "hy-nomas-query-correspondence-source-v1"
NATIVE_SCHEMA = "hy-nomas-query-correspondence-native-v1"


def sha256(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def fresh_paths(report: Path) -> tuple[Path, Path]:
    report = report.absolute()
    payload = report.with_suffix(".f32le")
    if report == payload or any(p.exists() or p.is_symlink() for p in (report, payload)):
        raise ValueError("report and payload must not exist")
    if not report.parent.is_dir():
        raise ValueError("output parent must exist")
    return report, payload


def checked_array(value: np.ndarray, shape: tuple[int, ...], *, bf16: bool = False) -> np.ndarray:
    if value.shape != shape or value.dtype != np.dtype("<f4"):
        raise ValueError("tensor shape or F32 dtype mismatch")
    if not np.all(np.isfinite(value)):
        raise ValueError("tensor must be finite")
    value = np.ascontiguousarray(value)
    if bf16 and np.any(value.view("<u4") & 0xFFFF):
        raise ValueError("input must be lossless BF16 expansion")
    return value


def publish_input(report: Path, query: np.ndarray, key: np.ndarray,
                  provenance: dict[str, Any]) -> tuple[Path, str]:
    report, payload = fresh_paths(report)
    arrays = {"query": checked_array(query, SHAPE, bf16=True),
              "key": checked_array(key, SHAPE, bf16=True)}
    document = {
        "schema": INPUT_SCHEMA, "schema_version": 1,
        "geometry": {"heads": 16, "positions": 144, "head_dim": 128},
        "provenance": provenance, "tensors": {},
    }
    raw = bytearray()
    for name, value in arrays.items():
        data = value.tobytes()
        document["tensors"][name] = {
            "dtype": "F32", "source_dtype": "BF16", "byte_order": "little", "shape": list(SHAPE),
            "offset_bytes": len(raw), "nbytes": len(data), "sha256": sha256(data),
        }
        raw.extend(data)
    document["payload"] = {"path": payload.name, "bytes": len(raw), "sha256": sha256(raw)}
    with payload.open("xb") as stream:
        stream.write(raw)
    with report.open("x") as stream:
        stream.write(json.dumps(document, indent=2) + "\n")
    return report, file_sha256(report)


def load_input(report: Path, trusted_sha256: str) -> dict[str, Any]:
    if report.is_symlink() or not report.is_file():
        raise ValueError("input manifest must be a regular non-symlink file")
    if file_sha256(report) != trusted_sha256:
        raise ValueError("input manifest digest mismatch")
    document = json.loads(report.read_text())
    if (document.get("schema") != INPUT_SCHEMA or document.get("schema_version") != 1
            or document.get("geometry") != {"heads": 16, "positions": 144, "head_dim": 128}):
        raise ValueError("input schema or geometry mismatch")
    arrays = load_payload(report, document, {"query": SHAPE, "key": SHAPE}, bf16=True)
    return {**arrays, "document": document, "path": str(report.resolve()), "sha256": trusted_sha256}


def load_payload(report: Path, document: dict[str, Any], shapes: dict[str, tuple[int, ...]],
                 *, bf16: bool = False) -> dict[str, np.ndarray]:
    descriptor = document["payload"]
    filename = descriptor["path"]
    if type(filename) is not str or Path(filename).name != filename or filename in (".", ".."):
        raise ValueError("payload path must be a basename")
    path = report.parent / filename
    if path.is_symlink() or not path.is_file():
        raise ValueError("payload must be a regular non-symlink file")
    raw = path.read_bytes()
    if len(raw) != descriptor["bytes"] or sha256(raw) != descriptor["sha256"]:
        raise ValueError("payload length or digest mismatch")
    if set(document["tensors"]) != set(shapes):
        raise ValueError("tensor inventory mismatch")
    cursor = 0
    arrays = {}
    for name, shape in shapes.items():
        record = document["tensors"][name]
        size = math.prod(shape) * 4
        if (record["dtype"] != "F32" or record["shape"] != list(shape)
                or type(record["offset_bytes"]) is not int or record["offset_bytes"] != cursor
                or record["nbytes"] != size or (bf16 and (record.get("source_dtype") != "BF16"
                                                         or record.get("byte_order") != "little"))):
            raise ValueError("tensor range, shape or dtype mismatch")
        data = raw[cursor:cursor + size]
        if len(data) != size or sha256(data) != record["sha256"]:
            raise ValueError("tensor range digest mismatch")
        arrays[name] = checked_array(np.frombuffer(data, dtype="<f4").reshape(shape), shape, bf16=bf16)
        cursor += size
    if cursor != len(raw):
        raise ValueError("payload contains unclaimed bytes")
    return arrays


def control_arrays():
    rng = np.random.default_rng(20261106)
    query = (rng.integers(0, 2, SHAPE, dtype=np.int32) * 2 - 1).astype("<f4")
    return ((query.copy(), query.copy()),
            (query.copy(), np.roll(query, 17, axis=0)),
            (np.zeros(SHAPE, dtype="<f4"), np.zeros(SHAPE, dtype="<f4")))


def run_source(admitted: dict[str, Any], report: Path) -> Path:
    report, payload = fresh_paths(report)
    if os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK") not in (None, "0"):
        raise ValueError("MPS CPU fallback must be disabled")
    os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "0"
    import torch
    if not torch.backends.mps.is_available():
        raise ValueError("the equation oracle requires GPU MPS")
    query = torch.from_numpy(admitted["query"].copy()).to("mps")
    key = torch.from_numpy(admitted["key"].copy()).to("mps")
    with torch.inference_mode():
        scores = torch.matmul(query.permute(1, 0, 2), key.permute(1, 2, 0))
        scores = scores * (1.0 / math.sqrt(SHAPE[2]))
        diagonal = scores.diagonal(dim1=-2, dim2=-1)
        maximum, argmax = scores.max(dim=-1)
        other = scores.masked_fill(torch.eye(144, device="mps", dtype=torch.bool).unsqueeze(0), -torch.inf)
        margin = diagonal - other.max(dim=-1).values
        greater_count = (scores > diagonal.unsqueeze(-1)).sum(dim=-1).to(torch.float32)
        self_mass = torch.softmax(scores, dim=-1).diagonal(dim1=-2, dim2=-1)
        metrics = torch.stack((diagonal, maximum, margin, greater_count,
                               argmax.to(torch.float32), self_mass), dim=-1)
        packed = torch.cat((scores.flatten(), metrics.flatten()))
    # The only explicit terminal GPU drain and one joint export, after all math.
    torch.mps.synchronize()
    raw = packed.cpu().numpy().astype("<f4", copy=False).tobytes()
    outputs = {"scores": SCORE_SHAPE, "metrics": METRIC_SHAPE}
    tensors = {}
    cursor = 0
    for name, shape in outputs.items():
        size = math.prod(shape) * 4
        tensors[name] = {"dtype": "F32", "shape": list(shape), "offset_bytes": cursor,
                         "nbytes": size, "sha256": sha256(raw[cursor:cursor + size])}
        cursor += size
    document = {
        "schema": SOURCE_SCHEMA, "schema_version": 1,
        "input_manifest_sha256": admitted["sha256"], "input_manifest": admitted["path"],
        "input_payload_sha256": admitted["document"]["payload"]["sha256"],
        "input_tensors": {name: {key: admitted["document"]["tensors"][name][key]
                                  for key in ("dtype", "source_dtype", "shape", "sha256")}
                          for name in ("query", "key")},
        "metrics_order": METRICS, "tensors": tensors,
        "payload": {"path": payload.name, "bytes": len(raw), "sha256": sha256(raw)},
        "oracle": {"device": "mps", "torch_version": torch.__version__, "dtype": "F32",
                   "cpu_fallback": False,
                   "script_path": str(Path(__file__).resolve()), "script_sha256": file_sha256(Path(__file__)),
                   "equation": "QK^T/sqrt(128); no BF16 score rounding; source-frame keys only"},
        "execution": {"explicit_terminal_waits": 1, "joint_exports": 1,
                      "intermediate_readbacks": 0, "model_launched": False},
    }
    with payload.open("xb") as stream:
        stream.write(raw)
    with report.open("x") as stream:
        stream.write(json.dumps(document, indent=2) + "\n")
    return report


def compare(actual: np.ndarray, expected: np.ndarray, *, discrete_columns=()) -> dict[str, Any]:
    if actual.shape != expected.shape or not np.all(np.isfinite(actual)) or not np.all(np.isfinite(expected)):
        return {"passed": False, "reason": "shape or nonfinite value"}
    delta = np.abs(actual.astype(np.float64) - expected.astype(np.float64))
    failures = delta > ATOL + RTOL * np.abs(expected.astype(np.float64))
    for column in discrete_columns:
        failures[..., column] |= actual[..., column] != expected[..., column]
    return {"passed": not bool(np.any(failures)), "elements": actual.size,
            "failures": int(np.count_nonzero(failures)), "max_abs": float(np.max(delta)),
            "atol": ATOL, "rtol": RTOL, "exact_discrete_columns": list(discrete_columns)}


def audit_metrics(scores: np.ndarray, metrics: np.ndarray) -> dict[str, Any]:
    """Independent post-drain verification; never part of GPU execution."""
    values = checked_array(scores, SCORE_SHAPE).astype(np.float64)
    rows = np.arange(144)
    diagonal = values[:, rows, rows]
    maximum = values.max(axis=-1)
    argmax = values.argmax(axis=-1)
    other = values.copy()
    other[:, rows, rows] = -np.inf
    exponent = np.exp(values - maximum[..., None])
    expected = np.stack((diagonal, maximum, diagonal - other.max(axis=-1),
                         (values > diagonal[..., None]).sum(axis=-1), argmax,
                         exponent[:, rows, rows] / exponent.sum(axis=-1)), axis=-1)
    return compare(metrics, expected, discrete_columns=(3, 4))


def summarize(scores: np.ndarray, metrics: np.ndarray) -> dict[str, Any]:
    scores = checked_array(scores, SCORE_SHAPE)
    metrics = checked_array(metrics, METRIC_SHAPE)
    rows = np.arange(144)
    masks = {"all": np.ones(144, dtype=bool),
             "authored_alex_head_rectangle": ((rows // 16 < 4) & (rows % 16 >= 4) & (rows % 16 < 8)),
             "authored_resume_rectangle": ((rows // 16 >= 2) & (rows // 16 < 8)
                                            & (rows % 16 >= 10) & (rows % 16 < 15))}
    result = {}
    for name, mask in masks.items():
        selected = metrics[:, mask]
        result[name] = {
            "head_row_pairs": int(selected.shape[0] * selected.shape[1]),
            "strict_self_max_fraction": float(np.mean(selected[..., 2] > 0)),
            "diagonal_argmax_fraction": float(np.mean(selected[..., 4] == rows[mask][None, :])),
            "strictly_greater_count_mean": float(np.mean(selected[..., 3], dtype=np.float64)),
            "self_mass_mean": float(np.mean(selected[..., 5], dtype=np.float64)),
            "uniform_self_mass": 1 / 144,
            "margin_mean": float(np.mean(selected[..., 2], dtype=np.float64)),
        }
    return result


def validate_consumed_input(document: dict[str, Any], admitted: dict[str, Any]) -> None:
    expected = admitted["document"]
    if document.get("input_payload_sha256") != expected["payload"]["sha256"]:
        raise ValueError("consumed input payload digest mismatch")
    tensors = document.get("input_tensors", {})
    if set(tensors) != {"query", "key"}:
        raise ValueError("consumed input tensor inventory mismatch")
    for name in ("query", "key"):
        if any(tensors[name].get(key) != expected["tensors"][name][key]
               for key in ("dtype", "source_dtype", "shape", "sha256")):
            raise ValueError("consumed input tensor identity mismatch")


def qualify(input_path: Path, trusted_input: str, source_path: Path, native_path: Path,
            *, verify_model_capture: bool = False) -> dict[str, Any]:
    admitted = load_input(input_path, trusted_input)
    lineage = None
    if verify_model_capture:
        spec = importlib.util.spec_from_file_location(
            "hy_nomas_query_capture", Path(__file__).with_name("hy_nomas_query_capture.py"))
        capture = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(capture)
        lineage = capture.verify_published_input(input_path, trusted_input)
    reports = {}
    arrays = {}
    for label, path, schema in (("source", source_path, SOURCE_SCHEMA), ("native", native_path, NATIVE_SCHEMA)):
        if path.is_symlink() or not path.is_file():
            raise ValueError("output manifest must be a regular non-symlink file")
        document = json.loads(path.read_text())
        if (document.get("schema") != schema or document.get("schema_version") != 1
                or document.get("input_manifest_sha256") != trusted_input
                or document.get("metrics_order") != METRICS):
            raise ValueError("output schema, input identity or metric order mismatch")
        validate_consumed_input(document, admitted)
        arrays[label] = load_payload(path, document, {"scores": SCORE_SHAPE, "metrics": METRIC_SHAPE})
        reports[label] = {"path": str(path.resolve()), "sha256": file_sha256(path)}
        if label == "source":
            if (document["oracle"]["script_sha256"] != file_sha256(Path(__file__))
                    or document["oracle"].get("device") != "mps"
                    or document["oracle"].get("dtype") != "F32"
                    or document["oracle"].get("cpu_fallback") is not False):
                raise ValueError("source equation implementation or GPU dtype changed")
            if document.get("execution") != {
                "explicit_terminal_waits": 1, "joint_exports": 1,
                "intermediate_readbacks": 0, "model_launched": False,
            }:
                raise ValueError("source GPU execution contract failed")
        if label == "native":
            execution = document["execution"]
            if any(execution.get(name) != value for name, value in {
                "device": "metal", "named_bytes_after_setup": 3741696,
                "named_encode_allocations": 0, "encode_readbacks": 0,
                "encode_waits": 0, "terminal_waits": 1, "terminal_readbacks": 2,
            }.items()):
                raise ValueError("native chain instrumentation failed")
    comparisons = {
        "scores": compare(arrays["native"]["scores"], arrays["source"]["scores"]),
        "metrics": compare(arrays["native"]["metrics"], arrays["source"]["metrics"], discrete_columns=(3, 4)),
        "source_metric_audit": audit_metrics(arrays["source"]["scores"], arrays["source"]["metrics"]),
        "native_metric_audit": audit_metrics(arrays["native"]["scores"], arrays["native"]["metrics"]),
    }
    return {"schema": "hy-nomas-query-correspondence-qualification-v1",
            "passed": all(item["passed"] for item in comparisons.values()),
            "claim_scope": "authenticated_model_capture_affinity" if lineage is not None else "submitted_array_affinity",
            "capture_lineage": lineage,
            "input": {"path": admitted["path"], "sha256": trusted_input,
                      "provenance": admitted["document"].get("provenance", {})},
            "reports": reports, "comparisons": comparisons,
            "summary": summarize(arrays["native"]["scores"], arrays["native"]["metrics"]),
            "scope": "F32 source-frame affinity diagnostic; no STCD, warp, causal rollout or acting claim"}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    controls = commands.add_parser("controls")
    controls.add_argument("--directory", type=Path, required=True)
    source = commands.add_parser("source")
    source.add_argument("--manifest", type=Path, required=True)
    source.add_argument("--manifest-sha256", required=True)
    source.add_argument("--report", type=Path, required=True)
    check = commands.add_parser("qualify")
    check.add_argument("--manifest", type=Path, required=True)
    check.add_argument("--manifest-sha256", required=True)
    check.add_argument("--source", type=Path, required=True)
    check.add_argument("--native", type=Path, required=True)
    check.add_argument("--report", type=Path, required=True)
    check.add_argument("--verify-model-capture", action="store_true",
                       help="Require authenticated model Q/K lineage; provenance labels alone do not qualify it")
    args = parser.parse_args(argv)
    if args.command == "controls":
        args.directory.mkdir(parents=False, exist_ok=False)
        for name, (query, key) in zip(("positive", "shuffled", "ties"), control_arrays()):
            path, digest = publish_input(args.directory / f"{name}.json", query, key,
                                         {"kind": f"synthetic_{name}", "seed": 20261106})
            print(json.dumps({"case": name, "path": str(path), "sha256": digest}))
    elif args.command == "source":
        admitted = load_input(args.manifest, args.manifest_sha256)
        print(json.dumps({"report": str(run_source(admitted, args.report))}))
    else:
        if args.report.exists() or args.report.is_symlink():
            raise ValueError("qualification output must not exist")
        result = qualify(args.manifest, args.manifest_sha256, args.source, args.native,
                         verify_model_capture=args.verify_model_capture)
        with args.report.open("x") as stream:
            stream.write(json.dumps(result, indent=2) + "\n")
        print(json.dumps(result))
        return 0 if result["passed"] else 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
