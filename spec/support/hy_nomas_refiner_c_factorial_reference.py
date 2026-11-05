#!/usr/bin/env python3
"""Source refiner-only c/text factorial oracle for the accepted V224 pair.

CPU admission authenticates the parent transfer reference and both of its
inputs before torch is imported. Source execution replays the same pinned
refiner, with c-bank interventions only at the input to the individual-token
refiner, then exports five checkpoints for each of five cases.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import tempfile
from typing import Any, Mapping

ROOT = Path(__file__).resolve().parents[2]
SUPPORT = ROOT / "spec" / "support"
if str(SUPPORT) not in sys.path:
    sys.path.insert(0, str(SUPPORT))

import hy_nomas_refiner_transfer_reference as transfer  # noqa: E402

SCHEMA = "hy-nomas-refiner-c-factorial-reference-v1"
SCHEMA_VERSION = 1
CASE_ORDER = ("A0", "A1", "B", "X10", "X01")
PARENT_CASES = ("A0", "A1", "B")
CHECKPOINTS = transfer.CHECKPOINTS
ATOL = transfer.ATOL
RTOL = transfer.RTOL
FACTORS = {
    "A0": {"text_case": "A0", "c_case": "A0", "override_c": False},
    "A1": {"text_case": "A0", "c_case": "A0", "override_c": True},
    "B": {"text_case": "B", "c_case": "B", "override_c": False},
    "X10": {"text_case": "B", "c_case": "A0", "override_c": True},
    "X01": {"text_case": "A0", "c_case": "B", "override_c": True},
}
EXECUTION_KEYS = {
    "case_order", "scope", "source_precision", "source_attention_query_chunks",
    "parent_replay", "source_aa",
}
DOCUMENT_KEYS = {
    "schema", "schema_version", "upstream_revision", "parent_reference", "source",
    "geometry", "timestep", "mask", "action_spans", "normalization", "precision",
    "execution", "case_factors", "payload", "tensors",
}
STAT_KEYS = {
    "checkpoint", "elements", "max_abs", "mean_abs", "rms_abs",
    "outside_atol_rtol", "all_finite", "passed", "atol", "rtol",
}


class ReferenceError(transfer.ReferenceError):
    """Fail-closed factorial schema, provenance, source, or payload error."""


def _same(a: Any, b: Any, label: str) -> None:
    if a != b:
        raise ReferenceError(f"{label} does not match authenticated parent")


def _record_shape(checkpoint: str) -> list[int]:
    return transfer._output_shape("A0", checkpoint)


def _elements(checkpoint: str) -> int:
    return math.prod(_record_shape(checkpoint))


def _validate_stats(record: Any, label: str, expected_checkpoint: str,
                    expected_elements: int) -> None:
    if type(record) is not dict or set(record) != STAT_KEYS:
        raise ReferenceError(f"{label} comparison record inventory mismatch")
    for key in ("max_abs", "mean_abs", "rms_abs"):
        value = record.get(key)
        if type(value) not in (int, float) or not math.isfinite(float(value)) or value < 0:
            raise ReferenceError(f"{label} comparison statistic malformed: {key}")
    if (record.get("checkpoint") != expected_checkpoint
            or type(record.get("elements")) is not int
            or record["elements"] != expected_elements
            or type(record.get("outside_atol_rtol")) is not int
            or record["outside_atol_rtol"] != 0
            or record.get("all_finite") is not True
            or record.get("passed") is not True
            or record.get("atol") != ATOL
            or record.get("rtol") != RTOL):
        raise ReferenceError(f"{label} comparison failed tolerance or shape")


def _validate_factor_map(value: Any) -> None:
    if type(value) is not dict or set(value) != set(CASE_ORDER):
        raise ReferenceError("factorial case inventory mismatch")
    for case in CASE_ORDER:
        record = value[case]
        if type(record) is not dict or set(record) != {"text_case", "c_case", "override_c"}:
            raise ReferenceError(f"factor map malformed: {case}")
        if type(record.get("override_c")) is not bool or record != FACTORS[case]:
            raise ReferenceError(f"factor map mismatch: {case}")


def validate_reference_document(document: Mapping[str, Any],
                                parent: Mapping[str, Any],
                                parent_path: str | Path,
                                parent_sha256: str) -> None:
    """Validate exact child schema and inheritance without importing torch."""
    if type(document) is not dict or set(document) != DOCUMENT_KEYS:
        raise ReferenceError("factorial reference document keys mismatch")
    if (document.get("schema") != SCHEMA
            or type(document.get("schema_version")) is not int
            or document["schema_version"] != SCHEMA_VERSION):
        raise ReferenceError("factorial reference schema/version mismatch")
    transfer.validate_reference_document(parent)
    parent_path = Path(parent_path)
    if not parent_path.is_absolute():
        raise ReferenceError("parent reference path must be absolute")
    transfer._digest(parent_sha256, "parent reference SHA-256")
    parent_ref = document.get("parent_reference")
    if (type(parent_ref) is not dict or set(parent_ref) != {"path", "sha256"}
            or parent_ref.get("path") != str(parent_path)
            or parent_ref.get("sha256") != parent_sha256):
        raise ReferenceError("parent reference binding mismatch")
    if document.get("upstream_revision") != parent["upstream_revision"]:
        raise ReferenceError("factorial upstream revision mismatch")
    for key in ("source", "geometry", "timestep", "mask", "action_spans",
                "normalization", "precision"):
        _same(document.get(key), parent.get(key), f"factorial {key}")
    _validate_factor_map(document.get("case_factors"))

    execution = document.get("execution")
    if type(execution) is not dict or set(execution) != EXECUTION_KEYS:
        raise ReferenceError("factorial execution contract keys mismatch")
    if (execution.get("case_order") != list(CASE_ORDER)
            or execution.get("scope") != "refiner only; no full DiT"
            or execution.get("source_precision") != parent["execution"]["source_precision"]
            or execution.get("source_attention_query_chunks") != 2):
        raise ReferenceError("factorial execution scope or source precision mismatch")
    replay = execution.get("parent_replay")
    if type(replay) is not dict or set(replay) != set(PARENT_CASES):
        raise ReferenceError("parent replay case inventory mismatch")
    for case in PARENT_CASES:
        records = replay[case]
        if type(records) is not dict or set(records) != set(CHECKPOINTS):
            raise ReferenceError(f"parent replay checkpoint inventory mismatch: {case}")
        for checkpoint in CHECKPOINTS:
            _validate_stats(records[checkpoint], f"parent replay {case}.{checkpoint}",
                            f"parent.{case}.{checkpoint}", _elements(checkpoint))
    aa = execution.get("source_aa")
    if type(aa) is not dict or set(aa) != set(CHECKPOINTS):
        raise ReferenceError("source A/A checkpoint inventory mismatch")
    for checkpoint in CHECKPOINTS:
        _validate_stats(aa[checkpoint], f"source A/A {checkpoint}",
                        f"A0/A1.{checkpoint}", _elements(checkpoint))

    payload = document.get("payload")
    if (type(payload) is not dict
            or set(payload) != {"basename", "bytes", "sha256", "encoding"}
            or type(payload.get("basename")) is not str
            or Path(payload["basename"]).name != payload["basename"]
            or payload.get("encoding") != "f32le_bf16_expansion"
            or type(payload.get("bytes")) is not int or payload["bytes"] <= 0):
        raise ReferenceError("factorial output payload metadata mismatch")
    transfer._digest(payload.get("sha256"), "factorial payload SHA-256")
    expected_names = {f"{case}.{checkpoint}" for case in CASE_ORDER for checkpoint in CHECKPOINTS}
    tensors = document.get("tensors")
    if type(tensors) is not dict or set(tensors) != expected_names:
        raise ReferenceError("factorial output tensor inventory mismatch")
    cursor = 0
    for case in CASE_ORDER:
        for checkpoint in CHECKPOINTS:
            name = f"{case}.{checkpoint}"
            record = tensors[name]
            shape = _record_shape(checkpoint)
            nbytes = math.prod(shape) * 4
            if (type(record) is not dict
                    or set(record) != {"dtype", "source_dtype", "shape", "offset_bytes",
                                       "nbytes", "sha256", "checkpoint_boundary"}
                    or record.get("dtype") != "F32"
                    or record.get("source_dtype") != "BF16"
                    or record.get("shape") != shape
                    or type(record.get("offset_bytes")) is not int
                    or record["offset_bytes"] != cursor
                    or type(record.get("nbytes")) is not int or record["nbytes"] != nbytes
                    or record.get("checkpoint_boundary") != checkpoint):
                raise ReferenceError(f"factorial tensor layout mismatch: {name}")
            transfer._digest(record.get("sha256"), f"{name} SHA-256")
            cursor += nbytes
    if payload["bytes"] != cursor:
        raise ReferenceError("factorial payload extent mismatch")


def _load_tensor(stream: Any, record: Mapping[str, Any], label: str, np: Any) -> Any:
    stream.seek(record["offset_bytes"])
    raw = stream.read(record["nbytes"])
    if len(raw) != record["nbytes"] or hashlib.sha256(raw).hexdigest() != record["sha256"]:
        raise ReferenceError(f"{label} range digest mismatch")
    values = np.frombuffer(raw, dtype="<f4")
    bits = values.view("<u4")
    if (not bool(np.isfinite(values).all())
            or bool(np.any(bits & np.uint32(0xFFFF)))):
        raise ReferenceError(f"{label} is not finite BF16-expanded F32")
    return values.reshape(record["shape"])


def _expected_comparisons(document: Mapping[str, Any], parent: Mapping[str, Any],
                          child_path: Path, parent_payload_path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    try:
        import numpy as np
    except ImportError as error:
        raise ReferenceError(f"NumPy is required for factorial checks: {error}") from error
    replay = {case: {} for case in PARENT_CASES}
    aa = {}
    with child_path.open("rb") as child_stream, parent_payload_path.open("rb") as parent_stream:
        for case in PARENT_CASES:
            for checkpoint in CHECKPOINTS:
                name = f"{case}.{checkpoint}"
                child_values = _load_tensor(child_stream, document["tensors"][name], name, np)
                parent_values = _load_tensor(parent_stream, parent["tensors"][name],
                                             f"parent {name}", np)
                replay[case][checkpoint] = transfer.compare_checkpoint(
                    f"parent.{case}.{checkpoint}", parent_values, child_values,
                )
        for checkpoint in CHECKPOINTS:
            lhs = _load_tensor(child_stream, document["tensors"][f"A0.{checkpoint}"],
                               f"A0.{checkpoint}", np)
            rhs = _load_tensor(child_stream, document["tensors"][f"A1.{checkpoint}"],
                               f"A1.{checkpoint}", np)
            aa[checkpoint] = transfer.compare_checkpoint(
                f"A0/A1.{checkpoint}", lhs, rhs,
            )
    return replay, aa


def validate_reference_payload(document: Mapping[str, Any], payload_path: str | Path,
                               parent: Mapping[str, Any],
                               parent_payload_path: str | Path,
                               parent_path: str | Path,
                               parent_sha256: str) -> None:
    """Authenticate output bytes and recompute source A/A and parent replay."""
    validate_reference_document(document, parent, parent_path, parent_sha256)
    child_path = transfer._regular_file(payload_path, "factorial output payload")
    parent_path = transfer._regular_file(parent_payload_path, "parent output payload")
    record = document["payload"]
    if (child_path.name != record["basename"]
            or child_path.stat().st_size != record["bytes"]
            or transfer.sha256_file(child_path) != record["sha256"]):
        raise ReferenceError("factorial output payload digest/extent mismatch")
    transfer.validate_reference_payload(parent, parent_path)
    # Validate every child range before comparing any of them.
    try:
        import numpy as np
    except ImportError as error:
        raise ReferenceError(f"NumPy is required for factorial checks: {error}") from error
    with child_path.open("rb") as stream:
        for name, tensor in document["tensors"].items():
            _load_tensor(stream, tensor, name, np)
    replay, aa = _expected_comparisons(document, parent, child_path, parent_path)
    if document["execution"]["source_aa"] != aa:
        raise ReferenceError("source A/A statistics do not match checkpoint bytes")
    if document["execution"]["parent_replay"] != replay:
        raise ReferenceError("parent replay statistics do not match checkpoint bytes")
    for case in PARENT_CASES:
        if any(not result["passed"] for result in replay[case].values()):
            raise ReferenceError(f"source parent replay is outside tolerance: {case}")
    if any(not result["passed"] for result in aa.values()):
        raise ReferenceError("source A/A no-op intervention is outside tolerance")


def _input_from_parent(parent_path: Path, parent: Mapping[str, Any], label: str) -> dict[str, Any]:
    case_label = "A" if label in ("A0", "A1") else "B"
    record = parent["inputs"]["cases"][label]
    case_manifest_path = parent_path.parent / record["manifest_path"]
    case_manifest_path = transfer._regular_file(case_manifest_path, f"{label} case manifest")
    raw_manifest = case_manifest_path.read_bytes()
    if hashlib.sha256(raw_manifest).hexdigest() != record["manifest_sha256"]:
        raise ReferenceError(f"{label} case manifest digest mismatch")
    case_manifest = transfer._json_no_duplicates(raw_manifest, f"{label} case manifest")
    transfer._validate_case_manifest(case_manifest)
    input_path = transfer._regular_file(
        case_manifest_path.parent / case_manifest["payload"]["basename"], f"{label} Qwen input",
    )
    origin_record = case_manifest["origin_manifest"]
    origin_path = transfer._regular_file(origin_record["path"], f"{label} input origin")
    if transfer.sha256_file(origin_path) != origin_record["sha256"]:
        raise ReferenceError(f"{label} origin manifest digest mismatch")
    authenticated = transfer.authenticate_case_input(input_path, origin_path, case_label)
    expected_case_record = transfer._case_reference_record(
        authenticated, case_manifest_path, input_path, case_manifest,
    )
    if expected_case_record != record:
        raise ReferenceError(f"{label} parent input record differs from authenticated source")
    return authenticated


def admit_parent(parent_arg: str | Path, expected_sha256: str) -> dict[str, Any]:
    """Authenticate parent reference, input provenance, source checkout and weights."""
    parent_path = transfer._regular_file(parent_arg, "parent reference")
    if transfer.sha256_file(parent_path) != expected_sha256:
        raise ReferenceError("parent reference digest mismatch")
    parent = transfer._read_json(parent_path, "parent reference")
    transfer.validate_reference_document(parent)
    parent_payload = transfer._regular_file(
        parent_path.parent / parent["payload"]["basename"], "parent output payload",
    )
    transfer.validate_reference_payload(parent, parent_payload)
    source = parent["source"]
    upstream_path = Path(source["upstream_path"]).expanduser()
    if upstream_path.is_symlink():
        raise ReferenceError("pinned upstream source path must not be a symlink")
    try:
        upstream_path = upstream_path.resolve(strict=True)
    except OSError:
        raise ReferenceError("pinned upstream source path is unavailable") from None
    if not upstream_path.is_dir():
        raise ReferenceError("pinned upstream source path is not a directory")
    source_hashes = transfer._verify_pinned_source(upstream_path)
    if source_hashes != source["source_files"]:
        raise ReferenceError("parent pinned source hashes changed")
    weights = source["text_weights"]
    weights_path = transfer._regular_file(weights["path"], "source text weights")
    weights_manifest_path = transfer._regular_file(weights["manifest_path"], "source text weight manifest")
    verified_weights = transfer._validate_weights(weights_path, weights_manifest_path)
    for key in ("path", "sha256", "bytes", "manifest_path", "manifest_sha256", "tensor_count",
                "stored_dtype", "loaded_dtype", "conversion"):
        if verified_weights[key] != weights[key]:
            raise ReferenceError(f"parent text-weight binding changed: {key}")
    input_a = _input_from_parent(parent_path, parent, "A0")
    input_a1 = _input_from_parent(parent_path, parent, "A1")
    input_b = _input_from_parent(parent_path, parent, "B")
    if input_a["normalized_bytes"] != input_a1["normalized_bytes"]:
        raise ReferenceError("parent A0/A1 source text differs")
    prepared = transfer.prepare_pair(input_a["normalized_bytes"], input_b["normalized_bytes"])
    if prepared["A0"] != input_a["normalized_bytes"] or prepared["B"] != input_b["normalized_bytes"]:
        raise ReferenceError("parent source inputs differ from normalized V224 pair")
    return {
        "path": parent_path,
        "sha256": expected_sha256,
        "document": parent,
        "payload_path": parent_payload,
        "inputs": {"A0": prepared["A0"], "A1": prepared["A1"], "B": prepared["B"]},
        "upstream_path": upstream_path,
        "weights_path": weights_path,
        "weights_manifest_path": weights_manifest_path,
        "weights_manifest": verified_weights,
    }


def _capture_factorial(torch: Any, modules: Mapping[str, Any], state: dict[str, Any],
                       source_inputs: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    captures: dict[str, dict[str, Any]] = {case: {} for case in CASE_ORDER}
    active = {"case": ""}
    c_banks: dict[str, Any] = {}
    hooks = []

    def save_output(name: str):
        def hook(_module: Any, _inputs: tuple[Any, ...], output: Any) -> None:
            case = active["case"]
            if not isinstance(output, torch.Tensor) or output.dtype != torch.bfloat16:
                raise ReferenceError(f"source checkpoint is not BF16: {case}.{name}")
            value = output[:, :transfer.VALID_ROWS] if name.startswith("block") else output
            captures[case][name] = value.detach().contiguous().clone()
        return hook

    def mean_pre_hook(_module: Any, inputs: tuple[Any, ...]) -> None:
        value = inputs[0]
        if not isinstance(value, torch.Tensor) or value.dtype != torch.float32:
            raise ReferenceError("source masked-mean input must be observed as F32 before BF16 cast")
        captures[active["case"]]["masked_mean"] = value.to(torch.bfloat16).detach().contiguous().clone()

    def c_pre_hook(_module: Any, inputs: tuple[Any, ...]) -> tuple[Any, ...] | None:
        case = active["case"]
        if len(inputs) < 2 or not isinstance(inputs[1], torch.Tensor):
            raise ReferenceError(f"source individual-token refiner c input missing: {case}")
        if FACTORS[case]["override_c"]:
            c_case = FACTORS[case]["c_case"]
            if c_case not in c_banks:
                raise ReferenceError(f"source c bank was not captured before intervention: {case}")
            value = c_banks[c_case]
        else:
            value = inputs[1]
        if value.dtype != torch.bfloat16 or tuple(value.shape) != (1, transfer.REFINER_DIM):
            raise ReferenceError(f"source c intervention shape/dtype mismatch: {case}")
        captures[case]["c"] = value.detach().contiguous().clone()
        if FACTORS[case]["override_c"]:
            return (inputs[0], value, *inputs[2:])
        return None

    refiner = modules["txt_in"]
    hooks.append(refiner.c_embedder.register_forward_pre_hook(mean_pre_hook))
    hooks.append(refiner.individual_token_refiner.register_forward_pre_hook(c_pre_hook))
    hooks.append(modules["time_in"].register_forward_hook(save_output("vec")))
    for index, block in enumerate(refiner.individual_token_refiner.blocks):
        hooks.append(block.register_forward_hook(save_output(f"block{index}.output")))
    try:
        with torch.inference_mode():
            for case in CASE_ORDER:
                active["case"] = case
                state["attention_block"] = 0
                state["attention_query_chunks"] = 0
                source_inputs["slot"].copy_(source_inputs["source"][FACTORS[case]["text_case"]])
                with torch.autocast("mps", dtype=torch.bfloat16):
                    vector = modules["time_in"](source_inputs["timestep"])
                    if vector.dtype != torch.bfloat16:
                        raise ReferenceError(f"source timestep vector is not BF16: {case}")
                    result = refiner(source_inputs["slot"], source_inputs["timestep"], source_inputs["mask"])
                if not isinstance(result, torch.Tensor) or result.dtype != torch.bfloat16:
                    raise ReferenceError(f"source refiner output is not BF16: {case}")
                if (state.get("attention_block") != transfer.BLOCKS
                        or state.get("attention_query_chunks") != transfer.BLOCKS):
                    raise ReferenceError(f"source attention adapter coverage mismatch: {case}")
                if set(captures[case]) != set(CHECKPOINTS):
                    raise ReferenceError(f"source checkpoint inventory mismatch: {case}")
                if case in ("A0", "B"):
                    c_banks[case] = captures[case]["c"].detach().clone()
    finally:
        active["case"] = ""
        for hook in hooks:
            hook.remove()
    return captures


def _export_factorial(torch: Any, np: Any,
                      captures: Mapping[str, Mapping[str, Any]]) -> dict[str, dict[str, Any]]:
    keys = [(case, cp) for case in CASE_ORDER for cp in CHECKPOINTS]
    expanded = []
    ranges = {}
    shapes = {}
    cursor = 0
    for case, checkpoint in keys:
        tensor = captures[case][checkpoint]
        shape = _record_shape(checkpoint)
        if tensor.dtype != torch.bfloat16 or list(tensor.shape) != shape:
            raise ReferenceError(f"source checkpoint shape/dtype changed: {case}.{checkpoint}")
        value = tensor.to(dtype=torch.float32).reshape(-1)
        expanded.append(value)
        ranges[(case, checkpoint)] = (cursor, cursor + value.numel())
        shapes[(case, checkpoint)] = shape
        cursor += value.numel()
    packed_device = torch.cat(expanded).contiguous()
    torch.mps.synchronize()
    packed = packed_device.to(device="cpu").numpy()
    if packed.dtype != np.float32 or packed.size != cursor or not bool(np.isfinite(packed).all()):
        raise ReferenceError("packed factorial source export is malformed or nonfinite")
    result = {case: {} for case in CASE_ORDER}
    for case, checkpoint in keys:
        start, stop = ranges[(case, checkpoint)]
        result[case][checkpoint] = np.array(
            packed[start:stop].reshape(shapes[(case, checkpoint)]), dtype="<f4", copy=True,
        )
    return result


def _build_payload(cases: Mapping[str, Mapping[str, Any]]) -> tuple[dict[str, Any], bytes]:
    chunks = []
    tensors = {}
    cursor = 0
    for case in CASE_ORDER:
        for checkpoint in CHECKPOINTS:
            name = f"{case}.{checkpoint}"
            values = cases[case][checkpoint]
            shape = list(values.shape)
            if shape != _record_shape(checkpoint):
                raise ReferenceError(f"source checkpoint shape mismatch: {name}")
            raw = values.astype("<f4", copy=False).tobytes(order="C")
            tensors[name] = {
                "dtype": "F32", "source_dtype": "BF16", "shape": shape,
                "offset_bytes": cursor, "nbytes": len(raw),
                "sha256": hashlib.sha256(raw).hexdigest(),
                "checkpoint_boundary": checkpoint,
            }
            chunks.append(raw)
            cursor += len(raw)
    return tensors, b"".join(chunks)


def _stats(cases: Mapping[str, Mapping[str, Any]], parent: Mapping[str, Any],
           parent_payload_path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    try:
        import numpy as np
    except ImportError as error:
        raise ReferenceError(f"NumPy is required for checkpoint comparison: {error}") from error
    replay = {case: {} for case in PARENT_CASES}
    for case in PARENT_CASES:
        for checkpoint in CHECKPOINTS:
            name = f"{case}.{checkpoint}"
            record = parent["tensors"][name]
            with parent_payload_path.open("rb") as stream:
                expected = _load_tensor(stream, record, f"parent {name}", np)
            replay[case][checkpoint] = transfer.compare_checkpoint(
                f"parent.{name}", expected, cases[case][checkpoint],
            )
    aa = {
        checkpoint: transfer.compare_checkpoint(
            f"A0/A1.{checkpoint}", cases["A0"][checkpoint], cases["A1"][checkpoint],
        ) for checkpoint in CHECKPOINTS
    }
    for case in PARENT_CASES:
        if any(not record["passed"] for record in replay[case].values()):
            raise ReferenceError(f"source parent replay is outside tolerance: {case}")
    if any(not record["passed"] for record in aa.values()):
        raise ReferenceError("source A/A no-op intervention is outside tolerance")
    return replay, aa


def run_reference(*, parent_reference_arg: str | Path,
                  parent_reference_sha256: str,
                  output_manifest_arg: str | Path) -> Path:
    """Authenticate the parent, run the five source cases on MPS, publish outputs."""
    admitted = admit_parent(parent_reference_arg, parent_reference_sha256)
    parent_path = admitted["path"]
    parent = admitted["document"]
    output_manifest = Path(output_manifest_arg).expanduser()
    if not output_manifest.is_absolute():
        output_manifest = Path.cwd() / output_manifest
    output_manifest = output_manifest.parent.resolve(strict=True) / output_manifest.name
    output_payload = output_manifest.with_name(output_manifest.stem + ".factorial.f32le")
    if output_manifest.exists() or output_payload.exists():
        raise ReferenceError("refusing to overwrite factorial reference output")
    try:
        import numpy as np
        import torch
    except ImportError as error:
        raise ReferenceError(f"source MPS runtime dependency missing: {error}") from error
    if not torch.backends.mps.is_built() or not torch.backends.mps.is_available():
        raise ReferenceError("source factorial reference is MPS-only; CPU fallback is forbidden")
    modules, state, _adapter = transfer._module_from_pinned_source(
        torch, admitted["upstream_path"], admitted["weights_path"],
        admitted["weights_manifest"], torch.device("mps"),
    )
    source_inputs = transfer._prepare_source_inputs(torch, np, admitted["inputs"])
    captures = _capture_factorial(torch, modules, state, source_inputs)
    source_cases = _export_factorial(torch, np, captures)
    parent_replay, source_aa = _stats(source_cases, parent, admitted["payload_path"])
    tensors, payload_bytes = _build_payload(source_cases)
    document = {
        "schema": SCHEMA,
        "schema_version": SCHEMA_VERSION,
        "upstream_revision": parent["upstream_revision"],
        "parent_reference": {"path": str(parent_path), "sha256": parent_reference_sha256},
        "source": parent["source"],
        "geometry": parent["geometry"],
        "timestep": parent["timestep"],
        "mask": parent["mask"],
        "action_spans": parent["action_spans"],
        "normalization": parent["normalization"],
        "precision": parent["precision"],
        "execution": {
            "case_order": list(CASE_ORDER),
            "scope": "refiner only; no full DiT",
            "source_precision": parent["execution"]["source_precision"],
            "source_attention_query_chunks": transfer.BLOCKS,
            "parent_replay": parent_replay,
            "source_aa": source_aa,
        },
        "case_factors": FACTORS,
        "payload": {
            "basename": output_payload.name,
            "bytes": len(payload_bytes),
            "sha256": hashlib.sha256(payload_bytes).hexdigest(),
            "encoding": "f32le_bf16_expansion",
        },
        "tensors": tensors,
    }
    validate_reference_document(document, parent, parent_path, parent_reference_sha256)
    manifest_raw = json.dumps(document, sort_keys=True, indent=2).encode("utf-8") + b"\n"
    output_manifest.parent.mkdir(parents=True, exist_ok=True)
    created: list[Path] = []
    with tempfile.TemporaryDirectory(prefix=".hy-nomas-c-factorial-", dir=output_manifest.parent) as temp:
        temp_path = Path(temp)
        staged_payload = temp_path / output_payload.name
        staged_manifest = temp_path / output_manifest.name
        for path, raw in ((staged_payload, payload_bytes), (staged_manifest, manifest_raw)):
            with path.open("xb") as stream:
                stream.write(raw)
                stream.flush()
                os.fsync(stream.fileno())
        validate_reference_payload(document, staged_payload, parent, admitted["payload_path"],
                                   parent_path, parent_reference_sha256)
        try:
            os.replace(staged_payload, output_payload)
            created.append(output_payload)
            os.replace(staged_manifest, output_manifest)
            created.append(output_manifest)
        except Exception:
            for path in created:
                path.unlink(missing_ok=True)
            raise
    return output_manifest


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent-reference", required=True, type=Path)
    parser.add_argument("--parent-reference-sha256", required=True)
    parser.add_argument("--output-manifest", required=True, type=Path)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    try:
        output = run_reference(
            parent_reference_arg=args.parent_reference,
            parent_reference_sha256=args.parent_reference_sha256,
            output_manifest_arg=args.output_manifest,
        )
    except (ReferenceError, OSError) as error:
        print(f"refiner c-factorial reference rejected: {error}", file=sys.stderr)
        return 2
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
