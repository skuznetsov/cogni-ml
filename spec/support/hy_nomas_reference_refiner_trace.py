#!/usr/bin/env python3
"""Capture the actual T0 text-refiner prelude before DiT block 0.

This opt-in trace reuses the authenticated structured continuation admission
and the existing source preblock sentinel. It adds observational hooks to the
pinned upstream refiner and exports the existing nine preblock tensors plus
native-comparable refiner checkpoints in one terminal device-to-host copy.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
import sys
from typing import Any, Iterator, Mapping
from contextlib import contextmanager

ROOT = Path(__file__).resolve().parents[2]
SUPPORT = ROOT / "spec" / "support"
EXAMPLE_SUPPORT = ROOT / "examples" / "support"
for _path in (SUPPORT, EXAMPLE_SUPPORT):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

import hy_nomas_structured_reference_control as structured  # noqa: E402
import hy_nomas_reference_preblock_control as preblock  # noqa: E402

base = structured.base

SCHEMA = "cogni_ml.hy_nomas_reference_refiner_source_trace/v1"
SCHEMA_VERSION = 1
COMMON_TENSOR_ORDER = (
    "input_compact_valid",
    "masked_mean",
    "timestep",
    "refiner_timestep_sinusoidal",
    "refiner_timestep_mlp1",
    "refiner_timestep_silu",
    "refiner_timestep",
    "context_linear1",
    "context_silu",
    "context_embed",
    "c",
    "input_embed",
    "time_in.sinusoidal",
    "time_in.mlp1",
    "time_in.silu",
    "vec",
)
SOURCE_ONLY_TENSOR_ORDER = (
    "refiner_timestep_sinusoidal_raw",
    "time_in.sinusoidal_raw",
    "refiner_mask",
)
TENSOR_ORDER = preblock.TENSOR_ORDER + COMMON_TENSOR_ORDER + SOURCE_ONLY_TENSOR_ORDER
TRACE_EVENT_ORDER = (
    "timestep",
    "time_in.sinusoidal_raw",
    "time_in.sinusoidal",
    "time_in.mlp1",
    "time_in.silu",
    "vec",
    "input_compact_valid",
    "refiner_mask",
    "refiner_timestep_sinusoidal_raw",
    "refiner_timestep_sinusoidal",
    "refiner_timestep_mlp1",
    "refiner_timestep_silu",
    "refiner_timestep",
    "masked_mean",
    "context_linear1",
    "context_silu",
    "context_embed",
    "input_embed",
    "c",
)
_TIME_EMBED_LAYERS = "hymm.models.dit.modules.embed_layers"
_TIME_EMBED_RELATIVE = Path("hymm/models/dit/modules/embed_layers.py")


class RefinerTraceError(RuntimeError):
    """Fail-closed source refiner trace error."""


def _expected_checkpoint_shapes(valid_text_tokens: int) -> dict[str, tuple[int, ...]]:
    if type(valid_text_tokens) is not int or not 0 < valid_text_tokens <= base.QWEN_ROWS:
        raise RefinerTraceError("valid_text_tokens is outside the authenticated Qwen extent")
    return {
        "input_compact_valid": (1, valid_text_tokens, 3584),
        "masked_mean": (1, 3584),
        "timestep": (1,),
        "refiner_timestep_sinusoidal": (1, 256),
        "refiner_timestep_mlp1": (1, 2048),
        "refiner_timestep_silu": (1, 2048),
        "refiner_timestep": (1, 2048),
        "context_linear1": (1, 2048),
        "context_silu": (1, 2048),
        "context_embed": (1, 2048),
        "c": (1, 2048),
        "input_embed": (1, valid_text_tokens, 2048),
        "time_in.sinusoidal": (1, 256),
        "time_in.mlp1": (1, 2048),
        "time_in.silu": (1, 2048),
        "vec": (1, 2048),
        "refiner_timestep_sinusoidal_raw": (1, 256),
        "time_in.sinusoidal_raw": (1, 256),
        "refiner_mask": (1, base.QWEN_ROWS),
    }


def _checkpoint_source_types(precision_control_f32: bool = False) -> dict[str, str]:
    if type(precision_control_f32) is not bool:
        raise RefinerTraceError("precision-control mode must be a boolean")
    module_type = "F32" if precision_control_f32 else "BF16"
    values = {name: module_type for name in COMMON_TENSOR_ORDER}
    values["timestep"] = "F32"
    values.update({name: "F32" for name in SOURCE_ONLY_TENSOR_ORDER})
    values["refiner_mask"] = "BOOL"
    return values


def _all_source_types(precision_control_f32: bool = False) -> dict[str, str]:
    return {
        **preblock._source_types(precision_control_f32),
        **_checkpoint_source_types(precision_control_f32),
    }


def _shape(tensor: Any) -> tuple[int, ...]:
    try:
        return tuple(int(value) for value in tensor.shape)
    except (AttributeError, TypeError, ValueError):
        raise RefinerTraceError("captured tensor has no valid shape") from None


def _dtype_name(tensor: Any) -> str:
    dtype = str(getattr(tensor, "dtype", "")).split(".")[-1]
    if dtype == "bfloat16":
        return "BF16"
    if dtype == "float32":
        return "F32"
    if dtype == "bool":
        return "BOOL"
    raise RefinerTraceError(f"captured tensor has unsupported dtype: {dtype or 'missing'}")


def _device_type(tensor: Any) -> str | None:
    return getattr(getattr(tensor, "device", None), "type", None)


def validate_snapshot_layout(
    snapshot: Mapping[str, Any], valid_text_tokens: int, *,
    event_order: tuple[str, ...] | list[str] = TRACE_EVENT_ORDER,
    require_mps: bool = False,
    precision_control_f32: bool = False,
) -> dict[str, tuple[int, ...]]:
    """Validate exact source inventory, chronology, shapes, and source dtypes."""
    if tuple(snapshot) != TENSOR_ORDER:
        raise RefinerTraceError("source refiner tensor inventory or order differs from its contract")
    if tuple(event_order) != TRACE_EVENT_ORDER:
        raise RefinerTraceError("source refiner checkpoint capture order differs from upstream forward order")
    shapes = {
        **preblock._expected_shapes(valid_text_tokens),
        **_expected_checkpoint_shapes(valid_text_tokens),
    }
    source_types = _all_source_types(precision_control_f32)
    for name in TENSOR_ORDER:
        tensor = snapshot[name]
        if _shape(tensor) != shapes[name]:
            raise RefinerTraceError(f"captured tensor geometry differs from the source/native contract: {name}")
        if _dtype_name(tensor) != source_types[name]:
            raise RefinerTraceError(f"captured source dtype differs from the selected precision contract: {name}")
        if require_mps and _device_type(tensor) != "mps":
            raise RefinerTraceError(f"captured tensor did not remain on MPS: {name}")
    return shapes


def _record_checkpoint(trace: dict[str, Any], name: str, tensor: Any) -> None:
    if not callable(getattr(tensor, "detach", None)) or not hasattr(tensor, "shape"):
        raise RefinerTraceError(f"expected actual source tensor at checkpoint {name}")
    values = trace.setdefault("checkpoints", {})
    if name not in TENSOR_ORDER or name in preblock.TENSOR_ORDER:
        raise RefinerTraceError(f"unknown refiner checkpoint: {name}")
    if name in values:
        raise RefinerTraceError(f"source refiner checkpoint executed more than once: {name}")
    values[name] = tensor
    trace.setdefault("events", []).append(name)


def _register_input_hook(module: Any, trace: dict[str, Any], index: int, name: str,
                         transform: Any = None) -> Any:
    if not callable(getattr(module, "register_forward_pre_hook", None)):
        raise RefinerTraceError(f"pinned source module cannot capture input: {name}")

    def hook(_module: Any, args: tuple[Any, ...]) -> None:
        if len(args) <= index:
            raise RefinerTraceError(f"source module input is missing at checkpoint {name}")
        tensor = args[index]
        _record_checkpoint(trace, name, transform(tensor) if transform else tensor)

    return module.register_forward_pre_hook(hook)


def _register_output_hook(module: Any, trace: dict[str, Any], name: str,
                          transform: Any = None) -> Any:
    if not callable(getattr(module, "register_forward_hook", None)):
        raise RefinerTraceError(f"pinned source module cannot capture output: {name}")

    def hook(_module: Any, _args: tuple[Any, ...], output: Any) -> None:
        _record_checkpoint(trace, name, transform(output) if transform else output)

    return module.register_forward_hook(hook)


def _install_refiner_hooks(
    modules: Mapping[str, Any], trace: dict[str, Any], valid_text_tokens: int,
    embed_layers: Any,
) -> None:
    """Attach observations to the loaded source modules; no operation is replaced."""
    if trace.get("hook_handles"):
        raise RefinerTraceError("refiner hooks were installed more than once")
    if set(modules) != {"time_in", "txt_in", "img_in", "final_layer"}:
        raise RefinerTraceError("loaded pinned prelude module inventory changed")
    time_in = modules["time_in"]
    refiner = modules["txt_in"]
    for attr in ("t_embedder", "c_embedder", "input_embedder", "individual_token_refiner"):
        if not hasattr(refiner, attr):
            raise RefinerTraceError(f"pinned text refiner is missing module: {attr}")
    for parent in (time_in, refiner.t_embedder):
        mlp = getattr(parent, "mlp", None)
        if mlp is None or len(mlp) != 3:
            raise RefinerTraceError("pinned source timestep MLP inventory changed")
    if not callable(getattr(embed_layers, "timestep_embedding", None)):
        raise RefinerTraceError("pinned source timestep embedding function is unavailable")
    trace["hook_handles"] = handles = []
    trace["events"] = []
    trace["checkpoints"] = {}
    trace["active_sinusoid"] = None
    trace["sinusoid_calls"] = {"time_in": 0, "refiner": 0}

    def begin(module_name: str):
        def hook(_module: Any, args: tuple[Any, ...]) -> None:
            if not args:
                raise RefinerTraceError(f"source timestep input is missing at {module_name}")
            if trace["active_sinusoid"] is not None:
                raise RefinerTraceError("source timestep embedding stages unexpectedly nested")
            trace["active_sinusoid"] = module_name
            if module_name == "time_in":
                _record_checkpoint(trace, "timestep", args[0])
            else:
                trace["refiner_timestep_input"] = args[0]
        return hook

    def end(module_name: str, result_name: str):
        def hook(_module: Any, _args: tuple[Any, ...], output: Any) -> None:
            if trace.get("active_sinusoid") != module_name:
                raise RefinerTraceError(f"source timestep embedding stage did not close: {module_name}")
            if result_name is not None:
                _record_checkpoint(trace, result_name, output)
            trace["active_sinusoid"] = None
        return hook

    original_timestep_embedding = embed_layers.timestep_embedding

    def timestep_embedding(*args: Any, **kwargs: Any) -> Any:
        result = original_timestep_embedding(*args, **kwargs)
        active = trace.get("active_sinusoid")
        if active == "time_in":
            trace["sinusoid_calls"]["time_in"] += 1
            _record_checkpoint(trace, "time_in.sinusoidal_raw", result)
        elif active == "refiner":
            trace["sinusoid_calls"]["refiner"] += 1
            _record_checkpoint(trace, "refiner_timestep_sinusoidal_raw", result)
        return result

    def capture_valid_qwen(_module: Any, args: tuple[Any, ...]) -> None:
        if len(args) <= 2:
            _raise_missing("T0 Qwen input or refiner mask")
        mask = args[2]
        _validate_refiner_mask_tensor(mask)
        trace["input_compact_full"] = args[0]
        trace["refiner_mask_full"] = mask
        _record_checkpoint(trace, "input_compact_valid", args[0][:, :valid_text_tokens, :])
        _record_checkpoint(trace, "refiner_mask", mask)

    embed_layers.timestep_embedding = timestep_embedding
    trace["timestep_embedding_patch"] = (embed_layers, original_timestep_embedding)
    try:
        handles.append(time_in.register_forward_pre_hook(begin("time_in")))
        handles.append(time_in.register_forward_hook(end("time_in", None)))
        handles.append(refiner.register_forward_pre_hook(capture_valid_qwen))
        handles.append(refiner.t_embedder.register_forward_pre_hook(
            begin("refiner")
        ))
        handles.append(refiner.t_embedder.register_forward_hook(end("refiner", "refiner_timestep")))
        handles.append(_register_input_hook(time_in.mlp[0], trace, 0, "time_in.sinusoidal"))
        handles.append(_register_output_hook(time_in.mlp[0], trace, "time_in.mlp1"))
        handles.append(_register_output_hook(time_in.mlp[1], trace, "time_in.silu"))
        handles.append(_register_output_hook(time_in.mlp[2], trace, "vec"))
        handles.append(_register_input_hook(
            refiner.t_embedder.mlp[0], trace, 0, "refiner_timestep_sinusoidal"
        ))
        handles.append(_register_output_hook(
            refiner.t_embedder.mlp[0], trace, "refiner_timestep_mlp1"
        ))
        handles.append(_register_output_hook(
            refiner.t_embedder.mlp[1], trace, "refiner_timestep_silu"
        ))
        handles.append(_register_input_hook(refiner.c_embedder, trace, 0, "masked_mean"))
        handles.append(_register_output_hook(refiner.c_embedder.linear_1, trace, "context_linear1"))
        handles.append(_register_output_hook(refiner.c_embedder.act_1, trace, "context_silu"))
        handles.append(_register_output_hook(refiner.c_embedder.linear_2, trace, "context_embed"))
        handles.append(_register_output_hook(
            refiner.input_embedder, trace, "input_embed",
            lambda output: output[:, :valid_text_tokens, :],
        ))
        handles.append(_register_input_hook(refiner.individual_token_refiner, trace, 1, "c"))
    except Exception:
        try:
            for handle in reversed(handles):
                handle.remove()
        finally:
            handles.clear()
            embed_layers.timestep_embedding = original_timestep_embedding
            trace.pop("timestep_embedding_patch", None)
        raise


def _raise_missing(label: str) -> None:
    raise RefinerTraceError(f"source module input is missing at checkpoint: {label}")


def _validate_refiner_mask_tensor(mask: Any) -> None:
    if _shape(mask) != (1, base.QWEN_ROWS):
        raise RefinerTraceError("actual source refiner mask has unexpected shape")
    if _dtype_name(mask) != "BOOL":
        raise RefinerTraceError("actual source refiner mask is not boolean")
    if _device_type(mask) != "mps":
        raise RefinerTraceError("actual source refiner mask did not remain on MPS")
    is_contiguous = getattr(mask, "is_contiguous", None)
    if not callable(is_contiguous) or not is_contiguous():
        raise RefinerTraceError("actual source refiner mask is not contiguous")


@contextmanager
def _install_refiner_trace_runtime(
    runtime: Any, trace: dict[str, Any], valid_text_tokens: int, upstream: Path,
) -> Iterator[None]:
    """Wrap preblock's loaded-module path and restore all observational hooks."""
    from unittest.mock import patch

    original_loader = runtime.attention_reference._load_prelude_weights
    trace["loader_calls"] = 0

    def load_prelude(*args: Any, **kwargs: Any) -> dict[str, Any]:
        trace["loader_calls"] += 1
        if trace["loader_calls"] != 1:
            raise RefinerTraceError("source prelude loader executed more than once")
        modules = original_loader(*args, **kwargs)
        import sys as _sys

        embed_layers = _sys.modules.get(_TIME_EMBED_LAYERS)
        expected_path = (Path(upstream) / _TIME_EMBED_RELATIVE).resolve(strict=True)
        actual_path = Path(getattr(embed_layers, "__file__", "")).resolve(strict=True) if embed_layers else None
        if embed_layers is None or actual_path != expected_path:
            raise RefinerTraceError("loaded timestep function is not from the authenticated pinned upstream source")
        _install_refiner_hooks(modules, trace, valid_text_tokens, embed_layers)
        return modules

    with patch.object(runtime.attention_reference, "_load_prelude_weights", load_prelude):
        try:
            yield
        finally:
            handles = trace.get("hook_handles", ())
            try:
                for handle in reversed(handles):
                    handle.remove()
            finally:
                if hasattr(handles, "clear"):
                    handles.clear()
                patch_record = trace.get("timestep_embedding_patch")
                if patch_record is not None:
                    embed_layers, original = patch_record
                    embed_layers.timestep_embedding = original
                    trace.pop("timestep_embedding_patch", None)


def _snapshot_from_capture(
    capture: Mapping[str, Any], trace: Mapping[str, Any], valid_text_tokens: int,
) -> dict[str, Any]:
    text_stage = capture.get("txt_in")
    consumed_args = text_stage.get("consumed_args", ()) if isinstance(text_stage, Mapping) else ()
    if trace.get("input_compact_full") is not (consumed_args[0] if consumed_args else None):
        raise RefinerTraceError("captured compact Qwen rows are not the exact consumed T0 refiner input")
    if (len(consumed_args) <= 2
            or trace.get("refiner_mask_full") is not consumed_args[2]):
        raise RefinerTraceError("captured source refiner mask is not the exact consumed txt_in argument 2")
    if trace.get("loader_calls") != 1 or trace.get("active_sinusoid") is not None:
        raise RefinerTraceError("source refiner trace did not complete one clean loaded-module pass")
    if trace.get("sinusoid_calls") != {"time_in": 1, "refiner": 1}:
        raise RefinerTraceError("pinned timestep embedding was not observed exactly once per source stage")
    if tuple(trace.get("events", ())) != TRACE_EVENT_ORDER:
        raise RefinerTraceError("source refiner checkpoints are missing, duplicated, or out of order")
    checkpoints = trace.get("checkpoints")
    if type(checkpoints) is not dict or tuple(checkpoints) != TRACE_EVENT_ORDER:
        raise RefinerTraceError("source refiner checkpoint inventory differs from the exact event sequence")
    base_snapshot = preblock._snapshot_from_capture(capture, valid_text_tokens)
    global_stage = capture["time_in"]
    if trace.get("refiner_timestep_input") is not global_stage["args"][0]:
        raise RefinerTraceError("text refiner did not consume the source writer timestep")
    if checkpoints["timestep"] is not global_stage["args"][0]:
        raise RefinerTraceError("captured source timestep is detached from the writer prelude")
    if checkpoints["vec"] is not global_stage["output"]:
        raise RefinerTraceError("captured source vec is detached from the actual time_in output")
    if checkpoints["input_compact_valid"].shape[1] != valid_text_tokens:
        raise RefinerTraceError("captured source Qwen prefix differs from the authenticated valid row count")
    snapshot = {**base_snapshot, **{name: checkpoints[name] for name in COMMON_TENSOR_ORDER}}
    snapshot.update({name: checkpoints[name] for name in SOURCE_ONLY_TENSOR_ORDER})
    return {name: snapshot[name] for name in TENSOR_ORDER}


def _validate_trace_binding(
    capture: Mapping[str, Any], snapshot: Mapping[str, Any],
    admitted: Mapping[str, Any], *, precision_control_f32: bool,
) -> None:
    preblock._validate_writer_call(
        capture, {name: snapshot[name] for name in preblock.TENSOR_ORDER}, admitted,
        precision_control_f32=precision_control_f32,
    )
    if capture.get("block_loader_calls") != 1 or capture.get("sentinel_calls") != 1:
        raise RefinerTraceError("source trace did not stop at exactly the first block boundary")
    if _device_type(snapshot["input_compact_valid"]) != "mps":
        raise RefinerTraceError("captured valid Qwen prefix left the MPS device")
    if _dtype_name(snapshot["input_compact_valid"]) != ("F32" if precision_control_f32 else "BF16"):
        raise RefinerTraceError("captured valid Qwen prefix differs from consumed refiner precision")
    if _dtype_name(snapshot["timestep"]) != "F32":
        raise RefinerTraceError("captured source writer timestep is not F32")
    if snapshot["input_compact_valid"].shape != (1, admitted["t0_valid_tokens"], 3584):
        raise RefinerTraceError("captured compact valid Qwen prefix has unexpected geometry")
    _validate_refiner_mask_tensor(snapshot["refiner_mask"])


def _pack_payload(
    torch: Any, snapshot: Mapping[str, Any], valid_text_tokens: int, *,
    precision_control_f32: bool = False,
) -> tuple[bytes, dict[str, dict[str, Any]], dict[str, str], dict[str, str]]:
    shapes = validate_snapshot_layout(
        snapshot, valid_text_tokens, require_mps=True,
        precision_control_f32=precision_control_f32,
    )
    source_types = _all_source_types(precision_control_f32)
    export_types = {
        **preblock._EXPORT_TYPES,
        **{name: "F32" for name in COMMON_TENSOR_ORDER + SOURCE_ONLY_TENSOR_ORDER},
    }
    byte_parts = []
    records: dict[str, dict[str, Any]] = {}
    offset = 0
    for name in TENSOR_ORDER:
        tensor = snapshot[name].detach().contiguous()
        if export_types[name] == "F32":
            tensor = tensor.to(dtype=torch.float32)
        if _dtype_name(tensor) != export_types[name]:
            raise RefinerTraceError(f"source export conversion failed: {name}")
        byte_parts.append(tensor.view(torch.uint8).reshape(-1))
        nbytes = math.prod(shapes[name]) * (2 if export_types[name] == "BF16" else 4)
        records[name] = {
            "dtype": export_types[name],
            "shape": list(shapes[name]),
            "offset": offset,
            "nbytes": nbytes,
        }
        offset += nbytes
    packed = torch.cat(byte_parts, dim=0).cpu()
    try:
        payload = packed.numpy().tobytes(order="C")
    except (AttributeError, RuntimeError, TypeError) as error:
        raise RefinerTraceError(f"terminal tensor export failed: {error}") from None
    if len(payload) != offset:
        raise RefinerTraceError("terminal source tensor export byte extent differs from its inventory")
    for record in records.values():
        start = record["offset"]
        record["sha256"] = hashlib.sha256(payload[start:start + record["nbytes"]]).hexdigest()
    _validate_payload_finite(payload, records)
    _validate_refiner_mask_payload(payload, records, valid_text_tokens)
    return payload, records, source_types, export_types


def _validate_refiner_mask_payload(
    payload: bytes, records: Mapping[str, Mapping[str, Any]], valid_text_tokens: int,
) -> None:
    record = records.get("refiner_mask")
    expected_shape = [1, base.QWEN_ROWS]
    if (type(record) is not dict or record.get("dtype") != "F32"
            or record.get("shape") != expected_shape
            or record.get("nbytes") != 4 * base.QWEN_ROWS):
        raise RefinerTraceError("exported source refiner mask differs from its F32 tensor contract")
    if type(valid_text_tokens) is not int or not 0 < valid_text_tokens <= base.QWEN_ROWS:
        raise RefinerTraceError("valid text count is outside the exported source mask extent")
    start = int(record["offset"])
    actual = payload[start:start + int(record["nbytes"])]
    expected = b"\x00\x00\x80\x3f" * valid_text_tokens + b"\x00" * (
        4 * (base.QWEN_ROWS - valid_text_tokens)
    )
    if actual != expected:
        raise RefinerTraceError("exported source refiner mask does not equal the authenticated valid prefix")


def _validate_payload_finite(payload: bytes, records: Mapping[str, Mapping[str, Any]]) -> None:
    try:
        import numpy as np
    except ImportError as error:
        raise RefinerTraceError("NumPy is required for terminal source trace validation") from error
    for name, record in records.items():
        start = int(record["offset"])
        raw = payload[start:start + int(record["nbytes"])]
        if record["dtype"] == "F32":
            if not np.isfinite(np.frombuffer(raw, dtype="<f4")).all():
                raise RefinerTraceError(f"terminal source tensor contains non-finite values: {name}")
        else:
            words = np.frombuffer(raw, dtype="<u2")
            if np.any((words & 0x7F80) == 0x7F80):
                raise RefinerTraceError(f"terminal source tensor contains non-finite values: {name}")


def _manifest(
    args: Any, admitted: Mapping[str, Any], payload_name: str, payload: bytes,
    records: Mapping[str, Mapping[str, Any]], source_types: Mapping[str, str],
    export_types: Mapping[str, str], *, precision_control_f32: bool,
) -> dict[str, Any]:
    base_records = {name: records[name] for name in preblock.TENSOR_ORDER}
    document = preblock._manifest(
        args, admitted, payload_name, payload, base_records,
        precision_control_f32=precision_control_f32,
    )
    document["schema"] = SCHEMA
    document["schema_version"] = SCHEMA_VERSION
    document["scope"] = (
        "actual source-F32 precision-control T0 prelude and refiner checkpoints captured immediately before DiT block 0; "
        "no block forward or K/V write"
        if precision_control_f32 else
        "actual source-BF16 T0 prelude and refiner checkpoints captured immediately before DiT block 0; "
        "no block forward or K/V write"
    )
    document["tensor_order"] = list(TENSOR_ORDER)
    document["tensors"] = dict(records)
    execution = document["execution"]
    execution["source_dtype_by_tensor"] = dict(source_types)
    execution["export_dtype_by_tensor"] = dict(export_types)
    execution["refiner_trace"] = {
        "common_tensor_order": list(COMMON_TENSOR_ORDER),
        "source_only_tensor_order": list(SOURCE_ONLY_TENSOR_ORDER),
        "event_order": list(TRACE_EVENT_ORDER),
        "full_text_refiner_encode_executed": True,
        "text_refiner_entry_calls": 1,
        "input_compact_valid": "exact valid prefix of the consumed txt_in input, after the selected source precision prehook",
        "refiner_mask": "actual bool txt_in argument 2, after source prehooks; exact consumed object, shape/device/contiguity validated and F32 export checked against the authenticated valid prefix",
        "masked_mean": "exact input to the pinned source c_embedder, after its mask multiply, sum, and division",
        "c": "exact argument 1 consumed by the pinned source individual_token_refiner",
        "input_embed": "exact valid prefix of the pinned source input_embedder output",
        "timestep_sinusoidal": "source-only raw F32 function output plus native-comparable actual input to mlp[0] after linear_dtype cast",
        "capture_method": "read-only source module hooks and scoped observation of the pinned timestep_embedding function result",
        "valid_prefix_tokens": admitted["t0_valid_tokens"],
        "upstream_embed_layers_sha256": admitted["source_hashes"].get(_TIME_EMBED_RELATIVE.as_posix()),
        "source_forward_computations_replayed": False,
    }
    execution["semantic_notes"].extend([
        "all common refiner checkpoints are the actual source tensors observed at their producing module or consuming argument; none are recomputed",
        "source-only raw sinusoid tensors precede the upstream linear_dtype cast; native-comparable sinusoid tensors are the exact mlp[0] inputs",
        "source-only refiner_mask is exported as F32 in the same terminal transfer and checked against the valid-prefix mask bytes after that transfer",
        "the source masked mean is captured at c_embedder input; this trace does not substitute the native mean implementation",
    ])
    return document


def _publish(output: Path, payload_path: Path, document: dict[str, Any], payload: bytes) -> str:
    document["payload"] = payload_path.name
    encoded = (json.dumps(document, separators=(",", ":"), allow_nan=False) + "\n").encode("utf-8")
    payload_identity = None
    try:
        if os.path.lexists(output) or os.path.lexists(payload_path):
            raise FileExistsError("trace manifest or sibling payload already exists")
        base.continuation_inputs._publish_noreplace(payload_path, payload)
        stat = payload_path.stat()
        payload_identity = (stat.st_dev, stat.st_ino)
        base.continuation_inputs._publish_noreplace(output, encoded)
    except Exception as error:
        if payload_identity is not None and os.path.lexists(payload_path):
            try:
                stat = payload_path.stat()
                if (stat.st_dev, stat.st_ino) == payload_identity:
                    payload_path.unlink(missing_ok=True)
            except OSError:
                pass
        raise RefinerTraceError(f"could not publish fresh source refiner trace: {error}") from None
    return hashlib.sha256(encoded).hexdigest()


def _cpu_plan(admitted: Mapping[str, Any], args: Any) -> dict[str, Any]:
    plan = preblock._cpu_plan(admitted, args)
    plan.update({
        "schema": SCHEMA,
        "tensor_names": list(TENSOR_ORDER),
        "common_tensor_names": list(COMMON_TENSOR_ORDER),
        "source_only_tensor_names": list(SOURCE_ONLY_TENSOR_ORDER),
        "capture_boundary": "actual text refiner and source prelude before DiT block 0",
        "hooked_forward_computations_replayed": False,
        "terminal_cpu_exports": 1,
    })
    return plan


def _execute(args: Any, admitted: dict[str, Any], base_args: Any,
             action_valid_tokens: int) -> dict[str, Any]:
    precision_control_f32 = getattr(args, "precision_control_f32", False)
    output, payload_path = preblock.output_paths(args.output)
    base.single._configure_mps_runtime()
    torch, scheduler, diffusers_version = base.single._prepare_cpu_scheduler(
        Path(admitted["upstream_path"]), "source-bf16"
    )
    original_factory = base._cache_attention_dispatch_factory
    trace: dict[str, Any] = {}
    capture: dict[str, Any] = {"events": []}

    def make_dispatch(runtime_torch: Any, state: dict[str, Any]) -> Any:
        return structured._action_cache_attention_dispatch_factory(
            runtime_torch, state, admitted["action_profile"],
            Path(admitted["upstream_path"]), legacy_factory=original_factory,
        )

    try:
        base._cache_attention_dispatch_factory = make_dispatch
        with structured._bind_dynamic_base_contract(action_valid_tokens, admitted["dit_text_rows"]):
            with preblock._install_runtime_patches(
                base, capture, precision_control_f32=precision_control_f32,
                expected_t0_qwen=admitted["t0_qwen"], expected_admitted=admitted,
            ):
                with _install_refiner_trace_runtime(
                    base, trace, admitted["t0_valid_tokens"], Path(admitted["upstream_path"]),
                ):
                    try:
                        with torch.inference_mode():
                            base._run_control(admitted, base_args, torch, scheduler, diffusers_version)
                    except preblock._StopBeforeBlock0:
                        pass
                    else:
                        raise RefinerTraceError("source control returned without reaching the block-0 sentinel")
        base_snapshot = preblock._snapshot_from_capture(capture, admitted["t0_valid_tokens"])
        preblock.validate_snapshot_layout(
            base_snapshot, admitted["t0_valid_tokens"], require_mps=True,
            precision_control_f32=precision_control_f32,
        )
        preblock._validate_writer_call(
            capture, base_snapshot, admitted, precision_control_f32=precision_control_f32,
        )
        snapshot = _snapshot_from_capture(capture, trace, admitted["t0_valid_tokens"])
        validate_snapshot_layout(
            snapshot, admitted["t0_valid_tokens"], require_mps=True,
            precision_control_f32=precision_control_f32,
        )
        _validate_trace_binding(
            capture, snapshot, admitted, precision_control_f32=precision_control_f32,
        )
        payload, records, source_types, export_types = _pack_payload(
            torch, snapshot, admitted["t0_valid_tokens"],
            precision_control_f32=precision_control_f32,
        )
    finally:
        base._cache_attention_dispatch_factory = original_factory
    document = _manifest(
        args, admitted, payload_path.name, payload, records, source_types, export_types,
        precision_control_f32=precision_control_f32,
    )
    manifest_sha = _publish(output, payload_path, document, payload)
    return {
        "schema": SCHEMA,
        "status": "captured_source_refiner_preblock",
        "precision_mode": "source-f32-precision-control" if precision_control_f32 else "source-bf16",
        "manifest": str(output),
        "manifest_sha256": manifest_sha,
        "payload": str(payload_path),
        "payload_sha256": document["payload_sha256"],
        "payload_bytes": document["payload_bytes"],
        "loaded_dit_blocks": 0,
        "block_forward_calls": 0,
        "cache_kv_written": False,
        "terminal_cpu_exports": 1,
        "tensor_names": list(TENSOR_ORDER),
    }


def _parse_args(argv: list[str] | None = None):
    return preblock._parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    try:
        args = _parse_args(argv)
        admitted, base_args, action_valid_tokens = structured._authenticate(args)
        if args.plan_only:
            print(json.dumps(_cpu_plan(admitted, args), sort_keys=True))
            return 0
        preblock.output_paths(args.output)
        result = _execute(args, admitted, base_args, action_valid_tokens)
    except (RefinerTraceError, preblock.PreblockTraceError,
            structured.StructuredControlError, base.ControlError,
            base.single.ControlError, OSError, ValueError, RuntimeError) as error:
        print(json.dumps({"schema": SCHEMA, "status": "rejected", "reason": str(error)}, sort_keys=True), file=sys.stderr)
        return 2
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
