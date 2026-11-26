#!/usr/bin/env python3
"""Capture the admitted T0 tensors immediately before DiT block 0.

This diagnostic reuses structured-control admission and the continuation's real
T0 preprocessing. The default preserves source BF16; an opt-in F32 precision
control widens the actual BF16 writer latent and Qwen input before their global
prelude modules. Both modes stop at the first block call and do not load any
DiT block weights, execute block 0, or write cache K/V.
"""

from __future__ import annotations

import argparse
from contextlib import ExitStack, contextmanager
import hashlib
import json
import math
import os
from pathlib import Path
import sys
from typing import Any, Iterator, Mapping

ROOT = Path(__file__).resolve().parents[2]
SUPPORT = ROOT / "spec" / "support"
EXAMPLE_SUPPORT = ROOT / "examples" / "support"
for _path in (SUPPORT, EXAMPLE_SUPPORT):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

import hy_nomas_structured_reference_control as structured  # noqa: E402

base = structured.base

SCHEMA = "cogni_ml.hy_nomas_reference_preblock_source_trace/v1"
SCHEMA_VERSION = 1
TENSOR_ORDER = (
    "reference.latent_bf16",
    "cache.noisy_latent_f32",
    "block.img_f32",
    "block.txt_valid_f32",
    "block.vec_f32",
    "block.vec_txt_f32",
    "block.rope_cos_f32",
    "block.rope_sin_f32",
    "cache.timestep_f32",
)
STAGE_ORDER = ("time_in", "txt_in", "img_in", "block0")
F32_EXPORT_NAMES = TENSOR_ORDER[1:]
_EXPORT_TYPES = {name: ("BF16" if name == "reference.latent_bf16" else "F32")
                 for name in TENSOR_ORDER}
_SOURCE_TYPES = {
    "reference.latent_bf16": "BF16",
    "cache.noisy_latent_f32": "BF16",
    "block.img_f32": "BF16",
    "block.txt_valid_f32": "BF16",
    "block.vec_f32": "BF16",
    "block.vec_txt_f32": "BF16",
    "block.rope_cos_f32": "F32",
    "block.rope_sin_f32": "F32",
    "cache.timestep_f32": "F32",
}
_F32_PRELUDE_NAMES = (
    "block.img_f32",
    "block.txt_valid_f32",
    "block.vec_f32",
    "block.vec_txt_f32",
)


class PreblockTraceError(RuntimeError):
    """Fail-closed source preblock trace error."""


class _StopBeforeBlock0(BaseException):
    """Private control flow raised by the sentinel in place of block 0."""


def _expected_shapes(valid_text_tokens: int) -> dict[str, tuple[int, ...]]:
    if type(valid_text_tokens) is not int or not 0 < valid_text_tokens <= base.QWEN_ROWS:
        raise PreblockTraceError("valid_text_tokens is outside the authenticated Qwen extent")
    return {
        "reference.latent_bf16": tuple(base.LATENT_SEGMENT_SHAPE),
        "cache.noisy_latent_f32": tuple(base.LATENT_SEGMENT_SHAPE),
        "block.img_f32": (1, base.VIDEO_TOKENS, base.DIM),
        "block.txt_valid_f32": (1, valid_text_tokens, base.DIM),
        "block.vec_f32": (1, base.DIM),
        "block.vec_txt_f32": (1, base.DIM),
        "block.rope_cos_f32": (base.VIDEO_TOKENS, base.HEAD_DIM),
        "block.rope_sin_f32": (base.VIDEO_TOKENS, base.HEAD_DIM),
        "cache.timestep_f32": (1,),
    }


def _shape(tensor: Any) -> tuple[int, ...]:
    try:
        return tuple(int(value) for value in tensor.shape)
    except (AttributeError, TypeError, ValueError):
        raise PreblockTraceError("captured tensor has no valid shape") from None


def _dtype_name(tensor: Any) -> str:
    dtype = str(getattr(tensor, "dtype", "" )).split(".")[-1]
    if dtype == "bfloat16":
        return "BF16"
    if dtype == "float32":
        return "F32"
    raise PreblockTraceError(f"captured tensor has unsupported dtype: {dtype or 'missing'}")


def _source_types(precision_control_f32: bool = False) -> dict[str, str]:
    if type(precision_control_f32) is not bool:
        raise PreblockTraceError("precision-control mode must be a boolean")
    types = dict(_SOURCE_TYPES)
    if precision_control_f32:
        for name in _F32_PRELUDE_NAMES:
            types[name] = "F32"
    return types


def _device_type(tensor: Any) -> str | None:
    device = getattr(tensor, "device", None)
    return getattr(device, "type", None)


def validate_snapshot_layout(
    snapshot: Mapping[str, Any], valid_text_tokens: int, *,
    stage_order: tuple[str, ...] | list[str] = STAGE_ORDER,
    require_mps: bool = False,
    precision_control_f32: bool = False,
) -> dict[str, tuple[int, ...]]:
    """Check names, boundary order, geometry, source dtype, and optional device."""
    if tuple(snapshot) != TENSOR_ORDER:
        raise PreblockTraceError(
            "source preblock tensor inventory or order differs from the nine-name contract"
        )
    if tuple(stage_order) != STAGE_ORDER:
        raise PreblockTraceError("source preblock stages are missing or out of order")
    source_types = _source_types(precision_control_f32)
    shapes = _expected_shapes(valid_text_tokens)
    for name in TENSOR_ORDER:
        tensor = snapshot[name]
        if _shape(tensor) != shapes[name]:
            raise PreblockTraceError(
                f"captured tensor geometry differs from the source/native contract: {name}"
            )
        if _dtype_name(tensor) != source_types[name]:
            raise PreblockTraceError(
                f"captured source dtype differs from the BF16 boundary contract: {name}"
            )
        if require_mps and _device_type(tensor) != "mps":
            raise PreblockTraceError(f"captured tensor did not remain on MPS: {name}")
    return shapes


def output_paths(output: str | Path) -> tuple[Path, Path]:
    """Validate a fresh manifest/payload pair without creating either file."""
    manifest = Path(output).expanduser()
    if (not manifest.is_absolute() or manifest.suffix.lower() != ".json"
            or not manifest.parent.is_dir()):
        raise PreblockTraceError("--output must be an absolute .json path in an existing directory")
    payload = manifest.with_name(f"{manifest.stem}.tensors.bin")
    if (manifest.is_symlink() or payload.is_symlink()
            or os.path.lexists(manifest) or os.path.lexists(payload)):
        raise PreblockTraceError("trace manifest and .tensors.bin payload must both be fresh files")
    return manifest, payload


def _record_module_hooks(
    modules: Mapping[str, Any], capture: dict[str, Any], *,
    precision_control_f32: bool = False, float32_dtype: Any = None,
) -> list[Any]:
    handles = []
    capture.setdefault("events", [])
    capture["hook_handles"] = handles
    if precision_control_f32 and float32_dtype is None:
        raise PreblockTraceError("F32 prelude cast has no explicit float32 dtype")
    for name in ("time_in", "txt_in", "img_in"):
        module = modules.get(name)
        if module is None or not callable(getattr(module, "register_forward_pre_hook", None)):
            raise PreblockTraceError(f"pinned global module cannot be traced: {name}")

        def pre_hook(_module: Any, args: tuple[Any, ...], *, stage: str = name) -> Any:
            if stage in capture:
                raise PreblockTraceError(f"pinned global stage executed more than once: {stage}")
            capture["events"].append(stage)
            capture[stage] = {"args": args}
            if precision_control_f32 and stage in ("txt_in", "img_in"):
                if not args or _dtype_name(args[0]) != "BF16":
                    raise PreblockTraceError(
                        f"F32 precision control expected the exact BF16 source input at {stage}"
                    )
                consumed = list(args)
                consumed[0] = args[0].to(dtype=float32_dtype)
                if _dtype_name(consumed[0]) != "F32":
                    raise PreblockTraceError(f"F32 precision cast failed at {stage}")
                capture[stage]["cast_input"] = consumed[0]
                return tuple(consumed)
            return None

        def post_hook(
            _module: Any, args: tuple[Any, ...], output: Any, *, stage: str = name,
        ) -> None:
            record = capture.get(stage)
            if type(record) is not dict:
                raise PreblockTraceError(f"pinned global stage output has no matching input: {stage}")
            record["consumed_args"] = args
            record["output"] = output

        handles.append(module.register_forward_pre_hook(pre_hook))
        handles.append(module.register_forward_hook(post_hook))
    return handles


class _FirstBlockSentinel:
    def __init__(self, capture: dict[str, Any]):
        self.capture = capture

    def __call__(self, **kwargs: Any) -> Any:
        capture = self.capture
        if capture.get("sentinel_calls", 0) != 0:
            raise PreblockTraceError("source continuation reached more than one block boundary")
        capture["sentinel_calls"] = 1
        capture["events"].append("block0")
        capture["block_kwargs"] = kwargs
        raise _StopBeforeBlock0()


@contextmanager
def _install_runtime_patches(
    runtime: Any, capture: dict[str, Any], *, precision_control_f32: bool = False,
    expected_t0_qwen: Any = None, expected_admitted: Any = None,
) -> Iterator[None]:
    """Replace only block loading/cache setup; restore every hook on all exits."""
    sentinel = _FirstBlockSentinel(capture)
    capture["sentinel"] = sentinel
    original_load_prelude = runtime.attention_reference._load_prelude_weights
    original_install_refiner_attention = runtime.attention_reference._install_refiner_attention

    def load_prelude(*args: Any, **kwargs: Any) -> dict[str, Any]:
        prelude_args = list(args)
        prelude_kwargs = dict(kwargs)
        float32_dtype = None
        if precision_control_f32:
            if not prelude_args:
                raise PreblockTraceError("F32 prelude loader did not receive its runtime torch module")
            float32_dtype = prelude_args[0].float32
            if len(prelude_args) >= 6:
                prelude_args[5] = float32_dtype
            elif "dtype" in prelude_kwargs:
                prelude_kwargs["dtype"] = float32_dtype
            else:
                raise PreblockTraceError("F32 prelude loader has no dtype argument to override")
        modules = original_load_prelude(*prelude_args, **prelude_kwargs)
        _record_module_hooks(
            modules, capture, precision_control_f32=precision_control_f32,
            float32_dtype=float32_dtype,
        )
        return modules

    def install_refiner_attention(torch: Any, state: dict[str, Any], *args: Any,
                                  **kwargs: Any) -> Any:
        if precision_control_f32:
            state["mode"] = "f32"
        return original_install_refiner_attention(torch, state, *args, **kwargs)

    original_load_clean = runtime._load_t0_clean
    original_load_qwen = runtime._load_qwen
    original_load_epsilon = runtime._load_cache_epsilon

    def load_clean(*args: Any, **kwargs: Any) -> Any:
        value = original_load_clean(*args, **kwargs)
        capture["clean_t0"] = value
        return value

    def load_qwen(*args: Any, **kwargs: Any) -> Any:
        source = args[2] if len(args) > 2 else kwargs.get("source")
        if expected_t0_qwen is not None and source is not expected_t0_qwen:
            raise PreblockTraceError("source preblock trace attempted to load a non-T0 Qwen tensor")
        value = original_load_qwen(*args, **kwargs)
        capture["t0_qwen_load_calls"] = capture.get("t0_qwen_load_calls", 0) + 1
        capture["t0_qwen"] = value
        return value

    def load_epsilon(*args: Any, **kwargs: Any) -> Any:
        source = args[2] if len(args) > 2 else kwargs.get("admitted")
        if expected_admitted is not None and source is not expected_admitted:
            raise PreblockTraceError("source preblock trace attempted to reload another cache input")
        value = original_load_epsilon(*args, **kwargs)
        capture["cache_epsilon_load_calls"] = capture.get("cache_epsilon_load_calls", 0) + 1
        capture["cache_epsilon"] = value
        return value

    def load_sentinel(*_args: Any, **_kwargs: Any) -> list[Any]:
        capture["block_loader_calls"] = capture.get("block_loader_calls", 0) + 1
        return [sentinel]

    def bind_nothing(*_args: Any, **_kwargs: Any) -> None:
        # The replacement is not a DiT block and is never asked to resolve source globals.
        return None

    def no_cache_allocation(*_args: Any, **_kwargs: Any) -> tuple[list[Any], list[Any]]:
        # The sentinel stops before any cache read/write; one unused slot is enough.
        return ([object()], [object()])

    with ExitStack() as stack:
        from unittest.mock import patch

        stack.enter_context(patch.object(runtime.single, "_load_all_blocks", load_sentinel))
        stack.enter_context(patch.object(runtime.two_segment, "_bind_runtime_cache_functions", bind_nothing))
        stack.enter_context(patch.object(runtime, "_new_caches", no_cache_allocation))
        stack.enter_context(patch.object(runtime, "_load_t0_clean", load_clean))
        stack.enter_context(patch.object(runtime, "_load_qwen", load_qwen))
        stack.enter_context(patch.object(runtime, "_load_cache_epsilon", load_epsilon))
        stack.enter_context(patch.object(
            runtime.attention_reference, "_install_refiner_attention", install_refiner_attention,
        ))
        stack.enter_context(patch.object(runtime.attention_reference, "_load_prelude_weights", load_prelude))
        try:
            yield
        finally:
            for handle in capture.get("hook_handles", ()):
                handle.remove()


def _snapshot_from_capture(capture: Mapping[str, Any], valid_text_tokens: int) -> dict[str, Any]:
    if tuple(capture.get("events", ())) != STAGE_ORDER:
        raise PreblockTraceError("actual T0 preblock stage order differs from time/text/image/block0")
    if capture.get("block_loader_calls") != 1 or capture.get("sentinel_calls") != 1:
        raise PreblockTraceError("source execution did not stop at exactly the first block boundary")
    clean = capture.get("clean_t0")
    stages = {name: capture.get(name) for name in ("time_in", "txt_in", "img_in")}
    if any(type(stage) is not dict or "args" not in stage or "output" not in stage
           for stage in stages.values()):
        raise PreblockTraceError("source prelude is missing an observed input or output stage")
    kwargs = capture.get("block_kwargs")
    if type(kwargs) is not dict:
        raise PreblockTraceError("source first-block call was not captured")
    if len(stages["time_in"]["args"]) < 1 or len(stages["img_in"]["args"]) < 1:
        raise PreblockTraceError("source timestep or image embedding input is missing")
    freqs = kwargs.get("freqs_cis")
    if not isinstance(freqs, (tuple, list)) or len(freqs) != 2:
        raise PreblockTraceError("source first-block RoPE pair is missing")
    text = kwargs.get("txt")
    try:
        valid_text = text[:, :valid_text_tokens]
    except (IndexError, TypeError):
        raise PreblockTraceError("source first-block text rows cannot be cropped to the valid prefix") from None
    snapshot = {
        "reference.latent_bf16": clean,
        "cache.noisy_latent_f32": stages["img_in"]["args"][0],
        "block.img_f32": kwargs.get("img"),
        "block.txt_valid_f32": valid_text,
        "block.vec_f32": kwargs.get("vec"),
        "block.vec_txt_f32": kwargs.get("vec_txt"),
        "block.rope_cos_f32": freqs[0],
        "block.rope_sin_f32": freqs[1],
        "cache.timestep_f32": stages["time_in"]["args"][0],
    }
    return snapshot


def _validate_writer_call(
    capture: Mapping[str, Any], snapshot: Mapping[str, Any], admitted: Mapping[str, Any],
    *, precision_control_f32: bool = False,
) -> None:
    kwargs = capture["block_kwargs"]
    stages = {name: capture[name] for name in ("time_in", "txt_in", "img_in")}
    writer_call = admitted["forward_plan"][0]
    expected_scalar_args = {
        "cache_valid_len": writer_call["cache_valid_len"],
        "update_cache": writer_call["update_cache"],
        "cache_txt": writer_call["cache_txt"],
        "attn_param": None,
        "is_flash": False,
        "cache_attn_mode": "torch",
        "txt_tokens_per_block": admitted["dit_text_rows"],
    }
    if any(kwargs.get(name) != expected for name, expected in expected_scalar_args.items()):
        raise PreblockTraceError("captured first-block call differs from the admitted T0 writer contract")
    if kwargs.get("prompt_video_boundaries") != [0, base.VIDEO_TOKENS]:
        raise PreblockTraceError("captured first-block video boundary differs from the T0 segment")
    if kwargs.get("kv_cache") is None or kwargs.get("cache_mask") is None:
        raise PreblockTraceError("captured first-block cache arguments are incomplete")
    if capture.get("t0_qwen_load_calls") != 1 or capture.get("cache_epsilon_load_calls") != 1:
        raise PreblockTraceError("source T0 Qwen or cache epsilon was not loaded exactly once")
    epsilon = capture.get("cache_epsilon")
    if (_shape(epsilon) != tuple(base.LATENT_SEGMENT_SHAPE)
            or _dtype_name(epsilon) != "BF16" or _device_type(epsilon) != "mps"):
        raise PreblockTraceError("source cache epsilon is not the authenticated BF16 T0 segment on MPS")
    if (kwargs.get("vec") is not stages["time_in"]["output"]
            or kwargs.get("vec_txt") is not stages["time_in"]["output"]
            or kwargs.get("img") is not stages["img_in"]["output"]
            or snapshot["cache.noisy_latent_f32"] is not stages["img_in"]["args"][0]
            or snapshot["cache.timestep_f32"] is not stages["time_in"]["args"][0]):
        raise PreblockTraceError("captured source block inputs are detached from their observed prelude outputs")
    text_args = stages["txt_in"]["args"]
    if (len(text_args) < 3 or text_args[0] is not capture.get("t0_qwen")
            or text_args[1] is not stages["time_in"]["args"][0]
            or _dtype_name(text_args[0]) != "BF16"
            or _device_type(text_args[0]) != "mps"
            or _shape(stages["txt_in"]["output"]) != (1, base.QWEN_ROWS, base.DIM)):
        raise PreblockTraceError(
            "captured text refiner did not consume the loaded T0 Qwen, writer timestep, and full extent"
        )
    expected_prelude_dtype = "F32" if precision_control_f32 else "BF16"
    if any(_dtype_name(stages[name]["output"]) != expected_prelude_dtype
           for name in ("time_in", "txt_in", "img_in")):
        raise PreblockTraceError("captured global prelude output dtype differs from the selected precision mode")
    if precision_control_f32:
        for name in ("txt_in", "img_in"):
            consumed = stages[name].get("consumed_args")
            if (not isinstance(consumed, tuple) or not consumed
                    or _dtype_name(consumed[0]) != "F32"
                    or _device_type(consumed[0]) != "mps"
                    or consumed[0] is not stages[name].get("cast_input")):
                raise PreblockTraceError(
                    f"F32 prelude did not consume its exact widened BF16 input at {name}"
                )
    else:
        if stages["txt_in"].get("consumed_args", text_args)[0] is not text_args[0]:
            raise PreblockTraceError("source-BF16 text refiner input identity changed")
        image_args = stages["img_in"]["args"]
        if stages["img_in"].get("consumed_args", image_args)[0] is not image_args[0]:
            raise PreblockTraceError("source-BF16 patch embed input identity changed")
    text_mask = kwargs.get("text_mask")
    if (text_mask is None or _shape(text_mask) != (1, admitted["dit_text_rows"])
            or str(getattr(text_mask, "dtype", "")).split(".")[-1] != "bool"):
        raise PreblockTraceError("captured T0 text mask differs from the admitted DIT row geometry")
    if tuple(snapshot) != TENSOR_ORDER:
        raise PreblockTraceError("captured source tensor order changed before export")
    if precision_control_f32:
        if any(_dtype_name(snapshot[name]) != "F32" for name in _F32_PRELUDE_NAMES):
            raise PreblockTraceError("source F32 precision-control tensor inventory changed before export")
        if any(_dtype_name(snapshot[name]) != "BF16" for name in (
            "reference.latent_bf16", "cache.noisy_latent_f32",
        )):
            raise PreblockTraceError("source F32 precision control altered the clean or noisy BF16 writer input")


def _pack_payload(
    torch: Any, snapshot: Mapping[str, Any], valid_text_tokens: int, *,
    precision_control_f32: bool = False,
) -> tuple[bytes, dict[str, dict[str, Any]]]:
    shapes = validate_snapshot_layout(
        snapshot, valid_text_tokens, require_mps=True,
        precision_control_f32=precision_control_f32,
    )
    byte_parts = []
    records: dict[str, dict[str, Any]] = {}
    offset = 0
    for name in TENSOR_ORDER:
        tensor = snapshot[name].detach().contiguous()
        if name in F32_EXPORT_NAMES:
            tensor = tensor.to(dtype=torch.float32)
        expected_type = _EXPORT_TYPES[name]
        if _dtype_name(tensor) != expected_type:
            raise PreblockTraceError(f"source export conversion failed: {name}")
        byte_tensor = tensor.view(torch.uint8).reshape(-1)
        byte_parts.append(byte_tensor)
        nbytes = math.prod(shapes[name]) * (2 if expected_type == "BF16" else 4)
        records[name] = {
            "dtype": expected_type,
            "shape": list(shapes[name]),
            "offset": offset,
            "nbytes": nbytes,
        }
        offset += nbytes
    packed = torch.cat(byte_parts, dim=0).cpu()
    try:
        payload = packed.numpy().tobytes(order="C")
    except (AttributeError, RuntimeError, TypeError) as error:
        raise PreblockTraceError(f"terminal tensor export failed: {error}") from None
    if len(payload) != offset:
        raise PreblockTraceError("terminal tensor export byte extent differs from the admitted inventory")
    for name, record in records.items():
        start = record["offset"]
        record["sha256"] = hashlib.sha256(payload[start:start + record["nbytes"]]).hexdigest()
    _validate_payload_finite(payload, records)
    return payload, records


def _validate_payload_finite(payload: bytes, records: Mapping[str, Mapping[str, Any]]) -> None:
    """Check exported values on CPU, after the one terminal device-to-host copy."""
    try:
        import numpy as np
    except ImportError as error:
        raise PreblockTraceError("NumPy is required for terminal source trace validation") from error
    for name in TENSOR_ORDER:
        record = records[name]
        start = int(record["offset"])
        stop = start + int(record["nbytes"])
        raw = payload[start:stop]
        if record["dtype"] == "F32":
            values = np.frombuffer(raw, dtype="<f4")
            if not np.isfinite(values).all():
                raise PreblockTraceError(f"terminal source tensor contains non-finite values: {name}")
        else:
            words = np.frombuffer(raw, dtype="<u2")
            if np.any((words & 0x7F80) == 0x7F80):
                raise PreblockTraceError(f"terminal source tensor contains non-finite values: {name}")


def _manifest(
    args: argparse.Namespace, admitted: Mapping[str, Any], payload_name: str,
    payload: bytes, tensor_records: Mapping[str, Mapping[str, Any]], *,
    precision_control_f32: bool = False,
) -> dict[str, Any]:
    t0_qwen = admitted["t0_qwen"]
    t1_qwen = admitted["t1_qwen"]
    action = admitted["action_profile"]
    source_hashes = admitted["source_hashes"]
    document = {
        "schema": SCHEMA,
        "schema_version": SCHEMA_VERSION,
        "scope": (
            "actual source-F32 precision-control T0 preprocessing captured immediately before block 0; "
            "no block forward or K/V write"
            if precision_control_f32 else
            "actual source-BF16 T0 preprocessing captured immediately before block 0; no block forward or K/V write"
        ),
        "source": {
            "structured_control_schema": structured.SCHEMA,
            "conditioning_manifest_sha256": args.conditioning_manifest_sha256,
            "conditioning_payload_sha256": admitted["payload_sha256"],
            "t0_qwen_manifest_sha256": t0_qwen["manifest_sha256"],
            "t0_qwen_payload_sha256": t0_qwen["payload_sha256"],
            "t0_qwen_tensor_sha256": t0_qwen["qwen_record"]["sha256"],
            "t1_qwen_manifest_sha256": t1_qwen["manifest_sha256"],
            "t1_qwen_payload_sha256": t1_qwen["payload_sha256"],
            "t1_qwen_tensor_sha256": t1_qwen["qwen_record"]["sha256"],
            "action_profile_sha256": action["profile_sha256"],
            "clean_t0_latent_bank_sha256": admitted["sha256"],
            "cache_index_sha256": admitted["cache_index_sha256"],
            "global_shard_sha256": admitted["global_shard"]["sha256"],
            "upstream_revision": admitted["upstream_revision"],
            "upstream_source_sha256": source_hashes,
        },
        "execution": {
            "mode": "source-f32-precision-control" if precision_control_f32 else "source-bf16",
            "device": "mps",
            "t0_valid_text_tokens": admitted["t0_valid_tokens"],
            "dit_text_rows": admitted["dit_text_rows"],
            "block_loader_calls": 1,
            "loaded_dit_blocks": 0,
            "block_weight_tensor_loads": 0,
            "block_forward_calls": 0,
            "first_block_boundary_calls": 1,
            "terminal_cpu_exports": 1,
            "cache_kv_written": False,
            "observed_stage_order": list(STAGE_ORDER),
            "writer_call": admitted["forward_plan"][0],
            "source_dtype_by_tensor": _source_types(precision_control_f32),
            "export_dtype_by_tensor": _EXPORT_TYPES,
            "semantic_notes": [
                "cache.noisy_latent_f32 is the actual BF16-rounded source writer latent widened to F32 for file export",
                (
                    "the F32 precision-control prelude outputs are the exact tensors handed to source block 0 after widening the actual BF16 writer latent and Qwen input"
                    if precision_control_f32 else
                    "the other block inputs are the exact tensors handed to source block 0, widened only where required by the native fixture format"
                ),
                "this trace proves the preblock boundary only; it does not execute DiT block 0 or claim cache K/V parity",
                "CPU admission still authenticates the pinned 54-shard weight index and identities; the trace loader instantiates none of those block tensors",
            ],
        },
        "valid_text_tokens": admitted["t0_valid_tokens"],
        "payload": payload_name,
        "payload_bytes": len(payload),
        "payload_sha256": hashlib.sha256(payload).hexdigest(),
        "tensor_order": list(TENSOR_ORDER),
        "tensors": dict(tensor_records),
    }
    if precision_control_f32:
        document["execution"]["prelude_precision_control"] = {
            "module_weights": "exact pinned BF16 checkpoint values widened to F32",
            "refiner_attention_adapter": "F32 installed before the pinned refiner binds its imported attention symbol",
            "img_in_input": "actual source BF16 T0 writer latent widened to F32 at the module prehook",
            "txt_in_input": "actual authenticated source BF16 T0 Qwen tensor widened to F32 at the module prehook",
            "time_in_input": "unchanged source F32 writer timestep",
            "writer_blend": "unchanged source BF16 clean/noise blend and BF16 cache-epsilon input",
            "block_execution": "sentinel stops before block 0; no K/V write",
        }
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
        raise PreblockTraceError(f"could not publish fresh source preblock trace: {error}") from None
    return hashlib.sha256(encoded).hexdigest()


def _cpu_plan(admitted: Mapping[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    plan = {
        "schema": SCHEMA,
        "status": "validated_plan_only",
        "model_launched": False,
        "mps_device_used": False,
        "conditioning_manifest_sha256": args.conditioning_manifest_sha256,
        "conditioning_payload_sha256": admitted["payload_sha256"],
        "t0_qwen_manifest_sha256": admitted["t0_qwen"]["manifest_sha256"],
        "t1_qwen_manifest_sha256": admitted["t1_qwen"]["manifest_sha256"],
        "action_profile_sha256": admitted["action_profile"]["profile_sha256"],
        "clean_t0_latent_bank_sha256": admitted["sha256"],
        "cache_index_sha256": admitted["cache_index_sha256"],
        "upstream_revision": admitted["upstream_revision"],
        "valid_text_tokens": admitted["t0_valid_tokens"],
        "dit_text_rows": admitted["dit_text_rows"],
        "tensor_names": list(TENSOR_ORDER),
        "capture_boundary": "before source block 0",
        "planned_loaded_dit_blocks": 0,
    }
    if getattr(args, "precision_control_f32", False):
        plan["precision_mode"] = "source-f32-precision-control"
    return plan


def _execute(args: argparse.Namespace, admitted: dict[str, Any], base_args: argparse.Namespace,
             action_valid_tokens: int) -> dict[str, Any]:
    precision_control_f32 = getattr(args, "precision_control_f32", False)
    output, payload_path = output_paths(args.output)
    base.single._configure_mps_runtime()
    torch, scheduler, diffusers_version = base.single._prepare_cpu_scheduler(
        Path(admitted["upstream_path"]), "source-bf16"
    )
    original_factory = base._cache_attention_dispatch_factory

    def make_dispatch(runtime_torch: Any, state: dict[str, Any]) -> Any:
        return structured._action_cache_attention_dispatch_factory(
            runtime_torch, state, admitted["action_profile"],
            Path(admitted["upstream_path"]), legacy_factory=original_factory,
        )

    capture: dict[str, Any] = {"events": []}
    snapshot: dict[str, Any] | None = None
    try:
        base._cache_attention_dispatch_factory = make_dispatch
        with structured._bind_dynamic_base_contract(action_valid_tokens, admitted["dit_text_rows"]):
            with _install_runtime_patches(
                base, capture, precision_control_f32=precision_control_f32,
                expected_t0_qwen=admitted["t0_qwen"], expected_admitted=admitted,
            ):
                try:
                    with torch.inference_mode():
                        base._run_control(
                            admitted, base_args, torch, scheduler, diffusers_version
                        )
                except _StopBeforeBlock0:
                    pass
                else:
                    raise PreblockTraceError("source control returned without reaching the block-0 sentinel")
        snapshot = _snapshot_from_capture(capture, admitted["t0_valid_tokens"])
        validate_snapshot_layout(
            snapshot, admitted["t0_valid_tokens"], require_mps=True,
            precision_control_f32=precision_control_f32,
        )
        _validate_writer_call(
            capture, snapshot, admitted, precision_control_f32=precision_control_f32,
        )
        payload, records = _pack_payload(
            torch, snapshot, admitted["t0_valid_tokens"],
            precision_control_f32=precision_control_f32,
        )
    finally:
        base._cache_attention_dispatch_factory = original_factory
    document = _manifest(
        args, admitted, payload_path.name, payload, records,
        precision_control_f32=precision_control_f32,
    )
    manifest_sha = _publish(output, payload_path, document, payload)
    return {
        "schema": SCHEMA,
        "status": (
            "captured_source_f32_precision_control_t0_preblock"
            if precision_control_f32 else "captured_source_t0_preblock"
        ),
        "precision_mode": "source-f32-precision-control" if precision_control_f32 else "source-bf16",
        "manifest": str(output),
        "manifest_sha256": manifest_sha,
        "payload": str(payload_path),
        "payload_sha256": document["payload_sha256"],
        "payload_bytes": document["payload_bytes"],
        "loaded_dit_blocks": 0,
        "block_forward_calls": 0,
        "terminal_cpu_exports": 1,
        "tensor_names": list(TENSOR_ORDER),
    }


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    raw_argv = list(sys.argv[1:] if argv is None else argv)
    precision_flags = raw_argv.count("--precision-control-f32")
    if precision_flags > 1:
        raise PreblockTraceError("--precision-control-f32 may be supplied only once")
    original_builder = structured._argument_parser

    def extended_builder() -> argparse.ArgumentParser:
        parser = original_builder()
        parser.description = __doc__
        parser.add_argument(
            "--precision-control-f32",
            action="store_true",
            help="run the same T0 preblock path with F32 global prelude modules and inputs widened from BF16",
        )
        return parser

    structured._argument_parser = extended_builder
    try:
        args = structured._parse_args(raw_argv)
    finally:
        structured._argument_parser = original_builder
    if getattr(args, "precision_control_f32", False) != (precision_flags == 1):
        raise PreblockTraceError("precision-control argument parsing changed unexpectedly")
    return args


def main(argv: list[str] | None = None) -> int:
    try:
        args = _parse_args(argv)
        admitted, base_args, action_valid_tokens = structured._authenticate(args)
        if args.plan_only:
            print(json.dumps(_cpu_plan(admitted, args), sort_keys=True))
            return 0
        output_paths(args.output)
        result = _execute(args, admitted, base_args, action_valid_tokens)
    except (PreblockTraceError, structured.StructuredControlError, base.ControlError,
            base.single.ControlError, OSError, ValueError, RuntimeError) as error:
        print(json.dumps({"schema": SCHEMA, "status": "rejected", "reason": str(error)}, sort_keys=True), file=sys.stderr)
        return 2
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
