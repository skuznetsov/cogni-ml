#!/usr/bin/env python3
"""Capture actual source BF16 T0 token-refiner block boundaries before DiT block 0.

This wrapper reuses the authenticated source preblock/refiner trace runtime. It
adds observational hooks for the two pinned text-refiner blocks, keeps all
activations on MPS through the existing terminal pack, and emits the original
prelude trace plus a separate block-trace manifest and payload.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
import re
import sys
from contextlib import contextmanager
from typing import Any, Mapping

ROOT = Path(__file__).resolve().parents[2]
SUPPORT = ROOT / "spec" / "support"
EXAMPLE_SUPPORT = ROOT / "examples" / "support"
for _path in (SUPPORT, EXAMPLE_SUPPORT):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

import hy_nomas_reference_refiner_trace as source_trace  # noqa: E402

preblock = source_trace.preblock
structured = source_trace.structured
base = source_trace.base

SCHEMA = "cogni_ml.hy_nomas_reference_refiner_blocks_source_trace/v1"
SCHEMA_VERSION = 1
PINNED_REVISION = "911cea6458a89c2663513ff7099561ec190dd491"
VALID_TEXT_TOKENS = 263
TEXT_ROWS = base.QWEN_ROWS
HIDDEN_DIM = base.DIM
HEADS = base.HEADS
HEAD_DIM = base.HEAD_DIM
FFN_DIM = 4 * HIDDEN_DIM

_BLOCK_STAGES = (
    "condition_silu", "adaLN", "gate_msa", "gate_mlp", "norm1", "qkv",
    "q", "k", "v", "attention", "proj", "residual1", "norm2",
    "mlp_fc1", "mlp_silu", "mlp_fc2", "output",
)
BLOCK_TENSOR_ORDER = (
    ("block0.input", "c")
    + tuple(f"block0.{stage}" for stage in _BLOCK_STAGES)
    + ("block1.input",)
    + tuple(f"block1.{stage}" for stage in _BLOCK_STAGES)
)
BLOCK1_INPUT_ALIAS = {"block1.input": "block0.output"}
BLOCK_UNIQUE_PAYLOAD_ORDER = tuple(
    name for name in BLOCK_TENSOR_ORDER if name not in BLOCK1_INPUT_ALIAS
)
_HEX = re.compile(r"[0-9a-f]{64}\Z")
_TOKEN_REFINER_MODULE = "hymm.models.text_encoder.token_refiner"
_BLOCK_EVENT_ORDER = (
    "c",
    "block0.input", "block0.condition_silu", "block0.adaLN", "block0.norm1",
    "block0.qkv", "block0.q", "block0.k", "block0.v", "block0.attention",
    "block0.proj", "block0.gate_msa", "block0.residual1", "block0.norm2",
    "block0.mlp_fc1", "block0.mlp_silu", "block0.mlp_fc2", "block0.gate_mlp",
    "block0.output",
    "block1.input", "block1.condition_silu", "block1.adaLN", "block1.norm1",
    "block1.qkv", "block1.q", "block1.k", "block1.v", "block1.attention",
    "block1.proj", "block1.gate_msa", "block1.residual1", "block1.norm2",
    "block1.mlp_fc1", "block1.mlp_silu", "block1.mlp_fc2", "block1.gate_mlp",
    "block1.output",
)


class BlockTraceError(RuntimeError):
    """Fail-closed source block trace error."""


def expected_shapes(valid_text_tokens: int = VALID_TEXT_TOKENS) -> dict[str, tuple[int, ...]]:
    """Return canonical native-comparable shapes for the admitted V263 trace."""
    if type(valid_text_tokens) is not int or valid_text_tokens != VALID_TEXT_TOKENS:
        raise BlockTraceError("source block trace is scoped to admitted T0 V263")
    shapes: dict[str, tuple[int, ...]] = {
        "block0.input": (1, TEXT_ROWS, HIDDEN_DIM),
        "c": (1, HIDDEN_DIM),
    }
    for block in range(2):
        prefix = f"block{block}."
        shapes.update({
            prefix + "condition_silu": (1, HIDDEN_DIM),
            prefix + "adaLN": (1, 2 * HIDDEN_DIM),
            prefix + "gate_msa": (1, HIDDEN_DIM),
            prefix + "gate_mlp": (1, HIDDEN_DIM),
            prefix + "norm1": (1, TEXT_ROWS, HIDDEN_DIM),
            prefix + "qkv": (1, TEXT_ROWS, 3, HIDDEN_DIM),
            prefix + "q": (1, TEXT_ROWS, HIDDEN_DIM),
            prefix + "k": (1, TEXT_ROWS, HIDDEN_DIM),
            prefix + "v": (1, TEXT_ROWS, HIDDEN_DIM),
            prefix + "attention": (1, TEXT_ROWS, HIDDEN_DIM),
            prefix + "proj": (1, TEXT_ROWS, HIDDEN_DIM),
            prefix + "residual1": (1, TEXT_ROWS, HIDDEN_DIM),
            prefix + "norm2": (1, TEXT_ROWS, HIDDEN_DIM),
            prefix + "mlp_fc1": (1, TEXT_ROWS, FFN_DIM),
            prefix + "mlp_silu": (1, TEXT_ROWS, FFN_DIM),
            prefix + "mlp_fc2": (1, TEXT_ROWS, HIDDEN_DIM),
            prefix + "output": (1, TEXT_ROWS, HIDDEN_DIM),
        })
    shapes["block1.input"] = shapes["block0.output"]
    return {name: shapes[name] for name in BLOCK_TENSOR_ORDER}


def _tensor_dtype(tensor: Any) -> str:
    dtype = str(getattr(tensor, "dtype", "")).split(".")[-1]
    if dtype == "bool":
        return "BOOL"
    return preblock._dtype_name(tensor)


def _tensor_shape(tensor: Any) -> tuple[int, ...]:
    return preblock._shape(tensor)


def _tensor_device(tensor: Any) -> str | None:
    return preblock._device_type(tensor)


def _source_shape_allowed(name: str, actual: tuple[int, ...], canonical: tuple[int, ...]) -> bool:
    if name.endswith((".q", ".k", ".v")):
        return actual in (canonical, (1, TEXT_ROWS, HEADS, HEAD_DIM))
    return actual == canonical


def validate_snapshot(
    snapshot: Mapping[str, Any], *, valid_text_tokens: int = VALID_TEXT_TOKENS,
    require_mps: bool = True,
) -> dict[str, tuple[int, ...]]:
    """Validate exact 37-name order, full-row geometry, and source BF16 types."""
    shapes = expected_shapes(valid_text_tokens)
    if tuple(snapshot) != BLOCK_TENSOR_ORDER:
        raise BlockTraceError("block tensor inventory or order differs from the trace contract")
    for name in BLOCK_TENSOR_ORDER:
        tensor = snapshot[name]
        actual_shape = _tensor_shape(tensor)
        if not _source_shape_allowed(name, actual_shape, shapes[name]):
            raise BlockTraceError(
                f"source block tensor shape differs from the full-row contract: {name}"
            )
        if _tensor_dtype(tensor) != "BF16":
            raise BlockTraceError(f"source block tensor dtype differs from BF16: {name}")
        if require_mps and _tensor_device(tensor) != "mps":
            raise BlockTraceError(f"source block tensor did not remain on MPS: {name}")
    return shapes


def validate_input_aliases(captured: Mapping[str, Any]) -> None:
    """Require block 1 to consume the exact block 0 output tensor."""
    previous = captured.get("block0.output")
    current = captured.get("block1.input")
    if previous is None or current is not previous:
        raise BlockTraceError("block1.input is not the exact block0.output tensor")


def _record(state: dict[str, Any], name: str, tensor: Any) -> None:
    if not callable(getattr(tensor, "detach", None)) or not hasattr(tensor, "shape"):
        raise BlockTraceError(f"expected an observed source tensor at {name}")
    values = state.setdefault("values", {})
    if name in values:
        raise BlockTraceError(f"source block checkpoint executed more than once: {name}")
    values[name] = tensor
    state.setdefault("events", []).append(name)


def _install_attention_observer_after_bind(state: dict[str, Any]) -> None:
    """Wrap the adapter only after the source runtime's final adapter rebind."""
    if state.get("attention_hook_installed") is True:
        raise BlockTraceError("source refiner attention observer was installed more than once")
    token_module = state.get("token_module")
    expected_attention = state.get("original_attention")
    observer_factory = state.get("attention_observer_factory")
    if token_module is None or not callable(expected_attention) or not callable(observer_factory):
        raise BlockTraceError("source refiner attention observer was not prepared before adapter binding")
    if getattr(token_module, "attention", None) is not expected_attention:
        raise BlockTraceError("source refiner attention adapter changed before observer installation")
    refiner = state.get("refiner_instance")
    individual = getattr(refiner, "individual_token_refiner", None)
    blocks = getattr(individual, "blocks", ())
    if len(blocks) != 2:
        raise BlockTraceError("source refiner attention binding lost its two-block instance")
    for block in blocks:
        forward = getattr(block, "forward", None)
        implementation = getattr(forward, "__func__", forward)
        if getattr(implementation, "__globals__", None) is not token_module.__dict__:
            raise BlockTraceError("source refiner block forward is detached from the authenticated module globals")
    token_module.attention = observer_factory(expected_attention)
    state["attention_hook_installed"] = True


def _restore_block_globals(state: Mapping[str, Any]) -> None:
    """Restore module globals recorded before installing source observers."""
    for module, original_attention, original_apply_gate in reversed(
        state.get("restore_globals", ()),
    ):
        module.attention = original_attention
        module.apply_gate = original_apply_gate


@contextmanager
def _observe_refiner_adapter_binding(single: Any, state: dict[str, Any]):
    """Install attention observation after the loaded-refiner adapter rebind.

    The source runner binds its attention adapter once before module loading and
    again after loading. The latter assignment overwrites any earlier function
    observer, so install the observer immediately after that final binding.
    """
    binder_name = "_bind_refiner_attention_adapter"
    original_binder = getattr(single, binder_name, None)
    if not callable(original_binder):
        raise BlockTraceError("source runtime cannot bind its refiner attention adapter")

    def bind_adapter(*args: Any, **kwargs: Any) -> Any:
        result = original_binder(*args, **kwargs)
        module_instance = kwargs.get("module_instance")
        if module_instance is None and len(args) >= 4:
            module_instance = args[3]
        if module_instance is not None:
            refiner_instance = state.get("refiner_instance")
            # _prepare_cpu_scheduler runs a pinned call-contract preflight
            # before this trace loads its admitted refiner. Those synthetic
            # instances must retain the base binder unchanged.
            if refiner_instance is None:
                return result
            if module_instance is not refiner_instance:
                raise BlockTraceError("source runtime rebound attention for another refiner instance")
            _install_attention_observer_after_bind(state)
        return result

    setattr(single, binder_name, bind_adapter)
    try:
        yield
    finally:
        try:
            setattr(single, binder_name, original_binder)
        finally:
            _restore_block_globals(state)


def _install_block_hooks(
    modules: Mapping[str, Any], refiner_trace_state: dict[str, Any],
    state: dict[str, Any], *, upstream: Path,
) -> None:
    """Attach no-op observers to the two already loaded source blocks."""
    import sys as _sys

    refiner = modules.get("txt_in")
    individual = getattr(refiner, "individual_token_refiner", None)
    blocks = getattr(individual, "blocks", None)
    if refiner is None or individual is None or blocks is None or len(blocks) != 2:
        raise BlockTraceError("pinned source refiner did not load exactly two blocks")

    token_module = _sys.modules.get(_TOKEN_REFINER_MODULE)
    expected_path = (Path(upstream) / "hymm/models/text_encoder/token_refiner.py").resolve(strict=True)
    actual_path = Path(getattr(token_module, "__file__", "")).resolve(strict=True) if token_module else None
    if token_module is None or actual_path != expected_path:
        raise BlockTraceError("loaded token refiner is not from the authenticated pinned source")

    handles = refiner_trace_state.get("hook_handles")
    if not isinstance(handles, list):
        raise BlockTraceError("source refiner trace did not expose its scoped hook lifecycle")

    def save_output(module: Any, name: str, transform: Any = None) -> None:
        def hook(_module: Any, _inputs: tuple[Any, ...], output: Any) -> None:
            _record(state, name, transform(output) if transform is not None else output)
        handles.append(module.register_forward_hook(hook))

    def save_input(module: Any, name: str, index: int = 0) -> None:
        def hook(_module: Any, inputs: tuple[Any, ...]) -> None:
            if len(inputs) <= index:
                raise BlockTraceError(f"source module input is missing at {name}")
            _record(state, name, inputs[index])
        handles.append(module.register_forward_pre_hook(hook))

    def input_embedder_output(_module: Any, _inputs: tuple[Any, ...], output: Any) -> None:
        if state.get("input_embedder_output") is not None:
            raise BlockTraceError("source input embedder ran more than once")
        state["input_embedder_output"] = output

    handles.append(refiner.input_embedder.register_forward_hook(input_embedder_output))
    state["refiner_instance"] = refiner

    def save_c(_module: Any, inputs: tuple[Any, ...]) -> None:
        if len(inputs) < 2:
            raise BlockTraceError("source individual refiner did not receive c")
        value = inputs[1]
        source_c = refiner_trace_state.get("checkpoints", {}).get("c")
        if value is not source_c:
            raise BlockTraceError("block trace c differs from the authenticated source refiner c")
        state["c"] = value
        state.setdefault("events", []).append("c")

    handles.append(individual.register_forward_pre_hook(save_c))

    def individual_output(_module: Any, _inputs: tuple[Any, ...], output: Any) -> None:
        if state.get("individual_output") is not None:
            raise BlockTraceError("source individual refiner ran more than once")
        state["individual_output"] = output

    handles.append(individual.register_forward_hook(individual_output))

    for index, block in enumerate(blocks):
        prefix = f"block{index}."

        def save_block_input(_module: Any, inputs: tuple[Any, ...], *, block_index: int = index) -> None:
            if len(inputs) < 2:
                raise BlockTraceError(f"source refiner block {block_index} lacks input or c")
            x, c = inputs[:2]
            if c is not state.get("c"):
                raise BlockTraceError(f"source refiner block {block_index} consumed another c tensor")
            if block_index == 0:
                if x is not state.get("input_embedder_output"):
                    raise BlockTraceError("block0.input is detached from the full input_embedder output")
                _record(state, "block0.input", x)
            else:
                previous = state.get("values", {}).get("block0.output")
                if x is not previous:
                    raise BlockTraceError("block1.input is not the exact block0.output tensor")
                _record(state, "block1.input", x)

        handles.append(block.register_forward_pre_hook(save_block_input))
        save_output(block.adaLN_modulation[0], prefix + "condition_silu")
        save_output(block.adaLN_modulation[1], prefix + "adaLN")
        for module, name in (
            (block.norm1, prefix + "norm1"),
            (block.self_attn_qkv, prefix + "qkv"),
            (block.self_attn_proj, prefix + "proj"),
            (block.norm2, prefix + "norm2"),
            (block.mlp.fc1, prefix + "mlp_fc1"),
            (block.mlp.act, prefix + "mlp_silu"),
            (block.mlp.fc2, prefix + "mlp_fc2"),
        ):
            transform = None
            if name.endswith(".qkv"):
                transform = lambda output: output.view(
                    output.shape[0], output.shape[1], 3, HIDDEN_DIM,
                )
            save_output(module, name, transform)
        save_input(block.norm2, prefix + "residual1")

        def save_block_output(
            _module: Any, _inputs: tuple[Any, ...], output: Any, *, block_index: int = index,
        ) -> None:
            _record(state, f"block{block_index}.output", output)

        handles.append(block.register_forward_hook(save_block_output))

    original_attention = getattr(token_module, "attention", None)
    original_apply_gate = getattr(token_module, "apply_gate", None)
    if not callable(original_attention) or not callable(original_apply_gate):
        raise BlockTraceError("pinned token refiner attention/gate callables are unavailable")

    def attention_observer_factory(original: Any) -> Any:
        def attention(q: Any, k: Any, v: Any, drop_rate: float = 0,
                      attn_mask: Any = None, causal: bool = False) -> Any:
            call = int(state.get("attention_calls", 0))
            if call >= 2:
                raise BlockTraceError("source refiner executed more than two attention calls")
            expected_qkv = (1, TEXT_ROWS, HEADS, HEAD_DIM)
            if any(_tensor_shape(tensor) != expected_qkv for tensor in (q, k, v)):
                raise BlockTraceError("source attention Q/K/V layout differs from [1,1280,16,128]")
            if (attn_mask is None or _tensor_shape(attn_mask) != (1, TEXT_ROWS)
                    or _tensor_dtype(attn_mask) != "BOOL" or _tensor_device(attn_mask) != "mps"
                    or drop_rate != 0 or causal):
                raise BlockTraceError("source refiner attention escaped the full bool-mask contract")
            for name, tensor in zip(("q", "k", "v"), (q, k, v), strict=True):
                _record(state, f"block{call}.{name}", tensor)
            output = original(
                q, k, v, drop_rate=drop_rate, attn_mask=attn_mask, causal=causal,
            )
            _record(state, f"block{call}.attention", output)
            state["attention_calls"] = call + 1
            return output
        return attention

    def apply_gate(x: Any, gate: Any = None, tanh: bool = False) -> Any:
        call = int(state.get("gate_calls", 0))
        if call >= 4 or gate is None:
            raise BlockTraceError("source refiner gate call count differs from two gated residuals per block")
        block_index, position = divmod(call, 2)
        gate_name = "gate_msa" if position == 0 else "gate_mlp"
        _record(state, f"block{block_index}.{gate_name}", gate)
        state["gate_calls"] = call + 1
        return original_apply_gate(x, gate, tanh=tanh)

    state["token_module"] = token_module
    state["original_attention"] = original_attention
    state["attention_observer_factory"] = attention_observer_factory
    state["attention_hook_installed"] = False
    state.setdefault("restore_globals", []).append(
        (token_module, original_attention, original_apply_gate),
    )
    token_module.apply_gate = apply_gate


def _block_snapshot(state: Mapping[str, Any], refiner_trace_state: Mapping[str, Any]) -> dict[str, Any]:
    values = state.get("values")
    if not isinstance(values, dict):
        raise BlockTraceError("source refiner block checkpoints were not captured")
    if state.get("attention_hook_installed") is not True:
        raise BlockTraceError("source refiner attention observer was not installed after adapter binding")
    attention_calls = state.get("attention_calls", 0)
    if attention_calls != 2:
        raise BlockTraceError(
            f"source refiner attention call count differs: observed {attention_calls}, expected 2"
        )
    gate_calls = state.get("gate_calls", 0)
    if gate_calls != 4:
        raise BlockTraceError(
            f"source refiner gate call count differs: observed {gate_calls}, expected 4"
        )
    if tuple(state.get("events", ())) != _BLOCK_EVENT_ORDER:
        raise BlockTraceError("source refiner block checkpoints are missing, duplicated, or out of order")
    output = values.get("block1.output")
    if output is not state.get("individual_output"):
        raise BlockTraceError("final source refiner output differs from the observed individual-refiner result")
    if state.get("c") is not refiner_trace_state.get("checkpoints", {}).get("c"):
        raise BlockTraceError("captured source c differs from the pinned refiner argument")
    values["c"] = state["c"]
    if set(values) != set(BLOCK_TENSOR_ORDER):
        raise BlockTraceError("source refiner block checkpoint inventory is incomplete or unexpected")
    validate_input_aliases(values)
    snapshot = {name: values[name] for name in BLOCK_TENSOR_ORDER}
    validate_snapshot(snapshot)
    return snapshot


def _source_shapes(name: str, tensor: Any) -> tuple[int, ...]:
    if name.endswith((".q", ".k", ".v")):
        return (1, TEXT_ROWS, HEADS, HEAD_DIM)
    if name.endswith(".qkv"):
        return (1, TEXT_ROWS, 3, HIDDEN_DIM)
    return _tensor_shape(tensor)


def pack_snapshot(
    torch: Any, snapshot: Mapping[str, Any], valid_text_tokens: int,
    *, source_trace_state: Mapping[str, Any] | None = None,
    block_state: Mapping[str, Any] | None = None,
    precision_control_f32: bool = False,
) -> tuple[bytes, dict[str, dict[str, Any]], dict[str, str], dict[str, str]]:
    """Pack the old source payload and new block tensors in one terminal readback."""
    if precision_control_f32:
        raise BlockTraceError("source refiner block trace admits source-BF16 mode only")
    if source_trace_state is None or block_state is None:
        raise BlockTraceError("source and block hook states are required for terminal packing")
    block_snapshot = _block_snapshot(block_state, source_trace_state)
    shapes = source_trace.validate_snapshot_layout(snapshot, valid_text_tokens, require_mps=True)
    validate_snapshot(block_snapshot, valid_text_tokens=valid_text_tokens)

    source_types = source_trace._all_source_types(False)
    export_types = {
        **preblock._EXPORT_TYPES,
        **{name: "F32" for name in source_trace.COMMON_TENSOR_ORDER + source_trace.SOURCE_ONLY_TENSOR_ORDER},
    }
    byte_parts = []
    source_records: dict[str, dict[str, Any]] = {}
    offset = 0
    for name in source_trace.TENSOR_ORDER:
        tensor = snapshot[name].detach().contiguous()
        if export_types[name] == "F32":
            tensor = tensor.to(dtype=torch.float32)
        if source_trace._dtype_name(tensor) != export_types[name]:
            raise BlockTraceError(f"source prelude export dtype changed at {name}")
        byte_parts.append(tensor.view(torch.uint8).reshape(-1))
        nbytes = math.prod(shapes[name]) * (2 if export_types[name] == "BF16" else 4)
        source_records[name] = {
            "dtype": export_types[name], "shape": list(shapes[name]),
            "offset": offset, "nbytes": nbytes,
        }
        offset += nbytes
    source_bytes = offset

    block_records: dict[str, dict[str, Any]] = {}
    export_shapes = expected_shapes(valid_text_tokens)
    source_checkpoints = {
        name: name for name in BLOCK_TENSOR_ORDER
        if name not in BLOCK1_INPUT_ALIAS and name not in ("block0.input", "c")
    }
    source_checkpoints.update({"block0.input": "input_embed", "c": "c"})
    for name in BLOCK_UNIQUE_PAYLOAD_ORDER:
        tensor = block_snapshot[name].detach()
        if name.endswith(".qkv"):
            tensor = tensor.view(1, TEXT_ROWS, 3, HIDDEN_DIM)
        tensor = tensor.contiguous().to(dtype=torch.float32)
        if source_trace._dtype_name(block_snapshot[name]) != "BF16":
            raise BlockTraceError(f"source block tensor is not BF16 at {name}")
        byte_parts.append(tensor.view(torch.uint8).reshape(-1))
        shape = export_shapes[name]
        nbytes = math.prod(shape) * sizeof_float32()
        block_records[name] = {
            "dtype": "F32", "source_dtype": "BF16", "shape": list(shape),
            "source_shape": list(_source_shapes(name, block_snapshot[name])),
            "source_checkpoint": source_checkpoints[name],
            "offset": offset - source_bytes, "nbytes": nbytes,
        }
        offset += nbytes

    packed = torch.cat(byte_parts, dim=0).cpu()
    try:
        joined_payload = packed.numpy().tobytes(order="C")
    except (AttributeError, RuntimeError, TypeError) as error:
        raise BlockTraceError(f"terminal source tensor export failed: {error}") from None
    if len(joined_payload) != offset:
        raise BlockTraceError("terminal source block export byte extent differs from the inventory")
    source_payload = joined_payload[:source_bytes]
    block_payload = joined_payload[source_bytes:]

    for records, payload in ((source_records, source_payload), (block_records, block_payload)):
        for record in records.values():
            start = record["offset"]
            if records is source_records:
                raw = payload[start:start + record["nbytes"]]
            else:
                raw = payload[start:start + record["nbytes"]]
            record["sha256"] = hashlib.sha256(raw).hexdigest()

    alias_record = dict(block_records["block0.output"])
    alias_record.update({
        "source_checkpoint": "block0.output",
        "alias_of": "block0.output",
    })
    block_records["block1.input"] = alias_record
    block_records = {name: block_records[name] for name in BLOCK_TENSOR_ORDER}

    source_trace._validate_payload_finite(source_payload, source_records)
    source_trace._validate_refiner_mask_payload(source_payload, source_records, valid_text_tokens)
    source_trace._validate_payload_finite(block_payload, block_records)
    state = block_state
    if isinstance(state, dict):
        state["packed_payload"] = block_payload
        state["packed_records"] = block_records
    return source_payload, source_records, source_types, export_types


def sizeof_float32() -> int:
    return 4


def manifest_contract(
    *, source_manifest: Mapping[str, Any], source_manifest_sha256: str,
    payload_name: str, payload: bytes, records: Mapping[str, Mapping[str, Any]],
    valid_text_tokens: int = VALID_TEXT_TOKENS,
) -> dict[str, Any]:
    """Build the separate sidecar manifest linked to the exact source prelude."""
    shapes = expected_shapes(valid_text_tokens)
    if (source_manifest.get("schema") != source_trace.SCHEMA
            or source_manifest.get("execution", {}).get("mode") != "source-bf16"
            or source_manifest.get("execution", {}).get("device") != "mps"):
        raise BlockTraceError("block trace source manifest is not an MPS source-BF16 prelude")
    if (source_manifest.get("valid_text_tokens") != VALID_TEXT_TOKENS
            or source_manifest.get("source", {}).get("upstream_revision") != PINNED_REVISION):
        raise BlockTraceError("block trace source manifest differs from pinned T0 V263")
    if not isinstance(source_manifest_sha256, str) or _HEX.fullmatch(source_manifest_sha256) is None:
        raise BlockTraceError("source refiner manifest SHA-256 is malformed")
    if (not isinstance(source_manifest.get("payload_sha256"), str)
            or _HEX.fullmatch(source_manifest["payload_sha256"]) is None):
        raise BlockTraceError("source refiner payload SHA-256 is malformed")
    if set(records) != set(BLOCK_TENSOR_ORDER):
        raise BlockTraceError("block trace manifest tensor inventory is incomplete")

    source = source_manifest.get("source", {})
    source_fields = {
        "upstream_revision": source.get("upstream_revision"),
        "upstream_source_sha256": source.get("upstream_source_sha256"),
        "conditioning_manifest_sha256": source.get("conditioning_manifest_sha256"),
        "t0_qwen_manifest_sha256": source.get("t0_qwen_manifest_sha256"),
        "t0_qwen_payload_sha256": source.get("t0_qwen_payload_sha256"),
    }
    return {
        "schema": SCHEMA,
        "schema_version": SCHEMA_VERSION,
        "status": "captured_source_refiner_blocks",
        "scope": (
            "actual source-BF16 T0 V263 text-refiner block inputs and outputs on MPS; "
            "two text-refiner blocks only; no DiT block forward or K/V write"
        ),
        "source": source_fields,
        "execution": {
            "mode": "source-bf16",
            "device": "mps",
            "valid_text_tokens": VALID_TEXT_TOKENS,
            "processed_text_rows": TEXT_ROWS,
            "padded_text_rows": TEXT_ROWS - VALID_TEXT_TOKENS,
            "refiner_block_forward_calls": 2,
            "refiner_attention_calls": 2,
            "dit_block_forward_calls": 0,
            "loaded_dit_blocks": 0,
            "cache_kv_written": False,
            "terminal_cpu_exports": 1,
            "source_dtype_by_tensor": {name: "BF16" for name in BLOCK_TENSOR_ORDER},
            "export_dtype_by_tensor": {name: "F32" for name in BLOCK_TENSOR_ORDER},
            "unique_payload_tensor_count": len(BLOCK_UNIQUE_PAYLOAD_ORDER),
            "attention_backend": (
                "unpadded PyTorch MPS SDPA over the valid prefix with zero-padded query outputs; "
                "not pinned FlashAttention or Magi/CUDA parity"
            ),
            "attention_internal_checkpoints": "fused backend internals are not observed",
            "capture_method": "observational PyTorch module/function hooks; no neural recomputation",
        },
        "valid_text_tokens": VALID_TEXT_TOKENS,
        "processed_tokens": TEXT_ROWS,
        "padding_tokens": TEXT_ROWS - VALID_TEXT_TOKENS,
        "device": "mps",
        "producer_dtype": "BF16",
        "full_text_refiner_encode_executed": True,
        "full_upstream_t0_preprocessing_parity": False,
        "capture_boundary": "after_block1_output",
        "blocks_executed": 2,
        "tensor_order": list(BLOCK_TENSOR_ORDER),
        "unique_payload_tensor_order": list(BLOCK_UNIQUE_PAYLOAD_ORDER),
        "aliases": dict(BLOCK1_INPUT_ALIAS),
        "payload": payload_name,
        "payload_bytes": len(payload),
        "payload_sha256": hashlib.sha256(payload).hexdigest(),
        "tensors": {name: dict(records[name]) for name in BLOCK_TENSOR_ORDER},
        "expected_shapes": {name: list(shapes[name]) for name in BLOCK_TENSOR_ORDER},
        "source_refiner_manifest_sha256": source_manifest_sha256,
        "source_refiner_payload_sha256": source_manifest["payload_sha256"],
    }


def _output_paths(output: str | Path) -> tuple[Path, Path, Path, Path]:
    source_manifest, source_payload = preblock.output_paths(output)
    blocks_manifest = source_manifest.with_name(source_manifest.stem + ".blocks.json")
    blocks_payload = source_manifest.with_name(source_manifest.stem + ".blocks.tensors.bin")
    if (blocks_manifest.is_symlink() or blocks_payload.is_symlink()
            or os.path.lexists(blocks_manifest) or os.path.lexists(blocks_payload)):
        raise BlockTraceError("block trace sidecar manifest and payload must both be fresh files")
    return source_manifest, source_payload, blocks_manifest, blocks_payload


def _cpu_plan(admitted: Mapping[str, Any], args: Any) -> dict[str, Any]:
    plan = source_trace._cpu_plan(admitted, args)
    plan.update({
        "block_trace_schema": SCHEMA,
        "block_trace_tensor_order": list(BLOCK_TENSOR_ORDER),
        "block_trace_shapes": {name: list(shape) for name, shape in expected_shapes().items()},
        "block_trace_output": "sibling <source-output-stem>.blocks.json and .blocks.tensors.bin",
        "block_trace_terminal_cpu_exports": 1,
        "block_trace_neural_recomputation": False,
    })
    return plan


def _execute(
    args: Any, admitted: dict[str, Any], base_args: Any, action_valid_tokens: int,
) -> dict[str, Any]:
    _, _, blocks_manifest_path, blocks_payload_path = _output_paths(args.output)
    block_state: dict[str, Any] = {"attention_calls": 0, "gate_calls": 0}
    original_install = source_trace._install_refiner_hooks
    original_pack = source_trace._pack_payload

    def install(modules: Mapping[str, Any], trace_state: dict[str, Any],
                valid_tokens: int, embed_layers: Any) -> None:
        original_install(modules, trace_state, valid_tokens, embed_layers)
        block_state["trace_state"] = trace_state
        _install_block_hooks(
            modules, trace_state, block_state, upstream=Path(admitted["upstream_path"]),
        )

    def pack(torch: Any, snapshot: Mapping[str, Any], valid_tokens: int, *,
             precision_control_f32: bool = False):
        trace_state = block_state.get("trace_state")
        if not isinstance(trace_state, dict):
            raise BlockTraceError("block hooks were not installed before source packing")
        return pack_snapshot(
            torch, snapshot, valid_tokens, source_trace_state=trace_state,
            block_state=block_state, precision_control_f32=precision_control_f32,
        )

    from unittest.mock import patch

    with _observe_refiner_adapter_binding(base.single, block_state):
        with patch.object(source_trace, "_install_refiner_hooks", install):
            with patch.object(source_trace, "_pack_payload", pack):
                source_result = source_trace._execute(
                    args, admitted, base_args, action_valid_tokens,
                )

    source_manifest_path = Path(source_result["manifest"])
    source_manifest_bytes = source_manifest_path.read_bytes()
    source_manifest = json.loads(source_manifest_bytes)
    source_manifest_sha256 = hashlib.sha256(source_manifest_bytes).hexdigest()
    if source_manifest_sha256 != source_result["manifest_sha256"]:
        raise BlockTraceError("published source prelude manifest hash changed before block linkage")
    if block_state.get("packed_payload") is None or block_state.get("packed_records") is None:
        raise BlockTraceError("single terminal pack did not produce the block checkpoint payload")
    payload = block_state["packed_payload"]
    records = block_state["packed_records"]
    document = manifest_contract(
        source_manifest=source_manifest,
        source_manifest_sha256=source_manifest_sha256,
        payload_name=blocks_payload_path.name,
        payload=payload,
        records=records,
        valid_text_tokens=admitted["t0_valid_tokens"],
    )
    document["execution"]["observed_attention_calls"] = block_state["attention_calls"]
    document["execution"]["observed_gate_calls"] = block_state["gate_calls"]
    manifest_sha256 = source_trace._publish(
        blocks_manifest_path, blocks_payload_path, document, payload,
    )
    return {
        **source_result,
        "block_trace_manifest": str(blocks_manifest_path),
        "block_trace_manifest_sha256": manifest_sha256,
        "block_trace_payload": str(blocks_payload_path),
        "block_trace_payload_sha256": document["payload_sha256"],
        "block_trace_payload_bytes": len(payload),
        "block_trace_tensor_names": list(BLOCK_TENSOR_ORDER),
        "block_trace_unique_payload_tensor_count": len(BLOCK_UNIQUE_PAYLOAD_ORDER),
    }


def main(argv: list[str] | None = None) -> int:
    try:
        args = source_trace._parse_args(argv)
        admitted, base_args, action_valid_tokens = structured._authenticate(args)
        if args.plan_only:
            print(json.dumps(_cpu_plan(admitted, args), sort_keys=True))
            return 0
        if getattr(args, "precision_control_f32", False):
            raise BlockTraceError("source refiner block trace admits source-BF16 mode only")
        _output_paths(args.output)
        result = _execute(args, admitted, base_args, action_valid_tokens)
    except (BlockTraceError, source_trace.RefinerTraceError, preblock.PreblockTraceError,
            structured.StructuredControlError, base.ControlError,
            base.single.ControlError, OSError, ValueError, RuntimeError) as error:
        print(json.dumps({"schema": SCHEMA, "status": "rejected", "reason": str(error)}, sort_keys=True), file=sys.stderr)
        return 2
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
