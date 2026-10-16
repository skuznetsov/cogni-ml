#!/usr/bin/env python3
"""Capture scalar and full-sequence pinned MLX backbone prefill intermediates.

This offline diagnostic accepts already-computed input embeddings. It invokes
the official Qwen3.5 text backbone once with the full row sequence and once per
row with an independent cache. It retains upstream intermediate references and
materializes them only after both runs have completed.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import pathlib
import platform
import re
import subprocess
import sys
from typing import Any

import numpy as np


FORMAT = "drama3-backbone-prefill-intermediates-v1"
UPSTREAM_REPOSITORY = "https://github.com/cguldogan/mlx-audio"
DEFAULT_MOE_LAYER = 39
INTERMEDIATE_ATOL = 6e-5
ENDPOINT_ATOL = 3e-5
RTOL = 3e-4
_SAFE_TENSOR_NAME = re.compile(r"^[A-Za-z0-9_.-]+$")


class CaptureError(ValueError):
    """Raised when capture inputs or retained source stages are incomplete."""


def _route_values_in_original_order(
    mx: Any,
    values: Any,
    *,
    rows: int,
    top_k: int,
    inverse_order: Any | None,
    name: str,
) -> Any:
    """Return route values as [token, original-rank, channel], without host reads."""
    shape = tuple(int(dim) for dim in values.shape)
    if len(shape) == 4 and shape[:3] == (rows, top_k, 1):
        values = values[:, :, 0, :]
        shape = tuple(int(dim) for dim in values.shape)
    elif len(shape) == 3 and shape[0] == rows * top_k and shape[1] == 1:
        values = values[:, 0, :]
        shape = tuple(int(dim) for dim in values.shape)

    if len(shape) == 3 and shape[:2] == (rows, top_k):
        return values
    if len(shape) != 2 or shape[0] != rows * top_k:
        raise CaptureError(
            f"{name} has route shape {shape}; expected [rows, top_k, channels] "
            "or flattened [rows*top_k, channels]"
        )
    if inverse_order is None:
        raise CaptureError(f"{name} is flattened but has no upstream inverse route order")
    inverse_shape = tuple(int(dim) for dim in inverse_order.shape)
    if int(np.prod(inverse_shape)) != rows * top_k:
        raise CaptureError(
            f"{name} inverse route order has shape {inverse_shape}; "
            f"expected {rows * top_k} indices"
        )
    # Upstream _scatter_unsort uses sorted_values[inv_order] to restore each
    # original token/rank slot. Array indexing remains an MLX GPU operation.
    values = values[inverse_order.reshape((-1,))]
    return values.reshape((rows, top_k, shape[-1]))


def _canonicalize_moe_routes(
    mx: Any,
    tensors: dict[str, Any],
    *,
    rows: int,
    top_k: int,
    layer: int,
) -> dict[str, Any]:
    """Add MLX GPU views sorted by expert ID while preserving upstream tensors."""
    base = f"backbone.layers.{layer}.mlp."

    def get(stage: str) -> Any:
        name = base + stage
        if name not in tensors:
            raise CaptureError(f"cannot canonicalize routes without {name}")
        return tensors[name]

    expert_ids = get("expert_ids")
    if tuple(int(dim) for dim in expert_ids.shape) != (rows, top_k):
        raise CaptureError(
            f"expert_ids has shape {expert_ids.shape}; expected {(rows, top_k)}"
        )
    rank_order = mx.argsort(expert_ids, axis=-1)
    canonical: dict[str, Any] = {
        base + "canonical_rank_order": rank_order,
        base + "canonical_expert_ids": mx.take_along_axis(
            expert_ids, rank_order, axis=-1
        ),
    }

    switch_ids = get("switch_expert_ids")
    switch_shape = tuple(int(dim) for dim in switch_ids.shape)
    inverse_order = tensors.get(base + "switch_inverse_order")
    if len(switch_shape) == 1:
        if inverse_order is None:
            raise CaptureError("flattened switch expert IDs lack upstream inverse order")
        reconstructed_ids = switch_ids[inverse_order.reshape((-1,))].reshape(
            (rows, top_k)
        )
    elif switch_shape == (rows, top_k):
        reconstructed_ids = switch_ids
    else:
        raise CaptureError(
            f"switch_expert_ids has shape {switch_shape}; expected {(rows, top_k)} "
            "or a flattened sorted route vector"
        )
    canonical[base + "canonical_route_ids_from_switch"] = mx.take_along_axis(
        reconstructed_ids, rank_order, axis=-1
    )

    for stage in ("selected_weights", "selected_probabilities"):
        raw_name = base + stage
        if raw_name not in tensors:
            continue
        values = tensors[raw_name]
        if tuple(int(dim) for dim in values.shape) != (rows, top_k):
            raise CaptureError(
                f"{stage} has shape {values.shape}; expected {(rows, top_k)}"
            )
        canonical[base + "canonical_" + stage] = mx.take_along_axis(
            values, rank_order, axis=-1
        )

    inverse = tensors.get(base + "switch_inverse_order")
    for stage in (
        "expert_gate_outputs",
        "expert_up_outputs",
        "expert_activation_outputs",
        "expert_down_outputs",
        "expert_weighted_outputs",
    ):
        values = _route_values_in_original_order(
            mx,
            get(stage),
            rows=rows,
            top_k=top_k,
            inverse_order=inverse,
            name=stage,
        )
        expanded_order = mx.broadcast_to(
            mx.expand_dims(rank_order, axis=-1), values.shape
        )
        canonical[base + "canonical_" + stage] = mx.take_along_axis(
            values, expanded_order, axis=-2
        )
    return canonical


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path: pathlib.Path) -> tuple[bytes, dict[str, Any]]:
    raw = path.read_bytes()
    value = json.loads(raw.decode("utf-8"))
    if not isinstance(value, dict):
        raise CaptureError("input JSON must contain an object")
    return raw, value


def read_input_embeddings(
    *,
    inputs_path: pathlib.Path | None,
    input_raw_path: pathlib.Path | None,
    input_width: int | None,
    expected_width: int,
    rows: int | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Load an `input_embeddings` JSON bundle or a little-endian raw F32 matrix."""
    if (inputs_path is None) == (input_raw_path is None):
        raise CaptureError("select exactly one of --inputs or --input-raw")

    if inputs_path is not None:
        raw, bundle = _read_json(inputs_path)
        if "input_embeddings" not in bundle:
            raise CaptureError("input JSON must contain `input_embeddings`")
        values = np.asarray(bundle["input_embeddings"], dtype=np.float32)
        source = {
            "path": inputs_path.name,
            "format": "json.input_embeddings",
            "sha256": _sha256_bytes(raw),
            "bytes": len(raw),
        }
    else:
        if input_width is None or input_width <= 0:
            raise CaptureError("--input-width must be positive with --input-raw")
        if input_width != expected_width:
            raise CaptureError(
                f"raw input width {input_width} differs from model hidden size {expected_width}"
            )
        assert input_raw_path is not None
        raw = input_raw_path.read_bytes()
        row_bytes = input_width * np.dtype("<f4").itemsize
        if not raw or len(raw) % row_bytes:
            raise CaptureError(
                f"raw F32 input has {len(raw)} bytes; expected a non-empty multiple of {row_bytes}"
            )
        values = np.frombuffer(raw, dtype=np.dtype("<f4")).reshape(-1, input_width)
        source = {
            "path": input_raw_path.name,
            "format": "raw.little_endian.float32",
            "sha256": _sha256_bytes(raw),
            "bytes": len(raw),
            "width": input_width,
        }

    if values.ndim != 2 or values.shape[1] != expected_width:
        raise CaptureError(
            f"input_embeddings must have shape [rows, {expected_width}], got {values.shape}"
        )
    if not values.shape[0] or not np.isfinite(values).all():
        raise CaptureError("input_embeddings must be non-empty and finite")
    if rows is not None:
        if rows <= 0 or rows > values.shape[0]:
            raise CaptureError(f"--rows must be between 1 and {values.shape[0]}")
        values = values[:rows]

    values = np.asarray(values, dtype=np.dtype("<f4"), order="C")
    source.update(
        {
            "available_rows": int(
                np.frombuffer(raw, dtype=np.dtype("<f4")).size // expected_width
                if input_raw_path is not None
                else np.asarray(bundle["input_embeddings"]).shape[0]
            ),
            "used_rows": int(values.shape[0]),
            "width": int(values.shape[1]),
            "used_f32_sha256": _sha256_bytes(values.tobytes(order="C")),
        }
    )
    return values, source


def _load_backbone(
    mlx_root: pathlib.Path, model_dir: pathlib.Path
) -> tuple[Any, Any, Any, Any, dict[str, Any]]:
    from replay_mlx_backbone import (
        MLX_REVISION,
        MODEL_REPOSITORY,
        MODEL_REVISION,
        load_selected_weights,
        selected_backbone_map,
    )

    import mlx.core as mx
    import mlx.nn as nn

    mlx_root = mlx_root.resolve()
    model_dir = model_dir.resolve()
    actual_revision = subprocess.check_output(
        ["git", "-C", str(mlx_root), "rev-parse", "HEAD"], text=True
    ).strip()
    if actual_revision != MLX_REVISION:
        raise RuntimeError(f"MLX-audio source is {actual_revision}; expected {MLX_REVISION}")

    sys.path.insert(0, str(mlx_root))
    from mlx_audio.lm.models import qwen3_5
    from mlx_audio.lm.models.qwen3_5 import Qwen3_5TextModel, TextModelArgs
    from mlx_audio.utils import apply_quantization

    mx.set_default_device(mx.gpu)
    if mx.default_device() != mx.gpu:
        raise RuntimeError(f"failed to select MLX GPU device: {mx.default_device()}")

    config_path = model_dir / "config.json"
    index_path = model_dir / "model.safetensors.index.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    index = json.loads(index_path.read_text(encoding="utf-8"))
    selected = selected_backbone_map(index["weight_map"])
    text_config = config["text_config"]
    args = TextModelArgs.from_dict(text_config)
    layer_types = text_config.get("layer_types")
    if layer_types is not None:
        expected_types = [
            "full_attention" if (i + 1) % args.full_attention_interval == 0 else "linear_attention"
            for i in range(args.num_hidden_layers)
        ]
        if list(layer_types) != expected_types:
            raise ValueError("checkpoint layer_types do not match pinned Qwen3.5 interval")

    class BackboneOnly(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.backbone = Qwen3_5TextModel(args)

    model = BackboneOnly()
    apply_quantization(model, config, selected)
    loaded = load_selected_weights(model, model_dir, selected, mx)
    if loaded is not None:
        raise AssertionError("unexpected weight hash return without hash_prefix")
    model.eval()

    source_files = {
        "mlx_audio/lm/models/qwen3_5.py": mlx_root / "mlx_audio/lm/models/qwen3_5.py",
        "mlx_audio/lm/models/switch_layers.py": mlx_root / "mlx_audio/lm/models/switch_layers.py",
        "mlx_audio/lm/models/activations.py": mlx_root / "mlx_audio/lm/models/activations.py",
        "mlx_audio/lm/models/base.py": mlx_root / "mlx_audio/lm/models/base.py",
        "mlx_audio/lm/models/cache.py": mlx_root / "mlx_audio/lm/models/cache.py",
        "mlx_audio/lm/models/gated_delta.py": mlx_root / "mlx_audio/lm/models/gated_delta.py",
        "mlx_audio/lm/models/rope_utils.py": mlx_root / "mlx_audio/lm/models/rope_utils.py",
        "mlx_audio/utils.py": mlx_root / "mlx_audio/utils.py",
    }
    metadata = {
        "args": args,
        "config_path": config_path,
        "index_path": index_path,
        "selected": selected,
        "model_repository": MODEL_REPOSITORY,
        "model_revision": MODEL_REVISION,
        "source_hashes": {
            **{name: _sha256_file(path) for name, path in source_files.items()},
            "config.json": _sha256_file(config_path),
            "model.safetensors.index.json": _sha256_file(index_path),
        },
        "upstream_revision": actual_revision,
        "runtime": {
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "mlx": importlib.metadata.version("mlx"),
            "safetensors": importlib.metadata.version("safetensors"),
            "torch": importlib.metadata.version("torch"),
            "numpy": np.__version__,
            "device": str(mx.default_device()),
        },
    }
    return mx, nn, qwen3_5, model, metadata


class PrefillCapture:
    """Retain official decoder and selected MoE locals through Python tracing."""

    def __init__(
        self,
        mx: Any,
        nn: Any,
        qwen3_5: Any,
        layers: list[Any],
        hidden_size: int,
        moe_layer: int,
    ) -> None:
        if not 0 <= moe_layer < len(layers):
            raise CaptureError(f"MoE layer {moe_layer} is outside {len(layers)} layers")
        self.mx = mx
        self.hidden_size = hidden_size
        self.moe_layer = moe_layer
        self.layer_ids = {id(layer): i for i, layer in enumerate(layers)}
        self.target_layer = layers[moe_layer]
        self.target_moe = self.target_layer.mlp
        if not isinstance(self.target_moe, qwen3_5.SparseMoeBlock):
            raise CaptureError(f"layer {moe_layer} is not an upstream SparseMoeBlock")
        self.top_k = int(self.target_moe.top_k)
        self.target_switch_mlp = self.target_moe.switch_mlp
        self.target_activation = self.target_switch_mlp.activation
        self.decoder_code = qwen3_5.DecoderLayer.__call__.__code__
        self.sparse_code = qwen3_5.SparseMoeBlock.__call__.__code__
        self.switch_code = type(self.target_switch_mlp).__call__.__code__
        self.activation_code = type(self.target_activation).__call__.__code__
        self.linear_code = nn.Linear.__call__.__code__
        self.arm: str | None = None
        self.refs: dict[str, dict[str, list[Any]]] = {"scalar": {}, "sequence": {}}
        self.moe_refs: dict[str, list[dict[str, Any]]] = {"scalar": [], "sequence": []}
        self._layer_calls: dict[str, dict[int, int]] = {"scalar": {}, "sequence": {}}
        self._old_trace = sys.gettrace()
        self._old_array_sum = mx.array.sum
        self._active_moe: dict[str, Any] | None = None

    def _append(self, table: dict[str, list[Any]], name: str, value: Any) -> None:
        table.setdefault(name, []).append(value)

    def _trace_decoder(self, layer: int):
        assert self.arm is not None
        arm = self.arm
        seen: set[str] = set()
        key = f"backbone.layers.{layer}"

        def local_trace(frame: Any, event: str, arg: Any):
            if event == "line":
                values = frame.f_locals
                if "h" in values and "attention_residual" not in seen:
                    self._append(
                        self.refs[arm], f"{key}.attention_residual", values["h"]
                    )
                    seen.add("attention_residual")
                if "out" in values and "hidden" not in seen:
                    self._append(self.refs[arm], f"{key}.hidden", values["out"])
                    seen.add("hidden")
            elif event == "return":
                missing = {"attention_residual", "hidden"} - seen
                if missing:
                    raise CaptureError(f"layer {layer} did not expose locals {sorted(missing)}")
            return local_trace

        return local_trace

    def _trace_sparse(self, frame: Any, event: str, arg: Any):
        record = self._active_moe
        if record is None:
            raise CaptureError("target MoE trace fired without an active capture record")
        if event == "line":
            values = frame.f_locals
            raw_gates = record.get("router_logits")
            gates = values.get("gates")
            if gates is not None and raw_gates is not None and gates is not raw_gates:
                record.setdefault("router_probabilities", gates)
            inds = values.get("inds")
            if inds is not None:
                record.setdefault("expert_ids", inds)
            scores = values.get("scores")
            if scores is not None:
                if "selected_probabilities" not in record:
                    record["selected_probabilities"] = scores
                elif scores is not record["selected_probabilities"]:
                    record.setdefault("selected_weights", scores)
            y = values.get("y")
            if y is not None:
                record.setdefault("expert_down_outputs", y)
            shared_y = values.get("shared_y")
            if shared_y is not None:
                record.setdefault("shared_expert_output", shared_y)
        elif event == "return":
            record["moe_output_official"] = arg
            shared = frame.f_locals.get("shared_y")
            if shared is not None:
                record["shared_weighted_output"] = shared
            self._active_moe = None
        return self._trace_sparse

    def _trace_switch(self, frame: Any, event: str, arg: Any):
        if event == "line" and self._active_moe is not None:
            values = frame.f_locals
            idx = values.get("idx")
            if idx is not None:
                # In the sequence arm, wait until _gather_sort has replaced
                # idx and populated inv_order. In the scalar arm, do_sort is
                # false and idx is already in original token/rank order.
                do_sort = bool(values.get("do_sort", False))
                inverse = values.get("inv_order")
                if not do_sort or inverse is not None:
                    self._active_moe.setdefault("switch_expert_ids", idx)
                    if inverse is not None:
                        self._active_moe.setdefault("switch_inverse_order", inverse)
            for local, stage in (
                ("x_gate", "expert_gate_outputs"),
                ("x_up", "expert_up_outputs"),
            ):
                if local in values:
                    self._active_moe.setdefault(stage, values[local])
        return self._trace_switch

    def _trace_activation(self, frame: Any, event: str, arg: Any):
        if event == "return" and self._active_moe is not None:
            self._active_moe.setdefault("expert_activation_outputs", arg)
        return self._trace_activation

    def _trace_linear(self, frame: Any, event: str, arg: Any):
        if (
            event == "return"
            and self._active_moe is not None
            and frame.f_locals.get("self") is self.target_moe.gate
        ):
            self._active_moe["router_logits"] = arg
        elif (
            event == "return"
            and self._active_moe is not None
            and frame.f_locals.get("self") is self.target_moe.shared_expert_gate
        ):
            self._active_moe["shared_gate_logits"] = arg
        return self._trace_linear

    def _array_sum(self, array: Any, *args: Any, **kwargs: Any):
        result = self._old_array_sum(array, *args, **kwargs)
        record = self._active_moe
        if record is not None:
            axis = args[0] if args else kwargs.get("axis")
            shape = tuple(int(dim) for dim in array.shape)
            if len(shape) >= 3 and shape[-2:] == (self.top_k, self.hidden_size):
                normalized_axis = axis if axis is None or axis >= 0 else len(shape) + axis
                if normalized_axis == len(shape) - 2:
                    record["expert_weighted_outputs"] = array
                    record["routed_sum_official"] = result
        return result

    def trace(self, frame: Any, event: str, arg: Any):
        if event == "call":
            code = frame.f_code
            module = frame.f_locals.get("self")
            if code is self.decoder_code and id(module) in self.layer_ids:
                layer = self.layer_ids[id(module)]
                assert self.arm is not None
                calls = self._layer_calls[self.arm]
                calls[layer] = calls.get(layer, 0) + 1
                return self._trace_decoder(layer)
            if code is self.sparse_code and module is self.target_moe:
                if self.arm is None or self._active_moe is not None:
                    raise CaptureError("nested or unselected target MoE capture")
                record: dict[str, Any] = {"normalized_input": frame.f_locals["x"]}
                self.moe_refs[self.arm].append(record)
                self._active_moe = record
                return self._trace_sparse
            if (
                self._active_moe is not None
                and code is self.switch_code
                and module is self.target_switch_mlp
            ):
                return self._trace_switch
            if (
                self._active_moe is not None
                and code is self.activation_code
                and module is self.target_activation
            ):
                return self._trace_activation
            if self._active_moe is not None and code is self.linear_code:
                return self._trace_linear
        return None

    def install(self) -> None:
        capture = self

        def traced_sum(array: Any, *args: Any, **kwargs: Any):
            return capture._array_sum(array, *args, **kwargs)

        self.mx.array.sum = traced_sum
        sys.settrace(self.trace)

    def uninstall(self) -> None:
        sys.settrace(self._old_trace)
        self.mx.array.sum = self._old_array_sum
        self._active_moe = None

    def arm_tensors(
        self,
        mx: Any,
        arm: str,
        final_outputs: list[Any],
        rows: int,
        layer_count: int,
    ) -> dict[str, Any]:
        if arm not in self.refs:
            raise CaptureError(f"unknown capture arm {arm}")
        expected_calls = rows if arm == "scalar" else 1
        for layer in range(layer_count):
            if self._layer_calls[arm].get(layer) != expected_calls:
                raise CaptureError(
                    f"{arm} arm called layer {layer} {self._layer_calls[arm].get(layer, 0)} times; "
                    f"expected {expected_calls}"
                )
        if len(final_outputs) != expected_calls:
            raise CaptureError(f"{arm} arm retained {len(final_outputs)} final outputs")

        tensors: dict[str, Any] = {}
        for name, refs in self.refs[arm].items():
            tensors[name] = self._rows_array(mx, arm, refs, rows)
        tensors["backbone.hidden"] = self._rows_array(mx, arm, final_outputs, rows)
        expected_hidden = (rows, self.hidden_size)
        if tensors["backbone.hidden"].shape != expected_hidden:
            raise CaptureError(
                f"{arm} backbone.hidden has shape {tensors['backbone.hidden'].shape}; "
                f"expected {expected_hidden}"
            )
        for layer in range(layer_count):
            for stage in ("attention_residual", "hidden"):
                name = f"backbone.layers.{layer}.{stage}"
                if name not in tensors or tensors[name].shape != expected_hidden:
                    shape = tensors[name].shape if name in tensors else None
                    raise CaptureError(
                        f"{arm} {name} has shape {shape}; expected {expected_hidden}"
                    )

        moe_records = self.moe_refs[arm]
        if len(moe_records) != expected_calls:
            raise CaptureError(
                f"{arm} arm captured {len(moe_records)} layer {self.moe_layer} MoE calls; "
                f"expected {expected_calls}"
            )
        stage_names = (
            "normalized_input",
            "router_logits",
            "router_probabilities",
            "expert_ids",
            "selected_probabilities",
            "selected_weights",
            "expert_gate_outputs",
            "expert_up_outputs",
            "expert_activation_outputs",
            "expert_down_outputs",
            "expert_weighted_outputs",
            "routed_sum_official",
            "shared_expert_output",
            "shared_gate_logits",
            "shared_weighted_output",
            "moe_output_official",
            "switch_expert_ids",
        )
        for stage in stage_names:
            refs = [record[stage] for record in moe_records if stage in record]
            if len(refs) == expected_calls:
                tensors[self._moe_tensor_name(stage)] = (
                    self._rows_array(mx, arm, refs, rows)
                )
        required = (
            "normalized_input",
            "router_logits",
            "router_probabilities",
            "expert_ids",
            "selected_probabilities",
            "selected_weights",
            "expert_gate_outputs",
            "expert_up_outputs",
            "expert_activation_outputs",
            "expert_down_outputs",
            "expert_weighted_outputs",
            "routed_sum_official",
            "shared_expert_output",
            "shared_gate_logits",
            "shared_weighted_output",
            "moe_output_official",
            "switch_expert_ids",
        )
        missing = [stage for stage in required if not all(stage in record for record in moe_records)]
        if missing:
            raise CaptureError(f"layer {self.moe_layer} omitted selected MoE stages: {missing}")

        inverse_refs = [
            record["switch_inverse_order"]
            for record in moe_records
            if "switch_inverse_order" in record
        ]
        if inverse_refs:
            if len(inverse_refs) != expected_calls:
                raise CaptureError("incomplete route inverse-order metadata")
            tensors[self._moe_tensor_name("switch_inverse_order")] = self._rows_array(
                mx, arm, inverse_refs, rows
            )

        weighted_name = self._moe_tensor_name("expert_weighted_outputs")
        weighted = tensors[weighted_name]
        # The prefixes sum the exact per-expert weighted contributions that
        # the official MoE immediately reduces. No projections are recomputed.
        prefixes = [weighted[..., 0, :]]
        for rank in range(1, self.top_k):
            prefixes.append(prefixes[-1] + weighted[..., rank, :])
        tensors[self._moe_tensor_name("per_rank_prefix_sums")] = mx.stack(prefixes, axis=-2)
        tensors.update(
            _canonicalize_moe_routes(
                mx,
                tensors,
                rows=rows,
                top_k=self.top_k,
                layer=self.moe_layer,
            )
        )
        return tensors

    def _moe_tensor_name(self, stage: str) -> str:
        aliases = {
            "routed_sum_official": "routed_sum",
            "moe_output_official": "moe_output",
        }
        return f"backbone.layers.{self.moe_layer}.mlp.{aliases.get(stage, stage)}"

    @staticmethod
    def _rows_array(mx: Any, arm: str, refs: list[Any], rows: int):
        if not refs:
            raise CaptureError("cannot assemble an empty captured stage")
        if arm == "sequence":
            if len(refs) != 1:
                raise CaptureError("sequence stage must have one retained reference")
            ref = refs[0]
            return ref[0] if len(ref.shape) > 0 and int(ref.shape[0]) == 1 else ref
        if len(refs) != rows:
            raise CaptureError(f"scalar stage has {len(refs)} rows; expected {rows}")
        if all(
            len(ref.shape) >= 2 and int(ref.shape[0]) == 1 and int(ref.shape[1]) == 1
            for ref in refs
        ):
            return mx.concatenate(refs, axis=1)[0]
        return mx.concatenate(refs, axis=0)


def _write_arm(
    output_dir: pathlib.Path, arm: str, tensors: dict[str, Any]
) -> list[dict[str, Any]]:
    arm_dir = output_dir / arm
    arm_dir.mkdir()
    records: list[dict[str, Any]] = []
    for name in sorted(tensors):
        if not _SAFE_TENSOR_NAME.fullmatch(name):
            raise CaptureError(f"unsafe tensor name {name!r}")
        value = np.asarray(tensors[name])
        if value.dtype.kind in "iu":
            dtype = "int32"
            array = np.asarray(value, dtype=np.dtype("<i4"), order="C")
            extension = "i32"
        else:
            dtype = "float32"
            array = np.asarray(value, dtype=np.dtype("<f4"), order="C")
            if not np.isfinite(array).all():
                raise FloatingPointError(f"{arm}/{name}: non-finite float32 value")
            extension = "f32"
        data = array.tobytes(order="C")
        relative = pathlib.Path(arm) / f"{name}.{extension}"
        (output_dir / relative).write_bytes(data)
        records.append(
            {
                "name": name,
                "file": relative.as_posix(),
                "shape": [int(dim) for dim in array.shape],
                "dtype": dtype,
                "bytes": len(data),
                "sha256": _sha256_bytes(data),
            }
        )
    return records


def run_capture(
    *,
    mlx_root: pathlib.Path,
    model_dir: pathlib.Path,
    output_dir: pathlib.Path,
    inputs_path: pathlib.Path | None = None,
    input_raw_path: pathlib.Path | None = None,
    input_width: int | None = None,
    rows: int | None = None,
    moe_layer: int = DEFAULT_MOE_LAYER,
) -> dict[str, Any]:
    output_dir = output_dir.resolve()
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite existing output directory: {output_dir}")
    mx, nn, qwen3_5, model, metadata = _load_backbone(mlx_root, model_dir)
    embeddings, input_metadata = read_input_embeddings(
        inputs_path=inputs_path,
        input_raw_path=input_raw_path,
        input_width=input_width,
        expected_width=int(metadata["args"].hidden_size),
        rows=rows,
    )
    row_count = int(embeddings.shape[0])
    hidden_size = int(metadata["args"].hidden_size)
    layer_count = len(model.backbone.layers)
    capture = PrefillCapture(mx, nn, qwen3_5, model.backbone.layers, hidden_size, moe_layer)
    input_mx = mx.array(embeddings[None, :, :])
    sequence_outputs: list[Any] = []
    scalar_outputs: list[Any] = []
    cache_contracts: dict[str, Any] = {}

    try:
        capture.install()
        capture.arm = "sequence"
        sequence_cache = model.backbone.make_cache()
        sequence_outputs.append(
            model.backbone(
                inputs=mx.zeros((1, row_count), dtype=mx.int32),
                cache=sequence_cache,
                input_embeddings=input_mx,
            )
        )
        cache_contracts["sequence"] = {
            "fresh_cache": True,
            "call_count": 1,
            "input_shape": [1, row_count, hidden_size],
            "implicit_positions": list(range(row_count)),
            "chunk_size": row_count,
        }

        capture.arm = "scalar"
        scalar_cache = model.backbone.make_cache()
        for row in range(row_count):
            scalar_outputs.append(
                model.backbone(
                    inputs=mx.zeros((1, 1), dtype=mx.int32),
                    cache=scalar_cache,
                    input_embeddings=input_mx[:, row : row + 1, :],
                )
            )
        cache_contracts["scalar"] = {
            "fresh_cache": True,
            "call_count": row_count,
            "input_shape_per_call": [1, 1, hidden_size],
            "implicit_positions": list(range(row_count)),
            "chunk_size": 1,
        }
    finally:
        capture.uninstall()

    sequence_tensors = capture.arm_tensors(
        mx, "sequence", sequence_outputs, row_count, layer_count
    )
    scalar_tensors = capture.arm_tensors(mx, "scalar", scalar_outputs, row_count, layer_count)
    # One terminal evaluation covers both arms and every retained layer/MoE
    # reference. Host materialization begins only after this call.
    mx.eval(*sequence_tensors.values(), *scalar_tensors.values())

    output_dir.mkdir(parents=True)
    sequence_records = _write_arm(output_dir, "sequence", sequence_tensors)
    scalar_records = _write_arm(output_dir, "scalar", scalar_tensors)

    model_dir = model_dir.resolve()
    mlx_root = mlx_root.resolve()
    source_hashes = {
        **metadata["source_hashes"],
        "scripts/drama3/replay_mlx_backbone.py": _sha256_file(
            pathlib.Path(__file__).with_name("replay_mlx_backbone.py")
        ),
        "scripts/drama3/prefill_intermediates.py": _sha256_file(pathlib.Path(__file__)),
    }
    manifest = {
        "format": FORMAT,
        "scope": (
            "Offline direct Qwen3_5TextModel backbone calls over supplied identical input embeddings. "
            "This is not an end-to-end Fish Omni tower capture and makes no native parity claim."
        ),
        "model": {
            "repository": metadata["model_repository"],
            "revision": metadata["model_revision"],
        },
        "upstream": {
            "repository": UPSTREAM_REPOSITORY,
            "revision": metadata["upstream_revision"],
            "source_root": str(mlx_root),
        },
        "checkpoint": {
            "model_dir": str(model_dir),
            "config_sha256": metadata["source_hashes"]["config.json"],
            "index_sha256": metadata["source_hashes"]["model.safetensors.index.json"],
            "selected_backbone_tensor_count": len(metadata["selected"]),
            "selected_backbone_shards": sorted(set(metadata["selected"].values())),
            "shard_sha256_recorded": False,
        },
        "runtime": metadata["runtime"],
        "input": input_metadata,
        "cache_position_contract": (
            "Each arm creates a fresh upstream backbone cache; positions are implicit cache offsets 0..T-1. "
            "Both arms receive the same F32 input embedding rows."
        ),
        "capture_contract": {
            "decoder_boundaries": ["attention_residual", "hidden"],
            "selected_moe_layer": moe_layer,
            "moe_stages": [
                "normalized_input",
                "router_logits",
                "router_probabilities",
                "expert_ids",
                "switch_expert_ids",
                "switch_inverse_order",
                "selected_probabilities",
                "selected_weights",
                "expert_gate_outputs",
                "expert_up_outputs",
                "expert_activation_outputs",
                "expert_down_outputs",
                "expert_weighted_outputs",
                "routed_sum",
                "per_rank_prefix_sums",
            "canonical_rank_order",
            "canonical_expert_ids",
            "canonical_route_ids_from_switch",
            "canonical_selected_weights",
            "canonical_expert_gate_outputs",
            "canonical_expert_up_outputs",
            "canonical_expert_activation_outputs",
            "canonical_expert_down_outputs",
            "canonical_expert_weighted_outputs",
                "shared_expert_output",
                "shared_gate_logits",
                "shared_weighted_output",
                "moe_output",
            ],
            "expert_gate_up_order": (
                "Exact SwitchGLU gate/up outputs before the pinned scatter-unsort. The sequence "
                "arm sorts routes when rows*top_k >= 64; scalar calls do not. `switch_expert_ids` "
                "records the route expert IDs in projection order and `switch_inverse_order` "
                "records the upstream scatter-unsort index. `expert_ids` remains the original "
                "argpartition order. Gate/up tensor shapes are recorded verbatim because the "
                "upstream sorted and unsorted layouts differ. `canonical_*` tensors restore the "
                "official inverse route order on GPU, then sort each token's routes by expert ID. "
                "`canonical_route_ids_from_switch` is the independently reconstructed ID table "
                "to falsify an incorrect inverse-order mapping."
            ),
            "switch_inverse_order_presence": (
                "Present only when the upstream SwitchGLU sorts routes (indices.size >= 64); "
                "one-row scalar calls for this pinned model remain below that threshold."
            ),
            "per_rank_prefix_sums": (
                "Diagnostic cumulative additions over the exact official weighted expert outputs; "
                "no router or expert projection is recomputed."
            ),
            "evaluation": "One mx.eval after both arm calls and all references are retained.",
        },
        "tolerances": {
            "intermediate": {"atol": INTERMEDIATE_ATOL, "rtol": RTOL, "status": "existing unchanged contract"},
            "endpoint": {"atol": ENDPOINT_ATOL, "rtol": RTOL, "status": "existing unchanged contract"},
        },
        "known_open_gate": {
            "status": "not_closed",
            "scope": "Prior matched-input native/upstream row 1 layer 39 down-path mismatch.",
            "coordinate": {"row": 1, "layer": 39, "hidden_index": 33},
            "note": "This capture records both upstream arms; it does not resolve or waive the existing MoE39/down discrepancy.",
        },
        "source_hashes": source_hashes,
        "arms": {
            "sequence": {**cache_contracts["sequence"], "tensors": sequence_records},
            "scalar": {**cache_contracts["scalar"], "tensors": scalar_records},
        },
    }
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return manifest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mlx-root", type=pathlib.Path, required=True)
    parser.add_argument("--model-dir", type=pathlib.Path, required=True)
    inputs = parser.add_mutually_exclusive_group(required=True)
    inputs.add_argument("--inputs", type=pathlib.Path, help="JSON bundle with input_embeddings")
    inputs.add_argument("--input-raw", type=pathlib.Path, help="little-endian raw F32 [rows,width]")
    parser.add_argument("--input-width", type=int, help="required raw input width; must match model hidden size")
    parser.add_argument("--rows", type=int, help="capture this prefix row count; defaults to all input rows")
    parser.add_argument("--moe-layer", type=int, default=DEFAULT_MOE_LAYER)
    parser.add_argument("--output-dir", type=pathlib.Path, required=True)
    args = parser.parse_args(argv)
    manifest = run_capture(
        mlx_root=args.mlx_root,
        model_dir=args.model_dir,
        inputs_path=args.inputs,
        input_raw_path=args.input_raw,
        input_width=args.input_width,
        rows=args.rows,
        moe_layer=args.moe_layer,
        output_dir=args.output_dir,
    )
    print(
        json.dumps(
            {
                "output_dir": str(args.output_dir.resolve()),
                "rows": manifest["input"]["used_rows"],
                "sequence_tensors": len(manifest["arms"]["sequence"]["tensors"]),
                "scalar_tensors": len(manifest["arms"]["scalar"]["tensors"]),
                "manifest": str((args.output_dir / "manifest.json").resolve()),
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
