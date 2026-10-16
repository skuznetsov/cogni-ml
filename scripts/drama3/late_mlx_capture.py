#!/usr/bin/env python3
"""Capture matched scalar/sequence MLX intermediates for DRAMA-3 late layers.

This diagnostic runs the pinned upstream text backbone over supplied F32
embeddings. It preserves the official operator outputs and selected Python
locals for decoder layers 34 through 39; it does not recompute captured stages.
All retained arrays are materialized once, after both arms have completed.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
from typing import Any

try:
    from . import prefill_intermediates as _prefill
except ImportError:  # Direct script invocation puts this directory on sys.path.
    import prefill_intermediates as _prefill


FORMAT = "drama3-late-matched-intermediates-v1"
DEFAULT_SELECTED_LAYERS = tuple(range(34, 40))

_MOE_STAGES = (
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
    "shared_gate_output",
    "shared_up_output",
    "shared_activation_output",
    "shared_expert_output",
    "shared_gate_logits",
    "shared_weighted_output",
    "moe_output",
    "canonical_rank_order",
    "canonical_expert_ids",
    "canonical_route_ids_from_switch",
    "canonical_selected_probabilities",
    "canonical_selected_weights",
    "canonical_expert_gate_outputs",
    "canonical_expert_up_outputs",
    "canonical_expert_activation_outputs",
    "canonical_expert_down_outputs",
    "canonical_expert_weighted_outputs",
)

_REQUIRED_MOE_STAGES = tuple(
    stage
    for stage in _MOE_STAGES
    if stage not in {
        "switch_inverse_order",
        "per_rank_prefix_sums",
        "canonical_rank_order",
        "canonical_expert_ids",
        "canonical_route_ids_from_switch",
        "canonical_selected_probabilities",
        "canonical_selected_weights",
        "canonical_expert_gate_outputs",
        "canonical_expert_up_outputs",
        "canonical_expert_activation_outputs",
        "canonical_expert_down_outputs",
        "canonical_expert_weighted_outputs",
    }
)

_MOE_RECORD_KEYS = {
    "routed_sum": "routed_sum_official",
    "moe_output": "moe_output_official",
}


def _missing_required_moe_stages(records: list[dict[str, Any]]) -> list[str]:
    """Report absent exported stages while honoring official-return record keys."""
    return [
        stage
        for stage in _REQUIRED_MOE_STAGES
        if not all(_MOE_RECORD_KEYS.get(stage, stage) in record for record in records)
    ]


def parse_layers(value: str) -> tuple[int, ...]:
    """Parse comma-separated layer IDs and inclusive ranges such as ``34-39``."""
    layers: list[int] = []
    try:
        for item in value.split(","):
            item = item.strip()
            if not item:
                raise ValueError
            if "-" in item:
                start_text, end_text = item.split("-", 1)
                start, end = int(start_text), int(end_text)
                if start < 0 or end < start:
                    raise ValueError
                layers.extend(range(start, end + 1))
            else:
                layer = int(item)
                if layer < 0:
                    raise ValueError
                layers.append(layer)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "layers must be non-negative IDs or inclusive ranges, e.g. 34-39"
        ) from exc
    if not layers or len(layers) != len(set(layers)):
        raise argparse.ArgumentTypeError("layers must be non-empty and contain no duplicates")
    return tuple(sorted(layers))


def capture_contract(selected_layers: tuple[int, ...]) -> dict[str, Any]:
    """Return the stable manifest description, independent of MLX availability."""
    return {
        "decoder_boundaries": ["attention_residual", "hidden"],
        "selected_moe_layers": list(selected_layers),
        "selected_decoder_operands": ["decoder_input", "attention_output"],
        "moe_stages": list(_MOE_STAGES),
        "expert_gate_up_order": (
            "Raw switch projection and activation tensors preserve the upstream layout. "
            "The sequence arm sorts routes when rows*top_k >= 64; scalar calls do not. "
            "switch_expert_ids and optional switch_inverse_order record that official "
            "projection order. canonical_* tensors restore official route order on GPU "
            "and then sort each token's routes by expert ID."
        ),
        "shared_stage_contract": (
            "shared_gate_output and shared_up_output are the official shared MLP "
            "projection returns; shared_activation_output is the exact input passed "
            "to the shared down projection; shared_expert_output is that projection's "
            "official return. shared_gate_logits is the separate scalar mixture gate, "
            "and shared_weighted_output is the official weighted shared contribution."
        ),
        "evaluation": "One mx.eval after both arm calls and all references are retained.",
    }


def write_used_input(
    output_dir: pathlib.Path,
    embeddings: Any,
    input_metadata: dict[str, Any],
) -> dict[str, Any]:
    """Export the exact used rows for native replay and bind them to metadata."""
    input_dir = output_dir / "input"
    input_dir.mkdir()
    used_embeddings = embeddings.astype("<f4", copy=False, order="C")
    input_bytes = used_embeddings.tobytes(order="C")
    digest = _prefill._sha256_bytes(input_bytes)
    if digest != input_metadata["used_f32_sha256"]:
        raise _prefill.CaptureError("exported input bytes do not match used_f32_sha256")
    relative_path = "input/used_embeddings.f32"
    (output_dir / relative_path).write_bytes(input_bytes)
    input_metadata = dict(input_metadata)
    input_metadata["used_f32_file"] = relative_path
    input_metadata["used_f32_bytes"] = len(input_bytes)
    return input_metadata


class LateLayerCapture:
    """Trace official decoder/MoE returns for several late layers at once."""

    def __init__(
        self,
        mx: Any,
        nn: Any,
        qwen3_5: Any,
        layers: list[Any],
        hidden_size: int,
        selected_layers: tuple[int, ...],
    ) -> None:
        self.mx = mx
        self.hidden_size = hidden_size
        if not selected_layers or len(selected_layers) != len(set(selected_layers)):
            raise _prefill.CaptureError("selected layers must be non-empty and unique")
        self.selected_layers = tuple(sorted(selected_layers))
        self.layer_ids = {id(layer): i for i, layer in enumerate(layers)}
        invalid = [layer for layer in selected_layers if layer < 0 or layer >= len(layers)]
        if invalid:
            raise _prefill.CaptureError(f"selected layers are outside the backbone: {invalid}")

        self.target_moes: dict[int, Any] = {}
        self.moe_layer_by_id: dict[int, int] = {}
        self.target_switches: dict[int, Any] = {}
        self.target_activations: dict[int, Any] = {}
        self.top_k: dict[int, int] = {}
        self.linear_stages: dict[int, tuple[int, str]] = {}
        for layer_id in selected_layers:
            moe = layers[layer_id].mlp
            if not isinstance(moe, qwen3_5.SparseMoeBlock):
                raise _prefill.CaptureError(f"layer {layer_id} is not a SparseMoeBlock")
            self.target_moes[id(moe)] = moe
            self.moe_layer_by_id[id(moe)] = layer_id
            self.target_switches[id(moe.switch_mlp)] = moe.switch_mlp
            self.target_activations[id(moe.switch_mlp.activation)] = moe.switch_mlp.activation
            self.top_k[layer_id] = int(moe.top_k)
            for module, stage in (
                (moe.gate, "router_logits"),
                (moe.shared_expert.gate_proj, "shared_gate_output"),
                (moe.shared_expert.up_proj, "shared_up_output"),
                (moe.shared_expert.down_proj, "shared_expert_output"),
                (moe.shared_expert_gate, "shared_gate_logits"),
            ):
                self.linear_stages[id(module)] = (layer_id, stage)

        first_switch = next(iter(self.target_switches.values()))
        self.decoder_code = qwen3_5.DecoderLayer.__call__.__code__
        self.sparse_code = qwen3_5.SparseMoeBlock.__call__.__code__
        self.switch_code = type(first_switch).__call__.__code__
        self.activation_code = type(first_switch.activation).__call__.__code__
        self.linear_code = nn.Linear.__call__.__code__
        self.arm: str | None = None
        self.refs: dict[str, dict[str, list[Any]]] = {"scalar": {}, "sequence": {}}
        self.moe_refs: dict[str, dict[int, list[dict[str, Any]]]] = {
            arm: {layer: [] for layer in selected_layers}
            for arm in ("scalar", "sequence")
        }
        self._layer_calls: dict[str, dict[int, int]] = {"scalar": {}, "sequence": {}}
        self._old_trace = sys.gettrace()
        self._old_array_sum = mx.array.sum
        self._active_moe: dict[str, Any] | None = None

    def _append(self, name: str, value: Any) -> None:
        assert self.arm is not None
        self.refs[self.arm].setdefault(name, []).append(value)

    def _trace_decoder(self, layer: int, seen: set[str]):
        assert self.arm is not None
        arm = self.arm
        key = f"backbone.layers.{layer}"

        def local_trace(frame: Any, event: str, arg: Any):
            if event == "line":
                values = frame.f_locals
                for local, stage in (
                    ("r", "attention_output"),
                    ("h", "attention_residual"),
                    ("out", "hidden"),
                ):
                    if local in values and stage not in seen:
                        self._append(f"{key}.{stage}", values[local])
                        seen.add(stage)
            elif event == "return":
                required = {"attention_residual", "hidden"}
                if layer in self.selected_layers:
                    required.update(("decoder_input", "attention_output"))
                missing = required - seen
                if missing:
                    raise _prefill.CaptureError(
                        f"layer {layer} did not expose locals {sorted(missing)}"
                    )
            return local_trace

        return local_trace

    def _trace_sparse(self, frame: Any, event: str, arg: Any):
        record = self._active_moe
        if record is None:
            raise _prefill.CaptureError("selected MoE trace fired without an active record")
        if event == "line":
            values = frame.f_locals
            raw_logits = record.get("router_logits")
            gates = values.get("gates")
            if gates is not None and raw_logits is not None and gates is not raw_logits:
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
                if "shared_expert_output" not in record:
                    record["shared_expert_output"] = shared_y
                elif shared_y is not record["shared_expert_output"]:
                    record["shared_weighted_output"] = shared_y
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
        module = frame.f_locals.get("self")
        mapped = self.linear_stages.get(id(module))
        record = self._active_moe
        if event == "return" and mapped is not None and record is not None:
            layer, stage = mapped
            if layer == record["layer"]:
                record[stage] = arg
        return self._trace_linear

    def _array_sum(self, array: Any, *args: Any, **kwargs: Any):
        result = self._old_array_sum(array, *args, **kwargs)
        record = self._active_moe
        if record is not None:
            layer = int(record["layer"])
            axis = args[0] if args else kwargs.get("axis")
            shape = tuple(int(dim) for dim in array.shape)
            if len(shape) >= 3 and shape[-2:] == (self.top_k[layer], self.hidden_size):
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
                seen: set[str] = set()
                if layer in self.selected_layers:
                    self._append(
                        f"backbone.layers.{layer}.decoder_input", frame.f_locals["x"]
                    )
                    seen.add("decoder_input")
                return self._trace_decoder(layer, seen)
            if code is self.sparse_code and id(module) in self.target_moes:
                if self.arm is None or self._active_moe is not None:
                    raise _prefill.CaptureError("nested or unselected selected-layer MoE capture")
                layer = self.moe_layer_by_id[id(module)]
                record: dict[str, Any] = {
                    "layer": layer,
                    "normalized_input": frame.f_locals["x"],
                }
                self.moe_refs[self.arm][layer].append(record)
                self._active_moe = record
                return self._trace_sparse
            if (
                self._active_moe is not None
                and code is self.switch_code
                and id(module) in self.target_switches
            ):
                return self._trace_switch
            if (
                self._active_moe is not None
                and code is self.activation_code
                and id(module) in self.target_activations
            ):
                return self._trace_activation
            if self._active_moe is not None and code is self.linear_code:
                mapped = self.linear_stages.get(id(module))
                if mapped is not None and mapped[0] == self._active_moe["layer"]:
                    if mapped[1] == "shared_expert_output":
                        self._active_moe["shared_activation_output"] = frame.f_locals["x"]
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

    @staticmethod
    def _rows_array(mx: Any, arm: str, refs: list[Any], rows: int):
        return _prefill.PrefillCapture._rows_array(mx, arm, refs, rows)

    def _moe_name(self, layer: int, stage: str) -> str:
        aliases = {"routed_sum_official": "routed_sum", "moe_output_official": "moe_output"}
        return f"backbone.layers.{layer}.mlp.{aliases.get(stage, stage)}"

    def arm_tensors(
        self,
        mx: Any,
        arm: str,
        final_outputs: list[Any],
        rows: int,
        layer_count: int,
    ) -> dict[str, Any]:
        if arm not in self.refs:
            raise _prefill.CaptureError(f"unknown capture arm {arm}")
        expected_calls = rows if arm == "scalar" else 1
        for layer in range(layer_count):
            if self._layer_calls[arm].get(layer) != expected_calls:
                raise _prefill.CaptureError(
                    f"{arm} arm called layer {layer} {self._layer_calls[arm].get(layer, 0)} times; "
                    f"expected {expected_calls}"
                )
        if len(final_outputs) != expected_calls:
            raise _prefill.CaptureError(f"{arm} arm retained {len(final_outputs)} final outputs")

        tensors: dict[str, Any] = {
            name: self._rows_array(mx, arm, refs, rows)
            for name, refs in self.refs[arm].items()
        }
        tensors["backbone.hidden"] = self._rows_array(mx, arm, final_outputs, rows)
        expected_hidden = (rows, self.hidden_size)
        for name, value in tensors.items():
            if (
                name == "backbone.hidden"
                or name.endswith(".attention_residual")
                or name.endswith(".hidden")
                or name.endswith(".decoder_input")
                or name.endswith(".attention_output")
            ):
                if tuple(int(dim) for dim in value.shape) != expected_hidden:
                    raise _prefill.CaptureError(
                        f"{arm} {name} has shape {value.shape}; expected {expected_hidden}"
                    )

        for layer in self.selected_layers:
            layer_refs = self.moe_refs[arm][layer]
            if len(layer_refs) != expected_calls:
                raise _prefill.CaptureError(
                    f"{arm} captured {len(layer_refs)} MoE calls for layer {layer}; expected {expected_calls}"
                )
            missing = _missing_required_moe_stages(layer_refs)
            if missing:
                raise _prefill.CaptureError(f"layer {layer} omitted selected MoE stages: {missing}")
            for stage in _MOE_STAGES:
                if stage == "switch_inverse_order":
                    refs = [record[stage] for record in layer_refs if stage in record]
                    if refs:
                        if len(refs) != expected_calls:
                            raise _prefill.CaptureError(
                                f"incomplete layer {layer} route inverse-order metadata"
                            )
                        tensors[self._moe_name(layer, stage)] = self._rows_array(
                            mx, arm, refs, rows
                        )
                    continue
                if stage.startswith("canonical_") or stage == "per_rank_prefix_sums":
                    continue
                source_stage = _MOE_RECORD_KEYS.get(stage, stage)
                refs = [record[source_stage] for record in layer_refs]
                tensors[self._moe_name(layer, stage)] = self._rows_array(mx, arm, refs, rows)

            weighted_name = self._moe_name(layer, "expert_weighted_outputs")
            weighted = tensors[weighted_name]
            prefixes = [weighted[..., 0, :]]
            for rank in range(1, self.top_k[layer]):
                prefixes.append(prefixes[-1] + weighted[..., rank, :])
            tensors[self._moe_name(layer, "per_rank_prefix_sums")] = mx.stack(
                prefixes, axis=-2
            )
            tensors.update(
                _prefill._canonicalize_moe_routes(
                    mx,
                    tensors,
                    rows=rows,
                    top_k=self.top_k[layer],
                    layer=layer,
                )
            )
        return tensors


def run_capture(
    *,
    mlx_root: pathlib.Path,
    model_dir: pathlib.Path,
    output_dir: pathlib.Path,
    inputs_path: pathlib.Path | None = None,
    input_raw_path: pathlib.Path | None = None,
    input_width: int | None = None,
    rows: int | None = None,
    selected_layers: tuple[int, ...] = DEFAULT_SELECTED_LAYERS,
) -> dict[str, Any]:
    output_dir = output_dir.resolve()
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite existing output directory: {output_dir}")
    mx, nn, qwen3_5, model, metadata = _prefill._load_backbone(mlx_root, model_dir)
    embeddings, input_metadata = _prefill.read_input_embeddings(
        inputs_path=inputs_path,
        input_raw_path=input_raw_path,
        input_width=input_width,
        expected_width=int(metadata["args"].hidden_size),
        rows=rows,
    )
    row_count = int(embeddings.shape[0])
    hidden_size = int(metadata["args"].hidden_size)
    layer_count = len(model.backbone.layers)
    capture = LateLayerCapture(
        mx, nn, qwen3_5, model.backbone.layers, hidden_size, selected_layers
    )
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
    # Keep all MLX work lazy through both arms; perform a single terminal sync.
    mx.eval(*sequence_tensors.values(), *scalar_tensors.values())

    output_dir.mkdir(parents=True)
    input_metadata = write_used_input(output_dir, embeddings, input_metadata)
    sequence_records = _prefill._write_arm(output_dir, "sequence", sequence_tensors)
    scalar_records = _prefill._write_arm(output_dir, "scalar", scalar_tensors)
    model_dir = model_dir.resolve()
    mlx_root = mlx_root.resolve()
    source_hashes = {
        **metadata["source_hashes"],
        "scripts/drama3/replay_mlx_backbone.py": _prefill._sha256_file(
            pathlib.Path(__file__).with_name("replay_mlx_backbone.py")
        ),
        "scripts/drama3/prefill_intermediates.py": _prefill._sha256_file(
            pathlib.Path(_prefill.__file__)
        ),
        "scripts/drama3/late_mlx_capture.py": _prefill._sha256_file(pathlib.Path(__file__)),
    }
    manifest = {
        "format": FORMAT,
        "scope": (
            "Offline direct Qwen3_5TextModel backbone calls over supplied identical input embeddings. "
            "This records upstream observations only and makes no native parity claim."
        ),
        "model": {
            "repository": metadata["model_repository"],
            "revision": metadata["model_revision"],
        },
        "upstream": {
            "repository": _prefill.UPSTREAM_REPOSITORY,
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
            **capture_contract(selected_layers),
            "decoder_layer_count": layer_count,
        },
        "tolerances": {
            "intermediate": {
                "atol": _prefill.INTERMEDIATE_ATOL,
                "rtol": _prefill.RTOL,
                "status": "existing unchanged contract",
            },
            "endpoint": {
                "atol": _prefill.ENDPOINT_ATOL,
                "rtol": _prefill.RTOL,
                "status": "existing unchanged contract",
            },
        },
        "known_open_gate": {
            "status": "not_closed",
            "scope": "Prior matched-input native/upstream row 1 layer 39 down-path mismatch.",
            "coordinate": {"row": 1, "layer": 39, "hidden_index": 33},
            "note": "This capture does not resolve or waive the existing MoE39/down discrepancy.",
        },
        "source_hashes": source_hashes,
        "arms": {
            "sequence": {**cache_contracts["sequence"], "tensors": sequence_records},
            "scalar": {**cache_contracts["scalar"], "tensors": scalar_records},
        },
    }
    (output_dir / "manifest.json").write_text(
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
    parser.add_argument("--rows", type=int, help="capture this prefix row count; defaults to all rows")
    parser.add_argument(
        "--layers",
        type=parse_layers,
        default=DEFAULT_SELECTED_LAYERS,
        help="selected MoE layers as IDs/ranges (default: 34-39)",
    )
    parser.add_argument("--output-dir", type=pathlib.Path, required=True)
    args = parser.parse_args(argv)
    manifest = run_capture(
        mlx_root=args.mlx_root,
        model_dir=args.model_dir,
        inputs_path=args.inputs,
        input_raw_path=args.input_raw,
        input_width=args.input_width,
        rows=args.rows,
        selected_layers=tuple(args.layers),
        output_dir=args.output_dir,
    )
    print(
        json.dumps(
            {
                "output_dir": str(args.output_dir.resolve()),
                "rows": manifest["input"]["used_rows"],
                "selected_layers": list(args.layers),
                "sequence_tensors": len(manifest["arms"]["sequence"]["tensors"]),
                "scalar_tensors": len(manifest["arms"]["scalar"]["tensors"]),
                "manifest": str((args.output_dir / "manifest.json").resolve()),
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
