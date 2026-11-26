#!/usr/bin/env python3
"""Capture pinned MLX MoE intermediates for one layer without changing its math.

The full-run mode executes the complete pinned backbone and retains the
row-1/layer-39 MLX values from the official SparseMoeBlock frame. The isolated
mode replays the same official block from the emitted normalized-input fixture.
All host conversion happens after the selected MLX values have been evaluated.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import pathlib
import platform
import subprocess
import sys
from typing import Any

import numpy as np
import torch

from replay_mlx_backbone import (
    MLX_REVISION,
    MODEL_REPOSITORY,
    MODEL_REVISION,
    load_selected_weights,
    selected_backbone_map,
    sha256_file,
)


INPUT_FORMAT = "drama3-moe-input-v1"
TRACE_FORMAT = "drama3-moe-layer-diagnostic-v1"
REFERENCE_TRACE_FORMAT = "drama3-reference-traces-v1"
UPSTREAM_REPOSITORY = "https://github.com/cguldogan/mlx-audio"
TARGET_ROW = 1
TARGET_LAYER = 39
TARGET_ACTIVATION_NAME = "post_attention_layernorm"


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def activation_bytes(values: Any) -> bytes:
    array = np.asarray(values, dtype=np.dtype("<f4"))
    if not np.isfinite(array).all():
        raise FloatingPointError("MoE activation contains non-finite values")
    return array.tobytes(order="C")


def activation_sha256(values: Any) -> str:
    return sha256_bytes(activation_bytes(values))


def read_json_bytes(path: pathlib.Path) -> tuple[bytes, Any]:
    raw = path.read_bytes()
    return raw, json.loads(raw.decode("utf-8"))


def target_moe_weight_hash(
    model_dir: pathlib.Path, selected: dict[str, str]
) -> dict[str, Any]:
    """Hash the raw checkpoint tensors used by the target layer's MoE."""
    from safetensors import safe_open

    prefix = f"backbone.layers.{TARGET_LAYER}.mlp."
    by_shard: dict[str, list[str]] = {}
    for name, shard in selected.items():
        if name.startswith(prefix):
            by_shard.setdefault(shard, []).append(name)
    names = sorted(name for shard_names in by_shard.values() for name in shard_names)
    if not names:
        raise ValueError(f"checkpoint index has no tensors under {prefix}")

    digest = hashlib.sha256()
    dtypes: dict[str, str] = {}
    shapes: dict[str, list[int]] = {}
    for shard in sorted(by_shard):
        with safe_open(str(model_dir / shard), framework="pt", device="cpu") as tensors:
            available = set(tensors.keys())
            missing = [name for name in by_shard[shard] if name not in available]
            if missing:
                raise ValueError(f"{shard}: missing target MoE tensors: {missing[:5]}")
            for name in sorted(by_shard[shard]):
                tensor = tensors.get_tensor(name)
                raw = tensor.contiguous().view(torch.uint8).numpy()
                dtype = str(tensor.dtype).removeprefix("torch.")
                shape = [int(dimension) for dimension in tensor.shape]
                digest.update(name.encode("utf-8") + b"\0")
                digest.update(dtype.encode("ascii") + b"\0")
                digest.update(np.asarray(shape, dtype="<i8").tobytes())
                digest.update(raw)
                dtypes[name] = dtype
                shapes[name] = shape
                del raw, tensor
    return {
        "sha256": digest.hexdigest(),
        "tensor_names": names,
        "tensor_dtypes": dtypes,
        "tensor_shapes": shapes,
    }


class UpstreamCapture:
    """Retain selected references through the original MLX computation."""

    def __init__(
        self,
        mx: Any,
        qwen3_5: Any,
        nn: Any,
        layers: list[Any],
        hidden_size: int,
    ) -> None:
        self.mx = mx
        self.hidden_size = hidden_size
        self.layer_ids = {id(layer): index for index, layer in enumerate(layers)}
        if not 0 <= TARGET_LAYER < len(layers):
            raise ValueError(f"model has no layer {TARGET_LAYER}")
        self.target_layer = layers[TARGET_LAYER]
        self.target_moe = self.target_layer.mlp
        if not isinstance(self.target_moe, qwen3_5.SparseMoeBlock):
            raise TypeError(f"layer {TARGET_LAYER} does not use SparseMoeBlock")
        self.target_switch_mlp = self.target_moe.switch_mlp
        self.target_activation = self.target_switch_mlp.activation
        self.decoder_code = qwen3_5.DecoderLayer.__call__.__code__
        self.sparse_code = qwen3_5.SparseMoeBlock.__call__.__code__
        self.switch_mlp_code = type(self.target_switch_mlp).__call__.__code__
        self.activation_code = type(self.target_activation).__call__.__code__
        self.linear_code = nn.Linear.__call__.__code__
        self.current_row = 0
        self.active_target = False
        self.target_call_count = 0
        self.stage_refs: dict[str, Any] = {}
        self.control_refs: dict[str, Any] = {}
        self.prefix_ref: Any | None = None
        self._old_array_sum = mx.array.sum
        self._old_sigmoid = mx.sigmoid
        self._old_trace = sys.gettrace()

    def _store_once(self, name: str, value: Any) -> None:
        if name not in self.stage_refs:
            self.stage_refs[name] = value

    def _trace_target_sparse(self, frame: Any, event: str, arg: Any) -> Any:
        if event == "line":
            values = frame.f_locals
            gates = values.get("gates")
            if gates is not None:
                if "router_logits" not in self.stage_refs:
                    self._store_once("router_logits", gates)
                elif "router_softmax" not in self.stage_refs and gates is not self.stage_refs["router_logits"]:
                    self._store_once("router_softmax", gates)

            if "inds" in values:
                self._store_once("expert_ids_argpartition_order", values["inds"])
            if "scores" in values:
                self._store_once("selected_probabilities", values["scores"])
                if "y" in values:
                    self._store_once("selected_weights", values["scores"])
            if "y" in values:
                self._store_once("expert_down_outputs", values["y"])
            if "shared_y" in values:
                self._store_once("shared_expert_output", values["shared_y"])
            return self._trace_target_sparse

        if event == "return":
            self.stage_refs["moe_output_official"] = arg
            # A line trace fires before the statement runs, so observing the
            # second assignment to shared_y on a line event sees the raw shared
            # expert result. At return, the local is the exact gated value
            # consumed by the official `return y + shared_y` expression.
            shared_y = frame.f_locals.get("shared_y")
            if shared_y is None:
                raise RuntimeError("SparseMoeBlock returned without local shared_y")
            self.stage_refs["shared_weighted_output"] = shared_y
            self.active_target = False
        return self._trace_target_sparse

    def _trace_decoder(self, frame: Any, event: str, arg: Any) -> Any:
        if event == "line":
            layer = self.layer_ids[id(frame.f_locals["self"])]
            values = frame.f_locals
            if "h" in values:
                self.control_refs[
                    f"backbone.row{self.current_row}.layers.{layer}.attention_residual"
                ] = values["h"]
            if "out" in values:
                self.control_refs[
                    f"backbone.row{self.current_row}.layers.{layer}.hidden"
                ] = values["out"]
        return self._trace_decoder

    def _trace_switch_mlp(self, frame: Any, event: str, arg: Any) -> Any:
        if event == "line":
            values = frame.f_locals
            # These are the actual expert projection returns consumed by the
            # official activation call. Do not recompute either projection.
            if "x_gate" in values:
                self._store_once("expert_gate_outputs", values["x_gate"])
            if "x_up" in values:
                self._store_once("expert_up_outputs", values["x_up"])
        return self._trace_switch_mlp

    def _trace_activation(self, frame: Any, event: str, arg: Any) -> Any:
        if event == "return":
            # Capture the exact SwiGLU return that SwitchGLU passes to down_proj.
            self._store_once("expert_activation_outputs", arg)
        return self._trace_activation

    def _trace_linear(self, frame: Any, event: str, arg: Any) -> Any:
        if event == "return":
            linear = frame.f_locals.get("self")
            if linear is self.target_moe.gate:
                self.stage_refs["router_logits"] = arg
            elif linear is self.target_moe.shared_expert_gate:
                self.stage_refs["shared_gate_logits"] = arg
        return self._trace_linear

    def trace(self, frame: Any, event: str, arg: Any) -> Any:
        if event == "call":
            code = frame.f_code
            module = frame.f_locals.get("self")
            if code is self.decoder_code and id(module) in self.layer_ids:
                return self._trace_decoder
            if (
                code is self.sparse_code
                and module is self.target_moe
                and self.current_row == TARGET_ROW
            ):
                self.target_call_count += 1
                if self.target_call_count > 1:
                    raise RuntimeError("target MoE was invoked more than once")
                self.active_target = True
                self.stage_refs = {"normalized_input": frame.f_locals["x"]}
                return self._trace_target_sparse
            if (
                code is self.switch_mlp_code
                and module is self.target_switch_mlp
                and self.active_target
            ):
                return self._trace_switch_mlp
            if (
                code is self.activation_code
                and module is self.target_activation
                and self.active_target
            ):
                return self._trace_activation
            if (
                code is self.linear_code
                and self.active_target
                and (
                    module is self.target_moe.gate
                    or module is self.target_moe.shared_expert_gate
                )
            ):
                return self._trace_linear
        return None

    def _array_sum(self, array: Any, *args: Any, **kwargs: Any) -> Any:
        result = self._old_array_sum(array, *args, **kwargs)
        axis = args[0] if args else kwargs.get("axis")
        shape = tuple(int(dimension) for dimension in array.shape)
        if self.active_target and len(shape) == 4 and shape[-2:] == (8, self.hidden_size):
            normalized_axis = axis if axis is None or axis >= 0 else len(shape) + axis
            if normalized_axis == 2:
                self.stage_refs["expert_weighted_outputs"] = array
                self.stage_refs["routed_sum_official"] = result
        return result

    def _sigmoid(self, value: Any, *args: Any, **kwargs: Any) -> Any:
        result = self._old_sigmoid(value, *args, **kwargs)
        # SparseMoeBlock also invokes sigmoid inside its shared SwiGLU. Identify
        # the shared scalar gate by the exact Linear output, not by call order.
        if self.active_target and value is self.stage_refs.get("shared_gate_logits"):
            self.stage_refs["shared_gate_sigmoid"] = result
        return result

    def install(self) -> None:
        capture = self

        def traced_array_sum(array: Any, *args: Any, **kwargs: Any) -> Any:
            return capture._array_sum(array, *args, **kwargs)

        self.mx.array.sum = traced_array_sum
        self.mx.sigmoid = self._sigmoid
        sys.settrace(self.trace)

    def uninstall(self) -> None:
        sys.settrace(self._old_trace)
        self.mx.array.sum = self._old_array_sum
        self.mx.sigmoid = self._old_sigmoid
        self.active_target = False

    def add_diagnostic_prefixes(self) -> None:
        weighted = self.stage_refs.get("expert_weighted_outputs")
        if weighted is None:
            raise RuntimeError("official weighted expert-sum operand was not captured")
        prefixes = [weighted[0, 0, 0, :]]
        for rank in range(1, 8):
            prefixes.append(prefixes[-1] + weighted[0, 0, rank, :])
        self.prefix_ref = self.mx.stack(prefixes, axis=0)


def source_hashes(
    mlx_root: pathlib.Path,
    model_dir: pathlib.Path,
    config_path: pathlib.Path,
    index_path: pathlib.Path,
    input_bundle_hash: str,
    moe_hash: dict[str, Any],
) -> dict[str, str]:
    return {
        "config.json": sha256_file(config_path),
        "model.safetensors.index.json": sha256_file(index_path),
        "mlx_audio/lm/models/qwen3_5.py": sha256_file(
            mlx_root / "mlx_audio/lm/models/qwen3_5.py"
        ),
        "mlx_audio/lm/models/switch_layers.py": sha256_file(
            mlx_root / "mlx_audio/lm/models/switch_layers.py"
        ),
        "mlx_audio/utils.py": sha256_file(mlx_root / "mlx_audio/utils.py"),
        "scripts/drama3/replay_mlx_backbone.py": sha256_file(
            pathlib.Path(__file__).with_name("replay_mlx_backbone.py")
        ),
        "scripts/drama3/replay_mlx_moe.py": sha256_file(pathlib.Path(__file__)),
        "input_bundle": input_bundle_hash,
        "target_moe_weights": moe_hash["sha256"],
    }


def stage_record(
    value: Any,
    dtype: str = "float32",
    shape: list[int] | None = None,
    capture_kind: str = "upstream_local",
    expert_ids: list[int] | None = None,
) -> dict[str, Any]:
    np_dtype = np.int32 if dtype == "int32" else np.float32
    array = np.asarray(value, dtype=np_dtype)
    actual_shape = list(array.shape) if shape is None else shape
    if int(np.prod(actual_shape, dtype=np.int64)) != array.size:
        raise ValueError(f"stage shape {actual_shape} does not match {array.size} values")
    array = array.reshape(actual_shape)
    if dtype == "float32" and not np.isfinite(array).all():
        raise FloatingPointError("captured MoE stage contains non-finite values")
    record: dict[str, Any] = {
        "shape": actual_shape,
        "dtype": dtype,
        "values": array.reshape(-1).tolist(),
        "capture_kind": capture_kind,
    }
    if expert_ids is not None:
        record["expert_ids"] = expert_ids
    return record


def check_capture_control(
    refs: dict[str, Any],
    control_path: pathlib.Path,
    input_bundle_hash: str,
    mx: Any,
) -> dict[str, Any]:
    control_raw, control = read_json_bytes(control_path)
    if control.get("format") != REFERENCE_TRACE_FORMAT:
        raise ValueError(f"unexpected control trace format: {control.get('format')!r}")
    manifest_path = control_path.with_suffix(control_path.suffix + ".manifest.json")
    manifest_raw, manifest = read_json_bytes(manifest_path)
    if manifest.get("trace_sha256") != sha256_bytes(control_raw):
        raise ValueError("control trace does not match its sidecar manifest hash")
    if manifest.get("inputs", {}).get("sha256") != input_bundle_hash:
        raise ValueError("control trace was produced from a different input bundle")
    if manifest.get("model") != {
        "repository": MODEL_REPOSITORY,
        "revision": MODEL_REVISION,
    }:
        raise ValueError("control trace model identity does not match the pinned checkpoint")
    if manifest.get("upstream", {}).get("mlx-audio") != {
        "repository": UPSTREAM_REPOSITORY,
        "revision": MLX_REVISION,
    }:
        raise ValueError("control trace upstream identity does not match the pinned source")
    expected = {item["name"]: item for item in control.get("traces", [])}
    if len(expected) != len(control.get("traces", [])):
        raise ValueError("control trace contains duplicate names")
    if set(expected) != set(refs):
        missing = sorted(set(expected) - set(refs))
        extra = sorted(set(refs) - set(expected))
        raise ValueError(f"capture/control trace names differ; missing={missing[:5]} extra={extra[:5]}")

    mismatches: list[str] = []
    for name, value in refs.items():
        current = np.asarray(value, dtype=np.float32).reshape(-1)
        reference = np.asarray(expected[name]["values"], dtype=np.float32).reshape(-1)
        if current.size != reference.size or current.tobytes() != reference.tobytes():
            mismatches.append(name)
    return {
        "passed": not mismatches,
        "comparison": "float32 byte equality",
        "reference_path": control_path.name,
        "reference_sha256": sha256_bytes(control_raw),
        "reference_manifest_sha256": sha256_bytes(manifest_raw),
        "input_bundle_sha256": input_bundle_hash,
        "expected_trace_count": len(expected),
        "observed_trace_count": len(refs),
        "mismatches": mismatches,
    }


def load_model(
    mlx_root: pathlib.Path,
    model_dir: pathlib.Path,
) -> tuple[Any, Any, Any, Any, pathlib.Path, pathlib.Path, dict[str, str], dict[str, Any]]:
    import mlx.core as mx
    import mlx.nn as nn
    from safetensors import __version__ as safetensors_version

    sys.path.insert(0, str(mlx_root))
    from mlx_audio.lm.models import qwen3_5
    from mlx_audio.lm.models.qwen3_5 import Qwen3_5TextModel, TextModelArgs
    from mlx_audio.utils import apply_quantization

    mx.set_default_device(mx.gpu)
    if mx.default_device() != mx.gpu:
        raise RuntimeError(f"failed to select MLX GPU device: {mx.default_device()}")

    actual_revision = subprocess.check_output(
        ["git", "-C", str(mlx_root), "rev-parse", "HEAD"], text=True
    ).strip()
    if actual_revision != MLX_REVISION:
        raise RuntimeError(f"MLX-audio source is {actual_revision}; expected {MLX_REVISION}")

    config_path = model_dir / "config.json"
    index_path = model_dir / "model.safetensors.index.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    index = json.loads(index_path.read_text(encoding="utf-8"))
    selected = selected_backbone_map(index["weight_map"])
    args = TextModelArgs.from_dict(config["text_config"])
    if args.num_hidden_layers <= TARGET_LAYER:
        raise ValueError(f"model has only {args.num_hidden_layers} layers")
    if args.num_experts_per_tok != 8:
        raise ValueError(f"expected top-8 routing; found top-{args.num_experts_per_tok}")
    expected_config = {
        "hidden_size": 2048,
        "moe_intermediate_size": 512,
        "num_experts": 256,
        "num_experts_per_tok": 8,
    }
    actual_config = {name: int(getattr(args, name)) for name in expected_config}
    if actual_config != expected_config:
        raise ValueError(f"pinned model MoE dimensions differ: {actual_config}")
    if not args.norm_topk_prob:
        raise ValueError("pinned model must normalize the selected top-k probabilities")

    class BackboneOnly(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.backbone = Qwen3_5TextModel(args)

    model = BackboneOnly()
    apply_quantization(model, config, selected)
    load_selected_weights(model, model_dir, selected, mx)
    model.eval()
    moe_hash = target_moe_weight_hash(model_dir, selected)
    runtime = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "mlx": importlib.metadata.version("mlx"),
        "device": str(mx.default_device()),
        "numpy": np.__version__,
        "torch": torch.__version__,
        "safetensors": safetensors_version,
    }
    return mx, nn, qwen3_5, model, config_path, index_path, selected, {
        "args": args,
        "runtime": runtime,
        "moe_hash": moe_hash,
        "config": config,
        "model_config": actual_config,
    }


def validate_input_fixture(
    fixture: dict[str, Any], hidden_size: int, source_hash_map: dict[str, str]
) -> np.ndarray:
    if fixture.get("format") != INPUT_FORMAT:
        raise ValueError(f"expected {INPUT_FORMAT}, found {fixture.get('format')!r}")
    if fixture.get("coordinate") != {"row": TARGET_ROW, "layer": TARGET_LAYER}:
        raise ValueError("MoE fixture coordinate is not row 1/layer 39")
    if fixture.get("activation_name") != TARGET_ACTIVATION_NAME:
        raise ValueError("MoE fixture activation is not post_attention_layernorm")
    if fixture.get("activation_dtype") != "float32":
        raise ValueError("MoE fixture activation must be float32")
    if fixture.get("activation_shape") != [hidden_size]:
        raise ValueError(f"MoE fixture activation shape must be [{hidden_size}]")
    if fixture.get("model") != {"repository": MODEL_REPOSITORY, "revision": MODEL_REVISION}:
        raise ValueError("MoE fixture model identity does not match the pinned checkpoint")
    if fixture.get("upstream") != {"repository": UPSTREAM_REPOSITORY, "revision": MLX_REVISION}:
        raise ValueError("MoE fixture upstream identity does not match the pinned source")
    hashes = fixture.get("source_hashes")
    if not isinstance(hashes, dict):
        raise ValueError("MoE fixture is missing source_hashes")
    for name, expected in source_hash_map.items():
        if hashes.get(name) != expected:
            raise ValueError(f"MoE fixture source hash mismatch for {name}")
    values = np.asarray(fixture.get("activation"), dtype=np.float32)
    if values.shape != (hidden_size,) or not np.isfinite(values).all():
        raise ValueError("MoE fixture activation has invalid shape or non-finite values")
    if fixture.get("activation_sha256") != activation_sha256(values):
        raise ValueError("MoE fixture activation SHA-256 does not match its values")
    return values


def build_stages(capture: UpstreamCapture, hidden_size: int) -> tuple[dict[str, Any], list[int]]:
    refs = capture.stage_refs
    required = (
        "normalized_input",
        "router_logits",
        "router_softmax",
        "expert_ids_argpartition_order",
        "selected_probabilities",
        "selected_weights",
        "expert_gate_outputs",
        "expert_up_outputs",
        "expert_activation_outputs",
        "expert_down_outputs",
        "expert_weighted_outputs",
        "routed_sum_official",
        "shared_gate_logits",
        "shared_gate_sigmoid",
        "shared_expert_output",
        "shared_weighted_output",
        "moe_output_official",
    )
    missing = [name for name in required if name not in refs]
    if missing:
        raise RuntimeError(f"upstream MoE stages were not captured: {missing}")

    expert_ids = np.asarray(refs["expert_ids_argpartition_order"], dtype=np.int32).reshape(-1)
    if expert_ids.size != 8:
        raise ValueError(f"expected 8 raw expert IDs, found {expert_ids.size}")
    id_list = [int(value) for value in expert_ids]
    expert_count = int(np.asarray(refs["router_logits"]).size)
    expert_gate_values = np.asarray(refs["expert_gate_outputs"], dtype=np.float32).reshape(8, -1)
    expert_up_values = np.asarray(refs["expert_up_outputs"], dtype=np.float32).reshape(8, -1)
    expert_activation_values = np.asarray(
        refs["expert_activation_outputs"], dtype=np.float32
    ).reshape(8, -1)
    expert_values = np.asarray(refs["expert_down_outputs"], dtype=np.float32).reshape(8, hidden_size)
    weighted_values = np.asarray(refs["expert_weighted_outputs"], dtype=np.float32).reshape(8, hidden_size)

    stages = {
        "normalized_input": stage_record(
            refs["normalized_input"], shape=[hidden_size], capture_kind="upstream_sparse_argument"
        ),
        "router_logits": stage_record(
            refs["router_logits"], shape=[expert_count], capture_kind="upstream_linear_return"
        ),
        "router_softmax": stage_record(
            refs["router_softmax"], shape=[expert_count], capture_kind="upstream_local"
        ),
        "expert_ids_argpartition_order": stage_record(
            expert_ids, dtype="int32", shape=[8], capture_kind="upstream_local"
        ),
        "selected_probabilities": stage_record(
            refs["selected_probabilities"], shape=[8], capture_kind="upstream_local"
        ),
        "selected_weights": stage_record(
            refs["selected_weights"], shape=[8], capture_kind="upstream_local"
        ),
        "expert_gate_outputs": stage_record(
            expert_gate_values,
            shape=list(expert_gate_values.shape),
            capture_kind="upstream_switch_mlp_gate_projection_return",
            expert_ids=id_list,
        ),
        "expert_up_outputs": stage_record(
            expert_up_values,
            shape=list(expert_up_values.shape),
            capture_kind="upstream_switch_mlp_up_projection_return",
            expert_ids=id_list,
        ),
        "expert_activation_outputs": stage_record(
            expert_activation_values,
            shape=list(expert_activation_values.shape),
            capture_kind="upstream_swiglu_return",
            expert_ids=id_list,
        ),
        "expert_down_outputs": stage_record(
            expert_values,
            shape=[8, hidden_size],
            capture_kind="upstream_switch_mlp_return",
            expert_ids=id_list,
        ),
        "expert_weighted_outputs": stage_record(
            weighted_values,
            shape=[8, hidden_size],
            capture_kind="upstream_sum_operand",
            expert_ids=id_list,
        ),
        "routed_sum_official": stage_record(
            refs["routed_sum_official"], shape=[hidden_size], capture_kind="upstream_sum_return"
        ),
        "shared_gate_logits": stage_record(
            refs["shared_gate_logits"], shape=[1], capture_kind="upstream_linear_return"
        ),
        "shared_gate_sigmoid": stage_record(
            refs["shared_gate_sigmoid"], shape=[1], capture_kind="upstream_sigmoid_return"
        ),
        "shared_expert_output": stage_record(
            refs["shared_expert_output"], shape=[hidden_size], capture_kind="upstream_local"
        ),
        "shared_weighted_output": stage_record(
            refs["shared_weighted_output"],
            shape=[hidden_size],
            capture_kind="upstream_return_local",
        ),
        "moe_output_official": stage_record(
            refs["moe_output_official"], shape=[hidden_size], capture_kind="upstream_return"
        ),
    }
    if capture.prefix_ref is None:
        raise RuntimeError("diagnostic prefix sums were not prepared")
    stages["diagnostic_prefix_sums_argpartition_order"] = stage_record(
        capture.prefix_ref,
        shape=[8, hidden_size],
        capture_kind="diagnostic_recompute_sequential_add",
        expert_ids=id_list,
    )
    return stages, id_list


def run(
    mlx_root: pathlib.Path,
    model_dir: pathlib.Path,
    output_path: pathlib.Path,
    inputs_path: pathlib.Path | None,
    control_path: pathlib.Path | None,
    input_fixture_path: pathlib.Path | None,
    input_fixture_out: pathlib.Path | None,
) -> tuple[dict[str, Any], bool]:
    if (inputs_path is None) == (input_fixture_path is None):
        raise ValueError("select exactly one of --inputs or --input-fixture")
    full_run = inputs_path is not None
    if full_run and (control_path is None or input_fixture_out is None):
        raise ValueError("full-run mode requires --control-trace and --input-fixture-out")
    if not full_run and (control_path is not None or input_fixture_out is not None):
        raise ValueError("--control-trace and --input-fixture-out are full-run-only")

    mlx_root = mlx_root.resolve()
    model_dir = model_dir.resolve()
    output_path = output_path.resolve()
    if inputs_path is not None:
        inputs_path = inputs_path.resolve()
    if control_path is not None:
        control_path = control_path.resolve()
    if input_fixture_path is not None:
        input_fixture_path = input_fixture_path.resolve()
    if input_fixture_out is not None:
        input_fixture_out = input_fixture_out.resolve()

    mx, nn, qwen3_5, model, config_path, index_path, selected, metadata = load_model(
        mlx_root, model_dir
    )
    args = metadata["args"]
    hashes = source_hashes(
        mlx_root,
        model_dir,
        config_path,
        index_path,
        sha256_file(inputs_path) if inputs_path is not None else "",
        metadata["moe_hash"],
    )
    fixture_bytes: bytes | None = None
    fixture: dict[str, Any] | None = None
    if input_fixture_path is not None:
        fixture_bytes, fixture = read_json_bytes(input_fixture_path)
        fixture_source_checks = {
            name: hashes[name]
            for name in (
                "config.json",
                "model.safetensors.index.json",
                "mlx_audio/lm/models/qwen3_5.py",
                "mlx_audio/lm/models/switch_layers.py",
                "mlx_audio/utils.py",
                "scripts/drama3/replay_mlx_backbone.py",
                "scripts/drama3/replay_mlx_moe.py",
                "target_moe_weights",
            )
        }
        activation = validate_input_fixture(fixture, args.hidden_size, fixture_source_checks)
        hashes["input_bundle"] = fixture["source_hashes"]["input_bundle"]
        input_raw_sha = sha256_bytes(fixture_bytes)
    else:
        input_raw, input_bundle = read_json_bytes(inputs_path)
        if sha256_bytes(input_raw) != hashes["input_bundle"]:
            raise AssertionError("input bundle hash changed while loading")
        embeddings = np.asarray(input_bundle["input_embeddings"], dtype=np.float32)
        if embeddings.ndim != 2 or embeddings.shape[1] != args.hidden_size:
            raise ValueError(
                f"input_embeddings must have shape [rows, {args.hidden_size}], got {embeddings.shape}"
            )
        if embeddings.shape[0] <= TARGET_ROW or not np.isfinite(embeddings).all():
            raise ValueError("input bundle must contain finite row 1 embeddings")
        activation = None
        input_raw_sha = sha256_bytes(input_raw)

    capture = UpstreamCapture(mx, qwen3_5, nn, model.backbone.layers, args.hidden_size)
    final_outputs: list[Any] = []
    try:
        capture.install()
        if full_run:
            cache = model.backbone.make_cache()
            for row, embedding in enumerate(embeddings):
                capture.current_row = row
                result = model.backbone(
                    inputs=mx.zeros((1, 1), dtype=mx.int32),
                    cache=cache,
                    input_embeddings=mx.array(embedding[None, None, :]),
                )
                final_outputs.append(result)
        else:
            capture.current_row = TARGET_ROW
            model.backbone.layers[TARGET_LAYER].mlp(
                mx.array(activation[None, None, :])
            )
    finally:
        capture.uninstall()

    if capture.target_call_count != 1:
        raise RuntimeError(f"expected one target MoE call, captured {capture.target_call_count}")

    if full_run:
        for row, output in enumerate(final_outputs):
            capture.control_refs[f"backbone.row{row}.hidden"] = output

    capture.add_diagnostic_prefixes()
    eval_values = list(capture.stage_refs.values()) + list(capture.control_refs.values())
    if capture.prefix_ref is not None:
        eval_values.append(capture.prefix_ref)
    mx.eval(*eval_values)

    if full_run:
        control_status = check_capture_control(
            capture.control_refs, control_path, hashes["input_bundle"], mx
        )
        if control_status["observed_trace_count"] != len(embeddings) * (
            2 * args.num_hidden_layers + 1
        ):
            raise RuntimeError("full-run capture did not retain the expected 243 control arrays")
        attention_residual = capture.control_refs.get(
            f"backbone.row{TARGET_ROW}.layers.{TARGET_LAYER}.attention_residual"
        )
        hidden_output = capture.control_refs.get(
            f"backbone.row{TARGET_ROW}.layers.{TARGET_LAYER}.hidden"
        )
        backbone_output = capture.control_refs.get(f"backbone.row{TARGET_ROW}.hidden")
        if attention_residual is None or hidden_output is None or backbone_output is None:
            raise RuntimeError("full-run capture omitted target residual or hidden output")
        normalized_values = np.asarray(capture.stage_refs["normalized_input"], dtype=np.float32).reshape(
            args.hidden_size
        )
        fixture_bundle = {
            "format": INPUT_FORMAT,
            "coordinate": {"row": TARGET_ROW, "layer": TARGET_LAYER},
            "activation_name": TARGET_ACTIVATION_NAME,
            "activation_shape": [args.hidden_size],
            "activation_dtype": "float32",
            "activation": normalized_values.tolist(),
            "activation_sha256": activation_sha256(normalized_values),
            "model": {"repository": MODEL_REPOSITORY, "revision": MODEL_REVISION},
            "upstream": {"repository": UPSTREAM_REPOSITORY, "revision": MLX_REVISION},
            "source_hashes": hashes,
        }
        input_fixture_out.parent.mkdir(parents=True, exist_ok=True)
        fixture_serialized = json.dumps(
            fixture_bundle, separators=(",", ":"), allow_nan=False
        ).encode("utf-8") + b"\n"
        input_fixture_out.write_bytes(fixture_serialized)
        fixture_raw_sha = sha256_bytes(fixture_serialized)
    else:
        control_status = None
        fixture_bundle = fixture
        fixture_raw_sha = input_raw_sha

    stages, expert_ids = build_stages(capture, args.hidden_size)
    if full_run:
        stages["attention_residual"] = stage_record(
            attention_residual,
            shape=[args.hidden_size],
            capture_kind="upstream_decoder_local",
        )
        stages["hidden_output"] = stage_record(
            hidden_output,
            shape=[args.hidden_size],
            capture_kind="upstream_target_decoder_output",
        )
        stages["backbone_output"] = stage_record(
            backbone_output,
            shape=[args.hidden_size],
            capture_kind="upstream_final_norm_output",
        )

    source_hash_map = fixture_bundle["source_hashes"]
    bundle = {
        "format": TRACE_FORMAT,
        "mode": "full_run" if full_run else "isolated_replay",
        "coordinate": {"row": TARGET_ROW, "layer": TARGET_LAYER},
        "model": {"repository": MODEL_REPOSITORY, "revision": MODEL_REVISION},
        "upstream": {"repository": UPSTREAM_REPOSITORY, "revision": MLX_REVISION},
        "runtime": metadata["runtime"],
        "model_config": metadata["model_config"],
        "source_hashes": source_hash_map,
        "input": {
            "format": INPUT_FORMAT,
            "path": input_fixture_out.name if full_run else input_fixture_path.name,
            "sha256": fixture_raw_sha,
            "activation_sha256": fixture_bundle["activation_sha256"],
            "shape": [args.hidden_size],
            "dtype": "float32",
            "activation_name": TARGET_ACTIVATION_NAME,
        },
        "expert_rank_to_id": [
            {"rank": rank, "expert_id": expert_id}
            for rank, expert_id in enumerate(expert_ids)
        ],
        "capture_control": control_status,
        "stages": stages,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(bundle, separators=(",", ":"), allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return bundle, bool(control_status is None or control_status["passed"])


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mlx-root", type=pathlib.Path, required=True)
    parser.add_argument("--model-dir", type=pathlib.Path, required=True)
    parser.add_argument("--inputs", type=pathlib.Path, help="full-run input embedding bundle")
    parser.add_argument("--input-fixture", type=pathlib.Path, help="isolated normalized-input fixture")
    parser.add_argument("--control-trace", type=pathlib.Path, help="no-capture 243-trace baseline")
    parser.add_argument("--input-fixture-out", type=pathlib.Path, help="write captured activation fixture")
    parser.add_argument("--output", type=pathlib.Path, required=True)
    args = parser.parse_args()
    bundle, passed = run(
        args.mlx_root,
        args.model_dir,
        args.output,
        args.inputs,
        args.control_trace,
        args.input_fixture,
        args.input_fixture_out,
    )
    print(
        json.dumps(
            {
                "output": str(args.output.resolve()),
                "mode": bundle["mode"],
                "capture_control_passed": passed,
                "input_fixture": bundle["input"]["path"],
            }
        )
    )
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
