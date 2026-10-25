#!/usr/bin/env python3
"""Capture one official pretrained Qwen2.5-VL decoder block for localization.

The input hidden state and expected output are authenticated snapshots from a
trusted 26-layer prefix fixture. Only the selected BF16 decoder layer is
loaded. The diagnostic stores stage activations and their provenance, never
weights.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import platform
import sys
from typing import Any, Callable


SCHEMA = "cogni_ml.qwen25_vl_pretrained_decoder_block_stage_reference"
SCHEMA_VERSION = 1
MODEL_ID = "Qwen/Qwen2.5-VL-7B-Instruct"
MODEL_REVISION = "eb6ba55b40ada64977147b52e50a4145ee89ac4f"
TRUSTED_PREFIX_SHA256 = "40c665e0ce5903f43411d39a46379e8dda64ddbe41bd058f15552c877ccba96e"
TRUSTED_PREFIX_S1280_SHA256 = "b78616cc1dc3b1deb1215b06ecc49fbf60e3b38477f24aec2a0ccd60da745d3a"
TRUSTED_PREFIX_SHA256S = frozenset((TRUSTED_PREFIX_SHA256, TRUSTED_PREFIX_S1280_SHA256))
MAX_PREFIX_MANIFEST_BYTES = 8 * 1024 * 1024
MAX_DIAGNOSTIC_SIDECAR_BYTES = 512 * 1024 * 1024
MAX_BLOCK_MPS_ESTIMATE_BYTES = 3 * 1024 * 1024 * 1024
MPS_RESERVE_FRACTION = 0.15

STAGE_PREFIX_NAMES = (
    "rotary_cos",
    "rotary_sin",
)
RMS_TRACE_STAGE_SUFFIXES = (
    "variance_f32",
    "variance_plus_eps_f32",
    "rsqrt_f32",
    "normalized_f32",
    "normalized_bf16",
)
RMS_SITE_MODULES = {
    "input": "input_layernorm",
    "post-attention": "post_attention_layernorm",
}
RMS_SOURCE_FILE = "models/qwen2/modeling_qwen2.py"
RMS_SOURCE_SHA256 = "08742a7c82a8713bc22f632f01c97b7eca3ef0e947180e7d6c9beec4897c4ff8"


def _load_pretrained_producer() -> Any:
    path = Path(__file__).resolve().with_name("qwen25_pretrained_reference.py")
    spec = importlib.util.spec_from_file_location("qwen25_pretrained_reference_for_block", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("could not load the pinned pretrained prefix producer")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _validate_layer_index(layer_index: Any, prefix_layer_count: int = 26) -> int:
    if (
        isinstance(layer_index, bool)
        or not isinstance(layer_index, int)
        or isinstance(prefix_layer_count, bool)
        or not isinstance(prefix_layer_count, int)
        or prefix_layer_count <= 0
        or not 0 <= layer_index < prefix_layer_count
    ):
        raise ValueError(f"layer index must be an integer in 0..{prefix_layer_count - 1}")
    return layer_index


def _read_prefix_snapshot(
    manifest_path: Path,
    expected_sha256: str,
    validate_manifest: Callable[[Any], None],
) -> dict[str, Any]:
    """Read and authenticate exact manifest and sidecar snapshots once."""
    if not _is_sha256(expected_sha256):
        raise ValueError("trusted prefix manifest SHA-256 is malformed")
    if expected_sha256 not in TRUSTED_PREFIX_SHA256S:
        raise ValueError("trusted prefix manifest digest does not match a pinned prefix fixture")
    manifest_path = Path(manifest_path).expanduser()
    if manifest_path.is_symlink():
        raise ValueError("prefix manifest must be a regular non-symlink file")
    manifest_path = manifest_path.resolve(strict=True)
    if not manifest_path.is_file():
        raise ValueError("prefix manifest must be a regular non-symlink file")
    manifest_bytes = manifest_path.read_bytes()
    if not 0 < len(manifest_bytes) <= MAX_PREFIX_MANIFEST_BYTES:
        raise ValueError("prefix manifest size is outside the supported range")
    actual_manifest_sha256 = _sha256_bytes(manifest_bytes)
    if actual_manifest_sha256 != expected_sha256:
        raise ValueError("prefix manifest SHA-256 does not match its trusted digest")
    try:
        document = json.loads(manifest_bytes)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("trusted prefix manifest is not valid JSON") from error
    validate_manifest(document)

    sidecar = document["sidecars"]["bf16"]
    name = sidecar["path"]
    if (
        not isinstance(name, str)
        or not name
        or name in {".", ".."}
        or "/" in name
        or "\\" in name
        or "\x00" in name
    ):
        raise ValueError("prefix sidecar path must be a sibling basename")
    if sidecar.get("encoding") != "bfloat16-le":
        raise ValueError("prefix sidecar encoding is not BF16 little-endian")
    sidecar_size = sidecar.get("nbytes")
    sidecar_sha256 = sidecar.get("sha256")
    if (
        isinstance(sidecar_size, bool)
        or not isinstance(sidecar_size, int)
        or sidecar_size <= 0
        or sidecar_size > MAX_DIAGNOSTIC_SIDECAR_BYTES
        or sidecar_size % 2
        or not _is_sha256(sidecar_sha256)
    ):
        raise ValueError("prefix sidecar size or digest is invalid")
    root = manifest_path.parent.resolve(strict=True)
    sidecar_path = root / name
    if (
        sidecar_path.is_symlink()
        or not sidecar_path.is_file()
        or sidecar_path.resolve(strict=True).parent != root
    ):
        raise ValueError("prefix BF16 sidecar must be a regular sibling file")
    sidecar_bytes = sidecar_path.read_bytes()
    if len(sidecar_bytes) != sidecar_size:
        raise ValueError("prefix sidecar size does not match its manifest")
    actual_sidecar_sha256 = _sha256_bytes(sidecar_bytes)
    if actual_sidecar_sha256 != sidecar_sha256:
        raise ValueError("prefix sidecar SHA-256 does not match its manifest")
    return {
        "manifest_path": str(manifest_path),
        "manifest_sha256": actual_manifest_sha256,
        "manifest_bytes": manifest_bytes,
        "document": document,
        "sidecar_path": str(sidecar_path),
        "sidecar_bytes": sidecar_bytes,
        "sidecar_sha256": actual_sidecar_sha256,
    }


def _prefix_state_bytes(
    snapshot: dict[str, Any], layer_index: int, producer: Any
) -> tuple[bytes, bytes, list[int], dict[str, Any]]:
    layer_index = _validate_layer_index(layer_index, producer.PREFIX_LAYER_COUNT)
    document = snapshot["document"]
    inputs = document["inputs"]
    capture = document["capture"]
    shape = capture["hidden_states"]["shape"]
    sequence_length = len(inputs["token_ids"])
    expected_shape = [producer.PREFIX_LAYER_COUNT + 1, 1, sequence_length, producer.REAL_TEXT_GEOMETRY["hidden_size"]]
    if shape != expected_shape:
        raise ValueError("trusted prefix state tensor has an unexpected shape")
    state_nbytes = math.prod(shape[1:]) * 2
    if len(snapshot["sidecar_bytes"]) != math.prod(shape) * 2:
        raise ValueError("trusted prefix sidecar does not contain the declared states")
    input_start = layer_index * state_nbytes
    output_start = (layer_index + 1) * state_nbytes
    input_state = snapshot["sidecar_bytes"][input_start : input_start + state_nbytes]
    output_state = snapshot["sidecar_bytes"][output_start : output_start + state_nbytes]
    if len(input_state) != state_nbytes or len(output_state) != state_nbytes:
        raise ValueError("trusted prefix is missing the selected input or output state")
    request = producer._validate_inputs(
        inputs["token_ids"],
        inputs["attention_mask"],
        producer.REAL_TEXT_GEOMETRY["vocab_size"],
        inputs["position_offset"],
        producer.REAL_TEXT_GEOMETRY["hidden_size"],
    )
    return input_state, output_state, expected_shape, request


def _diagnostic_inputs(request: dict[str, Any], prefix_inputs: dict[str, Any]) -> dict[str, Any]:
    """Carry the authenticated prefix's exact rotary inputs into the block record."""
    request_keys = (
        "token_ids",
        "attention_mask",
        "position_ids_f32",
        "position_ids_for_official_model",
        "multimodal_position_shape",
        "position_offset",
    )
    for key in request_keys:
        if prefix_inputs.get(key) != request.get(key):
            raise ValueError(f"authenticated prefix input disagrees with the block request: {key}")
    inv_freq = prefix_inputs.get("inv_freq_f32")
    if (
        not isinstance(inv_freq, list)
        or len(inv_freq) != 64
        or any(isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) for value in inv_freq)
    ):
        raise ValueError("authenticated prefix inverse frequencies are malformed")
    return {key: request[key] for key in request_keys} | {
        "inv_freq_f32": [float(value) for value in inv_freq]
    }


def _verify_config_generated_inv_freq(prefix_inv_freq: Any, official_inv_freq: Any) -> None:
    if (
        not isinstance(prefix_inv_freq, list)
        or not isinstance(official_inv_freq, list)
        or len(prefix_inv_freq) != 64
        or len(official_inv_freq) != 64
    ):
        raise ValueError("official rotary inverse-frequency geometry is invalid")
    for index, (prefix_value, official_value) in enumerate(zip(prefix_inv_freq, official_inv_freq)):
        if (
            isinstance(prefix_value, bool)
            or not isinstance(prefix_value, (int, float))
            or isinstance(official_value, bool)
            or not isinstance(official_value, (int, float))
            or not math.isfinite(prefix_value)
            or not math.isfinite(official_value)
            or float(prefix_value) != float(official_value)
        ):
            raise ValueError(f"official config-generated inverse frequency differs from prefix input at {index}")


def _safe_checkpoint_paths(root: Path, filenames: set[str]) -> None:
    for filename in filenames:
        if (
            not isinstance(filename, str)
            or not filename
            or filename in {".", ".."}
            or "/" in filename
            or "\\" in filename
            or "\x00" in filename
        ):
            raise ValueError("checkpoint filenames must be sibling basenames")
        path = root / filename
        if path.is_symlink() or path.resolve(strict=True).parent != root:
            raise ValueError(f"checkpoint file must be a regular file under its root: {filename}")


def _load_selected_checkpoint(
    checkpoint_root: Path, receipt_path: Path, layer_index: int, producer: Any
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    layer_index = _validate_layer_index(layer_index, producer.PREFIX_LAYER_COUNT)
    root = checkpoint_root.resolve(strict=True)
    if not root.is_dir():
        raise ValueError("checkpoint root is not a directory")
    receipt_path = Path(receipt_path).expanduser()
    if receipt_path.is_symlink():
        raise ValueError("checkpoint receipt must be a regular non-symlink file")
    receipt_path = receipt_path.resolve(strict=True)
    receipt_bytes = receipt_path.read_bytes()
    parsed_receipt_sha256 = _sha256_bytes(receipt_bytes)
    receipt_data = json.loads(receipt_bytes)
    receipt_files = producer._validate_receipt_metadata(receipt_data)
    if not isinstance(receipt_data, dict):
        raise ValueError("checkpoint receipt is malformed")

    metadata_names = {"config.json", "model.safetensors.index.json"}
    _safe_checkpoint_paths(root, metadata_names)
    metadata_source = producer._verify_receipt_files(
        root, receipt_path, receipt_data, metadata_names
    )
    if metadata_source["receipt_sha256"] != parsed_receipt_sha256:
        raise ValueError("checkpoint receipt changed during verification")
    config_bytes = (root / "config.json").read_bytes()
    index_bytes = (root / "model.safetensors.index.json").read_bytes()
    if _sha256_bytes(config_bytes) != metadata_source["verified_files"]["config.json"]["sha256"]:
        raise ValueError("checkpoint config changed after receipt verification")
    if _sha256_bytes(index_bytes) != metadata_source["verified_files"]["model.safetensors.index.json"]["sha256"]:
        raise ValueError("checkpoint index changed after receipt verification")
    model_config = json.loads(config_bytes)
    text_values = producer._text_config(model_config)
    index = json.loads(index_bytes)
    weight_map = index.get("weight_map") if isinstance(index, dict) else None
    if not isinstance(weight_map, dict):
        raise ValueError("checkpoint safetensors index has no weight_map")

    layer_prefix = f"model.layers.{layer_index}."
    tensor_names = (
        "input_layernorm.weight",
        "self_attn.q_proj.weight",
        "self_attn.q_proj.bias",
        "self_attn.k_proj.weight",
        "self_attn.k_proj.bias",
        "self_attn.v_proj.weight",
        "self_attn.v_proj.bias",
        "self_attn.o_proj.weight",
        "post_attention_layernorm.weight",
        "mlp.gate_proj.weight",
        "mlp.up_proj.weight",
        "mlp.down_proj.weight",
    )
    shard_names: set[str] = set()
    for tensor_name in tensor_names:
        key = layer_prefix + tensor_name
        shard = weight_map.get(key)
        if not isinstance(shard, str) or shard not in receipt_files:
            raise ValueError(f"checkpoint index/receipt is missing selected tensor {key}")
        shard_names.add(shard)
    if not shard_names or any(not shard.endswith(".safetensors") for shard in shard_names):
        raise ValueError("selected checkpoint layer does not reference safetensors shards")
    _safe_checkpoint_paths(root, shard_names)
    selected_source = producer._verify_receipt_files(
        root, receipt_path, receipt_data, shard_names
    )
    if selected_source["receipt_sha256"] != parsed_receipt_sha256:
        raise ValueError("checkpoint receipt changed during selected-shard verification")
    verified_files = dict(metadata_source["verified_files"])
    verified_files.update(selected_source["verified_files"])
    checkpoint = {
        "repo_id": producer.MODEL_ID,
        "revision": producer.MODEL_REVISION,
        "root": str(root),
        "receipt_path": str(receipt_path),
        "receipt_sha256": metadata_source["receipt_sha256"],
        "config_sha256": verified_files["config.json"]["sha256"],
        "index_sha256": verified_files["model.safetensors.index.json"]["sha256"],
        "verified_files": verified_files,
        "index_weight_map": weight_map,
        "text_config": text_values,
    }
    return model_config, index, checkpoint


def _stage_shapes(layer_index: int, sequence_length: int, geometry: dict[str, Any], helper: Any) -> dict[str, list[int]]:
    hidden = geometry["hidden_size"]
    intermediate = geometry["intermediate_size"]
    heads = geometry["num_attention_heads"]
    kv_heads = geometry["num_key_value_heads"]
    head_dim = geometry["head_dim"]
    common = [1, sequence_length, hidden]
    stage_shapes = {
        "rotary_cos": [sequence_length, head_dim],
        "rotary_sin": [sequence_length, head_dim],
        f"layers.{layer_index}.input_layernorm": common,
        f"layers.{layer_index}.self_attn.q_proj": common,
        f"layers.{layer_index}.self_attn.k_proj": [1, sequence_length, kv_heads * head_dim],
        f"layers.{layer_index}.self_attn.v_proj": [1, sequence_length, kv_heads * head_dim],
        "post_rope_q": [1, heads, sequence_length, head_dim],
        "post_rope_k": [1, kv_heads, sequence_length, head_dim],
        "attended": common,
        f"layers.{layer_index}.self_attn.o_proj": common,
        "residual_after_attention": common,
        f"layers.{layer_index}.post_attention_layernorm": common,
        f"layers.{layer_index}.mlp.gate_proj": [1, sequence_length, intermediate],
        f"layers.{layer_index}.mlp.up_proj": [1, sequence_length, intermediate],
        "silu_product": [1, sequence_length, intermediate],
        f"layers.{layer_index}.mlp.down_proj": common,
        "hidden_states_out": common,
    }
    helper_stage_names = {name.format(i=layer_index) for name in helper.STAGE_NAMES}
    if set(stage_shapes).difference(STAGE_PREFIX_NAMES) != helper_stage_names:
        raise RuntimeError("diagnostic stage map no longer matches the pinned capture helper")
    return stage_shapes


def _normalize_rms_site(rms_site: Any) -> str:
    if not isinstance(rms_site, str) or rms_site not in RMS_SITE_MODULES:
        raise ValueError("RMS trace site must be input or post-attention")
    return RMS_SITE_MODULES[rms_site]


def _rms_trace_shapes(
    layer_index: int,
    sequence_length: int,
    hidden_size: int,
    rms_site: str = "input_layernorm",
) -> dict[str, list[int]]:
    if rms_site not in {"input_layernorm", "post_attention_layernorm"}:
        raise ValueError("RMS trace module site is unsupported")
    prefix = f"layers.{layer_index}.{rms_site}."
    return {
        prefix + "variance_f32": [1, sequence_length, 1],
        prefix + "variance_plus_eps_f32": [1, sequence_length, 1],
        prefix + "rsqrt_f32": [1, sequence_length, 1],
        prefix + "normalized_f32": [1, sequence_length, hidden_size],
        prefix + "normalized_bf16": [1, sequence_length, hidden_size],
    }


def _qualified_rms_trace_entries(
    layer_index: int,
    rms_stages: dict[str, Any],
    rms_site: str = "input_layernorm",
) -> list[tuple[str, Any, str]]:
    layer_index = _validate_layer_index(layer_index)
    if rms_site not in {"input_layernorm", "post_attention_layernorm"}:
        raise ValueError("RMS trace module site is unsupported")
    if set(rms_stages) != set(RMS_TRACE_STAGE_SUFFIXES):
        raise RuntimeError("RMS trace tensor inventory differs from its expected stage suffixes")
    return [
        (
            f"layers.{layer_index}.{rms_site}.{name}",
            rms_stages[name],
            "bfloat16" if name.endswith("_bf16") else "float32",
        )
        for name in RMS_TRACE_STAGE_SUFFIXES
    ]


@contextmanager
def _observe_official_rmsnorm(module: Any, torch: Any):
    """Observe actual ops inside one official RMSNorm call without substituting math."""
    from torch.utils._python_dispatch import TorchDispatchMode

    captured: dict[str, Any] = {}
    original_forward = module.forward

    class RMSDispatchObserver(TorchDispatchMode):
        def __torch_dispatch__(self, function: Any, types: Any, args: Any = (), kwargs: Any = None) -> Any:
            result = function(*args, **(kwargs or {}))
            schema = getattr(function, "_schema", None)
            operator = getattr(schema, "name", "")
            if not isinstance(result, torch.Tensor):
                return result
            if operator == "aten::mean":
                name = "variance_f32"
            elif operator == "aten::add" and result.dtype == torch.float32:
                name = "variance_plus_eps_f32"
            elif operator == "aten::rsqrt":
                name = "rsqrt_f32"
            elif operator == "aten::mul" and result.dtype == torch.float32:
                name = "normalized_f32"
            elif operator == "aten::mul" and result.dtype == torch.bfloat16:
                candidates = [
                    value for value in args
                    if isinstance(value, torch.Tensor)
                    and value.dtype == torch.bfloat16
                    and tuple(value.shape) == tuple(result.shape)
                ]
                if len(candidates) != 1:
                    raise RuntimeError("official RMSNorm gain multiply did not expose one normalized BF16 operand")
                name = "normalized_bf16"
                if name in captured:
                    raise RuntimeError(f"official RMSNorm emitted a duplicate diagnostic op: {name}")
                captured[name] = candidates[0]
                return result
            else:
                return result
            if name in captured:
                raise RuntimeError(f"official RMSNorm emitted a duplicate diagnostic op: {name}")
            captured[name] = result
            return result

    def observed_forward(hidden_states: Any) -> Any:
        if "input_bf16" in captured:
            raise RuntimeError("official RMSNorm received more than one diagnostic input")
        captured["input_bf16"] = hidden_states
        with RMSDispatchObserver():
            return original_forward(hidden_states)

    module.forward = observed_forward
    try:
        yield captured
    finally:
        module.forward = original_forward


def _validate_rms_trace(
    captured: dict[str, Any], expected_shapes: dict[str, list[int]], torch: Any, require_mps: bool
) -> None:
    expected_names = {name.rsplit(".", 1)[-1] for name in expected_shapes}
    if set(captured) != expected_names | {"input_bf16"}:
        raise RuntimeError(
            "official RMSNorm did not produce the complete dispatch trace: "
            f"missing={sorted((expected_names | {'input_bf16'}) - set(captured))}, "
            f"unexpected={sorted(set(captured) - (expected_names | {'input_bf16'}))}"
        )
    input_tensor = captured["input_bf16"]
    expected_input_shape = next(
        shape for name, shape in expected_shapes.items() if name.endswith(".normalized_f32")
    )
    if input_tensor.dtype != torch.bfloat16 or list(input_tensor.shape) != expected_input_shape:
        raise RuntimeError("official RMSNorm input does not match the captured BF16 hidden state")
    if require_mps and input_tensor.device.type != "mps":
        raise RuntimeError("official RMSNorm input escaped MPS")
    for qualified_name, shape in expected_shapes.items():
        name = qualified_name.rsplit(".", 1)[-1]
        tensor = captured[name]
        expected_dtype = torch.bfloat16 if name == "normalized_bf16" else torch.float32
        if tensor.dtype != expected_dtype or list(tensor.shape) != shape:
            raise RuntimeError(f"official RMSNorm trace geometry changed: {qualified_name}")
        if require_mps and tensor.device.type != "mps":
            raise RuntimeError(f"official RMSNorm trace escaped MPS: {qualified_name}")


def _validate_rms_input_binding(
    rms_site: str,
    observed_input_bytes: bytes,
    prefix_input_bytes: bytes,
    residual_after_attention_bytes: bytes,
    layer_index: int,
) -> str:
    if rms_site == "input_layernorm":
        expected_input = prefix_input_bytes
        source_description = f"authenticated h{layer_index}"
    elif rms_site == "post_attention_layernorm":
        expected_input = residual_after_attention_bytes
        source_description = "captured residual_after_attention stage"
    else:
        raise ValueError("RMS trace module site is unsupported")
    if observed_input_bytes != expected_input:
        raise RuntimeError(f"official {rms_site} input differs from {source_description}")
    return _sha256_bytes(observed_input_bytes)


def _verified_rms_source_sha256(transformers_sources: dict[str, Any]) -> str:
    source = transformers_sources.get(RMS_SOURCE_FILE)
    if not isinstance(source, dict) or source.get("sha256") != RMS_SOURCE_SHA256:
        raise RuntimeError("pinned Qwen2RMSNorm source record is missing or changed")
    return RMS_SOURCE_SHA256


def _validate_official_rmsnorm_instance(
    module: Any, official_class: Any, transformers_sources: dict[str, Any]
) -> None:
    if type(module) is not official_class:
        raise RuntimeError("selected layer input norm is not the imported official Qwen2RMSNorm class")
    if (
        type(module).__module__ != "transformers.models.qwen2.modeling_qwen2"
        or type(module).__name__ != "Qwen2RMSNorm"
    ):
        raise RuntimeError("selected layer input norm class is not defined by the pinned Qwen2 source module")
    _verified_rms_source_sha256(transformers_sources)


def _estimate_block_mps_bytes(
    expected_weight_shapes: dict[str, tuple[int, ...]],
    stage_shapes: dict[str, list[int]],
    sequence_length: int,
    hidden_size: int,
    intermediate_size: int,
) -> int:
    weights = sum(math.prod(shape) * 2 for shape in expected_weight_shapes.values())
    captures = sum(math.prod(shape) * 2 for shape in stage_shapes.values())
    extra_activation_capture = sequence_length * intermediate_size * 2
    rms_trace_capture = sequence_length * (hidden_size * 6 + 3 * 4)
    hidden_input_and_output = sequence_length * hidden_size * 2 * 2
    workspace = sequence_length * (4 * hidden_size + 4 * intermediate_size) * 2
    fixed_slack = 256 * 1024 * 1024
    return weights + captures + extra_activation_capture + rms_trace_capture + hidden_input_and_output + workspace + fixed_slack


def _require_block_memory_budget(estimated_bytes: int, current_bytes: int, recommended_bytes: int) -> None:
    if estimated_bytes > MAX_BLOCK_MPS_ESTIMATE_BYTES:
        raise MemoryError("one-block diagnostic estimate exceeds the 3 GiB limit")
    if (
        any(isinstance(value, bool) or not isinstance(value, int) or value < 0 for value in (estimated_bytes, current_bytes, recommended_bytes))
        or recommended_bytes <= 0
    ):
        raise ValueError("one-block MPS memory measurements are invalid")
    allowed = int(recommended_bytes * (1.0 - MPS_RESERVE_FRACTION))
    if current_bytes + estimated_bytes > allowed:
        raise MemoryError("one-block diagnostic exceeds the MPS limit with a 15% reserve")


def _state_tensor_from_bytes(state_bytes: bytes, shape: list[int], torch: Any, device: Any) -> Any:
    if len(state_bytes) != math.prod(shape) * 2:
        raise ValueError("authenticated BF16 input state has the wrong byte length")
    words = torch.frombuffer(bytearray(state_bytes), dtype=torch.uint16)
    cpu_state = words.view(torch.bfloat16).reshape(shape)
    state = cpu_state.to(device=device)
    if state.device.type != "mps" or state.dtype != torch.bfloat16 or list(state.shape) != shape:
        raise RuntimeError("authenticated prefix hidden state did not remain BF16 on MPS")
    return state


def _tensor_bf16_bytes(tensor: Any, torch: Any) -> bytes:
    if tensor.device.type != "mps" or tensor.dtype != torch.bfloat16:
        raise RuntimeError("diagnostic activation must remain BF16 on MPS through the terminal wait")
    cpu_words = tensor.detach().contiguous().view(torch.uint16).to(device="cpu")
    return memoryview(cpu_words.numpy()).cast("B").tobytes()


def _tensor_f32_bytes(tensor: Any, torch: Any) -> bytes:
    if tensor.device.type != "mps" or tensor.dtype != torch.float32:
        raise RuntimeError("RMS diagnostic activation must remain F32 on MPS through the terminal wait")
    cpu_values = tensor.detach().contiguous().to(device="cpu", dtype=torch.float32)
    return memoryview(cpu_values.numpy()).cast("B").tobytes()


def _validate_layer_weight_digests(
    loaded_weights: list[dict[str, Any]], prefix_document: dict[str, Any], layer_index: int
) -> None:
    prefix_sources = {
        source["key"]: source
        for source in prefix_document["weight_sources"]
        if source["key"].startswith(f"model.layers.{layer_index}.")
    }
    loaded_sources = {source["key"]: source for source in loaded_weights}
    if len(prefix_sources) != 12 or set(prefix_sources) != set(loaded_sources):
        raise ValueError("selected layer weight inventory differs from the authenticated prefix")
    for key, loaded in loaded_sources.items():
        reference = prefix_sources[key]
        for field in ("shard", "dtype", "shape", "nbytes", "tensor_sha256"):
            if loaded[field] != reference[field]:
                raise ValueError(f"selected layer weight provenance differs from trusted prefix: {key}")


def _append_sidecar_record(
    sidecars: dict[str, bytearray], name: str, dtype: str, shape: list[int], raw: bytes
) -> dict[str, Any]:
    byte_width = {"bfloat16": 2, "float32": 4}.get(dtype)
    if byte_width is None:
        raise ValueError(f"unsupported diagnostic dtype: {dtype}")
    expected_nbytes = math.prod(shape) * byte_width
    if len(raw) != expected_nbytes:
        raise RuntimeError(f"captured {name} byte count does not match its declared dtype and shape")
    sidecar_name = "bf16" if dtype == "bfloat16" else "f32"
    payload = sidecars.setdefault(sidecar_name, bytearray())
    if len(payload) + expected_nbytes > MAX_DIAGNOSTIC_SIDECAR_BYTES:
        raise ValueError(f"diagnostic {sidecar_name} sidecar exceeds its bounded size")
    record = {
        "name": name,
        "dtype": dtype,
        "shape": shape,
        "storage": {
            "sidecar": sidecar_name,
            "offset_bytes": len(payload),
            "nbytes": expected_nbytes,
        },
    }
    payload.extend(raw)
    return record


def _pack_diagnostic_sidecars(
    layer_index: int,
    hidden_states_in: Any,
    stages: dict[str, Any],
    extra_stages: dict[str, Any],
    rms_stages: dict[str, Any],
    stage_order: list[str],
    expected_shapes: dict[str, list[int]],
    torch: Any,
    rms_site: str = "input_layernorm",
) -> tuple[dict[str, bytes], dict[str, Any], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    layer_index = _validate_layer_index(layer_index)
    rms_entries = _qualified_rms_trace_entries(layer_index, rms_stages, rms_site)
    sidecar_buffers: dict[str, bytearray] = {}
    records: dict[str, dict[str, Any]] = {}
    tensor_entries = (
        [("hidden_states_in", hidden_states_in, "bfloat16")]
        + [(name, stages[name], "bfloat16") for name in stage_order]
        + [(name, tensor, "bfloat16") for name, tensor in extra_stages.items()]
        + rms_entries
    )
    for name, tensor, dtype in tensor_entries:
        expected_shape = expected_shapes[name]
        if list(tensor.shape) != expected_shape:
            raise RuntimeError(f"captured {name} shape differs from the native diagnostic contract")
        if tensor.device.type != "mps":
            raise RuntimeError(f"captured {name} escaped MPS before the terminal wait")
        if dtype == "bfloat16":
            if tensor.dtype != torch.bfloat16:
                raise RuntimeError(f"captured {name} dtype differs from BF16 contract")
            raw = _tensor_bf16_bytes(tensor, torch)
        else:
            if tensor.dtype != torch.float32:
                raise RuntimeError(f"captured {name} dtype differs from F32 contract")
            raw = _tensor_f32_bytes(tensor, torch)
        records[name] = _append_sidecar_record(sidecar_buffers, name, dtype, expected_shape, raw)

    if "hidden_states_in" not in records:
        raise RuntimeError("diagnostic sidecar is missing hidden_states_in")
    stage_records = [records[name] for name in stage_order]
    extra_stage_records = [records[name] for name in extra_stages]
    rms_stage_records = [records[name] for name, _, _ in rms_entries]
    hidden_record = {key: value for key, value in records["hidden_states_in"].items() if key != "name"}
    return (
        {name: bytes(payload) for name, payload in sidecar_buffers.items()},
        hidden_record,
        stage_records,
        extra_stage_records,
        rms_stage_records,
    )


def _fresh_output_paths(output_path: Path) -> tuple[Path, dict[str, Path]]:
    output_path = output_path.expanduser().absolute()
    parent = output_path.parent.resolve(strict=True)
    if not parent.is_dir():
        raise ValueError("diagnostic output parent must already exist")
    manifest_path = parent / output_path.name
    sidecars = {
        "bf16": parent / f"{output_path.name}.bf16le",
        "f32": parent / f"{output_path.name}.f32le",
    }
    if manifest_path.exists() or manifest_path.is_symlink() or any(
        path.exists() or path.is_symlink() for path in sidecars.values()
    ):
        raise FileExistsError("diagnostic manifest and sidecar output paths must be fresh")
    return manifest_path, sidecars


def _write_diagnostic(
    output_path: Path, document: dict[str, Any], sidecar_bytes: dict[str, bytes]
) -> tuple[Path, Path]:
    manifest_path, sidecar_paths = _fresh_output_paths(output_path)
    if set(sidecar_bytes) != set(sidecar_paths):
        raise ValueError("diagnostic sidecar payload set is incomplete")
    document["sidecars"] = {
        name: ({
            "path": path.name,
            "encoding": "bfloat16-le" if name == "bf16" else "float32-le",
            "nbytes": len(sidecar_bytes[name]),
            "sha256": _sha256_bytes(sidecar_bytes[name]),
        } | ({"byte_order": "little", "storage_dtype": "float32"} if name == "f32" else {}))
        for name, path in sidecar_paths.items()
    }
    created_sidecars = []
    try:
        for name, path in sidecar_paths.items():
            with path.open("xb") as output:
                created_sidecars.append(path)
                output.write(sidecar_bytes[name])
                output.flush()
                os.fsync(output.fileno())
        with manifest_path.open("xb") as output:
            output.write((json.dumps(document, indent=2, sort_keys=True) + "\n").encode("utf-8"))
            output.flush()
            os.fsync(output.fileno())
    except Exception:
        if not manifest_path.exists():
            for path in created_sidecars:
                if path.exists():
                    path.unlink()
        raise
    return manifest_path, sidecar_paths["bf16"]


def _run(args: argparse.Namespace) -> dict[str, Any]:
    if os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK") not in (None, "0"):
        raise RuntimeError("PYTORCH_ENABLE_MPS_FALLBACK must be unset or 0")
    os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "0"
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    if sys.byteorder != "little":
        raise RuntimeError("checkpoint BF16 source preparation requires a little-endian host")

    producer = _load_pretrained_producer()
    layer_index = _validate_layer_index(args.layer_index, producer.PREFIX_LAYER_COUNT)
    rms_site = _normalize_rms_site(args.rms_site)
    _fresh_output_paths(args.output)
    prefix_snapshot = _read_prefix_snapshot(
        args.prefix_reference,
        args.prefix_reference_sha256,
        lambda document: producer._validate_manifest(document, manifest_path=None),
    )
    input_state_bytes, expected_output_bytes, _, request = _prefix_state_bytes(
        prefix_snapshot, layer_index, producer
    )
    prefix_document = prefix_snapshot["document"]
    prefix_config = prefix_document["config"]
    if prefix_config.get("attention_implementation") != "sdpa":
        raise ValueError("trusted prefix does not use official SDPA attention")

    model_config_json, index, checkpoint = _load_selected_checkpoint(
        args.checkpoint_root, args.checkpoint_receipt, layer_index, producer
    )
    text_values = producer._text_config(model_config_json)
    if text_values != checkpoint["text_config"]:
        raise RuntimeError("checkpoint text configuration changed during verification")

    import torch
    from safetensors import safe_open

    if not torch.backends.mps.is_built() or not torch.backends.mps.is_available():
        raise RuntimeError("pretrained block oracle requires MPS; CPU fallback is forbidden")
    helper = producer._load_upstream_helper()
    modeling, configuration, transformers_sources, transformers_version = helper._import_official(torch)
    torch_sources = helper._verify_torch_sources(torch)
    config = producer._parse_text_config(configuration, text_values)
    config.num_hidden_layers = producer.SOURCE_LAYER_COUNT
    config.use_cache = False
    config._attn_implementation = "sdpa"
    config_record = producer._materialize_config_record(config, request["sequence_length"])
    if config_record["sequence_length"] != prefix_config.get("sequence_length"):
        raise ValueError("trusted prefix and official config disagree on sequence length")

    with torch.device("meta"):
        shape_probe = modeling.Qwen2_5_VLDecoderLayer(config, layer_idx=layer_index)
    _validate_official_rmsnorm_instance(
        getattr(shape_probe, rms_site), modeling.Qwen2RMSNorm, transformers_sources
    )
    expected_weight_shapes = producer._expected_weight_names(shape_probe)
    stage_shapes = _stage_shapes(layer_index, request["sequence_length"], config_record, helper)
    rms_stage_shapes = _rms_trace_shapes(
        layer_index, request["sequence_length"], config_record["hidden_size"], rms_site
    )
    extra_stage_name = f"layers.{layer_index}.mlp.act_fn"
    extra_stage_shapes = {
        extra_stage_name: [1, request["sequence_length"], config_record["intermediate_size"]]
    }
    estimate = _estimate_block_mps_bytes(
        expected_weight_shapes,
        stage_shapes,
        request["sequence_length"],
        config_record["hidden_size"],
        config_record["intermediate_size"],
    )
    if estimate > MAX_BLOCK_MPS_ESTIMATE_BYTES:
        raise MemoryError("one-block diagnostic estimate exceeds the 3 GiB limit")
    _require_block_memory_budget(
        estimate,
        int(torch.mps.current_allocated_memory()),
        int(torch.mps.recommended_max_memory()),
    )
    del shape_probe

    weight_map = checkpoint["index_weight_map"]
    device = torch.device("mps")
    with torch.device("meta"):
        layer = modeling.Qwen2_5_VLDecoderLayer(config, layer_idx=layer_index)
    selected_rmsnorm = getattr(layer, rms_site)
    _validate_official_rmsnorm_instance(
        selected_rmsnorm, modeling.Qwen2RMSNorm, transformers_sources
    )
    if producer._expected_weight_names(layer) != expected_weight_shapes:
        raise RuntimeError("selected official decoder layer parameter geometry changed")
    layer, weight_sources = producer._load_decoder_layer(
        layer_index,
        layer,
        expected_weight_shapes,
        Path(checkpoint["root"]),
        weight_map,
        torch,
        safe_open,
    )
    _validate_layer_weight_digests(weight_sources, prefix_document, layer_index)

    hidden_shape = [1, request["sequence_length"], config_record["hidden_size"]]
    input_hidden_states = _state_tensor_from_bytes(input_state_bytes, hidden_shape, torch, device)
    rotary = modeling.Qwen2_5_VLRotaryEmbedding(config=config, device=device)
    rotary.eval()
    position_values = [request["position_offset"] + position for position in range(request["sequence_length"])]
    multimodal_positions = torch.tensor(
        [position_values, position_values, position_values], device=device, dtype=torch.int64
    ).reshape(3, 1, request["sequence_length"])
    binary_mask = torch.tensor(request["attention_mask"], device=device, dtype=torch.int64).reshape(1, -1)
    cache_position = torch.arange(request["sequence_length"], device=device, dtype=torch.int64)

    with torch.inference_mode():
        cos, sin = rotary(input_hidden_states, multimodal_positions)
        causal_mask = modeling.create_causal_mask(
            config=config,
            input_embeds=input_hidden_states,
            attention_mask=binary_mask,
            cache_position=cache_position,
            past_key_values=None,
            position_ids=None,
        )
        if causal_mask is None or not isinstance(causal_mask, torch.Tensor):
            raise RuntimeError("official causal-mask helper omitted the required right-padding mask")
        if causal_mask.device.type != "mps" or causal_mask.dtype != torch.bool:
            raise RuntimeError("official causal mask must remain bool on MPS")
        expected_mask_shape = (1, 1, request["sequence_length"], request["sequence_length"])
        if tuple(causal_mask.shape) != expected_mask_shape:
            raise RuntimeError("official causal mask has an unexpected shape")
        activation_capture = []

        def capture_mlp_activation(_module: Any, _inputs: Any, output: Any) -> None:
            if isinstance(output, torch.Tensor):
                if output.device.type != "mps" or output.dtype != torch.bfloat16:
                    raise RuntimeError("official SiLU activation escaped BF16 MPS")
                activation_capture.append(output.detach().clone())

        activation_hook = layer.mlp.act_fn.register_forward_hook(capture_mlp_activation)
        try:
            with _observe_official_rmsnorm(selected_rmsnorm, torch) as captured_rms_short_names:
                output_hidden_states, captured = helper._capture_layer(
                    layer,
                    layer_index,
                    input_hidden_states,
                    causal_mask,
                    None,
                    (cos, sin),
                    modeling,
                    torch,
                )
        finally:
            activation_hook.remove()
        if len(activation_capture) != 1:
            raise RuntimeError("official MLP activation did not produce exactly one diagnostic output")
        captured_stages = {
            "rotary_cos": cos[0, 0],
            "rotary_sin": sin[0, 0],
            **{name: captured[name] for name in captured if name != "hidden_states_in"},
        }
        stage_order = [*STAGE_PREFIX_NAMES, *(name.format(i=layer_index) for name in helper.STAGE_NAMES)]
        if set(captured_stages) != set(stage_order):
            raise RuntimeError("official layer capture did not produce the complete native stage set")
        captured_extra_stages = {extra_stage_name: activation_capture[0]}
        if list(captured_extra_stages[extra_stage_name].shape) != extra_stage_shapes[extra_stage_name]:
            raise RuntimeError("official SiLU activation has an unexpected shape")
        _validate_rms_trace(captured_rms_short_names, rms_stage_shapes, torch, require_mps=True)
        captured_rms_stages = {
            name: captured_rms_short_names[name]
            for name in RMS_TRACE_STAGE_SUFFIXES
        }
        for name, expected_shape in stage_shapes.items():
            tensor = captured_stages.get(name)
            if tensor is None or tensor.device.type != "mps" or tensor.dtype != torch.bfloat16:
                raise RuntimeError(f"official captured stage is not BF16 on MPS: {name}")
            if list(tensor.shape) != expected_shape:
                raise RuntimeError(f"official captured stage shape changed: {name}")
        if captured["hidden_states_in"].dtype != torch.bfloat16 or captured["hidden_states_in"].device.type != "mps":
            raise RuntimeError("official layer input capture must remain BF16 on MPS")
        # The single explicit device wait occurs after the one real block.
        torch.mps.synchronize()

    rms_input_bytes = _tensor_bf16_bytes(captured_rms_short_names["input_bf16"], torch)
    residual_after_attention_bytes = b""
    if rms_site == "post_attention_layernorm":
        residual_after_attention_bytes = _tensor_bf16_bytes(
            captured["residual_after_attention"], torch
        )
    rms_input_sha256 = _validate_rms_input_binding(
        rms_site,
        rms_input_bytes,
        input_state_bytes,
        residual_after_attention_bytes,
        layer_index,
    )
    official_inv_freq = (
        rotary.inv_freq.detach()
        .to(device="cpu", dtype=torch.float32)
        .contiguous()
        .reshape(-1)
        .tolist()
    )
    _verify_config_generated_inv_freq(
        prefix_document["inputs"]["inv_freq_f32"], official_inv_freq
    )
    actual_output_bytes = _tensor_bf16_bytes(output_hidden_states, torch)
    if actual_output_bytes != expected_output_bytes:
        mismatch_words = sum(
            actual_output_bytes[offset : offset + 2] != expected_output_bytes[offset : offset + 2]
            for offset in range(0, min(len(actual_output_bytes), len(expected_output_bytes)), 2)
        )
        raise RuntimeError(
            "standalone official block output does not reproduce the authenticated prefix state "
            f"h{layer_index + 1}: {mismatch_words} BF16 words differ"
        )

    sidecar_bytes, hidden_record, stage_records, extra_stage_records, rms_stage_records = _pack_diagnostic_sidecars(
        layer_index,
        captured["hidden_states_in"],
        captured_stages,
        captured_extra_stages,
        captured_rms_stages,
        stage_order,
        {"hidden_states_in": hidden_shape, **stage_shapes, **extra_stage_shapes, **rms_stage_shapes},
        torch,
        rms_site=rms_site,
    )
    prefix_hidden_digest = _sha256_bytes(input_state_bytes)
    prefix_output_digest = _sha256_bytes(expected_output_bytes)
    diagnostic = {
        "schema": SCHEMA,
        "schema_version": SCHEMA_VERSION,
        "checkpoint": {
            key: checkpoint[key]
            for key in ("repo_id", "revision", "receipt_sha256", "config_sha256", "index_sha256", "verified_files")
        },
        "prefix_reference": {
            "manifest_sha256": prefix_snapshot["manifest_sha256"],
            "sidecar_sha256": prefix_snapshot["sidecar_sha256"],
            "input_state_index": layer_index,
            "expected_output_state_index": layer_index + 1,
            "state_indexing": prefix_document["capture"]["state_indexing"],
            "input_state_sha256": prefix_hidden_digest,
            "expected_output_state_sha256": prefix_output_digest,
        },
        "config": config_record,
        "inputs": _diagnostic_inputs(request, prefix_document["inputs"]),
        "layer": {
            "index": layer_index,
            "weights": weight_sources,
        },
        "hidden_states_in": hidden_record,
        "stages": stage_records,
        "extra_stages": extra_stage_records,
        "rms_trace": {
            "site": rms_site,
            "source": {
                "schema_version": 1,
                "source_file": RMS_SOURCE_FILE,
                "source_sha256": _verified_rms_source_sha256(transformers_sources),
                "capture_method": "TorchDispatchMode pass-through around the official Qwen2RMSNorm.forward call",
            },
            "input_state_sha256": rms_input_sha256,
            "stages": rms_stage_records,
        },
        "provenance": {
            "producer_source_sha256": producer._sha256_file(Path(__file__).resolve()),
            "upstream_helper_source_sha256": producer._sha256_file(producer.UPSTREAM_HELPER_PATH),
            "transformers_sources": transformers_sources,
            "torch_sources": torch_sources,
            "python_version": platform.python_version(),
            "platform": platform.platform(),
            "torch_version": str(torch.__version__),
            "transformers_version": transformers_version,
            "device": "mps",
            "activation_dtype": "bfloat16",
            "attention_implementation": "sdpa",
            "explicit_mps_synchronizations": 1,
            "config_generated_inv_freq_bitexact": True,
            "rms_trace_site_input_bitexact": True,
            "rms_trace_site": rms_site,
            **({"input_layernorm_input_bitexact": True} if rms_site == "input_layernorm" else {}),
            "mps_memory_estimate_bytes": estimate,
            "mps_reserved_fraction": MPS_RESERVE_FRACTION,
            "prefix_output_bitexact": True,
        },
    }
    _write_diagnostic(args.output, diagnostic, sidecar_bytes)
    return diagnostic


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-root", required=True, type=Path)
    parser.add_argument("--checkpoint-receipt", required=True, type=Path)
    parser.add_argument("--prefix-reference", required=True, type=Path)
    parser.add_argument("--prefix-reference-sha256", required=True)
    parser.add_argument("--layer-index", type=int, default=24)
    parser.add_argument(
        "--rms-site",
        choices=tuple(RMS_SITE_MODULES),
        default="input",
        help="official RMSNorm site to trace (default: input)",
    )
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    diagnostic = _run(args)
    print(
        json.dumps(
            {
                "schema": diagnostic["schema"],
                "layer_index": diagnostic["layer"]["index"],
                "prefix_output_bitexact": diagnostic["provenance"]["prefix_output_bitexact"],
                "stage_count": len(diagnostic["stages"]),
                "output": str(args.output),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
