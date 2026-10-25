#!/usr/bin/env python3
"""Produce a bounded actual-weight Hy-Nomas text-conditioning oracle fixture.

This opt-in helper reads only the authenticated Qwen conditioning sidecar, the
42-tensor text/time subset, and one actual DiT block subset. It runs the pinned
``SingleTokenRefinerPerBlock`` and ``MMDoubleStreamBlock`` on MPS in F32 with
weights rounded through BF16. The full 1280-row masked refiner is the source
reference; a compact 23-row control is compared at every checkpoint before a
fixture is admitted. All checkpoint tensors stay on MPS until one packed final
readback. This is not CUDA/BF16 activation parity or full generation.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import math
import os
from pathlib import Path
import re
import sys
import tempfile
import types
from typing import Any


PINNED_REVISION = "911cea6458a89c2663513ff7099561ec190dd491"
DEFAULT_UPSTREAM = Path(
    "/opt/mldata_ssd/cogni-ml/video-scene-20261019/upstream/hy-nomas-20261024"
)
DEFAULT_QWEN_REFERENCE = Path(
    "/opt/mldata_ssd/cogni-ml/qwen25-pretrained-20261024/fixtures/prefix-natural-tram-v1.json"
)
DEFAULT_QWEN_SIDECAR = DEFAULT_QWEN_REFERENCE.with_suffix(".bf16le")
DEFAULT_TEXT_WEIGHTS = Path(
    "/opt/mldata_ssd/cogni-ml/hy-nomas-text-conditioning-20261025/weights/text-time-f32.safetensors"
)
DEFAULT_TEXT_MANIFEST = Path(str(DEFAULT_TEXT_WEIGHTS) + ".manifest.json")
DEFAULT_BLOCK_WEIGHTS = Path(
    "/opt/mldata_ssd/cogni-ml/hy-nomas-bf16-20261024/weights/distilled-block0-f32.safetensors"
)
DEFAULT_BLOCK_MANIFEST = Path(str(DEFAULT_BLOCK_WEIGHTS) + ".manifest.json")
QWEN_REFERENCE_SHA256 = "eb64e5c31bc2178436b2b7dd411af45dea7fa43a8e3ff37a146bc411fac03ba1"
QWEN_SIDECAR_SHA256 = "e7ebc9a784124afc20503435d84d42e0110d7c02054d3fc63fd586ec69ed4995"
TEXT_WEIGHTS_SHA256 = "58313f6141adafc12629b46c73f1bc270aa64c3e0972a8995875ba76d9068d8f"
TEXT_MANIFEST_SHA256 = "0863d49ba53d54db5ad7ff5313099c3e93003cf304e729f4a377098f230f462f"
BLOCK_WEIGHTS_SHA256 = "6810bf7cd7f9a8ac4d66240eee514637a7ccaa47dc9de02e8bc26ae4543530d1"
BLOCK_MANIFEST_SHA256 = "c7afec8da5ac22a37d8e38b42c9f5ecdabab7929b5c40da9260e7c30c9c73016"

ATOL = 2e-5
RTOL = 2e-5
SCHEMA = "hy-nomas-text-conditioning-reference-v1"
SIDE_CAR_LIMIT = 64 * 1024 * 1024
TEXT_SOURCE = {
    "repository": "tencent/Hy-Nomas",
    "revision": "bac719a80c8d02f91c485fca804bb8221e149b70",
    "checkpoint_path": "DiT/distilled/diffusion_pytorch_model_ema_0_995.safetensors",
    "checkpoint_lfs_sha256": "73819bcf3b82b090116dd1a1748795de95eef703c304e7443963d00ea8aa4d70",
    "source_git_commit": PINNED_REVISION,
}
QWEN_SOURCE = {
    "reference_sha256": QWEN_REFERENCE_SHA256,
    "sidecar_sha256": QWEN_SIDECAR_SHA256,
    "state_index": 26,
    "crop": [108, 1388],
}
GEOMETRY = {
    "batches": 1,
    "text_tokens": 1280,
    "valid_tokens": 23,
    "input_dim": 3584,
    "dim": 2048,
    "heads": 16,
    "head_dim": 128,
    "ffn_dim": 8192,
    "video_tokens": 6,
    "segments": 1,
    "chunk_size_tokens": 0,
    "text_seg_pos_emb_num": 0,
}

# These helpers and the complete actual modules are independently checked
# against the pinned source checkout before import or execution.
PINNED_SOURCE_SHA256 = {
    "hymm/models/dit/nomas_cpt.py": "acf827bb7feb84c9046618737146544843a3f50c3500a3b16925dadeb6c5bdee",
    "hymm/models/text_encoder/token_refiner.py": "7c67d81b81d9b183c3261eb4bf87bb04e0bda3142ce096afc2e64aa722433eab",
    "hymm/models/dit/modules/embed_layers.py": "5b055e383bbf532ea5aa49ee506031d96bea3e3d5b8da3a104d4bd875c2d0af3",
    "hymm/models/dit/modules/int8_stored_bf16_linear.py": "2bfd28f483c0da76df5ece7b23d0d3f62ef38725317b13e55d41ce997812122a",
    "hymm/models/dit/modules/activation_layers.py": "1e509b5e5f9527d199d6f0206f452e56fdb1bd25ce3ccc06fb47f3a16060ca39",
    "hymm/models/dit/modules/norm_layers.py": "2bcd5b29c951051ee1afaeac08258a3ef1dea3205b821f236594719331127044",
    "hymm/models/dit/modules/modulate_layers.py": "abfb87acc9a972b40d0f36cfa832f0816d978fc13f93236aa1dfc902aac5caee",
    "hymm/models/dit/modules/mlp_layers.py": "b77d862d17cd2c7db6237931b80f33b6447b7ba4db5918d3fbdfd17a6fb47e07",
    "hymm/models/dit/modules/posemb_layers.py": "27b729d04185dfd7d65ecf9ba0d761d4864153d3e656908997d60a6971afc1aa",
    "hymm/models/dit/modules/attention/__init__.py": "2a59284ac06057ac622228d280a32fc9e907a333e8ef0efcd526928626858450",
    "hymm/utils/helpers.py": "51b7a08377843b5eecc59e6fd41f10317d2de096d30fc3154491aacd27fefdfb",
}


class ReferenceError(RuntimeError):
    """A fail-closed input, provenance, execution, or fixture error."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _digest_arg(value: str, label: str) -> str:
    if re.fullmatch(r"[0-9a-f]{64}", value) is None:
        raise ReferenceError(f"{label} must be 64 lowercase hexadecimal characters")
    return value


def _load_trusted_manifest(path: Path, supplied_sha256: str, *, kind: str) -> dict[str, Any]:
    expected = _digest_arg(supplied_sha256, f"{kind} manifest digest")
    resolved = path.resolve(strict=True)
    if not resolved.is_file() or resolved.is_symlink() or resolved.stat().st_size > 2 * 1024 * 1024:
        raise ReferenceError(f"{kind} manifest must be a bounded regular file")
    actual = sha256_file(resolved)
    if actual != expected:
        raise ReferenceError(f"{kind} manifest digest mismatch: {actual} != {expected}")
    try:
        value = json.loads(resolved.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, RecursionError):
        raise ReferenceError(f"{kind} manifest is not valid UTF-8 JSON") from None
    if not isinstance(value, dict):
        raise ReferenceError(f"{kind} manifest root must be an object")
    return value


def _verify_text_manifest(manifest: dict[str, Any], weights_path: Path, weights_sha256: str) -> None:
    if manifest.get("format") != "hy-nomas-text-subset-provenance-v1":
        raise ReferenceError("text manifest format mismatch")
    source = manifest.get("source")
    subset = manifest.get("subset")
    if not isinstance(source, dict) or not isinstance(subset, dict):
        raise ReferenceError("text manifest provenance is incomplete")
    for key, expected in TEXT_SOURCE.items():
        if source.get(key) != expected:
            raise ReferenceError(f"text manifest source mismatch: {key}")
    if subset.get("dtype") != "F32" or subset.get("tensor_count") != 42:
        raise ReferenceError("text manifest must authenticate exactly 42 F32 tensors")
    if subset.get("file_sha256") != weights_sha256 or subset.get("file_bytes") != weights_path.stat().st_size:
        raise ReferenceError("text manifest does not match the exact subset bytes")
    if source.get("payload_sha256_observed") != "8b60f66d65d54332ecd96aa89ba4fb6c4565cb11cefccee50f4f68138301b1a5":
        raise ReferenceError("text manifest range payload digest mismatch")


def _verify_block_manifest(manifest: dict[str, Any], weights_path: Path, weights_sha256: str) -> None:
    output = manifest.get("output")
    source = manifest.get("source")
    if manifest.get("format") != "hy-nomas-block-subset-provenance-v1" or not isinstance(output, dict) or not isinstance(source, dict):
        raise ReferenceError("block manifest format/provenance mismatch")
    if output.get("sha256") != weights_sha256 or output.get("size_bytes") != weights_path.stat().st_size:
        raise ReferenceError("block manifest does not match the exact subset bytes")
    if source.get("source_git_commit") != PINNED_REVISION or source.get("repository") != TEXT_SOURCE["repository"]:
        raise ReferenceError("block manifest source revision mismatch")
    if source.get("revision") != TEXT_SOURCE["revision"]:
        raise ReferenceError("block manifest checkpoint revision mismatch")


def validate_qwen_reference(reference: dict[str, Any], *, reference_sha256: str,
                            sidecar_bytes: int, sidecar_sha256: str) -> list[int]:
    """Validate the existing Qwen capture and return its CPU-known cropped mask."""
    if _digest_arg(reference_sha256, "Qwen reference digest") != QWEN_REFERENCE_SHA256:
        raise ReferenceError("Qwen reference digest is not the qualified fixture")
    if sidecar_sha256 != QWEN_SIDECAR_SHA256 or sidecar_bytes != 268_627_968:
        raise ReferenceError("Qwen BF16 sidecar digest or byte count mismatch")
    if reference.get("schema") != "cogni_ml.qwen25_vl_pretrained_decoder_prefix_reference" or reference.get("schema_version") != 1:
        raise ReferenceError("Qwen source schema mismatch")
    provenance = reference.get("provenance")
    capture = reference.get("capture")
    sidecars = reference.get("sidecars")
    inputs = reference.get("inputs")
    if not all(isinstance(item, dict) for item in (provenance, capture, sidecars, inputs)):
        raise ReferenceError("Qwen source provenance is incomplete")
    if provenance.get("activation_dtype") != "bfloat16" or capture.get("captured_state_count") != 27:
        raise ReferenceError("Qwen capture dtype/state count mismatch")
    bf16 = sidecars.get("bf16")
    if not isinstance(bf16, dict) or bf16.get("encoding") != "bfloat16-le" or bf16.get("nbytes") != sidecar_bytes or bf16.get("sha256") != sidecar_sha256:
        raise ReferenceError("Qwen sidecar metadata mismatch")
    mask = inputs.get("attention_mask")
    if not isinstance(mask, list) or len(mask) != 1388 or any(type(value) is not int or value not in (0, 1) for value in mask):
        raise ReferenceError("Qwen attention mask does not have the pinned shape")
    cropped = mask[108:1388]
    if len(cropped) != 1280 or sum(cropped) != 23 or cropped != [1] * 23 + [0] * 1257:
        raise ReferenceError("Qwen crop is not the qualified 23-row valid prefix")
    return [index for index, value in enumerate(cropped) if value]


def validate_reference_document(document: dict[str, Any]) -> None:
    """Validate the admitted fixture envelope without importing torch."""
    if (not isinstance(document, dict) or document.get("schema") != SCHEMA
            or type(document.get("schema_version")) is not int or document.get("schema_version") != 1):
        raise ReferenceError("missing or invalid text-conditioning schema")
    if document.get("upstream_revision") != PINNED_REVISION:
        raise ReferenceError("text-conditioning upstream revision mismatch")
    geometry = document.get("geometry")
    if (not isinstance(geometry, dict) or geometry != GEOMETRY
            or any(type(value) is not int for value in geometry.values())):
        raise ReferenceError("text-conditioning geometry mismatch")
    precision = document.get("precision")
    if not isinstance(precision, dict) or precision.get("resident_weights") != "BF16" or precision.get("compute") != "F32" or precision.get("attention") not in ("f32", "bf16-boundaries"):
        raise ReferenceError("unsupported text-conditioning precision policy")
    qwen = document.get("qwen_source")
    if not isinstance(qwen, dict) or qwen != QWEN_SOURCE:
        raise ReferenceError("text-conditioning Qwen source mismatch")
    mask = document.get("mask")
    if (not isinstance(mask, list) or len(mask) != 1280
            or any(type(value) is not int for value in mask)
            or mask != [1] * 23 + [0] * 1257):
        raise ReferenceError("text-conditioning mask mismatch")
    if type(document.get("timestep")) not in (int, float) or float(document["timestep"]) != 500.0:
        raise ReferenceError("text-conditioning timestep mismatch")
    compaction = document.get("compaction")
    if not isinstance(compaction, dict) or type(compaction.get("passed")) is not bool or not isinstance(compaction.get("comparisons"), list):
        raise ReferenceError("text-conditioning compaction report is malformed")
    if not compaction["passed"]:
        raise ReferenceError("text-conditioning compaction gate failed")


def validate_tensor_layout(tensors: dict[str, Any], sidecar_bytes: int) -> None:
    if type(sidecar_bytes) is not int or not 0 < sidecar_bytes <= SIDE_CAR_LIMIT:
        raise ReferenceError("tensor sidecar size outside limit")
    if not isinstance(tensors, dict) or not tensors:
        raise ReferenceError("tensor inventory is empty")
    for name, metadata in tensors.items():
        if not isinstance(name, str) or not isinstance(metadata, dict) or metadata.get("dtype") != "F32":
            raise ReferenceError("invalid F32 tensor metadata")
        if type(metadata.get("offset_bytes")) is not int:
            raise ReferenceError(f"invalid tensor offset: {name}")
    cursor = 0
    for name, metadata in sorted(tensors.items(), key=lambda item: item[1]["offset_bytes"]):
        shape = metadata.get("shape")
        if not isinstance(shape, list) or not shape or len(shape) > 4 or any(type(dim) is not int or dim <= 0 for dim in shape):
            raise ReferenceError(f"invalid tensor shape for {name}")
        count = math.prod(shape)
        size = count * 4
        if type(metadata.get("offset_bytes")) is not int or metadata["offset_bytes"] != cursor or metadata.get("nbytes") != size:
            raise ReferenceError(f"noncontiguous or malformed tensor range: {name}")
        if size > sidecar_bytes - cursor:
            raise ReferenceError(f"tensor range exceeds sidecar: {name}")
        cursor += size
    if cursor != sidecar_bytes:
        raise ReferenceError("tensor sidecar has trailing or missing bytes")


def _text_state(torch: Any, path: Path) -> dict[str, Any]:
    from safetensors import safe_open

    with safe_open(str(path), framework="pt", device="cpu") as source:
        keys = set(source.keys())
        expected = {name for name in keys if name.startswith("txt_in.")} | {
            "time_in.mlp.0.weight", "time_in.mlp.0.bias",
            "time_in.mlp.2.weight", "time_in.mlp.2.bias",
        }
        if len(keys) != 42 or keys != expected or len([key for key in keys if key.startswith("txt_in.")]) != 38:
            raise ReferenceError("text subset does not contain the exact 38 txt_in and 4 time_in keys")
        state: dict[str, Any] = {}
        for name in sorted(keys):
            value = source.get_tensor(name)
            if value.dtype != torch.float32 or not bool(torch.isfinite(value).all().item()):
                raise ReferenceError(f"text subset tensor is not finite F32: {name}")
            if name.startswith("txt_in."):
                state[name.removeprefix("txt_in.")] = value
        return state


def _time_state(torch: Any, path: Path) -> dict[str, Any]:
    from safetensors import safe_open

    with safe_open(str(path), framework="pt", device="cpu") as source:
        all_keys = set(source.keys())
        names = {"time_in.mlp.0.weight", "time_in.mlp.0.bias", "time_in.mlp.2.weight", "time_in.mlp.2.bias"}
        if len(all_keys) != 42 or {name for name in all_keys if name.startswith("time_in.")} != names:
            raise ReferenceError("text subset key inventory changed")
        return {name.removeprefix("time_in."): source.get_tensor(name) for name in sorted(names)}


def _round_module_to_bf16_values(torch: Any, module: Any, device: Any) -> None:
    from hy_nomas_reference import _round_parameters_bf16_in_place

    _round_parameters_bf16_in_place(torch, module, device)


def _capture_module(module: Any, target: dict[str, Any], key: str,
                    transform: Any = None) -> Any:
    def hook(_module: Any, _inputs: tuple[Any, ...], output: Any) -> None:
        if not isinstance(output, __import__("torch").Tensor):
            raise ReferenceError(f"expected tensor checkpoint at {key}")
        value = transform(output) if transform is not None else output
        target[key] = value.detach().clone()

    return module.register_forward_hook(hook)


def _capture_input(module: Any, target: dict[str, Any], key: str,
                   input_index: int = 0, transform: Any = None) -> Any:
    def hook(_module: Any, inputs: tuple[Any, ...]) -> None:
        value = inputs[input_index]
        if not isinstance(value, __import__("torch").Tensor):
            raise ReferenceError(f"expected tensor input checkpoint at {key}")
        value = transform(value) if transform is not None else value
        target[key] = value.detach().clone()

    return module.register_forward_pre_hook(hook)


def _install_text_attention(torch: Any, *, valid_indices: list[int],
                            precision: str, capture: dict[str, Any]) -> tuple[Any, Any]:
    """Inject an MPS SDPA adapter that unpads known valid rows without CPU reads."""
    if precision not in ("f32", "bf16-boundaries"):
        raise ReferenceError("unsupported attention precision")
    module_name = "hymm.models.dit.modules.attention"
    attention_module = types.ModuleType(module_name)

    def attention(q: Any, k: Any, v: Any, drop_rate: float = 0,
                  attn_mask: Any = None, causal: bool = False) -> Any:
        if drop_rate != 0 or causal or q.device.type != "mps" or k.device.type != "mps" or v.device.type != "mps":
            raise ReferenceError("text adapter admits only MPS F32 unmasked-dropout noncausal attention")
        if q.dtype != torch.float32 or k.dtype != torch.float32 or v.dtype != torch.float32:
            raise ReferenceError("text adapter requires F32 Q/K/V")
        if q.shape != k.shape or q.shape != v.shape or q.ndim != 4:
            raise ReferenceError("text adapter Q/K/V shape mismatch")
        full_tokens = int(q.shape[1])
        if attn_mask is None:
            if full_tokens != len(valid_indices):
                raise ReferenceError("compact refiner call must use the admitted valid-row extent")
            indices = torch.arange(full_tokens, device=q.device, dtype=torch.int64)
        else:
            if tuple(attn_mask.shape) != (int(q.shape[0]), full_tokens) or attn_mask.dtype != torch.bool or attn_mask.device.type != "mps":
                raise ReferenceError("full refiner mask must be an MPS bool [B,S] tensor")
            if full_tokens == 1280:
                indices = torch.tensor(valid_indices, device=q.device, dtype=torch.int64)
            elif full_tokens == len(valid_indices):
                indices = torch.arange(full_tokens, device=q.device, dtype=torch.int64)
            else:
                raise ReferenceError("masked refiner path must be full 1280 or compact valid-prefix extent")
        q_valid = q.index_select(1, indices)
        k_valid = k.index_select(1, indices)
        v_valid = v.index_select(1, indices)
        if precision == "bf16-boundaries":
            q_valid = q_valid.to(torch.bfloat16).to(torch.float32)
            k_valid = k_valid.to(torch.bfloat16).to(torch.float32)
            v_valid = v_valid.to(torch.bfloat16).to(torch.float32)
        qh = q_valid.permute(0, 2, 1, 3).contiguous()
        kh = k_valid.permute(0, 2, 1, 3).contiguous()
        vh = v_valid.permute(0, 2, 1, 3).contiguous()
        attended = torch.nn.functional.scaled_dot_product_attention(
            qh, kh, vh, attn_mask=None, dropout_p=0.0, is_causal=False
        ).permute(0, 2, 1, 3).contiguous()
        result_valid = attended.reshape(attended.shape[0], attended.shape[1], -1)
        if precision == "bf16-boundaries":
            result_valid = result_valid.to(torch.bfloat16).to(torch.float32)
        if attn_mask is None:
            result = result_valid
        else:
            result = torch.zeros(
                (int(q.shape[0]), full_tokens, int(result_valid.shape[-1])),
                dtype=torch.float32, device=q.device,
            ).index_copy(1, indices, result_valid)
        return result

    attention_module.attention = attention  # type: ignore[attr-defined]
    sys.modules[module_name] = attention_module
    parent = sys.modules.get("hymm.models.dit.modules")
    if parent is not None:
        setattr(parent, "attention", attention_module)
    return attention_module, attention


def _install_refiner_hooks(torch: Any, model: Any, capture_state: dict[str, Any]) -> list[Any]:
    hooks = []
    target = capture_state["current"]
    hooks.extend([
        _capture_input(model.c_embedder, target, "masked_mean"),
        _capture_module(model.t_embedder.mlp[0], target, "refiner_timestep_mlp1"),
        _capture_module(model.t_embedder.mlp[1], target, "refiner_timestep_silu"),
        _capture_module(model.t_embedder.mlp[2], target, "refiner_timestep"),
        _capture_module(model.c_embedder.linear_1, target, "context_linear1"),
        _capture_module(model.c_embedder.act_1, target, "context_silu"),
        _capture_module(model.c_embedder.linear_2, target, "context_embed"),
        _capture_module(model.input_embedder, target, "input_embed"),
        _capture_input(model.individual_token_refiner, target, "c", input_index=1),
        _capture_module(model.individual_token_refiner, target, "projected_text"),
    ])
    for index, block in enumerate(model.individual_token_refiner.blocks):
        prefix = f"block{index}"
        hooks.extend([
            _capture_module(block.adaLN_modulation[0], target, f"{prefix}.condition_silu"),
            _capture_module(block.adaLN_modulation[1], target, f"{prefix}.adaLN"),
            _capture_module(
                block.adaLN_modulation,
                target,
                f"{prefix}.gate_msa",
                lambda value: value.chunk(2, dim=-1)[0],
            ),
            _capture_module(
                block.adaLN_modulation,
                target,
                f"{prefix}.gate_mlp",
                lambda value: value.chunk(2, dim=-1)[1],
            ),
            _capture_module(block.norm1, target, f"{prefix}.norm1"),
            _capture_module(block.self_attn_qkv, target, f"{prefix}.qkv"),
            _capture_module(
                block.self_attn_q_norm,
                target,
                f"{prefix}.q",
                lambda value: value.reshape(value.shape[0], value.shape[1], -1),
            ),
            _capture_module(
                block.self_attn_k_norm,
                target,
                f"{prefix}.k",
                lambda value: value.reshape(value.shape[0], value.shape[1], -1),
            ),
            _capture_module(block.self_attn_proj, target, f"{prefix}.proj"),
            _capture_input(block.norm2, target, f"{prefix}.residual1"),
            _capture_module(block.norm2, target, f"{prefix}.norm2"),
            _capture_module(block.mlp.fc1, target, f"{prefix}.mlp_fc1"),
            _capture_module(block.mlp.act, target, f"{prefix}.mlp_silu"),
            _capture_module(block.mlp.fc2, target, f"{prefix}.mlp_fc2"),
            _capture_module(block, target, f"{prefix}.output"),
        ])
    return hooks


def _capture_refiner_attention_outputs(capture_state: dict[str, Any], attention: Any,
                                      torch: Any) -> Any:
    """The refiner adapter records its output at each call, in block order."""
    original = attention

    def wrapped(q: Any, k: Any, v: Any, drop_rate: float = 0,
                attn_mask: Any = None, causal: bool = False) -> Any:
        result = original(q, k, v, drop_rate, attn_mask, causal)
        index = int(capture_state.get("attention_block", 0))
        capture_state["current"][f"block{index}.attention"] = result.detach().clone()
        capture_state["attention_block"] = index + 1
        return result

    return wrapped


def _capture_refiner_qkv_parts(model: Any, capture_state: dict[str, Any], torch: Any) -> list[Any]:
    hooks = []
    for index, block in enumerate(model.individual_token_refiner.blocks):
        prefix = f"block{index}"

        def save_v(_module: Any, _inputs: tuple[Any, ...], output: Any,
                   stage_prefix: str = prefix, heads_num: int = block.heads_num) -> None:
            value = output.chunk(3, dim=-1)[2].reshape(
                output.shape[0], output.shape[1], heads_num,
                output.shape[-1] // 3 // heads_num,
            )
            capture_state["current"][f"{stage_prefix}.v"] = value.reshape(
                value.shape[0], value.shape[1], -1
            ).detach().clone()
        hooks.append(block.self_attn_qkv.register_forward_hook(save_v))
    return hooks


def _full_expected_from_capture(torch: Any, capture: dict[str, Any], valid_indices: list[int]) -> dict[str, Any]:
    expected: dict[str, Any] = {}
    indices = torch.tensor(valid_indices, device=next(iter(capture.values())).device, dtype=torch.int64)
    for name, value in capture.items():
        if name == "attention_adapter.query":
            continue
        if value.ndim >= 3 and int(value.shape[1]) == 1280:
            value = value.index_select(1, indices)
        expected[name] = value.detach().clone()
    return expected


def _compare_compaction(torch: Any, full: dict[str, Any], compact: dict[str, Any],
                        valid_indices: list[int]) -> tuple[dict[str, dict[str, Any]], list[Any]]:
    if set(full) != set(compact):
        raise ReferenceError(
            f"full/compact checkpoint inventory mismatch: missing={sorted(set(full)-set(compact))}, extra={sorted(set(compact)-set(full))}"
        )
    comparisons: dict[str, dict[str, Any]] = {}
    metrics: list[Any] = []
    indices: list[Any] = []
    for name in sorted(full):
        left = full[name]
        right = compact[name]
        if left.ndim >= 3 and int(left.shape[1]) == 1280:
            index = torch.tensor(valid_indices, device=left.device, dtype=torch.int64)
            left = left.index_select(1, index)
        if tuple(left.shape) != tuple(right.shape):
            raise ReferenceError(f"full/compact checkpoint shape mismatch at {name}: {tuple(left.shape)} != {tuple(right.shape)}")
        delta = (left.float() - right.float()).abs().reshape(-1)
        allowed = ATOL + RTOL * right.float().abs().reshape(-1)
        left_flat = left.float().reshape(-1)
        right_flat = right.float().reshape(-1)
        nonfinite = ~torch.isfinite(left_flat) | ~torch.isfinite(right_flat)
        failed = nonfinite | (delta > allowed)
        fail_count = failed.to(torch.int32).sum().to(torch.float32)
        report_delta = torch.where(
            torch.isfinite(delta), delta, torch.full_like(delta, float("inf"))
        )
        max_abs = report_delta.max()
        first_candidate = torch.where(
            failed,
            torch.arange(delta.numel(), device=delta.device, dtype=torch.int64),
            torch.full((delta.numel(),), delta.numel(), device=delta.device, dtype=torch.int64),
        ).min()
        first = torch.where(
            fail_count > 0,
            first_candidate,
            torch.full((), -1, device=delta.device, dtype=torch.int64),
        )
        first_i = first.clamp(min=0).to(torch.int64)
        observed = torch.where(first >= 0, left_flat[first_i], torch.zeros((), device=left.device))
        expected = torch.where(first >= 0, right_flat[first_i], torch.zeros((), device=right.device))
        metrics.append(torch.stack((max_abs, fail_count, first.to(torch.float32), observed, expected)))
        comparisons[name] = {"shape": list(right.shape)}
    return comparisons, metrics


def _insert_record(tensors: dict[str, Any], name: str, value: Any) -> None:
    if name in tensors:
        raise ReferenceError(f"duplicate tensor record {name}")
    if value.dtype != __import__("torch").float32:
        value = value.to(dtype=__import__("torch").float32)
    if not value.is_contiguous():
        value = value.contiguous()
    tensors[name] = value.detach()


def _serialize_tensor_records(torch: Any, records: dict[str, Any],
                              extra_tensors: list[Any]) -> tuple[bytes, dict[str, Any], list[float]]:
    names = sorted(records)
    flat: list[Any] = []
    finite_checks: list[Any] = []
    metadata: dict[str, Any] = {}
    offset = 0
    for name in names:
        tensor = records[name]
        if tensor.device.type != "mps" or tensor.dtype != torch.float32:
            raise ReferenceError(f"non-MPS F32 checkpoint: {name}")
        tensor = tensor.contiguous()
        size = int(tensor.numel()) * 4
        metadata[name] = {
            "dtype": "F32",
            "shape": list(tensor.shape),
            "offset_bytes": offset,
            "nbytes": size,
        }
        offset += size
        flat.append(tensor.reshape(-1))
        finite_checks.append(torch.isfinite(tensor).all().to(dtype=torch.float32).reshape(1))
    if not flat or offset > SIDE_CAR_LIMIT:
        raise ReferenceError("oracle checkpoint sidecar exceeds the 64 MiB bound")
    fixture_float_count = offset // 4
    all_parts = flat + finite_checks + [value.reshape(-1).to(dtype=torch.float32) for value in extra_tensors]
    packed = torch.cat(all_parts).to(device="cpu", dtype=torch.float32).contiguous()
    packed_array = packed.numpy()
    finite_start = fixture_float_count
    finite_end = finite_start + len(finite_checks)
    if not bool((packed_array[finite_start:finite_end] == 1.0).all()):
        raise ReferenceError("nonfinite tensor value in the final packed MPS readback")
    extra_values_start = finite_end
    packed_metrics = packed_array[extra_values_start:].tolist()
    raw = packed_array[:fixture_float_count].astype("<f4", copy=False).tobytes()
    if len(raw) != offset:
        raise ReferenceError("packed fixture byte extent mismatch")
    # The single CPU copy above is the only activation/checkpoint readback.
    return raw, metadata, packed_metrics


def _comparison_reports(comparisons: dict[str, dict[str, Any]],
                        packed_metrics: list[float]) -> list[dict[str, Any]]:
    if len(packed_metrics) != len(comparisons) * 5:
        raise ReferenceError("compaction metric inventory does not match the comparison set")
    reports = []
    for index, (name, shape) in enumerate(comparisons.items()):
        values = packed_metrics[index * 5 : index * 5 + 5]
        max_abs, failures_f, first_f, observed, expected_value = values
        failures = int(failures_f)
        first = int(first_f)
        shape_values = shape["shape"]
        mismatch = None
        if first >= 0:
            if first >= math.prod(shape_values):
                raise ReferenceError(f"first mismatch index is outside checkpoint {name}")
            remainder = first
            coordinate = [0] * len(shape_values)
            for axis in range(len(shape_values) - 1, -1, -1):
                coordinate[axis] = remainder % shape_values[axis]
                remainder //= shape_values[axis]
            mismatch = {
                "flat_index": first,
                "coordinate": coordinate,
                "actual_full": observed if math.isfinite(observed) else None,
                "expected_compact": expected_value if math.isfinite(expected_value) else None,
            }
        reports.append({
            "name": name,
            "shape": shape_values,
            "elements": math.prod(shape_values),
            "failures": failures,
            "max_abs": max_abs if math.isfinite(max_abs) else None,
            "nonfinite": not math.isfinite(max_abs) or (
                mismatch is not None and (
                    mismatch["actual_full"] is None or mismatch["expected_compact"] is None
                )
            ),
            "first_mismatch": mismatch,
        })
    return reports


def _publish_noreplace(path: Path, data: bytes) -> None:
    path = path.resolve()
    if not path.parent.is_dir():
        raise FileNotFoundError(f"output parent must already exist: {path.parent}")
    if path.exists() or path.is_symlink():
        raise FileExistsError(f"oracle output already exists: {path}")
    with tempfile.NamedTemporaryFile(dir=path.parent, prefix=f".{path.name}.", delete=False) as stream:
        temporary = Path(stream.name)
        stream.write(data)
        stream.flush()
        os.fsync(stream.fileno())
    try:
        os.link(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _fixture_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path, help="new fixture JSON path in an existing scratch directory")
    parser.add_argument("--text-weights", type=Path, default=DEFAULT_TEXT_WEIGHTS)
    parser.add_argument("--text-weights-sha256", required=True)
    parser.add_argument("--text-manifest", type=Path, default=DEFAULT_TEXT_MANIFEST)
    parser.add_argument("--text-manifest-sha256", required=True)
    parser.add_argument("--block-weights", type=Path, default=DEFAULT_BLOCK_WEIGHTS)
    parser.add_argument("--block-weights-sha256", required=True)
    parser.add_argument("--block-manifest", type=Path, default=DEFAULT_BLOCK_MANIFEST)
    parser.add_argument("--block-manifest-sha256", required=True)
    parser.add_argument("--qwen-reference", type=Path, default=DEFAULT_QWEN_REFERENCE)
    parser.add_argument("--qwen-sidecar", type=Path, default=DEFAULT_QWEN_SIDECAR)
    parser.add_argument("--upstream", type=Path, default=DEFAULT_UPSTREAM)
    parser.add_argument("--attention-precision", choices=("f32", "bf16-boundaries"), default="f32")
    return parser


def produce_fixture(args: argparse.Namespace) -> dict[str, Any]:
    """Run the two bounded MPS controls and publish a fixture on success."""
    if os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK") not in (None, "0"):
        raise ReferenceError("PYTORCH_ENABLE_MPS_FALLBACK must be unset or 0")
    os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "0"
    output = args.output.resolve()
    if output.suffix.lower() != ".json" or output.exists() or output.is_symlink() or not output.parent.is_dir():
        raise ReferenceError("--output must be a new .json file in an existing directory")
    failure_report_path = output.with_name(f"{output.stem}.compaction-failure.json")
    if failure_report_path.exists() or failure_report_path.is_symlink():
        raise ReferenceError(f"compaction failure report already exists: {failure_report_path}")

    text_path = args.text_weights.resolve(strict=True)
    block_path = args.block_weights.resolve(strict=True)
    for label, path in (("text weights", text_path), ("block weights", block_path)):
        if not path.is_file() or path.is_symlink() or path.suffix != ".safetensors":
            raise ReferenceError(f"{label} must be a regular safetensors file")
    text_sha = _digest_arg(args.text_weights_sha256, "text weights digest")
    block_sha = _digest_arg(args.block_weights_sha256, "block weights digest")
    if sha256_file(text_path) != text_sha or text_sha != TEXT_WEIGHTS_SHA256:
        raise ReferenceError("text weights digest mismatch or not the pinned extracted subset")
    if sha256_file(block_path) != block_sha or block_sha != BLOCK_WEIGHTS_SHA256:
        raise ReferenceError("block weights digest mismatch or not the pinned distilled block-0 subset")
    text_manifest = _load_trusted_manifest(args.text_manifest, args.text_manifest_sha256, kind="text")
    block_manifest = _load_trusted_manifest(args.block_manifest, args.block_manifest_sha256, kind="block")
    _verify_text_manifest(text_manifest, text_path, text_sha)
    _verify_block_manifest(block_manifest, block_path, block_sha)

    qwen_reference_path = args.qwen_reference.resolve(strict=True)
    qwen_sidecar_path = args.qwen_sidecar.resolve(strict=True)
    qwen_reference_sha = sha256_file(qwen_reference_path)
    qwen_sidecar_sha = sha256_file(qwen_sidecar_path)
    if qwen_reference_sha != QWEN_REFERENCE_SHA256 or qwen_sidecar_sha != QWEN_SIDECAR_SHA256:
        raise ReferenceError("Qwen source hash mismatch")
    try:
        qwen_reference = json.loads(qwen_reference_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, RecursionError):
        raise ReferenceError("Qwen reference is invalid JSON") from None
    valid_indices = validate_qwen_reference(
        qwen_reference,
        reference_sha256=qwen_reference_sha,
        sidecar_bytes=qwen_sidecar_path.stat().st_size,
        sidecar_sha256=qwen_sidecar_sha,
    )

    from hy_nomas_reference import (
        REAL_WIDTH,
        _capture_hooks,
        _extract_pinned_block,
        _install_upstream_modules,
        _load_checkpoint_state,
        _make_case_inputs,
        _round_parameters_bf16_in_place,
        _verify_upstream,
    )

    upstream = args.upstream.resolve(strict=True)
    source_hashes = _verify_upstream(upstream)
    for relative, expected in PINNED_SOURCE_SHA256.items():
        path = upstream / relative
        if not path.is_file() or sha256_file(path) != expected:
            raise ReferenceError(f"pinned source digest mismatch: {relative}")
        source_hashes[relative] = expected

    import numpy as np
    import torch
    from safetensors import safe_open

    if not torch.backends.mps.is_built() or not torch.backends.mps.is_available():
        raise ReferenceError("MPS is required; CPU fallback is forbidden")
    device = torch.device("mps")
    if torch.get_default_dtype() != torch.float32:
        torch.set_default_dtype(torch.float32)

    # Read only selected state 26 and the 1280-row crop from the existing BF16
    # sidecar. uint16 reinterpretation preserves the BF16 source bits exactly.
    qwen_map = np.memmap(
        qwen_sidecar_path, mode="r", dtype="<u2", shape=(27, 1, 1388, 3584)
    )
    raw_rows = np.array(qwen_map[26, 0, 108:1388], copy=True)
    del qwen_map
    qwen_bf16_cpu = torch.from_numpy(raw_rows).view(torch.bfloat16)
    del raw_rows
    qwen_full = qwen_bf16_cpu.unsqueeze(0).to(device=device, dtype=torch.float32)
    del qwen_bf16_cpu
    valid_set = set(valid_indices)
    mask_values = [1 if index in valid_set else 0 for index in range(1280)]
    full_mask = torch.tensor([mask_values], dtype=torch.bool, device=device)
    compact_indices = torch.tensor(valid_indices, dtype=torch.int64, device=device)
    compact_input = qwen_full.index_select(1, compact_indices)
    compact_mask = torch.ones((1, len(valid_indices)), dtype=torch.bool, device=device)
    if tuple(qwen_full.shape) != (1, 1280, 3584) or tuple(compact_input.shape) != (1, 23, 3584):
        raise ReferenceError("Qwen MPS input geometry mismatch")

    helpers = _install_upstream_modules(upstream)
    # Install the exact pinned refiner module but replace only the CUDA-only
    # attention dispatch with the bounded MPS adapter below.
    attention_state: dict[str, Any] = {"current": None, "attention_block": 0}
    attention_module, refiner_attention = _install_text_attention(
        torch, valid_indices=valid_indices, precision=args.attention_precision,
        capture=attention_state,
    )
    refiner_attention_wrapped = _capture_refiner_attention_outputs(attention_state, refiner_attention, torch)
    attention_module.attention = refiner_attention_wrapped  # type: ignore[attr-defined]
    text_dir = upstream / "hymm/models/text_encoder"
    from hy_nomas_reference import _namespace
    _namespace("hymm.models.text_encoder", text_dir)
    text_module = importlib.import_module("hymm.models.text_encoder.token_refiner")
    embed_layers = importlib.import_module("hymm.models.dit.modules.embed_layers")

    # The exact upstream refiner is constructed on meta and populated from the
    # exact 38 txt_in tensors. A separate official TimestepEmbedder uses the 4
    # time_in tensors for the DiT's shared vec/vec_txt modulation.
    text_state = _text_state(torch, text_path)
    time_state = _time_state(torch, text_path)
    refiner = text_module.SingleTokenRefinerPerBlock(
        in_channels=3584, hidden_size=2048, heads_num=16, depth=2,
        mlp_width_ratio=4.0, mlp_drop_rate=0.0, act_type="silu",
        qk_norm=False, qkv_bias=True, attn_mode="torch", seg_pos_emb_num=0,
        dtype=torch.float32, device=torch.device("meta"),
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

    timestep = torch.tensor([500.0], dtype=torch.float32, device=device)
    refiner_capture_state: dict[str, Any] = {"current": {}, "attention_block": 0}
    refiner_hooks = _install_refiner_hooks(torch, refiner, refiner_capture_state)
    refiner_hooks.extend(_capture_refiner_qkv_parts(refiner, refiner_capture_state, torch))
    original_timestep_embedding = embed_layers.timestep_embedding
    sinusoidal_target: dict[str, Any] = {"key": None, "current": None}

    def capture_timestep_embedding(t: Any, *args: Any, **kwargs: Any) -> Any:
        result = original_timestep_embedding(t, *args, **kwargs)
        key = sinusoidal_target.get("key")
        if key is not None:
            sinusoidal_target["current"][key] = result.detach().clone()
        return result

    embed_layers.timestep_embedding = capture_timestep_embedding
    time_capture_state: dict[str, Any] = {}
    time_hooks = [
        _capture_module(time_in.mlp[0], time_capture_state, "time_in.mlp1"),
        _capture_module(time_in.mlp[1], time_capture_state, "time_in.silu"),
    ]
    try:
        with torch.inference_mode():
            sinusoidal_target.update(key="time_in.sinusoidal", current={})
            vec = time_in(timestep)
            time_capture = dict(sinusoidal_target["current"])
            for hook in time_hooks:
                hook.remove()
            time_hooks.clear()
            sinusoidal_target.update(key="refiner_timestep_sinusoidal", current={})
            refiner_capture_state["current"].clear()
            attention_state["current"] = refiner_capture_state["current"]
            attention_state["attention_block"] = 0
            full_output = refiner(qwen_full, timestep, full_mask)
            full_capture = dict(refiner_capture_state["current"])
            full_capture.update(sinusoidal_target["current"])

            sinusoidal_target.update(key="refiner_timestep_sinusoidal", current={})
            refiner_capture_state["current"].clear()
            attention_state["current"] = refiner_capture_state["current"]
            attention_state["attention_block"] = 0
            compact_output = refiner(compact_input, timestep, compact_mask)
            compact_capture = dict(refiner_capture_state["current"])
            compact_capture.update(sinusoidal_target["current"])
    finally:
        embed_layers.timestep_embedding = original_timestep_embedding
        for hook in time_hooks:
            hook.remove()
        for hook in refiner_hooks:
            hook.remove()

    # The full-width capture is the source oracle. Compare every full checkpoint
    # after selecting valid rows against the compact execution before admission.
    full_valid = _full_expected_from_capture(torch, full_capture, valid_indices)
    comparisons, comparison_metrics = _compare_compaction(torch, full_capture, compact_capture, valid_indices)
    expected_refiner = {
        "input_compact": compact_input,
        "masked_mean": full_capture["masked_mean"],
        "refiner_timestep_sinusoidal": full_capture["refiner_timestep_sinusoidal"],
        "refiner_timestep_mlp1": full_capture["refiner_timestep_mlp1"],
        "refiner_timestep_silu": full_capture["refiner_timestep_silu"],
        "refiner_timestep": full_capture["refiner_timestep"],
        "context_linear1": full_capture["context_linear1"],
        "context_silu": full_capture["context_silu"],
        "context_embed": full_capture["context_embed"],
        "c": full_capture["c"],
        "input_embed": full_valid["input_embed"],
        "projected_text": full_valid["projected_text"],
        "time_in.sinusoidal": time_capture["time_in.sinusoidal"],
        "time_in.mlp1": time_capture_state["time_in.mlp1"],
        "time_in.silu": time_capture_state["time_in.silu"],
        "vec": vec,
    }
    # Preserve the alias the native plan exposes for the shared 256-wide sine
    # vector used by both timestep embedders.
    expected_refiner["time_in.sinusoidal"] = time_capture["time_in.sinusoidal"]
    expected_refiner["timestep"] = timestep
    for name, value in full_valid.items():
        if name.startswith("block0.") or name.startswith("block1."):
            expected_refiner[name] = value
    for index in (0, 1):
        expected_refiner[f"block{index}.residual2"] = expected_refiner[f"block{index}.output"]

    # Keep only valid-row source checkpoints and the inputs needed by the
    # bounded DiT call. This releases the padded trace and refiner parameters
    # before the 604 MB block subset is loaded.
    refiner_capture_state["current"].clear()
    attention_state["current"] = None
    del full_capture, compact_capture, full_output, compact_output
    del refiner, time_in, qwen_full, full_mask, compact_mask, compact_indices
    del refiner_capture_state, attention_state
    if hasattr(torch, "mps") and hasattr(torch.mps, "empty_cache"):
        torch.mps.empty_cache()

    # Build the pinned real-width DiT block, load its actual 32 F32 weights, and
    # use the actual text output plus the shared actual time_in vector.
    from hy_nomas_reference import (
        REAL_WIDTH,
        _build_sdpa_adapter,
        _capture_hooks,
        _extract_pinned_block,
        _make_case_inputs,
    )
    block_capture_state: dict[str, Any] = {"current": {}, "attention_replay": {}}
    block_receipt: dict[str, Any] = {"calls": 0, "attention_precision": args.attention_precision}
    block_adapter = _build_sdpa_adapter(torch, block_capture_state, block_receipt, args.attention_precision)
    posemb_layers = helpers[-1]

    def apply_rotary_capture(q: Any, k: Any, freqs_cis: Any, head_first: bool = False) -> tuple[Any, Any]:
        q_rot, k_rot = posemb_layers.apply_rotary_emb(q, k, freqs_cis, head_first=head_first)
        current = block_capture_state["current"]
        current["img.attn.q_rope"] = q_rot.detach().clone()
        current["img.attn.k_rope"] = k_rot.detach().clone()
        return q_rot, k_rot

    block_type = _extract_pinned_block(upstream, torch, block_adapter, apply_rotary_capture, helpers)
    block = block_type(
        hidden_size=2048, heads_num=16, mlp_width_ratio=4.0,
        mlp_act_type="gelu_tanh", attn_mode="magi_mp_causal",
        qk_norm=True, qk_norm_type="rms", qkv_bias=True,
        dtype=torch.float32, device=torch.device("meta"),
    )
    block_state, block_source_dtype = _load_checkpoint_state(torch, block_path, dict(REAL_WIDTH))
    if block_source_dtype != "F32":
        raise ReferenceError("DiT source block subset must be F32 before BF16 rounding")
    block.load_state_dict(block_state, strict=True, assign=True)
    block.to(device=device, dtype=torch.float32).eval()
    del block_state
    _round_parameters_bf16_in_place(torch, block, device)
    block_hooks = _capture_hooks(block, block_capture_state, heads=16, head_dim=128)
    case_inputs_cpu, _ = _make_case_inputs(torch, posemb_layers, torch.device("cpu"), "broadcast", dict(REAL_WIDTH))
    # The shared broadcast fixture exercises B=2. This slice admits exactly
    # one prompt and uses its first deterministic video input only.
    video = case_inputs_cpu["img"][:1].contiguous().to(device=device, dtype=torch.float32)
    rope_cos = case_inputs_cpu["video_cos"].to(device=device, dtype=torch.float32)
    rope_sin = case_inputs_cpu["video_sin"].to(device=device, dtype=torch.float32)
    if tuple(video.shape) != (1, 6, 2048) or tuple(vec.shape) != (1, 2048):
        raise ReferenceError("DiT handoff input geometry differs from admitted single-prompt slice")
    # Upstream scalar-time path passes one shared [1,2048] `vec` as both block
    # streams' modulation. The one segment has no per-token text-time expansion.
    block_capture_state["current"] = {}
    block_capture_state["attention_replay"] = {}
    attn_param = {
        "num_prompts": 1,
        "txt_tokens_per_block": 23,
        "prompt_video_boundaries_tokens": [0, 6],
        "prompt_video_boundaries": [0, 6],
        "chunk_size_tokens": 0,
        "seg_ar_attention_backend": "magi",
    }
    with torch.inference_mode():
        img_output, txt_output = block(
            img=video, txt=full_valid["projected_text"], vec=vec, vec_txt=vec,
            freqs_cis=(rope_cos, rope_sin), text_mask=None, attn_param=attn_param,
            is_flash=False, txt_tokens_per_block=23, prompt_video_boundaries=[0, 6],
        )
    if block_receipt["calls"] != 1:
        raise ReferenceError("DiT block adapter did not execute exactly once")
    block_capture_state["current"]["img.output"] = img_output.detach().clone()
    block_capture_state["current"]["txt.output"] = txt_output.detach().clone()
    if len(block_capture_state["current"]) != 40:
        raise ReferenceError(f"DiT block checkpoint count mismatch: {len(block_capture_state['current'])}")
    for hook in block_hooks:
        hook.remove()

    records: dict[str, Any] = {}
    for name, value in expected_refiner.items():
        _insert_record(records, f"refiner.{name}", value)
    _insert_record(records, "input.video", video)
    _insert_record(records, "input.rope_cos", rope_cos)
    _insert_record(records, "input.rope_sin", rope_sin)
    for name, value in block_capture_state["current"].items():
        _insert_record(records, f"dit.{name}", value)

    # Add aliases the native accessor exposes. They intentionally duplicate
    # values in the serialized stream so every named public checkpoint has a
    # contiguous, independently addressable record.
    # Compare metrics and all required checkpoints in one packed GPU readback.
    extra_metrics = [metric.reshape(-1) for metric in comparison_metrics]
    raw_tensor_bytes, tensor_metadata, packed_metrics = _serialize_tensor_records(torch, records, extra_metrics)
    comparisons_out = _comparison_reports(comparisons, packed_metrics)
    compaction_passed = all(item["failures"] == 0 for item in comparisons_out)
    if not compaction_passed:
        failure_document = {
            "schema": "hy-nomas-text-conditioning-compaction-failure-v1",
            "schema_version": 1,
            "upstream_revision": PINNED_REVISION,
            "producer": {
                "path": str(Path(__file__).resolve()),
                "source_sha256": sha256_file(Path(__file__).resolve()),
                "argv": list(sys.argv),
                "python": sys.version.split()[0],
                "torch": torch.__version__,
                "numpy": np.__version__,
                "device": "mps",
                "activation_readback": "one packed final F32 tensor sidecar readback",
            },
            "source": {
                "text_checkpoint": TEXT_SOURCE,
                "text_manifest_sha256": args.text_manifest_sha256,
                "text_manifest_source_file": str(args.text_manifest.resolve()),
                "text_payload_sha256": text_manifest["source"]["payload_sha256_observed"],
                "block_checkpoint": {
                    "file_sha256": block_sha,
                    "file_bytes": block_path.stat().st_size,
                    "manifest_sha256": args.block_manifest_sha256,
                    "manifest_source_file": str(args.block_manifest.resolve()),
                    "source_revision": block_manifest["source"]["revision"],
                    "source_payload_sha256": block_manifest["source"]["payload_sha256_observed"],
                },
                "upstream_source_sha256": source_hashes,
            },
            "weights": {
                "text": {"path": str(text_path), "nbytes": text_path.stat().st_size,
                         "sha256": text_sha, "manifest_sha256": args.text_manifest_sha256},
                "block": {"path": str(block_path), "nbytes": block_path.stat().st_size,
                          "sha256": block_sha, "manifest_sha256": args.block_manifest_sha256},
            },
            "qwen_source": QWEN_SOURCE,
            "mask": mask_values,
            "timestep": 500.0,
            "geometry": GEOMETRY,
            "precision": {
                "resident_weights": "BF16",
                "compute": "F32",
                "attention": args.attention_precision,
                "weight_control": "source F32 parameters rounded through BF16 then represented in F32 for MPS compute",
                "cuda_bf16_parity": False,
            },
            "tolerance": {"atol": ATOL, "rtol": RTOL},
            "compaction": {"passed": False, "comparisons": comparisons_out},
            "failure": {
                "kind": "full masked and compact refiner traces differ",
                "fixture_created": False,
                "tensor_sidecar_published": False,
            },
        }
        _publish_noreplace(
            failure_report_path,
            (json.dumps(failure_document, indent=2, sort_keys=True) + "\n").encode("utf-8"),
        )
        raise ReferenceError(
            "full masked and compact refiner traces differ beyond unchanged atol=rtol=2e-5; "
            f"metrics saved to {failure_report_path}"
        )

    text_manifest_sha = args.text_manifest_sha256
    block_manifest_sha = args.block_manifest_sha256
    tensor_name = f"{output.stem}.tensors.f32le"
    tensor_path = output.parent / tensor_name
    tensor_sha = hashlib.sha256(raw_tensor_bytes).hexdigest()
    document = {
        "schema": SCHEMA,
        "schema_version": 1,
        "upstream_revision": PINNED_REVISION,
        "producer": {
            "path": str(Path(__file__).resolve()),
            "source_sha256": sha256_file(Path(__file__).resolve()),
            "argv": list(sys.argv),
            "python": sys.version.split()[0],
            "torch": torch.__version__,
            "numpy": np.__version__,
            "device": "mps",
            "activation_readback": "one packed final F32 tensor sidecar readback",
        },
        "source": {
            "text_checkpoint": TEXT_SOURCE,
            "text_manifest_sha256": text_manifest_sha,
            "text_manifest_source_file": str(args.text_manifest.resolve()),
            "text_payload_sha256": text_manifest["source"]["payload_sha256_observed"],
            "block_checkpoint": {
                "file_sha256": block_sha,
                "file_bytes": block_path.stat().st_size,
                "manifest_sha256": block_manifest_sha,
                "manifest_source_file": str(args.block_manifest.resolve()),
                "source_revision": block_manifest["source"]["revision"],
                "source_payload_sha256": block_manifest["source"]["payload_sha256_observed"],
            },
            "upstream_source_sha256": source_hashes,
        },
        "weights": {
            "text": {
                "path": str(text_path), "nbytes": text_path.stat().st_size,
                "sha256": text_sha, "manifest_sha256": text_manifest_sha,
            },
            "block": {
                "path": str(block_path), "nbytes": block_path.stat().st_size,
                "sha256": block_sha, "manifest_sha256": block_manifest_sha,
            },
        },
        "qwen_source": QWEN_SOURCE,
        "mask": mask_values,
        "timestep": 500.0,
        "geometry": GEOMETRY,
        "precision": {
            "resident_weights": "BF16",
            "compute": "F32",
            "attention": args.attention_precision,
            "weight_control": "source F32 parameters rounded through BF16 then represented in F32 for MPS compute",
            "cuda_bf16_parity": False,
        },
        "tolerance": {"atol": ATOL, "rtol": RTOL},
        "compaction": {"passed": compaction_passed, "comparisons": comparisons_out},
        "tensor_file": {"name": tensor_name, "nbytes": len(raw_tensor_bytes), "sha256": tensor_sha},
        "tensors": tensor_metadata,
    }
    validate_reference_document(document)
    validate_tensor_layout(tensor_metadata, len(raw_tensor_bytes))
    json_bytes = (json.dumps(document, indent=2, sort_keys=True) + "\n").encode("utf-8")
    # Publish the larger data file first, then the JSON authority. Both are
    # new-only hard-link publishes; on any failure the visible JSON is absent.
    _publish_noreplace(tensor_path, raw_tensor_bytes)
    try:
        _publish_noreplace(output, json_bytes)
    except Exception:
        tensor_path.unlink(missing_ok=True)
        raise
    return {
        "status": "created",
        "output": str(output),
        "tensor_file": str(tensor_path),
        "tensor_file_sha256": tensor_sha,
        "tensor_count": len(tensor_metadata),
        "tensor_bytes": len(raw_tensor_bytes),
        "compaction_passed": compaction_passed,
        "upstream_revision": PINNED_REVISION,
    }


def main(argv: list[str] | None = None) -> int:
    parser = _fixture_parser()
    args = parser.parse_args(argv)
    try:
        result = produce_fixture(args)
    except (OSError, ValueError, ReferenceError, ImportError, RuntimeError) as exc:
        raise SystemExit(str(exc)) from exc
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
