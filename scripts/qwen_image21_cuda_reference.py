#!/usr/bin/env python3
"""Pinned Diffusers oracle and strict fixture comparator for one CUDA Qwen block.

Fixture creation is intentionally separate from execution.  A Crystal probe
exports the exact GGUF block-0 packed bytes and their F32 dequantization, then
runs the CUDA block using the packed bytes.  This script loads the F32 view into
the pinned official Diffusers QwenImage21TransformerBlock, executes it on CUDA,
and records every checkpoint consumed by the Crystal comparator.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import subprocess
import sys
from typing import Any

import numpy as np


SCHEMA = "qwen-image21-cuda-block-fixture"
SCHEMA_VERSION = 1
CANDIDATE_SCHEMA = "qwen-image21-cuda-block-candidate"
REFERENCE_SCHEMA = "qwen-image21-cuda-block-reference"
DIFFUSERS_COMMIT = "8b3c707ebd3ec4881f4190cf42931da07eaf3b65"
GGUF_REPO = "realrebelai/Qwen-Image-2.1_GGUFs"
GGUF_REVISION = "8d393b750593a72ee040fe7d40611f479ccee679"
GGUF_FILENAME = "Qwen-Image-2.1-Q4.gguf"
GGUF_SIZE = 5_959_127_264
GGUF_SHA256 = "51998ad7c068ce7d68e233237537900ffe874ab4d5c72e20758f5f18ceb15b8a"

CHECKPOINTS = (
    "norm1",
    "q_proj",
    "k_proj",
    "v",
    "q_norm",
    "k_norm",
    "q_rope",
    "k_rope",
    "attended",
    "attn_out",
    "state",
    "norm2",
    "gate_up",
    "activated",
    "mlp_out",
    "output",
)
WEIGHT_NAMES = ("to_q", "to_k", "to_v", "to_out", "norm_q", "norm_k", "gate_up", "mlp_out")
INPUT_NAMES = (
    "hidden",
    "modulation",
    "modulation_rows",
    "positions",
    "image_ids",
    "key_valid",
    "target_token_mask",
)

# These are frozen gates from existing repository evidence: the tiny CPU block
# golden uses 2e-5, while the existing real mixed-quant one-block gate uses
# max_abs < 1e-2 and cosine > 0.999999.  The operator checkpoints inherit the
# same strict real-block absolute bound; no command-line tolerance is accepted.
TOLERANCES = {
    "tiny_max_abs": 2.0e-5,
    "real_operator_max_abs": 1.0e-2,
    "real_output_max_abs": 1.0e-2,
    "real_output_cosine_min_exclusive": 0.999999,
}


class FixtureError(ValueError):
    pass


class ParityError(AssertionError):
    pass


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _write_json(path: Path, value: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def _safe_path(root: Path, relative: str) -> Path:
    candidate = root / relative
    resolved_root = root.resolve()
    resolved = candidate.resolve()
    if resolved != resolved_root and resolved_root not in resolved.parents:
        raise FixtureError(f"fixture path escapes its directory: {relative!r}")
    return candidate


def _save_tensor(root: Path, relative: str, values: Any, dtype: str) -> dict[str, Any]:
    if dtype == "float32-le":
        array = np.ascontiguousarray(values, dtype="<f4")
    elif dtype == "int32-le":
        array = np.ascontiguousarray(values, dtype="<i4")
    elif dtype == "uint8":
        array = np.ascontiguousarray(values, dtype=np.uint8)
    else:
        raise FixtureError(f"unsupported fixture dtype {dtype}")
    if dtype == "float32-le" and not bool(np.isfinite(array).all()):
        raise FixtureError(f"non-finite tensor cannot be written: {relative}")
    data = array.tobytes(order="C")
    path = _safe_path(root, relative)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return {
        "file": relative,
        "dtype": dtype,
        "shape": [int(dim) for dim in array.shape],
        "nbytes": len(data),
        "sha256": _sha256(data),
    }


def _read_tensor(root: Path, descriptor: dict[str, Any], name: str) -> np.ndarray:
    required = {"file", "dtype", "shape", "nbytes", "sha256"}
    if not isinstance(descriptor, dict) or set(descriptor) != required:
        raise FixtureError(f"{name} descriptor fields are incomplete or unexpected")
    path = _safe_path(root, descriptor["file"])
    if not path.is_file():
        raise FixtureError(f"missing {name} tensor file: {descriptor['file']}")
    data = path.read_bytes()
    if len(data) != descriptor["nbytes"]:
        raise FixtureError(f"{name} byte count mismatch")
    if _sha256(data) != descriptor["sha256"]:
        raise FixtureError(f"{name} sha256 mismatch")
    dtypes = {"float32-le": np.dtype("<f4"), "int32-le": np.dtype("<i4"), "uint8": np.dtype("u1")}
    dtype = dtypes.get(descriptor["dtype"])
    if dtype is None:
        raise FixtureError(f"{name} has unsupported dtype {descriptor['dtype']!r}")
    shape = descriptor["shape"]
    if not isinstance(shape, list) or any(not isinstance(x, int) or x <= 0 for x in shape):
        raise FixtureError(f"{name} has invalid shape")
    expected_bytes = math.prod(shape) * dtype.itemsize
    if expected_bytes != len(data):
        raise FixtureError(f"{name} shape does not match byte count")
    return np.frombuffer(data, dtype=dtype).reshape(shape).copy()


def _tiny_weights() -> dict[str, tuple[np.ndarray, str]]:
    dim, intermediate = 6, 5

    def matrix(out_dim: int, in_dim: int, phase: int) -> np.ndarray:
        return np.asarray(
            [(((index * 17 + phase * 13) % 29) - 14) / 37.0 for index in range(out_dim * in_dim)],
            dtype=np.float32,
        ).reshape(out_dim, in_dim)

    return {
        "to_q": (matrix(dim, dim, 1), "F32"),
        "to_k": (matrix(dim, dim, 2), "F32"),
        "to_v": (matrix(dim, dim, 3), "F32"),
        "to_out": (matrix(dim, dim, 4), "F32"),
        "norm_q": (np.asarray([1.0, 0.9, 1.1, 0.8, 1.2, 0.95], dtype=np.float32), "F32"),
        "norm_k": (np.asarray([0.85, 1.05, 0.9, 1.15, 0.8, 1.1], dtype=np.float32), "F32"),
        "gate_up": (matrix(2 * intermediate, dim, 5), "F32"),
        "mlp_out": (matrix(dim, intermediate, 6), "F32"),
    }


def _fixture_inputs(kind: str) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    if kind == "tiny_mini_quad":
        dim, heads, head_dim, intermediate = 6, 1, 6, 5
        eps, axes_dims = 1.0e-6, [2, 2, 2]
        tokens = 6
        hidden = np.asarray([(((index * 11) % 23) - 11) / 19.0 for index in range(tokens * dim)], dtype=np.float32).reshape(tokens, dim)
        row_values = [(((index * 7) % 31) - 15) / 101.0 for index in range(2 * 4 * dim)]
        modulation_rows = np.asarray(row_values, dtype=np.float32).reshape(2, 4 * dim)
        positions = np.asarray(
            [[0, 0, 0], [1, 1, 1], [2, -1, -1], [2, -1, 0], [2, 0, -1], [2, 0, 0]],
            dtype=np.int32,
        )
        image_ids = np.asarray([-1, -1, 0, 0, 0, 0], dtype=np.int32)
        key_valid = np.asarray([1, 0, 1, 1, 1, 1], dtype=np.uint8)
        target_mask = np.asarray([0, 0, 1, 1, 1, 1], dtype=np.uint8)
        img_shapes = [[1, 2, 2]]
        kind_name = "tiny_mini_quad"
    elif kind == "real_block0_mini_quad":
        dim, heads, head_dim, intermediate = 4096, 32, 128, 12288
        eps, axes_dims = 1.0e-6, [16, 56, 56]
        tokens = 6
        hidden = np.asarray([(((index * 17 + 11) % 257) - 128) / 193.0 for index in range(tokens * dim)], dtype=np.float32).reshape(tokens, dim)
        row_values = [(((index * 13 + 7) % 101) - 50) / 401.0 for index in range(2 * 4 * dim)]
        modulation_rows = np.asarray(row_values, dtype=np.float32).reshape(2, 4 * dim)
        positions = np.asarray(
            [[0, 0, 0], [1, 1, 1], [2, -1, -1], [2, -1, 0], [2, 0, -1], [2, 0, 0]],
            dtype=np.int32,
        )
        image_ids = np.asarray([-1, -1, 0, 0, 0, 0], dtype=np.int32)
        key_valid = np.asarray([1, 0, 1, 1, 1, 1], dtype=np.uint8)
        target_mask = np.asarray([0, 0, 1, 1, 1, 1], dtype=np.uint8)
        img_shapes = [[1, 2, 2]]
        kind_name = "real_block0_mini_quad"
    else:
        raise FixtureError(f"unknown fixture kind {kind!r}")

    expanded = np.where(target_mask[:, None].astype(bool), modulation_rows[0], modulation_rows[1]).astype(np.float32)
    config = {
        "hidden_dim": dim,
        "heads": heads,
        "head_dim": head_dim,
        "intermediate_dim": intermediate,
        "eps": eps,
        "rope_theta": 10000.0,
        "axes_dims": axes_dims,
        "img_shapes": img_shapes,
        "prefix_length": 2,
    }
    tensors = {
        "hidden": hidden,
        "modulation": expanded,
        "modulation_rows": modulation_rows,
        "positions": positions,
        "image_ids": image_ids,
        "key_valid": key_valid,
        "target_token_mask": target_mask,
    }
    return {"kind": kind_name, "tokens": tokens, "config": config}, tensors


def prepare_fixture(output_dir: str | Path, kind: str) -> Path:
    root = Path(output_dir)
    if root.exists() and any(root.iterdir()):
        raise FileExistsError(f"fixture directory is not empty: {root}")
    root.mkdir(parents=True, exist_ok=True)
    fixture, tensors = _fixture_inputs(kind)
    input_descriptors: dict[str, Any] = {}
    dtypes = {
        "hidden": "float32-le",
        "modulation": "float32-le",
        "modulation_rows": "float32-le",
        "positions": "int32-le",
        "image_ids": "int32-le",
        "key_valid": "uint8",
        "target_token_mask": "uint8",
    }
    for name, values in tensors.items():
        input_descriptors[name] = _save_tensor(root, f"inputs/{name}.bin", values, dtypes[name])

    weights: dict[str, Any] = {}
    if kind == "tiny_mini_quad":
        for name, (values, quant_type) in _tiny_weights().items():
            packed = _save_tensor(root, f"weights/{name}.packed", values, "float32-le")
            dequantized = _save_tensor(root, f"weights/{name}.f32", values, "float32-le")
            weights[name] = {
                "shape": [int(dim) for dim in values.shape],
                "quant_type": quant_type,
                "packed_file": packed["file"],
                "packed_nbytes": packed["nbytes"],
                "packed_sha256": packed["sha256"],
                "dequantized": dequantized,
            }

    manifest: dict[str, Any] = {
        "schema": SCHEMA,
        "schema_version": SCHEMA_VERSION,
        "fixture": fixture,
        "model": None if kind == "tiny_mini_quad" else {
            "repo": GGUF_REPO,
            "revision": GGUF_REVISION,
            "filename": GGUF_FILENAME,
            "size": GGUF_SIZE,
            "sha256": GGUF_SHA256,
            "block_index": 0,
        },
        "inputs": input_descriptors,
        "weights": weights,
        "reference": None,
        "tolerances": TOLERANCES.copy(),
    }
    _write_json(root / "manifest.json", manifest)
    return root / "manifest.json"


def _load_manifest(root: Path) -> dict[str, Any]:
    path = root / "manifest.json"
    if not path.is_file():
        raise FixtureError("missing manifest.json")
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise FixtureError(f"invalid fixture manifest: {exc}") from exc
    if manifest.get("schema") != SCHEMA or manifest.get("schema_version") != SCHEMA_VERSION:
        raise FixtureError("unsupported fixture schema or version")
    if manifest.get("tolerances") != TOLERANCES:
        raise FixtureError("fixture tolerance policy differs from the frozen repository gates")
    fixture = manifest.get("fixture")
    if not isinstance(fixture, dict) or fixture.get("kind") not in {"tiny_mini_quad", "real_block0_mini_quad"}:
        raise FixtureError("unknown or missing fixture kind")
    if fixture.get("tokens") != 6 or fixture.get("config", {}).get("prefix_length") != 2:
        raise FixtureError("fixture must preserve the six-token mini-Quad layout with a two-token prefix")
    if set(manifest.get("inputs", {})) != set(INPUT_NAMES):
        raise FixtureError("fixture input list is incomplete or unexpected")
    if manifest["fixture"]["kind"] == "tiny_mini_quad" and set(manifest.get("weights", {})) != set(WEIGHT_NAMES):
        raise FixtureError("tiny fixture weight list is incomplete or unexpected")
    return manifest


def _validate_inputs(root: Path, manifest: dict[str, Any]) -> dict[str, np.ndarray]:
    inputs = {name: _read_tensor(root, desc, f"input {name}") for name, desc in manifest["inputs"].items()}
    fixture = manifest["fixture"]
    config = fixture["config"]
    tokens = fixture["tokens"]
    dim = config["hidden_dim"]
    expected_shapes = {
        "hidden": [tokens, dim],
        "modulation": [tokens, 4 * dim],
        "modulation_rows": [2, 4 * dim],
        "positions": [tokens, 3],
        "image_ids": [tokens],
        "key_valid": [tokens],
        "target_token_mask": [tokens],
    }
    for name, shape in expected_shapes.items():
        if list(inputs[name].shape) != shape:
            raise FixtureError(f"input {name} has shape {inputs[name].shape}; expected {tuple(shape)}")
    for mask_name in ("key_valid", "target_token_mask"):
        if not bool(np.logical_or(inputs[mask_name] == 0, inputs[mask_name] == 1).all()):
            raise FixtureError(f"input {mask_name} must contain only zero or one")
    if not np.array_equal(inputs["image_ids"] >= 0, inputs["target_token_mask"].astype(bool)):
        raise FixtureError("fixture image ids and target-token mask disagree")
    selected = np.where(inputs["target_token_mask"][:, None].astype(bool), inputs["modulation_rows"][0], inputs["modulation_rows"][1])
    if not np.array_equal(inputs["modulation"], selected):
        raise FixtureError("expanded modulation does not match official two-row target-mask selection")
    if int(inputs["key_valid"].sum()) < 1:
        raise FixtureError("fixture must have at least one valid attention key")
    return inputs


def _torch_commit(diffusers_module: Any, transformer_module: Any) -> str:
    """Require source provenance for the exact upstream pin, rather than trust a version string."""
    candidates = [Path(transformer_module.__file__).resolve(), Path(diffusers_module.__file__).resolve()]
    for file_path in candidates:
        for parent in (file_path.parent, *file_path.parents):
            if (parent / ".git").exists():
                try:
                    result = subprocess.run(
                        ["git", "-C", str(parent), "rev-parse", "HEAD"],
                        check=True,
                        capture_output=True,
                        text=True,
                    )
                except (OSError, subprocess.CalledProcessError):
                    continue
                return result.stdout.strip().lower()
    try:
        direct_url = importlib.metadata.distribution("diffusers").read_text("direct_url.json")
        if direct_url:
            data = json.loads(direct_url)
            return str(data.get("vcs_info", {}).get("commit_id", "")).lower()
    except (importlib.metadata.PackageNotFoundError, json.JSONDecodeError, OSError):
        pass
    raise FixtureError("cannot establish Diffusers source commit; use the pinned Git checkout/install")


def _module_tensor(module: Any, args: tuple[Any, ...], kwargs: dict[str, Any]) -> Any:
    value = args[0] if args else kwargs.get("hidden_states")
    if value is None or not hasattr(value, "detach"):
        raise FixtureError(f"checkpoint hook received an unexpected input for {module}")
    return value.detach()


def _load_weights_into_block(block: Any, root: Path, manifest: dict[str, Any], device: Any) -> None:
    import torch

    weights = manifest.get("weights")
    if not isinstance(weights, dict) or set(weights) != set(WEIGHT_NAMES):
        raise FixtureError("block weight set is incomplete; run the GGUF export probe first")
    config = manifest["fixture"]["config"]
    dim, intermediate = config["hidden_dim"], config["intermediate_dim"]
    expected = {
        "to_q": [dim, dim],
        "to_k": [dim, dim],
        "to_v": [dim, dim],
        "to_out": [dim, dim],
        "norm_q": [config["head_dim"]],
        "norm_k": [config["head_dim"]],
        "gate_up": [2 * intermediate, dim],
        "mlp_out": [dim, intermediate],
    }
    loaded: dict[str, Any] = {}
    for name in WEIGHT_NAMES:
        entry = weights[name]
        descriptor = entry.get("dequantized")
        values = _read_tensor(root, descriptor, f"weight {name}")
        if list(values.shape) != expected[name] or list(entry.get("shape", [])) != expected[name]:
            raise FixtureError(f"weight {name} shape mismatch; expected {expected[name]}")
        if not bool(np.isfinite(values).all()):
            raise FixtureError(f"weight {name} contains non-finite values")
        loaded[name] = torch.from_numpy(np.ascontiguousarray(values, dtype=np.float32)).to(device=device)

    with torch.no_grad():
        for attr, source in (("to_q", "to_q"), ("to_k", "to_k"), ("to_v", "to_v")):
            getattr(block.attn, attr).weight.copy_(loaded[source])
        block.attn.to_out[0].weight.copy_(loaded["to_out"])
        block.attn.norm_q.weight.copy_(loaded["norm_q"])
        block.attn.norm_k.weight.copy_(loaded["norm_k"])
        block.img_mlp.gate_layer.weight.copy_(loaded["gate_up"][:intermediate])
        block.img_mlp.proj.weight.copy_(loaded["gate_up"][intermediate:])
        block.img_mlp.out.weight.copy_(loaded["mlp_out"])


def run_upstream_oracle(fixture_dir: str | Path) -> Path:
    root = Path(fixture_dir)
    manifest = _load_manifest(root)
    inputs = _validate_inputs(root, manifest)
    if manifest["fixture"]["kind"] == "real_block0_mini_quad":
        model = manifest.get("model")
        if not isinstance(model, dict) or model.get("repo") != GGUF_REPO or model.get("revision") != GGUF_REVISION or model.get("sha256") != GGUF_SHA256:
            raise FixtureError("real block fixture is not tied to the pinned GGUF repository revision and file hash")
        if model.get("size") != GGUF_SIZE or model.get("filename") != GGUF_FILENAME or model.get("block_index") != 0:
            raise FixtureError("real block fixture model identity does not match pinned block-0 asset")

    try:
        import torch
        import diffusers
        from diffusers.models.transformers import transformer_qwenimage21 as upstream
    except Exception as exc:
        raise FixtureError(f"pinned PyTorch and Diffusers Qwen-Image 2.1 imports are required: {exc}") from exc

    commit = _torch_commit(diffusers, upstream)
    if commit != DIFFUSERS_COMMIT:
        raise FixtureError(f"Diffusers commit mismatch: expected {DIFFUSERS_COMMIT}, got {commit or 'unknown'}")
    if not torch.cuda.is_available():
        raise FixtureError("upstream oracle requires CUDA; CPU fallback is forbidden")
    device = torch.device("cuda")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    torch.manual_seed(0)

    fixture = manifest["fixture"]
    config = fixture["config"]
    dim, heads, head_dim, intermediate = (
        config["hidden_dim"], config["heads"], config["head_dim"], config["intermediate_dim"]
    )
    block = upstream.QwenImage21TransformerBlock(
        dim=dim,
        num_attention_heads=heads,
        attention_head_dim=head_dim,
        mlp_ratio=max(1, intermediate // dim),
        eps=float(config["eps"]),
    )
    # Tiny synthetic dimensions deliberately retain I=5 while D=6.  Replace
    # only the official feed-forward submodule's linears to preserve that exact
    # fixture shape; its forward equation and activation remain upstream code.
    block.img_mlp = upstream.QwenImage21SwiGLUFeedForward(dim, intermediate)
    block = block.to(device=device, dtype=torch.float32).eval()
    _load_weights_into_block(block, root, manifest, device)

    hidden = torch.from_numpy(inputs["hidden"]).to(device=device, dtype=torch.float32).unsqueeze(0)
    modulation_rows = torch.from_numpy(inputs["modulation_rows"]).to(device=device, dtype=torch.float32)
    positions = torch.from_numpy(inputs["positions"]).to(device=device)
    image_ids = torch.from_numpy(inputs["image_ids"].astype(np.int64)).to(device=device)
    key_valid = torch.from_numpy(inputs["key_valid"].astype(np.bool_)).to(device=device).unsqueeze(0)
    target_mask = torch.from_numpy(inputs["target_token_mask"].astype(np.bool_)).to(device=device)
    image_pad_mask = image_ids >= 0
    rope = upstream.QwenImage21Rope(theta=int(config["rope_theta"]), axes_dim=list(config["axes_dims"])).to(device)
    rotary_emb = rope(
        [tuple(int(x) for x in shape) for shape in config["img_shapes"]],
        image_pad_mask,
        device,
    )
    prefix_length = int(config["prefix_length"])
    prefix_ids = inputs["image_ids"][:prefix_length]
    segments: list[tuple[int, int, bool]] = []
    start = 0
    for index in range(1, prefix_length + 1):
        if index == prefix_length or prefix_ids[index] != prefix_ids[start]:
            segments.append((start, index, bool(prefix_ids[start] < 0)))
            start = index
    if not segments or segments[-1][1] != prefix_length:
        raise FixtureError("prefix segment construction failed")

    captures: dict[str, Any] = {}
    handles = []

    def capture_output(name: str):
        def hook(_module: Any, _args: Any, output: Any) -> None:
            captures[name] = output.detach()
        return hook

    def capture_input(name: str):
        def hook(module: Any, args: tuple[Any, ...], kwargs: dict[str, Any]) -> None:
            captures[name] = _module_tensor(module, args, kwargs)
        return hook

    handles.extend([
        block.attn.register_forward_pre_hook(capture_input("norm1"), with_kwargs=True),
        block.attn.to_q.register_forward_hook(capture_output("q_proj")),
        block.attn.to_k.register_forward_hook(capture_output("k_proj")),
        block.attn.to_v.register_forward_hook(capture_output("v")),
        block.attn.norm_q.register_forward_hook(capture_output("q_norm")),
        block.attn.norm_k.register_forward_hook(capture_output("k_norm")),
        block.attn.to_out[0].register_forward_pre_hook(capture_input("attended"), with_kwargs=True),
        block.attn.register_forward_hook(capture_output("attn_out")),
        block.img_norm2.register_forward_pre_hook(capture_input("state"), with_kwargs=True),
        block.img_mlp.gate_layer.register_forward_pre_hook(capture_input("norm2"), with_kwargs=True),
        block.img_mlp.gate_layer.register_forward_hook(capture_output("gate")),
        block.img_mlp.proj.register_forward_hook(capture_output("up")),
        block.img_mlp.out.register_forward_pre_hook(capture_input("activated"), with_kwargs=True),
        block.img_mlp.out.register_forward_hook(capture_output("mlp_out")),
        block.register_forward_hook(capture_output("output")),
    ])

    original_apply_rope = upstream.apply_rotary_emb_qwen
    rope_calls = 0

    def trace_rope(x: Any, freqs_cis: Any, use_real: bool = True, use_real_unbind_dim: int = -1) -> Any:
        nonlocal rope_calls
        output = original_apply_rope(x, freqs_cis, use_real=use_real, use_real_unbind_dim=use_real_unbind_dim)
        if rope_calls == 0:
            captures["q_rope"] = output.detach()
        elif rope_calls == 1:
            captures["k_rope"] = output.detach()
        else:
            raise FixtureError("unexpected extra upstream RoPE application")
        rope_calls += 1
        return output

    upstream.apply_rotary_emb_qwen = trace_rope
    try:
        with torch.no_grad():
            result = block(
                hidden_states=hidden,
                modulation=modulation_rows,
                rotary_emb=rotary_emb,
                target_token_mask=target_mask,
                segments=segments,
                key_valid=key_valid,
            )
        if not torch.equal(result, captures["output"]):
            raise FixtureError("block output hook did not capture the returned upstream tensor")
        if rope_calls != 2:
            raise FixtureError(f"expected two upstream RoPE applications, observed {rope_calls}")
        if "gate" not in captures or "up" not in captures:
            raise FixtureError("upstream SwiGLU projection hooks did not run")
        captures["gate_up"] = torch.cat((captures.pop("gate"), captures.pop("up")), dim=-1)
    finally:
        upstream.apply_rotary_emb_qwen = original_apply_rope
        for handle in handles:
            handle.remove()

    if set(captures) != set(CHECKPOINTS):
        raise FixtureError(f"oracle checkpoint set mismatch: missing={sorted(set(CHECKPOINTS) - set(captures))}")

    # Mirror the Crystal device-to-device chain guard: the first upstream
    # output becomes the hidden input to a second invocation of the same
    # official block, with identical weights and conditioning.
    with torch.no_grad():
        chain_result = block(
            hidden_states=result,
            modulation=modulation_rows,
            rotary_emb=rotary_emb,
            target_token_mask=target_mask,
            segments=segments,
            key_valid=key_valid,
        )
    if not bool(torch.isfinite(chain_result).all()):
        raise FixtureError("upstream device-chain output contains non-finite values")

    checkpoint_shapes = _checkpoint_shapes(fixture)
    reference_dir = root / "reference"
    if reference_dir.exists():
        raise FileExistsError(f"reference directory already exists: {reference_dir}")
    reference_dir.mkdir(parents=True)
    descriptors: dict[str, Any] = {}
    for name in CHECKPOINTS:
        tensor = captures[name]
        if name in {"q_norm", "k_norm", "q_rope", "k_rope"}:
            tensor = tensor.flatten(2, 3)
        tensor = tensor.reshape(checkpoint_shapes[name]).detach().to(device="cpu", dtype=torch.float32).contiguous()
        if not bool(torch.isfinite(tensor).all()):
            raise FixtureError(f"upstream checkpoint {name} contains non-finite values")
        descriptors[name] = _save_tensor(root, f"reference/{name}.f32", tensor.numpy(), "float32-le")

    chain_tensor = chain_result.reshape(checkpoint_shapes["output"]).detach().to(device="cpu", dtype=torch.float32).contiguous()
    chain_descriptor = _save_tensor(root, "reference/device_chain_output.f32", chain_tensor.numpy(), "float32-le")

    reference = {
        "schema": REFERENCE_SCHEMA,
        "schema_version": 1,
        "source": "official Diffusers QwenImage21TransformerBlock.forward with official QwenImage21Rope/apply_rotary_emb_qwen",
        "diffusers_commit": commit,
        "torch_version": str(torch.__version__),
        "device": torch.cuda.get_device_name(device),
        "dtype": "float32",
        "tf32": False,
        "attention_processor": type(block.attn.processor).__name__,
        "sdpa": {
            "flash_enabled": bool(torch.backends.cuda.flash_sdp_enabled()),
            "mem_efficient_enabled": bool(torch.backends.cuda.mem_efficient_sdp_enabled()),
            "math_enabled": bool(torch.backends.cuda.math_sdp_enabled()),
        },
        "weights_source": "exact GGUF block-0 dequantized F32 files; original packed bytes retained in fixture",
        "checkpoints": descriptors,
        "guards": {
            "device_chain": "same_official_block_second_invocation_with_first_output_as_hidden_states",
            "device_chain_output": chain_descriptor,
        },
    }
    manifest["reference"] = reference
    _write_json(root / "manifest.json", manifest)
    return root / "manifest.json"


def _checkpoint_shapes(fixture: dict[str, Any]) -> dict[str, list[int]]:
    tokens = int(fixture["tokens"])
    dim = int(fixture["config"]["hidden_dim"])
    intermediate = int(fixture["config"]["intermediate_dim"])
    shapes = {name: [tokens, dim] for name in CHECKPOINTS}
    shapes["gate_up"] = [tokens, 2 * intermediate]
    shapes["activated"] = [tokens, intermediate]
    return shapes


def _candidate_manifest(root: Path, candidate_dir: Path, fixture_manifest_bytes: bytes) -> dict[str, Any]:
    path = candidate_dir / "candidate.json"
    if not path.is_file():
        raise FixtureError("missing candidate.json from CUDA block probe")
    try:
        candidate = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise FixtureError(f"invalid candidate manifest: {exc}") from exc
    if candidate.get("schema") != CANDIDATE_SCHEMA or candidate.get("schema_version") != 1:
        raise FixtureError("unsupported CUDA candidate schema or version")
    expected_fixture_sha = _sha256(fixture_manifest_bytes)
    if candidate.get("fixture_manifest_sha256") != expected_fixture_sha:
        raise FixtureError("candidate was produced from a different fixture manifest")
    if set(candidate.get("checkpoints", {})) != set(CHECKPOINTS):
        raise FixtureError("candidate checkpoint set is incomplete or unexpected")
    guards = candidate.get("guards")
    if not isinstance(guards, dict):
        raise FixtureError("candidate device-chain and repeated-input guards are missing")
    if guards.get("device_chain") != "runner_b_input=runner_a.output_device_ptr":
        raise FixtureError("candidate did not chain runner A's device output directly into runner B")
    if guards.get("repeat_input_output_bitwise_equal") is not True:
        raise FixtureError("candidate repeated-input stale-state guard failed")
    return candidate


def compare_fixture(fixture_dir: str | Path, candidate_dir: str | Path) -> dict[str, Any]:
    root = Path(fixture_dir)
    manifest_bytes = (root / "manifest.json").read_bytes()
    manifest = _load_manifest(root)
    if not isinstance(manifest.get("reference"), dict):
        raise FixtureError("fixture has no official upstream reference checkpoints")
    if manifest["reference"].get("schema") != REFERENCE_SCHEMA or manifest["reference"].get("diffusers_commit") != DIFFUSERS_COMMIT:
        raise FixtureError("reference provenance is not the pinned upstream Diffusers commit")
    _validate_inputs(root, manifest)
    expected_shapes = _checkpoint_shapes(manifest["fixture"])
    if set(manifest["reference"].get("checkpoints", {})) != set(CHECKPOINTS):
        raise FixtureError("upstream reference checkpoint set is incomplete or unexpected")
    candidate_root = Path(candidate_dir)
    candidate = _candidate_manifest(root, candidate_root, manifest_bytes)

    tiny = manifest["fixture"]["kind"] == "tiny_mini_quad"
    max_abs_limit = TOLERANCES["tiny_max_abs"] if tiny else None
    results: dict[str, Any] = {}
    for name in CHECKPOINTS:
        shape = expected_shapes[name]
        reference = _read_tensor(root, manifest["reference"]["checkpoints"][name], f"reference {name}")
        actual = _read_tensor(candidate_root, candidate["checkpoints"][name], f"candidate {name}")
        if list(reference.shape) != shape or list(actual.shape) != shape:
            raise FixtureError(f"checkpoint {name} has wrong shape; expected {shape}")
        if not bool(np.isfinite(reference).all()) or not bool(np.isfinite(actual).all()):
            raise FixtureError(f"checkpoint {name} contains non-finite values")
        delta = np.abs(reference.astype(np.float64) - actual.astype(np.float64))
        max_abs = float(delta.max(initial=0.0))
        ref64 = reference.astype(np.float64).ravel()
        got64 = actual.astype(np.float64).ravel()
        denom = float(np.linalg.norm(ref64) * np.linalg.norm(got64))
        cosine = 1.0 if denom == 0.0 and np.array_equal(reference, actual) else (float(np.dot(ref64, got64) / denom) if denom else 0.0)
        if tiny:
            passed = max_abs <= float(max_abs_limit)
            bound = float(max_abs_limit)
        else:
            bound = TOLERANCES["real_output_max_abs"] if name == "output" else TOLERANCES["real_operator_max_abs"]
            passed = max_abs < float(bound)
            if name == "output":
                passed = passed and cosine > TOLERANCES["real_output_cosine_min_exclusive"]
        results[name] = {"max_abs": max_abs, "max_abs_limit": bound, "cosine": cosine, "passed": passed}

    reference_guard = manifest["reference"].get("guards")
    if not isinstance(reference_guard, dict) or reference_guard.get("device_chain") != "same_official_block_second_invocation_with_first_output_as_hidden_states":
        raise FixtureError("pinned upstream device-chain reference is missing")
    chain_reference = _read_tensor(root, reference_guard.get("device_chain_output"), "reference device-chain output")
    chain_actual = _read_tensor(candidate_root, candidate["guards"].get("chain_output"), "candidate device-chain output")
    chain_shape = expected_shapes["output"]
    if list(chain_reference.shape) != chain_shape or list(chain_actual.shape) != chain_shape:
        raise FixtureError(f"device-chain output has wrong shape; expected {chain_shape}")
    if not bool(np.isfinite(chain_reference).all()) or not bool(np.isfinite(chain_actual).all()):
        raise FixtureError("device-chain output contains non-finite values")
    chain_delta = np.abs(chain_reference.astype(np.float64) - chain_actual.astype(np.float64))
    chain_max_abs = float(chain_delta.max(initial=0.0))
    chain_ref64 = chain_reference.astype(np.float64).ravel()
    chain_got64 = chain_actual.astype(np.float64).ravel()
    chain_denom = float(np.linalg.norm(chain_ref64) * np.linalg.norm(chain_got64))
    chain_cosine = 1.0 if chain_denom == 0.0 and np.array_equal(chain_reference, chain_actual) else (
        float(np.dot(chain_ref64, chain_got64) / chain_denom) if chain_denom else 0.0
    )
    if tiny:
        chain_bound = TOLERANCES["tiny_max_abs"]
        chain_passed = chain_max_abs <= chain_bound
    else:
        chain_bound = TOLERANCES["real_output_max_abs"]
        chain_passed = chain_max_abs < chain_bound and chain_cosine > TOLERANCES["real_output_cosine_min_exclusive"]
    results["device_chain_output"] = {
        "max_abs": chain_max_abs,
        "max_abs_limit": chain_bound,
        "cosine": chain_cosine,
        "passed": chain_passed,
    }
    failures = [name for name, row in results.items() if not row["passed"]]
    if failures:
        details = ", ".join(f"{name}: max_abs={results[name]['max_abs']:.9g}" for name in failures)
        raise ParityError(f"CUDA block diverged from pinned upstream at {', '.join(failures)} ({details})")
    return {
        "fixture_kind": manifest["fixture"]["kind"],
        "checkpoint_count": len(CHECKPOINTS),
        "guard_count": 2,
        "max_abs": max(row["max_abs"] for row in results.values()),
        "checks": results,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    prepare = commands.add_parser("prepare-tiny", help="write deterministic mini-Quad synthetic inputs and F32 weights")
    prepare.add_argument("fixture_dir", type=Path)
    real = commands.add_parser("prepare-real", help="write six-token real-shaped block-0 inputs; export GGUF weights with the Crystal probe")
    real.add_argument("fixture_dir", type=Path)
    oracle = commands.add_parser("oracle", help="run pinned official Diffusers block on CUDA and capture checkpoints")
    oracle.add_argument("fixture_dir", type=Path)
    compare = commands.add_parser("compare", help="compare CUDA checkpoint files against frozen strict bounds")
    compare.add_argument("fixture_dir", type=Path)
    compare.add_argument("candidate_dir", type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.command == "prepare-tiny":
            print(prepare_fixture(args.fixture_dir, "tiny_mini_quad"))
        elif args.command == "prepare-real":
            print(prepare_fixture(args.fixture_dir, "real_block0_mini_quad"))
        elif args.command == "oracle":
            print(run_upstream_oracle(args.fixture_dir))
        elif args.command == "compare":
            result = compare_fixture(args.fixture_dir, args.candidate_dir)
            print(json.dumps(result, sort_keys=True))
        else:
            raise FixtureError(f"unknown command {args.command}")
    except (FixtureError, ParityError, FileExistsError, OSError, ValueError) as exc:
        print(f"qwen_image21_cuda_reference: error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
