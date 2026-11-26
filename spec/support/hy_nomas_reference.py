#!/usr/bin/env python3
# Based on Tencent Hy-Nomas at 911cea6458a89c2663513ff7099561ec190dd491.
# Copyright (C) 2026 Tencent. Apache-2.0. Modified for native Metal validation.
"""Create bounded MPS oracle fixtures from the pinned Hy-Nomas block.

By default, the fixture uses a tiny config. ``--real-width`` uses the Nomas-G
hidden/FFN/head geometry with the same few tokens; it does not load real weights
unless a trusted, exact single-block safetensors file is supplied. The fixture
compiles the exact ``MMDoubleStreamBlock`` class AST from the
pinned upstream source, with its actual pinned modulation, normalization, MLP,
activation, and RoPE modules. Only attention dispatch is replaced with an
explicit MPS-only PyTorch SDPA adapter for one unmasked segment. The optional
``bf16-boundaries`` precision rounds F32 concatenated Q/K/V through BF16 and
rounds the F32 SDPA output through BF16 while keeping SDPA itself F32. This is
a boundary control, not BF16 SDPA or Magi/CUDA numerical parity. Neither mode
loads a full model, camera conditioning, history, or media.

The ``bf16_weights_roundtrip_f32_compute`` control uses BF16-rounded weight
values with F32 parameters and computation. Synthetic fixtures store that
control as a sidecar; trusted real weights retain their exact source sidecar
and are rounded one MPS parameter at a time. This is not native BF16 activation
coverage.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import importlib
import importlib.util
import json
import math
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import time
import types
from typing import Any


PINNED_REVISION = "911cea6458a89c2663513ff7099561ec190dd491"
DEFAULT_UPSTREAM = Path(
    "/opt/mldata_ssd/cogni-ml/video-scene-20261019/upstream/hy-nomas-20261024"
)
SEED_WEIGHTS = 20261024
SEED_INPUTS = 20261025
ATTENTION_PRECISIONS = ("f32", "bf16-boundaries")

# Hashes cover the exact block AST, every executed upstream helper, and the
# source that defines the one-segment attention contract. Any drift fails
# closed before upstream code is imported or executed.
PINNED_SOURCE_SHA256 = {
    "hymm/models/dit/nomas_cpt.py": "acf827bb7feb84c9046618737146544843a3f50c3500a3b16925dadeb6c5bdee",
    "hymm/models/dit/modules/activation_layers.py": "1e509b5e5f9527d199d6f0206f452e56fdb1bd25ce3ccc06fb47f3a16060ca39",
    "hymm/models/dit/modules/norm_layers.py": "2bcd5b29c951051ee1afaeac08258a3ef1dea3205b821f236594719331127044",
    "hymm/models/dit/modules/modulate_layers.py": "abfb87acc9a972b40d0f36cfa832f0816d978fc13f93236aa1dfc902aac5caee",
    "hymm/models/dit/modules/mlp_layers.py": "b77d862d17cd2c7db6237931b80f33b6447b7ba4db5918d3fbdfd17a6fb47e07",
    "hymm/models/dit/modules/posemb_layers.py": "27b729d04185dfd7d65ecf9ba0d761d4864153d3e656908997d60a6971afc1aa",
    "hymm/models/dit/modules/attention/__init__.py": "2a59284ac06057ac622228d280a32fc9e907a333e8ef0efcd526928626858450",
    "hymm/models/dit/modules/attention/magi_attn.py": "3cdd9234db6ddd006a04af53b878a6876242ed4fa352bbcfe43038519492eb47",
    "hymm/utils/helpers.py": "51b7a08377843b5eecc59e6fd41f10317d2de096d30fc3154491aacd27fefdfb",
}

TINY = {
    "geometry_label": "tiny_synthetic",
    "batches": 1,
    "heads": 2,
    "head_dim": 32,
    "dim": 64,
    "ffn_dim": 256,
    "video_tokens": 6,
    "text_tokens": 3,
    "img_modulation_rows": 0,
    "txt_modulation_rows": 0,
    "segment_count": 1,
    "chunk_size_tokens": 0,
    "history_tokens": 0,
    "has_mask": False,
    "has_camera": False,
    "action_windows": False,
    "norm_eps": 1e-6,
    "mlp_act_type": "gelu_tanh",
    "qk_norm": True,
    "qk_norm_type": "rms",
    "qkv_bias": True,
    "rope_dim_list": [8, 12, 12],
    "rope_grid": [1, 2, 3],
    "rope_theta": 256.0,
}

REAL_WIDTH = {
    **TINY,
    "geometry_label": "nomas_g_real_width_synthetic_probe",
    "heads": 16,
    "head_dim": 128,
    "dim": 2048,
    "ffn_dim": 8192,
    "rope_dim_list": [16, 56, 56],
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _emit_stage_telemetry(enabled: bool, stage: str, **fields: Any) -> None:
    if not enabled:
        return
    marker = {"stage": stage, "monotonic_ns": time.monotonic_ns(), **fields}
    print(json.dumps(marker, sort_keys=True), file=sys.stderr, flush=True)


def _verify_upstream(root: Path) -> dict[str, str]:
    root = root.resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"Upstream source directory does not exist: {root}")
    try:
        revision = subprocess.check_output(
            ["git", "-C", str(root), "rev-parse", "HEAD"], text=True
        ).strip()
    except (OSError, subprocess.CalledProcessError) as exc:
        raise RuntimeError(f"Could not identify upstream Git revision at {root}") from exc
    if revision != PINNED_REVISION:
        raise RuntimeError(
            f"Pinned upstream revision mismatch: {revision} != {PINNED_REVISION}"
        )

    actual_hashes: dict[str, str] = {}
    for relative, expected in PINNED_SOURCE_SHA256.items():
        path = root / relative
        if not path.is_file():
            raise FileNotFoundError(f"Pinned upstream source is missing: {path}")
        actual = _sha256(path)
        if actual != expected:
            raise RuntimeError(
                f"Pinned source hash mismatch for {relative}: {actual} != {expected}"
            )
        actual_hashes[relative] = actual
    return actual_hashes


def _namespace(name: str, directory: Path) -> None:
    if name in sys.modules:
        module = sys.modules[name]
        module_path = getattr(module, "__path__", None)
        if not module_path or Path(module_path[0]).resolve() != directory.resolve():
            raise RuntimeError(
                f"Refusing conflicting imported package {name}: {module_path}"
            )
        return
    module = types.ModuleType(name)
    module.__path__ = [str(directory)]  # type: ignore[attr-defined]
    module.__package__ = name
    sys.modules[name] = module
    if "." in name:
        parent_name, child_name = name.rsplit(".", 1)
        parent = sys.modules.get(parent_name)
        if parent is not None:
            setattr(parent, child_name, module)


def _load_source_module(name: str, path: Path) -> types.ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load pinned upstream source module: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _install_upstream_modules(root: Path) -> tuple[Any, Any, Any, Any, Any, Any, Any]:
    """Load only direct pinned block dependencies, avoiding full model imports."""
    package_paths = {
        "hymm": root / "hymm",
        "hymm.models": root / "hymm" / "models",
        "hymm.models.dit": root / "hymm" / "models" / "dit",
        "hymm.models.dit.modules": root / "hymm" / "models" / "dit" / "modules",
        "hymm.utils": root / "hymm" / "utils",
    }
    for name, directory in package_paths.items():
        _namespace(name, directory)

    activation_layers = _load_source_module(
        "hymm.models.dit.modules.activation_layers",
        root / "hymm/models/dit/modules/activation_layers.py",
    )
    norm_layers = _load_source_module(
        "hymm.models.dit.modules.norm_layers",
        root / "hymm/models/dit/modules/norm_layers.py",
    )
    modulate_layers = _load_source_module(
        "hymm.models.dit.modules.modulate_layers",
        root / "hymm/models/dit/modules/modulate_layers.py",
    )
    # MLP's constructor helper is imported from the actual pinned helpers file;
    # its forward, including activation and both linears, is the pinned source.
    importlib.import_module("hymm.utils.helpers")
    mlp_layers = _load_source_module(
        "hymm.models.dit.modules.mlp_layers",
        root / "hymm/models/dit/modules/mlp_layers.py",
    )
    posemb_layers = _load_source_module(
        "hymm.models.dit.modules.posemb_layers",
        root / "hymm/models/dit/modules/posemb_layers.py",
    )
    return (
        activation_layers.get_activation_layer,
        norm_layers.get_norm_layer,
        modulate_layers.ModulateDiT,
        modulate_layers.modulate,
        modulate_layers.apply_gate,
        mlp_layers.MLP,
        posemb_layers,
    )


def _extract_pinned_block(
    root: Path,
    torch: Any,
    adapter: Any,
    apply_rotary_emb: Any,
    helpers: tuple[Any, Any, Any, Any, Any, Any, Any],
) -> Any:
    """Compile the unchanged upstream class statement, not a local rewrite."""
    source_path = root / "hymm/models/dit/nomas_cpt.py"
    tree = ast.parse(source_path.read_text())
    block_node = next(
        (
            node
            for node in tree.body
            if isinstance(node, ast.ClassDef) and node.name == "MMDoubleStreamBlock"
        ),
        None,
    )
    if block_node is None:
        raise RuntimeError("Pinned nomas_cpt.py has no MMDoubleStreamBlock class")

    (
        get_activation_layer,
        get_norm_layer,
        ModulateDiT,
        modulate,
        apply_gate,
        MLP,
        _posemb_layers,
    ) = helpers
    mlp_module = sys.modules["hymm.models.dit.modules.mlp_layers"]
    namespace: dict[str, Any] = {
        "__name__": "hymm.models.dit.nomas_cpt_pinned_block",
        "torch": torch,
        "nn": torch.nn,
        "Any": Any,
        "Dict": dict,
        "List": list,
        "Optional": __import__("typing").Optional,
        "Tuple": __import__("typing").Tuple,
        "Union": __import__("typing").Union,
        "MoeConfig": Any,
        "HunYuanMoE": torch.nn.Module,
        "ModulateDiT": ModulateDiT,
        "get_activation_layer": get_activation_layer,
        "get_norm_layer": get_norm_layer,
        "MLP": MLP,
        "fixed_m_linear": mlp_module.fixed_m_linear,
        "apply_rotary_emb": apply_rotary_emb,
        "parallel_attention": adapter,
        "modulate": modulate,
        "apply_gate": apply_gate,
        "rearrange": importlib.import_module("einops").rearrange,
        "_os": os,
    }
    wrapper = ast.Module(body=[block_node], type_ignores=[])
    code = compile(wrapper, str(source_path), "exec")
    exec(code, namespace)
    return namespace["MMDoubleStreamBlock"]


def _expected_block_keys(hidden: int, ffn: int, heads: int) -> dict[str, list[int]]:
    expected: dict[str, list[int]] = {}
    for stream in ("img", "txt"):
        expected[f"{stream}_mod.linear.weight"] = [6 * hidden, hidden]
        expected[f"{stream}_mod.linear.bias"] = [6 * hidden]
        for projection in ("q", "k", "v", "proj"):
            prefix = f"{stream}_attn_{projection}"
            expected[f"{prefix}.weight"] = [hidden, hidden]
            expected[f"{prefix}.bias"] = [hidden]
        for norm in ("q_norm", "k_norm"):
            expected[f"{stream}_attn_{norm}.weight"] = [hidden // heads]
        expected[f"{stream}_mlp.fc1.weight"] = [ffn, hidden]
        expected[f"{stream}_mlp.fc1.bias"] = [ffn]
        expected[f"{stream}_mlp.fc2.weight"] = [hidden, ffn]
        expected[f"{stream}_mlp.fc2.bias"] = [hidden]
    return expected


def _trusted_weights_path(
    weights: Path | None,
    expected_sha256: str | None,
    *,
    real_width: bool,
) -> tuple[Path, str] | None:
    if (weights is None) != (expected_sha256 is None):
        raise ValueError("--weights and --weights-sha256 must be supplied together")
    if weights is None:
        return None
    if not real_width:
        raise ValueError("--weights is admitted only with --real-width")
    assert expected_sha256 is not None
    if re.fullmatch(r"[0-9a-fA-F]{64}", expected_sha256) is None:
        raise ValueError("--weights-sha256 must be exactly 64 hexadecimal characters")

    path = weights.resolve(strict=True)
    if not path.is_file():
        raise ValueError("--weights must name a regular safetensors file")
    if path.suffix.lower() != ".safetensors":
        raise ValueError("--weights accepts safetensors files only")
    actual_sha256 = _sha256(path)
    if actual_sha256 != expected_sha256.lower():
        raise ValueError(
            f"Trusted checkpoint SHA256 mismatch: {actual_sha256} != {expected_sha256.lower()}"
        )
    return path, actual_sha256


def _load_checkpoint_state(
    torch: Any,
    path: Path,
    config: dict[str, Any],
) -> tuple[dict[str, Any], str]:
    """Load only an exact, finite, single-block F32 or BF16 safetensors file."""
    try:
        from safetensors import safe_open
    except ImportError as exc:
        raise RuntimeError("Real block weights require the safetensors package") from exc

    expected = _expected_block_keys(
        config["dim"], config["ffn_dim"], config["heads"]
    )
    prefixes = "double_blocks.0."
    expected_source = {f"{prefixes}{name}": (name, shape) for name, shape in expected.items()}
    with safe_open(str(path), framework="pt", device="cpu") as source:
        actual_keys = set(source.keys())
        if actual_keys != set(expected_source):
            raise RuntimeError(
                "Trusted safetensors must contain exactly the 32 double_blocks.0 "
                f"block tensors: missing={sorted(set(expected_source) - actual_keys)}, "
                f"unexpected={sorted(actual_keys - set(expected_source))}"
            )

        state: dict[str, Any] = {}
        source_dtype: str | None = None
        for source_name, (relative_name, expected_shape) in expected_source.items():
            tensor = source.get_tensor(source_name)
            if list(tensor.shape) != expected_shape:
                raise RuntimeError(
                    f"Trusted block shape mismatch for {source_name}: "
                    f"{list(tensor.shape)} != {expected_shape}"
                )
            if tensor.dtype == torch.float32:
                dtype_name = "F32"
            elif tensor.dtype == torch.bfloat16:
                dtype_name = "BF16"
            else:
                raise RuntimeError(
                    f"Trusted block tensor {source_name} has unsupported dtype {tensor.dtype}"
                )
            if source_dtype is not None and dtype_name != source_dtype:
                raise RuntimeError(
                    "Trusted block tensors must use one dtype throughout; "
                    f"found both {source_dtype} and {dtype_name}"
                )
            source_dtype = dtype_name
            if not bool(torch.isfinite(tensor).all().item()):
                raise RuntimeError(f"Trusted block tensor {source_name} contains non-finite values")
            state[relative_name] = tensor

    if source_dtype is None or len(state) != 32:
        raise RuntimeError("Trusted checkpoint did not provide all 32 block tensors")
    return state, source_dtype


def _synthetic_state(torch: Any, model: Any, config: dict[str, Any]) -> dict[str, Any]:
    generator = torch.Generator(device="cpu").manual_seed(SEED_WEIGHTS)
    state: dict[str, Any] = {}
    for name, parameter in model.named_parameters():
        shape = tuple(parameter.shape)
        if name.endswith("_q_norm.weight") or name.endswith("_k_norm.weight"):
            value = 1.0 + 0.02 * torch.randn(shape, generator=generator, dtype=torch.float32)
        elif name.endswith(".weight") and parameter.ndim >= 2:
            fan_out, fan_in = shape[0], shape[1]
            std = math.sqrt(2.0 / float(fan_in + fan_out))
            value = torch.randn(shape, generator=generator, dtype=torch.float32) * std
        else:
            value = torch.randn(shape, generator=generator, dtype=torch.float32) * 0.02
        state[name] = value.contiguous()

    expected = _expected_block_keys(
        config["dim"], config["ffn_dim"], config["heads"]
    )
    if set(state) != set(expected):
        raise RuntimeError(
            "Pinned block parameter keys changed: "
            f"missing={sorted(set(expected) - set(state))}, "
            f"extra={sorted(set(state) - set(expected))}"
        )
    for name, value in state.items():
        if list(value.shape) != expected[name]:
            raise RuntimeError(
                f"Pinned block parameter shape changed for {name}: "
                f"{list(value.shape)} != {expected[name]}"
            )
    return state


def _tensor_record(tensor: Any) -> dict[str, Any]:
    values = (
        tensor.detach()
        .to(device="cpu", dtype=__import__("torch").float32)
        .contiguous()
        .reshape(-1)
        .tolist()
    )
    return {"shape": list(tensor.shape), "values": values}


def _tensor_records(tensors: dict[str, Any], torch: Any) -> dict[str, dict[str, Any]]:
    """Read all MPS checkpoints back in one transfer for this block call."""
    names = sorted(tensors)
    flattened = [tensors[name].detach().to(dtype=torch.float32).reshape(-1) for name in names]
    sizes = [int(value.numel()) for value in flattened]
    joined = torch.cat(flattened).to(device="cpu", dtype=torch.float32).contiguous()
    values = joined.tolist()
    records: dict[str, dict[str, Any]] = {}
    offset = 0
    for name, size in zip(names, sizes):
        records[name] = {
            "shape": list(tensors[name].shape),
            "values": values[offset : offset + size],
        }
        offset += size
    if offset != len(values):
        raise RuntimeError("Oracle checkpoint readback extent mismatch")
    return records


def _device_matches_expected(expected: Any, actual: Any) -> bool:
    """Compare a requested torch.device with a tensor's possibly indexed device."""
    if expected.type != actual.type:
        return False
    return expected.index is None or expected.index == actual.index


def _self_test_device_matching(torch: Any) -> None:
    """Exercise device matching using torch.device values only; no device access."""
    cases = (
        (torch.device("mps"), torch.device("mps:0"), True),
        (torch.device("mps"), torch.device("cpu"), False),
        (torch.device("mps:1"), torch.device("mps:0"), False),
    )
    for expected, actual, admitted in cases:
        observed = _device_matches_expected(expected, actual)
        if observed is not admitted:
            raise RuntimeError(
                f"Device-match self-test failed: {expected} vs {actual}: "
                f"expected {admitted}, got {observed}"
            )


def _round_parameters_bf16_in_place(torch: Any, model: Any, device: Any) -> None:
    """Round F32 parameters through BF16 one tensor at a time on their device."""
    parameters = list(model.named_parameters())
    if not parameters:
        raise RuntimeError("Cannot round an empty block parameter set")
    with torch.no_grad():
        for name, parameter in parameters:
            if (
                not _device_matches_expected(device, parameter.device)
                or parameter.dtype != torch.float32
            ):
                raise RuntimeError(
                    f"Expected F32 parameter {name} on {device}, got "
                    f"{parameter.dtype} on {parameter.device}"
                )
            rounded = parameter.detach().to(dtype=torch.bfloat16).to(dtype=torch.float32)
            parameter.copy_(rounded)
            del rounded


def _partition_tensor_records(
    records: dict[str, dict[str, Any]],
) -> tuple[
    dict[str, dict[str, Any]],
    dict[str, dict[str, Any]],
    dict[str, dict[str, Any]],
]:
    """Separate public block checkpoints from fixture-private tensor records."""
    checkpoints: dict[str, dict[str, Any]] = {}
    inputs: dict[str, dict[str, Any]] = {}
    attention_replay: dict[str, dict[str, Any]] = {}
    for name, record in records.items():
        if name.startswith("$input."):
            suffix = name.removeprefix("$input.")
            destination = inputs
        elif name.startswith("$attention_replay."):
            suffix = name.removeprefix("$attention_replay.")
            destination = attention_replay
        elif name.startswith("$"):
            raise RuntimeError(f"Unknown private tensor record prefix: {name}")
        else:
            checkpoints[name] = record
            continue
        if not suffix or suffix in destination:
            raise RuntimeError(f"Invalid or duplicate private tensor record: {name}")
        destination[suffix] = record
    return checkpoints, inputs, attention_replay


def _self_test_record_partition() -> None:
    """Offline guard for public/private fixture record separation."""
    public = {"shape": [1], "values": [1.0]}
    image = {"shape": [1, 2], "values": [2.0, 3.0]}
    query = {"shape": [1, 1, 1, 1], "values": [4.0]}
    checkpoints, inputs, replay = _partition_tensor_records(
        {
            "joint.attention": public,
            "$input.img": image,
            "$attention_replay.query": query,
        }
    )
    if checkpoints != {"joint.attention": public}:
        raise RuntimeError("Private records leaked into public checkpoints")
    if inputs != {"img": image}:
        raise RuntimeError("Input records were not partitioned correctly")
    if replay != {"query": query}:
        raise RuntimeError("Attention replay records were not partitioned correctly")
    try:
        _partition_tensor_records({"$unknown.private": public})
    except RuntimeError as exc:
        if "Unknown private tensor record prefix" not in str(exc):
            raise
    else:
        raise RuntimeError("Unknown private record prefix did not fail closed")


def _capture_hooks(model: Any, capture_state: dict[str, Any], heads: int, head_dim: int) -> list[Any]:
    hooks: list[Any] = []

    def save_output(module: Any, key: str, transform: Any = None) -> None:
        def hook(_module: Any, _inputs: tuple[Any, ...], output: Any) -> None:
            if not isinstance(output, __import__("torch").Tensor):
                return
            value = transform(output) if transform is not None else output
            current = capture_state["current"]
            current[key] = value.detach().clone()

        hooks.append(module.register_forward_hook(hook))

    def save_input(module: Any, key: str) -> None:
        def hook(_module: Any, inputs: tuple[Any, ...]) -> None:
            value = inputs[0]
            current = capture_state["current"]
            current[key] = value.detach().clone()

        hooks.append(module.register_forward_pre_hook(hook))

    torch = __import__("torch")
    for stream in ("img", "txt"):
        save_output(getattr(model, f"{stream}_mod"), f"{stream}.modulation")
        norm1 = getattr(model, f"{stream}_norm1")
        save_output(norm1, f"{stream}.attn.norm")
        qkv_modules = {}
        for projection in ("q", "k", "v"):
            linear = getattr(model, f"{stream}_attn_{projection}")
            qkv_modules[projection] = linear
            save_input(linear, f"{stream}.attn.modulated")
            save_output(
                linear,
                f"{stream}.attn.{projection}",
                lambda output, h=heads, d=head_dim: output.reshape(
                    output.shape[0], output.shape[1], h, d
                ),
            )
        for norm in ("q_norm", "k_norm"):
            save_output(
                getattr(model, f"{stream}_attn_{norm}"),
                f"{stream}.attn.{norm}",
            )
        save_output(
            getattr(model, f"{stream}_attn_proj"),
            f"{stream}.attn.projected",
        )

        norm2 = getattr(model, f"{stream}_norm2")
        save_input(norm2, f"{stream}.after_attention")
        save_output(norm2, f"{stream}.ff.norm")
        mlp = getattr(model, f"{stream}_mlp")
        save_input(mlp.fc1, f"{stream}.ff.modulated")
        save_output(mlp.fc1, f"{stream}.ff.hidden")
        save_output(mlp.act, f"{stream}.ff.activated")
        save_output(mlp.fc2, f"{stream}.ff.projected")
    return hooks


def _build_sdpa_adapter(
    torch: Any,
    capture_state: dict[str, Any],
    receipt: dict[str, Any],
    attention_precision: str,
) -> Any:
    if attention_precision not in ATTENTION_PRECISIONS:
        raise ValueError(f"Unsupported attention precision {attention_precision!r}")
    functional = torch.nn.functional

    def parallel_attention(
        q: tuple[Any, Any],
        k: tuple[Any, Any],
        v: tuple[Any, Any],
        img_q_len: int,
        img_kv_len: int,
        attn_mode: str | None = None,
        text_mask: Any = None,
        attn_param: dict[str, Any] | None = None,
    ) -> Any:
        if attn_mode != "magi_mp_causal":
            raise RuntimeError(f"Oracle adapter rejects attention mode {attn_mode!r}")
        if text_mask is not None:
            raise RuntimeError("Oracle fixture admits only full-valid text and no mask")
        if attn_param is None:
            raise RuntimeError("Oracle fixture requires explicit one-segment metadata")
        if attn_param.get("num_prompts") != 1:
            raise RuntimeError("Oracle fixture admits exactly one video segment")
        if attn_param.get("chunk_size_tokens", 0) != 0:
            raise RuntimeError("Oracle fixture admits segment-level attention only")
        if attn_param.get("prompt_video_boundaries_tokens") != [0, img_q_len]:
            raise RuntimeError("Oracle fixture requires a single full-video segment")
        if attn_param.get("prompt_video_boundaries") != [0, img_q_len]:
            raise RuntimeError("Oracle fixture requires one matching video boundary")
        if attn_param.get("txt_tokens_per_block") != q[1].shape[1]:
            raise RuntimeError("Oracle fixture requires one complete text block")
        if img_q_len != img_kv_len:
            raise RuntimeError("Oracle fixture does not admit cached/history video keys")

        query = torch.cat((q[0], q[1]), dim=1)
        key = torch.cat((k[0], k[1]), dim=1)
        value = torch.cat((v[0], v[1]), dim=1)
        if query.ndim != 4 or query.shape[0] <= 0:
            raise RuntimeError(f"Unexpected oracle Q shape: {tuple(query.shape)}")
        if key.shape != query.shape or value.shape != query.shape:
            raise RuntimeError("Oracle Q/K/V shapes differ")
        if any(tensor.device.type != "mps" for tensor in (query, key, value)):
            raise RuntimeError("Hy-Nomas oracle attention requires MPS Q/K/V tensors")
        if any(tensor.dtype != torch.float32 for tensor in (query, key, value)):
            raise RuntimeError("Hy-Nomas oracle attention requires F32 Q/K/V tensors")

        if attention_precision == "bf16-boundaries":
            # Match the wrapper's visible cast boundaries while keeping this
            # oracle's attention math itself in F32. This intentionally does
            # not model Magi/FlashAttention's internal BF16 arithmetic.
            query = query.to(dtype=torch.bfloat16).to(dtype=torch.float32)
            key = key.to(dtype=torch.bfloat16).to(dtype=torch.float32)
            value = value.to(dtype=torch.bfloat16).to(dtype=torch.float32)

        capture = capture_state["current"]
        capture["joint.query"] = query.detach().clone()
        capture["joint.key"] = key.detach().clone()
        capture["joint.value"] = value.detach().clone()

        query_h = query.permute(0, 2, 1, 3).contiguous()
        key_h = key.permute(0, 2, 1, 3).contiguous()
        value_h = value.permute(0, 2, 1, 3).contiguous()
        attended = functional.scaled_dot_product_attention(
            query_h,
            key_h,
            value_h,
            attn_mask=None,
            dropout_p=0.0,
            is_causal=False,
        )
        attention_f32 = attended.permute(0, 2, 1, 3).contiguous().flatten(2)
        if attention_f32.device.type != "mps":
            raise RuntimeError("PyTorch SDPA returned a non-MPS tensor")
        result = attention_f32
        if attention_precision == "bf16-boundaries":
            result = result.to(dtype=torch.bfloat16).to(dtype=torch.float32)

        capture["joint.attention"] = result.detach().clone()
        capture_state["attention_replay"] = {
            "query": query.detach().clone(),
            "key": key.detach().clone(),
            "value": value.detach().clone(),
            "attention_f32": attention_f32.detach().clone(),
            "attention": result.detach().clone(),
        }
        capture["img.attn.raw"] = result[:, :img_q_len].detach().clone()
        capture["txt.attn.raw"] = result[:, img_q_len:].detach().clone()
        receipt["calls"] += 1
        receipt["last_q_shape"] = list(query.shape)
        receipt["last_sdpa_shape"] = list(query_h.shape)
        receipt["last_qkv_dtype"] = str(query.dtype).removeprefix("torch.")
        receipt["last_output_dtype"] = str(result.dtype).removeprefix("torch.")
        return result

    return parallel_attention


def _make_case_inputs(
    torch: Any, posemb_layers: Any, device: Any, name: str, config: dict[str, Any]
) -> tuple[dict[str, Any], dict[str, Any]]:
    generator = torch.Generator(device="cpu").manual_seed(
        SEED_INPUTS + (0 if name == "broadcast" else 1)
    )
    hidden = config["dim"]
    video_tokens = config["video_tokens"]
    text_tokens = config["text_tokens"]
    batch = 2 if name == "broadcast" else 1

    def random(shape: tuple[int, ...], scale: float) -> Any:
        return torch.randn(shape, generator=generator, dtype=torch.float32) * scale

    if name == "broadcast":
        vec = random((batch, hidden), 0.35)
        vec_txt = random((batch, hidden), 0.35)
    elif name == "per_token":
        vec = random((batch, video_tokens, hidden), 0.35)
        vec_txt = random((batch, text_tokens, hidden), 0.35)
    else:
        raise ValueError(f"Unknown fixture case {name}")

    cos, sin = posemb_layers.get_nd_rotary_pos_embed(
        config["rope_dim_list"],
        tuple(config["rope_grid"]),
        theta=config["rope_theta"],
        use_real=True,
        device=torch.device("cpu"),
    )
    cpu_values = {
        "img": random((batch, video_tokens, hidden), 0.4),
        "txt": random((batch, text_tokens, hidden), 0.4),
        "vec": vec,
        "vec_txt": vec_txt,
        "video_cos": cos.to(device="cpu", dtype=torch.float32),
        "video_sin": sin.to(device="cpu", dtype=torch.float32),
    }
    device_values = {
        name: value.to(device=device, dtype=torch.float32)
        for name, value in cpu_values.items()
    }
    return cpu_values, device_values


def _run_case(
    torch: Any,
    block: Any,
    capture_state: dict[str, Any],
    receipt: dict[str, Any],
    device_inputs: dict[str, Any],
    *,
    variant: str,
    config: dict[str, Any],
) -> tuple[dict[str, Any], tuple[Any, Any]]:
    capture: dict[str, Any] = {}
    capture_state["current"] = capture
    capture_state["attention_replay"] = {}
    before_calls = receipt["calls"]
    video_tokens = config["video_tokens"]
    text_tokens = config["text_tokens"]
    attn_param = {
        "num_prompts": 1,
        "txt_tokens_per_block": text_tokens,
        "prompt_video_boundaries_tokens": [0, video_tokens],
        "chunk_size_tokens": 0,
        "seg_ar_attention_backend": "magi",
    }
    with torch.inference_mode():
        img_out, txt_out = block(
            img=device_inputs["img"],
            txt=device_inputs["txt"],
            vec=device_inputs["vec"],
            vec_txt=device_inputs["vec_txt"],
            freqs_cis=(device_inputs["video_cos"], device_inputs["video_sin"]),
            text_mask=None,
            attn_param=attn_param,
            is_flash=False,
            txt_tokens_per_block=text_tokens,
            prompt_video_boundaries=[0, video_tokens],
        )
    if receipt["calls"] != before_calls + 1:
        raise RuntimeError("Pinned block did not make exactly one SDPA call")
    capture["img.output"] = img_out.detach().clone()
    capture["txt.output"] = txt_out.detach().clone()

    required = {
        "joint.query",
        "joint.key",
        "joint.value",
        "joint.attention",
        "img.modulation",
        "img.attn.norm",
        "img.attn.modulated",
        "img.attn.q",
        "img.attn.k",
        "img.attn.v",
        "img.attn.q_norm",
        "img.attn.k_norm",
        "img.attn.q_rope",
        "img.attn.k_rope",
        "img.attn.raw",
        "img.attn.projected",
        "img.after_attention",
        "img.ff.norm",
        "img.ff.modulated",
        "img.ff.hidden",
        "img.ff.activated",
        "img.ff.projected",
        "img.output",
        "txt.modulation",
        "txt.attn.norm",
        "txt.attn.modulated",
        "txt.attn.q",
        "txt.attn.k",
        "txt.attn.v",
        "txt.attn.q_norm",
        "txt.attn.k_norm",
        "txt.attn.raw",
        "txt.attn.projected",
        "txt.after_attention",
        "txt.ff.norm",
        "txt.ff.modulated",
        "txt.ff.hidden",
        "txt.ff.activated",
        "txt.ff.projected",
        "txt.output",
    }
    if set(capture) != required:
        raise RuntimeError(
            f"Pinned block trace keys differ: missing={sorted(required - set(capture))}, "
            f"extra={sorted(set(capture) - required)}"
        )

    replay_tensors = capture_state["attention_replay"]
    if set(replay_tensors) != {"query", "key", "value", "attention_f32", "attention"}:
        raise RuntimeError("Pinned block did not provide a complete attention replay")
    all_tensors = dict(capture)
    all_tensors["$input.img"] = device_inputs["img"]
    all_tensors["$input.txt"] = device_inputs["txt"]
    all_tensors.update({f"$attention_replay.{key}": value for key, value in replay_tensors.items()})
    records = _tensor_records(all_tensors, torch)
    checkpoints, input_records, attention_replay = _partition_tensor_records(records)
    _assert_nontrivial(input_records, checkpoints)
    result = {
        "precision_variant": variant,
        "outputs": {
            "img": checkpoints["img.output"],
            "txt": checkpoints["txt.output"],
        },
        "checkpoints": checkpoints,
        "attention_precision": receipt["attention_precision"],
        "attention_replay": attention_replay,
        "attention_receipt": {
            "calls": 1,
            "joint_qkv_shape": receipt["last_q_shape"],
            "sdpa_q_shape": receipt["last_sdpa_shape"],
            "mask": None,
            "causal": False,
            "dropout_p": 0.0,
            "precision": receipt["attention_precision"],
            "qkv_dtype": receipt["last_qkv_dtype"],
            "output_dtype": receipt["last_output_dtype"],
        },
    }
    return result, (img_out.detach().clone(), txt_out.detach().clone())


def _make_perturbation(torch: Any, image: Any, case_index: int) -> Any:
    generator = torch.Generator(device="cpu").manual_seed(20261026 + case_index)
    return torch.randn(
        image.shape,
        generator=generator,
        dtype=torch.float32,
    ) * 0.17


def _assert_perturbation_response(
    delta: Any,
    base_result: dict[str, Any],
    perturbed_result: dict[str, Any],
) -> None:
    base_img_out = base_result["outputs"]["img"]["values"]
    perturbed_img_out = perturbed_result["outputs"]["img"]["values"]
    img_input_delta = delta.reshape(-1).tolist()
    if not any(
        abs((new - old) - input_delta) > 1e-7
        for old, new, input_delta in zip(base_img_out, perturbed_img_out, img_input_delta)
    ):
        raise RuntimeError(
            "Video perturbation response is only the residual identity path; "
            "synthetic attention/MLP gates may be zero"
        )

    base_txt_out = base_result["outputs"]["txt"]["values"]
    perturbed_txt_out = perturbed_result["outputs"]["txt"]["values"]
    if not any(abs(new - old) > 1e-8 for old, new in zip(base_txt_out, perturbed_txt_out)):
        raise RuntimeError(
            "Video perturbation did not affect text output through joint attention"
        )


def _chain_device_inputs(device_inputs: dict[str, Any], outputs: tuple[Any, Any]) -> dict[str, Any]:
    chained = dict(device_inputs)
    chained["img"], chained["txt"] = outputs
    return chained


def _assert_nontrivial(
    inputs: dict[str, Any], checkpoints: dict[str, Any]
) -> None:
    for key in ("img.modulation", "txt.modulation", "joint.attention"):
        values = checkpoints[key]["values"]
        if not values or max(abs(value) for value in values) <= 1e-8:
            raise RuntimeError(f"Synthetic oracle produced a trivial {key}")
    for stream, input_key in (("img", "img"), ("txt", "txt")):
        output = checkpoints[f"{stream}.output"]["values"]
        source = inputs[input_key]["values"]
        if len(output) != len(source) or not any(
            abs(left - right) > 1e-7 for left, right in zip(output, source)
        ):
            raise RuntimeError(f"Synthetic oracle {stream} output did not change its input")
    for key, tensor in checkpoints.items():
        if any(not math.isfinite(value) for value in tensor["values"]):
            raise RuntimeError(f"Non-finite value in synthetic oracle checkpoint {key}")


def _bindings(state: dict[str, Any], variant_dtype: str) -> list[dict[str, Any]]:
    return [
        {
            "source": f"double_blocks.0.{name}",
            "block_relative": name,
            "shape": list(tensor.shape),
            "dtype": variant_dtype,
        }
        for name, tensor in state.items()
    ]


def _save_fixture(
    output: Path,
    torch: Any,
    source_hashes: dict[str, str],
    revision: str,
    state_f32: dict[str, Any],
    state_bf16: dict[str, Any],
    cases: list[dict[str, Any]],
    receipt: dict[str, Any],
    config: dict[str, Any],
    attention_precision: str,
    source_state: dict[str, Any] | None = None,
    source_dtype: str | None = None,
    source_provenance: dict[str, Any] | None = None,
) -> None:
    from safetensors.torch import save_file

    if output.exists() or output.is_symlink():
        raise FileExistsError(f"Output JSON must be a new file: {output}")
    if output.suffix.lower() != ".json":
        raise ValueError("--output must name a .json reference file")
    if not output.parent.is_dir():
        raise FileNotFoundError(f"Output parent must already exist: {output.parent}")

    if attention_precision not in ATTENTION_PRECISIONS:
        raise ValueError(f"Unsupported attention precision {attention_precision!r}")
    if (source_state is None) != (source_dtype is None):
        raise ValueError("Real source state and dtype must be supplied together")
    if (source_state is None) != (source_provenance is None):
        raise ValueError("Real source provenance must accompany source tensors")

    synthetic = source_state is None
    weights_f32_name = f"{output.stem}.weights_f32.safetensors"
    weights_bf16_name = f"{output.stem}.weights_bf16.safetensors"
    weights_source_name = f"{output.stem}.weights_source.safetensors"
    manifest_name = f"{output.stem}.manifest.json"
    sidecar_names = [weights_f32_name, weights_bf16_name] if synthetic else [weights_source_name]
    final_paths = [output.parent / name for name in sidecar_names]
    final_paths.extend([output.parent / manifest_name, output])
    for final_path in final_paths:
        if final_path.exists() or final_path.is_symlink():
            raise FileExistsError(f"Fixture sidecar must be a new file: {final_path}")

    stage = output.parent / f".{output.name}.tmp-{os.getpid()}"
    if stage.exists() or stage.is_symlink():
        raise FileExistsError(f"Temporary output path already exists: {stage}")
    stage.mkdir()
    published: list[Path] = []
    try:
        if synthetic:
            save_file(
                {
                    f"double_blocks.0.{name}": value.contiguous()
                    for name, value in state_f32.items()
                },
                str(stage / weights_f32_name),
                metadata={
                    "format": "hy-nomas-synthetic-double-stream-block",
                    "upstream_revision": revision,
                    "precision": "F32",
                    "weights_are_synthetic": "true",
                },
            )
            save_file(
                {
                    f"double_blocks.0.{name}": value.to(dtype=torch.bfloat16).contiguous()
                    for name, value in state_bf16.items()
                },
                str(stage / weights_bf16_name),
                metadata={
                    "format": "hy-nomas-synthetic-double-stream-block",
                    "upstream_revision": revision,
                    "precision": "BF16 weights only; fixture computes in F32",
                    "weights_are_synthetic": "true",
                },
            )
            sidecars = {
                "f32": (weights_f32_name, state_f32, "F32", "F32", False),
                "bf16_weights_roundtrip_f32_compute": (
                    weights_bf16_name,
                    state_bf16,
                    "BF16",
                    "BF16",
                    False,
                ),
            }
        else:
            assert source_state is not None and source_dtype is not None
            assert source_provenance is not None
            save_file(
                {
                    f"double_blocks.0.{name}": value.contiguous()
                    for name, value in source_state.items()
                },
                str(stage / weights_source_name),
                metadata={
                    "format": "hy-nomas-trusted-double-stream-block",
                    "upstream_revision": revision,
                    "precision": source_dtype,
                    "weights_are_synthetic": "false",
                    "source_checkpoint_sha256": source_provenance["file_sha256"],
                },
            )
            # Both reference controls consume the exact source tensors. The
            # BF16 resident variant rounds an F32 source at preparation time;
            # if the source is already BF16, no conversion is admitted.
            sidecars = {
                "f32": (weights_source_name, source_state, source_dtype, "F32", False),
                "bf16_weights_roundtrip_f32_compute": (
                    weights_source_name,
                    source_state,
                    source_dtype,
                    "BF16",
                    source_dtype == "F32",
                ),
            }
        memory_policy = {
            "physical_memory_reserve_percent": 15,
            "runtime_budget_bytes": None,
            "budget_enforced_by": "external cache-aware runtime guard; producer does not sample host memory",
        }
        artifacts: dict[str, dict[str, Any]] = {}
        for variant_name, (filename, binding_state, source_weight_dtype, resident_dtype, allow_cast) in sidecars.items():
            path = stage / filename
            artifacts[variant_name] = {
                "weights_file": {
                    "path": filename,
                    "sha256": _sha256(path),
                    "bytes": path.stat().st_size,
                    "dtype": source_weight_dtype,
                },
                "checkpoint_bindings": _bindings(binding_state, source_weight_dtype),
                "compute_dtype": "F32",
                "resident_dtype": resident_dtype,
                "allow_f32_to_bf16": allow_cast,
                "attention_precision": attention_precision,
                "weights_are_synthetic": synthetic,
                "memory_policy": memory_policy,
            }
        artifacts["bf16_weights_roundtrip_f32_compute"]["compute_control"] = (
            "BF16-rounded weights cast back to F32 parameters; F32 activations and compute"
        )
        if source_provenance is not None:
            for artifact in artifacts.values():
                artifact["source_checkpoint"] = source_provenance
        for case in cases:
            case["variants"]["f32"].update(artifacts["f32"])
            case["variants"]["bf16_weights_roundtrip_f32_compute"].update(
                artifacts["bf16_weights_roundtrip_f32_compute"]
            )
        reference_kind = (
            "upstream_block_bf16_boundary_control_mps_sdpa"
            if attention_precision == "bf16-boundaries"
            else "upstream_block_f32_math_mps_sdpa"
        )
        attention_precision_note = (
            "F32 concatenated Q/K/V round through BF16 before F32 SDPA; F32 SDPA output rounds through BF16 afterward. "
            "This boundary control is not Magi/FlashAttention BF16 parity. Magi's output dtype was not executed here; "
            "BF16 output followed by wrapper cast-back is only a hypothesis for the selected kernel."
            if attention_precision == "bf16-boundaries"
            else "F32 SDPA Q/K/V/output. The pinned Magi wrapper casts F32 Q/K/V through BF16; its output cast-back is conditional on kernel output dtype."
        )
        provenance = {
            "revision": revision,
            "upstream_revision": revision,
            "file_sha256": source_hashes,
            "source_sha256": source_hashes,
            "block_source": "hymm/models/dit/nomas_cpt.py:887-1288",
            "class_loading": "compile exact pinned MMDoubleStreamBlock AST statement; direct pinned helper modules",
            "attention_adapter": "oracle-only MPS PyTorch SDPA replacement for parallel_attention",
            "attention_precision": attention_precision,
            "attention_precision_note": attention_precision_note,
            "torch_version": torch.__version__,
            "mps_device": "mps",
        }
        if source_provenance is not None:
            provenance["source_checkpoint"] = source_provenance
        weights_by_file: dict[str, dict[str, Any]] = {}
        bindings_by_file: dict[str, list[dict[str, Any]]] = {}
        for artifact in artifacts.values():
            weights_file = artifact["weights_file"]
            filename = weights_file["path"]
            weights_by_file[filename] = {
                "weights_file": weights_file,
                "compute_dtype": "F32",
                "weights_are_synthetic": synthetic,
            }
            bindings_by_file[filename] = artifact["checkpoint_bindings"]
        manifest = {
            "schema_version": 2,
            "artifact": "Hy-Nomas MMDoubleStreamBlock single-segment oracle",
            "provenance": provenance,
            "scope": {
                "reference_kind": reference_kind,
                "attention_precision": attention_precision,
                "pinned_magi_runtime_numeric_parity": False,
                "block": "MMDoubleStreamBlock only; no model or pipeline instantiated",
                "segment_mode": "single_full",
                "history": "none",
                "text_mask": "none_all_valid",
                "camera": "unsupported_absent",
                "segment_count": 1,
                "chunk_size_tokens": 0,
                "history_tokens": 0,
                "has_mask": False,
                "has_camera": False,
                "action_windows": False,
                "segment_contract": "one full video segment; num_prompts=1; boundaries=[0,video_tokens]; chunk_size_tokens=0",
                "attention_semantics": "each query sees all video tokens in the one segment and all tokens in its one fully valid text block",
                "history_or_cache": "none; no KV cache and no preceding segments",
                "text_mask_semantics": "no mask; all text tokens are valid",
                "camera_semantics": "separate post-block path; absent and not represented by a no-op",
                "weights_are_synthetic": synthetic,
                "other_excluded_paths": [
                    "multi-segment causal prefix",
                    "chunk-level causal attention",
                    "history/cache attention",
                    "action-window masking",
                    *( [] if not synthetic else ["real checkpoint weights"] ),
                    "native BF16 activations",
                ],
            },
            "config": config,
            "memory_policy": memory_policy,
            "weights_are_synthetic": synthetic,
            "source": source_provenance,
            "weights": weights_by_file,
            "cases": [case["name"] for case in cases],
            "attention_adapter_receipt": {
                "sdpa_calls_total": receipt["calls"],
                "requires_all_qkv_on_mps": True,
                "explicit_attention_mask": None,
                "is_causal": False,
                "attention_precision": attention_precision,
                "qkv_dtype": receipt.get("last_qkv_dtype"),
                "output_dtype": receipt.get("last_output_dtype"),
                "bf16_output_hypothesis": "Magi kernel output dtype was not executed or confirmed; wrapper casts result back only when its dtype is BF16",
            },
        }
        reference = {
            "schema_version": 2,
            "provenance": provenance,
            "scope": manifest["scope"],
            "weights_are_synthetic": synthetic,
            "source": source_provenance,
            "config": config,
            "checkpoint_bindings": bindings_by_file,
            "manifest_file": manifest_name,
            "cases": cases,
        }
        (stage / manifest_name).write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n"
        )
        (stage / output.name).write_text(
            json.dumps(reference, indent=2, sort_keys=True) + "\n"
        )
        for name, final_path in zip(
            (*sidecar_names, manifest_name, output.name),
            final_paths,
        ):
            (stage / name).rename(final_path)
            published.append(final_path)
    except Exception:
        for path in published:
            path.unlink(missing_ok=True)
        shutil.rmtree(stage, ignore_errors=True)
        raise
    else:
        shutil.rmtree(stage)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--upstream",
        type=Path,
        default=DEFAULT_UPSTREAM,
        help="pinned Hy-Nomas checkout at revision " + PINNED_REVISION,
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="new .json reference path; weights and manifest are sibling sidecars",
    )
    parser.add_argument(
        "--structure-self-test",
        action="store_true",
        help="run the offline tensor-record partition guard without importing PyTorch/MPS",
    )
    parser.add_argument(
        "--device-match-self-test",
        action="store_true",
        help="test torch.device type/index matching without querying or using a GPU",
    )
    parser.add_argument(
        "--stage-telemetry",
        action="store_true",
        help="emit monotonic JSON stage markers to stderr; does not sample memory or synchronize MPS",
    )
    parser.add_argument(
        "--real-width",
        action="store_true",
        help=(
            "use Nomas-G C2048/H16/D128/F8192 geometry with only six video and "
            "three text tokens; optional --weights admits one trusted block only"
        ),
    )
    parser.add_argument(
        "--attention-precision",
        choices=ATTENTION_PRECISIONS,
        default="f32",
        help=(
            "attention boundary control: f32 or bf16-boundaries (BF16 QKV/output "
            "rounding around F32 MPS SDPA; not Magi/CUDA parity)"
        ),
    )
    parser.add_argument(
        "--weights",
        type=Path,
        help="trusted safetensors containing exactly double_blocks.0 (requires --real-width)",
    )
    parser.add_argument(
        "--weights-sha256",
        help="trusted SHA-256 of the exact --weights file bytes (required with --weights)",
    )
    args = parser.parse_args()
    if args.structure_self_test:
        _self_test_record_partition()
        print(json.dumps({"tensor_record_partition": "pass"}, sort_keys=True))
        return 0
    if args.device_match_self_test:
        import torch

        _self_test_device_matching(torch)
        print(json.dumps({"device_matching": "pass"}, sort_keys=True))
        return 0
    if args.output is None:
        parser.error("--output is required unless --structure-self-test is selected")
    config = dict(REAL_WIDTH if args.real_width else TINY)
    try:
        trusted_weights = _trusted_weights_path(
            args.weights,
            args.weights_sha256,
            real_width=args.real_width,
        )
    except (ValueError, FileNotFoundError, OSError) as exc:
        parser.error(str(exc))
    if trusted_weights is not None:
        config["geometry_label"] = "nomas_g_real_width_trusted_single_block_probe"

    fallback = os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK")
    if fallback not in (None, "0"):
        raise SystemExit("PYTORCH_ENABLE_MPS_FALLBACK must be unset or 0")
    os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "0"
    fixed_m = os.environ.get("HY_TEXT_MLP_FC2_FIXED_M", "0") or "0"
    if fixed_m != "0":
        raise SystemExit("HY_TEXT_MLP_FC2_FIXED_M must be unset or 0 for this oracle")

    upstream = args.upstream.resolve()
    source_hashes = _verify_upstream(upstream)

    import torch

    source_state: dict[str, Any] | None = None
    source_dtype: str | None = None
    source_provenance: dict[str, Any] | None = None
    if trusted_weights is not None:
        weights_path, weights_sha256 = trusted_weights
        try:
            source_state, source_dtype = _load_checkpoint_state(torch, weights_path, config)
        except (RuntimeError, OSError, ValueError) as exc:
            raise SystemExit(str(exc)) from exc
        source_provenance = {
            "format": "safetensors",
            "file_sha256": weights_sha256,
            "weights_are_synthetic": False,
            "source_dtype": source_dtype,
            "tensor_prefix": "double_blocks.0.",
            "tensor_count": 32,
        }
        _emit_stage_telemetry(
            args.stage_telemetry,
            "source_loaded",
            source_dtype=source_dtype,
            tensor_count=len(source_state),
        )

    if not torch.backends.mps.is_built() or not torch.backends.mps.is_available():
        raise SystemExit("This oracle requires an available PyTorch MPS device; CPU fallback is forbidden")
    device = torch.device("mps")
    if torch.get_default_dtype() != torch.float32:
        torch.set_default_dtype(torch.float32)

    helpers = _install_upstream_modules(upstream)
    posemb_layers = helpers[-1]
    capture_state: dict[str, Any] = {"current": {}}
    receipt: dict[str, Any] = {
        "calls": 0,
        "attention_precision": args.attention_precision,
    }

    def apply_rotary_capture(q: Any, k: Any, freqs_cis: Any, head_first: bool = False) -> tuple[Any, Any]:
        original = posemb_layers.apply_rotary_emb
        if q.device.type != "mps" or k.device.type != "mps":
            raise RuntimeError("Hy-Nomas oracle RoPE requires MPS Q/K tensors")
        q_rot, k_rot = original(q, k, freqs_cis, head_first=head_first)
        capture = capture_state["current"]
        capture["img.attn.q_rope"] = q_rot.detach().clone()
        capture["img.attn.k_rope"] = k_rot.detach().clone()
        return q_rot, k_rot

    adapter = _build_sdpa_adapter(
        torch, capture_state, receipt, args.attention_precision
    )
    block_type = _extract_pinned_block(
        upstream,
        torch,
        adapter,
        apply_rotary_capture,
        helpers,
    )
    block = block_type(
        hidden_size=config["dim"],
        heads_num=config["heads"],
        mlp_width_ratio=config["ffn_dim"] / config["dim"],
        mlp_act_type=config["mlp_act_type"],
        attn_mode="magi_mp_causal",
        qk_norm=config["qk_norm"],
        qk_norm_type=config["qk_norm_type"],
        qkv_bias=config["qkv_bias"],
        dtype=torch.float32,
        # Trusted real weights already occupy CPU memory. Constructing a full
        # random CPU block first needlessly adds another model-sized copy at
        # the 4 GiB guard boundary; meta parameters are replaced by the exact
        # safetensors tensors below, then moved to MPS for computation.
        device=torch.device("meta" if source_state is not None else "cpu"),
    )
    block.eval()
    if source_state is None:
        state_f32 = _synthetic_state(torch, block, config)
        state_bf16 = {
            name: tensor.to(dtype=torch.bfloat16).to(dtype=torch.float32).contiguous()
            for name, tensor in state_f32.items()
        }
        block.load_state_dict(state_f32, strict=True)
        block.to(device=device, dtype=torch.float32)
    else:
        if source_dtype not in ("F32", "BF16"):
            raise RuntimeError("Trusted source dtype was not established before block loading")
        # Keep the source tensors as the F32-control input mapping. load_state_dict
        # casts BF16 source values into the F32 MPS parameters when needed.
        state_f32 = source_state
        # An already-BF16 source is its own BF16-resident control. For F32
        # source weights, the BF16-rounded/F32-compute control is prepared
        # in place on MPS for each case to avoid a retained model-sized copy.
        state_bf16 = source_state if source_dtype == "BF16" else {}
        block.load_state_dict(source_state, strict=True, assign=True)
        block.to(device=device, dtype=torch.float32)
    _emit_stage_telemetry(
        args.stage_telemetry,
        "model_resident",
        geometry=config["geometry_label"],
        source_dtype=source_dtype or "synthetic",
        device=device.type,
    )
    hooks = _capture_hooks(
        block,
        capture_state,
        config["heads"],
        config["head_dim"],
    )

    cases: list[dict[str, Any]] = []
    try:
        for case_index, case_name in enumerate(("broadcast", "per_token")):
            cpu_inputs, device_inputs = _make_case_inputs(
                torch, posemb_layers, device, case_name, config
            )
            batch = cpu_inputs["img"].shape[0]
            if case_name == "broadcast":
                modulation_rows = {
                    "batches": batch,
                    "img_modulation_rows": 0,
                    "txt_modulation_rows": 0,
                }
            else:
                modulation_rows = {
                    "batches": batch,
                    "img_modulation_rows": config["video_tokens"],
                    "txt_modulation_rows": config["text_tokens"],
                }
            variant_results: dict[str, Any] = {}
            _emit_stage_telemetry(
                args.stage_telemetry,
                "variant_start",
                case=case_name,
                variant="f32",
            )
            block.load_state_dict(state_f32, strict=True)
            base_f32, base_f32_outputs = _run_case(
                torch,
                block,
                capture_state,
                receipt,
                device_inputs,
                variant="f32",
                config=config,
            )
            chain_f32, _ = _run_case(
                torch,
                block,
                capture_state,
                receipt,
                _chain_device_inputs(device_inputs, base_f32_outputs),
                variant="f32_chain_second_application",
                config=config,
            )
            delta = _make_perturbation(torch, cpu_inputs["img"], case_index)
            perturbed_cpu = dict(cpu_inputs)
            perturbed_cpu["img"] = cpu_inputs["img"] + delta
            perturbed_device = dict(device_inputs)
            perturbed_device["img"] = perturbed_cpu["img"].to(
                device=device, dtype=torch.float32
            )
            perturbed_f32, _ = _run_case(
                torch,
                block,
                capture_state,
                receipt,
                perturbed_device,
                variant="f32_video_perturbation",
                config=config,
            )
            _assert_perturbation_response(delta, base_f32, perturbed_f32)
            base_f32["chain_expected"] = {
                "input_source": "outputs of this variant's first application",
                "outputs": chain_f32["outputs"],
                "checkpoints": chain_f32["checkpoints"],
            }
            base_f32["video_perturbation"] = {
                "seed": 20261026 + case_index,
                "scale": 0.17,
                "delta": _tensor_record(delta),
                "input_img": _tensor_record(perturbed_cpu["img"]),
                "outputs": perturbed_f32["outputs"],
                "checkpoints": perturbed_f32["checkpoints"],
            }
            variant_results["f32"] = base_f32
            _emit_stage_telemetry(
                args.stage_telemetry,
                "variant_complete",
                case=case_name,
                variant="f32",
            )

            _emit_stage_telemetry(
                args.stage_telemetry,
                "variant_start",
                case=case_name,
                variant="bf16_weights_roundtrip_f32_compute",
            )
            if source_state is not None and source_dtype == "F32":
                _round_parameters_bf16_in_place(torch, block, device)
            else:
                block.load_state_dict(state_bf16, strict=True)
            base_bf16, base_bf16_outputs = _run_case(
                torch,
                block,
                capture_state,
                receipt,
                device_inputs,
                variant="bf16_weights_roundtrip_f32_compute",
                config=config,
            )
            chain_bf16, _ = _run_case(
                torch,
                block,
                capture_state,
                receipt,
                _chain_device_inputs(device_inputs, base_bf16_outputs),
                variant="bf16_weights_roundtrip_f32_compute_chain_second_application",
                config=config,
            )
            perturbed_bf16, _ = _run_case(
                torch,
                block,
                capture_state,
                receipt,
                perturbed_device,
                variant="bf16_weights_roundtrip_f32_compute_video_perturbation",
                config=config,
            )
            _assert_perturbation_response(
                delta, base_bf16, perturbed_bf16
            )
            base_bf16["chain_expected"] = {
                "input_source": "outputs of this variant's first application",
                "outputs": chain_bf16["outputs"],
                "checkpoints": chain_bf16["checkpoints"],
            }
            base_bf16["video_perturbation"] = {
                "seed": 20261026 + case_index,
                "scale": 0.17,
                "delta": _tensor_record(delta),
                "input_img": _tensor_record(perturbed_cpu["img"]),
                "outputs": perturbed_bf16["outputs"],
                "checkpoints": perturbed_bf16["checkpoints"],
            }
            variant_results["bf16_weights_roundtrip_f32_compute"] = base_bf16
            _emit_stage_telemetry(
                args.stage_telemetry,
                "variant_complete",
                case=case_name,
                variant="bf16_weights_roundtrip_f32_compute",
            )
            cases.append(
                {
                    "name": case_name,
                    "config_overrides": modulation_rows,
                    "modulation_layout": {
                        "img": "broadcast" if case_name == "broadcast" else "per_token",
                        "txt": "broadcast" if case_name == "broadcast" else "per_token",
                    },
                    "inputs": {
                        key: _tensor_record(value)
                        for key, value in cpu_inputs.items()
                    },
                    "variants": variant_results,
                }
            )
    finally:
        for hook in hooks:
            hook.remove()

    if receipt["calls"] != len(cases) * 6:
        raise RuntimeError(f"Unexpected total SDPA calls: {receipt['calls']}")
    _emit_stage_telemetry(args.stage_telemetry, "fixture_write_begin")
    _save_fixture(
        args.output.resolve(),
        torch,
        source_hashes,
        PINNED_REVISION,
        state_f32,
        state_bf16,
        cases,
        receipt,
        config,
        args.attention_precision,
        source_state,
        source_dtype,
        source_provenance,
    )
    _emit_stage_telemetry(args.stage_telemetry, "fixture_write_end")
    print(
        json.dumps(
            {
                "status": "created",
                "output": str(args.output.resolve()),
                "manifest": str(args.output.resolve().with_name(args.output.stem + ".manifest.json")),
                "cases": [case["name"] for case in cases],
                "device": "mps",
                "geometry_label": config["geometry_label"],
                "attention_calls": receipt["calls"],
                "upstream_revision": PINNED_REVISION,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
