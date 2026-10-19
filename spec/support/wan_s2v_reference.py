#!/usr/bin/env python3
"""Generate tiny Wan S2V block/audio fixtures from the pinned upstream modules.

Copyright 2024-2025 The Alibaba Wan Team Authors.
Apache-2.0. Adapted fixture harness for cogni-ml; upstream pin:
1ea34ff48f87168174e12956e200b1d908b1c5ff.

The block and injector are instantiated and invoked from the pinned Wan source.
Only attention is adapted from CUDA-only flash/distributed calls to PyTorch SDPA.
The upstream FP64 complex RoPE table is setup data; its real-valued application
is performed on MPS in F32. Neural operators and activations remain on MPS.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import importlib.util
import json
import os
import sys
import types
from pathlib import Path
from typing import Any

os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "0"

import torch
import torch.nn as nn
import torch.nn.functional as F


REVISION = "1ea34ff48f87168174e12956e200b1d908b1c5ff"
DIFFUSERS_VERSION = "0.31.0"
DEFAULT_UPSTREAM = Path("/opt/mldata_ssd/cogni-ml/video-scene-20261019/wan-audit")
SOURCE_SHA256 = {
    "wan_modules_model.py": "8b39115298ca7322806c19b3165b3f435a94fe4a58f0624aec24f8e7f4997432",
    "wan_modules_s2v_model_s2v.py": "91fb14df44edab2d01b5634420b463c5f1563aecdf6dac905da6b3acac3145cc",
    "wan_modules_s2v_audio_utils.py": "76931d7291f69a6b6ed3a7a1a9a106ad8b5f5d494a451db67718b4fbcc4dbcfd",
    "wan_modules_s2v_s2v_utils.py": "e45ae6c274b17e4fdddcb0439b598897a37679b8934ebd71f87255f92d5f0b2f",
    "wan_modules_s2v_auxi_blocks.py": "6ff7b5560036e5f0eaaa34460a328afcdbd7bd62baf93b699d5f6d2d556f95d6",
    # Exact AdaLayerNorm source snapshot from Hugging Face Diffusers v0.31.0.
    "diffusers-normalization-v0.31.0.py": "7f2d2086f669e19bc008d71743f3963017274fbc6ad86d0ab9bf18db211e7c5a",
}


def _verify_sources(upstream: Path) -> None:
    """Fail closed if the external source snapshot differs from the reviewed pin."""
    for name, expected in SOURCE_SHA256.items():
        path = upstream / name
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
        if actual != expected:
            raise RuntimeError(f"Pinned source hash mismatch for {name}: {actual}")


def _module(name: str, package: bool = False) -> types.ModuleType:
    result = types.ModuleType(name)
    if package:
        result.__path__ = []  # type: ignore[attr-defined]
    sys.modules[name] = result
    return result


def _load_source(name: str, path: Path) -> types.ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load pinned upstream source: {path}")
    result = importlib.util.module_from_spec(spec)
    sys.modules[name] = result
    spec.loader.exec_module(result)
    return result


def _install_import_shims(upstream: Path) -> tuple[types.ModuleType, types.ModuleType, types.ModuleType, types.ModuleType]:
    """Load upstream model classes while stubbing unused model-level imports."""
    for name in ("wan", "wan.modules", "wan.modules.s2v", "wan.distributed"):
        _module(name, package=True)

    diffusers = _module("diffusers", package=True)
    diffusers.__path__ = []  # type: ignore[attr-defined]
    diffusers_config = _module("diffusers.configuration_utils")
    diffusers_config.ConfigMixin = type("ConfigMixin", (), {})
    diffusers_config.register_to_config = lambda fn: fn
    diffusers.models = _module("diffusers.models", package=True)
    modeling_utils = _module("diffusers.models.modeling_utils")
    modeling_utils.ModelMixin = nn.Module
    attention_module = _module("diffusers.models.attention")

    # Extract, rather than retype, the exact pinned Diffusers AdaLayerNorm class.
    diffusers_path = upstream / "diffusers-normalization-v0.31.0.py"
    tree = ast.parse(diffusers_path.read_text())
    ada_node = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "AdaLayerNorm")
    ada_tree = ast.Module(body=[ada_node], type_ignores=[])
    ada_namespace: dict[str, Any] = {"torch": torch, "nn": nn, "Optional": Any}
    exec(compile(ada_tree, str(diffusers_path), "exec"), ada_namespace)
    attention_module.AdaLayerNorm = ada_namespace["AdaLayerNorm"]

    seq_parallel = _module("wan.distributed.sequence_parallel")
    seq_parallel.distributed_attention = lambda *args, **kwargs: None
    seq_parallel.gather_forward = lambda x, **kwargs: x
    seq_parallel.get_rank = lambda: 0
    seq_parallel.get_world_size = lambda: 1

    # The model-level flash entry point is replaced after loading by the explicit
    # SDPA adapter below. No flash-attention dependency is imported here.
    attention_stub = _module("wan.modules.attention")
    attention_stub.flash_attention = lambda *args, **kwargs: None

    model = _load_source("wan.modules.model", upstream / "wan_modules_model.py")

    # These classes are unused in a single transformer-block fixture. Their
    # imports are stubbed so the actual block/audio source can be imported.
    auxi = _module("wan.modules.s2v.auxi_blocks")
    auxi.MotionEncoder_tc = type("MotionEncoder_tc", (nn.Module,), {})
    motioner = _module("wan.modules.s2v.motioner")
    motioner.FramePackMotioner = type("FramePackMotioner", (nn.Module,), {})
    motioner.MotionerTransformers = type("MotionerTransformers", (nn.Module,), {})

    s2v_utils = _load_source("wan.modules.s2v.s2v_utils", upstream / "wan_modules_s2v_s2v_utils.py")
    audio = _load_source("wan.modules.s2v.audio_utils", upstream / "wan_modules_s2v_audio_utils.py")
    s2v = _load_source("wan.modules.s2v.s2v_model", upstream / "wan_modules_s2v_model_s2v.py")
    return model, audio, s2v, s2v_utils


def _flat(tensor: torch.Tensor) -> list[float]:
    return tensor.detach().to(device="cpu", dtype=torch.float32).contiguous().view(-1).tolist()


def _capture(hooks: list[Any], module: nn.Module, key: str, destination: dict[str, torch.Tensor], *, input: bool = False) -> None:
    def save(_module: nn.Module, inputs: tuple[Any, ...], output: Any = None) -> None:
        value = inputs[0] if input else output
        if isinstance(value, torch.Tensor):
            destination[key] = value.detach().clone()

    hooks.append(module.register_forward_hook(save) if not input else module.register_forward_pre_hook(save))


def generate(upstream: Path, seed: int = 7031) -> dict[str, Any]:
    _verify_sources(upstream)
    if not torch.backends.mps.is_available():
        raise RuntimeError("MPS is unavailable; refusing to use CPU neural fallback")
    device = torch.device("mps")
    torch.manual_seed(seed)
    torch.mps.manual_seed(seed)
    torch.set_num_threads(1)
    model_module, audio_module, s2v_module, s2v_utils = _install_import_shims(upstream)

    b, seq, ctx_len, dim, heads, ffn_dim = 2, 12, 3, 32, 2, 64
    video_frames, spatial, audio_tokens, split = 2, 4, 5, 8
    head_dim = dim // heads
    block = s2v_module.WanS2VAttentionBlock(
        dim=dim,
        ffn_dim=ffn_dim,
        num_heads=heads,
        window_size=(-1, -1),
        qk_norm=True,
        cross_attn_norm=True,
        eps=1e-6,
    )
    for name, parameter in block.named_parameters():
        if name.endswith("norm_q.weight") or name.endswith("norm_k.weight") or name.endswith("norm3.weight"):
            nn.init.uniform_(parameter, 0.8, 1.2)
        elif name.endswith("bias"):
            nn.init.normal_(parameter, mean=0.0, std=0.02)
        else:
            nn.init.normal_(parameter, mean=0.0, std=0.035)
    block = block.to(device=device, dtype=torch.float32).eval()

    injector = audio_module.AudioInjector_WAN(
        all_modules=[block],
        all_modules_names=["root.transformer_blocks.0"],
        dim=dim,
        num_heads=heads,
        inject_layer=[0],
        enable_adain=True,
        adain_dim=dim,
    ).to(device=device, dtype=torch.float32).eval()
    for name, parameter in injector.named_parameters():
        if name.endswith("norm_q.weight") or name.endswith("norm_k.weight"):
            nn.init.uniform_(parameter, 0.8, 1.2)
        elif name.endswith("norm.weight"):
            nn.init.uniform_(parameter, 0.8, 1.2)
        elif name.endswith("bias"):
            nn.init.normal_(parameter, mean=0.0, std=0.02)
        else:
            nn.init.normal_(parameter, mean=0.0, std=0.035)

    # All token vectors are deterministic MPS tensors. Prefix coordinates reach
    # representative late-video positions; the suffix is a negative-time motion
    # grid whose source RoPE coefficients use the conjugate positive row.
    x = torch.randn((b, seq, dim), device=device, dtype=torch.float32) * 0.15
    modulation = torch.randn((b, 6, 2, dim), device=device, dtype=torch.float32) * 0.08
    context = torch.randn((b, ctx_len, dim), device=device, dtype=torch.float32) * 0.12
    local_audio = torch.randn((b, video_frames, audio_tokens, dim), device=device, dtype=torch.float32) * 0.12
    global_audio = torch.randn((b, video_frames, 1, dim), device=device, dtype=torch.float32) * 0.12

    # Each batch member has its own late-video grid and negative-time motion
    # grid. The first member retains the prior boundary positions; the second
    # probes distinct batch indexing at all three axes.
    batch_grids = [
        (
            ([255, 511, 1022], [257, 513, 1024], [2, 2, 2]),
            ([-30, 0, 0], [-29, 2, 2], [1, 2, 2]),
        ),
        (
            ([301, 700, 800], [303, 702, 802], [2, 2, 2]),
            ([-45, 10, 20], [-44, 12, 22], [1, 2, 2]),
        ),
    ]
    positions: list[list[list[int]]] = []
    for video_grid, motion_grid in batch_grids:
        sample_positions: list[list[int]] = []
        video_start, _, video_shape = video_grid
        for t in range(video_shape[0]):
            for h in range(video_shape[1]):
                for w in range(video_shape[2]):
                    sample_positions.append(
                        [video_start[0] + t, video_start[1] + h, video_start[2] + w]
                    )
        motion_start, _, motion_shape = motion_grid
        for t in range(motion_shape[0]):
            for h in range(motion_shape[1]):
                for w in range(motion_shape[2]):
                    sample_positions.append(
                        [motion_start[0] + t, motion_start[1] + h, motion_start[2] + w]
                    )
        if len(sample_positions) != seq:
            raise RuntimeError("position grids must cover every S2V sequence token")
        positions.append(sample_positions)

    for name, tensor in {
        "x": x,
        "modulation": modulation,
        "context": context,
        "local_audio": local_audio,
        "global_audio": global_audio,
    }.items():
        if torch.equal(tensor[0], tensor[1]):
            raise RuntimeError(f"fixture {name} must differ between batch members")

    # Exact upstream complex FP64 coefficients are precomputed on CPU as setup
    # data only. The adapter applies the real rotation to MPS activations in F32.
    freqs = torch.cat(
        [
            model_module.rope_params(1024, head_dim - 4 * (head_dim // 6)),
            model_module.rope_params(1024, 2 * (head_dim // 6)),
            model_module.rope_params(1024, 2 * (head_dim // 6)),
        ],
        dim=1,
    )
    position_tables = []
    for video_grid, motion_grid in batch_grids:
        grid_sizes = [
            [
                torch.tensor([video_grid[0]]),
                torch.tensor([video_grid[1]]),
                torch.tensor([video_grid[2]]),
            ],
            [
                torch.tensor([motion_grid[0]]),
                torch.tensor([motion_grid[1]]),
                torch.tensor([motion_grid[2]]),
            ],
        ]
        position_tables.append(
            s2v_utils.rope_precompute(
                torch.zeros((1, seq, heads, head_dim), dtype=torch.float32),
                grid_sizes,
                freqs,
            )
        )
    position_table = torch.cat(position_tables, dim=0)
    # Convert the setup-only official FP64/complex table to resident MPS F32
    # cosine/sine once. MPS doesn't implement complex128 arithmetic.
    rope_real = position_table.real.to(device=device, dtype=torch.float32).reshape(b, seq, heads, head_dim // 2)
    rope_imag = position_table.imag.to(device=device, dtype=torch.float32).reshape(b, seq, heads, head_dim // 2)

    checkpoint: dict[str, torch.Tensor] = {}
    hooks: list[Any] = []
    _capture(hooks, block.norm1, "norm1", checkpoint)
    _capture(hooks, block.self_attn, "self_mod", checkpoint, input=True)
    for name in ("q", "k", "v"):
        _capture(hooks, getattr(block.self_attn, name), f"self_{name}", checkpoint)
    _capture(hooks, block.self_attn.norm_q, "self_q_norm", checkpoint)
    _capture(hooks, block.self_attn, "self_projection", checkpoint)
    _capture(hooks, block.norm3, "self_residual", checkpoint, input=True)
    _capture(hooks, block.norm3, "cross_norm", checkpoint)
    for name in ("q", "k", "v"):
        _capture(hooks, getattr(block.cross_attn, name), f"cross_{name}", checkpoint)
    _capture(hooks, block.cross_attn, "cross_projection", checkpoint)
    _capture(hooks, block.norm2, "cross_residual", checkpoint, input=True)
    _capture(hooks, block.norm2, "ffn_norm", checkpoint)
    _capture(hooks, block.ffn, "ffn_mod", checkpoint, input=True)
    _capture(hooks, block.ffn[0], "ffn_in", checkpoint)
    _capture(hooks, block.ffn[1], "ffn_gelu", checkpoint)
    _capture(hooks, block.ffn[2], "ffn_out", checkpoint)

    sdpa_outputs: list[torch.Tensor] = []

    def sdpa_adapter(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, *args: Any, **kwargs: Any) -> torch.Tensor:
        # Wan's CUDA flash/distributed attention is replaced only at this kernel
        # boundary. No padding/mask applies to the tiny equal-length fixture.
        output = F.scaled_dot_product_attention(
            q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2),
            attn_mask=None, dropout_p=0.0, is_causal=False,
        ).transpose(1, 2).contiguous()
        sdpa_outputs.append(output.detach().clone())
        return output

    rope_calls = 0

    def rope_mps_adapter(q: torch.Tensor, grid_sizes: Any, coefficients: torch.Tensor) -> torch.Tensor:
        nonlocal rope_calls
        if q.device.type != "mps":
            raise RuntimeError(f"RoPE activation left MPS: {q.device}")
        real = rope_real[:, : q.shape[1], :, :]
        imag = rope_imag[:, : q.shape[1], :, :]
        pairs = q.reshape(b, seq, heads, head_dim // 2, 2)
        real_x, imag_x = pairs[..., 0], pairs[..., 1]
        result = torch.stack((real_x * real - imag_x * imag, real_x * imag + imag_x * real), dim=-1)
        result = result.reshape_as(q)
        checkpoint["self_q_rope" if rope_calls == 0 else "self_k_rope"] = result.detach().clone()
        rope_calls += 1
        return result

    s2v_module.flash_attention = sdpa_adapter
    model_module.flash_attention = sdpa_adapter
    s2v_module.rope_apply = rope_mps_adapter

    # Ensure no model parameters or fixture activations silently stay on CPU.
    for module in (block, injector):
        if any(parameter.device.type != "mps" for parameter in module.parameters()):
            raise RuntimeError("A neural parameter is not resident on MPS")
    with torch.inference_mode():
        block_out = block(
            x,
            [modulation, torch.tensor(split)],
            torch.tensor([seq] * b),
            torch.tensor([[2, 2, 2]] * b),
            position_table,
            context,
            torch.tensor([ctx_len] * b),
        )
    if len(sdpa_outputs) != 2 or rope_calls != 2:
        raise RuntimeError("Upstream S2V block did not execute the expected self/cross path")
    checkpoint["self_attention"] = sdpa_outputs[0]
    checkpoint["cross_attention"] = sdpa_outputs[1]
    checkpoint["output"] = block_out.detach().clone()
    # The hidden state after the self residual is the input to norm3; norm2's
    # pre-hook records the state after context cross attention.

    # Capture the exact AdaLayerNorm operations instantiated by upstream's
    # AudioInjector_WAN. This is the real after_transformer_block implementation.
    ada = injector.injector_adain_layers[0]
    _capture(hooks, ada.silu, "global_silu", checkpoint)
    _capture(hooks, ada.norm, "audio_norm", checkpoint)
    _capture(hooks, ada.linear, "adain_projection", checkpoint)
    _capture(hooks, ada, "adain_video", checkpoint)
    attention = injector.injector[0]
    for name in ("q", "k", "v"):
        _capture(hooks, getattr(attention, name), f"audio_{name}", checkpoint)
    _capture(hooks, attention, "audio_projection", checkpoint)
    gathered = block_out[:, : split].reshape(b * video_frames, spatial, dim)
    checkpoint["gathered_video"] = gathered.detach().clone()

    model_obj = types.SimpleNamespace(
        audio_injector=injector,
        merged_audio_emb=local_audio,
        audio_emb_global=global_audio,
        original_seq_len=split,
        use_context_parallel=False,
        enbale_adain=True,
        adain_mode="attn_norm",
    )
    # Invoke the actual upstream after-block hook. Its residual is separately
    # checked to preserve all reference/motion suffix values.
    with torch.inference_mode():
        audio_out = s2v_module.WanModel_S2V.after_transformer_block(model_obj, 0, block_out)
    if len(sdpa_outputs) != 3:
        raise RuntimeError("Upstream audio injector did not execute its attention path")
    checkpoint["audio_attention"] = sdpa_outputs[2]
    if not torch.equal(audio_out[:, split:], block_out[:, split:]):
        raise RuntimeError("Upstream audio hook unexpectedly modified the suffix")
    checkpoint["audio_output"] = audio_out.detach().clone()

    for hook in hooks:
        hook.remove()

    # Export only source-derived model state and actual checkpoint activations.
    weights: dict[str, torch.Tensor] = {}
    weights["block_modulation"] = block.modulation.detach().reshape(6, dim)
    weights["norm3_weight"] = block.norm3.weight.detach()
    weights["norm3_bias"] = block.norm3.bias.detach()
    for prefix, module in (("self", block.self_attn), ("cross", block.cross_attn)):
        for name in ("q", "k", "v", "o"):
            weights[f"{prefix}_{name}_weight"] = getattr(module, name).weight.detach()
            weights[f"{prefix}_{name}_bias"] = getattr(module, name).bias.detach()
        weights[f"{prefix}_norm_q_weight"] = module.norm_q.weight.detach()
        weights[f"{prefix}_norm_k_weight"] = module.norm_k.weight.detach()
    weights["ffn_in_weight"] = block.ffn[0].weight.detach()
    weights["ffn_in_bias"] = block.ffn[0].bias.detach()
    weights["ffn_out_weight"] = block.ffn[2].weight.detach()
    weights["ffn_out_bias"] = block.ffn[2].bias.detach()
    adain = injector.injector_adain_layers[0]
    weights["adain_weight"] = adain.linear.weight.detach()
    weights["adain_bias"] = adain.linear.bias.detach()
    for name in ("q", "k", "v", "o"):
        weights[f"audio_{name}_weight"] = getattr(attention, name).weight.detach()
        weights[f"audio_{name}_bias"] = getattr(attention, name).bias.detach()
    weights["audio_q_norm_weight"] = attention.norm_q.weight.detach()
    weights["audio_k_norm_weight"] = attention.norm_k.weight.detach()

    source_hashes = dict(SOURCE_SHA256)
    inputs = {
        "x": x,
        "modulation": modulation,
        "positions": torch.tensor(positions, dtype=torch.float32, device=device),
        "context": context,
        "local_audio": local_audio,
        "global_audio": global_audio,
    }
    expected = {name: value for name, value in checkpoint.items() if name in {
        "norm1", "self_mod", "self_q", "self_k", "self_v", "self_q_norm", "self_q_rope", "self_k_rope",
        "self_attention", "self_projection", "self_residual", "cross_norm", "cross_q", "cross_k", "cross_v",
        "cross_attention", "cross_projection", "cross_residual", "ffn_norm", "ffn_mod", "ffn_in", "ffn_gelu",
        "ffn_out", "output", "gathered_video", "audio_norm", "global_silu", "adain_projection", "adain_video",
        "audio_q", "audio_k", "audio_v", "audio_attention", "audio_projection", "audio_output",
    }}
    missing = sorted(set((
        "norm1", "self_mod", "self_q", "self_k", "self_v", "self_q_norm", "self_q_rope", "self_k_rope", "self_attention",
        "self_projection", "self_residual", "cross_norm", "cross_q", "cross_k", "cross_v", "cross_attention",
        "cross_projection", "cross_residual", "ffn_norm", "ffn_mod", "ffn_in", "ffn_gelu", "ffn_out", "output",
        "gathered_video", "audio_norm", "global_silu", "adain_projection", "adain_video", "audio_q", "audio_k",
        "audio_v", "audio_attention", "audio_projection", "audio_output",
    )) - set(expected))
    if missing:
        raise RuntimeError(f"Upstream hooks did not capture checkpoints: {missing}")

    return {
        "upstream": {
            "repo": "https://github.com/Wan-Video/Wan2.2",
            "revision": REVISION,
            "diffusers_version": DIFFUSERS_VERSION,
            "source_sha256": source_hashes,
            "oracle_device": "mps",
            "fallback_disabled": True,
            "attention_adapter": "torch.nn.functional.scaled_dot_product_attention replacing CUDA-only flash/distributed backend; upstream module forward and projections remain active",
            "rope_adapter": "official rope_precompute FP64 complex CPU setup table called per batch member to preserve its per-sample sequence buckets; split-real F32 coefficient application on MPS",
            "seed": seed,
        },
        "config": {
            "batches": b,
            "seq": seq,
            "context_tokens": ctx_len,
            "dim": dim,
            "heads": heads,
            "ffn_dim": ffn_dim,
            "split": split,
            "theta": 10000.0,
            "video_frames": video_frames,
            "spatial_tokens": spatial,
            "audio_tokens": audio_tokens,
            "audio_attention_query_rows_per_frame": spatial,
        },
        "inputs": {name: _flat(value) for name, value in inputs.items()},
        "weights": {name: _flat(value) for name, value in weights.items()},
        "expected": {name: _flat(value) for name, value in expected.items()},
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--upstream", type=Path, default=Path(os.environ.get("WAN_S2V_UPSTREAM_DIR", DEFAULT_UPSTREAM)))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=7031)
    args = parser.parse_args()
    result = generate(args.upstream, args.seed)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, separators=(",", ":")) + "\n")
    print(json.dumps({"output": str(args.output), "checkpoints": len(result["expected"]), "source_revision": REVISION}))


if __name__ == "__main__":
    main()
