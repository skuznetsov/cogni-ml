#!/usr/bin/env python3
"""Stream official Wan2.2 S2V weights through a tiny-spatial pinned-MPS oracle.

The checkpoint is never converted into a resident CPU F32 model. Base modules
are uploaded from their original BF16 tensors and converted to F32 on MPS. The
official transformer blocks and audio injectors are streamed one layer at a
time; an oracle-only MPS synchronization retires each layer's weights before
the next layer is loaded. Activations stay on MPS throughout the forward. The
fixture stores input values and an explicit source-to-native binding contract
in JSON, while F32 expected stage traces live in a sibling safetensors file.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any

os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "0"

import torch
import torch.nn as nn
import torch.nn.functional as F
from safetensors import safe_open
from safetensors.torch import save_file

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from wan_s2v_reference import (  # noqa: E402
    CHECKPOINT_REPO,
    CHECKPOINT_REVISION,
    DEFAULT_UPSTREAM,
    REVISION,
    SOURCE_SHA256,
    _install_import_shims,
    _verify_sources,
)
from wan_s2v_transformer_reference import position_groups  # noqa: E402


OFFICIAL = {
    "batches": 1,
    "context_tokens": 3,
    "dim": 5120,
    "heads": 40,
    "ffn_dim": 13824,
    "official_layers": 40,
    "video_frames": 2,
    "spatial_tokens": 16,
    "latent_height": 8,
    "latent_width": 8,
    "official_injection_ids": [0, 4, 8, 12, 16, 20, 24, 27, 30, 33, 36, 39],
    "theta": 10000.0,
    "text_dim": 4096,
    "text_tokens": 3,
    "freq_dim": 256,
    "in_dim": 16,
    "out_dim": 16,
    "audio_dim": 1024,
    "patch_tile_rows": 4,
    "motion_frames": 5,
}
MOTIONER_SOURCE = "wan_modules_s2v_motioner.py"
MOTIONER_SHA256 = "db348a3ca0dfb22122a78aebdcf162ba76d4d4e79b7b763c684d819279ac957a"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def flat(tensor: torch.Tensor) -> list[float]:
    return tensor.detach().to(device="cpu", dtype=torch.float32).contiguous().reshape(-1).tolist()


def checkpoint_index(path: Path) -> tuple[Path, dict[str, str]]:
    if path.is_dir():
        index_path = path / "diffusion_pytorch_model.safetensors.index.json"
    elif path.name.endswith(".index.json"):
        index_path = path
    else:
        raise ValueError("--checkpoint must be a Wan safetensors directory or its .index.json")
    if not index_path.is_file():
        raise FileNotFoundError(f"Wan checkpoint index is missing: {index_path}")
    raw = json.loads(index_path.read_text())
    weight_map = raw.get("weight_map")
    if not isinstance(weight_map, dict) or not weight_map:
        raise ValueError(f"Wan checkpoint index has no weight_map: {index_path}")
    return index_path, weight_map


def official_bindings(config: dict[str, Any], weight_map: dict[str, str]) -> list[dict[str, Any]]:
    """Mirror WanS2VTransformer.checkpoint_bindings without reading payloads."""
    dim = config["dim"]
    ffn = config["ffn_dim"]
    bindings: list[dict[str, Any]] = []

    def add(target: str, source: str, shape: tuple[int, ...]) -> None:
        shard = weight_map.get(source)
        if shard is None:
            raise KeyError(f"Official checkpoint is missing bound source tensor {source}")
        bindings.append({"target": target, "source": source, "shape": list(shape),
                         "dtype": "BF16", "shard": shard})

    injections = {layer: index for index, layer in enumerate(config["injection_ids"])}
    for layer in range(config["layers"]):
        block = f"blocks.{layer}"
        target = f"block.{layer}."
        add(target + "block_modulation", block + ".modulation", (1, 6, dim))
        add(target + "norm3_weight", block + ".norm3.weight", (dim,))
        add(target + "norm3_bias", block + ".norm3.bias", (dim,))
        for attention in ("self", "cross"):
            source_prefix = f"{block}.{attention}_attn"
            for name in ("q", "k", "v", "o"):
                add(target + f"{attention}_{name}_weight", f"{source_prefix}.{name}.weight", (dim, dim))
                add(target + f"{attention}_{name}_bias", f"{source_prefix}.{name}.bias", (dim,))
            add(target + f"{attention}_norm_q_weight", source_prefix + ".norm_q.weight", (dim,))
            add(target + f"{attention}_norm_k_weight", source_prefix + ".norm_k.weight", (dim,))
        add(target + "ffn_in_weight", block + ".ffn.0.weight", (ffn, dim))
        add(target + "ffn_in_bias", block + ".ffn.0.bias", (ffn,))
        add(target + "ffn_out_weight", block + ".ffn.2.weight", (dim, ffn))
        add(target + "ffn_out_bias", block + ".ffn.2.bias", (dim,))

        if layer in injections:
            injector = injections[layer]
            audio_target = f"audio.{injector}."
            adain_source = f"audio_injector.injector_adain_layers.{injector}.linear"
            add(audio_target + "adain_weight", adain_source + ".weight", (2 * dim, dim))
            add(audio_target + "adain_bias", adain_source + ".bias", (2 * dim,))
            audio_source = f"audio_injector.injector.{injector}"
            for name in ("q", "k", "v", "o"):
                add(audio_target + f"audio_{name}_weight", f"{audio_source}.{name}.weight", (dim, dim))
                add(audio_target + f"audio_{name}_bias", f"{audio_source}.{name}.bias", (dim,))
            add(audio_target + "audio_q_norm_weight", audio_source + ".norm_q.weight", (dim,))
            add(audio_target + "audio_k_norm_weight", audio_source + ".norm_k.weight", (dim,))

    def extra(source: str, shape: tuple[int, ...]) -> None:
        add("model." + source, source, shape)

    extra("patch_embedding.weight", (dim, config["in_dim"], 1, 2, 2))
    extra("patch_embedding.bias", (dim,))
    extra("text_embedding.0.weight", (dim, config["text_dim"]))
    extra("text_embedding.0.bias", (dim,))
    extra("text_embedding.2.weight", (dim, dim))
    extra("text_embedding.2.bias", (dim,))
    extra("time_embedding.0.weight", (dim, config["freq_dim"]))
    extra("time_embedding.0.bias", (dim,))
    extra("time_embedding.2.weight", (dim, dim))
    extra("time_embedding.2.bias", (dim,))
    extra("time_projection.1.weight", (6 * dim, dim))
    extra("time_projection.1.bias", (6 * dim,))
    extra("cond_encoder.weight", (dim, config["in_dim"], 1, 2, 2))
    extra("cond_encoder.bias", (dim,))
    extra("head.head.weight", (config["out_dim"] * 4, dim))
    extra("head.head.bias", (config["out_dim"] * 4,))
    extra("head.modulation", (1, 2, dim))
    extra("frame_packer.proj.weight", (dim, config["in_dim"], 1, 2, 2))
    extra("frame_packer.proj.bias", (dim,))
    extra("frame_packer.proj_2x.weight", (dim, config["in_dim"], 2, 4, 4))
    extra("frame_packer.proj_2x.bias", (dim,))
    extra("frame_packer.proj_4x.weight", (dim, config["in_dim"], 4, 8, 8))
    extra("frame_packer.proj_4x.bias", (dim,))
    extra("trainable_cond_mask.weight", (3, dim))
    if len(bindings) != config["layers"] * 27 + len(config["injection_ids"]) * 12 + 24:
        raise RuntimeError(f"Unexpected native transformer binding count: {len(bindings)}")
    return bindings


def verify_contract(index_path: Path, bindings: list[dict[str, Any]]) -> None:
    by_shard: dict[str, list[dict[str, Any]]] = {}
    for binding in bindings:
        by_shard.setdefault(binding["shard"], []).append(binding)
    for shard_name, entries in by_shard.items():
        shard = index_path.parent / shard_name
        if not shard.is_file():
            raise FileNotFoundError(f"Wan checkpoint shard is missing: {shard}")
        with safe_open(str(shard), framework="pt", device="cpu") as handle:
            for entry in entries:
                tensor_slice = handle.get_slice(entry["source"])
                if tensor_slice.get_shape() != entry["shape"]:
                    raise RuntimeError(
                        f"Checkpoint shape mismatch for {entry['source']}: "
                        f"{tensor_slice.get_shape()} != {entry['shape']}"
                    )
                if tensor_slice.get_dtype() != "BF16":
                    raise RuntimeError(f"Strict real-stack oracle requires original BF16 {entry['source']}")


def load_prefixed_module(
    module: nn.Module,
    prefixes: tuple[str, ...],
    weight_map: dict[str, str],
    checkpoint_root: Path,
    device: torch.device,
) -> None:
    """Assign one source module's BF16 state, upload BF16, then convert on MPS."""
    source_names = [name for name in weight_map if any(name.startswith(prefix) for prefix in prefixes)]
    if not source_names:
        raise RuntimeError(f"No checkpoint tensors found for source prefix {prefixes}")
    state: dict[str, torch.Tensor] = {}
    for shard_name in sorted({weight_map[name] for name in source_names}):
        shard = checkpoint_root / shard_name
        names = [name for name in source_names if weight_map[name] == shard_name]
        with safe_open(str(shard), framework="pt", device="cpu") as handle:
            for source in names:
                matching = [prefix for prefix in prefixes if source.startswith(prefix)]
                if len(matching) != 1:
                    raise RuntimeError(f"Ambiguous checkpoint prefix for {source}")
                target = source[len(matching[0]):]
                tensor = handle.get_tensor(source)
                if tensor.dtype is not torch.bfloat16:
                    raise RuntimeError(f"Expected original BF16 checkpoint storage for {source}: {tensor.dtype}")
                state[target] = tensor

    expected = set(module.state_dict().keys())
    actual = set(state)
    if actual != expected:
        raise RuntimeError(
            f"Module checkpoint keys differ for {prefixes}: "
            f"missing={sorted(expected - actual)}, extra={sorted(actual - expected)}"
        )
    module.load_state_dict(state, strict=True, assign=True)
    del state
    gc.collect()
    module.to(device=device, dtype=torch.bfloat16)
    module.to(dtype=torch.float32)
    for name, value in module.state_dict().items():
        if value.device.type != "mps" or value.dtype != torch.float32:
            raise RuntimeError(f"Checkpoint module was not converted to MPS F32: {prefixes}::{name}")


def retire_module(module: nn.Module) -> None:
    module.to_empty(device=torch.device("meta"))
    gc.collect()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True,
                        help="Official Wan safetensors directory or index JSON")
    parser.add_argument("--upstream", type=Path, default=DEFAULT_UPSTREAM)
    parser.add_argument("--layers", type=int, choices=(1, 2, 40), default=40,
                        help="Run an official prefix; 40 exercises every distinct block and all injectors")
    args = parser.parse_args()
    _verify_sources(args.upstream)
    motioner_path = args.upstream / MOTIONER_SOURCE
    if sha256(motioner_path) != MOTIONER_SHA256:
        raise RuntimeError(f"Pinned FramePack source hash mismatch: {motioner_path}")
    if not torch.backends.mps.is_available():
        raise RuntimeError("MPS is required; CPU neural fallback is forbidden")

    index_path, weight_map = checkpoint_index(args.checkpoint)
    checkpoint_root = index_path.parent
    config = dict(OFFICIAL)
    config["layers"] = args.layers
    config["injection_ids"] = [layer for layer in OFFICIAL["official_injection_ids"] if layer < args.layers]
    bindings = official_bindings(config, weight_map)
    verify_contract(index_path, bindings)

    torch.manual_seed(7107)
    torch.mps.manual_seed(7107)
    torch.set_num_threads(1)
    device = torch.device("mps")
    model_module, _audio_module, s2v_module, s2v_utils = _install_import_shims(args.upstream)

    class FixtureCausalAudioEncoder(nn.Module):
        def __init__(self, *_args: Any, **_kwargs: Any) -> None:
            super().__init__()

        def forward(self, *_args: Any, **_kwargs: Any) -> torch.Tensor:
            raise RuntimeError("CausalAudioEncoder is outside this streamed transformer oracle")

    s2v_module.CausalAudioEncoder = FixtureCausalAudioEncoder
    with torch.device("meta"):
        model = s2v_module.WanModel_S2V(
            cond_dim=config["in_dim"], audio_dim=config["audio_dim"], num_audio_token=4,
            enable_adain=True, audio_inject_layers=config["injection_ids"], zero_init=False,
            zero_timestep=True, enable_motioner=False, enable_framepack=False,
            patch_size=(1, 2, 2), text_len=config["text_tokens"], in_dim=config["in_dim"],
            dim=config["dim"], ffn_dim=config["ffn_dim"], freq_dim=config["freq_dim"],
            text_dim=config["text_dim"], out_dim=config["out_dim"], num_heads=config["heads"],
            num_layers=config["layers"], window_size=(-1, -1), qk_norm=True,
            cross_attn_norm=True, eps=1e-6,
        )
        frame_packer = nn.Module()
        frame_packer.add_module("proj", nn.Conv3d(config["in_dim"], config["dim"], (1, 2, 2), stride=(1, 2, 2)))
        frame_packer.add_module("proj_2x", nn.Conv3d(config["in_dim"], config["dim"], (2, 4, 4), stride=(2, 4, 4)))
        frame_packer.add_module("proj_4x", nn.Conv3d(config["in_dim"], config["dim"], (4, 8, 8), stride=(4, 8, 8)))
        model.frame_packer = frame_packer
    model.eval()

    # model.freqs is a plain setup tensor. Build it outside the meta-device
    # context with the exact Wan rotary frequency split used by the pinned code.
    head_dim = config["dim"] // config["heads"]
    model.freqs = torch.cat([
        model_module.rope_params(1024, head_dim - 4 * (head_dim // 6)),
        model_module.rope_params(1024, 2 * (head_dim // 6)),
        model_module.rope_params(1024, 2 * (head_dim // 6)),
    ], dim=1)

    # Load shared source modules from original BF16 tensors, upload them to MPS,
    # and expand to F32 only after they are resident there.
    for module, prefixes in (
        (model.patch_embedding, ("patch_embedding.",)),
        (model.text_embedding, ("text_embedding.",)),
        (model.time_embedding, ("time_embedding.",)),
        (model.time_projection, ("time_projection.",)),
        (model.cond_encoder, ("cond_encoder.",)),
        (model.head, ("head.",)),
        (model.trainable_cond_mask, ("trainable_cond_mask.",)),
        (model.frame_packer, ("frame_packer.",)),
    ):
        load_prefixed_module(module, prefixes, weight_map, checkpoint_root, device)

    def randn(shape: tuple[int, ...], scale: float) -> torch.Tensor:
        return torch.randn(shape, device=device, dtype=torch.float32) * scale

    b = config["batches"]
    channels = config["in_dim"]
    frames = config["video_frames"]
    height, width = config["latent_height"], config["latent_width"]
    video = randn((b, channels, frames, height, width), 0.12)
    condition = randn((b, channels, frames, height, width), 0.12)
    no_pose_condition = condition.detach().clone()
    reference = randn((b, channels, 1, height, width), 0.12)
    motion = randn((b, channels, config["motion_frames"], height, width), 0.12)
    context = randn((b, config["text_tokens"], config["text_dim"]), 0.12)
    timestep_value = 1.25
    timestep = torch.tensor([timestep_value], device=device, dtype=torch.float32)
    local_audio = randn((b, frames, 5, config["dim"]), 0.12)
    global_audio = randn((b, frames, 1, config["dim"]), 0.12)

    with torch.inference_mode():
        video_tokens = model.patch_embedding(video).flatten(2).transpose(1, 2)
        video_tokens = video_tokens + model.trainable_cond_mask.weight[0].reshape(1, 1, -1)
        reference_tokens = model.patch_embedding(reference).flatten(2).transpose(1, 2)
        reference_tokens = reference_tokens + model.trainable_cond_mask.weight[1].reshape(1, 1, -1)

        padded = torch.zeros((b, channels, 19, height, width), device=device, dtype=torch.float32)
        padded[:, :, -config["motion_frames"]:, :, :] = motion
        packer = model.frame_packer
        recent = packer.proj(padded[:, :, 18:19]).flatten(2).transpose(1, 2)
        middle = packer.proj_2x(padded[:, :, 16:18]).flatten(2).transpose(1, 2)
        far = packer.proj_4x(padded[:, :, :16]).flatten(2).transpose(1, 2)
        motion_tokens = torch.cat((recent, middle, far), dim=1)
        motion_tokens = motion_tokens + model.trainable_cond_mask.weight[2].reshape(1, 1, -1)
        sequence = torch.cat((video_tokens, reference_tokens, motion_tokens), dim=1)
        split = frames * config["spatial_tokens"]
        sequence[:, :split, :] += model.cond_encoder(condition).flatten(2).transpose(1, 2)
        normal_initial = sequence.detach().clone()

        zero_condition = torch.zeros_like(condition)
        drop_sequence = torch.cat((video_tokens, reference_tokens), dim=1)
        drop_sequence[:, :split, :] += model.cond_encoder(zero_condition).flatten(2).transpose(1, 2)
        drop_initial = drop_sequence.detach().clone()

        projected_context = model.text_embedding(context)
        timestep_setup = torch.tensor([timestep_value, 0.0], device="cpu", dtype=torch.float32)
        time_input = model_module.sinusoidal_embedding_1d(config["freq_dim"], timestep_setup).float().to(device)
        time_all = model.time_embedding(time_input)
        projected_time = model.time_projection(time_all).reshape(b + 1, 6, config["dim"])
        time_embedding = time_all[:b]
        modulation = torch.stack((projected_time[:b], projected_time[b:].expand(b, -1, -1)), dim=2)

        oracle_model = type("Oracle", (), {"s2v_utils": s2v_utils, "model": model})()
        coefficients, positions = position_groups(oracle_model, config)
        drop_coefficients, drop_positions = position_groups(oracle_model, config, include_framepack=False)
        rope_state: dict[str, torch.Tensor] = {}

        def rope_mps_adapter(x: torch.Tensor, _grid_sizes: Any, _freqs: torch.Tensor) -> torch.Tensor:
            if x.device.type != "mps":
                raise RuntimeError(f"Wan real-stack activation left MPS: {x.device}")
            rope_real = rope_state["real"]
            rope_imag = rope_state["imag"]
            pairs = x.reshape(b, x.shape[1], config["heads"], head_dim // 2, 2)
            real_x, imag_x = pairs[..., 0], pairs[..., 1]
            result = torch.stack(
                (real_x * rope_real - imag_x * rope_imag,
                 real_x * rope_imag + imag_x * rope_real), dim=-1,
            )
            return result.reshape_as(x)

        def sdpa_adapter(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
                         *_args: Any, **_kwargs: Any) -> torch.Tensor:
            qh, kh, vh = (item.transpose(1, 2).contiguous() for item in (q, k, v))
            if any(item.device.type != "mps" or not item.is_contiguous() for item in (qh, kh, vh)):
                raise RuntimeError("Wan real-stack attention requires contiguous MPS Q/K/V")
            output = F.scaled_dot_product_attention(qh, kh, vh, dropout_p=0.0, is_causal=False)
            manual_scores = torch.matmul(qh, kh.transpose(-2, -1)) / (qh.shape[-1] ** 0.5)
            manual = torch.matmul(torch.softmax(manual_scores, dim=-1), vh)
            tolerance = 2e-5 + 2e-5 * manual.abs()
            if not torch.isfinite(output).all() or torch.any((output - manual).abs() > tolerance):
                raise RuntimeError("MPS SDPA disagrees with the independent F32 attention equation")
            return output.transpose(1, 2).contiguous()

        model_module.flash_attention = sdpa_adapter
        s2v_module.flash_attention = sdpa_adapter
        model_module.rope_apply = rope_mps_adapter
        s2v_module.rope_apply = rope_mps_adapter
        model.original_seq_len = split
        model.merged_audio_emb = local_audio
        model.audio_emb_global = global_audio
        grid_sizes = torch.tensor([[frames, height // 2, width // 2]], dtype=torch.long)
        seq_lens = torch.tensor([sequence.shape[1]], dtype=torch.long)
        drop_seq_lens = torch.tensor([drop_sequence.shape[1]], dtype=torch.long)
        trace_tensors: dict[str, torch.Tensor] = {
            "normal.initial_sequence": normal_initial,
            "drop.initial_sequence": drop_initial,
            "normal.projected_context": projected_context.detach().clone(),
            "drop.projected_context": projected_context.detach().clone(),
            "normal.time_embedding": time_embedding.detach().clone(),
            "drop.time_embedding": time_embedding.detach().clone(),
            "normal.modulation": modulation.detach().clone(),
            "drop.modulation": modulation.detach().clone(),
            "normal.zero_condition": zero_condition.detach().clone(),
        }

        # Retain independent stage outputs on MPS while the exact source block
        # and injector weights are streamed. The sole per-layer sync protects
        # the oracle's parameter lifetime; no activation is moved to CPU here.
        normal_current = sequence
        drop_current = drop_sequence
        injection_index = 0
        for layer in range(config["layers"]):
            block = model.blocks[layer]
            load_prefixed_module(block, (f"blocks.{layer}.",), weight_map, checkpoint_root, device)
            is_injected = injection_index < len(config["injection_ids"]) and config["injection_ids"][injection_index] == layer
            injector = None
            adain = None
            if is_injected:
                injector = model.audio_injector.injector[injection_index]
                adain = model.audio_injector.injector_adain_layers[injection_index]
                load_prefixed_module(injector, (f"audio_injector.injector.{injection_index}.",),
                                     weight_map, checkpoint_root, device)
                load_prefixed_module(adain, (f"audio_injector.injector_adain_layers.{injection_index}.",),
                                     weight_map, checkpoint_root, device)

            rope_state["real"] = coefficients.real.to(device=device, dtype=torch.float32)
            rope_state["imag"] = coefficients.imag.to(device=device, dtype=torch.float32)
            normal_current = block(
                normal_current, [modulation, torch.tensor(split)], seq_lens,
                grid_sizes, coefficients, projected_context, None,
            )
            trace_tensors[f"normal.block.{layer:02d}"] = normal_current.detach().clone()
            if is_injected:
                normal_current = model.after_transformer_block(layer, normal_current)
                trace_tensors[f"normal.audio.{injection_index:02d}"] = normal_current.detach().clone()

            rope_state["real"] = drop_coefficients.real.to(device=device, dtype=torch.float32)
            rope_state["imag"] = drop_coefficients.imag.to(device=device, dtype=torch.float32)
            drop_current = block(
                drop_current, [modulation, torch.tensor(split)], drop_seq_lens,
                grid_sizes, drop_coefficients, projected_context, None,
            )
            trace_tensors[f"drop.block.{layer:02d}"] = drop_current.detach().clone()
            if is_injected:
                drop_current = model.after_transformer_block(layer, drop_current)
                trace_tensors[f"drop.audio.{injection_index:02d}"] = drop_current.detach().clone()

            torch.mps.synchronize()
            retire_module(block)
            if injector is not None and adain is not None:
                retire_module(injector)
                retire_module(adain)
                injection_index += 1

        if injection_index != len(config["injection_ids"]):
            raise RuntimeError("Not every selected audio injection was executed")
        prediction = model.head(normal_current[:, :split], time_embedding)
        latent = model.unpatchify(
            [prediction[0]], torch.tensor([[frames, height // 2, width // 2]], device="cpu"),
        )[0]
        drop_prediction = model.head(drop_current[:, :split], time_embedding)
        drop_latent = model.unpatchify(
            [drop_prediction[0]], torch.tensor([[frames, height // 2, width // 2]], device="cpu"),
        )[0]
        trace_tensors.update({
            "normal.prediction": prediction.detach().clone(),
            "normal.latent_output": latent.detach().clone(),
            "drop.prediction": drop_prediction.detach().clone(),
            "drop.latent_output": drop_latent.detach().clone(),
        })
        torch.mps.synchronize()

    traces_path = args.output.with_suffix(".traces.safetensors")
    traces_path.parent.mkdir(parents=True, exist_ok=True)
    cpu_traces: dict[str, torch.Tensor] = {}
    trace_manifest: list[dict[str, Any]] = []
    for name, value in trace_tensors.items():
        cpu_value = value.detach().to(device="cpu", dtype=torch.float32).contiguous()
        cpu_traces[name] = cpu_value
        trace_manifest.append({"name": name, "shape": list(cpu_value.shape), "dtype": "F32"})
        del value
    save_file(cpu_traces, str(traces_path))
    del cpu_traces
    gc.collect()

    fixture = {
        "upstream": {
            "repo_revision": REVISION,
            "source_sha256": SOURCE_SHA256,
            "motioner_source": MOTIONER_SOURCE,
            "motioner_source_sha256": MOTIONER_SHA256,
            "device": "mps",
            "weight_upload": "original BF16 tensor upload, then MPS F32 conversion",
            "neural_arithmetic": "F32",
            "fallback_disabled": True,
            "oracle_layer_syncs": config["layers"],
            "activation_cpu_bounce": False,
            "torch_version": torch.__version__,
        },
        "checkpoint": {
            "repo": CHECKPOINT_REPO,
            "revision": CHECKPOINT_REVISION,
            "index_file": index_path.name,
            "index_sha256": sha256(index_path),
            "weight_map_count": len(weight_map),
            "streamed_weight_dtype": "BF16",
            "payload_sha256": "not-computed; use repo revision, index digest, and native file-identity checks",
        },
        "config": config,
        "inputs": {
            "video_latents": flat(video),
            "condition_latents": flat(condition),
            "no_pose_condition_latents": flat(no_pose_condition),
            "reference_latents": flat(reference),
            "motion_latents": flat(motion),
            "raw_context": flat(context),
            "timestep": flat(timestep),
            "positions": [coordinate for position in positions for coordinate in position],
            "drop_positions": [coordinate for position in drop_positions for coordinate in position],
            "local_audio": flat(local_audio),
            "global_audio": flat(global_audio),
        },
        "weight_contract": bindings,
        "traces": {
            "file": traces_path.name,
            "sha256": sha256(traces_path),
            "bytes": traces_path.stat().st_size,
            "format": "safetensors-f32",
            "manifest": trace_manifest,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(fixture, allow_nan=False, separators=(",", ":")))
    print(json.dumps({
        "fixture": str(args.output.resolve()), "fixture_bytes": args.output.stat().st_size,
        "traces": str(traces_path.resolve()), "trace_bytes": traces_path.stat().st_size,
        "trace_sha256": fixture["traces"]["sha256"], "device": "mps",
        "arithmetic": "F32", "layers": config["layers"],
        "injections": len(config["injection_ids"]), "tokens": sequence.shape[1],
        "bindings": len(bindings),
    }))


if __name__ == "__main__":
    main()
