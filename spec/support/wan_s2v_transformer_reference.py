#!/usr/bin/env python3
"""Generate a tiny pinned-MPS full Wan S2V transformer oracle.

The fixture runs the actual pinned WanModel_S2V patch/text/time/block/head
modules, with exact FramePack Conv3d modules and setup RoPE coefficients. The
native spec compares every captured F32 value using the fixed 2e-5 absolute +
2e-5 relative gate. Both tiny transformer layers have distinct weights, and
both layers exercise AdaIN plus audio cross-attention injection.
"""

from __future__ import annotations

import argparse
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

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from wan_s2v_reference import DEFAULT_UPSTREAM, REVISION, SOURCE_SHA256, _install_import_shims, _verify_sources  # noqa: E402


TINY = {
    "batches": 1,
    "context_tokens": 3,
    "dim": 16,
    "heads": 2,
    "ffn_dim": 32,
    "layers": 2,
    "video_frames": 2,
    "spatial_tokens": 16,
    "latent_height": 8,
    "latent_width": 8,
    "injection_ids": [0, 1],
    "theta": 10000.0,
    "text_dim": 4,
    "text_tokens": 3,
    "freq_dim": 4,
    "in_dim": 4,
    "out_dim": 2,
    "audio_dim": 8,
    "patch_tile_rows": 4,
    "motion_frames": 5,
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def flat(tensor: torch.Tensor) -> list[float]:
    return tensor.detach().to(device="cpu", dtype=torch.float32).contiguous().reshape(-1).tolist()


def source_weights(model: nn.Module) -> tuple[dict[str, list[float]], dict[str, list[int]]]:
    state = dict(model.state_dict())
    selected = {
        name: tensor
        for name, tensor in state.items()
        if not name.startswith("casual_audio_encoder.")
    }
    if hasattr(model, "frame_packer"):
        selected.update(
            {f"frame_packer.{name}": tensor for name, tensor in model.frame_packer.state_dict().items()}
        )
    return (
        {name: flat(tensor) for name, tensor in selected.items()},
        {name: list(tensor.shape) for name, tensor in selected.items()},
    )


def position_groups(
    model: nn.Module, cfg: dict[str, Any], include_framepack: bool = True
) -> tuple[torch.Tensor, list[list[int]]]:
    """Build upstream position coefficients and their equivalent integer triples."""
    height, width = cfg["latent_height"], cfg["latent_width"]
    half_h, half_w = height // 2, width // 2

    def grid(origin: tuple[int, int, int], end: tuple[int, int, int], span: tuple[int, int, int]):
        return [torch.tensor([origin]), torch.tensor([end]), torch.tensor([span])]

    grids = [
        grid((0, 0, 0), (cfg["video_frames"], half_h, half_w), (cfg["video_frames"], half_h, half_w)),
        grid((30, 0, 0), (31, half_h, half_w), (1, half_h, half_w)),
    ]
    if include_framepack:
        grids.extend(
            (
                grid((-1, 0, 0), (0, half_h, half_w), (1, half_h, half_w)),
                grid((-3, 0, 0), (-2, height // 4, width // 4), (1, half_h, half_w)),
                grid((-19, 0, 0), (-15, height // 8, width // 8), (16, half_h, half_w)),
            )
        )

    positions: list[list[int]] = []
    for frame in range(cfg["video_frames"]):
        positions.extend([[frame, h, w] for h in range(half_h) for w in range(half_w)])
    positions.extend([[30, h, w] for h in range(half_h) for w in range(half_w)])
    if include_framepack:
        positions.extend([[-1, h, w] for h in range(half_h) for w in range(half_w)])
        middle_h = [0] if height // 4 == 1 else [0, half_h - 1]
        middle_w = [0] if width // 4 == 1 else [0, half_w - 1]
        positions.extend([[-3, h, w] for h in middle_h for w in middle_w])
        far_h = [0] if height // 8 == 1 else [0, half_h - 1]
        far_w = [0] if width // 8 == 1 else [0, half_w - 1]
        positions.extend([[-frame, h, w] for frame in (19, 14, 9, 4) for h in far_h for w in far_w])

    seq = len(positions)
    # rope_precompute receives attention-head activations: the last axis is
    # head_dim (not model dim). Its third axis is the head count, but rotary
    # coefficients are shared across heads. Keeping this source shape also
    # makes the frequency split match WanModel_S2V.freqs for tiny configs.
    head_dim = cfg["dim"] // cfg["heads"]
    dummy = torch.zeros((1, seq, cfg["heads"], head_dim), dtype=torch.float32)
    coefficients = model.s2v_utils.rope_precompute(dummy, grids, model.model.freqs)
    if coefficients.shape[1] != seq:
        raise RuntimeError(f"RoPE groups produced {coefficients.shape[1]} rows for {seq} positions")
    return coefficients, positions


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--upstream", type=Path, default=DEFAULT_UPSTREAM)
    args = parser.parse_args()
    _verify_sources(args.upstream)
    if not torch.backends.mps.is_available():
        raise RuntimeError("MPS is required; CPU neural fallback is forbidden")

    torch.manual_seed(7107)
    torch.mps.manual_seed(7107)
    torch.set_num_threads(1)
    device = torch.device("mps")
    model_module, _audio_module, s2v_module, s2v_utils = _install_import_shims(args.upstream)

    # This fixture drives the admitted transformer blocks and audio injectors
    # directly. CausalAudioEncoder has its own native fixture; constructing the
    # pinned Python encoder here would instantiate MotionEncoder_tc, which is
    # intentionally import-stubbed by _install_import_shims. Keep the unused
    # module parameter-free so it cannot add unrelated state to this oracle.
    class FixtureCausalAudioEncoder(nn.Module):
        def __init__(self, *_args: Any, **_kwargs: Any) -> None:
            super().__init__()

        def forward(self, *_args: Any, **_kwargs: Any) -> torch.Tensor:
            raise RuntimeError("CausalAudioEncoder is outside the transformer fixture")

    s2v_module.CausalAudioEncoder = FixtureCausalAudioEncoder
    cfg = TINY
    model = s2v_module.WanModel_S2V(
        cond_dim=cfg["in_dim"], audio_dim=cfg["audio_dim"], num_audio_token=4,
        enable_adain=True, audio_inject_layers=cfg["injection_ids"], zero_init=False,
        zero_timestep=True, enable_motioner=False, enable_framepack=False,
        patch_size=(1, 2, 2), text_len=cfg["text_tokens"], in_dim=cfg["in_dim"],
        dim=cfg["dim"], ffn_dim=cfg["ffn_dim"], freq_dim=cfg["freq_dim"],
        text_dim=cfg["text_dim"], out_dim=cfg["out_dim"], num_heads=cfg["heads"],
        num_layers=cfg["layers"], window_size=(-1, -1), qk_norm=True,
        cross_attn_norm=True, eps=1e-6,
    ).to(device=device, dtype=torch.float32).eval()
    frame_packer = nn.Module()
    frame_packer.add_module("proj", nn.Conv3d(cfg["in_dim"], cfg["dim"], (1, 2, 2), stride=(1, 2, 2)))
    frame_packer.add_module("proj_2x", nn.Conv3d(cfg["in_dim"], cfg["dim"], (2, 4, 4), stride=(2, 4, 4)))
    frame_packer.add_module("proj_4x", nn.Conv3d(cfg["in_dim"], cfg["dim"], (4, 8, 8), stride=(4, 8, 8)))
    model.frame_packer = frame_packer.to(device=device, dtype=torch.float32).eval()

    def randn(shape: tuple[int, ...], scale: float) -> torch.Tensor:
        return torch.randn(shape, device=device, dtype=torch.float32) * scale

    b, channels, frames = cfg["batches"], cfg["in_dim"], cfg["video_frames"]
    height, width = cfg["latent_height"], cfg["latent_width"]
    video = randn((b, channels, frames, height, width), 0.12)
    condition = randn((b, channels, frames, height, width), 0.12)
    reference = randn((b, channels, 1, height, width), 0.12)
    motion = randn((b, channels, cfg["motion_frames"], height, width), 0.12)
    context = randn((b, cfg["text_tokens"], cfg["text_dim"]), 0.12)
    timestep_value = 1.25
    # The runtime-facing native input remains resident on MPS. The reference
    # sinusoid is setup from this caller-owned scalar because PyTorch MPS does
    # not support the pinned source's required F64 outer product.
    timestep = torch.tensor([timestep_value], device=device, dtype=torch.float32)
    local_audio = randn((b, frames, 5, cfg["dim"]), 0.12)
    global_audio = randn((b, frames, 1, cfg["dim"]), 0.12)

    with torch.inference_mode():
        video_tokens = model.patch_embedding(video).flatten(2).transpose(1, 2)
        video_tokens = video_tokens + model.trainable_cond_mask.weight[0].reshape(1, 1, -1)
        reference_tokens = model.patch_embedding(reference).flatten(2).transpose(1, 2)
        reference_tokens = reference_tokens + model.trainable_cond_mask.weight[1].reshape(1, 1, -1)

        padded = torch.zeros((b, channels, 19, height, width), device=device, dtype=torch.float32)
        padded[:, :, -cfg["motion_frames"] :, :, :] = motion
        packer = model.frame_packer
        recent = packer.proj(padded[:, :, 18:19]).flatten(2).transpose(1, 2)
        middle = packer.proj_2x(padded[:, :, 16:18]).flatten(2).transpose(1, 2)
        far = packer.proj_4x(padded[:, :, :16]).flatten(2).transpose(1, 2)
        motion_tokens = torch.cat((recent, middle, far), dim=1)
        motion_tokens = motion_tokens + model.trainable_cond_mask.weight[2].reshape(1, 1, -1)
        sequence = torch.cat((video_tokens, reference_tokens, motion_tokens), dim=1)
        sequence[:, : video_tokens.shape[1], :] += model.cond_encoder(condition).flatten(2).transpose(1, 2)
        raw_sequence = sequence.detach().clone()

        projected_context = model.text_embedding(context)
        timesteps_setup = torch.tensor([timestep_value, 0.0], device="cpu", dtype=torch.float32)
        time_input = model_module.sinusoidal_embedding_1d(cfg["freq_dim"], timesteps_setup).float().to(device)
        time_all = model.time_embedding(time_input)
        projected_time = model.time_projection(time_all).reshape(b + 1, 6, cfg["dim"])
        time_embedding = time_all[:b]
        modulation = torch.stack((projected_time[:b], projected_time[b:].expand(b, -1, -1)), dim=2)

        coefficients, positions = position_groups(
            type("Oracle", (), {"s2v_utils": s2v_utils, "model": model})(), cfg
        )
        rope_real = coefficients.real.to(device=device, dtype=torch.float32)
        rope_imag = coefficients.imag.to(device=device, dtype=torch.float32)

        def rope_mps_adapter(x: torch.Tensor, _grid_sizes: Any, _freqs: torch.Tensor) -> torch.Tensor:
            if x.device.type != "mps":
                raise RuntimeError(f"RoPE activation left MPS: {x.device}")
            pairs = x.reshape(b, x.shape[1], cfg["heads"], cfg["dim"] // cfg["heads"] // 2, 2)
            real_x, imag_x = pairs[..., 0], pairs[..., 1]
            result = torch.stack(
                (real_x * rope_real - imag_x * rope_imag, real_x * rope_imag + imag_x * rope_real),
                dim=-1,
            )
            return result.reshape_as(x)

        def sdpa_adapter(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
                         *_args: Any, **_kwargs: Any) -> torch.Tensor:
            qh, kh, vh = (item.transpose(1, 2).contiguous() for item in (q, k, v))
            if any(item.device.type != "mps" or not item.is_contiguous() for item in (qh, kh, vh)):
                raise RuntimeError("Wan attention oracle requires contiguous MPS Q/K/V")
            output = F.scaled_dot_product_attention(qh, kh, vh, dropout_p=0.0, is_causal=False)
            manual_scores = torch.matmul(qh, kh.transpose(-2, -1)) / (qh.shape[-1] ** 0.5)
            manual = torch.matmul(torch.softmax(manual_scores, dim=-1), vh)
            tolerance = 2e-5 + 2e-5 * manual.abs()
            if not torch.isfinite(output).all() or torch.any((output - manual).abs() > tolerance):
                raise RuntimeError("MPS SDPA disagrees with the independent F32 attention equation")
            return output.transpose(1, 2).contiguous()

        model_module.flash_attention = sdpa_adapter
        s2v_module.flash_attention = sdpa_adapter
        s2v_module.rope_apply = rope_mps_adapter
        sequence_outputs = []
        audio_outputs = []
        split = frames * cfg["spatial_tokens"]
        grid_sizes = torch.tensor([[frames, height // 2, width // 2]])
        seq_lens = torch.tensor([sequence.shape[1]])
        model.original_seq_len = split
        model.merged_audio_emb = local_audio
        model.audio_emb_global = global_audio
        for layer, block in enumerate(model.blocks):
            sequence = block(
                sequence,
                [modulation, torch.tensor(split)],
                seq_lens,
                grid_sizes,
                coefficients,
                projected_context,
                None,
            )
            sequence_outputs.append(sequence.detach().clone())
            sequence = model.after_transformer_block(layer, sequence)
            audio_outputs.append(sequence.detach().clone())

        head_hidden = sequence[:, :split]
        prediction = model.head(head_hidden, time_embedding)
        latent = model.unpatchify(
            [prediction[0]],
            torch.tensor([[frames, height // 2, width // 2]], device="cpu"),
        )[0]

        # Official first inference request: drop the FramePack tokens and zero
        # the no-pose COND after the upstream VAE result. CondEncoder still
        # runs on zeros, retaining its learned bias exactly as in Wan's path.
        drop_condition = torch.zeros_like(condition)
        drop_sequence = torch.cat((video_tokens, reference_tokens), dim=1)
        drop_sequence[:, :split, :] += model.cond_encoder(drop_condition).flatten(2).transpose(1, 2)
        drop_raw_sequence = drop_sequence.detach().clone()
        drop_coefficients, drop_positions = position_groups(
            type("Oracle", (), {"s2v_utils": s2v_utils, "model": model})(),
            cfg,
            include_framepack=False,
        )
        rope_real = drop_coefficients.real.to(device=device, dtype=torch.float32)
        rope_imag = drop_coefficients.imag.to(device=device, dtype=torch.float32)
        drop_sequence_outputs = []
        drop_audio_outputs = []
        drop_seq_lens = torch.tensor([drop_sequence.shape[1]])
        model.original_seq_len = split
        model.merged_audio_emb = local_audio
        model.audio_emb_global = global_audio
        for layer, block in enumerate(model.blocks):
            drop_sequence = block(
                drop_sequence,
                [modulation, torch.tensor(split)],
                drop_seq_lens,
                grid_sizes,
                drop_coefficients,
                projected_context,
                None,
            )
            drop_sequence_outputs.append(drop_sequence.detach().clone())
            drop_sequence = model.after_transformer_block(layer, drop_sequence)
            drop_audio_outputs.append(drop_sequence.detach().clone())
        drop_prediction = model.head(drop_sequence[:, :split], time_embedding)
        drop_latent = model.unpatchify(
            [drop_prediction[0]],
            torch.tensor([[frames, height // 2, width // 2]], device="cpu"),
        )[0]

    weights, weight_shapes = source_weights(model)
    expected_positions = [position for position in positions]
    torch.mps.synchronize()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fixture = {
        "upstream": {
            "revision": REVISION,
            "source_sha256": SOURCE_SHA256,
            "device": "mps",
            "arithmetic": "F32",
            "fallback_disabled": True,
            "torch_version": torch.__version__,
        },
        "config": cfg,
        "inputs": {
            "video_latents": flat(video),
            "condition_latents": flat(condition),
            "no_pose_condition_latents": flat(condition),
            "reference_latents": flat(reference),
            "motion_latents": flat(motion),
            "raw_context": flat(context),
            "timestep": flat(timestep),
            "positions": [coordinate for position in expected_positions for coordinate in position],
            "drop_positions": [coordinate for position in drop_positions for coordinate in position],
            "local_audio": flat(local_audio),
            "global_audio": flat(global_audio),
        },
        "weights": weights,
        "weight_shapes": weight_shapes,
        "expected": {
            "sequence": flat(raw_sequence),
            "projected_context": flat(projected_context),
            "time_embedding": flat(time_embedding),
            "modulation": flat(modulation),
            "block0_output": flat(sequence_outputs[0]),
            "audio0_output": flat(audio_outputs[0]),
            "block1_output": flat(sequence_outputs[1]),
            "audio1_output": flat(audio_outputs[1]),
            "prediction": flat(prediction),
            "latent_output": flat(latent),
            "zero_condition": flat(drop_condition),
            "drop_no_pose": {
                "sequence": flat(drop_raw_sequence),
                "projected_context": flat(projected_context),
                "time_embedding": flat(time_embedding),
                "modulation": flat(modulation),
                "block1_output": flat(drop_sequence_outputs[1]),
                "audio1_output": flat(drop_audio_outputs[1]),
                "prediction": flat(drop_prediction),
                "latent_output": flat(drop_latent),
            },
        },
    }
    args.output.write_text(json.dumps(fixture, allow_nan=False, separators=(",", ":")))
    print(json.dumps({"fixture": str(args.output.resolve()), "bytes": args.output.stat().st_size,
                      "sha256": sha256(args.output), "device": "mps", "arithmetic": "F32",
                      "layers": cfg["layers"], "tokens": len(expected_positions)}))


if __name__ == "__main__":
    main()
