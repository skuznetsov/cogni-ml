#!/usr/bin/env python3
"""Pinned Wan2.1 VAE MPS oracle for the native F32 Metal implementation.

The reference records deterministic image/video encodes, a 22-latent decode,
and the actual first/second temporal cache states produced by the upstream
module. Neural arithmetic is performed on MPS; this script has no CPU fallback.

Copyright 2024-2025 The Alibaba Wan Team Authors. Apache-2.0.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "0"
import torch
from safetensors.torch import load_file

REVISION = "1ea34ff48f87168174e12956e200b1d908b1c5ff"
SOURCE_SHA256 = "d0dfe9eccd0cdbcb37ffde9887579be58f4c1bb6fc2dbb1b72e8916e170df180"
WEIGHT_SHA256 = "2ae212ca06c67f00681d5e155d1a954f5ce81eeedb9f5ffd344829817718991f"
MEAN = [-0.7571, -0.7089, -0.9113, 0.1075, -0.1745, 0.9653, -0.1517, 1.5508,
        0.4134, -0.0715, 0.5517, -0.3632, -0.1922, -0.9497, 0.2503, -0.2921]
STD = [2.8184, 1.4541, 2.3275, 2.6558, 1.2196, 1.7708, 2.6052, 2.0743,
       3.2687, 2.1526, 2.8652, 1.5579, 1.6382, 1.1253, 2.8251, 1.9160]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def flat(tensor: torch.Tensor) -> list[float]:
    return tensor.detach().to(device="cpu", dtype=torch.float32).contiguous().view(-1).tolist()


def cache_manifest(cache: list[object]) -> list[object]:
    result: list[object] = []
    for item in cache:
        if item is None or isinstance(item, str):
            result.append(item)
        else:
            result.append({"shape": list(item.shape), "dtype": str(item.dtype)})
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--upstream", type=Path, required=True,
                        help="Directory containing pinned wan_modules_vae2_1.py")
    parser.add_argument("--checkpoint", type=Path, required=True,
                        help="Exact F32 Wan2.1_VAE.safetensors")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--height", type=int, default=8)
    parser.add_argument("--width", type=int, default=8)
    parser.add_argument("--motion-frames", type=int, default=73)
    parser.add_argument("--decode-latents", type=int, default=22)
    parser.add_argument("--infer-frames", type=int, default=84,
                        help="Upstream first-repeat clip window before initial-motion drop")
    parser.add_argument("--deliver-frames", type=int, default=80,
                        help="Explicit terminal crop after the upstream initial drop")
    args = parser.parse_args()

    source = args.upstream / "wan_modules_vae2_1.py"
    if sha256(source) != SOURCE_SHA256:
        raise RuntimeError("Pinned Wan2.1 VAE source hash mismatch")
    if sha256(args.checkpoint) != WEIGHT_SHA256:
        raise RuntimeError("Wan2.1 VAE checkpoint hash mismatch")
    if not torch.backends.mps.is_available():
        raise RuntimeError("MPS is required; CPU neural fallback is forbidden")
    if args.height <= 0 or args.width <= 0 or args.height % 8 or args.width % 8:
        raise ValueError("height and width must be positive multiples of eight")
    if args.motion_frames < 1 or (args.motion_frames - 1) % 4:
        raise ValueError("motion frame count must be 1+4n")
    if args.decode_latents < 1 or args.infer_frames < 1 or args.deliver_frames < 1:
        raise ValueError("decode and delivered frame counts must be positive")

    namespace: dict[str, object] = {"__name__": "wan_vae_pinned_reference"}
    exec(compile(source.read_text(), str(source), "exec"), namespace)
    model_cls = namespace["WanVAE_"]
    cfg = dict(dim=96, z_dim=16, dim_mult=[1, 2, 4, 4], num_res_blocks=2,
               attn_scales=[], temperal_downsample=[False, True, True], dropout=0.0)
    with torch.device("meta"):
        model = model_cls(**cfg)
    state = load_file(str(args.checkpoint), device="cpu")
    if len(state) != 194 or any(t.dtype != torch.float32 for t in state.values()):
        raise RuntimeError("Expected the exact 194-tensor F32 Wan2.1 VAE asset")
    incompatible = model.load_state_dict(state, strict=True, assign=True)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise RuntimeError(f"VAE state mismatch: {incompatible}")
    del state
    model = model.eval().requires_grad_(False).to(device="mps", dtype=torch.float32)
    mean = torch.tensor(MEAN, dtype=torch.float32, device="mps").view(1, 16, 1, 1, 1)
    inv_std = torch.tensor([1.0 / x for x in STD], dtype=torch.float32,
                           device="mps").view(1, 16, 1, 1, 1)
    scale = [mean, inv_std]

    torch.manual_seed(9173)
    image_index = torch.arange(3 * args.height * args.width, dtype=torch.float32)
    image = (torch.sin(image_index * 0.071) * 0.8).reshape(1, 3, 1, args.height, args.width)
    motion_index = torch.arange(3 * args.motion_frames * args.height * args.width,
                                dtype=torch.float32)
    motion = (torch.sin(motion_index * 0.013) * 0.55 +
              torch.cos(motion_index * 0.0031) * 0.2).reshape(
                  1, 3, args.motion_frames, args.height, args.width)
    latent_index = torch.arange(16 * args.decode_latents * (args.height // 8) * (args.width // 8),
                                dtype=torch.float32)
    latent = (torch.cos(latent_index * 0.037) * 0.6).reshape(
        1, 16, args.decode_latents, args.height // 8, args.width // 8)

    with torch.inference_mode():
        image_latent = model.encode(image.to("mps"), scale)
        motion_latent = model.encode(motion.to("mps"), scale)
        decoded = model.decode(latent.to("mps"), scale).clamp(-1, 1)
        # Match the upstream first-repeat path: last infer_frames, drop the
        # initial three motion frames, then retain the requested 80-frame clip.
        upstream_window = decoded[:, :, -args.infer_frames:]
        after_motion_drop = upstream_window[:, :, 3:]
        if after_motion_drop.shape[2] < args.deliver_frames:
            raise ValueError("decode output is too short for requested delivery crop")
        delivered = after_motion_drop[:, :, :args.deliver_frames]

        # Trace first, second, and later chunk calls/cache lists so the fixture
        # distinguishes warm-up from steady-state causal history.
        model.clear_cache()
        encode_trace: list[dict[str, object]] = []
        input_mps = motion.to("mps")
        chunk_count = 1 + (args.motion_frames - 1) // 4
        for chunk_index in range(min(3, chunk_count)):
            start = 0 if chunk_index == 0 else 1 + 4 * (chunk_index - 1)
            end = 1 if chunk_index == 0 else start + 4
            feature_index = [0]
            encoded = model.encoder(input_mps[:, :, start:end],
                                    feat_cache=model._enc_feat_map,
                                    feat_idx=feature_index)
            encode_trace.append({
                "chunk_index": chunk_index,
                "input_offset": start,
                "input_frames": end - start,
                "output_frames": int(encoded.shape[2]),
                "feat_idx": feature_index[0],
                "cache": cache_manifest(model._enc_feat_map),
            })

        # Trace first, second, and later per-latent decoder calls, including
        # both temporal upsamplers' Rep -> two-frame cache transition.
        model.clear_cache()
        raw_latent = latent.to("mps")
        unscaled = raw_latent / inv_std + mean
        projected = model.conv2(unscaled)
        decode_trace: list[dict[str, object]] = []
        for frame in range(min(3, args.decode_latents)):
            feature_index = [0]
            decoded_chunk = model.decoder(projected[:, :, frame:frame + 1],
                                          feat_cache=model._feat_map,
                                          feat_idx=feature_index)
            decode_trace.append({
                "latent_index": frame,
                "input_frames": 1,
                "output_frames": int(decoded_chunk.shape[2]),
                "feat_idx": feature_index[0],
                "cache": cache_manifest(model._feat_map),
            })
    torch.mps.synchronize()

    latent_frames = 1 + (args.motion_frames - 1) // 4
    decoded_frames = 1 + (args.decode_latents - 1) * 4
    report = {
        "revision": REVISION,
        "source_sha256": SOURCE_SHA256,
        "device": "mps",
        "arithmetic": "F32",
        "storage": "F32",
        "checkpoint": str(args.checkpoint),
        "weight_sha256": WEIGHT_SHA256,
        "config": {**cfg, "batches": 1, "height": args.height,
                    "width": args.width, "motion_frames": args.motion_frames,
                    "motion_latent_frames": latent_frames,
                    "decode_latent_frames": args.decode_latents,
                    "decoded_frames": decoded_frames,
                    "infer_frames": args.infer_frames,
                    "upstream_drop_frames": 3,
                    "deliver_frames": args.deliver_frames,
                    "delivered_source_offset": decoded_frames - args.infer_frames + 3},
        "inputs": {"image": flat(image), "motion": flat(motion), "latent": flat(latent)},
        "expected": {
            "image_latent": flat(image_latent),
            "motion_latent": flat(motion_latent),
            "decoded": flat(decoded),
            "delivered": flat(delivered),
        },
        "cache_trace": {"encode": encode_trace, "decode": decode_trace},
        "torch_version": torch.__version__,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, allow_nan=False, separators=(",", ":")))
    print(json.dumps({"fixture": str(args.output), "device": "mps",
                      "motion_latents": latent_frames, "decoded_frames": decoded_frames,
                      "encoder_cache_slots": len(encode_trace[0]["cache"]),
                      "decoder_cache_slots": len(decode_trace[0]["cache"])}))


if __name__ == "__main__":
    main()
