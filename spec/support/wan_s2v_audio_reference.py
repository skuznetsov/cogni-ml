#!/usr/bin/env python3
"""Create a pinned MPS fixture for Wan S2V's raw causal audio encoder.

The native spec consumes this fixture to compare every weighted-feature,
causal-convolution, LayerNorm, SiLU, local-pack, and global-projection result.
All input synthesis and neural arithmetic runs on MPS with fallback disabled;
CPU transfers happen only when exporting the completed fixture.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "0"
import torch
import torch.nn.functional as F
from safetensors import safe_open
from safetensors.torch import save_file


WAN_SOURCE_COMMIT = "1ea34ff48f87168174e12956e200b1d908b1c5ff"
WAN_AUDIO_UTILS_SHA256 = "76931d7291f69a6b6ed3a7a1a9a106ad8b5f5d494a451db67718b4fbcc4dbcfd"
WAN_AUXI_BLOCKS_SHA256 = "6ff7b5560036e5f0eaaa34460a328afcdbd7bd62baf93b699d5f6d2d556f95d6"

ROOT = Path("/opt/mldata_ssd/cogni-ml/video-scene-20261019/wan-audit")
TINY = {
    "feature_layers": 3,
    "input_dim": 6,
    "hidden_dim": 8,
    "local_tokens": 2,
    "bucket_frames": 9,
    "prefix_frames": 4,
    "drop_frames": 2,
}
PRODUCTION = {
    "feature_layers": 25,
    "input_dim": 1_024,
    "hidden_dim": 5_120,
    "local_tokens": 4,
    "bucket_frames": 84,
    "prefix_frames": 17,
    "drop_frames": 5,
}

PRODUCTION_SOURCES = {
    "casual_audio_encoder.weights": "casual_audio_encoder.weights",
    "casual_audio_encoder.encoder.conv1_local.conv.weight": "casual_audio_encoder.encoder.conv1_local.conv.weight",
    "casual_audio_encoder.encoder.conv1_local.conv.bias": "casual_audio_encoder.encoder.conv1_local.conv.bias",
    "casual_audio_encoder.encoder.conv1_global.conv.weight": "casual_audio_encoder.encoder.conv1_global.conv.weight",
    "casual_audio_encoder.encoder.conv1_global.conv.bias": "casual_audio_encoder.encoder.conv1_global.conv.bias",
    "casual_audio_encoder.encoder.conv2.conv.weight": "casual_audio_encoder.encoder.conv2.conv.weight",
    "casual_audio_encoder.encoder.conv2.conv.bias": "casual_audio_encoder.encoder.conv2.conv.bias",
    "casual_audio_encoder.encoder.conv3.conv.weight": "casual_audio_encoder.encoder.conv3.conv.weight",
    "casual_audio_encoder.encoder.conv3.conv.bias": "casual_audio_encoder.encoder.conv3.conv.bias",
    "casual_audio_encoder.encoder.final_linear.weight": "casual_audio_encoder.encoder.final_linear.weight",
    "casual_audio_encoder.encoder.final_linear.bias": "casual_audio_encoder.encoder.final_linear.bias",
    "casual_audio_encoder.encoder.padding_tokens": "casual_audio_encoder.encoder.padding_tokens",
}


def checkpoint_index(path: Path) -> tuple[dict[str, Path], Path | None]:
    if path.is_dir():
        index = path / "diffusion_pytorch_model.safetensors.index.json"
    elif path.name.endswith(".index.json"):
        index = path
    else:
        return {name: path for name in PRODUCTION_SOURCES}, None
    if not index.is_file():
        raise FileNotFoundError(f"Wan safetensors index is missing: {index}")
    data = json.loads(index.read_text())
    weight_map = data.get("weight_map")
    if not isinstance(weight_map, dict):
        raise ValueError(f"Wan safetensors index has no weight_map: {index}")
    mapping: dict[str, Path] = {}
    for name in PRODUCTION_SOURCES:
        shard = weight_map.get(name)
        if not isinstance(shard, str):
            raise KeyError(f"Wan checkpoint is missing required tensor {name}")
        mapping[name] = index.parent / shard
    return mapping, index


def load_production_weights(checkpoint: Path, device: torch.device) -> tuple[dict[str, torch.Tensor], dict[str, str]]:
    mapping, index = checkpoint_index(checkpoint)
    by_shard: dict[Path, list[str]] = {}
    for name, shard in mapping.items():
        by_shard.setdefault(shard, []).append(name)
    tensors: dict[str, torch.Tensor] = {}
    source_dtypes: dict[str, str] = {}
    for shard, names in by_shard.items():
        if not shard.is_file():
            raise FileNotFoundError(f"Wan checkpoint shard is missing: {shard}")
        with safe_open(str(shard), framework="pt", device="cpu") as handle:
            for name in names:
                value = handle.get_tensor(name)
                source_dtypes[name] = str(value.dtype)
                tensors[name] = value.to(device=device, dtype=torch.float32)
    return tensors, {
        "checkpoint": str(checkpoint.resolve()),
        "checkpoint_index": str(index.resolve()) if index else "",
        "checkpoint_index_sha256": sha256(index) if index else "",
        "checkpoint_shards": json.dumps(sorted(str(path.resolve()) for path in by_shard)),
        "source_dtypes": json.dumps(source_dtypes, sort_keys=True),
    }


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify_upstream() -> None:
    expected = {
        "wan_modules_s2v_audio_utils.py": WAN_AUDIO_UTILS_SHA256,
        "wan_modules_s2v_auxi_blocks.py": WAN_AUXI_BLOCKS_SHA256,
    }
    for name, digest in expected.items():
        path = ROOT / name
        actual = sha256(path)
        if actual != digest:
            raise RuntimeError(f"pinned Wan source digest mismatch: {path}: {actual}")


def causal_conv(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor, stride: int) -> torch.Tensor:
    # Upstream CausalConv1d uses replicate padding of (kernel_size - 1, 0).
    return F.conv1d(F.pad(x, (2, 0), mode="replicate"), weight, bias, stride=stride)


def as_trace(tensor: torch.Tensor) -> list[float]:
    return tensor.detach().contiguous().reshape(-1).tolist()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path,
                        help="Wan safetensors shard, index, or directory for production-width mode")
    parser.add_argument("--input-features", type=Path,
                        help="optional safetensors file with `features` [layers, channels, buckets]")
    parser.add_argument("--input-feature-key", default="features")
    parser.add_argument("--bucket-frames", type=int, default=84)
    parser.add_argument("--prefix-frames", type=int, default=17)
    parser.add_argument("--drop-frames", type=int, default=5)
    parser.add_argument("--memory-budget-bytes", type=int,
                        help="cap for weights, F32 staging, scratch, outputs, and complete traces")
    args = parser.parse_args()
    verify_upstream()
    if not torch.backends.mps.is_available():
        raise RuntimeError("MPS is required; CPU neural fallback is forbidden")

    torch.manual_seed(1729)
    device = torch.device("mps")
    production = args.checkpoint is not None
    config = dict(PRODUCTION if production else TINY)
    if production:
        config.update({
            "bucket_frames": args.bucket_frames,
            "prefix_frames": args.prefix_frames,
            "drop_frames": args.drop_frames,
        })
    elif args.input_features:
        raise ValueError("--input-features requires production --checkpoint mode")

    layers = config["feature_layers"]
    channels = config["input_dim"]
    hidden = config["hidden_dim"]
    tokens = config["local_tokens"]
    bucket_frames = config["bucket_frames"]
    prefix_frames = config["prefix_frames"]
    drop_frames = config["drop_frames"]
    quarter = hidden // 4
    half = hidden // 2

    def randn(shape: tuple[int, ...], scale: float) -> torch.Tensor:
        return torch.randn(shape, device=device, dtype=torch.float32) * scale

    if production:
        assert args.checkpoint is not None
        weights, checkpoint_metadata = load_production_weights(args.checkpoint, device)
        if args.input_features:
            with safe_open(str(args.input_features), framework="pt", device="cpu") as handle:
                features = handle.get_tensor(args.input_feature_key).to(device=device, dtype=torch.float32)
            if features.ndim == 3:
                features = features.unsqueeze(0)
            expected_shape = (1, layers, channels, bucket_frames)
            if tuple(features.shape) != expected_shape:
                raise ValueError(f"input feature shape must be {expected_shape}, got {tuple(features.shape)}")
        else:
            features = randn((1, layers, channels, bucket_frames), 0.2)
    else:
        # Source-compatible checkpoint state names and tensor ranks. Synthetic
        # values keep this fixture small while exercising every production op.
        weights = {
            "casual_audio_encoder.weights": torch.full((1, layers, 1, 1), 0.01, device=device),
            "casual_audio_encoder.encoder.conv1_local.conv.weight": randn((tokens * quarter, channels, 3), 0.08),
            "casual_audio_encoder.encoder.conv1_local.conv.bias": randn((tokens * quarter,), 0.015),
            "casual_audio_encoder.encoder.conv1_global.conv.weight": randn((quarter, channels, 3), 0.08),
            "casual_audio_encoder.encoder.conv1_global.conv.bias": randn((quarter,), 0.015),
            "casual_audio_encoder.encoder.conv2.conv.weight": randn((half, quarter, 3), 0.08),
            "casual_audio_encoder.encoder.conv2.conv.bias": randn((half,), 0.015),
            "casual_audio_encoder.encoder.conv3.conv.weight": randn((hidden, half, 3), 0.08),
            "casual_audio_encoder.encoder.conv3.conv.bias": randn((hidden,), 0.015),
            "casual_audio_encoder.encoder.final_linear.weight": randn((hidden, hidden), 0.08),
            "casual_audio_encoder.encoder.final_linear.bias": randn((hidden,), 0.015),
            "casual_audio_encoder.encoder.padding_tokens": randn((1, 1, 1, hidden), 0.02),
        }
        checkpoint_metadata = {}
        features = randn((1, layers, channels, bucket_frames), 0.2)
    layer_weights = weights["casual_audio_encoder.weights"]
    normalized_weights = F.silu(layer_weights)
    weight_sum = normalized_weights.sum(dim=1, keepdim=True)
    weighted = ((features * normalized_weights) / weight_sum).sum(dim=1).transpose(1, 2)
    prefix = weighted[:, 0:1, :].expand(-1, prefix_frames, -1)
    weighted = torch.cat((prefix, weighted), dim=1)
    trace: dict[str, torch.Tensor] = {"weighted": weighted[0]}

    def conv(name: str, x: torch.Tensor, stride: int) -> torch.Tensor:
        return causal_conv(
            x,
            weights[f"casual_audio_encoder.encoder.{name}.conv.weight"],
            weights[f"casual_audio_encoder.encoder.{name}.conv.bias"],
            stride,
        )

    # Local branch: conv1_local produces distinct contiguous token-head
    # channel groups. The following shared causal convolutions operate over
    # each head as an independent batch row.
    local = conv("conv1_local", weighted.transpose(1, 2), 1)
    t1 = local.shape[-1]
    local = local.reshape(1, tokens, quarter, t1).permute(0, 1, 3, 2).reshape(tokens, t1, quarter)
    trace["local_conv1"] = local
    local = F.layer_norm(local, (quarter,), eps=1e-6)
    trace["local_norm1"] = local
    local = F.silu(local)
    trace["local_act1"] = local

    local = conv("conv2", local.permute(0, 2, 1), 2)
    t2 = local.shape[-1]
    local = local.transpose(1, 2)
    trace["local_conv2"] = local
    local = F.layer_norm(local, (half,), eps=1e-6)
    trace["local_norm2"] = local
    local = F.silu(local)
    trace["local_act2"] = local

    local = conv("conv3", local.permute(0, 2, 1), 2)
    t3 = local.shape[-1]
    local = local.transpose(1, 2)
    trace["local_conv3"] = local
    local = F.layer_norm(local, (hidden,), eps=1e-6)
    trace["local_norm3"] = local
    local = F.silu(local)
    trace["local_act3"] = local
    local = local.permute(1, 0, 2).unsqueeze(0)
    pad = weights["casual_audio_encoder.encoder.padding_tokens"].expand(1, t3, 1, hidden)
    local = torch.cat((local, pad), dim=2)[:, drop_frames:, :, :]
    trace["local_output"] = local[0]

    # Global branch shares conv2/conv3 and non-affine norms but has its own
    # first convolution and final hidden-to-hidden affine projection.
    global_hidden = conv("conv1_global", weighted.transpose(1, 2), 1).transpose(1, 2)
    trace["global_conv1"] = global_hidden[0]
    global_hidden = F.layer_norm(global_hidden, (quarter,), eps=1e-6)
    trace["global_norm1"] = global_hidden[0]
    global_hidden = F.silu(global_hidden)
    trace["global_act1"] = global_hidden[0]

    global_hidden = conv("conv2", global_hidden.transpose(1, 2), 2).transpose(1, 2)
    trace["global_conv2"] = global_hidden[0]
    global_hidden = F.layer_norm(global_hidden, (half,), eps=1e-6)
    trace["global_norm2"] = global_hidden[0]
    global_hidden = F.silu(global_hidden)
    trace["global_act2"] = global_hidden[0]

    global_hidden = conv("conv3", global_hidden.transpose(1, 2), 2).transpose(1, 2)
    trace["global_conv3"] = global_hidden[0]
    global_hidden = F.layer_norm(global_hidden, (hidden,), eps=1e-6)
    trace["global_norm3"] = global_hidden[0]
    global_hidden = F.silu(global_hidden)
    trace["global_act3"] = global_hidden[0]
    linear_weight = weights["casual_audio_encoder.encoder.final_linear.weight"]
    linear_bias = weights["casual_audio_encoder.encoder.final_linear.bias"]
    global_hidden = F.linear(global_hidden, linear_weight, linear_bias)
    trace["global_linear"] = global_hidden[0]
    global_output = global_hidden[:, drop_frames:, :]
    trace["global_output"] = global_output[0]

    # One terminal synchronization, then host conversion for safetensors and
    # JSON. No CPU operation participates in the model or reference math.
    torch.mps.synchronize()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    weights_path = args.output.with_suffix(".safetensors")
    state_cpu = {name: tensor.detach().to(device="cpu").contiguous() for name, tensor in weights.items()}
    save_file(state_cpu, str(weights_path), metadata={
        "wan_source_commit": WAN_SOURCE_COMMIT,
        "wan_audio_utils_sha256": WAN_AUDIO_UTILS_SHA256,
        "wan_auxi_blocks_sha256": WAN_AUXI_BLOCKS_SHA256,
        "arithmetic": "F32",
        "device": "mps",
    })
    feature_cpu = features[0].detach().to(device="cpu").contiguous()
    traces_cpu = {name: as_trace(value.to(device="cpu")) for name, value in trace.items()}
    input_frames = bucket_frames + prefix_frames
    conv2_frames = (input_frames + 1) // 2
    conv3_frames = (conv2_frames + 1) // 2
    video_frames = conv3_frames - drop_frames
    activation_sizes = [
        tokens * input_frames * quarter,
        tokens * conv2_frames * half,
        tokens * conv3_frames * hidden,
        input_frames * quarter,
        conv2_frames * half,
        conv3_frames * hidden,
    ]
    scratch_bytes = (input_frames * channels + 2 * max(activation_sizes)) * 4
    feature_values = layers * channels * bucket_frames
    local_output_values = video_frames * (tokens + 1) * hidden
    global_output_values = video_frames * hidden
    trace_values = sum(value.numel() for value in trace.values())
    reserve_bytes = scratch_bytes + (
        feature_values + 2 * local_output_values + 2 * global_output_values + trace_values
    ) * 4
    weight_bytes = sum(tensor.numel() * 4 for tensor in weights.values())
    memory_budget = args.memory_budget_bytes
    if memory_budget is None:
        memory_budget = 2 * 1024**3 if production else 512 * 1024**2
    planned_peak = 2 * weight_bytes + reserve_bytes
    if planned_peak > memory_budget:
        raise MemoryError(f"native fixture plan {planned_peak} exceeds configured budget {memory_budget}")
    output_cpu = {
        "local_output": traces_cpu["local_output"],
        "global_output": traces_cpu["global_output"],
    }
    fixture = {
        "mode": "production" if production else "tiny",
        "wan_source_commit": WAN_SOURCE_COMMIT,
        "wan_audio_utils_sha256": WAN_AUDIO_UTILS_SHA256,
        "wan_auxi_blocks_sha256": WAN_AUXI_BLOCKS_SHA256,
        "device": "mps",
        "arithmetic": "F32",
        "torch_version": torch.__version__,
        **checkpoint_metadata,
        "weight_sha256": sha256(weights_path),
        "weights": str(weights_path.resolve()),
        "memory_budget_bytes": memory_budget,
        "planned_peak_bytes": planned_peak,
        "config": config,
        "input_features": feature_cpu.reshape(-1).tolist(),
        "input_shape": list(feature_cpu.shape),
        "conv_frames": [t1, t2, t3],
        "video_frames": local.shape[1],
        "expected": traces_cpu,
        "outputs": output_cpu,
    }
    args.output.write_text(json.dumps(fixture, allow_nan=False, separators=(",", ":")))
    print(json.dumps({
        "fixture": str(args.output),
        "weights": str(weights_path),
        "trace_names": list(traces_cpu),
        "mode": "production" if production else "tiny",
        "conv_frames": [t1, t2, t3],
        "video_frames": video_frames,
        "weight_bytes_f32": weight_bytes,
        "scratch_bytes": scratch_bytes,
        "planned_peak_bytes": planned_peak,
        "memory_budget_bytes": memory_budget,
        "device": "mps",
        "torch_version": torch.__version__,
    }))


if __name__ == "__main__":
    main()
