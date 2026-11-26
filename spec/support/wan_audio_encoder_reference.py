#!/usr/bin/env python3
"""Generate tiny or production XLSR/Wan parity fixtures.

The native spec loads F32 tensors into Metal and compares every captured
Wav2Vec2 hidden state after Wan interpolation and 16-fps bucketing. Tiny mode
uses a seeded model; production mode loads the local XLSR checkpoint. Both pin
the exact Transformers source implementing the neural operators.
"""

from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import os
from pathlib import Path

# Any unsupported MPS operation must fail; the oracle may never silently execute
# a neural operator on CPU.
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "0"
import numpy as np
import torch
import torch.nn.functional as F
# Wav2Vec2 does not need vision support. This oracle runs in a mixed venv/
# Miniconda environment where importing the optional torchvision package can
# fail during duplicate fake-op registration, before Wav2Vec2 is loaded. Mark
# that unrelated optional dependency unavailable before importing model classes.
from transformers.utils import import_utils as transformers_import_utils

transformers_import_utils._torchvision_available = False
import transformers
from safetensors.torch import save_file
from transformers import Wav2Vec2Config, Wav2Vec2FeatureExtractor, Wav2Vec2ForCTC, Wav2Vec2Model
from transformers.models.wav2vec2 import modeling_wav2vec2


PINNED_TRANSFORMERS_VERSION = "4.57.3"
PINNED_MODEL_REVISION = "dab4e9c55bbe4c8c4d03db1c2c98c7f0ac9c454b"
PINNED_MODELING_SHA256 = "1c19d14f99c623c71aa8a058235978a06a8af4defb5e1f54321f7a6b7984e572"
SAMPLE_RATE = 16_000
TARGET_FRAMES = 6  # Deliberately yields three repeated six-frame buckets.
PRODUCTION_TARGET_FRAMES = 84
VIDEO_FPS = 30
TARGET_FPS = 16


def verify_upstream_source() -> str:
    if transformers.__version__ != PINNED_TRANSFORMERS_VERSION:
        raise RuntimeError(
            f"expected Transformers {PINNED_TRANSFORMERS_VERSION}, got {transformers.__version__}"
        )
    source = Path(inspect.getfile(modeling_wav2vec2)).resolve()
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    if digest != PINNED_MODELING_SHA256:
        raise RuntimeError(f"pinned Wav2Vec2 source digest mismatch: {source} {digest}")
    return str(source)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def make_tiny_model() -> Wav2Vec2ForCTC:
    torch.manual_seed(1729)
    config = Wav2Vec2Config(
        vocab_size=32,
        conv_dim=(8, 8, 8, 8, 8, 8, 8),
        conv_kernel=(10, 3, 3, 3, 3, 2, 2),
        conv_stride=(5, 2, 2, 2, 2, 2, 2),
        conv_bias=True,
        feat_extract_norm="layer",
        feat_extract_activation="gelu",
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=2,
        num_attention_heads=4,
        hidden_act="gelu",
        hidden_dropout=0.0,
        attention_dropout=0.0,
        activation_dropout=0.0,
        feat_proj_dropout=0.0,
        layerdrop=0.0,
        do_stable_layer_norm=True,
        num_conv_pos_embeddings=8,
        num_conv_pos_embedding_groups=4,
        layer_norm_eps=1e-5,
        mask_time_prob=0.0,
        mask_feature_prob=0.0,
    )
    config._attn_implementation = "eager"
    if not torch.backends.mps.is_available():
        raise RuntimeError("MPS is required; CPU neural fallback is forbidden")
    model = Wav2Vec2ForCTC(config).to(device="mps", dtype=torch.float32)
    model.eval()
    return model


def sample_indices(interp_len: int, target_frames: int) -> tuple[list[int], int, int]:
    scale = VIDEO_FPS / TARGET_FPS
    repeats = int(interp_len / (target_frames * scale)) + 1
    bucket_count = repeats * target_frames
    padded_frames = int(np.ceil(bucket_count / TARGET_FPS * VIDEO_FPS)) - interp_len
    total_frames = interp_len + padded_frames
    duration = bucket_count / TARGET_FPS
    times = np.linspace(0.0, duration, bucket_count, endpoint=False)
    indices = np.round(times * VIDEO_FPS).astype(np.int64)
    indices = np.clip(indices, 0, total_frames - 1).tolist()
    return indices, repeats, padded_frames


def read_native_trace_array(path: Path, name: str, expected_values: int | None) -> np.ndarray:
    """Read one flat native trace from a potentially large JSON capture."""
    import mmap

    key = json.dumps(name, separators=(",", ":")).encode("ascii") + b":"
    with path.open("rb") as stream:
        with mmap.mmap(stream.fileno(), 0, access=mmap.ACCESS_READ) as mapped:
            trace_start = mapped.find(b'"trace":{')
            if trace_start < 0:
                raise ValueError(f"native capture has no trace object: {path}")
            key_start = mapped.find(key, trace_start)
            if key_start < 0:
                raise ValueError(f"native capture has no trace.{name}: {path}")
            array_start = mapped.find(b"[", key_start + len(key))
            array_end = mapped.find(b"]", array_start + 1)
            if array_start < 0 or array_end < 0:
                raise ValueError(f"native trace.{name} is not a flat JSON array")
            values = np.fromstring(mapped[array_start + 1:array_end], sep=",", dtype=np.float32)
    if expected_values is not None and values.size != expected_values:
        raise ValueError(
            f"native trace.{name} has {values.size} values; expected {expected_values}"
        )
    if not np.isfinite(values).all():
        raise ValueError(f"native trace.{name} contains non-finite values")
    return values


def run_conv1_same_input_probe(
    trunk: Wav2Vec2Model,
    native_json: Path,
    output_path: Path,
    conv_dims: list[int],
    conv_kernels: list[int],
    conv_strides: list[int],
) -> None:
    """Run pinned MPS Conv1d on the exact native post-conv0 tensor."""
    conv0_values = read_native_trace_array(native_json, "feature_conv_0", None)
    if conv0_values.size % conv_dims[0] != 0:
        raise ValueError("native trace.feature_conv_0 does not match the checkpoint channel width")
    conv0_length = conv0_values.size // conv_dims[0]
    native_input = torch.from_numpy(conv0_values).reshape(1, conv0_length, conv_dims[0]).to("mps")
    conv1_length = (conv0_length - conv_kernels[1]) // conv_strides[1] + 1
    conv1 = trunk.feature_extractor.conv_layers[1].conv
    with torch.inference_mode():
        raw = conv1(native_input.transpose(1, 2).contiguous())
        raw_sequence_major = raw.transpose(1, 2).contiguous()
        torch.mps.synchronize()
        result = raw_sequence_major.detach().to(device="cpu").reshape(-1).contiguous().clone()
    expected_values = conv1_length * conv_dims[1]
    if result.numel() != expected_values:
        raise RuntimeError(f"pinned MPS conv1 returned {result.numel()} values; expected {expected_values}")
    if not torch.isfinite(result).all():
        raise RuntimeError("pinned MPS conv1 returned non-finite values")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    save_file({"feature_conv_1_raw_from_native_feature_conv_0": result}, str(output_path), metadata={
        "transformers_version": PINNED_TRANSFORMERS_VERSION,
        "modeling_source_sha256": PINNED_MODELING_SHA256,
        "arithmetic": "F32",
        "device": "mps",
        "input_trace": "feature_conv_0 from native JSON",
    })
    print(json.dumps({
        "probe_output": str(output_path),
        "input_trace": str(native_json),
        "input_shape": [1, conv0_length, conv_dims[0]],
        "output_shape": [1, conv1_length, conv_dims[1]],
        "device": "mps",
    }))


def run_feature_extractor_same_input_probe(
    trunk: Wav2Vec2Model,
    native_json: Path,
    output_path: Path,
    conv_dims: list[int],
    conv_kernels: list[int],
    conv_strides: list[int],
) -> None:
    """Run the pinned full feature extractor on the native normalized waveform.

    This discriminator holds the first feature-extractor input fixed while
    comparing every available conv, LayerNorm, and GELU boundary. It emits no
    transformer activations and only copies results to CPU at terminal export.
    """
    normalized_values = read_native_trace_array(native_json, "normalized_waveform", None)
    input_length = int(normalized_values.size)
    output_lengths: list[int] = []
    current_length = input_length
    for kernel, stride in zip(conv_kernels, conv_strides, strict=True):
        current_length = (current_length - kernel) // stride + 1
        if current_length <= 0:
            raise ValueError("native normalized waveform is too short for the XLSR feature extractor")
        output_lengths.append(current_length)

    trace: dict[str, torch.Tensor] = {}

    def capture_feature_stage(name: str):
        def hook(_module, _inputs, output):
            trace[name] = output.transpose(1, 2).contiguous().detach().clone()
        return hook

    def capture_conv_output(name: str):
        def hook(_module, _inputs, output):
            trace[name] = output.transpose(1, 2).contiguous().detach().clone()
        return hook

    def capture_norm_output(name: str):
        def hook(_module, _inputs, output):
            trace[name] = output.detach().clone()
        return hook

    handles = []
    conv_layers = trunk.feature_extractor.conv_layers
    if len(conv_layers) != len(conv_dims):
        raise ValueError("pinned XLSR feature-extractor layer count does not match checkpoint config")
    for index, conv_layer in enumerate(conv_layers):
        handle = conv_layer.conv.register_forward_hook(capture_conv_output(f"feature_conv_{index}_raw"))
        handles.append(handle)
        norm_module = getattr(conv_layer, "layer_norm", None)
        if norm_module is not None:
            handle = norm_module.register_forward_hook(capture_norm_output(f"feature_conv_{index}_norm"))
            handles.append(handle)
        handle = conv_layer.register_forward_hook(capture_feature_stage(f"feature_conv_{index}"))
        handles.append(handle)

    normalized = torch.from_numpy(normalized_values).to(device="mps").reshape(1, input_length)
    try:
        with torch.inference_mode():
            result = trunk.feature_extractor(normalized)
        if result.shape[-1] != output_lengths[-1] or result.shape[1] != conv_dims[-1]:
            raise RuntimeError("pinned MPS feature extractor returned unexpected output geometry")
        torch.mps.synchronize()
        tensors = {
            f"trace.{name}": value.detach().to(device="cpu").contiguous().reshape(-1).clone()
            for name, value in trace.items()
        }
    finally:
        for handle in handles:
            if handle is not None:
                handle.remove()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    save_file(tensors, str(output_path), metadata={
        "transformers_version": PINNED_TRANSFORMERS_VERSION,
        "modeling_source_sha256": PINNED_MODELING_SHA256,
        "arithmetic": "F32",
        "device": "mps",
        "input_trace": "normalized_waveform from native JSON",
    })
    print(json.dumps({
        "probe_output": str(output_path),
        "input_trace": str(native_json),
        "input_shape": [1, input_length],
        "conv_output_lengths": output_lengths,
        "captured_traces": sorted(tensors),
        "device": "mps",
    }))


def run_position_same_input_probe(
    trunk: Wav2Vec2Model,
    native_json: Path,
    output_path: Path,
    hidden_size: int,
    position_kernel: int,
    position_groups: int,
) -> None:
    """Run pinned grouped position Conv1d/GELU on native projection+scale.

    Feeding the exact native `feature_projection` and `position_scale` values
    separates the positional Conv1d and activation from all earlier XLSR
    arithmetic. The scale is intentionally supplied by the native capture;
    the checkpoint's weight_v and bias remain MPS-resident.
    """
    projection_values = read_native_trace_array(native_json, "feature_projection", None)
    scale_values = read_native_trace_array(native_json, "position_scale", position_kernel)
    if projection_values.size % hidden_size != 0:
        raise ValueError("native trace.feature_projection does not match the checkpoint hidden width")
    frames = projection_values.size // hidden_size
    projection = torch.from_numpy(projection_values).reshape(1, frames, hidden_size).to(device="mps")
    scale = torch.from_numpy(scale_values).to(device="mps").reshape(1, 1, position_kernel)
    convolution = trunk.encoder.pos_conv_embed.conv
    if hasattr(convolution, "parametrizations"):
        weight_v = convolution.parametrizations.weight.original1
    else:
        weight_v = convolution.weight_v
    if tuple(weight_v.shape) != (hidden_size, hidden_size // position_groups, position_kernel):
        raise ValueError("pinned XLSR positional weight shape does not match its config")
    if convolution.bias is None:
        raise ValueError("pinned XLSR positional Conv1d must have a bias")

    with torch.inference_mode():
        # This reproduces Wav2Vec2PositionalConvEmbedding.forward's order:
        # Conv1d with symmetric K//2 padding, SamePad crop, then exact GELU.
        effective_weight = weight_v * scale
        full = F.conv1d(
            projection.transpose(1, 2).contiguous(),
            effective_weight,
            convolution.bias,
            stride=1,
            padding=position_kernel // 2,
            groups=position_groups,
        )
        expected_full_frames = frames + (1 if position_kernel % 2 == 0 else 0)
        if full.shape[-1] != expected_full_frames:
            raise RuntimeError(
                f"pinned positional Conv1d returned {full.shape[-1]} frames; "
                f"expected {expected_full_frames} before SamePad"
            )
        raw = full[..., :frames] if position_kernel % 2 == 0 else full
        activated = F.gelu(raw, approximate="none")
        raw_sequence_major = raw.transpose(1, 2).contiguous()
        activated_sequence_major = activated.transpose(1, 2).contiguous()
        torch.mps.synchronize()
        tensors = {
            "position_conv_raw": raw_sequence_major.detach().to(device="cpu").reshape(-1).contiguous().clone(),
            "position_conv": activated_sequence_major.detach().to(device="cpu").reshape(-1).contiguous().clone(),
        }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    save_file(tensors, str(output_path), metadata={
        "transformers_version": PINNED_TRANSFORMERS_VERSION,
        "model_revision": PINNED_MODEL_REVISION,
        "modeling_source_sha256": PINNED_MODELING_SHA256,
        "arithmetic": "F32",
        "device": "mps",
        "input_trace": "feature_projection and position_scale from native JSON",
        "native_capture_sha256": sha256(native_json),
        "samepad": "symmetric K//2 conv then crop final value for even kernels",
        "weight_norm": "native scale supplied; checkpoint weight_v and bias",
    })
    print(json.dumps({
        "probe_output": str(output_path),
        "input_trace": str(native_json),
        "input_shape": [1, frames, hidden_size],
        "position_scale_values": int(scale_values.size),
        "raw_shape": [1, frames, hidden_size],
        "device": "mps",
    }))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path,
                        help="fixture JSON path; required except in same-input probe mode")
    parser.add_argument("--checkpoint", type=Path,
                        help="local XLSR model.safetensors or model directory; enables real-checkpoint mode")
    parser.add_argument("--same-input-feature-conv0-json", type=Path,
                        help="native JSON capture; run only pinned MPS conv1 on its feature_conv_0")
    parser.add_argument("--same-input-probe-output", type=Path,
                        help="safetensors path for same-input conv1 output")
    parser.add_argument("--same-input-feature-extractor-json", type=Path,
                        help="native JSON capture; rerun pinned XLSR feature extractor on its normalized waveform")
    parser.add_argument("--same-input-feature-extractor-output", type=Path,
                        help="safetensors path for same-input feature-extractor stage captures")
    parser.add_argument("--same-input-position-json", type=Path,
                        help="native JSON capture; rerun positional Conv1d/GELU on its feature_projection and position_scale")
    parser.add_argument("--same-input-position-output", type=Path,
                        help="safetensors path for same-input positional Conv1d and GELU captures")
    parser.add_argument("--input-samples", type=int,
                        help="waveform length (defaults to 80000 in production, 16000 in tiny mode)")
    parser.add_argument("--memory-budget-bytes", type=int, default=4 * 1024 * 1024 * 1024,
                        help="production native weight/staging/resident admission budget")
    args = parser.parse_args()
    source_path = verify_upstream_source()
    production = args.checkpoint is not None
    target_frames = PRODUCTION_TARGET_FRAMES if production else TARGET_FRAMES
    input_samples = args.input_samples if args.input_samples is not None else (80_000 if production else SAMPLE_RATE)
    if input_samples <= 0:
        raise ValueError("--input-samples must be positive")

    if production:
        assert args.checkpoint is not None
        checkpoint_dir = args.checkpoint if args.checkpoint.is_dir() else args.checkpoint.parent
        checkpoint_file = args.checkpoint if args.checkpoint.is_file() else checkpoint_dir / "model.safetensors"
        if not checkpoint_file.is_file():
            raise FileNotFoundError(f"local XLSR model.safetensors is missing: {checkpoint_file}")
        # Wav2Vec2Model recognizes and strips the `wav2vec2.` prefix in the
        # pinned Wav2Vec2ForCTC checkpoint, so the unused CTC head is not part
        # of this reference model.
        model = Wav2Vec2Model.from_pretrained(
            str(checkpoint_dir), local_files_only=True, attn_implementation="eager"
        ).to(device="mps", dtype=torch.float32)
        model.eval()
        trunk = model
        checkpoint_metadata = {
            "checkpoint": str(checkpoint_file.resolve()),
            "checkpoint_bytes": checkpoint_file.stat().st_size,
        }
        target_config = model.config
        model_config = {
            "conv_dims": list(target_config.conv_dim),
            "conv_kernels": list(target_config.conv_kernel),
            "conv_strides": list(target_config.conv_stride),
            "hidden_size": target_config.hidden_size,
            "intermediate_size": target_config.intermediate_size,
            "heads": target_config.num_attention_heads,
            "layers": target_config.num_hidden_layers,
            "position_kernel": target_config.num_conv_pos_embeddings,
            "position_groups": target_config.num_conv_pos_embedding_groups,
            "feature_norm": target_config.feat_extract_norm,
            "stable_layer_norm": target_config.do_stable_layer_norm,
            "eps": target_config.layer_norm_eps,
        }
    else:
        model = make_tiny_model()
        trunk = model.wav2vec2
        checkpoint_metadata = {}
        model_config = {
            "conv_dims": [8] * 7,
            "conv_kernels": [10, 3, 3, 3, 3, 2, 2],
            "conv_strides": [5, 2, 2, 2, 2, 2, 2],
            "hidden_size": 16,
            "intermediate_size": 32,
            "heads": 4,
            "layers": 2,
            "position_kernel": 8,
            "position_groups": 4,
            "feature_norm": "layer",
            "stable_layer_norm": True,
            "eps": 1e-5,
        }
    trunk.config._attn_implementation = "eager"

    if args.same_input_feature_conv0_json is not None:
        if not production or args.same_input_probe_output is None:
            parser.error("same-input probe requires --checkpoint and --same-input-probe-output")
        run_conv1_same_input_probe(
            trunk,
            args.same_input_feature_conv0_json,
            args.same_input_probe_output,
            model_config["conv_dims"],
            model_config["conv_kernels"],
            model_config["conv_strides"],
        )
        return
    if args.same_input_feature_extractor_json is not None:
        if not production or args.same_input_feature_extractor_output is None:
            parser.error(
                "feature-extractor same-input probe requires --checkpoint and "
                "--same-input-feature-extractor-output"
            )
        run_feature_extractor_same_input_probe(
            trunk,
            args.same_input_feature_extractor_json,
            args.same_input_feature_extractor_output,
            model_config["conv_dims"],
            model_config["conv_kernels"],
            model_config["conv_strides"],
        )
        return
    if args.same_input_position_json is not None:
        if not production or args.same_input_position_output is None:
            parser.error("same-input position probe requires --checkpoint and --same-input-position-output")
        run_position_same_input_probe(
            trunk,
            args.same_input_position_json,
            args.same_input_position_output,
            model_config["hidden_size"],
            model_config["position_kernel"],
            model_config["position_groups"],
        )
        return
    if args.output is None:
        parser.error("--output is required outside same-input probe mode")

    # Save only the actual XLSR trunk. The CTC head is not a Wan conditioning
    # dependency. Map PyTorch's current parametrization keys to the legacy
    # safetensors names used by the pinned production XLSR checkpoint.
    state = {}
    if not production:
        for name, tensor in trunk.state_dict().items():
            name = name.replace("parametrizations.weight.original0", "weight_g")
            name = name.replace("parametrizations.weight.original1", "weight_v")
            state[f"wav2vec2.{name}"] = tensor.detach().to(device="cpu").contiguous()

    # Match the five-second oracle input's construction at a shorter size,
    # making the seven valid Conv1d stages and repeated/padded buckets cheap.
    indices = np.arange(input_samples, dtype=np.float32)
    raw_cpu = (0.3 * np.sin(indices * 0.017) + 0.1 * np.cos(indices * 0.0031)).astype(np.float32)
    # The upstream Wav2Vec2FeatureExtractor performs this preprocessing in
    # NumPy. Keep it as a non-neural CPU control while the actual neural input
    # uses MPS population moments, matching this fixture's arithmetic policy.
    extractor = Wav2Vec2FeatureExtractor(
        feature_size=1,
        sampling_rate=SAMPLE_RATE,
        padding_value=0.0,
        do_normalize=True,
        return_attention_mask=False,
    )
    control = extractor(raw_cpu, sampling_rate=SAMPLE_RATE, return_tensors="pt").input_values
    raw = torch.from_numpy(raw_cpu).to(device="mps")
    normalized = (raw - raw.mean()) * torch.rsqrt(raw.var(unbiased=False) + 1e-7)
    normalized = normalized.unsqueeze(0)
    # Capture the exact first StableLayerNorm encoder block boundary. Hooks
    # retain MPS tensors; only the terminal fixture export reads them to CPU.
    trace: dict[str, torch.Tensor] = {}
    trace["normalized_waveform"] = normalized.detach().clone()

    def capture_output(name: str):
        def hook(_module, _inputs, output):
            value = output[0] if isinstance(output, tuple) else output
            trace[name] = value.detach().clone()
        return hook

    def capture_input(name: str):
        def hook(_module, inputs):
            value = inputs[0]
            trace[name] = value.detach().clone()
        return hook

    def capture_feature_stage(name: str):
        def hook(_module, _inputs, output):
            # Native feature-convolution tensors are sequence-major [T,C].
            trace[name] = output.transpose(1, 2).contiguous().detach().clone()
        return hook

    def capture_conv_output(name: str):
        def hook(_module, _inputs, output):
            # Conv1d returns [B,C,T]; native trace buffers use [T,C].
            trace[name] = output.transpose(1, 2).contiguous().detach().clone()
        return hook

    def capture_position_conv_raw(_module, _inputs, output):
        # Conv1d emits T+1 for even positional kernels. SamePad removes the
        # final value before GELU; retain that exact pre-activation stage.
        position_frames = int(trunk._get_feat_extract_output_lengths(input_samples).item())
        trace["position_conv_raw"] = output[..., :position_frames].transpose(1, 2).contiguous().detach().clone()

    # The whole feature-layer hook is after Conv1d, per-frame LayerNorm, and
    # GELU, matching Wav2Vec2LayerNormConvLayer.forward in pinned source.
    hook_handles = [
        conv_layer.register_forward_hook(capture_feature_stage(f"feature_conv_{i}"))
        for i, conv_layer in enumerate(trunk.feature_extractor.conv_layers)
    ]
    first_conv = trunk.feature_extractor.conv_layers[0]
    second_conv = trunk.feature_extractor.conv_layers[1]
    hook_handles.extend([
        first_conv.conv.register_forward_hook(capture_conv_output("feature_conv_0_raw")),
        # ConvLayer.forward transposes [B,C,T] to [B,T,C] before this module.
        first_conv.layer_norm.register_forward_hook(capture_output("feature_conv_0_norm")),
        second_conv.conv.register_forward_hook(capture_conv_output("feature_conv_1_raw")),
    ])
    hook_handles.extend([
        trunk.feature_projection.layer_norm.register_forward_hook(capture_output("feature_norm")),
        trunk.feature_projection.projection.register_forward_hook(capture_output("feature_projection")),
        trunk.encoder.pos_conv_embed.conv.register_forward_hook(capture_position_conv_raw),
        trunk.encoder.pos_conv_embed.register_forward_hook(capture_output("position_conv")),
        # The encoder passes the positional sum into dropout before its first
        # layer. In eval mode dropout is identity, so this is the native sum.
        trunk.encoder.dropout.register_forward_pre_hook(capture_input("position_add")),
    ])
    position_conv = trunk.encoder.pos_conv_embed.conv
    if hasattr(position_conv, "parametrizations"):
        weight_g = position_conv.parametrizations.weight.original0
        weight_v = position_conv.parametrizations.weight.original1
    else:
        weight_g = position_conv.weight_g
        weight_v = position_conv.weight_v
    # `torch._weight_norm(v, g, dim=2)` uses this exact norm-except-dim
    # reduction in the parametrized positional Conv1d. Keep the trace on MPS.
    with torch.inference_mode():
        position_norm = torch.norm_except_dim(weight_v, 2, 2)
        trace["position_scale"] = (weight_g / position_norm).reshape(-1).clone()

    layer = trunk.encoder.layers[0]
    hook_handles.extend([
        layer.layer_norm.register_forward_hook(capture_output("pre_attention_norm")),
        layer.attention.q_proj.register_forward_hook(capture_output("query")),
        layer.attention.k_proj.register_forward_hook(capture_output("key")),
        layer.attention.v_proj.register_forward_hook(capture_output("value")),
        layer.attention.out_proj.register_forward_pre_hook(capture_input("attention")),
        layer.attention.register_forward_hook(capture_output("attention_projection")),
        layer.final_layer_norm.register_forward_pre_hook(capture_input("attention_residual")),
        layer.final_layer_norm.register_forward_hook(capture_output("ffn_norm")),
        layer.feed_forward.intermediate_dense.register_forward_hook(capture_output("ffn_pre_gelu")),
        layer.feed_forward.intermediate_dropout.register_forward_hook(capture_output("ffn_post_gelu")),
        layer.feed_forward.output_dense.register_forward_hook(capture_output("ffn_output")),
        layer.register_forward_hook(capture_output("layer_output")),
    ])
    with torch.inference_mode():
        result = trunk(normalized, output_hidden_states=True, return_dict=True)
    for hook_handle in hook_handles:
        hook_handle.remove()

    states = torch.cat(result.hidden_states, dim=0)
    feature_frames = int(trunk._get_feat_extract_output_lengths(input_samples).item())
    feature_rate = SAMPLE_RATE // int(np.prod(model_config["conv_strides"]))
    interpolation_frames = feature_frames * VIDEO_FPS // feature_rate
    interpolated = F.interpolate(
        states.transpose(1, 2),
        size=interpolation_frames,
        align_corners=True,
        mode="linear",
    ).transpose(1, 2)
    indices, repeats, padded_frames = sample_indices(interpolation_frames, target_frames)
    bucket_count = len(indices)
    buckets = torch.zeros((states.shape[0], bucket_count, states.shape[2]),
                          device="mps", dtype=torch.float32)
    for bucket, source_index in enumerate(indices):
        if source_index < interpolation_frames:
            buckets[:, bucket, :] = interpolated[:, source_index, :]
    output = buckets.permute(0, 2, 1).contiguous()

    # Single terminal sync/readback: the native test compares against exported
    # host values, while all reference neural operations above ran on MPS.
    torch.mps.synchronize()
    normalized_cpu = normalized.detach().to(device="cpu").float().reshape(-1)
    control_cpu = control[0].float()
    normalization_control_max_abs = float((normalized_cpu - control_cpu).abs().max())
    if normalization_control_max_abs > 2e-5:
        raise RuntimeError(
            "MPS population normalization diverges from Wav2Vec2FeatureExtractor: "
            f"max_abs={normalization_control_max_abs}"
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    if production:
        assert args.checkpoint is not None
        checkpoint_dir = args.checkpoint if args.checkpoint.is_dir() else args.checkpoint.parent
        weights_path = args.checkpoint if args.checkpoint.is_file() else checkpoint_dir / "model.safetensors"
    else:
        weights_path = args.output.with_suffix(".safetensors")
        save_file(state, str(weights_path), metadata={
            "transformers_version": PINNED_TRANSFORMERS_VERSION,
            "modeling_source_sha256": PINNED_MODELING_SHA256,
            "arithmetic": "F32",
        })
    output_cpu = output.detach().to(device="cpu").contiguous()
    expected = None
    expected_file = None
    expected_file_sha256 = None
    if production:
        expected_file = args.output.with_suffix(".expected.safetensors")
        expected_tensors = {
            # safetensors rejects shared storage between entries. Each saved
            # checkpoint is therefore an independent terminal export, even
            # when it is a view of the same MPS-produced output tensor.
            "output": output_cpu.reshape(-1).contiguous().clone(),
        }
        for i in range(output_cpu.shape[0]):
            expected_tensors[f"hidden_state_{i}"] = output_cpu[i].reshape(-1).contiguous().clone()
        for name, tensor in trace.items():
            expected_tensors[f"trace.{name}"] = (
                tensor.detach().to(device="cpu").contiguous().reshape(-1).clone()
            )
        save_file(expected_tensors, str(expected_file), metadata={
            "transformers_version": PINNED_TRANSFORMERS_VERSION,
            "model_revision": PINNED_MODEL_REVISION,
            "modeling_source_sha256": PINNED_MODELING_SHA256,
            "arithmetic": "F32",
            "device": "mps",
        })
        expected_file_sha256 = sha256(expected_file)
    else:
        expected = {
            f"hidden_state_{i}": output_cpu[i].reshape(-1).tolist()
            for i in range(output_cpu.shape[0])
        }
        expected["output"] = output_cpu.reshape(-1).tolist()
        expected["trace"] = {
            name: tensor.detach().to(device="cpu").contiguous().reshape(-1).tolist()
            for name, tensor in trace.items()
        }
    config = {
        "input_samples": input_samples,
        **model_config,
        "video_fps": VIDEO_FPS,
        "target_fps": TARGET_FPS,
        "target_frames": target_frames,
    }
    fixture = {
        "mode": "production" if production else "tiny",
        "transformers_version": transformers.__version__,
        "model_revision": PINNED_MODEL_REVISION,
        "modeling_source": source_path,
        "modeling_source_sha256": PINNED_MODELING_SHA256,
        "weight_sha256": None if production else hashlib.sha256(weights_path.read_bytes()).hexdigest(),
        "weights": str(weights_path.resolve()),
        **checkpoint_metadata,
        "device": "mps",
        "arithmetic": "F32",
        "normalization_control_max_abs": normalization_control_max_abs,
        "seed": 1729,
        "config": config,
        "input_values": raw_cpu.tolist(),
        "feature_frames": feature_frames,
        "interpolation_frames": interpolation_frames,
        "bucket_repeats": repeats,
        "bucket_frames": bucket_count,
        "padded_interpolation_frames": padded_frames,
        "sample_indices": indices,
        "output_shape": list(output.shape),
        "trace_shapes": {
            name: list(tensor.shape) for name, tensor in trace.items()
        },
        "expected": expected,
        "expected_safetensors": str(expected_file.resolve()) if expected_file else None,
        "expected_safetensors_sha256": expected_file_sha256,
        "memory_budget_bytes": args.memory_budget_bytes if production else 512 * 1024 * 1024,
    }
    args.output.write_text(json.dumps(fixture, allow_nan=False, separators=(",", ":")))
    print(json.dumps({
        "fixture": str(args.output),
        "weights": str(weights_path),
        "checkpoint_bytes": checkpoint_metadata.get("checkpoint_bytes"),
        "expected_safetensors": str(expected_file) if expected_file else None,
        "hidden_states": len(result.hidden_states),
        "output_shape": list(output.shape),
        "bucket_repeats": repeats,
        "modeling_source_sha256": PINNED_MODELING_SHA256,
    }))


if __name__ == "__main__":
    main()
