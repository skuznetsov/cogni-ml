#!/usr/bin/env python3
"""Create a pinned, three-arm Qwen-Image 2.1 LoRA visual comparison.

The arms are a four-evaluation frozen-base trajectory, a three-evaluation
hybrid using the deterministically seeded zero-B adapter, and the same hybrid
using the saved two-step rank-4 adapter. The adapter is active only for the
trained coarse sigma[0] -> sigma[2] interval; the last two evaluations use the
frozen base. This is an illustrative bounded reference control, not a claim
that the checkpoint is a generally capable 3-5-step student or that this
training-side FP32 Euler evaluator is bit-identical to the full pipeline.

Example (all inputs are local and hash-pinned; output must not already exist):
    python3 scripts/qwen_image21_lora_visual_compare.py \\
        --model-dir /path/to/pinned/transformer \\
        --vae-model-dir /path/to/pinned/model-root \\
        --conditioning /path/to/red-cube/manifest.json \\
        --heldout-conditioning /path/to/castle/manifest.json \\
        --heldout-manifest-sha256 <castle-manifest-sha256> \\
        --heldout-payload-sha256 <castle-payload-sha256> \\
        --checkpoint-dir /path/to/rank4-checkpoint \\
        --checkpoint-manifest-sha256 <checkpoint-manifest-sha256> \\
        --output-dir /path/to/new/visual-compare
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import os
import re
import shutil
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import qwen_image21_lora_adapter_checkpoint as adapter_checkpoint
import qwen_image21_lora_adapter_replay as adapter_replay
import qwen_image21_lora_distill_control as distill
import qwen_image21_lora_grad_probe as probe
import qwen_image21_vae_decode as vae_decode


BASE4_STEP_INDICES = (0, 1, 2, 3)
HYBRID3_STEP_INDICES = (0, 2, 3)
TRAIN_PROMPT = "red cube"
SAVED_OPTIMIZER_STEPS = 2
PARITY_ABS_TOL = 1e-6
PARITY_REL_TOL = 1e-6
CHECKPOINT_MANIFEST_SHA256 = (
    "266572b3d0f21b70b83c57e1ee80a8b9d5cb3d30fa62ce722e8ed9f2c0bce7f3"
)
CHECKPOINT_PAYLOAD_SHA256 = (
    "e124404bce8842d056242f216ff89a83db8fd4b63193affea19a6d12c6101f41"
)
HELDOUT_MANIFEST_SHA256 = (
    "17a57ee3b07a64c0bf0f9a559ea1e721d75e4ebea95f3930abe21314fea1c01f"
)
HELDOUT_PAYLOAD_SHA256 = (
    "921cdc9a14685877887733ab0f032cc8cfb2b258d421edfdb38e00e8497f4ea6"
)
VAE_CONFIG_SHA256 = "9785d527b278cb8b210e9f48a8028d92a4190968ce7e24c8fc85d9d1b82f6ba2"
VAE_WEIGHTS_SHA256 = "a07a1b7c4ee2966a1b3bdc37de9b4f983d56937e46619f709a80b6e490675417"
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


@dataclass(frozen=True)
class Trajectory:
    final_latents: Any
    steps: tuple[dict[str, Any], ...]


def _validate_schedule_contract(schedule: probe.FlowMatchSchedule) -> None:
    """Require the complete pinned four-step schedule, including its terminal zero."""
    distill._validate_schedule(schedule)
    if schedule.teacher_step_indices != (0, 1) or schedule.student_step_indices != (0,):
        raise ValueError(
            "visual comparison requires the pinned two-fine/one-coarse training route"
        )
    if len(schedule.sigmas) != 5 or len(schedule.timesteps) != 4:
        raise ValueError(
            "visual comparison requires four timestep evaluations and five sigma nodes"
        )


def _route_sigma_delta(
    schedule: probe.FlowMatchSchedule, schedule_index: int, interval: str
) -> float:
    if interval == "coarse":
        if schedule_index != 0:
            raise ValueError(
                "the trained coarse interval is only valid at schedule index zero"
            )
        return schedule.sigmas[2] - schedule.sigmas[0]
    if interval == "base":
        return schedule.sigmas[schedule_index + 1] - schedule.sigmas[schedule_index]
    raise ValueError(f"unknown Euler interval kind: {interval}")


def _validate_model_adapter(model: Any) -> Any:
    target = dict(model.named_modules()).get(probe.TARGET_MODULE)
    if (
        target is None
        or not hasattr(target, "lora_A")
        or not hasattr(target, "lora_B")
        or probe.ADAPTER_NAME not in target.lora_A
        or probe.ADAPTER_NAME not in target.lora_B
    ):
        raise ValueError(f"expected rank-4 adapter at {probe.TARGET_MODULE}")
    if not isinstance(getattr(target, "disable_adapters", None), bool):
        raise ValueError("target PEFT layer does not expose its adapter-disabled state")
    if not callable(getattr(model, "disable_adapters", None)):
        raise ValueError(
            "pinned transformer cannot disable the LoRA adapter for base tail steps"
        )
    if not callable(getattr(model, "enable_adapters", None)):
        raise ValueError(
            "pinned transformer cannot re-enable the LoRA adapter after base tail steps"
        )
    return target


def _require_adapter_enabled(model: Any, expected_enabled: bool) -> None:
    target = _validate_model_adapter(model)
    observed_enabled = not target.disable_adapters
    if observed_enabled is not expected_enabled:
        expected = "enabled" if expected_enabled else "disabled"
        raise ValueError(f"LoRA target layer must be {expected} at this route position")


def _one_euler_evaluation(
    model: Any,
    bundle: Any,
    state: Any,
    schedule: probe.FlowMatchSchedule,
    schedule_index: int,
    *,
    interval: str,
    device: Any,
    model_dtype: Any,
    adapter_enabled: bool | None,
) -> tuple[Any, dict[str, Any]]:
    import torch

    velocity = probe._forward_velocity(
        model,
        bundle,
        state,
        schedule.timesteps[schedule_index],
        device,
        model_dtype,
    ).float()
    if velocity.shape != state.shape or not probe._is_finite(velocity):
        raise ValueError("visual comparison received an invalid transformer velocity")
    if interval == "coarse":
        sigma_from = schedule.sigmas[0]
        sigma_to = schedule.sigmas[2]
    else:
        sigma_from = schedule.sigmas[schedule_index]
        sigma_to = schedule.sigmas[schedule_index + 1]
    # Keep the saved training evaluator's explicit FP32 Euler recurrence. This
    # intentionally does not claim full-pipeline scheduler dtype parity.
    next_state = probe.euler_endpoint(state, (velocity,), (sigma_from, sigma_to))
    if next_state.dtype != torch.float32 or not probe._is_finite(next_state):
        raise ValueError("FP32 Euler state became invalid during visual comparison")
    return next_state, {
        "schedule_index": schedule_index,
        "timestep": schedule.timesteps[schedule_index],
        "sigma_from": sigma_from,
        "sigma_to": sigma_to,
        "interval": interval,
        "adapter_enabled": adapter_enabled,
        "velocity_dtype_for_euler": "float32",
        "state_dtype_after_step": "float32",
    }


def sample_base4(
    model: Any,
    bundle: Any,
    schedule: probe.FlowMatchSchedule,
    *,
    device: Any,
    model_dtype: Any,
) -> Trajectory:
    """Run all four adjacent pinned steps using the frozen base transformer."""
    import torch

    _validate_schedule_contract(schedule)
    if model.training:
        raise ValueError("base transformer must be in eval mode")
    state = (
        bundle.initial_target_latents.detach()
        .to(device=device, dtype=torch.float32)
        .clone()
    )
    if tuple(state.shape) != (1, 256, 64):
        raise ValueError("visual comparison initial latents must have shape [1,256,64]")
    steps = []
    with torch.inference_mode():
        for schedule_index in BASE4_STEP_INDICES:
            state, step = _one_euler_evaluation(
                model,
                bundle,
                state,
                schedule,
                schedule_index,
                interval="base",
                device=device,
                model_dtype=model_dtype,
                adapter_enabled=None,
            )
            steps.append(step)
    return Trajectory(state.detach().to(device="cpu"), tuple(steps))


def sample_hybrid3(
    model: Any,
    bundle: Any,
    schedule: probe.FlowMatchSchedule,
    *,
    device: Any,
    model_dtype: Any,
    adapter_label: str,
) -> Trajectory:
    """Use the adapter for trained 0->2 coarse interval, then frozen base tail."""
    import torch

    _validate_schedule_contract(schedule)
    if model.training:
        raise ValueError("hybrid transformer must be in eval mode")
    if not isinstance(adapter_label, str) or not adapter_label:
        raise ValueError("hybrid adapter label must be non-empty")
    target = _validate_model_adapter(model)
    if target.disable_adapters:
        raise ValueError("hybrid route must begin with the LoRA adapter enabled")
    state = (
        bundle.initial_target_latents.detach()
        .to(device=device, dtype=torch.float32)
        .clone()
    )
    if tuple(state.shape) != (1, 256, 64):
        raise ValueError("visual comparison initial latents must have shape [1,256,64]")
    steps = []
    with torch.inference_mode():
        state, step = _one_euler_evaluation(
            model,
            bundle,
            state,
            schedule,
            0,
            interval="coarse",
            device=device,
            model_dtype=model_dtype,
            adapter_enabled=True,
        )
        step["adapter_label"] = adapter_label
        steps.append(step)

        # Do not let the checkpoint act at an interval it was not trained on.
        disable_started = True
        try:
            model.disable_adapters()
            _require_adapter_enabled(model, False)
            for schedule_index in (2, 3):
                state, step = _one_euler_evaluation(
                    model,
                    bundle,
                    state,
                    schedule,
                    schedule_index,
                    interval="base",
                    device=device,
                    model_dtype=model_dtype,
                    adapter_enabled=False,
                )
                step["adapter_label"] = adapter_label
                steps.append(step)
        finally:
            if disable_started:
                model.enable_adapters()
                _require_adapter_enabled(model, True)
    return Trajectory(state.detach().to(device="cpu"), tuple(steps))


def _seeded_adapter_b_is_zero(model: Any) -> bool:
    import torch

    target = _validate_model_adapter(model)
    a_weight = target.lora_A[probe.ADAPTER_NAME].weight
    b_weight = target.lora_B[probe.ADAPTER_NAME].weight
    if tuple(a_weight.shape) != (4, 4096) or tuple(b_weight.shape) != (4096, 4):
        raise ValueError(
            "seeded adapter must be exactly rank four on the pinned 4096-wide Q projection"
        )
    if a_weight.dtype != torch.float32 or b_weight.dtype != torch.float32:
        raise ValueError("seeded rank-4 adapter parameters must be FP32")
    if not bool(torch.isfinite(a_weight).all().item()) or not bool(
        torch.isfinite(b_weight).all().item()
    ):
        raise ValueError("seeded adapter contains a non-finite parameter")
    if bool(torch.count_nonzero(b_weight).item()):
        raise ValueError(
            "seeded hybrid control requires the exact zero-B initialization"
        )
    return True


def check_seeded_coarse_parity(
    model: Any,
    bundle: Any,
    schedule: probe.FlowMatchSchedule,
    *,
    device: Any,
    model_dtype: Any,
) -> dict[str, Any]:
    """Falsify an unexpected first-interval effect from the zero-B adapter."""
    import torch

    _validate_schedule_contract(schedule)
    _seeded_adapter_b_is_zero(model)
    _require_adapter_enabled(model, True)
    initial = bundle.initial_target_latents.detach().to(
        device=device, dtype=torch.float32
    )
    with torch.inference_mode():
        enabled_velocity = probe._forward_velocity(
            model, bundle, initial, schedule.timesteps[0], device, model_dtype
        ).float()
        enabled_endpoint = probe.euler_endpoint(
            initial, (enabled_velocity,), (schedule.sigmas[0], schedule.sigmas[2])
        )
        disable_started = True
        try:
            model.disable_adapters()
            _require_adapter_enabled(model, False)
            disabled_velocity = probe._forward_velocity(
                model, bundle, initial, schedule.timesteps[0], device, model_dtype
            ).float()
            disabled_endpoint = probe.euler_endpoint(
                initial, (disabled_velocity,), (schedule.sigmas[0], schedule.sigmas[2])
            )
        finally:
            if disable_started:
                model.enable_adapters()
                _require_adapter_enabled(model, True)
    velocity_delta = (enabled_velocity - disabled_velocity).detach().float()
    endpoint_delta = (enabled_endpoint - disabled_endpoint).detach().float()
    velocity_max = float(velocity_delta.abs().max().item())
    endpoint_rmse = float(endpoint_delta.square().mean().sqrt().item())
    endpoint_max = float(endpoint_delta.abs().max().item())
    if not all(
        math.isfinite(value) for value in (velocity_max, endpoint_rmse, endpoint_max)
    ):
        raise ValueError("seeded adapter/base parity produced non-finite metrics")
    if not torch.allclose(
        enabled_velocity, disabled_velocity, rtol=PARITY_REL_TOL, atol=PARITY_ABS_TOL
    ) or not torch.allclose(
        enabled_endpoint, disabled_endpoint, rtol=PARITY_REL_TOL, atol=PARITY_ABS_TOL
    ):
        raise ValueError("seeded adapter failed zero-B first coarse-step/base parity")
    return {
        "scope": "same-base, same-noise first coarse interval only",
        "velocity_max_abs": velocity_max,
        "coarse_endpoint_rmse": endpoint_rmse,
        "coarse_endpoint_max_abs": endpoint_max,
        "absolute_tolerance": PARITY_ABS_TOL,
        "relative_tolerance": PARITY_REL_TOL,
        "parity_passed": True,
    }


def latent_distance(left: Any, right: Any) -> dict[str, float]:
    import torch

    if tuple(left.shape) != tuple(right.shape):
        raise ValueError("latent distance requires equal tensor shapes")
    delta = left.detach().to(device="cpu", dtype=torch.float32) - right.detach().to(
        device="cpu", dtype=torch.float32
    )
    if not bool(torch.isfinite(delta).all().item()):
        raise ValueError("latent distance contains NaN or infinity")
    return {
        "mae": float(delta.abs().mean().item()),
        "rmse": float(delta.square().mean().sqrt().item()),
        "max_abs": float(delta.abs().max().item()),
    }


def rgba_distance(left: Any, right: Any) -> dict[str, float | int | str]:
    import numpy as np

    left = np.asarray(left)
    right = np.asarray(right)
    if left.shape != right.shape or left.ndim != 3 or left.shape[-1] != 4:
        raise ValueError("RGBA distance requires equal HxWx4 arrays")
    if left.dtype != np.uint8 or right.dtype != np.uint8:
        raise ValueError("RGBA distance requires uint8 images")
    delta = left.astype(np.float32) - right.astype(np.float32)
    absolute = np.abs(delta)
    different_values = int(np.count_nonzero(delta))
    different_pixels = int(np.count_nonzero(np.any(delta != 0, axis=-1)))
    alpha_delta = delta[..., 3]
    return {
        "mae": float(absolute.mean()),
        "rmse": float(np.sqrt(np.mean(np.square(delta)))),
        "max_abs": float(absolute.max()),
        "different_channel_values": different_values,
        "total_channel_values": int(delta.size),
        "different_pixels": different_pixels,
        "total_pixels": int(delta.shape[0] * delta.shape[1]),
        "different_alpha_values": int(np.count_nonzero(alpha_delta)),
        "channel_scale": "raw integer RGBA levels on a 0-255 scale",
    }


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def verify_vae_files(
    config_path: str | Path,
    weights_path: str | Path,
    *,
    expected_config_sha256: str = VAE_CONFIG_SHA256,
    expected_weights_sha256: str = VAE_WEIGHTS_SHA256,
) -> dict[str, str]:
    for digest, label in (
        (expected_config_sha256, "VAE config"),
        (expected_weights_sha256, "VAE weights"),
    ):
        if not isinstance(digest, str) or _SHA256_RE.fullmatch(digest) is None:
            raise ValueError(
                f"{label} SHA256 pin must be a lowercase 64-character digest"
            )
    try:
        config_sha256 = probe.sha256_file(config_path)
        weights_sha256 = probe.sha256_file(weights_path)
    except OSError as exc:
        raise ValueError("pinned VAE config or weights are missing/unreadable") from exc
    if config_sha256 != expected_config_sha256:
        raise ValueError("VAE config SHA256 mismatch")
    if weights_sha256 != expected_weights_sha256:
        raise ValueError("VAE weights SHA256 mismatch")
    return {"config_sha256": config_sha256, "weights_sha256": weights_sha256}


def verify_vae_snapshot(model_dir: str | Path) -> dict[str, Any]:
    root = Path(model_dir).resolve(strict=True)
    vae_dir = root / "vae"
    config_path = vae_dir / "config.json"
    weights_path = vae_dir / "diffusion_pytorch_model.safetensors"
    hashes = verify_vae_files(config_path, weights_path)
    metadata_dir = root / ".cache" / "huggingface" / "download" / "vae"
    metadata = {}
    for filename in ("config.json", "diffusion_pytorch_model.safetensors"):
        metadata_path = metadata_dir / f"{filename}.metadata"
        try:
            lines = metadata_path.read_text(encoding="utf-8").splitlines()
        except OSError as exc:
            raise ValueError(
                f"missing pinned-revision VAE metadata for {filename}"
            ) from exc
        if len(lines) < 2 or lines[0].lower() != probe.MODEL_REVISION:
            raise ValueError(
                f"VAE {filename} is not attested to pinned model revision {probe.MODEL_REVISION}"
            )
        metadata[filename] = {"revision": lines[0], "metadata_digest": lines[1]}
    try:
        config = json.loads(config_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError("pinned VAE config cannot be parsed") from exc
    if (
        config.get("z_dim") != 64
        or config.get("out_channels") != 4
        or config.get("scale_factor_spatial") != 16
    ):
        raise ValueError("pinned VAE config geometry differs from Qwen-Image 2.1")
    return {
        "model_dir": str(root),
        "model_revision": probe.MODEL_REVISION,
        **hashes,
        "revision_metadata": metadata,
    }


def validate_output_dir(output_dir: str | Path, *, repo_root: str | Path) -> Path:
    output = Path(output_dir).expanduser().resolve(strict=False)
    repo = Path(repo_root).resolve(strict=True)
    try:
        output.relative_to(repo)
    except ValueError:
        pass
    else:
        raise ValueError("output directory must be outside the repository")
    if output.exists() or output.is_symlink():
        raise ValueError(f"output directory must not exist: {output}")
    if output == output.parent:
        raise ValueError("output directory must name a new child path")
    return output


def _verify_checkpoint_inputs(args: argparse.Namespace) -> dict[str, Any]:
    manifest_sha256 = adapter_replay.verify_checkpoint_manifest_pin(
        args.checkpoint_dir, args.checkpoint_manifest_sha256
    )
    if manifest_sha256 != CHECKPOINT_MANIFEST_SHA256:
        raise ValueError(
            "checkpoint is not the pinned two-step rank-4 visual-comparison adapter"
        )
    manifest_path = Path(args.checkpoint_dir) / adapter_checkpoint.MANIFEST_NAME
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError("cannot read pinned adapter checkpoint manifest") from exc
    payload = manifest.get("payload")
    training = manifest.get("training")
    if not isinstance(payload, dict) or not isinstance(training, dict):
        raise ValueError("pinned adapter checkpoint metadata is malformed")
    payload_path = Path(args.checkpoint_dir) / adapter_checkpoint.PAYLOAD_NAME
    payload_sha256 = probe.sha256_file(payload_path)
    if (
        payload_sha256 != CHECKPOINT_PAYLOAD_SHA256
        or payload.get("sha256") != payload_sha256
    ):
        raise ValueError("checkpoint payload is not the pinned saved rank-4 adapter")
    if training.get("optimizer_steps") != SAVED_OPTIMIZER_STEPS:
        raise ValueError(
            "visual comparison requires the saved two-optimizer-step checkpoint"
        )
    return {
        "manifest_sha256": manifest_sha256,
        "payload_sha256": payload_sha256,
        "optimizer_steps": SAVED_OPTIMIZER_STEPS,
    }


def _validate_bundle_pair(
    train_bundle: Any, heldout_bundle: Any, schedule: Any
) -> None:
    adapter_replay.validate_conditioning_bundles(train_bundle, heldout_bundle, schedule)
    distill._validate_bundle_geometry(train_bundle, schedule, "training")
    distill._validate_bundle_geometry(heldout_bundle, schedule, "held-out")
    if not bool(re.search(r"\bcastle\b", heldout_bundle.prompt, flags=re.IGNORECASE)):
        raise ValueError("held-out conditioning prompt must identify a castle")
    if not bool(
        (train_bundle.initial_target_latents == heldout_bundle.initial_target_latents)
        .all()
        .item()
    ):
        raise ValueError(
            "red-cube and castle bundles must contain byte-identical same-seed initial noise"
        )
    if train_bundle.initial_target_latents.detach().cpu().contiguous().view(
        -1
    ).shape != (256 * 64,):
        raise ValueError("conditioning initial latent geometry changed")


def _initial_noise_sha256(bundle: Any) -> str:
    import torch

    array = bundle.initial_target_latents.detach().to(device="cpu", dtype=torch.float32)
    payload = array.contiguous().numpy().astype("<f4", copy=False).tobytes()
    return _sha256_bytes(payload)


def _adapter_pair_hashes(model: Any) -> dict[str, str]:
    import torch

    target = _validate_model_adapter(model)
    result = {}
    for label in ("lora_A", "lora_B"):
        tensor = getattr(target, label)[probe.ADAPTER_NAME].weight
        raw = (
            tensor.detach()
            .to(device="cpu", dtype=torch.float32)
            .contiguous()
            .numpy()
            .tobytes()
        )
        result[f"{label}_sha256"] = _sha256_bytes(raw)
    return result


def _base_projection_sha256(model: Any) -> str:
    import torch

    name = f"{probe.TARGET_MODULE}.weight"
    parameter = dict(model.named_parameters()).get(name)
    if parameter is None:
        raise ValueError("fresh verified base lacks the exact Q projection weight")
    # Promoting BF16 values to FP32 is injective and avoids NumPy's lack of
    # direct BF16 byte-array support while retaining the quantized base values.
    raw = (
        parameter.detach()
        .to(device="cpu", dtype=torch.float32)
        .contiguous()
        .numpy()
        .tobytes()
    )
    return _sha256_bytes(raw)


def _decode_latents(vae: Any, torch: Any, latents: Any) -> Any:
    import numpy as np

    if tuple(latents.shape) != (1, 256, 64):
        raise ValueError("VAE decode requires one [256,64] final latent tensor")
    values = latents.detach().to(device="cpu", dtype=torch.float32).contiguous().numpy()
    payload = values[0].astype("<f4", copy=False).tobytes()
    latent_array = vae_decode.tokens_hwc_to_vae_input(payload, 16, 16)
    model_latents = torch.from_numpy(latent_array).to(device="cpu", dtype=torch.float32)
    means = torch.tensor(vae.config.latents_mean, dtype=torch.float32).view(
        1, 64, 1, 1, 1
    )
    stds = torch.tensor(vae.config.latents_std, dtype=torch.float32).view(
        1, 64, 1, 1, 1
    )
    with torch.inference_mode():
        decoded = vae.decode(model_latents * stds + means, return_dict=False)[0]
    expected = (1, 4, 1, 256, 256)
    if tuple(decoded.shape) != expected:
        raise RuntimeError(
            f"pinned Qwen VAE returned {tuple(decoded.shape)}, expected {expected}"
        )
    frame = decoded[0, :, 0].float()
    if not bool(torch.isfinite(frame).all().item()):
        raise RuntimeError("pinned Qwen VAE returned non-finite decoded pixels")
    pixels = (
        frame.permute(1, 2, 0)
        .div(2.0)
        .add(0.5)
        .clamp(0.0, 1.0)
        .mul(255.0)
        .round()
        .to(torch.uint8)
        .cpu()
        .numpy()
    )
    if pixels.shape != (256, 256, 4) or pixels.dtype != np.uint8:
        raise RuntimeError("pinned Qwen VAE output is not a 256x256 uint8 RGBA image")
    return pixels


def _source_hashes(
    *,
    schedule_path: Path,
    train_bundle: Any,
    heldout_bundle: Any,
    checkpoint_identity: dict[str, Any],
    vae_identity: dict[str, Any],
) -> dict[str, Any]:
    files = {
        "visual_compare": Path(__file__).resolve(),
        "probe": Path(probe.__file__).resolve(),
        "distill": Path(distill.__file__).resolve(),
        "checkpoint": Path(adapter_checkpoint.__file__).resolve(),
        "replay": Path(adapter_replay.__file__).resolve(),
        "vae_decoder": Path(vae_decode.__file__).resolve(),
        "flow_fixture": Path(schedule_path).resolve(),
    }
    source_digests = {name: probe.sha256_file(path) for name, path in files.items()}
    train_digests = adapter_checkpoint._bundle_digests(train_bundle, "train")
    heldout_digests = adapter_checkpoint._bundle_digests(heldout_bundle, "held-out")
    model_snapshot = {
        "config_sha256": probe.CONFIG_SHA256,
        "index_sha256": probe.INDEX_SHA256,
        "shards": dict(probe.SHARD_SHA256),
    }
    return {
        "python_sources_and_fixture": source_digests,
        "model_snapshot": model_snapshot,
        "train_conditioning": train_digests,
        "heldout_conditioning": heldout_digests,
        "adapter_checkpoint": checkpoint_identity,
        "vae": vae_identity,
        "model_revision": probe.MODEL_REVISION,
        "diffusers_commit": probe.DIFFUSERS_COMMIT,
        "flow_fixture_sha256": probe.FIXTURE_SHA256,
    }


def _write_new_artifact_dir(
    output_dir: Path, files: dict[str, Any], manifest: dict[str, Any]
) -> None:
    """Stage all six PNGs plus a content manifest, then publish as one directory."""
    import numpy as np
    from PIL import Image

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    if output_dir.exists() or output_dir.is_symlink():
        raise ValueError(f"output directory must not exist: {output_dir}")
    stage_dir = Path(
        tempfile.mkdtemp(prefix=f".{output_dir.name}.stage-", dir=output_dir.parent)
    )
    try:
        png_records = {}
        for filename, pixels in files.items():
            array = np.asarray(pixels)
            if array.shape != (256, 256, 4) or array.dtype != np.uint8:
                raise ValueError(f"refusing invalid decoded image: {filename}")
            image_path = stage_dir / filename
            Image.fromarray(array, mode="RGBA").save(image_path, format="PNG")
            png_records[filename] = {
                "sha256": probe.sha256_file(image_path),
                "bytes": image_path.stat().st_size,
                "width": int(array.shape[1]),
                "height": int(array.shape[0]),
                "mode": "RGBA",
            }
        manifest["images"] = png_records
        manifest_bytes = (json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode(
            "utf-8"
        )
        (stage_dir / "manifest.json").write_bytes(manifest_bytes)
        parsed = json.loads((stage_dir / "manifest.json").read_text(encoding="utf-8"))
        if set(parsed["images"]) != set(files) or len(parsed["images"]) != 6:
            raise ValueError("staged visual comparison must contain exactly six images")
        for filename, record in parsed["images"].items():
            path = stage_dir / filename
            if probe.sha256_file(path) != record["sha256"]:
                raise ValueError(
                    f"staged image SHA256 changed before publication: {filename}"
                )
        if output_dir.exists() or output_dir.is_symlink():
            raise ValueError(f"output directory appeared during staging: {output_dir}")
        os.rename(stage_dir, output_dir)
    finally:
        if stage_dir.exists():
            shutil.rmtree(stage_dir)


def run_comparison(
    args: argparse.Namespace, *, torch_module: Any | None = None
) -> dict[str, Any]:
    """Run two sequential verified transformer loads, then one CPU VAE decode pass."""
    if torch_module is None:
        import torch as torch_module
    torch = torch_module
    repo_root = Path(__file__).resolve().parents[1]
    output_dir = validate_output_dir(args.output_dir, repo_root=repo_root)

    schedule = probe.load_flow_match_schedule(args.schedule)
    _validate_schedule_contract(schedule)
    train_bundle = probe.load_conditioning_bundle(args.conditioning)
    heldout_bundle = distill.load_heldout_conditioning_bundle(
        args.heldout_conditioning,
        expected_manifest_sha256=args.heldout_manifest_sha256,
        expected_payload_sha256=args.heldout_payload_sha256,
    )
    _validate_bundle_pair(train_bundle, heldout_bundle, schedule)
    if (
        args.heldout_manifest_sha256 != HELDOUT_MANIFEST_SHA256
        or args.heldout_payload_sha256 != HELDOUT_PAYLOAD_SHA256
    ):
        raise ValueError(
            "held-out bundle is not the pinned castle visual-comparison fixture"
        )
    checkpoint_identity = _verify_checkpoint_inputs(args)
    vae_identity = verify_vae_snapshot(args.vae_model_dir)
    noise_sha = _initial_noise_sha256(train_bundle)
    if noise_sha != _initial_noise_sha256(heldout_bundle):
        raise ValueError(
            "training and held-out bundles do not share byte-identical initial noise"
        )

    installed_diffusers_commit = probe.require_pinned_diffusers_installation()
    if installed_diffusers_commit != probe.DIFFUSERS_COMMIT:
        raise RuntimeError(
            "installed Diffusers commit does not match the pinned Qwen-Image 2.1 source"
        )
    installed_peft_version, installed_peft_path = (
        probe.require_compatible_peft_installation()
    )
    device = probe._resolve_device(torch, "mps")
    if device.type != "mps":
        raise ValueError(
            "visual comparison is restricted to the pinned MPS/BF16 transformer path"
        )
    model_dtype = torch.bfloat16
    bundles = (("red_cube", train_bundle), ("castle", heldout_bundle))
    memory_guards: list[dict[str, Any]] = []
    latents: dict[str, dict[str, Any]] = {name: {} for name, _bundle in bundles}
    parity: dict[str, Any] = {}

    verified_base = adapter_replay._load_verified_base(
        args, torch=torch, device=device, dtype=model_dtype, memory_guards=memory_guards
    )
    base_model = verified_base.transformer
    adapter_replay._require_base_frozen(base_model)
    base_model_hash = _base_projection_sha256(base_model)
    for name, bundle in bundles:
        latents[name]["base4"] = sample_base4(
            base_model,
            bundle,
            schedule,
            device=device,
            model_dtype=model_dtype,
        ).final_latents

    probe._install_probe_adapter(base_model, adapter_dtype=torch.float32, device=device)
    _seeded_adapter_b_is_zero(base_model)
    seeded_hashes = _adapter_pair_hashes(base_model)
    for name, bundle in bundles:
        parity[name] = check_seeded_coarse_parity(
            base_model,
            bundle,
            schedule,
            device=device,
            model_dtype=model_dtype,
        )
        latents[name]["seeded_hybrid3"] = sample_hybrid3(
            base_model,
            bundle,
            schedule,
            device=device,
            model_dtype=model_dtype,
            adapter_label="seeded_zero_b",
        ).final_latents
        adapter_replay._require_no_gradients(base_model, f"seeded {name}")

    del base_model
    del verified_base
    adapter_replay._release_model_memory(torch, device)

    verified_trained = adapter_replay._load_verified_base(
        args, torch=torch, device=device, dtype=model_dtype, memory_guards=memory_guards
    )
    trained_model = verified_trained.transformer
    adapter_replay._require_base_frozen(trained_model)
    trained_base_hash = _base_projection_sha256(trained_model)
    if trained_base_hash != base_model_hash:
        raise ValueError(
            "fresh trained-adapter base Q projection differs from the baseline/seeded base"
        )
    checkpoint_record = adapter_checkpoint.load_adapter(
        verified_trained,
        args.checkpoint_dir,
        train_bundle=train_bundle,
        heldout_bundle=heldout_bundle,
        flow_fixture_path=args.schedule,
        device=device,
    )
    if (
        checkpoint_record.manifest_sha256 != checkpoint_identity["manifest_sha256"]
        or checkpoint_record.payload_sha256 != checkpoint_identity["payload_sha256"]
    ):
        raise ValueError(
            "loaded checkpoint identity changed after preflight validation"
        )
    trained_hashes = _adapter_pair_hashes(trained_model)
    for name, bundle in bundles:
        latents[name]["trained_hybrid3"] = sample_hybrid3(
            trained_model,
            bundle,
            schedule,
            device=device,
            model_dtype=model_dtype,
            adapter_label="saved_two_step_rank4",
        ).final_latents
        adapter_replay._require_no_gradients(trained_model, f"trained {name}")

    del trained_model
    del verified_trained
    adapter_replay._release_model_memory(torch, device)

    vae_torch, vae, vae_device, vae_dtype = vae_decode._load_local_vae(
        Path(args.vae_model_dir), "cpu", "float32"
    )
    if vae_device != "cpu" or vae_dtype != torch.float32:
        raise RuntimeError(
            "visual comparison requires one CPU/FP32 VAE load after releasing the transformer"
        )
    images: dict[str, Any] = {}
    try:
        for prompt_name in ("red_cube", "castle"):
            for arm in ("base4", "seeded_hybrid3", "trained_hybrid3"):
                filename = f"{prompt_name}_{arm}.png"
                images[filename] = _decode_latents(
                    vae, vae_torch, latents[prompt_name][arm]
                )
    finally:
        del vae
        gc.collect()

    comparisons: dict[str, Any] = {}
    for prompt_name in ("red_cube", "castle"):
        baseline_latent = latents[prompt_name]["base4"]
        baseline_image = images[f"{prompt_name}_base4.png"]
        seeded_image = images[f"{prompt_name}_seeded_hybrid3.png"]
        trained_image = images[f"{prompt_name}_trained_hybrid3.png"]
        comparisons[prompt_name] = {
            "seeded_hybrid3_vs_base4": {
                "latent": latent_distance(
                    latents[prompt_name]["seeded_hybrid3"], baseline_latent
                ),
                "rgba": rgba_distance(seeded_image, baseline_image),
            },
            "trained_hybrid3_vs_base4": {
                "latent": latent_distance(
                    latents[prompt_name]["trained_hybrid3"], baseline_latent
                ),
                "rgba": rgba_distance(trained_image, baseline_image),
            },
            "trained_hybrid3_vs_seeded_hybrid3": {
                "latent": latent_distance(
                    latents[prompt_name]["trained_hybrid3"],
                    latents[prompt_name]["seeded_hybrid3"],
                ),
                "rgba": rgba_distance(trained_image, seeded_image),
            },
        }

    source_hashes = _source_hashes(
        schedule_path=args.schedule,
        train_bundle=train_bundle,
        heldout_bundle=heldout_bundle,
        checkpoint_identity=checkpoint_identity,
        vae_identity=vae_identity,
    )
    report = {
        "schema": "qwen-image21-lora-visual-compare",
        "schema_version": 1,
        "status": "visual_comparison_completed",
        "claim_scope": (
            "two single-seed illustrative comparisons only; not evidence of a generally capable "
            "3-5-step student, image quality, or full-pipeline numerical parity"
        ),
        "method": {
            "description": "bounded reference Diffusers transformer control with training-side FP32 explicit Euler recurrence",
            "bitwise_full_pipeline_parity_claimed": False,
            "cfg_enabled": False,
            "kv_cache_enabled": False,
            "vae_decode": "one pinned local Qwen-Image 2.1 VAE instance, CPU/FP32, after transformer release",
            "arms": {
                "base4": [
                    "base sigma[0]->sigma[1] at timestep[0]",
                    "base sigma[1]->sigma[2] at timestep[1]",
                    "base sigma[2]->sigma[3] at timestep[2]",
                    "base sigma[3]->sigma[4] at timestep[3]",
                ],
                "seeded_hybrid3": [
                    "seeded zero-B rank-4 adapter sigma[0]->sigma[2] at timestep[0]",
                    "base with adapter disabled sigma[2]->sigma[3] at timestep[2]",
                    "base with adapter disabled sigma[3]->sigma[4] at timestep[3]",
                ],
                "trained_hybrid3": [
                    "saved two-step rank-4 adapter sigma[0]->sigma[2] at timestep[0]",
                    "base with adapter disabled sigma[2]->sigma[3] at timestep[2]",
                    "base with adapter disabled sigma[3]->sigma[4] at timestep[3]",
                ],
            },
            "same_noise_sha256": noise_sha,
            "seeded_first_coarse_parity": parity,
            "latent_update": "state + (sigma_to - sigma_from) * velocity.float(); FP32 throughout",
        },
        "model": {
            "repo": "Qwen/Qwen-Image-2.1",
            "revision": probe.MODEL_REVISION,
            "diffusers_commit": probe.DIFFUSERS_COMMIT,
            "installed_diffusers_commit": installed_diffusers_commit,
            "installed_peft_version": installed_peft_version,
            "installed_peft_path": installed_peft_path,
            "device": str(device),
            "transformer_dtype": "bfloat16",
            "adapter_dtype": "float32",
            "base_target_projection_sha256_load1": base_model_hash,
            "base_target_projection_sha256_load2": trained_base_hash,
            "seeded_adapter_pair_sha256": seeded_hashes,
            "saved_adapter_pair_sha256": trained_hashes,
            "checkpoint_optimizer_steps": checkpoint_identity["optimizer_steps"],
        },
        "conditioning": {
            "red_cube_prompt": train_bundle.prompt,
            "castle_prompt": heldout_bundle.prompt,
            "red_cube_geometry": {
                "width": 256,
                "height": 256,
                "latent_tokens": 256,
                "latent_channels": 64,
            },
            "castle_geometry": {
                "width": 256,
                "height": 256,
                "latent_tokens": 256,
                "latent_channels": 64,
            },
            "seed": 7,
        },
        "schedule": {
            "fixture_sha256": schedule.fixture_sha256,
            "sigmas": list(schedule.sigmas),
            "timesteps": list(schedule.timesteps),
            "base4_indices": list(BASE4_STEP_INDICES),
            "hybrid3_indices": list(HYBRID3_STEP_INDICES),
        },
        "memory_guards": memory_guards,
        "source_hashes": source_hashes,
        "distances_from_base4": comparisons,
        "distance_semantics": {
            "latent": "FP32 values in model latent units",
            "rgba": "raw integer channel levels on a 0-255 scale",
            "quality_metric_or_threshold": "none",
        },
    }
    _write_new_artifact_dir(output_dir, images, report)
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model-dir",
        required=True,
        type=Path,
        help="local pinned transformer directory",
    )
    parser.add_argument(
        "--vae-model-dir",
        required=True,
        type=Path,
        help="local pinned model root containing vae/",
    )
    parser.add_argument(
        "--conditioning",
        required=True,
        type=Path,
        help="pinned red-cube conditioning manifest",
    )
    parser.add_argument(
        "--heldout-conditioning",
        required=True,
        type=Path,
        help="pinned castle conditioning manifest",
    )
    parser.add_argument("--heldout-manifest-sha256", required=True)
    parser.add_argument("--heldout-payload-sha256", required=True)
    parser.add_argument(
        "--checkpoint-dir",
        required=True,
        type=Path,
        help="saved rank-4 adapter checkpoint",
    )
    parser.add_argument("--checkpoint-manifest-sha256", required=True)
    parser.add_argument(
        "--schedule",
        type=Path,
        default=Path(__file__).resolve().parents[1]
        / "spec/fixtures/qwen_image21_flow_match_diffusers.json",
        help="pinned full four-step FlowMatch fixture",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        type=Path,
        help="new directory for six PNGs and manifest",
    )
    parser.add_argument("--mps-memory-fraction", type=float, default=0.70)
    return parser


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = build_parser()
    args = parser.parse_args(argv)
    for option, value in (
        ("--heldout-manifest-sha256", args.heldout_manifest_sha256),
        ("--heldout-payload-sha256", args.heldout_payload_sha256),
        ("--checkpoint-manifest-sha256", args.checkpoint_manifest_sha256),
    ):
        if _SHA256_RE.fullmatch(value or "") is None:
            parser.error(f"{option} must contain 64 lowercase hexadecimal characters")
    if args.heldout_manifest_sha256 != HELDOUT_MANIFEST_SHA256:
        parser.error(
            "--heldout-manifest-sha256 does not match the pinned castle bundle"
        )
    if args.heldout_payload_sha256 != HELDOUT_PAYLOAD_SHA256:
        parser.error(
            "--heldout-payload-sha256 does not match the pinned castle payload"
        )
    if args.checkpoint_manifest_sha256 != CHECKPOINT_MANIFEST_SHA256:
        parser.error(
            "--checkpoint-manifest-sha256 does not match the pinned two-step adapter"
        )
    if not 0.1 <= args.mps_memory_fraction <= 0.85:
        parser.error("--mps-memory-fraction must be in [0.1, 0.85]")
    return args


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        report = run_comparison(args)
    except (OSError, RuntimeError, ValueError, MemoryError) as exc:
        print(f"qwen_image21_lora_visual_compare: error: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(report, indent=2, sort_keys=True))
    print(f"saved six decoded RGBA images and manifest to {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
