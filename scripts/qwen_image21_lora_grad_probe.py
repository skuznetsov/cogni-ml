#!/usr/bin/env python3
"""Run a no-optimizer-step, frozen-base Qwen-Image 2.1 LoRA gradient probe.

The probe compares one coarse student Euler endpoint against a two-substep
endpoint from the same frozen base transformer. It is a gradient-plumbing and
numerical-stability check only; it does not train or evaluate image quality.
Text/VAE are intentionally not loaded: their pinned conditioning exchange is
precomputed in the validated bundle.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import os
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence


MODEL_REVISION = "790c92633540aa0cb11d9abf19eb46d861714758"
DIFFUSERS_COMMIT = "8b3c707ebd3ec4881f4190cf42931da07eaf3b65"
TARGET_MODULE = "transformer_blocks.31.attn.to_q"
ADAPTER_NAME = "grad_probe"
RANK = 4
ADAPTER_INIT_SEED = 20260925
MIN_PEFT_VERSION = "0.18.0"
FIXTURE_SHA256 = "62ff646b0a98b743f56466c5d68afcc93fa42b8172a12cad6545f6dec5750581"
CONDITIONING_MANIFEST_SHA256 = "daa5fdf4c2616730f025e7365756cb5dfd6d7c8bfac76ba27c806bb7fe004fa6"
CONDITIONING_PAYLOAD_SHA256 = "56bfb2e98e22d193242a66a1bf2ea905482aecbc6e1ee6ad877775d66238bade"
CONFIG_SHA256 = "56ae3281c4e6c2d1aa3658252d187488071815fd79bef15808bb0205fc1a2241"
INDEX_SHA256 = "17987f6623b1c814d0ef55a137d99142b7b3b040eb1bf241b5575dd35af803a2"
SHARD_SHA256 = {
    "diffusion_pytorch_model-00001-of-00002.safetensors": "9e6bc2d641e67bf277895ea8777141044a38f3edb7101bc469b2961dd7c36b4b",
    "diffusion_pytorch_model-00002-of-00002.safetensors": "3aaf234dcbe128530479735854a346b5e3e66283b7c11db56f836bbd1c13ebaa",
}
GIB = 1024**3
EXPECTED_TENSOR_SHAPES = {
    "encoder_hidden_states": (10, 4096),
    "encoder_hidden_states_mask": (10,),
    "encoder_img_mask": (10,),
    "initial_target_latents": (256, 64),
}
EXPECTED_TENSOR_DTYPES = {
    "encoder_hidden_states": "float32-le",
    "encoder_hidden_states_mask": "uint8",
    "encoder_img_mask": "uint8",
    "initial_target_latents": "float32-le",
}


@dataclass(frozen=True)
class FlowMatchSchedule:
    sigmas: tuple[float, ...]
    timesteps: tuple[float, ...]
    teacher_step_indices: tuple[int, ...] = (0, 1)
    student_step_indices: tuple[int, ...] = (0,)
    fixture_sha256: str = FIXTURE_SHA256


@dataclass(frozen=True)
class ConditioningBundle:
    manifest_path: Path
    payload_sha256: str
    prompt: str
    img_shapes: list[list[tuple[int, int, int]]]
    encoder_hidden_states: Any
    encoder_hidden_states_mask: Any
    encoder_img_mask: Any
    initial_target_latents: Any


@dataclass(frozen=True)
class ProbeResult:
    target_module: str
    rank: int
    loss: float
    adapter_gradient_norm: float
    adapter_gradients_finite: bool
    teacher_endpoint_rms: float
    student_endpoint_rms: float
    trainable_parameter_names: tuple[str, ...]
    base_parameters_have_no_grad: bool
    optimizer_step_performed: bool = False


def sha256_file(path: str | Path, chunk_bytes: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while True:
            chunk = handle.read(chunk_bytes)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def require_sha256(path: str | Path, expected: str, label: str) -> str:
    actual = sha256_file(path)
    if actual != expected:
        raise ValueError(f"{label} SHA256 mismatch: expected {expected}, got {actual}")
    return actual


def load_flow_match_schedule(path: str | Path) -> FlowMatchSchedule:
    path = Path(path)
    require_sha256(path, FIXTURE_SHA256, "flow-match fixture")
    try:
        fixture = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read pinned flow-match fixture: {path}") from exc
    provenance = fixture.get("provenance", {})
    if (
        provenance.get("model_repo") != "Qwen/Qwen-Image-2.1"
        or provenance.get("model_revision") != MODEL_REVISION
        or provenance.get("diffusers_commit") != DIFFUSERS_COMMIT
    ):
        raise ValueError("flow-match fixture provenance is not the pinned Qwen-Image 2.1 source")
    schedule = next(
        (
            entry
            for entry in fixture.get("schedules", [])
            if entry.get("steps") == 4 and entry.get("image_seq_len") == 256
        ),
        None,
    )
    if schedule is None or schedule.get("mu") != 0.5:
        raise ValueError("fixture must contain the 4-step, 256-token, mu=0.5 schedule")
    sigmas = tuple(float(value) for value in schedule.get("sigmas", ()))
    timesteps = tuple(float(value) for value in schedule.get("timesteps", ()))
    if len(sigmas) != 5 or len(timesteps) != 4:
        raise ValueError("pinned 4-step schedule must have five sigma nodes and four timesteps")
    expected_sigmas = (1.0, 0.744611382484436, 0.4266734719276428, 0.019999980926513672, 0.0)
    expected_times = (1000.0, 744.6113891601562, 426.6734619140625, 19.999980926513672)
    if sigmas != expected_sigmas or timesteps != expected_times:
        raise ValueError("pinned nested sigma/timestep values changed")
    return FlowMatchSchedule(sigmas=sigmas, timesteps=timesteps)


def euler_endpoint(initial: Any, velocities: Sequence[Any], sigmas: Sequence[float]):
    """Apply explicit Euler over the supplied adjacent sigma nodes."""
    if len(sigmas) != len(velocities) + 1:
        raise ValueError("Euler endpoint needs one more sigma node than velocities")
    result = initial
    for index, velocity in enumerate(velocities):
        result = result + (float(sigmas[index + 1]) - float(sigmas[index])) * velocity
    return result


def load_conditioning_bundle(
    manifest_path: str | Path, *, verify_pinned_bundle: bool = True
) -> ConditioningBundle:
    import numpy as np
    import torch

    manifest_path = Path(manifest_path)
    if verify_pinned_bundle:
        require_sha256(manifest_path, CONDITIONING_MANIFEST_SHA256, "conditioning manifest")
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read conditioning manifest: {manifest_path}") from exc
    if manifest.get("schema") != "qwen-image21-conditioning" or manifest.get("schema_version") != 1:
        raise ValueError("unsupported Qwen-Image 2.1 conditioning bundle schema")
    model = manifest.get("model", {})
    if model.get("repo") != "Qwen/Qwen-Image-2.1" or model.get("revision") != MODEL_REVISION:
        raise ValueError("conditioning bundle is not from the pinned Qwen-Image 2.1 revision")
    runtime = manifest.get("runtime", {})
    if runtime.get("diffusers_commit") != DIFFUSERS_COMMIT:
        raise ValueError("conditioning bundle Diffusers source commit is not pinned")
    image = manifest.get("image", {})
    if (
        image.get("width") != 256
        or image.get("height") != 256
        or image.get("latent_height") != 16
        or image.get("latent_width") != 16
        or image.get("img_shapes") != [[1, 16, 16]]
    ):
        raise ValueError("conditioning bundle must describe a 256x256, 16x16 target")

    payload_name = manifest.get("payload_file")
    if not isinstance(payload_name, str) or Path(payload_name).name != payload_name:
        raise ValueError("conditioning payload_file must be a local filename")
    payload_path = manifest_path.parent / payload_name
    try:
        payload = payload_path.read_bytes()
    except OSError as exc:
        raise ValueError(f"cannot read conditioning payload: {payload_path}") from exc
    if len(payload) != manifest.get("payload_nbytes"):
        raise ValueError("conditioning payload size does not match manifest")
    payload_hash = hashlib.sha256(payload).hexdigest()
    if payload_hash != manifest.get("payload_sha256"):
        raise ValueError("conditioning payload SHA256 does not match manifest")
    if verify_pinned_bundle and payload_hash != CONDITIONING_PAYLOAD_SHA256:
        raise ValueError("conditioning payload does not match the pinned bundle SHA256")

    descriptors = manifest.get("tensors", {})
    if set(descriptors) != set(EXPECTED_TENSOR_SHAPES):
        raise ValueError("conditioning manifest tensor set is incomplete or unexpected")
    arrays = {}
    expected_offset = 0
    for name, shape in EXPECTED_TENSOR_SHAPES.items():
        descriptor = descriptors[name]
        dtype_name = EXPECTED_TENSOR_DTYPES[name]
        expected_bytes = math.prod(shape) * (4 if dtype_name == "float32-le" else 1)
        offset = descriptor.get("offset_bytes")
        nbytes = descriptor.get("nbytes")
        if (
            descriptor.get("shape") != list(shape)
            or descriptor.get("dtype") != dtype_name
            or offset != expected_offset
            or nbytes != expected_bytes
        ):
            raise ValueError(f"conditioning tensor descriptor is invalid: {name}")
        if expected_offset + expected_bytes > len(payload):
            raise ValueError(f"conditioning tensor exceeds payload: {name}")
        np_dtype = "<f4" if dtype_name == "float32-le" else "u1"
        array = np.frombuffer(payload, dtype=np_dtype, count=math.prod(shape), offset=offset)
        arrays[name] = array.reshape(shape).copy()
        expected_offset += expected_bytes
    if expected_offset != len(payload):
        raise ValueError("conditioning payload has unreferenced trailing bytes")
    for name in ("encoder_hidden_states", "initial_target_latents"):
        if not bool(np.isfinite(arrays[name]).all()):
            raise ValueError(f"conditioning tensor contains non-finite values: {name}")
    for name in ("encoder_hidden_states_mask", "encoder_img_mask"):
        if not bool(np.logical_or(arrays[name] == 0, arrays[name] == 1).all()):
            raise ValueError(f"conditioning mask must contain only 0 or 1: {name}")
    if not bool(arrays["encoder_hidden_states_mask"].any()):
        raise ValueError("conditioning text mask must contain at least one valid token")
    if bool(arrays["encoder_img_mask"].any()):
        raise ValueError("red-cube text-only bundle must not contain condition-image slots")

    tensors = {
        name: torch.from_numpy(array).unsqueeze(0)
        for name, array in arrays.items()
    }
    img_shapes = [[tuple(int(axis) for axis in shape) for shape in image["img_shapes"]]]
    return ConditioningBundle(
        manifest_path=manifest_path,
        payload_sha256=payload_hash,
        prompt=str(manifest.get("prompt", "")),
        img_shapes=img_shapes,
        encoder_hidden_states=tensors["encoder_hidden_states"],
        encoder_hidden_states_mask=tensors["encoder_hidden_states_mask"].bool(),
        encoder_img_mask=tensors["encoder_img_mask"].bool(),
        initial_target_latents=tensors["initial_target_latents"],
    )


def build_forward_kwargs(
    bundle: ConditioningBundle,
    hidden_states: Any,
    *,
    scheduler_timestep: float,
    device: Any,
    model_dtype: Any,
) -> dict[str, Any]:
    """Mirror the pinned pipeline's dtype cast, timestep scaling, and target slots."""
    import torch

    device = torch.device(device)
    image_mask = bundle.encoder_img_mask.to(device=device, dtype=torch.bool)
    target_slots = torch.ones(
        (image_mask.shape[0], hidden_states.shape[1] // 4),
        dtype=torch.bool,
        device=device,
    )
    image_mask = torch.cat((image_mask, target_slots), dim=1)
    raw_timestep = torch.tensor(
        [float(scheduler_timestep)], device=device, dtype=model_dtype
    )
    return {
        "hidden_states": hidden_states.to(device=device, dtype=model_dtype),
        "encoder_hidden_states": bundle.encoder_hidden_states.to(device=device, dtype=model_dtype),
        "encoder_hidden_states_mask": bundle.encoder_hidden_states_mask.to(device=device, dtype=torch.bool),
        "img_shapes": bundle.img_shapes,
        "img_mask": image_mask,
        # Pipeline casts the scheduler value to latent dtype first, then divides.
        "timestep": raw_timestep / 1000.0,
        "attention_kwargs": {},
        "return_dict": False,
    }


def _model_sample(result: Any):
    if hasattr(result, "sample"):
        return result.sample
    if isinstance(result, (tuple, list)):
        if not result:
            raise ValueError("transformer returned an empty tuple")
        return result[0]
    return result


def _is_finite(value: Any) -> bool:
    import torch

    return bool(torch.isfinite(value).all().item())


def _forward_velocity(
    model: Any,
    bundle: ConditioningBundle,
    state: Any,
    scheduler_timestep: float,
    device: Any,
    model_dtype: Any,
):
    import torch

    kwargs = build_forward_kwargs(
        bundle,
        state,
        scheduler_timestep=scheduler_timestep,
        device=device,
        model_dtype=model_dtype,
    )
    velocity = _model_sample(model(**kwargs))
    target_tokens = state.shape[1]
    if velocity.ndim != 3 or velocity.shape[0] != 1 or velocity.shape[-1] != 64 or velocity.shape[1] < target_tokens:
        raise ValueError(f"Qwen transformer output must end in [1, {target_tokens}, 64]")
    velocity = velocity[:, -target_tokens:, :]
    if not _is_finite(velocity):
        raise ValueError("transformer produced a non-finite velocity")
    return velocity


def _install_probe_adapter(model: Any, *, adapter_dtype: Any, device: Any) -> tuple[str, ...]:
    import torch

    require_compatible_peft_installation()
    from peft import LoraConfig, inject_adapter_in_model

    config = LoraConfig(
        r=RANK,
        lora_alpha=RANK,
        target_modules=[TARGET_MODULE],
        lora_dropout=0.0,
        bias="none",
        init_lora_weights=True,
    )
    device = torch.device(device)
    cpu_rng_state = torch.random.get_rng_state()
    mps_rng_state = torch.mps.get_rng_state() if device.type == "mps" else None
    try:
        cpu_generator = torch.Generator(device="cpu").manual_seed(ADAPTER_INIT_SEED)
        torch.random.set_rng_state(cpu_generator.get_state())
        if mps_rng_state is not None:
            torch.mps.manual_seed(ADAPTER_INIT_SEED)
        if callable(getattr(model, "add_adapter", None)):
            model.add_adapter(config, adapter_name=ADAPTER_NAME)
            if callable(getattr(model, "set_adapter", None)):
                model.set_adapter(ADAPTER_NAME)
        else:
            inject_adapter_in_model(config, model, adapter_name=ADAPTER_NAME)
    finally:
        torch.random.set_rng_state(cpu_rng_state)
        if mps_rng_state is not None:
            torch.mps.set_rng_state(mps_rng_state)

    trainable_names = []
    for name, parameter in model.named_parameters():
        is_adapter = "lora_" in name and ADAPTER_NAME in name
        parameter.requires_grad_(is_adapter)
        parameter.grad = None
        if is_adapter:
            parameter.data = parameter.data.to(device=device, dtype=adapter_dtype)
            trainable_names.append(name)
    if len(trainable_names) != 2:
        raise ValueError(f"expected exactly LoRA A and B trainables, found {trainable_names}")
    named_modules = dict(model.named_modules())
    wrapped = named_modules.get(TARGET_MODULE)
    if wrapped is None or not hasattr(wrapped, "lora_A") or ADAPTER_NAME not in wrapped.lora_A:
        raise ValueError(f"PEFT did not wrap the exact target projection {TARGET_MODULE}")
    lora_a = wrapped.lora_A[ADAPTER_NAME].weight
    lora_b = wrapped.lora_B[ADAPTER_NAME].weight
    if lora_a.shape[0] != RANK or lora_b.shape[1] != RANK:
        raise ValueError(f"LoRA projection rank is not {RANK}: A={tuple(lora_a.shape)}, B={tuple(lora_b.shape)}")
    if any(not parameter.requires_grad for name, parameter in model.named_parameters() if "lora_" in name and ADAPTER_NAME in name):
        raise ValueError("not all selected LoRA parameters are trainable")
    if any(parameter.requires_grad for name, parameter in model.named_parameters() if name not in trainable_names):
        raise ValueError("a frozen base parameter became trainable")
    return tuple(trainable_names)


def run_gradient_probe(
    transformer: Any,
    bundle: ConditioningBundle,
    schedule: FlowMatchSchedule,
    *,
    device: Any,
    adapter_dtype: Any,
) -> ProbeResult:
    """Backpropagate once through rank-4 LoRA and deliberately do not step."""
    import torch
    import torch.nn.functional as F

    device = torch.device(device)
    model_dtype = next(transformer.parameters()).dtype
    for parameter in transformer.parameters():
        parameter.requires_grad_(False)
        parameter.grad = None
    transformer.eval()
    initial = bundle.initial_target_latents.detach().to(device=device, dtype=torch.float32)

    # Frozen same-base teacher: integrate the nested fixture nodes 0->1->2.
    teacher_state = initial.clone()
    with torch.no_grad():
        for index in schedule.teacher_step_indices:
            teacher_velocity = _forward_velocity(
                transformer,
                bundle,
                teacher_state,
                schedule.timesteps[index],
                device,
                model_dtype,
            ).float()
            teacher_state = euler_endpoint(
                teacher_state,
                (teacher_velocity,),
                (schedule.sigmas[index], schedule.sigmas[index + 1]),
            )
            if not _is_finite(teacher_state):
                raise ValueError("teacher endpoint became non-finite")
    teacher_endpoint = teacher_state.detach()

    trainable_names = _install_probe_adapter(
        transformer, adapter_dtype=adapter_dtype, device=device
    )
    transformer.eval()  # LoRA dropout is also kept inactive for a deterministic probe.
    with torch.enable_grad():
        student_velocity = _forward_velocity(
            transformer,
            bundle,
            initial,
            schedule.timesteps[schedule.student_step_indices[0]],
            device,
            model_dtype,
        ).float()
        student_endpoint = euler_endpoint(
            initial,
            (student_velocity,),
            (schedule.sigmas[0], schedule.sigmas[2]),
        )
        if not _is_finite(student_endpoint):
            raise ValueError("student endpoint became non-finite")
        loss = F.mse_loss(student_endpoint, teacher_endpoint, reduction="mean")
        if not _is_finite(loss):
            raise ValueError("endpoint loss is non-finite")
        loss.backward()

    adapter_parameters = [parameter for name, parameter in transformer.named_parameters() if name in trainable_names]
    gradient_squares = 0.0
    for parameter in adapter_parameters:
        if parameter.grad is None:
            raise ValueError("a selected LoRA parameter has no gradient")
        if not _is_finite(parameter.grad):
            raise ValueError("a selected LoRA gradient is non-finite")
        gradient_squares += float(parameter.grad.detach().float().pow(2).sum().item())
    gradient_norm = math.sqrt(gradient_squares)
    if not math.isfinite(gradient_norm) or gradient_norm == 0.0:
        raise ValueError("selected LoRA gradient norm is zero or non-finite")
    base_has_grad = any(
        parameter.grad is not None
        for name, parameter in transformer.named_parameters()
        if name not in trainable_names
    )
    if base_has_grad:
        raise ValueError("frozen transformer base unexpectedly received a gradient")
    return ProbeResult(
        target_module=TARGET_MODULE,
        rank=RANK,
        loss=float(loss.detach().item()),
        adapter_gradient_norm=gradient_norm,
        adapter_gradients_finite=True,
        teacher_endpoint_rms=float(teacher_endpoint.square().mean().sqrt().item()),
        student_endpoint_rms=float(student_endpoint.detach().square().mean().sqrt().item()),
        trainable_parameter_names=trainable_names,
        base_parameters_have_no_grad=not base_has_grad,
        optimizer_step_performed=False,
    )


def verify_snapshot_integrity(model_dir: str | Path) -> tuple[dict[str, Any], int]:
    """Fail closed on any unpinned config, index, shard, or HF metadata revision."""
    model_dir = Path(model_dir).resolve()
    config_path = model_dir / "config.json"
    index_path = model_dir / "diffusion_pytorch_model.safetensors.index.json"
    require_sha256(config_path, CONFIG_SHA256, "transformer config")
    require_sha256(index_path, INDEX_SHA256, "transformer safetensors index")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    expected_config = {
        "_class_name": "QwenImage21Transformer2DModel",
        "in_channels": 64,
        "out_channels": 64,
        "num_layers": 32,
        "attention_head_dim": 128,
        "num_attention_heads": 32,
        "context_in_dim": 4096,
        "axes_dims_rope": [16, 56, 56],
        "causal_condition": True,
    }
    if any(config.get(key) != value for key, value in expected_config.items()):
        raise ValueError("transformer config dimensions or class differ from the pinned source")
    index = json.loads(index_path.read_text(encoding="utf-8"))
    weight_map = index.get("weight_map", {})
    if len(weight_map) != 297:
        raise ValueError(f"expected 297 indexed transformer tensors, found {len(weight_map)}")
    if weight_map.get(f"{TARGET_MODULE}.weight") != "diffusion_pytorch_model-00002-of-00002.safetensors":
        raise ValueError("pinned index does not map the selected LoRA projection to shard 2")
    shard_names = sorted(set(weight_map.values()))
    if shard_names != sorted(SHARD_SHA256):
        raise ValueError(f"unexpected safetensors shard set: {shard_names}")

    metadata_dir = model_dir.parent / ".cache" / "huggingface" / "download" / "transformer"
    files = {"config.json": CONFIG_SHA256, index_path.name: INDEX_SHA256, **SHARD_SHA256}
    total_size = 0
    for filename, expected_hash in files.items():
        path = model_dir / filename
        actual_hash = require_sha256(path, expected_hash, filename)
        if filename.endswith(".safetensors"):
            total_size += path.stat().st_size
        metadata_path = metadata_dir / f"{filename}.metadata"
        try:
            metadata_lines = metadata_path.read_text(encoding="utf-8").splitlines()
        except OSError as exc:
            raise ValueError(f"missing Hugging Face revision metadata for {filename}") from exc
        if len(metadata_lines) < 2 or metadata_lines[0].lower() != MODEL_REVISION:
            raise ValueError(f"{filename} metadata is not from pinned revision {MODEL_REVISION}")
        if not re.fullmatch(r"[0-9a-fA-F]+", actual_hash):
            raise ValueError(f"invalid SHA256 while validating {filename}")

    try:
        from safetensors import safe_open

        with safe_open(model_dir / "diffusion_pytorch_model-00002-of-00002.safetensors", framework="pt", device="cpu") as file:
            shape = file.get_slice(f"{TARGET_MODULE}.weight").get_shape()
    except Exception as exc:
        raise ValueError(f"cannot inspect selected projection in safetensors header: {exc}") from exc
    if list(shape) != [4096, 4096]:
        raise ValueError(f"selected projection must be [4096, 4096], got {shape}")
    if total_size <= 0:
        raise ValueError("transformer checkpoint shards are empty")
    return config, total_size


def validate_diffusers_direct_url(direct_url_text: str | None) -> str:
    if not direct_url_text:
        raise RuntimeError("installed Diffusers direct_url.json is missing; cannot attest pinned source commit")
    try:
        direct_url = json.loads(direct_url_text)
    except json.JSONDecodeError as exc:
        raise RuntimeError("installed Diffusers direct_url.json is invalid") from exc
    vcs_info = direct_url.get("vcs_info", {})
    commit = vcs_info.get("commit_id")
    if vcs_info.get("vcs") != "git" or commit != DIFFUSERS_COMMIT:
        raise RuntimeError(
            f"installed Diffusers must be git commit {DIFFUSERS_COMMIT}; observed {commit!r}"
        )
    return commit


def require_pinned_diffusers_installation() -> str:
    try:
        direct_url_text = importlib.metadata.distribution("diffusers").read_text("direct_url.json")
    except importlib.metadata.PackageNotFoundError as exc:
        raise RuntimeError("pinned Diffusers installation is unavailable") from exc
    return validate_diffusers_direct_url(direct_url_text)


def validate_peft_version(version: str) -> str:
    from packaging.version import InvalidVersion, Version

    try:
        installed = Version(version)
    except InvalidVersion as exc:
        raise RuntimeError(f"installed PEFT version is invalid: {version!r}") from exc
    if installed < Version(MIN_PEFT_VERSION):
        raise RuntimeError(
            f"PEFT >= {MIN_PEFT_VERSION} is required for the pinned Diffusers adapter path "
            f"with Transformers 5; observed {version}"
        )
    return version


def require_compatible_peft_installation() -> tuple[str, str]:
    try:
        distribution = importlib.metadata.distribution("peft")
    except importlib.metadata.PackageNotFoundError as exc:
        raise RuntimeError("compatible PEFT installation is unavailable") from exc

    version = validate_peft_version(distribution.version)
    try:
        import peft
    except ImportError as exc:
        raise RuntimeError("installed PEFT package cannot be imported") from exc
    distribution_path = Path(distribution.locate_file("peft")).resolve()
    imported_path = Path(peft.__file__).resolve().parent
    if imported_path != distribution_path:
        raise RuntimeError(
            "imported PEFT package path does not match its installed distribution: "
            f"{imported_path} != {distribution_path}"
        )
    return version, str(imported_path)


def _parse_darwin_memory(
    pressure_output: str, vm_stat_output: str
) -> tuple[int, int, int]:
    """Return (pressure estimate, vm_stat reclaimable floor, physical RAM)."""
    total_match = re.search(
        r"The system has (\d+) \((\d+) pages with a page size of (\d+)\)",
        pressure_output,
    )
    percentage_match = re.search(
        r"System-wide memory free percentage:\s*(\d+)%", pressure_output
    )
    page_match = re.search(r"page size of (\d+) bytes", vm_stat_output)
    if total_match is None or percentage_match is None or page_match is None:
        raise RuntimeError("cannot parse macOS memory-pressure/vm_stat output")
    total, physical_pages, pressure_page_size = map(int, total_match.groups())
    page_size = int(page_match.group(1))
    if physical_pages * pressure_page_size != total or page_size != pressure_page_size:
        raise RuntimeError("macOS memory-pressure and vm_stat physical-memory reports disagree")
    if total <= 0 or not 0 <= int(percentage_match.group(1)) <= 100:
        raise RuntimeError("macOS reported invalid physical memory or pressure percentage")

    labels = ("Pages free", "Pages inactive", "Pages speculative", "Pages purgeable")
    pages = {}
    for label in labels:
        match = re.search(rf"^{re.escape(label)}:\s+(\d+)", vm_stat_output, re.MULTILINE)
        if match is None:
            raise RuntimeError(f"cannot read {label} from vm_stat")
        pages[label] = int(match.group(1))
    # memory_pressure rounds to integer percentage; discount one point so that
    # the estimate is conservative. vm_stat is retained as a separate floor,
    # not combined with the more optimistic pressure estimate.
    pressure_percent = max(0, int(percentage_match.group(1)) - 1)
    pressure_available = total * pressure_percent // 100
    raw_reclaimable = sum(pages.values()) * page_size
    return pressure_available, raw_reclaimable, total


def _system_memory_bytes() -> tuple[int, int, int]:
    """Return pressure-estimated available, independent reclaimable, total RAM."""
    if sys.platform == "darwin":
        pressure_output = subprocess.run(
            ["/usr/bin/memory_pressure", "-Q"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout
        vm_stat_output = subprocess.run(
            ["/usr/bin/vm_stat"], check=True, capture_output=True, text=True
        ).stdout
        return _parse_darwin_memory(pressure_output, vm_stat_output)
    page_size = os.sysconf("SC_PAGE_SIZE")
    total = os.sysconf("SC_PHYS_PAGES") * page_size
    try:
        available_pages = os.sysconf("SC_AVPHYS_PAGES")
    except (AttributeError, ValueError):
        raise RuntimeError("cannot query available host memory on this platform")
    available = int(available_pages * page_size)
    return available, available, int(total)


def guard_memory_budget(
    checkpoint_bytes: int,
    *,
    device_type: str,
    device_limit_bytes: int | None,
    current_device_bytes: int,
    pressure_estimated_available_host_bytes: int,
    raw_reclaimable_host_bytes: int,
    total_host_bytes: int,
    memory_fraction: float = 0.70,
) -> dict[str, int | float | str]:
    if not (0.1 <= memory_fraction <= 0.85):
        raise ValueError("memory fraction must be in [0.1, 0.85]")
    # A full model/device copy can coexist briefly with CPU checkpoint storage;
    # reserve an additional 2 GiB for the last-block graph and runtime workspaces.
    estimated_peak = 2 * checkpoint_bytes + 2 * GIB
    if device_type == "mps":
        if device_limit_bytes is None:
            raise ValueError("MPS recommended memory limit is unavailable")
        device_budget = int(device_limit_bytes * memory_fraction)
        if current_device_bytes + estimated_peak > device_budget:
            raise MemoryError(
                f"MPS guard: estimated {estimated_peak / GIB:.2f} GiB + current "
                f"{current_device_bytes / GIB:.2f} GiB exceeds {memory_fraction:.0%} "
                f"budget {device_budget / GIB:.2f} GiB"
            )
    elif device_type == "cuda":
        if device_limit_bytes is None:
            raise ValueError("CUDA free-memory limit is unavailable")
        device_budget = int(device_limit_bytes * memory_fraction)
        if current_device_bytes + estimated_peak > device_budget:
            raise MemoryError("CUDA memory guard rejects the estimated peak allocation")
    elif device_type == "cpu":
        if estimated_peak > pressure_estimated_available_host_bytes:
            raise MemoryError(
                f"CPU guard: projected peak estimate {estimated_peak / GIB:.2f} GiB exceeds "
                f"pressure-estimated RAM {pressure_estimated_available_host_bytes / GIB:.2f} GiB"
            )
    else:
        raise ValueError(f"unsupported execution device type: {device_type}")
    reserve_bytes = int(total_host_bytes * 0.20)
    if pressure_estimated_available_host_bytes - estimated_peak < reserve_bytes:
        raise MemoryError(
            "host-memory guard requires pressure-estimated RAM minus projected peak "
            "to retain at least 20% of physical RAM"
        )
    if raw_reclaimable_host_bytes < reserve_bytes:
        raise MemoryError(
            "host-memory guard requires the independent raw reclaimable estimate "
            "to retain at least 20% of physical RAM"
        )
    return {
        "estimated_peak_bytes": estimated_peak,
        "estimated_peak_not_guaranteed": True,
        "pressure_estimated_available_host_bytes": pressure_estimated_available_host_bytes,
        "raw_reclaimable_host_bytes": raw_reclaimable_host_bytes,
        "total_host_bytes": total_host_bytes,
        "device_budget_bytes": int(device_limit_bytes * memory_fraction) if device_limit_bytes else 0,
        "memory_fraction": memory_fraction,
        "device_type": device_type,
    }


def _resolve_device(torch: Any, requested: str):
    if requested == "auto":
        if torch.backends.mps.is_available():
            return torch.device("mps")
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")
    device = torch.device(requested)
    if device.type == "mps" and not torch.backends.mps.is_available():
        raise ValueError("MPS was requested but is unavailable")
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA was requested but is unavailable")
    if device.type not in {"cpu", "mps", "cuda"}:
        raise ValueError("device must be auto, cpu, mps, or cuda")
    return device


def _guard_local_memory(torch: Any, checkpoint_bytes: int, device: Any, memory_fraction: float):
    pressure_available, raw_reclaimable, total = _system_memory_bytes()
    limit = current = None
    if device.type == "mps":
        limit = int(torch.mps.recommended_max_memory())
        current = int(torch.mps.driver_allocated_memory())
    elif device.type == "cuda":
        free_bytes, limit_bytes = torch.cuda.mem_get_info(device)
        limit = int(limit_bytes)
        current = int(limit_bytes - free_bytes)
    return guard_memory_budget(
        checkpoint_bytes,
        device_type=device.type,
        device_limit_bytes=limit,
        current_device_bytes=current or 0,
        pressure_estimated_available_host_bytes=pressure_available,
        raw_reclaimable_host_bytes=raw_reclaimable,
        total_host_bytes=total,
        memory_fraction=memory_fraction,
    )


def _set_device_memory_cap(torch: Any, device: Any, memory_fraction: float) -> None:
    if device.type == "mps":
        setter = getattr(torch.mps, "set_per_process_memory_fraction", None)
        if not callable(setter):
            raise RuntimeError("this PyTorch build cannot enforce the MPS per-process memory cap")
        setter(float(memory_fraction))


def _load_transformer(model_dir: Path, device: Any, dtype: Any):
    from diffusers import QwenImage21Transformer2DModel

    model = QwenImage21Transformer2DModel.from_pretrained(
        model_dir,
        dtype=dtype,
        low_cpu_mem_usage=True,
        local_files_only=True,
        use_safetensors=True,
    )
    model.to(device=device)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
        parameter.grad = None
    return model


def _parse_dtype(torch: Any, name: str):
    return {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}[name]


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", required=True, type=Path, help="local pinned transformer/ directory")
    parser.add_argument("--conditioning", required=True, type=Path, help="qwen_image21_conditioning.json")
    parser.add_argument(
        "--schedule",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "spec/fixtures/qwen_image21_flow_match_diffusers.json",
    )
    parser.add_argument("--device", choices=("mps",), default="mps")
    parser.add_argument("--dtype", choices=("bfloat16",), default="bfloat16")
    parser.add_argument("--mps-memory-fraction", type=float, default=0.70)
    args = parser.parse_args(argv)

    import torch

    schedule = load_flow_match_schedule(args.schedule)
    bundle = load_conditioning_bundle(args.conditioning)
    installed_diffusers_commit = require_pinned_diffusers_installation()
    installed_peft_version, installed_peft_path = require_compatible_peft_installation()
    model_config, checkpoint_bytes = verify_snapshot_integrity(args.model_dir)
    device = _resolve_device(torch, args.device)
    dtype = _parse_dtype(torch, args.dtype)
    memory = _guard_local_memory(torch, checkpoint_bytes, device, args.mps_memory_fraction)
    _set_device_memory_cap(torch, device, args.mps_memory_fraction)
    transformer = _load_transformer(args.model_dir, device, dtype)
    result = run_gradient_probe(
        transformer,
        bundle,
        schedule,
        device=device,
        adapter_dtype=torch.float32,
    )
    report = {
        "status": "gradient_probe_passed",
        "claim_scope": "one gradient-plumbing/stability probe; no optimizer step; no quality evidence",
        "model_repo": "Qwen/Qwen-Image-2.1",
        "model_revision": MODEL_REVISION,
        "diffusers_commit": DIFFUSERS_COMMIT,
        "installed_diffusers_commit": installed_diffusers_commit,
        "installed_peft_version": installed_peft_version,
        "installed_peft_path": installed_peft_path,
        "transformer_class": model_config["_class_name"],
        "device": str(device),
        "model_dtype": args.dtype,
        "adapter_dtype": "float32",
        "adapter_init_seed": ADAPTER_INIT_SEED,
        "conditioning_payload_sha256": bundle.payload_sha256,
        "flow_fixture_sha256": schedule.fixture_sha256,
        "checkpoint_bytes": checkpoint_bytes,
        "memory_guard": memory,
        "target_module": result.target_module,
        "rank": result.rank,
        "trainable_parameter_names": result.trainable_parameter_names,
        "loss": result.loss,
        "adapter_gradient_norm": result.adapter_gradient_norm,
        "adapter_gradients_finite": result.adapter_gradients_finite,
        "base_parameters_have_no_grad": result.base_parameters_have_no_grad,
        "teacher_endpoint_rms": result.teacher_endpoint_rms,
        "student_endpoint_rms": result.student_endpoint_rms,
        "optimizer_step_performed": result.optimizer_step_performed,
        "teacher_semantics": "frozen same-base two Euler substeps over fixture nodes sigma[0:3]",
        "student_semantics": "one trainable coarse Euler update sigma[0] to sigma[2]",
    }
    if device.type == "mps":
        torch.mps.synchronize()
        report["mps_memory_after_probe"] = {
            "driver_allocated_bytes": int(torch.mps.driver_allocated_memory()),
            "current_allocated_bytes": int(torch.mps.current_allocated_memory()),
            "recommended_max_bytes": int(torch.mps.recommended_max_memory()),
        }
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
