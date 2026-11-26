#!/usr/bin/env python3
"""Run bounded rank-4 LoRA endpoint distillation for Qwen-Image 2.1.

The frozen same-base teacher integrates two adjacent fixture substeps once.
The student learns one coarse Euler endpoint with an adapter on the final
attention Q projection. An opt-in, content-bound adapter checkpoint can be
written after held-out evaluation; this makes no image-quality claim.
"""

from __future__ import annotations

import argparse
import json
import math
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Sequence

import qwen_image21_lora_grad_probe as probe
import qwen_image21_lora_adapter_checkpoint as adapter_checkpoint


MAX_STEPS = 8
DEFAULT_STEPS = 2
DEFAULT_LEARNING_RATE = 1e-3
MAX_LEARNING_RATE = 1e-2
MIN_LEARNING_RATE = 1e-8
DEFAULT_MAX_GRAD_NORM = 1.0
MAX_GRAD_NORM = 10.0
TARGET_TOKENS = 256
TARGET_LATENT_CHANNELS = 64
TARGET_IMG_SHAPES = [[(1, 16, 16)]]
PINNED_SIGMAS = (
    1.0,
    0.744611382484436,
    0.4266734719276428,
    0.019999980926513672,
    0.0,
)
PINNED_TIMESTEPS = (1000.0, 744.6113891601562, 426.6734619140625, 19.999980926513672)


@dataclass(frozen=True)
class DistillationStep:
    step: int
    pre_step_loss: float
    post_step_loss: float
    post_step_loss_finite: bool
    preclip_gradient_norm: float
    postclip_gradient_norm: float
    adapter_weights_finite: bool
    memory_after: dict[str, int | float | str]


@dataclass(frozen=True)
class HeldoutEvaluation:
    conditioning_manifest_sha256: str
    conditioning_payload_sha256: str
    initial_eval_loss: float
    final_eval_loss: float
    teacher_target_detached: bool
    teacher_endpoint_rms: float

    @property
    def no_grad_context_delta(self) -> float:
        """Return final minus initial held-out loss from the shared no-grad evaluator."""
        return self.final_eval_loss - self.initial_eval_loss


@dataclass(frozen=True)
class DistillationResult:
    target_module: str
    rank: int
    trainable_parameter_names: tuple[str, ...]
    steps: tuple[DistillationStep, ...]
    initial_eval_loss: float
    initial_adapter_weights: dict[str, Any]
    teacher_target_detached: bool
    base_parameters_have_no_grad: bool
    teacher_endpoint_rms: float
    heldout_evaluation: HeldoutEvaluation | None = None

    @property
    def final_eval_loss(self) -> float:
        return self.steps[-1].post_step_loss

    @property
    def no_grad_context_delta(self) -> float:
        """Return final minus initial loss, both measured by the shared no-grad evaluator."""
        return self.final_eval_loss - self.initial_eval_loss


def validate_training_parameters(
    *, steps: int, learning_rate: float, max_grad_norm: float
) -> None:
    if isinstance(steps, bool) or not isinstance(steps, int) or not 1 <= steps <= MAX_STEPS:
        raise ValueError(f"steps must be an integer in [1, {MAX_STEPS}]")
    if (
        not math.isfinite(learning_rate)
        or not MIN_LEARNING_RATE <= learning_rate <= MAX_LEARNING_RATE
    ):
        raise ValueError(
            f"learning rate must be finite and in [{MIN_LEARNING_RATE:g}, {MAX_LEARNING_RATE:g}]"
        )
    if not math.isfinite(max_grad_norm) or not 0.0 < max_grad_norm <= MAX_GRAD_NORM:
        raise ValueError(f"gradient clip norm must be finite and in (0, {MAX_GRAD_NORM:g}]")


def _validate_schedule(schedule: probe.FlowMatchSchedule) -> None:
    if schedule.fixture_sha256 != probe.FIXTURE_SHA256:
        raise ValueError("distillation requires the pinned Qwen-Image 2.1 flow-match fixture")
    if schedule.teacher_step_indices != (0, 1) or schedule.student_step_indices != (0,):
        raise ValueError("distillation requires teacher steps (0, 1) and student step (0,)")
    if schedule.sigmas != PINNED_SIGMAS or schedule.timesteps != PINNED_TIMESTEPS:
        raise ValueError("distillation schedule nodes differ from the pinned flow-match fixture")


def _validate_bundle_geometry(
    bundle: probe.ConditioningBundle, schedule: probe.FlowMatchSchedule, label: str
) -> None:
    _validate_schedule(schedule)
    latent_shape = tuple(bundle.initial_target_latents.shape)
    if latent_shape != (1, TARGET_TOKENS, TARGET_LATENT_CHANNELS):
        raise ValueError(
            f"{label} conditioning bundle must use the 256-token target geometry"
        )
    if bundle.img_shapes != TARGET_IMG_SHAPES:
        raise ValueError(f"{label} conditioning bundle img_shapes differ from the 256x256 fixture")
    hidden_shape = tuple(bundle.encoder_hidden_states.shape)
    if (
        len(hidden_shape) != 3
        or hidden_shape[0] != 1
        or hidden_shape[1] < 1
        or hidden_shape[2] != 4096
    ):
        raise ValueError(
            f"{label} conditioning hidden states must have batch size one and width 4096"
        )
    expected_mask_shape = hidden_shape[:2]
    if (
        tuple(bundle.encoder_hidden_states_mask.shape) != expected_mask_shape
        or tuple(bundle.encoder_img_mask.shape) != expected_mask_shape
    ):
        raise ValueError(f"{label} conditioning masks do not match the encoder sequence geometry")


def _validate_heldout_bundle(
    training_bundle: probe.ConditioningBundle,
    heldout_bundle: probe.ConditioningBundle,
    schedule: probe.FlowMatchSchedule,
) -> tuple[str, str]:
    """Require held-out data to be a separately manifested same-geometry bundle."""
    _validate_bundle_geometry(training_bundle, schedule, "training")
    _validate_bundle_geometry(heldout_bundle, schedule, "held-out")
    training_manifest = Path(training_bundle.manifest_path).resolve()
    heldout_manifest = Path(heldout_bundle.manifest_path).resolve()
    if training_manifest == heldout_manifest:
        raise ValueError("held-out evaluation requires a different conditioning manifest")
    try:
        training_manifest_sha256 = probe.sha256_file(training_manifest)
        heldout_manifest_sha256 = probe.sha256_file(heldout_manifest)
    except OSError as exc:
        raise ValueError("held-out evaluation requires readable conditioning manifests") from exc
    if training_manifest_sha256 == heldout_manifest_sha256:
        raise ValueError("held-out evaluation cannot reuse the training conditioning manifest")
    if training_bundle.payload_sha256 == heldout_bundle.payload_sha256:
        raise ValueError("held-out evaluation requires a distinct conditioning payload")
    if training_bundle.prompt == heldout_bundle.prompt:
        raise ValueError("held-out evaluation requires a distinct conditioning prompt")
    return heldout_manifest_sha256, heldout_bundle.payload_sha256


def _validate_heldout_manifest_provenance(manifest_path: Path) -> None:
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read held-out conditioning manifest: {manifest_path}") from exc
    if not isinstance(manifest, dict):
        raise ValueError("held-out bundle has invalid CPU conditioning provenance")
    model = manifest.get("model", {})
    runtime = manifest.get("runtime", {})
    noise = manifest.get("noise", {})
    image = manifest.get("image", {})
    if not all(isinstance(section, dict) for section in (model, runtime, noise, image)):
        raise ValueError("held-out bundle has invalid CPU conditioning provenance")
    if (
        model.get("repo") != "Qwen/Qwen-Image-2.1"
        or model.get("revision") != probe.MODEL_REVISION
        or runtime.get("diffusers_commit") != probe.DIFFUSERS_COMMIT
        or runtime.get("device") != "cpu"
        or noise.get("source_dtype") != "bfloat16"
        or noise.get("generator_device") != "cpu"
        or noise.get("seed") != 7
        or image.get("width") != 256
        or image.get("height") != 256
        or image.get("latent_height") != 16
        or image.get("latent_width") != 16
        or image.get("img_shapes") != [[1, 16, 16]]
    ):
        raise ValueError("held-out bundle has invalid CPU conditioning provenance")


def load_heldout_conditioning_bundle(
    manifest_path: str | Path,
    *,
    expected_manifest_sha256: str,
    expected_payload_sha256: str,
) -> probe.ConditioningBundle:
    """Load a variable-prompt held-out bundle under explicit content pins."""
    manifest_path = Path(manifest_path)
    for label, digest in (
        ("manifest", expected_manifest_sha256),
        ("payload", expected_payload_sha256),
    ):
        if not re.fullmatch(r"[0-9a-f]{64}", digest):
            raise ValueError(f"held-out {label} SHA256 pin must be 64 lowercase hex characters")
    probe.require_sha256(
        manifest_path, expected_manifest_sha256, "held-out conditioning manifest"
    )
    bundle = probe.load_conditioning_bundle(
        manifest_path, verify_pinned_bundle=False
    )
    if bundle.payload_sha256 != expected_payload_sha256:
        raise ValueError("held-out conditioning payload SHA256 mismatch")
    _validate_heldout_manifest_provenance(manifest_path)
    # The loader and provenance validator read the manifest independently. Recheck
    # after both reads so a concurrent rewrite cannot leave us with tensors loaded
    # under a manifest other than the explicitly pinned one.
    probe.require_sha256(
        manifest_path, expected_manifest_sha256, "held-out conditioning manifest"
    )
    return bundle


def _compute_teacher_endpoint(
    transformer: Any,
    bundle: probe.ConditioningBundle,
    schedule: probe.FlowMatchSchedule,
    *,
    device: Any,
    model_dtype: Any,
):
    """Compute the frozen base's two-substep endpoint exactly once."""
    import torch

    initial = bundle.initial_target_latents.detach().to(device=device, dtype=torch.float32)
    teacher_state = initial.clone()
    with torch.no_grad():
        for index in schedule.teacher_step_indices:
            velocity = probe._forward_velocity(
                transformer,
                bundle,
                teacher_state,
                schedule.timesteps[index],
                device,
                model_dtype,
            ).float()
            teacher_state = probe.euler_endpoint(
                teacher_state,
                (velocity,),
                (schedule.sigmas[index], schedule.sigmas[index + 1]),
            )
            if not probe._is_finite(teacher_state):
                raise ValueError("teacher endpoint became non-finite")
    return initial, teacher_state.detach()


def _evaluate_student_loss(
    transformer: Any,
    bundle: probe.ConditioningBundle,
    initial: Any,
    teacher_endpoint: Any,
    schedule: probe.FlowMatchSchedule,
    *,
    device: Any,
    model_dtype: Any,
) -> float:
    """Measure the student endpoint loss in the shared no-grad eval context."""
    import torch
    import torch.nn.functional as F

    transformer.eval()
    with torch.no_grad():
        student_velocity = probe._forward_velocity(
            transformer,
            bundle,
            initial,
            schedule.timesteps[schedule.student_step_indices[0]],
            device,
            model_dtype,
        ).float()
        student_endpoint = probe.euler_endpoint(
            initial,
            (student_velocity,),
            (schedule.sigmas[0], schedule.sigmas[2]),
        )
        loss = F.mse_loss(student_endpoint, teacher_endpoint, reduction="mean")
        if not probe._is_finite(student_endpoint):
            raise ValueError("student evaluation endpoint is non-finite")
        if not probe._is_finite(loss):
            raise ValueError("student evaluation loss is non-finite")
        return float(loss.item())


def _evaluate_heldout_loss(
    transformer: Any,
    bundle: probe.ConditioningBundle,
    initial: Any,
    teacher_endpoint: Any,
    schedule: probe.FlowMatchSchedule,
    *,
    adapter_parameters: Sequence[Any],
    device: Any,
    model_dtype: Any,
) -> float:
    """Measure held-out loss while proving evaluation left the adapter untouched."""
    import torch

    adapter_before = {
        parameter: parameter.detach().clone() for parameter in adapter_parameters
    }

    def restore_if_mutated() -> bool:
        mutated = any(
            not torch.equal(parameter.detach(), adapter_before[parameter])
            for parameter in adapter_parameters
        )
        if mutated:
            _restore_adapter_weights(adapter_parameters, adapter_before)
        return mutated

    try:
        loss = _evaluate_student_loss(
            transformer,
            bundle,
            initial,
            teacher_endpoint,
            schedule,
            device=device,
            model_dtype=model_dtype,
        )
    except Exception:
        restore_if_mutated()
        raise
    if restore_if_mutated():
        raise ValueError("held-out no-grad evaluation changed adapter weights; weights restored")
    if any(parameter.grad is not None for parameter in transformer.parameters()):
        raise ValueError("held-out no-grad evaluation unexpectedly populated gradients")
    return loss


def _gradient_norm(parameters: Sequence[Any]) -> float:
    squared = 0.0
    for parameter in parameters:
        if parameter.grad is None:
            raise ValueError("a selected LoRA parameter has no gradient")
        if not probe._is_finite(parameter.grad):
            raise ValueError("a selected LoRA gradient is non-finite")
        squared += float(parameter.grad.detach().float().pow(2).sum().item())
    return math.sqrt(squared)


def _has_base_gradient(transformer: Any, trainable_names: set[str]) -> bool:
    return any(
        parameter.grad is not None
        for name, parameter in transformer.named_parameters()
        if name not in trainable_names
    )


def _freeze_all_parameters(transformer: Any) -> None:
    for parameter in transformer.parameters():
        parameter.requires_grad_(False)
        parameter.grad = None


def _require_exact_adapter_contract(
    transformer: Any, trainable_names: tuple[str, ...]
) -> set[str]:
    """Fail closed if the installer selects anything beyond the pinned LoRA A/B."""
    expected = {
        f"{probe.TARGET_MODULE}.lora_A.{probe.ADAPTER_NAME}.weight",
        f"{probe.TARGET_MODULE}.lora_B.{probe.ADAPTER_NAME}.weight",
    }
    selected = set(trainable_names)
    if len(trainable_names) != 2 or selected != expected:
        _freeze_all_parameters(transformer)
        raise ValueError("adapter selection must be the exact LoRA A/B parameter pair")
    named = dict(transformer.named_parameters())
    if not expected.issubset(named):
        _freeze_all_parameters(transformer)
        raise ValueError("the exact LoRA A/B parameter pair is missing from the transformer")
    for name, parameter in named.items():
        if name in selected and not parameter.requires_grad:
            _freeze_all_parameters(transformer)
            raise ValueError(f"selected LoRA parameter unexpectedly frozen: {name}")
        if name not in selected and parameter.requires_grad:
            _freeze_all_parameters(transformer)
            raise ValueError(f"base parameter unexpectedly trainable: {name}")
        if name not in selected and parameter.grad is not None:
            _freeze_all_parameters(transformer)
            raise ValueError(f"base parameter unexpectedly has a gradient: {name}")
    return selected


def _restore_adapter_weights(parameters: Sequence[Any], previous: dict[Any, Any]) -> None:
    import torch

    with torch.no_grad():
        for parameter in parameters:
            parameter.copy_(previous[parameter])


def memory_snapshot(torch: Any, device: Any) -> dict[str, int | float | str]:
    """Capture device memory and an independent host-memory observation."""
    device_type = device.type
    if device_type == "mps":
        torch.mps.synchronize()
        device_memory = {
            "driver_allocated_bytes": int(torch.mps.driver_allocated_memory()),
            "current_allocated_bytes": int(torch.mps.current_allocated_memory()),
            "recommended_max_bytes": int(torch.mps.recommended_max_memory()),
        }
    elif device_type == "cuda":
        torch.cuda.synchronize(device)
        free_bytes, total_bytes = torch.cuda.mem_get_info(device)
        device_memory = {
            "allocated_bytes": int(torch.cuda.memory_allocated(device)),
            "reserved_bytes": int(torch.cuda.memory_reserved(device)),
            "free_bytes": int(free_bytes),
            "total_bytes": int(total_bytes),
        }
    else:
        device_memory = {}
    pressure_available, raw_reclaimable, total_host = probe._system_memory_bytes()
    return {
        "device_type": device_type,
        **device_memory,
        "pressure_estimated_available_host_bytes": int(pressure_available),
        "raw_reclaimable_host_bytes": int(raw_reclaimable),
        "total_host_bytes": int(total_host),
    }


def train_distillation(
    transformer: Any,
    bundle: probe.ConditioningBundle,
    schedule: probe.FlowMatchSchedule,
    *,
    device: Any,
    adapter_dtype: Any,
    conditioning_manifest_sha256_pin: str | None = None,
    heldout_bundle: probe.ConditioningBundle | None = None,
    heldout_manifest_sha256_pin: str | None = None,
    steps: int = DEFAULT_STEPS,
    learning_rate: float = DEFAULT_LEARNING_RATE,
    max_grad_norm: float = DEFAULT_MAX_GRAD_NORM,
    memory_snapshot_fn: Callable[[Any, Any], dict[str, int | float | str]] | None = None,
    on_step: Callable[[DistillationStep], None] | None = None,
) -> DistillationResult:
    """Train only the seeded rank-4 adapter against one fixed teacher target."""
    import torch
    import torch.nn.functional as F

    validate_training_parameters(
        steps=steps, learning_rate=learning_rate, max_grad_norm=max_grad_norm
    )
    _validate_schedule(schedule)
    if conditioning_manifest_sha256_pin is not None:
        if not re.fullmatch(r"[0-9a-f]{64}", conditioning_manifest_sha256_pin):
            raise ValueError("conditioning manifest SHA256 pin must be 64 lowercase hex characters")
        probe.require_sha256(
            bundle.manifest_path,
            conditioning_manifest_sha256_pin,
            "conditioning manifest",
        )
    heldout_manifest_sha256: str | None = None
    heldout_payload_sha256: str | None = None
    if heldout_bundle is not None:
        heldout_manifest_sha256, heldout_payload_sha256 = _validate_heldout_bundle(
            bundle, heldout_bundle, schedule
        )
        if heldout_manifest_sha256_pin is not None:
            if not re.fullmatch(r"[0-9a-f]{64}", heldout_manifest_sha256_pin):
                raise ValueError("held-out manifest SHA256 pin must be 64 lowercase hex characters")
            if heldout_manifest_sha256 != heldout_manifest_sha256_pin:
                raise ValueError("held-out conditioning manifest SHA256 mismatch")
    elif heldout_manifest_sha256_pin is not None:
        raise ValueError("held-out manifest SHA256 pin requires a held-out conditioning bundle")
    device = torch.device(device)
    if adapter_dtype != torch.float32:
        raise ValueError("distillation adapter dtype must be FP32")
    model_dtype = next(transformer.parameters()).dtype
    _freeze_all_parameters(transformer)
    transformer.eval()

    # The target is constructed once from the unwrapped, frozen base before LoRA is attached.
    initial, teacher_endpoint = _compute_teacher_endpoint(
        transformer,
        bundle,
        schedule,
        device=device,
        model_dtype=model_dtype,
    )
    if teacher_endpoint.requires_grad:
        raise ValueError("teacher endpoint must be detached from autograd")
    heldout_initial: Any | None = None
    heldout_teacher_endpoint: Any | None = None
    if heldout_bundle is not None:
        heldout_initial, heldout_teacher_endpoint = _compute_teacher_endpoint(
            transformer,
            heldout_bundle,
            schedule,
            device=device,
            model_dtype=model_dtype,
        )
        if heldout_teacher_endpoint.requires_grad:
            raise ValueError("held-out teacher endpoint must be detached from autograd")

    trainable_names_tuple = probe._install_probe_adapter(
        transformer, adapter_dtype=adapter_dtype, device=device
    )
    trainable_names = _require_exact_adapter_contract(transformer, trainable_names_tuple)
    adapter_parameters = [
        parameter
        for name, parameter in transformer.named_parameters()
        if name in trainable_names
    ]
    if len(adapter_parameters) != len(trainable_names_tuple) or not adapter_parameters:
        raise ValueError("selected LoRA parameters changed during adapter installation")
    if any(parameter.dtype != torch.float32 for parameter in adapter_parameters):
        raise ValueError("distillation adapter parameters must remain FP32")
    if any(not probe._is_finite(parameter) for parameter in adapter_parameters):
        raise ValueError("seeded adapter contains non-finite initial weights")
    initial_adapter_weights = {
        name: parameter.detach().clone()
        for name, parameter in transformer.named_parameters()
        if name in trainable_names
    }

    # Capture the seeded adapter's baseline with the exact no-grad evaluation
    # path also used after every update. This is an observation, not an update.
    for parameter in transformer.parameters():
        parameter.grad = None
    initial_eval_loss = _evaluate_student_loss(
        transformer,
        bundle,
        initial,
        teacher_endpoint,
        schedule,
        device=device,
        model_dtype=model_dtype,
    )
    if _has_base_gradient(transformer, trainable_names):
        raise ValueError(
            "frozen transformer base unexpectedly received a gradient during no-grad evaluation"
        )
    if any(parameter.grad is not None for parameter in adapter_parameters):
        raise ValueError("adapter unexpectedly received a gradient during no-grad evaluation")
    heldout_initial_eval_loss: float | None = None
    heldout_final_eval_loss: float | None = None
    if heldout_bundle is not None:
        assert heldout_initial is not None and heldout_teacher_endpoint is not None
        heldout_initial_eval_loss = _evaluate_heldout_loss(
            transformer,
            heldout_bundle,
            heldout_initial,
            heldout_teacher_endpoint,
            schedule,
            adapter_parameters=adapter_parameters,
            device=device,
            model_dtype=model_dtype,
        )

    optimizer = torch.optim.AdamW(
        adapter_parameters, lr=learning_rate, weight_decay=0.0
    )
    snapshot = memory_snapshot_fn or memory_snapshot
    reports: list[DistillationStep] = []
    try:
        for step_index in range(steps):
            optimizer.zero_grad(set_to_none=True)
            transformer.eval()
            with torch.enable_grad():
                student_velocity = probe._forward_velocity(
                    transformer,
                    bundle,
                    initial,
                    schedule.timesteps[schedule.student_step_indices[0]],
                    device,
                    model_dtype,
                ).float()
                student_endpoint = probe.euler_endpoint(
                    initial,
                    (student_velocity,),
                    (schedule.sigmas[0], schedule.sigmas[2]),
                )
                if not probe._is_finite(student_endpoint):
                    raise ValueError("student endpoint became non-finite")
                loss = F.mse_loss(student_endpoint, teacher_endpoint, reduction="mean")
                if not probe._is_finite(loss):
                    raise ValueError("endpoint loss is non-finite")
                loss.backward()

            # All fail-closed guards run before clipping and before AdamW.step().
            if _has_base_gradient(transformer, trainable_names):
                raise ValueError("frozen transformer base unexpectedly received a gradient")
            # Validate every adapter gradient before torch's clipping helper can
            # turn a non-finite norm into a runtime error or mutate the tensors.
            _gradient_norm(adapter_parameters)
            preclip_tensor = torch.nn.utils.clip_grad_norm_(
                adapter_parameters,
                max_norm=max_grad_norm,
                error_if_nonfinite=True,
            )
            preclip_norm = float(preclip_tensor.detach().item())
            if not math.isfinite(preclip_norm):
                raise ValueError("adapter gradient norm is non-finite")
            if preclip_norm == 0.0:
                raise ValueError("adapter gradient norm is zero")
            postclip_norm = _gradient_norm(adapter_parameters)
            if not math.isfinite(postclip_norm):
                raise ValueError("postclip adapter gradient norm is non-finite")
            if _has_base_gradient(transformer, trainable_names):
                raise ValueError("frozen transformer base unexpectedly received a gradient")

            pre_step_loss = float(loss.detach().item())
            before_step = {
                parameter: parameter.detach().clone()
                for parameter in adapter_parameters
            }
            try:
                optimizer.step()
            except Exception:
                _restore_adapter_weights(adapter_parameters, before_step)
                raise
            adapter_weights_finite = all(
                probe._is_finite(parameter) for parameter in adapter_parameters
            )
            if not adapter_weights_finite:
                _restore_adapter_weights(adapter_parameters, before_step)
                raise ValueError(
                    "AdamW update produced non-finite adapter weights; update rolled back"
                )
            post_step_loss: float | None = None
            post_step_loss_finite = False
            try:
                # The shared evaluator makes initial/final no-grad losses
                # directly comparable and always measures the updated adapter.
                post_step_loss = _evaluate_student_loss(
                    transformer,
                    bundle,
                    initial,
                    teacher_endpoint,
                    schedule,
                    device=device,
                    model_dtype=model_dtype,
                )
                post_step_loss_finite = math.isfinite(post_step_loss)
            except Exception as exc:
                _restore_adapter_weights(adapter_parameters, before_step)
                raise ValueError(
                    "post-step student loss could not be measured; adapter update rolled back"
                ) from exc
            if not post_step_loss_finite:
                _restore_adapter_weights(adapter_parameters, before_step)
                raise ValueError(
                    "post-step student loss is non-finite; adapter update rolled back"
                )
            if post_step_loss is None:
                _restore_adapter_weights(adapter_parameters, before_step)
                raise ValueError(
                    "post-step student loss is missing; adapter update rolled back"
                )
            if _has_base_gradient(transformer, trainable_names):
                _restore_adapter_weights(adapter_parameters, before_step)
                raise ValueError(
                    "frozen transformer base unexpectedly received a post-step gradient"
                )

            try:
                memory_after = snapshot(torch, device)
            except Exception:
                _restore_adapter_weights(adapter_parameters, before_step)
                raise
            report = DistillationStep(
                step=step_index + 1,
                pre_step_loss=pre_step_loss,
                post_step_loss=post_step_loss,
                post_step_loss_finite=post_step_loss_finite,
                preclip_gradient_norm=preclip_norm,
                postclip_gradient_norm=postclip_norm,
                adapter_weights_finite=adapter_weights_finite,
                memory_after=memory_after,
            )
            reports.append(report)
            if on_step is not None:
                on_step(report)
        if heldout_bundle is not None:
            assert heldout_initial is not None and heldout_teacher_endpoint is not None
            # End the optimizer phase before the final held-out observation.
            # The evaluation must neither consume nor leave gradients for AdamW.
            for parameter in transformer.parameters():
                parameter.grad = None
            heldout_final_eval_loss = _evaluate_heldout_loss(
                transformer,
                heldout_bundle,
                heldout_initial,
                heldout_teacher_endpoint,
                schedule,
                adapter_parameters=adapter_parameters,
                device=device,
                model_dtype=model_dtype,
            )
    except Exception:
        for parameter in transformer.parameters():
            parameter.grad = None
        raise

    heldout_evaluation = None
    if heldout_bundle is not None:
        assert heldout_manifest_sha256 is not None
        assert heldout_payload_sha256 is not None
        assert heldout_initial_eval_loss is not None
        assert heldout_final_eval_loss is not None
        assert heldout_teacher_endpoint is not None
        heldout_evaluation = HeldoutEvaluation(
            conditioning_manifest_sha256=heldout_manifest_sha256,
            conditioning_payload_sha256=heldout_payload_sha256,
            initial_eval_loss=heldout_initial_eval_loss,
            final_eval_loss=heldout_final_eval_loss,
            teacher_target_detached=not heldout_teacher_endpoint.requires_grad,
            teacher_endpoint_rms=float(
                heldout_teacher_endpoint.square().mean().sqrt().item()
            ),
        )

    return DistillationResult(
        target_module=probe.TARGET_MODULE,
        rank=probe.RANK,
        trainable_parameter_names=trainable_names_tuple,
        steps=tuple(reports),
        initial_eval_loss=initial_eval_loss,
        initial_adapter_weights=initial_adapter_weights,
        teacher_target_detached=not teacher_endpoint.requires_grad,
        base_parameters_have_no_grad=not _has_base_gradient(transformer, trainable_names),
        teacher_endpoint_rms=float(teacher_endpoint.square().mean().sqrt().item()),
        heldout_evaluation=heldout_evaluation,
    )


def parse_args(argv: Sequence[str] | None = None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", required=True, type=Path, help="local pinned transformer/ directory")
    parser.add_argument("--conditioning", required=True, type=Path, help="qwen_image21_conditioning.json")
    parser.add_argument(
        "--heldout-conditioning",
        type=Path,
        help="optional distinct 256x256 conditioning bundle for endpoint evaluation",
    )
    parser.add_argument(
        "--heldout-manifest-sha256",
        help="required expected SHA256 for --heldout-conditioning manifest",
    )
    parser.add_argument(
        "--heldout-payload-sha256",
        help="required expected SHA256 for --heldout-conditioning payload",
    )
    parser.add_argument(
        "--schedule",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "spec/fixtures/qwen_image21_flow_match_diffusers.json",
    )
    parser.add_argument("--device", choices=("mps",), default="mps")
    parser.add_argument("--dtype", choices=("bfloat16",), default="bfloat16")
    parser.add_argument("--mps-memory-fraction", type=float, default=0.70)
    parser.add_argument("--steps", type=int, default=DEFAULT_STEPS)
    parser.add_argument("--learning-rate", type=float, default=DEFAULT_LEARNING_RATE)
    parser.add_argument("--grad-clip-norm", type=float, default=DEFAULT_MAX_GRAD_NORM)
    parser.add_argument(
        "--adapter-output",
        type=Path,
        help="optional new directory for exact rank-4 FP32 LoRA checkpoint; requires held-out evaluation",
    )
    args = parser.parse_args(argv)
    try:
        validate_training_parameters(
            steps=args.steps,
            learning_rate=args.learning_rate,
            max_grad_norm=args.grad_clip_norm,
        )
    except ValueError as exc:
        parser.error(str(exc))
    heldout_pins = (args.heldout_manifest_sha256, args.heldout_payload_sha256)
    if args.heldout_conditioning is None:
        if any(value is not None for value in heldout_pins):
            parser.error("held-out SHA256 pins require --heldout-conditioning")
    else:
        if any(value is None for value in heldout_pins):
            parser.error(
                "--heldout-conditioning requires both --heldout-manifest-sha256 and "
                "--heldout-payload-sha256"
            )
        for name, value in zip(
            ("--heldout-manifest-sha256", "--heldout-payload-sha256"), heldout_pins
        ):
            if not re.fullmatch(r"[0-9a-f]{64}", value or ""):
                parser.error(f"{name} must contain 64 lowercase hexadecimal characters")
    if args.adapter_output is not None:
        if args.heldout_conditioning is None:
            parser.error("--adapter-output requires --heldout-conditioning and both SHA256 pins")
        if args.adapter_output.exists():
            parser.error("--adapter-output must name a new directory")
    return args


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    import torch

    schedule = probe.load_flow_match_schedule(args.schedule)
    bundle = probe.load_conditioning_bundle(args.conditioning)
    heldout_bundle = None
    if args.heldout_conditioning is not None:
        assert args.heldout_manifest_sha256 is not None
        assert args.heldout_payload_sha256 is not None
        heldout_bundle = load_heldout_conditioning_bundle(
            args.heldout_conditioning,
            expected_manifest_sha256=args.heldout_manifest_sha256,
            expected_payload_sha256=args.heldout_payload_sha256,
        )
        _validate_heldout_bundle(bundle, heldout_bundle, schedule)
    installed_diffusers_commit = probe.require_pinned_diffusers_installation()
    installed_peft_version, installed_peft_path = probe.require_compatible_peft_installation()
    model_config, checkpoint_bytes = probe.verify_snapshot_integrity(args.model_dir)
    device = probe._resolve_device(torch, args.device)
    dtype = probe._parse_dtype(torch, args.dtype)
    memory_guard = probe._guard_local_memory(
        torch, checkpoint_bytes, device, args.mps_memory_fraction
    )
    probe._set_device_memory_cap(torch, device, args.mps_memory_fraction)
    transformer = probe._load_transformer(args.model_dir, device, dtype)

    # Bundle tensors were loaded before model initialization; recheck their source
    # pins at the training boundary so a concurrent manifest rewrite cannot alter
    # the provenance attached to this run.
    probe.require_sha256(
        args.conditioning,
        probe.CONDITIONING_MANIFEST_SHA256,
        "conditioning manifest",
    )
    if args.heldout_conditioning is not None:
        assert args.heldout_manifest_sha256 is not None
        probe.require_sha256(
            args.heldout_conditioning,
            args.heldout_manifest_sha256,
            "held-out conditioning manifest",
        )

    def log_step(step: DistillationStep) -> None:
        print(
            json.dumps({"event": "distillation_step", **asdict(step)}, sort_keys=True),
            flush=True,
        )

    result = train_distillation(
        transformer,
        bundle,
        schedule,
        device=device,
        adapter_dtype=torch.float32,
        conditioning_manifest_sha256_pin=probe.CONDITIONING_MANIFEST_SHA256,
        heldout_bundle=heldout_bundle,
        heldout_manifest_sha256_pin=args.heldout_manifest_sha256,
        steps=args.steps,
        learning_rate=args.learning_rate,
        max_grad_norm=args.grad_clip_norm,
        on_step=log_step,
    )
    checkpoint_record = None
    if args.adapter_output is not None:
        assert heldout_bundle is not None
        saved = adapter_checkpoint.export_adapter(
            transformer,
            args.adapter_output,
            train_bundle=bundle,
            heldout_bundle=heldout_bundle,
            flow_fixture_path=args.schedule,
            training_result=result,
        )
        checkpoint_record = {
            "directory": str(saved.directory),
            "manifest_sha256": saved.manifest_sha256,
            "payload_sha256": saved.payload_sha256,
            "trainable_parameter_names": saved.trainable_parameter_names,
        }
    report = {
        "status": "distillation_control_completed",
        "claim_scope": "bounded endpoint distillation control only; no image quality claim",
        "model_repo": "Qwen/Qwen-Image-2.1",
        "model_revision": probe.MODEL_REVISION,
        "diffusers_commit": probe.DIFFUSERS_COMMIT,
        "installed_diffusers_commit": installed_diffusers_commit,
        "installed_peft_version": installed_peft_version,
        "installed_peft_path": installed_peft_path,
        "transformer_class": model_config["_class_name"],
        "device": str(device),
        "model_dtype": args.dtype,
        "adapter_dtype": "float32",
        "adapter_init_seed": probe.ADAPTER_INIT_SEED,
        "conditioning_payload_sha256": bundle.payload_sha256,
        "flow_fixture_sha256": schedule.fixture_sha256,
        "checkpoint_bytes": checkpoint_bytes,
        "memory_guard": memory_guard,
        "target_module": result.target_module,
        "rank": result.rank,
        "trainable_parameter_names": result.trainable_parameter_names,
        "base_parameters_have_no_grad": result.base_parameters_have_no_grad,
        "teacher_endpoint_rms": result.teacher_endpoint_rms,
        "heldout_evaluation": (
            {
                **asdict(result.heldout_evaluation),
                "no_grad_context_delta": result.heldout_evaluation.no_grad_context_delta,
            }
            if result.heldout_evaluation is not None
            else None
        ),
        "initial_eval_loss": result.initial_eval_loss,
        "final_eval_loss": result.final_eval_loss,
        "no_grad_context_delta": result.no_grad_context_delta,
        "no_grad_context_delta_definition": (
            "final_eval_loss - initial_eval_loss; both use the shared no-grad student endpoint evaluator"
        ),
        "teacher_semantics": "fixed frozen same-base two Euler substeps over fixture nodes sigma[0:3]",
        "student_semantics": "one trainable coarse Euler update sigma[0] to sigma[2]",
        "optimizer": "AdamW",
        "weight_decay": 0.0,
        "optimizer_steps": len(result.steps),
        "learning_rate": args.learning_rate,
        "grad_clip_norm": args.grad_clip_norm,
        "checkpoint_written": checkpoint_record is not None,
        "adapter_checkpoint": checkpoint_record,
    }
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
