#!/usr/bin/env python3
"""Save and reload the bounded Qwen-Image 2.1 rank-4 LoRA adapter.

This opt-in checkpoint contains only the final Q projection's FP32 LoRA A/B
weights. Its manifest binds those tensors to the pinned source, flow fixture,
and exact train/heldout conditioning bundles; it makes no image-quality claim.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, Callable, Mapping

import qwen_image21_lora_grad_probe as probe


SCHEMA = "qwen-image21-lora-adapter-checkpoint"
SCHEMA_VERSION = 1
PAYLOAD_NAME = "adapter.safetensors"
MANIFEST_NAME = "manifest.json"
RANK = 4
ALPHA = 4
ALPHA_OVER_RANK = 1.0
ADAPTER_DTYPE = "float32"
MAX_OPTIMIZER_STEPS = 8
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_EXPECTED_TENSORS = (
    f"{probe.TARGET_MODULE}.lora_A.{probe.ADAPTER_NAME}.weight",
    f"{probe.TARGET_MODULE}.lora_B.{probe.ADAPTER_NAME}.weight",
)


@dataclass(frozen=True)
class AdapterCheckpoint:
    directory: Path
    manifest: dict[str, Any]
    manifest_sha256: str
    payload_sha256: str
    trainable_parameter_names: tuple[str, ...]


_VERIFIED_BASE_SEAL = object()


def _freeze_json(value: Any) -> Any:
    if isinstance(value, dict):
        return MappingProxyType(
            {key: _freeze_json(item) for key, item in value.items()}
        )
    if isinstance(value, list):
        return tuple(_freeze_json(item) for item in value)
    return value


class VerifiedBase:
    """Transformer loaded from a snapshot verified by ``load_verified_base``.

    The constructor is intentionally sealed. ``load_adapter`` accepts this
    provenance wrapper rather than a caller-supplied transformer, since tensor
    dimensions alone cannot establish which base model is being adapted.
    The wrapper fields cannot be rebound, but callers must not mutate the
    transformer's parameters between verification and adapter loading.
    """

    __slots__ = (
        "transformer",
        "model_dir",
        "config",
        "source_revision",
        "diffusers_commit",
        "_seal",
    )

    def __init__(
        self,
        transformer: Any,
        model_dir: Path,
        config: Mapping[str, Any],
        *,
        _seal: object | None = None,
    ) -> None:
        if _seal is not _VERIFIED_BASE_SEAL:
            raise TypeError("VerifiedBase instances must come from load_verified_base")
        object.__setattr__(self, "transformer", transformer)
        object.__setattr__(self, "model_dir", model_dir)
        object.__setattr__(self, "config", _freeze_json(dict(config)))
        object.__setattr__(self, "source_revision", probe.MODEL_REVISION)
        object.__setattr__(self, "diffusers_commit", probe.DIFFUSERS_COMMIT)
        object.__setattr__(self, "_seal", _seal)

    def __setattr__(self, name: str, value: Any) -> None:
        raise AttributeError("VerifiedBase is immutable after construction")

    def __delattr__(self, name: str) -> None:
        raise AttributeError("VerifiedBase is immutable after construction")


def load_verified_base(
    model_dir: str | Path,
    *,
    device: Any,
    dtype: Any,
    before_load: Callable[[Mapping[str, Any], int], None] | None = None,
) -> VerifiedBase:
    """Verify the pinned snapshot, then load its transformer from that path.

    ``before_load`` is a narrow hook for resource guards that need the verified
    config and total checkpoint bytes before model allocation. It runs after
    the full integrity check and before loading; it cannot replace the checked
    path used by the loader.
    """
    if before_load is not None and not callable(before_load):
        raise TypeError("before_load must be callable")
    try:
        resolved_model_dir = Path(model_dir).resolve(strict=True)
    except OSError as exc:
        raise ValueError(f"pinned model directory is missing: {model_dir}") from exc
    installed_diffusers_commit = probe.require_pinned_diffusers_installation()
    if installed_diffusers_commit != probe.DIFFUSERS_COMMIT:
        raise RuntimeError(
            "installed Diffusers commit does not match the pinned source"
        )
    config, checkpoint_bytes = probe.verify_snapshot_integrity(resolved_model_dir)
    if (
        not isinstance(config, dict)
        or isinstance(checkpoint_bytes, bool)
        or not isinstance(checkpoint_bytes, int)
        or checkpoint_bytes <= 0
    ):
        raise ValueError("pinned snapshot verification returned malformed metadata")
    verified_config = _freeze_json(config)
    if before_load is not None:
        before_load(verified_config, checkpoint_bytes)
    transformer = probe._load_transformer(resolved_model_dir, device, dtype)
    if transformer is None:
        raise ValueError("pinned transformer loader returned no model")
    _base_projection(transformer)
    return VerifiedBase(
        transformer, resolved_model_dir, verified_config, _seal=_VERIFIED_BASE_SEAL
    )


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _is_sha256(value: Any) -> bool:
    return isinstance(value, str) and _SHA256_RE.fullmatch(value) is not None


def _require_sha256(value: Any, label: str) -> str:
    if not _is_sha256(value):
        raise ValueError(f"{label} must be a lowercase SHA256 digest")
    return value


def _parse_json(raw: bytes, label: str) -> dict[str, Any]:
    def object_without_duplicates(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate JSON key in {label}: {key}")
            result[key] = value
        return result

    def reject_constant(value):
        raise ValueError(f"invalid JSON constant in {label}: {value}")

    try:
        result = json.loads(
            raw.decode("utf-8"),
            object_pairs_hook=object_without_duplicates,
            parse_constant=reject_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot parse {label} JSON") from exc
    if not isinstance(result, dict):
        raise ValueError(f"{label} JSON must contain an object")
    return result


def _bundle_digests(bundle: Any, label: str) -> dict[str, str]:
    manifest_path = getattr(bundle, "manifest_path", None)
    if not isinstance(manifest_path, (str, Path)):
        raise ValueError(f"{label} bundle has no manifest path")
    manifest_path = Path(manifest_path)
    try:
        raw_manifest = manifest_path.read_bytes()
    except OSError as exc:
        raise ValueError(
            f"cannot read {label} conditioning manifest: {manifest_path}"
        ) from exc
    manifest = _parse_json(raw_manifest, f"{label} conditioning manifest")
    model = manifest.get("model")
    runtime = manifest.get("runtime")
    if (
        not isinstance(model, dict)
        or model.get("repo") != "Qwen/Qwen-Image-2.1"
        or model.get("revision") != probe.MODEL_REVISION
    ):
        raise ValueError(f"{label} conditioning manifest source revision mismatch")
    if (
        not isinstance(runtime, dict)
        or runtime.get("diffusers_commit") != probe.DIFFUSERS_COMMIT
    ):
        raise ValueError(f"{label} conditioning manifest Diffusers commit mismatch")

    payload_name = manifest.get("payload_file")
    if (
        not isinstance(payload_name, str)
        or not payload_name
        or Path(payload_name).name != payload_name
        or payload_name in {".", ".."}
    ):
        raise ValueError(f"{label} conditioning payload filename is malformed")
    payload_path = manifest_path.parent / payload_name
    try:
        payload_sha256 = _sha256_file(payload_path)
    except OSError as exc:
        raise ValueError(
            f"cannot read {label} conditioning payload: {payload_path}"
        ) from exc
    declared_payload_sha256 = manifest.get("payload_sha256")
    if (
        not _is_sha256(declared_payload_sha256)
        or payload_sha256 != declared_payload_sha256
    ):
        raise ValueError(f"{label} conditioning payload SHA256 mismatch")
    bundle_payload_sha256 = getattr(bundle, "payload_sha256", None)
    if bundle_payload_sha256 != payload_sha256:
        raise ValueError(f"{label} bundle payload SHA256 mismatch")
    return {
        "manifest_sha256": _sha256_bytes(raw_manifest),
        "payload_sha256": payload_sha256,
    }


def _flow_fixture_digest(path: str | Path) -> str:
    try:
        actual = _sha256_file(path)
    except OSError as exc:
        raise ValueError(f"cannot read flow fixture: {path}") from exc
    if actual != probe.FIXTURE_SHA256:
        raise ValueError(
            f"flow fixture SHA256 mismatch: expected {probe.FIXTURE_SHA256}, got {actual}"
        )
    return actual


def _expected_shapes(base_projection: Any) -> dict[str, tuple[int, int]]:
    input_features = getattr(base_projection, "in_features", None)
    output_features = getattr(base_projection, "out_features", None)
    if (
        isinstance(input_features, bool)
        or not isinstance(input_features, int)
        or input_features <= 0
        or isinstance(output_features, bool)
        or not isinstance(output_features, int)
        or output_features <= 0
    ):
        raise ValueError(
            f"target projection dimensions are malformed: {probe.TARGET_MODULE}"
        )
    return {
        _EXPECTED_TENSORS[0]: (RANK, input_features),
        _EXPECTED_TENSORS[1]: (output_features, RANK),
    }


def _base_projection(transformer: Any) -> Any:
    modules = dict(transformer.named_modules())
    target = modules.get(probe.TARGET_MODULE)
    if target is None:
        raise ValueError(f"target projection is missing: {probe.TARGET_MODULE}")
    return target


def _tensor_sha256(tensor: Any) -> str:
    import torch

    if tensor.dtype != torch.float32:
        raise ValueError("adapter tensor must be FP32")
    canonical = tensor.detach().to(device="cpu", dtype=torch.float32).contiguous()
    return _sha256_bytes(canonical.view(torch.uint8).numpy().tobytes())


def _adapter_weights(
    transformer: Any,
) -> tuple[dict[str, Any], dict[str, tuple[int, ...]]]:
    import torch

    named_parameters = dict(transformer.named_parameters())
    trainable_names = tuple(
        name for name, value in named_parameters.items() if value.requires_grad
    )
    if len(trainable_names) != len(_EXPECTED_TENSORS) or set(trainable_names) != set(
        _EXPECTED_TENSORS
    ):
        raise ValueError(
            "trainable parameters must be the exact LoRA A/B parameter pair"
        )
    if not set(_EXPECTED_TENSORS).issubset(named_parameters):
        raise ValueError("the exact LoRA A/B parameter pair is incomplete")
    for name, parameter in named_parameters.items():
        if name in _EXPECTED_TENSORS:
            if parameter.dtype != torch.float32:
                raise ValueError(f"adapter tensor must be FP32: {name}")
            if not bool(torch.isfinite(parameter).all().item()):
                raise ValueError(f"adapter tensor contains non-finite values: {name}")
        elif parameter.requires_grad:
            raise ValueError(f"unexpected trainable transformer parameter: {name}")

    projection = _base_projection(transformer)
    lora_a = getattr(projection, "lora_A", None)
    lora_b = getattr(projection, "lora_B", None)
    if (
        lora_a is None
        or lora_b is None
        or probe.ADAPTER_NAME not in lora_a
        or probe.ADAPTER_NAME not in lora_b
    ):
        raise ValueError(
            f"PEFT did not inject the exact target adapter: {probe.TARGET_MODULE}"
        )
    adapter_layer = lora_a[probe.ADAPTER_NAME]
    b_layer = lora_b[probe.ADAPTER_NAME]
    weights = {
        _EXPECTED_TENSORS[0]: adapter_layer.weight,
        _EXPECTED_TENSORS[1]: b_layer.weight,
    }
    expected_shapes = _expected_shapes(projection)
    for name, weight in weights.items():
        if tuple(weight.shape) != expected_shapes[name]:
            raise ValueError(f"adapter tensor shape mismatch: {name}")
    alpha = getattr(projection, "lora_alpha", {}).get(probe.ADAPTER_NAME)
    scaling = getattr(projection, "scaling", {}).get(probe.ADAPTER_NAME)
    if (
        alpha != ALPHA
        or not isinstance(scaling, (int, float))
        or not math.isclose(float(scaling), ALPHA_OVER_RANK, rel_tol=0.0, abs_tol=0.0)
    ):
        raise ValueError("adapter alpha/rank must equal 1")
    return weights, expected_shapes


def _finite_loss(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be a finite nonnegative number")
    number = float(value)
    if not math.isfinite(number) or number < 0.0:
        raise ValueError(f"{label} must be a finite nonnegative number")
    return number


def _validate_training_result(
    training_result: Any, heldout_digests: Mapping[str, str]
) -> tuple[int, dict[str, Any]]:
    names = getattr(training_result, "trainable_parameter_names", None)
    if (
        not isinstance(names, (tuple, list))
        or len(names) != 2
        or set(names) != set(_EXPECTED_TENSORS)
    ):
        raise ValueError(
            "training result must report the exact LoRA A/B parameter pair"
        )
    steps = getattr(training_result, "steps", None)
    try:
        count = len(steps)
    except TypeError as exc:
        raise ValueError("training result optimizer steps are malformed") from exc
    if isinstance(count, bool) or not 1 <= count <= MAX_OPTIMIZER_STEPS:
        raise ValueError(f"optimizer steps must be in [1, {MAX_OPTIMIZER_STEPS}]")
    if getattr(training_result, "base_parameters_have_no_grad", None) is not True:
        raise ValueError("training result does not certify frozen base parameters")
    if getattr(training_result, "teacher_target_detached", None) is not True:
        raise ValueError(
            "training result does not certify a detached train teacher target"
        )
    heldout = getattr(training_result, "heldout_evaluation", None)
    if heldout is None:
        raise ValueError("training result must include a held-out evaluation")
    if getattr(heldout, "teacher_target_detached", None) is not True:
        raise ValueError(
            "held-out evaluation does not certify a detached teacher target"
        )
    if (
        getattr(heldout, "conditioning_manifest_sha256", None)
        != heldout_digests["manifest_sha256"]
    ):
        raise ValueError(
            "held-out evaluation manifest hash does not match the exported bundle"
        )
    if (
        getattr(heldout, "conditioning_payload_sha256", None)
        != heldout_digests["payload_sha256"]
    ):
        raise ValueError(
            "held-out evaluation payload hash does not match the exported bundle"
        )

    evaluation = {
        "train_initial_eval_loss": _finite_loss(
            getattr(training_result, "initial_eval_loss", None), "initial training loss"
        ),
        "train_final_eval_loss": _finite_loss(
            getattr(training_result, "final_eval_loss", None), "final training loss"
        ),
        "heldout_initial_eval_loss": _finite_loss(
            getattr(heldout, "initial_eval_loss", None), "initial held-out loss"
        ),
        "heldout_final_eval_loss": _finite_loss(
            getattr(heldout, "final_eval_loss", None), "final held-out loss"
        ),
        "base_parameters_have_no_grad": True,
        "train_teacher_target_detached": True,
        "heldout_teacher_target_detached": True,
    }
    return count, evaluation


def _tensor_descriptors(weights: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    return {
        name: {
            "dtype": ADAPTER_DTYPE,
            "shape": list(weight.shape),
            "sha256": _tensor_sha256(weight),
        }
        for name, weight in sorted(weights.items())
    }


def _manifest_for(
    *,
    train_digests: Mapping[str, str],
    heldout_digests: Mapping[str, str],
    flow_fixture_sha256: str,
    optimizer_steps: int,
    evaluation: dict[str, Any],
    tensor_descriptors: dict[str, dict[str, Any]],
    payload_sha256: str,
    payload_nbytes: int,
) -> dict[str, Any]:
    return {
        "schema": SCHEMA,
        "schema_version": SCHEMA_VERSION,
        "model_repo": "Qwen/Qwen-Image-2.1",
        "source_revision": probe.MODEL_REVISION,
        "diffusers_commit": probe.DIFFUSERS_COMMIT,
        "adapter": {
            "target_module": probe.TARGET_MODULE,
            "adapter_name": probe.ADAPTER_NAME,
            "rank": RANK,
            "alpha": ALPHA,
            "alpha_over_rank": ALPHA_OVER_RANK,
            "dtype": ADAPTER_DTYPE,
            "bias": "none",
            "dropout": 0.0,
        },
        "flow_fixture_sha256": flow_fixture_sha256,
        "training": {
            "optimizer": "AdamW",
            "optimizer_steps": optimizer_steps,
            "adapter_init_seed": probe.ADAPTER_INIT_SEED,
        },
        "evaluation": evaluation,
        "bundles": {
            "train": dict(train_digests),
            "heldout": dict(heldout_digests),
        },
        "payload": {
            "file": PAYLOAD_NAME,
            "sha256": payload_sha256,
            "nbytes": payload_nbytes,
        },
        "tensors": tensor_descriptors,
    }


def export_adapter(
    transformer: Any,
    output_dir: str | Path,
    *,
    train_bundle: Any,
    heldout_bundle: Any,
    flow_fixture_path: str | Path,
    training_result: Any,
) -> AdapterCheckpoint:
    """Write a new, fail-closed two-file adapter checkpoint directory.

    The adapter payload is written first and ``manifest.json`` last. Existing
    output paths are never replaced. ``training_result`` is the result returned
    by ``train_distillation`` and supplies the exact trainable allowlist and
    optimizer-step count.
    """
    from safetensors.torch import save_file

    flow_fixture_sha256 = _flow_fixture_digest(flow_fixture_path)
    train_digests = _bundle_digests(train_bundle, "train")
    heldout_digests = _bundle_digests(heldout_bundle, "heldout")
    optimizer_steps, evaluation = _validate_training_result(
        training_result, heldout_digests
    )
    expected_names = tuple(getattr(training_result, "trainable_parameter_names"))
    weights, expected_shapes = _adapter_weights(transformer)
    if set(expected_names) != set(weights):
        raise ValueError(
            "training result trainable names do not match the transformer LoRA A/B pair"
        )
    for name, weight in weights.items():
        if tuple(weight.shape) != expected_shapes[name]:
            raise ValueError(f"adapter tensor shape mismatch: {name}")

    # Validate all source evidence before creating the output directory.
    import torch

    tensors = {
        name: value.detach().to(device="cpu", dtype=torch.float32).contiguous()
        for name, value in weights.items()
    }
    descriptors = _tensor_descriptors(tensors)

    output_dir = Path(output_dir)
    if not output_dir.name or output_dir == output_dir.parent:
        raise ValueError("output directory must name a new child path")
    if output_dir.exists():
        raise ValueError(f"checkpoint output already exists: {output_dir}")
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    if output_dir.exists():
        raise ValueError(f"checkpoint output already exists: {output_dir}")

    staging_dir = Path(
        tempfile.mkdtemp(prefix=f".{output_dir.name}.stage-", dir=output_dir.parent)
    )
    try:
        staged_payload = staging_dir / PAYLOAD_NAME
        save_file(tensors, staged_payload, metadata={"format": "pt"})
        payload_sha256 = _sha256_file(staged_payload)
        manifest = _manifest_for(
            train_digests=train_digests,
            heldout_digests=heldout_digests,
            flow_fixture_sha256=flow_fixture_sha256,
            optimizer_steps=optimizer_steps,
            evaluation=evaluation,
            tensor_descriptors=descriptors,
            payload_sha256=payload_sha256,
            payload_nbytes=staged_payload.stat().st_size,
        )
        manifest_bytes = (json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode(
            "utf-8"
        )
        staged_manifest = staging_dir / MANIFEST_NAME
        staged_manifest.write_bytes(manifest_bytes)

        # mkdir is the no-overwrite claim. Publish data before the manifest so
        # a reader can treat a missing manifest as an incomplete checkpoint.
        try:
            output_dir.mkdir()
        except FileExistsError as exc:
            raise ValueError(f"checkpoint output already exists: {output_dir}") from exc
        os.rename(staged_payload, output_dir / PAYLOAD_NAME)
        os.rename(staged_manifest, output_dir / MANIFEST_NAME)
    finally:
        shutil.rmtree(staging_dir, ignore_errors=True)

    manifest_sha256 = _sha256_file(output_dir / MANIFEST_NAME)
    return AdapterCheckpoint(
        directory=output_dir,
        manifest=manifest,
        manifest_sha256=manifest_sha256,
        payload_sha256=payload_sha256,
        trainable_parameter_names=tuple(sorted(_EXPECTED_TENSORS)),
    )


def _require_exact_keys(value: Any, expected: set[str], label: str) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != expected:
        raise ValueError(f"{label} fields are malformed or unexpected")
    return value


def _validate_manifest(
    manifest: dict[str, Any],
    *,
    train_bundle: Any,
    heldout_bundle: Any,
    flow_fixture_path: str | Path,
    payload_path: Path,
    expected_shapes: dict[str, tuple[int, int]],
) -> tuple[str, dict[str, Any]]:
    _require_exact_keys(
        manifest,
        {
            "schema",
            "schema_version",
            "model_repo",
            "source_revision",
            "diffusers_commit",
            "adapter",
            "flow_fixture_sha256",
            "training",
            "evaluation",
            "bundles",
            "payload",
            "tensors",
        },
        "checkpoint manifest",
    )
    if manifest["schema"] != SCHEMA or manifest["schema_version"] != SCHEMA_VERSION:
        raise ValueError("unsupported adapter checkpoint schema")
    if manifest["model_repo"] != "Qwen/Qwen-Image-2.1":
        raise ValueError("adapter checkpoint model repository mismatch")
    if manifest["source_revision"] != probe.MODEL_REVISION:
        raise ValueError("adapter checkpoint source revision mismatch")
    if manifest["diffusers_commit"] != probe.DIFFUSERS_COMMIT:
        raise ValueError("adapter checkpoint Diffusers commit mismatch")

    adapter = _require_exact_keys(
        manifest["adapter"],
        {
            "target_module",
            "adapter_name",
            "rank",
            "alpha",
            "alpha_over_rank",
            "dtype",
            "bias",
            "dropout",
        },
        "adapter configuration",
    )
    if adapter != {
        "target_module": probe.TARGET_MODULE,
        "adapter_name": probe.ADAPTER_NAME,
        "rank": RANK,
        "alpha": ALPHA,
        "alpha_over_rank": ALPHA_OVER_RANK,
        "dtype": ADAPTER_DTYPE,
        "bias": "none",
        "dropout": 0.0,
    }:
        raise ValueError("adapter checkpoint configuration mismatch")

    flow_sha256 = _require_sha256(
        manifest["flow_fixture_sha256"], "flow fixture SHA256"
    )
    if flow_sha256 != _flow_fixture_digest(flow_fixture_path):
        raise ValueError("adapter checkpoint flow fixture hash mismatch")
    training = _require_exact_keys(
        manifest["training"],
        {"optimizer", "optimizer_steps", "adapter_init_seed"},
        "training metadata",
    )
    steps = training["optimizer_steps"]
    if (
        training["optimizer"] != "AdamW"
        or isinstance(steps, bool)
        or not isinstance(steps, int)
        or not 1 <= steps <= MAX_OPTIMIZER_STEPS
        or training["adapter_init_seed"] != probe.ADAPTER_INIT_SEED
    ):
        raise ValueError("adapter checkpoint training metadata mismatch")

    evaluation = _require_exact_keys(
        manifest["evaluation"],
        {
            "train_initial_eval_loss",
            "train_final_eval_loss",
            "heldout_initial_eval_loss",
            "heldout_final_eval_loss",
            "base_parameters_have_no_grad",
            "train_teacher_target_detached",
            "heldout_teacher_target_detached",
        },
        "evaluation evidence",
    )
    for field in (
        "train_initial_eval_loss",
        "train_final_eval_loss",
        "heldout_initial_eval_loss",
        "heldout_final_eval_loss",
    ):
        _finite_loss(evaluation[field], field.replace("_", " "))
    if any(
        evaluation[field] is not True
        for field in (
            "base_parameters_have_no_grad",
            "train_teacher_target_detached",
            "heldout_teacher_target_detached",
        )
    ):
        raise ValueError("adapter checkpoint evaluation guards are not satisfied")

    bundles = _require_exact_keys(
        manifest["bundles"], {"train", "heldout"}, "bundle digests"
    )
    expected_bundles = {
        "train": _bundle_digests(train_bundle, "train"),
        "heldout": _bundle_digests(heldout_bundle, "heldout"),
    }
    for label in ("train", "heldout"):
        actual = _require_exact_keys(
            bundles[label],
            {"manifest_sha256", "payload_sha256"},
            f"{label} bundle digests",
        )
        for field, expected in expected_bundles[label].items():
            _require_sha256(actual[field], f"{label} bundle {field}")
            if actual[field] != expected:
                raise ValueError(
                    f"adapter checkpoint {label} bundle hash mismatch: {field}"
                )

    payload = _require_exact_keys(
        manifest["payload"], {"file", "sha256", "nbytes"}, "payload descriptor"
    )
    if payload["file"] != PAYLOAD_NAME or payload_path.name != PAYLOAD_NAME:
        raise ValueError("adapter checkpoint payload filename mismatch")
    payload_sha256 = _require_sha256(payload["sha256"], "adapter payload SHA256")
    if (
        isinstance(payload["nbytes"], bool)
        or not isinstance(payload["nbytes"], int)
        or payload["nbytes"] <= 0
    ):
        raise ValueError("adapter checkpoint payload size is malformed")
    try:
        payload_nbytes = payload_path.stat().st_size
        actual_payload_sha256 = _sha256_file(payload_path)
    except OSError as exc:
        raise ValueError("adapter checkpoint payload is missing or unreadable") from exc
    if payload_nbytes != payload["nbytes"] or actual_payload_sha256 != payload_sha256:
        raise ValueError("adapter checkpoint payload hash or size mismatch")

    tensor_descriptors = manifest["tensors"]
    if not isinstance(tensor_descriptors, dict) or set(tensor_descriptors) != set(
        _EXPECTED_TENSORS
    ):
        raise ValueError("adapter tensor descriptor set is incomplete or unexpected")
    for name in _EXPECTED_TENSORS:
        descriptor = _require_exact_keys(
            tensor_descriptors[name],
            {"dtype", "shape", "sha256"},
            f"tensor descriptor {name}",
        )
        _require_sha256(descriptor["sha256"], f"tensor {name} SHA256")
        if descriptor["dtype"] != ADAPTER_DTYPE:
            raise ValueError(f"adapter tensor dtype mismatch: {name}")
        if descriptor["shape"] != list(expected_shapes[name]):
            raise ValueError(f"adapter tensor shape mismatch: {name}")
    return payload_sha256, tensor_descriptors


def _load_adapter_into_transformer(
    transformer: Any,
    checkpoint_dir: str | Path,
    *,
    train_bundle: Any,
    heldout_bundle: Any,
    flow_fixture_path: str | Path,
    device: Any,
) -> AdapterCheckpoint:
    """Internal implementation; caller must supply a verified pinned base."""
    import torch
    from safetensors.torch import load_file

    checkpoint_dir = Path(checkpoint_dir)
    manifest_path = checkpoint_dir / MANIFEST_NAME
    payload_path = checkpoint_dir / PAYLOAD_NAME
    if (
        not checkpoint_dir.is_dir()
        or not manifest_path.is_file()
        or not payload_path.is_file()
    ):
        raise ValueError("adapter checkpoint pair is incomplete or missing")
    try:
        entries = {entry.name for entry in checkpoint_dir.iterdir()}
    except OSError as exc:
        raise ValueError("cannot inspect adapter checkpoint directory") from exc
    if entries != {MANIFEST_NAME, PAYLOAD_NAME}:
        raise ValueError("adapter checkpoint directory contains unexpected files")
    try:
        raw_manifest = manifest_path.read_bytes()
    except OSError as exc:
        raise ValueError(
            "adapter checkpoint manifest is missing or unreadable"
        ) from exc
    manifest = _parse_json(raw_manifest, "adapter checkpoint manifest")

    # Validate the base projection's dimensions while it is still unmodified.
    projection = _base_projection(transformer)
    if hasattr(projection, "lora_A") or hasattr(projection, "lora_B"):
        raise ValueError(
            "load_adapter requires a fresh transformer without an injected LoRA adapter"
        )
    expected_shapes = _expected_shapes(projection)
    payload_sha256, descriptors = _validate_manifest(
        manifest,
        train_bundle=train_bundle,
        heldout_bundle=heldout_bundle,
        flow_fixture_path=flow_fixture_path,
        payload_path=payload_path,
        expected_shapes=expected_shapes,
    )
    try:
        tensors = load_file(payload_path, device="cpu")
    except Exception as exc:
        raise ValueError("adapter safetensors payload is malformed") from exc
    if set(tensors) != set(_EXPECTED_TENSORS):
        raise ValueError("adapter safetensors tensor set is incomplete or unexpected")
    for name in _EXPECTED_TENSORS:
        tensor = tensors[name]
        descriptor = descriptors[name]
        if tensor.dtype != torch.float32:
            raise ValueError(f"adapter tensor dtype mismatch: {name}")
        if (
            list(tensor.shape) != descriptor["shape"]
            or tuple(tensor.shape) != expected_shapes[name]
        ):
            raise ValueError(f"adapter tensor shape mismatch: {name}")
        if not bool(torch.isfinite(tensor).all().item()):
            raise ValueError(f"adapter tensor contains non-finite values: {name}")
        if _tensor_sha256(tensor) != descriptor["sha256"]:
            raise ValueError(f"adapter tensor SHA256 mismatch: {name}")

    probe.require_compatible_peft_installation()
    try:
        trainable_names = probe._install_probe_adapter(
            transformer, adapter_dtype=torch.float32, device=torch.device(device)
        )
    except Exception as exc:
        raise ValueError(
            "could not inject the pinned rank-4 FP32 LoRA adapter"
        ) from exc
    if len(trainable_names) != 2 or set(trainable_names) != set(_EXPECTED_TENSORS):
        raise ValueError(
            "injected trainable parameters are not the exact LoRA A/B pair"
        )
    installed_weights, installed_shapes = _adapter_weights(transformer)
    if installed_shapes != expected_shapes:
        raise ValueError("injected adapter shape differs from the checkpoint target")
    with torch.no_grad():
        for name, parameter in installed_weights.items():
            parameter.copy_(
                tensors[name].to(device=parameter.device, dtype=parameter.dtype)
            )
    for name, parameter in installed_weights.items():
        if not bool(torch.isfinite(parameter).all().item()):
            raise ValueError(f"reloaded adapter tensor became non-finite: {name}")

    return AdapterCheckpoint(
        directory=checkpoint_dir,
        manifest=manifest,
        manifest_sha256=_sha256_bytes(raw_manifest),
        payload_sha256=payload_sha256,
        trainable_parameter_names=tuple(trainable_names),
    )


def load_adapter(
    verified_base: VerifiedBase,
    checkpoint_dir: str | Path,
    *,
    train_bundle: Any,
    heldout_bundle: Any,
    flow_fixture_path: str | Path,
    device: Any,
) -> AdapterCheckpoint:
    """Validate then inject weights into a base created by ``load_verified_base``.

    Raw transformers are rejected even if their target dimensions match: those
    dimensions do not establish the pinned base revision.
    """
    if (
        not isinstance(verified_base, VerifiedBase)
        or verified_base._seal is not _VERIFIED_BASE_SEAL
        or verified_base.source_revision != probe.MODEL_REVISION
        or verified_base.diffusers_commit != probe.DIFFUSERS_COMMIT
    ):
        raise TypeError("load_adapter requires a VerifiedBase from load_verified_base")
    return _load_adapter_into_transformer(
        verified_base.transformer,
        checkpoint_dir,
        train_bundle=train_bundle,
        heldout_bundle=heldout_bundle,
        flow_fixture_path=flow_fixture_path,
        device=device,
    )


def load_pinned_transformer_adapter(
    model_dir: str | Path,
    checkpoint_dir: str | Path,
    *,
    train_bundle: Any,
    heldout_bundle: Any,
    flow_fixture_path: str | Path,
    device: Any,
    dtype: Any,
    before_load: Callable[[Mapping[str, Any], int], None] | None = None,
) -> tuple[VerifiedBase, AdapterCheckpoint]:
    """Verify/load a pinned base, then inject a checkpointed adapter.

    Returns the verified base wrapper as well as the checkpoint record. Use the
    two-step ``load_verified_base``/``load_adapter`` APIs when frozen teacher
    endpoints must be computed before adapter injection.
    """
    verified_base = load_verified_base(
        model_dir,
        device=device,
        dtype=dtype,
        before_load=before_load,
    )
    checkpoint = load_adapter(
        verified_base,
        checkpoint_dir,
        train_bundle=train_bundle,
        heldout_bundle=heldout_bundle,
        flow_fixture_path=flow_fixture_path,
        device=device,
    )
    return verified_base, checkpoint
