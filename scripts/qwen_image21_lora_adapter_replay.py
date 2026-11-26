#!/usr/bin/env python3
"""Replay a pinned Qwen-Image 2.1 LoRA checkpoint against its recorded losses.

The command is opt-in and MPS/BF16-only. It checks the pinned official
transformer snapshot, the exact ``red cube`` training conditioning, a
user-supplied SHA256-pinned castle holdout, and the pinned flow fixture. It
reconstructs the seeded adapter's initial losses, then reloads a fresh verified
transformer and the saved adapter for final losses. Comparisons use an explicit
absolute/relative numerical tolerance; this is not a claim of bitwise parity or
image quality.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import re
from pathlib import Path
from typing import Any, Mapping, Sequence

import qwen_image21_lora_adapter_checkpoint as adapter_checkpoint
import qwen_image21_lora_distill_control as distill
import qwen_image21_lora_grad_probe as probe


# The model and FP32 adapter tensors are pinned, and the same BF16/MPS
# evaluator is used for training and replay. These are an acceptance envelope,
# not an empirical variance estimate: 1e-6 absolute and 1e-4 relative tolerate
# small BF16/MPS reduction-order differences while the held-out loss pair's
# combined envelope is separately required to stay below its recorded gain.
LOSS_ABS_TOL = 1e-6
LOSS_REL_TOL = 1e-4
TRAIN_PROMPT = "red cube"
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_LOSS_RULE = "absolute_delta <= max(abs_tol, rel_tol * abs(expected))"


def _require_sha256(value: Any, label: str) -> str:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise ValueError(f"{label} must be a lowercase SHA256 digest")
    return value


def _finite_loss(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be a finite nonnegative number")
    number = float(value)
    if not math.isfinite(number) or number < 0.0:
        raise ValueError(f"{label} must be a finite nonnegative number")
    return number


def verify_checkpoint_manifest_pin(
    checkpoint_dir: str | Path, expected_sha256: str | None
) -> str | None:
    """Check an optional manifest pin before loading the model snapshot."""
    if expected_sha256 is None:
        return None
    expected = _require_sha256(expected_sha256, "checkpoint manifest SHA256")
    manifest_path = Path(checkpoint_dir) / adapter_checkpoint.MANIFEST_NAME
    try:
        raw_manifest = manifest_path.read_bytes()
    except OSError as exc:
        raise ValueError(
            f"cannot read adapter checkpoint manifest: {manifest_path}"
        ) from exc
    actual = hashlib.sha256(raw_manifest).hexdigest()
    if actual != expected:
        raise ValueError(
            "adapter checkpoint manifest SHA256 differs from the explicit pin: "
            f"expected {expected}, got {actual}"
        )
    return actual


def compare_loss(label: str, observed: Any, expected: Any) -> dict[str, Any]:
    """Compare one endpoint loss using the documented non-bitwise tolerance."""
    actual = _finite_loss(observed, f"observed {label} loss")
    reference = _finite_loss(expected, f"checkpoint {label} loss")
    delta = actual - reference
    allowed = max(LOSS_ABS_TOL, LOSS_REL_TOL * abs(reference))
    return {
        "expected": reference,
        "observed": actual,
        "delta": delta,
        "absolute_delta": abs(delta),
        "allowed_absolute_delta": allowed,
        "within_tolerance": abs(delta) <= allowed,
        "rule": _LOSS_RULE,
        "abs_tol": LOSS_ABS_TOL,
        "rel_tol": LOSS_REL_TOL,
        "bitwise_equality_required": False,
    }


def validate_conditioning_bundles(
    train_bundle: Any, heldout_bundle: Any, schedule: Any
) -> None:
    """Require the pinned red-cube train bundle and a distinct castle holdout."""
    if getattr(train_bundle, "prompt", None) != TRAIN_PROMPT:
        raise ValueError("training prompt must be exactly 'red cube'")
    heldout_prompt = getattr(heldout_bundle, "prompt", None)
    if (
        not isinstance(heldout_prompt, str)
        or re.search(r"\bcastle\b", heldout_prompt, flags=re.IGNORECASE) is None
    ):
        raise ValueError("held-out prompt must identify a castle")
    distill._validate_heldout_bundle(train_bundle, heldout_bundle, schedule)


def _verify_bundle_digests(
    train_bundle: Any,
    heldout_bundle: Any,
    *,
    heldout_manifest_sha256: str,
    heldout_payload_sha256: str,
) -> dict[str, dict[str, str]]:
    train_digests = adapter_checkpoint._bundle_digests(train_bundle, "train")
    if (
        train_digests["manifest_sha256"] != probe.CONDITIONING_MANIFEST_SHA256
        or train_digests["payload_sha256"] != probe.CONDITIONING_PAYLOAD_SHA256
    ):
        raise ValueError("training bundle is not the pinned red-cube conditioning")

    heldout_digests = adapter_checkpoint._bundle_digests(heldout_bundle, "held-out")
    if (
        heldout_digests["manifest_sha256"] != heldout_manifest_sha256
        or heldout_digests["payload_sha256"] != heldout_payload_sha256
    ):
        raise ValueError(
            "held-out castle bundle no longer matches its explicit SHA256 pins"
        )
    return {"train": train_digests, "heldout": heldout_digests}


def _expected_adapter_names() -> tuple[str, str]:
    return (
        f"{probe.TARGET_MODULE}.lora_A.{probe.ADAPTER_NAME}.weight",
        f"{probe.TARGET_MODULE}.lora_B.{probe.ADAPTER_NAME}.weight",
    )


def _require_adapter_names(names: Any, label: str) -> tuple[str, str]:
    expected = set(_expected_adapter_names())
    if (
        not isinstance(names, (tuple, list))
        or len(names) != 2
        or set(names) != expected
    ):
        raise ValueError(f"{label} must select the exact pinned LoRA A/B pair")
    return tuple(names)


def _require_base_frozen(transformer: Any) -> None:
    for name, parameter in transformer.named_parameters():
        if parameter.requires_grad or parameter.grad is not None:
            raise ValueError(
                f"fresh base parameter is not frozen and gradient-free: {name}"
            )


def _require_no_gradients(transformer: Any, label: str) -> None:
    if any(parameter.grad is not None for parameter in transformer.parameters()):
        raise ValueError(f"{label} no-grad evaluation unexpectedly populated gradients")


def _compute_teacher_pair(
    transformer: Any,
    train_bundle: Any,
    heldout_bundle: Any,
    schedule: Any,
    *,
    device: Any,
    model_dtype: Any,
) -> tuple[tuple[Any, Any], tuple[Any, Any]]:
    train_initial, train_teacher = distill._compute_teacher_endpoint(
        transformer,
        train_bundle,
        schedule,
        device=device,
        model_dtype=model_dtype,
    )
    heldout_initial, heldout_teacher = distill._compute_teacher_endpoint(
        transformer,
        heldout_bundle,
        schedule,
        device=device,
        model_dtype=model_dtype,
    )
    for label, initial, teacher in (
        ("training", train_initial, train_teacher),
        ("held-out", heldout_initial, heldout_teacher),
    ):
        if initial.requires_grad or teacher.requires_grad:
            raise ValueError(f"{label} teacher endpoint must be detached")
    return (train_initial, train_teacher), (heldout_initial, heldout_teacher)


def _evaluate_pair(
    transformer: Any,
    bundles: tuple[Any, Any],
    targets: tuple[tuple[Any, Any], tuple[Any, Any]],
    schedule: Any,
    *,
    device: Any,
    model_dtype: Any,
    label: str,
) -> tuple[float, float]:
    values = []
    for bundle, (initial, teacher), split in zip(
        bundles, targets, ("train", "heldout")
    ):
        value = distill._evaluate_student_loss(
            transformer,
            bundle,
            initial,
            teacher,
            schedule,
            device=device,
            model_dtype=model_dtype,
        )
        _require_no_gradients(transformer, f"{label} {split}")
        values.append(_finite_loss(value, f"{label} {split}"))
    return values[0], values[1]


def _release_model_memory(torch: Any, device: Any) -> None:
    gc.collect()
    if device.type == "mps":
        torch.mps.synchronize()
        empty_cache = getattr(torch.mps, "empty_cache", None)
        if not callable(empty_cache):
            raise RuntimeError(
                "this PyTorch build cannot empty the MPS cache between replay passes"
            )
        empty_cache()


def _load_verified_base(
    args: argparse.Namespace,
    *,
    torch: Any,
    device: Any,
    dtype: Any,
    memory_guards: list[dict[str, Any]],
) -> Any:
    """Verify the official snapshot, guard memory, then construct from that path."""

    def before_load(_config: Mapping[str, Any], checkpoint_bytes: int) -> None:
        memory = probe._guard_local_memory(
            torch, checkpoint_bytes, device, args.mps_memory_fraction
        )
        probe._set_device_memory_cap(torch, device, args.mps_memory_fraction)
        memory_guards.append(memory)

    # load_verified_base calls verify_snapshot_integrity(model_dir) and then
    # _load_transformer on that same directory, invoking the guard only after
    # the pinned hashes pass and before model allocation.
    verified = adapter_checkpoint.load_verified_base(
        args.model_dir,
        device=device,
        dtype=dtype,
        before_load=before_load,
    )
    if Path(verified.model_dir).resolve() != Path(args.model_dir).resolve():
        raise ValueError(
            "verified base model directory differs from the requested pinned snapshot"
        )
    if verified.transformer.training:
        raise ValueError("verified base transformer must be in eval mode")
    return verified


def run_replay(
    args: argparse.Namespace, *, torch_module: Any | None = None
) -> dict[str, Any]:
    """Run a no-optimizer initial/final replay against one pinned checkpoint."""
    if torch_module is None:
        import torch as torch_module
    torch = torch_module

    # Fail on a stale or substituted checkpoint before allocating the large
    # transformer; load_adapter performs its full compatibility/hash checks
    # again when injecting the saved tensors.
    checkpoint_manifest_sha256_preflight = verify_checkpoint_manifest_pin(
        args.checkpoint_dir, args.checkpoint_manifest_sha256
    )

    schedule = probe.load_flow_match_schedule(args.schedule)
    train_bundle = probe.load_conditioning_bundle(args.conditioning)
    heldout_bundle = distill.load_heldout_conditioning_bundle(
        args.heldout_conditioning,
        expected_manifest_sha256=args.heldout_manifest_sha256,
        expected_payload_sha256=args.heldout_payload_sha256,
    )
    validate_conditioning_bundles(train_bundle, heldout_bundle, schedule)
    bundle_digests = _verify_bundle_digests(
        train_bundle,
        heldout_bundle,
        heldout_manifest_sha256=args.heldout_manifest_sha256,
        heldout_payload_sha256=args.heldout_payload_sha256,
    )

    installed_diffusers_commit = probe.require_pinned_diffusers_installation()
    installed_peft_version, installed_peft_path = (
        probe.require_compatible_peft_installation()
    )
    device = probe._resolve_device(torch, "mps")
    if device.type != "mps":
        raise ValueError("reference adapter replay is restricted to MPS")
    model_dtype = torch.bfloat16
    memory_guards: list[dict[str, Any]] = []

    # Pass 1 reproduces training's exact seeded initial adapter evaluation.
    # Both frozen teacher endpoints are computed before adapter injection.
    verified_initial = _load_verified_base(
        args,
        torch=torch,
        device=device,
        dtype=model_dtype,
        memory_guards=memory_guards,
    )
    initial_transformer = verified_initial.transformer
    _require_base_frozen(initial_transformer)
    bundle_digests = _verify_bundle_digests(
        train_bundle,
        heldout_bundle,
        heldout_manifest_sha256=args.heldout_manifest_sha256,
        heldout_payload_sha256=args.heldout_payload_sha256,
    )
    initial_targets = _compute_teacher_pair(
        initial_transformer,
        train_bundle,
        heldout_bundle,
        schedule,
        device=device,
        model_dtype=model_dtype,
    )
    initial_names = _require_adapter_names(
        probe._install_probe_adapter(
            initial_transformer,
            adapter_dtype=torch.float32,
            device=device,
        ),
        "seeded initial adapter",
    )
    distill._require_exact_adapter_contract(initial_transformer, initial_names)
    initial_losses = _evaluate_pair(
        initial_transformer,
        (train_bundle, heldout_bundle),
        initial_targets,
        schedule,
        device=device,
        model_dtype=model_dtype,
        label="initial",
    )
    del initial_targets
    del initial_transformer
    del verified_initial
    _release_model_memory(torch, device)

    # Pass 2 obtains another verified fresh base. Teacher endpoints are again
    # computed while the base is frozen and before the saved adapter is loaded.
    verified_final = _load_verified_base(
        args,
        torch=torch,
        device=device,
        dtype=model_dtype,
        memory_guards=memory_guards,
    )
    final_transformer = verified_final.transformer
    _require_base_frozen(final_transformer)
    bundle_digests = _verify_bundle_digests(
        train_bundle,
        heldout_bundle,
        heldout_manifest_sha256=args.heldout_manifest_sha256,
        heldout_payload_sha256=args.heldout_payload_sha256,
    )
    final_targets = _compute_teacher_pair(
        final_transformer,
        train_bundle,
        heldout_bundle,
        schedule,
        device=device,
        model_dtype=model_dtype,
    )
    checkpoint_record = adapter_checkpoint.load_adapter(
        verified_final,
        args.checkpoint_dir,
        train_bundle=train_bundle,
        heldout_bundle=heldout_bundle,
        flow_fixture_path=args.schedule,
        device=device,
    )
    if args.checkpoint_manifest_sha256 is not None:
        if checkpoint_record.manifest_sha256 != args.checkpoint_manifest_sha256:
            raise ValueError(
                "adapter checkpoint manifest SHA256 differs from the explicit pin"
            )
    if (
        checkpoint_manifest_sha256_preflight is not None
        and checkpoint_record.manifest_sha256 != checkpoint_manifest_sha256_preflight
    ):
        raise ValueError(
            "adapter checkpoint manifest changed after the preflight pin check"
        )
    final_names = _require_adapter_names(
        checkpoint_record.trainable_parameter_names, "reloaded adapter"
    )
    distill._require_exact_adapter_contract(final_transformer, final_names)
    final_losses = _evaluate_pair(
        final_transformer,
        (train_bundle, heldout_bundle),
        final_targets,
        schedule,
        device=device,
        model_dtype=model_dtype,
        label="final",
    )
    del final_targets

    evaluation = checkpoint_record.manifest["evaluation"]
    loss_comparisons = {
        "train_initial": compare_loss(
            "train_initial",
            initial_losses[0],
            evaluation["train_initial_eval_loss"],
        ),
        "train_final": compare_loss(
            "train_final",
            final_losses[0],
            evaluation["train_final_eval_loss"],
        ),
        "heldout_initial": compare_loss(
            "heldout_initial",
            initial_losses[1],
            evaluation["heldout_initial_eval_loss"],
        ),
        "heldout_final": compare_loss(
            "heldout_final",
            final_losses[1],
            evaluation["heldout_final_eval_loss"],
        ),
    }
    heldout_expected_improvement = float(
        evaluation["heldout_initial_eval_loss"]
    ) - float(evaluation["heldout_final_eval_loss"])
    heldout_tolerance_sum = (
        loss_comparisons["heldout_initial"]["allowed_absolute_delta"]
        + loss_comparisons["heldout_final"]["allowed_absolute_delta"]
    )
    if heldout_expected_improvement <= 0.0:
        raise ValueError("checkpoint must record a positive held-out loss improvement")
    tolerance_visibility = heldout_tolerance_sum < heldout_expected_improvement
    if not tolerance_visibility:
        raise ValueError(
            "loss tolerance could obscure the checkpoint's held-out improvement"
        )
    matches = all(item["within_tolerance"] for item in loss_comparisons.values())

    source_hashes = {
        "model_snapshot": {
            "config_sha256": probe.CONFIG_SHA256,
            "index_sha256": probe.INDEX_SHA256,
            "shards": dict(probe.SHARD_SHA256),
        },
        "train_conditioning": bundle_digests["train"],
        "heldout_conditioning": bundle_digests["heldout"],
        "flow_fixture_sha256": schedule.fixture_sha256,
        "adapter_checkpoint": {
            "manifest_sha256": checkpoint_record.manifest_sha256,
            "payload_sha256": checkpoint_record.payload_sha256,
        },
    }
    report = {
        "status": "reference_replay_passed"
        if matches
        else "reference_replay_loss_mismatch",
        "claim_scope": (
            "four endpoint losses replayed from one pinned Diffusers transformer and adapter checkpoint; "
            "no image-quality claim"
        ),
        "comparison": {
            "absolute_tolerance": LOSS_ABS_TOL,
            "relative_tolerance": LOSS_REL_TOL,
            "rule": _LOSS_RULE,
            "bitwise_parity_claimed": False,
        },
        "loss_comparisons": loss_comparisons,
        "heldout_improvement_visibility": {
            "checkpoint_initial_minus_final": heldout_expected_improvement,
            "sum_of_initial_and_final_tolerances": heldout_tolerance_sum,
            "tolerance_sum_below_improvement": tolerance_visibility,
            "replayed_initial_minus_final": initial_losses[1] - final_losses[1],
        },
        "model_repo": "Qwen/Qwen-Image-2.1",
        "model_revision": probe.MODEL_REVISION,
        "diffusers_commit": probe.DIFFUSERS_COMMIT,
        "installed_diffusers_commit": installed_diffusers_commit,
        "installed_peft_version": installed_peft_version,
        "installed_peft_path": installed_peft_path,
        "transformer_class": type(final_transformer).__name__,
        "device": str(device),
        "model_dtype": "bfloat16",
        "adapter_dtype": "float32",
        "adapter_init_seed": probe.ADAPTER_INIT_SEED,
        "checkpoint_manifest_sha256_pin": args.checkpoint_manifest_sha256,
        "memory_guards": memory_guards,
        "snapshot_verification": {
            "passes": 2,
            "method": "checkpoint.load_verified_base verifies the pinned snapshot then loads from that same model_dir",
            "load_adapter_base_identity_scope": (
                "this CLI supplies a VerifiedBase produced from the pinned model_dir; "
                "load_adapter is not a general arbitrary-transformer identity proof"
            ),
        },
        "source_hashes": source_hashes,
    }
    return report


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model-dir",
        required=True,
        type=Path,
        help="local official pinned transformer snapshot",
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
        help="castle conditioning manifest",
    )
    parser.add_argument(
        "--heldout-manifest-sha256",
        required=True,
        help="explicit SHA256 pin for the castle manifest",
    )
    parser.add_argument(
        "--heldout-payload-sha256",
        required=True,
        help="explicit SHA256 pin for the castle payload",
    )
    parser.add_argument(
        "--checkpoint-dir",
        required=True,
        type=Path,
        help="adapter checkpoint directory",
    )
    parser.add_argument(
        "--checkpoint-manifest-sha256",
        help="optional explicit SHA256 pin for checkpoint manifest",
    )
    parser.add_argument(
        "--schedule",
        type=Path,
        default=Path(__file__).resolve().parents[1]
        / "spec/fixtures/qwen_image21_flow_match_diffusers.json",
        help="pinned four-step flow-match fixture",
    )
    parser.add_argument("--device", choices=("mps",), default="mps")
    parser.add_argument("--dtype", choices=("bfloat16",), default="bfloat16")
    parser.add_argument("--mps-memory-fraction", type=float, default=0.70)
    args = parser.parse_args(argv)
    for option, value in (
        ("--heldout-manifest-sha256", args.heldout_manifest_sha256),
        ("--heldout-payload-sha256", args.heldout_payload_sha256),
    ):
        if _SHA256_RE.fullmatch(value or "") is None:
            parser.error(f"{option} must contain 64 lowercase hexadecimal characters")
    if (
        args.checkpoint_manifest_sha256 is not None
        and _SHA256_RE.fullmatch(args.checkpoint_manifest_sha256) is None
    ):
        parser.error(
            "--checkpoint-manifest-sha256 must contain 64 lowercase hexadecimal characters"
        )
    if not 0.1 <= args.mps_memory_fraction <= 0.85:
        parser.error("--mps-memory-fraction must be in [0.1, 0.85]")
    return args


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    report = run_replay(args)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["status"] == "reference_replay_passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
