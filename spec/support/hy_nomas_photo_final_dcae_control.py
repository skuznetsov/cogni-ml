#!/usr/bin/env python3
"""Admit an original upstream DCAE decode of the final photo T1 latent.

The paired T0+final-T1 latent bank is joined to the current four-step DiT
checkpoint control and a fresh native PixelUMM output receipt before the
upstream DCAE decoder can run. Plan mode reads CPU metadata only. Execute mode
uses the existing one-terminal-transfer MPS decoder path under the unchanged
18 GiB cap and 10% physical-memory reserve. The cross-decoder RGB comparison
is descriptive and does not claim numerical parity between decoders.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import sys
from typing import Any, Mapping


SUPPORT = Path(__file__).resolve().parent
REPO_ROOT = SUPPORT.parents[1]
EXAMPLES_SUPPORT = REPO_ROOT / "examples" / "support"
for entry in (str(EXAMPLES_SUPPORT), str(SUPPORT)):
    if entry not in sys.path:
        sys.path.insert(0, entry)

import dcae_v_checkpoint  # noqa: E402,F401
import hy_nomas_photo_dcae_decode_control as dcae  # noqa: E402
import hy_nomas_photo_pixelumm_control as photo  # noqa: E402


ControlError = dcae.ControlError

SCHEMA = "cogni_ml.hy_nomas_photo_final_dcae_decode/v1"
FINAL_PAIRED_BANK_SHA256 = "19250cf779b9218f8c2ae9055c6272fcd01444de1c30deaa37bf477eef97ff42"
CHECKPOINT_CONTROL_SHA256 = "dbc43825526bc484f804bb88069b2f3a9c8644a4541f956c44b87685e5fe6915"
FINAL_X0_SHA256 = "8b21a039c9a18e85b24bd2590de198bf2243ef25aa8acc0c55dd9b515b7e11e3"
PREFIX_CONTROL_RGB8_SHA256 = "98f3710e8cc4fba50ad1bcd66863c81e1d3f3a0f932a02a5c06e535d59619978"

PAIRED_LATENT_SHAPE = (1, 32, 60, 9, 16)
FULL_PIXEL_SHAPE = tuple(photo.base.PIXELS_SHAPE)
FINAL_RENDER_START = 120
FINAL_PIXEL_CROP = (120, 240)
PIXEL_FRAME_SHAPE = (1, 3, 120, 288, 512)
GIB = 1024**3
EXTERNAL_CAP_BYTES = 18 * GIB
RESERVE_FRACTION = 0.10


def _require_digest(actual: str, expected: str, label: str) -> None:
    if actual != expected:
        raise ControlError(f"{label} SHA-256 mismatch")


def _coerce_control_error(error: ValueError) -> ControlError:
    if isinstance(error, ControlError):
        return error
    return ControlError(str(error))


def validate_final_bank(bank: bytes) -> dict[str, Any]:
    """Require the pinned paired final bank, its original T0, and exact final x0."""
    actual = hashlib.sha256(bank).hexdigest()
    _require_digest(actual, FINAL_PAIRED_BANK_SHA256, "final paired60 bank")
    try:
        diagnostic = photo.validate_bank(bank, "final", FINAL_X0_SHA256)
    except ValueError as error:
        raise _coerce_control_error(error) from None
    return {**diagnostic, "paired_bank_sha256": actual,
            "shape": list(PAIRED_LATENT_SHAPE)}


def _validate_checkpoint_metrics(receipt: Mapping[str, Any]) -> None:
    expected_comparisons = ["velocity_bf16", "x0_bf16", "after_euler_bf16"]
    expected_metrics = ["max_abs", "rmse", "outside_tolerance", "unequal", "nonfinite"]
    if (receipt.get("comparison_order") != expected_comparisons
            or receipt.get("metric_order") != expected_metrics):
        raise ControlError("checkpoint comparison metric order is missing or changed")
    metrics = receipt.get("metrics")
    if type(metrics) is not list or len(metrics) != 4:
        raise ControlError("checkpoint receipt must contain exactly four step metric rows")
    for step, comparisons in enumerate(metrics, start=1):
        if type(comparisons) is not list or len(comparisons) != 3:
            raise ControlError(f"checkpoint step {step} must contain three comparison rows")
        for comparison in comparisons:
            if (type(comparison) is not list or len(comparison) != 5
                    or any(type(value) not in (int, float) or not math.isfinite(value)
                           for value in comparison)):
                raise ControlError("checkpoint receipt contains malformed or non-finite metrics")
            max_abs, rmse, outside_tolerance, unequal, nonfinite = comparison
            if (max_abs < 0 or rmse < 0 or outside_tolerance != 0
                    or unequal != 0 or nonfinite != 0):
                raise ControlError("checkpoint comparison has outside tolerance or non-finite values")


def validate_checkpoint_control(
    receipt: Mapping[str, Any], *, receipt_sha256: str,
    bank_sha256: str, bank_path: Path,
) -> dict[str, Any]:
    """Join the fixed checkpoint receipt to its exact step-four paired artifact."""
    _require_digest(receipt_sha256, CHECKPOINT_CONTROL_SHA256, "checkpoint comparison")
    _validate_checkpoint_metrics(receipt)
    if receipt.get("step4_x0_equals_final_endpoint") is not True:
        raise ControlError("checkpoint receipt does not certify the exact step-four final endpoint")
    expected_runtime = {
        "torch": "2.8.0", "device": "mps", "intermediate_readbacks": 0,
        "explicit_synchronize_calls": 0, "terminal_transfers": 1,
    }
    if receipt.get("runtime") != expected_runtime:
        raise ControlError("checkpoint receipt violates the GPU terminal-transfer contract")
    bank_path = dcae._regular_path(bank_path, "final paired photo bank")
    try:
        artifact = photo.validate_checkpoint_join(
            dict(receipt), "step4", bank_sha256, FINAL_X0_SHA256,
        )
    except ValueError as error:
        raise _coerce_control_error(error) from None
    artifact_path = artifact.get("path")
    if not isinstance(artifact_path, str) or not Path(artifact_path).is_absolute():
        raise ControlError("named step-four paired bank path must be absolute")
    try:
        resolved_artifact = dcae._regular_path(
            Path(artifact_path), "named step-four paired bank",
        )
        resolved_bank = bank_path.resolve(strict=True)
    except OSError:
        raise ControlError("named step-four paired bank path is unavailable") from None
    if resolved_artifact != resolved_bank:
        raise ControlError("paired bank path differs from the named step-four checkpoint artifact")
    native_x0 = receipt.get("native_x0_sha256")
    if (type(native_x0) is not list or len(native_x0) != 4
            or native_x0[3] != FINAL_X0_SHA256):
        raise ControlError("checkpoint receipt does not end at the pinned final x0 endpoint")
    return artifact


def validate_pixelumm_receipt(
    receipt: Mapping[str, Any], *, bank_sha256: str, bank_path: Path,
    pixels_sha256: str, pixels_path: Path, pixels_bytes: int,
) -> dict[str, Any]:
    """Require exact native PixelUMM resources, output bytes, and terminal metrics."""
    expected_bytes = math.prod(FULL_PIXEL_SHAPE) * 4
    if type(pixels_bytes) is not int or pixels_bytes != expected_bytes:
        raise ControlError("native PixelUMM pixel output extent mismatch")
    pixels_path = dcae._regular_path(pixels_path, "native PixelUMM pixels")
    bank_path = dcae._regular_path(bank_path, "final paired photo bank")
    try:
        receipt_output = receipt.get("output")
        receipt_latent = receipt.get("latent")
        if (not isinstance(receipt_output, str) or not Path(receipt_output).is_absolute()
                or not isinstance(receipt_latent, str) or not Path(receipt_latent).is_absolute()):
            raise ControlError("native PixelUMM receipt output and latent paths must be absolute")
        if Path(receipt_output).is_symlink() or Path(receipt_latent).is_symlink():
            raise ControlError("native PixelUMM receipt paths must not be symlinks")
        if Path(receipt_output).resolve(strict=True) != pixels_path.resolve(strict=True):
            raise ControlError("native PixelUMM receipt names a different output path")
        if Path(receipt_latent).resolve(strict=True) != bank_path.resolve(strict=True):
            raise ControlError("native PixelUMM receipt names a different paired bank path")
        photo.validate_replay(dict(receipt), bank_sha256, pixels_sha256)
    except ValueError as error:
        raise _coerce_control_error(error) from None
    return {"path": pixels_path, "sha256": pixels_sha256,
            "shape": list(FULL_PIXEL_SHAPE), "crop": list(FINAL_PIXEL_CROP),
            "bytes": expected_bytes}


def final_memory_estimate(base_estimate: Mapping[str, Any]) -> dict[str, Any]:
    """Name the final diagnostic's resident buffers without treating them as a peak bound."""
    latent_f32 = math.prod(PAIRED_LATENT_SHAPE) * 4
    latent_f32_copy = latent_f32
    latent_bf16 = math.prod(PAIRED_LATENT_SHAPE) * 2
    full_output_bf16 = math.prod((1, 3, 240, 288, 512)) * 2
    frame_f32 = math.prod(PIXEL_FRAME_SHAPE) * 4
    pixel_reference_full_f32 = math.prod(FULL_PIXEL_SHAPE) * 4
    pixel_reference_crop_f32 = math.prod(PIXEL_FRAME_SHAPE) * 4
    rgb8_terminal = math.prod((120, 288, 512, 3))
    combined_terminal_pack = 2 * rgb8_terminal + 16
    inventory = {
        "host_dcae_decoder_bf16_checkpoint_bytes": dcae.DECODER_BF16_BYTES,
        "device_dcae_decoder_bf16_weights_bytes": dcae.DECODER_BF16_BYTES,
        "host_paired60_latent_f32_source_bytes": latent_f32,
        "host_paired60_latent_f32_execution_copy_bytes": latent_f32_copy,
        "device_paired60_latent_bf16_bytes": latent_bf16,
        "device_full240_dcae_bf16_tile_and_output_bytes": 2 * full_output_bf16,
        "host_pixelumm_full240_f32_read_buffer_bytes": pixel_reference_full_f32,
        "host_pixelumm_final120_f32_selected_crop_bytes": pixel_reference_crop_f32,
        "device_pixelumm_final120_f32_crop_bytes": pixel_reference_crop_f32,
        "device_final120_f32_dcae_display_crop_bytes": frame_f32,
        "device_final120_f32_cross_decoder_delta_bytes": frame_f32,
        "device_final120_f32_squared_delta_staging_bytes": frame_f32,
        "device_prefix120_f32_input_cast_bytes": frame_f32,
        "device_prefix120_f32_render_staging_bytes": 2 * frame_f32,
        "device_final120_f32_render_staging_bytes": 2 * frame_f32,
        "device_prefix120_rgb8_staging_bytes": rgb8_terminal,
        "device_final120_rgb8_staging_bytes": rgb8_terminal,
        "device_final_counter_metrics_bytes": 16,
        "device_combined_terminal_pack_bytes": combined_terminal_pack,
        "host_combined_terminal_pack_bytes": combined_terminal_pack,
        "host_final120_rgb8_write_copy_bytes": rgb8_terminal,
        "host_prefix120_rgb8_hash_copy_bytes": rgb8_terminal,
    }
    named_bytes = sum(inventory.values())
    return {
        **dict(base_estimate),
        "named_bytes": named_bytes,
        "named_gib": named_bytes / GIB,
        "named_inventory": inventory,
        "external_cap_bytes": EXTERNAL_CAP_BYTES,
        "physical_reserve_fraction": RESERVE_FRACTION,
        "external_guard_required": True,
        "opaque_mps_workspace_included": False,
    }


def _read_hashed_json(path: Path, label: str, expected_sha256: str) -> tuple[dict, str]:
    path = dcae._regular_path(path, label)
    receipt, raw = dcae._read_json(path, label)
    digest = hashlib.sha256(raw).hexdigest()
    _require_digest(digest, expected_sha256, label)
    return receipt, digest


def _admit_external_artifacts(args: argparse.Namespace) -> dict[str, Any]:
    bank_path = dcae._regular_path(args.paired_latents, "final paired photo bank")
    bank = dcae._read_regular(bank_path, "final paired photo bank",
                              max_bytes=2 * photo.SEGMENT_BYTES)
    bank_sha256 = hashlib.sha256(bank).hexdigest()
    _require_digest(bank_sha256, args.paired_latents_sha256, "paired photo bank argument")
    bank_diagnostic = validate_final_bank(bank)

    checkpoint_path = dcae._regular_path(args.checkpoint_control, "checkpoint comparison")
    checkpoint, checkpoint_sha256 = _read_hashed_json(
        checkpoint_path, "checkpoint comparison", args.checkpoint_control_sha256,
    )
    _require_digest(checkpoint_sha256, CHECKPOINT_CONTROL_SHA256, "current checkpoint comparison")
    checkpoint_artifact = validate_checkpoint_control(
        checkpoint, receipt_sha256=checkpoint_sha256,
        bank_sha256=bank_sha256, bank_path=bank_path,
    )

    pixel_receipt_path = dcae._regular_path(args.pixelumm_receipt, "native PixelUMM receipt")
    pixel_receipt, pixel_receipt_sha256 = _read_hashed_json(
        pixel_receipt_path, "native PixelUMM receipt", args.pixelumm_receipt_sha256,
    )
    output_value = pixel_receipt.get("output")
    if not isinstance(output_value, str) or not Path(output_value).is_absolute():
        raise ControlError("native PixelUMM output path must be absolute")
    pixels_path = dcae._regular_path(Path(output_value), "native PixelUMM pixels")
    pixels_bytes = pixels_path.stat().st_size
    pixels_sha256 = dcae._sha256_file(pixels_path)
    pixel_reference = validate_pixelumm_receipt(
        pixel_receipt, bank_sha256=bank_sha256, bank_path=bank_path,
        pixels_sha256=pixels_sha256, pixels_path=pixels_path,
        pixels_bytes=pixels_bytes,
    )
    return {
        "bank_path": bank_path, "bank_sha256": bank_sha256,
        "bank_diagnostic": bank_diagnostic,
        "checkpoint_path": checkpoint_path,
        "checkpoint_sha256": checkpoint_sha256,
        "checkpoint_artifact": checkpoint_artifact,
        "pixel_receipt_path": pixel_receipt_path,
        "pixel_receipt_sha256": pixel_receipt_sha256,
        "pixelumm_reference": pixel_reference,
    }


def preflight(args: argparse.Namespace, torch: Any) -> dict[str, Any]:
    """Authenticate the paired diagnostic artifacts before DCAE metadata admission."""
    external = _admit_external_artifacts(args)
    admitted = dcae.admit(args, torch)
    admitted.update({
        "decode_input": {
            "path": external["bank_path"], "sha256": external["bank_sha256"],
            "shape": list(PAIRED_LATENT_SHAPE),
            "render_start_frame": FINAL_RENDER_START,
            "scope": "authenticated original-photo T0 plus step-four final x0; render T1 only",
        },
        "pixelumm_reference": external["pixelumm_reference"],
        "prefix_control": {"expected_rgb8_sha256": PREFIX_CONTROL_RGB8_SHA256},
        "output_schema": SCHEMA,
        "counterfactual": {
            "kind": "final-photo-step4-x0",
            "paired_latent_bank_sha256": external["bank_sha256"],
            "checkpoint_control_sha256": external["checkpoint_sha256"],
            "checkpoint_artifact_path": str(external["checkpoint_artifact"]["path"]),
            "native_pixelumm_receipt_sha256": external["pixel_receipt_sha256"],
            "native_pixelumm_pixels_sha256": external["pixelumm_reference"]["sha256"],
            "scope": "upstream DCAE-V alone decodes the original-photo T0 plus final x0; render frames [120,240)",
            "native_dcae_decoder_used": False,
            "dit_rerun": False,
            "native_pixelumm_rerun": False,
            "cross_decoder_appearance_comparison": "descriptive RGB difference only; no parity threshold",
            "checkpoint_comparison_atol": dcae.ATOL,
            "checkpoint_comparison_rtol": dcae.RTOL,
        },
    })
    admitted["memory_estimate"] = final_memory_estimate(admitted["memory_estimate"])
    admitted["photo_diagnostic"] = external["bank_diagnostic"]
    return admitted


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, default=dcae.DEFAULT_ROOT)
    parser.add_argument("--upstream", type=Path, default=dcae.DEFAULT_UPSTREAM)
    parser.add_argument("--photo", type=Path, default=dcae.DEFAULT_PHOTO)
    parser.add_argument("--weights", type=Path, default=dcae.DEFAULT_WEIGHTS)
    parser.add_argument("--output-dir", type=Path, default=dcae.DEFAULT_OUTPUT_DIR)
    parser.add_argument("--paired-latents", type=Path, required=True)
    parser.add_argument("--paired-latents-sha256", required=True)
    parser.add_argument("--checkpoint-control", type=Path, required=True)
    parser.add_argument("--checkpoint-control-sha256", required=True)
    parser.add_argument("--pixelumm-receipt", type=Path, required=True)
    parser.add_argument("--pixelumm-receipt-sha256", required=True)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--plan-only", action="store_true",
                      help="authenticate CPU metadata and receipts without MPS execution")
    mode.add_argument("--execute", action="store_true",
                      help="run only the upstream DCAE decoder on MPS")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    try:
        import torch

        admitted = preflight(args, torch)
        if args.execute:
            result = dcae.execute(admitted, torch)
        else:
            result = {
                "admission": "passed", "model_launched": False,
                "mps_work_started": False, "schema": SCHEMA,
                "decode_input": admitted["decode_input"],
                "counterfactual": admitted["counterfactual"],
                "memory_estimate": admitted["memory_estimate"],
                "output_dir": str(admitted["output_dir"]),
            }
        print(json.dumps(result, sort_keys=True, allow_nan=False, default=str))
        return 0
    except (ControlError, ValueError, OSError, KeyError, TypeError, RuntimeError) as error:
        print(f"final-photo DCAE control failed: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
