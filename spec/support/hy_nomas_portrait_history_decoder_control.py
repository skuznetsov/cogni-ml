#!/usr/bin/env python3
"""Qualify a T0-history arm's PixelUMM output on exactly the same latents.

The original portrait still authenticates the weights, source, noise and RoPE.
Only the supplied T1 prefix may change. Its clean T0 must remain byte-identical,
and its synthetic future must be the declared three copies of frames [50,60).
Default mode is CPU admission; execution has one terminal MPS transfer.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import sys

import hy_nomas_portrait_tail_control as tail

base = tail.base


def channel_prefix(raw: bytes, *, frames: int, selected: int) -> bytes:
    frame_bytes = math.prod(tail.LATENT60_SHAPE[3:]) * 4
    expected = tail.LATENT60_SHAPE[1] * frames * frame_bytes
    if len(raw) != expected or not 0 < selected <= frames:
        raise base.ControlError("channel prefix geometry mismatch")
    return b"".join(raw[channel * frames * frame_bytes:
                        channel * frames * frame_bytes + selected * frame_bytes]
                    for channel in range(tail.LATENT60_SHAPE[1]))


def validate_arm_prefix(raw60: bytes, accepted_bank: bytes) -> None:
    if len(raw60) != math.prod(tail.LATENT60_SHAPE) * 4:
        raise base.ControlError("same-arm 60-latent extent mismatch")
    if channel_prefix(raw60, frames=60, selected=30) != channel_prefix(
            accepted_bank, frames=base.LATENT_BANK_SHAPE[2], selected=30):
        raise base.ControlError("history arm changed the clean T0 reference")


def validate_native_noise(receipt: dict) -> None:
    required = {
        "pixel_noise_manifest_sha256": base.INPUT_MANIFEST_SHA256,
        "pixel_noise_record_sha256": base.NOISE_SHA256,
        "continuation_manifest_sha256": base.INPUT_MANIFEST_SHA256,
        "decoder_rope_cos_record_sha256": base.ROPE_COS_SHA256,
        "decoder_rope_sin_record_sha256": base.ROPE_SIN_SHA256,
        "decoder_weight_cache_sha256": base.CACHE_MANIFEST_SHA256,
        "stem_weights_sha256": base.STEM_SHA256,
        "head_weights_sha256": base.HEAD_SHA256,
    }
    if any(receipt.get(key) != expected for key, expected in required.items()):
        raise base.ControlError("same-arm decoder noise, RoPE or weight identity changed")


def history_probe_metadata(keep_t0_latents, hide_t0_caption_keys, t1_current_text_kv):
    if t1_current_text_kv and not hide_t0_caption_keys:
        raise base.ControlError("T1 current text K/V requires hidden T0 caption keys")
    if hide_t0_caption_keys and keep_t0_latents != 30:
        raise base.ControlError("hiding T0 caption keys requires all 30 T0 video latents")
    return {
        "keep_t0_latents": keep_t0_latents,
        "t0_hide_caption_keys": hide_t0_caption_keys,
        "t1_current_text_kv": t1_current_text_kv,
    }


def preflight(args):
    history_metadata = history_probe_metadata(
        args.keep_t0_latents,
        getattr(args, "hide_t0_caption_keys", False),
        getattr(args, "t1_current_text_kv", False),
    )
    args.input_latents = None
    args.input_latents_sha256 = None
    admitted = base.preflight(args)
    prefix_path = args.prefix60.expanduser()
    if not prefix_path.is_absolute():
        raise base.ControlError("same-arm prefix60 must use an absolute path")
    prefix = base._read_regular(prefix_path, "same-arm prefix60",
                                max_bytes=math.prod(tail.LATENT60_SHAPE) * 4)
    prefix_sha = hashlib.sha256(prefix).hexdigest()
    base._require_digest(prefix_sha, args.prefix60_sha256, "same-arm prefix60")
    bank = base._read_regular(admitted["bank_path"], "accepted portrait latent bank",
                             max_bytes=math.prod(base.LATENT_BANK_SHAPE) * 4)
    validate_arm_prefix(prefix, bank)
    custom_path = args.custom_latents.expanduser()
    if not custom_path.is_absolute():
        raise base.ControlError("same-arm custom90 must use an absolute path")
    custom = base._read_regular(custom_path, "same-arm custom90", max_bytes=tail.LATENT90_BYTES)
    custom_sha = tail.validate_latent_artifact_bytes(custom, args.custom_latents_sha256)
    tail.validate_cyclic_tail_bytes(
        custom, channels=32, frames=90, frame_bytes=9 * 16 * 4,
        prefix=prefix, prefix_frames=60, cycle_start=50, cycle_frames=10, repeats=3,
    )
    native_pixels, native_receipt, receipt_sha = tail._read_native90(args, custom_path, custom_sha)
    validate_native_noise(native_receipt)
    admitted.update({
        "custom90_path": custom_path.resolve(strict=True),
        "custom90_sha256": custom_sha,
        "custom90_prefix_sha256": prefix_sha,
        "native90_pixels_path": native_pixels,
        "native90_receipt": native_receipt,
        "native90_receipt_sha256": receipt_sha,
        "history_probe": {
            **history_metadata,
            "same_arm_prefix60": str(prefix_path),
            "same_arm_prefix60_sha256": prefix_sha,
            "clean_t0_unchanged": True,
            "scope": "decoder parity only; history mask labels are caller metadata",
        },
    })
    return admitted


def main(argv=None):
    parser = tail._parser()
    parser.description = __doc__
    parser.add_argument("--prefix60", type=Path, required=True)
    parser.add_argument("--prefix60-sha256", required=True)
    parser.add_argument("--keep-t0-latents", type=int, choices=(10, 30), required=True)
    parser.add_argument("--hide-t0-caption-keys", action="store_true")
    parser.add_argument("--t1-current-text-kv", action="store_true",
                        help="record the admitted current T1 text K/V rescue metadata")
    args = parser.parse_args(argv)
    try:
        admitted = preflight(args)
        if not args.execute:
            print(json.dumps({"admission": "passed", "model_launched": False,
                              "history_probe": admitted["history_probe"],
                              "native90_receipt_sha256": admitted["native90_receipt_sha256"]}, indent=2))
            return 0
        result = tail.execute(admitted, historical_controls=False)
    except (base.ControlError, OSError, RuntimeError, ValueError, KeyError, TypeError) as error:
        print(f"portrait history decoder control failed: {error}", file=sys.stderr)
        return 2
    print(json.dumps({"comparison": result["comparison"], "gates": result["gates"],
                      "output": result["upstream_t1_display_path"]}, indent=2))
    return 0 if all(result["gates"].values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
