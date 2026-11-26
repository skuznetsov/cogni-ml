#!/usr/bin/env python3
"""Replay a context-source arm through pinned PixelUMM on the same latents.

The arm's actual DiT bank authenticates both T0 and T1. This does not require
generated T0 to preserve the photograph, or claim that context-source changes
isolate temporal motion. All model computation remains in the existing MPS
decoder with one terminal transfer.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys

import hy_nomas_portrait_history_decoder_control as history

tail = history.tail
base = history.base


def validate_dit_prefix(receipt: dict, bank: bytes, prefix: bytes, mode: str) -> None:
    required = {
        "schema": "cogni_ml.hy_nomas_portrait_t0_history_probe/v1",
        "t0_context_mode": mode,
        "inputs_sha256": base.INPUT_MANIFEST_SHA256,
        "upstream_revision": base.UPSTREAM_REVISION,
        "source_bf16": True,
        "segments": 2,
        "steps_per_segment": 4,
        "keep_t0_video_latents": 30,
        "t0_hide_caption_keys": False,
        "t1_current_text_kv": False,
    }
    if mode not in ("repeated-photo", "generated") or any(
            type(receipt.get(k)) is not type(v) or receipt.get(k) != v
            for k, v in required.items()):
        raise base.ControlError("DiT context mode or frozen geometry mismatch")
    if len(bank) != 32 * 180 * 9 * 16 * 4:
        raise base.ControlError("DiT context bank extent mismatch")
    actual = hashlib.sha256(bank).hexdigest()
    if receipt.get("output", {}).get("latent_bank_sha256") != actual:
        raise base.ControlError("DiT context bank byte identity mismatch")
    expected = history.channel_prefix(bank, frames=180, selected=60)
    if prefix != expected:
        raise base.ControlError("decoder prefix differs from actual same-arm DiT bank")
    for k in ("chain_preterminal_allocations", "chain_preterminal_readbacks",
              "chain_preterminal_waits"):
        value = receipt.get("metrics", {}).get(k)
        if type(value) is not int or value != 0:
            raise base.ControlError("DiT context chain counter mismatch")
    waits = receipt.get("metrics", {}).get("dit_terminal_waits")
    if type(waits) is not int or waits != 1:
        raise base.ControlError("DiT context terminal wait counter mismatch")


def preflight(args):
    args.input_latents = args.input_latents_sha256 = None
    admitted = base.preflight(args)
    receipt_path = args.dit_receipt.expanduser()
    if not receipt_path.is_absolute():
        raise base.ControlError("DiT receipt must use an absolute path")
    raw = base._read_regular(receipt_path, "same-arm DiT receipt", max_bytes=1024 * 1024)
    digest = hashlib.sha256(raw).hexdigest()
    base._require_digest(digest, args.dit_receipt_sha256, "same-arm DiT receipt")
    receipt = json.loads(raw, object_pairs_hook=base._unique_object)
    bank_path = Path(receipt["output"]["latent_bank"])
    if not bank_path.is_absolute():
        raise base.ControlError("same-arm DiT bank must use an absolute path")
    bank = base._read_regular(bank_path, "same-arm DiT bank", max_bytes=32 * 180 * 9 * 16 * 4)
    prefix_path = args.prefix60.expanduser()
    if not prefix_path.is_absolute():
        raise base.ControlError("same-arm prefix must use an absolute path")
    prefix = base._read_regular(prefix_path, "same-arm prefix", max_bytes=32 * 60 * 9 * 16 * 4)
    prefix_sha = hashlib.sha256(prefix).hexdigest()
    base._require_digest(prefix_sha, args.prefix60_sha256, "same-arm prefix")
    validate_dit_prefix(receipt, bank, prefix, args.t0_context)
    if args.t0_context == "repeated-photo":
        old_bank = base._read_regular(admitted["bank_path"], "accepted photo bank",
                                      max_bytes=32 * 180 * 9 * 16 * 4)
        history.validate_arm_prefix(prefix, old_bank)
    custom_path = args.custom_latents.expanduser()
    if not custom_path.is_absolute():
        raise base.ControlError("same-arm custom90 must use an absolute path")
    custom = base._read_regular(custom_path, "same-arm custom90", max_bytes=tail.LATENT90_BYTES)
    custom_sha = tail.validate_latent_artifact_bytes(custom, args.custom_latents_sha256)
    tail.validate_cyclic_tail_bytes(custom, channels=32, frames=90, frame_bytes=9 * 16 * 4,
                                    prefix=prefix, prefix_frames=60, cycle_start=50,
                                    cycle_frames=10, repeats=3)
    pixels, native, native_sha = tail._read_native90(args, custom_path, custom_sha)
    history.validate_native_noise(native)
    admitted.update({
        "custom90_path": custom_path.resolve(strict=True),
        "custom90_sha256": custom_sha, "custom90_prefix_sha256": prefix_sha,
        "native90_pixels_path": pixels, "native90_receipt": native,
        "native90_receipt_sha256": native_sha,
        "history_probe": {"t0_context_mode": args.t0_context,
                          "same_arm_dit_receipt_sha256": digest,
                          "same_arm_dit_bank_sha256": hashlib.sha256(bank).hexdigest(),
                          "same_arm_prefix60_sha256": prefix_sha,
                          "scope": "decoder parity on actual same-arm T0/T1; no identity claim"},
    })
    return admitted


def main(argv=None):
    parser = tail._parser()
    parser.description = __doc__
    parser.add_argument("--t0-context", choices=("repeated-photo", "generated"), required=True)
    parser.add_argument("--dit-receipt", type=Path, required=True)
    parser.add_argument("--dit-receipt-sha256", required=True)
    parser.add_argument("--prefix60", type=Path, required=True)
    parser.add_argument("--prefix60-sha256", required=True)
    args = parser.parse_args(argv)
    try:
        admitted = preflight(args)
        if not args.execute:
            print(json.dumps({"admission": "passed", "model_launched": False,
                              "context_probe": admitted["history_probe"]}, indent=2))
            return 0
        result = tail.execute(admitted, historical_controls=False)
    except (base.ControlError, OSError, RuntimeError, ValueError, KeyError, TypeError) as error:
        print(f"T0 context decoder control failed: {error}", file=sys.stderr)
        return 2
    print(json.dumps({"comparison": result["comparison"], "gates": result["gates"]}, indent=2))
    return 0 if all(result["gates"].values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
