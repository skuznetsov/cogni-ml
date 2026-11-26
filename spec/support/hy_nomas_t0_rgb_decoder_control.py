#!/usr/bin/env python3
"""Same-latent PixelUMM gate for DCAE moving/repeated-last-frame T0 arms."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys

import hy_nomas_portrait_history_decoder_control as history
import hy_nomas_t0_rgb_artifacts as rgb

tail = history.tail
base = history.base
SCHEMA = "cogni_ml.hy_nomas_t0_rgb_history_probe/v1"
BANK_BYTES = 32 * 180 * 9 * 16 * 4
LATENT_BYTES = 32 * 30 * 9 * 16 * 4


def latent_segment(bank: bytes, start: int) -> bytes:
    if len(bank) != BANK_BYTES or start not in (0, 30):
        raise base.ControlError("RGB history bank or segment extent differs")
    frame_bytes = 9 * 16 * 4
    return b"".join(bank[(channel * 180 + start) * frame_bytes:
                         (channel * 180 + start + 30) * frame_bytes]
                    for channel in range(32))


def validate_dit_prefix(receipt: dict, bank: bytes, prefix: bytes, mode: str,
                        dcae_latent: bytes) -> None:
    required = {
        "schema": SCHEMA, "context_mode": mode,
        "inputs_sha256": base.INPUT_MANIFEST_SHA256,
        "upstream_revision": base.UPSTREAM_REVISION,
        "source_bf16": True, "segments": 2, "steps_per_segment": 4,
    }
    if mode not in rgb.MODES or any(type(receipt.get(k)) is not type(v)
                                   or receipt.get(k) != v for k, v in required.items()):
        raise base.ControlError("RGB history mode, precision or frozen geometry differs")
    if len(bank) != BANK_BYTES:
        raise base.ControlError("RGB history bank extent differs")
    output = receipt.get("output", {})
    if output.get("latent_bank_sha256") != hashlib.sha256(bank).hexdigest():
        raise base.ControlError("RGB history actual bank digest differs")
    if prefix != history.channel_prefix(bank, frames=180, selected=60):
        raise base.ControlError("decoder prefix differs from actual same-arm RGB history bank")
    t0, t1 = latent_segment(bank, 0), latent_segment(bank, 30)
    if len(dcae_latent) != LATENT_BYTES or t0 != dcae_latent:
        raise base.ControlError("clean T0 differs from the actual same-arm DCAE endpoint")
    for key, raw in (("t0_sha256", t0), ("t1_sha256", t1),
                     ("prefix60_sha256", prefix), ("dcae_latent_f32_sha256", dcae_latent)):
        if output.get(key) != hashlib.sha256(raw).hexdigest():
            raise base.ControlError(f"RGB history actual endpoint digest differs: {key}")
    metrics = receipt.get("metrics", {})
    for key, expected in (("chain_preterminal_allocations", 0),
                          ("chain_preterminal_readbacks", 0),
                          ("chain_preterminal_waits", 0), ("terminal_waits", 1)):
        if type(metrics.get(key)) is not int or metrics.get(key) != expected:
            raise base.ControlError("RGB history preterminal or terminal counter differs")


def _artifact(path_arg, digest, label: str, maximum: int) -> tuple[Path, bytes]:
    path = Path(path_arg)
    if not path.is_absolute():
        raise base.ControlError(f"{label} must use an absolute path")
    raw = base._read_regular(path, label, max_bytes=maximum)
    base._require_digest(hashlib.sha256(raw).hexdigest(), digest, label)
    return path.resolve(strict=True), raw


def preflight(args):
    args.input_latents = args.input_latents_sha256 = None
    # The old receipt binds shared weights/noise/RoPE only. Its photo T0 is
    # never substituted for either new DCAE endpoint.
    admitted = base.preflight(args)
    receipt_path, raw = _artifact(args.dit_receipt, args.dit_receipt_sha256,
                                  "same-arm RGB DiT receipt", 1024 * 1024)
    receipt = json.loads(raw, object_pairs_hook=base._unique_object)
    output = receipt["output"]
    bank_path, bank = _artifact(output["latent_bank"], output["latent_bank_sha256"],
                                "same-arm RGB DiT bank", BANK_BYTES)
    prefix_path, prefix = _artifact(args.prefix60, args.prefix60_sha256,
                                    "same-arm prefix60", 32 * 60 * 9 * 16 * 4)
    dcae_path, dcae = _artifact(output["dcae_latent_f32"], output["dcae_latent_f32_sha256"],
                                "same-arm DCAE endpoint", LATENT_BYTES)
    validate_dit_prefix(receipt, bank, prefix, args.context_mode, dcae)

    source = rgb.validate_source(args.source_pixels)
    clip_path = Path(output["normalized_clip_bf16"])
    clip = rgb.validate_clip_bytes(source, clip_path, args.context_mode)
    if clip["sha256"] != output["normalized_clip_bf16_sha256"]:
        raise base.ControlError("actual normalized clip differs from its DiT receipt")
    del source

    custom_path, custom = _artifact(args.custom_latents, args.custom_latents_sha256,
                                    "same-arm cyclic90", tail.LATENT90_BYTES)
    tail.validate_latent_artifact_bytes(custom, args.custom_latents_sha256)
    tail.validate_cyclic_tail_bytes(custom, channels=32, frames=90, frame_bytes=9 * 16 * 4,
                                    prefix=prefix, prefix_frames=60, cycle_start=50,
                                    cycle_frames=10, repeats=3)
    pixels, native, native_sha = tail._read_native90(args, custom_path, args.custom_latents_sha256)
    history.validate_native_noise(native)
    admitted.update({
        "custom90_path": custom_path, "custom90_sha256": args.custom_latents_sha256,
        "custom90_prefix_sha256": args.prefix60_sha256,
        "native90_pixels_path": pixels, "native90_receipt": native,
        "native90_receipt_sha256": native_sha,
        "history_probe": {"context_mode": args.context_mode,
                          "same_arm_dit_receipt": str(receipt_path),
                          "same_arm_dit_receipt_sha256": args.dit_receipt_sha256,
                          "same_arm_dit_bank": str(bank_path),
                          "same_arm_dit_bank_sha256": output["latent_bank_sha256"],
                          "same_arm_prefix60": str(prefix_path),
                          "same_arm_prefix60_sha256": args.prefix60_sha256,
                          "same_arm_dcae_endpoint": str(dcae_path),
                          "normalized_input_check": clip,
                          "scope": "same-arm decoder parity; motion and likeness are separate"},
    })
    return admitted


def main(argv=None):
    parser = tail._parser()
    parser.description = __doc__
    parser.add_argument("--context-mode", choices=rgb.MODES, required=True)
    parser.add_argument("--source-pixels", type=Path, required=True)
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
        print(f"RGB history decoder control failed: {error}", file=sys.stderr)
        return 2
    print(json.dumps({"comparison": result["comparison"], "gates": result["gates"]}, indent=2))
    return 0 if all(result["gates"].values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
