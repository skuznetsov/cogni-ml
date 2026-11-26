#!/usr/bin/env python3
"""Compare a photo-derived counterfactual bank through native/source PixelUMM.

The first segment must be the authenticated clean original-photo DCAE latent.
The second is either its exact repeat, a named captured clean estimate, or the
accepted final DiT endpoint. Decoder noise, positions and windows are unchanged.
This proves decoder parity for supplied latents, not DiT checkpoint provenance.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import struct
import sys

import hy_nomas_portrait_decoder_control as base

SCHEMA = "cogni_ml.hy_nomas_photo_pixelumm_control/v1"
T0_SHA = "6730c2618b6534768d626bbbd70119a46e2808bbcce7a7c20e6dd87e8e22e7ed"
FINAL_SHA = "8b21a039c9a18e85b24bd2590de198bf2243ef25aa8acc0c55dd9b515b7e11e3"
FRAME_BYTES = 9 * 16 * 4
SEGMENT_BYTES = 32 * 30 * FRAME_BYTES
DIT_CAPTURE_SHA = "5ebe729905d69a52590be9bd8029be5731f738bb36fed956951f10bc340555a3"


def segments(bank: bytes) -> tuple[bytes, bytes]:
    if len(bank) != 2 * SEGMENT_BYTES:
        raise base.ControlError("paired bank must be F32 [1,32,60,9,16]")
    # All admitted model checkpoints are BF16-expanded F32. A noisy F32 state
    # or an unrounded velocity cannot silently become a clean-estimate input.
    for (word,) in struct.iter_unpack("<I", bank):
        if word & 0xffff or word & 0x7f800000 == 0x7f800000:
            raise base.ControlError("paired bank must contain finite BF16-expanded values")
    result = []
    for offset in (0, 30 * FRAME_BYTES):
        result.append(b"".join(bank[c * 60 * FRAME_BYTES + offset:
                                    c * 60 * FRAME_BYTES + offset + 30 * FRAME_BYTES]
                               for c in range(32)))
    return tuple(result)


def validate_bank(bank: bytes, kind: str, checkpoint_sha: str) -> dict:
    if kind not in ("clean", "step1", "step2", "step3", "step4", "final"):
        raise base.ControlError("unknown clean-estimate kind")
    t0, t1 = segments(bank)
    base._require_digest(hashlib.sha256(t0).hexdigest(), T0_SHA, "original photo T0")
    actual = hashlib.sha256(t1).hexdigest()
    base._require_digest(actual, checkpoint_sha, "supplied second segment")
    if kind == "clean" and t1 != t0:
        raise base.ControlError("clean control must repeat original photo T0 exactly")
    if kind == "final":
        base._require_digest(actual, FINAL_SHA, "accepted original-photo final DiT endpoint")
    return {"kind": kind, "t0_sha256": T0_SHA, "second_segment_sha256": actual,
            "checkpoint_semantics_scope": "supplied BF16-expanded clean estimate; DiT provenance requires the separate capture comparison"}


def validate_replay(receipt: dict, bank_sha: str, pixels_sha: str) -> None:
    expected = {
        "schema": "cogni_ml.hy_nomas_decode_latent/v1",
        "upstream_revision": base.UPSTREAM_REVISION,
        "latent_sha256": bank_sha, "latent_bank_sha256": bank_sha,
        "latent_shape": [1, 32, 60, 9, 16],
        "latent_bank_shape": [1, 32, 60, 9, 16],
        "latent_prefix_start_frame": 0, "pixel_noise_prefix_start_frame": 0,
        "continuation_manifest_sha256": base.INPUT_MANIFEST_SHA256,
        "pixel_noise_manifest_sha256": base.INPUT_MANIFEST_SHA256,
        "pixel_noise_record_sha256": base.NOISE_SHA256,
        "decoder_weight_cache_sha256": base.CACHE_MANIFEST_SHA256,
        "stem_weights_sha256": base.STEM_SHA256, "head_weights_sha256": base.HEAD_SHA256,
        "decoder_rope_cos_record_sha256": base.ROPE_COS_SHA256,
        "decoder_rope_sin_record_sha256": base.ROPE_SIN_SHA256,
        "pixel_noise_shape": list(base.NOISE_PREFIX_SHAPE),
        "pixelumm": {"steps": 1, "window_latents": 10, "overlap_latents": 5,
                     "starts": list(base.WINDOW_STARTS)},
        "output_shape": list(base.PIXELS_SHAPE), "output_sha256": pixels_sha,
    }
    if any(type(receipt.get(k)) is not type(v) or receipt.get(k) != v for k, v in expected.items()):
        raise base.ControlError("native replay differs in bank/noise/positions/weights/output")
    for key, value in {"decoder_encode_allocations": 0, "decoder_encode_readbacks": 0,
                       "decoder_preterminal_waits": 0, "decoder_waits": 1}.items():
        actual = receipt.get("metrics", {}).get(key)
        if type(actual) is not int or actual != value:
            raise base.ControlError("native replay violates preterminal GPU boundary")


def validate_checkpoint_join(receipt: dict, kind: str, bank_sha: str, checkpoint_sha: str) -> dict:
    if (receipt.get("schema") != "cogni_ml.hy_nomas_photo_dit_checkpoint_control/v1"
            or receipt.get("passed") is not True
            or receipt.get("source_capture_sha256") != DIT_CAPTURE_SHA
            or receipt.get("atol") != base.ATOL or receipt.get("rtol") != base.RTOL):
        raise base.ControlError("checkpoint comparison source/gate mismatch")
    step = int(kind.removeprefix("step"))
    native_shas = receipt.get("native_x0_sha256", [])
    if len(native_shas) != 4 or native_shas[step - 1] != checkpoint_sha:
        raise base.ControlError("named DiT step differs from the supplied clean estimate")
    records = [r for r in receipt.get("artifacts", []) if r.get("name") == f"native-step{step}-paired60"]
    if (len(records) != 1 or records[0].get("sha256") != bank_sha
            or records[0].get("shape") != [1, 32, 60, 9, 16]):
        raise base.ControlError("paired bank differs from its named checkpoint artifact")
    return records[0]


def preflight(args):
    args.input_latents = args.input_latents_sha256 = None
    admitted = base.preflight(args)
    bank = base._read_regular(args.paired_latents, "paired photo bank", max_bytes=2 * SEGMENT_BYTES)
    bank_sha = hashlib.sha256(bank).hexdigest()
    base._require_digest(bank_sha, args.paired_latents_sha256, "paired photo bank")
    diagnostic = validate_bank(bank, args.kind, args.checkpoint_sha256)
    if args.kind.startswith("step"):
        if args.checkpoint_control is None or args.checkpoint_control_sha256 is None:
            raise base.ControlError("named DiT step requires an authenticated checkpoint comparison")
        control, control_raw = base._read_json(args.checkpoint_control, "checkpoint comparison")
        control_sha = hashlib.sha256(control_raw).hexdigest()
        base._require_digest(control_sha, args.checkpoint_control_sha256, "checkpoint comparison")
        artifact = validate_checkpoint_join(control, args.kind, bank_sha, args.checkpoint_sha256)
        if Path(artifact["path"]).resolve(strict=True) != args.paired_latents.resolve(strict=True):
            raise base.ControlError("paired bank path differs from checkpoint comparison")
        diagnostic["checkpoint_control_sha256"] = control_sha
    elif args.checkpoint_control is not None or args.checkpoint_control_sha256 is not None:
        raise base.ControlError("checkpoint comparison applies only to named DiT estimates")
    receipt, raw = base._read_json(args.native_receipt, "native paired decoder receipt")
    receipt_sha = hashlib.sha256(raw).hexdigest()
    base._require_digest(receipt_sha, args.native_receipt_sha256, "native paired decoder receipt")
    pixels = Path(receipt["output"])
    if (not pixels.is_absolute() or pixels.is_symlink() or not pixels.is_file()
            or pixels.stat().st_size != 3 * 240 * 288 * 512 * 4):
        raise base.ControlError("native paired pixel path/extent mismatch")
    pixels_sha = base._sha256_file(pixels)
    validate_replay(receipt, bank_sha, pixels_sha)
    if Path(receipt["latent"]).resolve(strict=True) != args.paired_latents.resolve(strict=True):
        raise base.ControlError("native decoder used a different latent path")
    admitted.update({"receipt": receipt, "receipt_sha256": receipt_sha,
                     "bank_path": args.paired_latents, "bank_sha256": bank_sha,
                     "prefix_sha256": bank_sha, "latent_path": args.paired_latents,
                     "latent_sha256": bank_sha, "custom_latents": True,
                     "pixels_path": pixels, "pixels_sha256": pixels_sha,
                     "output_schema": SCHEMA, "photo_diagnostic": diagnostic,
                     "comparison_scope": "same matched second-segment display pixels [120,240); no encoder/DiT rerun or likeness claim"})
    return admitted


def main(argv=None):
    parser = base._parser()
    parser.description = __doc__
    parser.add_argument("--paired-latents", type=Path, required=True)
    parser.add_argument("--paired-latents-sha256", required=True)
    parser.add_argument("--native-receipt", type=Path, required=True)
    parser.add_argument("--native-receipt-sha256", required=True)
    parser.add_argument("--checkpoint-sha256", required=True)
    parser.add_argument("--checkpoint-control", type=Path)
    parser.add_argument("--checkpoint-control-sha256")
    parser.add_argument("--kind", choices=("clean", "step1", "step2", "step3", "step4", "final"), required=True)
    args = parser.parse_args(argv)
    try:
        admitted = preflight(args)
        if not args.execute:
            print(json.dumps({"admission": "passed", "model_launched": False,
                              "photo_diagnostic": admitted["photo_diagnostic"]}, indent=2))
            return 0
        result = base.execute(admitted)
    except (base.ControlError, OSError, ValueError, KeyError, TypeError, RuntimeError) as error:
        print(f"photo PixelUMM control failed: {error}", file=sys.stderr)
        return 2
    print(json.dumps(result["comparison"], indent=2))
    return 0 if result["comparison"]["result"] == "match" else 1


if __name__ == "__main__":
    raise SystemExit(main())
