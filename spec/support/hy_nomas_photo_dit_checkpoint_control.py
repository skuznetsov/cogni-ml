#!/usr/bin/env python3
"""GPU replay of authenticated portrait DiT checkpoints, with one terminal pack.

No model is loaded here. Historical upstream BF16 velocities/Euler states are
replayed against fresh native captures. The clean estimate is x_sigma-sigma*v,
computed in F32 then rounded to BF16, rather than a noisy post-Euler state.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import struct
import sys

import hy_nomas_photo_pixelumm_control as photo
base = photo.base

SHAPE = (1, 32, 30, 9, 16)
COUNT = math.prod(SHAPE)
SIGMAS = (1.0, 0.9375, 0.8333333134651184, 0.625)
DELTAS = (-0.0625, -0.10416668653488159, -0.2083333134651184, -0.625)
CAPTURE_SHA = "5ebe729905d69a52590be9bd8029be5731f738bb36fed956951f10bc340555a3"
SOURCE_RECEIPT_SHA = "55658d3937b971e0c1e8ef365757a994d1fa0a23febc0de533ffc9f702e064e5"
CAPTURE_CONTRACT = {
    "source": "original-photo reference T0, source-BF16 native T1",
    "capture_chain": "same DiT command chain; one terminal wait",
    "capture_values": "BF16-effective velocity, BF16-rounded x0 estimate, BF16-after-Euler latent",
}


def validate_native_capture(native):
    expected = {
        "schema": "cogni_ml.hy_nomas_portrait_t0_history_probe/v1",
        "reference_t1_step_capture": True,
        "reference_t1_step_capture_contract": CAPTURE_CONTRACT,
        "reference_t1_step_capture_buffers_bytes": 12 * COUNT * 4,
        "inputs_sha256": base.INPUT_MANIFEST_SHA256,
        "weight_index_sha256": "a235ca592b47161472c7fd94cd3d886fce164480cd0571179c8860002063cbb0",
        "baseline_receipt_sha256": base.NATIVE_RECEIPT_SHA256,
        "baseline_latent_bank_sha256": base.LATENT_BANK_SHA256,
        "t0_context_mode": "repeated-photo", "upstream_revision": base.UPSTREAM_REVISION,
        "source_bf16": True, "keep_t0_video_latents": 30,
        "t0_hide_caption_keys": False, "t1_current_text_kv": False,
        "t0_reference_initialization": True, "generated_t0_control": False,
        "segments": 2, "steps_per_segment": 4,
    }
    if any(type(native.get(k)) is not type(v) or native.get(k) != v for k, v in expected.items()):
        raise base.ControlError("native capture contract or original-photo source lineage mismatch")
    for key, value in {"chain_preterminal_allocations": 0, "chain_preterminal_readbacks": 0,
                       "chain_preterminal_waits": 0, "dit_terminal_waits": 1,
                       "reference_t1_captured_steps": 4, "t0_denoise_steps": 0,
                       "t0_cache_writer_forwards": 1}.items():
        actual = native.get("metrics", {}).get(key)
        if type(actual) is not int or actual != value:
            raise base.ControlError("native capture violates chain boundary or capture count")


def bank_segment(bank: bytes, start: int) -> bytes:
    if len(bank) != 32 * 180 * 9 * 16 * 4 or start not in (0, 30):
        raise base.ControlError("native capture bank extent/segment mismatch")
    frame_bytes = 9 * 16 * 4
    return b"".join(bank[(channel * 180 + start) * frame_bytes:
                             (channel * 180 + start + 30) * frame_bytes]
                    for channel in range(32))


def validate_steps(records):
    if not isinstance(records, list) or len(records) != 4:
        raise base.ControlError("four original-photo T1 steps are required")
    for index, record in enumerate(records):
        if type(record.get("step")) is not int or record["step"] != index + 1:
            raise base.ControlError("native sigma/dt or step order mismatch")
        # Crystal emits shortest-roundtrip F32 decimal strings. Compare IEEE
        # F32 bits, not their F64 JSON parses; this introduces no tolerance.
        for key, expected in (("sigma_f32", SIGMAS[index]), ("dt_f32", DELTAS[index])):
            value = record.get(key)
            try:
                matches = (type(value) is float and math.isfinite(value)
                           and struct.pack("<f", value) == struct.pack("<f", expected))
            except (OverflowError, struct.error):
                matches = False
            if not matches:
                raise base.ControlError("native sigma/dt or step order mismatch")
        for kind in ("velocity_bf16", "x0_bf16", "after_euler_bf16"):
            item = record.get(kind, {})
            if (item.get("shape") != list(SHAPE) or item.get("dtype") != "BF16"
                    or item.get("encoding") != "f32le_bf16_expansion"):
                raise base.ControlError("native checkpoint representation mismatch")


def preflight(args):
    root = args.run_root.resolve(strict=True)
    source, source_raw = base._read_json(root / "photo-upstream-dit-v1/receipt.json", "upstream DiT receipt")
    base._require_digest(hashlib.sha256(source_raw).hexdigest(), SOURCE_RECEIPT_SHA, "upstream DiT receipt")
    if (source.get("upstream_revision") != base.UPSTREAM_REVISION
            or source.get("conditioning_manifest_sha256") != base.INPUT_MANIFEST_SHA256
            or source.get("schedule", {}).get("sigmas_f32") != list(SIGMAS) + [0.0]
            or source.get("schedule", {}).get("dt_f32") != list(DELTAS)):
        raise base.ControlError("historical upstream conditioning/schedule mismatch")
    captures = root / "photo-upstream-dit-v1/captures.f32le"
    base._require_digest(base._sha256_file(captures), CAPTURE_SHA, "historical upstream captures")
    if source["capture_payload"]["sha256"] != CAPTURE_SHA:
        raise base.ControlError("source capture receipt digest mismatch")
    source_records = {r["name"]: r for r in source["capture_payload"]["records"]}
    for step in range(1, 5):
        for kind in ("velocity", "after_euler"):
            record = source_records[f"t1.step{step}.{kind}"]
            if record["shape"] != list(SHAPE) or record["nbytes"] != COUNT * 4:
                raise base.ControlError("source checkpoint extent mismatch")
            base._require_digest(base._sha256_range(captures, record["offset"], record["nbytes"]),
                                 record["sha256"], "source checkpoint")
    native, raw = base._read_json(args.native_receipt, "native step-capture receipt")
    native_sha = hashlib.sha256(raw).hexdigest()
    base._require_digest(native_sha, args.native_receipt_sha256, "native step-capture receipt")
    validate_native_capture(native)
    bank = base._read_regular(Path(native["output"]["latent_bank"]), "native terminal bank",
                              max_bytes=32 * 180 * 9 * 16 * 4)
    base._require_digest(hashlib.sha256(bank).hexdigest(), native["output"]["latent_bank_sha256"], "native terminal bank")
    base._require_digest(hashlib.sha256(bank_segment(bank, 0)).hexdigest(), photo.T0_SHA, "native original-photo T0")
    base._require_digest(hashlib.sha256(bank_segment(bank, 30)).hexdigest(), photo.FINAL_SHA, "native accepted final endpoint")
    records = native.get("reference_t1_step_captures")
    validate_steps(records)
    for record in records:
        for kind in ("velocity_bf16", "x0_bf16", "after_euler_bf16"):
            item = record[kind]
            raw = base._read_regular(Path(item["path"]), "native step tensor", max_bytes=COUNT * 4)
            if len(raw) != COUNT * 4:
                raise base.ControlError("native step tensor extent mismatch")
            base._require_digest(hashlib.sha256(raw).hexdigest(), item["sha256"], "native step tensor")
    base._require_digest(records[-1]["x0_bf16"]["sha256"], photo.FINAL_SHA, "step-four native x0 endpoint")
    t0 = root / "photo-upstream-dcae-t0-v1/latent.f32le"
    base._require_digest(base._sha256_file(t0), photo.T0_SHA, "source-derived original photo T0")
    manifest, raw = base._read_json(root / "photo-continuation-v2.json", "noise manifest")
    base._require_digest(hashlib.sha256(raw).hexdigest(), base.INPUT_MANIFEST_SHA256, "noise manifest")
    payload = root / manifest["payload"]
    base._require_digest(base._sha256_file(payload), base.INPUT_PAYLOAD_SHA256, "noise payload")
    noise_record = manifest["tensors"]["input.latent_bank"]
    if noise_record["shape"] != [1, 32, 180, 9, 16] or noise_record["dtype"] != "F32":
        raise base.ControlError("initial DiT noise geometry mismatch")
    output = base._fresh_output_dir(args.output_dir)
    return dict(root=root, source=source, source_records=source_records, captures=captures,
                native=native, native_sha=native_sha, records=records, t0=t0,
                payload=payload, noise_record=noise_record, output=output)


def execute(admitted):
    import torch
    if (torch.__version__ != "2.8.0" or not torch.backends.mps.is_available()
            or os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK") != "0"):
        raise base.ControlError("checkpoint replay requires Torch2.8 MPS with fallback disabled")
    device = torch.device("mps")

    def load(path, offset=0, count=COUNT, shape=SHAPE):
        size = path.stat().st_size // 4
        return torch.from_file(str(path), shared=False, size=size, dtype=torch.float32).narrow(
            0, offset // 4, count).view(shape).to(device)

    t0 = load(admitted["t0"])
    record = admitted["noise_record"]
    noise = load(admitted["payload"], record["offset"], record["nbytes"] // 4,
                 (1, 32, 180, 9, 16))
    current = noise[:, :, 30:60].to(torch.bfloat16).float()
    del noise
    outputs, comparisons = [], []
    with torch.inference_mode():
        for index, native_record in enumerate(admitted["records"]):
            values = {}
            for kind in ("velocity", "after_euler"):
                meta = admitted["source_records"][f"t1.step{index+1}.{kind}"]
                values[kind] = load(admitted["captures"], meta["offset"])
            # Separate F32 multiplication/subtraction matches the upstream flow
            # contract. BF16 conversion occurs only after the clean estimate.
            scaled = values["velocity"] * SIGMAS[index]
            estimate = (current - scaled).to(torch.bfloat16).float()
            values["x0"] = estimate
            native_estimate = None
            for kind, source_kind in (("velocity_bf16", "velocity"), ("x0_bf16", "x0"),
                                      ("after_euler_bf16", "after_euler")):
                actual = load(Path(native_record[kind]["path"]))
                reference = values[source_kind]
                delta = (actual - reference).abs()
                comparisons.append(torch.stack((delta.max(), delta.square().mean().sqrt(),
                    (delta > base.ATOL + base.RTOL * reference.abs()).float().sum(),
                    (actual != reference).float().sum(),
                    (~torch.isfinite(actual) | ~torch.isfinite(reference)).float().sum())))
                if source_kind == "x0":
                    native_estimate = actual
            outputs.append((f"source-step{index+1}-x0", estimate))
            outputs.append((f"native-step{index+1}-paired60", torch.cat((t0, native_estimate), dim=2)))
            current = values["after_euler"]
        packed = torch.cat([value.reshape(-1) for _, value in outputs] +
                           [torch.stack(comparisons).reshape(-1)])
        # Sole terminal device-to-host transfer; no model or per-step host read.
        cpu = packed.cpu()
    out = admitted["output"]
    out.mkdir(mode=0o700)
    records, offset = [], 0
    for name, value in outputs:
        count = value.numel()
        path = out / (name + ".f32le")
        sha = base._write_fresh(path, cpu[offset:offset+count].numpy().astype("<f4", copy=False))
        records.append(dict(name=name, path=str(path), sha256=sha, shape=list(value.shape)))
        offset += count
    metrics = cpu[offset:].view(4, 3, 5).tolist()
    final_x0_matches = records[-2]["sha256"] == photo.FINAL_SHA
    result = {"schema": "cogni_ml.hy_nomas_photo_dit_checkpoint_control/v1",
              "scope": "historical upstream checkpoints vs fresh native captures; no upstream DiT rerun",
              "native_receipt_sha256": admitted["native_sha"], "source_capture_sha256": CAPTURE_SHA,
              "sigmas_f32": list(SIGMAS), "dt_f32": list(DELTAS),
              "formula": "BF16(F32(x_sigma)-F32(sigma)*F32(BF16(v)))",
              "comparison_order": ["velocity_bf16", "x0_bf16", "after_euler_bf16"],
              "metric_order": ["max_abs", "rmse", "outside_tolerance", "unequal", "nonfinite"],
              "metrics": metrics, "atol": base.ATOL, "rtol": base.RTOL,
              "native_x0_sha256": [r["x0_bf16"]["sha256"] for r in admitted["records"]],
              "step4_x0_equals_final_endpoint": final_x0_matches,
              "passed": final_x0_matches and all(row[2] == 0 and row[4] == 0 for step in metrics for row in step),
              "artifacts": records, "runtime": {"torch": torch.__version__, "device": "mps",
                  "intermediate_readbacks": 0, "explicit_synchronize_calls": 0, "terminal_transfers": 1}}
    (out / "receipt.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, default=base.DEFAULT_ROOT)
    parser.add_argument("--native-receipt", type=Path, required=True)
    parser.add_argument("--native-receipt-sha256", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args(argv)
    try:
        admitted = preflight(args)
        if not args.execute:
            print(json.dumps({"admitted": True, "model_loaded": False, "GPU_work_started": False}))
            return 0
        result = execute(admitted)
    except (base.ControlError, OSError, RuntimeError, ValueError, KeyError, TypeError) as error:
        print(f"photo checkpoint control failed: {error}", file=sys.stderr)
        return 2
    print(json.dumps({"passed": result["passed"], "metrics": result["metrics"]}, indent=2))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
