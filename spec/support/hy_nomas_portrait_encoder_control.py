#!/usr/bin/env python3
"""Compare the portrait's native clean T0 against the pinned upstream DCAE-V encoder.

The default path is read-only preflight. ``--execute`` runs BF16 encoder math
on MPS and compares the generated 30-frame latent prefix with the supplied
native T0 on MPS, then performs one packed terminal transfer. Run execution
under an external 18 GiB memory cap with a 10% host-memory reserve; MPSGraph
private workspace is opaque to the named-byte estimate below.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Any


SUPPORT = Path(__file__).resolve().parent
EXAMPLES_SUPPORT = SUPPORT.parents[1] / "examples" / "support"
if str(EXAMPLES_SUPPORT) not in sys.path:
    sys.path.insert(0, str(EXAMPLES_SUPPORT))
if str(SUPPORT) not in sys.path:
    sys.path.insert(0, str(SUPPORT))

import dcae_v_checkpoint as checkpoint  # noqa: E402
import dcae_v_reference as reference  # noqa: E402


DEFAULT_ROOT = Path("/opt/mldata_ssd/cogni-ml/hy-nomas-native-context-20261106.ui7736ue")
DEFAULT_UPSTREAM = Path("/opt/mldata_ssd/models/hy-nomas/upstream-911cea6458a8")
DEFAULT_ENCODER_WEIGHTS = Path("/opt/mldata_ssd/models/hy-nomas/dcae-v-encoder-bf16.safetensors")
DEFAULT_OUTPUT_DIR = DEFAULT_ROOT / "photo-upstream-dcae-t0-v1"
UPSTREAM_REVISION = "911cea6458a89c2663513ff7099561ec190dd491"
DCAE_MODULE_REVISION = "a31c0fe0bcc493be188fd37ddba4ce2f674becb4"
DCAE_MODULE_SHA256 = "38163ebead9a6bb643d27843022c1757c45c6b8638c334a476aee5b373edf8c1"
DCAE_CHECKPOINT_SHA256 = "ebbee483c80890c7795a2babc06e1cb4d42564174cf3bae2b698c0589d95cc3b"
RECOVERED_ENCODER_SHA256 = "d0bbacbebe8960b147388d0bf61ec8bbc0eb8d575574d6ec5c5abe4878c8d6c1"
NATIVE_RECEIPT_SHA256 = "ca8be135ef659497594dec9647c72240f911be8cfd6be989f0740fbf76cd66f5"
REFERENCE_MANIFEST_SHA256 = "04dd1a12f22d41336499b1d3d9d83d2bc92313900cd6116d3565392da04e6628"
REFERENCE_IMAGE_SHA256 = "d38704c64cffcb83e1c43218964cff8824d63a3a3bca9e77dcc88369a0756ed9"
REFERENCE_PIXELS_SHA256 = "c5d6ea9b0ade1f91946844f22f36b764231bccb37098f5cb61ba543fe0e884b2"
ENCODER_TENSOR_COUNT = 169
ENCODER_BF16_BYTES = 1_833_310_016
REFERENCE_PIXEL_BYTES = 442_368
REFERENCE_INPUT_SHAPE = (1, 3, 120, 288, 512)
REFERENCE_LATENT_SHAPE = (1, 32, 30, 9, 16)
LATENT_BANK_SHAPE = (1, 32, 180, 9, 16)
ATOL = RTOL = 2e-5
GIB = 1024**3
EXTERNAL_CAP_BYTES = 18 * GIB
EXTERNAL_RESERVE_FRACTION = 0.10
NATIVE_ENCODER_NAMED_PEAK_BYTES = 4_654_862_144


def _json_no_duplicates(raw: bytes, label: str) -> dict[str, Any]:
    def object_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        value: dict[str, Any] = {}
        for key, item in pairs:
            if key in value:
                raise ValueError(f"duplicate key in {label}: {key}")
            value[key] = item
        return value

    try:
        value = json.loads(raw, object_pairs_hook=object_pairs)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid JSON in {label}: {exc}") from None
    if type(value) is not dict:
        raise ValueError(f"{label} root must be an object")
    return value


def _required_file(path: Path, label: str) -> Path:
    path = Path(path).expanduser()
    reference._identity(path)
    return path.resolve(strict=True)


def _git_head(root: Path) -> str:
    result = subprocess.run(
        ["git", "-C", str(root), "rev-parse", "HEAD"],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    return result.stdout.strip()


def _memory_estimate() -> dict[str, Any]:
    # The native receipt reports named MPS residency. Add the simultaneously
    # staged CPU safetensors state; private MPSGraph workspace stays excluded.
    named_bytes = NATIVE_ENCODER_NAMED_PEAK_BYTES + ENCODER_BF16_BYTES
    return {
        "named_estimate_bytes": named_bytes,
        "named_estimate_gib": named_bytes / GIB,
        "native_named_peak_bytes": NATIVE_ENCODER_NAMED_PEAK_BYTES,
        "cpu_encoder_staging_bytes": ENCODER_BF16_BYTES,
        "external_cap_bytes": EXTERNAL_CAP_BYTES,
        "external_reserve_fraction": EXTERNAL_RESERVE_FRACTION,
        "external_reserve_basis": "physical memory; byte estimate omitted",
        "opaque_mpsgraph_workspace_included": False,
    }


def admit(args: argparse.Namespace) -> dict[str, Any]:
    root = args.input_root.resolve(strict=True)
    reference_manifest_path = _required_file(root / "photo-reference/manifest.json", "reference manifest")
    if reference.sha256_file(reference_manifest_path) != REFERENCE_MANIFEST_SHA256:
        raise ValueError("reference image manifest SHA-256 differs from the native portrait receipt")
    reference_manifest = _json_no_duplicates(reference_manifest_path.read_bytes(), "reference manifest")
    expected_reference = {
        "schema": "cogni_ml.hy_nomas_reference_image/v1",
        "schema_version": 1,
        "upstream_revision": DCAE_MODULE_REVISION,
        "checkpoint_sha256": DCAE_CHECKPOINT_SHA256,
        "encoder_safetensors_sha256": RECOVERED_ENCODER_SHA256,
        "image_sha256": REFERENCE_IMAGE_SHA256,
        "payload": "pixels.rgb8",
        "payload_bytes": REFERENCE_PIXEL_BYTES,
        "payload_sha256": REFERENCE_PIXELS_SHA256,
        "width": 512,
        "height": 288,
        "frames": 120,
        "fit_policy": "exif-rgb-lanczos-pad-black-center",
    }
    if any(reference_manifest.get(key) != value for key, value in expected_reference.items()):
        raise ValueError("reference manifest differs from the admitted same-photo contract")

    pixels_path = _required_file(root / "photo-reference/pixels.rgb8", "cached RGB8 pixels")
    if (pixels_path.stat().st_size != REFERENCE_PIXEL_BYTES
            or reference.sha256_file(pixels_path) != REFERENCE_PIXELS_SHA256):
        raise ValueError("cached RGB8 payload changed")

    receipt_path = _required_file(root / "photo-native-v2/receipt.json", "native receipt")
    if reference.sha256_file(receipt_path) != NATIVE_RECEIPT_SHA256:
        raise ValueError("native receipt SHA-256 differs from the admitted portrait")
    native_receipt = _json_no_duplicates(receipt_path.read_bytes(), "native receipt")
    if (native_receipt.get("reference_manifest_sha256") != REFERENCE_MANIFEST_SHA256
            or native_receipt.get("reference_image_sha256") != REFERENCE_IMAGE_SHA256
            or native_receipt.get("reference_payload_sha256") != REFERENCE_PIXELS_SHA256
            or native_receipt.get("reference_encoder_sha256") != RECOVERED_ENCODER_SHA256
            or native_receipt.get("technical_reference_latent_frames") != 30
            or native_receipt.get("reference_initialization") is not True):
        raise ValueError("native receipt does not bind the supplied photo/T0 encoder contract")

    bank_path = _required_file(root / "photo-native-v2/latent-bank.f32le", "native latent bank")
    bank_sha256 = native_receipt.get("diagnostics", {}).get("latent_bank_sha256")
    expected_bank_bytes = math.prod(LATENT_BANK_SHAPE) * 4
    if (bank_path.stat().st_size != expected_bank_bytes
            or reference.sha256_file(bank_path) != bank_sha256):
        raise ValueError("native latent bank size or receipt hash mismatch")

    upstream = args.upstream.resolve(strict=True)
    head = _git_head(upstream)
    if head != UPSTREAM_REVISION:
        raise ValueError(f"upstream checkout revision mismatch: {head} != {UPSTREAM_REVISION}")
    module_path = _required_file(upstream / "hymm/models/vae/autoencoder_dcae_v.py", "upstream DCAE module")
    module_sha256 = reference.sha256_file(module_path)
    if module_sha256 != DCAE_MODULE_SHA256:
        raise ValueError(f"upstream DCAE module SHA-256 mismatch: {module_sha256}")

    weights_path = _required_file(args.encoder_weights, "recovered encoder safetensors")
    weights_sha256 = reference.sha256_file(weights_path)
    if weights_sha256 != RECOVERED_ENCODER_SHA256:
        raise ValueError("recovered encoder safetensors SHA-256 mismatch")
    expected_shapes, upstream_metadata = checkpoint.upstream_encoder_shapes(module_path)
    total_shape_bytes = sum(math.prod(shape) * 2 for shape in expected_shapes.values())
    if len(expected_shapes) != ENCODER_TENSOR_COUNT or total_shape_bytes != ENCODER_BF16_BYTES:
        raise ValueError("upstream encoder tensor schema differs from the 169-tensor BF16 asset")

    output = args.output_dir
    if not output.is_absolute() or output.exists() or output.is_symlink():
        raise ValueError("output-dir must be a fresh absolute path")
    output_parent = output.parent.resolve(strict=True)
    if not output_parent.is_dir():
        raise ValueError("output-dir parent must be a directory")
    output = output_parent / output.name
    return {
        "root": root,
        "reference_manifest_path": reference_manifest_path,
        "reference_manifest": reference_manifest,
        "pixels_path": pixels_path,
        "receipt_path": receipt_path,
        "native_receipt": native_receipt,
        "bank_path": bank_path,
        "bank_sha256": bank_sha256,
        "upstream_path": upstream,
        "module_path": module_path,
        "module_sha256": module_sha256,
        "upstream_metadata": upstream_metadata,
        "expected_shapes": expected_shapes,
        "weights_path": weights_path,
        "weights_sha256": weights_sha256,
        "producer_sha256": reference.sha256_file(Path(__file__)),
        "output_dir": output,
        "memory_estimate": _memory_estimate(),
    }


def _read_encoder_weights(admitted: dict[str, Any], torch: Any) -> dict[str, Any]:
    try:
        from safetensors import safe_open
    except ImportError as exc:
        raise ValueError(f"safetensors is required to load the recovered encoder: {exc}") from None
    try:
        with safe_open(str(admitted["weights_path"]), framework="pt", device="cpu") as archive:
            keys = archive.keys()
            if set(keys) != set(admitted["expected_shapes"]):
                raise ValueError("recovered encoder tensor names differ from upstream schema")
            state = {name: archive.get_tensor(name) for name in keys}
    except Exception as exc:
        if isinstance(exc, ValueError):
            raise
        raise ValueError(f"cannot read recovered encoder safetensors: {exc}") from None
    return checkpoint.validate_encoder_state(state, admitted["expected_shapes"], torch)


def _stats_on_device(torch: Any, candidate: Any, native: Any) -> Any:
    delta = (candidate - native).abs()
    limit = ATOL + RTOL * native.abs()
    unequal = candidate.contiguous().view(torch.int32) != native.contiguous().view(torch.int32)
    return torch.stack((
        delta.max(),
        delta.square().mean().sqrt(),
        (delta > limit).sum().to(torch.float32),
        unequal.sum().to(torch.float32),
        unequal.any().logical_not().to(torch.float32),
    ))


def execute(admitted: dict[str, Any]) -> dict[str, Any]:
    import numpy as np
    import torch

    if torch.__version__ != "2.8.0" or not torch.backends.mps.is_available():
        raise ValueError("execution requires the admitted Torch 2.8.0 MPS runtime")
    if os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK") != "0":
        raise ValueError("PYTORCH_ENABLE_MPS_FALLBACK must be exactly '0'")

    # CPU state loading is hash/schema checked before only the encoder is
    # materialized on MPS. The decoder stays meta and is never invoked.
    upstream = reference._load_module(admitted["module_path"])
    config = upstream.dc_ae_v_f32t4c32_config()
    with torch.device("meta"):
        model = upstream.DCAEV(config)
    model.encoder.to(dtype=torch.bfloat16)
    model.encoder.to_empty(device="mps")
    encoder_state = _read_encoder_weights(admitted, torch)
    local_names = {name.removeprefix("encoder."): value for name, value in encoder_state.items()}
    model.encoder.load_state_dict(local_names, strict=True)
    del local_names, encoder_state
    model.eval()

    pixels = np.fromfile(admitted["pixels_path"], dtype=np.uint8)
    if pixels.size != REFERENCE_PIXEL_BYTES:
        raise ValueError("cached RGB8 payload extent changed during execution")
    pixels = torch.from_numpy(pixels.reshape(288, 512, 3).copy()).permute(2, 0, 1)
    pixels = pixels.to(device="mps", dtype=torch.bfloat16).div_(127.5).sub_(1)
    clip = pixels[None, :, None].repeat(1, 1, 120, 1, 1).contiguous()
    del pixels
    if tuple(clip.shape) != REFERENCE_INPUT_SHAPE or clip.dtype is not torch.bfloat16:
        raise ValueError("prepared upstream clip differs from the native reference input contract")

    native_np = np.fromfile(admitted["bank_path"], dtype="<f4").reshape(LATENT_BANK_SHAPE)
    native_np = np.ascontiguousarray(native_np[:, :, :30, :, :])
    native = torch.from_numpy(native_np).to(device="mps", dtype=torch.float32)
    started = time.monotonic()
    with torch.no_grad():
        latent = model.encode(clip)
    enqueue_seconds = time.monotonic() - started
    if tuple(latent.shape) != REFERENCE_LATENT_SHAPE or latent.dtype is not torch.bfloat16:
        raise ValueError(f"upstream latent mismatch: shape={tuple(latent.shape)} dtype={latent.dtype}")

    candidate = latent.float()
    stats = _stats_on_device(torch, candidate, native)
    # One packed MPS->CPU transfer contains the F32-expanded latent and all
    # comparison scalars. There is no intermediate activation or scalar readback.
    packed = torch.cat((candidate.reshape(-1), stats)).contiguous().cpu().numpy()
    through_terminal_seconds = time.monotonic() - started
    values = packed[: math.prod(REFERENCE_LATENT_SHAPE)].astype("<f4", copy=False)
    metric_values = packed[math.prod(REFERENCE_LATENT_SHAPE):]
    if not np.isfinite(values).all() or not np.isfinite(metric_values).all():
        raise ValueError("upstream T0 output or comparison metrics are non-finite")

    output = admitted["output_dir"]
    output.mkdir()
    latent_path = output / "latent.f32le"
    latent_bytes = values.tobytes(order="C")
    with latent_path.open("xb") as stream:
        stream.write(latent_bytes)
    stats = {
        "max_abs": float(metric_values[0]),
        "rmse": float(metric_values[1]),
        "outside_tolerance": int(metric_values[2]),
        "unequal_f32_values": int(metric_values[3]),
        "bitwise_equal": bool(metric_values[4]),
        "values": math.prod(REFERENCE_LATENT_SHAPE),
        "atol": ATOL,
        "rtol": RTOL,
    }
    receipt = {
        "schema": "cogni_ml.hy_nomas_portrait_dcae_control/v1",
        "upstream_revision": UPSTREAM_REVISION,
        "dcae_module_history_revision": DCAE_MODULE_REVISION,
        "dcae_module_sha256": admitted["module_sha256"],
        "recovered_encoder_sha256": admitted["weights_sha256"],
        "reference_manifest_sha256": REFERENCE_MANIFEST_SHA256,
        "reference_payload_sha256": REFERENCE_PIXELS_SHA256,
        "native_receipt_sha256": NATIVE_RECEIPT_SHA256,
        "native_latent_bank_sha256": admitted["bank_sha256"],
        "producer_sha256": admitted["producer_sha256"],
        "runtime": {"torch": torch.__version__, "device": "mps", "dtype": "BF16",
                    "mps_fallback": False, "autocast": False},
        "input": {"shape": list(REFERENCE_INPUT_SHAPE), "dtype": "BF16",
                  "normalization": "RGB8 BF16 pixels: div_(127.5), then sub_(1); identical frame repeated 120 times",
                  "tile_frames": 40, "temporal_feature_cache": True},
        "latent": {"shape": list(REFERENCE_LATENT_SHAPE), "dtype": "BF16",
                   "export_dtype": "F32", "path": str(latent_path),
                   "sha256": hashlib.sha256(latent_bytes).hexdigest()},
        "comparison": stats,
        "seconds_encoder_enqueue": enqueue_seconds,
        "seconds_through_terminal": through_terminal_seconds,
        "intermediate_readbacks": 0,
        "explicit_mps_synchronize_calls": 0,
        "terminal_transfer_count": 1,
        "memory_estimate": admitted["memory_estimate"],
        "external_guard": {"cap_bytes": EXTERNAL_CAP_BYTES,
                           "reserve_fraction": EXTERNAL_RESERVE_FRACTION,
                           "required_external": True,
                           "enforcement_verified_by_this_process": False},
    }
    receipt_path = output / "receipt.json"
    receipt_path.write_text(json.dumps(receipt, indent=2, allow_nan=False) + "\n")
    return receipt


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--upstream", type=Path, default=DEFAULT_UPSTREAM)
    parser.add_argument("--encoder-weights", type=Path, default=DEFAULT_ENCODER_WEIGHTS,
                        help="recovered 169-tensor BF16 safetensors (SHA-256 is pinned)")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--execute", action="store_true",
                        help="run the encoder; requires external 18 GiB cap and 10%% reserve")
    args = parser.parse_args()
    try:
        admitted = admit(args)
        if not args.execute:
            result = {
                "validated": True,
                "model_launched": False,
                "upstream_revision": UPSTREAM_REVISION,
                "dcae_module_sha256": admitted["module_sha256"],
                "encoder_sha256": admitted["weights_sha256"],
                "reference_payload_sha256": REFERENCE_PIXELS_SHA256,
                "native_latent_bank_sha256": admitted["bank_sha256"],
                "memory_estimate": admitted["memory_estimate"],
                "output_dir": str(admitted["output_dir"]),
            }
        else:
            result = execute(admitted)
        print(json.dumps(result, sort_keys=True, allow_nan=False))
        return 0
    except (OSError, ValueError, subprocess.CalledProcessError) as exc:
        parser.error(str(exc))
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
