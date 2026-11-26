#!/usr/bin/env python3
"""Run a pinned RGB-history DCAE T0 through the frozen portrait writer and T1."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import re
import subprocess
import sys
from types import SimpleNamespace
from typing import Any

SUPPORT = Path(__file__).resolve().parent
EXAMPLES_SUPPORT = SUPPORT.parents[1] / "examples" / "support"
for path in (SUPPORT, EXAMPLES_SUPPORT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import dcae_v_checkpoint as dcae_checkpoint  # noqa: E402
import dcae_v_reference as dcae_reference  # noqa: E402
import hy_nomas_portrait_dit_control as portrait  # noqa: E402
import hy_nomas_portrait_encoder_control as encoder_control  # noqa: E402
import hy_nomas_t0_rgb_artifacts as rgb_artifacts  # noqa: E402

DEFAULT_ROOT = portrait.DEFAULT_ROOT
DEFAULT_UPSTREAM = Path("/opt/mldata_ssd/models/hy-nomas/upstream-911cea6458a8")
DEFAULT_ENCODER_WEIGHTS = Path("/opt/mldata_ssd/models/hy-nomas/dcae-v-encoder-bf16.safetensors")
DEFAULT_RGB_DIR = DEFAULT_ROOT / "photo-generated-t0-v1/native-decoder-generated-v1"
DECODER_RECEIPT_SHA256 = "7e5a3ee991fdb3bbcb6604ef0008ae2254e96f8161d4c81053df77e9b0d7af07"
UPSTREAM_REVISION = "911cea6458a89c2663513ff7099561ec190dd491"
NATIVE_SCHEMA = "cogni_ml.hy_nomas_t0_rgb_history_probe/v1"
LATENT_BANK_SHAPE = (1, 32, 180, 9, 16)
LATENT_SEGMENT_SHAPE = (1, 32, 30, 9, 16)
LATENT_BANK_BYTES = math.prod(LATENT_BANK_SHAPE) * 4
LATENT_SEGMENT_BYTES = math.prod(LATENT_SEGMENT_SHAPE) * 4
CLIP_ELEMENTS = math.prod(rgb_artifacts.CLIP_SHAPE)
CLIP_BYTES = rgb_artifacts.CLIP_BYTES
SOURCE_SHAPE = rgb_artifacts.SOURCE_SHAPE
SOURCE_BYTES = rgb_artifacts.SOURCE_BYTES
SOURCE_SHA256 = rgb_artifacts.SOURCE_SHA256
GIB = 1024 ** 3
EXTERNAL_CAP_BYTES = 18 * GIB
EXTERNAL_RESERVE_FRACTION = 0.10
HEX = re.compile(r"^[0-9a-f]{64}$")


def validate_context_mode(mode: str) -> str:
    if mode not in rgb_artifacts.MODES:
        raise ValueError("--context-mode must be moving or repeat-last")
    return mode


def validate_decoder_receipt_document(receipt: dict[str, Any]) -> dict[str, Any]:
    expected = {
        "schema": "cogni_ml.hy_nomas_decode_latent/v1",
        "upstream_revision": UPSTREAM_REVISION,
        "output_bytes": rgb_artifacts.SOURCE_BYTES,
        "output_sha256": rgb_artifacts.SOURCE_SHA256,
        "output_dtype": "F32 little-endian planar RGB",
        "output_shape": list(rgb_artifacts.SOURCE_SHAPE),
        "frames": 360,
        "width": 512,
        "height": 288,
        "full_upstream_video_parity": False,
    }
    if not isinstance(receipt, dict) or any(receipt.get(key) != value for key, value in expected.items()):
        raise ValueError("decoder receipt differs from the frozen generated-pixel artifact")
    return {
        "schema": receipt["schema"],
        "upstream_revision": receipt["upstream_revision"],
        "pixels_sha256": receipt["output_sha256"],
        "pixels_bytes": receipt["output_bytes"],
        "shape": receipt["output_shape"],
        "dtype": receipt["output_dtype"],
        "full_upstream_video_parity": False,
    }


def _json_file(path: Path, label: str) -> dict[str, Any]:
    try:
        value = portrait.source.attention_reference._json_no_duplicates(
            Path(path).read_bytes(), label
        )
    except (OSError, ValueError) as error:
        raise ValueError(f"cannot read {label}: {error}") from None
    if type(value) is not dict:
        raise ValueError(f"{label} root must be an object")
    return value


def _regular_file(path: Path, label: str) -> Path:
    path = Path(path).expanduser()
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"{label} must be a regular file")
    return path.resolve(strict=True)


def _sha256(path: Path) -> str:
    return portrait.source._sha256(Path(path))


def validate_native_receipt_document(
    receipt: dict[str, Any],
    *,
    mode: str,
    frozen_input_identity: dict[str, str],
    native_t1_sha256: str,
) -> dict[str, Any]:
    mode = validate_context_mode(mode)
    if not isinstance(receipt, dict) or receipt.get("schema") != NATIVE_SCHEMA:
        raise ValueError("native receipt schema differs from the RGB history probe")
    if receipt.get("context_mode") != mode:
        raise ValueError("native receipt context-mode differs from the source arm")
    if receipt.get("frozen_input_identity") != frozen_input_identity:
        raise ValueError("native receipt frozen input identity differs")
    if not isinstance(native_t1_sha256, str) or HEX.fullmatch(native_t1_sha256) is None:
        raise ValueError("native T1 digest must be lowercase SHA-256")
    metrics = receipt.get("metrics")
    if not isinstance(metrics, dict) or any(
        type(metrics.get(key)) is not int or metrics.get(key) != 0
        for key in (
            "chain_preterminal_allocations",
            "chain_preterminal_readbacks",
            "chain_preterminal_waits",
        )
    ) or type(metrics.get("terminal_waits")) is not int or metrics.get("terminal_waits") != 1:
        raise ValueError("native terminal chain metrics differ")
    output = receipt.get("output")
    if not isinstance(output, dict):
        raise ValueError("native receipt output record is missing")
    required = (
        "latent_bank", "latent_bank_sha256", "t0_sha256", "t1_sha256",
        "prefix60_sha256", "normalized_clip_bf16", "normalized_clip_bf16_sha256",
        "dcae_latent_f32", "dcae_latent_f32_sha256",
    )
    if any(key not in output for key in required):
        raise ValueError("native receipt output record is incomplete")
    for key in (
        "latent_bank_sha256", "t0_sha256", "t1_sha256", "prefix60_sha256",
        "normalized_clip_bf16_sha256", "dcae_latent_f32_sha256",
    ):
        if not isinstance(output[key], str) or HEX.fullmatch(output[key]) is None:
            raise ValueError(f"native receipt output digest is invalid: {key}")
    for key in ("latent_bank", "normalized_clip_bf16", "dcae_latent_f32"):
        if not isinstance(output[key], str) or not output[key]:
            raise ValueError(f"native receipt output path is invalid: {key}")
    if output["dcae_latent_f32_sha256"] != output["t0_sha256"]:
        raise ValueError("native DCAE endpoint digest differs from the T0 bank slice")
    if output["t1_sha256"] != native_t1_sha256:
        raise ValueError("native T1 digest differs from the supplied endpoint")
    if output["normalized_clip_bf16_sha256"] != rgb_artifacts.EXPECTED_CLIP_SHA256[mode]:
        raise ValueError("native normalized clip digest differs from the independent input expectation")
    return {"schema": NATIVE_SCHEMA, "context_mode": mode, "metrics": metrics,
            "output": output}


def _resolve_output(receipt_path: Path, value: str, label: str) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = Path(receipt_path).parent / path
    return _regular_file(path, label)


def _slice_sha256(path: Path, start: int, stop: int) -> str:
    return portrait.tensor_time_slice_sha256(
        path, {"offset": 0, "shape": list(LATENT_BANK_SHAPE)}, start, stop, itemsize=4
    )


def memory_estimate(mode: str) -> dict[str, Any]:
    mode = validate_context_mode(mode)
    source_copy_bytes = (3 * 120 if mode == "moving" else 3) * 288 * 512 * 4
    dit_prior_bytes = int(4.5 * GIB)
    dcae_reference_bytes = encoder_control.NATIVE_ENCODER_NAMED_PEAK_BYTES
    cpu_encoder_bytes = encoder_control.ENCODER_BF16_BYTES
    cpu_clip_staging_bytes = source_copy_bytes + CLIP_BYTES
    return {
        "external_cap_bytes": EXTERNAL_CAP_BYTES,
        "external_reserve_fraction": EXTERNAL_RESERVE_FRACTION,
        "prior_dit_peak_estimate_bytes": dit_prior_bytes,
        "prior_dit_peak_estimate_basis": "previous source control estimate; not measured in this process",
        "dcae_native_named_reference_bytes": dcae_reference_bytes,
        "cpu_encoder_weight_staging_bytes": cpu_encoder_bytes,
        "cpu_rgb_and_native_clip_staging_bytes": cpu_clip_staging_bytes,
        "conservative_named_total_estimate_bytes": (
            dit_prior_bytes + dcae_reference_bytes + cpu_encoder_bytes + cpu_clip_staging_bytes
        ),
        "dcae_reference_is_torch_mps_measurement": False,
        "opaque_mps_workspace_included": False,
        "external_guard_enforcement_verified_by_this_process": False,
    }


def admit_native_artifacts(
    receipt_path: Path,
    receipt_sha256: str,
    mode: str,
    frozen_input_identity: dict[str, str],
    native_t1_path: Path,
    native_t1_sha256: str,
) -> dict[str, Any]:
    receipt_path = _regular_file(receipt_path, "native RGB history receipt")
    if not isinstance(receipt_sha256, str) or HEX.fullmatch(receipt_sha256) is None:
        raise ValueError("native receipt requires lowercase SHA-256")
    if _sha256(receipt_path) != receipt_sha256:
        raise ValueError("native RGB history receipt SHA-256 differs")
    receipt = _json_file(receipt_path, "native RGB history receipt")
    contract = validate_native_receipt_document(
        receipt,
        mode=mode,
        frozen_input_identity=frozen_input_identity,
        native_t1_sha256=native_t1_sha256,
    )
    output = contract["output"]
    bank_path = _resolve_output(receipt_path, output["latent_bank"], "native latent bank")
    clip_path = _resolve_output(
        receipt_path, output["normalized_clip_bf16"], "native normalized BF16 clip"
    )
    t0_path = _resolve_output(receipt_path, output["dcae_latent_f32"], "native DCAE T0 endpoint")
    t1_path = _regular_file(native_t1_path, "native T1 endpoint")
    native_t1_sha256 = _sha256(t1_path)
    for path, expected_size, expected_sha, label in (
        (bank_path, LATENT_BANK_BYTES, output["latent_bank_sha256"], "native latent bank"),
        (clip_path, CLIP_BYTES, output["normalized_clip_bf16_sha256"], "native normalized clip"),
        (t0_path, LATENT_SEGMENT_BYTES, output["dcae_latent_f32_sha256"], "native DCAE T0 endpoint"),
        (t1_path, LATENT_SEGMENT_BYTES, output["t1_sha256"], "native T1 endpoint"),
    ):
        if path.stat().st_size != expected_size or _sha256(path) != expected_sha:
            raise ValueError(f"{label} extent or SHA-256 differs")
    if _slice_sha256(bank_path, 0, 30) != output["t0_sha256"]:
        raise ValueError("native T0 hash differs from latent-bank prefix")
    if _slice_sha256(bank_path, 30, 60) != output["t1_sha256"]:
        raise ValueError("native T1 hash differs from latent-bank suffix")
    if _slice_sha256(bank_path, 0, 60) != output["prefix60_sha256"]:
        raise ValueError("native 60-latent prefix hash differs")
    if _sha256(bank_path) != output["latent_bank_sha256"]:
        raise ValueError("native full latent-bank hash differs")
    return {
        "receipt_path": receipt_path,
        "receipt_sha256": receipt_sha256,
        "receipt": receipt,
        "bank_path": bank_path,
        "normalized_clip_path": clip_path,
        "t0_path": t0_path,
        "t1_path": t1_path,
        "context_mode": mode,
        "output": output,
    }


def admit_decoder_artifact(rgb_dir: Path) -> dict[str, Any]:
    rgb_dir = Path(rgb_dir).resolve(strict=True)
    receipt_path = _regular_file(rgb_dir / "receipt.json", "generated-pixel decoder receipt")
    if _sha256(receipt_path) != DECODER_RECEIPT_SHA256:
        raise ValueError("generated-pixel decoder receipt SHA-256 differs")
    receipt = _json_file(receipt_path, "generated-pixel decoder receipt")
    identity = validate_decoder_receipt_document(receipt)
    pixels_path = _regular_file(rgb_dir / "pixels.f32le", "generated RGB pixels")
    pixels = rgb_artifacts.validate_source(pixels_path)
    pixels._mmap.close()
    return {"receipt_path": receipt_path, "receipt_sha256": DECODER_RECEIPT_SHA256,
            "receipt": receipt, "pixels_path": pixels_path, "identity": identity}


def admit(args: argparse.Namespace) -> dict[str, Any]:
    mode = validate_context_mode(args.context_mode)
    decoder = admit_decoder_artifact(args.rgb_dir)
    t1_path = _regular_file(args.native_t1_latents, "native T1 endpoint")
    if _sha256(t1_path) != args.native_t1_latents_sha256:
        raise ValueError("native T1 endpoint SHA-256 differs")

    portrait_args = SimpleNamespace(
        input_root=args.input_root,
        upstream=args.upstream,
        cache_index=args.cache_index,
        t0_context="photo",
        t0_history_keep_latents=None,
        hide_t0_caption_keys=False,
        t1_current_text_kv=False,
        native_t0_latents=None,
        native_t0_latents_sha256=None,
        native_latents=t1_path,
        native_latents_sha256=args.native_t1_latents_sha256,
    )
    # The native receipt authenticates the DCAE endpoint and BF16 source clip;
    # validate it after the source admits the fixed shared-scene identity.
    native_receipt_path = _regular_file(args.native_receipt, "native RGB history receipt")
    if (not isinstance(args.native_receipt_sha256, str)
            or HEX.fullmatch(args.native_receipt_sha256) is None
            or _sha256(native_receipt_path) != args.native_receipt_sha256):
        raise ValueError("native RGB history receipt SHA-256 differs")
    native_receipt = _json_file(native_receipt_path, "native RGB history receipt")
    output = native_receipt.get("output")
    if not isinstance(output, dict) or not isinstance(output.get("dcae_latent_f32"), str):
        raise ValueError("native receipt omits the DCAE T0 endpoint")
    t0_path = _resolve_output(native_receipt_path, output["dcae_latent_f32"], "native DCAE T0 endpoint")
    t0_sha256 = output.get("dcae_latent_f32_sha256")
    portrait_args.native_t0_latents = t0_path
    portrait_args.native_t0_latents_sha256 = t0_sha256
    admitted = portrait.admit(portrait_args)
    native = admit_native_artifacts(
        native_receipt_path,
        args.native_receipt_sha256,
        mode,
        admitted["frozen_input_identity"],
        t1_path,
        args.native_t1_latents_sha256,
    )
    upstream = args.upstream.resolve(strict=True)
    revision = subprocess.check_output(
        ["git", "-C", str(upstream), "rev-parse", "HEAD"], text=True,
        stderr=subprocess.DEVNULL,
    ).strip()
    if revision != UPSTREAM_REVISION:
        raise ValueError("upstream revision differs from pinned DCAE source")
    dcae_module = encoder_control._required_file(
        upstream / "hymm/models/vae/autoencoder_dcae_v.py", "pinned DCAE module"
    )
    module_sha = _sha256(dcae_module)
    if module_sha != encoder_control.DCAE_MODULE_SHA256:
        raise ValueError("pinned DCAE module SHA-256 differs")
    encoder_weights = encoder_control._required_file(args.encoder_weights, "DCAE encoder weights")
    weights_sha = _sha256(encoder_weights)
    if weights_sha != encoder_control.RECOVERED_ENCODER_SHA256:
        raise ValueError("recovered DCAE encoder SHA-256 differs")
    expected_shapes, upstream_metadata = dcae_checkpoint.upstream_encoder_shapes(dcae_module)
    expected_bytes = sum(math.prod(shape) * 2 for shape in expected_shapes.values())
    if (len(expected_shapes) != encoder_control.ENCODER_TENSOR_COUNT
            or expected_bytes != encoder_control.ENCODER_BF16_BYTES):
        raise ValueError("DCAE encoder tensor schema differs from the pinned BF16 weights")
    output_dir = args.output_dir
    if not output_dir.is_absolute() or output_dir.exists() or output_dir.is_symlink():
        raise ValueError("output-dir must be a fresh absolute path")
    output_dir = output_dir.parent.resolve(strict=True) / output_dir.name
    admitted.update(
        t0_context=f"rgb-{mode}",
        rgb_context_mode=mode,
        rgb_decoder=decoder,
        rgb_pixels_path=decoder["pixels_path"],
        rgb_source_identity=decoder["identity"],
        native_rgb=native,
        dcae_module_path=dcae_module,
        dcae_module_sha256=module_sha,
        dcae_weights_path=encoder_weights,
        dcae_weights_sha256=weights_sha,
        dcae_expected_shapes=expected_shapes,
        dcae_upstream_metadata=upstream_metadata,
        output_dir=output_dir,
        producer_sha256=_sha256(Path(__file__)),
    )
    return admitted


class DCAEClipProvider:
    """Prepare one source BF16 clip, encode it, then return its GPU latent."""

    def __init__(self, admitted: dict[str, Any]):
        self.admitted = admitted
        self.model = None
        self.clip = None
        self.native_clip = None
        self.clip_comparison = None
        self.clip_elements = CLIP_ELEMENTS

    def prepare(self, torch, np, admitted, device):
        upstream = dcae_reference._load_module(admitted["dcae_module_path"])
        config = upstream.dc_ae_v_f32t4c32_config()
        with torch.device("meta"):
            model = upstream.DCAEV(config)
        model.encoder.to(dtype=torch.bfloat16)
        model.encoder.to_empty(device=device)
        encoder_state = encoder_control._read_encoder_weights(
            {"weights_path": admitted["dcae_weights_path"],
             "expected_shapes": admitted["dcae_expected_shapes"]},
            torch,
        )
        local_names = {name.removeprefix("encoder."): value for name, value in encoder_state.items()}
        model.encoder.load_state_dict(local_names, strict=True)
        del local_names, encoder_state
        model.eval()

        source_map = np.memmap(
            admitted["rgb_pixels_path"], mode="r", dtype="<f4",
            shape=rgb_artifacts.SOURCE_SHAPE,
        )
        if admitted["rgb_context_mode"] == "moving":
            selected = np.array(source_map[:, :, :120], copy=True)
        else:
            selected = np.array(source_map[:, :, 119:120], copy=True)
        source_map._mmap.close()
        rgb = torch.from_numpy(selected).to(device=device, dtype=torch.float32)
        del selected
        if admitted["rgb_context_mode"] == "repeat-last":
            rgb = rgb.repeat(1, 1, 120, 1, 1)
        clip = rgb.clamp_(0.0, 1.0).mul_(2.0).sub_(1.0).to(dtype=torch.bfloat16).contiguous()
        del rgb
        native_raw = np.memmap(
            admitted["native_rgb"]["normalized_clip_path"], mode="r", dtype="<u2",
            shape=rgb_artifacts.CLIP_SHAPE,
        )
        native_u16 = torch.from_numpy(np.array(native_raw, copy=True))
        native_raw._mmap.close()
        native_clip = native_u16.view(torch.bfloat16).to(device=device).contiguous()
        del native_u16
        if tuple(clip.shape) != rgb_artifacts.CLIP_SHAPE or clip.dtype is not torch.bfloat16:
            raise ValueError("prepared RGB history clip differs from the DCAE input contract")
        if tuple(native_clip.shape) != rgb_artifacts.CLIP_SHAPE or native_clip.dtype is not torch.bfloat16:
            raise ValueError("native normalized BF16 clip differs from the DCAE input contract")
        self.model = model
        self.clip = clip
        self.native_clip = native_clip

    def __call__(self, torch, np, admitted, device):
        with torch.no_grad():
            latent = self.model.encode(self.clip)
        if tuple(latent.shape) != LATENT_SEGMENT_SHAPE or latent.dtype is not torch.bfloat16:
            raise ValueError("upstream DCAE returned an unexpected RGB-history T0 latent")
        self.clip_comparison = portrait._comparison_stats(torch, self.clip, self.native_clip)
        del self.model, self.clip, self.native_clip
        self.model = self.clip = self.native_clip = None
        return latent

    def receipt_record(self, admitted: dict[str, Any]) -> dict[str, Any]:
        mode = admitted["rgb_context_mode"]
        return {
            "scope": (
                "matched moving-generated versus repeated-last-frame DCAE T0 context packages "
                "with the fixed T1 inputs and source BF16 schedule; no photo likeness or isolated "
                "motion-causality claim"
            ),
            "kind": "DCAE encoding of authenticated native-decoder RGB history",
            "mode": mode,
            "t0_denoise_forwards": 0,
            "t0_dcae_encode_calls": 1,
            "t0_cache_writer_forwards": 1,
            "writer_sigma": portrait.source.WRITER_SIGMA,
            "writer_timestep": portrait.source.WRITER_TIMESTEP,
            "source_frames": [0, 120] if mode == "moving" else [119, 120],
            "source_frame_repetitions": 1 if mode == "moving" else 120,
            "source_pixels_sha256": admitted["rgb_source_identity"]["pixels_sha256"],
            "decoder_receipt_sha256": admitted["rgb_decoder"]["receipt_sha256"],
            "normalization": (
                "F32 clamp to [0,1], multiply by 2, subtract 1, cast to BF16 RNE"
            ),
            "normalized_clip_bf16_sha256": admitted["native_rgb"]["output"][
                "normalized_clip_bf16_sha256"
            ],
            "native_receipt_sha256": admitted["native_rgb"]["receipt_sha256"],
            "dcae_module_sha256": admitted["dcae_module_sha256"],
            "dcae_encoder_sha256": admitted["dcae_weights_sha256"],
            "frozen_input_identity": admitted["frozen_input_identity"],
            "memory_estimate": memory_estimate(mode),
            "source_rng_parity": "not applicable; RGB history and all T1 inputs are authenticated",
        }


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--rgb-dir", type=Path, default=DEFAULT_RGB_DIR)
    parser.add_argument("--upstream", type=Path, default=DEFAULT_UPSTREAM)
    parser.add_argument("--cache-index", type=Path, default=portrait.source.single.DEFAULT_CACHE_INDEX)
    parser.add_argument("--encoder-weights", type=Path, default=DEFAULT_ENCODER_WEIGHTS)
    parser.add_argument("--context-mode", required=True, choices=rgb_artifacts.MODES)
    parser.add_argument("--native-receipt", type=Path, required=True)
    parser.add_argument("--native-receipt-sha256", required=True)
    parser.add_argument("--native-t1-latents", type=Path, required=True)
    parser.add_argument("--native-t1-latents-sha256", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--execute", action="store_true",
                        help="run source DCAE and fixed T1; requires the external 18 GiB guard")
    return parser


def main() -> int:
    parser = _build_parser()
    args = parser.parse_args()
    try:
        admitted = admit(args)
        if not args.execute:
            print(json.dumps({
                "validated": True,
                "model_launched": False,
                "context_mode": admitted["rgb_context_mode"],
                "source_pixels_sha256": admitted["rgb_source_identity"]["pixels_sha256"],
                "decoder_receipt_sha256": admitted["rgb_decoder"]["receipt_sha256"],
                "native_receipt_sha256": admitted["native_rgb"]["receipt_sha256"],
                "frozen_input_identity": admitted["frozen_input_identity"],
                "memory_estimate": memory_estimate(admitted["rgb_context_mode"]),
                "output_dir": str(admitted["output_dir"]),
            }, sort_keys=True, allow_nan=False))
        else:
            provider = DCAEClipProvider(admitted)
            result = portrait.execute(
                admitted,
                admitted["output_dir"],
                clean_provider=provider,
                context_record=provider.receipt_record(admitted),
            )
            print(json.dumps({
                "context_mode": admitted["rgb_context_mode"],
                "result": result["result"],
                "endpoint_comparisons": result["endpoint_comparisons"],
            }, sort_keys=True, allow_nan=False))
        return 0
    except (OSError, ValueError, subprocess.CalledProcessError) as error:
        parser.error(str(error))
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
