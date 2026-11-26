#!/usr/bin/env python3
"""Compare an authenticated Echo decoder seam using two pinned PixelUMM windows.

This diagnostic reuses the same-latent MPS block adapter and authenticated
weight/cache admission.  It runs no DiT and compares only the selected 20-frame
overlap after the pinned upstream temporal blender and native BF16 display
conversion.  Run it beneath ``examples/support/wan_memory_guard.py``.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import sys
from typing import Mapping

SUPPORT_DIR = Path(__file__).resolve().parent
if str(SUPPORT_DIR) not in sys.path:
    sys.path.insert(0, str(SUPPORT_DIR))

import hy_nomas_continuation_inputs as continuation_inputs
import hy_nomas_decoder_same_latent as same_latent

torch = same_latent.torch
block_reference = same_latent.block_reference
cache_inventory = same_latent.cache_inventory
checkpoint_inventory = same_latent.checkpoint_inventory
head_reference = same_latent.head_reference
head_checkpoint = same_latent.head_checkpoint
real_input = same_latent.real_input

SCHEMA = "cogni_ml.hy_nomas_decoder_boundary/v1"
NATIVE_SCHEMA = "cogni_ml.hy_nomas_native_continuation/v1"
ECHO_ROOT = Path("/opt/mldata_ssd/cogni-ml/hy-nomas-echo-20261030")
RUN_ROOT = ECHO_ROOT / "results/sequence-echo-20s"
DEFAULT_RECEIPT = RUN_ROOT / "native/receipt.json"
DEFAULT_INPUTS = ECHO_ROOT / "conditioning/continuation-inputs-echo-20s.json"
DIAGNOSTICS_DIR = RUN_ROOT / "diagnostics-decoder"

LATENT_SHAPE = continuation_inputs.LATENT_BANK_SHAPE
NOISE_SHAPE = continuation_inputs.PIXEL_NOISE_SHAPE
NATIVE_PIXEL_SHAPE = (1, 3, 480, 288, 512)
LATENT_BYTES = math.prod(LATENT_SHAPE) * 4
NOISE_BYTES = math.prod(NOISE_SHAPE) * 4
NATIVE_PIXEL_BYTES = math.prod(NATIVE_PIXEL_SHAPE) * 4
WINDOW_LATENTS = 10
WINDOW_PIXEL_FRAMES = WINDOW_LATENTS * continuation_inputs.PIXELS_PER_LATENT
OVERLAP_LATENTS = 5
OVERLAP_PIXEL_FRAMES = OVERLAP_LATENTS * continuation_inputs.PIXELS_PER_LATENT
ATOL = RTOL = 2e-5
GIB = 1024**3
# The pinned reference estimate covers one decoder window. Reserve an extra GiB
# for two hidden streams, the 60-frame local blend, and both window outputs.
TWO_WINDOW_WORKING_SET_BYTES = GIB
METRIC_CHUNK_ELEMENTS = 1 << 20


class DecoderBoundaryError(ValueError):
    """Echo receipt, input bundle, pinned decoder, or MPS admission failed."""


def _digest(value: object) -> bool:
    return (isinstance(value, str) and len(value) == 64
            and all(character in "0123456789abcdef" for character in value))


def seam_geometry(seam_frame: int) -> dict[str, int | list[int]]:
    """Return the two global windows and compared range for a 5-second seam."""
    if type(seam_frame) is not int or seam_frame not in (120, 240, 360):
        raise DecoderBoundaryError("seam frame must be one of 120, 240, or 360")
    if seam_frame % continuation_inputs.PIXELS_PER_LATENT:
        raise DecoderBoundaryError("seam frame is not aligned to a latent boundary")
    latent_boundary = seam_frame // continuation_inputs.PIXELS_PER_LATENT
    first_latent = latent_boundary - OVERLAP_LATENTS
    second_latent = latent_boundary
    first_pixel = first_latent * continuation_inputs.PIXELS_PER_LATENT
    second_pixel = second_latent * continuation_inputs.PIXELS_PER_LATENT
    compare_stop = seam_frame + OVERLAP_PIXEL_FRAMES
    if (first_latent < 0 or second_latent + WINDOW_LATENTS > LATENT_SHAPE[2]
            or compare_stop > NATIVE_PIXEL_SHAPE[2]):
        raise DecoderBoundaryError("selected seam lacks two complete admitted windows")
    return {
        "seam_frame": seam_frame,
        "latent_boundary": latent_boundary,
        "latent_starts": [first_latent, second_latent],
        "pixel_starts": [first_pixel, second_pixel],
        "noise_frame_range": [first_pixel, second_pixel + WINDOW_PIXEL_FRAMES],
        "compare_frame_range": [seam_frame, compare_stop],
        "local_compare_frame_range": [OVERLAP_PIXEL_FRAMES, WINDOW_PIXEL_FRAMES],
    }


def slice_ncthw_time(tensor, start: int, stop: int):
    """Select frames along NCTHW axis 2, preserving all channels and pixels."""
    if (not isinstance(tensor, torch.Tensor) or tensor.ndim != 5
            or type(start) is not int or type(stop) is not int
            or start < 0 or stop <= start or stop > tensor.shape[2]):
        raise DecoderBoundaryError("invalid NCTHW temporal slice")
    return tensor.narrow(2, start, stop - start)


def validate_native_receipt_document(
    document: Mapping, *, expected_inputs_sha256: str,
) -> dict[str, object]:
    """Admit the exact continuation run contract before reading its tensors."""
    if type(document) is not dict or document.get("schema") != NATIVE_SCHEMA:
        raise DecoderBoundaryError("unsupported native continuation receipt schema")
    if document.get("upstream_revision") != block_reference.UPSTREAM_REVISION:
        raise DecoderBoundaryError("native continuation upstream revision mismatch")
    for field, expected in (
        ("segments", 4), ("frames", 480), ("clean_latent_frames", 120),
        ("width", 512), ("height", 288), ("fps", 24),
        ("steps_per_segment", 4), ("shift", 5), ("cfg", 1),
    ):
        if type(document.get(field)) is not int or document[field] != expected:
            raise DecoderBoundaryError(f"native continuation {field} mismatch")
    if document.get("latent_bank_shape") != list(LATENT_SHAPE):
        raise DecoderBoundaryError("native continuation latent-bank geometry mismatch")
    if (not _digest(expected_inputs_sha256)
            or document.get("inputs_sha256") != expected_inputs_sha256):
        raise DecoderBoundaryError("native receipt does not authenticate the supplied input digest")
    if not isinstance(document.get("inputs"), str) or not Path(document["inputs"]).is_absolute():
        raise DecoderBoundaryError("native continuation input-manifest path is malformed")
    if not isinstance(document.get("prompt"), str) or not document["prompt"]:
        raise DecoderBoundaryError("native continuation prompt is missing")
    if document.get("continuation_experimental") is not True:
        raise DecoderBoundaryError("native continuation receipt is not marked experimental")
    expected_pixelumm = {
        "steps": 1, "window_latents": WINDOW_LATENTS,
        "overlap_latents": OVERLAP_LATENTS, "global_windows": 23,
    }
    if document.get("pixelumm") != expected_pixelumm:
        raise DecoderBoundaryError("native continuation PixelUMM window contract mismatch")
    diagnostics = document.get("diagnostics")
    if type(diagnostics) is not dict:
        raise DecoderBoundaryError("native continuation diagnostics are missing")
    for path_key, hash_key in (("latent_bank", "latent_bank_sha256"),
                               ("pixels", "pixels_sha256")):
        path_value = diagnostics.get(path_key)
        if not isinstance(path_value, str) or not Path(path_value).is_absolute():
            raise DecoderBoundaryError(f"native diagnostics.{path_key} path is malformed")
        if not _digest(diagnostics.get(hash_key)):
            raise DecoderBoundaryError(f"native diagnostics.{hash_key} is malformed")
    return dict(diagnostics)


def authenticate_file(
    path_value: str | Path, *, expected_sha256: str, expected_bytes: int, what: str,
) -> Path:
    """Require an absolute, regular, non-symlink artifact with its authorized identity."""
    try:
        path = same_latent._canonical_file(path_value, what)
    except same_latent.SameLatentError as error:
        raise DecoderBoundaryError(str(error)) from error
    if type(expected_bytes) is not int or expected_bytes <= 0 or path.stat().st_size != expected_bytes:
        raise DecoderBoundaryError(f"{what} byte extent does not match admitted geometry")
    if not _digest(expected_sha256) or same_latent._sha256_file(path) != expected_sha256:
        raise DecoderBoundaryError(f"{what} SHA-256 mismatch")
    return path


def _verify_input_payload(document: Mapping, payload_path: Path) -> None:
    """Hash aggregate and each packed continuation record in one bounded pass."""
    try:
        continuation_inputs.validate_document(
            document, payload_bytes=payload_path.stat().st_size
        )
    except continuation_inputs.InputsError as error:
        raise DecoderBoundaryError(f"continuation input manifest rejected: {error}") from error
    if payload_path.is_symlink() or not payload_path.is_file():
        raise DecoderBoundaryError("continuation input payload must be a regular non-symlink file")
    aggregate = hashlib.sha256()
    with payload_path.open("rb") as stream:
        for name, _, _ in continuation_inputs.TENSOR_SPECS:
            record = document["tensors"][name]
            if stream.tell() != record["offset"]:
                raise DecoderBoundaryError("continuation payload has a gap or overlap")
            record_hash = hashlib.sha256()
            remaining = record["nbytes"]
            while remaining:
                block = stream.read(min(8 * 1024 * 1024, remaining))
                if not block:
                    raise DecoderBoundaryError(f"continuation payload tensor is truncated: {name}")
                aggregate.update(block)
                record_hash.update(block)
                remaining -= len(block)
            if record_hash.hexdigest() != record["sha256"]:
                raise DecoderBoundaryError(f"continuation tensor digest mismatch: {name}")
        if stream.read(1):
            raise DecoderBoundaryError("continuation payload has trailing bytes")
    if aggregate.hexdigest() != document["payload_sha256"]:
        raise DecoderBoundaryError("continuation aggregate payload digest mismatch")


def validate_input_bundle(
    manifest_path: Path, *, expected_sha256: str, receipt: Mapping,
) -> tuple[dict, Path, dict]:
    """Bind the manifest path, digest, prompt, upstream revision, and packed bytes."""
    try:
        manifest_path = same_latent._canonical_file(manifest_path, "continuation input manifest")
        manifest, raw = same_latent._read_json(manifest_path, "continuation input manifest")
        receipt_manifest = same_latent._canonical_file(
            receipt["inputs"], "native receipt continuation input manifest"
        )
    except (same_latent.SameLatentError, KeyError, TypeError) as error:
        raise DecoderBoundaryError(str(error)) from error
    actual_sha256 = hashlib.sha256(raw).hexdigest()
    if (not _digest(expected_sha256) or actual_sha256 != expected_sha256
            or actual_sha256 != receipt.get("inputs_sha256")):
        raise DecoderBoundaryError("continuation manifest digest differs from CLI and receipt")
    if receipt_manifest != manifest_path:
        raise DecoderBoundaryError("continuation manifest path differs from native receipt")
    try:
        continuation_inputs.validate_document(manifest)
    except continuation_inputs.InputsError as error:
        raise DecoderBoundaryError(f"continuation input manifest rejected: {error}") from error
    if manifest.get("upstream_revision") != block_reference.UPSTREAM_REVISION:
        raise DecoderBoundaryError("continuation input upstream revision mismatch")
    if manifest.get("prompt") != receipt.get("prompt"):
        raise DecoderBoundaryError("native receipt prompt differs from continuation inputs")
    payload_path = manifest_path.parent / manifest["payload"]
    try:
        payload_path = same_latent._canonical_file(payload_path, "continuation input payload")
    except same_latent.SameLatentError as error:
        raise DecoderBoundaryError(str(error)) from error
    _verify_input_payload(manifest, payload_path)
    noise = manifest["tensors"]["decoder.noise"]
    if (noise["dtype"] != "F32" or noise["shape"] != list(NOISE_SHAPE)
            or noise["nbytes"] != NOISE_BYTES):
        raise DecoderBoundaryError("continuation decoder-noise geometry mismatch")
    return manifest, payload_path, noise


def _check_echo_path(path: Path, expected: Path, what: str) -> Path:
    try:
        canonical = same_latent._canonical_file(path, what)
        expected_canonical = expected.resolve(strict=True)
    except (OSError, same_latent.SameLatentError) as error:
        raise DecoderBoundaryError(f"{what} is unavailable") from error
    if canonical != expected_canonical:
        raise DecoderBoundaryError(f"{what} is outside the authenticated Echo run")
    return canonical


def preflight(
    *, receipt_path: Path, receipt_sha256: str,
    inputs_path: Path, inputs_sha256: str,
) -> dict[str, object]:
    """Authenticate all CPU-side provenance before any MPS model allocation."""
    receipt_path = _check_echo_path(receipt_path, DEFAULT_RECEIPT, "native continuation receipt")
    receipt, raw_receipt = same_latent._read_json(receipt_path, "native continuation receipt")
    actual_receipt_sha = hashlib.sha256(raw_receipt).hexdigest()
    if not _digest(receipt_sha256) or actual_receipt_sha != receipt_sha256:
        raise DecoderBoundaryError("native continuation receipt SHA-256 mismatch")
    diagnostics = validate_native_receipt_document(
        receipt, expected_inputs_sha256=inputs_sha256
    )
    inputs_path = _check_echo_path(inputs_path, DEFAULT_INPUTS, "continuation input manifest")
    manifest, payload_path, noise_record = validate_input_bundle(
        inputs_path, expected_sha256=inputs_sha256, receipt=receipt
    )
    latent_path = _check_echo_path(
        Path(diagnostics["latent_bank"]), RUN_ROOT / "native/latent-bank.f32le",
        "native latent bank",
    )
    pixels_path = _check_echo_path(
        Path(diagnostics["pixels"]), RUN_ROOT / "native/pixels.f32le", "native display pixels"
    )
    authenticate_file(
        latent_path, expected_sha256=diagnostics["latent_bank_sha256"],
        expected_bytes=LATENT_BYTES, what="native latent bank",
    )
    authenticate_file(
        pixels_path, expected_sha256=diagnostics["pixels_sha256"],
        expected_bytes=NATIVE_PIXEL_BYTES, what="native display pixels",
    )

    sources = same_latent._validate_upstream_source_metadata()
    raw_header, stem_path, stem_receipt_path = same_latent._authenticate_stem()
    head = same_latent._authenticate_head(raw_header)
    cache = same_latent._authenticate_full36_cache()
    cache_manifest, cache_path, cache_payload_path, cache_manifest_sha, cache_payload_sha = cache
    return {
        "receipt": receipt,
        "receipt_sha256": actual_receipt_sha,
        "diagnostics": diagnostics,
        "manifest": manifest,
        "manifest_sha256": inputs_sha256,
        "payload_path": payload_path,
        "payload_sha256": manifest["payload_sha256"],
        "noise_record": noise_record,
        "latent_path": latent_path,
        "pixels_path": pixels_path,
        "source_metadata": sources,
        "raw_header": raw_header,
        "stem_path": stem_path,
        "stem_receipt_path": stem_receipt_path,
        "head": head,
        "cache_manifest": cache_manifest,
        "cache_path": cache_path,
        "cache_payload_path": cache_payload_path,
        "cache_manifest_sha256": cache_manifest_sha,
        "cache_payload_sha256": cache_payload_sha,
    }


def _memory_preflight(arch, cache_resident_allowance_bytes: int) -> dict[str, int | str]:
    """Apply the pinned one-window estimate plus a two-window working-set reserve."""
    memory = same_latent._mps_preflight(
        arch, cache_resident_allowance_bytes=cache_resident_allowance_bytes
    )
    memory["two_window_working_set_reserve_bytes"] = TWO_WINDOW_WORKING_SET_BYTES
    memory["reference_peak_estimate_bytes"] += TWO_WINDOW_WORKING_SET_BYTES
    memory["estimated_peak_bytes"] += TWO_WINDOW_WORKING_SET_BYTES
    available = memory["physical_bytes"] - memory["driver_allocated_bytes"]
    if available - memory["estimated_peak_bytes"] < memory["required_reserve_bytes"]:
        raise DecoderBoundaryError(
            "two-window cache-aware MPS preflight would leave less than 10% physical memory"
        )
    return memory


def _mps_slice(path: Path, shape: tuple[int, ...], start: int, stop: int, device):
    full_bytes = math.prod(shape) * 4
    mapped = same_latent._mapped_f32(
        path, offset=0, nbytes=full_bytes, shape=shape
    )
    selected = slice_ncthw_time(mapped, start, stop).to(device=device)
    del mapped
    return selected


def _display_range_bf16(raw):
    """Mirror native finalize: BF16-rounded multiply, add, then clamp."""
    scaled = (raw.to(dtype=torch.float32) * 0.5).to(dtype=torch.bfloat16).to(dtype=torch.float32)
    shifted = (scaled + 0.5).to(dtype=torch.bfloat16).to(dtype=torch.float32)
    return shifted.clamp(0.0, 1.0)


def _queue_comparison(actual, expected):
    if tuple(actual.shape) != tuple(expected.shape) or actual.ndim != 5:
        raise DecoderBoundaryError("seam comparison geometry mismatch")
    left = actual.reshape(-1)
    right = expected.reshape(-1)
    max_chunks = []
    sum_chunks = []
    mismatch_chunks = []
    finite_chunks = []
    for start in range(0, left.numel(), METRIC_CHUNK_ELEMENTS):
        stop = min(start + METRIC_CHUNK_ELEMENTS, left.numel())
        a = left[start:stop]
        b = right[start:stop]
        delta = (a - b).abs()
        max_chunks.append(delta.max())
        sum_chunks.append(delta.sum())
        mismatch_chunks.append((delta > (ATOL + RTOL * b.abs())).any())
        finite_chunks.append(torch.isfinite(a).all().logical_and(torch.isfinite(b).all()))
    return torch.stack((
        torch.stack(max_chunks).max(),
        torch.stack(sum_chunks).sum() / left.numel(),
        torch.stack(mismatch_chunks).any().to(torch.float32),
        torch.stack(finite_chunks).all().logical_not().to(torch.float32),
    ))


def _output_paths(seam_frame: int) -> tuple[Path, Path]:
    if type(seam_frame) is not int or seam_frame not in (120, 240, 360):
        raise DecoderBoundaryError("invalid seam output frame")
    try:
        run_root = RUN_ROOT.resolve(strict=True)
        if DIAGNOSTICS_DIR.is_symlink():
            raise DecoderBoundaryError("diagnostics-decoder may not be a symlink")
        if not DIAGNOSTICS_DIR.exists():
            DIAGNOSTICS_DIR.mkdir(mode=0o700)
        output_dir = DIAGNOSTICS_DIR.resolve(strict=True)
    except OSError as error:
        raise DecoderBoundaryError("Echo run root is unavailable for diagnostic output") from error
    if output_dir.parent != run_root:
        raise DecoderBoundaryError("diagnostic output directory is outside the Echo run")
    stem = f"decoder-boundary-{seam_frame}"
    output_json = output_dir / f"{stem}.json"
    output_pixels = output_dir / f"{stem}.f32le"
    if os.path.lexists(output_json) or os.path.lexists(output_pixels):
        raise DecoderBoundaryError("refusing to replace an existing seam diagnostic")
    return output_json, output_pixels


def run_boundary(
    admission: Mapping[str, object], *, seam_frame: int,
    output_json: Path, output_pixels: Path,
) -> dict[str, object]:
    """Decode exactly two global windows, blend their seam, and compare display pixels."""
    geometry = seam_geometry(seam_frame)
    starts = geometry["latent_starts"]
    first_pixel, second_pixel = geometry["pixel_starts"]
    noise_start, noise_stop = geometry["noise_frame_range"]
    compare_start, compare_stop = geometry["compare_frame_range"]
    local_start, local_stop = geometry["local_compare_frame_range"]
    device = torch.device("mps")

    source, model_sha = block_reference.load_upstream(same_latent.UPSTREAM)
    if model_sha != real_input.UPSTREAM_MODEL_SHA256:
        raise DecoderBoundaryError("pinned upstream model.py source digest changed")
    dcae_source, source_again, autoencoder_source = real_input._load_modules(same_latent.UPSTREAM)
    if (source_again is not source
            and Path(source_again.__file__).resolve() != Path(source.__file__).resolve()):
        raise DecoderBoundaryError("upstream module import identity changed")
    if same_latent._sha256_file(Path(dcae_source.__file__)) != real_input.DCAE_ENCODER_SHA256:
        raise DecoderBoundaryError("pinned upstream DCAE source digest changed")
    if same_latent._sha256_file(Path(autoencoder_source.__file__)) != real_input.PIXELUMM_NOISE_SHA256:
        raise DecoderBoundaryError("pinned upstream decoder/blender source digest changed")
    block_reference.validate_preview_oracle_runtime(same_latent.GRID, "mps")
    arch = source.QWEN3_8B_GEN
    cache_manifest = admission["cache_manifest"]
    cache_payload_path = admission["cache_payload_path"]
    cache_words = same_latent._load_full36_cache(admission)[2]
    memory = _memory_preflight(arch, cache_payload_path.stat().st_size)

    header_path = same_latent._canonical_file(
        checkpoint_inventory.ROOT / "source" / "pixelumm-header.json",
        "pinned PixelUMM checkpoint header",
    )
    plan, inverse_cpu, _ = block_reference.make_attention_plan(source, same_latent.GRID, tile=8)
    permutation = torch.tensor(plan.permutation, dtype=torch.long, device=device)
    inverse = inverse_cpu.to(device=device)
    cos, sin = source.mrope_tables(arch, same_latent.GRID, device)
    cos = cos.index_select(1, permutation)
    sin = sin.index_select(1, permutation)

    stem = real_input._load_stem(source, device)
    final_norm, head, head_runtime_weights, head_source = head_reference._load_head_state(
        source=source,
        subset_path=admission["head"]["subset_path"],
        raw_header=admission["raw_header"],
        receipt_path=admission["head"]["receipt_path"],
    )
    del head_runtime_weights

    latent_begin, latent_end = starts[0], starts[1] + WINDOW_LATENTS
    latent_union = _mps_slice(
        admission["latent_path"], LATENT_SHAPE, latent_begin, latent_end, device
    )
    # The packed file is mapped on CPU; only the 60-frame seam neighborhood is
    # materialized on MPS. Slice time on axis 2 before any reshape/flattening.
    payload = admission["payload_path"]
    payload_cpu = torch.from_file(
        str(payload), shared=False, size=payload.stat().st_size // 4,
        dtype=torch.float32,
    )
    noise_record = admission["noise_record"]
    noise_full_cpu = payload_cpu.narrow(
        0, noise_record["offset"] // 4, noise_record["nbytes"] // 4
    ).view(NOISE_SHAPE)
    noise_union = slice_ncthw_time(noise_full_cpu, noise_start, noise_stop).to(device=device)
    del payload_cpu, noise_full_cpu
    native_pixels_cpu = same_latent._mapped_f32(
        admission["pixels_path"], offset=0, nbytes=NATIVE_PIXEL_BYTES,
        shape=NATIVE_PIXEL_SHAPE,
    )
    native_seam = slice_ncthw_time(native_pixels_cpu, compare_start, compare_stop).to(device=device)
    del native_pixels_cpu

    condition = type("_ConditionProjection", (), {})()
    condition.cond_proj = stem.cond_proj
    condition.arch = arch
    current_windows = []
    layer_checks = []
    with torch.inference_mode():
        with torch.autocast(device_type="mps", dtype=torch.bfloat16):
            for latent_start, pixel_start in zip(starts, (first_pixel, second_pixel)):
                latent_window = slice_ncthw_time(
                    latent_union, latent_start - latent_begin,
                    latent_start - latent_begin + WINDOW_LATENTS,
                )
                noise_window = slice_ncthw_time(
                    noise_union, pixel_start - noise_start,
                    pixel_start - noise_start + WINDOW_PIXEL_FRAMES,
                )
                xt_tokens = stem.embed(source.patchify(noise_window.to(stem.embed.weight.dtype)))
                condition_tokens = source.DCAEVPixelUMMDecoder._condition_tokens(
                    condition, latent_window.to(stem.cond_proj.weight.dtype)
                )
                current_windows.append((xt_tokens + condition_tokens).index_select(1, permutation))
                del latent_window, noise_window, xt_tokens, condition_tokens

            for block_index in range(36):
                block_record = cache_manifest["blocks"][str(block_index)]
                source_dir = cache_inventory._source_dir_for_block(36, block_index)
                layer, checked, receipt_sha = block_reference._load_layer_for_block(
                    source,
                    source_dir / block_record["subset_file"],
                    source_dir / block_record["source_receipt_file"],
                    header_path,
                    device,
                    block_index,
                )
                if (block_record.get("subset_sha256") != checked["subset_sha256"]
                        or block_record.get("subset_bytes") != checked["subset_bytes"]
                        or block_record.get("source_extent") != checked["block_data_offsets"]
                        or block_record.get("source_receipt_sha256") != receipt_sha):
                    raise DecoderBoundaryError(
                        f"BF16 cache provenance differs from source for block {block_index}"
                    )
                layer_checks.extend(block_reference._queue_cache_weight_word_checks(
                    layer,
                    cache_manifest=cache_manifest,
                    cache_words=cache_words,
                    block_index=block_index,
                    device=device,
                ).values())
                for index, current in enumerate(current_windows):
                    current_windows[index], captures = block_reference.run_layer_reference(
                        layer, current, cos, sin, plan
                    )
                    if captures:
                        raise DecoderBoundaryError("unexpected decoder capture was retained")
                del layer

        blender = autoencoder_source._TemporalBlender(
            total_frames=WINDOW_PIXEL_FRAMES + OVERLAP_PIXEL_FRAMES
        )
        for index, local_window_start in enumerate((0, OVERLAP_PIXEL_FRAMES)):
            with torch.autocast(device_type="mps", dtype=torch.bfloat16):
                decoder_tokens = final_norm(current_windows[index])
                decoder_tokens = decoder_tokens.index_select(1, inverse)
            prediction_tokens = head(decoder_tokens)
            prediction = source.unpatchify(prediction_tokens, same_latent.GRID)
            blender.add(prediction, local_window_start)
            del decoder_tokens, prediction_tokens, prediction

        merged_raw = blender.result()
        merged_seam = slice_ncthw_time(merged_raw, local_start, local_stop)
        upstream_display = _display_range_bf16(merged_seam)
        metric_vector = _queue_comparison(upstream_display, native_seam)
        cache_ok = torch.stack(layer_checks).all()
        final_metrics = torch.cat((metric_vector, cache_ok.logical_not().to(torch.float32).view(1)))

    # All input, decoder, blend, and comparison work above remains on the MPS
    # queue. One terminal boundary permits the only result readback.
    torch.mps.synchronize()
    max_abs, mean_abs, mismatch, nonfinite, bad_cache = final_metrics.cpu().tolist()
    output_cpu = upstream_display.to(dtype=torch.float32).contiguous().cpu()
    del final_metrics, cache_words, latent_union, noise_union, native_seam, merged_raw

    if nonfinite:
        raise DecoderBoundaryError("seam comparison contains non-finite pixels")
    if bad_cache:
        raise DecoderBoundaryError("upstream source weights differ from authenticated BF16 cache")
    try:
        with output_pixels.open("xb") as stream:
            output_cpu.numpy().astype("<f4", copy=False).tofile(stream)
            stream.flush()
            os.fsync(stream.fileno())
    except OSError as error:
        raise DecoderBoundaryError(f"cannot write seam pixels: {error}") from error
    output_sha = same_latent._sha256_file(output_pixels)
    result = {
        "schema": SCHEMA,
        "upstream_revision": block_reference.UPSTREAM_REVISION,
        "upstream_model_sha256": model_sha,
        "native_receipt_sha256": admission["receipt_sha256"],
        "native_input_manifest_sha256": admission["manifest_sha256"],
        "continuation_payload_sha256": admission["payload_sha256"],
        "native_latent_bank_sha256": admission["diagnostics"]["latent_bank_sha256"],
        "native_display_pixels_sha256": admission["diagnostics"]["pixels_sha256"],
        "decoder_cache_manifest_sha256": admission["cache_manifest_sha256"],
        "decoder_cache_payload_sha256": admission["cache_payload_sha256"],
        "stem_subset_sha256": checkpoint_inventory.validate_named_subset_receipt(
            admission["raw_header"],
            checkpoint_inventory.parse_header(admission["stem_receipt_path"].read_bytes()),
            admission["stem_path"],
            "stem",
        )["subset_sha256"],
        "head_subset_sha256": head_source["subset_sha256"],
        "head_receipt_sha256": head_source["receipt_sha256"],
        "seam": geometry,
        "windows": {
            "latent_frames": WINDOW_LATENTS,
            "pixel_frames": WINDOW_PIXEL_FRAMES,
            "overlap_pixel_frames": OVERLAP_PIXEL_FRAMES,
            "noise_tensor": "authenticated continuation decoder.noise F32 NCTHW",
            "merge": "pinned upstream _TemporalBlender; local starts 0 and 20 preserve global seam weights",
        },
        "comparison": {
            "representation": "display F32 [0,1]",
            "native_source": "saved sequence pixels.f32le after BF16-rounded multiply/add and clamp",
            "upstream_transform": "BF16-rounded multiply by 0.5, BF16-rounded add 0.5, clamp [0,1]",
            "atol": ATOL,
            "rtol": RTOL,
            "compared_global_frames": [compare_start, compare_stop],
            "compared_shape": [1, 3, OVERLAP_PIXEL_FRAMES, 288, 512],
        },
        "memory_preflight": memory,
        "weight_word_checks": len(layer_checks),
        "metrics": {
            "max_abs_error": max_abs,
            "mean_abs_error": mean_abs,
            "any_value_outside_tolerance": bool(mismatch),
            "all_values_finite": not bool(nonfinite),
            "all_source_weights_match_bf16_cache": not bool(bad_cache),
            "result": "match" if not mismatch else "mismatch",
        },
        "upstream_display_pixels": str(output_pixels),
        "upstream_display_pixels_bytes": NATIVE_PIXEL_BYTES // (NATIVE_PIXEL_SHAPE[2] // OVERLAP_PIXEL_FRAMES),
        "upstream_display_pixels_sha256": output_sha,
    }
    try:
        with output_json.open("x", encoding="utf-8") as stream:
            json.dump(result, stream, indent=2)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
    except OSError as error:
        raise DecoderBoundaryError(f"cannot write seam result manifest: {error}") from error
    result["output_json"] = str(output_json)
    return result


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--native-receipt", type=Path, default=DEFAULT_RECEIPT)
    parser.add_argument("--native-receipt-sha256", required=True)
    parser.add_argument("--inputs", type=Path, default=DEFAULT_INPUTS)
    parser.add_argument("--inputs-sha256", required=True)
    parser.add_argument("--seam-frame", type=int, choices=(120, 240, 360), default=120)
    parser.add_argument(
        "--admission-only", action="store_true",
        help="authenticate Echo artifacts and pinned decoder inputs without importing/running MPS models",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        seam_geometry(args.seam_frame)
        admitted = preflight(
            receipt_path=args.native_receipt,
            receipt_sha256=args.native_receipt_sha256,
            inputs_path=args.inputs,
            inputs_sha256=args.inputs_sha256,
        )
        if args.admission_only:
            summary = {
                "schema": SCHEMA,
                "admission": "passed",
                "receipt_sha256": admitted["receipt_sha256"],
                "input_manifest_sha256": admitted["manifest_sha256"],
                "input_payload_sha256": admitted["payload_sha256"],
                "seam": seam_geometry(args.seam_frame),
                "upstream_revision": admitted["source_metadata"]["revision"],
                "decoder_cache_payload_sha256": admitted["cache_payload_sha256"],
            }
            print(json.dumps(summary, indent=2))
            return 0
        output_json, output_pixels = _output_paths(args.seam_frame)
        result = run_boundary(
            admitted, seam_frame=args.seam_frame,
            output_json=output_json, output_pixels=output_pixels,
        )
    except (
        DecoderBoundaryError,
        same_latent.SameLatentError,
        block_reference.ReferenceError,
        head_reference.HeadReferenceError,
        real_input.RealInputError,
        cache_inventory.CacheError,
        checkpoint_inventory.InventoryError,
        continuation_inputs.InputsError,
        head_checkpoint.HeadCheckpointError,
        RuntimeError,
        OSError,
    ) as error:
        print(f"Hy-Nomas decoder boundary oracle failed: {error}", file=sys.stderr)
        return 2
    print(json.dumps({
        "output": result["output_json"],
        "seam_frame": args.seam_frame,
        "result": result["metrics"]["result"],
        "memory_preflight": result["memory_preflight"],
    }, indent=2))
    return 0 if result["metrics"]["result"] == "match" else 1


if __name__ == "__main__":
    raise SystemExit(main())
