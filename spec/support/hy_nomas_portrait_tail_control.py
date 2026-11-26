#!/usr/bin/env python3
"""Run a pinned PixelUMM 90-latent synthetic-tail decoder counterfactual.

The extra 30 latents repeat the original [50:60] cycle three times. This is
synthetic future context, and the intervention jointly changes future context,
local temporal RoPE positions, and overlap blending. It does not rerun DiT.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time
from typing import Mapping

import hy_nomas_portrait_decoder_control as base

ControlError = base.ControlError
WINDOW_LATENTS = 10
OVERLAP_LATENTS = 5
LATENT90_SHAPE = (1, 32, 90, 9, 16)
LATENT60_SHAPE = (1, 32, 60, 9, 16)
PIXELS360_SHAPE = (1, 3, 360, 288, 512)
PIXELS240_SHAPE = base.PIXELS_SHAPE
NOISE360_SHAPE = (1, 3, 360, 288, 512)
WINDOW_STARTS = tuple(range(0, 81, 5))
T1_CROP = (120, 240)
PREFIX_CROP = (120, 220)
TAIL_CROP = (220, 240)
EYE_XYXY = (185, 85, 335, 150)
ATOL = RTOL = 2e-5
LATENT90_BYTES = math.prod(LATENT90_SHAPE) * 4
PIXELS360_BYTES = math.prod(PIXELS360_SHAPE) * 4
NATIVE_SCHEMA = "cogni_ml.hy_nomas_decode_latent/v1"
NATIVE_EXECUTABLE_SHA256 = "2c857f83d5b95633ba4d671c910f1d2a9552b3a2e80df10852a19021e0f380fa"
SCHEMA = "cogni_ml.hy_nomas_portrait_tail_control/v1"


def window_starts(frames: int) -> tuple[int, ...]:
    if type(frames) is not int or frames < WINDOW_LATENTS:
        raise ControlError("latent frame count must be an integer >= 10")
    return tuple(range(0, frames - WINDOW_LATENTS + 1, OVERLAP_LATENTS))


def validate_latent_artifact_bytes(data: bytes, expected_sha256: str,
                                   *, expected_bytes: int = LATENT90_BYTES) -> str:
    if len(data) != expected_bytes:
        raise ControlError("custom latent artifact extent mismatch")
    actual = hashlib.sha256(data).hexdigest()
    base._require_digest(actual, expected_sha256, "custom latent artifact")
    return actual


def _channel_prefix_hash(data: bytes, *, channels: int, frames: int,
                         frame_bytes: int, prefix_frames: int) -> str:
    digest = hashlib.sha256()
    channel_bytes = frames * frame_bytes
    for channel in range(channels):
        start = channel * channel_bytes
        digest.update(data[start:start + prefix_frames * frame_bytes])
    return digest.hexdigest()


def validate_cyclic_tail_bytes(data: bytes, *, channels: int, frames: int,
                               frame_bytes: int, prefix: bytes,
                               prefix_frames: int, cycle_start: int,
                               cycle_frames: int, repeats: int) -> None:
    if (type(channels) is not int or channels <= 0 or type(frames) is not int
            or type(frame_bytes) is not int or frame_bytes <= 0
            or type(prefix_frames) is not int or type(cycle_start) is not int
            or type(cycle_frames) is not int or type(repeats) is not int
            or prefix_frames <= 0 or cycle_start < 0 or cycle_frames <= 0 or repeats <= 0
            or cycle_start + cycle_frames > prefix_frames
            or frames != prefix_frames + cycle_frames * repeats
            or len(prefix) != channels * prefix_frames * frame_bytes
            or len(data) != channels * frames * frame_bytes):
        raise ControlError("cyclic extension geometry is malformed")
    prefix_channel_bytes = prefix_frames * frame_bytes
    target_channel_bytes = frames * frame_bytes
    tail_bytes = cycle_frames * frame_bytes
    for channel in range(channels):
        source_start = channel * prefix_channel_bytes
        target_start = channel * target_channel_bytes
        if data[target_start:target_start + prefix_channel_bytes] != prefix[
                source_start:source_start + prefix_channel_bytes]:
            raise ControlError("custom latent common prefix differs from original capture")
        cycle = data[target_start + cycle_start * frame_bytes:
                     target_start + (cycle_start + cycle_frames) * frame_bytes]
        expected_tail = cycle * repeats
        actual_tail = data[target_start + prefix_channel_bytes:
                            target_start + target_channel_bytes]
        if len(expected_tail) != repeats * tail_bytes or actual_tail != expected_tail:
            raise ControlError("custom latent cyclic extension differs from source [50:60] repeated")


def make_cyclic_tail_bytes(prefix: bytes, *, channels: int, frames: int,
                           frame_bytes: int, prefix_frames: int,
                           cycle_start: int, cycle_frames: int,
                           repeats: int) -> bytes:
    if len(prefix) != channels * frames * frame_bytes:
        raise ControlError("source latent prefix extent mismatch")
    source_channel_bytes = frames * frame_bytes
    output = bytearray()
    for channel in range(channels):
        start = channel * source_channel_bytes
        source = prefix[start:start + source_channel_bytes]
        cycle = source[cycle_start * frame_bytes:(cycle_start + cycle_frames) * frame_bytes]
        if (len(cycle) != cycle_frames * frame_bytes or prefix_frames != frames
                or cycle_start + cycle_frames > prefix_frames):
            raise ControlError("cyclic extension geometry is malformed")
        output.extend(source)
        output.extend(cycle * repeats)
    return bytes(output)


def validate_native90_receipt(receipt: dict, *, receipt_sha256: str,
                              expected_receipt_sha256: str,
                              latent_sha256: str) -> None:
    base._require_digest(receipt_sha256, expected_receipt_sha256, "native90 receipt SHA-256")
    expected = {
        "schema": NATIVE_SCHEMA,
        "executable_sha256": NATIVE_EXECUTABLE_SHA256,
        "latent_sha256": latent_sha256,
        "latent_bank_sha256": latent_sha256,
        "latent_bytes": LATENT90_BYTES,
        "latent_dtype": "F32 little-endian",
        "latent_shape": list(LATENT90_SHAPE),
        "latent_frames": 90,
        "latent_selected_frames": 90,
        "latent_prefix_start_frame": 0,
        "latent_bank_shape": list(LATENT90_SHAPE),
        "latent_bank_frames": 90,
        "frames": 360, "width": 512, "height": 288, "fps": 24,
        "output_dtype": "F32 little-endian planar RGB",
        "output_bytes": PIXELS360_BYTES,
        "output_shape": list(PIXELS360_SHAPE),
    }
    if type(receipt) is not dict or any(receipt.get(k) != v for k, v in expected.items()):
        raise ControlError("native90 receipt geometry or identity mismatch")
    pixelumm = receipt.get("pixelumm")
    if (type(pixelumm) is not dict or pixelumm.get("steps") != 1
            or pixelumm.get("window_latents") != WINDOW_LATENTS
            or pixelumm.get("overlap_latents") != OVERLAP_LATENTS
            or pixelumm.get("starts") != list(WINDOW_STARTS)):
        raise ControlError("native90 receipt PixelUMM window geometry mismatch")
    metrics = receipt.get("metrics")
    if type(metrics) is not dict:
        raise ControlError("native90 receipt decoder counters are missing")
    required_counters = {
        "decoder_encode_allocations": 0,
        "decoder_encode_readbacks": 0,
        "decoder_preterminal_waits": 0,
        "decoder_waits": 1,
    }
    for key, value in required_counters.items():
        if type(metrics.get(key)) is not int or metrics[key] != value:
            raise ControlError(f"native90 receipt counter rejected: {key}")
    for key in ("latent_sha256", "latent_bank_sha256", "output_sha256"):
        digest = receipt.get(key)
        if type(digest) is not str or len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
            raise ControlError(f"native90 receipt digest malformed: {key}")


def _read_native90(args, custom_path: Path, custom_sha: str) -> tuple[Path, dict, str]:
    receipt_path = args.native_receipt.expanduser()
    if receipt_path.is_symlink() or not receipt_path.is_file():
        raise ControlError("native90 receipt must be a regular non-symlink file")
    raw = base._read_regular(receipt_path, "native90 receipt", max_bytes=8 * 1024 * 1024)
    actual_receipt_sha = hashlib.sha256(raw).hexdigest()
    base._require_digest(actual_receipt_sha, args.native_sha256.lower(), "native90 receipt SHA-256")
    try:
        receipt = json.loads(raw.decode("utf-8"), object_pairs_hook=base._unique_object)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ControlError("cannot parse native90 receipt") from error
    validate_native90_receipt(receipt, receipt_sha256=actual_receipt_sha,
                              expected_receipt_sha256=args.native_sha256.lower(),
                              latent_sha256=custom_sha)
    try:
        receipt_latent = Path(receipt["latent"]).expanduser().resolve(strict=True)
    except (KeyError, OSError, TypeError) as error:
        raise ControlError("native90 receipt latent path is invalid") from error
    if receipt_latent != custom_path.resolve(strict=True):
        raise ControlError("native90 receipt names a different custom latent artifact")
    output_value = receipt.get("output")
    if type(output_value) is not str or not output_value:
        raise ControlError("native90 receipt output path is missing")
    output_path = Path(output_value).expanduser()
    if not output_path.is_absolute():
        output_path = receipt_path.parent / output_path
    if output_path.is_symlink() or not output_path.is_file() or output_path.name != "pixels.f32le":
        raise ControlError("native90 output must be a regular pixels.f32le file")
    if output_path.stat().st_size != PIXELS360_BYTES:
        raise ControlError("native90 output extent mismatch")
    output_sha = base._sha256_file(output_path)
    base._require_digest(output_sha, receipt["output_sha256"], "native90 output")
    return output_path.resolve(strict=True), receipt, actual_receipt_sha


def preflight(args) -> dict[str, object]:
    # Retain the original 60-latent receipt, saved noise/RoPE, upstream, cache,
    # and weight authentication without altering the established helper.
    admission = base.preflight(args)
    custom_path = args.custom_latents.expanduser()
    if not custom_path.is_absolute():
        custom_path = Path.cwd() / custom_path
    if custom_path.is_symlink() or not custom_path.is_file():
        raise ControlError("--custom-latents must be a regular non-symlink F32 file")
    raw = base._read_regular(custom_path, "custom 90-latent artifact", max_bytes=LATENT90_BYTES)
    custom_sha = validate_latent_artifact_bytes(raw, args.custom_latents_sha256.lower())
    channels = LATENT90_SHAPE[1]
    frame_bytes = math.prod(LATENT90_SHAPE[3:]) * 4
    original_prefix = bytearray()
    bank_path = admission["bank_path"]
    source_frames = base.LATENT_BANK_SHAPE[2]
    source_channel_bytes = source_frames * frame_bytes
    prefix_channel_bytes = LATENT60_SHAPE[2] * frame_bytes
    with bank_path.open("rb") as stream:
        for channel in range(channels):
            stream.seek(channel * source_channel_bytes)
            chunk = stream.read(prefix_channel_bytes)
            if len(chunk) != prefix_channel_bytes:
                raise ControlError("native bank is truncated while checking custom90 prefix")
            original_prefix.extend(chunk)
    validate_cyclic_tail_bytes(
        raw, channels=channels, frames=LATENT90_SHAPE[2], frame_bytes=frame_bytes,
        prefix=bytes(original_prefix), prefix_frames=60, cycle_start=50,
        cycle_frames=10, repeats=3,
    )
    prefix_sha = _channel_prefix_hash(raw, channels=channels, frames=90,
                                      frame_bytes=frame_bytes, prefix_frames=60)
    base._require_digest(prefix_sha, admission["prefix_sha256"], "custom90 common 60-latent prefix")
    native_pixels, native_receipt, native_receipt_sha = _read_native90(
        args, custom_path, custom_sha
    )
    admission.update({
        "custom90_path": custom_path.resolve(strict=True),
        "custom90_sha256": custom_sha,
        "custom90_prefix_sha256": prefix_sha,
        "native90_pixels_path": native_pixels,
        "native90_receipt": native_receipt,
        "native90_receipt_sha256": native_receipt_sha,
    })
    return admission


def _runtime_preflight(torch, block_reference, arch, cache_bytes: int) -> dict[str, object]:
    if torch.__version__ != "2.6.0":
        raise ControlError(f"decoder execution requires Torch 2.6.0; got {torch.__version__}")
    if not torch.backends.mps.is_available():
        raise ControlError("MPS unavailable; CPU fallback is prohibited")
    if os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK") != "0":
        raise ControlError("set PYTORCH_ENABLE_MPS_FALLBACK=0 for decoder execution")
    grid = (10, 18, 32)
    base_peak = block_reference._estimated_peak_bytes(
        grid, arch, layers=36, resident_layers=1,
        capture_stages=block_reference.FULL_TARGET_STAGES,
        full36_capture_policy=block_reference.THIRTY_SIX_CAPTURE_POLICY,
    )
    physical = block_reference._physical_memory_bytes()
    allocated = int(torch.mps.driver_allocated_memory())
    hidden_per_window = math.prod(grid) * arch.hidden_size * 2
    persistent_windows = len(WINDOW_STARTS) * hidden_per_window
    large_tensors = (PIXELS360_BYTES + math.prod(PIXELS360_SHAPE) * 2
                     + PIXELS360_BYTES + math.prod(PIXELS240_SHAPE) * 4)
    one_block = 402_653_184
    estimate = base_peak + persistent_windows + large_tensors + one_block + 2 * base.GIB
    reserve = int(physical * base.RESERVE_FRACTION)
    if estimate > base.MAX_PROCESS_BYTES:
        raise ControlError("estimated 90-latent decoder footprint exceeds the 18 GiB guard cap")
    if physical - allocated - estimate < reserve:
        raise ControlError("estimated 90-latent footprint leaves less than 10% physical memory free")
    return {
        "physical_bytes": physical, "mps_allocated_bytes": allocated,
        "estimated_peak_bytes": estimate, "estimated_peak_gib": estimate / base.GIB,
        "process_cap_bytes": base.MAX_PROCESS_BYTES, "reserve_floor_bytes": reserve,
        "cache_payload_bytes_cpu_mapped": cache_bytes,
        "windows": len(WINDOW_STARTS), "persistent_window_bytes": persistent_windows,
        "guard": "wan_memory_guard.py --memory-accounting file-cache --runtime-floor-pct 10 --max-footprint-mib 18432 --max-swap-growth-mib 64",
    }


def execute(admission: Mapping[str, object], *, historical_controls: bool = True) -> dict[str, object]:
    started_at = time.perf_counter()
    import torch

    support = Path(__file__).resolve().parent
    if str(support) not in sys.path:
        sys.path.insert(0, str(support))
    import hy_nomas_pixelumm_real_input as real_input
    import hy_nomas_pixelumm_reference as block_reference
    import hy_nomas_pixelumm_checkpoint as checkpoint_inventory
    import hy_nomas_pixelumm_head_checkpoint as head_checkpoint
    import pixelumm_generation_reference as pixelumm_reference

    upstream = admission["upstream"]
    source, model_sha = block_reference.load_upstream(upstream)
    if model_sha != base.MODEL_SHA256:
        raise ControlError("upstream model source changed after CPU preflight")
    _, _, autoencoder_source = real_input._load_modules(upstream)
    if base._sha256_file(Path(autoencoder_source.__file__)) != base.BLENDER_SOURCE_SHA256:
        raise ControlError("pinned PixelUMM blender source changed after CPU preflight")
    try:
        block_reference.validate_preview_oracle_runtime((10, 18, 32), "mps")
    except block_reference.ReferenceError as error:
        raise ControlError(str(error)) from error
    device = torch.device("mps")
    arch = source.QWEN3_8B_GEN
    memory = _runtime_preflight(torch, block_reference, arch,
                                admission["cache_payload_path"].stat().st_size)
    raw_header = (admission["cache_payload_path"].parent / "pixelumm-header.json").read_bytes()
    try:
        checkpoint_inventory.parse_pinned_header(raw_header)
    except checkpoint_inventory.InventoryError as error:
        raise ControlError(f"pinned checkpoint header rejected: {error}") from error
    real_input.STEM_SUBSET_PATH = admission["stem_path"]
    stem = real_input._load_stem(source, device)
    from safetensors import safe_open
    try:
        with safe_open(str(admission["head_path"]), framework="pt", device="cpu") as handle:
            if set(handle.keys()) != set(head_checkpoint.HEAD_TENSORS):
                raise ControlError("head safetensors keys differ from pinned inventory")
            head_state = {name: handle.get_tensor(name) for name in head_checkpoint.HEAD_TENSORS}
    except ControlError:
        raise
    except Exception as error:
        raise ControlError("cannot load pinned PixelUMM head safetensors") from error
    expected_head = {
        "decoder.final_norm.weight": (4096,),
        "decoder.head.linear.bias": (3072,),
        "decoder.head.linear.weight": (3072, 4096),
        "decoder.head.norm.weight": (4096,),
    }
    if any(value.dtype != torch.float32 or tuple(value.shape) != expected_head[name]
           for name, value in head_state.items()):
        raise ControlError("head safetensors dtype or shape differs from pinned source")
    final_norm = source.RMSNorm(arch.hidden_size, arch.rms_norm_eps).to(device=device, dtype=torch.bfloat16)
    pixel_head = source.PixelHead(arch.hidden_size).to(device=device, dtype=torch.bfloat16)
    head_destinations = {
        "decoder.final_norm.weight": final_norm.weight,
        "decoder.head.norm.weight": pixel_head.norm.weight,
        "decoder.head.linear.weight": pixel_head.linear.weight,
        "decoder.head.linear.bias": pixel_head.linear.bias,
    }
    with torch.no_grad():
        for name, parameter in head_destinations.items():
            parameter.copy_(head_state[name].to(device=device, dtype=torch.bfloat16))
    del head_state

    latent_cpu = torch.from_file(str(admission["custom90_path"]), shared=False,
                                 size=math.prod(LATENT90_SHAPE), dtype=torch.float32).view(LATENT90_SHAPE)
    latent = latent_cpu.to(device=device)
    del latent_cpu
    baseline_cpu = torch.from_file(str(admission["pixels_path"]), shared=False,
                                   size=math.prod(PIXELS240_SHAPE), dtype=torch.float32).view(PIXELS240_SHAPE)
    baseline_pixels = baseline_cpu.to(device=device)
    del baseline_cpu
    native90_cpu = torch.from_file(str(admission["native90_pixels_path"]), shared=False,
                                   size=math.prod(PIXELS360_SHAPE), dtype=torch.float32).view(PIXELS360_SHAPE)
    native90 = native90_cpu.to(device=device)
    del native90_cpu

    payload_bytes = admission["cache_payload_path"].stat().st_size
    cache_bytes = torch.from_file(str(admission["cache_payload_path"]), shared=False,
                                  size=payload_bytes, dtype=torch.uint8)
    payload = torch.from_file(str(admission["payload_path"]), shared=False,
                              size=admission["payload_path"].stat().st_size // 4,
                              dtype=torch.float32)
    noise_record = admission["noise_record"]
    noise = payload.narrow(0, noise_record["offset"] // 4,
                           noise_record["nbytes"] // 4).view(base.NOISE_SHAPE)
    noise = noise.narrow(2, 0, NOISE360_SHAPE[2]).to(device=device)
    cos_record = admission["rope_records"]["decoder.rope_cos"]
    sin_record = admission["rope_records"]["decoder.rope_sin"]
    cos = payload.narrow(0, cos_record["offset"] // 4,
                         cos_record["nbytes"] // 4).view(base.DECODER_ROPE_SHAPE).to(device=device)
    sin = payload.narrow(0, sin_record["offset"] // 4,
                         sin_record["nbytes"] // 4).view(base.DECODER_ROPE_SHAPE).to(device=device)
    del payload

    plan, inverse_cpu, _ = block_reference.make_attention_plan(source, (10, 18, 32), tile=8)
    permutation = torch.tensor(plan.permutation, dtype=torch.long, device=device)
    inverse = inverse_cpu.to(device=device)
    condition = type("_ConditionProjection", (), {})()
    condition.cond_proj = stem.cond_proj
    condition.arch = arch
    windows = []
    blender = autoencoder_source._TemporalBlender(total_frames=360)
    raw_old_tail = None
    with torch.inference_mode():
        with torch.autocast(device_type="mps", dtype=torch.bfloat16):
            for start in WINDOW_STARTS:
                latent_window = latent[:, :, start:start + WINDOW_LATENTS]
                noise_window = noise[:, :, start * 4:(start + WINDOW_LATENTS) * 4]
                xt_tokens = stem.embed(source.patchify(noise_window.to(stem.embed.weight.dtype)))
                condition_tokens = source.DCAEVPixelUMMDecoder._condition_tokens(
                    condition, latent_window.to(stem.cond_proj.weight.dtype)
                )
                windows.append((xt_tokens + condition_tokens).index_select(1, permutation))
                del latent_window, noise_window, xt_tokens, condition_tokens
            for block_index in range(36):
                layer = base._cache_layer(torch, source, arch, cache_bytes,
                                          admission["cache_manifest"], block_index, device)
                for index, current in enumerate(windows):
                    windows[index], captures = block_reference.run_layer_reference(layer, current, cos, sin, plan)
                    if captures:
                        raise ControlError("unexpected intermediate capture was retained")
                del layer
        for index, start in enumerate(WINDOW_STARTS):
            with torch.autocast(device_type="mps", dtype=torch.bfloat16):
                decoder_tokens = final_norm(windows[index]).index_select(1, inverse)
            prediction = source.unpatchify(pixel_head(decoder_tokens), (10, 18, 32))
            if start == 50:
                if prediction.dtype is not torch.bfloat16 or prediction.shape[2] != 40:
                    raise ControlError("window-50 A/A prediction has unexpected dtype or extent")
                raw_old_tail = prediction[:, :, -20:].contiguous()
            blender.add(prediction, start * 4)
            del decoder_tokens, prediction
    raw = blender.result()
    if raw.device.type != "mps" or tuple(raw.shape) != PIXELS360_SHAPE or raw.dtype is not torch.bfloat16:
        raise ControlError("upstream 90-latent PixelUMM result has unexpected dtype/device/geometry")
    display = pixelumm_reference.postprocess_bf16(torch, raw)
    upstream_t1 = display[:, :, T1_CROP[0]:T1_CROP[1]]
    native_t1 = native90[:, :, T1_CROP[0]:T1_CROP[1]]
    baseline_unaffected = display[:, :, PREFIX_CROP[0]:PREFIX_CROP[1]]
    baseline_expected = baseline_pixels[:, :, PREFIX_CROP[0]:PREFIX_CROP[1]]
    tail_display = pixelumm_reference.postprocess_bf16(torch, raw_old_tail)
    tail_expected = baseline_pixels[:, :, TAIL_CROP[0]:TAIL_CROP[1]]
    equal_prefix = (baseline_unaffected.contiguous().view(torch.int32)
                    == baseline_expected.contiguous().view(torch.int32)).all()
    equal_old_tail = (tail_display.contiguous().view(torch.int32)
                      == tail_expected.contiguous().view(torch.int32)).all()
    actual, expected = upstream_t1.reshape(-1), native_t1.reshape(-1)
    maxima, sums, mismatches, finite = [], [], [], []
    for begin in range(0, actual.numel(), 1 << 20):
        stop = min(begin + (1 << 20), actual.numel())
        left, right = actual[begin:stop], expected[begin:stop]
        delta = (left - right).abs()
        maxima.append(delta.max())
        sums.append(delta.sum())
        mismatches.append((delta > (ATOL + RTOL * right.abs())).any())
        finite.append(torch.isfinite(left).all().logical_and(torch.isfinite(right).all()))
    eye = upstream_t1[:, :, 1:, EYE_XYXY[1]:EYE_XYXY[3], EYE_XYXY[0]:EYE_XYXY[2]]
    eye_prev = upstream_t1[:, :, :-1, EYE_XYXY[1]:EYE_XYXY[3], EYE_XYXY[0]:EYE_XYXY[2]]
    eye_mae = (eye - eye_prev).abs().mean(dim=(0, 1, 3, 4))
    metrics = torch.stack((
        torch.stack(maxima).max(), torch.stack(sums).sum() / actual.numel(),
        torch.stack(mismatches).any().to(torch.float32),
        torch.stack(finite).all().logical_not().to(torch.float32),
        torch.isfinite(raw).all().logical_not().to(torch.float32),
        equal_prefix.logical_not().to(torch.float32),
        equal_old_tail.logical_not().to(torch.float32),
    ))
    packed = torch.cat((upstream_t1.contiguous().view(-1), eye_mae, metrics))
    cpu = packed.cpu()  # sole MPS synchronization and device-to-host transfer
    del packed, metrics, latent, noise, native_t1, native90, baseline_pixels
    del display, raw, raw_old_tail, tail_display, windows, cache_bytes, cos, sin
    pixel_count = math.prod((1, 3, 120, 288, 512))
    output_array = cpu[:pixel_count].numpy().astype("<f4", copy=False)
    eye_values = cpu[pixel_count:pixel_count + 119].tolist()
    max_abs, mean_abs, mismatch, nonfinite, raw_nonfinite, prefix_changed, old_tail_changed = (
        cpu[pixel_count + 119:].tolist()
    )
    out_dir = admission["output_dir"]
    out_dir.mkdir(mode=0o700)
    output_path = out_dir / "upstream-t1-display.f32le"
    output_sha = base._write_fresh(output_path, output_array)
    gates = {
        "native90_t1_within_tolerance": not bool(mismatch) and not bool(nonfinite) and not bool(raw_nonfinite),
        "baseline_unaffected_100_frames_bitexact": not bool(prefix_changed),
        "baseline_window50_unblended_tail20_bitexact": not bool(old_tail_changed),
    }
    result = {
        "schema": SCHEMA,
        "experiment_scope": "synthetic cyclic future context plus changed local temporal RoPE and overlap blending; no DiT rerun; not a missing-context-only causal test",
        "synthetic_future_context": {"source_latent_cycle": [50, 60], "repeats": 3,
                                     "custom_latent_frames": 90, "prefix60_bitexact": True},
        "upstream_revision": base.UPSTREAM_REVISION,
        "upstream_model_sha256": model_sha,
        "native_baseline_receipt_sha256": admission["receipt_sha256"],
        "native90_receipt_sha256": admission["native90_receipt_sha256"],
        "native90_output_sha256": admission["native90_receipt"]["output_sha256"],
        "custom90_latent_sha256": admission["custom90_sha256"],
        "common_prefix_sha256": admission["custom90_prefix_sha256"],
        "decoder_noise_record_sha256": admission["noise_record"]["sha256"],
        "decoder_noise_source_shape": list(base.NOISE_SHAPE),
        "decoder_noise_selected_axis2": [0, 360],
        "decoder_rope_cos_sha256": base.ROPE_COS_SHA256,
        "decoder_rope_sin_sha256": base.ROPE_SIN_SHA256,
        "decoder_cache_manifest_sha256": admission["cache_manifest_sha256"],
        "decoder_cache_payload_sha256": admission["cache_payload_sha256"],
        "stem_weights_sha256": base.STEM_SHA256,
        "head_weights_sha256": base.HEAD_SHA256,
        "decoder": {"latent_shape": list(LATENT90_SHAPE), "window_starts": list(WINDOW_STARTS),
                    "window_latents": WINDOW_LATENTS, "overlap_latents": OVERLAP_LATENTS,
                    "blocks": 36, "dit_rerun": False,
                    "blend": "pinned upstream _TemporalBlender cosine overlap"},
        "comparison": {"native_pixel_crop": list(T1_CROP), "shape": [1, 3, 120, 288, 512],
                       "atol": ATOL, "rtol": RTOL, "max_abs_error": max_abs,
                       "mean_abs_error": mean_abs,
                       "any_value_outside_tolerance": bool(mismatch),
                       "all_values_finite": not bool(nonfinite or raw_nonfinite),
                       "result": "match" if gates["native90_t1_within_tolerance"] else "mismatch"},
        "baseline_controls": {"unaffected_pixel_range": list(PREFIX_CROP),
                               "unaffected_bitexact": gates["baseline_unaffected_100_frames_bitexact"],
                               "window50_raw_tail_pixel_range": list(TAIL_CROP),
                               "window50_unblended_tail_bitexact": gates["baseline_window50_unblended_tail20_bitexact"]},
        "eye_roi_adjacent_frame_rgb_mae": {"xyxy": list(EYE_XYXY), "transition_count": 119,
                                            "values": eye_values,
                                            "peak": max(eye_values),
                                            "peak_transition_t1_index": eye_values.index(max(eye_values)) + 1},
        "gates": gates,
        "memory_preflight": memory,
        "runtime": {"torch_version": torch.__version__,
                    "pytorch_enable_mps_fallback": os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK"),
                    "wall_seconds": time.perf_counter() - started_at,
                    "intermediate_readbacks": 0, "explicit_sync_calls": 0,
                    "terminal_transfer_count": 1},
        "upstream_t1_display_path": str(output_path),
        "upstream_t1_display_sha256": output_sha,
        "single_terminal_mps_sync_and_pack_transfer": True,
    }
    if not historical_controls:
        # A caller authenticates a new same-arm 60-latent prefix and native90
        # output. Historical portrait equality is then neither required nor
        # claimed: the upstream/native replay is the numerical control.
        result["schema"] = "cogni_ml.hy_nomas_portrait_history_decoder_control/v1"
        result["experiment_scope"] = (
            "same-arm native/upstream PixelUMM replay of authenticated T0+T1 "
            "and a cyclic synthetic future; no DiT rerun or causal decoder claim"
        )
        result["historical_fixture_receipt_sha256"] = result.pop("native_baseline_receipt_sha256")
        result["same_arm_prefix60_sha256"] = result.pop("common_prefix_sha256")
        result["synthetic_future_context"] = {
            "source_latent_cycle": [50, 60], "repeats": 3,
            "custom_latent_frames": 90, "same_arm_prefix60_bitexact": True,
        }
        result.pop("baseline_controls")
        result["gates"] = {"native90_t1_within_tolerance": gates["native90_t1_within_tolerance"]}
        result["history_probe"] = admission["history_probe"]
    receipt_path = out_dir / "receipt.json"
    with receipt_path.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    return result


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--preflight", action="store_true")
    mode.add_argument("--execute", action="store_true")
    parser.add_argument("--run-root", type=Path, default=base.DEFAULT_ROOT)
    parser.add_argument("--upstream", type=Path, default=base.DEFAULT_UPSTREAM)
    parser.add_argument("--weight-cache-manifest", type=Path, default=base.DEFAULT_CACHE)
    parser.add_argument("--stem-weights", type=Path, default=base.DEFAULT_STEM)
    parser.add_argument("--head-weights", type=Path, default=base.DEFAULT_HEAD)
    parser.add_argument("--custom-latents", type=Path, required=True)
    parser.add_argument("--custom-latents-sha256", required=True)
    parser.add_argument("--native-receipt", type=Path, required=True)
    parser.add_argument("--native-sha256", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    # Adapt the established 60-latent preflight argument contract without
    # passing the 90-frame artifact through its deliberately strict 60-frame slot.
    args.input_latents = None
    args.input_latents_sha256 = None
    try:
        admission = preflight(args)
        if not args.execute:
            print(json.dumps({"schema": SCHEMA, "mode": "preflight", "admission": "passed",
                              "custom90_latent_sha256": admission["custom90_sha256"],
                              "common_prefix_sha256": admission["custom90_prefix_sha256"],
                              "native90_receipt_sha256": admission["native90_receipt_sha256"],
                              "native90_output_sha256": admission["native90_receipt"]["output_sha256"],
                              "window_starts": list(WINDOW_STARTS),
                              "t1_pixel_crop": list(T1_CROP),
                              "decoder_noise_selected_axis2": [0, 360],
                              "output_dir": str(admission["output_dir"]),
                              "source_metadata": admission["source_metadata"],
                              "dit_rerun": False}, indent=2))
            return 0
        result = execute(admission)
    except (ControlError, OSError, RuntimeError, ValueError, KeyError, TypeError) as error:
        print(f"portrait tail decoder control failed: {error}", file=sys.stderr)
        return 2
    passed = all(result["gates"].values()) and result["comparison"]["result"] == "match"
    print(json.dumps({"output": result["upstream_t1_display_path"],
                      "gates": result["gates"], "comparison": result["comparison"],
                      "eye_roi_peak": result["eye_roi_adjacent_frame_rgb_mae"]["peak"],
                      "memory_preflight": result["memory_preflight"]}, indent=2))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
