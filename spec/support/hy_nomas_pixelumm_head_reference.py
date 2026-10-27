#!/usr/bin/env python3
"""Produce head-only PixelUMM reference fixtures from authenticated block35 edges.

The oracle imports the pinned upstream model source and executes only its
global final RMSNorm, inverse tile permutation, PixelHead RMSNorm/Linear, and
unpatchify on MPS. It never constructs or runs the 36-block decoder.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import mmap
import os
from pathlib import Path
import platform
import struct
import subprocess
import sys
from typing import Mapping

import torch

import hy_nomas_pixelumm_checkpoint as checkpoint
import hy_nomas_pixelumm_head_checkpoint as head_checkpoint
import hy_nomas_pixelumm_reference as block_reference

UPSTREAM_REVISION = "911cea6458a89c2663513ff7099561ec190dd491"
UPSTREAM_MODEL_PATH = "hymm/models/vae/pixelumm_decoder/model.py"
UPSTREAM_MODEL_SHA256 = "bea01a8490f8e9a4076da83d4f9fbb3cba7bd6701df5c28fb2e2c04f00131eec"
SCHEMA = "cogni_ml.pixelumm_head_reference/v1"
ROOT = head_checkpoint.ROOT
SOURCE_DIR = ROOT / "source"
RESULTS_DIR = ROOT / "results"
DEFAULT_UPSTREAM = Path(
    "/opt/mldata_ssd/cogni-ml/video-scene-20261019/upstream/hy-nomas-20261024"
)
FULL36_RESULTS_DIR = Path(
    "/opt/mldata_ssd/cogni-ml/hy-nomas-pixelumm-36-20261027/results"
).resolve()
GRID = (6, 44, 80)
TILE_TOKENS = 8
HIDDEN = 4096
TUBE_DIM = 3072
TOKENS = math.prod(GRID)
ATOL = RTOL = 2e-5
GIB = 1024**3
DEFAULT_MAX_MPS_BYTES = 4 * GIB
TEMPORARY_MARGIN_BYTES = 1 * GIB
CAPTURE_ORDER = (
    "final_norm_tile_major", "restored_thw", "head_norm_thw",
    "linear_thw", "output_thw",
)
HEAD_INPUT_NBYTES = TOKENS * HIDDEN * 2
HEAD_WEIGHT_NBYTES = head_checkpoint.HEAD_BF16_BYTES
HEAD_LINEAR_NBYTES = TOKENS * TUBE_DIM * 2
HEAD_OUTPUT_NBYTES = 1 * 3 * (GRID[0] * 4) * (GRID[1] * 16) * (GRID[2] * 16) * 2
HEAD_STATIC_RETAINED_BYTES = (
    HEAD_INPUT_NBYTES + HEAD_WEIGHT_NBYTES
    + 3 * HEAD_INPUT_NBYTES + HEAD_LINEAR_NBYTES + HEAD_OUTPUT_NBYTES
)
HEAD_ESTIMATED_PEAK_BYTES = HEAD_STATIC_RETAINED_BYTES + TEMPORARY_MARGIN_BYTES

_FULL36_INPUTS = {
    42: {
        "manifest": "thirty-six-reference-full42-v4.json",
        "manifest_sha256": "6efc8a15dc83c3e5e0d1cff824522ea8e98e3649d7178d143b8aaf35f2a0e592",
        "payload": "thirty-six-reference-full42-v4.bin",
        "payload_bytes": 20_661_356_544,
        "payload_sha256": "d3879fa5f1bcf0aa3227127f312b6190e47ce97eef36e2aaa3952cb41975bb89",
        "record_sha256": "d70e09efa819ae38e5f3fba9f58ccde4375ca1e220e966beb573d3d4a5ca15b4",
        "record_offset": 20_488_341_504,
    },
    43: {
        "manifest": "thirty-six-reference-full43-v2.json",
        "manifest_sha256": "429c010be440de4834bf5f0a0a8d9e658579d808d5e8f4b68605ebbf088ff7c8",
        "payload": "thirty-six-reference-full43-v2.bin",
        "payload_bytes": 20_661_356_544,
        "payload_sha256": "a976d84925fd262e81c047cc0436cefd40f6cba4baa7947e8887d6ca37b9a963",
        "record_sha256": "523a6f7af4f2d14c576859df2b9ee9414f999e5ec62bc2dad971a046dff80192",
        "record_offset": 20_488_341_504,
    },
}


class HeadReferenceError(ValueError):
    """The source, capture, head subset, geometry, or runtime is unsupported."""


def _digest(value: object) -> bool:
    return (isinstance(value, str) and len(value) == 64
            and all(char in "0123456789abcdef" for char in value))


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def independent_tile_major_permutation(
    grid: tuple[int, int, int], tile: int = TILE_TOKENS,
) -> tuple[int, ...]:
    """Enumerate tile-major positions directly, independently of sort keys."""
    if (len(grid) != 3 or any(type(value) is not int or value <= 0 for value in grid)
            or type(tile) is not int or tile <= 0):
        raise HeadReferenceError("grid and tile must contain positive integers")
    frames, height, width = grid
    result = []
    for tile_h in range((height + tile - 1) // tile):
        for tile_w in range((width + tile - 1) // tile):
            for frame in range(frames):
                for h in range(tile_h * tile, min((tile_h + 1) * tile, height)):
                    for w in range(tile_w * tile, min((tile_w + 1) * tile, width)):
                        result.append((frame * height + h) * width + w)
    if len(result) != math.prod(grid) or len(set(result)) != len(result):
        raise HeadReferenceError("independent tile-major enumeration is not a bijection")
    return tuple(result)


def independent_inverse(permutation: tuple[int, ...]) -> tuple[int, ...]:
    """Return THW-index -> tile-major-index mapping and reject false positives."""
    count = len(permutation)
    if (not count or any(type(value) is not int for value in permutation)
            or set(permutation) != set(range(count))):
        raise HeadReferenceError("tile permutation must be a complete bijection")
    inverse = [0] * count
    for tile_major_index, thw_index in enumerate(permutation):
        inverse[thw_index] = tile_major_index
    return tuple(inverse)


def permutation_sha256(permutation: tuple[int, ...]) -> str:
    if any(type(value) is not int or not -(2**31) <= value < 2**31 for value in permutation):
        raise HeadReferenceError("permutation values must fit signed int32")
    raw = struct.pack(f"<{len(permutation)}i", *permutation)
    return hashlib.sha256(raw).hexdigest()


def validate_tile_order(
    permutation, inverse, grid: tuple[int, int, int], tile: int = TILE_TOKENS,
) -> str:
    """Compare a source permutation and inverse to independent nested-loop maps."""
    expected = independent_tile_major_permutation(grid, tile)
    actual = tuple(permutation)
    actual_inverse = tuple(inverse)
    if (any(type(value) is not int for value in actual)
            or any(type(value) is not int for value in actual_inverse)):
        raise HeadReferenceError("source tile permutation and inverse must contain integer indices")
    expected_inverse = independent_inverse(expected)
    if actual != expected or actual_inverse != expected_inverse:
        raise HeadReferenceError("source tile permutation or inverse differs from independent mapping")
    return permutation_sha256(expected)


def make_head_order(source, grid=GRID, tile: int = TILE_TOKENS):
    """Derive the upstream inverse without constructing a full attention mask."""
    grid = tuple(grid)
    if grid != GRID:
        raise HeadReferenceError("MPS oracle input is pinned to the full [6,44,80] grid")
    # This existing adapter invokes the pinned helper on small grids and uses
    # its exact source permutation formula for the full grid to avoid building
    # a quadratic FlexAttention mask.
    plan, inverse, _ = block_reference.make_attention_plan(source, grid, tile)
    digest = validate_tile_order(plan.permutation, inverse.tolist(), grid, tile)
    if plan.permutation_sha256 != digest:
        raise HeadReferenceError("source permutation digest differs from independent mapping")
    return plan, inverse, digest


def expected_fixture_payload_bytes(grid=GRID) -> int:
    grid = tuple(grid)
    count = math.prod(grid)
    input_bytes = count * HIDDEN * 2
    linear_bytes = count * TUBE_DIM * 2
    output_bytes = 3 * grid[0] * 4 * grid[1] * 16 * grid[2] * 16 * 2
    return (
        input_bytes + HEAD_WEIGHT_NBYTES + 3 * input_bytes + linear_bytes + output_bytes
    )


def estimate_mps_peak_bytes(grid=GRID) -> int:
    """Retained head tensors plus a conservative 1 GiB kernel/workspace margin."""
    return expected_fixture_payload_bytes(grid) + TEMPORARY_MARGIN_BYTES


def memory_preflight(
    *, physical_bytes: int, driver_allocated_bytes: int,
    recommended_max_bytes: int, estimated_peak_bytes: int,
    max_mps_bytes: int = DEFAULT_MAX_MPS_BYTES,
) -> dict[str, int | float]:
    """Enforce the allocator cap and reserve at least 10% of physical memory."""
    positive = (physical_bytes, recommended_max_bytes, estimated_peak_bytes, max_mps_bytes)
    if any(type(value) is not int or value <= 0 for value in positive):
        raise HeadReferenceError("memory preflight requires positive integer byte counts")
    if (type(driver_allocated_bytes) is not int or driver_allocated_bytes < 0
            or driver_allocated_bytes > physical_bytes):
        raise HeadReferenceError("driver allocated bytes must be within physical memory")
    floor = physical_bytes // 10
    available = physical_bytes - driver_allocated_bytes - floor
    if estimated_peak_bytes > max_mps_bytes:
        raise HeadReferenceError("estimated MPS peak exceeds the requested head allocator cap")
    if max_mps_bytes > available:
        raise HeadReferenceError("MPS allocator cap would cross the required 10% physical-memory floor")
    if max_mps_bytes > recommended_max_bytes:
        raise HeadReferenceError("MPS allocator cap exceeds the recommended device limit")
    fraction = max_mps_bytes / recommended_max_bytes
    if not 0.0 < fraction <= 1.0:
        raise HeadReferenceError("MPS allocator fraction is outside (0,1]")
    return {
        "physical_bytes": physical_bytes,
        "driver_allocated_bytes": driver_allocated_bytes,
        "recommended_max_bytes": recommended_max_bytes,
        "estimated_peak_bytes": estimated_peak_bytes,
        "max_mps_bytes": max_mps_bytes,
        "required_floor_bytes": floor,
        "headroom_after_cap_bytes": available - max_mps_bytes,
        "mps_memory_fraction": fraction,
    }


def _source_manifest_contract(manifest: Mapping, seed: int, manifest_sha256: str) -> dict:
    """Validate the immutable full36 fixture header and selected output record."""
    try:
        pinned = _FULL36_INPUTS[seed]
    except KeyError as error:
        raise HeadReferenceError("head fixtures admit only full36 seeds 42 and 43") from error
    if not _digest(manifest_sha256) or manifest_sha256 != pinned["manifest_sha256"]:
        raise HeadReferenceError("full36 input manifest digest is not an admitted seed fixture")
    if (manifest.get("schema") != "cogni_ml.pixelumm_thirty_six_block_reference/v1"
            or manifest.get("upstream_revision") != UPSTREAM_REVISION
            or manifest.get("upstream_model_path") != UPSTREAM_MODEL_PATH
            or manifest.get("upstream_model_sha256") != UPSTREAM_MODEL_SHA256
            or manifest.get("checkpoint_header_sha256") != checkpoint.HEADER_SHA256
            or manifest.get("grid") != list(GRID)
            or type(manifest.get("tile_tokens")) is not int
            or manifest.get("tile_tokens") != TILE_TOKENS
            or manifest.get("token_order") != "tile_major"
            or manifest.get("captures_are_tile_major") is not True
            or manifest.get("runtime_dtype") != "BF16"
            or manifest.get("full36_capture_policy") != "two_bank_survivors"
            or manifest.get("permutation_sha256") != "4e81c6026e99ee2c17c176c4394682ca632602c198c3065ad311a83a3839cf33"
            or manifest.get("payload") != pinned["payload"]
            or type(manifest.get("payload_bytes")) is not int
            or manifest.get("payload_bytes") != pinned["payload_bytes"]
            or manifest.get("payload_sha256") != pinned["payload_sha256"]):
        raise HeadReferenceError("full36 source metadata does not match the pinned head input")
    if (manifest.get("comparison") != {"atol": ATOL, "rtol": RTOL}
            or not isinstance(manifest.get("input_provenance"), dict)
            or manifest["input_provenance"].get("kind") != "real_latent_xt_pixelumm_stem"
            or not isinstance(manifest.get("input_lineage"), dict)):
        raise HeadReferenceError("full36 input provenance or comparison metadata is malformed")
    try:
        block_reference._validate_full36_capture_inventory(
            policy=manifest.get("full36_capture_policy"),
            capture_order=manifest.get("capture_order"),
            blocks=manifest.get("blocks"),
        )
    except block_reference.ReferenceError as error:
        raise HeadReferenceError(f"full36 capture order is invalid: {error}") from error
    captures = manifest.get("captures")
    if not isinstance(captures, dict) or "block35.output" not in captures:
        raise HeadReferenceError("full36 fixture lacks the final block35.output capture")
    record = captures["block35.output"]
    expected_record = {
        "dtype": "BF16", "shape": [1, TOKENS, HIDDEN],
        "offset": pinned["record_offset"], "nbytes": HEAD_INPUT_NBYTES,
        "sha256": pinned["record_sha256"],
    }
    if record != expected_record:
        raise HeadReferenceError("block35.output record has a false-positive dtype, shape, extent, or digest")
    return {
        "manifest_sha256": manifest_sha256,
        "payload": pinned["payload"],
        "payload_bytes": pinned["payload_bytes"],
        "payload_sha256": pinned["payload_sha256"],
        "record": expected_record,
    }


def _load_full36_input(seed: int, *, device: torch.device) -> tuple[torch.Tensor, dict]:
    pinned = _FULL36_INPUTS.get(seed)
    if pinned is None:
        raise HeadReferenceError("head fixtures admit only full36 seeds 42 and 43")
    manifest_path = FULL36_RESULTS_DIR / pinned["manifest"]
    if manifest_path.is_symlink() or not manifest_path.is_file():
        raise HeadReferenceError("pinned full36 input manifest is missing or is a symlink")
    if manifest_path.resolve() != manifest_path:
        raise HeadReferenceError("full36 input manifest must use its canonical path")
    raw_manifest = manifest_path.read_bytes()
    if len(raw_manifest) > 2 * 1024 * 1024:
        raise HeadReferenceError("full36 input manifest exceeds the metadata limit")
    manifest_sha = hashlib.sha256(raw_manifest).hexdigest()
    try:
        manifest = checkpoint.parse_header(raw_manifest)
    except checkpoint.InventoryError as error:
        raise HeadReferenceError(f"cannot parse full36 input manifest: {error}") from error
    source = _source_manifest_contract(manifest, seed, manifest_sha)
    payload_path = manifest_path.parent / source["payload"]
    if payload_path.is_symlink() or not payload_path.is_file():
        raise HeadReferenceError("pinned full36 input payload is missing or is a symlink")
    before = payload_path.stat()
    if before.st_size != source["payload_bytes"]:
        raise HeadReferenceError("pinned full36 input payload size changed")
    record = source["record"]
    with payload_path.open("rb") as stream:
        mapped = mmap.mmap(stream.fileno(), 0, access=mmap.ACCESS_COPY)
    try:
        record_bytes = memoryview(mapped)[record["offset"]:record["offset"] + record["nbytes"]]
        record_sha = hashlib.sha256(record_bytes).hexdigest()
        del record_bytes
        if record_sha != record["sha256"]:
            raise HeadReferenceError("full36 block35.output bytes do not match the pinned record digest")
        byte_tensor = torch.frombuffer(
            mapped, dtype=torch.uint8, count=record["nbytes"], offset=record["offset"]
        )
        cpu = byte_tensor.view(torch.bfloat16).reshape(record["shape"])
        tensor = cpu.to(device=device)
        del cpu, byte_tensor
    finally:
        mapped.close()
    after = payload_path.stat()
    if ((before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns)
            != (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns)):
        raise HeadReferenceError("full36 input payload changed while reading block35.output")
    provenance = {
        "kind": "authenticated_full36_block35_output",
        "seed": seed,
        "manifest": manifest_path.name,
        "manifest_sha256": source["manifest_sha256"],
        "payload": source["payload"],
        "payload_sha256": source["payload_sha256"],
        "payload_bytes": source["payload_bytes"],
        "capture": "block35.output",
        "capture_sha256": record["sha256"],
        "capture_offset": record["offset"],
        "capture_nbytes": record["nbytes"],
        "capture_token_order": "tile_major",
    }
    return tensor, provenance


def _regular_file(path: Path, what: str, *, max_bytes: int | None = None) -> Path:
    if path.is_symlink() or not path.is_file():
        raise HeadReferenceError(f"{what} must be a regular non-symlink file")
    if max_bytes is not None and path.stat().st_size > max_bytes:
        raise HeadReferenceError(f"{what} exceeds its metadata size limit")
    try:
        canonical = path.resolve(strict=True)
    except (OSError, RuntimeError) as error:
        raise HeadReferenceError(f"cannot resolve {what}") from error
    if canonical != path.absolute():
        raise HeadReferenceError(f"{what} must use its canonical path")
    return canonical


def _load_head_state(
    *, source, subset_path: Path, raw_header: bytes, receipt_path: Path,
) -> tuple[object, object, dict[str, torch.Tensor], dict]:
    try:
        from safetensors import safe_open
    except ImportError as error:
        raise HeadReferenceError("safetensors is required to load the authenticated head subset") from error
    receipt_path = _regular_file(receipt_path, "head subset receipt", max_bytes=2 * 1024 * 1024)
    try:
        receipt = checkpoint.parse_header(receipt_path.read_bytes())
        checked = head_checkpoint.validate_head_subset_receipt(raw_header, receipt, subset_path)
    except (checkpoint.InventoryError, head_checkpoint.HeadCheckpointError) as error:
        raise HeadReferenceError(f"authenticated head subset rejected: {error}") from error
    expected_digests = (
        "hf_revision", "source_commit", "source_path", "header_sha256",
        "source_data_offsets", "source_f32_bytes", "subset_sha256",
    )
    checked_source = {key: receipt[key] for key in expected_digests}
    if (receipt.get("hf_revision") != checkpoint.REVISION
            or receipt.get("source_commit") != checkpoint.SOURCE_COMMIT
            or receipt.get("source_path") != checkpoint.SOURCE_PATH
            or receipt.get("header_sha256") != checkpoint.HEADER_SHA256
            or receipt.get("source_data_offsets") != [head_checkpoint.HEAD_DATA_START,
                                                        head_checkpoint.HEAD_DATA_END]
            or receipt.get("source_f32_bytes") != head_checkpoint.HEAD_F32_BYTES
            or checked["subset_sha256"] != receipt.get("subset_sha256")):
        raise HeadReferenceError("head receipt does not bind the pinned checkpoint source")
    subset_path = _regular_file(subset_path, "head safetensors subset")
    try:
        with safe_open(str(subset_path), framework="pt", device="cpu") as handle:
            if set(handle.keys()) != set(head_checkpoint.HEAD_TENSORS):
                raise HeadReferenceError("head subset safetensors keys differ from the pinned inventory")
            state = {name: handle.get_tensor(name) for name in head_checkpoint.HEAD_TENSORS}
    except HeadReferenceError:
        raise
    except Exception as error:
        raise HeadReferenceError("cannot load the authenticated head safetensors subset") from error
    expected = {
        "decoder.final_norm.weight": (4096,),
        "decoder.head.linear.bias": (3072,),
        "decoder.head.linear.weight": (3072, 4096),
        "decoder.head.norm.weight": (4096,),
    }
    if any(tensor.dtype != torch.float32 or tuple(tensor.shape) != expected[name]
           or not tensor.is_contiguous() for name, tensor in state.items()):
        raise HeadReferenceError("head safetensors values differ from the pinned F32 parameter shapes")

    arch = source.QWEN3_8B_GEN
    final_norm = source.RMSNorm(arch.hidden_size, arch.rms_norm_eps)
    head = source.PixelHead(arch.hidden_size)
    final_norm = final_norm.to(device="mps", dtype=torch.bfloat16)
    head = head.to(device="mps", dtype=torch.bfloat16)
    destinations = {
        "decoder.final_norm.weight": final_norm.weight,
        "decoder.head.norm.weight": head.norm.weight,
        "decoder.head.linear.weight": head.linear.weight,
        "decoder.head.linear.bias": head.linear.bias,
    }
    with torch.no_grad():
        for name, parameter in destinations.items():
            parameter.copy_(state[name].to(device="mps", dtype=torch.bfloat16))
    runtime_weights = {
        "final_norm_weight": final_norm.weight,
        "head_norm_weight": head.norm.weight,
        "head_linear_weight": head.linear.weight,
        "head_linear_bias": head.linear.bias,
    }
    return final_norm, head, runtime_weights, {
        "subset_sha256": checked["subset_sha256"],
        "receipt_sha256": _sha256_file(receipt_path),
        "source_data_offsets": [head_checkpoint.HEAD_DATA_START,
                                head_checkpoint.HEAD_DATA_END],
        "source_f32_bytes": head_checkpoint.HEAD_F32_BYTES,
        "runtime_bf16_bytes": head_checkpoint.HEAD_BF16_BYTES,
        "full_file_hash_verified": False,
    }


def _physical_memory_bytes() -> int:
    if sys.platform == "darwin":
        try:
            return int(subprocess.check_output(
                ["sysctl", "-n", "hw.memsize"], text=True, stderr=subprocess.DEVNULL
            ).strip())
        except (OSError, ValueError, subprocess.CalledProcessError) as error:
            raise HeadReferenceError("cannot determine physical memory for the 10% floor") from error
    try:
        return int(os.sysconf("SC_PHYS_PAGES") * os.sysconf("SC_PAGE_SIZE"))
    except (AttributeError, OSError, ValueError) as error:
        raise HeadReferenceError("cannot determine physical memory for the 10% floor") from error


def _mps_preflight(max_mps_bytes: int) -> dict:
    if not torch.backends.mps.is_built() or not torch.backends.mps.is_available():
        raise HeadReferenceError("PyTorch MPS is unavailable; the head oracle has no CPU fallback")
    if os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK", "").strip().lower() in {
        "1", "true", "yes", "on",
    }:
        raise HeadReferenceError("PYTORCH_ENABLE_MPS_FALLBACK is enabled")
    if (not hasattr(torch.mps, "recommended_max_memory")
            or not hasattr(torch.mps, "driver_allocated_memory")
            or not hasattr(torch.mps, "set_per_process_memory_fraction")):
        raise HeadReferenceError("PyTorch MPS lacks required memory guard APIs")
    try:
        preflight = memory_preflight(
            physical_bytes=_physical_memory_bytes(),
            driver_allocated_bytes=int(torch.mps.driver_allocated_memory()),
            recommended_max_bytes=int(torch.mps.recommended_max_memory()),
            estimated_peak_bytes=estimate_mps_peak_bytes(),
            max_mps_bytes=max_mps_bytes,
        )
        torch.mps.set_per_process_memory_fraction(float(preflight["mps_memory_fraction"]))
    except HeadReferenceError:
        raise
    except Exception as error:
        raise HeadReferenceError(f"MPS memory preflight failed: {error}") from error
    return preflight


def make_fixture(
    *, seed: int, upstream: Path | str = DEFAULT_UPSTREAM,
    root: Path | str = ROOT, max_mps_bytes: int = DEFAULT_MAX_MPS_BYTES,
) -> dict:
    """Create one fresh full-J head fixture; seed 42 and 43 are separate calls."""
    if type(seed) is not int or seed not in _FULL36_INPUTS:
        raise HeadReferenceError("head fixtures admit only full36 seeds 42 and 43")
    root_path = Path(root).expanduser()
    if not root_path.is_absolute() or root_path.is_symlink():
        raise HeadReferenceError("head artifact root must be an absolute non-symlink path")
    root_path = root_path.resolve()
    if root_path != ROOT.resolve():
        raise HeadReferenceError("head reference output root must be the dedicated 20261027 head root")
    source_dir = root_path / "source"
    results_dir = root_path / "results"
    for directory, label in ((source_dir, "head source directory"),
                             (results_dir, "head results directory")):
        if directory.is_symlink() or not directory.is_dir() or directory.resolve() != directory:
            raise HeadReferenceError(f"{label} must be an existing canonical directory")
    header_path = _regular_file((checkpoint.ROOT / "source" / "pixelumm-header.json").resolve(),
                                "pinned PixelUMM header", max_bytes=2 * 1024 * 1024)
    raw_header = header_path.read_bytes()
    try:
        checkpoint.parse_pinned_header(raw_header)
    except checkpoint.InventoryError as error:
        raise HeadReferenceError(f"pinned PixelUMM header rejected: {error}") from error
    header_sha = hashlib.sha256(raw_header).hexdigest()
    if header_sha != checkpoint.HEADER_SHA256:
        raise HeadReferenceError("pinned PixelUMM header digest changed")

    upstream_path = Path(upstream).expanduser().resolve()
    source, source_sha = block_reference.load_upstream(upstream_path)
    if source_sha != UPSTREAM_MODEL_SHA256:
        raise HeadReferenceError("pinned upstream PixelUMM model source digest changed")
    plan, inverse, permutation_digest = make_head_order(source)
    memory = _mps_preflight(max_mps_bytes)

    subset_path = _regular_file(source_dir / "head-f32.safetensors", "head safetensors subset")
    final_norm, head, runtime_weights, checkpoint_source = _load_head_state(
        source=source,
        subset_path=subset_path,
        raw_header=raw_header,
        receipt_path=source_dir / "head-f32.receipt.json",
    )
    device = torch.device("mps")
    block35_output, input_provenance = _load_full36_input(seed, device=device)
    if tuple(block35_output.shape) != (1, TOKENS, HIDDEN) or block35_output.dtype != torch.bfloat16:
        raise HeadReferenceError("loaded block35.output differs from the admitted BF16 full grid")
    inverse_device = inverse.to(device=device)

    with torch.no_grad():
        # Match DCAEVPixelUMMDecoder.forward after its 36-layer loop. The global
        # final norm is still in tile-major order; only then does source code
        # apply tokens[:, inverse]. The source PixelHead executes after autocast.
        final_norm_tile_major = final_norm(block35_output)
        restored_thw = final_norm_tile_major[:, inverse_device]
        head_norm_thw = head.norm(restored_thw)
        linear_thw = head.linear(head_norm_thw)
        output_thw = source.unpatchify(linear_thw, GRID)

    expected_shapes = {
        "final_norm_tile_major": (1, TOKENS, HIDDEN),
        "restored_thw": (1, TOKENS, HIDDEN),
        "head_norm_thw": (1, TOKENS, HIDDEN),
        "linear_thw": (1, TOKENS, TUBE_DIM),
        "output_thw": (1, 3, 24, 704, 1280),
    }
    stage_tensors = {
        "final_norm_tile_major": final_norm_tile_major,
        "restored_thw": restored_thw,
        "head_norm_thw": head_norm_thw,
        "linear_thw": linear_thw,
        "output_thw": output_thw,
    }
    for name, tensor in stage_tensors.items():
        if (tuple(tensor.shape) != expected_shapes[name]
                or tensor.dtype != torch.bfloat16 or tensor.device.type != "mps"):
            raise HeadReferenceError(
                f"upstream head stage shape, dtype, or device mismatch: {name}"
            )

    # One explicit terminal inference synchronization; all CPU serialization
    # and hashing occurs afterward.
    torch.mps.synchronize()

    metadata = {
        "upstream_revision": UPSTREAM_REVISION,
        "upstream_model_path": UPSTREAM_MODEL_PATH,
        "upstream_model_sha256": source_sha,
        "hf_revision": checkpoint.REVISION,
        "hf_source_path": checkpoint.SOURCE_PATH,
        "checkpoint_header_sha256": header_sha,
        "head_source": checkpoint_source,
        "input_provenance": input_provenance,
        "grid": list(GRID),
        "tokens": TOKENS,
        "tile_tokens": TILE_TOKENS,
        "token_order": "tile_major",
        "output_token_order": "thw",
        "output_layout": "BCTHW",
        "permutation_sha256": permutation_digest,
        "runtime_dtype": "BF16",
        "comparison": {"atol": ATOL, "rtol": RTOL},
        "atol": ATOL,
        "rtol": RTOL,
        "source_config": {
            "hidden_size": HIDDEN,
            "rms_norm_eps": 1e-6,
            "tubelet": [4, 16, 16],
            "tube_dim": TUBE_DIM,
        },
        "head_operations": {
            "global_final_norm": "pinned source.RMSNorm; BF16 input and gain; eps=1e-6",
            "inverse_tile_permutation": "pinned source local_attention_layout inverse; no full BlockMask",
            "head_norm": "pinned source.PixelHead.norm (torch.nn.RMSNorm); BF16; eps=1e-6",
            "linear": "pinned source.PixelHead.linear (torch.nn.Linear); BF16; bias=true; autocast exited",
            "unpatchify": "pinned source.unpatchify; token values ordered (tubelet_t, patch_h, patch_w, channel)",
        },
        "capture_order": list(CAPTURE_ORDER),
        "memory_preflight": {
            **memory,
            "retained_head_bytes": HEAD_STATIC_RETAINED_BYTES,
            "temporary_margin_bytes": TEMPORARY_MARGIN_BYTES,
            "estimated_peak_formula": "input + BF16 head weights + five stage outputs + 1GiB temporary margin",
        },
        "execution": {
            "loaded_decoder_blocks": 0,
            "ran_full_decoder": False,
            "ran_generation": False,
            "explicit_terminal_mps_synchronize_count": 1,
            "cpu_fallback": False,
        },
        "torch_version": torch.__version__,
        "python_version": platform.python_version(),
    }
    output_path = results_dir / f"pixelumm-head-full{seed}-v1.json"
    fixture = block_reference._write_fixture(
        output_path,
        metadata=metadata,
        inputs={"block35_output": block35_output},
        weights=runtime_weights,
        captures=stage_tensors,
        schema=SCHEMA,
    )
    expected_payload_bytes = expected_fixture_payload_bytes()
    if fixture["payload_bytes"] != expected_payload_bytes:
        raise HeadReferenceError("head fixture payload differs from exact tensor byte estimate")
    return {
        "fixture": str(output_path.resolve()),
        "payload": str(output_path.with_suffix(".bin").resolve()),
        "schema": SCHEMA,
        "seed": seed,
        "grid": list(GRID),
        "tokens": TOKENS,
        "capture_order": fixture["capture_order"],
        "payload_bytes": fixture["payload_bytes"],
        "payload_sha256": fixture["payload_sha256"],
        "permutation_sha256": permutation_digest,
        "estimated_peak_bytes": memory["estimated_peak_bytes"],
        "input_capture_sha256": input_provenance["capture_sha256"],
        "head_subset_sha256": checkpoint_source["subset_sha256"],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--upstream", type=Path, default=DEFAULT_UPSTREAM)
    parser.add_argument("--seed", type=int, choices=(42, 43), required=True)
    parser.add_argument("--max-mps-bytes", type=int, default=DEFAULT_MAX_MPS_BYTES)
    args = parser.parse_args(argv)
    try:
        result = make_fixture(
            seed=args.seed, upstream=args.upstream, root=args.root,
            max_mps_bytes=args.max_mps_bytes,
        )
    except (HeadReferenceError, OSError, RuntimeError) as error:
        print(json.dumps({"error": str(error)}), file=sys.stderr)
        return 2
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
