#!/usr/bin/env python3
"""Build independent Q4_K plus fixed-stride F16-delta Wan references.

Every arm shares the pinned canonical ggml Q4_K base. MPS selects, packs, and
reconstructs rare per-block corrections, then computes projection references
with ``torch.nn.functional.linear`` and MPS fallback disabled. Small scalar
and vector CPU wire controls are exposed for offline contract checks. The
script never loads the full Wan model and its fixed per-block quota is an
ablation, not a globally optimal allocation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import operator
import os
import shutil
import struct
import sys
import tempfile
from pathlib import Path
from typing import Any

import numpy as np

import wan_affine_budget_reference as affine_reference
import wan_q4k_range_reference as q4k_reference

SOURCE_ROWS = 4
INPUT_DIM = 5120
OUTPUT_DIM = 13824
BENCH_ROWS = 4480
BLOCK_VALUES = 256
Q4_K_BLOCK_BYTES = 144
Q4_K_BYTES = 39_813_120
Q4_K_BLOCKS = OUTPUT_DIM * INPUT_DIM // BLOCK_VALUES
DELTA_RECORD_BYTES = 3
FORMAT_DESCRIPTOR_BYTES = 16
LOGICAL_BF16_WEIGHT_BYTES = OUTPUT_DIM * INPUT_DIM * 2
DECODED_F32_WEIGHT_BYTES = OUTPUT_DIM * INPUT_DIM * 4
PATCHES_BY_ARM = {
    "q4_k": 0,
    "q4_k_delta1": 1,
    "q4_k_delta4": 4,
    "q4_k_delta8": 8,
}
FORMAT_IDS = {name: 0 if patches == 0 else 0x100 + patches for name, patches in PATCHES_BY_ARM.items()}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def file_record(path: Path, dtype: str, shape: list[int], root: Path) -> dict[str, Any]:
    return {
        "path": path.relative_to(root).as_posix(),
        "dtype": dtype,
        "shape": shape,
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def _patch_count(value: int) -> int:
    try:
        count = operator.index(value)
    except TypeError as exc:
        raise ValueError("patch count must be an integer between 0 and 256") from exc
    if isinstance(value, (bool, np.bool_)) or count < 0 or count > BLOCK_VALUES:
        raise ValueError("patch count must be an integer between 0 and 256")
    return count


def _block_f32(values: np.ndarray, label: str) -> np.ndarray:
    result = np.asarray(values, dtype=np.float32)
    if result.shape != (BLOCK_VALUES,):
        raise ValueError(f"{label} block must contain exactly 256 F32 values")
    if not np.isfinite(result).all():
        raise ValueError(f"{label} block contains a non-finite value")
    return result


def select_patch_positions(scores: np.ndarray, count: int) -> np.ndarray:
    """Return positions in descending score order, breaking ties by low index."""
    quota = _patch_count(count)
    values = np.asarray(scores, dtype=np.float32)
    if values.shape != (BLOCK_VALUES,):
        raise ValueError("patch scores must contain exactly 256 values")
    if not np.isfinite(values).all():
        raise ValueError("patch scores contain a non-finite value")
    if np.any(values < np.float32(0.0)):
        raise ValueError("patch scores must be non-negative")
    indices = np.arange(BLOCK_VALUES, dtype=np.int64)
    # lexsort uses the last key first: descending score, then ascending index.
    order = np.lexsort((indices, np.negative(values, dtype=np.float32)))
    return order[:quota].astype(np.uint8, copy=False)


def _scalar_positions(scores: np.ndarray, count: int) -> list[int]:
    """Independent scalar selector used only as a CPU control."""
    quota = _patch_count(count)
    values = np.asarray(scores, dtype=np.float32)
    if values.shape != (BLOCK_VALUES,) or not np.isfinite(values).all():
        raise ValueError("scalar patch scores must be 256 finite F32 values")
    if np.any(values < np.float32(0.0)):
        raise ValueError("scalar patch scores must be non-negative")
    return sorted(range(BLOCK_VALUES), key=lambda index: (-float(values[index]), index))[:quota]


def _half_delta(value: np.float32) -> np.float16:
    with np.errstate(over="ignore", under="ignore", invalid="ignore"):
        result = np.asarray(value, dtype=np.float32).astype("<f2")[()]
    if not np.isfinite(result):
        raise ValueError("selected F16 delta overflows to a non-finite value")
    return np.float16(result)


def pack_delta_block_scalar(
    original: np.ndarray, base: np.ndarray, patches: int
) -> bytes:
    """Serialize one block as rank-ordered ``U8 position + little-endian F16``."""
    quota = _patch_count(patches)
    source = _block_f32(original, "original")
    decoded = _block_f32(base, "base")
    residual = np.subtract(source, decoded, dtype=np.float32)
    scores = np.abs(residual).astype(np.float32, copy=False)
    positions = _scalar_positions(scores, quota)
    result = bytearray()
    for position in positions:
        half = _half_delta(np.float32(residual[position]))
        result.extend(struct.pack("<B", position))
        result.extend(struct.pack("<e", float(half)))
    return bytes(result)


def decode_delta_block_scalar(base: np.ndarray, sidecar: bytes | bytearray | memoryview, patches: int) -> np.ndarray:
    """Decode one strict fixed-quota sidecar block using scalar F32 additions."""
    quota = _patch_count(patches)
    decoded = _block_f32(base, "base").copy()
    expected = quota * DELTA_RECORD_BYTES
    if len(sidecar) != expected:
        raise ValueError(f"sidecar block must be exactly {expected} bytes")
    seen: set[int] = set()
    for record in range(quota):
        offset = record * DELTA_RECORD_BYTES
        position = struct.unpack_from("<B", sidecar, offset)[0]
        if position in seen:
            raise ValueError("sidecar block contains a duplicate patch position")
        seen.add(position)
        delta = np.float32(struct.unpack_from("<e", sidecar, offset + 1)[0])
        if not np.isfinite(delta):
            raise ValueError("sidecar block contains a non-finite F16 delta")
        decoded[position] = np.add(decoded[position], delta, dtype=np.float32)
    return decoded


def pack_delta_blocks(
    original: np.ndarray, base: np.ndarray, patches: int
) -> tuple[bytes, np.ndarray]:
    """Vector CPU microfixture writer and reconstructed-weight control."""
    quota = _patch_count(patches)
    source = np.asarray(original, dtype=np.float32)
    decoded = np.asarray(base, dtype=np.float32)
    if source.ndim != 2 or source.shape[1] != BLOCK_VALUES or source.shape != decoded.shape:
        raise ValueError("delta batches must have matching [blocks,256] F32 shapes")
    if not np.isfinite(source).all() or not np.isfinite(decoded).all():
        raise ValueError("delta batch contains a non-finite value")
    if quota == 0:
        return b"", decoded.copy()

    residual = np.subtract(source, decoded, dtype=np.float32)
    scores = np.abs(residual).astype(np.float32, copy=False)
    indices = np.broadcast_to(np.arange(BLOCK_VALUES, dtype=np.int64), scores.shape)
    order = np.lexsort((indices, np.negative(scores, dtype=np.float32)), axis=1)
    positions = order[:, :quota].astype(np.uint8, copy=False)
    rows = np.arange(source.shape[0], dtype=np.int64)[:, None]
    selected_residuals = residual[rows, positions.astype(np.int64)]
    with np.errstate(over="ignore", under="ignore", invalid="ignore"):
        selected_half = selected_residuals.astype("<f2")
    if not np.isfinite(selected_half).all():
        raise ValueError("selected F16 delta overflows to a non-finite value")

    wire = np.empty((source.shape[0], quota, DELTA_RECORD_BYTES), dtype=np.uint8)
    wire[:, :, 0] = positions
    wire[:, :, 1:] = selected_half.view(np.uint8).reshape(source.shape[0], quota, 2)
    reconstructed = decoded.copy()
    selected_base = decoded[rows, positions.astype(np.int64)]
    corrected = np.add(selected_base, selected_half.astype(np.float32), dtype=np.float32)
    reconstructed[rows, positions.astype(np.int64)] = corrected
    return wire.tobytes(), reconstructed


def decode_delta_blocks(
    base: np.ndarray, sidecar: bytes | bytearray | memoryview, patches: int
) -> np.ndarray:
    """Decode rank-ordered fixed-stride blocks with vectorized F32 additions."""
    quota = _patch_count(patches)
    decoded = np.asarray(base, dtype=np.float32)
    if decoded.ndim != 2 or decoded.shape[1] != BLOCK_VALUES:
        raise ValueError("delta base must have shape [blocks,256]")
    if not np.isfinite(decoded).all():
        raise ValueError("delta base contains a non-finite value")
    expected = decoded.shape[0] * quota * DELTA_RECORD_BYTES
    if len(sidecar) != expected:
        if len(sidecar) % DELTA_RECORD_BYTES:
            raise ValueError("sidecar size must be a multiple of the 3-byte record size")
        raise ValueError(f"sidecar must be exactly {expected} bytes")
    result = decoded.copy()
    if quota == 0:
        return result

    records = np.frombuffer(sidecar, dtype=np.uint8).reshape(decoded.shape[0], quota, DELTA_RECORD_BYTES)
    positions = records[:, :, 0].astype(np.int64, copy=False)
    ordered = np.sort(positions, axis=1)
    if quota > 1 and np.any(ordered[:, 1:] == ordered[:, :-1]):
        raise ValueError("sidecar contains duplicate patch positions in a block")
    delta_halves = records[:, :, 1:].copy().reshape(-1).view("<f2")
    deltas = delta_halves.astype(np.float32).reshape(decoded.shape[0], quota)
    if not np.isfinite(deltas).all():
        raise ValueError("sidecar contains a non-finite F16 delta")
    rows = np.arange(decoded.shape[0], dtype=np.int64)[:, None]
    selected_base = np.take_along_axis(decoded, positions, axis=1)
    corrected = np.add(selected_base, deltas, dtype=np.float32)
    result[rows, positions] = corrected
    return result


def format_id_for_patches(patches: int) -> int:
    quota = _patch_count(patches)
    if quota not in (0, 1, 4, 8):
        raise ValueError("wire descriptor supports only baseline, 1, 4, or 8 patches")
    return 0 if quota == 0 else 0x100 + quota


def write_descriptor(path: Path, format_id: int) -> None:
    try:
        format_id = operator.index(format_id)
    except TypeError as exc:
        raise ValueError("unsupported Q4_K delta format ID") from exc
    if format_id not in (0, 0x101, 0x104, 0x108):
        raise ValueError("unsupported Q4_K delta format ID")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(struct.pack("<4I", 0x31425157, format_id, OUTPUT_DIM, INPUT_DIM))


def _mps_pack_and_reconstruct(
    original_f32_path: Path,
    base_f32_path: Path,
    arm_paths: dict[int, tuple[Path, Path]],
    chunk_blocks: int = 8192,
) -> dict[str, str]:
    """Select, F16-pack, reconstruct, and export full-weight blocks on MPS."""
    if sys.byteorder != "little":
        raise RuntimeError("the Q4_K delta wire format requires a little-endian host")
    os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "0"
    import torch

    if not torch.backends.mps.is_available():
        raise RuntimeError("MPS is required for full-weight patch selection and reconstruction")
    try:
        original = np.memmap(
            original_f32_path,
            dtype="<f4",
            mode="r",
            shape=(OUTPUT_DIM * INPUT_DIM,),
        )
        base = np.memmap(
            base_f32_path,
            dtype="<f4",
            mode="r",
            shape=(OUTPUT_DIM * INPUT_DIM,),
        )
        outputs: dict[int, tuple[Any, Any]] = {}
        for quota, (sidecar_path, decoded_path) in arm_paths.items():
            sidecar_stream = sidecar_path.open("wb")
            decoded_stream = decoded_path.open("wb")
            outputs[quota] = (sidecar_stream, decoded_stream)

        for first in range(0, Q4_K_BLOCKS, chunk_blocks):
            last = min(Q4_K_BLOCKS, first + chunk_blocks)
            first_value = first * BLOCK_VALUES
            last_value = last * BLOCK_VALUES
            # Bounded pageable CPU staging is read from pinned F32 fixtures,
            # then uploaded to MPS; no CPU selection or reconstruction occurs.
            original_cpu = np.ascontiguousarray(
                original[first_value:last_value].reshape(last - first, BLOCK_VALUES)
            )
            base_cpu = np.ascontiguousarray(
                base[first_value:last_value].reshape(last - first, BLOCK_VALUES)
            )
            original_mps = torch.from_numpy(original_cpu).to(device="mps", dtype=torch.float32)
            base_mps = torch.from_numpy(base_cpu).to(device="mps", dtype=torch.float32)
            residual = torch.sub(original_mps, base_mps)
            scores = torch.abs(residual)
            # One boundary check per bounded chunk prevents non-finite scores
            # from silently changing stable ranking or the F16 wire values.
            if not bool(torch.isfinite(scores).all().item()):
                raise ValueError("MPS patch selection encountered a non-finite residual")
            ranked_positions = torch.argsort(scores, dim=1, descending=True, stable=True)[:, :8]

            for quota, (sidecar_stream, decoded_stream) in outputs.items():
                if quota == 0:
                    reconstructed = base_mps
                else:
                    # Stable descending argsort retains ascending local indices
                    # for equal scores because each row starts in index order.
                    positions = ranked_positions[:, :quota]
                    selected_base = torch.gather(base_mps, 1, positions)
                    selected_residual = torch.gather(residual, 1, positions)
                    delta_half = selected_residual.to(torch.float16)
                    delta_bytes = delta_half.contiguous().view(torch.uint8).reshape(last - first, quota, 2)
                    packed = torch.stack(
                        (positions.to(torch.uint8), delta_bytes[:, :, 0], delta_bytes[:, :, 1]),
                        dim=-1,
                    )
                    packed_cpu = packed.to(device="cpu").contiguous().numpy()
                    half_values = packed_cpu[:, :, 1:].copy().reshape(-1).view("<f2")
                    if not np.isfinite(half_values).all():
                        raise ValueError("selected F16 delta overflows to a non-finite value")
                    sidecar_stream.write(packed_cpu.tobytes())
                    corrected = torch.add(selected_base, delta_half.to(torch.float32))
                    reconstructed = base_mps.scatter(1, positions, corrected)

                # This CPU copy synchronizes the queued MPS reconstruction and
                # writes a diagnostic F32 tensor outside the native timed path.
                reconstructed_cpu = reconstructed.to(device="cpu").contiguous().numpy()
                reconstructed_cpu.astype("<f4", copy=False).tofile(decoded_stream)
                torch.mps.synchronize()

        for sidecar_stream, decoded_stream in outputs.values():
            sidecar_stream.flush()
            decoded_stream.flush()
            sidecar_stream.close()
            decoded_stream.close()
        del original, base
        versions = {"torch": torch.__version__, "numpy": np.__version__, "python": sys.version}
        try:
            torch.mps.empty_cache()
        except (AttributeError, RuntimeError):
            pass
        return versions
    except BaseException:
        for streams in locals().get("outputs", {}).values():
            for stream in streams:
                try:
                    stream.close()
                except Exception:
                    pass
        raise


def _write_baseline_projection(
    source_paths: dict[str, Path], stage: Path, rows: int, input_path: Path, bias_path: Path
) -> tuple[dict[str, Any], dict[str, str] | None]:
    baseline_dir = stage / "baseline"
    baseline_dir.mkdir(parents=True, exist_ok=True)
    decoded_path = baseline_dir / "decoded_weight.f32"
    projection_path = baseline_dir / "expected_projection.f32"
    affine_reference.copy_or_link(source_paths["bf16_decoded"], decoded_path)
    if rows == BENCH_ROWS:
        affine_reference.copy_or_link(source_paths["bf16_projection"], projection_path)
        versions = None
        projection_reference = "pinned same-geometry MPS fixture"
    else:
        versions = affine_reference.write_candidate_projection(
            decoded_path, input_path, bias_path, projection_path, rows
        )
        projection_reference = "fresh same-geometry MPS F.linear"
    return (
        {
            "decoded_weight": file_record(decoded_path, "F32", [OUTPUT_DIM, INPUT_DIM], stage),
            "expected_projection": file_record(projection_path, "F32", [rows, OUTPUT_DIM], stage),
            "projection_reference": projection_reference,
            "control": "original BF16 source weight converted to F32",
        },
        versions,
    )


def _variant_storage(patches: int, payload_path: Path, sidecar_path: Path, descriptor_path: Path) -> dict[str, Any]:
    base_bytes = payload_path.stat().st_size
    sidecar_bytes = sidecar_path.stat().st_size
    descriptor_bytes = descriptor_path.stat().st_size
    total = base_bytes + sidecar_bytes + descriptor_bytes
    expected_sidecar = Q4_K_BLOCKS * patches * DELTA_RECORD_BYTES
    if base_bytes != Q4_K_BYTES or sidecar_bytes != expected_sidecar or descriptor_bytes != FORMAT_DESCRIPTOR_BYTES:
        raise RuntimeError("Q4_K delta candidate has an unexpected storage component size")
    correction_count = Q4_K_BLOCKS * patches
    weight_count = Q4_K_BLOCKS * BLOCK_VALUES
    return {
        "blocks": Q4_K_BLOCKS,
        "patches_per_block": patches,
        "correction_count": correction_count,
        "delta_record_bytes": DELTA_RECORD_BYTES,
        "base_payload_bytes": base_bytes,
        "sidecar_bytes": sidecar_bytes,
        "descriptor_bytes": descriptor_bytes,
        "package_bytes": total,
        "bits_per_weight": total * 8 / weight_count,
        "logical_storage_bytes": total,
        "resident_storage_bytes": total,
        "logical_weight_bytes_bf16": LOGICAL_BF16_WEIGHT_BYTES,
        "decoded_f32_diagnostic_bytes": DECODED_F32_WEIGHT_BYTES,
        "diagnostic_exports_counted_in_resident_storage": False,
        "resident_storage_basis": "serialized canonical base, fixed-stride sidecar, and 16-byte descriptor; no padding or index arrays",
    }


def _validate_sampled_canonical_decode(payload_path: Path, decoded_path: Path) -> dict[str, Any]:
    """Cross-check three canonical Q4_K records against the pinned decoded F32."""
    decoded = np.memmap(
        decoded_path,
        dtype="<f4",
        mode="r",
        shape=(OUTPUT_DIM * INPUT_DIM,),
    )
    block_ids = (0, Q4_K_BLOCKS // 2, Q4_K_BLOCKS - 1)
    max_abs = 0.0
    compared_values = 0
    violations = 0
    with payload_path.open("rb") as stream:
        for block_id in block_ids:
            stream.seek(block_id * Q4_K_BLOCK_BYTES)
            record = stream.read(Q4_K_BLOCK_BYTES)
            if len(record) != Q4_K_BLOCK_BYTES:
                raise RuntimeError("canonical Q4_K payload ended during sampled decode control")
            actual = q4k_reference.decode_q4k_record(record)
            reference = np.asarray(
                decoded[block_id * BLOCK_VALUES : (block_id + 1) * BLOCK_VALUES],
                dtype=np.float32,
            )
            if not np.isfinite(actual).all() or not np.isfinite(reference).all():
                raise RuntimeError("canonical Q4_K sampled decode contains a non-finite value")
            error = np.abs(np.subtract(actual, reference, dtype=np.float32))
            tolerance = np.float32(2e-5) + np.float32(2e-5) * np.abs(reference)
            violations += int(np.count_nonzero(error > tolerance))
            max_abs = max(max_abs, float(np.max(error)))
            compared_values += BLOCK_VALUES
    del decoded
    if violations:
        raise RuntimeError("sampled independent Q4_K decode differs from the pinned decoded base")
    return {
        "decoder": "wan_q4k_range_reference.decode_q4k_record",
        "block_ids": list(block_ids),
        "compared_values": compared_values,
        "max_abs": max_abs,
        "tolerance_violations": violations,
        "predicate": "finite(actual) && abs(actual-reference) <= 2e-5 + 2e-5*abs(reference)",
    }


def create_reference_fixture(fixture_dir: Path, output_dir: Path, rows: int) -> Path:
    if rows not in (SOURCE_ROWS, BENCH_ROWS):
        raise ValueError("--rows must be 4 or 4480")
    fixture_dir = fixture_dir.expanduser().resolve()
    output_dir = output_dir.expanduser().resolve()
    if os.path.lexists(output_dir):
        raise FileExistsError(f"immutable output already exists: {output_dir}")
    output_dir.parent.mkdir(parents=True, exist_ok=True)

    source_manifest, source_paths = affine_reference.validate_fixture(fixture_dir)
    stage = Path(tempfile.mkdtemp(prefix=f".{output_dir.name}.", dir=output_dir.parent))
    try:
        source_records, copied_sources = affine_reference.source_records(source_paths, stage)
        source_input = np.memmap(
            source_paths["input"], dtype="<f4", mode="r", shape=(SOURCE_ROWS, INPUT_DIM)
        )
        source_bf16 = np.memmap(
            source_paths["original_weight"],
            dtype="<u2",
            mode="r",
            shape=(OUTPUT_DIM * INPUT_DIM,),
        )

        baseline, torch_versions = _write_baseline_projection(
            source_paths, stage, rows, copied_sources["input"], copied_sources["bias"]
        )
        baseline_projection_path = stage / baseline["expected_projection"]["path"]

        variant_paths: dict[str, dict[str, Path]] = {}
        mps_arm_paths: dict[int, tuple[Path, Path]] = {}
        for name, patches in PATCHES_BY_ARM.items():
            variant_dir = stage / "variants" / name
            variant_dir.mkdir(parents=True, exist_ok=True)
            payload_path = variant_dir / "payload.bin"
            sidecar_path = variant_dir / "sidecar.bin"
            descriptor_path = variant_dir / "descriptor.bin"
            decoded_path = variant_dir / "decoded_weight.f32"
            projection_path = variant_dir / "expected_projection.f32"
            affine_reference.copy_or_link(source_paths["q4_payload"], payload_path)
            write_descriptor(descriptor_path, format_id_for_patches(patches))
            if patches == 0:
                sidecar_path.write_bytes(b"")
                affine_reference.copy_or_link(source_paths["q4_decoded"], decoded_path)
            else:
                mps_arm_paths[patches] = (sidecar_path, decoded_path)
            variant_paths[name] = {
                "payload": payload_path,
                "sidecar": sidecar_path,
                "descriptor": descriptor_path,
                "decoded_weight": decoded_path,
                "expected_projection": projection_path,
            }

        canonical_decode_control = _validate_sampled_canonical_decode(
            variant_paths["q4_k"]["payload"], variant_paths["q4_k"]["decoded_weight"]
        )

        versions = _mps_pack_and_reconstruct(
            source_paths["bf16_decoded"], source_paths["q4_decoded"], mps_arm_paths
        )
        torch_versions = torch_versions or versions

        input_rows = np.asarray(source_input[:SOURCE_ROWS], dtype=np.float32)
        variants: dict[str, dict[str, Any]] = {}
        for name, patches in PATCHES_BY_ARM.items():
            paths = variant_paths[name]
            projection_versions = affine_reference.write_candidate_projection(
                paths["decoded_weight"],
                copied_sources["input"],
                copied_sources["bias"],
                paths["expected_projection"],
                rows,
            )
            torch_versions = torch_versions or projection_versions
            weight_quality = affine_reference.distortion_metrics(
                source_bf16, paths["decoded_weight"], input_rows
            )
            projection_quality = affine_reference.projection_quality(
                paths["expected_projection"], baseline_projection_path, rows
            )
            sidecar_bytes = Q4_K_BLOCKS * patches * DELTA_RECORD_BYTES
            variant = {
                "patches": patches,
                "payload": file_record(
                    paths["payload"],
                    "Q4_K",
                    [OUTPUT_DIM, INPUT_DIM // BLOCK_VALUES, Q4_K_BLOCK_BYTES],
                    stage,
                ),
                "sidecar": file_record(
                    paths["sidecar"],
                    "U8F16LE_3B",
                    [Q4_K_BLOCKS, patches, DELTA_RECORD_BYTES],
                    stage,
                ),
                "descriptor": file_record(
                    paths["descriptor"], "WQB1Descriptor", [FORMAT_DESCRIPTOR_BYTES], stage
                ),
                "decoded_weight": file_record(
                    paths["decoded_weight"], "F32", [OUTPUT_DIM, INPUT_DIM], stage
                ),
                "expected_projection": file_record(
                    paths["expected_projection"], "F32", [rows, OUTPUT_DIM], stage
                ),
                "projection_reference": "fresh same-geometry MPS F.linear with fallback disabled",
                "storage": _variant_storage(
                    patches, paths["payload"], paths["sidecar"], paths["descriptor"]
                ),
                "cpu_quality": {
                    "decode_device": "MPS" if patches else "pinned source fixture",
                    "evaluation_device": "CPU",
                    "weight_distortion": weight_quality,
                },
                "quality_vs_original_bf16": projection_quality,
            }
            if paths["sidecar"].stat().st_size != sidecar_bytes:
                raise RuntimeError(f"{name} sidecar has an unexpected fixed-stride size")
            variants[name] = variant

        del source_bf16, source_input
        own_path = Path(__file__).resolve()
        support_path = Path(affine_reference.__file__).resolve()
        q4k_support_path = Path(q4k_reference.__file__).resolve()
        manifest = {
            "schema": "wan_q4k_delta_reference_v1",
            "created_by": own_path.name,
            "geometry": {
                "rows": rows,
                "source_rows": SOURCE_ROWS,
                "input_dim": INPUT_DIM,
                "output_dim": OUTPUT_DIM,
                "block_values": BLOCK_VALUES,
                "blocks_per_output_row": INPUT_DIM // BLOCK_VALUES,
                "block_count": Q4_K_BLOCKS,
                "input_layout": "row_major periodic source-row tile",
            },
            "comparison": {
                "dtype": "F32",
                "atol": 2e-5,
                "rtol": 2e-5,
                "finite": True,
                "predicate": "finite(actual) && abs(actual-reference) <= 2e-5 + 2e-5*abs(reference)",
                "scope": "native output parity against the same arm's independent MPS F.linear reference",
                "lossy_quality_scope": "projection quality relative to the original BF16 output is separate from same-format parity",
            },
            "selection": {
                "score": "abs(F32(original BF16 weight) - F32(canonical Q4_K decoded base))",
                "scope": "independent contiguous 256-value Q4_K block",
                "quota_per_block": True,
                "rank_order": "descending absolute residual; equal residuals use ascending local position",
                "wire_order": "selected positions remain in descending residual rank order",
                "delta": "F16 round-to-nearest-ties-to-even of F32(original minus base)",
                "reconstruction": "separate F32(base + F32(F16 delta)) operation",
                "global_optimality_claim": False,
                "quality_scope": "the 4 unique seeded projection rows only; rows=4480 periodically tiles those same 4 rows",
            },
            "oracle": {
                "device": "MPS",
                "fallback_disabled": True,
                "selection_packing_reconstruction_device": "MPS",
                "projection": "torch.nn.functional.linear(input, decoded_weight, bias)",
                "torch_version": torch_versions["torch"],
                "numpy_version": np.__version__,
                "python_version": sys.version,
                "chunk_blocks": 8192,
                "cpu_staging_and_sync": "Pinned F32 original/base memmaps are read in 8192-block pageable NumPy chunks and uploaded to MPS. Residual finite checking, stable selection, F16 conversion, byte interleave, and F32 reconstruction run on MPS. One bounded residual finite check synchronizes each chunk; copied sidecar bytes receive a CPU F16-finiteness check during export. Sidecar and decoded-weight copies return to CPU per chunk for immutable file export and synchronize queued MPS work. F.linear references return to CPU for file export and explicitly synchronize. Full-weight CPU distortion metrics scan exported decoded tensors in bounded output-row chunks after MPS generation. These staging and diagnostic exports are outside the native timed path.",
            },
            "canonical_base_decode_control": canonical_decode_control,
            "source_fixture": {
                "schema": source_manifest["schema"],
                "manifest_sha256": affine_reference.FIXTURE_MANIFEST_SHA256,
                "real_video_derived": source_manifest.get("fixture_policy", {}).get("input_is_real_video_derived"),
                "checkpoint_weight_sha256": affine_reference.WEIGHT_BF16_SHA256,
                "input_sha256": affine_reference.INPUT_SHA256,
                "bias_f32_sha256": affine_reference.BIAS_F32_SHA256,
                "q4_k_sha256": affine_reference.Q4_K_SHA256,
                "q4_decoded_sha256": source_manifest["variants"]["q4_k"]["decoded_weight"]["sha256"],
            },
            "input": source_records["input"],
            "bias": source_records["bias"],
            "original_weight": source_records["original_weight"],
            "baseline": baseline,
            "variants": variants,
            "producers": {
                "pinned_q4_k": source_manifest.get("producers", {}).get("q4_k"),
                "independent_reference": {
                    "script": own_path.name,
                    "script_sha256": sha256_file(own_path),
                    "selection_pack_reconstruction": "PyTorch MPS stable descending argsort; F16 wire delta; fixed three-byte little-endian records; separate F32 reconstruction add",
                    "fixture_validation_and_metrics": {
                        "script": support_path.name,
                        "script_sha256": sha256_file(support_path),
                    },
                    "q4k_decode_control": {
                        "script": q4k_support_path.name,
                        "script_sha256": sha256_file(q4k_support_path),
                    },
                },
            },
            "fixture_policy": {
                "immutable_output": True,
                "full_wan_model_loaded": False,
                "real_video_quality_claim": False,
                "source_rows_are_periodically_tiled_when_rows_4480": True,
            },
        }
        manifest_path = stage / "manifest.json"
        with manifest_path.open("w", encoding="utf-8") as stream:
            json.dump(manifest, stream, indent=2, sort_keys=True, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        if os.path.lexists(output_dir):
            raise FileExistsError(f"immutable output appeared while preparing: {output_dir}")
        os.replace(stage, output_dir)
        return output_dir
    except BaseException:
        shutil.rmtree(stage, ignore_errors=True)
        raise


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixture-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--rows", type=int, choices=(SOURCE_ROWS, BENCH_ROWS), required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output = create_reference_fixture(args.fixture_dir, args.output_dir, args.rows)
    print(output / "manifest.json")


if __name__ == "__main__":
    main()
