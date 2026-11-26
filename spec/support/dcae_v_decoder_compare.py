#!/usr/bin/env python3
"""Compare authenticated upstream and native DCAE-V BF16 decoder artifacts."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import re
import stat
import sys
from typing import Any, Iterable, Sequence


SCHEMA = "cogni_ml.dcae_v_decoder_comparison/v1"
ATOL = 2e-5
RTOL = 2e-5
HEADER_LIMIT = 16 * 1024 * 1024
DEFAULT_CHUNK_ELEMENTS = 262144
MAX_CHUNK_ELEMENTS = 1048576
SMOKE_LATENT_SHAPE = (1, 32, 20, 1, 1)
FULL_LATENT_SHAPE = (1, 32, 60, 9, 16)
# Complete causal prefixes of the authenticated paired bank. These are
# output-only profiling controls, not another independent upstream execution.
PROFILE_LATENT_SHAPES = ((1, 32, 10, 9, 16), (1, 32, 20, 9, 16))
_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_TRACE_KEY = re.compile(r"trace\.(project_in|stages\.[1-6]|project_out)\.tile([0-9]+)\Z")
_TRACE_BOUNDARY_ORDER = (
    "project_in", "stages.6", "stages.5", "stages.4", "stages.3",
    "stages.2", "stages.1", "project_out",
)


class CompareError(ValueError):
    """An authenticated artifact, decoder schema, or comparison input is invalid."""


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise CompareError(f"duplicate JSON field: {key}")
        result[key] = value
    return result


def _valid_sha256(value: object) -> bool:
    return type(value) is str and _SHA256.fullmatch(value) is not None


def _hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class TensorArtifact:
    """Validated BF16 safetensors ranges read in bounded flat chunks."""

    def __init__(self, path: str | Path, *, expected_sha256: str):
        if not _valid_sha256(expected_sha256):
            raise CompareError("caller SHA-256 must be 64 lowercase hexadecimal characters")
        source = Path(path).expanduser()
        if not source.is_absolute() or source.is_symlink() or not source.is_file():
            raise CompareError("tensor artifact must be an absolute regular non-symlink file")
        self.path = source.resolve(strict=True)
        self._identity = self._stat_identity()
        if self._identity[2] < 10:
            raise CompareError("safetensors artifact is truncated")
        observed_sha256 = _hash_file(self.path)
        self.check_identity()
        if observed_sha256 != expected_sha256:
            raise CompareError("tensor artifact SHA-256 differs from caller pin")
        self.sha256 = observed_sha256
        self.tensors: dict[str, dict[str, Any]] = {}
        self.metadata: dict[str, str] = {}
        self.data_offset = 0
        self._read_header()
        self.check_identity()

    def _stat_identity(self) -> tuple[int, int, int, int, int]:
        try:
            info = self.path.stat(follow_symlinks=False)
        except OSError as error:
            raise CompareError("tensor artifact cannot be statted") from error
        if not stat.S_ISREG(info.st_mode):
            raise CompareError("tensor artifact is not a regular file")
        return (info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns, info.st_ctime_ns)

    def check_identity(self) -> None:
        if self._stat_identity() != self._identity:
            raise CompareError("tensor artifact changed while being compared")

    def _read_header(self) -> None:
        with self.path.open("rb") as stream:
            prefix = stream.read(8)
            if len(prefix) != 8:
                raise CompareError("safetensors header prefix is truncated")
            header_bytes = int.from_bytes(prefix, "little", signed=False)
            if header_bytes < 2 or header_bytes > HEADER_LIMIT or 8 + header_bytes > self._identity[2]:
                raise CompareError("safetensors header extent is invalid")
            raw_header = stream.read(header_bytes)
        try:
            header = json.loads(
                raw_header.decode("utf-8").rstrip(" "), object_pairs_hook=_unique_object
            )
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise CompareError("safetensors header JSON is invalid") from error
        if type(header) is not dict:
            raise CompareError("safetensors header must be a JSON object")
        metadata = header.pop("__metadata__", {})
        if type(metadata) is not dict or any(
            type(key) is not str or type(value) is not str for key, value in metadata.items()
        ):
            raise CompareError("safetensors metadata must map strings to strings")
        self.metadata = metadata
        self.data_offset = 8 + header_bytes
        ranges: list[tuple[int, int, str]] = []
        for name, record in header.items():
            if type(name) is not str or not name or type(record) is not dict or set(record) != {
                "dtype", "shape", "data_offsets"
            }:
                raise CompareError(f"invalid safetensors record: {name}")
            dtype = record["dtype"]
            shape = record["shape"]
            offsets = record["data_offsets"]
            if type(dtype) is not str or dtype != "BF16":
                raise CompareError(f"decoder artifact tensor must be BF16: {name}")
            if type(shape) is not list or not shape or any(
                type(extent) is not int or extent <= 0 for extent in shape
            ):
                raise CompareError(f"invalid safetensors shape: {name}")
            if type(offsets) is not list or len(offsets) != 2 or any(
                type(offset) is not int for offset in offsets
            ) or offsets[0] < 0 or offsets[1] < offsets[0]:
                raise CompareError(f"invalid safetensors data range: {name}")
            count = math.prod(shape)
            expected_bytes = count * 2
            if offsets[1] - offsets[0] != expected_bytes:
                raise CompareError(f"safetensors byte extent does not match BF16 shape: {name}")
            self.tensors[name] = {
                "dtype": dtype,
                "shape": tuple(shape),
                "count": count,
                "offset": offsets[0],
                "nbytes": expected_bytes,
            }
            ranges.append((offsets[0], offsets[1], name))
        if not self.tensors:
            raise CompareError("safetensors artifact contains no tensors")
        ranges.sort()
        cursor = 0
        for start, end, name in ranges:
            if start != cursor:
                raise CompareError(f"safetensors data ranges overlap or have gaps: {name}")
            cursor = end
        if self.data_offset + cursor != self._identity[2]:
            raise CompareError("safetensors ranges do not cover the complete data section")

    def read_words(self, name: str, start: int, count: int):
        import numpy as np

        self.check_identity()
        record = self.tensors[name]
        if start < 0 or count < 0 or start + count > record["count"]:
            raise CompareError(f"tensor read is outside the admitted range: {name}")
        byte_offset = self.data_offset + record["offset"] + start * 2
        byte_count = count * 2
        with self.path.open("rb") as stream:
            stream.seek(byte_offset)
            raw = stream.read(byte_count)
        self.check_identity()
        if len(raw) != byte_count:
            raise CompareError(f"safetensors tensor range was truncated: {name}")
        return np.frombuffer(raw, dtype="<u2")


def _expected_output_shape(latent_shape: Sequence[int]) -> tuple[int, ...]:
    batch, _channels, frames, height, width = latent_shape
    return (batch, 3, frames * 4, height * 32, width * 32)


def _expected_trace_shape(boundary: str, latent_shape: Sequence[int]) -> tuple[int, ...]:
    batch, _channels, _frames, height, width = latent_shape
    if boundary == "project_in":
        return (batch, 1024, 10, height, width)
    if boundary == "stages.6":
        return (batch, 1024, 10, height, width)
    if boundary == "stages.5":
        return (batch, 1024, 40, height, width)
    scale_by_stage = {"stages.4": (1024, 2), "stages.3": (512, 4),
                      "stages.2": (512, 8), "stages.1": (256, 16)}
    if boundary in scale_by_stage:
        channels, scale = scale_by_stage[boundary]
        return (batch, channels, 40, height * scale, width * scale)
    if boundary == "project_out":
        return (batch, 3, 40, height * 32, width * 32)
    raise CompareError(f"unknown decoder trace boundary: {boundary}")


def _ordered_tensor_names(names: Iterable[str]) -> list[str]:
    names = set(names)
    ordered = [name for name in ("input.latent", "output.pixels") if name in names]
    trace_records: list[tuple[int, int, str]] = []
    for name in names - set(ordered):
        match = _TRACE_KEY.fullmatch(name)
        if match is None:
            raise CompareError(f"unexpected decoder artifact tensor: {name}")
        boundary, tile_value = match.groups()
        trace_records.append((int(tile_value), _TRACE_BOUNDARY_ORDER.index(boundary), name))
    ordered.extend(name for _tile, _boundary, name in sorted(trace_records))
    return ordered


def _validate_decoder_schema(artifact: TensorArtifact) -> None:
    tensors = artifact.tensors
    if "input.latent" not in tensors or "output.pixels" not in tensors:
        raise CompareError("decoder artifact requires input.latent and output.pixels")
    latent_shape = tensors["input.latent"]["shape"]
    if latent_shape not in (SMOKE_LATENT_SHAPE, FULL_LATENT_SHAPE, *PROFILE_LATENT_SHAPES):
        raise CompareError("input.latent shape is outside the pinned smoke/final-bank geometries")
    if tensors["output.pixels"]["shape"] != _expected_output_shape(latent_shape):
        raise CompareError("output.pixels shape does not match the pinned decoder geometry")

    trace_names = [name for name in tensors if name.startswith("trace.")]
    if latent_shape == FULL_LATENT_SHAPE or latent_shape in PROFILE_LATENT_SHAPES:
        if trace_names:
            raise CompareError("the full paired-bank artifact must be output-only")
        return
    if not trace_names:
        raise CompareError("the smoke artifact requires at least one complete stage trace tile")
    tile_groups: dict[int, set[str]] = {}
    for name in trace_names:
        match = _TRACE_KEY.fullmatch(name)
        if match is None:
            raise CompareError(f"invalid decoder trace tensor name: {name}")
        boundary, tile_value = match.groups()
        tile = int(tile_value)
        if str(tile) != tile_value:
            raise CompareError(f"decoder trace tile index is not canonical: {name}")
        if tile >= latent_shape[2] // 10:
            raise CompareError(f"trace tile index is outside the latent sequence: {name}")
        boundaries = tile_groups.setdefault(tile, set())
        if boundary in boundaries:
            raise CompareError(f"decoder trace boundary is duplicated for tile {tile}: {name}")
        boundaries.add(boundary)
        if tensors[name]["shape"] != _expected_trace_shape(boundary, latent_shape):
            raise CompareError(f"decoder trace shape differs from the pinned stage graph: {name}")
    expected_boundaries = set(_TRACE_BOUNDARY_ORDER)
    for tile, boundaries in tile_groups.items():
        if boundaries != expected_boundaries:
            missing = sorted(expected_boundaries - boundaries)
            raise CompareError(f"decoder trace tile {tile} is incomplete; missing={missing}")


def _float32_from_bf16_words(words):
    import numpy as np

    widened = words.astype("<u4")
    np.left_shift(widened, 16, out=widened)
    return widened.view("<f4")


def _compare_tensor(reference: TensorArtifact, candidate: TensorArtifact, name: str,
                    *, chunk_elements: int) -> dict[str, Any]:
    import numpy as np

    count = reference.tensors[name]["count"]
    bitwise_mismatches = 0
    finite_input_bitwise_mismatches = 0
    nonfinite_count = 0
    tolerance_failures = 0
    zero_reference_nonzero_candidate_count = 0
    max_abs_error: float | None = None
    max_relative_error: float | None = None
    is_input = name == "input.latent"

    for start in range(0, count, chunk_elements):
        size = min(chunk_elements, count - start)
        ref_words = reference.read_words(name, start, size)
        cand_words = candidate.read_words(name, start, size)
        bitwise_mismatches += int(np.count_nonzero(ref_words != cand_words))
        ref_values = _float32_from_bf16_words(ref_words).astype(np.float64)
        cand_values = _float32_from_bf16_words(cand_words).astype(np.float64)
        ref_finite = np.isfinite(ref_values)
        cand_finite = np.isfinite(cand_values)
        finite_pair = ref_finite & cand_finite
        nonfinite = ~finite_pair
        nonfinite_count += int(np.count_nonzero(nonfinite))
        if is_input:
            finite_input_bitwise_mismatches += int(
                np.count_nonzero((ref_words != cand_words) & finite_pair)
            )
        if not finite_pair.any():
            continue
        difference = np.abs(cand_values[finite_pair] - ref_values[finite_pair])
        chunk_max_abs = float(np.max(difference))
        max_abs_error = chunk_max_abs if max_abs_error is None else max(max_abs_error, chunk_max_abs)
        ref_finite_values = ref_values[finite_pair]
        nonzero_reference = ref_finite_values != 0.0
        zero_reference_differences = difference[~nonzero_reference]
        zero_reference_nonzero_candidate_count += int(
            np.count_nonzero(zero_reference_differences != 0.0)
        )
        if nonzero_reference.any():
            relative = difference[nonzero_reference] / np.abs(ref_finite_values[nonzero_reference])
            chunk_max_relative = float(np.max(relative))
            max_relative_error = (
                chunk_max_relative if max_relative_error is None
                else max(max_relative_error, chunk_max_relative)
            )
        if not is_input:
            tolerance = ATOL + RTOL * np.abs(ref_finite_values)
            tolerance_failures += int(np.count_nonzero(difference > tolerance))

    if is_input:
        failed_count = finite_input_bitwise_mismatches + nonfinite_count
        tolerance_failures = 0
    else:
        failed_count = tolerance_failures + nonfinite_count
    return {
        "shape": list(reference.tensors[name]["shape"]),
        "dtype": "BF16",
        "elements": count,
        "bitwise_mismatch_count": bitwise_mismatches,
        "nonfinite_count": nonfinite_count,
        "tolerance_failure_count": tolerance_failures,
        "zero_reference_nonzero_candidate_count": zero_reference_nonzero_candidate_count,
        "max_abs_error": max_abs_error,
        "max_relative_error": max_relative_error,
        "failed_count": failed_count,
        "passed": failed_count == 0,
    }


def compare_artifacts(reference_path: str | Path, reference_sha256: str,
                      candidate_path: str | Path, candidate_sha256: str, *,
                      chunk_elements: int = DEFAULT_CHUNK_ELEMENTS) -> dict[str, Any]:
    """Compare the upstream reference with native output using bounded CPU chunks."""
    if type(chunk_elements) is not int or not 1 <= chunk_elements <= MAX_CHUNK_ELEMENTS:
        raise CompareError(f"chunk-elements must be from 1 through {MAX_CHUNK_ELEMENTS}")
    reference = TensorArtifact(reference_path, expected_sha256=reference_sha256)
    candidate = TensorArtifact(candidate_path, expected_sha256=candidate_sha256)
    if set(reference.tensors) != set(candidate.tensors):
        missing = sorted(set(reference.tensors) - set(candidate.tensors))
        extra = sorted(set(candidate.tensors) - set(reference.tensors))
        raise CompareError(f"tensor names differ; candidate missing={missing}, extra={extra}")
    for name in reference.tensors:
        ref_record = reference.tensors[name]
        candidate_record = candidate.tensors[name]
        if ref_record["dtype"] != candidate_record["dtype"] or ref_record["shape"] != candidate_record["shape"]:
            raise CompareError(f"BF16 tensor schema/shape differs: {name}")
    _validate_decoder_schema(reference)
    _validate_decoder_schema(candidate)
    ordered_names = _ordered_tensor_names(reference.tensors)

    tensor_results: dict[str, dict[str, Any]] = {}
    first_failed_tensor: str | None = None
    first_failed_trace: str | None = None
    for name in ordered_names:
        result = _compare_tensor(reference, candidate, name, chunk_elements=chunk_elements)
        tensor_results[name] = result
        if not result["passed"] and first_failed_tensor is None:
            first_failed_tensor = name
        if name.startswith("trace.") and not result["passed"] and first_failed_trace is None:
            first_failed_trace = name
    reference.check_identity()
    candidate.check_identity()
    total_failed = sum(item["failed_count"] for item in tensor_results.values())
    total_bitwise = sum(item["bitwise_mismatch_count"] for item in tensor_results.values())
    total_nonfinite = sum(item["nonfinite_count"] for item in tensor_results.values())
    total_tolerance_failures = sum(item["tolerance_failure_count"] for item in tensor_results.values())
    return {
        "schema": SCHEMA,
        "passed": total_failed == 0,
        "reference_sha256": reference.sha256,
        "candidate_sha256": candidate.sha256,
        "reference_metadata": reference.metadata,
        "candidate_metadata": candidate.metadata,
        "tolerance": {"atol": ATOL, "rtol": RTOL,
                      "rule": "abs(candidate-reference) <= atol + rtol*abs(reference)"},
        "input_rule": "input.latent must be bitwise identical BF16",
        "relative_error_rule": (
            "abs(candidate-reference)/abs(reference), finite elements with nonzero reference"
        ),
        "chunk_elements": chunk_elements,
        "tensor_order": ordered_names,
        "tensor_count": len(ordered_names),
        "total_failed_count": total_failed,
        "total_bitwise_mismatch_count": total_bitwise,
        "total_nonfinite_count": total_nonfinite,
        "total_tolerance_failure_count": total_tolerance_failures,
        "first_failed_tensor": first_failed_tensor,
        "first_failed_trace": first_failed_trace,
        "tensors": tensor_results,
    }


def _write_report(path: str | Path, summary: dict[str, Any]) -> None:
    target = Path(path).expanduser()
    if not target.is_absolute() or target.is_symlink() or target.exists():
        raise CompareError("report path must be a fresh absolute path")
    try:
        parent = target.parent.resolve(strict=True)
    except OSError as error:
        raise CompareError("report parent directory must already exist") from error
    if not parent.is_dir():
        raise CompareError("report parent directory must already exist")
    payload = (json.dumps(summary, sort_keys=True, allow_nan=False, indent=2) + "\n").encode()
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(parent / target.name, flags, 0o600)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
    except BaseException:
        try:
            (parent / target.name).unlink()
        except OSError:
            pass
        raise


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", required=True, type=Path)
    parser.add_argument("--reference-sha256", required=True)
    parser.add_argument("--candidate", required=True, type=Path)
    parser.add_argument("--candidate-sha256", required=True)
    parser.add_argument("--report", type=Path, help="optional fresh absolute JSON report path")
    parser.add_argument("--chunk-elements", type=int, default=DEFAULT_CHUNK_ELEMENTS)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        summary = compare_artifacts(
            args.reference,
            args.reference_sha256,
            args.candidate,
            args.candidate_sha256,
            chunk_elements=args.chunk_elements,
        )
        if args.report is not None:
            _write_report(args.report, summary)
        print(json.dumps(summary, sort_keys=True, allow_nan=False))
        return 0 if summary["passed"] else 1
    except (CompareError, OSError, ValueError, TypeError) as error:
        print(f"DCAE-V decoder comparison failed: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
