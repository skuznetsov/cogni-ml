#!/usr/bin/env python3
"""Compare standalone native DRAMA-3 dense projections to pinned MLX outputs."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import struct
import sys
from pathlib import Path
from typing import Any


PROJECTION = "backbone.layers.0.linear_attn"
PROJECTION_SIZES = {"qkv": 8192, "z": 4096, "b": 32, "a": 32}
REFERENCE_FORMAT = "drama3-dense-projection-reference-v1"
NATIVE_FORMAT = "drama3-dense-projection-native-v1"
INPUT_FORMAT = "drama3-dense-projection-inputs-v1"
MODEL_REPOSITORY = "cguldogan/drama-3-mlx-q4"
MODEL_REVISION = "5331ab509359c0e4a7e353a19e255ddf3e8152a8"
UPSTREAM_REVISION = "e7488935ec2ebfafd38ef2b58e290d79f3d05dc7"
INPUT_WIDTH = 2048
DEFAULT_ATOL = 6e-5
DEFAULT_RTOL = 3e-4
REQUIRED_SOURCE_HASHES = {
    "config.json",
    "model.safetensors.index.json",
    "mlx_audio/lm/models/qwen3_5.py",
    "input_bundle",
    "layer0_projection_tensors",
}


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def load(path: Path, expected_format: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read {path}: {exc}") from exc
    require(isinstance(value, dict), f"{path}: expected a JSON object")
    require(value.get("format") == expected_format, f"{path}: unexpected format")
    require(value.get("projection") == PROJECTION, f"{path}: unexpected projection")
    return value


def require_sha256(value: Any, where: str) -> str:
    require(isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None,
            f"{where}: expected lowercase SHA-256 hex")
    return value


def validate_source_hashes(value: Any, where: str) -> dict[str, str]:
    require(isinstance(value, dict), f"{where}: expected source_hashes object")
    require(REQUIRED_SOURCE_HASHES <= set(value), f"{where}: missing required source hashes")
    return {
        name: require_sha256(digest, f"{where}.{name}")
        for name, digest in value.items()
    }


def values_for(record: Any, shape: list[int], where: str) -> list[float]:
    require(isinstance(record, dict), f"{where}: expected a record object")
    require(record.get("shape") == shape, f"{where}: expected shape {shape}")
    require(record.get("dtype") == "float32", f"{where}: expected float32 values")
    values = record.get("values")
    require(isinstance(values, list) and len(values) == math.prod(shape),
            f"{where}: value count does not match shape")
    result = []
    for index, value in enumerate(values):
        require(not isinstance(value, bool) and isinstance(value, (int, float)),
                f"{where}[{index}]: expected a numeric value")
        number = float(value)
        require(math.isfinite(number), f"{where}[{index}]: non-finite value")
        try:
            packed = struct.pack("!f", number)
        except (OverflowError, struct.error) as exc:
            raise ValueError(f"{where}[{index}]: value is outside float32 range") from exc
        value32 = struct.unpack("!f", packed)[0]
        require(math.isfinite(value32), f"{where}[{index}]: non-finite float32 value")
        result.append(value32)
    return result


def index_rows(bundle: dict[str, Any], source: str) -> dict[int, dict[str, Any]]:
    rows = bundle.get("rows")
    require(isinstance(rows, list) and bool(rows), f"{source}: rows must be non-empty")
    result: dict[int, dict[str, Any]] = {}
    for item in rows:
        require(isinstance(item, dict), f"{source}: row must be an object")
        row = item.get("row")
        require(type(row) is int and row >= 0, f"{source}: invalid row index")
        require(row not in result, f"{source}: duplicate row {row}")
        result[row] = item
    return result


def load_input_fixture(
    reference_path: Path, reference: dict[str, Any]
) -> tuple[dict[str, Any], str, str]:
    fixture_record = reference.get("input_fixture")
    require(
        isinstance(fixture_record, dict),
        "upstream reference is missing its input fixture record",
    )
    fixture_name = fixture_record.get("path")
    require(
        isinstance(fixture_name, str)
        and fixture_name not in ("", ".", "..")
        and Path(fixture_name).name == fixture_name
        and "/" not in fixture_name
        and "\\" not in fixture_name,
        "upstream input fixture must be a sibling filename",
    )
    claimed_digest = require_sha256(fixture_record.get("sha256"), "upstream input fixture hash")
    fixture_path = reference_path.parent / fixture_name
    try:
        fixture_bytes = fixture_path.read_bytes()
    except OSError as exc:
        raise ValueError(f"cannot read input fixture {fixture_path}: {exc}") from exc
    actual_digest = hashlib.sha256(fixture_bytes).hexdigest()
    require(
        actual_digest == claimed_digest,
        "upstream input fixture bytes do not match reference hash",
    )
    try:
        fixture = json.loads(fixture_bytes)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot parse input fixture {fixture_path}: {exc}") from exc
    require(isinstance(fixture, dict), "input fixture: expected a JSON object")
    require(fixture.get("format") == INPUT_FORMAT, "input fixture: unexpected format")
    require(fixture.get("projection") == PROJECTION, "input fixture: unexpected projection")
    require(fixture.get("input_dtype") == "float32", "input fixture: expected float32 input")
    require(fixture.get("input_shape") == [INPUT_WIDTH],
            f"input fixture: expected input shape [{INPUT_WIDTH}]")
    require(fixture.get("source_model") == MODEL_REPOSITORY and
            fixture.get("source_revision") == MODEL_REVISION,
            "input fixture: unexpected model identity")
    return fixture, actual_digest, fixture_name


def validate_input_rows(fixture: dict[str, Any], source: str) -> dict[int, dict[str, Any]]:
    rows = index_rows(fixture, source)
    for row, item in rows.items():
        values = item.get("values")
        require(isinstance(values, list) and len(values) == INPUT_WIDTH,
                f"{source}: row {row} must contain {INPUT_WIDTH} input values")
        for index, value in enumerate(values):
            require(not isinstance(value, bool) and isinstance(value, (int, float)) and
                    math.isfinite(float(value)),
                    f"{source}: row {row} input[{index}] must be finite numeric data")
            try:
                value32 = struct.unpack("!f", struct.pack("!f", float(value)))[0]
            except (OverflowError, struct.error) as exc:
                raise ValueError(
                    f"{source}: row {row} input[{index}] is outside float32 range"
                ) from exc
            require(math.isfinite(value32),
                    f"{source}: row {row} input[{index}] is non-finite float32 data")
    return rows


def validate_reference_provenance(reference: dict[str, Any]) -> dict[str, str]:
    require(reference.get("source_model") == MODEL_REPOSITORY and
            reference.get("source_revision") == MODEL_REVISION and
            reference.get("upstream_revision") == UPSTREAM_REVISION,
            "upstream reference model or source revision differs from the pinned coordinates")
    hashes = validate_source_hashes(reference.get("source_hashes"), "upstream source_hashes")
    return hashes


def validate_native_identity(trace: dict[str, Any], where: str) -> None:
    model = trace.get("model")
    require(isinstance(model, str) and bool(model), f"{where}: missing model path")
    expected_metadata = {
        "qkv": {"shape": [8192, INPUT_WIDTH], "dtype": "BF16"},
        "z": {"shape": [4096, INPUT_WIDTH], "dtype": "BF16"},
        "b": {"shape": [32, INPUT_WIDTH], "dtype": "BF16"},
        "a": {"shape": [32, INPUT_WIDTH], "dtype": "BF16"},
    }
    require(trace.get("weight_metadata") == expected_metadata,
            f"{where}: expected pinned dense BF16 projection weight shapes")


def compare(
    reference_path: Path,
    tree_path: Path,
    simd_path: Path,
    atol: float = DEFAULT_ATOL,
    rtol: float = DEFAULT_RTOL,
) -> dict[str, Any]:
    require(math.isfinite(atol) and atol >= 0, "atol must be finite and non-negative")
    require(math.isfinite(rtol) and rtol >= 0, "rtol must be finite and non-negative")
    require(atol <= DEFAULT_ATOL and rtol <= DEFAULT_RTOL,
            f"tolerances may only be tightened from atol={DEFAULT_ATOL} and rtol={DEFAULT_RTOL}")
    reference = load(reference_path, REFERENCE_FORMAT)
    tree = load(tree_path, NATIVE_FORMAT)
    simd = load(simd_path, NATIVE_FORMAT)
    reference_hashes = validate_reference_provenance(reference)
    fixture, fixture_hash, fixture_name = load_input_fixture(reference_path, reference)
    fixture_hashes = validate_source_hashes(
        fixture.get("source_hashes"), "input fixture source_hashes"
    )
    require(fixture_hashes == reference_hashes,
            "input fixture and upstream reference source hashes differ")
    fixture_rows = validate_input_rows(fixture, str(reference_path.parent / fixture_name))
    require(tree.get("mode") == "tree", f"{tree_path}: expected tree mode")
    require(simd.get("mode") == "simd", f"{simd_path}: expected simd mode")
    require(tree.get("input_fixture") == simd.get("input_fixture"),
            "tree and simd traces used different input fixtures")
    native_fixture_path = tree.get("input_fixture")
    require(isinstance(native_fixture_path, str) and
            Path(native_fixture_path).name == fixture_name,
            "native traces do not name the upstream input fixture")
    require(tree.get("input_fixture_sha256") == fixture_hash and
            simd.get("input_fixture_sha256") == fixture_hash,
            "native traces do not match the upstream input fixture")
    validate_native_identity(tree, str(tree_path))
    validate_native_identity(simd, str(simd_path))
    require(tree.get("model") == simd.get("model"),
            "tree and simd traces used different model paths")
    reference_rows = index_rows(reference, str(reference_path))
    tree_rows = index_rows(tree, str(tree_path))
    simd_rows = index_rows(simd, str(simd_path))
    expected_rows = set(reference_rows)
    require(set(fixture_rows) == expected_rows,
            "input fixture row set differs from upstream reference")
    require(set(tree_rows) == expected_rows, "tree trace row set differs from upstream reference")
    require(set(simd_rows) == expected_rows, "simd trace row set differs from upstream reference")

    counts = {name: 0 for name in PROJECTION_SIZES}
    max_abs = {name: 0.0 for name in PROJECTION_SIZES}
    worst: dict[str, dict[str, Any]] = {}
    for row in sorted(expected_rows):
        ref_projections = reference_rows[row].get("projections")
        tree_projections = tree_rows[row].get("projections")
        simd_projections = simd_rows[row].get("projections")
        for name, width in PROJECTION_SIZES.items():
            require(isinstance(ref_projections, dict) and name in ref_projections,
                    f"reference row {row}: missing {name}")
            require(isinstance(tree_projections, dict) and name in tree_projections,
                    f"tree row {row}: missing {name}")
            require(isinstance(simd_projections, dict) and name in simd_projections,
                    f"simd row {row}: missing {name}")
            shape = [width]
            wanted = values_for(ref_projections[name], shape, f"reference row {row} {name}")
            tree_values = values_for(tree_projections[name], shape, f"tree row {row} {name}")
            simd_values = values_for(simd_projections[name], shape, f"simd row {row} {name}")
            for index, (expected, baseline, candidate) in enumerate(
                zip(wanted, tree_values, simd_values)
            ):
                if struct.pack("!f", baseline) != struct.pack("!f", candidate):
                    raise AssertionError(
                        f"tree/simd mismatch at row {row} {name}[{index}]: "
                        f"tree={baseline:.9g} simd={candidate:.9g}"
                    )
                absolute = abs(candidate - expected)
                counts[name] += 1
                if absolute > max_abs[name]:
                    max_abs[name] = absolute
                    worst[name] = {
                        "row": row,
                        "index": index,
                        "reference": expected,
                        "native": candidate,
                        "allowed_abs": atol + rtol * abs(expected),
                    }
                if absolute > atol + rtol * abs(expected):
                    raise AssertionError(
                        f"upstream mismatch at row {row} {name}[{index}]: "
                        f"reference={expected:.9g} native={candidate:.9g} "
                        f"abs={absolute:.3g} allowed={atol + rtol * abs(expected):.3g}"
                    )
    return {
        "format": "drama3-dense-projection-comparison-v1",
        "status": "match",
        "projection": PROJECTION,
        "reference": reference_path.name,
        "tree": tree_path.name,
        "simd": simd_path.name,
        "rows": sorted(expected_rows),
        "atol": atol,
        "rtol": rtol,
        "tree_simd_bitwise_equal": True,
        "value_counts": counts,
        "max_absolute_error": max_abs,
        "worst_values": worst,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", required=True, type=Path,
                        help="upstream qkv/z/b/a sidecar emitted by replay_mlx_backbone.py")
    parser.add_argument("--tree", required=True, type=Path,
                        help="native trace produced with dense reduction tree")
    parser.add_argument("--simd", required=True, type=Path,
                        help="native trace produced with dense reduction simd")
    parser.add_argument("--atol", type=float, default=DEFAULT_ATOL)
    parser.add_argument("--rtol", type=float, default=DEFAULT_RTOL)
    args = parser.parse_args()
    try:
        report = compare(args.reference, args.tree, args.simd, args.atol, args.rtol)
    except (AssertionError, ValueError) as exc:
        print(json.dumps({"status": "mismatch", "error": str(exc)}, sort_keys=True), file=sys.stderr)
        return 1
    print(json.dumps(report, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
