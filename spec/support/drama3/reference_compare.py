"""Strict comparator for DRAMA-3 native/reference activation traces."""

from __future__ import annotations

import json
import math
import struct
from pathlib import Path
from typing import Any


TRACE_FORMAT = "drama3-reference-traces-v1"


class TraceFormatError(ValueError):
    pass


def _load(path: str | Path) -> dict[str, Any]:
    try:
        with Path(path).open(encoding="utf-8") as stream:
            value = json.load(stream)
    except (OSError, json.JSONDecodeError) as exc:
        raise TraceFormatError(f"cannot read trace JSON {path}: {exc}") from exc
    if not isinstance(value, dict) or value.get("format") != TRACE_FORMAT:
        raise TraceFormatError(f"{path}: expected format {TRACE_FORMAT!r}")
    if not isinstance(value.get("traces"), list):
        raise TraceFormatError(f"{path}: traces must be a list")
    return value


def _validate_records(bundle: dict[str, Any], source: str) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for index, item in enumerate(bundle["traces"]):
        where = f"{source}:traces[{index}]"
        if not isinstance(item, dict):
            raise TraceFormatError(f"{where}: record must be an object")
        name = item.get("name")
        if not isinstance(name, str) or not name:
            raise TraceFormatError(f"{where}: name must be a non-empty string")
        if name in result:
            raise TraceFormatError(f"{where}: duplicate trace name {name!r}")
        shape = item.get("shape")
        if (
            not isinstance(shape, list)
            or not shape
            or any(type(dimension) is not int or dimension <= 0 for dimension in shape)
        ):
            raise TraceFormatError(f"{where}: shape must contain positive integers")
        dtype = item.get("dtype")
        if dtype != "float32":
            raise TraceFormatError(f"{where}: dtype must be 'float32', got {dtype!r}")
        values = item.get("values")
        count = math.prod(shape)
        if not isinstance(values, list) or len(values) != count:
            raise TraceFormatError(
                f"{where}: shape {shape} requires {count} values, got "
                f"{len(values) if isinstance(values, list) else type(values).__name__}"
            )
        clean_values: list[float] = []
        for value_index, value in enumerate(values):
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise TraceFormatError(f"{where}: value {value_index} is not numeric")
            number = float(value)
            if not math.isfinite(number):
                raise TraceFormatError(f"{where}: non-finite value at element {value_index}")
            try:
                value32 = struct.unpack("!f", struct.pack("!f", number))[0]
            except (OverflowError, struct.error) as exc:
                raise TraceFormatError(
                    f"{where}: value {value_index} is outside the float32 range"
                ) from exc
            if not math.isfinite(value32):
                raise TraceFormatError(f"{where}: value {value_index} is not finite as float32")
            clean_values.append(value32)
        result[name] = {"shape": shape, "dtype": dtype, "values": clean_values}
    return result


def compare_bundles(
    reference_path: str | Path,
    actual_path: str | Path,
    *,
    atol: float = 3e-5,
    rtol: float = 3e-4,
    prefixes: tuple[str, ...] = (),
) -> dict[str, Any]:
    if atol < 0 or rtol < 0 or not math.isfinite(atol) or not math.isfinite(rtol):
        raise ValueError("atol and rtol must be finite and non-negative")
    reference = _validate_records(_load(reference_path), str(reference_path))
    actual = _validate_records(_load(actual_path), str(actual_path))
    if any(not isinstance(prefix, str) or not prefix for prefix in prefixes):
        raise ValueError("trace prefixes must be non-empty strings")
    if prefixes:
        reference = {
            name: record
            for name, record in reference.items()
            if any(name.startswith(prefix) for prefix in prefixes)
        }
        actual = {
            name: record
            for name, record in actual.items()
            if any(name.startswith(prefix) for prefix in prefixes)
        }
        if not reference:
            raise TraceFormatError(f"prefixes select no reference traces: {prefixes}")
    missing = sorted(reference.keys() - actual.keys())
    unexpected = sorted(actual.keys() - reference.keys())
    if missing or unexpected:
        pieces = []
        if missing:
            pieces.append(f"missing traces: {missing[:8]}")
        if unexpected:
            pieces.append(f"unmatched traces: {unexpected[:8]}")
        raise TraceFormatError("; ".join(pieces))

    max_abs = 0.0
    max_rel = 0.0
    worst_name = ""
    worst_index = 0
    count = 0
    for name in sorted(reference):
        expected = reference[name]
        observed = actual[name]
        if expected["shape"] != observed["shape"]:
            raise TraceFormatError(
                f"{name}: shape mismatch {observed['shape']} != {expected['shape']}"
            )
        if expected["dtype"] != observed["dtype"]:
            raise TraceFormatError(
                f"{name}: dtype mismatch {observed['dtype']} != {expected['dtype']}"
            )
        for index, (wanted, got) in enumerate(zip(expected["values"], observed["values"])):
            absolute = abs(got - wanted)
            relative = absolute / max(abs(wanted), 1e-12)
            if absolute > max_abs:
                max_abs, max_rel = absolute, relative
                worst_name, worst_index = name, index
            count += 1
            if absolute > atol + rtol * abs(wanted):
                raise AssertionError(
                    f"{name}[{index}] differs: expected {wanted:.9g}, actual {got:.9g}, "
                    f"abs={absolute:.3g}, allowed={atol + rtol * abs(wanted):.3g}"
                )
    return {
        "status": "match",
        "trace_count": len(reference),
        "value_count": count,
        "atol": atol,
        "rtol": rtol,
        "prefixes": list(prefixes),
        "max_abs_error": max_abs,
        "max_relative_error_at_max_abs": max_rel,
        "worst_trace": worst_name,
        "worst_flat_index": worst_index,
    }
