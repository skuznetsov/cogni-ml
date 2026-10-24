#!/usr/bin/env python3
"""Emit same-input PyTorch MPS SiLU references from a frozen Hy-Nomas fixture.

Only the source fixture's ``vec`` and ``vec_txt`` records are extracted. This
does not load checkpoint tensors or instantiate/execute a Hy-Nomas block. The
four isolated results are diagnostic controls for the pinned modulation SiLU.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import mmap
import os
from pathlib import Path
import re
import struct
import sys
from typing import Any


PINNED_REVISION = "911cea6458a89c2663513ff7099561ec190dd491"
DEFAULT_UPSTREAM = Path(
    "/opt/mldata_ssd/cogni-ml/video-scene-20261019/upstream/hy-nomas-20261024"
)
REFERENCE_KIND = "hy_nomas_same_input_pointwise_diagnostic"
_STRUCTURAL = re.compile(rb'["{}\[\]]')
_PRIMITIVE_END = re.compile(rb"[,}\]\s]")
_F32_MAX = 3.4028234663852886e38


class DiagnosticError(Exception):
    """A bounded rejection of an input artifact or runtime."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _object_without_duplicate_keys(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise DiagnosticError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _reject_json_constant(value: str) -> None:
    raise DiagnosticError(f"non-standard JSON constant: {value}")


class _FixtureScanner:
    """Read only fixture metadata and vec input records from the large JSON."""

    def __init__(self, data: Any):
        self.data = data
        self.size = len(data)

    def _ws(self, pos: int) -> int:
        while pos < self.size and self.data[pos] in b" \t\r\n":
            pos += 1
        return pos

    def _expect(self, pos: int, token: bytes) -> int:
        pos = self._ws(pos)
        if self.data[pos : pos + len(token)] != token:
            raise DiagnosticError(f"invalid fixture JSON near byte {pos}")
        return pos + len(token)

    def _string_end(self, pos: int) -> int:
        if pos >= self.size or self.data[pos] != ord('"'):
            raise DiagnosticError(f"expected a JSON string near byte {pos}")
        quote = self.data.find(b'"', pos + 1)
        while quote >= 0:
            slash = quote - 1
            while slash > pos and self.data[slash] == ord("\\"):
                slash -= 1
            if (quote - slash - 1) % 2 == 0:
                return quote + 1
            quote = self.data.find(b'"', quote + 1)
        raise DiagnosticError("unterminated JSON string in source fixture")

    def _decode(self, start: int, end: int) -> Any:
        try:
            return json.loads(
                bytes(self.data[start:end]),
                object_pairs_hook=_object_without_duplicate_keys,
                parse_constant=_reject_json_constant,
            )
        except DiagnosticError:
            raise
        except (UnicodeDecodeError, json.JSONDecodeError, RecursionError) as exc:
            raise DiagnosticError(f"invalid selected fixture JSON: {exc}") from None

    def _skip_value(self, pos: int) -> int:
        pos = self._ws(pos)
        if pos >= self.size:
            raise DiagnosticError("unexpected end of source fixture")
        first = self.data[pos]
        if first == ord('"'):
            return self._string_end(pos)
        if first in (ord("{"), ord("[")):
            stack = [ord("}") if first == ord("{") else ord("]")]
            cursor = pos + 1
            while stack:
                match = _STRUCTURAL.search(self.data, cursor)
                if match is None:
                    raise DiagnosticError("unterminated JSON container in source fixture")
                token_pos = match.start()
                token = self.data[token_pos]
                if token == ord('"'):
                    cursor = self._string_end(token_pos)
                elif token == ord("{"):
                    stack.append(ord("}"))
                    cursor = token_pos + 1
                elif token == ord("["):
                    stack.append(ord("]"))
                    cursor = token_pos + 1
                elif stack[-1] == token:
                    stack.pop()
                    cursor = token_pos + 1
                else:
                    raise DiagnosticError(f"unbalanced JSON container near byte {token_pos}")
            return cursor
        match = _PRIMITIVE_END.search(self.data, pos)
        if match is None:
            return self.size
        if match.start() == pos:
            raise DiagnosticError(f"empty JSON value near byte {pos}")
        return match.start()

    def _member(self, pos: int) -> tuple[str, int]:
        pos = self._ws(pos)
        key_end = self._string_end(pos)
        key = self._decode(pos, key_end)
        if not isinstance(key, str):
            raise DiagnosticError("fixture object key must be a string")
        value_start = self._expect(key_end, b":")
        return key, self._ws(value_start)

    def _metadata_value(self, pos: int) -> tuple[Any, int]:
        end = self._skip_value(pos)
        return self._decode(pos, end), end

    def _record_value(self, pos: int) -> tuple[dict[str, Any], int]:
        value, end = self._metadata_value(pos)
        if not isinstance(value, dict) or set(value) != {"shape", "values"}:
            raise DiagnosticError("vec input must be a tensor record with shape and values")
        return value, end

    def _parse_inputs(self, pos: int) -> tuple[dict[str, Any], int]:
        pos = self._expect(pos, b"{")
        inputs: dict[str, Any] = {}
        seen: set[str] = set()
        pos = self._ws(pos)
        if pos < self.size and self.data[pos] == ord("}"):
            return inputs, pos + 1
        while True:
            key, value_pos = self._member(pos)
            if key in seen:
                raise DiagnosticError(f"duplicate input key: {key}")
            seen.add(key)
            if key in ("vec", "vec_txt"):
                value, pos = self._record_value(value_pos)
                inputs[key] = value
            else:
                pos = self._skip_value(value_pos)
            pos = self._ws(pos)
            if pos < self.size and self.data[pos] == ord("}"):
                return inputs, pos + 1
            pos = self._expect(pos, b",")

    def _parse_case(self, pos: int) -> tuple[dict[str, Any], int]:
        pos = self._expect(pos, b"{")
        case: dict[str, Any] = {}
        seen: set[str] = set()
        pos = self._ws(pos)
        if pos < self.size and self.data[pos] == ord("}"):
            return case, pos + 1
        while True:
            key, value_pos = self._member(pos)
            if key in seen:
                raise DiagnosticError(f"duplicate case key: {key}")
            seen.add(key)
            if key == "inputs":
                case[key], pos = self._parse_inputs(value_pos)
            elif key in ("name", "config_overrides"):
                case[key], pos = self._metadata_value(value_pos)
            else:
                pos = self._skip_value(value_pos)
            pos = self._ws(pos)
            if pos < self.size and self.data[pos] == ord("}"):
                return case, pos + 1
            pos = self._expect(pos, b",")

    def _parse_cases(self, pos: int) -> tuple[list[dict[str, Any]], int]:
        pos = self._expect(pos, b"[")
        cases: list[dict[str, Any]] = []
        pos = self._ws(pos)
        if pos < self.size and self.data[pos] == ord("]"):
            return cases, pos + 1
        while True:
            case, pos = self._parse_case(pos)
            cases.append(case)
            pos = self._ws(pos)
            if pos < self.size and self.data[pos] == ord("]"):
                return cases, pos + 1
            pos = self._expect(pos, b",")

    def extract(self) -> dict[str, Any]:
        pos = self._expect(0, b"{")
        selected: dict[str, Any] = {}
        seen: set[str] = set()
        pos = self._ws(pos)
        if pos < self.size and self.data[pos] == ord("}"):
            raise DiagnosticError("source fixture metadata is empty")
        while True:
            key, value_pos = self._member(pos)
            if key in seen:
                raise DiagnosticError(f"duplicate top-level fixture key: {key}")
            seen.add(key)
            if key == "cases":
                selected[key], pos = self._parse_cases(value_pos)
            elif key in ("schema_version", "provenance", "scope", "config", "source", "weights_are_synthetic"):
                selected[key], pos = self._metadata_value(value_pos)
            else:
                pos = self._skip_value(value_pos)
            pos = self._ws(pos)
            if pos < self.size and self.data[pos] == ord("}"):
                pos += 1
                break
            pos = self._expect(pos, b",")
        if self._ws(pos) != self.size:
            raise DiagnosticError("trailing bytes after source fixture JSON")
        return selected


def _record_shape(record: Any, *, context: str) -> tuple[int, ...]:
    if not isinstance(record, dict) or set(record) != {"shape", "values"}:
        raise DiagnosticError(f"{context} must contain shape and values")
    shape, values = record["shape"], record["values"]
    if not isinstance(shape, list) or not shape or not isinstance(values, list):
        raise DiagnosticError(f"{context} has an invalid shape or value list")
    count = 1
    normalized = []
    for dimension in shape:
        if isinstance(dimension, bool) or not isinstance(dimension, int) or dimension <= 0:
            raise DiagnosticError(f"{context} has an invalid shape dimension")
        count *= dimension
        normalized.append(dimension)
    if count != len(values):
        raise DiagnosticError(f"{context} value count does not match its shape")
    for value in values:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise DiagnosticError(f"{context} contains a non-numeric value")
        try:
            number = float(value)
        except (OverflowError, ValueError):
            raise DiagnosticError(f"{context} contains a value outside F64 range") from None
        if not math.isfinite(number) or abs(number) > _F32_MAX:
            raise DiagnosticError(f"{context} contains a non-finite or out-of-F32 value")
    return tuple(normalized)


def _f32_record_sha256(record: dict[str, Any]) -> str:
    digest = hashlib.sha256()
    values = record["values"]
    for start in range(0, len(values), 4096):
        chunk = values[start : start + 4096]
        try:
            digest.update(struct.pack(f"<{len(chunk)}f", *chunk))
        except (OverflowError, struct.error):
            raise DiagnosticError("source input cannot be represented as F32") from None
    return digest.hexdigest()


def _validate_source(selected: dict[str, Any]) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    if selected.get("schema_version") != 2:
        raise DiagnosticError("source fixture must use schema_version 2")
    scope = selected.get("scope")
    provenance = selected.get("provenance")
    config = selected.get("config")
    if not isinstance(scope, dict) or not isinstance(provenance, dict) or not isinstance(config, dict):
        raise DiagnosticError("source fixture is missing scope, provenance, or config")
    if scope.get("reference_kind") != "upstream_block_bf16_boundary_control_mps_sdpa" or scope.get("attention_precision") != "bf16-boundaries":
        raise DiagnosticError("source fixture is outside the BF16-boundary control scope")
    if provenance.get("revision") != PINNED_REVISION or provenance.get("upstream_revision") != PINNED_REVISION:
        raise DiagnosticError("source fixture upstream revision is not pinned")
    if provenance.get("mps_device") != "mps":
        raise DiagnosticError("source fixture is not an MPS oracle")
    if not isinstance(provenance.get("file_sha256"), dict) or provenance.get("file_sha256") != provenance.get("source_sha256"):
        raise DiagnosticError("source fixture pinned-source hash fields disagree")
    synthetic = selected.get("weights_are_synthetic")
    source = selected.get("source")
    if not isinstance(synthetic, bool):
        raise DiagnosticError("source fixture must state whether its weights are synthetic")
    if synthetic:
        if source is not None:
            raise DiagnosticError("synthetic source fixture unexpectedly names a checkpoint")
    elif (
        not isinstance(source, dict)
        or source.get("format") != "safetensors"
        or source.get("weights_are_synthetic") is not False
        or source.get("tensor_prefix") != "double_blocks.0."
        or source.get("tensor_count") != 32
        or not isinstance(source.get("file_sha256"), str)
        or re.fullmatch(r"[0-9a-f]{64}", source["file_sha256"]) is None
    ):
        raise DiagnosticError("actual source fixture checkpoint provenance is invalid")
    for key in ("batches", "dim", "video_tokens", "text_tokens"):
        value = config.get(key)
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise DiagnosticError(f"source fixture config has invalid {key}")

    raw_cases = selected.get("cases")
    if not isinstance(raw_cases, list) or len(raw_cases) != 2:
        raise DiagnosticError("source fixture must contain exactly broadcast and per_token cases")
    cases = {}
    for case in raw_cases:
        if not isinstance(case, dict) or not isinstance(case.get("name"), str):
            raise DiagnosticError("source fixture case is missing its name")
        name = case["name"]
        if name in cases:
            raise DiagnosticError(f"duplicate source fixture case: {name}")
        if name not in ("broadcast", "per_token"):
            raise DiagnosticError(f"unexpected source fixture case: {name}")
        inputs = case.get("inputs")
        if not isinstance(inputs, dict) or not {"vec", "vec_txt"} <= set(inputs):
            raise DiagnosticError(f"{name} is missing original vec/vec_txt records")
        overrides = case.get("config_overrides", {})
        if not isinstance(overrides, dict):
            raise DiagnosticError(f"{name} config_overrides must be an object")
        case_config = _case_shape_config(config, case)
        batches = case_config["batches"]
        for stream, input_name in (("img", "vec"), ("txt", "vec_txt")):
            record = inputs[input_name]
            shape = _record_shape(record, context=f"{name}.inputs.{input_name}")
            if name == "broadcast":
                expected = (batches, config["dim"])
            else:
                tokens = config["video_tokens"] if stream == "img" else config["text_tokens"]
                expected = (batches, tokens, config["dim"])
            if shape != expected:
                raise DiagnosticError(
                    f"{name}.inputs.{input_name} shape {shape} does not match {expected}"
                )
        cases[name] = case
    if set(cases) != {"broadcast", "per_token"}:
        raise DiagnosticError("source fixture case inventory mismatch")
    return config, cases


def _case_shape_config(config: dict[str, Any], case: dict[str, Any]) -> dict[str, int]:
    """Resolve only dimensions needed to validate the frozen SiLU input layout."""
    overrides = case.get("config_overrides", {})
    if not isinstance(overrides, dict):
        raise DiagnosticError("source case config_overrides must be an object")
    batches = overrides.get("batches", config.get("batches"))
    if isinstance(batches, bool) or not isinstance(batches, int) or batches <= 0:
        raise DiagnosticError(f"{case.get('name', 'case')} has invalid batch count")
    shape_config = {
        "batches": batches,
        "dim": config["dim"],
        "video_tokens": config["video_tokens"],
        "text_tokens": config["text_tokens"],
    }
    for key, value in shape_config.items():
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise DiagnosticError(f"source fixture has invalid resolved {key}")
    return shape_config


def _fresh_output(path: Path) -> Path:
    output = path.expanduser().absolute()
    if output.exists() or output.is_symlink():
        raise DiagnosticError("--output must be a fresh path")
    if output.suffix.lower() != ".json":
        raise DiagnosticError("--output must name a .json file")
    if not output.parent.is_dir():
        raise DiagnosticError("--output parent directory must already exist")
    return output


def _atomic_write_fresh(path: Path, value: dict[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    if temporary.exists() or temporary.is_symlink():
        raise DiagnosticError("temporary output path already exists")
    try:
        with temporary.open("x", encoding="utf-8") as stream:
            json.dump(value, stream, allow_nan=False, separators=(",", ":"))
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _load_pinned_activation(upstream: Path) -> Any:
    module_path = upstream / "hymm/models/dit/modules/activation_layers.py"
    spec = importlib.util.spec_from_file_location(
        "_hy_nomas_pointwise_activation_layers", module_path
    )
    if spec is None or spec.loader is None:
        raise DiagnosticError("could not load pinned activation_layers.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.get_activation_layer("silu")


def _records_from_mps(torch: Any, tensors: dict[str, Any]) -> dict[str, dict[str, Any]]:
    names = sorted(tensors)
    flattened = [tensors[name].detach().to(dtype=torch.float32).reshape(-1) for name in names]
    sizes = [int(value.numel()) for value in flattened]
    joined = torch.cat(flattened)
    if joined.device.type != "mps":
        raise DiagnosticError("batched pointwise capture left MPS")
    torch.mps.synchronize()
    values = joined.to(device="cpu", dtype=torch.float32).contiguous().tolist()
    result: dict[str, dict[str, Any]] = {}
    offset = 0
    for name, size in zip(names, sizes):
        result[name] = {
            "shape": list(tensors[name].shape),
            "values": values[offset : offset + size],
        }
        offset += size
    if offset != len(values):
        raise DiagnosticError("batched pointwise capture length mismatch")
    return result


def _build_pointwise_records(
    torch: Any,
    config: dict[str, Any],
    selected_cases: dict[str, Any],
    device: Any,
    upstream: Path,
) -> tuple[list[dict[str, Any]], dict[str, str]]:
    silu_type = _load_pinned_activation(upstream)
    silu = silu_type()
    if not isinstance(silu, torch.nn.SiLU):
        raise DiagnosticError("pinned get_activation_layer('silu') did not return nn.SiLU")
    tensors: dict[str, Any] = {}
    operations: list[dict[str, Any]] = []
    input_hashes: dict[str, str] = {}
    with torch.inference_mode():
        for case_name in ("broadcast", "per_token"):
            case = selected_cases[case_name]
            inputs = case["inputs"]
            case_config = _case_shape_config(config, case)
            for stream, source_name in (("img", "vec"), ("txt", "vec_txt")):
                source_record = inputs[source_name]
                shape = _record_shape(source_record, context=f"{case_name}.{source_name}")
                name = f"{case_name}.{stream}.vec_silu"
                x = torch.tensor(
                    source_record["values"], dtype=torch.float32, device=device
                ).reshape(shape).contiguous()
                if x.device.type != "mps" or x.dtype != torch.float32:
                    raise DiagnosticError("SiLU input was not materialized as MPS F32")
                output = silu(x)
                if output.device.type != "mps" or output.dtype != torch.float32:
                    raise DiagnosticError("pinned SiLU output was not MPS F32")
                rounded = output.to(dtype=torch.bfloat16).to(dtype=torch.float32)
                if rounded.device.type != "mps" or rounded.dtype != torch.float32:
                    raise DiagnosticError("BF16 diagnostic round-trip left MPS F32")
                key = name
                tensors[f"{key}.input"] = x
                tensors[f"{key}.expected"] = output
                tensors[f"{key}.rounded_expected"] = rounded
                input_hash = _f32_record_sha256(source_record)
                input_hashes[key] = input_hash
                operations.append(
                    {
                        "name": name,
                        "kind": "silu",
                        "case": case_name,
                        "case_config": case_config,
                        "stream": stream,
                        "source_input": f"{case_name}.inputs.{source_name}",
                        "input_sha256_f32_le": input_hash,
                    }
                )

    records = _records_from_mps(torch, tensors)
    for operation in operations:
        prefix = operation["name"]
        operation["inputs"] = {"x": records[f"{prefix}.input"]}
        operation["expected"] = records[f"{prefix}.expected"]
        operation["rounded_expected"] = records[f"{prefix}.rounded_expected"]
        if _f32_record_sha256(operation["inputs"]["x"]) != operation["input_sha256_f32_le"]:
            raise DiagnosticError(f"MPS input changed during capture: {prefix}")
    return operations, input_hashes


def _structure_self_test() -> None:
    synthetic = {
        "schema_version": 2,
        "provenance": {
            "revision": PINNED_REVISION,
            "upstream_revision": PINNED_REVISION,
            "mps_device": "mps",
            "file_sha256": {},
            "source_sha256": {},
        },
        "scope": {
            "reference_kind": "upstream_block_bf16_boundary_control_mps_sdpa",
            "attention_precision": "bf16-boundaries",
        },
        "config": {"batches": 1, "dim": 2, "video_tokens": 2, "text_tokens": 1},
        "source": None,
        "weights_are_synthetic": True,
        "cases": [
            {
                "name": "broadcast",
                "config_overrides": {"batches": 2},
                "inputs": {
                    "vec": {"shape": [2, 2], "values": [0.25, -0.5, 0.75, -1.0]},
                    "vec_txt": {"shape": [2, 2], "values": [0.75, -1.0, 0.25, -0.5]},
                },
                "variants": {"ignored": [{"values": [9.0] * 1000}]},
            },
            {
                "name": "per_token",
                "config_overrides": {"batches": 1},
                "inputs": {
                    "vec": {"shape": [1, 2, 2], "values": [0.0, 0.125, -0.25, 0.5]},
                    "vec_txt": {"shape": [1, 1, 2], "values": [-0.75, 1.0]},
                },
                "variants": {"ignored": [{"values": [8.0] * 1000}]},
            },
        ],
        "variants": {"large_payload": [{"values": [7.0] * 1000}]},
    }
    encoded = json.dumps(synthetic, separators=(",", ":")).encode("utf-8")
    selected = _FixtureScanner(encoded).extract()
    config, cases = _validate_source(selected)
    if len(cases) != 2 or cases["broadcast"]["inputs"]["vec"]["values"] != [0.25, -0.5, 0.75, -1.0]:
        raise AssertionError("streaming fixture extraction did not retain vec inputs")
    if _case_shape_config(config, cases["broadcast"])["batches"] != 2:
        raise AssertionError("per-case batch override was not preserved")
    if len(_f32_record_sha256(cases["per_token"]["inputs"]["vec"])) != 64:
        raise AssertionError("input provenance hash was not produced")
    try:
        _FixtureScanner(b'{"schema_version":2,"schema_version":2}').extract()
    except DiagnosticError:
        pass
    else:
        raise AssertionError("duplicate top-level keys were admitted")
    print(json.dumps({"structure_self_test": "pass", "fixture_payloads_materialized": 0}, sort_keys=True))


def _arguments(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixture", type=Path, help="hash-pinned frozen block fixture JSON")
    parser.add_argument("--fixture-sha256", help="SHA-256 of exact source fixture bytes")
    parser.add_argument(
        "--upstream",
        type=Path,
        default=DEFAULT_UPSTREAM,
        help="pinned Hy-Nomas checkout at revision " + PINNED_REVISION,
    )
    parser.add_argument("--output", type=Path, help="new diagnostic JSON path")
    parser.add_argument(
        "--structure-self-test",
        action="store_true",
        help="run offline scanner and record guards without importing PyTorch or using MPS",
    )
    args = parser.parse_args(argv)
    if args.structure_self_test:
        return args
    if args.fixture is None or args.fixture_sha256 is None or args.output is None:
        parser.error("--fixture, --fixture-sha256, and --output are required")
    if re.fullmatch(r"[0-9a-fA-F]{64}", args.fixture_sha256) is None:
        parser.error("--fixture-sha256 must be exactly 64 hexadecimal characters")
    return args


def main(argv: list[str] | None = None) -> int:
    args = _arguments(argv)
    if args.structure_self_test:
        _structure_self_test()
        return 0

    fixture_path = args.fixture.expanduser().resolve(strict=True)
    output_path = _fresh_output(args.output)
    if not fixture_path.is_file() or fixture_path.suffix.lower() != ".json":
        raise DiagnosticError("--fixture must name a regular .json file")
    fixture_sha256 = _sha256(fixture_path)
    if fixture_sha256 != args.fixture_sha256.lower():
        raise DiagnosticError("source fixture SHA-256 does not match --fixture-sha256")

    try:
        with fixture_path.open("rb") as stream:
            with mmap.mmap(stream.fileno(), 0, access=mmap.ACCESS_READ) as mapped:
                selected = _FixtureScanner(mapped).extract()
    except (OSError, ValueError) as exc:
        raise DiagnosticError(f"could not scan source fixture: {exc}") from None
    config, selected_cases = _validate_source(selected)
    provenance = selected["provenance"]

    support_dir = Path(__file__).resolve().parent
    if str(support_dir) not in sys.path:
        sys.path.insert(0, str(support_dir))
    import hy_nomas_reference as oracle

    upstream = args.upstream.expanduser().resolve()
    try:
        upstream_hashes = oracle._verify_upstream(upstream)
    except (OSError, RuntimeError, ValueError) as exc:
        raise DiagnosticError(f"pinned upstream verification failed: {exc}") from None
    fixture_hashes = provenance.get("file_sha256")
    if fixture_hashes != upstream_hashes:
        raise DiagnosticError("source fixture pinned-source hashes do not match upstream")

    if os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK") not in (None, "0"):
        raise DiagnosticError("PYTORCH_ENABLE_MPS_FALLBACK must be unset or 0")
    os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "0"
    try:
        import torch
    except ImportError as exc:
        raise DiagnosticError("PyTorch is required to emit the MPS SiLU diagnostic") from exc
    if not re.match(r"^2\.8(?:\.|$)", str(torch.__version__)):
        raise DiagnosticError(f"PyTorch 2.8 is required, got {torch.__version__}")
    if not torch.backends.mps.is_built() or not torch.backends.mps.is_available():
        raise DiagnosticError("PyTorch MPS is required; CPU fallback is forbidden")
    fixture_torch_version = provenance.get("torch_version")
    if fixture_torch_version != torch.__version__:
        raise DiagnosticError(
            "source fixture and pointwise producer PyTorch versions differ: "
            f"{fixture_torch_version!r} != {torch.__version__!r}"
        )

    operations, input_hashes = _build_pointwise_records(
        torch, config, selected_cases, torch.device("mps"), upstream
    )
    if len(operations) != 4:
        raise DiagnosticError("pointwise inventory must contain exactly four SiLU operations")

    source = selected.get("source")
    source_checkpoint_sha256 = None
    if isinstance(source, dict):
        source_checkpoint_sha256 = source.get("file_sha256")
    result = {
        "schema_version": 1,
        "kind": REFERENCE_KIND,
        "upstream_revision": PINNED_REVISION,
        "source_fixture_sha256": fixture_sha256,
        "source_checkpoint_sha256": source_checkpoint_sha256,
        "weights_are_synthetic": selected.get("weights_are_synthetic"),
        "config": config,
        "provenance": {
            "upstream_revision": PINNED_REVISION,
            "upstream_source_sha256": upstream_hashes,
            "source_fixture_provenance": provenance,
            "reference_device": "mps",
            "compute_dtype": "F32",
            "operation": "pinned get_activation_layer('silu') -> torch.nn.SiLU",
            "boundary_diagnostic": "F32 output round-tripped through torch.bfloat16 to F32",
            "torch_version": torch.__version__,
            "producer": Path(__file__).name,
            "checkpoint_loaded": False,
            "block_instantiated": False,
            "input_provenance": "SHA-256 of exact source-input F32 little-endian bytes",
            "terminal_mps_synchronize_count": 1,
            "input_hashes_f32_le": input_hashes,
        },
        "operations": operations,
    }
    _atomic_write_fresh(output_path, result)
    print(
        json.dumps(
            {
                "status": "created",
                "output": str(output_path),
                "source_fixture_sha256": fixture_sha256,
                "operations": len(operations),
                "reference_device": "mps",
                "checkpoint_loaded": False,
                "block_instantiated": False,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except DiagnosticError as exc:
        print(f"hy_nomas_pointwise_reference: {exc}", file=sys.stderr)
        raise SystemExit(2) from None
