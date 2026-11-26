#!/usr/bin/env python3
"""Build and compare one exact-input PyTorch MPS BF16 linear control.

The builder authenticates a saved first-step block-0 capture and pinned
upstream source, then runs the unchanged ModulateDiT SiLU and linear modules
on MPS. The native comparator checks the saved graph output byte-for-byte
against that PyTorch result; it does not use a numeric tolerance.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import sys
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
SUPPORT = ROOT / "spec" / "support"
if str(SUPPORT) not in sys.path:
    sys.path.insert(0, str(SUPPORT))

import hy_nomas_real_attention_reference as block_capture  # noqa: E402
import hy_nomas_reference as pinned_block  # noqa: E402


SCHEMA = "cogni_ml.hy_nomas_bf16_linear_case/v1"
NATIVE_SCHEMA = "cogni_ml.hy_nomas_bf16_linear_capture/v1"
PINNED_REVISION = "911cea6458a89c2663513ff7099561ec190dd491"
INPUT_WIDTH = 2048
OUTPUT_WIDTH = 12_288
OPERATION = "pinned img_mod.act(vec.to_bfloat16); img_mod.linear(silu)"
DEFAULT_UPSTREAM = pinned_block.DEFAULT_UPSTREAM
HEX = set("0123456789abcdef")


class ReferenceError(RuntimeError):
    """A fail-closed source, capture, or comparison error."""


def _digest(value: Any, label: str) -> str:
    if (not isinstance(value, str) or len(value) != 64
            or any(char not in HEX for char in value)):
        raise ReferenceError(f"{label} must be a lowercase SHA-256 digest")
    return value


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _regular_file(path_arg: str | Path, label: str) -> Path:
    path = Path(path_arg).expanduser()
    if not path.is_absolute():
        path = Path.cwd() / path
    try:
        resolved = path.resolve(strict=True)
    except OSError as error:
        raise ReferenceError(f"{label} does not exist: {path}") from error
    if not resolved.is_file() or path.is_symlink():
        raise ReferenceError(f"{label} must be a regular non-symlink file")
    return resolved


def _read_json(path: Path, label: str) -> dict[str, Any]:
    try:
        def unique(pairs):
            result = {}
            for key, value in pairs:
                if key in result:
                    raise ReferenceError(f"duplicate key in {label}: {key}")
                result[key] = value
            return result

        value = json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=unique)
    except ReferenceError:
        raise
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, RecursionError) as error:
        raise ReferenceError(f"invalid {label} JSON") from error
    if type(value) is not dict:
        raise ReferenceError(f"{label} root must be an object")
    return value


def _publish_noreplace(path: Path, data: bytes, label: str) -> None:
    try:
        with path.open("xb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
    except FileExistsError as error:
        raise ReferenceError(f"refusing to replace existing {label}: {path}") from error


def _capture_input_vec(capture: dict[str, Any], payload: Path) -> bytes:
    records = capture["tensors"]
    record = records["input.vec"]
    if (record["dtype"] != "F32" or record["shape"] != [INPUT_WIDTH]
            or type(record["offset"]) is not int or record["offset"] < 0
            or record["nbytes"] != INPUT_WIDTH * 4):
        raise ReferenceError("captured input.vec descriptor does not match [1,2048] F32")
    with payload.open("rb") as stream:
        stream.seek(record["offset"])
        data = stream.read(record["nbytes"])
    if len(data) != record["nbytes"] or _sha256_bytes(data) != record["sha256"]:
        raise ReferenceError("captured input.vec bytes do not match their descriptor")
    return data


def _load_pinned_modulation(torch: Any, upstream: Path,
                            block_weights: Path) -> Any:
    try:
        from safetensors import safe_open
    except ImportError as error:
        raise ReferenceError("safetensors is required for the pinned BF16 block") from error

    get_activation_layer, _norm, ModulateDiT, *_rest = pinned_block._install_upstream_modules(upstream)
    device = torch.device("meta")
    module = ModulateDiT(
        hidden_size=INPUT_WIDTH,
        factor=6,
        act_layer=get_activation_layer("silu"),
        dtype=torch.bfloat16,
        device=device,
    )
    expected = {
        "double_blocks.0.img_mod.linear.weight": (OUTPUT_WIDTH, INPUT_WIDTH),
        "double_blocks.0.img_mod.linear.bias": (OUTPUT_WIDTH,),
    }
    state = {}
    with safe_open(str(block_weights), framework="pt", device="cpu") as source:
        for key, shape in expected.items():
            tensor = source.get_tensor(key)
            if tensor.dtype != torch.bfloat16 or tuple(tensor.shape) != shape:
                raise ReferenceError(f"pinned modulation parameter dtype/shape mismatch: {key}")
            relative = key.removeprefix("double_blocks.0.img_mod.")
            state[relative] = tensor.contiguous()
    if set(state) != {"linear.weight", "linear.bias"}:
        raise ReferenceError("pinned modulation parameter inventory mismatch")
    module.load_state_dict(state, strict=True, assign=True)
    module.to(device=torch.device("mps"), dtype=torch.bfloat16).eval()
    return module


def build_case(args: argparse.Namespace) -> dict[str, Any]:
    output_parent = args.output.expanduser().parent.resolve(strict=True)
    args.output = output_parent / args.output.name
    payload_path = args.output.with_suffix(".tensors.bin")
    if args.output.suffix.lower() != ".json" or args.output.exists() or args.output.is_symlink():
        raise ReferenceError("--output must name a new .json file")
    if payload_path.exists() or payload_path.is_symlink():
        raise ReferenceError("refusing to replace an existing case tensor payload")

    capture_path = _regular_file(args.capture, "native first-step capture")
    capture_sha = _digest(args.capture_sha256, "capture SHA-256")
    capture = _read_json(capture_path, "native first-step capture")
    capture_payload, _names = block_capture._validate_flat_capture(
        capture, capture_path, capture_sha,
    )
    source_info, _prelude_inputs = block_capture._authenticate_sources(capture)
    block_capture._validate_capture_aliases(capture, capture_payload, source_info)
    upstream = args.upstream.resolve(strict=True)
    source_hashes = pinned_block._verify_upstream(upstream)

    fallback = os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK")
    if fallback not in (None, "0"):
        raise ReferenceError("PYTORCH_ENABLE_MPS_FALLBACK must be unset or 0")
    os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "0"
    try:
        import numpy as np
        import torch
    except ImportError as error:
        raise ReferenceError("NumPy and PyTorch are required to build the MPS case") from error
    if not torch.backends.mps.is_available():
        raise ReferenceError("PyTorch MPS is unavailable; refusing a CPU fallback")

    vec_bytes = _capture_input_vec(capture, capture_payload)
    vec_np = np.frombuffer(vec_bytes, dtype="<f4").copy().reshape(1, INPUT_WIDTH)
    vec = torch.from_numpy(vec_np).to(device="mps", dtype=torch.bfloat16)
    block = _load_pinned_modulation(torch, upstream, Path(source_info["block0_weights"]))
    with torch.inference_mode():
        silu = block.act(vec)
        expected = block.linear(silu)
    if (silu.device.type != "mps" or expected.device.type != "mps"
            or silu.dtype != torch.bfloat16 or expected.dtype != torch.bfloat16
            or tuple(silu.shape) != (1, INPUT_WIDTH)
            or tuple(expected.shape) != (1, OUTPUT_WIDTH)):
        raise ReferenceError("pinned PyTorch modulation did not return expected MPS BF16 tensors")

    input_bytes = silu.detach().to(device="cpu", dtype=torch.float32).contiguous().numpy().astype("<f4", copy=False).tobytes(order="C")
    expected_bytes = expected.detach().to(device="cpu", dtype=torch.float32).contiguous().numpy().astype("<f4", copy=False).tobytes(order="C")
    payload_bytes = vec_bytes + input_bytes + expected_bytes
    _publish_noreplace(payload_path, payload_bytes, "case tensor payload")

    records = {}
    cursor = 0
    for name, shape, data in (
        ("input.vec", [1, INPUT_WIDTH], vec_bytes),
        ("linear.input", [1, INPUT_WIDTH], input_bytes),
        ("linear.expected", [1, OUTPUT_WIDTH], expected_bytes),
    ):
        records[name] = {
            "dtype": "F32",
            "shape": shape,
            "offset": cursor,
            "nbytes": len(data),
            "sha256": _sha256_bytes(data),
        }
        cursor += len(data)
    document = {
        "schema": SCHEMA,
        "upstream_revision": PINNED_REVISION,
        "upstream_source_sha256": source_hashes,
        "operation": OPERATION,
        "torch_device": "mps",
        "torch_input_dtype": "bfloat16",
        "torch_linear_output_dtype": "bfloat16",
        "capture_manifest": str(capture_path),
        "capture_manifest_sha256": capture_sha,
        "capture_payload_sha256": capture["payload_sha256"],
        "input_vec_sha256": capture["tensors"]["input.vec"]["sha256"],
        "weight_index_sha256": capture["weight_index_sha256"],
        "block_weights_sha256": source_info["block0_weights_sha256"],
        "geometry": {"rows": 1, "input_width": INPUT_WIDTH, "output_width": OUTPUT_WIDTH},
        "payload": payload_path.name,
        "payload_bytes": len(payload_bytes),
        "payload_sha256": _sha256_bytes(payload_bytes),
        "tensors": records,
    }
    manifest_bytes = (json.dumps(document, indent=2, sort_keys=True, allow_nan=False) + "\n").encode("utf-8")
    _publish_noreplace(args.output, manifest_bytes, "case manifest")
    return {
        "case_manifest": str(args.output),
        "case_manifest_sha256": _sha256_bytes(manifest_bytes),
        "payload": str(payload_path),
        "payload_bytes": len(payload_bytes),
        "operation": OPERATION,
        "device": "mps",
        "output_dtype": "bfloat16",
    }


def _read_record(payload: Path, record: dict[str, Any], label: str,
                 expected_shape: list[int]) -> bytes:
    if (record.get("dtype") != "F32" or record.get("shape") != expected_shape
            or type(record.get("offset")) is not int or record["offset"] < 0
            or type(record.get("nbytes")) is not int
            or record["nbytes"] != 4 * math.prod(expected_shape)):
        raise ReferenceError(f"{label} descriptor mismatch")
    with payload.open("rb") as stream:
        stream.seek(record["offset"])
        data = stream.read(record["nbytes"])
    if len(data) != record["nbytes"] or _sha256_bytes(data) != _digest(record.get("sha256"), f"{label} SHA-256"):
        raise ReferenceError(f"{label} payload digest/extent mismatch")
    return data


def _verify_case_payload(case: dict[str, Any], payload: Path) -> None:
    geometry = case.get("geometry")
    if (type(geometry) is not dict or geometry != {
            "rows": 1, "input_width": INPUT_WIDTH, "output_width": OUTPUT_WIDTH,
    }):
        raise ReferenceError("case geometry mismatch")
    records = case.get("tensors")
    if type(records) is not dict or set(records) != {"input.vec", "linear.input", "linear.expected"}:
        raise ReferenceError("case tensor inventory mismatch")
    expected_layout = (
        ("input.vec", [1, INPUT_WIDTH], 0),
        ("linear.input", [1, INPUT_WIDTH], INPUT_WIDTH * 4),
        ("linear.expected", [1, OUTPUT_WIDTH], 2 * INPUT_WIDTH * 4),
    )
    for name, shape, offset in expected_layout:
        record = records[name]
        if type(record) is not dict or record.get("offset") != offset:
            raise ReferenceError(f"case tensor offset mismatch: {name}")
        _read_record(payload, record, name, shape)
    expected_bytes = 2 * INPUT_WIDTH * 4 + OUTPUT_WIDTH * 4
    if type(case.get("payload_bytes")) is not int or case["payload_bytes"] != expected_bytes:
        raise ReferenceError("case payload byte extent mismatch")


def compare_capture(args: argparse.Namespace) -> dict[str, Any]:
    case_path = _regular_file(args.case, "case manifest")
    case_sha = _digest(args.case_sha256, "case manifest SHA-256")
    if _sha256_file(case_path) != case_sha:
        raise ReferenceError("case manifest digest differs from supplied digest")
    case = _read_json(case_path, "case manifest")
    if (case.get("schema") != SCHEMA or case.get("upstream_revision") != PINNED_REVISION
            or case.get("operation") != OPERATION or case.get("torch_device") != "mps"
            or case.get("torch_input_dtype") != "bfloat16"
            or case.get("torch_linear_output_dtype") != "bfloat16"):
        raise ReferenceError("case manifest schema or upstream revision mismatch")
    case_payload = _regular_file(case_path.parent / case.get("payload", ""), "case payload")
    if (case_payload.parent != case_path.parent or case_payload.name != case.get("payload")
            or case_payload.stat().st_size != case.get("payload_bytes")
            or _sha256_file(case_payload) != _digest(case.get("payload_sha256"), "case payload SHA-256")):
        raise ReferenceError("case payload identity mismatch")
    _verify_case_payload(case, case_payload)
    native_path = _regular_file(args.native, "native result")
    native_sha = _digest(args.native_sha256, "native result SHA-256")
    if _sha256_file(native_path) != native_sha:
        raise ReferenceError("native result manifest digest differs from supplied digest")
    native = _read_json(native_path, "native result")
    if (native.get("schema") != NATIVE_SCHEMA
            or native.get("case_manifest") != str(case_path)
            or native.get("case_manifest_sha256") != case_sha
            or native.get("case_payload_sha256") != case.get("payload_sha256")
            or native.get("weight_index_sha256") != case.get("weight_index_sha256")
            or native.get("block_weights_sha256") != case.get("block_weights_sha256")
            or native.get("upstream_revision") != PINNED_REVISION
            or native.get("operation") != "resident BF16 img_mod.linear weights+bias; F32 storage input cast to BF16"
            or type(native.get("named_allocations_during_encode")) is not int
            or native.get("named_allocations_during_encode") != 0
            or type(native.get("activation_readbacks_during_encode")) is not int
            or native.get("activation_readbacks_during_encode") != 0
            or type(native.get("terminal_wait_count")) is not int
            or native.get("terminal_wait_count") != 1):
        raise ReferenceError("native capture does not bind to this authenticated case")
    geometry = case.get("geometry")
    if (type(geometry) is not dict or native.get("geometry") != geometry
            or native.get("variant") not in {"bf16_graph_ops", "f32_accumulation_bf16_output"}):
        raise ReferenceError("native result geometry or graph variant mismatch")
    native_payload_name = native.get("payload")
    if (not isinstance(native_payload_name, str) or Path(native_payload_name).name != native_payload_name
            or not native_payload_name.endswith(".tensors.bin")):
        raise ReferenceError("native result payload must be a sibling tensor file")
    native_payload = _regular_file(native_path.parent / native_payload_name, "native result payload")
    if native_payload.parent != native_path.parent:
        raise ReferenceError("native result payload must be adjacent to its manifest")
    native_records = native.get("tensors")
    if type(native_records) is not dict or set(native_records) != {"linear.output"}:
        raise ReferenceError("native result tensor inventory mismatch")
    native_record = native_records["linear.output"]
    expected_count = int(geometry["rows"]) * int(geometry["output_width"])
    expected_shape = [int(geometry["rows"]), int(geometry["output_width"])]
    expected_output_bytes = expected_count * 4
    if (type(native.get("payload_bytes")) is not int
            or native.get("payload_bytes") != expected_output_bytes):
        raise ReferenceError("native result payload has an unexpected extent")
    if (type(native_record) is not dict or type(native_record.get("offset")) is not int
            or native_record.get("offset") != 0):
        raise ReferenceError("native output must occupy the complete sibling payload")
    actual_bytes = _read_record(native_payload, native_record, "linear.output", expected_shape)
    if (native_payload.stat().st_size != native.get("payload_bytes")
            or _sha256_file(native_payload) != _digest(native.get("payload_sha256"), "native payload SHA-256")):
        raise ReferenceError("native result payload manifest mismatch")
    expected_record = case.get("tensors", {}).get("linear.expected")
    expected_bytes = _read_record(case_payload, expected_record, "linear.expected", expected_shape)
    if len(actual_bytes) != len(expected_bytes) or len(actual_bytes) != expected_count * 4:
        raise ReferenceError("actual and expected output extents differ")
    import numpy as np
    bad_count = 0
    first_bad = None
    max_abs = 0.0
    actual_values = np.frombuffer(actual_bytes, dtype="<f4")
    expected_values = np.frombuffer(expected_bytes, dtype="<f4")
    if not np.isfinite(actual_values).all() or not np.isfinite(expected_values).all():
        raise ReferenceError("BF16 linear comparison requires finite actual and expected values")
    expected_non_bf16 = int(np.count_nonzero(expected_values.view("<u4") & 0xFFFF))
    actual_non_bf16 = int(np.count_nonzero(actual_values.view("<u4") & 0xFFFF))
    if expected_non_bf16:
        raise ReferenceError("Torch BF16 case contains values not representable in BF16")
    chunk_elements = 1_048_576
    for offset in range(0, expected_count, chunk_elements):
        stop = min(expected_count, offset + chunk_elements)
        actual_chunk = actual_values[offset:stop]
        expected_chunk = expected_values[offset:stop]
        different = actual_chunk.view("<u4") != expected_chunk.view("<u4")
        count = int(np.count_nonzero(different))
        if count:
            if first_bad is None:
                first_bad = offset + int(np.flatnonzero(different)[0])
            bad_count += count
            delta = np.abs(actual_chunk[different] - expected_chunk[different])
            chunk_max = float(np.nan_to_num(delta, nan=0.0, posinf=float("inf")).max(initial=0.0))
            max_abs = max(max_abs, chunk_max)
    return {
        "bitwise_equal": bad_count == 0,
        "variant": native["variant"],
        "elements": expected_count,
        "different_elements": bad_count,
        "first_difference_flat_index": first_bad,
        "max_abs_difference": max_abs,
        "expanded_bf16_storage": actual_non_bf16 == 0,
        "nonzero_low16_elements": actual_non_bf16,
        "comparison": "raw F32 output bytes; no tolerance",
    }


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(description=__doc__)
    subparsers = root.add_subparsers(dest="command", required=True)
    build = subparsers.add_parser("build", help="run the pinned Torch MPS BF16 linear")
    build.add_argument("--capture", type=Path, required=True)
    build.add_argument("--capture-sha256", required=True)
    build.add_argument("--upstream", type=Path, default=DEFAULT_UPSTREAM)
    build.add_argument("--output", type=Path, required=True)
    build.set_defaults(function=build_case)
    compare = subparsers.add_parser("compare", help="compare native output bytes to Torch MPS")
    compare.add_argument("--case", type=Path, required=True)
    compare.add_argument("--case-sha256", required=True)
    compare.add_argument("--native", type=Path, required=True)
    compare.add_argument("--native-sha256", required=True)
    compare.set_defaults(function=compare_capture)
    return root


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    try:
        result = args.function(args)
    except (ReferenceError, block_capture.ReferenceError, RuntimeError, OSError, ValueError) as error:
        raise SystemExit(str(error)) from error
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
