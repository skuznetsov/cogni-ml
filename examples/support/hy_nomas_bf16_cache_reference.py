#!/usr/bin/env python3
"""Stream the exact pinned Hy-Nomas block-0 weights into a BF16 cache.

The source is the authenticated F32 block-0 safetensors subset. Each tensor is
copied to MPS and converted there to BF16; the output keeps the original
``double_blocks.0.*`` names and shapes. This helper does not instantiate or
run a model. Its saved intermediate comparison scope is the authenticated
block-0 capture from the prior two-block upstream oracle, whose parameters
were BF16-rounded but stored and computed in F32.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import struct
import sys
from typing import Any, BinaryIO


ROOT = Path(__file__).resolve().parents[2]
SPEC_SUPPORT = ROOT / "spec" / "support"
if str(SPEC_SUPPORT) not in sys.path:
    sys.path.insert(0, str(SPEC_SUPPORT))

import hy_nomas_pretrained_serial_reference as serial_reference  # noqa: E402


SCHEMA = "hy-nomas-bf16-cache-reference-v1"
SERIAL_REFERENCE_SHA256 = "6bca7992819d7590f0226beff5ac12dcf76975af8ff4ab7d6de829a087a9fe8f"
SERIAL_SIDECAR_SHA256 = "3e8be928791a6e1308b377fa436f4f45ff3de6abfc64ecd11ba8ffbd52f01a35"
SERIAL_SIDECAR_BYTES = 13_006_848
DEFAULT_SERIAL_REFERENCE = Path(
    "/opt/mldata_ssd/cogni-ml/hy-nomas-pretrained-chain-20261025/fixtures/pretrained-serial-v1.json"
)
DEFAULT_BLOCK0_WEIGHTS = serial_reference.DEFAULT_BLOCK0_WEIGHTS
DEFAULT_BLOCK0_MANIFEST = serial_reference.DEFAULT_BLOCK0_MANIFEST
ATOL = serial_reference.ATOL
RTOL = serial_reference.RTOL
MAX_REFERENCE_BYTES = 8 * 1024 * 1024
MAX_MANIFEST_BYTES = 2 * 1024 * 1024

ReferenceError = serial_reference.ReferenceError


def _valid_sha256(value: Any) -> bool:
    return isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_json(path: Path, *, label: str, max_bytes: int) -> dict[str, Any]:
    if (path.is_symlink() or not path.is_file()
            or path.stat().st_size <= 0 or path.stat().st_size > max_bytes):
        raise ReferenceError(f"{label} must be a bounded regular file")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, RecursionError):
        raise ReferenceError(f"{label} is not valid UTF-8 JSON") from None
    if not isinstance(value, dict):
        raise ReferenceError(f"{label} root must be an object")
    return value


def validate_serial_reference_envelope(
    document: dict[str, Any],
    *,
    reference_sha256: str,
    sidecar_sha256: str,
    sidecar_nbytes: int,
) -> None:
    """Admit only the exact saved first-block capture used by native replay."""
    if reference_sha256 != SERIAL_REFERENCE_SHA256:
        raise ReferenceError("serial reference SHA is not the pinned capture")
    if (sidecar_sha256 != SERIAL_SIDECAR_SHA256
            or type(sidecar_nbytes) is not int
            or sidecar_nbytes != SERIAL_SIDECAR_BYTES):
        raise ReferenceError("serial reference sidecar pin mismatch")
    try:
        serial_reference.validate_reference_document(document)
    except (ReferenceError, KeyError, TypeError, ValueError) as exc:
        raise ReferenceError(f"serial reference validation failed: {exc}") from None

    producer = document.get("producer", {})
    handoff = producer.get("gpu_handoff") if isinstance(producer, dict) else None
    if (not isinstance(producer, dict)
            or producer.get("block_invocations") != ["block0", "block1"]
            or producer.get("attention_adapter_calls_by_block") != [1, 1]
            or not isinstance(handoff, dict)
            or handoff.get("same_mps_tensor_objects") is not True):
        raise ReferenceError("serial reference does not prove the saved two-block MPS run")
    block0 = document["source"]["blocks"]["block0"]
    pin = serial_reference.BLOCK_PINS["block0"]
    if (block0.get("block_index") != 0
            or block0.get("file_sha256") != pin["file_sha256"]
            or block0.get("manifest_sha256") != pin["manifest_sha256"]
            or block0.get("source_payload_sha256") != pin["source_payload_sha256"]
            or block0.get("tensor_prefix") != "double_blocks.0."
            or block0.get("tensor_count") != 32):
        raise ReferenceError("serial reference block0 source pin mismatch")
    if (document.get("tolerance") != {"atol": ATOL, "rtol": RTOL}
            or document.get("precision") != serial_reference.PRECISION):
        raise ReferenceError("serial reference precision or tolerance mismatch")


def _trusted_serial_reference(path_arg: Path, supplied_sha256: str) -> tuple[Path, dict[str, Any], Path]:
    if supplied_sha256 != SERIAL_REFERENCE_SHA256:
        raise ReferenceError("serial reference SHA is not the pinned capture")
    if path_arg.is_symlink():
        raise ReferenceError("serial reference symlinks are not admitted")
    path = path_arg.resolve(strict=True)
    if not path.is_file() or path.is_symlink() or path.stat().st_size > MAX_REFERENCE_BYTES:
        raise ReferenceError("serial reference must be a bounded regular JSON file")
    actual_sha256 = _sha256_file(path)
    if actual_sha256 != supplied_sha256:
        raise ReferenceError("serial reference SHA-256 mismatch")
    document = _load_json(path, label="serial reference", max_bytes=MAX_REFERENCE_BYTES)
    metadata = document.get("tensor_file")
    if not isinstance(metadata, dict):
        raise ReferenceError("serial reference tensor sidecar metadata is missing")
    sidecar_name = metadata.get("name")
    if (not isinstance(sidecar_name, str)
            or Path(sidecar_name).name != sidecar_name
            or not sidecar_name.endswith(".tensors.f32le")):
        raise ReferenceError("serial reference sidecar must be a sibling basename")
    sidecar_path = path.parent / sidecar_name
    if (sidecar_path.is_symlink() or not sidecar_path.is_file()
            or sidecar_path.stat().st_size != SERIAL_SIDECAR_BYTES):
        raise ReferenceError("serial reference sidecar is missing or has the wrong size")
    sidecar_sha256 = _sha256_file(sidecar_path)
    if sidecar_sha256 != SERIAL_SIDECAR_SHA256:
        raise ReferenceError("serial reference sidecar SHA-256 mismatch")
    validate_serial_reference_envelope(
        document,
        reference_sha256=actual_sha256,
        sidecar_sha256=sidecar_sha256,
        sidecar_nbytes=sidecar_path.stat().st_size,
    )
    return path, document, sidecar_path


def _block0_tensor_shapes() -> dict[str, list[int]]:
    relative = serial_reference.block_reference._expected_block_keys(2048, 8192, 16)
    return {f"double_blocks.0.{name}": shape for name, shape in sorted(relative.items())}


def build_safetensors_header(
    tensors: dict[str, tuple[list[int], int]],
) -> tuple[bytes, dict[str, dict[str, Any]]]:
    """Build a deterministic BF16 safetensors header and its data ranges."""
    if not isinstance(tensors, dict) or not tensors:
        raise ReferenceError("BF16 tensor inventory must be a non-empty mapping")
    offsets: dict[str, dict[str, Any]] = {}
    header_values: dict[str, Any] = {}
    offset = 0
    for name in sorted(tensors):
        shape, nbytes = tensors[name]
        if (not isinstance(name, str) or not name or name == "__metadata__"
                or not isinstance(shape, list) or not shape
                or any(type(dimension) is not int or dimension <= 0 for dimension in shape)
                or type(nbytes) is not int or nbytes <= 0):
            raise ReferenceError("invalid BF16 tensor name, shape, or byte extent")
        element_count = 1
        for dimension in shape:
            element_count *= dimension
        if nbytes != element_count * 2:
            raise ReferenceError(f"BF16 tensor byte extent mismatch: {name}")
        end = offset + nbytes
        record = {
            "dtype": "BF16",
            "shape": shape,
            "data_offsets": [offset, end],
        }
        header_values[name] = record
        offsets[name] = {
            "dtype": "BF16",
            "shape": shape,
            "offset_bytes": offset,
            "nbytes": nbytes,
        }
        offset = end
    header_values["__metadata__"] = {"format": "pt"}
    encoded = json.dumps(header_values, separators=(",", ":"), sort_keys=True).encode("utf-8")
    encoded += b" " * ((-len(encoded)) % 8)
    return struct.pack("<Q", len(encoded)) + encoded, offsets


def _validate_source_inventory(source: Any, expected: dict[str, list[int]]) -> None:
    actual = set(source.keys())
    if actual != set(expected):
        raise ReferenceError(
            "block0 source tensor inventory mismatch: "
            f"missing={sorted(set(expected) - actual)}, unexpected={sorted(actual - set(expected))}"
        )
    for name, shape in expected.items():
        tensor_slice = source.get_slice(name)
        if tensor_slice.get_dtype() != "F32" or tensor_slice.get_shape() != shape:
            raise ReferenceError(f"block0 source dtype/shape mismatch: {name}")


def _safetensors_module() -> Any:
    try:
        from safetensors import safe_open
    except ImportError as exc:
        raise ReferenceError("block0 cache creation requires safetensors") from exc
    return safe_open


def _prepare_outputs(output: Path, report: Path) -> tuple[Path, Path]:
    if (output.suffix.lower() != ".safetensors"
            or report.suffix.lower() != ".json"
            or output == report
            or not output.parent.is_dir()
            or not report.parent.is_dir()):
        raise ReferenceError("output and report must be fresh paths in existing directories")
    for label, path in (("output", output), ("report", report)):
        if os.path.lexists(path):
            raise FileExistsError(f"{label} already exists: {path}")
    output_tmp = output.with_name(f".{output.name}.{os.getpid()}.partial")
    report_tmp = report.with_name(f".{report.name}.{os.getpid()}.partial")
    for path in (output_tmp, report_tmp):
        if os.path.lexists(path):
            raise FileExistsError(f"temporary output already exists: {path}")
    return output_tmp, report_tmp


def _write_file_fsync(path: Path, writer: Any) -> None:
    created = False
    try:
        with path.open("xb") as stream:
            created = True
            writer(stream)
            stream.flush()
            os.fsync(stream.fileno())
    except Exception:
        if created:
            path.unlink(missing_ok=True)
        raise


def _publish_temp_noreplace(temporary: Path, destination: Path) -> None:
    if temporary.is_symlink() or not temporary.is_file():
        raise ReferenceError("publication staging file must be a regular file")
    if os.path.lexists(destination):
        raise FileExistsError(f"output already exists: {destination}")
    os.link(temporary, destination)
    temporary.unlink()


def publish_noreplace(path: Path, data: bytes) -> None:
    """Atomically publish a small file without replacing a prior artifact."""
    if path.is_symlink():
        raise ReferenceError(f"refusing symlink output: {path}")
    if not path.parent.is_dir():
        raise FileNotFoundError(f"output parent must already exist: {path.parent}")
    if os.path.lexists(path):
        raise FileExistsError(f"output already exists: {path}")
    temporary = path.with_name(f".{path.name}.{os.getpid()}.partial")
    temporary_owned = False
    try:
        _write_file_fsync(temporary, lambda stream: stream.write(data))
        temporary_owned = True
        _publish_temp_noreplace(temporary, path)
        temporary_owned = False
    finally:
        if temporary_owned:
            temporary.unlink(missing_ok=True)


def _capture_hashes(document: dict[str, Any], sidecar_path: Path) -> dict[str, dict[str, Any]]:
    try:
        import numpy as np
    except ImportError as exc:
        raise ReferenceError("validating the saved F32 captures requires NumPy") from exc
    tensors = document["tensors"]
    capture_hashes: dict[str, dict[str, Any]] = {}
    with sidecar_path.open("rb") as stream:
        for relative_name in serial_reference.BLOCK_CAPTURE_NAMES:
            name = f"block0.{relative_name}"
            metadata = tensors[name]
            offset = metadata["offset_bytes"]
            nbytes = metadata["nbytes"]
            stream.seek(offset)
            raw = stream.read(nbytes)
            if len(raw) != nbytes:
                raise ReferenceError(f"short read in saved block0 capture: {name}")
            if not bool(np.isfinite(np.frombuffer(raw, dtype="<f4")).all()):
                raise ReferenceError(f"non-finite value in saved block0 capture: {name}")
            capture_hashes[name] = {
                "dtype": "F32",
                "shape": list(metadata["shape"]),
                "nbytes": nbytes,
                "sha256": hashlib.sha256(raw).hexdigest(),
                "finite": True,
            }
    return capture_hashes


def _source_metadata(path: Path, digest: str, manifest_path: Path,
                      manifest_sha256: str, manifest: dict[str, Any]) -> dict[str, Any]:
    return {
        "block_index": 0,
        "path": str(path),
        "file_bytes": path.stat().st_size,
        "file_sha256": digest,
        "manifest_path": str(manifest_path),
        "manifest_sha256": manifest_sha256,
        "source": manifest["source"],
        "tensor_prefix": "double_blocks.0.",
        "tensor_count": 32,
    }


def _fixture_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path,
                        help="new BF16 safetensors cache path in an existing scratch directory")
    parser.add_argument("--report", required=True, type=Path,
                        help="new JSON report path in an existing scratch directory")
    parser.add_argument("--reference", type=Path, default=DEFAULT_SERIAL_REFERENCE)
    parser.add_argument("--reference-sha256", required=True)
    parser.add_argument("--block0-weights", type=Path, default=DEFAULT_BLOCK0_WEIGHTS)
    parser.add_argument("--block0-weights-sha256", required=True)
    parser.add_argument("--block0-manifest", type=Path, default=DEFAULT_BLOCK0_MANIFEST)
    parser.add_argument("--block0-manifest-sha256", required=True)
    return parser


def produce_cache(args: argparse.Namespace) -> dict[str, Any]:
    if os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK") not in (None, "0"):
        raise ReferenceError("PYTORCH_ENABLE_MPS_FALLBACK must be unset or 0")
    os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "0"

    output = args.output.absolute()
    report_path = args.report.absolute()
    output_tmp, report_tmp = _prepare_outputs(output, report_path)

    reference_path, reference_doc, reference_sidecar = _trusted_serial_reference(
        args.reference, args.reference_sha256
    )
    weights_path, weights_sha256, manifest_path, manifest = serial_reference._trusted_weight_pair(
        name="block0",
        weights=args.block0_weights,
        weights_sha256=args.block0_weights_sha256,
        manifest_path=args.block0_manifest,
        manifest_sha256=args.block0_manifest_sha256,
    )

    try:
        import torch
    except ImportError as exc:
        raise ReferenceError("block0 BF16 cache conversion requires PyTorch") from exc
    safe_open = _safetensors_module()
    if not torch.backends.mps.is_built() or not torch.backends.mps.is_available():
        raise ReferenceError("MPS is required; CPU fallback is forbidden")
    if torch.get_default_dtype() != torch.float32:
        torch.set_default_dtype(torch.float32)
    device = torch.device("mps")
    expected_shapes = _block0_tensor_shapes()
    source_values: dict[str, Any] = {}
    converted: dict[str, Any] = {}
    finite_flags: list[Any] = []
    finite_reduction: Any | None = None
    conversion_synchronized = False
    finite_readback_pending = False

    try:
        with safe_open(str(weights_path), framework="pt", device="cpu") as source:
            _validate_source_inventory(source, expected_shapes)
            for name in sorted(expected_shapes):
                source_tensor = source.get_tensor(name)
                if (source_tensor.dtype != torch.float32
                        or list(source_tensor.shape) != expected_shapes[name]):
                    raise ReferenceError(f"block0 source tensor is invalid: {name}")
                source_value = source_tensor.to(device=device, dtype=torch.float32).contiguous()
                if (source_value.device.type != "mps" or source_value.dtype != torch.float32
                        or list(source_value.shape) != expected_shapes[name]):
                    raise ReferenceError(f"MPS F32 source transfer changed tensor metadata: {name}")
                value = source_value.to(dtype=torch.bfloat16).contiguous()
                if (value.device.type != "mps" or value.dtype != torch.bfloat16
                        or list(value.shape) != expected_shapes[name]):
                    raise ReferenceError(f"MPS BF16 conversion changed tensor metadata: {name}")
                source_values[name] = source_value
                converted[name] = value
                finite_flags.append(torch.isfinite(source_value).all())
                del source_tensor, source_value, value
        # Enqueue the aggregate reduction before the sole explicit completion
        # point; only its one-element result is read back after the drain.
        finite_reduction = torch.stack(finite_flags).all()
        torch.mps.synchronize()
        conversion_synchronized = True
        finite_readback_pending = True
        finite_on_cpu = finite_reduction.to(device="cpu")
        finite_readback_pending = False
        if not bool(finite_on_cpu.item()):
            raise ReferenceError("non-finite value in MPS F32 source tensors")
        del finite_on_cpu
        source_values.clear()
        finite_flags.clear()
        finite_reduction = None
    except Exception as exc:
        if ((not conversion_synchronized or finite_readback_pending)
                and (source_values or converted or finite_flags or finite_reduction is not None)):
            try:
                torch.mps.synchronize()
            except Exception as drain_exc:
                raise ReferenceError(
                    "block0 BF16 conversion failed and MPS completion could not be confirmed: "
                    f"{type(drain_exc).__name__}: {drain_exc}"
                ) from None
        source_values.clear()
        converted.clear()
        finite_flags.clear()
        finite_reduction = None
        if isinstance(exc, ReferenceError):
            raise
        raise ReferenceError(
            f"block0 MPS BF16 conversion failed: {type(exc).__name__}: {exc}"
        ) from None

    tensor_specs = {
        name: (shape, math_product(shape) * 2)
        for name, shape in expected_shapes.items()
    }
    header, layout = build_safetensors_header(tensor_specs)
    tensor_records: dict[str, dict[str, Any]] = {}
    artifact_hash = hashlib.sha256()
    output_tmp_owned = False

    try:
        output_stream = output_tmp.open("xb")
        output_tmp_owned = True
        with output_stream as stream:
            stream.write(header)
            artifact_hash.update(header)
            for name in sorted(expected_shapes):
                cpu_tensor = converted[name].to(device="cpu").contiguous()
                if (cpu_tensor.dtype != torch.bfloat16
                        or list(cpu_tensor.shape) != expected_shapes[name]):
                    raise ReferenceError(f"BF16 host staging changed tensor metadata: {name}")
                raw = cpu_tensor.view(torch.uint16).numpy().astype("<u2", copy=False).tobytes()
                expected_nbytes = layout[name]["nbytes"]
                if len(raw) != expected_nbytes:
                    raise ReferenceError(f"BF16 byte extent mismatch: {name}")
                stream.write(raw)
                artifact_hash.update(raw)
                tensor_records[name] = {
                    **layout[name],
                    "source_dtype": "F32",
                    "sha256": hashlib.sha256(raw).hexdigest(),
                }
                del cpu_tensor, raw
            stream.flush()
            os.fsync(stream.fileno())
    except Exception:
        if output_tmp_owned:
            output_tmp.unlink(missing_ok=True)
            output_tmp_owned = False
        raise
    finally:
        converted.clear()

    expected_payload_bytes = sum(value[1] for value in tensor_specs.values())
    output_bytes = output_tmp.stat().st_size
    if output_bytes != len(header) + expected_payload_bytes:
        if output_tmp_owned:
            output_tmp.unlink(missing_ok=True)
            output_tmp_owned = False
        raise ReferenceError("BF16 safetensors output size mismatch")
    file_sha256 = _sha256_file(output_tmp)
    if file_sha256 != artifact_hash.hexdigest():
        if output_tmp_owned:
            output_tmp.unlink(missing_ok=True)
            output_tmp_owned = False
        raise ReferenceError("BF16 safetensors streamed and saved hashes differ")

    reference_tensor_file = reference_doc["tensor_file"]
    try:
        capture_tensors = _capture_hashes(reference_doc, reference_sidecar)
    except Exception:
        if output_tmp_owned:
            output_tmp.unlink(missing_ok=True)
            output_tmp_owned = False
        raise
    intermediate_reference = {
        "kind": "authenticated saved block0 upstream capture; block not rerun",
        "reference_path": str(reference_path),
        "reference_sha256": SERIAL_REFERENCE_SHA256,
        "tensor_file_name": reference_tensor_file["name"],
        "tensor_file_sha256": SERIAL_SIDECAR_SHA256,
        "tensor_file_nbytes": SERIAL_SIDECAR_BYTES,
        "source_schema": serial_reference.SCHEMA,
        "source_upstream_revision": serial_reference.PINNED_REVISION,
        "source_producer_sha256": reference_doc["producer"]["source_sha256"],
        "block_index": 0,
        "capture_tensor_count": len(serial_reference.BLOCK_CAPTURE_NAMES),
        "capture_tensors": capture_tensors,
        "tolerance": {"atol": ATOL, "rtol": RTOL},
        "precision": dict(serial_reference.PRECISION),
        "scope": "BF16-rounded parameter values in F32 storage and F32 compute; not physical BF16-resident upstream execution",
    }
    document = {
        "schema": SCHEMA,
        "schema_version": 1,
        "upstream_revision": serial_reference.PINNED_REVISION,
        "producer": {
            "path": str(Path(__file__).resolve()),
            "source_sha256": _sha256_file(Path(__file__).resolve()),
            "argv": list(sys.argv),
            "python": sys.version.split()[0],
            "torch": torch.__version__,
            "device": "mps",
            "conversion": "source F32 tensor copied to MPS and cast to BF16",
            "converted_tensor_count": len(tensor_records),
            "explicit_terminal_waits": 1,
            "model_invocations": 0,
            "weight_cache_host_readbacks": len(tensor_records),
        },
        "source": {
            "block0": _source_metadata(
                weights_path,
                weights_sha256,
                manifest_path,
                serial_reference.BLOCK_PINS["block0"]["manifest_sha256"],
                manifest,
            ),
            "serial_reference": {
                "path": str(reference_path),
                "reference_sha256": SERIAL_REFERENCE_SHA256,
                "tensor_file_name": reference_tensor_file["name"],
                "tensor_file_sha256": SERIAL_SIDECAR_SHA256,
                "tensor_file_nbytes": SERIAL_SIDECAR_BYTES,
            },
        },
        "cache": {
            "name": output.name,
            "nbytes": output_bytes,
            "sha256": file_sha256,
            "format": "safetensors",
            "dtype": "BF16",
            "tensor_count": len(tensor_records),
            "payload_nbytes": expected_payload_bytes,
            "tensors": tensor_records,
        },
        "intermediate_reference": intermediate_reference,
        "tolerance": {"atol": ATOL, "rtol": RTOL},
        "limitations": [
            "The helper converts checkpoint weights and does not instantiate or run a neural block.",
            "The saved block0 comparison is the prior authenticated MPS reference with BF16-rounded values held in F32 parameters and F32 compute.",
            "This helper authenticates and indexes the intermediate reference but performs no numerical comparison; the native probe owns that gate.",
            "This artifact does not establish full-model residency, BF16 activation parity, or generation behavior.",
        ],
    }
    json_bytes = (json.dumps(document, indent=2, sort_keys=True) + "\n").encode("utf-8")
    report_tmp_owned = False
    output_published = False
    report_published = False
    try:
        _write_file_fsync(report_tmp, lambda stream: stream.write(json_bytes))
        report_tmp_owned = True
        _publish_temp_noreplace(output_tmp, output)
        output_tmp_owned = False
        output_published = True
        _publish_temp_noreplace(report_tmp, report_path)
        report_tmp_owned = False
        report_published = True
    except Exception:
        if output_published and not report_published:
            output.unlink(missing_ok=True)
        raise
    finally:
        if output_tmp_owned:
            output_tmp.unlink(missing_ok=True)
        if report_tmp_owned:
            report_tmp.unlink(missing_ok=True)

    return {
        "status": "created",
        "output": str(output),
        "output_sha256": file_sha256,
        "report": str(report_path),
        "tensor_count": len(tensor_records),
        "payload_nbytes": expected_payload_bytes,
        "capture_count": len(intermediate_reference["capture_tensors"]),
        "explicit_terminal_waits": 1,
    }


def math_product(shape: list[int]) -> int:
    product = 1
    for dimension in shape:
        product *= dimension
    return product


def main(argv: list[str] | None = None) -> int:
    args = _fixture_parser().parse_args(argv)
    try:
        result = produce_cache(args)
    except (OSError, ValueError, ReferenceError, ImportError, RuntimeError) as exc:
        raise SystemExit(str(exc)) from exc
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
