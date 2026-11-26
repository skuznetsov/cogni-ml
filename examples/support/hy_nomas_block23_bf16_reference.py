#!/usr/bin/env python3
"""Convert one authenticated original Hy-Nomas block 2/3 from F32 to MPS BF16.

This is a tensorwise conversion reference, not a neural-block execution. CPU
admission authenticates the complete selected subset and manifest before Torch
is imported. The only conversion is F32 transfer to MPS followed by MPS BF16
cast; output tensors are serialized from MPS readbacks into safetensors.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import stat
import struct
import sys
import tempfile
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
SPEC_SUPPORT = ROOT / "spec" / "support"
if str(SPEC_SUPPORT) not in sys.path:
    sys.path.insert(0, str(SPEC_SUPPORT))

import hy_nomas_block23_checkpoint_subset as subset  # noqa: E402


ReferenceError = subset.SubsetError
SCHEMA = "hy-nomas-block23-bf16-reference-v1"
MAX_MANIFEST_BYTES = 2 * 1024 * 1024
MAX_HEADER_BYTES = subset.base.MAX_HEADER_BYTES
ALLOWED_ARTIFACT_ROOT = subset.ALLOWED_OUTPUT_ROOT
DEFAULT_WEIGHTS = {
    index: ALLOWED_ARTIFACT_ROOT / "weights" / f"distilled-block{index}-f32.safetensors"
    for index in (2, 3)
}
DEFAULT_MANIFESTS = {
    index: Path(str(DEFAULT_WEIGHTS[index]) + ".manifest.json") for index in (2, 3)
}
DEFAULT_CACHE = {
    index: ALLOWED_ARTIFACT_ROOT / "cache" / f"distilled-block{index}-bf16.safetensors"
    for index in (2, 3)
}
DEFAULT_REPORT = {
    index: ALLOWED_ARTIFACT_ROOT / "fixtures" / f"block{index}-bf16-reference-v1.json"
    for index in (2, 3)
}

# Derived from the authenticated source block-0 contract and then renamed to
# the actual target index. Values are immutable by convention; tests may patch
# the mapping to tiny shapes while exercising the CPU admission path.
BLOCK_TENSOR_SHAPES: dict[int, dict[str, list[int]]] = {
    index: {name: list(shape) for name, shape in subset.expected_tensors(index).items()}
    for index in (2, 3)
}
_UNCONFIRMED_MPS_REFERENCES: list[tuple[Any, ...]] = []


def checked_block_index(value: Any) -> int:
    try:
        return subset.checked_block_index(value)
    except ReferenceError:
        raise ReferenceError("block index must identify actual block 2 or 3") from None


def _block_index_argument(text: str) -> int:
    try:
        return checked_block_index(int(text))
    except ValueError as exc:
        raise argparse.ArgumentTypeError("block index must be 2 or 3") from exc
    except ReferenceError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from exc


def block_tensor_shapes(block_index: int) -> dict[str, list[int]]:
    block_index = checked_block_index(block_index)
    return dict(BLOCK_TENSOR_SHAPES[block_index])


def shape_product(shape: list[int] | tuple[int, ...]) -> int:
    if (not isinstance(shape, (list, tuple)) or not shape
            or any(type(dimension) is not int or dimension <= 0 for dimension in shape)):
        raise ReferenceError("tensor shape must contain positive integer dimensions")
    product = 1
    for dimension in shape:
        product *= dimension
        if product > subset.base.SOURCE_FILE_SIZE:
            raise ReferenceError("tensor shape exceeds the pinned checkpoint size")
    return product


def block_payload_bytes(block_index: int) -> int:
    return sum(shape_product(shape) * 4 for shape in block_tensor_shapes(block_index).values())


def source_header_metadata(block_index: int) -> dict[str, str]:
    block_index = checked_block_index(block_index)
    base = subset.base
    return {
        "format": "pt",
        "hy_nomas_hf_revision": base.HF_REVISION,
        "hy_nomas_source_file": base.CHECKPOINT_PATH,
        "hy_nomas_source_lfs_sha256_declared": base.SOURCE_LFS_SHA256,
        "hy_nomas_source_lfs_sha256_verified": "false",
        "hy_nomas_source_git_commit": base.SOURCE_GIT_COMMIT,
        "hy_nomas_block_index": str(block_index),
        "hy_nomas_subset": f"double_blocks.{block_index} F32 stored bytes; no conversion",
    }


def _valid_sha256(value: Any) -> bool:
    return isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_json_bytes(raw: bytes, *, label: str) -> dict[str, Any]:
    try:
        value = subset.base._json_object(raw, context=label)
    except (ReferenceError, TypeError, ValueError) as exc:
        raise ReferenceError(f"{label} is not valid bounded JSON: {exc}") from None
    return value


def validate_manifest_document(
    document: dict[str, Any],
    *,
    block_index: int,
    source_path: Path,
    source_sha256: str,
    source_file_bytes: int,
) -> dict[str, Any]:
    """Check the subset extractor's provenance schema against the selected block."""
    block_index = checked_block_index(block_index)
    if not isinstance(document, dict) or document.get("format") != "hy-nomas-block-subset-provenance-v1":
        raise ReferenceError("source manifest format mismatch")
    source = document.get("source")
    subset_record = document.get("subset")
    output = document.get("output")
    if not all(isinstance(value, dict) for value in (source, subset_record, output)):
        raise ReferenceError("source manifest provenance sections are missing")

    base = subset.base
    expected_source = {
        "repository": base.HF_REPO,
        "revision": base.HF_REVISION,
        "checkpoint_path": base.CHECKPOINT_PATH,
        "checkpoint_size_bytes": base.SOURCE_FILE_SIZE,
        "checkpoint_lfs_sha256": base.SOURCE_LFS_SHA256,
        "checkpoint_lfs_sha256_status": (
            "declared by pinned HF API; whole file not downloaded or verified"
        ),
        "source_git_commit": base.SOURCE_GIT_COMMIT,
        "header_sha256_observed": subset.EXPECTED_HEADER_SHA256,
        "range_start_inclusive": subset.BLOCK_HTTP_RANGES[block_index][0],
        "range_end_inclusive": subset.BLOCK_HTTP_RANGES[block_index][1],
        "authentication_precondition": "HTTPS/TLS to pinned HF revision and its HTTPS range redirects",
    }
    for key, expected in expected_source.items():
        if source.get(key) != expected:
            raise ReferenceError(f"source manifest source {key} mismatch")
    payload_sha256 = source.get("payload_sha256_observed")
    if not _valid_sha256(payload_sha256):
        raise ReferenceError("source manifest payload digest is invalid")

    names = sorted(block_tensor_shapes(block_index))
    expected_subset = {
        "block_index": block_index,
        "tensor_prefix": f"double_blocks.{block_index}.",
        "tensor_count": 32,
        "tensor_names": names,
        "dtype": "F32",
        "stored_bytes_preserved": True,
        "conversion_performed": False,
        "payload_bytes": block_payload_bytes(block_index),
    }
    for key, expected in expected_subset.items():
        if subset_record.get(key) != expected:
            raise ReferenceError(f"source manifest subset {key} mismatch")
    if (output.get("filename") != source_path.name
            or output.get("size_bytes") != source_file_bytes
            or output.get("sha256") != source_sha256):
        raise ReferenceError("source manifest output digest, size, or filename mismatch")
    return document


def _file_identity(path: Path) -> tuple[int, int, int, int, int]:
    value = path.stat(follow_symlinks=False)
    if not stat.S_ISREG(value.st_mode) or path.is_symlink():
        raise ReferenceError("source and manifest must be regular non-symlink files")
    return value.st_dev, value.st_ino, value.st_size, value.st_mtime_ns, value.st_ctime_ns


def _same_file(left: Path, right: Path) -> bool:
    try:
        return os.path.samefile(left, right)
    except FileNotFoundError:
        return False


def _read_source_header(path: Path) -> tuple[dict[str, Any], int, int]:
    with path.open("rb") as stream:
        prefix = stream.read(8)
        if len(prefix) != 8:
            raise ReferenceError("source safetensors prefix is truncated")
        header_nbytes = struct.unpack("<Q", prefix)[0]
        if (header_nbytes <= 0 or header_nbytes > MAX_HEADER_BYTES
                or header_nbytes % 8 != 0):
            raise ReferenceError("source safetensors header length is invalid")
        raw = stream.read(header_nbytes)
        if len(raw) != header_nbytes:
            raise ReferenceError("source safetensors header is truncated")
    header = _load_json_bytes(raw, label="source safetensors header")
    return header, header_nbytes, 8 + header_nbytes


class SourceTensor:
    __slots__ = ("name", "shape", "start", "end")

    def __init__(self, name: str, shape: list[int], start: int, end: int):
        self.name = name
        self.shape = shape
        self.start = start
        self.end = end


class SourceInventory:
    __slots__ = ("block_index", "path", "file_bytes", "header_bytes", "data_offset",
                 "payload_bytes", "payload_sha256", "file_sha256", "tensors",
                 "file_identity", "manifest_path", "manifest_sha256", "manifest")

    def __init__(self, *, block_index: int, path: Path, file_bytes: int, header_bytes: int,
                 data_offset: int, payload_bytes: int, payload_sha256: str, file_sha256: str,
                 tensors: tuple[SourceTensor, ...], file_identity: tuple[int, int, int, int, int],
                 manifest_path: Path, manifest_sha256: str, manifest: dict[str, Any]):
        self.block_index = block_index
        self.path = path
        self.file_bytes = file_bytes
        self.header_bytes = header_bytes
        self.data_offset = data_offset
        self.payload_bytes = payload_bytes
        self.payload_sha256 = payload_sha256
        self.file_sha256 = file_sha256
        self.tensors = tensors
        self.file_identity = file_identity
        self.manifest_path = manifest_path
        self.manifest_sha256 = manifest_sha256
        self.manifest = manifest


def parse_source_inventory(path: Path, block_index: int) -> SourceInventory:
    """Validate names, header metadata, F32 shapes and complete payload coverage."""
    block_index = checked_block_index(block_index)
    source_path = Path(path)
    identity_before = _file_identity(source_path)
    expected = block_tensor_shapes(block_index)
    expected_payload = block_payload_bytes(block_index)
    header, header_nbytes, data_offset = _read_source_header(source_path)
    metadata = header.pop("__metadata__", None)
    if metadata != source_header_metadata(block_index):
        raise ReferenceError(f"block{block_index} source header metadata mismatch")
    if set(header) != set(expected):
        raise ReferenceError(f"block{block_index} source tensor inventory mismatch")

    records: list[SourceTensor] = []
    for name, shape in expected.items():
        descriptor = header.get(name)
        if not isinstance(descriptor, dict) or set(descriptor) != {"dtype", "shape", "data_offsets"}:
            raise ReferenceError(f"invalid source descriptor: {name}")
        offsets = descriptor["data_offsets"]
        source_shape = descriptor.get("shape")
        if (descriptor.get("dtype") != "F32" or not isinstance(source_shape, list)
                or any(type(dimension) is not int for dimension in source_shape)
                or source_shape != shape
                or not isinstance(offsets, list) or len(offsets) != 2
                or any(type(value) is not int for value in offsets)):
            raise ReferenceError(f"source dtype, shape, or offsets mismatch: {name}")
        start, end = offsets
        tensor_bytes = shape_product(shape) * 4
        if start < 0 or end - start != tensor_bytes:
            raise ReferenceError(f"source tensor extent mismatch: {name}")
        records.append(SourceTensor(name, list(shape), start, end))

    records.sort(key=lambda record: (record.start, record.end, record.name))
    cursor = 0
    for record in records:
        if record.start != cursor or record.end <= record.start:
            raise ReferenceError("source tensor payload must be disjoint and contiguous")
        cursor = record.end
    if cursor != expected_payload:
        raise ReferenceError("source tensor payload coverage mismatch")

    file_bytes = source_path.stat().st_size
    if file_bytes != data_offset + expected_payload:
        raise ReferenceError("source safetensors file size or trailing bytes mismatch")
    payload_digest = hashlib.sha256()
    with source_path.open("rb") as stream:
        stream.seek(data_offset)
        remaining = expected_payload
        while remaining:
            chunk = stream.read(min(8 * 1024 * 1024, remaining))
            if not chunk:
                raise ReferenceError("source safetensors payload is truncated")
            payload_digest.update(chunk)
            remaining -= len(chunk)
        if stream.read(1):
            raise ReferenceError("source safetensors has trailing bytes")
    identity_after = _file_identity(source_path)
    if identity_after != identity_before:
        raise ReferenceError("source safetensors changed during validation")
    file_sha256 = _sha256_file(source_path)
    if _file_identity(source_path) != identity_before:
        raise ReferenceError("source safetensors changed while hashing")
    return SourceInventory(
        block_index=block_index,
        path=source_path.resolve(strict=True),
        file_bytes=file_bytes,
        header_bytes=header_nbytes,
        data_offset=data_offset,
        payload_bytes=expected_payload,
        payload_sha256=payload_digest.hexdigest(),
        file_sha256=file_sha256,
        tensors=tuple(records),
        file_identity=identity_before,
        manifest_path=Path(""),
        manifest_sha256="",
        manifest={},
    )


def validate_source_pair(
    *,
    weights: Path,
    weights_sha256: str,
    manifest_path: Path,
    manifest_sha256: str,
    block_index: int,
) -> SourceInventory:
    """Authenticate caller-pinned subset+manifest before any Torch import."""
    block_index = checked_block_index(block_index)
    if not _valid_sha256(weights_sha256) or not _valid_sha256(manifest_sha256):
        raise ReferenceError("caller source and manifest SHA-256 pins are required")
    weights_arg = Path(weights).expanduser()
    manifest_arg = Path(manifest_path).expanduser()
    if weights_arg.is_symlink() or manifest_arg.is_symlink():
        raise ReferenceError("source checkpoint and manifest symlinks are not admitted")
    weights_resolved = weights_arg.resolve(strict=True)
    manifest_resolved = manifest_arg.resolve(strict=True)
    weight_identity = _file_identity(weights_resolved)
    manifest_identity = _file_identity(manifest_resolved)
    if _same_file(weights_resolved, manifest_resolved):
        raise ReferenceError("source checkpoint and manifest must be distinct files")
    if weights_resolved.suffix != ".safetensors":
        raise ReferenceError("source weights must be a safetensors file")
    if manifest_identity[2] <= 0 or manifest_identity[2] > MAX_MANIFEST_BYTES:
        raise ReferenceError("source manifest exceeds its size bound")
    with manifest_resolved.open("rb") as stream:
        manifest_bytes = stream.read(MAX_MANIFEST_BYTES + 1)
    if len(manifest_bytes) > MAX_MANIFEST_BYTES or len(manifest_bytes) != manifest_identity[2]:
        raise ReferenceError("source manifest exceeds its size bound or changed while reading")
    if _file_identity(manifest_resolved) != manifest_identity:
        raise ReferenceError("source manifest changed while reading")
    observed_manifest_sha = hashlib.sha256(manifest_bytes).hexdigest()
    if observed_manifest_sha != manifest_sha256:
        raise ReferenceError("source manifest SHA-256 mismatch")
    manifest = _load_json_bytes(manifest_bytes, label="source manifest")

    inventory = parse_source_inventory(weights_resolved, block_index)
    if inventory.file_sha256 != weights_sha256:
        raise ReferenceError("source safetensors SHA-256 mismatch")
    validate_manifest_document(
        manifest,
        block_index=block_index,
        source_path=weights_resolved,
        source_sha256=inventory.file_sha256,
        source_file_bytes=inventory.file_bytes,
    )
    if inventory.payload_sha256 != manifest["source"]["payload_sha256_observed"]:
        raise ReferenceError("source payload SHA-256 differs from its manifest")
    if _file_identity(weights_resolved) != weight_identity:
        raise ReferenceError("source safetensors changed during admission")
    inventory.manifest_path = manifest_resolved
    inventory.manifest_sha256 = observed_manifest_sha
    inventory.manifest = manifest
    return inventory


def validate_source_unchanged(inventory: SourceInventory) -> None:
    """Fail if a source path was replaced or modified after admission."""
    if _file_identity(inventory.path) != inventory.file_identity:
        raise ReferenceError("source safetensors changed after admission")


def build_bf16_header(block_index: int) -> tuple[bytes, dict[str, dict[str, Any]]]:
    """Create a deterministic BF16 safetensors header for all selected tensors."""
    block_index = checked_block_index(block_index)
    header: dict[str, Any] = {
        "__metadata__": {
            "format": "pt",
            "hy_nomas_block_index": str(block_index),
            "hy_nomas_subset": f"double_blocks.{block_index} MPS F32->BF16 tensorwise conversion",
        }
    }
    records: dict[str, dict[str, Any]] = {}
    cursor = 0
    for name, shape in sorted(block_tensor_shapes(block_index).items()):
        nbytes = shape_product(shape) * 2
        header[name] = {"dtype": "BF16", "shape": shape,
                        "data_offsets": [cursor, cursor + nbytes]}
        records[name] = {"dtype": "BF16", "shape": shape,
                         "offset_bytes": cursor, "nbytes": nbytes}
        cursor += nbytes
    encoded = json.dumps(header, sort_keys=True, separators=(",", ":")).encode("utf-8")
    encoded += b" " * ((-len(encoded)) % 8)
    if len(encoded) > MAX_HEADER_BYTES:
        raise ReferenceError("BF16 safetensors header exceeds its bound")
    return struct.pack("<Q", len(encoded)) + encoded, records


def _prepare_outputs(output: Path, report: Path) -> tuple[Path, Path]:
    output = output.absolute()
    report = report.absolute()
    if (output == report or output.suffix.lower() != ".safetensors"
            or report.suffix.lower() != ".json"
            or not output.parent.is_dir() or not report.parent.is_dir()):
        raise ReferenceError("output and report must be distinct files in existing directories")
    if os.path.lexists(output) or os.path.lexists(report):
        raise FileExistsError("output or report already exists")
    if _same_file(output.parent, report.parent) and output.name == report.name:
        raise ReferenceError("output and report paths alias")
    return output, report


def _new_temp(destination: Path) -> tuple[int, Path]:
    fd, name = tempfile.mkstemp(prefix=f".{destination.name}.", suffix=".partial",
                                dir=destination.parent)
    return fd, Path(name)


def _link_temp_no_replace(temporary: Path, destination: Path) -> tuple[int, int]:
    value = temporary.lstat()
    if not stat.S_ISREG(value.st_mode) or temporary.is_symlink():
        raise ReferenceError("publication staging path must be a regular file")
    os.link(temporary, destination)
    return value.st_dev, value.st_ino


def _remove_if_same(path: Path, identity: tuple[int, int] | None) -> None:
    if identity is None:
        return
    try:
        current = path.lstat()
        if (current.st_dev, current.st_ino) == identity:
            path.unlink()
            subset.base._fsync_directory(path.parent)
    except FileNotFoundError:
        pass


def publish_output_pair(
    output_temp: Path,
    output: Path,
    report_temp: Path,
    report: Path,
) -> tuple[tuple[int, int], tuple[int, int]]:
    """No-replace publish both outputs, rolling back only files this call linked."""
    output_identity: tuple[int, int] | None = None
    report_identity: tuple[int, int] | None = None
    try:
        output_identity = _link_temp_no_replace(output_temp, output)
        report_identity = _link_temp_no_replace(report_temp, report)
        subset.base._fsync_directory(output.parent)
        if report.parent != output.parent:
            subset.base._fsync_directory(report.parent)
        return output_identity, report_identity
    except BaseException:
        _remove_if_same(report, report_identity)
        _remove_if_same(output, output_identity)
        raise
    finally:
        output_temp.unlink(missing_ok=True)
        report_temp.unlink(missing_ok=True)


def _safetensors_module() -> Any:
    try:
        from safetensors import safe_open
    except ImportError as exc:
        raise ReferenceError("BF16 conversion requires safetensors") from exc
    return safe_open


def _validate_safe_source(source: Any, block_index: int) -> None:
    expected = block_tensor_shapes(block_index)
    actual = set(source.keys())
    if actual != set(expected):
        raise ReferenceError(f"block{block_index} safetensors inventory changed after admission")
    for name, shape in expected.items():
        item = source.get_slice(name)
        if item.get_dtype() != "F32" or item.get_shape() != shape:
            raise ReferenceError(f"block{block_index} safetensors metadata mismatch: {name}")


def _to_cpu_bytes(tensor: Any, torch: Any, name: str, expected_shape: list[int]) -> bytes:
    cpu_tensor = tensor.to(device="cpu").contiguous()
    if (cpu_tensor.dtype != torch.bfloat16
            or list(cpu_tensor.shape) != expected_shape):
        raise ReferenceError(f"BF16 host readback changed tensor metadata: {name}")
    raw = cpu_tensor.view(torch.uint16).numpy().astype("<u2", copy=False).tobytes()
    return raw


def _write_bytes_fsync(path: Path, raw: bytes) -> None:
    with path.open("xb") as stream:
        stream.write(raw)
        stream.flush()
        os.fsync(stream.fileno())


def produce_cache(args: argparse.Namespace) -> dict[str, Any]:
    block_index = checked_block_index(getattr(args, "block_index", 2))
    weights_path = Path(getattr(args, "weights", None) or DEFAULT_WEIGHTS[block_index])
    manifest_path = Path(getattr(args, "manifest", None) or DEFAULT_MANIFESTS[block_index])
    output, report = _prepare_outputs(
        Path(getattr(args, "output", DEFAULT_CACHE[block_index])),
        Path(getattr(args, "report", DEFAULT_REPORT[block_index])),
    )
    weights_sha256 = getattr(args, "weights_sha256", None)
    manifest_sha256 = getattr(args, "manifest_sha256", None)
    if not _valid_sha256(weights_sha256) or not _valid_sha256(manifest_sha256):
        raise ReferenceError("caller source and manifest SHA-256 pins are required")

    # Fully validate the source and manifest before importing Torch or touching MPS.
    inventory = validate_source_pair(
        weights=weights_path,
        weights_sha256=weights_sha256,
        manifest_path=manifest_path,
        manifest_sha256=manifest_sha256,
        block_index=block_index,
    )
    try:
        import torch
    except ImportError as exc:
        raise ReferenceError("BF16 conversion requires PyTorch") from exc
    safe_open = _safetensors_module()
    if os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK") not in (None, "0"):
        raise ReferenceError("PYTORCH_ENABLE_MPS_FALLBACK must be unset or 0")
    os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "0"
    if not torch.backends.mps.is_built() or not torch.backends.mps.is_available():
        raise ReferenceError("MPS is required; CPU fallback is forbidden")
    if torch.get_default_dtype() != torch.float32:
        torch.set_default_dtype(torch.float32)

    expected_shapes = block_tensor_shapes(block_index)
    output_header, output_layout = build_bf16_header(block_index)
    converted: dict[str, Any] = {}
    source_values: dict[str, Any] = {}
    finite_flags: list[Any] = []
    finite_reduction: Any | None = None
    terminal_wait_completed = False
    conversion_started = False
    output_temp: Path | None = None
    report_temp: Path | None = None
    output_temp_owned = False
    report_temp_owned = False
    output_identity: tuple[int, int] | None = None
    report_identity: tuple[int, int] | None = None
    post_drain_mps_work_pending = False
    completion_uncertain = False
    try:
        with safe_open(str(inventory.path), framework="pt", device="cpu") as source:
            _validate_safe_source(source, block_index)
            for name in sorted(expected_shapes):
                source_tensor = source.get_tensor(name)
                if (source_tensor.dtype != torch.float32
                        or list(source_tensor.shape) != expected_shapes[name]):
                    raise ReferenceError(f"source tensor metadata mismatch: {name}")
                conversion_started = True
                source_value = source_tensor.to(device="mps", dtype=torch.float32).contiguous()
                if (source_value.device.type != "mps" or source_value.dtype != torch.float32
                        or list(source_value.shape) != expected_shapes[name]):
                    raise ReferenceError(f"MPS source transfer changed tensor metadata: {name}")
                source_values[name] = source_value
                value = source_value.to(dtype=torch.bfloat16).contiguous()
                if (value.device.type != "mps" or value.dtype != torch.bfloat16
                        or list(value.shape) != expected_shapes[name]):
                    raise ReferenceError(f"MPS BF16 conversion changed tensor metadata: {name}")
                converted[name] = value
                finite_flags.append(torch.isfinite(source_value).all())
                finite_flags.append(torch.isfinite(value).all())
                del source_tensor, source_value, value

        validate_source_unchanged(inventory)

        finite_reduction = torch.stack(finite_flags).all()
        torch.mps.synchronize()
        terminal_wait_completed = True
        finite_on_cpu = finite_reduction.to(device="cpu")
        if not bool(finite_on_cpu.item()):
            raise ReferenceError("non-finite source or BF16 result in MPS conversion")
        del finite_on_cpu
        source_values.clear()
        finite_flags.clear()
        finite_reduction = None

        output_fd, output_temp = _new_temp(output)
        output_temp_owned = True
        output_digest = hashlib.sha256()
        tensor_records: dict[str, dict[str, Any]] = {}
        with os.fdopen(output_fd, "wb") as stream:
            stream.write(output_header)
            output_digest.update(output_header)
            for name in sorted(expected_shapes):
                post_drain_mps_work_pending = True
                raw = _to_cpu_bytes(converted[name], torch, name, expected_shapes[name])
                post_drain_mps_work_pending = False
                expected_nbytes = output_layout[name]["nbytes"]
                if len(raw) != expected_nbytes:
                    raise ReferenceError(f"BF16 tensor byte extent mismatch: {name}")
                stream.write(raw)
                output_digest.update(raw)
                tensor_records[name] = {
                    **output_layout[name],
                    "source_dtype": "F32",
                    "sha256": hashlib.sha256(raw).hexdigest(),
                }
                del raw
            stream.flush()
            os.fsync(stream.fileno())
        converted.clear()

        expected_output_bytes = len(output_header) + sum(
            record["nbytes"] for record in output_layout.values()
        )
        if output_temp.stat().st_size != expected_output_bytes:
            raise ReferenceError("BF16 safetensors output size mismatch")
        output_sha256 = _sha256_file(output_temp)
        if output_sha256 != output_digest.hexdigest():
            raise ReferenceError("BF16 streamed and saved output digests differ")

        source_details = {
            "block_index": block_index,
            "path": str(inventory.path),
            "file_bytes": inventory.file_bytes,
            "file_sha256": inventory.file_sha256,
            "manifest_path": str(inventory.manifest_path),
            "manifest_sha256": inventory.manifest_sha256,
            "source": inventory.manifest["source"],
            "tensor_prefix": f"double_blocks.{block_index}.",
            "tensor_count": len(inventory.tensors),
        }
        report_document = {
            "schema": SCHEMA,
            "schema_version": 1,
            "producer": {
                "path": str(Path(__file__).resolve()),
                "source_sha256": _sha256_file(Path(__file__).resolve()),
                "argv": list(sys.argv),
                "python": sys.version.split()[0],
                "torch": torch.__version__,
                "device": "mps",
                "operation": "each source F32 tensor copied to MPS then cast to BF16",
                "block_index": block_index,
                "converted_tensor_count": len(tensor_records),
                "explicit_terminal_waits": 1,
                "model_invocations": 0,
                "weight_tensor_host_readbacks": len(tensor_records),
                "source_mutation_checked_after_admission": True,
            },
            "source": source_details,
            "cache": {
                "name": output.name,
                "nbytes": output_temp.stat().st_size,
                "sha256": output_sha256,
                "format": "safetensors",
                "dtype": "BF16",
                "block_index": block_index,
                "tensor_count": len(tensor_records),
                "payload_nbytes": sum(record["nbytes"] for record in output_layout.values()),
                "tensors": tensor_records,
            },
            "scope": {
                "kind": "tensorwise parameter conversion oracle",
                "source_storage": "original checkpoint subset stored as F32",
                "source_values": "F32 values loaded from the authenticated subset",
                "conversion_device": "MPS",
                "result_storage": "raw BF16 values serialized as safetensors",
                "neural_block_invocations": 0,
                "numerical_block_reference": False,
                "tolerance_comparison": False,
            },
            "limitations": [
                "This artifact verifies tensorwise MPS F32-to-BF16 conversion only; it does not execute a neural block.",
                "Source integrity is bound by caller-supplied whole-file and manifest SHA-256 values.",
                "The source checkpoint's declared whole-file LFS digest is not verified by this subset extraction.",
                "No CUDA, Magi, full-model residency, generation, speed, or physical-peak-memory claim is made.",
            ],
        }
        report_bytes = (json.dumps(report_document, indent=2, sort_keys=True) + "\n").encode()
        report_fd, report_temp = _new_temp(report)
        report_temp_owned = True
        with os.fdopen(report_fd, "wb") as stream:
            stream.write(report_bytes)
            stream.flush()
            os.fsync(stream.fileno())

        output_identity, report_identity = publish_output_pair(
            output_temp, output, report_temp, report
        )
        output_temp_owned = False
        report_temp_owned = False
        return {
            "status": "created",
            "block_index": block_index,
            "output": str(output),
            "output_sha256": output_sha256,
            "report": str(report),
            "tensor_count": len(tensor_records),
            "explicit_terminal_waits": 1,
        }
    except BaseException as exc:
        if conversion_started and (not terminal_wait_completed or post_drain_mps_work_pending):
            try:
                torch.mps.synchronize()
            except Exception as drain_exc:
                _UNCONFIRMED_MPS_REFERENCES.append((
                    source_values,
                    converted,
                    finite_flags,
                    finite_reduction,
                    locals().get("source_tensor"),
                    locals().get("source_value"),
                    locals().get("value"),
                    locals().get("finite_on_cpu"),
                ))
                completion_uncertain = True
                raise ReferenceError(
                    "MPS conversion failed and completion could not be confirmed: "
                    f"{type(drain_exc).__name__}: {drain_exc}"
                ) from None
        if isinstance(exc, ReferenceError):
            raise
        raise ReferenceError(f"block{block_index} conversion failed: {type(exc).__name__}: {exc}") from None
    finally:
        if not completion_uncertain:
            source_values.clear()
            converted.clear()
            finite_flags.clear()
            finite_reduction = None
        if output_temp_owned and output_temp is not None:
            output_temp.unlink(missing_ok=True)
        if report_temp_owned and report_temp is not None:
            report_temp.unlink(missing_ok=True)
        if report_identity is None:
            _remove_if_same(output, output_identity)
        elif output_identity is None:
            _remove_if_same(report, report_identity)


def _fixture_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--block-index", type=_block_index_argument, default=2,
                        help="actual DiT block index 2 or 3 (default: 2)")
    parser.add_argument("--weights", "--block-weights", dest="weights", type=Path,
                        default=None, help="authenticated extracted original-F32 block subset")
    parser.add_argument("--weights-sha256", "--block-weights-sha256",
                        dest="weights_sha256", required=True,
                        help="caller-pinned whole safetensors file SHA-256")
    parser.add_argument("--manifest", "--block-manifest", dest="manifest", type=Path,
                        default=None, help="authenticated source subset manifest")
    parser.add_argument("--manifest-sha256", "--block-manifest-sha256",
                        dest="manifest_sha256", required=True,
                        help="caller-pinned whole manifest SHA-256")
    parser.add_argument("--output", type=Path, default=None,
                        help="new raw BF16 safetensors output")
    parser.add_argument("--report", type=Path, default=None,
                        help="new conversion report output")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _fixture_parser().parse_args(argv)
    try:
        result = produce_cache(args)
    except (OSError, ValueError, ReferenceError, ImportError, RuntimeError) as exc:
        message = str(exc) if isinstance(exc, ReferenceError) else type(exc).__name__
        print(f"BF16 conversion failed: {message}", file=sys.stderr)
        return 2
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
