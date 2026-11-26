"""Bounded, lossless activation trace writer for diagnostic MPS snapshots.

Importing this module does not import PyTorch.  Device snapshots stay on MPS
until ``flush``; each tensor is then exported separately to a raw little-endian
payload shard.  A trace remains incomplete until ``finish(expected_names)``.
"""

from __future__ import annotations

import copy
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import stat
import sys
from numbers import Integral
from typing import Iterable


SCHEMA = "cogni-ml.activation-trace"
SCHEMA_VERSION = 1
_DTYPE_BYTES = {"float16": 2, "bfloat16": 2, "float32": 4}
_NAME_PART = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")
_SHARD_NAME = re.compile(r"^batch-[0-9]{6,}\.bin$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_CHUNK_BYTES = 1024 * 1024


class ActivationTraceWriter:
    """Write scoped activation snapshots into a new directory without overwrite."""

    def __init__(
        self,
        directory: Path,
        max_total_bytes: int,
        max_batch_bytes: int,
        min_free_bytes: int,
        metadata: dict,
    ) -> None:
        self.directory = Path(directory)
        self._validate_limits(max_total_bytes, max_batch_bytes, min_free_bytes)
        if not isinstance(metadata, dict):
            raise ValueError("metadata must be a JSON-serializable object")
        try:
            self.metadata = copy.deepcopy(metadata)
            json.dumps(self.metadata, sort_keys=True, allow_nan=False)
        except (TypeError, ValueError) as exc:
            raise ValueError("metadata must be a JSON-serializable object") from exc

        self.max_total_bytes = max_total_bytes
        self.max_batch_bytes = max_batch_bytes
        self.min_free_bytes = min_free_bytes
        self._records: list[dict] = []
        self._shards: list[dict] = []
        self._pending: list[dict] = []
        self._names: set[str] = set()
        self._reserved_bytes = 0
        self._finished = False
        self._failed = False

        # A caller must choose a fresh leaf under its canonical data root.
        # mkdir is intentionally exclusive so prior evidence cannot be replaced.
        if ".." in self.directory.parts:
            raise ValueError("trace directory must not contain parent traversal")
        initial_document = self._document(complete=False, expected_names=None)
        initial_index_bytes = self._json_bytes(initial_document)
        initial_marker = b"activation trace not finalized\n"
        free = shutil.disk_usage(self.directory.parent).free
        if free - len(initial_marker) - len(initial_index_bytes) < self.min_free_bytes:
            raise OSError("insufficient free space reserve for trace initialization")
        self.directory.mkdir(mode=0o700, parents=False, exist_ok=False)
        self._write_exclusive(self.directory / ".incomplete", initial_marker)
        self._write_index_bytes(initial_index_bytes)

    @staticmethod
    def _validate_limits(max_total_bytes: int, max_batch_bytes: int, min_free_bytes: int) -> None:
        for label, value, allow_zero in (
            ("max_total_bytes", max_total_bytes, False),
            ("max_batch_bytes", max_batch_bytes, False),
            ("min_free_bytes", min_free_bytes, True),
        ):
            if isinstance(value, bool) or not isinstance(value, int):
                raise ValueError(f"{label} must be an integer")
            if value < (0 if allow_zero else 1):
                raise ValueError(f"{label} is out of range")

    @property
    def pending_bytes(self) -> int:
        return sum(item["nbytes"] for item in self._pending)

    @property
    def complete(self) -> bool:
        return self._finished

    def record(self, name: str, tensor) -> None:
        self._ensure_writable()
        self._validate_name(name)
        if name in self._names:
            raise ValueError(f"duplicate activation name: {name}")

        dtype = self._tensor_dtype(tensor)
        shape, nbytes = self._tensor_size(tensor, dtype)
        if self.pending_bytes + nbytes > self.max_batch_bytes:
            raise ValueError(
                f"activation batch quota exceeded: {self.pending_bytes + nbytes} > {self.max_batch_bytes}"
            )
        if self._reserved_bytes + nbytes > self.max_total_bytes:
            raise ValueError(
                f"activation total quota exceeded: {self._reserved_bytes + nbytes} > {self.max_total_bytes}"
            )

        original_device = self._mps_device(tensor)
        # This ordering snapshots the source immediately while keeping the
        # copy on the same device; CPU export is deferred to flush().
        snapshot = tensor.detach().contiguous().clone()
        if self._mps_device(snapshot) != original_device:
            raise ValueError("activation snapshot changed device")
        if self._tensor_dtype(snapshot) != dtype:
            raise ValueError("activation snapshot changed dtype")
        snapshot_shape, snapshot_nbytes = self._tensor_size(snapshot, dtype)
        if snapshot_shape != shape or snapshot_nbytes != nbytes:
            raise ValueError("activation snapshot changed shape or dtype size")

        self._pending.append({
            "name": name,
            "dtype": dtype,
            "shape": shape,
            "nbytes": nbytes,
            "tensor": snapshot,
        })
        self._names.add(name)
        self._reserved_bytes += nbytes

    def flush(self) -> bool:
        """Persist the pending GPU snapshots as one bounded batch shard."""
        self._ensure_writable()
        if not self._pending:
            return False

        shard_name = f"batch-{len(self._shards):06d}.bin"
        shard_path = self.directory / shard_name
        byte_offset = 0
        placeholder_records = list(self._records)
        for item in self._pending:
            placeholder_records.append({
                "sequence": len(placeholder_records),
                "name": item["name"],
                "shard": shard_name,
                "offset": byte_offset,
                "nbytes": item["nbytes"],
                "dtype": item["dtype"],
                "shape": item["shape"],
                "sha256": "0" * 64,
            })
            byte_offset += item["nbytes"]
        placeholder_shards = list(self._shards) + [{
            "sequence": len(self._shards),
            "name": shard_name,
            "nbytes": byte_offset,
            "sha256": "0" * 64,
            "record_names": [item["name"] for item in self._pending],
        }]
        predicted = self._document(
            complete=False,
            expected_names=None,
            records=placeholder_records,
            shards=placeholder_shards,
        )
        predicted_index_bytes = self._json_bytes(predicted)
        self._check_disk_reserve(byte_offset + len(predicted_index_bytes))

        records_for_shard: list[dict] = []
        shard_hash = hashlib.sha256()
        try:
            with shard_path.open("xb") as shard_file:
                offset = 0
                for item in self._pending:
                    raw = self._export_raw(item["tensor"], item["dtype"])
                    if len(raw) != item["nbytes"]:
                        raise ValueError(
                            f"exported byte count mismatch for {item['name']}: "
                            f"{len(raw)} != {item['nbytes']}"
                        )
                    record_hash = hashlib.sha256(raw).hexdigest()
                    shard_file.write(raw)
                    shard_hash.update(raw)
                    records_for_shard.append({
                        "sequence": len(self._records) + len(records_for_shard),
                        "name": item["name"],
                        "shard": shard_name,
                        "offset": offset,
                        "nbytes": item["nbytes"],
                        "dtype": item["dtype"],
                        "shape": item["shape"],
                        "sha256": record_hash,
                    })
                    offset += len(raw)
                    del raw
                shard_file.flush()
                os.fsync(shard_file.fileno())
            self._fsync_directory()

            actual_records = self._records + records_for_shard
            actual_shard = {
                "sequence": len(self._shards),
                "name": shard_name,
                "nbytes": byte_offset,
                "sha256": shard_hash.hexdigest(),
                "record_names": [item["name"] for item in self._pending],
            }
            new_shards = self._shards + [actual_shard]
            document = self._document(
                complete=False,
                expected_names=None,
                records=actual_records,
                shards=new_shards,
            )
            index_bytes = self._json_bytes(document)
            self._check_disk_reserve(len(index_bytes))
            self._write_index_bytes(index_bytes)
        except Exception:
            # A partially written shard/index is deliberately retained under
            # the incomplete marker for inspection, and this writer is unusable.
            self._failed = True
            self._pending.clear()
            raise

        self._records = actual_records
        self._shards = new_shards
        self._pending.clear()
        return True

    def finish(self, expected_names: Iterable[str]) -> None:
        """Flush and mark complete only when the caller's exact scope is present."""
        self._ensure_writable()
        if isinstance(expected_names, (str, bytes)):
            raise ValueError("expected_names must be an iterable of activation names")
        expected = list(expected_names)
        for name in expected:
            self._validate_name(name)
        if len(expected) != len(set(expected)):
            raise ValueError("expected scope contains duplicate activation names")
        if set(expected) != self._names:
            missing = sorted(set(expected) - self._names)
            extra = sorted(self._names - set(expected))
            raise ValueError(f"activation scope mismatch: missing={missing}, extra={extra}")

        self.flush()
        document = self._document(
            complete=True,
            expected_names=sorted(expected),
        )
        index_bytes = self._json_bytes(document)
        self._check_disk_reserve(len(index_bytes))
        self._write_index_bytes(index_bytes)
        try:
            (self.directory / ".incomplete").unlink()
            self._fsync_directory()
        except Exception:
            self._failed = True
            raise
        self._finished = True

    def close(self, expected_names: Iterable[str]) -> None:
        """Alias for finish; an expected scope is mandatory."""
        self.finish(expected_names)

    def _ensure_writable(self) -> None:
        if self._finished:
            raise RuntimeError("activation trace is already complete")
        if self._failed:
            raise RuntimeError("activation trace writer failed; trace remains incomplete")

    @staticmethod
    def _validate_name(name: str) -> None:
        if not isinstance(name, str) or not name or "\\" in name or name.startswith("/"):
            raise ValueError("activation name must be a safe relative name")
        parts = name.split("/")
        if any(part in ("", ".", "..") or not _NAME_PART.fullmatch(part) for part in parts):
            raise ValueError(f"activation name is not safe: {name!r}")

    @staticmethod
    def _tensor_dtype(tensor) -> str:
        device = getattr(tensor, "device", None)
        if getattr(device, "type", None) != "mps":
            raise ValueError("activation tensors must be on MPS")
        layout = getattr(tensor, "layout", None)
        if str(layout) not in ("torch.strided", "strided"):
            raise ValueError("activation tensors must use dense strided layout")
        if bool(getattr(tensor, "is_sparse", False)) or bool(getattr(tensor, "is_quantized", False)):
            raise ValueError("sparse and quantized activations are not supported")
        dtype = str(getattr(tensor, "dtype", "")).removeprefix("torch.")
        if dtype not in _DTYPE_BYTES:
            raise ValueError(f"unsupported activation dtype: {dtype or 'unknown'}")
        floating = getattr(tensor, "is_floating_point", None)
        if floating is not None:
            is_floating = floating() if callable(floating) else bool(floating)
            if not is_floating:
                raise ValueError("activation dtype must be floating point")
        return dtype

    @staticmethod
    def _mps_device(tensor):
        device = getattr(tensor, "device", None)
        if getattr(device, "type", None) != "mps":
            raise ValueError("activation tensors must remain on MPS")
        return device

    @staticmethod
    def _tensor_size(tensor, dtype: str) -> tuple[list[int], int]:
        try:
            shape = list(tensor.shape)
        except Exception as exc:
            raise ValueError("activation tensor has no valid shape") from exc
        if any(isinstance(dim, bool) or not isinstance(dim, Integral) or dim < 0 for dim in shape):
            raise ValueError("activation tensor has invalid dimensions")
        shape = [int(dim) for dim in shape]
        count = 1
        for dim in shape:
            count *= dim
        numel_method = getattr(tensor, "numel", None)
        if callable(numel_method):
            numel = numel_method()
            if isinstance(numel, bool) or int(numel) != count:
                raise ValueError("activation tensor shape does not match its element count")
        return shape, count * _DTYPE_BYTES[dtype]

    @staticmethod
    def _export_raw(tensor, dtype: str) -> bytes:
        # Test doubles can provide exact raw bytes without importing torch.
        # Real BF16 tensors need a dtype reinterpretation because NumPy does not
        # natively expose torch.bfloat16; this preserves every original bit.
        test_export = getattr(tensor, "_activation_trace_raw_bytes", None)
        if callable(test_export):
            raw = test_export(dtype)
            if not isinstance(raw, bytes):
                raise ValueError("raw tensor exporter must return bytes")
            return raw

        cpu_tensor = tensor
        if dtype == "bfloat16":
            import torch  # lazy: only needed when a real BF16 payload is flushed

            cpu_tensor = tensor.view(torch.uint16).cpu()
        else:
            cpu_tensor = tensor.cpu()
        array = cpu_tensor.numpy()
        raw = array.tobytes(order="C")
        byteorder = getattr(getattr(array, "dtype", None), "byteorder", "<")
        if byteorder == ">" or (byteorder == "=" and sys.byteorder == "big"):
            raw = array.byteswap().tobytes(order="C")
        return raw

    def _document(
        self,
        *,
        complete: bool,
        expected_names: list[str] | None,
        records: list[dict] | None = None,
        shards: list[dict] | None = None,
    ) -> dict:
        use_records = self._records if records is None else records
        use_shards = self._shards if shards is None else shards
        return {
            "schema": SCHEMA,
            "version": SCHEMA_VERSION,
            "complete": complete,
            "metadata": self.metadata,
            "limits": {
                "max_total_bytes": self.max_total_bytes,
                "max_batch_bytes": self.max_batch_bytes,
                "min_free_bytes": self.min_free_bytes,
            },
            "expected_names": expected_names,
            "total_bytes": sum(record["nbytes"] for record in use_records),
            "records": use_records,
            "shards": use_shards,
        }

    @staticmethod
    def _json_bytes(document: dict) -> bytes:
        return (json.dumps(document, sort_keys=True, indent=2, allow_nan=False) + "\n").encode("utf-8")

    def _check_disk_reserve(self, write_bytes: int) -> None:
        free = shutil.disk_usage(self.directory).free
        if free - write_bytes < self.min_free_bytes:
            raise OSError(
                f"insufficient free space reserve: need {write_bytes} bytes and "
                f"retain {self.min_free_bytes}, have {free}"
            )

    @staticmethod
    def _write_exclusive(path: Path, payload: bytes) -> None:
        with path.open("xb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())

    def _write_index_bytes(self, payload: bytes) -> None:
        temporary = self.directory / ".index.json.tmp"
        if temporary.exists() or temporary.is_symlink():
            raise FileExistsError(f"index temporary file already exists: {temporary}")
        index_path = self.directory / "index.json"
        if index_path.is_symlink() or (index_path.exists() and not index_path.is_file()):
            raise OSError("trace index is not a regular file")
        self._write_exclusive(temporary, payload)
        os.replace(temporary, index_path)
        self._fsync_directory()

    def _fsync_directory(self) -> None:
        flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
        descriptor = os.open(self.directory, flags)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)


def validate_trace(path: Path, *, require_complete: bool = True) -> dict:
    """Validate trace structure, scope, sizes and hashes with bounded reads.

    The returned object is the parsed aggregate index.  By default an
    incomplete trace is rejected; callers inspecting a recoverable diagnostic
    can request ``require_complete=False`` and still receive full byte checks.
    """
    root = Path(path)
    if ".." in root.parts:
        raise ValueError("trace path must not contain parent traversal")
    try:
        root_mode = root.lstat().st_mode
    except OSError as exc:
        raise ValueError(f"trace directory is unavailable: {root}") from exc
    if not stat.S_ISDIR(root_mode) or stat.S_ISLNK(root_mode):
        raise ValueError("trace path must be a real directory, not a symlink")

    try:
        entries = list(root.iterdir())
    except OSError as exc:
        raise ValueError("cannot enumerate trace directory") from exc
    for entry in entries:
        mode = entry.lstat().st_mode
        if stat.S_ISLNK(mode):
            raise ValueError(f"trace contains a symlink: {entry.name}")
        if not stat.S_ISREG(mode):
            raise ValueError(f"trace contains an unexpected non-file entry: {entry.name}")

    marker = root / ".incomplete"
    index_path = root / "index.json"
    if not index_path.is_file():
        raise ValueError("trace index is missing")
    if marker.is_symlink():
        raise ValueError("incomplete marker must not be a symlink")
    if marker.exists() and not marker.is_file():
        raise ValueError("incomplete marker is not a regular file")
    if marker.exists() and require_complete:
        raise ValueError("activation trace is incomplete")

    try:
        with _open_regular_nofollow(index_path, text=True) as handle:
            index = json.load(handle, object_pairs_hook=_unique_object)
    except (OSError, UnicodeError, json.JSONDecodeError, ValueError) as exc:
        raise ValueError(f"trace index is invalid: {exc}") from exc
    if not isinstance(index, dict):
        raise ValueError("trace index must be an object")
    if index.get("schema") != SCHEMA or index.get("version") != SCHEMA_VERSION:
        raise ValueError("trace schema/version is unsupported")
    if not isinstance(index.get("complete"), bool):
        raise ValueError("trace completion flag is invalid")
    if index["complete"] != (not marker.exists()):
        raise ValueError("trace completion flag disagrees with incomplete marker")
    if not isinstance(index.get("metadata"), dict):
        raise ValueError("trace metadata is invalid")
    limits = index.get("limits")
    if not isinstance(limits, dict) or any(
        isinstance(limits.get(key), bool) or not isinstance(limits.get(key), int) or limits[key] < 0
        for key in ("max_total_bytes", "max_batch_bytes", "min_free_bytes")
    ):
        raise ValueError("trace byte limits are invalid")
    if limits["max_total_bytes"] <= 0 or limits["max_batch_bytes"] <= 0:
        raise ValueError("trace byte limits are invalid")

    records = index.get("records")
    shards = index.get("shards")
    if not isinstance(records, list) or not isinstance(shards, list):
        raise ValueError("trace records or shards are invalid")
    if isinstance(index.get("total_bytes"), bool) or not isinstance(index.get("total_bytes"), int):
        raise ValueError("trace total byte count is invalid")

    names: set[str] = set()
    records_by_shard: dict[str, list[dict]] = {}
    calculated_total = 0
    for sequence, record in enumerate(records):
        if not isinstance(record, dict):
            raise ValueError("trace record is invalid")
        if (
            isinstance(record.get("sequence"), bool)
            or not isinstance(record.get("sequence"), int)
            or record["sequence"] != sequence
        ):
            raise ValueError("trace record sequence is not contiguous")
        name = record.get("name")
        ActivationTraceWriter._validate_name(name)
        if name in names:
            raise ValueError(f"trace contains duplicate record name: {name}")
        names.add(name)
        shard_name = record.get("shard")
        if not isinstance(shard_name, str) or not _SHARD_NAME.fullmatch(shard_name):
            raise ValueError(f"trace record has unsafe shard path: {shard_name!r}")
        dtype = record.get("dtype")
        if dtype not in _DTYPE_BYTES:
            raise ValueError(f"trace record has unsupported dtype: {dtype!r}")
        shape = record.get("shape")
        if not isinstance(shape, list) or any(
            isinstance(dim, bool) or not isinstance(dim, int) or dim < 0 for dim in shape
        ):
            raise ValueError(f"trace record has invalid shape: {name}")
        count = 1
        for dim in shape:
            count *= dim
        expected_nbytes = count * _DTYPE_BYTES[dtype]
        for key in ("offset", "nbytes"):
            if isinstance(record.get(key), bool) or not isinstance(record.get(key), int) or record[key] < 0:
                raise ValueError(f"trace record has invalid {key}: {name}")
        if record["nbytes"] != expected_nbytes:
            raise ValueError(f"trace record byte count disagrees with dtype/shape: {name}")
        if not isinstance(record.get("sha256"), str) or not _SHA256.fullmatch(record["sha256"]):
            raise ValueError(f"trace record hash is invalid: {name}")
        records_by_shard.setdefault(shard_name, []).append(record)
        calculated_total += expected_nbytes

    if calculated_total != index["total_bytes"]:
        raise ValueError("trace aggregate byte count is inconsistent")
    if calculated_total > limits["max_total_bytes"]:
        raise ValueError("trace exceeds declared total byte limit")

    shard_names: set[str] = set()
    for expected_sequence, shard in enumerate(shards):
        if not isinstance(shard, dict):
            raise ValueError("trace shard entry is invalid")
        if (
            isinstance(shard.get("sequence"), bool)
            or not isinstance(shard.get("sequence"), int)
            or shard["sequence"] != expected_sequence
        ):
            raise ValueError("trace shard sequence field is not contiguous")
        shard_name = shard.get("name")
        if not isinstance(shard_name, str) or not _SHARD_NAME.fullmatch(shard_name):
            raise ValueError(f"trace has unsafe shard path: {shard_name!r}")
        if shard_name in shard_names:
            raise ValueError(f"trace has duplicate shard: {shard_name}")
        shard_names.add(shard_name)
        if shard_name != f"batch-{expected_sequence:06d}.bin":
            raise ValueError("trace shard sequence is not contiguous")
        shard_records = records_by_shard.get(shard_name, [])
        declared_record_names = shard.get("record_names")
        if declared_record_names != [record["name"] for record in shard_records]:
            raise ValueError(f"trace shard record list is inconsistent: {shard_name}")
        offset = 0
        for record in shard_records:
            if record["offset"] != offset:
                raise ValueError(f"trace record offsets are not contiguous: {record['name']}")
            offset += record["nbytes"]
        if isinstance(shard.get("nbytes"), bool) or shard.get("nbytes") != offset:
            raise ValueError(f"trace shard byte count is inconsistent: {shard_name}")
        if offset > limits["max_batch_bytes"]:
            raise ValueError(f"trace shard exceeds declared batch byte limit: {shard_name}")
        if not isinstance(shard.get("sha256"), str) or not _SHA256.fullmatch(shard["sha256"]):
            raise ValueError(f"trace shard hash is invalid: {shard_name}")

    if set(records_by_shard) != shard_names:
        raise ValueError("trace records and shards do not match")
    actual_shards = {entry.name for entry in entries if entry.name.endswith(".bin")}
    if actual_shards != shard_names:
        raise ValueError("trace contains missing or unindexed shard files")
    expected_entries = {"index.json", *((".incomplete",) if marker.exists() else ()), *shard_names}
    actual_entries = {entry.name for entry in entries}
    if actual_entries != expected_entries:
        raise ValueError("trace contains unexpected or missing files")

    for shard in shards:
        shard_name = shard["name"]
        shard_path = root / shard_name
        if not shard_path.is_file() or shard_path.is_symlink():
            raise ValueError(f"trace shard is missing or unsafe: {shard_name}")
        shard_hash = hashlib.sha256()
        with _open_regular_nofollow(shard_path) as handle:
            if os.fstat(handle.fileno()).st_size != shard["nbytes"]:
                raise ValueError(f"trace shard size/truncation mismatch: {shard_name}")
            for record in records_by_shard[shard_name]:
                remaining = record["nbytes"]
                record_hash = hashlib.sha256()
                while remaining:
                    chunk = handle.read(min(_CHUNK_BYTES, remaining))
                    if not chunk:
                        raise ValueError(f"trace shard is truncated: {shard_name}")
                    remaining -= len(chunk)
                    record_hash.update(chunk)
                    shard_hash.update(chunk)
                if record_hash.hexdigest() != record["sha256"]:
                    raise ValueError(f"trace record hash mismatch: {record['name']}")
            if handle.read(1):
                raise ValueError(f"trace shard has trailing bytes: {shard_name}")
        if shard_hash.hexdigest() != shard["sha256"]:
            raise ValueError(f"trace shard hash mismatch: {shard_name}")

    expected_names = index.get("expected_names")
    if index["complete"]:
        if not isinstance(expected_names, list):
            raise ValueError("complete trace has invalid expected scope")
        for name in expected_names:
            ActivationTraceWriter._validate_name(name)
        if len(expected_names) != len(set(expected_names)):
            raise ValueError("complete trace has duplicate expected names")
        if set(expected_names) != names:
            raise ValueError("complete trace does not match expected scope")
    elif expected_names is not None:
        raise ValueError("incomplete trace must not declare a completed scope")
    return index


def _unique_object(pairs: list[tuple[str, object]]) -> dict:
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _open_regular_nofollow(path: Path, *, text: bool = False):
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(path, flags)
    try:
        if not stat.S_ISREG(os.fstat(descriptor).st_mode):
            raise ValueError(f"not a regular trace file: {path.name}")
        if text:
            return os.fdopen(descriptor, "r", encoding="utf-8")
        return os.fdopen(descriptor, "rb")
    except Exception:
        os.close(descriptor)
        raise
