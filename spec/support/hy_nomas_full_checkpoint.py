#!/usr/bin/env python3
"""Download the complete pinned Hy-Nomas DiT safetensors checkpoint.

The downloader handles only bytes. It authenticates the immutable HF
revision, the complete safetensors header, and the final LFS SHA-256. Network
reads are range bounded and streamed into a resumable partial file under the
designated external SSD root. It imports no tensor framework.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
from dataclasses import dataclass
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import stat
import struct
import sys
import tempfile
import time
from typing import Any, Callable, Iterator
from urllib.parse import quote
from urllib.request import Request, build_opener

import hy_nomas_checkpoint_subset as subset


SCHEMA = "hy-nomas-full-checkpoint-download-v1"
ALLOWED_OUTPUT_ROOT = Path("/opt/mldata_ssd/cogni-ml/hy-nomas-q4-dit-20261025")
DEFAULT_OUTPUT = ALLOWED_OUTPUT_ROOT / "weights" / "diffusion_pytorch_model_ema_0_995.safetensors"
SOURCE_HEADER_SHA256 = "cbfa83dc81af315b425fd783e0ce42d8ff09df772952ef2a8aff48493e21b200"
SOURCE_TENSOR_COUNT = 2268
DOUBLE_BLOCK_COUNT = 54
MAX_CHUNK_BYTES = 64 * 1024 * 1024
MIN_CHUNK_BYTES = 64 * 1024
MAX_TRANSFER_MULTIPLIER = 2
MAX_DEADLINE_SECONDS = 24 * 60 * 60
STATE_LIMIT = 16 * 1024
READ_CHUNK_BYTES = 1024 * 1024


class DownloadError(RuntimeError):
    """A safe-to-report pinned-source, resource, or transfer rejection."""


@dataclass(frozen=True)
class CheckpointSpec:
    repository: str
    revision: str
    path: str
    size_bytes: int
    sha256: str
    header_bytes: int
    header_sha256: str


PINNED = CheckpointSpec(
    repository=subset.HF_REPO,
    revision=subset.HF_REVISION,
    path=subset.CHECKPOINT_PATH,
    size_bytes=subset.SOURCE_FILE_SIZE,
    sha256=subset.SOURCE_LFS_SHA256,
    header_bytes=subset.EXPECTED_HEADER_BYTES,
    header_sha256=SOURCE_HEADER_SHA256,
)


def _source_record(spec: CheckpointSpec) -> dict[str, Any]:
    return {
        "repository": spec.repository,
        "revision": spec.revision,
        "path": spec.path,
        "size_bytes": spec.size_bytes,
        "sha256": spec.sha256,
        "header_bytes": spec.header_bytes,
        "header_sha256": spec.header_sha256,
    }


def _json_object(raw: bytes, *, label: str) -> dict[str, Any]:
    def reject_duplicates(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise DownloadError(f"duplicate key in {label}")
            result[key] = value
        return result

    try:
        value = json.loads(raw.decode("utf-8"), object_pairs_hook=reject_duplicates)
    except DownloadError:
        raise
    except (UnicodeDecodeError, json.JSONDecodeError, RecursionError):
        raise DownloadError(f"invalid {label} JSON") from None
    if not isinstance(value, dict):
        raise DownloadError(f"{label} must be a JSON object")
    return value


def validate_checkpoint_header(raw: bytes, source_size: int = PINNED.size_bytes,
                               spec: CheckpointSpec = PINNED) -> None:
    """Authenticate and structurally validate all 2268 tensor extents.

    The existing subset parser checks every safetensors descriptor for legal
    dtype, shape/byte extent, in-bounds ranges, and contiguous full-payload
    coverage. This layer additionally binds the exact source header and all
    54 actual double-block inventories to their known block-0 shapes.
    """
    if not isinstance(raw, bytes) or len(raw) != spec.header_bytes:
        raise DownloadError("safetensors header length differs from the pinned source")
    if hashlib.sha256(raw).hexdigest() != spec.header_sha256:
        raise DownloadError("safetensors header SHA-256 differs from the pinned source")
    if spec != PINNED:
        # The private test seam may exercise transfer behavior with a synthetic
        # CheckpointSpec. Production CLI always supplies PINNED.
        raise DownloadError("full tensor inventory validation is only defined for the pinned source")
    try:
        subset.parse_safetensors_header(raw, source_size)
    except subset.SubsetError as exc:
        raise DownloadError(f"pinned safetensors header validation failed: {exc}") from None
    header = subset._json_object(raw, context="pinned safetensors header")
    header.pop("__metadata__", None)
    if len(header) != SOURCE_TENSOR_COUNT:
        raise DownloadError("full checkpoint tensor count is not 2268")
    for block_index in range(DOUBLE_BLOCK_COUNT):
        prefix = f"double_blocks.{block_index}."
        expected = {
            name.replace("double_blocks.0.", prefix, 1): shape
            for name, shape in subset.EXPECTED_TENSORS.items()
        }
        names = {name for name in header if name.startswith(prefix)}
        if names != set(expected):
            raise DownloadError(f"double block {block_index} tensor inventory mismatch")
        for name, shape in expected.items():
            descriptor = header[name]
            if (not isinstance(descriptor, dict)
                    or descriptor.get("dtype") != "F32"
                    or descriptor.get("shape") != list(shape)):
                raise DownloadError(f"double block {block_index} descriptor mismatch: {name}")


def validate_budgets(*, spec: CheckpointSpec, deadline_seconds: int,
                     max_transfer_bytes: int, disk_reserve_bytes: int,
                     chunk_bytes: int) -> None:
    if (type(deadline_seconds) is not int or deadline_seconds <= 0
            or deadline_seconds > MAX_DEADLINE_SECONDS):
        raise DownloadError("deadline must be between 1 second and 24 hours")
    minimum_transfer = spec.size_bytes + 8 + spec.header_bytes
    if (type(max_transfer_bytes) is not int or max_transfer_bytes < minimum_transfer
            or max_transfer_bytes > MAX_TRANSFER_MULTIPLIER * spec.size_bytes):
        raise DownloadError("transfer budget must cover the full source plus header and be at most 2x source size")
    if type(disk_reserve_bytes) is not int or disk_reserve_bytes <= 0:
        raise DownloadError("disk reserve must be a positive explicit byte count")
    if (type(chunk_bytes) is not int or chunk_bytes < MIN_CHUNK_BYTES
            or chunk_bytes > MAX_CHUNK_BYTES):
        raise DownloadError("range chunk size must be between 64 KiB and 64 MiB")


def _output_path(path: str | os.PathLike[str], *, root: Path = ALLOWED_OUTPUT_ROOT) -> Path:
    root = Path(root).expanduser().resolve(strict=True)
    candidate = Path(path).expanduser()
    if not candidate.is_absolute():
        candidate = Path.cwd() / candidate
    if candidate.parent.is_symlink():
        raise DownloadError("output parent symlinks are not admitted")
    parent = candidate.parent.resolve(strict=True)
    output = parent / candidate.name
    if (not output.is_relative_to(root) or output == root
            or output.suffix != ".safetensors" or candidate.name in ("", ".", "..")):
        raise DownloadError("output must be a .safetensors file under the designated SSD root")
    if not parent.is_dir():
        raise DownloadError("output parent must be an existing directory")
    return output


def initialize_output_root(root: Path = ALLOWED_OUTPUT_ROOT) -> None:
    """Create only this task's external SSD directories, rejecting symlinks."""
    root = Path(root)
    if root.exists() and root.is_symlink():
        raise DownloadError("external output root symlinks are not admitted")
    root.mkdir(parents=True, exist_ok=True)
    for name in ("weights", "logs", "tmp", "cache"):
        path = root / name
        if path.exists() and path.is_symlink():
            raise DownloadError(f"external output directory {name} is a symlink")
        path.mkdir(exist_ok=True)
        if not path.is_dir():
            raise DownloadError(f"external output path {name} is not a directory")


def _identity(path: Path) -> tuple[int, int, int]:
    try:
        info = path.lstat()
    except FileNotFoundError:
        raise DownloadError("partial download file is missing") from None
    if not stat.S_ISREG(info.st_mode):
        raise DownloadError("partial download path must be a regular non-symlink file")
    return info.st_dev, info.st_ino, info.st_size


def _sha256_prefix(path: Path, nbytes: int, *, deadline: float,
                   clock: Callable[[], float]) -> Any:
    digest = hashlib.sha256()
    remaining = nbytes
    with path.open("rb") as stream:
        while remaining:
            if clock() >= deadline:
                raise DownloadError("deadline expired while validating partial download")
            chunk = stream.read(min(READ_CHUNK_BYTES, remaining))
            if not chunk:
                raise DownloadError("partial download is shorter than committed state")
            digest.update(chunk)
            remaining -= len(chunk)
    return digest


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _write_state(path: Path, state: dict[str, Any]) -> None:
    if path.is_symlink():
        raise DownloadError("partial state path symlink is not admitted")
    encoded = (json.dumps(state, sort_keys=True, separators=(",", ":")) + "\n").encode()
    if len(encoded) > STATE_LIMIT:
        raise DownloadError("partial state exceeds its fixed size bound")
    fd, temporary_name = tempfile.mkstemp(prefix=".hy-nomas-state-", suffix=".tmp",
                                           dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        _fsync_directory(path.parent)
    finally:
        temporary.unlink(missing_ok=True)


def _read_header_prefix(path: Path, spec: CheckpointSpec) -> bytes:
    expected = 8 + spec.header_bytes
    with path.open("rb") as stream:
        raw = stream.read(expected)
    if len(raw) != expected:
        raise DownloadError("partial file does not contain the complete pinned header")
    if struct.unpack("<Q", raw[:8])[0] != spec.header_bytes:
        raise DownloadError("partial safetensors header length mismatch")
    return raw


def _load_state(state_path: Path, partial_path: Path, *, spec: CheckpointSpec,
                max_transfer_bytes: int, deadline: float,
                clock: Callable[[], float],
                header_validator: Callable[[bytes, int, CheckpointSpec], None]) -> tuple[dict[str, Any], Any]:
    if state_path.is_symlink() or partial_path.is_symlink():
        raise DownloadError("partial state and data must not be symlinks")
    try:
        raw = state_path.read_bytes()
    except OSError:
        raise DownloadError("partial state cannot be read") from None
    if len(raw) > STATE_LIMIT:
        raise DownloadError("partial state exceeds its fixed size bound")
    state = _json_object(raw, label="partial state")
    expected_keys = {
        "schema", "source", "committed_bytes", "prefix_sha256",
        "transfer_reserved_bytes", "partial_device", "partial_inode",
    }
    if set(state) != expected_keys or state.get("schema") != SCHEMA:
        raise DownloadError("partial state schema or fields mismatch")
    if state.get("source") != _source_record(spec):
        raise DownloadError("partial state belongs to a different pinned source")
    committed = state.get("committed_bytes")
    reserved = state.get("transfer_reserved_bytes")
    device, inode = state.get("partial_device"), state.get("partial_inode")
    prefix_bytes = 8 + spec.header_bytes
    if (type(committed) is not int or committed < 0
            or 0 < committed < prefix_bytes
            or committed > spec.size_bytes or type(reserved) is not int
            or reserved < committed or reserved > max_transfer_bytes
            or type(device) is not int or device < 0
            or type(inode) is not int or inode <= 0):
        raise DownloadError("partial state byte counts or file identity are invalid")
    partial_device, partial_inode, partial_size = _identity(partial_path)
    if (partial_device != device or partial_inode != inode
            or partial_size < committed or partial_size > spec.size_bytes):
        raise DownloadError("partial file identity or committed extent mismatch")
    hasher = _sha256_prefix(partial_path, committed, deadline=deadline, clock=clock)
    if hasher.hexdigest() != state.get("prefix_sha256"):
        raise DownloadError("partial prefix digest differs from its durable state")
    if committed >= prefix_bytes:
        prefix = _read_header_prefix(partial_path, spec)
        header_validator(prefix[8:], spec.size_bytes, spec)
    if partial_size != committed:
        with partial_path.open("r+b") as stream:
            stream.truncate(committed)
            stream.flush()
            os.fsync(stream.fileno())
    return state, hasher


@contextmanager
def _exclusive_lock(lock_path: Path) -> Iterator[None]:
    flags = os.O_CREAT | os.O_RDWR
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(lock_path, flags, 0o600)
    except OSError:
        raise DownloadError("could not open downloader lock") from None
    try:
        if not stat.S_ISREG(os.fstat(descriptor).st_mode):
            raise DownloadError("downloader lock must be a regular file")
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise DownloadError("another downloader owns this output") from None
        yield
    finally:
        os.close(descriptor)


def _checkpoint_url(spec: CheckpointSpec) -> str:
    return (f"https://huggingface.co/{spec.repository}/resolve/{spec.revision}/"
            f"{quote(spec.path, safe='/')}")


def _fetch_api_metadata(spec: CheckpointSpec, deadline: float,
                        clock: Callable[[], float]) -> None:
    if clock() >= deadline:
        raise DownloadError("download deadline expired before source metadata check")
    if spec != PINNED:
        raise DownloadError("network API admission is restricted to the pinned production source")
    try:
        url = subset._api_metadata_url()
        if not subset._trusted_hf_https_url(url):
            raise DownloadError("metadata URL is outside the trusted HF HTTPS domain")
        remaining = deadline - clock()
        if remaining <= 0:
            raise DownloadError("download deadline expired before source metadata request")
        response = build_opener(subset._TrustedRedirectHandler()).open(
            Request(url, headers={
                "Accept-Encoding": "identity",
                "User-Agent": "hy-nomas-full-checkpoint/1",
            }, method="GET"),
            timeout=max(0.001, min(45.0, remaining)),
        )
        if not subset._trusted_hf_https_url(response.geturl()):
            response.close()
            raise DownloadError("metadata request redirected outside the trusted HF HTTPS domain")
        with response:
            if clock() >= deadline:
                raise DownloadError("download deadline expired during source metadata check")
            raw = subset._read_bounded(response, subset.MAX_API_METADATA_BYTES)
        if clock() >= deadline:
            raise DownloadError("download deadline expired during source metadata check")
        subset.validate_api_metadata(raw)
    except DownloadError:
        raise
    except subset.SubsetError as exc:
        raise DownloadError(f"pinned HF source metadata rejected: {exc}") from None
    except Exception as exc:
        raise DownloadError(f"pinned HF metadata request failed ({type(exc).__name__})") from None


def _open_range(url: str, start: int, end: int, deadline: float,
                clock: Callable[[], float]):
    if clock() >= deadline:
        raise DownloadError("download deadline expired before HTTP range request")
    if not subset._trusted_hf_https_url(url):
        raise DownloadError("range URL is outside the trusted HF HTTPS domain")
    remaining = deadline - clock()
    if remaining <= 0:
        raise DownloadError("download deadline expired before HTTP range request")
    headers = {
        "Accept-Encoding": "identity",
        "User-Agent": "hy-nomas-full-checkpoint/1",
        "Range": f"bytes={start}-{end}",
    }
    request = Request(url, headers=headers, method="GET")
    try:
        response = build_opener(subset._TrustedRedirectHandler()).open(
            request, timeout=max(0.001, min(45.0, remaining))
        )
    except Exception as exc:
        raise DownloadError(f"HTTPS range request failed ({type(exc).__name__})") from None
    if not subset._trusted_hf_https_url(response.geturl()):
        response.close()
        raise DownloadError("range request redirected outside the trusted HF HTTPS domain")
    return response


def _read_header(url: str, *, spec: CheckpointSpec, deadline: float,
                 clock: Callable[[], float], opener: Callable[..., Any],
                 header_validator: Callable[[bytes, int, CheckpointSpec], None]) -> bytes:
    stop = 7 + spec.header_bytes
    try:
        with opener(url, 0, stop, deadline, clock) as response:
            subset._validate_range_response(response, 0, stop, spec.size_bytes)
            raw = bytearray()
            while len(raw) < stop + 1:
                if clock() >= deadline:
                    raise DownloadError("deadline expired while reading checkpoint header")
                piece = response.read(min(64 * 1024, stop + 1 - len(raw)))
                if not piece:
                    break
                raw.extend(piece)
    except subset.SubsetError as exc:
        raise DownloadError(f"checkpoint header range rejected: {exc}") from None
    if len(raw) != stop + 1:
        raise DownloadError("checkpoint header range is truncated")
    prefix = bytes(raw[:8])
    if struct.unpack("<Q", prefix)[0] != spec.header_bytes:
        raise DownloadError("checkpoint safetensors header length mismatch")
    header = bytes(raw[8:])
    header_validator(header, spec.size_bytes, spec)
    return bytes(raw)


def _disk_guard(parent: Path, remaining_bytes: int, reserve_bytes: int,
                disk_usage: Callable[[str | os.PathLike[str]], Any]) -> None:
    free = disk_usage(parent).free
    if free < remaining_bytes + reserve_bytes:
        raise DownloadError("external SSD free space is below remaining download plus explicit reserve")


def _initial_state(spec: CheckpointSpec, path: Path,
                   transfer_reserved: int = 0) -> tuple[dict[str, Any], Any]:
    fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_RDWR, 0o600)
    try:
        with os.fdopen(fd, "w+b") as stream:
            stream.flush()
            os.fsync(stream.fileno())
        info = path.lstat()
        state = {
            "schema": SCHEMA,
            "source": _source_record(spec),
            "committed_bytes": 0,
            "prefix_sha256": hashlib.sha256(b"").hexdigest(),
            "transfer_reserved_bytes": transfer_reserved,
            "partial_device": info.st_dev,
            "partial_inode": info.st_ino,
        }
        return state, hashlib.sha256()
    except BaseException:
        # This path was created exclusively by this call.
        path.unlink(missing_ok=True)
        raise


def _save_state(path: Path, state: dict[str, Any], hasher: Any,
                committed_bytes: int, transfer_reserved: int) -> dict[str, Any]:
    updated = dict(state)
    updated["committed_bytes"] = committed_bytes
    updated["prefix_sha256"] = hasher.hexdigest()
    updated["transfer_reserved_bytes"] = transfer_reserved
    _write_state(path, updated)
    return updated


def _append_range(partial: Path, state_path: Path, state: dict[str, Any],
                  hasher: Any, *, spec: CheckpointSpec, chunk_bytes: int,
                  max_transfer_bytes: int, deadline: float,
                  clock: Callable[[], float], opener: Callable[..., Any],
                  disk_guard: Callable[[int], None]) -> tuple[dict[str, Any], Any, int]:
    committed = state["committed_bytes"]
    stop = min(spec.size_bytes, committed + chunk_bytes) - 1
    request_bytes = stop - committed + 1
    reserved = state["transfer_reserved_bytes"] + request_bytes
    if reserved > max_transfer_bytes:
        raise DownloadError("range would exceed the explicit transfer budget")
    # Reserve a whole range before dispatch. Abrupt process death cannot cause
    # a retry to exceed the caller's total-transfer budget.
    state = _save_state(state_path, state, hasher, committed, reserved)
    disk_guard(spec.size_bytes - committed)
    previous_hasher = hasher.copy()
    try:
        with partial.open("r+b", buffering=0) as sink:
            sink.seek(committed)
            with opener(_checkpoint_url(spec), committed, stop, deadline, clock) as response:
                try:
                    subset._validate_range_response(response, committed, stop, spec.size_bytes)
                except subset.SubsetError as exc:
                    raise DownloadError(f"checkpoint range rejected: {exc}") from None
                remaining = request_bytes
                while remaining:
                    if clock() >= deadline:
                        raise DownloadError("download deadline expired during a range")
                    piece = response.read(min(READ_CHUNK_BYTES, remaining))
                    if not isinstance(piece, bytes) or not piece:
                        raise DownloadError("checkpoint range response was truncated")
                    if len(piece) > remaining:
                        raise DownloadError("checkpoint range response exceeded its requested extent")
                    if sink.write(piece) != len(piece):
                        raise DownloadError("short local write while saving checkpoint range")
                    hasher.update(piece)
                    remaining -= len(piece)
                if response.read(1) != b"":
                    raise DownloadError("checkpoint range response contained trailing bytes")
            sink.flush()
            os.fsync(sink.fileno())
        committed += request_bytes
        state = _save_state(state_path, state, hasher, committed, reserved)
        return state, hasher, committed
    except BaseException:
        with partial.open("r+b", buffering=0) as sink:
            sink.truncate(state["committed_bytes"])
            sink.flush()
            os.fsync(sink.fileno())
        hasher = previous_hasher
        # Keep the reservation durable even if a response failed midway.
        state = _save_state(state_path, state, hasher, state["committed_bytes"], reserved)
        raise


def _publish(partial: Path, state_path: Path, target: Path,
             expected_sha256: str, expected_size: int, *, deadline: float,
             clock: Callable[[], float]) -> None:
    digest = hashlib.sha256()
    total = 0
    with partial.open("rb") as stream:
        while True:
            if clock() >= deadline:
                raise DownloadError("deadline expired while hashing completed checkpoint")
            chunk = stream.read(READ_CHUNK_BYTES)
            if not chunk:
                break
            digest.update(chunk)
            total += len(chunk)
    if total != expected_size:
        raise DownloadError("completed checkpoint size differs from pinned source")
    if digest.hexdigest() != expected_sha256:
        raise DownloadError("completed checkpoint SHA-256 differs from pinned LFS digest; partial retained")
    subset.atomic_publish_no_replace(partial, target)
    try:
        _fsync_directory(target.parent)
    except OSError:
        # atomic_publish_no_replace has already linked and fsynced the target.
        pass
    state_path.unlink(missing_ok=True)
    _fsync_directory(target.parent)


def _download_checkpoint(
    output: str | os.PathLike[str], *, deadline_seconds: int,
    max_transfer_bytes: int, disk_reserve_bytes: int,
    chunk_bytes: int = MAX_CHUNK_BYTES,
    root: Path = ALLOWED_OUTPUT_ROOT,
    spec: CheckpointSpec = PINNED,
    metadata_check: Callable[[CheckpointSpec, float, Callable[[], float]], None] | None = None,
    opener: Callable[..., Any] | None = None,
    header_validator: Callable[[bytes, int, CheckpointSpec], None] = validate_checkpoint_header,
    disk_usage: Callable[[str | os.PathLike[str]], Any] = shutil.disk_usage,
    clock: Callable[[], float] = time.monotonic,
    progress: Callable[[int, int], None] | None = None,
) -> dict[str, Any]:
    """Private core; injected source/transport hooks exist only for CPU tests.

    Production callers use download_checkpoint(), which fixes every
    provenance, network, and destination authority to the pinned source.
    """
    validate_budgets(
        spec=spec, deadline_seconds=deadline_seconds,
        max_transfer_bytes=max_transfer_bytes,
        disk_reserve_bytes=disk_reserve_bytes, chunk_bytes=chunk_bytes,
    )
    target = _output_path(output, root=root)
    if target.exists() or target.is_symlink():
        raise DownloadError("final checkpoint already exists; refusing replacement")
    partial = target.with_name(target.name + ".partial")
    state_path = target.with_name(target.name + ".partial.json")
    lock_path = target.with_name(target.name + ".lock")
    deadline = clock() + deadline_seconds
    open_range = opener or _open_range
    check_metadata = metadata_check or _fetch_api_metadata

    with _exclusive_lock(lock_path):
        if target.exists() or target.is_symlink():
            raise DownloadError("final checkpoint appeared while acquiring lock")
        has_partial = os.path.lexists(partial)
        has_state = os.path.lexists(state_path)
        if has_partial != has_state:
            raise DownloadError("partial data and state must both exist or both be absent")
        check_metadata(spec, deadline, clock)

        if has_partial:
            state, hasher = _load_state(
                state_path, partial, spec=spec,
                max_transfer_bytes=max_transfer_bytes,
                deadline=deadline, clock=clock,
                header_validator=header_validator,
            )
            committed = state["committed_bytes"]
            reserved = state["transfer_reserved_bytes"]
        else:
            _disk_guard(target.parent, spec.size_bytes,
                        disk_reserve_bytes, disk_usage)
            state, hasher = _initial_state(spec, partial)
            try:
                _write_state(state_path, state)
            except BaseException:
                if not os.path.lexists(state_path):
                    try:
                        info = partial.lstat()
                        if (info.st_dev == state["partial_device"]
                                and info.st_ino == state["partial_inode"]):
                            partial.unlink()
                    except FileNotFoundError:
                        pass
                raise
            committed = state["committed_bytes"]
            reserved = state["transfer_reserved_bytes"]

        if committed == 0:
            prefix_size = 8 + spec.header_bytes
            next_reserved = reserved + prefix_size
            if next_reserved > max_transfer_bytes:
                raise DownloadError("header range would exceed the explicit transfer budget")
            _disk_guard(target.parent, spec.size_bytes,
                        disk_reserve_bytes, disk_usage)
            state = _save_state(state_path, state, hasher, 0, next_reserved)
            reserved = next_reserved
            prefix = _read_header(
                _checkpoint_url(spec), spec=spec, deadline=deadline,
                clock=clock, opener=open_range, header_validator=header_validator,
            )
            with partial.open("r+b", buffering=0) as stream:
                stream.seek(0)
                if stream.write(prefix) != len(prefix):
                    raise DownloadError("short local write while saving checkpoint header")
                stream.flush()
                os.fsync(stream.fileno())
            hasher.update(prefix)
            committed = len(prefix)
            state = _save_state(state_path, state, hasher, committed, reserved)

        while committed < spec.size_bytes:
            if clock() >= deadline:
                raise DownloadError("download deadline expired before next range")
            _disk_guard(target.parent, spec.size_bytes - committed,
                        disk_reserve_bytes, disk_usage)
            previous = committed
            state, hasher, committed = _append_range(
                partial, state_path, state, hasher,
                spec=spec, chunk_bytes=chunk_bytes,
                max_transfer_bytes=max_transfer_bytes,
                deadline=deadline, clock=clock, opener=open_range,
                disk_guard=lambda remaining: _disk_guard(
                    target.parent, remaining, disk_reserve_bytes, disk_usage
                ),
            )
            reserved = state["transfer_reserved_bytes"]
            if progress is not None:
                progress(committed, spec.size_bytes)
            if committed <= previous:
                raise DownloadError("download made no forward progress")

        _publish(partial, state_path, target, spec.sha256, spec.size_bytes,
                 deadline=deadline, clock=clock)
        return {
            "schema": SCHEMA,
            "path": str(target),
            "size_bytes": spec.size_bytes,
            "sha256": spec.sha256,
            "header_sha256": spec.header_sha256,
            "network_bytes_reserved": reserved,
            "source_revision": spec.revision,
            "source_path": spec.path,
        }


def download_checkpoint(
    output: str | os.PathLike[str], *, deadline_seconds: int,
    max_transfer_bytes: int, disk_reserve_bytes: int,
    chunk_bytes: int = MAX_CHUNK_BYTES,
    progress: Callable[[int, int], None] | None = None,
) -> dict[str, Any]:
    """Download only the hard-pinned source to the designated SSD root."""
    return _download_checkpoint(
        output,
        deadline_seconds=deadline_seconds,
        max_transfer_bytes=max_transfer_bytes,
        disk_reserve_bytes=disk_reserve_bytes,
        chunk_bytes=chunk_bytes,
        root=ALLOWED_OUTPUT_ROOT,
        spec=PINNED,
        progress=progress,
    )


def _positive_int(text: str) -> int:
    try:
        value = int(text)
    except ValueError:
        raise argparse.ArgumentTypeError("expected a positive integer") from None
    if value <= 0:
        raise argparse.ArgumentTypeError("expected a positive integer")
    return value


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--deadline-seconds", type=_positive_int, required=True,
                        help="wall-clock limit for API checks, transfer, and final SHA verification")
    parser.add_argument("--max-transfer-bytes", type=_positive_int, required=True,
                        help="cumulative reserved HTTP range bytes across resumes")
    parser.add_argument("--disk-reserve-bytes", type=_positive_int, required=True,
                        help="free bytes to retain after accounting for the remaining download")
    parser.add_argument("--chunk-bytes", type=_positive_int, default=MAX_CHUNK_BYTES)
    args = parser.parse_args(argv)
    try:
        initialize_output_root(ALLOWED_OUTPUT_ROOT)

        last_report = [0]

        def report_progress(done: int, total: int) -> None:
            if done == total or done - last_report[0] >= 512 * 1024 * 1024:
                last_report[0] = done
                print(f"downloaded {done}/{total} bytes", file=sys.stderr, flush=True)

        result = download_checkpoint(
            args.output,
            deadline_seconds=args.deadline_seconds,
            max_transfer_bytes=args.max_transfer_bytes,
            disk_reserve_bytes=args.disk_reserve_bytes,
            chunk_bytes=args.chunk_bytes,
            progress=report_progress,
        )
    except (DownloadError, subset.SubsetError, OSError, ValueError) as exc:
        if isinstance(exc, (DownloadError, subset.SubsetError)):
            message = str(exc)
        elif isinstance(exc, FileExistsError):
            message = "final checkpoint already exists; refusing replacement"
        else:
            message = f"local operation failed ({type(exc).__name__})"
        print(f"download failed: {message}", file=sys.stderr)
        return 2
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
