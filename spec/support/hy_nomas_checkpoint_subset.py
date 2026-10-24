#!/usr/bin/env python3
"""Extract only the audited first Hy-Nomas double-stream block from HF.

This intentionally handles bytes and safetensors metadata only. It does not
decode or convert checkpoint tensors. The extractor accepts exactly one pinned
official file and one audited contiguous block extent; unfamiliar metadata
fails closed.
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
from typing import BinaryIO
from urllib.error import HTTPError, URLError
from urllib.parse import quote, urlsplit
from urllib.request import HTTPRedirectHandler, Request, build_opener


HF_REPO = "tencent/Hy-Nomas"
HF_REVISION = "bac719a80c8d02f91c485fca804bb8221e149b70"
CHECKPOINT_PATH = "DiT/distilled/diffusion_pytorch_model_ema_0_995.safetensors"
SOURCE_LFS_SHA256 = "73819bcf3b82b090116dd1a1748795de95eef703c304e7443963d00ea8aa4d70"
SOURCE_FILE_SIZE = 34_213_991_280
SOURCE_GIT_COMMIT = "911cea6458a89c2663513ff7099561ec190dd491"

EXPECTED_HEADER_BYTES = 252_904
SOURCE_DATA_OFFSET = 8 + EXPECTED_HEADER_BYTES
BLOCK_DATA_START = 968_017_408
BLOCK_DATA_END = 1_572_244_992
EXPECTED_RANGE_START = SOURCE_DATA_OFFSET + BLOCK_DATA_START
EXPECTED_RANGE_END = SOURCE_DATA_OFFSET + BLOCK_DATA_END - 1
EXPECTED_BLOCK_BYTES = BLOCK_DATA_END - BLOCK_DATA_START
MAX_HEADER_BYTES = 16 * 1024 * 1024
MAX_API_METADATA_BYTES = 2 * 1024 * 1024
MAX_SINGLE_RANGE_BYTES = EXPECTED_BLOCK_BYTES
ALLOWED_OUTPUT_ROOT = Path("/opt/mldata_ssd/cogni-ml/hy-nomas-bf16-20261024")


def _expected_tensors() -> dict[str, tuple[int, ...]]:
    expected: dict[str, tuple[int, ...]] = {}
    for stream in ("img", "txt"):
        root = f"double_blocks.0.{stream}"
        expected[f"{root}_mod.linear.weight"] = (12_288, 2_048)
        expected[f"{root}_mod.linear.bias"] = (12_288,)
        for projection in ("q", "k", "v", "proj"):
            expected[f"{root}_attn_{projection}.weight"] = (2_048, 2_048)
            expected[f"{root}_attn_{projection}.bias"] = (2_048,)
        for norm in ("q_norm", "k_norm"):
            expected[f"{root}_attn_{norm}.weight"] = (128,)
        # The native arena calls these mlp_fc1/mlp_fc2; official checkpoint
        # keys preserve the module boundary as `mlp.fc1`/`mlp.fc2`.
        expected[f"{root}_mlp.fc1.weight"] = (8_192, 2_048)
        expected[f"{root}_mlp.fc1.bias"] = (8_192,)
        expected[f"{root}_mlp.fc2.weight"] = (2_048, 8_192)
        expected[f"{root}_mlp.fc2.bias"] = (2_048,)
    return expected


EXPECTED_TENSORS = _expected_tensors()


class SubsetError(Exception):
    """A safe-to-report rejection of untrusted or out-of-scope input."""


class TensorRecord:
    __slots__ = ("name", "dtype", "shape", "data_start", "data_end")

    def __init__(self, name: str, dtype: str, shape: tuple[int, ...],
                 data_start: int, data_end: int) -> None:
        self.name = name
        self.dtype = dtype
        self.shape = shape
        self.data_start = data_start
        self.data_end = data_end


class HeaderPlan:
    __slots__ = ("tensors", "range_start", "range_end", "block_bytes")

    def __init__(self, tensors: tuple[TensorRecord, ...], range_start: int,
                 range_end: int, block_bytes: int) -> None:
        self.tensors = tensors
        self.range_start = range_start
        self.range_end = range_end
        self.block_bytes = block_bytes


_DTYPE_BYTES = {
    "BOOL": 1,
    "U8": 1,
    "I8": 1,
    "F8_E4M3": 1,
    "F8_E4M3FN": 1,
    "F8_E5M2": 1,
    "F8_E8M0": 1,
    "U16": 2,
    "I16": 2,
    "F16": 2,
    "BF16": 2,
    "U32": 4,
    "I32": 4,
    "F32": 4,
    "U64": 8,
    "I64": 8,
    "F64": 8,
}


def _object_without_duplicate_keys(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise SubsetError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _json_object(raw: bytes, *, context: str):
    try:
        value = json.loads(raw.decode("utf-8"), object_pairs_hook=_object_without_duplicate_keys)
    except SubsetError:
        raise
    except (UnicodeDecodeError, json.JSONDecodeError, RecursionError) as exc:
        raise SubsetError(f"invalid {context} JSON") from None
    if not isinstance(value, dict):
        raise SubsetError(f"{context} JSON must be an object")
    return value


def _is_int(value) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _shape_nbytes(shape: tuple[int, ...], dtype: str) -> int:
    try:
        itemsize = _DTYPE_BYTES[dtype]
    except KeyError:
        raise SubsetError(f"unsupported safetensors dtype: {dtype}") from None
    count = 1
    for dimension in shape:
        if not _is_int(dimension) or dimension < 0:
            raise SubsetError("invalid tensor shape dimension")
        count *= dimension
        if count > SOURCE_FILE_SIZE:
            raise SubsetError("tensor shape exceeds source-size bound")
    return count * itemsize


def parse_safetensors_header(raw: bytes, source_size: int) -> HeaderPlan:
    """Validate the full source layout and return the audited block's range."""
    if not isinstance(raw, bytes) or len(raw) > MAX_HEADER_BYTES:
        raise SubsetError("header exceeds bounded header size")
    if not _is_int(source_size) or source_size != SOURCE_FILE_SIZE:
        raise SubsetError("source size does not match pinned file size")

    header = _json_object(raw, context="safetensors header")
    if len(raw) != EXPECTED_HEADER_BYTES:
        raise SubsetError("header length does not match audited source")
    metadata_present = "__metadata__" in header
    metadata = header.pop("__metadata__", None)
    if metadata_present:
        if not isinstance(metadata, dict) or any(
            not isinstance(key, str) or not isinstance(value, str)
            for key, value in metadata.items()
        ):
            raise SubsetError("invalid safetensors metadata")

    data_size = source_size - SOURCE_DATA_OFFSET
    records = []
    for name, descriptor in header.items():
        if not isinstance(name, str) or not isinstance(descriptor, dict):
            raise SubsetError("invalid safetensors tensor descriptor")
        if set(descriptor) != {"dtype", "shape", "data_offsets"}:
            raise SubsetError(f"unrecognized fields in descriptor for {name}")
        dtype = descriptor.get("dtype")
        shape_value = descriptor.get("shape")
        offsets = descriptor.get("data_offsets")
        if not isinstance(dtype, str) or not isinstance(shape_value, list):
            raise SubsetError(f"invalid descriptor for {name}")
        if not isinstance(offsets, list) or len(offsets) != 2:
            raise SubsetError(f"invalid offsets for {name}")
        if any(not _is_int(value) for value in offsets):
            raise SubsetError(f"invalid offsets for {name}")
        start, end = offsets
        if start < 0 or end < start or end > data_size:
            raise SubsetError(f"out-of-bounds offsets for {name}")
        if any(not _is_int(dimension) or dimension < 0 for dimension in shape_value):
            raise SubsetError(f"invalid shape for {name}")
        shape = tuple(shape_value)
        if end - start != _shape_nbytes(shape, dtype):
            raise SubsetError(f"shape byte count does not match offsets for {name}")
        records.append(TensorRecord(name, dtype, shape, start, end))

    records.sort(key=lambda item: (item.data_start, item.data_end, item.name))
    cursor = 0
    for record in records:
        if record.data_start != cursor:
            raise SubsetError("source tensor ranges are not contiguous and nonoverlapping")
        cursor = record.data_end
    if cursor != data_size:
        raise SubsetError("source tensor ranges do not cover the complete payload")

    target_names = {name for name in header if name.startswith("double_blocks.0.")}
    if target_names != set(EXPECTED_TENSORS):
        missing = len(set(EXPECTED_TENSORS) - target_names)
        extra = len(target_names - set(EXPECTED_TENSORS))
        raise SubsetError(f"block 0 tensor inventory mismatch (missing={missing}, extra={extra})")
    target_by_name = {record.name: record for record in records
                      if record.name in EXPECTED_TENSORS}
    for name, expected_shape in EXPECTED_TENSORS.items():
        record = target_by_name[name]
        if record.dtype != "F32":
            raise SubsetError(f"block tensor dtype mismatch for {name}")
        if record.shape != expected_shape:
            raise SubsetError(f"block tensor shape mismatch for {name}")

    ordered = tuple(sorted(target_by_name.values(),
                           key=lambda item: (item.data_start, item.data_end, item.name)))
    if not ordered or ordered[0].data_start != BLOCK_DATA_START:
        raise SubsetError("block 0 starts at an unexpected source offset")
    if ordered[-1].data_end != BLOCK_DATA_END:
        raise SubsetError("block 0 ends at an unexpected source offset")
    for left, right in zip(ordered, ordered[1:]):
        if left.data_end != right.data_start:
            raise SubsetError("block 0 ranges are not ordered and contiguous")

    block_bytes = ordered[-1].data_end - ordered[0].data_start
    absolute_start = SOURCE_DATA_OFFSET + ordered[0].data_start
    absolute_end = SOURCE_DATA_OFFSET + ordered[-1].data_end - 1
    if (block_bytes != EXPECTED_BLOCK_BYTES
            or absolute_start != EXPECTED_RANGE_START
            or absolute_end != EXPECTED_RANGE_END):
        raise SubsetError("block 0 extent does not match audited range")
    return HeaderPlan(ordered, absolute_start, absolute_end, block_bytes)


def make_subset_header(plan: HeaderPlan) -> bytes:
    """Build a deterministic safetensors header with block offsets rebased to 0."""
    if (plan.block_bytes != EXPECTED_BLOCK_BYTES
            or plan.range_start != EXPECTED_RANGE_START
            or plan.range_end != EXPECTED_RANGE_END
            or {item.name for item in plan.tensors} != set(EXPECTED_TENSORS)):
        raise SubsetError("refusing to serialize an unrecognized block plan")
    by_offset = sorted(plan.tensors,
                       key=lambda tensor: (tensor.data_start, tensor.data_end, tensor.name))
    cursor = 0
    for item in by_offset:
        if item.data_start != BLOCK_DATA_START + cursor:
            raise SubsetError("block tensors do not form the expected output payload")
        cursor += item.data_end - item.data_start
    if cursor != EXPECTED_BLOCK_BYTES:
        raise SubsetError("block tensors do not cover the expected output payload")

    value = {
        "__metadata__": {
            "format": "pt",
            "hy_nomas_hf_revision": HF_REVISION,
            "hy_nomas_source_file": CHECKPOINT_PATH,
            "hy_nomas_source_lfs_sha256_declared": SOURCE_LFS_SHA256,
            "hy_nomas_source_lfs_sha256_verified": "false",
            "hy_nomas_source_git_commit": SOURCE_GIT_COMMIT,
            "hy_nomas_subset": "double_blocks.0 F32 stored bytes; no conversion",
        }
    }
    for item in sorted(plan.tensors, key=lambda tensor: tensor.name):
        start = item.data_start - BLOCK_DATA_START
        end = item.data_end - BLOCK_DATA_START
        if start < 0 or end < start or item.dtype != "F32":
            raise SubsetError("invalid source extent for block tensor")
        value[item.name] = {
            "dtype": item.dtype,
            "shape": list(item.shape),
            "data_offsets": [start, end],
        }
    encoded = json.dumps(value, separators=(",", ":"), sort_keys=True).encode("utf-8")
    padding = (-len(encoded)) % 8
    encoded += b" " * padding
    if len(encoded) > MAX_HEADER_BYTES:
        raise SubsetError("rebased header exceeds bounded size")
    return struct.pack("<Q", len(encoded)) + encoded


def _header_values(headers, name: str):
    if hasattr(headers, "get_all"):
        values = headers.get_all(name)
        return [] if values is None else list(values)
    value = headers.get(name)
    return [] if value is None else [value]


def _one_header(headers, name: str, *, required: bool = True):
    values = _header_values(headers, name)
    if len(values) != (1 if required else min(len(values), 1)):
        raise SubsetError(f"missing or duplicate {name} response header")
    if not values:
        return None
    return str(values[0]).strip()


def _validate_range_response(response, start: int, end: int, total: int) -> int:
    status = getattr(response, "status", None)
    if status != 206:
        raise SubsetError("range request did not return HTTP 206")
    content_range = _one_header(response.headers, "Content-Range")
    match = re.fullmatch(r"bytes ([0-9]+)-([0-9]+)/([0-9]+)", content_range or "")
    if match is None or tuple(map(int, match.groups())) != (start, end, total):
        raise SubsetError("range response Content-Range mismatch")
    expected = end - start + 1
    content_length = _one_header(response.headers, "Content-Length")
    if content_length is None or not re.fullmatch(r"[0-9]+", content_length):
        raise SubsetError("invalid range response Content-Length")
    if int(content_length) != expected:
        raise SubsetError("range response Content-Length mismatch")
    encodings = _header_values(response.headers, "Content-Encoding")
    if len(encodings) > 1 or (encodings and str(encodings[0]).strip().lower() != "identity"):
        raise SubsetError("range response is not identity encoded")
    if start < 0 or end < start or total <= end or expected > MAX_SINGLE_RANGE_BYTES:
        raise SubsetError("requested range exceeds audited transfer bounds")
    return expected


def copy_verified_range_body(response, start: int, end: int, total: int,
                             sink: BinaryIO, *, chunk_size: int = 1024 * 1024) -> str:
    """Copy an exact bounded range and return its locally observed SHA-256."""
    expected = _validate_range_response(response, start, end, total)
    if not _is_int(chunk_size) or chunk_size <= 0 or chunk_size > 8 * 1024 * 1024:
        raise SubsetError("invalid bounded range chunk size")
    remaining = expected
    digest = hashlib.sha256()
    while remaining:
        chunk = response.read(min(chunk_size, remaining))
        if not isinstance(chunk, bytes) or not chunk:
            raise SubsetError("truncated range response body")
        if len(chunk) > remaining:
            raise SubsetError("range reader returned more than requested")
        written = sink.write(chunk)
        if written != len(chunk):
            raise SubsetError("short write while saving range")
        digest.update(chunk)
        remaining -= len(chunk)
    if response.read(1) != b"":
        raise SubsetError("range response body exceeded requested bytes")
    return digest.hexdigest()


def _read_bounded(response, limit: int) -> bytes:
    if getattr(response, "status", None) != 200:
        raise SubsetError("metadata request did not return HTTP 200")
    encodings = _header_values(response.headers, "Content-Encoding")
    if len(encodings) > 1 or (encodings and str(encodings[0]).strip().lower() != "identity"):
        raise SubsetError("metadata response is not identity encoded")
    content_lengths = _header_values(response.headers, "Content-Length")
    if len(content_lengths) > 1:
        raise SubsetError("duplicate metadata Content-Length")
    if content_lengths:
        value = str(content_lengths[0]).strip()
        if not re.fullmatch(r"[0-9]+", value) or int(value) > limit:
            raise SubsetError("metadata response exceeds bounded size")
    chunks = bytearray()
    while len(chunks) <= limit:
        chunk = response.read(min(64 * 1024, limit + 1 - len(chunks)))
        if not chunk:
            break
        chunks.extend(chunk)
    if len(chunks) > limit:
        raise SubsetError("metadata response exceeds bounded size")
    if content_lengths and len(chunks) != int(str(content_lengths[0]).strip()):
        raise SubsetError("truncated metadata response")
    return bytes(chunks)


def validate_api_metadata(raw: bytes) -> None:
    if not isinstance(raw, bytes) or len(raw) > MAX_API_METADATA_BYTES:
        raise SubsetError("API metadata exceeds bounded size")
    metadata = _json_object(raw, context="HF API metadata")
    if metadata.get("sha") != HF_REVISION:
        raise SubsetError("HF API returned an unexpected revision")
    siblings = metadata.get("siblings")
    if not isinstance(siblings, list):
        raise SubsetError("HF API metadata has no sibling inventory")
    matches = [item for item in siblings
               if isinstance(item, dict) and item.get("rfilename") == CHECKPOINT_PATH]
    if len(matches) != 1:
        raise SubsetError("pinned checkpoint is absent or duplicated in HF inventory")
    checkpoint = matches[0]
    if checkpoint.get("size") != SOURCE_FILE_SIZE:
        raise SubsetError("HF checkpoint size mismatch")
    lfs = checkpoint.get("lfs")
    if not isinstance(lfs, dict) or lfs.get("sha256") != SOURCE_LFS_SHA256:
        raise SubsetError("HF checkpoint hash mismatch")


def validate_transfer_budget(max_payload_bytes: int) -> None:
    """Check the maximum selected tensor payload, excluding bounded metadata."""
    if not _is_int(max_payload_bytes) or max_payload_bytes < EXPECTED_BLOCK_BYTES:
        raise SubsetError("payload transfer budget is below audited block size")
    if max_payload_bytes > SOURCE_FILE_SIZE:
        raise SubsetError("payload transfer budget exceeds pinned source size")


def atomic_publish_no_replace(source: Path, target: Path) -> None:
    """Publish a completed same-directory file without replacing any target."""
    source = Path(source)
    target = Path(target)
    if source.parent.resolve() != target.parent.resolve():
        raise SubsetError("atomic publication requires one output directory")
    source_stat = source.lstat()
    if not stat.S_ISREG(source_stat.st_mode):
        raise SubsetError("publication source must be a regular file")
    os.link(source, target, follow_symlinks=False)
    try:
        _fsync_directory(target.parent)
    except OSError:
        # Preserve the completed temporary file and roll back our link if the
        # directory cannot confirm publication durability.
        _remove_if_same(target, source_stat.st_dev, source_stat.st_ino)
        raise
    try:
        source.unlink()
    except OSError:
        # The published target is already complete and durable; caller cleanup
        # can retry removal of the temporary name.
        return
    try:
        _fsync_directory(target.parent)
    except OSError:
        # The target was durably linked before temporary-name cleanup.
        pass


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _trusted_hf_https_url(url: str) -> bool:
    try:
        parsed = urlsplit(url)
        host = (parsed.hostname or "").lower().rstrip(".")
        port = parsed.port
    except ValueError:
        return False
    trusted_suffixes = ("huggingface.co", "hf.co")
    trusted_host = any(host == suffix or host.endswith("." + suffix)
                       for suffix in trusted_suffixes)
    return (parsed.scheme.lower() == "https" and trusted_host
            and port in (None, 443) and parsed.username is None
            and parsed.password is None)


class _TrustedRedirectHandler(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        if not _trusted_hf_https_url(newurl):
            raise SubsetError("HTTPS redirect refused outside the trusted HF domain")
        return super().redirect_request(req, fp, code, msg, headers, newurl)


def _request(url: str, *, byte_range: tuple[int, int] | None = None):
    if not _trusted_hf_https_url(url):
        raise SubsetError("request endpoint is outside the trusted HF HTTPS domain")
    headers = {
        "Accept-Encoding": "identity",
        "User-Agent": "hy-nomas-bounded-subset/1",
    }
    if byte_range is not None:
        headers["Range"] = f"bytes={byte_range[0]}-{byte_range[1]}"
    request = Request(url, headers=headers, method="GET")
    try:
        response = build_opener(_TrustedRedirectHandler()).open(request, timeout=45)
    except HTTPError as error:
        raise SubsetError(f"HTTPS request failed with status {error.code}") from None
    except URLError:
        raise SubsetError("HTTPS request failed before response") from None
    if not _trusted_hf_https_url(response.geturl()):
        response.close()
        raise SubsetError("HTTPS request redirected outside the trusted HF domain")
    return response


def _read_range(url: str, start: int, end: int, total: int) -> tuple[bytes, str]:
    from io import BytesIO

    if end - start + 1 > MAX_HEADER_BYTES + 8:
        raise SubsetError("in-memory range reads are restricted to safetensors headers")
    sink = BytesIO()
    with _request(url, byte_range=(start, end)) as response:
        digest = copy_verified_range_body(response, start, end, total, sink)
    return sink.getvalue(), digest


def _api_metadata_url() -> str:
    return (f"https://huggingface.co/api/models/{HF_REPO}/revision/"
            f"{HF_REVISION}?blobs=true")


def _fetch_plan(file_url: str) -> HeaderPlan:
    prefix, _ = _read_range(file_url, 0, 7, SOURCE_FILE_SIZE)
    if len(prefix) != 8:
        raise SubsetError("safetensors length prefix is truncated")
    header_length = struct.unpack("<Q", prefix)[0]
    if header_length != EXPECTED_HEADER_BYTES or header_length > MAX_HEADER_BYTES:
        raise SubsetError("safetensors header length differs from audited source")
    header, _ = _read_range(file_url, 8, 8 + header_length - 1, SOURCE_FILE_SIZE)
    return parse_safetensors_header(header, SOURCE_FILE_SIZE)


class _HashingSink:
    def __init__(self, stream: BinaryIO) -> None:
        self.stream = stream
        self.digest = hashlib.sha256()

    def write(self, data: bytes) -> int:
        written = self.stream.write(data)
        if written != len(data):
            raise SubsetError("short write while saving subset")
        self.digest.update(data)
        return written

    def hexdigest(self) -> str:
        return self.digest.hexdigest()


def _output_path(path: str | os.PathLike[str]) -> Path:
    root = ALLOWED_OUTPUT_ROOT.resolve(strict=True)
    output = Path(path).expanduser()
    if not output.is_absolute():
        output = Path.cwd() / output
    parent = output.parent.resolve(strict=True)
    canonical = parent / output.name
    if not canonical.is_relative_to(root) or canonical == root:
        raise SubsetError("output must be inside the designated external SSD directory")
    if output.suffix != ".safetensors":
        raise SubsetError("output filename must end in .safetensors")
    if not parent.is_dir():
        raise SubsetError("output parent must already exist")
    return canonical


def _new_temp(parent: Path, suffix: str) -> tuple[int, Path]:
    fd, name = tempfile.mkstemp(prefix=".hy-nomas-", suffix=suffix, dir=parent)
    return fd, Path(name)


def _remove_if_same(path: Path, device: int, inode: int) -> None:
    try:
        current = path.lstat()
        if current.st_dev == device and current.st_ino == inode:
            path.unlink()
            _fsync_directory(path.parent)
    except FileNotFoundError:
        pass


def extract(output_path: str | os.PathLike[str], *,
            max_payload_bytes: int = EXPECTED_BLOCK_BYTES) -> tuple[str, str]:
    """Fetch only block 0 and atomically publish its F32 subset and manifest."""
    validate_transfer_budget(max_payload_bytes)
    output = _output_path(output_path)
    manifest = output.with_suffix(output.suffix + ".manifest.json")
    if os.path.lexists(output) or os.path.lexists(manifest):
        raise FileExistsError("output or provenance manifest already exists")

    checkpoint_url = (
        f"https://huggingface.co/{HF_REPO}/resolve/{HF_REVISION}/"
        f"{quote(CHECKPOINT_PATH, safe='/')}"
    )
    with _request(_api_metadata_url()) as response:
        api_raw = _read_bounded(response, MAX_API_METADATA_BYTES)
    validate_api_metadata(api_raw)
    plan = _fetch_plan(checkpoint_url)
    if plan.block_bytes > max_payload_bytes:
        raise SubsetError("selected block exceeds payload transfer budget")
    output_header = make_subset_header(plan)

    out_fd, out_temp = _new_temp(output.parent, ".safetensors.partial")
    manifest_fd = None
    manifest_temp = None
    published_manifest = None
    try:
        with os.fdopen(out_fd, "wb") as stream:
            sink = _HashingSink(stream)
            sink.write(output_header)
            with _request(checkpoint_url,
                          byte_range=(plan.range_start, plan.range_end)) as response:
                payload_digest = copy_verified_range_body(
                    response, plan.range_start, plan.range_end, SOURCE_FILE_SIZE, sink
                )
            stream.flush()
            os.fsync(stream.fileno())
            output_digest = sink.hexdigest()

        record = {
            "format": "hy-nomas-block-subset-provenance-v1",
            "source": {
                "repository": HF_REPO,
                "revision": HF_REVISION,
                "checkpoint_path": CHECKPOINT_PATH,
                "checkpoint_size_bytes": SOURCE_FILE_SIZE,
                "checkpoint_lfs_sha256": SOURCE_LFS_SHA256,
                "checkpoint_lfs_sha256_status": "declared_by_pinned_HF_API; whole file not downloaded or verified",
                "source_git_commit": SOURCE_GIT_COMMIT,
                "range_start_inclusive": plan.range_start,
                "range_end_inclusive": plan.range_end,
                "authentication_precondition": "HTTPS/TLS to the pinned HF revision and its HTTPS range redirects",
                "payload_sha256_observed": payload_digest,
            },
            "subset": {
                "tensor_prefix": "double_blocks.0.",
                "tensor_count": len(plan.tensors),
                "tensor_names": sorted(item.name for item in plan.tensors),
                "dtype": "F32",
                "stored_bytes_preserved": True,
                "conversion_performed": False,
                "payload_bytes": plan.block_bytes,
            },
            "output": {
                "filename": output.name,
                "size_bytes": len(output_header) + plan.block_bytes,
                "sha256": output_digest,
            },
        }
        manifest_bytes = (json.dumps(record, sort_keys=True, indent=2) + "\n").encode("utf-8")
        manifest_fd, manifest_temp = _new_temp(output.parent, ".manifest.partial")
        with os.fdopen(manifest_fd, "wb") as stream:
            manifest_fd = None
            stream.write(manifest_bytes)
            stream.flush()
            os.fsync(stream.fileno())

        manifest_stat = manifest_temp.lstat()
        atomic_publish_no_replace(manifest_temp, manifest)
        published_manifest = (manifest_stat.st_dev, manifest_stat.st_ino)
        atomic_publish_no_replace(out_temp, output)
        published_manifest = None
        _fsync_directory(output.parent)
        return output_digest, payload_digest
    except BaseException:
        if published_manifest is not None:
            _remove_if_same(manifest, *published_manifest)
        raise
    finally:
        if manifest_fd is not None:
            os.close(manifest_fd)
        for temporary in (out_temp, manifest_temp):
            if temporary is not None:
                try:
                    temporary.unlink()
                except FileNotFoundError:
                    pass


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", help="new .safetensors path under the designated external SSD directory")
    parser.add_argument("--max-payload-bytes", type=int,
                        default=EXPECTED_BLOCK_BYTES,
                        help="maximum selected checkpoint payload bytes (default: audited block size)")
    args = parser.parse_args(argv)
    try:
        output_digest, payload_digest = extract(
            args.output, max_payload_bytes=args.max_payload_bytes
        )
    except (SubsetError, OSError, ValueError) as error:
        # Never display URL-bearing exception text from an HTTPS implementation.
        if isinstance(error, SubsetError):
            message = str(error)
        elif isinstance(error, FileExistsError):
            message = "output or provenance manifest already exists"
        else:
            message = f"local operation failed ({type(error).__name__})"
        print(f"extract failed: {message}", file=sys.stderr)
        return 2
    print(f"published F32 block subset; sha256={output_digest}; payload_sha256={payload_digest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
