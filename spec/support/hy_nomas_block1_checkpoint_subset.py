#!/usr/bin/env python3
"""Fetch only the authenticated second Hy-Nomas DiT block's stored F32 bytes.

This reuses the first subset's bounded HTTPS/range/publication machinery. The
complete pinned header is authenticated; the whole checkpoint is not downloaded
or hashed. No tensor decoding or conversion is performed by this extractor.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import struct
import sys
from urllib.parse import quote

import hy_nomas_checkpoint_subset as base

SubsetError = base.SubsetError
EXPECTED_HEADER_BYTES = base.EXPECTED_HEADER_BYTES
EXPECTED_HEADER_SHA256 = "cbfa83dc81af315b425fd783e0ce42d8ff09df772952ef2a8aff48493e21b200"
BLOCK_DATA_START = 1_572_244_992
BLOCK_DATA_END = 2_176_472_576
EXPECTED_BLOCK_BYTES = 604_227_584
EXPECTED_RANGE_START = base.SOURCE_DATA_OFFSET + BLOCK_DATA_START
EXPECTED_RANGE_END = base.SOURCE_DATA_OFFSET + BLOCK_DATA_END - 1
ALLOWED_OUTPUT_ROOT = Path("/opt/mldata_ssd/cogni-ml/hy-nomas-pretrained-chain-20261025")
EXPECTED_TENSORS = {name.replace("double_blocks.0.", "double_blocks.1.", 1): shape
                    for name, shape in base.EXPECTED_TENSORS.items()}


def record_subset_header(raw: bytes) -> base.HeaderPlan:
    """Check all second-block names, shapes, dtypes and their exact extent."""
    header = base._json_object(raw, context="second block header")
    selected = {name for name in header if name.startswith("double_blocks.1.")}
    if selected != set(EXPECTED_TENSORS):
        raise SubsetError("second block tensor inventory mismatch")
    records = []
    for name, expected_shape in EXPECTED_TENSORS.items():
        value = header[name]
        if not isinstance(value, dict) or set(value) != {"dtype", "shape", "data_offsets"}:
            raise SubsetError("invalid second block tensor descriptor")
        shape, offsets = value["shape"], value["data_offsets"]
        if (value["dtype"] != "F32" or not isinstance(shape, list)
                or any(type(x) is not int for x in shape) or shape != list(expected_shape)):
            raise SubsetError("second block dtype or shape mismatch")
        if (not isinstance(offsets, list) or len(offsets) != 2
                or any(type(x) is not int for x in offsets)):
            raise SubsetError("invalid second block offsets")
        count = 4
        for dimension in expected_shape:
            count *= dimension
        start, end = offsets
        if start < BLOCK_DATA_START or end > BLOCK_DATA_END or end - start != count:
            raise SubsetError("second block tensor extent mismatch")
        records.append(base.TensorRecord(name, "F32", expected_shape, start, end))
    records.sort(key=lambda item: item.data_start)
    cursor = BLOCK_DATA_START
    for record in records:
        if record.data_start != cursor:
            raise SubsetError("second block tensors must be contiguous and disjoint")
        cursor = record.data_end
    if cursor != BLOCK_DATA_END:
        raise SubsetError("second block payload extent mismatch")
    return base.HeaderPlan(tuple(records), EXPECTED_RANGE_START, EXPECTED_RANGE_END,
                           EXPECTED_BLOCK_BYTES)


def parse_safetensors_header(raw: bytes, source_size: int) -> base.HeaderPlan:
    if source_size != base.SOURCE_FILE_SIZE or len(raw) != EXPECTED_HEADER_BYTES:
        raise SubsetError("pinned source header size mismatch")
    if hashlib.sha256(raw).hexdigest() != EXPECTED_HEADER_SHA256:
        raise SubsetError("pinned source header SHA-256 mismatch")
    # Check the full source layout, including the old block-0 contract.
    base.parse_safetensors_header(raw, source_size)
    return record_subset_header(raw)


def make_subset_header(plan: base.HeaderPlan) -> bytes:
    if (plan.block_bytes != EXPECTED_BLOCK_BYTES
            or plan.range_start != EXPECTED_RANGE_START or plan.range_end != EXPECTED_RANGE_END
            or len(plan.tensors) != 32 or {x.name for x in plan.tensors} != set(EXPECTED_TENSORS)):
        raise SubsetError("unrecognized second block subset plan")
    value = {"__metadata__": {
        "format": "pt", "hy_nomas_hf_revision": base.HF_REVISION,
        "hy_nomas_source_file": base.CHECKPOINT_PATH,
        "hy_nomas_source_lfs_sha256_declared": base.SOURCE_LFS_SHA256,
        "hy_nomas_source_lfs_sha256_verified": "false",
        "hy_nomas_source_git_commit": base.SOURCE_GIT_COMMIT,
        "hy_nomas_block_index": "1",
        "hy_nomas_subset": "double_blocks.1 F32 stored bytes; no conversion",
    }}
    cursor = 0
    for record in sorted(plan.tensors, key=lambda item: item.data_start):
        shape = EXPECTED_TENSORS[record.name]
        size = 4
        for dimension in shape:
            size *= dimension
        if (record.dtype != "F32" or record.shape != shape
                or record.data_start != BLOCK_DATA_START + cursor
                or record.data_end != record.data_start + size):
            raise SubsetError("second block descriptor changed after admission")
        value[record.name] = {"dtype": "F32", "shape": list(shape),
                              "data_offsets": [cursor, cursor + size]}
        cursor += size
    if cursor != EXPECTED_BLOCK_BYTES:
        raise SubsetError("second block payload coverage changed")
    encoded = json.dumps(value, separators=(",", ":"), sort_keys=True).encode()
    encoded += b" " * ((-len(encoded)) % 8)
    return struct.pack("<Q", len(encoded)) + encoded


def output_path(path: str | os.PathLike[str]) -> Path:
    root = ALLOWED_OUTPUT_ROOT.resolve(strict=True)
    candidate = Path(path).expanduser().absolute()
    parent = candidate.parent.resolve(strict=True)
    output = parent / candidate.name
    if output.suffix != ".safetensors" or not output.is_relative_to(root):
        raise SubsetError("output must be a safetensors file inside the designated external SSD root")
    return output


def extract(path: str | os.PathLike[str], *, max_payload_bytes: int = EXPECTED_BLOCK_BYTES):
    if type(max_payload_bytes) is not int or max_payload_bytes != EXPECTED_BLOCK_BYTES:
        raise SubsetError("payload transfer budget must equal the exact second block size")
    output = output_path(path)
    manifest = output.with_suffix(output.suffix + ".manifest.json")
    if os.path.lexists(output) or os.path.lexists(manifest):
        raise FileExistsError("output or provenance manifest already exists")
    url = (f"https://huggingface.co/{base.HF_REPO}/resolve/{base.HF_REVISION}/"
           f"{quote(base.CHECKPOINT_PATH, safe='/')}")
    with base._request(base._api_metadata_url()) as response:
        base.validate_api_metadata(base._read_bounded(response, base.MAX_API_METADATA_BYTES))
    prefix, _ = base._read_range(url, 0, 7, base.SOURCE_FILE_SIZE)
    if len(prefix) != 8 or struct.unpack("<Q", prefix)[0] != EXPECTED_HEADER_BYTES:
        raise SubsetError("source prefix differs from pinned header length")
    raw, header_sha = base._read_range(url, 8, 8 + EXPECTED_HEADER_BYTES - 1,
                                      base.SOURCE_FILE_SIZE)
    plan = parse_safetensors_header(raw, base.SOURCE_FILE_SIZE)
    encoded = make_subset_header(plan)
    fd, temp = base._new_temp(output.parent, ".block1.partial")
    manifest_fd, manifest_temp, published_manifest = None, None, None
    try:
        with os.fdopen(fd, "wb") as stream:
            sink = base._HashingSink(stream)
            sink.write(encoded)
            with base._request(url, byte_range=(plan.range_start, plan.range_end)) as response:
                payload_sha = base.copy_verified_range_body(
                    response, plan.range_start, plan.range_end, base.SOURCE_FILE_SIZE, sink)
            stream.flush()
            os.fsync(stream.fileno())
            output_sha = sink.hexdigest()
        record = {
            "format": "hy-nomas-block-subset-provenance-v1",
            "source": {
                "repository": base.HF_REPO, "revision": base.HF_REVISION,
                "checkpoint_path": base.CHECKPOINT_PATH,
                "checkpoint_size_bytes": base.SOURCE_FILE_SIZE,
                "checkpoint_lfs_sha256": base.SOURCE_LFS_SHA256,
                "checkpoint_lfs_sha256_status": "declared by pinned HF API; whole file not downloaded or verified",
                "source_git_commit": base.SOURCE_GIT_COMMIT,
                "header_sha256_observed": header_sha,
                "range_start_inclusive": plan.range_start,
                "range_end_inclusive": plan.range_end,
                "authentication_precondition": "HTTPS/TLS to pinned HF revision and its HTTPS range redirects",
                "payload_sha256_observed": payload_sha,
            },
            "subset": {"block_index": 1, "tensor_prefix": "double_blocks.1.",
                       "tensor_count": 32, "tensor_names": sorted(EXPECTED_TENSORS),
                       "dtype": "F32", "stored_bytes_preserved": True,
                       "conversion_performed": False, "payload_bytes": EXPECTED_BLOCK_BYTES},
            "output": {"filename": output.name, "size_bytes": len(encoded) + EXPECTED_BLOCK_BYTES,
                       "sha256": output_sha},
        }
        manifest_fd, manifest_temp = base._new_temp(output.parent, ".manifest.partial")
        with os.fdopen(manifest_fd, "wb") as stream:
            manifest_fd = None
            stream.write((json.dumps(record, indent=2, sort_keys=True) + "\n").encode())
            stream.flush()
            os.fsync(stream.fileno())
        stat = manifest_temp.lstat()
        base.atomic_publish_no_replace(manifest_temp, manifest)
        published_manifest = (stat.st_dev, stat.st_ino)
        base.atomic_publish_no_replace(temp, output)
        published_manifest = None
        return output_sha, payload_sha
    except BaseException:
        if published_manifest is not None:
            base._remove_if_same(manifest, *published_manifest)
        raise
    finally:
        if manifest_fd is not None:
            os.close(manifest_fd)
        for temporary in (temp, manifest_temp):
            if temporary is not None:
                temporary.unlink(missing_ok=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output")
    parser.add_argument("--max-payload-bytes", type=int, default=EXPECTED_BLOCK_BYTES)
    args = parser.parse_args()
    try:
        output_sha, payload_sha = extract(args.output, max_payload_bytes=args.max_payload_bytes)
    except (SubsetError, OSError, ValueError) as error:
        message = str(error) if isinstance(error, SubsetError) else type(error).__name__
        print(f"extract failed: {message}", file=sys.stderr)
        return 2
    print(json.dumps({"status": "created", "block_index": 1, "tensor_count": 32,
                      "payload_bytes": EXPECTED_BLOCK_BYTES,
                      "output_sha256": output_sha, "payload_sha256": payload_sha}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
