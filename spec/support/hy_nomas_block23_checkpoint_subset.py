#!/usr/bin/env python3
"""Fetch only authenticated original F32 checkpoint bytes for DiT block 2 or 3.

The full safetensors header is pinned and checked against the repository's
complete source-layout validator. Only the selected block's 32 contiguous F32
tensors are transferred; their stored bytes are preserved without conversion.
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
EXPECTED_BLOCK_BYTES = 604_227_584

# Relative payload extents copied from the individually named descriptors in
# the authenticated full source header. They are intentionally not inferred
# from neighboring block indices.
BLOCK_DATA_RANGES = {
    2: (8_218_748_416, 8_822_976_000),
    3: (14_865_251_840, 15_469_479_424),
}
BLOCK_HTTP_RANGES = {
    index: (base.SOURCE_DATA_OFFSET + start, base.SOURCE_DATA_OFFSET + end - 1)
    for index, (start, end) in BLOCK_DATA_RANGES.items()
}
ALLOWED_OUTPUT_ROOT = Path("/opt/mldata_ssd/cogni-ml/hy-nomas-four-block-20261025")


def checked_block_index(value: object) -> int:
    if type(value) is not int or value not in BLOCK_DATA_RANGES:
        raise SubsetError("block index must identify actual block 2 or 3")
    return value


def _block_index_argument(text: str) -> int:
    try:
        value = int(text)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("block index must be 2 or 3") from exc
    if value not in BLOCK_DATA_RANGES:
        raise argparse.ArgumentTypeError("block index must be 2 or 3")
    return value


def expected_tensors(block_index: int) -> dict[str, tuple[int, ...]]:
    block_index = checked_block_index(block_index)
    return {
        name.replace("double_blocks.0.", f"double_blocks.{block_index}.", 1): shape
        for name, shape in base.EXPECTED_TENSORS.items()
    }


def validate_transfer_budget(max_payload_bytes: int) -> None:
    if type(max_payload_bytes) is not int or max_payload_bytes != EXPECTED_BLOCK_BYTES:
        raise SubsetError("payload transfer budget must equal the exact selected block size")


def record_subset_header(raw: bytes, block_index: int) -> base.HeaderPlan:
    """Select a block only after checking its exact named inventory and extent."""
    block_index = checked_block_index(block_index)
    expected = expected_tensors(block_index)
    header = base._json_object(raw, context=f"block {block_index} header")
    prefix = f"double_blocks.{block_index}."
    selected = {name for name in header if name.startswith(prefix)}
    if selected != set(expected):
        raise SubsetError(f"block {block_index} tensor inventory mismatch")

    records = []
    for name, expected_shape in expected.items():
        value = header[name]
        if not isinstance(value, dict) or set(value) != {"dtype", "shape", "data_offsets"}:
            raise SubsetError(f"invalid block {block_index} tensor descriptor: {name}")
        shape, offsets = value["shape"], value["data_offsets"]
        if (value["dtype"] != "F32" or not isinstance(shape, list)
                or any(type(dimension) is not int for dimension in shape)
                or shape != list(expected_shape)):
            raise SubsetError(f"block {block_index} tensor dtype or shape mismatch: {name}")
        if (not isinstance(offsets, list) or len(offsets) != 2
                or any(type(offset) is not int for offset in offsets)):
            raise SubsetError(f"invalid block {block_index} tensor offsets: {name}")
        size = 4
        for dimension in expected_shape:
            size *= dimension
        start, end = offsets
        if end - start != size:
            raise SubsetError(f"block {block_index} tensor extent mismatch: {name}")
        records.append(base.TensorRecord(name, "F32", expected_shape, start, end))

    data_start, data_end = BLOCK_DATA_RANGES[block_index]
    records.sort(key=lambda record: (record.data_start, record.data_end, record.name))
    cursor = data_start
    for record in records:
        if record.data_start != cursor or record.data_end <= record.data_start:
            raise SubsetError(f"block {block_index} tensors must be disjoint and contiguous")
        cursor = record.data_end
    if cursor != data_end or cursor - data_start != EXPECTED_BLOCK_BYTES:
        raise SubsetError(f"block {block_index} source extent mismatch")

    range_start, range_end = BLOCK_HTTP_RANGES[block_index]
    return base.HeaderPlan(tuple(records), range_start, range_end, EXPECTED_BLOCK_BYTES)


def parse_safetensors_header(raw: bytes, source_size: int, block_index: int) -> base.HeaderPlan:
    block_index = checked_block_index(block_index)
    if source_size != base.SOURCE_FILE_SIZE or len(raw) != EXPECTED_HEADER_BYTES:
        raise SubsetError("pinned source header size mismatch")
    if hashlib.sha256(raw).hexdigest() != EXPECTED_HEADER_SHA256:
        raise SubsetError("pinned source header SHA-256 mismatch")
    # This validates the complete source inventory and all global extents,
    # including the independently audited block-0 contract.
    base.parse_safetensors_header(raw, source_size)
    return record_subset_header(raw, block_index)


def _block_index_from_plan(plan: base.HeaderPlan) -> int:
    for block_index in BLOCK_DATA_RANGES:
        if ({record.name for record in plan.tensors} == set(expected_tensors(block_index))
                and (plan.range_start, plan.range_end) == BLOCK_HTTP_RANGES[block_index]
                and plan.block_bytes == EXPECTED_BLOCK_BYTES):
            return block_index
    raise SubsetError("unrecognized block 2/3 subset plan")


def make_subset_header(plan: base.HeaderPlan) -> bytes:
    """Create a deterministic rebased safetensors header for one admitted plan."""
    block_index = _block_index_from_plan(plan)
    expected = expected_tensors(block_index)
    data_start, _ = BLOCK_DATA_RANGES[block_index]
    value = {"__metadata__": {
        "format": "pt",
        "hy_nomas_hf_revision": base.HF_REVISION,
        "hy_nomas_source_file": base.CHECKPOINT_PATH,
        "hy_nomas_source_lfs_sha256_declared": base.SOURCE_LFS_SHA256,
        "hy_nomas_source_lfs_sha256_verified": "false",
        "hy_nomas_source_git_commit": base.SOURCE_GIT_COMMIT,
        "hy_nomas_block_index": str(block_index),
        "hy_nomas_subset": f"double_blocks.{block_index} F32 stored bytes; no conversion",
    }}
    cursor = 0
    for record in sorted(plan.tensors, key=lambda item: (item.data_start, item.data_end, item.name)):
        shape = expected[record.name]
        size = 4
        for dimension in shape:
            size *= dimension
        if (record.dtype != "F32" or record.shape != shape
                or record.data_start != data_start + cursor
                or record.data_end != record.data_start + size):
            raise SubsetError(f"block {block_index} descriptor changed after admission")
        value[record.name] = {
            "dtype": "F32", "shape": list(shape), "data_offsets": [cursor, cursor + size]
        }
        cursor += size
    if cursor != EXPECTED_BLOCK_BYTES:
        raise SubsetError(f"block {block_index} subset payload coverage changed")
    encoded = json.dumps(value, separators=(",", ":"), sort_keys=True).encode("utf-8")
    encoded += b" " * ((-len(encoded)) % 8)
    if len(encoded) > base.MAX_HEADER_BYTES:
        raise SubsetError("rebased subset header exceeds bounded size")
    return struct.pack("<Q", len(encoded)) + encoded


def output_path(path: str | os.PathLike[str], block_index: int) -> Path:
    checked_block_index(block_index)
    root = ALLOWED_OUTPUT_ROOT.resolve(strict=True)
    candidate = Path(path).expanduser()
    if not candidate.is_absolute():
        candidate = Path.cwd() / candidate
    parent = candidate.parent.resolve(strict=True)
    output = parent / candidate.name
    if output.suffix != ".safetensors" or not output.is_relative_to(root) or output == root:
        raise SubsetError("output must be a safetensors file inside the designated external SSD root")
    if not parent.is_dir():
        raise SubsetError("output parent must already exist")
    return output


def extract(path: str | os.PathLike[str], block_index: int, *,
            max_payload_bytes: int = EXPECTED_BLOCK_BYTES) -> tuple[str, str]:
    """Stream one exact authenticated original-F32 block subset to external SSD."""
    block_index = checked_block_index(block_index)
    validate_transfer_budget(max_payload_bytes)
    output = output_path(path, block_index)
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
    raw, header_sha = base._read_range(
        url, 8, 8 + EXPECTED_HEADER_BYTES - 1, base.SOURCE_FILE_SIZE
    )
    plan = parse_safetensors_header(raw, base.SOURCE_FILE_SIZE, block_index)
    encoded = make_subset_header(plan)

    output_fd, output_temp = base._new_temp(output.parent, f".block{block_index}.partial")
    manifest_fd = None
    manifest_temp = None
    published_manifest = None
    published_output = None
    try:
        with os.fdopen(output_fd, "wb") as stream:
            sink = base._HashingSink(stream)
            sink.write(encoded)
            with base._request(url, byte_range=(plan.range_start, plan.range_end)) as response:
                payload_sha = base.copy_verified_range_body(
                    response, plan.range_start, plan.range_end, base.SOURCE_FILE_SIZE, sink
                )
            stream.flush()
            os.fsync(stream.fileno())
            output_sha = sink.hexdigest()

        manifest_record = {
            "format": "hy-nomas-block-subset-provenance-v1",
            "source": {
                "repository": base.HF_REPO,
                "revision": base.HF_REVISION,
                "checkpoint_path": base.CHECKPOINT_PATH,
                "checkpoint_size_bytes": base.SOURCE_FILE_SIZE,
                "checkpoint_lfs_sha256": base.SOURCE_LFS_SHA256,
                "checkpoint_lfs_sha256_status": (
                    "declared by pinned HF API; whole file not downloaded or verified"
                ),
                "source_git_commit": base.SOURCE_GIT_COMMIT,
                "header_sha256_observed": header_sha,
                "range_start_inclusive": plan.range_start,
                "range_end_inclusive": plan.range_end,
                "authentication_precondition": (
                    "HTTPS/TLS to pinned HF revision and its HTTPS range redirects"
                ),
                "payload_sha256_observed": payload_sha,
            },
            "subset": {
                "block_index": block_index,
                "tensor_prefix": f"double_blocks.{block_index}.",
                "tensor_count": len(plan.tensors),
                "tensor_names": sorted(record.name for record in plan.tensors),
                "dtype": "F32",
                "stored_bytes_preserved": True,
                "conversion_performed": False,
                "payload_bytes": plan.block_bytes,
            },
            "output": {
                "filename": output.name,
                "size_bytes": len(encoded) + plan.block_bytes,
                "sha256": output_sha,
            },
        }
        manifest_fd, manifest_temp = base._new_temp(output.parent, ".manifest.partial")
        with os.fdopen(manifest_fd, "wb") as stream:
            manifest_fd = None
            stream.write((json.dumps(manifest_record, indent=2, sort_keys=True) + "\n").encode())
            stream.flush()
            os.fsync(stream.fileno())
        manifest_stat = manifest_temp.lstat()
        base.atomic_publish_no_replace(manifest_temp, manifest)
        published_manifest = (manifest_stat.st_dev, manifest_stat.st_ino)
        output_stat = output_temp.lstat()
        base.atomic_publish_no_replace(output_temp, output)
        published_output = (output_stat.st_dev, output_stat.st_ino)
        base._fsync_directory(output.parent)
        published_manifest = None
        published_output = None
        return output_sha, payload_sha
    except BaseException:
        if published_output is not None:
            base._remove_if_same(output, *published_output)
        if published_manifest is not None:
            base._remove_if_same(manifest, *published_manifest)
        raise
    finally:
        if manifest_fd is not None:
            os.close(manifest_fd)
        for temporary in (output_temp, manifest_temp):
            if temporary is not None:
                temporary.unlink(missing_ok=True)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", help="new .safetensors path under the designated external SSD root")
    parser.add_argument("--block-index", type=_block_index_argument, required=True,
                        help="actual block index 2 or 3")
    parser.add_argument("--max-payload-bytes", type=int, default=EXPECTED_BLOCK_BYTES,
                        help="exact selected source payload transfer budget")
    args = parser.parse_args(argv)
    try:
        output_sha, payload_sha = extract(
            args.output, args.block_index, max_payload_bytes=args.max_payload_bytes
        )
    except (SubsetError, OSError, ValueError) as error:
        if isinstance(error, SubsetError):
            message = str(error)
        elif isinstance(error, FileExistsError):
            message = "output or provenance manifest already exists"
        else:
            message = f"local operation failed ({type(error).__name__})"
        print(f"extract failed: {message}", file=sys.stderr)
        return 2
    print(json.dumps({"status": "created", "block_index": args.block_index,
                      "tensor_count": 32, "payload_bytes": EXPECTED_BLOCK_BYTES,
                      "output_sha256": output_sha, "payload_sha256": payload_sha}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
