#!/usr/bin/env python3
"""Extract the audited Hy-Nomas text refiner and timestep weights only.

The producer validates the complete pinned safetensors header, then downloads
one contiguous byte range containing ``time_in`` and ``txt_in``. Source tensor
bytes remain F32 and are not decoded or converted. This is a bounded subset,
not a complete checkpoint verification.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import struct
import sys
import tempfile
from typing import Any, BinaryIO
from urllib.parse import quote


HF_REPO = "tencent/Hy-Nomas"
HF_REVISION = "bac719a80c8d02f91c485fca804bb8221e149b70"
CHECKPOINT_PATH = "DiT/distilled/diffusion_pytorch_model_ema_0_995.safetensors"
SOURCE_LFS_SHA256 = "73819bcf3b82b090116dd1a1748795de95eef703c304e7443963d00ea8aa4d70"
SOURCE_FILE_SIZE = 34_213_991_280
SOURCE_GIT_COMMIT = "911cea6458a89c2663513ff7099561ec190dd491"
EXPECTED_HEADER_BYTES = 252_904
EXPECTED_HEADER_SHA256 = "cbfa83dc81af315b425fd783e0ce42d8ff09df772952ef2a8aff48493e21b200"
SOURCE_DATA_OFFSET = 8 + EXPECTED_HEADER_BYTES
TEXT_DATA_START = 33_630_427_008
TEXT_DATA_END = 34_213_738_368
EXPECTED_RANGE_START = SOURCE_DATA_OFFSET + TEXT_DATA_START
EXPECTED_RANGE_END = SOURCE_DATA_OFFSET + TEXT_DATA_END - 1
EXPECTED_TEXT_BYTES = TEXT_DATA_END - TEXT_DATA_START
MAX_HEADER_BYTES = 16 * 1024 * 1024
MAX_API_METADATA_BYTES = 2 * 1024 * 1024
ALLOWED_OUTPUT_ROOT = Path(
    "/opt/mldata_ssd/cogni-ml/hy-nomas-text-conditioning-20261025"
)


def _expected_tensors() -> dict[str, tuple[int, ...]]:
    expected: dict[str, tuple[int, ...]] = {
        "time_in.mlp.0.bias": (2048,),
        "time_in.mlp.0.weight": (2048, 256),
        "time_in.mlp.2.bias": (2048,),
        "time_in.mlp.2.weight": (2048, 2048),
        "txt_in.c_embedder.linear_1.bias": (2048,),
        "txt_in.c_embedder.linear_1.weight": (2048, 3584),
        "txt_in.c_embedder.linear_2.bias": (2048,),
        "txt_in.c_embedder.linear_2.weight": (2048, 2048),
        "txt_in.input_embedder.bias": (2048,),
        "txt_in.input_embedder.weight": (2048, 3584),
        "txt_in.t_embedder.mlp.0.bias": (2048,),
        "txt_in.t_embedder.mlp.0.weight": (2048, 256),
        "txt_in.t_embedder.mlp.2.bias": (2048,),
        "txt_in.t_embedder.mlp.2.weight": (2048, 2048),
    }
    for block in range(2):
        root = f"txt_in.individual_token_refiner.blocks.{block}"
        expected[f"{root}.adaLN_modulation.1.bias"] = (4096,)
        expected[f"{root}.adaLN_modulation.1.weight"] = (4096, 2048)
        expected[f"{root}.mlp.fc1.bias"] = (8192,)
        expected[f"{root}.mlp.fc1.weight"] = (8192, 2048)
        expected[f"{root}.mlp.fc2.bias"] = (2048,)
        expected[f"{root}.mlp.fc2.weight"] = (2048, 8192)
        for norm in ("norm1", "norm2"):
            expected[f"{root}.{norm}.bias"] = (2048,)
            expected[f"{root}.{norm}.weight"] = (2048,)
        expected[f"{root}.self_attn_proj.bias"] = (2048,)
        expected[f"{root}.self_attn_proj.weight"] = (2048, 2048)
        expected[f"{root}.self_attn_qkv.bias"] = (6144,)
        expected[f"{root}.self_attn_qkv.weight"] = (6144, 2048)
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
    __slots__ = ("tensors", "range_start", "range_end", "payload_bytes")

    def __init__(self, tensors: tuple[TensorRecord, ...], range_start: int,
                 range_end: int, payload_bytes: int) -> None:
        self.tensors = tensors
        self.range_start = range_start
        self.range_end = range_end
        self.payload_bytes = payload_bytes


def _object_without_duplicate_keys(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise SubsetError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _decode_header(raw: bytes) -> dict[str, Any]:
    if not isinstance(raw, bytes) or len(raw) > MAX_HEADER_BYTES:
        raise SubsetError("header exceeds bounded size")
    try:
        value = json.loads(
            raw.decode("utf-8"), object_pairs_hook=_object_without_duplicate_keys
        )
    except SubsetError:
        raise
    except (UnicodeDecodeError, json.JSONDecodeError, RecursionError):
        raise SubsetError("invalid safetensors header JSON") from None
    if not isinstance(value, dict):
        raise SubsetError("safetensors header must be an object")
    return value


def _record_subset_header(raw: bytes) -> HeaderPlan:
    """Validate exact text tensors and their audited contiguous source range."""
    header = _decode_header(raw)
    metadata = header.pop("__metadata__", None)
    if metadata is not None and (
        not isinstance(metadata, dict)
        or any(not isinstance(key, str) or not isinstance(value, str)
               for key, value in metadata.items())
    ):
        raise SubsetError("invalid safetensors metadata")

    names = {name for name in header if name.startswith(("time_in.", "txt_in."))}
    if names != set(EXPECTED_TENSORS):
        missing = sorted(set(EXPECTED_TENSORS) - names)
        extra = sorted(names - set(EXPECTED_TENSORS))
        raise SubsetError(
            f"text subset tensor inventory mismatch (missing={len(missing)}, extra={len(extra)})"
        )

    records: list[TensorRecord] = []
    for name, expected_shape in EXPECTED_TENSORS.items():
        descriptor = header[name]
        if not isinstance(descriptor, dict) or set(descriptor) != {
            "dtype", "shape", "data_offsets"
        }:
            raise SubsetError(f"invalid tensor descriptor for {name}")
        dtype = descriptor["dtype"]
        shape = descriptor["shape"]
        offsets = descriptor["data_offsets"]
        if dtype != "F32" or shape != list(expected_shape):
            raise SubsetError(f"text subset dtype or shape mismatch for {name}")
        if (not isinstance(offsets, list) or len(offsets) != 2
                or any(type(value) is not int for value in offsets)):
            raise SubsetError(f"invalid tensor offsets for {name}")
        start, end = offsets
        expected_bytes = 4
        for dimension in expected_shape:
            expected_bytes *= dimension
        if start < 0 or end - start != expected_bytes:
            raise SubsetError(f"text tensor byte count mismatch for {name}")
        records.append(TensorRecord(name, dtype, expected_shape, start, end))

    by_offset = sorted(records, key=lambda item: (item.data_start, item.data_end, item.name))
    if (by_offset[0].data_start != TEXT_DATA_START
            or by_offset[-1].data_end != TEXT_DATA_END):
        raise SubsetError("text subset extent differs from pinned source")
    for left, right in zip(by_offset, by_offset[1:]):
        if left.data_end != right.data_start:
            raise SubsetError("text subset tensors are not contiguous and nonoverlapping")
    if sum(item.data_end - item.data_start for item in by_offset) != EXPECTED_TEXT_BYTES:
        raise SubsetError("text subset payload byte count differs from pinned source")
    return HeaderPlan(
        tuple(by_offset), EXPECTED_RANGE_START, EXPECTED_RANGE_END, EXPECTED_TEXT_BYTES
    )


def parse_safetensors_header(raw: bytes, source_size: int) -> HeaderPlan:
    """Validate the exact upstream header and return only its text-weight range."""
    if source_size != SOURCE_FILE_SIZE:
        raise SubsetError("source size does not match pinned file size")
    if len(raw) != EXPECTED_HEADER_BYTES:
        raise SubsetError("header length does not match audited source")
    if hashlib.sha256(raw).hexdigest() != EXPECTED_HEADER_SHA256:
        raise SubsetError("pinned source header SHA-256 mismatch")

    # Reuse the pre-existing full-layout validator: it checks all source tensor
    # extents and the previously audited block-0 range. Its output is discarded.
    support_dir = Path(__file__).resolve().parent
    if str(support_dir) not in sys.path:
        sys.path.insert(0, str(support_dir))
    import hy_nomas_checkpoint_subset as block_subset

    block_subset.parse_safetensors_header(raw, source_size)
    return _record_subset_header(raw)


def make_subset_header(plan: HeaderPlan) -> bytes:
    """Serialize the audited 42 F32 tensors with payload offsets rebased to 0."""
    if (plan.payload_bytes != EXPECTED_TEXT_BYTES
            or plan.range_start != EXPECTED_RANGE_START
            or plan.range_end != EXPECTED_RANGE_END
            or {item.name for item in plan.tensors} != set(EXPECTED_TENSORS)):
        raise SubsetError("refusing to serialize an unrecognized text subset plan")
    cursor = 0
    value: dict[str, Any] = {
        "__metadata__": {
            "format": "pt",
            "hy_nomas_hf_revision": HF_REVISION,
            "hy_nomas_source_file": CHECKPOINT_PATH,
            "hy_nomas_source_lfs_sha256_declared": SOURCE_LFS_SHA256,
            "hy_nomas_source_lfs_sha256_verified": "false",
            "hy_nomas_source_git_commit": SOURCE_GIT_COMMIT,
            "hy_nomas_subset": "time_in and txt_in F32 stored bytes; no conversion",
        }
    }
    for record in sorted(plan.tensors, key=lambda item: (item.data_start, item.data_end)):
        if record.data_start != TEXT_DATA_START + cursor:
            raise SubsetError("text subset source payload is not the audited contiguous range")
        size = record.data_end - record.data_start
        value[record.name] = {
            "dtype": record.dtype,
            "shape": list(record.shape),
            "data_offsets": [cursor, cursor + size],
        }
        cursor += size
    if cursor != EXPECTED_TEXT_BYTES:
        raise SubsetError("text subset tensors do not cover the expected payload")
    encoded = json.dumps(value, separators=(",", ":"), sort_keys=True).encode("utf-8")
    encoded += b" " * ((-len(encoded)) % 8)
    if len(encoded) > MAX_HEADER_BYTES:
        raise SubsetError("rebased text subset header exceeds bounded size")
    return struct.pack("<Q", len(encoded)) + encoded


def validate_api_metadata(raw: bytes) -> None:
    if not isinstance(raw, bytes) or len(raw) > MAX_API_METADATA_BYTES:
        raise SubsetError("HF API metadata exceeds bounded size")
    try:
        value = json.loads(raw.decode("utf-8"), object_pairs_hook=_object_without_duplicate_keys)
    except SubsetError:
        raise
    except (UnicodeDecodeError, json.JSONDecodeError, RecursionError):
        raise SubsetError("invalid HF API metadata JSON") from None
    if not isinstance(value, dict) or value.get("sha") != HF_REVISION:
        raise SubsetError("HF API returned an unexpected revision")
    siblings = value.get("siblings")
    if not isinstance(siblings, list):
        raise SubsetError("HF API metadata has no sibling inventory")
    matches = [item for item in siblings
               if isinstance(item, dict) and item.get("rfilename") == CHECKPOINT_PATH]
    if len(matches) != 1:
        raise SubsetError("pinned checkpoint is absent or duplicated in HF inventory")
    item = matches[0]
    if item.get("size") != SOURCE_FILE_SIZE:
        raise SubsetError("HF checkpoint size mismatch")
    lfs = item.get("lfs")
    if not isinstance(lfs, dict) or lfs.get("sha256") != SOURCE_LFS_SHA256:
        raise SubsetError("HF checkpoint declared hash mismatch")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _output_path(path: str | os.PathLike[str]) -> Path:
    root = ALLOWED_OUTPUT_ROOT.resolve(strict=True)
    requested = Path(path).expanduser()
    if not requested.is_absolute():
        requested = Path.cwd() / requested
    parent = requested.parent.resolve(strict=True)
    canonical = parent / requested.name
    if not canonical.is_relative_to(root) or canonical == root:
        raise SubsetError("output must be inside the designated external SSD directory")
    if canonical.suffix != ".safetensors" or not parent.is_dir():
        raise SubsetError("output must name a .safetensors file in an existing directory")
    return canonical


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _publish_no_replace(source: Path, target: Path) -> None:
    if source.parent.resolve() != target.parent.resolve():
        raise SubsetError("atomic publication requires one output directory")
    os.link(source, target, follow_symlinks=False)
    try:
        _fsync_directory(target.parent)
    except OSError:
        target.unlink(missing_ok=True)
        raise
    source.unlink()


class _HashingSink:
    def __init__(self, stream: BinaryIO) -> None:
        self.stream = stream
        self.digest = hashlib.sha256()

    def write(self, data: bytes) -> int:
        written = self.stream.write(data)
        if written != len(data):
            raise SubsetError("short write while saving text subset")
        self.digest.update(data)
        return written

    def hexdigest(self) -> str:
        return self.digest.hexdigest()


def extract(output_path: str | os.PathLike[str], *,
            max_payload_bytes: int = EXPECTED_TEXT_BYTES) -> tuple[str, str]:
    """Fetch the exact F32 ``time_in`` + ``txt_in`` range and new manifest."""
    if type(max_payload_bytes) is not int or max_payload_bytes < EXPECTED_TEXT_BYTES:
        raise SubsetError("payload transfer budget is below the audited text subset size")
    if max_payload_bytes > EXPECTED_TEXT_BYTES:
        raise SubsetError("payload transfer budget exceeds the exact audited subset")
    output = _output_path(output_path)
    manifest = output.with_suffix(output.suffix + ".manifest.json")
    if os.path.lexists(output) or os.path.lexists(manifest):
        raise FileExistsError("output or provenance manifest already exists")

    support_dir = Path(__file__).resolve().parent
    if str(support_dir) not in sys.path:
        sys.path.insert(0, str(support_dir))
    import hy_nomas_checkpoint_subset as block_subset

    file_url = (
        f"https://huggingface.co/{HF_REPO}/resolve/{HF_REVISION}/"
        f"{quote(CHECKPOINT_PATH, safe='/')}"
    )
    with block_subset._request(block_subset._api_metadata_url()) as response:
        metadata_raw = block_subset._read_bounded(response, MAX_API_METADATA_BYTES)
    validate_api_metadata(metadata_raw)

    prefix, _ = block_subset._read_range(file_url, 0, 7, SOURCE_FILE_SIZE)
    if len(prefix) != 8 or struct.unpack("<Q", prefix)[0] != EXPECTED_HEADER_BYTES:
        raise SubsetError("safetensors prefix differs from pinned header length")
    header, header_sha = block_subset._read_range(
        file_url, 8, 8 + EXPECTED_HEADER_BYTES - 1, SOURCE_FILE_SIZE
    )
    if header_sha != EXPECTED_HEADER_SHA256:
        raise SubsetError("pinned source header SHA-256 mismatch")
    plan = parse_safetensors_header(header, SOURCE_FILE_SIZE)
    output_header = make_subset_header(plan)

    fd, temp_name = tempfile.mkstemp(prefix=".hy-nomas-text-", suffix=".partial",
                                     dir=output.parent)
    temp = Path(temp_name)
    manifest_temp = output.parent / f".{manifest.name}.{os.getpid()}.partial"
    published: list[Path] = []
    try:
        with os.fdopen(fd, "wb") as stream:
            sink = _HashingSink(stream)
            sink.write(output_header)
            with block_subset._request(
                file_url, byte_range=(plan.range_start, plan.range_end)
            ) as response:
                payload_sha = block_subset.copy_verified_range_body(
                    response,
                    plan.range_start,
                    plan.range_end,
                    SOURCE_FILE_SIZE,
                    sink,
                )
            stream.flush()
            os.fsync(stream.fileno())
            output_sha = sink.hexdigest()

        record = {
            "format": "hy-nomas-text-subset-provenance-v1",
            "source": {
                "repository": HF_REPO,
                "revision": HF_REVISION,
                "checkpoint_path": CHECKPOINT_PATH,
                "checkpoint_size_bytes": SOURCE_FILE_SIZE,
                "checkpoint_lfs_sha256": SOURCE_LFS_SHA256,
                "checkpoint_lfs_sha256_status": "declared by pinned HF API; whole file not downloaded or verified",
                "source_git_commit": SOURCE_GIT_COMMIT,
                "header_sha256_observed": header_sha,
                "range_start_inclusive": plan.range_start,
                "range_end_inclusive": plan.range_end,
                "payload_sha256_observed": payload_sha,
            },
            "subset": {
                "format": "safetensors",
                "dtype": "F32",
                "tensor_count": len(plan.tensors),
                "tensor_prefixes": ["time_in.", "txt_in."],
                "payload_bytes": plan.payload_bytes,
                "payload_range_relative_to_source_data": [TEXT_DATA_START, TEXT_DATA_END],
                "file_sha256": output_sha,
                "file_bytes": len(output_header) + plan.payload_bytes,
                "tensor_names": sorted(EXPECTED_TENSORS),
            },
        }
        with manifest_temp.open("xb") as stream:
            stream.write(json.dumps(record, indent=2, sort_keys=True).encode("utf-8") + b"\n")
            stream.flush()
            os.fsync(stream.fileno())

        _publish_no_replace(temp, output)
        published.append(output)
        _publish_no_replace(manifest_temp, manifest)
        published.append(manifest)
        return output_sha, payload_sha
    except Exception:
        for path in published:
            path.unlink(missing_ok=True)
        raise
    finally:
        temp.unlink(missing_ok=True)
        manifest_temp.unlink(missing_ok=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True,
                        help="new .safetensors path under the external SSD allowlist")
    parser.add_argument("--max-payload-bytes", type=int, default=EXPECTED_TEXT_BYTES,
                        help="must equal the exact pinned 583,311,360-byte range")
    args = parser.parse_args()
    try:
        output_sha, payload_sha = extract(
            args.output, max_payload_bytes=args.max_payload_bytes
        )
    except (FileNotFoundError, FileExistsError, OSError, SubsetError) as exc:
        parser.error(str(exc))
    print(json.dumps({
        "status": "created",
        "output": str(args.output.resolve()),
        "manifest": str(args.output.resolve().with_suffix(args.output.suffix + ".manifest.json")),
        "output_sha256": output_sha,
        "payload_sha256": payload_sha,
        "payload_bytes": EXPECTED_TEXT_BYTES,
        "tensor_count": len(EXPECTED_TENSORS),
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
