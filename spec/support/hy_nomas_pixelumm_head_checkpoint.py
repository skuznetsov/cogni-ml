#!/usr/bin/env python3
"""Authenticate and extract the contiguous PixelUMM global decoder head range.

This is a bounded byte-transfer helper. It never decodes tensors or performs
inference. The pinned header identifies four adjacent F32 tensors; the receipt
binds their exact source offsets and every transferred range byte.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile

import hy_nomas_pixelumm_checkpoint as checkpoint
import hy_nomas_checkpoint_subset as transport

ROOT = Path("/opt/mldata_ssd/cogni-ml/hy-nomas-pixelumm-head-20261027")
SOURCE_DIR = ROOT / "source"
DEFAULT_HEADER = checkpoint.ROOT / "source" / "pixelumm-header.json"
DEFAULT_OUTPUT = SOURCE_DIR / "head-f32.safetensors"
RANGE_CHUNK_BYTES = 64 * 1024 * 1024

HEAD_TENSORS = {
    "decoder.final_norm.weight": ((4096,), (52_510_720, 52_527_104)),
    "decoder.head.linear.bias": ((3072,), (52_527_104, 52_539_392)),
    "decoder.head.linear.weight": ((3072, 4096), (52_539_392, 102_871_040)),
    "decoder.head.norm.weight": ((4096,), (102_871_040, 102_887_424)),
}
HEAD_DATA_START = 52_510_720
HEAD_DATA_END = 102_887_424
HEAD_F32_BYTES = HEAD_DATA_END - HEAD_DATA_START
HEAD_BF16_BYTES = HEAD_F32_BYTES // 2
HEAD_KEYS_IN_SOURCE_ORDER = tuple(sorted(
    HEAD_TENSORS, key=lambda name: HEAD_TENSORS[name][1][0]
))


class HeadCheckpointError(ValueError):
    """The supplied header, subset, receipt, or budget is not admitted."""


def validate_transfer_budget(max_payload_bytes: int) -> None:
    """Require enough budget for the exact audited range and no extra range."""
    if type(max_payload_bytes) is not int or max_payload_bytes < HEAD_F32_BYTES:
        raise HeadCheckpointError("payload transfer budget is below the exact head range")
    if max_payload_bytes > HEAD_F32_BYTES:
        raise HeadCheckpointError("payload transfer budget exceeds the exact head range")


def head_subset_header(header: dict) -> tuple[bytes, int, int, tuple[str, ...]]:
    """Build a rebased safetensors header for the exact four contiguous tensors."""
    selected = {name: header[name] for name in HEAD_TENSORS if name in header}
    if set(selected) != set(HEAD_TENSORS):
        raise HeadCheckpointError("head tensor ownership mismatch")
    def source_start(item):
        value = item[1]
        offsets = value.get("data_offsets") if isinstance(value, dict) else None
        if (not isinstance(offsets, list) or len(offsets) != 2
                or any(type(offset) is not int for offset in offsets)):
            raise HeadCheckpointError(f"pinned head tensor offsets are malformed: {item[0]}")
        return offsets[0]

    ordered = sorted(selected.items(), key=source_start)
    cursor = HEAD_DATA_START
    result = {}
    for name, value in ordered:
        shape, offsets = HEAD_TENSORS[name]
        if (not isinstance(value, dict) or value.get("dtype") != "F32"
                or value.get("shape") != list(shape)
                or value.get("data_offsets") != list(offsets)):
            raise HeadCheckpointError(f"pinned head tensor metadata mismatch: {name}")
        lo, hi = offsets
        if lo != cursor:
            raise HeadCheckpointError("head source range contains a gap or overlap")
        result[name] = {
            "dtype": "F32", "shape": list(shape),
            "data_offsets": [lo - HEAD_DATA_START, hi - HEAD_DATA_START],
        }
        cursor = hi
    if cursor != HEAD_DATA_END or cursor - HEAD_DATA_START != HEAD_F32_BYTES:
        raise HeadCheckpointError("head source range does not match the audited byte extent")
    raw = json.dumps(result, separators=(",", ":")).encode()
    raw += b" " * (-len(raw) % 8)
    encoded = len(raw).to_bytes(8, "little") + raw
    return encoded, HEAD_DATA_START, HEAD_DATA_END, tuple(name for name, _ in ordered)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _regular_file(path: Path, what: str) -> Path:
    if path.is_symlink() or not path.is_file():
        raise HeadCheckpointError(f"{what} must be a regular non-symlink file")
    try:
        resolved = path.resolve(strict=True)
    except (OSError, RuntimeError) as error:
        raise HeadCheckpointError(f"cannot resolve {what}") from error
    if not resolved.is_file():
        raise HeadCheckpointError(f"{what} must resolve to a regular file")
    return resolved


def _expected_range_chunks() -> tuple[tuple[int, int], ...]:
    return tuple(
        (start + 8 + checkpoint.HEADER_BYTES,
         min(start + RANGE_CHUNK_BYTES, HEAD_DATA_END) + 8 + checkpoint.HEADER_BYTES - 1)
        for start in range(HEAD_DATA_START, HEAD_DATA_END, RANGE_CHUNK_BYTES)
    )


def extract_head_subset(
    raw_header: bytes, output: Path | str | None = None, *,
    root: Path | str = ROOT, max_payload_bytes: int = HEAD_F32_BYTES,
) -> dict:
    """Fetch the pinned head range to a fresh subset and write its receipt.

    This function performs network I/O when called. The default root is the
    dedicated head artifact directory, never the older block extractor root.
    """
    validate_transfer_budget(max_payload_bytes)
    try:
        header = checkpoint.parse_pinned_header(raw_header)
        inventory = checkpoint.inventory(header)
        encoded, lo, hi, keys = head_subset_header(header)
    except checkpoint.InventoryError as error:
        raise HeadCheckpointError(f"pinned PixelUMM header rejected: {error}") from error

    root_path = Path(root).expanduser()
    if not root_path.is_absolute() or root_path.is_symlink():
        raise HeadCheckpointError("head artifact root must be an absolute non-symlink path")
    root_path = root_path.resolve()
    source_dir = root_path / "source"
    source_dir.mkdir(parents=True, exist_ok=True)
    if source_dir.is_symlink() or not source_dir.is_dir() or source_dir.resolve() != source_dir:
        raise HeadCheckpointError("head source directory must be a canonical directory")
    output_path = Path(output if output is not None else root_path / "source" / "head-f32.safetensors").expanduser()
    if not output_path.is_absolute():
        output_path = (root_path / "source" / output_path).absolute()
    if output_path.parent.resolve() != (root_path / "source").resolve():
        raise HeadCheckpointError("head subset output must be in the dedicated head source directory")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    receipt_path = output_path.with_suffix(".receipt.json")
    if any(path.exists() or path.is_symlink() for path in (output_path, receipt_path)):
        raise HeadCheckpointError("refusing to replace an existing head subset or receipt")

    chunks = []
    partial_path: Path | None = None
    receipt_partial_path: Path | None = None
    partial_identity: tuple[int, int] | None = None
    receipt_partial_identity: tuple[int, int] | None = None
    published_subset = False
    published_receipt = False
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb", prefix=".head-f32.", suffix=".partial",
            dir=output_path.parent, delete=False,
        ) as sink:
            partial_path = Path(sink.name)
            stat = os.fstat(sink.fileno())
            partial_identity = (stat.st_dev, stat.st_ino)
            sink.write(encoded)
            for start in range(lo, hi, RANGE_CHUNK_BYTES):
                end = min(start + RANGE_CHUNK_BYTES, hi) - 1
                absolute_start = start + 8 + checkpoint.HEADER_BYTES
                absolute_end = end + 8 + checkpoint.HEADER_BYTES
                endpoint = checkpoint.URL + (
                    f"?pixelumm_head_range={absolute_start}-{absolute_end}"
                )
                with transport._request(
                    endpoint, byte_range=(absolute_start, absolute_end)
                ) as response:
                    digest = transport.copy_verified_range_body(
                        response, absolute_start, absolute_end,
                        checkpoint.SOURCE_BYTES, sink,
                    )
                chunks.append({
                    "start": absolute_start,
                    "end_inclusive": absolute_end,
                    "sha256": digest,
                })
            sink.flush()
            os.fsync(sink.fileno())
        expected_size = len(encoded) + hi - lo
        if partial_path.stat().st_size != expected_size:
            raise HeadCheckpointError("head subset has an unexpected file extent")
        receipt_data = {
            **inventory,
            "selected_subset": "decoder_head",
            "tensor_keys": list(keys),
            "source_data_offsets": [lo, hi],
            "source_f32_bytes": hi - lo,
            "range_chunks": chunks,
            "subset_file": output_path.name,
            "subset_bytes": expected_size,
            "subset_sha256": _sha256_file(partial_path),
        }
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", prefix=".head-f32-receipt.",
            suffix=".partial", dir=output_path.parent, delete=False,
        ) as sink:
            receipt_partial_path = Path(sink.name)
            stat = os.fstat(sink.fileno())
            receipt_partial_identity = (stat.st_dev, stat.st_ino)
            json.dump(receipt_data, sink, indent=2)
            sink.write("\n")
            sink.flush()
            os.fsync(sink.fileno())
        os.link(partial_path, output_path)
        published_subset = True
        os.link(receipt_partial_path, receipt_path)
        published_receipt = True
        for temporary in (partial_path, receipt_partial_path):
            try:
                temporary.unlink()
            except OSError:
                # The final pair is already published; a stale private temp
                # file is recoverable and must not turn success into a retry trap.
                pass
        return receipt_data
    except BaseException as error:
        # Remove only paths created by this call. In particular, if receipt
        # publication fails after the subset link, remove that owned orphan so
        # a transient failure does not permanently block a safe retry.
        if published_subset and not published_receipt and partial_identity is not None:
            try:
                output_identity = output_path.lstat()
                if (output_identity.st_dev, output_identity.st_ino) == partial_identity:
                    output_path.unlink()
            except OSError:
                pass
        for temporary, identity in (
            (partial_path, partial_identity),
            (receipt_partial_path, receipt_partial_identity),
        ):
            if temporary is not None and identity is not None:
                try:
                    temp_identity = temporary.lstat()
                    if (temp_identity.st_dev, temp_identity.st_ino) == identity:
                        temporary.unlink()
                except OSError:
                    pass
        if isinstance(error, Exception):
            raise HeadCheckpointError(f"head range extraction failed: {error}") from error
        raise


def validate_head_subset_receipt(
    raw_header: bytes, receipt: dict, subset_path: Path | str,
) -> dict:
    """Re-derive the exact head selection and verify the local subset bytes."""
    try:
        header = checkpoint.parse_pinned_header(raw_header)
        expected_inventory = checkpoint.inventory(header)
        encoded, lo, hi, keys = head_subset_header(header)
    except checkpoint.InventoryError as error:
        raise HeadCheckpointError(f"pinned PixelUMM header rejected: {error}") from error
    if not isinstance(receipt, dict):
        raise HeadCheckpointError("head subset receipt must be an object")
    for field in (
        "hf_revision", "source_commit", "source_path", "source_file_bytes",
        "header_bytes", "header_sha256", "stored_dtype", "full_file_hash_verified",
    ):
        if receipt.get(field) != expected_inventory[field]:
            raise HeadCheckpointError(f"head subset receipt {field} mismatch")
    if (receipt.get("selected_subset") != "decoder_head"
            or receipt.get("tensor_keys") != list(keys)
            or receipt.get("source_data_offsets") != [lo, hi]
            or receipt.get("source_f32_bytes") != hi - lo):
        raise HeadCheckpointError("head subset receipt ownership or source extent mismatch")
    expected_chunks = _expected_range_chunks()
    chunks = receipt.get("range_chunks")
    if not isinstance(chunks, list) or len(chunks) != len(expected_chunks):
        raise HeadCheckpointError("head subset receipt range inventory mismatch")
    for item, (start, end_inclusive) in zip(chunks, expected_chunks):
        if not isinstance(item, dict):
            raise HeadCheckpointError("head subset receipt range chunk is malformed")
        digest = item.get("sha256") if isinstance(item, dict) else None
        if (type(item.get("start")) is not int or item.get("start") != start
                or type(item.get("end_inclusive")) is not int
                or item.get("end_inclusive") != end_inclusive
                or not isinstance(digest, str) or len(digest) != 64
                or any(char not in "0123456789abcdef" for char in digest)):
            raise HeadCheckpointError("head subset receipt range chunk is malformed")

    path = Path(subset_path).expanduser()
    if path.is_symlink() or not path.is_file():
        raise HeadCheckpointError("head subset must be a regular non-symlink file")
    path = path.resolve()
    if path.name != receipt.get("subset_file"):
        raise HeadCheckpointError("head subset receipt filename mismatch")
    expected_size = len(encoded) + hi - lo
    if (type(receipt.get("subset_bytes")) is not int
            or receipt["subset_bytes"] != expected_size or path.stat().st_size != expected_size):
        raise HeadCheckpointError("head subset file size mismatch")

    aggregate = hashlib.sha256()
    actual_chunks = []
    with path.open("rb") as stream:
        prefix = stream.read(len(encoded))
        if prefix != encoded:
            raise HeadCheckpointError("head subset safetensors header differs from the pinned selection")
        aggregate.update(prefix)
        for item, (start, end_inclusive) in zip(chunks, expected_chunks):
            nbytes = end_inclusive - start + 1
            digest = hashlib.sha256()
            remaining = nbytes
            while remaining:
                block = stream.read(min(8 * 1024 * 1024, remaining))
                if not block:
                    raise HeadCheckpointError("head subset payload is truncated")
                aggregate.update(block)
                digest.update(block)
                remaining -= len(block)
            actual_chunks.append(digest.hexdigest())
        if stream.read(1):
            raise HeadCheckpointError("head subset has trailing data")
    if any(item["sha256"] != actual for item, actual in zip(chunks, actual_chunks)):
        raise HeadCheckpointError("head source range digest does not match subset payload")
    subset_sha = aggregate.hexdigest()
    if receipt.get("subset_sha256") != subset_sha:
        raise HeadCheckpointError("head subset digest does not match receipt")
    return {
        "selected_subset": "decoder_head", "subset_file": path.name,
        "subset_bytes": expected_size, "subset_sha256": subset_sha,
        "source_data_offsets": [lo, hi], "source_f32_bytes": hi - lo,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--header", type=Path, default=DEFAULT_HEADER)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--max-payload-bytes", type=int, default=HEAD_F32_BYTES)
    args = parser.parse_args(argv)
    try:
        header_path = _regular_file(args.header.expanduser().absolute(), "pinned header")
        raw_header = header_path.read_bytes()
        output = args.output or (args.root / "source" / "head-f32.safetensors")
        receipt = extract_head_subset(
            raw_header, output, root=args.root,
            max_payload_bytes=args.max_payload_bytes,
        )
    except (HeadCheckpointError, checkpoint.InventoryError, OSError) as error:
        print(json.dumps({"error": str(error)}), file=sys.stderr)
        return 2
    print(json.dumps({
        "subset": str(Path(output).resolve()),
        "receipt": str(Path(output).with_suffix(".receipt.json").resolve()),
        "source_data_offsets": receipt["source_data_offsets"],
        "source_f32_bytes": receipt["source_f32_bytes"],
        "subset_sha256": receipt["subset_sha256"],
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
