#!/usr/bin/env python3
"""Inventory the pinned PixelUMM header and extract one decoder layer.

This metadata/byte-transfer tool performs no tensor conversion or inference.
It reuses the existing strict bounded HF range transport. The full-file LFS
hash is advertised provenance; only fetched bytes receive observed hashes.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import struct
import sys

import hy_nomas_checkpoint_subset as transport

REVISION = "c67f2c52b3fbe0193cbc202769bbe5cb7264807b"
SOURCE_COMMIT = "911cea6458a89c2663513ff7099561ec190dd491"
SOURCE_PATH = "VAE/PixelUMM_FT/diffusion_pytorch_model.safetensors"
SOURCE_BYTES = 31_553_868_864
HEADER_BYTES = 75_192
PAYLOAD_BYTES = SOURCE_BYTES - 8 - HEADER_BYTES
HEADER_SHA256 = "7e7a92cfe84d54c0da3a132099a9472766ae0034042515a0f76b6198bbd09d11"
ADVERTISED_LFS_SHA256 = "0bbfe8fd44bd2ebf1059fe65079caba50fd6f52ea8e1b5b35393e3050b9b8378"
ROOT = Path("/opt/mldata_ssd/cogni-ml/hy-nomas-pixelumm-20261026")
URL = f"https://huggingface.co/tencent/Hy-Nomas/resolve/{REVISION}/{SOURCE_PATH}"


class InventoryError(ValueError):
    pass


def decoder_schema():
    result = {
        "decoder.cond_proj.bias": (16384,),
        "decoder.cond_proj.weight": (16384, 32),
        "decoder.embed.bias": (4096,),
        "decoder.embed.weight": (4096, 3072),
        "decoder.final_norm.weight": (4096,),
        "decoder.head.linear.bias": (3072,),
        "decoder.head.linear.weight": (3072, 4096),
        "decoder.head.norm.weight": (4096,),
    }
    for index in range(36):
        prefix = f"decoder.layers.{index}."
        layer = {"input_norm.weight": (4096,), "post_attn_norm.weight": (4096,)}
        for projection, shape in (("q", (4096, 4096)), ("k", (1024, 4096)),
                                  ("v", (1024, 4096)), ("o", (4096, 4096))):
            layer[f"attn.{projection}_proj.weight"] = shape
        for projection in ("q", "k"):
            for section in ("t", "hw"):
                layer[f"attn.{projection}_norm_{section}.weight"] = (64,)
        for projection, shape in (("gate", (12288, 4096)), ("up", (12288, 4096)),
                                  ("down", (4096, 12288))):
            layer[f"mlp.{projection}_proj.weight"] = shape
        result.update({prefix + name: shape for name, shape in layer.items()})
    return result


def _object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise InventoryError("duplicate JSON object key")
        result[key] = value
    return result


def parse_header(raw):
    try:
        value = json.loads(raw, object_pairs_hook=_object)
    except (ValueError, UnicodeDecodeError) as error:
        raise InventoryError("invalid header JSON") from error
    if not isinstance(value, dict):
        raise InventoryError("header must be a JSON object")
    return value


def parse_pinned_header(raw):
    if len(raw) != HEADER_BYTES or hashlib.sha256(raw).hexdigest() != HEADER_SHA256:
        raise InventoryError("header identity mismatch")
    value = parse_header(raw)
    inventory(value)
    return value


def inventory(header):
    expected = decoder_schema()
    tensors = {k: v for k, v in header.items() if k != "__metadata__"}
    decoder = {k: v for k, v in tensors.items() if k.startswith("decoder.")}
    encoder = {k: v for k, v in tensors.items() if k.startswith("encoder.")}
    if len(tensors) != 645 or len(encoder) != 169 or set(decoder) != set(expected):
        raise InventoryError("checkpoint ownership/key inventory mismatch")
    if len(decoder) + len(encoder) != len(tensors):
        raise InventoryError("unknown tensor ownership")
    extents = []
    sizes = {}
    for name, value in tensors.items():
        if not isinstance(value, dict) or value.get("dtype") != "F32":
            raise InventoryError("all pinned tensor storage must be F32")
        shape, offsets = value.get("shape"), value.get("data_offsets")
        if not isinstance(shape, list) or not shape or not all(type(n) is int and n > 0 for n in shape):
            raise InventoryError("invalid tensor shape")
        if name in expected and tuple(shape) != expected[name]:
            raise InventoryError("decoder tensor shape mismatch")
        if not isinstance(offsets, list) or len(offsets) != 2 or not all(type(n) is int for n in offsets):
            raise InventoryError("invalid tensor offsets")
        start, end = offsets
        if not 0 <= start < end <= PAYLOAD_BYTES or end - start != 4 * math.prod(shape):
            raise InventoryError("tensor byte extent mismatch")
        extents.append((start, end))
        sizes[name] = end - start
    cursor = 0
    for start, end in sorted(extents):
        if start != cursor:
            raise InventoryError("payload overlap or hole")
        cursor = end
    if cursor != PAYLOAD_BYTES:
        raise InventoryError("unowned payload tail")
    decoder_bytes = sum(sizes[k] for k in decoder)
    block_bytes = sum(v for k, v in sizes.items() if k.startswith("decoder.layers.0."))
    return {"schema": "cogni_ml.pixelumm_checkpoint_inventory", "schema_version": 1,
            "hf_revision": REVISION, "source_commit": SOURCE_COMMIT, "source_path": SOURCE_PATH,
            "source_file_bytes": SOURCE_BYTES, "header_bytes": HEADER_BYTES,
            "header_sha256": HEADER_SHA256, "advertised_lfs_sha256": ADVERTISED_LFS_SHA256,
            "full_file_hash_verified": False, "stored_dtype": "F32", "runtime_dtype": "BF16",
            "tensor_count": len(tensors), "decoder_tensor_count": len(decoder),
            "encoder_tensor_count": len(encoder), "decoder_f32_bytes": decoder_bytes,
            "decoder_bf16_bytes": decoder_bytes // 2,
            "encoder_f32_bytes": PAYLOAD_BYTES - decoder_bytes,
            "block0_f32_bytes": block_bytes, "block0_bf16_bytes": block_bytes // 2,
            "config": {"layers": 36, "dim": 4096, "ffn_dim": 12288, "heads": 32,
                       "kv_heads": 8, "head_dim": 128, "rms_eps": 1e-6}}


def block_schema(index: int) -> dict[str, tuple[int, ...]]:
    if type(index) is not int or not 0 <= index < 36:
        raise InventoryError("decoder block index must be in [0, 36)")
    prefix = f"decoder.layers.{index}."
    return {name.removeprefix(prefix): shape
            for name, shape in decoder_schema().items() if name.startswith(prefix)}


def subset_header(header, block_index: int = 0):
    if type(block_index) is not int or not 0 <= block_index < 36:
        raise InventoryError("decoder block index must be in [0, 36)")
    prefix = f"decoder.layers.{block_index}."
    expected = {prefix + name: shape for name, shape in block_schema(block_index).items()}
    selected = {k: v for k, v in header.items() if k.startswith(prefix)}
    if len(selected) != 13:
        raise InventoryError(f"expected exactly 13 layer-{block_index} tensors")
    if set(selected) != set(expected):
        raise InventoryError(f"layer-{block_index} tensor ownership mismatch")
    ordered = sorted(selected.items(), key=lambda item: item[1]["data_offsets"][0])
    start = ordered[0][1]["data_offsets"][0]
    cursor = start
    result = {}
    for name, value in ordered:
        if value.get("dtype") != "F32" or tuple(value.get("shape", ())) != expected[name]:
            raise InventoryError(f"layer-{block_index} tensor shape/dtype mismatch")
        lo, hi = value["data_offsets"]
        if lo != cursor:
            raise InventoryError(f"layer-{block_index} extent is not contiguous")
        result[name] = {"dtype": "F32", "shape": value["shape"],
                        "data_offsets": [lo - start, hi - start]}
        cursor = hi
    raw = json.dumps(result, separators=(",", ":")).encode()
    raw += b" " * (-len(raw) % 8)
    return struct.pack("<Q", len(raw)) + raw, start, cursor


def _sha_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def extract_block(raw_header, output, block_index: int):
    header = parse_pinned_header(raw_header)
    if type(block_index) is not int or not 0 <= block_index < 36:
        raise InventoryError("decoder block index must be in [0, 36)")
    output = Path(output)
    # Scope the network-writing command to its task's external-SSD scratch.
    if output.parent.resolve() != (ROOT / "source").resolve():
        raise InventoryError("subset output must be in the task source directory")
    receipt = output.with_suffix(".receipt.json")
    partial = output.with_suffix(output.suffix + ".partial")
    if output.exists() or receipt.exists() or partial.exists():
        raise InventoryError("refusing to replace an existing subset or receipt")
    encoded, lo, hi = subset_header(header, block_index)
    chunks = []
    with partial.open("xb") as sink:
        sink.write(encoded)
        for start in range(lo, hi, 64 * 1024 * 1024):
            end = min(start + 64 * 1024 * 1024, hi) - 1
            absolute_start = start + 8 + HEADER_BYTES
            absolute_end = end + 8 + HEADER_BYTES
            # Separate immutable query keys avoid a stale cached Range response.
            endpoint = URL + f"?pixelumm_range={absolute_start}-{absolute_end}"
            with transport._request(endpoint, byte_range=(absolute_start, absolute_end)) as response:
                digest = transport.copy_verified_range_body(
                    response, absolute_start, absolute_end, SOURCE_BYTES, sink)
            chunks.append({"start": absolute_start, "end_inclusive": absolute_end, "sha256": digest})
            print(json.dumps({"downloaded_layer_bytes": end + 1 - lo,
                              "total_layer_bytes": hi - lo}), flush=True)
    if partial.stat().st_size != len(encoded) + hi - lo:
        raise InventoryError("subset file extent mismatch")
    receipt_data = {**inventory(header), "selected_block": block_index,
                    "block_data_offsets": [lo, hi], "block_f32_bytes": hi - lo,
                    "range_chunks": chunks,
                    "subset_file": output.name, "subset_bytes": partial.stat().st_size,
                    "subset_sha256": _sha_file(partial)}
    # No overwrite even if another caller creates the destination during transfer.
    import os
    os.link(partial, output)
    partial.unlink()
    with receipt.open("x") as sink:
        json.dump(receipt_data, sink, indent=2)
        sink.write("\n")
    return receipt_data


def extract_block0(raw_header, output):
    """Backward-compatible block-zero extractor."""
    return extract_block(raw_header, output, 0)


def validate_block_receipt(raw_header, receipt, subset_path, block_index: int) -> dict:
    """Validate a local extracted block against the pinned header and receipt."""
    header = parse_pinned_header(raw_header)
    if type(block_index) is not int or not 0 <= block_index < 36:
        raise InventoryError("decoder block index must be in [0, 36)")
    if not isinstance(receipt, dict):
        raise InventoryError("block receipt must be a JSON object")
    expected_inventory = inventory(header)
    for field in ("hf_revision", "source_commit", "source_path", "source_file_bytes",
                  "header_bytes", "header_sha256", "stored_dtype", "full_file_hash_verified"):
        if receipt.get(field) != expected_inventory[field]:
            raise InventoryError(f"block receipt {field} mismatch")
    if receipt.get("selected_block") != block_index:
        raise InventoryError("block receipt index mismatch")
    encoded, lo, hi = subset_header(header, block_index)
    block_bytes = hi - lo
    has_offsets = "block_data_offsets" in receipt
    has_byte_count = "block_f32_bytes" in receipt
    legacy_block0 = block_index == 0 and not has_offsets and not has_byte_count
    if legacy_block0:
        # The already-fetched v1 block-zero receipt predates these redundant
        # fields. Its extent is rederived from the authenticated pinned header.
        pass
    elif not has_offsets or not has_byte_count:
        raise InventoryError("block receipt is missing source extent metadata")
    if has_offsets and receipt["block_data_offsets"] != [lo, hi]:
        raise InventoryError("block receipt source extent mismatch")
    if has_byte_count and receipt["block_f32_bytes"] != block_bytes:
        raise InventoryError("block receipt source byte count mismatch")
    chunks = receipt.get("range_chunks")
    if not isinstance(chunks, list) or not chunks:
        raise InventoryError("block receipt is missing fetched range chunks")
    expected_chunks = []
    for start in range(lo, hi, 64 * 1024 * 1024):
        end_exclusive = min(start + 64 * 1024 * 1024, hi)
        expected_chunks.append((start + 8 + HEADER_BYTES, end_exclusive + 8 + HEADER_BYTES - 1))
    if len(chunks) != len(expected_chunks):
        raise InventoryError("block receipt range chunk count mismatch")
    for item, (start, end_inclusive) in zip(chunks, expected_chunks):
        if (not isinstance(item, dict) or item.get("start") != start
                or item.get("end_inclusive") != end_inclusive):
            raise InventoryError("block receipt range chunk extent mismatch")
        digest = item.get("sha256")
        if not isinstance(digest, str) or len(digest) != 64 or any(
            c not in "0123456789abcdef" for c in digest
        ):
            raise InventoryError("invalid block receipt range digest")
    subset_path = Path(subset_path).expanduser().resolve()
    if receipt.get("subset_file") != subset_path.name:
        raise InventoryError("block receipt filename mismatch")
    if not subset_path.is_file():
        raise InventoryError("block subset file is missing")
    if receipt.get("subset_bytes") != subset_path.stat().st_size:
        raise InventoryError("block subset byte count mismatch")
    if subset_path.stat().st_size != len(encoded) + block_bytes:
        raise InventoryError("block subset extent does not match its pinned tensors")
    digest = hashlib.sha256()
    actual_chunk_hashes = []
    with subset_path.open("rb") as stream:
        prefix = stream.read(len(encoded))
        if prefix != encoded:
            raise InventoryError("block subset header differs from the pinned selection")
        digest.update(prefix)
        for start, end_inclusive in expected_chunks:
            chunk_bytes = end_inclusive - start + 1
            chunk = stream.read(chunk_bytes)
            if len(chunk) != chunk_bytes:
                raise InventoryError("block subset payload is truncated")
            digest.update(chunk)
            actual_chunk_hashes.append(hashlib.sha256(chunk).hexdigest())
        if stream.read(1):
            raise InventoryError("block subset has trailing data")
    for item, actual in zip(chunks, actual_chunk_hashes):
        if item["sha256"] != actual:
            raise InventoryError("block receipt range digest does not match subset payload")
    digest = digest.hexdigest()
    if receipt.get("subset_sha256") != digest:
        raise InventoryError("block subset digest does not match its receipt")
    return {
        "selected_block": block_index,
        "subset_file": subset_path.name,
        "subset_bytes": subset_path.stat().st_size,
        "subset_sha256": digest,
        "block_data_offsets": [lo, hi],
        "block_f32_bytes": block_bytes,
        "legacy_extent_derived_from_pinned_header": legacy_block0,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--header", required=True, type=Path)
    parser.add_argument("--report", required=True, type=Path)
    parser.add_argument("--extract-block0", type=Path)
    parser.add_argument("--extract-block", nargs=2, action="append", metavar=("INDEX", "PATH"))
    args = parser.parse_args()
    # Reject known collisions before any large transfer or subset mutation.
    if args.report.exists() or args.report.is_symlink():
        raise InventoryError("refusing to replace an existing inventory report")
    if args.report.parent.resolve() != (ROOT / "results").resolve():
        raise InventoryError("report must be in the task results directory")
    raw = args.header.read_bytes()
    header = parse_pinned_header(raw)
    report = inventory(header)
    requests = []
    if args.extract_block0:
        requests.append((0, args.extract_block0))
    for raw_index, path in args.extract_block or ():
        try:
            index = int(raw_index)
        except ValueError as error:
            raise InventoryError("--extract-block INDEX must be an integer") from error
        requests.append((index, Path(path)))
    if len({index for index, _ in requests}) != len(requests):
        raise InventoryError("each decoder block may be extracted only once per invocation")
    # Validate every destination before starting any bounded source transfer.
    for index, path in requests:
        if type(index) is not int or not 0 <= index < 36:
            raise InventoryError("decoder block index must be in [0, 36)")
        path = Path(path)
        if path.parent.resolve() != (ROOT / "source").resolve():
            raise InventoryError("subset output must be in the task source directory")
        receipt = path.with_suffix(".receipt.json")
        partial = path.with_suffix(path.suffix + ".partial")
        if any(candidate.exists() or candidate.is_symlink()
               for candidate in (path, receipt, partial)):
            raise InventoryError("refusing to replace an existing subset or receipt")
    subsets = {}
    for index, path in requests:
        subsets[str(index)] = extract_block(raw, path, index)
    if args.extract_block0:
        report["subset"] = subsets["0"]
    if args.extract_block:
        report["subsets"] = subsets
    with args.report.open("x") as sink:
        json.dump(report, sink, indent=2)
        sink.write("\n")
    print(json.dumps({"report": str(args.report), "decoder_bf16_bytes": report["decoder_bf16_bytes"]}))


if __name__ == "__main__":
    try:
        main()
    except (InventoryError, transport.SubsetError) as error:
        print(f"PixelUMM inventory rejected: {error}", file=sys.stderr)
        raise SystemExit(1)
