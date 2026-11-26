#!/usr/bin/env python3
"""Offline admission tests for the bounded Hy-Nomas checkpoint extractor."""

import io
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest

SUPPORT = Path(__file__).resolve().parent / "support"
sys.path.insert(0, str(SUPPORT))
TEMP_ROOT = Path("/opt/mldata_ssd/cogni-ml/hy-nomas-bf16-20261024")
TEMP_ROOT.mkdir(parents=True, exist_ok=True)

import hy_nomas_checkpoint_subset as subset  # noqa: E402


# Literal names and dimensions copied from the audited pinned safetensors
# header. Keep this independent of EXPECTED_TENSORS so a bad expectation cannot
# manufacture its own positive fixture.
OFFICIAL_BLOCK_TENSORS = {
    "double_blocks.0.img_mod.linear.weight": (12_288, 2_048),
    "double_blocks.0.img_mod.linear.bias": (12_288,),
    "double_blocks.0.img_attn_q.weight": (2_048, 2_048),
    "double_blocks.0.img_attn_q.bias": (2_048,),
    "double_blocks.0.img_attn_k.weight": (2_048, 2_048),
    "double_blocks.0.img_attn_k.bias": (2_048,),
    "double_blocks.0.img_attn_v.weight": (2_048, 2_048),
    "double_blocks.0.img_attn_v.bias": (2_048,),
    "double_blocks.0.img_attn_proj.weight": (2_048, 2_048),
    "double_blocks.0.img_attn_proj.bias": (2_048,),
    "double_blocks.0.img_attn_q_norm.weight": (128,),
    "double_blocks.0.img_attn_k_norm.weight": (128,),
    "double_blocks.0.img_mlp.fc1.weight": (8_192, 2_048),
    "double_blocks.0.img_mlp.fc1.bias": (8_192,),
    "double_blocks.0.img_mlp.fc2.weight": (2_048, 8_192),
    "double_blocks.0.img_mlp.fc2.bias": (2_048,),
    "double_blocks.0.txt_mod.linear.weight": (12_288, 2_048),
    "double_blocks.0.txt_mod.linear.bias": (12_288,),
    "double_blocks.0.txt_attn_q.weight": (2_048, 2_048),
    "double_blocks.0.txt_attn_q.bias": (2_048,),
    "double_blocks.0.txt_attn_k.weight": (2_048, 2_048),
    "double_blocks.0.txt_attn_k.bias": (2_048,),
    "double_blocks.0.txt_attn_v.weight": (2_048, 2_048),
    "double_blocks.0.txt_attn_v.bias": (2_048,),
    "double_blocks.0.txt_attn_proj.weight": (2_048, 2_048),
    "double_blocks.0.txt_attn_proj.bias": (2_048,),
    "double_blocks.0.txt_attn_q_norm.weight": (128,),
    "double_blocks.0.txt_attn_k_norm.weight": (128,),
    "double_blocks.0.txt_mlp.fc1.weight": (8_192, 2_048),
    "double_blocks.0.txt_mlp.fc1.bias": (8_192,),
    "double_blocks.0.txt_mlp.fc2.weight": (2_048, 8_192),
    "double_blocks.0.txt_mlp.fc2.bias": (2_048,),
}


def _padded_header(value):
    raw = json.dumps(value, separators=(",", ":"), sort_keys=True).encode("utf-8")
    if len(raw) > subset.EXPECTED_HEADER_BYTES:
        raise AssertionError("synthetic header exceeded audited header size")
    return raw + b" " * (subset.EXPECTED_HEADER_BYTES - len(raw))


def _valid_header(*, reverse_block_order=False):
    payload_size = subset.SOURCE_FILE_SIZE - subset.SOURCE_DATA_OFFSET
    cursor = 0
    value = {"__metadata__": {"fixture": "offline"}}

    before_bytes = subset.BLOCK_DATA_START
    value["outside.before"] = {
        "dtype": "F32",
        "shape": [before_bytes // 4],
        "data_offsets": [cursor, cursor + before_bytes],
    }
    cursor += before_bytes

    for name, shape in sorted(OFFICIAL_BLOCK_TENSORS.items(),
                               reverse=reverse_block_order):
        byte_count = 4
        for dimension in shape:
            byte_count *= dimension
        value[name] = {
            "dtype": "F32",
            "shape": list(shape),
            "data_offsets": [cursor, cursor + byte_count],
        }
        cursor += byte_count

    remaining = payload_size - cursor
    part = 0
    chunk_bytes = 1_000_000_000
    while remaining:
        amount = min(remaining, chunk_bytes)
        amount -= amount % 4
        if amount <= 0:
            raise AssertionError("bad synthetic remainder")
        value[f"outside.after.{part}"] = {
            "dtype": "F32",
            "shape": [amount // 4],
            "data_offsets": [cursor, cursor + amount],
        }
        cursor += amount
        remaining -= amount
        part += 1
    if cursor != payload_size:
        raise AssertionError("synthetic tensor payload does not cover source")
    return _padded_header(value)


class FakeResponse:
    def __init__(self, body, *, status=206, content_range=None,
                 content_length=None, content_encoding=None):
        self.status = status
        self.headers = {}
        if content_range is not None:
            self.headers["Content-Range"] = content_range
        if content_length is not None:
            self.headers["Content-Length"] = str(content_length)
        if content_encoding is not None:
            self.headers["Content-Encoding"] = content_encoding
        self._body = io.BytesIO(body)

    def read(self, size=-1):
        return self._body.read(size)


class HyNomasCheckpointSubsetTests(unittest.TestCase):
    def test_known_positive_header_matches_exact_block_inventory(self):
        self.assertEqual(subset.EXPECTED_TENSORS, OFFICIAL_BLOCK_TENSORS)
        plan = subset.parse_safetensors_header(_valid_header(), subset.SOURCE_FILE_SIZE)
        self.assertEqual(plan.range_start, subset.EXPECTED_RANGE_START)
        self.assertEqual(plan.range_end, subset.EXPECTED_RANGE_END)
        self.assertEqual(plan.block_bytes, subset.EXPECTED_BLOCK_BYTES)
        self.assertEqual(len(plan.tensors), 32)
        self.assertEqual({item.name for item in plan.tensors}, set(subset.EXPECTED_TENSORS))
        self.assertEqual([item.data_start for item in plan.tensors][0], subset.BLOCK_DATA_START)
        self.assertEqual([item.data_end for item in plan.tensors][-1], subset.BLOCK_DATA_END)

    def test_rejects_native_arena_mlp_aliases_as_checkpoint_names(self):
        value = json.loads(_valid_header())
        for stream in ("img", "txt"):
            for layer in ("fc1", "fc2"):
                for parameter in ("weight", "bias"):
                    official = f"double_blocks.0.{stream}_mlp.{layer}.{parameter}"
                    logical = f"double_blocks.0.{stream}_mlp_{layer}.{parameter}"
                    value[logical] = value.pop(official)
        with self.assertRaisesRegex(subset.SubsetError, "inventory mismatch"):
            subset.parse_safetensors_header(_padded_header(value), subset.SOURCE_FILE_SIZE)

    def test_rebased_output_header_preserves_names_and_contiguous_f32_ranges(self):
        for reverse in (False, True):
            with self.subTest(reverse_source_tensor_order=reverse):
                plan = subset.parse_safetensors_header(
                    _valid_header(reverse_block_order=reverse), subset.SOURCE_FILE_SIZE
                )
                raw = subset.make_subset_header(plan)
                header_length = int.from_bytes(raw[:8], "little")
                header = json.loads(raw[8:8 + header_length])
                entries = [value for name, value in header.items()
                           if name != "__metadata__"]
                ranges = sorted((entry["data_offsets"], entry["dtype"], entry["shape"])
                                for entry in entries)
                self.assertEqual(len(entries), 32)
                self.assertTrue(all(dtype == "F32" for _, dtype, _ in ranges))
                self.assertEqual(ranges[0][0][0], 0)
                self.assertEqual(ranges[-1][0][1], subset.EXPECTED_BLOCK_BYTES)
                self.assertTrue(all(left[0][1] == right[0][0]
                                    for left, right in zip(ranges, ranges[1:])))
                self.assertEqual(len(raw) % 8, 0)

    def test_rejects_duplicate_json_keys(self):
        raw = (b'{"double_blocks.0.img_attn_k.bias":'
               b'{"dtype":"F32","dtype":"F32","shape":[2048],'
               b'"data_offsets":[968017408,968025600]}}')
        with self.assertRaisesRegex(subset.SubsetError, "duplicate"):
            subset.parse_safetensors_header(raw, subset.SOURCE_FILE_SIZE)

    def test_rejects_wrong_shape_and_block_extent(self):
        value = json.loads(_valid_header())
        value["double_blocks.0.img_attn_k.bias"]["shape"] = [2047]
        with self.assertRaisesRegex(subset.SubsetError, "shape"):
            subset.parse_safetensors_header(_padded_header(value), subset.SOURCE_FILE_SIZE)

        value = json.loads(_valid_header())
        value["double_blocks.0.img_attn_k.bias"]["data_offsets"][0] += 4
        with self.assertRaises(subset.SubsetError):
            subset.parse_safetensors_header(_padded_header(value), subset.SOURCE_FILE_SIZE)

    def test_rejects_truncated_or_oversized_header_and_bad_source_size(self):
        valid = _valid_header()
        with self.assertRaisesRegex(subset.SubsetError, "header"):
            subset.parse_safetensors_header(valid[:-1], subset.SOURCE_FILE_SIZE)
        with self.assertRaisesRegex(subset.SubsetError, "header"):
            subset.parse_safetensors_header(b" " * (subset.MAX_HEADER_BYTES + 1),
                                            subset.SOURCE_FILE_SIZE)
        with self.assertRaisesRegex(subset.SubsetError, "size"):
            subset.parse_safetensors_header(valid, subset.SOURCE_FILE_SIZE - 1)

    def test_range_reader_accepts_exact_positive_and_rejects_bad_response(self):
        body = b"bounded-payload"
        total = 100
        start, end = 10, 10 + len(body) - 1
        response = FakeResponse(body, content_range=f"bytes {start}-{end}/{total}",
                                content_length=len(body))
        sink = io.BytesIO()
        digest = subset.copy_verified_range_body(response, start, end, total, sink,
                                                  chunk_size=3)
        self.assertEqual(sink.getvalue(), body)
        self.assertEqual(digest, hashlib.sha256(body).hexdigest())

        bad_responses = [
            FakeResponse(body, status=200, content_range=f"bytes {start}-{end}/{total}",
                         content_length=len(body)),
            FakeResponse(body, content_range=f"bytes {start + 1}-{end}/{total}",
                         content_length=len(body)),
            FakeResponse(body, content_range=f"bytes {start}-{end}/{total + 1}",
                         content_length=len(body)),
            FakeResponse(body, content_range=f"bytes {start}-{end}/{total}",
                         content_length=len(body), content_encoding="gzip"),
            FakeResponse(body[:-1], content_range=f"bytes {start}-{end}/{total}",
                         content_length=len(body)),
            FakeResponse(body + b"x", content_range=f"bytes {start}-{end}/{total}",
                         content_length=len(body) + 1),
        ]
        for bad in bad_responses:
            with self.subTest(headers=bad.headers, status=bad.status):
                with self.assertRaises(subset.SubsetError):
                    subset.copy_verified_range_body(bad, start, end, total, io.BytesIO(),
                                                     chunk_size=3)

    def test_api_metadata_is_pinned_and_budget_is_checked(self):
        self.assertEqual(
            subset._api_metadata_url(),
            f"https://huggingface.co/api/models/{subset.HF_REPO}/revision/"
            f"{subset.HF_REVISION}?blobs=true",
        )
        metadata = {
            "sha": subset.HF_REVISION,
            "siblings": [{
                "rfilename": subset.CHECKPOINT_PATH,
                "size": subset.SOURCE_FILE_SIZE,
                "lfs": {"sha256": subset.SOURCE_LFS_SHA256},
            }],
        }
        subset.validate_api_metadata(json.dumps(metadata).encode())
        metadata["siblings"][0]["lfs"]["sha256"] = "0" * 64
        with self.assertRaisesRegex(subset.SubsetError, "hash"):
            subset.validate_api_metadata(json.dumps(metadata).encode())
        with self.assertRaisesRegex(subset.SubsetError, "budget"):
            subset.validate_transfer_budget(subset.EXPECTED_BLOCK_BYTES - 1)
        subset.validate_transfer_budget(subset.EXPECTED_BLOCK_BYTES)

    def test_https_redirect_allowlist_is_domain_bounded_and_keeps_urls_private(self):
        for url in (
            "https://huggingface.co/api/models/test",
            "https://cdn-lfs.huggingface.co/object?signature=redacted",
            "https://cas-bridge.xethub.hf.co/object?signature=redacted",
        ):
            with self.subTest(url=url):
                self.assertTrue(subset._trusted_hf_https_url(url))
        for url in (
            "http://huggingface.co/file",
            "https://huggingface.co.attacker.invalid/file",
            "https://attacker-huggingface.co/file",
            "https://huggingface.co:8443/file",
            "https://user@huggingface.co/file",
        ):
            with self.subTest(url=url):
                self.assertFalse(subset._trusted_hf_https_url(url))

    def test_atomic_publish_refuses_overwrite(self):
        with tempfile.TemporaryDirectory(dir=TEMP_ROOT) as directory:
            source = Path(directory) / "subset.partial"
            target = Path(directory) / "subset.safetensors"
            source.write_bytes(b"new")
            target.write_bytes(b"existing")
            with self.assertRaises(FileExistsError):
                subset.atomic_publish_no_replace(source, target)
            self.assertEqual(source.read_bytes(), b"new")
            self.assertEqual(target.read_bytes(), b"existing")

    def test_atomic_publish_moves_complete_same_directory_file(self):
        with tempfile.TemporaryDirectory(dir=TEMP_ROOT) as directory:
            source = Path(directory) / "subset.partial"
            target = Path(directory) / "subset.safetensors"
            source.write_bytes(b"complete-new-output")
            subset.atomic_publish_no_replace(source, target)
            self.assertEqual(target.read_bytes(), b"complete-new-output")
            self.assertFalse(source.exists())


if __name__ == "__main__":
    unittest.main()
