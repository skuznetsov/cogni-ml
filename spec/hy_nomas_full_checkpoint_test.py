#!/usr/bin/env python3
"""CPU-only falsifiers for the bounded pinned full-checkpoint downloader."""

from __future__ import annotations

import hashlib
import io
import json
from pathlib import Path
import struct
import sys
import tempfile
import types
import unittest

SUPPORT = Path(__file__).resolve().parent / "support"
sys.path.insert(0, str(SUPPORT))

import hy_nomas_full_checkpoint as downloader


HEADER = b'{"ok":true}'
PREFIX = struct.pack("<Q", len(HEADER)) + HEADER
PAYLOAD = bytes((index * 17 + 3) & 0xFF for index in range(131_117))
SOURCE_BYTES = PREFIX + PAYLOAD


def tiny_spec(*, digest: str | None = None, revision: str = "test-pinned-revision"):
    return downloader.CheckpointSpec(
        repository="tencent/Hy-Nomas",
        revision=revision,
        path="DiT/distilled/test.safetensors",
        size_bytes=len(SOURCE_BYTES),
        sha256=digest or hashlib.sha256(SOURCE_BYTES).hexdigest(),
        header_bytes=len(HEADER),
        header_sha256=hashlib.sha256(HEADER).hexdigest(),
    )


class RangeResponse(io.BytesIO):
    def __init__(self, body: bytes, start: int, end: int, total: int):
        super().__init__(body)
        self.status = 206
        self.headers = {
            "Content-Range": f"bytes {start}-{end}/{total}",
            "Content-Length": str(len(body)),
            "Content-Encoding": "identity",
        }


class MemorySource:
    def __init__(self, source: bytes, *, fail_after_payload_ranges: int | None = None):
        self.source = source
        self.fail_after_payload_ranges = fail_after_payload_ranges
        self.payload_ranges = 0
        self.ranges: list[tuple[int, int]] = []

    def __call__(self, url, start, end, deadline, clock):
        if start != 0:
            self.payload_ranges += 1
            if self.fail_after_payload_ranges is not None and (
                self.payload_ranges > self.fail_after_payload_ranges
            ):
                raise downloader.DownloadError("injected transport interruption")
        if end >= len(self.source) or start < 0 or end < start:
            raise AssertionError("test requested an invalid source range")
        self.ranges.append((start, end))
        return RangeResponse(self.source[start:end + 1], start, end, len(self.source))


class DownloaderTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name) / "root"
        self.weights = self.root / "weights"
        self.weights.mkdir(parents=True)
        self.output = self.weights / "model.safetensors"
        self.disk_usage = lambda _path: types.SimpleNamespace(free=10**9)

    def tearDown(self):
        self.temporary.cleanup()

    @staticmethod
    def header_validator(raw, source_size, spec):
        if raw != HEADER or source_size != len(SOURCE_BYTES):
            raise downloader.DownloadError("synthetic header rejected")
        if hashlib.sha256(raw).hexdigest() != spec.header_sha256:
            raise downloader.DownloadError("synthetic header digest rejected")

    def run_download(self, *, spec=None, opener=None, max_transfer_bytes=None):
        spec = spec or tiny_spec()
        return downloader._download_checkpoint(
            self.output,
            deadline_seconds=60,
            max_transfer_bytes=max_transfer_bytes or 2 * spec.size_bytes,
            disk_reserve_bytes=1024,
            chunk_bytes=64 * 1024,
            root=self.root,
            spec=spec,
            metadata_check=lambda *_args: None,
            opener=opener or MemorySource(SOURCE_BYTES),
            header_validator=self.header_validator,
            disk_usage=self.disk_usage,
        )

    def test_malformed_pinned_header_rejected(self):
        with self.assertRaisesRegex(downloader.DownloadError, "header length"):
            downloader.validate_checkpoint_header(b"bad")

    def test_exact_digest_publishes_after_safe_resume(self):
        interrupted = MemorySource(SOURCE_BYTES, fail_after_payload_ranges=1)
        with self.assertRaisesRegex(downloader.DownloadError, "interruption"):
            self.run_download(opener=interrupted)

        partial = self.output.with_name(self.output.name + ".partial")
        state_path = self.output.with_name(self.output.name + ".partial.json")
        state = json.loads(state_path.read_text())
        self.assertEqual(state["committed_bytes"], len(PREFIX) + 64 * 1024)
        self.assertFalse(self.output.exists())
        self.assertTrue(partial.exists())

        resumed = self.run_download(opener=MemorySource(SOURCE_BYTES))
        self.assertEqual(self.output.read_bytes(), SOURCE_BYTES)
        self.assertEqual(hashlib.sha256(self.output.read_bytes()).hexdigest(),
                         hashlib.sha256(SOURCE_BYTES).hexdigest())
        self.assertEqual(resumed["sha256"], hashlib.sha256(SOURCE_BYTES).hexdigest())
        self.assertGreater(resumed["network_bytes_reserved"], len(SOURCE_BYTES))
        self.assertFalse(partial.exists())
        self.assertFalse(state_path.exists())

    def test_header_request_reservation_survives_interruption(self):
        class FailHeader:
            def __init__(self):
                self.source = MemorySource(SOURCE_BYTES)

            def __call__(self, url, start, end, deadline, clock):
                if start == 0:
                    raise downloader.DownloadError("injected header interruption")
                return self.source(url, start, end, deadline, clock)

        with self.assertRaisesRegex(downloader.DownloadError, "header interruption"):
            self.run_download(opener=FailHeader())
        state_path = self.output.with_name(self.output.name + ".partial.json")
        state = json.loads(state_path.read_text())
        self.assertEqual(state["committed_bytes"], 0)
        self.assertEqual(state["transfer_reserved_bytes"], len(PREFIX))

        result = self.run_download(opener=MemorySource(SOURCE_BYTES))
        self.assertEqual(self.output.read_bytes(), SOURCE_BYTES)
        self.assertGreater(result["network_bytes_reserved"],
                           len(SOURCE_BYTES) + len(PREFIX) - 1)

    def test_stale_source_state_is_rejected_before_resuming(self):
        interrupted = MemorySource(SOURCE_BYTES, fail_after_payload_ranges=0)
        with self.assertRaisesRegex(downloader.DownloadError, "interruption"):
            self.run_download(opener=interrupted)
        state_path = self.output.with_name(self.output.name + ".partial.json")
        state = json.loads(state_path.read_text())
        state["source"]["revision"] = "stale-revision"
        state_path.write_text(json.dumps(state))

        with self.assertRaisesRegex(downloader.DownloadError, "different pinned source"):
            self.run_download(opener=MemorySource(SOURCE_BYTES))

    def test_digest_mismatch_never_publishes(self):
        bad_digest = "0" * 64
        with self.assertRaisesRegex(downloader.DownloadError, "differs from pinned LFS digest"):
            self.run_download(spec=tiny_spec(digest=bad_digest))
        self.assertFalse(self.output.exists())
        self.assertTrue(self.output.with_name(self.output.name + ".partial").exists())

    def test_partial_prefix_tamper_is_rejected(self):
        interrupted = MemorySource(SOURCE_BYTES, fail_after_payload_ranges=1)
        with self.assertRaisesRegex(downloader.DownloadError, "interruption"):
            self.run_download(opener=interrupted)
        partial = self.output.with_name(self.output.name + ".partial")
        with partial.open("r+b") as stream:
            stream.seek(len(PREFIX))
            byte = stream.read(1)
            stream.seek(len(PREFIX))
            stream.write(bytes([byte[0] ^ 0xFF]))

        with self.assertRaisesRegex(downloader.DownloadError, "prefix digest"):
            self.run_download(opener=MemorySource(SOURCE_BYTES))

    def test_range_transfer_and_disk_guards(self):
        spec = tiny_spec()
        with self.assertRaisesRegex(downloader.DownloadError, "transfer budget"):
            downloader.validate_budgets(
                spec=spec, deadline_seconds=60,
                max_transfer_bytes=spec.size_bytes,
                disk_reserve_bytes=1024, chunk_bytes=64 * 1024,
            )
        with self.assertRaisesRegex(downloader.DownloadError, "deadline"):
            downloader.validate_budgets(
                spec=spec, deadline_seconds=0,
                max_transfer_bytes=2 * spec.size_bytes,
                disk_reserve_bytes=1024, chunk_bytes=64 * 1024,
            )
        with self.assertRaisesRegex(downloader.DownloadError, "reserve"):
            downloader.validate_budgets(
                spec=spec, deadline_seconds=60,
                max_transfer_bytes=2 * spec.size_bytes,
                disk_reserve_bytes=0, chunk_bytes=64 * 1024,
            )

        with self.assertRaisesRegex(downloader.DownloadError, "free space"):
            downloader._download_checkpoint(
                self.output, deadline_seconds=60,
                max_transfer_bytes=2 * spec.size_bytes,
                disk_reserve_bytes=10**9, chunk_bytes=64 * 1024,
                root=self.root, spec=spec, metadata_check=lambda *_args: None,
                opener=MemorySource(SOURCE_BYTES), header_validator=self.header_validator,
                disk_usage=lambda _path: types.SimpleNamespace(free=10),
            )
        self.assertFalse(self.output.exists())
        self.assertFalse(self.output.with_name(self.output.name + ".partial").exists())

    def test_range_chunk_budget_boundary(self):
        spec = tiny_spec()
        common = {
            "spec": spec,
            "deadline_seconds": 60,
            "max_transfer_bytes": 2 * spec.size_bytes,
            "disk_reserve_bytes": 1024,
        }
        downloader.validate_budgets(**common, chunk_bytes=downloader.MAX_CHUNK_BYTES)
        with self.assertRaisesRegex(downloader.DownloadError, "range chunk size"):
            downloader.validate_budgets(
                **common, chunk_bytes=downloader.MAX_CHUNK_BYTES + 1
            )

    def test_output_path_must_stay_under_designated_root(self):
        outside = Path(self.temporary.name) / "outside.safetensors"
        with self.assertRaisesRegex(downloader.DownloadError, "under the designated SSD root"):
            downloader._output_path(outside, root=self.root)

    def test_wrong_range_extent_is_rejected(self):
        class BadRange(MemorySource):
            def __call__(self, url, start, end, deadline, clock):
                return RangeResponse(
                    self.source[start:end + 1], start + 1, end + 1, len(self.source)
                )

        with self.assertRaisesRegex(downloader.DownloadError, "Content-Range"):
            self.run_download(opener=BadRange(SOURCE_BYTES))
        self.assertFalse(self.output.exists())

    def test_wrong_source_size_in_content_range_is_rejected(self):
        class WrongTotal(MemorySource):
            def __call__(self, url, start, end, deadline, clock):
                response = RangeResponse(
                    self.source[start:end + 1], start, end, len(self.source) + 1
                )
                return response

        with self.assertRaisesRegex(downloader.DownloadError, "Content-Range"):
            self.run_download(opener=WrongTotal(SOURCE_BYTES))
        self.assertFalse(self.output.exists())


if __name__ == "__main__":
    unittest.main()
