#!/usr/bin/env python3
"""CPU-only file identity falsifiers for Cinematic reference qualification."""

from __future__ import annotations

import os
from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "spec" / "support"))

from hy_nomas_cinematic_reference import (  # noqa: E402
    FileIdentityLedger,
    QualificationError,
    _write_report,
)


class CinematicReferenceIdentityTests(unittest.TestCase):
    def test_rejects_same_size_path_replacement_even_when_mtime_is_preserved(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            pinned = root / "pinned.bin"
            replacement = root / "replacement.bin"
            pinned.write_bytes(b"adapter")
            ledger = FileIdentityLedger()
            original = ledger.capture(pinned, "adapter")

            replacement.write_bytes(b"changed")
            os.utime(replacement, ns=(original.st_atime_ns, original.st_mtime_ns))
            os.replace(replacement, pinned)

            with self.assertRaisesRegex(QualificationError, "identity changed"):
                ledger.assert_current(pinned, "adapter")

    def test_rejects_in_place_mutation_before_report_publication(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            pinned = root / "adapter-config.json"
            pinned.write_bytes(b'{"alpha":2}')
            ledger = FileIdentityLedger()
            original = ledger.capture(pinned, "adapter config")
            pinned.write_bytes(b'{"alpha":3}')
            os.utime(pinned, ns=(original.st_atime_ns, original.st_mtime_ns + 1_000_000))
            report = root / "qualification.json"

            with self.assertRaisesRegex(QualificationError, "identity changed"):
                _write_report(report, {"status": "qualified"}, ledger=ledger)
            self.assertFalse(report.exists())

    def test_removes_report_if_input_changes_during_publication_check(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            pinned = root / "base-index.json"
            pinned.write_bytes(b"base")
            ledger = FileIdentityLedger()
            ledger.capture(pinned, "base index")

            class MutateOnPrepublishLedger(FileIdentityLedger):
                def assert_all_current(self, phase: str) -> None:
                    if phase == "during qualification report publication":
                        pinned.write_bytes(b"changed")
                    super().assert_all_current(phase)

            racing_ledger = MutateOnPrepublishLedger()
            racing_ledger.capture(pinned, "base index")
            report = root / "qualification.json"
            with self.assertRaisesRegex(QualificationError, "identity changed"):
                _write_report(report, {"status": "qualified"}, ledger=racing_ledger)
            self.assertFalse(report.exists())

    def test_rejects_symlink_at_authenticated_file_path(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            target = root / "target"
            link = root / "adapter.safetensors"
            target.write_bytes(b"adapter")
            link.symlink_to(target)
            with self.assertRaisesRegex(QualificationError, "regular non-symlink"):
                FileIdentityLedger().capture(link, "adapter")


if __name__ == "__main__":
    unittest.main()
