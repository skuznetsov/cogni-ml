"""CPU-only falsifiers for the terminal QBit quality comparison."""
from __future__ import annotations

import json
from pathlib import Path
import struct
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).parent / "support"))
import hy_nomas_cache_qbit_compare as comparison
import hy_nomas_sequence_resume_reference_test as fixture


class CacheQBitComparisonTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory(prefix="hy-nomas-qbit-compare-")
        self.addCleanup(self.temp.cleanup)
        directories = fixture._fixture(Path(self.temp.name))
        self.control, self.qbit = directories["control"], directories["resume"]
        self.count = (4320 + 557) * 16 * 128
        rows = self.count // 128
        records = [{"name": f"history.layer{i}.{kind}", "count": self.count, "rows": rows,
                    "logical_packed_bytes": rows * 96, "exact_bytes": self.count * 4,
                    "invalid_rows": 0, "tier_counts": {"p4": 0, "p5": rows, "p8": 0, "f32": 0}}
                   for i in range(54) for kind in ("key", "value")]
        state_sha = json.loads((self.qbit / "receipt.json").read_text())["state_input_sha256"]
        for directory, treatment in ((self.control, "exact"), (self.qbit, "qbit")):
            def edit(receipt):
                receipt.update(mode="resume", cache_roundtrip=treatment, state_input_sha256=state_sha,
                               valid_text_tokens=557, codec_bytes=0 if treatment == "exact" else 1000000, codec=None)
                if treatment == "qbit":
                    receipt["peak_named_bytes"] += 1000000
                    receipt["codec"] = {"resident_bytes": 1000000, "invalid_rows": 0, "records": records,
                                        "schema": "hy-nomas-cache-qbit-report-v1",
                                        "policy": dict(comparison.POLICY),
                                        "codebook": {"sha256": comparison.CODEBOOK_SHA256},
                                        "logical_packed_bytes": 108 * rows * 96,
                                        "exact_bytes": 108 * self.count * 4,
                                        "tier_counts": {"p4": 0, "p5": 108 * rows, "p8": 0, "f32": 0}}
            fixture._edit_receipt(directory, edit)
        self.change("timeline", 30 * 9 * 16, 0.01)
        self.change("b-step1", 0, 0.126)

    def change(self, name: str, element: int, value: float) -> None:
        path = self.qbit / f"{name}.f32le"
        with path.open("r+b") as stream:
            stream.seek(element * 4)
            stream.write(struct.pack("<f", value))
        fixture._refresh_artifact_sha(self.qbit, name)

    def compare(self) -> dict:
        return comparison.compare_runs(str(self.control), str(self.qbit))

    def test_reports_lossy_error_without_promoting_quality(self):
        result = self.compare()
        self.assertTrue(result["pair_integrity_passed"])
        self.assertFalse(result["b_latent"]["within_tolerance"])
        self.assertIn("pending visual review", result["quality_verdict"])

    def test_rejects_changed_a(self):
        self.change("timeline", 0, 1.0)
        with self.assertRaisesRegex(ValueError, "modified A"):
            self.compare()

    def test_rejects_changed_future_noise(self):
        self.change("timeline", 60 * 9 * 16, 1.0)
        with self.assertRaisesRegex(ValueError, "future noise"):
            self.compare()

    def test_rejects_mismatched_identity(self):
        fixture._edit_receipt(self.qbit, lambda r: r.update(inputs_sha256="a" * 64))
        with self.assertRaisesRegex(ValueError, "identity mismatch"):
            self.compare()

    def test_rejects_stale_selector(self):
        fixture._edit_receipt(self.qbit, lambda r: r["codec"]["policy"].update(p4_max_error_over_sigma=0.60))
        with self.assertRaisesRegex(ValueError, "selector policy mismatch"):
            self.compare()

    def test_rejects_stale_codebook(self):
        fixture._edit_receipt(self.qbit, lambda r: r["codec"]["codebook"].update(sha256="a" * 64))
        with self.assertRaisesRegex(ValueError, "codebook mismatch"):
            self.compare()

    def test_rejects_nonfinite_data(self):
        self.change("b-step1", 0, float("nan"))
        with self.assertRaisesRegex(ValueError, "non-finite"):
            self.compare()

    def test_rejects_underreported_header_bytes(self):
        fixture._edit_receipt(self.qbit, lambda r: r["codec"]["records"][0].update(logical_packed_bytes=0))
        with self.assertRaisesRegex(ValueError, "byte accounting"):
            self.compare()

    def test_rejects_inactive_treatment(self):
        self.change("timeline", 30 * 9 * 16, 0.0)
        self.change("b-step1", 0, 0.125)
        with self.assertRaisesRegex(ValueError, "treatment is inactive"):
            self.compare()


if __name__ == "__main__":
    unittest.main()
