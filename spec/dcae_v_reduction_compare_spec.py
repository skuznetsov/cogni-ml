"""CPU-only falsifiers for the GPU reduction diagnostic's rounding and gate."""
import importlib.util
import json
from fractions import Fraction
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


SOURCE = Path(__file__).resolve().parents[1] / "examples/support/dcae_v_reduction_compare.py"
SPEC = importlib.util.spec_from_file_location("dcae_v_reduction_compare", SOURCE)
compare = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(compare)


class ExactRoundSpec(unittest.TestCase):
    def test_midpoint_even_and_direct_bf16_avoids_f32_double_round(self):
        midpoint = Fraction(1) + Fraction(1, 256)
        epsilon = Fraction(1, 2**266)
        self.assertEqual(compare.exact_bits(midpoint, 8), 0x3f80)
        self.assertEqual(compare.exact_bits(midpoint + epsilon, 8), 0x3f81)
        self.assertEqual(compare.exact_bits(midpoint - epsilon, 8), 0x3f80)
        self.assertEqual(compare.exact_bits(midpoint + epsilon, 24), 0x3f808000)
        self.assertEqual(compare.exact_bits(Fraction(259, 256), 8), 0x3f82)

    def test_subnormal_min_normal_carry_overflow_and_cancellation(self):
        self.assertEqual(compare.exact_bits(Fraction(1, 2**134), 8), 0)
        self.assertEqual(compare.exact_bits(Fraction(3, 2**134), 8), 2)
        self.assertEqual(compare.exact_bits(Fraction(255, 2**134), 8), 0x80)
        self.assertEqual(compare.exact_bits(Fraction(-1, 2**150), 24), 0x80000000)
        self.assertEqual(compare.exact_bits(Fraction(3, 2**150), 24), 2)
        self.assertEqual(compare.exact_bits(Fraction(1 << 128), 8), 0x7f80)
        self.assertEqual(compare.exact_bits(Fraction(-(1 << 128)), 24), 0xff800000)
        self.assertEqual(compare.exact_bits(Fraction(), 8), 0)

    def test_nonfinite_evidence_is_a_reportable_failure(self):
        result = compare.metrics([0x7fc0], [0], [0])
        self.assertEqual(result["tolerance_failures"], 1)
        self.assertEqual(result["nonfinite"], 1)
        json.dumps(result, allow_nan=False)


class CandidateGateSpec(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="dcae-reduction-gate-")
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        vectors = self.root / "vectors"
        compare.make_vectors(vectors)
        expected = json.loads((vectors / "expected.json").read_text())
        compare.write_raw(self.root / "vectors.bf16", "H", expected["bf16"])
        compare.write_raw(self.root / "vectors.f32", "I", expected["f32"])
        manifest = {"schema": "dcae_conv3d_reduction_probe/v1", "samples": 8192,
                    "uniform_samples": 4096, "shape": [1, 512, 40, 3, 5],
                    "vectors_directory": str(vectors), "terminal_waits": 1,
                    "hot_metrics": {"allocations": 0, "readbacks": 0, "preterminal_waits": 0},
                    "sampling": "gate fixture only", "full_stats": list("abcdef"),
                    "candidates": ["order0-lanes1-tree0"]}
        (self.root / "manifest.json").write_text(json.dumps(manifest))
        for case in ("invalid", "valid"):
            compare.write_raw(self.root / f"{case}-indices.u32", "I", range(8192))
            compare.write_raw(self.root / f"{case}-full-stats.u32", "I", [0] * 6)
            for name in ("mps-pre.bf16", "mps-post.bf16", "original.bf16",
                         "order0-lanes1-tree0.pre.bf16", "order0-lanes1-tree0.post.bf16"):
                compare.write_raw(self.root / f"{case}-{name}", "H", [0] * 8192)
            for name in ("mps-f32.f32", "order0-lanes1-tree0.f32"):
                compare.write_raw(self.root / f"{case}-{name}", "f", [0] * 8192)

    def test_valid_control_and_required_candidate_pass(self):
        result, passed = compare.compare(self.root, "order0-lanes1-tree0", False, False)
        self.assertTrue(passed)
        self.assertTrue(result["required_candidate_passed"])

    def test_control_pass_cannot_hide_candidate_failure_or_cli_exit(self):
        path = self.root / "valid-order0-lanes1-tree0.pre.bf16"
        values = bytearray(path.read_bytes())
        values[:2] = b"\x80\x3f"
        path.write_bytes(values)
        result, passed = compare.compare(self.root, "order0-lanes1-tree0", False, True)
        self.assertTrue(result["controls_passed"])
        self.assertFalse(passed)
        run = subprocess.run([sys.executable, "-B", str(SOURCE), str(self.root),
                              "--require-candidate", "order0-lanes1-tree0", "--positive-control"],
                             capture_output=True, text=True)
        self.assertEqual(run.returncode, 1, run.stderr)
        self.assertFalse(json.loads(run.stdout)["gate_passed"])

    def test_missing_upstream_or_hot_readback_blocks_gate(self):
        self.assertFalse(compare.compare(self.root, None, True, False)[1])
        manifest = json.loads((self.root / "manifest.json").read_text())
        manifest["hot_metrics"]["readbacks"] = 1
        (self.root / "manifest.json").write_text(json.dumps(manifest))
        self.assertFalse(compare.compare(self.root, "order0-lanes1-tree0", False, False)[1])

    def test_biased_upstream_call_must_match_the_exposed_boundary(self):
        for case in ("invalid", "valid"):
            for name in ("pre", "post", "biased"):
                compare.write_raw(self.root / f"torch-{case}-{name}.bf16", "H", [0] * 8192)
            compare.write_raw(self.root / f"torch-{case}-indices.u32", "I", range(8192))
        self.assertTrue(compare.compare(self.root, "order0-lanes1-tree0", True, False)[1])
        path = self.root / "torch-valid-biased.bf16"
        values = bytearray(path.read_bytes())
        values[:2] = b"\x80\x3f"
        path.write_bytes(values)
        result, passed = compare.compare(self.root, "order0-lanes1-tree0", True, False)
        self.assertFalse(passed)
        self.assertFalse(result["controls_passed"])
        self.assertEqual(result["cases"]["valid"]["upstream_biased_call_vs_separate"]["bit_mismatches"], 1)

    def test_sampled_f32_rounding_cannot_disagree_with_zero_full_counter(self):
        path = self.root / "invalid-mps-f32.f32"
        values = bytearray(path.read_bytes())
        values[:4] = b"\x00\x00\x80\x3f"
        path.write_bytes(values)
        result, passed = compare.compare(self.root, "order0-lanes1-tree0", False, False)
        self.assertFalse(passed)
        self.assertFalse(result["controls_passed"])
        self.assertEqual(result["cases"]["invalid"]["sampled_mps_f32_round_vs_bf16_pre"]["bit_mismatches"], 1)

    def test_control_only_observation_never_passes_candidate_gate(self):
        result, passed = compare.compare(self.root, None, False, False)
        self.assertTrue(result["controls_passed"])
        self.assertFalse(result["gate_passed"])
        self.assertFalse(passed)
        run = subprocess.run([sys.executable, "-B", str(SOURCE), str(self.root), "--observe-only"],
                             capture_output=True, text=True)
        self.assertEqual(run.returncode, 0, run.stderr)
        self.assertEqual(json.loads(run.stdout)["mode"], "observation-only")
        self.assertFalse(json.loads(run.stdout)["gate_passed"])


if __name__ == "__main__":
    unittest.main()
