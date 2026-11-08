"""CPU-only falsifiers for the GPU reduction diagnostic's rounding and gate."""
import importlib.util
import json
from fractions import Fraction
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch


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

    def prefix_manifest(self):
        manifest = json.loads((self.root / "manifest.json").read_text())
        prefix = {"reference_path": "/pinned/reference.safetensors", "reference_sha256": "a" * 64,
                  "target_conv": compare.PREFIX_TARGET, "tile_count": 2,
                  "activation_shape": [1, 512, 40, 72, 128], "cache_shape": [1, 512, 1, 72, 128]}
        manifest.update(activation_prefix=prefix, shape=prefix["activation_shape"], weights_sha256="b" * 64)
        return manifest

    def prefix_report(self, manifest):
        report = dict(manifest["activation_prefix"], weights_sha256=manifest["weights_sha256"],
                      upstream_sha256=compare.PREFIX_UPSTREAM_SHA,
                      schema="cogni_ml.dcae_v_activation_prefix_upstream/v1",
                      checkpoint={"sha256": compare.PREFIX_CHECKPOINT_SHA},
                      upstream={"revision": compare.PREFIX_UPSTREAM_REVISION,
                                "module_sha256": compare.PREFIX_UPSTREAM_SHA},
                      native_weight_cache={"sha256": manifest["weights_sha256"],
                                           "selected_prefix_tensor_count": 90,
                                           "selected_prefix_bytes_bitwise_joined_to_original_checkpoint": 2096936960},
                      reference={"sha256": manifest["activation_prefix"]["reference_sha256"],
                                 "tensor": "input.latent", "tensor_dtype": "BF16",
                                 "tensor_shape": [1, 32, 20, 9, 16], "tensor_sha256": compare.PREFIX_INPUT_SHA,
                                 "lineage_audit_sha256": compare.PREFIX_LINEAGE_SHA,
                                 "full_source_reference_sha256": compare.PREFIX_SOURCE_REFERENCE_SHA},
                      prefix_weight_load={"materialized_tensor_count": 90,
                                          "materialized_bf16_bytes": 2096936960,
                                          "encoder_weights_materialized": False,
                                          "later_decoder_weights_materialized": False,
                                          "target_conv_weights_materialized": False}, artifacts={})
        for case in ("invalid", "valid"):
            for suffix, shape in (("input", report["activation_shape"]), ("cache", report["cache_shape"])):
                report["artifacts"][f"{case}-{suffix}.bf16"] = {
                    "dtype": "BF16", "shape": shape, "sha256": "same", "bytes": compare.math.prod(shape) * 2}
        return report

    def test_real_activation_without_independent_prefix_cannot_pass_candidate_gate(self):
        (self.root / "manifest.json").write_text(json.dumps(self.prefix_manifest()))
        result, passed = compare.compare(self.root, "order0-lanes1-tree0", False, False)
        self.assertTrue(result["required_candidate_passed"])
        self.assertFalse(result["controls_passed"])
        self.assertFalse(passed)

    def test_full_prefix_wrong_provenance_or_old_cache_rejects_even_with_passing_samples(self):
        manifest = self.prefix_manifest()
        oracle = self.root / "prefix-oracle"
        oracle.mkdir()
        report = self.prefix_report(manifest)
        (oracle / "upstream-prefix.json").write_text(json.dumps(report))
        transport = {"cold_cache_all_zero": True, "cached_input_is_previous_activation_tail": True}
        with patch.object(compare, "file_digest", return_value="same"), patch.object(compare, "causal_cache_invariant", return_value=transport):
            self.assertTrue(compare.authenticate_prefix(self.root, manifest, oracle)[1])
            report["reference_sha256"] = "c" * 64
            (oracle / "upstream-prefix.json").write_text(json.dumps(report))
            self.assertFalse(compare.authenticate_prefix(self.root, manifest, oracle)[1])
        report["reference_sha256"] = "a" * 64
        (oracle / "upstream-prefix.json").write_text(json.dumps(report))
        with patch.object(compare, "file_digest", side_effect=lambda p, _: "changed" if p == oracle / "valid-cache.bf16" else "same"), patch.object(compare, "causal_cache_invariant", return_value=transport):
            result, passed = compare.authenticate_prefix(self.root, manifest, oracle)
            self.assertFalse(passed)
            self.assertFalse(result["files"]["valid-cache.bf16"]["bitwise_equal"])

    def test_nested_lineage_and_artifact_claims_cannot_be_hidden_by_equal_bytes(self):
        manifest = self.prefix_manifest()
        oracle = self.root / "prefix-oracle"
        oracle.mkdir()
        mutations = [
            lambda r: r.update(schema="unknown/v1"),
            lambda r: r["checkpoint"].update(sha256="0" * 64),
            lambda r: r["upstream"].update(revision="0" * 40),
            lambda r: r["native_weight_cache"].update(selected_prefix_tensor_count=89),
            lambda r: r["reference"].update(lineage_audit_sha256="0" * 64),
            lambda r: r["prefix_weight_load"].update(later_decoder_weights_materialized=True),
            lambda r: r["artifacts"]["valid-cache.bf16"].update(sha256="0" * 64),
        ]
        transport = {"cold_cache_all_zero": True, "cached_input_is_previous_activation_tail": True}
        for i, mutate in enumerate(mutations):
            with self.subTest(mutation=i):
                report = self.prefix_report(manifest)
                mutate(report)
                (oracle / "upstream-prefix.json").write_text(json.dumps(report))
                with patch.object(compare, "file_digest", return_value="same"), patch.object(compare, "causal_cache_invariant", return_value=transport):
                    result, passed = compare.authenticate_prefix(self.root, manifest, oracle)
                    self.assertFalse(passed)
                    self.assertTrue(all(v["bitwise_equal"] for v in result["files"].values()))

    def test_prefix_extent_and_unsupported_capture_boundary_are_rejected(self):
        path = self.root / "tiny.bf16"
        path.write_bytes(b"\x80\x3f")
        with self.assertRaisesRegex(ValueError, "extent"):
            compare.file_digest(path, 4)
        manifest = self.prefix_manifest()
        manifest["activation_prefix"]["target_conv"] = "decoder.stages.2.op_list.2.main.conv1.conv"
        with self.assertRaisesRegex(ValueError, "boundary"):
            compare.authenticate_prefix(self.root, manifest, None)

    def test_old_cache_is_previous_ncthw_tail_not_overwritten_current_tail(self):
        compare.write_raw(self.root / "invalid-input.bf16", "H", [1, 2, 3, 4, 5, 6])
        compare.write_raw(self.root / "invalid-cache.bf16", "H", [0, 0])
        compare.write_raw(self.root / "valid-cache.bf16", "H", [3, 6])
        self.assertTrue(compare.causal_cache_invariant(self.root, [1, 2, 3, 1, 1])["cached_input_is_previous_activation_tail"])
        (self.root / "valid-cache.bf16").write_bytes(b"\x01\x00\x04\x00")
        self.assertFalse(compare.causal_cache_invariant(self.root, [1, 2, 3, 1, 1])["cached_input_is_previous_activation_tail"])


if __name__ == "__main__":
    unittest.main()
