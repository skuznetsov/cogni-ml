#!/usr/bin/env python3
"""CPU semantic oracle checks for Hy-Nomas action attention masks."""

from __future__ import annotations

import json
from pathlib import Path
import sys
import tempfile
import unittest

import torch


REPOSITORY = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY / "spec" / "support"))
import hy_nomas_action_attention_reference as reference  # noqa: E402


UPSTREAM_SOURCE = Path(
    "/opt/mldata_ssd/cogni-ml/hy-nomas-reference-handoff-20261101.pDnd7TV1"
    "/upstream-a31/hymm/models/dit/nomas_cpt.py"
)


class HyNomasActionAttentionReferenceSpec(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.source = reference.validate_upstream_source(UPSTREAM_SOURCE)
        cls.fixture = reference.make_fixture(cls.source)

    def test_source_range_builder_matches_native_label_matrix(self) -> None:
        fixture = self.fixture
        self.assertEqual(len(fixture.query_labels), reference.QUERY_ROWS)
        self.assertEqual(len(fixture.key_labels), reference.KEY_ROWS)
        self.assertEqual(len(fixture.q_ranges), 25)
        self.assertTrue(torch_equal(fixture.masked_mask, reference.build_label_mask(
            fixture.query_labels, fixture.key_labels
        )))

        global_end = reference.HISTORY_ROWS + reference.IMAGE_ROWS + reference.OFFSETS[0]
        action0_end = reference.HISTORY_ROWS + reference.IMAGE_ROWS + reference.OFFSETS[1]
        action1_end = reference.HISTORY_ROWS + reference.IMAGE_ROWS + reference.OFFSETS[2]
        self.assertGreater(reference.QUERY_ROWS, 256)
        self.assertEqual(fixture.query_labels[256:], [2, -1, -1])
        # Video action 0 sees its own text and all global keys, but never action 1.
        self.assertTrue(all(fixture.masked_mask[0, key] for key in range(action0_end)))
        self.assertFalse(any(fixture.masked_mask[0, key] for key in range(action0_end, reference.KEY_ROWS)))
        # The gap has global-only visibility, represented as query label 0.
        self.assertEqual(fixture.query_labels[3], 0)
        self.assertTrue(all(fixture.masked_mask[3, key] for key in range(global_end)))
        self.assertFalse(any(fixture.masked_mask[3, key] for key in range(global_end, reference.KEY_ROWS)))
        # The second action has its own keys and cannot see the first action's tail.
        self.assertEqual(fixture.query_labels[4], 2)
        self.assertFalse(any(fixture.masked_mask[4, key] for key in range(global_end, action0_end)))
        self.assertTrue(all(fixture.masked_mask[4, key] for key in range(action0_end, action1_end)))
        # Padded queries produce no visible keys; padded keys are never visible.
        self.assertEqual(fixture.query_labels[-2:], [-1, -1])
        self.assertEqual(fixture.key_labels[-2:], [-1, -1])
        self.assertFalse(bool(fixture.masked_mask[-2:, :].any()))
        self.assertFalse(bool(fixture.masked_mask[:, -2:].any()))

    def test_cpu_attention_sentinel_separates_masked_and_legacy(self) -> None:
        fixture = self.fixture
        self.assertTrue(bool(torch_isfinite(fixture.expected_masked).all()))
        self.assertTrue(bool(torch_isfinite(fixture.expected_legacy).all()))
        self.assertEqual(float(fixture.expected_masked[:, -2:, :, :].abs().max()), 0.0)
        self.assertEqual(float(fixture.expected_legacy[:, -2:, :, :].abs().max()), 0.0)
        delta = (fixture.expected_masked - fixture.expected_legacy).abs()
        self.assertGreater(float(delta[:, : reference.IMAGE_ROWS, :, :].max()), 1.0)
        # Two action value sentinels have opposite signs. A crossed action mask
        # would move the affected rows by far more than the native comparison tolerance.
        self.assertGreater(float(delta.max()), 1.0)
        self.assertEqual(reference.ATOL, 2.0e-5)

        crossed_query_labels = [
            3 - label if label in (1, 2) else label for label in fixture.query_labels
        ]
        crossed_mask = reference.build_label_mask(crossed_query_labels, fixture.key_labels)
        keys = torch.cat((fixture.history_key, fixture.current_key), dim=1)
        values = torch.cat((fixture.history_value, fixture.current_value), dim=1)
        crossed = reference.attention_f32(fixture.query, keys, values, crossed_mask)
        crossed_error = (crossed - fixture.expected_masked).abs()
        self.assertGreater(float(crossed_error.max()), 1.0)

    def test_fixture_manifest_authenticates_cpu_f32_inputs_and_outputs(self) -> None:
        with tempfile.TemporaryDirectory(prefix="hy-nomas-action-oracle-") as scratch:
            output = Path(scratch) / "fixture"
            manifest_path, manifest_sha = reference.write_fixture(output, UPSTREAM_SOURCE)
            raw_manifest = manifest_path.read_bytes()
            self.assertEqual(reference.sha256(raw_manifest), manifest_sha)
            manifest = json.loads(raw_manifest)
            self.assertEqual(
                manifest["upstream"]["source_file_sha256"], reference.UPSTREAM_SOURCE_SHA256
            )
            self.assertEqual(manifest["oracle"]["device"], "cpu")
            self.assertEqual(manifest["oracle"]["dtype"], "torch.float32")
            self.assertEqual(manifest["geometry"]["history_rows"], 3)
            self.assertEqual(manifest["geometry"]["query_rows"], reference.QUERY_ROWS)
            self.assertEqual(manifest["mask"]["query_labels"][3], 0)
            for descriptor in manifest["tensors"].values():
                payload = (output / descriptor["file"]).read_bytes()
                self.assertEqual(len(payload), descriptor["nbytes"])
                self.assertEqual(reference.sha256(payload), descriptor["sha256"])


def torch_equal(left, right) -> bool:
    return bool(torch.equal(left, right))


def torch_isfinite(value):
    return torch.isfinite(value)


if __name__ == "__main__":
    unittest.main()
