"""CPU publisher-to-qualifier contract test for the source render artifact."""
from contextlib import redirect_stderr
import io
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "spec" / "support"))
sys.path.insert(0, str(ROOT / "examples" / "support"))

import hy_nomas_soft_route_render_reference as source  # noqa: E402
import hy_nomas_soft_route_render_qualify as qualifier  # noqa: E402


class SourceRenderPublisherTests(unittest.TestCase):
    """Exercise real serialization and qualification with schema-only fixtures."""

    def test_output_preflight_rejects_aliases_and_collisions_before_admission(self):
        with tempfile.TemporaryDirectory(prefix="soft-route-output-preflight-") as temp:
            root = Path(temp).resolve()
            canonical_parent = root / "canonical"
            canonical_parent.mkdir()
            alias_parent = root / "alias"
            alias_parent.symlink_to(canonical_parent, target_is_directory=True)

            reports = [alias_parent / "aliased.json"]
            for existing_name in ("collision.json", "collision.f32le"):
                case_parent = root / existing_name.replace(".", "-")
                case_parent.mkdir()
                report = case_parent / "collision.json"
                (case_parent / existing_name).write_bytes(b"occupied")
                reports.append(report)

            with patch.object(source, "authenticate_inputs") as authenticate, \
                    patch.object(source.direct, "authenticate_model_sources") as authenticate_model:
                for report in reports:
                    stderr = io.StringIO()
                    with redirect_stderr(stderr):
                        status = source.main(["--execute", "--report", str(report)])
                    self.assertEqual(status, 2)
                    self.assertIn('"status": "rejected"', stderr.getvalue())
                    authenticate.assert_not_called()
                    authenticate_model.assert_not_called()

            race_parent = root / "publish-recheck"
            race_parent.mkdir()
            race_report = race_parent / "render.json"
            source._fresh_render_paths(race_report)
            (race_parent / "render.f32le").write_bytes(b"claimed after preflight")
            with self.assertRaisesRegex(source.RenderReferenceError, "render report and payload must be new files"):
                source._publish(race_report, {}, {}, {}, {})

    def test_legacy_runtime_geometry_is_configured_and_fail_closed(self):
        legacy = SimpleNamespace(DIT_TEXT_ROWS=576, CACHE_CAPACITY=4896)
        with self.assertRaisesRegex(source.RenderReferenceError, "legacy source cache geometry mismatch"):
            source._require_legacy_cache_geometry(legacy)

        expected = {
            "DIT_TEXT_ROWS": source.direct.TEXT_STORAGE_ROWS,
            "CACHE_CAPACITY": source.direct.PHYSICAL_CACHE_ROWS,
        }
        self.assertEqual(expected, {"DIT_TEXT_ROWS": 320, "CACHE_CAPACITY": 4640})
        self.assertEqual(source._configure_legacy_cache_geometry(legacy), expected)
        self.assertEqual(source._require_legacy_cache_geometry(legacy), expected)

        legacy.CACHE_CAPACITY = 4896
        with self.assertRaisesRegex(source.RenderReferenceError, "legacy source cache geometry mismatch"):
            source._require_legacy_cache_geometry(legacy)

    def fixture(self):
        model = {
            "upstream_path": "/fixture/pinned-upstream",
            "upstream_revision": source.direct.PINNED_REVISION,
            "source_hashes": {"fixture.py": "a" * 64},
            "cache_index_path": "/fixture/cache-index.json",
            "cache_shards": [
                {"path": f"/fixture/block-{index}.safetensors", "bytes": 1,
                 "sha256": f"{index + 1:064x}", "tensor_count": 32}
                for index in range(source.direct.BLOCKS)
            ],
            "global_shard": {"path": "/fixture/global.safetensors", "bytes": 1,
                             "sha256": "b" * 64, "tensor_count": 1},
            "total_weight_bytes": source.direct.BLOCKS + 1,
        }

        initial = {"path": "/fixture/initial-latent.f32le", "shape": list(source.LATENT_SHAPE),
                   "bytes": source.LATENT_BYTES, "sha256": source.PARENT_INITIAL_SHA256,
                   "source_range_offset": 17, "source_range_bytes": source.LATENT_BYTES}
        admitted = {
            "inputs": {"t1.initial_latent": initial},
            "frozen_inputs_path": "/fixture/frozen-inputs.json",
            "frozen_inputs_sha256": "c" * 64,
            "roi_manifest_path": "/fixture/roi.json",
            "roi_manifest_sha256": source.ROI_SOURCE_SHA256,
            "roi_payload_path": "/fixture/roi.f32le",
            "roi_payload_sha256": "d" * 64,
        }
        checked = {
            "parent": {"path": "/fixture/direct-parent.json", "sha256": source.soft.PARENT_SHA256,
                       "payload_path": "/fixture/direct-parent.f32le",
                       "document": {"payload": {"bytes": 1, "sha256": "e" * 64}}},
            "admitted": admitted,
            "parent_initial": initial,
            "anchor_path": "/fixture/t1-anchor.json", "anchor_sha256": source.ANCHOR_SHA256,
            "anchor_document": {"payload_bytes": 1},
            "anchor_payload_path": "/fixture/t1-anchor.f32le",
            "anchor_payload_sha256": source.ANCHOR_PAYLOAD_SHA256,
            "anchor_qwen_record": {"dtype": "BF16", "shape": list(source.QWEN_SHAPE),
                                    "offset": source.LATENT_BYTES, "nbytes": source.QWEN_BYTES,
                                    "sha256": source.ANCHOR_QWEN_SHA256},
            "anchor_unused_latent_record": {"dtype": "F32", "shape": list(source.LATENT_SHAPE),
                                             "offset": 0, "nbytes": source.LATENT_BYTES,
                                             "sha256": source.ANCHOR_UNUSED_LATENT_SHA256},
            "bias": {"path": "/fixture/bias.json", "sha256": source.soft.BIAS_SHA256,
                     "document": {"reference_image": "fixture.png", "maps": {},
                                  "bias_f32le_sha256": "f" * 64}},
            "noise_selection_path": "/fixture/noise-selection.json",
            "noise_selection": {"cpu_comparison": {}},
            "noise_recomputed": {"anchor_raw_vs_parent_bit_differences": 138_240,
                                 "anchor_bf16_rne_vs_parent_bit_differences": 138_117},
            "zero_step4_path": "/fixture/accepted-step4.f32le",
            "zero_step4_sha256": source.ACCEPTED_ZERO_STEP4_SHA256,
            "zero_step4_bytes": source.LATENT_BYTES,
        }
        captures = {
            name: np.full(shape, index, dtype="<f4")
            for index, (name, shape) in enumerate(source.TENSOR_SHAPES.items(), start=1)
        }
        receipt = {
            "case_order": list(source.CASE_ORDER), "steps": 4, "block_calls": 864,
            "t0_writer_block_calls": 216, "t1_arm_block_calls": 648, "refiner_calls": 4,
            "final_head_calls": 12, "euler_updates": 12, "attention_calls": 864,
            "t0_attention_calls": 216, "t1_attention_calls": 648,
            "attention_query_chunks": 15_552, "explicit_terminal_mps_synchronize_count": 1,
            "terminal_export_count": 1, "intermediate_readbacks": 0, "captures": 18,
        }
        return model, checked, captures, receipt

    def test_publish_roundtrips_through_root_qualifier(self):
        model, checked, captures, receipt = self.fixture()
        with tempfile.TemporaryDirectory(prefix="soft-route-source-publisher-") as temp:
            report = Path(temp).resolve() / "source.json"
            # This test covers the serialization interface, not upstream provenance or
            # replay math; those expensive checks are represented by explicit fixtures.
            gates = {"zero.step4.latent": {"bitwise_equal": True}}
            with patch.object(source, "_check_source_replay", return_value=gates):
                manifest, payload, document = source._publish(report, checked, model, captures, receipt)

            manifest_sha = source._sha256_file(manifest)
            loaded = qualifier.load_render(manifest, manifest_sha, "source")
            qualifier.validate_execution(document, "source")

            self.assertEqual(document["case_order"], list(source.CASE_ORDER))
            self.assertEqual(set(loaded["arrays"]), set(source.TENSOR_ORDER))
            self.assertEqual(document["payload"]["bytes"], 45_527_040)
            self.assertEqual(payload.stat().st_size, 45_527_040)
            cursor = 0
            for index, name in enumerate(source.TENSOR_ORDER, start=1):
                record = document["tensors"][name]
                self.assertEqual(record["offset_bytes"], cursor)
                self.assertEqual(record["checkpoint_boundary"], name.split(".", 1)[1])
                self.assertEqual(record["nbytes"], int(np.prod(source.TENSOR_SHAPES[name])) * 4)
                self.assertTrue(np.all(loaded["arrays"][name] == index))
                cursor += record["nbytes"]
            self.assertEqual(cursor, payload.stat().st_size)


if __name__ == "__main__":
    unittest.main()
