"""CPU-only contract tests for the bounded PixelUMM global-head oracle."""
from __future__ import annotations

import hashlib
import sys
from contextlib import nullcontext
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).parent / "support"))
import hy_nomas_pixelumm_checkpoint as checkpoint
import hy_nomas_pixelumm_head_checkpoint as head_checkpoint
import hy_nomas_pixelumm_head_reference as head_reference
import hy_nomas_pixelumm_reference as block_reference


def _minimal_head_header():
    return {
        name: {
            "dtype": "F32", "shape": list(shape), "data_offsets": list(offsets),
        }
        for name, (shape, offsets) in head_checkpoint.HEAD_TENSORS.items()
    }


def _full36_manifest(seed=42):
    pinned = head_reference._FULL36_INPUTS[seed]
    captures = {
        "block35.output": {
            "dtype": "BF16", "shape": [1, head_reference.TOKENS, head_reference.HIDDEN],
            "offset": pinned["record_offset"],
            "nbytes": head_reference.HEAD_INPUT_NBYTES,
            "sha256": pinned["record_sha256"],
        },
    }
    blocks = []
    for index in range(36):
        stages = block_reference._full36_capture_stages(index)
        blocks.append({
            "index": index,
            "capture_keys": [f"block{index}.{stage}" for stage in stages],
        })
    return {
        "schema": "cogni_ml.pixelumm_thirty_six_block_reference/v1",
        "upstream_revision": head_reference.UPSTREAM_REVISION,
        "upstream_model_path": head_reference.UPSTREAM_MODEL_PATH,
        "upstream_model_sha256": head_reference.UPSTREAM_MODEL_SHA256,
        "checkpoint_header_sha256": checkpoint.HEADER_SHA256,
        "grid": list(head_reference.GRID),
        "tile_tokens": head_reference.TILE_TOKENS,
        "token_order": "tile_major",
        "captures_are_tile_major": True,
        "runtime_dtype": "BF16",
        "full36_capture_policy": "two_bank_survivors",
        "permutation_sha256": "4e81c6026e99ee2c17c176c4394682ca632602c198c3065ad311a83a3839cf33",
        "payload": pinned["payload"],
        "payload_bytes": pinned["payload_bytes"],
        "payload_sha256": pinned["payload_sha256"],
        "comparison": {"atol": 2e-5, "rtol": 2e-5},
        "input_provenance": {"kind": "real_latent_xt_pixelumm_stem"},
        "input_lineage": {},
        "capture_order": list(block_reference._full36_capture_order()),
        "blocks": blocks,
        "captures": captures,
    }


class PixelUMMHeadCheckpointTests(unittest.TestCase):
    def test_exact_four_key_range_and_runtime_weight_size(self):
        encoded, start, end, keys = head_checkpoint.head_subset_header(_minimal_head_header())
        self.assertEqual((start, end), (52_510_720, 102_887_424))
        self.assertEqual(end - start, 50_376_704)
        self.assertEqual(head_checkpoint.HEAD_BF16_BYTES, 25_188_352)
        self.assertEqual(keys, (
            "decoder.final_norm.weight",
            "decoder.head.linear.bias",
            "decoder.head.linear.weight",
            "decoder.head.norm.weight",
        ))
        self.assertEqual(int.from_bytes(encoded[:8], "little") % 8, 0)

    def test_transfer_budget_rejects_one_byte_short_and_extra(self):
        with self.assertRaisesRegex(head_checkpoint.HeadCheckpointError, "below"):
            head_checkpoint.validate_transfer_budget(head_checkpoint.HEAD_F32_BYTES - 1)
        head_checkpoint.validate_transfer_budget(head_checkpoint.HEAD_F32_BYTES)
        with self.assertRaisesRegex(head_checkpoint.HeadCheckpointError, "exceeds"):
            head_checkpoint.validate_transfer_budget(head_checkpoint.HEAD_F32_BYTES + 1)
        with self.assertRaises(head_checkpoint.HeadCheckpointError):
            head_checkpoint.validate_transfer_budget(True)

    def test_false_positive_shape_and_offset_are_rejected(self):
        candidate = _minimal_head_header()
        candidate["decoder.head.linear.weight"]["shape"] = [3072, 4095]
        with self.assertRaises(head_checkpoint.HeadCheckpointError):
            head_checkpoint.head_subset_header(candidate)
        candidate = _minimal_head_header()
        candidate["decoder.head.linear.weight"]["data_offsets"][0] += 4
        with self.assertRaises(head_checkpoint.HeadCheckpointError):
            head_checkpoint.head_subset_header(candidate)

    def test_range_failure_cleans_private_partial_and_allows_retry(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output = root / "source" / "head-f32.safetensors"
            with (
                patch.object(checkpoint, "parse_pinned_header", return_value=_minimal_head_header()),
                patch.object(checkpoint, "inventory", return_value={}),
                patch.object(
                    head_checkpoint.transport, "_request",
                    side_effect=head_checkpoint.transport.SubsetError("transient range failure"),
                ),
            ):
                for _ in range(2):
                    with self.assertRaisesRegex(head_checkpoint.HeadCheckpointError, "range extraction failed"):
                        head_checkpoint.extract_head_subset(b"unused", output, root=root)
                    self.assertFalse(output.exists())
                    self.assertFalse(output.with_suffix(".receipt.json").exists())
                    self.assertEqual(list((root / "source").glob("*.partial")), [])
class PixelUMMHeadReferenceTests(unittest.TestCase):
    def test_full_geometry_permutation_and_inverse_have_pinned_digest(self):
        permutation = head_reference.independent_tile_major_permutation(head_reference.GRID)
        inverse = head_reference.independent_inverse(permutation)
        self.assertEqual(len(permutation), 21_120)
        self.assertEqual(
            head_reference.validate_tile_order(permutation, inverse, head_reference.GRID),
            "4e81c6026e99ee2c17c176c4394682ca632602c198c3065ad311a83a3839cf33",
        )
        with self.assertRaisesRegex(head_reference.HeadReferenceError, "bijection"):
            head_reference.independent_inverse((0, 0))
        reordered = tuple(reversed(permutation))
        reordered_inverse = head_reference.independent_inverse(reordered)
        with self.assertRaisesRegex(head_reference.HeadReferenceError, "differs"):
            head_reference.validate_tile_order(reordered, reordered_inverse, head_reference.GRID)

    def test_partial_tile_order_matches_pinned_source_helper(self):
        source, _ = block_reference.load_upstream(head_reference.DEFAULT_UPSTREAM)
        grid = (1, 2, 33)
        layout = source.local_attention_layout(grid, head_reference.TILE_TOKENS, "cpu")
        self.assertIsNotNone(layout)
        permutation, inverse, _ = layout
        expected = head_reference.independent_tile_major_permutation(
            grid, head_reference.TILE_TOKENS
        )
        self.assertEqual(tuple(permutation.tolist()), expected)
        self.assertEqual(
            head_reference.validate_tile_order(
                tuple(permutation.tolist()), tuple(inverse.tolist()), grid
            ),
            head_reference.permutation_sha256(expected),
        )

    def test_seed_manifest_pin_rejects_false_positive_capture_metadata(self):
        valid = _full36_manifest(42)
        checked = head_reference._source_manifest_contract(
            valid, 42, head_reference._FULL36_INPUTS[42]["manifest_sha256"]
        )
        self.assertEqual(checked["record"]["sha256"], head_reference._FULL36_INPUTS[42]["record_sha256"])

        wrong_order = dict(valid)
        wrong_order["capture_order"] = list(reversed(valid["capture_order"]))
        with self.assertRaisesRegex(head_reference.HeadReferenceError, "capture order"):
            head_reference._source_manifest_contract(
                wrong_order, 42, head_reference._FULL36_INPUTS[42]["manifest_sha256"]
            )

        wrong_record = dict(valid)
        wrong_record["captures"] = dict(valid["captures"])
        wrong_record["captures"]["block35.output"] = dict(
            valid["captures"]["block35.output"]
        )
        wrong_record["captures"]["block35.output"]["offset"] += 1
        with self.assertRaisesRegex(head_reference.HeadReferenceError, "false-positive"):
            head_reference._source_manifest_contract(
                wrong_record, 42, head_reference._FULL36_INPUTS[42]["manifest_sha256"]
            )

        wrong_digest = dict(valid)
        wrong_digest["payload_sha256"] = "0" * 64
        with self.assertRaisesRegex(head_reference.HeadReferenceError, "source metadata"):
            head_reference._source_manifest_contract(
                wrong_digest, 42, head_reference._FULL36_INPUTS[42]["manifest_sha256"]
            )

    def test_fixture_and_mps_memory_budget_are_exact_and_guarded(self):
        self.assertEqual(head_reference.expected_fixture_payload_bytes(), 976_771_072)
        self.assertEqual(head_reference.estimate_mps_peak_bytes(), 2_050_512_896)
        admitted = head_reference.memory_preflight(
            physical_bytes=64 * head_reference.GIB,
            driver_allocated_bytes=4 * head_reference.GIB,
            recommended_max_bytes=52 * head_reference.GIB,
            estimated_peak_bytes=head_reference.estimate_mps_peak_bytes(),
            max_mps_bytes=head_reference.estimate_mps_peak_bytes(),
        )
        self.assertEqual(admitted["required_floor_bytes"], (64 * head_reference.GIB) // 10)
        with self.assertRaisesRegex(head_reference.HeadReferenceError, "estimated MPS peak"):
            head_reference.memory_preflight(
                physical_bytes=64 * head_reference.GIB,
                driver_allocated_bytes=4 * head_reference.GIB,
                recommended_max_bytes=52 * head_reference.GIB,
                estimated_peak_bytes=head_reference.estimate_mps_peak_bytes(),
                max_mps_bytes=head_reference.estimate_mps_peak_bytes() - 1,
            )
        with self.assertRaisesRegex(head_reference.HeadReferenceError, "10%"):
            head_reference.memory_preflight(
                physical_bytes=64 * head_reference.GIB,
                driver_allocated_bytes=4 * head_reference.GIB,
                recommended_max_bytes=64 * head_reference.GIB,
                estimated_peak_bytes=head_reference.estimate_mps_peak_bytes(),
                max_mps_bytes=55 * head_reference.GIB,
            )

    def test_receipt_publish_failure_removes_owned_output_for_retry(self):
        encoded = b"test-header"
        data = b"abcd"
        digest = hashlib.sha256(data).hexdigest()

        def copy_small_range(_response, _start, _end, _total, sink):
            sink.write(data)
            return digest

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output = root / "source" / "head-f32.safetensors"
            real_link = head_checkpoint.os.link
            link_count = 0

            def fail_second_link(source, destination):
                nonlocal link_count
                link_count += 1
                if link_count == 2:
                    raise OSError("injected receipt publication failure")
                return real_link(source, destination)

            common_patches = (
                patch.object(checkpoint, "parse_pinned_header", return_value=_minimal_head_header()),
                patch.object(checkpoint, "inventory", return_value={}),
                patch.object(
                    head_checkpoint, "head_subset_header",
                    return_value=(encoded, 0, len(data), tuple(head_checkpoint.HEAD_TENSORS)),
                ),
                patch.object(head_checkpoint, "HEAD_F32_BYTES", len(data)),
                patch.object(head_checkpoint.transport, "_request", return_value=nullcontext(object())),
                patch.object(head_checkpoint.transport, "copy_verified_range_body", side_effect=copy_small_range),
            )
            with common_patches[0], common_patches[1], common_patches[2], common_patches[3], common_patches[4], common_patches[5], patch.object(
                head_checkpoint.os, "link", side_effect=fail_second_link
            ):
                with self.assertRaisesRegex(head_checkpoint.HeadCheckpointError, "injected receipt publication"):
                    head_checkpoint.extract_head_subset(
                        b"unused", output, root=root, max_payload_bytes=len(data)
                    )
            self.assertFalse(output.exists())
            self.assertFalse(output.with_suffix(".receipt.json").exists())
            self.assertEqual(list((root / "source").glob("*.partial")), [])

            link_count = 0
            with common_patches[0], common_patches[1], common_patches[2], common_patches[3], common_patches[4], common_patches[5]:
                receipt = head_checkpoint.extract_head_subset(
                    b"unused", output, root=root, max_payload_bytes=len(data)
                )
            self.assertTrue(output.is_file())
            self.assertTrue(output.with_suffix(".receipt.json").is_file())
            self.assertEqual(receipt["source_f32_bytes"], len(data))


if __name__ == "__main__":
    unittest.main()
