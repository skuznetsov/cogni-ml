"""CPU-only structural falsifiers for the bounded PixelUMM FFN windows fixture."""
from __future__ import annotations

import ast
import copy
import inspect
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).parent / "support"))
import hy_nomas_pixelumm_ffn_windows_reference as reference


def _cache_block(index: str) -> dict:
    expected = reference.BLOCK_SOURCE_EXPECTED[index]
    chunks = [
        {
            "start": 100 + offset,
            "end_inclusive": 200 + offset,
            "sha256": f"{(offset + 1) % 10}" * 64,
        }
        for offset in range(12)
    ]
    return {
        **expected,
        "source_range_chunks": chunks,
        "weight_keys": [f"block{index}.{name}" for name in reference._BLOCK_WEIGHT_NAMES],
    }


def _memory() -> dict:
    physical = 64 * reference.GIB
    allocated = 1 * reference.GIB
    recommended = 60 * reference.GIB
    cap = reference.MPS_CAP_BYTES
    floor = physical // 10
    return {
        "physical_bytes": physical,
        "driver_allocated_bytes": allocated,
        "recommended_max_bytes": recommended,
        "estimated_peak_bytes": 6 * reference.GIB,
        "max_mps_bytes": cap,
        "required_floor_bytes": floor,
        "headroom_after_cap_bytes": physical - allocated - floor - cap,
        "mps_memory_fraction": cap / recommended,
    }


def _record(dtype: str, shape: tuple[int, ...], offset: int, digest: str) -> dict:
    return {
        "dtype": dtype,
        "shape": list(shape),
        "offset": offset,
        "nbytes": reference._tensor_nbytes(dtype, shape),
        "sha256": digest,
    }


def valid_document() -> dict:
    cursor = 0
    windows = {}
    for name in ("w0", "w1"):
        starts = reference.WINDOWS[name]
        window = {
            "source_slices": {
                "latent_frame_start": starts["latent_frame_start"],
                "latent_frame_count": reference.LATENT_FRAME_COUNT,
                "noise_frame_start": starts["noise_frame_start"],
                "noise_frame_count": reference.NOISE_FRAME_COUNT,
                "latent_slice_sha256": "a" * 64 if name == "w0" else "b" * 64,
                "noise_slice_sha256": "c" * 64 if name == "w0" else "d" * 64,
            },
            "inputs": {},
            "captures": {"block0": {}, "block1": {}},
        }
        for input_name in ("x", "cos", "sin"):
            dtype, shape = reference._INPUT_SHAPES[input_name]
            digest = ("1" if name == "w0" else "2") * 64 if input_name == "x" else "e" * 64
            window["inputs"][input_name] = _record(dtype, shape, cursor, digest)
            cursor += window["inputs"][input_name]["nbytes"]
        for block_name in ("block0", "block1"):
            for stage in reference.CAPTURE_ORDER:
                dtype, shape = reference._STAGE_SHAPES[stage]
                digest = f"{(cursor // 2 + 3) % 10}" * 64
                window["captures"][block_name][stage] = _record(
                    dtype, shape, cursor, digest
                )
                cursor += window["captures"][block_name][stage]["nbytes"]
        windows[name] = window

    return {
        "schema": reference.SCHEMA,
        "upstream_revision": reference.UPSTREAM_REVISION,
        "upstream_model_sha256": reference.MODEL_SHA256,
        "model_sha256": reference.MODEL_SHA256,
        "torch_version": reference.TORCH_VERSION,
        "grid": list(reference.GRID),
        "comparison": dict(reference.COMPARISON),
        "token_order": "tile_major",
        "tile_tokens": reference.TILE_TOKENS,
        "runtime_dtype": reference.BF16,
        "capture_order": list(reference.CAPTURE_ORDER),
        "block_execution": {
            "block0_input": "inputs.x",
            "block1_input": "captures.block0.output",
            "edge": "same_window_output_to_next_input",
        },
        "input_source": {
            "manifest_path": str(reference.INPUT_MANIFEST),
            "manifest_sha256": reference.INPUT_MANIFEST_SHA256,
            "payload_path": str(reference.INPUT_MANIFEST.parent / "inputs.tensors.bin"),
            "payload_bytes": reference.INPUT_PAYLOAD_BYTES,
            "payload_sha256": reference.INPUT_PAYLOAD_SHA256,
            "tensors": reference.INPUT_SOURCE_RECORDS,
        },
        "input_provenance": {
            "kind": "saved_initial_noise_conditioning_upstream_stem",
            "model_inference": False,
            "interpretation": (
                "saved initial latent and decoder noise passed through the pinned "
                "upstream stem; not a DCAE-encoded pose or generated DiT latent"
            ),
            "x_expression": (
                "embed(patchify(decoder.noise.to(dtype))) + "
                "_condition_tokens(input.latent.to(dtype))"
            ),
            "window_offsets": reference.WINDOW_OFFSETS,
            "stem": {
                "subset_path": str(reference.STEM_SUBSET_PATH),
                "subset_sha256": reference.STEM_SUBSET_SHA256,
                "subset_bytes": reference.STEM_SUBSET_BYTES,
                "receipt_path": str(reference.STEM_RECEIPT_PATH),
                "receipt_sha256": reference.STEM_RECEIPT_SHA256,
                "weights_sha256": reference.STEM_WEIGHTS_SHA256,
            },
        },
        "weight_cache": {
            "manifest_path": str(reference.WEIGHT_CACHE_MANIFEST),
            "manifest_sha256": reference.WEIGHT_CACHE_MANIFEST_SHA256,
            "payload_path": str(
                reference.WEIGHT_CACHE_MANIFEST.parent
                / "pixelumm-thirty-six-bf16-cache.bin"
            ),
            "payload_bytes": reference.WEIGHT_CACHE_PAYLOAD_BYTES,
            "payload_sha256": reference.WEIGHT_CACHE_PAYLOAD_SHA256,
            "schema": "cogni_ml.pixelumm_bf16_weight_cache/v1",
            "upstream_revision": reference.UPSTREAM_REVISION,
            "upstream_model_sha256": reference.MODEL_SHA256,
            "blocks": {"0": _cache_block("0"), "1": _cache_block("1")},
        },
        "windows": windows,
        "payload": "pixelumm-ffn-windows-reference.bin",
        "payload_bytes": cursor,
        "payload_sha256": "f" * 64,
        "producer": {
            "device": "mps",
            "mps_fallback": False,
            "full_generation": False,
            "bounded_layer_calls": 4,
            "captures_per_call": 6,
            "explicit_mps_synchronizations": 1,
            "host_export": "packed tensor records after one terminal MPS synchronization",
            "python_version": "3.12.2",
            "memory_preflight": _memory(),
        },
    }


class PixelUMMFFNWindowsReferenceTest(unittest.TestCase):
    def test_fixed_two_window_two_block_inventory_is_admitted(self):
        document = valid_document()
        reference.validate_document(document, payload_bytes=document["payload_bytes"])
        self.assertEqual(document["payload_bytes"], 2_371_092_480)
        self.assertEqual(document["grid"], [10, 18, 32])
        self.assertEqual(document["capture_order"], list(reference.CAPTURE_ORDER))

    def test_block1_is_bound_to_same_window_block0_output(self):
        document = valid_document()
        document["block_execution"]["block1_input"] = "inputs.x"
        with self.assertRaisesRegex(reference.WindowsReferenceError, "lineage"):
            reference.validate_document(document)

    def test_real_generated_input_claim_is_rejected(self):
        document = valid_document()
        document["input_provenance"]["kind"] = "real_latent_xt_pixelumm_stem"
        with self.assertRaisesRegex(reference.WindowsReferenceError, "provenance"):
            reference.validate_document(document)

    def test_windows_must_have_distinct_saved_slices_and_stem_inputs(self):
        for field in ("latent_slice_sha256", "noise_slice_sha256"):
            document = valid_document()
            document["windows"]["w1"]["source_slices"][field] = (
                document["windows"]["w0"]["source_slices"][field]
            )
            with self.assertRaisesRegex(reference.WindowsReferenceError, "must differ"):
                reference.validate_document(document)

        document = valid_document()
        document["windows"]["w1"]["inputs"]["x"]["sha256"] = (
            document["windows"]["w0"]["inputs"]["x"]["sha256"]
        )
        with self.assertRaisesRegex(reference.WindowsReferenceError, "must differ"):
            reference.validate_document(document)

    def test_comparison_and_source_bindings_cannot_widen_or_drift(self):
        mutations = (
            lambda doc: doc.update(comparison={"atol": 1e-3, "rtol": 2e-5}),
            lambda doc: doc["input_source"].update(manifest_sha256="0" * 64),
            lambda doc: doc["weight_cache"].update(manifest_sha256="0" * 64),
            lambda doc: doc["weight_cache"]["blocks"]["1"].update(
                source_receipt_sha256="0" * 64
            ),
            lambda doc: doc["input_provenance"]["stem"].update(
                weights_sha256="0" * 64
            ),
        )
        for mutate in mutations:
            document = valid_document()
            mutate(document)
            with self.assertRaises(reference.WindowsReferenceError):
                reference.validate_document(document)

    def test_bad_record_shape_extent_precision_or_digest_is_rejected(self):
        mutations = (
            lambda record: record.update(dtype="F16"),
            lambda record: record.update(shape=[1, 1]),
            lambda record: record.update(offset=record["offset"] + 1),
            lambda record: record.update(nbytes=1),
            lambda record: record.update(sha256="z" * 64),
        )
        for mutate in mutations:
            document = valid_document()
            mutate(document["windows"]["w0"]["captures"]["block0"]["gate_proj"])
            with self.assertRaises(reference.WindowsReferenceError):
                reference.validate_document(document)

    def test_source_chunk_receipts_and_ten_percent_memory_floor_are_required(self):
        document = valid_document()
        del document["weight_cache"]["blocks"]["0"]["source_range_chunks"]
        with self.assertRaises(reference.WindowsReferenceError):
            reference.validate_document(document)

        document = valid_document()
        document["producer"]["memory_preflight"]["driver_allocated_bytes"] = 60 * reference.GIB
        with self.assertRaises(reference.WindowsReferenceError):
            reference.validate_document(document)

    def test_record_placement_maps_capture_key_to_correct_nested_boundary(self):
        windows = {
            "w0": {"inputs": {}, "captures": {"block0": {}, "block1": {}}},
            "w1": {"inputs": {}, "captures": {"block0": {}, "block1": {}}},
        }
        record = {"sha256": "a" * 64}
        reference._place_record(windows, "windows.w0.captures.block0.output", record)
        reference._place_record(windows, "windows.w1.inputs.x", record)
        self.assertIs(windows["w0"]["captures"]["block0"]["output"], record)
        self.assertIs(windows["w1"]["inputs"]["x"], record)

    def test_optional_stem_cleanup_has_an_initialized_local_sentinel(self):
        tree = ast.parse(inspect.getsource(reference.produce))
        function = next(
            node for node in tree.body
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
            and node.name == "produce"
        )
        has_none_sentinel = any(
            isinstance(node, ast.Assign)
            and any(isinstance(target, ast.Name) and target.id == "stem" for target in node.targets)
            and isinstance(node.value, ast.Constant) and node.value.value is None
            for node in ast.walk(function)
        )
        deletes_stem = any(
            isinstance(node, ast.Delete)
            and any(isinstance(target, ast.Name) and target.id == "stem" for target in node.targets)
            for node in ast.walk(function)
        )
        cleanup_assigns_sentinel = any(
            isinstance(node, ast.Try) and node.finalbody
            and any(
                isinstance(statement, ast.Assign)
                and any(isinstance(target, ast.Name) and target.id == "stem"
                        for target in statement.targets)
                and isinstance(statement.value, ast.Constant)
                and statement.value.value is None
                for statement in node.finalbody
            )
            for node in ast.walk(function)
        )
        self.assertTrue(has_none_sentinel)
        self.assertFalse(deletes_stem)
        self.assertTrue(cleanup_assigns_sentinel)


if __name__ == "__main__":
    unittest.main()
