#!/usr/bin/env python3
"""CPU-only admission and layout falsifiers for the full Q4 DiT oracle."""

from __future__ import annotations

import copy
from contextlib import redirect_stderr
import io
import json
import math
from pathlib import Path
import struct
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

SUPPORT = Path(__file__).resolve().parent / "support"
sys.path.insert(0, str(SUPPORT))

import hy_nomas_q4_dit_reference as reference  # noqa: E402
import hy_nomas_text_conditioning_reference as text_reference  # noqa: E402


def _digest(number: int) -> str:
    return f"{number:064x}"


def cache_index_document():
    blocks = []
    for index in range(reference.BLOCK_COUNT):
        base = index * 3 + 1
        blocks.append({
            "block_index": index,
            "manifest": f"block-{index:02d}.json",
            "manifest_sha256": _digest(base),
            "payload_sha256": _digest(base + 1),
            "bf16": f"block-{index:02d}.safetensors",
            "bf16_sha256": _digest(base + 2),
        })
    return {
        "schema": reference.CACHE_INDEX_SCHEMA,
        "schema_version": 1,
        "source": {"sha256": reference.SOURCE_SHA256, "bytes": reference.SOURCE_BYTES},
        "blocks": blocks,
        "globals": {"bf16": "globals.safetensors", "sha256": _digest(1000)},
    }


def q4_manifest(index: int = 7):
    entry = cache_index_document()["blocks"][index]
    layout = reference.expected_q4_tensor_layout(index)
    total = sum(descriptor["nbytes"] for descriptor in layout.values())
    return entry, {
        "schema": reference.CACHE_MANIFEST_SCHEMA,
        "schema_version": 1,
        "source": {"sha256": reference.SOURCE_SHA256, "bytes": reference.SOURCE_BYTES},
        "block_index": index,
        "geometry": {
            key: reference.GEOMETRY[key]
            for key in ("batches", "video_tokens", "text_tokens", "dim", "heads", "head_dim", "ffn_dim")
        },
        "codec": {
            "matrix": "q4_k", "ranges": 8, "block_elements": 256,
            "block_bytes": 144, "vector": "bf16-le",
        },
        "payload": {
            "basename": f"block-{index:02d}.bin",
            "nbytes": total,
            "sha256": entry["payload_sha256"],
        },
        "tensors": copy.deepcopy(layout),
    }


def _tensor_layout():
    tensors = {}
    offset = 0
    for name, shape in sorted(reference.EXPECTED_TENSOR_SHAPES.items()):
        nbytes = reference._element_count(shape) * 4
        tensors[name] = {
            "dtype": "F32", "shape": list(shape),
            "offset_bytes": offset, "nbytes": nbytes,
        }
        offset += nbytes
    return tensors, offset


def valid_document(mode: str = "q4", source_loading: str = "mapped"):
    tensors, tensor_bytes = _tensor_layout()
    index = cache_index_document()
    cache_blocks = [
        {
            "block_index": entry["block_index"],
            "manifest": entry["manifest"],
            "manifest_sha256": entry["manifest_sha256"],
            "payload_sha256": entry["payload_sha256"],
            "bf16": entry["bf16"],
            "bf16_sha256": entry["bf16_sha256"],
        }
        for entry in index["blocks"]
    ]
    precision = dict(reference.PRECISION)
    precision["mode"] = mode
    return {
        "schema": reference.SCHEMA,
        "schema_version": 1,
        "precision_mode": mode,
        "upstream": {
            "revision": reference.PINNED_REVISION,
            "source_sha256": dict(reference.conditioned_reference.PINNED_UPSTREAM_SOURCE_SHA256),
        },
        "source": {
            "checkpoint_path": "/weights/diffusion_pytorch_model_ema_0_995.safetensors",
            "checkpoint_sha256": reference.SOURCE_SHA256,
            "checkpoint_bytes": reference.SOURCE_BYTES,
            "checkpoint_header_sha256": reference.SOURCE_HEADER_SHA256,
            "cache_index": {"path": "/weights/q4cache/index.json", "sha256": _digest(2000)},
            "cache_block_order": list(range(reference.BLOCK_COUNT)),
            "cache_blocks": cache_blocks,
            "precision_source": (
                "original full F32 checkpoint" if mode == "bf16"
                else "pinned Q4_K block payloads plus original global F32 checkpoint"
            ),
        },
        "qwen_source": copy.deepcopy(text_reference.QWEN_SOURCE),
        "qwen_reference": {
            "path": "/fixtures/qwen26.json",
            "sha256": text_reference.QWEN_REFERENCE_SHA256,
        },
        "qwen_sidecar": {
            "path": "/fixtures/qwen26.bf16le",
            "sha256": text_reference.QWEN_SIDECAR_SHA256,
            "nbytes": 268_627_968,
        },
        "geometry": copy.deepcopy(reference.GEOMETRY),
        "input_generation": copy.deepcopy(reference.INPUT_GENERATION),
        "precision": precision,
        "mask": list(reference.MASK),
        "timestep": 500.0,
        "conditioning": copy.deepcopy(reference._CONDITIONING),
        "padding_policy": copy.deepcopy(reference._PADDING_POLICY),
        "tolerance": {"atol": reference.ATOL, "rtol": reference.RTOL},
        "producer": {
            "path": "/repo/spec/support/hy_nomas_q4_dit_reference.py",
            "source_sha256": _digest(3000),
            "device": "mps",
            "activation_readback": "one packed terminal F32 tensor sidecar readback",
            "terminal_activation_readbacks": 1,
            "attention_adapter": "PyTorch MPS SDPA F32, valid joint prefix only",
            "attention_adapter_calls": reference.BLOCK_COUNT,
            "query_chunks_per_block": reference.ATTENTION_CHUNKS_PER_BLOCK,
            "attention_query_chunk_calls": reference.BLOCK_COUNT * reference.ATTENTION_CHUNKS_PER_BLOCK,
            "padding_poison_control_chunks": reference.ATTENTION_CHUNKS_PER_BLOCK,
            "padding_poison_control_passed": True,
            "block_invocations": list(range(reference.BLOCK_COUNT)),
            "captured_block_outputs": list(reference.CAPTURE_BLOCKS),
            "block_handoff": "img/text outputs passed directly to the next block",
            "final_layer": "pinned FinalLayerPerBlock forward plus pinned pointwise unpatchify layout",
            "camera_conditioning": "off",
            "action_conditioning": "off",
            "full_generation": False,
            "weight_source_loading": {
                "block_weights": (
                    "q4_cache_payloads" if mode == "q4"
                    else ("owned_raw_f32_extents" if source_loading == "owned"
                          else "safetensors_mapped_tensor_views")
                ),
                "global_weights": "safetensors_mapped_tensor_views",
            },
        },
        "tensor_file": {
            "name": "full-dit.tensors.f32le",
            "nbytes": tensor_bytes,
            "sha256": _digest(4000),
            "encoding": "f32-le",
        },
        "tensors": tensors,
    }


class _FakeTensor:
    def __init__(self, owner, dtype, count):
        self.owner = owner
        self.dtype = dtype
        self.shape = [count]
        self.values = list(struct.unpack(f"<{count}f", owner))

    def reshape(self, shape):
        self.shape = tuple(shape)
        return self


class _FakeFinite:
    def __init__(self, result):
        self.result = result

    def all(self):
        return self

    def item(self):
        return self.result


class _FakeTorch:
    float32 = object()

    @staticmethod
    def frombuffer(buffer, *, dtype, count):
        if dtype is not _FakeTorch.float32:
            raise AssertionError("owned raw F32 bytes must retain their source dtype")
        return _FakeTensor(buffer, dtype, count)

    @staticmethod
    def isfinite(tensor):
        return _FakeFinite(all(math.isfinite(value) for value in tensor.values))


class HyNomasQ4DiTReferenceTests(unittest.TestCase):
    def test_source_loading_defaults_to_mapped_and_owned_is_bf16_only(self):
        required = [
            "--output", "/tmp/reference.json",
            "--precision", "bf16",
            "--weights-sha256", _digest(1),
            "--cache-index-sha256", _digest(2),
        ]
        self.assertEqual(reference._fixture_parser().parse_args(required).source_loading, "mapped")
        self.assertEqual(
            reference._fixture_parser().parse_args(required + ["--source-loading", "owned"])
            .source_loading,
            "owned",
        )

        q4_args = [
            "--output", "/tmp/reference.json",
            "--precision", "q4",
            "--weights-sha256", _digest(1),
            "--cache-index-sha256", _digest(2),
            "--source-loading", "owned",
        ]
        args = reference._fixture_parser().parse_args(q4_args)
        with patch.dict("os.environ", {"PYTORCH_ENABLE_MPS_FALLBACK": "0"}):
            with self.assertRaisesRegex(reference.ReferenceError, "only available for the BF16"):
                reference.produce_fixture(args)

    def test_stage_telemetry_uses_only_direct_mps_memory_counters(self):
        calls = []

        def counter(name, value):
            def read():
                calls.append(name)
                return value
            return read

        mps = types.SimpleNamespace(
            current_allocated_memory=counter("current", 123),
            driver_allocated_memory=counter("driver", 456),
            synchronize=lambda: self.fail("stage telemetry must not synchronize MPS"),
        )
        stream = io.StringIO()
        with redirect_stderr(stream):
            reference._emit_stage_telemetry(
                types.SimpleNamespace(mps=mps), "block_weights_ready", block_index=7
            )

        marker = json.loads(stream.getvalue())
        self.assertEqual(calls, ["current", "driver"])
        self.assertEqual(marker["stage"], "block_weights_ready")
        self.assertEqual(marker["block_index"], 7)
        self.assertEqual(marker["mps_current_allocated_bytes"], 123)
        self.assertEqual(marker["mps_driver_allocated_bytes"], 456)
        self.assertIsInstance(marker["monotonic_ns"], int)

    def test_admits_q4_and_bf16_fixture_envelopes_with_exact_full_j_inventory(self):
        for mode, source_loading in (("q4", "mapped"), ("bf16", "mapped"), ("bf16", "owned")):
            with self.subTest(mode=mode):
                document = valid_document(mode, source_loading)
                reference.validate_reference_document(document)
        self.assertEqual(len(reference.EXPECTED_TENSOR_SHAPES), 18)
        self.assertEqual(reference.EXPECTED_TENSOR_SHAPES["output.latent"], [1, 32, 6, 22, 40])
        self.assertLess(reference._expected_sidecar_bytes(), reference.SIDE_CAR_LIMIT)

    def test_schema_one_legacy_receipts_default_missing_weight_loading_by_precision(self):
        for mode, expected in (
            ("q4", {
                "block_weights": "q4_cache_payloads",
                "global_weights": "safetensors_mapped_tensor_views",
            }),
            ("bf16", {
                "block_weights": "safetensors_mapped_tensor_views",
                "global_weights": "safetensors_mapped_tensor_views",
            }),
        ):
            with self.subTest(mode=mode):
                document = valid_document(mode)
                document["producer"].pop("weight_source_loading")
                reference.validate_reference_document(document)
                self.assertNotIn("weight_source_loading", document["producer"])
                self.assertEqual(
                    reference._validated_weight_source_loading(document["producer"], mode),
                    expected,
                )

    def test_explicit_weight_loading_receipt_rejects_malformed_modes(self):
        for mode, loading in (
            ("q4", {
                "block_weights": "owned_raw_f32_extents",
                "global_weights": "safetensors_mapped_tensor_views",
            }),
            ("bf16", {
                "block_weights": "q4_cache_payloads",
                "global_weights": "safetensors_mapped_tensor_views",
            }),
            ("bf16", {
                "block_weights": "safetensors_mapped_tensor_views",
                "global_weights": "owned_raw_f32_extents",
            }),
            ("bf16", None),
        ):
            with self.subTest(mode=mode, loading=loading):
                document = valid_document(mode)
                document["producer"]["weight_source_loading"] = loading
                with self.assertRaises(reference.ReferenceError):
                    reference.validate_reference_document(document)

    def test_cache_index_requires_all_ordered_distinct_authenticated_entries(self):
        document = cache_index_document()
        reference.validate_cache_index_document(document)
        for mutate, message in (
            (lambda value: value["blocks"].reverse(), "order"),
            (lambda value: value["blocks"][8].update(block_index=7), "order"),
            (lambda value: value["blocks"][2].update(manifest="../block-02.json"), "basename"),
            (lambda value: value["blocks"][1].update(payload_sha256=value["blocks"][0]["payload_sha256"]), "duplicate"),
            (lambda value: value["source"].update(bytes=True), "source"),
        ):
            changed = copy.deepcopy(document)
            mutate(changed)
            with self.subTest(message=message), self.assertRaises(reference.ReferenceError):
                reference.validate_cache_index_document(changed)

    def test_q4_manifest_authenticates_tensor_names_codec_and_contiguous_offsets(self):
        entry, manifest = q4_manifest()
        reference.validate_q4_manifest(manifest, entry, 7)
        self.assertEqual(len(manifest["tensors"]), 32)
        for mutate in (
            lambda value: value.update(block_index=6),
            lambda value: value["tensors"].pop(next(iter(value["tensors"]))),
            lambda value: value["tensors"][next(iter(value["tensors"]))].update(offset_bytes=1),
            lambda value: value["payload"].update(nbytes=value["payload"]["nbytes"] + 1),
            lambda value: value["codec"].update(block_bytes=143),
            lambda value: value["geometry"].update(heads=True),
        ):
            changed = copy.deepcopy(manifest)
            mutate(changed)
            with self.subTest(mutation=mutate), self.assertRaises(reference.ReferenceError):
                reference.validate_q4_manifest(changed, entry, 7)

    def test_mps_decoder_probe_fixture_has_known_header_scale_min_and_bf16_words(self):
        packed, expected_q4, vector, expected_vector = reference._q4_decoder_probe_fixture()
        self.assertEqual(len(packed), 144)
        self.assertEqual(packed[:4], bytes((0x00, 0x30, 0x00, 0x30)))
        self.assertEqual(
            packed[4:16], bytes((65, 66, 67, 68, 74, 76, 78, 80, 0x21, 0x42, 0x63, 0x84))
        )
        self.assertEqual(len(expected_q4), 256)
        self.assertEqual(expected_q4[:4], [-1.25, -1.0, -0.75, -0.5])
        self.assertEqual(vector, bytes((0x80, 0x3F, 0x00, 0xC0, 0x00, 0x3F, 0x50, 0x40)))
        self.assertEqual(expected_vector, [1.0, -2.0, 0.5, 3.25])

    def test_mask_layout_and_unpatchify_keep_valid_prefix_and_nthwc_mapping(self):
        reference.validate_mask(reference.MASK)
        with self.assertRaisesRegex(reference.ReferenceError, "23-valid-row"):
            reference.validate_mask([0] + reference.MASK[1:])
        reference.validate_unpatchified_shape([1, 32, 6, 22, 40])
        with self.assertRaisesRegex(reference.ReferenceError, "unpatchified"):
            reference.validate_unpatchified_shape([1, 6, 22, 40, 32])

        import numpy as np

        flat = np.arange(reference.VIDEO_TOKENS * 32, dtype=np.int32).reshape(
            1, reference.VIDEO_TOKENS, 32
        )
        actual = reference._unpatchify(types.SimpleNamespace(einsum=np.einsum), flat)
        expected = flat.reshape(1, 6, 22, 40, 32).transpose(0, 4, 1, 2, 3)
        np.testing.assert_array_equal(actual, expected)

    def test_reference_rejects_precision_conditioning_padding_and_receipt_drift(self):
        for mutate in (
            lambda doc: doc["precision"].update(mode="other"),
            lambda doc: doc["conditioning"].update(crop_start=107),
            lambda doc: doc["qwen_source"].update(state_index=25),
            lambda doc: doc["padding_policy"].update(valid_joint_prefix=5304),
            lambda doc: doc["producer"].update(padding_poison_control_passed=False),
            lambda doc: doc["producer"].update(terminal_activation_readbacks=2),
            lambda doc: doc["tolerance"].update(rtol=1e-4),
            lambda doc: doc["source"].update(cache_block_order=list(reversed(range(54)))),
            lambda doc: doc["source"]["cache_blocks"][53].update(block_index=52),
            lambda doc: doc["producer"]["weight_source_loading"].update(
                block_weights="owned_raw_f32_extents"
            ),
        ):
            document = valid_document()
            mutate(document)
            with self.subTest(mutation=mutate), self.assertRaises(reference.ReferenceError):
                reference.validate_reference_document(document)

    def test_owned_block_reader_preserves_raw_f32_bytes_in_distinct_cpu_owners(self):
        first = struct.pack("<2f", 1.25, -3.5)
        second = struct.pack("<1f", 0.125)
        layout = {
            "double_blocks.0.a": ("F32", (2,), 0, len(first)),
            "double_blocks.0.b": ("F32", (1,), len(first), len(first) + len(second)),
        }
        expected_shapes = {
            "double_blocks.0.a": [2],
            "double_blocks.0.b": [1],
        }
        with tempfile.TemporaryDirectory() as scratch:
            path = Path(scratch) / "source.safetensors"
            path.write_bytes(bytes(reference.SOURCE_DATA_OFFSET) + first + second)
            with patch.object(reference, "expected_block_shapes", return_value=expected_shapes), \
                    patch.object(reference, "MAX_SOURCE_TENSOR_BYTES", 8), \
                    patch.object(reference, "MAX_SOURCE_BLOCK_BYTES", 12):
                identity = reference._identity(path)
                state = reference._load_owned_source_block_state(
                    _FakeTorch, path, identity, layout, 0
                )

            self.assertEqual(list(state), ["a", "b"])
            self.assertEqual(state["a"].values, [1.25, -3.5])
            self.assertEqual(state["b"].values, [0.125])
            self.assertEqual(state["a"].owner, bytearray(first))
            self.assertIsInstance(state["a"].owner, bytearray)
            self.assertIsNot(state["a"].owner, first)
            with path.open("r+b") as stream:
                stream.seek(reference.SOURCE_DATA_OFFSET)
                stream.write(bytes(len(first) + len(second)))
            self.assertEqual(state["a"].values, [1.25, -3.5])
            self.assertEqual(state["b"].values, [0.125])

    def test_owned_block_reader_rejects_wrong_names_shapes_dtype_and_overlapping_extents(self):
        base_layout = {
            "double_blocks.0.a": ("F32", (2,), 0, 8),
            "double_blocks.0.b": ("F32", (1,), 8, 12),
        }
        expected_shapes = {"double_blocks.0.a": [2], "double_blocks.0.b": [1]}
        with patch.object(reference, "expected_block_shapes", return_value=expected_shapes), \
                patch.object(reference, "MAX_SOURCE_TENSOR_BYTES", 8), \
                patch.object(reference, "MAX_SOURCE_BLOCK_BYTES", 12):
            reference._validated_source_block_extents(base_layout, 0)
            for changed in (
                {**base_layout, "double_blocks.0.extra": ("F32", (1,), 12, 16)},
                {**base_layout, "double_blocks.0.a": ("F16", (2,), 0, 8)},
                {**base_layout, "double_blocks.0.a": ("F32", (1,), 0, 8)},
                {**base_layout, "double_blocks.0.b": ("F32", (1,), 4, 8)},
            ):
                with self.subTest(layout=changed), self.assertRaises(reference.ReferenceError):
                    reference._validated_source_block_extents(changed, 0)

    def test_owned_reader_requires_exact_reads_and_rechecks_source_identity(self):
        class PartialThenEof:
            def __init__(self):
                self.calls = 0

            def readinto(self, destination):
                self.calls += 1
                if self.calls == 1:
                    destination[:2] = b"ab"
                    return 2
                return 0

        with self.assertRaisesRegex(reference.ReferenceError, "short read"):
            reference._read_exact_into(PartialThenEof(), 4, "double_blocks.0.a")

        first = struct.pack("<2f", 1.25, -3.5)
        second = struct.pack("<1f", 0.125)
        layout = {
            "double_blocks.0.a": ("F32", (2,), 0, 8),
            "double_blocks.0.b": ("F32", (1,), 8, 12),
        }
        expected_shapes = {"double_blocks.0.a": [2], "double_blocks.0.b": [1]}
        with tempfile.TemporaryDirectory() as scratch:
            path = Path(scratch) / "source.safetensors"
            path.write_bytes(bytes(reference.SOURCE_DATA_OFFSET) + first + second)
            with patch.object(reference, "expected_block_shapes", return_value=expected_shapes), \
                    patch.object(reference, "MAX_SOURCE_TENSOR_BYTES", 8), \
                    patch.object(reference, "MAX_SOURCE_BLOCK_BYTES", 12):
                identity = reference._identity(path)
                check_identity = reference._assert_source_file_identity
                calls = 0

                def change_before_after_check(stream, source_path, expected_identity):
                    nonlocal calls
                    calls += 1
                    if calls == 2:
                        source_path.write_bytes(source_path.read_bytes() + b"x")
                    check_identity(stream, source_path, expected_identity)

                with patch.object(reference, "_assert_source_file_identity",
                                  side_effect=change_before_after_check):
                    with self.assertRaisesRegex(reference.ReferenceError, "identity changed"):
                        reference._read_owned_source_block_buffers(path, identity, layout, 0)

    def test_tensor_sidecar_rejects_capture_growth_dtype_and_range_alias(self):
        document = valid_document()
        reference.validate_tensor_layout(document["tensors"], document["tensor_file"]["nbytes"])
        for mutate in (
            lambda value: value["block53.txt.output"].update(dtype="BF16"),
            lambda value: value["final.linear"].update(offset_bytes=4),
            lambda value: value.pop("block7.img.output"),
        ):
            changed = copy.deepcopy(document["tensors"])
            mutate(changed)
            with self.assertRaises(reference.ReferenceError):
                reference.validate_tensor_layout(changed, document["tensor_file"]["nbytes"])


if __name__ == "__main__":
    unittest.main()
