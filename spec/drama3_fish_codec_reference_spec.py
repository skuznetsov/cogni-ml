"""CPU-only contract tests for the pinned Fish codec reference exporter."""

from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path
from types import SimpleNamespace


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/drama3/fish_codec_reference.py"
SPEC = importlib.util.spec_from_file_location("fish_codec_reference_under_test", SCRIPT)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError(f"cannot load Fish reference exporter from {SCRIPT}")
reference = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(reference)


class FakeLayer:
    pass


class FakeTensor:
    shape = (1, 2, 3)

    def detach(self):
        return self

    def clone(self):
        return FakeTensor()


class FakeCodebook:
    def __init__(self):
        self.calls = 0
        self.output = FakeTensor()

    def embed_code(self, _codes):
        self.calls += 1
        return self.output


def make_fake_model():
    books = [SimpleNamespace(codebook=object(), out_proj=object()) for _ in range(10)]
    decoder_blocks = [
        SimpleNamespace(block=[FakeLayer() for _ in range(5)]) for _ in range(4)
    ]
    return SimpleNamespace(
        quantizer=SimpleNamespace(
            semantic_quantizer=SimpleNamespace(quantizers=books[:1]),
            quantizer=SimpleNamespace(quantizers=books[1:]),
            post_module=SimpleNamespace(layers=[FakeLayer() for _ in range(8)]),
            upsample=[[FakeLayer(), FakeLayer()], [FakeLayer(), FakeLayer()]],
        ),
        decoder=SimpleNamespace(
            model=[FakeLayer(), *decoder_blocks, FakeLayer(), FakeLayer(), FakeLayer()]
        ),
    )


class FishCodecReferenceSpec(unittest.TestCase):
    def test_pinned_source_gate_accepts_only_exact_clean_inputs(self):
        reference.validate_pin_metadata(
            reference.EXPECTED_FISH_REVISION, "", reference.EXPECTED_CODEC_SHA256
        )
        with self.assertRaisesRegex(ValueError, "revision mismatch"):
            reference.validate_pin_metadata("other", "", reference.EXPECTED_CODEC_SHA256)
        with self.assertRaisesRegex(ValueError, "must be clean"):
            reference.validate_pin_metadata(
                reference.EXPECTED_FISH_REVISION, " M changed.py\n", reference.EXPECTED_CODEC_SHA256
            )
        with self.assertRaisesRegex(ValueError, "SHA-256 mismatch"):
            reference.validate_pin_metadata(reference.EXPECTED_FISH_REVISION, "", "bad")

    def test_strict_state_dict_guard_checks_keys_and_shapes(self):
        expected = {"a": SimpleNamespace(shape=(2, 3)), "b": SimpleNamespace(shape=(4,))}
        supplied = {"a": SimpleNamespace(shape=(2, 5)), "c": SimpleNamespace(shape=(1,))}
        self.assertEqual(
            reference._state_dict_mismatches(expected, supplied),
            {
                "missing": ["b"],
                "unexpected": ["c"],
                "shape_mismatches": [
                    {"name": "a", "expected": [2, 3], "supplied": [2, 5]}
                ],
            },
        )
        self.assertFalse(
            any(
                reference._state_dict_mismatches(
                    {"a": SimpleNamespace(shape=(2, 3))},
                    {"a": SimpleNamespace(shape=(2, 3))},
                ).values()
            )
        )

    def test_original_graph_emits_the_58_expected_stage_names(self):
        books, names = reference._expected_stage_names(make_fake_model())
        self.assertEqual(len(books), 10)
        self.assertEqual(len(names), 58)
        self.assertEqual(names[:4], [
            "quantizer.book.0.embedding",
            "quantizer.book.0.projection",
            "quantizer.book.1.embedding",
            "quantizer.book.1.projection",
        ])
        self.assertEqual(names[20], "quantizer.codebook_sum")
        self.assertEqual(names[-3:], ["decoder.model.5", "decoder.model.6", "waveform"])

    def test_graph_guard_rejects_wrong_layer_count(self):
        model = make_fake_model()
        model.quantizer.post_module.layers.pop()
        with self.assertRaisesRegex(ValueError, "exactly 8"):
            reference._expected_stage_names(model)

    def test_embedding_capture_is_scoped_to_one_codebook_instance_and_restored(self):
        first = FakeCodebook()
        second = FakeCodebook()
        trace = reference._FishTrace()
        trace.add_method_capture(first, "embed_code", "book.0.embedding", "ntc")
        try:
            first_result = first.embed_code([0])
            second_result = second.embed_code([1])
            self.assertIs(first_result, first.output)
            self.assertIs(second_result, second.output)
            self.assertEqual(first.calls, 1)
            self.assertEqual(second.calls, 1)
            self.assertEqual(list(trace.values), ["book.0.embedding"])
            self.assertEqual(trace.layouts["book.0.embedding"], "ntc")
        finally:
            trace.close()
        self.assertNotIn("embed_code", first.__dict__)
        self.assertEqual(first.embed_code([2]).shape, (1, 2, 3))
        self.assertEqual(first.calls, 2)

    def test_device_guard_accepts_only_default_mps_index_or_exact_cuda_index(self):
        mps_default = SimpleNamespace(type="mps", index=None)
        mps_zero = SimpleNamespace(type="mps", index=0)
        mps_one = SimpleNamespace(type="mps", index=1)
        mps_requested = SimpleNamespace(type="mps", index=None)
        self.assertTrue(reference._device_matches(mps_default, mps_requested))
        self.assertTrue(reference._device_matches(mps_zero, mps_requested))
        self.assertFalse(reference._device_matches(mps_one, mps_requested))
        self.assertTrue(
            reference._device_matches(
                SimpleNamespace(type="cuda", index=0),
                SimpleNamespace(type="cuda", index=0),
            )
        )
        self.assertFalse(
            reference._device_matches(
                SimpleNamespace(type="cuda", index=1),
                SimpleNamespace(type="cuda", index=0),
            )
        )

    def test_native_frame_and_mlx_code_reports_parse_to_book_major(self):
        native = {
            "frames": [[17, *range(10)], [18, *range(10)]],
            "frame_shape": [2, 11],
        }
        codes, layout = reference.parse_codes_document(native)
        self.assertEqual(layout, "native_time_major_frames")
        self.assertEqual(codes[0], [0, 0])
        self.assertEqual(codes[9], [9, 9])

        mlx = {"output": {"generated_codes": [[0, 1] for _ in range(10)]}}
        codes, layout = reference.parse_codes_document(mlx)
        self.assertEqual(layout, "book_major_codes")
        self.assertEqual(codes[0], [0, 1])

    def test_chunked_mlx_requires_a_valid_explicit_index(self):
        report = {
            "chunks": [
                {"output": {"codes": [[0] for _ in range(10)]}},
                {"output": {"codes": [[1] for _ in range(10)]}},
            ]
        }
        codes, _ = reference.parse_codes_document(report, chunk_index=1)
        self.assertEqual(codes[0], [1])
        with self.assertRaisesRegex(ValueError, "explicit --chunk-index"):
            reference.parse_codes_document(report)
        with self.assertRaisesRegex(ValueError, "outside"):
            reference.parse_codes_document(report, chunk_index=2)

    def test_codebook_range_and_lengths_fail_closed(self):
        valid = [[0] for _ in range(10)]
        invalid = [row[:] for row in valid]
        invalid[0][0] = 4096
        with self.assertRaisesRegex(ValueError, "outside"):
            reference.validate_book_major_codes(invalid)
        ragged = [row[:] for row in valid]
        ragged[3].append(1)
        with self.assertRaisesRegex(ValueError, "same non-zero frame count"):
            reference.validate_book_major_codes(ragged)


if __name__ == "__main__":
    unittest.main()
