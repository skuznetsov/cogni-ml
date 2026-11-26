"""CPU-only input validation specs for the MLX codec cross-reference emitter."""

import importlib.util
import unittest
from pathlib import Path
from types import SimpleNamespace


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = REPO_ROOT / "scripts/drama3/codec_cross_reference.py"
SCRIPT_SPEC = importlib.util.spec_from_file_location(
    "drama3_codec_cross_reference_spec_target", SCRIPT_PATH
)
EMITTER = importlib.util.module_from_spec(SCRIPT_SPEC)
SCRIPT_SPEC.loader.exec_module(EMITTER)
parse_codes_document = EMITTER.parse_codes_document
validate_book_major_codes = EMITTER.validate_book_major_codes


def fake_upstream_model(upsample_count=2, post_layer_count=8):
    semantic_books = [object()]
    residual_books = [object() for _ in range(9)]
    decoder_blocks = [
        SimpleNamespace(block=[object() for _ in range(5)]) for _ in range(4)
    ]
    decoder_model = [object(), *decoder_blocks, object(), object(), object()]
    quantizer = SimpleNamespace(
        semantic_quantizer=SimpleNamespace(quantizers=semantic_books),
        quantizer=SimpleNamespace(quantizers=residual_books),
        post_module=SimpleNamespace(layers=[object() for _ in range(post_layer_count)]),
        upsample=[[object(), object()] for _ in range(upsample_count)],
    )
    return SimpleNamespace(quantizer=quantizer, decoder=SimpleNamespace(model=decoder_model))


def frame_rows(count=2):
    rows = []
    for frame in range(count):
        rows.append([248000 + frame, *[book * 10 + frame for book in range(10)]])
    return rows


class CodecCrossReferenceInputSpec(unittest.TestCase):
    def test_pinned_trace_manifest_has_58_unique_ordered_stages(self):
        books, names = EMITTER._expected_stage_names(fake_upstream_model())
        self.assertEqual(len(books), 10)
        self.assertEqual(len(names), 58)
        self.assertEqual(len(set(names)), 58)
        self.assertEqual(names[:4], [
            "quantizer.book.0.embedding",
            "quantizer.book.0.projection",
            "quantizer.book.1.embedding",
            "quantizer.book.1.projection",
        ])
        self.assertEqual(names[-3:], [
            "decoder.model.5",
            "decoder.model.6",
            "waveform",
        ])

    def test_trace_manifest_fails_closed_when_pinned_shape_changes(self):
        with self.assertRaisesRegex(ValueError, "exactly 8"):
            EMITTER._expected_stage_names(fake_upstream_model(post_layer_count=7))
        with self.assertRaisesRegex(ValueError, "expected 58"):
            EMITTER._expected_stage_names(fake_upstream_model(upsample_count=3))

    def test_native_frame_rows_drop_main_and_become_book_major(self):
        codes, layout = parse_codes_document({"frames": frame_rows()})
        self.assertEqual(layout, "native_time_major_frames")
        self.assertEqual(codes, [[book * 10 + frame for frame in range(2)] for book in range(10)])

    def test_official_mlx_output_frames_are_supported(self):
        codes, layout = parse_codes_document({"output": {"frames": frame_rows()}})
        self.assertEqual(layout, "mlx_time_major_frames")
        self.assertEqual(codes[9], [90, 91])

    def test_chunk_report_requires_and_selects_explicit_chunk_index(self):
        document = {
            "chunks": [
                {"output": {"frames": frame_rows(1)}},
                {"output": {"frames": frame_rows(3)}},
            ]
        }
        with self.assertRaisesRegex(ValueError, "--chunk-index"):
            parse_codes_document(document)
        codes, layout = parse_codes_document(document, chunk_index=1)
        self.assertEqual(layout, "mlx_chunk_time_major_frames")
        self.assertEqual(len(codes[0]), 3)
        with self.assertRaisesRegex(ValueError, "outside"):
            parse_codes_document(document, chunk_index=2)

    def test_book_major_mlx_codes_are_supported(self):
        codes = [[book * 10 + frame for frame in range(2)] for book in range(10)]
        parsed, layout = parse_codes_document({"output": {"codes": codes}})
        self.assertEqual(layout, "book_major_codes")
        self.assertEqual(parsed, codes)

    def test_exactly_ten_equal_length_books_and_expected_ranges_are_required(self):
        codes = [[0] for _ in range(10)]
        self.assertEqual(validate_book_major_codes(codes), codes)
        with self.assertRaisesRegex(ValueError, "exactly 10"):
            validate_book_major_codes(codes[:9])
        with self.assertRaisesRegex(ValueError, "same non-zero frame count"):
            validate_book_major_codes([[0], *([[]] * 9)])
        with self.assertRaisesRegex(ValueError, "book 0 code"):
            validate_book_major_codes([[4096], *([[0]] * 9)])
        with self.assertRaisesRegex(ValueError, "book 9 code"):
            validate_book_major_codes([[0], *([[0]] * 8), [1024]])

    def test_frame_rows_reject_bad_width_types_and_main_code_type(self):
        with self.assertRaisesRegex(ValueError, "11 integers"):
            parse_codes_document({"frames": [[1] * 10]})
        malformed = frame_rows(1)
        malformed[0][3] = True
        with self.assertRaisesRegex(ValueError, "integer"):
            parse_codes_document({"frames": malformed})
        malformed = frame_rows(1)
        malformed[0][0] = 1.5
        with self.assertRaisesRegex(ValueError, "integer"):
            parse_codes_document({"frames": malformed})

    def test_declared_shape_must_match_book_major_payload(self):
        codes = [[0, 1] for _ in range(10)]
        with self.assertRaisesRegex(ValueError, "codes_shape"):
            parse_codes_document({"output": {"codes_shape": [10, 3], "codes": codes}})
        with self.assertRaisesRegex(ValueError, "frame_shape"):
            parse_codes_document({"output": {"frame_shape": [3, 11], "frames": frame_rows(2)}})

    def test_chunk_index_is_only_accepted_for_chunk_reports(self):
        with self.assertRaisesRegex(ValueError, "only valid"):
            parse_codes_document({"frames": frame_rows()}, chunk_index=0)
        with self.assertRaisesRegex(ValueError, "non-negative"):
            parse_codes_document({"chunks": []}, chunk_index=-1)


if __name__ == "__main__":
    unittest.main()
