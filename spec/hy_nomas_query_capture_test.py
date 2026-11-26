"""CPU falsifiers for source block-0 Q/K receipt admission."""
import importlib.util
from pathlib import Path
import unittest


CAPTURE_SOURCE = (Path(__file__).resolve().parent / "support" / "hy_nomas_query_capture.py")
SPEC = importlib.util.spec_from_file_location("hy_nomas_query_capture_test_module", CAPTURE_SOURCE)
capture = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(capture)


def inventory_fixture():
    shapes = {
        "img.q_rope": (1, 2),
        "img.k_rope": (1, 2),
        "cache.k_bf16": (1, 2),
    }
    names = list(shapes)
    records = {}
    header = {"__metadata__": {}}
    for offset, name in reversed(list(enumerate(names))):
        records[name] = {
            "dtype": "BF16", "shape": list(shapes[name]), "nbytes": 4,
            "sha256": f"{offset + 1:064x}",
        }
        header[name] = {
            "dtype": "BF16", "shape": list(shapes[name]),
            "data_offsets": [offset * 4, (offset + 1) * 4],
        }
    return shapes, records, header


class QueryCaptureInventoryTests(unittest.TestCase):
    def validate(self, records, header, *, count=3, shapes=None, total_size=28):
        if shapes is None:
            shapes, _, _ = inventory_fixture()
        capture._validate_source_tensor_inventory(
            records, header, tensor_count=count, expected_shapes=shapes,
            data_start=16, total_size=total_size,
        )

    def test_receipt_and_header_key_order_is_not_semantic(self):
        shapes, records, header = inventory_fixture()
        self.validate(records, header, shapes=shapes)
        self.validate(dict(reversed(list(records.items()))),
                      dict(reversed(list(header.items()))), shapes=shapes)

    def test_exact_inventory_and_count_are_required(self):
        shapes, records, header = inventory_fixture()
        missing = dict(records)
        missing.pop("img.k_rope")
        with self.assertRaisesRegex(capture.QueryCaptureError, "inventory"):
            self.validate(missing, header, shapes=shapes)

        extra = dict(records)
        extra["unexpected.tensor"] = records["img.q_rope"]
        with self.assertRaisesRegex(capture.QueryCaptureError, "inventory"):
            self.validate(extra, header, shapes=shapes)

        with self.assertRaisesRegex(capture.QueryCaptureError, "inventory"):
            self.validate(records, header, count=4, shapes=shapes)

    def test_header_records_and_contiguous_ranges_are_checked(self):
        shapes, records, header = inventory_fixture()
        malformed = dict(header)
        malformed.pop("img.q_rope")
        with self.assertRaisesRegex(capture.QueryCaptureError, "Safetensors inventory"):
            self.validate(records, malformed, shapes=shapes)

        malformed = {name: dict(record) if isinstance(record, dict) else record
                     for name, record in header.items()}
        malformed["img.q_rope"]["shape"] = [1, 3]
        with self.assertRaisesRegex(capture.QueryCaptureError, "header record"):
            self.validate(records, malformed, shapes=shapes)

        malformed = {name: dict(record) if isinstance(record, dict) else record
                     for name, record in header.items()}
        malformed["img.k_rope"]["data_offsets"] = [0, 4]
        with self.assertRaisesRegex(capture.QueryCaptureError, "overlap or leave gaps"):
            self.validate(records, malformed, shapes=shapes)

    def test_receipt_record_schema_and_text_extent_are_strict(self):
        shapes, records, header = inventory_fixture()
        malformed = {name: dict(record) for name, record in records.items()}
        malformed["img.q_rope"]["unexpected"] = True
        with self.assertRaisesRegex(capture.QueryCaptureError, "receipt record"):
            self.validate(malformed, header, shapes=shapes)

        with self.assertRaisesRegex(capture.QueryCaptureError, "valid_text_tokens"):
            capture._expected_source_tensor_shapes(True)


if __name__ == "__main__":
    unittest.main()
