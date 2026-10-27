import json
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).parent / "support"))
import hy_nomas_pixelumm_checkpoint as pixel


def header():
    result = {}
    offset = 0
    for name, shape in pixel.decoder_schema().items():
        size = 4
        for extent in shape:
            size *= extent
        result[name] = {"dtype": "F32", "shape": list(shape),
                        "data_offsets": [offset, offset + size]}
        offset += size
    # Inventory validates the complete pinned header in production. The fixture
    # exercises structural validation without containing model tensor values.
    remaining = pixel.PAYLOAD_BYTES - offset
    for index in range(169):
        size = 4 if index < 168 else remaining - 168 * 4
        result[f"encoder.fixture_{index}"] = {
            "dtype": "F32", "shape": [size // 4],
            "data_offsets": [offset, offset + size]}
        offset += size
    return result


class InventoryTest(unittest.TestCase):
    def test_complete_structural_inventory(self):
        plan = pixel.inventory(header())
        self.assertEqual(plan["decoder_bf16_bytes"], 13_943_586_816)
        self.assertEqual(plan["block0_bf16_bytes"], 385_892_864)
        self.assertFalse(plan["full_file_hash_verified"])

    def test_missing_decoder_tensor_rejected(self):
        value = header()
        del value["decoder.layers.35.attn.q_norm_t.weight"]
        with self.assertRaises(pixel.InventoryError):
            pixel.inventory(value)

    def test_shape_dtype_and_boolean_offset_rejected(self):
        for field, value in (("shape", [4096, 4095]), ("dtype", "BF16"),
                             ("data_offsets", [True, 4])):
            candidate = header()
            candidate["decoder.layers.0.attn.q_proj.weight"][field] = value
            with self.assertRaises(pixel.InventoryError):
                pixel.inventory(candidate)

    def test_overlap_hole_and_tail_rejected(self):
        for delta in (-4, 4):
            candidate = header()
            candidate["decoder.layers.0.attn.q_proj.weight"]["data_offsets"] = [
                x + delta for x in candidate["decoder.layers.0.attn.q_proj.weight"]["data_offsets"]]
            with self.assertRaises(pixel.InventoryError):
                pixel.inventory(candidate)
        candidate = header()
        candidate["encoder.fixture_168"]["shape"][0] -= 1
        candidate["encoder.fixture_168"]["data_offsets"][1] -= 4
        with self.assertRaises(pixel.InventoryError):
            pixel.inventory(candidate)

    def test_unknown_ownership_rejected(self):
        candidate = header()
        candidate["other.weight"] = candidate.pop("encoder.fixture_168")
        with self.assertRaises(pixel.InventoryError):
            pixel.inventory(candidate)

    def test_header_authentication_and_duplicate_detection(self):
        with self.assertRaises(pixel.InventoryError):
            pixel.parse_pinned_header(json.dumps(header()).encode())
        with self.assertRaises(pixel.InventoryError):
            pixel.parse_header(b'{"x":{},"x":{}}')

    def test_subset_header_has_rebased_exact_extent(self):
        value = header()
        raw, start, end = pixel.subset_header(value)
        parsed = json.loads(raw[8:])
        self.assertEqual(len(parsed), 13)
        self.assertEqual(min(v["data_offsets"][0] for v in parsed.values()), 0)
        self.assertEqual(max(v["data_offsets"][1] for v in parsed.values()), end - start)
        self.assertEqual(end - start, 771_785_728)

    def test_extraction_cannot_bypass_header_identity(self):
        with patch.object(pixel.transport, "_request") as request:
            with self.assertRaises(pixel.InventoryError):
                pixel.extract_block0(json.dumps(header()).encode(), pixel.ROOT / "source" / "test.safetensors")
            request.assert_not_called()

    def test_existing_report_rejected_before_transfer(self):
        with patch.object(pixel.Path, "exists", return_value=True), \
                patch.object(pixel, "extract_block0") as extract, \
                patch.object(sys, "argv", ["inventory", "--header", "unused", "--report", "existing", "--extract-block0", "unused"]):
            with self.assertRaises(pixel.InventoryError):
                pixel.main()
            extract.assert_not_called()


if __name__ == "__main__":
    unittest.main()
