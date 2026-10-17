import struct
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from gemma4_prune_layers import _parse, inspect_gguf, prune_gguf


def _u32(value):
    return struct.pack("<I", value)


def _u64(value):
    return struct.pack("<Q", value)


def _string(value):
    encoded = value.encode("utf-8")
    return _u64(len(encoded)) + encoded


def _meta_string(value):
    return _string(value)


def _meta_uint32(value):
    return _u32(value)


def _meta_array(value_type, values):
    return _u32(value_type) + _u64(len(values)) + b"".join(values)


def write_gguf(
    path,
    *,
    ple_length=0,
    shared_kv_layers=0,
    extra_metadata=None,
    extra_tensors=None,
):
    metadata = [
        ("general.architecture", 8, _meta_string("gemma4")),
        ("general.name", 8, _meta_string("tiny-gemma4")),
        ("general.alignment", 4, _meta_uint32(32)),
        ("gemma4.block_count", 4, _meta_uint32(2)),
        ("gemma4.embedding_length_per_layer_input", 4, _meta_uint32(ple_length)),
        ("gemma4.attention.shared_kv_layers", 4, _meta_uint32(shared_kv_layers)),
        (
            "gemma4.attention.head_count_kv",
            9,
            _meta_array(5, [_u32(8), _u32(1)]),
        ),
        (
            "gemma4.attention.sliding_window_pattern",
            9,
            _meta_array(7, [b"\x01", b"\x00"]),
        ),
    ]
    metadata.extend(extra_metadata or [])

    tensors = [
        ("output_norm.weight", [2], 0, struct.pack("<2f", 1.0, 2.0)),
        ("rope_freqs.weight", [2], 0, struct.pack("<2f", 3.0, 4.0)),
        ("token_embd.weight", [2, 2], 0, struct.pack("<4f", 5.0, 6.0, 7.0, 8.0)),
        ("blk.0.attn.weight", [2, 2], 0, struct.pack("<4f", 9.0, 10.0, 11.0, 12.0)),
        ("blk.1.attn.weight", [2, 2], 0, struct.pack("<4f", 13.0, 14.0, 15.0, 16.0)),
    ]
    tensors.extend(extra_tensors or [])

    # Build the GGUF header twice: tensor offsets depend only on the fixed-size
    # tensor-info records, not on their payload bytes.
    def encode_header(offsets):
        out = bytearray(b"GGUF" + _u32(3) + _u64(len(tensors)) + _u64(len(metadata)))
        for key, value_type, raw in metadata:
            out += _string(key) + _u32(value_type) + raw
        for (name, dims, tensor_type, _), offset in zip(tensors, offsets):
            out += _string(name) + _u32(len(dims))
            for dim in dims:
                out += _u64(dim)
            out += _u32(tensor_type) + _u64(offset)
        return out

    alignment = 32
    probe = encode_header([0] * len(tensors))
    data_start = (len(probe) + alignment - 1) // alignment * alignment
    offsets = []
    cursor = 0
    for _, _, _, payload in tensors:
        cursor = (cursor + alignment - 1) // alignment * alignment
        offsets.append(cursor)
        cursor += len(payload)
    header = encode_header(offsets)
    data_start = (len(header) + alignment - 1) // alignment * alignment
    with open(path, "wb") as stream:
        stream.write(header)
        stream.write(b"\x00" * (data_start - len(header)))
        cursor = 0
        for payload, offset in zip((item[3] for item in tensors), offsets):
            stream.write(b"\x00" * (offset - cursor))
            stream.write(payload)
            cursor = offset + len(payload)


class Gemma4PruneLayersSpec(unittest.TestCase):
    def test_inspect_reports_architecture_counts_payload_and_layer_arrays(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "tiny.gguf"
            write_gguf(source)

            report = inspect_gguf(source)

            self.assertEqual(report["architecture"], "gemma4")
            self.assertEqual(report["decoder_layers"], 2)
            self.assertEqual(report["shared_kv_layers"], 0)
            self.assertEqual(report["tensor_count"], 5)
            self.assertEqual(report["parameter_count"], 16)
            self.assertEqual(report["tensor_payload_bytes"], 64)
            self.assertEqual(
                report["layer_metadata_arrays"]["gemma4.attention.head_count_kv"],
                [8, 1],
            )
            self.assertEqual(
                report["layer_metadata_arrays"]["gemma4.attention.sliding_window_pattern"],
                [True, False],
            )
            self.assertEqual(report["output_head_storage"], "token_embd.weight")
            self.assertEqual(len(report["source_sha256"]), 64)

    def test_prune_retains_prefix_and_globals_and_updates_layer_metadata(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "tiny.gguf"
            target = Path(directory) / "tiny-prefix-1.gguf"
            write_gguf(source)
            source_bytes = source.read_bytes()

            report = prune_gguf(source, target, 1)

            self.assertEqual(source.read_bytes(), source_bytes)
            self.assertTrue(target.exists())
            self.assertTrue(Path(str(target) + ".manifest.json").exists())
            self.assertEqual(report["original_decoder_layers"], 2)
            self.assertEqual(report["decoder_layers"], 1)
            self.assertEqual(report["pruned_decoder_layers"], 1)
            self.assertEqual(report["tensor_count"], 4)
            self.assertEqual(report["parameter_count"], 12)
            self.assertEqual(report["tensor_payload_bytes"], 48)
            self.assertEqual(
                report["layer_metadata_arrays"]["gemma4.attention.head_count_kv"], [8]
            )
            self.assertEqual(
                report["layer_metadata_arrays"]["gemma4.attention.sliding_window_pattern"],
                [True],
            )
            self.assertIn("experimental decoder prefix 1/2", report["output_name"])
            self.assertEqual(report["output_sha256"], report["output"]["sha256"])
            self.assertEqual(report["output"]["file_bytes"], target.stat().st_size)

            derived = inspect_gguf(target)
            self.assertEqual(derived["decoder_layers"], 1)
            self.assertEqual(derived["tensor_count"], 4)
            self.assertEqual(derived["parameter_count"], 12)
            self.assertEqual(derived["tensor_payload_bytes"], 48)

    def test_prune_refuses_nonzero_ple_before_creating_output(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "ple.gguf"
            target = Path(directory) / "must-not-exist.gguf"
            write_gguf(source, ple_length=4)

            with self.assertRaisesRegex(ValueError, "PLE|per-layer embedding"):
                prune_gguf(source, target, 1)

            self.assertFalse(target.exists())

    def test_prune_refuses_shared_kv_layers_before_creating_output(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "shared-kv.gguf"
            target = Path(directory) / "must-not-exist.gguf"
            write_gguf(source, shared_kv_layers=1)

            with self.assertRaisesRegex(ValueError, "shared KV"):
                prune_gguf(source, target, 1)

            self.assertFalse(target.exists())

    def test_prune_refuses_unknown_architecture_array_with_one_value_per_layer(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "unknown-array.gguf"
            target = Path(directory) / "must-not-exist.gguf"
            write_gguf(
                source,
                extra_metadata=[
                    ("gemma4.attention.unknown_schedule", 9, _meta_array(4, [_u32(1), _u32(2)]))
                ],
            )

            with self.assertRaisesRegex(ValueError, "unrecognized.*layer|layer.*metadata"):
                prune_gguf(source, target, 1)

            self.assertFalse(target.exists())

    def test_prune_preserves_quantized_payload_bytes_and_refuses_overwrite(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "quant.gguf"
            target = Path(directory) / "quant-prefix-1.gguf"
            # Official b11714 IDs: Q4_K=12 (256 values / 144 bytes) and
            # Q6_K=14 (256 values / 210 bytes). Payloads stay opaque and exact.
            q4_k = bytes((index * 7) % 256 for index in range(144))
            q6_k = bytes((index * 11) % 256 for index in range(210))
            write_gguf(
                source,
                extra_tensors=[
                    ("blk.0.q4_k", [256], 12, q4_k),
                    ("blk.0.q6_k", [256], 14, q6_k),
                ],
            )

            source_report = inspect_gguf(source)
            self.assertEqual(source_report["tensor_payload_bytes"], 64 + 144 + 210)
            report = prune_gguf(source, target, 1)
            self.assertEqual(report["tensor_payload_bytes"], 48 + 144 + 210)
            source_file = _parse(source)
            output_file = _parse(target)
            for tensor_name, expected in (("blk.0.q4_k", q4_k), ("blk.0.q6_k", q6_k)):
                source_tensor = next(t for t in source_file.tensors if t.name == tensor_name)
                output_tensor = next(t for t in output_file.tensors if t.name == tensor_name)
                with source.open("rb") as original, target.open("rb") as derived:
                    original.seek(source_file.data_start + source_tensor.offset)
                    derived.seek(output_file.data_start + output_tensor.offset)
                    self.assertEqual(original.read(source_tensor.payload_bytes), expected)
                    self.assertEqual(derived.read(output_tensor.payload_bytes), expected)

            with self.assertRaises(FileExistsError):
                prune_gguf(source, target, 1)

            self.assertTrue(target.exists())

    def test_prune_rejects_block_count_inconsistent_with_tensor_names(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "bad-block.gguf"
            target = Path(directory) / "must-not-exist.gguf"
            write_gguf(source, extra_tensors=[("blk.2.attn.weight", [2, 2], 0, b"x" * 16)])

            with self.assertRaisesRegex(ValueError, "block_count|outside|inconsistent"):
                prune_gguf(source, target, 1)

            self.assertFalse(target.exists())

    def test_output_and_manifest_collisions_never_clobber_existing_files(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "tiny.gguf"
            target = Path(directory) / "prefix.gguf"
            manifest = Path(str(target) + ".manifest.json")
            write_gguf(source)
            target.write_bytes(b"preexisting model")
            with self.assertRaises(FileExistsError):
                prune_gguf(source, target, 1)
            self.assertEqual(target.read_bytes(), b"preexisting model")

            target.unlink()
            manifest.write_text("preexisting manifest", encoding="utf-8")
            with self.assertRaises(FileExistsError):
                prune_gguf(source, target, 1)
            self.assertFalse(target.exists())
            self.assertEqual(manifest.read_text(encoding="utf-8"), "preexisting manifest")


if __name__ == "__main__":
    unittest.main()
