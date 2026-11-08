from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import torch
from safetensors import safe_open

ROOT = Path(__file__).resolve().parents[1]
SUPPORT = ROOT / "examples" / "support"
sys.path.insert(0, str(SUPPORT))
import dcae_v_decoder_checkpoint as conversion  # noqa: E402


class DCAEVDecoderCheckpointConversionTest(unittest.TestCase):
    expected_shapes = {
        "decoder.project_in.conv.weight": [2, 2],
        "decoder.project_in.conv.bias": [2],
    }

    def _checkpoint(self, path: Path, *, state: dict | None = None) -> tuple[str, int]:
        if state is None:
            state = {
                "encoder.project_in.weight": torch.tensor([1], dtype=torch.bfloat16),
                "decoder.project_in.conv.weight": torch.tensor(
                    [[1, 2], [3, 4]], dtype=torch.bfloat16
                ),
                "decoder.project_in.conv.bias": torch.tensor(
                    [5, 6], dtype=torch.bfloat16
                ),
            }
        torch.save({"model_state_dict": state}, path)
        raw = path.read_bytes()
        return hashlib.sha256(raw).hexdigest(), len(raw)

    def _convert(
        self, root: Path, source: Path, output: Path, digest: str, size: int
    ) -> dict:
        upstream = root / "autoencoder_dcae_v.py"
        upstream.write_text("# synthetic pinned upstream source\n", encoding="utf-8")
        upstream_digest = hashlib.sha256(upstream.read_bytes()).hexdigest()
        upstream_metadata = {
            "module_path": str(upstream.resolve()),
            "module_sha256": upstream_digest,
            "checkout_revision": conversion.PINNED_UPSTREAM_REVISION,
            "module_history_revision": conversion.PINNED_UPSTREAM_MODULE_REVISION,
            "config": {"variant": conversion.VARIANT},
        }
        with (
            patch.object(conversion, "PINNED_SOURCE_SHA256", digest),
            patch.object(conversion, "PINNED_SOURCE_BYTES", size),
            patch.object(conversion, "PINNED_UPSTREAM_MODULE_SHA256", upstream_digest),
            patch.object(conversion, "EXPECTED_TENSOR_COUNT", 2),
            patch.object(conversion, "EXPECTED_BF16_BYTES", 12),
            patch.object(conversion, "EXPECTED_ENCODER_TENSOR_COUNT", 1),
            patch.object(
                conversion,
                "upstream_decoder_shapes",
                return_value=(self.expected_shapes, upstream_metadata),
            ),
        ):
            return conversion.convert_checkpoint(
                source,
                output,
                expected_source_sha256=digest,
                expected_source_bytes=size,
                source_revision=conversion.PINNED_SOURCE_REVISION,
                upstream_module_path=upstream,
                expected_upstream_sha256=upstream_digest,
                upstream_revision=conversion.PINNED_UPSTREAM_REVISION,
            )

    def test_preserves_decoder_names_and_bf16_values_and_omits_encoder(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source, output = root / "source.pt", root / "decoder.safetensors"
            digest, size = self._checkpoint(source)
            manifest = self._convert(root, source, output, digest, size)

            self.assertTrue(output.is_file())
            self.assertEqual(manifest["source"]["sha256"], digest)
            self.assertEqual(manifest["conversion"]["tensor_count"], 2)
            self.assertEqual(manifest["conversion"]["payload_bytes"], 12)
            self.assertEqual(manifest["conversion"]["source_name_prefix"], "decoder.")
            self.assertFalse(manifest["conversion"]["encoder_included"])
            with safe_open(str(output), framework="pt", device="cpu") as archive:
                self.assertEqual(set(archive.keys()), set(self.expected_shapes))
                self.assertEqual(
                    archive.get_tensor("decoder.project_in.conv.weight").dtype,
                    torch.bfloat16,
                )
                self.assertTrue(
                    torch.equal(
                        archive.get_tensor("decoder.project_in.conv.weight"),
                        torch.tensor([[1, 2], [3, 4]], dtype=torch.bfloat16),
                    )
                )
            saved = json.loads(Path(str(output) + ".manifest.json").read_text())
            self.assertEqual(saved["schema"], conversion.SCHEMA)
            self.assertEqual(saved["output"]["sha256"], manifest["output"]["sha256"])

    def test_rejects_missing_or_unexpected_decoder_tensor(self):
        malformed = (
            {
                "encoder.project_in.weight": torch.tensor([1], dtype=torch.bfloat16),
                "decoder.project_in.conv.weight": torch.ones(
                    (2, 2), dtype=torch.bfloat16
                ),
            },
            {
                "encoder.project_in.weight": torch.tensor([1], dtype=torch.bfloat16),
                "decoder.project_in.conv.weight": torch.ones(
                    (2, 2), dtype=torch.bfloat16
                ),
                "decoder.project_in.conv.bias": torch.ones(2, dtype=torch.bfloat16),
                "decoder.unexpected.weight": torch.ones(1, dtype=torch.bfloat16),
            },
        )
        for state in malformed:
            with (
                self.subTest(names=sorted(state)),
                tempfile.TemporaryDirectory() as temporary,
            ):
                root = Path(temporary)
                source = root / "source.pt"
                digest, size = self._checkpoint(source, state=state)
                with self.assertRaises(conversion.CheckpointError):
                    self._convert(
                        root, source, root / "decoder.safetensors", digest, size
                    )

    def test_rejects_wrong_decoder_shape_dtype_or_non_tensor_entry(self):
        malformed = (
            {
                "encoder.project_in.weight": torch.tensor([1], dtype=torch.bfloat16),
                "decoder.project_in.conv.weight": torch.ones(
                    (2, 3), dtype=torch.bfloat16
                ),
                "decoder.project_in.conv.bias": torch.ones(2, dtype=torch.bfloat16),
            },
            {
                "encoder.project_in.weight": torch.tensor([1], dtype=torch.bfloat16),
                "decoder.project_in.conv.weight": torch.ones(
                    (2, 2), dtype=torch.float32
                ),
                "decoder.project_in.conv.bias": torch.ones(2, dtype=torch.bfloat16),
            },
            {
                "encoder.project_in.weight": torch.tensor([1], dtype=torch.bfloat16),
                "decoder.project_in.conv.weight": torch.ones(
                    (2, 2), dtype=torch.bfloat16
                ).T,
                "decoder.project_in.conv.bias": torch.ones(2, dtype=torch.bfloat16),
            },
            {
                "encoder.project_in.weight": torch.tensor([1], dtype=torch.bfloat16),
                "decoder.project_in.conv.weight": torch.ones(
                    (2, 2), dtype=torch.bfloat16
                ),
                "decoder.project_in.conv.bias": "not a tensor",
            },
        )
        for state in malformed:
            with self.subTest(state=state), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                source = root / "source.pt"
                digest, size = self._checkpoint(source, state=state)
                with self.assertRaises(conversion.CheckpointError):
                    self._convert(
                        root, source, root / "decoder.safetensors", digest, size
                    )

    def test_refuses_unknown_checkpoint_ownership_and_incomplete_encoder_inventory(
        self,
    ):
        states = (
            {
                "other.project.weight": torch.ones(1, dtype=torch.bfloat16),
                **{
                    name: torch.ones(shape, dtype=torch.bfloat16)
                    for name, shape in self.expected_shapes.items()
                },
            },
            {
                **{
                    name: torch.ones(shape, dtype=torch.bfloat16)
                    for name, shape in self.expected_shapes.items()
                },
            },
        )
        for state in states:
            with (
                self.subTest(names=sorted(state)),
                tempfile.TemporaryDirectory() as temporary,
            ):
                root = Path(temporary)
                source = root / "source.pt"
                digest, size = self._checkpoint(source, state=state)
                with self.assertRaises(conversion.CheckpointError):
                    self._convert(
                        root, source, root / "decoder.safetensors", digest, size
                    )

    def test_rejects_untrusted_source_and_upstream_pins(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source.pt"
            digest, size = self._checkpoint(source)
            upstream = root / "autoencoder_dcae_v.py"
            upstream.write_text("# synthetic upstream\n", encoding="utf-8")
            upstream_digest = hashlib.sha256(upstream.read_bytes()).hexdigest()
            with (
                patch.object(conversion, "PINNED_SOURCE_SHA256", digest),
                patch.object(conversion, "PINNED_SOURCE_BYTES", size),
                patch.object(
                    conversion, "PINNED_UPSTREAM_MODULE_SHA256", upstream_digest
                ),
                patch.object(conversion, "EXPECTED_TENSOR_COUNT", 2),
                patch.object(conversion, "EXPECTED_BF16_BYTES", 12),
                patch.object(conversion, "EXPECTED_ENCODER_TENSOR_COUNT", 1),
            ):
                with self.assertRaisesRegex(
                    conversion.CheckpointError, "upstream.*revision"
                ):
                    conversion.convert_checkpoint(
                        source,
                        root / "bad-revision.safetensors",
                        expected_source_sha256=digest,
                        expected_source_bytes=size,
                        source_revision=conversion.PINNED_SOURCE_REVISION,
                        upstream_module_path=upstream,
                        expected_upstream_sha256=upstream_digest,
                        upstream_revision="0" * 40,
                    )
                with self.assertRaisesRegex(
                    conversion.CheckpointError, "pinned Hy-Nomas source"
                ):
                    conversion.convert_checkpoint(
                        source,
                        root / "bad-source.safetensors",
                        expected_source_sha256="0" * 64,
                        expected_source_bytes=size,
                        source_revision=conversion.PINNED_SOURCE_REVISION,
                        upstream_module_path=upstream,
                        expected_upstream_sha256=upstream_digest,
                        upstream_revision=conversion.PINNED_UPSTREAM_REVISION,
                    )
                with self.assertRaisesRegex(
                    conversion.CheckpointError, "upstream.*pinned"
                ):
                    conversion.convert_checkpoint(
                        source,
                        root / "bad-upstream-sha.safetensors",
                        expected_source_sha256=digest,
                        expected_source_bytes=size,
                        source_revision=conversion.PINNED_SOURCE_REVISION,
                        upstream_module_path=upstream,
                        expected_upstream_sha256="0" * 64,
                        upstream_revision=conversion.PINNED_UPSTREAM_REVISION,
                    )
                with self.assertRaisesRegex(conversion.CheckpointError, "byte count"):
                    conversion.convert_checkpoint(
                        source,
                        root / "bad-source-size.safetensors",
                        expected_source_sha256=digest,
                        expected_source_bytes=size + 1,
                        source_revision=conversion.PINNED_SOURCE_REVISION,
                        upstream_module_path=upstream,
                        expected_upstream_sha256=upstream_digest,
                        upstream_revision=conversion.PINNED_UPSTREAM_REVISION,
                    )

    def test_refuses_output_replacement(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source, output = root / "source.pt", root / "decoder.safetensors"
            digest, size = self._checkpoint(source)
            output.write_bytes(b"keep")
            with self.assertRaisesRegex(conversion.CheckpointError, "overwrite"):
                self._convert(root, source, output, digest, size)
            self.assertEqual(output.read_bytes(), b"keep")

    def test_rejects_corrupt_or_unrestricted_pickle_input(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "corrupt.pt"
            source.write_bytes(b"not a torch checkpoint")
            digest = hashlib.sha256(source.read_bytes()).hexdigest()
            with self.assertRaisesRegex(
                conversion.CheckpointError, "weights_only=True"
            ):
                self._convert(
                    root,
                    source,
                    root / "decoder.safetensors",
                    digest,
                    source.stat().st_size,
                )

    def test_meta_upstream_schema_has_exact_pinned_decoder_inventory(self):
        if not conversion.DEFAULT_UPSTREAM_MODULE.is_file():
            self.skipTest("pinned upstream DCAE-V source is not installed")
        shapes, metadata = conversion.upstream_decoder_shapes()
        self.assertEqual(len(shapes), 171)
        self.assertEqual(
            sum(conversion._shape_product(shape) * 2 for shape in shapes.values()),
            2_443_960_856,
        )
        self.assertEqual(shapes["decoder.project_in.conv.weight"], [1024, 32, 3, 3, 3])
        self.assertEqual(
            shapes["decoder.project_out.op_list.2.conv.conv.weight"], [12, 256, 3, 3, 3]
        )
        self.assertEqual(
            metadata["module_sha256"], conversion.PINNED_UPSTREAM_MODULE_SHA256
        )


if __name__ == "__main__":
    unittest.main()
