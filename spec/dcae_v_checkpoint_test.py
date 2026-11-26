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
from safetensors.torch import save_file

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "examples" / "support"))
import dcae_v_checkpoint as conversion  # noqa: E402


class DCAEVCheckpointConversionTest(unittest.TestCase):
    expected_shapes = {
        "encoder.project_in.weight": [2, 2],
        "encoder.project_in.bias": [2],
    }

    def _checkpoint(self, path: Path, *, state: dict | None = None) -> tuple[str, int]:
        if state is None:
            state = {
                "encoder.project_in.weight": torch.tensor([[1, 2], [3, 4]], dtype=torch.bfloat16),
                "encoder.project_in.bias": torch.tensor([5, 6], dtype=torch.bfloat16),
                "decoder.project_out.weight": torch.ones((2, 2), dtype=torch.bfloat16),
            }
        torch.save({"model_state_dict": state}, path)
        raw = path.read_bytes()
        return hashlib.sha256(raw).hexdigest(), len(raw)

    def _convert(self, root: Path, source: Path, output: Path, digest: str, size: int) -> dict:
        with patch.object(conversion, "PINNED_SOURCE_SHA256", digest), patch.object(
            conversion, "PINNED_SOURCE_BYTES", size
        ):
            return conversion.convert_checkpoint(
                source,
                output,
                expected_source_sha256=digest,
                expected_source_bytes=size,
                source_revision=conversion.PINNED_SOURCE_REVISION,
                upstream_module_path=Path("/opt/mldata_ssd/cogni-ml/video-scene-20261019/upstream/hy-nomas-20261024/hymm/models/vae/autoencoder_dcae_v.py"),
                expected_shapes=self.expected_shapes,
            )

    def test_preserves_encoder_names_and_bf16_values_and_omits_decoder(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source, output = root / "source.pt", root / "encoder.safetensors"
            digest, size = self._checkpoint(source)
            with patch.object(conversion, "EXPECTED_TENSOR_COUNT", 2), patch.object(
                conversion, "EXPECTED_BF16_BYTES", 12
            ):
                manifest = self._convert(root, source, output, digest, size)
            self.assertTrue(output.is_file())
            self.assertEqual(manifest["source"]["sha256"], digest)
            self.assertEqual(manifest["conversion"]["tensor_count"], 2)
            self.assertEqual(manifest["conversion"]["payload_bytes"], 12)
            self.assertEqual(manifest["conversion"]["source_name_prefix"], "encoder.")
            with safe_open(str(output), framework="pt", device="cpu") as archive:
                self.assertEqual(set(archive.keys()), set(self.expected_shapes))
                self.assertEqual(archive.get_tensor("encoder.project_in.weight").dtype, torch.bfloat16)
                self.assertTrue(torch.equal(
                    archive.get_tensor("encoder.project_in.weight"),
                    torch.tensor([[1, 2], [3, 4]], dtype=torch.bfloat16),
                ))
            manifest_path = Path(str(output) + ".manifest.json")
            saved = json.loads(manifest_path.read_text())
            self.assertEqual(saved["output"]["sha256"], manifest["output"]["sha256"])

    def test_refuses_unrestricted_or_corrupt_pickle_input(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "corrupt.pt"
            source.write_bytes(b"not a torch checkpoint")
            digest = hashlib.sha256(source.read_bytes()).hexdigest()
            with patch.object(conversion, "EXPECTED_TENSOR_COUNT", 2), patch.object(
                conversion, "EXPECTED_BF16_BYTES", 12
            ):
                with self.assertRaisesRegex(conversion.CheckpointError, "safe memory-mapped"):
                    self._convert(root, source, root / "out.safetensors", digest, source.stat().st_size)

    def test_rejects_missing_or_unexpected_encoder_name(self):
        malformed_states = (
            {
                "encoder.project_in.weight": torch.ones((2, 2), dtype=torch.bfloat16),
            },
            {
                "encoder.project_in.weight": torch.ones((2, 2), dtype=torch.bfloat16),
                "encoder.project_in.bias": torch.ones(2, dtype=torch.bfloat16),
                "encoder.unexpected.weight": torch.ones((1,), dtype=torch.bfloat16),
            },
        )
        for state in malformed_states:
            with self.subTest(names=sorted(state)):
                with tempfile.TemporaryDirectory() as temporary:
                    root = Path(temporary)
                    source = root / "source.pt"
                    digest, size = self._checkpoint(source, state=state)
                    with patch.object(conversion, "EXPECTED_TENSOR_COUNT", 2), patch.object(
                        conversion, "EXPECTED_BF16_BYTES", 12
                    ):
                        with self.assertRaises(conversion.CheckpointError):
                            self._convert(root, source, root / "out.safetensors", digest, size)

    def test_rejects_wrong_encoder_shape_or_dtype(self):
        malformed_states = (
            {
                "encoder.project_in.weight": torch.ones((2, 3), dtype=torch.bfloat16),
                "encoder.project_in.bias": torch.ones(2, dtype=torch.bfloat16),
            },
            {
                "encoder.project_in.weight": torch.ones((2, 2), dtype=torch.float32),
                "encoder.project_in.bias": torch.ones(2, dtype=torch.bfloat16),
            },
        )
        for state in malformed_states:
            with self.subTest(shapes={key: tuple(value.shape) for key, value in state.items()}):
                with tempfile.TemporaryDirectory() as temporary:
                    root = Path(temporary)
                    source = root / "source.pt"
                    digest, size = self._checkpoint(source, state=state)
                    with patch.object(conversion, "EXPECTED_TENSOR_COUNT", 2), patch.object(
                        conversion, "EXPECTED_BF16_BYTES", 12
                    ):
                        with self.assertRaises(conversion.CheckpointError):
                            self._convert(root, source, root / "out.safetensors", digest, size)

    def test_requires_caller_source_digest_and_size_to_match(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source.pt"
            digest, size = self._checkpoint(source)
            with patch.object(conversion, "EXPECTED_TENSOR_COUNT", 2), patch.object(
                conversion, "EXPECTED_BF16_BYTES", 12
            ):
                with patch.object(conversion, "PINNED_SOURCE_SHA256", "0" * 64), patch.object(
                    conversion, "PINNED_SOURCE_BYTES", size
                ), self.assertRaisesRegex(conversion.CheckpointError, "SHA-256"):
                    conversion.convert_checkpoint(
                        source,
                        root / "bad-hash.safetensors",
                        expected_source_sha256="0" * 64,
                        expected_source_bytes=size,
                        source_revision=conversion.PINNED_SOURCE_REVISION,
                        upstream_module_path=Path("/opt/mldata_ssd/cogni-ml/video-scene-20261019/upstream/hy-nomas-20261024/hymm/models/vae/autoencoder_dcae_v.py"),
                        expected_shapes=self.expected_shapes,
                    )
                with patch.object(conversion, "PINNED_SOURCE_SHA256", digest), patch.object(
                    conversion, "PINNED_SOURCE_BYTES", size + 1
                ), self.assertRaisesRegex(conversion.CheckpointError, "size mismatch"):
                    conversion.convert_checkpoint(
                        source,
                        root / "bad-size.safetensors",
                        expected_source_sha256=digest,
                        expected_source_bytes=size + 1,
                        source_revision=conversion.PINNED_SOURCE_REVISION,
                        upstream_module_path=Path("/opt/mldata_ssd/cogni-ml/video-scene-20261019/upstream/hy-nomas-20261024/hymm/models/vae/autoencoder_dcae_v.py"),
                        expected_shapes=self.expected_shapes,
                    )


if __name__ == "__main__":
    unittest.main()
