from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import torch
from PIL import Image
from safetensors.torch import save_file

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "spec" / "support"))
import dcae_v_reference as reference  # noqa: E402


class DCAEVReferenceContractTest(unittest.TestCase):
    def _fixture(self, root: Path) -> tuple[Path, Path, dict[str, torch.Tensor]]:
        artifact_path = root / "reference.safetensors"
        manifest_path = root / "reference.manifest.json"
        shapes = reference._capture_shapes(40, 32, 32, 40)
        generator = torch.Generator(device="cpu").manual_seed(17031)
        tensors = {
            name: torch.randn(shape, generator=generator, dtype=torch.float32).to(torch.bfloat16)
            for name, shape in shapes.items()
        }
        output_bytes, output_sha256 = reference._write_safetensors_atomic(
            artifact_path,
            tensors,
            {"reference_schema": reference.SCHEMA},
        )
        image_path = root / "fixture.png"
        image_path.write_bytes(b"seeded-image-fixture")
        manifest = reference.build_reference_manifest(
            image_path=image_path,
            image_sha256=hashlib.sha256(image_path.read_bytes()).hexdigest(),
            image_bytes=image_path.stat().st_size,
            checkpoint_sha256=reference.CHECKPOINT_SHA256,
            checkpoint_bytes=reference.CHECKPOINT_BYTES,
            checkpoint_revision=reference.CHECKPOINT_REVISION,
            module_sha256="b" * 64,
            output_path=artifact_path,
            output_bytes=output_bytes,
            output_sha256=output_sha256,
            frames=40,
            height=32,
            width=32,
            torch_version="2.8.0-test",
        )
        reference._write_json_atomic(manifest_path, manifest)
        return artifact_path, manifest_path, tensors

    def _write_candidate(self, path: Path, tensors: dict[str, torch.Tensor]) -> None:
        save_file(tensors, str(path), metadata={"format": "pt"})

    def test_seeded_identical_candidate_passes_and_seeded_mismatch_fails(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            reference_path, manifest_path, tensors = self._fixture(root)
            candidate_path = root / "candidate.safetensors"
            candidate = {name: tensor.clone() for name, tensor in tensors.items()}
            self._write_candidate(candidate_path, candidate)

            passed = reference.compare_artifacts(reference_path, manifest_path, candidate_path)
            self.assertTrue(passed["all_passed"])
            self.assertTrue(all(record["passed"] for record in passed["comparisons"]))

            candidate["trace.stages.5.tile0"].view(-1)[17] += torch.tensor(
                1.0, dtype=torch.bfloat16
            )
            self._write_candidate(candidate_path, candidate)
            failed = reference.compare_artifacts(reference_path, manifest_path, candidate_path)
            self.assertFalse(failed["all_passed"])
            stage_report = next(
                item for item in failed["comparisons"]
                if item["name"] == "trace.stages.5.tile0"
            )
            self.assertEqual(stage_report["mismatches"], 1)

    def test_compare_cli_returns_failure_for_a_mismatch_and_writes_report(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            reference_path, manifest_path, tensors = self._fixture(root)
            candidate_path = root / "candidate.safetensors"
            candidate = {name: tensor.clone() for name, tensor in tensors.items()}
            candidate["latent"].view(-1)[0] += torch.tensor(1.0, dtype=torch.bfloat16)
            self._write_candidate(candidate_path, candidate)
            report_path = root / "comparison.json"
            argv = [
                "dcae_v_reference.py", "compare",
                "--reference", str(reference_path),
                "--manifest", str(manifest_path),
                "--candidate", str(candidate_path),
                "--report", str(report_path),
            ]
            with patch("sys.argv", argv):
                self.assertEqual(reference.main(), 1)
            self.assertFalse(json.loads(report_path.read_text())["all_passed"])

    def test_comparator_rejects_corrupt_header_and_tensor_inventory(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            reference_path, manifest_path, tensors = self._fixture(root)
            candidate_path = root / "candidate.safetensors"
            candidate_path.write_bytes(b"not a safetensors header")
            with self.assertRaisesRegex(reference.ReferenceError, "cannot read native candidate"):
                reference.compare_artifacts(reference_path, manifest_path, candidate_path)

            candidate = {name: tensor.clone() for name, tensor in tensors.items()}
            del candidate["trace.stages.5.tile0"]
            candidate["trace.stages.5.tile1"] = tensors["trace.stages.5.tile0"].clone()
            self._write_candidate(candidate_path, candidate)
            with self.assertRaisesRegex(reference.ReferenceError, "tensor names differ"):
                reference.compare_artifacts(reference_path, manifest_path, candidate_path)

    def test_comparator_rejects_shape_and_dtype_mismatch(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            reference_path, manifest_path, tensors = self._fixture(root)
            candidate_path = root / "candidate.safetensors"

            wrong_shape = {name: tensor.clone() for name, tensor in tensors.items()}
            wrong_shape["latent"] = wrong_shape["latent"][:, :, :-1].contiguous()
            self._write_candidate(candidate_path, wrong_shape)
            with self.assertRaisesRegex(reference.ReferenceError, "shape or dtype mismatch: latent"):
                reference.compare_artifacts(reference_path, manifest_path, candidate_path)

            wrong_dtype = {name: tensor.clone() for name, tensor in tensors.items()}
            wrong_dtype["latent"] = wrong_dtype["latent"].float()
            self._write_candidate(candidate_path, wrong_dtype)
            with self.assertRaisesRegex(reference.ReferenceError, "shape or dtype mismatch: latent"):
                reference.compare_artifacts(reference_path, manifest_path, candidate_path)

    def test_loader_rejects_corrupted_manifest_and_reference_digest(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            artifact_path, manifest_path, _tensors = self._fixture(root)
            manifest = json.loads(manifest_path.read_text())
            manifest["schema"] = "unknown"
            manifest_path.write_text(json.dumps(manifest))
            with self.assertRaisesRegex(reference.ReferenceError, "schema mismatch"):
                reference.load_reference_artifact(artifact_path, manifest_path)

            artifact_path.write_bytes(b"truncated safetensors")
            manifest["schema"] = reference.SCHEMA
            manifest_path.write_text(json.dumps(manifest))
            with self.assertRaisesRegex(reference.ReferenceError, "byte count differs"):
                reference.load_reference_artifact(artifact_path, manifest_path)

    def test_image_clip_matches_fit_normalize_then_pixel_repeat(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            image_path = root / "image.png"
            image = Image.new("RGB", (17, 9))
            image.putdata([(x * 11 % 256, y * 23 % 256, (x + y) * 7 % 256)
                           for y in range(9) for x in range(17)])
            image.save(image_path)

            clip = reference.reference_clip_from_image(
                image_path,
                height=32,
                width=64,
                frames=40,
                device="cpu",
                dtype=torch.bfloat16,
            )
            self.assertEqual(tuple(clip.shape), (1, 3, 40, 32, 64))
            self.assertEqual(clip.dtype, torch.bfloat16)
            self.assertTrue(torch.equal(clip[:, :, 0], clip[:, :, -1]))

            with Image.open(image_path) as source:
                fitted = reference.fit_image_to_frame(source, (64, 32))
                pixels = torch.from_numpy(__import__("numpy").array(fitted, copy=True))
            expected = pixels.permute(2, 0, 1).to(torch.bfloat16).div_(127.5).sub_(1)
            self.assertTrue(torch.equal(clip[0, :, 0], expected))


if __name__ == "__main__":
    unittest.main()
