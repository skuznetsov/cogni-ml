"""Reference ingestion pixels and identity are checked without a model launch."""
import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

from PIL import Image, ImageOps


MODULE = Path(__file__).resolve().parents[1] / "examples/support/hy_nomas_reference_image.py"
SPEC = importlib.util.spec_from_file_location("reference_image_preparation", MODULE)
prep = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(prep)


class ReferenceImagePreparationTest(unittest.TestCase):
    def test_pixels_match_upstream_fit_and_manifest_binds_bytes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            image = root / "portrait.png"
            source = Image.new("RGB", (37, 83), (220, 140, 60))
            source.save(image)
            result = prep.prepare_reference_image(image, root / "prepared", "a" * 64)
            manifest_path = Path(result["manifest"])
            manifest = json.loads(manifest_path.read_bytes())
            payload = (manifest_path.parent / manifest["payload"]).read_bytes()
            expected = ImageOps.pad(source, (512, 288), method=Image.Resampling.LANCZOS,
                                    color=(0, 0, 0), centering=(.5, .5)).tobytes()
            self.assertEqual(payload, expected)
            self.assertEqual(len(payload), 442368)
            self.assertEqual(payload[:3], bytes(3))
            self.assertEqual(manifest["image_sha256"], hashlib.sha256(image.read_bytes()).hexdigest())
            self.assertEqual(manifest["payload_sha256"], hashlib.sha256(payload).hexdigest())
            self.assertEqual(result["manifest_sha256"], hashlib.sha256(manifest_path.read_bytes()).hexdigest())
            self.assertEqual(manifest["encoder_safetensors_sha256"], "a" * 64)

    def test_exif_orientation_is_applied_before_fit(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            image = root / "rotated.jpg"
            source = Image.new("RGB", (24, 12), "red")
            source.paste("blue", (12, 0, 24, 12))
            exif = source.getexif()
            exif[274] = 6
            source.save(image, exif=exif)
            result = prep.prepare_reference_image(image, root / "prepared", "b" * 64)
            with Image.open(image) as loaded:
                expected = ImageOps.pad(ImageOps.exif_transpose(loaded).convert("RGB"),
                    (512, 288), method=Image.Resampling.LANCZOS, color=(0, 0, 0),
                    centering=(.5, .5)).tobytes()
            self.assertEqual((Path(result["manifest"]).parent / "pixels.rgb8").read_bytes(), expected)

    def test_invalid_identity_and_existing_output_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            image = root / "input.png"
            Image.new("RGB", (8, 8)).save(image)
            with self.assertRaises(ValueError):
                prep.prepare_reference_image(image, root / "new", "unchecked")
            self.assertFalse((root / "new").exists())
            with self.assertRaises(ValueError):
                prep.prepare_reference_image(image, root, "a" * 64)


if __name__ == "__main__":
    unittest.main()
