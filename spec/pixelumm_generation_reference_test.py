#!/usr/bin/env python3
"""CPU-only contract checks for the PixelUMM generation oracle."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest


SUPPORT = Path(__file__).resolve().parent / "support"
sys.path.insert(0, str(SUPPORT))

import pixelumm_generation_reference as reference  # noqa: E402


class PixelUMMGenerationReferenceTests(unittest.TestCase):
    def test_oracle_tensor_manifest_contract_is_exact(self):
        self.assertEqual(
            reference.TENSORS,
            (
                ("stem.output_bf16", "BF16", (1, 5760, 4096)),
                ("assembly.output_f32", "F32", (1, 3, 120, 288, 512)),
            ),
        )
        self.assertEqual(
            [reference.tensor_nbytes(dtype, shape) for _, dtype, shape in reference.TENSORS],
            [47_185_920, 212_336_640],
        )
        self.assertEqual(reference.SCHEMA, "cogni_ml.pixelumm_generation_reference/v1")

    def test_synthetic_window_formula_has_fixed_zero_based_window_index(self):
        self.assertEqual(
            tuple(reference.synthetic_value(index, 0) for index in range(4)),
            (-4.0, -3.9638671875, -3.927734375, -3.8916015625),
        )
        self.assertEqual(reference.synthetic_value(0, 1), -3.1103515625)
        self.assertNotEqual(reference.synthetic_value(0, 4), reference.synthetic_value(0, 0))
        self.assertEqual(
            reference.synthetic_value(5_898_324, 0),
            reference.synthetic_value(5_898_324 % 8192, 0),
        )

    def test_synthetic_window_repeats_small_exact_period_on_cpu(self):
        try:
            import torch
        except ImportError:
            self.skipTest("PyTorch is required to exercise tensor construction")
        window = reference.synthetic_window(torch, 0, "cpu").reshape(-1)
        indices = (0, 8191, 5_898_324, 16_777_217, window.numel() - 1)
        selected = window[list(indices)].float().tolist()
        expected = [
            float(torch.tensor(reference.synthetic_value(index, 0), dtype=torch.float32)
                  .to(torch.bfloat16).float().item())
            for index in indices
        ]
        self.assertEqual(selected, expected)

    def test_display_postprocess_preserves_bf16_operation_boundaries(self):
        try:
            import torch
        except ImportError:
            self.skipTest("PyTorch is required to exercise dtype rounding")
        raw = torch.tensor([-0.96484375, -0.44140625, 0.123046875, 1.5, -1.5],
                           dtype=torch.bfloat16)
        pixels = reference.postprocess_bf16(torch, raw)
        self.assertEqual(pixels.dtype, torch.float32)
        self.assertEqual(pixels.tolist(), [0.017578125, 0.279296875, 0.5625, 1.0, 0.0])

    def test_source_pin_validation_checks_revision_and_stem_merge_postprocess_sources(self):
        root = reference._verify_upstream_files(reference.DEFAULT_UPSTREAM)
        self.assertEqual(root, reference.DEFAULT_UPSTREAM.resolve())
        self.assertEqual(
            set(reference.SOURCE_SHA256),
            {reference.MODEL_PATH, reference.STREAMING_PATH, reference.POSTPROCESS_PATH},
        )

    def test_default_stem_is_canonical_and_symlink_final_component_is_rejected(self):
        self.assertFalse(reference.DEFAULT_STEM.is_symlink())
        self.assertEqual(reference.DEFAULT_STEM, reference.DEFAULT_STEM.resolve(strict=True))
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "target.bin"
            target.write_bytes(b"stem")
            alias = Path(directory) / "alias.bin"
            alias.symlink_to(target)
            with self.assertRaisesRegex(reference.ReferenceError, "must not be a symlink"):
                reference._canonical_file(alias, "stem checkpoint")

    def test_manifest_validation_rejects_unpinned_document_before_sidecar_access(self):
        malformed = {"schema": "other", "payload": "missing.tensors.bin"}
        raw = (json.dumps(malformed) + "\n").encode()
        with tempfile.TemporaryDirectory() as directory:
            manifest = Path(directory) / "bad.json"
            manifest.write_bytes(raw)
            with self.assertRaises(reference.ReferenceError):
                reference._validate_input_bundle(manifest, hashlib.sha256(raw).hexdigest())


if __name__ == "__main__":
    unittest.main()
