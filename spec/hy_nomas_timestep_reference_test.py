#!/usr/bin/env python3
"""CPU-only schema and extraction tests for the pinned timestep fixture."""

import copy
from pathlib import Path
import sys
import unittest

SUPPORT = Path(__file__).resolve().parent / "support"
sys.path.insert(0, str(SUPPORT))

import hy_nomas_timestep_reference as reference  # noqa: E402


def valid_document():
    return {
        "schema": reference.SCHEMA,
        "schema_version": 1,
        "upstream_revision": reference.PINNED_REVISION,
        "source": {
            "git_commit": reference.PINNED_REVISION,
            "path": reference.SOURCE_RELATIVE_PATH,
            "sha256": reference.SOURCE_SHA256,
            "function": "timestep_embedding",
            "execution": "single exact FunctionDef AST extracted from the pinned file",
        },
        "producer": {
            "source_sha256": "a" * 64,
            "argv": ["python3", "fixture.py"],
            "python": "3.13.0",
            "torch_version": "2.9.0",
            "torch_git_version": "b" * 40,
            "device": "mps",
            "activation_dtype": "F32",
            "mps_fallback": "disabled",
        },
        "spec": {
            "dim": reference.DIM,
            "max_period": reference.MAX_PERIOD,
            "timesteps": list(reference.TIMESTEPS),
        },
        "frequency_basis": {
            "dtype": "F32",
            "shape": [reference.DIM // 2],
            "bits": list(range(reference.DIM // 2)),
            "captured_from": "the single CPU torch.exp result before upstream .to(t.device)",
        },
        "expected": {
            "dtype": "F32",
            "shape": [len(reference.TIMESTEPS), reference.DIM],
            "values": [[0.0] * reference.DIM for _ in reference.TIMESTEPS],
            "computed_on": "MPS by the pinned upstream function for every listed timestep",
        },
    }


class HyNomasTimestepReferenceTests(unittest.TestCase):
    def test_admits_exact_pinned_frequency_and_dynamic_output_geometry(self):
        reference.validate_reference_document(valid_document())

    def test_rejects_wrong_source_timesteps_basis_shape_output_or_device(self):
        mutations = []

        wrong_source = valid_document()
        wrong_source["source"]["sha256"] = "0" * 64
        mutations.append(wrong_source)

        one_timestep = valid_document()
        one_timestep["spec"]["timesteps"] = [500.0]
        mutations.append(one_timestep)

        short_basis = valid_document()
        short_basis["frequency_basis"]["bits"] = [1] * 127
        mutations.append(short_basis)

        wrong_output_shape = valid_document()
        wrong_output_shape["expected"]["shape"] = [1, 256]
        mutations.append(wrong_output_shape)

        nonfinite_output = valid_document()
        nonfinite_output["expected"]["values"][4][0] = float("nan")
        mutations.append(nonfinite_output)

        wrong_device = valid_document()
        wrong_device["producer"]["device"] = "cpu"
        mutations.append(wrong_device)

        for document in mutations:
            with self.subTest(document=document):
                with self.assertRaises(reference.ReferenceError):
                    reference.validate_reference_document(document)

    def test_ast_extraction_selects_only_the_named_function_and_checks_signature(self):
        source = (
            b"def timestep_embedding(t, dim, max_period=10000):\n"
            b"    return t\n\n"
            b"def unrelated():\n"
            b"    return None\n"
        )
        module = reference._extract_function(source)
        self.assertEqual(len(module.body), 1)
        self.assertEqual(module.body[0].name, "timestep_embedding")
        compile(module, "synthetic_embed_layers.py", "exec")

        changed_signature = b"def timestep_embedding(t, dim):\n    return t\n"
        with self.assertRaisesRegex(reference.ReferenceError, "signature"):
            reference._extract_function(changed_signature)


if __name__ == "__main__":
    unittest.main()
