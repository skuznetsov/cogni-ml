#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import importlib.util
import json
import struct
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "qwen_image21_text_reference.py"
SPEC = importlib.util.spec_from_file_location("qwen_image21_text_reference", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


MODEL_REVISION = "790c92633540aa0cb11d9abf19eb46d861714758"
DIFFUSERS_COMMIT = "8b3c707ebd3ec4881f4190cf42931da07eaf3b65"


class FakeEncoding:
    def __init__(self, input_ids: torch.Tensor, attention_mask: torch.Tensor):
        self.input_ids = input_ids
        self.attention_mask = attention_mask
        self.mm_token_type_ids = torch.zeros_like(input_ids)

    def to(self, _device):
        return self


class FakeProcessor:
    def __init__(self, *, fail_on_call: bool = False):
        self.fail_on_call = fail_on_call
        self.tokenizer = SimpleNamespace(encode=lambda _text: [7])

    def apply_chat_template(self, *_args, **_kwargs):
        return [[11, 12, 13]]

    def __call__(self, **_kwargs):
        if self.fail_on_call:
            raise RuntimeError("synthetic processor failure")
        return FakeEncoding(
            torch.tensor([[101, 102, 103, 201, 202, 203]], dtype=torch.long),
            torch.tensor([[1, 1, 1, 1, 1, 1]], dtype=torch.long),
        )


class FakeNorm(torch.nn.Module):
    def forward(self, hidden_states):
        # A non-identity transform makes the pre-final-norm assertion meaningful.
        self.last_input = hidden_states
        self.last_output = hidden_states * 0.5
        return self.last_output


class FakeLanguageModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.norm = FakeNorm()


class FakeEncoder(torch.nn.Module):
    def __init__(self, *, num_hidden_layers: int = 1):
        super().__init__()
        self.model = SimpleNamespace(language_model=FakeLanguageModel())
        self.config = SimpleNamespace(
            text_config=SimpleNamespace(num_hidden_layers=num_hidden_layers)
        )

    def forward(self, input_ids, attention_mask, output_hidden_states, **_kwargs):
        assert output_hidden_states
        batch, sequence = input_ids.shape
        source = torch.arange(batch * sequence * 4096, dtype=torch.float32).reshape(
            batch, sequence, 4096
        ) + 1
        # Qwen's output list stores the result of the norm module. The official
        # pipeline hook replaces that result with its input for this call.
        final_state = self.model.language_model.norm(source)
        return SimpleNamespace(hidden_states=(torch.zeros_like(source), final_state))


def make_official_pipeline(*, fail_on_call: bool = False, num_hidden_layers: int = 1):
    from diffusers.pipelines.qwenimage21.pipeline_qwenimage21 import QwenImage21Pipeline

    processor = FakeProcessor(fail_on_call=fail_on_call)
    pipeline = QwenImage21Pipeline(
        scheduler=None,
        vae=None,
        text_encoder=FakeEncoder(num_hidden_layers=num_hidden_layers),
        processor=processor,
        transformer=None,
    )
    return pipeline, processor


class QwenImage21TextReferenceSpec(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.output_dir = Path(self.temp_dir.name)

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def synthetic_reference(self):
        input_ids = np.array([[101, 102, 103, 201, 202, 203]], dtype=np.int64)
        attention_mask = np.ones((1, 6), dtype=np.int64)
        layers = [
            np.full((1, 6, 4096), layer, dtype=np.float32)
            for layer in (0.0, 1.0, 2.0)
        ]
        return {
            "output_dir": self.output_dir,
            "prompt": "a tiny lighthouse",
            "model_revision": MODEL_REVISION,
            "raw_template_text": "<system></system><user>a tiny lighthouse</user>",
            "max_sequence_length": 8,
            "source_dtype": "float32",
            "device": "cpu",
            "runtime": {
                "python_version": "3.12.2",
                "torch_version": "2.6.0",
                "transformers_version": "5.17.0",
                "diffusers_version": "0.41.0.dev0",
                "diffusers_commit": DIFFUSERS_COMMIT,
            },
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "mm_token_type_ids": np.zeros_like(input_ids),
            "drop_idx": 3,
            "expected_hidden_state_count": 3,
            "pre_final_rmsnorm_embeddings": layers[-1][:, 3:, :],
            "layer_hidden_states": layers,
        }

    def test_capture_uses_official_pipeline_and_records_exact_processor_inputs(self) -> None:
        self.assertEqual(DIFFUSERS_COMMIT, MODULE._installed_diffusers_commit())
        pipeline, original_processor = make_official_pipeline()
        reference = MODULE.capture_pipeline_reference(
            pipeline,
            "a tiny lighthouse",
            max_sequence_length=8,
            device=torch.device("cpu"),
            source_dtype="float32",
        )

        expected_template = (
            "<|im_start|>system\nComprehend and analyze the provided prompt.<|im_end|>\n"
            "<|im_start|>user\na tiny lighthouse<|im_end|>\n"
            "<|im_start|>assistant\n"
        )
        self.assertEqual(expected_template, reference["raw_template_text"])
        self.assertEqual(
            {"padding": True, "padding_side": "left", "return_tensors": "pt"},
            reference["processor_kwargs"],
        )
        self.assertEqual(3, reference["drop_idx"])
        self.assertEqual(2, reference["expected_hidden_state_count"])
        self.assertEqual((1, 6), tuple(reference["input_ids"].shape))
        self.assertEqual((1, 3, 4096), tuple(reference["pre_final_rmsnorm_embeddings"].shape))
        self.assertEqual(2, len(reference["layer_hidden_states"]))
        self.assertEqual(original_processor, pipeline.processor)
        self.assertEqual({}, pipeline.text_encoder._forward_hooks)
        norm = pipeline.text_encoder.model.language_model.norm
        self.assertEqual({}, norm._forward_hooks)
        self.assertEqual(torch.float32, reference["pre_final_rmsnorm_embeddings"].dtype)
        self.assertTrue(torch.equal(norm.last_input, reference["layer_hidden_states"][-1]))
        self.assertTrue(torch.equal(
            reference["pre_final_rmsnorm_embeddings"],
            norm.last_input[:, 3:, :],
        ))
        # The mocked RMSNorm is deliberately non-identity. This catches capture
        # regressions where the official pipeline stops neutralizing final norm.
        self.assertFalse(torch.equal(norm.last_input, norm.last_output))

    def test_capture_rejects_hidden_state_list_shorter_than_text_config(self) -> None:
        # Model config says 2 decoder layers, so the output must contain the
        # embedding state plus both decoder states (3 total).
        pipeline, _original_processor = make_official_pipeline(num_hidden_layers=2)
        with self.assertRaisesRegex(RuntimeError, "expected 3 hidden states"):
            MODULE.capture_pipeline_reference(
                pipeline,
                "a tiny lighthouse",
                max_sequence_length=8,
                device=torch.device("cpu"),
                source_dtype="float32",
            )

    def test_writer_rejects_incomplete_state_list_against_capture_expectation(self) -> None:
        with self.assertRaisesRegex(ValueError, "expected 4 hidden states"):
            MODULE.write_reference_bundle(
                **self.synthetic_reference() | {"expected_hidden_state_count": 4}
            )

    def test_capture_cleans_hooks_and_restores_processor_when_official_call_raises(self) -> None:
        pipeline, original_processor = make_official_pipeline(fail_on_call=True)
        with self.assertRaisesRegex(RuntimeError, "synthetic processor failure"):
            MODULE.capture_pipeline_reference(
                pipeline,
                "a tiny lighthouse",
                max_sequence_length=8,
                device=torch.device("cpu"),
                source_dtype="float32",
            )
        self.assertEqual(original_processor, pipeline.processor)
        self.assertEqual({}, pipeline.text_encoder._forward_hooks)
        self.assertEqual({}, pipeline.text_encoder.model.language_model.norm._forward_hooks)

    def test_serializes_little_endian_tensors_with_per_tensor_and_payload_checksums(self) -> None:
        manifest_path = MODULE.write_reference_bundle(**self.synthetic_reference())
        self.assertEqual("qwen_image21_text_reference.json", manifest_path.name)
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        payload = (self.output_dir / manifest["payload_file"]).read_bytes()

        self.assertEqual("qwen-image21-text-reference", manifest["schema"])
        self.assertEqual(1, manifest["schema_version"])
        self.assertEqual(MODEL_REVISION, manifest["model"]["revision_sha"])
        self.assertEqual("a tiny lighthouse", manifest["prompt"])
        self.assertEqual(8, manifest["sequence"]["max_sequence_length"])
        self.assertIn("guard only", manifest["sequence"]["max_sequence_length_semantics"])
        self.assertEqual(3, manifest["sequence"]["drop_idx"])
        self.assertEqual(3, manifest["sequence"]["actual_sequence_length"])
        self.assertTrue(manifest["embedding"]["pre_final_rmsnorm"])
        self.assertEqual(3, manifest["embedding"]["expected_hidden_state_count"])
        self.assertEqual(DIFFUSERS_COMMIT, manifest["runtime"]["diffusers_commit"])
        self.assertEqual(hashlib.sha256(payload).hexdigest(), manifest["payload_sha256"])

        tensors = manifest["tensors"]
        self.assertEqual(
            [
                "input_ids",
                "attention_mask",
                "mm_token_type_ids",
                "pre_final_rmsnorm_embeddings",
                "hidden_state_000",
                "hidden_state_001",
                "hidden_state_002",
            ],
            list(tensors),
        )
        input_ids = tensors["input_ids"]
        embeddings = tensors["pre_final_rmsnorm_embeddings"]
        first_layer = tensors["hidden_state_000"]
        self.assertEqual("int64-le", input_ids["dtype"])
        self.assertEqual([1, 6], input_ids["shape"])
        self.assertEqual("float32-le", embeddings["dtype"])
        self.assertEqual([1, 3, 4096], embeddings["shape"])
        self.assertEqual("float32-le", first_layer["dtype"])
        self.assertEqual([1, 6, 4096], first_layer["shape"])
        self.assertEqual(101, struct.unpack_from("<q", payload, input_ids["offset_bytes"])[0])
        self.assertEqual(2.0, struct.unpack_from("<f", payload, embeddings["offset_bytes"])[0])
        for name, descriptor in tensors.items():
            start = descriptor["offset_bytes"]
            end = start + descriptor["nbytes"]
            self.assertEqual(descriptor["sha256"], hashlib.sha256(payload[start:end]).hexdigest(), name)

    def test_rejects_context_over_limit_bad_drop_index_and_shape_mismatch(self) -> None:
        with self.assertRaisesRegex(ValueError, "max_sequence_length"):
            MODULE.write_reference_bundle(**self.synthetic_reference() | {"max_sequence_length": 2})
        with self.assertRaisesRegex(ValueError, "drop_idx"):
            MODULE.write_reference_bundle(**self.synthetic_reference() | {"drop_idx": 7})
        with self.assertRaisesRegex(ValueError, "shape"):
            MODULE.write_reference_bundle(
                **self.synthetic_reference()
                | {"attention_mask": np.ones((1, 5), dtype=np.int64)}
            )

    def test_refuses_to_overwrite_reference_files(self) -> None:
        MODULE.write_reference_bundle(**self.synthetic_reference())
        with self.assertRaisesRegex(FileExistsError, "already exists"):
            MODULE.write_reference_bundle(**self.synthetic_reference())


if __name__ == "__main__":
    unittest.main()
