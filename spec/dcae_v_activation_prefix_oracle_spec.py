"""CPU-only contract tests for the bounded upstream activation-prefix oracle."""

import importlib.util
from pathlib import Path
import unittest


SOURCE = (
    Path(__file__).resolve().parents[1]
    / "examples/support/dcae_v_activation_prefix_oracle.py"
)
SPEC = importlib.util.spec_from_file_location("dcae_v_activation_prefix_oracle", SOURCE)
oracle = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(oracle)


class PrefixContractSpec(unittest.TestCase):
    def test_manifest_fields_and_export_names_match_root_comparator(self):
        fields = oracle.activation_prefix_fields()
        self.assertEqual(
            fields,
            {
                "reference_path": str(oracle.DEFAULT_REFERENCE),
                "reference_sha256": oracle.REFERENCE_SHA256,
                "target_conv": "decoder.stages.2.op_list.1.main.conv1.conv",
                "tile_count": 2,
                "activation_shape": [1, 512, 40, 72, 128],
                "cache_shape": [1, 512, 1, 72, 128],
            },
        )
        self.assertEqual(
            [record["name"] for record in oracle.terminal_inventory()],
            [
                "invalid-input.bf16",
                "invalid-cache.bf16",
                "valid-input.bf16",
                "valid-cache.bf16",
            ],
        )

    def test_fixed_bcthw_admission_rejects_wrong_geometry_and_bool_dimensions(self):
        self.assertEqual(
            oracle.validate_bcthw_shape(
                (1, 512, 40, 72, 128), oracle.ACTIVATION_SHAPE, "activation"
            ),
            oracle.ACTIVATION_SHAPE,
        )
        with self.assertRaisesRegex(oracle.OracleError, "pinned boundary"):
            oracle.validate_bcthw_shape(
                (1, 512, 40, 72, 127), oracle.ACTIVATION_SHAPE, "activation"
            )
        with self.assertRaisesRegex(oracle.OracleError, "positive integer"):
            oracle.validate_bcthw_shape(
                (True, 512, 40, 72, 128), oracle.ACTIVATION_SHAPE, "activation"
            )

    def test_tile_zero_seed_and_export_follow_upstream_old_cache_layout(self):
        try:
            import torch
        except ImportError as exc:
            self.skipTest(f"PyTorch is unavailable for the CPU tensor contract: {exc}")
        activation = (
            torch.arange(80, dtype=torch.float32)
            .reshape(1, 2, 40, 1, 1)
            .to(torch.bfloat16)
        )
        internal = oracle._target_internal_cache(
            torch,
            activation,
            chunk_length=40,
            temporal_pad=1,
            expected_shape=(1, 2, 40, 1, 1),
        )
        self.assertEqual(tuple(internal.shape), (1, 1, 2, 1, 1, 1))
        self.assertEqual(internal[:, 0, :, 0, 0, 0].tolist(), [[39.0, 79.0]])
        external = oracle._target_cache_bcthw(
            torch, internal, expected_shape=(1, 2, 1, 1, 1)
        )
        self.assertEqual(tuple(external.shape), (1, 2, 1, 1, 1))
        self.assertEqual(external[:, :, 0, 0, 0].tolist(), [[39.0, 79.0]])

    def test_only_weights_before_target_entry_are_selected(self):
        names = {
            "decoder.project_in.conv.weight": object(),
            "decoder.stages.6.op_list.0.main.conv1.conv.weight": object(),
            "decoder.stages.2.op_list.0.main.conv.conv.weight": object(),
            "decoder.stages.2.op_list.1.main.conv1.conv.weight": object(),
            "decoder.stages.1.op_list.0.main.conv.weight": object(),
            "encoder.project_in.conv.weight": object(),
        }
        selected = oracle._selected_state_names(names)
        self.assertEqual(
            selected,
            (
                "decoder.project_in.conv.weight",
                "decoder.stages.6.op_list.0.main.conv1.conv.weight",
                "decoder.stages.2.op_list.0.main.conv.conv.weight",
            ),
        )


if __name__ == "__main__":
    unittest.main()
