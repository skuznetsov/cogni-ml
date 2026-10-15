"""Adversarial contract checks for the layer-39 MoE trace comparator."""

import copy
import hashlib
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "drama3" / "moe_compare.py"
SPEC = importlib.util.spec_from_file_location("drama3_moe_compare_under_test", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
moe_compare = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(moe_compare)


def write_json(path: Path, value: dict, *, allow_nan: bool = False) -> None:
    path.write_text(json.dumps(value, separators=(",", ":"), allow_nan=allow_nan) + "\n",
                    encoding="utf-8")


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def source_hashes() -> dict[str, str]:
    return {name: f"{index:064x}" for index, name in enumerate(
        sorted(moe_compare.REQUIRED_SOURCE_HASHES), 1
    )}


def rank_map(ids: list[int]) -> list[dict[str, int]]:
    return [{"rank": rank, "expert_id": expert_id}
            for rank, expert_id in enumerate(ids)]


def tensor(values: list[float], shape: list[int], capture_kind: str,
           expert_ids: list[int] | None = None) -> dict:
    result = {"shape": shape, "dtype": "float32", "values": values,
              "capture_kind": capture_kind}
    if expert_ids is not None:
        result["expert_ids"] = list(expert_ids)
    return result


def make_stages(ids: list[int], activation: list[float], *, native: bool,
                include_boundaries: bool = True) -> dict:
    expert_count = moe_compare.TOP_K
    gate_values = [float(expert_id) / 8.0 for expert_id in ids for _ in range(512)]
    up_values = [0.5 + float(expert_id) / 16.0 for expert_id in ids for _ in range(512)]
    activation_values = [0.25 + float(expert_id) / 32.0
                         for expert_id in ids for _ in range(512)]
    down_values = [0.125 + float(expert_id) / 64.0
                   for expert_id in ids for _ in range(moe_compare.INPUT_WIDTH)]
    weighted_values = [0.25 for _ in range(expert_count * moe_compare.INPUT_WIDTH)]
    gate_capture = "official_in_path" if native else "upstream_switch_mlp_gate_projection_return"
    up_capture = "official_in_path" if native else "upstream_switch_mlp_up_projection_return"
    activation_capture = "official_in_path" if native else "upstream_swiglu_return"
    down_capture = "official_in_path" if native else "upstream_switch_mlp_return"
    diagnostic = "diagnostic_auxiliary" if native else "diagnostic_recompute_sequential_add"
    ids_record = tensor([float(value) for value in ids], [expert_count],
                        "official_in_path" if native else "upstream_local")
    stages = {
        "normalized_input": tensor(list(activation), [moe_compare.INPUT_WIDTH],
                                   "official_in_path" if native else "upstream_sparse_argument"),
        "router_logits": tensor([0.0] * moe_compare.EXPERT_COUNT,
                                [moe_compare.EXPERT_COUNT],
                                "official_in_path" if native else "upstream_linear_return"),
        "expert_ids_argpartition_order": ids_record,
        "selected_weights": tensor([0.125] * expert_count, [expert_count],
                                   "official_in_path" if native else "upstream_local"),
        "expert_gate_outputs": tensor(gate_values, [expert_count, 512], gate_capture,
                                       None if native else ids),
        "expert_up_outputs": tensor(up_values, [expert_count, 512], up_capture,
                                     None if native else ids),
        "expert_activation_outputs": tensor(activation_values, [expert_count, 512], activation_capture,
                                             None if native else ids),
        "expert_down_outputs": tensor(down_values,
                                      [expert_count, moe_compare.INPUT_WIDTH], down_capture,
                                      None if native else ids),
        "expert_weighted_outputs": tensor(weighted_values,
                                          [expert_count, moe_compare.INPUT_WIDTH],
                                          diagnostic if native else "upstream_sum_operand",
                                          None if native else ids),
        "routed_sum_official": tensor([2.0] * moe_compare.INPUT_WIDTH,
                                      [moe_compare.INPUT_WIDTH],
                                      "official_in_path" if native else "upstream_sum_return"),
        "diagnostic_prefix_sums_argpartition_order": tensor(
            [0.0] * (expert_count * moe_compare.INPUT_WIDTH),
            [expert_count, moe_compare.INPUT_WIDTH], diagnostic, ids),
        "shared_gate_logits": tensor([0.0], [1],
                                     "official_in_path" if native else "upstream_linear_return"),
        "shared_expert_output": tensor([1.0] * moe_compare.INPUT_WIDTH,
                                       [moe_compare.INPUT_WIDTH],
                                       "official_in_path" if native else "upstream_local"),
        "shared_weighted_output": tensor([0.5] * moe_compare.INPUT_WIDTH,
                                         [moe_compare.INPUT_WIDTH],
                                         diagnostic if native else "upstream_return_local"),
        "moe_output_official": tensor([2.5] * moe_compare.INPUT_WIDTH,
                                      [moe_compare.INPUT_WIDTH],
                                      "official_in_path" if native else "upstream_return"),
    }
    if not native:
        stages["router_softmax"] = tensor([1.0 / moe_compare.EXPERT_COUNT] * moe_compare.EXPERT_COUNT,
                                          [moe_compare.EXPERT_COUNT], "upstream_local")
        stages["selected_probabilities"] = tensor([0.125] * expert_count, [expert_count],
                                                  "upstream_local")
        stages["shared_gate_sigmoid"] = tensor([0.5], [1], "upstream_sigmoid_return")
    if include_boundaries:
        stages["attention_residual"] = tensor([3.0] * moe_compare.INPUT_WIDTH,
                                               [moe_compare.INPUT_WIDTH],
                                               "official_in_path" if native else "upstream_decoder_local")
        stages["hidden_output"] = tensor([0.75] * moe_compare.INPUT_WIDTH,
                                         [moe_compare.INPUT_WIDTH],
                                         "official_in_path" if native else
                                         "upstream_target_decoder_output")
    return stages


def capture_evidence(runs: list[dict]) -> dict:
    aggregate_names = set(moe_compare.EXPERT_MATRIX_STAGES) | {
        "diagnostic_prefix_sums_argpartition_order"
    }
    stage_values = sum(len(stage["values"])
                       for run in runs for stage in run["stages"].values())
    stage_read_calls = sum(
        (moe_compare.TOP_K if stage["shape"][0] == moe_compare.TOP_K and
         len(stage["shape"]) == 2 and stage_name in aggregate_names else 1)
        for run in runs for stage_name, stage in run["stages"].items()
    )
    count = 4
    compared_values = 4 * moe_compare.INPUT_WIDTH
    capture_reads = 2 * count
    graph_before = {"depth": 2, "submitted": 0, "completed": 0, "pending": 0,
                    "max_pending": 0, "completion_waits": 0, "retained_bytes": 0,
                    "max_retained_bytes": 0}
    graph_after = {"depth": 2, "submitted": 1, "completed": 1, "pending": 0,
                   "max_pending": 1, "completion_waits": 1, "retained_bytes": 0,
                   "max_retained_bytes": 1}
    return {
        "capture_control": {"passed": True, "comparison": "Float32 bitwise equality",
                            "count": count, "compared_values": compared_values,
                            "mismatches": []},
        "graph_before_drain": graph_before,
        "graph_after_drain": graph_after,
        "graph_after_serialization": dict(graph_after),
        "readbacks": {
            "control": 4 * (stage_read_calls + capture_reads),
            "terminal": 4 * (stage_values + 2 * compared_values),
        },
    }


def fixture(directory: Path, *, native_route_order: list[int] | None = None,
            native_rows_follow_ids: bool = True, full_input_matches: bool = True
            ) -> tuple[Path, Path, Path, Path]:
    hashes = source_hashes()
    activation = [0.125] * moe_compare.INPUT_WIDTH
    fixture_bundle = {
        "format": moe_compare.INPUT_FORMAT,
        "coordinate": moe_compare.COORDINATE,
        "activation_name": moe_compare.INPUT_NAME,
        "activation_shape": [moe_compare.INPUT_WIDTH],
        "activation_dtype": "float32",
        "activation": activation,
        "activation_sha256": moe_compare.canonical_f32_sha256(activation),
        "model": {"repository": moe_compare.MODEL_REPOSITORY,
                  "revision": moe_compare.MODEL_REVISION},
        "upstream": {"repository": moe_compare.UPSTREAM_REPOSITORY,
                     "revision": moe_compare.UPSTREAM_REVISION},
        "source_hashes": hashes,
    }
    fixture_path = directory / "layer39-input.json"
    write_json(fixture_path, fixture_bundle)
    fixture_digest = hashlib.sha256(fixture_path.read_bytes()).hexdigest()
    input_record = {
        "format": moe_compare.INPUT_FORMAT,
        "path": fixture_path.name,
        "sha256": fixture_digest,
        "activation_sha256": fixture_bundle["activation_sha256"],
        "shape": [moe_compare.INPUT_WIDTH],
        "dtype": "float32",
        "activation_name": moe_compare.INPUT_NAME,
    }
    model_config = {
        "hidden_size": moe_compare.INPUT_WIDTH,
        "moe_intermediate_size": moe_compare.INTERMEDIATE_WIDTH,
        "num_experts": moe_compare.EXPERT_COUNT,
        "num_experts_per_tok": moe_compare.TOP_K,
    }
    upstream_ids = list(range(moe_compare.TOP_K))
    upstream = {
        "format": moe_compare.UPSTREAM_FORMAT,
        "mode": "full_run",
        "coordinate": moe_compare.COORDINATE,
        "model": fixture_bundle["model"],
        "upstream": fixture_bundle["upstream"],
        "model_config": model_config,
        "source_hashes": hashes,
        "input": input_record,
        "expert_rank_to_id": rank_map(upstream_ids),
        "stages": make_stages(upstream_ids, activation, native=False),
    }
    upstream_path = directory / "upstream.json"
    write_json(upstream_path, upstream)

    native_ids = native_route_order or upstream_ids
    if native_rows_follow_ids:
        native_stages = make_stages(native_ids, activation, native=True)
    else:
        native_stages = make_stages(upstream_ids, activation, native=True)
        native_stages["expert_ids_argpartition_order"]["values"] = [float(v) for v in native_ids]
        native_stages["selected_weights"]["values"] = [0.125] * len(native_ids)
    full_activation = activation if full_input_matches else [0.25] + activation[1:]
    full_stages = make_stages(native_ids, full_activation, native=True)
    native_runs = [
        {"name": "full_native", "stages": full_stages,
         "expert_rank_to_id": rank_map(native_ids)},
        {"name": "isolated_upstream_input", "stages": native_stages,
         "expert_rank_to_id": rank_map(native_ids)},
    ]
    native = {
        "format": moe_compare.NATIVE_FORMAT,
        "coordinate": moe_compare.COORDINATE,
        "model": fixture_bundle["model"],
        "upstream": fixture_bundle["upstream"],
        "source_hashes": {
            "native": {name: f"{index + 100:064x}" for index, name in enumerate(
                ("backbone_resident.cr", "backbone_resident.metal", "full_inputs",
                 "upstream_input_fixture", "drama3_moe_provenance.cr"))},
            "fixture": hashes,
        },
        "model_weight_provenance": {
            "target_layer": moe_compare.COORDINATE["layer"],
            "target_moe_weights_sha256": hashes["target_moe_weights"],
            "tensor_count": 1536,
            "model_path": "/models/drama3-pinned",
            "config_sha256": hashes["config.json"],
            "index_sha256": hashes["model.safetensors.index.json"],
            "verified_before_gpu": True,
        },
        "input_fixture": {"path": fixture_path.name, "sha256": fixture_digest,
                          "activation_sha256": fixture_bundle["activation_sha256"]},
        "execution": {"batch_experts_requested": False,
                      "full_native_batch_path": False,
                      "isolated_batch_path": False,
                      "dense_reduction": "tree"},
        "runs": native_runs,
    }
    native.update(capture_evidence(native_runs))
    native_path = directory / "native.json"
    write_json(native_path, native)
    return upstream_path, native_path, fixture_path, directory


class Drama3MoeCompareControls(unittest.TestCase):
    def test_accepts_pinned_real_shape_traces_and_reports_diagnostics(self):
        with tempfile.TemporaryDirectory() as temporary:
            upstream, native, _, _ = fixture(Path(temporary))
            report = moe_compare.compare(upstream, native)

        self.assertTrue(report["qualified"])
        self.assertEqual(report["status"], "match")
        self.assertEqual(report["isolated_upstream_input"]["stages"]["routed_sum_official"]["status"],
                         "match")
        self.assertEqual(report["isolated_upstream_input"]["stages"]["expert_gate_outputs"]["status"],
                         "match")
        self.assertEqual(report["isolated_upstream_input"]["diagnostic_prefix_reductions"]
                         ["official_sum_stage"], "routed_sum_official")
        self.assertEqual(report["qualification_scope"],
                         "full_native_and_isolated_exact_input_replay")
        self.assertEqual(report["native_weight_provenance"], {
            "verified_before_gpu": True,
            "target_layer": moe_compare.COORDINATE["layer"],
            "tensor_count": 1536,
            "source_hashes_match_fixture": True,
        })

    def test_requires_loaded_weight_provenance_to_match_fixture_before_gpu(self):
        mutations = (
            ("missing record", lambda bundle: bundle.pop("model_weight_provenance"),
             "missing actual loaded-weight evidence"),
            ("not verified before GPU", lambda bundle: bundle["model_weight_provenance"].__setitem__(
                "verified_before_gpu", False), "must be true"),
            ("wrong target digest", lambda bundle: bundle["model_weight_provenance"].__setitem__(
                "target_moe_weights_sha256", "f" * 64),
             "does not match fixture source hash target_moe_weights"),
            ("wrong config digest", lambda bundle: bundle["model_weight_provenance"].__setitem__(
                "config_sha256", "f" * 64), "does not match fixture source hash config.json"),
            ("wrong index digest", lambda bundle: bundle["model_weight_provenance"].__setitem__(
                "index_sha256", "f" * 64),
             "does not match fixture source hash model.safetensors.index.json"),
            ("wrong layer", lambda bundle: bundle["model_weight_provenance"].__setitem__(
                "target_layer", 38), "expected layer 39"),
            ("empty tensor set", lambda bundle: bundle["model_weight_provenance"].__setitem__(
                "tensor_count", 0), "positive tensor count"),
            ("empty model path", lambda bundle: bundle["model_weight_provenance"].__setitem__(
                "model_path", " "), "expanded absolute model path"),
            ("relative model path", lambda bundle: bundle["model_weight_provenance"].__setitem__(
                "model_path", "models/drama3"), "expanded absolute model path"),
            ("unexpected provenance field", lambda bundle: bundle["model_weight_provenance"].__setitem__(
                "digest_source", "weight_store"), "unexpected evidence fields"),
        )
        for name, mutate, error in mutations:
            with self.subTest(name=name), tempfile.TemporaryDirectory() as temporary:
                upstream, native, _, _ = fixture(Path(temporary))
                candidate = load_json(native)
                mutate(candidate)
                write_json(native, candidate)
                with self.assertRaisesRegex(ValueError, error):
                    moe_compare.compare(upstream, native)

    def test_rejects_native_fixture_hashes_that_do_not_match_shared_fixture(self):
        with tempfile.TemporaryDirectory() as temporary:
            upstream, native, _, _ = fixture(Path(temporary))
            candidate = load_json(native)
            candidate["source_hashes"]["fixture"]["target_moe_weights"] = "f" * 64
            write_json(native, candidate)
            with self.assertRaisesRegex(ValueError,
                                       "native trace fixture source hashes differ"):
                moe_compare.compare(upstream, native)

    def test_rejects_legacy_flat_native_provenance(self):
        with tempfile.TemporaryDirectory() as temporary:
            upstream, native, _, _ = fixture(Path(temporary))
            candidate = load_json(native)
            candidate["source_hashes"] = candidate["source_hashes"]["native"]
            write_json(native, candidate)
            with self.assertRaisesRegex(ValueError, "expected native and fixture maps"):
                moe_compare.compare(upstream, native)

    def test_requires_real_four_vector_capture_control(self):
        mutations = (
            ("missing", lambda bundle: bundle.pop("capture_control"),
             "native.capture_control: missing evidence"),
            ("failed", lambda bundle: bundle["capture_control"].__setitem__("passed", False),
             "expected four bitwise-equal"),
            ("short comparison", lambda bundle: bundle["capture_control"].__setitem__(
                "compared_values", 8191), "expected four bitwise-equal"),
            ("mismatch list", lambda bundle: bundle["capture_control"].__setitem__(
                "mismatches", [33]), "expected four bitwise-equal"),
        )
        for name, mutate, error in mutations:
            with self.subTest(name=name), tempfile.TemporaryDirectory() as temporary:
                upstream, native, _, _ = fixture(Path(temporary))
                candidate = load_json(native)
                mutate(candidate)
                write_json(native, candidate)
                with self.assertRaisesRegex(ValueError, error):
                    moe_compare.compare(upstream, native)

    def test_requires_one_final_graph_drain_and_no_serialization_drift(self):
        mutations = (
            ("pre-drain wait", lambda bundle: bundle["graph_before_drain"].__setitem__(
                "completion_waits", 1), "expected idle depth-2 graph"),
            ("extra submission", lambda bundle: bundle["graph_after_drain"].__setitem__(
                "submitted", 2), "expected exactly one final submit"),
            ("serialization wait", lambda bundle: bundle["graph_after_serialization"].__setitem__(
                "completion_waits", 2), "graph changed during serialization"),
        )
        for name, mutate, error in mutations:
            with self.subTest(name=name), tempfile.TemporaryDirectory() as temporary:
                upstream, native, _, _ = fixture(Path(temporary))
                candidate = load_json(native)
                mutate(candidate)
                write_json(native, candidate)
                with self.assertRaisesRegex(ValueError, error):
                    moe_compare.compare(upstream, native)

    def test_rejects_extra_or_missing_terminal_reads(self):
        mutations = (
            ("extra status read", lambda bundle: bundle["readbacks"].__setitem__(
                "control", bundle["readbacks"]["control"] + 4),
             "readbacks.control: expected"),
            ("extra tensor read", lambda bundle: bundle["readbacks"].__setitem__(
                "terminal", bundle["readbacks"]["terminal"] + 4),
             "readbacks.terminal: expected"),
            ("missing ledger", lambda bundle: bundle.pop("readbacks"),
             "native.readbacks: expected"),
        )
        for name, mutate, error in mutations:
            with self.subTest(name=name), tempfile.TemporaryDirectory() as temporary:
                upstream, native, _, _ = fixture(Path(temporary))
                candidate = load_json(native)
                mutate(candidate)
                write_json(native, candidate)
                with self.assertRaisesRegex(ValueError, error):
                    moe_compare.compare(upstream, native)

    def test_reports_full_native_input_drift_without_failing_exact_input_replay(self):
        with tempfile.TemporaryDirectory() as temporary:
            upstream, native, _, _ = fixture(Path(temporary), full_input_matches=False)
            report = moe_compare.compare(upstream, native)

        self.assertTrue(report["qualified"])
        self.assertEqual(report["full_native"]["status"], "different_input")
        self.assertEqual(report["qualification_scope"],
                         "isolated_exact_input_replay_only; full_native_original_input_differs")
        self.assertFalse(report["full_native_original_discrepancy_qualified"])
        self.assertTrue(report["isolated_upstream_input"]["input_bitwise_equal"])

    def test_reindexes_correctly_when_native_expert_rank_order_differs(self):
        reordered = list(reversed(range(moe_compare.TOP_K)))
        with tempfile.TemporaryDirectory() as temporary:
            upstream, native, _, _ = fixture(Path(temporary), native_route_order=reordered)
            report = moe_compare.compare(upstream, native)

        self.assertTrue(report["qualified"])
        order = report["isolated_upstream_input"]["expert_order"]
        self.assertTrue(order["same_expert_set"])
        self.assertFalse(order["same_order"])
        self.assertEqual(report["isolated_upstream_input"]["stages"]
                         ["expert_gate_outputs"]["status"], "match")

    def test_rejects_positional_expert_rows_disguised_by_a_permuted_id_order(self):
        reordered = list(reversed(range(moe_compare.TOP_K)))
        with tempfile.TemporaryDirectory() as temporary:
            upstream, native, _, _ = fixture(
                Path(temporary), native_route_order=reordered,
                native_rows_follow_ids=False,
            )
            report = moe_compare.compare(upstream, native)

        self.assertFalse(report["qualified"])
        self.assertEqual(report["isolated_upstream_input"]["first_failing_stage"],
                         "expert_gate_outputs")
        self.assertEqual(report["isolated_upstream_input"]["stages"]
                         ["expert_gate_outputs"]["status"], "mismatch")

    def test_rejects_candidate_routed_sum_outside_fixed_tolerance(self):
        with tempfile.TemporaryDirectory() as temporary:
            upstream, native, _, _ = fixture(Path(temporary))
            candidate = load_json(native)
            stage = next(run for run in candidate["runs"]
                         if run["name"] == "isolated_upstream_input")
            stage = stage["stages"]["routed_sum_official"]
            stage["values"][33] += 0.01
            write_json(native, candidate)
            report = moe_compare.compare(upstream, native)

        self.assertFalse(report["qualified"])
        routed = report["isolated_upstream_input"]["stages"]["routed_sum_official"]
        self.assertEqual(routed["status"], "mismatch")
        self.assertGreater(routed["failures"], 0)

    def test_rejects_authoritative_stage_mislabeled_auxiliary(self):
        with tempfile.TemporaryDirectory() as temporary:
            upstream, native, _, _ = fixture(Path(temporary))
            candidate = load_json(native)
            run = next(run for run in candidate["runs"]
                       if run["name"] == "isolated_upstream_input")
            run["stages"]["expert_gate_outputs"]["capture_kind"] = "diagnostic_auxiliary"
            run["stages"]["expert_gate_outputs"]["values"][0] += 1.0
            write_json(native, candidate)
            with self.assertRaisesRegex(ValueError, "expected capture_kind"):
                moe_compare.compare(upstream, native)

    def test_rejects_old_shared_weighted_capture_frame(self):
        with tempfile.TemporaryDirectory() as temporary:
            upstream, native, _, _ = fixture(Path(temporary))
            candidate = load_json(upstream)
            candidate["stages"]["shared_weighted_output"]["capture_kind"] = "upstream_local"
            write_json(upstream, candidate)
            with self.assertRaisesRegex(ValueError, "expected capture_kind"):
                moe_compare.compare(upstream, native)

    def test_rejects_widened_tolerance(self):
        with tempfile.TemporaryDirectory() as temporary:
            upstream, native, _, _ = fixture(Path(temporary))
            with self.assertRaisesRegex(ValueError, "tolerances may only be tightened"):
                moe_compare.compare(upstream, native, atol=moe_compare.DEFAULT_ATOL * 2)

    def test_rejects_stale_fixture_bytes(self):
        with tempfile.TemporaryDirectory() as temporary:
            upstream, native, fixture_path, _ = fixture(Path(temporary))
            stale = load_json(fixture_path)
            stale["activation"][0] = 0.5
            write_json(fixture_path, stale)
            with self.assertRaisesRegex(ValueError, "stale fixture bytes"):
                moe_compare.compare(upstream, native)

    def test_rejects_wrong_model_and_upstream_revisions(self):
        for field, replacement in (("model", "0" * 40), ("upstream", "1" * 40)):
            with self.subTest(field=field), tempfile.TemporaryDirectory() as temporary:
                upstream, native, _, _ = fixture(Path(temporary))
                bundle = load_json(upstream)
                metadata = bundle[field]
                key = "revision"
                metadata[key] = replacement
                write_json(upstream, bundle)
                with self.assertRaisesRegex(ValueError, "wrong (model|upstream) metadata"):
                    moe_compare.compare(upstream, native)

    def test_rejects_wrong_model_config(self):
        with tempfile.TemporaryDirectory() as temporary:
            upstream, native, _, _ = fixture(Path(temporary))
            bundle = load_json(upstream)
            bundle["model_config"]["moe_intermediate_size"] = 256
            write_json(upstream, bundle)
            with self.assertRaisesRegex(ValueError, "expected pinned value 512"):
                moe_compare.compare(upstream, native)

    def test_rejects_duplicate_route_ids_and_incomplete_rank_mapping(self):
        mutations = (
            ("route", lambda run: run["stages"]["expert_ids_argpartition_order"]["values"].__setitem__(7, 0),
             "duplicate or missing expert IDs"),
            ("rank map", lambda run: run["expert_rank_to_id"][7].__setitem__("expert_id", 6),
             "duplicate, missing, or unexpected expert IDs"),
        )
        for name, mutate, message in mutations:
            with self.subTest(name=name), tempfile.TemporaryDirectory() as temporary:
                upstream, native, _, _ = fixture(Path(temporary))
                candidate = load_json(native)
                run = next(run for run in candidate["runs"]
                           if run["name"] == "isolated_upstream_input")
                mutate(run)
                write_json(native, candidate)
                with self.assertRaisesRegex(ValueError, message):
                    moe_compare.compare(upstream, native)

    def test_rejects_nonfinite_stage_values_and_wrong_isolated_input(self):
        with tempfile.TemporaryDirectory() as temporary:
            upstream, native, _, _ = fixture(Path(temporary))
            candidate = load_json(native)
            stage = next(run for run in candidate["runs"]
                         if run["name"] == "isolated_upstream_input")["stages"]["routed_sum_official"]
            stage["values"][0] = float("nan")
            write_json(native, candidate, allow_nan=True)
            with self.assertRaisesRegex(ValueError, "non-finite"):
                moe_compare.compare(upstream, native)

        with tempfile.TemporaryDirectory() as temporary:
            upstream, native, _, _ = fixture(Path(temporary))
            candidate = load_json(native)
            run = next(run for run in candidate["runs"]
                       if run["name"] == "isolated_upstream_input")
            run["stages"]["normalized_input"]["values"][0] = 0.5
            write_json(native, candidate)
            with self.assertRaisesRegex(ValueError, "isolated replay input differs"):
                moe_compare.compare(upstream, native)

    def test_upstream_full_isolated_control_is_bitwise_for_official_sums(self):
        for stage_name in ("routed_sum_official", "shared_weighted_output",
                           "moe_output_official", "hidden_output"):
            with self.subTest(stage=stage_name), tempfile.TemporaryDirectory() as temporary:
                upstream, native, _, directory = fixture(Path(temporary))
                full_reference = load_json(upstream)
                isolated = copy.deepcopy(full_reference)
                isolated["mode"] = "isolated_replay"
                isolated["stages"][stage_name]["values"][33] += 1e-6
                isolated_path = directory / "upstream-isolated.json"
                write_json(isolated_path, isolated)
                report = moe_compare.compare(upstream, native,
                                             upstream_isolated_path=isolated_path)

                self.assertFalse(report["qualified"])
                control = report["upstream_full_vs_isolated_control"]
                self.assertEqual(control["status"], "mismatch")
                self.assertIn(stage_name, control["failure_stages"])


if __name__ == "__main__":
    unittest.main()
