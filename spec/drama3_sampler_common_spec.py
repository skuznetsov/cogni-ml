#!/usr/bin/env python3
"""CPU-only contract tests for common-noise sampler diagnostics."""

from __future__ import annotations

import hashlib
import copy
import math
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts" / "drama3"))

import compare_sampler_common as common
import sampler_common_reference as mlx_reference


def f32(value: float) -> float:
    import struct

    return struct.unpack("<f", struct.pack("<f", value))[0]


def algorithm(case: dict, top_ids: list[int]) -> dict:
    scaled = [f32(value / case["temperature"]) for value in case["logits"]]
    maximum = max(scaled)
    weights = [f32(math.exp(f32(value - maximum))) for value in scaled]
    denominator = 0.0
    for weight in weights:
        denominator = f32(denominator + weight)
    full_p = [f32(weights[token] / denominator) for token in top_ids]
    full_cdf: list[float] = []
    cumulative = 0.0
    for probability in full_p:
        cumulative = f32(cumulative + probability)
        full_cdf.append(cumulative)
    kept = [(rank == 0) or (full_cdf[rank] <= case["top_p"])
            for rank in range(case["top_k"])]
    kept_weights = [weights[token] for token, keep in zip(top_ids, kept) if keep]
    retained = 0.0
    for weight in kept_weights:
        retained = f32(retained + weight)
    conditional = [f32(weights[token] / retained) if keep else 0.0
                   for token, keep in zip(top_ids, kept)]
    conditional_cdf: list[float] = []
    cumulative = 0.0
    for probability, keep in zip(conditional, kept):
        if keep:
            cumulative = f32(cumulative + probability)
            conditional_cdf.append(cumulative)
        else:
            conditional_cdf.append(0.0)
    kept_ids = [token for token, keep in zip(top_ids, kept) if keep]
    samples = []
    for draw in case["draws"]:
        if draw == 1.0:
            samples.append(kept_ids[-1])
        else:
            rank = next(i for i, value in enumerate(conditional_cdf[:len(kept_ids)])
                        if value > draw)
            samples.append(kept_ids[rank])
    gumbel_ids = []
    for row in case["noise_uniforms"]:
        scores = []
        for rank, (token, keep, uniform) in enumerate(zip(top_ids, kept, row)):
            if keep:
                score = f32(scaled[token] + f32(-math.log(-math.log(uniform))))
                scores.append((score, -rank, token))
        gumbel_ids.append(max(scores)[2])
    return {
        "sample_ids": samples,
        "topk_ids": top_ids,
        "topk_scaled_logits": [scaled[token] for token in top_ids],
        "kept_mask": kept,
        "kept_count": sum(kept),
        "full_vocab_normalization": denominator,
        "full_probabilities": full_p,
        "full_cdf": full_cdf,
        "retained_conditional_probabilities": conditional,
        "retained_conditional_cdf": conditional_cdf,
        "shared_gumbel_ids": gumbel_ids,
    }


def source_sha(relative: str) -> str:
    return hashlib.sha256((ROOT / relative).read_bytes()).hexdigest()


def make_fixture(*, tie: bool = False) -> dict:
    if tie:
        logits = [10.0, 10.0, 10.0, 10.0, 10.0, 0.0, 0.0]
        top_k = 4
        top_p = 1.0
        name = "synthetic.topk_boundary_tie"
        kind = "synthetic_topk_boundary_tie"
        arm = "synthetic"
        step = None
        expected_native = [0, 1, 2, 3]
        expected_mlx = [0, 1, 2, 4]
    else:
        logits = [float(32 - index) for index in range(32)]
        top_k = common.NORMAL[1]
        top_p = 0.9
        name = "synthetic.unit_vector"
        kind = "synthetic_sampler_vector"
        arm = "synthetic"
        step = None
        expected_native = expected_mlx = list(range(top_k))
    temperature = 1.0 if tie else common.NORMAL[0]
    case = common._case(name=name, role="main", source_arm=arm, step=step,
                        case_kind=kind, logits=[f32(value) for value in logits],
                        settings=(temperature, top_k, top_p))
    case = common._validate_case_input(case, "test-case")
    current = source_sha("src/ml/audio/drama3/kernels/resident.metal")
    fixture = {
        "format": common.FIXTURE_FORMAT,
        "provenance": {
            "native_sampler_source_sha256": current,
            "seeded_rng_parity_claimed": False,
        },
        "sampling_contract": common._canonical_sampling_contract(),
        "input_logit_comparisons": [],
        "pairs": [],
        "cases": [case],
    }
    return fixture, case, expected_native, expected_mlx


def make_probe(fixture: dict, case: dict, native_ids: list[int], mlx_ids: list[int], *,
               additional: tuple[dict, list[int], list[int]] | list[tuple[dict, list[int], list[int]]] | None = None) -> dict:
    entries = [(case, native_ids, mlx_ids)]
    if additional is not None:
        if isinstance(additional, tuple):
            entries.append(additional)
        else:
            entries.extend(additional)
    case_outputs = []
    diagnostic_bytes = 0
    terminal_bytes = 0
    for current, current_native_ids, current_mlx_ids in entries:
        case_output = {
            key: current[key] for key in ("name", "role", "source_arm", "step", "case_kind",
                                          "temperature", "top_k", "top_p", "draws")
        }
        case_output.update({f"native_{key}": value
                            for key, value in algorithm(current, current_native_ids).items()})
        case_output.update({f"mlx_{key}": value
                            for key, value in algorithm(current, current_mlx_ids).items()})
        k = current["top_k"]
        draws = len(current["draws"])
        diagnostic_bytes += (7 * k + 3 + draws) * 4
        terminal_bytes += draws * 4
        case_outputs.append(case_output)
    reads = {"diagnostic": diagnostic_bytes, "terminal": terminal_bytes,
             "control": 2 * len(entries) * 4}
    return {
        "format": common.PROBE_FORMAT,
        "fixture_sha256": "0" * 64,
        "sampler_kernel": "dr3_sample",
        "diagnostic_kernel": "dr3_sampler_compare",
        "resident_metal_source_sha256": source_sha("src/ml/audio/drama3/kernels/resident.metal"),
        "diagnostic_metal_source_sha256": source_sha("examples/support/drama3_sampler_compare.metal"),
        "cases": case_outputs,
        "queue": {
            "graph_before_work": {"depth": 2, "submitted": 0, "completed": 0,
                                   "pending": 0, "completion_waits": 0},
            "graph_before_final_drain": {"depth": 2, "submitted": 1, "completed": 0,
                                          "pending": 1, "completion_waits": 0},
            "graph_after_final_drain": {"depth": 2, "submitted": 1, "completed": 1,
                                        "pending": 0, "completion_waits": 1},
            "graph_after_terminal_reads": {"depth": 2, "submitted": 1, "completed": 1,
                                           "pending": 0, "completion_waits": 1},
            "readbacks_before_final_drain": {"diagnostic": 0, "terminal": 0, "control": 0},
            "readbacks_after_terminal_reads": reads,
            "submissions_before_final_drain": 1,
            "predrain_waits": 0,
            "final_drain_waits": 1,
            "diagnostic_bytes": diagnostic_bytes,
            "terminal_bytes": terminal_bytes,
            "finite_status_reads_after_drain": 2 * len(entries),
            "finite_status_readback_bytes": 8 * len(entries),
        },
    }


def write_json(path: Path, value: dict) -> str:
    payload = common._json_bytes(value)
    path.write_bytes(payload)
    return hashlib.sha256(payload).hexdigest()


def make_source_bound_fixture(directory: Path) -> dict:
    """A tiny valid trace-backed fixture for provenance mutation tests."""
    model_config = {
        "hidden_size": 4,
        "main_vocab_size": 4097,
        "tower_dim": 4,
        "tower_vocab_size": 4096,
        "num_codebooks": 10,
    }
    prompt_ids = [42]
    frames = [[250_000, *([0] * 10)]]
    source_report_value = {
        "model_path": str(directory),
        "prompt_ids": prompt_ids,
        "prompt_rows": len(prompt_ids),
        "frames": frames,
        "seed": 1,
        "temperature": common.NORMAL[0],
        "top_k": common.NORMAL[1],
        "top_p": common.NORMAL[2],
    }
    report_path = directory / "source-report.json"
    report_sha = write_json(report_path, source_report_value)
    report_meta = {"path": str(report_path.resolve()), "sha256": report_sha}
    source_report_provenance = {
        **report_meta,
        "value": source_report_value,
    }
    model_files = {"config.json": "a" * 64, "model.safetensors.index.json": "b" * 64}
    logits = [f32(32.0 - index) for index in range(4097)]
    source_refs = {}
    for arm in ("native", "upstream"):
        trace = {
            "format": common.TRACE_FORMAT,
            "source_report": report_meta,
            "prompt_ids": prompt_ids,
            "prompt_rows": len(prompt_ids),
            "frames": frames,
            "frame_count": len(frames),
            "model_config": model_config,
            "model_files": model_files,
            "execution": ({"device": "Device(gpu, 0)",
                           "float_checkpoint_weights_promoted_to_float32": True}
                          if arm == "upstream" else {"device": "Metal"}),
            "capture_steps": [0],
            "traces": [{
                "name": "generation.step0.main_logits",
                "shape": [len(logits)],
                "dtype": "float32",
                "values": logits,
            }],
        }
        if arm == "upstream":
            trace.update({
                "model": common.PINNED_MODEL,
                "upstream": common.PINNED_UPSTREAM,
                "source_hashes": {"fish_omni_tower": common.PINNED_TOWER_SHA256},
                "generation_parameters": {"seed": 1, "temperature": common.NORMAL[0],
                                           "top_k": common.NORMAL[1], "top_p": common.NORMAL[2]},
            })
        trace_path = directory / f"{arm}-trace.json"
        trace_sha = write_json(trace_path, trace)
        source_refs[arm] = {"path": str(trace_path.resolve()), "sha256": trace_sha}

    cases = []
    pairs = []
    for call, settings in (("main", common.NORMAL), ("main_retry", common.RETRY)):
        names = {}
        for arm in ("native", "upstream"):
            name = f"{arm}.step0.{call}"
            names[arm] = name
            cases.append(common._case(
                name=name, role="main", source_arm=arm, step=0, case_kind="trace",
                logits=logits, settings=settings,
            ))
        pairs.append({
            "name": f"step0.{call}",
            "role": "main",
            "step": 0,
            "native_case": names["native"],
            "upstream_case": names["upstream"],
            "settings": {"temperature": f32(settings[0]), "top_k": settings[1],
                         "top_p": f32(settings[2])},
        })
    source_comparison = {
        "name": "step0.main",
        "role": "main",
        "step": 0,
        "stage": "generation.step0.main_logits",
        "comparison": common._logit_stats(logits, logits),
    }
    native_sampler_sha = hashlib.sha256(
        (ROOT / "src/ml/audio/drama3/kernels/resident.metal").read_bytes()
    ).hexdigest()
    return {
        "format": common.FIXTURE_FORMAT,
        "provenance": {
            "source_report": source_report_provenance,
            "source_traces": source_refs,
            "model": common.PINNED_MODEL,
            "upstream": common.PINNED_UPSTREAM,
            "model_files": model_files,
            "upstream_source_hashes": {"fish_omni_tower": common.PINNED_TOWER_SHA256},
            "native_sampler_source_sha256": native_sampler_sha,
            "seeded_rng_parity_claimed": False,
            "native_random_tape": None,
        },
        "sampling_contract": common._canonical_sampling_contract(),
        "generation": {
            "prompt_rows": len(prompt_ids),
            "frame_count": len(frames),
            "capture_steps": [0],
            "model_config": model_config,
            "input_report_sha256": report_sha,
            "ras_token_mapping": None,
            "ras_window_size": None,
            "ras_checks": [],
        },
        "input_logit_comparisons": [source_comparison],
        "pairs": pairs,
        "cases": cases,
    }


def probe_for_fixture(fixture: dict) -> tuple[dict, str]:
    cases = fixture["cases"]
    ids = list(range(common.NORMAL[1]))
    probe = make_probe(
        fixture, cases[0], ids, ids,
        additional=[(case, ids, ids) for case in cases[1:]],
    )
    fixture_sha = hashlib.sha256(common._json_bytes(fixture)).hexdigest()
    probe["fixture_sha256"] = fixture_sha
    return probe, fixture_sha


class SamplerCommonSpec(unittest.TestCase):
    def test_source_bound_trace_fixture_passes_and_recomputes_lineage(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture = make_source_bound_fixture(Path(temporary))
            probe, digest = probe_for_fixture(fixture)
            report = common.compare_probe(fixture, probe, fixture_sha256=digest)
            self.assertEqual(report["supported_real_trace"]["status"], "pass")
            self.assertEqual(report["supported_real_trace"]["case_count"], 4)
            self.assertEqual(report["input_logit_comparisons"], fixture["input_logit_comparisons"])

    def test_trace_logits_cannot_be_rebound_to_matching_probe_output(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture = make_source_bound_fixture(Path(temporary))
            fixture["cases"][0]["logits"][0] = f32(fixture["cases"][0]["logits"][0] + 0.125)
            probe, digest = probe_for_fixture(fixture)
            with self.assertRaisesRegex(common.InvalidBundle,
                                        "logits differ from the hash-verified source trace stage"):
                common.compare_probe(fixture, probe, fixture_sha256=digest)

    def test_trace_cases_without_generation_or_source_pairs_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture = make_source_bound_fixture(Path(temporary))
            fixture.pop("generation")
            fixture["pairs"] = []
            probe, digest = probe_for_fixture(fixture)
            with self.assertRaisesRegex(common.InvalidBundle,
                                        "trace cases require generation metadata"):
                common.compare_probe(fixture, probe, fixture_sha256=digest)

    def test_trace_cases_without_complete_pairs_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture = make_source_bound_fixture(Path(temporary))
            fixture["pairs"] = []
            probe, digest = probe_for_fixture(fixture)
            with self.assertRaisesRegex(common.InvalidBundle,
                                        "trace cases require complete native/upstream source pairs"):
                common.compare_probe(fixture, probe, fixture_sha256=digest)

    def test_trace_cases_without_source_trace_references_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture = make_source_bound_fixture(Path(temporary))
            fixture["provenance"].pop("source_traces")
            probe, digest = probe_for_fixture(fixture)
            with self.assertRaisesRegex(common.InvalidBundle,
                                        "trace cases require native and upstream source trace references"):
                common.compare_probe(fixture, probe, fixture_sha256=digest)

    def test_source_trace_bytes_must_match_recorded_sha256(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture = make_source_bound_fixture(Path(temporary))
            source_path = Path(fixture["provenance"]["source_traces"]["native"]["path"])
            source_path.write_bytes(source_path.read_bytes() + b" ")
            probe, digest = probe_for_fixture(fixture)
            with self.assertRaisesRegex(common.InvalidBundle, "source trace SHA-256 mismatch"):
                common.compare_probe(fixture, probe, fixture_sha256=digest)

    def test_retry_settings_are_bound_to_contract_and_pair(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture = make_source_bound_fixture(Path(temporary))
            # Mutating both arms and their pair metadata must still fail before
            # output comparison: NORMAL cannot be relabeled as main_retry.
            for case in fixture["cases"]:
                if case["name"].endswith(".main_retry"):
                    case["temperature"] = f32(common.NORMAL[0])
                    case["top_p"] = f32(common.NORMAL[2])
            retry_pair = next(pair for pair in fixture["pairs"] if pair["name"].endswith("main_retry"))
            retry_pair["settings"] = {"temperature": f32(common.NORMAL[0]),
                                       "top_k": common.NORMAL[1], "top_p": f32(common.NORMAL[2])}
            probe, digest = probe_for_fixture(fixture)
            with self.assertRaisesRegex(common.InvalidBundle,
                                        "sampler settings disagree with canonical main_retry contract"):
                common.compare_probe(fixture, probe, fixture_sha256=digest)

    def test_pair_settings_cannot_disagree_with_canonical_retry_policy(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture = make_source_bound_fixture(Path(temporary))
            retry_pair = next(pair for pair in fixture["pairs"] if pair["name"].endswith("main_retry"))
            retry_pair["settings"]["top_p"] = f32(common.NORMAL[2])
            probe, digest = probe_for_fixture(fixture)
            with self.assertRaisesRegex(common.InvalidBundle, "pair.*disagree"):
                common.compare_probe(fixture, probe, fixture_sha256=digest)

    def test_sampling_contract_mutation_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture = make_source_bound_fixture(Path(temporary))
            fixture["sampling_contract"]["main_retry"]["top_p"] = 0.9
            probe, digest = probe_for_fixture(fixture)
            with self.assertRaisesRegex(common.InvalidBundle, "sampling_contract is malformed"):
                common.compare_probe(fixture, probe, fixture_sha256=digest)

    def test_wrong_retry_settings_can_pass_ras_tokens_but_not_contract_guard(self) -> None:
        tape_sha = "c" * 64
        uniforms = [0.8] * 2
        logits = [-100.0 - index for index in range(32)]
        logits[1] = 100.0
        ordered_ids = sorted(range(len(logits)), key=lambda index: (-logits[index], index))[:30]
        main_draw = common._native_draw_record({"sha256": tape_sha}, uniforms, 0, "main")
        retry_draw = common._native_draw_record({"sha256": tape_sha}, uniforms, 0, "main_retry")
        main = common._validate_case_input(common._case(
            name="native.step0.main", role="main", source_arm="native", step=0,
            case_kind="trace", logits=list(logits), settings=common.NORMAL,
            native_draw=main_draw), "main")
        wrong_retry = common._validate_case_input(common._case(
            name="native.step0.main_retry", role="main", source_arm="native", step=0,
            case_kind="trace", logits=list(logits), settings=common.NORMAL,
            native_draw=retry_draw), "retry")
        cases = {main["name"]: main, wrong_retry["name"]: wrong_retry}
        outputs = {
            main["name"]: {"native": algorithm(main, ordered_ids)},
            wrong_retry["name"]: {"native": algorithm(wrong_retry, ordered_ids)},
        }
        context = {
            "tape": {"sha256": tape_sha},
            "token_mapping": {"eos_token_id": 0, "semantic_start_token_id": 100,
                              "semantic_end_token_id": 200},
            "checks": [{"step": 0, "window_main_tokens": [100], "reported_main_token": 100,
                        "main_case": main["name"], "retry_case": wrong_retry["name"]}],
        }
        self.assertEqual(common._run_native_ras_replay(context, cases, outputs)["status"], "pass")
        contract = common._validate_sampling_contract({"sampling_contract": common._canonical_sampling_contract()})
        with self.assertRaisesRegex(common.InvalidBundle, "canonical main_retry contract"):
            common._validate_case_sampling_settings(wrong_retry, contract, "retry")

    def test_fixed_draw_vector_requires_endpoints_and_order(self) -> None:
        fixture, case, _, _ = make_fixture()
        malformed = dict(fixture["cases"][0])
        malformed["draws"] = list(case["draws"][:-1]) + [0.99]
        with self.assertRaises(common.InvalidBundle):
            common._validate_case_input(malformed, "bad-draws")

    def test_gumbel_noise_must_be_open_interval(self) -> None:
        fixture, _, _, _ = make_fixture()
        malformed = dict(fixture["cases"][0])
        malformed["noise_uniforms"] = [list(row) for row in malformed["noise_uniforms"]]
        malformed["noise_uniforms"][0][0] = 0.0
        with self.assertRaises(common.InvalidBundle):
            common._validate_case_input(malformed, "bad-noise")

    def test_native_preflight_binds_exact_fixture_bytes(self) -> None:
        fixture, _, _, _ = make_fixture()
        with self.assertRaises(common.InvalidBundle):
            mlx_reference._validate_inputs(fixture, {"format": common.PROBE_FORMAT}, "1" * 64)

    def test_stable_native_topk_order_is_enforced(self) -> None:
        _, case, _, _ = make_fixture(tie=True)
        row = algorithm(case, [1, 0, 2, 3])
        with self.assertRaisesRegex(common.InvalidBundle, "stable lower-ID"):
            common._validate_algorithm_record({f"native_{key}": value for key, value in row.items()},
                                              case, "native", "tie")

    def test_tie_mismatch_fails_strict_parity_but_not_real_trace_gate(self) -> None:
        fixture, real_case, real_native_ids, real_mlx_ids = make_fixture()
        _, tie_case, tie_native_ids, tie_mlx_ids = make_fixture(tie=True)
        fixture["cases"].append(tie_case)
        probe = make_probe(
            fixture,
            real_case,
            real_native_ids,
            real_mlx_ids,
            additional=(tie_case, tie_native_ids, tie_mlx_ids),
        )
        raw = common._json_bytes(fixture)
        digest = hashlib.sha256(raw).hexdigest()
        probe["fixture_sha256"] = digest
        report = common.compare_probe(fixture, probe, fixture_sha256=digest)
        item = next(row for row in report["sampler_cases"] if row["case_kind"] == "synthetic_topk_boundary_tie")
        self.assertEqual(report["status"], "not_applicable")
        self.assertEqual(report["supported_real_trace"]["status"], "not_applicable")
        self.assertEqual(report["supported_real_trace"]["case_count"], 0)
        self.assertEqual(report["strict_all_cases_parity"]["status"], "fail")
        self.assertFalse(report["strict_all_cases_parity"]["exact_id_parity"])
        self.assertEqual(report["summary"]["real_trace_case_count"], 0)
        self.assertEqual(report["summary"]["synthetic_case_count"], 2)
        self.assertEqual(report["summary"]["cases_with_exact_id_mismatches"], 1)
        self.assertTrue(item["topk_identity_ambiguity_due_to_exact_boundary_tie"])
        self.assertFalse(item["topk_set_equal"])
        self.assertFalse(item["exact_id_comparison"]["topk_set_equal"])
        self.assertIn("topk_set_equal", item["exact_id_mismatches"])
        self.assertTrue(any("synthetic.topk_boundary_tie" in failure
                            for failure in report["strict_all_cases_parity"]["failures"]))

    def test_terminal_singleton_row_extracts_full_vocab_normalization(self) -> None:
        class FakeArray:
            def tolist(self) -> list[list[float]]:
                return [[58.75]]

        self.assertEqual(mlx_reference._terminal_row_list(FakeArray()), [58.75])

        class WrongShape:
            def tolist(self) -> list[float]:
                return [58.75]

        with self.assertRaisesRegex(common.InvalidBundle, "unexpected MLX output shape"):
            mlx_reference._terminal_row_list(WrongShape())

    def test_finite_cdf_arithmetic_failures_are_reported_without_aborting_comparison(self) -> None:
        fixture, case, native_ids, mlx_ids = make_fixture()
        probe = make_probe(fixture, case, native_ids, mlx_ids)

        # Preserve finite F32 records but model an above-range CDF tail with
        # one monotonic decrease just over the existing 1e-7 guard. The
        # expected-softmax and probability checks should still reject gross
        # errors, but this small arithmetic discrepancy must be reportable.
        probe_case = probe["cases"][0]
        probe_case["mlx_full_cdf"][-2] = f32(1.00015)
        probe_case["mlx_full_cdf"][-1] = f32(f32(1.00015) - 1.1920928955078125e-7)
        fixture_sha = hashlib.sha256(common._json_bytes(fixture)).hexdigest()
        probe["fixture_sha256"] = fixture_sha

        report = common.compare_probe(fixture, probe, fixture_sha256=fixture_sha)
        result = report["sampler_cases"][0]
        failures = result["cdf_arithmetic_invariant_failures"]
        self.assertTrue(any(row["sampler"] == "mlx" and row["field"] == "full_cdf"
                            and row["kind"] == "decrease_exceeds_tolerance"
                            and row["index"] == case["top_k"] - 1
                            for row in failures))
        self.assertTrue(any(row["sampler"] == "mlx" and row["kind"] == "above_maximum"
                            for row in failures))
        self.assertTrue(result["full_probability_comparison"]["within_tolerance"])
        self.assertTrue(result["full_cdf_comparison"]["within_tolerance"])
        self.assertEqual(report["strict_all_cases_parity"]["status"], "fail")
        self.assertEqual(report["supported_real_trace"]["status"], "not_applicable")
        self.assertEqual(report["summary"]["cdf_arithmetic_invariant_failure_count"], len(failures))

        probe_case["mlx_full_cdf"][0] = 0.5
        with self.assertRaisesRegex(common.InvalidBundle, "full_cdf disagrees with full probabilities"):
            common.compare_probe(fixture, probe, fixture_sha256=fixture_sha)

        probe_case["mlx_full_cdf"][0] = 1.0
        probe_case["mlx_full_cdf"][-1] = float("nan")
        with self.assertRaisesRegex(common.InvalidBundle, "expected finite F32"):
            common.compare_probe(fixture, probe, fixture_sha256=fixture_sha)

    def test_native_tape_metadata_preserves_step_call_slot_and_uniform(self) -> None:
        tape = {"sha256": "a" * 64}
        native_draw = common._native_draw_record(tape, [0.25] * 22, 1, "main_retry")
        self.assertEqual(native_draw["flat_index"], 12)
        self.assertEqual(native_draw["slot"], 1)
        self.assertEqual(native_draw["uniform"], 0.25)
        case = common._case(name="native.step1.main_retry", role="main", source_arm="native",
                            step=1, case_kind="trace", logits=[0.0, 1.0],
                            settings=(1.0, 2, 1.0), native_draw=native_draw)
        normalized = common._validate_case_input(case, "tape-case")
        self.assertEqual(normalized["draws"][-1], 0.25)
        self.assertEqual(normalized["native_draw"]["flat_index"], 12)

        malformed = dict(case)
        malformed["native_draw"] = dict(native_draw, flat_index=13)
        with self.assertRaisesRegex(common.InvalidBundle, "flat_index disagrees"):
            common._validate_case_input(malformed, "bad-tape-case")

    def test_native_ras_replay_uses_actual_first_and_retry_draws(self) -> None:
        tape_sha = "b" * 64
        main_draw = {"tape_sha256": tape_sha, "step": 0, "slot": 0, "call": "main",
                     "flat_index": 0, "uniform": 0.8}
        retry_draw = {"tape_sha256": tape_sha, "step": 0, "slot": 1, "call": "main_retry",
                      "flat_index": 1, "uniform": 0.8}
        main_case = common._validate_case_input(common._case(
            name="native.step0.main", role="main", source_arm="native", step=0,
            case_kind="trace", logits=[0.0, 3.0, 2.0, 1.0],
            settings=(0.8, 2, 0.9), native_draw=main_draw), "main-case")
        retry_case = common._validate_case_input(common._case(
            name="native.step0.main_retry", role="main", source_arm="native", step=0,
            case_kind="trace", logits=[0.0, 3.0, 2.0, 1.0],
            settings=(1.0, 2, 0.95), native_draw=retry_draw), "retry-case")
        mapping = {"eos_token_id": 0, "semantic_start_token_id": 100,
                   "semantic_end_token_id": 103}
        context = {
            "tape": {"sha256": tape_sha},
            "token_mapping": mapping,
            "checks": [{"step": 0, "window_main_tokens": [100], "reported_main_token": 101,
                        "main_case": "native.step0.main", "retry_case": "native.step0.main_retry"}],
        }
        outputs = {
            main_case["name"]: {"native": algorithm(main_case, [1, 2])},
            retry_case["name"]: {"native": algorithm(retry_case, [1, 2])},
        }
        result = common._run_native_ras_replay(
            context, {main_case["name"]: main_case, retry_case["name"]: retry_case}, outputs)
        self.assertEqual(result["status"], "pass")
        row = result["checks"][0]
        self.assertTrue(row["retry_triggered"])
        self.assertEqual(row["main_draw"]["token_id"], 100)
        self.assertEqual(row["retry_draw"]["token_id"], 101)
        self.assertEqual(row["selected_token_id"], 101)

        context["checks"][0]["reported_main_token"] = 100
        mismatch = common._run_native_ras_replay(
            context, {main_case["name"]: main_case, retry_case["name"]: retry_case}, outputs)
        self.assertEqual(mismatch["status"], "fail")
        self.assertEqual(len(mismatch["failures"]), 1)

    def test_corrupt_full_normalization_is_rejected(self) -> None:
        fixture, case, native_ids, mlx_ids = make_fixture()
        probe = make_probe(fixture, case, native_ids, mlx_ids)
        probe["cases"][0]["native_full_vocab_normalization"] *= 2.0
        digest = hashlib.sha256(common._json_bytes(fixture)).hexdigest()
        probe["fixture_sha256"] = digest
        with self.assertRaisesRegex(common.InvalidBundle, "full_vocab_normalization"):
            common.compare_probe(fixture, probe, fixture_sha256=digest)


if __name__ == "__main__":
    unittest.main()
