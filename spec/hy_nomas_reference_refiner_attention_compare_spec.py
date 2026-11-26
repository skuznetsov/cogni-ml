"""Fail-closed artifact/layout and explanation-qualification checks."""
import copy
import hashlib
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parent / "support"))
import hy_nomas_reference_refiner_attention_compare as compare


def qualification():
    return dict(rows=[dict(kind=name, passed=True, atol=0.0, rtol=0.0) for name in compare.QUALIFICATION] +
                     [dict(kind=name, passed=False, atol=2e-5, rtol=2e-5)
                      for name in compare.NEGATIVE_QUALIFICATION],
                numeric_tolerance=dict(atol=2e-5, rtol=2e-5), terminal_cpu_exports=1,
                source_query_chunks=[256, 7], device="mps", torch_version="2.8.0",
                macos_version="26.6.2", os_build="fixture identity",
                source_mask=compare.replay.SOURCE_MASK,
                source_probabilities_scope=compare.replay.PROBABILITIES_SCOPE,
                explicit_stages_scope=compare.replay.EXPLICIT_SCOPE,
                full_generation_executed=False, attention_production_changed=False,
                upstream_revision=compare.replay.admission.PINNED_REVISION,
                source_upstream_source_sha256={name: "a"*64 for name in
                    compare.replay.admission.REQUIRED_UPSTREAM_FILES})


def packet():
    data = b"\0"*8
    return dict(tensor_order=["q"], tensors={"q": dict(dtype="F32", shape=[1, 2],
        offset=0, nbytes=8, sha256=hashlib.sha256(data).hexdigest())}), data


def native_capture(control=False, vector=False):
    order = list(compare.NATIVE_ORDER)
    tensors = {name: dict(dtype="F32", shape=[1]) for name in order}
    if control:
        order.extend(compare.CONNECTED_NATIVE_ORDER[len(compare.NATIVE_ORDER):])
        tensors.update({
            "bf16_boundary_attention": dict(dtype="F32", shape=list(compare.ATTENTION_SHAPE)),
            "bf16_boundary_attention_with_probabilities": dict(dtype="F32", shape=list(compare.ATTENTION_SHAPE)),
            "bf16_boundary_probabilities": dict(dtype="F32", shape=list(compare.SCORE_SHAPE)),
        })
    if vector:
        order.append("vector_tail_attention")
        tensors["vector_tail_attention"] = dict(
            dtype="F32", shape=list(compare.VECTOR_TAIL_SHAPE))
    extra = ((compare.CONNECTED_NAMED_BYTES if control else 0)
             + (compare.VECTOR_TAIL_BYTES if vector else 0))
    return dict(
        bf16_boundaries_control=control,
        vector_tail_control=vector,
        tensor_order=order,
        tensors=tensors,
        setup_named_bytes=compare.BASE_NAMED_BYTES + extra,
        max_named_bytes=compare.CAP,
        encode_allocations=0,
        encode_readbacks=0,
        preterminal_waits=0,
        terminal_waits=1,
    )


def connected_rows():
    exact = compare.CONNECTED_EXACT_COUNTS
    return ([dict(kind=name, passed=True, atol=0.0, rtol=0.0,
                  count=count, finite_count=count, max_abs=0.0, mae=0.0, rms=0.0,
                  mismatch_count=0)
             for name, count in exact.items()] +
            [dict(kind="source_tail_actual", passed=False, atol=2e-5, rtol=2e-5,
                  count=compare.VECTOR_TAIL_BYTES // 4,
                  finite_count=compare.VECTOR_TAIL_BYTES // 4, max_abs=0.4375,
                  mismatch_count=7499)])


def vector_tail_rows():
    rows = connected_rows()
    for name, count in (("vector_tail_actual_gate", compare.VECTOR_TAIL_BYTES // 4),
                        ("vector_tail_assembled_context_diagnostic",
                         compare.Q_BYTES // 4)):
        rows.append(dict(kind=name, passed=True, atol=0.0, rtol=0.0,
                         count=count, finite_count=count, max_abs=0.0, mae=0.0, rms=0.0,
                         mismatch_count=0))
    return rows


class AttentionCompareTests(unittest.TestCase):
    def test_default_capture_control_is_legacy_false_and_connected_order_is_exact(self):
        legacy = native_capture()
        legacy.pop("bf16_boundaries_control")
        legacy.pop("vector_tail_control")
        self.assertEqual(compare._validate_native_capture_contract(legacy), (False, False))
        self.assertEqual(compare._validate_native_capture_contract(native_capture(False)), (False, False))
        self.assertEqual(compare._validate_native_capture_contract(native_capture(True)), (True, False))
        self.assertEqual(compare._validate_native_capture_contract(native_capture(True, True)), (True, True))

    def test_vector_tail_flag_requires_exact_bool_and_bf16_control(self):
        for value in (0, 1, "true", None):
            bad = native_capture(True)
            bad["vector_tail_control"] = value
            with self.subTest(value=value), self.assertRaisesRegex(compare.Error, "exact bool"):
                compare._validate_native_capture_contract(bad)
        with self.assertRaisesRegex(compare.Error, "requires BF16"):
            compare._validate_native_capture_contract(native_capture(False, True))

    def test_connected_capture_rejects_forged_flag_missing_tensor_wrong_shape_and_nonbool(self):
        missing = native_capture(True)
        missing["tensor_order"] = list(compare.NATIVE_ORDER)
        missing["tensors"].pop("bf16_boundary_attention")
        with self.assertRaises(compare.Error):
            compare._validate_native_capture_contract(missing)

        wrong_shape = native_capture(True)
        wrong_shape["tensors"]["bf16_boundary_probabilities"]["shape"] = [1, 263, 16, 128]
        with self.assertRaises(compare.Error):
            compare._validate_native_capture_contract(wrong_shape)

        for key in ("bf16_boundaries_control", "vector_tail_control"):
            for value in (0, 1, "true", None):
                bad = native_capture(False)
                bad[key] = value
                with self.subTest(key=key, value=value), self.assertRaises(compare.Error):
                    compare._validate_native_capture_contract(bad)

    def test_vector_tail_capture_requires_order_shape_and_exact_extra_bytes(self):
        missing = native_capture(True, True)
        missing["tensors"].pop("vector_tail_attention")
        with self.assertRaises(compare.Error):
            compare._validate_native_capture_contract(missing)

        wrong_order = native_capture(True, True)
        wrong_order["tensor_order"].pop()
        with self.assertRaises(compare.Error):
            compare._validate_native_capture_contract(wrong_order)

        wrong_shape = native_capture(True, True)
        wrong_shape["tensors"]["vector_tail_attention"]["shape"] = [1, 8, 16, 128]
        with self.assertRaisesRegex(compare.Error, "vector tail tensor"):
            compare._validate_native_capture_contract(wrong_shape)

        for bad_bytes in (compare.BASE_NAMED_BYTES + compare.CONNECTED_NAMED_BYTES,
                          compare.BASE_NAMED_BYTES + compare.CONNECTED_NAMED_BYTES + compare.VECTOR_TAIL_BYTES + 4):
            bad = native_capture(True, True)
            bad["setup_named_bytes"] = bad_bytes
            with self.subTest(bytes=bad_bytes), self.assertRaisesRegex(compare.Error, "buffer budget"):
                compare._validate_native_capture_contract(bad)

    def test_connected_capture_rejects_budget_drift_and_candidate_when_disabled(self):
        bad_budget = native_capture(True)
        bad_budget["setup_named_bytes"] -= 4
        with self.assertRaisesRegex(compare.Error, "buffer budget"):
            compare._validate_native_capture_contract(bad_budget)

        disabled_with_candidate = native_capture(False)
        disabled_with_candidate["tensor_order"].append("bf16_boundary_attention")
        with self.assertRaises(compare.Error):
            compare._validate_native_capture_contract(disabled_with_candidate)

        vector_disabled = native_capture(True, False)
        vector_disabled["tensor_order"].append("vector_tail_attention")
        vector_disabled["tensors"]["vector_tail_attention"] = dict(
            dtype="F32", shape=list(compare.VECTOR_TAIL_SHAPE))
        with self.assertRaises(compare.Error):
            compare._validate_native_capture_contract(vector_disabled)

    def test_connected_gate_requires_main_actual_source_exactness_and_keeps_tail_non_gating(self):
        result = compare._qualify_connected_rows(connected_rows(), True)
        self.assertTrue(result["connected_main256_gate_passed"])
        self.assertTrue(result["connected_general_control_passed"])
        self.assertTrue(result["connected_instrumented_context_passed"])
        self.assertTrue(result["connected_probability_main256_gate_passed"])
        self.assertTrue(result["connected_probability_general_control_passed"])
        self.assertFalse(result["source_tail_actual_passed"])
        self.assertTrue(result["bf16_boundaries_control"])
        self.assertTrue(result["production_attention_changed"])

        failed_main = connected_rows()
        failed_main[0]["passed"] = False
        with self.assertRaisesRegex(compare.Error, "main256"):
            compare._qualify_connected_rows(failed_main, True)

    def test_vector_tail_exact_rows_gate_zero_metrics_and_preserve_tail_diagnostic(self):
        result = compare._qualify_connected_rows(vector_tail_rows(), True, True)
        self.assertTrue(result["vector_tail_control"])
        self.assertTrue(result["vector_tail_actual_gate_passed"])
        self.assertTrue(result["vector_tail_assembled_context_diagnostic_passed"])
        self.assertFalse(result["source_tail_actual_passed"])
        self.assertTrue(result["production_attention_changed"])

        for name in compare.VECTOR_TAIL_ROWS:
            for key, value in (("mismatch_count", 1), ("max_abs", 0.25),
                               ("finite_count", 1), ("passed", False),
                               ("mae", 0.25)):
                bad = vector_tail_rows()
                next(row for row in bad if row["kind"] == name)[key] = value
                with self.subTest(name=name, key=key), self.assertRaisesRegex(
                        compare.Error, "zero error and mismatches"):
                    compare._qualify_connected_rows(bad, True, True)

        bad = vector_tail_rows()
        bad = [row for row in bad if row["kind"] != "vector_tail_assembled_context_diagnostic"]
        with self.assertRaisesRegex(compare.Error, "coverage"):
            compare._qualify_connected_rows(bad, True, True)

    def test_connected_exact_rows_reject_forged_pass_and_disabled_vector_rows(self):
        for rows, enabled, vector in ((connected_rows(), True, False),
                                      (vector_tail_rows(), True, True),
                                      ([dict(kind="unrelated")], False, False)):
            with self.subTest(enabled=enabled, vector=vector), self.assertRaisesRegex(
                    compare.Error, "duplicate connected"):
                compare._qualify_connected_rows(rows + [dict(rows[0])], enabled, vector)

        forged = connected_rows()
        next(row for row in forged if row["kind"] == "connected_main256_gate")["mismatch_count"] = 1
        with self.assertRaisesRegex(compare.Error, "zero error and mismatches"):
            compare._qualify_connected_rows(forged, True)

        with self.assertRaisesRegex(compare.Error, "disabled BF16"):
            compare._qualify_connected_rows(
                [dict(kind="vector_tail_actual_gate")], False, False)

        with self.assertRaisesRegex(compare.Error, "requires BF16"):
            compare._qualify_connected_rows([], False, True)

    def test_default_result_has_no_connected_gates_and_reports_unchanged_attention(self):
        result = compare._qualify_connected_rows([], False, False)
        self.assertFalse(result["bf16_boundaries_control"])
        self.assertFalse(result["vector_tail_control"])
        self.assertFalse(result["production_attention_changed"])
        for name in ("connected_main256_gate_passed", "connected_general_control_passed",
                     "connected_instrumented_context_passed", "connected_probability_main256_gate_passed",
                     "connected_probability_general_control_passed", "source_tail_actual_passed",
                     "vector_tail_actual_gate_passed",
                     "vector_tail_assembled_context_diagnostic_passed"):
            self.assertIsNone(result[name])

    def test_qualified_explanation_requires_all_actual_source_and_control_rows(self):
        good = qualification()
        compare._qualify_replay(good)
        for name in compare.QUALIFICATION:
            bad = copy.deepcopy(good)
            next(row for row in bad["rows"] if row["kind"] == name)["passed"] = False
            with self.subTest(name=name), self.assertRaisesRegex(compare.Error, "gate failed"):
                compare._qualify_replay(bad)

    def test_missing_or_duplicate_qualification_cannot_pass_vacuously(self):
        for rows in ([], qualification()["rows"][1:], qualification()["rows"]*2):
            bad = qualification()
            bad["rows"] = rows
            with self.assertRaisesRegex(compare.Error, "coverage"):
                compare._qualify_replay(bad)

    def test_negative_controls_must_be_present_and_fail(self):
        for name in compare.NEGATIVE_QUALIFICATION:
            bad = qualification()
            next(row for row in bad["rows"] if row["kind"] == name)["passed"] = True
            with self.subTest(name=name), self.assertRaisesRegex(compare.Error, "sensitivity"):
                compare._qualify_replay(bad)
            bad["rows"] = [row for row in bad["rows"] if row["kind"] != name]
            with self.subTest(missing=name), self.assertRaisesRegex(compare.Error, "coverage"):
                compare._qualify_replay(bad)

    def test_raw_round_alternative_cannot_use_exact_only_falsifier(self):
        bad = qualification()
        bad["rows"][-1].update(atol=0.0, rtol=0.0)
        with self.assertRaisesRegex(compare.Error, "mixed tolerance"):
            compare._qualify_replay(bad)

    def test_runtime_scope_and_upstream_cannot_be_omitted_or_relabeled(self):
        keys = ("device", "torch_version", "macos_version", "os_build", "source_mask",
                "source_probabilities_scope", "explicit_stages_scope", "full_generation_executed",
                "attention_production_changed", "upstream_revision", "source_upstream_source_sha256")
        for key in keys:
            bad = qualification()
            bad.pop(key)
            with self.subTest(key=key), self.assertRaises(compare.Error):
                compare._qualify_replay(bad)
        for key, value in (("device", "cpu"), ("torch_version", "2.9.0"),
                           ("full_generation_executed", 0), ("attention_production_changed", True),
                           ("source_upstream_source_sha256", {"arbitrary.py": "a"*64})):
            bad = qualification()
            bad[key] = value
            with self.subTest(key=key, value=value), self.assertRaises(compare.Error):
                compare._qualify_replay(bad)

    def test_compiled_pin_inventory_cannot_be_partial_or_extended(self):
        self.assertTrue({"src/ml/metal/device.cr", "src/ml/video/hy_nomas_block.cr",
                         "src/ml/video/hy_nomas_mps_reductions_bridge.mm"}
                        <= set(compare.COMPILED_SOURCE_FILES))
        root = Path(compare.__file__).resolve().parents[2]
        pins = {name: hashlib.sha256((root/name).read_bytes()).hexdigest()
                for name in compare.COMPILED_SOURCE_FILES}
        compare._validate_compiled_pins(pins)
        for bad in (dict(list(pins.items())[1:]), {**pins, "extra.cr": "a"*64},
                    {**pins, compare.COMPILED_SOURCE_FILES[0]: "0"*64}):
            with self.assertRaises(compare.Error):
                compare._validate_compiled_pins(bad)

    def test_source_exact_candidate_cannot_pass_with_widened_row_tolerance(self):
        bad = qualification()
        next(row for row in bad["rows"] if row["kind"] == "scaled_only_p_vs_general_p")["atol"] = 2e-5
        with self.assertRaisesRegex(compare.Error, "exact row tolerance"):
            compare._qualify_replay(bad)

    def test_localization_requires_measured_main256_relations_and_divergences(self):
        rows = [dict(kind=name, passed=True, atol=0.0, rtol=0.0)
                for name in compare.LOCALIZATION_POSITIVE] + [
                dict(kind=name, passed=False, atol=2e-5, rtol=2e-5)
                for name in compare.LOCALIZATION_NEGATIVE]
        compare._qualify_localization(rows)
        for index, row in enumerate(rows):
            bad = copy.deepcopy(rows)
            bad[index]["passed"] = not row["passed"]
            with self.subTest(kind=row["kind"]), self.assertRaises(compare.Error):
                compare._qualify_localization(bad)
            bad = rows[:index] + rows[index+1:]
            with self.subTest(missing=row["kind"]), self.assertRaises(compare.Error):
                compare._qualify_localization(bad)
        bad = copy.deepcopy(rows)
        bad[0]["atol"] = 2e-5
        with self.assertRaises(compare.Error):
            compare._qualify_localization(bad)

    def test_tolerance_chunking_or_counter_cannot_drift(self):
        for key, value in (("numeric_tolerance", dict(atol=0.01, rtol=0.01)),
                           ("source_query_chunks", [263]), ("terminal_cpu_exports", True),
                           ("terminal_cpu_exports", 2)):
            bad = qualification()
            bad[key] = value
            with self.subTest(key=key), self.assertRaisesRegex(compare.Error, "scope differs"):
                compare._qualify_replay(bad)

    def test_layout_covers_exact_payload(self):
        doc, data = packet()
        compare._validate_layout(doc, data)
        with self.assertRaises(compare.Error):
            compare._validate_layout(doc, data+b"\0\0\0\0")
        with self.assertRaises(compare.Error):
            compare._validate_layout(doc, data[:-1])

    def test_record_extent_type_and_hash_poisoning(self):
        for key, value in (("offset", True), ("offset", 4), ("nbytes", 4),
                           ("shape", [True, 2]), ("dtype", "BF16"),
                           ("sha256", "0"*64)):
            doc, data = packet()
            doc["tensors"]["q"][key] = value
            with self.subTest(key=key), self.assertRaises(compare.Error):
                compare._validate_layout(doc, data)


if __name__ == "__main__":
    unittest.main()
