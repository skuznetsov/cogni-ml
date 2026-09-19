import copy
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

MODULE = Path(__file__).parents[1] / "scripts/check_metal_residency_footprint.py"
spec = importlib.util.spec_from_file_location("residency_check", MODULE)
check = importlib.util.module_from_spec(spec)
spec.loader.exec_module(check)


def fixture(mode):
    plain_mode = mode.removeprefix("file-")
    phases = ["baseline", "touched", "prepared"]
    phases += (["retained", "after_250ms", "after_1000ms", "after_5000ms", "released"]
               if plain_mode == "hold" else ["released", "after_250ms", "after_1000ms", "after_5000ms"])
    batches = []
    for cycle in range(1 if plain_mode == "hold" else 3):
        batch = {}
        for i, phase in enumerate(phases):
            live = phase in ("touched", "prepared", "retained") or (plain_mode == "hold" and phase.startswith("after_"))
            batch[phase] = dict(event="sample", mode=mode, cycle=cycle, phase=phase, bytes=check.SIZE,
                                elapsed_ms=cycle * 20000 + i * 2000, footprint=(16 << 20) + live * check.SIZE,
                                resident=(16 << 20) + live * check.SIZE, metal_allocated=0)
        batches.append(batch)
    return batches


class ResidencyCheckerTest(unittest.TestCase):
    def test_file_resident_metric_with_insensitive_footprint(self):
        runs = [fixture("file-" + mode) for mode in ("control", "request", "hold")]
        for run in runs:
            for batch in run:
                for row in batch.values():
                    row["footprint"] = 16 << 20
        result = check.evaluate(*runs, file_backed=True)
        self.assertTrue(result["bounded_task_accounting_recovery"])
        self.assertFalse(result["system_reclamation_proven"])
        runs[1][2]["after_5000ms"]["resident"] += check.SIZE
        self.assertFalse(check.evaluate(*runs, file_backed=True)["bounded_task_accounting_recovery"])

    def test_file_requires_held_resident_signal(self):
        runs = [fixture("file-" + mode) for mode in ("control", "request", "hold")]
        runs[2][0]["after_5000ms"]["resident"] -= check.SIZE
        with self.assertRaises(ValueError):
            check.evaluate(*runs, file_backed=True)

    def test_file_live_requested_mapping_can_be_invisible(self):
        runs = [fixture("file-" + mode) for mode in ("control", "request", "hold")]
        for batch in runs[1]:
            batch["prepared"]["resident"] = batch["baseline"]["resident"]
        result = check.evaluate(*runs, file_backed=True)
        self.assertTrue(result["bounded_task_accounting_recovery"])
        self.assertFalse(result["requested_live_resident_signal_detected"])
        self.assertEqual(result["residency_retention_verdict"], "unqualified")

    def test_file_matched_control_bound_is_independent(self):
        runs = [fixture("file-" + mode) for mode in ("control", "request", "hold")]
        runs[0][0]["after_5000ms"]["resident"] -= 7 << 20
        runs[1][0]["after_5000ms"]["resident"] += 7 << 20
        self.assertFalse(check.evaluate(*runs, file_backed=True)["bounded_task_accounting_recovery"])

    def test_file_reader_identity(self):
        for mode in ("file-control", "file-request", "file-hold"):
            rows = [dict(event="config", mode=mode, bytes=check.SIZE,
                         cycles=1 if mode == "file-hold" else 3,
                         device="Apple M2 Max", gpu_commands=0)]
            rows += [x for b in fixture(mode) for x in b.values()]
            rows += [dict(event="complete", mode=mode, gpu_commands=0)]
            with tempfile.TemporaryDirectory() as folder:
                path = Path(folder) / "run.log"
                path.write_text("\n".join(map(json.dumps, rows)) + "\n[EXIT: 0]")
                self.assertEqual(len(check.read_run(path, mode)), rows[0]["cycles"])
                with self.assertRaises(ValueError):
                    check.read_run(path, mode.removeprefix("file-"))

    def evaluate(self, request=None, control=None, hold=None):
        return check.evaluate(control or fixture("control"), request or fixture("request"), hold or fixture("hold"))

    def test_clean(self):
        self.assertTrue(self.evaluate()["bounded_reclamation"])

    def test_retention_and_cumulative_baseline(self):
        request = fixture("request")
        for i, batch in enumerate(request):
            # A rising per-cycle baseline cannot hide cumulative retention.
            for row in batch.values():
                row["footprint"] += (i + 1) * check.SIZE
        self.assertFalse(self.evaluate(request=request)["bounded_reclamation"])

    def test_small_residual_tolerance(self):
        request = fixture("request")
        for batch in request:
            batch["after_5000ms"]["footprint"] += 1024 * 1024
        self.assertTrue(self.evaluate(request=request)["bounded_reclamation"])

    def test_hold_must_be_detected(self):
        hold = fixture("hold")
        hold[0]["after_5000ms"]["footprint"] -= check.SIZE
        with self.assertRaises(ValueError):
            self.evaluate(hold=hold)

    def test_control_drift_is_inconclusive(self):
        control = fixture("control")
        control[0]["after_5000ms"]["footprint"] += check.SIZE
        with self.assertRaises(ValueError):
            self.evaluate(control=control)

    def test_touched_signal_required(self):
        request = fixture("request")
        request[1]["touched"]["footprint"] -= check.SIZE
        with self.assertRaises(ValueError):
            self.evaluate(request=request)

    def test_reader_rejects_invalid_evidence(self):
        rows = [dict(event="config", mode="request", bytes=check.SIZE, cycles=3,
                     device="Apple M2 Max", gpu_commands=0)]
        rows += [x for b in fixture("request") for x in b.values()]
        rows += [dict(event="complete", mode="request", gpu_commands=0)]
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "run.log"
            def read(items, trailer="[EXIT: 0]"):
                path.write_text("\n".join(map(json.dumps, items)) + "\n" + trailer)
                return check.read_run(path, "request")
            self.assertEqual(len(read(rows)), 3)
            mutants = [rows[:-1], rows[:2] + rows[3:]]
            for key, value in (("footprint", float("nan")), ("elapsed_ms", -1),
                               ("cycle", 99), ("phase", "fake"), ("bytes", 1)):
                mutant = copy.deepcopy(rows)
                mutant[2][key] = value
                mutants.append(mutant)
            mutant = copy.deepcopy(rows)
            mutant[7]["elapsed_ms"] = mutant[4]["elapsed_ms"] + 100
            mutants.append(mutant)
            for mutant in mutants:
                with self.assertRaises(ValueError):
                    read(mutant)
            for trailer in ("", "[EXIT: 1]", "[EXIT: 0]\n[KILL]",
                            "[EXIT: 0]\n[EXIT: 1]", "[EXIT: 0]\n[EXIT: 0]"):
                with self.assertRaises(ValueError):
                    read(rows, trailer)


if __name__ == "__main__":
    unittest.main()
