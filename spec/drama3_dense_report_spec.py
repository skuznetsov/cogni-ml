"""Fail-closed controls for DRAMA-3 dense projection benchmark evidence."""
import unittest

from scripts.drama3_dense_report import PROJECTIONS, SHAPES, qualify


def fixture(profiles=PROJECTIONS, pairs=1, large_repeats=8, tiny_repeats=128):
    out, err = [], []
    invocation = "dense-run-a"
    input_id = "fnv64:0123456789abcdef"
    out.append({"event": "dense_benchmark_invocation", "id": invocation})
    err.append({"event": "dense_benchmark_invocation", "id": invocation})
    repeats = {"qkv": large_repeats, "z": large_repeats,
               "b": tiny_repeats, "a": tiny_repeats}
    schedule = []
    for projection in profiles:
        schedule.extend((projection, True, index) for index in range(2))
        schedule.extend((projection, False, index) for index in range(pairs))
    verified = 0
    terminal_bytes = 0
    for position, (projection, warmup, pair) in enumerate(schedule):
        order = ["baseline", "candidate"] if pair % 2 == 0 else ["candidate", "baseline"]
        for slot, arm in enumerate(order):
            index = position * 2 + slot
            out.append({"event": "dense_benchmark_arm", "projection": projection,
                        "pair": pair, "warmup": warmup, "arm": arm,
                        "command_index": index, "repetitions": repeats[projection],
                        "shape": list(SHAPES[projection]), "weight_dtype": "BF16",
                        "input_id": input_id})
            err.append({"event": "drama3_gpu_command", "index": index,
                        "gpu_seconds": 0.001 if arm == "baseline" else 0.0009})
        expected_elements = repeats[projection] * SHAPES[projection][0]
        pair_bytes = 2 * expected_elements * 4
        verified += expected_elements
        terminal_bytes += pair_bytes
        completed = (position + 1) * 2
        out.append({"event": "dense_benchmark_pair", "projection": projection,
                    "pair": pair, "warmup": warmup, "order": order,
                    "waits_before_drain": 0, "readbacks_before_drain": 0,
                    "submissions": 2, "final_drain_waits": 2,
                    "exact_bitwise": True, "verified_elements": expected_elements,
                    "terminal_readback_bytes": pair_bytes,
                    "wall_seconds": 0.003,
                    "graph": {"depth": 2, "submitted": completed,
                              "completed": completed, "completion_waits": completed,
                              "pending": 0, "max_pending": 2, "retained_bytes": 0}})
    total_commands = len(schedule) * 2
    out.append({"event": "dense_benchmark_summary", "invocation_id": invocation,
                "model": "/models/drama3", "input_id": input_id,
                "input_source": "fixture.json#row=0", "input_row": 0,
                "candidate_mode": "simd", "projections": list(profiles),
                "shapes": {name: list(SHAPES[name]) for name in profiles},
                "pairs_per_projection": pairs, "warmups_per_projection": 2,
                "large_repeats": large_repeats, "tiny_repeats": tiny_repeats,
                "repeats": repeats,
                "weight_dtypes": {name: "BF16" for name in profiles},
                "graph": {"depth": 2, "submitted": total_commands,
                          "completed": total_commands, "completion_waits": total_commands,
                          "pending": 0, "max_pending": 2, "retained_bytes": 0},
                "readbacks": {"control": 0, "diagnostic": 0,
                              "terminal": terminal_bytes},
                "verified_elements": verified})
    return out, err


class DenseReportControls(unittest.TestCase):
    def test_accepts_complete_pairs_and_reports_measured_speedup(self):
        out, err = fixture()
        report = qualify(out, err)
        self.assertTrue(report["qualified"])
        self.assertEqual(report["timed_command_buffers"], 24)
        self.assertEqual(report["verified_elements"], (2 + 1) * (8 * 8192 + 8 * 4096 + 128 * 32 * 2))
        self.assertEqual(report["projections"]["qkv"]["candidate_speedup"]["median"], 1 / 0.9)

    def test_rejects_intermediate_waits_or_readbacks(self):
        for field in ("waits_before_drain", "readbacks_before_drain"):
            with self.subTest(field=field):
                out, err = fixture()
                next(row for row in out if row.get("event") == "dense_benchmark_pair")[field] = 1
                with self.assertRaisesRegex(ValueError, "intermediate synchronization or readback"):
                    qualify(out, err)

    def test_rejects_missing_or_reused_gpu_timestamps(self):
        out, err = fixture()
        with self.assertRaisesRegex(ValueError, "missing or reused GPU timing"):
            qualify(out, err[:-2])
        out, err = fixture()
        err[2]["index"] = err[1]["index"]
        with self.assertRaisesRegex(ValueError, "duplicate GPU timeline index"):
            qualify(out, err)

    def test_rejects_nonfinite_timing_and_cross_run_streams(self):
        for value in (0, -1, float("nan"), float("inf")):
            out, err = fixture()
            err[2]["gpu_seconds"] = value
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, "invalid GPU timestamp"):
                qualify(out, err)
        out, err = fixture()
        err[0]["id"] = "dense-run-b"
        with self.assertRaisesRegex(ValueError, "invocation identity"):
            qualify(out, err)

    def test_rejects_geometry_exactness_and_drain_mutations(self):
        mutations = (
            ("dense_benchmark_arm", "incorrect projection shape", lambda row: row.__setitem__("shape", [1, 2048])),
            ("dense_benchmark_pair", "bitwise output comparison omitted", lambda row: row.__setitem__("exact_bitwise", False)),
            ("dense_benchmark_pair", "incorrect paired completion boundary", lambda row: row.__setitem__("final_drain_waits", 1)),
            ("dense_benchmark_pair", "graph did not drain exactly once per command", lambda row: row["graph"].__setitem__("pending", 1)),
        )
        for event, message, mutation in mutations:
            with self.subTest(message=message):
                out, err = fixture()
                row = next(row for row in out if row.get("event") == event)
                mutation(row)
                with self.assertRaisesRegex(ValueError, message):
                    qualify(out, err)

    def test_rejects_candidate_input_mismatch_and_extra_timeline(self):
        out, err = fixture()
        arms = [row for row in out if row.get("event") == "dense_benchmark_arm"]
        arms[1]["input_id"] = "other-input"
        with self.assertRaisesRegex(ValueError, "arm used a different input fixture"):
            qualify(out, err)
        out, err = fixture()
        err.append(dict(next(row for row in err if row.get("event") == "drama3_gpu_command")))
        with self.assertRaisesRegex(ValueError, "duplicate GPU timeline index"):
            qualify(out, err)

    def test_rejects_boolean_indices_and_malformed_summary_geometry(self):
        out, err = fixture()
        out[-1]["pairs_per_projection"] = True
        with self.assertRaisesRegex(ValueError, "invalid repeat count"):
            qualify(out, err)
        out, err = fixture()
        out[-1]["shapes"]["qkv"][0] = True
        with self.assertRaisesRegex(ValueError, "unexpected projection shapes"):
            qualify(out, err)
        out, err = fixture()
        arms = [row for row in out if row.get("event") == "dense_benchmark_arm"]
        arms[0]["command_index"] = True
        with self.assertRaisesRegex(ValueError, "invalid benchmark command index"):
            qualify(out, err)
        out, err = fixture(large_repeats=9)
        with self.assertRaisesRegex(ValueError, "repeat count exceeds benchmark bound"):
            qualify(out, err)


if __name__ == "__main__":
    unittest.main()
