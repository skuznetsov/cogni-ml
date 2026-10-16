"""Offline contract checks for the fixed-request native prefill driver."""
from __future__ import annotations

import copy
import json
import sys
import tempfile
import unittest
from pathlib import Path
import wave


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts" / "drama3"))
import measure_prefill_requests as driver


def synthetic_record(*, chunk: int = 1, submitted: int = 20,
                     completion_waits: int = 20) -> dict:
    return {
        "wav_sha256": "a" * 64,
        "report": {
            "frames": [[21] + [0] * 10, [22] + [1] * 10],
            "worker_job": 1, "warm_worker": False,
            "samples": 176400,
            "sample_rate": 44100,
            "reference_sha256": None, "prompt_rows": 7,
            "text": "A fixed line", "seed": 1, "max_frames": 60,
            "temperature": 0.8, "top_k": 30, "top_p": 0.9,
            "sampler_inputs": {
                "dtype": "float32-le", "uniform_count": 660,
                "sha256": "b" * 64, "provenance": "host input tape uploaded once",
            },
            "resident_readbacks": {"control": 8, "terminal": 88},
            "resident_graph": {
                "submitted": submitted, "completed": submitted,
                "completion_waits": completion_waits, "pending": 0,
            },
            "chunk": chunk,
        },
    }


def synthetic_request_records() -> list[dict]:
    records = []
    for arm in driver.schedule(4):
        record = synthetic_record(chunk=arm["chunk"])
        record.update({
            "id": f"prefill-arm-{arm['arm']:02d}",
            "arm": arm["arm"],
            "chunk": arm["chunk"],
            "prime": arm["prime"],
            "request_wall_seconds": 2.0 + arm["arm"] / 10,
        })
        record["report"]["worker_job"] = arm["arm"] + 1
        record["report"]["warm_worker"] = arm["arm"] > 0
        records.append(record)
    return records


def synthetic_timeline(records: list[dict], *, rich: bool = True) -> str:
    lines = []
    index = 0
    gpu_clock = 100.0
    for record in records:
        lines.append(json.dumps({"event": "prefill_request_begin", "id": record["id"],
                                 "chunk": record["chunk"]}))
        for _ in range(record["report"]["resident_graph"]["submitted"]):
            event = {"event": "drama3_gpu_command", "index": index, "gpu_seconds": 0.001}
            if rich:
                event.update({"completion_status": 0, "gpu_timing_valid": True,
                              "gpu_start_s": gpu_clock, "gpu_end_s": gpu_clock + 0.001})
                gpu_clock += 0.003
            lines.append(json.dumps(event))
            index += 1
        lines.append(json.dumps({"event": "prefill_request_end", "id": record["id"],
                                 "chunk": record["chunk"]}))
    return "\n".join(lines) + "\n"


class PrefillRequestDriverSpec(unittest.TestCase):
    def test_schedule_primes_first_pair_then_uses_abba(self):
        arms = driver.schedule(16)
        self.assertEqual([arm["chunk"] for arm in arms], [1, 16, 1, 16, 16, 1])
        self.assertEqual([arm["prime"] for arm in arms], [True, True, False, False, False, False])
        self.assertEqual([arm["arm"] for arm in arms if not arm["prime"]], [2, 3, 4, 5])
        self.assertEqual([arm["position"] for arm in arms if not arm["prime"]], [2, 3, 4, 5])

    def test_chunk_bounds_fail_closed(self):
        for chunk in (1, 0, 33, -1):
            with self.subTest(chunk=chunk), self.assertRaises(driver.ProbeError):
                driver.schedule(chunk)

    def test_fixture_builds_one_fixed_teacher_forced_request(self):
        with tempfile.TemporaryDirectory() as directory:
            fixture_path = Path(directory) / "prefill-fixture.json.text"
            request = driver.fixture_request(
                {"input": {"text": "A fixed line", "reference_audio": "voice.wav",
                            "reference_text": "spoken line", "seed": 900,
                            "temperature": 0.01, "top_k": 1, "top_p": 0.2}},
                fixture_path, max_frames=60, seed=1,
            )
            self.assertEqual(request, {
                "text": "A fixed line", "reference_audio": str(Path(directory, "voice.wav").resolve()),
                "reference_text": "spoken line", "seed": 1, "max_frames": 60,
                "temperature": 0.8, "top_k": 30, "top_p": 0.9,
            })

    def test_fixture_requires_nonempty_text_and_valid_bounds(self):
        with tempfile.TemporaryDirectory() as directory:
            fixture_path = Path(directory) / "fixture.json"
            for fixture, max_frames, seed in (
                ({"text": "  "}, 60, 1), ({"text": 4}, 60, 1),
                ({"text": "ok"}, 0, 1), ({"text": "ok"}, 60, -1),
            ):
                with self.subTest(fixture=fixture, max_frames=max_frames, seed=seed):
                    with self.assertRaises(driver.ProbeError):
                        driver.fixture_request(fixture, fixture_path, max_frames, seed)

    def test_validates_codes_wav_samples_tape_readbacks_and_graph_against_scalar_prime(self):
        scalar = synthetic_record()
        driver.validate_record(scalar, "scalar", {
            "text": "A fixed line", "seed": 1, "max_frames": 60,
            "temperature": 0.8, "top_k": 30, "top_p": 0.9,
        })
        driver.validate_against_scalar_prime(copy.deepcopy(scalar), scalar, "candidate")

        mutations = (
            (lambda row: row["report"].update(frames=[[21] + [0] * 10]), "frames"),
            (lambda row: row["report"].update(samples=176399), "samples"),
            (lambda row: row["report"].update(reference_sha256="e" * 64), "reference_sha256"),
            (lambda row: row.update(wav_sha256="c" * 64), "WAV bytes"),
            (lambda row: row["report"]["sampler_inputs"].update(sha256="d" * 64), "sampler_inputs"),
            (lambda row: row["report"]["resident_readbacks"].update(control=12), "resident_readbacks"),
            (lambda row: row["report"]["resident_graph"].update(submitted=21, completed=21), "final-row elision"),
            (lambda row: row["report"]["resident_graph"].update(submitted=19, completed=19), "final-row elision"),
            (lambda row: row["report"]["resident_graph"].update(completion_waits=21), "final-row elision"),
            (lambda row: row["report"]["resident_graph"].update(completion_waits=19), "final-row elision"),
            (lambda row: row["report"]["resident_graph"].update(pending=1), "drained"),
            (lambda row: row["report"]["resident_graph"].update(completed=19), "drained"),
        )
        for mutate, message in mutations:
            candidate = copy.deepcopy(scalar)
            mutate(candidate)
            with self.subTest(message=message), self.assertRaisesRegex(driver.ProbeError, message):
                driver.validate_against_scalar_prime(candidate, scalar, "candidate")

    def test_batch_queue_counts_allow_only_the_derived_final_row_elision(self):
        scalar = synthetic_record()
        scalar["report"]["prompt_rows"] = 158
        scalar["report"]["resident_graph"].update(submitted=20, completed=20,
                                                   completion_waits=20)

        batched = synthetic_record(chunk=16, submitted=19, completion_waits=19)
        batched["report"]["prompt_rows"] = 158
        batched.update(chunk=16)
        driver.validate_against_scalar_prime(batched, scalar, "chunk-16 final multirow chunk")

        one_row_tail = synthetic_record(chunk=16, submitted=20, completion_waits=20)
        one_row_tail["report"]["prompt_rows"] = 17
        one_row_tail.update(chunk=16)
        scalar_one_row_tail = synthetic_record(submitted=20, completion_waits=20)
        scalar_one_row_tail["report"]["prompt_rows"] = 17
        driver.validate_against_scalar_prime(
            one_row_tail, scalar_one_row_tail, "chunk-16 one-row final chunk"
        )

        invalid = copy.deepcopy(batched)
        invalid["report"]["resident_graph"].update(submitted=20, completed=20,
                                                    completion_waits=20)
        with self.assertRaisesRegex(driver.ProbeError, "final-row elision"):
            driver.validate_against_scalar_prime(invalid, scalar, "wrong chunk-16 count")

    def test_requires_one_ordered_warm_worker_for_measured_jobs(self):
        for arm in range(6):
            report = {"worker_job": arm + 1, "warm_worker": arm > 0}
            driver.validate_worker_lifecycle(report, arm, f"arm {arm}")

        invalid = (
            ({"worker_job": 2, "warm_worker": True}, 0, "job order"),
            ({"worker_job": 2, "warm_worker": False}, 1, "persistent worker state"),
            ({"worker_job": True, "warm_worker": True}, 1, "job order"),
            ({"worker_job": 3, "warm_worker": True}, 1, "job order"),
        )
        for report, arm, message in invalid:
            with self.subTest(report=report, arm=arm), self.assertRaisesRegex(
                    driver.ProbeError, message):
                driver.validate_worker_lifecycle(report, arm, f"arm {arm}")

    def test_binds_duration_only_and_absolute_gpu_timings_to_request_scopes(self):
        records = synthetic_request_records()
        summary = driver.qualify_request_timeline(synthetic_timeline(records, rich=False), records, "0")
        self.assertEqual(summary["timing_mode"], "checked_duration_only")
        self.assertEqual(summary["completed_commands"], 120)
        self.assertEqual(records[0]["gpu_command_count"], 20)
        self.assertAlmostEqual(records[0]["completed_gpu_interval_sum_seconds"], 0.02)
        self.assertIsNone(records[0]["gpu_span_seconds"])
        self.assertIn("no absolute span", records[0]["gpu_timing_evidence"])

        records = synthetic_request_records()
        summary = driver.qualify_request_timeline(synthetic_timeline(records), records, "2")
        self.assertEqual(summary["timing_mode"], "absolute_valid_intervals")
        self.assertEqual(summary["requests"], 6)
        self.assertGreater(records[0]["gpu_span_seconds"],
                           records[0]["completed_gpu_interval_sum_seconds"])
        timing = driver.summarize_request_timings(records)
        self.assertEqual(timing["arms"]["scalar"]["samples"], 2)
        self.assertEqual(timing["arms"]["batched"]["samples"], 2)
        self.assertIn("completed_gpu_interval_sum", timing["descriptive_scalar_over_batched"])

    def test_rejects_missing_or_unscoped_gpu_commands_and_orphan_markers(self):
        records = synthetic_request_records()
        lines = synthetic_timeline(records).splitlines()
        command_index = next(i for i, line in enumerate(lines)
                             if '"event": "drama3_gpu_command"' in line)
        del lines[command_index]
        with self.assertRaisesRegex(driver.ProbeError, "command count"):
            driver.qualify_request_timeline("\n".join(lines), records, "2")

        records = synthetic_request_records()
        timeline = json.dumps({"event": "drama3_gpu_command", "index": 999,
                              "gpu_seconds": 1.0}) + "\n" + synthetic_timeline(records, rich=False)
        with self.assertRaisesRegex(driver.ProbeError, "outside worker request scopes"):
            driver.qualify_request_timeline(timeline, records, "0")

        records = synthetic_request_records()
        lines = synthetic_timeline(records).splitlines()
        begin_index = next(i for i, line in enumerate(lines)
                           if '"event": "prefill_request_begin"' in line)
        begin = json.loads(lines[begin_index])
        begin["chunk"] = 16
        lines[begin_index] = json.dumps(begin)
        with self.assertRaisesRegex(driver.ProbeError, "begin chunk mismatch"):
            driver.qualify_request_timeline("\n".join(lines), records, "2")

        records = synthetic_request_records()
        lines = synthetic_timeline(records).splitlines()
        del lines[0]
        with self.assertRaisesRegex(driver.ProbeError, "end without begin"):
            driver.qualify_request_timeline("\n".join(lines), records, "2")

        records = synthetic_request_records()
        lines = synthetic_timeline(records).splitlines()
        first_begin = next(i for i, line in enumerate(lines)
                           if 'prefill_request_begin' in line)
        second_begin = next(i for i, line in enumerate(lines[first_begin + 1:], first_begin + 1)
                            if 'prefill_request_begin' in line)
        first_scope = lines[first_begin:second_begin]
        second_end = next(i for i, line in enumerate(lines[second_begin:], second_begin)
                          if 'prefill_request_end' in line)
        second_scope = lines[second_begin:second_end + 1]
        reordered = lines[:first_begin] + second_scope + first_scope + lines[second_end + 1:]
        with self.assertRaisesRegex(driver.ProbeError, "timeline order"):
            driver.qualify_request_timeline("\n".join(reordered), records, "2")

    def test_rejects_out_of_order_gpu_command_indices(self):
        records = synthetic_request_records()
        lines = synthetic_timeline(records).splitlines()
        command_indices = [i for i, line in enumerate(lines)
                           if '"event": "drama3_gpu_command"' in line]
        first, second = command_indices[:2]
        first_event, second_event = json.loads(lines[first]), json.loads(lines[second])
        first_event["index"], second_event["index"] = second_event["index"], first_event["index"]
        lines[first], lines[second] = json.dumps(first_event), json.dumps(second_event)
        with self.assertRaisesRegex(driver.ProbeError, "indices are out of order"):
            driver.qualify_request_timeline("\n".join(lines), records, "2")

    def test_rejects_invalid_or_wrong_mode_gpu_timing(self):
        for mutation, depth, message in (
            (lambda event: event.update(completion_status=1), "2", "completion status"),
            (lambda event: event.update(gpu_timing_valid=False), "2", "interval is not valid"),
            (lambda event: event.update(gpu_end_s=event["gpu_start_s"]), "2", "positive duration"),
            (lambda event: event.update(gpu_timing_valid=True, gpu_start_s=1.0,
                                        gpu_end_s=1.001), "0", "duration-only completion"),
        ):
            records = synthetic_request_records()
            lines = synthetic_timeline(records, rich=depth == "2").splitlines()
            command_index = next(i for i, line in enumerate(lines)
                                 if '"event": "drama3_gpu_command"' in line)
            event = json.loads(lines[command_index])
            mutation(event)
            lines[command_index] = json.dumps(event)
            with self.subTest(depth=depth, message=message), self.assertRaisesRegex(driver.ProbeError, message):
                driver.qualify_request_timeline("\n".join(lines), records, depth)

    def test_rejects_malformed_report_and_sampler_metadata(self):
        cases = []
        bad_codes = synthetic_record()
        bad_codes["report"]["frames"] = [[21, "bad"]]
        cases.append((bad_codes, "integer code frames"))
        partial_codes = synthetic_record()
        partial_codes["report"]["frames"] = [[21, 0]]
        cases.append((partial_codes, "complete integer code frames"))
        bad_tape = synthetic_record()
        bad_tape["report"]["sampler_inputs"]["uniform_count"] = 0
        cases.append((bad_tape, "sampler input provenance"))
        wrong_provenance = synthetic_record()
        wrong_provenance["report"]["sampler_inputs"]["provenance"] = "invented tape env"
        cases.append((wrong_provenance, "sampler input provenance"))
        bad_readbacks = synthetic_record()
        bad_readbacks["report"]["resident_readbacks"]["terminal"] = -1
        cases.append((bad_readbacks, "resident_readbacks"))
        for record, message in cases:
            with self.subTest(message=message), self.assertRaisesRegex(driver.ProbeError, message):
                driver.validate_record(record, "synthetic")

    def test_wav_header_sample_count_matches_report(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "synthetic.wav"
            with wave.open(str(path), "wb") as audio:
                audio.setnchannels(1)
                audio.setsampwidth(2)
                audio.setframerate(44100)
                audio.writeframes(b"\x00\x00" * 3)
            driver.validate_wav_file(path, 3)
            with self.assertRaisesRegex(driver.ProbeError, "sample count"):
                driver.validate_wav_file(path, 4)

    def test_output_directory_must_be_new_and_existing_content_is_preserved(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "run"
            output.mkdir()
            sentinel = output / "sentinel"
            sentinel.write_text("keep", encoding="utf-8")
            with self.assertRaisesRegex(driver.ProbeError, "already exists"):
                driver.create_output_dir(output)
            self.assertEqual(sentinel.read_text(encoding="utf-8"), "keep")


if __name__ == "__main__":
    unittest.main()
