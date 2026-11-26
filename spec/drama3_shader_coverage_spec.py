#!/usr/bin/env python3
"""Focused guards for overlap-safe DRAMA-3 Shader Timeline coverage."""

import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts/drama3"
sys.path.insert(0, str(SCRIPTS))
SPEC = importlib.util.spec_from_file_location(
    "analyze_shader_coverage", SCRIPTS / "analyze_shader_coverage.py"
)
coverage = importlib.util.module_from_spec(SPEC)
assert SPEC is not None and SPEC.loader is not None
sys.modules[SPEC.name] = coverage
SPEC.loader.exec_module(coverage)


def _table(schema, columns, rows):
    pieces = ['<trace-query-result><node xpath="/trace-toc/run/data/table">']
    pieces.append(f'<schema name="{schema}">')
    pieces.extend(f'<col><mnemonic>{column}</mnemonic></col>' for column in columns)
    pieces.append("</schema>")
    for row in rows:
        pieces.append("<row>")
        for tag, value in zip(columns, row):
            if tag == "process" and value is not None:
                pieces.append(f'<process fmt="studio-host-profile ({value})"><pid>{value}</pid></process>')
            elif value is None:
                pieces.append("<sentinel/>")
            elif tag == "start":
                pieces.append(f'<start-time fmt="{_start_format(value)}">{value}</start-time>')
            elif tag == "duration":
                pieces.append(f'<duration fmt="{_duration_format(value)}">{value}</duration>')
            else:
                pieces.append(f"<{_xml_tag(tag)}>{value}</{_xml_tag(tag)}>")
        pieces.append("</row>")
    pieces.append("</node></trace-query-result>")
    return "".join(pieces)


def _xml_tag(column):
    return {
        "start": "start-time", "duration": "duration", "event-type": "metal-event-name",
        "cmdbuffer-id": "metal-command-buffer-id", "channel-name": "gpu-channel-name",
        "state": "gpu-state", "name": "metal-object-label", "label": "metal-object-label",
        "event-name": "gpu-event-name", "shader-type": "metal-object-label",
    }.get(column, column)


def _start_format(value):
    minutes, remainder = divmod(value, 60_000_000_000)
    seconds, remainder = divmod(remainder, 1_000_000_000)
    milliseconds, microseconds = divmod(remainder // 1_000, 1_000)
    return f"{minutes}:{seconds:02d}.{milliseconds:03d}.{microseconds:03d}"


def _duration_format(value):
    if value >= 1_000_000:
        return f"{value / 1_000_000:.2f} ms"
    return f"{value / 1_000:.2f} µs"


def _write_table(path, schema, columns, rows):
    path.write_text(_table(schema, columns, rows))
    return {
        "path": str(path.resolve()),
        "bytes": path.stat().st_size,
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }


def _refresh_export(capture_path, schema, export_path):
    manifest = json.loads(capture_path.read_text())
    entry = manifest["exports"][schema]
    entry["bytes"] = export_path.stat().st_size
    entry["sha256"] = hashlib.sha256(export_path.read_bytes()).hexdigest()
    capture_path.write_text(json.dumps(manifest))


def _report(job, warm):
    return {
        "worker_job": job,
        "warm_worker": warm,
        "text": coverage.EXPECTED_INPUT["text"],
        "seed": 1,
        "max_frames": 64,
        "temperature": 0.8,
        "top_k": 30,
        "top_p": 0.9,
        "reference_sha256": None,
        "prompt_rows": 29,
        "frames": list(range(60)),
        "sampler_inputs": {"sha256": "f" * 64},
        "resident_readbacks": {"control": 1, "terminal": 1},
        "resident_allocation_stats": {"allocations": 3, "requested_bytes": 64,
                                      "allocation_ns": 999},
        "resident_graph": {"submitted": 122, "completed": 122, "completion_waits": 122,
                            "pending": 0, "depth": 2, "max_pending": 2,
                            "retained_bytes": 0},
        "stopped_on_eos": True,
        "truncated": False,
        "samples": 10,
        "sample_rate": 44100,
    }


def _synthetic_capture(root):
    pid = 12345
    request_records = []
    for arm, repeats in (("off-before", 2), ("timeline", 4), ("off-after", 2)):
        for repeat in range(repeats):
            timeline = arm == "timeline"
            request_records.append({
                "arm": arm,
                "repeat": repeat,
                "pid": pid if timeline else pid + 1,
                "warm": repeat > 0,
                "timeline": timeline,
                "instruments": timeline and repeat == 2,
                "wav_sha256": "a" * 64,
                "report": _report(repeat + 1, repeat > 0),
            })

    command_lines = []
    for index in range(4 * 122):
        command_lines.append(json.dumps({
            "event": "drama3_gpu_command", "index": index,
            "completion_status": 0,
        }))
    stderr = root / "timeline.stderr"
    stderr.write_text("\n".join(command_lines) + "\n")

    submission_columns = ["start", "event-type", "process", "cmdbuffer-id"]
    submission_rows = [
        (1_000_000_000 + index * 20_010, "CommandBufferSubmission", pid, 1000 + index)
        for index in range(122)
    ]
    submission = root / "metal-application-command-buffer-submissions.xml"
    sub_record = _write_table(submission, coverage.TABLES["submissions"],
                              submission_columns, submission_rows)

    gpu_columns = ["start", "duration", "state", "channel-name", "process", "cmdbuffer-id"]
    gpu_rows = [
        (1_000_000_000 + index * 20_010, 20_000, "Active", "Compute", pid, 1000 + index)
        for index in range(122)
    ]
    gpu_rows.append((1_000_000_000, 20_000, "Active", "Vertex", None, 0))
    gpu = root / "metal-gpu-intervals.xml"
    gpu_record = _write_table(gpu, coverage.TABLES["gpu"], gpu_columns, gpu_rows)

    shader_columns = ["start", "duration", "name", "label", "event-name", "shader-type", "channel-name", "process"]
    shader_rows = [
        (1_000_000_000 + 30 * 20_010, 20_000, "dr3_dense (0)", "dr3_dense (0)",
         "Shader Timeline", "Compute", "Compute", pid),
        (1_000_000_000 + 30 * 20_010 + 5, 5, "dr3_route (1)", "dr3_route (1)",
         "Shader Timeline", "Compute", "Compute", pid),
        (1_000_000_000 + 30 * 20_010, 20_000, "dr3c_conv1d_coalesced (1)",
         "dr3c_conv1d_coalesced (1)", "Shader Timeline", "Compute", "Compute", pid),
    ]
    shader = root / "metal-shader-profiler-intervals.xml"
    shader_record = _write_table(shader, coverage.TABLES["shader"], shader_columns, shader_rows)

    manifest = {
        "format": "drama3-complete-decode-capture-v1",
        "binary_sha256": "b" * 64,
        "template_sha256": "c" * 64,
        "input": coverage.EXPECTED_INPUT,
        "requests": request_records,
        "exports": {
            coverage.TABLES["shader"]: shader_record,
            coverage.TABLES["gpu"]: gpu_record,
            coverage.TABLES["submissions"]: sub_record,
        },
        "raw_trace_removed": True,
    }
    capture_path = root / "capture.json"
    capture_path.write_text(json.dumps(manifest))
    return capture_path, stderr, shader, gpu, submission


class ShaderCoverageSpec(unittest.TestCase):
    PHASES = "prefill=0:28,selector=29:29,decode=30:89,dac=90:121"

    def test_sweep_separates_inclusive_exclusive_and_overlap_without_summing_rows(self):
        result = coverage._coverage({
            "a": [(0, 60)],
            "b": [(40, 100)],
        })
        self.assertEqual(result["a"], {"inclusive_units": 60, "exclusive_units": 40, "overlap_units": 20})
        self.assertEqual(result["b"], {"inclusive_units": 60, "exclusive_units": 40, "overlap_units": 20})

    def test_sample_intervals_are_unioned_and_clipped_to_decode_windows(self):
        windows = [(10, 30), (50, 80)]
        clipped = coverage._clip([(0, 20), (25, 55), (60, 90)], windows)
        self.assertEqual(clipped, [(10, 20), (25, 30), (50, 55), (60, 80)])
        self.assertEqual(coverage._length(clipped), 40)

    def test_phase_ranges_are_explicit_and_trajectory_specific(self):
        self.assertEqual(coverage._parse_phase_ranges(self.PHASES), coverage.EXPECTED_PHASES)
        for invalid in (
            "prefill=0:28,selector=29:29,decode=30:88,dac=89:121",
            "prefill=0:28,selector=29:29,decode=30:90,dac=90:121",
            "prefill=0:28,selector=29:29,decode=30:89",
        ):
            with self.subTest(invalid=invalid), self.assertRaises(coverage.CoverageError):
                coverage._parse_phase_ranges(invalid)

    def test_codec_label_conflict_rejects_phase_attribution_and_suppresses_families(self):
        with tempfile.TemporaryDirectory() as tmp:
            paths = _synthetic_capture(Path(tmp))
            report = coverage.analyze(*paths, self.PHASES)
        self.assertEqual(report["status"], "phase_attribution_rejected")
        self.assertEqual(report["structural_status"], "structurally_complete_sample_coverage")
        self.assertEqual(report["capture"]["native_command_indices_inclusive"], [244, 365])
        self.assertEqual(report["join"]["decode_command_buffer_count"], 60)
        self.assertEqual(report["coverage"]["decode_gpu_active_compute_union_units"], 1_200_000)
        self.assertEqual(report["coverage"]["shader_sample_interval_union_units"], 20_000)
        self.assertEqual(report["coverage"]["interval_unit_validation"]["integer_unit"], "nanoseconds")
        self.assertEqual(report["join"]["unattributed_gpu_vertex_rows_excluded"], 1)
        self.assertEqual(report["phase_attribution"]["status"], "rejected")
        self.assertEqual(report["phase_attribution"]["codec_namespace_rows_overlapping_decode_windows"], 1)
        self.assertEqual(report["phase_attribution"]["codec_namespace_labels_overlapping_decode_windows"],
                         ["dr3c_conv1d_coalesced (1)"])
        self.assertEqual(len(report["phase_attribution"]["representative_conflicts"]), 1)
        self.assertEqual(report["phase_attribution"]["representative_conflicts"][0]["decode_window_overlap_ns"],
                         20_000)
        self.assertFalse(report["coverage"]["phase_specific_tables_emitted"])
        self.assertIsNone(report["coverage"]["family_coverage"])
        self.assertIsNone(report["coverage"]["individual_shader_coverage"])
        self.assertFalse(report["coverage"]["durations_or_percentages_summed_across_rows"])
        self.assertTrue(any("not lower/upper bounds on actual shader runtime" in line
                            for line in report["limitations"]))

    def test_no_codec_conflict_still_does_not_admit_phase_tables_without_shader_command_key(self):
        with tempfile.TemporaryDirectory() as tmp:
            paths = list(_synthetic_capture(Path(tmp)))
            shader = paths[2]
            text = shader.read_text()
            text = text.replace("<metal-object-label>dr3c_conv1d_coalesced (1)</metal-object-label>",
                                "<metal-object-label>dr3_dense (1)</metal-object-label>")
            shader.write_text(text)
            _refresh_export(paths[0], coverage.TABLES["shader"], shader)
            report = coverage.analyze(*paths, self.PHASES)
        self.assertEqual(report["phase_attribution"]["status"], "rejected")
        self.assertEqual(report["phase_attribution"]["codec_namespace_rows_overlapping_decode_windows"], 0)
        self.assertIsNone(report["coverage"]["family_coverage"])

    def test_dropped_native_event_fails_closed(self):
        with tempfile.TemporaryDirectory() as tmp:
            paths = list(_synthetic_capture(Path(tmp)))
            stderr = paths[1]
            lines = stderr.read_text().splitlines()
            del lines[250]
            stderr.write_text("\n".join(lines) + "\n")
            with self.assertRaisesRegex(coverage.CoverageError, "command events"):
                coverage.analyze(*paths, self.PHASES)

    def test_duplicate_submission_command_buffer_fails_closed(self):
        with tempfile.TemporaryDirectory() as tmp:
            paths = list(_synthetic_capture(Path(tmp)))
            submission = paths[4]
            text = submission.read_text()
            text = text.replace("<metal-command-buffer-id>1001</metal-command-buffer-id>",
                                "<metal-command-buffer-id>1000</metal-command-buffer-id>")
            submission.write_text(text)
            # Update the export binding so the test reaches the duplicate-ID guard.
            manifest = json.loads(paths[0].read_text())
            entry = manifest["exports"][coverage.TABLES["submissions"]]
            entry["bytes"] = submission.stat().st_size
            entry["sha256"] = hashlib.sha256(submission.read_bytes()).hexdigest()
            paths[0].write_text(json.dumps(manifest))
            with self.assertRaisesRegex(coverage.CoverageError, "duplicate target command-buffer IDs"):
                coverage.analyze(*paths, self.PHASES)

    def test_unjoined_active_gpu_interval_fails_closed(self):
        with tempfile.TemporaryDirectory() as tmp:
            paths = list(_synthetic_capture(Path(tmp)))
            gpu = paths[3]
            text = gpu.read_text().replace("<metal-command-buffer-id>1000</metal-command-buffer-id>",
                                           "<metal-command-buffer-id>999999</metal-command-buffer-id>", 1)
            gpu.write_text(text)
            manifest = json.loads(paths[0].read_text())
            entry = manifest["exports"][coverage.TABLES["gpu"]]
            entry["bytes"] = gpu.stat().st_size
            entry["sha256"] = hashlib.sha256(gpu.read_bytes()).hexdigest()
            paths[0].write_text(json.dumps(manifest))
            with self.assertRaisesRegex(coverage.CoverageError, "no matching submission"):
                coverage.analyze(*paths, self.PHASES)

    def test_formatted_time_label_mismatch_fails_closed(self):
        element = coverage.ET.fromstring('<duration fmt="2.14 ms">2139667</duration>')
        evidence = coverage.TimeUnitEvidence()
        evidence.check(element)
        self.assertEqual(evidence.checked["duration"], 1)
        bad = coverage.ET.fromstring('<duration fmt="2.14 µs">2139667</duration>')
        with self.assertRaisesRegex(coverage.CoverageError, "inconsistent"):
            evidence.check(bad)

    def test_unattributed_active_compute_row_fails_closed(self):
        with tempfile.TemporaryDirectory() as tmp:
            paths = list(_synthetic_capture(Path(tmp)))
            gpu = paths[3]
            table = gpu.read_text()
            row = (
                '<row><start-time fmt="00:01.000.000">1000000000</start-time>'
                '<duration fmt="0.01 µs">10</duration><gpu-state>Active</gpu-state>'
                '<gpu-channel-name>Compute</gpu-channel-name><sentinel/>'
                '<metal-command-buffer-id>0</metal-command-buffer-id></row>'
            )
            gpu.write_text(table.replace("</node>", f"{row}</node>"))
            _refresh_export(paths[0], coverage.TABLES["gpu"], gpu)
            with self.assertRaisesRegex(coverage.CoverageError, "no valid process PID"):
                coverage.analyze(*paths, self.PHASES)

    def test_unattributed_vertex_row_with_command_buffer_fails_closed(self):
        with tempfile.TemporaryDirectory() as tmp:
            paths = list(_synthetic_capture(Path(tmp)))
            gpu = paths[3]
            table = gpu.read_text()
            row = (
                '<row><start-time fmt="00:01.000.000">1000000000</start-time>'
                '<duration fmt="0.01 µs">10</duration><gpu-state>Active</gpu-state>'
                '<gpu-channel-name>Vertex</gpu-channel-name><sentinel/>'
                '<metal-command-buffer-id>1000</metal-command-buffer-id></row>'
            )
            gpu.write_text(table.replace("</node>", f"{row}</node>"))
            _refresh_export(paths[0], coverage.TABLES["gpu"], gpu)
            with self.assertRaisesRegex(coverage.CoverageError, "no valid process PID"):
                coverage.analyze(*paths, self.PHASES)


if __name__ == "__main__":
    unittest.main()
