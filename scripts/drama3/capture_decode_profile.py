#!/usr/bin/env python3
"""Capture a complete fixed DRAMA-3 request, bracketed by untraced controls.

Run under run_safe.sh with an explicit timeout of at least 900 seconds. Supply a
Metal System Trace template with Shader Timeline
enabled. All model calls use the existing worker protocol; attaching Instruments
adds no inference command, wait, readback or sampler draw. A trace is diagnostic,
not a throughput benchmark. Raw captures are removed after selected exports.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import select
import shutil
import signal
import subprocess
import time
from xml.etree import ElementTree as ET

import measure_host_pairs as pairs


ENVIRONMENT_KEYS = (
    "PATH", "HOME", "USER", "LOGNAME", "SHELL", "LANG", "LC_ALL", "LC_CTYPE",
    "TMPDIR", "DEVELOPER_DIR", "SDKROOT", "COGNI_METAL_COMMAND_TIMEOUT_MS",
)
TABLES = (
    "metal-shader-profiler-intervals", "metal-application-intervals",
    "metal-application-command-buffer-submissions", "metal-gpu-intervals",
)
SCRATCH_ROOT = Path("/opt/mldata_ssd")


def environment(source: dict[str, str], timeline: bool) -> dict[str, str]:
    result = {key: source[key] for key in ENVIRONMENT_KEYS if key in source}
    result.update(pairs.CONTROLS)
    result.update(DRAMA3_GENERATION_SCRATCH="0", DRAMA3_HOST_PROFILE="0",
                  DRAMA3_DAC_CONV_LAYOUT="coalesced",
                  DRAMA3_METAL_TIMELINE="1" if timeline else "0")
    return result


def trajectory(report: dict) -> dict:
    # Allocation time is deliberately excluded: only counts/bytes are invariants.
    return {
        "allocations": {key: report["resident_allocation_stats"][key]
                        for key in ("allocations", "requested_bytes")},
        "graph": {key: report["resident_graph"][key] for key in
                  ("submitted", "completed", "completion_waits", "pending",
                   "depth", "max_pending", "retained_bytes")},
        "reads": report["resident_readbacks"],
        "frames": report["frames"], "sampler_inputs": report["sampler_inputs"],
        "termination": {key: report[key] for key in
                        ("stopped_on_eos", "truncated", "samples", "sample_rate", "prompt_rows")},
    }


def receive(process: subprocess.Popen, pending: bytearray, timeout: float = 180) -> dict:
    # Do not use TextIO.readline after select: a partial JSON line can still
    # block past the deadline. Preserve complete/partial lines across calls.
    deadline = time.monotonic() + timeout
    while True:
        if b"\n" in pending:
            line, _, rest = pending.partition(b"\n")
            pending[:] = rest
            event = json.loads(line)
            if event["event"] in ("ready", "result", "error"):
                if event["event"] == "error":
                    raise RuntimeError(event)
                return event
        else:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError(f"worker {process.pid} result exceeded {timeout}s")
            readable, _, _ = select.select([process.stdout], [], [], remaining)
            if readable:
                chunk = os.read(process.stdout.fileno(), 65536)
                if not chunk:
                    raise RuntimeError(f"worker {process.pid} exited before a result")
                pending.extend(chunk)


def stop_process(process: subprocess.Popen, first_signal: int | None,
                 grace: float) -> tuple[str, bool]:
    """Stop and reap before touching files the process may still be writing."""
    if first_signal is not None and process.poll() is None:
        process.send_signal(first_signal)
    forced = False
    try:
        output = process.communicate(timeout=grace)[0]
    except subprocess.TimeoutExpired:
        forced = True
        process.terminate()
        try:
            output = process.communicate(timeout=10)[0]
        except subprocess.TimeoutExpired:
            process.kill()
            output = process.communicate(timeout=10)[0]
    return output or "", forced


def cleanup_trace(trace: Path, recorders: list[subprocess.Popen]) -> bool:
    # Even SIGKILL can fail to finish promptly. Never remove a live writer's
    # files: retain ownership information for manual recovery in that case.
    if any(process.poll() is None for process in recorders):
        return False
    if trace.exists():
        shutil.rmtree(trace)
    return True


def sanitize_toc(path: Path) -> list[str]:
    temporary = path.with_suffix(path.suffix + ".sanitized")
    try:
        tree = ET.parse(path)
        for parent in tree.iter():
            for child in list(parent):
                if child.tag == "environment":
                    parent.remove(child)
        tree.write(temporary, encoding="utf-8", xml_declaration=True)
        temporary.replace(path)
        return sorted({table.get("schema") for table in tree.findall(".//table")
                       if table.get("schema")})
    except BaseException:
        # A partial export must not leave an unsanitized environment behind.
        path.unlink(missing_ok=True)
        temporary.unlink(missing_ok=True)
        raise


def export_trace(trace: Path, output: Path, env: dict[str, str]) -> dict:
    exports = {}
    # Never display the unsanitized TOC or export a process environment table.
    with (output / "export.log").open("x") as log:
        toc = output / "trace-toc.xml"
        subprocess.run(["xcrun", "xctrace", "export", "--input", str(trace),
                        "--toc", "--output", str(toc)], env=env,
                       stdout=log, stderr=subprocess.STDOUT, check=True)
        schemas = sanitize_toc(toc)
        for schema in TABLES:
            if schema not in schemas:
                continue
            path = output / (schema + ".xml")
            subprocess.run([
                "xcrun", "xctrace", "export", "--input", str(trace), "--xpath",
                f'/trace-toc/run[@number="1"]/data/table[@schema="{schema}"]',
                "--output", str(path),
            ], env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
            exports[schema] = {"path": str(path), "sha256": pairs.sha(path),
                               "bytes": path.stat().st_size}
    missing = set(TABLES) - set(exports)
    if missing:
        raise RuntimeError(f"missing trace tables: {sorted(missing)}")
    return exports


def start_trace(pid: int, template: Path, trace: Path, output: Path,
                env: dict[str, str], startup_timeout: float = 45,
                cleanup_grace: float = 120,
                recorders: list[subprocess.Popen] | None = None) -> subprocess.Popen:
    process = subprocess.Popen([
        "xcrun", "xctrace", "record", "--template", str(template), "--attach",
        str(pid), "--output", str(trace), "--time-limit", "90s", "--no-prompt",
    ], env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, bufsize=1)
    if recorders is not None:
        # Track ownership before startup, which can itself fail during reaping.
        recorders.append(process)
    lines = []
    pending = bytearray()
    try:
        deadline = time.monotonic() + startup_timeout
        while time.monotonic() < deadline:
            if b"\n" in pending:
                raw, _, rest = pending.partition(b"\n")
                pending[:] = rest
                line = raw.decode("utf-8", errors="replace") + "\n"
                lines.append(line)
                if "Recording started" in line or "Ctrl-C" in line:
                    return process
            else:
                readable, _, _ = select.select(
                    [process.stdout], [], [], max(0, deadline - time.monotonic()))
                if readable:
                    chunk = os.read(process.stdout.fileno(), 65536)
                    if not chunk:
                        break
                    pending.extend(chunk)
        raise RuntimeError("Instruments did not report recording start")
    except BaseException:
        more, _ = stop_process(process, signal.SIGINT, cleanup_grace)
        lines.append(more)
        raise
    finally:
        lines.append(pending.decode("utf-8", errors="replace"))
        (output / "record-start.log").write_text("".join(lines))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--template", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True,
                        help="New directory on the external scratch volume")
    args = parser.parse_args()
    for path in (args.binary, args.template, args.baseline):
        if not path.is_file():
            parser.error(f"missing input: {path}")
    if not (args.model / "config.json").is_file():
        parser.error("model must contain config.json")
    if not args.output.resolve().is_relative_to(SCRATCH_ROOT.resolve()):
        parser.error(f"output must be under {SCRATCH_ROOT}")
    # Do not erase or append to an existing experiment.
    args.output.mkdir(parents=True, exist_ok=False)
    output = args.output.resolve()
    baseline = json.loads(args.baseline.read_text())[0]
    records = []
    invariant = None
    manifest = {"format": "drama3-complete-decode-capture-v1",
                "binary": str(args.binary), "binary_sha256": pairs.sha(args.binary),
                "template_sha256": pairs.sha(args.template), "input": pairs.INPUT,
                "requests": records, "exports": None, "raw_trace_removed": False}

    def save():
        (output / "capture.json").write_text(json.dumps(manifest, indent=2) + "\n")

    trace = output / "complete-request.trace"
    recorders = []
    try:
        for arm, timeline, repeats in (("off-before", False, 2),
                                       ("timeline", True, 4),
                                       ("off-after", False, 2)):
            env = environment(os.environ, timeline)
            with (output / (arm + ".stderr")).open("x") as errors:
                worker = subprocess.Popen([
                    str(args.binary.resolve()), "--worker", "--model", str(args.model.resolve()),
                ], env=env, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                    stderr=errors, text=True)
                recording = None
                pending = bytearray()
                try:
                    assert receive(worker, pending)["event"] == "ready"
                    for repeat in range(repeats):
                        captured = timeline and repeat == 2
                        if captured:
                            recording = start_trace(worker.pid, args.template.resolve(),
                                                    trace, output, env, recorders=recorders)
                        path = output / f"{arm}-{repeat}.wav"
                        started = time.monotonic()
                        worker.stdin.write(json.dumps({"input": pairs.INPUT, "output": str(path)}) + "\n")
                        worker.stdin.flush()
                        event = receive(worker, pending)
                        wall = time.monotonic() - started
                        assert event["event"] == "result"
                        wav_sha = pairs.sha(path)
                        report = event["report"]
                        pairs.validate(report, wav_sha, baseline)
                        # Cold conversion caches legitimately allocate more than warm jobs.
                        if repeat > 0:
                            current = trajectory(report)
                            if invariant is None:
                                invariant = current
                            assert current == invariant, "warm request trajectory/ledger changed"
                        item = {"arm": arm, "repeat": repeat, "pid": worker.pid,
                                "warm": repeat > 0, "instruments": captured,
                                "timeline": timeline, "request_wall_seconds": wall,
                                "wav_sha256": wav_sha, "report": report,
                                "worker_gc": event.get("worker_gc")}
                        records.append(item)
                        save()
                        print(json.dumps({key: item[key] for key in
                              ("arm", "repeat", "request_wall_seconds", "instruments", "wav_sha256")}), flush=True)
                        if recording is not None:
                            if recording.poll() is not None:
                                raise RuntimeError("trace ended before the complete result")
                            more, forced = stop_process(recording, signal.SIGINT, 300)
                            (output / "record-stop.log").write_text(more)
                            assert not forced, "Instruments required forced termination"
                            assert recording.returncode == 0, "Instruments recording failed"
                            recording = None
                            manifest["exports"] = export_trace(trace, output, env)
                            shutil.rmtree(trace)
                            manifest["raw_trace_removed"] = True
                            save()
                finally:
                    try:
                        if recording is not None:
                            stop_process(recording, signal.SIGINT, 300)
                    finally:
                        worker.stdin.close()
                        worker.stdin = None
                        _, forced = stop_process(worker, None, 30)
                        if forced or worker.returncode != 0:
                            raise RuntimeError(f"worker {worker.pid} exited with {worker.returncode}; forced={forced}")
    finally:
        # These files belong exclusively to this newly-created experiment.
        # Ordinary exceptions/export failures clean up after reaping children.
        # SIGKILL/process-group termination can bypass Python cleanup: after an
        # interrupted run, verify/remove this owned trace outside run_safe.
        cleanup_confirmed = cleanup_trace(trace, recorders)
        if not cleanup_confirmed:
            manifest["recorder_cleanup_unresolved_pids"] = [
                process.pid for process in recorders if process.poll() is None]
        toc = output / "trace-toc.xml"
        if toc.exists():
            try:
                sanitize_toc(toc)
            except (ET.ParseError, OSError) as error:
                manifest["toc_removed_after_sanitize_error"] = type(error).__name__
        manifest["raw_trace_removed"] = cleanup_confirmed and not trace.exists()
        save()


if __name__ == "__main__":
    main()
