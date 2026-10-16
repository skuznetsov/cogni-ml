"""Counterbalanced native worker runs; no profiling or synchronization in the driver.

Run under scripts/run_safe.sh. Each worker has one unscored cold request followed
by warm repeats. The default order mirrors scratch/profile conditions in time.
Only known benchmark controls are emitted; inherited DRAMA3 controls are removed.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time


CONTROLS = {
    "DRAMA3_BACKEND": "metal",
    "DRAMA3_DAC_SCRATCH": "0",
    "DRAMA3_MEMORY_PROFILE": "0",
    "DRAMA3_ALLOCATION_PROFILE": "0",
    "DRAMA3_METAL_TIMELINE": "0",
    "DRAMA3_METAL_RESOURCES": "0",
    "DRAMA3_MOE_BATCH": "1",
    "DRAMA3_EXPERT_THREADS": "64",
    "DRAMA3_DENSE_REDUCTION": "tree",
    "DRAMA3_BACKBONE_DENSE_REDUCTION": "tree",
    "DRAMA3_DELTA_TRANSPOSE": "1",
    "DRAMA3_EXPERT_DOWN_REDUCTION": "native",
    "DRAMA3_MMAP_SHARED": "0",
    "DRAMA3_COGNIGRAPH_MAX_INFLIGHT": "2",
    "DRAMA3_TOWER_EMBEDDING_FUSION": "0",
}
INPUT = {
    "text": "Hello, this is a voice test.",
    "voice_id": None,
    "seed": 1,
    "max_frames": 64,
    "temperature": 0.8,
    "top_k": 30,
    "top_p": 0.9,
}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def validate(report, wav_sha, baseline):
    base = baseline["report"]
    for key in ("frames", "sampler_inputs", "resident_readbacks"):
        assert report[key] == base[key], key + " changed"
    for key in ("submitted", "completed", "completion_waits", "pending"):
        assert report["resident_graph"][key] == base["resident_graph"][key], key + " changed"
    assert wav_sha == baseline["wav_sha256"], "waveform changed"
    if report["resident_generation_scratch"]["enabled"]:
        for stats in report["resident_generation_scratch"]["workspaces"].values():
            assert stats["released"] == 1 and stats["bytes"] == 0 and stats["quarantined"] == 0, stats


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", required=True, type=Path)
    schedules = parser.add_mutually_exclusive_group()
    schedules.add_argument("--legacy-binary", type=Path,
                           help="Compare legacy/current/current/legacy with scratch and profiling off")
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--baseline", required=True, type=Path)
    parser.add_argument("--prefix", default="paired")
    schedules.add_argument("--schedule", help="Comma-separated scratch:profile conditions")
    schedules.add_argument("--dense-schedule", help="Comma-separated tree/simd/simd32 modes; scratch and host profiling stay off")
    schedules.add_argument("--resource-schedule", help="Comma-separated 0/1 resource observers; timeline and host profiling stay on")
    schedules.add_argument("--expert-threads-schedule", help="Comma-separated 32/64 thread requests; dense stays tree, scratch and profiling off")
    schedules.add_argument("--dac-layout-schedule", help="Comma-separated native/coalesced DAC layouts; scratch and profiling stay off")
    parser.add_argument("--warm-repeats", type=int, default=2)
    parser.add_argument("--timeline", action="store_true",
                        help="Read existing completion timestamps; does not introduce waits")
    args = parser.parse_args()
    resource_modes = None
    dac_layouts = None
    if args.dac_layout_schedule is not None:
        dac_layouts = [mode.strip() for mode in args.dac_layout_schedule.split(",")]
        assert dac_layouts and all(mode in ("native", "coalesced") for mode in dac_layouts), "DAC layouts must be native or coalesced"
        schedule = [("0", "0", args.binary, "tree", "64") for mode in dac_layouts]
    elif args.resource_schedule is not None:
        resource_modes = [mode.strip() for mode in args.resource_schedule.split(",")]
        assert resource_modes and all(mode in ("0", "1") for mode in resource_modes), "resource schedule must contain 0 or 1"
        schedule = [("0", "1", args.binary, "tree", "64") for mode in resource_modes]
    elif args.dense_schedule is not None:
        modes = [mode.strip() for mode in args.dense_schedule.split(",")]
        assert modes and all(mode in ("tree", "simd", "simd32") for mode in modes), "dense schedule modes must be tree, simd, or simd32"
        schedule = [("0", "0", args.binary, mode, "64") for mode in modes]
    elif args.expert_threads_schedule is not None:
        widths = [width.strip() for width in args.expert_threads_schedule.split(",")]
        assert widths and all(width in ("32", "64") for width in widths), "expert thread schedule must contain 32 or 64"
        schedule = [("0", "0", args.binary, "tree", width) for width in widths]
    elif args.legacy_binary:
        schedule = [("0", "0", binary, "tree", "64") for binary in
                    (args.legacy_binary, args.binary, args.binary, args.legacy_binary)]
    else:
        conditions = args.schedule or "0:0,1:0,1:1,0:1,0:1,1:1,1:0,0:0"
        conditions = [tuple(item.split(":")) for item in conditions.split(",")]
        assert all(len(item) == 2 and all(x in ("0", "1") for x in item) for item in conditions)
        schedule = [(scratch, profile, args.binary, "tree", "64") for scratch, profile in conditions]
    assert args.warm_repeats > 0
    args.output.mkdir(parents=True, exist_ok=True)
    manifest = args.output / (args.prefix + "-worker.json")
    assert not manifest.exists(), "refusing existing run"
    baseline = json.loads(args.baseline.read_text())[0]
    records = []
    for arm, (scratch, profile, binary, dense_reduction, expert_threads) in enumerate(schedule):
        binary_sha = sha(binary)
        env = {k: v for k, v in os.environ.items() if not k.startswith("DRAMA3_")}
        env.pop("QWEN35_NATIVE_TOKENIZER_OFF", None)
        if args.dense_schedule is not None:
            scratch, profile = "0", "0"
        controls = dict(
            CONTROLS,
            DRAMA3_GENERATION_SCRATCH=scratch,
            DRAMA3_HOST_PROFILE=profile,
            DRAMA3_DENSE_REDUCTION=dense_reduction,
            DRAMA3_EXPERT_THREADS=expert_threads,
            DRAMA3_BACKBONE_DENSE_REDUCTION="tree",
        )
        controls["DRAMA3_METAL_RESOURCES"] = resource_modes[arm] if resource_modes is not None else "0"
        if dac_layouts is not None:
            controls["DRAMA3_DAC_CONV_LAYOUT"] = dac_layouts[arm]
        if args.timeline or resource_modes is not None:
            controls["DRAMA3_METAL_TIMELINE"] = "1"
        env.update(controls)
        stderr_path = args.output / f"{args.prefix}-arm{arm}.stderr"
        started = time.monotonic()
        with stderr_path.open("x") as errors:
            process = subprocess.Popen(
                [str(binary), "--worker", "--model", str(args.model)],
                stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=errors, text=True, env=env,
            )
            try:
                line = process.stdout.readline()
                if not line:
                    raise RuntimeError("no ready event: " + stderr_path.read_text()[-1000:])
                ready = json.loads(line)
                assert ready["event"] == "ready", ready
                load_wall = time.monotonic() - started
                for repeat in range(args.warm_repeats + 1):
                    output = args.output / f"{args.prefix}-arm{arm}-repeat{repeat}.wav"
                    assert not output.exists(), output
                    load_before = os.getloadavg()
                    started = time.monotonic()
                    process.stdin.write(json.dumps({"input": INPUT, "output": str(output)}) + "\n")
                    process.stdin.flush()
                    while True:
                        line = process.stdout.readline()
                        if not line:
                            raise RuntimeError("worker exited: " + stderr_path.read_text()[-1000:])
                        event = json.loads(line)
                        if event["event"] == "error":
                            raise RuntimeError(event)
                        if event["event"] == "result":
                            break
                    wall = time.monotonic() - started
                    report = event["report"]
                    item = {
                        "binary": str(binary), "binary_sha256": binary_sha,
                        "arm": arm, "scratch": scratch, "profile": profile, "repeat": repeat,
                        "dense_reduction": dense_reduction,
                        "expert_threads": expert_threads,
                        "warm": repeat > 0, "effective_controls": controls,
                        "load_wall_seconds": load_wall, "request_wall_seconds": wall,
                        "worker_gc": event.get("worker_gc"),
                        "host_load_before": load_before, "host_load_after": os.getloadavg(),
                        "wav_sha256": sha(output), "report": report,
                    }
                    validate(report, item["wav_sha256"], baseline)
                    if binary == args.binary:
                        assert report["host_profile"]["enabled"] == (profile == "1")
                    records.append(item)
                    manifest.write_text(json.dumps(records, indent=2) + "\n")
                    print(json.dumps({k: item[k] for k in (
                        "arm", "scratch", "profile", "dense_reduction", "expert_threads", "repeat",
                        "effective_controls", "request_wall_seconds", "wav_sha256",
                    )}), flush=True)
            finally:
                process.stdin.close()
                assert process.wait(timeout=30) == 0
    print(json.dumps({"status": "PASS", "jobs": len(records), "exact_frames_and_wav": True,
                      "same_sampler_inputs_readbacks_waits": True}), flush=True)


if __name__ == "__main__":
    main()
