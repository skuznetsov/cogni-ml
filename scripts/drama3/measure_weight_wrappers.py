#!/usr/bin/env python3
"""Counterbalance raw weight wrappers inside each fixed, primed Studio worker.

The fresh arm releases all old shard buffers before recreating them over the
same mmap owners. No cache purge, remap, GPU wait or activation read is added.
The first request in each process is a separate cold prime, not a paired arm.
"""

import argparse
import json
import os
from pathlib import Path
import subprocess
import time

from measure_host_pairs import CONTROLS, INPUT, sha, validate


def qualify(previous, wrapper, warm, mode):
    before, after = wrapper["preparation"], wrapper["after_job"]
    if not warm:
        assert before["mappings"] == before["factory_calls"] == 0
        assert after["mappings"] > 0
        return
    assert before["mappings"] == after["mappings"] == previous["mappings"]
    assert before["mapped_bytes"] == after["mapped_bytes"] == previous["mapped_bytes"]
    count = before["mappings"] if mode == "fresh" else 0
    assert before["constructed"] == before["retired"] == before["same_backing"] == count
    assert before["factory_calls"] - previous["factory_calls"] == count
    assert after["factory_calls"] == before["factory_calls"], "unexpected lazy wrappers after prime"
    assert before["generation"] - previous["generation"] == (mode == "fresh")
    if mode == "fresh":
        assert before["tensor_views_cleared"] > 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", required=True, type=Path)
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--baseline", required=True, type=Path)
    parser.add_argument("--cycles", type=int, default=1)
    args = parser.parse_args()
    assert args.cycles > 0
    args.output.mkdir(parents=True, exist_ok=True)
    manifest = args.output / "wrapper-worker.json"
    assert not manifest.exists(), "refusing existing run"
    baseline = json.loads(args.baseline.read_text())[0]
    records = []
    binary_sha = sha(args.binary)
    controls = dict(CONTROLS, DRAMA3_GENERATION_SCRATCH="0", DRAMA3_HOST_PROFILE="1",
                    DRAMA3_METAL_TIMELINE="1", DRAMA3_METAL_RESOURCES="1",
                    DRAMA3_BACKBONE_DENSE_REDUCTION="tree", DRAMA3_EXPERT_THREADS="64")
    for worker, sequence in enumerate((
        ["persistent", "fresh", "fresh", "persistent"],
        ["fresh", "persistent", "persistent", "fresh"],
    )):
        env = {k: v for k, v in os.environ.items() if not k.startswith("DRAMA3_")}
        env.update(controls)
        stderr = args.output / f"worker{worker}.stderr"
        with stderr.open("x") as errors:
            process = subprocess.Popen([str(args.binary), str(args.model)], stdin=subprocess.PIPE,
                                       stdout=subprocess.PIPE, stderr=errors, text=True, env=env)
            failed = False
            try:
                ready = json.loads(process.stdout.readline())
                assert ready["event"] == "ready", ready
                previous = None
                for index, mode in enumerate(["persistent"] + sequence * args.cycles):
                    warm = index > 0
                    output = args.output / f"worker{worker}-job{index}-{mode}.wav"
                    assert not output.exists(), output
                    load_before = os.getloadavg()
                    started = time.monotonic()
                    process.stdin.write(json.dumps({"input": INPUT, "output": str(output), "wrappers": mode}) + "\n")
                    process.stdin.flush()
                    while True:
                        line = process.stdout.readline()
                        if not line:
                            raise RuntimeError("worker exited; inspect " + str(stderr))
                        event = json.loads(line)
                        if event["event"] == "error":
                            raise RuntimeError(event)
                        if event["event"] == "result":
                            break
                    wall = time.monotonic() - started
                    wav_sha = sha(output)
                    validate(event["report"], wav_sha, baseline)
                    assert event["report"]["host_profile"]["enabled"]
                    wrapper = event["weight_wrappers"]
                    qualify(previous, wrapper, warm, mode)
                    item = dict(worker=worker, job=index, warm=warm, mode=mode, output=str(output),
                                binary=str(args.binary), binary_sha256=binary_sha, effective_controls=controls,
                                request_wall_seconds=wall, host_load_before=load_before, host_load_after=os.getloadavg(),
                                wav_sha256=wav_sha, **{k: event[k] for k in ("report", "weight_wrappers", "worker_gc")})
                    records.append(item)
                    manifest.write_text(json.dumps(records, indent=2) + "\n")
                    previous = wrapper["after_job"]
                    print(json.dumps({k: item[k] for k in ("worker", "job", "warm", "mode", "request_wall_seconds", "wav_sha256")}), flush=True)
            except BaseException:
                failed = True
                raise
            finally:
                process.stdin.close()
                code = process.wait(timeout=30)
                if not failed:
                    assert code == 0
    print(json.dumps(dict(status="PASS", jobs=len(records), exact_frames_and_wav=True,
                          same_sampler_inputs_readbacks_waits=True, same_mmap_backing=True)), flush=True)


if __name__ == "__main__":
    main()
