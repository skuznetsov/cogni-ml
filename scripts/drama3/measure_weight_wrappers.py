#!/usr/bin/env python3
"""Measure persistent, view-cleared, and recreated mapped weight wrappers.

The default ``wrappers`` design preserves the established two-worker P/R run.
The ``views`` design uses one worker, one excluded persistent prime, then all
six P/V/R permutations in each seeded cycle. No profiling or synchronization is
added by the driver; it validates the example's passive receipts after each
terminal result.
"""

import argparse
import hashlib
import itertools
import json
import os
from pathlib import Path
import random
import subprocess
import time

from measure_host_pairs import CONTROLS, INPUT, sha, validate


DEFAULT_DESIGN = "wrappers"
WRAPPER_MODES = ("persistent", "fresh")
VIEWS_MODES = ("persistent", "views", "fresh")
_VIEWS_PERMUTATIONS = tuple(itertools.permutations(VIEWS_MODES))
_WRAPPER_ORDERS = (
    ("persistent", "fresh", "fresh", "persistent"),
    ("fresh", "persistent", "persistent", "fresh"),
)
_SOURCE_PATHS = (
    "scripts/drama3/measure_weight_wrappers.py",
    "scripts/drama3/measure_host_pairs.py",
    "examples/drama3_weight_wrappers.cr",
    "src/ml/audio/drama3/resident.cr",
    "src/ml/audio/drama3/resident_host_encoder.cr",
    "src/ml/audio/drama3/metal.cr",
    "src/ml/audio/drama3/studio/engine.cr",
)


def _positive_cycles(cycles):
    if type(cycles) is not int or cycles <= 0:
        raise ValueError("cycles must be a positive integer")


def build_views_schedule(cycles, seed):
    """Return a seeded shuffle of all six treatment permutations per cycle."""
    _positive_cycles(cycles)
    if type(seed) is not int:
        raise ValueError("seed must be an integer")
    rng = random.Random(seed)
    schedule = []
    for cycle in range(cycles):
        block = list(_VIEWS_PERMUTATIONS)
        rng.shuffle(block)
        schedule.extend({
            "cycle": cycle,
            "permutation": "-".join(order),
            "order": list(order),
        } for order in block)
    validate_views_schedule(schedule, cycles)
    return schedule


def validate_views_schedule(schedule, cycles):
    """Reject incomplete, duplicated, or unbalanced P/V/R six-permutation cycles."""
    _positive_cycles(cycles)
    if not isinstance(schedule, list) or len(schedule) != 6 * cycles:
        raise ValueError("views schedule must contain six unique mode permutations per cycle")
    expected_permutations = set(_VIEWS_PERMUTATIONS)
    for cycle in range(cycles):
        rows = [row for row in schedule if isinstance(row, dict) and row.get("cycle") == cycle]
        orders = []
        for row in rows:
            order = row.get("order")
            if (not isinstance(order, list) or len(order) != len(VIEWS_MODES)
                    or row.get("permutation") != "-".join(order)):
                raise ValueError("views schedule must contain six unique mode permutations per cycle")
            orders.append(tuple(order))
        if len(rows) != 6 or set(orders) != expected_permutations or len(set(orders)) != 6:
            raise ValueError("views schedule must contain six unique mode permutations per cycle")

        for position in range(len(VIEWS_MODES)):
            counts = {mode: sum(order[position] == mode for order in orders) for mode in VIEWS_MODES}
            if counts != {mode: 2 for mode in VIEWS_MODES}:
                raise ValueError("views schedule is not balanced by position")
        directions = {
            (left, right): sum(
                left == order[index] and right == order[index + 1]
                for order in orders for index in range(len(order) - 1)
            )
            for left in VIEWS_MODES for right in VIEWS_MODES if left != right
        }
        if directions != {(left, right): 2 for left in VIEWS_MODES
                          for right in VIEWS_MODES if left != right}:
            raise ValueError("views schedule is not balanced by direction")


def views_request_count(cycles):
    """One excluded persistent prime plus 18 scored jobs per complete cycle."""
    _positive_cycles(cycles)
    return 1 + 18 * cycles


def wrapper_orders(cycles):
    """Preserve the established mirrored P/R worker order for wrappers design."""
    _positive_cycles(cycles)
    return tuple(order * cycles for order in _WRAPPER_ORDERS)


def _counter(stats, key, label, minimum=0):
    value = stats[key]
    assert type(value) is int and value >= minimum, f"{label}.{key} is invalid: {value!r}"
    return value


def _validate_converted_cache(wrapper, previous_count=None):
    cache = wrapper["converted_cache"]
    before = _counter(cache, "before_count", "converted_cache")
    after = _counter(cache, "after_count", "converted_cache")
    assert cache["unchanged"] is True, "converted GPU weight cache changed"
    assert before == after, "converted GPU weight cache count changed"
    if previous_count is not None:
        assert before == previous_count, "converted GPU weight cache count differs from previous job"
    return after


def qualify(previous, wrapper, warm, mode, design="wrappers", previous_cache_count=None):
    """Qualify wrapper/view lifetime receipts for a prime or warm job."""
    assert design in ("wrappers", "views"), f"unknown design: {design}"
    assert mode in (WRAPPER_MODES if design == "wrappers" else VIEWS_MODES), mode
    assert wrapper["mode"] == mode, "reported treatment differs from requested mode"
    before, after = wrapper["preparation"], wrapper["after_job"]
    for stats, label in ((before, "preparation"), (after, "after_job")):
        for key in ("mappings", "mapped_bytes", "factory_calls", "generation",
                    "constructed", "retired", "same_backing"):
            _counter(stats, key, label)

    if not warm:
        assert mode == "persistent", "the cold prime must use the persistent treatment"
        assert before["mappings"] == before["factory_calls"] == 0
        assert after["mappings"] > 0
        if design == "views":
            for stats, label in ((before, "preparation"), (after, "after_job")):
                for key in ("view_generation", "tensor_views_created", "tensor_views",
                            "tensor_views_cleared"):
                    _counter(stats, key, label)
            assert before["tensor_views_cleared"] == 0
            assert after["tensor_views_cleared"] == 0
            assert after["view_generation"] == before["view_generation"]
            assert after["tensor_views_created"] >= before["tensor_views_created"]
            assert after["tensor_views"] >= 0
        return

    assert previous is not None
    mappings = _counter(previous, "mappings", "previous")
    mapped_bytes = _counter(previous, "mapped_bytes", "previous")
    previous_factory_calls = _counter(previous, "factory_calls", "previous")
    previous_generation = _counter(previous, "generation", "previous")
    assert before["mappings"] == after["mappings"] == mappings
    assert before["mapped_bytes"] == after["mapped_bytes"] == mapped_bytes
    recreated = mappings if mode == "fresh" else 0
    assert before["constructed"] == recreated, "unexpected wrapper construction count"
    assert before["retired"] == recreated, "unexpected wrapper retirement count"
    assert before["same_backing"] == recreated, "same_backing count does not cover every recreated wrapper"
    assert before["factory_calls"] - previous_factory_calls == recreated, "unexpected no-copy factory count"
    assert after["factory_calls"] == before["factory_calls"], "unexpected lazy wrappers after prime"
    assert before["generation"] - previous_generation == (1 if mode == "fresh" else 0)
    assert after["generation"] == before["generation"]
    assert after["constructed"] == after["retired"] == after["same_backing"] == 0

    if design == "wrappers":
        if mode == "fresh":
            assert before["tensor_views_cleared"] > 0
        return

    for stats, label in ((before, "preparation"), (after, "after_job")):
        for key in ("view_generation", "tensor_views_created", "tensor_views",
                    "tensor_views_cleared"):
            _counter(stats, key, label)
    assert after["tensor_views_cleared"] == 0, "after_job prepare(false) cleared views"
    clears_views = mode in ("views", "fresh")
    expected_view_generation = _counter(previous, "view_generation", "previous") + int(clears_views)
    previous_views_created = _counter(previous, "tensor_views_created", "previous")
    previous_views = _counter(previous, "tensor_views", "previous")
    assert before["view_generation"] == expected_view_generation
    assert after["view_generation"] == before["view_generation"]
    assert before["tensor_views_created"] == previous_views_created, \
        "prepare changed the cumulative TensorBuffer view creation count"
    if clears_views:
        assert before["tensor_views_cleared"] == previous_views, \
            "cleared TensorBuffer view count differs from the previous active cache"
        assert before["tensor_views"] == 0, "raw TensorBuffer cache remained populated after clear"
        assert after["tensor_views"] > 0, "raw TensorBuffer views were not rebuilt by the job"
        assert after["tensor_views_created"] - before["tensor_views_created"] == after["tensor_views"], \
            "new TensorBuffer view count differs from the post-job cache size"
    else:
        assert before["tensor_views_cleared"] == 0, "persistent treatment cleared views"
        assert before["tensor_views"] == previous_views, "persistent preparation changed active views"
        assert after["tensor_views_created"] == before["tensor_views_created"]
        assert before["tensor_views"] == after["tensor_views"] == previous_views

    _validate_converted_cache(wrapper, previous_cache_count)


def qualify_first_command_bindings(wrapper, report, expected_index, mapped_bytes):
    """Check the passive first-command histogram against the full mapped bank."""
    histogram = wrapper["first_command_bindings"]
    assert type(expected_index) is int and expected_index >= 0
    assert type(mapped_bytes) is int and mapped_bytes > 0
    assert type(histogram["command_index"]) is int
    assert histogram["command_index"] == expected_index, "unexpected first-command index"
    assert histogram["observed"] is True, "first-command bindings were not observed"
    assert histogram["submitted"] is True, "first command was not submitted"
    buffer_bindings = _counter(histogram, "buffer_bindings", "first_command_bindings")
    raw_bindings = _counter(histogram, "raw_weight_bindings", "first_command_bindings")
    assert raw_bindings <= buffer_bindings, "raw weight bindings exceed all buffer bindings"
    shards = histogram["shards"]
    assert isinstance(shards, list) and shards, "first-command shard coverage is empty"
    assert len(shards) == _counter(wrapper["after_job"], "mappings", "after_job")
    ordinals = []
    binding_sum = 0
    extent_sum = 0
    for row in shards:
        shard = _counter(row, "shard", "shard", minimum=1)
        extent = _counter(row, "mapped_bytes", f"shard {shard}", minimum=1)
        bindings = _counter(row, "bindings", f"shard {shard}")
        ordinals.append(shard)
        extent_sum += extent
        binding_sum += bindings
    assert ordinals == list(range(1, len(shards) + 1)), "known shard ordinals are not contiguous"
    assert extent_sum == mapped_bytes, "mapped shard extents do not cover the bank"
    assert binding_sum == raw_bindings, "per-shard counts differ from raw weight binding total"
    submitted = report["resident_graph"]["submitted"]
    assert type(submitted) is int and submitted > 0, "report has no submitted graph commands"
    return histogram


def _warm_histogram_signature(wrapper):
    histogram = wrapper["first_command_bindings"]
    return tuple((key, json.dumps(histogram[key], sort_keys=True)) for key in (
        "observed", "submitted", "buffer_bindings", "raw_weight_bindings", "shards",
    ))


def qualify_warm_binding_histogram(previous_wrapper, current_wrapper):
    assert _warm_histogram_signature(previous_wrapper) == _warm_histogram_signature(current_wrapper), \
        "warm first-command binding histograms differ"


def qualify_command_index_after(previous_wrapper, previous_report, current_wrapper):
    previous_index = previous_wrapper["first_command_bindings"]["command_index"]
    submitted = previous_report["resident_graph"]["submitted"]
    assert type(previous_index) is int and previous_index >= 0
    assert type(submitted) is int and submitted > 0
    current_index = current_wrapper["first_command_bindings"]["command_index"]
    assert current_index == previous_index + submitted, "unexpected first-command index"


def _source_hashes(repo_root):
    hashes = {}
    for relative in _SOURCE_PATHS:
        path = repo_root / relative
        if not path.is_file():
            raise FileNotFoundError(f"benchmark source is missing: {relative}")
        hashes[relative] = sha(path)
    return hashes


def _write_manifest(handle, records):
    handle.seek(0)
    handle.truncate()
    json.dump(records, handle, indent=2)
    handle.write("\n")
    handle.flush()


def _event_for_request(process, stderr_path):
    while True:
        line = process.stdout.readline()
        if not line:
            raise RuntimeError("worker exited; inspect " + str(stderr_path))
        event = json.loads(line)
        if event["event"] == "error":
            raise RuntimeError(event)
        if event["event"] == "result":
            return event


def _run_worker(*, design, worker, jobs, args, output_dir, manifest_handle, records,
                baseline, baseline_sha256, binary_sha256, source_hashes, controls):
    stderr_name = f"worker{worker}.stderr" if design == "wrappers" else f"views-worker{worker}.stderr"
    stderr_path = output_dir / stderr_name
    env = {key: value for key, value in os.environ.items() if not key.startswith("DRAMA3_")}
    env.pop("QWEN35_NATIVE_TOKENIZER_OFF", None)
    env.update(controls)
    previous_stats = None
    previous_cache_count = None
    previous_histogram = None
    previous_report = None
    expected_command_index = 0
    failed = False
    with stderr_path.open("x") as errors:
        process = None
        try:
            process = subprocess.Popen(
                [str(args.binary), str(args.model)], stdin=subprocess.PIPE,
                stdout=subprocess.PIPE, stderr=errors, text=True, env=env,
            )
            ready_line = process.stdout.readline()
            if not ready_line:
                raise RuntimeError("worker exited before ready; inspect " + str(stderr_path))
            ready = json.loads(ready_line)
            assert ready["event"] == "ready", ready

            for job in jobs:
                mode = job["mode"]
                output = output_dir / job["output_name"]
                assert not output.exists(), f"refusing existing waveform: {output}"
                load_before = os.getloadavg()
                started = time.monotonic()
                process.stdin.write(json.dumps({
                    "input": INPUT, "output": str(output), "wrappers": mode,
                }) + "\n")
                process.stdin.flush()
                event = _event_for_request(process, stderr_path)
                wall = time.monotonic() - started
                wav_sha256 = sha(output)
                report = event["report"]
                validate(report, wav_sha256, baseline)
                assert report["host_profile"]["enabled"]
                wrapper = event["weight_wrappers"]
                warm = bool(job["warm"])
                qualify(previous_stats, wrapper, warm=warm, mode=mode, design=design,
                        previous_cache_count=previous_cache_count)

                if design == "views":
                    mapped_bytes = wrapper["after_job"]["mapped_bytes"]
                    qualify_first_command_bindings(
                        wrapper, report, expected_command_index, mapped_bytes
                    )
                    if previous_histogram is not None:
                        qualify_command_index_after(previous_histogram, previous_report, wrapper)
                    if warm:
                        if previous_histogram is not None and previous_histogram.get("warm"):
                            # Prime is deliberately excluded from histogram equality: its F32
                            # conversion can take a different first-command path.
                            qualify_warm_binding_histogram(previous_histogram, wrapper)

                cache_count = None
                if "converted_cache" in wrapper:
                    if warm and design == "views":
                        cache_count = _validate_converted_cache(wrapper, previous_cache_count)
                    else:
                        cache_count = _counter(
                            wrapper["converted_cache"], "after_count", "converted_cache"
                        )
                item = {
                    "design": design,
                    "worker": worker,
                    "job": job["job"],
                    "cycle": job.get("cycle"),
                    "permutation": job.get("permutation"),
                    "position": job.get("position"),
                    "predecessor_mode": job.get("predecessor_mode"),
                    "warm": warm,
                    "mode": mode,
                    "output": str(output),
                    "binary": str(args.binary),
                    "binary_sha256": binary_sha256,
                    "baseline_sha256": baseline_sha256,
                    "source_sha256": source_hashes,
                    "effective_controls": controls,
                    "request_wall_seconds": wall,
                    "host_load_before": load_before,
                    "host_load_after": os.getloadavg(),
                    "wav_sha256": wav_sha256,
                    "report": report,
                    "weight_wrappers": wrapper,
                    "worker_gc": event.get("worker_gc"),
                }
                records.append(item)
                _write_manifest(manifest_handle, records)
                print(json.dumps({key: item[key] for key in (
                    "design", "worker", "job", "warm", "mode", "cycle", "position",
                    "request_wall_seconds", "wav_sha256",
                )}), flush=True)

                if design == "views":
                    previous_histogram = dict(wrapper, warm=warm)
                    previous_report = report
                    expected_command_index += report["resident_graph"]["submitted"]
                previous_stats = wrapper["after_job"]
                if cache_count is not None:
                    previous_cache_count = cache_count
        except BaseException:
            failed = True
            raise
        finally:
            if process is not None:
                if process.stdin is not None and not process.stdin.closed:
                    process.stdin.close()
                code = process.wait()
                if process.stdout is not None:
                    process.stdout.close()
                if not failed:
                    assert code == 0, f"worker exited with status {code}; inspect {stderr_path}"


def _wrapper_jobs(worker, cycles):
    sequence = wrapper_orders(cycles)[worker]
    jobs = [{"job": 0, "warm": False, "mode": "persistent", "output_name": f"worker{worker}-job0-persistent.wav"}]
    predecessor = "persistent"
    for index, mode in enumerate(sequence, start=1):
        jobs.append({
            "job": index, "warm": True, "mode": mode,
            "output_name": f"worker{worker}-job{index}-{mode}.wav",
            "cycle": (index - 1) // 4, "position": (index - 1) % 4,
            "predecessor_mode": predecessor,
        })
        predecessor = mode
    return jobs


def _views_jobs(cycles, seed):
    jobs = [{"job": 0, "warm": False, "mode": "persistent", "output_name": "worker0-prime-persistent.wav"}]
    predecessor = "persistent"
    job_index = 1
    for permutation_index, row in enumerate(build_views_schedule(cycles, seed)):
        for position, mode in enumerate(row["order"]):
            jobs.append({
                "job": job_index, "warm": True, "mode": mode,
                "output_name": f"worker0-cycle{row['cycle']}-perm{permutation_index}-job{position}-{mode}.wav",
                "cycle": row["cycle"], "permutation": row["permutation"],
                "position": position, "predecessor_mode": predecessor,
            })
            predecessor = mode
            job_index += 1
    assert len(jobs) == views_request_count(cycles)
    return jobs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", required=True, type=Path)
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--baseline", required=True, type=Path)
    parser.add_argument("--cycles", type=int, default=1)
    parser.add_argument("--design", choices=("wrappers", "views"), default=DEFAULT_DESIGN,
                        help="wrappers preserves the legacy two-worker P/R design; views runs one-worker P/V/R permutations")
    parser.add_argument("--seed", type=int, default=1, help="seed for the views permutation block order")
    args = parser.parse_args()
    _positive_cycles(args.cycles)
    if not args.binary.is_file():
        parser.error(f"binary does not exist: {args.binary}")
    if not args.model.is_dir():
        parser.error(f"model directory does not exist: {args.model}")
    if not args.baseline.is_file():
        parser.error(f"baseline does not exist: {args.baseline}")

    baseline_bytes = args.baseline.read_bytes()
    baseline_items = json.loads(baseline_bytes)
    if not isinstance(baseline_items, list) or not baseline_items:
        parser.error("baseline must be a non-empty JSON array")
    baseline = baseline_items[0]
    baseline_sha256 = hashlib.sha256(baseline_bytes).hexdigest()
    repo_root = Path(__file__).resolve().parents[2]
    source_hashes = _source_hashes(repo_root)
    binary_sha256 = sha(args.binary)

    controls = dict(
        CONTROLS,
        DRAMA3_GENERATION_SCRATCH="0",
        DRAMA3_HOST_PROFILE="1",
        DRAMA3_METAL_TIMELINE="1",
        DRAMA3_METAL_RESOURCES="1",
        DRAMA3_BACKBONE_DENSE_REDUCTION="tree",
        DRAMA3_EXPERT_THREADS="64",
    )
    args.output.mkdir(parents=True, exist_ok=True)
    manifest_name = "wrapper-worker.json" if args.design == "wrappers" else "views-worker.json"
    manifest = args.output / manifest_name
    with manifest.open("x") as handle:
        records = []
        _write_manifest(handle, records)
        if args.design == "wrappers":
            for worker in range(2):
                _run_worker(
                    design="wrappers", worker=worker,
                    jobs=_wrapper_jobs(worker, args.cycles), args=args,
                    output_dir=args.output, manifest_handle=handle, records=records,
                    baseline=baseline, baseline_sha256=baseline_sha256,
                    binary_sha256=binary_sha256, source_hashes=source_hashes,
                    controls=controls,
                )
            warm_jobs = [item for item in records if item["warm"]]
            print(json.dumps({
                "status": "PASS", "design": args.design, "jobs": len(records),
                "warm_jobs": len(warm_jobs), "baseline_sha256": baseline_sha256,
                "same_sampler_inputs_readbacks_waits": True,
                "same_mmap_backing": True,
            }), flush=True)
        else:
            _run_worker(
                design="views", worker=0, jobs=_views_jobs(args.cycles, args.seed), args=args,
                output_dir=args.output, manifest_handle=handle, records=records,
                baseline=baseline, baseline_sha256=baseline_sha256,
                binary_sha256=binary_sha256, source_hashes=source_hashes,
                controls=controls,
            )
            warm_jobs = [item for item in records if item["warm"]]
            assert len(records) == views_request_count(args.cycles)
            assert len(warm_jobs) == 18 * args.cycles
            print(json.dumps({
                "status": "PASS", "design": args.design, "jobs": len(records),
                "warm_jobs": len(warm_jobs), "cycles": args.cycles, "seed": args.seed,
                "baseline_sha256": baseline_sha256,
                "warm_first_command_histograms_equal": True,
                "same_sampler_inputs_readbacks_waits": True,
                "same_mmap_backing": True,
            }), flush=True)


if __name__ == "__main__":
    main()
