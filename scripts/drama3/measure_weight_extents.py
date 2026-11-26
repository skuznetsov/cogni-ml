#!/usr/bin/env python3
"""Measure whole-shard versus page-aligned tensor-resource layouts.

One Engine runs a cold prime followed by four S/N/N/S (or mirrored)
layout blocks. Each block has an excluded persistent prime and the balanced
P/V/V/P view-cache sham. Layout construction and retirement occur only during
excluded primes. This measures the net layout effect; resource count is part of
the layout and is not isolated as an extent-only cause.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

from measure_host_pairs import CONTROLS, INPUT, sha, validate
from measure_weight_wrappers import (
    _counter,
    _validate_converted_cache,
    qualify_command_index_after,
)


LAYOUTS = ("shard", "tensor")
WRAPPER_MODES = ("persistent", "views")
_WITHIN_LAYOUT_ORDER = ("persistent", "views", "views", "persistent")
_LAYOUT_ORDERS = {
    "shard-tensor-tensor-shard": ("shard", "tensor", "tensor", "shard"),
    "tensor-shard-shard-tensor": ("tensor", "shard", "shard", "tensor"),
}
DEFAULT_LAYOUT_ORDER = "shard-tensor-tensor-shard"
_SOURCE_PATHS = (
    "scripts/drama3/measure_weight_extents.py",
    "spec/drama3_weight_extents_driver_spec.py",
    "spec/drama3_weight_extents_spec.cr",
    "scripts/drama3/measure_weight_wrappers.py",
    "scripts/drama3/measure_host_pairs.py",
    "examples/drama3_weight_wrappers.cr",
    "src/ml/audio/drama3/resident.cr",
    "src/ml/audio/drama3/resident_host_encoder.cr",
    "src/ml/audio/drama3/metal.cr",
    "src/ml/audio/drama3/studio/engine.cr",
)


def extent_jobs(layout_order=DEFAULT_LAYOUT_ORDER):
    """Build the exact 20-request, one-Engine schedule with real predecessors."""
    if layout_order not in _LAYOUT_ORDERS:
        raise ValueError(f"unknown layout order: {layout_order}")
    jobs = []
    predecessor = None

    def append_job(*, block, layout, mode, prime, position=None):
        nonlocal predecessor
        index = len(jobs)
        job = {
            "job": index,
            "block": block,
            "expected_layout": layout,
            "mode": mode,
            "prime": prime,
            "cold_prime": index == 0,
            "scored": not prime,
            "warm": not prime,
            "position": position,
            "output_name": (
                f"job{index}-block{block}-{layout}-"
                f"{'prime' if prime else f'position{position}-{mode}'}.wav"
            ),
            "predecessor": predecessor,
        }
        jobs.append(job)
        predecessor = {
            "job": index,
            "block": block,
            "expected_layout": layout,
            "mode": mode,
            "prime": prime,
        }

    for block, layout in enumerate(_LAYOUT_ORDERS[layout_order]):
        append_job(block=block, layout=layout, mode="persistent", prime=True)
        for position, mode in enumerate(_WITHIN_LAYOUT_ORDER):
            append_job(block=block, layout=layout, mode=mode, prime=False,
                       position=position)

    validate_extent_jobs(jobs, layout_order)
    return jobs


def validate_extent_jobs(jobs, layout_order=DEFAULT_LAYOUT_ORDER):
    """Reject schedule edits that break the balanced blocks or provenance."""
    if layout_order not in _LAYOUT_ORDERS:
        raise ValueError(f"unknown layout order: {layout_order}")
    if not isinstance(jobs, list) or len(jobs) != 20:
        raise ValueError("extent schedule must contain four primes and sixteen scored jobs")
    predecessor = None
    for index, job in enumerate(jobs):
        if not isinstance(job, dict) or job.get("job") != index:
            raise ValueError("extent schedule job indices are not contiguous")
        block = job.get("block")
        if type(block) is not int or not 0 <= block < 4:
            raise ValueError("extent schedule block boundaries are invalid")
        if job.get("expected_layout") != _LAYOUT_ORDERS[layout_order][block]:
            raise ValueError("extent schedule layout does not match its declared block order")
        if job.get("predecessor") != predecessor:
            raise ValueError("extent schedule predecessor does not match the preceding request")
        block = index // 5
        position_in_block = index % 5
        if job.get("block") != block:
            raise ValueError("extent schedule block boundaries are invalid")
        if position_in_block == 0:
            if job.get("prime") is not True or job.get("mode") != "persistent":
                raise ValueError("each layout block must start with an excluded persistent prime")
            if job.get("scored") is not False or job.get("warm") is not False:
                raise ValueError("layout primes must be excluded from scored jobs")
            if job.get("position") is not None:
                raise ValueError("layout prime cannot have a scored position")
        else:
            expected_mode = _WITHIN_LAYOUT_ORDER[position_in_block - 1]
            if job.get("prime") is not False or job.get("mode") != expected_mode:
                raise ValueError("scored block must use the persistent/views/views/persistent order")
            if job.get("scored") is not True or job.get("warm") is not True:
                raise ValueError("all non-prime jobs must be scored warm requests")
            if job.get("position") != position_in_block - 1:
                raise ValueError("scored position is invalid")
        if job.get("cold_prime") is not (index == 0):
            raise ValueError("only the first prime may be the cold prime")
        expected_name = (
            f"job{index}-block{block}-{job['expected_layout']}-"
            f"{'prime' if position_in_block == 0 else f'position{position_in_block - 1}-{job['mode']}'}.wav"
        )
        if job.get("output_name") != expected_name:
            raise ValueError("extent schedule output names must be unique and deterministic")
        predecessor = {
            "job": index,
            "block": block,
            "expected_layout": job["expected_layout"],
            "mode": job["mode"],
            "prime": job["prime"],
        }
    if len({job["output_name"] for job in jobs}) != len(jobs):
        raise ValueError("extent schedule output names are not unique")


def _layout_narrow(layout):
    if layout not in LAYOUTS:
        raise ValueError(f"unknown resource layout: {layout}")
    return int(layout == "tensor")


def _validate_cache_shape(stats, label, expected_layout, expected_page_size=None,
                          expected_known_tensors=None, allow_empty_known=False):
    counters = (
        "narrow", "layout_changed", "page_size", "known_tensors", "shards",
        "mappings", "mapped_bytes", "unique_mapped_bytes", "overlapping_bytes",
        "factory_calls", "generation", "constructed", "retired", "same_backing",
        "view_generation", "tensor_views_created", "tensor_views",
        "tensor_views_cleared",
    )
    for key in counters:
        minimum = 1 if key == "page_size" or (key == "known_tensors" and not allow_empty_known) else 0
        _counter(stats, key, label, minimum=minimum)
    assert stats["narrow"] == _layout_narrow(expected_layout), \
        f"{label}.narrow differs from expected layout {expected_layout}"
    if expected_page_size is not None:
        assert stats["page_size"] == expected_page_size, f"{label}.page_size changed"
    if expected_known_tensors is not None:
        assert stats["known_tensors"] == expected_known_tensors, \
            f"{label}.known_tensors changed"
    assert stats["mapped_bytes"] == stats["unique_mapped_bytes"] + stats["overlapping_bytes"], \
        f"{label} byte totals do not partition into unique and overlapping extents"
    assert stats["mappings"] == 0 or stats["shards"] > 0, \
        f"{label} has resources without a mapped owner"
    assert stats["mappings"] > 0 or stats["shards"] == 0, \
        f"{label} has mapped owners without resources"
    assert stats["mapped_bytes"] > 0 or stats["mappings"] == 0, \
        f"{label} has zero-byte mapped resources"
    return stats


def _extent_union(resources):
    """Return total resource bytes and the union length of per-shard intervals."""
    total = 0
    intervals = {}
    for index, resource in enumerate(resources):
        shard = _counter(resource, "shard", f"resource {index}", minimum=1)
        offset = _counter(resource, "file_offset", f"resource {index}")
        length = _counter(resource, "mapped_bytes", f"resource {index}", minimum=1)
        _counter(resource, "bindings", f"resource {index}")
        intervals.setdefault(shard, []).append((offset, offset + length))
        total += length
    unique = 0
    for shard_intervals in intervals.values():
        shard_intervals.sort()
        start = end = None
        for left, right in shard_intervals:
            if start is None:
                start, end = left, right
            elif left <= end:
                end = max(end, right)
            else:
                unique += end - start
                start, end = left, right
        if start is not None:
            unique += end - start
    return total, unique, intervals


def qualify_first_command_bindings(wrapper, report, expected_index):
    """Validate per-resource receipts and recompute the byte-union counters."""
    histogram = wrapper["first_command_bindings"]
    after = wrapper["after_job"]
    assert type(expected_index) is int and expected_index >= 0
    assert type(histogram["command_index"]) is int
    assert histogram["command_index"] == expected_index, "unexpected first-command index"
    assert histogram["observed"] is True, "first-command bindings were not observed"
    assert histogram["submitted"] is True, "first command was not submitted"
    buffer_bindings = _counter(histogram, "buffer_bindings", "first_command_bindings", minimum=1)
    raw_bindings = _counter(histogram, "raw_weight_bindings", "first_command_bindings")
    assert raw_bindings <= buffer_bindings, "raw weight bindings exceed all buffer bindings"

    resources = histogram["resources"]
    assert isinstance(resources, list) and resources, "first-command resource list is empty"
    assert len(resources) == after["mappings"], "resource list does not cover every mapping"
    page_size = after["page_size"]
    assert all(resource["file_offset"] % page_size == 0 for resource in resources), \
        "resource file offset is not page aligned"
    assert all(resource["mapped_bytes"] % page_size == 0 for resource in resources), \
        "resource extent is not page aligned"
    total, unique, intervals = _extent_union(resources)
    assert total == after["mapped_bytes"], "resource extents do not sum to mapped_bytes"
    assert unique == after["unique_mapped_bytes"], "resource interval union differs from unique_mapped_bytes"
    assert total - unique == after["overlapping_bytes"], \
        "resource overlap differs from overlapping_bytes"
    assert len(intervals) == after["shards"], "resource list does not cover every mapped owner"
    assert sum(_counter(row, "bindings", "resource") for row in resources) == raw_bindings, \
        "per-resource counts differ from raw weight binding total"

    shards = histogram["shards"]
    assert isinstance(shards, list) and shards, "first-command shard coverage is empty"
    assert len(shards) == after["shards"], "shard histogram does not cover every mapped owner"
    assert [row["shard"] for row in shards] == list(range(1, len(shards) + 1)), \
        "known shard ordinals are not contiguous"
    grouped = {}
    for resource in resources:
        row = grouped.setdefault(resource["shard"], {"mapped_bytes": 0, "bindings": 0})
        row["mapped_bytes"] += resource["mapped_bytes"]
        row["bindings"] += resource["bindings"]
    for row in shards:
        shard = _counter(row, "shard", "shard", minimum=1)
        assert row["mapped_bytes"] == grouped[shard]["mapped_bytes"], \
            f"shard {shard} mapped_bytes is not the sum of resource extents"
        assert row["bindings"] == grouped[shard]["bindings"], \
            f"shard {shard} binding total differs from resource rows"
    submitted = report["resident_graph"]["submitted"]
    assert type(submitted) is int and submitted > 0, "report has no submitted graph commands"
    return histogram


def _warm_histogram_signature(wrapper):
    histogram = wrapper["first_command_bindings"]
    return tuple((key, json.dumps(histogram[key], sort_keys=True)) for key in (
        "observed", "submitted", "buffer_bindings", "raw_weight_bindings", "shards", "resources",
    ))


def _qualify_lifetimes(previous, wrapper, job, previous_cache_count):
    """Check one prime/scored request against its expected layout transition."""
    layout = job["expected_layout"]
    mode = job["mode"]
    prime = job["prime"]
    assert layout in LAYOUTS and mode in WRAPPER_MODES
    assert wrapper["layout"] == layout, "reported resource layout differs from expected_layout"
    assert wrapper["mode"] == mode, "reported view treatment differs from request"
    assert prime == (not job["scored"]), "prime/scored metadata is inconsistent"
    assert not prime or mode == "persistent", "only persistent requests may be layout primes"

    previous_after = None if previous is None else previous["after_job"]
    previous_layout = None if previous is None else previous["layout"]
    transition = previous_after is not None and previous_layout != layout
    expected_page = None if previous_after is None else previous_after["page_size"]
    expected_known = None if previous_after is None else previous_after["known_tensors"]
    before = _validate_cache_shape(
        wrapper["preparation"], "preparation", layout,
        expected_page_size=expected_page, expected_known_tensors=expected_known,
        allow_empty_known=previous_after is None,
    )
    after = _validate_cache_shape(
        wrapper["after_job"], "after_job", layout,
        expected_page_size=before["page_size"], expected_known_tensors=expected_known,
    )
    cold_layout_switch = previous_after is None and layout == "tensor"
    assert before["layout_changed"] == int(transition or cold_layout_switch), \
        "unexpected layout_changed preparation flag"
    assert after["layout_changed"] == 0, "after_job reports a layout transition"
    assert after["constructed"] == after["retired"] == after["same_backing"] == 0, \
        "after_job reports wrapper construction or retirement"
    assert after["tensor_views_cleared"] == 0, "after_job reports a tensor-view clear"

    if previous_after is None:
        assert prime and job["cold_prime"], "first request must be the cold layout prime"
        assert before["known_tensors"] == 0, "cold preparation must begin before any tensor is known"
        assert before["layout_changed"] == int(layout == "tensor"), \
            "cold preparation layout flag differs from the requested layout"
        for key in ("shards", "mappings", "mapped_bytes", "unique_mapped_bytes", "overlapping_bytes"):
            assert before[key] == 0, f"initial preparation.{key} must describe an empty wrapper bank"
        assert before["constructed"] == before["retired"] == before["same_backing"] == 0
        assert before["tensor_views"] == before["tensor_views_cleared"] == 0
        assert before["factory_calls"] == 0, "cold preparation unexpectedly created a resource"
        assert after["known_tensors"] > 0, "cold prime did not discover any tensor views"
        assert after["mappings"] > 0, "cold prime did not construct any mapped resource"
        assert after["shards"] > 0 and after["mapped_bytes"] > 0
        assert after["tensor_views"] > 0, "cold prime did not build raw tensor views"
        assert after["tensor_views"] == after["known_tensors"], \
            "cold prime did not build one raw view for every known tensor"
        assert after["tensor_views_created"] - before["tensor_views_created"] == after["known_tensors"], \
            "cold prime raw-view creation count differs from discovered tensors"
        assert after["factory_calls"] - before["factory_calls"] == after["mappings"], \
            "cold prime resource factory delta differs from constructed mappings"
        assert after["generation"] == before["generation"] \
            and after["view_generation"] == before["view_generation"], \
            "cold generation changed a layout prepared before request generation"
    elif transition:
        assert prime, "layout changes are only allowed at excluded primes"
        for key in ("known_tensors", "page_size"):
            assert before[key] == previous_after[key], f"layout transition changed {key}"
        assert before["shards"] == previous_after["shards"], \
            "layout transition changed the number of mmap owners"
        assert before["constructed"] == before["mappings"], \
            "transition did not construct every new resource before the prime"
        assert before["retired"] == previous_after["mappings"], \
            "transition did not retire every old resource before the prime"
        assert before["same_backing"] == previous_after["shards"], \
            "same_backing does not cover all unchanged mmap owners"
        assert before["factory_calls"] - previous_after["factory_calls"] == before["constructed"], \
            "factory-call delta differs from constructed resource count"
        assert before["generation"] == previous_after["generation"] + 1, \
            "layout transition did not advance wrapper generation exactly once"
        assert before["tensor_views_cleared"] == previous_after["tensor_views"], \
            "layout transition did not clear the previous raw-view cache"
        assert before["view_generation"] == previous_after["view_generation"] + 1, \
            "layout transition did not advance raw-view generation exactly once"
        assert before["tensor_views"] == before["known_tensors"], \
            "layout transition did not rebuild views for every known tensor"
        assert before["tensor_views_created"] - previous_after["tensor_views_created"] == before["known_tensors"], \
            "layout transition view creation count differs from known_tensors"
        for key in ("mappings", "mapped_bytes", "unique_mapped_bytes", "overlapping_bytes"):
            assert after[key] == before[key], f"prime changed post-transition {key}"
        for key in ("generation", "factory_calls", "tensor_views_created", "tensor_views"):
            assert after[key] == before[key], f"prime lazily changed {key} after layout preparation"
    else:
        for key in ("shards", "mappings", "mapped_bytes", "unique_mapped_bytes", "overlapping_bytes"):
            assert before[key] == previous_after[key], f"unchanged layout changed {key} during preparation"
        assert before["constructed"] == before["retired"] == before["same_backing"] == 0, \
            "unchanged layout constructed or retired resources"
        assert before["factory_calls"] == previous_after["factory_calls"]
        assert before["generation"] == previous_after["generation"]

        clear_views = mode == "views"
        expected_view_generation = previous_after["view_generation"] + int(clear_views)
        assert before["view_generation"] == expected_view_generation, \
            "view generation differs from requested persistent/views treatment"
        if clear_views:
            assert before["tensor_views_cleared"] == previous_after["tensor_views"]
            assert before["tensor_views"] == 0, "views treatment left stale raw views before generation"
            assert before["tensor_views_created"] == previous_after["tensor_views_created"]
        else:
            assert before["tensor_views_cleared"] == 0
            assert before["tensor_views"] == previous_after["tensor_views"]
            assert before["tensor_views_created"] == previous_after["tensor_views_created"]

        for key in ("mappings", "mapped_bytes", "unique_mapped_bytes", "overlapping_bytes"):
            assert after[key] == before[key], f"request changed resource layout metric {key}"
        assert after["generation"] == before["generation"]
        assert after["factory_calls"] == before["factory_calls"], \
            "scored request constructed a mapped resource"
        assert after["tensor_views_cleared"] == 0
        assert after["view_generation"] == before["view_generation"]
        if clear_views:
            assert after["tensor_views"] > 0, "views treatment did not rebuild raw tensor views"
            assert after["tensor_views_created"] - before["tensor_views_created"] == after["tensor_views"], \
                "views treatment rebuild count differs from active raw views"
        else:
            assert after["tensor_views"] == before["tensor_views"]
            assert after["tensor_views_created"] == before["tensor_views_created"]

    # The cold prime may populate converted F32 weights. Every later prime and
    # every scored job must preserve that cache exactly.
    if job["cold_prime"]:
        cache_count = _counter(wrapper["converted_cache"], "after_count", "converted_cache")
    else:
        cache_count = _validate_converted_cache(wrapper, previous_cache_count)
    return cache_count


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


def _run_worker(*, jobs, args, output_dir, manifest_handle, records, baseline,
                baseline_sha256, binary_sha256, source_hashes, controls):
    stderr_path = output_dir / "weight-extents-worker.stderr"
    env = {key: value for key, value in os.environ.items() if not key.startswith("DRAMA3_")}
    env.pop("QWEN35_NATIVE_TOKENIZER_OFF", None)
    env.update(controls)
    previous = None
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
                output = output_dir / job["output_name"]
                assert not output.exists(), f"refusing existing waveform: {output}"
                load_before = os.getloadavg()
                started = time.monotonic()
                process.stdin.write(json.dumps({
                    "input": INPUT,
                    "output": str(output),
                    "layout": job["expected_layout"],
                    "wrappers": job["mode"],
                }) + "\n")
                process.stdin.flush()
                event = _event_for_request(process, stderr_path)
                wall = time.monotonic() - started
                wav_sha256 = sha(output)
                report = event["report"]
                validate(report, wav_sha256, baseline)
                assert report["host_profile"]["enabled"]
                wrapper = event["weight_wrappers"]
                cache_count = _qualify_lifetimes(previous, wrapper, job, previous_cache_count)

                qualify_first_command_bindings(wrapper, report, expected_command_index)
                if previous_histogram is not None and not job["cold_prime"]:
                    qualify_command_index_after(previous_histogram, previous_report, wrapper)
                    if (previous_histogram.get("layout") == wrapper["layout"]
                            and previous_histogram.get("warm_histogram_eligible", False)):
                        assert _warm_histogram_signature(previous_histogram) == \
                            _warm_histogram_signature(wrapper), \
                            "resource histogram changed within a stable layout block"

                item = {
                    "design": "weight-extents",
                    "worker": 0,
                    "job": job["job"],
                    "block": job["block"],
                    "expected_layout": job["expected_layout"],
                    "layout": wrapper["layout"],
                    "prime": job["prime"],
                    "cold_prime": job["cold_prime"],
                    "scored": job["scored"],
                    "warm": job["warm"],
                    "mode": job["mode"],
                    "position": job["position"],
                    "predecessor": job["predecessor"],
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
                    "job", "block", "expected_layout", "prime", "mode", "position",
                    "predecessor", "request_wall_seconds", "wav_sha256",
                )}), flush=True)

                previous_histogram = dict(
                    wrapper,
                    warm_histogram_eligible=not job["cold_prime"],
                    layout=wrapper["layout"],
                )
                previous_report = report
                expected_command_index += report["resident_graph"]["submitted"]
                previous = {"layout": job["expected_layout"], "after_job": wrapper["after_job"]}
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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", required=True, type=Path)
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--baseline", required=True, type=Path)
    parser.add_argument("--layout-order", choices=tuple(_LAYOUT_ORDERS),
                        default=DEFAULT_LAYOUT_ORDER)
    args = parser.parse_args()
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
    jobs = extent_jobs(args.layout_order)

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
    manifest = args.output / "weight-extents-worker.json"
    with manifest.open("x") as handle:
        records = []
        _write_manifest(handle, records)
        _run_worker(
            jobs=jobs, args=args, output_dir=args.output, manifest_handle=handle,
            records=records, baseline=baseline, baseline_sha256=baseline_sha256,
            binary_sha256=binary_sha256, source_hashes=source_hashes, controls=controls,
        )
        assert len(records) == 20
        scored = [item for item in records if item["scored"]]
        primes = [item for item in records if item["prime"]]
        assert len(scored) == 16 and len(primes) == 4
        print(json.dumps({
            "status": "PASS",
            "design": "weight-extents",
            "layout_order": args.layout_order,
            "jobs": len(records),
            "scored_jobs": len(scored),
            "excluded_primes": len(primes),
            "same_sampler_inputs_readbacks_waits": True,
            "same_mmap_backing": True,
            "claim_scope": "net layout effect; resource count remains part of layout",
        }), flush=True)


if __name__ == "__main__":
    main()
