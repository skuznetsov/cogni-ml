#!/usr/bin/env python3
"""Offline schedule, layout-lifetime, and receipt mutation tests."""

import copy
import importlib.util
import sys
import unittest
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[1] / "scripts/drama3/measure_weight_extents.py"
sys.path.insert(0, str(SCRIPT.parent))
SPEC = importlib.util.spec_from_file_location("measure_weight_extents", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
driver = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = driver
SPEC.loader.exec_module(driver)


PAGE = 4096


def stats(
    *,
    layout="shard",
    mappings=2,
    mapped_bytes=12288,
    unique_mapped_bytes=12288,
    overlapping_bytes=0,
    shards=2,
    narrow=None,
    layout_changed=0,
    known_tensors=3,
    page_size=PAGE,
    factory_calls=2,
    generation=1,
    constructed=0,
    retired=0,
    same_backing=0,
    view_generation=0,
    tensor_views_created=3,
    tensor_views=3,
    tensor_views_cleared=0,
):
    return {
        "narrow": int(layout == "tensor") if narrow is None else narrow,
        "layout_changed": layout_changed,
        "page_size": page_size,
        "known_tensors": known_tensors,
        "shards": shards,
        "mappings": mappings,
        "mapped_bytes": mapped_bytes,
        "unique_mapped_bytes": unique_mapped_bytes,
        "overlapping_bytes": overlapping_bytes,
        "factory_calls": factory_calls,
        "generation": generation,
        "constructed": constructed,
        "retired": retired,
        "same_backing": same_backing,
        "view_generation": view_generation,
        "tensor_views_created": tensor_views_created,
        "tensor_views": tensor_views,
        "tensor_views_cleared": tensor_views_cleared,
    }


def shard_resources():
    return [
        {"shard": 1, "file_offset": 0, "mapped_bytes": 8192, "bindings": 4},
        {"shard": 2, "file_offset": 0, "mapped_bytes": 4096, "bindings": 2},
    ]


def tensor_resources():
    return [
        {"shard": 1, "file_offset": 0, "mapped_bytes": 8192, "bindings": 2},
        {"shard": 1, "file_offset": 4096, "mapped_bytes": 8192, "bindings": 2},
        {"shard": 2, "file_offset": 0, "mapped_bytes": 4096, "bindings": 2},
    ]


def histogram(resources, command_index=0):
    grouped = {}
    for resource in resources:
        row = grouped.setdefault(resource["shard"], {"mapped_bytes": 0, "bindings": 0})
        row["mapped_bytes"] += resource["mapped_bytes"]
        row["bindings"] += resource["bindings"]
    total = sum(resource["mapped_bytes"] for resource in resources)
    unique = 0
    for shard in set(resource["shard"] for resource in resources):
        ranges = sorted(
            (resource["file_offset"], resource["file_offset"] + resource["mapped_bytes"])
            for resource in resources if resource["shard"] == shard
        )
        start, end = ranges[0]
        shard_union = 0
        for left, right in ranges[1:]:
            if left <= end:
                end = max(end, right)
            else:
                shard_union += end - start
                start, end = left, right
        shard_union += end - start
        unique += shard_union
    return {
        "command_index": command_index,
        "observed": True,
        "submitted": True,
        "buffer_bindings": 17,
        "raw_weight_bindings": 6,
        "shards": [
            {"shard": shard, **grouped[shard]}
            for shard in sorted(grouped)
        ],
        "resources": copy.deepcopy(resources),
    }, total, unique


def wrapper(layout, mode, preparation, after_job, resources, *, cache_before=23,
            cache_after=23, unchanged=True, command_index=0):
    bindings, total, unique = histogram(resources, command_index)
    after_job["mapped_bytes"] = total
    after_job["unique_mapped_bytes"] = unique
    after_job["overlapping_bytes"] = total - unique
    if preparation["mappings"] > 0:
        preparation["mapped_bytes"] = total
        preparation["unique_mapped_bytes"] = unique
        preparation["overlapping_bytes"] = total - unique
    return {
        "layout": layout,
        "mode": mode,
        "preparation": preparation,
        "after_job": after_job,
        "converted_cache": {
            "before_count": cache_before,
            "after_count": cache_after,
            "unchanged": unchanged,
        },
        "first_command_bindings": bindings,
    }


def initial_prime():
    before = stats(
        mappings=0, mapped_bytes=0, unique_mapped_bytes=0, overlapping_bytes=0,
        shards=0, known_tensors=0, factory_calls=0, generation=0, tensor_views_created=0,
        tensor_views=0,
    )
    after = stats(generation=0)
    candidate = wrapper("shard", "persistent", before, after, shard_resources(),
                        cache_before=0, cache_after=23, unchanged=False)
    return candidate


def cold_tensor_prime():
    before = stats(
        layout="tensor", mappings=0, mapped_bytes=0, unique_mapped_bytes=0,
        overlapping_bytes=0, shards=0, known_tensors=0, layout_changed=1,
        factory_calls=0, generation=1, view_generation=1,
        tensor_views_created=0, tensor_views=0,
    )
    after = stats(
        layout="tensor", mappings=3, mapped_bytes=20480, unique_mapped_bytes=16384,
        overlapping_bytes=4096, shards=2, factory_calls=3, generation=1,
        view_generation=1, tensor_views_created=3, tensor_views=3,
    )
    return wrapper("tensor", "persistent", before, after, tensor_resources(),
                   cache_before=0, cache_after=23, unchanged=False)


def transition_prime(previous, layout="tensor"):
    old = previous["after_job"]
    resources = tensor_resources() if layout == "tensor" else shard_resources()
    bindings, total, unique = histogram(resources)
    before = stats(
        layout=layout, mappings=len(resources), mapped_bytes=total, unique_mapped_bytes=unique,
        overlapping_bytes=total - unique, shards=2, layout_changed=1,
        factory_calls=old["factory_calls"] + len(resources),
        generation=old["generation"] + 1, constructed=len(resources), retired=old["mappings"],
        same_backing=old["shards"],
        view_generation=old["view_generation"] + 1,
        tensor_views_created=old["tensor_views_created"] + old["known_tensors"],
        tensor_views=old["known_tensors"], tensor_views_cleared=old["tensor_views"],
    )
    after = copy.deepcopy(before)
    after.update(layout_changed=0, constructed=0, retired=0, same_backing=0,
                 tensor_views_cleared=0)
    return wrapper(layout, "persistent", before, after, resources)


def steady_request(previous, job):
    old = previous["after_job"]
    layout = job["expected_layout"]
    resources = tensor_resources() if layout == "tensor" else shard_resources()
    before = copy.deepcopy(old)
    before.update(layout_changed=0, constructed=0, retired=0, same_backing=0,
                  tensor_views_cleared=0)
    if job["mode"] == "views":
        before.update(view_generation=old["view_generation"] + 1,
                      tensor_views=0, tensor_views_cleared=old["tensor_views"])
    after = copy.deepcopy(before)
    after.update(layout_changed=0, constructed=0, retired=0, same_backing=0,
                 tensor_views_cleared=0)
    if job["mode"] == "views":
        after.update(tensor_views=after["known_tensors"],
                     tensor_views_created=before["tensor_views_created"] + after["known_tensors"])
    return wrapper(layout, job["mode"], before, after, resources)


class WeightExtentDriverSpec(unittest.TestCase):
    def test_schedule_is_balanced_and_records_real_cross_block_predecessors(self):
        jobs = driver.extent_jobs()
        self.assertEqual(len(jobs), 20)
        self.assertEqual([job["expected_layout"] for job in jobs[::5]],
                         ["shard", "tensor", "tensor", "shard"])
        self.assertEqual([job["mode"] for job in jobs[1:5]],
                         ["persistent", "views", "views", "persistent"])
        self.assertEqual(sum(job["prime"] for job in jobs), 4)
        self.assertEqual(sum(job["scored"] for job in jobs), 16)
        self.assertIsNone(jobs[0]["predecessor"])
        self.assertEqual(jobs[5]["predecessor"]["job"], 4)
        self.assertEqual(jobs[5]["predecessor"]["expected_layout"], "shard")
        self.assertEqual(jobs[10]["predecessor"]["job"], 9)
        self.assertEqual(jobs[10]["predecessor"]["expected_layout"], "tensor")
        self.assertEqual(jobs[15]["predecessor"]["job"], 14)
        self.assertEqual(jobs[15]["predecessor"]["expected_layout"], "tensor")
        mirrored = driver.extent_jobs("tensor-shard-shard-tensor")
        self.assertEqual([job["expected_layout"] for job in mirrored[::5]],
                         ["tensor", "shard", "shard", "tensor"])
        self.assertEqual(mirrored[5]["predecessor"]["expected_layout"], "tensor")

    def test_schedule_rejects_wrong_predecessor_layout_or_within_block_order(self):
        jobs = driver.extent_jobs()
        jobs[5]["predecessor"]["expected_layout"] = "tensor"
        with self.assertRaisesRegex(ValueError, "predecessor"):
            driver.validate_extent_jobs(jobs)

        jobs = driver.extent_jobs()
        jobs[7]["mode"] = "persistent"
        with self.assertRaisesRegex(ValueError, "persistent/views/views/persistent"):
            driver.validate_extent_jobs(jobs)

    def test_first_command_histogram_recomputes_extent_union_and_shard_sums(self):
        candidate = transition_prime(initial_prime())
        report = {"resident_graph": {"submitted": 10}}
        driver.qualify_first_command_bindings(candidate, report, expected_index=0)
        self.assertEqual(candidate["after_job"]["mapped_bytes"], 20480)
        self.assertEqual(candidate["after_job"]["unique_mapped_bytes"], 16384)
        self.assertEqual(candidate["after_job"]["overlapping_bytes"], 4096)

    def test_resource_histogram_rejects_extent_offset_union_and_binding_mutations(self):
        mutations = (
            ("misaligned offset", lambda row: row["first_command_bindings"]["resources"][1].__setitem__("file_offset", 1)),
            ("incorrect union", lambda row: row["after_job"].__setitem__("unique_mapped_bytes", 12288)),
            ("incorrect shard sum", lambda row: row["first_command_bindings"]["shards"][0].__setitem__("mapped_bytes", 8191)),
            ("raw binding total", lambda row: row["first_command_bindings"]["resources"][0].__setitem__("bindings", 3)),
            ("missing resource", lambda row: row["first_command_bindings"]["resources"].pop()),
        )
        for name, mutation in mutations:
            with self.subTest(name=name):
                candidate = copy.deepcopy(transition_prime(initial_prime()))
                mutation(candidate)
                with self.assertRaises(AssertionError):
                    driver.qualify_first_command_bindings(
                        candidate, {"resident_graph": {"submitted": 10}}, expected_index=0
                    )

    def test_initial_prime_and_layout_transition_require_empty_then_full_rebuild(self):
        cold = initial_prime()
        cold_job = driver.extent_jobs()[0]
        first_cache = driver._qualify_lifetimes(None, cold, cold_job, None)
        self.assertEqual(first_cache, 23)

        mirrored = cold_tensor_prime()
        mirrored_cache = driver._qualify_lifetimes(
            None, mirrored, driver.extent_jobs("tensor-shard-shard-tensor")[0], None
        )
        self.assertEqual(mirrored_cache, 23)

        transition = transition_prime(cold)
        transition_job = driver.extent_jobs()[5]
        next_cache = driver._qualify_lifetimes(
            {"layout": "shard", "after_job": cold["after_job"]},
            transition, transition_job, previous_cache_count=23,
        )
        self.assertEqual(next_cache, 23)

        mirrored_transition = transition_prime(mirrored, layout="shard")
        driver._qualify_lifetimes(
            {"layout": "tensor", "after_job": mirrored["after_job"]},
            mirrored_transition,
            driver.extent_jobs("tensor-shard-shard-tensor")[5],
            previous_cache_count=23,
        )

        for name, mutate in (
            ("constructor count", lambda row: row["preparation"].__setitem__("constructed", 2)),
            ("same mmap owners", lambda row: row["preparation"].__setitem__("same_backing", 1)),
            ("cache mutation", lambda row: row["converted_cache"].__setitem__("after_count", 24)),
        ):
            with self.subTest(name=name):
                candidate = copy.deepcopy(transition)
                mutate(candidate)
                with self.assertRaises(AssertionError):
                    driver._qualify_lifetimes(
                        {"layout": "shard", "after_job": cold["after_job"]},
                        candidate, transition_job, previous_cache_count=23,
                    )

    def test_transition_prime_must_not_construct_again_during_generation(self):
        cold = initial_prime()
        transition = transition_prime(cold)
        transition["after_job"]["factory_calls"] += 1
        transition["after_job"]["mappings"] += 1
        transition["after_job"]["mapped_bytes"] += PAGE
        transition["after_job"]["unique_mapped_bytes"] += PAGE
        with self.assertRaisesRegex(AssertionError, "prime changed post-transition"):
            driver._qualify_lifetimes(
                {"layout": "shard", "after_job": cold["after_job"]},
                transition, driver.extent_jobs()[5], previous_cache_count=23,
            )

    def test_views_only_shams_clear_and_rebuild_without_resource_changes(self):
        jobs = driver.extent_jobs()
        cold = initial_prime()
        cache_count = driver._qualify_lifetimes(None, cold, jobs[0], None)
        previous = {"layout": "shard", "after_job": cold["after_job"]}
        for job in jobs[1:5]:
            candidate = steady_request(previous, job)
            cache_count = driver._qualify_lifetimes(previous, candidate, job, cache_count)
            previous = {"layout": job["expected_layout"], "after_job": candidate["after_job"]}
        self.assertEqual(cache_count, 23)

        views = steady_request(previous, jobs[3])
        views["preparation"]["mappings"] += 1
        with self.assertRaisesRegex(AssertionError, "unchanged layout changed mappings"):
            driver._qualify_lifetimes(previous, views, jobs[3], previous_cache_count=23)


if __name__ == "__main__":
    unittest.main()
