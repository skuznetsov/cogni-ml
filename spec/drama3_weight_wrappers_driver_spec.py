#!/usr/bin/env python3
"""Offline controls for the DRAMA-3 weight-wrapper measurement driver."""

import importlib.util
import itertools
import copy
import sys
import unittest
from collections import Counter
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[1] / "scripts/drama3/measure_weight_wrappers.py"
sys.path.insert(0, str(SCRIPT.parent))
SPEC = importlib.util.spec_from_file_location("measure_weight_wrappers", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
driver = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = driver
SPEC.loader.exec_module(driver)


def wrapper_stats(
    *,
    mappings=3,
    mapped_bytes=12000,
    factory_calls=9,
    generation=4,
    constructed=0,
    retired=0,
    same_backing=0,
    view_generation=8,
    tensor_views_cleared=0,
    tensor_views_created=31,
    tensor_views=4,
):
    return {
        "mappings": mappings,
        "mapped_bytes": mapped_bytes,
        "factory_calls": factory_calls,
        "generation": generation,
        "constructed": constructed,
        "retired": retired,
        "same_backing": same_backing,
        "view_generation": view_generation,
        "tensor_views_cleared": tensor_views_cleared,
        "tensor_views_created": tensor_views_created,
        "tensor_views": tensor_views,
    }


def binding_histogram(command_index=41):
    return {
        "command_index": command_index,
        "observed": True,
        "submitted": True,
        "buffer_bindings": 19,
        "raw_weight_bindings": 6,
        "shards": [
            {"shard": 1, "mapped_bytes": 4000, "bindings": 4},
            {"shard": 2, "mapped_bytes": 3000, "bindings": 0},
            {"shard": 3, "mapped_bytes": 5000, "bindings": 2},
        ],
    }


def views_transition(mode):
    before = wrapper_stats()
    after = dict(before)
    if mode in ("views", "fresh"):
        before["view_generation"] += 1
        before["tensor_views_cleared"] = before["tensor_views"]
        before["tensor_views"] = 0
        after.update(
            view_generation=before["view_generation"],
            tensor_views_created=before["tensor_views_created"] + 3,
            tensor_views=3,
        )
    if mode == "fresh":
        before.update(
            factory_calls=before["factory_calls"] + before["mappings"],
            generation=before["generation"] + 1,
            constructed=before["mappings"],
            retired=before["mappings"],
            same_backing=before["mappings"],
        )
        after.update(
            factory_calls=before["factory_calls"],
            generation=before["generation"],
            constructed=0,
            retired=0,
            same_backing=0,
        )
    else:
        after.update(
            generation=before["generation"],
            constructed=0,
            retired=0,
            same_backing=0,
        )
    if mode == "persistent":
        after["tensor_views_created"] = before["tensor_views_created"]
        after["tensor_views"] = before["tensor_views"]
    return {
        "mode": mode,
        "preparation": before,
        "after_job": after,
        "converted_cache": {"before_count": 23, "after_count": 23, "unchanged": True},
        "first_command_bindings": binding_histogram(),
    }


def generated_report(submitted=41):
    return {"resident_graph": {"submitted": submitted}}


def baseline_fixture():
    report = {
        "frames": 60,
        "sampler_inputs": [1.0, 2.0],
        "resident_readbacks": {"control": 4, "diagnostic": 8, "terminal": 12},
        "resident_graph": {
            "submitted": 9,
            "completed": 9,
            "completion_waits": 9,
            "pending": 0,
        },
        "resident_generation_scratch": {"enabled": False},
    }
    return {"report": report, "wav_sha256": "f" * 64}, copy.deepcopy(report)


class WeightWrapperDriverSpec(unittest.TestCase):
    def test_views_schedule_is_seeded_and_balanced_by_position_and_direction(self):
        cycles = 3
        schedule = driver.build_views_schedule(cycles=cycles, seed=8421)
        self.assertEqual(schedule, driver.build_views_schedule(cycles=cycles, seed=8421))
        self.assertEqual(len(schedule), 6 * cycles)
        self.assertEqual(driver.views_request_count(cycles), 1 + 18 * cycles)
        driver.validate_views_schedule(schedule, cycles)

        permutations = set(itertools.permutations(("persistent", "views", "fresh")))
        for cycle in range(cycles):
            block = [tuple(item["order"]) for item in schedule if item["cycle"] == cycle]
            self.assertEqual(set(block), permutations)
            self.assertEqual(len(block), 6)

            positions = [Counter(order[index] for order in block) for index in range(3)]
            for counts in positions:
                self.assertEqual(counts, Counter({mode: 2 for mode in driver.VIEWS_MODES}))

            directions = Counter(
                (left, right)
                for order in block
                for left, right in zip(order, order[1:])
            )
            self.assertEqual(
                directions,
                Counter({(left, right): 2 for left in driver.VIEWS_MODES
                         for right in driver.VIEWS_MODES if left != right}),
            )

    def test_views_schedule_rejects_a_missing_permutation(self):
        schedule = driver.build_views_schedule(cycles=1, seed=2)
        schedule[-1]["order"] = list(schedule[0]["order"])
        with self.assertRaisesRegex(ValueError, "six unique mode permutations"):
            driver.validate_views_schedule(schedule, cycles=1)

    def test_wrappers_default_preserves_the_legacy_mirrored_persistent_fresh_orders(self):
        self.assertEqual(driver.DEFAULT_DESIGN, "wrappers")
        self.assertEqual(
            driver.wrapper_orders(cycles=2),
            (
                ("persistent", "fresh", "fresh", "persistent") * 2,
                ("fresh", "persistent", "persistent", "fresh") * 2,
            ),
        )
        self.assertEqual(driver.WRAPPER_MODES, ("persistent", "fresh"))

    def test_views_request_plan_has_one_prime_and_all_seeded_warm_permutations(self):
        jobs = driver._views_jobs(cycles=2, seed=8421)
        self.assertEqual(len(jobs), driver.views_request_count(2))
        self.assertEqual(jobs[0]["mode"], "persistent")
        self.assertFalse(jobs[0]["warm"])
        warm = jobs[1:]
        self.assertEqual(len(warm), 36)
        self.assertEqual(len({job["output_name"] for job in jobs}), len(jobs))
        self.assertEqual([job["predecessor_mode"] for job in warm[:1]], ["persistent"])
        for cycle in range(2):
            cycle_jobs = [job for job in warm if job["cycle"] == cycle]
            permutations = {}
            for job in cycle_jobs:
                permutations.setdefault(job["permutation"], []).append(job)
            self.assertEqual(len(permutations), 6)
            self.assertTrue(all(len(rows) == 3 for rows in permutations.values()))
            for rows in permutations.values():
                rows.sort(key=lambda job: job["position"])
                self.assertEqual([job["mode"] for job in rows], rows[0]["permutation"].split("-"))
        for previous, current in zip(jobs, jobs[1:]):
            if current["warm"]:
                self.assertEqual(current["predecessor_mode"], previous["mode"])

    def test_legacy_request_plan_keeps_mirrored_workers_and_prime_counts(self):
        left = driver._wrapper_jobs(worker=0, cycles=2)
        right = driver._wrapper_jobs(worker=1, cycles=2)
        self.assertEqual(len(left), 9)
        self.assertEqual(len(right), 9)
        self.assertEqual([job["mode"] for job in left[1:]], list(driver.wrapper_orders(2)[0]))
        self.assertEqual([job["mode"] for job in right[1:]], list(driver.wrapper_orders(2)[1]))
        self.assertEqual(left[0]["mode"], right[0]["mode"])

    def test_views_mode_clears_raw_views_without_recreating_mapped_wrappers(self):
        previous = wrapper_stats()
        for mode in ("persistent", "views"):
            with self.subTest(mode=mode):
                wrapper = views_transition(mode)
                driver.qualify(previous, wrapper, warm=True, mode=mode, design="views")

    def test_views_qualification_rejects_mode_count_generation_and_cache_mutations(self):
        mutations = (
            ("mode", lambda row: row.__setitem__("mode", "fresh")),
            ("wrapper factory calls", lambda row: row["preparation"].__setitem__("factory_calls", 10)),
            ("wrapper generation", lambda row: row["preparation"].__setitem__("generation", 5)),
            ("view generation", lambda row: row["preparation"].__setitem__("view_generation", 8)),
            ("views were not cleared", lambda row: row["preparation"].__setitem__("tensor_views_cleared", 0)),
            ("wrong cleared view count", lambda row: row["preparation"].__setitem__("tensor_views_cleared", 3)),
            ("prepare created views", lambda row: row["preparation"].__setitem__("tensor_views_created", 32)),
            ("post-job view count", lambda row: row["after_job"].__setitem__("tensor_views_created", 35)),
            ("converted cache changed", lambda row: row["converted_cache"].__setitem__("after_count", 24)),
        )
        for name, mutation in mutations:
            with self.subTest(name=name):
                wrapper = views_transition("views")
                mutation(wrapper)
                with self.assertRaises(AssertionError):
                    driver.qualify(wrapper_stats(), wrapper, warm=True, mode="views", design="views")

    def test_fresh_qualification_rejects_missing_same_backing_proof(self):
        wrapper = views_transition("fresh")
        wrapper["preparation"]["same_backing"] -= 1
        with self.assertRaisesRegex(AssertionError, "same_backing"):
            driver.qualify(wrapper_stats(), wrapper, warm=True, mode="fresh", design="views")

    def test_views_warm_cache_must_match_the_previous_count(self):
        wrapper = views_transition("views")
        with self.assertRaisesRegex(AssertionError, "differs from previous job"):
            driver.qualify(
                wrapper_stats(), wrapper, warm=True, mode="views", design="views",
                previous_cache_count=24,
            )

    def test_persistent_preparation_cannot_change_active_raw_views(self):
        wrapper = views_transition("persistent")
        wrapper["preparation"]["tensor_views"] += 1
        with self.assertRaisesRegex(AssertionError, "persistent preparation changed active views"):
            driver.qualify(
                wrapper_stats(), wrapper, warm=True, mode="persistent", design="views"
            )

    def test_binding_histogram_requires_complete_known_shard_coverage_and_job_index(self):
        wrapper = views_transition("persistent")
        report = generated_report(submitted=41)
        driver.qualify_first_command_bindings(wrapper, report, expected_index=41, mapped_bytes=12000)

        mutations = (
            ("missing zero-binding shard", lambda hist: hist["shards"].pop(1)),
            ("noncontiguous shard ordinals", lambda hist: hist["shards"][2].__setitem__("shard", 4)),
            ("raw binding sum", lambda hist: hist.__setitem__("raw_weight_bindings", 7)),
            ("mapped extent sum", lambda hist: hist["shards"][0].__setitem__("mapped_bytes", 3999)),
            ("command index", lambda hist: hist.__setitem__("command_index", 42)),
            ("observation", lambda hist: hist.__setitem__("observed", False)),
            ("submission", lambda hist: hist.__setitem__("submitted", False)),
        )
        for name, mutation in mutations:
            with self.subTest(name=name):
                candidate = copy.deepcopy(wrapper)
                mutation(candidate["first_command_bindings"])
                with self.assertRaises(AssertionError):
                    driver.qualify_first_command_bindings(
                        candidate, report, expected_index=41, mapped_bytes=12000
                    )

    def test_warm_binding_histograms_match_without_collapsing_command_indices(self):
        first = views_transition("persistent")
        second = views_transition("views")
        first["first_command_bindings"] = binding_histogram(command_index=41)
        second["first_command_bindings"] = binding_histogram(command_index=50)
        driver.qualify_warm_binding_histogram(first, second)

        second["first_command_bindings"]["shards"][2]["bindings"] += 1
        with self.assertRaisesRegex(AssertionError, "warm first-command binding histograms differ"):
            driver.qualify_warm_binding_histogram(first, second)

    def test_binding_command_index_advances_by_previous_report_submission_count(self):
        previous = views_transition("persistent")
        previous["first_command_bindings"] = binding_histogram(command_index=41)
        current = views_transition("views")
        current["first_command_bindings"] = binding_histogram(command_index=83)
        driver.qualify_command_index_after(previous, generated_report(submitted=42), current)
        with self.assertRaisesRegex(AssertionError, "unexpected first-command index"):
            driver.qualify_command_index_after(previous, generated_report(submitted=41), current)

    def test_report_and_waveform_mutations_against_saved_baseline_fail(self):
        baseline, report = baseline_fixture()
        driver.validate(report, baseline["wav_sha256"], baseline)
        changed_baseline = copy.deepcopy(baseline)
        changed_baseline["report"]["frames"] += 1
        with self.assertRaisesRegex(AssertionError, "frames changed"):
            driver.validate(report, baseline["wav_sha256"], changed_baseline)
        changed_waveform = copy.deepcopy(baseline)
        changed_waveform["wav_sha256"] = "0" * 64
        with self.assertRaisesRegex(AssertionError, "waveform changed"):
            driver.validate(report, baseline["wav_sha256"], changed_waveform)


if __name__ == "__main__":
    unittest.main()
