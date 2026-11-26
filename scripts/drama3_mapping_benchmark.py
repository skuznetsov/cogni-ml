#!/usr/bin/env python3
"""Controlled file-cache A/B harness for the DRAMA-3 native Metal CLI.

Darwin only. ``mincore-zero`` is a VM file-cache observation, not a claim about
SSD/controller caches. The resident arm is prepared by touching mapped pages.
The fixture-only ``--self-test`` never opens model weights or starts inference.
"""

from __future__ import annotations

import argparse
from contextlib import ExitStack
import ctypes
import errno
import hashlib
import json
import math
import mmap
import os
import platform
import re
import statistics
import subprocess
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable


ROOT = Path(__file__).resolve().parent.parent
GENERATOR = ROOT / "examples/drama3_generate.cr"
RUN_SAFE = ROOT / "scripts/run_safe.sh"
SOURCE_ROOT = ROOT / "src/ml/audio/drama3"
METAL_SOURCE_ROOT = ROOT / "src/ml/metal"
MODEL_FILES_EXPECTED = 6
FIXTURE_SIZE = 1024 * 1024
COPY_CHUNK = 16 * 1024 * 1024
MAX_WARM_REPAIR_PASSES = 2
# User-authorized experiment reserve: 15%. The observed cold drop was at
# least 45 percentage points; admission is an estimate, the sampled guard
# remains authoritative. Warm preparation has already faulted model pages.
NONRESIDENT_LAUNCH_FREE_MIN = 60
RESIDENT_LAUNCH_FREE_MIN = 20
RESERVE_STOP = 15
WAIT_HEADROOM_SECONDS = 60
RUN_TIMEOUT_SECONDS = 300
RUN_MAX_MEMORY_MIB = 40960
PROMPT = "Hello."
SEED = 0
MAX_FRAMES = 128
MS_SYNC = 0x10
MS_INVALIDATE = 0x02


class HarnessError(RuntimeError):
    pass


class HeadroomBlocked(HarnessError):
    pass


class CacheStateError(HarnessError):
    pass


class DarwinVM:
    """Small read-only Darwin mmap/mincore/msync bridge."""

    def __init__(self) -> None:
        if platform.system() != "Darwin":
            raise HarnessError("this cache-control harness requires Darwin")
        self.lib = ctypes.CDLL("/usr/lib/libSystem.B.dylib", use_errno=True)
        self.lib.mmap.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int,
                                  ctypes.c_int, ctypes.c_int, ctypes.c_longlong]
        self.lib.mmap.restype = ctypes.c_void_p
        self.lib.munmap.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
        self.lib.munmap.restype = ctypes.c_int
        self.lib.mincore.argtypes = [ctypes.c_void_p, ctypes.c_size_t,
                                     ctypes.POINTER(ctypes.c_ubyte)]
        self.lib.mincore.restype = ctypes.c_int
        self.lib.msync.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int]
        self.lib.msync.restype = ctypes.c_int
        self.lib.memcpy.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t]
        self.lib.memcpy.restype = ctypes.c_void_p
        self.page_size = os.sysconf("SC_PAGE_SIZE")
        self.map_shared = mmap.MAP_SHARED
        self.map_private = mmap.MAP_PRIVATE

    @staticmethod
    def _raise_errno(operation: str) -> None:
        code = ctypes.get_errno()
        raise OSError(code, f"{operation}: {os.strerror(code)}")

    def map_file(self, path: Path, private: bool = False) -> tuple[int, int]:
        before = checked_file_stat(path)
        flags = self.map_private if private else self.map_shared
        fd_flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
        fd = os.open(path, fd_flags)
        try:
            opened = os.fstat(fd)
            if stat_identity(opened) != stat_identity(before):
                raise HarnessError(f"model file changed while opening: {path.name}")
            ptr = self.lib.mmap(None, opened.st_size, mmap.PROT_READ, flags, fd, 0)
            if ptr == ctypes.c_void_p(-1).value:
                self._raise_errno("mmap")
            address = ctypes.cast(ptr, ctypes.c_void_p).value
            if address is None:
                raise HarnessError("mmap returned a null address")
            return address, opened.st_size
        finally:
            os.close(fd)

    def unmap(self, address: int, length: int) -> None:
        if self.lib.munmap(ctypes.c_void_p(address), length) != 0:
            self._raise_errno("munmap")

    def residency_vector(self, address: int, length: int) -> Any:
        count = (length + self.page_size - 1) // self.page_size
        vector = (ctypes.c_ubyte * count)()
        if self.lib.mincore(ctypes.c_void_p(address), length, vector) != 0:
            self._raise_errno("mincore")
        return vector

    def resident_pages(self, address: int, length: int) -> tuple[int, int]:
        vector = self.residency_vector(address, length)
        resident = sum(1 for item in vector if item & 1)
        return resident, len(vector)

    def missing_ranges(self, address: int, length: int) -> list[tuple[int, int]]:
        vector = self.residency_vector(address, length)
        ranges = []
        index = 0
        while index < len(vector):
            if vector[index] & 1:
                index += 1
                continue
            start = index
            while index < len(vector) and not vector[index] & 1:
                index += 1
            offset = start * self.page_size
            ranges.append((offset, min(length, index * self.page_size) - offset))
        return ranges

    def invalidate(self, address: int, length: int) -> None:
        if self.lib.msync(ctypes.c_void_p(address), length, MS_SYNC | MS_INVALIDATE) != 0:
            self._raise_errno("msync(MS_SYNC|MS_INVALIDATE)")

    def touch_all(self, address: int, length: int,
                  progress: Callable[[], None] | None = None) -> int:
        return self.touch_ranges(address, [(0, length)], progress)

    def touch_ranges(self, address: int, ranges: list[tuple[int, int]],
                     progress: Callable[[], None] | None = None) -> int:
        """Read selected mapped ranges into one reusable 16 MiB sink."""
        sink = ctypes.create_string_buffer(COPY_CHUNK)
        total = 0
        last_check = time.monotonic()
        for offset, length in ranges:
            copied = 0
            while copied < length:
                size = min(COPY_CHUNK, length - copied)
                self.lib.memcpy(ctypes.cast(sink, ctypes.c_void_p),
                                ctypes.c_void_p(address + offset + copied), size)
                copied += size
                total += size
                now = time.monotonic()
                if progress and now - last_check >= 0.5:
                    progress()
                    last_check = now
        if progress:
            progress()
        return total


def stat_identity(st: os.stat_result) -> dict[str, int]:
    return {"device": st.st_dev, "inode": st.st_ino, "size": st.st_size,
            "mtime_ns": st.st_mtime_ns, "ctime_ns": st.st_ctime_ns}


def checked_file_stat(path: Path) -> os.stat_result:
    try:
        st = path.lstat()
    except OSError as ex:
        raise HarnessError(f"cannot stat model file {path}: {ex}") from ex
    if path.is_symlink() or not os.path.isfile(path):
        raise HarnessError(f"model file must be a regular, non-symlink file: {path}")
    if st.st_size <= 0:
        raise HarnessError(f"model file is empty: {path}")
    return st


def model_files(model_dir: Path) -> list[Path]:
    if model_dir.is_symlink() or not model_dir.is_dir():
        raise HarnessError("--model must name a real directory, not a symlink")
    paths = sorted(model_dir.glob("*.safetensors"))
    if len(paths) != MODEL_FILES_EXPECTED:
        raise HarnessError(f"expected exactly six top-level .safetensors files; found {len(paths)}")
    for path in paths:
        checked_file_stat(path)
        if path.resolve(strict=True).parent != model_dir.resolve(strict=True):
            raise HarnessError(f"model weights must resolve directly inside --model: {path.name}")
    return paths


def snapshot_model(paths: list[Path]) -> dict[str, dict[str, int]]:
    return {path.name: stat_identity(checked_file_stat(path)) for path in paths}


def assert_model_unchanged(paths: list[Path], baseline: dict[str, dict[str, int]]) -> None:
    now = snapshot_model(paths)
    if now != baseline:
        raise HarnessError("model file stat identity changed during benchmark")


def lsof_pids(path: Path) -> list[int]:
    try:
        result = subprocess.run(["/usr/sbin/lsof", "-t", "--", str(path)],
                                text=True, capture_output=True, timeout=10)
    except (OSError, subprocess.TimeoutExpired) as ex:
        raise HarnessError(f"cannot verify open model descriptors with lsof: {ex}") from ex
    if result.returncode not in (0, 1):
        raise HarnessError(f"lsof failed for {path.name}: exit {result.returncode}")
    return [int(line) for line in result.stdout.splitlines() if line.strip().isdigit()]


def require_no_open_model_files(paths: list[Path]) -> None:
    open_files = [(path.name, lsof_pids(path)) for path in paths]
    open_files = [(name, pids) for name, pids in open_files if pids]
    if open_files:
        details = ", ".join(f"{name}:pids={pids}" for name, pids in open_files)
        raise HarnessError(f"model files have open descriptors; refusing cache control ({details})")


def query_cache(vm: DarwinVM, paths: list[Path]) -> dict[str, Any]:
    files: dict[str, dict[str, int]] = {}
    total_pages = resident_pages = 0
    for path in paths:
        address, length = vm.map_file(path)
        try:
            resident, pages = vm.resident_pages(address, length)
        finally:
            vm.unmap(address, length)
        files[path.name] = {"resident_pages": resident, "total_pages": pages}
        resident_pages += resident
        total_pages += pages
    return {"resident_pages": resident_pages, "total_pages": total_pages,
            "resident_bytes_mincore_estimate": resident_pages * vm.page_size,
            "total_bytes_mincore_estimate": total_pages * vm.page_size,
            "mincore_bytes_are_not_pinned_ram": True,
            "all_resident": resident_pages == total_pages,
            "zero_resident": resident_pages == 0, "files": files}


def parse_free_percent(output: str) -> int:
    match = re.search(r"System-wide memory free percentage:\s*(\d+)\s*%", output)
    if not match:
        raise HarnessError("cannot parse memory_pressure -Q free percentage")
    return int(match.group(1))


def free_percent() -> int:
    try:
        result = subprocess.run(["/usr/bin/memory_pressure", "-Q"],
                                text=True, capture_output=True, timeout=5)
    except (OSError, subprocess.TimeoutExpired) as ex:
        raise HarnessError(f"memory_pressure -Q failed: {ex}") from ex
    if result.returncode != 0:
        raise HarnessError(f"memory_pressure -Q exited {result.returncode}")
    return parse_free_percent(result.stdout + "\n" + result.stderr)


def wait_for_headroom(minimum: int, *, strict: bool, phase: str,
                      series: list[dict[str, Any]], timeout_seconds: int = WAIT_HEADROOM_SECONDS) -> int:
    started = time.monotonic()
    while True:
        value = free_percent()
        series.append({"phase": phase, "free_percent": value,
                       "wait_elapsed_seconds": round(time.monotonic() - started, 3)})
        admitted = value > minimum if strict else value >= minimum
        if admitted:
            return value
        if time.monotonic() - started >= timeout_seconds:
            op = ">" if strict else ">="
            raise HeadroomBlocked(f"free memory stayed below {op}{minimum}% for {timeout_seconds}s ({value}%)")
        time.sleep(min(2.0, max(0.1, timeout_seconds - (time.monotonic() - started))))


def check_prep_reserve(series: list[dict[str, Any]], phase: str) -> int:
    value = free_percent()
    series.append({"phase": phase, "free_percent": value})
    if value <= RESERVE_STOP:
        raise HeadroomBlocked(f"cache preparation crossed the {RESERVE_STOP}% free-memory reserve ({value}%)")
    return value


def normalize_nonresident(vm: DarwinVM, paths: list[Path],
                          series: list[dict[str, Any]]) -> dict[str, Any]:
    """Start each arm from an exact zero-mincore state using targeted invalidation."""
    before = query_cache(vm, paths)
    require_no_open_model_files(paths)
    started = time.perf_counter()
    for path in paths:
        address, length = vm.map_file(path)
        try:
            vm.invalidate(address, length)
            resident, pages = vm.resident_pages(address, length)
            if resident:
                raise CacheStateError(f"{path.name}: cache reset left {resident}/{pages} pages resident")
        finally:
            vm.unmap(address, length)
        check_prep_reserve(series, "during_cache_normalization")
    after = query_cache(vm, paths)
    if not after["zero_resident"]:
        raise CacheStateError("targeted cache normalization did not leave exactly zero mincore pages")
    return {"cache_before_normalization": before,
            "cache_after_normalization": after,
            "cache_normalization_seconds": time.perf_counter() - started}


def require_cache_state(snapshot: dict[str, Any], state: str, phase: str) -> None:
    key = "zero_resident" if state == "nonresident" else "all_resident"
    if not snapshot[key]:
        raise CacheStateError(
            f"{state} {phase}: {snapshot['resident_pages']}/{snapshot['total_pages']} "
            "pages resident; exact declared cache state required")


def prepare_cache(vm: DarwinVM, paths: list[Path], state: str,
                  series: list[dict[str, Any]],
                  record: dict[str, Any] | None = None) -> dict[str, Any]:
    record = {} if record is None else record
    before = query_cache(vm, paths)
    record["cache_before_prepare"] = before
    require_no_open_model_files(paths)
    if state == "nonresident":
        if not before["zero_resident"]:
            raise CacheStateError("nonresident target preparation must start from normalized zero mincore state")
        check_prep_reserve(series, "during_nonresident_prepare")
    elif state == "resident":
        try:
            # Mappings do not pin pages. Keep helpers alive, repair only missing
            # pages at most twice, then demand exact residency after cleanup.
            # ExitStack also releases helpers on a reserve/error exit.
            with ExitStack() as held:
                mappings = []
                for path in paths:
                    address, length = vm.map_file(path)
                    held.callback(vm.unmap, address, length)
                    mappings.append((path, address, length))

                def reserve_check() -> None:
                    check_prep_reserve(series, "during_resident_prepare")

                for path, address, length in mappings:
                    vm.touch_all(address, length, reserve_check)
                snapshot = query_cache(vm, paths)
                repairs = record["warm_repair_passes"] = []
                for iteration in range(MAX_WARM_REPAIR_PASSES):
                    if snapshot["all_resident"]:
                        break
                    repair = {"pass": iteration + 1, "before": snapshot, "files": {}}
                    repairs.append(repair)
                    started = time.perf_counter()
                    for path, address, length in mappings:
                        ranges = vm.missing_ranges(address, length)
                        detail = {"missing_ranges_offset_length": ranges,
                                  "read_bytes": sum(size for _, size in ranges)}
                        repair["files"][path.name] = detail
                        if ranges:
                            vm.touch_ranges(address, ranges, reserve_check)
                    snapshot = query_cache(vm, paths)
                    repair["after"] = snapshot
                    repair["seconds"] = time.perf_counter() - started
                    repair["read_bytes"] = sum(item["read_bytes"]
                                                for item in repair["files"].values())
                record["cache_while_helpers_mapped"] = snapshot
                reserve_check()
        except BaseException:
            # This query happens after ExitStack has released all helpers. Keep
            # the original exception even if collecting diagnostics also fails.
            try:
                record["cache_after_prepare_failure"] = query_cache(vm, paths)
            except Exception as error:
                record["cache_after_prepare_failure_error"] = str(error)
            raise
    else:
        raise HarnessError(f"unknown cache state: {state}")
    after = query_cache(vm, paths)
    record["cache_after_prepare"] = after
    require_cache_state(after, state, "after helper cleanup")
    check_prep_reserve(series, "after_cache_prepare_cleanup")
    return record


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def fixture_stat(path: Path) -> dict[str, int]:
    return stat_identity(path.stat())


def qualify_fixture(vm: DarwinVM, fixture_dir: Path) -> dict[str, Any]:
    """Prove fresh-map warm visibility, then same-map and fresh-map eviction."""
    with tempfile.TemporaryDirectory(prefix="drama3-cache-fixture-", dir=fixture_dir) as temp:
        path = Path(temp) / "allocated-1mib.bin"
        payload = hashlib.shake_256(b"drama3 non-sparse cache fixture v1").digest(FIXTURE_SIZE)
        with path.open("xb") as handle:
            written = handle.write(payload)
            if written != FIXTURE_SIZE:
                raise HarnessError("fixture write was short")
            handle.flush()
            os.fsync(handle.fileno())
        original_hash = sha256_file(path)
        original_stat = fixture_stat(path)

        address, length = vm.map_file(path)
        try:
            vm.touch_all(address, length)
            warm, pages = vm.resident_pages(address, length)
            if warm != pages:
                raise CacheStateError(f"fixture warm control was {warm}/{pages} pages")
            fresh_warm_results: dict[str, int] = {}
            for mode, private in (("private", True), ("shared", False)):
                fresh, fresh_length = vm.map_file(path, private=private)
                try:
                    fresh_warm, fresh_pages = vm.resident_pages(fresh, fresh_length)
                    fresh_warm_results[mode] = fresh_warm
                    if fresh_warm != pages or fresh_pages != pages:
                        raise CacheStateError(
                            f"fresh {mode} mapping did not observe warm file pages: "
                            f"{fresh_warm}/{fresh_pages}")
                finally:
                    vm.unmap(fresh, fresh_length)
            vm.invalidate(address, length)
            same_map, _ = vm.resident_pages(address, length)
            if same_map != 0:
                raise CacheStateError(f"same shared mapping retained {same_map}/{pages} pages after invalidate")
        finally:
            vm.unmap(address, length)

        fresh_results: dict[str, int] = {}
        for mode, private in (("private", True), ("shared", False)):
            address, length = vm.map_file(path, private=private)
            try:
                resident, fresh_pages = vm.resident_pages(address, length)
                fresh_results[mode] = resident
                if resident != 0 or fresh_pages != pages:
                    raise CacheStateError(f"fresh {mode} mapping was {resident}/{fresh_pages} pages")
            finally:
                vm.unmap(address, length)

        final_hash = sha256_file(path)
        final_stat = fixture_stat(path)
        if final_hash != original_hash or final_stat != original_stat:
            raise HarnessError("cache qualification changed fixture bytes or stat identity")

        # Hashing re-faults the fixture; the final invalidation must restore zero
        # residency before the first model arm.
        address, length = vm.map_file(path)
        try:
            vm.invalidate(address, length)
            final_resident, _ = vm.resident_pages(address, length)
            if final_resident != 0:
                raise CacheStateError("final fixture invalidation did not return mincore to zero")
        finally:
            vm.unmap(address, length)
        if fixture_stat(path) != original_stat:
            raise HarnessError("final fixture invalidation changed stat identity")
        return {"size_bytes": FIXTURE_SIZE, "allocated_by_full_write": True,
                "warm_pages": warm, "page_count": pages,
                "fresh_private_warm_pages": fresh_warm_results["private"],
                "fresh_shared_warm_pages": fresh_warm_results["shared"],
                "same_mapping_after_invalidate_pages": same_map,
                "fresh_private_pages": fresh_results["private"],
                "fresh_shared_pages": fresh_results["shared"],
                "sha256_before_after_equal": original_hash == final_hash,
                "stat_identity_before_after_equal": original_stat == final_stat,
                "final_resident_pages": final_resident,
                "device": original_stat["device"]}


def parse_csv_ints(value: str, allowed: set[int], label: str) -> list[int]:
    try:
        items = [int(item.strip()) for item in value.split(",") if item.strip()]
    except ValueError as ex:
        raise argparse.ArgumentTypeError(f"{label} must be a comma-separated integer list") from ex
    if not items or len(items) != len(set(items)) or any(item not in allowed for item in items):
        raise argparse.ArgumentTypeError(f"{label} values must be unique members of {sorted(allowed)}")
    return items


def evaluate_gate(rows: list[dict[str, Any]], cache_states: list[str], depths: list[int],
                  requested_pairs: int) -> dict[str, Any]:
    cells: dict[str, Any] = {}
    full_matrix = set(cache_states) == {"nonresident", "resident"} and set(depths) == {0, 2}
    all_ok = (full_matrix and requested_pairs >= 4
              and not any(not row.get("passed", False) for row in rows))
    for cache_state in cache_states:
        for depth in depths:
            cell_rows = [row for row in rows if row.get("cache_state") == cache_state
                         and row.get("depth") == depth and row.get("passed")]
            by_pair: dict[int, dict[str, dict[str, Any]]] = {}
            for row in cell_rows:
                by_pair.setdefault(int(row["pair"]), {})[row["mapping"]] = row
            ratios = []
            valid_pairs = 0
            for pair, arms in sorted(by_pair.items()):
                if set(arms) != {"private", "shared"}:
                    continue
                private_seconds = float(arms["private"]["process_seconds"])
                shared_seconds = float(arms["shared"]["process_seconds"])
                if not all(math.isfinite(value) and value > 0
                           for value in (private_seconds, shared_seconds)):
                    continue
                ratios.append({"pair": pair, "ratio": shared_seconds / private_seconds})
                valid_pairs += 1
            median = statistics.median(item["ratio"] for item in ratios) if ratios else None
            max_ratio = max((item["ratio"] for item in ratios), default=None)
            cell_ok = (valid_pairs == requested_pairs and requested_pairs >= 4
                       and median is not None and median <= 0.95
                       and max_ratio is not None and max_ratio <= 1.05)
            cells[f"{cache_state}/depth{depth}"] = {
                "complete_pairs": valid_pairs, "ratios_shared_over_private": ratios,
                "median_ratio": median, "max_pair_ratio": max_ratio,
                "meets_cell_gate": cell_ok,
            }
            all_ok = all_ok and cell_ok
    return {"predeclared_gate": {"minimum_pairs_per_cell": 4,
                                 "median_process_ratio_max": 0.95,
                                 "maximum_pair_ratio": 1.05,
                                 "process_ratio": "shared / private",
                                 "required_cache_states": ["nonresident", "resident"],
                                 "required_depths": [0, 2],
                                 "minimum_free_percent_during_cache_prepare": RESERVE_STOP,
                                 "minimum_free_percent_before_target_prepare": NONRESIDENT_LAUNCH_FREE_MIN,
                                 "minimum_free_percent_at_launch_nonresident": NONRESIDENT_LAUNCH_FREE_MIN,
                                 "minimum_free_percent_at_launch_resident": RESIDENT_LAUNCH_FREE_MIN,
                                 "stop_at_or_below_free_percent": RESERVE_STOP},
            "cells": cells, "promoted": all_ok}


def validate_report(report: dict[str, Any], baseline: dict[str, Any],
                    expected_depth: int) -> list[str]:
    errors: list[str] = []
    for key in ("text", "seed", "prompt_ids", "frames", "stopped_on_eos", "sample_rate", "samples"):
        if report.get(key) != baseline.get(key):
            errors.append(f"report field {key} differs from baseline")
    if report.get("backend") != "metal_resident":
        errors.append("backend is not metal_resident")
    if report.get("stopped_on_eos") is not True:
        errors.append("generation did not stop on EOS")
    if report.get("resident_readbacks") != baseline.get("resident_readbacks"):
        errors.append("resident readback ledger differs from baseline")
    graph = report.get("resident_graph")
    if not isinstance(graph, dict):
        errors.append("resident graph report is missing")
    else:
        submitted = graph.get("submitted")
        completed = graph.get("completed")
        waits = graph.get("completion_waits")
        max_pending = graph.get("max_pending")
        if graph.get("depth") != expected_depth:
            errors.append(f"resident graph depth is {graph.get('depth')}, expected {expected_depth}")
        if graph.get("pending") != 0 or graph.get("retained_bytes") != 0:
            errors.append("resident graph did not drain pending work and retained bytes")
        if submitted is None or submitted != completed:
            errors.append("resident graph submitted/completed counts differ")
        if submitted is None or waits != submitted:
            errors.append("resident graph completion waits do not match submitted count")
        if max_pending is None or not isinstance(max_pending, int) or max_pending > expected_depth:
            errors.append("resident graph max_pending exceeds the configured depth")
    return errors


def parse_time_real(stderr_text: str) -> float:
    matches = re.findall(r"^\s*([0-9]+(?:\.[0-9]+)?)\s+real\b", stderr_text, re.MULTILINE)
    if len(matches) != 1:
        raise HarnessError(f"expected one /usr/bin/time real duration, found {len(matches)}")
    return float(matches[0])


def atomic_json(path: Path, value: Any) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.{time.time_ns()}.tmp")
    with temporary.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def sha256_paths(paths: list[Path]) -> dict[str, str]:
    return {str(path.relative_to(ROOT)): sha256_file(path) for path in paths}


def source_files() -> list[Path]:
    files = [GENERATOR, RUN_SAFE, Path(__file__).resolve()]
    for root in (SOURCE_ROOT, METAL_SOURCE_ROOT):
        files.extend(sorted(p for p in root.rglob("*") if p.is_file()
                            and p.suffix in (".cr", ".metal", ".mm")))
    return sorted(set(files))


def binary_identity(path: Path) -> dict[str, Any]:
    st = checked_file_stat(path)
    if not os.access(path, os.X_OK):
        raise HarnessError("--binary must be executable")
    return {"stat": stat_identity(st), "sha256": sha256_file(path)}


def assert_arm_inputs_stable(args: argparse.Namespace,
                             progress_doc: dict[str, Any]) -> dict[str, Any]:
    current_binary = binary_identity(args.binary)
    expected_binary = progress_doc["binary"]
    if current_binary != {"stat": expected_binary["stat"], "sha256": expected_binary["sha256"]}:
        raise HarnessError("benchmark binary identity/hash changed during matrix")
    current_sources = sha256_paths(source_files())
    if current_sources != progress_doc["source_sha256_before"]:
        raise HarnessError("source hashes changed during benchmark matrix")
    return {"binary_sha256": current_binary["sha256"], "source_sha256": current_sources}


def git_metadata() -> dict[str, Any]:
    def git(*args: str) -> str:
        result = subprocess.run(["git", *args], cwd=ROOT, text=True,
                                capture_output=True, timeout=10)
        if result.returncode != 0:
            return "unavailable"
        return result.stdout.strip()
    return {"head": git("rev-parse", "HEAD"),
            "worktree_dirty": bool(git("status", "--porcelain"))}


def gpu_utilization_sample() -> dict[str, Any] | None:
    try:
        result = subprocess.run(["/usr/sbin/ioreg", "-r", "-c", "IOAccelerator", "-d", "1"],
                                text=True, capture_output=True, timeout=5)
    except (OSError, subprocess.TimeoutExpired):
        return None
    if result.returncode != 0:
        return None
    values: dict[str, float] = {}
    for key, number in re.findall(r'"(GPU Core Utilization|GPU Utilization|Device Utilization)"\s*=\s*([0-9.]+)', result.stdout):
        values[key] = float(number)
    return values or None


def host_cpu_aggregate() -> float | None:
    try:
        result = subprocess.run(["/bin/ps", "-A", "-o", "%cpu="], text=True,
                                capture_output=True, timeout=5)
    except (OSError, subprocess.TimeoutExpired):
        return None
    if result.returncode != 0:
        return None
    try:
        return round(sum(float(line.strip()) for line in result.stdout.splitlines()
                         if line.strip()), 2)
    except ValueError:
        return None


def load_baseline(wav_path: Path) -> tuple[dict[str, Any], bytes]:
    if not wav_path.is_file() or wav_path.is_symlink():
        raise HarnessError("--baseline-wav must be an existing regular non-symlink WAV")
    sidecar = Path(str(wav_path) + ".json")
    try:
        report = json.loads(sidecar.read_text(encoding="utf-8"))
        wav_bytes = wav_path.read_bytes()
    except (OSError, json.JSONDecodeError) as ex:
        raise HarnessError(f"cannot read baseline WAV/report: {ex}") from ex
    if report.get("text") != PROMPT or report.get("seed") != SEED:
        raise HarnessError("baseline sidecar must describe prompt Hello. with seed 0")
    if report.get("backend") != "metal_resident" or report.get("stopped_on_eos") is not True:
        raise HarnessError("baseline must be a completed resident Metal run that reached EOS")
    if not isinstance(report.get("prompt_ids"), list) or not isinstance(report.get("frames"), list):
        raise HarnessError("baseline sidecar lacks prompt_ids or frames")
    if not isinstance(report.get("resident_readbacks"), dict):
        raise HarnessError("baseline sidecar lacks resident_readbacks")
    return report, wav_bytes


def run_arm(args: argparse.Namespace, vm: DarwinVM, model_paths: list[Path],
            model_identity: dict[str, dict[str, int]], baseline: dict[str, Any],
            baseline_wav: bytes, out_dir: Path, cache_state: str, depth: int,
            pair: int, mapping: str, progress_doc: dict[str, Any],
            progress_path: Path) -> dict[str, Any]:
    stem = f"{cache_state}-depth{depth}-pair{pair:02d}-{mapping}"
    wav_path = out_dir / f"{stem}.wav"
    report_path = Path(str(wav_path) + ".json")
    stdout_path = out_dir / f"{stem}.stdout.log"
    stderr_path = out_dir / f"{stem}.stderr.log"
    for path in (wav_path, report_path, stdout_path, stderr_path):
        if path.exists():
            raise HarnessError(f"refusing to overwrite output: {path}")

    row: dict[str, Any] = {"cache_state": cache_state, "depth": depth, "pair": pair,
                           "mapping": mapping, "status": "preparing"}
    progress_doc["current_arm"] = row
    atomic_json(progress_path, progress_doc)
    series: list[dict[str, Any]] = []
    clear_started = time.perf_counter()
    prep_started = clear_started
    try:
        row["artifact_identity_before_arm"] = assert_arm_inputs_stable(args, progress_doc)
        row["free_percent_before_normalization"] = check_prep_reserve(series, "before_cache_normalization")
        row.update(normalize_nonresident(vm, model_paths, series))
        row["cache_normalization_seconds"] = time.perf_counter() - clear_started
        row["free_percent_after_normalization"] = check_prep_reserve(series, "after_cache_normalization")
        row["free_percent_before_prepare"] = wait_for_headroom(
            NONRESIDENT_LAUNCH_FREE_MIN, strict=False,
            phase="before_target_prepare_headroom_wait", series=series)
        prep_started = time.perf_counter()
        prepare_cache(vm, model_paths, cache_state, series, row)
        row["free_percent_after_prepare"] = check_prep_reserve(series, "after_cache_prepare")
        row["preparation_seconds"] = time.perf_counter() - prep_started
        launch_minimum = (RESIDENT_LAUNCH_FREE_MIN if cache_state == "resident"
                          else NONRESIDENT_LAUNCH_FREE_MIN)
        row["free_percent_at_launch"] = wait_for_headroom(
            launch_minimum, strict=False, phase="launch_headroom_wait", series=series)
        require_no_open_model_files(model_paths)
        assert_model_unchanged(model_paths, model_identity)
        row["gpu_utilization_before"] = gpu_utilization_sample()
        row["host_cpu_aggregate_before"] = host_cpu_aggregate()
        immediately_before = query_cache(vm, model_paths)
        row["cache_immediately_before_launch"] = immediately_before
        require_cache_state(immediately_before, cache_state, "immediately before launch")
        row["free_percent_series"] = series
        row["status"] = "running"
        progress_doc["current_arm"] = row
        atomic_json(progress_path, progress_doc)

        environment = os.environ.copy()
        for key in ("DRAMA3_PROFILE", "DRAMA3_METAL_TIMELINE", "COGNI_METAL_SUBMIT_PROFILE",
                    "COGNI_METAL_PIPELINE_PROFILE", "COGNI_METAL_PIPELINE_TRACE_PREFIX"):
            environment.pop(key, None)
        environment.update({
            "DRAMA3_BACKEND": "metal",
            "DRAMA3_MMAP_SHARED": "1" if mapping == "shared" else "0",
            "DRAMA3_COGNIGRAPH_MAX_INFLIGHT": str(depth),
            "DRAMA3_MOE_BATCH": "1",
            "DRAMA3_DELTA_TRANSPOSE": "1",
            "COGNI_RUN_SAFE_MIN_FREE_PCT": str(RESERVE_STOP),
            "COGNI_RUN_SAFE_WAIT_QUIET_SEC": "0",
            "COGNI_RUN_SAFE_REQUIRE_QUIET": "0",
            "RUN_SAFE_PASSTHROUGH_STDIO": "1",
        })
        command = [str(RUN_SAFE), "/usr/bin/time", str(RUN_TIMEOUT_SECONDS),
                   str(RUN_MAX_MEMORY_MIB), "-l", str(args.binary),
                   "--model", str(args.model.resolve()), "--text", PROMPT,
                   "--output", str(wav_path), "--max-frames", str(MAX_FRAMES),
                   "--seed", str(SEED)]
        started_wall = time.perf_counter()
        with stdout_path.open("xb") as stdout_file, stderr_path.open("xb") as stderr_file:
            launch_free = free_percent()
            series.append({"phase": "immediate_launch_sample", "free_percent": launch_free})
            row["free_percent_at_launch"] = launch_free
            if launch_free < launch_minimum:
                raise HeadroomBlocked(f"immediate launch sample {launch_free}% < {launch_minimum}%")
            process = subprocess.Popen(command, cwd=ROOT, env=environment,
                                       stdin=subprocess.DEVNULL, stdout=stdout_file,
                                       stderr=stderr_file, start_new_session=True)
            samples: list[dict[str, Any]] = []
            try:
                while process.poll() is None:
                    value = free_percent()
                    elapsed = time.perf_counter() - started_wall
                    samples.append({"elapsed_seconds": round(elapsed, 3), "free_percent": value})
                    series.append({"phase": "process_monitor", "elapsed_seconds": round(elapsed, 3),
                                   "free_percent": value})
                    if value <= RESERVE_STOP:
                        row["status"] = "aborted_memory_reserve"
                        row["reserve_violation"] = True
                        row["memory_pressure_process_samples"] = samples
                        row["free_percent_series"] = series
                        progress_doc["current_arm"] = row
                        atomic_json(progress_path, progress_doc)
                        terminate_own_wrapper(process)
                        raise HeadroomBlocked(f"free memory reached {value}% during arm; stopped own run_safe wrapper")
                    time.sleep(0.5)
                exit_code = process.wait()
            except BaseException:
                if process.poll() is None:
                    terminate_own_wrapper(process)
                raise
        row["outer_monitor_wall_seconds"] = time.perf_counter() - started_wall
        final_free = free_percent()
        samples.append({"elapsed_seconds": round(time.perf_counter() - started_wall, 3),
                        "free_percent": final_free, "phase": "post_process_final_sample"})
        row["memory_pressure_process_samples"] = samples
        series.append({"phase": "post_process_final_sample", "free_percent": final_free})
        row["post_process_free_percent"] = final_free
        if final_free <= RESERVE_STOP:
            row["status"] = "aborted_memory_reserve"
            row["reserve_violation"] = True
            raise HeadroomBlocked(f"post-process free memory was {final_free}%; invalidating arm")
        stderr_text = stderr_path.read_text(encoding="utf-8", errors="replace")
        row["process_seconds"] = parse_time_real(stderr_text)
        row["preparation_plus_process_seconds"] = (row["cache_normalization_seconds"]
                                                    + row["preparation_seconds"]
                                                    + row["process_seconds"])
        row["free_percent_series"] = series
        row["exit_code"] = exit_code
        row["gpu_utilization_after"] = gpu_utilization_sample()
        row["host_cpu_aggregate_after"] = host_cpu_aggregate()
        row["cache_after_process"] = query_cache(vm, model_paths)
        assert_model_unchanged(model_paths, model_identity)
        row["artifact_identity_after_arm"] = assert_arm_inputs_stable(args, progress_doc)
        if exit_code != 0:
            raise HarnessError(f"run_safe arm exited with status {exit_code}")
        if not wav_path.is_file() or not report_path.is_file():
            raise HarnessError("generator did not create WAV and JSON sidecar")
        generated_wav = wav_path.read_bytes()
        report = json.loads(report_path.read_text(encoding="utf-8"))
        parity_errors = validate_report(report, baseline, depth)
        if generated_wav != baseline_wav:
            parity_errors.append("WAV bytes differ from baseline")
        if parity_errors:
            raise HarnessError("parity/ownership validation failed: " + "; ".join(parity_errors))
        row["report_sha256"] = sha256_file(report_path)
        row["wav_sha256"] = hashlib.sha256(generated_wav).hexdigest()
        row["report_summary"] = {key: report.get(key) for key in
                                  ("backend", "prompt_rows", "frames", "stopped_on_eos",
                                   "resident_readbacks", "resident_submissions", "resident_graph")}
        row["passed"] = True
        row["status"] = "completed"
    except BaseException as ex:
        row["passed"] = False
        row["status"] = row.get("status") if row.get("status") == "aborted_memory_reserve" else "aborted"
        row["error"] = f"{type(ex).__name__}: {ex}"
        if not row.get("preparation_seconds"):
            row["preparation_seconds"] = time.perf_counter() - prep_started
        if row.get("process_seconds") is not None:
            row["preparation_plus_process_seconds"] = row["preparation_seconds"] + row["process_seconds"]
        row["free_percent_series"] = series
        progress_doc["rows"].append(row)
        progress_doc["current_arm"] = None
        progress_doc["last_error"] = row["error"]
        atomic_json(progress_path, progress_doc)
        raise
    progress_doc["rows"].append(row)
    progress_doc["current_arm"] = None
    progress_doc["last_error"] = None
    atomic_json(progress_path, progress_doc)
    return row


def terminate_own_wrapper(process: subprocess.Popen[bytes]) -> None:
    if process.poll() is not None:
        return
    # Signal run_safe itself; its TERM trap owns process-group cleanup.
    process.terminate()
    try:
        process.wait(timeout=30)
    except subprocess.TimeoutExpired:
        # If the wrapper is wedged, its watchdog kills the isolated workload
        # group when the wrapper exits.
        process.kill()
        process.wait(timeout=10)


def self_test(model_dir: Path) -> None:
    if platform.system() != "Darwin":
        raise HarnessError("--self-test requires Darwin libSystem mincore/msync")
    if model_dir.is_symlink() or not model_dir.is_dir():
        raise HarnessError("--self-test --model must name the existing model directory")
    vm = DarwinVM()
    qualification = qualify_fixture(vm, model_dir.resolve().parent)
    if qualification["device"] != os.stat(model_dir.resolve().parent).st_dev:
        raise HarnessError("cache fixture was not created on the model volume")
    if parse_free_percent("System-wide memory free percentage: 82%\n") != 82:
        raise HarnessError("memory-pressure parser qualification failed")
    fake_rows: list[dict[str, Any]] = []
    for state in ("nonresident", "resident"):
        for depth in (0, 2):
            for pair in range(4):
                private = 20.0 + pair / 10
                shared = private * 0.94
                for mapping, seconds in (("private", private), ("shared", shared)):
                    fake_rows.append({"cache_state": state, "depth": depth, "pair": pair,
                                      "mapping": mapping, "process_seconds": seconds,
                                      "passed": True})
    gate = evaluate_gate(fake_rows, ["nonresident", "resident"], [0, 2], 4)
    if not gate["promoted"]:
        raise HarnessError("promotion-gate positive control failed")
    bad_rows = [dict(row) for row in fake_rows]
    for row in bad_rows:
        if row["cache_state"] == "resident" and row["depth"] == 2 and row["mapping"] == "shared":
            row["process_seconds"] = 22.0
    if evaluate_gate(bad_rows, ["nonresident", "resident"], [0, 2], 4)["promoted"]:
        raise HarnessError("promotion-gate regression negative control failed")
    baseline = {"text": PROMPT, "seed": SEED, "prompt_ids": [1, 2], "frames": [[3]],
                "stopped_on_eos": True, "sample_rate": 44100, "samples": 4,
                "backend": "metal_resident", "resident_readbacks": {"control": 1, "terminal": 4}}
    candidate = dict(baseline)
    candidate["resident_graph"] = {"depth": 2, "submitted": 3, "completed": 3,
                                    "completion_waits": 3, "max_pending": 2,
                                    "pending": 0, "retained_bytes": 0}
    if validate_report(candidate, baseline, 2):
        raise HarnessError("report validator positive control failed")
    candidate["resident_graph"] = {"depth": 2, "submitted": 3, "completed": 2,
                                    "completion_waits": 2, "max_pending": 2,
                                    "pending": 1, "retained_bytes": 0}
    if not validate_report(candidate, baseline, 2):
        raise HarnessError("report validator pending-work negative control failed")
    if parse_time_real("run_safe log\n  12.34 real  8.00 user  1.00 sys\n") != 12.34:
        raise HarnessError("/usr/bin/time real-duration parser qualification failed")
    if evaluate_gate(fake_rows, ["resident"], [2], 4)["promoted"]:
        raise HarnessError("partial-matrix promotion negative control failed")
    if evaluate_gate(fake_rows[:-1], ["nonresident", "resident"], [0, 2], 4)["promoted"]:
        raise HarnessError("missing-arm promotion negative control failed")
    one_bad_pair = [dict(row) for row in fake_rows]
    for row in one_bad_pair:
        if (row["cache_state"] == "resident" and row["depth"] == 2
                and row["pair"] == 0 and row["mapping"] == "shared"):
            row["process_seconds"] = 21.2
    if evaluate_gate(one_bad_pair, ["nonresident", "resident"], [0, 2], 4)["promoted"]:
        raise HarnessError("single-bad-pair promotion negative control failed")
    print(json.dumps({"self_test": "passed", "scope": "fixture and pure checks only",
                      "qualification": qualification,
                      "promotion_gate_positive_control": True,
                      "promotion_gate_regression_negative_control": True,
                      "promotion_gate_partial_missing_single_bad_pair_controls": True,
                      "time_parser_control": True,
                      "report_validator_positive_and_negative_controls": True}, sort_keys=True))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, help="model directory (weights are not read by --self-test)")
    parser.add_argument("--binary", type=Path, help="same final native Metal generator binary for all arms")
    parser.add_argument("--baseline-wav", type=Path, help="baseline WAV with adjacent .wav.json report")
    parser.add_argument("--output-dir", type=Path, help="new output directory; existing paths are rejected")
    parser.add_argument("--pairs", type=int, default=4)
    parser.add_argument("--depths", default="0,2")
    parser.add_argument("--cache-states", default="nonresident,resident")
    parser.add_argument("--prepare-only", action="store_true", help="qualify and prepare cache states without inference")
    parser.add_argument("--self-test", action="store_true", help="fixture and pure checks only; no model reads or inference")
    args = parser.parse_args()
    try:
        args.depth_values = parse_csv_ints(args.depths, {0, 2}, "--depths")
        states = [item.strip() for item in args.cache_states.split(",") if item.strip()]
        if not states or len(states) != len(set(states)) or any(item not in ("nonresident", "resident") for item in states):
            raise argparse.ArgumentTypeError("--cache-states must be unique values from nonresident,resident")
        args.cache_values = states
    except argparse.ArgumentTypeError as ex:
        parser.error(str(ex))
    if args.pairs <= 0:
        parser.error("--pairs must be positive")
    if args.self_test:
        if args.model is None:
            parser.error("--self-test requires --model to place its fixture on the model volume")
        if args.binary or args.baseline_wav or args.output_dir or args.prepare_only:
            parser.error("--self-test cannot be combined with benchmark or prepare-only arguments")
        return args
    if args.model is None or args.output_dir is None:
        parser.error("--model and --output-dir are required")
    if not args.prepare_only and (args.binary is None or args.baseline_wav is None):
        parser.error("--binary and --baseline-wav are required unless --prepare-only is set")
    return args


def main() -> int:
    args = parse_args()
    if args.self_test:
        self_test(args.model.resolve(strict=True))
        return 0
    if platform.system() != "Darwin":
        raise HarnessError("controlled cache preparation is Darwin only")
    if args.model.is_symlink():
        raise HarnessError("--model must not be a symlink")
    args.model = args.model.resolve(strict=True)
    if args.binary is not None:
        args.binary = args.binary.resolve(strict=True)
    paths = model_files(args.model)
    vm = DarwinVM()

    args.output_dir = args.output_dir.expanduser().absolute()
    try:
        args.output_dir.mkdir(parents=True, exist_ok=False)
    except FileExistsError as ex:
        raise HarnessError(f"--output-dir already exists; refusing overwrite: {args.output_dir}") from ex
    progress_path = args.output_dir / "progress.json"
    progress_doc: dict[str, Any] = {
        "schema": "drama3-mapping-benchmark-v1", "created_utc": datetime.now(timezone.utc).isoformat(),
        "command": [sys.executable, *sys.argv],
        "status": "initializing", "scope": "Darwin VM file-cache state; not SSD/controller cache",
        "configuration": {"pairs": args.pairs, "depths": args.depth_values,
                           "cache_states": args.cache_values, "prepare_only": args.prepare_only,
                           "prompt": PROMPT, "seed": SEED, "max_frames": MAX_FRAMES,
                           "binary": str(args.binary.resolve()) if args.binary else None,
                           "model": str(args.model), "timeout_seconds": RUN_TIMEOUT_SECONDS,
                           "run_safe_max_memory_mib": RUN_MAX_MEMORY_MIB,
                           "reserve_stop_percent": RESERVE_STOP,
                           "nonresident_launch_free_min_percent": NONRESIDENT_LAUNCH_FREE_MIN,
                           "resident_launch_free_min_percent": RESIDENT_LAUNCH_FREE_MIN,
                           "max_warm_missing_page_repair_passes": MAX_WARM_REPAIR_PASSES,
                           "profile_environment": "unset",
                           "fixed_environment": {"DRAMA3_MOE_BATCH": "1", "DRAMA3_DELTA_TRANSPOSE": "1"}},
        "git": git_metadata(), "binary": None, "source_sha256_before": None,
        "model_stat_identity": snapshot_model(paths), "cache_qualification": None,
        "rows": [], "current_arm": None, "last_error": None,
    }
    atomic_json(progress_path, progress_doc)
    try:
        progress_doc["status"] = "qualifying_cache_control"
        atomic_json(progress_path, progress_doc)
        require_no_open_model_files(paths)
        progress_doc["cache_qualification"] = qualify_fixture(vm, args.model.parent)
        if args.binary:
            identity = binary_identity(args.binary)
            progress_doc["binary"] = {"path": str(args.binary), **identity}
        srcs = source_files()
        progress_doc["source_sha256_before"] = sha256_paths(srcs)
        baseline: dict[str, Any] = {}
        baseline_wav = b""
        if not args.prepare_only:
            baseline, baseline_wav = load_baseline(args.baseline_wav.resolve(strict=True))
            progress_doc["baseline"] = {
                "wav_path": str(args.baseline_wav.resolve()),
                "wav_sha256": hashlib.sha256(baseline_wav).hexdigest(),
                "report_path": str(Path(str(args.baseline_wav.resolve()) + ".json")),
                "report_sha256": sha256_file(Path(str(args.baseline_wav.resolve()) + ".json")),
                "prompt_ids": baseline["prompt_ids"], "frames": baseline["frames"],
                "stopped_on_eos": baseline["stopped_on_eos"],
                "resident_readbacks": baseline["resident_readbacks"],
            }
        progress_doc["status"] = "preparing_only" if args.prepare_only else "running"
        atomic_json(progress_path, progress_doc)

        if args.prepare_only:
            for state in args.cache_values:
                series: list[dict[str, Any]] = []
                row = {"cache_state": state, "status": "preparing"}
                progress_doc["current_arm"] = row
                atomic_json(progress_path, progress_doc)
                row["free_percent_before_normalization"] = check_prep_reserve(
                    series, "before_cache_normalization")
                row.update(normalize_nonresident(vm, paths, series))
                row["free_percent_after_normalization"] = check_prep_reserve(
                    series, "after_cache_normalization")
                row["free_percent_before_prepare"] = wait_for_headroom(
                    NONRESIDENT_LAUNCH_FREE_MIN, strict=False,
                    phase="before_target_prepare_headroom_wait", series=series)
                started = time.perf_counter()
                prepare_cache(vm, paths, state, series, row)
                row["preparation_seconds"] = time.perf_counter() - started
                row["free_percent_after_prepare"] = check_prep_reserve(series, "after_prepare_only")
                launch_minimum = (RESIDENT_LAUNCH_FREE_MIN if state == "resident"
                                  else NONRESIDENT_LAUNCH_FREE_MIN)
                row["free_percent_at_launch"] = wait_for_headroom(
                    launch_minimum, strict=False, phase="prepare_only_launch_headroom_wait",
                    series=series)
                row["cache_immediately_before_launch"] = query_cache(vm, paths)
                require_cache_state(row["cache_immediately_before_launch"], state,
                                    "after prepare-only headroom wait")
                row["free_percent_series"] = series
                row["passed"] = True
                row["status"] = "prepared"
                progress_doc["rows"].append(row)
                progress_doc["current_arm"] = None
                atomic_json(progress_path, progress_doc)
        else:
            cells = [(state, depth) for state in args.cache_values for depth in args.depth_values]
            # Pair-major, cell-round-robin schedule; every pair is alternating AB/BA.
            for pair in range(args.pairs):
                for cell_index in range(len(cells)):
                    state, depth = cells[(cell_index + pair) % len(cells)]
                    order = ("private", "shared") if pair % 2 == 0 else ("shared", "private")
                    for mapping in order:
                        try:
                            run_arm(args, vm, paths, progress_doc["model_stat_identity"],
                                    baseline, baseline_wav, args.output_dir, state, depth,
                                    pair, mapping, progress_doc, progress_path)
                        except BaseException as ex:
                            progress_doc["status"] = "blocked" if isinstance(ex, (HeadroomBlocked, CacheStateError)) else "failed"
                            progress_doc["last_error"] = f"{type(ex).__name__}: {ex}"
                            break
                    if progress_doc["status"] in ("blocked", "failed"):
                        break
                if progress_doc["status"] in ("blocked", "failed"):
                    break
        assert_model_unchanged(paths, progress_doc["model_stat_identity"])
        progress_doc["source_sha256_after"] = sha256_paths(srcs)
        if progress_doc["source_sha256_after"] != progress_doc["source_sha256_before"]:
            raise HarnessError("source hashes changed during benchmark")
        if args.binary:
            final_binary_identity = binary_identity(args.binary)
            progress_doc["binary_identity_after"] = final_binary_identity
            if (final_binary_identity["sha256"] != progress_doc["binary"]["sha256"]
                    or final_binary_identity["stat"] != progress_doc["binary"]["stat"]):
                raise HarnessError("benchmark binary identity/hash changed during matrix")
        progress_doc["gpu_utilization_final"] = gpu_utilization_sample()
        progress_doc["host_cpu_aggregate_final"] = host_cpu_aggregate()
        if not args.prepare_only:
            progress_doc["gate"] = evaluate_gate(progress_doc["rows"], args.cache_values,
                                                   args.depth_values, args.pairs)
            if progress_doc["status"] == "running":
                progress_doc["status"] = "complete" if all(row.get("passed") for row in progress_doc["rows"]) else "incomplete"
            progress_doc["promoted"] = progress_doc["status"] == "complete" and progress_doc["gate"]["promoted"]
        elif progress_doc["status"] == "preparing_only":
            progress_doc["status"] = "prepared"
        atomic_json(progress_path, progress_doc)
        return 0 if progress_doc["status"] in ("complete", "prepared") else 2
    except BaseException as ex:
        progress_doc["status"] = "blocked" if isinstance(ex, (HeadroomBlocked, CacheStateError)) else "failed"
        progress_doc["last_error"] = f"{type(ex).__name__}: {ex}"
        atomic_json(progress_path, progress_doc)
        raise


if __name__ == "__main__":
    try:
        sys.exit(main())
    except HarnessError as error:
        print(f"[DRAMA3-MAPPING] {error}", file=sys.stderr)
        sys.exit(2)
    except KeyboardInterrupt:
        print("[DRAMA3-MAPPING] interrupted; current arm is preserved in progress.json", file=sys.stderr)
        sys.exit(130)
