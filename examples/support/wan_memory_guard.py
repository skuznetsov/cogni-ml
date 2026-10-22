#!/usr/bin/env python3
"""Sampled macOS memory guard for bounded Wan microbenchmarks.

This is a launcher for one explicitly supplied command. It does not reserve
memory: a spike between samples can exceed a limit, and a descendant that
deliberately calls setsid() or setpgid() can escape the process group. It never
signals a process outside the new session/process group it created.
Cleanup waits are bounded. Lost process identity or an uninterruptible child is
reported as incomplete cleanup, never as successful containment.
"""

from __future__ import annotations

import argparse
import ctypes
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
import errno
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import time
from typing import Callable, Iterable

MIB = 1024 * 1024
RUSAGE_INFO_V2 = 2
RUNTIME_FLOOR_PCT = 15
FREE_SPECULATIVE = "free-speculative"
FILE_CACHE = "file-cache"
MEMORY_ACCOUNTING_POLICIES = (FREE_SPECULATIVE, FILE_CACHE)
EXIT_USAGE = 64
EXIT_ADMISSION = 75
EXIT_GUARD = 76
EXIT_TELEMETRY = 77
EXIT_TIMEOUT = 124


class TelemetryError(RuntimeError):
    """Host or process telemetry is absent, malformed, or inconsistent."""


@dataclass(frozen=True)
class Telemetry:
    pressure_free_pct: int
    free_spec_bytes: int
    physical_bytes: int
    swap_used_bytes: int
    group_footprint_bytes: int
    group_live_processes: int = 1
    speculative_bytes: int | None = None
    file_backed_bytes: int | None = None


@dataclass(frozen=True)
class VMPageCounts:
    free_bytes: int
    speculative_bytes: int
    file_backed_bytes: int | None

    @property
    def free_spec_bytes(self) -> int:
        return self.free_bytes + self.speculative_bytes

    @property
    def free_file_cache_bytes(self) -> int:
        # XNU includes speculative pages in its pageable external counter.
        if self.file_backed_bytes is None:
            raise TelemetryError("file-cache counters missing")
        return self.free_bytes + self.file_backed_bytes


@dataclass(frozen=True)
class GroupProcess:
    pid: int
    state: str


def _single_int(pattern: str, text: str, label: str) -> int:
    matches = re.findall(pattern, text, re.MULTILINE)
    if len(matches) != 1:
        raise TelemetryError(f"{label} missing, malformed, or duplicated")
    try:
        return int(matches[0])
    except (TypeError, ValueError) as error:
        raise TelemetryError(f"{label} is not an integer") from error


def parse_memory_pressure(text: str) -> int:
    """Parse the single free percentage emitted by memory_pressure -Q."""
    value = _single_int(
        r"^System-wide memory free percentage:\s*(\d+)%\s*$", text,
        "memory_pressure free percentage")
    if not 0 <= value <= 100:
        raise TelemetryError("memory_pressure percentage is outside 0..100")
    return value


def parse_vm_stat(text: str, *, require_file_cache: bool = False) -> VMPageCounts:
    """Parse printed free/speculative/file-backed counts without overlap.

    vm_stat prints free_count - speculative_count as "Pages free". File-backed
    pages already include speculative pages. The file-cache estimate does not
    establish that all file-backed pages are clean or immediately reclaimable.
    Cache telemetry is optional for the unchanged free-speculative policy.
    """
    page_size = _single_int(r"page size of\s*(\d+)\s+bytes", text, "vm_stat page size")
    free_pages = _single_int(r"^Pages free:\s*(\d+)\.?\s*$", text, "vm_stat free pages")
    speculative_pages = _single_int(
        r"^Pages speculative:\s*(\d+)\.?\s*$", text,
        "vm_stat speculative pages")
    if page_size <= 0 or min(free_pages, speculative_pages) < 0:
        raise TelemetryError("vm_stat contains a negative or zero capacity")
    try:
        file_backed_pages = _single_int(
            r"^File-backed pages:\s*(\d+)\.?\s*$", text, "vm_stat file-backed pages")
        if speculative_pages > file_backed_pages:
            raise TelemetryError("vm_stat speculative pages exceed file-backed pages")
        file_backed_bytes = file_backed_pages * page_size
    except TelemetryError:
        if require_file_cache:
            raise
        file_backed_bytes = None
    return VMPageCounts(free_pages * page_size, speculative_pages * page_size,
                        file_backed_bytes)


def parse_swap_used(text: str) -> int:
    """Parse sysctl vm.swapusage's used value into bytes."""
    matches = re.findall(
        r"\bused\s*=\s*(\d+(?:\.\d+)?)\s*([KMGT]?)(?:i?B)?(?=\s|$)",
        text, re.IGNORECASE)
    if len(matches) != 1:
        raise TelemetryError("vm.swapusage used value missing, malformed, or duplicated")
    amount, unit = matches[0]
    multiplier = {"": 1, "K": 1024, "M": 1024**2,
                  "G": 1024**3, "T": 1024**4}.get(unit.upper())
    if multiplier is None:
        raise TelemetryError("vm.swapusage used value has an unsupported unit")
    try:
        result = int(Decimal(amount) * multiplier)
    except (InvalidOperation, ValueError) as error:
        raise TelemetryError("vm.swapusage used value is invalid") from error
    if result < 0:
        raise TelemetryError("vm.swapusage used value is negative")
    return result


def _run_text(argv: list[str], *, timeout: float = 1.0,
              deadline: float | None = None,
              env: dict[str, str] | None = None) -> str:
    if deadline is not None:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise TelemetryError("runtime telemetry sample exceeded the monotonic deadline")
        timeout = min(timeout, remaining)
    try:
        result = subprocess.run(argv, check=True, capture_output=True, text=True,
                                timeout=timeout, env=env)
    except (OSError, subprocess.SubprocessError) as error:
        raise TelemetryError(f"telemetry command failed: {Path(argv[0]).name}") from error
    return result.stdout


def _physical_memory_bytes(deadline: float | None = None) -> int:
    value = _single_int(_single_int_pattern(),
                        _run_text(["/usr/sbin/sysctl", "-n", "hw.memsize"], deadline=deadline),
                        "physical memory")
    if value <= 0:
        raise TelemetryError("physical memory is not positive")
    return value


def _single_int_pattern() -> str:
    return r"^\s*(\d+)\s*$"


class _RUsageInfoV2(ctypes.Structure):
    # Exact field order from the active macOS SDK's sys/resource.h.
    _fields_ = [
        ("ri_uuid", ctypes.c_uint8 * 16),
        ("ri_user_time", ctypes.c_uint64),
        ("ri_system_time", ctypes.c_uint64),
        ("ri_pkg_idle_wkups", ctypes.c_uint64),
        ("ri_interrupt_wkups", ctypes.c_uint64),
        ("ri_pageins", ctypes.c_uint64),
        ("ri_wired_size", ctypes.c_uint64),
        ("ri_resident_size", ctypes.c_uint64),
        ("ri_phys_footprint", ctypes.c_uint64),
        ("ri_proc_start_abstime", ctypes.c_uint64),
        ("ri_proc_exit_abstime", ctypes.c_uint64),
        ("ri_child_user_time", ctypes.c_uint64),
        ("ri_child_system_time", ctypes.c_uint64),
        ("ri_child_pkg_idle_wkups", ctypes.c_uint64),
        ("ri_child_interrupt_wkups", ctypes.c_uint64),
        ("ri_child_pageins", ctypes.c_uint64),
        ("ri_child_elapsed_abstime", ctypes.c_uint64),
        ("ri_diskio_bytesread", ctypes.c_uint64),
        ("ri_diskio_byteswritten", ctypes.c_uint64),
    ]


class _DarwinSigval(ctypes.Union):
    _fields_ = [("sival_int", ctypes.c_int), ("sival_ptr", ctypes.c_void_p)]


class _DarwinSiginfo(ctypes.Structure):
    # Exact 64-bit siginfo_t layout from the active macOS SDK's sys/signal.h.
    _fields_ = [
        ("si_signo", ctypes.c_int),
        ("si_errno", ctypes.c_int),
        ("si_code", ctypes.c_int),
        ("si_pid", ctypes.c_int32),
        ("si_uid", ctypes.c_uint32),
        ("si_status", ctypes.c_int),
        ("si_addr", ctypes.c_void_p),
        ("si_value", _DarwinSigval),
        ("si_band", ctypes.c_long),
        ("__pad", ctypes.c_ulong * 7),
    ]


def _proc_pid_rusage(pid: int) -> int:
    if sys.platform != "darwin":
        raise TelemetryError("proc_pid_rusage is available only on macOS")
    try:
        libproc = ctypes.CDLL("/usr/lib/libproc.dylib", use_errno=True)
    except OSError as error:
        raise TelemetryError("cannot load libproc for process footprint") from error
    function = libproc.proc_pid_rusage
    function.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.POINTER(_RUsageInfoV2)]
    function.restype = ctypes.c_int
    usage = _RUsageInfoV2()
    if function(pid, RUSAGE_INFO_V2, ctypes.byref(usage)) != 0:
        number = ctypes.get_errno()
        raise OSError(number, os.strerror(number), pid)
    return int(usage.ri_phys_footprint)


def process_group_members(pgid: int, *, deadline: float | None = None) -> list[GroupProcess]:
    """Enumerate one process group; fail closed if ps cannot provide a full sample."""
    if not isinstance(pgid, int) or isinstance(pgid, bool) or pgid <= 1:
        raise TelemetryError("process-group identity is invalid")
    # ps(1): one -g selects this process group, including its unreaped leader.
    # compat(5): legacy mode ignores -g, so pin modern semantics for ps only.
    # Keep the existing one-second command cap and fail closed on every error.
    output = _run_text(["/bin/ps", "-g", str(pgid), "-o", "pid=,pgid=,stat="],
                       deadline=deadline, env={**os.environ, "COMMAND_MODE": "unix2003"})
    members: list[GroupProcess] = []
    seen: set[int] = set()
    try:
        for line in output.splitlines():
            fields = line.split(None, 2)
            if len(fields) != 3:
                raise TelemetryError("ps process list has a malformed row")
            pid, row_pgid = int(fields[0]), int(fields[1])
            state = fields[2].strip()
            if pid <= 0 or row_pgid <= 0 or not state:
                raise TelemetryError("ps process list has invalid process identity")
            if row_pgid != pgid or pid in seen:
                raise TelemetryError("ps group query returned a foreign or duplicate process")
            seen.add(pid)
            members.append(GroupProcess(pid, state))
    except ValueError as error:
        raise TelemetryError("ps process list has a nonnumeric pid or pgid") from error
    if not members:
        raise TelemetryError("ps group query returned no process rows")
    return members


def _group_usage(pgid: int, *, deadline: float | None = None) -> tuple[int, int]:
    members = process_group_members(pgid, deadline=deadline)
    if not members or not any(member.pid == pgid for member in members):
        raise TelemetryError("process-group leader missing from sample")
    total = 0
    live = 0
    for member in members:
        try:
            total += _proc_pid_rusage(member.pid)
        except OSError as error:
            # A process may exit between ps and proc_pid_rusage. Retry the group
            # membership check; skip only if that exact PID has left our group.
            if error.errno != errno.ESRCH:
                raise TelemetryError("proc_pid_rusage failed") from error
            if any(row.pid == member.pid for row in process_group_members(pgid, deadline=deadline)):
                raise TelemetryError("process footprint disappeared while still in group") from error
            continue
        if not member.state.startswith("Z"):
            live += 1
    return total, live


def collect_telemetry(pgid: int | None, deadline: float | None = None) -> Telemetry:
    """Collect one complete macOS sample; every source is required."""
    if sys.platform != "darwin":
        raise TelemetryError("Wan memory guard requires macOS")
    pressure = parse_memory_pressure(_run_text(["/usr/bin/memory_pressure", "-Q"],
                                               deadline=deadline))
    pages = parse_vm_stat(_run_text(["/usr/bin/vm_stat"], deadline=deadline))
    physical = _physical_memory_bytes(deadline)
    swap = parse_swap_used(_run_text(["/usr/sbin/sysctl", "-n", "vm.swapusage"],
                                     deadline=deadline))
    if pgid is None:
        footprint, live = 0, 0
    else:
        footprint, live = _group_usage(pgid, deadline=deadline)
    return Telemetry(pressure, pages.free_spec_bytes, physical, swap, footprint, live,
                     pages.speculative_bytes, pages.file_backed_bytes)


def _validate_sample(sample: Telemetry,
                     memory_accounting: str = FREE_SPECULATIVE) -> str | None:
    if memory_accounting not in MEMORY_ACCOUNTING_POLICIES:
        return "unknown memory-accounting policy"
    if not isinstance(sample, Telemetry):
        return "telemetry missing"
    values = (sample.pressure_free_pct, sample.free_spec_bytes, sample.physical_bytes,
              sample.swap_used_bytes, sample.group_footprint_bytes,
              sample.group_live_processes)
    if any(not isinstance(value, int) or isinstance(value, bool) for value in values):
        return "telemetry missing or noninteger"
    if not 0 <= sample.pressure_free_pct <= 100:
        return "pressure percentage outside 0..100"
    if sample.physical_bytes <= 0 or sample.free_spec_bytes < 0:
        return "physical/free telemetry is invalid"
    if sample.free_spec_bytes > sample.physical_bytes:
        return "free/speculative bytes exceed physical memory"
    if sample.swap_used_bytes < 0 or sample.group_footprint_bytes < 0:
        return "swap/footprint telemetry is invalid"
    if sample.group_live_processes < 0:
        return "process-group count is invalid"
    if memory_accounting == FREE_SPECULATIVE:
        # Optional cache diagnostics cannot change strict-policy admission.
        return None
    cache_values = (sample.speculative_bytes, sample.file_backed_bytes)
    if all(value is None for value in cache_values):
        return "file-cache counters missing" if memory_accounting == FILE_CACHE else None
    if any(not isinstance(value, int) or isinstance(value, bool) for value in cache_values):
        return "file-cache counters missing or noninteger"
    if not 0 <= sample.speculative_bytes <= sample.free_spec_bytes:
        return "speculative/free telemetry is inconsistent"
    if sample.file_backed_bytes < sample.speculative_bytes:
        return "speculative bytes exceed file-backed bytes"
    if _headroom_bytes(sample, FILE_CACHE) > sample.physical_bytes:
        return "free/file-cache bytes exceed physical memory"
    return None


def _headroom_bytes(sample: Telemetry, memory_accounting: str) -> int:
    if memory_accounting == FREE_SPECULATIVE:
        return sample.free_spec_bytes
    assert sample.speculative_bytes is not None and sample.file_backed_bytes is not None
    return sample.free_spec_bytes - sample.speculative_bytes + sample.file_backed_bytes


def _runtime_floor_bytes(sample: Telemetry) -> int:
    return (sample.physical_bytes * RUNTIME_FLOOR_PCT + 99) // 100


def assess_startup(sample: Telemetry, max_footprint_bytes: int, *,
                    memory_accounting: str = FREE_SPECULATIVE) -> str | None:
    invalid = _validate_sample(sample, memory_accounting)
    if invalid:
        return invalid
    if sample.pressure_free_pct < RUNTIME_FLOOR_PCT:
        return f"startup pressure {sample.pressure_free_pct}% is below {RUNTIME_FLOOR_PCT}%"
    required = _runtime_floor_bytes(sample) + max_footprint_bytes
    if _headroom_bytes(sample, memory_accounting) < required:
        if memory_accounting == FILE_CACHE:
            return ("startup headroom estimate (free + file-backed) is below the "
                    "15%-of-RAM policy floor plus the requested process-tree footprint budget")
        return ("startup headroom is below 15% physical memory plus the requested "
                "process-tree footprint budget")
    return None


def assess_runtime(sample: Telemetry, max_footprint_bytes: int,
                   max_swap_growth_bytes: int, initial_swap_bytes: int, *,
                   memory_accounting: str = FREE_SPECULATIVE) -> str | None:
    invalid = _validate_sample(sample, memory_accounting)
    if invalid:
        return invalid
    if sample.pressure_free_pct < RUNTIME_FLOOR_PCT:
        return f"pressure fell below {RUNTIME_FLOOR_PCT}%"
    if _headroom_bytes(sample, memory_accounting) < _runtime_floor_bytes(sample):
        if memory_accounting == FILE_CACHE:
            return "file-cache headroom estimate fell below the 15%-of-RAM policy floor"
        return "free/speculative bytes fell below the 15% physical-memory floor"
    if sample.group_footprint_bytes > max_footprint_bytes:
        return "process-tree footprint exceeded its requested maximum"
    if sample.swap_used_bytes - initial_swap_bytes > max_swap_growth_bytes:
        return "swap growth exceeded its requested maximum"
    return None


def process_is_gone_or_zombie(pid: int) -> bool:
    """Test helper: true when a PID is absent or cannot execute further."""
    try:
        result = subprocess.run(["/bin/ps", "-o", "stat=", "-p", str(pid)],
                                check=False, capture_output=True, text=True, timeout=2)
    except (OSError, subprocess.SubprocessError):
        return False
    state = result.stdout.strip()
    if result.returncode != 0 and not state:
        return True
    if result.returncode != 0:
        return False
    return not state or state.startswith("Z")


def _open_log(path: Path):
    try:
        stream = path.open("x", encoding="utf-8")
        os.fchmod(stream.fileno(), 0o600)
        return stream
    except OSError as error:
        raise ValueError(f"cannot create new log file: {path}") from error


def _log(stream, event: str, **fields) -> None:
    record = {"event": event, "monotonic": time.monotonic(), **fields}
    stream.write(json.dumps(record, sort_keys=True, separators=(",", ":")) + "\n")
    stream.flush()


def _log_sample(stream, sample: Telemetry, memory_accounting: str) -> None:
    _log(stream, "sample", pressure_free_pct=sample.pressure_free_pct,
         free_spec_bytes=sample.free_spec_bytes, physical_bytes=sample.physical_bytes,
         swap_used_bytes=sample.swap_used_bytes,
         group_footprint_bytes=sample.group_footprint_bytes,
         group_live_processes=sample.group_live_processes,
         speculative_bytes=sample.speculative_bytes, file_backed_bytes=sample.file_backed_bytes,
         headroom_bytes=_headroom_bytes(sample, memory_accounting))


def _libc_waitid_child_exit(pid: int) -> int | None:
    """Darwin fallback for Python runtimes that omit os.waitid."""
    if sys.platform != "darwin" or ctypes.sizeof(ctypes.c_void_p) != 8:
        raise TelemetryError("native waitid fallback requires 64-bit macOS")
    if (ctypes.sizeof(_DarwinSiginfo) != 104 or
            _DarwinSiginfo.si_pid.offset != 12 or
            _DarwinSiginfo.si_status.offset != 20):
        raise TelemetryError("Darwin siginfo_t layout is unsupported")
    try:
        libc = ctypes.CDLL(None, use_errno=True)
        function = libc.waitid
    except (OSError, AttributeError) as error:
        raise TelemetryError("libc waitid is unavailable") from error
    function.argtypes = [ctypes.c_int, ctypes.c_uint32,
                         ctypes.POINTER(_DarwinSiginfo), ctypes.c_int]
    function.restype = ctypes.c_int
    info = _DarwinSiginfo()
    # SDK sys/wait.h: P_PID=1, WEXITED=0x4, WNOHANG=0x1, WNOWAIT=0x20.
    if function(1, pid, ctypes.byref(info), 0x4 | 0x1 | 0x20) != 0:
        number = ctypes.get_errno()
        raise OSError(number, os.strerror(number), pid)
    if info.si_pid == 0:
        return None
    if info.si_pid != pid:
        raise TelemetryError("libc waitid returned a different child identity")
    if info.si_code == 1:  # CLD_EXITED
        return info.si_status
    if info.si_code in (2, 3):  # CLD_KILLED or CLD_DUMPED
        return 128 + info.si_status
    raise TelemetryError("libc waitid returned an unexpected child state")


def _peek_child_exit(pid: int):
    """Observe exit without reaping, pinning the session-leader PID/PGID."""
    names = ("P_PID", "WEXITED", "WNOHANG", "WNOWAIT")
    if callable(getattr(os, "waitid", None)) and all(hasattr(os, name) for name in names):
        try:
            result = os.waitid(os.P_PID, pid, os.WEXITED | os.WNOHANG | os.WNOWAIT)
        except (OSError, AttributeError) as error:
            raise TelemetryError("could not pin process-group leader identity") from error
    else:
        try:
            return _libc_waitid_child_exit(pid)
        except (OSError, AttributeError) as error:
            raise TelemetryError("could not pin process-group leader identity") from error
    if result is None or result.si_pid == 0:
        return None
    if result.si_code == getattr(os, "CLD_EXITED", 1):
        return result.si_status
    return 128 + result.si_status


def _signal_owned_group(child: subprocess.Popen, signum: int) -> None:
    # The session leader remains our unreaped child until cleanup completes.
    # Keeping it waitable with WNOWAIT prevents PID/PGID reuse in this interval.
    if child.pid <= 1 or child.returncode is not None:
        raise RuntimeError("refusing to signal invalid child process group")
    leader_exit = _peek_child_exit(child.pid)
    if leader_exit is None:
        try:
            current_pgid = os.getpgid(child.pid)
        except ProcessLookupError:
            # Darwin cannot query the PGID of an unreaped zombie. Recheck
            # without reaping to distinguish that race from lost identity.
            leader_exit = _peek_child_exit(child.pid)
            if leader_exit is None:
                raise RuntimeError("session leader identity disappeared before group signal")
        else:
            if current_pgid != child.pid:
                raise RuntimeError("session leader no longer owns its original process group")
    # Popen created this child with start_new_session=True, so its original
    # PGID is its pinned PID even when getpgid cannot query an exited child.
    # Process-list availability is not part of that identity proof. Darwin can
    # return EPERM for a zombie-only group, so skip it when that state can be
    # observed. A ps outage must still allow a signal to the pinned group.
    if leader_exit is not None:
        try:
            members = process_group_members(child.pid)
        except TelemetryError:
            pass
        else:
            if not any(not member.state.startswith("Z") for member in members):
                return
    # Keep the child unreaped through the final group signal.
    try:
        os.killpg(child.pid, signum)
    except ProcessLookupError:
        pass


def _terminate_owned_group(child: subprocess.Popen, grace_sec: float) -> None:
    _signal_owned_group(child, signal.SIGTERM)
    deadline = time.monotonic() + max(0.0, grace_sec)
    while time.monotonic() < deadline:
        try:
            members = process_group_members(child.pid)
            if not any(not item.state.startswith("Z") for item in members):
                # A missing leader alone is not evidence that descendants
                # stopped. An empty/zombie-only group plus its waitable exit is.
                if _peek_child_exit(child.pid) is not None:
                    return
        except TelemetryError:
            # The final signal revalidates waitability and original identity.
            break
        time.sleep(min(0.025, max(0.0, deadline - time.monotonic())))
    _signal_owned_group(child, signal.SIGKILL)
    # Delivering SIGKILL is not proof that every descendant has stopped. Keep
    # the leader unreaped until the original group is observed quiescent.
    deadline = time.monotonic() + max(0.1, grace_sec)
    last_error = None
    while time.monotonic() < deadline:
        try:
            members = process_group_members(child.pid)
            if (not any(not item.state.startswith("Z") for item in members) and
                    _peek_child_exit(child.pid) is not None):
                return
        except TelemetryError as error:
            last_error = error
        time.sleep(min(0.025, max(0.0, deadline - time.monotonic())))
    raise TelemetryError("owned process group did not become verifiably quiescent after SIGKILL") from last_error


def run_guarded(command: list[str], *, max_footprint_bytes: int,
                max_swap_growth_bytes: int, timeout_sec: float, log_path: Path,
                telemetry_provider: Callable[[int | None, float | None], Telemetry] = collect_telemetry,
                poll_interval_sec: float = 0.5,
                terminate_grace_sec: float = 1.0,
                memory_accounting: str = FREE_SPECULATIVE) -> int:
    """Run one command in a new session under startup and sampled runtime gates."""
    if (not command or max_footprint_bytes <= 0 or max_swap_growth_bytes < 0 or
            timeout_sec <= 0 or poll_interval_sec <= 0 or terminate_grace_sec < 0):
        raise ValueError("invalid command or nonpositive guard limits")
    if memory_accounting not in MEMORY_ACCOUNTING_POLICIES:
        raise ValueError("unknown memory-accounting policy")
    with _open_log(Path(log_path)) as stream:
        _log(stream, "limits", runtime_floor_pct=RUNTIME_FLOOR_PCT,
             max_footprint_bytes=max_footprint_bytes,
             max_swap_growth_bytes=max_swap_growth_bytes, timeout_sec=timeout_sec,
             memory_accounting=memory_accounting,
             headroom_formula="printed free + speculative" if memory_accounting == FREE_SPECULATIVE else "printed free + file-backed (includes speculative)",
             startup_rule="memory_pressure -Q >= 15% and selected headroom >= 15% physical + footprint budget",
             file_cache_limitation="file-backed bytes are a reclaimability estimate, not guaranteed clean or immediately free memory",
             limitation="sampled limits cannot prevent between-poll spikes or cover deliberate setsid/setpgid escape")
        try:
            initial = telemetry_provider(None, None)
        except Exception as error:
            _log(stream, "blocked", reason="telemetry unavailable",
                 detail=type(error).__name__)
            return EXIT_TELEMETRY
        invalid = _validate_sample(initial, memory_accounting)
        if invalid:
            _log(stream, "blocked", reason="telemetry unavailable", detail=invalid)
            return EXIT_TELEMETRY
        _log_sample(stream, initial, memory_accounting)
        startup_reason = assess_startup(initial, max_footprint_bytes,
                                         memory_accounting=memory_accounting)
        if startup_reason:
            _log(stream, "blocked", reason=startup_reason)
            return EXIT_ADMISSION

        previous_handlers = {}
        received: dict[str, int | None] = {"signal": None}

        def remember_signal(signum, _frame):
            received["signal"] = signum

        for signum in (signal.SIGTERM, signal.SIGINT):
            previous_handlers[signum] = signal.getsignal(signum)
            signal.signal(signum, remember_signal)

        child = None
        result_code = EXIT_TELEMETRY
        try:
            started = time.monotonic()
            try:
                child = subprocess.Popen(command, start_new_session=True)
            except OSError as error:
                _log(stream, "launch_failed", detail=type(error).__name__)
                return 126
            deadline = started + timeout_sec
            _log(stream, "started", pid=child.pid, pgid=child.pid)
            leader_code = None
            while True:
                now = time.monotonic()
                if received["signal"] is not None:
                    signum = received["signal"]
                    _log(stream, "stopping", reason="wrapper signal", signal=signum)
                    _terminate_owned_group(child, terminate_grace_sec)
                    result_code = 128 + signum
                    break
                if now >= deadline:
                    _log(stream, "stopping", reason="monotonic timeout")
                    _terminate_owned_group(child, terminate_grace_sec)
                    result_code = EXIT_TIMEOUT
                    break
                try:
                    sample = telemetry_provider(child.pid, deadline)
                    invalid = _validate_sample(sample, memory_accounting)
                    if invalid:
                        raise TelemetryError(invalid)
                except Exception as error:
                    _log(stream, "stopping", reason="telemetry unavailable",
                         detail=type(error).__name__)
                    _terminate_owned_group(child, terminate_grace_sec)
                    result_code = EXIT_TELEMETRY
                    break
                _log_sample(stream, sample, memory_accounting)
                if time.monotonic() >= deadline:
                    _log(stream, "stopping", reason="monotonic timeout")
                    _terminate_owned_group(child, terminate_grace_sec)
                    result_code = EXIT_TIMEOUT
                    break
                violation = assess_runtime(sample, max_footprint_bytes,
                                          max_swap_growth_bytes, initial.swap_used_bytes,
                                          memory_accounting=memory_accounting)
                if violation:
                    _log(stream, "stopping", reason=violation)
                    _terminate_owned_group(child, terminate_grace_sec)
                    result_code = EXIT_GUARD
                    break
                try:
                    if leader_code is None:
                        leader_code = _peek_child_exit(child.pid)
                except TelemetryError as error:
                    _log(stream, "stopping", reason="telemetry unavailable",
                         detail=type(error).__name__)
                    _terminate_owned_group(child, terminate_grace_sec)
                    result_code = EXIT_TELEMETRY
                    break
                if leader_code is not None and sample.group_live_processes == 0:
                    result_code = child.wait()
                    _log(stream, "finished", returncode=result_code)
                    return result_code
                time.sleep(min(poll_interval_sec, max(0.0, deadline - time.monotonic())))
            # Reap only after group termination, preserving the PGID until then.
            child.wait(timeout=max(2.0, terminate_grace_sec + 1.0))
            _log(stream, "finished", returncode=result_code)
            return result_code
        except (OSError, RuntimeError, subprocess.SubprocessError) as error:
            _log(stream, "cleanup_failed", detail=type(error).__name__,
                 reason="owned command did not reach verified termination and reaping")
            return EXIT_TELEMETRY
        finally:
            if child is not None and child.returncode is None:
                try:
                    _terminate_owned_group(child, terminate_grace_sec)
                except (OSError, RuntimeError) as error:
                    # Do not reap away the identity pin after group cleanup
                    # failed, or attempt an unbounded wait on a live child.
                    _log(stream, "cleanup_incomplete", phase="group_termination",
                         detail=type(error).__name__, pid=child.pid, leader_reaped=False)
                else:
                    try:
                        child.wait(timeout=max(2.0, terminate_grace_sec + 1.0))
                    except subprocess.TimeoutExpired:
                        _log(stream, "cleanup_incomplete", phase="leader_reap",
                             detail="TimeoutExpired", pid=child.pid, leader_reaped=False)
            for signum, previous in previous_handlers.items():
                signal.signal(signum, previous)


def _positive_int(text: str) -> int:
    try:
        value = int(text)
    except ValueError as error:
        raise argparse.ArgumentTypeError("must be an integer") from error
    if value <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return value


def _nonnegative_int(text: str) -> int:
    try:
        value = int(text)
    except ValueError as error:
        raise argparse.ArgumentTypeError("must be an integer") from error
    if value < 0:
        raise argparse.ArgumentTypeError("must be nonnegative")
    return value


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-footprint-mib", type=_positive_int, required=True)
    parser.add_argument("--max-swap-growth-mib", type=_nonnegative_int, required=True)
    parser.add_argument("--timeout-sec", type=_positive_int, required=True)
    parser.add_argument("--log", type=Path, required=True)
    parser.add_argument("--memory-accounting", choices=MEMORY_ACCOUNTING_POLICIES,
                        default=FREE_SPECULATIVE,
                        help="headroom policy: strict free+speculative (default), or free+file-backed estimate")
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    if not command:
        parser.error("expected -- command ...")
    try:
        return run_guarded(command,
                           max_footprint_bytes=args.max_footprint_mib * MIB,
                           max_swap_growth_bytes=args.max_swap_growth_mib * MIB,
                           timeout_sec=args.timeout_sec, log_path=args.log,
                           memory_accounting=args.memory_accounting)
    except (ValueError, OSError) as error:
        print(f"wan memory guard: {error}", file=sys.stderr)
        return EXIT_USAGE


if __name__ == "__main__":
    raise SystemExit(main())
