#!/usr/bin/env python3
"""Pressure-only admission check for the documented 64-GiB Qwen two-call replay.

Never launches a workload or changes a process. Exit 0 means this one snapshot
meets the operator-selected threshold, NOT that a GPU run is safe or correct.
Launch with a 30% runtime guard, 24-GiB cap, timeout and Metal lease.
See docs/qwen-prefill-command-trace.md for workload scope and refresh triggers.
"""

import json
import re
import subprocess

PROFILE_BYTES = 64 * 1024**3
RUNTIME_FLOOR_PCT = 30
# Explicit operator policy (2026-09-18), not a measured demand bound.
# This checker reports the runtime floor but does not configure the runner.
REQUIRED_INITIAL_PCT = 75


def assess(report):
    capacities = re.findall(r'^The system has (\d+) \([^\n]+\)\.$', report, re.MULTILINE)
    percentages = re.findall(r'^System-wide memory free percentage: (\d+)%$', report, re.MULTILINE)
    if len(capacities) != 1 or len(percentages) != 1:
        raise ValueError('missing, malformed or duplicate memory report')
    if int(capacities[0]) != PROFILE_BYTES:
        raise ValueError('profile applies only to the documented 64-GiB host')
    free = int(percentages[0])
    if not 0 <= free <= 100:
        raise ValueError('memory percentage out of range')
    return {
        'profile': 'qwen38_27b_q4_f32_two_call_7813_64gib',
        'free_pct': free,
        'required_initial_pct': REQUIRED_INITIAL_PCT,
        'runtime_floor_pct': RUNTIME_FLOOR_PCT,
        'memory_preflight_pass': free >= REQUIRED_INITIAL_PCT,
        'runtime_guard_still_required': True,
    }


def main():
    try:
        result = subprocess.run(['/usr/bin/memory_pressure', '-Q'],
                                check=True, capture_output=True, text=True, timeout=5)
        facts = assess(result.stdout)
    except (OSError, ValueError, subprocess.SubprocessError) as error:
        print(json.dumps({'memory_preflight_pass': False, 'error': type(error).__name__}))
        return 2
    print(json.dumps(facts))
    return 0 if facts['memory_preflight_pass'] else 75


if __name__ == '__main__':
    raise SystemExit(main())
