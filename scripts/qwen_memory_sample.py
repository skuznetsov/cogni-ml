#!/usr/bin/env python3
"""Read-only macOS memory sampler; never launches, signals or controls a workload.

RSS is a snapshot of PPID-linked descendants, not Metal allocation accounting.
Shared pages can be counted twice; reparented children can leave this snapshot.
Probe phases are last-flushed provider events, not kernel-level attribution.
"""

import argparse
import json
import os
from pathlib import Path
import re
import subprocess
import time


def parse_processes(text):
    rows = {}
    for line in text.splitlines():
        if not line.strip():
            continue
        fields = line.split(None, 3)
        if len(fields) != 4:
            raise ValueError('malformed process row')
        pid, ppid, rss = map(int, fields[:3])
        if pid <= 0 or ppid < 0 or rss < 0 or pid in rows:
            raise ValueError('invalid or duplicate process row')
        rows[pid] = {'pid': pid, 'ppid': ppid, 'rss_kib': rss,
                     'name': os.path.basename(fields[3])}
    if not rows:
        raise ValueError('empty process snapshot')
    return rows


def summarize_processes(rows, root_pid):
    members = {root_pid} if root_pid in rows else set()
    while True:
        more = {pid for pid, row in rows.items() if row['ppid'] in members} - members
        if not more:
            break
        members.update(more)
    external = sorted((r for pid, r in rows.items() if pid not in members),
                      key=lambda r: r['rss_kib'], reverse=True)[:5]
    return {'root_present': root_pid in rows, 'tree_pids': sorted(members),
            'tree_rss_kib': sum(rows[p]['rss_kib'] for p in members) if members else None,
            'external_top': [{k: row[k] for k in ('pid', 'rss_kib', 'name')} for row in external]}


def parse_free_percent(text):
    match = re.search(r'System-wide memory free percentage:\s*(\d+)%', text)
    if match is None or not 0 <= int(match[1]) <= 100:
        raise ValueError('missing or invalid memory free percentage')
    return int(match[1])


def phase_from_text(text):
    phase = 'before_call_1'
    for line in text.splitlines():
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue  # A concurrently written final line can be incomplete.
        if not isinstance(event, dict):
            continue
        kind, facts = event.get('event'), event.get('facts', {})
        call = facts.get('call') if isinstance(facts, dict) else None
        if call in (1, 2) and kind in ('call_begin', 'call_end'):
            phase = ('call_' if kind == 'call_begin' else 'after_call_') + str(call)
        elif kind in ('tool_gap', 'second_input', 'complete'):
            phase = kind
    return phase


def command_text(command):
    result = subprocess.run(command, check=True, capture_output=True, text=True, timeout=2)
    if len(result.stdout) > 1_048_576:
        raise ValueError('oversized command output')
    return result.stdout


def sample(root_pid, phase_log):
    started = time.monotonic()
    result = {'wall_time_ns': time.time_ns(), 'monotonic_s': started,
              'phase': None, 'system_free_pct': None, 'root_present': None,
              'tree_rss_kib': None, 'tree_pids': [], 'external_top': [], 'errors': []}
    try:
        rows = parse_processes(command_text(['/bin/ps', '-axo', 'pid=,ppid=,rss=,comm=']))
        result.update(summarize_processes(rows, root_pid))
    except (OSError, ValueError, subprocess.SubprocessError) as error:
        result['errors'].append('ps:' + type(error).__name__)
    try:
        result['system_free_pct'] = parse_free_percent(command_text(['/usr/bin/memory_pressure', '-Q']))
    except (OSError, ValueError, subprocess.SubprocessError) as error:
        result['errors'].append('memory_pressure:' + type(error).__name__)
    try:
        with phase_log.open() as source:
            text = source.read(262_145)
        if len(text) > 262_144:
            raise ValueError('oversized phase log')
        result['phase'] = phase_from_text(text)
    except (OSError, ValueError) as error:
        result['errors'].append('phase:' + type(error).__name__)
    result['collection_ms'] = (time.monotonic() - started) * 1000
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--pid', required=True, type=int, help='existing guarded process PID')
    parser.add_argument('--phase-log', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--seconds', type=float, default=310)
    parser.add_argument('--interval', type=float, default=2)
    args = parser.parse_args()
    if args.pid <= 1 or not 0 < args.seconds <= 330 or not 1 <= args.interval <= 10:
        parser.error('require pid > 1, 0 < seconds <= 330, and 1 <= interval <= 10')
    deadline = time.monotonic() + args.seconds
    valid = 0
    with args.output.open('x') as output:
        while time.monotonic() < deadline:
            row = sample(args.pid, args.phase_log)
            output.write(json.dumps(row) + '\n')
            output.flush()
            valid += not row['errors'] and row['root_present'] is True
            if row['root_present'] is False:
                break
            delay = min(args.interval, max(0, deadline - time.monotonic()))
            time.sleep(delay)
    return 0 if valid else 2


if __name__ == '__main__':
    raise SystemExit(main())
