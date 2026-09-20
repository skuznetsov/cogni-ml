#!/usr/bin/env python3
"""No GPU: qualify exact-process sampling against one bounded child fixture."""
import json
import os
from pathlib import Path
import subprocess
import sys


def main():
    sampler, fixture = map(lambda p: str(Path(p).resolve()), sys.argv[1:])
    child = subprocess.Popen([fixture], stdout=subprocess.PIPE, text=True)
    try:
        assert child.stdout.readline().strip() == 'ready'
        args = [sampler, str(child.pid), str(os.getpid()), fixture, '3']
        for index, wrong in [(1, '1'), (1, '-1'), (1, '0'),
                             (1, '2x'), (2, '1'), (3, '/wrong/path'),
                             (4, '0'), (4, '31'), (4, '1x')]:
            bad = args.copy()
            bad[index] = wrong
            result = subprocess.run(bad, capture_output=True, text=True, timeout=4)
            assert result.returncode != 0 and not result.stdout, (bad, result)
        result = subprocess.run(args, capture_output=True, text=True, timeout=5)
        assert result.returncode == 0, result.stderr
        rows = [json.loads(line) for line in result.stdout.splitlines()]
        assert rows[-1]['event'] == 'end' and rows[-1]['reason'] == 'deadline'
        samples = rows[:-1]
        assert 20 <= len(samples) <= 31
        assert len({r['start_abstime'] for r in samples}) == 1
        assert all(r['pid'] == child.pid and r['ppid'] == os.getpid() for r in samples)
        assert all(r['mach_begin_s'] <= r['mach_end_s'] for r in samples)
        assert all(a['mach_end_s'] <= b['mach_begin_s'] for a, b in zip(samples, samples[1:]))
        terminal = subprocess.run(args, capture_output=True, text=True, timeout=5)
        assert terminal.returncode == 0, terminal.stderr
        terminal_rows = [json.loads(line) for line in terminal.stdout.splitlines()]
        assert terminal_rows[-1]['event'] == 'end' and terminal_rows[-1]['reason'] == 'exited'
        assert child.wait(timeout=5) == 0
        control = json.loads(child.stdout.read())
        assert control['cpu_ms'] > 50 and .85 < control['cpu_to_getrusage_ratio'] < 1.15, control
        assert control['resident_delta'] >= 24 * 1024**2, control
        assert control['disk_read_delta'] >= 8 * 1024**2, control
        assert samples[-1]['resident_bytes'] - samples[0]['resident_bytes'] >= 24 * 1024**2
        assert samples[-1]['user_ns'] - samples[0]['user_ns'] > 50_000_000
        absent = subprocess.run(args, capture_output=True, text=True, timeout=5)
        assert absent.returncode != 0 and not absent.stdout
        print(json.dumps({'passed': True, 'samples': len(samples), 'control': control}))
    finally:
        # Fixture is our own bounded child, normally exits by itself in five seconds.
        child.wait(timeout=8)


if __name__ == '__main__':
    main()
