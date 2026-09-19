"""Run the standalone fake-device native test and check its actual JSON output.

Usage: python3 spec/metal_pipeline_profile_output_test.py /absolute/native-test
No real Metal device, model, or GPU command is created by that test.
"""
import json
import math
import subprocess
import sys


def check(binary):
    result = subprocess.run([binary], capture_output=True, text=True, timeout=15)
    assert result.returncode == 0, result.stderr
    assert 'native pipeline profile tests PASS' in result.stdout
    rows = [json.loads(line) for line in result.stderr.splitlines() if line.startswith('{')]
    expected = [
        ('source', 'source_ok', ['library', 'function', 'pipeline']),
        ('file', 'file_miss', ['library', 'function', 'pipeline']),
        ('file', 'file_hit', ['library', 'function', 'pipeline']),
        ('default', 'default_ok', ['function', 'pipeline']),
        ('source', 'source_failed', ['library']),
        ('source', 'source_failed', ['library', 'function']),
        ('source', 'source_failed', ['library', 'function', 'pipeline']),
        ('source', 'quoted"\nname', ['library', 'function', 'pipeline']),
    ]
    assert [(r['route'], r['function'], r['stage']) for r in rows] == [
        (route, function, stage) for route, function, stages in expected for stage in stages]
    assert [i for i, r in enumerate(rows) if not r['success']] == [11, 13, 16]
    assert [i for i, r in enumerate(rows) if r['cache_hit']] == [6]
    previous_end = 0
    for row in rows:
        assert row['event'] == 'metal_pipeline_profile' and row['timing_valid'] is True
        begin, end, elapsed = [row[k] for k in ('mach_begin_s', 'mach_end_s', 'elapsed_ms')]
        assert all(math.isfinite(v) for v in (begin, end, elapsed))
        assert 0 < begin <= end and previous_end <= begin
        assert abs((end - begin) * 1000 - elapsed) < 0.00001
        if not row['cache_hit']:
            assert elapsed >= 4, row  # Known 5ms delay in each fake API call.
        previous_end = end
    print(f'pipeline JSON qualification PASS: {len(rows)} intervals, disabled silence, '
          'three failures, file-library hit, escaped name, known delays; no GPU')


if __name__ == '__main__':
    check(sys.argv[1])
