#!/usr/bin/env python3
"""Qualify sampler command-group timings; never infer full-model speedups."""
from __future__ import annotations

import argparse
import json
import math
import statistics
from pathlib import Path


def events(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines()
            if line.startswith('{')]


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def qualify(stdout: list[dict], stderr: list[dict]) -> dict:
    summaries = [row for row in stdout if row.get('event') == 'sampler_benchmark_summary']
    require(len(summaries) == 1, 'missing or duplicate successful summary')
    summary = summaries[0]
    invocations = [[row['id'] for row in stream
                    if row.get('event') == 'sampler_benchmark_invocation']
                   for stream in (stdout, stderr)]
    require(invocations == [[summary.get('invocation_id')]] * 2 and
            isinstance(summary.get('invocation_id'), str) and bool(summary['invocation_id']),
            'missing or mismatched invocation identity')
    profiles = summary['profiles']
    pairs, warmups = summary['pairs_per_profile'], summary['warmups_per_profile']
    require(profiles and len(profiles) == len(set(profiles)), 'invalid profiles')
    require(1 <= pairs <= 15 and warmups == 2, 'invalid repeat count')
    require(summary['finite_guard_included'] is True, 'finite guard omitted')
    require((summary['main_calls'], summary['tower_calls'], summary['main_normal_calls'],
             summary['main_retry_calls']) == (36, 153, 18, 18), 'incorrect call geometry')

    gpu = {}
    for row in stderr:
        if row.get('event') != 'drama3_gpu_command':
            continue
        index, seconds = row['index'], row['gpu_seconds']
        require(index not in gpu, 'duplicate GPU timeline index')
        require(math.isfinite(seconds) and seconds > 0, 'invalid GPU timestamp')
        gpu[index] = seconds

    grouped = {}
    group_rows = [row for row in stdout if row.get('event') == 'sampler_benchmark_group']
    consumed_indices = set()
    for row in group_rows:
        key = (row['profile'], row['warmup'], row['pair'])
        groups = grouped.setdefault(key, {})
        role, index = row['role'], row['command_index']
        require(role in ('main', 'tower') and role not in groups, 'duplicate or invalid role')
        require(index in gpu and index not in consumed_indices, 'missing or reused GPU timing')
        require((row['calls'], row['vocab']) == ((36, 4097) if role == 'main' else (153, 4096)),
                'incorrect group shape')
        consumed_indices.add(index)
        groups[role] = {**row, 'gpu_seconds': gpu[index]}
    require(consumed_indices == set(gpu), 'unassigned GPU command')
    schedule = [(profile, True, index) for profile in profiles for index in range(warmups)]
    schedule += [(profile, False, index) for index in range(pairs)
                 for profile in (profiles if index % 2 == 0 else list(reversed(profiles)))]
    expected = set(schedule)
    require(set(grouped) == expected, 'incomplete group schedule')
    pair_rows = [row for row in stdout if row.get('event') == 'sampler_benchmark_pair']
    seen = set()
    samples = {profile: [] for profile in profiles}
    total_pairs = len(expected)
    require(set(gpu) == set(range(2 * total_pairs)), 'non-contiguous GPU timeline')
    for position, key in enumerate(schedule):
        require(sorted(group['command_index'] for group in grouped[key].values()) ==
                [position * 2, position * 2 + 1], 'incorrect global profile order')
    for row in pair_rows:
        key = (row['profile'], row['warmup'], row['pair'])
        require(key in expected and key not in seen, 'invalid or duplicate pair')
        seen.add(key)
        require(row['waits_before_drain'] == 0 and row['readbacks_before_drain'] == 0,
                'intermediate synchronization/readback')
        require(row['submissions'] == 2 and row['final_drain_waits'] == 2,
                'incorrect completion boundary')
        require(row['verified_outputs'] == 189, 'missing output verification')
        require(math.isfinite(row['wall_seconds']) and row['wall_seconds'] > 0,
                'invalid pair wall time')
        graph = row['graph']
        require(graph['depth'] == 2 and graph['max_pending'] == 2 and
                graph['pending'] == 0 and graph['retained_bytes'] == 0, 'graph did not drain')
        groups = grouped[key]
        require(set(groups) == {'main', 'tower'}, 'incomplete pair roles')
        ordered = sorted(groups, key=lambda role: groups[role]['command_index'])
        require(ordered == row['order'] == (['main', 'tower'] if row['pair'] % 2 == 0
                                           else ['tower', 'main']), 'wrong submission order')
        if not row['warmup']:
            main, tower = groups['main']['gpu_seconds'], groups['tower']['gpu_seconds']
            samples[row['profile']].append({
                'pair': row['pair'], 'order': ordered, 'main_gpu_seconds': main,
                'tower_gpu_seconds': tower, 'total_gpu_seconds': main + tower,
                'pair_wall_seconds': row['wall_seconds'],
            })
    require(seen == expected, 'missing pair verification')
    final = summary['graph']
    require(final['submitted'] == final['completed'] == final['completion_waits'] == total_pairs * 2,
            'unexpected total submissions or waits')
    require(final['pending'] == 0 and final['retained_bytes'] == 0, 'final resources retained')
    require(summary['readbacks'] == {'control': total_pairs * 189 * 4,
                                     'terminal': total_pairs * 189 * 4}, 'unexpected total readbacks')
    costs = {}
    for profile, rows in samples.items():
        costs[profile] = {'pairs': len(rows), 'samples': rows}
        for key in ('main_gpu_seconds', 'tower_gpu_seconds', 'total_gpu_seconds', 'pair_wall_seconds'):
            values = [row[key] for row in rows]
            costs[profile][key] = {'median': statistics.median(values), 'min': min(values), 'max': max(values)}
        costs[profile]['main_gpu_seconds_per_call'] = statistics.median(row['main_gpu_seconds'] / 36 for row in rows)
        costs[profile]['tower_gpu_seconds_per_call'] = statistics.median(row['tower_gpu_seconds'] / 153 for row in rows)
    return {'format': 'drama3-sampler-cost-v1', 'qualified': True,
            'invocation_id': summary['invocation_id'],
            'scope': 'Warm fixed-input Context.sample command-group GPU time, including finite guard; not production attribution or speedup.',
            'fixture': summary['fixture'], 'profiles': costs, 'warmups_per_profile': warmups,
            'verified_outputs': total_pairs * 189, 'timed_command_buffers': len(gpu),
            'no_wait_or_readback_before_pair_drain': True, 'summary': summary}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stdout', required=True, type=Path)
    parser.add_argument('--stderr', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    require(not args.output.exists(), 'output already exists')
    report = qualify(events(args.stdout), events(args.stderr))
    args.output.write_text(json.dumps(report, indent=2) + '\n')
    print(f"Qualified {report['timed_command_buffers']} commands / {report['verified_outputs']} outputs")


if __name__ == '__main__':
    main()
