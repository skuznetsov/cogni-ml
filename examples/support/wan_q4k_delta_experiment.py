#!/usr/bin/env python3
"""Source-bound, guarded launcher for the opt-in Q4_K sparse-delta experiment.

Builds reuse the six hash-pinned native objects from the prior qualified MPS
stage experiment. Each invocation has exclusive intent/completion receipts;
reference construction is outside the timed native GPU process.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
BASE = Path('/opt/mldata_ssd/cogni-ml/video-scene-20261019')
OUT = BASE / 'evidence/wan-q4k-delta-20261023'
BUILD = BASE / 'build/wan-q4k-delta-20261023'
PRIOR = BASE / 'evidence/wan-qbit-mps-stage-20261022-v2/build-v2-intent.json'
PRIOR_SHA = '4cf673fd89d04112bc72de3e26e98d85111876f8f9b698042df2341aa2f6bca4'
SOURCE_FIXTURE = BASE / 'evidence/wan-quantized-projection-20261022/fixture-4480-v3'
ORACLE_PYTHON = Path('/opt/mldata_ssd/cogni-ml/drama3-fish-reference-20261015/venv/bin/python')


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(2**20), b''):
            digest.update(chunk)
    return digest.hexdigest()


def capture(command):
    return subprocess.check_output(command, cwd=ROOT, text=True).strip()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=['build', 'oracle', 'cpu-spec', 'analysis-spec', 'spec', 'probe'])
    parser.add_argument('tag')
    parser.add_argument('--build-tag', default='v1')
    parser.add_argument('--fixture-tag', default='r4-v1')
    parser.add_argument('--rows', type=int, choices=[4, 4480], default=4)
    parser.add_argument('--reverse-first', action='store_true', help='Reverse the first warmup and paired order')
    parser.add_argument('--rounds', type=int, choices=[16, 32])
    parser.add_argument('--compare-block-validation', action='store_true',
                        help='Pair per-value and per-block correction validation in the native probe')
    parser.add_argument('--compare-fused', action='store_true',
                        help='Compare fused Q4_K delta projection against block-validated staged controls')
    args = parser.parse_args()
    if args.compare_block_validation and args.compare_fused:
        parser.error('--compare-block-validation and --compare-fused are mutually exclusive')
    if (args.compare_block_validation or args.compare_fused) and args.mode != 'probe':
        parser.error('comparison modes are only available in probe mode')
    if args.mode in ['spec', 'build', 'probe'] and not (ROOT/'spec/q4k_delta_linear_spec.cr').is_file():
        raise RuntimeError('Required native Q4_K delta linear spec is missing')
    if args.rounds is None:
        args.rounds = 16
    if args.rounds not in [16, 32]:
        parser.error('rounds must be 16 or 32 balanced measured rounds')
    for value in [args.tag, args.build_tag, args.fixture_tag]:
        if not value.isascii() or not value.replace('-', '').isalnum():
            parser.error('tags must be ASCII alphanumeric with optional hyphens')
    OUT.mkdir(parents=True, exist_ok=True)
    for directory in [BUILD, BUILD/'tmp', BUILD/'crystal-cache', BUILD/'clang-cache']:
        directory.mkdir(parents=True, exist_ok=True)
    key = f'{args.mode}-{args.tag}'
    paths = {s: OUT/f'{key}.{s}' for s in
             ['intent.json', 'completion.json', 'stdout.log', 'stderr.log', 'guard.jsonl']}
    if any(path.exists() for path in paths.values()):
        raise RuntimeError('Refusing to overwrite experiment evidence')
    if sha(PRIOR) != PRIOR_SHA:
        raise RuntimeError('Prior object receipt changed')
    prior = json.loads(PRIOR.read_text())
    old_command = prior['command']
    flags = old_command[old_command.index('--link-flags')+1]
    objects = [Path(x) for x in flags.split() if x.endswith('.o')]
    if len(objects) != 6:
        raise RuntimeError('Expected six qualified native objects')
    files = {str(PRIOR): PRIOR_SHA, str(Path(__file__).resolve()): sha(__file__)}
    for obj in objects:
        if sha(obj) != prior['files'][str(obj.resolve())]:
            raise RuntimeError(f'Native object changed: {obj}')
        files[str(obj)] = sha(obj)
    for rel in ['src/ml/metal/bridge.mm', 'src/ml/video/wan_t5_mps_bridge.mm',
                'src/ml/video/wan_t5_rms_mps_bridge.mm',
                'src/ml/video/wan_t5_attention_mps_bridge.mm',
                'src/ml/video/metal_mps_command_chain.mm', 'src/ml/video/metal_stage_io.mm']:
        source = ROOT/rel
        expected = prior['files'].get(str(source.resolve()))
        if expected is not None and sha(source) != expected:
            raise RuntimeError(f'Native source no longer matches object ancestry: {rel}')
        files[str(source)] = sha(source)
    patterns = ['examples/support/wan_q4k_delta_reference.py',
                'examples/support/wan_affine_budget_reference.py',
                'examples/support/wan_q4k_range_reference.py',
                'examples/support/wan_quantized_projection_reference.py',
                'examples/support/wan_memory_guard.py',
                'spec/wan_q4k_delta_reference_spec.py']
    if args.mode == 'analysis-spec':
        patterns += ['examples/support/wan_q4k_delta_results.py',
                     'spec/wan_q4k_delta_results_spec.py']
    if args.mode not in ['oracle', 'cpu-spec']:
        patterns += ['src/**/*.cr', 'src/**/*.metal', 'spec/support/*.cr',
                     'examples/wan_q4k_delta*.cr', 'examples/support/wan_affine_budget_metrics.cr',
                     'spec/q4k_delta_weights_spec.cr',
                     'spec/q4k_delta_linear_spec.cr']
    for pattern in patterns:
        for source in ROOT.glob(pattern):
            if source.is_file():
                files[str(source.resolve())] = sha(source)
    binary = BUILD/f'probe-{args.build_tag if args.mode != "build" else args.tag}'
    if args.mode == 'build':
        if binary.exists():
            raise RuntimeError('Refusing to overwrite a binary')
        command = [old_command[0], 'build', '--release',
                   'examples/wan_q4k_delta_benchmark.cr', '-o', str(binary), '--link-flags', flags]
    elif args.mode == 'oracle':
        fixture = OUT/f'fixture-{args.tag}'
        if fixture.exists():
            raise RuntimeError('Refusing to overwrite a fixture')
        manifest = SOURCE_FIXTURE/'manifest.json'
        files[str(manifest)] = sha(manifest)
        for source in SOURCE_FIXTURE.iterdir():
            if source.is_file() and source.suffix in ['.bin', '.f32', '.bf16']:
                files[str(source)] = sha(source)
        command = [str(ORACLE_PYTHON), '-B', 'examples/support/wan_q4k_delta_reference.py',
                   '--fixture-dir', str(SOURCE_FIXTURE), '--output-dir', str(fixture),
                   '--rows', str(args.rows)]
    elif args.mode == 'cpu-spec':
        command = [str(ORACLE_PYTHON), '-B', 'spec/wan_q4k_delta_reference_spec.py']
    elif args.mode == 'analysis-spec':
        command = [sys.executable, '-B', 'spec/wan_q4k_delta_results_spec.py']
    elif args.mode == 'spec':
        command = [old_command[0], 'spec', 'spec/q4k_delta_weights_spec.cr',
                   'spec/q4k_delta_linear_spec.cr', '--link-flags', flags]
    else:
        receipt_path = OUT/f'build-{args.build_tag}.completion.json'
        receipt = json.loads(receipt_path.read_text())
        if receipt['returncode'] or sha(binary) != receipt['binary_sha256']:
            raise RuntimeError('No successful source-bound build')
        for path, expected in receipt['sources'].items():
            if sha(path) != expected:
                raise RuntimeError(f'Source changed after build: {path}')
        files[str(receipt_path)] = sha(receipt_path)
        files[str(binary)] = sha(binary)
        fixture = OUT/f'fixture-{args.fixture_tag}'
        oracle_path = OUT/f'oracle-{args.fixture_tag}.completion.json'
        oracle = json.loads(oracle_path.read_text())
        if (oracle['returncode'] or oracle['changed_files'] or
                oracle.get('produced_manifest_sha256') != sha(fixture/'manifest.json')):
            raise RuntimeError('No successful source-bound oracle for this exact fixture')
        for rel in ['examples/support/wan_q4k_delta_reference.py',
                    'examples/support/wan_affine_budget_reference.py',
                    'examples/support/wan_q4k_range_reference.py',
                    'examples/support/wan_quantized_projection_reference.py']:
            path = str(ROOT/rel)
            if oracle['sources'].get(path) != sha(path):
                raise RuntimeError(f'Oracle producer changed: {rel}')
        files[str(oracle_path)] = sha(oracle_path)
        for source in fixture.rglob('*'):
            if source.is_file():
                files[str(source)] = sha(source)
        command = [str(binary), '--fixture='+str(fixture), '--rounds='+str(args.rounds)]
        if args.compare_block_validation:
            command += ['--compare-block-validation']
        if args.compare_fused:
            command += ['--compare-fused']
        if args.reverse_first:
            command += ['--reverse-first']
    deadline = 240 if args.mode == 'build' else 300
    intent = {'schema': 'wan_q4k_delta_intent_v1', 'command': command,
              'mode': args.mode, 'tag': args.tag, 'files': files,
              'sparse_delta_ablation': True, 'reverse_first': args.reverse_first,
              'block_validation_comparison': args.compare_block_validation,
              'fused_comparison': args.compare_fused,
              'head': capture(['git', 'rev-parse', 'HEAD']),
              'status': capture(['git', 'status', '--short']),
              'macos': platform.mac_ver()[0], 'created_unix': time.time(),
              'memory_accounting': 'file-cache', 'reserve_pct': 15,
              'max_footprint_mib': 4096, 'max_swap_growth_mib': 64,
              'timeout_sec': deadline, 'prior_object_receipt_sha256': PRIOR_SHA}
    if args.mode == 'build':
        archive = OUT/f'sources-{args.tag}'
        archive.mkdir(exist_ok=False)
        for file in files:
            source = Path(file)
            if source.is_relative_to(ROOT):
                target = archive/source.relative_to(ROOT)
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(source, target)
    paths['intent.json'].write_text(json.dumps(intent, indent=2, sort_keys=True)+'\n')
    env = os.environ.copy()
    env.update(TMPDIR=str(BUILD/'tmp'), CRYSTAL_CACHE_DIR=str(BUILD/'crystal-cache'),
               CLANG_MODULE_CACHE_PATH=str(BUILD/'clang-cache'), PYTHONDONTWRITEBYTECODE='1',
               PYTORCH_ENABLE_MPS_FALLBACK='0')
    guarded = [sys.executable, '-B', 'examples/support/wan_memory_guard.py',
               '--memory-accounting=file-cache', '--max-footprint-mib=4096',
               '--max-swap-growth-mib=64', f'--timeout-sec={deadline}',
               '--log='+str(paths['guard.jsonl']), '--', *command]
    with paths['stdout.log'].open('xb') as out, paths['stderr.log'].open('xb') as err:
        returncode = subprocess.run(guarded, cwd=ROOT, env=env, stdout=out, stderr=err).returncode
    changed = [p for p, expected in files.items() if not Path(p).is_file() or sha(p) != expected]
    completion = {'schema': 'wan_q4k_delta_completion_v1', 'returncode': returncode,
                  'changed_files': changed, 'intent_sha256': sha(paths['intent.json']),
                  'stdout_sha256': sha(paths['stdout.log']), 'stderr_sha256': sha(paths['stderr.log']),
                  'guard_sha256': sha(paths['guard.jsonl']),
                  'sources': {p: h for p, h in files.items() if Path(p).is_relative_to(ROOT)},
                  'produced_manifest_sha256': sha(fixture/'manifest.json')
                  if args.mode == 'oracle' and returncode == 0 else None,
                  'binary_sha256': sha(binary) if binary.is_file() else None}
    paths['completion.json'].write_text(json.dumps(completion, indent=2, sort_keys=True)+'\n')
    print(json.dumps({'mode': args.mode, 'tag': args.tag, 'returncode': returncode,
                      'changed_files': changed, 'evidence': str(OUT)}))
    if returncode:
        print(paths['stderr.log'].read_text()[-6000:])
    return returncode or bool(changed)


if __name__ == '__main__':
    raise SystemExit(main())
