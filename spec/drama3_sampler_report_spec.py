"""Controls for accepting asynchronous sampler timing evidence."""
import copy
import unittest

from scripts.drama3_sampler_report import qualify


def fixture(profiles=('control',), pairs=1):
    out, err = [], []
    schedule = [(profile, True, pair) for profile in profiles for pair in range(2)]
    schedule += [(profile, False, pair) for pair in range(pairs)
                 for profile in (profiles if pair % 2 == 0 else tuple(reversed(profiles)))]
    for command_pair, (profile, warmup, pair) in enumerate(schedule):
        order = ['main', 'tower'] if pair % 2 == 0 else ['tower', 'main']
        graph = {'depth': 2, 'submitted': (command_pair + 1) * 2,
                 'completed': (command_pair + 1) * 2, 'pending': 0, 'max_pending': 2,
                 'completion_waits': (command_pair + 1) * 2, 'retained_bytes': 0}
        for slot, role in enumerate(order):
            index = command_pair * 2 + slot
            out.append({'event': 'sampler_benchmark_group', 'profile': profile,
                        'pair': pair, 'warmup': warmup, 'role': role, 'command_index': index,
                        'calls': 36 if role == 'main' else 153,
                        'vocab': 4097 if role == 'main' else 4096})
            err.append({'event': 'drama3_gpu_command', 'index': index, 'gpu_seconds': 0.01})
        out.append({'event': 'sampler_benchmark_pair', 'profile': profile, 'pair': pair,
                    'warmup': warmup, 'order': order, 'waits_before_drain': 0,
                    'readbacks_before_drain': 0, 'submissions': 2, 'final_drain_waits': 2,
                    'verified_outputs': 189, 'graph': graph, 'wall_seconds': 0.03})
    out.append({'event': 'sampler_benchmark_summary', 'profiles': list(profiles),
                'pairs_per_profile': pairs, 'warmups_per_profile': 2, 'finite_guard_included': True,
                'main_calls': 36, 'tower_calls': 153, 'main_normal_calls': 18, 'main_retry_calls': 18,
                'graph': graph, 'readbacks': {'control': len(schedule) * 189 * 4,
                                            'terminal': len(schedule) * 189 * 4},
                'fixture': 'synthetic control only', 'invocation_id': 'invocation-a'})
    out.append({'event': 'sampler_benchmark_invocation', 'id': 'invocation-a'})
    err.append({'event': 'sampler_benchmark_invocation', 'id': 'invocation-a'})
    return out, err


class SamplerTimingControls(unittest.TestCase):
    def test_accepts_complete_schedule_and_excludes_warmups(self):
        report = qualify(*fixture())
        self.assertTrue(report['qualified'])
        self.assertEqual(report['profiles']['control']['pairs'], 1)
        self.assertEqual(report['profiles']['control']['total_gpu_seconds']['median'], 0.02)
        self.assertEqual(report['verified_outputs'], 567)

    def test_rejects_added_wait_and_readback(self):
        for key in ('waits_before_drain', 'readbacks_before_drain'):
            with self.subTest(key=key):
                out, err = fixture()
                out[2][key] = 1
                with self.assertRaisesRegex(ValueError, 'intermediate synchronization'):
                    qualify(out, err)

    def test_rejects_missing_or_reused_timer(self):
        out, err = fixture()
        with self.assertRaisesRegex(ValueError, 'missing or reused GPU timing'):
            qualify(out, err[:-2] + err[-1:])
        out[1]['command_index'] = out[0]['command_index']
        with self.assertRaisesRegex(ValueError, 'missing or reused GPU timing'):
            qualify(out, err)

    def test_rejects_zero_nan_and_duplicate_timestamps(self):
        for value in (0, float('nan'), -1):
            out, err = fixture()
            err[0]['gpu_seconds'] = value
            with self.assertRaisesRegex(ValueError, 'invalid GPU timestamp'):
                qualify(out, err)
        out, err = fixture()
        err.append(copy.deepcopy(err[0]))
        with self.assertRaisesRegex(ValueError, 'duplicate GPU timeline'):
            qualify(out, err)

    def test_rejects_partial_output_or_resource_drain(self):
        out, err = fixture()
        out[2]['verified_outputs'] -= 1
        with self.assertRaisesRegex(ValueError, 'missing output verification'):
            qualify(out, err)
        out, err = fixture()
        out[2]['graph']['retained_bytes'] = 4
        with self.assertRaisesRegex(ValueError, 'graph did not drain'):
            qualify(out, err)

    def test_rejects_missing_pair_and_omitted_guard(self):
        out, err = fixture()
        del out[2]
        with self.assertRaisesRegex(ValueError, 'missing pair verification'):
            qualify(out, err)
        out, err = fixture()
        out[-2]['finite_guard_included'] = False
        with self.assertRaisesRegex(ValueError, 'finite guard omitted'):
            qualify(out, err)

    def test_rejects_cross_run_streams_and_shifted_indices(self):
        out, err = fixture()
        err[-1]['id'] = 'invocation-b'
        with self.assertRaisesRegex(ValueError, 'invocation identity'):
            qualify(out, err)
        out, err = fixture()
        for row in out:
            if 'command_index' in row:
                row['command_index'] += 100
        for row in err:
            if 'index' in row:
                row['index'] += 100
        with self.assertRaisesRegex(ValueError, 'non-contiguous GPU timeline'):
            qualify(out, err)

    def test_rejects_invalid_wall_time(self):
        for value in (0, -1, float('nan'), float('inf')):
            out, err = fixture()
            out[2]['wall_seconds'] = value
            with self.assertRaisesRegex(ValueError, 'invalid pair wall time'):
                qualify(out, err)

    def test_rejects_wrong_cross_profile_order(self):
        out, err = fixture(('a', 'b'), pairs=2)
        self.assertTrue(qualify(out, err)['qualified'])
        for row in out:
            if not row.get('warmup', True) and row.get('pair') == 1:
                row['profile'] = 'b' if row['profile'] == 'a' else 'a'
        with self.assertRaisesRegex(ValueError, 'incorrect global profile order'):
            qualify(out, err)


if __name__ == '__main__':
    unittest.main()
