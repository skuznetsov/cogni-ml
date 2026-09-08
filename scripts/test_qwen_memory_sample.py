import unittest
from pathlib import Path
from unittest.mock import patch

from qwen_memory_sample import parse_free_percent, parse_processes, phase_from_text, sample, summarize_processes


class MemorySampleTests(unittest.TestCase):
    def test_descendants_not_unrelated_and_no_full_paths(self):
        rows = parse_processes('10 1 100 /tmp/guard\n11 10 200 /tmp/probe\n12 11 300 /tmp/child\n99 1 900 /Applications/Some App\n')
        result = summarize_processes(rows, 10)
        self.assertEqual(result['tree_rss_kib'], 600)
        self.assertEqual(result['tree_pids'], [10, 11, 12])
        self.assertEqual(result['external_top'][0], {'pid': 99, 'rss_kib': 900, 'name': 'Some App'})

    def test_missing_root_is_unknown_not_zero(self):
        result = summarize_processes(parse_processes('11 1 200 /tmp/probe'), 10)
        self.assertFalse(result['root_present'])
        self.assertIsNone(result['tree_rss_kib'])

    def test_cycles_terminate(self):
        result = summarize_processes(parse_processes('10 11 100 /tmp/a\n11 10 200 /tmp/b'), 10)
        self.assertEqual(result['tree_rss_kib'], 300)

    def test_malformed_or_empty_ps_is_not_zero_memory(self):
        for text in ('', 'garbage', '10 1 -2 /tmp/probe', '10 1 12 /tmp/a\n10 1 13 /tmp/b'):
            with self.assertRaises(ValueError):
                parse_processes(text)

    def test_free_percent_and_missing_metric(self):
        self.assertEqual(parse_free_percent('System-wide memory free percentage: 35%\n'), 35)
        for text in ('', 'System-wide memory free percentage: 101%'):
            with self.assertRaises(ValueError):
                parse_free_percent(text)

    def test_coarse_phase_does_not_copy_private_facts(self):
        text = '{"event":"call_begin","facts":{"call":1,"secret":"not emitted"}}\n'
        self.assertEqual(phase_from_text(text), 'call_1')
        self.assertEqual(phase_from_text(text + '{"event":"call_end","facts":{"call":1}}\n'), 'after_call_1')
        self.assertEqual(phase_from_text(text + '{"event":"tool_gap","facts":{}}\n'), 'tool_gap')
        self.assertEqual(phase_from_text(text + '{"event":"call_begin","facts":{"call":2}}\n'), 'call_2')
        self.assertEqual(phase_from_text('{"event":"complete","facts":{}}\n'), 'complete')

    def test_partial_last_line_keeps_last_complete_phase(self):
        text = '{"event":"call_begin","facts":{"call":1}}\n{"event":"call_end"'
        self.assertEqual(phase_from_text(text), 'call_1')

    def test_all_phase_labels(self):
        self.assertEqual(phase_from_text(''), 'before_call_1')
        for kind in ('tool_gap', 'second_input', 'complete'):
            self.assertEqual(phase_from_text('{"event":"' + kind + '"}'), kind)
        for call in (1, 2):
            for kind, prefix in (('call_begin', 'call_'), ('call_end', 'after_call_')):
                self.assertEqual(phase_from_text(
                    '{"event":"%s","facts":{"call":%d}}' % (kind, call)), prefix + str(call))

    def test_failed_sources_remain_unknown_without_error_payloads(self):
        with patch('qwen_memory_sample.command_text', side_effect=OSError('private data')):
            with patch.object(Path, 'open', side_effect=OSError('private path')):
                row = sample(10, Path('/unused'))
        for key in ('system_free_pct', 'tree_rss_kib', 'root_present', 'phase'):
            self.assertIsNone(row[key])
        self.assertEqual(row['errors'], ['ps:OSError', 'memory_pressure:OSError', 'phase:OSError'])
        self.assertNotIn('private', str(row))


if __name__ == '__main__':
    unittest.main()
