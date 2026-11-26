import copy
from pathlib import Path
import tempfile
import unittest

import torch

from data import PROGRAMS, make_prompt, infer_target, split_tasks, load_tasks
from tail import TailConfig, QwenTail, configure_mode


class DataTests(unittest.TestCase):
    def test_test_answers_never_affect_supervision(self):
        task = {"train": [{"input": [[1, 2], [3, 4]],
                           "output": [[3, 1], [4, 2]]}],
                "test": [{"input": [[5, 6]], "output": [[999]]}]}
        other = copy.deepcopy(task)
        other["test"][0]["output"] = [[123]]
        self.assertEqual(infer_target(task), infer_target(other))
        self.assertEqual(make_prompt(task), make_prompt(other))
        self.assertEqual(PROGRAMS[infer_target(task)].name, "rotate90")
        self.assertNotIn("999", make_prompt(task))

    def test_task_content_deduplicated_across_split(self):
        task = {"train": [{"input": [[1, 2]], "output": [[2, 1]]}],
                "test": [{"input": [[3, 4]], "output": [[4, 3]]}]}
        with self.assertRaisesRegex(ValueError, "duplicate"):
            split_tasks({"a": task, "b": copy.deepcopy(task)}, 42, 0.25)

    def test_eval_directory_is_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaisesRegex(ValueError, "evaluation"):
                load_tasks(Path(d) / "evaluation")

    def test_unsupported_and_ambiguous_tasks_have_no_target(self):
        unsupported = {"train": [{"input": [[1, 2]], "output": [[7, 8]]}]}
        ambiguous = {"train": [{"input": [[1]], "output": [[1]]}]}
        self.assertIsNone(infer_target(unsupported))
        self.assertIsNone(infer_target(ambiguous))


class TailTests(unittest.TestCase):
    def model(self):
        torch.manual_seed(7)
        return QwenTail(TailConfig(hidden=8, intermediate=12, heads=2,
                                   kv_heads=1, head_dim=4, rotary_dim=2), 8)

    def test_causal_future_does_not_change_prefix(self):
        m = self.model()
        x = torch.randn(4, 8)
        a = m.block(x)
        x[-1] += 100
        b = m.block(x)
        torch.testing.assert_close(a[:-1], b[:-1])

    def test_only_selected_weights_update(self):
        for mode in ["head", "last-block"]:
            with self.subTest(mode=mode):
                m = self.model()
                configure_mode(m, mode)
                before = {n: p.detach().clone() for n, p in m.named_parameters()}
                x = torch.randn(4, 8, requires_grad=True)
                opt = torch.optim.AdamW([p for p in m.parameters() if p.requires_grad], lr=.01)
                loss = torch.nn.functional.cross_entropy(m(x).unsqueeze(0), torch.tensor([2]))
                loss.backward()
                self.assertIsNone(x.grad)
                for n, p in m.named_parameters():
                    if not p.requires_grad:
                        self.assertIsNone(p.grad, n)
                opt.step()
                changed = {n for n, p in m.named_parameters() if not torch.equal(p, before[n])}
                self.assertTrue(changed)
                self.assertTrue(all((n == "head.weight") if mode == "head"
                                    else n.startswith("block.") for n in changed), changed)
                if mode == "last-block":
                    self.assertTrue(any("attn_q" in n for n in changed))
                    self.assertTrue(any("ffn_down" in n for n in changed))


if __name__ == "__main__":
    torch.set_num_threads(2)
    unittest.main()
