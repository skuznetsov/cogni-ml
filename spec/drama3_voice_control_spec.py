"""CPU-only controls for the bounded MLX Fish Omni generation helper."""

from __future__ import annotations

import contextlib
import importlib.util
import io
import tempfile
import unittest
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "drama3" / "generate_mlx_control.py"
SPEC = importlib.util.spec_from_file_location("drama3_mlx_control_under_test", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
control = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(control)


class FakeMx:
    """Small NumPy-backed CPU surface for the sampler operations under test."""

    float32 = np.float32
    int32 = np.int32
    inf = np.inf

    def __init__(self) -> None:
        self.array_calls: list[tuple[object, object]] = []

    def array(self, values: object, dtype: object = None) -> np.ndarray:
        self.array_calls.append((values, dtype))
        return np.asarray(values, dtype=dtype)

    @staticmethod
    def argpartition(values: np.ndarray, kth: int, axis: int = -1) -> np.ndarray:
        # Stable descending order makes equal-logit candidate order deterministic.
        del kth
        return np.argsort(-values, axis=axis, kind="stable")

    @staticmethod
    def take_along_axis(values: np.ndarray, indices: np.ndarray, axis: int) -> np.ndarray:
        return np.take_along_axis(values, indices, axis=axis)

    @staticmethod
    def argsort(values: np.ndarray, axis: int = -1) -> np.ndarray:
        return np.argsort(values, axis=axis, kind="stable")

    @staticmethod
    def logsumexp(values: np.ndarray, axis: int, keepdims: bool) -> np.ndarray:
        maximum = np.max(values, axis=axis, keepdims=True)
        result = maximum + np.log(np.sum(np.exp(values - maximum), axis=axis, keepdims=True))
        return result if keepdims else np.squeeze(result, axis=axis)

    @staticmethod
    def cumsum(values: np.ndarray, axis: int) -> np.ndarray:
        return np.cumsum(values, axis=axis, dtype=values.dtype)

    @staticmethod
    def exp(values: np.ndarray) -> np.ndarray:
        return np.exp(values)

    @staticmethod
    def where(condition: np.ndarray, left: object, right: np.ndarray) -> np.ndarray:
        return np.where(condition, left, right)

    @classmethod
    def softmax(cls, values: np.ndarray, axis: int = -1) -> np.ndarray:
        return cls.exp(values - cls.logsumexp(values, axis=axis, keepdims=True))

    @staticmethod
    def reshape(values: np.ndarray, shape: tuple[int, ...]) -> np.ndarray:
        return np.reshape(values, shape)

    @staticmethod
    def sum(values: np.ndarray, axis: int) -> np.ndarray:
        return np.sum(values, axis=axis)

    @staticmethod
    def minimum(left: np.ndarray, right: np.ndarray) -> np.ndarray:
        return np.minimum(left, right)

    @staticmethod
    def concatenate(values: list[np.ndarray], axis: int = 0) -> np.ndarray:
        return np.concatenate(values, axis=axis)


class MlxControlSamplerSpec(unittest.TestCase):
    def test_stock_and_tape_dispatch_cover_all_eleven_sampler_slots(self) -> None:
        mx = FakeMx()
        stock = control.SamplerControl(mx, None, {0})
        official_calls: list[tuple[object, float, int, float]] = []

        def official(logits: object, temperature: float, top_k: int, top_p: float) -> str:
            official_calls.append((logits, temperature, top_k, top_p))
            return "official"

        for slot in range(control.DRAW_STRIDE):
            self.assertEqual(stock.sample(official, slot, 0.8, 30, 0.9), "official")

        self.assertEqual(len(official_calls), 11)
        self.assertEqual(stock.draw_count, control.DRAW_STRIDE)
        self.assertEqual(
            [entry[1] for entry in stock.capture_refs],
            ["main", "main_retry", *[f"codebook_{index}" for index in range(1, 10)]],
        )
        self.assertEqual(mx.array_calls, [])

        mx = FakeMx()
        taped = control.SamplerControl(mx, [0.25] * control.DRAW_STRIDE, {0})
        logits = np.zeros((1, 4), dtype=np.float32)
        results = [taped.sample(official, logits, 1.0, 4, 0.6) for _ in range(control.DRAW_STRIDE)]

        self.assertEqual(len(mx.array_calls), 1, "the tape must be uploaded once and sliced per draw")
        self.assertEqual(taped.draw_count, control.DRAW_STRIDE)
        self.assertEqual(len(taped.capture_refs), control.DRAW_STRIDE)
        self.assertEqual([entry[1] for entry in taped.capture_refs], [
            "main", "main_retry", *[f"codebook_{index}" for index in range(1, 10)]
        ])
        self.assertTrue(all(int(result[0]) == 0 for result in results))

    def test_inverse_cdf_advances_on_equality_and_clamps_one_to_last_kept(self) -> None:
        logits = np.zeros((1, 4), dtype=np.float32)

        def sample_at(uniform: float) -> int:
            sampler = control.SamplerControl(FakeMx(), [uniform], set())
            result = sampler._shadow_sample(logits, 1.0, 4, 0.6, 0)
            return int(result[0])

        # Top-p keeps the first two equal candidates; their renormalized CDF is [0.5, 1].
        # U=0.5 must advance to candidate 1 on equality, matching native's d < 0 rule.
        self.assertEqual(sample_at(0.0), 0)
        self.assertEqual(sample_at(float(np.nextafter(np.float32(0.5), np.float32(0.0)))), 0)
        self.assertEqual(sample_at(0.5), 1)

        # U=1 would count all four CDF entries; the clamp must return the second retained
        # candidate rather than indexing into one of the two top-p-dropped candidates.
        self.assertEqual(sample_at(1.0), 1)


class OfficialChunkControlSpec(unittest.TestCase):
    def test_chunk_text_is_an_explicit_opt_in(self) -> None:
        required = [
            "--text", "Тише.", "--seed", "1", "--max-frames", "768",
            "--output", "out.wav", "--report", "out.json",
            "--model-dir", "model", "--mlx-root", "mlx-audio",
        ]
        self.assertFalse(control.parse_args(required).chunk_text)
        self.assertTrue(control.parse_args([*required, "--chunk-text"]).chunk_text)

    def test_chunk_capture_steps_use_global_budget_but_single_prompt_stays_bounded(self) -> None:
        required = [
            "--text", "Тише.", "--seed", "1", "--max-frames", "768",
            "--output", "out.wav", "--report", "out.json",
            "--model-dir", "model", "--mlx-root", "mlx-audio",
            "--capture-steps", "768",
        ]
        with contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit):
                control.parse_args(required)
        chunked = control.parse_args([*required, "--chunk-text"])
        self.assertEqual(chunked.capture_steps, (768,))
        control.validate_chunk_capture_steps(chunked.capture_steps, 768, 2)
        with self.assertRaisesRegex(ValueError, "frame budget 1536"):
            control.validate_chunk_capture_steps((1536,), 768, 2)

    def test_chunked_tape_budget_multiplies_per_chunk_cap(self) -> None:
        self.assertEqual(control.chunk_tape_frame_budget(768, 2), 1536)
        with self.assertRaisesRegex(ValueError, "positive"):
            control.chunk_tape_frame_budget(768, 0)

    def test_chunked_tape_loader_rejects_single_chunk_sized_tape(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "short-tape.json"
            path.write_text("[0.5,0.5]", encoding="utf-8")
            with self.assertRaisesRegex(SystemExit, "need at least 66"):
                control.load_random_tape(path, max_frames=2, chunk_count=3)

    def test_zero_history_spans_leave_prompt_unchanged(self) -> None:
        prompt = np.zeros((3, control.NUM_CODEBOOKS + 1), dtype=np.int32)
        prompt[:, 0] = [101, 102, 103]
        spans = control.find_zero_history_spans(prompt, [], semantic_begin=1000)
        rebuilt = control.replace_history_audio_rows(FakeMx(), prompt, spans, [], semantic_begin=1000)
        np.testing.assert_array_equal(rebuilt, prompt)

    def test_history_placeholder_spans_are_replaced_with_gpu_code_rows(self) -> None:
        semantic_begin = 1000
        row = np.zeros((control.NUM_CODEBOOKS + 1,), dtype=np.int32)
        row[0] = semantic_begin
        prompt = np.stack([
            np.r_[101, np.zeros(control.NUM_CODEBOOKS, dtype=np.int32)],
            row,
            row,
            np.r_[202, np.zeros(control.NUM_CODEBOOKS, dtype=np.int32)],
        ])
        spans = control.find_zero_history_spans(prompt, [2], semantic_begin)
        self.assertEqual(spans, [(1, 3)])

        codes = np.arange(control.NUM_CODEBOOKS * 2, dtype=np.uint32).reshape(control.NUM_CODEBOOKS, 2)
        rebuilt = control.replace_history_audio_rows(FakeMx(), prompt, spans, [codes], semantic_begin)
        expected_audio_rows = np.column_stack([
            codes[0].astype(np.int32) + semantic_begin,
            codes.T.astype(np.int32),
        ])
        np.testing.assert_array_equal(rebuilt[1:3], expected_audio_rows)
        np.testing.assert_array_equal(rebuilt[[0, 3]], prompt[[0, 3]])
        self.assertEqual(rebuilt.dtype, np.int32)

    def test_malformed_history_placeholder_spans_fail_closed(self) -> None:
        semantic_begin = 1000
        row = np.zeros((control.NUM_CODEBOOKS + 1,), dtype=np.int32)
        row[0] = semantic_begin
        short_prompt = np.stack([row, np.r_[202, np.zeros(control.NUM_CODEBOOKS, dtype=np.int32)]])
        with self.assertRaisesRegex(ValueError, "placeholder span lengths"):
            control.find_zero_history_spans(short_prompt, [2], semantic_begin)

        extra_prompt = np.stack([
            row,
            np.r_[202, np.zeros(control.NUM_CODEBOOKS, dtype=np.int32)],
            row,
        ])
        with self.assertRaisesRegex(ValueError, "placeholder span count"):
            control.find_zero_history_spans(extra_prompt, [1], semantic_begin)

        with self.assertRaisesRegex(ValueError, "prompt rows"):
            control.find_zero_history_spans(np.zeros((2, 3), dtype=np.int32), [1], semantic_begin)


if __name__ == "__main__":
    unittest.main()
