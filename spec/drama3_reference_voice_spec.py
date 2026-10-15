"""CPU-only validation for the pinned MLX reference-voice encoder fixture."""

from __future__ import annotations

import hashlib
import importlib.util
import tempfile
import unittest
import wave
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "drama3" / "reference_voice_reference.py"
SPEC = importlib.util.spec_from_file_location("drama3_reference_voice_under_test", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
reference = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(reference)


def write_wav(path: Path, *, channels: int = 1, width: int = 2, rate: int = 44_100,
              frames: bytes | None = None) -> bytes:
    if frames is None:
        frames = bytes(([-128, -1, 0, 1, 127] if width == 1 else [0] * 10))
    with wave.open(str(path), "wb") as output:
        output.setnchannels(channels)
        output.setsampwidth(width)
        output.setframerate(rate)
        output.writeframes(frames)
    return path.read_bytes()


class ReferenceVoiceInputSpec(unittest.TestCase):
    def test_pcm16_loader_preserves_input_bytes_and_records_both_hashes(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "reference.wav"
            raw = write_wav(path, frames=np.array([-32768, -1, 0, 16384, 32767], dtype="<i2").tobytes())

            loaded = reference.read_pcm16_wav(path)

        self.assertEqual(loaded["sample_rate"], 44_100)
        self.assertEqual(loaded["channels"], 1)
        self.assertEqual(loaded["sample_width_bytes"], 2)
        self.assertEqual(loaded["samples"], 5)
        self.assertEqual(loaded["pcm_bytes"], np.array([-32768, -1, 0, 16384, 32767], dtype="<i2").tobytes())
        self.assertEqual(loaded["wav_sha256"], hashlib.sha256(raw).hexdigest())
        self.assertEqual(loaded["pcm_sha256"], hashlib.sha256(loaded["pcm_bytes"]).hexdigest())

    def test_pcm16_loader_rejects_stereo_wrong_rate_non_pcm_and_empty_audio(self) -> None:
        cases = (
            ({"channels": 2}, "mono"),
            ({"rate": 24_000}, "44100"),
            ({"width": 3}, "16-bit"),
            ({"frames": b""}, "non-empty"),
        )
        for index, (kwargs, message) in enumerate(cases):
            with self.subTest(kwargs=kwargs), tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / f"bad-{index}.wav"
                write_wav(path, **kwargs)
                with self.assertRaisesRegex(ValueError, message):
                    reference.read_pcm16_wav(path)

    def test_transcript_requires_nonempty_text(self) -> None:
        for value in ("", " \t\n"):
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, "non-empty"):
                reference.validate_transcript(value)

    def test_stage_export_is_time_major_little_endian_float32(self) -> None:
        nch = np.array([[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]], dtype=np.float64)
        values = reference.stage_to_time_major_f32(nch, "nct", "encoder.block.0")
        self.assertEqual(values.shape, (3, 2))
        self.assertEqual(values.dtype, np.dtype("<f4"))
        np.testing.assert_array_equal(values, [[1, 4], [2, 5], [3, 6]])

    def test_stage_export_rejects_nonfinite_and_unsupported_shapes(self) -> None:
        with self.assertRaisesRegex(ValueError, "non-finite"):
            reference.stage_to_time_major_f32(np.array([[np.nan]], dtype=np.float32), "matrix", "bad")
        with self.assertRaisesRegex(ValueError, "layout"):
            reference.stage_to_time_major_f32(np.zeros((1, 2), dtype=np.float32), "unknown", "bad")

    def test_trace_captures_encoder_input_and_output_with_declared_layout(self) -> None:
        class Module:
            def __call__(self, values: np.ndarray) -> np.ndarray:
                return values + 1

        trace = reference._EncoderTrace(mx=None)
        module = Module()
        trace.add_module(
            module,
            "encoder.output",
            input_name="audio.padded",
            input_layout="nct",
        )
        audio = np.zeros((1, 1, 8), dtype=np.float32)
        with trace:
            result = module(audio)

        np.testing.assert_array_equal(trace.values["audio.padded"], audio)
        np.testing.assert_array_equal(trace.values["encoder.output"], result)
        self.assertEqual(trace.layouts, {"audio.padded": "nct", "encoder.output": "nct"})

    def test_stage_file_manifest_hashes_flat_little_endian_payload(self) -> None:
        values = np.array([[1.25, -2.5], [3.75, 4.0]], dtype="<f4")
        with tempfile.TemporaryDirectory() as directory:
            entry = reference._write_stage(
                Path(directory),
                0,
                "encoder.output",
                values,
                np,
            )
            payload = (Path(directory) / entry["file"]).read_bytes()

        self.assertEqual(entry["shape"], [2, 2])
        self.assertEqual(entry["dtype"], "float32-le")
        self.assertEqual(entry["row_axis"], "time")
        self.assertEqual(entry["channel_axis"], "features")
        self.assertEqual(entry["sha256"], hashlib.sha256(payload).hexdigest())
        self.assertEqual(payload, values.tobytes(order="C"))


if __name__ == "__main__":
    unittest.main()
