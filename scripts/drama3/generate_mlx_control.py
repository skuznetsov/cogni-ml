#!/usr/bin/env python3
"""Generate a bounded Fish Omni MLX control with an F32 core.

The backbone, main head, and tower use the repository's shardwise F32 loader;
the pinned MLX-audio sampler, post-load hook, and DAC remain in use. Optional
uniform tapes select tokens with an explicit inverse-CDF shadow sampler. That
path is useful for shared-uniform diagnostics and is not stock MLX RNG parity.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import importlib.metadata
import importlib.util
import json
import math
import os
import platform
import subprocess
import sys
import time
import wave
from pathlib import Path
from typing import Any


PINNED_REVISION = "e7488935ec2ebfafd38ef2b58e290d79f3d05dc7"
MODEL_REPOSITORY = "cguldogan/drama-3-mlx-q4"
MODEL_REVISION = "5331ab509359c0e4a7e353a19e255ddf3e8152a8"
SAMPLE_RATE = 44_100
FRAME_SAMPLES = 2_048
NUM_CODEBOOKS = 10
DRAW_STRIDE = 11  # main, repetition retry, then nine residual codebook heads
MAX_FRAMES_LIMIT = 1_024
TEMPERATURE = 0.8
TOP_K = 30
TOP_P = 0.9
RAS_TEMPERATURE = 1.0
RAS_TOP_K = 30
RAS_TOP_P = 0.95
RAS_WINDOW_FRAMES = 16
STARTED = time.perf_counter()

UPSTREAM_FILES = (
    "mlx_audio/utils.py",
    "mlx_audio/tts/utils.py",
    "mlx_audio/tts/models/base.py",
    "mlx_audio/tts/models/fish_omni/__init__.py",
    "mlx_audio/tts/models/fish_omni/config.py",
    "mlx_audio/tts/models/fish_omni/fish_omni.py",
    "mlx_audio/tts/models/fish_omni/prompt.py",
    "mlx_audio/tts/models/fish_omni/tower.py",
    "mlx_audio/lm/models/qwen3_5.py",
    "mlx_audio/lm/models/activations.py",
    "mlx_audio/lm/models/base.py",
    "mlx_audio/lm/models/cache.py",
    "mlx_audio/lm/models/gated_delta.py",
    "mlx_audio/lm/models/rope_utils.py",
    "mlx_audio/lm/models/switch_layers.py",
    "mlx_audio/codec/models/fish_s1_dac/__init__.py",
    "mlx_audio/codec/models/fish_s1_dac/fish_s1_dac.py",
)


def parse_capture_steps(value: str) -> tuple[int, ...]:
    """Parse comma-separated, zero-based generation steps."""
    if not value.strip():
        return ()
    try:
        steps = tuple(int(piece.strip(), 10) for piece in value.split(","))
    except ValueError as error:
        raise argparse.ArgumentTypeError("capture steps must be comma-separated integers") from error
    if any(step < 0 for step in steps):
        raise argparse.ArgumentTypeError("capture steps must be non-negative")
    if len(set(steps)) != len(steps):
        raise argparse.ArgumentTypeError("capture steps must not repeat")
    return tuple(sorted(steps))


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--text", required=True, help="Exact text to synthesize")
    parser.add_argument("--seed", required=True, type=int, help="MLX seed for stock-sampler runs")
    parser.add_argument("--max-frames", required=True, type=int, help="Maximum generation iterations")
    parser.add_argument(
        "--speaker-id",
        type=int,
        choices=range(16),
        help="Optional official prompt tag 0..15; a tag alone does not supply a reference voice",
    )
    parser.add_argument("--output", required=True, type=Path, help="New WAV output path")
    parser.add_argument("--report", required=True, type=Path, help="New JSON report path")
    parser.add_argument("--model-dir", required=True, type=Path, help="Local converted MLX checkpoint")
    parser.add_argument("--mlx-root", required=True, type=Path, help="Pinned mlx-audio checkout")
    parser.add_argument(
        "--replay-script",
        type=Path,
        default=Path(__file__).resolve().with_name("replay_mlx_generation.py"),
        help="F32 shard loader and source utilities (defaults to the sibling script)",
    )
    parser.add_argument(
        "--random-tape",
        type=Path,
        help="JSON array of shared uniforms in [0,1], one per sampler call",
    )
    parser.add_argument(
        "--capture-steps",
        type=parse_capture_steps,
        default=(),
        metavar="N[,N...]",
        help="Capture sampler-input logits for selected zero-based steps; saved beside the report",
    )
    parser.add_argument(
        "--chunk-text",
        action="store_true",
        help="Use official Model.generate text splitting and cross-chunk history (300-byte chunks)",
    )
    parser.add_argument(
        "--reference-audio",
        type=Path,
        help="One mono PCM16 44.1 kHz WAV whose voice and transcript condition generation",
    )
    parser.add_argument(
        "--reference-text",
        help="Exact transcript matching --reference-audio",
    )
    args = parser.parse_args(argv)
    if args.max_frames < 1:
        parser.error("--max-frames must be at least 1")
    if args.max_frames > MAX_FRAMES_LIMIT:
        parser.error(f"--max-frames must be at most {MAX_FRAMES_LIMIT}")
    if args.seed < 0:
        parser.error("--seed must be non-negative")
    if not args.chunk_text and any(step >= args.max_frames for step in args.capture_steps):
        parser.error("every --capture-steps value must be less than --max-frames")
    if (args.reference_audio is None) != (args.reference_text is None):
        parser.error("--reference-audio and --reference-text must be supplied together")
    if args.reference_audio is not None and args.chunk_text:
        parser.error("reference audio is not supported with --chunk-text")
    if args.reference_audio is not None and args.speaker_id not in (None, 0):
        parser.error("reference voice conditioning currently supports speaker 0 only")
    if args.reference_text is not None and not args.reference_text.strip():
        parser.error("--reference-text must be non-empty")
    return args


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def progress(stage: str, **details: Any) -> None:
    print(
        json.dumps(
            {"event": "progress", "stage": stage,
             "elapsed_seconds": round(time.perf_counter() - STARTED, 3), **details},
            ensure_ascii=False,
        ),
        file=sys.stderr,
        flush=True,
    )


def git_metadata(root: Path) -> dict[str, Any]:
    try:
        revision = subprocess.check_output(
            ["git", "-C", str(root), "rev-parse", "HEAD"], text=True, stderr=subprocess.PIPE
        ).strip()
        status = subprocess.check_output(
            ["git", "-C", str(root), "status", "--porcelain"], text=True, stderr=subprocess.PIPE
        )
    except (OSError, subprocess.CalledProcessError) as error:
        raise SystemExit(f"cannot identify pinned MLX-audio checkout: {error}") from error
    top = subprocess.check_output(
        ["git", "-C", str(root), "rev-parse", "--show-toplevel"],
        text=True,
        stderr=subprocess.PIPE,
    ).strip()
    if Path(top).resolve() != root.resolve():
        raise SystemExit(f"--mlx-root must be the git root: {root}")
    if revision != PINNED_REVISION:
        raise SystemExit(f"MLX-audio revision mismatch: expected {PINNED_REVISION}, got {revision}")
    return {"repository": str(root), "revision": revision, "worktree_clean": not bool(status.strip())}


def chunk_tape_frame_budget(max_frames: int, chunk_count: int) -> int:
    if (
        isinstance(max_frames, bool)
        or not isinstance(max_frames, int)
        or max_frames < 1
        or isinstance(chunk_count, bool)
        or not isinstance(chunk_count, int)
        or chunk_count < 1
    ):
        raise ValueError("max_frames and chunk_count must be positive integers")
    return max_frames * chunk_count


def validate_chunk_capture_steps(
    capture_steps: tuple[int, ...] | list[int], max_frames: int, chunk_count: int
) -> None:
    frame_budget = chunk_tape_frame_budget(max_frames, chunk_count)
    if any(step >= frame_budget for step in capture_steps):
        raise ValueError(
            f"every --capture-steps value must be less than the chunked frame budget {frame_budget}"
        )


def load_random_tape(
    path: Path, max_frames: int, chunk_count: int = 1
) -> tuple[list[float], bytes]:
    raw = path.read_bytes()
    try:
        value = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise SystemExit(f"invalid random-tape JSON at {path}: {error}") from error
    if isinstance(value, dict):
        value = value.get("uniforms")
    if not isinstance(value, list):
        raise SystemExit("random tape must be a JSON array or an object with a uniforms array")
    frame_budget = chunk_tape_frame_budget(max_frames, chunk_count)
    required_draws = frame_budget * DRAW_STRIDE
    if len(value) < required_draws:
        chunk_scope = (
            f" across {chunk_count} text chunks"
            if chunk_count > 1
            else ""
        )
        raise SystemExit(
            f"random tape has {len(value)} uniforms; need at least "
            f"{required_draws} for --max-frames {max_frames}{chunk_scope}"
        )
    uniforms: list[float] = []
    for index, item in enumerate(value):
        if isinstance(item, bool) or not isinstance(item, (int, float)):
            raise SystemExit(f"random tape uniforms[{index}] must be numeric")
        uniform = float(item)
        if not math.isfinite(uniform) or not 0.0 <= uniform <= 1.0:
            raise SystemExit(f"random tape uniforms[{index}] must be finite and in [0,1]")
        uniforms.append(uniform)
    return uniforms, raw


def find_zero_history_spans(
    prompt: Any,
    frame_counts: list[int] | tuple[int, ...],
    semantic_begin: int,
    num_codebooks: int = NUM_CODEBOOKS,
) -> list[tuple[int, int]]:
    """Find exact contiguous zero-code history rows in an official host prompt."""
    import numpy as np

    rows = np.asarray(prompt)
    expected_width = num_codebooks + 1
    if rows.ndim != 2 or rows.shape[1] != expected_width:
        raise ValueError(f"prompt rows must have shape [rows,{expected_width}], got {rows.shape}")
    counts = tuple(frame_counts)
    if any(
        isinstance(count, bool) or not isinstance(count, (int, np.integer)) or count < 1
        for count in counts
    ):
        raise ValueError("history placeholder frame counts must be positive integers")
    if not counts:
        return []

    placeholder = np.zeros((expected_width,), dtype=rows.dtype)
    placeholder[0] = semantic_begin
    marker_rows = np.all(rows == placeholder[None, :], axis=1)
    runs: list[tuple[int, int]] = []
    index = 0
    while index < len(marker_rows):
        if not marker_rows[index]:
            index += 1
            continue
        start = index
        while index < len(marker_rows) and marker_rows[index]:
            index += 1
        runs.append((start, index))

    if len(runs) != len(counts):
        raise ValueError(
            f"history placeholder span count mismatch: expected {len(counts)}, found {len(runs)}"
        )
    actual_counts = tuple(end - start for start, end in runs)
    if actual_counts != counts:
        raise ValueError(
            f"history placeholder span lengths mismatch: expected {counts}, found {actual_counts}"
        )
    return runs


def replace_history_audio_rows(
    mx: Any,
    prompt: Any,
    spans: list[tuple[int, int]],
    history_codes: list[Any] | tuple[Any, ...],
    semantic_begin: int,
    num_codebooks: int = NUM_CODEBOOKS,
) -> Any:
    """Build a GPU prompt with MLX code rows spliced into zero-history spans."""
    import numpy as np

    rows = np.asarray(prompt)
    expected_width = num_codebooks + 1
    if rows.ndim != 2 or rows.shape[1] != expected_width:
        raise ValueError(f"prompt rows must have shape [rows,{expected_width}], got {rows.shape}")
    if len(spans) != len(history_codes):
        raise ValueError("history placeholder span count does not match history code tensors")
    if not spans:
        return mx.array(rows, dtype=mx.int32)

    pieces: list[Any] = []
    cursor = 0
    for (start, end), codes in zip(spans, history_codes):
        if start < cursor or start < 0 or end <= start or end > len(rows):
            raise ValueError(f"malformed history placeholder span {(start, end)}")
        if tuple(codes.shape) != (num_codebooks, end - start):
            raise ValueError(
                f"history codes shape {tuple(codes.shape)} does not match span {(start, end)}"
            )
        if start > cursor:
            pieces.append(mx.array(rows[cursor:start], dtype=mx.int32))
        codes_i32 = codes.astype(mx.int32)
        semantic = codes_i32[0:1, :].T + int(semantic_begin)
        audio_rows = mx.concatenate([semantic, codes_i32.T], axis=1)
        pieces.append(audio_rows)
        cursor = end
    if cursor < len(rows):
        pieces.append(mx.array(rows[cursor:], dtype=mx.int32))
    return pieces[0] if len(pieces) == 1 else mx.concatenate(pieces, axis=0)


def build_reference_prompt_gpu(
    mx: Any,
    np: Any,
    official_build_prompt: Any,
    *,
    text: str,
    tokenizer: Any,
    codebook_size: int,
    semantic_begin: int,
    reference_text: str,
    reference_codes: Any,
) -> tuple[Any, Any, list[tuple[int, int]]]:
    """Build the official prompt skeleton and splice reference codes while on GPU."""
    shape = tuple(reference_codes.shape)
    if len(shape) != 2 or shape[0] != NUM_CODEBOOKS or shape[1] < 1:
        raise ValueError(f"reference codes must have shape [10,frames>=1], got {shape}")
    frame_count = int(shape[1])
    skeleton_codes = np.zeros((NUM_CODEBOOKS, frame_count), dtype=np.int64)
    skeleton = official_build_prompt(
        text,
        tokenizer,
        num_codebooks=NUM_CODEBOOKS,
        codebook_size=codebook_size,
        semantic_start=semantic_begin,
        prompt_texts=[reference_text],
        prompt_tokens=[skeleton_codes],
    )
    spans = find_zero_history_spans(skeleton, [frame_count], semantic_begin)
    prompt = replace_history_audio_rows(
        mx,
        skeleton,
        spans,
        [reference_codes],
        semantic_begin,
    )
    return prompt, skeleton, spans


def verify_reference_prompt(
    np: Any,
    official_build_prompt: Any,
    *,
    actual_prompt: Any,
    text: str,
    tokenizer: Any,
    codebook_size: int,
    semantic_begin: int,
    reference_text: str,
    reference_codes: Any,
) -> Any:
    """Rebuild the full official prompt at terminal readback and compare rows."""
    official_prompt = official_build_prompt(
        text,
        tokenizer,
        num_codebooks=NUM_CODEBOOKS,
        codebook_size=codebook_size,
        semantic_start=semantic_begin,
        prompt_texts=[reference_text],
        prompt_tokens=[np.asarray(reference_codes, dtype=np.int64)],
    )
    actual = np.asarray(actual_prompt)
    expected = np.asarray(official_prompt)
    if (
        actual.dtype != expected.dtype
        or actual.shape != expected.shape
        or actual.tobytes(order="C") != expected.tobytes(order="C")
    ):
        raise ValueError("GPU reference prompt differs from official builder with encoded reference codes")
    return official_prompt


def read_reference_pcm16_wav(path: Path) -> dict[str, Any]:
    """Validate a single 44.1 kHz mono PCM16 reference before loading MLX."""
    import wave
    try:
        with wave.open(str(path), "rb") as source:
            channels = source.getnchannels()
            sample_width = source.getsampwidth()
            sample_rate = source.getframerate()
            samples = source.getnframes()
            compression = source.getcomptype()
            pcm_bytes = source.readframes(samples)
    except (OSError, wave.Error) as error:
        raise ValueError(f"cannot read reference WAV: {error}") from error
    if compression != "NONE":
        raise ValueError("reference WAV must use uncompressed PCM")
    if channels != 1:
        raise ValueError(f"reference WAV must be mono, got {channels} channels")
    if sample_width != 2:
        raise ValueError(f"reference WAV must be PCM 16-bit, got {sample_width * 8}-bit")
    if sample_rate != SAMPLE_RATE:
        raise ValueError(f"reference WAV sample rate must be 44100 Hz, got {sample_rate}")
    if samples < 1 or not pcm_bytes:
        raise ValueError("reference WAV must contain non-empty audio")
    if samples > SAMPLE_RATE * 10:
        raise ValueError("reference WAV must be no longer than 10 seconds")
    if len(pcm_bytes) != samples * 2:
        raise ValueError("reference WAV PCM byte count does not match its frame count")
    return {
        "path": path.resolve(),
        "samples": samples,
        "pcm_bytes": pcm_bytes,
        "wav_sha256": sha256_file(path),
        "pcm_sha256": sha256_bytes(pcm_bytes),
    }


def pcm16_bytes(audio: Any, np: Any) -> bytes:
    samples = np.asarray(audio)
    if samples.ndim != 1 or samples.dtype != np.dtype("float32"):
        raise SystemExit(f"decoded audio must be float32 mono, got {samples.dtype} {samples.shape}")
    if not np.isfinite(samples).all():
        raise SystemExit("decoded audio contains non-finite samples")
    clipped = np.clip(samples, np.float32(-1.0), np.float32(1.0)).astype(np.float32, copy=False)
    scaled = np.multiply(clipped, np.float32(32767.0), dtype=np.float32)
    rounded = np.where(
        scaled >= np.float32(0.0),
        np.floor(scaled + np.float32(0.5)),
        np.ceil(scaled - np.float32(0.5)),
    )
    return rounded.astype("<i2").tobytes(order="C")


def make_wav_bytes(pcm: bytes) -> bytes:
    import io

    if not pcm or len(pcm) % 2:
        raise SystemExit("PCM16 output must contain at least one complete sample")
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(SAMPLE_RATE)
        wav.writeframes(pcm)
    return buffer.getvalue()


def atomic_create(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise SystemExit(f"refusing to overwrite existing output: {path}")
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, path)
    except FileExistsError as error:
        raise SystemExit(f"refusing to overwrite existing output: {path}") from error
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


class SamplerControl:
    """Capture logits and optionally substitute shared uniforms for MLX draws."""

    def __init__(self, mx: Any, uniforms: list[float] | None, capture_steps: set[int]):
        self.mx = mx
        self.uniform_count = 0 if uniforms is None else len(uniforms)
        self.uniform_tape = (
            None if uniforms is None else mx.array(uniforms, dtype=mx.float32)
        )
        self.capture_steps = capture_steps
        self.draw_count = 0
        self.frame_count = 0
        self.last_main = None
        self.capture_refs: list[tuple[int, str, Any]] = []
        self.hidden_capture_refs: list[tuple[int, Any]] = []

    def _label(self, slot: int) -> str:
        if slot == 0:
            return "main"
        if slot == 1:
            return "main_retry"
        return f"codebook_{slot - 1}"

    def _shadow_sample(
        self, logits: Any, temperature: float, top_k: int, top_p: float, draw: int
    ) -> Any:
        mx = self.mx
        if top_k < 1:
            raise ValueError(f"top_k must be at least 1; got {top_k}")
        scaled = logits.astype(mx.float32) / max(float(temperature), 1e-5)
        k = min(int(top_k), scaled.shape[-1])
        ids = mx.argpartition(-scaled, kth=k - 1, axis=-1)[..., :k]
        values = mx.take_along_axis(scaled, ids, axis=-1)
        order = mx.argsort(-values, axis=-1)
        ids = mx.take_along_axis(ids, order, axis=-1)
        values = mx.take_along_axis(values, order, axis=-1)
        full_log_norm = mx.logsumexp(scaled, axis=-1, keepdims=True)
        full_cumulative = mx.cumsum(mx.exp(values - full_log_norm), axis=-1)
        dropped = full_cumulative > float(top_p)
        dropped[..., 0] = False
        masked = mx.where(dropped, -mx.inf, values)
        probabilities = mx.softmax(masked, axis=-1)
        cumulative = mx.cumsum(probabilities, axis=-1)
        u = mx.reshape(self.uniform_tape[draw : draw + 1], (1, 1))
        kept_count = mx.sum(~dropped, axis=-1)
        choice = mx.minimum(
            mx.sum(cumulative <= u, axis=-1), kept_count - 1
        ).astype(mx.int32)
        return mx.take_along_axis(ids, choice[..., None], axis=-1).squeeze(-1)

    def sample(self, official: Any, logits: Any, temperature: float, top_k: int, top_p: float) -> Any:
        draw = self.draw_count
        step, slot = divmod(draw, DRAW_STRIDE)
        if step in self.capture_steps:
            self.capture_refs.append((step, self._label(slot), logits))
        self.draw_count += 1
        if self.uniform_tape is None:
            return official(logits, temperature, top_k, top_p)
        return self._shadow_sample(logits, temperature, top_k, top_p, draw)

    def track_main(self, original: Any, *args: Any, **kwargs: Any) -> Any:
        result = original(*args, **kwargs)
        self.last_main = result
        self.frame_count += 1
        return result

    def capture_main_hidden(self, step: int, hidden: Any) -> None:
        if step in self.capture_steps:
            self.hidden_capture_refs.append((step, hidden))


class ChunkPromptAdapter:
    """Keep official chunk/history orchestration while avoiding history readback in-loop."""

    def __init__(
        self,
        mx: Any,
        np: Any,
        fish_omni: Any,
        official_build_prompt: Any,
        semantic_begin: int,
    ) -> None:
        self.mx = mx
        self.np = np
        self.fish_omni = fish_omni
        self.official_build_prompt = official_build_prompt
        self.semantic_begin = int(semantic_begin)
        self.original_module_builder = fish_omni.build_prompt
        self.records: list[dict[str, Any]] = []

    def install(self) -> None:
        self.fish_omni.build_prompt = self._build_prompt

    def restore(self) -> None:
        self.fish_omni.build_prompt = self.original_module_builder

    def _build_prompt(
        self,
        text: str,
        tokenizer: Any,
        *,
        num_codebooks: int,
        codebook_size: int,
        semantic_start: int,
        prompt_texts: Any = (),
        prompt_tokens: Any = (),
        history: Any = (),
    ) -> Any:
        if int(semantic_start) != self.semantic_begin:
            raise ValueError("semantic start changed during chunked prompt construction")
        history_turns = list(history)
        history_codes: list[Any] = []
        frame_counts: list[int] = []
        placeholder_history: list[tuple[str, Any]] = []
        for turn_text, codes in history_turns:
            shape = tuple(codes.shape)
            if len(shape) != 2 or shape[0] != num_codebooks or shape[1] < 1:
                raise ValueError(f"invalid history codes shape without readback: {shape}")
            history_codes.append(codes)
            frame_counts.append(int(shape[1]))
            placeholder_history.append(
                (turn_text, self.np.zeros(shape, dtype=self.np.int64))
            )

        started = time.perf_counter()
        host_prompt = self.official_build_prompt(
            text,
            tokenizer,
            num_codebooks=num_codebooks,
            codebook_size=codebook_size,
            semantic_start=semantic_start,
            prompt_texts=prompt_texts,
            prompt_tokens=prompt_tokens,
            history=placeholder_history,
        )
        spans = find_zero_history_spans(host_prompt, frame_counts, semantic_start, num_codebooks)
        gpu_prompt = replace_history_audio_rows(
            self.mx,
            host_prompt,
            spans,
            history_codes,
            semantic_start,
            num_codebooks,
        )
        self.records.append(
            {
                "text": text,
                "prompt_texts": tuple(prompt_texts),
                "prompt_tokens": tuple(prompt_tokens),
                "history_turns": history_turns,
                "placeholder_spans": spans,
                "prompt_ref": gpu_prompt,
                "prompt_build_seconds": time.perf_counter() - started,
            }
        )
        return gpu_prompt


class FixedHistoryCandidates:
    """Supply the exact selected history to the official terminal prompt rebuild."""

    def __init__(self, turns: list[tuple[str, Any]]) -> None:
        self.turns = turns

    def candidates(self):
        yield self.turns


def run_chunk_text_control(
    args: argparse.Namespace,
    *,
    model: Any,
    mx: Any,
    np: Any,
    mlx: Any,
    fish_omni: Any,
    tower_module: Any,
    official_build_prompt: Any,
    sampler: SamplerControl,
    chunk_texts: list[str],
    generation_text: str,
    output_path: Path,
    report_path: Path,
    capture_path: Path,
    uniforms: list[float] | None,
    random_tape_bytes: bytes | None,
    upstream: dict[str, Any],
    source_hashes: dict[str, str],
    replay_path: Path,
    replay_sha: str,
    replay_revision: str,
    model_dir: Path,
    config_bytes: bytes,
    index_bytes: bytes,
    indexed_shards: list[str],
    loaded_core_tensors: int,
    previous_cache_limit: int,
    load_timings: dict[str, float],
) -> int:
    """Run the opt-in official high-level chunk/history path with GPU history splicing."""
    chunk_output_paths = [
        output_path.with_name(
            f"{output_path.stem}.chunk-{index + 1:03d}{output_path.suffix or '.wav'}"
        )
        for index in range(len(chunk_texts))
    ]
    all_output_paths = [output_path, report_path, *chunk_output_paths]
    if args.capture_steps:
        all_output_paths.append(capture_path)
    if len(set(all_output_paths)) != len(all_output_paths):
        raise SystemExit("chunk WAV, combined WAV, report, and capture paths must differ")
    for path in all_output_paths:
        if path.exists():
            raise SystemExit(f"refusing to overwrite existing output: {path}")

    adapter = ChunkPromptAdapter(
        mx, np, fish_omni, official_build_prompt, int(model.semantic_begin)
    )
    original_fish_sampler = fish_omni.sample_top_k_top_p
    original_tower_sampler = tower_module.sample_top_k_top_p
    original_sample_main = model.sample_main
    original_generate_codes = model.generate_codes
    original_decode_codes = model.decode_codes
    chunk_call_index = 0
    decode_call_index = 0

    def controlled_fish_sampler(logits: Any, temperature: float, top_k: int, top_p: float) -> Any:
        return sampler.sample(original_fish_sampler, logits, temperature, top_k, top_p)

    def controlled_tower_sampler(logits: Any, temperature: float, top_k: int, top_p: float) -> Any:
        return sampler.sample(original_tower_sampler, logits, temperature, top_k, top_p)

    def tracked_sample_main(*call_args: Any, **call_kwargs: Any) -> Any:
        return sampler.track_main(original_sample_main, *call_args, **call_kwargs)

    def tracked_generate_codes(prompt: Any, *call_args: Any, **call_kwargs: Any) -> Any:
        nonlocal chunk_call_index
        if chunk_call_index >= len(adapter.records):
            raise RuntimeError("official generation requested codes without a recorded chunk prompt")
        record = adapter.records[chunk_call_index]
        chunk_call_index += 1
        record["frame_start"] = sampler.frame_count
        record["draw_start"] = sampler.draw_count
        started = time.perf_counter()
        codes = original_generate_codes(prompt, *call_args, **call_kwargs)
        record["codes_ref"] = codes
        record["frame_end"] = sampler.frame_count
        record["draw_end"] = sampler.draw_count
        record["last_main_ref"] = sampler.last_main
        record["generation_seconds"] = time.perf_counter() - started
        return codes

    def tracked_decode_codes(codes: Any) -> Any:
        nonlocal decode_call_index
        while (
            decode_call_index < len(adapter.records)
            and adapter.records[decode_call_index].get("codes_ref") is None
        ):
            decode_call_index += 1
        if decode_call_index >= len(adapter.records):
            raise RuntimeError("official decoder was called without a successful chunk")
        record = adapter.records[decode_call_index]
        decode_call_index += 1
        started = time.perf_counter()
        audio = original_decode_codes(codes)
        record["audio_ref"] = audio
        record["decode_seconds"] = time.perf_counter() - started
        return audio

    adapter.install()
    fish_omni.sample_top_k_top_p = controlled_fish_sampler
    tower_module.sample_top_k_top_p = controlled_tower_sampler
    model.sample_main = tracked_sample_main
    model.generate_codes = tracked_generate_codes
    model.decode_codes = tracked_decode_codes
    mx.random.seed(args.seed)
    generation_started = time.perf_counter()
    progress(
        "official_chunk_generation_started",
        chunk_count=len(chunk_texts),
        max_frames_per_chunk=args.max_frames,
        sampler="shared_uniform_inverse_cdf_shadow" if uniforms is not None else "official_mlx_categorical",
    )
    try:
        results = list(
            model.generate(
                generation_text,
                max_tokens=args.max_frames,
                temperature=TEMPERATURE,
                top_k=TOP_K,
                top_p=TOP_P,
                chunk_length=fish_omni.DRAMA3_CHUNK_BYTES,
                single_chunk=False,
            )
        )
    finally:
        adapter.restore()
        fish_omni.sample_top_k_top_p = original_fish_sampler
        tower_module.sample_top_k_top_p = original_tower_sampler
        model.sample_main = original_sample_main
        model.generate_codes = original_generate_codes
        model.decode_codes = original_decode_codes
    generation_seconds = time.perf_counter() - generation_started

    if [record["text"] for record in adapter.records] != chunk_texts:
        raise SystemExit("official Model.generate chunk sequence differed from the pinned splitter")
    if chunk_call_index != len(chunk_texts):
        raise SystemExit(
            f"official Model.generate made {chunk_call_index} chunk calls; expected {len(chunk_texts)}"
        )
    successful_records = [record for record in adapter.records if record.get("codes_ref") is not None]
    if len(results) != len(successful_records):
        raise SystemExit("official Model.generate results do not match successful chunk code outputs")
    for result, record in zip(results, successful_records):
        if result.text != record["text"]:
            raise SystemExit("official Model.generate result text did not match its prompt chunk")
    if sampler.draw_count != sampler.frame_count * DRAW_STRIDE:
        raise SystemExit(
            f"Fish Omni sampler-call count {sampler.draw_count} did not equal "
            f"{sampler.frame_count} frames * {DRAW_STRIDE} draws"
        )

    terminal_eval_started = time.perf_counter()
    eval_values: list[Any] = []
    for record in adapter.records:
        eval_values.append(record["prompt_ref"])
        for key in ("codes_ref", "audio_ref", "last_main_ref"):
            if record.get(key) is not None:
                eval_values.append(record[key])
    eval_values.extend(entry[2] for entry in sampler.capture_refs)
    if eval_values:
        mx.eval(*eval_values)
    terminal_eval_seconds = time.perf_counter() - terminal_eval_started

    terminal_readback_started = time.perf_counter()
    eos_id = int(model.eos_id)
    semantic_begin = int(model.semantic_begin)
    chunk_reports: list[dict[str, Any]] = []
    all_audio: list[Any] = []
    capture_arrays: dict[str, Any] = {}
    capture_manifest: list[dict[str, Any]] = []

    for index, record in enumerate(adapter.records):
        prompt_np = np.asarray(record["prompt_ref"])
        official_prompt = model.build_chunk_prompt(
            record["text"],
            record["prompt_texts"],
            record["prompt_tokens"],
            FixedHistoryCandidates(record["history_turns"]),
        )
        official_prompt_np = np.asarray(official_prompt)
        prompt_equal = (
            prompt_np.dtype == official_prompt_np.dtype
            and prompt_np.shape == official_prompt_np.shape
            and prompt_np.tobytes(order="C") == official_prompt_np.tobytes(order="C")
        )
        if not prompt_equal:
            raise SystemExit(
                f"GPU history prompt adapter differed from official prompt for chunk {index + 1}"
            )

        codes_ref = record.get("codes_ref")
        if codes_ref is None:
            codes_np = np.empty((NUM_CODEBOOKS, 0), dtype=np.int32)
            codes_dtype = None
        else:
            codes_dtype = str(codes_ref.dtype)
            codes_np = np.asarray(codes_ref)
        if codes_np.ndim != 2 or codes_np.shape[0] != NUM_CODEBOOKS:
            raise SystemExit(f"chunk {index + 1} returned invalid code shape {codes_np.shape}")
        if codes_np.size:
            codebook_size = int(model.config.audio_decoder_config.vocab_size)
            if not np.issubdtype(codes_np.dtype, np.integer):
                raise SystemExit(f"chunk {index + 1} codes are not integer: {codes_np.dtype}")
            if int(codes_np.min()) < 0 or int(codes_np.max()) >= codebook_size:
                raise SystemExit(f"chunk {index + 1} returned out-of-range code IDs")

        audio_ref = record.get("audio_ref")
        if audio_ref is None:
            audio_np = np.empty((0,), dtype=np.float32)
            audio_dtype = None
        else:
            audio_dtype = str(audio_ref.dtype)
            audio_np = np.asarray(audio_ref)
            if audio_np.ndim != 1 or audio_np.dtype != np.dtype("float32"):
                raise SystemExit(
                    f"chunk {index + 1} audio must be float32 mono, got {audio_np.dtype} {audio_np.shape}"
                )
            all_audio.append(audio_np)
        record["audio_np"] = audio_np

        iterations = int(record["frame_end"] - record["frame_start"])
        if iterations > args.max_frames:
            raise SystemExit(f"chunk {index + 1} exceeded its per-chunk frame cap")
        last_main_ref = record.get("last_main_ref")
        last_main = (
            int(np.asarray(last_main_ref).reshape(-1)[0])
            if last_main_ref is not None
            else None
        )
        hit_eos = last_main == eos_id
        returned_frames = int(codes_np.shape[1])
        status = "eos" if hit_eos else (
            "max_frames" if iterations == args.max_frames else "returned_before_cap_unknown"
        )
        frames = [
            [
                semantic_begin + int(codes_np[0, frame]),
                *[int(codes_np[book, frame]) for book in range(NUM_CODEBOOKS)],
            ]
            for frame in range(returned_frames)
        ]
        chunk_wav_bytes = b""
        chunk_wav_sha = None
        chunk_wav_path = None
        if audio_np.size:
            pcm = pcm16_bytes(audio_np, np)
            chunk_wav_bytes = make_wav_bytes(pcm)
            chunk_wav_sha = sha256_bytes(chunk_wav_bytes)
            chunk_wav_path = chunk_output_paths[index]
        record["chunk_wav_bytes"] = chunk_wav_bytes

        chunk_reports.append(
            {
                "index": index,
                "text": record["text"],
                "text_utf8_sha256": sha256_bytes(record["text"].encode("utf-8")),
                "prompt_shape": list(prompt_np.shape),
                "prompt_ids": [int(value) for value in prompt_np[:, 0].tolist()],
                "prompt_sha256": sha256_bytes(prompt_np.tobytes(order="C")),
                "official_prompt_byte_equal": prompt_equal,
                "prompt_build_seconds": record["prompt_build_seconds"],
                "generation_seconds": record["generation_seconds"],
                "decode_seconds": record.get("decode_seconds", 0.0),
                "generation_iterations": iterations,
                "returned_frames": returned_frames,
                "eos_detected": hit_eos,
                "last_sampled_main_token": last_main,
                "max_frames_per_chunk": args.max_frames,
                "frame_cap_reached": iterations == args.max_frames and not hit_eos,
                "termination_status": status,
                "draw_interval": {
                    "start_inclusive": record["draw_start"],
                    "end_exclusive": record["draw_end"],
                    "draw_count": record["draw_end"] - record["draw_start"],
                },
                "history_turn_count": len(record["history_turns"]),
                "output": {
                    "wav_path": str(chunk_wav_path) if chunk_wav_path is not None else None,
                    "wav_sha256": chunk_wav_sha,
                    "frame_shape": [returned_frames, NUM_CODEBOOKS + 1],
                    "frames": frames,
                    "codes_shape": [NUM_CODEBOOKS, returned_frames],
                    "codebook_dtype": codes_dtype,
                    "decoded_audio_dtype": audio_dtype,
                    "samples": int(audio_np.size),
                    "seconds": float(audio_np.size / SAMPLE_RATE),
                },
            }
        )

    terminal_readback_seconds = time.perf_counter() - terminal_readback_started
    combined_audio = (
        np.concatenate(all_audio).astype(np.float32, copy=False)
        if all_audio
        else np.empty((0,), dtype=np.float32)
    )
    combined_wav_bytes = (
        make_wav_bytes(pcm16_bytes(combined_audio, np)) if combined_audio.size else b""
    )

    capture_file_sha = None
    if sampler.capture_refs:
        import io

        for capture_step, label, logits in sampler.capture_refs:
            key = f"step_{capture_step:05d}_{label}"
            values = np.asarray(logits, dtype=np.float32)
            if values.ndim != 2 or values.shape[0] != 1 or not np.isfinite(values).all():
                raise SystemExit(f"captured sampler logits {key} have invalid values or shape {values.shape}")
            capture_arrays[key] = values
            capture_manifest.append(
                {"key": key, "step": capture_step, "sampler": label,
                 "shape": list(values.shape), "dtype": "float32"}
            )
        buffer = io.BytesIO()
        np.savez_compressed(buffer, **capture_arrays)
        capture_bytes = buffer.getvalue()
        atomic_create(capture_path, capture_bytes)
        capture_file_sha = sha256_bytes(capture_bytes)

    for report_entry, chunk_path in zip(chunk_reports, chunk_output_paths):
        wav_path = report_entry["output"]["wav_path"]
        if wav_path is not None:
            retained_wav = adapter.records[report_entry["index"]]["chunk_wav_bytes"]
            atomic_create(chunk_path, retained_wav)
    if combined_wav_bytes:
        atomic_create(output_path, combined_wav_bytes)

    try:
        mlx_version = importlib.metadata.version("mlx")
    except importlib.metadata.PackageNotFoundError:
        mlx_version = getattr(mlx, "__version__", "unknown")
    source_hashes = dict(source_hashes)
    source_hashes["scripts/drama3/replay_mlx_generation.py"] = replay_sha
    source_hashes["scripts/drama3/generate_mlx_control.py"] = sha256_file(Path(__file__).resolve())

    report: dict[str, Any] = {
        "format": "drama3-mlx-chunk-control-v1",
        "claim_scope": "pinned Fish Omni F32-core control through official Model.generate text splitting and history orchestration; per-chunk shared-uniform sampling is an inverse-CDF shadow, not native RNG parity or a voice-identity claim",
        "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "upstream": upstream,
        "source_hashes": source_hashes,
        "runtime": {
            "python": sys.version,
            "python_executable": sys.executable,
            "platform": platform.platform(),
            "mlx_version": mlx_version,
            "replay_loader_path": str(replay_path),
            "replay_loader_sha256": replay_sha,
            "replay_loader_git_revision": replay_revision,
            "default_device": str(mx.default_device()),
            "hub_network_policy": {"HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1",
                                   "HF_HUB_DISABLE_TELEMETRY": "1"},
            "allocator_cache_limit_bytes": 0,
            "previous_allocator_cache_limit_bytes": previous_cache_limit,
            "prefill_chunk_size": 1,
            "host_readback_scope": "official per-frame EOS/decode checks plus terminal prompt/code/audio/logit conversion and exact official prompt verification; history code values are not read back while constructing chunk prompts",
        },
        "model": {
            "repository": MODEL_REPOSITORY,
            "revision": MODEL_REVISION,
            "path": str(model_dir),
            "model_type": model.model_type,
            "config_sha256": sha256_bytes(config_bytes),
            "index_sha256": sha256_bytes(index_bytes),
            "indexed_shards": indexed_shards,
            "core_loader": {
                "kind": "replay_mlx_generation.py shardwise loader",
                "floating_checkpoint_tensors": "promoted to F32 before mx.array",
                "packed_integer_tensors": "preserved at checkpoint integer dtype",
                "quantization_layout": "official apply_quantization before shardwise load",
                "loaded_core_tensor_count": loaded_core_tensors,
                "official_post_load_hook": True,
                "codec": "official Fish S1 DAC loader, checkpoint dtype",
            },
            "sample_rate": SAMPLE_RATE,
            "num_codebooks": NUM_CODEBOOKS,
            "codebook_size": int(model.config.audio_decoder_config.vocab_size),
            "semantic_begin": int(model.semantic_begin),
            "eos_id": int(model.eos_id),
        },
        "input": {
            "text": args.text,
            "text_utf8_sha256": sha256_bytes(args.text.encode("utf-8")),
            "generation_text": generation_text,
            "speaker_id": args.speaker_id if args.speaker_id is not None else 0,
            "speaker_tag": f"<|speaker:{args.speaker_id if args.speaker_id is not None else 0}|>",
            "speaker_tag_injected_by_cli": args.speaker_id is not None,
            "prompt_schedule": "official Model.generate split_text/chunk loop and history; GPU history rows use zero placeholders then terminal byte-exact official prompt verification",
        },
        "settings": {
            "seed": args.seed,
            "seed_semantics": "seeds MLX stock RNG; with --random-tape the supplied uniforms control all 11 draws per generation iteration and seed does not control those choices",
            "chunk_text": True,
            "chunk_length_bytes": int(fish_omni.DRAMA3_CHUNK_BYTES),
            "single_chunk": False,
            "chunk_count": len(chunk_texts),
            "max_frames_per_chunk": args.max_frames,
            "max_total_frame_budget": chunk_tape_frame_budget(args.max_frames, len(chunk_texts)),
            "temperature": TEMPERATURE,
            "top_k": TOP_K,
            "top_p": TOP_P,
            "repetition_aware_retry": {"temperature": RAS_TEMPERATURE, "top_k": RAS_TOP_K,
                                       "top_p": RAS_TOP_P, "window_frames": RAS_WINDOW_FRAMES},
            "sampler": "shared_uniform_inverse_cdf_shadow" if uniforms is not None else "pinned_official_mlx_categorical",
            "sampler_draw_order": ["main", "main_retry", *[f"codebook_{i}" for i in range(1, 10)]],
            "draws_per_iteration": DRAW_STRIDE,
            "random_tape": None if uniforms is None else {
                "path": str(args.random_tape.expanduser().resolve()),
                "sha256": sha256_bytes(random_tape_bytes or b""),
                "uniform_count": len(uniforms),
                "required_uniform_count": chunk_tape_frame_budget(args.max_frames, len(chunk_texts)) * DRAW_STRIDE,
                "used_uniform_count": sampler.draw_count,
                "uniform_domain": "inclusive [0,1]; 1 selects the last retained candidate",
                "host_uploads": 1,
                "sampler_scope": "one preuploaded tape is consumed continuously across official chunks; same top-k/top-p filtering and native endpoint rule, though numeric operation order and precision may diverge",
            },
            "capture_steps_global": list(args.capture_steps),
        },
        "timings_seconds": {
            **load_timings,
            "prompt_build_sum": sum(record["prompt_build_seconds"] for record in adapter.records),
            "official_generate_with_decode": generation_seconds,
            "generation_codes_sum": sum(record["generation_seconds"] for record in adapter.records),
            "decode_codes_sum": sum(record.get("decode_seconds", 0.0) for record in adapter.records),
            "terminal_eval": terminal_eval_seconds,
            "terminal_readback_and_prompt_verification": terminal_readback_seconds,
        },
        "termination": {
            "chunk_count": len(chunk_reports),
            "generation_iterations": sampler.frame_count,
            "sampler_draw_count": sampler.draw_count,
            "all_chunks_at_cap": all(item["frame_cap_reached"] for item in chunk_reports),
            "statuses": [item["termination_status"] for item in chunk_reports],
        },
        "chunks": chunk_reports,
        "output": {
            "wav_path": str(output_path) if combined_wav_bytes else None,
            "wav_sha256": sha256_bytes(combined_wav_bytes) if combined_wav_bytes else None,
            "samples": int(combined_audio.size),
            "seconds": float(combined_audio.size / SAMPLE_RATE),
            "sample_rate": SAMPLE_RATE,
            "chunk_wav_paths": [entry["output"]["wav_path"] for entry in chunk_reports],
        },
        "sampler_capture": {
            "path": str(capture_path) if capture_manifest else None,
            "sha256": capture_file_sha,
            "entries": capture_manifest,
            "trajectory_scope": "captures follow this chunked autoregressive run; cross-run logit comparison is common-prefix evidence only through equal prior frames",
            "readback_policy": "captured input-logit arrays are retained as MLX arrays and converted only after terminal mx.eval",
        },
        "voice_support": {
            "official_generate_voice_argument": "accepted but discarded by pinned Fish Omni Model.generate",
            "speaker_tags": "prompt syntax supports <|speaker:0|> through <|speaker:15|>; tags identify segments but do not provide reference voice identity by themselves",
            "identity_conditioning": "requires matching reference audio/transcript or ref_codes/ref_texts in the official high-level API",
            "reference_voice_supplied": False,
        },
        "report_path": str(report_path),
    }
    report_bytes = (json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n").encode("utf-8")
    atomic_create(report_path, report_bytes)
    progress(
        "chunk_outputs_complete",
        chunks=len(chunk_reports),
        frames=sampler.frame_count,
        samples=int(combined_audio.size),
        statuses=[item["termination_status"] for item in chunk_reports],
    )
    print(
        json.dumps(
            {
                "report": str(report_path),
                "wav": str(output_path) if combined_wav_bytes else None,
                "chunks": len(chunk_reports),
                "frames": sampler.frame_count,
                "termination": report["termination"]["statuses"],
            },
            ensure_ascii=False,
        )
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    progress("input_validation_started")
    output_path = args.output.expanduser().resolve()
    report_path = args.report.expanduser().resolve()
    mlx_root = args.mlx_root.expanduser().resolve()
    model_dir = args.model_dir.expanduser().resolve()
    replay_path = args.replay_script.expanduser().resolve()
    capture_path = report_path.with_suffix(report_path.suffix + ".sampler_logits.npz")
    prompt_capture_path = report_path.with_suffix(report_path.suffix + ".prompt_rows.f32")
    if (
        output_path == report_path
        or output_path == capture_path
        or report_path == capture_path
        or (args.reference_audio is not None and prompt_capture_path in (output_path, report_path, capture_path))
    ):
        raise SystemExit("--output, --report, and generated capture path must differ")
    for path in (output_path, report_path):
        if path.exists():
            raise SystemExit(f"refusing to overwrite existing output: {path}")
    if args.capture_steps and capture_path.exists():
        raise SystemExit(f"refusing to overwrite existing capture output: {capture_path}")
    if args.reference_audio is not None and prompt_capture_path.exists():
        raise SystemExit(f"refusing to overwrite existing prompt capture output: {prompt_capture_path}")
    if not mlx_root.is_dir() or not model_dir.is_dir() or not replay_path.is_file():
        raise SystemExit("--mlx-root, --model-dir, and --replay-script must exist")
    if not args.text:
        raise SystemExit("--text must not be empty")
    reference_wav = None
    if args.reference_audio is not None:
        reference_path = args.reference_audio.expanduser().resolve()
        try:
            reference_wav = read_reference_pcm16_wav(reference_path)
        except ValueError as error:
            raise SystemExit(str(error)) from error

    upstream = git_metadata(mlx_root)
    if not upstream["worktree_clean"]:
        raise SystemExit("--mlx-root must be clean at the pinned MLX-audio revision")
    source_hashes = {name: sha256_file(mlx_root / name) for name in UPSTREAM_FILES}
    for name, digest in source_hashes.items():
        if len(digest) != 64:
            raise SystemExit(f"failed to hash pinned source file {name}")
    replay_sha = sha256_file(replay_path)
    replay_repo = Path(
        subprocess.check_output(
            ["git", "-C", str(replay_path.parent), "rev-parse", "--show-toplevel"],
            text=True,
            stderr=subprocess.PIPE,
        ).strip()
    ).resolve()
    replay_revision = subprocess.check_output(
        ["git", "-C", str(replay_repo), "rev-parse", "HEAD"],
        text=True,
        stderr=subprocess.PIPE,
    ).strip()

    config_path = model_dir / "config.json"
    index_path = model_dir / "model.safetensors.index.json"
    for path in (config_path, index_path, model_dir / "tokenizer.json", model_dir / "codec.safetensors"):
        if not path.is_file():
            raise SystemExit(f"required local model file is missing: {path}")
    config_bytes = config_path.read_bytes()
    index_bytes = index_path.read_bytes()
    try:
        config = json.loads(config_bytes)
        index = json.loads(index_bytes)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise SystemExit(f"invalid local model config/index JSON: {error}") from error
    weight_map = index.get("weight_map") if isinstance(index, dict) else None
    if not isinstance(weight_map, dict) or not weight_map:
        raise SystemExit("model index must contain a non-empty weight_map")
    indexed_shards = sorted({str(shard) for shard in weight_map.values()})
    for shard in indexed_shards:
        if Path(shard).name != shard or not (model_dir / shard).is_file():
            raise SystemExit(f"invalid or missing local checkpoint shard: {shard}")
    if config.get("model_type") != "fish_omni":
        raise SystemExit(f"expected fish_omni checkpoint, got {config.get('model_type')!r}")

    uniforms = None
    random_tape_bytes = None
    if args.random_tape is not None and not args.chunk_text:
        uniforms, random_tape_bytes = load_random_tape(args.random_tape.expanduser().resolve(), args.max_frames)
    if not args.chunk_text:
        progress("input_validation_complete", indexed_shard_count=len(indexed_shards))

    # Local-only paths and offline flags keep the generation run from fetching files.
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
    sys.path.insert(0, str(mlx_root))
    replay_spec = importlib.util.spec_from_file_location("drama3_replay_mlx_generation", replay_path)
    if replay_spec is None or replay_spec.loader is None:
        raise SystemExit(f"cannot load F32 loader: {replay_path}")
    replay = importlib.util.module_from_spec(replay_spec)
    replay_spec.loader.exec_module(replay)
    if getattr(replay, "MLX_REVISION", None) != PINNED_REVISION:
        raise SystemExit("replay loader does not target the pinned MLX-audio revision")

    import mlx
    import mlx.core as mx
    import numpy as np
    from mlx_audio.utils import apply_quantization
    import mlx_audio.tts.models.fish_omni.fish_omni as fish_omni
    import mlx_audio.tts.models.fish_omni.tower as tower_module
    from mlx_audio.tts.models.fish_omni.config import ModelConfig
    from mlx_audio.tts.models.fish_omni.prompt import build_prompt, validate_text

    generation_text = (
        f"<|speaker:{args.speaker_id}|> {args.text}"
        if args.speaker_id is not None
        else args.text
    )
    try:
        validate_text(generation_text, "text")
        if args.reference_text is not None:
            validate_text(args.reference_text, "reference 1")
    except ValueError as error:
        raise SystemExit(str(error)) from error
    chunk_texts: list[str] = []
    if args.chunk_text:
        chunk_texts = fish_omni.split_text(
            generation_text, fish_omni.DRAMA3_CHUNK_BYTES
        ) or [generation_text]
        try:
            validate_chunk_capture_steps(
                args.capture_steps, args.max_frames, len(chunk_texts)
            )
        except ValueError as error:
            raise SystemExit(str(error)) from error
        if args.random_tape is not None:
            uniforms, random_tape_bytes = load_random_tape(
                args.random_tape.expanduser().resolve(),
                args.max_frames,
                chunk_count=len(chunk_texts),
            )
        chunk_output_paths = [
            output_path.with_name(
                f"{output_path.stem}.chunk-{index + 1:03d}{output_path.suffix or '.wav'}"
            )
            for index in range(len(chunk_texts))
        ]
        planned_paths = [output_path, report_path, *chunk_output_paths]
        if args.capture_steps:
            planned_paths.append(capture_path)
        if args.reference_audio is not None:
            planned_paths.append(prompt_capture_path)
        if len(set(planned_paths)) != len(planned_paths):
            raise SystemExit("chunk WAV, combined WAV, report, and capture paths must differ")
        for path in planned_paths:
            if path.exists():
                raise SystemExit(f"refusing to overwrite existing output: {path}")
        progress(
            "input_validation_complete",
            indexed_shard_count=len(indexed_shards),
            chunk_count=len(chunk_texts),
            max_total_frame_budget=chunk_tape_frame_budget(args.max_frames, len(chunk_texts)),
        )

    mx.set_default_device(mx.gpu)
    if mx.default_device() != mx.gpu:
        raise SystemExit(f"failed to select MLX GPU: {mx.default_device()}")
    previous_cache_limit = int(mx.set_cache_limit(0))
    old_prefill_chunk_size = fish_omni.PREFILL_CHUNK_SIZE
    fish_omni.PREFILL_CHUNK_SIZE = 1
    progress("mlx_runtime_configured", default_device=str(mx.default_device()), cache_limit_bytes=0,
             prefill_chunk_size=1)

    model_config_started = time.perf_counter()
    parsed_config = ModelConfig.from_dict(config)
    model = fish_omni.Model(parsed_config)
    model_construct_seconds = time.perf_counter() - model_config_started
    selected = replay.selected_core_map(weight_map)

    quant_started = time.perf_counter()
    apply_quantization(model, config, selected)
    quantization_seconds = time.perf_counter() - quant_started

    load_started = time.perf_counter()
    loaded_core_tensors = replay.load_core_weights(model, model_dir, selected, mx)
    core_load_seconds = time.perf_counter() - load_started
    if loaded_core_tensors != len(selected):
        raise SystemExit(f"loaded {loaded_core_tensors} core tensors; expected {len(selected)}")

    eval_started = time.perf_counter()
    model.eval()
    model_eval_seconds = time.perf_counter() - eval_started
    hook_started = time.perf_counter()
    model = fish_omni.Model.post_load_hook(model, model_dir)
    post_load_seconds = time.perf_counter() - hook_started
    load_seconds = model_construct_seconds + quantization_seconds + core_load_seconds + model_eval_seconds + post_load_seconds

    if model.model_type != "fish_omni" or model.config.sample_rate != SAMPLE_RATE:
        raise SystemExit("loaded model type or sample rate differs from the checked checkpoint")
    if model.config.audio_decoder_config.num_codebooks != NUM_CODEBOOKS:
        raise SystemExit("loaded model does not have ten audio codebooks")
    reference_codes_gpu = None
    reference_lengths_gpu = None
    reference_encode_seconds = 0.0
    if reference_wav is not None:
        if model.codec is None:
            raise SystemExit("reference audio requires the checkpoint's loaded Fish S1 DAC codec")
        pcm_i16 = np.frombuffer(reference_wav["pcm_bytes"], dtype=np.dtype("<i2"))
        pcm_f32 = pcm_i16.astype(np.float32, copy=False)
        reference_audio_gpu = (
            mx.array(pcm_f32[None, None, :], dtype=mx.float32) / 32768.0
        )
        expected_reference_frames = math.ceil(reference_wav["samples"] / FRAME_SAMPLES)
        reference_encode_started = time.perf_counter()
        encoded_reference, reference_lengths_gpu = model.codec.encode(reference_audio_gpu)
        if (
            len(encoded_reference.shape) != 3
            or tuple(encoded_reference.shape[:2]) != (1, NUM_CODEBOOKS)
            or int(encoded_reference.shape[2]) != expected_reference_frames
        ):
            raise SystemExit(
                "reference DAC returned invalid code geometry: "
                f"{tuple(encoded_reference.shape)}, expected [1,{NUM_CODEBOOKS},{expected_reference_frames}]"
            )
        if tuple(reference_lengths_gpu.shape) != (1,):
            raise SystemExit(
                f"reference DAC returned invalid feature lengths shape {tuple(reference_lengths_gpu.shape)}"
            )
        reference_codes_gpu = encoded_reference[0, :, :expected_reference_frames]
        reference_encode_seconds = time.perf_counter() - reference_encode_started
    if args.chunk_text:
        sampler = SamplerControl(mx, uniforms, set(args.capture_steps))
        load_timings = {
            "model_load_total": load_seconds,
            "model_construct": model_construct_seconds,
            "quantization_layout": quantization_seconds,
            "core_shardwise_load": core_load_seconds,
            "model_eval": model_eval_seconds,
            "post_load_hook_and_codec": post_load_seconds,
        }
        try:
            return run_chunk_text_control(
                args,
                model=model,
                mx=mx,
                np=np,
                mlx=mlx,
                fish_omni=fish_omni,
                tower_module=tower_module,
                official_build_prompt=build_prompt,
                sampler=sampler,
                chunk_texts=chunk_texts,
                generation_text=generation_text,
                output_path=output_path,
                report_path=report_path,
                capture_path=capture_path,
                uniforms=uniforms,
                random_tape_bytes=random_tape_bytes,
                upstream=upstream,
                source_hashes=source_hashes,
                replay_path=replay_path,
                replay_sha=replay_sha,
                replay_revision=replay_revision,
                model_dir=model_dir,
                config_bytes=config_bytes,
                index_bytes=index_bytes,
                indexed_shards=indexed_shards,
                loaded_core_tensors=loaded_core_tensors,
                previous_cache_limit=previous_cache_limit,
                load_timings=load_timings,
            )
        finally:
            fish_omni.PREFILL_CHUNK_SIZE = old_prefill_chunk_size

    prompt_started = time.perf_counter()
    reference_prompt_spans: list[tuple[int, int]] = []
    if reference_codes_gpu is None:
        # Preserve the original no-reference prompt path and its exact host builder.
        prompt = build_prompt(
            generation_text,
            model.tokenizer,
            num_codebooks=NUM_CODEBOOKS,
            codebook_size=int(model.config.audio_decoder_config.vocab_size),
            semantic_start=int(model.semantic_begin),
        )
        prompt_ids = [int(value) for value in prompt[:, 0].tolist()]
    else:
        prompt, _skeleton, reference_prompt_spans = build_reference_prompt_gpu(
            mx,
            np,
            build_prompt,
            text=generation_text,
            tokenizer=model.tokenizer,
            codebook_size=int(model.config.audio_decoder_config.vocab_size),
            semantic_begin=int(model.semantic_begin),
            reference_text=args.reference_text,
            reference_codes=reference_codes_gpu,
        )
        # The prompt and encoded reference stay on GPU until generation completes.
        prompt_ids = None
    prompt_seconds = time.perf_counter() - prompt_started
    if prompt_ids is not None and not prompt_ids:
        raise SystemExit("prompt builder produced no token rows")

    sampler = SamplerControl(mx, uniforms, set(args.capture_steps))
    original_fish_sampler = fish_omni.sample_top_k_top_p
    original_tower_sampler = tower_module.sample_top_k_top_p
    original_sample_main = model.sample_main

    def controlled_fish_sampler(logits: Any, temperature: float, top_k: int, top_p: float) -> Any:
        return sampler.sample(original_fish_sampler, logits, temperature, top_k, top_p)

    def controlled_tower_sampler(logits: Any, temperature: float, top_k: int, top_p: float) -> Any:
        return sampler.sample(original_tower_sampler, logits, temperature, top_k, top_p)

    def tracked_sample_main(*call_args: Any, **call_kwargs: Any) -> Any:
        if call_args and sampler.frame_count == 0 and 0 in sampler.capture_steps:
            sampler.capture_main_hidden(sampler.frame_count, call_args[0])
        return sampler.track_main(original_sample_main, *call_args, **call_kwargs)

    fish_omni.sample_top_k_top_p = controlled_fish_sampler
    tower_module.sample_top_k_top_p = controlled_tower_sampler
    model.sample_main = tracked_sample_main
    mx.random.seed(args.seed)
    generation_started = time.perf_counter()
    progress("official_generation_started", max_frames=args.max_frames,
             sampler="shared_uniform_inverse_cdf_shadow" if uniforms is not None else "official_mlx_categorical")
    try:
        codes = model.generate_codes(
            prompt,
            max_new_tokens=args.max_frames,
            temperature=TEMPERATURE,
            top_k=TOP_K,
            top_p=TOP_P,
        )
    finally:
        fish_omni.sample_top_k_top_p = original_fish_sampler
        tower_module.sample_top_k_top_p = original_tower_sampler
        model.sample_main = original_sample_main
        fish_omni.PREFILL_CHUNK_SIZE = old_prefill_chunk_size
    generation_seconds = time.perf_counter() - generation_started
    if sampler.draw_count != sampler.frame_count * DRAW_STRIDE:
        raise SystemExit(
            f"Fish Omni sampler-call count {sampler.draw_count} did not equal "
            f"{sampler.frame_count} frames * {DRAW_STRIDE} draws"
        )

    decode_started = time.perf_counter()
    audio = model.decode_codes(codes) if codes is not None else None
    decode_seconds = time.perf_counter() - decode_started
    terminal_eval_started = time.perf_counter()
    eval_values = [value for value in (codes, audio, sampler.last_main) if value is not None]
    eval_values.extend(entry[2] for entry in sampler.capture_refs)
    eval_values.extend(hidden for _step, hidden in sampler.hidden_capture_refs)
    if reference_codes_gpu is not None:
        eval_values.extend((reference_codes_gpu, reference_lengths_gpu, prompt))
    if eval_values:
        mx.eval(*eval_values)
    terminal_eval_seconds = time.perf_counter() - terminal_eval_started

    terminal_readback_started = time.perf_counter()
    if codes is None:
        codes_np = np.empty((NUM_CODEBOOKS, 0), dtype=np.int32)
        audio_np = np.empty((0,), dtype=np.float32)
        codes_dtype = None
        audio_dtype = None
    else:
        codes_dtype = str(codes.dtype)
        audio_dtype = str(audio.dtype)
        codes_np = np.asarray(codes)
        audio_np = np.asarray(audio)
    prompt_np = None
    reference_codes_np = None
    reference_codes_sha = None
    reference_lengths = None
    reference_prompt_verified = None
    prompt_capture_bytes = None
    prompt_capture_sha = None
    prompt_rows_sha = None
    if reference_codes_gpu is not None:
        if reference_lengths_gpu is None:
            raise SystemExit("reference encoder did not return feature lengths")
        reference_codes_np = np.asarray(reference_codes_gpu)
        if reference_codes_np.shape != (NUM_CODEBOOKS, math.ceil(reference_wav["samples"] / FRAME_SAMPLES)):
            raise SystemExit(f"terminal reference code shape is invalid: {reference_codes_np.shape}")
        if not np.issubdtype(reference_codes_np.dtype, np.integer):
            raise SystemExit(f"terminal reference codes are not integer: {reference_codes_np.dtype}")
        reference_lengths_np = np.asarray(reference_lengths_gpu)
        if not np.issubdtype(reference_lengths_np.dtype, np.integer):
            raise SystemExit(f"terminal reference feature lengths are not integer: {reference_lengths_np.dtype}")
        reference_lengths = [int(value) for value in reference_lengths_np.reshape(-1).tolist()]
        expected_reference_frames = math.ceil(reference_wav["samples"] / FRAME_SAMPLES)
        if reference_lengths != [expected_reference_frames]:
            raise SystemExit(
                f"terminal reference feature lengths {reference_lengths} do not match "
                f"PCM frame count {expected_reference_frames}"
            )
        for book_index, limit in enumerate((4096,) + (1024,) * 9):
            row = reference_codes_np[book_index]
            if row.size and (int(row.min()) < 0 or int(row.max()) >= limit):
                raise SystemExit(f"reference code IDs for book {book_index} are out of range")
        reference_codes_np = np.asarray(reference_codes_np, dtype=np.dtype("<i4"), order="C")
        reference_codes_sha = sha256_bytes(reference_codes_np.tobytes(order="C"))
        prompt_np = np.asarray(prompt)
        official_prompt = verify_reference_prompt(
            np,
            build_prompt,
            actual_prompt=prompt_np,
            text=generation_text,
            tokenizer=model.tokenizer,
            codebook_size=int(model.config.audio_decoder_config.vocab_size),
            semantic_begin=int(model.semantic_begin),
            reference_text=args.reference_text,
            reference_codes=reference_codes_np,
        )
        reference_prompt_verified = True
        if prompt_np.ndim != 2 or prompt_np.shape[1] != NUM_CODEBOOKS + 1:
            raise SystemExit(f"terminal reference prompt has invalid shape {prompt_np.shape}")
        prompt_ids = [int(value) for value in prompt_np[:, 0].tolist()]
        prompt_capture_values = np.asarray(prompt_np, dtype=np.dtype("<f4"), order="C")
        if not np.isfinite(prompt_capture_values).all() or not np.array_equal(
            prompt_capture_values, np.rint(prompt_capture_values)
        ):
            raise SystemExit("terminal reference prompt rows must be finite exact integer values")
        prompt_capture_bytes = prompt_capture_values.tobytes(order="C")
        prompt_capture_sha = sha256_bytes(prompt_capture_bytes)
        prompt_rows_sha = sha256_bytes(prompt_capture_bytes)
        if prompt_np.shape != official_prompt.shape:
            raise SystemExit("verified official reference prompt unexpectedly changed shape")
    last_main = int(np.asarray(sampler.last_main).reshape(-1)[0]) if sampler.last_main is not None else None
    capture_arrays: dict[str, Any] = {}
    capture_manifest: list[dict[str, Any]] = []
    for capture_step, label, logits in sampler.capture_refs:
        key = f"step_{capture_step:05d}_{label}"
        values = np.asarray(logits, dtype=np.float32)
        if values.ndim != 2 or values.shape[0] != 1 or not np.isfinite(values).all():
            raise SystemExit(f"captured sampler logits {key} have invalid values or shape {values.shape}")
        capture_arrays[key] = values
        capture_manifest.append({"key": key, "step": capture_step, "sampler": label,
                                 "shape": list(values.shape), "dtype": "float32"})
    for capture_step, hidden in sampler.hidden_capture_refs:
        key = f"step_{capture_step:05d}_prompt_backbone_hidden"
        values = np.asarray(hidden, dtype=np.float32)
        if values.ndim != 3 or values.shape[:2] != (1, 1) or not np.isfinite(values).all():
            raise SystemExit(
                f"captured prompt backbone hidden {key} has invalid values or shape {values.shape}"
            )
        values = values[:, 0, :]
        capture_arrays[key] = values
        capture_manifest.append({
            "key": key,
            "step": capture_step,
            "kind": "prompt.backbone.hidden",
            "shape": list(values.shape),
            "dtype": "float32",
        })
    terminal_readback_seconds = time.perf_counter() - terminal_readback_started

    if codes_np.ndim != 2 or codes_np.shape[0] != NUM_CODEBOOKS:
        raise SystemExit(f"official generation returned invalid code shape {codes_np.shape}")
    if codes_np.size:
        codebook_size = int(model.config.audio_decoder_config.vocab_size)
        if not np.issubdtype(codes_np.dtype, np.integer):
            raise SystemExit(f"official code output is not integer: {codes_np.dtype}")
        if int(codes_np.min()) < 0 or int(codes_np.max()) >= codebook_size:
            raise SystemExit("official generation returned out-of-range code IDs")
    frame_count = int(codes_np.shape[1])
    if audio_np.size and (audio_np.ndim != 1 or audio_np.dtype != np.dtype("float32")):
        raise SystemExit(f"decoded audio must be float32 mono, got {audio_np.dtype} {audio_np.shape}")
    if audio_np.size and audio_np.size < frame_count:
        raise SystemExit("decoded audio is implausibly shorter than the generated code count")

    eos_id = int(model.eos_id)
    hit_eos = last_main == eos_id
    termination_status = "eos" if hit_eos else (
        "max_frames" if sampler.frame_count == args.max_frames else "returned_before_cap_unknown"
    )
    semantic_begin = int(model.semantic_begin)
    frames = [
        [semantic_begin + int(codes_np[0, index]), *[int(codes_np[book, index]) for book in range(NUM_CODEBOOKS)]]
        for index in range(frame_count)
    ]
    pcm = pcm16_bytes(audio_np, np) if audio_np.size else b""
    wav_bytes = make_wav_bytes(pcm) if pcm else b""

    capture_file_sha = None
    if capture_arrays:
        import io
        buffer = io.BytesIO()
        np.savez_compressed(buffer, **capture_arrays)
        capture_bytes = buffer.getvalue()
        atomic_create(capture_path, capture_bytes)
        capture_file_sha = sha256_bytes(capture_bytes)

    if prompt_capture_bytes is not None:
        atomic_create(prompt_capture_path, prompt_capture_bytes)

    if wav_bytes:
        atomic_create(output_path, wav_bytes)
    try:
        mlx_version = importlib.metadata.version("mlx")
    except importlib.metadata.PackageNotFoundError:
        mlx_version = getattr(mlx, "__version__", "unknown")

    source_hashes["scripts/drama3/replay_mlx_generation.py"] = replay_sha
    source_hashes["scripts/drama3/generate_mlx_control.py"] = sha256_file(Path(__file__).resolve())
    report: dict[str, Any] = {
        "format": "drama3-mlx-control-v1",
        "claim_scope": "pinned Fish Omni F32-core generation control; stock MLX categorical sampling unless an explicitly labeled shared-uniform inverse-CDF shadow tape is supplied; not native RNG parity or a voice-identity claim",
        "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "upstream": upstream,
        "source_hashes": source_hashes,
        "runtime": {
            "python": sys.version,
            "python_executable": sys.executable,
            "platform": platform.platform(),
            "mlx_version": mlx_version,
            "replay_loader_path": str(replay_path),
            "replay_loader_sha256": replay_sha,
            "replay_loader_git_revision": replay_revision,
            "default_device": str(mx.default_device()),
            "hub_network_policy": {"HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1",
                                   "HF_HUB_DISABLE_TELEMETRY": "1"},
            "allocator_cache_limit_bytes": 0,
            "previous_allocator_cache_limit_bytes": previous_cache_limit,
            "prefill_chunk_size": 1,
            "host_readback_scope": (
                "official Fish Omni per-frame scalar EOS check, official DAC decoded-length scalar, "
                "one terminal sampled-main scalar for EOS classification, and terminal output/logit "
                "conversion after mx.eval; reference codes and the complete GPU-built prompt are read "
                "only after generation for hash and exact official-builder verification; no per-token "
                "activation readback"
                if reference_wav is not None
                else "official Fish Omni per-frame scalar EOS check, official DAC decoded-length scalar, "
                "one terminal sampled-main scalar for EOS classification, and terminal output/logit "
                "conversion after mx.eval; no per-token activation readback"
            ),
        },
        "model": {
            "repository": MODEL_REPOSITORY,
            "revision": MODEL_REVISION,
            "path": str(model_dir),
            "model_type": model.model_type,
            "config_sha256": sha256_bytes(config_bytes),
            "index_sha256": sha256_bytes(index_bytes),
            "indexed_shards": indexed_shards,
            "core_loader": {
                "kind": "replay_mlx_generation.py shardwise loader",
                "floating_checkpoint_tensors": "promoted to F32 before mx.array",
                "packed_integer_tensors": "preserved at checkpoint integer dtype",
                "quantization_layout": "official apply_quantization before shardwise load",
                "loaded_core_tensor_count": loaded_core_tensors,
                "official_post_load_hook": True,
                "codec": "official Fish S1 DAC loader, checkpoint dtype",
            },
            "sample_rate": SAMPLE_RATE,
            "num_codebooks": NUM_CODEBOOKS,
            "codebook_size": int(model.config.audio_decoder_config.vocab_size),
            "semantic_begin": semantic_begin,
            "eos_id": eos_id,
        },
        "input": {
            "text": args.text,
            "text_utf8_sha256": sha256_bytes(args.text.encode("utf-8")),
            "prompt_rows": len(prompt_ids),
            "prompt_ids": prompt_ids,
            "speaker_id": args.speaker_id if args.speaker_id is not None else 0,
            "speaker_tag": f"<|speaker:{args.speaker_id if args.speaker_id is not None else 0}|>",
            "speaker_tag_injected_by_cli": args.speaker_id is not None,
            "prompt_schedule": (
                "official prompt builder creates a zero-code reference skeleton; MLX reference code "
                "rows are spliced before GPU prefill; one-row prefill chunks"
                if reference_wav is not None
                else "official prompt builder; one-row prefill chunks"
            ),
        },
        "settings": {
            "seed": args.seed,
            "seed_semantics": "seeds MLX stock RNG; with --random-tape the supplied uniforms control all 11 draws per generation iteration and seed does not control those choices",
            "max_frames": args.max_frames,
            "temperature": TEMPERATURE,
            "top_k": TOP_K,
            "top_p": TOP_P,
            "repetition_aware_retry": {"temperature": RAS_TEMPERATURE, "top_k": RAS_TOP_K,
                                       "top_p": RAS_TOP_P, "window_frames": RAS_WINDOW_FRAMES},
            "sampler": "shared_uniform_inverse_cdf_shadow" if uniforms is not None else "pinned_official_mlx_categorical",
            "sampler_draw_order": ["main", "main_retry", *[f"codebook_{i}" for i in range(1, 10)]],
            "draws_per_iteration": DRAW_STRIDE,
            "random_tape": None if uniforms is None else {
                "path": str(args.random_tape.expanduser().resolve()),
                "sha256": sha256_bytes(random_tape_bytes or b""),
                "uniform_count": len(uniforms),
                "used_uniform_count": sampler.draw_count,
                "uniform_domain": "inclusive [0,1]; 1 selects the last retained candidate",
                "host_uploads": 1,
                "sampler_scope": "same top-k and top-p candidate filtering; MLX F32 inverse CDF advances on exact CDF equality like Crystal's subtractive rule and clamps U=1 to the last retained candidate; numeric operation order and precision may still diverge from native",
            },
            "capture_steps": list(args.capture_steps),
        },
        "timings_seconds": {
            "model_load_total": load_seconds,
            "model_construct": model_construct_seconds,
            "quantization_layout": quantization_seconds,
            "core_shardwise_load": core_load_seconds,
            "model_eval": model_eval_seconds,
            "post_load_hook_and_codec": post_load_seconds,
            "prompt_build": prompt_seconds,
            "generation": generation_seconds,
            "decode_codes": decode_seconds,
            "reference_encode": reference_encode_seconds,
            "terminal_eval": terminal_eval_seconds,
            "terminal_readback_and_capture_conversion": terminal_readback_seconds,
        },
        "termination": {
            "status": termination_status,
            "eos_detected": hit_eos,
            "last_sampled_main_token": last_main,
            "generation_iterations": sampler.frame_count,
            "returned_frames": frame_count,
            "max_frames": args.max_frames,
            "frame_cap_reached": sampler.frame_count == args.max_frames and not hit_eos,
            "reason_source": "terminal scalar from the final sample_main result; EOS frame itself is omitted by official generate_codes",
        },
        "output": {
            "wav_path": str(output_path) if wav_bytes else None,
            "wav_sha256": sha256_bytes(wav_bytes) if wav_bytes else None,
            "frame_shape": [frame_count, NUM_CODEBOOKS + 1],
            "frames": frames,
            "codes_shape": [NUM_CODEBOOKS, frame_count],
            "codes": codes_np.astype(np.int32, copy=False).tolist(),
            "codebook_dtype": codes_dtype,
            "decoded_audio_dtype": audio_dtype,
            "samples": int(audio_np.size),
            "seconds": float(audio_np.size / SAMPLE_RATE),
        },
        "sampler_capture": {
            "path": str(capture_path) if capture_manifest else None,
            "sha256": capture_file_sha,
            "entries": capture_manifest,
            "trajectory_scope": "captures follow this run's autoregressive trajectory; cross-run logit comparison is common-prefix evidence only through equal prior frames, otherwise use teacher-forced common-prefix replay",
            "readback_policy": "captured input-logit arrays are retained as MLX arrays and converted only after terminal mx.eval; they are not inspected during generation",
        },
        "reference_voice": None if reference_wav is None else {
            "supplied": True,
            "audio": {
                "path": str(reference_wav["path"]),
                "wav_sha256": reference_wav["wav_sha256"],
                "pcm_sha256": reference_wav["pcm_sha256"],
                "sample_rate": SAMPLE_RATE,
                "channels": 1,
                "sample_width_bits": 16,
                "samples": reference_wav["samples"],
                "normalization": "upload signed PCM16 values as F32; divide by 32768.0 on MLX GPU",
            },
            "transcript": args.reference_text,
            "transcript_utf8_sha256": sha256_bytes(args.reference_text.encode("utf-8")),
            "encoder": "pinned Fish S1 DAC.encode; codec weights retained at checkpoint F32",
            "codes": {
                "shape": list(reference_codes_np.shape),
                "layout": "codebook-major",
                "dtype": "int32-le",
                "sha256": reference_codes_sha,
                "feature_lengths": reference_lengths,
                "encoder_seconds": reference_encode_seconds,
            },
            "prompt_rows": {
                "path": str(prompt_capture_path),
                "shape": list(prompt_np.shape),
                "row_axis": "prompt_row",
                "channel_axis": "main_plus_10_codebooks",
                "dtype": "float32-le",
                "file_sha256": prompt_capture_sha,
                "rows_sha256": prompt_rows_sha,
                "reference_placeholder_spans": [list(span) for span in reference_prompt_spans],
                "official_builder_byte_equal_after_generation": reference_prompt_verified,
            },
            "complete_conditioning": True,
        },
        "voice_support": {
            "official_generate_voice_argument": "accepted but discarded by pinned Fish Omni Model.generate",
            "speaker_tags": "prompt syntax supports <|speaker:0|> through <|speaker:15|>; tags identify segments but do not provide reference voice identity by themselves",
            "identity_conditioning": "requires matching reference audio/transcript or ref_codes/ref_texts in the official high-level API",
            "this_run": {
                "speaker_id": args.speaker_id if args.speaker_id is not None else 0,
                "speaker_tag_in_prompt": True,
                "speaker_tag_injected_by_cli": args.speaker_id is not None,
                "reference_voice_supplied": reference_wav is not None,
                "interpretation": (
                    "reference audio and its matching transcript are included in speaker-0 prompt conditioning"
                    if reference_wav is not None
                    else "the prompt builder defaults to speaker 0; an explicit --speaker-id changes prompt structure only. Neither a tag nor a changed seed promises a distinct or stable speaker identity"
                ),
            },
        },
        "report_path": str(report_path),
    }
    report_bytes = (json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n").encode("utf-8")
    atomic_create(report_path, report_bytes)
    progress("outputs_complete", frames=frame_count, samples=int(audio_np.size),
             eos=hit_eos, capture_entries=len(capture_manifest))
    print(json.dumps({"report": str(report_path), "wav": str(output_path) if wav_bytes else None,
                      "frames": frame_count, "samples": int(audio_np.size),
                      "termination": termination_status}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
