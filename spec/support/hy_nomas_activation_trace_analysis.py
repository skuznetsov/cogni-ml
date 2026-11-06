"""Selected-record, CPU-only summaries for completed Hy-Nomas traces.

This deliberately validates only selected payload ranges.  It checks the whole
index and trace directory shape, but does not reread every unselected shard or
claim that selected hashes certify the full payload corpus.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import re
import stat
import sys

import numpy as np


TRACE_SCHEMA = "cogni-ml.activation-trace"
TRACE_VERSION = 1
ANALYSIS_SCHEMA = "cogni_ml.hy_nomas_activation_trace_analysis/v1"
GRID = (30, 9, 16)
LATENT_SHAPE = (1, 32, *GRID)
IMAGE_ROWS = math.prod(GRID)
DEFAULT_LAYERS = (0, 13, 26, 39, 53)
DEFAULT_STEPS = (1, 2, 3, 4)
DTYPE_BYTES = {"float16": 2, "bfloat16": 2, "float32": 4}
NAME_PART = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")
SHARD_NAME = re.compile(r"^batch-[0-9]{6,}\.bin$")
SHA256 = re.compile(r"^[0-9a-f]{64}$")
READ_CHUNK_BYTES = 1024 * 1024
ROW_CHUNK = 32

REGIONS = {
    "Alex ROI": {"h": (0, 9), "w": (1, 8), "label": "coarse fixed coordinate ROI"},
    "Resume ROI": {"h": (2, 8), "w": (10, 15), "label": "coarse fixed coordinate ROI"},
    "rest": {"h": None, "w": None, "label": "coarse complement of the two fixed ROIs"},
}


class TraceAnalysisError(ValueError):
    """The selected trace index or payload cannot support this summary."""


def _safe_name(name: str) -> None:
    if not isinstance(name, str) or not name or "\\" in name or name.startswith("/"):
        raise TraceAnalysisError("activation name must be a safe relative name")
    parts = name.split("/")
    if any(part in ("", ".", "..") or not NAME_PART.fullmatch(part) for part in parts):
        raise TraceAnalysisError(f"activation name is not safe: {name!r}")


def _json_object_unique(pairs: list[tuple[str, object]]) -> dict:
    value = {}
    for key, item in pairs:
        if key in value:
            raise TraceAnalysisError(f"duplicate JSON key: {key}")
        value[key] = item
    return value


def _hash_format(value, label: str) -> None:
    if not isinstance(value, str) or not SHA256.fullmatch(value):
        raise TraceAnalysisError(f"invalid SHA256 for {label}")


def _integer(value, label: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise TraceAnalysisError(f"invalid {label}")
    return value


def _open_regular_nofollow(path: Path, *, text: bool = False):
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    try:
        if not stat.S_ISREG(os.fstat(descriptor).st_mode):
            raise TraceAnalysisError(f"not a regular trace file: {path.name}")
        if text:
            return os.fdopen(descriptor, "r", encoding="utf-8")
        return os.fdopen(descriptor, "rb")
    except Exception:
        os.close(descriptor)
        raise


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(READ_CHUNK_BYTES), b""):
            digest.update(chunk)
    return digest.hexdigest()


class SelectedTraceReader:
    """Validate trace metadata and read/hash one selected record at a time."""

    def __init__(self, path: Path):
        self.root = Path(path)
        if ".." in self.root.parts:
            raise TraceAnalysisError("trace path must not contain parent traversal")
        try:
            root_mode = self.root.lstat().st_mode
        except OSError as exc:
            raise TraceAnalysisError(f"trace directory is unavailable: {self.root}") from exc
        if not stat.S_ISDIR(root_mode) or stat.S_ISLNK(root_mode):
            raise TraceAnalysisError("trace path must be a real directory, not a symlink")

        try:
            entries = list(self.root.iterdir())
        except OSError as exc:
            raise TraceAnalysisError("cannot enumerate trace directory") from exc
        for entry in entries:
            mode = entry.lstat().st_mode
            if stat.S_ISLNK(mode):
                raise TraceAnalysisError(f"trace contains a symlink: {entry.name}")
            if not stat.S_ISREG(mode):
                raise TraceAnalysisError(f"trace contains a non-file entry: {entry.name}")

        if (self.root / ".incomplete").exists():
            raise TraceAnalysisError("analysis accepts complete traces only")
        index_path = self.root / "index.json"
        try:
            with _open_regular_nofollow(index_path) as handle:
                raw_index = handle.read()
        except OSError as exc:
            raise TraceAnalysisError("trace index is missing or unsafe") from exc
        self.index_sha256 = hashlib.sha256(raw_index).hexdigest()
        try:
            self.index = json.loads(raw_index, object_pairs_hook=_json_object_unique)
        except (UnicodeError, json.JSONDecodeError, TraceAnalysisError) as exc:
            raise TraceAnalysisError(f"trace index is invalid: {exc}") from exc
        self._validate_index(entries)
        self.records = {record["name"]: record for record in self.index["records"]}
        self.shards = {shard["name"]: shard for shard in self.index["shards"]}
        self.validated_records: dict[str, dict] = {}

    def _validate_index(self, entries: list[Path]) -> None:
        index = self.index
        if not isinstance(index, dict):
            raise TraceAnalysisError("trace index must be an object")
        if index.get("schema") != TRACE_SCHEMA or index.get("version") != TRACE_VERSION:
            raise TraceAnalysisError("trace schema/version is unsupported")
        if index.get("complete") is not True:
            raise TraceAnalysisError("analysis accepts complete traces only")
        metadata = index.get("metadata")
        if not isinstance(metadata, dict):
            raise TraceAnalysisError("trace metadata is invalid")
        arm = metadata.get("arm")
        if not isinstance(arm, str) or arm == "T0":
            raise TraceAnalysisError("trace metadata does not identify a T1 arm")
        _safe_name(arm)
        self.arm = arm

        limits = index.get("limits")
        if not isinstance(limits, dict):
            raise TraceAnalysisError("trace byte limits are invalid")
        max_total = _integer(limits.get("max_total_bytes"), "max_total_bytes", minimum=1)
        max_batch = _integer(limits.get("max_batch_bytes"), "max_batch_bytes", minimum=1)
        _integer(limits.get("min_free_bytes"), "min_free_bytes")
        records = index.get("records")
        shards = index.get("shards")
        expected_names = index.get("expected_names")
        if not isinstance(records, list) or not isinstance(shards, list):
            raise TraceAnalysisError("trace records or shards are invalid")
        if not isinstance(expected_names, list):
            raise TraceAnalysisError("complete trace has no expected scope")

        expected_set: set[str] = set()
        for name in expected_names:
            _safe_name(name)
            if name in expected_set:
                raise TraceAnalysisError("trace expected scope has duplicates")
            expected_set.add(name)

        record_names: set[str] = set()
        by_shard: dict[str, list[dict]] = {}
        total = 0
        for sequence, record in enumerate(records):
            if not isinstance(record, dict):
                raise TraceAnalysisError("trace record is invalid")
            if _integer(record.get("sequence"), "record sequence") != sequence:
                raise TraceAnalysisError("trace record sequence is not contiguous")
            name = record.get("name")
            _safe_name(name)
            if name in record_names:
                raise TraceAnalysisError(f"trace has duplicate record: {name}")
            record_names.add(name)
            dtype = record.get("dtype")
            if dtype not in DTYPE_BYTES:
                raise TraceAnalysisError(f"unsupported record dtype: {dtype!r}")
            shape = record.get("shape")
            if not isinstance(shape, list) or any(
                isinstance(dim, bool) or not isinstance(dim, int) or dim < 0 for dim in shape
            ):
                raise TraceAnalysisError(f"invalid shape for record: {name}")
            elements = math.prod(shape)
            nbytes = _integer(record.get("nbytes"), "record byte count")
            offset = _integer(record.get("offset"), "record offset")
            if nbytes != elements * DTYPE_BYTES[dtype]:
                raise TraceAnalysisError(f"record byte count disagrees with dtype/shape: {name}")
            _hash_format(record.get("sha256"), name)
            shard_name = record.get("shard")
            if not isinstance(shard_name, str) or not SHARD_NAME.fullmatch(shard_name):
                raise TraceAnalysisError(f"record has unsafe shard path: {shard_name!r}")
            by_shard.setdefault(shard_name, []).append(record)
            total += nbytes

        if record_names != expected_set:
            raise TraceAnalysisError("complete trace expected scope and record names differ")
        if total != _integer(index.get("total_bytes"), "trace total_bytes") or total > max_total:
            raise TraceAnalysisError("trace total byte count is inconsistent or over quota")

        shard_names: set[str] = set()
        for sequence, shard in enumerate(shards):
            if not isinstance(shard, dict):
                raise TraceAnalysisError("trace shard entry is invalid")
            if _integer(shard.get("sequence"), "shard sequence") != sequence:
                raise TraceAnalysisError("trace shard sequence is not contiguous")
            name = shard.get("name")
            if not isinstance(name, str) or not SHARD_NAME.fullmatch(name):
                raise TraceAnalysisError(f"unsafe shard path: {name!r}")
            if name != f"batch-{sequence:06d}.bin" or name in shard_names:
                raise TraceAnalysisError("trace shard names or sequence are inconsistent")
            shard_names.add(name)
            records_here = by_shard.get(name, [])
            if shard.get("record_names") != [record["name"] for record in records_here]:
                raise TraceAnalysisError(f"shard record names are inconsistent: {name}")
            offset = 0
            for record in records_here:
                if record["offset"] != offset:
                    raise TraceAnalysisError(f"record offsets are not contiguous: {record['name']}")
                offset += record["nbytes"]
            if _integer(shard.get("nbytes"), "shard byte count") != offset or offset > max_batch:
                raise TraceAnalysisError(f"shard byte count is inconsistent or over quota: {name}")
            _hash_format(shard.get("sha256"), name)

        if set(by_shard) != shard_names:
            raise TraceAnalysisError("trace records and shard index differ")
        actual_entries = {entry.name for entry in entries}
        expected_entries = {"index.json", *shard_names}
        if actual_entries != expected_entries:
            raise TraceAnalysisError("trace directory contains unexpected or missing files")
        for name, shard in ((item["name"], item) for item in shards):
            path = self.root / name
            try:
                mode = path.lstat().st_mode
            except OSError as exc:
                raise TraceAnalysisError(f"trace shard is missing: {name}") from exc
            if not stat.S_ISREG(mode) or stat.S_ISLNK(mode):
                raise TraceAnalysisError(f"trace shard is not a regular file: {name}")
            if path.lstat().st_size != shard["nbytes"]:
                raise TraceAnalysisError(f"trace shard size/truncation mismatch: {name}")

        self.record_count = len(records)
        self.payload_bytes = total

    def record_info(self, name: str, *, expected_shape=None, expected_dtype=None) -> dict:
        _safe_name(name)
        try:
            record = self.records[name]
        except KeyError as exc:
            raise TraceAnalysisError(f"missing selected record: {name}") from exc
        if expected_shape is not None and tuple(record["shape"]) != tuple(expected_shape):
            raise TraceAnalysisError(
                f"selected record shape mismatch for {name}: "
                f"{tuple(record['shape'])} != {tuple(expected_shape)}"
            )
        if expected_dtype is not None:
            expected = {expected_dtype} if isinstance(expected_dtype, str) else set(expected_dtype)
            if record["dtype"] not in expected:
                raise TraceAnalysisError(
                    f"selected record dtype mismatch for {name}: {record['dtype']} not in {sorted(expected)}"
                )
        return record

    def read_f32(self, name: str, *, expected_shape=None, expected_dtype=None) -> np.ndarray:
        """Hash and decode exactly one record; BF16 is expanded by bit shift."""
        record = self.record_info(name, expected_shape=expected_shape, expected_dtype=expected_dtype)
        shard_name = record["shard"]
        shard = self.shards[shard_name]
        path = self.root / shard_name
        try:
            with _open_regular_nofollow(path) as handle:
                if os.fstat(handle.fileno()).st_size != shard["nbytes"]:
                    raise TraceAnalysisError(f"trace shard size/truncation mismatch: {shard_name}")
                if record["offset"] + record["nbytes"] > shard["nbytes"]:
                    raise TraceAnalysisError(f"selected record range exceeds shard: {name}")
                handle.seek(record["offset"])
                payload = bytearray(record["nbytes"])
                view = memoryview(payload)
                digest = hashlib.sha256()
                position = 0
                while position < len(payload):
                    chunk = handle.read(min(READ_CHUNK_BYTES, len(payload) - position))
                    if not chunk:
                        raise TraceAnalysisError(f"selected record range is truncated: {name}")
                    view[position:position + len(chunk)] = chunk
                    digest.update(chunk)
                    position += len(chunk)
                if digest.hexdigest() != record["sha256"]:
                    raise TraceAnalysisError(f"selected record hash mismatch: {name}")
        except OSError as exc:
            raise TraceAnalysisError(f"cannot read selected record {name}: {exc}") from exc

        dtype = record["dtype"]
        shape = tuple(record["shape"])
        if dtype == "bfloat16":
            bf16 = np.frombuffer(payload, dtype="<u2")
            bits32 = bf16.astype("<u4")
            del bf16, payload
            bits32 <<= 16
            values = bits32.view("<f4").reshape(shape)
        elif dtype == "float16":
            f16 = np.frombuffer(payload, dtype="<f2")
            values = f16.astype("<f4").reshape(shape)
            del f16, payload
        else:
            values = np.frombuffer(payload, dtype="<f4").reshape(shape)
        self.validated_records[name] = {
            "name": name,
            "sequence": record["sequence"],
            "shard": shard_name,
            "offset": record["offset"],
            "nbytes": record["nbytes"],
            "dtype": dtype,
            "shape": shape,
            "sha256": record["sha256"],
        }
        return values


def _region_masks() -> dict[str, np.ndarray]:
    alex = np.zeros(GRID, dtype=bool)
    alex[:, 0:9, 1:8] = True
    resume = np.zeros(GRID, dtype=bool)
    resume[:, 2:8, 10:15] = True
    return {
        "Alex ROI": alex.reshape(-1),
        "Resume ROI": resume.reshape(-1),
        "rest": (~(alex | resume)).reshape(-1),
    }


def _rms_rows(values: np.ndarray, mask: np.ndarray) -> tuple[float, int]:
    rows, width = values.shape
    if rows != IMAGE_ROWS or mask.shape != (IMAGE_ROWS,):
        raise TraceAnalysisError("image activation does not match the 30x9x16 token grid")
    sum_squares = 0.0
    count = 0
    for start in range(0, rows, ROW_CHUNK):
        stop = min(rows, start + ROW_CHUNK)
        selected = values[start:stop][mask[start:stop]]
        if not selected.size:
            continue
        if not np.isfinite(selected).all():
            raise TraceAnalysisError("selected activation contains non-finite values")
        sum_squares += float(np.square(selected, dtype=np.float64).sum(dtype=np.float64))
        count += selected.size
    if count == 0:
        raise TraceAnalysisError("ROI has no activation elements")
    return math.sqrt(sum_squares / count), count


def _rms_diff_rows(left: np.ndarray, right: np.ndarray, mask: np.ndarray) -> tuple[float, int]:
    if left.shape != right.shape or left.ndim != 2:
        raise TraceAnalysisError("activation residual operands have different shapes")
    rows = left.shape[0]
    if rows != IMAGE_ROWS or mask.shape != (IMAGE_ROWS,):
        raise TraceAnalysisError("image activation does not match the 30x9x16 token grid")
    sum_squares = 0.0
    count = 0
    for start in range(0, rows, ROW_CHUNK):
        stop = min(rows, start + ROW_CHUNK)
        row_mask = mask[start:stop]
        if not row_mask.any():
            continue
        delta = left[start:stop] - right[start:stop]
        selected = delta[row_mask]
        if not np.isfinite(selected).all():
            raise TraceAnalysisError("selected activation residual contains non-finite values")
        sum_squares += float(np.square(selected, dtype=np.float64).sum(dtype=np.float64))
        count += selected.size
    if count == 0:
        raise TraceAnalysisError("ROI has no residual elements")
    return math.sqrt(sum_squares / count), count


def _latent_temporal_rms(values: np.ndarray, h_range: tuple[int, int] | None,
                         w_range: tuple[int, int] | None) -> tuple[float, int]:
    if tuple(values.shape) != LATENT_SHAPE:
        raise TraceAnalysisError(f"latent shape mismatch: {tuple(values.shape)}")
    if h_range is None:
        selected = [(h, w) for h in range(GRID[1]) for w in range(GRID[2])
                    if not (0 <= h < 9 and 1 <= w < 8)
                    and not (2 <= h < 8 and 10 <= w < 15)]
    else:
        assert w_range is not None
        selected = [(h, w) for h in range(*h_range) for w in range(*w_range)]
    sum_squares = 0.0
    count = 0
    for t in range(GRID[0] - 1):
        for h, w in selected:
            delta = values[0, :, t + 1, h, w] - values[0, :, t, h, w]
            if not np.isfinite(delta).all():
                raise TraceAnalysisError("selected latent contains non-finite values")
            sum_squares += float(np.square(delta, dtype=np.float64).sum(dtype=np.float64))
            count += delta.size
    if count == 0:
        raise TraceAnalysisError("ROI has no adjacent latent pairs")
    return math.sqrt(sum_squares / count), count


def _validate_selection(layers, steps) -> tuple[tuple[int, ...], tuple[int, ...]]:
    try:
        selected_layers = tuple(layers)
        selected_steps = tuple(steps)
    except TypeError as exc:
        raise TraceAnalysisError("layers and steps must be finite sequences") from exc
    if (not selected_layers or any(isinstance(item, bool) or not isinstance(item, int)
                                   or item < 0 or item >= 54 for item in selected_layers)
            or len(set(selected_layers)) != len(selected_layers)):
        raise TraceAnalysisError("layers must be distinct integers in [0, 53]")
    if (not selected_steps or any(isinstance(item, bool) or not isinstance(item, int)
                                  or item < 1 or item > 4 for item in selected_steps)
            or len(set(selected_steps)) != len(selected_steps)):
        raise TraceAnalysisError("steps must be distinct integers in [1, 4]")
    return selected_layers, selected_steps


def analyze_trace(path: Path, *, layers=DEFAULT_LAYERS, steps=DEFAULT_STEPS) -> dict:
    """Summarize selected T1 block records and every selected Euler latent."""
    selected_layers, selected_steps = _validate_selection(layers, steps)
    reader = SelectedTraceReader(path)
    arm = reader.arm
    hidden = None
    selected_names: list[str] = []
    for step in selected_steps:
        for layer in selected_layers:
            prefix = f"step{step}.{arm}.block{layer:02d}.img."
            for boundary in ("input", "after_attention", "output"):
                name = prefix + boundary
                info = reader.record_info(name)
                shape = tuple(info["shape"])
                if len(shape) != 3 or shape[0] != 1 or shape[1] != IMAGE_ROWS or shape[2] <= 0:
                    raise TraceAnalysisError(f"selected image activation has wrong shape: {name}")
                if info["dtype"] != "bfloat16":
                    raise TraceAnalysisError(f"selected image activation must preserve BF16: {name}")
                if hidden is None:
                    hidden = shape[2]
                elif shape[2] != hidden:
                    raise TraceAnalysisError("selected image activation hidden sizes differ")
                selected_names.append(name)
        latent_name = f"step{step}.{arm}.head.after_euler"
        reader.record_info(latent_name, expected_shape=LATENT_SHAPE, expected_dtype="bfloat16")
        selected_names.append(latent_name)

    masks = _region_masks()
    block_metrics = []
    for step in selected_steps:
        for layer in selected_layers:
            prefix = f"step{step}.{arm}.block{layer:02d}.img."
            input_name = prefix + "input"
            after_name = prefix + "after_attention"
            output_name = prefix + "output"
            expected_shape = (1, IMAGE_ROWS, hidden)

            image_input = reader.read_f32(input_name, expected_shape=expected_shape,
                                          expected_dtype="bfloat16").reshape(IMAGE_ROWS, hidden)
            input_stats = {region: _rms_rows(image_input, mask) for region, mask in masks.items()}

            after_attention = reader.read_f32(after_name, expected_shape=expected_shape,
                                              expected_dtype="bfloat16").reshape(IMAGE_ROWS, hidden)
            attention_stats = {
                region: _rms_diff_rows(after_attention, image_input, mask)
                for region, mask in masks.items()
            }
            del image_input

            output = reader.read_f32(output_name, expected_shape=expected_shape,
                                     expected_dtype="bfloat16").reshape(IMAGE_ROWS, hidden)
            ffn_stats = {
                region: _rms_diff_rows(output, after_attention, mask)
                for region, mask in masks.items()
            }
            del after_attention

            # Re-read input rather than retaining a third full 34 MiB array.
            image_input = reader.read_f32(input_name, expected_shape=expected_shape,
                                          expected_dtype="bfloat16").reshape(IMAGE_ROWS, hidden)
            update_stats = {
                region: _rms_diff_rows(output, image_input, mask)
                for region, mask in masks.items()
            }
            del image_input, output

            region_metrics = {}
            for region, mask in masks.items():
                input_rms, input_count = input_stats[region]
                attention_rms, attention_count = attention_stats[region]
                ffn_rms, ffn_count = ffn_stats[region]
                update_rms, update_count = update_stats[region]
                if len({input_count, attention_count, ffn_count, update_count}) != 1:
                    raise TraceAnalysisError("activation metric element counts disagree")
                region_metrics[region] = {
                    "element_count": input_count,
                    "input_rms": input_rms,
                    "attention_residual_rms": attention_rms,
                    "ffn_residual_rms": ffn_rms,
                    "relative_update": update_rms / input_rms if input_rms else None,
                }
            block_metrics.append({"step": step, "layer": layer, "arm": arm,
                                 "activation_shape": list(expected_shape),
                                 "regions": region_metrics})

    temporal_metrics = []
    for step in selected_steps:
        name = f"step{step}.{arm}.head.after_euler"
        latent = reader.read_f32(name, expected_shape=LATENT_SHAPE, expected_dtype="bfloat16")
        region_metrics = {}
        for region, definition in REGIONS.items():
            rms, count = _latent_temporal_rms(latent, definition["h"], definition["w"])
            region_metrics[region] = {"rms": rms, "element_count": count}
        temporal_metrics.append({"step": step, "arm": arm,
                                 "latent_shape": list(LATENT_SHAPE),
                                 "regions": region_metrics})
        del latent

    return {
        "schema": ANALYSIS_SCHEMA,
        "trace_directory": str(Path(path).resolve()),
        "trace_index_sha256": reader.index_sha256,
        "validation_scope": "selected_only",
        "trace": {"complete": True, "record_count": reader.record_count,
                  "payload_bytes": reader.payload_bytes,
                  "unselected_payloads_hashed": False,
                  "per_shard_hashes_recomputed": False},
        "selection": {"arm": arm, "steps": list(selected_steps),
                      "layers": list(selected_layers),
                      "selected_record_count": len(set(selected_names))},
        "regions": {
            name: {"label": definition["label"], "grid": list(GRID),
                   "h_range": list(definition["h"]) if definition["h"] is not None else None,
                   "w_range": list(definition["w"]) if definition["w"] is not None else None}
            for name, definition in REGIONS.items()
        },
        "definitions": {
            "input_rms": "RMS of img.input values",
            "attention_residual_rms": "RMS(img.after_attention - img.input)",
            "ffn_residual_rms": "RMS(img.output - img.after_attention)",
            "relative_update": "RMS(img.output - img.input) / RMS(img.input); null if denominator is zero",
            "temporal_adjacent_latent_rms": (
                "RMS(after_euler[t+1] - after_euler[t]) over latent channels and the selected coarse ROI; "
                "a descriptive noise/feature-distance statistic, not optical flow or motion"
            ),
            "interpretation_boundary": (
                "All values are descriptive tensor statistics. ROI names are coarse fixed coordinate boxes; "
                "no motion, identity, or semantic inference is made."
            ),
        },
        "block_metrics": block_metrics,
        "temporal_latent_metrics": temporal_metrics,
        "selected_record_hashes": [reader.validated_records[name] for name in selected_names],
    }


def _parse_int_list(raw: str) -> tuple[int, ...]:
    try:
        return tuple(int(item) for item in raw.split(","))
    except ValueError as exc:
        raise argparse.ArgumentTypeError("expected comma-separated integers") from exc


def _write_fresh_json(path: Path, document: dict, *, trace_path: Path) -> None:
    output = Path(path)
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"refusing to overwrite output: {output}")
    trace_root = Path(trace_path).resolve(strict=True)
    target = output.parent.resolve(strict=True) / output.name
    if target.resolve(strict=False).is_relative_to(trace_root):
        raise ValueError("analysis output must be outside the activation trace directory")
    payload = (json.dumps(document, sort_keys=True, indent=2, allow_nan=False) + "\n").encode("utf-8")
    descriptor = os.open(output, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    with os.fdopen(descriptor, "wb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace", type=Path, required=True,
                        help="completed RUN/activations directory")
    parser.add_argument("--output", type=Path, required=True,
                        help="fresh JSON summary path; existing files are never overwritten")
    parser.add_argument("--layers", type=_parse_int_list, default=DEFAULT_LAYERS,
                        help="comma-separated block indices (default: 0,13,26,39,53)")
    parser.add_argument("--steps", type=_parse_int_list, default=DEFAULT_STEPS,
                        help="comma-separated Euler steps (default: 1,2,3,4)")
    args = parser.parse_args(argv)
    result = analyze_trace(args.trace, layers=args.layers, steps=args.steps)
    _write_fresh_json(args.output, result, trace_path=args.trace)
    print(json.dumps({"output": str(args.output),
                      "trace_index_sha256": result["trace_index_sha256"],
                      "selected_record_count": result["selection"]["selected_record_count"]},
                     sort_keys=True))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, TraceAnalysisError, ValueError) as exc:
        print(f"activation trace analysis failed: {exc}", file=sys.stderr)
        raise SystemExit(2)
