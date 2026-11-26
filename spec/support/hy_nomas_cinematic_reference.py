#!/usr/bin/env python3
"""Bounded independent qualification against upstream Hy-Nomas merge().

This tool qualifies only the selected blocks' merged DiT weights and the
unchanged cache assets. It does not qualify a full model trajectory, decoder,
or perceptual quality. For each selected block it supplies exactly that
block's 32 FP32 base tensors and 28 active adapter tensors to the pinned
upstream merge implementation, then compares the resulting MPS BF16 words to
the candidate cache. It never loads the full base or adapter checkpoint.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path
import re
import stat
import sys
import tempfile
import types
from typing import Any, Iterator, Mapping


ROOT = Path(__file__).resolve().parents[2]
SUPPORT = ROOT / "examples" / "support"
sys.path.insert(0, str(SUPPORT))

import hy_nomas_cinematic_cache as contract  # noqa: E402


PINNED_UPSTREAM_MERGE_SHA256 = (
    "92e24f385660a2d37c2e23c23fd4f66f3ba1b0f8195a9e59ba500ff15a7bbee4"
)
HASH_CHUNK = 8 * 1024 * 1024
COMPARE_CHUNK_WORDS = 1024 * 1024
MAX_FIRST_MISMATCHES = 8


class QualificationError(ValueError):
    """An input failed the pinned Cinematic qualification contract."""


class FileIdentityLedger:
    """Keep the exact lstat identity of every artifact used by qualification."""

    def __init__(self) -> None:
        self._identities: dict[str, tuple[int, int, int, int]] = {}
        self._labels: dict[str, set[str]] = {}

    @staticmethod
    def _absolute(path: Path) -> Path:
        return Path(os.path.abspath(os.fspath(path)))

    def capture(self, path: Path, label: str) -> os.stat_result:
        absolute = self._absolute(path)
        try:
            info = absolute.lstat()
        except OSError as exc:
            raise QualificationError(f"{label} is unavailable: {absolute}: {exc}") from None
        if not stat.S_ISREG(info.st_mode):
            raise QualificationError(f"{label} must be a regular non-symlink file: {absolute}")
        identity = (info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns)
        key = str(absolute)
        prior = self._identities.get(key)
        if prior is not None and prior != identity:
            raise QualificationError(f"{label} file identity changed: {absolute}")
        self._identities[key] = identity
        self._labels.setdefault(key, set()).add(label)
        return info

    def assert_current(self, path: Path, label: str) -> os.stat_result:
        return self.capture(path, label)

    def assert_all_current(self, phase: str) -> None:
        for path, expected in tuple(self._identities.items()):
            labels = ", ".join(sorted(self._labels[path]))
            try:
                current = Path(path).lstat()
            except OSError as exc:
                raise QualificationError(f"{labels} became unavailable {phase}: {path}: {exc}") from None
            if not stat.S_ISREG(current.st_mode):
                raise QualificationError(f"{labels} is no longer a regular file {phase}: {path}")
            actual = (current.st_dev, current.st_ino, current.st_size, current.st_mtime_ns)
            if actual != expected:
                raise QualificationError(f"{labels} file identity changed {phase}: {path}")

    def report(self) -> list[dict[str, Any]]:
        return [
            {
                "path": path,
                "labels": sorted(self._labels[path]),
                "dev": identity[0],
                "ino": identity[1],
                "size": identity[2],
                "mtime_ns": identity[3],
            }
            for path, identity in sorted(self._identities.items())
        ]


def _json_pairs_no_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise QualificationError(f"duplicate JSON object key: {key}")
        result[key] = value
    return result


def _read_json(path: Path, label: str, *, limit: int = contract.MAX_JSON_BYTES,
               ledger: FileIdentityLedger | None = None) -> tuple[dict[str, Any], bytes]:
    info = ledger.capture(path, label) if ledger else _regular_file(path, label)
    size = info.st_size
    if size <= 0 or size > limit:
        raise QualificationError(f"{label} has an invalid byte size: {size}")
    raw = path.read_bytes()
    if len(raw) != size:
        raise QualificationError(f"{label} changed while reading")
    if ledger:
        ledger.assert_current(path, label)
    try:
        value = json.loads(raw.decode("utf-8"), object_pairs_hook=_json_pairs_no_duplicates)
    except (UnicodeDecodeError, json.JSONDecodeError, RecursionError) as exc:
        raise QualificationError(f"{label} is not duplicate-free UTF-8 JSON: {exc}") from None
    if not isinstance(value, dict):
        raise QualificationError(f"{label} root must be an object")
    return value, raw


def _regular_file(path: Path, label: str) -> os.stat_result:
    try:
        info = path.lstat()
    except OSError as exc:
        raise QualificationError(f"{label} is unavailable: {path}: {exc}") from None
    if not stat.S_ISREG(info.st_mode):
        raise QualificationError(f"{label} must be a regular non-symlink file: {path}")
    return info


def _stable_hash(path: Path, label: str, *, ledger: FileIdentityLedger | None = None) -> tuple[str, int]:
    before = ledger.capture(path, label) if ledger else _regular_file(path, label)
    digest = hashlib.sha256()
    try:
        with path.open("rb") as stream:
            opened = os.fstat(stream.fileno())
            if (opened.st_dev, opened.st_ino, opened.st_size, opened.st_mtime_ns) != (
                before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns
            ):
                raise QualificationError(f"{label} changed before hashing")
            while chunk := stream.read(HASH_CHUNK):
                digest.update(chunk)
            after = os.fstat(stream.fileno())
            if (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns) != (
                before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns
            ):
                raise QualificationError(f"{label} changed while hashing")
    except OSError as exc:
        raise QualificationError(f"could not hash {label}: {exc}") from None
    after_path = ledger.assert_current(path, label) if ledger else _regular_file(path, label)
    if (after_path.st_dev, after_path.st_ino, after_path.st_size, after_path.st_mtime_ns) != (
        before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns
    ):
        raise QualificationError(f"{label} path changed while hashing")
    return digest.hexdigest(), before.st_size


def _pinned_hash(path: Path, label: str, expected: str, expected_bytes: int | None = None,
                 *, ledger: FileIdentityLedger | None = None) -> dict[str, Any]:
    digest, size = _stable_hash(path, label, ledger=ledger)
    if digest != expected or (expected_bytes is not None and size != expected_bytes):
        raise QualificationError(f"{label} SHA-256 or byte pin mismatch")
    return {"path": str(path.resolve()), "sha256": digest, "bytes": size}


def _compare_files(left: Path, right: Path, label: str, expected_sha: str | None = None,
                   expected_bytes: int | None = None,
                   *, ledger: FileIdentityLedger | None = None) -> dict[str, Any]:
    left_info = ledger.capture(left, f"{label} candidate") if ledger else _regular_file(left, f"{label} candidate")
    right_info = ledger.capture(right, f"{label} base") if ledger else _regular_file(right, f"{label} base")
    if left_info.st_size != right_info.st_size:
        raise QualificationError(f"{label} candidate/base byte sizes differ")
    if expected_bytes is not None and left_info.st_size != expected_bytes:
        raise QualificationError(f"{label} byte count differs from manifest")

    same_inode = (left_info.st_dev, left_info.st_ino) == (right_info.st_dev, right_info.st_ino)
    if same_inode:
        digest, size = _stable_hash(left, label, ledger=ledger)
        if size != left_info.st_size:
            raise QualificationError(f"{label} changed while hashing")
        if expected_sha is not None and digest != expected_sha:
            raise QualificationError(f"{label} actual SHA-256 differs from the pinned index")
        if ledger:
            ledger.assert_current(right, f"{label} base")
        return {
            "candidate_path": str(left.resolve()),
            "base_path": str(right.resolve()),
            "bytes": size,
            "candidate_sha256": digest,
            "base_sha256": digest,
            "bytes_equal": True,
            "same_inode": True,
        }
    left_digest = hashlib.sha256()
    right_digest = hashlib.sha256()
    equal = True
    try:
        with left.open("rb") as left_stream, right.open("rb") as right_stream:
            for left_chunk in iter(lambda: left_stream.read(HASH_CHUNK), b""):
                right_chunk = right_stream.read(len(left_chunk))
                if len(right_chunk) != len(left_chunk):
                    raise QualificationError(f"{label} changed size during comparison")
                left_digest.update(left_chunk)
                right_digest.update(right_chunk)
                if left_chunk != right_chunk:
                    equal = False
            if right_stream.read(1):
                raise QualificationError(f"{label} changed size during comparison")
    except OSError as exc:
        raise QualificationError(f"could not compare {label}: {exc}") from None
    left_sha = left_digest.hexdigest()
    right_sha = right_digest.hexdigest()
    if not equal or left_sha != right_sha:
        raise QualificationError(f"{label} actual bytes differ from the base cache")
    if expected_sha is not None and (left_sha != expected_sha or right_sha != expected_sha):
        raise QualificationError(f"{label} actual SHA-256 differs from the pinned index")
    if ledger:
        ledger.assert_current(left, f"{label} candidate")
        ledger.assert_current(right, f"{label} base")
    return {
        "candidate_path": str(left.resolve()),
        "base_path": str(right.resolve()),
        "bytes": left_info.st_size,
        "candidate_sha256": left_sha,
        "base_sha256": right_sha,
        "bytes_equal": True,
        "same_inode": same_inode,
    }


def _parse_blocks(value: str) -> list[int]:
    if value == "all":
        return list(range(contract.BLOCK_COUNT))
    try:
        result = [int(part.strip()) for part in value.split(",")]
    except ValueError:
        raise QualificationError("--blocks must be 'all' or comma-separated block indices") from None
    if not result or len(set(result)) != len(result) or any(
        index < 0 or index >= contract.BLOCK_COUNT for index in result
    ):
        raise QualificationError("--blocks must contain unique indices in 0...53")
    return result


def _validate_index_pair(candidate_path: Path, candidate_sha: str,
                         base_source: Path, *, ledger: FileIdentityLedger
                         ) -> tuple[dict[str, Any], dict[str, Any], Path, dict[str, Any]]:
    if re.fullmatch(r"[0-9a-f]{64}", candidate_sha) is None:
        raise QualificationError("--candidate-index-sha must be a lowercase SHA-256")
    candidate, candidate_raw = _read_json(candidate_path, "candidate index", ledger=ledger)
    actual_candidate_sha = hashlib.sha256(candidate_raw).hexdigest()
    if actual_candidate_sha != candidate_sha:
        raise QualificationError("candidate index SHA-256 pin mismatch")
    contract.validate_cache_index_document(
        candidate,
        expected_source_sha256=contract.BASE_SOURCE_SHA256,
        expected_source_bytes=contract.BASE_SOURCE_BYTES,
    )

    base_index_path = base_source.parent / "q4-cache-v2" / "index.json"
    base, base_raw = _read_json(base_index_path, "base Q4 cache index", ledger=ledger)
    base_sha = hashlib.sha256(base_raw).hexdigest()
    contract.validate_base_index_document(base, actual_sha256=base_sha)
    for index, (candidate_entry, base_entry) in enumerate(
        zip(candidate["blocks"], base["blocks"], strict=True)
    ):
        for field in ("manifest", "manifest_sha256", "payload_sha256"):
            if candidate_entry[field] != base_entry[field]:
                raise QualificationError(f"candidate Q4 ancestry differs from base at block {index}: {field}")
        if candidate_entry["bf16"] != f"block-{index}-cinematic-bf16.safetensors":
            raise QualificationError(f"candidate BF16 block basename mismatch at block {index}")
    if candidate["globals"] != base["globals"]:
        raise QualificationError("candidate globals index identity differs from base")

    lineage_path = candidate_path.parent / "cinematic-lineage.json"
    lineage, lineage_raw = _read_json(lineage_path, "candidate Cinematic lineage", ledger=ledger)
    lineage_sha = hashlib.sha256(lineage_raw).hexdigest()
    contract.validate_lineage_document(lineage, expected_index_sha256=actual_candidate_sha)
    return candidate, base, base_index_path.parent, {
        "candidate_index_sha256": actual_candidate_sha,
        "base_index_path": str(base_index_path.resolve()),
        "base_index_sha256": base_sha,
        "lineage_path": str(lineage_path.resolve()),
        "lineage_sha256": lineage_sha,
    }


def _compare_shared_assets(candidate: Mapping[str, Any], base: Mapping[str, Any],
                           candidate_dir: Path, base_dir: Path, *,
                           ledger: FileIdentityLedger) -> dict[str, Any]:
    reports: list[dict[str, Any]] = []
    for index, (candidate_entry, base_entry) in enumerate(
        zip(candidate["blocks"], base["blocks"], strict=True)
    ):
        manifest_name = base_entry["manifest"]
        candidate_manifest = candidate_dir / manifest_name
        base_manifest = base_dir / manifest_name
        ledger.assert_current(candidate_manifest, f"Q4 manifest block {index} candidate")
        ledger.assert_current(base_manifest, f"Q4 manifest block {index} base")
        payload_path, manifest_sha = contract._validate_q4_manifest(base_manifest, base_entry, index)
        ledger.assert_current(base_manifest, f"Q4 manifest block {index} base")
        payload_doc, _ = _read_json(base_manifest, f"base Q4 manifest block {index}", ledger=ledger)
        payload = payload_doc["payload"]
        candidate_payload = candidate_dir / payload["basename"]
        reports.append(_compare_files(
            candidate_manifest, base_manifest, f"Q4 manifest block {index}",
            expected_sha=manifest_sha, ledger=ledger,
        ))
        reports.append(_compare_files(
            candidate_payload, payload_path, f"Q4 payload block {index}",
            expected_sha=base_entry["payload_sha256"], expected_bytes=payload["nbytes"], ledger=ledger,
        ))

    globals_name = base["globals"]["bf16"]
    candidate_globals = candidate_dir / globals_name
    base_globals = base_dir / globals_name
    globals_header = _read_header(base_globals, "base globals BF16", ledger=ledger)
    if any(record.dtype != "BF16" for record in globals_header.tensors.values()):
        raise QualificationError("base globals cache contains a non-BF16 tensor")
    reports.append(_compare_files(
        candidate_globals, base_globals, "globals BF16", expected_sha=base["globals"]["sha256"], ledger=ledger,
    ))
    return {
        "assets_checked": len(reports),
        "assets_equal": all(item["bytes_equal"] for item in reports),
        "assets": reports,
    }


@contextmanager
def _tensor_file(path: Path, torch: Any, *, ledger: FileIdentityLedger | None = None,
                 label: str = "safetensors") -> Iterator[Any]:
    from safetensors import safe_open

    if ledger:
        ledger.assert_current(path, label)
    try:
        with safe_open(str(path), framework="pt", device="cpu") as handle:
            yield handle
    finally:
        if ledger:
            ledger.assert_current(path, label)


def _load_tensors(path: Path, keys: list[str], torch: Any, *,
                  ledger: FileIdentityLedger | None = None, label: str = "safetensors") -> dict[str, Any]:
    tensors: dict[str, Any] = {}
    with _tensor_file(path, torch, ledger=ledger, label=label) as handle:
        available = set(handle.keys())
        missing = sorted(set(keys) - available)
        if missing:
            raise QualificationError(f"{path} is missing requested tensors: {missing[:5]}")
        for key in keys:
            tensors[key] = handle.get_tensor(key).contiguous()
    return tensors


def _read_header(path: Path, label: str, *, ledger: FileIdentityLedger) -> Any:
    ledger.assert_current(path, label)
    header = contract.read_safetensors_header(path, label=label)
    ledger.assert_current(path, label)
    return header


def _ensure_finite(tensors: Mapping[str, Any], label: str) -> int:
    examined = 0
    for name, tensor in tensors.items():
        if not bool(tensor.is_floating_point()):
            raise QualificationError(f"{label} tensor is not floating point: {name}")
        if not bool(tensor.isfinite().all().item()):
            raise QualificationError(f"{label} contains a non-finite tensor value: {name}")
        examined += tensor.numel()
    return examined


def _word_view(tensor: Any, torch: Any) -> Any:
    return tensor.detach().contiguous().view(torch.uint16).numpy()


def _word_sha(words: Any) -> str:
    return hashlib.sha256(words.astype("<u2", copy=False).tobytes(order="C")).hexdigest()


def _compare_words(expected: Any, actual: Any, torch: Any) -> dict[str, Any]:
    expected_words = _word_view(expected, torch).reshape(-1)
    actual_words = _word_view(actual, torch).reshape(-1)
    if expected_words.shape != actual_words.shape:
        raise QualificationError("BF16 tensor word counts differ")
    mismatch_count = 0
    first: list[dict[str, Any]] = []
    for start in range(0, expected_words.size, COMPARE_CHUNK_WORDS):
        end = min(start + COMPARE_CHUNK_WORDS, expected_words.size)
        differences = expected_words[start:end] != actual_words[start:end]
        count = int(differences.sum())
        mismatch_count += count
        if count and len(first) < MAX_FIRST_MISMATCHES:
            for local in differences.nonzero()[0][:MAX_FIRST_MISMATCHES - len(first)]:
                flat_index = start + int(local)
                first.append({
                    "flat_index": flat_index,
                    "expected_word_hex": f"{int(expected_words[flat_index]):04x}",
                    "actual_word_hex": f"{int(actual_words[flat_index]):04x}",
                })
    return {
        "values_compared": int(expected_words.size),
        "mismatch_count": mismatch_count,
        "expected_words_sha256": _word_sha(expected_words),
        "actual_words_sha256": _word_sha(actual_words),
        "first_mismatches": first,
    }


def _import_pinned_merge(path: Path, *, ledger: FileIdentityLedger) -> Any:
    digest, size = _stable_hash(path, "upstream merge_lora.py", ledger=ledger)
    ledger.assert_current(path, "upstream merge_lora.py")
    source = path.read_bytes()
    if len(source) != size or hashlib.sha256(source).hexdigest() != digest:
        raise QualificationError("upstream merge_lora.py changed after hash verification")
    ledger.assert_current(path, "upstream merge_lora.py")
    if digest != PINNED_UPSTREAM_MERGE_SHA256:
        raise QualificationError("upstream merge_lora.py SHA-256 pin mismatch")
    module = types.ModuleType("hy_nomas_pinned_merge_lora")
    module.__file__ = str(path)
    exec(compile(source, str(path), "exec"), module.__dict__)
    if not callable(getattr(module, "merge", None)):
        raise QualificationError("pinned upstream source does not expose merge()")
    return module


def _merge_block_with_upstream(
    *, module: Any,
    base_source: Path,
    adapter_path: Path,
    adapter_config: Path,
    adapter_config_bytes: bytes,
    block_index: int,
    source_tensors: dict[str, Any],
    adapter_tensors: dict[str, Any],
    torch: Any,
    device: Any,
    ledger: FileIdentityLedger,
) -> dict[str, Any]:
    captured: dict[str, Any] = {}
    source_resolved = base_source.resolve(strict=True)
    adapter_resolved = adapter_path.resolve(strict=True)
    active_names = {stem + ".weight" for stem in module.collect_pairs(adapter_tensors)}
    if len(active_names) != len(contract.ACTIVE_TARGET_SHAPES):
        raise QualificationError(f"block {block_index} adapter subset is not 14 complete pairs")

    def controlled_load_file(path: str, device: str = "cpu") -> dict[str, Any]:
        resolved = Path(path).resolve(strict=True)
        if resolved == source_resolved:
            ledger.assert_current(base_source, "base FP32 source")
            return source_tensors
        if resolved == adapter_resolved:
            ledger.assert_current(adapter_path, "Cinematic adapter")
            return adapter_tensors
        raise QualificationError(f"upstream merge requested an unapproved safetensors file: {resolved}")

    def capture_save_file(tensors: Mapping[str, Any], filename: str, metadata: Any = None) -> None:
        if captured:
            raise QualificationError("upstream merge attempted more than one save")
        captured.update({
            key: value.detach().to(device="cpu", dtype=torch.float32).contiguous()
            for key, value in tensors.items() if key in active_names
        })
        Path(filename).write_bytes(b"")

    original_load_file = module.load_file
    original_save_file = module.save_file
    module.load_file = controlled_load_file
    module.save_file = capture_save_file
    try:
        ledger.assert_current(adapter_config, "adapter config")
        ledger.assert_current(base_source, "base FP32 source")
        ledger.assert_current(adapter_path, "Cinematic adapter")
        with tempfile.TemporaryDirectory(prefix=f"hy-nomas-reference-{block_index}-") as temporary:
            adapter_dir = Path(temporary) / "adapter"
            adapter_dir.mkdir()
            (adapter_dir / "adapter.safetensors").symlink_to(adapter_path)
            # The upstream function reads this file directly. Feed it the
            # already hash-verified bytes so a source-path swap cannot change
            # rank/alpha between identity checks and merge().
            (adapter_dir / "adapter_config.json").write_bytes(adapter_config_bytes)
            output_path = Path(temporary) / "merged.safetensors"
            module.merge(base_source, adapter_dir, output_path, str(device))
            output_path.unlink(missing_ok=True)
        ledger.assert_current(adapter_config, "adapter config")
        ledger.assert_current(base_source, "base FP32 source")
        ledger.assert_current(adapter_path, "Cinematic adapter")
    finally:
        module.load_file = original_load_file
        module.save_file = original_save_file

    expected_names = active_names
    if set(captured) != expected_names or len(captured) != len(contract.ACTIVE_TARGET_SHAPES):
        raise QualificationError(
            f"upstream merge captured the wrong target set for block {block_index}: "
            f"expected={len(expected_names)} actual={len(captured)}"
        )
    return captured


def _qualify_blocks(
    *, blocks: list[int], candidate: Mapping[str, Any], base: Mapping[str, Any],
    candidate_dir: Path, base_dir: Path, source_header: Any, adapter_path: Path,
    adapter_config: Path, adapter_config_bytes: bytes, base_source: Path,
    adapter_pairs: Mapping[str, Any], merge_module: Any,
    torch: Any, device: Any, ledger: FileIdentityLedger,
) -> dict[str, Any]:
    active_targets = set(contract.active_target_names())
    block_reports: list[dict[str, Any]] = []
    total_values = 0
    total_mismatches = 0
    for block_index in blocks:
        source_keys = sorted(contract._expected_block_names(source_header.tensors, block_index))
        active_names = sorted(
            target for target in active_targets if target.startswith(f"double_blocks.{block_index}.")
        )
        adapter_keys = sorted(
            key
            for target in active_names
            for key in (adapter_pairs[target].a_key, adapter_pairs[target].b_key)
        )
        source_tensors = _load_tensors(base_source, source_keys, torch,
                                       ledger=ledger, label="base FP32 source")
        adapter_tensors = _load_tensors(adapter_path, adapter_keys, torch,
                                        ledger=ledger, label="Cinematic adapter")
        source_finite = _ensure_finite(source_tensors, f"base block {block_index}")
        adapter_finite = _ensure_finite(adapter_tensors, f"adapter block {block_index}")
        for name in source_keys:
            record = source_header.tensors[name]
            if source_tensors[name].dtype != torch.float32 or tuple(source_tensors[name].shape) != record.shape:
                raise QualificationError(f"loaded FP32 source tensor dtype/shape changed: {name}")
        for target in active_names:
            pair = adapter_pairs[target]
            if tuple(adapter_tensors[pair.a_key].shape) != pair.a.shape or \
                    tuple(adapter_tensors[pair.b_key].shape) != pair.b.shape:
                raise QualificationError(f"loaded adapter tensor shape changed for {target}")

        candidate_entry = candidate["blocks"][block_index]
        base_entry = base["blocks"][block_index]
        candidate_block_path = candidate_dir / candidate_entry["bf16"]
        base_block_path = base_dir / base_entry["bf16"]
        candidate_block_sha, _ = _stable_hash(candidate_block_path,
            f"candidate BF16 block {block_index}", ledger=ledger)
        base_block_sha, _ = _stable_hash(base_block_path,
            f"base BF16 block {block_index}", ledger=ledger)
        if candidate_block_sha != candidate_entry["bf16_sha256"]:
            raise QualificationError(f"candidate BF16 block hash mismatch at block {block_index}")
        if base_block_sha != base_entry["bf16_sha256"]:
            raise QualificationError(f"base BF16 block hash mismatch at block {block_index}")
        candidate_header = _read_header(candidate_block_path,
            f"candidate BF16 block {block_index}", ledger=ledger)
        base_header = _read_header(base_block_path,
            f"base BF16 block {block_index}", ledger=ledger)
        contract._validate_bf16_block_header(candidate_header, source_header.tensors, block_index)
        contract._validate_bf16_block_header(base_header, source_header.tensors, block_index)

        merged = _merge_block_with_upstream(
            module=merge_module, base_source=base_source, adapter_path=adapter_path,
            adapter_config=adapter_config, adapter_config_bytes=adapter_config_bytes,
            block_index=block_index,
            source_tensors=source_tensors, adapter_tensors=adapter_tensors,
            torch=torch, device=device, ledger=ledger,
        )
        tensor_results: list[dict[str, Any]] = []
        block_values = 0
        block_mismatches = 0
        with _tensor_file(candidate_block_path, torch, ledger=ledger,
                           label=f"candidate BF16 block {block_index}") as candidate_file, \
                _tensor_file(base_block_path, torch, ledger=ledger,
                             label=f"base BF16 block {block_index}") as base_file:
            for name in source_keys:
                actual = candidate_file.get_tensor(name).contiguous()
                if actual.dtype != torch.bfloat16 or not bool(actual.isfinite().all().item()):
                    raise QualificationError(f"candidate BF16 tensor is non-BF16 or non-finite: {name}")
                if name in active_names:
                    merged_f32 = merged[name]
                    if not bool(merged_f32.isfinite().all().item()):
                        raise QualificationError(f"upstream merge produced non-finite F32 values: {name}")
                    expected = merged_f32.to(device=device, dtype=torch.bfloat16).to(device="cpu").contiguous()
                    expected = expected.detach()
                    if tuple(expected.shape) != tuple(actual.shape):
                        raise QualificationError(f"upstream merged BF16 shape differs for {name}")
                    compared = _compare_words(expected, actual, torch)
                    record = {
                        "tensor": name,
                        "kind": "upstream_merge_bf16",
                        "shape": list(actual.shape),
                        "expected_sha256": compared["expected_words_sha256"],
                        "candidate_sha256": compared["actual_words_sha256"],
                        "values_compared": compared["values_compared"],
                        "mismatch_count": compared["mismatch_count"],
                        "first_mismatches": compared["first_mismatches"],
                    }
                    del expected, merged_f32
                else:
                    reference = base_file.get_tensor(name).contiguous()
                    if reference.dtype != torch.bfloat16 or not bool(reference.isfinite().all().item()):
                        raise QualificationError(f"base BF16 tensor is non-BF16 or non-finite: {name}")
                    compared = _compare_words(reference, actual, torch)
                    record = {
                        "tensor": name,
                        "kind": "unchanged_base_bf16",
                        "shape": list(actual.shape),
                        "base_sha256": compared["expected_words_sha256"],
                        "candidate_sha256": compared["actual_words_sha256"],
                        "values_compared": compared["values_compared"],
                        "mismatch_count": compared["mismatch_count"],
                        "first_mismatches": compared["first_mismatches"],
                    }
                tensor_results.append(record)
                block_values += compared["values_compared"]
                block_mismatches += compared["mismatch_count"]
                del actual

        if len(tensor_results) != contract.BLOCK_TENSOR_COUNT:
            raise QualificationError(f"block {block_index} did not compare all 32 tensors")
        if sum(item["kind"] == "upstream_merge_bf16" for item in tensor_results) != len(active_names):
            raise QualificationError(f"block {block_index} did not compare all 14 active targets")
        total_values += block_values
        total_mismatches += block_mismatches
        block_reports.append({
            "block_index": block_index,
            "source_fp32_tensors_loaded": len(source_tensors),
            "adapter_fp32_tensors_loaded": len(adapter_tensors),
            "active_targets_merged": len(active_names),
            "unchanged_tensors_compared": contract.BLOCK_TENSOR_COUNT - len(active_names),
            "finite_source_values_checked": source_finite,
            "finite_adapter_values_checked": adapter_finite,
            "candidate_block_sha256": candidate_block_sha,
            "base_block_sha256": base_block_sha,
            "values_compared": block_values,
            "mismatch_count": block_mismatches,
            "tensors": tensor_results,
        })
        del source_tensors, adapter_tensors, merged
        try:
            torch.mps.empty_cache()
        except (AttributeError, RuntimeError):
            pass
    return {
        "blocks_requested": blocks,
        "blocks_unqualified": sorted(set(range(contract.BLOCK_COUNT)) - set(blocks)),
        "blocks_checked": block_reports,
        "values_compared": total_values,
        "mismatch_count": total_mismatches,
    }


def _write_report(path: Path, report: dict[str, Any], *, ledger: FileIdentityLedger) -> None:
    if path.name in ("", ".", ".."):
        raise QualificationError("--output must name a report file")
    parent = path.parent.resolve(strict=True)
    output = parent / path.name
    if os.path.lexists(output):
        raise FileExistsError(f"refusing to replace existing qualification report: {output}")
    raw = (json.dumps(report, indent=2, sort_keys=True) + "\n").encode("utf-8")
    ledger.assert_all_current("before qualification report publication")
    descriptor = os.open(output, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(raw)
            stream.flush()
            os.fsync(stream.fileno())
        ledger.assert_all_current("during qualification report publication")
    except Exception:
        output.unlink(missing_ok=True)
        raise


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-index", type=Path, required=True)
    parser.add_argument("--candidate-index-sha", required=True)
    parser.add_argument("--adapter", type=Path, required=True)
    parser.add_argument("--adapter-config", type=Path, required=True)
    parser.add_argument("--base-source", type=Path, required=True)
    parser.add_argument("--upstream-merge", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--blocks", default="0,53",
                        help="comma-separated blocks to qualify, or 'all' (default: 0,53)")
    return parser


def _absolute_argument(path: Path) -> Path:
    # Keep the final path component visible to lstat so input symlinks are
    # rejected instead of silently resolved before identity capture.
    return Path(os.path.abspath(os.fspath(path.expanduser())))


def main(argv: list[str] | None = None) -> int:
    args = make_parser().parse_args(argv)
    try:
        ledger = FileIdentityLedger()
        reference_tool_path = Path(__file__).resolve()
        reference_tool_sha, _ = _stable_hash(reference_tool_path, "reference tool", ledger=ledger)
        blocks = _parse_blocks(args.blocks)
        candidate_index_path = _absolute_argument(args.candidate_index)
        adapter_path = _absolute_argument(args.adapter)
        config_path = _absolute_argument(args.adapter_config)
        base_source = _absolute_argument(args.base_source)
        upstream_path = _absolute_argument(args.upstream_merge)

        candidate, base, base_dir, index_facts = _validate_index_pair(
            candidate_index_path, args.candidate_index_sha, base_source, ledger=ledger
        )
        source_header = _read_header(base_source, "base FP32 source", ledger=ledger)
        if source_header.file_bytes != contract.BASE_SOURCE_BYTES:
            raise QualificationError("base FP32 source byte pin mismatch")
        source_header_sha = hashlib.sha256(source_header.raw_header[8:]).hexdigest()
        if source_header_sha != contract.BASE_SOURCE_HEADER_SHA256:
            raise QualificationError("base FP32 source header pin mismatch")
        contract.validate_base_source_inventory(source_header.tensors)
        source_identity = _pinned_hash(
            base_source, "base FP32 source", contract.BASE_SOURCE_SHA256,
            contract.BASE_SOURCE_BYTES, ledger=ledger,
        )

        config, config_raw = _read_json(config_path, "adapter config", limit=16 * 1024,
                                        ledger=ledger)
        config_sha = hashlib.sha256(config_raw).hexdigest()
        contract.validate_config_document(
            config, actual_sha256=config_sha, actual_bytes=len(config_raw),
        )
        config_identity = {"path": str(config_path), "sha256": config_sha, "bytes": len(config_raw)}

        adapter_header = _read_header(adapter_path, "Cinematic adapter", ledger=ledger)
        if adapter_header.file_bytes != contract.ADAPTER_BYTES:
            raise QualificationError("adapter byte pin mismatch")
        adapter_pairs = contract.validate_adapter_inventory(adapter_header.tensors.items())
        adapter_identity = _pinned_hash(
            adapter_path, "Cinematic adapter", contract.ADAPTER_SHA256,
            contract.ADAPTER_BYTES, ledger=ledger,
        )
        adapter_identity["header_sha256"] = hashlib.sha256(adapter_header.raw_header[8:]).hexdigest()
        adapter_identity["tensor_count"] = len(adapter_header.tensors)
        adapter_identity["pair_count"] = len(adapter_pairs)

        upstream_identity = _pinned_hash(
            upstream_path, "upstream merge_lora.py", PINNED_UPSTREAM_MERGE_SHA256,
            ledger=ledger,
        )
        shared_assets = _compare_shared_assets(candidate, base, candidate_index_path.parent,
                                               base_dir, ledger=ledger)

        fallback = os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK")
        if fallback not in (None, "0"):
            raise QualificationError("PYTORCH_ENABLE_MPS_FALLBACK must be unset or 0")
        os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "0"
        import torch

        if not torch.backends.mps.is_built() or not torch.backends.mps.is_available():
            raise QualificationError("MPS is required; CPU fallback is forbidden")
        device = torch.device("mps")
        merge_module = _import_pinned_merge(upstream_path, ledger=ledger)
        result = _qualify_blocks(
            blocks=blocks, candidate=candidate, base=base,
            candidate_dir=candidate_index_path.parent, base_dir=base_dir,
            source_header=source_header, adapter_path=adapter_path,
            adapter_config=config_path, adapter_config_bytes=config_raw, base_source=base_source,
            adapter_pairs=adapter_pairs,
            merge_module=merge_module, torch=torch, device=device, ledger=ledger,
        )
        ledger.assert_all_current("after tensor qualification")
        report = {
            "schema": "cogni_ml.hy_nomas_cinematic_reference_qualification/v1",
            "status": "qualified" if result["mismatch_count"] == 0 else "mismatch",
            "claim_scope": (
                "selected block merge outputs and byte-identical unchanged cache controls; "
                "not full trajectory, decoder parity, or perceptual quality"
            ),
            "merge_contract": {
                "formula": "BF16_on_MPS(F32_base + (F32_B @ F32_A) * 2)",
                "upstream_order": "delta=(B@A)*scale; merged=base+delta; return F32; cast once to BF16 on MPS",
                "scale": contract.MERGE_SCALE,
                "active_targets_per_block": len(contract.ACTIVE_TARGET_SHAPES),
                "skipped_action_targets_total": contract.SKIPPED_ACTION_TARGET_COUNT,
                "skipped_action_target_names": contract.skipped_action_targets(),
                "action_conditioning": False,
                "base_dtype": "F32",
                "adapter_dtype": "F32",
                "output_dtype": "BF16",
            },
            "scope": result,
            "provenance": {
                **index_facts,
                "candidate_index_path": str(candidate_index_path),
                "candidate_index_sha256": args.candidate_index_sha,
                "base_source": source_identity,
                "base_source_header_sha256": source_header_sha,
                "adapter": {
                    "repo": contract.UPSTREAM_REPO,
                    "revision": contract.ADAPTER_REVISION,
                    "path": contract.ADAPTER_HUB_PATH,
                    **adapter_identity,
                    "config": config_identity,
                },
                "upstream_merge": {
                    "repo": contract.UPSTREAM_REPO,
                    "revision": contract.UPSTREAM_REVISION,
                    **upstream_identity,
                    "function": "merge()",
                    "device": "mps",
                },
                "mps": {
                    "device": str(device),
                    "torch_version": torch.__version__,
                    "is_built": bool(torch.backends.mps.is_built()),
                    "is_available": bool(torch.backends.mps.is_available()),
                    "fallback": os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK"),
                },
                "reference_tool_sha256": reference_tool_sha,
            },
            "shared_assets": shared_assets,
            "file_identities": ledger.report(),
        }
        _write_report(args.output, report, ledger=ledger)
        print(json.dumps({
            "status": report["status"],
            "blocks_checked": len(blocks),
            "blocks_unqualified": result["blocks_unqualified"],
            "values_compared": result["values_compared"],
            "mismatch_count": result["mismatch_count"],
            "report": str(args.output.resolve()),
        }, sort_keys=True))
        return 0 if result["mismatch_count"] == 0 else 1
    except (QualificationError, contract.CinematicCacheError, FileExistsError, OSError) as exc:
        print(f"hy_nomas_cinematic_reference: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
