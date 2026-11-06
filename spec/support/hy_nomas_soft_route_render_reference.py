#!/usr/bin/env python3
"""Authenticated four-step source oracle for the soft-route render.

The render replays the pinned BF16 MPS DiT from the accepted parent latent.
The conditioning anchor supplies only the action Qwen states; its separate
noise tensor is authenticated and explicitly excluded from the replay.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import math
import mmap
import os
from pathlib import Path
import sys
import tempfile
from typing import Any, Mapping

ROOT = Path(__file__).resolve().parents[2]
SUPPORT = ROOT / "spec" / "support"
if str(SUPPORT) not in sys.path:
    sys.path.insert(0, str(SUPPORT))

import hy_nomas_dit_route_factorial_reference as direct  # noqa: E402
import hy_nomas_soft_route_reference as soft  # noqa: E402
import hy_nomas_generation_inputs as generation_inputs  # noqa: E402

SCHEMA = "hy-nomas-soft-route-render-v1"
SCHEMA_VERSION = 1
NOISE_SCHEMA = "cogni_ml.hy_nomas_soft_route_render_noise_selection/v1"
NOISE_SELECTION_SHA256 = "8abfe96dc1a2b8ee3cd47f5bfac606c5f25630d95c637b535ec7811b122107ef"
ANCHOR_SHA256 = "4a7e2cc71586b0380735f6301c382efcae7ba7fc6ed3eaa0c43b37eb4b57c3c5"
ANCHOR_PAYLOAD_SHA256 = "44c8fc254429411f3c52ccf26a6eecc5101a2e16d67c979b90247d7178817208"
ANCHOR_QWEN_SHA256 = "673d17f11549d89bf5bb4010b22cd9e3e4923ed3071b55a0c7e7348c4e50faf9"
ANCHOR_UNUSED_LATENT_SHA256 = "61a5ed5ec852b8abf9c3d799df4fca420eae5e320adb7596d2da0a4c1670cf33"
PARENT_INITIAL_SHA256 = "503615dc8920afe80d441fa025380c4b764b1e81f07d3a44525127010ddb2e5a"
ACCEPTED_ZERO_STEP4_SHA256 = "a66c0c3e3de3f070b7f74ff54a4316b04c3860c2385efb830f15f17490561df5"
ROI_SOURCE_SHA256 = "ee7a830eb287cffd5d4ee2297d1ce5bd99e7f26a3dcc36442358c12f6ba72dd8"
DEFAULT_ANCHOR = Path(
    "/opt/mldata_ssd/cogni-ml/rolewright-sequential-actions-20261105.w7m2k9qa/anchor/conditioning/t1-inputs-anchor.json"
)
DEFAULT_PARENT = soft.DEFAULT_PARENT_REFERENCE
DEFAULT_BIAS = soft.DEFAULT_BIAS_MANIFEST
DEFAULT_NOISE_SELECTION = Path(
    "/opt/mldata_ssd/cogni-ml/hy-nomas-soft-routes-20261105.j0wgy7yl/render-v1-noise-selection.json"
)
DEFAULT_ZERO_STEP4 = Path(
    "/opt/mldata_ssd/cogni-ml/rolewright-sequential-actions-20261105.w7m2k9qa/roi/native/result/t1-step4.f32le"
)
DEFAULT_UPSTREAM = Path("/opt/mldata_ssd/models/hy-nomas/upstream-911cea6458a8")
DEFAULT_CACHE_INDEX = direct.DEFAULT_CACHE_INDEX
CASE_ORDER = ("zero", "A1_high", "A2_high")
LATENT_SHAPE = (1, 32, 30, 9, 16)
LATENT_BYTES = math.prod(LATENT_SHAPE) * 4
QWEN_SHAPE = (1, 1280, 3584)
QWEN_BYTES = math.prod(QWEN_SHAPE) * 2
TENSOR_SHAPES: dict[str, tuple[int, ...]] = {
    **{f"{case}.step{step}.latent": LATENT_SHAPE for case in CASE_ORDER for step in range(1, 5)},
    **{f"{case}.step1.velocity": LATENT_SHAPE for case in CASE_ORDER},
    "zero.step1.patch": (1, 4320, 2048),
    "zero.step1.text_valid": (1, 224, 2048),
    "zero.step1.vector": (1, 2048),
}
TENSOR_ORDER = tuple(
    [f"{case}.step{step}.latent" for case in CASE_ORDER for step in range(1, 5)]
    + [f"{case}.step1.velocity" for case in CASE_ORDER]
    + ["zero.step1.patch", "zero.step1.text_valid", "zero.step1.vector"]
)
ATOL = 2e-5
RTOL = 2e-5
STEPS = 4
SHIFT = 5.0
_HASH_CHUNK = 8 * 1024 * 1024


class RenderReferenceError(RuntimeError):
    """Fail-closed artifact, source, render, or replay admission error."""


def _require_legacy_cache_geometry(source_module: Any) -> dict[str, int]:
    """Require the legacy source runtime to use the admitted compact cache geometry."""
    expected = {
        "DIT_TEXT_ROWS": direct.TEXT_STORAGE_ROWS,
        "CACHE_CAPACITY": direct.PHYSICAL_CACHE_ROWS,
    }
    if direct.VIDEO_ROWS + direct.TEXT_STORAGE_ROWS != direct.PHYSICAL_CACHE_ROWS:
        raise RenderReferenceError("admitted compact cache geometry is internally inconsistent")
    observed = {name: getattr(source_module, name, None) for name in expected}
    if observed != expected:
        raise RenderReferenceError(
            "legacy source cache geometry mismatch: "
            f"expected={expected}, observed={observed}"
        )
    return expected


def _configure_legacy_cache_geometry(source_module: Any) -> dict[str, int]:
    """Set the pinned legacy runtime globals to the authenticated compact geometry."""
    if direct.VIDEO_ROWS + direct.TEXT_STORAGE_ROWS != direct.PHYSICAL_CACHE_ROWS:
        raise RenderReferenceError("admitted compact cache geometry is internally inconsistent")
    source_module.DIT_TEXT_ROWS = direct.TEXT_STORAGE_ROWS
    source_module.CACHE_CAPACITY = direct.PHYSICAL_CACHE_ROWS
    return _require_legacy_cache_geometry(source_module)


def _fresh_render_paths(output_manifest: Path) -> tuple[Path, Path]:
    """Admit only a canonical existing parent and two unclaimed output paths."""
    try:
        parent_dir = output_manifest.parent.resolve(strict=True)
        if output_manifest.parent != parent_dir or not parent_dir.is_dir():
            raise RenderReferenceError("report parent directory must be canonical and existing")
        manifest = parent_dir / output_manifest.name
        payload = parent_dir / f"{manifest.stem}.f32le"
        if os.path.lexists(manifest) or os.path.lexists(payload):
            raise RenderReferenceError("render report and payload must be new files")
        return manifest, payload
    except (OSError, RuntimeError) as error:
        raise RenderReferenceError(f"render output path invalid: {error}") from None


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(_HASH_CHUNK), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_file(value: str | Path, label: str) -> Path:
    path = Path(value).expanduser()
    try:
        resolved = path.resolve(strict=True)
    except OSError:
        raise RenderReferenceError(f"{label} does not exist: {path}") from None
    if path.is_symlink() or not resolved.is_file():
        raise RenderReferenceError(f"{label} must be a non-symlink regular file: {path}")
    return resolved


def _same_file(left: str | Path, right: str | Path, label: str) -> bool:
    try:
        return Path(left).expanduser().resolve(strict=True) == Path(right).expanduser().resolve(strict=True)
    except OSError:
        raise RenderReferenceError(f"{label} path is unavailable") from None


def _read_json(path: Path, label: str, expected_sha256: str | None = None) -> dict[str, Any]:
    if expected_sha256 is not None and _sha256_file(path) != expected_sha256:
        raise RenderReferenceError(f"{label} SHA-256 mismatch")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise RenderReferenceError(f"{label} JSON could not be read: {error}") from None
    if type(value) is not dict:
        raise RenderReferenceError(f"{label} must be a JSON object")
    return value


def _tensor_record(document: Mapping[str, Any], name: str, label: str) -> dict[str, Any]:
    records = document.get("tensors")
    record = records.get(name) if type(records) is dict else None
    if type(record) is not dict:
        raise RenderReferenceError(f"{label} tensor record missing: {name}")
    return record


def _validate_noise_selection(
    path: Path, expected_sha256: str, anchor_path: Path, anchor_document: Mapping[str, Any],
    parent_path: Path, parent: Mapping[str, Any], admitted: Mapping[str, Any],
) -> dict[str, Any]:
    if expected_sha256 != NOISE_SELECTION_SHA256:
        raise RenderReferenceError("noise-selection SHA differs from the frozen CPU comparison")
    document = _read_json(path, "noise-selection record", NOISE_SELECTION_SHA256)
    if set(document) != {"schema", "anchor", "cpu_comparison", "decision", "selected_parent"}:
        raise RenderReferenceError("noise-selection record keyset mismatch")
    if document.get("schema") != NOISE_SCHEMA:
        raise RenderReferenceError("noise-selection schema mismatch")
    anchor = document["anchor"]
    if (type(anchor) is not dict or set(anchor) != {
            "manifest_path", "manifest_sha256", "payload_path", "payload_realpath", "payload_sha256",
            "payload_bytes", "latent_record", "dtype", "shape", "offset_bytes", "nbytes", "sha256",
            "bf16_rne_expanded_sha256"}
            or not _same_file(anchor.get("manifest_path", ""), anchor_path, "noise-selection anchor")
            or anchor.get("manifest_sha256") != ANCHOR_SHA256
            or not _same_file(anchor.get("payload_path", ""),
                              anchor_path.parent / anchor_document["payload"],
                              "noise-selection anchor payload")
            or anchor.get("payload_sha256") != ANCHOR_PAYLOAD_SHA256
            or anchor.get("payload_bytes") != anchor_document["payload_bytes"]
            or anchor.get("latent_record") != "input.latent" or anchor.get("dtype") != "F32"
            or anchor.get("shape") != list(LATENT_SHAPE) or anchor.get("offset_bytes") != 0
            or anchor.get("nbytes") != LATENT_BYTES or anchor.get("sha256") != ANCHOR_UNUSED_LATENT_SHA256
            or anchor.get("bf16_rne_expanded_sha256") != "397957a4cf3095bce530f712dda87ba9f1b16b6322447509972f9cfd0b54569b"):
        raise RenderReferenceError("noise-selection anchor input binding mismatch")
    decision = document["decision"]
    expected_decision = {
        "anchor_input_latent_used": False,
        "authoritative_initial_latent": "authenticated direct-route parent t1.initial_latent",
        "reason": "anchor input.latent is a different noise tensor; neither raw F32 nor BF16 RNE expansion matches the accepted parent initial latent",
        "render_contract": "use the parent latent for matched zero/A1_high/A2_high replay; bind the conditioning anchor only for Qwen and record its unused latent explicitly",
        "selected_record": "t1.initial_latent",
        "selected_source_dtype": "BF16",
        "selected_storage_dtype": "F32 lossless BF16 expansion",
    }
    if decision != expected_decision:
        raise RenderReferenceError("noise-selection decision mismatch")
    selected = document["selected_parent"]
    parent_initial = admitted["inputs"]["t1.initial_latent"]
    if (type(selected) is not dict or set(selected) != {
            "source_reference_path", "source_reference_sha256", "frozen_inputs_path", "frozen_inputs_sha256",
            "record", "path", "realpath", "shape", "dtype", "source_dtype", "file_offset_bytes",
            "file_nbytes", "file_sha256", "accepted_source_range_offset_bytes",
            "accepted_source_range_nbytes", "accepted_source_range_sha256", "matches_accepted_source_record"}
            or selected.get("source_reference_sha256") != soft.PARENT_SHA256
            or not _same_file(selected.get("source_reference_path", ""), parent_path, "noise-selection parent")
            or not _same_file(selected.get("frozen_inputs_path", ""), admitted["frozen_inputs_path"], "noise-selection frozen inputs")
            or selected.get("frozen_inputs_sha256") != admitted["frozen_inputs_sha256"]
            or selected.get("record") != "t1.initial_latent"
            or not _same_file(selected.get("path", ""), parent_initial["path"], "selected parent initial latent")
            or not _same_file(selected.get("realpath", ""), parent_initial["path"], "selected parent latent realpath")
            or selected.get("shape") != list(LATENT_SHAPE)
            or selected.get("dtype") != "F32 lossless expansion of BF16"
            or selected.get("source_dtype") != "BF16"
            or selected.get("file_offset_bytes") != 0 or selected.get("file_nbytes") != LATENT_BYTES
            or selected.get("file_sha256") != PARENT_INITIAL_SHA256
            or selected.get("accepted_source_range_offset_bytes") != direct.INPUT_SOURCE_RANGES["t1.initial_latent"][0]
            or selected.get("accepted_source_range_nbytes") != LATENT_BYTES
            or selected.get("accepted_source_range_sha256") != PARENT_INITIAL_SHA256
            or selected.get("matches_accepted_source_record") is not True):
        raise RenderReferenceError("noise-selection chosen parent latent binding mismatch")
    comparison = document["cpu_comparison"]
    if (type(comparison) is not dict or comparison.get("element_count") != math.prod(LATENT_SHAPE)
            or comparison.get("anchor_raw_f32_vs_selected_parent_bit_differences") != math.prod(LATENT_SHAPE)
            or comparison.get("anchor_bf16_rne_expanded_vs_selected_parent_bit_differences") != 138117
            or comparison.get("large_tensor_copy_saved") is not False):
        raise RenderReferenceError("noise-selection CPU comparison evidence mismatch")
    return document


def _verify_noise_difference(anchor_path: Path, anchor_document: Mapping[str, Any],
                              selected_path: Path) -> dict[str, Any]:
    """Recompute the selected-noise distinction using only the 138240-element CPU ranges."""
    import numpy as np

    anchor_record = anchor_document["tensors"]["input.latent"]
    raw_anchor = np.memmap(anchor_path.parent / anchor_document["payload"], mode="r", dtype="<f4",
                           offset=anchor_record["offset"], shape=LATENT_SHAPE)
    chosen = np.memmap(selected_path, mode="r", dtype="<f4", shape=LATENT_SHAPE)
    anchor_bits = np.asarray(raw_anchor).view(np.uint32)
    chosen_bits = np.asarray(chosen).view(np.uint32)
    rne_bits = ((anchor_bits.astype(np.uint64) + 0x7FFF + ((anchor_bits >> 16) & 1)) & 0xFFFF0000).astype(np.uint32)
    rne = rne_bits.view(np.float32)
    raw_diff = int(np.count_nonzero(anchor_bits != chosen_bits))
    rounded_diff = int(np.count_nonzero(rne_bits != chosen_bits))
    if raw_diff != math.prod(LATENT_SHAPE) or rounded_diff != 138117:
        raise RenderReferenceError("CPU noise-selection mismatch counts differ from frozen evidence")
    if _array_sha(rne) != "397957a4cf3095bce530f712dda87ba9f1b16b6322447509972f9cfd0b54569b":
        raise RenderReferenceError("CPU BF16-RNE-expanded anchor digest mismatch")
    del raw_anchor, chosen
    return {"anchor_raw_vs_parent_bit_differences": raw_diff,
            "anchor_bf16_rne_vs_parent_bit_differences": rounded_diff}


def _array_sha(array: Any) -> str:
    import numpy as np
    return hashlib.sha256(np.asarray(array).astype("<f4", copy=False).tobytes(order="C")).hexdigest()


def authenticate_inputs(
    anchor_path: str | Path = DEFAULT_ANCHOR, anchor_sha256: str = ANCHOR_SHA256,
    parent_path: str | Path = DEFAULT_PARENT, parent_sha256: str = soft.PARENT_SHA256,
    bias_path: str | Path = DEFAULT_BIAS, bias_sha256: str = soft.BIAS_SHA256,
    noise_selection_path: str | Path = DEFAULT_NOISE_SELECTION,
    noise_selection_sha256: str = NOISE_SELECTION_SHA256,
    zero_step4_path: str | Path = DEFAULT_ZERO_STEP4,
) -> dict[str, Any]:
    """Authenticate four source inputs and the accepted zero-render target."""
    if anchor_sha256 != ANCHOR_SHA256 or parent_sha256 != soft.PARENT_SHA256:
        raise RenderReferenceError("anchor or direct parent SHA differs from the pinned source lineage")
    try:
        anchor_manifest = direct._require_file_digest(anchor_path, ANCHOR_SHA256, "T1 conditioning anchor")
        anchor_document = direct._read_json(anchor_manifest, "T1 conditioning anchor")
        generation_inputs.validate_document(anchor_document, payload_bytes=anchor_document.get("payload_bytes"))
        if (anchor_document.get("schema") != generation_inputs.DYNAMIC_SCHEMA
                or anchor_document.get("schema_version") != generation_inputs.DYNAMIC_SCHEMA_VERSION
                or anchor_document.get("upstream_revision") != direct.PINNED_REVISION
                or anchor_document.get("geometry", {}).get("qwen_valid_tokens") != direct.T1_VALID_TEXT_ROWS
                or anchor_document.get("geometry", {}).get("dit_latent_shape") != list(LATENT_SHAPE)):
            raise RenderReferenceError("T1 anchor schema, source revision, or geometry mismatch")
        anchor_payload = direct._require_file_digest(
            anchor_manifest.parent / anchor_document["payload"], ANCHOR_PAYLOAD_SHA256,
            "T1 conditioning anchor payload",
        )
        with anchor_payload.open("rb") as stream:
            with mmap.mmap(stream.fileno(), 0, access=mmap.ACCESS_READ) as mapped:
                payload_view = memoryview(mapped)
                try:
                    generation_inputs.validate_payload(anchor_document, payload_view)
                finally:
                    payload_view.release()
        anchor_qwen = _tensor_record(anchor_document, "input.qwen_bf16", "T1 anchor")
        anchor_latent = _tensor_record(anchor_document, "input.latent", "T1 anchor")
        if (anchor_qwen != {"dtype": "BF16", "shape": list(QWEN_SHAPE), "offset": LATENT_BYTES,
                            "nbytes": QWEN_BYTES, "sha256": ANCHOR_QWEN_SHA256}
                or anchor_latent != {"dtype": "F32", "shape": list(LATENT_SHAPE), "offset": 0,
                                     "nbytes": LATENT_BYTES, "sha256": ANCHOR_UNUSED_LATENT_SHA256}):
            raise RenderReferenceError("T1 anchor Qwen or unused latent record mismatch")

        parent_manifest = soft.authenticate_parent_reference(parent_path, parent_sha256)
        admitted = direct.authenticate_inputs()
        parent_roi = parent_manifest["document"].get("source_inputs", {}).get("roi", {})
        if (parent_roi.get("manifest_sha256") != direct.ROI_MANIFEST_SHA256
                or not _same_file(parent_roi.get("manifest_path", ""), admitted["roi_manifest_path"],
                                  "direct parent accepted ROI source")):
            raise RenderReferenceError("direct parent is not bound to the accepted ROI source")
        parent_document = parent_manifest["document"]
        if parent_document.get("schema") != direct.SCHEMA:
            raise RenderReferenceError("direct route parent schema mismatch")
        parent_initial = admitted["inputs"]["t1.initial_latent"]
        if parent_initial["sha256"] != PARENT_INITIAL_SHA256:
            raise RenderReferenceError("accepted parent initial latent digest mismatch")
        bias = soft.authenticate_bias_manifest(bias_path, bias_sha256)
        support = soft.validate_bias_support(bias["vectors"])

        noise_path = direct._require_file_digest(noise_selection_path, NOISE_SELECTION_SHA256,
                                                 "noise-selection CPU evidence")
        noise = _validate_noise_selection(noise_path, noise_selection_sha256, anchor_manifest,
                                          anchor_document, parent_manifest["path"], parent_document, admitted)
        noise_recomputed = _verify_noise_difference(anchor_manifest, anchor_document, parent_initial["path"])

        baseline = _canonical_file(zero_step4_path, "accepted zero step-4 baseline")
        baseline_sha = _sha256_file(baseline)
        if baseline_sha != ACCEPTED_ZERO_STEP4_SHA256 or baseline.stat().st_size != LATENT_BYTES:
            raise RenderReferenceError("accepted zero step-4 baseline digest or byte count mismatch")
    except (direct.ReferenceError, soft.SoftRouteError, generation_inputs.InputsError) as error:
        raise RenderReferenceError(str(error)) from None

    return {
        "anchor_path": str(anchor_manifest), "anchor_sha256": ANCHOR_SHA256,
        "anchor_document": anchor_document, "anchor_payload_path": str(anchor_payload),
        "anchor_payload_sha256": ANCHOR_PAYLOAD_SHA256,
        "anchor_qwen_record": anchor_qwen, "anchor_unused_latent_record": anchor_latent,
        "parent": parent_manifest, "parent_document": parent_document,
        "admitted": admitted, "parent_initial": parent_initial,
        "bias": bias, "bias_support": support,
        "noise_selection_path": str(noise_path), "noise_selection": noise,
        "noise_recomputed": noise_recomputed,
        "zero_step4_path": str(baseline), "zero_step4_sha256": baseline_sha,
        "zero_step4_bytes": baseline.stat().st_size,
    }


def _input_document(checked: Mapping[str, Any]) -> dict[str, Any]:
    parent = checked["parent"]
    admitted = checked["admitted"]
    initial = checked["parent_initial"]
    qwen = checked["anchor_qwen_record"]
    latent = checked["anchor_unused_latent_record"]
    return {
        "anchor": {
            "path": checked["anchor_path"], "sha256": checked["anchor_sha256"],
            "payload_path": checked["anchor_payload_path"], "payload_bytes": checked["anchor_document"]["payload_bytes"],
            "payload_sha256": checked["anchor_payload_sha256"],
            "qwen_record": {"name": "input.qwen_bf16", **qwen},
            "unused_latent": {"name": "input.latent", **latent, "used": False},
        },
        "parent": {
            "path": parent["path"], "sha256": parent["sha256"],
            "payload_path": parent["payload_path"],
            "payload_bytes": parent["document"]["payload"]["bytes"],
            "payload_sha256": parent["document"]["payload"]["sha256"],
            "frozen_inputs_path": admitted["frozen_inputs_path"],
            "frozen_inputs_sha256": admitted["frozen_inputs_sha256"],
            "selected_initial_latent": {
                "record": "t1.initial_latent", "path": initial["path"], "dtype": "F32",
                "source_dtype": "BF16", "shape": initial["shape"], "offset_bytes": 0,
                "nbytes": initial["bytes"], "sha256": initial["sha256"],
                "source_range_offset_bytes": initial["source_range_offset"],
                "source_range_bytes": initial["source_range_bytes"],
            },
            "accepted_roi_manifest_path": admitted["roi_manifest_path"],
            "accepted_roi_manifest_sha256": admitted["roi_manifest_sha256"],
            "accepted_roi_payload_path": admitted["roi_payload_path"],
            "accepted_roi_payload_sha256": admitted["roi_payload_sha256"],
        },
        "bias_manifest": {
            "path": checked["bias"]["path"], "sha256": checked["bias"]["sha256"],
            "schema": soft.BIAS_SCHEMA, "case_order": list(soft.CASE_ORDER),
            "reference_image": checked["bias"]["document"]["reference_image"],
            "maps": checked["bias"]["document"]["maps"],
            "bias_f32le_sha256": checked["bias"]["document"]["bias_f32le_sha256"],
        },
        "noise_selection": {
            "path": checked["noise_selection_path"], "sha256": NOISE_SELECTION_SHA256,
            "selected_record": "t1.initial_latent", "anchor_input_latent_used": False,
            "cpu_comparison": checked["noise_selection"]["cpu_comparison"],
            "recomputed": checked["noise_recomputed"],
        },
        "accepted_zero_step4": {
            "path": checked["zero_step4_path"], "sha256": checked["zero_step4_sha256"],
            "bytes": checked["zero_step4_bytes"], "dtype": "F32 lossless expansion of BF16",
            "source_dtype": "BF16", "shape": list(LATENT_SHAPE),
        },
    }


def _comparison_record(label: str, expected: Any, actual: Any) -> dict[str, Any]:
    import numpy as np

    expected_f32 = np.asarray(expected, dtype=np.float32)
    actual_f32 = np.asarray(actual, dtype=np.float32)
    if (expected_f32.shape != actual_f32.shape or not np.isfinite(expected_f32).all()
            or not np.isfinite(actual_f32).all()):
        raise RenderReferenceError(f"source output shape/nonfinite mismatch: {label}")
    bitwise_equal = np.array_equal(expected_f32.view(np.uint32), actual_f32.view(np.uint32))
    left = expected_f32.astype(np.float64)
    right = actual_f32.astype(np.float64)
    delta = np.abs(left - right)
    outside = int(np.count_nonzero(delta > ATOL + RTOL * np.abs(left)))
    return {"checkpoint": label, "elements": int(delta.size), "max_abs": float(delta.max(initial=0.0)),
            "outside_atol_rtol": outside, "bitwise_equal": bool(bitwise_equal),
            "passed": outside == 0}


def validate_captures(captures: Mapping[str, Any], *, cases: tuple[str, ...] = CASE_ORDER) -> dict[str, Any]:
    import numpy as np

    if not cases or len(set(cases)) != len(cases) or any(case not in CASE_ORDER for case in cases):
        raise RenderReferenceError("source render case selection mismatch")
    order = tuple(name for name in TENSOR_ORDER if name.split(".")[0] in cases
                  or name in ("zero.step1.patch", "zero.step1.text_valid", "zero.step1.vector"))
    if type(captures) is not dict or set(captures) != set(order):
        raise RenderReferenceError("source render capture inventory mismatch")
    result = {}
    for name in order:
        value = np.asarray(captures[name])
        if (value.shape != TENSOR_SHAPES[name] or value.dtype != np.dtype("float32")
                or not np.isfinite(value).all()):
            raise RenderReferenceError(f"source render capture geometry/dtype/nonfinite mismatch: {name}")
        result[name] = value
    return result


def _check_source_replay(checked: Mapping[str, Any], captures: Mapping[str, Any]) -> dict[str, Any]:
    import numpy as np

    records = checked["admitted"]["parent_checkpoints"]
    roi_payload = Path(checked["admitted"]["roi_payload_path"])
    gates: dict[str, Any] = {}
    direct_parent_names = {
        "zero.step1.velocity": "t1.step1.velocity",
        "zero.step1.patch": "t1.step1.patch",
        "zero.step1.text_valid": "t1.step1.text_valid",
        "zero.step1.vector": "t1.step1.vector",
    }
    roi_records = checked["admitted"]["roi_document"]["capture_payload"]["records"]
    by_name = {item["name"]: item for item in roi_records}
    for output_name, source_name in direct_parent_names.items():
        record = by_name.get(source_name)
        if type(record) is not dict:
            raise RenderReferenceError(f"accepted ROI step-1 record missing: {source_name}")
        source = np.memmap(roi_payload, mode="r", dtype="<f4", offset=record["offset"],
                           shape=tuple(record["shape"]))
        actual = captures[output_name]
        stats = _comparison_record(f"zero:{source_name}", source, actual)
        if not stats["bitwise_equal"]:
            raise RenderReferenceError(f"zero source step-1 replay is not bitwise exact: {source_name}")
        gates[output_name] = stats
        del source
    # The accepted final image alone cannot prove that the zero arm followed
    # the same four-step trajectory: an intermediate mismatch could cancel
    # later. Gate each source Euler boundary against the accepted ROI capture.
    for step in range(1, STEPS + 1):
        source_name = f"t1.step{step}.after_euler"
        output_name = f"zero.step{step}.latent"
        record = by_name.get(source_name)
        if (type(record) is not dict or record.get("shape") != list(LATENT_SHAPE)
                or record.get("nbytes") != LATENT_BYTES or record.get("source_dtype") != "BF16"
                or type(record.get("offset")) is not int or record.get("offset", -1) < 0):
            raise RenderReferenceError(f"accepted ROI Euler-boundary record missing or malformed: {source_name}")
        source = np.memmap(roi_payload, mode="r", dtype="<f4", offset=record["offset"],
                           shape=tuple(record["shape"]))
        stats = _comparison_record(f"zero:{source_name}", source, captures[output_name])
        if not stats["bitwise_equal"]:
            raise RenderReferenceError(f"zero source Euler-boundary replay is not bitwise exact: {source_name}")
        gates[output_name] = stats
        del source
    parent_source_payload = Path(checked["parent"]["payload_path"])
    parent_velocity = checked["parent_document"]["tensors"]["M11.velocity"]
    source_velocity = np.memmap(
        parent_source_payload, mode="r", dtype="<f4", offset=parent_velocity["offset_bytes"],
        shape=tuple(parent_velocity["shape"]),
    )
    velocity_stats = _comparison_record("zero:direct_parent.M11.velocity", source_velocity,
                                        captures["zero.step1.velocity"])
    if not velocity_stats["bitwise_equal"]:
        raise RenderReferenceError("zero step-1 velocity differs from authenticated direct parent source")
    gates["direct_parent.M11.velocity"] = velocity_stats
    del source_velocity
    baseline = np.memmap(checked["zero_step4_path"], mode="r", dtype="<f4", shape=LATENT_SHAPE)
    final = captures["zero.step4.latent"]
    final_stats = _comparison_record("zero:accepted_step4", baseline, final)
    if not final_stats["bitwise_equal"]:
        raise RenderReferenceError("zero source step-4 replay is not bitwise exact to accepted baseline")
    gates["zero.step4.latent"] = final_stats
    del baseline
    return gates


def _self_test_source_replay_gate(checked: Mapping[str, Any]) -> None:
    """Exercise positive accepted replay and a seeded one-bit trajectory failure on CPU."""
    import numpy as np

    roi_payload = Path(checked["admitted"]["roi_payload_path"])
    by_name = {item["name"]: item
               for item in checked["admitted"]["roi_document"]["capture_payload"]["records"]}
    source_names = {
        "zero.step1.velocity": "t1.step1.velocity",
        "zero.step1.patch": "t1.step1.patch",
        "zero.step1.text_valid": "t1.step1.text_valid",
        "zero.step1.vector": "t1.step1.vector",
        **{f"zero.step{step}.latent": f"t1.step{step}.after_euler"
           for step in range(1, STEPS + 1)},
    }
    captures: dict[str, Any] = {}
    for output_name, source_name in source_names.items():
        record = by_name.get(source_name)
        if type(record) is not dict:
            raise RenderReferenceError(f"CPU replay-gate fixture is missing {source_name}")
        captures[output_name] = np.memmap(
            roi_payload, mode="r", dtype="<f4", offset=record["offset"], shape=tuple(record["shape"]),
        )
    _check_source_replay(checked, captures)

    corrupted = dict(captures)
    corrupted_latent = np.asarray(captures["zero.step2.latent"]).copy()
    corrupted_latent.view(np.uint32).reshape(-1)[0] ^= 1
    corrupted["zero.step2.latent"] = corrupted_latent
    try:
        _check_source_replay(checked, corrupted)
    except RenderReferenceError as error:
        if "Euler-boundary replay is not bitwise exact" not in str(error):
            raise RenderReferenceError(f"CPU replay-gate falsifier failed for the wrong reason: {error}") from None
    else:
        raise RenderReferenceError("CPU replay-gate falsifier accepted a one-bit step-2 latent change")


def _device_masks(torch: Any, source_masks: Mapping[str, Any], biases: Mapping[str, Any], device: Any) -> dict[str, list[Any]]:
    import numpy as np

    if set(source_masks) != set(direct.CASE_ORDER) or "M11" not in source_masks:
        raise RenderReferenceError("pinned route mask inventory mismatch")
    source = source_masks["M11"]
    if len(source) != 18:
        raise RenderReferenceError("pinned M11 mask chunk count mismatch")
    zero: list[Any] = []
    for index, chunk in enumerate(source):
        expected_rows = min(256, direct.QUERY_ROWS - index * 256)
        if (tuple(chunk.shape) != (expected_rows, direct.KEY_ROWS) or chunk.dtype != torch.bool
                or chunk.device.type != "cpu"):
            raise RenderReferenceError("pinned M11 mask chunk geometry mismatch")
        zero.append(chunk.to(device=device))
    result: dict[str, list[Any]] = {"zero": zero}
    for case, action in (("A1_high", "A1"), ("A2_high", "A2")):
        vector = np.asarray(biases[case], dtype=np.float32)
        chunks = []
        for index, chunk in enumerate(source):
            try:
                additive = __import__("hy_nomas_dit_route_factorial_runtime")._soft_route_additive_chunk(
                    torch, chunk, vector, action, index * 256,
                )
            except Exception as error:
                raise RenderReferenceError(f"source soft-route mask construction failed: {case}: {error}") from None
            chunks.append(additive.to(device=device))
        result[case] = chunks
    return result


def _dispatch_factory(torch: Any, state: dict[str, Any], masks: Mapping[str, Any], legacy_factory: Any) -> Any:
    legacy = legacy_factory(torch, state)

    def dispatch(q_full: Any, full_k: Any, full_v: Any, q_mask: Any,
                 kv_mask: Any, cache_valid_len: int, update_cache: bool,
                 cache_txt: bool) -> Any:
        call = state["call"]
        if call["condition"] == "t0_caption":
            state["receipt"]["t0_attention_calls"] += 1
            return legacy(q_full, full_k, full_v, q_mask, kv_mask,
                          cache_valid_len, update_cache, cache_txt)
        if (call["condition"] != "t1_action" or call["valid_text_tokens"] != direct.T1_VALID_TEXT_ROWS
                or cache_valid_len != direct.PHYSICAL_CACHE_ROWS or update_cache is not False
                or cache_txt is not True or tuple(q_full.shape) != (1, direct.PHYSICAL_CACHE_ROWS, direct.HEADS, direct.HEAD_DIM)
                or tuple(full_k.shape) != (1, 2 * direct.PHYSICAL_CACHE_ROWS, direct.HEADS, direct.HEAD_DIM)
                or tuple(full_v.shape) != tuple(full_k.shape)
                or tuple(q_mask.shape) != (1, direct.PHYSICAL_CACHE_ROWS)
                or tuple(kv_mask.shape) != (1, 2 * direct.PHYSICAL_CACHE_ROWS)
                or any(t.device.type != "mps" or t.dtype != torch.bfloat16 for t in (q_full, full_k, full_v))
                or any(t.device != q_full.device or t.dtype != torch.bool for t in (q_mask, kv_mask))):
            raise RenderReferenceError("render T1 cache attention escaped pinned BF16 geometry")
        q = q_full[:, :direct.QUERY_ROWS].permute(0, 2, 1, 3).contiguous()
        compact_k = torch.cat((full_k[:, :direct.HISTORY_ROWS],
                               full_k[:, direct.PHYSICAL_CACHE_ROWS:direct.PHYSICAL_CACHE_ROWS + direct.QUERY_ROWS]), dim=1)
        compact_v = torch.cat((full_v[:, :direct.HISTORY_ROWS],
                               full_v[:, direct.PHYSICAL_CACHE_ROWS:direct.PHYSICAL_CACHE_ROWS + direct.QUERY_ROWS]), dim=1)
        keys = compact_k.permute(0, 2, 1, 3).contiguous()
        values = compact_v.permute(0, 2, 1, 3).contiguous()
        result = torch.zeros(tuple(q_full.shape), device=q_full.device, dtype=torch.bfloat16)
        case = state["case"]
        chunks = masks[case]
        for index, start in enumerate(range(0, direct.QUERY_ROWS, 256)):
            stop = min(start + 256, direct.QUERY_ROWS)
            attended = torch.nn.functional.scaled_dot_product_attention(
                q[:, :, start:stop], keys, values,
                attn_mask=chunks[index][None, None, :, :], dropout_p=0.0, is_causal=False,
            )
            result[:, start:stop].copy_(attended.permute(0, 2, 1, 3))
        receipt = state["receipt"]
        receipt["attention_calls"] += 1
        receipt["t1_attention_calls"] += 1
        receipt["attention_query_chunks"] += len(chunks)
        return result

    return dispatch


def _load_f32_input(torch: Any, np: Any, record: Mapping[str, Any], device: Any) -> Any:
    values = np.fromfile(record["path"], dtype="<f4")
    if values.size * 4 != record["bytes"]:
        raise RenderReferenceError("authenticated parent tensor changed extent")
    return torch.from_numpy(values.reshape(record["shape"])).to(device=device, dtype=torch.bfloat16)


def _load_anchor_qwen(torch: Any, np: Any, checked: Mapping[str, Any], device: Any) -> Any:
    record = checked["anchor_qwen_record"]
    payload = Path(checked["anchor_payload_path"])
    raw = np.memmap(payload, mode="r", dtype="<u2", offset=record["offset"], shape=QWEN_SHAPE)
    copied = np.array(raw, copy=True)
    del raw
    return torch.from_numpy(copied).view(torch.bfloat16).to(device=device, dtype=torch.bfloat16)


def _run_source(checked: Mapping[str, Any], model: Mapping[str, Any], upstream_path: Path,
                *, cases: tuple[str, ...] = CASE_ORDER, observer: Any = None) -> tuple[dict[str, Any], dict[str, Any]]:
    """Run four Euler steps with a single staged BF16 block and one layer cache."""
    import numpy as np

    if not cases or len(set(cases)) != len(cases) or any(case not in CASE_ORDER for case in cases):
        raise RenderReferenceError("source render case selection mismatch")
    # The existing CLI and publisher retain the full three-arm contract. A
    # diagnostic caller may replay a subset without changing its equations.
    first_case = cases[0]
    support_path = Path("/opt/mldata_ssd/cogni-ml/hy-nomas-final-euler-20261104.c7zj7lnM")
    if str(support_path) not in sys.path:
        sys.path.insert(0, str(support_path))
    try:
        source = importlib.import_module("replay_source_autocast_full_base")
    except Exception as error:
        raise RenderReferenceError(f"pinned source runtime import failed: {type(error).__name__}: {error}") from None
    legacy_geometry = _configure_legacy_cache_geometry(source)
    source.single._configure_mps_runtime()
    torch, scheduler, diffusers_version = source.single._prepare_cpu_scheduler(upstream_path, "source-bf16")
    if os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK") != "0":
        raise RenderReferenceError("MPS fallback must be explicitly disabled")
    if not torch.backends.mps.is_built() or not torch.backends.mps.is_available():
        raise RenderReferenceError("MPS is required; CPU model fallback is forbidden")
    if (scheduler.timesteps.device.type != "cpu" or scheduler.sigmas.device.type != "cpu"
            or len(scheduler.timesteps) != STEPS or len(scheduler.sigmas) != STEPS + 1):
        raise RenderReferenceError("pinned four-step Euler schedule geometry mismatch")
    times = scheduler.timesteps.detach().cpu().numpy().astype("<f4", copy=False)
    deltas = (scheduler.sigmas[1:] - scheduler.sigmas[:-1]).detach().cpu().numpy().astype("<f4", copy=False)
    if (not np.array_equal(times, np.asarray([1000.0, 937.5, 833.333313, 625.0], dtype="<f4"))
            or deltas[0] != np.float32(-0.0625)):
        raise RenderReferenceError("pinned source Euler schedule differs from the accepted four-step schedule")

    device, dtype = torch.device("mps"), torch.bfloat16
    timesteps = scheduler.timesteps.to(device=device, dtype=torch.float32)
    deltas = (scheduler.sigmas[1:] - scheduler.sigmas[:-1]).to(device=device, dtype=torch.float32)
    helpers = source.block_reference._install_upstream_modules(upstream_path)
    attention_state = {"mode": "bf16", "compare": source.single._NoopCompare(),
                       "receipt": {"attention_calls": 0, "attention_query_chunks": 0}}
    regular_attention = source.attention_reference._build_attention_adapter(torch, attention_state)
    block_type = source.block_reference._extract_pinned_block(
        upstream_path, torch, regular_attention, helpers[-1].apply_rotary_emb, helpers,
    )
    block = block_type(
        hidden_size=direct.HIDDEN, heads_num=direct.HEADS, mlp_width_ratio=4.0,
        mlp_act_type="gelu_tanh", attn_mode="magi_mp_causal", qk_norm=True,
        qk_norm_type="rms", qkv_bias=True, dtype=dtype, device=torch.device("meta"),
    )
    block.to_empty(device=device).eval()
    expected_block = source.cache_reference._BLOCK_RELATIVE_SHAPES
    if set(block.state_dict()) != set(expected_block):
        raise RenderReferenceError("reusable BF16 source block tensor inventory mismatch")

    refiner_state = {"mode": "bf16", "valid_text_tokens": direct.T1_VALID_TEXT_ROWS,
                     "attention_block": 0, "attention_query_chunks": 0,
                     "compare": source.single._NoopCompare()}
    _, refiner_attention = source.attention_reference._install_refiner_attention(torch, refiner_state)
    source.single._bind_refiner_attention_adapter(upstream_path, refiner_attention, refiner_state)
    globals_ = source.attention_reference._load_prelude_weights(
        torch, Path(model["global_shard"]["path"]), upstream_path, helpers, device, dtype,
    )
    source.single._bind_refiner_attention_adapter(
        upstream_path, refiner_attention, refiner_state, globals_["txt_in"],
    )
    final_type = sys.modules["hymm.models.dit.modules.mlp_layers"].FinalLayerPerBlock
    final = final_type(direct.HIDDEN, 1, 32, helpers[0]("silu"),
                       device=torch.device("meta"), dtype=dtype)
    from safetensors import safe_open
    global_path = Path(model["global_shard"]["path"])
    with safe_open(str(global_path), framework="pt", device="cpu") as weight_file:
        final_state = {name: weight_file.get_tensor(f"final_layer.{name}") for name in final.state_dict()}
        if any(value.dtype != dtype or tuple(value.shape) != tuple(final.state_dict()[name].shape)
               for name, value in final_state.items()):
            raise RenderReferenceError("source final-layer weights have wrong BF16 geometry")
        final.load_state_dict(final_state, strict=True, assign=True)
        final.to(device=device, dtype=dtype).eval()
    del final_state

    source_masks, mask_receipt = direct.build_pinned_source_masks(torch, upstream_path)
    soft.validate_bias_support(checked["bias"]["vectors"], source_masks["M11"])
    masks = _device_masks(torch, source_masks, checked["bias"]["vectors"], device)
    del source_masks

    state = {"dtype": dtype, "device_type": "mps", "call": None, "case": None,
             "attention_layout": "compact", "t0_valid_tokens": direct.T0_VALID_TEXT_ROWS,
             "receipt": {"attention_calls": 0, "attention_query_chunks": 0,
                         "t0_attention_calls": 0, "t1_attention_calls": 0}}
    _require_legacy_cache_geometry(source)
    original_factory = source._cache_attention_dispatch_factory
    try:
        source._cache_attention_dispatch_factory = lambda torch_, state_: _dispatch_factory(
            torch_, state_, masks, original_factory,
        )
        cache_function = source._build_pinned_cache_function(torch, upstream_path, state)
    finally:
        source._cache_attention_dispatch_factory = original_factory
    source.two_segment._bind_runtime_cache_functions([block], None, cache_function)
    if observer is not None:
        observer.install(torch, block, globals_, state)

    cache = {key: torch.zeros((1, direct.PHYSICAL_CACHE_ROWS, direct.HEADS, direct.HEAD_DIM),
                              device=device, dtype=dtype) for key in ("k", "v")}
    cache_mask = torch.zeros((1, direct.PHYSICAL_CACHE_ROWS), device=device, dtype=torch.bool)
    parent_inputs = checked["admitted"]["inputs"]
    writer_start = [
        _load_f32_input(torch, np, parent_inputs["t0.patch"], device),
        torch.zeros((1, direct.TEXT_STORAGE_ROWS, direct.HIDDEN), device=device, dtype=dtype),
    ]
    writer_start[1][:, :direct.T0_VALID_TEXT_ROWS].copy_(
        _load_f32_input(torch, np, parent_inputs["t0.text_valid"], device)
    )
    t0_vector = _load_f32_input(torch, np, parent_inputs["t0.vector"], device)
    initial = _load_f32_input(torch, np, parent_inputs["t1.initial_latent"], device)
    qwen = _load_anchor_qwen(torch, np, checked, device)
    t0_mask = torch.arange(direct.TEXT_STORAGE_ROWS, device=device)[None, :] < direct.T0_VALID_TEXT_ROWS
    t1_mask = torch.arange(direct.TEXT_STORAGE_ROWS, device=device)[None, :] < direct.T1_VALID_TEXT_ROWS
    qwen_mask_values = checked["anchor_document"]["qwen_source"].get("cropped_attention_mask")
    if (type(qwen_mask_values) is not list or len(qwen_mask_values) != QWEN_SHAPE[1]
            or qwen_mask_values != [1] * direct.T1_VALID_TEXT_ROWS + [0] * (QWEN_SHAPE[1] - direct.T1_VALID_TEXT_ROWS)):
        raise RenderReferenceError("anchor Qwen mask is not the authenticated 224-row prefix")
    qwen_mask = torch.tensor([qwen_mask_values], device=device, dtype=torch.bool)
    ropes = []
    for name in ("input.rope_cos", "input.rope_sin"):
        values = []
        record = checked["admitted"]["continuation_document"]["tensors"][name]
        bank_bytes = direct.VIDEO_ROWS * direct.HEAD_DIM * 4
        with Path(checked["admitted"]["continuation_payload_path"]).open("rb") as stream:
            for bank in (0, 1):
                stream.seek(record["offset"] + bank * bank_bytes)
                raw = stream.read(bank_bytes)
                if len(raw) != bank_bytes:
                    raise RenderReferenceError("selected source RoPE bank is truncated")
                values.append(torch.from_numpy(np.frombuffer(raw, dtype="<f4").copy().reshape(direct.VIDEO_ROWS, direct.HEAD_DIM))
                              .to(device=device, dtype=torch.float32))
        ropes.append(values)
    rope_t0 = (ropes[0][0], ropes[1][0])
    rope_t1 = (ropes[0][1], ropes[1][1])

    latents = {case: initial.clone() for case in cases}
    captured: list[tuple[str, Any]] = []
    def capture(name: str, tensor: Any) -> None:
        if tensor.device.type != "mps" or tensor.dtype is not dtype:
            raise RenderReferenceError(f"source render capture escaped BF16 MPS: {name}")
        captured.append((name, tensor.detach().contiguous().clone()))

    receipt = {
        "case_order": list(cases), "schedule": ("four outer Euler steps, each with layer-outer T0 then zero/A1_high/A2_high T1 arms"
            if cases == CASE_ORDER else "four outer Euler steps, each with layer-outer T0 then selected T1 arms"),
        "full_blocks_per_arm": 54, "steps": STEPS, "block_calls": 216 * (1 + len(cases)),
        "t0_writer_block_calls": 216, "t1_arm_block_calls": 216 * len(cases), "weights_staged": 216,
        "refiner_calls": STEPS, "final_head_calls": STEPS * len(cases),
        "euler_updates": STEPS * len(cases), "one_resident_block": True,
        "one_layer_kv_cache": True, "source_range_builder_invocations": 1,
        "source_mask_cases": 5, "explicit_terminal_mps_synchronize_count": 0,
        "terminal_export_count": 0, "intermediate_readbacks": 0,
        "attention_calls": 0, "t0_attention_calls": 0, "t1_attention_calls": 0,
        "attention_query_chunks": 0, "device": "mps", "activation_dtype": "BF16",
        "diffusers_version": str(diffusers_version), "legacy_cache_geometry": legacy_geometry,
    }

    cache_dir = Path(model["cache_index_path"]).parent
    with torch.inference_mode(), torch.autocast("mps", dtype=dtype):
        for step in range(STEPS):
            timestep = timesteps[step].reshape(1)
            if observer is not None:
                observer.begin_prelude(step)
            vector = globals_["time_in"](timestep)
            refiner_state["valid_text_tokens"] = direct.T1_VALID_TEXT_ROWS
            refiner_state["attention_block"] = 0
            refiner_state["attention_query_chunks"] = 0
            full_text = globals_["txt_in"](qwen, timestep, qwen_mask)
            if refiner_state["attention_block"] != 2 or tuple(full_text.shape) != (1, 1280, direct.HIDDEN):
                raise RenderReferenceError("pinned action text refiner did not complete both blocks")
            patches = {case: globals_["img_in"](latents[case]) for case in cases}
            if any(tuple(patch.shape) != (1, direct.VIDEO_ROWS, direct.HIDDEN) for patch in patches.values()):
                raise RenderReferenceError("pinned T1 patch embedding geometry mismatch")
            if observer is not None:
                observer.end_prelude(patches)
            if step == 0:
                capture("zero.step1.patch", patches[first_case])
                capture("zero.step1.text_valid", full_text[:, :direct.T1_VALID_TEXT_ROWS])
                capture("zero.step1.vector", vector)
            base_text = full_text[:, :direct.TEXT_STORAGE_ROWS].contiguous()
            arms: dict[str, list[Any]] = {
                case: [patches[case], base_text.clone()] for case in cases
            }
            writer = [writer_start[0].clone(), writer_start[1].clone()]
            for layer, entry in enumerate(model["cache_index"]["blocks"]):
                path = source.single._canonical_file(cache_dir / entry["bf16"], "BF16 block shard")
                if source.single._file_identity(path) != model["cache_identities"][str(path)]:
                    raise RenderReferenceError("BF16 block shard changed after authentication")
                with safe_open(str(path), framework="pt", device="cpu") as weights:
                    if set(weights.keys()) != {f"double_blocks.{layer}.{name}" for name in expected_block}:
                        raise RenderReferenceError("source BF16 block tensor inventory changed")
                    for name, target in block.state_dict().items():
                        value = weights.get_tensor(f"double_blocks.{layer}.{name}")
                        if value.dtype is not dtype or tuple(value.shape) != tuple(expected_block[name]):
                            raise RenderReferenceError("source BF16 block tensor shape/dtype changed")
                        target.copy_(value, non_blocking=True)
                if source.single._file_identity(path) != model["cache_identities"][str(path)]:
                    raise RenderReferenceError("BF16 block shard changed while staging")
                cache_mask.zero_()
                state["call"] = {"condition": "t0_caption", "valid_text_tokens": direct.T0_VALID_TEXT_ROWS,
                                 "cache_valid_len": 0, "update_cache": True, "cache_txt": True}
                if observer is not None:
                    observer.begin_block(step, layer, "T0", writer[0], writer[1], t0_vector, rope_t0)
                writer = block(
                    img=writer[0], txt=writer[1], vec=t0_vector, vec_txt=t0_vector,
                    freqs_cis=rope_t0, text_mask=t0_mask, attn_param=None, is_flash=False,
                    txt_tokens_per_block=direct.TEXT_STORAGE_ROWS,
                    prompt_video_boundaries=[0, direct.VIDEO_ROWS], kv_cache=cache,
                    cache_valid_len=0, update_cache=True, cache_txt=True,
                    cache_attn_mode="torch", cache_mask=cache_mask,
                )
                writer = list(writer)
                if observer is not None:
                    observer.end_block(writer)
                state["call"] = {"condition": "t1_action", "valid_text_tokens": direct.T1_VALID_TEXT_ROWS,
                                 "cache_valid_len": direct.PHYSICAL_CACHE_ROWS,
                                 "update_cache": False, "cache_txt": True}
                for case in cases:
                    state["case"] = case
                    if observer is not None:
                        observer.begin_block(step, layer, case, arms[case][0], arms[case][1], vector, rope_t1)
                    arms[case] = list(block(
                        img=arms[case][0], txt=arms[case][1], vec=vector, vec_txt=vector,
                        freqs_cis=rope_t1, text_mask=t1_mask, attn_param=None, is_flash=False,
                        txt_tokens_per_block=direct.TEXT_STORAGE_ROWS,
                        prompt_video_boundaries=[0, direct.VIDEO_ROWS], kv_cache=cache,
                        cache_valid_len=direct.PHYSICAL_CACHE_ROWS, update_cache=False,
                        cache_txt=True, cache_attn_mode="torch", cache_mask=cache_mask,
                    ))
                    if observer is not None:
                        observer.end_block(arms[case])
            for case in cases:
                if observer is not None:
                    observer.begin_head(step, case)
                prediction_tokens = globals_["final_layer"](arms[case][0], vector)
                if tuple(prediction_tokens.shape) != (1, direct.VIDEO_ROWS, 32):
                    raise RenderReferenceError("pinned final head output shape mismatch")
                prediction = prediction_tokens.reshape(1, 30, 9, 16, 32).permute(0, 4, 1, 2, 3).contiguous()
                if step == 0:
                    capture(f"{case}.step1.velocity", prediction)
                latents[case] = (latents[case].float() + prediction.float() *
                                 deltas[step]).to(dtype)
                capture(f"{case}.step{step + 1}.latent", latents[case])
                if observer is not None:
                    observer.end_head(step, case, prediction, latents[case])
                del prediction_tokens, prediction
            del arms, patches, base_text, full_text, vector, writer

    receipt["attention_calls"] = state["receipt"]["attention_calls"]
    receipt["t0_attention_calls"] = state["receipt"]["t0_attention_calls"]
    receipt["t1_attention_calls"] = state["receipt"]["t1_attention_calls"]
    receipt["attention_query_chunks"] = state["receipt"]["attention_query_chunks"]
    expected_counts = {"attention_calls": 216 * (1 + len(cases)), "t0_attention_calls": 216,
                       "t1_attention_calls": 216 * len(cases),
                       "attention_query_chunks": 216 * (1 + len(cases)) * 18}
    if any(receipt[key] != value for key, value in expected_counts.items()):
        raise RenderReferenceError("source render call-count receipt mismatch")
    if len(captured) != 5 * len(cases) + 3:
        raise RenderReferenceError("source render capture count mismatch")
    layout = [(name, tuple(tensor.shape), int(tensor.numel())) for name, tensor in captured]
    packed = torch.cat([tensor.float().reshape(-1) for _, tensor in captured], dim=0).contiguous()
    torch.mps.synchronize()
    receipt["explicit_terminal_mps_synchronize_count"] = 1
    receipt["terminal_export_count"] = 1
    host = packed.cpu().numpy()
    outputs: dict[str, Any] = {}
    cursor = 0
    for name, shape, count in layout:
        outputs[name] = host[cursor:cursor + count].reshape(shape).copy()
        cursor += count
    if cursor != host.size:
        raise RenderReferenceError("terminal source export inventory mismatch")
    receipt["captures"] = len(outputs)
    receipt["total_block_invocations"] = expected_counts["attention_calls"]
    receipt["weights_staged"] = 216
    receipt["mask_builder"] = mask_receipt
    receipt["scheduler"] = source.single._schedule_record(scheduler)
    receipt["autocast"] = "pinned source BF16 MPS"
    if observer is not None:
        receipt["diagnostic_activation_exports"] = observer.receipt()
        receipt["intermediate_readbacks"] = len(observer.expected)
        observer.uninstall()
    return validate_captures(outputs, cases=cases), receipt


def _publish(output_manifest: Path, checked: Mapping[str, Any], model: Mapping[str, Any],
             captures: Mapping[str, Any], receipt: Mapping[str, Any]) -> tuple[Path, Path, dict[str, Any]]:
    import numpy as np

    manifest, payload = _fresh_render_paths(output_manifest)
    captures = validate_captures(captures)
    gates = _check_source_replay(checked, captures)
    parent_dir = manifest.parent
    payload_temp: Path | None = None
    manifest_temp: Path | None = None
    payload_published = False
    manifest_published = False
    try:
        hasher = hashlib.sha256()
        records: dict[str, Any] = {}
        cursor = 0
        fd, name = tempfile.mkstemp(prefix=f".{payload.name}.", suffix=".tmp", dir=parent_dir)
        payload_temp = Path(name)
        with os.fdopen(fd, "wb") as stream:
            for tensor_name in TENSOR_ORDER:
                array = np.asarray(captures[tensor_name], dtype="<f4")
                raw = array.tobytes(order="C")
                expected = math.prod(TENSOR_SHAPES[tensor_name]) * 4
                if len(raw) != expected:
                    raise RenderReferenceError(f"render tensor byte count mismatch: {tensor_name}")
                records[tensor_name] = {
                    "dtype": "F32", "source_dtype": "BF16", "shape": list(TENSOR_SHAPES[tensor_name]),
                    "offset_bytes": cursor, "nbytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest(),
                    "checkpoint_boundary": tensor_name.split(".", 1)[1],
                }
                stream.write(raw)
                hasher.update(raw)
                cursor += len(raw)
            stream.flush()
            os.fsync(stream.fileno())
        source_document = direct.source_document(model)
        document = {
            "schema": SCHEMA, "schema_version": SCHEMA_VERSION,
            "upstream_revision": direct.PINNED_REVISION,
            "case_order": list(CASE_ORDER),
            "bindings": {
                "parent_sha256": soft.PARENT_SHA256,
                "bias_sha256": soft.BIAS_SHA256,
                "roi_source_sha256": direct.ROI_MANIFEST_SHA256,
                "noise_selection_sha256": NOISE_SELECTION_SHA256,
                "initial_latent_sha256": PARENT_INITIAL_SHA256,
                "t1_conditioning_sha256": ANCHOR_SHA256,
                "t1_qwen_sha256": ANCHOR_QWEN_SHA256,
            },
            "inputs": _input_document(checked),
            "source": source_document,
            "geometry": {"latent_shape": list(LATENT_SHAPE), "video_rows": direct.VIDEO_ROWS,
                         "text_storage_rows": direct.TEXT_STORAGE_ROWS,
                         "valid_text_rows": direct.T1_VALID_TEXT_ROWS,
                         "case_order": list(CASE_ORDER), "steps": STEPS},
            "precision": {"resident_weights": "BF16", "activation_dtype": "BF16",
                          "input_storage": "F32 lossless promotion of BF16 saved states",
                          "attention": "pinned source cache body with compact MPS SDPA chunks of 256; BF16 Q/K/V",
                          "mps_fallback": False, "cuda_parity_claimed": False,
                          "bias_storage": "authored BF16-expanded F32 promoted to BF16 before SDPA"},
            "execution": {"case_order": list(CASE_ORDER), "scope": "four-step saved-state direct DiT render; no DCAE",
                          "exact_source_replay": gates, "runtime_receipt": dict(receipt),
                          "zero_step4_bitwise_exact": gates["zero.step4.latent"]["bitwise_equal"]},
            "implementation": {"path": str(Path(__file__).resolve()), "sha256": _sha256_file(Path(__file__).resolve()),
                               "direct_runtime_path": str(SUPPORT / "hy_nomas_dit_route_factorial_runtime.py"),
                               "direct_runtime_sha256": _sha256_file(SUPPORT / "hy_nomas_dit_route_factorial_runtime.py"),
                               "soft_route_path": str(SUPPORT / "hy_nomas_soft_route_reference.py"),
                               "soft_route_sha256": _sha256_file(SUPPORT / "hy_nomas_soft_route_reference.py")},
            "payload": {"basename": payload.name, "bytes": cursor, "sha256": hasher.hexdigest(),
                        "encoding": "f32le_bf16_expansion"},
            "tensors": records,
        }
        payload_bytes = cursor
        if set(records) != set(TENSOR_ORDER) or payload_bytes != sum(math.prod(TENSOR_SHAPES[name]) * 4 for name in TENSOR_ORDER):
            raise RenderReferenceError("render manifest tensor inventory/extent mismatch")
        fd, name = tempfile.mkstemp(prefix=f".{manifest.name}.", suffix=".tmp", dir=parent_dir)
        manifest_temp = Path(name)
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as stream:
            json.dump(document, stream, sort_keys=True, separators=(",", ":"), allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.link(payload_temp, payload)
        payload_published = True
        payload_temp.unlink()
        payload_temp = None
        os.link(manifest_temp, manifest)
        manifest_published = True
        manifest_temp.unlink()
        manifest_temp = None
        return manifest, payload, document
    except Exception:
        if manifest_published:
            manifest.unlink(missing_ok=True)
        if payload_published:
            payload.unlink(missing_ok=True)
        raise
    finally:
        for temporary in (payload_temp, manifest_temp):
            if temporary is not None:
                temporary.unlink(missing_ok=True)


def _parser(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument("--plan-only", action="store_true")
    modes.add_argument("--admit-only", action="store_true")
    modes.add_argument("--execute", action="store_true")
    parser.add_argument("--anchor", type=Path, default=DEFAULT_ANCHOR)
    parser.add_argument("--anchor-sha256", default=ANCHOR_SHA256)
    parser.add_argument("--parent", type=Path, default=DEFAULT_PARENT)
    parser.add_argument("--parent-sha256", default=soft.PARENT_SHA256)
    parser.add_argument("--bias-manifest", type=Path, default=DEFAULT_BIAS)
    parser.add_argument("--bias-sha256", default=soft.BIAS_SHA256)
    parser.add_argument("--noise-selection", type=Path, default=DEFAULT_NOISE_SELECTION)
    parser.add_argument("--noise-selection-sha256", default=NOISE_SELECTION_SHA256)
    parser.add_argument("--zero-step4", type=Path, default=DEFAULT_ZERO_STEP4)
    parser.add_argument("--upstream", type=Path, default=DEFAULT_UPSTREAM)
    parser.add_argument("--cache-index", type=Path, default=DEFAULT_CACHE_INDEX)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args(argv)
    for value, label in ((args.anchor_sha256, "--anchor-sha256"), (args.parent_sha256, "--parent-sha256"),
                         (args.bias_sha256, "--bias-sha256"),
                         (args.noise_selection_sha256, "--noise-selection-sha256")):
        if type(value) is not str or len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
            parser.error(f"{label} must be a lowercase SHA-256 digest")
    if args.execute and args.report is None:
        parser.error("--report is required with --execute")
    if not args.execute and args.report is not None:
        parser.error("--report is valid only with --execute")
    if args.execute and not args.report.is_absolute():
        parser.error("--report must be an absolute path")
    return args


def main(argv: list[str] | None = None) -> int:
    args = _parser(argv)
    try:
        if args.execute:
            _fresh_render_paths(args.report)
        checked = authenticate_inputs(
            args.anchor, args.anchor_sha256, args.parent, args.parent_sha256,
            args.bias_manifest, args.bias_sha256, args.noise_selection, args.noise_selection_sha256,
            args.zero_step4,
        )
        if args.plan_only:
            print(json.dumps({"schema": SCHEMA, "status": "plan_validated", "model_launched": False,
                              "mps_device_used": False, "cases": list(CASE_ORDER), "steps": STEPS,
                              "named_output_bytes": sum(math.prod(TENSOR_SHAPES[name]) * 4 for name in TENSOR_ORDER),
                              "selected_initial_latent_sha256": PARENT_INITIAL_SHA256,
                              "anchor_latent_used": False}, sort_keys=True))
            return 0
        if args.admit_only:
            _self_test_source_replay_gate(checked)
            print(json.dumps({"schema": SCHEMA, "status": "inputs_admitted_cpu_only", "model_launched": False,
                              "mps_device_used": False, "cases": list(CASE_ORDER), "steps": STEPS,
                              "bias_support": checked["bias_support"],
                              "selected_initial_latent_sha256": PARENT_INITIAL_SHA256,
                              "anchor_latent_used": False,
                              "source_replay_gate_self_test": "accepted path and seeded one-bit step-2 corruption passed"},
                             sort_keys=True))
            return 0
        model = direct.authenticate_model_sources(args.cache_index, args.upstream)
        captures, receipt = _run_source(checked, model, Path(model["upstream_path"]))
        manifest, payload, document = _publish(args.report, checked, model, captures, receipt)
        print(json.dumps({"schema": SCHEMA, "status": "source_render_completed",
                          "model_launched": True, "mps_device_used": True,
                          "manifest": str(manifest), "manifest_sha256": _sha256_file(manifest),
                          "payload": str(payload), "payload_sha256": document["payload"]["sha256"],
                          "tensor_count": len(document["tensors"]), "runtime_receipt": receipt}, sort_keys=True))
        return 0
    except (RenderReferenceError, direct.ReferenceError, soft.SoftRouteError,
            generation_inputs.InputsError) as error:
        print(json.dumps({"schema": SCHEMA, "status": "rejected", "error": str(error)}, sort_keys=True),
              file=sys.stderr)
        return 2
    except Exception as error:
        print(json.dumps({"schema": SCHEMA, "status": "rejected",
                          "error": f"{type(error).__name__}: {error}"}, sort_keys=True), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
