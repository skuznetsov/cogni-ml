#!/usr/bin/env python3
"""Run the pinned MLX top-k/top-p path on fixed logits and common noise.

This diagnostic mirrors pinned ``sample_top_k_top_p`` through the candidate
mask, then applies shared-uniform inverse-CDF and shared-Gumbel shadows. It
does not call MLX's seeded random stream and does not claim seeded-sampler
parity. GPU arrays remain live until one terminal ``mx.eval``; JSON conversion
happens only after that evaluation.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import sys
from pathlib import Path
from typing import Any

try:
    import compare_sampler_common as common
except ModuleNotFoundError:  # Support importing this file from the repository root.
    from scripts.drama3 import compare_sampler_common as common


PINNED_MLX_VERSION = "0.31.1"


def _read_once(path: Path) -> tuple[bytes, Any]:
    raw = path.read_bytes()
    try:
        parsed = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise common.InvalidBundle(f"{path}: invalid UTF-8 JSON") from error
    return raw, parsed


def _validate_inputs(fixture: Any, native: Any, fixture_sha256: str) -> list[dict[str, Any]]:
    common._require(isinstance(fixture, dict), "fixture: expected JSON object")
    common._require(fixture.get("format") == common.FIXTURE_FORMAT, "fixture format mismatch")
    common._require(common._sha(fixture_sha256, "fixture SHA-256") == fixture_sha256,
                    "fixture SHA-256 malformed")
    common._require(isinstance(native, dict), "native output: expected JSON object")
    common._require(native.get("format") == common.PROBE_FORMAT, "native output format mismatch")
    common._require(native.get("fixture_sha256") == fixture_sha256,
                    "native output is bound to different fixture bytes")
    provenance = fixture.get("provenance")
    common._require(isinstance(provenance, dict), "fixture.provenance: expected object")
    common._require(provenance.get("upstream") == common.PINNED_UPSTREAM,
                    "fixture upstream pin mismatch")
    sampling = fixture.get("sampling_contract")
    common._require(isinstance(sampling, dict), "fixture.sampling_contract: expected object")
    common._require(sampling.get("atol") == common.ATOL and sampling.get("rtol") == common.RTOL,
                    "fixture tolerance differs from the fixed sampler contract")
    common._require(provenance.get("seeded_rng_parity_claimed") is False,
                    "fixture must not claim seeded RNG parity")

    raw_cases = fixture.get("cases")
    common._require(isinstance(raw_cases, list) and bool(raw_cases),
                    "fixture.cases: expected nonempty array")
    cases: list[dict[str, Any]] = []
    names: set[str] = set()
    for index, raw in enumerate(raw_cases):
        case = common._validate_case_input(raw, f"fixture.cases[{index}]")
        common._require(case["name"] not in names, f"fixture.cases[{index}].name: duplicate")
        names.add(case["name"])
        cases.append(case)

    native_cases = native.get("cases")
    common._require(isinstance(native_cases, list), "native output.cases: expected array")
    native_by_name: dict[str, dict[str, Any]] = {}
    case_by_name = {case["name"]: case for case in cases}
    for index, row in enumerate(native_cases):
        where = f"native output.cases[{index}]"
        common._require(isinstance(row, dict), f"{where}: expected object")
        name = row.get("name")
        common._require(isinstance(name, str) and name in names, f"{where}.name: unknown case")
        common._require(name not in native_by_name, f"{where}.name: duplicate case")
        case = case_by_name[name]
        for key in ("role", "source_arm", "step", "case_kind"):
            common._require(row.get(key) == case[key], f"{where}.{key}: metadata mismatch")
        common._require(common._f32(row.get("temperature"), f"{where}.temperature") == case["temperature"],
                        f"{where}.temperature: mismatch")
        common._require(common._strict_int(row.get("top_k"), f"{where}.top_k", minimum=1) == case["top_k"],
                        f"{where}.top_k: mismatch")
        common._require(common._f32(row.get("top_p"), f"{where}.top_p") == case["top_p"],
                        f"{where}.top_p: mismatch")
        common._require(common._number_list(row.get("draws"), f"{where}.draws", len(case["draws"])) == case["draws"],
                        f"{where}.draws: mismatch")
        common._validate_algorithm_record(row, case, "native", where)
        native_by_name[name] = row
    common._require(set(native_by_name) == names, "native output cases do not exactly match fixture cases")
    common._validate_probe_lineage(native, fixture)
    common._validate_queue_ledger(native, cases)
    return cases


def _terminal_row_list(array: Any) -> list[Any]:
    """Extract the sole row from an evaluated [1, width] MLX diagnostic."""
    value = array.tolist()
    if isinstance(value, list) and len(value) == 1 and isinstance(value[0], list):
        return value[0]
    raise common.InvalidBundle("unexpected MLX output shape after terminal evaluation")


def _make_mlx_outputs(cases: list[dict[str, Any]], mx: Any) -> tuple[list[dict[str, Any]], int]:
    """Construct every diagnostic lazily and evaluate all arrays once."""
    staged: list[tuple[dict[str, Any], dict[str, Any]]] = []
    eval_refs: list[Any] = []

    for case in cases:
        scaled = mx.array(case["logits"], dtype=mx.float32).reshape((1, -1))
        scaled = scaled.astype(mx.float32) / max(float(case["temperature"]), 1e-5)
        k = min(int(case["top_k"]), int(scaled.shape[-1]))

        # Keep operator order aligned with the pinned tower.py implementation.
        top_ids = mx.argpartition(-scaled, kth=k - 1, axis=-1)[..., :k]
        top_values = mx.take_along_axis(scaled, top_ids, axis=-1)
        order = mx.argsort(-top_values, axis=-1)
        top_ids = mx.take_along_axis(top_ids, order, axis=-1)
        top_values = mx.take_along_axis(top_values, order, axis=-1)

        log_norm = mx.logsumexp(scaled, axis=-1, keepdims=True)
        full_probabilities = mx.exp(top_values - log_norm)
        full_cdf = mx.cumsum(full_probabilities, axis=-1)
        dropped = full_cdf > float(case["top_p"])
        dropped[..., 0] = False
        masked = mx.where(dropped, -mx.inf, top_values)

        # The pinned categorical consumes these masked logits. The two outputs
        # below replace its random source only with declared common variates.
        conditional_probabilities = mx.softmax(masked, axis=-1)
        conditional_cdf_all = mx.cumsum(conditional_probabilities, axis=-1)
        conditional_cdf = mx.where(dropped, 0.0, conditional_cdf_all)
        kept_count = mx.sum(~dropped, axis=-1)
        normalization = mx.exp(log_norm - mx.max(scaled, axis=-1, keepdims=True))

        draws = mx.array(case["draws"], dtype=mx.float32).reshape((-1, 1))
        cdf_choice = mx.minimum(
            mx.sum(conditional_cdf_all <= draws, axis=-1), kept_count - 1
        ).astype(mx.int32)
        ids_for_draws = mx.broadcast_to(top_ids, (len(case["draws"]), k))
        sample_ids = mx.take_along_axis(ids_for_draws, cdf_choice[..., None], axis=-1).squeeze(-1)

        noise = mx.array(case["noise_uniforms"], dtype=mx.float32)
        gumbel = -mx.log(-mx.log(noise))
        gumbel_scores = mx.where(dropped, -mx.inf, top_values) + gumbel
        gumbel_rank = mx.argmax(gumbel_scores, axis=-1).astype(mx.int32)
        gumbel_ids = mx.take_along_axis(ids_for_draws, gumbel_rank[..., None], axis=-1).squeeze(-1)

        arrays = {
            "topk_ids": top_ids,
            "topk_scaled_logits": top_values,
            "kept_mask": ~dropped,
            "kept_count": kept_count,
            "full_vocab_normalization": normalization,
            "full_probabilities": full_probabilities,
            "full_cdf": full_cdf,
            "retained_conditional_probabilities": conditional_probabilities,
            "retained_conditional_cdf": conditional_cdf,
            "sample_ids": sample_ids,
            "shared_gumbel_ids": gumbel_ids,
        }
        staged.append((case, arrays))
        eval_refs.extend(arrays.values())

    if not eval_refs:
        raise common.InvalidBundle("no MLX arrays were constructed")
    mx.eval(*eval_refs)

    outputs: list[dict[str, Any]] = []
    for case, arrays in staged:
        top_ids = [int(value) for value in _terminal_row_list(arrays["topk_ids"])]
        sample_ids = [int(value) for value in arrays["sample_ids"].tolist()]
        gumbel_ids = [int(value) for value in arrays["shared_gumbel_ids"].tolist()]
        count = int(arrays["kept_count"].tolist()[0])
        values = {
            "mlx_topk_ids": top_ids,
            "mlx_topk_scaled_logits": [float(value) for value in _terminal_row_list(arrays["topk_scaled_logits"])],
            "mlx_kept_mask": [bool(value) for value in _terminal_row_list(arrays["kept_mask"])],
            "mlx_kept_count": count,
            "mlx_full_vocab_normalization": float(_terminal_row_list(arrays["full_vocab_normalization"])[0]),
            "mlx_full_probabilities": [float(value) for value in _terminal_row_list(arrays["full_probabilities"])],
            "mlx_full_cdf": [float(value) for value in _terminal_row_list(arrays["full_cdf"])],
            "mlx_retained_conditional_probabilities": [
                float(value) for value in _terminal_row_list(arrays["retained_conditional_probabilities"])
            ],
            "mlx_retained_conditional_cdf": [
                float(value) for value in _terminal_row_list(arrays["retained_conditional_cdf"])
            ],
            "mlx_sample_ids": sample_ids,
            "mlx_shared_gumbel_ids": gumbel_ids,
        }
        outputs.append(values)
    return outputs, len(eval_refs)


def _write_new(path: Path, value: dict[str, Any]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() or path.is_symlink():
        raise FileExistsError(f"refusing to overwrite existing output: {path}")
    payload = (json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode()
    with path.open("xb") as stream:
        stream.write(payload)
    return hashlib.sha256(payload).hexdigest()


def run(fixture_path: Path, native_path: Path, output_path: Path, mlx_root: Path,
        *, validate_only: bool = False) -> dict[str, Any]:
    fixture_raw, fixture = _read_once(fixture_path)
    native_raw, native = _read_once(native_path)
    fixture_sha = hashlib.sha256(fixture_raw).hexdigest()
    native_sha = hashlib.sha256(native_raw).hexdigest()
    cases = _validate_inputs(fixture, native, fixture_sha)
    sampler_source = common._pinned_sampler_metadata(mlx_root)
    if validate_only:
        return {
            "status": "valid",
            "fixture_sha256": fixture_sha,
            "native_output_sha256": native_sha,
            "case_count": len(cases),
            "pinned_sampler_source": sampler_source,
        }

    if output_path.resolve() in (fixture_path.resolve(), native_path.resolve()):
        raise common.InvalidBundle("output path must differ from fixture and native output")

    try:
        runtime_version = importlib.metadata.version("mlx")
    except importlib.metadata.PackageNotFoundError as error:
        raise common.InvalidBundle("MLX Python package is not installed in this interpreter") from error
    common._require(runtime_version == PINNED_MLX_VERSION,
                    f"expected MLX {PINNED_MLX_VERSION}, found {runtime_version}")

    import mlx.core as mx

    mx.set_default_device(mx.gpu)
    common._require(mx.default_device() == mx.gpu,
                    f"failed to select MLX GPU: {mx.default_device()}")
    mlx_fields, evaluated_array_count = _make_mlx_outputs(cases, mx)

    native_cases = {row["name"]: row for row in native["cases"]}
    merged_cases: list[dict[str, Any]] = []
    for case, mlx_record in zip(cases, mlx_fields, strict=True):
        merged = dict(native_cases[case["name"]])
        merged.update(mlx_record)
        merged_cases.append(merged)

    merged_output = dict(native)
    merged_output["cases"] = merged_cases
    merged_output["native_output_sha256"] = native_sha
    merged_output["lineage"] = {
        "fixture_sha256": fixture_sha,
        "native_output_sha256": native_sha,
        "mlx_reference": {
            "upstream": common.PINNED_UPSTREAM,
            "sampler_source": sampler_source,
            "mlx_version": runtime_version,
            "device": str(mx.default_device()),
            "operator_path": "pinned sample_top_k_top_p through masked logits; shared-uniform inverse-CDF and shared-Gumbel shadows",
            "randomness": "declared common variates; no seeded MLX RNG replay claim",
            "terminal_evaluation_calls": 1,
            "evaluated_array_count": evaluated_array_count,
            "host_conversion_after_terminal_evaluation": True,
        },
    }
    merged_output["mlx_reference_source_sha256"] = sampler_source["source_sha256"]
    output_sha = _write_new(output_path, merged_output)
    return {
        "status": "complete",
        "output": str(output_path.resolve()),
        "output_sha256": output_sha,
        "fixture_sha256": fixture_sha,
        "native_output_sha256": native_sha,
        "case_count": len(cases),
        "evaluated_array_count": evaluated_array_count,
        "device": str(mx.default_device()),
        "mlx_version": runtime_version,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixture", type=Path, required=True)
    parser.add_argument("--native-output", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--mlx-root", type=Path, default=common.PINNED_MLX_ROOT)
    parser.add_argument("--validate-only", action="store_true",
                        help="validate fixture/native lineage without importing MLX or running GPU work")
    args = parser.parse_args(argv)
    try:
        result = run(args.fixture, args.native_output, args.output, args.mlx_root,
                     validate_only=args.validate_only)
        print(json.dumps(result, sort_keys=True))
        return 0
    except (common.InvalidBundle, FileExistsError, OSError, ValueError, RuntimeError) as error:
        print(json.dumps({"status": "invalid", "error": str(error)}, sort_keys=True), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
