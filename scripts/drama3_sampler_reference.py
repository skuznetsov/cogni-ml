#!/usr/bin/env python3
"""Build deterministic DRAMA-3 sampler fixtures from pinned CPU-side evidence.

This is deliberately not a model runner. It reads retained tower logits, mirrors
the scalar FP32 Metal sampler for expected IDs, and executes only the pinned
Fish-Speech sampler function on CPU to audit support and tie behavior.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import math
import struct
import subprocess
from pathlib import Path
from typing import Any


REFERENCE_ROOT = Path("/private/tmp/cogni-drama3-reference")
UPSTREAM_ROOT = REFERENCE_ROOT / "fish-speech"
UPSTREAM_REVISION = "541566932abbaebcb0a40e562a1f846738383f38"
UPSTREAM_SOURCE = Path(
    "fish_speech/models/text2semantic/fish_omni/tower.py"
)
UPSTREAM_SOURCE_SHA256 = "71f47e85bcc51c1b644fd29d863598bf87208dbd86d479791d699629b7b92d8c"
NATIVE_TRACES = REFERENCE_ROOT / "native_actual_tower_traces.json"
NATIVE_TRACE_SHA256 = "310d002c0aa6c9dc8748fcd25e7b6b7ea1fd82926c0268e8663fdb3029da0db8"
TORCH_TRACES = REFERENCE_ROOT / "torch_actual_tower_traces.json"
TORCH_TRACE_SHA256 = "784057f0d440401183b8db66c1ab07d34e262525a358c7944207558b5e2debaa"
TORCH_MANIFEST = REFERENCE_ROOT / "torch_actual_tower_traces.json.manifest.json"
DEFAULT_REFERENCE_OUTPUT = REFERENCE_ROOT / "sampler-cost-20261014-r2"
DEFAULT_RUN_OUTPUT = Path(
    "/Users/sergey/.cache/cogni-ml/drama3-runs/sampler-cost-20261014-r2"
)

DRAWS = [0.0, 0.01, 0.1, 0.25, 0.5, 0.75, 0.9, 0.99, 1.0]
TEMPERATURE = 0.8
TOP_K = 30
TOP_P = 0.9
RETRY_TEMPERATURE = 1.0
RETRY_TOP_P = 0.95


def f32(value: float) -> float:
    return struct.unpack("<f", struct.pack("<f", float(value)))[0]


def add32(left: float, right: float) -> float:
    return f32(f32(left) + f32(right))


def sub32(left: float, right: float) -> float:
    return f32(f32(left) - f32(right))


def mul32(left: float, right: float) -> float:
    return f32(f32(left) * f32(right))


def div32(left: float, right: float) -> float:
    return f32(f32(left) / f32(right))


def exp32(value: float) -> float:
    # Metal exp() is implementation-specific; this models its FP32 result by
    # rounding the host double exp to FP32 and records that limitation below.
    return f32(math.exp(f32(value)))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def pinned_source() -> tuple[str, str, str, Any]:
    revision = subprocess.check_output(
        ["git", "-C", str(UPSTREAM_ROOT), "rev-parse", "HEAD"], text=True
    ).strip()
    if revision != UPSTREAM_REVISION:
        raise RuntimeError(f"upstream revision mismatch: {revision}")
    source_path = UPSTREAM_ROOT / UPSTREAM_SOURCE
    source = source_path.read_text(encoding="utf-8")
    source_hash = sha256(source_path)
    if source_hash != UPSTREAM_SOURCE_SHA256:
        raise RuntimeError(f"pinned tower.py hash mismatch: {source_hash}")
    tree = ast.parse(source, filename=str(source_path))
    function = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "sample_top_k_top_p"
    )
    function_source = ast.get_source_segment(source, function)
    if function_source is None:
        raise RuntimeError("could not extract pinned sampler source")
    function_hash = hashlib.sha256(function_source.encode("utf-8")).hexdigest()
    return revision, source_hash, function_hash, function


def load_tower_rows() -> tuple[dict[str, list[float]], dict[str, Any]]:
    if sha256(NATIVE_TRACES) != NATIVE_TRACE_SHA256:
        raise RuntimeError("native tower trace hash mismatch")
    if sha256(TORCH_TRACES) != TORCH_TRACE_SHA256:
        raise RuntimeError("upstream torch tower trace hash mismatch")
    manifest = json.loads(TORCH_MANIFEST.read_text(encoding="utf-8"))
    if manifest["trace_sha256"] != TORCH_TRACE_SHA256:
        raise RuntimeError("upstream trace manifest hash mismatch")
    if manifest["upstream"]["fish-speech"]["revision"] != UPSTREAM_REVISION:
        raise RuntimeError("trace manifest upstream revision mismatch")

    native = json.loads(NATIVE_TRACES.read_text(encoding="utf-8"))
    upstream = json.loads(TORCH_TRACES.read_text(encoding="utf-8"))
    if native.get("format") != "drama3-reference-traces-v1":
        raise RuntimeError("native trace format mismatch")
    if upstream.get("format") != "drama3-reference-traces-v1":
        raise RuntimeError("upstream trace format mismatch")

    def collect(bundle: dict[str, Any]) -> dict[str, list[float]]:
        result: dict[str, list[float]] = {}
        for trace in bundle["traces"]:
            name = trace["name"]
            if ".head_logits." not in name:
                continue
            if trace.get("dtype") != "float32" or trace.get("shape") != [1, 4096]:
                raise RuntimeError(f"unexpected head-logit trace shape: {name}")
            values = [f32(v) for v in trace["values"]]
            if len(values) != 4096 or not all(math.isfinite(v) for v in values):
                raise RuntimeError(f"invalid head-logit values: {name}")
            result[name] = values
        return result

    native_rows = collect(native)
    upstream_rows = collect(upstream)
    if len(native_rows) != 18 or set(native_rows) != set(upstream_rows):
        raise RuntimeError("expected 18 matching native/upstream tower head rows")
    return native_rows, upstream_rows


def native_scalar(
    logits: list[float], temperature: float, top_k: int, top_p: float, draws: list[float]
) -> tuple[list[int], dict[str, Any]]:
    temp = max(f32(temperature), f32(1e-5))
    p_limit = f32(top_p)
    scaled = [div32(value, temp) for value in logits]
    maximum = max(scaled)
    values = [0.0] * top_k
    ids = [0] * top_k
    comparison_count = 0
    for token_id, value in enumerate(scaled):
        pos = min(token_id, top_k)
        while pos > 0:
            comparison_count += 1
            previous_value = values[pos - 1]
            previous_id = ids[pos - 1]
            if value > previous_value or (value == previous_value and token_id < previous_id):
                if pos < top_k:
                    values[pos] = values[pos - 1]
                    ids[pos] = ids[pos - 1]
                pos -= 1
            else:
                break
        if pos < top_k:
            values[pos] = value
            ids[pos] = token_id

    denominator = 0.0
    for value in logits:
        scaled_value = div32(value, temp)
        denominator = add32(denominator, exp32(sub32(scaled_value, maximum)))

    kept_values: list[float] = []
    kept_ids: list[int] = []
    cumulative = 0.0
    cumulative_values: list[float] = []
    for rank in range(top_k):
        weight = exp32(sub32(values[rank], maximum))
        cumulative = add32(cumulative, div32(weight, denominator))
        cumulative_values.append(cumulative)
        if rank == 0 or cumulative <= p_limit:
            kept_values.append(weight)
            kept_ids.append(ids[rank])

    total = 0.0
    for weight in kept_values:
        total = add32(total, weight)
    result: list[int] = []
    draw_margins: list[float] = []
    normalized_cdf: list[float] = []
    for raw_draw in draws:
        draw = f32(raw_draw)
        threshold = mul32(draw, total)
        chosen = kept_ids[-1]
        nearest_transition = math.inf
        running = 0.0
        for token_id, weight in zip(kept_ids, kept_values):
            threshold = sub32(threshold, weight)
            nearest_transition = min(nearest_transition, abs(threshold))
            running = add32(running, weight)
            if threshold < 0.0:
                chosen = token_id
                break
        result.append(chosen)
        draw_margins.append(nearest_transition)
        normalized_cdf.append(div32(running, total))

    sorted_scaled = sorted(scaled, reverse=True)
    top_k_gap = sub32(sorted_scaled[top_k - 1], sorted_scaled[top_k])
    p_cutoff = next((c - p_limit for c in cumulative_values if c > p_limit), None)
    if p_cutoff is None:
        p_cutoff = p_limit - cumulative_values[-1]
    native_probabilities = {
        str(token_id): div32(weight, total)
        for token_id, weight in zip(kept_ids, kept_values)
    }
    return result, {
        "native_support_ids": kept_ids,
        "native_conditional_probabilities": native_probabilities,
        "native_denominator_fp32": denominator,
        "native_top_k_scaled_gap": top_k_gap,
        "native_top_p_decision_margin": f32(p_cutoff),
        "native_draw_transition_margins_scaled": draw_margins,
        "native_cdf_at_draws": normalized_cdf,
        "insertion_comparison_count": comparison_count,
        "insertion_comparisons_per_logit": comparison_count / len(logits),
    }


def load_upstream_function(function_node: Any) -> Any:
    import torch

    module = ast.Module(body=[function_node], type_ignores=[])
    ast.fix_missing_locations(module)
    namespace: dict[str, Any] = {
        "torch": torch,
        "Tensor": torch.Tensor,
        "MIN_TEMPERATURE": 1e-5,
    }
    exec(compile(module, "pinned_fish_speech_tower.py", "exec"), namespace)
    return namespace["sample_top_k_top_p"]


def upstream_audit(
    function: Any,
    logits: list[float],
    temperature: float,
    top_k: int,
    top_p: float,
    target_ids: list[int],
    seed: int,
) -> dict[str, Any]:
    import torch

    tensor = torch.tensor(logits, dtype=torch.float32, device="cpu")
    values, sorted_ids = torch.sort(
        tensor.float() / temperature, descending=True, dim=-1
    )
    ranks = torch.arange(values.shape[-1], device=values.device)
    dropped = (values.softmax(-1).cumsum(-1) > top_p) | (ranks >= top_k)
    dropped[..., 0] = False
    probabilities = values.masked_fill(dropped, -torch.inf).softmax(-1)
    kept_ranks = torch.nonzero(~dropped, as_tuple=False).reshape(-1)
    support_ids = [int(sorted_ids[rank].item()) for rank in kept_ranks]
    support_probs = {
        str(int(sorted_ids[rank].item())): float(probabilities[rank].item())
        for rank in kept_ranks
    }
    full_cdf = values.softmax(-1).cumsum(-1)
    upto = min(top_k, len(logits))
    p_boundary = min(abs(float(full_cdf[rank].item()) - top_p) for rank in range(upto))
    p_last = float(full_cdf[upto - 1].item())
    upstream_top_k_gap = float(values[top_k - 1].item() - values[top_k].item())

    original_exponential = torch.Tensor.exponential_
    forced_outputs: dict[str, int | None] = {}
    forced_failures: dict[str, str] = {}
    try:
        for target in sorted(set(target_ids)):
            if target not in support_ids:
                forced_outputs[str(target)] = None
                forced_failures[str(target)] = "scalar target is absent from upstream retained support"
                continue
            target_rank = support_ids.index(target)

            def injected_exponential(
                noise: Any, *args: Any, _rank: int = target_rank, **kwargs: Any
            ) -> Any:
                noise.fill_(1.0e30)
                noise[_rank] = 1.0e-30
                return noise

            torch.Tensor.exponential_ = injected_exponential
            observed = function(
                tensor,
                temperature,
                top_k,
                top_p,
                check_finite=True,
            )
            forced_outputs[str(target)] = int(observed.item())
    finally:
        torch.Tensor.exponential_ = original_exponential

    torch.manual_seed(seed)
    seeded_output = int(
        function(tensor, temperature, top_k, top_p, check_finite=True).item()
    )
    return {
        "upstream_support_ids_cpu": support_ids,
        "upstream_conditional_probabilities_cpu": support_probs,
        "upstream_top_k_scaled_gap_cpu": upstream_top_k_gap,
        "upstream_top_p_min_cdf_margin_cpu": p_boundary,
        "upstream_cdf_at_top_k_cpu": p_last,
        "upstream_scalar_target_forced_ids": forced_outputs,
        "upstream_scalar_target_force_failures": forced_failures,
        "upstream_seeded_cpu_id": seeded_output,
        "upstream_seeded_cpu_id_in_support": seeded_output in support_ids,
        "upstream_support_top_k_tie_order_is_not_api_guaranteed": upstream_top_k_gap == 0.0,
    }


def support_probability_comparison(
    native_info: dict[str, Any], upstream_info: dict[str, Any]
) -> tuple[bool, float]:
    native_probs = native_info["native_conditional_probabilities"]
    upstream_probs = upstream_info["upstream_conditional_probabilities_cpu"]
    return (
        native_info["native_support_ids"] == upstream_info["upstream_support_ids_cpu"],
        max(
            (
                abs(native_probs.get(key, 0.0) - upstream_probs.get(key, 0.0))
                for key in set(native_probs) | set(upstream_probs)
            ),
            default=0.0,
        ),
    )


def build_cases(
    native_rows: dict[str, list[float]], upstream_rows: dict[str, list[float]], function: Any
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    cases: list[dict[str, Any]] = []
    audits: list[dict[str, Any]] = []

    def add_pair(
        profile: str,
        tower_name: str,
        tower_logits: list[float],
        main_logits: list[float],
        main_provenance: dict[str, Any],
        tower_provenance: dict[str, Any],
    ) -> None:
        for role, name, logits, provenance in (
            ("main", f"{profile}.main", main_logits, main_provenance),
            ("tower", f"{profile}.tower", tower_logits, tower_provenance),
        ):
            normal_ids, native_info = native_scalar(logits, TEMPERATURE, TOP_K, TOP_P, DRAWS)
            up_info = upstream_audit(
                function,
                logits,
                TEMPERATURE,
                TOP_K,
                TOP_P,
                normal_ids,
                seed=0xD3A300 + len(cases),
            )
            case: dict[str, Any] = {
                "name": name,
                "role": role,
                "profile": profile,
                "logits": logits,
                "draws": DRAWS,
                "expected_ids": normal_ids,
                "temperature": TEMPERATURE,
                "top_k": TOP_K,
                "top_p": TOP_P,
                "provenance": provenance,
            }
            if role == "main":
                retry_ids, retry_info = native_scalar(
                    logits, RETRY_TEMPERATURE, TOP_K, RETRY_TOP_P, DRAWS
                )
                case["retry_expected_ids"] = retry_ids
                retry_audit = upstream_audit(
                    function,
                    logits,
                    RETRY_TEMPERATURE,
                    TOP_K,
                    RETRY_TOP_P,
                    retry_ids,
                    seed=0xD3B300 + len(cases),
                )
            else:
                retry_info = None
                retry_audit = None
            support_equal, probability_error = support_probability_comparison(
                native_info, up_info
            )
            if retry_info is not None and retry_audit is not None:
                retry_support_equal, retry_probability_error = support_probability_comparison(
                    retry_info, retry_audit
                )
            else:
                retry_support_equal = None
                retry_probability_error = None
            cases.append(case)
            audits.append(
                {
                    "name": name,
                    "role": role,
                    "profile": profile,
                    "provenance": provenance,
                    "native": native_info,
                    "upstream": up_info,
                    "retry_native": retry_info,
                    "retry_upstream": retry_audit,
                    "native_upstream_support_equal": support_equal,
                    "native_upstream_probability_max_abs_error": probability_error,
                    "retry_native_upstream_support_equal": retry_support_equal,
                    "retry_native_upstream_probability_max_abs_error": retry_probability_error,
                    "retry_non_tie_top_k": retry_info is not None
                    and retry_info["native_top_k_scaled_gap"] != 0.0,
                }
            )

    actual_names = [
        "tower.decode.row0.head_logits.0",
        "tower.prefill.row0.head_logits.0",
        "tower.decode.row0.head_logits.8",
    ]
    for trace_name in actual_names:
        tower_logits = native_rows[trace_name]
        # Main head logits are not retained. This constructs an explicit proxy:
        # EOS=0 followed by one native tower head's semantic-size logits.
        main_logits = [0.0, *tower_logits]
        short = trace_name.removeprefix("tower.").replace(".row0.head_logits.", "_head")
        profile = f"tower_trace_{short}_plus_eos_proxy"
        add_pair(
            profile,
            trace_name,
            tower_logits,
            main_logits,
            {
                "kind": "synthetic_tower_trace_plus_eos_proxy",
                "source_trace": trace_name,
                "synthetic_eos_logit": 0.0,
                "actual_main_logits_retained": False,
                "limitation": "not a captured main-head distribution; only a length-4097 data-dependent proxy",
            },
            {
                "kind": "retained_native_pinned_q4_tower_trace",
                "source_trace": trace_name,
                "upstream_comparison_trace": trace_name,
                "model_revision": "5331ab509359c0e4a7e353a19e255ddf3e8152a8",
            },
        )

    for profile, pattern in (
        ("synthetic_ascending_insertion_worst", "ascending"),
        ("synthetic_descending_insertion_low", "descending"),
        ("synthetic_flat_ties", "flat"),
        ("synthetic_k_cutoff_ties", "k_cutoff_ties"),
    ):
        def make_logits(size: int) -> list[float]:
            if pattern == "ascending":
                return [f32(index / 1000.0) for index in range(size)]
            if pattern == "descending":
                return [f32(-index / 1000.0) for index in range(size)]
            if pattern == "flat":
                return [0.0] * size
            # One unique maximum followed by a 31-token tie straddling k=30.
            return [0.01] + [0.0] * 31 + [-0.001] * (size - 32)

        tower_logits = make_logits(4096)
        main_logits = make_logits(4097)
        synthetic_info = {
            "kind": "deterministic_synthetic_sampler_stress_case",
            "pattern": pattern,
            "actual_model_logits": False,
        }
        add_pair(
            profile,
            profile,
            tower_logits,
            main_logits,
            {**synthetic_info, "role_shape": 4097},
            {**synthetic_info, "role_shape": 4096},
        )
    return cases, audits


def _finite_number(value: Any) -> bool:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return False
    try:
        return math.isfinite(float(value))
    except (OverflowError, ValueError):
        return False


def _valid_ids(values: Any, vocabulary_size: int) -> bool:
    return isinstance(values, list) and all(
        isinstance(value, int)
        and not isinstance(value, bool)
        and 0 <= value < vocabulary_size
        for value in values
    )


def verify_fixture(path: Path) -> dict[str, Any]:
    fixture = json.loads(path.read_text(encoding="utf-8"))
    if fixture.get("format") != "drama3-sampler-fixture-v1":
        raise RuntimeError("fixture format mismatch")
    cases = fixture.get("cases")
    if not isinstance(cases, list) or len(cases) != 14:
        raise RuntimeError("fixture should contain 7 paired profiles")
    by_profile: dict[str, list[dict[str, Any]]] = {}
    seen_names: set[str] = set()
    main_cases: list[dict[str, Any]] = []
    for case in cases:
        if not isinstance(case, dict):
            raise RuntimeError("fixture case must be an object")
        name = case.get("name")
        if not isinstance(name, str) or not name or name in seen_names:
            raise RuntimeError("fixture case names must be nonempty and unique")
        seen_names.add(name)
        profile = case.get("profile")
        if not isinstance(profile, str) or not profile:
            raise RuntimeError(f"invalid profile in {name}")
        role = case.get("role")
        if role not in ("main", "tower"):
            raise RuntimeError(f"invalid role in {name}: {role!r}")
        by_profile.setdefault(profile, []).append(case)
        want_size = 4097 if role == "main" else 4096
        logits = case.get("logits")
        if not isinstance(logits, list) or len(logits) != want_size or not all(
            _finite_number(value) for value in logits
        ):
            raise RuntimeError(f"invalid logits in {name}")
        draws = case.get("draws")
        if (
            not isinstance(draws, list)
            or not draws
            or not all(_finite_number(value) and 0.0 <= value <= 1.0 for value in draws)
        ):
            raise RuntimeError(f"invalid draws in {name}")
        if not _finite_number(case.get("temperature")) or case["temperature"] != TEMPERATURE:
            raise RuntimeError(f"unsupported temperature in {name}")
        top_k = case.get("top_k")
        if isinstance(top_k, bool) or not isinstance(top_k, int) or top_k != TOP_K:
            raise RuntimeError(f"unsupported top_k in {name}")
        if not _finite_number(case.get("top_p")) or case["top_p"] != TOP_P:
            raise RuntimeError(f"unsupported top_p in {name}")
        expected_ids = case.get("expected_ids")
        if not _valid_ids(expected_ids, want_size) or len(expected_ids) != len(draws):
            raise RuntimeError(f"normal expected IDs are invalid in {name}")
        recomputed, _ = native_scalar(logits, TEMPERATURE, TOP_K, TOP_P, draws)
        if recomputed != expected_ids:
            raise RuntimeError(f"normal oracle mismatch in {name}")
        if role == "main":
            retry_ids = case.get("retry_expected_ids")
            if not _valid_ids(retry_ids, want_size) or len(retry_ids) != len(draws):
                raise RuntimeError(f"retry expected IDs are invalid in {name}")
            retry, _ = native_scalar(
                logits, RETRY_TEMPERATURE, TOP_K, RETRY_TOP_P, draws
            )
            if retry != retry_ids:
                raise RuntimeError(f"retry oracle mismatch in {name}")
            main_cases.append(case)
        elif "retry_expected_ids" in case:
            raise RuntimeError(f"tower case must not contain retry IDs: {name}")
    for profile, pair in by_profile.items():
        if len(pair) != 2 or {item["role"] for item in pair} != {"main", "tower"}:
            raise RuntimeError(f"profile pairing mismatch: {profile}")

    _, _, _, function_node = pinned_source()
    upstream_function = load_upstream_function(function_node)
    retry_audits: list[dict[str, Any]] = []
    for index, case in enumerate(main_cases):
        logits = case["logits"]
        _, native_info = native_scalar(
            logits, RETRY_TEMPERATURE, TOP_K, RETRY_TOP_P, case["draws"]
        )
        upstream_info = upstream_audit(
            upstream_function,
            logits,
            RETRY_TEMPERATURE,
            TOP_K,
            RETRY_TOP_P,
            target_ids=[],
            seed=0xD3B300 + index,
        )
        support_equal, probability_error = support_probability_comparison(
            native_info, upstream_info
        )
        retry_audits.append(
            {
                "name": case["name"],
                "native_upstream_support_equal": support_equal,
                "native_upstream_probability_max_abs_error": probability_error,
                "native_top_k_scaled_gap": native_info["native_top_k_scaled_gap"],
                "non_tie_top_k": native_info["native_top_k_scaled_gap"] != 0.0,
            }
        )
    non_tie_retries = [item for item in retry_audits if item["non_tie_top_k"]]
    return {
        "status": "verified",
        "case_count": len(cases),
        "profile_count": len(by_profile),
        "all_expected_ids_recomputed": True,
        "retry_case_count": len(retry_audits),
        "retry_support_equal_count": sum(
            item["native_upstream_support_equal"] for item in retry_audits
        ),
        "retry_probability_max_abs_error": max(
            (item["native_upstream_probability_max_abs_error"] for item in retry_audits),
            default=0.0,
        ),
        "non_tie_retry_case_count": len(non_tie_retries),
        "non_tie_retry_support_equal_count": sum(
            item["native_upstream_support_equal"] for item in non_tie_retries
        ),
        "non_tie_retry_probability_max_abs_error": max(
            (
                item["native_upstream_probability_max_abs_error"]
                for item in non_tie_retries
            ),
            default=0.0,
        ),
        "retry_audits": retry_audits,
    }


def preflight_output_paths(*destinations: Path) -> list[Path]:
    outputs: list[Path] = []
    seen: set[Path] = set()
    for destination in destinations:
        output = destination.expanduser().resolve()
        if output in seen:
            continue
        seen.add(output)
        if output.exists() and not output.is_dir():
            raise NotADirectoryError(f"output destination is not a directory: {output}")
        for filename in ("oracle-fixture.json", "oracle-report.json"):
            artifact = output / filename
            if artifact.exists() or artifact.is_symlink():
                raise FileExistsError(
                    f"refusing to overwrite existing oracle artifact: {artifact}"
                )
        outputs.append(output)
    return outputs


def create_outputs(reference_out: Path, run_out: Path) -> None:
    outputs = preflight_output_paths(reference_out, run_out)
    revision, source_hash, function_hash, function_node = pinned_source()
    native_rows, upstream_rows = load_tower_rows()
    cases, case_audits = build_cases(
        native_rows, upstream_rows, load_upstream_function(function_node)
    )
    fixture = {
        "format": "drama3-sampler-fixture-v1",
        "provenance": {
            "oracle": "native-style scalar FP32 port of src/ml/audio/drama3/kernels/resident.metal::dr3_sample",
            "upstream_sampler": "AST-extracted exact sample_top_k_top_p function, CPU execution only",
            "upstream_revision": revision,
            "upstream_source": str(UPSTREAM_SOURCE),
            "upstream_source_sha256": source_hash,
            "upstream_function_sha256": function_hash,
            "native_trace_path": str(NATIVE_TRACES),
            "native_trace_sha256": NATIVE_TRACE_SHA256,
            "upstream_trace_path": str(TORCH_TRACES),
            "upstream_trace_sha256": TORCH_TRACE_SHA256,
            "trace_manifest_sha256": sha256(TORCH_MANIFEST),
            "main_logits_retained": False,
            "sampling_rng_contract": "scalar fixture draws are inverse-CDF uniforms; upstream uses an exponential race, so RNG-identical IDs are not claimed",
        },
        "cases": cases,
    }
    payload = json.dumps(fixture, separators=(",", ":"), allow_nan=False) + "\n"
    for output in outputs:
        output.mkdir(parents=True, exist_ok=True)
        (output / "oracle-fixture.json").write_text(payload, encoding="utf-8")

    fixture_hash = hashlib.sha256(payload.encode("utf-8")).hexdigest()
    # Pair each native row with its retained upstream torch counterpart and
    # record exact observed source deltas rather than imply identical logits.
    trace_comparisons = []
    for trace_name in (
        "tower.decode.row0.head_logits.0",
        "tower.prefill.row0.head_logits.0",
        "tower.decode.row0.head_logits.8",
    ):
        diffs = [abs(a - b) for a, b in zip(native_rows[trace_name], upstream_rows[trace_name])]
        trace_comparisons.append(
            {
                "name": trace_name,
                "native_vs_upstream_torch_max_abs_logit_delta": max(diffs),
                "native_vs_upstream_torch_nonzero_delta_count": sum(value != 0 for value in diffs),
            }
        )

    max_ascending = max(
        audit["native"]["insertion_comparison_count"]
        for audit in case_audits
        if audit["profile"] == "synthetic_ascending_insertion_worst"
    )
    min_flat = min(
        audit["native"]["insertion_comparison_count"]
        for audit in case_audits
        if audit["profile"] == "synthetic_flat_ties"
    )
    report = {
        "format": "drama3-sampler-oracle-report-v1",
        "fixture_path": str(run_out / "oracle-fixture.json"),
        "fixture_sha256": fixture_hash,
        "upstream": {
            "repository": "https://github.com/fishaudio/fish-speech",
            "revision": revision,
            "source_path": str(UPSTREAM_SOURCE),
            "source_sha256": source_hash,
            "function_sha256": function_hash,
            "torch_version": __import__("torch").__version__,
            "function_executed_on": "CPU only",
            "forced_noise_interception": "patched Tensor.exponential_ temporarily so each in-support scalar target wins the upstream probability/noise argmax; method restored in finally",
        },
        "trace_search": {
            "native_trace_sha256": NATIVE_TRACE_SHA256,
            "upstream_torch_trace_sha256": TORCH_TRACE_SHA256,
            "native_trace_rows": 726,
            "upstream_trace_rows": 726,
            "retained_tower_head_logit_rows": 18,
            "retained_main_head_logit_rows": 0,
            "native_upstream_pair_checks": trace_comparisons,
        },
        "numeric_limits": {
            "native_oracle": "FP32 rounding after division, exp, addition, multiply, and subtraction; top-k ties favor lower token ID; uniform inverse-CDF selection mirrors dr3_sample",
            "exp_approximation": "Python math.exp rounded to FP32; Metal exp implementation may differ by ULPs",
            "reduction_order": "denominator is serial FP32 in native oracle as in dr3_sample; upstream torch softmax uses its own CPU reduction/cumsum order",
            "upstream_comparison": "support and conditional probabilities are computed by the exact pinned sampler preprocessing operators on CPU; this is not CUDA kernel arithmetic",
            "rng": "fixture draws are uniforms; upstream source samples exponential noise; no same-seed ID parity claim",
            "ties": "native scalar tie rule is lower token ID; upstream torch.sort is run on CPU and does not promise a stable tie order unless requested, so CPU tie output is diagnostic only",
            "margins": "each case audit records top-k scaled-logit gap, top-p CDF decision margin, and draw-transition distance; zero margins mean exact ties or boundary cases, while positive margins do not bound Metal exp/reduction ULP error",
        },
        "mini_quad": {
            "strongest_proxy_failure": "one-spike and flat synthetic vectors hide data-dependent insertion work: ascending-by-ID logits repeatedly bubble every new token through all 30 retained slots",
            "discriminator": {
                "profile": "synthetic_ascending_insertion_worst",
                "main_comparisons_per_call": next(
                    audit["native"]["insertion_comparison_count"]
                    for audit in case_audits
                    if audit["profile"] == "synthetic_ascending_insertion_worst" and audit["role"] == "main"
                ),
                "tower_comparisons_per_call": next(
                    audit["native"]["insertion_comparison_count"]
                    for audit in case_audits
                    if audit["profile"] == "synthetic_ascending_insertion_worst" and audit["role"] == "tower"
                ),
                "comparison_count_ratio_to_flat_profile_max": max_ascending / min_flat,
                "interpretation": "compare root's per-profile GPU timing across ascending, flat, and retained tower rows; if the ascending cost expands disproportionately, insertion work is a stronger next optimization target than further full-vocabulary denominator reduction",
            },
        },
        "case_audits": case_audits,
        "fixture_verification": None,
    }
    report["fixture_verification"] = verify_fixture(run_out / "oracle-fixture.json")
    report_payload = json.dumps(report, indent=2, allow_nan=False) + "\n"
    for output in outputs:
        (output / "oracle-report.json").write_text(report_payload, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-out", type=Path, default=DEFAULT_REFERENCE_OUTPUT)
    parser.add_argument("--run-out", type=Path, default=DEFAULT_RUN_OUTPUT)
    parser.add_argument("--verify-fixture", type=Path)
    args = parser.parse_args()
    if args.verify_fixture:
        print(json.dumps(verify_fixture(args.verify_fixture), sort_keys=True))
        return
    create_outputs(args.reference_out, args.run_out)
    print(
        json.dumps(
            {
                "status": "generated",
                "fixture": str(args.run_out / "oracle-fixture.json"),
                "report": str(args.run_out / "oracle-report.json"),
                "case_count": 14,
                "profile_count": 7,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
