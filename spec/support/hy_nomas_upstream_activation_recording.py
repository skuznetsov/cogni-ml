#!/usr/bin/env python3
"""Bounded named-boundary recording of an accepted upstream BF16 DiT replay.

This is a diagnostic, not a timing path. Model arithmetic remains on MPS;
lossless activation exports drain each block to external SSD. Opaque SDPA/MPS
scratch, Qwen internals and PixelUMM internals are deliberately out of scope.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import re
import shutil
import sys
import time
from typing import Any

SUPPORT = Path(__file__).resolve().parent
if str(SUPPORT) not in sys.path:
    sys.path.insert(0, str(SUPPORT))
import hy_nomas_soft_route_render_reference as render  # noqa: E402

GIB = 1024**3
SCHEMA = "cogni_ml.hy_nomas_upstream_activation_recording/v1"
BASELINE = Path("/opt/mldata_ssd/cogni-ml/hy-nomas-soft-routes-20261105.j0wgy7yl/render-v1/source.json")
BASELINE_SHA256 = "c1034866068583cb662ac8f03468371df5997781d95413f24a432d0d9ab13d07"
BASELINE_PAYLOAD_SHA256 = "7a6aaee1339832106c52578fa6b2aff2c6f925bb50346153c4ce6bffc8d26d09"
SOURCE_RUNTIME = Path("/opt/mldata_ssd/cogni-ml/hy-nomas-final-euler-20261104.c7zj7lnM/replay_source_autocast_full_base.py")


def sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def model_bindings(model: dict) -> dict:
    def path_only(value: Any) -> str:
        if isinstance(value, Path):
            return str(value)
        raise TypeError(f"unsupported model provenance type: {type(value).__name__}")
    return json.loads(json.dumps({key: model[key] for key in (
        "upstream_path", "upstream_revision", "source_hashes", "cache_index_path",
        "cache_shards", "global_shard", "total_weight_bytes")}, default=path_only))


def diagnostic_source_bindings() -> dict[str, dict[str, str]]:
    paths = [Path(__file__), Path(render.__file__), SOURCE_RUNTIME] + [SUPPORT / f"{name}.py" for name in (
        "hy_nomas_activation_trace", "hy_nomas_dit_route_factorial_reference",
        "hy_nomas_soft_route_reference", "hy_nomas_bf16_dit_quality_control",
        "hy_nomas_continuation_inputs", "hy_nomas_generation_inputs",
        "hy_nomas_q4_dit_reference", "hy_nomas_real_attention_reference",
        "hy_nomas_reference", "hy_nomas_two_segment_quality_control")]
    return {path.name: {"path": str(path.resolve(strict=True)), "sha256": sha256(path)} for path in paths}


def authenticate_baseline(path: Path = BASELINE) -> tuple[dict, Path]:
    if path.is_symlink() or not path.is_file() or sha256(path) != BASELINE_SHA256:
        raise ValueError("accepted source manifest missing or changed")
    baseline = json.loads(path.read_text())
    record = baseline["payload"]
    basename = record["basename"]
    if (not isinstance(basename, str) or Path(basename).name != basename
            or basename in (".", "..")):
        raise ValueError("accepted source payload must be a sibling basename")
    payload = path.parent / basename
    if (payload.is_symlink() or not payload.is_file()
            or payload.stat().st_size != record["bytes"]
            or record["sha256"] != BASELINE_PAYLOAD_SHA256
            or sha256(payload) != BASELINE_PAYLOAD_SHA256):
        raise ValueError("accepted source payload missing or changed")
    return baseline, payload


def fresh_output(path: Path) -> Path:
    if not path.is_absolute() or path.exists() or path.is_symlink():
        raise ValueError("recording output must be a fresh absolute directory")
    parent = path.parent.resolve(strict=True)
    if parent != path.parent:
        raise ValueError("recording output parent must be canonical")
    if not parent.is_relative_to(Path("/opt/mldata_ssd").resolve(strict=True)):
        raise ValueError("activation recording must use the external data SSD")
    p = plan()
    if shutil.disk_usage(parent).free < p["planned_total_upper_bound_bytes"] + p["disk_reserve_bytes"]:
        raise ValueError("insufficient external SSD space for the complete bounded trace")
    return path


def block_shapes() -> dict[str, tuple[int, ...]]:
    """Exact named-boundary inventory for one compact source block call."""
    d = render.direct
    result = {"input.vec": (1, d.HIDDEN),
              "rope.cos": (d.VIDEO_ROWS, d.HEAD_DIM),
              "rope.sin": (d.VIDEO_ROWS, d.HEAD_DIM)}
    for stream, rows in (("img", d.VIDEO_ROWS), ("txt", d.TEXT_STORAGE_ROWS)):
        shape, head = (1, rows, d.HIDDEN), (1, rows, d.HEADS, d.HEAD_DIM)
        for name in ("input", "norm1", "qkv_input", "q_projection", "k_projection",
                     "v_projection", "attn_projected", "after_attention", "norm2",
                     "ff_input", "ff_projected", "output", "attention_output"):
            result[f"{stream}.{name}"] = shape
        for name in ("q_norm", "k_norm", "attention_q", "attention_k", "attention_v"):
            result[f"{stream}.{name}"] = head
        for name in ("ff_hidden", "ff_activated"):
            result[f"{stream}.{name}"] = (1, rows, 4 * d.HIDDEN)
        result[f"{stream}.mod_silu"] = (1, d.HIDDEN)
        result[f"{stream}.modulation"] = (1, 6 * d.HIDDEN)
    for name in ("rope.q_before", "rope.k_before", "rope.q_after", "rope.k_after"):
        result[name] = (1, d.VIDEO_ROWS, d.HEADS, d.HEAD_DIM)
    for name in ("cache.k", "cache.v"):
        result[name] = (1, d.PHYSICAL_CACHE_ROWS, d.HEADS, d.HEAD_DIM)
    return result


def block_dtypes() -> dict[str, str]:
    # MPS autocast keeps affine-free LayerNorm and its modulated result F32.
    # Preserve that source boundary; never round it to fit the trace estimate.
    f32 = {"rope.cos", "rope.sin"} | {
        f"{stream}.{name}" for stream in ("img", "txt")
        for name in ("norm1", "qkv_input", "norm2", "ff_input")}
    return {name: "float32" if name in f32 else "bfloat16" for name in block_shapes()}


def plan() -> dict[str, Any]:
    shapes = block_shapes()
    dtypes = block_dtypes()
    batch = sum(math.prod(shape) * (4 if dtypes[name] == "float32" else 2)
                for name, shape in shapes.items())
    calls = 4 * 54 * 2
    return {"schema": SCHEMA, "plan_only_does_not_launch_model": True, "steps": 4,
            "blocks": 54, "block_calls": calls, "block_records_per_call": len(shapes),
            "block_batch_bytes": batch, "all_block_bytes": batch * calls,
            "auxiliary_batches": 8, "auxiliary_batch_ceiling_bytes": GIB,
            "planned_total_upper_bound_bytes": batch * calls + 8 * GIB,
            "max_batch_bytes": GIB, "max_total_bytes": 300 * GIB,
            "disk_reserve_bytes": 32 * GIB, "process_cap_mib": 16384,
            "memory_floor_pct": 10, "swap_growth_cap_mib": 64,
            "scope": "named DiT boundaries plus prelude/refiner/head outputs; no fused MPS scratch or decoder/Qwen internals",
            "recording_exports_are_diagnostic": True}


class Observer:
    """Observe exact source tensors without substituting arithmetic results."""
    def __init__(self, writer: Any, arm: str, *, pilot: bool = False):
        self.writer, self.arm, self.pilot = writer, arm, pilot
        self.context: str | None = None
        self.local: set[str] = set()
        self.hooks: list[Any] = []
        self.restore: list[tuple[dict, str, Any]] = []
        self.expected: set[str] = set()
        self.block_calls = 0
        self.exports = 0
        self.pilot_checked = False
        self.pilot_calls: list[dict] = []

    def keep(self, name: str, value: Any) -> None:
        if self.context is None:
            return
        if name in self.local:
            raise ValueError(f"duplicate activation boundary: {name}")
        if ".block" in self.context:
            expected = block_shapes().get(name)
            if expected is None or tuple(value.shape) != expected:
                raise ValueError(f"block activation shape mismatch: {name}")
            # The byte plan must describe actual source storage, not merely a
            # hoped-for precision. Refuse an unexpected promotion before clone.
            expected_dtype = getattr(self.torch, block_dtypes()[name])
            if value.dtype != expected_dtype:
                raise ValueError(f"block activation dtype mismatch: {name}: {value.dtype}")
        self.local.add(name)
        key = f"{self.context}.{name}"
        self.writer.record(key, value)
        self.expected.add(key)
        if self.pilot and self.pilot_calls and name in ("img.norm1", "txt.norm1"):
            self.pilot_calls[-1].setdefault("values", {})[name] = value.detach().clone()

    def _context(self, name: str) -> None:
        if self.context is not None:
            raise ValueError("activation batch was not drained")
        self.context, self.local = name, set()

    def _drain(self) -> None:
        self.writer.flush()
        self.exports += 1
        self.context = None

    def install(self, torch: Any, block: Any, globals_: dict, state: dict) -> None:
        self.torch, self.block = torch, block
        self.state = state
        if self.pilot:
            def copy_value(value: Any) -> Any:
                if isinstance(value, torch.Tensor):
                    if value.device.type != "mps":
                        raise ValueError("pilot source call escaped MPS")
                    return value.detach().clone()
                if isinstance(value, dict):
                    return {key: copy_value(item) for key, item in value.items()}
                if isinstance(value, (list, tuple)):
                    return type(value)(copy_value(item) for item in value)
                return value
            def remember_call(_m: Any, args: Any, kwargs: dict) -> None:
                if args:
                    raise ValueError("pilot requires the source keyword call boundary")
                self.pilot_calls.append({"kwargs": copy_value(kwargs),
                    "call": dict(state["call"]), "case": state["case"],
                    "arm": self.current_call[2]})
            self.hooks.append(block.register_forward_pre_hook(remember_call, with_kwargs=True))
        def output(module: Any, name: str) -> None:
            self.hooks.append(module.register_forward_hook(
                lambda _m, _a, value, key=name: self.keep(key, value)))
        def input_(module: Any, name: str) -> None:
            self.hooks.append(module.register_forward_pre_hook(
                lambda _m, args, key=name: self.keep(key, args[0])))
        for stream in ("img", "txt"):
            mod = getattr(block, f"{stream}_mod")
            output(mod.act, f"{stream}.mod_silu")
            output(mod, f"{stream}.modulation")
            for attr, name in (("norm1", "norm1"), ("attn_q", "q_projection"),
                               ("attn_k", "k_projection"), ("attn_v", "v_projection"),
                               ("attn_q_norm", "q_norm"), ("attn_k_norm", "k_norm"),
                               ("attn_proj", "attn_projected"), ("norm2", "norm2")):
                output(getattr(block, f"{stream}_{attr}"), f"{stream}.{name}")
            input_(getattr(block, f"{stream}_attn_q"), f"{stream}.qkv_input")
            input_(getattr(block, f"{stream}_norm2"), f"{stream}.after_attention")
            mlp = getattr(block, f"{stream}_mlp")
            input_(mlp.fc1, f"{stream}.ff_input")
            for attr, name in (("fc1", "ff_hidden"), ("act", "ff_activated"), ("fc2", "ff_projected")):
                output(getattr(mlp, attr), f"{stream}.{name}")
        namespace = block.forward_with_cache.__globals__
        original_rope = namespace["apply_rotary_emb"]
        original_attention = namespace["causal_attention_with_cache_mp"]
        def rope(q: Any, k: Any, freqs: Any, **kwargs: Any) -> Any:
            self.keep("rope.q_before", q)
            self.keep("rope.k_before", k)
            result = original_rope(q, k, freqs, **kwargs)
            self.keep("rope.q_after", result[0])
            self.keep("rope.k_after", result[1])
            return result
        def attention(q: Any, k: Any, v: Any, **kwargs: Any) -> Any:
            for stream, index in (("img", 0), ("txt", 1)):
                for name, tensors in (("q", q), ("k", k), ("v", v)):
                    self.keep(f"{stream}.attention_{name}", tensors[index])
            result = original_attention(q, k, v, **kwargs)
            for stream, value in zip(("img", "txt"), result):
                self.keep(f"{stream}.attention_output", value)
            for name in ("k", "v"):
                self.keep(f"cache.{name}", kwargs["kv_cache"][name])
            return result
        for key, replacement in (("apply_rotary_emb", rope), ("causal_attention_with_cache_mp", attention)):
            self.restore.append((namespace, key, namespace[key]))
            namespace[key] = replacement
        # Leaf modules expose refiner/head computation, including its attention
        # projections. Functional SDPA scratch remains excluded from this trace.
        for root_name in ("time_in", "txt_in", "img_in", "final_layer"):
            root = globals_[root_name]
            for name, module in root.named_modules():
                if any(module.children()) or isinstance(module, (torch.nn.Identity, torch.nn.Dropout)):
                    continue
                key = f"{root_name}.{name or 'output'}"
                def hook(_m: Any, _a: Any, value: Any, key: str = key) -> None:
                    if self.context is not None and ".block" not in self.context:
                        if isinstance(value, torch.Tensor) and value.is_floating_point():
                            self.keep(key, value)
                self.hooks.append(module.register_forward_hook(hook))

    def begin_prelude(self, step: int) -> None:
        self._context(f"step{step + 1}.prelude")

    def end_prelude(self, patches: dict) -> None:
        for arm, value in patches.items():
            self.keep(f"{arm}.patch", value)
        self._drain()

    def begin_block(self, step: int, layer: int, arm: str, image: Any, text: Any, vector: Any, rope: Any) -> None:
        self._context(f"step{step + 1}.{arm}.block{layer:02d}")
        self.current_call = (step, layer, arm)
        self.keep("img.input", image)
        self.keep("txt.input", text)
        self.keep("input.vec", vector)
        self.keep("rope.cos", rope[0])
        self.keep("rope.sin", rope[1])

    def end_block(self, outputs: list[Any]) -> None:
        self.keep("img.output", outputs[0])
        self.keep("txt.output", outputs[1])
        expected = block_shapes()
        if self.local != set(expected):
            raise ValueError(f"block trace inventory mismatch: missing={set(expected)-self.local}, extra={self.local-set(expected)}")
        # Shapes are checked before marking this batch complete; original dtype
        # bytes are retained, including any source F32 boundaries.
        self._drain()
        self.block_calls += 1
        print(json.dumps({"event": "block_trace_flushed", "step": self.current_call[0] + 1,
                          "layer": self.current_call[1], "arm": self.current_call[2],
                          "block_calls": self.block_calls, "monotonic": time.monotonic()}), flush=True)
        if self.pilot and not self.pilot_checked:
            self.pilot_calls[-1]["outputs"] = [value.detach().clone() for value in outputs]
            # Exercise the cache-writing and cache-reading paths in one layer.
            if self.block_calls == 2:
                self.pilot_checked = True
                raise PilotComplete(self.pilot_calls)

    def begin_head(self, step: int, arm: str) -> None:
        self._context(f"step{step + 1}.{arm}.head")

    def end_head(self, step: int, arm: str, prediction: Any, latent: Any) -> None:
        self.keep("velocity", prediction)
        self.keep("after_euler", latent)
        self._drain()

    def receipt(self) -> dict[str, Any]:
        return {"block_calls": self.block_calls, "batch_exports": self.exports,
                "record_count": len(self.expected), "source_math_device": "mps",
                "exports_add_diagnostic_waits": True}

    def expected_scope(self, *, pilot: bool) -> set[str]:
        """Reject skipped/repeated steps or layers independently of call counts."""
        contexts = {(0, 0, "T0"), (0, 0, self.arm)} if pilot else {
            (step, layer, arm) for step in range(4) for layer in range(54)
            for arm in ("T0", self.arm)}
        blocks = {f"step{step + 1}.{arm}.block{layer:02d}.{name}"
                  for step, layer, arm in contexts for name in block_shapes()}
        observed_blocks = {name for name in self.expected if re.match(r"^step[1-4]\.[^.]+\.block[0-9]{2}\.", name)}
        if observed_blocks != blocks or self.block_calls != len(contexts):
            raise ValueError("recording skipped or repeated a block context")
        # Refiner/head leaf inventory is discovered from the pinned modules.
        # Every later step must expose the identical set of observed leaves.
        leaves = self.expected - blocks
        prelude_first = {name.removeprefix("step1.prelude.") for name in leaves
                         if name.startswith("step1.prelude.")}
        if not prelude_first:
            raise ValueError("recording omitted prelude leaf outputs")
        expected_leaves = {f"step{step + 1}.prelude.{name}"
                           for step in range(1 if pilot else 4) for name in prelude_first}
        if not pilot:
            prefix = f"step1.{self.arm}.head."
            head_first = {name.removeprefix(prefix) for name in leaves if name.startswith(prefix)}
            if not {"velocity", "after_euler"}.issubset(head_first) or len(head_first) <= 2:
                raise ValueError("recording omitted head leaf outputs")
            expected_leaves |= {f"step{step + 1}.{self.arm}.head.{name}"
                                for step in range(4) for name in head_first}
        if leaves != expected_leaves:
            raise ValueError("recording prelude/head inventory changed between steps")
        return blocks | expected_leaves

    def uninstall(self) -> None:
        for hook in self.hooks:
            hook.remove()
        self.hooks.clear()
        for namespace, key, original in reversed(self.restore):
            namespace[key] = original
        self.restore.clear()


class PilotComplete(Exception):
    def __init__(self, calls: list[dict]):
        super().__init__("bounded two-call pilot completed")
        self.calls = calls


def compare_replay(captures: dict, baseline: dict, payload: Path, arm: str) -> list[dict]:
    import numpy as np
    records = baseline["tensors"]
    result = []
    for step in range(1, 5):
        name = f"{arm}.step{step}.latent"
        record = records[name]
        expected = np.memmap(payload, dtype="<f4", mode="r", offset=record["offset_bytes"], shape=tuple(record["shape"]))
        actual = captures[name]
        if actual.dtype != np.dtype("<f4") or actual.shape != expected.shape or not np.isfinite(actual).all():
            raise ValueError(f"recorded Euler boundary geometry/dtype mismatch: {name}")
        exact = np.array_equal(expected.view("<u4"), actual.view("<u4"))
        result.append({"name": name, "bitwise_equal": bool(exact),
                       "max_abs_error": float(np.max(np.abs(actual - expected)))})
        if not exact:
            raise ValueError(f"recording changed the accepted upstream Euler boundary: {name}")
    return result


def validate_pilot_payload(directory: Path, calls: list[dict], index: dict) -> list[dict]:
    """Compare serialized words with an independent F32-expanded source export.

    The writer reinterprets BF16 as uint16 before its CPU export. This check
    instead expands the live source to F32 on MPS, then compares its high words
    with the saved BF16 words. It also exercises original F32 LayerNorm bytes.
    Full shard hashes must already have been validated by validate_trace().
    """
    import numpy as np
    records = {record["name"]: record for record in index["records"]}
    result = []
    incoming_cache = calls[1]["kwargs"]["kv_cache"]
    for call in calls:
        values = dict(call["values"])
        values.update(zip(("img.output", "txt.output"), call["outputs"]))
        values.update({f"cache.{name}": incoming_cache[name] for name in ("k", "v")})
        for name, source in values.items():
            key = f"step1.{call['arm']}.block00.{name}"
            record = records[key]
            words = source.detach().float().cpu().numpy().view("<u4")
            if record["dtype"] == "bfloat16":
                if np.any(words & 0xffff):
                    raise ValueError(f"pilot source BF16 expansion is not exact: {key}")
                words = (words >> 16).astype("<u2")
            elif record["dtype"] != "float32":
                raise ValueError(f"unexpected pilot serialization dtype: {key}")
            stored = np.memmap(directory / record["shard"], mode="r", dtype=words.dtype,
                               offset=record["offset"], shape=tuple(record["shape"]))
            equal = np.array_equal(words, stored)
            result.append({"name": key, "dtype": record["dtype"], "bitwise_equal": bool(equal)})
            if not equal:
                raise ValueError(f"pilot serialization changed source words: {key}")
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--plan-only", action="store_true")
    mode.add_argument("--pilot", action="store_true")
    mode.add_argument("--execute", action="store_true")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--arm", choices=render.CASE_ORDER, default="A1_high")
    args = parser.parse_args(argv)
    if args.plan_only:
        print(json.dumps(plan(), sort_keys=True))
        return 0
    if args.output is None:
        parser.error("--output is required for a recording")
    args.output = fresh_output(args.output)
    from hy_nomas_activation_trace import ActivationTraceWriter, validate_trace
    checked = render.authenticate_inputs()
    # Authenticate the accepted source payload independently of trace metadata.
    baseline, payload = authenticate_baseline()
    model = render.direct.authenticate_model_sources(render.DEFAULT_CACHE_INDEX, render.DEFAULT_UPSTREAM)
    p = plan()
    metadata = {"schema": SCHEMA, "plan": p, "arm": args.arm,
                "accepted_source": {"path": str(BASELINE.resolve()), "sha256": sha256(BASELINE)},
                "inputs": render._input_document(checked),
                "model": model_bindings(model),
                "upstream_revision": render.direct.PINNED_REVISION,
                "render_helper_sha256": sha256(Path(render.__file__)),
                "recorder_sha256": sha256(Path(__file__)),
                "diagnostic_sources": diagnostic_source_bindings(),
                "precision": "upstream BF16 MPS autocast; original activation storage dtype",
                "scope": p["scope"], "diagnostic_only": True}
    args.output.mkdir(mode=0o700)
    writer = ActivationTraceWriter(args.output / "activations", p["max_total_bytes"], p["max_batch_bytes"],
                                   p["disk_reserve_bytes"], metadata)
    observer = Observer(writer, args.arm, pilot=args.pilot)
    try:
        try:
            captures, receipt = render._run_source(checked, model, render.DEFAULT_UPSTREAM,
                                                  cases=(args.arm,), observer=observer)
        except PilotComplete as stop:
            observer.uninstall()
            torch = observer.torch
            metrics = []
            if [call["arm"] for call in stop.calls] != ["T0", args.arm]:
                raise ValueError("pilot did not capture the writer then reader call")
            with torch.inference_mode(), torch.autocast("mps", dtype=torch.bfloat16):
                for call in stop.calls:
                    observer.state["call"], observer.state["case"] = call["call"], call["case"]
                    replay = observer.block(**call["kwargs"])
                    metrics.extend((a.contiguous().view(torch.int16) == b.contiguous().view(torch.int16)).all()
                                   for a, b in zip(call["outputs"], replay))
                written, incoming = (call["kwargs"] for call in stop.calls)
                metrics.extend((written["kv_cache"][key].view(torch.int16)
                                == incoming["kv_cache"][key].view(torch.int16)).all()
                               for key in ("k", "v"))
                metrics.append((written["cache_mask"] == incoming["cache_mask"]).all())
                same = torch.stack(metrics)
            if not bool(same.all().cpu().item()):
                raise ValueError("traced/untraced source block pilot changed output or cache lineage")
            index = validate_trace(args.output / "activations", require_complete=False)
            serialized = validate_pilot_payload(args.output / "activations", stop.calls, index)
            writer.finish(observer.expected_scope(pilot=True))
            validate_trace(args.output / "activations")
            (args.output / "replay.json").write_text(json.dumps({"schema": SCHEMA, "pilot": True,
                "block_outputs_bitwise_equal": True, "cache_lineage_bitwise_equal": True,
                "serialized_boundary_gates": serialized, "trace_hashes_validated": True,
                "pilot_arms": [call["arm"] for call in stop.calls],
                "runtime": observer.receipt()}, indent=2) + "\n")
            return 0
        if observer.block_calls != 432:
            raise ValueError("single-arm recording did not execute all 432 block calls")
        gates = compare_replay(captures, baseline, payload, args.arm)
        import numpy as np
        latent_path = args.output / "final-latent.f32le"
        with latent_path.open("xb") as stream:
            np.asarray(captures[f"{args.arm}.step4.latent"], dtype="<f4").tofile(stream)
        print(json.dumps({"event": "validating_serialized_trace", "monotonic": time.monotonic()}), flush=True)
        index = validate_trace(args.output / "activations", require_complete=False)
        writer.finish(observer.expected_scope(pilot=False))
        (args.output / "replay.json").write_text(json.dumps({"schema": SCHEMA, "pilot": False,
            "all_euler_boundaries_exact": True, "gates": gates, "runtime": receipt,
            "trace_hashes_validated": True, "trace_records": len(index["records"]),
            "trace_payload_bytes": index["total_bytes"],
            "final_latent": {"path": str(latent_path.resolve()), "sha256": sha256(latent_path),
                             "shape": list(render.LATENT_SHAPE), "dtype": "float32"}}, indent=2) + "\n")
        return 0
    finally:
        observer.uninstall()


if __name__ == "__main__":
    raise SystemExit(main())
