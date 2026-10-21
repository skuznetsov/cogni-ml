#!/usr/bin/env python3
"""Compare staged native MPS attention candidates with PyTorch 2.8 SDPA.

This diagnostic uses fixed B=1,Q=128,K=36608,H=40,D=128 F16 inputs. It
allocates Q/K/V on MPS, fills them through the native bridge, and shares those
same storage buffers with PyTorch, Wan's pinned fallback, and the local native
attention bridge. It never exports activation data.
"""

from __future__ import annotations

import argparse
import ctypes
import hashlib
import json
import os
import platform
import struct
import sys
import time
from pathlib import Path
from typing import Any

sys.dont_write_bytecode = True


Q, K, H, D = 128, 36_608, 40, 128
Q_N = Q * H * D
K_N = K * H * D
S_N = H * Q * K
F16_BYTES = 2
F32_BYTES = 4
SCALE = 1.0 / (D**0.5)
TOLERANCE = {"atol": 2e-5, "rtol": 2e-5}
MEMORY_ROOT = Path("/opt/mldata_ssd")
DEFAULT_WAN_SOURCE = Path(
    "/opt/mldata_ssd/cogni-ml/video-scene-20261019/wan-audit/wan_modules_attention.py"
)
WAN_SOURCE_SHA256 = "23fe7c6f6e4065242d95e5e188cb2d1a16bd283f05dca7e7e878977158fcfdbc"
FIXTURE_FORMULA = (
    "avalanche32(native_flat_index + salt*0x9e3779b9); "
    "((hash & 1023) - 512) / 256; salts Q=1,K=2,V=3"
)
MIB = 1024**2
GIB = 1024**3
EXPLICIT_CAP_BYTES = 5 * GIB


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def candidate_stage_order(phase: str) -> str:
    if phase == "qk-scale":
        return "QK -> optional epsilon for F32 QK -> F32 scale"
    if phase == "round":
        return "QK -> optional epsilon for F32 QK -> F32 scale -> F16 round"
    if phase == "softmax":
        return (
            "QK -> optional epsilon for F32 QK -> F32 scale -> F16 round "
            "-> softmax (ends before PV)"
        )
    if phase == "wide":
        return (
            "F16 inputs -> F32 arithmetic through QK/scale/softmax/PV "
            "-> F16 final cast"
        )
    return "upstream SDPA and native endpoint comparisons"


def phase_estimates() -> dict[str, int]:
    q_bytes = Q_N * F16_BYTES
    kv_bytes = K_N * F16_BYTES
    score16 = S_N * F16_BYTES
    score32 = S_N * F32_BYTES
    # Includes all simultaneously live explicit tensors, not private MPSGraph
    # workspaces or allocator caching.
    common = q_bytes + 2 * kv_bytes + 8 * q_bytes + 256 * 5 * 4
    return {
        "endpoints": (
            common + 2 * score16
        ),
        "qk-scale": common + 2 * score16 + 2 * score32,
        "round": common + 3 * score16 + score32,
        "softmax": common + 4 * score16 + score32,
        "wide": common + 2 * score16,
    }


def plan_payload(phase: str) -> dict[str, Any]:
    estimates = phase_estimates()
    return {
        "status": "plan_only",
        "phase": phase,
        "geometry": {"B": 1, "Q": Q, "K": K, "H": H, "D": D, "dtype": "float16"},
        "elements": {"q": Q_N, "k_v": K_N, "score_probability": S_N},
        "explicit_user_tensor_estimate_bytes": estimates[phase],
        "explicit_cap_bytes": EXPLICIT_CAP_BYTES,
        "explicit_estimate_under_cap": estimates[phase] <= EXPLICIT_CAP_BYTES,
        "mpsgraph_workspace_included": False,
        "tolerance": TOLERANCE,
        "scale_f32": struct.unpack("<f", struct.pack("<f", SCALE))[0],
        "candidate_stage_order": candidate_stage_order(phase),
        "fixture_formula": FIXTURE_FORMULA,
        "scratch_root": str(MEMORY_ROOT),
        "torch_imported_in_process": "torch" in sys.modules,
        "external_memory_guard_required": {
            "process_footprint_limit_mib": 5120,
            "runtime_free_floor_percent": 15,
        },
    }


def _below(root: Path, child: Path) -> bool:
    try:
        child.resolve().relative_to(root.resolve())
        return True
    except ValueError:
        return False


def _ctypes_library(path: Path) -> ctypes.CDLL:
    lib = ctypes.CDLL(str(path))
    p = ctypes.c_void_p
    i32 = ctypes.c_int32
    u64 = ctypes.c_uint64
    lib.wan_stage_create.argtypes = [i32, ctypes.POINTER(p), ctypes.POINTER(p)]
    lib.wan_stage_create.restype = p
    lib.wan_stage_encode.argtypes = [p]
    lib.wan_stage_encode.restype = i32
    lib.wan_stage_release.argtypes = [p]
    lib.wan_stage_release.restype = None
    lib.wan_stage_fill.argtypes = [p, u64, i32]
    lib.wan_stage_fill.restype = i32
    lib.wan_stage_legacy_private_rejected.argtypes = [
        ctypes.POINTER(p), ctypes.POINTER(p),
    ]
    lib.wan_stage_legacy_private_rejected.restype = i32
    lib.wan_stage_compare.argtypes = [
        p, p, p, u64, i32, i32, i32, i32, i32,
    ]
    lib.wan_stage_compare.restype = i32
    return lib


class Probe:
    def __init__(self, torch: Any, library: ctypes.CDLL, slots: list[str]):
        self.torch = torch
        self.lib = library
        self.slots = slots
        self.slot_index = {name: i for i, name in enumerate(slots)}
        self.metrics = torch.zeros((len(slots), 5), device="mps", dtype=torch.int32)
        self.plans: list[ctypes.c_void_p] = []
        self.pointer_arrays: list[Any] = []
        self.tensors: list[Any] = []

    def retain(self, *tensors: Any) -> None:
        self.tensors.extend(tensors)

    @staticmethod
    def pointer(tensor: Any) -> ctypes.c_void_p:
        if tensor.device.type != "mps" or not tensor.is_contiguous():
            raise RuntimeError("native buffers must be contiguous MPS tensors")
        if tensor.storage_offset() != 0:
            raise RuntimeError("native buffers must have zero storage offset")
        storage = tensor.untyped_storage()
        required = tensor.numel() * tensor.element_size()
        if storage.nbytes() < required:
            raise RuntimeError("native buffer storage is shorter than the tensor")
        return ctypes.c_void_p(storage.data_ptr())

    def make_stage(self, kind: int, inputs: list[Any], outputs: list[Any]) -> ctypes.c_void_p:
        # Native plans borrow Tensor-owned MTLBuffers. Keep every source and
        # destination alive until the queued GPU work and final metrics readback.
        self.retain(*inputs, *outputs)
        in_array = (ctypes.c_void_p * len(inputs))(*(self.pointer(x) for x in inputs))
        out_array = (ctypes.c_void_p * len(outputs))(*(self.pointer(x) for x in outputs))
        plan = self.lib.wan_stage_create(kind, in_array, out_array)
        if not plan:
            raise RuntimeError(f"wan_stage_create(kind={kind}) failed")
        self.pointer_arrays.extend((in_array, out_array))
        self.plans.append(plan)
        return plan

    def encode(self, plan: ctypes.c_void_p, kind: int) -> None:
        result = self.lib.wan_stage_encode(plan)
        if result != 0:
            raise RuntimeError(f"wan_stage_encode(kind={kind}) returned {result}")

    def fill(self, tensor: Any, salt: int) -> None:
        count = tensor.numel()
        expected = Q_N if count == Q_N else K_N if count == K_N else -1
        if count != expected:
            raise RuntimeError(f"unexpected fixture element count {count}")
        result = self.lib.wan_stage_fill(self.pointer(tensor), count, salt)
        if result != 0:
            raise RuntimeError(f"wan_stage_fill(salt={salt}) returned {result}")

    def compare(
        self,
        label: str,
        actual: Any,
        reference: Any,
        *,
        n: int,
        actual_type: int = 1,
        reference_type: int = 1,
        layout: int = 0,
        control: int = 0,
    ) -> None:
        expected_dtypes = {0: self.torch.float32, 1: self.torch.float16}
        if actual_type not in expected_dtypes or reference_type not in expected_dtypes:
            raise ValueError("comparison tensor type must be declared as F32 or F16")
        if actual.dtype != expected_dtypes[actual_type]:
            raise ValueError(f"{label}: actual tensor dtype disagrees with declared type")
        if reference.dtype != expected_dtypes[reference_type]:
            raise ValueError(f"{label}: reference tensor dtype disagrees with declared type")
        if n <= 0 or n != actual.numel() or n != reference.numel():
            raise ValueError(f"{label}: comparison span must equal both tensor sizes")
        if self.metrics.dtype != self.torch.int32 or not self.metrics.is_contiguous():
            raise RuntimeError("metric buffer must be contiguous int32")
        if self.metrics.numel() != len(self.slots) * 5:
            raise RuntimeError("metric buffer must contain five int32 values per slot")
        slot = self.slot_index[label]
        if slot < 0 or (slot + 1) * 5 > self.metrics.numel():
            raise RuntimeError(f"{label}: metric slot is outside the allocated buffer")
        result = self.lib.wan_stage_compare(
            self.pointer(actual),
            self.pointer(reference),
            self.pointer(self.metrics),
            n,
            actual_type,
            reference_type,
            layout,
            slot,
            control,
        )
        if result != 0:
            raise RuntimeError(f"wan_stage_compare({label}) returned {result}")

    def read_metrics_once(self) -> dict[str, dict[str, int | float]]:
        # This is the sole host readback. Inputs and all activation outputs stay
        # on MPS; only the compact integer metric table crosses to CPU.
        rows = self.metrics.cpu().tolist()
        result: dict[str, dict[str, int | float]] = {}
        for name, row in zip(self.slots, rows, strict=True):
            result[name] = {
                "changed": int(row[0]),
                "violations": int(row[1]),
                "nonfinite": int(row[2]),
                "max_abs": _float_from_bits(row[3]),
                "max_tolerance_ratio": _float_from_bits(row[4]),
            }
        return result

    def release(self) -> None:
        for plan in reversed(self.plans):
            self.lib.wan_stage_release(plan)
        self.plans.clear()


def _float_from_bits(value: int) -> float:
    return struct.unpack("<f", struct.pack("<I", value & 0xFFFFFFFF))[0]


def _run_phase(args: argparse.Namespace) -> dict[str, Any]:
    if os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK") != "0":
        raise RuntimeError("MPS fallback must be disabled")
    if not args.library or not args.library.is_file():
        raise FileNotFoundError("--library must name the built native bridge")
    if args.output and not _below(MEMORY_ROOT, args.output):
        raise ValueError("--output must be under /opt/mldata_ssd")
    if args.output and args.output.exists():
        raise FileExistsError(f"result already exists: {args.output}")
    if not args.wan_source.is_file():
        raise FileNotFoundError(f"pinned Wan source missing: {args.wan_source}")

    # Import only after fallback has been disabled. The reference module also
    # verifies the pinned upstream source hash and blocks FlashAttention.
    import torch

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import wan_s2v_attention_precision_reference as oracle

    if not torch.backends.mps.is_available():
        raise RuntimeError("MPS is required")
    if torch.__version__.split("+", 1)[0] != "2.8.0":
        raise RuntimeError(f"PyTorch 2.8.0 is required, got {torch.__version__}")
    if platform.system() != "Darwin":
        raise RuntimeError("this diagnostic requires macOS")
    macos_version = platform.mac_ver()[0]
    try:
        macos_major = int(macos_version.split(".", 1)[0])
    except (AttributeError, ValueError):
        raise RuntimeError(f"cannot determine macOS version: {macos_version!r}")
    if macos_major < 15:
        raise RuntimeError(
            f"PyTorch 2.8 MPS math SDPA dispatch requires macOS 15 or newer, got {macos_version}"
        )
    if not hasattr(torch.ops.aten, "_scaled_dot_product_attention_math_for_mps"):
        raise RuntimeError("pinned PyTorch MPS math SDPA op is unavailable")

    source_sha = oracle.sha256_file(args.wan_source.resolve())
    if source_sha != WAN_SOURCE_SHA256:
        raise RuntimeError("pinned Wan source SHA-256 mismatch")
    upstream = oracle.load_pinned_attention_source(args.wan_source.resolve())
    repo_root = Path(__file__).resolve().parents[2]
    bridge_source = repo_root / "examples/support/wan_attention_f16_stages_bridge.mm"
    production_source = repo_root / "src/ml/video/wan_attention_mps_bridge.mm"
    if not bridge_source.is_file() or not production_source.is_file():
        raise FileNotFoundError("native bridge source is missing")

    lib = _ctypes_library(args.library.resolve())
    q = torch.empty((1, Q, H, D), device="mps", dtype=torch.float16)
    k = torch.empty((1, K, H, D), device="mps", dtype=torch.float16)
    v = torch.empty((1, K, H, D), device="mps", dtype=torch.float16)
    if not all(t.is_contiguous() and t.storage_offset() == 0 for t in (q, k, v)):
        raise RuntimeError("Q/K/V storage must be contiguous and zero-offset")

    if args.phase == "endpoints":
        slots = [
            "actual_y_finite", "actual_p_finite", "wan_vs_actual_y",
            "prefix6_p_vs_actual_p", "prefix6_ctx_vs_actual_y",
            "prefix5_vs_actual_y", "isolated_actual_pv_vs_actual_y",
            "opaque_vs_actual_y", "prefix5_vs_opaque", "control_plus_one",
            "control_nan",
        ]
    elif args.phase == "qk-scale":
        slots = [
            "prefix_qk_vs_torch_matmul",
            "prefix_scale32_vs_torch", "isolated_scale32_vs_torch",
        ]
    elif args.phase == "round":
        slots = ["prefix_scaled16_vs_torch", "isolated_scaled16_vs_torch"]
    elif args.phase == "softmax":
        slots = [
            "actual_p_finite", "prefix_softmax_vs_torch",
            "isolated_softmax_vs_torch", "actual_p_vs_explicit_softmax",
        ]
    else:
        slots = [
            "actual_y_finite", "wide_f32_vs_actual_y", "wide_f32_vs_opaque",
        ]

    probe = Probe(torch, lib, slots)
    probe.retain(q, k, v)
    try:
        with torch.inference_mode():
            probe.fill(q, 1)
            probe.fill(k, 2)
            probe.fill(v, 3)
            qh, kh, vh = (x.transpose(1, 2).to(dtype=torch.float16) for x in (q, k, v))
            probe.retain(qh, kh, vh)
            actual_y = actual_p = None
            if args.phase in ("endpoints", "softmax", "wide"):
                actual_y, actual_p = torch.ops.aten._scaled_dot_product_attention_math_for_mps.default(
                    qh, kh, vh, None, 0.0, False, None, scale=None
                )
                if args.phase == "wide":
                    probe.retain(actual_y)
                    del actual_p
                    actual_p = None
                else:
                    probe.retain(actual_y, actual_p)
                if args.phase in ("endpoints", "wide"):
                    probe.compare("actual_y_finite", actual_y, actual_y, n=Q_N)
                if args.phase in ("endpoints", "softmax"):
                    probe.compare("actual_p_finite", actual_p, actual_p, n=S_N)

            if args.phase == "endpoints":
                # Official pinned Wan fallback is the source-level endpoint.
                wan_y = upstream.attention(
                    q, k, v, q_lens=None, k_lens=None, dropout_p=0.0,
                    softmax_scale=None, q_scale=None, causal=False,
                    window_size=(-1, -1), deterministic=False,
                    dtype=torch.float16, fa_version=None,
                )
                probe.retain(wan_y)
                probe.compare("wan_vs_actual_y", wan_y, actual_y, n=Q_N, layout=1)

                prefix6_p = torch.empty((1, H, Q, K), device="mps", dtype=torch.float16)
                prefix6_ctx = torch.empty((1, H, Q, D), device="mps", dtype=torch.float16)
                plan6 = probe.make_stage(6, [q, k, v], [prefix6_p, prefix6_ctx])
                probe.encode(plan6, 6)
                probe.compare("prefix6_p_vs_actual_p", prefix6_p, actual_p, n=S_N)
                probe.compare("prefix6_ctx_vs_actual_y", prefix6_ctx, actual_y, n=Q_N)

                prefix5_ctx = torch.empty((1, H, Q, D), device="mps", dtype=torch.float16)
                plan5 = probe.make_stage(5, [q, k, v], [prefix5_ctx])
                probe.encode(plan5, 5)
                probe.compare("prefix5_vs_actual_y", prefix5_ctx, actual_y, n=Q_N)

                pv_ctx = torch.empty((1, H, Q, D), device="mps", dtype=torch.float16)
                pv_plan = probe.make_stage(11, [actual_p, v], [pv_ctx])
                probe.encode(pv_plan, 11)
                probe.compare("isolated_actual_pv_vs_actual_y", pv_ctx, actual_y, n=Q_N)

                opaque_y = torch.empty((1, Q, H, D), device="mps", dtype=torch.float16)
                opaque_spare = torch.empty_like(opaque_y)
                private_inputs = (ctypes.c_void_p * 3)(
                    *(probe.pointer(x) for x in (q, k, v))
                )
                private_outputs = (ctypes.c_void_p * 2)(
                    probe.pointer(opaque_y), probe.pointer(opaque_spare)
                )
                legacy_private_result = lib.wan_stage_legacy_private_rejected(
                    private_inputs, private_outputs
                )
                if legacy_private_result != 0:
                    raise RuntimeError(
                        "existing public typed constructor accepted private buffers "
                        f"or guard failed ({legacy_private_result})"
                    )
                opaque_plan = probe.make_stage(0, [q, k, v], [opaque_y, opaque_spare])
                probe.encode(opaque_plan, 0)
                probe.compare("opaque_vs_actual_y", opaque_y, actual_y, n=Q_N, layout=1)
                probe.compare("prefix5_vs_opaque", opaque_y, prefix5_ctx, n=Q_N, layout=1)

                probe.compare("control_plus_one", actual_y, actual_y, n=Q_N, control=1)
                probe.compare("control_nan", actual_y, actual_y, n=Q_N, control=2)

            elif args.phase == "qk-scale":
                score16 = torch.empty((1, H, Q, K), device="mps", dtype=torch.float16)
                scaled32 = torch.empty((1, H, Q, K), device="mps", dtype=torch.float32)
                native16 = torch.empty_like(score16)
                native32 = torch.empty_like(scaled32)
                probe.retain(score16, scaled32, native16, native32)

                qk_plan = probe.make_stage(1, [q, k, v], [native16])
                probe.encode(qk_plan, 1)
                torch.matmul(qh, kh.transpose(-2, -1), out=score16)
                probe.compare("prefix_qk_vs_torch_matmul", native16, score16, n=S_N)

                scale_plan = probe.make_stage(2, [q, k, v], [native32])
                probe.encode(scale_plan, 2)
                scale = torch.tensor(SCALE, device="mps", dtype=torch.float32)
                scaled32.copy_(score16)
                scaled32.mul_(scale)
                probe.compare("prefix_scale32_vs_torch", native32, scaled32,
                              n=S_N, actual_type=0, reference_type=0)

                isolated_scale = probe.make_stage(8, [score16], [native32])
                probe.encode(isolated_scale, 8)
                probe.compare("isolated_scale32_vs_torch", native32, scaled32,
                              n=S_N, actual_type=0, reference_type=0)

            elif args.phase == "round":
                score16 = torch.empty((1, H, Q, K), device="mps", dtype=torch.float16)
                scaled32 = torch.empty((1, H, Q, K), device="mps", dtype=torch.float32)
                rounded16 = torch.empty((1, H, Q, K), device="mps", dtype=torch.float16)
                native16 = torch.empty_like(score16)
                probe.retain(score16, scaled32, rounded16, native16)

                torch.matmul(qh, kh.transpose(-2, -1), out=score16)
                scale = torch.tensor(SCALE, device="mps", dtype=torch.float32)
                scaled32.copy_(score16)
                scaled32.mul_(scale)
                rounded16.copy_(scaled32)
                rounded_prefix = probe.make_stage(3, [q, k, v], [native16])
                probe.encode(rounded_prefix, 3)
                probe.compare("prefix_scaled16_vs_torch", native16, rounded16, n=S_N)

                isolated_round = probe.make_stage(9, [score16], [native16])
                probe.encode(isolated_round, 9)
                probe.compare("isolated_scaled16_vs_torch", native16, rounded16, n=S_N)

            elif args.phase == "softmax":
                score16 = torch.empty((1, H, Q, K), device="mps", dtype=torch.float16)
                scaled32 = torch.empty((1, H, Q, K), device="mps", dtype=torch.float32)
                prob16 = torch.empty((1, H, Q, K), device="mps", dtype=torch.float16)
                native16 = torch.empty_like(score16)
                probe.retain(score16, scaled32, prob16, native16)

                torch.matmul(qh, kh.transpose(-2, -1), out=score16)
                scale = torch.tensor(SCALE, device="mps", dtype=torch.float32)
                scaled32.copy_(score16)
                scaled32.mul_(scale)
                # Reuse score16 as rounded logits after the QK source is consumed.
                score16.copy_(scaled32)
                torch.ops.aten._softmax.out(score16, -1, False, out=prob16)
                prob_prefix = probe.make_stage(4, [q, k, v], [native16])
                probe.encode(prob_prefix, 4)
                probe.compare("prefix_softmax_vs_torch", native16, prob16, n=S_N)

                isolated_softmax = probe.make_stage(10, [score16], [native16])
                probe.encode(isolated_softmax, 10)
                probe.compare("isolated_softmax_vs_torch", native16, prob16, n=S_N)
                probe.compare("actual_p_vs_explicit_softmax", actual_p, prob16, n=S_N)

            else:  # wide
                wide_y = torch.empty((1, H, Q, D), device="mps", dtype=torch.float16)
                wide_plan = probe.make_stage(12, [q, k, v], [wide_y])
                probe.encode(wide_plan, 12)
                probe.compare("wide_f32_vs_actual_y", wide_y, actual_y, n=Q_N)
                opaque_y = torch.empty((1, Q, H, D), device="mps", dtype=torch.float16)
                opaque_spare = torch.empty_like(opaque_y)
                opaque_plan = probe.make_stage(0, [q, k, v], [opaque_y, opaque_spare])
                probe.encode(opaque_plan, 0)
                probe.compare("wide_f32_vs_opaque", opaque_y, wide_y, n=Q_N, layout=1)

        result = probe.read_metrics_once()
        controls_pass = None
        if args.phase == "endpoints":
            controls_pass = (
                result["control_plus_one"]["changed"] == Q_N
                and result["control_plus_one"]["violations"] == Q_N
                and result["control_plus_one"]["nonfinite"] == 0
                and result["control_nan"]["changed"] == Q_N
                and result["control_nan"]["violations"] == Q_N
                and result["control_nan"]["nonfinite"] == Q_N
            )
        if controls_pass is False:
            raise RuntimeError("native comparator positive/NaN controls failed")
        passed = all(
            metric["violations"] == 0 and metric["nonfinite"] == 0
            for name, metric in result.items()
            if not name.startswith("control_")
        )
        return {
            "status": "complete",
            "phase": args.phase,
            "candidate_comparisons_passed": passed,
            "controls_passed": controls_pass,
            "metrics": result,
            "tolerance": TOLERANCE,
            "scale_f32": struct.unpack("<f", struct.pack("<f", SCALE))[0],
            "candidate_stage_order": candidate_stage_order(args.phase),
            "geometry": {"B": 1, "Q": Q, "K": K, "H": H, "D": D, "dtype": "float16"},
            "fixture_formula": FIXTURE_FORMULA,
            "shared_qkv_storage_passed_to_all_native_stages": True,
            "torch_version": torch.__version__,
            "macos_version": platform.mac_ver()[0],
            "mps_fallback_disabled": True,
            "legacy_private_bindings_rejected": True if args.phase == "endpoints" else None,
            "wan_source": str(args.wan_source.resolve()),
            "wan_source_sha256": source_sha,
            "bridge_library": str(args.library.resolve()),
            "bridge_library_sha256": sha256_file(args.library.resolve()),
            "stage_bridge_source_sha256": sha256_file(bridge_source),
            "production_bridge_source_sha256": sha256_file(production_source),
            "driver_sha256": sha256_file(Path(__file__).resolve()),
            "explicit_user_tensor_estimate_bytes": phase_estimates()[args.phase],
            "explicit_cap_bytes": EXPLICIT_CAP_BYTES,
            "mpsgraph_workspace_included": False,
            "external_memory_guard_required": {
                "process_footprint_limit_mib": 5120,
                "runtime_free_floor_percent": 15,
            },
            "activation_readbacks": 0,
            "metric_readbacks": 1,
            "actual_p_and_y_from_internal_sdpa_op": args.phase in (
                "endpoints", "softmax", "wide"
            ),
            "candidate_prefix_outputs_are_not_internal_taps": True,
            "opaque_sdpa_internals_exposed": False,
            "elapsed_seconds": round(time.monotonic() - args.started_at, 3),
        }
    finally:
        probe.release()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--phase",
        choices=("endpoints", "qk-scale", "round", "softmax", "wide"),
        default="endpoints",
    )
    parser.add_argument("--plan-only", action="store_true")
    parser.add_argument("--library", type=Path)
    parser.add_argument("--wan-source", type=Path, default=DEFAULT_WAN_SOURCE)
    parser.add_argument("--output", type=Path, help="optional JSON result path under /opt/mldata_ssd")
    args = parser.parse_args()
    args.started_at = time.monotonic()

    if args.plan_only:
        payload = plan_payload(args.phase)
        print(json.dumps(payload, sort_keys=True))
        return 0 if payload["explicit_estimate_under_cap"] else 2

    os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "0"
    try:
        payload = _run_phase(args)
        if args.output:
            output = args.output.resolve()
            output.parent.mkdir(parents=True, exist_ok=True)
            tmp = output.with_name(output.name + ".tmp")
            tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
            tmp.replace(output)
        print(json.dumps(payload, sort_keys=True))
        # Candidate mismatches are the diagnostic result; process success is
        # reserved for bridge/measurement failures or failed comparator controls.
        return 0
    except (FileNotFoundError, FileExistsError, OSError, RuntimeError, ValueError) as error:
        print(json.dumps({"status": "error", "error": str(error)}), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
