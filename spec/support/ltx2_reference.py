#!/usr/bin/env python3
# Adapted from Lightricks/LTX-2 source at e98f6d927d83c92ef2be6aec76c8440b19306e39.
# Changed for a native Metal/Crystal parity fixture; covered by the LTX-2.x
# Community License (LICENSE-2_x). This notice does not grant relicensing rights.
"""Create an MPS fixture by executing the pinned upstream LTX-2 block.

No CPU execution fallback is admitted. The tiny shape keeps the oracle probe
small while exercising the actual upstream dual-stream module topology.
"""

from __future__ import annotations

import argparse
from dataclasses import replace
import hashlib
import json
import os
from pathlib import Path
import random
import sys
from types import ModuleType
from typing import Any

PINNED_REVISION = "e98f6d927d83c92ef2be6aec76c8440b19306e39"
DEFAULT_SOURCE_ROOT = Path(
    "/opt/mldata_ssd/cogni-ml/video-scene-20261019/ltx-audit/source/packages/ltx-core/src"
)
DEFAULT_OUTPUT = Path(
    "/opt/mldata_ssd/cogni-ml/video-scene-20261019/ltx-audit/ltx2-dual-stream-mps.json"
)
SEED = 20261019
PERTURBATION_SEED = 20261020
PINNED_SOURCE_SHA256 = {
    "ltx_core/model/transformer/transformer.py": "6d24b7294be9afb28d69156b9a17d59477d01b43547156cf79767cee1f9f24d5",
    "ltx_core/model/transformer/attention.py": "e65abeb1536d0ae8b0f1236a913f2b5675fab61ba09d90fe33ccc03f496cabbd",
    "ltx_core/model/transformer/feed_forward.py": "e6c743950e90b0540499674c688271c5aab45101e6f776fbdde78f50bff12567",
    "ltx_core/model/transformer/gelu_approx.py": "c9d008a2db4f945ee39d24ff6c191fc6c15093ad021d86648b6a70dc16d6da55",
    "ltx_core/model/transformer/rope.py": "6be5f15cc736544a49aa775fdf0befea35be3f10a9a2977160c0eaeb5bbd9fe4",
    "ltx_core/model/transformer/transformer_args.py": "dbb97d200f6de3f82155638978294d30227adaa9131736b6970ff270509b72da",
    "ltx_core/model/transformer/adaln.py": "87be8b1ec9b1395156d2e0f22834ed993af31d684198109d33e61c68f53cd065",
    "ltx_core/model/transformer/timestep_embedding.py": "3a75ff5ef6231bed408f10279a4fb3f0b76c7b027ebaf0aae8c0255fb5338b97",
    "ltx_core/model/transformer/modality.py": "b37d99f659b4e4888ba9cdaed562890648105e6b158245e97ba3ee9123be22db",
    "ltx_core/utils.py": "d4acb090aa380165e47739ec5c9dc89d58e9ab2b20e7db06c45c1c7554ab4810",
}

# Must be set before importing torch. A nonzero caller setting is rejected so
# unsupported MPS operators cannot silently execute on CPU.
fallback_setting = os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK")
if fallback_setting not in (None, "0"):
    raise SystemExit("PYTORCH_ENABLE_MPS_FALLBACK must be 0")
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "0"


def namespace(name: str, directory: Path) -> ModuleType:
    module = ModuleType(name)
    module.__path__ = [str(directory)]  # type: ignore[attr-defined]
    sys.modules[name] = module
    return module


def import_upstream(source_root: Path) -> tuple[Any, ...]:
    # Upstream package __init__ files import the whole model stack. Namespace
    # packages let this fixture execute the exact pinned block and its direct
    # dependencies without importing unrelated inference/pipeline modules.
    namespace("ltx_core", source_root / "ltx_core")
    namespace("ltx_core.model", source_root / "ltx_core" / "model")
    namespace("ltx_core.model.transformer", source_root / "ltx_core" / "model" / "transformer")

    guidance = namespace("ltx_core.guidance", source_root / "ltx_core" / "guidance")
    perturbations = ModuleType("ltx_core.guidance.perturbations")
    perturbations.BatchedPerturbationConfig = type("BatchedPerturbationConfig", (), {})
    perturbations.PerturbationType = type("PerturbationType", (), {})
    sys.modules[perturbations.__name__] = perturbations
    guidance.perturbations = perturbations  # type: ignore[attr-defined]

    import torch
    from torch.nn.attention import SDPBackend

    from ltx_core.model.transformer.attention import AttentionOps, PytorchAttention
    from ltx_core.model.transformer.rope import LTXRopeType, precompute_freqs_cis
    from ltx_core.model.transformer.transformer import BasicAVTransformerBlock, TransformerConfig
    from ltx_core.model.transformer.transformer_args import AdaLNTimestepEmbedding, TransformerArgs

    return (
        torch,
        SDPBackend,
        AttentionOps,
        PytorchAttention,
        LTXRopeType,
        precompute_freqs_cis,
        BasicAVTransformerBlock,
        TransformerConfig,
        AdaLNTimestepEmbedding,
        TransformerArgs,
    )


def deterministic_values(count: int, seed: int, *, near_one: bool = False) -> list[float]:
    rng = random.Random(seed)
    if near_one:
        return [1.0 + rng.uniform(-0.015, 0.015) for _ in range(count)]
    return [rng.uniform(-0.035, 0.035) for _ in range(count)]


def tensor_values(tensor: Any) -> list[float]:
    import torch

    return tensor.detach().to(device="cpu", dtype=torch.float32).contiguous().view(-1).tolist()


def serialize_values(value: Any) -> list[float]:
    if hasattr(value, "detach"):
        return tensor_values(value)
    return [float(item) for item in value]


def pe_values(pe: tuple[Any, Any]) -> tuple[list[float], list[float]]:
    # Upstream uses [B,H,T,D/2]; the native diagnostic buffers are [T,H,D/2].
    return tuple(
        tensor.permute(0, 2, 1, 3).contiguous().view(-1).tolist()
        for tensor in pe
    )  # type: ignore[return-value]


def make_positions(torch: Any, tokens: int, axes: int) -> Any:
    positions = torch.zeros((1, axes, tokens, 2), dtype=torch.float32)
    for token in range(tokens):
        for axis in range(axes):
            start = float(token * (axis + 1))
            positions[0, axis, token, 0] = start
            positions[0, axis, token, 1] = start + 1.0
    return positions


def build_pe(
    torch: Any,
    precompute_freqs_cis: Any,
    positions: Any,
    dim: int,
    heads: int,
    rope_type: Any,
) -> tuple[Any, Any]:
    return precompute_freqs_cis(
        positions,
        dim=dim,
        out_dtype=torch.float32,
        max_pos=[20, 2048, 2048] if positions.shape[1] == 3 else [4096],
        use_middle_indices_grid=True,
        num_attention_heads=heads,
        rope_type=rope_type.SPLIT,
    )


def repeat_rows(tensor: Any, rows: int) -> Any:
    return tensor.expand(tensor.shape[0], rows, tensor.shape[-1]).contiguous()


def build_case_inputs(
    torch: Any,
    block: Any,
    AdaLNTimestepEmbedding: Any,
    video_x: Any,
    audio_x: Any,
    video_context: Any,
    audio_context: Any,
    video_pe: tuple[Any, Any],
    audio_pe: tuple[Any, Any],
    video_cross_pe: tuple[Any, Any],
    audio_cross_pe: tuple[Any, Any],
    video_tokens: int,
    audio_tokens: int,
    video_dim: int,
    audio_dim: int,
    video_context_dim: int,
    audio_context_dim: int,
    device: Any,
) -> tuple[dict[str, Any], Any, Any, Any, Any, Any, Any]:
    # Use a nontrivial per-token perturbation mask. The native adapter receives
    # the exact product of this mask and the gate to match upstream's two
    # elementwise products while keeping the mask outside the module contract.
    video_mask = torch.tensor([0.25, 0.625, 1.0], device=device, dtype=torch.float32).view(1, video_tokens, 1)
    audio_mask = torch.tensor([1.0, 0.5, 0.0, 0.75], device=device, dtype=torch.float32).view(1, audio_tokens, 1)

    video_mod = torch.linspace(-0.012, 0.013, 6 * video_dim, device=device).view(1, 1, 6 * video_dim)
    audio_mod = torch.linspace(0.011, -0.014, 6 * audio_dim, device=device).view(1, 1, 6 * audio_dim)
    video_timestep = AdaLNTimestepEmbedding(video_mod, torch.zeros((1, 1, video_dim), device=device))
    audio_timestep = AdaLNTimestepEmbedding(audio_mod, torch.zeros((1, 1, audio_dim), device=device))

    def base_values(table_name: str, timestep: Any, indices: slice, rows: int) -> tuple[Any, Any, Any]:
        table = getattr(block, table_name)
        return tuple(repeat_rows(value, rows)
                     for value in block.get_ada_values(table, 1, timestep, indices))  # type: ignore[return-value]

    video_sa_shift, video_sa_scale, video_sa_gate = base_values("scale_shift_table", video_timestep, slice(0, 3), video_tokens)
    video_ff_shift, video_ff_scale, video_ff_gate = base_values("scale_shift_table", video_timestep, slice(3, 6), video_tokens)
    audio_sa_shift, audio_sa_scale, audio_sa_gate = base_values("audio_scale_shift_table", audio_timestep, slice(0, 3), audio_tokens)
    audio_ff_shift, audio_ff_scale, audio_ff_gate = base_values("audio_scale_shift_table", audio_timestep, slice(3, 6), audio_tokens)

    # AV scale/shift values are two table rows per direction, shared over all
    # tokens. Their native counterpart is expanded into each token's input row.
    video_cross_scale_shift = torch.linspace(-0.009, 0.008, 4 * video_dim, device=device).view(1, 1, 4 * video_dim)
    audio_cross_scale_shift = torch.linspace(0.007, -0.01, 4 * audio_dim, device=device).view(1, 1, 4 * audio_dim)
    video_cross_gate = torch.linspace(-0.006, 0.009, video_dim, device=device).view(1, 1, video_dim)
    audio_cross_gate = torch.linspace(0.008, -0.007, audio_dim, device=device).view(1, 1, audio_dim)
    video_cross_shift_timestep = video_cross_scale_shift
    audio_cross_shift_timestep = audio_cross_scale_shift
    video_cross_gate_timestep = video_cross_gate
    audio_cross_gate_timestep = audio_cross_gate

    # Upstream returns (scale, shift, gate); the Metal module accepts shift and
    # scale separately, with the AV gate pre-multiplied by the perturbation mask.
    video_a2v_scale, video_a2v_shift, video_a2v_gate = block.get_av_ca_ada_values(
        block.scale_shift_table_a2v_ca_video,
        1,
        video_timestep,
        video_cross_shift_timestep,
        video_cross_gate_timestep,
        slice(0, 2),
    )
    audio_a2v_scale, audio_a2v_shift, _ = block.get_av_ca_ada_values(
        block.scale_shift_table_a2v_ca_audio,
        1,
        audio_timestep,
        audio_cross_shift_timestep,
        audio_cross_gate_timestep,
        slice(0, 2),
    )
    audio_v2a_scale, audio_v2a_shift, audio_v2a_gate = block.get_av_ca_ada_values(
        block.scale_shift_table_a2v_ca_audio,
        1,
        audio_timestep,
        audio_cross_shift_timestep,
        audio_cross_gate_timestep,
        slice(2, 4),
    )
    video_v2a_scale, video_v2a_shift, _ = block.get_av_ca_ada_values(
        block.scale_shift_table_a2v_ca_video,
        1,
        video_timestep,
        video_cross_shift_timestep,
        video_cross_gate_timestep,
        slice(2, 4),
    )
    video_a2v_gate_rows = repeat_rows(video_a2v_gate, video_tokens) * video_mask
    audio_v2a_gate_rows = repeat_rows(audio_v2a_gate, audio_tokens) * audio_mask

    inputs = {
        "video_x": video_x,
        "audio_x": audio_x,
        "video_context": video_context,
        "audio_context": audio_context,
        "video_sa_shift": video_sa_shift,
        "video_sa_scale": video_sa_scale,
        "video_sa_gate": video_sa_gate,
        "video_ff_shift": video_ff_shift,
        "video_ff_scale": video_ff_scale,
        "video_ff_gate": video_ff_gate,
        "audio_sa_shift": audio_sa_shift,
        "audio_sa_scale": audio_sa_scale,
        "audio_sa_gate": audio_sa_gate,
        "audio_ff_shift": audio_ff_shift,
        "audio_ff_scale": audio_ff_scale,
        "audio_ff_gate": audio_ff_gate,
        "a2v_video_shift": repeat_rows(video_a2v_shift, video_tokens),
        "a2v_video_scale": repeat_rows(video_a2v_scale, video_tokens),
        "a2v_video_gate": video_a2v_gate_rows,
        "a2v_audio_shift": repeat_rows(audio_a2v_shift, audio_tokens),
        "a2v_audio_scale": repeat_rows(audio_a2v_scale, audio_tokens),
        "v2a_audio_shift": repeat_rows(audio_v2a_shift, audio_tokens),
        "v2a_audio_scale": repeat_rows(audio_v2a_scale, audio_tokens),
        "v2a_audio_gate": audio_v2a_gate_rows,
        "v2a_video_shift": repeat_rows(video_v2a_shift, video_tokens),
        "v2a_video_scale": repeat_rows(video_v2a_scale, video_tokens),
    }
    for name, pe in (
        ("video_self", video_pe),
        ("audio_self", audio_pe),
        ("video_cross", video_cross_pe),
        ("audio_cross", audio_cross_pe),
    ):
        cosine, sine = pe_values(pe)
        inputs[f"{name}_cos"] = cosine
        inputs[f"{name}_sin"] = sine

    video_args = {
        "x": video_x,
        "context": video_context,
        "context_mask": None,
        "timestep_embedding": video_timestep,
        "positional_embeddings": video_pe,
        "cross_positional_embeddings": video_cross_pe,
        "cross_scale_shift_timestep": video_cross_shift_timestep,
        "cross_gate_timestep": video_cross_gate_timestep,
        "enabled": True,
        "cross_attn_perturbation_mask": video_mask,
    }
    audio_args = {
        "x": audio_x,
        "context": audio_context,
        "context_mask": None,
        "timestep_embedding": audio_timestep,
        "positional_embeddings": audio_pe,
        "cross_positional_embeddings": audio_cross_pe,
        "cross_scale_shift_timestep": audio_cross_shift_timestep,
        "cross_gate_timestep": audio_cross_gate_timestep,
        "enabled": True,
        "cross_attn_perturbation_mask": audio_mask,
    }
    return inputs, video_args, audio_args, video_a2v_gate, audio_v2a_gate, video_mask, audio_mask


def file_manifest(source_root: Path) -> dict[str, str]:
    missing = [relative for relative in PINNED_SOURCE_SHA256 if not (source_root / relative).is_file()]
    if missing:
        raise SystemExit(f"missing pinned upstream source files: {', '.join(missing)}")
    return {
        relative: hashlib.sha256((source_root / relative).read_bytes()).hexdigest()
        for relative in PINNED_SOURCE_SHA256
    }


def verify_source_manifest(source_root: Path) -> dict[str, str]:
    actual = file_manifest(source_root)
    mismatches = {
        relative: {"expected": expected, "actual": actual.get(relative)}
        for relative, expected in PINNED_SOURCE_SHA256.items()
        if actual.get(relative) != expected
    }
    if mismatches:
        raise SystemExit(
            "upstream source hash mismatch for pinned revision "
            f"{PINNED_REVISION}: {json.dumps(mismatches, sort_keys=True)}"
        )
    return actual


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--source-root",
        type=Path,
        default=DEFAULT_SOURCE_ROOT,
        help="Pinned upstream `ltx_core/src` snapshot",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="External fixture output path",
    )
    parser.add_argument(
        "--device",
        choices=("mps",),
        default="mps",
        help="Oracle device (CPU is intentionally not an option)",
    )
    args = parser.parse_args()

    source_root = args.source_root.resolve()
    if not (source_root / "ltx_core/model/transformer/transformer.py").is_file():
        raise SystemExit(f"missing pinned upstream source under {source_root}")
    # Refuse to import or execute a source-root that merely claims the pinned
    # revision but contains different bytes.
    source_manifest = verify_source_manifest(source_root)

    (
        torch,
        SDPBackend,
        AttentionOps,
        PytorchAttention,
        LTXRopeType,
        precompute_freqs_cis,
        BasicAVTransformerBlock,
        TransformerConfig,
        AdaLNTimestepEmbedding,
        TransformerArgs,
    ) = import_upstream(source_root)
    if not torch.backends.mps.is_available():
        raise SystemExit("MPS unavailable: refusing CPU oracle")
    device = torch.device("mps")

    config = {
        "video_dim": 32,
        "video_context_dim": 24,
        "video_context_tokens": 2,
        "video_tokens": 3,
        "video_heads": 2,
        "video_head_dim": 16,
        "audio_dim": 16,
        "audio_context_dim": 12,
        "audio_context_tokens": 2,
        "audio_tokens": 4,
        "audio_heads": 2,
        "audio_head_dim": 8,
        "video_modulation_rows": 3,
        "audio_modulation_rows": 4,
        "norm_eps": 1e-6,
    }
    video = TransformerConfig(
        dim=config["video_dim"],
        heads=config["video_heads"],
        d_head=config["video_head_dim"],
        context_dim=config["video_context_dim"],
        apply_gated_attention=False,
        cross_attention_adaln=False,
        ff_bias=True,
    )
    audio = TransformerConfig(
        dim=config["audio_dim"],
        heads=config["audio_heads"],
        d_head=config["audio_head_dim"],
        context_dim=config["audio_context_dim"],
        apply_gated_attention=False,
        cross_attention_adaln=False,
        ff_bias=True,
    )
    sdpa = PytorchAttention(priority=[SDPBackend.MATH])
    block = BasicAVTransformerBlock(
        video=video,
        audio=audio,
        rope_type=LTXRopeType.SPLIT,
        norm_eps=config["norm_eps"],
        attention_ops=AttentionOps(attention_function=sdpa, masked_attention_function=sdpa),
    )
    with torch.no_grad():
        for name, parameter in block.named_parameters():
            shape = tuple(parameter.shape)
            count = parameter.numel()
            vals = deterministic_values(
                count,
                SEED + sum((i + 1) * ord(ch) for i, ch in enumerate(name)),
                near_one=name.endswith("q_norm.weight") or name.endswith("k_norm.weight"),
            )
            # Strong gates make an omitted residual branch fail the fixed parity gate.
            if name in ("scale_shift_table", "audio_scale_shift_table"):
                for row in (2, 5):
                    vals[row * shape[1] : (row + 1) * shape[1]] = [
                        0.65 + 0.02 * ((column % 3) - 1) for column in range(shape[1])
                    ]
            elif name in ("scale_shift_table_a2v_ca_video", "scale_shift_table_a2v_ca_audio"):
                vals[4 * shape[1] : 5 * shape[1]] = [
                    0.75 + 0.02 * ((column % 3) - 1) for column in range(shape[1])
                ]
            parameter.copy_(torch.tensor(vals, dtype=torch.float32).reshape(shape))
    block = block.to(device=device, dtype=torch.float32).eval()

    video_pos = make_positions(torch, config["video_tokens"], 3)
    audio_pos = make_positions(torch, config["audio_tokens"], 1)
    video_pe = build_pe(torch, precompute_freqs_cis, video_pos, 32, config["video_heads"], LTXRopeType)
    audio_pe = build_pe(torch, precompute_freqs_cis, audio_pos, 16, config["audio_heads"], LTXRopeType)
    video_cross_pe = build_pe(torch, precompute_freqs_cis, video_pos, 16, config["audio_heads"], LTXRopeType)
    audio_cross_pe = build_pe(torch, precompute_freqs_cis, audio_pos, 16, config["audio_heads"], LTXRopeType)
    video_pe = tuple(value.to(device=device) for value in video_pe)
    audio_pe = tuple(value.to(device=device) for value in audio_pe)
    video_cross_pe = tuple(value.to(device=device) for value in video_cross_pe)
    audio_cross_pe = tuple(value.to(device=device) for value in audio_cross_pe)

    hook_outputs: dict[str, Any] = {}
    hooks = []
    module_names = {
        "video.attn1.output": block.attn1,
        "video.attn2.output": block.attn2,
        "video.audio_to_video_attn.output": block.audio_to_video_attn,
        "video.ff.output": block.ff,
        "audio.audio_attn1.output": block.audio_attn1,
        "audio.audio_attn2.output": block.audio_attn2,
        "audio.video_to_audio_attn.output": block.video_to_audio_attn,
        "audio.audio_ff.output": block.audio_ff,
    }

    def save_output(name: str) -> Any:
        def hook(_module: Any, _inputs: Any, output: Any) -> None:
            hook_outputs[name] = output.detach()

        return hook

    for name, module in module_names.items():
        hooks.append(module.register_forward_hook(save_output(name)))

    # A source-hashed deterministic parameter set is serialized alongside each
    # case so native code consumes the exact values executed by the upstream block.
    weights = {name: tensor_values(value) for name, value in block.state_dict().items()}

    cases = []
    final_outputs = []
    try:
        for case_name in ("baseline", "seeded_perturbation"):
            case_seed = SEED if case_name == "baseline" else PERTURBATION_SEED
            # Both cases share all base inputs; the second changes video_x only.
            video_rng = random.Random(SEED + 1)
            audio_rng = random.Random(SEED + 2)
            video_x = torch.tensor(
                [video_rng.uniform(-0.4, 0.4) for _ in range(3 * config["video_dim"])],
                device=device,
                dtype=torch.float32,
            ).view(1, config["video_tokens"], config["video_dim"])
            audio_x = torch.tensor(
                [audio_rng.uniform(-0.4, 0.4) for _ in range(4 * config["audio_dim"])],
                device=device,
                dtype=torch.float32,
            ).view(1, config["audio_tokens"], config["audio_dim"])
            if case_name == "seeded_perturbation":
                perturb_rng = random.Random(PERTURBATION_SEED)
                perturb = torch.tensor(
                    [perturb_rng.uniform(-0.07, 0.08) for _ in range(video_x.numel())],
                    device=device,
                    dtype=torch.float32,
                ).view_as(video_x)
                video_x = video_x + perturb
            video_context = torch.tensor(
                deterministic_values(2 * config["video_context_dim"], SEED + 31),
                device=device,
                dtype=torch.float32,
            ).view(1, 2, config["video_context_dim"])
            audio_context = torch.tensor(
                deterministic_values(2 * config["audio_context_dim"], SEED + 47),
                device=device,
                dtype=torch.float32,
            ).view(1, 2, config["audio_context_dim"])

            input_tensors, video_arg_data, audio_arg_data, video_gate, audio_gate, video_mask, audio_mask = build_case_inputs(
                torch,
                block,
                AdaLNTimestepEmbedding,
                video_x,
                audio_x,
                video_context,
                audio_context,
                video_pe,
                audio_pe,
                video_cross_pe,
                audio_cross_pe,
                config["video_tokens"],
                config["audio_tokens"],
                config["video_dim"],
                config["audio_dim"],
                config["video_context_dim"],
                config["audio_context_dim"],
                device,
            )
            video_args = TransformerArgs(**video_arg_data)
            audio_args = TransformerArgs(**audio_arg_data)
            hook_outputs.clear()
            with torch.inference_mode():
                actual_video, actual_audio = block(video_args, audio_args)

                # Mirror only the upstream residual boundaries using module
                # outputs captured from the actual upstream forward call.
                v_after_sa = video_x + hook_outputs["video.attn1.output"] * input_tensors["video_sa_gate"].view_as(video_x)
                a_after_sa = audio_x + hook_outputs["audio.audio_attn1.output"] * input_tensors["audio_sa_gate"].view_as(audio_x)
                v_after_text = v_after_sa + hook_outputs["video.attn2.output"]
                a_after_text = a_after_sa + hook_outputs["audio.audio_attn2.output"]
                v_after_av = v_after_text + hook_outputs["video.audio_to_video_attn.output"] * video_gate * video_mask
                a_after_av = a_after_text + hook_outputs["audio.video_to_audio_attn.output"] * audio_gate * audio_mask
                v_out = v_after_av + hook_outputs["video.ff.output"] * input_tensors["video_ff_gate"].view_as(video_x)
                a_out = a_after_av + hook_outputs["audio.audio_ff.output"] * input_tensors["audio_ff_gate"].view_as(audio_x)

                for mirrored, actual, label in (
                    (v_out, actual_video.x, "video"),
                    (a_out, actual_audio.x, "audio"),
                ):
                    if not torch.allclose(mirrored, actual, atol=2e-6, rtol=2e-6):
                        max_error = (mirrored - actual).abs().max().item()
                        raise RuntimeError(f"upstream residual mirror mismatch in {label}: {max_error}")

                for name in (
                    "video.audio_to_video_attn.output",
                    "audio.video_to_audio_attn.output",
                ):
                    if hook_outputs[name].abs().max().item() < 1e-3:
                        raise RuntimeError(f"{name} is too small to discriminate a missing AV branch")
                if (v_after_av - v_after_text).abs().max().item() < 1e-4:
                    raise RuntimeError("video AV residual contribution is too small")
                if (a_after_av - a_after_text).abs().max().item() < 1e-4:
                    raise RuntimeError("audio AV residual contribution is too small")

            checkpoints = {
                **hook_outputs,
                "video.after_attn1": v_after_sa,
                "video.after_attn2": v_after_text,
                "video.after_av_cross_attention": v_after_av,
                "video.output": actual_video.x,
                "audio.after_audio_attn1": a_after_sa,
                "audio.after_audio_attn2": a_after_text,
                "audio.after_av_cross_attention": a_after_av,
                "audio.output": actual_audio.x,
            }

            # Execute the actual upstream block a second time with its first
            # result as input: the oracle for resident-buffer block chaining.
            hook_outputs.clear()
            chained_video_args = replace(video_args, x=actual_video.x)
            chained_audio_args = replace(audio_args, x=actual_audio.x)
            with torch.inference_mode():
                chained_video, chained_audio = block(chained_video_args, chained_audio_args)
                chained_v_after_sa = actual_video.x + hook_outputs["video.attn1.output"] * input_tensors["video_sa_gate"].view_as(video_x)
                chained_a_after_sa = actual_audio.x + hook_outputs["audio.audio_attn1.output"] * input_tensors["audio_sa_gate"].view_as(audio_x)
                chained_v_after_text = chained_v_after_sa + hook_outputs["video.attn2.output"]
                chained_a_after_text = chained_a_after_sa + hook_outputs["audio.audio_attn2.output"]
                chained_v_after_av = chained_v_after_text + hook_outputs["video.audio_to_video_attn.output"] * video_gate * video_mask
                chained_a_after_av = chained_a_after_text + hook_outputs["audio.video_to_audio_attn.output"] * audio_gate * audio_mask
                chained_v_out = chained_v_after_av + hook_outputs["video.ff.output"] * input_tensors["video_ff_gate"].view_as(video_x)
                chained_a_out = chained_a_after_av + hook_outputs["audio.audio_ff.output"] * input_tensors["audio_ff_gate"].view_as(audio_x)

                for mirrored, actual, label in (
                    (chained_v_out, chained_video.x, "chained video"),
                    (chained_a_out, chained_audio.x, "chained audio"),
                ):
                    if not torch.allclose(mirrored, actual, atol=2e-6, rtol=2e-6):
                        max_error = (mirrored - actual).abs().max().item()
                        raise RuntimeError(f"upstream residual mirror mismatch in {label}: {max_error}")

            chain_checkpoints = {
                **hook_outputs,
                "video.after_attn1": chained_v_after_sa,
                "video.after_attn2": chained_v_after_text,
                "video.after_av_cross_attention": chained_v_after_av,
                "video.output": chained_video.x,
                "audio.after_audio_attn1": chained_a_after_sa,
                "audio.after_audio_attn2": chained_a_after_text,
                "audio.after_av_cross_attention": chained_a_after_av,
                "audio.output": chained_audio.x,
            }
            cases.append(
                {
                    "name": case_name,
                    "seed": case_seed,
                    "inputs": {name: serialize_values(value) for name, value in input_tensors.items()},
                    "expected": {name: tensor_values(value) for name, value in checkpoints.items()},
                    "chain_expected": {name: tensor_values(value) for name, value in chain_checkpoints.items()},
                }
            )
            final_outputs.append((actual_video.x.detach(), actual_audio.x.detach()))
    finally:
        for hook in hooks:
            hook.remove()

    if torch.allclose(final_outputs[0][0], final_outputs[1][0], atol=1e-5, rtol=1e-5):
        raise RuntimeError("seeded perturbation did not change video output")
    if torch.allclose(final_outputs[0][1], final_outputs[1][1], atol=1e-5, rtol=1e-5):
        raise RuntimeError("seeded video perturbation did not reach audio output")

    output = {
        "upstream": {
            "repo": "https://github.com/Lightricks/LTX-2",
            "revision": PINNED_REVISION,
            "license": "LTX-2.x Community License (LICENSE-2_x)",
            "source_files_sha256": source_manifest,
            "class": "ltx_core.model.transformer.transformer.BasicAVTransformerBlock",
        },
        "oracle": {
            "device": "mps",
            "dtype": "float32",
            "attention": "upstream PytorchAttention pinned to SDPBackend.MATH",
            "pytorch_mps_fallback": False,
            "seed": SEED,
            "perturbation_seed": PERTURBATION_SEED,
            "fixture_shape_only": True,
            "production_dimensions_admitted": False,
        },
        "config": config,
        "weights": weights,
        "cases": cases,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, separators=(",", ":")) + "\n")
    print(f"wrote {args.output} ({len(cases)} MPS cases, revision {PINNED_REVISION})")


if __name__ == "__main__":
    main()
