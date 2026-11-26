#!/usr/bin/env python3
"""CPU-only work model for the pinned Hy-Nomas preview geometries.

No model, tensor library, weights, GPU initialization, or timing is involved.
Counts are not latency predictions, utilization, or physical DRAM traffic.
Refresh the assumptions after model, layout, window, or token-policy changes.
"""

from collections import Counter
import json


def attention_tiles(temporal, height, width, tile=8):
    """Count the ragged, clamped neighborhoods in AttentionLayout."""
    rows, cols = (height + tile - 1) // tile, (width + tile - 1) // tile
    tokens = temporal * height * width
    if rows <= 3 and cols <= 3:
        return [(tokens, tokens)]
    hs = [min(tile, height - i * tile) for i in range(rows)]
    ws = [min(tile, width - j * tile) for j in range(cols)]
    result = []
    for i in range(rows):
        for j in range(cols):
            ci, cj = max(1, min(i, rows - 2)), max(1, min(j, cols - 2))
            kh = sum(hs[max(0, ci - 1):min(rows, ci + 2)])
            kw = sum(ws[max(0, cj - 1):min(cols, cj + 2)])
            result.append((temporal * hs[i] * ws[j], temporal * kh * kw))
    assert sum(q for q, _ in result) == tokens
    return result


def decoder_work(temporal, height, width, windows):
    d, f, kv, heads, dh, blocks = 4096, 12288, 1024, 32, 128, 36
    j = temporal * height * width
    tiles = attention_tiles(temporal, height, width)
    qk_pairs = sum(q * k for q, k in tiles)
    scores = heads * qk_pairs
    ffn = 6 * j * d * f
    projection = 4 * j * d * (d + kv)
    attention = 4 * dh * scores
    head = 2 * j * d * 3072
    assert height % 2 == width % 2 == 0
    stem = 2 * j * 3072 * d + 2 * (j // 4) * 32 * 16384
    passes = windows * blocks
    return {
        "shape_thw": [temporal, height, width],
        "tokens_per_window": j,
        "windows": windows,
        "aggregate_ffn_rows": windows * j,
        "tiles_per_window": len(tiles),
        "tile_histogram": [
            {"q": q, "k": k, "count": n}
            for (q, k), n in sorted(Counter(tiles).items())
        ],
        "qk_pairs_per_head_per_window": qk_pairs,
        "ffn_flops": passes * ffn,
        "projection_flops": passes * projection,
        "attention_flops": passes * attention,
        "blocks_flops": passes * (ffn + projection + attention),
        "head_flops": windows * head,
        "stem_flops": windows * stem,
        "blocks_head_flops": passes * (ffn + projection + attention) + windows * head,
        "stem_head_flops": windows * (stem + head),
        "all_counted_flops": passes * (ffn + projection + attention)
        + windows * (stem + head),
        "logical_score_probability_write_bytes": passes * 2 * 4 * scores,
        "logical_expanded_kv_duplicate_write_bytes":
        passes * 2 * (heads - 8) * sum(k for _, k in tiles) * dh * 4,
    }


def low_resolution_dit_work(steps):
    """First segment: stored text=64, valid text=23, refiner rows=1280."""
    d, f, video, stored, valid = 2048, 8192, 4320, 64, 23
    s, a = video + stored, video + valid
    block = 24 * d * d + 8 * d * d * s + 4 * d * f * s + 4 * d * a * a
    refiner_block = (
        4 * d * d + 8 * 1280 * d * d + 4 * 1280 * d * f
        + 4 * d * valid * valid
    )
    refiner_embed = 2 * 1280 * 3584 * d
    time_mlp = 2 * 256 * d + 2 * d * d
    context_mlp = 2 * 3584 * d + 2 * d * d
    refiner = 2 * refiner_block + refiner_embed + 2 * time_mlp + context_mlp
    adapters = 2 * video * 32 * d + 2 * video * d * 32 + 4 * d * d
    return steps * (54 * block + refiner + adapters)


def dcae_low_resolution_work():
    """Pinned DCAE-V dense-conv work for 120x288x512; not a time bound.

    Three causal 10-latent tiles, no temporal overlap, 32 channels, s32/t4.
    Count padded 3x3x3 convolution products; omit elementwise/cache/shuffle work.
    """
    t, h, w, channels = 30, 9, 16, 1024
    parts = {"project_in": 2 * 27 * t * h * w * 32 * channels}
    # Decoder runs stage6 first; each later row upsamples before its ResBlocks.
    for stage, output_channels, depth, sf, tf in (
        (6, 1024, 4, 1, 1),
        (5, 1024, 4, 1, 4),
        (4, 1024, 4, 2, 1),
        (3, 512, 4, 2, 1),
        (2, 512, 10, 2, 1),
        (1, 256, 5, 2, 1),
    ):
        if stage != 6:
            # PixelShuffle's convolution expands channels at the input grid.
            parts[f"stage{stage}_upsample"] = (
                2 * 27 * t * h * w * channels * output_channels * sf * sf * tf
            )
            t, h, w, channels = t * tf, h * sf, w * sf, output_channels
        parts[f"stage{stage}_residuals"] = (
            depth * 2 * (2 * 27 * t * h * w * channels * channels)
        )
    # Stage0 has depth0. Project-out is a spatial x2 PixelShuffle to RGB.
    parts["project_out"] = 2 * 27 * t * h * w * channels * (3 * 4)
    assert (t, h * 2, w * 2) == (120, 288, 512)
    return {
        "latent_shape_cthw": [32, 30, 9, 16],
        "output_shape_cthw": [3, 120, 288, 512],
        "latent_tile_frames": 10,
        "tile_count": 3,
        "components_flops": parts,
        "residual_convolution_flops": sum(
            v for k, v in parts.items() if k.endswith("_residuals")
        ),
        "all_counted_flops": sum(parts.values()),
        "not_a_latency_or_physical_instruction_bound": True,
    }


def main():
    low = decoder_work(10, 18, 32, 5)
    diagnostic = decoder_work(6, 44, 80, 1)
    saved = decoder_work(10, 26, 48, 5)
    dcae = dcae_low_resolution_work()
    # Cross-check the prior published model and the reviewed ragged tile counts.
    assert low["blocks_flops"] == 473459746406400
    assert low["stem_head_flops"] == 1457101209600
    assert diagnostic["qk_pairs_per_head_per_window"] == 69672960
    assert saved["qk_pairs_per_head_per_window"] == 64972800
    assert low["all_counted_flops"] + low_resolution_dit_work(8) == 734577437442048
    assert dcae["residual_convolution_flops"] == 542059769364480
    assert dcae["all_counted_flops"] == 602486530375680
    presets = []
    for steps in (4, 8):
        dit = low_resolution_dit_work(steps)
        total = dit + low["all_counted_flops"]
        presets.append({
            "dit_steps": steps,
            "dit_flops_including_refiner_adapters": dit,
            "pixelumm_flops": low["all_counted_flops"],
            "total_flops": total,
            "pixelumm_fraction_of_counted_work": low["all_counted_flops"] / total,
        })
    print(json.dumps({
        "schema": "cogni_ml.hy_nomas_work_budget/v1",
        "scope": "major matrix/dense-convolution operations; two FLOPs/MAC; no GPU or timing",
        "excluded": ["elementwise", "reductions", "casts", "live Qwen",
                     "resource preparation", "export", "later-segment history"],
        "physical_dram_traffic_measured": False,
        "source_anchors": [
            "src/ml/video/pixelumm_decoder_block_mps.cr:AttentionLayout",
            "examples/support/hy_nomas_generation.cr:Decoder",
            "docs/hy-nomas-generation-frontier.md:Operation-count roofline",
            "https://github.com/Tencent-Hunyuan/Hy-Nomas/blob/"
            "effb8a49b83e6f1d35e26710db35194f2fd952a7/"
            "hymm/models/vae/autoencoder_dcae_v.py#L1509",
        ],
        "decoder_cases": {"low_resolution": low, "diagnostic": diagnostic,
                          "saved_high_resolution": saved},
        "low_resolution_presets": presets,
        "dcae_low_resolution": dcae,
        "self_checks": "passed",
    }, indent=2))


if __name__ == "__main__":
    main()
