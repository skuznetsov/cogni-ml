# Raspberry Pi 5 V3D Allowed-Head Route

Status: proposed product slice, backed by Pi probe evidence.

## Summary

Naive Raspberry Pi 5 GPU offload is refuted for Qwen3.5 0.8B/2B decode. Full FFN/head/attention dense paths are slower than the Pi CPU denominator. The viable route is narrower: exact constrained Q6 head evaluation for certified finite token sets from tool/literal constraints.

This route should be used only when the decoder can prove a finite allowed-token frontier. Otherwise fall back to the normal CPU/full-head path.

## Measured Anchors

Pi target:

- Device: V3D 7.1.7.0 / Mesa V3DV.
- Qwen3.5-2B CPU decode denominator: about `123.3ms/token` (`8.11 tok/s`) at best measured decode-only setting.
- Full 2B Q6 tied head on V3D: `328.010ms` for `248320` rows, not viable as a full-head route.

Allowed-head route:

- Indexed full-head prepack resident size: `606.25MiB`.
- Prepack-once cost in probe: about `52s`.
- Cached blob load in probe: about `125-130ms`.
- Product code should keep the prepack resident in-process and avoid per-generation reload.

Allowed rows:

| allowed rows | V3D ms | CPU oracle ms | route |
| ---: | ---: | ---: | --- |
| 3 | 0.180 | 0.105 | CPU |
| 8 | 0.186 | 0.299 | V3D |
| 13 | 0.189 | 0.475 | V3D |
| 16 | 0.191 | 0.604 | V3D |
| 32 | 0.328 | 1.188 | V3D |
| 64 | 0.380 | 2.383 | V3D |
| 128 | 0.610 | 4.815 | V3D |
| 256 | 0.922 | 9.642 | V3D |
| 1024 | 2.734 | 38.846 | V3D |
| 4096 sorted | 8.621 | 155.674 | V3D |
| 16384 sorted | 22.124 | 621.220 | V3D |

Actual Qwen tokenizer frontiers:

- Tool-call prefix over sample tools: `allowed=3`.
- Function-name starts: `allowed=2`.
- Mid-name continuations: typically `1-6`.
- Required/optional parameter open/close literals: `allowed=2`.
- Booleans: `allowed=8`.
- Bounded integer `1..8`: `allowed=8`.
- Enum `fast/safe/minimal`: `allowed=13`.

`scripts/estimate_qwen35_allowed_frontiers.cr MODEL.gguf` prints each frontier with `route`, estimated `est_v3d_ms` / `est_cpu_ms`, and an `ids_csv` field that can be passed directly to the Pi probe as `RPI5_ROW_IDS_CSV`.

Use `scripts/rpi5_q6_frontier_probe.sh LABEL IDS_CSV [REPEATS]` from the local checkout to run one of those frontiers through `ssh raspberrypi.local` without hand-writing the remote Vulkan environment.

Use `scripts/rpi5_q6_frontier_suite.sh MODEL.gguf [MAX_FRONTIERS]` to run estimator-selected frontiers. It defaults to `ROUTE_FILTER=V3D`; use `LABEL_REGEX=...`, `REPEATS=...`, and `DRY_RUN=1` to narrow or preview the run.
Set `RAW_OUTPUT=0` to suppress the full probe log; the suite still emits a machine-readable `frontier_result` row with a measured `verdict`, estimated and measured `gpu_ms` / `cpu_ms`, estimate ratios, `speedup`, `top1_match`, and throttle state. `MIN_SPEEDUP` controls the `V3D_CLEAR` cutoff; default is `1.25`.

Actual row-id probe:

- `tool_call_prefix:start`, ids `27,60638,248058`: `0.174ms`, top1 matched CPU.
- `read_file.limit` bounded integer, ids `16,17,18,19,20,21,22,23`: `0.232ms` V3D vs `0.291ms` CPU, top1 matched CPU.
- `edit_mode.mode`, 13 ids: `0.181ms`, top1 matched CPU.
- Suite wrapper check with `LABEL_REGEX='finite_values:(read_file\.limit|edit_mode\.mode)' REPEATS=40`: `read_file.limit` measured `0.239ms` V3D vs `0.295ms` CPU, and `edit_mode.mode` measured `0.234ms` V3D vs `0.487ms` CPU; both matched CPU top1 and reported `throttled=0x0`.
- Summary-only wrapper check with `RAW_OUTPUT=0 LABEL_REGEX='finite_values:read_file\.limit' REPEATS=30` emitted `verdict=V3D_NEAR`, `gpu_ms=0.261`, `cpu_ms=0.283`, `speedup=1.083x`, `v3d_est_ratio=1.403`, and `top1_match=true`, confirming `allowed=8` is positive but close to the CPU/V3D boundary.
- Summary-only wrapper check with `RAW_OUTPUT=0 LABEL_REGEX='finite_values:edit_mode\.mode' REPEATS=30` emitted `verdict=V3D_CLEAR`, `gpu_ms=0.267`, `cpu_ms=0.487`, `speedup=1.822x`, and `top1_match=true`, giving the current clean example for `allowed=13`.

## Route Policy

Use CPU for singleton/tiny frontiers unless the GPU path is already batched or resident in a command-buffer corridor.

Use V3D when:

- `allowed_count >= 8` under warmed resident-prepack conditions, or
- multiple constrained steps can be grouped/amortized, or
- logits/top1 already stay on the GPU side.

Use `QWEN35_ALLOWED_HEAD_CPU_MAX=7` as the current Pi-oriented policy override for resident Q6 allowed-head experiments. Keep it configurable: cold process startup, grammar row locality, and future V3D kernels can move the cutoff.

Sort allowed token ids for medium/broad finite sets. Sorting was neutral at `1024` rows but improved `4096` rows from `10.550ms` to `8.617ms`.

Runtime smoke on Qwen3.5-2B Q4_K_M with `QWEN35_CONSTRAINED_TOOL_CALL_PREFIX=1`, `QWEN35_ALLOWED_HEAD_CPU_MAX=7`, and `QWEN35_METAL_PROFILE=1` produced a valid parsed `edit_mode(mode=safe,dry_run=true)` tool call. Profile route markers showed `allowed_head.cpu_selected_metal_hidden=28` for tiny frontiers and `allowed_head.metal_q6=2` for finite value frontiers; the Q6 allowed-head matmul shapes were `head_top1_allowed8` and `head_top1_allowed13`.

## Product Boundary

Inputs:

- Hidden state vector for the current token.
- Certified `allowed_token_ids : Array(Int32)` from `Qwen35Constraints`.
- Resident prepacked Q6 tied head.

Output:

- Exact top1 token id among allowed ids.

Fallbacks:

- Empty allowed set: fall back to normal unconstrained decode.
- Too-small set below policy threshold: CPU row-dot/top1. The implemented runtime knob is `QWEN35_ALLOWED_HEAD_CPU_MAX=N`; default `0` preserves existing behavior.
- Too-broad set or unknown grammar state: normal full-head path.
- Missing V3D runtime/prepack: CPU path.

Kill switch:

- `QWEN35_ALLOWED_HEAD_CPU_MAX=0` disables the CPU-threshold override.
- A future V3D product adapter should still expose a direct V3D kill switch such as `QWEN35_RPI5_V3D_ALLOWED_HEAD_OFF=1`.

## Definition of Done

Focused correctness:

- For real tokenizer-derived frontiers, V3D top1 token id matches CPU top1.
- Include frontiers for tool prefix, function name, parameter open/close, boolean, bounded integer, and enum values.

Performance:

- Resident prepack setup is paid once per process/model.
- Per-step V3D timing for `allowed >= 8` is below CPU row-dot timing on Pi under warmed resident-prepack conditions.
- Tiny-frontier CPU fallback is active and measured.

Safety:

- Route only activates from a certified finite frontier.
- Full fallback remains exact and default-safe.
- Allowed ids are range-checked against vocab/head rows.

## Next Slice

Continue the default-off runtime adapter:

1. Build `TokenTextIndex` once per tokenizer.
2. For literal/tool states, compute `allowed_token_ids`.
3. Apply the route policy threshold (`QWEN35_ALLOWED_HEAD_CPU_MAX=7` for current Pi resident-Q6 experiments).
4. Use the implemented hidden/head split to avoid full-head logits when routing to CPU selected rows.
5. Behind the V3D gate, replace CPU row-dot with resident indexed Q6 head.
6. Verify exact generated token parity on constrained tool-call smokes.
