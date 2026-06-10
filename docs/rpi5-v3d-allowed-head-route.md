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

`scripts/estimate_qwen35_allowed_frontiers.cr MODEL.gguf` prints each frontier with `route`, estimated `est_v3d_ms` / `est_cpu_ms`, and an `ids_csv` field that can be passed directly to the Pi probe as `RPI5_ROW_IDS_CSV`. Its default `policy_cpu_max_allowed=12` is conservative for unbatched real frontiers; set `QWEN35_ALLOWED_HEAD_CPU_MAX=7` to re-enable near-boundary `allowed=8` V3D route experiments.

Use `scripts/rpi5_q6_frontier_probe.sh LABEL IDS_CSV [REPEATS]` from the local checkout to run one of those frontiers through `ssh raspberrypi.local` without hand-writing the remote Vulkan environment. Set `RPI5_WARMUPS=N` for untimed dispatches before measurement.

Use `scripts/rpi5_q6_frontier_suite.sh MODEL.gguf [MAX_FRONTIERS]` to run estimator-selected frontiers. It defaults to `ROUTE_FILTER=V3D`; use `LABEL_REGEX=...`, `REPEATS=...`, and `DRY_RUN=1` to narrow or preview the run.
Set `RAW_OUTPUT=0` to suppress the full probe log; the suite still emits a machine-readable `frontier_result` row with a measured `verdict`, estimated and measured `gpu_ms` / `cpu_ms`, estimate ratios, `speedup`, `top1_match`, and throttle state. `MIN_SPEEDUP` controls the `V3D_CLEAR` cutoff; default is `1.25`.

Use `scripts/rpi5_q6_policy_calibrate.sh MODEL.gguf` to run the current policy frontiers and emit `calibration_policy`. The default labels cover `tool_call_prefix:start` (`allowed=3`), `read_file.limit` (`allowed=8`), and `edit_mode.mode` (`allowed=13`). Calibration defaults to `RPI5_WARMUPS=3` so first-submit latency is excluded from the averaged GPU time.

Use `scripts/rpi5_q6_policy_repeat_sweep.sh MODEL.gguf` to rerun calibration across `REPEAT_COUNTS` and expose near-boundary stability. This is a measurement stability check for the current one-frontier submit path, not command-buffer batching.

Use `scripts/rpi5_q6_batch_sweep.sh MODEL.gguf` to run a proxy batching sweep over `RPI5_BATCH`. Each batch row uses a distinct synthetic hidden vector with the same allowed-token frontier, so it measures command/dispatch amortization and shader batch indexing.

Use `scripts/rpi5_q6_multi_frontier_sweep.sh MODEL.gguf` to group several tokenizer-derived frontiers into one `q6idx` submit with per-row frontier offsets/counts.

Use `scripts/estimate_rpi5_allowed_batch_plan.cr MODEL.gguf` to plan grouped frontier submits from the estimator output. It is a planning model over measured probe anchors, not a product benchmark.

Use `scripts/plan_rpi5_from_frontier_trace.cr TRACE.log` to parse real `QWEN35_CONSTRAINT_FRONTIER_TRACE=1` generation logs into CPU/V3D grouping candidates.

Use `scripts/rpi5_q6_probe_trace_groups.sh TRACE.log` to run the planned V3D groups from a runtime trace log directly on the Pi probe.

Use `QWEN35_ALLOWED_HEAD_CAPTURE_PATH=/tmp/allowed_head.jsonl` during a constrained structured run to capture the real pre-output-norm hidden vector paired with each runtime `allowed_ids` set. This forces the exact hidden-readback/selected-row path and is intended for replaying real hidden rows through the Pi probe; it is not a product route by itself.

Use `scripts/export_allowed_head_capture_replay.cr CAPTURE.jsonl OUT.f32 [MAX_ROWS]` to convert captured hidden rows into a raw little-endian Float32 batch and print the matching `ids_groups`. Set `MIN_ALLOWED=4` to export only rows above the current tiny-CPU policy. Copy `OUT.f32` to the Pi probe directory and pass it through `RPI5_X_F32_LOAD=OUT.f32` with `RPI5_ROW_IDS_CSV_BATCH`.

Actual row-id probe:

- `tool_call_prefix:start`, ids `27,60638,248058`: `0.174ms`, top1 matched CPU.
- `read_file.limit` bounded integer, ids `16,17,18,19,20,21,22,23`: `0.232ms` V3D vs `0.291ms` CPU, top1 matched CPU.
- `edit_mode.mode`, 13 ids: `0.181ms`, top1 matched CPU.
- Suite wrapper check with `LABEL_REGEX='finite_values:(read_file\.limit|edit_mode\.mode)' REPEATS=40`: `read_file.limit` measured `0.239ms` V3D vs `0.295ms` CPU, and `edit_mode.mode` measured `0.234ms` V3D vs `0.487ms` CPU; both matched CPU top1 and reported `throttled=0x0`.
- Summary-only wrapper check with `RAW_OUTPUT=0 LABEL_REGEX='finite_values:read_file\.limit' REPEATS=30` emitted `verdict=V3D_NEAR`, `gpu_ms=0.261`, `cpu_ms=0.283`, `speedup=1.083x`, `v3d_est_ratio=1.403`, and `top1_match=true`, confirming `allowed=8` is positive but close to the CPU/V3D boundary.
- Summary-only wrapper check with `RAW_OUTPUT=0 LABEL_REGEX='finite_values:edit_mode\.mode' REPEATS=30` emitted `verdict=V3D_CLEAR`, `gpu_ms=0.267`, `cpu_ms=0.487`, `speedup=1.822x`, and `top1_match=true`, giving the current clean example for `allowed=13`.
- Policy calibration with `RAW_OUTPUT=0 REPEATS=40 scripts/rpi5_q6_policy_calibrate.sh MODEL.gguf` emitted `policy_cpu_max_allowed=3`, `v3d_clear_min_allowed=13`, `near_boundary_allowed=8`. The measured rows were `allowed=3 verdict=CPU_WINS`, `allowed=8 verdict=V3D_NEAR`, and `allowed=13 verdict=V3D_CLEAR`.
- Warmed policy repeat sweep with `RAW_OUTPUT=0 RPI5_WARMUPS=3 REPEAT_COUNTS='20 40' scripts/rpi5_q6_policy_repeat_sweep.sh MODEL.gguf` emitted stable `V3D_CLEAR` for both `read_file.limit` (`allowed=8`, `gpu_ms=0.144/0.143`, `cpu_ms=0.289/0.289`) and `edit_mode.mode` (`allowed=13`, `gpu_ms=0.147/0.147`, `cpu_ms=0.488/0.481`), all with `top1_match=true` and `throttled=0x0`.
- Batch proxy sweep with `RAW_OUTPUT=0 RPI5_WARMUPS=3 BATCH_COUNTS='1 2 4 8' scripts/rpi5_q6_batch_sweep.sh MODEL.gguf` preserved CPU/GPU parity across all rows (`max_abs_diff<=4.47e-7`, `throttled=0x0`). `read_file.limit` (`allowed=8`) improved from `0.168ms/row` at batch 1 to `0.126ms/row` at batch 8, with speedup rising `1.719x -> 2.398x`; `edit_mode.mode` (`allowed=13`) improved from `0.169ms/row` to `0.1265ms/row`, with speedup `2.967x -> 3.828x`. A follow-up `allowed=8` sweep through batch 32 saturated near `0.1215ms/row`.
- Multi-frontier batch proof with `RAW_OUTPUT=1 RPI5_WARMUPS=3 MAX_FRONTIERS=3 REPEATS=30 scripts/rpi5_q6_multi_frontier_sweep.sh MODEL.gguf` grouped `tool_call_prefix:start` (`allowed=3`), `read_file.limit` (`allowed=8`), and `edit_mode.mode` (`allowed=13`) into one `q6idx13_l256` submit. It measured `gpu_ms=0.393`, `cpu_ms=0.915`, `gpu_ms_per_frontier=0.131`, `speedup=2.326x`, `max_abs_diff=2.98e-7`, `throttled=0x0`.
- Batch planner with default labels estimates the same `3/8/13` group at `grouped_v3d_ms=0.393`, `grouped_vs_cpu=2.237x`, and `grouped_vs_unbatched_v3d=1.412x`.
- Tool frontier trace for `edit_mode` has six ranked frontier corridors. Blind V3D grouping estimates `0.786ms`, but hybrid routing with `CPU_TINY_MAX=3` leaves four tiny `allowed<=3` corridors on CPU and groups the two finite-value corridors (`allowed=13/8`) on V3D. The measured two-frontier Pi run reported `gpu_ms=0.281`, `cpu_ms=0.802`, `speedup=2.852x`, `max_abs_diff=1.94e-7`, `throttled=0x0`; the trace planner estimates hybrid total `0.596ms` vs all-CPU `1.089ms`.
- Runtime trace parser smoke on a short Qwen3.5-0.8B `edit_mode` run parsed four prefix-stage rows (`allowed=3/4/5/1`) and emitted one weak V3D candidate group (`allowed=4/5`) with estimated `hybrid_vs_cpu=1.108x`. This confirms the parser path and also shows why tiny/near-tiny frontiers should stay policy-gated.
- Trace-log-to-Pi probe wrapper on the same short log ran the planned `allowed=4/5` group as `q6idx5_l256`, measured `gpu_ms=0.273`, `cpu_ms=0.359`, `speedup=1.316x`, `max_abs_diff=2.09e-7`, `throttled=0x0`. This confirms end-to-end trace extraction and probe execution, while keeping the near-tiny route classified as marginal.
- Longer runtime trace on Qwen3.5-0.8B with `QWEN35_CONSTRAINED_TOOL_CALL_PREFIX=1`, `QWEN35_CONSTRAINT_FRONTIER_TRACE=1`, one `edit_mode` tool, and `n_gen=64` reached real value-literal rows and emitted a valid parsed `edit_mode(mode=safe,dry_run=true)` tool call. The trace had `31` constrained frontier rows, including two `value_literal allowed=8` rows. `scripts/plan_rpi5_from_frontier_trace.cr` planned six V3D groups with `hybrid_total_ms=3.1624` vs all-CPU `4.1306` (`1.306x`). Live `scripts/rpi5_q6_probe_trace_groups.sh` on the same log measured grouped Pi probe speedups of `1.424x`, `1.519x`, `1.794x`, `1.787x`, `1.455x`, and `1.326x`, with `max_abs_diff<=3.58e-7` and `throttled=0x0`.
- Runtime capture smoke with `QWEN35_ALLOWED_HEAD_CAPTURE_PATH=/tmp/qwen35_allowed_head_capture_20260609_220135.jsonl` on the same `edit_mode` prompt emitted `31` capture rows for `31` trace rows, kept the parsed tool call as `edit_mode(mode=safe,dry_run=true)`, and wrote rows with `source=metal_hidden`, `allowed_ids`, `hidden_dim=1024`, and the pre-output-norm hidden vector. This validates local capture of real hidden rows; replay on Pi is still pending.
- Real-hidden 2B replay: Qwen3.5-2B capture on the same prompt emitted `31` rows with `hidden_dim=2048`, matching the resident 2B Q6 tied head on the Pi. All-V3D replay of those rows as one `q6idx8_l256` batch measured `gpu_ms=3.521`, `cpu_ms=4.231`, `speedup=1.202x`, `max_abs_diff=1.14e-5`, `throttled=0x0`; this is positive but confirms tiny-frontier dilution. Filtering with `MIN_ALLOWED=4` exported `16` V3D rows and measured `gpu_ms=1.847`, `cpu_ms=3.205`, `speedup=1.735x`, `max_abs_diff=8.58e-6`, `throttled=0x0`. Combining that with the all-vs-filter CPU delta for the `15` tiny rows gives a replay hybrid estimate of about `2.873ms` vs all-CPU `4.231ms` (`1.47x`).

## Route Policy

Use CPU for singleton/tiny frontiers unless the GPU path is already batched or resident in a command-buffer corridor. Current real-frontier calibration sets the conservative CPU clear zone at `allowed<=3`.

Use V3D when:

- `allowed_count >= 8` under warmed resident-probe calibration, or
- multiple constrained steps can be grouped/amortized, or
- logits/top1 already stay on the GPU side.

Treat `allowed=8` as sensitive to measurement corridor until the product adapter owns a warmed resident command path. Without warmups it has shown near-boundary/noisy results; with `RPI5_WARMUPS=3` it measured as `V3D_CLEAR`.
When several constrained steps can share one resident submit, batching improves the margin. The probe now supports different frontiers per batch row, but it still uses synthetic hidden rows; product batching must wire real decode hidden states and grammar frontiers into the same resident command path before promotion.

The longer runtime trace confirms that finite value-literal frontiers occur in the real structured decode path, not only in the offline frontier estimator. It does not yet prove product RPi5 speedup because the Pi probe consumes the real row-id groups but still generates synthetic hidden vectors.

`QWEN35_ALLOWED_HEAD_CAPTURE_PATH` plus `RPI5_X_F32_LOAD` closes the probe-side synthetic hidden gap for offline replay. The remaining product adapter step is resident transport of real hidden rows to the Pi/V3D allowed-head path without JSONL files, SSH process boundaries, or per-run prepack reloads.

Use `QWEN35_ALLOWED_HEAD_CPU_MAX=7` as the Pi-oriented policy override for warmed resident Q6 allowed-head experiments. A conservative unbatched policy can still prefer `QWEN35_ALLOWED_HEAD_CPU_MAX=12` until the product V3D adapter reproduces the warmed corridor. Keep it configurable: cold process startup, grammar row locality, and future V3D kernels can move the cutoff.

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
3. Apply the route policy threshold (`QWEN35_ALLOWED_HEAD_CPU_MAX=12` for conservative unbatched Pi resident-Q6 experiments; lower only when accepting near-boundary routes or after fresh calibration).
4. Use the implemented hidden/head split to avoid full-head logits when routing to CPU selected rows.
5. Behind the V3D gate, replace CPU row-dot with resident indexed Q6 head.
6. Replace synthetic probe hidden rows with product decode hidden rows for grouped constrained steps.
7. Use `QWEN35_CONSTRAINT_FRONTIER_TRACE=1` on short structured smokes to capture the real runtime allowed-head frontier rows that an RPi5 adapter should group.
6. Verify exact generated token parity on constrained tool-call smokes.
