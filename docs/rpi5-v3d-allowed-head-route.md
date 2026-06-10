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

Use `scripts/rpi5_q6_capture_replay.sh CAPTURE.jsonl 4` to automate the conversion, copy, all-row replay, filtered replay, and hybrid estimate.

Use `scripts/rpi5_q6_capture_smoke.sh` to run the full local 2B `edit_mode` constrained capture smoke and then replay it on the Pi.

Use `scripts/rpi5_q6_allowed_head_regression.sh` as the default local pre-adapter gate. It runs marker smoke plus capture smoke without Pi replay; set `FULL_REPLAY=1` to include the Pi replay step.

Use `scripts/rpi5_q6_transport_budget.sh CAPTURE.jsonl 4` to measure the current non-product transport envelope around the remote Pi probe. This separates the resident-kernel opportunity from SSH/process/Vulkan setup and prepack-load tax, and prints a `resident_budget_result` row that includes mapped hidden/row-id upload before each resident submit.

Use `scripts/rpi5_q6_resident_budget_gate.sh CAPTURE.jsonl 4` as a fail-closed resident-budget gate. It requires resident upload request speedup over CPU selected-row fallback, bounded `max_abs_diff`, and clean throttle state.

Use `scripts/rpi5_q6_resident_stream_bench.sh CAPTURE.jsonl 4` to replay captured rows as a resident per-step request stream: one Vulkan setup, one prepacked Q6 head load, then one hidden/row-id upload and submit per captured constrained step. This is the closest current probe to a product worker boundary, but it still runs outside the real decode loop.

Use `scripts/rpi5_q6_resident_stream_gate.sh CAPTURE.jsonl 4` as a fail-closed resident stream gate. It requires per-step resident stream speedup over CPU selected-row fallback, bounded `max_abs_diff`, zero top1 mismatches, clean throttle state, and at least `MIN_REPEATS=10` timing repeats by default.

Use `scripts/rpi5_q6_resident_stdin_smoke.sh CAPTURE.jsonl 4` to test the daemon-like stdin request boundary. The C probe keeps Vulkan/prepack resident, then accepts `bin<TAB>ids_csv` frames followed by raw Float32 hidden bytes and returns `resident_stdin_result` rows. The legacy `hidden.f32<TAB>ids_csv` request form remains available for debugging.

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
- Capture replay wrapper check with `RAW_OUTPUT=0 REPEATS=30 RPI5_WARMUPS=3 bash scripts/rpi5_q6_capture_replay.sh /tmp/qwen35_2b_allowed_head_capture_20260609_220934.jsonl 4` reproduced the same shape: all rows `gpu_ms=3.517`, `cpu_ms=4.259`, `speedup=1.211x`; filtered rows `gpu_ms=1.846`, `cpu_ms=3.223`, `speedup=1.746x`; hybrid estimate `2.882ms` vs all-CPU `4.259ms` (`1.478x`).
- End-to-end capture smoke wrapper check with `RAW_OUTPUT=0 REPEATS=30 RPI5_WARMUPS=3 scripts/rpi5_q6_capture_smoke.sh` produced a fresh 2B structured capture (`31` trace rows, `31` capture rows), preserved parsed `edit_mode(mode=safe,dry_run=true)`, then replayed on Pi. Results: all rows `gpu_ms=3.520`, `cpu_ms=4.250`, `speedup=1.207x`; filtered rows `gpu_ms=1.845`, `cpu_ms=3.233`, `speedup=1.753x`; hybrid estimate `2.862ms` vs all-CPU `4.250ms` (`1.485x`), `throttled=0x0`.
- Full regression gate with `FULL_REPLAY=1 RAW_OUTPUT=0 REPEATS=30 RPI5_WARMUPS=3 scripts/rpi5_q6_allowed_head_regression.sh` passed marker checks, produced a fresh 2B capture (`31` trace rows, `31` capture rows), and replayed on Pi. Results: all rows `gpu_ms=3.517`, `cpu_ms=4.220`, `speedup=1.200x`; filtered rows `gpu_ms=1.844`, `cpu_ms=3.227`, `speedup=1.750x`; hybrid estimate `2.837ms` vs all-CPU `4.220ms` (`1.487x`), `throttled=0x0`.
- Transport budget check with `RAW_OUTPUT=0 REPEATS=30 RPI5_WARMUPS=3 scripts/rpi5_q6_transport_budget.sh /tmp/qwen35_2b_allowed_head_capture_20260609_223005.jsonl 4` replayed the filtered `16` real-hidden rows with `gpu_ms=1.844`, `cpu_ms=3.332`, `speedup=1.807x`, and `max_abs_diff=8.58307e-06`, but the current remote probe envelope measured `remote_wall_ms=656.856`, including `prepack_load_ms=127.827` and estimated setup overhead `464.845ms` (`252.1x` one averaged GPU dispatch). The same run emitted `resident_budget_result` with `resident_kernel_ms=1.844`, `resident_upload_request_ms=1.854`, `resident_upload_request_ms_per_row=0.115875`, `cpu_selected_ms_per_row=0.208250`, and `upload_request_speedup=1.797x`. This is a falsifier for SSH/process-per-call product routing and a target for the resident adapter: mapped hidden/row-id upload is not the dominant remaining tax.
- Resident budget gate with `REPEATS=30 RPI5_WARMUPS=3 scripts/rpi5_q6_resident_budget_gate.sh /tmp/qwen35_2b_allowed_head_capture_20260609_223005.jsonl 4` passed with `upload_request_speedup=1.792`, `max_abs_diff=8.58307e-06`, threshold `MIN_UPLOAD_SPEEDUP=1.25`, and `throttled=0x0`.
- Full regression gate now runs the resident budget gate when `FULL_REPLAY=1`. Smoke `FULL_REPLAY=1 RAW_OUTPUT=0 REPEATS=10 RPI5_WARMUPS=3 scripts/rpi5_q6_allowed_head_regression.sh` produced a fresh `31` row capture, replayed all rows at `gpu_ms=3.517`, `cpu_ms=4.405`, replayed filtered rows at `gpu_ms=1.846`, `cpu_ms=3.330`, estimated hybrid `2.921ms` vs all-CPU `4.405ms` (`1.508x`), then passed `resident_budget_gate_result` with `upload_request_speedup=1.866`, `max_abs_diff=8.58307e-06`, and `throttled=0x0`.
- Resident stream bench with `RAW_OUTPUT=0 REPEATS=30 scripts/rpi5_q6_resident_stream_bench.sh /tmp/qwen35_2b_allowed_head_capture_20260609_223005.jsonl 4` replayed the filtered `16` rows as independent per-step requests inside one resident probe process. It measured `resident_stream_ms=0.160` per request vs CPU selected-row `0.212ms` (`1.326x`), with `max_abs_diff=8.58307e-06`, `top1_mismatches=0`, one-time `prepack_load_ms=127.349`, and `throttled=0x0`. Boundary: this removes SSH/process/Vulkan setup from the per-request path, but still uses a probe-side worker rather than the product decode runtime.
- Full regression gate now also runs the resident stream gate when `FULL_REPLAY=1`. Smoke `FULL_REPLAY=1 RAW_OUTPUT=0 REPEATS=5 RPI5_WARMUPS=2 scripts/rpi5_q6_allowed_head_regression.sh` preserved marker smokes and parsed `edit_mode(mode=safe,dry_run=true)`, produced a fresh `31` row capture, replayed filtered rows at `gpu_ms=1.844`, `cpu_ms=3.255`, passed resident budget gate with `upload_request_speedup=1.745`, then passed resident stream gate with `stream_speedup=1.342`, `max_abs_diff=8.58307e-06`, `top1_mismatches=0`, and `throttled=0x0`.
- Remote temp cleanup is fail-safe for the capture replay and resident stream wrappers. The capture replay wrapper now removes remote `.f32` artifacts even if the probe exits non-zero, and resident stream uses a remote `EXIT` trap. Adversary check: after short replay and one fail-closed `REPEATS=2` stream gate attempt, `ssh raspberrypi.local 'ls ~/cogni-ml-vulkan-probe/rpi5_resident_stream_*.f32 ~/cogni-ml-vulkan-probe/rpi5_allowed_head_*.f32 2>/dev/null || true'` printed no files.
- Resident stream timing gate now rejects undersampled runs before contacting the Pi: `REPEATS=2 scripts/rpi5_q6_resident_stream_gate.sh ...` exits `2` with `REPEATS=2 below resident stream gate minimum 10`. The normal `REPEATS=30` gate still passes on the same capture with `stream_speedup=1.326`, `max_abs_diff=8.58307e-06`, `top1_mismatches=0`, and `throttled=0x0`.
- Resident stdin smoke with `MAX_ROWS=2 RPI5_WARMUPS=3 scripts/rpi5_q6_resident_stdin_smoke.sh /tmp/qwen35_2b_allowed_head_capture_20260609_223005.jsonl 4` now sends hidden vectors as binary stdin frames, not per-request `.f32` files. A follow-up gate run emitted `resident_stdin_result` rows with `top1_match=true`, `max_abs_diff<=2.86102e-06`, top1 ids plus `gpu_top1_logit`/`cpu_top1_logit`, and `throttled=0x0`; remote stdin temp cleanup was empty. Boundary: this is a daemon-protocol proof that streams the local replay payload directly over SSH stdin, not the final product IPC or Crystal runtime adapter.
- `ML::GGUF::Qwen35Rpi5AllowedHeadClient` now owns the Crystal-side binary-frame writer and `resident_stdin_result` parser for the probe protocol, and `scripts/rpi5_q6_resident_stdin_smoke.sh` uses `scripts/rpi5_q6_resident_stdin_frames.cr` instead of bash `printf/dd` loops for request framing. Focused specs cover little-endian Float32 framing, batch export, empty/out-of-range request rejection, old/new result parsing, ignored non-result rows, malformed-row fail-closed behavior, hidden-batch metadata mismatch, fail-closed conversion into the `{id, logit}` tuple expected by `forward_top1_allowed`, and an injectable single-request transport facade. Boundary: this is a typed protocol/client contract for the future adapter; the hot decode path still does not call the RPi worker.
- `Qwen35CPU.forward_top1_allowed` now has a default-off resident transport hook after the hidden/head split. It only runs for Q6 output heads above `QWEN35_ALLOWED_HEAD_CPU_MAX`, outside capture mode, and only when `Qwen35CPU.allowed_head_resident_transport` is explicitly set; missing or invalid resident rows fall through to the exact CPU selected-row fallback. Boundary: this is a wiring point, not the final RPi worker implementation or a speed claim.
- `Qwen35Rpi5AllowedHeadClient::ResidentProcessTransport` is the first long-lived runtime transport primitive: it starts one process with stdin/stdout pipes, writes binary request frames through the existing protocol helper, skips worker preamble until the next `resident_stdin_result` row, serializes concurrent callers with a mutex, and closes/fails silent on transport errors so `top1_allowed?` keeps the CPU fallback exact. Focused spec evidence uses a fake resident worker and proves two requests share one process. Boundary: this still needs generator/env wiring and a real SSH/Vulkan worker command before it is a product route.
- Full replay regression now runs the resident stdin gate when `FULL_REPLAY=1`. Smoke `FULL_REPLAY=1 RAW_OUTPUT=0 REPEATS=10 RPI5_WARMUPS=2 scripts/rpi5_q6_allowed_head_regression.sh` preserved marker smokes and parsed `edit_mode(mode=safe,dry_run=true)`, passed capture replay, resident budget gate, resident stream gate, and resident stdin gate, then ended with `allowed_head_regression_result marker=ok capture=ok full_replay=1 resident_budget_gate=1 resident_stream_gate=1 resident_stdin_gate=1`.

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

With `QWEN35_METAL_PROFILE=1`, route markers distinguish the policy reason before the selected-row fallback: `allowed_head.cpu_threshold`, `allowed_head.q6_off`, or `allowed_head.capture`, followed by the hidden source marker such as `allowed_head.cpu_selected_metal_hidden`.

Marker smoke checks:

- `QWEN35_ALLOWED_HEAD_CPU_MAX=7` produced `allowed_head.cpu_threshold=28`, `allowed_head.cpu_selected_metal_hidden=28`, and `allowed_head.metal_q6=2` while preserving parsed `edit_mode(mode=safe,dry_run=true)`.
- `QWEN35_ALLOWED_HEAD_Q6_OFF=1` produced `allowed_head.q6_off=30` and `allowed_head.cpu_selected_metal_hidden=30` while preserving the same parsed tool call.

Use `scripts/rpi5_q6_marker_smoke.sh` to rerun both marker checks as a local regression gate.

## Stateful Boundary

Constrained decode is causal and state-mutating: each `forward_top1_allowed` consumes the previous token, mutates KV/DeltaNet state, and only then reveals the next hidden row. A product V3D adapter must therefore treat the normal greedy path as a per-step resident call, not as a blind batch of future constrained decisions.

Legal batching frames:

- Offline replay from captured hidden rows, as implemented by `QWEN35_ALLOWED_HEAD_CAPTURE_PATH` plus `scripts/rpi5_q6_capture_replay.sh`.
- Independent requests or externally supplied hidden rows where state mutation has already happened.
- A future speculative corridor that checkpoints state, computes proposal hidden rows, verifies exact accepted tokens, and rolls back on rejection.

Illegal promotion:

- Grouping future constrained decode steps before their predecessor token is known.
- Treating replay-batch speedup as end-to-end decode speedup without transport, state, and rollback accounting.

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
- `QWEN35_ALLOWED_HEAD_Q6_OFF=1` disables the Q6 allowed-head fast route and forces the exact hidden-readback/CPU selected-row fallback.

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
5. Wire `Qwen35Rpi5AllowedHeadClient::ResidentProcessTransport` into an explicit generator/env adapter for the real SSH/Vulkan command, then run a constrained tool-call parity smoke with `QWEN35_ALLOWED_HEAD_CPU_MAX=7`.
6. Keep CPU selected-row fallback for `allowed <= QWEN35_ALLOWED_HEAD_CPU_MAX`, missing V3D runtime, transport errors, and kill-switch activation.
7. Verify exact generated token parity on constrained tool-call smokes.
8. Only introduce multi-row batching under an explicit replay/independent-request/speculative-corridor frame.
