# DiffusionGemma Text Probe

`scripts/diffusion_gemma_text_probe.sh` is the current text-first entrypoint for
the native bounded sparse DiffusionGemma prototype.

It runs the native sparse-loop smoke with:

- text prompt tokenization through the Gemma4 GGUF tokenizer bridge;
- text canvas tokenization without BOS;
- pipe-separated one-token candidate texts;
- decoded canvas/candidate diagnostics;
- optional wrapper-level regression gates.

Default probe:

```sh
scripts/diffusion_gemma_text_probe.sh
```

Default inputs are:

```text
PROMPT=Say:
CANVAS=Hello
CANDIDATES=Hello|world
STEPS=1
WARMUPS=1
REPEATS=2
FORMAT=tsv
```

Useful regression gate:

```sh
EXPECT_CANVAS_LEN=1 EXPECT_CHOSEN=Hello EXPECT_MIN_PROB=0.9 scripts/diffusion_gemma_text_probe.sh
```

Override the text probe:

```sh
PROMPT='Complete:' \
CANVAS='Hello' \
CANDIDATES='Hello|world' \
EXPECT_CHOSEN=Hello \
scripts/diffusion_gemma_text_probe.sh
```

The wrapper prints the full smoke output before applying expectation checks.
Expectation gates require TSV output:

- `EXPECT_CHOSEN=TEXT` checks the first data row's `last_chosen_texts`.
- `EXPECT_MIN_PROB=F` checks every first-row `last_argmax_probabilities` value.
- `EXPECT_CANVAS_LEN=N` checks the first data row's `canvas_len`.
- `SMOKE_RUNNER=/path/to/runner` can replace the underlying smoke command for
  wrapper-level tests.

## Boundaries

This is a bounded sparse candidate prototype, not full DiffusionGemma text
generation. Candidate texts must each tokenize to exactly one token. The
tokenizer bridge still uses the DiffusionGemma-capable `llama-tokenize` oracle
for encoding text; native decoding is qualitative diagnostics only.

The default local paths are:

```text
DIFFUSION_GEMMA_MODEL=$HOME/.cache/lm-studio/models/unsloth/diffusiongemma-26B-A4B-it-GGUF/diffusiongemma-26B-A4B-it-Q4_K_M.gguf
DIFFUSION_GEMMA_LLAMA_TOKENIZE=$HOME/SrcArchives/AI/llama.cpp-diffusiongemma-pr/build-dg/bin/llama-tokenize
```

Use `scripts/diffusion_gemma_sparse_loop_smoke.sh` directly when you need
token-id, canvas-length, candidate-count, or TSV sweep controls that are outside
the text-first wrapper.

## Prompt Perf Probe

`scripts/diffusion_gemma_prompt_perf_probe.sh` runs the same bounded sparse
decode with short, medium, and long text prompts and writes one TSV row per
case:

```sh
TIMEOUT_SECONDS=45 OUT=/tmp/diffusion_gemma_prompt_perf.tsv scripts/diffusion_gemma_prompt_perf_probe.sh
```

The probe builds or reuses `DIFFUSION_GEMMA_SPARSE_LOOP_BIN` once, runs cases
sequentially, and records `ok`, `timeout`, or `failed` rows so a slow long case
does not discard earlier measurements. The main timing column for prompt-size
work is `prompt_cache_ms`; `loop_ms_median` tracks the bounded sparse candidate
loop after the prompt cache is built. `prompt_route_ms` exposes any prompt MoE
route precompute work that happens before cache construction. The
`prompt_projection_backend` column reports whether prompt Q/K/V projection
matmuls used the default CPU path or the experimental Metal path.
`prompt_projection_ms` and `prompt_materialize_ms` split prompt-cache time into
Q/K/V projection and prompt-row materialization phases. The sparse-loop smoke
also accepts `--cache-warmups N` and `--cache-repeats N`; the perf wrapper maps
those to `CACHE_WARMUPS` and `CACHE_REPEATS` and reports
`prompt_cache_ms_samples`, `prompt_projection_ms_samples`, and
`prompt_materialize_ms_samples`. The wrapper also reports sparse-loop phase
timings: `loop_prediction_ms`, `loop_update_ms`, `loop_regenerate_ms`, and
`loop_proposal_ms`, plus matching sample columns, so prompt-cache work and
canvas decode-loop work can be optimized separately. `loop_prediction_ms` is
further split into `loop_decode_stack_ms` and `loop_output_head_ms`; the decode
stack is split again into Q/K/V projection, context, attention output, shared
FFN, MoE FFN, and combine/scale timing columns.

Use cache repeats when separating Metal cold-start effects from steady-state
prompt projection timing:

```sh
DIFFUSION_GEMMA_PROMPT_PROJ_METAL=1 \
CACHE_WARMUPS=1 CACHE_REPEATS=2 TIMEOUT_SECONDS=45 \
OUT=/tmp/diffusion_gemma_prompt_perf_cache_repeats.tsv \
scripts/diffusion_gemma_prompt_perf_probe.sh
```

The sparse smoke defaults to a decode-only prompt cache: it stores the prompt
attention projections needed by canvas decode but skips materializing final
prompt rows for the last requested layer. Use
`--materialize-prompt-final-rows` on `scripts/diffusion_gemma_sparse_loop_smoke.sh`
or `MATERIALIZE_PROMPT_FINAL_ROWS=1` on the perf probe when comparing against
the conservative full-cache path.

Experimental prompt projection acceleration is opt-in:

```sh
DIFFUSION_GEMMA_PROMPT_PROJ_METAL=1 TIMEOUT_SECONDS=45 scripts/diffusion_gemma_prompt_perf_probe.sh
```

Set `DIFFUSION_GEMMA_PROMPT_PROJ_METAL_OFF=1` to force the CPU route, or
leave `DIFFUSION_GEMMA_PROMPT_PROJ_METAL` unset for the default CPU route. When
the opt-in Metal route rejects a supported-call shape with an exception, the
probe fails fast instead of hiding the error. The Metal route uses
`DIFFUSION_GEMMA_PROMPT_PROJ_METAL_MIN_BATCH=16` by default so short prompts do
not pay one-command-buffer-per-projection overhead.

For focused local performance probes,
`DIFFUSION_GEMMA_PROMPT_PROJ_METAL_MIN_BATCH=1` also lets single-row layer Q/K/V
projections use the same opt-in Metal matmul helper. Keep this as an experiment
gate: tiny Metal dispatches are noisier than the default CPU route, but they
directly target the current one-canvas-row decode-stack bottleneck.

## ABBA Promotion Gate

Use `scripts/diffusion_gemma_prompt_variant_abba.sh` for route A/B checks that
need host snapshots, quiet gating, row summaries, and a suite-level promotion
decision.

Canonical quiet promotion command for the fixed-high context batch route:

```sh
LOG_DIR=/tmp/diffusiongemma_context_batch_quiet_abba \
SEQUENCE='base variant variant base' \
BASE_ENV='DIFFUSION_GEMMA_CONTEXT_METAL_BATCH_ROWS_OFF=1' \
VARIANT_ENV='' \
DIFFUSION_GEMMA_CONTEXT_METAL=1 \
DIFFUSION_GEMMA_PROMPT_PROJ_METAL=1 \
DIFFUSION_GEMMA_PROMPT_PROJ_METAL_MIN_BATCH=1 \
DIFFUSION_GEMMA_FUSED_QK_NORM_ROPE=1 \
SYNTHETIC_PROMPT_LENGTHS=64,128,256 \
CANVAS='Hello world' \
CACHE_WARMUPS=1 \
CACHE_REPEATS=2 \
TIMEOUT_SECONDS=90 \
QUIET_MS=15000 \
LOAD_THRESHOLD=30 \
TOTAL_THRESHOLD=90 \
REQUIRE_QUIET=1 \
REQUIRE_CANDIDATE=1 \
PROMOTION_FORMAT=json \
scripts/diffusion_gemma_prompt_variant_abba.sh
```

Interpretation rules:

- `REQUIRE_QUIET=1` fails before model work if the host is too noisy.
- `REQUIRE_CANDIDATE=1` fails after summaries unless the whole suite is
  `candidate_speedup`.
- `PROMOTION_FORMAT=json` emits a structured suite decision with
  `suite_decision`, `decision_reason`, and minimum loop/context speedups.
- `CHECK_QUIET_ONLY=1` runs the same host snapshot and quiet gate path without
  building or running the model, even if the host is quiet.
- Treat rows with `blocked_by_host_noise` or `blocked_by_range` as branch
  evidence only, not promotion evidence.

Offline summaries for saved ABBA directories:

```sh
scripts/diffusion_gemma_abba_dir_summary.py /tmp/run64 /tmp/run128 /tmp/run256
scripts/diffusion_gemma_abba_promotion_summary.py --format json --roots /tmp/run64 /tmp/run128 /tmp/run256
scripts/diffusion_gemma_quiet_snapshot_summary.py --format json /tmp/diffusiongemma_quiet_check_only_current
```

Safe quiet-window polling without model work:

```sh
scripts/diffusion_gemma_quiet_gate_check.sh
```

## Artifact Suite pp/tg Summary

`scripts/diffusion_gemma_pp_tg_summary.py` also accepts artifact-suite gate or
promotion stdout files. For mixed fast/exact fallback gates it reports the
unsafe all-fast throughput and the certified mixed throughput separately:

```sh
scripts/diffusion_gemma_pp_tg_summary.py /tmp/diffusiongemma_full30_prompt_ffn_mixed_promotion_offline_20260614042452.stdout
```

The suite fields are still probe rates, not ordinary autoregressive pp/tg.
Use `mixed_pp_like_tok_s`, `mixed_layers_s`, and
`mixed_prompt_layer_rows_s` for the certified fast/exact route; use the unsafe
columns only as a diagnostic for how much the rejected fast windows would have
saved without the certificate boundary.

## Mixed Route Plan

`scripts/diffusion_gemma_prompt_artifact_suite_gate.sh` can write a JSONL
route plan with one summary row and one row per prompt/canvas window:

```sh
SUITE_MIXED_FALLBACK_GATE=1 \
SUITE_ROUTE_PLAN_OUT=/tmp/diffusiongemma_route_plan.jsonl \
scripts/diffusion_gemma_prompt_artifact_suite_gate.sh
```

`scripts/diffusion_gemma_prompt_artifact_suite_promotion.sh` writes this plan
to `LOG_DIR/route_plan.jsonl` by default for `gate` and `all` stages. Window
rows use `selected_route=variant_fast` only for certified candidate windows;
rejected windows use `selected_route=base_exact` while preserving the unsafe
variant timing and artifact path for diagnostics.

Product/runtime code should load this with
`ML::GGUF::DiffusionGemmaMixedRoutePlan.from_jsonl(path)`. The loader fails
closed for `audit_only` or `reject` summaries by default, rejects duplicate
windows and count mismatches, and requires each `variant_fast` window to carry
a `variant_route_artifact`. Use `variant_route_artifact_map` for certified fast
windows and `exact_fallback_windows_spec` for explicit exact fallback windows.

For shell/runtime handoff, `scripts/diffusion_gemma_mixed_route_plan_env.cr`
loads the same JSONL and emits `DIFFUSION_GEMMA_MIXED_*` assignments:

```sh
crystal scripts/diffusion_gemma_mixed_route_plan_env.cr --plan LOG_DIR/route_plan.jsonl
```

The helper checks that selected fast-route artifacts exist by default and exits
`4` if a selected artifact is missing. It intentionally emits
`DIFFUSION_GEMMA_MIXED_FAST_ROUTE_ARTIFACT_MAP`, not `SUITE_*`, because mixed
runtime selection has exact fallback windows while suite gates expect complete
per-window maps.

Before launching another heavy gate, inspect the mixed plan offline:

```sh
scripts/diffusion_gemma_mixed_route_plan_atlas.py LOG_DIR/route_plan.jsonl
```

The atlas ranks windows by certified mixed wall time and folds in child
`gate_metric` phase rows when the child logs still exist. Treat it as an
LTP/WBA controller: if exact fallback dominates the recomputed mixed `Phi`,
fix that certificate/fallback boundary before micro-tuning accepted fast
windows.

When a mixed-plan atlas points at one failing certificate window, inspect the
row-level certificate:

```sh
scripts/diffusion_gemma_output_cert_atlas.py CHILD_LOG_DIR/output_cert.tsv
```

This reports argmax and sampled failures per canvas row and prints an
optimistic row-local fallback lower bound. That bound is not a legal route by
itself: row-local fallback is only promotable after the runtime proves the
exact prompt-cache/hidden boundary can be narrowed or reused.

Use `--reuse-count N` and `--min-speedup S` to turn the same certificate into a
dual-cache branch-selection estimate:

```sh
scripts/diffusion_gemma_output_cert_atlas.py CHILD_LOG_DIR/output_cert.tsv \
  --reuse-count 2 \
  --min-speedup 1.10
```

By default the helper uses `--canvas-attention full`, matching the current
DiffusionGemma decode mask where every canvas query can attend every canvas
key. Under that boundary, the row-local figure is printed only as a lower bound
and the legal fallback grain is `canvas_band`; a future runtime path must prove
base-cache reuse and exact band predict fallback before promotion. The reported
`break_even_uses` is therefore a target for a future runtime boundary, not speed
evidence by itself.

`scripts/diffusion_gemma_prompt_output_cert_probe.cr` can also consume the same
plan directly:

```sh
crystal build scripts/diffusion_gemma_prompt_output_cert_probe.cr \
  -o /tmp/diffusion_gemma_prompt_output_cert_probe \
  --link-flags="$(pwd)/build/bridge.o -framework Metal -framework Foundation -framework MetalPerformanceShaders -lc++"

/tmp/diffusion_gemma_prompt_output_cert_probe \
  --mixed-route-plan LOG_DIR/route_plan.jsonl \
  --dry-run-route-selection
```

For `variant_fast` windows, the probe runs the variant arm with the selected
route artifact. For `base_exact` fallback windows, it runs the variant side
through the base env and does not load the rejected/unsafe variant artifact.
Explicit `--*-route-artifact*` options are rejected when `--mixed-route-plan` is
set, so a mixed plan remains the single route authority.
