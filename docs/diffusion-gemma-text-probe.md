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
When a mixed rerun selects a pre-cached exact fallback artifact, the suite
writer records that artifact under `base_route_artifact` and leaves
`variant_route_artifact` empty unless an actual variant artifact was loaded.

Product/runtime code should load this with
`ML::GGUF::DiffusionGemmaMixedRoutePlan.from_jsonl(path)`. The loader fails
closed for `audit_only` or `reject` summaries by default, rejects duplicate
windows and count mismatches, and requires each `variant_fast` window to carry
a `variant_route_artifact`. Use `variant_route_artifact_map` for certified fast
windows, `exact_fallback_windows_spec` for explicit exact fallback windows, and
`exact_fallback_route_artifact_map` when a route plan carries optional base
artifacts for exact fallback replay.

For shell/runtime handoff, `scripts/diffusion_gemma_mixed_route_plan_env.cr`
loads the same JSONL and emits `DIFFUSION_GEMMA_MIXED_*` assignments:

```sh
crystal scripts/diffusion_gemma_mixed_route_plan_env.cr --plan LOG_DIR/route_plan.jsonl
```

The helper checks that selected runtime artifacts exist by default and exits
`4` if a selected artifact is missing. It emits separate
`DIFFUSION_GEMMA_MIXED_FAST_ROUTE_ARTIFACT_MAP`,
`DIFFUSION_GEMMA_MIXED_EXACT_FALLBACK_ROUTE_ARTIFACT_MAP`, and
`DIFFUSION_GEMMA_MIXED_SELECTED_ROUTE_ARTIFACT_MAP` variables instead of
`SUITE_*`, because mixed runtime selection may have exact fallback windows while
suite gates expect complete per-window maps.

If base artifacts are prepared after the mixed plan already exists, attach them
without hand-editing JSONL:

```sh
scripts/diffusion_gemma_attach_fallback_artifacts.py \
  LOG_DIR/route_plan.jsonl \
  --prepare-log PREPARE_LOG_WITH_SUITE_BASE_ROUTE_ARTIFACT_MAP \
  --out LOG_DIR/route_plan_with_base_fallback.jsonl
```

The helper only accepts `base_exact` windows in the map, checks artifact files
by default, and clears fallback `variant_route_artifact` fields unless
`--keep-fallback-variant-artifacts` is passed. Use the derived plan as the
single route authority for the next mixed gate.

To run the whole fallback-replay measurement path when the host is quiet:

```sh
MIXED_ROUTE_PLAN=LOG_DIR/route_plan.jsonl \
scripts/diffusion_gemma_fallback_replay_gate.sh
```

The wrapper derives `base_exact` windows from the plan, prepares only
`SUITE_ARTIFACT_ARMS=base` artifacts for those windows, attaches the resulting
base map, and then runs the mixed gate from the derived route plan. It defaults
to `CHECK_QUIET=1` and `VARIANT_PROFILE=prompt-ffn-resident` for the current
full-depth route-plan family. Use `DRY_RUN=1` to inspect commands, or
`FALLBACK_REPLAY_STAGE=prepare|attach|gate` to split the workflow. After a
successful gate, the wrapper writes `fallback_replay_route_plan_atlas.txt`,
`fallback_replay_route_plan_atlas.tsv`, and `fallback_replay_pp_tg.tsv` under
`LOG_DIR`; set `FALLBACK_SUMMARY=0` to skip these best-effort summaries.

Before launching another heavy gate, inspect the mixed plan offline:

```sh
scripts/diffusion_gemma_mixed_route_plan_atlas.py LOG_DIR/route_plan.jsonl
```

The atlas ranks windows by certified mixed wall time and folds in child
`gate_metric` phase rows when the child logs still exist. Treat it as an
LTP/WBA controller: if exact fallback dominates the recomputed mixed `Phi`,
fix that certificate/fallback boundary before micro-tuning accepted fast
windows. For exact fallback windows, the atlas also reads the child
`output_cert_log` when available and prints a cert-derived dual-cache
canvas-band fallback estimate. Compare it cautiously: `selected_route_ms` comes
from route-plan ABBA timing, while `dual_cache_band_ms` comes from output-cert
timing. If `dual_cache_band_vs_selected_route < 1`, the known mixed route should
keep `base_exact` and the next runtime work should reduce the exact fallback
itself rather than try variant-first fallback.

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
variant artifact. For `base_exact` fallback windows, it runs the variant side
through the base env and never loads the rejected/unsafe variant artifact; if
the route plan carries a `base_route_artifact`, that base artifact is loaded as
the selected runtime artifact with `arm=base`.
Explicit `--*-route-artifact*` options are rejected when `--mixed-route-plan` is
set, so a mixed plan remains the single route authority.

For the next fallback experiment outside `--mixed-route-plan`, the certificate
and ABBA probes can intentionally load a route artifact whose stored metadata
belongs to another arm/env role. This is explicit and still fail-closed because
the shared v2 route-artifact loader validates the requested metadata:

```sh
/tmp/diffusion_gemma_prompt_output_cert_probe \
  --variant-route-artifact /tmp/base_p4096_c8192_pl16_l30.tsv \
  --variant-route-artifact-expected-arm base \
  --variant-route-artifact-env-role base

/tmp/diffusion_gemma_prompt_cache_abba \
  --full-routes \
  --materialize-final-rows \
  --variant-route-artifact /tmp/base_p4096_c8192_pl16_l30.tsv \
  --variant-route-artifact-expected-arm base \
  --variant-route-artifact-env-role base
```

Use this only as a foreign-route replay experiment: it can test whether the
variant/exact-fast runtime profile improves a fallback window while replaying
base routes. Promotion still requires the output certificate and ABBA timing to
pass on the same artifact and a recomputed mixed route plan to reduce the
fallback-dominated Phi.

The certified gate wrapper forwards the same controls to both child probes via
shared defaults:

```sh
BASE_ROUTE_ARTIFACT_EXPECTED_ARM=base \
VARIANT_ROUTE_ARTIFACT_EXPECTED_ARM=base \
BASE_ROUTE_ARTIFACT_ENV_ROLE=base \
VARIANT_ROUTE_ARTIFACT_ENV_ROLE=base \
ABBA_VARIANT_ROUTE_ARTIFACT=/tmp/base_p4096_c8192_pl16_l30.tsv \
scripts/diffusion_gemma_prompt_certified_variant_gate.sh
```

Use the `CERT_*` or `ABBA_*` prefixed forms when certificate and ABBA need
different explicit artifacts. These expected-metadata overrides are rejected
when `MIXED_ROUTE_PLAN`/`CERT_MIXED_ROUTE_PLAN`/`ABBA_MIXED_ROUTE_PLAN` is set,
because a mixed route plan owns the selected artifact arm/env role.

The fallback replay wrapper has explicit frames. The default
`FALLBACK_REPLAY_MODE=selected` attaches prepared base artifacts back into a
derived mixed plan and measures the known `base_exact` fallback. The experimental
`foreign` mode prepares/reuses those same base artifacts, but feeds them through
the suite's variant artifact map with `VARIANT_ROUTE_ARTIFACT_EXPECTED_ARM=base`
and `VARIANT_ROUTE_ARTIFACT_ENV_ROLE=base`:

```sh
FALLBACK_REPLAY_MODE=foreign \
MIXED_ROUTE_PLAN=/tmp/diffusiongemma_route_plan_full30_20260614043810/route_plan.jsonl \
scripts/diffusion_gemma_fallback_replay_gate.sh
```

Use `DRY_RUN=1` first to inspect the derived fallback windows, base artifact
map, and `foreign_gate_cmd`.

Use `FALLBACK_REPLAY_MODE=compare` when the next quiet window should measure
both frames from the same prepared or reused base artifacts. It writes separate
`gate_selected` and `gate_foreign` directories, plus separate atlas and pp/tg
summary files:

```sh
FALLBACK_REPLAY_MODE=compare \
MIXED_ROUTE_PLAN=/tmp/diffusiongemma_route_plan_full30_20260614043810/route_plan.jsonl \
scripts/diffusion_gemma_fallback_replay_gate.sh
```

When reusing existing artifacts in compare mode, pass them through
`FALLBACK_BASE_ROUTE_ARTIFACT_MAP` so selected and foreign replay use the same
base map. `FALLBACK_FOREIGN_BASE_ROUTE_ARTIFACT_MAP` is foreign-only and is
rejected for compare mode.

The promotion wrapper records and forwards the same expected-metadata controls
into its nested suite gate, so a dry-run `gate_cmd` is self-contained for
foreign replay. Non-default expected metadata is still rejected when a mixed
route plan is active, because the plan owns selected artifact semantics.
