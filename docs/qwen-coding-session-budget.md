# Coding-session timing budget

This is an offline measurement boundary, not an engine speed promotion.
The intended value is time to a correct task result with instruction and safety
constraints retained. Provider throughput alone is only a diagnostic.

`scripts/qwen_coding_session_budget.py` consumes trusted, complete, sequential
headless CrystalBall CogniQwen logs plus externally measured session wall time:

```sh
python3 scripts/qwen_coding_session_budget.py --log /absolute/harness.stdout.log --wall-ms 123456
PYTHONDONTWRITEBYTECODE=1 python3 -m unittest discover -s spec -p qwen_coding_session_budget_spec.py -v
```

Admitted: one performance record per `Calling LLM` thought; equal records from
different calls count separately. The plain final performance summary is a
duplicate and is ignored. Sum render, load, tokenize, cache lookup, prefill plus
first-token selection, and decode body. Never add `greedy_ms`: it contains
prefill/decode. Preserve both provider-unattributed and outside-provider time.
The latter includes startup, tools, orchestration, provider work outside its
timer, and any enclosing runner overhead. It is **not** measured tool time.
These are host-wall intervals, not GPU utilization or per-kernel costs.

Reject absent, unpaired, unfinished, duplicate-field or numerically inconsistent records;
nonfinite/negative measured durations; and provider totals exceeding session wall.
Only `route`, `total_ms` and the six listed timing leaves are required; unused
parent timers, token counts, TPS fields and route names are not validated.
Allow at most 1 ms rounding discrepancy. Hidden-test time must not be included
in the input wall. The caller must declare whether guard startup is included.
Require exactly one `Agent finished` marker after the calls; it establishes
log termination, not correctness. Text logs are trusted input, not
injection-resistant telemetry. Parallel, concatenated, retried-without-record,
provider-error or call-truncated logs are not complete budgets. A wrong task
answer may still have a complete timing budget; quality is assessed separately.

Quality stays separate: execute fixture tests after the agent exits and inspect
the changed files. Do not infer correctness from exit zero, final prose, ECS,
top-1 agreement, or a timing record. A fixture hidden from the prompt is not an
OS-sealed evaluation. One task is a routing baseline, not general coding
capability or a protocol A/B result.

No production changes are admitted by this slice. Before optimization, retain
the full tool-mode system prompt, model, task, limits and cache policy; compare
repeated matched successful sessions and report failure outcomes separately.
Refresh the parser contract when CrystalBall emission/timing ownership changes.
Rollback is removal of this offline script; it is not wired into inference.

## Observed baseline, 2026-09-07

Source: cogni-ml `51de8bf1`, CrystalBall `4d52b33`; fresh release harness,
Apple M2 Max, Qwen3.8-27B Q4_K_M. No production source changes in this slice.
Build from CrystalBall with a fresh temporary `CRYSTAL_CACHE_DIR`:

```sh
crystal build src/harness.cr --release -o /absolute/tmp/harness --link-flags='-framework Metal -framework Foundation -lc++'
```

The isolated `eval/tasks/cogni_qwen_quadrumvirate/happy-path/project` fixture
exposes only its project and prompt to the harness. Copy `hidden_spec.cr` into
the temporary project's spec directory only after the process exits. The
original fails 4/4 visible-plus-hidden checks; a separate oracle copy with
`Math.max` for the upper endpoint passes 4/4. The live output is byte-identical
to the broken original and fails 4/4. Do not overwrite that evidence with the
oracle fix.

Run configuration: clear inherited `QWEN35_*`, `CB_COGNI_QWEN_*`, and
`COGNI_RUN_SAFE_*` experimental switches; select the explicit Qwen3.8 model
file. Effort `none`, compact tool prompt OFF, harness Quadrumvirate OFF,
post-read no-tools OFF, phase enforcement OFF, rehydration OFF, constrained
tools ON, resident prefix ON with reserve 1024, disk prompt cache OFF.
This is the ordinary F32-KV provider baseline, **not adaptive-QBit evidence**.
Prefill chunk 2048, append groups 1, cooldown 50 ms. Metal lease wait 10000 ms,
command timeout 180000 ms. The safe runner retains a 35% free-memory floor,
24576 MiB process-tree cap and 300-second limit, without a quiet-host wait.
Harness flags: `--model cogni-qwen --llm-max-tokens 256 --llm-timeout 180
--shell-timeout 60 --max-turns 8 --auto-approve-verify --no-rehydration
--allow-tools read_file,grep,glob,list_directory,shell,edit_file`, followed by
the unchanged fixture prompt. Stdin is closed. Normal tool-mode system
instructions are retained; project bootstrap/rehydration is not tested.

Observed result: guarded external wall **101927.5 ms**, including runner
startup but excluding later tests. First call: 7813 prompt tokens, 27 output
tokens, provider wall 90897.9 ms, prefill/first-token 84209.4 ms, decode body
6326.7 ms. The model only ran `ls -la`. The second call records a resident
prefix hit and then Metal `completion_status=-6`; stderr names
`Impacting Interactivity`. This is a correlation, not proof that prefix reuse
caused the GPU failure. No driver timeout or memory-guard kill is reported.
Sampled free memory was 85% before launch, 55% during, and 77% after exit;
these samples are not a peak-memory trace.

The adapter rescues the exception into ordinary response text
(`src/llm/cogni_qwen.cr:883-885`), and the harness prints `Agent finished` and
returns **exit 0**. The error call also repeats previous usage counters. These
are concrete reasons not to use process status or token totals as success.
The new budget CLI rejects the real log with exit 1 (missing error-call timing
fields), rather than reporting a successful first-call-only session budget.
This rejection is covered by a regression test.

Verification: 10 budget unit tests and 4 existing coding-scorer tests pass;
duplicate-field and post-call-truncation/concatenation tests were observed
failing before rejection was added.
The successful synthetic logs qualify parsing, not a successful model run.
Source review finds no leaf-timer overlap, including cache continuation;
adversary scope is trusted current timing records, not full-record validation.
**Open:** no time-to-correct-code baseline or speed improvement is established.
Next, separate error propagation from the Metal failure: a CPU-only provider
error regression can qualify fail-closed harness status; a bounded two-call
prefix-reuse probe must discriminate reuse from scheduling/interactivity before
changing engine defaults. Do not lengthen watchdogs to obtain a pass.

Temporary evidence (may expire): `/private/tmp/qwen-coding-budget.BS00g9/`,
`harness.stdout.log`, `harness.stderr.log`, `run.json`, `run_baseline.py`,
`work/`, and separate `oracle/`. The initial sandbox invocation aborted before
workload isolation; the live invocation used the same guards outside sandbox.
Harness SHA256 `08baae4dc8cee9d626b41a3c91986727b07cfa5183d9c160751bfb3017c04a76`;
bridge SHA256 `48bb1469e2a473d30a94ab102df91268d549a4dd3710b076a0e59d137691005a`.
