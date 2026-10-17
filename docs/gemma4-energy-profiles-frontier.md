# Gemma 4 energy profiles

Status: implemented; scoped NVIDIA/llama.cpp and raw Fluke checks passed.
Sustained GPU stability, energy-optimal defaults and answer-quality rankings
remain unestablished.

## Boundary

The Linux integration uses a local llama.cpp server as the full-depth inference
backend. cogni-ml owns the Eco (0/256), Balanced (256/512), and Quality
(1024/1024) reasoning/answer token budgets. These are initial policy defaults,
not measured quality or energy optima. Existing Metal diagnostics keep their
current behavior. Layer truncation, synthetic CUDA probes as useful inference,
and guaranteed instantaneous watt limits are rejected.

Thinking uses Gemma 4's native thought channel. A bounded reasoning completion
is followed by an answer completion with the closed channel in its actual
prompt; closing a thought channel must never consume the answer allowance.
Every generated token, including stop/control tokens, counts against its phase
allowance. Context overflow, unexpected phase termination, backend errors, and
unsupported model architecture fail explicitly. The report records forced
closure and answer truncation. Timing includes both requests and re-prefill.

VictoriaMetrics is read through `/api/v1/export` raw JSONL. Exactly one physical
series is required. Merge split rows, sort timestamps, deduplicate identical
samples, reject conflicting duplicates and non-finite/negative values. Integrate
piecewise linearly over bracketing samples with a bounded maximum gap. Missing
coverage is unavailable, never zero. Fluke current multiplied by assumed 116 V
is apparent power (VA); its integral is VA s. Active watts yield joules. An
explicit assumed power factor can yield *estimated* joules, with provenance.
The collector's `fluke_current_amperes` series and its stable labels are the
physical source; no per-request labels are added to that stream.

Comparison keeps raw profile runs, prompt/model/settings identity, wall-clock
windows, phase counts and outcomes. A cheaper truncated/incorrect answer is not
evidence of better efficiency. No automatic quality ranking is admitted.

## Execution and falsifiers

Risk: CAUTION (new CLI and token/measurement semantics). Rollback: revert the
single feature commit; deployment lives in `/build/persisten/cogni-gemma4-energy`
and owns only the processes it starts. Existing dirty paths are out of scope.

1. Add failing budget/backend tests, then `src/ml/gguf/gemma4_energy_profile.cr`
   and `src/ml/gguf/gemma4_energy_runner.cr`. Guard zero/negative limits,
   natural/forced channel closure, answer allowance, EOS and context overflow.
2. Add failing raw-series/integration tests, then
   `src/ml/energy/victoria_metrics.cr`. Guard gaps, ambiguity, duplicate rows,
   missing boundaries and VA/J distinction.
3. Add `bin/gemma4_energy.cr` and a route from `bin/gemma4_generate.cr`.
   Exercise independent budget overrides and all three profiles on gputer;
   compare raw measurement windows there. Missing live measurements remain an
   explicit limitation, not a successful physical-energy test.

DoD: focused Crystal specs pass on gputer; CLI builds there; real Gemma 4
inference returns separate bounded phase counts and nonempty answers for all
profiles; an adversarial forced-close run preserves the answer allowance;
VictoriaMetrics comparison passes real-server integration tests and reports
coverage/provenance for live Fluke data if available. Broader guard: existing
Gemma chat specs and the existing generation wrapper dry-run contract.

Primary contracts: [Gemma formatting](https://ai.google.dev/gemma/docs/core/prompt-formatting-gemma4),
[llama-server API](https://github.com/ggml-org/llama.cpp/blob/master/tools/server/README.md),
[VictoriaMetrics raw export](https://docs.victoriametrics.com/victoriametrics/url-examples/#apiv1export).

## Direct gputer evidence, 2026-10-17

The isolated deployment is `/build/persisten/cogni-gemma4-energy` on the
user-provided gputer host. The original remote checkout was not modified.
`tools/INSTALLATION.md` records installation digests, build commands, service
configuration and logs. Both services bind only to loopback: llama-server on
8080 and VictoriaMetrics on 8428; VM retains samples for seven days.

- NVIDIA RTX 5060 Ti, llama.cpp b11714 (`c0fd78626`), context 8192, one slot,
  greedy decoding, no prompt cache or continuous batching. Loader evidence in
  `tools/llama-server.log` reports **49/49 layers offloaded to CUDA**. This is
  full-depth inference through llama.cpp; native cogni-ml CUDA layer probes and
  Radeon generation are outside this result.
- `models/gemma-4-12B-it-Q4_K_M.gguf`, 7,381,382,944 bytes, SHA-256
  `95d83ba36642b1f385fb906b5962a71763361be3bac930a709945f72d97473f8`.
  The public lmstudio-community Gemma 4 GGUF was verified before loading.
- Crystal 1.22.0, OpenSSL 3.0.13, VictoriaMetrics v1.154.0. Focused specs passed
  **22 examples, zero failures**, including the disposable real-VM import/export
  fixture. The renderer passed **10 Python tests**. Its fixtures are synthetic,
  separately stored, and never counted as physical measurements.
- `reports/defaults.json`: nine real requests, three rounds with rotating
  profile order, for `What is 17 times 23? Give a short answer.` All completed,
  returned `17 times 23 is 391.`, respected separate phase caps and had live
  Fluke coverage. Eco generated 0 reasoning / 13 answer tokens; Balanced and
  Quality both ended reasoning naturally after 177 tokens and generated 13
  answer tokens. Caps do not force the model to consume the allowance.
- `reports/forced-close.json`: reasoning cap 8, answer cap 64. Reasoning closed
  after exactly 8 tokens; the independent answer used 13 tokens and returned
  the correct result. `reports/truncated-answer.json`: a 3-token answer was
  explicitly truncated and was not approved for quality comparison.
- `reports/missing-data.json`: selecting a missing circuit with
  `--measure-only --measurement-wait 0 --require-measurement` returned exit 3
  and unavailable measurements, without manufacturing zero consumption.
- Imported reports now require complete, coherent phase counts and termination
  fields. A zero-token fabricated answer failed the new falsifier after the
  import guard was repaired. Imported inference remains explicitly unverified;
  structural validation is not an authenticity check. Existing generation
  wrapper old/new dry-run contracts passed on gputer; the existing Gemma chat
  suite passed locally (nine examples, CPU-only).

The longer enterprise explanation prompt completed for all three profiles in
`reports/comparison.json`: 0/132, 256/132 and 1024/136 actual reasoning/answer
tokens, with 2.940/8.811/26.707 s request windows. Reasoning reached the cap in
Balanced and Quality, while all three answers terminated normally. A subsequent
three-round attempt (`reports/comparison-repeated.json`) completed its first
round, then the NVIDIA runtime aborted at 19:02:10 UTC with CUDA launch timeout,
watchdog and Xid 8. OOM counters were zero. The controller recorded failed
requests with incomplete usage and retained their physical cost windows.
The owned llama-server was restored; its crash log is retained as
`tools/llama-server-crash-20261017T190210Z.log`. The root cause is unresolved;
do not claim a successful nine-run long-request benchmark or sustained GPU
stability from these observations.

Live `fluke_current_amperes{circuit="gputer",mode="ac"}` and matching validity
arrived from the Mac BLE collector through an ephemeral loopback-to-SSH command
relay. SSH TCP forwarding was disabled. Raw notification timestamps and physical
labels were preserved. All physical results use assumed 116 V and **VA / VA s**;
`active_energy_joules` remains null. The current signal does not establish real
active watts or joules without another measurement or a stated PF assumption.

An eight-round clock probe with minimum RTT 16.4 ms estimated the server clock
33.4 ms ahead of the collector host, with about 8.2 ms transport uncertainty.
No clocks were changed. BLE notifications were about four per second; meter
refresh/averaging can lag GPU changes. The ~0.35 s Eco arithmetic runs are too
short to establish a causal energy ratio, and carryover from earlier load is
visible. Other host activity remains uncontrolled. No quiet-host, general
efficiency improvement, or quality superiority claim is admitted.

Adversary verdict: **ROBUST within the functional token-cap, explicit-failure,
raw-coverage and unit boundaries tested above**. The largest remaining inference
assumption is that the configured llama-server honors its reported full model
and token accounting. Physical efficiency conclusions require matched useful
answers, longer controlled runs, calibration and appropriate power measurement.
Refresh this evidence after source, backend/model/tokenizer, GPU, collector,
clock synchronization, measurement configuration or calibration changes; missing
raw evidence makes a measurement unavailable and reopens the dependent claim.
