# Experimental Gemma 4 decoder depth

Status: the bounded 48/32/16/48 experiment completed on gputer on 2026-10-17.
User-authorized hackathon experiment; answer accuracy is not an acceptance
criterion. Normal energy profiles continue to execute the full model.

## Hypothesis and discriminator

Retaining fewer decoder blocks may shorten inference and reduce integrated GPU
and circuit cost for a fixed generated-token count. It need not reduce
instantaneous GPU utilization or power: the remaining blocks can keep the GPU
busy. Compare fixed generated work, elapsed time, GPU power/utilization and
Fluke apparent-power windows separately.

`--gpu-layers` changes CPU/GPU placement and is not a decoder-depth experiment.
The proposed reversible probe changes the loaded model's block-count metadata
to retain a prefix of decoder blocks, while using the original final norm and
output head. Admit a reduced case only if the backend actually constructs fewer
blocks and the model's KV-sharing/layout requirements remain compatible. Such
an early exit has no answer-quality guarantee and must be labelled experimental.

## Bounds

Risk: CAUTION. Use the owned deployment in
`/build/persisten/cogni-gemma4-energy`, immutable original GGUF bytes, loopback services
and bounded requests. Preserve the full-model startup configuration and restore
it after the experiment. Do not reset the GPU, change power limits, stop other
workloads or mutate the model file. A previous NVIDIA watchdog/Xid 8 is an
unresolved stability risk; retain failures instead of reporting them as savings.

DoD: logs establish actual decoder counts and complete GPU offload for each
accepted depth; the same prompt/settings and exact requested generated-token
count are used; live GPU telemetry and Fluke coverage are recorded per request;
missing coverage or backend failure is explicit; the original full model is
healthy after rollback. Reject an incompatible metadata override rather than
calling CPU offload or a malformed load a successful shallow model.

The collector/VM metrology and raw-sample guards remain as documented in
[energy profiles](gemma4-energy-profiles.md): Fluke uses VA / VA s, whereas
NVIDIA's reported active board power uses W / J and is scoped to the GPU.

## Mechanism and measurement

The inspected original has 48 decoder blocks, no PLE embeddings and no shared
KV suffix. Its two per-layer arrays (KV head count and sliding-window pattern)
have 48 entries. A scalar block-count override leaves incompatible arrays.
The prefix writer therefore creates new GGUFs with blocks `0..N-1`, cropped
layer metadata and the unchanged non-block tensors. The token embedding is
also the tied output-head storage in this model. Original weights are neither
decoded nor requantized. Unsupported layouts fail explicitly.

`scripts/gemma4_prune_layers.py` inspects stored tensor element counts and
payload bytes, and creates prefix copies with SHA-256 manifests. These counts
describe unique stored parameters, not a direct wattage estimate.

`bin/gemma4_layer_experiment.cr` sends a fixed prompt with greedy decoding,
`cache_prompt=false`, `ignore_eos=true` and an exact 256-token limit. It requires
256 returned token IDs and a token-limit stop. Continuing beyond EOS is
intentional synthetic work; it does not establish useful-answer efficiency.
The decoder count supplied to this controller requires corroborating loader
logs (`n_layer_all=N`, `N+1/N+1` GPU offload), not merely a report label.

`scripts/gemma4_gpu_telemetry.py` samples the supported NVIDIA
`power.draw.instant` board reading every 250 ms. It preserves raw JSONL and
imports a separate `cogni_gpu_power_watts{host="gputer",gpu="0",run="..."}`
series into VictoriaMetrics. The experiment reuses cogni-ml's raw export and
bracketing/gap guards to integrate W into J. Sensor update frequency may be
slower than polling. GPU utilization is a sample mean, reported separately.
GPU board power includes other GPU activity; Fluke includes the whole circuit.

Parameter count alone cannot determine watts. A calibrated predictor can use
retained weight bytes, token throughput, context, batch size, precision and
clocks; its coefficients require measurements on the target hardware. The
directly observed quantities here are mean board W and its request-window
integral J, with no idle-power subtraction. Fewer blocks may save J while
leaving instantaneous W or utilization nearly unchanged.

A possible calibrated model is `P ~= P_idle + e_byte * B/t + e_op * F/t`,
where `B` is bytes actually moved, `F` is executed operations, and `t` is
elapsed time. Stored parameter count does not determine these quantities or
the hardware-specific coefficients. This experiment does not fit a wattage
predictor from three model sizes. NVIDIA documents the board reading in
[nvidia-smi](https://docs.nvidia.com/deploy/nvidia-smi/index.html).

## Reproduction

Use the original GGUF as an immutable source and fresh output paths. On gputer,
the deployment root is `/build/persisten/cogni-gemma4-energy`.

```sh
python3 scripts/gemma4_prune_layers.py inspect /path/to/gemma4.gguf
python3 scripts/gemma4_prune_layers.py prune /path/to/gemma4.gguf \
  --keep-blocks 32 --output /path/to/gemma4-depth32-experimental.gguf
python3 scripts/gemma4_prune_layers.py prune /path/to/gemma4.gguf \
  --keep-blocks 16 --output /path/to/gemma4-depth16-experimental.gguf
crystal build bin/gemma4_layer_experiment.cr --release -o /path/to/gemma4_layer_experiment
```

Run the existing Fluke collector continuously into VictoriaMetrics with
`circuit="gputer"`. Start the bounded GPU collector before inference and use
the same campaign label in all batches:

```sh
python3 scripts/gemma4_gpu_telemetry.py --run-label depth-example \
  --duration-seconds 300 --interval-ms 250 --output /path/to/new-gpu.jsonl \
  --vm-url http://127.0.0.1:8428
```

In another shell, load one model at a time using the established llama-server
flags: `--ctx-size 8192 --gpu-layers 999 --parallel 1 --no-cache-prompt
--cache-reuse 0 --no-cont-batching`. Verify its decoder count and complete GPU
offload in the loader log before setting `--layers`. With that service ready:

```sh
/path/to/gemma4_layer_experiment --layers 32 --label depth32 \
  --campaign-id depth-example --model-file /path/to/gemma4-depth32-experimental.gguf \
  --tokens 256 --trials 3 --output /path/to/new-depth32.json
python3 scripts/gemma4_layer_report.py merge \
  --input /path/to/depth48-before.json /path/to/new-depth32.json \
  /path/to/depth16.json /path/to/depth48-after.json --output /path/to/comparison.html
```

Use separate new reports for each batch and restore the original startup script
and full model after testing. The 48-before / 32 / 16 / 48-after ordering is a
drift check, not randomized isolation from other workloads. Backend failures
remain in reports and block the fixed-token gate. Invalid configuration is
rejected before inference; missing measurements remain unavailable.

## Observed gputer result

Campaign `depth-20261017-a` used an RTX 5060 Ti, driver 595.104.02,
llama.cpp b11714 (commit `c0fd78626`) and Gemma 4 12B Q4_K_M. Each row below
averages three completed trials with exactly 256 generated tokens. All 12 GPU
and Fluke windows had bracketing coverage; loading, hashing, eight-token warmup
and waits were outside those windows. The prompt digest is
`aec49f7658576d37a23d3180917cf7f53285f9ebf55db09ca6194a2202c42986`.

| Batch | Decoder blocks | Stored parameters | Seconds | GPU W | Fluke VA | GPU J/request | Fluke VA s/request |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Full before | 48 | 11,907,350,576 | 5.658 | 159.7 | 234.8 | 903.8 | 1328.7 |
| Prefix 32 | 32 | 8,267,880,992 | 3.970 | 161.5 | 230.5 | 640.9 | 914.9 |
| Prefix 16 | 16 | 4,628,411,408 | 2.316 | 144.8 | 202.1 | 335.1 | 468.3 |
| Full after | 48 | 11,907,350,576 | 5.635 | 163.5 | 235.3 | 921.4 | 1326.3 |

Relative to the initial full-model batch, the 16-block batch used about 63%
less measured GPU energy and took about 59% less time. The 32-block batch
reduced energy while its mean watts stayed near the full model. This refutes
using stored weight count as a proportional wattage estimator in this setup.
The 16-block trial means ranged from 134.9 to 154.0 W, so the mean-power effect
needs longer, isolated windows before a precise claim. Sensor averaging,
shared GPU activity and short current windows remain limits. This is fixed
synthetic token work; it does not establish quality-preserving energy savings.

## Paired Fluke and NVIDIA visualization

The report pairs both sources by their recorded request boundaries. A pair
requires matching integer start/end timestamps, matching durations, compatible
W/VA classes, and strictly ordered curves spanning the full window. Missing or
shifted coverage stays unavailable. Per-label paired means require every trial
to pass these checks and the exact-token check; the renderer does not silently
drop incomplete trials from a label mean. Matching recorded windows does not
establish hardware synchronization or remove sensor lag, averaging or clock
offset. No idle subtraction or time shift is applied.

The preserved `depth-20261017-a` report has 12/12 matched pairs. Relative to the
initial full-model batch, depth 16 reduced GPU J/request by 62.9% and the Fluke
VA s/request integral by 64.8%. These are separate changes within each source.
NVIDIA measures active GPU-board power; Fluke covers the measured circuit and,
without a measured power factor, supplies apparent power. Do not subtract W
from VA, infer GPU share, or calculate PSU efficiency from this pair. See
[Fluke's electrical glossary](https://www.fluke.com/en-us/learn/blog/electrical/electrical-glossary)
for the active/apparent power distinction.

`gemma4_layer_report.py` includes an interactive Canvas explorer and the paired
static traces in the full report. `--presentation-only` creates a compact,
self-contained HTML page for the hackathon pitch:

```sh
python3 scripts/gemma4_layer_report.py /path/to/comparison.json \
  --output /path/to/fluke-nvidia-comparison.html
python3 scripts/gemma4_layer_report.py /path/to/comparison.json \
  --presentation-only --output /path/to/depth-explorer.html
node --test spec/test_gemma4_depth_chart.js
```

Copy `gemma4_depth_chart.js` and `gemma4_depth_chart.css` alongside the Python
renderer when deploying its source. The output embeds both assets and the
measured curves; opening it requires no CDN, package install or network access.
The Canvas sizing and interaction patterns are adapted from Tokenomics Viewer's
`public/index.html` under its retained ISC notice. The source checkout is named
`tokenomics` locally. Neither that repository nor the team's published pitch is
modified by generating the report.

The explorer switches between NVIDIA W and Fluke VA, individual trials, and
J/token or W/token. Two aligned 2D panels share elapsed milliseconds: solid
power above and a same-color dashed ratio below. Fluke ratios retain VA/token
and VA·s/token. The two 48-layer batches remain separate to expose drift.
Bounded time-parametric Catmull–Rom curves smooth the display only; energy
integrates the original piecewise-linear samples, with interpolated request
boundaries and no extrapolation or synthetic zeros. Zoom/pan and point
inspection affect presentation only. Duplicate
label/trial identities, non-exact runs and unmatched windows are excluded
explicitly from the explorer; the full report retains their original records.
The plot is a historical measurement view, not a live feed or useful-answer
efficiency claim.

For a static pitch site, copy the generated page to that site's directory and
link to it, or embed it without extra dependencies:

```html
<iframe src="./depth-explorer.html" title="Measured Gemma 4 power traces"
        style="width:100%;height:1100px;border:0" loading="lazy"></iframe>
```

The current visualization passes 27 Python report tests locally and on gputer,
22 JavaScript contract/projection tests locally, and JavaScript syntax checking.
The full and compact HTML outputs are checked against gputer rendering.
Browser checks cover sensor/trial/ratio switching, zoom/reset, point inspection,
the PNG action, and 375/611 px layouts. These checks
verify the rendering and rejection guards, not sensor calibration or model
quality. The pitch bundle's separate evidence index records source and artifact
hashes without replacing the original experiment index.

### Observed token counters

Add `--token-progress` to the layer experiment to count token IDs in the
completion stream. The observer retains receipt timestamps, checks the final
count against fixed work, and sends `cogni_llm_generated_tokens_total` through
Prometheus remote-write v1 to the configured loopback VictoriaMetrics service.
Its bounded labels identify campaign, host, run label, depth, trial and answer
phase. Missing or malformed streams fail explicitly; failed metric writes
preserve the report and return a failing CLI status.

When counters are available, the chart uses cumulative received N(t), held
between observations, and omits ratios before the first token. The historical
depth campaign has no token timestamps: its dashed curves explicitly use final
N=256 instead. W/token is instantaneous power divided by the chosen token count;
it is distinct from energy/token and throughput. Timing is observer receipt on
gputer, not a hardware timestamp for each GPU token.

Direct probe `token-progress-20261017-b` completed 256 tokens at 48 layers in
5,588.13 ms. All 257 recorded counter points, including zero and terminal
anchors, exactly match raw VictoriaMetrics export timestamps and values. Both
GPU and Fluke request windows were available. This verifies the observer and
transport on the pinned backend, not a new depth comparison or answer quality.
The probe JSON, raw export and roundtrip check are preserved with the report
artifacts. Focused software checks are:

```sh
crystal spec spec/gemma4_token_progress_spec.cr spec/prometheus_remote_write_spec.cr \
  spec/gemma4_layer_experiment_spec.cr spec/gemma4_energy_cli_spec.cr
node --test spec/test_gemma4_depth_chart.js
python3 -m unittest discover -s spec -p test_gemma4_layer_report.py
```

The Crystal checks pass 19 examples locally and on gputer. Rollback: omit
`--token-progress` or revert this feature; historical reports remain readable.
Refresh the live counter evidence after backend, clock, stream, wire encoding,
or measurement changes. The longer BLE capture ended with a collector storage
error after the retained request; this probe does not establish sustained
collector operation.

## Model provenance and rollback

Loader logs show `n_layer_all` 48/32/16/48 and GPU offload 49/49, 33/33,
17/17, 49/49 respectively. Every retained tensor payload was independently
hashed against its original counterpart (446 tensors at depth 32; 225 at
depth 16). The original GGUF SHA-256 remained
`95d83ba36642b1f385fb906b5962a71763361be3bac930a709945f72d97473f8`.
Derived digests are:

- Depth 32: `c64b8b038823f5193e0694cad850f75c66e08b4b43cc77cb3301bdda746ea9ac`.
- Depth 16: `d685e10c5ba33d5f913740e5ea0ad588f9c0b412a3e0d43a7b07ac4241b65234`.

The original full model was restored and completed its final three trials.
No new backend failure occurred in these bounded runs; the earlier watchdog
incident remains an unresolved sustained-load risk. Source evidence is under
the deployment's `reports/depth-20261017-a-*` and `tools/llama-depth*.log`;
the preserved final startup log is `llama-depth48-after.log` in the local
artifact bundle. Refresh results on model, backend, GPU/driver, prompt/settings,
collector, clocks, sensor calibration or concurrent-load changes, or evidence
loss. Focused test commands are:

```sh
crystal spec spec/gemma4_layer_experiment_spec.cr
python3 -m unittest discover -s spec -p test_gemma4_prune_layers.py
python3 -m unittest discover -s spec -p test_gemma4_gpu_telemetry.py
python3 -m unittest discover -s spec -p test_gemma4_layer_report.py
```


## Same-host UT61B campaign (2026-10-17)

Campaign `depth-ut61b-20261017-c` repeats the prefix-depth experiment with the
UT61B USB adapter connected directly to gputer as `/dev/ttyUSB0`. It uses the
same owned deployment, model hashes, prompt, greedy settings and exact
256-token workload. Batches are `depth48-before`, `depth32`, `depth16` and
`depth48-after`, with three requests each. The full 48-layer service is restored
before the final batch and remains healthy after both collectors stop.

Both bounded collectors run for 300 seconds on gputer. GPU power, utilization
and memory are captured every 250 ms; UT61B retains complete serial frames and
rejected candidates with receipt epoch milliseconds and monotonic timestamps.
Actual received-token counters use the same host's epoch clock, guarded against
wall/monotonic request-duration drift. These are observer receipt timestamps,
not device-side sampling timestamps. A common host removes inter-host clock
offset but does not remove meter averaging, transport or token-delivery lag.

Raw-only GPU and UT61B captures are subsequently sent through Prometheus
remote-write v1 by `scripts/gemma4_same_clock_report.py`. Crystal already writes
the token counters through that protocol. The postprocessor exports all seven
metric families from VictoriaMetrics, rejects missing/conflicting/extra samples
within the selected capture ranges, and compares timestamps exactly and values
with absolute tolerance `1e-12` before constructing measurements. The UT61B
held valid-AC state merges every mode, including unknown and rejected frames;
request boundaries must be bracketed, with raw gaps at most 2000 ms.

A successful write can precede raw-export visibility. If a prior attempt wrote
successfully but failed the bounded readback, `--verify-existing` performs the
same full comparison without sending another write. Its output explicitly
records `write_attempted=false` and does not invent write-response statistics.
Existing evidence paths are never overwritten; retain a failed partial export
and choose a fresh prefix if one already exists.

From `/build/persisten/cogni-gemma4-energy`, after the four batch reports and
bounded captures have closed:

```sh
python3 source/scripts/gemma4_same_clock_report.py \
  --input reports/depth-ut61b-20261017-c-depth48-before.json \
    reports/depth-ut61b-20261017-c-depth32.json \
    reports/depth-ut61b-20261017-c-depth16.json \
    reports/depth-ut61b-20261017-c-depth48-after.json \
  --gpu-jsonl reports/depth-ut61b-20261017-c-gpu.jsonl \
  --ut61b-jsonl reports/depth-ut61b-20261017-c-ut61b.jsonl \
  --output reports/depth-ut61b-20261017-c-comparison.json \
  --evidence-prefix reports/depth-ut61b-20261017-c-vm
```

The new comparison explicitly identifies `circuit_meter=ut61b` and
`measurement_clock=gputer`. Original batch reports retain their original
measurements; the postprocessor replaces those legacy circuit fields using
fresh UT61B samples, without relabeling Fluke evidence. Legacy reports continue
to render as Fluke, and mixed-meter merges or inconsistent metric identities
are rejected. UT61B records cannot claim active circuit W/J.

The compact UT61B presentation is the demo's single-source view, using solid
VA and same-color dashed VA s/token or VA/token against actual cumulative
received tokens. NVIDIA remains in the separate full technical comparison.
Historical inputs and the earlier complete presentation bundle are retained
under `previous-f4bb6457/` in the task artifact. Restoring the archived HTML is
the chart rollback; restoring the normal startup script is the model rollback.


Observed means (three requests per row, 256 tokens each):

| Batch | Time (s) | GPU W | GPU J | UT61B VA | UT61B VA s | UT61B VA s/token |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 48 before | 5.601 | 156.36 | 875.72 | 246.91 | 1382.88 | 5.402 |
| 32 | 3.973 | 153.97 | 611.83 | 232.24 | 922.84 | 3.605 |
| 16 | 2.322 | 135.28 | 314.08 | 227.82 | 529.07 | 2.067 |
| 48 after | 5.631 | 167.33 | 942.23 | 254.23 | 1431.54 | 5.592 |

The reduced-depth UT61B request integrals are 33.3% and 61.7% below the initial
48-layer batch, while mean apparent power changes much less. The final full
batch's UT61B integral is 3.52% higher than the initial full batch; GPU energy
is 7.59% higher. Keep both baselines visible rather than hiding drift or claiming
an invariant causal savings ratio. These short windows include prefill and
sensor lag, exclude loading/warmup/waits, and do not establish useful-answer
quality, calibration, quiet-host operation or comparative meter accuracy.

The 300-second capture retained 1199 GPU samples with zero sample errors and
507 valid AC frames plus one initial partial frame before any request. Raw VM
exports match 8207 unique metric samples: 1199 each for GPU W, utilization and
memory; 508 UT61B validity, 507 current, 507 apparent VA; and 3088 observed-token
counter points across 12 series. No same-millisecond UT61B rows coalesced.
The initial write was accepted before raw exports became visible; the final
`--verify-existing` readback sent no second write. An independent artifact
checker, importing no collector/writer/integrator implementation, reconstructs
UT61B display values, verifies every raw sample, and recomputes all 24 power
curves and window integrals. The GPU/UT61B wall-minus-monotonic receipt offsets
occupy a 1.017 ms corridor; this is not a sensor-latency measurement.

Evidence is retained as `depth-ut61b-20261017-c-*`,
`check_same_clock_evidence.py`, `ut61b-nvidia-comparison-evidence.json`,
`ut61b-nvidia-comparison.html` and `depth-explorer.html` in the task artifact's
`layer-depth/`. Raw captures, VM exports and source batch reports also remain
under the owned gputer deployment's `reports/`. Refresh on changes to source,
model/backend, GPU/driver, workload, clock, USB/meter/adapter, serial settings,
voltage assumption or calibration, concurrent activity, or raw-evidence loss.


Verification for this source state: the following 69 Python tests passed both
locally and on gputer; 27 Node chart tests passed locally, including a canvas
mount with actual-token progress. Browser checks exercised all three trials,
both UT61B ratio modes, point inspection, zoom/reset, PNG prepared status, and
375/611 px layouts. The full technical view mounted and switched from NVIDIA
to UT61B. PNG downloaded bytes were not inspected. Compact and full HTML hashes
match the independently rendered gputer copies. The independent raw-evidence
checker passed locally and on gputer. The scoped adversary verdict is ROBUST
for acquisition/readback, unit separation and these rendered paths; causal
savings, calibration and useful-answer efficiency remain unverified.

```sh
python3 -m unittest spec/test_gemma4_same_clock_report.py spec/test_gemma4_layer_report.py spec/test_ut61b_telemetry.py spec/test_prometheus_remote_write.py
node spec/test_gemma4_depth_chart.js
node --check scripts/gemma4_depth_chart.js
git diff --check
# From the retained artifact directory, without Python -O:
python3 check_same_clock_evidence.py
```


### Demo energy estimate with assumed power factor

At the operator's explicit request, the compact UT61B demo now shows estimated
J/token with PF=1 assumed. Its payload declares
`circuit_energy_estimate={power_factor: 1, basis: "assumed"}`. This is admitted
only for the compact UT61B presentation; invalid factors/bases or other meter
contexts fail validation. Estimated energy is `integral(VA * PF) / N(t)`;
PF=1 leaves the numeric ratio equal to VA s/token. Measured upper-panel VA,
raw VA s integrals, VM series and technical report units remain unchanged.
The selected metric, lower canvas title, cards, tooltip and explanatory copy
identify the estimate or PF assumption. VA/token remains the alternative mode.

DoD: 32 Python report tests and 28 Node chart tests, including the formerly
failing estimate test, preserve raw values and reject missing-context or invalid
PF metadata. All checks passed locally; all 32 Python report tests also passed
on gputer. Browser verification confirmed estimated J/token and PF=1 after
mounting and switching both ratio modes, including the lower canvas title.
Local and gputer compact/full HTML hashes matched. Restore the prior renderer
and HTML for rollback. Refresh if PF becomes measured or its assumed value,
source data, chart contract or renderer changes. This estimate does not resolve
meter calibration or establish measured active circuit energy.
