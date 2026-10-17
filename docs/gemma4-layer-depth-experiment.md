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

| Batch | Decoder blocks | Stored parameters | Seconds | GPU W | GPU J/request | Fluke VA s/request |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Full before | 48 | 11,907,350,576 | 5.658 | 159.7 | 903.8 | 1328.7 |
| Prefix 32 | 32 | 8,267,880,992 | 3.970 | 161.5 | 640.9 | 914.9 |
| Prefix 16 | 16 | 4,628,411,408 | 2.316 | 144.8 | 335.1 | 468.3 |
| Full after | 48 | 11,907,350,576 | 5.635 | 163.5 | 921.4 | 1326.3 |

Relative to the initial full-model batch, the 16-block batch used about 63%
less measured GPU energy and took about 59% less time. The 32-block batch
reduced energy while its mean watts stayed near the full model. This refutes
using stored weight count as a proportional wattage estimator in this setup.
The 16-block trial means ranged from 134.9 to 154.0 W, so the mean-power effect
needs longer, isolated windows before a precise claim. Sensor averaging,
shared GPU activity and short current windows remain limits. This is fixed
synthetic token work; it does not establish quality-preserving energy savings.

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
