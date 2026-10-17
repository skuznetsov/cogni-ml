# Gemma 4 energy profiles

cogni-ml controls two independent generation allowances and compares the whole
measured circuit over each request window. On Linux this entrypoint uses a local
llama.cpp server for full-depth Gemma 4 inference. It does not turn the existing
CUDA layer probes into a generation backend. NVIDIA is the tested backend;
this feature does not implement Gemma 4 generation on Radeon.

| Profile | Reasoning token cap | Answer token cap |
| --- | ---: | ---: |
| Eco | 0 | 256 |
| Balanced | 256 | 512 |
| Quality | 1024 | 1024 |

These are starting policies, not established quality or energy rankings. Eco
omits the thinking instruction. The other profiles generate a bounded native
Gemma 4 thought channel, close it, and issue a separately bounded answer
continuation. Closing reasoning never consumes the answer allowance. The report
contains generated counts including control tokens, forced closure, answer
truncation, and the answer for human quality review. Private reasoning stays in
memory and is not written to the report.

## Run

Use Crystal with OpenSSL development metadata (`pkg-config`) available. Start a
compatible local llama-server with the complete Gemma 4 GGUF, enough context,
and one generation slot. For reproducible initial comparisons, disable prompt
caching and continuous batching:

```sh
llama-server --model /models/gemma-4-12B-it-Q4_K_M.gguf \
  --host 127.0.0.1 --port 8080 --ctx-size 8192 --gpu-layers 999 \
  --parallel 1 --no-cache-prompt --cache-reuse 0 --no-cont-batching
crystal build bin/gemma4_energy.cr -o build/gemma4_energy
build/gemma4_energy --dry-run
build/gemma4_energy --profiles eco,balanced,quality \
  --prompt 'What is 17 times 23? Give a short answer.' \
  --server-url http://127.0.0.1:8080 \
  --model-file /models/gemma-4-12B-it-Q4_K_M.gguf \
  --vm-url http://127.0.0.1:8428 \
  --vm-selector 'fluke_current_amperes{circuit="gputer",mode="ac"}' \
  --require-measurement --output comparison.json
python3 scripts/gemma4_energy_report.py comparison.json --output comparison.html
```

`--model-file` is optional; it hashes the actual loaded path and requires the
controller and server to share a filesystem. Without it, model provenance is
the server's reported path and tokenizer checks. Hashing and model loading are
outside the measured request window. The llama-server is trusted to run the
specified full model and honor generation limits; this is not hardware
attestation.

The existing entrypoint also accepts `--energy-profile balanced` or
`--compare-energy-profiles` with these energy options. Its previous Metal
profiler path remains available without those flags.

```sh
crystal run bin/gemma4_generate.cr -- --energy-profile balanced \
  --reasoning-tokens 128 --answer-tokens 256 --prompt 'Explain this result.'
```

Budget overrides apply to every selected profile. `--repeats 3` rotates the
profile order each round. Optional `--warmups N` makes short unmeasured Eco
requests first. Defaults are deterministic greedy decoding, seed 1, no prompt
cache, and zero warmups. Use the same prompt, model, hardware load, and settings;
allow temperature to settle or randomize external workload separately when
making stronger experimental claims.

## Measurement contract

The BLE collector writes `fluke_current_amperes{mode="ac",...}` and companion
`fluke_measurement_valid` samples to VictoriaMetrics. Include stable circuit
and device labels in `--vm-selector` if more than one meter is present. Exactly
one series with matching validity labels must cover each entire window.

The controller reads raw `/api/v1/export` samples, merges split rows, and
integrates by trapezoids with clipped boundary interpolation. It requires a
sample on either side and rejects gaps over `--max-gap-ms` (default 5000).
Validity uses held status, not interpolation. A missing or invalid measurement
is **unavailable**, never zero. `--measurement-wait 8` allows collector delivery
and a trailing sample. Increase it for buffering or use `--measure-only` later:

```sh
build/gemma4_energy --measure-only comparison.json \
  --vm-selector 'fluke_current_amperes{circuit="gputer",mode="ac"}' \
  --output comparison-measured.json --require-measurement
```

Imported reports must have complete, consistent phase accounting and stop
outcomes. They are marked `imported_unverified`: structural validation cannot
attest that the recorded inference actually happened.

Request timing uses monotonic elapsed time and wall-clock epoch windows for
cross-host correlation. Synchronize collector and server clocks. Measured cost
includes both HTTP completions, request overhead, and the second prefill; it
excludes model load, warmups, model hashing, and the delivery wait. Whole-circuit
readings include other CPU/GPU activity and idle consumption. Sampling resolution
limits what short request curves can establish.

AC RMS current times an assumed 116 V is **apparent power in VA**, with integral
in **VA s**. The default report and graph use those units. They do not claim
active watts or joules. `--assumed-power-factor PF` yields explicitly estimated
joules. For an actual active-power sensor use `--measurement-kind active_power`
with an explicitly selected source metric ending in `_watts`; the source must
really measure active watts. Its integral is joules. A metric name alone does
not calibrate an instrument.

Remote VM authentication uses `COGNI_VM_BEARER_TOKEN` or `--vm-token-file`.
A bearer token requires HTTPS unless VM is on loopback. Keep the collector and
inference server inside the intended perimeter. For meters connected to a Mac,
deliver its remote-write endpoint through an authorized private connection.
If SSH disables TCP forwarding, an SSH command transport or offline delivery
can still carry samples; missing data remains unavailable until delivered.

Exit status is 1 for a failed inference and 3 when `--require-measurement` lacks
coverage. A token-limited answer remains explicitly truncated even if the
inference request succeeded. Reports never automatically approve answer quality;
inspect answers before comparing cost for useful work.

## Checks

```sh
crystal spec spec/gemma4_energy_profile_spec.cr \
  spec/gemma4_energy_runner_spec.cr spec/gemma4_energy_cli_spec.cr \
  spec/victoria_metrics_energy_spec.cr
PYTHONDONTWRITEBYTECODE=1 python3 -m unittest spec.test_gemma4_energy_report
```

The real VM HTTP fixture is opt-in. Point `COGNI_TEST_VM_URL` at a disposable VM
instance before running `spec/victoria_metrics_energy_http_spec.cr`; its synthetic
`cogni_energy_fixture_watts` series tests import/export arithmetic and is never
physical Fluke evidence. See [frontier and evidence](gemma4-energy-profiles-frontier.md)
for the admitted scope and current hardware validation.

On the measured gputer, the built controller is
`/build/persisten/cogni-gemma4-energy/tools/gemma4_energy`; the loaded GGUF is
under the same root in `models/`, and real reports are in `reports/`. The local
services are already running on ports 8080 and 8428. Use the controller directly
over SSH with the selector above. Live capture still needs the Mac BLE collector
to deliver samples; old measurements can be re-read until VM retention expires.
