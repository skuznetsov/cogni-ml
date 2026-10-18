# DRAMA-3 prefill memory investigation

One bounded, single-model MLX prefill completed all 269 rows without
reproducing multi-GiB per-row growth. Most allocation occurred while loading
the F32 core and codec. Cache growth matched the source prediction. The cause
of the older interrupted incident remains **IN_PROGRESS**: this run does not
prove duplicate Studio processes caused it, or exclude retention in later
generation and repeated requests. No causal model/cache/kernel fix is claimed.
The diagnostic process exited; Studio was not started.

## Observed boundary and source identity

The private fixed-coarse-code probe stopped after reporting entry into its
first 269-row `prefill_A`. It produced no generation/WAV result and had no
per-row or per-stage memory counters. Its log does not identify the row at
which memory grew. Preserve the aborted receipt and original script under
`/opt/mldata_ssd/cogni-ml/drama3-voice-diagnosis-20261018/`.

This audit uses native base `60a28599`, MLX-audio
`e7488935ec2ebfafd38ef2b58e290d79f3d05dc7`, and checkpoint
`cguldogan/drama-3-mlx-q4@5331ab509359c0e4a7e353a19e255ddf3e8152a8`.
The preparatory source audit used only source, config/index JSON, and
safetensors headers. The separately authorized single-model runtime follow-up
is recorded below. Neither phase started Studio.

## Static discriminators

`replay_mlx_generation.py:164` promotes floating core weights to F32 and keeps
packed U32 tensors. Header-only accounting gives:

| Core inventory | Bytes | GiB |
| --- | ---: | ---: |
| Stored indexed tensors | 23,993,186,560 | 22.35 |
| Same tensors after F32 floating promotion | 31,363,559,936 | 29.21 |
| Packed U32 payload, included above | 16,622,813,184 | 15.48 |

There are 1,167 indexed core tensors. The codec file separately contains
1,572,013,576 bytes of F32 tensors and 301,989,888 bytes of boolean masks. File
bytes and conversion arithmetic do not establish physical residency, private
process memory, or an allocation peak. Lazy initialization, mapped pages,
conversion intermediates, and allocator reservations remain separate terms.

The pinned upstream sources are under
`mlx_audio/tts/models/fish_omni/` and `mlx_audio/lm/models/`:

- `fish_omni.py:168-188` replaces the current hidden value and evaluates hidden
  plus every cache state on each one-row prefill iteration. `tower.py:260-282`
  processes the corresponding shifted pairs without an output accumulation
  list. Generation capture lists are not reached at the observed failure.
- `cache.py:104-176` grows KV backing in 256-position blocks. There are 10
  backbone full-attention caches and 13 tower caches, each with two F32 arrays
  of shape `[1, 2, capacity, 256]`. One arm therefore owns 23 MiB at capacity
  256 and 46 MiB at capacity 512. Row 257 adds **23 MiB**, with a bounded
  old/new-buffer resize transient. At 269 prompt rows the tower has 268 pairs.
- `qwen3_5.py:294-344` retains fixed recurrent states and convolution tails in
  the 30 linear-attention layers: 60 MiB plus about 2.81 MiB. The complete
  predictable F32 cache array total at capacity 512 is about **108.81 MiB**.
- `qwen3_5.py:218-234` and `switch_layers.py:76-91,177-200` select eight experts
  and use packed `gather_qmm`; this Python path does not explicitly dequantize
  every expert. Kernel scratch and allocator behavior are not ruled out.

Native Studio has a different ownership path. `studio/server.cr` and
`studio/backend.cr` own one serial consumer and one persistent worker per App;
`examples/drama3_studio.cr` runs GC after successful requests and exits the
worker after an error. Static converted weights intentionally persist per
worker Context. `studio/engine.cr:51-64` bounds cache capacity by
`prompt_rows + max_frames - 1 <= 4096`, rather than the checkpoint's 262144.
The config-derived maximum recurrent/KV allocation is **432 MiB** at capacity
4096, excluding weights and other scratch. Native reference encoding can
create 108 MiB early activations and multiple fresh residual intermediates;
it occurs before prefill. Neither is a demonstrated cause of the separate
upstream probe's growth. Two live engine processes may overlap working sets;
the post-shutdown system-free percentage cannot assign per-process ownership.

## Diagnostic preparation and its limits

`scripts/drama3/mlx_memory.py --model-dir <local-checkpoint>` performs only the
header inventory above. It never imports MLX, Torch, or model modules and
never reads tensor payloads. The separate `MemoryTrace` helper is opt-in for an
isolated, single-threaded pinned MLX diagnostic:

```python
# Future caller only: mx/model/prompt have already been prepared by that probe.
# Open a fresh JSONL file on the external SSD with mode "x".
fish_omni.PREFILL_CHUNK_SIZE = 1
trace = MemoryTrace(mx, stream, budget_bytes=40 * 1024**3)
trace.snapshot("core_loaded")
hidden = trace.prefill(model, prompt, backbone_cache, tower_cache, stage="prefill_A")
```

Record snapshots before model construction, after construction, after
quantization, after core load, after the codec hook, and after the existing
parameter evaluation. `prefill` wraps the existing `mx.eval` call and records
each return, including rows 1, 256, 257, and 269. It delegates the original
arguments exactly once, returns the original result, and restores the function
in `finally`. No additional explicit GPU evaluation/synchronization, tensor
readback, cache clear, precision change, or allocator-policy change is added.
The observer has now been exercised in the bounded runtime follow-up below.

Records contain active/cached/peak MLX counters, current process RSS, and cache
backing shapes/dtypes/bytes. They contain no activations or audio. Cache metadata
uses backing arrays directly; `KVCache.state` would crop to the valid prefix
and hide spare capacity. Records are flushed immediately and no array handles
are retained by the recorder. Peak MLX memory is a process high-water counter,
not a per-row peak. RSS and MLX counters are separately sampled signals;
neither is an exact unified-memory physical-footprint measurement. The RSS
subprocess and JSON writes add host overhead, so traced timings cannot support
a performance claim.

The explicit budget checks the maximum of process RSS, active-plus-cached
MLX bytes, and cumulative MLX peak-active bytes before prefill and after each
existing evaluation. The tracer never resets the peak counter. A CPU falsifier
exposed that the original guard recorded the peak but omitted it from its
decision; the guard now includes a transient peak even after active memory
falls. A violation flushes the record and stops before encoding another row.
It cannot interrupt a peak inside one evaluation or before a snapshot, and
cannot protect against other processes. The external `run_safe` system-pressure
guard and user-authorized 15% floor still apply. Counter/telemetry errors fail
closed; original evaluation errors propagate. No budget value grants permission
to restart a model.

The `mx.eval` hook is process-global. Use only in the isolated single-threaded
probe with `PREFILL_CHUNK_SIZE=1`; do not install it in Studio or a concurrent
runner. The helper rejects the upstream default of 128, or an unrecognized
method, before calling prefill or installing the hook. It checks the bound
method's schedule global without changing upstream settings. A successful
return also requires one observed evaluation per prompt row.
This count is a drift guard, not proof of source identity: the caller must
still verify the pinned clean upstream revision and record script hashes.

## Verification and continuation

Executed without importing the GPU/model runtime:

```text
python3 -B spec/drama3_mlx_memory_spec.py
  11 tests pass
python3 -B scripts/drama3/mlx_memory.py --model-dir <local-checkpoint>
  1167 tensors; 31,363,559,936 F32-core bytes
git diff --check
  clean
```

CPU falsifiers cover packed/promotion arithmetic, bad index paths and missing
tensors, backing capacity versus valid prefix, one-for-one evaluation
delegation, chunk-schedule rejection before model entry, pre-entry and per-row
budget failure, released-but-over-budget peak detection, original-error propagation,
hook restoration, schedule drift, absence of retained array handles, and no
MLX/Torch/model import. These establish diagnostic logic, not GPU correctness,
physical memory causality, or an upstream numerical gate. The runtime result
below adds a scoped upstream self-consistency check. Existing native/upstream
tolerances are unchanged; no new independent full native gate was run.

## Bounded single-model runtime follow-up, 2026-10-18

The user authorized one first-prefill measurement, without Studio. The private
harness reused the aborted probe's exact text and saved `[10,178]` voice codes;
it constructed the same `[269,11]` prompt. Inputs/audio remain on the external
SSD and are not in Git. It loaded one model, including the original codec
hook, through the same F32 core loader; packed U32 weights stayed packed.
Pinned upstream was clean at `e7488935ec2ebfafd38ef2b58e290d79f3d05dc7`.
Runtime: MLX/MLX-Metal 0.31.1, NumPy 1.26.4, Torch 2.6.0; 64 GiB host.
Known DRAMA/MLX runtime scans and Studio ports 8787/8788 were clear before
launch and again afterward. Unrelated processes were untouched.

```text
PYTHONOPTIMIZE=0 PYTHONDONTWRITEBYTECODE=1 TMPDIR=<external-output>/tmp \
COGNI_RUN_SAFE_MIN_FREE_PCT=15 RUN_SAFE_PASSTHROUGH_STDIO=1 \
scripts/run_safe.sh <pinned-venv-python> 900 40960 \
  <external-output>/bounded_prefill.py
```

The child also rejects optimized Python, source drift and a matching second
runtime before model import. Its observer used a 40 GiB budget; the external
runner used the same RSS-tree limit, a 900-second timeout and the 15% floor.
All 285 snapshots passed their budget checks. The process exited 0 and both
Studio ports remained closed. System free memory was 82% before and after;
sparse external polls reached 27%. That is not a continuously sampled minimum;
the guard log reports no floor-triggered stop.

| Stage / completed row | MLX active MiB | MLX cached MiB | Logical cache backing MiB |
| --- | ---: | ---: | ---: |
| Constructed / quantized | 0.01 | <0.01 | 0 |
| Core shard 1 / 2 / 3 | 6,411.62 / 12,724.24 / 19,061.48 | 0 | 0 |
| Core shard 4 / 5 | 25,566.10 / 29,910.63 | 0 | 0 |
| Codec hook / parameter eval | 31,989.33 | 255.77 | 0 |
| Prefill row 1 | 32,076.34 | 254.67 | 85.8125 |
| Prefill row 2 | 32,078.97 | 252.04 | 85.8125 |
| Prefill row 256 | 32,079.04 | 251.98 | 85.8125 |
| Prefill row 257 | 32,104.06 | 253.83 | 108.8125 |
| Prefill row 268, before checkpoint copy | 32,102.05 | 237.96 | 108.8125 |
| Prefill row 269, with diagnostic checkpoint | 32,182.65 | 230.17 | 108.8125 + 82.8125 checkpoint |

The core's measured active allocation is 29.2096 GiB; the codec hook raises
it to 31.2396 GiB. Construction and quantization were lazy. Rows 2 through 256
range from 31.3235 to 31.3273 GiB active; row 256 is only 76,812 bytes above row
2. Across those rows, active-plus-cached spans only 0.375 MiB. RSS still drifts
by about 7.4 MiB; this is not evidence of zero host retention. The exact 23 MiB
logical cache step at row 257 matches the source prediction.
Active/cached samples have small asynchronous lifetime fluctuations; a cache
metadata change must not be equated to the difference between two physical
footprint samples.

For numerical checking only, the harness deep-copied 82.8125 MiB of logical
backbone backing after row 268. The copy is lazy; row 269 includes this retained
diagnostic state. Its physical increment was not isolated with a no-clone
control. An immediate post-copy budget sample was also taken. There is no
second full prefill, teacher cache, generation, sampler, or DAC decode.
Peak-active was 31.4318 GiB through prefill, and
31.4894 GiB including tail replay and terminal GPU reductions. The largest
sampled active-plus-cached value was 31.6531 GiB, below the 40 GiB budget.
Sampled RSS peaked at 3.4772 GiB during loading and stayed near 1.28 GiB in
prefill; it clearly does not represent all Metal-buffer ownership. Do not add
RSS and MLX counters or promote them to exact physical residency.

The original 269 per-row `mx.eval` calls were delegated unchanged. Stage/row
sampling adds no explicit GPU eval, wait or activation readback; it does add
host RSS queries and JSON writes. The final check reads one five-number GPU
receipt. Instrumented prefill took 33.554 seconds, while the trace from runtime
import through terminal checks spans 82.087 seconds. These are diagnostic
timings, not a speed benchmark. The runner's `after ~56s` is an approximate loop
counter and must not replace the monotonic clock.

### Upstream check and remaining falsifier

The prompt built by GPU splicing matched official `build_prompt` exactly on
GPU. After the original prefill, `_backbone_step` replayed the final row against
the copied pre-final backbone cache. Upstream deliberately skips tower prefill
on this last row, so replay also skips it. Endpoint hidden and all backbone
cache arrays were finite, with maximum absolute replay difference **0.0**.
A hidden perturbation of +1 produced difference 1.0, and a small array
deepcopy/source-mutation control passed. This does not stress mutation of every
cache layout, check tower finiteness, or compare all intermediate rows.

No saved golden exists for this exact 269-row prompt. The existing private
312-row controls use a different prompt and were rejected as a reference.
The successful check establishes matched input and same-upstream tail replay
consistency, not independent native/upstream parity. The prior strict native
layer-39 discrepancy remains open; tolerances were not widened.

Root execution and correlated mini-Quad review support **ROBUST** only for the
bounded single-prefill observation and this numerical scope. Stable rows make
intrinsic multi-GiB rowwise retention an unsupported explanation for this run.
Concurrent full models remain a plausible historical contributor, but only
one arm was measured and the old incident had no counters. Repeated requests,
generation retention and native Studio lifetimes remain untested. Inspect
single-worker ownership and restart/cleanup boundaries before any separately
authorized Studio restart; do not infer that the full voice experiment is safe
from this first-prefill result alone.

The source/artifact digest index and exact selected counters are in
`.landmark/evidence/drama3-prefill-memory.json`; full private trace, harness and
receipts are under
`/opt/mldata_ssd/cogni-ml/drama3-prefill-memory-20261018-2246/`.
Refresh after source, checkpoint contents, precision, chunk schedule,
MLX/Metal/driver, device, input or concurrent-process changes. Weight payloads
were loaded as-is; this run recorded config/index hashes rather than rehashing
all tensor payloads. The bounded runtime grant is consumed; model and Studio
remain stopped at handoff.
