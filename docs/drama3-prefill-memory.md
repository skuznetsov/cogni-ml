# DRAMA-3 prefill memory investigation

The cause of the interrupted MLX probe's memory growth remains **IN_PROGRESS**.
Source inspection rejects KV-cache growth alone as a multi-GiB explanation.
No causal model, cache, precision, or kernel fix has been established. The new
diagnostic is CPU-tested preparation for a future measurement, not a leak fix.
The model and Studio remain stopped by user instruction.

## Observed boundary and source identity

The private fixed-coarse-code probe stopped after reporting entry into its
first 269-row `prefill_A`. It produced no generation/WAV result and had no
per-row or per-stage memory counters. Its log does not identify the row at
which memory grew. Preserve the aborted receipt and original script under
`/opt/mldata_ssd/cogni-ml/drama3-voice-diagnosis-20261018/`.

This audit uses native base `60a28599`, MLX-audio
`e7488935ec2ebfafd38ef2b58e290d79f3d05dc7`, and checkpoint
`cguldogan/drama-3-mlx-q4@5331ab509359c0e4a7e353a19e255ddf3e8152a8`.
Only source, config/index JSON, and safetensors headers were inspected. No MLX
import, model/Studio launch, weight tensor load, or GPU experiment was used for
this investigation.

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
Actual MLX runtime behavior of the observer has not been exercised.

Records contain active/cached/peak MLX counters, current process RSS, and cache
backing shapes/dtypes/bytes. They contain no activations or audio. Cache metadata
uses backing arrays directly; `KVCache.state` would crop to the valid prefix
and hide spare capacity. Records are flushed immediately and no array handles
are retained by the recorder. Peak MLX memory is a process high-water counter,
not a per-row peak. RSS and MLX counters are separately sampled signals;
neither is an exact unified-memory physical-footprint measurement. The RSS
subprocess and JSON writes add host overhead, so traced timings cannot support
a performance claim.

The explicit budget checks the larger of process RSS and active-plus-cached
MLX bytes before prefill and after each existing evaluation. A budget violation
flushes the offending record and stops before encoding another row. It cannot
prevent a peak inside one evaluation or before a snapshot, and cannot protect
against other processes. The normal external `run_safe` system-pressure guard
and user-authorized 15% floor still apply to any future run. Counter/telemetry
errors fail closed and abort the diagnostic; original evaluation errors are
propagated. No budget value grants permission to restart a model.

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
  10 tests pass
python3 -B scripts/drama3/mlx_memory.py --model-dir <local-checkpoint>
  1167 tensors; 31,363,559,936 F32-core bytes
git diff --check
  clean
```

CPU falsifiers cover packed/promotion arithmetic, bad index paths and missing
tensors, backing capacity versus valid prefix, one-for-one evaluation
delegation, chunk-schedule rejection before model entry, pre-entry and per-row
budget failure, original-error propagation,
hook restoration, schedule drift, absence of retained array handles, and no
MLX/Torch/model import. These establish diagnostic logic, not GPU correctness,
physical memory causality, or an upstream numerical gate. Tolerances are
unchanged; no new upstream numerical result exists.

Next discriminator, only after runtime investigation is authorized: use a fresh
external output directory, keep Studio stopped, and measure the first prefill
with an explicit budget and stage/row telemetry. Fixed recurrent-state
initialization and one 23 MiB KV step predict the cache portion. Growth beyond
those terms should be localized to initial weight residency, row temporaries,
or allocator reservation before changing kernels. Do not repeat the complete
two-arm voice experiment as the first memory probe. Refresh this audit after
source, checkpoint, precision, chunk schedule, MLX/driver, or device changes.
