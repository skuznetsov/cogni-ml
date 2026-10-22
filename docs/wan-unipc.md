# Native Wan Flow-UniPC

The opt-in `ML::Video::WanUniPC` implements the pinned Wan defaults: flow
prediction, order two, `bh2`, `predict_x0`, lower-order final prediction and CFG
4.5. Samples, predictions and solver history stay in F32 Metal buffers. Five
caller-prepared scratch buffers are reused across steps and serial requests.
`encode_step` and `guidance` do not allocate GPU buffers, submit, wait or read
activations. Schedule scalars and the small coefficient system are setup work.

Source: `Wan-Video/Wan2.2@1ea34ff48f87168174e12956e200b1d908b1c5ff`,
`wan/utils/fm_solvers_unipc.py`, SHA256
`0dec8c7ed17f6f2049275c6848113314da6ccec1c8db5bdc89df43c05c6038d9`.
The oracle extracts the actual scheduler class from this hash-checked source.
It replaces unavailable Diffusers configuration plumbing, solves constant small
schedule matrices on CPU, and clones CPU sigma scalar views before MPS use.
The last adapter prevents PyTorch 2.8.0 MPS from reusing a different scalar at
the same underlying storage address. It changes storage ownership, not solver
equations. No neural activation has a CPU fallback.

## Executed gate

On Apple M2 Max, PyTorch 2.8.0 MPS references and the native Metal implementation
passed the unchanged per-value finite gate
`abs(actual-reference) <= 2e-5 + 2e-5*abs(reference)` for 1, 2 and 40 steps.
Each fixture contains 257-element samples and conditional/unconditional model
outputs. Every step compares guidance, converted current prediction, corrected
sample and predicted output. This is 1,028 / 2,056 / 41,120 compared values.
Each focused native invocation reported two examples, zero failures, zero
errors and zero pending cases.

Each invocation also encodes two complete schedules in one serial command,
reusing the same five scratch buffers. Repeated outputs are bit-identical;
the cumulative GPU-buffer allocation count is unchanged during encoding.
Negative checks cover degenerate schedules, duplicate integer timesteps,
mutation through schedule getters, exhausted schedules and unsafe aliases.
The correlated mini-Quadrumvirate review found the bounded algorithm/history/
alias claim ROBUST after the executed gate. It is not independent attestation.

This does not establish end-to-end video generation, full-transformer parity,
concurrent use across command buffers, performance or perceptual quality.
`reset!` is host metadata only: encode the preceding request completely first,
retain all buffers, and preserve GPU command order. Other scheduler modes are
outside this API.

### Reduced-step gate, 2026-10-22

Fresh upstream MPS fixtures and a fresh native release build pass at 8, 12,
16 and 40 steps with the same finite per-value tolerance. Each invocation
reports two examples with zero failures, errors or pending cases. Every
schedule is recomputed from its requested count; all sigmas and integer
timesteps match the matching upstream fixture exactly. The repeated schedule
reuses scratch without new encode-time GPU allocations and is bit-identical.

| Steps | Compared intermediate values | Worst error / allowed error |
| --- | --- | --- |
| 8 | 8,224 | 0.02269 |
| 12 | 12,336 | 0.02386 |
| 16 | 16,448 | 0.03238 |
| 40 | 41,120 | 0.46278 |

These 78,128 comparisons use supplied conditional/unconditional predictions;
they establish solver parity, not denoiser accuracy or video quality. Native
40-step inference remains the relative quality control for the separate
short-video experiment. Existing full-model upstream drift is unaffected.

Set `WAN_UNIPC_EXPECT_STEPS` in a comparison matrix to reject a stale or
mislabelled fixture. An eight-step fixture with expected count 12 was rejected
before Metal initialization (exit 1); the four matching counts passed. All
oracle, build and native runs used the cache-aware memory guard with the
unchanged 15% floor and 64 MiB maximum swap growth; logs contain clean terminal
events. These are sampled limits, not continuous peak-memory guarantees.

Evidence: `/opt/mldata_ssd/cogni-ml/video-scene-20261019/evidence/wan-steps-short-20261022`.
The report `scheduler-gate.json` pins fixtures, actual traces, tested sources,
binary and guard logs; SHA-256
`52e98fec4ac9bd3e85b115aaa5058e4dbf9f109f7f7168ee6775269bb5073701`.
The logs do not capture invocation arguments/environment or bind the executable
hash before launch; saved trace parity is stronger evidence than the receipt's
invocation lineage. An independent recomparison of those saved arrays found
zero violations; it was a same-lineage review, not a second GPU execution.
Refresh by rebuilding the spec and regenerating each oracle fixture at its
declared count. For each count, execute the spec through
`examples/support/wan_memory_guard.py --memory-accounting=file-cache
--max-footprint-mib=2048 --max-swap-growth-mib=64 --timeout-sec=90 --log=FILE
-- BINARY`, with both `WAN_UNIPC_FIXTURE` and `WAN_UNIPC_EXPECT_STEPS` set.
No model generation was timed by this gate.

## Reproduce

Use a pinned source checkout or a directory containing the unmodified source as
`wan_utils_fm_solvers_unipc.py`. An MPS-capable PyTorch 2.8.0 environment is
required. Store fixtures and build products outside the repository.

```sh
PYTORCH_ENABLE_MPS_FALLBACK=0 COGNI_RUN_SAFE_MIN_FREE_PCT=15 \
  scripts/run_safe.sh "$ORACLE_PYTHON" 60 2048 \
  spec/support/wan_unipc_reference.py --upstream "$WAN_SOURCE" \
  --steps 40 --output "$EVIDENCE/unipc40.json"

crystal build spec/wan_unipc_spec.cr -o "$BUILD/wan_unipc_spec" \
  --link-flags="$BUILD/bridge.o -framework Metal -framework Foundation -lc++ -fuse-ld=/usr/bin/ld"

WAN_UNIPC_FIXTURE="$EVIDENCE/unipc40.json" COGNI_RUN_SAFE_MIN_FREE_PCT=15 \
  scripts/run_safe.sh "$BUILD/wan_unipc_spec" 60 2048
```

Repeat the oracle and native commands with 1 and 2 steps. A missing fixture is
pending, not a successful oracle gate. The run above uses a 15% free-memory stop
floor and serial GPU execution; no Studio or service is involved.

Local evidence directory:
`/opt/mldata_ssd/cogni-ml/video-scene-20261019/evidence/wan-full`.
Final logs: `unipc{1,2,40}-native-final.log`; build log: `unipc-build.log`.
Fixture SHA256 digests:

| Steps | SHA256 |
| --- | --- |
| 1 | `bb10b52441b895410f29b5ec4625fe740d81b33e5125e4774e39bfa5de53d4e2` |
| 2 | `5fdca309381f29fb9979a4628d11b32e0d58bb56588d409fedfd8ba10f84b6d1` |
| 40 | `2bdfc240e5d07f326ff9c4ca6c0665a6b01e5b87850f874ff3de5dcb583f6751` |

Refresh this gate after changes to the native/oracle source, upstream revision,
solver configuration, arithmetic, tolerance, device or runtime. Missing or
changed fixtures invalidate the local receipt; hashes alone do not attest a
build or its execution.
