# Complete DRAMA-3 decode capture

The fixed short request completed under Shader Timeline with all 122 Metal
commands present in the native timing stream. Eight complete requests on the
same rebuilt binary preserve full frame codes, sampler tape and WAV against
the saved baseline. All five warm requests preserve allocation counts/bytes,
command/wait counts and readback ledgers. This qualifies a diagnostic capture;
it does not establish a speedup or strict upstream compatibility.

## Request and controls

The input is `Hello, this is a voice test.`, seed 1, max 64 frames, temperature 0.8,
top-k30 and top-p0.9. It stops on EOS after 60 frames and produces 122,880 samples
at 44,100 Hz (2.786394558s). WAV SHA256 is
`38f2530ed1c90f0caba45e222432909ac5d92cc3633e4c34a21fbe39d693e07d`.
The pinned checkpoint is `cguldogan/drama-3-mlx-q4` revision
`5331ab509359c0e4a7e353a19e255ddf3e8152a8`.

The sequence uses three fresh workers: profiler-off cold/warm; timeline-enabled
cold/warm, one captured warm request, another untraced warm request; profiler-off
cold/warm. Warm wall times are:

| Arm | Instruments attached | Wall seconds |
| --- | --- | ---: |
| Off before | No | 26.733517 |
| Native timeline before | No | 27.293311 |
| Complete Shader Timeline | Yes | 29.169552 |
| Native timeline after | No | 27.688088 |
| Off after | No | 23.661635 |

The two profiler-off observations give RTF 8.492–9.594 on this active host.
They are a bracket, not a causal estimate of profiler overhead or a stable
throughput distribution. Shader Timeline wall time is not production throughput.

Every request submits, completes and waits on 122 commands, with graph depth 2,
max pending 2 and final pending 0. Existing control reads are 496 B; terminal reads
are 494,160 B. Each warm request records 135,590 allocations requesting
4,079,562,724 B cumulatively. This is not live resident memory. Production
activations are not copied to the host by this capture; timestamps are read
after existing waits. No inference kernel, synchronization or default changes.

## Native command windows

The fixed request has 29 prefill commands, one initial selector/boundary command
(including the prompt-window dispatch), 60 mixed
backbone/tower decode commands and 32 DAC commands. GPU execution window unions
for the timeline-enabled warm requests are:

| Phase | Before trace | Captured | After trace |
| --- | ---: | ---: | ---: |
| Prefill | 4.623050s | 4.567721s | 4.559935s |
| Decode | 11.226656s | 11.083234s | 11.072031s |
| DAC | 2.688797s | 2.634473s | 2.623002s |

These are unions of complete command execution windows, not per-kernel runtime.
Different phases can overlap; do not add phase unions as a request-time model.
The first prefill wait includes 6.325/8.579/7.930 s before GPU execution in these
three requests, while its GPU command takes about 0.168 s. This locates the delay
before execution; it does not identify page-in, resource registration, driver
work or another process as its cause. The existing diagnostic kernelStart/End
timestamps describe CPU-side driver scheduling, not shader runtime.

## Rejected shader attribution and next discriminator

The selected exports contain 122 unique target-process command submissions.
Their order-based association with native commands 244–365 yields 60 decode
buffers, and the decode Active/Compute window union is 11.083234335s, within
86ns of the native timestamp union. This supports the command-window analysis;
native and xctrace events still lack a shared command identity key.

It does not validate the shader-family distribution. Shader Timeline intervals
labelled `dr3c_conv1d_coalesced` and `dr3c_layer_norm` overlap decode and even
prefill windows. This request has no voice reference, and the source calls DAC
only after generation (`studio/engine.cr`, `Engine#generate`); those `dr3c_*`
dispatches are confined to codec implementations. The intervals therefore
provide a counterexample to phase-specific family attribution. This is not
evidence that DAC executed during decode. The cause in the sample semantics or
trace association remains unresolved. Nanosecond units, complete command
counts and overlap-safe interval arithmetic do not resolve it.

The v2 analyzer records 5,033 such rows across 16 labels and suppresses both
per-family and per-shader phase tables (`status=phase_attribution_rejected`).
For example, `dr3c_conv1d_coalesced (1)` starts at trace time 14.463539291s
with a 214.625µs interval entirely inside the mapped decode windows. The
source-order contradiction is retained as evidence, not explained away.

Do not use the earlier `shader-coverage-v1.json` distribution as a runtime
ranking, bounds on family cost, or a basis for changing defaults. The next
performance target remains decode, whose native command windows occupy about
11.1s for 60 frames. The next discriminator is a diagnostic-only replay of
captured decode operands through homogeneous production-kernel families, with
batched command drains, existing GPU timestamps and exact GPU output checks.
An isolated generation-versus-DAC capture can separately test shader-label
attribution. Neither probe alone promotes a candidate: unchanged upstream
metrics and profiler-off complete-WAV pairs are still required.

## Numerical and reliability scope

The rebuilt binary includes the separately committed watchdog lifetime fix
`6ab91a4a`; see [watchdog qualification](metal-watchdog-exit.md). All three workers
in this capture exit 0 after their complete request graphs drain.

The GPU-only tower replay reuses the pinned 87-row MLX fixture. All 2,025 raw
comparison records, tolerances, command graph and readbacks equal the saved
pre-fix report. All 675 native off/on sites remain exact and the first decode
frame equals MLX. The same three late residual layer 8 sites fail the unchanged MLX
tolerances in both native arms; strict upstream compatibility remains false.
This is a bounded no-regression replay at prefill row 85/decode row 0, not a
complete backbone/DAC upstream gate. Its compact GPU metrics cross only the
terminal diagnostic boundary (40,500 B), with one diagnostic command/wait.

## Reproduction and evidence

Use `scripts/drama3/capture_decode_profile.py` under `scripts/run_safe.sh` with
an explicit timeout of 900 seconds or longer, the pinned model, a saved complete baseline,
and the installed Metal System Trace + Shader Timeline template. The script
pins inference controls, launches an allowlisted environment, checks complete
trajectories, exports selected tables, sanitizes the trace TOC and removes its
owned raw trace after the recorder is reaped. If reaping cannot be confirmed,
the current harness retains the trace and records the unresolved recorder PID.
Process-group SIGKILL can bypass Python cleanup, so verify/remove that
experiment's owned trace after its children stop. Never inspect an unsanitized
TOC or raw capture.

Artifacts are under
`/opt/mldata_ssd/cogni-ml/drama3-decode-profile-20261017/`.
`commands.json` binds the actual capture to its executed harness SHA256
`421a318a1825fc67cab3aded3c0e4aea8292361b13cea0d80174105058d5014e`,
retained as `capture-harness-executed-v1.py`. The committed harness additionally
bounds partial recorder-start lines and preserves a live recorder's trace;
those later guards are CPU-tested, not claimed as exercised by this capture.
The observed recorder start succeeded,
and recording ended naturally before selected exports. Raw trace is absent.

The scoped index is `.landmark/evidence/drama3-complete-decode-profile.json`.
`profile-command-receipts.json` pins the five successful build/capture/replay
commands. `profile-validation-v1.json` records 10 capture guards and 11 analyzer
guards passing, native timeline analysis and the unchanged upstream comparison.
Root replay of the full selected exports equals `shader-coverage-v2.json`.
The artifact check matches 89 recorded runtime source files, all selected export
digests and all eight WAVs. These checks and correlated mini-Quad review support
the bounded capture result; they do not establish independent replication or
strict upstream compatibility.

Refresh on source, compiler/runtime, model, fixture, sampler input, tolerance,
queue, device/OS or artifact changes; timing also decays with host load. Retain
existing production selectors and numerical tolerances. A future kernel
candidate needs captured-input GPU parity/timing, upstream comparisons and
profiler-off full-request pairs before promotion.
