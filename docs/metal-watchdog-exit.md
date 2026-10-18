# Metal watchdog state at process exit

The command watchdog could lock its file-static mutex after C++ static teardown
had destroyed that mutex. A DRAMA worker aborted at exit after its final complete
request, with `std::system_error: mutex lock failed: Invalid argument` in the
watchdog dispatch callback. Its recorded bridge SHA256 matches the source at
`034354781281a04cc1858d7ce3d45e267530e7be`.

`spec/metal_watchdog_exit_probe.mm` reproduces that ordering without a model. It
completes one empty real Metal command, checks that the wait registry is empty,
and retains the actual watchdog dispatch source. A sentinel constructed before
the bridge globals wakes the real callback after those globals are destroyed.
The same final probe source gives exit134 and the incident's invalid-mutex
exception against the recorded bridge; the patched bridge gives exit0 and
`WATCHDOG_LATE_CALLBACK_COMPLETED`. This establishes static teardown as a
sufficient cause matching the incident, without excluding another fault in the
original inference process.

The fix places the mutex and record vector together in one intentionally
process-lifetime `GSCommandWaitState`. Callback, registration and unregistration
use the same mutex. The state is allocated once on first use and is not destroyed
while dispatch callbacks can still execute. This retains the small registry and
its vector capacity until process exit. No FFI, kernel, deadline, inference
submission, wait or readback changes.

The bounded guards cover normal and late real-bridge exit, disabled watchdog,
concurrent registration and removal with the callback observing both active
records, and the unchanged terminating timeout path (expected exit124). The
existing CPU-only submit/timeline profiling checks also pass. The legacy-registry
adapter exists only in the probe, selected by
`-DWATCHDOG_PROBE_HAS_STATE_API=0` plus `WATCHDOG_BRIDGE_INCLUDE`; production has
one implementation.

Exact build/run commands, source and binary hashes, raw baseline/candidate logs,
and expected exit statuses are indexed in
`.landmark/evidence/metal-watchdog-exit.json`. External artifacts are under
`/opt/mldata_ssd/cogni-ml/drama3-decode-profile-20261017/drama_watchdog_probe/`.
Run tiny Metal probes serially under `scripts/run_safe.sh`; a nonzero timeout
mode is an expected discriminator, not a passing ordinary exit.

Rollback is the isolated watchdog lifetime commit. Refresh these claims after
changes to bridge state ownership, registration, watchdog callbacks, teardown,
compiler/runtime or device/OS. This does not certify arbitrary process shutdown
with abandoned application threads or pending Metal commands.
