# UT61B current telemetry

The UT61B collector records a bounded current capture as JSONL with Python's
standard library. It uses POSIX `termios`, `select`, and modem-control ioctls;
it does not require pyserial. The serial settings are 2400 baud, 8 data bits,
no parity, one stop bit, DTR high, and RTS low. Capture duration must be greater
than zero and no more than 300 seconds. The output path is created exclusively,
so an existing report is never replaced.

## Capture

```sh
python3 scripts/ut61b_telemetry.py capture \
  --port /dev/cu.usbserial-10 \
  --duration 60 \
  --output /tmp/ut61b-gputer.jsonl \
  --circuit gputer \
  --voltage 116
```

The voltage is an assumption used only for AC apparent power. The default is
116 V; an explicitly supplied value is also recorded as an assumption. Each
JSONL row keeps the raw bytes as hex, UTC receipt time in epoch milliseconds,
monotonic receipt nanoseconds, circuit, AC/DC mode when recognizable, validity,
and a rejection reason. `current_amperes` appears only on valid current frames.
If the bounded capture receives no valid frames, it retains the report and
exits with status 2.

The decoder follows the observable 14-byte FS9922 UT61B frame described by the
[sigrok FS9922 driver](https://raw.githubusercontent.com/sigrokproject/libsigrok/master/src/dmm/fs9922.c):
sign, four digits, separator, decimal exponent, mode flags, status flags, unit
prefix, current-unit byte, bargraph, CR, LF. Decimal byte values `0`, `1`, `2`,
and `4` mean powers of ten `0`, `-3`, `-2`, and `-1`. The current-unit byte
must equal `0x40`; no-prefix, milli, and micro readings are scaled to amperes.
Mode-byte bit 0 indicates that the display bargraph is active; it is retained
in the raw frame and ignored for the current value. Other reserved mode bits,
ambiguous prefixes or AC/DC flags, HOLD, REL, MIN, MAX, and negative AC
readings are rejected. Status-byte bit 1 marks the unsupported nano multiplier
and remains rejected; bit 3 indicates auto-power-off and is recorded as
`auto_poweroff_active` without changing the current. The overrange digit payload
`?0:?` is rejected as overload; other malformed digits are also rejected.
Negative DC readings are retained. Rejected input is never converted to zero.

The framing and decoder tests verify this software contract. They do not verify
that a particular meter/adapter emits this frame at the selected serial
settings. If received bytes do not contain the documented CRLF framing, the
collector records rejected bytes and exits nonzero when no valid frame arrives.

## VictoriaMetrics import

Copy the report to the gputer host over SSH, then run the writer there. For
example, `gputer` below is an SSH host alias configured by the operator:

```sh
scp /tmp/ut61b-gputer.jsonl gputer:/tmp/ut61b-gputer.jsonl
ssh gputer 'python3 /path/to/cogni-ml/scripts/prometheus_remote_write.py \
  --input /tmp/ut61b-gputer.jsonl \
  --vm-url http://127.0.0.1:8428'
```

The writer only accepts credential-free loopback HTTP URLs and sends Prometheus
remote-write v1 protobuf in raw Snappy blocks. It uses bounded retries for
network errors, HTTP 5xx, and HTTP 429; other HTTP 4xx responses fail without a
retry. The command prints sample, series, batch, attempt, and response counts,
and exits nonzero on failure. The JSONL report remains available if a write
fails. No TCP forwarding is needed when the writer runs on the VM host.

The physical metrics are distinct from Fluke telemetry:

- `ut61b_measurement_valid{circuit,meter,mode}` records each valid or rejected
  candidate (`1` or `0`).
- `ut61b_current_amperes{circuit,meter,mode}` is emitted only for valid current
  readings.
- `ut61b_apparent_power_va{circuit,meter,mode,voltage_assumption_v,voltage_source,voltage_provenance}`
  is emitted only for valid AC current, using the recorded assumed voltage.

Apparent power remains in VA; it is never relabeled as watts. DC current has no
apparent-power sample. Remote-write timestamps have millisecond resolution, so
if multiple rows for the same series share a millisecond, the latest row is
sent for that timestamp and the number of coalesced samples appears in the CLI
summary. The JSONL source retains every row and its receipt clocks.

## Observed hardware and transport check

The 2026-10-17 Mac capture on `/dev/cu.usbserial-10` retained 21 frame
candidates. Six recognizable current frames were initially rejected because
mode-byte bit 0, the bargraph flag, was incorrectly considered reserved. The
original JSONL remains unchanged. A separate `ut61b-20261017-redecoded.jsonl`
preserves each original receipt clock and raw frame, adds the source row and
decoder SHA-256, and admits six AC values from 0.864 to 1.003 A with the
corrected decoder. The other 15 candidates remain rejected.

On the owned gputer deployment, the Python writer sent those rows to real
VictoriaMetrics `/api/v1/write`: 29 samples in five series, one successful
request, four same-millisecond validity samples coalesced. Raw export matched
all sample timestamps exactly and all values within absolute tolerance
`1e-12`. The preserved verification script and `ut61b-20261017-vm-export.json`
are in the task artifact's `layer-depth/` directory and in
`/build/persisten/cogni-gemma4-energy/reports/` on gputer.

Two subsequent 15-second captures received no valid frames (nine and four
rejected candidates). The operator confirmed that data transmission mode was
already enabled.

After the operator reseated the optical/USB adapter, two new captures with the
same decoder and serial settings received 34 valid frames in 20 seconds and
68 in 40 seconds, respectively. No malformed complete frame was observed; the
first capture retained one partial frame at the bounded stop. AC current ranged
from 0.850 to 1.212 A across the two captures. The median receipt interval was
582 and 589 ms, respectively, and neither capture reported a low-battery flag.
Raw input is preserved in `ut61b-20261017-reseated.jsonl` and
`ut61b-20261017-reseated-confirm.jsonl` in the same artifact and gputer report
directories.

Both fresh captures were imported through real Prometheus remote-write v1:
103 and 204 metric samples, respectively, with no same-ms coalescing. Raw VM
export matched all 307 timestamps exactly and values within `1e-12`. The first
immediate export after the second import was empty; a later export matched
without resending the input. An HTTP write success alone is therefore not the
readback check. The separate `verify_ut61b_live_roundtrip.py` and corresponding
`*-reseated*-vm-export.json` summaries preserve the fresh-capture scope; the
earlier retained-frame proof remains unchanged.

This establishes two successful bounded captures after reseating, rather than
a long-duration stability or causal diagnosis of the previous corruption.
Meter calibration and a synchronized UT61B/NVIDIA comparison remain unverified.
Do not treat series wiring as evidence that this meter is more accurate than
Fluke.

Software DoD, run locally and on gputer:

```sh
python3 -m unittest spec/test_ut61b_telemetry.py spec/test_prometheus_remote_write.py
```

All 25 tests pass, including the captured bargraph frame, split CR/LF,
APO/nano/overrange flags, rejected-data omission, protobuf/Snappy framing and
bounded HTTP retry behavior. Re-run the hardware capture and raw-export check
after changes to the meter mode, adapter, USB driver, serial settings, decoder,
writer or VictoriaMetrics. Rollback is to stop the bounded capture and retain
the JSONL; the tool does not alter the meter's measurement settings or the
server's GPU configuration.


## Same-host depth campaign

The later `depth-ut61b-20261017-c` campaign connects the USB adapter directly to
gputer (`/dev/ttyUSB0`) and records UT61B, NVIDIA and received token counters on
that host's clock. In 300 seconds it retains 507 valid AC frames and one initial
partial frame before all request windows. All 12 requests have bracketing
current coverage and a held valid-AC state, with a maximum 2000 ms raw gap.
This supersedes the synchronized-comparison blocker for these bounded windows.

The actual GPU, UT61B and token metrics match 8207 raw-export samples after
remote-write; 1522 belong to the three UT61B metric families. A delayed export
was verified without resending. Full methods, drift, results, recovery mode,
reproduction and artifact names are in
[the layer-depth experiment](gemma4-layer-depth-experiment.md#same-host-ut61b-campaign-2026-10-17).
The demo now uses UT61B as its single circuit source. This operational choice
does not establish greater calibration accuracy than Fluke. The assumed
116 V still yields apparent VA / VA s; active circuit W / J remain unavailable.
