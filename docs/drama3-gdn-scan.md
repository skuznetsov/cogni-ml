# DRAMA-3 known-input GDN scan qualification

The production-dimension affine scan is rejected for promotion on the tested
short prefill blocks: it changes native output/state bits and is slower even
when all numerical differences fit the unchanged tolerance metric. A separate
single-launch, serial-token control is bit-exact and shows a small local gain.
Neither path is integrated into production. This experiment makes no new
full-prefill, autoregressive decode, WAV-speed or audio-quality claim.

## Scope and implementation

Base revision: `e4131aaafe1c32a017daa39e57658c73139f9dd3`, plus the standalone
diagnostic sources recorded in the receipt. No production source changes.
Run on Apple M2 Max, 38 GPU cores, 64 GiB, macOS 26.6.2 (25G83), Xcode 27.1
(27A9275), Crystal 1.22.0 / LLVM 23.1.3. Checkpoint:
`cguldogan/drama-3-mlx-q4@5331ab509359c0e4a7e353a19e255ddf3e8152a8`.

`examples/drama3_gdn_scan_probe.cr` builds the actual 29-row no-reference prompt
for `Hello, this is a voice test.` and captures layers 0 and 38 at the existing
transposed recurrence. GPU copies retain normalized/scaled q/k, v, gates,
parameters, every output and every boundary state. Capture covers rows 0..28.
Replay uses the complete 29-row zero-state range and rows 7..25 with the actual
nonzero incoming state. Dimensions are 16 key heads, 32 value heads, K=V=128.

The three diagnostic modes use the same inputs:

- `sequential`: initial-state GPU copy plus the unchanged production recurrence
  dispatch for each row.
- `serial_tile`: one dispatch with private per-lane state and the original
  ascending-key F32 folds, decay, sigmoid and token order. It writes all outputs
  and final state. Tokens still execute sequentially inside the dispatch.
- `affine_scan`: independently build each block's (A,B), compose inclusive
  prefixes, transport S0 to each block, then replay every block for all outputs
  and the final state. The entire corridor is timed, not only its best stage.

For row state `r`, exact real arithmetic gives `r' = r A + B`,
`A = g(I - beta k k^T)`, `B = beta v k^T`. Earlier-then-later composition is
`(Al Ar, Bl Ar + Br)`. The GPU summary constructs A from identity basis states
with v=0 and B from zero states with actual v. This algebra does not make the
rounded F32 recurrence affine. All new shader folds retain ascending key order
and `fp contract(off)`; composition still changes where rounding occurs.

Block sizes are 8, 16 and the complete range length. Size 8 gives four blocks
for 29 rows and three for 19 rows, so later replay outputs consume composed
prefixes. Size 16 has only two blocks: the second consumes a direct summary,
not a composition. The one-block case is a degenerate control.

## Observed correctness

Six release Metal examples pass: four focused diagnostic examples and two
existing prefill examples. They cover offset views, production key/value widths,
ragged blocks, zero/nonzero states, both graph depths, shape/owner/mode rejection
before encoding, one-ULP, large finite and NaN/Inf comparator controls.

Actual-input probes ran at CogniGraph depths 0 and 2. Each has 12 validation
groups and 16 timing cohorts, with two warmup and eight measured counterbalanced
pairs per cohort. Both depths produce identical numerical receipts:

- Sequential replay reproduces the captured output and final state exactly.
- Every serial-tile output and final state is bit-exact in all 12 groups per
  depth and all 80 timed pairs including warmups across both depths.
- Affine scan fails the native bit gate in 6/12 groups per depth: all four
  size-8 groups and both nonzero-state size-16 groups. Maximum absolute error
  over outputs, prefixes and state is `3.8146973e-5`; all values are finite.
- All affine receipts pass the separate unchanged metric
  `6e-5 + 3e-4 * abs(reference)`. The largest error/tolerance ratio is
  `0.05352709`. This metric does not override the native bit gate.
- From S0=0, size 8 first differs at output row 16, where a composed prefix
  first feeds replay. From the actual nonzero S0 at base 7, size 8 first differs
  at local output row 8. Size-16 zero-state and single-block controls are exact;
  those cases do not certify composed-prefix replay.

Production sources are unchanged. Diagnostic arm encoding checks that its
graph/readback ledger is unchanged before and after each arm. Per run,
the terminal ledger is 495 submitted/completed commands, zero pending/retained
leases, 19,920 diagnostic bytes and 688 finite-control bytes read. Depth 2 reaches
two pending commands; depth 0 completes on submit. These are probe boundaries,
not a claim that existing production EOS control reads have been removed.

Capture itself adds GPU copies and 120 MiB of all-row state tables for the two
layers, outside the timed arms. Report `scratch_bytes` counts only the two affine
summary buffers and block prefix states: 40/30 MiB at size 8 for 29/19 rows,
20 MiB at size 16, and 10 MiB for one block. Output/final-state buffers, capture
tables, model resources and validation allocations are additional and excluded.

## Observed local timings

Each cell below is the median of eight paired GPU-command ratios
`sequential / candidate`. These are warm resident recurrence-corridor intervals,
not complete model requests. Depth-2 intervals may overlap and are not additive
phase costs. Shared-host load was uncontrolled; no confidence interval or causal
end-to-end improvement is claimed.

| Layer | Incoming state / rows | Serial tile, depth 0 | Serial tile, depth 2 | Affine size 8, depth 0 | Affine size 8, depth 2 |
|---|---|---:|---:|---:|---:|
| 0 | zero / 29 | 1.0581x | 1.0567x | 0.3967x | 0.6012x |
| 0 | actual nonzero / 19 | 1.0486x | 1.0375x | 0.3743x | 0.5268x |
| 38 | zero / 29 | 1.0671x | 1.0591x | 0.3904x | 0.6013x |
| 38 | actual nonzero / 19 | 1.0573x | 1.0422x | 0.3699x | 0.5385x |

The pooled descriptive serial-tile medians are 1.0581x / 1.0548x at depths 0/2.
At depth 0, the 29-row sequential command medians are 1.6430–1.6720 ms versus
1.5501–1.5521 ms for serial tile. This saves about 0.1 ms per tested recurrence,
not a large fraction of full generation. Encoding medians favor serial tile too,
but integration must charge convolution/q/k preparation, output normalization,
state/history commits and complete requests. The current rowwise prefill path
does not yet expose the required normalized tables as one production input.

Every affine timing cohort, including size 16 and the one-block controls, has a
median ratio below 1. The useful size-8 scans are 1.66–2.70 times slower. Dense
summary/prefix work and its scratch traffic outweigh saved dependency depth on
these short blocks. This rejects this dense implementation and workload; it
does not prove that all compact/WY or longer-window scans must lose.

## Evidence, reproduction and limits

Artifacts are on the external SSD under
`/opt/mldata_ssd/cogni-ml/drama3-gdn-scan-20261018/`: `depth0.json`, `depth0.log`,
`depth2.json`, `depth2.log`, analyzed summaries, build logs and spec logs. JSON
command indices require the matching STDERR GPU timeline log. The analyzer
joins each timed arm to one unique valid timing event, excludes warmups and
recomputes the declared gates using validation and all pair receipts, including
warmups. Ten offline tests pass; 16 deliberately damaged actual-receipt cases
are rejected, and warmup drift changes the final gates even when initial
validation passes. Scoped source review is ROBUST; it is correlated review,
not independent GPU replication. Analyzer exit 0 means the evidence was parsed;
the reported candidate gates can still fail. Its hashes bind each summary to
both files; these are locally produced receipts, not an independent attestation.
Selected diagnostic, capture, input-preparation and comparison sources are
checked against compile-time bytes before setup and before report creation.
The `source_tree_checked_before_and_after` field means those 19 listed files,
not the entire checkout. Model config/tokenizer/index and all five model shards
are hashed after execution. The DAC checkpoint is not used in this experiment.

Run from the isolated DRAMA worktree, with an existing matching bridge object:

```bash
export TMPDIR=/opt/mldata_ssd/cogni-ml/tmp
export CRYSTAL_CACHE_DIR=/opt/mldata_ssd/cogni-ml/crystal-cache
export COGNI_RUN_SAFE_MIN_FREE_PCT=15
export RUN_SAFE_PASSTHROUGH_STDIO=1
bridge=/opt/mldata_ssd/cogni-ml/drama3-decode-profile-20261017/bridge.o
out=/opt/mldata_ssd/cogni-ml/drama3-gdn-scan-20261018
model=/Users/sergey/.cache/cogni-ml/drama3-mlx-q4-5331ab50
scripts/run_safe.sh /opt/homebrew/bin/crystal 360 10000 spec --release -Ddrama3_metal \
  spec/drama3_gdn_scan_spec.cr spec/drama3_prefill_spec.cr \
  --link-flags="$bridge -framework Metal -framework Foundation -lc++ -framework MetalPerformanceShaders -fuse-ld=/usr/bin/ld"
scripts/run_safe.sh /opt/homebrew/bin/crystal 360 10000 build --release -Ddrama3_metal \
  examples/drama3_gdn_scan_probe.cr -o "$out/probe" \
  --link-flags="$bridge -framework Metal -framework Foundation -lc++ -framework MetalPerformanceShaders -fuse-ld=/usr/bin/ld"
# Use fresh output names: the probe refuses to overwrite an existing report.
scripts/run_safe.sh "$out/probe" 360 18000 "$model" "$out/repeat-depth0.json" 0 > "$out/repeat-depth0.log" 2>&1
scripts/run_safe.sh "$out/probe" 360 18000 "$model" "$out/repeat-depth2.json" 2 > "$out/repeat-depth2.log" 2>&1
python3 scripts/drama3/analyze_gdn_scan.py "$out/repeat-depth0.json" "$out/repeat-depth0.log"
python3 scripts/drama3/analyze_gdn_scan.py "$out/repeat-depth2.json" "$out/repeat-depth2.log"
python3 -m unittest discover -s spec -p drama3_gdn_scan_report_spec.py
```

No new MLX oracle or WAV was generated. Production is unchanged; the earlier
strict MLX row-1/layer-39 discrepancy remains a separate open numerical issue.
The user reports good audible quality in the examples they heard. Neither that
observation nor this recurrence experiment establishes when the old discrepancy
affects sound. Tolerances remain unchanged.

Next bounded implementation candidate: integrate the order-preserving serial
token tile into known-input prefill, preserving convolution history and every
output/state bit, then check saved upstream boundaries and full-WAV pairs.
Keep the dense affine scan diagnostic-only. Refresh after source/compiler/driver,
device, checkpoint, prompt, state range, queue-depth or control-stack changes;
lost or mismatched logs invalidate timing evidence. Rollback is omitting the
standalone probe: it has no production routing to disable.
