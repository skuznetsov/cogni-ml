# Hy-Nomas DiT text compaction, 2026-10-26

The opt-in M64 path removes 1,216 of 1,280 stored DiT text rows after the
unchanged text refiner. Attention still uses exactly 23 valid text tokens.
On the fixed saved-input Q4 forward, all 14 selected outputs are bitwise equal
to the frozen upstream control, including the final latent. Six adjacent timing
pairs all favor compaction; their median padded/compact wall-time ratio is
1.119019, or 10.64% less compact time. This is one observed run with substantial
absolute drift, not a video-throughput or stable-latency result.

The padded path remains the default. The complete camera-off native forward
probe selects the qualified shape with `--compact-text`; its default compact
storage is 64 rows. `--compact-text-tokens N` is an explicit diagnostic override,
not qualification of other shapes. The generic serial API requires callers to
choose and qualify their storage geometry; omitting storage there means the
exact valid prefix. No complete diffusion generation or pixel decode ran.

## Contract and implementation

The admitted fixture is batch one, 5,280 video tokens, D2048/H16/Dh128/F8192,
54 ordered blocks and a 23-token valid text prefix within 1,280 producer rows.
It is a first segment with no history, camera, action or chunk mask. Qwen's
saved state-26 crop, the full text refiner, conditioning, all weight bytes,
5,303-token joint attention, RoPE and the final image head are unchanged.
Unsupported compaction contexts fail before encoding.

The serial workspace allocates compact text input, text scratch and serial
output banks once. It copies the 64-row prefix on the existing GPU command
chain, then passes GPU buffers directly between blocks. The workspace validates
the full source extent, compact extent, identities, aliases and every serial
binding before appending work. It rejects release while its chain is in flight.

Two text Q/K RMS plans retain the original 1,280-token MPSGraph mean geometry
and full square/mean scratch. Their native square and affine normalization
dispatch only the 64 active storage rows. The reduction is over head features,
not across text rows; inactive square/mean rows may contain NaN and are never
consumed by active normalization. Preserving this graph shape costs about
20 MiB compared with shrinking that scratch too, but preserves the numerical
contract. The default Wan T5 RMS path still dispatches all prepared rows.

Sources: [serial ownership and GPU copy](../src/ml/video/hy_nomas_serial_mps.cr),
[active/full RMS geometry](../src/ml/video/hy_nomas_mps_rms.cr),
[RMS dispatch](../src/ml/video/wan_t5.cr),
[batch/prefix admission](../spec/hy_nomas_serial_compaction_admission_spec.cr).

## Refuted routes and numerical gate

Changing M can change MPS arithmetic despite row independence in exact
mathematics. The original M23 compact full forward failed the unchanged
`atol=rtol=2e-5` gate. On an isolated real block-0 text FFN-down projection,
M23 and M32 each have 15 tolerance failures versus M1280, maximum absolute
error 0.000205994; M64/128/256/512 are bitwise equal to M1280 for the first
23 rows. This is a single projection result, not qualification of those larger
sizes throughout the model.

The first block diagnostic then located an earlier raw difference at text Q/K
RMS. M64 alone still had one downstream FFN projection tolerance failure.
Retaining the full mean geometry removes that difference: all 40 selected
block-0 checkpoints are bitwise equal, including Q/K normalization, attention,
FFN and residual outputs. Inactive RMS scratch was NaN-poisoned before encoding.

The qualified full54 path passes 14 selected comparisons: block 0/7/53 image
outputs and valid text prefixes, projected conditioning text and vector,
image input, final norm/modulation/modulated/linear values and latent. Every
compared value is finite; tolerance failures and raw bit differences are zero.
The frozen upstream fixture uses the same decoded Q4 weights and F32 math.
This turn replays that fixture; it does not rerun or replace its producer.

Standalone clean/reused requests and four validation requests bracketing the
paired timings pass. The compact reused requests GPU-back up the refiner
output, poison excluded text rows 23..1279 and all registered scratch with
NaN, run the serial stack, and restore the full refiner output on GPU.
Compared output hashes remain exact. The 41 stored but excluded rows remain
outside attention. These tests do not compare every boundary of all 54 blocks.

## Full-forward timing and residency

Both padded and compact sessions are resident in one process before timing.
Each epoch has opposite warm-up order, followed by three alternating AB/BA
pairs. Diagnostics bracket four warm-ups and 12 timed forwards; comparisons
and poisoning are excluded from timed samples. Each sample includes the
unchanged patch/refiner, Q4 staging, 54 blocks and final head, plus the compact
GPU input copy where applicable. It starts at the saved Qwen boundary.

| Epoch | Padded median wall seconds | Compact median wall seconds | Median adjacent padded/compact ratio |
| --- | ---: | ---: | ---: |
| Padded-first warm-up | 8.119861 | 7.182976 | 1.111864 |
| Compact-first warm-up | 11.721074 | 10.481859 | 1.119812 |

All six pair ratios exceed one: range 1.035913..1.173759, median 1.119019.
The paired GPU-interval ratio is 1.120324, also six wins. Absolute epoch medians
drift upward 44.35% padded and 45.93% compact; the cause is not established.
Do not treat the across-epoch medians as a paired comparison or infer stable
seconds per second of video. No profiler was active.

Named GPU residency falls from 10,895,796,804 to 10,060,081,732 bytes:
835,715,072 bytes saved (797.000 MiB, 7.67%). This includes full RMS scratch and
the compact diagnostic refiner backup. It excludes private MPS allocations.
The paired process holds both sessions, 20,955,878,536 named bytes in total;
its peak sampled physical footprint is not a per-mode memory comparison.

Every timed forward has 1,244 existing command segments, one terminal wait,
zero tracked Metal allocations during encode and zero preterminal activation
readbacks. Diagnostic requests read 14 outputs only after completion. Initial
authenticated input/weight loading is outside this no-readback statement.

The file-cache-aware guard completes with exit zero under a 10% physical-memory
floor, 22 GiB process-group cap, 64 MiB system-swap-growth stop and 900-second
deadline. Peak sampled group footprint is 20.09684 GiB; minimum sampled
headroom is 15.25545 GiB, above the 6.4 GiB floor. Swap growth is zero.
File-backed headroom estimates reclaimability, and periodic sampling does not
bound between-poll peaks. No guard was weakened.

## Verification, evidence and rollback

Executed checks:

```sh
make BUILD_DIR=/opt/mldata_ssd/cogni-ml/hy-nomas-text-compaction-20261026/build/native \
  hy_nomas_text_row_stages_probe hy_nomas_q4_dit_probe \
  hy_nomas_text_compaction_benchmark spec_hy_nomas_q4_dit spec_hy_nomas_conditioned_eight
python3 -B -m unittest spec/hy_nomas_text_compaction_analysis_test.py
python3 -B examples/support/hy_nomas_text_compaction_analysis.py \
  /opt/mldata_ssd/cogni-ml/hy-nomas-text-compaction-20261026/reports/paired-rms64-v1.json \
  --guard /opt/mldata_ssd/cogni-ml/hy-nomas-text-compaction-20261026/logs/paired-rms64-v1.guard.jsonl
git diff --check
```

The build/spec invocation ran inside the external 10% guard and passed 22
Crystal examples; the CPU analyzer suite passes 12 tests. The analyzer checks
receipts and summarizes timing, not tensor bytes or a speedup acceptance gate.
The actual native tensor comparisons above provide the numerical evidence.
The M64 CLI default has a separate guarded clean/poison replay after rebuilding.

Exact runtime argv, reports, guard logs, source snapshots and preserved binaries
are under `/opt/mldata_ssd/cogni-ml/hy-nomas-text-compaction-20261026/`.
The curated [receipt](../.landmark/evidence/2026-10-26-hy-nomas-text-compaction.json)
binds their identities. Binary identities were captured after execution; this
is not a hermetic build attestation. Earlier failed source iterations were
not frozen and are recorded as observations, not reproducible source snapshots.

Correlated Mini-Quad source, geometry, alias, poison, admission and timing review
returned ROBUST within this fixed-fixture scope; it is not independent replication.
Rollback is to omit `--compact-text` or leave serial compaction disabled.
Requalify after changes to source, MPS/toolchain/device, input/weight identity,
text length or mask/segment/history/precision/lifetime contracts, or loss of
external evidence. Do not infer other prompt lengths from this single fixture.

The pre-existing Q4-versus-original-BF16 quality failure remains open; exact
same-Q4 compaction parity does not repair it. Before a generation-quality or
video-speed claim, qualify weight quality and inventory/port the decoder,
then measure the full producer/scheduler/DiT/decoder pipeline. The next useful
performance discriminator is that decoder inventory, followed by an admitted
bounded decoder block, rather than another text GEMM tile sweep.
