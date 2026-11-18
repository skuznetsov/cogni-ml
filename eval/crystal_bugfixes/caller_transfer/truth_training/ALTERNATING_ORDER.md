# Alternating the first target during full SFT

Alternating `true` and `false` reduced first-token skew on both datasets at all three checkpoints. All six checkpoints retained a preference for one literal. Complete pairs improved at three of the six checkpoints, fell at two and tied at one. The best trained checkpoint solved 6/16 pairs. Tool-four solved 7/16.

All 32 held-out prompts come from four families with executable Crystal truth tables. The required first token is `true` on 16 prompts and `false` on 16 prompts. A first-token pair requires the correct literal to lead by logit on both sides. A sampled complete pair requires both two-literal responses to match their respective truth tables. Mean log-odds are the mean of `logit(true) - logit(false)` over the 32 prompts. There were zero ties.

Each arrow shows the [shuffled run](SEMANTIC_CHECKPOINTS.md) followed by the alternating run.

| Training set | Step | True leads of 32 | Mean log-odds of true | First-token pairs of 16 | Sampled complete pairs of 16 |
|---|---:|---:|---:|---:|---:|
| Original | 10 | 30 → 12 | +1.205 → −0.330 | 4 → 5 | 2 → 6 |
| Original | 25 | 4 → 22 | −0.389 → +0.132 | 5 → 2 | 4 → 5 |
| Original | 50 | 29 → 10 | +0.344 → −0.229 | 5 → 3 | 4 → 4 |
| Diverse | 10 | 31 → 2 | +0.759 → −0.495 | 5 → 5 | 5 → 4 |
| Diverse | 25 | 0 → 25 | −0.685 → +0.203 | 4 → 5 | 4 → 2 |
| Diverse | 50 | 32 → 20 | +0.335 → +0.105 | 4 → 5 | 2 → 3 |

The step-25 comparison holds the exact 25-example multiset fixed on each dataset. Reordering turns the preference from `false` to `true` on both. Sampled complete pairs move in opposite directions. On the original dataset, the six step-10 wins all have invariant truth tables. The alternating run solves one changed-table pair at step 25 on the original dataset and zero at the other five checkpoints.

## Schedule and evaluation

Each arm starts from tool-four on the same Qwen3.5-2B Q4_K_M GGUF. The native Crystal trainer uses full completion SFT, Adam at 0.0001, training seed 42 and rank 8 on Q, K, V, O, gate, up and down in four full-attention blocks. It saves after updates 10, 25 and 50 in one run per dataset, with the same optimizer state across saves.

Each 32-row epoch uses the original seeded shuffle. The alternating order preserves the relative order within each first-label group and interleaves the groups starting with `true`. Actual first-target token IDs in both logs confirm 5:5, 13:12 and 25:25 at the three checkpoints, with 40, 100 and 200 supervised tokens. Every row appears once in the first 32 updates. At steps 10 and 50 the alternating prefix replaces two true-first visits with two false-first visits compared with the shuffled prefix. At step 25 the change is confined to order.

The control on the original order matched each of the first ten archived losses and the adapter bytes. The initial attempt detected a rebuild of the binary at qualification. The recorded runs both used a separately frozen binary after this control was reproduced. All 28 projections and 56 factors changed at each new checkpoint. Two runs on the same alternating schedule and four held-out families establish the local comparison. The generation seed was fixed at 101 with context 4,096, temperature 0.8, top-k 40 and a cap of 128 new tokens.

Independent scoring from the raw first-literal probabilities and direct answer strings reproduced the preference and complete-pair counts at all six checkpoints. A second run of the coordinator revalidated the cached results.

## Reproduction

Run from the populated native QLoRA worktree. Revalidate the recorded runs with the retained binary and artifacts.

```sh
python3 -B eval/crystal_bugfixes/caller_transfer/truth_training/alternate_checkpoints.py
```

For a fresh run, use an unused output directory and compile the driver there.

```sh
trial=build/qlora/scale-2b/own-vs-peer-v1/diagnostics/helper-truth-training-v1/semantic-diversity-v1/alternating-order-replay
mkdir "$trial"
crystal build eval/crystal_bugfixes/caller_transfer/truth_training/train_alternating_truth.cr --release -o "$trial/native-checkpoints" --link-flags="$PWD/build/bridge.o -framework Metal -framework Foundation -lc++"
python3 -B eval/crystal_bugfixes/caller_transfer/truth_training/alternate_checkpoints.py --driver "$trial/native-checkpoints" --output "$trial"
```

The expected signals are the bitwise control match, two audited 50-update runs, six saved checkpoints and the counts above. Recorded adapters, logs and observations are under `build/qlora/scale-2b/own-vs-peer-v1/diagnostics/helper-truth-training-v1/semantic-diversity-v1/alternating-order-v2`.

| Recorded artifact | SHA-256 |
|---|---|
| Protocol | `13f2ec22ecc0a101064ad8e9ba6ebf6de520b15ec807f9699f6a7710695362b9` |
| Results | `b5dc11bf8440ed9c14369c8894a6a49781e83b6ba14d8feb172af27bc6e3002b` |
