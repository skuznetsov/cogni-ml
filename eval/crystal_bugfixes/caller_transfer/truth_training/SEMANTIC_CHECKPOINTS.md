# Full-SFT checkpoints on Crystal caller truth tables

The first-token preference moved toward `true` by step 10 on both training sets, toward `false` at step 25, and back toward `true` at step 50. The best trained checkpoint on these sampled pairs was diverse SFT at step 10 with 5/16. Tool-four solved 7/16.

All checkpoints received the same 32 prompts from four held-out families. The executed Crystal truth tables require `true` first on 16 prompts and `false` first on the other 16. A complete pair requires both sampled two-literal answers to match their full tables. A first-token pair requires the correct literal to have the higher logit on both sides when comparing `true` with `false`. Mean log-odds are the mean of `logit(true) - logit(false)` over all 32 prompts. There were zero ties.

| Training set | Step | True leads | Mean log-odds of true | Correct false-first leaders | First-token pairs | Sampled complete pairs | Sampled flip pairs |
|---|---:|---:|---:|---:|---:|---:|---:|
| Tool-four | 0 | 12/32 | −0.115 | 13/16 | 8/16 | 7/16 | 1/8 |
| Original | 10 | 30/32 | +1.205 | 1/16 | 4/16 | 2/16 | 1/8 |
| Original | 25 | 4/32 | −0.389 | 15/16 | 5/16 | 4/16 | 0/8 |
| Original | 50 | 29/32 | +0.344 | 3/16 | 5/16 | 4/16 | 0/8 |
| Diverse | 10 | 31/32 | +0.759 | 1/16 | 5/16 | 5/16 | 2/8 |
| Diverse | 25 | 0/32 | −0.685 | 16/16 | 4/16 | 4/16 | 0/8 |
| Diverse | 50 | 32/32 | +0.335 | 0/16 | 4/16 | 2/16 | 0/8 |

On the original set, steps 25 and 50 both solved four pairs with opposite first-token preferences. Two pairs solved at step 25 were lost and two others were gained. On the diverse set, step 10 solved two pairs whose required table changes between sides. All complete pairs at steps 25 and 50 on both sets had invariant tables.

The two training streams have the same target at every visited position. The first target was `true` on 7 of the first 10 updates, 6 of updates 11–25, and 14 of updates 26–50. The direction of these interval counts agrees with the observed preference at each endpoint. The experiment kept the label order fixed and measured a single generation seed, so the cause of the reversals remains open.

## Training and evaluation

The four new checkpoints each start from tool-four with fresh optimizer state on the same Qwen3.5-2B Q4_K_M GGUF. The pinned native Crystal trainer uses Adam at 0.0001, seed 42 and rank 8 on Q, K, V, O, gate, up and down in blocks 11, 15, 19 and 23. Its update count controls where the loop stops. The replays capped at 10 and 25 updates matched the archived prefill and recovery events, initial validation loss, visit order and every recorded update loss. All 28 projections and 56 factors changed from tool-four in each replay. The target counts were 40, 100 and 200 at steps 10, 25 and 50. Step 0 and both step 50 adapters and their observations came from the revalidated [50-step comparison](SEMANTIC_DIVERSITY.md).

All generations used 4,096 context, seed 101, temperature 0.8, top-k 40 and a limit of 128 new tokens. An independent pass reconstructed the first-token preferences from raw probabilities and the complete pair totals from exact lowercase Boolean answers. Its totals agreed at all seven checkpoints. Perturbing a replayed update loss by 0.01 triggered the prefix guard.

## Reproduction

Run from the populated native QLoRA worktree. The default output directory holds the four new adapter files, visit and loss audits, and evaluator observations. Running the command again revalidates those records. Pass `--output` with an unused directory to train and evaluate all four intermediate snapshots again.

```sh
python3 -B eval/crystal_bugfixes/caller_transfer/truth_training/semantic_checkpoints.py
```

The expected signals are four matching prefixes and the per-checkpoint totals above. The output path is `build/qlora/scale-2b/own-vs-peer-v1/diagnostics/helper-truth-training-v1/semantic-diversity-v1/checkpoint-sweep-v1`.

| Recorded artifact | SHA-256 |
|---|---|
| Protocol | `4fa959529e89d47daa0e7c5b25f75bd1c8914e9d22d9a71c99510411d21a7a64` |
| Results | `330cf10a1b0b7a9ec83533001c820c308efe2291d589b92146e36f9b68964b7d` |
