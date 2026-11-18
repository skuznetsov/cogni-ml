# Reversing the alternating label order

Switching to `false` → `true` shifted mean first-token log-odds toward the new last label in five of six comparisons. The majority label changed once, on the original dataset at step 25. The final training label agreed with both the majority and the mean sign at two of six reversed checkpoints and five of six forward checkpoints.

Each arrow compares the [forward schedule](ALTERNATING_ORDER.md) with the reversed schedule.

| Training set | Step | Last target | `true` leads of 32 | Mean log-odds of `true` | Complete pairs of 16 |
|---|---:|---|---:|---:|---:|
| Original | 10 | `false` → `true` | 12 → 15 | −0.330 → −0.167 | 6 → 7 |
| Original | 25 | `true` → `false` | 22 → 15 | +0.132 → −0.031 | 5 → 6 |
| Original | 50 | `false` → `true` | 10 → 11 | −0.229 → −0.192 | 4 → 4 |
| Diverse | 10 | `false` → `true` | 2 → 6 | −0.495 → −0.376 | 4 → 4 |
| Diverse | 25 | `true` → `false` | 25 → 22 | +0.203 → +0.146 | 2 → 2 |
| Diverse | 50 | `false` → `true` | 20 → 20 | +0.105 → +0.095 | 3 → 3 |

All four matched-multiset comparisons at steps 10 and 50 kept the same majority label. On the diverse dataset at step 50, `true` still led on 20 prompts as its mean log-odds fell by 0.011. The number of complete pairs stayed at three, with one pair gained and one lost.

Reversing the order added one complete pair at steps 10 and 25 on the original dataset. The other four counts held steady. The best reversed checkpoint solved seven pairs, matching `tool-four`. The evaluation uses one generation seed across four families.

## Method

Both native QLoRA runs continue full-completion SFT from `tool-four` on the same Qwen3.5-2B Q4_K_M GGUF. Each uses Adam at 0.0001, training seed 42 and rank 8 on Q, K, V, O, gate, up and down in full-attention blocks 11, 15, 19 and 23. Each run trains for 50 updates and saves three snapshots with a single optimizer state. Both datasets have 32 training rows and each update supervises four tokens.

Every adjacent `true`, `false` visit was swapped, preserving the seeded order within each label group. The logged first-token IDs confirm `true:false` counts of 5:5, 12:13 and 25:25 in the reversed prefixes. The exact row multiplicities match between schedules at steps 10 and 50. Step 25 replaces one `true`-first visit with a `false`-first visit. The experiment measures the effect of reversing order throughout the prefix.

The 32 prompts cover the same 16 pairs in four held-out families. The expected first literal is balanced 16:16. A `true` lead is a higher first-token logit than `false`. The mean log-odds average the logit difference. A complete pair requires both sampled two-literal answers to match the Crystal truth tables. Generation uses context 4,096, seed 101, temperature 0.8, top-k 40 and up to 128 new tokens.

The experiment uses the pinned native executable from the forward run. The recorded original-order control matches all ten archived losses and the adapter bytes. The new logs and audits confirm the visit order, token IDs, 40, 100 and 200 supervised tokens and changes to all 28 projections and 56 factors. Independent scoring from the raw probabilities and answers reproduced the pair counts and preferences across both schedules. A replay of the coordinator revalidated the saved results.

## Reproduction

Run from the populated native QLoRA worktree with the frozen forward artifacts and executable.

```sh
python3 -B eval/crystal_bugfixes/caller_transfer/truth_training/reverse_order_checkpoints.py
```

Use `--output` with an unused directory for two fresh training runs. The expected signals are the archived control match, two audited 50-update runs and the six checkpoint comparisons above. Saved adapters, logs and raw observations are under `build/qlora/scale-2b/own-vs-peer-v1/diagnostics/helper-truth-training-v1/semantic-diversity-v1/reversed-order-v1`.

| Recorded artifact | SHA-256 |
|---|---|
| Protocol | `7b2acba06798a0f1a878edcd028245eb3ffbd88cb801c2e5f227fa822e06707a` |
| Results | `f6096d853814d1c5052531348aa74f904688c30b052dbc6b23d3fe38a78aad03` |
