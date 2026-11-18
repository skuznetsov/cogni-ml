# Teaching a 2B adapter to infer a helper's truth table

The caller either returns a Boolean helper's result or negates it. The prompt supplies the public result for inputs 0 and 1, shows the caller and blank helper, and asks for two Boolean literals in a specified order. The supervised target contains only the answer. This isolates one step in debugging before code generation or tool selection.

Four training templates and four separately named evaluation templates share three Crystal source layouts. Each template crosses the public result at 0, caller negation, and requested input order, giving eight conversations. Every public contract has opposite outputs at 0 and 1. Statement order is balanced across the split. Within a matched sign pair only the caller's exclamation mark changes. Within a matched order pair only the requested input order changes.

The primary unit is a sign pair with both answers correct. Each split has 16 such pairs. A stronger check requires all four answers for one public contract across both requested orders. There are eight contracts per split. We also repeat four unmodified prompts from the earlier caller-transfer probe, which describes public behavior in prose and uses Crystal's implicit return. Those four prompts were examined in the previous experiment.

The parent is tool-four on Qwen3.5-2B Q4_K_M. The child starts from exactly that adapter and takes 50 native QLoRA steps with training seed 42, rank 8, and the Q, K, V, O, gate, up and down projections of blocks 11, 15, 19 and 23. A separate integer-output conversation supplies the trainer's required validation input. It has no role in model selection. All answers are sampled once per adapter with seed 101, temperature 0.8, top-k 40, a 4,096-token budget, and a 128-token output cap. The scorer uses the previous probe's one-line normalization and comma-separated Boolean parser.

## Reproduction

From the populated native QLoRA worktree, run:

```sh
python3 -B eval/crystal_bugfixes/caller_transfer/truth_training/experiment.py
```

Use `--output PATH` for a fresh output directory. The default is `build/qlora/scale-2b/own-vs-peer-v1/diagnostics/helper-truth-training-v1`. The harness pins the training JSONL, separate instrument, cases and prompts, binaries, and code before scoring the parent or fitting the child. It compiles and executes 256 public-output checks for the teacher and its opposite. The raw generations, training log, factor audit and scored rows are kept in the output directory.

## Recorded run

The parent and trained adapter each produced 68 parseable answers. The native trainer visited all 32 training records in 50 steps, with 200 supervised tokens including end tokens, and changed both Float32 factors in all 28 projections. It took 292.2 seconds. The four archived parent completions reproduced the earlier raw answers.

| Matched requests | Parent correct | Trained correct | Parent complete sign pairs | Trained complete sign pairs | Correct across both signs and both query orders, parent → trained |
|---|---:|---:|---:|---:|---:|
| Training prompts | 15/32 | 21/32 | 1/16 | 5/16 | 0/8 → 0/8 |
| Fresh identifiers | 17/32 | 22/32 | 2/16 | 6/16 | 0/8 → 0/8 |
| Four archived prompts | 2/4 | 2/4 | 0/2 | 0/2 | Only one query order tested |

| Fresh template | Parent correct | Trained correct | Parent complete sign pairs | Trained complete sign pairs |
|---|---:|---:|---:|---:|
| Mosaic | 3/8 | 6/8 | 0/4 | 2/4 |
| Silver | 5/8 | 5/8 | 1/4 | 1/4 |
| Velvet | 5/8 | 6/8 | 1/4 | 2/4 |
| Pine | 4/8 | 5/8 | 0/4 | 1/4 |

On the fresh templates, direct callers went from 13/16 to 16/16 correct and inverted callers from 4/16 to 6/16. The child gained five complete pairs and lost one that the parent solved. Ten of its sixteen sign pairs still produced the same answer on both sides. Every remaining individual error was on an inverted caller, where it returned the public values in the requested order. It solved 5/8 inverted cases requiring `true, false` and 1/8 requiring `false, true`.

For example, take Silver with public values false at 0 and true at 1. When its caller negates the helper, the trained adapter answered `true, false` for both query orders. The first order was correct and the reversed order failed. The parent had answered `false, true` both times, succeeding on the reversed order. The fresh-template aggregate thus includes a concrete reversal of a previously correct pair. On the two old prose contracts both adapters solved only the direct callers.

This is a single generation seed, four fresh templates in three familiar layouts, and a two-valued domain with complementary outputs. The counts measure one inference step. The effect of architecture width and transfer to executable repairs remain open. A later scorer, data change, or sampler change requires a fresh evaluation.

| Evidence | SHA-256 |
|---|---|
| Frozen protocol | `b2c7d180def9c26c8b87aabcedc965a5b60e4c044e1bbc84f87cad02c885fb10` |
| Combined scored rows | `bf44b71bba64e13802ccc541fbc3d046fc0b2084809506d46b74bb7e6f9c3535` |
| Training audit | `9688ce78e0db627fb174d772f4ae296d593b1b986e071478c0a704355942c76d` |
| Child adapter | `7b20bf1e61d7134694ecc64b6d2c7a0e6026a640e0c8bf68b98527a3faac6716` |
