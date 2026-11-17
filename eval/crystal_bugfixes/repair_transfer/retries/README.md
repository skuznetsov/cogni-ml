# Two diagnostic retries on rewritten Crystal fixtures

This follow-up resumes the twelve failed conversations from the [single-repair transfer experiment](../README.md). Each of three rewritten fixtures is paired with both frozen adapters and both original draft contexts. Attempt 0 is the archived, unedited model response. Each additional attempt retains the complete conversation, appends the model's raw previous response and gives it the measured compiler error or failing public spec output for that response. A format rejection receives parser feedback. The branch ends on a public pass or after two additional samples. Hidden tests run separately and their results do not enter the dialogue.

The run uses the same Qwen3.5-2B Q4_K_M, native sampler, adapters, seed 101 on every request, top-k 40 and temperature 0.8. Each request reserves 128 output tokens within a 4,096-token total budget. The scorer uses the earlier one-line normalization. The source template, visible examples and hidden examples are identical to the parent run.

## Observed repairs

All twelve conversations used both retries. In each round seven answers failed compilation and five compiled and failed the public target. Fifteen of the 24 replies changed the previous answer. Every conversation finished each round with the same failing examples or compilation stage as attempt 0. The broken source controls compiled and failed their targets in all three fixtures. The teacher passed all ten examples across the three fixtures.

The first label in each pair names the adapter, the second the source of the draft in the original dialogue. The `own` adapter trained on errors from its parent, the `peer` adapter on errors from another adapter. Counts include the public and hidden checks together. A compiler failure prevented test execution. For mask predicates, “always true” describes the two allowed mask bytes, 0 and 1.

| Rewritten case | Adapter / draft | Retry 1 | Retry 2 |
|---|---|---|---|
| Mean gradient | own / own | Compile, `@shape` | Compile, `@dim1` |
| Mean gradient | own / peer | Compile, `@shape` | Compile, `@elements` |
| Mean gradient | peer / own | Compile, `@shape` | Compile, `@shape`, added `.double` |
| Mean gradient | peer / peer | Compile, `@shape` | Compile, `@dim1` |
| Conditioning mask | own / own | 2/4, low bit set | 2/4, always true |
| Conditioning mask | own / peer | 2/4, always true | 2/4, always true |
| Conditioning mask | peer / own | 2/4, low bit set | 2/4, always true |
| Conditioning mask | peer / peer | 2/4, always true | 2/4, always true (verbatim) |
| Trace phase | own / own | Compile, repeated assignment | Compile, repeated assignment |
| Trace phase | own / peer | 0/3, string key | 0/3, string key |
| Trace phase | peer / own | Compile, repeated assignment | Compile, repeated assignment |
| Trace phase | peer / peer | Compile, repeated assignment | Compile, repeated assignment |

On mean gradient every first retry appended `.to_f32` to the same unresolved `@shape.numel`. The compiler identified the unresolved field. The original excerpt contained the usable field `@input_shape`.

On the mask, both invalid-byte checks stayed green. Both the decoding target and the reversed-order regression kept failing. The public error changed from unexpected image placeholders to a lack of valid text tokens when the predicate switched to always true. The caller negates the off-mask predicate, so the helper must select zero.

All eight trace replies repeated their respective previous raw answers verbatim. In the own/own dialogue, removing the duplicated `completed =` alone had passed 3/3 in the parent experiment's separate replay.

The retry prompts used 774–3,121 prefix tokens. The native sampler recovered prefill on three requests with the same prompt and seed. The observations cover three selected failed rewrites at one seed. The [paired control](blind-control.md) compares these outcomes with neutral retry instructions on the same conversations.

| Evidence in the output directory | SHA-256 |
|---|---|
| `protocol.json` | `0b096aa5ed9bece180b3e2fdcd336291eaaa26d8a6ce2eb824abc7ef0b2670ba` |
| `results.json` | `41da7ee40dfd3696298bd140614ae434d5d78a4a3269e9d148a1c2fe2d0a8453` |

## Reproduction

Run from the populated native worktree with `python3 -B eval/crystal_bugfixes/repair_transfer/retries/run_retries.py`. The default output is `build/qlora/scale-2b/own-vs-peer-v1/diagnostics/fix-transfer-retries-v1`. For a fresh run, supply a new directory with `--output`. The protocol file pins the parent experiment, weights and helpers. Per-attempt artifacts include the prompts, native responses, separate public and full scores, compiler logs and executable spec logs.
