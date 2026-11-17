# Transfer of four reproduced Qwen3.5-2B repairs

Both frozen continuations had reproduced the same four teacher lines with their respective training-context drafts. This experiment uses that selected cohort. The `own` continuation trained on errors from its parent and the `peer` continuation trained on errors from another adapter. Both saw the same teacher lines.

The rename condition preserves the archived problem, excerpt boundaries and repair-dialogue format. It alpha-renames the identifiers and the corresponding draft. The Metal record fields are also constructor keywords, so the fixture uses their updated names. The harness checks failure signatures on each transformed source, retains the original spec diagnostic formatting, and takes compiler columns from the fresh build.

The rewrite condition starts independently from the same historical sources. It changes the surrounding implementation and adapts each historical draft to its new slot. For the trace fixture the original early returns become a constant false key and a truthiness key. All rewritten draft diagnostics come from compiling and running the transformed fixtures. Both models see both draft contexts. Each sampled answer gets a single repair opportunity. Answers can use any one-line expression that satisfies the checks.

| Case | Changed implementation | Checks on each transformed answer |
|---|---|---|
| Mean gradient | Multiplies by a reciprocal scale | 2D mean, 1D regression, 3D input with a non-unit upstream gradient |
| Metal stage timing | Reduces an array of stage durations | All eight durations, zero readback, fractional durations with nonzero readback |
| Conditioning mask | Builds an off-mask and inverts it | Binary decoding, rejection of out-of-domain bytes, reversed ordering, invalid image-mask byte |
| Trace phase | Selects a phase from a table keyed by completion | False completion, nil and exception paths, true completion |

Every transformed seeded source must compile and fail its named target. Every transformed reference must compile and pass all checks. The historical and renamed reference lines must each fail when literally pasted into every rewritten slot. Additional regression examples stay out of the model's feedback. The unchanged-source calibration repeats the two original matched contexts with freshly rendered diagnostics and the original two examples per fixture.

## Observed results

The frozen Qwen3.5-2B Q4_K_M and both adapters ran at seed 101, temperature 0.8 and top-k 40. The 4,096-token limit includes a 128-token output allowance. The 24 distinct prompts used 496–1,017 prefix tokens. There were 32 transformed responses and eight unchanged-source calibration responses. The native sampler recovered the original own/own Metal request by retrying prefill on the same prompt and seed.

In each cell the numerator is the number of passed Crystal examples after successful compilation. **Compile** marks a compiler rejection. **Format** marks a multiline response that could not be inserted under the one-line contract. The columns name the adapter followed by the source of the supplied failed draft.

| Variation | Case | own / own | own / peer | peer / own | peer / peer |
|---|---|---:|---:|---:|---:|
| Rename | Mean gradient | 3/3 | 3/3 | 3/3 | 3/3 |
| Rename | Metal timings | 0/3 | 0/3 | 3/3 | 0/3 |
| Rename | Conditioning mask | 4/4 | 4/4 | 4/4 | 4/4 |
| Rename | Trace phase | Format | Compile | Compile | 3/3 |
| Rewrite | Mean gradient | Compile | Compile | Compile | Compile |
| Rewrite | Metal timings | 3/3 | 3/3 | 3/3 | 3/3 |
| Rewrite | Conditioning mask | 2/4 | 2/4 | 2/4 | 2/4 |
| Rewrite | Trace phase | Compile | 0/3 | Compile | Compile |

On the original implementation each adapter reproduced the teacher verbatim on all four matched contexts and passed both original examples in each case. After renaming, the peer adapter fixed Metal on the own draft by supplying the four missing terms in a different order. The three unsuccessful sums double-counted the four terms above the hole. The peer trace repair on the peer draft used an equivalent nil comparison.

All four rewritten mean-gradient answers used the required reciprocal but referenced an unknown `@shape` instance variable. All four rewritten mask answers returned a predicate equivalent to `mask_byte == 1` inside the off-mask helper. The target and the reversed-order example failed. Both invalid-byte regressions passed. Three rewritten trace answers repeated `completed =` inside the existing assignment and stopped at compilation. The fourth produced a string for the Boolean lookup and failed all three examples. Only the rewritten Metal fixture passed in the primary run, with every response returning `@readback_ms`.

A secondary replay removed one leading literal `completed = ` from each of the three trace answers that repeated it. On the own adapter and own draft, the remaining expression `(call_return === nil) ? false : true` passed 3/3. The peer adapter's own-draft expression passed 0/3 and its peer-draft expression passed 2/3, failing the false-completion target. These three replays reuse the sampled text and are excluded from the table. The artifacts are in `posthoc-trace-prefix/results.json`.

The eight transformed teacher controls passed. Every literal transplant of either the archived teacher or the renamed teacher into the corresponding rewrite failed. These results describe four selected seen defect families at one seed.

| Local evidence file | SHA-256 |
|---|---|
| `seal.json` | `732ac608e6f54c361d4a8410745c4ed6418a001e65651b3064825aa8340e8d84` |
| `results.json` | `89b3f50ddd483aa6026c675f07aa0bd84b2bb90815ae2e3b1ddfdb6a51015d7f` |
| `posthoc-trace-prefix/results.json` | `dcfc6ca0e57670d2c5ffdc23275ab1f5e5cf8ff7a17a3ac8da3f9754a0c70ba8` |

## Reproduction

Run from the populated native worktree:

```sh
python3 -B eval/crystal_bugfixes/repair_transfer/run.py freeze
python3 -B eval/crystal_bugfixes/repair_transfer/run.py generate
```

Freeze records the source templates, both drafts, their actual failures, hidden checks, exact prompts and token counts. Generation validates the seal, model, adapters and sampler. Use `--output` with a new directory for a new freeze. By default the local evidence is under `build/qlora/scale-2b/own-vs-peer-v1/diagnostics/fix-transfer-v1`. The `fixtures` tree contains controls and prompts. The `runs` tree contains every raw response, compiler status and executable spec log.
