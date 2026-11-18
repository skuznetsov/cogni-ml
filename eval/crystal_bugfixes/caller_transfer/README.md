# Following a Boolean value through a Crystal caller

This probe compares the frozen native Qwen3.5-2B tool-four adapter with its own-error and peer-error continuations. Both continuations previously received 50 updates on twelve matched repair examples. This probe performs inference only.

Each of two authored fixtures has two implementations. A public method either uses a helper's Boolean result directly or inverts it. Within a family, both implementations share the public contract, helper, symbol index, tests, and broken expression. The two families use opposite public conventions for 0 and 1. A family counts as a successful transfer only when both variants pass the lane being measured. The correct caller also occupies a different position in each family's symbol index.

The protocol freezes all four fixtures and the harness before generation. Each variant must compile with a constant broken helper and fail the target, pass all executable Crystal specs with its reference repair, and fail the target with the other variant's repair.

| Lane | Information supplied | Measurement |
|---|---|---|
| Truth table | Requirement, helper, actual caller | Both required helper returns, in input order 0 then 1 |
| Shown caller | Requirement, helper, actual caller | Free one-line patch passes Crystal compilation, target and regressions |
| Given table | Requirement, helper, correct helper returns | Same executable criterion with the intermediate reasoning supplied |
| Method selection | Requirement, helper, index of method signatures | One voluntary `OPEN name` or `PROCEED`. Record whether the opened method consumes the helper |
| Repair after selection | Selection dialogue and the actual body of the selected method, if any | Free one-line patch passes Crystal compilation, target and regressions |

The selective repair lane sees its own raw action. If the choice is invalid, the model is told that no method was opened and still gets to produce a patch. The other lanes are fresh conversations. A successful prediction or patch never supplies an answer to another lane. Direct and inverted variants have identical model-visible selection prompts. This makes selecting a useful source distinct from inferring which implementation the model is facing. Selection tests a single lookup in a small supplied index.

The native sampler uses seed 101, temperature 0.8, top-k 40, and a 128-token generation allowance within a 4,096-token total window. Each lane receives one generation. Repairs are inserted into the authored helper and scored by the existing isolated Crystal compiler and spec runner.

## Observations

All four references passed their six (chunk dispatch) or seven (span retention) specs. In all four fixtures the broken helper compiled and failed a target. Substituting the opposite reference made both target examples fail. Each pair of source files differs by one `!` in the caller. The helper, rest of the implementation, and tests match within each pair.

| Adapter | Required helper table given caller | Opened actual caller (distinct indexes) | Compiled with caller shown | Passed with caller shown | Passed with table and public contract | Passed after method selection |
|---|---:|---:|---:|---:|---:|---:|
| tool-four | 2/4 | 0/2 | 1/4 | 0/4 | 0/4 | 0/4 |
| own | 2/4 | 0/2 | 4/4 | 2/4 | 1/4 | 1/4 |
| peer | 2/4 | 0/2 | 4/4 | 2/4 | 1/4 | 0/4 |

Every truth answer was a valid pair of literals. For chunk dispatch the three adapters returned `false, true` in both implementations. For span retention all returned `true, false` in both. Those are precisely the public truth tables. No adapter changed its prediction when the caller inverted the value. No primary repair lane passed both versions of either family.

The following table counts passed specs when the actual caller was shown and the model was asked for a patch. A rejected response stopped before compilation.

| Fixture | tool-four | own | peer |
|---|---:|---:|---:|
| Chunk dispatch, direct | Compile error | 6/6 | 6/6 |
| Chunk dispatch, inverted | 4/6 | 3/6 | 3/6 |
| Span retention, direct | Format rejection | 7/7 | 7/7 |
| Span retention, inverted | Compile error | 4/7 | 4/7 |

All three adapters returned valid actions. The two continuations chose `PROCEED` for both distinct indexes. Tool-four chose `PROCEED` for chunk dispatch and opened the unrelated `overlaps?` method for spans. Repeating each index verbatim for the opposite caller implementation yielded the same choice. The own adapter's one successful patch after selection was on direct chunk dispatch with no method opened.

The 60 primary generations used 161–343 prefix tokens and required no native prefill recovery. The fixtures are fresh authored implementations. The adapters' training histories already include flag gating, the treatment of false/nil in a caller, and nearby chunk/span concepts. Neither the foundation model's training corpus nor a causal effect of model width was tested. The signatures and domain language supply lexical hints to the choice of a caller. The check measures a single optional lookup in a supplied index at one seed.

| Primary artifact | SHA-256 |
|---|---|
| `fixtures.json` | `55458529e3955eee0c2f3c20213383f5603b77f57cc79a645206dbd349bd3ed0` |
| `protocol.json` | `dae67f6d3bbca57a54744d20e08992a46c62be3a4368c55e8e48e5743858c891` |
| `results.json` | `5d388b7b8a1a513105fc2dae0d2dae98e0a813fb0a40b7661d68ed1e58cff092` |

## Follow-up with the helper contract alone

The primary given-table prompt still described the public behavior. After seeing that the models often repeated that behavior we removed only its introductory sentence and the following blank line. All remaining message content and sampling settings were retained. With a shared helper there were exactly two distinct requests per adapter. Each of the six sampled responses was tested against both fixture variants requiring its table, giving twelve executable scores. Each request used 137 prefix tokens. The follow-up was exploratory.

| Required helper returns (0, 1) | tool-four | own | peer |
|---|---|---|---|
| false, true | Compile error | Both fixtures passed | Both fixtures passed |
| true, false | Compile error | Selected 1, both fixtures failed | Returned true on both inputs, both fixtures failed |

Tool-four emitted a conditional return with no Boolean result on its other path in both cases. For the reverse table, own returned `value == 1` and peer returned `value == 0 ? true : value == 1`. Both continuations had already produced a working zero predicate in the primary shown-caller lane for direct span retention. This variation across prompts limits claims of an intrinsic inability to express the predicate.

The supplementary results hash is `db9174a86b53899405ad6bb5e0d844231e7f2ffa58ce23bc81f036f723cea1e9`. The protocol hash is `95d8d5df4026c4b58874cd15e43bb29e77c9c11848f93a2fa88c4259a28463d5`.

## Reproduction

Run from the populated native experiment worktree:

```sh
python3 -B eval/crystal_bugfixes/caller_transfer/probe.py
python3 -B eval/crystal_bugfixes/caller_transfer/table_only.py
```

Evidence is written under `build/qlora/scale-2b/own-vs-peer-v1/diagnostics/caller-transfer-v1`. Use `--output` with a fresh directory for a new run. The supplement accepts `--primary` to choose a completed primary run and `--output` to choose its own destination. The protocol pins the GGUF, three adapters, sampler, evaluation code, token counter and fixtures. Per-lane artifacts include raw requests and completions, plus separate build and test logs for compiled repairs.
