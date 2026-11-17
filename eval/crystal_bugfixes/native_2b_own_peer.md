# Native Qwen3.5-2B repair comparison

On the first repair of an identical failed draft, tool-four passed 2 of 18 schedules, its continuation on its own errors passed 0, and its continuation on peer errors passed 0. Both tool-four successes copied the inner arrays with `@options.map(&.dup)` on the same task. None of the three adapters solved a fresh task within its retry budget. On this cohort, the choice of error source yielded no additional fix.

Both continuations started at tool-four. 12 paired training problems, 50 updates and 838 supervised tokens per arm. The target completions and order of training examples matched at every update. Own drafts came from tool-four, peer drafts from diagnostics-only. GGUF Q4_K_M, rank 8, seven projections on blocks 11, 15, 19 and 23.

Seed schedules 101, 202 and 303. Context 4096 including 128 reserved generation tokens. The evaluator scored 268 responses. The longest prompt plus its generation reserve used 622 tokens. Fresh permits one initial answer and two repairs. Shared starts from a common hand-selected failed draft and its computed diagnostic, then permits two repairs. Every native response is stored in evaluation/runs. A pass requires compilation and all named fixture examples.

The entries below are successes across the three paired seed schedules. Within each cell, the first figure is success on the initial model response in that mode and the second figure is success within the full budget.

| Transfer | Task | Fresh parent | Fresh own | Fresh peer | Shared parent | Shared own | Shared peer |
|---|---|---:|---:|---:|---:|---:|---:|
| familiar | corridor-nested-view | 0 / 0 | 0 / 0 | 0 / 0 | 2 / 2 | 0 / 0 | 0 / 0 |
| familiar | layernorm-normalized-shape-domain | 0 / 0 | 0 / 0 | 0 / 0 | 0 / 1 | 0 / 0 | 0 / 0 |
| unfamiliar | hadamard-latest-tie | 0 / 0 | 0 / 0 | 0 / 0 | 0 / 0 | 0 / 0 | 0 / 0 |
| unfamiliar | victoriametrics-validity-endpoint | 0 / 0 | 0 / 0 | 0 / 0 | 0 / 0 | 0 / 0 | 0 / 0 |
| external | dimension-large-fraction | 0 / 0 | 0 / 0 | 0 / 0 | 0 / 0 | 0 / 0 | 0 / 0 |
| external | flex-zero-weight-distribution | 0 / 0 | 0 / 0 | 0 / 0 | 0 / 0 | 0 / 0 | 0 / 0 |

## First response by failure stage

| Mode | Arm | Pass | Format | Compile | Tests | Runtime |
|---|---|---:|---:|---:|---:|---:|
| fresh | parent | 0 | 5 | 8 | 5 | 0 |
| fresh | own | 0 | 1 | 17 | 0 | 0 |
| fresh | peer | 0 | 2 | 16 | 0 | 0 |
| shared_error | parent | 2 | 0 | 6 | 10 | 0 |
| shared_error | own | 0 | 0 | 14 | 4 | 0 |
| shared_error | peer | 0 | 0 | 13 | 5 | 0 |

## First repair of the shared failure

Counts below pool the first repair on the three seed schedules for each task. Each cell gives passes / format rejections / compiler failures / test failures / runtime failures, in this order.

| Task | Parent | Own | Peer |
|---|---:|---:|---:|
| corridor-nested-view | 2 / 0 / 0 / 1 / 0 | 0 / 0 / 2 / 1 / 0 | 0 / 0 / 2 / 1 / 0 |
| layernorm-normalized-shape-domain | 0 / 0 / 1 / 2 / 0 | 0 / 0 / 3 / 0 / 0 | 0 / 0 / 2 / 1 / 0 |
| hadamard-latest-tie | 0 / 0 / 0 / 3 / 0 | 0 / 0 / 1 / 2 / 0 | 0 / 0 / 1 / 2 / 0 |
| victoriametrics-validity-endpoint | 0 / 0 / 3 / 0 / 0 | 0 / 0 / 3 / 0 / 0 | 0 / 0 / 3 / 0 / 0 |
| dimension-large-fraction | 0 / 0 / 1 / 2 / 0 | 0 / 0 / 2 / 1 / 0 | 0 / 0 / 2 / 1 / 0 |
| flex-zero-weight-distribution | 0 / 0 / 1 / 2 / 0 | 0 / 0 / 3 / 0 / 0 | 0 / 0 / 3 / 0 / 0 |

## Independent counterexamples

Three tasks also had independent additional checks frozen before the first holdout generation. Only candidates accepted by the primary fixtures went through this second replay. The denominators below count initially accepted candidate occurrences. Identical outputs on different seed schedules were replayed once per task.

| Task | Arm | Passed extra / original accepted |
|---|---|---:|
| corridor-nested-view | parent | 2 / 2 |
| corridor-nested-view | own | 0 / 0 |
| corridor-nested-view | peer | 0 / 0 |
| hadamard-latest-tie | parent | 0 / 0 |
| hadamard-latest-tie | own | 0 / 0 |
| hadamard-latest-tie | peer | 0 / 0 |
| dimension-large-fraction | parent | 0 / 0 |
| dimension-large-fraction | own | 0 / 0 |
| dimension-large-fraction | peer | 0 / 0 |

Only two independently defined tasks per transfer stratum and one training seed per treatment. Evaluation seeds repeat the same problems. Differences are descriptive and do not establish a population-level improvement.


## Supplementary replay of all accepted candidates

The three remaining tasks had additional guards designed after inspecting the parent outputs and before inspecting either continuation. These new checks exposed an accepted LayerNorm answer that checked the first axis and whether the product was zero. A shape with two negative axes has a positive product. Each acceptance was replayed against the extra checks for its task. The original early stop was retained, so a candidate rejected on replay received no additional attempt.

| Mode | Arm | First pass surviving extra checks | Budget pass surviving extra checks |
|---|---|---:|---:|
| fresh | parent | 0 / 18 | 0 / 18 |
| fresh | own | 0 / 18 | 0 / 18 |
| fresh | peer | 0 / 18 | 0 / 18 |
| shared_error | parent | 2 / 18 | 2 / 18 |
| shared_error | own | 0 / 18 | 0 / 18 |
| shared_error | peer | 0 / 18 | 0 / 18 |

## Paired repairs and training data

For the primary first repair, paired schedules are counted once for each comparison.

| Pair (left / right) | Both pass | Only left | Only right | Both fail |
|---|---:|---:|---:|---:|
| own/peer | 0 | 0 | 0 | 18 |
| own/parent | 0 | 0 | 2 | 16 |
| peer/parent | 0 | 0 | 2 | 16 |

Eight paired generation rounds on each of the 12 training problems yielded 192 native drafts. Selection required the same failure stage and favored pairs with different drafts and diagnostics. The selected pairs had eight compile failures and four test failures. All 12 drafts differed, and nine diagnostics differed. Teacher answers passed the target and adjacent regression tests for each case. One preflight included a variable temporary path. Training on that case used the two identical saved generation prompts.

The six evaluation teacher lines and source paths have zero literal matches in the 20, 39 and 46 training records from the parent’s successive training stages, the 27 diagnostics-only training records, and each 12-row continuation set. Literal matching provides a limited contamination check. Some tasks intentionally exercise familiar mechanisms.

## Training fit and transfer conditions

The native trainer records the teacher-completion loss before each update. Each row below averages the first and last observed loss per training problem with equal problem weight. Each problem is revisited four or five times. These are training observations.

| Arm | Mean at first visit | Mean at last visit | Problems with lower loss |
|---|---:|---:|---:|
| own | 1.464 | 0.242 | 12 / 12 |
| peer | 1.388 | 0.204 | 12 / 12 |

All 12 training prompts explicitly contain the original failing test output as well as later feedback for the student's draft. All six evaluation prompts give a task description and source. In shared-error mode, the actual diagnostic for the common draft then follows. All six common drafts compile and fail a behavioral check. This distribution shift is shared by both continued arms. Its contribution to absolute repair performance was not measured.

The native prefill fallback retried a smaller chunk when the original prefix produced nonfinite values. Training rows recovered (own 1, peer 0). Evaluation requests recovered (parent 1, own 2, peer 1). Recovery reused the same sampling request. Every final answer in the comparison had a scored response.


## Reproduction

The experiment was run from the native QLoRA worktree at revision `74c02fe5ed5f211eeb337c91d2111ee0b0087418`. All 12 training defects were seeded in project code pinned to local synthetic revision `838e6602c58027cdcdac676fa0994c5339f8d877` (`codex/own-peer-v1-train-fixtures`). Student responses, compiler diagnostics, test replays and optimizer steps are measured. The external repository is crystal_tui at `16979139610f21b3bb8f77ca093564c4516b2ce4`. Family novelty is relative to the experimental fine-tuning data. The contents of model pretraining are unknown.

Run from `build/qlora/scale-2b/own-vs-peer-v1` in the same populated worktree. The native model, sampler and initial adapter fingerprints are in `train/own/audit.json`. The original fixture seal is in `holdout/seal.json`.

```sh
python3 -B data/generate_pairs.py audit
python3 -B train/run.py own
python3 -B train/run.py peer
for arm in parent own peer; do python3 -B evaluation/run.py "$arm" --seal evaluation/frozen-inputs.json; done
python3 -B evaluation/run.py report
python3 -B evaluation/verify.py
python3 -B holdout-support/extra-checks/replay.py replay
python3 -B evaluation/counterexamples/replay.py replay
python3 -B evaluation/summarize.py
```

## Evidence fingerprints

Paths are relative to `build/qlora/scale-2b/own-vs-peer-v1`.

| File | SHA-256 |
|---|---|
| `protocol.json` | `0e237b953b1870ebe17eb16072be3acbd0ccd10f2defa6aab92163abaf62c5b7` |
| `data/manifest.json` | `5409d1ade85bad312a3c74475a6421844c0ce49883bb61702c881fbe8129aed2` |
| `data/own.jsonl` | `e412b4b09c293f25c54d8d3ce1316f7a4d00f985f6d0266c60ab78b370b9bbed` |
| `data/peer.jsonl` | `d85b6d2d6a100090e2de9cb5a2be8d1ee153f29d131c98ff5e6a67d961c1e91f` |
| `holdout/seal.json` | `a74eb8edd754a54ad1beb5a7c354a3717465ca8c2510d660dbe753ba74d66cb6` |
| `evaluation/frozen-inputs.json` | `befe54f561ec72eca7b2a931259020e45cb03ef299beb3084fd2e498491829a0` |
| `train/matched.json` | `22495823a2edf7c1031798be2579b19319410788d787b49fe6c4313a1dcd370f` |
| `evaluation/comparison.json` | `89620b34dd36bb388abe203e3e2f2b4aa01c10204f20ef0142afc5af02135b81` |
| `evaluation/audit.json` | `337583334b68f71b2ff9752f5bc74df7e9bc9414a053f064c668220072d605ec` |
| `holdout-support/extra-checks/results.json` | `d9a31f4cc6c45b8528d7f987615aa834abf2de634613b9bba0038733d627b504` |
| `evaluation/counterexamples/results.json` | `f7592ffe017b34813d52b2a4d1bd3752c8d5f312907170a65099832cbecca36f` |
| `train/own/adapter.bin` | `20144e02c683364ef9c048d23a845afb0cd27d479dd120d2f8ef20e8b7a31c8f` |
| `train/peer/adapter.bin` | `f29b7192700a5c3db4cdb5b49b15cfef5c44ea9f74c4b36ca856e46d9861c42b` |

## Free generation on the training problems

A follow-up used the same native sampler and frozen weights. For each training row, the sampler received all four messages up to the failed draft and its diagnostic. It generated one answer with seed 101, temperature 0.8, top-k 40 and a 128-token generation budget. Teacher completions were withheld. Each one-line candidate was compiled and run against the original training specs. In the second condition only the original failure log in the first user message was removed. The description, code, draft and feedback on that draft were preserved.

| Prompt | Model | Passed training specs / 12 | Survived two supplementary checks / 12 | Matched teacher line / 12 |
|---|---|---:|---:|---:|
| Own-error training rows | tool-four | 0 | 0 | 0 |
| Own-error training rows | own continuation | 6 | 4 | 4 |
| Peer-error training rows | tool-four | 0 | 0 | 0 |
| Peer-error training rows | peer continuation | 5 | 4 | 4 |
| Own-error rows, original failure log omitted | own continuation | 5 | 3 | 2 |
| Peer-error rows, original failure log omitted | peer continuation | 6 | 5 | 5 |

The supplementary checks were designed after inspecting accepted answers in the full training contexts. Softmax on logits 1000 and 999 in a row with a nonzero offset caught index-based substitutes for the row maximum. A reasoning budget of `Int32::MAX` plus one answer token exposed an `OverflowError` where the Gemma profile requires an `ArgumentError`. Both teacher lines passed. Only the two affected tasks received additional checks. Teacher matches use the same one-line normalization as the training scorer.

These 72 sampled answers measure recall and sensitivity on seen problems at one seed. The unchanged held-out report above measures transfer. The adapter files each contain 1,277,952 FP32 values. The GGUF header identifies 24 Q4_K and four Q6_K matrices among the 28 adapted base projections.

The local evidence root is `build/qlora/scale-2b/own-vs-peer-v1/diagnostics/seen-repair`. Run `python3 -B probe.py`, `python3 -B probe.py --omit-original-failure` and `python3 -B guards.py` there. The saved results have these fingerprints.

| File | SHA-256 |
|---|---|
| `results.json` | `a177a66178a9ae5a53866f4fee9e541e264caab779eecba0afcad80097e52c9d` |
| `without-initial-failure/results.json` | `2a4b84472724b4d16dd7c325330dea90f031eee4c17aa9786e0b3c30e603a7fa` |
| `guards/results.json` | `1dc520e04b53f58cc00f25b019e3cee205aa6637e458c4b742e642b5308e72c4` |
