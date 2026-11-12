# Crystal bugfix pilot

This pilot measures one-shot patch generation for [Qwen3.5-0.8B](https://huggingface.co/Qwen/Qwen3.5-0.8B) on localized Crystal bugs. Each prompt gives a reconstructed bug report, a target file, and excerpts from the parent of a historical fix. Excerpts come from neighborhoods around the reference diff, with extra context where needed. This gives the model localization assistance and may reveal how many parts need editing.

[cases.json](cases.json) pins the history, source, spec, split, and expected test results for each case. The reference answer changes one production file. Tests use the fixed revision's full spec, a focused excerpt, or an adaptation of the historical regression that uses APIs shared by both revisions. The reader adaptation tests malformed GGUF cleanup on the existing default constructor. The tokenizer adaptation loads typed metadata through the shared GGUF loader.

| Split | Cases | Families |
| --- | --- | --- |
| Train | cr001–cr005 | N-gram risk policy and prompt-state artifact codecs |
| Validation | cr006 | GGUF reader resource cleanup |
| Held out | cr007–cr010 | Constrained tool decoding, GGUF special tokens, and ClickHouse HTTP framing |

## Export and replay

From the repository root, with Python 3.12+, Git, and Crystal on `PATH`:

```sh
python3 scripts/crystal_bugfix_dataset.py export build/crystal-bugfix-pilot-v1
python3 scripts/crystal_bugfix_dataset.py verify
```

The output directory must be new. Pass only `build/crystal-bugfix-pilot-v1/sft` to the trainer. It contains chat-format `train.jsonl` and `valid.jsonl` with historical diffs as assistant messages. `heldout/test.jsonl` contains case IDs and user messages for generation. [MLX LM data format](https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/LORA.md#data)

Replay checks the same spec at the historical parent and after applying the exported diff. It requires the expected behavioral failure before, the exact reference source after applying the diff, and a clean spec run after. All ten cases were replayed with Crystal 1.22.1 on macOS arm64 in CPU-only mode. The HTTP case uses a loopback server and the reader case needs a file-descriptor inventory at `/proc/self/fd` or `/dev/fd`. The cases need no model weights, GPU, or ClickHouse server.

## Compare the checkpoint and adapter

Pin one checkpoint revision and use the same tokenizer, chat template, thinking mode, precision, context and generation limits, and decoding settings in both runs. Generate one answer for each held-out prompt with the adapter disabled and again with it enabled. Keep the manifest, reference diffs, specs, and repository access out of both generation runs. Fix the training settings using train and validation before looking at held-out results.

Save each model answer as a unified diff. Prepare the historical replay directory and print the scoring command:

```sh
python3 scripts/crystal_bugfix_dataset.py prepare cr007 answer.diff build/replay-cr007
```

The command restricts the diff to the named source file. Run the printed Crystal command from the replay directory inside a disposable environment with no credentials or external network access. Keep loopback enabled for cr010. Give both runs the same time limit. Record patch rejection, compilation failure, behavioral failure or timeout, and a clean pass per case. A clean pass requires the manifest's full example count, exit status zero, and zero failures, errors, and pending examples. Review green patches for scope and regressions beyond the focused specs. Alternative patches can pass the same tests.

The split keeps each source file, historical spec, and bug family together. It establishes which tasks were excluded from this adapter's training and validation. Pretraining exposure is unknown. Five training patches and four held-out tasks support a smoke test and a case-by-case error comparison. A wider dataset and additional families are needed to estimate coding gains.

## Expanded fragment dataset

[fragments.json](fragments.json) pins 27 cases from the same history. The small training cohort reuses the five pilot fixes in fragment format. The expanded cohort adds 15 fixes from the shortlist and the wider repository history, for 20 fixes across 16 recorded families. Both cohorts share one validation example for the llama.cpp context ABI. The six new held-out tasks cover four source files and are disjoint from both training cohorts and their validation by source file, spec, and recorded bug family. The four original held-out cases remain available through the pilot.

| Held out | Regression | Failure before the fix |
| --- | --- | --- |
| cr104 | Nomic NeoX rotary pairing | Wrong rotation values |
| cr105 | Nomic MoE routing probabilities | Wrong weights for selected experts |
| cr106 | Causal decoder KV buffer writes | Unsupported offset parameter |
| cr107 | Causal decoder geometry | Invalid dimensions accepted |
| cr121 | Nomic matrix route device policy | Wrong automatic device admission |
| cr122 | Metal ComputeGraph buffer bindings | Unsupported length and partition parameters |

Export and replay the cases from the repository root with Python 3.12+, Git and Crystal.

~~~sh
python3 scripts/crystal_bugfix_fragments.py export build/crystal-bugfix-fragments-v2/data
python3 scripts/crystal_bugfix_fragments.py verify
~~~

The exporter writes two chat-format SFT directories, `sft-5` and `sft-expanded`. Their training sets are nested and their validation files are identical. `heldout/test.jsonl` has only IDs and user prompts. Keep the manifests, fixtures and reference answers outside the generation environment. With the pinned pilot Qwen tokenizer and thinking disabled, the longest training record is 2,397 tokens after chat templating. The longest held-out prompt is 578 tokens.

Each prompt gives editable source ranges from the historical parent. The model returns one complete Crystal code block per range, in order. Replay splices these blocks into the parent file and runs the same spec before and after replacement. Most references reproduce the historical target file exactly. Three cases use focused historical excerpts, with a separate green replay of the complete fixed target file. The test suites are focused regression checks and may cover only part of a historical commit. Two held-out parents fail to compile at the named call site. The other parent replays reach assertions or surface the expected runtime errors.

The replays use Crystal 1.22.1 on macOS arm64. Most use `-Dcpu_only`. The llama wrapper fixtures use native-call doubles and require `libllama` on the linker path. The QBit owner and ComputeGraph fixtures link the archived Metal bridge with Xcode's `clang++`. The QBit fixture checks device availability and admission with an empty snapshot. The graph fixture compiles dependencies using test handles. Both stop before GPU command submission. Tiny synthetic GGUF fixtures exercise the relevant model code without a downloaded checkpoint.

Use a saved model response to prepare an isolated historical replay.

~~~sh
python3 scripts/crystal_bugfix_fragments.py prepare cr104 answer.txt build/replay-cr104
~~~

Run the printed spec command from the replay directory in a disposable environment without credentials or external network access. Record replacement rejection, compilation failure, spec failure, timeout or pass for each attempt. For the base, LoRA-5 and LoRA-20 comparison, pin the same checkpoint and tokenizer, prompt and decoding settings, attempt count and feedback. Keep the two adapters' rank, placement, update budget and optimizer settings matched. These replays verify the dataset and provide no model comparison scores.
