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
