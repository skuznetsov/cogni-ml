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

## Native Qwen3.5-0.8B QLoRA

The experimental Crystal trainer runs against a frozen Qwen3.5-0.8B GGUF.
It caches the first 23 blocks' outputs per training example and trains only
FP32 rank-8, alpha-16 factors on the final block's Q (including its attention
gate), K, V, O, and FFN gate, up and down projections. The optimizer is
`ML::Optim::Adam` with learning rate `0.0001`. Training computes next-token
cross-entropy on the assistant completion and its end token. The output head
uses the dequantized GGUF values for the loss and hidden gradient, with logits
processed in 32-token tiles. Metal executes the quantized layer projections,
and Accelerate executes the adapter multiplications and dense gradients on
macOS. The checkpoint has 200,704 FP32 parameters and binds them to the GGUF
SHA-256. Base generation uses the same native route with a zero adapter update.

The comparison used `Qwen3.5-0.8B-Q4_K_M.gguf` with SHA-256
`b4e71cb32694733ba40c69e595a0524dfbc2cabe3c28fb7fef99eed4cf8730bf`.
Run these commands from the repository root on macOS.

~~~sh
mkdir -p build/qlora
xcrun clang++ -c src/ml/metal/bridge.mm -o build/qlora/bridge.o \
  -std=c++17 -fobjc-arc -fPIC -fno-objc-msgsend-selector-stubs
crystal build --release bin/qwen35_qlora.cr -o build/qlora/qwen35_qlora \
  --link-flags="$PWD/build/qlora/bridge.o -framework Metal -framework Foundation -lc++"
python3 scripts/crystal_bugfix_fragments.py export build/qlora/data
model=/absolute/path/to/Qwen3.5-0.8B-Q4_K_M.gguf
shasum -a 256 "$model"
for cohort in sft-5 sft-expanded; do
  build/qlora/qwen35_qlora train "$model" \
    "build/qlora/data/$cohort/train.jsonl" "build/qlora/data/$cohort/valid.jsonl" \
    "build/qlora/$cohort.adapter" 50 42 2>"build/qlora/$cohort.log"
done
~~~

`qwen35_qlora generate MODEL ADAPTER_OR_DASH MAX_NEW_TOKENS` reads NDJSON
chat requests on stdin and writes replies to stdout. Use `-` for the base
arm. Generation uses the embedded GGUF tokenizer, greedy decoding, and the
no-thinking chat prompt. The experiment caps training examples at 2,400
tokens and total generation context at 4,096 tokens.

Evaluate each arm against the six held-out families with the same replay
protocol. The output directory must be a new child of `build/qlora`.

~~~sh
for arm in base sft-5 sft-expanded; do
  adapter=-
  if [ "$arm" != base ]; then
    adapter="build/qlora/$arm.adapter"
  fi
  python3 scripts/crystal_bugfix_qlora_eval.py \
    --binary build/qlora/qwen35_qlora --model "$model" --adapter "$adapter" \
    --arm "$arm" --output "build/qlora/$arm" --max-new-tokens 768 --repairs 2
done
~~~

The evaluator preserves each attempt's prompt, answer and failure diagnostics.
The replay runner validates the macOS sandbox before compiling and executing
the focused spec. An evaluator exit code of 1 means one or more cases failed
after the allotted attempts. Exit code 2 indicates an infrastructure error.

### Four full-attention blocks

Append `4` to the training command to adapt the last four full-attention
blocks. Their zero-based indices on the pinned GGUF are 11, 15, 19 and 23.
The first 11 layers are cached per example. Gradients pass through the nine
frozen DeltaNet and SwiGLU blocks between adapters. Their forward projections
use the frozen GGUF weights and their transpose products use the dequantized
weights. The recurrent backward pass recomputes bounded state segments.
The four-block checkpoint stores 802,816 FP32 parameters and records the
block indices and GGUF digest. The single-block checkpoint format is still
supported.

~~~sh
build/qlora/qwen35_qlora train "$model" \
  build/qlora/data/sft-expanded/train.jsonl build/qlora/data/sft-expanded/valid.jsonl \
  build/qlora/sft-expanded-four.adapter 50 42 4
for arm in sft-expanded sft-expanded-four; do
  python3 scripts/crystal_bugfix_qlora_eval.py \
    --binary build/qlora/qwen35_qlora --model "$model" \
    --adapter "build/qlora/$arm.adapter" --full-blocks 4 \
    --arm "$arm" --output "build/qlora/matched-$arm" --max-new-tokens 768 --repairs 2
done
~~~

Both evaluation arms traverse the same four-block generation route. Loading a
single-block checkpoint on this route places its factors in the final block and
zeros the other three adapter updates.

### Six full-attention blocks

Append `6` to the training command to adapt all six full-attention blocks
on the pinned GGUF. Their zero-based indices are 3, 7, 11, 15, 19 and 23.
The checkpoint records the same v2 metadata and stores 1,204,224 FP32
parameters. The generation command infers the route from the checkpoint.
To compare the existing four-block checkpoint with the six-block checkpoint,
run both evaluator arms with `--full-blocks 6`. The four-block checkpoint
loads at layers 11, 15, 19 and 23 and the first two adapter updates are zero.

### Four blocks with repair examples

Collect model mistakes from the expanded training split. Each accepted
conversation includes the original prompt, a generated replacement that
fails compilation or the focused spec, the replay diagnostic, and the
historical replacement that passes the same spec. The failed replacement
and diagnostic are context for the final assistant completion. Gold
answers are used by the scorer and as training targets.

Run these commands with the exported fragment dataset, the native
generator, and the existing four-block checkpoint. Use fresh output paths.

~~~sh
bin=build/qlora/qwen35_qlora
expanded=build/qlora/data/sft-expanded
control=build/qlora/sft-expanded-four.adapter
run=build/qlora/four-repair
python3 scripts/crystal_bugfix_qlora_collect.py --binary "$bin" \
  --model "$model" --adapter "$control" --output "$run/current-candidates"
python3 scripts/crystal_bugfix_qlora_collect.py --binary "$bin" \
  --model "$model" --adapter - --stage compile --output "$run/stock-candidates"
crystal build --release eval/crystal_bugfixes/qlora_token_counts.cr \
  -o "$run/token_counts"
python3 scripts/crystal_bugfix_qlora_assemble.py --expanded "$expanded" \
  --model "$model" --token-counter "$run/token_counts" --output "$run/data" \
  "$run/current-candidates/candidates.jsonl" "$run/stock-candidates/candidates.jsonl"
"$bin" train "$model" "$run/data/train.jsonl" "$run/data/valid.jsonl" \
  "$run/sft-expanded-repair-four.adapter" 50 42 4 2>"$run/train.log"
~~~

Assembly checks the recorded prompt, response and replay logs against the
training manifest. It adds distinct failed answers with their passing fixes
when they fit the 2,400-token training limit. It preserves the original
one-shot rows and validation file byte for byte. The assembly manifest
records the accepted histories and file digests. Collection uses the same
sandbox as held-out evaluation, with UTC for Crystal's tempfile operations.
Replay diagnostics can include timings and other process-specific text.
Use the recorded JSONL and candidate order to reproduce an existing run.

Both arms use the same four-block route, quantized checkpoint, six held-out
prompts, greedy generation limit of 768 tokens, and up to two repair rounds.
Each round receives its own previous answer and diagnostic. The repair
adapter starts from fresh factors with the same seed and 50 updates as the
control. The added examples change the shuffled training mixture and the
number of one-shot updates.

~~~sh
for arm in control repair; do
  checkpoint="$control"
  if [ "$arm" = repair ]; then
    checkpoint="$run/sft-expanded-repair-four.adapter"
  fi
  python3 scripts/crystal_bugfix_qlora_eval.py \
    --binary "$bin" --model "$model" --adapter "$checkpoint" --full-blocks 4 \
    --arm "$arm" --output "$run/eval-$arm" --max-new-tokens 768 --repairs 2
done
~~~

The first run added 19 repair histories from 14 training cases, with 13
test diagnostics and six compiler diagnostics. At 50 updates, the new
mixture used 25 one-shot and 25 repair updates. Both checkpoints solved
0/6 held-out cases and reached tests on 4/6. The control compiled 12/18
responses and the repair-trained checkpoint compiled 8/18. On cr105 and
cr107 the new checkpoint introduced compile errors after test feedback.
On cr106 it reached the generation cap with an unfinished replacement
on all three attempts. These measurements use one seed.
