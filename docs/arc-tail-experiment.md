# ARC last-block experiment

Status: numerical plumbing verified; real ARC pilot blocked by unavailable data.
No ARC quality claim. No real-data training has run.

Scope: real ARC training tasks, frozen local Qwen3.5 GGUF prefix, and a
differentiable final full-attention block. Public evaluation is excluded from
training. Existing runtime sources and GGUF weights are not modified.

First slice: choose one short DSL program from identity, rotations, reflections,
transpose, and nonzero crop. Supervision is inferred from each task's **train
pairs only**. This restricted grammar is a coverage limitation, not a general ARC
solver. Unsupported tasks must remain in the coverage denominator.

Compare original pretrained label logits, trained label-head rows, and trained
last block with the label head frozen. The head is a restricted vocabulary
projection, not full-vocabulary language-model fine-tuning. Native cogni-ml
exports the frozen prefix; PyTorch supplies backward for the final block.

Split by entire tasks, deterministically, before deriving train targets. Never
provide test outputs to prompt construction, target derivation, or optimization.
Report exact-grid and all-test exact-task scores separately from label accuracy.

Falsifiers:
- Differentiable final-block outputs must match native outputs before training.
- In head mode, block weights must be unchanged; in block mode, head weights
  must be unchanged. Frozen inputs must have no gradient.
- A changed test output must not change training labels/prompts or split.
- Train/holdout task IDs and content hashes must be disjoint.
- Unsupported tasks and ambiguous training-derived targets must be explicit.
- Real ARC data is required; synthetic tasks are excluded from training.

Stop on parity failure, nonfinite loss, unavailable data, or resource limits.
No dependency installation, downloads, API calls, submissions, or commits.
Checkpoint deployment in native inference, full-vocabulary generation, and
contest performance remain outside this initial slice.

## Reproduction

Worktree: `/Users/sergey/Projects/Crystal/cogni-ml-arc-tail-experiment`, detached
at `7cccdd8b`. New files are uncommitted. The original checkout is untouched.
Dependencies already installed: Python 3.12, torch 2.6, numpy, gguf 0.18; Crystal
1.21. No Transformers upgrade or downloaded weights are required.

```sh
cd /Users/sergey/Projects/Crystal/cogni-ml-arc-tail-experiment
PY=/opt/homebrew/Caskroom/miniconda/base/bin/python
MODEL="$HOME/.cache/lm-studio/models/lmstudio-community/Qwen3.5-0.8B-GGUF/Qwen3.5-0.8B-Q8_0.gguf"

"$PY" -B -m unittest discover -s scripts/arc_tail -p 'test_*.py'
CRYSTAL_CACHE_DIR=/tmp/arc-tail-crystal-cache crystal build -Dcpu_only --release \
  scripts/arc_tail/export.cr -o /tmp/arc_tail_export \
  --link-flags=-fuse-ld=/usr/bin/ld

# New output filenames are required: existing files are refused.
(ulimit -t 120; /tmp/arc_tail_export --model "$MODEL" --parity-only \
  --output /tmp/arc-tail-parity-new.jsonl)
"$PY" -B scripts/arc_tail/run.py parity --model "$MODEL" \
  --features /tmp/arc-tail-parity-new.jsonl
```

CPU-only is the portable correctness reference. The `/usr/bin/ld` selection
avoids the local lld/Xcode TAPI incompatibility. Crystal cache writes are kept
under `/tmp`. Native prefix extraction on CPU may be slow; do not infer contest
speed from these checks.

For actual ARC data, after providing a real training directory:

```sh
DATA_DIR=/actual/path/to/arc-agi-2/training
RUN_DIR=/tmp/arc-tail-real-run-new
"$PY" -B scripts/arc_tail/run.py prepare --data-dir "$DATA_DIR" --out "$RUN_DIR" \
  --seed 42 --max-train 8 --max-holdout 8
(ulimit -t 120; /tmp/arc_tail_export --model "$MODEL" \
  --requests "$RUN_DIR/requests.jsonl" --output "$RUN_DIR/features.jsonl" \
  --max-records 16 --max-tokens 256)
"$PY" -B scripts/arc_tail/run.py train --model "$MODEL" \
  --features "$RUN_DIR/features.jsonl" --out "$RUN_DIR" \
  --epochs 3 --lr 0.0001 --max-seconds 120
```

An interrupted export is not a valid corpus. The trainer requires every selected
task to have either a feature row or an explicit token-cap exclusion. If CPU
extraction hits the cap, use a separately verified faster prefix backend rather
than silently accepting partial coverage.

The run saves task/source/model hashes, parity, changed weight names, excluded
tasks, train/holdout exact-grid and exact-task metrics, the deterministic DSL
search comparator, and separate delta checkpoints. Holdout results do not tune
the run. Evaluation data is not admitted as the training directory.

Evidence: `docs/arc-tail-verification.json`. Six unit tests pass. Native/PyTorch
parity max absolute error on four sequence positions, including nonzero positions, is
`1.430511474609375e-6`. A seeded changed weight is rejected by the parity checker.
The real-data preparation command stops on missing task JSONs before creating
an output directory. No synthetic training dataset was substituted.
