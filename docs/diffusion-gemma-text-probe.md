# DiffusionGemma Text Probe

`scripts/diffusion_gemma_text_probe.sh` is the current text-first entrypoint for
the native bounded sparse DiffusionGemma prototype.

It runs the native sparse-loop smoke with:

- text prompt tokenization through the Gemma4 GGUF tokenizer bridge;
- text canvas tokenization without BOS;
- pipe-separated one-token candidate texts;
- decoded canvas/candidate diagnostics;
- optional wrapper-level regression gates.

Default probe:

```sh
scripts/diffusion_gemma_text_probe.sh
```

Default inputs are:

```text
PROMPT=Say:
CANVAS=Hello
CANDIDATES=Hello|world
STEPS=1
WARMUPS=1
REPEATS=2
FORMAT=tsv
```

Useful regression gate:

```sh
EXPECT_CHOSEN=Hello EXPECT_MIN_PROB=0.9 scripts/diffusion_gemma_text_probe.sh
```

Override the text probe:

```sh
PROMPT='Complete:' \
CANVAS='Hello' \
CANDIDATES='Hello|world' \
EXPECT_CHOSEN=Hello \
scripts/diffusion_gemma_text_probe.sh
```

The wrapper prints the full smoke output before applying expectation checks.
Expectation gates require TSV output:

- `EXPECT_CHOSEN=TEXT` checks the first data row's `last_chosen_texts`.
- `EXPECT_MIN_PROB=F` checks every first-row `last_argmax_probabilities` value.
- `SMOKE_RUNNER=/path/to/runner` can replace the underlying smoke command for
  wrapper-level tests.

## Boundaries

This is a bounded sparse candidate prototype, not full DiffusionGemma text
generation. Candidate texts must each tokenize to exactly one token. The
tokenizer bridge still uses the DiffusionGemma-capable `llama-tokenize` oracle
for encoding text; native decoding is qualitative diagnostics only.

The default local paths are:

```text
DIFFUSION_GEMMA_MODEL=$HOME/.cache/lm-studio/models/unsloth/diffusiongemma-26B-A4B-it-GGUF/diffusiongemma-26B-A4B-it-Q4_K_M.gguf
DIFFUSION_GEMMA_LLAMA_TOKENIZE=$HOME/SrcArchives/AI/llama.cpp-diffusiongemma-pr/build-dg/bin/llama-tokenize
```

Use `scripts/diffusion_gemma_sparse_loop_smoke.sh` directly when you need
token-id, canvas-length, candidate-count, or TSV sweep controls that are outside
the text-first wrapper.
