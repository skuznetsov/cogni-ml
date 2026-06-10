# DiffusionGemma Resident Server Plan

## Current state

The PR `llama-diffusion-gemma-server` is a resident forward/logits server. It loads the GGUF once and accepts request-file paths over stdin. For each request it writes `C * n_vocab` float32 canvas logits to `R.resp`.

For the local Unsloth Q4_K_M model:

- `canvas = 256`
- `vocab = 262144`
- response per denoise step = `256 * 262144 * 4 = 268435456` bytes = `256 MiB`
- `MODE=fast-quality` at 10 denoise steps would move `2.5 GiB` of logits through files

This refutes a naive Python entropy-bound driver over the current full-logits protocol as the next speed path. It may remove repeated model load, but it adds a large file IPC corridor before sampling.

## Legal next move

Move the entropy-bound denoise loop into the resident server process and return only final tokens/text plus compact timing.

Use the existing llama.cpp PR implementation instead of duplicating sampler logic:

- Link `examples/diffusion-gemma-server` against `llama-diffusion` and `llama-common`.
- Add a text request mode, for example `TEXT\t<prompt>` on stdin.
- Tokenize and format the prompt in-process using the same chat-template path as `llama-diffusion-cli`.
- Call `diffusion_generate_entropy_bound(...)` with the same defaults as the CLI:
  - `max_denoising_steps = 10` for fast-quality, `16` for quality fallback
  - `kv_cache = true`
  - `max_length = n_input + diffusion.canvas_length`
  - entropy-bound metadata defaults with explicit env overrides only after parity
- Trim the canvas with the same EOG/repetition-loop logic as the CLI.
- Return a compact line such as `TEXT_OK\tsteps=10\ttotal_ms=...\t<escaped text>`.

## DoD

1. Build target:
   `cmake --build /Users/sergey/SrcArchives/AI/llama.cpp-diffusiongemma-pr/build-dg --config Release -j 8 --target llama-diffusion-gemma-server`

2. Resident load smoke still passes:
   `READY_TIMEOUT=180 scripts/diffusion_gemma_server_smoke.sh`

3. Text-mode smoke:
   - start server once
   - send two `TEXT` requests
   - both return clean text without reloading the model
   - stderr contains one model load and one `READY`

4. Quality parity gate:
   Compare text-mode output against `MODE=fast-quality RAW_OUTPUT=0 scripts/diffusion_gemma_proto.sh` on the default prompt/seed. Exact string parity is not required if RNG consumption differs, but answer extraction must be clean and the raw response must not expose `<|channel>` traces.

5. Transport gate:
   Text-mode response must be below `16 KiB` per request. If the protocol still writes full logits, the move is rejected.

## Rejected path

Do not build a Python driver around the existing `R.resp` full-logits file protocol for speed work. It is useful for diagnostics, but the byte corridor is too large for the demo acceleration target.
