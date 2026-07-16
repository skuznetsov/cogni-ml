# Cogni-ML

Crystal machine learning library with native Apple Silicon GPU acceleration.

Cogni-ML is currently two things:

- A general Crystal ML toolkit: tensors, autograd, NN layers, optimizers, GGUF readers, and llama.cpp bindings.
- A native Metal inference lab for GGUF models, with production-oriented work on `nomic-embed-text-v2-moe` embeddings and Qwen 3.5 text generation.

## Highlights

- Native Metal embedding pipeline for `nomic-embed-text-v2-moe`.
- Native Qwen 3.5 9B GGUF inference path for Apple Silicon Metal.
- Q4_K/Q5_K/Q6_K/Q8_0 quantized matmul kernels.
- Chunked Qwen 3.5 prefill, decode wave scheduling, prompt-state cache restore, and exact speculative decode harnesses.
- ComputeGraph wave scheduling with offset-aware barrier optimization.
- Crystal autograd engine, NN layers, and Adam/AdamW optimizers.
- llama.cpp FFI bindings for general GGUF model access.

## Architecture

```text
src/ml/
  core/         Tensor, Shape, MetalBuffer
  autograd/     Variable, GradFn backward pass
  nn/           Linear, LayerNorm, MultiHeadAttention, ViT
  optim/        Adam/AdamW
  llm/          llama.cpp FFI bindings
  gguf/         GGUF reader, tokenizer, dequantization, Qwen35, NomicBertMoE
  metal/        Device, ComputeEncoder, ComputeGraph, GraphEncoder
```

## Qwen 3.5 Native Metal

The native Qwen path targets `Qwen3.5-9B-Q4_K_M.gguf` on Apple Silicon. The code supports:

- Qwen 3.5 GGUF metadata and tokenizer loading.
- Q4_K, Q5_K, Q6_K, and Q8_0 quantized projections.
- Full-attention layers with GQA, partial RoPE, KV cache writes, and fused output projection.
- DeltaNet/recurrent layers with GPU-resident recurrent state and chunked prefill scan.
- Chunked prefill with final-token top1 shortcut.
- Decode wave scheduling to reduce command-buffer boundaries.
- Native Qwen BPE tokenizer with an external `llama-tokenize` fallback for A/B.
- Exact prompt-state save/restore, tokenized-prompt reuse, and longest-prefix prompt cache.
- Exact speculative decode harnesses:
  - neural draft with Qwen 3.5 0.8B Q8_0,
  - n-gram/cache draft for repeated/generated-template text,
  - target-verifier chunks with row-batched top1 for larger accepted chunks.

The 9B Q4_K_M path is the primary verified target. Qwen 3.6 27B is a scale-up target, but it should be treated as experimental until local correctness and performance runs are completed.

### Model Layout

The developer CLIs default to local LM Studio / llama.cpp-style paths:

```text
~/.cache/lm-studio/models/lmstudio-community/Qwen3.5-9B-GGUF/Qwen3.5-9B-Q4_K_M.gguf
~/.cache/lm-studio/models/lmstudio-community/Qwen3.5-0.8B-GGUF/Qwen3.5-0.8B-Q8_0.gguf
~/SrcArchives/AI/llama.cpp/build/bin/llama-tokenize
~/SrcArchives/AI/llama.cpp/build/bin/llama-bench
```

Most benchmark/probe CLIs also accept `--model`, `--target`, `--draft`, `--tokenizer-bin`, or environment overrides. `bin/qwen35_generate.cr` is intentionally a small demo and currently uses its constants at the top of the file.

### Build Qwen CLIs

Build the CPU-only GGUF/Qwen metadata smoke on Linux, CUDA hosts, or any environment where Metal is unavailable:

```sh
crystal build -Dcpu_only bin/qwen35_gguf_info.cr -o build/qwen35_gguf_info
./build/qwen35_gguf_info --model /path/to/Qwen3.5-9B-Q4_K_M.gguf
./build/qwen35_gguf_info --model /path/to/Qwen3.5-0.8B-Q8_0.gguf --load-weights
```

This entrypoint intentionally does not run inference. It verifies GGUF parsing, Qwen 3.5/3.6 hparams, tensor inventory, and the structured `Qwen35Weights` loader without pulling the Metal bridge into a Linux build.

Build the minimal Crystal CUDA Driver API smoke on NVIDIA/Linux hosts:

```sh
crystal build bin/cuda_driver_smoke.cr -o build/cuda_driver_smoke
./build/cuda_driver_smoke 4096
```

The CUDA smoke is a backend boundary probe only: it links `libcuda`, loads embedded PTX, launches a vector-add kernel, and checks the result.

CUDA probe code uses `src/ml/cuda/driver.cr` for reusable CUDA context, module/function, launch, copy, synchronize, and device-buffer ownership. This is intentionally small: it owns the raw CUDA Driver API lifecycle and calls, while higher-level layer execution is still probe-local until the CUDA backend split is promoted.
It also provides `ML::CUDA::ResidentSequenceRunner`, a thin lifecycle facade for resident sequence probes with explicit `upload_weights`, `reset_sequence`, `run_sequence`, and `read_outputs` phases.
`src/ml/cuda/qwen_recurrent_layer_runner.cr` is the first Qwen-specific runner extraction: it owns one recurrent layer's CUDA modules, device buffers, kernel parameters, weight upload, sequence reset, token launch graph, and output readback. `QwenRecurrentLayerRunner::Weights.load` owns GGUF tensor lookup, tensor-shape/type validation, and raw weight reads for the runner, including recurrent-layer `ffn_down` tensors stored as either Q4_K or Q6_K. CPU-reference comparison intentionally remains in the probe.

Build the first quantized CUDA correctness probe on NVIDIA/Linux hosts:

```sh
crystal build -Dcpu_only bin/cuda_q8_gemv_probe.cr -o build/cuda_q8_gemv_probe
./build/cuda_q8_gemv_probe \
  --model /path/to/Qwen3.5-0.8B-Q8_0.gguf \
  --tensor blk.0.ffn_up.weight \
  --kernel warp4 \
  --reps 100 \
  --warmup 10
```

`cuda_q8_gemv_probe` loads a real GGUF Q8_0 tensor, launches a Crystal-driven CUDA Driver API GEMV kernel over the raw GGUF block layout, and compares against the existing CPU `QuantMatmul` reference. `--kernel scalar` keeps the first one-thread-per-output-row correctness kernel; the default `--kernel warp4` maps four output rows to four warps per thread block and is the current faster probe shape. This is still a standalone backend-boundary probe, not an optimized Qwen CUDA inference path yet. The current full `qwen35_generate` CLI remains Metal-first.

Build the first Q4_K CUDA correctness probe for Qwen 9B/27B-style target tensors:

```sh
crystal build -Dcpu_only bin/cuda_q4k_gemv_probe.cr -o build/cuda_q4k_gemv_probe
./build/cuda_q4k_gemv_probe \
  --model /path/to/Qwen3.5-9B-Q4_K_M.gguf \
  --tensor blk.0.attn_gate.weight \
  --kernel warp4 \
  --reps 20 \
  --warmup 3
```

`cuda_q4k_gemv_probe` uses the raw GGUF Q4_K block layout (`d`, `dmin`, 12-byte packed scales/mins, 128-byte packed nibbles) and checks the CUDA output against the CPU `QuantMatmul` Q4_K reference. `--kernel scalar` keeps the first correctness kernel; the default `--kernel warp4` maps four output rows to four warps per block and is the current faster probe shape.

Build the Q6_K CUDA correctness/speed probe for Q4_K_M tensors that remain in Q6_K:

```sh
crystal build -Dcpu_only bin/cuda_q6k_gemv_probe.cr -o build/cuda_q6k_gemv_probe
./build/cuda_q6k_gemv_probe \
  --model /path/to/Qwen3.5-9B-Q4_K_M.gguf \
  --tensor blk.0.ffn_down.weight \
  --kernel warp4 \
  --reps 10 \
  --warmup 2
```

`cuda_q6k_gemv_probe` covers the GGUF Q6_K block layout (`ql`, `qh`, signed scales, `d`) used by output/value/down projections in mixed-quant target models. Like the Q4_K/Q8_0 probes, it is a standalone backend primitive check; full CUDA Qwen execution is still a separate backend split.

Build the first GPU-resident FFN sequence probe:

```sh
crystal build -Dcpu_only bin/cuda_ffn_sequence_probe.cr -o build/cuda_ffn_sequence_probe
./build/cuda_ffn_sequence_probe \
  --model /path/to/Qwen3.5-9B-Q4_K_M.gguf \
  --layer 0 \
  --reps 10 \
  --warmup 2
```

`cuda_ffn_sequence_probe` composes the checked CUDA primitives as `Q4_K ffn_gate + Q4_K ffn_up -> SwiGLU -> Q6_K ffn_down` while keeping the input, intermediate activations, and output projection input GPU-resident. Only the final hidden vector is copied back for comparison against the CPU `QuantMatmul` FFN reference.

Build the full-attention input projection bundle probe:

```sh
crystal build -Dcpu_only bin/cuda_attn_projection_probe.cr -o build/cuda_attn_projection_probe
./build/cuda_attn_projection_probe \
  --model /path/to/Qwen3.5-9B-Q4_K_M.gguf \
  --layer 3 \
  --reps 10 \
  --warmup 2
```

`cuda_attn_projection_probe` runs `Q4_K attn_q + Q4_K attn_k + Q6_K attn_v` from one GPU-resident hidden vector and copies Q/K/V back only after all projections complete. It targets full-attention layers such as `blk.3` in Qwen3.5 9B.
The probe now routes through `ML::CUDA::QwenFullAttnProjectionRunner`, supports `--tokens N`, and keeps Q/K/V outputs GPU-resident until the final correctness readback. It is the reusable input-projection boundary for future full-attention/KV CUDA work, not a complete full-attention layer runner yet.

Build the full-attention Q/K normalization + RoPE + KV-cache boundary probe:

```sh
crystal build -Dcpu_only bin/cuda_full_attn_kv_probe.cr -o build/cuda_full_attn_kv_probe
./build/cuda_full_attn_kv_probe \
  --model /path/to/Qwen3.5-9B-Q4_K_M.gguf \
  --layer 3 \
  --tokens 4 \
  --start-pos 2 \
  --max-seq 12
```

`cuda_full_attn_kv_probe` now routes through `ML::CUDA::QwenFullAttnLayerRunner`, a residual-hidden-to-final-hidden wrapper around the projection runner and `ML::CUDA::QwenFullAttnKVRunner`. The projection runner can apply the initial `attn_norm` on CUDA from residual hidden states before Q/K/V projection; Q is split into normalized/RoPE'd Q and gate, K is RMSNormed/RoPE'd, K/V rows are appended to a CUDA-resident cache at `start_pos`, a correctness-first serial CUDA kernel computes GQA scores, softmax, value reduction, and Q-gate multiplication, the resident gated attention output is projected through `attn_output.weight`, and the layer tail runs residual add, post-attention RMSNorm, FFN gate/up/SwiGLU/down, and final residual. It checks Q, gate, K, gated attention output, projected attention output, final hidden, K-cache, and V-cache against the CPU Qwen reference. This is now a clean one-layer semantics probe with device input/output hooks for mixed-stack composition, but it is not yet an end-to-end Linux decode path: full/recurrent stack scheduling, logits/top1, tokenizer/sampling, restored nonzero prefix KV, and a faster attention kernel remain separate gates.

Build the Q5_K CUDA recurrent-QKV probe:

```sh
crystal build -Dcpu_only bin/cuda_q5k_gemv_probe.cr -o build/cuda_q5k_gemv_probe
./build/cuda_q5k_gemv_probe \
  --model /path/to/Qwen3.5-9B-Q4_K_M.gguf \
  --tensor blk.0.attn_qkv.weight \
  --reps 10 \
  --warmup 2
```

`cuda_q5k_gemv_probe` covers the GGUF Q5_K block layout used by recurrent-layer combined `attn_qkv.weight` tensors in the current Qwen3.5 9B Q4_K_M file.

Build the recurrent-layer projection bundle probe:

```sh
crystal build -Dcpu_only bin/cuda_recurrent_projection_probe.cr -o build/cuda_recurrent_projection_probe
./build/cuda_recurrent_projection_probe \
  --model /path/to/Qwen3.5-9B-Q4_K_M.gguf \
  --layer 0 \
  --reps 10 \
  --warmup 2
```

`cuda_recurrent_projection_probe` runs `Q5_K attn_qkv + Q4_K attn_gate + Q4_K ssm_alpha + Q4_K ssm_beta` from one GPU-resident hidden vector and copies the four outputs back only after all kernels complete. It is the first CUDA recurrent projection-bundle proof; DeltaNet recurrence, convolution, state updates, and `ssm_out` remain separate work.

Build the synthetic DeltaNet output slice probe:

```sh
crystal build -Dcpu_only bin/cuda_deltanet_output_probe.cr -o build/cuda_deltanet_output_probe
./build/cuda_deltanet_output_probe \
  --model /path/to/Qwen3.5-9B-Q4_K_M.gguf \
  --layer 0 \
  --reps 10 \
  --warmup 2
```

`cuda_deltanet_output_probe` runs a synthetic CUDA DeltaNet state update, applies post RMSNorm/SiLU gating on GPU, and feeds the result directly into the real Q4_K `ssm_out.weight` projection. It is a stateful boundary probe, not a full recurrent layer: recurrent conv prep, alpha/beta transforms, residuals, and FFN remain separate work.

Build the recurrent prep/output slice probe:

```sh
crystal build -Dcpu_only bin/cuda_recurrent_prep_output_probe.cr -o build/cuda_recurrent_prep_output_probe
./build/cuda_recurrent_prep_output_probe \
  --model /path/to/Qwen3.5-9B-Q4_K_M.gguf \
  --layer 0 \
  --tokens 4 \
  --reps 10 \
  --warmup 2
```

`cuda_recurrent_prep_output_probe` now composes one full recurrent layer token slice: input RMSNorm, real recurrent projection bundle (`attn_qkv`, `attn_gate`, `ssm_alpha`, `ssm_beta`), recurrent conv prep, alpha/beta transforms, DeltaNet, post RMSNorm/SiLU, Q4_K `ssm_out`, residual add, post-attention RMSNorm, Q4_K FFN gate/up, SwiGLU, Q6_K FFN down, and final residual. `--tokens N` runs a GPU-resident sequence through persistent conv/SSM state and compares all token outputs plus final recurrent states against the CPU reference. The probe now separates one-time weight upload from per-sequence input/state reset and prints `weight_upload_ms`; timed `cuda_ms_per_token` excludes the persistent weight upload. GGUF recurrent-layer tensor loading is routed through `QwenRecurrentLayerRunner::Weights.load`, so the probe no longer manually passes every raw tensor into the runner constructor. It is still a standalone one-layer probe, not an end-to-end Linux decoder.

Build the recurrent multi-layer stack scaffold:

```sh
crystal build -Dcpu_only bin/cuda_recurrent_stack_probe.cr -o build/cuda_recurrent_stack_probe
./build/cuda_recurrent_stack_probe \
  --model /path/to/Qwen3.5-9B-Q4_K_M.gguf \
  --layers 0,2,4 \
  --tokens 2
```

`cuda_recurrent_stack_probe` chains multiple `QwenRecurrentLayerRunner` instances and compares the final hidden sequence plus each layer's recurrent conv/SSM state against the CPU reference. The default path hands each recurrent layer's CUDA output buffer directly to the next layer's CUDA input; `--host-handoff` keeps the older host-copy route as a debug oracle. This is still a recurrent-only scaffold, not an end-to-end Linux decoder.

Build the mixed recurrent/full-attention CUDA stack probe:

```sh
crystal build -Dcpu_only bin/cuda_mixed_stack_probe.cr -o build/cuda_mixed_stack_probe
./build/cuda_mixed_stack_probe \
  --model /path/to/Qwen3.5-9B-Q4_K_M.gguf \
  --layers 0,1,2,3,4 \
  --tokens 2 \
  --start-pos 2 \
  --max-seq 12
```

`cuda_mixed_stack_probe` composes `QwenRecurrentLayerRunner` and `QwenFullAttnLayerRunner` in model layer order with device-resident hidden handoff across recurrent/full-attention boundaries, then runs `QwenOutputHeadRunner` for output RMSNorm, quantized lm-head projection, and resident top1. The layer/head loop is now owned by `ML::CUDA::QwenMixedStackRunner`, which is the first model-slice decode-state object for CUDA. By default the probe copies back only the CUDA top1 id/value plus hidden/state debug outputs; pass `--read-logits` to also copy full logits for attribution, and `--profile-phases` to insert per-layer/head synchronizations and print attribution lines. It compares the final hidden sequence, top1, recurrent conv/SSM states, and full-attention KV cache rows against the CPU reference. This is the first mixed-stack CUDA correctness scaffold through resident top1; it still stops before tokenizer/sampling, repeated full-model decode ownership, and an optimized topK/sampling kernel. The current resident top1 is a simple two-phase partial-scan/reduce kernel: correct, but not yet promoted as a speed-optimized head.

Experimental CUDA exact cache-replay fast lane:

```sh
crystal build --release --no-debug -Dcpu_only \
  bin/cuda_mixed_stack_probe.cr \
  -o build/cuda_mixed_stack_probe

./build/cuda_mixed_stack_probe \
  --model /path/to/Qwen3.5-9B-Q4_K_M.gguf \
  --all-layers \
  --max-seq 256 \
  --greedy-loop-tokens 64 \
  --greedy-loop-probe-chunk-gamma 16 \
  --greedy-loop-probe-chunk-active-verify \
  --greedy-loop-probe-ngram \
  --greedy-loop-probe-ngram-source-history "$SOURCE_TOKEN_IDS" \
  --greedy-loop-probe-ngram-replay-start 1 \
  --greedy-loop-probe-ngram-cursor-only \
  --greedy-loop-probe-ngram-trusted-source \
  --greedy-loop-probe-ngram-schedule 64 \
  --skip-debug-readback
```

This path is exact verification of a validated source/cache cursor, not a
general-purpose sampler. It first checks that the live prefix matches the source
history at the replay cursor. If the prefix gate fails, the active verifier path
is disabled before any proposals are trusted and the probe falls back to plain
greedy target decode. If the gate passes, proposal chunks are verified through
the same resident CUDA stack and rejected chunks restore the exact target state.
Use smaller schedules such as `4,4,8,16` for weaker proposal sources where early
reject economics matter; use bulk schedules such as `64` only for artifact-level
trusted replay where the source history was produced by the same model/cache
contract and every proposed token is still target-verified before commit.

Trusted artifact restore lower-bound:

```sh
./build/cuda_mixed_stack_probe \
  --model /path/to/Qwen3.5-9B-Q4_K_M.gguf \
  --all-layers \
  --max-seq 256 \
  --known-replay-history "$SOURCE_TOKEN_IDS" \
  --known-replay-start 1 \
  --known-replay-tokens 64 \
  --known-replay-trusted-artifact-restore \
  --skip-debug-readback
```

For a conservative host-backed artifact simulation, add:

```sh
  --known-replay-trusted-artifact-host-restore
```

To snapshot only KV rows up to the restored cursor instead of the full
`max_seq` cache, use:

```sh
  --known-replay-trusted-artifact-live-kv
```

To include diagnostic artifact file IO and hash timing, add:

```sh
  --known-replay-trusted-artifact-io-probe \
  --known-replay-trusted-artifact-io-path /path/to/artifact.bin
```

This is a different contract from verified replay. It simulates a cache artifact
that already contains the exact token span and the post-span decode state for
the same model/config/source hash. The timed region restores that state and
emits cached tokens; it does not recompute the verifier body. Use it only for
session/cache artifacts whose model hash, tokenizer hash, prefix hash, token
span, and state artifact hash have been validated.

The default trusted-artifact probe snapshots/restores state inside device memory
and is a pure lower bound. The host-backed variant copies the complete decode
state to host memory, poisons the runner, then times host-to-device restore. On
the RTX 5060 Ti Qwen3.5-9B snapshot with `max_seq=128`, that state is
`61,079,552` bytes; restore costs `12.259ms` for a 64-token artifact and
`12.331ms` for a non-zero start9/24-token artifact. This is still not full
production IO: durable lookup, disk reads, hashing, and live-length KV trimming
remain outside the measured region.

The live-KV variant preserves exact output while reducing only the full-attention
cache portion. On the same host/model, start1/gen64 stores `56,885,248` bytes
and restores in `11.358ms`; start9/gen24 stores `54,788,096` bytes and restores
in `10.984ms`. The modest delta is a useful finding: recurrent DeltaNet state,
not KV capacity, dominates this short-context artifact.

On the persistent reefy storage path, the live-KV start1/gen64 artifact breaks
down into `52,690,944` recurrent-state bytes and `4,194,304` KV bytes. The
diagnostic file path measured write `17.695ms`, read `42.046ms`, SHA-256
verification `59.556ms`, and H2D restore `13.003ms`. For start9/gen24, the
artifact is `52,690,944` recurrent bytes plus `2,097,152` KV bytes; write/read/
hash/restore measured `18.249/42.994/53.814/12.825ms`. The SHA-256 timing uses
the host `sha256sum`/`shasum` tool to avoid adding OpenSSL linkage to the CUDA
probe, so treat it as a product-boundary diagnostic rather than a kernel metric.
An experimental contiguous-read reconstruction reduced start1/gen64 read time
from `42.046ms` to `36.310ms` while preserving `64/64` exact output, but start9/
gen24 remained noisy (`43.618ms` read). This points to some avoidable allocation/
fragmentation cost, but not enough to change the main conclusion: cache artifacts
need async prefetch or a resident artifact service to stay off the decode
critical path.
A first scalar CPU recurrent-state block-INT8 codec diagnostic compresses the
`52,690,944` recurrent bytes to `13.18-14.00MB` (`25.0-26.6%`) depending on
block size. On start1/gen64, block sizes `64/256/1024/4096` measured relative
RMSE `0.008798/0.012300/0.015476/0.019087`; encode/decode stayed around
`457-469ms` / `121-125ms`. A follow-up restore gate now decodes the block-INT8
recurrent buffers back into a host snapshot, restores that state, and runs one
continuation token against an exact uncompressed restored state. On the RTX 5060
Ti host, block sizes `64/256/1024/4096` preserved the same next top1 on
start9/gen24, and block `256/4096` also preserved the same next top1 on a later
start33/gen8 cursor. The multi-step source-aligned gate is stronger: block4096
preserved `16/16` continuation top1 ids on start9/gen24, and block256/block4096
preserved `8/8` continuation top1 ids on start33/gen8. The free-run greedy gate
now starts both exact-restored and decoded-INT8-restored states from the same
token and feeds back each path's own generated top1; block256 preserved `8/8`
free-run ids on start33/gen8, and block4096 preserved `16/16` free-run ids on
start9/gen24. A wider six-cursor free-run sweep changed the conclusion: block256
preserved `16/16` parity on all six tested cursors, and block1024 also preserved
`16/16` on those cursors, but block4096 drifted on later cursors (`15/16`,
`15/16`, and `6/16`). So block4096 is not a trusted restore format; block1024 is
the current best compression/parity trade-off candidate. This is still not a
production codec: the current scalar CPU codec is too slow for the critical
path, and these gates cover one prompt/history, not the full prompt distribution.

Build the Metal bridge once:

```sh
make build/bridge.o
```

Build the practical generation demo:

```sh
crystal build --release --no-debug \
  --link-flags="$(pwd)/build/bridge.o -framework Metal -framework Foundation -lc++" \
  bin/qwen35_generate.cr \
  -o build/qwen35_generate
```

Run greedy generation:

```sh
./build/qwen35_generate "The capital of France is" 64
```

Enable exact n-gram speculative decode for repeated text:

```sh
QWEN35_NGRAM_DECODE=1 ./build/qwen35_generate "The capital of France is" 64
```

Use the conservative automatic decode policy:

```sh
QWEN35_DECODE_POLICY=auto ./build/qwen35_generate "The capital of France is" 64
```

`auto` is the product-safe proposal-aware profile: it uses exact n-gram/cache
proposals only when they are large enough to amortize verification, enables the
candidate-shape risk gate plus a runtime corridor detector for untrusted suffix
replay, and otherwise falls back to exact target decoding without invoking the
neural draft model.

Enable exact neural speculative decode with the Qwen 3.5 0.8B draft:

```sh
QWEN35_SPECULATIVE_DECODE=1 \
QWEN35_HEAD_FULL_ROWS_GUARDED=1 \
./build/qwen35_generate "The capital of France is" 64
```

Experimental same-weight proposal-route memory:

```sh
crystal build --release --no-debug \
  bin/qwen35_proposal_route_memory.cr \
  -o build/qwen35_proposal_route_memory

QWEN35_ROUTE_CAL_ROOT=/tmp/qwen35_route_memory \
QWEN35_ROUTE_CAL_GEN=16 \
QWEN35_ROUTE_CAL_GAMMA=4 \
QWEN35_ROUTE_CAL_UPDOWN_RANK=4 \
QWEN35_ROUTE_CAL_UPDOWN_LAYERS=0,2,4 \
scripts/qwen35_proposal_route_calibrate.sh
```

The route cache stores certified proposal-body choices such as `baseline` or
`pca_updown` for the GPU self-spec probe path. It is intentionally not a
`qwen35_generate` decode-policy switch yet: the product generator currently has
greedy, n-gram/cache, external-draft speculative, and GGUF-MTP paths, while the
PCA-updown same-weight proposal body lives in
`bin/qwen35_deltanet_fixed_basis_probe.cr`. Use route memory to avoid repeating
online route calibration in that probe corridor; do not count it as a product
generation speedup until the same verifier/proposal corridor is wired into the
normal generator. `qwen35_generate` can resolve the route table with
`QWEN35_SELF_SPEC_ROUTE_MEMORY_ROOT` for diagnostics and future wiring, but it
prints `product_self_spec=unsupported` and leaves the decode path unchanged.
If `QWEN35_SELF_SPEC_UPDOWN_ADAPTERS_PATH` is also set, the generator validates
the referenced PCA-updown adapter artifact against the selected route layers and
rank through `ML::GGUF::Qwen35SelfSpecPlan`, then reports
`plan=pca_updown|invalid_adapter|baseline|route_miss` and
`adapter_artifact=valid|invalid` without executing the adapter.
The PCA-updown FFN adapter data path is now shared in
`src/ml/gguf/qwen35_ffn_updown_adapter.cr`: it owns the centered low-rank
projection math, Hadamard/symmetric quant-dequant helpers, and the
`qwen35_ffn_updown_adapter_v1` artifact dump/load format. The heavy
self-spec scheduler is still probe-local.

Manual route lookup/seed example:

```sh
./build/qwen35_proposal_route_memory \
  --root /tmp/qwen35_route_memory \
  --prompt "def square(x): return x * x\n" \
  --route-key code_square \
  --route pca_updown \
  --rank 4 \
  --layers 0,2,4
```

Enable Qwen chat-template prompting and XML-style function calls:

```sh
QWEN35_CHAT=1 \
QWEN35_CHAT_SYSTEM="You are a tool-using assistant." \
QWEN35_TOOLS_JSON='[{"type":"function","function":{"name":"get_weather","description":"Get weather for a city","parameters":{"type":"object","properties":{"city":{"type":"string"}},"required":["city"]}}}]' \
./build/qwen35_generate "Weather in Paris?" 64
```

`QWEN35_TOOLS_JSON` uses the tool schema shape embedded in the Qwen
`tokenizer.chat_template`. Qwen 3.5/3.6 GGUFs use an XML-ish tool-call format,
for example `<tool_call><function=...><parameter=...>...`, not a constrained
JSON-schema decoder. The CLI renders the model prompt with the Qwen chat tokens,
uses the rendered prompt for tokenization and prompt-cache keys, and prints a
parsed JSON summary when generated text contains Qwen `<tool_call>` blocks.

For harnesses that expect normal JSON function-calling, keep the model-facing
Qwen XML template and normalize only at the host boundary:

```sh
QWEN35_CHAT=1 \
QWEN35_TOOL_RESPONSE_JSON=simple \
QWEN35_TOOLS_JSON='[{"type":"function","function":{"name":"read_file","description":"Read a file","parameters":{"type":"object","properties":{"path":{"type":"string"}},"required":["path"]}}}]' \
./build/qwen35_generate "Read src/foo.cr" 64
```

The emitted `=== Tool response JSON ===` block uses the CrystalBall-compatible
shape. When `QWEN35_TOOLS_JSON` is available, argument normalization is
schema-aware for basic scalar types, so string fields stay strings while
integer/number/boolean fields are emitted as typed JSON values:

```json
{"content":null,"tool_calls":[{"name":"read_file","arguments":{"path":"src/foo.cr"}}]}
```

Set `QWEN35_TOOL_RESPONSE_JSON=openai` to emit OpenAI-style wrappers with
`id`, `type`, and `function.arguments` as a JSON string. Multi-turn harnesses
can pass OpenAI-style messages through `QWEN35_MESSAGES_JSON`; assistant
`tool_calls` are rendered back into Qwen XML blocks and `tool` messages are
rendered as Qwen `tool` role messages.

`../crystal_ball` can use this backend through its `CogniQwen` provider:

```sh
crystal build --release bin/qwen35_generate -o build/qwen35_generate
cd ../crystal_ball
CB_COGNI_QWEN=1 \
CB_COGNI_QWEN_BIN=../cogni-ml/build/qwen35_generate \
crystal run src/cli.cr
```

The lightweight adapter is useful for testing prompt/render and output parsing
without loading the model:

```sh
printf '<tool_call>\n<function=read_file>\n<parameter=path>\nsrc/foo.cr\n</parameter>\n</function>\n</tool_call>\n' \
  | crystal run bin/qwen35_tool_json_adapter.cr -- --parse-output --format=simple

printf '{"messages":[{"role":"user","content":"Read src/foo.cr"}],"tools":[]}' \
  | crystal run bin/qwen35_tool_json_adapter.cr -- --render-request
```

Enable exact prompt cache:

```sh
QWEN35_PROMPT_CACHE=1 \
QWEN35_SESSION_ID=demo \
./build/qwen35_generate "The capital of France is" 64
```

With prompt cache enabled, `qwen35_generate` now caches two independent
artifacts:

- tokenized prompt text, so repeated prompts avoid the older external tokenizer
  process path;
- exact prompt state, including optional full-prompt next-token metadata for
  long generations.

The prompt-state lookup is longest-prefix based. If the new prompt extends a
cached prompt, the CLI restores the verified prefix state and exact-replays only
the uncached suffix before decode. A local product smoke on Qwen3.5 9B Q4_K_M
restored `12/14` prompt tokens, replayed `2`, skipped normal prompt prefill
(`prefill_ms=0.0`), and used `cache_route=prompt_state_restore`.
`QWEN35_PROMPT_CACHE_FULL_HIT_MIN_GEN` controls when a full-prompt hit can use
stored next-token metadata directly; the default favors longer generations so
short requests do not merely shift first model work from `prefill_ms` to
`decode_ms`.

When `QWEN35_PROMPT_CACHE_FAST_FORWARD=1` and source-history cache is enabled,
the CLI also writes a direct per-key output fast-forward certificate for fully
generated spans. On a repeated same-session terminal request, this lets
`qwen35_generate` validate model/session/prompt/output hashes and emit cached
token ids/text before opening the GGUF. The older tokenized-prompt +
source-history + manifest scan remains as a fail-closed fallback for legacy or
tampered direct certificates.

Fast-forward state artifacts can also be stored in the guarded compressed v2
format. Set `QWEN35_PROMPT_CACHE_ARTIFACT_CODEC=recurrent-bf16` to compress
recurrent DeltaNet state while keeping KV rows raw; Metal restores these
validated artifacts through the mmap encoded path. `recurrent-int8` remains
explicitly gated by `QWEN35_PROMPT_CACHE_METAL_INT8_RESTORE=1`.

The native tokenizer is the default. Set `QWEN35_NATIVE_TOKENIZER_OFF=1` only
when comparing against the external `llama-tokenize` bootstrap path.

Each generation ends with a compact request-phase summary:

```text
request summary: total_ms=... model_load_ms=... draft_load_ms=... tokenize_ms=... token_cache_hit=... state_prepare_ms=... source_history_lookup_ms=... cache_restore_ms=... prefill_ms=... decode_ms=... source_history_save_ms=... prompt_tokens=... output_tokens=...
```

Use `total_ms` for one-shot CLI latency. Use `decode_ms / output_tokens` only
when comparing decoder loops after the model, tokenizer, prompt cache, and
state setup costs are already accounted for.

Measure warm resident-process request latency without changing product CLI
semantics:

```sh
crystal build --release --no-debug \
  --link-flags="$(pwd)/build/bridge.o -framework Metal -framework Foundation -framework MetalPerformanceShaders -lc++" \
  bin/qwen35_warm_request_probe.cr \
  -o build/qwen35_warm_request_probe

./build/qwen35_warm_request_probe \
  --gen 16 \
  --requests 5 \
  --warmups 1 \
  --quiet \
  "The capital of France is"
```

This probe loads model/tokenizer once, runs explicit warmup requests outside the
measured set, then reports per-request total/tokenize/state-prepare/prefill/decode
timings. Use it to evaluate daemon/server-mode economics; do not mix it with
one-shot `qwen35_generate` totals.

To isolate exact source-history replay economics without one-shot CLI cache,
disk, or lazy Metal compile tax, run the resident replay mode:

```sh
./build/qwen35_warm_request_probe \
  --source-replay \
  --gen 64 \
  --requests 3 \
  --warmups 1 \
  --quiet \
  "alpha beta gamma delta alpha beta gamma delta"

./build/qwen35_warm_request_probe \
  --source-replay \
  --metal-profile \
  --gen 64 \
  --requests 1 \
  --warmups 1 \
  --quiet \
  "alpha beta gamma delta alpha beta gamma delta"
```

`--source-replay` seeds one exact generated span, restores the prompt state
inside the same resident process, and verifies the known span in one chunk with
the same tail-skip contract as source-history cache replay. It is a
cache/session replay measurement, not plain generation throughput.

To measure the real prompt-cache `Store` restore path in the same resident
process, use `--prompt-cache-replay`. `--resident-states=1` enables the hot
in-memory restored-state template path:

```sh
./build/qwen35_warm_request_probe \
  --prompt-cache-replay \
  --resident-states 1 \
  --gen 64 \
  --requests 3 \
  --warmups 1 \
  --quiet \
  "alpha beta gamma delta alpha beta gamma delta"
```

To measure validated session-cache fast-forward, use
`--prompt-cache-fast-forward`. This mode seeds an exact span once, stores the
state after the processed part of that span, validates the full token-history
hash, restores the cached state, and emits the cached output ids without running
the verifier body:

```sh
./build/qwen35_warm_request_probe \
  --prompt-cache-fast-forward \
  --resident-states 1 \
  --reuse-request-state \
  --gen 64 \
  --requests 3 \
  --warmups 1 \
  --quiet \
  "alpha beta gamma delta alpha beta gamma delta"
```

This is a cache-hit/session fast-forward measurement, not generation
throughput. It is exact only when the cached span and state artifact are
validated for the same model, tokenizer, prompt/session tokens, and runtime
contract. `--reuse-request-state` models a daemon/server state pool: cached
state restore overwrites the same prepared destination buffers each request
instead of allocating a fresh destination state. On a local M2 Max Qwen3.5 9B
Q4_K_M smoke, resident fast-forward measured `~4.5 ms` total for 64 cached
tokens (`~0.07 ms/tok`) before request-state pooling; a later 16-token
BF16/live-KV smoke measured p50 total `~0.6-1.0 ms` with request-state reuse.
The same Store source replay path remains slower because it still runs the exact
bulk verifier body.

To measure the terminal direct-output certificate path inside a resident
process, use `--prompt-cache-direct-output`. This mode seeds the same exact span
and certificate once, then times only `Store#lookup_output_fast_forward`
validation and cached id emission; it intentionally performs no state restore,
prefill, or decoder work:

```sh
./build/qwen35_warm_request_probe \
  --prompt-cache-direct-output \
  --gen 16 \
  --requests 7 \
  --warmups 2 \
  --quiet \
  "The capital of France is"
```

On a local M2 Max Qwen3.5 9B Q4_K_M smoke, this measured p50 `0.009 ms`
total for a 16-token cached span after warmup. The same build measured hot
BF16/live-KV state fast-forward with `--resident-states=1` and
`--reuse-request-state` at p50 `0.587 ms` on the same prompt/settings. Treat
direct-output certificate hits as the terminal repeated-output path; use state
fast-forward only when the caller needs a continuation state after the cached
span.

To exercise the resident serving route order directly, use
`--prompt-cache-serving-route`. Terminal requests try the direct output
certificate first. If the caller requires continuation state after the cached
span, add `--serving-route-continuation`; this bypasses terminal id emission and
uses the validated state fast-forward corridor instead:

```sh
./build/qwen35_warm_request_probe \
  --prompt-cache-serving-route \
  --gen 16 \
  --requests 7 \
  --warmups 2 \
  --quiet \
  "The capital of France is"

./build/qwen35_warm_request_probe \
  --prompt-cache-serving-route \
  --serving-route-continuation \
  --resident-states 1 \
  --reuse-request-state \
  --artifact-codec recurrent-bf16 \
  --live-kv-artifacts \
  --gen 16 \
  --requests 7 \
  --warmups 2 \
  --quiet \
  "The capital of France is"
```

The request summaries include `route=direct_output` or
`route=state_fast_forward_continuation`. A local M2 Max smoke measured p50
`0.010 ms` for the terminal direct route and p50 `0.661 ms` for the
continuation-state route on the same 16-token cached span.
`--serving-route-active-cursor` is an upper-bound server-shape probe: it
prewarms the continuation state once and then measures the already-owned
active-session handoff, avoiding the reusable-cache copy that a shared Store
must perform for isolation.
For matrix gates, set `QWEN35_MATRIX_MODE=serving-route`; the TSV includes a
`routes` column so policy changes are visible in apples-to-apples comparisons.
For fail-closed diagnostics, `--serving-route-direct-miss` omits the direct
output certificate while keeping the exact-known-span artifact. Terminal
requests then fall back to
`route=source_history_direct_output_fallback` without restoring state; requests
that need continuation state still use the validated state corridor.
The route decision is also available to product code as
`ML::GGUF::Qwen35ServingRoute.serve_exact_cached_span` in
`src/ml/gguf/qwen35_serving_route.cr`.
For resident servers, `ML::GGUF::Qwen35ResidentSession` in
`src/ml/gguf/qwen35_resident_session.cr` owns route counters and an optional
active continuation cursor while keeping shared Store restores copy-safe.

For a repeatable raw-vs-compressed cache-artifact gate, use the matrix runner:

```sh
scripts/qwen35_cache_artifact_matrix.sh
```

By default it builds/uses `/tmp/qwen35_warm_request_probe_matrix`, runs five
prompt classes through `--prompt-cache-fast-forward`, and reports tab-separated
rows with `mode`, artifact codec, live-KV policy, request totals, and phase
timings. Useful overrides:

```sh
QWEN35_MATRIX_PROMPT_LIMIT=1 \
QWEN35_MATRIX_REQUESTS=1 \
QWEN35_MATRIX_WARMUPS=0 \
QWEN35_MATRIX_CODECS="raw recurrent-bf16" \
QWEN35_MATRIX_LIVE_KV="0 1" \
QWEN35_MATRIX_REUSE_REQUEST_STATE=1 \
scripts/qwen35_cache_artifact_matrix.sh

QWEN35_MATRIX_MODE=direct-output \
QWEN35_MATRIX_REQUESTS=7 \
QWEN35_MATRIX_WARMUPS=2 \
scripts/qwen35_cache_artifact_matrix.sh
```

`QWEN35_MATRIX_MODE=direct-output` collapses the artifact axes to
`codec=direct` and `live_kv=na`, because this mode measures only the resident
direct output-certificate validator and cached id emission.

Use the matrix for cache-artifact economics only. It does not measure first-run
prefill or plain generation throughput.

For prefill/decode profiling logs, use the graph-atlas helper to turn
`Qwen35Metal.Profile` reports into ranked LTP/WBA windows:

```sh
QWEN35_PREFILL_PHASE_PROFILE=1 /tmp/qwen35_prefill_attribution \
  --prompt=64 --warmup=1 --reps=1 --prepare-state \
  --load-warning-threshold=0 --load-total-warning-threshold=0 \
  > /tmp/qwen35_prefill_profile.log

scripts/qwen35_profile_atlas.cr /tmp/qwen35_prefill_profile.log --top=8
```

The atlas is an attribution aid, not a benchmark by itself. Phase profiling
intentionally changes command-buffer boundaries, so use paired wall timing
before promoting any kernel or scheduling change.

Useful Qwen environment switches:

| Variable | Effect |
|---|---|
| `QWEN35_PROMPT_CACHE=1` | Enable exact prompt-state cache lookup/save in `qwen35_generate`. |
| `QWEN35_PROMPT_CACHE_ROOT=/path` | Override prompt-cache artifact root. |
| `QWEN35_PROMPT_TOKEN_CACHE_OFF=1` | Disable tokenized-prompt cache lookup/save while keeping prompt-state cache enabled. |
| `QWEN35_PROMPT_CACHE_FAST_FORWARD=1` | With prompt cache and source-history enabled, save and use validated post-span state artifacts so exact session-cache hits can emit cached spans without verifier recompute. Default off; falls back when source prefix, token hash, or artifact validation fails. |
| `QWEN35_PROMPT_CACHE_PREWEIGHT_FAST_FORWARD_OFF=1` | Diagnostic switch: disable pre-weight terminal fast-forward exits so `qwen35_generate` can exercise the post-weight serving-route/state-restore path. Default off; normal product routing still tries zero-GGUF/zero-weight terminal hits first. |
| `QWEN35_PROMPT_CACHE_ARTIFACT_CODEC=recurrent-bf16` | Store validated fast-forward state artifacts in compressed v2 BF16 recurrent format. KV rows remain raw. Default is raw; `recurrent-int8` is available only with the explicit Metal INT8 gate. |
| `QWEN35_PROMPT_CACHE_ARTIFACT_CODEC_BLOCK=8` | Block size for `recurrent-int8` prompt-cache artifacts. Ignored for raw/BF16. |
| `QWEN35_PROMPT_CACHE_LIVE_KV_ARTIFACTS=1` | Store prompt-cache artifacts in v3 live-KV format: full recurrent state plus only live full-attention KV rows. This is opt-in while broader validation continues. |
| `QWEN35_PROMPT_CACHE_METAL_INT8_RESTORE=1` | Explicitly allow Metal restore of validated `recurrent-int8` artifacts. Default off because INT8 remains approximate and needs stronger prompt/session validation than BF16. |
| `QWEN35_PROMPT_CACHE_FULL_HIT_MIN_GEN=64` | Minimum requested generation length before a full-prompt cache hit can skip suffix replay and use stored next-token metadata. Lower values are useful for experiments but can move first model work into `decode_ms` without improving total wall time. |
| `QWEN35_PROMPT_CACHE_RESIDENT_STATES=0` | Number of restored prompt-cache states to keep hot in a resident process. `0` disables the in-memory state cache. Positive values avoid rereading and redecompressing `.qkv` artifacts on repeated same-process session hits. |
| `QWEN35_SELF_SPEC_ROUTE_MEMORY_ROOT=/path` | Diagnostic/future-wiring hook: resolve a same-weight self-spec proposal route in `qwen35_generate` using the shared route table. Current product generation does not execute the PCA-updown self-spec runner, so hits are reported and the decode path remains unchanged. |
| `QWEN35_SELF_SPEC_ROUTE_KEY=KEY` | Optional caller-certified key for `QWEN35_SELF_SPEC_ROUTE_MEMORY_ROOT`; without it, lookup uses exact prompt text and token ids. |
| `QWEN35_SELF_SPEC_UPDOWN_ADAPTERS_PATH=/path.json` | Optional diagnostic/future-wiring hook paired with `QWEN35_SELF_SPEC_ROUTE_MEMORY_ROOT`: validate a `qwen35_ffn_updown_adapter_v1` artifact for a selected `pca_updown` route and report validity without changing decode behavior. |
| `QWEN35_NATIVE_TOKENIZER_OFF=1` | Disable the native Crystal Qwen BPE encoder and use the external `llama-tokenize` bootstrap path. |
| `QWEN35_PREPARE_STATE_OFF=1` | Disable eager Metal state-buffer preparation in `qwen35_generate`. By default the CLI prepares KV/DeltaNet buffers before timing prompt ingest. |
| `QWEN35_METAL_PROFILE=1` | Enable Metal dispatch/profile attribution for the timed decode region in `qwen35_generate`. The report includes wave/group timings, prefill/source-replay phase traces, matmul logical traffic, conversion traffic, and CPU fallback counts. |
| `QWEN35_CHAT=1` | Render the input through the minimal Qwen 3.5/3.6 chat-template path before tokenization. |
| `QWEN35_CHAT_SYSTEM="..."` | Optional system message used by the chat-template renderer. |
| `QWEN35_TOOLS_JSON='[...]'` | Enable Qwen XML-style function-calling prompt rendering and parsed `<tool_call>` output reporting. The value must be a JSON array of tool definitions. |
| `QWEN35_MESSAGES_JSON='[...]'` | Render an OpenAI/CrystalBall-style message array through the Qwen chat-template path. Supports `user`, `system`, `assistant` with `tool_calls`, and `tool` messages. |
| `QWEN35_TOOL_RESPONSE_JSON=simple\|openai` | Emit a machine-readable `=== Tool response JSON ===` block after generation. `simple` matches CrystalBall's local provider shape; `openai` wraps calls as OpenAI-style function tool calls. With `QWEN35_TOOLS_JSON`, scalar arguments are normalized using the tool schema where possible. |
| `QWEN35_CONSTRAINED_LITERAL_PREFIX='...'` | Experimental greedy-only constrained decoding probe. Forces an exact literal prefix using tokenizer-derived allowed-token frontiers and the constrained Q6 head path, then falls back to normal greedy decode after the literal completes. Currently incompatible with prompt-cache/speculative/n-gram fast paths. |
| `QWEN35_CONSTRAINED_TOOL_CALL_PREFIX=1` | Experimental greedy-only structured tool-call mode. Requires `QWEN35_TOOLS_JSON`; constrains Qwen XML `<tool_call>\n<function=...>\n` prefixes over the available function names, required `<parameter=...>\n` tags in schema order, lets the model choose schema-valid optional parameter tags before final close, constrains finite enum/boolean values and small bounded integer ranges, falls back for free-form single-line values, constrains the inter-parameter and final closing tags after value newlines, stops generation after a complete constrained tool call, batches deterministic literal spans by default, and emits a `tool constraint summary` line with constrained-stage, forced-span, and free-form fallback counts. |
| `QWEN35_CONSTRAINED_FORCE_SPAN_OFF=1` | Disable deterministic literal-span batching inside constrained structured decoding. This is a diagnostic kill switch; the default batches grammar-proven spans with exact body-only state updates and skips intermediate lm-head ranking. |
| `QWEN35_DECODE_POLICY=greedy\|ngram\|speculative\|mtp\|auto` | Explicit decode-mode selector. `auto` chooses the exact fail-closed n-gram path with risk gating; explicit policy overrides legacy mode envs. `mtp` requires `QWEN35_MTP_GGUF_PATH` and remains explicit/default-off. |
| `QWEN35_TRACE_STEPS_OFF=1` | Suppress per-token/per-cycle trace lines in `qwen35_generate` while keeping summaries and final output. |
| `QWEN35_QUIET=1` | Alias for suppressing per-step traces in `qwen35_generate`; useful for cleaner local timing. |
| `QWEN35_NGRAM_DECODE=1` | Enable exact n-gram speculative decode in `qwen35_generate`. |
| `QWEN35_NGRAM_GAMMA=32` | Maximum n-gram verifier chunk size. |
| `QWEN35_NGRAM_MIN=6` | Minimum repeated suffix length before n-gram drafting. |
| `QWEN35_NGRAM_MAX=8` | Maximum suffix length to search for n-gram drafting. |
| `QWEN35_NGRAM_MIN_CANDIDATES=N` | Skip n-gram proposals shorter than `N` candidates. In `auto`, the default is `8`; explicit `ngram` keeps the old default `0` unless set. |
| `QWEN35_NGRAM_STAGE_MIN=N` | Split only n-gram verifier chunks with at least `N` candidates into staged subchunks. In `auto`, the default is `QWEN35_NGRAM_GAMMA + 1`, so the common full chunk is kept intact unless overridden. Explicit `ngram` keeps the old default `0` unless set. |
| `QWEN35_NGRAM_RISK_GATE=0\|1` | In `auto`, the exact candidate-shape risk gate is enabled by default; set `0` to disable. Explicit `ngram` keeps the old default off unless set to `1`. |
| `QWEN35_NGRAM_RISK_MIN_SIZE=16` | Candidate size threshold used by the n-gram risk gate. This is independent from `QWEN35_NGRAM_STAGE_MIN`, so staging can be disabled without weakening fail-closed risk checks. |
| `QWEN35_NGRAM_CORRIDOR_GATE=0\|1` | In `auto`, require runtime repeat-corridor evidence for untrusted local suffix n-gram proposals. Trusted source-history cursor replay bypasses this gate after prefix validation. |
| `QWEN35_NGRAM_CORRIDOR_MATCH_LEN_MIN=8` | In `auto`, allow untrusted suffix proposals when the repeated suffix match reaches this length. Explicit `ngram` defaults to `0`. |
| `QWEN35_NGRAM_CORRIDOR_LAG8_MIN=2.0` | In `auto`, lag-8 repetition alone is disabled as a product certificate after template false positives; set a lower value only for A/B. Explicit `ngram` defaults to `0.5`. |
| `QWEN35_NGRAM_RECURSIVE_OFF=1` | Disable recursive n-gram extension through scratch history. |
| `QWEN35_NGRAM_DISABLE_AFTER_REJECT_OFF=1` | Exploration mode: keep trying n-gram chunks after first rejection. |
| `QWEN35_NGRAM_REPLAY_ON_REJECT=1` | Research/fast-path mode: skip n-gram target-state backups and rebuild the exact target state only after a non-final n-gram reject. Use with the default `auto` risk gate; it can regress badly when a large bad n-gram chunk is forced through verification. |
| `QWEN35_NGRAM_CACHE_MIN_REMAINING=64` | In `auto`, require at least this many remaining requested tokens before using source-history/cache n-gram proposals. This keeps short cache-hit requests on the cheaper exact replay/greedy path. Explicit `ngram` defaults to `0` unless set. |
| `QWEN35_SPECULATIVE_DECODE=1` | Enable exact neural speculative decode in `qwen35_generate` using the 0.8B draft. |
| `QWEN35_DRAFT_MODEL=/path` | Override the Qwen 3.5 draft GGUF used by neural speculative decode. |
| `QWEN35_SPEC_GAMMA=4` | Initial neural draft chunk size in `qwen35_generate`. |
| `QWEN35_SPEC_MAX_GAMMA=32` | Maximum adaptive neural draft chunk size. |
| `QWEN35_SPEC_PLAIN_FALLBACK_OFF=1` | Disable target-only fallback after low-gamma speculative rejection. Useful for A/B experiments; default fallback is faster on rejection-heavy prompts. |
| `QWEN35_SPEC_PLAIN_FALLBACK_GAMMA=2` | Gamma threshold at or below which rejected neural speculative decode falls back to target-only generation. |
| `QWEN35_SPEC_BOOTSTRAP_GAMMA=N` | Default-off neural speculative jump after a fully accepted initial chunk. Can help 100%-accept runs; may regress prompts that reject after an accepted prefix. |
| `QWEN35_SPEC_SINGLE_FAST_OFF=1` | Disable the exact gamma=1 accepted-token fast path in neural speculative decode. Mostly useful when target-only fallback is disabled for A/B experiments. |
| `QWEN35_SPEC_VERIFY=chunk-inplace\|hybrid\|serial` | Choose neural speculative verifier strategy. Default `chunk-inplace` is best for high-accept prompts; `hybrid` can help first-cycle partial-reject prompts. |
| `QWEN35_SPEC_SKIP_DRAFT_BEFORE_FALLBACK_OFF=1` | Disable the exact optimization that skips draft resync work when a rejection is guaranteed to enter target-only fallback. |
| `QWEN35_SPEC_SKIP_DRAFT_BACKUP_BEFORE_FALLBACK_OFF=1` | Disable the matching draft-backup skip before fallback-bound speculative chunks. |
| `QWEN35_HEAD_FULL_ROWS_GUARDED=1` | Experimental speculative-verifier accelerator for large accepted chunks; uses a margin guard and exact fallback for low-margin rows. |
| `QWEN35_HEAD_FULL_ROWS_MARGIN=0.25` | Margin threshold for the guarded full-row verifier route. Higher is safer but falls back more often. |
| `QWEN35_FFN_DOWN_ADD_FUSED_OFF=1` | Disable decode-wave FFN-down residual-add fusion for Q4/Q6 target and Q8 draft experiments. |
| `QWEN35_Q4K_PAIR_H16_MIN_BATCH=64` | Tune the prefill Q4 gate/up shared H16 conversion threshold. The current default enables sharing from pp64 upward after refreshed A/B showed a small exact win. |
| `QWEN35_Q4K_H16_B48_OFF=1` | Disable the exact 48-token Q4_K H16 prefill tile. The default route uses this only for exact 48-token chunks. |
| `QWEN35_Q4K_H16_B64_OFF=1` | Disable the exact wide-batch Q4_K H16 prefill GEMM. The default route uses a 64-token batch tile for prompt chunks that are exact multiples of 64; irregular chunk sizes stay on the older 32-token tile to avoid tail regressions. |
| `QWEN35_Q4K_H16_B64_TAIL_MIN=N` | Experimental exact prefill probe: for non-64 prompt chunks at least `N` tokens, allow the B64 Q4_K H16 tile and its tail-safe B64 up+SwiGLU fusion on the underfilled final tile. Exact row-pack shapes with dedicated tiles stay on their proven routes. Default off because prior long-prompt tail evidence is shape-sensitive; use for A/B only. |
| `QWEN35_Q4K_H16_B80_OFF=1` | Disable the exact 80-token Q4_K H16 prefill tile. The default route uses this only for exact 80-token chunks. |
| `QWEN35_Q4K_H16_B96_OFF=1` | Disable the exact 96-token Q4_K H16 prefill tile. The default route uses this only for exact 96-token chunks; wider irregular chunks still avoid B96 after pp160/pp192 regressions. |
| `QWEN35_Q4K_H16_B112_OFF=1` | Disable the exact 112-token Q4_K H16 prefill tile. The default route uses this only for exact 112-token chunks. |
| `QWEN35_ADDNORM_H16_FFN=1` | Experimental exact prefill route: residual add+RMSNorm writes H16 rows directly for following Q4_H16 FFN gate/up GEMMs. Default off; current evidence is mixed and it is retained only as an attribution/probe knob. |
| `QWEN35_SWIGLU_H16_DOWN=1` | Experimental exact prefill route: SwiGLU writes H16 activations directly for FFN-down Q4/Q5/Q6 batch GEMM consumers. Default off; it removes large conversion traffic at long prompts, but paired wall timing is mixed and not a promotion signal yet. |
| `QWEN35_RMSNORM_H16_PROJ=1` | Experimental exact prefill route: RMSNorm writes f32 plus H16 rows so Q4/Q5/Q6 projection GEMMs can skip their separate input conversion. Default off; useful for attribution, but current paired wall timing is noise-level. |
| `QWEN35_DN_POST_H16_OPROJ=1` | Experimental exact prefill route: recurrent DeltaNet post-gate writes H16 rows directly for following o_proj GEMMs. Default off; completes the conversion-atlas probe set, but current wall timing is neutral. |
| `QWEN35_REC_PROJ_SHARED_H16_OFF=1` | Disable the exact recurrent prefill projection optimization that shares one H16 input conversion between Q5 qkv and Q4 gate GEMMs. |
| `QWEN35_PREFILL_CHUNK_OFF=1` | Force older non-chunked prefill path. |
| `QWEN35_DECODE_WAVE_OFF=1` | Force older non-wave decode path. |

Structured tool-call span batching can be regression-tested with:

```sh
REPS=2 scripts/qwen35_structured_span_suite.sh
```

### Library Integration

The current Qwen API is low-level and intended for native inference experiments:

```crystal
require "ml/gguf/qwen35_cpu"
require "ml/gguf/qwen35_weights"

model = "/path/to/Qwen3.5-9B-Q4_K_M.gguf"
weights = ML::GGUF::Qwen35Weights.from_gguf(model)
state = ML::GGUF::Qwen35CPU::State.new(weights.hparams, max_seq: 1024)

prompt_ids = [760_i32, 6511_i32, 314_i32, 9338_i32, 13_i32]
next_id, next_logit = ML::GGUF::Qwen35CPU.prefill_tokens_top1(weights, prompt_ids, 0, state)

64.times do |i|
  puts next_id
  next_id, next_logit = ML::GGUF::Qwen35CPU.forward_top1(weights, next_id, prompt_ids.size + i, state)
end
```

When linking an executable that uses Metal, include the bridge object and Apple frameworks:

```sh
crystal build your_app.cr \
  --link-flags="$(pwd)/build/bridge.o -framework Metal -framework Foundation -lc++"
```

For CPU-only builds:

```sh
crystal build -Dcpu_only your_app.cr
```

### Benchmark Against llama.cpp

Build the matched benchmark:

```sh
crystal build --release --no-debug \
  --link-flags="$(pwd)/build/bridge.o -framework Metal -framework Foundation -lc++" \
  bin/benchmark_qwen_vs_llama.cr \
  -o build/benchmark_qwen_vs_llama
```

Run a normal first-run prefill/decode comparison:

```sh
./build/benchmark_qwen_vs_llama \
  --model ~/.cache/lm-studio/models/lmstudio-community/Qwen3.5-9B-GGUF/Qwen3.5-9B-Q4_K_M.gguf \
  --llama-bench ~/SrcArchives/AI/llama.cpp/build/bin/llama-bench \
  --prompt=64 \
  --gen=64 \
  --reps=5 \
  --warmup=2
```

For publishable measurements, wait for a quiet host:

```sh
./build/benchmark_qwen_vs_llama \
  --prompt=64 \
  --gen=64 \
  --reps=5 \
  --warmup=2 \
  --wait-quiet-ms=60000 \
  --require-quiet
```

Additional benchmark modes:

```sh
# Default native decode now matches llama-bench `tg`: decoder body only,
# with no output logits/head readback. Product-shaped greedy decode is:
./build/benchmark_qwen_vs_llama --native-decode-top1

# Fresh State per repetition, but Metal state buffers are prepared before
# the timed prefill. This measures prompt ingest without first-touch buffer
# allocation/zeroing in the timed region.
./build/benchmark_qwen_vs_llama --native-prefill-prepare-state

# State buffers allocated once, then reset between reps.
./build/benchmark_qwen_vs_llama --native-prefill-prealloc

# Exact prompt-cache restore after one seeded native prefill.
./build/benchmark_qwen_vs_llama --native-prefill-cache

# Prompt-cache prefix restore plus exact replay of the last N prompt tokens.
./build/benchmark_qwen_vs_llama --native-prefill-cache-prefix-suffix=8
```

Latest guarded relaxed-host Qwen3.5-9B Q4_K_M body-only rows on M2 Max:

| prompt/gen | cogni-ml pp | llama.cpp pp | pp gap | cogni-ml tg | llama.cpp tg | tg gap |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 64/64 | 480.06 tok/s | 458.85 tok/s | +4.62% | 53.75 tok/s | 48.15 tok/s | +11.63% |
| 256/64 | 567.60 tok/s | 566.21 tok/s | +0.25% | 53.43 tok/s | 48.26 tok/s | +10.72% |
| 1024/64 | 582.74 tok/s | 574.77 tok/s | +1.39% | 53.14 tok/s | 48.24 tok/s | +10.16% |

These rows use `--native-prefill-prealloc`, `--threads=8`, and disabled load
warnings. Treat them as guarded relaxed measurements, not publishable quiet-host
ABBA evidence.

For Qwen3.6 MTP / quant baselines, first inspect the local/HF matrix:

```sh
crystal build --release bin/qwen36_mtp_baseline_matrix.cr \
  -o build/qwen36_mtp_baseline_matrix

./build/qwen36_mtp_baseline_matrix
```

The matrix prints local-path detection plus llama.cpp command lines for the
current plain `Q4_K_M` target and external MTP GGUF baselines such as
`unsloth/Qwen3.6-27B-MTP-GGUF:IQ4_NL`,
`unsloth/Qwen3.6-27B-MTP-GGUF:UD-Q4_K_XL`, and
`AtomicChat/Qwen3.6-27B-UDT-MTP-GGUF:Q4_K_XL_MTP`. Use `--run-available` only
after the relevant files are local and the llama.cpp binary advertises
`draft-mtp`.

`benchmark_qwen_vs_llama` also accepts llama.cpp KV-cache options for matching
long-context/product configurations:

```sh
./build/benchmark_qwen_vs_llama \
  --model ~/.cache/lm-studio/models/lmstudio-community/Qwen3.6-27B-GGUF/Qwen3.6-27B-Q4_K_M.gguf \
  --prompt=512 \
  --gen=128 \
  --llama-cache-k=q8_0 \
  --llama-cache-v=q4_0
```

Boundary: native `cogni-ml` currently supports the K-quants used by our
Q4_K_M/Q5_K/Q6_K/Q8_0 paths. IQ/UD MTP GGUFs are external llama.cpp baselines
until native IQ/UD quant loaders are implemented.

Fresh local M2 Max 64GB relaxed-load snapshot after the shared-H16 recurrent projection cleanup, Qwen 3.5 9B Q4_K_M, llama.cpp `llama-bench`, `prompt=64`, `gen=64`, `reps=3`, `warmup=1`, flash-attention off:

| Mode | cogni-ml | llama.cpp | Gap |
|---|---:|---:|---:|
| First-run prefill | 426.70 tok/s p50 | 455.10 tok/s avg | -6.24% |
| Fresh state, prepared Metal buffers | 449.73 tok/s p50 | 464.78 tok/s avg | -3.24% |
| Prefill with preallocated state | 448.60 tok/s p50 | 465.91 tok/s avg | -3.71% |
| Prompt-cache restore | 1350.65 tok/s p50 | 465.80 tok/s avg | +189.97% |
| Plain greedy decode, first-run bench | 48.67 tok/s p50 | 46.67 tok/s avg | +4.29% |
| Plain greedy decode, prepared-state bench | 48.59 tok/s p50 | 46.43 tok/s avg | +4.67% |
| Plain greedy decode, preallocated bench | 48.51 tok/s p50 | 46.63 tok/s avg | +4.05% |
| Plain greedy decode, prompt-cache bench | 48.52 tok/s p50 | 46.58 tok/s avg | +4.18% |

Notes:

- The table is a local engineering snapshot, not a lab-clean public benchmark.
- First-run prefill is still behind llama.cpp on this machine. The native wins currently come from state reuse, prompt-cache restore, and exact speculative decode.
- `--native-prefill-prepare-state` uses a fresh `State` per repetition but calls `Qwen35CPU.prepare_state_metal!` before timing. This is useful for server-style latency where a session object can be prepared before the prompt arrives.
- `--native-prefill-cache` measures exact restore of a previously computed prompt state; it is not a first-run prefill replacement.
- Short decode runs are noisy on a desktop system. The two plain decode rows above are intentionally both shown: treat plain decode as parity-to-faster, not as a stable public margin without a quiet rerun.

Same-host CUDA snapshot, RTX 5060 Ti, Qwen 3.5 9B Q4_K_M, `gen=64`:

| Mode | Speed | Notes |
|---|---:|---|
| llama.cpp CUDA plain decode | 67.18 tok/s, 14.89 ms/tok | `llama-bench` `tg64` on the same host/model |
| cogni-ml CUDA plain greedy probe | ~42.6-42.9 tok/s, ~23.3-23.5 ms/tok | full resident CUDA target path, no proposal reuse |
| cogni-ml CUDA source/cache cursor, risk-gated | 71.76 tok/s, 13.94 ms/tok | exact output; 62/62 active-verified tokens plus two serial cursor advances |
| cogni-ml CUDA trusted source/cache cursor | 86.61 tok/s, 11.55 ms/tok | exact output; 64/64 active-verified tokens, chunks `4,4,8,16,16,16` |
| cogni-ml CUDA trusted bulk replay | 91.69 tok/s, 10.91 ms/tok | exact output; 64/64 active-verified tokens, one `64`-token WBA chunk |
| cogni-ml CUDA trusted artifact restore, device-resident | restore-only lower bound: 0.189 ms / 64 cached tokens | no verifier recompute; requires validated post-span state artifact |
| cogni-ml CUDA trusted artifact restore, host-backed | 12.259 ms / 64 cached tokens | restores a full `max_seq=128` decode state from host memory; excludes durable IO/hash lookup |
| cogni-ml CUDA trusted artifact restore, host-backed live KV | 11.358 ms / 64 cached tokens | restores recurrent state plus live KV rows only; recurrent state dominates at short context |
| cogni-ml CUDA trusted artifact IO/hash, live KV | write/read/hash/restore: 17.695/42.046/59.556/13.003 ms | persistent-path diagnostic for a 56.9 MB artifact; exact cached output still `64/64` |
| cogni-ml CUDA recurrent block-INT8 artifact codec | recurrent bytes `52.7MB -> ~13.2MB`; one-token continuation parity passed on tested cursors | scalar CPU encode/decode is too slow; treat as codec feasibility evidence, not production restore |
| invalid trusted cursor | ~42.3 tok/s, ~23.64 ms/tok | fails closed: zero proposals, active verifier disabled, near plain fallback |

CUDA cache-replay caveats:

- The fast rows are not arbitrary generation. They measure exact replay from a validated session/source cursor, which is the intended primitive for prompt/session cache hits and repeated known spans.
- Exact greedy parity is preserved by verifying every proposed token through the target stack before committing it.
- Wrong cursors must fail closed. The trusted-source mode is only trusted after the source-prefix gate passes; otherwise it falls back near plain decode speed instead of accepting proposal tokens.
- The host-backed live-KV row excludes durable cache lookup, artifact IO, and hashing; the IO/hash row includes a simple persistent-path file write/read plus host SHA-256 tool timing, but still excludes pg/session lookup and production artifact indexing.
- Trusted artifact restore is a cache-hit fast-forward path, not speculative decoding. It is only exact if the restored state artifact and emitted token span are validated against the same model/tokenizer/config/source contract.

Local Metal resident cache snapshot, M2 Max, Qwen 3.5 9B Q4_K_M, prompt
`alpha beta gamma delta alpha beta gamma delta`:

| Mode | Speed | Notes |
|---|---:|---|
| cogni-ml Metal plain greedy resident | `~24.97 ms/tok` | fresh request state, same process, no proposal reuse |
| cogni-ml Metal exact source replay | `~4.60 ms/tok` | synthetic resident prompt-state copy plus exact bulk verifier |
| cogni-ml Metal Store source replay, resident states off | `~5.82 ms/tok` | real prompt-cache Store path; cold artifact restore dominates restore phase |
| cogni-ml Metal Store source replay, resident states on | `~4.51-4.55 ms/tok` | real Store hot state path; remaining wall is verifier body |
| cogni-ml Metal Store fast-forward, resident states off | `~1.45 ms/tok` for 64 cached tokens | cold artifact restore; no verifier body |
| cogni-ml Metal Store fast-forward, resident states on | `~0.07 ms/tok` for 64 cached tokens; `~0.02 ms/tok` for 256 cached tokens | hot validated state restore plus cached token emission; no verifier body |
| `qwen35_generate` direct output fast-forward | `~1-2 ms` total for a 16-token cached span even with 5k irrelevant legacy manifest rows per cache file | one-shot CLI hit reads a per-key output certificate, validates prompt/output/text/exact-span hashes before opening GGUF, emits cached ids/text, and exits |
| Metal encoded BF16 artifact read+restore | `~26.35 ms` for a 28.4MB BF16 recurrent artifact in a one-prompt smoke | direct encoded BF16 recurrent decode into prepared Metal buffers; avoids CPU BF16 decode path measured at `~715 ms`, but resident decoded templates remain faster at `~1.7 ms` restore-only |

Latest focused resident-cache smoke, M2 Max, Qwen3.5 9B Q4_K_M, prompt14/gen8
(`/tmp/qwen35_warm_cache_modes_20260528095435`): plain resident greedy p50
`265.185 ms`; Store replay without resident state p50 `156.728 ms` with p50
restore `44.817 ms`; Store replay with `--resident-states 2` p50 `115.932 ms`
with p50 restore `3.367 ms`; state fast-forward p50 `3.657 ms`; serving-route
continuation p50 `4.304 ms`; active cursor p50 `0.007 ms`; direct-output
certificate p50 `0.015 ms`. These are warm-process session-cache numbers, not
one-shot CLI latency.

Metal cache caveat: source replay and fast-forward are different contracts.
Source replay still verifies the known span through the exact target stack.
Fast-forward skips that body and is only legal after validating the cached
post-span state artifact and full emitted-token history. The CLI output-only
hit also requires cached generated text for exactly the requested output length;
it is exact for the emitted cached text, but it does not restore a continuation
state because the process exits.

### Speculative Decode Harnesses

Neural draft harness with Qwen 3.5 0.8B Q8_0:

```sh
crystal build --release --no-debug \
  --link-flags="$(pwd)/build/bridge.o -framework Metal -framework Foundation -lc++" \
  bin/qwen35_speculative_accept.cr \
  -o build/qwen35_speculative_accept

./build/qwen35_speculative_accept \
  --target ~/.cache/lm-studio/models/lmstudio-community/Qwen3.5-9B-GGUF/Qwen3.5-9B-Q4_K_M.gguf \
  --draft ~/.cache/lm-studio/models/lmstudio-community/Qwen3.5-0.8B-GGUF/Qwen3.5-0.8B-Q8_0.gguf \
  --tokens 64 \
  --ngram \
  "The capital of France is"
```

Cheap-proposal-only policy for repeated/template spans:

```sh
QWEN35_SPEC_NGRAM_MIN_CANDIDATES=8 \
./build/qwen35_speculative_accept \
  --target ~/.cache/lm-studio/models/lmstudio-community/Qwen3.6-27B-GGUF/Qwen3.6-27B-Q4_K_M.gguf \
  --draft ~/.cache/lm-studio/models/lmstudio-community/Qwen3.5-0.8B-GGUF/Qwen3.5-0.8B-Q8_0.gguf \
  --tokens 32 \
  --ngram \
  --ngram-risk-gate \
  --ngram-target-only \
  "alpha beta gamma alpha beta gamma alpha beta gamma alpha"
```

Target-only n-gram speculative harness:

```sh
crystal build --release --no-debug \
  --link-flags="$(pwd)/build/bridge.o -framework Metal -framework Foundation -lc++" \
  bin/qwen35_ngram_speculative.cr \
  -o build/qwen35_ngram_speculative

./build/qwen35_ngram_speculative \
  --tokens 64 \
  --gamma 32 \
  --min-ngram 6 \
  "The capital of France is"
```

Both harnesses replay/check exact greedy target output by default unless their CLI explicitly says otherwise.

Fresh local speculative smoke, same M2 Max 64GB and Qwen 3.5 9B target:

| Mode / prompt | Effective speed | Plain target | Notes |
|---|---:|---:|---|
| Neural draft, `The capital of France is` | 15.38 ms/tok, 65.01 tok/s | 21.98 ms/tok, 45.49 tok/s | 100% accepted, 64/64 candidates |
| Neural draft, `def fibonacci(n):` | 21.06 ms/tok, 47.48 tok/s | 21.71 ms/tok, 46.07 tok/s | falls back after rejection; small but safe win |
| N-gram + neural, `The capital of France is` | 10.10 ms/tok, 98.98 tok/s | 21.91 ms/tok, 45.64 tok/s | repeated-text path, 48/48 n-gram candidates accepted |
| Experimental guarded full-row verifier + neural, `The capital of France is` | 14.32 ms/tok, 69.82 tok/s | 22.36 ms/tok, 44.73 tok/s | `QWEN35_HEAD_FULL_ROWS_GUARDED=1`, 0 fallback rows in this run |
| Experimental guarded full-row verifier + n-gram + neural, `The capital of France is` | 9.20 ms/tok, 108.64 tok/s | noisy target run | `QWEN35_HEAD_FULL_ROWS_GUARDED=1`, 48/48 n-gram candidates accepted |

Speculative decode caveats:

- The speculative paths are exact greedy verification paths, not approximate sampling shortcuts.
- Neural speculative speed depends on draft acceptance. High-accept prompts are faster; rejection-heavy prompts quickly fall back to plain target decode.
- In `qwen35_generate`, neural speculative decode is useful for longer high-accept generations. In a local 64-token smoke, `The capital of France is` measured `20.40 ms/tok` greedy, `16.61 ms/tok` neural speculative, and `15.10 ms/tok` neural speculative with guarded full-row verification. A 32-token smoke was slower due fixed draft/verifier overhead.
- N-gram speculation is a workload-specialized path for repeated/generated-template text. `QWEN35_DECODE_POLICY=auto` is the recommended product profile: risk-gated, runtime-corridor gated for untrusted local suffix replay, `min_candidates=8`, no neural fallback, and exact target-only fallback outside clean cheap-copy spans.
- Prompt cache and source-history n-gram are request-level accelerators, so judge them by `total_ms`, not decode-only `ms/tok`. In a local repeated long-generation smoke, native tokenization removed the external tokenizer cost (`~560 ms -> ~0.1 ms`), tokenized-prompt cache made repeated tokenization effectively `0.0 ms`, and full-prompt cache plus source-history n-gram accepted `128/128` generated tokens with total wall around `1.55 s`. For short generations, the default `QWEN35_PROMPT_CACHE_FULL_HIT_MIN_GEN=64` avoids a misleading path that only shifts first model work between `cache_restore_ms` and `decode_ms`.
- In the research acceptance harness, `--ngram-target-only` / `QWEN35_SPEC_NGRAM_TARGET_ONLY=1` skips neural draft fallback after the cheap n-gram proposal source and uses exact target-only steps instead. On a 27B mixed JSONL gate it beat neural default on `5/7` prompts with paired ratio `0.909x` when combined with `--ngram-risk-gate`, but its average speed was near plain target decode; the real win is on clean repeated spans.
- The n-gram risk gate now also catches small-period prefix overruns such as IP/YAML tails. A focused 27B probe kept clean `alpha beta gamma` repeats at `~2.58x` while turning a YAML overrun from `0.80x` into fail-closed target-only `1.006x`.
- The productization smoke after the structured-tail fix kept clean 27B repeats fast (`~2.55x` over plain), made a YAML-like overrun fail closed (`1.003x`), and had `ngram_target_only_risk` beat neural default on the tiny paired suite (`0.815x` ratio). Treat this as opt-in CLI evidence, not a broad default claim.
- `QWEN35_NGRAM_REPLAY_ON_REJECT=1` is exact but deliberately opt-in. It removes rollback-copy overhead on high-confidence accepted n-gram chunks; on the local 27B+0.8B repeat8 harness it improved `ngram_router16_risk` from `30.85` to `30.21 ms/tok`. A forced no-risk YAML reject regressed from `71.53` to `95.86 ms/tok`, so this is not a broad default.
- N-gram verifier chunks temporarily disable guarded full-row verification even if `QWEN35_HEAD_FULL_ROWS_GUARDED=1`, because partial n-gram rejection exposed a close-row guard failure during adversarial CLI testing.
- `QWEN35_HEAD_FULL_ROWS_GUARDED=1` is still an experimental research switch. The harness checks final output against plain greedy target output, but the route is not broad-defaulted because it relies on a full-row F16 top1 margin guard.
- These numbers are effective decode throughput after prompt prefill; they do not make first-run prefill faster.

## Native Metal Embeddings

The embedding path targets `nomic-embed-text-v2-moe` with a fully native Metal compute pipeline.

```crystal
require "ml"
require "ml/gguf/nomic_bert"
require "ml/gguf/metal_backend"
require "ml/metal/compute_graph"

ML::Metal::Device.init!
model = ML::GGUF::NomicBertMoE.from_gguf("path/to/model.gguf", ML::GGUF::MetalBackend.new)

embedding = model.embed("Your text here")
```

### Embedding Performance

Apple M2 Max, 38 GPU cores:

| Tokens | Latency |
|---:|---:|
| 20 | 14 ms |
| 94 | 16 ms |
| 196 | 33 ms |
| 433 | 70 ms |

### Embedding Pipeline Internals

- simdgroup-matrix GEMM for Q5_K/Q6_K dequant+multiply.
- Batched expert GEMM for MoE experts.
- ComputeGraph wave scheduling with offset-aware dependency analysis.
- Fused QKV split/RoPE, gate/softmax/top-k, scatter, and norm kernels.
- GPU-driven dispatch where useful.

`NOMIC_SIMDGROUP_MATRIX=auto` is the default hardware policy. The verified M2
path keeps matrix GEMM, matrix attention, and batched matrix MoE enabled. On
`Apple M5 Max`, those Nomic routes currently fail closed to scalar-SIMD Metal
because a measured embedding-parity run diverged with sequence length. Use
`NOMIC_SIMDGROUP_MATRIX=on` only for an explicit M5 diagnostic probe;
`NOMIC_SIMDGROUP_MATRIX=off` forces the conservative corridor on any device.
The fallback is correctness-oriented and can be substantially slower.

## Supported Models

| Model | Format | Status |
|---|---|---|
| `Qwen3.5-9B` | GGUF Q4_K_M | Native Metal text generation path, active optimization target. |
| `Qwen3.5-0.8B` | GGUF Q8_0 | Native draft model path for speculative decode harnesses. |
| `Qwen3.6-27B` | GGUF Q4_K_M target | Planned/experimental scale-up target. |
| `nomic-embed-text-v2-moe` | GGUF Q5_K_M | Native Metal embedding pipeline. |
| BERT-like encoders | GGUF | Via `NomicBertMoE` when the architecture matches. |
| Other Llama/Qwen/Mistral-style models | GGUF | Via llama.cpp bindings. |

## Installation

```yaml
# shard.yml
dependencies:
  cogni-ml:
    github: skuznetsov/cogni-ml
    version: ~> 0.40.0
```

## Build And Test

```sh
make build
make spec
```

CPU-only:

```sh
make build_cpu
make spec_cpu
```

llama.cpp helper targets:

```sh
make llama
make llama_env
```

The Makefile searches common local, Homebrew, and system library locations for `libllama`. Override with `LLAMA_DIR`, `LLAMA_BUILD`, or `LLAMA_LIB_DIR` if needed.

## Quick Start

### Tensor + Autograd

```crystal
require "ml"

x = ML::Autograd::Variable.rand(2, 3, requires_grad: true, device: ML::Tensor::Device::CPU)
layer = ML::NN::Linear.new(3, 4, device: ML::Tensor::Device::CPU)

out = layer.forward(x)
loss = out.mean
loss.backward

opt = ML::Optim::Adam.new(layer.parameters)
opt.step
opt.zero_grad
```

### LLM Inference Through llama.cpp

```crystal
require "ml/llm/llama"

ML::LLM.init
model = ML::LLM::Model.new("path/to/model.gguf")
gen = ML::LLM::Generator.new(model)
puts gen.ask("What is Crystal?", max_tokens: 100)
ML::LLM.cleanup
```

### GGUF Embeddings

```crystal
require "ml"
require "ml/gguf/nomic_bert"
require "ml/gguf/metal_backend"
require "ml/metal/compute_graph"

ML::Metal::Device.init!
model = ML::GGUF::NomicBertMoE.from_gguf(
  "nomic-embed-text-v2-moe.Q5_K_M.gguf",
  ML::GGUF::MetalBackend.new
)

vec = model.embed("Crystal programming language")
puts "dim=#{vec.size}"

vecs = model.embed_batch(["Hello", "World", "Crystal"])
```

## Metal Kernels

| Kernel | Purpose |
|---|---|
| `gemm_q4k.metal` | Q4_K GEMV/GEMM paths for Qwen. |
| `gemm_q56k.metal` | Q5_K/Q6_K/Q8_0 GEMV, top1, and helper kernels for Qwen. |
| `gemm_mm.metal` | simdgroup-matrix GEMM for Q5_K/Q6_K and batched expert variants. |
| `gemm_simd.metal` | Scalar SIMD GEMM fallback. |
| `ffn_qwen35.metal` | Qwen FFN, add, RMSNorm, and activation helpers. |
| `delta_net.metal` | Qwen 3.5 DeltaNet/recurrent kernels. |
| `fullattn_qwen35.metal` | Qwen full-attention prefill/decode helpers. |
| `attn_decode_qwen35.metal` | Qwen gated attention decode. |
| `attention_matmul.metal` | Flash-style attention matrix helpers. |
| `bert_fp16.metal` | Nomic/BERT fused ops. |
| `nn.metal` | General NN ops. |

## Platform Support

| Platform | GPU | CPU | Status |
|---|---|---|---|
| macOS Apple Silicon | Metal | Yes | Primary target. |
| macOS Intel | Metal | Yes | Supported for general Metal paths; Qwen performance focus is Apple Silicon. |
| Linux | Experimental CUDA probes | Yes | Use `-Dcpu_only` for GGUF/metadata and CUDA probe CLIs; full Qwen generation is still Metal-first. |
| FreeBSD | No native Metal | Untested CPU-only | Not a primary CI target. |

NVIDIA/CUDA support is currently an experimental backend-probe track, not a full decoder. The Qwen native generation path remains Metal-first.

## Build Flags

| Flag | Effect |
|---|---|
| `-Dcpu_only` | Disable Metal and build pure CPU paths. |
| `-Duse_gguf` | Enable GGUF model loading where applicable. |

## License

MIT
