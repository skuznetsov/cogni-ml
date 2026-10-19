# Wan S2V transformer composition gate

`WanS2VTransformer` composes the native patch, text, time, attention-block,
audio-injector and prediction-head operations. Its default topology binds all
40 distinct blocks and 12 official injection sites. The 1,248 transformer
bindings plus 12 separately qualified causal-audio bindings cover the 1,260
tensors in the pinned BF16 checkpoint. Binding coverage is not full-forward
numerical evidence.

## GPU and ownership boundary

Setup prepares pipelines, resident parameters, constant position/frequency
metadata and all activation storage. Encode borrows a serial caller encoder;
it does not create/submit/wait for command buffers, read activations on the
host, or allocate tracked Metal buffers. Weights retain BF16 storage with F32
neural operations. The caller must keep inputs, parameters, trace destinations
and exclusive workspaces alive until completion.

The per-layer scratch reuses consumed buffers for in-place modulation,
normalization, RoPE and GELU, and for later attention/FFN phases. Cross-attention
starts after self-attention has consumed Q and the attention/projection
temporaries. The residual storage remains live independently. Audio gather,
normalization and AdaIN share storage; Q/K normalization is in place.
Independent scratch buffers still reject overlap, including separate no-copy
wrappers over the same region.

At the default retained-history geometry (36,634 sequence tokens), executable
scratch accounting reports 8,799,085,056 block bytes and 2,689,310,720 injector
bytes, compared with 24,542,691,840 bytes for the earlier combined layout.
This excludes weights, sequence ping-pong, conditioning, VAE and private driver
resources. It is not an observed process peak or the 832x512 first-chunk budget.

`wan_s2v_memory.cr` adds opt-in `prepare_reused_scratch!` for two distinct empty
arenas. It shares the consumed block normalization, Q, attention and projection
buffers with the following audio injector. Both arenas are one exclusive
workspace: keep them alive together until completion, and release them together.
`reused_scratch_bytes` and `workspace_resident_bytes` count their union rather
than summing aliases twice. The default separate-arena preparation is unchanged.

## Executed evidence

The pinned upstream revision is
`1ea34ff48f87168174e12956e200b1d908b1c5ff`, using PyTorch 2.8.0 MPS with
fallback disabled. All gates retain finite per-value
`abs(actual-reference) <= 2e-5 + 2e-5*abs(reference)`.

- A two-layer fixture uses distinct weights and two distinct audio injections.
  It checks raw patch/FramePack sequence, projected text, timestep embedding,
  modulation, both block/injector outputs, prediction head and unpatchified
  latents. Both the retained-history path and first-request empty FramePack /
  zero no-pose condition path pass: four examples, zero failures/errors/pending.
- The prior real official block-zero fixture is rerun in F32 and BF16 resident
  modes after the scratch changes. It includes intermediate traces, repeated
  encode/chaining, allocation and forbidden-overlap checks. Each mode reports
  six examples, zero failures/errors/pending.
- A stack-wide guard rejects scratch/activation/trace overlap with any resident
  parameter before encoding the first layer, including a future layer's weight.
  The tiny forward and this counterexample pass together after the guard change.
- The opt-in cross-arena layout and the separate layout each pass the same
  retained/drop two-layer fixture, allocation/overlap guards and exact union
  accounting: five examples, zero failures/errors/pending per layout. These
  results establish serial lifetimes at tiny spatial geometry, not full 480p
  memory or forty-layer propagation. Releasing either arena early is forbidden.

External local receipts are under
`/opt/mldata_ssd/cogni-ml/video-scene-20261019/evidence/wan-full/`:
`transformer-final-build.log`, `transformer-final-native.log`,
`block-alias-build.log`, `block-alias-f32-native.log`, and
`block-alias-bf16-native.log`, `transformer-global-guard-build.log`, and
`transformer-global-guard-native.log`. Fixtures/parameters stay outside the repository.
The cross-arena receipts are `transformer-reused-build.log`,
`transformer-reused-native.log` and `transformer-separate-scratch-native.log`.

From a build linked against the current Metal bridge, run:

```sh
WAN_S2V_TRANSFORMER_FIXTURE="$evidence/transformer-tiny-drop.json" \
  COGNI_RUN_SAFE_MIN_FREE_PCT=15 \
  scripts/run_safe.sh "$build/wan_s2v_transformer_spec" 120 4096
WAN_S2V_TRANSFORMER_FIXTURE="$evidence/transformer-tiny-drop.json" \
  WAN_S2V_REUSE_SCRATCH=1 COGNI_RUN_SAFE_MIN_FREE_PCT=15 \
  scripts/run_safe.sh "$build/wan_s2v_transformer_spec" 120 4096
WAN_S2V_BLOCK_FIXTURE="$official_block/wan-official.json" \
  WAN_S2V_WEIGHT_DTYPE=f32 COGNI_RUN_SAFE_MIN_FREE_PCT=15 \
  scripts/run_safe.sh "$build/wan_s2v_block_spec" 180 8192
WAN_S2V_BLOCK_FIXTURE="$official_block/wan-official.json" \
  WAN_S2V_WEIGHT_DTYPE=bf16 COGNI_RUN_SAFE_MIN_FREE_PCT=15 \
  scripts/run_safe.sh "$build/wan_s2v_block_spec" 180 8192
```

## Remaining frontier

The tiny composition and real single-block gates do not establish propagation
through all 40 distinct production layers, full spatial memory/performance,
the complete conditioning chain, or generated video quality/speed. Standard
attention remains the composition default; the tiled attention primitive needs
its own block/composition admission. Workspace exclusivity across overlapping
commands is a caller precondition, not a runtime lease.

Refresh this evidence when the implementation, upstream/oracle/checkpoint,
precision, geometry, tolerance, runtime, device or driver changes, or the
referenced artifacts are missing or changed. Full-pipeline admission remains
closed until its own strict gates pass.
