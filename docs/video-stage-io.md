# Serial GPU stage storage

`MetalStagePool`, `StageIO`, and `MPSCommandChain` are opt-in infrastructure
for sharing weight and scratch storage across serial video stages. They do
not admit the complete Wan pipeline or establish video generation speed.

The pool reserves page-aligned anonymous memory and creates no-copy Metal
views. Allocations within one phase are disjoint, including different lanes;
different phases intentionally overlap. Each individual view respects the
device's maximum buffer length. Arena uploads use separate buffers so host
setup constants cannot overwrite another phase's queued work. Weight views
carry an explicit storage dtype and Arena ownership lease.

All views, graph plans, and source files must remain live and immutable until
terminal completion. Prepare every I/O plan before starting. For each stage:
pause the chain, begin its I/O plan, resume and encode its consumers, pause,
retire the stage, and optionally continue the command chain asynchronously.
The next I/O stage waits for the previous GPU retirement before overwriting
the shared storage. I/O completion notifications publish readiness to a Metal
shared event; they neither wait on the host nor download activations.

`MPSCommandChain` refreshes the current native command buffer after MPSGraph
continuations. Borrowed encoders are invalid after pause/resume. Completion
commits the final root and performs one terminal host wait, checks every
tracked root, and sums their GPU timestamps. Only afterward may callers finish
I/O and release plans, Arenas, views, and pools. No finalizer silently waits.

Source inode, size, and modification time are checked before enqueue. This
assumes immutable local checkpoint files; it is not a file lock or a defense
against concurrent adversarial mutation. An asynchronous I/O error is latched
and wakes the chain so completion can reject its entire output. That real
device-error path has not been experimentally injected. Changed-file rejection
before enqueue is tested separately.

## Bounded verification

On Apple M2 Max, the focused six-example gate passes. It covers twelve I/O
stages alternating two physically overlapping views, an unaligned file offset,
exact GPU-consumer results, continued command roots, one terminal host wait,
zero tracked MetalBuffer allocations during encoding, typed Arena ownership,
capacity and lifetime guards, and changed-source rejection. It does not test
multi-gigabyte I/O throughput or the complete model's peak memory.

Run the common primitive and staging regression target serially:

```sh
COGNI_RUN_SAFE_MIN_FREE_PCT=15 scripts/run_safe.sh make 180 6144 \
  spec_video_staging BUILD_DIR=/opt/mldata_ssd/cogni-ml/video-build
```

The evidence for the initial focused gate is in external scratch at
`/opt/mldata_ssd/cogni-ml/video-scene-20261019/evidence/wan-full/`:
`stage-specs-build.log`, `stage-specs-native.log`, and the broader
`staging-common-build.log` / `staging-common-native.log` (23 examples, zero
failures/errors/pending). Refresh the gate after
changing platform/runtime, command continuation behavior, stage ordering,
source immutability, pool layout, or ownership lifetimes. The bounded serial
contract does not authorize concurrent use of a pool.
