# Hy-Nomas full DiT and Q4 loader frontier

Status: VERIFIED for the scoped execution gates below, 2026-10-25; Q4 quality
and direct Q4 GEMM remain unqualified. The complete checkpoint and compact Q4
loader execute one saved-input, first-segment DiT forward through all 54 blocks.
Both BF16 and identical-decoded-Q4 native controls match the selected upstream
captures. This slice does not run diffusion generation or decode video. The
10% cache-aware physical-memory reserve, 64 MiB swap-growth stop, bounded
process footprints and per-command deadlines remain mandatory.

## Admitted and rejected surface

The existing eight-block BF16 path remains the numerical control. New download,
cache, full-stack and pre/postprocessing code is opt-in until its checks pass.
Target: batch one, one segment, no history, camera or action conditioning,
pinned checkpoint geometry and the existing explicit valid-text-prefix MPS
attention adapter. Do not infer CUDA/Magi padding or BF16-activation parity.
Downloaded checkpoint tensors outside the selected forward are inventoried;
their presence does not establish executable camera/action support.

Q4 is a lossy representation. Preserve finite `atol=rtol=2e-5` for native versus
upstream execution of identical decoded weights. Measure Q4 versus original
BF16 separately and report failures without widening that tolerance or calling
the representations equivalent. Norms and biases retain BF16. Inference,
weight reconstruction and successor handoffs stay on GPU. File I/O, header
validation, cache persistence and terminal diagnostics are setup/control work.

## Plan, risks and rollback

1. Stream and authenticate the complete 34,213,991,280-byte F32 source at HF
   revision `bac719a80c8d02f91c485fca804bb8221e149b70`. Check the pinned header,
   all 2,268 tensor extents, 54 block inventories and complete LFS SHA-256;
   publish atomically and reject incomplete, stale or altered artifacts.
2. Pack and persist a versioned Q4_K matrix/BF16 vector cache, with explicit
   source, tensor geometry, quantizer and byte identities. Admit aggregate
   resident and staging storage before Metal allocation. Reuse a bounded GPU
   weight stage and shared block scratch; allocate nothing during encode.
3. Connect all 54 ordered blocks and required first-segment input, text/time
   and final projection modules. Preflight every resource before the first
   dispatch. Keep all owners rooted through the existing terminal wait.
4. Run focused CPU/Metal falsifiers and upstream same-input one-forward gates,
   including BF16 control, Q4 decoding/forward parity, finite Q4-versus-BF16
   error, scratch-poison repeat and wrong-cache rejection. No diffusion loop.

Risk: CAUTION (new cache format, lossy weights, full model resource lifetime).
Likeliest failures are an incorrect source/cache mapping, an overwritten
successor input, CPU fallback, hidden setup staging or an unaccounted memory
peak. Guard malformed late tensors before Metal; poison reused storage and
test a wrong last block; track encode allocations, early activation reads and
terminal waits. Do not launch concurrent GPU probes. Download and CPU-only
inspection may overlap bounded work if aggregate headroom remains guarded.
Rollback: leave the new opt-in loader/forward unused; existing BF16 probes and
defaults remain the control. Do not alter previous evidence files.

## Definition of done

The full source digest passes; the persisted compact cache reloads under
authenticated inventory and byte budgets; all 54 blocks execute in order in a
single GPU command chain without intermediate host reads/waits or tracked
encode allocations. Required first-segment DiT input and output modules pass
the matching upstream gate. BF16 and same-decoded-Q4 controls retain the finite
2e-5 gate; Q4-versus-BF16 divergence is a separate measured result. Record
guarded peak/headroom/swap and a scoped mini-Quad adversary verdict. A partial
stack or cache-only result is IN_PROGRESS, not a full-DiT completion claim.

Refresh all affected checks after source, format, quantizer, geometry, mask,
precision, ownership, toolchain, OS or device changes. Failure rejects the new
route and preserves the existing BF16 control. Exact commands and observations
will be added after execution; plans and hashes alone are not runtime evidence.

## Observed cache and component evidence

The complete source was downloaded and authenticated at the declared revision:
34,213,991,280 bytes, SHA-256
`73819bcf3b82b090116dd1a1748795de95eef703c304e7443963d00ea8aa4d70`.
The fresh `q4-cache-v2` contains all 54 blocks and 48 global bindings. Its index
SHA-256 is `a235ca592b47161472c7fd94cd3d886fce164480cd0571179c8860002063cbb0`.
Block Q4/vector payloads total 4,593,162,240 bytes, compared with
16,314,338,848 bytes of BF16 block control files including their headers.
The common BF16 file is 308,713,168 bytes including its header. This is a
3.55x block-storage reduction, not a measured inference speedup.

The second pack scans every actual resident BF16 destination on GPU before
publishing any file, including vectors, globals and finite-F32 values that
overflow during BF16 rounding. All 55 packing phases completed with zero
tracked MetalBuffer encode allocations, no preterminal reads and one terminal
wait per offline phase. Maximum admitted per-phase named storage was
991,276,040 bytes; the external guard sampled a 2,007,786,136-byte peak group
physical footprint, 31,338,643,456-byte minimum cache-aware headroom and no
swap growth. The new cache's tensor/index bytes match the first pack; the
second receipt adds the full resident-finiteness gate.

Executed component checks:

```sh
python3 -B -m unittest spec/hy_nomas_full_checkpoint_test.py spec/hy_nomas_q4_dit_reference_test.py
make BUILD_DIR=/opt/mldata_ssd/cogni-ml/hy-nomas-q4-dit-20261025/build/native-v1 spec_hy_nomas_q4_dit
```

The CPU suite passed 25 tests; the guarded native suite passed 7 examples.
The independent PyTorch/MPS tiny Q4/BF16 decoder discriminator also passed its
exact-value gate. Release packer and full native probe builds passed. These
checks by themselves do not establish full-stack numerical parity. The existing
eight-block admission/serial/refiner suite also passed 11 examples after the
new full-stack opt-in was added (`regression-eight-v3`).

## Observed full BF16 and Q4 forwards

The independent pinned upstream Python/MPS forwards completed all 54 blocks:
`oracle-full-bf16-owned-v3` reads the authenticated original F32 block bytes and
rounds the weights to BF16 on MPS; `oracle-full-q4-v3` independently decodes the
authenticated Q4 cache on MPS. Both execute the pinned upstream blocks and use
byte-identical latent, saved Qwen, and RoPE inputs. Native runs
`native-full-bf16-owned-v3` and `native-full-q4-v3` each completed two requests.
Every one of the 14 captured outputs, including blocks 0, 7 and 53 and the final
latent, matched the corresponding upstream F32 bytes exactly. Terminal CPU
inspection confirmed nonconstant, finite values. This qualifies the selected
saved-input first-segment forward, not every intermediate tensor or an
end-to-end diffusion sample. The saved Qwen boundary is not a live native
Qwen26-to-DiT composition test.

| Native mode | Named bytes | Request 1, seconds | Request 2, seconds |
| --- | ---: | ---: | ---: |
| BF16 | 22,616,779,332 | 6.420382625 | 6.415047375 |
| Q4 with reused BF16 stage | 10,895,796,804 | 6.545072166 | 6.563472875 |

Times cover encode start through terminal completion. Q4 named storage includes
4,593,162,240 bytes of resident compact block payloads. Each request had zero
tracked MetalBuffer encode allocations, zero preterminal activation reads,
1,244 command segments and one terminal wait. The second request poisoned 214
exposed mutable buffers and reproduced every captured hash. In-flight release
was rejected. Hidden MPS workspace and immutable weights were not poisoned.

| Guarded run | Sampled peak group footprint | Minimum cache-aware headroom |
| --- | ---: | ---: |
| Upstream BF16 owned v3 | 5,081,615,144 | 39,216,349,184 |
| Native BF16 v3 | 23,014,044,752 | 24,003,624,960 |
| Upstream Q4 v3 | 5,112,858,880 | 40,364,376,064 |
| Native Q4 v3 | 11,268,248,400 | 38,154,354,688 |

All four guards exited zero with no swap growth. Periodic guard values cannot
exclude between-poll spikes or hidden driver allocations. The earlier Q4 v1
native requests took approximately 13 seconds; that latency did not reproduce
in v3. The BF16/Q4 runs are not an interleaved performance experiment, so they
do not establish the cost of staging or a relative speedup. These are one-DiT
forward observations, not seconds of video per wall-clock second.

Both v3 oracles ran from the preserved 59-file source snapshot
`source/oracle-v3` under the external evidence root. Manifest SHA-256:
`27198bafbc826502123b806dc6e117028bcb572788f1a9aed9fb857a90296aaf`;
producer SHA-256:
`fd960c6bae7c7386145be6f616b05ee8d7661ec562675f079b77b369f4519ea8`.
The v3 Q4 sidecar reproduces the v1 Q4 sidecar bytes. v1 is retained as a
historical observation; its exact historical producer bytes were not preserved,
so the frozen v3 rerun supplies the qualification lineage.

Mini-Quad adversary verdict: ROBUST for the measured saved-input scope. A
separate read-only audit checked all frozen source hashes, both fixture/sidecar
identities, all 14 captured native readback digests against raw expected slices,
input equality, telemetry and guarded exits. This is correlated review, not an
independent implementation. Native and upstream both use Apple MPS reductions;
CUDA/Magi and BF16-activation equivalence are outside the gate.

## Quantization error is a separate frontier

The postterminal comparison `reports/q4-vs-bf16-v3.json` authenticates both
fixtures and streams their corresponding output spans. All 14 outputs are
finite and nontrivial; the four raw inputs and common conditioning outputs are
exact. Q4 versus original BF16 is not within the unchanged finite 2e-5 gate:
62,059,101 values in 10 captured tensors exceed it. The quantization delta grows
through the stack:

| Output | Relative L2 error against BF16 | Cosine similarity |
| --- | ---: | ---: |
| Block 0 image output | 0.0131818 | 0.999913 |
| Block 7 image output | 0.0530202 | 0.998597 |
| Block 53 image output | 0.597551 | 0.809744 |
| Final latent | 1.184475 | 0.638643 |

The final latent has RMSE 0.323331 and maximum absolute error 1.776829. This
does not prove visible artifacts, but it precludes treating the present Q4 pack
as a negligible perturbation or qualified quality-preserving replacement.
Keep the loader opt-in. Before promoting compact weights, localize sensitive
matrices with identical-input projections and test higher precision or sparse
corrections against the BF16 control. Faster arithmetic cannot by itself repair
an inaccurate packed representation. No tolerance was widened.

## Reference-memory frontier and direct-Q4 boundary

Two full original-source BF16 oracle attempts stopped on the unchanged 64 MiB
system-swap-growth guard. The second reached the start of block 44. Allocator
telemetry showed a stable approximately 3.34 GB of MPS driver allocation and
approximately 1.07 GB of live MPS tensors after each released block; meanwhile
system compressed memory grew substantially. This refutes accumulation of all
54 F32 GPU weight sets as the explanation for that execution. Host source
mapping, copy registration and allocation lifetimes remain under investigation.
Do not repeat the original-source full run without a discriminating bounded
probe. The sampled process physical footprint alone did not expose this peak;
the independent host swap stop remains essential. Cleanup could not verify
termination within its bounded grace; subsequent process/group inspection
found no oracle process. This does not prove the earlier cleanup succeeded at
its deadline.

The bounded two-block transfer-only mapped/clone controls both completed under
the guard without swap growth (`host-transfer-mmap-v1`,
`host-transfer-clone-v1`). Both released live MPS tensors to zero and retained
1,082,523,648 bytes of driver allocation. These short controls do not reproduce
or explain the full-forward host growth. PyTorch's dense CPU-to-MPS copy wraps
the tensor's page-rounded byte span, not the entire backing checkpoint storage;
the proposed whole-checkpoint-per-upload mechanism is rejected. No memory
threshold or cleanup identity check was relaxed.

The owned-source BF16 alternative passed all 54 blocks under the unchanged
guard. It reads each authenticated original F32 byte extent into bounded owned
CPU storage, then uses the same MPS upload, GPU BF16 rounding and pinned
upstream block. Mapped loading remains the default; `--source-loading owned`
explicitly selects the alternative and is rejected for Q4 mode. Exact source
inventory/identity, malformed extents, overlapping ranges and short-read checks
remain enforced. This passing alternative does not diagnose the mapped path's
full-forward growth. It does not substitute the native-produced BF16 cache for
the independent original source. The upstream oracle's default CPU-to-MPS
transfers can block internally; its single explicit terminal activation
readback is not evidence of a synchronization-free upstream weight-upload path.

The staged native path keeps all block payloads in Q4, reconstructs only one
reused BF16 block stage, and reuses the MPS projection stage. It does not expand
the entire compact cache into a BF16 model. Direct tiled Q4 GEMM is a separate
candidate: Qwen's `simd_mm_q4k_f32` consumes Q4 payloads and F32 activations but
uses F16 operands with F32 accumulation/output. Reusing that kernel changes the
current DiT arithmetic contract. Test it against identical packed weights and
captured activations with the unchanged gate before promotion; keep operand
rounding error separate from quantization error. A direct kernel's lower weight
traffic is not by itself evidence of a faster full DiT forward.

The bounded direct-Q4 discriminator ran on the saved block-0 text FFN down
projection: M23/K8192/N2048, identical Q4 payload and activation. It reuses the
existing Qwen tile without a decoded full-weight buffer for the direct branch.
Qwen's F32-input and preconverted-F16-input branches are bit exact. Against the
qualified BF16-decoded-Q4/F32-MPS control, the direct output has 46,593 failures
out of 47,104 values at unchanged 2e-5, maximum absolute error 0.0834351.
The MPS control with both weights and activations rounded to F16 reduces the
direct-kernel difference to 17 failures, maximum absolute error 0.000247955 and
relative L2 1.41135e-6. This separates the large operand-rounding change from
the smaller residual, but does not isolate reduction order from details of
in-register Q4 decoding. The direct kernel is not promoted.

All six probe branches are finite. They run on one chain with one terminal
packed read, one wait, no tracked encode allocation or preterminal activation
read, and 189,871,104 named bytes for the combined diagnostic. The external
source/report are `tmp/direct-q4/direct_q4_probe.cr` and `result-v1.json`; guarded
run `direct-q4-real-projection-v1` exits zero to indicate diagnostic completion,
not a passed arithmetic gate. Its combined 0.00967546-second GPU interval covers
all six branches, decoding and output packing; it is not an isolated direct
kernel speed measurement. M23 does not represent the full DiT token geometry.

Next falsifiers: preserve the desired decoded-weight and activation arithmetic
inside a direct kernel before a full-block timing comparison, and localize the
much larger packed-weight error with BF16 controls before a quality claim.
Native binary/object/source identities are recorded in
`source/native-runtime-identity-v3.json`. That post-run capture plus the release
build log binds the available runtime artifacts; it is weaker than the frozen
oracle snapshot and is not a hermetic native build attestation.

Runtime evidence is under
`/opt/mldata_ssd/cogni-ml/hy-nomas-q4-dit-20261025/`. Exact guarded argv and logs
are in `logs/`; the v3 comparison reports are in `reports/`; the matching
upstream fixtures and sidecars are in `weights/`. The curated repository packet
is `.landmark/evidence/2026-10-25-hy-nomas-q4-dit.json`. Refresh the scoped
runtime checks after any change listed in the definition of done. A packet is
an index to evidence, not a substitute for the executed neural checks.
