# Paired control for diagnostic retries

The frozen Qwen3.5-2B Q4_K_M and its two adapters continued the same twelve conversations as in the [diagnostic retry experiment](README.md). The pairing key is the rewritten fixture, adapter, source of the failed draft, and retry number. Both arms share the initial problem, its original diagnostics, the draft, and the archived attempt-0 response. The intervention starts with the message immediately following attempt 0.

After each response the blind arm receives just the existing instruction “Revise the same hole. Return only the replacement line.” The diagnostic arm also receives a failure notice, the compiler or spec label, and the measured public diagnostic. Each arm retains its own raw responses in the dialogue. The second round therefore measures the accumulated effect of the retry policy. The two blind requests are issued regardless of score. The archived diagnostic run also reached both rounds on every branch. Compilers and tests score each output separately from generation.

Both arms use seed 101, temperature 0.8, top-k 40, a 4,096-token context limit, and a 128-token output allowance. They share the model, adapters, native sampler, source and suites by digest. Removing feedback also shortens the prompt. The comparison covers the decision to supply the complete diagnostic message under this retry policy.

## Paired results

All 24 paired outcomes agreed on compilation, target, and the exact set of failing examples. The two arms had 17 different raw answers at corresponding positions. Each round and arm had seven compiler rejections, five compiled answers and zero target or full-suite passes. No answer was rejected for format or length.

Each row applies separately to every listed adapter/draft pair. Ratios count all passed Crystal examples for a compiled answer.

| Rewrite | Adapter / draft | Diagnostic 1 | Blind 1 | Diagnostic 2 | Blind 2 |
|---|---|---|---|---|---|
| Mean gradient | All four pairs | Compile error | Compile error | Compile error | Compile error |
| Conditioning mask | All four pairs | 2/4 | 2/4 | 2/4 | 2/4 |
| Trace phase | own/own, peer/own, peer/peer | Compile error | Compile error | Compile error | Compile error |
| Trace phase | own/peer | 0/3 | 0/3 | 0/3 | 0/3 |

On the first retry, 8/12 diagnostic responses differed from the preceding raw answer, compared with 2/12 blind responses. On the second retry those counts were 7/12 and 6/12. The four first blind gradient retries reused the unresolved `@shape.numel`. The four first diagnostic retries added a float conversion to that expression. In the second round the blind own adapter guessed `@elements` on both drafts, while the blind peer adapter kept `@shape`. All eight second-round gradient responses across both arms still failed compilation.

Every second blind mask response selected byte 1 within the valid binary domain. Every second diagnostic response selected both bytes. In both arms only the two invalid-byte regressions passed. The binary decoding target and reversed-order regression failed.

Three trace pairs continued to duplicate the assignment already present before the hole. In the peer/peer blind arm, the first retry changed the expression to `completed = !call_return` and the second repeated it. The own/peer pair continued to compile and use a string key in a Boolean-keyed table. All three examples errored on that pair.

Blind prefixes used 641–1,083 tokens and diagnostic prefixes used 774–3,121. The largest totals including the output allowance were 1,211 and 3,249 tokens respectively. No blind generation needed prefill recovery. The diagnostic arm recovered three requests at the same seed and prompt. These are paired results for three previously unsuccessful rewrites at one seed.

| Local blind evidence | SHA-256 |
|---|---|
| `protocol.json` | `6cbf7aed96501414d85c19c0cb65f12a9f0900b0ae784eb864e6ffca4643207f` |
| `results.json` | `cfa328e21c388816637f7b6d288b4c1417bbd423e4d9528923cb10cc728ac70d` |

## Reproduction

From the populated native worktree run:

```sh
python3 -B eval/crystal_bugfixes/repair_transfer/retries/run_blind.py
```

The reference outputs are in `build/qlora/scale-2b/own-vs-peer-v1/diagnostics/fix-transfer-retries-v1`. The blind run writes to the sibling `fix-transfer-blind-retries-v1` directory. Set `--output` to a fresh path for an independent run. Each new attempt saves the exact request, native reply, and separate compiler and spec logs for the public and full suites.
