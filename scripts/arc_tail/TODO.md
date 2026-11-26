# ARC last-block pilot ledger

- [x] Isolate new code from the user's dirty cogni-ml checkout.
- [x] Load final Qwen3.5 block from the existing local GGUF.
- [x] Implement frozen-prefix native export and differentiable full-attention tail.
- [x] Check causal attention, weight update isolation, and test-answer exclusion.
- [x] Record independent native/PyTorch numerical parity and seeded-defect sensitivity.
- [ ] Prepare a task-level split from actual ARC training files.
- [ ] Run baseline/head/last-block comparison with exact-match scoring.
- [ ] Decide whether to broaden the DSL or train full-vocabulary generation.

Data blocker: the known ARC paths refer to an unavailable external volume.
The user requested real ARC data only; no synthetic training corpus is admitted.
