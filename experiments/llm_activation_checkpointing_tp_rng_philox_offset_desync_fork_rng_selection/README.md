# Round 317: Activation Checkpointing & `TP/SP` `CUDA RNG Philox (seed, offset)` Desync Selection

Deterministic evaluator comparing 4 RNG state management architectures under Activation Checkpointing and `TP=8 + SP` across forward-backward bitwise match (`%`), RNG overhead (`ms`), and gradient relative error (`%`).
