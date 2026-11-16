# Round 323: LLM Optimizer Parameter-Group Routing (`RMSNorm` Weight-Decay Exclusion & `Muon` 2D vs `AdamW` 1D/Embedding) Selection

Deterministic evaluator comparing 4 optimizer parameter-group routing architectures on an `8B` LLM across validation perplexity (`PPL`), `RMSNorm` $\gamma$ norm stability, and convergence speedup (`x`).
