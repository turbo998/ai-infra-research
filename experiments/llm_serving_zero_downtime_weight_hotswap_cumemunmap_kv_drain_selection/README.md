# Round 313: Zero-Downtime LLM Weight Hot-Swapping (`In-Place DMA Weight Refit` & CUDAGraph Preservation) Selection

Deterministic evaluator comparing 4 production `70B` LLM checkpoint hot-swapping architectures across replica blackout window (`s`), extra surge GPU requirement (`%`), and `CUDAGraph` preservation.
