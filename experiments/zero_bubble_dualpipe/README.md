# Zero-Bubble & DeepSeek DualPipe Pipeline Parallelism Simulator

Deterministic offline model verifying `1F1B`, `Interleaved-1F1B`, `ZB-H1`, `ZB-H2`, `ZB-V`, `DualPipe`, and `DualPipeV`:
- Exact $B$ (`dgrad`) and $W$ (`wgrad`, `WeightGradStore` FIFO queue) decomposition across `PP in {8, 16, 32}` and `M in {16, 32, 64, 128}`.
- Bidirectional `DualPipe` / `DualPipeV` overlapped `F&B` hiding `EP All-to-All` and P2P communication.
- Mathematical verification of post-validation optimizer rollback (`rollback_adamw`).
