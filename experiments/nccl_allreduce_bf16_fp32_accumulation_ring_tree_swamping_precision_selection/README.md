# Round 324: `NCCL All-Reduce` `BF16` 256-Rank Linear Ring Swamping (`7-Bit Mantissa` Absorption) vs. Pairwise Tree + `FP32` Register Accumulation Selection

Deterministic evaluator comparing 4 `256`-GPU gradient reduction architectures across small-gradient signal retention (`%`), relative RMSE (`%`), and `All-Reduce` latency (`ms`).
