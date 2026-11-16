# Round 321: `FlashAttention` Varlen (`cu_seqlens`) Sequence-Length Skew & Persistent Dynamic Tile Scheduler Selection

Deterministic evaluator comparing 4 `FlashAttention` varlen tile scheduling architectures on a skewed batch (`1x 64K + 63x 512` tokens) across kernel latency (`ms`), SM utilization (`%`), and effective `TFLOPS`.
