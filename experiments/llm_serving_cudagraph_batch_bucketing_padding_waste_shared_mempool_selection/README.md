# Round 325: LLM Serving `CUDAGraph` Batch-Size Bucketing, Padding Waste & Shared `graph_pool_handle` Selection

Deterministic evaluator comparing 4 `CUDAGraph` batch bucketing and memory pool architectures up to `B=256` across reserved graph HBM (`GB`), average padding waste (`%`), and `B=129` step latency (`ms`).
