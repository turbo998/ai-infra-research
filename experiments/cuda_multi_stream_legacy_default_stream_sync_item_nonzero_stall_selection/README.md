# Round 306: CUDA Multi-Stream Overlap, Legacy Default `Stream 0` Implicit Barrier & `.item() / nonzero()` Host-Sync Stall Selection

Deterministic evaluator comparing 4 multi-stream compute-communication overlap architectures across `Comm-Compute` overlap (`%`), step latency (`ms`), and host-device synchronization bubble (`ms`).
