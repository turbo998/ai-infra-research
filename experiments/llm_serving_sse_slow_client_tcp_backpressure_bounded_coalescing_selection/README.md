# Round 307: LLM Serving SSE Slow-Client TCP Backpressure, Unbounded Queue OOM & Token-Coalescing Ring Buffer Selection

Deterministic evaluator comparing 4 streaming SSE output backpressure architectures under `10%` slow/stalled TCP clients across healthy client P99 `TPOT` (`ms`), gateway memory (`GB`), and stalled KV cache hold duration (`s`).
