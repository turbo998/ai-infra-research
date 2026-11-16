# Round 316: LLM Serving `prompt_logprobs` (`echo=True`) `[N, V]` Logits `OOM` & Chunked-Prefill Streaming `LSE` Selection

Deterministic evaluator comparing 4 `prompt_logprobs` execution architectures (`8 x 32K` tokens, `V=152,064`) across peak HBM footprint (`GB`), batch latency (`ms`), and chunk-boundary continuity.
