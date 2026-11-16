# Round 314: `FlashAttention` Logit Soft-Capping (`Gemma-2/3 / Grok` `tanh`) `SFU` Bottleneck & Rational Padé `FMA` Selection

Deterministic evaluator comparing 4 `attn_logit_softcapping` execution paths at `32K` context across effective `TFLOPS`, `32K` prefill latency (`ms`), and maximum numerical error.
