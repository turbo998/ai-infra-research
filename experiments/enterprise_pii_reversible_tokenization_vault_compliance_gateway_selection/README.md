# Enterprise Reversible PII Tokenization (`Presidio + GLiNER ONNX + Redis Vault` vs. Destructive `[REDACTED]`) Selection (`#504`)

Reproducible analytical model comparing 4 enterprise PII data-masking and compliance gateway architectures at `100 RPS`:
- **Tier 1 (`Destructive [REDACTED] Regex Masking`)**: `74.0%` PII recall and **`52.0%` downstream LLM task accuracy** (collapses multiple entities into identical `[REDACTED]` tokens and breaks tool calls).
- **Tier 2 (`8B/70B LLM-Based PII Scrubber`)**: `97.5%` PII recall, but adds **`+385.0ms` latency** and burns **`16x H200 GPUs`**.
- **Tier 3 (`Static Pseudonymization Without Streaming SSE Re-Hydration`)**: `84.5%` downstream accuracy (fails when surrogate tokens split across SSE chunks).
- **Tier 4 (`Hybrid Checksum-Regex + GLiNER/DeBERTa ONNX + Ephemeral Redis Vault + Streaming Aho-Corasick Re-Hydration`)**: Achieves **`98.6%` PII recall**, preserves **`94.2%` downstream LLM reasoning & tool-call accuracy (`+42.2 pts`)**, and adds only **`+4.5ms` gateway latency** with **`0` extra GPUs**.

## Run
```bash
python experiments/enterprise_pii_reversible_tokenization_vault_compliance_gateway_selection/run.py
```
